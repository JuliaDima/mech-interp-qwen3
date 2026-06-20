"""Within-class permutation null test for concept localization.

For any registered concept, constructs K null datasets by shuffling
positive-class prompts against other positive-class prompts within
tokenisation-length-matched groups, then runs the identical delta-extraction
pipeline on each.  The resulting null delta norms form an empirical null
distribution against which the real concept signal is tested.

The null hypothesis is:
    H0 — the contrastive pairing is not necessary.  The observed ||delta_l||
    arises from within-class variation rather than from the concept-critical
    contrast between positive and negative instances.

Under H0 the expected mean delta at every layer is zero.  Rejection requires
that the real ||delta_l|| at the peak layer exceeds the 95th percentile of
the null distribution across K permutations.

Null pair construction
----------------------
For each (template, tokenisation-length) group, the positive-class prompts
are randomly split into two halves and paired across halves.  Both elements
of a null pair belong to the positive class, so the concept-critical signal
is absent by construction.  Within-group pairing guarantees that tokenisation
lengths match, so no pairs are discarded for length mismatch.

Usage
-----
    # Run real extraction + K null permutations:
    python -m experiments.concept_localization.pipeline.run_null_permutation \\
        --concept carry --k 20

    # Run all concepts sequentially:
    python -m experiments.concept_localization.pipeline.run_null_permutation \\
        --concept all --k 20

    # Load existing real deltas to skip re-running the model:
    python -m experiments.concept_localization.pipeline.run_null_permutation \\
        --concept carry --k 20 \\
        --real_deltas runs/concept_localization/carry/deltas.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.concept_localization.concept_pair import ConceptPair
from experiments.concept_localization.extract_deltas_generic import extract_layer_deltas_generic
from experiments.concept_localization.pipeline.run_concept import CONCEPTS, _load_concept
from experiments.plot_style import GRAY, VIOLET, apply
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("null_permutation")

_MODEL = "Qwen/Qwen3-4B"
_TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"


# ── Null pair construction ────────────────────────────────────────────────────

def _group_by_length(
    pairs: list[ConceptPair],
    tokenizer,
    context_keys: list[str] | None = None,
) -> dict[tuple, list[int]]:
    """Return pair indices grouped by (template, token-length [, context-values...]).

    Within each group all positive prompts are the same length, so null pairs
    are guaranteed to match in tokenisation length.

    context_keys: optional list of pair.meta field names to additionally group
    by.  For transitive_ordering use ['a', 'b'] so that null pairs always share
    the same (a, b) context and only c varies — matching the structure of the
    real pairs.
    """
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        length = len(tokenizer(p.prompt_pos, add_special_tokens=False).input_ids)
        ctx = tuple(str(p.meta.get(k, "")) for k in (context_keys or []))
        groups[(p.template, length) + ctx].append(i)
    return dict(groups)


def make_shuffled_pairs(
    pairs: list[ConceptPair],
    groups: dict[tuple[str, int], list[int]],
    rng: random.Random,
) -> list[ConceptPair]:
    """Construct null pairs by pairing positive-class prompts with each other.

    Within each length-matched group the positive prompts are randomly
    shuffled and split in half; each element of the first half is paired with
    the corresponding element of the second half.  Both sides of the resulting
    pair are positive-class instances, so the concept-critical signal is absent
    by construction and the expected mean delta is zero.

    Groups with fewer than 2 members are skipped.
    """
    null_pairs: list[ConceptPair] = []
    for idx_list in groups.values():
        if len(idx_list) < 2:
            continue
        shuffled = idx_list[:]
        rng.shuffle(shuffled)
        mid = len(shuffled) // 2
        first_half = shuffled[:mid]
        second_half = shuffled[mid : mid + len(first_half)]
        for i, j in zip(first_half, second_half):
            null_pairs.append(
                ConceptPair(
                    prompt_pos=pairs[i].prompt_pos,
                    prompt_neg=pairs[j].prompt_pos,  # both positive class
                    label_pos="null",
                    label_neg="null",
                    template=pairs[i].template,
                )
            )
    return null_pairs


# ── Norm extraction ───────────────────────────────────────────────────────────

def _null_norms_from_cache(
    npz_path: Path,
    layers: list[int],
    k: int,
    seed: int,
) -> np.ndarray:
    """Compute K null delta norms by re-pairing cached pos residuals — no model needed.

    sweep_residuals.npz stores H_L{l} of shape (2N, d_model): even rows = pos,
    odd rows = neg.  Null pairs both sides from the pos class, so we shuffle
    pos rows among themselves and compute mean(H[perm1]) - mean(H[perm2]).

    Returns null_raw_norms of shape (k, len(layers)).
    """
    data = np.load(str(npz_path), allow_pickle=True)
    null_raw_norms = np.zeros((k, len(layers)), dtype=np.float32)
    rng = np.random.default_rng(seed)

    # pos rows: even indices 0,2,4,...
    # We only need the H matrices for layers we care about.
    for li, l in enumerate(layers):
        key = f"H_L{l}"
        if key not in data:
            continue
        H_l = data[key].astype(np.float32)          # (2N, d_model)
        H_pos = H_l[0::2]                            # (N, d_model)
        N = len(H_pos)
        if N < 2:
            continue
        mid = N // 2
        for ki in range(k):
            perm = rng.permutation(N)
            first  = H_pos[perm[:mid]].mean(axis=0)
            second = H_pos[perm[mid: mid + mid]].mean(axis=0)
            null_raw_norms[ki, li] = float(np.linalg.norm(first - second))

    return null_raw_norms


def _raw_act_norms(
    model,
    pairs: list[ConceptPair],
    layers: list[int],
    device: torch.device,
    dtype: torch.dtype,
    anchor_mode: str,
) -> tuple[np.ndarray, dict[int, float]]:
    """Return (raw_delta_norms, mean_act_norms) for the given pairs.

    raw_delta_norms[l] = ||delta_l|| (not yet divided by mean_act_norm or max).
    mean_act_norms[l]  = E[||h_l||] over both pos and neg prompts.
    """
    results = extract_layer_deltas_generic(
        model, pairs, layers, device, dtype,
        per_template=False,
        anchor_mode=anchor_mode,
    )
    ld = results["all"]
    raw = np.zeros(len(layers))
    for li, l in enumerate(layers):
        if l in ld.delta:
            raw[li] = ld.delta[l].norm().item()
    return raw, dict(ld.mean_act_norm)


def _act_normalised(
    raw: np.ndarray,
    layers: list[int],
    mean_act_norms: dict[int, float],
) -> np.ndarray:
    """Divide each layer's raw norm by E[||h_l||] to remove residual-stream scale growth."""
    out = np.zeros_like(raw)
    for li, l in enumerate(layers):
        scale = mean_act_norms.get(l, 1.0)
        out[li] = raw[li] / scale if scale > 1e-8 else raw[li]
    return out


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_null_comparison(
    layers: list[int],
    real_norms: np.ndarray,
    null_norms: np.ndarray,
    concept: str,
    out_path: Path,
    k: int,
) -> None:
    """Plot real delta norms against the K-sample null distribution.

    All curves share the same y-scale: they are divided by the real curve's
    maximum activation-normalised norm, so the real signal peaks at 1.0 and
    the null curves show their actual magnitude relative to the real signal.
    """
    apply()
    fig, ax = plt.subplots(figsize=(8, 4))

    null_mean = null_norms.mean(axis=0)
    null_std  = null_norms.std(axis=0)
    null_p95  = np.percentile(null_norms, 95, axis=0)

    # Individual null runs as faint lines
    for ki in range(len(null_norms)):
        ax.plot(layers, null_norms[ki], color=GRAY, lw=0.6, alpha=0.25)

    # Null band (mean ± 1 SD) and 95th percentile
    ax.fill_between(
        layers, null_mean - null_std, null_mean + null_std,
        color=GRAY, alpha=0.20,
        label=f"null mean $\\pm$ 1 SD  ({k} shuffles)",
    )
    ax.plot(layers, null_p95, color=GRAY, lw=1.0, ls=":", label="null 95th percentile")
    ax.plot(layers, null_mean, color=GRAY, lw=1.4)

    # Real signal
    ax.plot(
        layers, real_norms, color=VIOLET, lw=2.4, zorder=5,
        label=r"real $\|\delta_l\| / \max_l \|\delta_l\|$",
    )

    # Mark peak layer
    peak_layer = int(layers[int(real_norms.argmax())])
    ax.axvline(peak_layer, color=VIOLET, lw=1.0, ls="--", alpha=0.55, zorder=4)
    ax.text(
        peak_layer + 0.4, 1.03,
        f"L{peak_layer}",
        fontsize=8, color=VIOLET, va="bottom",
    )

    ax.set_title(
        f"{concept} — within-class permutation null  (K = {k})",
        fontsize=11,
    )
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel(
        r"$\|\delta_l\| / \max_l \|\delta_l^\mathrm{real}\|$",
        fontsize=9,
    )
    ax.set_xlim(-0.5, max(layers) + 0.5)
    ax.set_xticks(range(0, max(layers) + 1, 5))
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved null comparison plot → %s", out_path)


# ── Main runner ───────────────────────────────────────────────────────────────

def run_null_permutation(
    concept: str,
    model_name: str,
    transcoder_set: str,
    n_per_template: int,
    k: int,
    out_dir: Path,
    dtype_str: str = "bfloat16",
    seed: int = 42,
    real_deltas_path: Path | None = None,
    anchor_mode: str = "delimiter",
    context_keys: list[str] | None = None,
    template: str = "T0",
    model=None,
    pairs=None,
) -> dict:
    """Run the permutation null test for one concept and return a summary dict.

    model and pairs can be passed in to avoid reloading when looping over
    multiple anchors (see --anchor_modes in the CLI).
    """
    device = get_default_device()
    dtype = parse_dtype(dtype_str)

    if pairs is None:
        log.info("Loading pairs for concept '%s'  (n_per_template=%d)", concept, n_per_template)
        pairs = _load_concept(concept, n_per_template, seed)
    if template is None:
        raise ValueError("Null permutation must use a single template; use --template T0/T1/T2")
    pairs = [p for p in pairs if p.template == template]
    log.info(
        "Filtered to template %s: %d pairs. Multi-template data is only for run_concept/causal plots.",
        template,
        len(pairs),
    )

    if model is None:
        log.info("Loading model %s", model_name)
        ts_obj, _ = load_transcoder_from_hub(
            transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
        )
        model = AttributionModel.from_pretrained_and_transcoders(
            model_name, ts_obj, dtype=dtype, device=device
        )
        model.eval()

    n_layers = model.cfg.n_layers
    layers = list(range(n_layers))

    # ── Real delta norms ──────────────────────────────────────────────────────
    mean_act_norms: dict[int, float] = {}

    if real_deltas_path is not None and real_deltas_path.exists():
        log.info("Loading real deltas from %s", real_deltas_path)
        saved = torch.load(real_deltas_path, map_location="cpu", weights_only=False)
        all_deltas: dict[int, torch.Tensor] = saved.get("all", {})
        real_raw = np.zeros(len(layers))
        for li, l in enumerate(layers):
            if l in all_deltas:
                real_raw[li] = all_deltas[l].norm().item()

        # Load mean_act_norms from the sibling results.json if present
        results_json = real_deltas_path.parent / "results.json"
        if results_json.exists():
            with open(results_json) as f:
                saved_results = json.load(f)
            mean_act_norms = {
                int(k_str): v
                for k_str, v in saved_results.get("mean_act_norm", {}).items()
            }
            log.info("Loaded mean_act_norms from %s", results_json)
    else:
        log.info("Running real delta extraction…")
        real_raw, mean_act_norms = _raw_act_norms(
            model, pairs, layers, device, dtype, anchor_mode
        )

    # Activation-normalised version (original): divides each layer by E[||h_l||]
    # then by the peak.  Used for statistics (z-score, empirical p).
    real_act = _act_normalised(real_raw, layers, mean_act_norms)
    real_scale_act = real_act.max()
    real_norms = real_act / real_scale_act if real_scale_act > 1e-8 else real_act

    # Raw-max version: ||delta_l|| / max_l(||delta_real_l||) — no E normalisation.
    # Consistent with the per-anchor norms_raw / norms_raw.max() used in plots.
    real_scale_raw = real_raw.max()
    real_norms_maxnorm = real_raw / real_scale_raw if real_scale_raw > 1e-8 else real_raw

    # ── Group pairs by tokenisation length for length-safe null construction ──
    log.info("Computing tokenisation-length groups…")
    groups = _group_by_length(pairs, model.tokenizer, context_keys=context_keys)
    total_null_pairs_per_run = sum(
        len(idx) // 2 for idx in groups.values() if len(idx) >= 2
    )
    log.info(
        "%d groups  →  ~%d null pairs per permutation",
        len(groups),
        total_null_pairs_per_run,
    )
    for key, idxs in sorted(groups.items()):
        log.info("  %s: %d pairs", key, len(idxs))

    # ── K null permutations ───────────────────────────────────────────────────
    rng = random.Random(seed)
    # null_raw_norms[ki, li] = ||delta_l|| for null run ki at layer li
    null_raw_norms = np.zeros((k, len(layers)))

    # Fast path: if sweep_residuals.npz exists alongside the anchor dir, reuse
    # the cached H matrices instead of re-running the model K times.
    sweep_npz_path = out_dir.parent / "sweep" / "sweep_residuals.npz"
    if sweep_npz_path.exists():
        log.info("Using cached sweep residuals for null permutations: %s", sweep_npz_path)
        null_raw_norms = _null_norms_from_cache(sweep_npz_path, layers, k, seed)
    else:
        for ki in range(k):
            log.info("Null permutation %d / %d", ki + 1, k)
            null_pairs = make_shuffled_pairs(pairs, groups, rng)
            log.info("  %d null pairs", len(null_pairs))
            null_raw, _ = _raw_act_norms(
                model, null_pairs, layers, device, dtype, anchor_mode
            )
            null_raw_norms[ki] = null_raw

    # Activation-normalised null (original): same E[||h_l||] and real_scale_act as real.
    null_norms = np.zeros_like(null_raw_norms)
    for ki in range(k):
        null_act = _act_normalised(null_raw_norms[ki], layers, mean_act_norms)
        null_norms[ki] = null_act / real_scale_act if real_scale_act > 1e-8 else null_act

    # Raw-max null: ||delta_null|| / max_l(||delta_real||) — same denominator as real_norms_maxnorm.
    null_norms_maxnorm = np.zeros_like(null_raw_norms)
    for ki in range(k):
        null_norms_maxnorm[ki] = null_raw_norms[ki] / real_scale_raw if real_scale_raw > 1e-8 else null_raw_norms[ki]

    # ── Statistics ────────────────────────────────────────────────────────────
    peak_idx   = int(real_norms.argmax())
    peak_layer = layers[peak_idx]
    null_at_peak = null_norms[:, peak_idx]
    empirical_p  = float((null_at_peak >= real_norms[peak_idx]).mean())
    null_std_pk  = float(null_at_peak.std())
    z_score      = float(
        (real_norms[peak_idx] - null_at_peak.mean()) / (null_std_pk + 1e-8)
    )

    log.info(
        "Peak L%d  real=%.4f  null_mean=%.4f  null_std=%.4f  "
        "null_p95=%.4f  z=%.2f  p=%.4f",
        peak_layer,
        float(real_norms[peak_idx]),
        float(null_at_peak.mean()),
        null_std_pk,
        float(np.percentile(null_at_peak, 95)),
        z_score,
        empirical_p,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "concept":             concept,
        "k":                   k,
        "n_per_template":      n_per_template,
        "seed":                seed,
        "anchor_mode":         anchor_mode,
        "n_groups":            len(groups),
        "null_pairs_per_run":  total_null_pairs_per_run,
        "peak_layer":          peak_layer,
        "real_norm_at_peak":   round(float(real_norms[peak_idx]), 4),
        "null_mean_at_peak":   round(float(null_at_peak.mean()), 4),
        "null_std_at_peak":    round(null_std_pk, 4),
        "null_p95_at_peak":    round(float(np.percentile(null_at_peak, 95)), 4),
        "z_score":             round(z_score, 3),
        "empirical_p_value":   round(empirical_p, 4),
        "layers":              layers,
        "real_norms":          [round(v, 5) for v in real_norms.tolist()],
        "null_norms":          [[round(v, 5) for v in row] for row in null_norms.tolist()],
        # Raw-max normalised versions (no E[||h||] division) — used in anchor_sensitivity plots.
        "real_norms_maxnorm":  [round(v, 5) for v in real_norms_maxnorm.tolist()],
        "null_norms_maxnorm":  [[round(v, 5) for v in row] for row in null_norms_maxnorm.tolist()],
    }

    results_path = out_dir / "null_permutation.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved null results → %s", results_path)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_null_comparison(
        layers, real_norms, null_norms, concept,
        out_dir / "null_permutation.png",
        k,
    )

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--concept", required=True,
        choices=CONCEPTS + ["all"],
        help="Concept name or 'all' to run every registered concept.",
    )
    parser.add_argument("--model",           default=_MODEL)
    parser.add_argument("--transcoder_set",  default=_TRANSCODER_SET)
    parser.add_argument("--n",    type=int,  default=100, help="Pairs per template")
    parser.add_argument("--template", default="T0",
                        help="Single template for per-anchor null permutation")
    parser.add_argument("--k",    type=int,  default=20,  help="Number of null permutations")
    parser.add_argument("--seed", type=int,  default=42)
    parser.add_argument("--dtype",           default="bfloat16")
    parser.add_argument("--anchor_mode",  default="delimiter",
                        help="Single anchor position: 'delimiter', 'last', or integer string.")
    parser.add_argument(
        "--anchor_modes", default=None,
        help=(
            "Run null for multiple anchors in one pass (model loaded once). "
            "Accepts comma-separated integer positions (e.g. '5,6,7,8') or "
            "'topN' to select N anchors from emergence.npy. "
            "Overrides --anchor_mode when set."
        ),
    )
    parser.add_argument(
        "--context_keys", default=None,
        help=(
            "Comma-separated pair.meta field names to group null pairs by, "
            "in addition to template and token length.  Use 'a,b' for "
            "transitive_ordering so that null pairs always share the same "
            "(a, b) context and only c varies."
        ),
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Output directory.  Default: runs/concept_localization/<concept>/null/",
    )
    parser.add_argument(
        "--real_deltas", default=None,
        metavar="PATH",
        help=(
            "Path to an existing deltas.pt produced by run_concept.py.  "
            "When provided the real extraction step is skipped.  "
            "A sibling results.json is loaded automatically for mean_act_norms.  "
            "Ignored when --concept all is used."
        ),
    )
    args = parser.parse_args()

    concepts = CONCEPTS if args.concept == "all" else [args.concept]
    ctx_keys = [k.strip() for k in args.context_keys.split(",")] if args.context_keys else None

    summaries = []
    for concept in concepts:
        log.info("=" * 60)
        log.info("Concept: %s", concept)
        log.info("=" * 60)

        # ── Resolve anchor list ───────────────────────────────────────────────
        if args.anchor_modes:
            import re as _re
            from experiments.concept_localization.plots.plot_anchor_analysis import (
                load_emergence, top_k_anchors,
            )
            _top_m = _re.fullmatch(r"top(\d+)", args.anchor_modes)
            if _top_m:
                k_anch = int(_top_m.group(1))
                em = load_emergence(concept)
                if em is None:
                    log.warning("emergence.npy not found — falling back to delimiter.")
                    anchor_list = ["delimiter"]
                else:
                    n_nonzero = sum(
                        1 for a in range(em["norms_raw"].shape[0])
                        if em["norms_raw"][a].max() > 1e-8
                    )
                    anchors = top_k_anchors(em, concept, k=min(k_anch, n_nonzero))
                    anchor_list = [str(idx) for idx, _, _ in anchors]
            else:
                anchor_list = [m.strip() for m in args.anchor_modes.split(",")]
        else:
            anchor_list = [args.anchor_mode]

        multi = len(anchor_list) > 1

        # ── Load model + pairs once for all anchors ───────────────────────────
        device = get_default_device()
        dtype = parse_dtype(args.dtype)
        log.info("Loading model %s", args.model)
        ts_obj, _ = load_transcoder_from_hub(
            args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
        )
        shared_model = AttributionModel.from_pretrained_and_transcoders(
            args.model, ts_obj, dtype=dtype, device=device
        )
        shared_model.eval()
        shared_pairs = _load_concept(concept, args.n, args.seed)
        log.info("Loaded %d pairs before template filtering", len(shared_pairs))

        # ── Loop over anchors ─────────────────────────────────────────────────
        for anchor_mode in anchor_list:
            if multi:
                log.info("--- anchor %s ---", anchor_mode)

            # Derive out_dir
            suffix = f"_pos{anchor_mode}" if anchor_mode != "delimiter" else ""
            if args.out_dir and not multi:
                out_dir = Path(args.out_dir)
            else:
                out_dir = Path(f"runs/concept_localization/{concept}/null{suffix}")
            out_dir.mkdir(parents=True, exist_ok=True)

            # real_deltas: only usable for single-anchor runs
            real_deltas: Path | None = None
            if not multi and args.concept != "all" and args.real_deltas:
                real_deltas = Path(args.real_deltas)
            elif not multi:
                auto = Path(f"runs/concept_localization/{concept}/deltas.pt")
                if auto.exists():
                    log.info("Auto-detected real deltas at %s", auto)
                    real_deltas = auto

            summary = run_null_permutation(
                concept=concept,
                model_name=args.model,
                transcoder_set=args.transcoder_set,
                n_per_template=args.n,
                k=args.k,
                out_dir=out_dir,
                dtype_str=args.dtype,
                seed=args.seed,
                real_deltas_path=real_deltas,
                anchor_mode=anchor_mode,
                context_keys=ctx_keys,
                template=args.template,
                model=shared_model,
                pairs=shared_pairs,
            )
            summaries.append(summary)

    if len(summaries) > 1:
        _print_summary_table(summaries)


def _print_summary_table(summaries: list[dict]) -> None:
    """Print a compact summary table for all-concept runs."""
    header = f"{'concept':<25}  {'peak_L':>6}  {'real':>6}  {'null_p95':>8}  {'z':>6}  {'p':>6}"
    log.info(header)
    log.info("-" * len(header))
    for s in summaries:
        log.info(
            "%-25s  %6d  %6.3f  %8.3f  %6.2f  %6.4f",
            s["concept"],
            s["peak_layer"],
            s["real_norm_at_peak"],
            s["null_p95_at_peak"],
            s["z_score"],
            s["empirical_p_value"],
        )


if __name__ == "__main__":
    main()
