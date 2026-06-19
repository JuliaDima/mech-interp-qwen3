"""Feature-direction modulation on a concept dataset.

All interventions operate on the raw model (no transcoder reconstruction):
  ablation  → subtract act_f * W_dec[f] from the real MLP output
  injection → add delta * W_dec[f] unconditionally at anchor_pos

When no features are specified the script runs in eval-only mode: it reports
raw-model concept accuracy on the chosen dataset and template, then exits.

Examples:
    # Eval-only — just report baseline accuracy on T0 carry
    python -m experiments.concept_localization.pipeline.run_feature_modulation \\
        --concept carry

    # Ablate joint features (subtract their decoder directions from MLP output)
    python -m experiments.concept_localization.pipeline.run_feature_modulation \\
        --concept carry --joint --features L4_F126502 L19_F23877 \\
        --sweep_dir runs/concept_localization/carry/carry_T0/anchor_rank2_pos9/sweep

    # Inject feature directions on wrong pos pairs
    python -m experiments.concept_localization.pipeline.run_feature_modulation \\
        --concept carry --features L4_F126502 L19_F23877 \\
        --inject_delta 5.0 --anchor_pos 10 \\
        --sweep_dir runs/concept_localization/carry/carry_T0/anchor_rank2_pos9/sweep
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.concept_localization.pipeline.run_concept import (
    CONCEPTS,
    _MODEL,
    _TRANSCODER_SET,
    _load_concept,
)
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.interventions import ablate_feature_directions, inject_feature_directions
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype
from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("feature_modulation")


# ---------------------------------------------------------------------------
# data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    layer: int
    feature_id: int

    @property
    def key(self) -> str:
        return f"L{self.layer}_F{self.feature_id}"


@dataclass
class SplitMetrics:
    n: int
    accuracy: float
    mean_correct_prob: float


@dataclass
class EvalMetrics:
    all: SplitMetrics
    pos: SplitMetrics
    neg: SplitMetrics
    skipped: int


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def parse_feature_name(name: str) -> FeatureSpec:
    nums = [int(x) for x in re.findall(r"\d+", name)]
    if len(nums) != 2:
        raise ValueError(
            f"Feature {name!r} must contain exactly two integers: layer and feature id"
        )
    return FeatureSpec(name=name, layer=nums[0], feature_id=nums[1])


@torch.no_grad()
def _show_examples(model, pairs: list, n: int = 2, max_new_tokens: int = 50) -> None:
    """Print greedy-decoded outputs for the first n pairs (pos + neg each)."""
    tok = model.tokenizer
    device = model.cfg.device
    for pair in pairs[:n]:
        for label, prompt, expected in [
            ("pos", pair.prompt_pos, pair.predict_pos or pair.label_pos),
            ("neg", pair.prompt_neg, pair.predict_neg or pair.label_neg),
        ]:
            ids = tokenize_qwen_input(
                tok(prompt, add_special_tokens=False).input_ids, tok, device
            ).unsqueeze(0)
            out_ids = model.generate(
                ids, max_new_tokens=max_new_tokens,
                do_sample=False, prepend_bos=False, verbose=False,
            )
            new_tokens = out_ids[0, ids.shape[1]:]
            raw = tok.decode(new_tokens, skip_special_tokens=True)
            got = raw.strip().split()[0] if raw.strip() else ""
            marker = "✓" if got.rstrip(".,!?").lower() == expected.strip().lower() else "✗"
            print(f"  [{label}] {marker} expect={expected!r:8s}  got={got!r:8s}  -> {raw[:80]!r}")


def _dedupe_preserving_order(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        key = parse_feature_name(name).key
        if key in seen:
            log.info("Skipping duplicate feature %s", name)
            continue
        seen.add(key)
        out.append(name)
    return out


def _load_sweep_examples(sweep_dir: Path) -> list[dict]:
    """Load sweep examples, accepting sweep_examples.pkl or sweep_dataset_examples.pkl."""
    path = sweep_dir / "sweep_dataset_examples.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"No sweep examples file found in {sweep_dir}")


def _ensure_sweep_activations(sweep_dir: Path, feature_keys: list[str], model) -> "dict[str, np.ndarray]":
    """Compute per-feature activations from sweep_residuals.npz.

    Returns a plain dict {feature_key: np.ndarray of shape (N,)} where N is
    2*n_pairs (interleaved pos/neg).  No file is written — the computation is
    just a few linear-layer applications over the small residual matrix and is
    fast enough to redo on each call.
    """
    from collections import defaultdict

    residuals_path = sweep_dir / "sweep_residuals.npz"
    if not residuals_path.exists():
        raise FileNotFoundError(f"sweep_residuals.npz not found in {sweep_dir}")

    from experiments.concept_localization.sweep_utils import apply_transcoder_all

    residuals = np.load(residuals_path)
    layers_to_feats: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for key in feature_keys:
        spec = parse_feature_name(key)
        layers_to_feats[spec.layer].append((spec.feature_id, key))

    acts_dict: dict[str, np.ndarray] = {}
    for layer, feat_list in sorted(layers_to_feats.items()):
        h_key = f"H_L{layer}"
        if h_key not in residuals.files:
            log.warning("Layer %d not in sweep_residuals — skipping: %s", layer, [k for _, k in feat_list])
            continue
        H = residuals[h_key]  # (N, d_model)
        all_acts = apply_transcoder_all(model, layer, H)  # (N, n_features)
        for feat_id, key in feat_list:
            acts_dict[key] = all_acts[:, feat_id]
        log.info("  Layer %d: computed activations for %d features", layer, len(feat_list))

    return acts_dict


def load_feature_names(args: argparse.Namespace) -> list[str]:
    names = list(args.features or [])
    if args.features_file:
        path = Path(args.features_file)
        text = path.read_text().strip()
        if path.suffix == ".json":
            data = json.loads(text)
            if isinstance(data, list):
                names.extend(str(x) for x in data)
            elif isinstance(data, dict) and ("pos" in data or "neg" in data):
                # edec_features.json format: {"pos": [...], "neg": [...]}
                direction = getattr(args, "features_direction", "pos")
                rows = data.get(direction, [])
                names.extend(r["feature"] for r in rows)
            else:
                raise ValueError(
                    "--features_file JSON must be a list or edec_features.json dict with 'pos'/'neg' keys"
                )
        else:
            names.extend(line.strip() for line in text.splitlines() if line.strip())
    return _dedupe_preserving_order(names)



def _filter_active_prompts(
    feature_keys: list[str],
    npz,
    examples: list[dict],
    pairs: list,
) -> list[dict]:
    """Return individual prompts where any feature in feature_keys has activation > 0.

    Filters at the prompt level (pos/neg independently), not the pair level.
    Returns a flat list of {"prompt", "target", "split", "pair_idx"} dicts.
    pairs must be the same template-filtered list used to generate the sweep, so
    pairs[ex["pair_idx"]] resolves to the correct ConceptPair.
    """
    # Track which (pair_idx, side) has any feature firing. side: 0=pos, 1=neg.
    active_sides: set[tuple[int, int]] = set()
    missing: list[str] = []
    for key in feature_keys:
        if key not in npz:
            missing.append(key)
            continue
        acts = npz[key]  # shape (2*n_pairs,): interleaved [pos_0, neg_0, pos_1, neg_1, ...]
        for ex in examples:
            i = ex["pair_idx"]
            if acts[2 * i] > 0:
                active_sides.add((i, 0))
            if acts[2 * i + 1] > 0:
                active_sides.add((i, 1))

    if missing:
        log.warning("Features not in the input file (skipped for filter): %s", missing)

    prompts: list[dict] = []
    for ex in examples:
        i = ex["pair_idx"]
        pair = pairs[i]
        if (i, 0) in active_sides:
            prompts.append({
                "prompt": pair.prompt_pos,
                "target": pair.predict_pos,
                "split": "pos",
                "pair_idx": i,
            })
        if (i, 1) in active_sides:
            prompts.append({
                "prompt": pair.prompt_neg,
                "target": pair.predict_neg,
                "split": "neg",
                "pair_idx": i,
            })

    n_pos = sum(1 for p in prompts if p["split"] == "pos")
    n_neg = sum(1 for p in prompts if p["split"] == "neg")
    log.info("Active prompts: %d pos + %d neg = %d total", n_pos, n_neg, len(prompts))
    return prompts


def _pairs_to_prompts(pairs: list) -> list[dict]:
    """Flatten ConceptPair list into individual prompt dicts for evaluate()."""
    prompts = []
    for pair in pairs:
        prompts.append({
            "prompt": pair.prompt_pos,
            "target": pair.predict_pos,
            "split": "pos",
        })
        prompts.append({
            "prompt": pair.prompt_neg,
            "target": pair.predict_neg,
            "split": "neg",
        })
    return prompts


def _split_metrics(rows: list[dict], split: str) -> SplitMetrics:
    items = rows if split == "all" else [r for r in rows if r["split"] == split]
    if not items:
        return SplitMetrics(n=0, accuracy=0.0, mean_correct_prob=0.0)
    return SplitMetrics(
        n=len(items),
        accuracy=sum(1.0 for r in items if r["correct"]) / len(items),
        mean_correct_prob=sum(float(r["correct_prob"]) for r in items) / len(items),
    )


def _metrics(rows: list[dict], skipped: int) -> EvalMetrics:
    return EvalMetrics(
        all=_split_metrics(rows, "all"),
        pos=_split_metrics(rows, "pos"),
        neg=_split_metrics(rows, "neg"),
        skipped=skipped,
    )


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model,
    prompts: list[dict],
    feature_map: dict[int, list[int]] | None = None,
    alpha: float = 0.0,
    batch_size: int = 8,
    desc: str | None = None,
    inject_deltas_by_layer: dict[int, dict[int, float]] | None = None,
    inject_positions: list[int] | None = None,
) -> EvalMetrics:
    """Teacher-forced full-answer accuracy over a flat list of prompt dicts.

    Each prompt dict: {"prompt": str, "target": str, "split": "pos"|"neg"}.

    feature_map=None                → plain model (no intervention)
    feature_map + inject_deltas_by_layer → add delta_f * W_dec[f] unconditionally at inject_positions (induce feature activation even if it does not fire)
    feature_map                     → subtract (1-alpha) * act_f * W_dec[f] from real MLP output
                                      alpha=0 → full ablation, alpha=0.5 → half, alpha=1 → no-op
    """
    rows: list[dict] = []
    skipped = 0

    if desc is None:
        if feature_map is None:
            desc = "raw-model"
        else:
            parts = []
            for layer in sorted(feature_map):
                fids = feature_map[layer]
                parts.append(
                    f"L{layer}"
                    + (f"_F{fids[0]}" if len(fids) == 1 else f"[{len(fids)}feat]" if fids else "_tc")
                )
            desc = "+".join(parts) or "tc-baseline"

    examples_by_len: dict[int, list] = {}
    for orig_idx, p in enumerate(prompts):
        prompt_ids = model.tokenizer(p["prompt"], add_special_tokens=False).input_ids
        answer_ids = model.tokenizer(p["target"], add_special_tokens=False).input_ids
        if not answer_ids:
            skipped += 1
            continue
        examples_by_len.setdefault(len(prompt_ids) + len(answer_ids), []).append(
            (p["split"], prompt_ids, answer_ids, orig_idx)
        )

    total = sum(len(v) for v in examples_by_len.values())
    with tqdm(total=total, desc=desc) as pbar:
        for items in examples_by_len.values():
            for start in range(0, len(items), max(1, batch_size)):
                batch = items[start : start + batch_size]
                tokens = torch.stack(
                    [
                        tokenize_qwen_input(
                            prompt_ids + answer_ids, model.tokenizer, model.cfg.device
                        )
                        for _, prompt_ids, answer_ids, _orig in batch
                    ],
                    dim=0,
                )
                if inject_deltas_by_layer is not None:
                    logits = inject_feature_directions(
                        model, tokens, inject_deltas_by_layer, inject_positions
                    )
                elif feature_map is None:
                    logits = model(tokens)
                else:
                    logits = ablate_feature_directions(model, tokens, feature_map, alpha=alpha)

                for i, (split, prompt_ids, answer_ids, orig_idx) in enumerate(batch):
                    n = len(prompt_ids)
                    correct_prob = torch.softmax(logits[i, n], dim=-1)[answer_ids[0]].item()
                    pred_toks = [int(logits[i, n + j].argmax()) for j in range(len(answer_ids))]
                    all_correct = all(p == t for p, t in zip(pred_toks, answer_ids))
                    rows.append({
                        "split": split, "correct": all_correct, "correct_prob": correct_prob,
                        "orig_idx":  orig_idx,
                        "pred_toks": pred_toks,
                        # keep pred_first for tok_change_rate in diff_metrics
                        "pred_first": pred_toks[0] if pred_toks else None,
                    })
                pbar.update(len(batch))

    return _metrics(rows, skipped), rows


def diff_metrics(
    baseline: EvalMetrics, modulated: EvalMetrics,
    base_rows: list[dict], mod_rows: list[dict],
) -> dict:
    out = {"baseline": asdict(baseline), "modulated": asdict(modulated), "change": {}}
    for split in ("all", "pos", "neg"):
        base = getattr(baseline, split)
        mod = getattr(modulated, split)
        br = base_rows if split == "all" else [r for r in base_rows if r["split"] == split]
        mr = mod_rows if split == "all" else [r for r in mod_rows if r["split"] == split]
        improved_frac = (
            sum(1 for b, m in zip(br, mr) if m["correct_prob"] > b["correct_prob"]) / len(br)
            if br else 0.0
        )
        rel_changes = [
            (m["correct_prob"] - b["correct_prob"]) / b["correct_prob"] * 100.0
            for b, m in zip(br, mr) if b["correct_prob"] > 1e-12
        ]
        mean_rel_p_change = sum(rel_changes) / len(rel_changes) if rel_changes else float("nan")
        tok_change_rate = (
            sum(1 for b, m in zip(br, mr) if m.get("pred_first") != b.get("pred_first")) / len(br)
            if br else 0.0
        )
        out["change"][split] = {
            "accuracy": mod.accuracy - base.accuracy,
            "mean_correct_prob": mod.mean_correct_prob - base.mean_correct_prob,
            "improved_frac": improved_frac,
            "mean_rel_p_change": mean_rel_p_change,
            "tok_change_rate": tok_change_rate,
        }
    return out




# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def print_eval_report(raw: EvalMetrics) -> None:
    print("\n=== Concept Accuracy (raw model) ===")
    print(f"{'split':<6} {'n':>5} {'accuracy':>10} {'mean_p(correct)':>16}")
    print("-" * 42)
    for split in ("all", "pos", "neg"):
        m = getattr(raw, split)
        print(f"{split:<6} {m.n:>5} {m.accuracy:>10.1%} {m.mean_correct_prob:>16.4f}")


def print_modulation_report(results: dict) -> None:
    alpha = results["config"]["alpha"]
    mode = "ablation" if alpha == 0.0 else ("amplification" if alpha > 1.0 else f"alpha={alpha}")
    print(f"\n=== Feature modulation report ({mode}) ===")
    print(
        "feature,split,n,"
        "baseline_acc,modulated_acc,delta_acc,"
        "baseline_p,modulated_p,delta_p,"
        "rel_delta_p_pct,improved_frac,tok_change_rate"
    )
    for row in results["features"]:
        m = row["metrics"]
        for split in ("all", "pos", "neg"):
            base = m["baseline"][split]
            mod  = m["modulated"][split]
            chg  = m["change"][split]
            rel  = chg.get("mean_rel_p_change", float("nan"))
            tok  = chg.get("tok_change_rate", float("nan"))
            print(
                f"{row['feature']},{split},{base['n']},"
                f"{base['accuracy']:.4f},{mod['accuracy']:.4f},{chg['accuracy']:+.4f},"
                f"{base['mean_correct_prob']:.6f},{mod['mean_correct_prob']:.6f},"
                f"{chg['mean_correct_prob']:+.6f},"
                f"{rel:+.2f},{chg['improved_frac']:.4f},{tok:.4f}"
            )


def print_token_comparison(
    feature_key: str,
    prompts_r: list[dict],
    base_rows: list[dict],
    mod_rows: list[dict],
    tokenizer,
) -> None:
    """Print per-prompt first 3 predicted tokens comparison between baseline and ablated model."""
    def _dec(tok_id):
        return tokenizer.decode([tok_id]).strip() if tok_id is not None else "-"

    # Rows from evaluate() are reordered by sequence length for batching.
    # Re-sort both row lists back to the original prompts_r order via orig_idx.
    base_sorted = sorted(base_rows, key=lambda r: r["orig_idx"])
    mod_sorted  = sorted(mod_rows,  key=lambda r: r["orig_idx"])

    print(f"\n--- Token comparison: {feature_key} ---")
    print(f"{'#':>3}  {'sp':<3}  {'target':<8}  {'baseline':<12}  {'ablated':<12}  chg  prompt")
    print("-" * 100)
    for idx, (p, br, mr) in enumerate(zip(prompts_r, base_sorted, mod_sorted)):
        base_str = tokenizer.decode(br.get("pred_toks", [])).strip()
        mod_str  = tokenizer.decode(mr.get("pred_toks", [])).strip()
        target   = str(p.get("target", ""))
        marker   = "★" if base_str != mod_str else " "
        prompt_tail = str(p.get("prompt", ""))[-40:]
        print(f"{idx:>3}  {p['split']:<3}  {target:<8}  {base_str:<12}  {mod_str:<12}  {marker}    ...{prompt_tail}")


def plot_results(results: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import experiments.plot_style as ps

    rows = sorted(
        results["features"],
        key=lambda row: row["metrics"]["change"]["all"]["accuracy"],
    )
    if not rows:
        return []

    ps.apply()
    labels = [row["feature"] for row in rows]
    y = list(range(len(rows)))
    offsets = {"all": -0.24, "pos": 0.0, "neg": 0.24}
    colors = {"all": ps.NAVY, "pos": ps.VIOLET, "neg": ps.TEAL}

    alpha = results["config"]["alpha"]
    mode = "ablation" if alpha == 0.0 else ("amplification" if alpha > 1.0 else f"α={alpha}")

    fig_h = max(5.0, 0.36 * len(rows) + 1.8)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, fig_h), sharey=True)

    for ax, metric, title, xlabel in [
        (axes[0], "accuracy", "Accuracy change", "modulated − baseline accuracy"),
        (axes[1], "mean_correct_prob", "Correct-probability change", "modulated − baseline mean p(correct)"),
    ]:
        ax.axvline(0.0, color=ps.GRAY, lw=1.0, ls="--", alpha=0.8)
        for split in ("all", "pos", "neg"):
            xs = [row["metrics"]["change"][split][metric] for row in rows]
            ys = [v + offsets[split] for v in y]
            ax.plot(xs, ys, "o", ms=4.5, color=colors[split], label=split, alpha=0.95)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(axis="x", color="#E0E0E0", lw=0.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=8)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_ylabel("modulated feature", fontsize=9)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].legend(title="split", fontsize=8, title_fontsize=8, loc="best")

    config = results.get("config", {})
    fig.suptitle(
        f"{config.get('concept', 'concept')} feature modulation ({mode})",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    paths = [out_dir / "feature_modulation_summary.pdf"]
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# heatmap grid (edec_topk_grid style)
# ---------------------------------------------------------------------------


def _save_heatmap_grid(
    features: list[FeatureSpec],
    sweep_npz,
    sweep_examples: list[dict],
    out_path: Path,
    concept: str,
    anchor_label: str,
    feature_scores: "dict[str, dict] | None" = None,
) -> None:
    """Save an edec_topk_grid-style PDF for the modulated features.

    Reuses plot_feature_heatmap_grid from visualize.py and _bin_to_heatmap
    from delta_feature_projections.py.  Silently skips when examples lack
    the per-operand digit metadata required for binning (non-carry concepts).
    """
    if not sweep_examples or not all(
        "a_pos" in ex.get("meta", {}) for ex in sweep_examples[:3]
    ):
        log.info("Sweep examples lack digit metadata (a_pos/b_pos) — skipping heatmap grid")
        return

    try:
        from experiments.concept_localization.plots.visualize import plot_feature_heatmap_grid
        from experiments.concept_localization.analyze import FeatureMatch
        from experiments.concept_localization.pipeline.delta_feature_projections import _bin_to_heatmap
    except ImportError as exc:
        log.warning("Heatmap grid skipped (import error: %s)", exc)
        return

    coverage = np.zeros((10, 10), dtype=bool)
    for ex in sweep_examples:
        m = ex.get("meta", {})
        for ak, bk in (("a_pos", "b_pos"), ("a_neg", "b_neg")):
            if ak in m and bk in m:
                coverage[int(m[ak]) % 10, int(m[bk]) % 10] = True

    matrices: dict[tuple[int, int], np.ndarray] = {}
    projections: dict[int, list] = {}
    for f in features:
        if f.key not in sweep_npz:
            log.warning("Feature %s not in sweep_activations — skipping from heatmap", f.key)
            continue
        acts = sweep_npz[f.key]  # (2*n_pairs,) interleaved pos/neg
        grid = _bin_to_heatmap(acts, sweep_examples)  # (10, 10) mean activations
        max_val = float(grid.max())
        if max_val > 0:
            grid = grid / max_val  # normalise to [0, 1] for white→violet cmap
        matrices[(f.layer, f.feature_id)] = grid

        scores = (feature_scores or {}).get(f.key, {})
        projections.setdefault(f.layer, []).append(FeatureMatch(
            feature_id=f.feature_id,
            projection=scores.get("score", 0.0),
            cos_sim=scores.get("dec_cos", 0.0),
            layer=f.layer,
            enc_cos_sim=scores.get("enc_cos", 0.0),
        ))

    if not matrices:
        log.info("No features with sweep activations — skipping heatmap grid")
        return

    plot_feature_heatmap_grid(
        matrices=matrices,
        projections=projections,
        out_path=out_path,
        concept=concept,
        anchor_label=anchor_label,
        xlabel="a",
        ylabel="b",
        ncols=5,
        coverage=coverage,
    )
    log.info("Saved heatmap grid → %s", out_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--concept", required=True, choices=CONCEPTS)
    parser.add_argument(
        "--features", nargs="+", default=None,
        help="Features to modulate, e.g. L13_F56616 L16_F34883. "
             "Omit to run eval-only mode (report baseline accuracy and exit).",
    )
    parser.add_argument("--features_file", default=None, help="Text, JSON list, or edec_features.json dict")
    parser.add_argument("--features_direction", default="pos", choices=["pos", "neg"],
                        help="Which direction to load from edec_features.json ('pos'=positive, 'neg'=negative)")
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--template", default="T0",
                        help="Template filter (T0/T1/T2/None). Default T0.")
    parser.add_argument(
        "--alpha", type=float, default=0.0,
        help="Feature scale factor: 0=ablate, 0<α<1=suppress, α=1=no-op, α>1=amplify, α<0=invert",
    )
    parser.add_argument("--joint", action="store_true",
                        help="Modulate all features simultaneously")
    parser.add_argument(
        "--sweep_dir", default=None, type=Path,
        help="Path to sweep dir containing sweep residuals and sweep examples. "
             "Required in modulation mode: each feature is evaluated only on prompts where it fires.",
    )
    parser.add_argument(
        "--inject_delta", type=float, default=None,
        help="If set, inject delta * W_dec[f] unconditionally at --anchor_pos for each feature. "
             "Unlike alpha, this forces the feature direction even when the feature does not fire.",
    )
    parser.add_argument(
        "--inject_split", default="pos", choices=["pos", "neg", "all"],
        help="Which split to inject on: 'pos', 'neg', or 'all'. Default 'pos'.",
    )
    parser.add_argument(
        "--anchor_pos", type=int, default=10,
        help="Sequence position (0-indexed, post-sink-token) at which to inject. "
             "Default 10 = original position 9 (units digit of second operand) + 1 sink token.",
    )
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    feature_names = load_feature_names(args)
    eval_only = not feature_names

    if not eval_only and args.sweep_dir is None:
        parser.error("--sweep_dir is required in modulation mode")

    log.info("Generating pairs for concept '%s'", args.concept)
    all_pairs = _load_concept(args.concept, 200, args.seed)
    if args.template and args.template.lower() != "none":
        all_pairs = [p for p in all_pairs if p.template == args.template]
    log.info("Loaded %d pairs for concept '%s' template '%s'", len(all_pairs), args.concept, args.template)

    if not eval_only:
        log.info("sweep_dir=%s — each feature evaluated only on prompts where it fires", args.sweep_dir)

    device = get_default_device()
    dtype = parse_dtype(args.dtype)
    log.info("Loading model %s", args.model)
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    model.eval()

    # --- eval-only mode ---
    if eval_only:
        log.info("No features specified — running eval-only mode")
        print("\n--- sample outputs (raw model) ---")
        _show_examples(model, all_pairs, n=2, max_new_tokens=50)
        print()
        raw, _ = evaluate(model, _pairs_to_prompts(all_pairs), feature_map=None,
                          batch_size=args.batch_size, desc="raw-model")
        print_eval_report(raw)

        tmpl_suffix = f"_{args.template}" if args.template and args.template.lower() != "none" else ""
        out_dir = Path(args.out_dir or f"runs/concept_localization/{args.concept}/eval_only")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"concept_accuracy{tmpl_suffix}.json"
        out_path.write_text(json.dumps({
            "config": {
                "concept": args.concept,
                "template": args.template,

                "seed": args.seed,
            },
            "raw_model": {split: asdict(getattr(raw, split)) for split in ("all", "pos", "neg")},
        }, indent=2))
        log.info("Saved → %s", out_path)
        return

    # --- inject mode: force feature direction on a chosen split ---
    if args.inject_delta is not None:
        if not feature_names:
            parser.error("--inject_delta requires --features or --features_file")
        features = [parse_feature_name(n) for n in feature_names]
        joint_feature_map: dict[int, list[int]] = {}
        for f in features:
            joint_feature_map.setdefault(f.layer, []).append(f.feature_id)
        joint_key = "+".join(f.key for f in features)

        # Build prompt list for the chosen split
        inject_prompts: list[dict] = []
        for p in all_pairs:
            if args.inject_split in ("pos", "all"):
                inject_prompts.append({"prompt": p.prompt_pos, "target": p.predict_pos, "split": "pos"})
            if args.inject_split in ("neg", "all"):
                inject_prompts.append({"prompt": p.prompt_neg, "target": p.predict_neg, "split": "neg"})

        # Compute per-feature deltas: use mean activation from sweep when available,
        # fall back to --inject_delta value passed as argument, for features not in the sweep.
        sweep_npz_inj = None
        if args.sweep_dir is not None:
            inj_keys = [parse_feature_name(n).key for n in feature_names]
            sweep_npz_inj = _ensure_sweep_activations(args.sweep_dir, inj_keys, model)

        inject_deltas_by_layer: dict[int, dict[int, float]] = {}
        for f in features:
            if sweep_npz_inj is not None and f.key in sweep_npz_inj:
                pos_acts = sweep_npz_inj[f.key][0::2]  # pos activations (interleaved)
                firing = pos_acts[pos_acts > 0]
                delta = float(firing.mean()) if len(firing) > 0 else args.inject_delta
                source = f"sweep mean ({len(firing)} firing)"
            else:
                delta = args.inject_delta # hard-coded alpha value for inducting activation
                source = "--inject_delta fallback"
            inject_deltas_by_layer.setdefault(f.layer, {})[f.feature_id] = delta
            log.info("  %s: delta=%.4f (%s)", f.key, delta, source)

        log.info(
            "Injecting per-feature deltas at anchor_pos=%d on %d '%s' prompts",
            args.anchor_pos, len(inject_prompts), args.inject_split,
        )

        _, base_rows = evaluate(model, inject_prompts, feature_map=None,
                                batch_size=args.batch_size, desc="inject-baseline")
        _, inj_rows = evaluate(
            model, inject_prompts,
            inject_deltas_by_layer=inject_deltas_by_layer,
            inject_positions=[args.anchor_pos],
            batch_size=args.batch_size, desc="inject-modulated",
        )

        base_sorted = sorted(base_rows, key=lambda r: r["orig_idx"])
        inj_sorted  = sorted(inj_rows,  key=lambda r: r["orig_idx"])

        delta_summary = ", ".join(f"{f.key}={inject_deltas_by_layer[f.layer][f.feature_id]:.2f}" for f in features)
        print(f"\n--- Injection comparison: {joint_key} (deltas=[{delta_summary}], pos={args.anchor_pos}, split={args.inject_split}) ---")
        print(f"{'#':>3}  {'sp':<3}  {'target':<10}  {'baseline':<12}  {'injected':<12}  {'Δp(correct)':>12}  chg  prompt")
        print("-" * 120)
        for idx, (p, br, ir) in enumerate(zip(inject_prompts, base_sorted, inj_sorted)):
            base_str = model.tokenizer.decode(br.get("pred_toks", [])).strip()
            inj_str  = model.tokenizer.decode(ir.get("pred_toks", [])).strip()
            target   = str(p.get("target", ""))
            delta_p  = ir["correct_prob"] - br["correct_prob"]
            marker   = "★" if base_str != inj_str else " "
            prompt_tail = str(p.get("prompt", ""))[-35:]
            print(
                f"{idx:>3}  {p['split']:<3}  {target:<10}  {base_str:<12}  {inj_str:<12}  "
                f"{delta_p:>+12.4f}  {marker}    ...{prompt_tail}"
            )

        n_changed = sum(1 for b, i in zip(base_sorted, inj_sorted) if b.get("pred_toks") != i.get("pred_toks"))
        mean_delta_p = sum(i["correct_prob"] - b["correct_prob"] for b, i in zip(base_sorted, inj_sorted)) / len(base_sorted)
        base_acc = sum(1 for r in base_sorted if r["correct"]) / len(base_sorted)
        inj_acc  = sum(1 for r in inj_sorted  if r["correct"]) / len(inj_sorted)
        print(f"\nSummary ({len(inject_prompts)} '{args.inject_split}' prompts): "
              f"acc {base_acc:.1%} → {inj_acc:.1%} | "
              f"mean Δp(correct) = {mean_delta_p:+.4f} | "
              f"{n_changed} predictions changed")
        return

    # --- modulation mode ---
    features = [parse_feature_name(n) for n in feature_names]

    for feature in features:
        if feature.layer < 0 or feature.layer >= model.cfg.n_layers:
            raise ValueError(f"{feature.name!r} has invalid layer {feature.layer}")
        n_features = model.transcoders[feature.layer].W_enc.shape[0]
        if feature.feature_id < 0 or feature.feature_id >= n_features:
            raise ValueError(
                f"{feature.name!r}: feature id {feature.feature_id} out of range "
                f"(layer {feature.layer} has {n_features} features)"
            )

    alpha = args.alpha
    mode_str = (
        "ablation" if alpha == 0.0
        else "amplification" if alpha > 1.0
        else f"suppression (α={alpha})" if 0.0 < alpha < 1.0
        else f"inversion (α={alpha})"
    )
    log.info("Mode: %s | features: %s", mode_str, [f.key for f in features])

    # Load sweep files once — shared across all features
    feature_keys_all = [parse_feature_name(n).key for n in feature_names]
    sweep_npz = _ensure_sweep_activations(args.sweep_dir, feature_keys_all, model)
    sweep_examples = _load_sweep_examples(args.sweep_dir)

    if args.joint:
        joint_feature_map: dict[int, list[int]] = {}
        for f in features:
            joint_feature_map.setdefault(f.layer, []).append(f.feature_id)
        joint_key = "+".join(f.key for f in features)
        log.info("Joint modulation of %d features: %s", len(features), joint_key)

        feature_keys_joint = [f.key for f in features]
        prompts_r = _filter_active_prompts(feature_keys_joint, sweep_npz, sweep_examples, all_pairs)
        if not prompts_r:
            raise ValueError("No prompts where any joint feature fires — check sweep_dir")
        log.info("Joint: %d active prompts (any feature fires)", len(prompts_r))

        baseline, base_rows = evaluate(model, prompts_r, feature_map=None,
                                       batch_size=args.batch_size, desc="joint-baseline")
        modulated, mod_rows = evaluate(model, prompts_r, feature_map=joint_feature_map,
                                       alpha=alpha, batch_size=args.batch_size, desc="joint-modulated")

        print_token_comparison(joint_key, prompts_r, base_rows, mod_rows, model.tokenizer)
        rows = [{"feature": joint_key, "features": [f.key for f in features],
                 "metrics": diff_metrics(baseline, modulated, base_rows, mod_rows)}]

    else:
        rows = []
        for feature in features:
            prompts_r = _filter_active_prompts([feature.key], sweep_npz, sweep_examples, all_pairs)
            if not prompts_r:
                log.warning("Feature %s fires on no prompts — skipping", feature.key)
                continue
            log.info("Feature %s: %d active prompts (%d pos, %d neg)",
                     feature.key, len(prompts_r),
                     sum(1 for p in prompts_r if p["split"] == "pos"),
                     sum(1 for p in prompts_r if p["split"] == "neg"))

            # Baseline
            baseline, base_rows = evaluate(
                model, prompts_r, feature_map=None,
                batch_size=args.batch_size, desc=f"{feature.key}_base",
            )

            # Modulated: subtract (1-alpha) * act_f * W_dec[f] from real MLP output
            modulated_map = {feature.layer: [feature.feature_id]}
            modulated, mod_rows = evaluate(
                model, prompts_r, feature_map=modulated_map,
                alpha=alpha, batch_size=args.batch_size,
            )

            print_token_comparison(feature.key, prompts_r, base_rows, mod_rows, model.tokenizer)
            rows.append({
                "feature": feature.key,
                "input_name": feature.name,
                "layer": feature.layer,
                "feature_id": feature.feature_id,
                "metrics": diff_metrics(baseline, modulated, base_rows, mod_rows),
            })

    alpha_tag = f"alpha{alpha}".replace(".", "p").replace("-", "neg")
    default_out = f"runs/concept_localization/{args.concept}/modulation_{alpha_tag}"
    out_dir = Path(args.out_dir or default_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "config": {
            "concept": args.concept,
            "model": args.model,
            "transcoder_set": args.transcoder_set,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "template": args.template,
            "alpha": alpha,
            "joint": args.joint,
            "features_file": args.features_file,
            "sweep_dir": str(args.sweep_dir),
        },
        "features": rows,
    }
    out_path = out_dir / "feature_modulation.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Saved → %s", out_path)

    plot_paths = plot_results(results, out_dir)
    for p in plot_paths:
        log.info("Saved plot → %s", p)

    # Load scores (dec_cos, enc_cos) from edec_features.json if the features came from one,
    # so the heatmap panel titles show the alignment scores.
    feature_scores: dict[str, dict] | None = None
    if args.features_file:
        fp = Path(args.features_file)
        if fp.suffix == ".json":
            try:
                data = json.loads(fp.read_text())
                if isinstance(data, dict) and ("pos" in data or "neg" in data):
                    direction = getattr(args, "features_direction", "pos")
                    feature_scores = {
                        row["feature"]: row
                        for row in data.get(direction, [])
                        if "feature" in row
                    }
            except Exception:
                pass

    _save_heatmap_grid(
        features=features,
        sweep_npz=sweep_npz,
        sweep_examples=sweep_examples,
        out_path=out_dir / "modulation_features_grid.pdf",
        concept=args.concept,
        anchor_label=args.sweep_dir.parent.name,
        feature_scores=feature_scores,
    )

    print_modulation_report(results)


if __name__ == "__main__":
    main()
