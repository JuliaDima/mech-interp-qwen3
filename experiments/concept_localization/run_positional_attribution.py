"""Which token position near the end of an expression holds the most concept info?

Two modes:

  Default  — runs at a single layer per concept (the peak patching layer from
             results.json, falling back to --layer).  Produces a bar chart.

  --sweep  — runs across every layer in one forward+backward pass per pair.
             Produces a heatmap (layers x positions) per concept showing where
             concept information concentrates and at which depth.

Scores are cosine(grad[layer, pos], Δh[layer, pos]), so magnitude of activations
does not confound position comparisons.

Usage
-----
    python -m experiments.concept_localization.run_positional_attribution
    python -m experiments.concept_localization.run_positional_attribution --concepts gcd residue_class
    python -m experiments.concept_localization.run_positional_attribution --sweep
    python -m experiments.concept_localization.run_positional_attribution --sweep --n_layers 36
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import experiments.plot_style as ps
from experiments.concept_localization.causal_analysis import (
    run_positional_attribution,
    run_positional_attribution_sweep,
)
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_pos_attr")

_MODEL = "Qwen/Qwen3-4B"
_TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"
_RUNS_DIR = _REPO_ROOT / "runs" / "concept_localization"
_DEFAULT_LAYER = 18


ALL_CONCEPTS = [
    "carry",
    "gcd",
    "residue_class",
    "transitive_ordering",
    "triangle_inequality",
    "perfect_square",
    "decimal_termination",
    "geometric_series",
    "dot_product",
    "conservation",
    "momentum",
    "doppler",
    "wave_interference",
    "causal_direction",
    "negation_scope",
    "syllogism",
    "balanced_parentheses",
]

_CONCEPT_REGISTRY: dict[str, tuple[str, str]] = {
    "carry": ("data.concept_datasets.carry_dataset", "generate_carry_pairs"),
    "gcd": ("data.concept_datasets.gcd_dataset", "generate_gcd_pairs"),
    "residue_class": ("data.concept_datasets.residue_class_dataset", "generate_residue_pairs"),
    "transitive_ordering": (
        "data.concept_datasets.transitive_ordering_dataset",
        "generate_ordering_pairs",
    ),
    "triangle_inequality": (
        "data.concept_datasets.triangle_inequality_dataset",
        "generate_triangle_pairs",
    ),
    "perfect_square": (
        "data.concept_datasets.perfect_square_dataset",
        "generate_perfect_square_pairs",
    ),
    "decimal_termination": (
        "data.concept_datasets.decimal_termination_dataset",
        "generate_decimal_pairs",
    ),
    "geometric_series": (
        "data.concept_datasets.geometric_series_dataset",
        "generate_geometric_pairs",
    ),
    "dot_product": ("data.concept_datasets.dot_product_sign_dataset", "generate_dot_pairs"),
    "conservation": ("data.concept_datasets.conservation_dataset", "generate_conservation_pairs"),
    "momentum": ("data.concept_datasets.momentum_conservation_dataset", "generate_momentum_pairs"),
    "doppler": ("data.concept_datasets.doppler_shift_dataset", "generate_doppler_pairs"),
    "wave_interference": ("data.concept_datasets.wave_interference_dataset", "generate_wave_pairs"),
    "causal_direction": ("data.concept_datasets.causal_direction_dataset", "generate_causal_pairs"),
    "negation_scope": ("data.concept_datasets.negation_scope_dataset", "generate_negation_pairs"),
    "syllogism": ("data.concept_datasets.syllogism_dataset", "generate_syllogism_pairs"),
    "balanced_parentheses": (
        "data.concept_datasets.balanced_parentheses_dataset",
        "generate_parentheses_pairs",
    ),
}


def _peak_causal_layer(concept: str, fallback: int) -> int:
    """Read peak patching layer from results.json, fall back to fallback."""
    results_path = _RUNS_DIR / concept / "results.json"
    if results_path.exists():
        data = json.loads(results_path.read_text())
        causal = data.get("causal")
        if causal and "all" in causal:
            pm = causal["all"].get("patching_mean", {})
            if pm:
                return int(max(pm, key=lambda k: pm[k], default=fallback))
    return fallback


def _load_pairs(concept: str, n: int, seed: int):
    if concept not in _CONCEPT_REGISTRY:
        raise ValueError(f"Unknown concept: {concept!r}")
    mod_name, fn_name = _CONCEPT_REGISTRY[concept]
    mod = __import__(mod_name, fromlist=[fn_name])
    return getattr(mod, fn_name)(n, seed=seed)


def _clean_tok(t: str) -> str:
    """Convert BPE token labels to human-readable form; lone spaces become [TR. SPACE]."""
    clean = t.replace("Ġ", "")  # strip Ġ (GPT-2 space prefix)
    return "[TR. SPACE]" if not clean or clean.isspace() else clean


# ── single-layer bar chart ────────────────────────────────────────────────────


def plot_positional_attribution(
    results: dict[str, tuple[dict[int, list[float]], list[str]]],
    out_path: Path,
    fallback_layer: int,
) -> None:
    """One subplot per concept: bar chart of mean cosine attribution by position."""
    concepts = list(results.keys())
    n = len(concepts)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    ps.apply()
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes_flat = list(np.array(axes).flatten()) if n > 1 else [axes]

    colors = [ps.NAVY, ps.TEAL, ps.VIOLET, ps.MAUVE, ps.RED, ps.GRAY]

    for idx, concept in enumerate(concepts):
        ax = axes_flat[idx]
        scores, token_labels = results[concept]

        rel_positions = sorted(scores.keys())
        means = [float(np.mean(scores[r])) if scores[r] else 0.0 for r in rel_positions]
        stds = [float(np.std(scores[r])) / max(len(scores[r]) ** 0.5, 1) for r in rel_positions]

        x_labels = [f"{_clean_tok(token_labels[r])}\n[−{r}]" for r in rel_positions]

        bar_colors = [colors[i % len(colors)] for i in range(len(rel_positions))]
        bars = ax.bar(
            range(len(rel_positions)), means, yerr=stds, color=bar_colors, capsize=4, alpha=0.85
        )

        best = int(np.argmax(means))
        bars[best].set_edgecolor("black")
        bars[best].set_linewidth(2.0)

        ax.set_xticks(range(len(rel_positions)))
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.axhline(0, color=ps.GRAY, linewidth=0.8, linestyle="--")

        layer = _peak_causal_layer(concept, fallback_layer)
        ax.set_title(f"{concept}  (L{layer})", fontsize=10)
        ax.set_ylabel("cos(grad, Δh)  (attribution)")
        ax.set_xlabel("Token position from end  (−0 = last)")

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Positional attribution: which end-of-expression token carries most concept info?\n"
        "Score = cos(grad, Δh) — normalised for magnitude  |  highlighted bar = winner per concept",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved positional attribution plot → %s", out_path)


# ── full layer-sweep heatmap ──────────────────────────────────────────────────


def _draw_sweep_heatmap(
    ax,
    concept: str,
    scores: dict[int, dict[int, list[float]]],
    token_labels: list[str],
    prompt_example: str | None = None,
    show_ylabel: bool = True,
    show_colorbar: bool = True,
    fig=None,
) -> None:
    """Draw a positional sweep heatmap onto an existing axis."""
    layers = sorted(scores.keys())
    n_tail = len(token_labels)

    mat = np.zeros((len(layers), n_tail))
    for row, l in enumerate(layers):
        for col in range(n_tail):
            vals = scores[l][col]
            mat[row, col] = float(np.mean(vals)) if vals else 0.0

    winner_per_layer = mat.argmax(axis=1)

    vmax = max(abs(mat).max(), 1e-6)
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    if show_colorbar and fig is not None:
        fig.colorbar(im, ax=ax, label="cos(grad, Δh)", shrink=0.9)

    ax.plot(
        winner_per_layer,
        np.arange(len(layers)),
        color="white",
        linewidth=1.8,
        alpha=0.9,
    )
    ax.scatter(winner_per_layer, np.arange(len(layers)), color="white", s=14, zorder=5, alpha=0.8)

    ax.set_xticks(range(n_tail))
    ax.set_xticklabels(
        [f"{_clean_tok(token_labels[i])}\n[−{i}]" for i in range(n_tail)], fontsize=8
    )
    ax.set_yticks(range(0, len(layers), max(1, len(layers) // 12)))
    if show_ylabel:
        ax.set_yticklabels(
            [str(layers[i]) for i in range(0, len(layers), max(1, len(layers) // 12))],
            fontsize=8,
        )
        ax.set_ylabel("Layer", fontsize=9)
    else:
        ax.set_yticklabels([])

    ax.set_xlabel("Token position from end  (−0 = last)", fontsize=9)

    title = f"{concept}\nwhite curve = winning position per layer"
    if prompt_example:
        # Truncate long prompts to keep the title readable
        max_chars = 52
        disp = prompt_example if len(prompt_example) <= max_chars else prompt_example[:max_chars - 1] + "…"
        title = f"{concept}\n{disp}"
    ax.set_title(title, fontsize=9)


def plot_positional_sweep_combined(
    concept: str,
    template_results: dict[str, tuple[dict[int, dict[int, list[float]]], list[str]]],
    template_prompts: dict[str, str],
    out_path: Path,
) -> None:
    """One figure with three side-by-side heatmap subplots, one per template.

    template_results: {tmpl: (scores, token_labels)}
    template_prompts: {tmpl: example_prompt_pos string}
    """
    ps.apply()
    tmpls = sorted(template_results.keys())
    n = len(tmpls)
    n_tail = len(next(iter(template_results.values()))[1])

    fig, axes = plt.subplots(
        1, n,
        figsize=(max(5, n_tail * 1.0) * n + 0.5, 7),
        gridspec_kw={"wspace": 0.06},
    )
    if n == 1:
        axes = [axes]

    for i, tmpl in enumerate(tmpls):
        scores, token_labels = template_results[tmpl]
        prompt_ex = template_prompts.get(tmpl)
        _draw_sweep_heatmap(
            axes[i],
            tmpl,
            scores,
            token_labels,
            prompt_example=prompt_ex,
            show_ylabel=(i == 0),
            show_colorbar=(i == n - 1),
            fig=fig,
        )

    fig.suptitle(
        f"{concept.replace('_', ' ')}  —  positional attribution by template",
        fontsize=11,
        y=1.01,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved combined sweep figure → %s", out_path)


# ── sweep summary chart ───────────────────────────────────────────────────────


def plot_sweep_summary(
    concept_summaries: list[tuple[str, int, str, float]],
    out_path: Path,
) -> None:
    """Horizontal bar chart: one bar per concept, showing peak sweep score and best anchor token.

    concept_summaries: list of (concept, best_pos_from_end, token_label, peak_score).
    """
    concepts = [c for c, _, _, _ in concept_summaries]
    positions = [pos for _, pos, _, _ in concept_summaries]
    tok_labels = [tok for _, _, tok, _ in concept_summaries]
    scores = [s for _, _, _, s in concept_summaries]

    ps.apply()
    n = len(concepts)
    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.52)))

    y = np.arange(n)
    bar_colors = [ps.TEAL if s >= 0.3 else ps.NAVY for s in scores]
    bars = ax.barh(y, scores, color=bar_colors, alpha=0.85, height=0.62)

    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in concepts], fontsize=9)

    x_max = max(scores) if scores else 1.0
    for bar, pos, tok in zip(bars, positions, tok_labels):
        label = f"{_clean_tok(tok)}  [−{pos}]"
        ax.text(
            bar.get_width() + 0.005 * x_max,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_xlim(0, x_max * 1.35)
    ax.axvline(0, color=ps.GRAY, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Peak cos(grad, Δh) across all layers", fontsize=10)
    ax.set_title(
        "Best anchor token per concept — positional attribution sweep\n"
        "bar label shows token and distance from sequence end",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved sweep summary → %s", out_path)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--concepts", nargs="+", default=ALL_CONCEPTS, help="Which concepts to run (default: all)"
    )
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--n", type=int, default=50, help="Pairs per concept (first n used)")
    parser.add_argument(
        "--layer",
        type=int,
        default=_DEFAULT_LAYER,
        help="Single-layer fallback when no causal results exist (ignored with --sweep)",
    )
    parser.add_argument(
        "--anchor",
        default="delimiter",
        choices=["delimiter", "last"],
        help="Anchor token selection: 'delimiter' uses the last structural delimiter (default); 'last' uses the final token",
    )
    parser.add_argument(
        "--n_layers", type=int, default=36, help="Total number of model layers (used with --sweep)"
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Sweep all layers and produce a heatmap per concept"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_dir", default=None, help="Output directory (default: runs/concept_localization)"
    )
    args = parser.parse_args()

    _base_dir = Path(args.out_dir or _RUNS_DIR)
    _base_dir.mkdir(parents=True, exist_ok=True)

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

    all_layers = list(range(args.n_layers))

    if args.sweep:
        out_dir = _base_dir / "anchor_token_sweep"
        out_dir.mkdir(parents=True, exist_ok=True)
        concept_summaries: list[tuple[str, int, str, float]] = []

        for concept in args.concepts:
            log.info("Concept: %s  (sweep, %d layers)", concept, args.n_layers)
            # Load up to args.n pairs per template; group before sweeping
            all_pairs = _load_pairs(concept, args.n, args.seed)
            template_keys = list(dict.fromkeys(p.template for p in all_pairs))
            groups = {t: [p for p in all_pairs if p.template == t][: args.n] for t in template_keys}

            global_best_score = -float("inf")
            global_best_pos = 0
            global_best_tok = "?"

            # Collect per-template results for combined figure
            template_results: dict[str, tuple] = {}
            template_prompts: dict[str, str] = {}

            for tmpl, grp in groups.items():
                log.info("  template %s  (%d pairs)", tmpl, len(grp))

                scores, token_labels = run_positional_attribution_sweep(
                    model,
                    grp,
                    layers=all_layers,
                    device=device,
                    dtype=dtype,
                    anchor=args.anchor,
                )

                example_prompt = grp[0].prompt_pos if grp else None
                template_results[tmpl] = (scores, token_labels)
                if example_prompt:
                    template_prompts[tmpl] = example_prompt

                for l in all_layers:
                    rel_positions = sorted(scores[l].keys())
                    means = {
                        r: float(np.mean(scores[l][r])) if scores[l][r] else 0.0
                        for r in rel_positions
                    }
                    best = max(means, key=lambda r: means[r])
                    tok = token_labels[best] if best < len(token_labels) else "?"
                    log.info(
                        "  [%s] L%-2d  best=−%d (%r)  score=%.4f", tmpl, l, best, tok, means[best]
                    )
                    if means[best] > global_best_score:
                        global_best_score = means[best]
                        global_best_pos = best
                        global_best_tok = tok

            concept_summaries.append((concept, global_best_pos, global_best_tok, global_best_score))
            log.info(
                "  %s  →  global best pos=−%d (%r)  score=%.4f",
                concept,
                global_best_pos,
                global_best_tok,
                global_best_score,
            )

            if len(template_results) > 1:
                plot_positional_sweep_combined(
                    concept,
                    template_results,
                    template_prompts,
                    out_dir / f"{concept}_combined_positional_sweep.png",
                )

        plot_sweep_summary(concept_summaries, out_dir / "positional_sweep_summary.png")

    else:
        out_dir = _base_dir
        all_results: dict[str, tuple[dict[int, list[float]], list[str]]] = {}

        for concept in args.concepts:
            layer = _peak_causal_layer(concept, args.layer)
            log.info("Concept: %s  →  layer %d", concept, layer)

            pairs = _load_pairs(concept, args.n, args.seed)[: args.n]

            scores, token_labels = run_positional_attribution(
                model, pairs, layer=layer, device=device, dtype=dtype, anchor=args.anchor
            )

            all_results[concept] = (scores, token_labels)

            rel_positions = sorted(scores.keys())
            means = {r: float(np.mean(scores[r])) if scores[r] else 0.0 for r in rel_positions}
            best = max(means, key=lambda r: means[r])
            log.info(
                "  %s: best position = −%d (%r)  score=%.4f",
                concept,
                best,
                token_labels[best] if best < len(token_labels) else "?",
                means[best],
            )
            for r in rel_positions:
                tok = token_labels[r] if r < len(token_labels) else "?"
                log.info("    pos −%d  %r  mean=%.4f  n=%d", r, tok, means[r], len(scores[r]))

        plot_positional_attribution(
            all_results,
            out_dir / "positional_attribution.png",
            fallback_layer=args.layer,
        )


if __name__ == "__main__":
    main()
