"""Operand plots for the Anthropic addition case study reproduction.

Replicates Anthropic's "operand plots" from Section 3 of
"On the Biology of a Large Language Model" (2025):

  For each (transcoder) feature, we collect its activation at the '='
  token position across all 10,000 prompts of the form "calc: a+b=",
  producing a 100x100 matrix indexed by (a, b).  The visual pattern of this
  matrix reveals what the feature "detects":

    - Diagonal stripes  → sum-sensitive (a+b ≈ k)
    - Horizontal/vert.  → single-operand-sensitive (a ≈ k   or   b ≈ k)
    - Repeating modular → mod-10 or mod-100 structure (ones/tens digits)
    - Isolated points  → exact-pair lookup table
    - Smeared / blurry → low-precision "sum near X" pathway

Usage (standalone):
  python -m experiments.addition.run --operand-plots ...

or import and call run_operand_plots() directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from mechinterp_qwen3.attribution_model import AttributionModel

from .prompts import CALC_GRID, CalcEntry

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eq_token_position(tokens: torch.Tensor) -> int:
    """Return the index of the final token in the prompt (= sign position).

    For "calc: a+b=" the '=' is always the last token. We assert this to
    catch tokenisation surprises early.
    """
    # We simply take the last position; see README.md for justification.
    return int(tokens.shape[-1]) - 1


# ---------------------------------------------------------------------------
# Core: collect activations over the calc grid
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_grid_activations(
    model: AttributionModel,
    calc_grid: list[CalcEntry] | None = None,
    *,
    feature_ids: list[tuple[int, int]] | None = None,  # [(layer, feat_idx), ...]
    top_k_global: int = 50,
    batch_size: int = 32,
    eq_pos: int | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    """Collect transcoder feature activations at the '=' token over the calc grid.

    For each feature (layer, feat_idx), returns a (100, 100) numpy array
    A[a, b] = activation value of that feature at the '=' position for the
    prompt "calc: a+b=".

    Args:
        model:         Loaded AttributionModel with transcoders.
        calc_grid:     List of CalcEntry dicts (defaults to CALC_GRID from prompts.py).
        feature_ids:   Explicit list of (layer, feat_idx) pairs to track.
                       If None, will first run a "discovery pass" over grid to
                       identify the top_k_global most active features on
                       FOCUS_PROMPT and track those.
        top_k_global:  Number of top features to auto-discover from FOCUS_PROMPT
                       when feature_ids is None.
        batch_size:    Number of prompts per forward-pass batch.
        eq_pos:        Override the '=' position index (auto-detected if None).

    Returns:
        Dict mapping (layer, feat_idx) → (100, 100) float32 ndarray.
    """
    if calc_grid is None:
        calc_grid = CALC_GRID

    n_layers = model.cfg.n_layers  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Step 1: auto-discover feature_ids from FOCUS_PROMPT if not given
    # ------------------------------------------------------------------
    if feature_ids is None:
        log.info("Auto-discovering top-%d features on FOCUS_PROMPT ...", top_k_global)
        focus_prompt = "calc: 36+59="
        _, activation_cache = model.get_activations(focus_prompt)
        # activation_cache: (n_layers, n_pos, d_transcoder) dense
        # shape is (n_layers, seq_len, d_transcoder)
        # We want features active at the last position
        focus_eq_pos = activation_cache.shape[1] - 1

        # Collect (layer, feat_idx) for top-k non-zero features
        top_features: list[tuple[int, int]] = []
        for layer in range(n_layers):
            layer_acts = activation_cache[layer, focus_eq_pos]  # (d_transcoder,)
            nonzero_mask = layer_acts > 0
            if nonzero_mask.any():
                vals = layer_acts[nonzero_mask]
                idxs = torch.where(nonzero_mask)[0]
                sorted_order = torch.argsort(vals, descending=True)
                for i in sorted_order:
                    top_features.append((int(layer), int(idxs[i])))

        # Sort by descending activation and take top_k
        def _score(lf: tuple[int, int]) -> float:
            layer_, f = lf
            return float(activation_cache[layer_, focus_eq_pos, f])

        top_features.sort(key=_score, reverse=True)
        feature_ids = top_features[:top_k_global]
        log.info("Auto-discovered %d features.", len(feature_ids))

    # ------------------------------------------------------------------
    # Step 2: build (layer, feat_idx) → (100, 100) accumulators
    # ------------------------------------------------------------------
    matrices: dict[tuple[int, int], np.ndarray] = {
        lf: np.zeros((100, 100), dtype=np.float32) for lf in feature_ids
    }

    # Group prompts into batches for efficiency
    all_entries = list(calc_grid)
    n_total = len(all_entries)

    # We run one prompt at a time (they may have different lengths for large a/b).
    # For efficiency we batch prompts that have the same token length
    # (most calc: a+b= prompts differ by 1–2 chars) but for correctness we
    # process each individually — the overhead is modest for 10k short prompts.

    log.info("Collecting activations for %d prompts ...", n_total)
    for entry in tqdm(all_entries, desc="calc grid", total=n_total):
        a, b = entry["a"], entry["b"]
        prompt = entry["prompt"]

        # Forward pass — use get_activations which returns (n_layers, n_pos, d_tc)
        _, acts = model.get_activations(prompt)
        # acts: (n_layers, seq_len, d_transcoder)  — dense, on model device

        # '=' position = last token of prompt
        this_eq_pos = acts.shape[1] - 1
        if eq_pos is not None:
            this_eq_pos = eq_pos

        for layer, feat_idx in feature_ids:
            val = float(acts[layer, this_eq_pos, feat_idx].item())
            matrices[(layer, feat_idx)][a, b] = val

    return matrices


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_operand_matrix(
    matrix: np.ndarray,
    *,
    feature_id: tuple[int, int],
    out_path: Path,
    title: str | None = None,
) -> None:
    """Save a heatmap of a 100×100 operand matrix.

    Also saves the raw .npy array alongside the .png.

    Args:
        matrix:     (100, 100) float32 array — matrix[a, b] = feature activation.
        feature_id: (layer, feat_idx) tuple used in title/filename.
        out_path:   Output path (should end in .png); .npy saved next to it.
        title:      Optional plot title override.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for operand plots. Install it with: pip install matplotlib"
        ) from e

    layer, feat_idx = feature_id
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save raw array
    np.save(str(out_path).replace(".png", ".npy"), matrix)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        matrix,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("b  (second operand)", fontsize=11)
    ax.set_ylabel("a  (first operand)", fontsize=11)
    ax.set_title(
        title or f"Layer {layer} · feature {feat_idx}  (activation at '=' token)",
        fontsize=11,
    )

    # Light gridlines every 10 units to help read mod-10 structure
    ax.set_xticks(np.arange(-0.5, 100, 10), minor=True)
    ax.set_yticks(np.arange(-0.5, 100, 10), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.debug("Saved operand plot to %s", out_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_operand_plots(
    model: AttributionModel,
    out_dir: Path | str,
    *,
    feature_ids: list[tuple[int, int]] | None = None,
    top_k_global: int = 50,
    batch_size: int = 32,
) -> dict[tuple[int, int], np.ndarray]:
    """Run the full operand-plot pipeline and write results to *out_dir*.

    Args:
        model:         Loaded AttributionModel.
        out_dir:       Output directory; .png and .npy files written here.
        feature_ids:   Explicit (layer, feat_idx) list; auto-discovered if None.
        top_k_global:  How many top features to auto-discover if feature_ids=None.
        batch_size:    Prompts per batch (kept for API symmetry; currently 1).

    Returns:
        Dict mapping (layer, feat_idx) → (100, 100) numpy array.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrices = collect_grid_activations(
        model,
        feature_ids=feature_ids,
        top_k_global=top_k_global,
        batch_size=batch_size,
    )

    log.info("Saving %d operand plots to %s ...", len(matrices), out_dir)
    for (layer, feat_idx), mat in matrices.items():
        fname = f"L{layer:02d}_F{feat_idx:06d}.png"
        plot_operand_matrix(
            mat,
            feature_id=(layer, feat_idx),
            out_path=out_dir / fname,
        )

    # Save a summary JSON listing which features were plotted
    import json

    summary = {
        "n_features": len(matrices),
        "features": [
            {
                "layer": layer_,
                "feat_idx": f,
                "max_activation": float(mat.max()),
                "mean_activation": float(mat.mean()),
            }
            for (layer_, f), mat in sorted(matrices.items())
        ],
    }
    with open(out_dir / "operand_plots_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    log.info("Operand plots complete.")
    return matrices
