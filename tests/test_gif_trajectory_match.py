"""Test that the GIF norm trajectory at the delimiter matches the concept localization run.

Loads two pre-computed artefacts from disk — no model required:
  - runs/concept_localization/carry/emergence.npy  (saved by make_gif.py)
  - runs/concept_localization/carry/deltas.pt      (saved by run_concept.py)

Checks that the raw ‖δ_l‖ trajectory at the delimiter position in the GIF
correlates strongly with the ‖δ_l‖ trajectory from the main concept run.

If either file is missing the test is skipped — run the GIF and concept
localization first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

_RUNS = Path(__file__).resolve().parents[1] / "runs" / "concept_localization" / "carry"
_GIF_NORMS = _RUNS / "emergence.npy"
_DELTAS = _RUNS / "deltas.pt"

pytestmark = pytest.mark.skipif(
    not _GIF_NORMS.exists() or not _DELTAS.exists(),
    reason="emergence.npy or deltas.pt not found — run make_gif and run_concept first",
)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


@pytest.fixture(scope="module")
def gif_data():
    return np.load(_GIF_NORMS, allow_pickle=True).item()


@pytest.fixture(scope="module")
def main_deltas():
    return torch.load(_DELTAS, map_location="cpu", weights_only=False)


def test_delimiter_frame_norm_trajectory_pearson(gif_data, main_deltas):
    """Pearson r between GIF delimiter-frame norms and main-run norms must exceed 0.90."""
    delim = gif_data["delimiter_pos"]
    layers = gif_data["layers"]

    gif_norms = gif_data["norms_raw"][delim]           # (n_layers,)

    agg = main_deltas.get("all", {})
    main_norms = np.array([
        agg[l].float().norm().item() if l in agg else 0.0
        for l in layers
    ])

    r = _pearson(gif_norms, main_norms)
    print(f"\n  delimiter_pos={delim}  n_pairs_gif={gif_data['n_pairs']}")
    print(f"  Pearson r (raw norms): {r:.4f}")

    assert r > 0.90, (
        f"Pearson r={r:.4f} below 0.90 — GIF and main-run norm trajectories diverge. "
        "Check that length filtering is applied in make_gif.py and that both runs "
        "use the same anchor token."
    )


def test_delimiter_frame_norm_trajectory_cosine(gif_data, main_deltas):
    """Cosine similarity between the two norm vectors (treated as 1-D signals) > 0.95."""
    delim = gif_data["delimiter_pos"]
    layers = gif_data["layers"]

    gif_norms = gif_data["norms_raw"][delim]
    agg = main_deltas.get("all", {})
    main_norms = np.array([
        agg[l].float().norm().item() if l in agg else 0.0
        for l in layers
    ])

    cos = _cosine(gif_norms, main_norms)
    print(f"\n  cosine similarity of norm vectors: {cos:.4f}")

    assert cos > 0.95, (
        f"Cosine similarity={cos:.4f} below 0.95 between GIF and main-run norm trajectories."
    )


def test_delimiter_position_nonzero(gif_data, main_deltas):
    """Sanity: the delimiter frame should have non-trivial norms (concept is active there)."""
    delim = gif_data["delimiter_pos"]
    gif_norms = gif_data["norms_raw"][delim]
    assert gif_norms.max() > 1e-6, "All norms zero at delimiter — extraction may have failed"


def test_gif_norms_shape(gif_data):
    """Saved norms array must have shape (n_frames, n_layers) with n_frames <= seq_len."""
    norms = gif_data["norms_raw"]
    delim = gif_data["delimiter_pos"]
    assert norms.ndim == 2
    assert norms.shape[0] == delim + 1, (
        f"Expected {delim + 1} frames (0..delimiter), got {norms.shape[0]}"
    )
