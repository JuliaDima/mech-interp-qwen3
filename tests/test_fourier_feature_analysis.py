"""Tests for fourier_feature_analysis.py.

Covers:
  - signed_freq: FFT-index to signed-frequency conversion
  - classify_mode: all eight mode categories
  - mode_energy_breakdown: energy fractions match pure synthetic signals
  - fourier_decompose_matrix / fourier_reconstruct / fourier_r2: round-trip fidelity
  - find_min_k: stops at minimal K for a single-mode signal
  - dominant_mode_direction: returns correct label
  - _parse_layer_sel: all selector forms
  - _aggregate_grids: mean-per-cell computation
  - _score_grids_batch: agrees with mode_energy_breakdown; pure signals score correctly
  - _tex_escape: percent escaping does NOT double-escape
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.concept_localization.concept_fits.fourier_feature_analysis import (
    _aggregate_grids,
    _build_scan_masks,
    _parse_layer_sel,
    _score_grids_batch,
    _tex_escape,
    classify_mode,
    dominant_mode_direction,
    find_min_k,
    fourier_decompose_matrix,
    fourier_r2,
    fourier_reconstruct,
    mode_energy_breakdown,
    signed_freq,
)

N = 10


# ── helpers ───────────────────────────────────────────────────────────────────

def pure_mode_grid(u: int, v: int, N: int = 10, phase: float = 0.0) -> np.ndarray:
    """Return a grid = cos(2π(u·a + v·b)/N + phase), shape (N, N)."""
    a = np.arange(N)
    b = np.arange(N)
    A, B = np.meshgrid(a, b, indexing="ij")
    return np.cos(2 * math.pi * (u * A + v * B) / N + phase)


# ── signed_freq ───────────────────────────────────────────────────────────────

class TestSignedFreq:
    def test_zero(self):
        assert signed_freq(0, 10) == 0

    def test_positive_low(self):
        assert signed_freq(3, 10) == 3

    def test_nyquist(self):
        assert signed_freq(5, 10) == 5

    def test_negative_wrap(self):
        assert signed_freq(9, 10) == -1

    def test_halfway_negative(self):
        assert signed_freq(6, 10) == -4


# ── classify_mode ─────────────────────────────────────────────────────────────

class TestClassifyMode:
    def test_dc(self):
        assert classify_mode(0, 0) == "mean / DC"

    def test_col_only(self):
        # u=0, v≠0 → depends on b only
        assert classify_mode(0, 3) == "row-only / b-only"

    def test_row_only(self):
        # v=0, u≠0 → depends on a only
        assert classify_mode(3, 0) == "column-only / a-only"

    def test_parity(self):
        # parity must be classified BEFORE sum/diff (5,5 satisfies u==v but is parity)
        assert classify_mode(5, 5) == "parity / (-1)^(a+b)"
        assert classify_mode(-5, 5) == "parity / (-1)^(a+b)"
        assert classify_mode(5, -5) == "parity / (-1)^(a+b)"

    def test_sum(self):
        assert classify_mode(2, 2) == "iso-sum / a+b"
        assert classify_mode(-3, -3) == "iso-sum / a+b"

    def test_diff(self):
        assert classify_mode(1, -1) == "iso-difference / b-a"
        assert classify_mode(-4, 4) == "iso-difference / b-a"

    def test_mixed(self):
        assert classify_mode(1, 2) == "mixed"
        assert classify_mode(3, -2) == "mixed"

    def test_row_parity(self):
        assert classify_mode(5, 0) == "row-parity / (-1)^a"

    def test_col_parity(self):
        assert classify_mode(0, 5) == "col-parity / (-1)^b"


# ── mode_energy_breakdown ─────────────────────────────────────────────────────

class TestModeEnergyBreakdown:
    def _C_for(self, grid: np.ndarray) -> np.ndarray:
        return np.fft.fft2(grid - grid.mean()) / (N * N)

    def test_pure_diff_signal(self):
        # X = cos(2π(b-a)/N): u=-1, v=1  → 100% diff
        X = pure_mode_grid(-1, 1)
        C = self._C_for(X)
        bd = mode_energy_breakdown(C)
        assert bd["diff"] == pytest.approx(1.0, abs=1e-6)
        for k in ("sum", "parity", "row", "col", "mixed"):
            assert bd[k] == pytest.approx(0.0, abs=1e-6)

    def test_pure_sum_signal(self):
        X = pure_mode_grid(2, 2)
        C = self._C_for(X)
        bd = mode_energy_breakdown(C)
        assert bd["sum"] == pytest.approx(1.0, abs=1e-6)

    def test_pure_parity_signal(self):
        # parity: (5, 5) after signed conversion
        X = pure_mode_grid(5, 5)
        C = self._C_for(X)
        bd = mode_energy_breakdown(C)
        assert bd["parity"] == pytest.approx(1.0, abs=1e-6)

    def test_pure_row_signal(self):
        # row: v=0, u=3 → depends only on a
        X = pure_mode_grid(3, 0)
        C = self._C_for(X)
        bd = mode_energy_breakdown(C)
        assert bd["row"] == pytest.approx(1.0, abs=1e-6)

    def test_pure_col_signal(self):
        X = pure_mode_grid(0, 2)
        C = self._C_for(X)
        bd = mode_energy_breakdown(C)
        assert bd["col"] == pytest.approx(1.0, abs=1e-6)

    def test_fractions_sum_to_one(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((N, N))
        C = self._C_for(X)
        bd = mode_energy_breakdown(C)
        assert sum(bd.values()) == pytest.approx(1.0, abs=1e-6)

    def test_flat_grid_returns_zeros(self):
        X = np.ones((N, N))
        C = self._C_for(X)
        bd = mode_energy_breakdown(C)
        assert all(v == 0.0 for v in bd.values())


# ── fourier round-trip ────────────────────────────────────────────────────────

class TestFourierRoundTrip:
    def test_single_mode_reconstructs_exactly(self):
        # A pure cosine with a fixed phase should reconstruct perfectly with K=1
        X = pure_mode_grid(-1, 1, phase=0.7)
        mu, modes, C = fourier_decompose_matrix(X, K=1)
        Xhat = fourier_reconstruct(mu, modes, N=N)
        assert fourier_r2(X, Xhat) == pytest.approx(1.0, abs=1e-5)

    def test_mean_is_subtracted_correctly(self):
        X = pure_mode_grid(1, -1) + 3.0
        mu, modes, C = fourier_decompose_matrix(X, K=1)
        assert mu == pytest.approx(3.0, abs=1e-6)

    def test_r2_improves_with_more_modes(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((N, N))
        _, _, _, _, _, Xhat1 = find_min_k(X, r2_target=1.1, k_max=1)  # force k=1
        _, _, _, _, _, Xhat8 = find_min_k(X, r2_target=1.1, k_max=8)
        r2_1 = fourier_r2(X, Xhat1)
        r2_8 = fourier_r2(X, Xhat8)
        assert r2_8 >= r2_1

    def test_reconstruct_uses_all_modes(self):
        X = pure_mode_grid(2, -3, phase=1.2)
        mu, modes, C = fourier_decompose_matrix(X, K=4)
        Xhat = fourier_reconstruct(mu, modes, N=N)
        assert fourier_r2(X, Xhat) == pytest.approx(1.0, abs=1e-4)

    def test_constant_grid_r2_is_nan(self):
        X = np.ones((N, N)) * 5.0
        mu, modes, C = fourier_decompose_matrix(X, K=1)
        Xhat = fourier_reconstruct(mu, modes, N=N)
        r2 = fourier_r2(X, Xhat)
        assert math.isnan(r2)


# ── find_min_k ────────────────────────────────────────────────────────────────

class TestFindMinK:
    def test_single_mode_needs_k1(self):
        X = pure_mode_grid(-1, 1, phase=0.3)
        k, r2, modes, mu, C, Xhat = find_min_k(X, r2_target=0.99, k_max=8)
        assert k == 1
        assert r2 == pytest.approx(1.0, abs=1e-4)

    def test_falls_back_to_kmax_when_target_not_reached(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((N, N))
        k, r2, modes, mu, C, Xhat = find_min_k(X, r2_target=0.9999, k_max=3)
        assert k == 3

    def test_returns_six_tuple(self):
        X = pure_mode_grid(1, -1)
        result = find_min_k(X, r2_target=0.9, k_max=4)
        assert len(result) == 6
        k_used, r2, modes, mu, C, Xhat = result
        assert isinstance(k_used, int)
        assert isinstance(modes, list)
        assert Xhat.shape == (N, N)


# ── dominant_mode_direction ───────────────────────────────────────────────────

class TestDominantModeDirection:
    def test_dominant_diff(self):
        bd = {"diff": 0.8, "sum": 0.1, "parity": 0.05, "row": 0.03, "col": 0.01, "mixed": 0.01}
        result = dominant_mode_direction(bd)
        assert result.startswith("iso-difference")

    def test_dominant_parity(self):
        bd = {"diff": 0.1, "sum": 0.1, "parity": 0.6, "row": 0.1, "col": 0.05, "mixed": 0.05}
        result = dominant_mode_direction(bd)
        assert result.startswith("parity")

    def test_secondary_above_threshold_appears(self):
        bd = {"diff": 0.6, "sum": 0.3, "parity": 0.0, "row": 0.05, "col": 0.05, "mixed": 0.0}
        result = dominant_mode_direction(bd)
        assert "sum" in result  # secondary is reported


# ── _parse_layer_sel ──────────────────────────────────────────────────────────

class TestParseLayerSel:
    AVAILABLE = list(range(36))

    def test_all(self):
        assert _parse_layer_sel("all", self.AVAILABLE) == self.AVAILABLE

    def test_single(self):
        assert _parse_layer_sel("13", self.AVAILABLE) == [13]

    def test_comma_list(self):
        assert _parse_layer_sel("11,13,15", self.AVAILABLE) == [11, 13, 15]

    def test_range(self):
        assert _parse_layer_sel("11-14", self.AVAILABLE) == [11, 12, 13, 14]

    def test_mixed(self):
        assert _parse_layer_sel("0,11-13,35", self.AVAILABLE) == [0, 11, 12, 13, 35]

    def test_filters_unavailable(self):
        assert _parse_layer_sel("1,99,2", [1, 2, 3]) == [1, 2]

    def test_all_filters_to_available(self):
        assert _parse_layer_sel("all", [5, 10]) == [5, 10]


# ── _aggregate_grids ──────────────────────────────────────────────────────────

class TestAggregateGrids:
    def test_single_cell_mean(self):
        # Two examples both in cell (2, 3), single feature
        acts = np.array([[1.0], [3.0]], dtype=np.float32)
        a_mod = np.array([2, 2])
        b_mod = np.array([3, 3])
        grids = _aggregate_grids(acts, a_mod, b_mod, N=5)
        assert grids.shape == (1, 5, 5)
        assert grids[0, 2, 3] == pytest.approx(2.0)

    def test_separate_cells(self):
        acts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        a_mod = np.array([0, 1])
        b_mod = np.array([0, 1])
        grids = _aggregate_grids(acts, a_mod, b_mod, N=4)
        # Feature 0: cell(0,0)=1.0, cell(1,1)=3.0
        assert grids[0, 0, 0] == pytest.approx(1.0)
        assert grids[0, 1, 1] == pytest.approx(3.0)
        # Feature 1: cell(0,0)=2.0, cell(1,1)=4.0
        assert grids[1, 0, 0] == pytest.approx(2.0)
        assert grids[1, 1, 1] == pytest.approx(4.0)

    def test_empty_cells_are_nan(self):
        acts = np.array([[1.0]], dtype=np.float32)
        a_mod = np.array([0])
        b_mod = np.array([0])
        grids = _aggregate_grids(acts, a_mod, b_mod, N=3)
        # All cells except (0,0) should be NaN
        assert np.isnan(grids[0, 1, 2])
        assert np.isnan(grids[0, 2, 0])

    def test_multiple_features(self):
        # 4 examples, 3 features
        acts = np.arange(12, dtype=np.float32).reshape(4, 3)
        a_mod = np.array([0, 0, 1, 2])
        b_mod = np.array([0, 0, 1, 2])
        grids = _aggregate_grids(acts, a_mod, b_mod, N=5)
        assert grids.shape == (3, 5, 5)
        # Cell (0,0): mean of rows 0 and 1 for each feature
        for f in range(3):
            expected = (acts[0, f] + acts[1, f]) / 2
            assert grids[f, 0, 0] == pytest.approx(expected)


# ── _score_grids_batch ────────────────────────────────────────────────────────

class TestScoreGridsBatch:
    def test_pure_diff_high_diff_energy(self):
        X = pure_mode_grid(-1, 1).astype(np.float32)
        X = (X - X.min()) / (X.max() - X.min())  # normalise to [0,1]
        scores = _score_grids_batch(X[None], N=N)
        assert scores["diff"][0] == pytest.approx(1.0, abs=1e-4)
        assert scores["structured_energy"][0] == pytest.approx(1.0, abs=1e-4)

    def test_pure_parity_high_parity_energy(self):
        X = pure_mode_grid(5, 5).astype(np.float32)
        X = (X - X.min()) / (X.max() - X.min())
        scores = _score_grids_batch(X[None], N=N)
        assert scores["parity"][0] == pytest.approx(1.0, abs=1e-4)

    def test_fractions_sum_to_one(self):
        rng = np.random.default_rng(7)
        grids = rng.standard_normal((5, N, N)).astype(np.float32)
        scores = _score_grids_batch(grids, N=N)
        cats = ["diff", "sum", "parity", "row", "col", "mixed"]
        for i in range(5):
            total = sum(scores[c][i] for c in cats)
            assert total == pytest.approx(1.0, abs=1e-5)

    def test_batch_matches_mode_energy_breakdown(self):
        # _score_grids_batch and mode_energy_breakdown must agree on energy fractions
        X = pure_mode_grid(2, -3).astype(np.float32)
        X = (X - X.min()) / (X.max() - X.min() + 1e-12)

        scores = _score_grids_batch(X[None], N=N)

        Xd = X.astype(np.float64)
        C = np.fft.fft2(Xd - Xd.mean()) / (N * N)
        bd = mode_energy_breakdown(C)

        for cat in ("diff", "sum", "parity", "row", "col", "mixed"):
            assert scores[cat][0] == pytest.approx(bd[cat], abs=1e-4), \
                f"Mismatch for category '{cat}'"

    def test_structured_energy_definition(self):
        rng = np.random.default_rng(3)
        grids = rng.standard_normal((3, N, N)).astype(np.float32)
        scores = _score_grids_batch(grids, N=N)
        for i in range(3):
            expected = scores["diff"][i] + scores["sum"][i] + scores["parity"][i]
            assert scores["structured_energy"][i] == pytest.approx(expected, abs=1e-8)

    def test_output_shape(self):
        grids = np.zeros((7, N, N), dtype=np.float32)
        scores = _score_grids_batch(grids, N=N)
        for k in ("diff", "sum", "parity", "row", "col", "mixed", "structured_energy", "top_mode_amp"):
            assert scores[k].shape == (7,)
        assert len(scores["top_mode_type"]) == 7


# ── _tex_escape ───────────────────────────────────────────────────────────────

class TestTexEscape:
    def test_percent_escapes_to_backslash_percent(self):
        result = _tex_escape("50%")
        assert result == r"50\%"

    def test_plain_percent_not_double_escaped(self):
        # A plain "%" string must become "\%", not "\textbackslash{}\%"
        result = _tex_escape("%")
        assert result == r"\%"
        assert "textbackslash" not in result

    def test_underscore(self):
        assert _tex_escape("a_b") == r"a\_b"

    def test_ampersand(self):
        assert _tex_escape("a & b") == r"a \& b"

    def test_dollar(self):
        assert _tex_escape("$x$") == r"\$x\$"

    def test_no_latex_in_plain_text(self):
        assert _tex_escape("hello world") == "hello world"

    def test_backslash_escaped_to_textbackslash(self):
        result = _tex_escape("a\\b")
        assert "textbackslash" in result

    def test_energy_summary_typical_output(self):
        # Simulate what _feature_tex_page builds for bd_summary
        # Keys and values produce "diff: 51%" which _tex_escape should make "diff: 51\%"
        bd_summary = "diff: 51%; mixed: 25%"
        result = _tex_escape(bd_summary)
        assert result == r"diff: 51\%; mixed: 25\%"
        assert "{}" not in result  # no leftover textbackslash artifacts
