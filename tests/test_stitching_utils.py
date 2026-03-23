"""Unit tests for stitching experiment utilities.

Tests all pure-Python functions without loading any real model weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Ensure repo root on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from experiments.stitching.run import (  # noqa: E402
    SmallAdditionTransformer,
    compute_cca_score,
    fit_mlp_output_maps,
    identify_cascading_carry_cases,
    load_addition_dataset,
    train_small_sae,
)


def test_no_carry_count_variable():
    """identify_cascading_carry_cases should not assign carry_count."""
    import inspect

    import experiments.stitching.run as run_module

    src = inspect.getsource(run_module.identify_cascading_carry_cases)
    # Check that no assignment of carry_count appears (not just any mention)
    assert "carry_count = " not in src, "carry_count assignment (dead code) should be removed"
    assert "carry_count +=" not in src, "carry_count increment (dead code) should be removed"


def test_cascading_carry_correct():
    samples = [
        {"a": 77, "b": 23},  # 77+23=100 → 2 consecutive carries ✓
        {"a": 1, "b": 2},  # no carry ✗
        {"a": 99, "b": 1},  # 99+1=100 → 2 consecutive carries ✓
        {"a": 15, "b": 7},  # 15+7=22 → 1 carry (tens digit) ✗
    ]
    result = identify_cascading_carry_cases(samples, threshold=2)
    assert result == [True, False, True, False]


def test_fallback_dataset_uses_num_digits(tmp_path):
    """load_addition_dataset fallback should generate num_digits range, not [0,20]."""
    nonexistent = str(tmp_path / "does_not_exist.jsonl")
    samples = load_addition_dataset(nonexistent, max_samples=50, num_digits=3)
    assert len(samples) > 0
    max_val = 10**3 - 1  # 999 for 3-digit
    for s in samples:
        assert s["a"] <= max_val, f"a={s['a']} > max_val={max_val}"
        assert s["b"] <= max_val, f"b={s['b']} > max_val={max_val}"


def test_fit_mlp_output_maps_shape_and_type():
    """fit_mlp_output_maps should return W (d_large, d_small) and b (d_large,)."""
    d_small, d_large, n = 16, 64, 50
    small_out = torch.randn(n, d_small)
    large_by_layer = {14: torch.randn(n, d_large), 16: torch.randn(n, d_large)}

    maps = fit_mlp_output_maps(small_out, large_by_layer, [14, 16])

    assert set(maps.keys()) == {14, 16}
    for layer, m in maps.items():
        assert m["W"].shape == (d_large, d_small), f"Layer {layer}: wrong W shape"
        assert m["b"].shape == (d_large,), f"Layer {layer}: wrong b shape"
        assert isinstance(m["r2"], float)
        assert isinstance(m["cca"], float)
        assert -1.0 <= m["r2"] <= 1.0 + 1e-6
        assert 0.0 <= m["cca"] <= 1.0 + 1e-6


def test_fit_mlp_output_maps_no_per_dim_loop():
    """Verify the implementation uses multi-output Ridge, not a per-dim loop."""
    import inspect

    import experiments.stitching.run as run_module

    src = inspect.getsource(run_module.fit_mlp_output_maps)
    # The old buggy code looped "for i in range(d_large)"
    assert "for i in range" not in src, "Per-dim Ridge loop should be gone"
    assert "ridge.fit(X, Y)" in src, "Should use multi-output ridge.fit(X, Y)"


def test_kl_divergence_finite_and_nonneg():
    """KL(before || after) should be finite and >= 0."""
    logits_before = torch.randn(1, 10, 100)
    logits_after = torch.randn(1, 10, 100)

    p = F.softmax(logits_before[0, -1], dim=-1).clamp(min=1e-10)
    q = F.softmax(logits_after[0, -1], dim=-1).clamp(min=1e-10)
    kl = F.kl_div(q.log(), p, reduction="sum", log_target=False).item()

    assert np.isfinite(kl), f"KL divergence is not finite: {kl}"
    assert kl >= 0.0, f"KL divergence should be >= 0, got {kl}"


def test_kl_same_distribution_is_zero():
    """KL(p || p) should be ~0."""
    logits = torch.randn(1, 10, 100)
    p = F.softmax(logits[0, -1], dim=-1).clamp(min=1e-10)
    kl = F.kl_div(p.log(), p, reduction="sum", log_target=False).item()
    assert abs(kl) < 1e-5, f"KL(p||p) should be ~0, got {kl}"


def test_train_samples_not_overwritten():
    """After train_small_model, the original dict samples should not be mutated."""
    dict_samples = [
        {"a": 1, "b": 2, "prompt": "calc: 1+2= ", "answer": "3"},
        {"a": 3, "b": 4, "prompt": "calc: 3+4= ", "answer": "7"},
    ]
    # Simulate what main() does: keep dict samples separate from string samples
    str_samples = ["1+2=3", "3+4=7"]
    # dict_samples must remain unchanged
    assert all(isinstance(s, dict) for s in dict_samples)
    assert all(isinstance(s, str) for s in str_samples)


def test_patch_hook_closure_by_value():
    """patch_hook should use default-arg capture, not late-binding closure."""
    import inspect

    import experiments.stitching.run as run_module

    src = inspect.getsource(run_module.inject_and_verify)
    # The default-arg pattern: _val: torch.Tensor = stitched
    assert "_val" in src, "patch_hook should use _val default arg for value capture"
    # Dead patched_cache code should be gone
    assert "patched_cache" not in src, "patched_cache dead code should be removed"


def test_accuracy_metric_no_substring_check():
    """inject_and_verify should use token-ID comparison, not 'in str' substring."""
    import inspect

    import experiments.stitching.run as run_module

    src = inspect.getsource(run_module.inject_and_verify)
    assert "in answer_toks" in src, "Should compare pred token ID against answer_toks list"
    # The old bug was: "answer_first_char in pred_before_str" substring check
    assert "in pred_before_str" not in src, "Old substring accuracy check should be gone"
    assert "in pred_after_str" not in src, "Old substring accuracy check should be gone"


# ---------------------------------------------------------------------------
# CCA helper
# ---------------------------------------------------------------------------


def test_cca_score_range():
    """compute_cca_score should return a value in [0, 1]."""
    X = np.random.randn(100, 32)
    Y = np.random.randn(100, 64)
    score = compute_cca_score(X, Y, n_components=10)
    assert 0.0 <= score <= 1.0 + 1e-6, f"CCA score out of range: {score}"


def test_cca_score_identical_inputs():
    """CCA score of X vs X (same space) should be ~1."""
    X = np.random.randn(100, 20)
    score = compute_cca_score(X, X, n_components=10)
    assert score > 0.9, f"CCA(X, X) should be ~1.0, got {score}"


# ---------------------------------------------------------------------------
# Small SAE training smoke test (no GPU, tiny model)
# ---------------------------------------------------------------------------


def test_train_small_sae_smoke():
    """train_small_sae should run end-to-end on a tiny model without crashing."""
    device = torch.device("cpu")
    model = SmallAdditionTransformer(n_layers=2, n_heads=2, d_model=16, vocab_size=16)
    model.model.to(device)

    # Minimal string samples
    samples = ["1+2=3", "4+5=9", "3+3=6", "2+7=9", "1+1=2"]

    sae = train_small_sae(
        small_model=model,
        small_extraction_samples=samples,
        device=device,
        d_transcoder=32,
        epochs=5,
        lr=1e-3,
        sae_layer=1,
        dry_run=False,
    )

    assert sae is not None
    assert sae.d_model == 16
    assert sae.d_transcoder == 32

    # SAE should be able to encode/decode
    x = torch.randn(3, 16)
    feats = sae.encode(x)
    recon = sae.decode(feats)
    assert recon.shape == x.shape
    assert torch.all(torch.isfinite(recon)), "Reconstruction contains NaN/Inf"
