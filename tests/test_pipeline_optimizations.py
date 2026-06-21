"""Tests for pipeline optimization correctness.

Each test establishes that an optimized implementation produces the same
result as the existing reference implementation, within bfloat16 numerical
tolerance.  Run BEFORE and AFTER implementing the optimizations.

Tests:
  1. batched_residuals_match   — batched collect_layer_residuals == one-by-one
  2. null_from_cache           — null delta norms from cached H == from model re-run
  3. bfloat16_transcoder_ok    — bfloat16 preserves sparsity pattern vs float32
  4. float16_transcoder_warn   — float16 can change which features activate (documents the risk)
  5. all_anchors_single_pass   — extracting multiple anchors in one pass == separate passes

All model-loading tests require RDS and are auto-skipped without it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_HUB = Path("/rds/user/eid23/hpc-work/p28/cache/hf/hub")
_REQUIRES_RDS = pytest.mark.skipif(
    not (_HUB / "models--mwhanna--qwen3-4b-transcoders").exists(),
    reason="RDS cache not mounted",
)

_MODEL_ID = "Qwen/Qwen3-4B"
_TC_ID = "mwhanna/qwen3-4b-transcoders"

# Fixture: load model once per session so tests don't each pay the load cost.
@pytest.fixture(scope="session")
def model_bf16():
    # Check inside the fixture so session-scoped setup is skipped correctly.
    if not (_HUB / "models--mwhanna--qwen3-4b-transcoders").exists():
        pytest.skip("RDS cache not mounted")
    # Check that weight shards actually exist, not just the directory.
    qwen_dir = _HUB / "models--Qwen--Qwen3-4B"
    if not any(qwen_dir.rglob("*.safetensors")):
        pytest.skip("Qwen3-4B weights not in RDS cache")
    from mechinterp_qwen3.attribution_model import AttributionModel
    from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
    from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype
    dtype = parse_dtype("bfloat16")
    device = get_default_device()
    tc_set, _ = load_transcoder_from_hub(_TC_ID, dtype=dtype, lazy_encoder=True, lazy_decoder=True)
    m = AttributionModel.from_pretrained_and_transcoders(_MODEL_ID, tc_set, dtype=dtype, device=device)
    m.eval()
    return m


def _make_prompts(model):
    """Four short prompts: two pairs with the same token length."""
    prompts = [
        "calc: gcd(35,7)=",
        "calc: gcd(36,7)=",
        "calc: gcd(14,7)=",
        "calc: gcd(15,7)=",
    ]
    from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input
    device = model.cfg.device
    # Return (token_list, anchor=last_token) pairs — anchor at the "=" token
    result = []
    for p in prompts:
        ids = model.tokenizer(p, add_special_tokens=False).input_ids
        result.append((ids, len(ids) - 1))
    return result


# ---------------------------------------------------------------------------
# 1. Batched residual collection == one-by-one
# ---------------------------------------------------------------------------

@_REQUIRES_RDS
def test_batched_residuals_match(model_bf16):
    """collect_layer_residuals_batched must match the existing one-by-one version."""
    from experiments.concept_localization.analyze import collect_layer_residuals

    prompts_and_anchors = _make_prompts(model_bf16)
    layers = [0, 5, 10, 20, 35]

    # Reference: current one-by-one implementation
    H_ref = collect_layer_residuals(model_bf16, prompts_and_anchors, layers)

    # Optimized: batched (to be implemented in analyze.py)
    # Import will fail until implemented — test will ERROR, not PASS, until then.
    from experiments.concept_localization.analyze import collect_layer_residuals_batched
    H_batch = collect_layer_residuals_batched(model_bf16, prompts_and_anchors, layers, batch_size=2)

    for l in layers:
        # bfloat16 → float32 round-trip tolerance: ~1e-2 relative
        np.testing.assert_allclose(
            H_batch[l], H_ref[l], rtol=1e-2, atol=1e-3,
            err_msg=f"Layer {l}: batched vs one-by-one residuals differ"
        )


# ---------------------------------------------------------------------------
# 2. Null from cached H == null from model re-run
# ---------------------------------------------------------------------------

@_REQUIRES_RDS
def test_null_from_cache_matches_model(model_bf16):
    """Null delta norms computed by re-pairing cached H must equal those from
    running the model on shuffled pairs."""
    from experiments.concept_localization.analyze import collect_layer_residuals
    from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input

    prompts_and_anchors = _make_prompts(model_bf16)  # 4 prompts = 2 pairs
    layers = [0, 10, 35]
    rng = np.random.default_rng(0)

    # Collect H for all 4 prompts (interleaved pos/neg)
    H = collect_layer_residuals(model_bf16, prompts_and_anchors, layers)
    # H[l] shape: (4, d_model), indices 0,2 = pos, 1,3 = neg

    # Null from cache: re-pair pos[perm] - neg (fixed)
    pos_idx = np.array([0, 2])  # pos prompts
    neg_idx = np.array([1, 3])  # neg prompts
    perm = rng.permutation(len(pos_idx))
    null_from_cache = {}
    for l in layers:
        null_from_cache[l] = float(np.linalg.norm(
            H[l][pos_idx[perm]].mean(0) - H[l][neg_idx].mean(0)
        ))

    # Null from model: run the model again on the shuffled pairs
    shuffled_pairs = []
    for pi, ni in zip(pos_idx[perm], neg_idx):
        shuffled_pairs.append((prompts_and_anchors[pi][0], prompts_and_anchors[pi][1]))
        shuffled_pairs.append((prompts_and_anchors[ni][0], prompts_and_anchors[ni][1]))
    H_reshuffled = collect_layer_residuals(model_bf16, shuffled_pairs, layers)

    for l in layers:
        delta_from_model = float(np.linalg.norm(
            H_reshuffled[l][0::2].mean(0) - H_reshuffled[l][1::2].mean(0)
        ))
        assert abs(null_from_cache[l] - delta_from_model) < 1e-3, (
            f"Layer {l}: null_from_cache={null_from_cache[l]:.6f} "
            f"!= null_from_model={delta_from_model:.6f}"
        )


# ---------------------------------------------------------------------------
# 3. bfloat16 transcoder preserves sparsity pattern vs float32
# ---------------------------------------------------------------------------

@_REQUIRES_RDS
def test_bfloat16_transcoder_ok(model_bf16):
    """bfloat16 transcoder activations should match float32 within 1% of features."""
    from experiments.concept_localization.sweep_utils import apply_transcoder_all
    from experiments.concept_localization.analyze import collect_layer_residuals

    prompts_and_anchors = _make_prompts(model_bf16)
    layer = 10
    H = collect_layer_residuals(model_bf16, prompts_and_anchors, [layer])
    H_l = H[layer].astype(np.float32)  # (4, 2560)

    # bfloat16 path (current — model is already bfloat16)
    acts_bf16 = apply_transcoder_all(model_bf16, layer, H_l)  # (4, d_tc)

    # float32 path: upcast encoder weights temporarily
    tc = model_bf16.transcoders[layer]
    W_enc_f32 = tc.W_enc.detach().float()
    b_enc_f32 = tc.b_enc.detach().float()
    dev = W_enc_f32.device
    H_t = torch.from_numpy(H_l).to(dev)
    pre = H_t @ W_enc_f32.T + b_enc_f32
    from mechinterp_qwen3.transcoder.single_layer_transcoder import JumpReLU
    act_fn = tc.activation_function
    if isinstance(act_fn, JumpReLU):
        thr = act_fn.threshold.detach().float()
        acts_f32 = (pre * (pre > thr)).cpu().numpy()
    else:
        acts_f32 = torch.relu(pre).cpu().numpy()

    # Active features in both
    active_f32  = (acts_f32  > 0).any(axis=0)
    active_bf16 = (acts_bf16 > 0).any(axis=0)
    n_active_f32 = active_f32.sum()

    # bfloat16 must agree with float32 on ≥99% of active features
    agreement = (active_f32 & active_bf16).sum() / max(n_active_f32, 1)
    assert agreement >= 0.99, (
        f"bfloat16 activates only {agreement:.1%} of float32-active features at layer {layer}"
    )


# ---------------------------------------------------------------------------
# 4. float16 transcoder risk — documents known failure mode, not a pass/fail
# ---------------------------------------------------------------------------

@_REQUIRES_RDS
def test_float16_transcoder_warns(model_bf16):
    """Document that float16 can silently change which transcoder features activate.

    This test PASSES if float16 matches float32 (good) or if it differs (just
    prints a warning). It is a canary, not a hard assertion — the important
    thing is knowing the magnitude of the drift before trusting float16 outputs.
    """
    from experiments.concept_localization.analyze import collect_layer_residuals
    from mechinterp_qwen3.transcoder.single_layer_transcoder import JumpReLU

    prompts_and_anchors = _make_prompts(model_bf16)
    layer = 10
    H = collect_layer_residuals(model_bf16, prompts_and_anchors, [layer])
    H_l = H[layer].astype(np.float32)

    tc = model_bf16.transcoders[layer]
    dev = tc.W_enc.device
    results = {}
    for dtype_name, dtype in [("float32", torch.float32), ("float16", torch.float16)]:
        W = tc.W_enc.detach().to(dtype)
        b = tc.b_enc.detach().to(dtype)
        H_t = torch.from_numpy(H_l).to(device=dev, dtype=dtype)
        pre = H_t @ W.T + b
        act_fn = tc.activation_function
        if isinstance(act_fn, JumpReLU):
            thr = act_fn.threshold.detach().to(dtype)
            acts = (pre * (pre > thr)).float().cpu().numpy()
        else:
            acts = torch.relu(pre).float().cpu().numpy()
        results[dtype_name] = (acts > 0).any(axis=0)

    n_f32 = results["float32"].sum()
    n_agree = (results["float32"] & results["float16"]).sum()
    agreement = n_agree / max(n_f32, 1)
    print(f"\nfloat16 vs float32 at layer {layer}: {agreement:.1%} sparsity agreement "
          f"({n_f32} active in f32, {n_agree} also active in f16)")

    if agreement < 0.99:
        pytest.warns(UserWarning,
            match="float16 changes transcoder sparsity pattern")
    # Always passes — this is documentation, not a hard requirement.


# ---------------------------------------------------------------------------
# 5. All anchors extracted in one pass == separate passes
# ---------------------------------------------------------------------------

@_REQUIRES_RDS
def test_all_anchors_single_pass(model_bf16):
    """Extracting multiple anchor positions in one forward pass must match
    running the model separately for each anchor position."""
    from experiments.concept_localization.analyze import collect_layer_residuals

    # Two prompts, two different anchor positions
    prompts = _make_prompts(model_bf16)[:2]  # use first 2 prompts
    layers = [10, 35]

    # Reference: run separately for each anchor position
    anchor_a = prompts[0][1]      # last token of first prompt
    anchor_b = max(0, anchor_a - 2)  # two positions earlier (same prompt length)

    H_anchor_a = collect_layer_residuals(model_bf16,
        [(ids, anchor_a) for ids, _ in prompts], layers)
    H_anchor_b = collect_layer_residuals(model_bf16,
        [(ids, anchor_b) for ids, _ in prompts], layers)

    # Optimized: extract both anchors in one pass
    # collect_layer_residuals_multi_anchor to be implemented
    from experiments.concept_localization.analyze import collect_layer_residuals_multi_anchor
    H_multi = collect_layer_residuals_multi_anchor(
        model_bf16, [(ids, [anchor_a, anchor_b]) for ids, _ in prompts], layers
    )
    # H_multi[l] shape: (n_prompts, n_anchors, d_model)

    for l in layers:
        np.testing.assert_allclose(
            H_multi[l][:, 0, :], H_anchor_a[l], rtol=1e-2, atol=1e-3,
            err_msg=f"Layer {l} anchor_a: multi-anchor vs single-anchor differ"
        )
        np.testing.assert_allclose(
            H_multi[l][:, 1, :], H_anchor_b[l], rtol=1e-2, atol=1e-3,
            err_msg=f"Layer {l} anchor_b: multi-anchor vs single-anchor differ"
        )
