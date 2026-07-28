"""Tests for 1D bar plot generation in delta_feature_pipeline.py.

Test 1 (unit): _bin_to_1d_bar with synthetic data — no model needed.
Test 2 (integration): load H_L10 from saved sweep + transcoder, compute
  feature 158993 activations, verify per-residue means match edec_features.json.
  Requires RDS cache access (run on cluster or login node with RDS mounted).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from experiments.concept_localization.pipeline.delta_feature_pipeline import _bin_to_1d_bar

_ANCHOR_DIR = (
    _REPO_ROOT
    / "runs/concept_localization/residue_class/residue_class_T0/anchor_rank4_pos6/sweep"
)
_REQUIRES_RDS = pytest.mark.skipif(
    not Path("/rds/user/eid23/hpc-work/p28/cache/hf/hub/models--mwhanna--qwen3-4b-transcoders").exists(),
    reason="RDS cache not mounted",
)
_REQUIRES_SWEEP = pytest.mark.skipif(
    not (_ANCHOR_DIR / "sweep_residuals.npz").exists(),
    reason="Sweep residuals not found",
)


# --- Unit test: _bin_to_1d_bar ---

def _make_examples(residues_pos, residues_neg, m=7):
    """Build minimal example dicts for _bin_to_1d_bar."""
    return [
        {"meta": {"a_pos": rp, "a_neg": rn, "m": m}}
        for rp, rn in zip(residues_pos, residues_neg)
    ]


def test_bin_to_1d_bar_basic():
    """Single example per residue, activation = residue value."""
    m = 7
    residues = list(range(m))
    examples = _make_examples(residues, residues, m)
    # acts_col interleaves pos/neg: [pos0, neg0, pos1, neg1, ...]
    acts_col = np.array([float(r) for r in residues for _ in (0, 1)], dtype=np.float32)
    pos, neg = _bin_to_1d_bar(acts_col, examples, m)
    np.testing.assert_allclose(pos, np.arange(m, dtype=np.float32))
    np.testing.assert_allclose(neg, np.arange(m, dtype=np.float32))


def test_bin_to_1d_bar_mean():
    """Two examples with same residue — result should be their mean."""
    m = 7
    examples = _make_examples([1, 1], [2, 2], m)
    acts_col = np.array([3.0, 0.0, 5.0, 0.0], dtype=np.float32)  # pos: 3,5; neg ignored
    pos, neg = _bin_to_1d_bar(acts_col, examples, m)
    assert abs(pos[1] - 4.0) < 1e-5  # mean(3,5) = 4
    assert pos[0] == 0.0  # no examples at residue 0
    assert neg[2] == 0.0  # neg acts are 0 (second element of each pair)


def test_bin_to_1d_bar_residue_class_structure():
    """residue_class: all pos have a≡1, so blue bar only at x=1."""
    m = 7
    # 3 pos examples all ≡1, neg examples at 2,3,4
    examples = _make_examples([8, 15, 22], [2, 3, 4], m)  # 8%7=1, 15%7=1, 22%7=1
    acts_col = np.array([10., 5., 10., 5., 10., 5.], dtype=np.float32)
    pos, neg = _bin_to_1d_bar(acts_col, examples, m)
    # pos: only bucket 1 should be filled
    assert pos[1] == pytest.approx(10.0)
    for r in [0, 2, 3, 4, 5, 6]:
        assert pos[r] == 0.0
    # neg: buckets 2,3,4 filled
    assert neg[2] == pytest.approx(5.0)
    assert neg[3] == pytest.approx(5.0)
    assert neg[4] == pytest.approx(5.0)
    assert neg[1] == 0.0


# --- Integration test: compare computed activations to stored JSON ---

def _load_model_and_residuals():
    """Shared setup for integration tests — load model + sweep residuals."""
    import torch
    from mechinterp_qwen3.attribution_model import AttributionModel
    from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
    from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype

    dtype = parse_dtype("bfloat16")
    device = get_default_device()
    tc_set, _ = load_transcoder_from_hub(
        "mwhanna/qwen3-4b-transcoders", dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        "Qwen/Qwen3-4B", tc_set, dtype=dtype, device=device
    )
    model.eval()

    data = np.load(str(_ANCHOR_DIR / "sweep_residuals.npz"), allow_pickle=True)
    with open(_ANCHOR_DIR / "sweep_dataset_examples.pkl", "rb") as f:
        examples = pickle.load(f)
    return model, data, examples


@_REQUIRES_SWEEP
@_REQUIRES_RDS
def test_top7_feature_activations_match_json():
    """Compute activations for all top-7 pos features, verify mean_pos/mean_neg vs JSON.

    Loads the transcoder for each unique layer once, then checks:
      - |computed mean_pos - json mean_pos| < 0.1
      - |computed mean_neg - json mean_neg| < 0.1
      - computed std_pos within 20% of json std_pos (or both ~0)
    """
    from experiments.concept_localization.sweep_utils import apply_transcoder_all

    model, data, examples = _load_model_and_residuals()

    with open(_ANCHOR_DIR / "delta_feature_projections_enc_dec/edec_features.json") as f:
        d = json.load(f)
    all_rows = d["pos"] + d["neg"]

    # Cache activations per layer to avoid recomputing
    layer_acts: dict[int, np.ndarray] = {}

    print(f"\n{'Feature':<20} {'mean_pos_json':>14} {'mean_pos_comp':>14} {'mean_neg_json':>14} {'mean_neg_comp':>14}")
    for row in all_rows:
        layer, fid = row["layer"], row["feature_id"]
        if layer not in layer_acts:
            H_l = data[f"H_L{layer}"].astype(np.float32)
            layer_acts[layer] = apply_transcoder_all(model, layer, H_l)

        acts_col = layer_acts[layer][:, fid]
        mean_pos_comp = float(acts_col[0::2].mean())
        mean_neg_comp = float(acts_col[1::2].mean())
        std_pos_comp  = float(acts_col[0::2].std())

        print(f"{row['feature']:<20} {row['mean_pos']:>14.4f} {mean_pos_comp:>14.4f} "
              f"{row['mean_neg']:>14.4f} {mean_neg_comp:>14.4f}")

        assert abs(mean_pos_comp - row["mean_pos"]) < 0.1, (
            f"{row['feature']}: mean_pos computed={mean_pos_comp:.4f} json={row['mean_pos']:.4f}"
        )
        assert abs(mean_neg_comp - row["mean_neg"]) < 0.1, (
            f"{row['feature']}: mean_neg computed={mean_neg_comp:.4f} json={row['mean_neg']:.4f}"
        )
        # std check only when signal is non-trivial
        if row["std_pos"] > 0.01:
            assert abs(std_pos_comp - row["std_pos"]) / row["std_pos"] < 0.20, (
                f"{row['feature']}: std_pos computed={std_pos_comp:.4f} json={row['std_pos']:.4f}"
            )


@_REQUIRES_SWEEP
@_REQUIRES_RDS
def test_feature_activation_bar_structure():
    """For residue_class, verify bar plot structure: pos only at x=1, neg absent at x=0,1."""
    from experiments.concept_localization.sweep_utils import apply_transcoder_all

    model, data, examples = _load_model_and_residuals()

    # Use L10_F158993 (all pos have a≡1 mod 7 by construction)
    LAYER, FEAT_ID = 10, 158993
    H_l = data[f"H_L{LAYER}"].astype(np.float32)
    acts = apply_transcoder_all(model, LAYER, H_l)
    acts_col = acts[:, FEAT_ID]

    pos_bar, neg_bar = _bin_to_1d_bar(acts_col, examples, modulus=7)

    # pos only at residue 1
    assert pos_bar[1] > 0, "pos bar at residue 1 should be nonzero"
    for r in [0, 2, 3, 4, 5, 6]:
        assert pos_bar[r] == 0.0, f"pos_bar[{r}] should be 0 (no pos examples at residue {r})"

    # neg absent at residues 0 and 1
    assert neg_bar[0] == 0.0, "no neg examples at residue 0"
    assert neg_bar[1] == 0.0, "neg examples should not have a≡1 mod 7"

    # pos bar height matches computed mean_pos
    mean_pos = float(acts_col[0::2].mean())
    assert abs(pos_bar[1] - mean_pos) < 0.01


# --- Survival stats test (requires attribution graphs to have been run) ---

_SURVIVAL_FILE = _REPO_ROOT / "runs/concept_localization/residue_class/feature_survival/survival_stats.json"
_REQUIRES_SURVIVAL = pytest.mark.skipif(
    not _SURVIVAL_FILE.exists(),
    reason="residue_class survival_stats.json not generated yet — run attribution_feature_survival.py first",
)


@_REQUIRES_SURVIVAL
def test_top7_features_survival_rates():
    """Report survival rate and pos_enrichment for the top-7 features in the attr graph."""
    with open(_ANCHOR_DIR / "delta_feature_projections_enc_dec/edec_features.json") as f:
        d = json.load(f)
    with open(_SURVIVAL_FILE) as f:
        surv = json.load(f)

    surv_by_key = {f["feature_key"]: f for f in surv["features"]}
    n_total = surv["config"]["n_total_graphs"]

    all_rows = d["pos"] + d["neg"]
    print(f"\n{'Feature':<20} {'dec_cos':>8} {'survival':>10} {'pos_enrich':>11} {'in_graph?':>10}")
    for row in all_rows:
        key = row["feature"]
        s = surv_by_key.get(key)
        survival = s["survival_rate"] if s else 0.0
        enrichment = s.get("pos_enrichment", float("nan")) if s else float("nan")
        in_graph = "YES" if s else "no"
        print(f"{key:<20} {row['dec_cos']:>8.3f} {survival:>10.3f} {enrichment:>11.3f} {in_graph:>10}")

    # At least one of the top-7 should appear in some graph
    appearing = [r for r in all_rows if r["feature"] in surv_by_key]
    assert len(appearing) > 0, "None of the top-7 delta-aligned features appear in any attr graph"
