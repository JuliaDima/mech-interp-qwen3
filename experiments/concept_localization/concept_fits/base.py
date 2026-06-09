"""Shared scaffolding for concept-specific PySR anchor runs.

Each concept script defines:
  - _VAR_NAMES:  list[str]           — variable names passed to PySR
  - build_Xy_fn: (npz, examples, key, orig_indices) -> (X, y) | (None, None)
  - plot_fn:     (X, y, model, key, out_path, r2_threshold) -> r2

Then calls base.run_anchor(...) and base.make_parser(...).

If sweep/edec_pysr/sweep_activations.npz is missing, run_anchor automatically
generates it by loading the model, scanning active features, projecting onto
E_dec, and capturing residuals — so no separate extraction step is needed.
"""
from __future__ import annotations

import importlib
import json
import pickle
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SWEEPS_DIR = _REPO_ROOT / "scripts" / "sweeps"
for p in (_REPO_ROOT, _REPO_ROOT / "src", _SWEEPS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fit_pysr_sweep import _fit_pysr
from sweep_utils import apply_transcoder_all

from experiments.concept_localization.analyze import (
    collect_layer_residuals,
    project_onto_E_dec_model,
)
from experiments.concept_localization.extract_deltas_generic import _resolve_anchor
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype

_MODEL = "Qwen/Qwen3-4B"
_TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"


def _load_concept(name: str, n: int, seed: int):
    mod = importlib.import_module(f"data.concept_datasets.{name}_dataset")
    for fn in [
        f"generate_{name}_pairs",
        f"generate_{name.split('_')[-1]}_pairs",
        "generate_decimal_pairs",
        "generate_large_prime_pairs",
        "generate_wave_pairs",
    ]:
        if hasattr(mod, fn):
            return getattr(mod, fn)(n, seed=seed)
    for attr in dir(mod):
        if attr.startswith("generate_") and attr.endswith("_pairs"):
            return getattr(mod, attr)(n, seed=seed)
    raise ValueError(f"No generate function found in {name}_dataset")


def _resolve_top_edec(
    anchor_dir: Path,
    model,
    top_k: int,
    active_features: dict[int, set],
) -> tuple[list[tuple[int, int]], list[float]]:
    raw = torch.load(str(anchor_dir / "deltas.pt"), map_location="cpu")
    per_layer = project_onto_E_dec_model(model, raw["all"], top_k=1000)
    flat = []
    for ms in per_layer.values():
        for m in ms:
            if m.feature_id not in active_features.get(m.layer, set()):
                continue
            flat.append((abs(m.cos_sim), m.layer, m.feature_id, m.cos_sim))
    flat.sort(reverse=True)
    top = flat[:top_k]
    return [(l, f) for _, l, f, _ in top], [c for _, _, _, c in top]


def _build_inputs_and_examples(model, pairs, anchor_mode, max_pairs):
    inputs, examples = [], []
    for pair in pairs[:max_pairs]:
        ids_pos = model.tokenizer(pair.prompt_pos, add_special_tokens=False).input_ids
        ids_neg = model.tokenizer(pair.prompt_neg, add_special_tokens=False).input_ids
        if len(ids_pos) != len(ids_neg):
            continue
        anchor = _resolve_anchor(ids_pos, model.tokenizer, anchor_mode, None, None)
        inputs.append((ids_pos, anchor))
        inputs.append((ids_neg, anchor))
        examples.append({
            "pair_idx": len(examples),
            "template": pair.template,
            "meta": pair.meta,
            "label_pos": pair.label_pos,
        })
    return inputs, examples


def _ensure_edec_data(
    anchor_dir: Path,
    concept: str,
    n_pairs: int = 200,
    top_k: int = 15,
    seed: int = 42,
    dtype_str: str = "bfloat16",
) -> bool:
    """Generate sweep/edec_pysr/sweep_activations.npz if it doesn't exist.

    Returns True if data is available after the call, False if skipped.
    """
    edec_dir = anchor_dir / "sweep" / "edec_pysr"
    acts_path = edec_dir / "sweep_activations.npz"
    ex_path   = edec_dir / "sweep_examples.pkl"

    if acts_path.exists() and ex_path.exists():
        return True

    print(f"  [edec] {anchor_dir.name}: generating edec_pysr data…")

    cfg_path = anchor_dir / "results.json"
    if not cfg_path.exists():
        print(f"  [skip] {anchor_dir.name}: no results.json — cannot determine anchor_mode")
        return False

    cfg = json.loads(cfg_path.read_text())["config"]
    anchor_mode = str(cfg.get("anchor_mode", cfg.get("anchor_pos", "9")))

    device = get_default_device()
    dtype  = parse_dtype(dtype_str)

    tc_set, _ = load_transcoder_from_hub(_TRANSCODER_SET, dtype=dtype,
                                         lazy_encoder=True, lazy_decoder=True)
    model = AttributionModel.from_pretrained_and_transcoders(
        _MODEL, tc_set, dtype=dtype, device=device)
    model.eval()

    pairs = _load_concept(concept, n_pairs, seed)
    inputs, examples = _build_inputs_and_examples(model, pairs, anchor_mode, n_pairs)
    if not inputs:
        print(f"  [skip] {anchor_dir.name}: no valid pairs for {concept} at anchor {anchor_mode}")
        return False

    all_layers = list(range(len(model.transcoders)))
    print(f"  [edec] scanning active features across {len(all_layers)} layers…")
    H_scan = collect_layer_residuals(model, inputs[:40], all_layers)

    active_features: dict[int, set] = {}
    for layer in all_layers:
        acts = apply_transcoder_all(model, layer, H_scan[layer])
        active_features[layer] = set(np.where(acts.max(axis=0) > 0)[0].tolist())

    features, _cos_sims = _resolve_top_edec(anchor_dir, model, top_k, active_features)
    if not features:
        print(f"  [skip] {anchor_dir.name}: no top E_dec features found")
        return False

    layers = sorted({l for l, _ in features})
    print(f"  [edec] top-{top_k} features: {[f'L{l}_F{f}' for l, f in features]}")
    H_all = collect_layer_residuals(model, inputs, layers)

    acts_1d: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for layer in layers:
            acts = apply_transcoder_all(model, layer, H_all[layer])
            for l, fid in features:
                if l == layer:
                    acts_1d[f"L{l}_F{fid}"] = acts[:, fid].astype(np.float32)

    edec_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(acts_path, **acts_1d)
    with open(ex_path, "wb") as f:
        pickle.dump(examples, f)
    # Save feature metadata so overview plots can be regenerated without the model
    features_meta = [{"key": f"L{l}_F{fid}", "layer": l, "feature_id": fid, "cos_sim": float(cs)}
                     for (l, fid), cs in zip(features, _cos_sims)]
    with open(edec_dir / "edec_features.json", "w") as f:
        json.dump(features_meta, f, indent=2)
    print(f"  [edec] saved → {edec_dir}")

    # Free model memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True


def run_anchor(
    anchor_dir: Path,
    concept: str,
    var_names: list[str],
    build_Xy_fn: Callable,
    plot_fn: Callable,
    top_k: int = 15,
    niterations: int = 40,
    r2_threshold: float = 0.0,
    seed: int = 42,
    template: str = "T0",
    n_pairs: int = 200,
    dtype: str = "bfloat16",
    features: list[str] | None = None,
) -> list[dict]:
    """Load (or generate) edec_pysr sweep data, rank features, fit PySR, plot, save summary.

    If sweep/edec_pysr/sweep_activations.npz is missing, automatically runs the
    model forward passes to extract E_dec-projected feature activations first.
    """
    sweep_dir = anchor_dir / "sweep" / "edec_pysr"
    acts_path = sweep_dir / "sweep_activations.npz"
    ex_path   = sweep_dir / "sweep_examples.pkl"

    if not acts_path.exists() or not ex_path.exists():
        ok = _ensure_edec_data(anchor_dir, concept,
                               n_pairs=n_pairs, top_k=top_k, seed=seed, dtype_str=dtype)
        if not ok:
            return []

    npz      = np.load(acts_path)
    examples = pickle.load(open(ex_path, "rb"))

    # Filter by template, tracking original row indices for interleaved pos/neg activations.
    if template:
        filtered = [(i, ex) for i, ex in enumerate(examples)
                    if ex.get("template") == template]
        if not filtered:
            print(f"  [skip] {anchor_dir.name}: no examples for template={template}")
            return []
        orig_indices, examples = zip(*filtered)
        orig_indices = list(orig_indices)
        examples     = list(examples)
    else:
        orig_indices = list(range(len(examples)))

    out_dir = anchor_dir / "sweep" / "edec_pysr" / f"{concept}_pysr"
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = [k for k in npz.files if k.startswith("L")]
    if not keys:
        print(f"  [skip] {anchor_dir.name}: no feature arrays in npz")
        return []

    def _pos_neg_delta(k: str) -> float:
        a = np.asarray(npz[k])
        pos_vals = [a[2 * i]     for i in orig_indices if 2 * i     < len(a)]
        neg_vals = [a[2 * i + 1] for i in orig_indices if 2 * i + 1 < len(a)]
        if not pos_vals or not neg_vals:
            return 0.0
        return float(np.mean(pos_vals) - np.mean(neg_vals))

    if features:
        keys_sorted = [k for k in features if k in npz.files]
    else:
        keys_sorted = sorted(keys, key=lambda k: abs(_pos_neg_delta(k)), reverse=True)[:top_k]
    print(f"  {anchor_dir.name}: fitting {len(keys_sorted)} features (template={template or 'all'})")

    results = []
    for key in keys_sorted:
        X, y = build_Xy_fn(npz, examples, key, orig_indices)
        if X is None:
            continue
        print(f"  Fitting {key}  ({len(y)} pts)  y=[{y.min():.3g}, {y.max():.3g}]")
        model   = _fit_pysr(X, y, var_names, niterations=niterations, seed=seed)
        best_eq = str(model.get_best()["equation"])
        print(f"    best: {best_eq}")

        r2 = plot_fn(X, y, model, key, out_dir / f"pysr_{key}.pdf", r2_threshold)

        results.append({
            "feature":      key,
            "anchor":       anchor_dir.name,
            "variables":    var_names,
            "best_equation": best_eq,
            "r2":           round(float(r2), 4) if not np.isnan(r2) else None,
            "pos_neg_delta": round(_pos_neg_delta(key), 4),
        })

    summary_path = out_dir / f"pysr_{concept}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Summary → {summary_path}")
    return results


def make_parser(concept: str, anchors_root: Path):
    """Return a standard ArgumentParser for concept PySR scripts."""
    import argparse
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--anchor_dir",  type=Path,
                       help="Single anchor dir (must contain sweep/edec_pysr/)")
    group.add_argument("--all_anchors", action="store_true",
                       help=f"Run all anchors under {anchors_root}")
    ap.add_argument("--top_k",        type=int,   default=15)
    ap.add_argument("--features",     nargs="+",  default=None,
                    help="Run PySR only on these feature keys, e.g. L30_F103271")
    ap.add_argument("--n_pairs",      type=int,   default=200,
                    help="Pairs to load for edec extraction (if data not yet cached)")
    ap.add_argument("--niterations",  type=int,   default=40)
    ap.add_argument("--r2_threshold", type=float, default=0.0)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--dtype",        type=str,   default="bfloat16",
                    help="Model dtype for edec extraction")
    ap.add_argument("--template",     type=str,   default="T0",
                    help="Filter to this template (T0/T1/T2). Empty string = all.")
    return ap


def resolve_anchor_dirs(args, anchors_root: Path) -> list[Path]:
    if args.all_anchors:
        dirs = sorted(p for p in anchors_root.iterdir()
                      if p.is_dir() and p.name.startswith("anchor_"))
        if not dirs:
            print(f"No anchor dirs found under {anchors_root}")
        return dirs
    return [args.anchor_dir]
