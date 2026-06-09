"""PySR on top-k transcoder features for MODULAR concepts.

For concepts with modular or grid structure: carry, gcd, residue_class.

  - carry       uses the 10×10 ones-digit grid mode (a mod 10, b mod 10)
  - gcd / residue_class  use generic-meta mode with numeric fields (a, b, g/m)

For concepts whose task is better described by relational or inequality
structure (transitive_ordering, triangle_inequality, …), use the
concept-specific scripts in this directory instead — they add derived
variables tailored to each concept's semantics.

Selects features by |cos(δ_l, e^dec_f)| across all layers, computed from the
loaded model's transcoders and cached to edec_features.npz in the anchor dir
for reuse. Generates their activations over the concept dataset at the anchor
used for the run, then runs PySR via fit_pysr_sweep.fit_feature.

Only fits whose R² exceeds --r2_threshold are plotted.

Usage
-----
    python experiments/concept_localization/concept_fits/pysr_modular.py \\
        --anchor_dir runs/concept_localization/carry/carry_T0/anchor_rank5_pos9 \\
        --concept carry --top_k 15 --r2_threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "scripts" / "sweeps"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sweep_utils import apply_transcoder_all
from fit_pysr_sweep import fit_feature, _meta_mode, _carry_table
from run_concept_sweep import _load_concept
import experiments.plot_style as ps
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


def _resolve_top_edec(anchor_dir: Path, model, top_k: int, active_features: dict[int, set[int]]) -> tuple[list[tuple[int, int]], list[float]]:
    """Global top-k (layer, feat_id) by |cos_sim| from the E_dec projection.

    Restricted to features that fired on at least one scan example (active_features),
    then ranked purely by |cos_sim| with the concept delta.
    """
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
    """Build (token_ids, anchor) inputs and aligned example metadata in lockstep.

    Only pairs whose pos/neg tokenisations share a length are kept, so that
    acts row 2*i / 2*i+1 always corresponds to examples[i].
    """
    inputs, examples = [], []
    for pair in pairs[:max_pairs]:
        ids_pos = model.tokenizer(pair.prompt_pos, add_special_tokens=False).input_ids
        ids_neg = model.tokenizer(pair.prompt_neg, add_special_tokens=False).input_ids
        if len(ids_pos) != len(ids_neg):
            continue
        anchor = _resolve_anchor(ids_pos, model.tokenizer, anchor_mode, None, None)
        inputs.append((ids_pos, anchor))
        inputs.append((ids_neg, anchor))
        examples.append({"pair_idx": len(examples), "template": pair.template,
                         "meta": pair.meta, "label_pos": pair.label_pos})
    return inputs, examples


def _plot_topk_grid(features, cos_sims, npz, examples, out_path, ncols=5):
    """Grid of the top-k features' actual activation heatmaps (carry mode only).

    Shows every top-k feature regardless of fit quality, so it complements the
    R²-gated per-feature PySR plots.
    """
    cmap = LinearSegmentedColormap.from_list("white_violet", ["white", ps.VIOLET])
    cmap.set_bad("white")

    items = []
    for (l, fid), cs in zip(features, cos_sims):
        key = f"L{l}_F{fid}"
        if key not in npz:
            continue
        try:
            _, _, grid, _ = _carry_table(key, npz, examples)
        except Exception:
            continue
        items.append((l, fid, cs, grid))
    if not items:
        return

    ps.apply()
    n = len(items)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.4),
                             squeeze=False)
    for idx, (l, fid, cs, grid) in enumerate(items):
        ax = axes[idx // ncols][idx % ncols]
        ax.imshow(grid.T, origin="lower", aspect="equal", cmap=cmap, vmin=0, vmax=1)
        ax.set_title(rf"$L^{{{l}}}_{{{fid}}}$  cs={cs:+.2f}", fontsize=8)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("a", fontsize=7, labelpad=1)
        ax.set_ylabel("b", fontsize=7, labelpad=1)
        ax.tick_params(length=0, labelsize=5.5)
        for sp in ax.spines.values():
            sp.set_color(ps.GRAY)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(r"Top E_dec features — actual activations  (a mod 10 $\times$ b mod 10)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved top-k grid → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--anchor_dir", required=True,
                    help="Anchor run dir containing deltas.pt and results.json")
    ap.add_argument("--concept", default="carry")
    ap.add_argument("--top_k", type=int, default=15)
    ap.add_argument("--n_pairs", type=int, default=200)
    ap.add_argument("--niterations", type=int, default=40)
    ap.add_argument("--r2_threshold", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    anchor_dir = Path(args.anchor_dir)
    out_dir = anchor_dir / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Anchor mode from the run config so activations match the delta capture point
    cfg = json.loads((anchor_dir / "results.json").read_text())["config"]
    anchor_mode = str(cfg.get("anchor_mode", cfg.get("anchor_pos", "9")))

    device, dtype = get_default_device(), parse_dtype(args.dtype)
    tc_set, _ = load_transcoder_from_hub(_TRANSCODER_SET, dtype=dtype,
                                         lazy_encoder=True, lazy_decoder=True)
    model = AttributionModel.from_pretrained_and_transcoders(_MODEL, tc_set,
                                                             dtype=dtype, device=device)
    model.eval()

    pairs = _load_concept(args.concept, args.n_pairs, args.seed)
    inputs, examples = _build_inputs_and_examples(model, pairs, anchor_mode, args.n_pairs)
    if not inputs:
        print(f"  [skip] no valid pairs for {args.concept} at anchor {anchor_mode}")
        return

    # Find features that fire on any of the scan examples, then rank by cos_sim with the concept delta.
    all_layers = list(range(len(model.transcoders)))
    print(f"Scanning active features across layers (using 40 examples): {all_layers}")
    H_scan = collect_layer_residuals(model, inputs[:40], all_layers)

    active_features = {}
    for layer in all_layers:
        acts = apply_transcoder_all(model, layer, H_scan[layer])
        active_ids = np.where(acts.max(axis=0) > 0)[0].tolist()
        active_features[layer] = set(active_ids)
        print(f"  Layer {layer}: {len(active_ids)} active features / {acts.shape[1]}")

    features, cos_sims = _resolve_top_edec(anchor_dir, model, args.top_k, active_features)
    layers = sorted({l for l, _ in features})
    print(f"Top-{args.top_k} active E_dec features: {[f'L{l}_F{f}' for l, f in features]}")
    print(f"Concept: {args.concept}  anchor mode: {anchor_mode}  layers: {layers}")

    # Now capture residuals for all inputs, but only at the layers containing the resolved top-k features
    print(f"Capturing residuals for all examples at layers: {layers}")
    H_all = collect_layer_residuals(model, inputs, layers)

    acts_1d: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for layer in layers:
            acts = apply_transcoder_all(model, layer, H_all[layer])
            for l, fid in features:
                if l == layer:
                    acts_1d[f"L{l}_F{fid}"] = acts[:, fid].astype(np.float32)

    # Persist a minimal sweep cache (so re-plotting needs no model)
    edec_dir = out_dir / "edec_pysr"
    edec_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(edec_dir / "sweep_activations.npz", **acts_1d)
    with open(edec_dir / "sweep_examples.pkl", "wb") as f:
        pickle.dump(examples, f)

    npz = np.load(edec_dir / "sweep_activations.npz")
    mode = _meta_mode(examples)
    if mode == "skip":
        print(f"  [skip] {args.concept} has no numeric meta fields — PySR skipped")
        return

    # Overview grid of all top-k actual activations (carry mode; ungated by R²)
    if mode == "carry":
        _plot_topk_grid(features, cos_sims, npz, examples,
                        edec_dir / "edec_topk_grid.pdf")

    results = []
    for l, fid in features:
        r = fit_feature(f"L{l}_F{fid}", npz, examples, mode, edec_dir,
                        niterations=args.niterations, r2_threshold=args.r2_threshold)
        if r:
            results.append(r)

    with (edec_dir / "pysr_edec_summary.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(f"PySR E_dec summary → {edec_dir / 'pysr_edec_summary.json'}")


if __name__ == "__main__":
    main()
