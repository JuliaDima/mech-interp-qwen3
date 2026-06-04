"""PySR on the top-k transcoder features most aligned with the concept delta.

Selects features by |cos(δ_l, e^dec_f)| across all layers, computed from the
loaded model's transcoders and cached to edec_features.npz
in the anchor dir for reuse. Generates their activations over the concept
dataset at the anchor used for the run, then runs PySR via fit_pysr_sweep.fit_feature.

Carry uses the 10×10 ones-digit grid; every other concept uses generic mode,
fitting activations against the numeric meta variables. Only fits whose R²
exceeds --r2_threshold are plotted.

Usage
-----
    python scripts/sweeps/pysr_top_edec_features.py \\
        --anchor_dir runs/concept_localization/carry/anchor_rank5_pos9 \\
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
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


def _resolve_top_edec(anchor_dir: Path, model, top_k: int) -> list[tuple[int, int]]:
    """Global top-k (layer, feat_id) by |cos_sim| from the E_dec projection.

    Reads a cached edec_features.npz if present; otherwise computes it from the
    loaded model's transcoders (no disk-cache path, no extra model run) and
    persists it for reuse.
    """
    npz_path = anchor_dir / "edec_features.npz"
    if npz_path.exists():
        d = np.load(npz_path)
        return list(zip(d["layers"].tolist(), d["feat_ids"].tolist()))

    raw = torch.load(str(anchor_dir / "deltas.pt"), map_location="cpu")
    per_layer = project_onto_E_dec_model(model, raw["all"], top_k=top_k)
    flat = [(abs(m.cos_sim), m.layer, m.feature_id, m.cos_sim)
            for ms in per_layer.values() for m in ms]
    flat.sort(reverse=True)
    top = flat[:top_k]
    np.savez(
        npz_path,
        layers=np.array([l for _, l, _, _ in top], dtype=np.int32),
        feat_ids=np.array([f for _, _, f, _ in top], dtype=np.int64),
        cos_sims=np.array([c for _, _, _, c in top], dtype=np.float32),
    )
    print(f"Saved top-{top_k} E_dec features → {npz_path}")
    return [(l, f) for _, l, f, _ in top]


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
        ax.set_xticks(range(0, 10, 3)); ax.set_yticks(range(0, 10, 3))
        ax.tick_params(length=0, labelsize=6)
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
    ap.add_argument("--r2_threshold", type=float, default=0.5)
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

    # Projection uses the loaded model's transcoders (cached to npz for reuse)
    features = _resolve_top_edec(anchor_dir, model, args.top_k)
    layers = sorted({l for l, _ in features})
    print(f"Top-{args.top_k} E_dec features: {[f'L{l}_F{f}' for l, f in features]}")
    print(f"Concept: {args.concept}  anchor mode: {anchor_mode}  layers: {layers}")

    pairs = _load_concept(args.concept, args.n_pairs, args.seed)
    inputs, examples = _build_inputs_and_examples(model, pairs, anchor_mode, args.n_pairs)
    if not inputs:
        print(f"  [skip] no valid pairs for {args.concept} at anchor {anchor_mode}")
        return
    H = collect_layer_residuals(model, inputs, layers)

    acts_1d: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for layer in layers:
            acts = apply_transcoder_all(model, layer, H[layer])
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
        cos_sims = np.load(anchor_dir / "edec_features.npz")["cos_sims"].tolist()
        _plot_topk_grid(features, cos_sims, npz, examples,
                        out_dir / "edec_topk_grid.pdf")

    results = []
    for l, fid in features:
        r = fit_feature(f"L{l}_F{fid}", npz, examples, mode, out_dir,
                        niterations=args.niterations, r2_threshold=args.r2_threshold)
        if r:
            results.append(r)

    with (out_dir / "pysr_edec_summary.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(f"PySR E_dec summary → {out_dir / 'pysr_edec_summary.json'}")


if __name__ == "__main__":
    main()
