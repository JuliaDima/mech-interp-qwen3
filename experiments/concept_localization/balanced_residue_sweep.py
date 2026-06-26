"""
Balanced mean-activation sweep for the residue_class concept.

Generates n_per_class single prompts for each residue class 0..m-1 (default m=7),
runs them through the model, applies the transcoder at every layer, and saves the
per-class mean feature activation as an npz file.

Output keys in the npz:
  mean_L{i}   float32 (m, d_tc)  — mean transcoder activation per residue class, layer i
  labels      int32   (N,)        — residue class for each prompt (0..m-1)
  prompts     object  (N,)        — the prompt strings
  n_per_class int                 — N // m
  modulus     int                 — m

Usage
-----
    sbatch scripts/sbatch_run.sh \\
        python -m experiments.concept_localization.balanced_residue_sweep \\
        --anchor_pos 3 --n_per_class 30

    # larger balanced set:
    sbatch scripts/sbatch_run.sh \\
        python -m experiments.concept_localization.balanced_residue_sweep \\
        --anchor_pos 3 --n_per_class 100
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
    sys.path.insert(0, str(_REPO))

_MODULUS  = 7
_TEMPLATE = "calc: {a}%7= "
_OUT_DIR  = _REPO / "runs/concept_localization/residue_class/balanced_mean_acts"


def _generate_prompts(n_per_class: int, modulus: int, seed: int) -> tuple[list[str], list[int]]:
    """n_per_class 3-digit prompts per residue class; returns (prompts, labels)."""
    rng = random.Random(seed)
    prompts, labels = [], []
    for r in range(modulus):
        k_min = max(1, (100 - r + modulus - 1) // modulus)
        k_max = (999 - r) // modulus
        pool  = list(range(k_min, k_max + 1))
        rng.shuffle(pool)
        count = 0
        for k in pool:
            if count >= n_per_class:
                break
            a = modulus * k + r
            if 100 <= a <= 999:
                prompts.append(_TEMPLATE.format(a=a))
                labels.append(r)
                count += 1
        if count < n_per_class:
            raise RuntimeError(f"Only {count}/{n_per_class} 3-digit numbers for r={r} mod {modulus}")
    return prompts, labels


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--anchor_pos",  type=int, default=3,
                    help="Token position in the prompt to capture (without sink offset)")
    ap.add_argument("--n_per_class", type=int, default=30,
                    help="Number of prompts per residue class")
    ap.add_argument("--modulus",     type=int, default=_MODULUS)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--out_dir",     type=Path, default=_OUT_DIR)
    ap.add_argument("--model",           default=None)
    ap.add_argument("--transcoder_set",  default=None)
    ap.add_argument("--dtype",           default="bfloat16")
    ap.add_argument("--device",          default=None)
    ap.add_argument("--batch_size",  type=int, default=32)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── load model ────────────────────────────────────────────────────────────
    from scripts.model_config import default_model, default_transcoder_set
    from mechinterp_qwen3.attribution_model import AttributionModel
    from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
    from mechinterp_qwen3.utils.model_utils import parse_dtype

    dtype    = parse_dtype(args.dtype)
    model_id = args.model or default_model()
    tc_id    = args.transcoder_set or default_transcoder_set()

    print(f"Loading model {model_id} + transcoders {tc_id}...")
    tc_set, _ = load_transcoder_from_hub(tc_id, dtype=dtype, lazy_encoder=True, lazy_decoder=True)
    model = AttributionModel.from_pretrained_and_transcoders(
        model_id, tc_set, dtype=dtype, device=device
    )
    model.eval()
    n_layers = len(model.transcoders)
    print(f"  {n_layers} layers, device={device}")

    # ── generate prompts ──────────────────────────────────────────────────────
    prompts, labels = _generate_prompts(args.n_per_class, args.modulus, args.seed)
    N = len(prompts)
    print(f"Generated {N} prompts ({args.n_per_class} per class × {args.modulus} classes)")

    # tokenize
    inputs: list[tuple[list[int], int]] = []
    for p in prompts:
        ids = model.tokenizer(p, add_special_tokens=False).input_ids
        inputs.append((ids, args.anchor_pos))

    # ── capture residual stream ───────────────────────────────────────────────
    from experiments.concept_localization.analyze import collect_layer_residuals_batched
    from experiments.concept_localization.sweep_utils import apply_transcoder_all

    all_layers = list(range(n_layers))
    print(f"Capturing residual stream at anchor pos {args.anchor_pos} across {n_layers} layers...")
    H = collect_layer_residuals_batched(
        model, inputs, all_layers, batch_size=args.batch_size
    )  # {layer: (N, d_model) float32}

    # ── compute per-class mean transcoder activations ─────────────────────────
    labels_arr = np.array(labels, dtype=np.int32)
    modulus    = args.modulus
    out: dict[str, np.ndarray] = {
        "labels":      labels_arr,
        "prompts":     np.array(prompts, dtype=object),
        "n_per_class": np.array(args.n_per_class),
        "modulus":     np.array(modulus),
    }

    print(f"Computing mean transcoder activations per class...")
    with torch.no_grad():
        for layer in all_layers:
            if layer not in H:
                print(f"  WARN: layer {layer} missing from residuals; skipping")
                continue
            acts = apply_transcoder_all(model, layer, H[layer])  # (N, d_tc)
            # mean per residue class
            d_tc = acts.shape[1]
            mean_acts = np.zeros((modulus, d_tc), dtype=np.float32)
            for r in range(modulus):
                mask = labels_arr == r
                if mask.any():
                    mean_acts[r] = acts[mask].mean(axis=0)
            out[f"mean_L{layer}"] = mean_acts
            print(f"  L{layer}: d_tc={d_tc}, frac_active={float((acts > 0).any(axis=0).mean()):.3f}")

    # ── save ──────────────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"mean_acts_anchor{args.anchor_pos}_n{args.n_per_class}_seed{args.seed}.npz"
    np.savez_compressed(out_path, **out)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved → {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
