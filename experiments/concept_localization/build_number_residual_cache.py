"""Run inference once on all 900 three-digit numbers (100-999), capturing the residual
stream at the "ones_a" position (last digit of N) across all 36 layers.

No concept-pair dataset needed: prompt is a fixed minimal prefix "calc: {n}", truncated
right after the last digit. Causal attention means the residual stream at that position
is identical to what it would be in a longer sentence (e.g. "calc: {n}%7= ") — nothing
after that position can affect it — so truncating there is free, just fewer tokens to run.

This gives uniform per-residue coverage for any modulus M in [2, 100]: 900/M samples per
residue class (>= 9 even at M=100), vs. ~200 samples total and highly uneven coverage
from the gcd/residue_class/prime concept sweeps, which were built to vary around specific
moduli rather than to cover all residues of an arbitrary M.

Needs the model loaded — submit via sbatch.

Usage:
    sbatch scripts/sbatch_run.sh python -m experiments.concept_localization.build_number_residual_cache
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.concept_localization.analyze import collect_layer_residuals_batched
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype
from scripts.model_config import default_model, default_transcoder_set

OUT_PATH = Path("runs/concept_localization/mod_m_scan/number_cache/number_residuals.npz")


def main() -> None:
    device = get_default_device()
    dtype = parse_dtype("bfloat16")
    model_id = default_model()
    tc_id = default_transcoder_set()

    print(f"Loading {model_id} + transcoders {tc_id} on {device}...")
    tc_set, _ = load_transcoder_from_hub(tc_id, dtype=dtype, lazy_encoder=True, lazy_decoder=True)
    model = AttributionModel.from_pretrained_and_transcoders(model_id, tc_set, dtype=dtype, device=device)
    model.eval()

    ns = list(range(100, 1000))
    inputs = []
    for n in ns:
        ids = model.tokenizer(f"calc: {n}", add_special_tokens=False).input_ids
        inputs.append((ids, len(ids) - 1))  # anchor = last token = ones digit of n

    layers = list(range(model.cfg.n_layers))
    print(f"Running {len(ns)} prompts x {len(layers)} layers...")
    with torch.no_grad():
        H = collect_layer_residuals_batched(model, inputs, layers, batch_size=256)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        n=np.array(ns, dtype=np.int64),
        **{f"H_L{l}": H[l] for l in layers},
    )
    print(f"Saved -> {OUT_PATH}  ({len(ns)} numbers x {len(layers)} layers)")


if __name__ == "__main__":
    main()
