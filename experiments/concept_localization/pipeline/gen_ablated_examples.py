"""Greedy multi-token generation under joint feature ablation, for a handful of
prompts from one concept-localization anchor -- baseline vs each of the
default_encdec / cohens_d configs already picked by
topk_edec_ablation_compare.py (reads their feature lists straight from that
anchor's feature_modulation_compare.json).

Unlike the single-position top-k dumps (generated_tokens.jsonl), this actually
decodes --max_new_tokens tokens autoregressively with the ablation hooks held
active for the whole generation, via HookedTransformer's persistent
model.hooks(...) context (the hooks fire on every step, not just once).

Usage:
    sbatch scripts/sbatch_run.sh python -m experiments.concept_localization.pipeline.gen_ablated_examples \
        --concept gcd --anchor anchor_rank3_pos6 --n_pairs 4 --max_new_tokens 10
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import torch

from experiments.concept_localization.pipeline.run_concept import _MODEL, _TRANSCODER_SET, _load_concept
from experiments.concept_localization.pipeline.run_feature_modulation import parse_feature_name
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.interventions import make_capture_hook, make_subtract_hook
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype
from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input


def _feature_map(feature_keys: list[str]) -> dict[int, list[int]]:
    fm: dict[int, list[int]] = {}
    for key in feature_keys:
        spec = parse_feature_name(key)
        fm.setdefault(spec.layer, []).append(spec.feature_id)
    return fm


def _build_hooks(model, feature_map: dict[int, list[int]], alpha: float = 0.0):
    hooks = []
    for layer, feat_ids in feature_map.items():
        if not feat_ids:
            continue
        sub_name, sub_fn = make_subtract_hook(model, layer, feat_ids, alpha=alpha)
        cap_name, cap_fn = make_capture_hook(model, layer, sub_fn)
        hooks.append((cap_name, cap_fn))
        hooks.append((sub_name, sub_fn))
    return hooks


@torch.no_grad()
def generate(model, prompt: str, feature_map: dict[int, list[int]] | None, max_new_tokens: int) -> str:
    tok = model.tokenizer
    ids = tokenize_qwen_input(
        tok(prompt, add_special_tokens=False).input_ids, tok, model.cfg.device
    ).unsqueeze(0)
    hooks = _build_hooks(model, feature_map) if feature_map else []
    ctx = model.hooks(fwd_hooks=hooks) if hooks else contextlib.nullcontext()
    with ctx:
        out_ids = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False, prepend_bos=False, verbose=False,
        )
    new_tokens = out_ids[0, ids.shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--n_pairs", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=10)
    ap.add_argument("--template", default="T0")
    ap.add_argument("--model", default=_MODEL)
    ap.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    anchor_dir = Path(f"runs/concept_localization/{args.concept}/{args.concept}_T0/{args.anchor}")
    compare_path = anchor_dir / "modulation_topk10_compare" / "feature_modulation_compare.json"
    cfg = json.loads(compare_path.read_text())["config"]
    fm_default = _feature_map(cfg["default_features"])
    fm_cohens = _feature_map(cfg["cohens_d_features"])
    print(f"default_encdec features: {cfg['default_features']}")
    print(f"cohens_d features:       {cfg['cohens_d_features']}")

    device = get_default_device()
    dtype = parse_dtype(args.dtype)
    print(f"Loading model {args.model}")
    transcoder_set, _ = load_transcoder_from_hub(args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True)
    model = AttributionModel.from_pretrained_and_transcoders(args.model, transcoder_set, dtype=dtype, device=device)
    model.eval()

    all_pairs = _load_concept(args.concept, 200, args.seed)
    if args.template and args.template.lower() != "none":
        all_pairs = [p for p in all_pairs if p.template == args.template]

    for pair in all_pairs[: args.n_pairs]:
        for side, prompt in (("pos", pair.prompt_pos), ("neg", pair.prompt_neg)):
            print(f"\n[{side}] {prompt!r}")
            for cfg_name, fm in (("baseline", None), ("default_encdec", fm_default), ("cohens_d", fm_cohens)):
                out = generate(model, prompt, fm, args.max_new_tokens)
                print(f"  {cfg_name:15s}: {out!r}")


if __name__ == "__main__":
    main()
