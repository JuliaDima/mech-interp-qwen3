"""Full-answer argmax accuracy across all concept datasets.

For each concept pair the model sees the raw prompt (no answer prefix hint)
and we check whether argmax matches every answer token.  Reported for:

  - plain model:           AttributionModel forward pass, original MLP outputs
  - model + transcoders:   same model, MLP outputs replaced by transcoder reconstructions

The transcoder accuracy reveals how faithfully the transcoders reconstruct
the model's computations.

Usage
-----
    # All concepts, T0 template, 100 pairs each
    python experiments/concept_localization/eval_concept_accuracy.py

    # Skip transcoder reconstruction evaluation (faster)
    python experiments/concept_localization/eval_concept_accuracy.py --no_transcoder

    # Specific concept
    python experiments/concept_localization/eval_concept_accuracy.py --concepts carry gcd

    # Via SLURM
    sbatch --time=01:00:00 scripts/sbatch_run.sh \\
        python experiments/concept_localization/eval_concept_accuracy.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.concept_localization.run_concept import (
    CONCEPTS,
    _MODEL,
    _TRANSCODER_SET,
    _load_concept,
)
from mechinterp_qwen3.interventions import run_with_transcoder_reconstruction
from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input


@torch.no_grad()
def _full_answer_correct(model, prompt: str, answer: str, *, reconstruct: bool = False) -> bool:
    """Full-answer argmax accuracy using AttributionModel with tokenize_qwen_input.

    tokenize_qwen_input prepends a sink token, so logits[n+j] predicts answer_ids[j].

    reconstruct=False: plain model forward pass (original MLP outputs)
    reconstruct=True:  MLP outputs replaced by transcoder reconstructions
    """
    tokenizer = model.tokenizer
    device = model.cfg.device
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    if not answer_ids:
        return False
    tokens = tokenize_qwen_input(prompt_ids + answer_ids, tokenizer, device).unsqueeze(0)
    if reconstruct:
        logits = run_with_transcoder_reconstruction(model, tokens)[0]
    else:
        logits = model(tokens)[0]
    n = len(prompt_ids)
    return all(
        int(logits[n + j].argmax()) == tok_id
        for j, tok_id in enumerate(answer_ids)
    )


def eval_concept(
    model,
    concept: str,
    n: int,
    seed: int,
    template: str,
    *,
    with_transcoder: bool = True,
) -> dict:
    pairs = _load_concept(concept, n, seed)
    if template:
        pairs = [p for p in pairs if p.template == template]
    if not pairs:
        return {"concept": concept, "n": 0, "accuracy": None, "acc_pos": None, "acc_neg": None}

    plain_pos = plain_neg = tc_pos = tc_neg = total = 0

    for pair in tqdm(pairs, desc=concept, leave=False):
        pred_pos = pair.predict_pos or pair.label_pos
        pred_neg = pair.predict_neg or pair.label_neg

        if _full_answer_correct(model, pair.prompt_pos, pred_pos):
            plain_pos += 1
        if _full_answer_correct(model, pair.prompt_neg, pred_neg):
            plain_neg += 1

        if with_transcoder:
            if _full_answer_correct(model, pair.prompt_pos, pred_pos, reconstruct=True):
                tc_pos += 1
            if _full_answer_correct(model, pair.prompt_neg, pred_neg, reconstruct=True):
                tc_neg += 1

        total += 1

    result = {
        "concept":  concept,
        "template": template or "all",
        "n_pairs":  total,
        "accuracy": round((plain_pos + plain_neg) / (2 * total), 4),
        "acc_pos":  round(plain_pos / total, 4),
        "acc_neg":  round(plain_neg / total, 4),
    }
    if with_transcoder:
        result["tc_accuracy"] = round((tc_pos + tc_neg) / (2 * total), 4)
        result["tc_acc_pos"]  = round(tc_pos / total, 4)
        result["tc_acc_neg"]  = round(tc_neg / total, 4)
    return result


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--concepts",       nargs="+", default=CONCEPTS)
    ap.add_argument("--n",              type=int,  default=100,
                    help="Pairs to load per concept (before template filter)")
    ap.add_argument("--template",       type=str,  default="T0",
                    help="Template to filter to. Empty string = all templates.")
    ap.add_argument("--seed",           type=int,  default=42)
    ap.add_argument("--dtype",          default="bfloat16")
    ap.add_argument("--no_transcoder",  action="store_true",
                    help="Skip the transcoder reconstruction evaluation (faster)")
    ap.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    ap.add_argument("--out",            type=Path, default=None,
                    help="JSON output path (default: runs/concept_localization/accuracy_teacher_forced.json)")
    args = ap.parse_args()

    out_path = args.out or (
        _REPO_ROOT / "runs" / "concept_localization" / "accuracy_teacher_forced.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from mechinterp_qwen3.attribution_model import AttributionModel
    from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
    from mechinterp_qwen3.utils.model_utils import parse_dtype

    dtype = parse_dtype(args.dtype)
    print(f"Loading {_MODEL} + transcoders ({args.transcoder_set}) on {device} ({dtype})…")
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True,
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        _MODEL, transcoder_set, dtype=dtype, device=device,
    )
    model.eval()

    results = []
    for concept in args.concepts:
        try:
            r = eval_concept(model, concept, args.n, args.seed, args.template,
                             with_transcoder=not args.no_transcoder)
            results.append(r)
            tc_str = (f"  tc={r['tc_accuracy']:.3f} (pos={r['tc_acc_pos']:.3f} neg={r['tc_acc_neg']:.3f})"
                      if "tc_accuracy" in r else "")
            print(f"  {concept:30s}  acc={r['accuracy']:.3f} "
                  f"(pos={r['acc_pos']:.3f} neg={r['acc_neg']:.3f})"
                  f"  n={r['n_pairs']}{tc_str}")
        except Exception as e:
            print(f"  {concept:30s}  ERROR: {e}")
            results.append({"concept": concept, "error": str(e)})

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")

    valid = [r for r in results if r.get("accuracy") is not None]
    if valid:
        has_tc = any("tc_accuracy" in r for r in valid)
        header = f"{'concept':30s}  {'acc':>6}  {'pos':>6}  {'neg':>6}"
        if has_tc:
            header += f"  {'tc_acc':>6}  {'tc_pos':>6}  {'tc_neg':>6}"
        header += f"  {'n':>5}"
        print(f"\n{header}")
        print("-" * len(header))
        for r in sorted(valid, key=lambda x: x["accuracy"], reverse=True):
            line = (f"  {r['concept']:28s}  {r['accuracy']:6.3f}  "
                    f"{r['acc_pos']:6.3f}  {r['acc_neg']:6.3f}")
            if has_tc:
                line += (f"  {r.get('tc_accuracy', float('nan')):6.3f}  "
                         f"{r.get('tc_acc_pos', float('nan')):6.3f}  "
                         f"{r.get('tc_acc_neg', float('nan')):6.3f}")
            line += f"  {r['n_pairs']:5d}"
            print(line)
        mean_acc = sum(r["accuracy"] for r in valid) / len(valid)
        print(f"\n  {'mean':28s}  {mean_acc:6.3f}")


if __name__ == "__main__":
    main()
