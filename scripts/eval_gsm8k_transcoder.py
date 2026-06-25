"""Evaluate Qwen3-4B on GSM8K in two modes:
  - baseline: normal forward pass (transcoders loaded but not substituted)
  - substituted: every MLP output replaced by the transcoder reconstruction

Usage
-----
    python scripts/eval_gsm8k_transcoder.py                        # both modes, 100 examples
    python scripts/eval_gsm8k_transcoder.py --mode baseline        # baseline only
    python scripts/eval_gsm8k_transcoder.py --mode substituted     # substituted only
    python scripts/eval_gsm8k_transcoder.py --n_examples 1319      # full test set
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from transformers import AutoModelForCausalLM, AutoTokenizer

from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype
from scripts.model_config import add_model_config_arg, default_model, default_transcoder_set, resolve_model_args

_MODEL = default_model()
_TRANSCODER_SET = default_transcoder_set()

_SYSTEM_PROMPT = (
    "You are a math problem solver. Think step by step. "
    "At the end of your answer, write the final numeric answer as: #### <number>"
)

# Qwen3 chat template: /no_think disables the <think> block for speed
_PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{question} /no_think<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _format_prompt(question: str) -> str:
    return _PROMPT_TEMPLATE.format(system=_SYSTEM_PROMPT, question=question)


def _extract_answer(text: str) -> str | None:
    """Parse the #### N answer from model output."""
    m = re.search(r"####\s*([-\d,\.]+)", text)
    if m is None:
        return None
    return m.group(1).replace(",", "").strip()


def _extract_gold(answer: str) -> str:
    m = re.search(r"####\s*([-\d,\.]+)", answer)
    assert m, f"No gold answer found in: {answer!r}"
    return m.group(1).replace(",", "").strip()


@torch.no_grad()
def _greedy_decode(hf_model, hf_tokenizer, token_ids: list[int],
                   max_new_tokens: int = 512, device=None) -> str:
    """Baseline greedy decode via HF generate() — uses KV cache for speed."""
    ids = torch.tensor([token_ids], device=device)
    out = hf_model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=hf_tokenizer.eos_token_id,
    )
    generated = out[0, len(token_ids):]
    return hf_tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def _greedy_decode_substituted(model, token_ids: list[int], max_new_tokens: int = 512) -> str:
    """Greedy decode with every MLP output replaced by its transcoder reconstruction.

    Hooks are built once per example (not per token) and reused across the
    token loop. Still no KV cache — inherent to HookedTransformer — but avoids
    the per-token hook-rebuild overhead.
    """
    tc_set  = model.transcoders
    n_layers = len(model.blocks)
    device   = model.cfg.device

    pre_mlp_cache: dict[int, torch.Tensor] = {}
    input_hooks, output_hooks = [], []
    for layer in range(n_layers):
        tc = tc_set.transcoders[layer]

        def _cache(acts, hook, _l=layer):
            pre_mlp_cache[_l] = acts
            return acts

        def _substitute(acts, hook, _l=layer, _tc=tc):
            return _tc(pre_mlp_cache.pop(_l)).to(acts.dtype)

        input_hooks.append((f"blocks.{layer}.{model.feature_input_hook}", _cache))
        output_hooks.append((f"blocks.{layer}.{model.original_feature_output_hook}", _substitute))

    all_hooks = input_hooks + output_hooks
    ids = torch.tensor([token_ids], device=device)
    generated = []
    for _ in range(max_new_tokens):
        logits  = model.run_with_hooks(ids, fwd_hooks=all_hooks)
        next_id = int(logits[0, -1].argmax())
        if next_id == model.tokenizer.eos_token_id:
            break
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    return model.tokenizer.decode(generated, skip_special_tokens=True)


def evaluate(model, hf_model, hf_tokenizer, examples: list[dict],
             mode: str, max_new_tokens: int = 512,
             start_idx: int = 0, out_path: Path | None = None) -> float:
    """Evaluate examples[start_idx:], saving partial results to out_path after each one."""
    device = next(hf_model.parameters()).device

    # Load prior results if resuming
    results: list[dict] = []
    if out_path and out_path.exists():
        import json as _json
        prior = _json.loads(out_path.read_text())
        results = prior.get(mode, [])

    correct = sum(1 for r in results if r["correct"])

    for i, ex in enumerate(tqdm(examples[start_idx:], desc=f"[{mode}]",
                                initial=start_idx, total=len(examples))):
        global_i = start_idx + i
        prompt    = _format_prompt(ex["question"])
        token_ids = hf_tokenizer(prompt, add_special_tokens=False).input_ids

        if mode == "baseline":
            output = _greedy_decode(hf_model, hf_tokenizer, token_ids,
                                    max_new_tokens, device)
        else:
            output = _greedy_decode_substituted(model, token_ids, max_new_tokens)

        pred = _extract_answer(output)
        gold = _extract_gold(ex["answer"])
        is_correct = pred == gold
        if is_correct:
            correct += 1

        results.append({"idx": global_i, "pred": pred, "gold": gold, "correct": is_correct, "output": output})

        # Save after every example so a killed job can resume
        if out_path:
            _save_partial(out_path, mode, results)

    total = start_idx + len(examples[start_idx:])
    acc = correct / total if total else 0.0
    print(f"[{mode}] {correct}/{total}  accuracy={acc:.3f}")
    return acc


def _save_partial(out_path: Path, mode: str, results: list[dict]) -> None:
    import json as _json
    data: dict = {}
    if out_path.exists():
        try:
            data = _json.loads(out_path.read_text())
        except Exception:
            pass
    data[mode] = results
    out_path.write_text(_json.dumps(data, indent=2))


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_model_config_arg(ap)
    ap.add_argument("--model", default=None)
    ap.add_argument("--transcoder_set", default=None)
    ap.add_argument("--mode", choices=["baseline", "substituted", "both"], default="both")
    ap.add_argument("--n_examples", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start_idx", type=int, default=0,
                    help="Resume from this example index (skip already-done examples)")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON file for partial + final results. Auto-detects start_idx if file exists.")
    args = ap.parse_args()
    resolve_model_args(args)

    out_path = args.out or (
        Path("runs") / "gsm8k" / f"gsm8k_{args.mode}_{args.n_examples}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect start_idx from existing partial results
    start_idx = args.start_idx
    if start_idx == 0 and out_path.exists():
        import json as _json
        try:
            prior = _json.loads(out_path.read_text())
            modes = ["baseline", "substituted"] if args.mode == "both" else [args.mode]
            for m in modes:
                done = len(prior.get(m, []))
                if done > 0:
                    print(f"  Resuming {m} from example {done} (found partial results)")
                    start_idx = max(start_idx, done)
        except Exception:
            pass

    print(f"Loading GSM8K test set ({args.n_examples} examples)…")
    ds = load_dataset("gsm8k", "main", split="test")
    examples = list(ds.select(range(min(args.n_examples, len(ds)))))

    device = get_default_device()
    dtype  = parse_dtype(args.dtype)
    print(f"Loading {args.model} on {device} ({dtype})…")

    # HF model used for baseline (has KV cache via generate())
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model     = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device)
    hf_model.eval()

    # AttributionModel needed only for substituted mode
    model = None
    if args.mode in ("substituted", "both"):
        tc_set, _ = load_transcoder_from_hub(args.transcoder_set, dtype=dtype,
                                             lazy_encoder=False, lazy_decoder=False)
        model = AttributionModel.from_pretrained_and_transcoders(
            args.model, tc_set, dtype=dtype, device=device)
        model.eval()

    modes = ["baseline", "substituted"] if args.mode == "both" else [args.mode]
    accs = {}
    for mode in modes:
        accs[mode] = evaluate(model, hf_model, hf_tokenizer,
                              examples, mode, args.max_new_tokens,
                              start_idx=start_idx, out_path=out_path)

    print(f"\nFinal results → {out_path}")
    if len(accs) == 2:
        drop = accs["baseline"] - accs["substituted"]
        print(f"Accuracy drop from transcoder substitution: {drop:+.3f}")


if __name__ == "__main__":
    main()
