from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .io import read_jsonl, sha256_file, write_json, write_jsonl
from .utils_seed import SeedConfig, set_all_seeds


def now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def get_git_commit() -> str | None:
    try:
        import subprocess

        r = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return r
    except Exception:
        return None


def setup_hf_env() -> None:
    # Disable Xet / CAS backend (prevents flaky large-model downloads)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    # Disable hf_transfer unless explicitly enabled by user
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    # Conservative timeouts for slow or unstable networks
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "300")


@torch.inference_mode()
def generate_one(
    model, tok, prompt: str, max_new_tokens: int, temperature: float, top_p: float
) -> str:
    # Format as a chat message for Instruct models
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tok(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0.0
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tok.eos_token_id,
    )
    # Only decode the new tokens.
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tok.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument(
        "--prompts", type=str, default="src/mechinterp_qwen3/prompts/greater_than.jsonl"
    )
    ap.add_argument("--run_dir", type=str, default="runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--max_new_tokens", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)  # greedy by default
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=80)
    args = ap.parse_args()

    set_all_seeds(SeedConfig(seed=args.seed))

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        args.dtype
    ]
    run_id = now_run_id()
    run_path = Path(args.run_dir) / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(args.prompts)
    prompt_rows = read_jsonl(prompt_path)[: args.limit]

    # Save an exact copy of prompts used (freezes the dataset for the run)
    copied_prompts_path = run_path / "prompts.jsonl"
    write_jsonl(copied_prompts_path, prompt_rows)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    outputs = []
    for r in prompt_rows:
        completion = generate_one(
            model,
            tok,
            r["prompt"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        # Normalize baseline answer: take first non-whitespace char if any.
        stripped = completion.strip()
        first_char = stripped[0] if stripped else ""
        outputs.append(
            {
                "prompt_id": r["prompt_id"],
                "expected": r.get("expected"),
                "completion_raw": completion,
                "completion_first_char": first_char,
            }
        )

    out_path = run_path / "outputs.jsonl"
    write_jsonl(out_path, outputs)

    # Minimal accuracy (for this behaviour)
    correct = 0
    total = 0
    for o in outputs:
        if o["expected"] is None:
            continue
        total += 1
        if o["completion_first_char"] == o["expected"]:
            correct += 1

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "git_commit": get_git_commit(),
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "paths": {
            "prompts_src": str(prompt_path),
            "prompts_sha256": sha256_file(prompt_path) if prompt_path.exists() else None,
            "prompts_used": str(copied_prompts_path),
            "outputs": str(out_path),
        },
        "system": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "metrics": {
            "n": len(outputs),
            "accuracy_first_char": (correct / total) if total else None,
            "correct": correct,
            "total": total,
        },
    }
    write_json(run_path / "manifest.json", manifest)

    print(json.dumps(manifest["metrics"], indent=2))


if __name__ == "__main__":
    setup_hf_env()  # to avoid flaky downloads
    main()
