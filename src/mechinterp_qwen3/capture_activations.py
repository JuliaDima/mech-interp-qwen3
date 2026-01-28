from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .hooks import MLPHookManager
from .io import read_jsonl, write_json
from .utils_seed import SeedConfig, set_all_seeds


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--run_path", type=str, required=True, help="Path to an existing run folder (contains prompts.jsonl).")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=str, default="4,12,20,28", help="Comma-separated layer ids to capture.")
    ap.add_argument("--limit", type=int, default=80)
    args = ap.parse_args()

    set_all_seeds(SeedConfig(seed=args.seed))

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    layers: List[int] = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    run_path = Path(args.run_path)
    prompts_path = run_path / "prompts.jsonl"
    if not prompts_path.exists():
        raise FileNotFoundError(f"Expected {prompts_path} to exist. Run baseline first.")

    prompt_rows = read_jsonl(prompts_path)[: args.limit]

    acts_dir = run_path / "activations"
    acts_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    hooker = MLPHookManager(model, layer_ids=layers)
    hooker.install()

    saved = 0
    for r in prompt_rows:
        hooker.clear_cache()

        # Prompt-only forward pass:
        inputs = tok(r["prompt"], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        _ = model(**inputs)

        # Save per prompt (keeps variable seq lens easy)
        prompt_id = r["prompt_id"]
        out: Dict[str, Any] = {
            "prompt_id": prompt_id,
            "behaviour": r.get("behaviour"),
            "layers": layers,
            "seq_len": int(inputs["input_ids"].shape[1]),
        }

        # Save tensors in separate .pt file to avoid JSON bloat
        tensor_payload = {
            "input_ids": inputs["input_ids"][0].detach().to("cpu"),
            "attention_mask": inputs.get("attention_mask", None)[0].detach().to("cpu") if "attention_mask" in inputs else None,
            "per_layer": {},
        }

        for lid in layers:
            la = hooker.cache[lid]
            if la.mlp_in is None or la.mlp_out is None:
                raise RuntimeError(f"Missing activations for layer {lid}. Hook path may be wrong for this model.")
            tensor_payload["per_layer"][lid] = {
                "mlp_in": la.mlp_in,   # [seq, d_model]
                "mlp_out": la.mlp_out, # [seq, d_model]
            }

        pt_path = acts_dir / f"{prompt_id}.pt"
        torch.save(tensor_payload, pt_path)

        out["tensor_file"] = str(pt_path)
        write_json(acts_dir / f"{prompt_id}.meta.json", out)

        saved += 1

    hooker.remove()

    summary = {
        "saved": saved,
        "layers": layers,
        "dtype": args.dtype,
        "device": args.device,
        "model": args.model,
        "run_path": str(run_path),
    }
    write_json(run_path / "activation_capture_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()