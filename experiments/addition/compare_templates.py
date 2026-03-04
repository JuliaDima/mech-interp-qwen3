import os
import random
import sys

import torch
from tqdm import tqdm

# 1. Setup paths to ensure we can import from the source tree
repo_root = "/home/eid23/mechinterp-qwen-3B-Instruct/mechinterp-qwen3"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
os.chdir(repo_root)

from experiments.addition.prompts import CALC_GRID  # noqa: E402
from mechinterp_qwen3.attribution_model import AttributionModel  # noqa: E402
from mechinterp_qwen3.dataset_generation.generate_add_dataset import (  # noqa: E402
    TEMPLATES,
    build_prompt,
)
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub  # noqa: E402

# 2. Configuration
model_name = "Qwen/Qwen3-4B"
transcoder_set = "mwhanna/qwen3-4b-transcoders"
n_samples = 100  # Number of random prompts to test per template
device = "cuda" if torch.cuda.is_available() else "cpu"


def run_benchmark():
    print(f"Loading model {model_name} on {device}...")
    transcoder, _ = load_transcoder_from_hub(transcoder_set, dtype=torch.bfloat16)
    model = AttributionModel.from_pretrained_and_transcoders(
        model_name, transcoder, dtype=torch.bfloat16, device=device
    )

    # Test on random prompts from the grid
    random.seed(42)
    subset = random.sample(CALC_GRID, min(n_samples, len(CALC_GRID)))

    template_results = {}

    print(f"\nBenchmarking {len(subset)} prompts across {len(TEMPLATES)} templates...")

    for tid, fmt in TEMPLATES.items():
        print(f"\nTesting template {tid}: {fmt!r}")
        correct = 0

        failures_seen = 0
        for entry in tqdm(subset, desc=f"Template {tid}"):
            a, b = entry["a"], entry["b"]
            prompt = build_prompt(tid, a, b)
            target_str = str(a + b)

            # Get target token ID (first token of answer)
            target_token_id = model.to_tokens(target_str, prepend_bos=False)[0, 0].item()

            # Tokenize using the standardized tokenize_qwen_input
            tokens = model.tokenize_qwen_input(prompt).to(device)
            if tokens.ndim == 1:
                tokens = tokens.unsqueeze(0)

            with torch.no_grad():
                logits = model(tokens)
                last_logits = logits[0, -1, :]
                predicted_token_id = torch.argmax(last_logits).item()

                if predicted_token_id == target_token_id:
                    correct += 1
                elif failures_seen < 3:
                    failures_seen += 1
                    topk_vals, topk_ids = torch.topk(torch.softmax(last_logits, dim=-1), k=3)
                    top_preds = [
                        f"'{model.tokenizer.decode([tid])}' ({val:.1%})"
                        for tid, val in zip(topk_ids, topk_vals, strict=False)
                    ]
                    print(f"\n  Failure on {a}+{b}:")
                    print(f"    Target: '{target_str[0]}' (ID {target_token_id})")
                    print(
                        f"    Got   : '{model.tokenizer.decode([predicted_token_id])}' (ID {predicted_token_id})"
                    )
                    print(f"    Top 3 : {', '.join(top_preds)}")

        acc = correct / len(subset)
        template_results[tid] = acc
        print(f"Accuracy for {tid}: {acc:.2%}")

    # 3. Final Summary Table
    print("\n" + "=" * 70)
    print(f"{'ID':5s} | {'Template String':35s} | {'Accuracy':10s}")
    print("-" * 70)
    # Sort by accuracy descending
    sorted_results = sorted(template_results.items(), key=lambda x: x[1], reverse=True)
    for tid, acc in sorted_results:
        print(f"{tid:5s} | {TEMPLATES[tid][:35]:35s} | {acc:.2%}")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
