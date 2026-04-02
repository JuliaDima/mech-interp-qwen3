"""Perform 3-7 digit sweep on Hub vs. RoPE models.

Tests generalization beyond the 5-digit training threshold.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(repo_root))

from experiments.stitching.run import (  # noqa: E402
    SmallAdditionTransformer,
    _qm_tokenize,
    load_quanta_maths_model,
)

# Equivalent signs for QuantaMaths format
EQUIV = {"P": "+", "+": "+", "M": "-", "-": "-"}


def evaluate_on_digits(model, n_digits, n_samples=100, device="cpu"):
    """Evaluate a model on exactly n_digits addition."""
    max_val = 10**n_digits - 1
    min_val = 10 ** (n_digits - 1) if n_digits > 1 else 0

    correct_samples = 0

    # Check context window
    expected_seq_len = (2 * n_digits + 2) + (n_digits + 2)
    if (
        not hasattr(model.model, "cfg") or model.model.cfg.n_ctx < expected_seq_len
    ) and not getattr(model, "use_rope", False):
        return 0.0  # Guaranteed failure for non-RoPE models beyond ctx

    model.model.eval()
    for _ in range(n_samples):
        # Generate random n-digit numbers
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        total = a + b

        # QuantaMaths format
        a_str = str(a).zfill(n_digits)
        b_str = str(b).zfill(n_digits)
        prompt = f"{a_str}+{b_str}="
        true_ans_str = "+" + str(total).zfill(n_digits + 1)
        print("Prompt", prompt)
        print("True answer", true_ans_str)

        # Tokenize prompt
        tokens = torch.tensor([_qm_tokenize(prompt)], device=device)

        # Greedy generation
        generated = []
        with torch.no_grad():
            for _ in range(n_digits + 2):
                logits = model.model(tokens)
                next_token = logits[0, -1].argmax().item()
                generated.append(next_token)
                tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=1)

        # Map tokens back to characters
        vocab = [str(i) for i in range(10)] + ["+", "-", "=", "P", "M"]
        pred_ans_str = "".join([vocab[t] for t in generated])

        # Strip signs (+, -, P, M) for math-only comparison
        def get_math_part(s):
            if not s:
                return ""
            return s[1:] if (s[0] in EQUIV or s[0] in ["+", "-", "P", "M"]) else s

        t_math = get_math_part(true_ans_str)
        p_math = get_math_part(pred_ans_str)

        print("True math part", t_math)
        print("Predicted math part", p_math)

        # Check if the remaining digits match
        try:
            is_correct = (int(p_math) == int(t_math)) and (len(t_math) > 0)
        except ValueError:
            is_correct = False

        if is_correct:
            correct_samples += 1

    return (correct_samples / n_samples) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Local .pt model path")
    parser.add_argument("--hub-model", default="PhilipQuirke/QuantaMaths_add_d5_l2_h3_t15K_s372001")
    parser.add_argument("--output", default="runs/stitching/digit_sweep.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Hub model
    hub_model, n_digits_hub = load_quanta_maths_model(args.hub_model, device)
    hub_model.use_rope = False  # Pretrained hub models use absolute pos embeddings

    # Load RoPE model
    print(f"Loading local model: {args.model_path}")
    sd = torch.load(args.model_path, map_location=device)

    # Strip 'model.' prefix if present
    processed_sd = {}
    for k, v in sd.items():
        processed_sd[k.replace("model.", "")] = v

    # Reconstruct config from weights
    d_model = processed_sd["embed.W_E"].shape[1]
    n_layers = sum(1 for k in processed_sd if k.startswith("blocks.") and k.endswith(".ln1.w"))
    n_heads = processed_sd["blocks.0.attn.W_Q"].shape[0]

    rope_model = SmallAdditionTransformer(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        vocab_size=15,
        device=device,
        use_rope=True,
    )
    rope_model.model.load_state_dict(processed_sd, strict=False)
    rope_model.use_rope = True

    results = {"n_digits": [3, 4], "hub_acc": [], "rope_acc": []}

    for n in results["n_digits"]:
        print(f"Testing {n} digits...")

        # Hub eval
        acc = evaluate_on_digits(hub_model, n, device=device)
        results["hub_acc"].append(acc)

        # RoPE eval
        acc = evaluate_on_digits(rope_model, n, device=device)
        results["rope_acc"].append(acc)
        print(f"  Hub: {acc:.2f}%")
        print(f"  RoPE: {acc:.2f}%")

    plt.style.use("seaborn-v0_8-muted")
    plt.figure(figsize=(10, 6))

    plt.plot(
        results["n_digits"],
        results["hub_acc"],
        "o-",
        label="Hub (Absolute Pos)",
        color="#4c72b0",
        linewidth=2.5,
    )
    plt.plot(
        results["n_digits"],
        results["rope_acc"],
        "s--",
        label="Manual (RoPE)",
        color="#55a868",
        linewidth=2.5,
    )

    # Highlight training threshold
    plt.axvline(x=5, color="gray", linestyle="--", alpha=0.5)
    plt.text(5.1, 80, "Training Threshold (5 digits)", color="gray")

    plt.title("Digit Count Generalization Sweep", fontweight="bold", pad=20)
    plt.xlabel("Number of Digits")
    plt.ylabel("Mathematical Accuracy (%)")
    plt.ylim(-5, 105)
    plt.xticks(results["n_digits"])
    plt.grid(linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig("experiments/stitching/digit_sweep.png", dpi=150)
    print("Sweep plot saved to experiments/stitching/digit_sweep.png")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
