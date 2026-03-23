import matplotlib
import torch

matplotlib.use("Agg")
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from huggingface_hub import hf_hub_download

from experiments.stitching.run import _load_small_model, _qm_make_sample


def plot_shift_sensitivity_intuitive():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "PhilipQuirke/QuantaMaths_add_d5_l1_h3_t15K_s372001"

    print(f"Downloading {model_id}...")
    path = hf_hub_download(repo_id=model_id, filename="model.pth")

    print(f"Loading {model_id}...")
    args = SimpleNamespace(hub_model=model_id)
    model = _load_small_model(args, Path(path), device)
    model.model.eval()

    char_map = {str(i): i for i in range(10)}
    char_map.update({"+": 10, "=": 11, " ": 12})

    def manual_tokenize(text):
        return [char_map.get(c, 13) for c in text]

    a, b = 12345, 67890
    sample = _qm_make_sample(a, b, 5)
    full_text = sample["full"]
    orig_tokens = manual_tokenize(full_text)
    n_ctx = model.model.cfg.n_ctx

    shifts = [-2, -1, 0, 1, 2]
    accuracies = []
    labels = []

    for s in shifts:
        if s < 0:
            display_text = full_text[-s:] + "_" * (-s)
            shifted = orig_tokens[-s:] + [12] * (-s)
        elif s > 0:
            display_text = " " * s + full_text[:-s]
            shifted = [12] * s + orig_tokens[:-s]
        else:
            display_text = full_text
            shifted = list(orig_tokens)

        shifted = (shifted + [12] * n_ctx)[:n_ctx]
        input_bits = torch.tensor([shifted], device=device)

        with torch.no_grad():
            logits = model.model(input_bits)
            targets = orig_tokens[12:19]
            preds = logits[0, 11:18].argmax(dim=-1).cpu().numpy()
            acc = float(np.mean(preds == targets))
            accuracies.append(acc)

            clean_display = display_text[:12].replace(" ", "_")
            labels.append(f"Shift {s}\n'{clean_display}...'")
            print(f"Shift {s}: Accuracy {acc:.2f}")

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    colors = ["#d65f5f" if s != 0 else "#4878d0" for s in shifts]
    plt.bar(shifts, accuracies, color=colors, edgecolor="black", alpha=0.8, width=0.6)

    plt.title("Fixed-Width 'Stiffness' Demo", fontsize=15, fontweight="bold")
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Input Shift", fontsize=12)
    plt.xticks(shifts, labels, fontsize=9)
    plt.ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig("runs/stitching/shift_sensitivity.png")
    print("Saved runs/stitching/shift_sensitivity.png")


if __name__ == "__main__":
    Path("runs/stitching").mkdir(parents=True, exist_ok=True)
    plot_shift_sensitivity_intuitive()
