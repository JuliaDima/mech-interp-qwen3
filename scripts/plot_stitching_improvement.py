#!/usr/bin/env python3
"""Evaluate original vs stitched Qwen3-4B on random samples and a 2D grid."""

import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from experiments.stitching.run import (  # noqa: E402
    _load_large_model,
    _load_small_model,
    _load_small_sae,
    get_small_model_tokenizer,
)


def evaluate_pairs(
    pairs,
    large_model,
    small_model,
    small_sae,
    W,
    b_vec,
    tokenize_small,
    sae_layer,
    best_large_layer,
    batch_size=32,
    device="cuda",
):
    n_digits = 5
    probs_before_all = []
    probs_after_all = []

    with torch.no_grad():
        for start_idx in tqdm(range(0, len(pairs), batch_size), desc="Evaluating pairs"):
            batch_pairs = pairs[start_idx : start_idx + batch_size]

            large_tokens_list = []
            small_ids_list = []
            ans_tokens_list = []

            for a, b in batch_pairs:
                a_str = f"{a:0>{n_digits}}"
                b_str = f"{b:0>{n_digits}}"
                ans_str = str(a + b)
                prompt = f"{a_str}+{b_str}="

                large_tokens_list.append(large_model.to_tokens(prompt)[0])

                small_text = f"{a_str}+{b_str}={a+b}"
                small_ids_list.append(tokenize_small(small_text))

                ans_token = large_model.tokenizer.encode(ans_str[0], add_special_tokens=False)[0]
                ans_tokens_list.append(ans_token)

            small_ids = torch.tensor(small_ids_list, device=device, dtype=torch.long)

            # Find '=' position in small_text ("0000a+0000b=" -> 5+1+5=11)
            eq_small_idx = 11

            cache_sm = {}

            def _hook_sm(act, hook, _pos=eq_small_idx, cache_sm=cache_sm):
                cache_sm["v"] = act[:, _pos, :].clone()
                return act

            with small_model.model.hooks(
                fwd_hooks=[(f"blocks.{sae_layer}.hook_resid_mid", _hook_sm)]
            ):
                small_model.model(small_ids)

            resid_mid = cache_sm["v"].to(dtype=torch.float32)
            feats = small_sae.encode(resid_mid)
            small_mlp_out = small_sae.decode(feats)
            stitched = (small_mlp_out @ W.T + b_vec).to(dtype=large_model.cfg.dtype)

            max_target_len = max(len(t) for t in large_tokens_list)
            padded_large_tokens = []
            eq_large_list = []
            for t in large_tokens_list:
                pad_len = max_target_len - len(t)
                padded_t = F.pad(t, (0, pad_len), value=0)
                padded_large_tokens.append(padded_t)
                eq_large_list.append(len(t) - 1)
            large_tokens = torch.stack(padded_large_tokens).to(device)

            logits_before = large_model(large_tokens)

            def patch_hook(act, hook, _val=stitched, _pos_list=eq_large_list):
                act = act.clone()
                for b_idx, p in enumerate(_pos_list):
                    act[b_idx, p, :] = _val[b_idx, :]
                return act

            with large_model.hooks(
                fwd_hooks=[(f"blocks.{best_large_layer}.hook_mlp_out", patch_hook)]
            ):
                logits_after = large_model(large_tokens)

            for idx in range(len(batch_pairs)):
                a, b = batch_pairs[idx]
                pos = eq_large_list[idx]
                ans_tok = ans_tokens_list[idx]

                pb = F.softmax(logits_before[idx, pos], dim=-1)[ans_tok].item()
                pa = F.softmax(logits_after[idx, pos], dim=-1)[ans_tok].item()

                probs_before_all.append(pb)
                probs_after_all.append(pa)

    return np.array(probs_before_all), np.array(probs_after_all)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path("runs/stitching")

    class Args:
        pass

    args = Args()
    args.model = "Qwen/Qwen3-4B"
    args.transcoder_set = "mwhanna/qwen3-4b-transcoders"
    args.small_sae_layer = None
    args.small_model_layers = 2
    args.small_model_num_digits = 5
    args.hub_model = "PhilipQuirke/QuantaMaths_add_d5_l1_h3_t15K_s372001"

    dtype = torch.bfloat16

    print("Loading models...")
    large_model = _load_large_model(args, dtype, device=device)
    small_model = _load_small_model(args, out_root / "small_model.pt", device)
    small_sae = _load_small_sae(args, out_root / "small_sae.safetensors", device)
    stitch_maps = torch.load(out_root / "stitch_maps.pt", map_location=device)

    small_model.model.cfg.n_ctx = 19
    tokenize_small = get_small_model_tokenizer(small_model, max_len=19)

    best_large_layer, best_map = max(
        stitch_maps.items(), key=lambda x: x[1].get("cca", x[1].get("r2", 0.0))
    )
    W = best_map["W"].to(device)
    b_vec = best_map["b"].to(device)

    sae_layer = small_model.n_layers - 1

    large_model.eval()
    small_model.model.eval()
    small_sae.eval()

    # 1. Evaluate 10 random large problems for greedy gain bar chart
    print("\n--- Evaluating Random 5-digit Problems ---")
    random.seed(42)
    random_pairs = [(random.randint(10000, 99999), random.randint(10000, 99999)) for _ in range(10)]
    pb_rand, pa_rand = evaluate_pairs(
        random_pairs,
        large_model,
        small_model,
        small_sae,
        W,
        b_vec,
        tokenize_small,
        sae_layer,
        best_large_layer,
        batch_size=10,
        device=device,
    )

    results = []
    for i, (a, b) in enumerate(random_pairs):
        gain = (pa_rand[i] - pb_rand[i]) * 100
        print(f"Sample {i+1}: {a}+{b} = {a+b} -> Gain: {gain:+.2f} pp")
        results.append(
            {
                "problem": f"{a}+{b}",
                "Base": pb_rand[i] * 100,
                "Patched": pa_rand[i] * 100,
                "Gain": gain,
            }
        )
    df = pd.DataFrame(results)

    sns.set_theme(style="whitegrid", palette="magma")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="problem", y="Gain", palette="viridis")
    plt.xticks(rotation=45)
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.title(
        "Functional Alignment: Probability Boost ($P_{correct}$)", fontsize=16, fontweight="bold"
    )
    plt.ylabel("Absolute Gain (Percentage Points)", fontsize=12)
    plt.xlabel("Addition Problem (5-digit)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_root / "stitching_improvement.png", dpi=300)
    plt.close()
    print(f"Saved bar chart to {out_root / 'stitching_improvement.png'}")

    # 2. Evaluate Grid [0, 30] x [0, 30]
    print("\n--- Evaluating Grid [0, 30] ---")
    max_val = 30
    grid_pairs = [(a, b) for a in range(max_val + 1) for b in range(max_val + 1)]
    pb_grid, pa_grid = evaluate_pairs(
        grid_pairs,
        large_model,
        small_model,
        small_sae,
        W,
        b_vec,
        tokenize_small,
        sae_layer,
        best_large_layer,
        batch_size=32,
        device=device,
    )

    grid_before = np.zeros((max_val + 1, max_val + 1))
    grid_after = np.zeros((max_val + 1, max_val + 1))

    for idx, (a, b) in enumerate(grid_pairs):
        grid_before[b, a] = pb_grid[idx]
        grid_after[b, a] = pa_grid[idx]

    avg_diff = np.mean(grid_after - grid_before) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"]
    cmap_prob = LinearSegmentedColormap.from_list("prob", colors, N=100)

    im1 = ax1.imshow(grid_before, cmap=cmap_prob, aspect="auto", origin="lower", vmin=0, vmax=1)
    ax1.set_title("Original Qwen3-4B", fontsize=15)
    ax1.set_xlabel("a", fontsize=12)
    ax1.set_ylabel("b", fontsize=12)
    fig.colorbar(im1, ax=ax1, label="P(correct first answer digit)")

    im2 = ax2.imshow(grid_after, cmap=cmap_prob, aspect="auto", origin="lower", vmin=0, vmax=1)
    ax2.set_title(rf"Stitched Qwen3-4B ($\Delta$ {avg_diff:+.1f}pp)", fontsize=15)
    ax2.set_xlabel("a", fontsize=12)
    ax2.set_ylabel("b", fontsize=12)
    fig.colorbar(im2, ax=ax2, label="P(correct first answer digit)")

    for ax in [ax1, ax2]:
        ax.set_xticks(np.arange(0, max_val + 1, 5))
        ax.set_yticks(np.arange(0, max_val + 1, 5))
        ax.grid(True, alpha=0.2, color="white", linewidth=0.5)

        for i in range(max_val + 1):
            for j in range(max_val + 1):
                carry_current = ((i % 10) + (j % 10)) >= 10
                if i < max_val:
                    carry_right = (((i + 1) % 10) + (j % 10)) >= 10
                    if carry_current != carry_right:
                        ax.axvline(i + 0.5, color="black", linewidth=0.5, alpha=0.4)
                if j < max_val:
                    carry_up = ((i % 10) + ((j + 1) % 10)) >= 10
                    if carry_current != carry_up:
                        ax.axhline(j + 0.5, color="black", linewidth=0.5, alpha=0.4)

    plt.suptitle("Probability of Correct First Digit: Stitched vs Original", fontsize=16, y=1.02)
    plt.tight_layout()

    plt.savefig(out_root / "stitching_grid_comparison.png", dpi=300)
    print(f"Saved heatmaps to {out_root / 'stitching_grid_comparison.png'}")


if __name__ == "__main__":
    main()
