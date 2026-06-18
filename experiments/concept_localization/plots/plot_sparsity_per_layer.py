"""Plot transcoder feature sparsity per layer and test pos vs neg differences.

Samples pos and neg prompts from the concept dataset, runs forward passes,
computes mean active features per layer, and runs a Welch t-test per layer
to check whether carry (pos) and no-carry (neg) prompts differ in sparsity.

Usage:
    python -m experiments.concept_localization.plots.plot_sparsity_per_layer \\
        --concept carry --sample 25 --template T0
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.concept_localization.pipeline.run_concept import (
    CONCEPTS, _MODEL, _TRANSCODER_SET, _load_concept,
)
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype
from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input

CARRY_LAYERS = [13, 14, 16, 21]


@torch.no_grad()
def count_active_per_layer(model, prompts: list[str]) -> np.ndarray:
    """Returns (n_prompts, n_layers): mean active features per token position."""
    n_layers = model.cfg.n_layers
    n_features = model.transcoders[0].W_enc.shape[0]
    results = np.zeros((len(prompts), n_layers), dtype=np.float32)

    for pi, prompt in enumerate(prompts):
        tokens = tokenize_qwen_input(prompt, model.tokenizer, model.cfg.device)

        mlp_inputs: dict[int, torch.Tensor] = {}
        hooks = []
        for layer in range(n_layers):
            def _hook(acts, hook, _l=layer):
                mlp_inputs[_l] = acts.detach().squeeze(0)
                return acts
            hooks.append((f"blocks.{layer}.{model.feature_input_hook}", _hook))
        model.run_with_hooks(tokens.unsqueeze(0), fwd_hooks=hooks)

        for layer in range(n_layers):
            tc = model.transcoders[layer]
            h_in = mlp_inputs[layer].to(tc.W_enc.dtype)
            acts = F.relu(F.linear(h_in, tc.W_enc, tc.b_enc))
            # mean active features per position (as count, not fraction)
            results[pi, layer] = (acts > 0).float().sum(dim=-1).mean().item()
            del acts
        del mlp_inputs

        if (pi + 1) % 10 == 0:
            print(f"  {pi + 1}/{len(prompts)} prompts done")

    return results


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "** "
    if p < 0.05:
        return "*  "
    return "   "


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", required=True, choices=CONCEPTS)
    parser.add_argument("--sample", type=int, default=25, help="Pairs to sample (gives sample pos + sample neg)")
    parser.add_argument("--template", default="T0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    device = get_default_device()
    dtype = parse_dtype("bfloat16")

    print(f"Loading model {_MODEL}...")
    transcoder_set, _ = load_transcoder_from_hub(
        _TRANSCODER_SET, dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        _MODEL, transcoder_set, dtype=dtype, device=device
    )
    model.eval()

    all_pairs = _load_concept(args.concept, args.n, args.seed)
    if args.template:
        all_pairs = [p for p in all_pairs if p.template == args.template]
    pairs = random.Random(args.seed).sample(all_pairs, min(args.sample, len(all_pairs)))

    pos_prompts = [p.prompt_pos for p in pairs]
    neg_prompts = [p.prompt_neg for p in pairs]

    print(f"\nProcessing {len(pos_prompts)} pos prompts...")
    pos_counts = count_active_per_layer(model, pos_prompts)  # (n, n_layers)
    print(f"Processing {len(neg_prompts)} neg prompts...")
    neg_counts = count_active_per_layer(model, neg_prompts)

    n_layers = model.cfg.n_layers
    layers = np.arange(n_layers)

    mean_pos = pos_counts.mean(axis=0)
    mean_neg = neg_counts.mean(axis=0)
    std_pos  = pos_counts.std(axis=0)
    std_neg  = neg_counts.std(axis=0)
    mean_all = np.concatenate([pos_counts, neg_counts]).mean(axis=0)
    std_all  = np.concatenate([pos_counts, neg_counts]).std(axis=0)

    # Welch t-test per layer (pos vs neg)
    pvalues = np.array([
        stats.ttest_ind(pos_counts[:, l], neg_counts[:, l], equal_var=False).pvalue
        for l in range(n_layers)
    ])
    sig = pvalues < 0.05

    # --- plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import experiments.plot_style as ps
        ps.apply()
        navy, violet, teal, gray = ps.NAVY, ps.VIOLET, ps.TEAL, ps.GRAY
    except Exception:
        navy, violet, teal, gray = "#1f3a5f", "#7b3fa0", "#2a7f6f", "#888888"

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    # Top panel: mean active features per layer
    width = 0.38
    ax.bar(layers - width/2, mean_pos, width, color=violet, alpha=0.75, label="pos (carry)")
    ax.bar(layers + width/2, mean_neg, width, color=teal,   alpha=0.75, label="neg (no-carry)")
    ax.errorbar(layers - width/2, mean_pos, yerr=std_pos, fmt="none",
                color=gray, elinewidth=0.7, capsize=1.5)
    ax.errorbar(layers + width/2, mean_neg, yerr=std_neg, fmt="none",
                color=gray, elinewidth=0.7, capsize=1.5)

    # Mark significant layers with stars above the taller bar
    for l in range(n_layers):
        if sig[l]:
            y_top = max(mean_pos[l] + std_pos[l], mean_neg[l] + std_neg[l]) * 1.05
            ax.text(l, y_top, sig_stars(pvalues[l]).strip(), ha="center",
                    va="bottom", fontsize=7, color="black")

    for cl in CARRY_LAYERS:
        ax.axvline(cl, color="red", lw=0.8, ls="--", alpha=0.4)

    ax.set_ylabel("Mean active features / position", fontsize=10)
    ax.set_title(
        f"Transcoder sparsity per layer — {args.concept}, template={args.template}, "
        f"n={len(pairs)} pairs each",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", color="#E0E0E0", lw=0.5)

    # Bottom panel: pos - neg difference with significance shading
    diff = mean_pos - mean_neg
    colors = [violet if d > 0 else teal for d in diff]
    ax2.bar(layers, diff, color=colors, alpha=0.75)
    for l in range(n_layers):
        if sig[l]:
            ax2.bar(l, diff[l], color=colors[l], alpha=1.0, edgecolor="black", linewidth=0.8)
    ax2.axhline(0, color=gray, lw=0.8)
    for cl in CARRY_LAYERS:
        ax2.axvline(cl, color="red", lw=0.8, ls="--", alpha=0.4)
        ax2.text(cl + 0.15, ax2.get_ylim()[0] * 0.9 if diff[cl] < 0 else ax2.get_ylim()[1] * 0.9,
                 f"L{cl}", color="red", fontsize=7, va="center")

    ax2.set_xlabel("Layer", fontsize=10)
    ax2.set_ylabel("pos − neg", fontsize=10)
    ax2.grid(axis="y", color="#E0E0E0", lw=0.5)
    ax2.set_xticks(layers[::2])

    fig.tight_layout()

    out_dir = Path(args.out_dir or f"runs/concept_localization/{args.concept}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sparsity_per_layer_{args.template}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved → {out_path}")

    # Print summary table
    print(f"\n{'Layer':>6} {'pos mean':>10} {'neg mean':>10} {'diff':>8} {'p-value':>10} sig")
    print("-" * 56)
    for l in range(n_layers):
        marker = " <--" if l in CARRY_LAYERS else ""
        print(
            f"{l:>6} {mean_pos[l]:>10.1f} {mean_neg[l]:>10.1f} "
            f"{diff[l]:>+8.1f} {pvalues[l]:>10.4f} {sig_stars(pvalues[l])}{marker}"
        )


if __name__ == "__main__":
    main()
