"""Visualise steering evaluation results from steer_results_in.json."""

import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import GRAY, MAUVE, NAVY, TEAL, VIOLET, apply, phase_vlines, shade_phases

apply()

IN_PATH = Path("runs/fsm_router/steer_results_in.json")
OUT_DIR = Path("runs/fsm_router")

with open(IN_PATH) as f:
    records = json.load(f)

baseline_first = records[0]["first_token_acc"] - records[0]["delta_first_token_acc"]
baseline_full = records[0]["full_acc"] - records[0]["delta_full_acc"]
baseline_digit = records[0]["digit_acc"] - records[0]["delta_digit_acc"]

layers = sorted(set(r["layer"] for r in records))
alphas = sorted(set(r["alpha"] for r in records))
n_layers, n_alphas = len(layers), len(alphas)
layer_idx = {l: i for i, l in enumerate(layers)}
alpha_idx = {a: i for i, a in enumerate(alphas)}


def _matrix(metric):
    m = np.full((n_layers, n_alphas), np.nan)
    for r in records:
        m[layer_idx[r["layer"]], alpha_idx[r["alpha"]]] = r[metric]
    return m


delta_full = _matrix("delta_full_acc")
delta_first = _matrix("delta_first_token_acc")
delta_digit = _matrix("delta_digit_acc")

best_idx = np.unravel_index(np.nanargmax(delta_full), delta_full.shape)
best_layer = layers[best_idx[0]]
best_alpha = alphas[best_idx[1]]
best_delta = delta_full[best_idx]

# ── Figure 1: heatmap (layer × alpha) ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
titles = ["Δ full-answer accuracy (pp)", "Δ first-token accuracy (pp)", "Δ digit accuracy (pp)"]
matrices = [delta_full, delta_first, delta_digit]

for ax, mat, title in zip(axes, matrices, titles, strict=False):
    vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)), 1.0)
    im = ax.imshow(
        mat,
        aspect="auto",
        cmap="RdYlGn",
        norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        origin="upper",
    )
    ax.set_xticks(range(n_alphas))
    ax.set_xticklabels([f"α={a}" for a in alphas], fontsize=9)
    ax.set_yticks(range(0, n_layers, 5))
    ax.set_yticklabels([str(layers[i]) for i in range(0, n_layers, 5)], fontsize=8)
    ax.set_xlabel("Steering amplitude α", fontsize=9)
    ax.set_ylabel("Injection layer", fontsize=9)
    ax.set_title(title)
    ax.grid(False)

    if title.startswith("Δ full"):
        ax.add_patch(
            plt.Rectangle(
                (best_idx[1] - 0.5, best_idx[0] - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#222222",
                linewidth=2.2,
            )
        )
        ax.text(
            best_idx[1],
            best_idx[0],
            f"+{best_delta:.1f}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#111111",
        )

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Δ accuracy (pp)", fontsize=8)

fig.suptitle(
    f"Local FSM steering sweep — in-distribution\n"
    f"baseline:  full = {baseline_full:.1f}%,  first-token = {baseline_first:.1f}%,  digit = {baseline_digit:.1f}%",
    fontsize=11,
    y=1.02,
)
fig.savefig(OUT_DIR / "steer_heatmap.pdf")
plt.close(fig)
print("Saved steer_heatmap.pdf")

# ── Figure 2: line plots per alpha ────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
metric_triples = [
    (delta_full, "Δ full-answer accuracy (pp)"),
    (delta_first, "Δ first-token accuracy (pp)"),
    (delta_digit, "Δ digit accuracy (pp)"),
]
alpha_colors = [NAVY, VIOLET, TEAL, MAUVE]

for ax, (mat, label) in zip(axes2, metric_triples, strict=False):
    for ai, (alpha, color) in enumerate(zip(alphas, alpha_colors, strict=False)):
        lw = 2.2 if alpha == best_alpha else 1.3
        ls = "-" if alpha == best_alpha else "--"
        ax.plot(
            layers,
            mat[:, ai],
            color=color,
            linewidth=lw,
            linestyle=ls,
            label=f"α = {alpha}",
            alpha=0.9,
        )
    ax.axhline(0, color=GRAY, linewidth=0.8, linestyle=":")
    shade_phases(ax)
    phase_vlines(ax)
    ax.set_xlabel("Injection layer", fontsize=9)
    ax.set_ylabel(label, fontsize=9)
    ax.set_title(label)
    ax.set_xticks(range(0, n_layers, 5))
    ax.legend(fontsize=8, loc="upper right")

    if "full" in label:
        ax.axvline(
            best_layer, color="#333333", linewidth=1.4, linestyle=":", label=f"best L={best_layer}"
        )
        ax.annotate(
            f"L={best_layer}, α={best_alpha}\n+{best_delta:.1f} pp",
            xy=(best_layer, best_delta),
            xytext=(best_layer + 2.5, best_delta - 6),
            fontsize=8,
            color=GRAY,
            arrowprops=dict(arrowstyle="->", lw=1.0, color=GRAY),
        )

fig2.suptitle("Steering improvement vs injection layer — in-distribution, local mode", fontsize=11)
fig2.savefig(OUT_DIR / "steer_line.pdf")
plt.close(fig2)
print("Saved steer_line.pdf")

# ── Figure 3: absolute accuracy at best alpha ─────────────────────────────────
best_ai = alpha_idx[best_alpha]
full_abs = baseline_full + delta_full[:, best_ai]
first_abs = baseline_first + delta_first[:, best_ai]
digit_abs = baseline_digit + delta_digit[:, best_ai]

fig3, ax3 = plt.subplots(figsize=(11, 4.5))
ax3.plot(layers, full_abs, color=NAVY, linewidth=2.2, label="Full-answer")
ax3.plot(layers, first_abs, color=VIOLET, linewidth=2.2, label="First-token")
ax3.plot(layers, digit_abs, color=TEAL, linewidth=1.8, linestyle="--", label="Digit")
ax3.axhline(baseline_full, color=NAVY, linewidth=0.9, linestyle=":", alpha=0.5)
ax3.axhline(baseline_first, color=VIOLET, linewidth=0.9, linestyle=":", alpha=0.5)
ax3.axvline(
    best_layer, color=GRAY, linewidth=1.4, linestyle=":", label=f"Best layer (L={best_layer})"
)
shade_phases(ax3)
phase_vlines(ax3)
ax3.set_xlabel("Injection layer", fontsize=10)
ax3.set_ylabel("Accuracy (%)", fontsize=10)
ax3.set_title(
    f"Absolute accuracy at α = {best_alpha} (best amplitude) — in-distribution\n"
    "dotted lines show unsteered baseline",
)
ax3.legend(fontsize=9)
ax3.set_xticks(range(0, n_layers, 2))

fig3.savefig(OUT_DIR / "steer_best_alpha.pdf")
plt.close(fig3)
print("Saved steer_best_alpha.pdf")

print(f"\nBest:  layer={best_layer}  α={best_alpha}  Δfull={best_delta:+.1f} pp")
print(
    f"Base:  full={baseline_full:.1f}%  first-token={baseline_first:.1f}%  digit={baseline_digit:.1f}%"
)
