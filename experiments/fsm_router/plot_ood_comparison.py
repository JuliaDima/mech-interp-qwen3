"""In-distribution vs OOD steering comparison plot."""

import json
import sys

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import GRAY, NAVY, PHASE_BOUNDS, PHASE_LABELS, VIOLET, apply, phase_vlines

apply()

IN_PATH = Path("runs/fsm_router/steer_results_in.json")
OOD_PATH = Path("runs/fsm_router/steer_results_ood.json")
OUT_DIR = Path("runs/fsm_router")

ALPHA = 5.0

# ── load ──────────────────────────────────────────────────────────────────────
with open(IN_PATH) as f:
    in_records = json.load(f)

in_alpha = [r for r in in_records if r["alpha"] == ALPHA]
in_layers = sorted(set(r["layer"] for r in in_alpha))
in_delta_full = {r["layer"]: r["delta_full_acc"] for r in in_alpha}
in_delta_first = {r["layer"]: r["delta_first_token_acc"] for r in in_alpha}
in_base_full = in_records[0]["full_acc"] - in_records[0]["delta_full_acc"]
in_base_first = in_records[0]["first_token_acc"] - in_records[0]["delta_first_token_acc"]

with open(OOD_PATH) as f:
    ood_records = json.load(f)

ood_layers = sorted(set(r["layer"] for r in ood_records))
ood_delta_full = {r["layer"]: r["delta_full_acc"] for r in ood_records}
ood_delta_first = {r["layer"]: r["delta_first_token_acc"] for r in ood_records}
ood_base_full = ood_records[0]["full_acc"] - ood_records[0]["delta_full_acc"]
ood_base_first = ood_records[0]["first_token_acc"] - ood_records[0]["delta_first_token_acc"]

best_in_l = max(in_delta_full, key=in_delta_full.get)
best_ood_l = max(ood_delta_full, key=ood_delta_full.get)

# ── figure: two panels ────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(11, 7.5),
    sharex=True,
    gridspec_kw={"hspace": 0.07},
)


def _decorate(ax):
    phase_vlines(ax)
    ax.axhline(0, color=GRAY, linewidth=0.7, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── panel 1: delta full-answer accuracy ───────────────────────────────────────
in_df = [in_delta_full[l] for l in in_layers]
ood_df = [ood_delta_full[l] for l in ood_layers]

ax1.fill_between(in_layers, in_df, alpha=0.10, color=NAVY)
ax1.plot(
    in_layers,
    in_df,
    color=NAVY,
    linewidth=2.2,
    label=f"In-distribution (0–999),  baseline {in_base_full:.1f}%",
)
ax1.plot(
    ood_layers,
    ood_df,
    color=VIOLET,
    linewidth=2.0,
    linestyle="--",
    marker="o",
    markersize=5,
    markerfacecolor="white",
    markeredgewidth=1.4,
    label=f"OOD (1,000–9,999),  baseline {ood_base_full:.1f}%",
)

# peak verticals
ax1.axvline(best_in_l, color=NAVY, linewidth=1.0, linestyle=":", alpha=0.7)
ax1.axvline(best_ood_l, color=VIOLET, linewidth=1.0, linestyle=":", alpha=0.7)

# peak annotations
ax1.annotate(
    f"L={best_in_l}\n$+${in_delta_full[best_in_l]:.1f} pp",
    xy=(best_in_l, in_delta_full[best_in_l]),
    xytext=(best_in_l - 6, in_delta_full[best_in_l] - 9),
    fontsize=8.5,
    color=NAVY,
    ha="right",
    arrowprops=dict(arrowstyle="->", lw=0.9, color=NAVY),
)
ax1.annotate(
    f"L={best_ood_l}\n$+${ood_delta_full[best_ood_l]:.1f} pp",
    xy=(best_ood_l, ood_delta_full[best_ood_l]),
    xytext=(best_ood_l + 2.5, ood_delta_full[best_ood_l] + 4),
    fontsize=8.5,
    color=VIOLET,
    ha="left",
    arrowprops=dict(arrowstyle="->", lw=0.9, color=VIOLET),
)

_decorate(ax1)
ax1.set_ylabel("Δ full-answer accuracy (pp)", fontsize=10)
ax1.legend(fontsize=9, loc="upper right")
ax1.set_title(
    "Steering generalisation: in-distribution vs OOD  (α = 5.0)",
    fontsize=11,
    pad=22,
)

# phase labels
for (lo, hi), name in zip(PHASE_BOUNDS, PHASE_LABELS, strict=False):
    ax1.text(
        (lo + hi) / 2,
        1.01,
        name,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#666666",
        style="italic",
        transform=ax1.get_xaxis_transform(),
    )

# ── panel 2: absolute accuracy ────────────────────────────────────────────────
in_abs = [in_base_full + in_delta_full[l] for l in in_layers]
ood_abs = [ood_base_full + ood_delta_full[l] for l in ood_layers]

ax2.fill_between(in_layers, in_abs, in_base_full, alpha=0.10, color=NAVY)
ax2.plot(in_layers, in_abs, color=NAVY, linewidth=2.2, label="In-distribution steered")
ax2.plot(
    ood_layers,
    ood_abs,
    color=VIOLET,
    linewidth=2.0,
    linestyle="--",
    marker="o",
    markersize=5,
    markerfacecolor="white",
    markeredgewidth=1.4,
    label="OOD steered",
)

ax2.axhline(
    in_base_full,
    color=NAVY,
    linewidth=1.0,
    linestyle=":",
    alpha=0.55,
    label=f"In-dist baseline ({in_base_full:.1f}%)",
)
ax2.axhline(
    ood_base_full,
    color=VIOLET,
    linewidth=1.0,
    linestyle=":",
    alpha=0.55,
    label=f"OOD baseline ({ood_base_full:.1f}%)",
)

_decorate(ax2)
ax2.set_ylabel("Full-answer accuracy (%)", fontsize=10)
ax2.set_xlabel("Injection layer", fontsize=10)
ax2.set_xticks(range(0, 36, 2))
ax2.legend(fontsize=9, loc="upper right", ncol=2)

fig.savefig(OUT_DIR / "ood_comparison.pdf")
plt.close(fig)
print(f"Saved {OUT_DIR / 'ood_comparison.pdf'}")

print(
    f"\nIn-dist:  baseline={in_base_full:.1f}%  peak L={best_in_l}  Δ={in_delta_full[best_in_l]:+.1f} pp"
)
print(
    f"OOD:      baseline={ood_base_full:.1f}%  peak L={best_ood_l}  Δ={ood_delta_full[best_ood_l]:+.1f} pp"
)
print(f"Peak shift: L={best_in_l} → L={best_ood_l} (+{best_ood_l - best_in_l} layers)")
