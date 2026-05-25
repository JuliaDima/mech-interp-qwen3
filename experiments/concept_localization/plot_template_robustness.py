"""Template-robustness norm-trajectory plots.

One output: concept_per_template.pdf
  Small-multiples grid, 4 concepts per row. Each panel shows the normalised
  delta-norm trajectory for templates T0/T1/T2 (thin dashed) plus the
  cross-template mean (bold) with a ±1 std band.

Run:
    python -m experiments.concept_localization.plot_template_robustness
Loads deltas.pt which stores {key: {layer: tensor}} for keys
"all", "T0", "T1", "T2".
"""

from __future__ import annotations

import json
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.plot_style import (
    GRAY,
    MAUVE,
    NAVY,
    TEAL,
    VIOLET,
    apply,
)

_TEMPLATE_COLORS = {"T0": NAVY, "T1": TEAL, "T2": MAUVE}

_GROUP_STYLE = {
    "modular":  {"fc": "#F5E6D0", "ec": "#D4823A", "tc": "#7A3A00", "label": "modular"},
    "logical":  {"fc": "#E8E4F0", "ec": "#6C5B8E", "tc": "#3D2E6B", "label": "logical / state"},
    "physical": {"fc": "#DFF0EC", "ec": "#1A7A6E", "tc": "#0D4A42", "label": "physical / linear"},
}


apply()

BASE = Path("runs/concept_localization")
N_LAYERS = 36
LAYERS = np.arange(N_LAYERS)
TMPL_KEYS = ["T0", "T1", "T2"]

CONCEPTS = [
    # (key,                    display label,               group,       color,     ls,   lw)
    # — modular arithmetic —
    ("carry",                  "Carry (mod 10)",            "modular",   VIOLET,    "-",  2.2),
    ("residue_class",          "Residue class",             "modular",   "#C0444A", "-",  2.0),
    ("gcd",                    "GCD divisibility",          "modular",   "#D4823A", "-",  2.0),
    ("perfect_square",         "Perfect square",            "modular",   "#9B59B6", "-",  2.0),
    ("decimal_termination",    "Decimal termination",       "modular",   "#E67E22", "-",  2.0),
    # — logical / structural —
    ("transitive_ordering",    "Transitive ordering",       "logical",   NAVY,      "--", 2.0),
    ("negation_scope",         "Negation scope",            "logical",   MAUVE,     "--", 2.0),
    ("causal_direction",       "Causal direction",          "logical",   GRAY,      "--", 2.0),
    ("balanced_parentheses",   "Balanced parentheses",      "logical",   "#1A7A6E", "--", 2.0),
    ("syllogism",              "Syllogism",                 "logical",   "#6C5B8E", "--", 2.0),
    # — physical / continuous —
    ("conservation",           "Energy conservation",       "physical",  TEAL,      ":",  2.0),
    ("momentum_conservation",  "Momentum conservation",     "physical",  "#2E86AB", ":",  2.0),
    ("doppler_shift",          "Doppler shift",             "physical",  "#3B7A57", ":",  2.0),
    ("wave_interference",      "Wave interference",         "physical",  "#A0522D", ":",  2.0),
    ("geometric_series",       "Geometric series",          "physical",  "#8B6914", ":",  2.0),
    ("triangle_inequality",    "Triangle inequality",       "physical",  "#5C7A3E", ":",  2.0),
    ("dot_product_sign",       "Dot product sign",          "physical",  "#7B5EA7", ":",  2.0),
]


def load_concept_data(
    subdir: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str] | None:
    """Return (raw_norms, act_norms, anchor_token) or None if data missing.

    raw_norms[key][l]  = ‖δ_l‖
    act_norms[key][l]  = ‖δ_l‖ / E[‖h_l‖]   (activation-normalised)
    """
    deltas_path = BASE / subdir / "deltas.pt"
    if not deltas_path.exists():
        return None
    data = torch.load(deltas_path, map_location="cpu")

    mean_act_norm: dict[int, float] = {}
    anchor_tok = "?"
    results_path = BASE / subdir / "results.json"
    if results_path.exists():
        res = json.loads(results_path.read_text())
        cfg = res.get("config", {})
        anchor_tok = cfg.get("anchor_token", cfg.get("anchor_mode", "?"))
        mean_act_norm = {int(k): float(v) for k, v in res.get("mean_act_norm", {}).items()}

    raw_norms: dict[str, np.ndarray] = {}
    act_norms: dict[str, np.ndarray] = {}
    for key, layer_dict in data.items():
        arr_raw = np.zeros(N_LAYERS)
        arr_act = np.zeros(N_LAYERS)
        for layer, vec in layer_dict.items():
            l = int(layer)
            raw = float(vec.float().norm())
            arr_raw[l] = raw
            scale = mean_act_norm.get(l, 1.0)
            arr_act[l] = raw / scale if scale > 1e-8 else raw
        raw_norms[key] = arr_raw
        act_norms[key] = arr_act

    return raw_norms, act_norms, anchor_tok


def norm_array(arr: np.ndarray) -> np.ndarray:
    mx = arr.max()
    return arr / mx if mx > 1e-8 else arr


concept_data: dict[str, dict[str, np.ndarray]] = {}
concept_act_data: dict[str, dict[str, np.ndarray]] = {}
concept_anchor: dict[str, str] = {}
for key, label, group, color, ls, lw in CONCEPTS:
    result = load_concept_data(key)
    if result is None:
        print(f"  [warn] {key}: no deltas.pt, skipping")
        continue
    concept_data[key], concept_act_data[key], concept_anchor[key] = result

available = [
    (key, label, group, color, ls, lw)
    for key, label, group, color, ls, lw in CONCEPTS
    if key in concept_data
]

NCOLS = 4
nrows = int(np.ceil(len(available) / NCOLS))
fig, axes = plt.subplots(
    nrows, NCOLS,
    figsize=(NCOLS * 3.6, nrows * 3.2),
    gridspec_kw={"hspace": 0.6, "wspace": 0.35},
    squeeze=False,
)
axes_flat = axes.flatten()

_ACT_COLOR = "#27ae60"

for idx, (key, label, group, color, ls, lw) in enumerate(available):
    ax = axes_flat[idx]
    norms = concept_data[key]
    act_norms = concept_act_data[key]

    tmpl_keys_present = [t for t in TMPL_KEYS if t in norms]
    tmpl_arrays = [norms[t] for t in tmpl_keys_present]
    tmpl_normed = [norm_array(a) for a in tmpl_arrays]

    for t, a_norm in zip(tmpl_keys_present, tmpl_normed, strict=False):
        ax.plot(
            LAYERS, a_norm,
            color=_TEMPLATE_COLORS.get(t, GRAY),
            linestyle="--", linewidth=0.85, alpha=0.65, label=t,
        )

    mean_norm = np.stack(tmpl_normed).mean(0)
    std_norm = np.stack(tmpl_normed).std(0)
    ax.plot(LAYERS, mean_norm, color=color, linestyle="-", linewidth=2.0, label=r"$\|\delta_l\| / \max_l(\|\delta_l\|)$", zorder=5)
    ax.fill_between(
        LAYERS, mean_norm - std_norm, mean_norm + std_norm,
        color=color, alpha=0.18, linewidth=0,
    )

    # Activation-normalised mean: ‖δ‖ / E[‖h‖] / max
    act_arrays = [act_norms[t] for t in tmpl_keys_present if t in act_norms]
    if act_arrays:
        act_normed = [norm_array(a) for a in act_arrays]
        act_mean = np.stack(act_normed).mean(0)
        ax.plot(LAYERS, act_mean, color=_ACT_COLOR, linestyle="--",
                linewidth=1.6, label=r"$(\|\delta_l\| / \mathbb{E}\|\mathbf{h}_l\|) / \max_l(\|\delta_l\| / \mathbb{E}\|\mathbf{h}_l\|)$", zorder=6)

    ax.set_title(label, fontsize=8.5, pad=4)
    ax.set_xlabel("Layer", fontsize=7.5)
    ax.set_ylabel("Normalised to [0, 1]", fontsize=7.5)
    ax.set_xlim(-0.5, 35.5)
    ax.set_xticks(range(0, 36, 9))
    ax.set_ylim(-0.05, 1.15)
    ax.tick_params(labelsize=7)

    gs = _GROUP_STYLE[group]
    ax.text(
        0.99, 0.97, gs["label"],
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=5.8, color=gs["tc"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor=gs["fc"],
                  edgecolor=gs["ec"], linewidth=0.6),
    )

    anchor_tok = concept_anchor.get(key, "?")
    ax.text(
        0.01, 0.97, f'anchor: "{anchor_tok}"',
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=5.8, color=GRAY,
        family="monospace",
    )

    if idx == 0:
        legend_handles = [
            Line2D([0], [0], color=_TEMPLATE_COLORS[t], linestyle="--", linewidth=0.85, label=t)
            for t in tmpl_keys_present
        ] + [
            Line2D([0], [0], color=color, linestyle="-", linewidth=2.0, label=r"$\|\delta_l\| / \max_l(\|\delta_l\|)$"),
            Line2D([0], [0], color=_ACT_COLOR, linestyle="--", linewidth=1.6, label=r"$(\|\delta_l\| / \mathbb{E}\|\mathbf{h}_l\|) / \max_l(\|\delta_l\| / \mathbb{E}\|\mathbf{h}_l\|)$"),
        ]
        ax.legend(handles=legend_handles, fontsize=6.5, loc="upper left",
                  framealpha=0.85, edgecolor="#cccccc")

for ax in axes_flat[len(available):]:
    ax.set_visible(False)

fig.suptitle(
    "Template robustness: per-concept normalised delta-norm trajectories (±1 std)",
    fontsize=11,
    y=1.01,
)

out = BASE / "concept_per_template.pdf"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
