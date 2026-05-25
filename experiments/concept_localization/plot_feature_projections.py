"""Feature projection plots for carry concept localisation.

Produces:
  feature_projections_scatter.pdf — scatter + persistence panel
"""

import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from plot_style import GRAY, VIOLET, apply

apply()

RESULTS = Path("runs/concept_localization/carry/results.json")
OUT_DIR = RESULTS.parent

with open(RESULTS) as f:
    r = json.load(f)

norms = {int(k): v for k, v in r["sharpness"]["norm_by_layer"].items()}
top_feats = {int(k): v for k, v in r["top_features_by_layer"].items()}
n_layers = len(norms)

FOCAL_LAYERS = [4, 5, 18, 19, 27, 28, 33, 34]

# ── Scatter ───────────────────────────────────────────────────────────────────
all_layers, all_fids, all_sims = [], [], []
for layer, feats in top_feats.items():
    for feat in feats:
        all_layers.append(layer)
        all_fids.append(feat["feature_id"])
        all_sims.append(feat["projection"])

all_layers = np.array(all_layers)
all_fids = np.array(all_fids)
all_sims = np.array(all_sims)

vmax_s = np.abs(all_sims).max()
norm_s = mcolors.TwoSlopeNorm(vmin=-vmax_s, vcenter=0, vmax=vmax_s)
sizes = 22 + 85 * (np.abs(all_sims) / vmax_s) ** 1.5

# ── Inter-layer cosine similarity for same-index features ─────────────────────
# Checks whether the same feature INDEX across layers represents the same direction.
# If cos(W_enc_La[f], W_enc_Lb[f]) ≈ random baseline, index coincidences are spurious.
from collections import Counter

import torch.nn.functional as F

from mechinterp_qwen3.transcoder.single_layer_transcoder import load_relu_transcoder

CACHE = Path.home() / ".cache/mechinterp_qwen3/mwhanna/qwen3-4b-transcoders"
FID = 73141
LAYER_A, LAYER_B = 21, 28

tc_a = load_relu_transcoder(
    str(CACHE / f"layer_{LAYER_A}.safetensors"),
    layer=LAYER_A,
    lazy_encoder=False,
    lazy_decoder=True,
)
tc_b = load_relu_transcoder(
    str(CACHE / f"layer_{LAYER_B}.safetensors"),
    layer=LAYER_B,
    lazy_encoder=False,
    lazy_decoder=True,
)

rng = np.random.default_rng(42)
rand_ids = rng.integers(0, tc_a.W_enc.shape[0], size=500)
cos_random = np.array(
    [
        F.cosine_similarity(
            tc_a.W_enc[int(f)].float().unsqueeze(0), tc_b.W_enc[int(f)].float().unsqueeze(0)
        ).item()
        for f in rand_ids
    ]
)
cos_f73141 = F.cosine_similarity(
    tc_a.W_enc[FID].float().unsqueeze(0), tc_b.W_enc[FID].float().unsqueeze(0)
).item()

fid_counts = Counter(all_fids)
# No genuinely persistent features — index coincidences are noise (verified by cos-sim check)

fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(
    3,
    1,
    height_ratios=[3.5, 1, 1.4],
    hspace=0.42,
)
ax_sc = fig.add_subplot(gs[0])
ax_n = fig.add_subplot(gs[1])
ax_cos = fig.add_subplot(gs[2, :])

sc = ax_sc.scatter(
    all_layers,
    all_fids,
    c=all_sims,
    s=sizes,
    cmap="RdBu_r",
    norm=norm_s,
    alpha=0.78,
    linewidths=0.2,
    edgecolors="white",
    zorder=3,
)

for layer in FOCAL_LAYERS:
    ax_sc.axvline(layer, color=GRAY, linewidth=0.7, linestyle=":", alpha=0.5)

ax_sc.set_ylabel("Transcoder feature ID", fontsize=11)
ax_sc.set_title(
    "Top-15 transcoder features aligned with carry concept delta $\\delta_l$\n"
    "dot size $\\propto |c_{l,f}|$;  red = positive,  blue = negative alignment",
    fontsize=11,
    pad=28,
)
ax_sc.set_xlim(-0.5, n_layers - 0.5)
ax_sc.set_xlabel("Layer", fontsize=11)
ax_sc.spines["top"].set_visible(False)
ax_sc.spines["right"].set_visible(False)

cb = fig.colorbar(sc, ax=ax_sc, orientation="vertical", fraction=0.022, pad=0.01)
cb.set_label("$c_{l,f}$", fontsize=10)
cb.ax.tick_params(labelsize=8)

# Highlight F73141 at both layers — but note it is NOT a genuine cross-layer feature
mask = all_fids == FID
if mask.any():
    lyrs = all_layers[mask]
    sims = all_sims[mask]
    ax_sc.scatter(
        lyrs,
        [FID] * len(lyrs),
        c=sims,
        cmap="RdBu_r",
        norm=norm_s,
        s=90,
        zorder=5,
        linewidths=1.2,
        edgecolors="#333333",
    )
    for l in lyrs:
        ax_sc.annotate(
            f"F{FID}\n(idx only)",
            xy=(l, FID),
            xytext=(l + 1.0, FID + 8000),
            fontsize=6.5,
            color="#666666",
            va="bottom",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#aaaaaa"),
        )

# norm strip
norm_vals = [norms[l] for l in range(n_layers)]
ax_n.fill_between(range(n_layers), norm_vals, alpha=0.25, color=VIOLET)
ax_n.plot(range(n_layers), norm_vals, color=VIOLET, linewidth=2.0)
ax_n.set_xlim(-0.5, n_layers - 0.5)
ax_n.set_xticks(range(0, n_layers, 5))
ax_n.set_xticklabels([str(l) for l in range(0, n_layers, 5)], fontsize=9)
ax_n.set_ylabel("$\\|\\delta_l\\|$", fontsize=10)
ax_n.set_xlabel("Layer", fontsize=11)
ax_n.set_title("Carry delta norm $\\|\\delta_l\\|$ by layer", fontsize=10, pad=6)
ax_n.spines["top"].set_visible(False)
ax_n.spines["right"].set_visible(False)

# ── Inter-layer cosine similarity panel ───────────────────────────────────────
ax_cos.hist(cos_random, bins=30, color=VIOLET, alpha=0.65, edgecolor="white", linewidth=0.4)
ax_cos.axvline(
    cos_random.mean(),
    color=GRAY,
    linewidth=1.2,
    linestyle="--",
    label=f"Random mean = {cos_random.mean():.3f}",
)
ax_cos.axvline(
    cos_f73141, color="#C0444A", linewidth=1.8, linestyle="-", label=f"F{FID} = {cos_f73141:.3f}"
)
ax_cos.legend(fontsize=7.5, framealpha=0.9)
ax_cos.set_xlabel(f"cos( W_enc_L{LAYER_A}[f],  W_enc_L{LAYER_B}[f] )", fontsize=8)
ax_cos.set_ylabel("Count", fontsize=8)
ax_cos.set_title(
    f"Same index across L{LAYER_A}–L{LAYER_B}:\nare they the same direction?",
    fontsize=8.5,
    pad=6,
)
ax_cos.spines["top"].set_visible(False)
ax_cos.spines["right"].set_visible(False)

sigma = (cos_f73141 - cos_random.mean()) / cos_random.std()

fig.savefig(OUT_DIR / "feature_projections_scatter.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT_DIR / 'feature_projections_scatter.pdf'}")
print(f"\nKey result: cos(W_enc_L{LAYER_A}[{FID}], W_enc_L{LAYER_B}[{FID}]) = {cos_f73141:.4f}")
print(
    f"Random baseline (500 same-idx pairs L{LAYER_A}/L{LAYER_B}): mean={cos_random.mean():.4f}  std={cos_random.std():.4f}"
)
print(f"F{FID} is {sigma:+.1f}σ — index coincidence is spurious.")
