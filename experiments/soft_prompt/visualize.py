"""
Visualization suite for the soft-prompt experiment.

Produces six figures saved to <out_root>/figures/:
  fig1_prompt_templates.png   — prompt format & example prompts
  fig2_digit_coverage.png     — dataset (a,b) coverage scatter
  fig3_correct_grid.png       — wrong→correct transitions per (a mod 10, b mod 10)
  fig4_steering.png           — cos(Δ,sv_base) vs cos(Δ,sv_pfx) by layer + delta norms
  fig5_prefix_vectors.png     — learned prefix: norms, cosine matrix, PCA
  fig6_architecture.png       — soft-prompt architecture diagram
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update(
    {
        "font.family": "monospace",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ── colour palette ─────────────────────────────────────────────────────────────
C_BASE = "#4878D0"  # blue
C_PFX = "#EE854A"  # orange
C_WRONG = "#D65F5F"  # red
C_CORR = "#6ACC65"  # green
C_BOTH = "#C8A8E9"  # purple
C_GRAY = "#BBBBBB"
C_DARK = "#333333"


# ── helpers ────────────────────────────────────────────────────────────────────


def savefig(fig: plt.Figure, path: Path, name: str) -> None:
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 – Prompt templates
# ══════════════════════════════════════════════════════════════════════════════


def fig_prompt_templates(dataset: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    title_y = 0.96
    ax.text(
        0.5,
        title_y,
        "Prompt Template & Example Inputs",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        transform=ax.transAxes,
    )

    # Template box
    template_str = (
        'Template (T0):    "calc: {a}+{b}= "\n'
        "                   └─ literal digits, no chain-of-thought\n\n"
        "Answer format:     First generated token must equal hundreds digit of (a+b)\n"
        "                   (evaluated greedily, first token only)"
    )
    ax.text(
        0.05,
        0.82,
        template_str,
        ha="left",
        va="top",
        fontsize=10,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", fc="#F0F4FF", ec=C_BASE, lw=1.5),
    )

    # Pick 4 diverse examples
    examples = [s for s in dataset if s.get("a") is not None][:4]
    # Try to find carry / no-carry examples
    carry_ex = [s for s in dataset if (s["a"] % 10 + s["b"] % 10) >= 10][:2]
    nocarry_ex = [s for s in dataset if (s["a"] % 10 + s["b"] % 10) < 10][:2]
    examples = (carry_ex + nocarry_ex)[:4]

    col_x = [0.05, 0.55]
    row_y = [0.48, 0.20]
    for i, s in enumerate(examples):
        x = col_x[i % 2]
        y = row_y[i // 2]
        a, b = s["a"], s["b"]
        true_ans = s["true_answer_str"]
        greedy = s.get("greedy_completion_str", "?")
        carry_flag = "carry" if (a % 10 + b % 10) >= 10 else "no carry"
        base_ok = greedy.strip().startswith(true_ans[0]) if greedy else False
        color = C_CORR if base_ok else C_WRONG
        ex_str = (
            f'Input:   "calc: {a}+{b}= "\n'
            f"Target:   {true_ans}    [{carry_flag}]\n"
            f"Baseline: {greedy.strip()[:6] or '?'}    "
            f"[{'✓' if base_ok else '✗'}]"
        )
        ax.text(
            x,
            y,
            ex_str,
            ha="left",
            va="top",
            fontsize=9,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFFDE7", ec=color, lw=1.5),
        )

    ax.text(
        0.5,
        0.04,
        f"Dataset: {len(dataset)} samples   |   a ∈ [0, 999]   |   b ∈ [0, 999]",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#666666",
        transform=ax.transAxes,
    )

    savefig(fig, out, "fig1_prompt_templates.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 – Digit coverage scatter
# ══════════════════════════════════════════════════════════════════════════════


def fig_digit_coverage(dataset: list[dict], samples_path: Path | None, out: Path) -> None:
    a_vals = np.array([s["a"] for s in dataset if s.get("a") is not None])
    b_vals = np.array([s["b"] for s in dataset if s.get("b") is not None])

    # Load per-sample correctness if available
    corr_base = corr_pfx = None
    if samples_path and samples_path.exists():
        with open(samples_path) as f:
            sdata = json.load(f)
        corr_base = np.array([s["base_correct"] for s in sdata])
        corr_pfx = np.array([s["pfx_correct"] for s in sdata])
        eval_a = np.array([s["a"] for s in sdata])
        eval_b = np.array([s["b"] for s in sdata])

    fig, axes = plt.subplots(
        1, 2 if corr_base is not None else 1, figsize=(14 if corr_base is not None else 6, 5)
    )
    if corr_base is None:
        axes = [axes]

    # Left: full dataset scatter
    ax = axes[0]
    ax.scatter(a_vals, b_vals, s=4, alpha=0.3, color=C_BASE, rasterized=True)
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_title(f"Dataset coverage  (n={len(a_vals)})")
    ax.set_xlim(-10, 1010)
    ax.set_ylim(-10, 1010)

    if corr_base is not None:
        ax = axes[1]
        # colour by outcome: both correct / base only / pfx only / both wrong
        both_ok = corr_base & corr_pfx
        pfx_fixed = ~corr_base & corr_pfx  # wrong→correct
        pfx_broke = corr_base & ~corr_pfx  # correct→wrong
        both_bad = ~corr_base & ~corr_pfx

        def _plot(mask, color, label, zorder=1):
            ax.scatter(
                eval_a[mask],
                eval_b[mask],
                s=18,
                alpha=0.8,
                color=color,
                label=label,
                zorder=zorder,
                rasterized=True,
            )

        _plot(both_bad, C_WRONG, "wrong (both)", 1)
        _plot(both_ok, C_GRAY, "correct (both)", 2)
        _plot(pfx_broke, C_BOTH, "prefix broke it", 3)
        _plot(pfx_fixed, C_CORR, "prefix fixed it", 4)

        ax.set_xlabel("a")
        ax.set_ylabel("b")
        ax.set_title("Eval samples: correctness change with prefix")
        ax.set_xlim(-10, 1010)
        ax.set_ylim(-10, 1010)
        ax.legend(loc="upper left", fontsize=8, markerscale=2)

    fig.tight_layout()
    savefig(fig, out, "fig2_digit_coverage.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 – Wrong → correct grid (bucketed by units digit a%10, b%10)
# ══════════════════════════════════════════════════════════════════════════════


def fig_correct_grid(samples_path: Path | None, out: Path) -> None:
    if samples_path is None or not samples_path.exists():
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Per-sample eval data not yet available.\nRe-run: --eval --force",
            ha="center",
            va="center",
            fontsize=12,
            color="#888888",
        )
        savefig(fig, out, "fig3_correct_grid.png")
        return

    with open(samples_path) as f:
        sdata = json.load(f)

    grid_pfx_fixed = np.zeros((10, 10), dtype=int)
    grid_pfx_broke = np.zeros((10, 10), dtype=int)
    grid_both_bad = np.zeros((10, 10), dtype=int)
    grid_both_ok = np.zeros((10, 10), dtype=int)
    grid_total = np.zeros((10, 10), dtype=int)

    for s in sdata:
        if s["a"] is None or s["b"] is None:
            continue
        row = s["a"] % 10
        col = s["b"] % 10
        grid_total[row, col] += 1
        bc, pc = s["base_correct"], s["pfx_correct"]
        if not bc and pc:
            grid_pfx_fixed[row, col] += 1
        elif bc and not pc:
            grid_pfx_broke[row, col] += 1
        elif not bc and not pc:
            grid_both_bad[row, col] += 1
        else:
            grid_both_ok[row, col] += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # y-axis: 0 at bottom, 9 at top (standard math convention).
    # imshow places row-0 at the top, so we flip the data vertically.
    # After flipping: visual row i = data row (9-i) = a%10 = (9-i).
    # y tick label for visual row i → a%10 = 9-i.
    ytick_labels = [str(9 - i) for i in range(10)]
    xtick_labels = [str(i) for i in range(10)]

    def _heatmap(ax, data, title, cmap, vmax=None):
        flipped = data[::-1]  # flip so a%10=0 is at bottom
        im = ax.imshow(flipped, cmap=cmap, aspect="equal", vmin=0, vmax=vmax or data.max() or 1)
        ax.set_xticks(range(10))
        ax.set_xticklabels(xtick_labels, fontsize=8)
        ax.set_yticks(range(10))
        ax.set_yticklabels(ytick_labels, fontsize=8)
        ax.set_xlabel("b mod 10")
        ax.set_ylabel("a mod 10")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
        # annotate cells (i in visual row coords → a%10 = 9-i)
        for i in range(10):
            for j in range(10):
                v = data[9 - i, j]  # data row = 9-visual_row
                if v > 0:
                    ax.text(
                        j,
                        i,
                        str(v),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if v > data.max() * 0.5 else C_DARK,
                    )

    _heatmap(axes[0], grid_pfx_fixed, "prefix fixed  (wrong→correct)", "Greens")
    _heatmap(axes[1], grid_both_bad, "still wrong   (both wrong)", "Reds")

    # Carry boundary as a staircase along cell edges.
    # In flipped visual coords: visual_row k = a%10 = (9-k).
    # Carry starts at col = (10 - a%10) = k+1, so the cell-edge is at col = k+0.5.
    # Path: enter from top at col=0.5, step right by 1 each row.
    xs_stair, ys_stair = [], []
    for step in range(9):  # step = visual_row, a%10 = 9-step  (9 down to 1)
        col = step + 0.5  # b-edge where carry begins for this a
        top_y = step - 0.5 if step > 0 else -0.5
        xs_stair.extend([col, col, col + 1.0])
        ys_stair.extend([top_y, step + 0.5, step + 0.5])
    xs_stair.append(9.5)
    ys_stair.append(8.5)  # exit to the right

    for ax in axes:
        ax.plot(
            xs_stair,
            ys_stair,
            color="gold",
            lw=2,
            label="carry boundary  (upper-right = carry)",
            zorder=5,
        )
        ax.legend(loc="upper left", fontsize=7)

    fig.suptitle(
        "Correctness transitions bucketed by units digit  (a mod 10, b mod 10)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    savefig(fig, out, "fig3_correct_grid.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 – Steering analysis
# ══════════════════════════════════════════════════════════════════════════════


def fig_steering(analysis_path: Path, out: Path) -> None:
    with open(analysis_path) as f:
        data = json.load(f)

    layers = [d["layer"] for d in data]
    delta_norms = [d["delta_norm"] for d in data]
    cos_base = [d["cos_delta_sv_base"] for d in data]
    cos_pfx = [d["cos_delta_sv_pfx"] for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- cos alignment ---
    ax1.plot(
        layers, cos_base, "o-", color=C_BASE, lw=2, ms=6, label="cos(Δ, sv_base)  [14 correct]"
    )
    ax1.plot(layers, cos_pfx, "s-", color=C_PFX, lw=2, ms=6, label="cos(Δ, sv_pfx)   [155 correct]")
    ax1.axhline(0, color=C_GRAY, lw=1, linestyle=":")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Cosine similarity")
    ax1.set_title("Prefix Δ alignment with carry steering vector")
    ax1.set_ylim(-0.1, 0.75)
    ax1.legend(fontsize=9)
    ax1.set_xticks(layers)

    # shaded band for "well-aligned" threshold
    ax1.axhspan(0.5, 0.75, alpha=0.08, color=C_CORR, label="_nolegend_")
    ax1.text(layers[-1], 0.52, "≥0.5", fontsize=8, color=C_CORR, va="bottom", ha="right")

    # --- delta norms ---
    ax2.bar(layers, delta_norms, color=C_BASE, alpha=0.7, width=2.5)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("||Δ||  (L2 norm)")
    ax2.set_title("Mean prefix-induced activation delta norm")
    ax2.set_xticks(layers)

    for layer_, n in zip(layers, delta_norms, strict=False):
        ax2.text(layer_, n + 0.5, f"{n:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Soft-prompt steering analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, out, "fig4_steering.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 – Prefix vectors
# ══════════════════════════════════════════════════════════════════════════════


def fig_prefix_vectors(ckpt_path: Path, out: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    P = ckpt["state_dict"]["prefix"].float().numpy()  # (k, d_model)
    k, d = P.shape

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. L2 norm per prefix token ────────────────────────────────────────
    ax_norm = fig.add_subplot(gs[0, 0])
    norms = np.linalg.norm(P, axis=1)
    ax_norm.bar(range(k), norms, color=C_PFX, alpha=0.85)
    ax_norm.set_xlabel("Prefix token index")
    ax_norm.set_ylabel("L2 norm")
    ax_norm.set_title("Prefix token norms")
    ax_norm.set_xticks(range(k))
    for i, n in enumerate(norms):
        ax_norm.text(i, n + 0.001, f"{n:.2f}", ha="center", va="bottom", fontsize=7)

    # ── 2. Cosine similarity matrix ────────────────────────────────────────
    ax_cos = fig.add_subplot(gs[0, 1])
    P_normed = P / (norms[:, None] + 1e-8)
    cos_mat = P_normed @ P_normed.T
    im = ax_cos.imshow(cos_mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax_cos.set_xticks(range(k))
    ax_cos.set_xticklabels([str(i) for i in range(k)], fontsize=8)
    ax_cos.set_yticks(range(k))
    ax_cos.set_yticklabels([str(i) for i in range(k)], fontsize=8)
    ax_cos.set_title("Cosine similarity between prefix tokens")
    plt.colorbar(im, ax=ax_cos, fraction=0.046)
    for i in range(k):
        for j in range(k):
            ax_cos.text(
                j,
                i,
                f"{cos_mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if abs(cos_mat[i, j]) > 0.6 else C_DARK,
            )

    # ── 3. PCA: 2D projection of prefix tokens ─────────────────────────────
    ax_pca = fig.add_subplot(gs[0, 2])
    # Simple PCA via SVD
    P_c = P - P.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(P_c, full_matrices=False)
    coords = P_c @ Vt[:2].T  # (k, 2)
    sc = ax_pca.scatter(coords[:, 0], coords[:, 1], c=range(k), cmap="plasma", s=80, zorder=3)
    for i, (x, y) in enumerate(coords):
        ax_pca.text(x, y + 0.002, str(i), ha="center", va="bottom", fontsize=8)
    ax_pca.set_xlabel("PC 1")
    ax_pca.set_ylabel("PC 2")
    ax_pca.set_title("PCA of prefix tokens (2D)")
    plt.colorbar(sc, ax=ax_pca, label="token index", fraction=0.046)

    # ── 4. Heatmap of raw prefix values (k × first 128 dims) ──────────────
    ax_heat = fig.add_subplot(gs[1, :])
    show_dims = min(256, d)
    im2 = ax_heat.imshow(
        P[:, :show_dims],
        aspect="auto",
        cmap="RdBu_r",
        vmin=-np.abs(P[:, :show_dims]).max(),
        vmax=np.abs(P[:, :show_dims]).max(),
    )
    ax_heat.set_ylabel("Prefix token")
    ax_heat.set_xlabel(f"Embedding dim (first {show_dims})")
    ax_heat.set_yticks(range(k))
    ax_heat.set_yticklabels([str(i) for i in range(k)], fontsize=8)
    ax_heat.set_title(f"Prefix vector values — first {show_dims} of {d} dims  (d_model={d})")
    plt.colorbar(im2, ax=ax_heat, fraction=0.02, pad=0.01)

    fig.suptitle(
        "Learned prefix vectors  (k=10 tokens, d_model=2560)", fontsize=13, fontweight="bold"
    )
    savefig(fig, out, "fig5_prefix_vectors.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 – Architecture diagram
# ══════════════════════════════════════════════════════════════════════════════


def fig_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_aspect("equal")

    def box(x, y, w, h, fc, ec, label, fontsize=8, bold=False):
        rect = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.5, zorder=2
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            zorder=3,
            wrap=True,
        )

    def arrow(x0, y0, x1, y1, color=C_DARK, style="-|>"):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle=style, color=color, lw=1.5),
            zorder=4,
        )

    # ─── Title ───────────────────────────────────────────────────────────────
    ax.text(
        7,
        7.6,
        "Soft-Prompt Architecture  (embedding-level prefix injection)",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    # ─── Input token sequence ─────────────────────────────────────────────
    prefix_color = "#FFD580"
    real_color = "#C8E6C9"
    tok_y = 0.3
    tok_h = 0.7
    tok_w = 0.82

    # Prefix tokens (k=10 pad tokens)
    for i in range(4):
        box(
            0.3 + i * (tok_w + 0.08),
            tok_y,
            tok_w,
            tok_h,
            prefix_color,
            C_PFX,
            f"PAD\n({i})",
            fontsize=7,
        )
    ax.text(4.0, tok_y + tok_h / 2, "···", ha="center", va="center", fontsize=12)
    # Real input tokens: "calc: 123+456= "
    real_tokens = ["calc:", "1", "2", "3", "+", "4", "5", "6", "=", "▢"]
    for i, tok in enumerate(real_tokens[:8]):
        box(4.4 + i * (tok_w + 0.06), tok_y, tok_w, tok_h, real_color, "#4CAF50", tok, fontsize=7)
    ax.text(11.4, tok_y + tok_h / 2, "···", ha="center", va="center", fontsize=12)

    ax.text(
        2.0,
        tok_y - 0.25,
        "← k=10 prefix (pad IDs)",
        ha="center",
        va="top",
        fontsize=8,
        color=C_PFX,
        style="italic",
    )
    ax.text(
        8.5,
        tok_y - 0.25,
        "← real input tokens",
        ha="center",
        va="top",
        fontsize=8,
        color="#4CAF50",
        style="italic",
    )

    # ─── Embedding layer (hook_embed) ─────────────────────────────────────
    emb_y = 1.7
    box(0.2, emb_y, 13.2, 0.9, "#E3F2FD", C_BASE, "", fontsize=9)
    ax.text(
        6.8,
        emb_y + 0.45,
        "Embedding Layer  (hook_embed)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=C_BASE,
    )

    # Hook annotation
    box(
        9.5,
        emb_y + 0.9,
        3.5,
        0.7,
        "#FFF8E1",
        C_PFX,
        "HOOK: replace prefix embeds\nwith learned vectors P₀…P₉",
        fontsize=7.5,
    )
    ax.annotate(
        "",
        xy=(9.5, emb_y + 0.6),
        xytext=(10.5, emb_y + 0.9),
        arrowprops=dict(arrowstyle="-|>", color=C_PFX, lw=1.5),
        zorder=4,
    )

    # Arrows from tokens to embedding
    for i in range(9):
        ax.annotate(
            "",
            xy=(0.7 + i * 0.9 + 0.4, emb_y),
            xytext=(0.7 + i * 0.9 + 0.4, tok_y + tok_h),
            arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.8),
            zorder=1,
        )

    # ─── Transformer blocks (stacked) ────────────────────────────────────
    block_y_start = 3.1
    block_h = 0.52
    block_gap = 0.08
    n_show = 5
    labels = [
        "Transformer Block 0",
        "Transformer Block 1",
        "···",
        "Transformer Block 27",
        "Transformer Block 35",
    ]
    colors = ["#E8EAF6", "#E8EAF6", "#FFFFFF", "#E8EAF6", "#E8EAF6"]
    for i, (label, color) in enumerate(zip(labels, colors, strict=False)):
        y = block_y_start + i * (block_h + block_gap)
        if label == "···":
            ax.text(7, y + block_h / 2 + 0.05, label, ha="center", va="center", fontsize=14)
        else:
            box(0.2, y, 13.2, block_h, color, "#9FA8DA", label, fontsize=9)
        if i < n_show - 1 and label != "···":
            next_label = labels[i + 1]
            if next_label != "···":
                arrow(7, y + block_h, 7, y + block_h + block_gap + 0.01, color=C_GRAY)

    # Arrows: embedding → block 0
    arrow(7, emb_y + 0.9, 7, block_y_start - 0.05, color=C_BASE)

    # ─── Output logits ────────────────────────────────────────────────────
    out_y = block_y_start + n_show * (block_h + block_gap) + 0.1
    box(
        3.5,
        out_y,
        6.6,
        0.75,
        "#F3E5F5",
        "#9C27B0",
        "Logits at last real position  →  argmax  →  predicted digit",
        fontsize=9,
        bold=False,
    )
    arrow(7, out_y - 0.1, 7, out_y, color="#9C27B0")

    # ─── Learnable P legend ───────────────────────────────────────────────
    box(10.8, 3.2, 2.9, 2.2, "#FFFDE7", C_PFX, "", fontsize=8)
    ax.text(
        12.25,
        5.2,
        "Learned prefix P",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=C_PFX,
    )
    lines = [
        "shape: (k=10, d=2560)",
        "params: 25,600",
        "init: N(0, 0.02)",
        "opt:  Adam lr=3e-4",
        "only prefix is trained;",
        "model weights frozen",
    ]
    for j, line in enumerate(lines):
        ax.text(11.0, 5.0 - j * 0.28, line, ha="left", va="center", fontsize=7.5, color=C_DARK)

    savefig(fig, out, "fig6_architecture.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 – Prefix nearest-neighbour tokens
# ══════════════════════════════════════════════════════════════════════════════


def _load_embedding_matrix(model_name: str = "Qwen/Qwen3-4B") -> tuple[np.ndarray, list[str]]:
    """Load only the embedding weight and tokenizer — no GPU needed."""
    import os

    from transformers import AutoTokenizer

    # Try loading just the embed_tokens weight via safetensors (much lighter)
    cache_root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    # Find safetensors shards in cache
    import glob as _glob

    pattern = os.path.join(cache_root, "**", "model*.safetensors")
    shards = sorted(_glob.glob(pattern, recursive=True))

    W_E = None
    for shard in shards:
        try:
            from safetensors import safe_open

            with safe_open(shard, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                embed_key = next((k for k in keys if "embed_tokens.weight" in k), None)
                if embed_key:
                    W_E = f.get_tensor(embed_key).float().numpy()
                    print(f"  Loaded embed matrix from {shard}  shape={W_E.shape}")
                    break
        except Exception:
            continue

    if W_E is None:
        raise RuntimeError("Could not find embed_tokens.weight in HF cache safetensors shards.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return W_E, tokenizer


def fig_prefix_nn(
    ckpt_path: Path, out: Path, model_name: str = "Qwen/Qwen3-4B", top_k: int = 8
) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    P = ckpt["state_dict"]["prefix"].float().numpy()  # (k, d_model)
    k, d = P.shape

    print("  Loading embedding matrix (CPU only, safetensors)…")
    try:
        W_E, tokenizer = _load_embedding_matrix(model_name)
    except RuntimeError as e:
        print(f"  SKIP fig7: {e}")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, str(e), ha="center", va="center", fontsize=9, wrap=True)
        savefig(fig, out, "fig7_prefix_nn.png")
        return

    # Normalise both for cosine similarity
    P_n = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    WE_n = W_E / (np.linalg.norm(W_E, axis=1, keepdims=True) + 1e-8)

    cos_all = P_n @ WE_n.T  # (k, vocab_size)

    # Top-k nearest neighbours per prefix token
    nn_idx = np.argsort(-cos_all, axis=1)[:, :top_k]  # (k, top_k)
    nn_cos = cos_all[np.arange(k)[:, None], nn_idx]

    def decode(tok_id: int) -> str:
        s = tokenizer.decode([tok_id]).strip()
        return repr(s) if (not s or s.isspace()) else s[:12]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2.5, 1]})

    # ── Left: table of top-k neighbours per token ─────────────────────────
    ax = axes[0]
    ax.axis("off")
    col_labels = [f"#{i + 1} token  (cos)" for i in range(top_k)]
    row_labels = [f"P{i}" for i in range(k)]
    cell_text = []
    for i in range(k):
        row = []
        for j in range(top_k):
            tok = decode(nn_idx[i, j])
            cos = nn_cos[i, j]
            row.append(f"{tok}\n({cos:.2f})")
        cell_text.append(row)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 2.2)
    # colour prefix token rows alternately
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#DDEEFF")
        elif c == -1:
            cell.set_facecolor("#FFF8E1")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if r % 2 == 0 else "#F9F9F9")
    ax.set_title(
        f"Top-{top_k} nearest real tokens per prefix vector  (cosine similarity)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )

    # ── Right: heatmap of max cosine similarity per prefix token ─────────
    ax2 = axes[1]
    max_cos = cos_all.max(axis=1)  # (k,)
    mean_cos = cos_all.mean(axis=1)

    x = np.arange(k)
    ax2.barh(x, max_cos, height=0.4, left=0, color=C_PFX, alpha=0.85, label="max cos")
    ax2.barh(x + 0.4, mean_cos, height=0.4, left=0, color=C_BASE, alpha=0.85, label="mean cos")
    ax2.set_yticks(x + 0.2)
    ax2.set_yticklabels([f"P{i}" for i in range(k)], fontsize=9)
    ax2.set_xlabel("Cosine similarity to vocab")
    ax2.set_title("Distance from token manifold", fontsize=10, fontweight="bold")
    ax2.axvline(1.0, color=C_GRAY, lw=1, linestyle=":")
    ax2.legend(fontsize=8)
    # Annotate: if max_cos < 0.5, token is in "novel" space
    for i, mc in enumerate(max_cos):
        label = "novel" if mc < 0.5 else ""
        ax2.text(mc + 0.01, i + 0.0, label, va="center", fontsize=7, color=C_WRONG)

    fig.suptitle("What real tokens do the prefix vectors resemble?", fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, out, "fig7_prefix_nn.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize soft-prompt experiment results")
    p.add_argument("--out_root", default="runs/soft_prompt")
    p.add_argument("--dataset_path", default="data/addition_3digit.jsonl")
    p.add_argument("--mode", default="soft_prompt", choices=["soft_prompt", "prefix_tuning"])
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument(
        "--figures",
        nargs="+",
        type=int,
        default=None,
        help="Which figures to generate (1-7). Default: all",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.out_root)
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    mode = args.mode
    figs = set(args.figures) if args.figures else set(range(1, 8))

    # Paths
    dataset_path = Path(args.dataset_path)
    analysis_path = root / f"analysis_{mode}.json"
    ckpt_path = root / f"prefix_{mode}.pt"
    samples_path = root / f"eval_{mode}_samples.json"

    print(f"Output directory: {out}")

    # Load dataset once
    dataset = []
    if dataset_path.exists():
        with open(dataset_path) as f:
            dataset = [json.loads(line) for line in f]

    if 1 in figs:
        print("[1/6] Prompt templates")
        fig_prompt_templates(dataset, out)

    if 2 in figs:
        print("[2/6] Digit coverage")
        sp = samples_path if samples_path.exists() else None
        fig_digit_coverage(dataset, sp, out)

    if 3 in figs:
        print("[3/6] Correct grid")
        sp = samples_path if samples_path.exists() else None
        fig_correct_grid(sp, out)

    if 4 in figs:
        if not analysis_path.exists():
            print(f"[4/6] SKIP — {analysis_path} not found")
        else:
            print("[4/6] Steering analysis")
            fig_steering(analysis_path, out)

    if 5 in figs:
        if not ckpt_path.exists():
            print(f"[5/6] SKIP — {ckpt_path} not found")
        else:
            print("[5/6] Prefix vectors")
            fig_prefix_vectors(ckpt_path, out)

    if 6 in figs:
        print("[6/7] Architecture diagram")
        fig_architecture(out)

    if 7 in figs:
        if not ckpt_path.exists():
            print(f"[7/7] SKIP — {ckpt_path} not found")
        else:
            print("[7/7] Prefix nearest-neighbour tokens")
            fig_prefix_nn(ckpt_path, out, model_name=args.model)

    print(f"\nDone. All figures in {out}/")


if __name__ == "__main__":
    main()
