"""Cross-concept correlation analysis.

For every (concept, anchor) pair collects:
  - anchor rank (1 = best by abruptness)
  - z_score and empirical_p from null permutation
  - sharpness_index and peak_layer from run_concept
  - causal patching effect and grad-dot-delta alignment (positive sum)
  - abruptness score (ranking criterion in top_k_anchors)
  - peak prominence / non-monotonicity on the double-normalised trajectory
  - peak prominence on the raw-norm trajectory

Then prints Spearman correlation tables and saves scatter plots.

Usage
-----
    python scripts/analyze_concept_correlations.py
    python scripts/analyze_concept_correlations.py --out_dir runs/concept_localization/stats
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.plot_style import apply

_BASE = _REPO_ROOT / "runs" / "concept_localization"
_DECAY = 3.0

# Characters counted as "mathematical" for the math-fraction feature.
# Excludes punctuation that appears in natural language (. , : -)
_MATH_CHARS = set("0123456789+*/=%()[]→^·>")


# ── Prompt structure features (per concept, T0 only) ─────────────────────────

def _prompt_features(concept: str) -> dict:
    """Return prompt_n_chars and prompt_math_frac for the T0 positive prompt.

    math_frac = |{chars that are digits or math operators}| / |non-space chars|
    Returns NaN fields if the dataset cannot be loaded.
    """
    nan = {"prompt_n_chars": float("nan"), "prompt_math_frac": float("nan")}
    try:
        import importlib
        mod = importlib.import_module(f"data.concept_datasets.{concept}_dataset")
        fn  = next(f for f in dir(mod) if f.startswith("generate_") and f.endswith("_pairs"))
        pairs = getattr(mod, fn)(5, seed=42)
        t0 = next((p for p in pairs if p.template == "T0"), None)
        if t0 is None:
            return nan
        prompt = t0.prompt_pos
        non_space = [c for c in prompt if c != " "]
        if not non_space:
            return nan
        math_frac = sum(1 for c in non_space if c in _MATH_CHARS) / len(non_space)
        return {
            "prompt_n_chars":  len(prompt),
            "prompt_math_frac": round(math_frac, 4),
        }
    except Exception:
        return nan


# ── Metric helpers ────────────────────────────────────────────────────────────

def _abruptness(traj: np.ndarray) -> float:
    """Early-weighted max step (ranking criterion used in top_k_anchors)."""
    m = traj.max()
    if m < 1e-8:
        return 0.0
    t = traj / m
    w = 2
    diffs = t[w:] - t[:-w]
    weights = np.exp(-_DECAY * np.arange(len(diffs)) / len(diffs))
    return float((diffs * weights).max())


def _peak_prominence(traj: np.ndarray) -> float:
    """Non-monotonicity: max over i of min(c_i - min(c[:i]), c_i - min(c[i+1:]))."""
    best = 0.0
    for i in range(1, len(traj) - 1):
        left_drop = traj[i] - traj[:i].min()
        right_drop = traj[i] - traj[i + 1:].min()
        best = max(best, min(left_drop, right_drop))
    return best


def _causal_scalar(causal_block: dict | None, key: str) -> float:
    """Sum positive layer values from a causal dict entry."""
    if causal_block is None:
        return float("nan")
    vals = list(causal_block.get(key, {}).values())
    pos = [v for v in vals if v > 0]
    return sum(pos) if pos else 0.0


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_records() -> list[dict]:
    records = []
    _prompt_cache: dict[str, dict] = {}   # concept → prompt features

    for concept_dir in sorted(_BASE.iterdir()):
        if not concept_dir.is_dir():
            continue
        concept = concept_dir.name
        if concept in ("stats",):
            continue

        if concept not in _prompt_cache:
            _prompt_cache[concept] = _prompt_features(concept)

        em_path = concept_dir / "emergence.npy"
        if not em_path.exists():
            continue
        em = np.load(em_path, allow_pickle=True).item()
        norms_raw: np.ndarray = em["norms_raw"]        # (n_anchors, n_layers)
        act_norms_raw: np.ndarray = em["act_norms_raw"]  # (n_anchors, n_layers)

        for anchor_dir in sorted(concept_dir.glob("anchor_rank*_pos*")):
            m = re.match(r"anchor_rank(\d+)_pos(\d+)", anchor_dir.name)
            if not m:
                continue
            rank = int(m.group(1))
            pos  = int(m.group(2))

            # ── null ─────────────────────────────────────────────────────────
            null_path = anchor_dir / "null" / "null_permutation.json"
            if not null_path.exists():
                continue
            null = json.loads(null_path.read_text())

            # ── results ──────────────────────────────────────────────────────
            res_path = anchor_dir / "results.json"
            if not res_path.exists():
                continue
            res = json.loads(res_path.read_text())

            causal_all = (res.get("causal") or {}).get("all")

            # ── trajectory metrics from emergence.npy ────────────────────────
            if pos >= norms_raw.shape[0]:
                continue
            raw_traj = norms_raw[pos]
            act_traj = act_norms_raw[pos]

            # Double-normalised (act-norm then /max) — same as top_k_anchors
            act_m = act_traj.max()
            dbl_traj = act_traj / act_m if act_m > 1e-8 else act_traj

            records.append({
                "concept":           concept,
                "anchor_rank":       rank,
                "anchor_pos":        pos,
                # null metrics
                "z_score":           null["z_score"],
                "empirical_p":       null["empirical_p_value"],
                "null_mean_at_peak": null["null_mean_at_peak"],
                # sharpness
                "peak_layer":        res["sharpness"]["peak_layer"],
                "sharpness_index":   res["sharpness"]["sharpness_index"],
                # causal
                "causal_patch_pos":  _causal_scalar(causal_all, "patching_mean"),
                "grad_dot_delta":    _causal_scalar(causal_all, "grad_dot_delta_mean"),
                # trajectory metrics
                "abruptness":        _abruptness(dbl_traj),
                "nm_dbl":            _peak_prominence(dbl_traj),   # non-monotonicity on double-norm
                "nm_raw":            _peak_prominence(raw_traj / (raw_traj.max() + 1e-12)),  # on raw-norm
                "raw_peak_norm":     float(raw_traj.max()),
                "act_peak_norm":     float(act_traj.max()),
                **_prompt_cache[concept],
            })

    return records


# ── Analysis ──────────────────────────────────────────────────────────────────

METRICS = [
    "z_score", "empirical_p",
    "causal_patch_pos", "grad_dot_delta",
    "sharpness_index", "peak_layer",
    "abruptness", "nm_dbl", "nm_raw",
    "raw_peak_norm", "act_peak_norm",
    "anchor_rank",
    "prompt_n_chars", "prompt_math_frac",
]

METRIC_LABELS = {
    "z_score":           "z-score (null sep.)",
    "empirical_p":       "empirical p",
    "causal_patch_pos":  "causal patching (pos sum)",
    "grad_dot_delta":    "grad·delta alignment",
    "sharpness_index":   "sharpness index",
    "peak_layer":        "peak layer",
    "abruptness":        "abruptness (ranking score)",
    "nm_dbl":            "non-monotonicity (dbl-norm)",
    "nm_raw":            "non-monotonicity (raw-norm)",
    "raw_peak_norm":     "raw peak ||δ||",
    "act_peak_norm":     "act-norm peak ||δ||/E[||h||]",
    "anchor_rank":       "anchor rank (1=best)",
    "prompt_n_chars":    "prompt length (chars, T0)",
    "prompt_math_frac":  "math fraction (T0)",
}


def spearman_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in METRICS if c in df.columns]
    rho_mat = np.full((len(cols), len(cols)), np.nan)
    p_mat   = np.full((len(cols), len(cols)), np.nan)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            mask = df[c1].notna() & df[c2].notna()
            if mask.sum() < 5:
                continue
            r, p = stats.spearmanr(df.loc[mask, c1], df.loc[mask, c2])
            rho_mat[i, j] = r
            p_mat[i, j]   = p
    rho_df = pd.DataFrame(rho_mat, index=cols, columns=cols)
    p_df   = pd.DataFrame(p_mat,   index=cols, columns=cols)
    return rho_df, p_df


def _sig(p: float) -> str:
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "   "))


def print_pairwise_table(df: pd.DataFrame, targets: list[str], title: str) -> None:
    """Print Spearman ρ for every ordered pair in targets, sorted by |ρ|."""
    print(f"\n=== {title} ===")
    rows = []
    for x in targets:
        for y in targets:
            if x >= y or x not in df.columns or y not in df.columns:
                continue
            mask = df[x].notna() & df[y].notna()
            if mask.sum() < 5:
                continue
            r, p = stats.spearmanr(df.loc[mask, x], df.loc[mask, y])
            rows.append((x, y, r, p, mask.sum()))
    rows.sort(key=lambda t: -abs(t[2]))
    w = max(len(METRIC_LABELS.get(c, c)) for c in targets)
    for x, y, r, p, n in rows:
        lx = METRIC_LABELS.get(x, x)
        ly = METRIC_LABELS.get(y, y)
        print(f"  {lx:<{w}}  ↔  {ly:<{w}}  ρ={r:+.3f}  p={p:.3f}  {_sig(p)}  n={n}")


def print_correlations_with_z(df: pd.DataFrame) -> None:
    """Print Spearman ρ of every metric against z_score, sorted by |ρ|."""
    print("\n=== Spearman ρ with z_score (sorted by |ρ|) ===")
    print(f"  n = {df['z_score'].notna().sum()} (concept, anchor) pairs\n")
    rows = []
    for col in METRICS:
        if col == "z_score" or col not in df.columns:
            continue
        mask = df["z_score"].notna() & df[col].notna()
        if mask.sum() < 5:
            continue
        r, p = stats.spearmanr(df.loc[mask, "z_score"], df.loc[mask, col])
        rows.append((col, r, p, mask.sum()))
    rows.sort(key=lambda x: -abs(x[1]))
    for col, r, p, n in rows:
        print(f"  {METRIC_LABELS.get(col, col):<40}  ρ={r:+.3f}  p={p:.3f}  {_sig(p)}")


def print_rank_breakdown(df: pd.DataFrame) -> None:
    """For each rank, show median z_score and causal."""
    print("\n=== Median metrics by anchor rank ===")
    g = df.groupby("anchor_rank")[["z_score", "causal_patch_pos", "grad_dot_delta",
                                    "abruptness", "nm_dbl"]].median()
    print(g.round(3).to_string())


def print_per_concept(df: pd.DataFrame) -> None:
    """Best anchor (rank 1) per concept: z_score and causal."""
    print("\n=== Rank-1 anchor per concept ===")
    r1 = df[df["anchor_rank"] == 1].sort_values("z_score", ascending=False)
    print(r1[["concept", "anchor_pos", "z_score", "empirical_p",
              "causal_patch_pos", "grad_dot_delta",
              "sharpness_index", "abruptness", "nm_dbl"]].to_string(index=False))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_corr_heatmap(rho_df: pd.DataFrame, p_df: pd.DataFrame, out_path: Path) -> None:
    apply()
    labels = [METRIC_LABELS.get(c, c) for c in rho_df.columns]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(9, n * 0.9), max(7, n * 0.8)))
    im = ax.imshow(rho_df.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(labels, fontsize=7.5)
    for i in range(n):
        for j in range(n):
            v = rho_df.values[i, j]
            p = p_df.values[i, j]
            if np.isnan(v):
                continue
            sig = "*" if p < 0.05 else ""
            ax.text(j, i, f"{v:.2f}{sig}", ha="center", va="center",
                    fontsize=6.5, color="white" if abs(v) > 0.5 else "black")
    ax.set_title("Spearman ρ — all (concept, anchor) pairs  (* p<0.05)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_key_scatters(df: pd.DataFrame, out_path: Path) -> None:
    """z_score vs causal / abruptness / nm_dbl, coloured by anchor_rank."""
    apply()
    pairs = [
        ("abruptness",       "z_score",          "Abruptness  →  null separation"),
        ("nm_dbl",           "z_score",          "Non-monotonicity (dbl-norm)  →  null separation"),
        ("causal_patch_pos", "z_score",          "Causal patching  →  null separation"),
        ("grad_dot_delta",   "z_score",          "Grad·delta  →  null separation"),
        ("abruptness",       "causal_patch_pos", "Abruptness  →  causal patching"),
        ("nm_dbl",           "causal_patch_pos", "Non-monotonicity  →  causal patching"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    colours = {1: "#2563EB", 2: "#16A34A", 3: "#D97706", 4: "#DC2626"}
    for ax, (x_col, y_col, title) in zip(axes.flat, pairs):
        for rank, grp in df.groupby("anchor_rank"):
            mask = grp[x_col].notna() & grp[y_col].notna()
            ax.scatter(grp.loc[mask, x_col], grp.loc[mask, y_col],
                       color=colours.get(rank, "gray"), s=35, alpha=0.75,
                       label=f"rank {rank}", zorder=3)
            # concept labels for rank-1 only
            if rank == 1:
                for _, row in grp[mask].iterrows():
                    ax.text(row[x_col], row[y_col], row["concept"][:5],
                            fontsize=5.5, ha="left", va="bottom", color="#2563EB", alpha=0.8)
        r, p = stats.spearmanr(df[x_col].dropna(), df[y_col][df[x_col].notna()])
        ax.set_xlabel(METRIC_LABELS.get(x_col, x_col), fontsize=8)
        ax.set_ylabel(METRIC_LABELS.get(y_col, y_col), fontsize=8)
        ax.set_title(f"{title}\nρ={r:+.3f}  p={p:.3f}", fontsize=8)
        ax.tick_params(labelsize=7)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=colours[r], markersize=7, label=f"rank {r}")
               for r in sorted(colours)]
    axes.flat[-1].legend(handles=handles, fontsize=7)
    fig.suptitle("Concept localization — metric correlations\n(each point = one concept × anchor)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_signal_scatters(df: pd.DataFrame, out_path: Path) -> None:
    """Focused scatter grid: causal/grad·delta vs trajectory shape metrics."""
    apply()
    pairs = [
        ("causal_patch_pos", "grad_dot_delta",   "Causal patching  ↔  grad·delta"),
        ("nm_dbl",           "causal_patch_pos", "Non-monotonicity  ↔  causal patching"),
        ("nm_dbl",           "grad_dot_delta",   "Non-monotonicity  ↔  grad·delta"),
        ("abruptness",       "nm_dbl",           "Abruptness  ↔  non-monotonicity"),
        ("act_peak_norm",    "causal_patch_pos", "Act-norm peak  ↔  causal patching"),
        ("act_peak_norm",    "nm_dbl",           "Act-norm peak  ↔  non-monotonicity"),
        ("sharpness_index",  "causal_patch_pos", "Sharpness  ↔  causal patching"),
        ("sharpness_index",  "grad_dot_delta",   "Sharpness  ↔  grad·delta"),
        ("nm_dbl",           "nm_raw",           "Non-monotonicity: dbl-norm  ↔  raw-norm"),
    ]
    ncols, nrows = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 12))
    colours = {1: "#2563EB", 2: "#16A34A", 3: "#D97706", 4: "#DC2626"}
    for ax, (x_col, y_col, title) in zip(axes.flat, pairs):
        for rank, grp in df.groupby("anchor_rank"):
            mask = grp[x_col].notna() & grp[y_col].notna()
            ax.scatter(grp.loc[mask, x_col], grp.loc[mask, y_col],
                       color=colours.get(rank, "gray"), s=30, alpha=0.75,
                       label=f"rank {rank}", zorder=3)
        mask = df[x_col].notna() & df[y_col].notna()
        r, p = stats.spearmanr(df.loc[mask, x_col], df.loc[mask, y_col])
        ax.set_xlabel(METRIC_LABELS.get(x_col, x_col), fontsize=8)
        ax.set_ylabel(METRIC_LABELS.get(y_col, y_col), fontsize=8)
        ax.set_title(f"{title}\nρ={r:+.3f}  p={p:.3f}  {_sig(p)}", fontsize=8)
        ax.tick_params(labelsize=7)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=colours[r], markersize=7, label=f"rank {r}")
               for r in sorted(colours)]
    axes.flat[-1].legend(handles=handles, fontsize=7, loc="upper left")
    fig.suptitle("Signal metric correlations — causal, trajectory, shape\n"
                 "(each point = one concept × anchor, colour = anchor rank)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out_dir", default=str(_BASE / "stats"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading records …")
    records = load_all_records()
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} (concept, anchor) pairs across {df['concept'].nunique()} concepts.")

    csv_path = out_dir / "concept_anchor_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV → {csv_path}")

    print_per_concept(df)
    print_rank_breakdown(df)
    print_correlations_with_z(df)

    # ── Core signal metric pairs ──────────────────────────────────────────────
    signal_metrics = [
        "causal_patch_pos", "grad_dot_delta",
        "abruptness", "nm_dbl", "nm_raw",
        "sharpness_index", "act_peak_norm", "raw_peak_norm",
    ]
    print_pairwise_table(df, signal_metrics,
                         "All pairwise Spearman ρ — signal metrics (sorted by |ρ|)")

    # ── Causal ↔ gradient-dot-delta vs trajectory, broken down by rank ────────
    print("\n=== Causal patching ↔ grad·delta, by anchor rank ===")
    for rank, grp in df.groupby("anchor_rank"):
        mask = grp["causal_patch_pos"].notna() & grp["grad_dot_delta"].notna()
        if mask.sum() < 4:
            continue
        r, p = stats.spearmanr(grp.loc[mask, "causal_patch_pos"], grp.loc[mask, "grad_dot_delta"])
        print(f"  rank {rank}  n={mask.sum()}  ρ={r:+.3f}  p={p:.3f}  {_sig(p)}")

    print("\n=== Non-monotonicity (dbl-norm) ↔ causal / grad·delta, by rank ===")
    for col in ("causal_patch_pos", "grad_dot_delta"):
        print(f"  vs {METRIC_LABELS[col]}:")
        for rank, grp in df.groupby("anchor_rank"):
            mask = grp["nm_dbl"].notna() & grp[col].notna()
            if mask.sum() < 4:
                continue
            r, p = stats.spearmanr(grp.loc[mask, "nm_dbl"], grp.loc[mask, col])
            print(f"    rank {rank}  n={mask.sum()}  ρ={r:+.3f}  p={p:.3f}  {_sig(p)}")

    print("\n=== Abruptness ↔ non-monotonicity (are they measuring the same thing?) ===")
    for col in ("nm_dbl", "nm_raw"):
        mask = df["abruptness"].notna() & df[col].notna()
        r, p = stats.spearmanr(df.loc[mask, "abruptness"], df.loc[mask, col])
        print(f"  abruptness ↔ {METRIC_LABELS[col]:<40}  ρ={r:+.3f}  p={p:.3f}  {_sig(p)}")

    # ── Prompt structure correlations ────────────────────────────────────────
    print("\n=== Prompt structure ↔ localization metrics (Spearman ρ) ===")
    print(f"  {'Localization metric':<36}  {'prompt_n_chars':>16}  {'prompt_math_frac':>16}")
    print("  " + "─" * 72)
    localization_metrics = [
        "z_score", "sharpness_index", "abruptness", "nm_dbl",
        "act_peak_norm", "causal_patch_pos", "grad_dot_delta",
    ]
    for col in localization_metrics:
        row_parts = []
        for struct_col in ("prompt_n_chars", "prompt_math_frac"):
            mask = df[col].notna() & df[struct_col].notna()
            if mask.sum() < 4:
                row_parts.append(f"{'n/a':>16}")
                continue
            r, p = stats.spearmanr(df.loc[mask, col], df.loc[mask, struct_col])
            row_parts.append(f"  ρ={r:+.3f} {_sig(p):>3}")
        label = METRIC_LABELS.get(col, col)
        print(f"  {label:<36}{''.join(row_parts)}")

    # also print per-concept prompt features for context
    print("\n  Per-concept prompt features (T0):")
    pf = (df[["concept", "prompt_n_chars", "prompt_math_frac"]]
          .drop_duplicates("concept")
          .sort_values("prompt_math_frac", ascending=False))
    print(f"  {'Concept':<28}  {'n_chars':>8}  {'math_frac':>10}")
    for _, row in pf.iterrows():
        print(f"  {row['concept']:<28}  {row['prompt_n_chars']:>8.0f}  {row['prompt_math_frac']:>10.3f}")

    rho_df, p_df = spearman_table(df)
    heatmap_path = out_dir / "correlation_heatmap.png"
    plot_corr_heatmap(rho_df, p_df, heatmap_path)
    print(f"\nSaved heatmap → {heatmap_path}")

    scatter_path = out_dir / "key_scatter_plots.png"
    plot_key_scatters(df, scatter_path)
    print(f"Saved scatter → {scatter_path}")

    signal_scatter_path = out_dir / "signal_metric_scatters.png"
    plot_signal_scatters(df, signal_scatter_path)
    print(f"Saved signal scatter → {signal_scatter_path}")


if __name__ == "__main__":
    main()
