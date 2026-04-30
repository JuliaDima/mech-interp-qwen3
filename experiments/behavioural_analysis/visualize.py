"""Visualisation and summary table generation for the behavioural sweep.

All functions take a pandas DataFrame (the raw results CSV) and an output
directory path and write plots + tables to that directory.

Column schema expected in the DataFrame:
    operation, template, digit_count, carry_type, seed, problem,
    ground_truth, model_answer, correct, per_digit_correct,
    per_digit_confidence, consistent, extraction_failed
"""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Colour palette for templates
_TEMPLATE_COLORS = {"T1": "#1f77b4", "T2": "#ff7f0e", "T3": "#2ca02c", "T4": "#d62728"}

_CARRY_LINESTYLES = {"carry_free": "-", "carry_heavy": "--", "none": "-"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_list_col(series: pd.Series) -> list[list]:
    """Parse a column of stringified Python lists back to Python lists."""
    return [ast.literal_eval(x) if isinstance(x, str) else x for x in series]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 1. Accuracy vs digit count
# ---------------------------------------------------------------------------


def plot_accuracy_vs_digit_count(df: pd.DataFrame, operation: str, out_dir: Path) -> None:
    """One line per (template, carry_type) combination."""
    _ensure_dir(out_dir)
    sub = df[df["operation"] == operation].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for template in ["T1", "T2", "T3", "T4"]:
        t_df = sub[sub["template"] == template]
        carry_types = t_df["carry_type"].unique()
        for ct in sorted(carry_types):
            ct_df = t_df[t_df["carry_type"] == ct]
            acc = ct_df.groupby("digit_count")["correct"].mean()
            label = f"{template}" + (f" ({ct})" if ct != "none" else "")
            ls = _CARRY_LINESTYLES.get(ct, "-")
            ax.plot(
                acc.index,
                acc.values,
                marker="o",
                color=_TEMPLATE_COLORS[template],
                linestyle=ls,
                label=label,
            )

    ax.set_xlabel("Digit count (n)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs digit count — {operation}")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(1, 9))
    ax.axhline(0.9, color="grey", linestyle=":", linewidth=0.8)
    ax.axhline(0.7, color="grey", linestyle=":", linewidth=0.8)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax.legend(loc="lower left", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"accuracy_{operation}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Per-digit error heatmap
# ---------------------------------------------------------------------------


def plot_per_digit_error_heatmap(df: pd.DataFrame, operation: str, out_dir: Path) -> None:
    """Rows = digit positions (0=LSB), columns = digit counts."""
    _ensure_dir(out_dir)
    sub = df[(df["operation"] == operation) & (df["template"] == "T1")].copy()
    if sub.empty:
        return

    digit_counts = sorted(sub["digit_count"].unique())
    max_pos = 0

    # Collect per-digit correct lists
    records = []
    for _, row in sub.iterrows():
        pdc = _parse_list_col(pd.Series([row["per_digit_correct"]]))[0]
        for pos, correct in enumerate(pdc):
            records.append({"digit_count": row["digit_count"], "pos": pos, "correct": int(correct)})
        max_pos = max(max_pos, len(pdc))

    if not records:
        return

    pos_df = pd.DataFrame(records)
    pivot = pos_df.pivot_table(index="pos", columns="digit_count", values="correct", aggfunc="mean")
    # Error rate = 1 - accuracy
    error_pivot = 1 - pivot

    fig, ax = plt.subplots(figsize=(max(6, len(digit_counts) * 0.8), max(4, max_pos * 0.5)))
    sns.heatmap(
        error_pivot,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
    )
    ax.set_xlabel("Digit count (n)")
    ax.set_ylabel("Digit position (0 = LSB / ones)")
    ax.set_title(f"Per-digit error rate — {operation} (T1)")
    fig.tight_layout()
    fig.savefig(out_dir / f"per_digit_error_{operation}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Confidence degradation curve
# ---------------------------------------------------------------------------


def plot_confidence_degradation(df: pd.DataFrame, operation: str, out_dir: Path) -> None:
    """Mean token probability on correct answer digit vs digit count (T1 only)."""
    _ensure_dir(out_dir)
    sub = df[(df["operation"] == operation) & (df["template"] == "T1")].copy()
    if sub.empty:
        return

    rows = []
    for _, row in sub.iterrows():
        pdc = _parse_list_col(pd.Series([row["per_digit_confidence"]]))[0]
        if pdc:
            rows.append({"digit_count": row["digit_count"], "mean_conf": float(np.mean(pdc))})
    if not rows:
        return

    conf_df = pd.DataFrame(rows)
    agg = conf_df.groupby("digit_count")["mean_conf"].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(agg.index, agg.values, marker="o", color=_TEMPLATE_COLORS["T1"])
    ax.set_xlabel("Digit count (n)")
    ax.set_ylabel("Mean token probability (correct digit)")
    ax.set_title(f"Confidence degradation — {operation} (T1)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(1, 9))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"confidence_{operation}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Consistency rate vs digit count
# ---------------------------------------------------------------------------


def plot_consistency_rate(df: pd.DataFrame, operation: str, out_dir: Path) -> None:
    _ensure_dir(out_dir)
    sub = df[(df["operation"] == operation) & (df["template"] == "T1")].copy()
    if sub.empty:
        return

    cons = sub.groupby("digit_count")["consistent"].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cons.index, cons.values, marker="o", color="#9467bd")
    ax.set_xlabel("Digit count (n)")
    ax.set_ylabel("Consistency rate (fraction of runs identical)")
    ax.set_title(f"Consistency vs digit count — {operation} (T1)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(1, 9))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"consistency_{operation}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Summary table
# ---------------------------------------------------------------------------


def make_summary_table(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """For each operation: digit count where accuracy first drops below 90/70/50% on T1.

    For operations with sub-variants (exponentiation by exponent; modular by
    modulus) each variant is a separate row.

    Returns the DataFrame and also saves it as summary_table.csv and .json.
    """
    _ensure_dir(out_dir)

    t1 = df[(df["template"] == "T1") & (df["carry_type"] == "none")].copy()
    # For add/sub, average over carry types
    add_sub = df[
        (df["template"] == "T1") & (df["operation"].isin(["addition", "subtraction"]))
    ].copy()
    t1 = pd.concat([t1, add_sub], ignore_index=True)

    thresholds = [0.9, 0.7, 0.5]
    rows = []

    for (op, ct), grp in t1.groupby(["operation", "carry_type"]):
        acc_by_n = grp.groupby("digit_count")["correct"].mean().sort_index()
        row: dict = {"operation": op, "carry_type": ct}
        for thresh in thresholds:
            col = f"first_n_below_{int(thresh * 100)}pct"
            below = acc_by_n[acc_by_n < thresh]
            row[col] = int(below.index[0]) if not below.empty else ">8"
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["operation", "carry_type"])
    summary.to_csv(out_dir / "summary_table.csv", index=False)
    summary.to_json(out_dir / "summary_table.json", orient="records", indent=2)
    return summary


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_all_plots(df: pd.DataFrame, out_dir: Path) -> None:
    """Generate all plots and summary table for every operation in df."""
    out_dir = Path(out_dir)
    operations = df["operation"].unique()
    for op in operations:
        plot_accuracy_vs_digit_count(df, op, out_dir)
        plot_per_digit_error_heatmap(df, op, out_dir)
        plot_confidence_degradation(df, op, out_dir)
    make_summary_table(df, out_dir)
