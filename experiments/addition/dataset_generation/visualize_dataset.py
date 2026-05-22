"""Advanced visualizations for addition dataset analysis.

Provides multiple visualization types inspired by Anthropic's mechanistic interpretability
work, with additional scientific insights for understanding model arithmetic capabilities.

usage: # will take default values from config.yaml, if not provided

   python -m src.mechinterp_qwen3.dataset_generation.visualize_dataset \
     data/addition_dataset.jsonl \
     --output_dir plots/grid \
     --template T0
"""

import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .generate_dataset_with_predictions import TEMPLATES as _TEMPLATES


def _template_label(template_id: str) -> str:
    """Return a short label like 'T0: "calc: {a}+{b}= "'."""
    fmt = _TEMPLATES.get(template_id)
    if fmt is None:
        return template_id
    fmt_display = fmt.replace("\n", "\\n")
    return f'{template_id}: "{fmt_display}"'


try:
    import seaborn as sns

    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def load_dataset(jsonl_path: Path) -> list[dict]:
    """Load dataset from JSONL file."""
    records = []
    with jsonl_path.open() as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_accuracy_sweep(json_path: Path, template_id: str = "T0") -> list[dict]:
    """Convert accuracy_sweep.json to the record format expected by visualize_dataset.

    The accuracy sweep stores one entry per (a, b) pair with top-level fields
    ``target_prob``, ``answer_str``, and ``predicted_str``.  This function
    wraps each entry into the ``per_pos`` schema so existing plot functions
    work unchanged.
    """
    with json_path.open() as f:
        data = json.load(f)
    results = data["results"]

    records = []
    for r in results:
        tail = r["prompt"].replace("calc: ", "").replace("= ", "").strip()
        a, b = map(int, tail.split("+"))
        pred = r["predicted_str"].strip()
        records.append(
            {
                "a": a,
                "b": b,
                "template_id": template_id,
                "per_pos": [
                    {
                        "prob_true": r["target_prob"],
                        "true_str": r["answer_str"][0] if r["answer_str"] else "",
                        "topk_strs": [pred[0] if pred else ""],
                        "topk_probs": [r.get("predicted_prob", 0.0)],
                    }
                ],
            }
        )
    return records


def classify_carry(a: int, b: int) -> Literal["no_carry", "single_carry", "multi_carry"]:
    """Classify addition by carry pattern."""
    carry_count = 0
    carry = 0
    a_str, b_str = (
        str(a).zfill(max(len(str(a)), len(str(b)))),
        str(b).zfill(max(len(str(a)), len(str(b)))),
    )
    for i in range(len(a_str) - 1, -1, -1):
        digit_sum = int(a_str[i]) + int(b_str[i]) + carry
        if digit_sum >= 10:
            carry_count += 1
            carry = 1
        else:
            carry = 0
    return "no_carry" if carry_count == 0 else "single_carry" if carry_count == 1 else "multi_carry"


def create_probability_heatmap(
    records: list[dict],
    template_id: str = "T0",
    position: int = 0,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (12, 10),
):
    """Create 2D heatmap of P(correct) for first answer token.

    Shows model confidence as a function of (a, b) coordinates.
    Inspired by Anthropic's circuit analysis visualizations.

    Args:
        records: Dataset records
        template_id: Which template to visualize
        position: Which answer token position (0 = first digit)
        output_path: Where to save figure
        figsize: Figure size
    """
    # Filter by template
    template_records = [r for r in records if r["template_id"] == template_id]

    # Determine grid size
    max_a = max(r["a"] for r in template_records)
    max_b = max(r["b"] for r in template_records)

    # Create probability grid
    prob_grid = np.full((max_b + 1, max_a + 1), np.nan)

    for rec in template_records:
        if len(rec["per_pos"]) > position:
            prob_grid[rec["b"], rec["a"]] = rec["per_pos"][position]["prob_true"]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Custom colormap: red (low) -> yellow -> green (high)
    colors = ["#d73027", "#fc8d59", "#fee090", "#e0f3f8", "#91bfdb", "#4575b4"]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("prob", colors, N=n_bins)

    # Plot heatmap
    im = ax.imshow(
        prob_grid,
        cmap=cmap,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="P(correct token)")
    cbar.ax.set_ylabel("P(correct token)", rotation=270, labelpad=20)

    # Labels
    ax.set_xlabel("a (first operand)", fontsize=12)
    ax.set_ylabel("b (second operand)", fontsize=12)
    ax.set_title(
        f"Model Confidence Map: Position {position} | {_template_label(template_id)}\n"
        f"Probability of correct {['first', 'second', 'third', 'fourth'][position]} digit",
        fontsize=14,
        pad=20,
    )

    # Add grid
    ax.set_xticks(np.arange(0, max_a + 1, 5))
    ax.set_yticks(np.arange(0, max_b + 1, 5))
    ax.grid(True, alpha=0.3, color="white", linewidth=0.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig, ax


def create_carry_structure_plot(
    records: list[dict],
    template_id: str = "T0",
    position: int = 0,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (12, 10),
):
    """Visualize carry pattern structure with overlaid probability.

    Shows three regions (no-carry, single-carry, multi-carry) with
    probability overlaid. Reveals if model uses carry patterns.

    Args:
        records: Dataset records
        template_id: Which template to visualize
        position: Which answer token position
        output_path: Where to save figure
        figsize: Figure size
    """
    template_records = [r for r in records if r["template_id"] == template_id]

    max_a = max(r["a"] for r in template_records)
    max_b = max(r["b"] for r in template_records)

    # Create grids
    carry_grid = np.zeros((max_b + 1, max_a + 1))
    prob_grid = np.full((max_b + 1, max_a + 1), np.nan)

    carry_map = {"no_carry": 0, "single_carry": 1, "multi_carry": 2}

    for rec in template_records:
        a, b = rec["a"], rec["b"]
        carry_type = classify_carry(a, b)
        carry_grid[b, a] = carry_map[carry_type]

        if len(rec["per_pos"]) > position:
            prob_grid[b, a] = rec["per_pos"][position]["prob_true"]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: Carry structure
    cmap_carry = LinearSegmentedColormap.from_list("carry", ["#2ecc71", "#f39c12", "#e74c3c"], N=3)
    im1 = ax1.imshow(carry_grid, cmap=cmap_carry, aspect="auto", origin="lower", vmin=0, vmax=2)

    cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2])
    cbar1.ax.set_yticklabels(["No carry", "Single carry", "Multi carry"])

    ax1.set_xlabel("a", fontsize=12)
    ax1.set_ylabel("b", fontsize=12)
    ax1.set_title("Carry Pattern Structure", fontsize=13)
    ax1.grid(True, alpha=0.3, color="white", linewidth=0.5)

    # Right: Probability with carry boundaries
    colors = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"]
    cmap_prob = LinearSegmentedColormap.from_list("prob", colors, N=100)

    im2 = ax2.imshow(prob_grid, cmap=cmap_prob, aspect="auto", origin="lower", vmin=0, vmax=1)

    plt.colorbar(im2, ax=ax2, label="P(correct)")

    # Overlay carry region boundaries
    for i in range(max_a + 1):
        for j in range(max_b + 1):
            # Draw thin borders where carry pattern changes
            if i < max_a and carry_grid[j, i] != carry_grid[j, i + 1]:
                ax2.axvline(i + 0.5, color="black", linewidth=0.5, alpha=0.4)
            if j < max_b and carry_grid[j, i] != carry_grid[j + 1, i]:
                ax2.axhline(j + 0.5, color="black", linewidth=0.5, alpha=0.4)

    ax2.set_xlabel("a", fontsize=12)
    ax2.set_ylabel("b", fontsize=12)
    ax2.set_title(f"P(correct) with Carry Boundaries | Pos {position}", fontsize=13)
    ax2.grid(True, alpha=0.2, color="white", linewidth=0.5)

    plt.suptitle(f"Carry Pattern Analysis | {_template_label(template_id)}", fontsize=15, y=0.98)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig, (ax1, ax2)


def create_diagonal_analysis(
    records: list[dict],
    template_id: str = "T0",
    position: int = 0,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (14, 6),
):
    """Analyze model performance along key diagonals.

    Diagonals represent constant sums (a+b=const). Shows if model
    uses sum-based or operand-based strategies.

    Args:
        records: Dataset records
        template_id: Which template to visualize
        position: Which answer token position
        output_path: Where to save figure
        figsize: Figure size
    """
    template_records = [r for r in records if r["template_id"] == template_id]

    # Group by sum
    sum_to_records = {}
    for rec in template_records:
        s = rec["a"] + rec["b"]
        if s not in sum_to_records:
            sum_to_records[s] = []
        sum_to_records[s].append(rec)

    # Compute mean probability per sum
    sums = sorted(sum_to_records.keys())
    mean_probs = []
    std_probs = []

    for s in sums:
        recs = sum_to_records[s]
        probs = [r["per_pos"][position]["prob_true"] for r in recs if len(r["per_pos"]) > position]
        if probs:
            mean_probs.append(np.mean(probs))
            std_probs.append(np.std(probs))
        else:
            mean_probs.append(np.nan)
            std_probs.append(np.nan)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: Mean probability by sum
    ax1.plot(
        sums, mean_probs, "o-", linewidth=2, markersize=4, color="#3498db", label="Mean P(correct)"
    )
    ax1.fill_between(
        sums,
        np.array(mean_probs) - np.array(std_probs),
        np.array(mean_probs) + np.array(std_probs),
        alpha=0.3,
        color="#3498db",
        label="±1 std dev",
    )
    ax1.set_xlabel("Sum (a + b)", fontsize=12)
    ax1.set_ylabel("Mean P(correct)", fontsize=12)
    ax1.set_title(f"Confidence vs Sum | Pos {position}", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="lower right", fontsize=10)

    # Right: Variance by sum (shows consistency)
    ax2.plot(
        sums,
        std_probs,
        "o-",
        linewidth=2,
        markersize=4,
        color="#e74c3c",
        label="Std Dev P(correct)",
    )
    ax2.set_xlabel("Sum (a + b)", fontsize=12)
    ax2.set_ylabel("Std Dev P(correct)", fontsize=12)
    ax2.set_title("Consistency vs Sum", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=10)

    plt.suptitle(
        f"Diagonal Analysis: Does model use sum-based strategy? | {_template_label(template_id)}",
        fontsize=14,
        y=0.98,
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig, (ax1, ax2)


def create_entropy_map(
    records: list[dict],
    template_id: str = "T0",
    position: int = 0,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (12, 10),
):
    """Visualize prediction entropy (uncertainty) across the grid.

    Higher entropy = model is uncertain between multiple tokens.
    Lower entropy = model is confident (even if wrong).

    Args:
        records: Dataset records
        template_id: Which template to visualize
        position: Which answer token position
        output_path: Where to save figure
        figsize: Figure size
    """
    template_records = [r for r in records if r["template_id"] == template_id]

    max_a = max(r["a"] for r in template_records)
    max_b = max(r["b"] for r in template_records)

    entropy_grid = np.full((max_b + 1, max_a + 1), np.nan)

    for rec in template_records:
        if len(rec["per_pos"]) > position:
            # Compute entropy from top-k probabilities
            topk_probs = rec["per_pos"][position]["topk_probs"]
            topk_probs = np.array(topk_probs)
            topk_probs = topk_probs[topk_probs > 0]  # Remove zeros

            entropy = -np.sum(topk_probs * np.log2(topk_probs + 1e-10))
            entropy_grid[rec["b"], rec["a"]] = entropy

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Colormap: blue (low entropy/confident) -> red (high entropy/uncertain)
    cmap = plt.cm.RdYlBu_r

    im = ax.imshow(entropy_grid, cmap=cmap, aspect="auto", origin="lower")

    cbar = plt.colorbar(im, ax=ax, label="Entropy (bits)")
    cbar.ax.set_ylabel("Prediction Entropy (bits)", rotation=270, labelpad=20)

    ax.set_xlabel("a (first operand)", fontsize=12)
    ax.set_ylabel("b (second operand)", fontsize=12)
    ax.set_title(
        f"Prediction Uncertainty Map | Position {position} | {_template_label(template_id)}\n"
        f"Higher entropy = model is more uncertain",
        fontsize=14,
        pad=20,
    )

    ax.grid(True, alpha=0.3, color="white", linewidth=0.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig, ax


def create_error_analysis(
    records: list[dict],
    template_id: str = "T0",
    position: int = 0,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (14, 10),
):
    """Analyze what the model predicts when it's wrong.

    Shows: (1) where model is correct/wrong, (2) what it predicts instead.

    Args:
        records: Dataset records
        template_id: Which template to visualize
        position: Which answer token position
        output_path: Where to save figure
        figsize: Figure size
    """
    template_records = [r for r in records if r["template_id"] == template_id]

    max_a = max(r["a"] for r in template_records)
    max_b = max(r["b"] for r in template_records)

    # Create grids
    correct_grid = np.full((max_b + 1, max_a + 1), np.nan)
    predicted_grid = np.full((max_b + 1, max_a + 1), np.nan)
    true_grid = np.full((max_b + 1, max_a + 1), np.nan)

    for rec in template_records:
        if len(rec["per_pos"]) > position:
            a, b = rec["a"], rec["b"]
            true_token = rec["per_pos"][position]["true_str"]
            pred_token = rec["per_pos"][position]["topk_strs"][0]  # Top-1 prediction

            # Store as integers for visualization
            true_digit = int(true_token) if true_token.isdigit() else -1
            pred_digit = int(pred_token) if pred_token.isdigit() else -1

            correct_grid[b, a] = 1.0 if pred_token == true_token else 0.0
            predicted_grid[b, a] = pred_digit
            true_grid[b, a] = true_digit

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Left: Correct/incorrect map
    cmap_correct = LinearSegmentedColormap.from_list("correct", ["#e74c3c", "#2ecc71"], N=2)
    im1 = axes[0].imshow(
        correct_grid, cmap=cmap_correct, aspect="auto", origin="lower", vmin=0, vmax=1
    )
    axes[0].set_title("Correct Predictions", fontsize=13)
    axes[0].set_xlabel("a", fontsize=11)
    axes[0].set_ylabel("b", fontsize=11)
    cbar1 = plt.colorbar(im1, ax=axes[0], ticks=[0, 1])
    cbar1.ax.set_yticklabels(["Wrong", "Correct"])

    # Middle: True digits
    im2 = axes[1].imshow(true_grid, cmap="viridis", aspect="auto", origin="lower", vmin=0, vmax=9)
    axes[1].set_title(f"True Digit at Pos {position}", fontsize=13)
    axes[1].set_xlabel("a", fontsize=11)
    axes[1].set_ylabel("b", fontsize=11)
    plt.colorbar(im2, ax=axes[1], label="Digit value")

    # Right: Predicted digits
    im3 = axes[2].imshow(
        predicted_grid, cmap="viridis", aspect="auto", origin="lower", vmin=0, vmax=9
    )
    axes[2].set_title(f"Predicted Digit at Pos {position}", fontsize=13)
    axes[2].set_xlabel("a", fontsize=11)
    axes[2].set_ylabel("b", fontsize=11)
    plt.colorbar(im3, ax=axes[2], label="Digit value")

    for ax in axes:
        ax.grid(True, alpha=0.3, color="white", linewidth=0.5)

    plt.suptitle(
        f"Error Analysis: Where and How Model Fails | {_template_label(template_id)}",
        fontsize=15,
        y=0.98,
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig, axes


def create_positional_cascade(
    records: list[dict],
    template_id: str = "T0",
    max_positions: int = 3,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (16, 5),
):
    """Show how confidence cascades across positions (teacher forcing effect).

    Args:
        records: Dataset records
        template_id: Which template to visualize
        max_positions: How many positions to show
        output_path: Where to save figure
        figsize: Figure size
    """
    template_records = [r for r in records if r["template_id"] == template_id]

    if not template_records:
        print(f"No records found for template {template_id}!")
        return None

    max_a = max(r["a"] for r in template_records)
    max_b = max(r["b"] for r in template_records)

    # Create subplots
    fig, axes = plt.subplots(1, max_positions, figsize=figsize)

    if max_positions == 1:
        axes = [axes]

    colors = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"]
    cmap = LinearSegmentedColormap.from_list("prob", colors, N=100)

    for pos in range(max_positions):
        prob_grid = np.full((max_b + 1, max_a + 1), np.nan)

        for rec in template_records:
            if pos < len(rec["per_pos"]):
                prob_grid[rec["b"], rec["a"]] = rec["per_pos"][pos]["prob_true"]

        im = axes[pos].imshow(prob_grid, cmap=cmap, aspect="auto", origin="lower", vmin=0, vmax=1)

        axes[pos].set_title(f"Position {pos}", fontsize=13)
        axes[pos].set_xlabel("a", fontsize=11)

        if pos == 0:
            axes[pos].set_ylabel("b", fontsize=11)

        axes[pos].grid(True, alpha=0.3, color="white", linewidth=0.5)

        # Add mean probability annotation
        mean_prob = np.nanmean(prob_grid)
        axes[pos].text(
            0.95,
            0.95,
            f"μ={mean_prob:.3f}",
            transform=axes[pos].transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Add shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="P(correct)")

    plt.suptitle(
        f"Confidence Cascade Across Positions | {_template_label(template_id)}\n"
        f"Shows teacher-forcing effect: confidence increases with more context",
        fontsize=14,
        y=1.02,
    )

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig, axes


def create_comprehensive_report(
    jsonl_path: Path,
    output_dir: Path,
    template_id: str = "T0",
):
    """Generate all visualizations for a dataset.

    Args:
        jsonl_path: Path to JSONL dataset
        output_dir: Directory to save visualizations
        template_id: Template to analyze
    """
    print(f"Loading dataset from {jsonl_path}...")
    records = load_dataset(jsonl_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating visualizations for template {template_id}...")

    # 1. Probability heatmaps for each position
    max_pos = max(len(r["per_pos"]) for r in records if r["template_id"] == template_id)
    for pos in range(min(max_pos, 3)):
        print(f"  Creating probability heatmap (position {pos})...")
        create_probability_heatmap(
            records,
            template_id=template_id,
            position=pos,
            output_path=output_dir / f"heatmap_pos{pos}.png",
        )
        plt.close()

    # 2. Carry structure analysis
    print("  Creating carry structure analysis...")
    create_carry_structure_plot(
        records,
        template_id=template_id,
        position=0,
        output_path=output_dir / "carry_structure.png",
    )
    plt.close()

    # 3. Diagonal analysis
    print("  Creating diagonal analysis...")
    create_diagonal_analysis(
        records,
        template_id=template_id,
        position=0,
        output_path=output_dir / "diagonal_analysis.png",
    )
    plt.close()

    # 4. Entropy map
    print("  Creating entropy map...")
    create_entropy_map(
        records,
        template_id=template_id,
        position=0,
        output_path=output_dir / "entropy_map.png",
    )
    plt.close()

    # 5. Error analysis
    print("  Creating error analysis...")
    create_error_analysis(
        records,
        template_id=template_id,
        position=0,
        output_path=output_dir / "error_analysis.png",
    )
    plt.close()

    # 6. Positional cascade
    print("  Creating positional cascade...")
    create_positional_cascade(
        records,
        template_id=template_id,
        max_positions=min(max_pos, 3),
        output_path=output_dir / "positional_cascade.png",
    )
    plt.close()

    print(f"\n✓ All visualizations saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize addition dataset")
    parser.add_argument(
        "dataset_path", type=Path, help="Path to JSONL dataset or accuracy_sweep.json"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("visualizations"),
        help="Output directory for plots",
    )
    parser.add_argument("--template", type=str, default="T0", help="Template to visualize")
    parser.add_argument(
        "--accuracy_sweep",
        action="store_true",
        help="Load from accuracy_sweep.json format instead of JSONL",
    )

    args = parser.parse_args()

    if args.accuracy_sweep:
        records = load_accuracy_sweep(args.dataset_path, template_id=args.template)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        create_carry_structure_plot(
            records,
            template_id=args.template,
            position=0,
            output_path=args.output_dir / "carry_structure.png",
        )
        plt.close()
        create_probability_heatmap(
            records,
            template_id=args.template,
            position=0,
            output_path=args.output_dir / "heatmap_pos0.png",
        )
        plt.close()
        create_diagonal_analysis(
            records,
            template_id=args.template,
            position=0,
            output_path=args.output_dir / "diagonal_analysis.png",
        )
        plt.close()
        create_error_analysis(
            records,
            template_id=args.template,
            position=0,
            output_path=args.output_dir / "error_analysis.png",
        )
        plt.close()
        print(f"Saved plots to {args.output_dir}")
    else:
        create_comprehensive_report(args.dataset_path, args.output_dir, args.template)
