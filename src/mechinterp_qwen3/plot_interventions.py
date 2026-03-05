import matplotlib.pyplot as plt
import numpy as np

"""
miq plot-interventions runs/addition/interventions/intervention_results.json --out_dir plots/
"""


def plot_causal_importance(results, out_dir):
    """Plot 1: Bar chart showing constrained delta logit (causal importance)."""
    groups = [r["group"] for r in results]
    deltas = [r["delta_logit_constrained"] for r in results]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(groups, deltas, color="skyblue", edgecolor="black")

    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Causal Importance of Feature Groups (Constrained Patching)", fontsize=14, pad=15)
    plt.ylabel("Δ Logit (Correct Token)", fontsize=12)
    plt.xlabel("Feature Group", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        va = "bottom" if height < 0 else "top"
        y_pos = height - 0.5 if height < 0 else height + 0.5
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            f"{height:+.2f}",
            ha="center",
            va=va,
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(out_dir / "causal_importance_logit.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_probability_impact(results, out_dir):
    """Plot 2: Bar chart showing constrained delta probability."""
    groups = [r["group"] for r in results]
    deltas = [r["delta_prob_constrained"] for r in results]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(groups, deltas, color="salmon", edgecolor="black")

    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Impact on Output Probability (Constrained Patching)", fontsize=14, pad=15)
    plt.ylabel("Δ Probability", fontsize=12)
    plt.xlabel("Feature Group", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    # Add values
    for bar in bars:
        height = bar.get_height()
        va = "bottom" if height < 0 else "top"
        y_pos = height - 0.05 if height < 0 else height + 0.05
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            f"{height:+.2f}",
            ha="center",
            va=va,
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(out_dir / "probability_impact.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_leakage_comparison(results, out_dir):
    """Plot 3: Comparison of Unconstrained vs Constrained Delta Logit."""
    groups = [r["group"] for r in results]
    unc_deltas = [r["delta_logit_unconstrained"] for r in results]
    con_deltas = [r["delta_logit_constrained"] for r in results]

    x = np.arange(len(groups))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(
        x - width / 2,
        unc_deltas,
        width,
        label="Unconstrained",
        color="lightgray",
        edgecolor="black",
    )
    plt.bar(
        x + width / 2, con_deltas, width, label="Constrained", color="steelblue", edgecolor="black"
    )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Δ Logit", fontsize=12)
    plt.title("Upstream Leakage: Unconstrained vs Constrained Patching", fontsize=14, pad=15)
    plt.xticks(x, groups, rotation=45, ha="right")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "leakage_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_layer_locations(results, out_dir):
    """Plot 4: Distribution of features across layers for each group."""
    plt.figure(figsize=(10, 4))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for i, (r, color) in enumerate(zip(results, colors, strict=False)):
        group = r["group"]
        layers = r.get("layers", [])  # Use get with default to avoid KeyError if missing

        if not layers:
            continue

        # Plot each layer as a point. Add slight vertical jitter to separate groups visually
        y_pos = np.full(len(layers), i)
        plt.scatter(
            layers,
            y_pos,
            label=f"{group} (n={r.get('n_features', len(layers))})",
            color=color,
            alpha=0.7,
            edgecolors="black",
            s=100,
        )

    plt.yticks(range(len(results)), [r["group"] for r in results])
    plt.xlabel("Network Layer", fontsize=12)
    plt.title("Feature Locations by Layer", fontsize=14, pad=15)
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    # Ensure x-axis spans reasonable layer numbers automatically
    all_layers = [L for r in results for L in r.get("layers", [])]
    if all_layers:
        plt.xlim(min(all_layers) - 1, max(all_layers) + 2)

    plt.tight_layout()
    plt.savefig(out_dir / "layer_locations.png", dpi=300, bbox_inches="tight")
    plt.close()
