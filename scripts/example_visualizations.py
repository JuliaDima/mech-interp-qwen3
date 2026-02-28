#!/usr/bin/env python3
"""Example: Generate custom visualizations programmatically.

This shows how to use the visualization API for custom analysis beyond the CLI.
"""

from pathlib import Path

from src.mechinterp_qwen3.visualize_dataset import (
    classify_carry,
    create_carry_structure_plot,
    create_comprehensive_report,
    create_entropy_map,
    create_positional_cascade,
    create_probability_heatmap,
    load_dataset,
)

# ============================================================================
# Example 1: Generate all visualizations for a specific template
# ============================================================================


def example_comprehensive():
    """Generate all visualizations at once."""
    print("Example 1: Comprehensive report")
    print("=" * 60)

    create_comprehensive_report(
        jsonl_path=Path("data/addition_grid.jsonl"),
        output_dir=Path("visualizations/example1"),
        template_id="T0",
    )


# ============================================================================
# Example 2: Generate specific visualizations with custom settings
# ============================================================================


def example_custom():
    """Generate specific visualizations with custom parameters."""
    print("\nExample 2: Custom visualizations")
    print("=" * 60)

    records = load_dataset(Path("data/addition_grid.jsonl"))

    # Create a larger heatmap with custom size
    print("Creating large heatmap...")
    fig, ax = create_probability_heatmap(
        records,
        template_id="T0",
        position=0,
        output_path=Path("visualizations/example2/large_heatmap.png"),
        figsize=(16, 14),
    )

    # Create entropy map for position 1
    print("Creating entropy map for position 1...")
    create_entropy_map(
        records,
        template_id="T0",
        position=1,
        output_path=Path("visualizations/example2/entropy_pos1.png"),
        figsize=(14, 12),
    )

    print("✓ Custom visualizations saved to visualizations/example2/")


# ============================================================================
# Example 3: Compare multiple templates
# ============================================================================


def example_compare_templates():
    """Generate visualizations for all templates to compare."""
    print("\nExample 3: Compare templates")
    print("=" * 60)

    records = load_dataset(Path("data/addition_grid.jsonl"))

    templates = ["T0", "T1", "T2"]

    for template in templates:
        # Check if template exists in dataset
        template_records = [r for r in records if r["template_id"] == template]
        if not template_records:
            print(f"  Skipping {template} (not in dataset)")
            continue

        print(f"  Generating for template {template}...")

        # Create probability heatmap for each template
        create_probability_heatmap(
            records,
            template_id=template,
            position=0,
            output_path=Path(f"visualizations/example3/heatmap_{template}.png"),
            figsize=(12, 10),
        )

    print("✓ Template comparison saved to visualizations/example3/")


# ============================================================================
# Example 4: Analyze specific carry patterns
# ============================================================================


def example_carry_analysis():
    """Deep dive into carry pattern analysis."""
    print("\nExample 4: Carry pattern analysis")
    print("=" * 60)

    records = load_dataset(Path("data/addition_grid.jsonl"))

    # Filter by template
    t0_records = [r for r in records if r["template_id"] == "T0"]

    # Analyze carry pattern distribution
    carry_counts = {"no_carry": 0, "single_carry": 0, "multi_carry": 0}
    carry_probs = {"no_carry": [], "single_carry": [], "multi_carry": []}

    for rec in t0_records:
        carry_type = classify_carry(rec["a"], rec["b"])
        carry_counts[carry_type] += 1

        if rec["per_pos"]:
            carry_probs[carry_type].append(rec["per_pos"][0]["prob_true"])

    print("\nCarry pattern distribution:")
    for carry_type, count in carry_counts.items():
        mean_prob = (
            sum(carry_probs[carry_type]) / len(carry_probs[carry_type])
            if carry_probs[carry_type]
            else 0
        )
        print(f"  {carry_type}: {count} cases, mean P(correct)={mean_prob:.4f}")

    # Generate carry structure visualization
    create_carry_structure_plot(
        records,
        template_id="T0",
        position=0,
        output_path=Path("visualizations/example4/carry_deep_dive.png"),
        figsize=(16, 8),
    )

    print("✓ Carry analysis saved to visualizations/example4/")


# ============================================================================
# Example 5: Find interesting cases for circuit analysis
# ============================================================================


def example_find_interesting():
    """Find interesting cases for detailed circuit analysis."""
    print("\nExample 5: Find interesting cases")
    print("=" * 60)

    records = load_dataset(Path("data/addition_grid.jsonl"))
    t0_records = [r for r in records if r["template_id"] == "T0"]

    # Find cases with high entropy but low probability (model confused)
    import numpy as np

    confused_cases = []
    confident_wrong = []
    perfect_cases = []

    for rec in t0_records:
        if not rec["per_pos"]:
            continue

        pos0 = rec["per_pos"][0]
        prob = pos0["prob_true"]

        # Compute entropy
        topk_probs = np.array(pos0["topk_probs"])
        topk_probs = topk_probs[topk_probs > 0]
        entropy = -np.sum(topk_probs * np.log2(topk_probs + 1e-10))

        # Categorize
        if entropy > 2.0 and prob < 0.3:
            confused_cases.append((rec["a"], rec["b"], prob, entropy))
        elif entropy < 0.5 and prob < 0.3:
            confident_wrong.append((rec["a"], rec["b"], prob, entropy))
        elif prob > 0.95 and entropy < 0.3:
            perfect_cases.append((rec["a"], rec["b"], prob, entropy))

    print(f"\nFound {len(confused_cases)} confused cases (high entropy, low prob)")
    print("Top 5 confused cases:")
    for a, b, p, e in sorted(confused_cases, key=lambda x: -x[3])[:5]:
        print(f"  {a} + {b}: P={p:.3f}, Entropy={e:.3f}")

    print(f"\nFound {len(confident_wrong)} confident-wrong cases (low entropy, low prob)")
    print("Top 5 confident-wrong cases (MOST INTERESTING!):")
    for a, b, p, e in sorted(confident_wrong, key=lambda x: x[1])[:5]:
        print(f"  {a} + {b}: P={p:.3f}, Entropy={e:.3f}")

    print(f"\nFound {len(perfect_cases)} perfect cases (high prob, low entropy)")
    print("Top 5 perfect cases:")
    for a, b, p, e in sorted(perfect_cases, key=lambda x: -x[2])[:5]:
        print(f"  {a} + {b}: P={p:.3f}, Entropy={e:.3f}")

    print("\n✓ These cases are good targets for circuit analysis!")


# ============================================================================
# Example 6: Analyze positional information flow
# ============================================================================


def example_positional_flow():
    """Analyze how information flows across positions."""
    print("\nExample 6: Positional information flow")
    print("=" * 60)

    records = load_dataset(Path("data/addition_grid.jsonl"))

    # Filter to records with multi-token answers
    multi_token = [r for r in records if r["template_id"] == "T0" and len(r["per_pos"]) >= 2]

    # Compute average probability gain from pos 0 to pos 1
    gains = []
    for rec in multi_token:
        p0 = rec["per_pos"][0]["prob_true"]
        p1 = rec["per_pos"][1]["prob_true"]
        gains.append(p1 - p0)

    mean_gain = sum(gains) / len(gains)
    print(f"\nAverage probability gain from position 0 → 1: {mean_gain:.4f}")
    print("This shows how much the first digit helps predict the second digit.")

    # Generate cascade visualization
    create_positional_cascade(
        records,
        template_id="T0",
        max_positions=2,
        output_path=Path("visualizations/example6/info_flow.png"),
        figsize=(14, 6),
    )

    print("✓ Information flow analysis saved to visualizations/example6/")


# ============================================================================
# Run all examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VISUALIZATION EXAMPLES")
    print("=" * 60)

    # Check if dataset exists
    if not Path("data/addition_grid.jsonl").exists():
        print("\nERROR: data/addition_grid.jsonl not found!")
        print("Please generate the dataset first:")
        print("  python -m mechinterp_qwen3.dataset_generation \\")
        print("    --output_path data/addition_grid.jsonl \\")
        print("    --sampling_strategy grid \\")
        print("    --max_value 20 \\")
        print("    --templates T0 T1 T2")
        exit(1)

    # Run examples (comment out ones you don't want)
    example_comprehensive()  # Generates all visualizations
    # example_custom()  # Custom sizes and positions
    # example_compare_templates()  # Compare T0, T1, T2
    # example_carry_analysis()  # Deep dive into carries
    example_find_interesting()  # Find cases for circuit analysis
    # example_positional_flow()  # Analyze information flow

    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 60)
    print("\nCheck the visualizations/ directory for outputs.")
