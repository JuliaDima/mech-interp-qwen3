#!/usr/bin/env python3
"""Run a robustness experiment across multiple addition prompts to test circuit stability.

This script:
1. Generates 20 carry and 20 no-carry prompt pairs.
2. Runs `miq attribute` and `miq intervene` for each pair via subprocess to avoid
   memory fragmentation / OOM errors on GPUs.
3. Collects the intervention results.
4. Aggregates data (mean/std) per feature group.
5. Produces publication-style visualization plots.
"""

import csv
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate_prompt_pairs(num_carry=20, num_no_carry=20, seed=42):
    random.seed(seed)

    carry_pairs = []
    no_carry_pairs = []

    # We want a, b in [0, 99].
    # For carry: a%10 + b%10 >= 10. We need to find a b_perturb such that a%10 + b_perturb%10 < 10.
    # For no-carry: a%10 + b%10 < 10. We need to find a b_perturb such that a%10 + b_perturb%10 >= 10.

    while len(carry_pairs) < num_carry or len(no_carry_pairs) < num_no_carry:
        a = random.randint(0, 99)
        b = random.randint(0, 99)

        a_ones = a % 10
        b_ones = b % 10

        is_carry = (a_ones + b_ones) >= 10

        if is_carry and len(carry_pairs) < num_carry:
            # Pick a b_perturb that causes no-carry.
            # We want a_ones + b_perturb_ones < 10
            # Since a_ones >= 1 (otherwise carry is impossible), we can just set b_perturb_ones = 0.
            b_perturb = (b // 10) * 10  # set last digit to 0
            carry_pairs.append({"a": a, "b": b, "b_perturb": b_perturb, "type": "carry"})

        elif not is_carry and len(no_carry_pairs) < num_no_carry:
            # Pick a b_perturb that causes carry.
            # We want a_ones + b_perturb_ones >= 10.
            # We can set b_perturb_ones = 9 (which ensures >= 10 as long as a_ones >= 1).
            # If a_ones == 0, we can't create a carry just by changing b_ones (0+9=9 < 10).
            if a_ones == 0:
                continue
            b_perturb = (b // 10) * 10 + 9
            no_carry_pairs.append({"a": a, "b": b, "b_perturb": b_perturb, "type": "no_carry"})

    return carry_pairs + no_carry_pairs


def run_pipeline(pairs, transcoder_set="mwhanna/qwen3-4b-transcoders"):
    for idx, pair in enumerate(pairs):
        a, b, b_perturb = pair["a"], pair["b"], pair["b_perturb"]
        slug = f"addition_{a}_{b}"
        graph_dir = Path(f"benchmark_graphs/{slug}")
        graph_path = graph_dir / "attribution_graph.pt"
        out_dir = Path(f"runs/addition/interventions/{slug}")

        print(f"\n[{idx + 1}/{len(pairs)}] Processing {slug} ({pair['type']})")
        print(f"  Base prompt: calc: {a}+{b}=")
        print(f"  Perturbed  : calc: {a}+{b_perturb}=")

        # Step 1: miq attribute
        if not graph_path.exists():
            print("  => Running miq attribute...")
            graph_dir.mkdir(parents=True, exist_ok=True)
            cmd_attr = [
                "miq",
                "attribute",
                "--prompt",
                f"calc: {a}+{b}= ",
                "--slug",
                slug,
                "--graph_file_dir",
                str(graph_dir),
                "--graph_output_path",
                str(graph_path),
            ]
            subprocess.run(cmd_attr, check=True)
        else:
            print(f"  => Graph found at {graph_path}, skipping attribute...")

        # Step 2: miq intervene
        results_path = out_dir / "intervention_results.json"
        if not results_path.exists():
            print("  => Running miq intervene...")
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd_int = [
                "miq",
                "intervene",
                "--graph_path",
                str(graph_path),
                "--transcoder_set",
                transcoder_set,
                "--prompt",
                f"calc: {a}+{b}= ",
                "--perturbed_prompt",
                f"calc: {a}+{b_perturb}= ",
                "--out_dir",
                str(out_dir),
            ]
            subprocess.run(cmd_int, check=True)

            # Clean up the entire graph dir to save disk space
            if graph_dir.exists():
                print(f"  => Cleaning up {graph_dir}")
                import shutil

                shutil.rmtree(graph_dir, ignore_errors=True)
        else:
            print(f"  => Results found at {results_path}, skipping intervene...")

        # Also clean up existing graph if we just skipped intervene
        if graph_dir.exists():
            import shutil

            shutil.rmtree(graph_dir, ignore_errors=True)


def collect_data(pairs):
    all_records = []

    for pair in pairs:
        a, b = pair["a"], pair["b"]
        slug = f"addition_{a}_{b}"
        results_path = Path(f"runs/addition/interventions/{slug}/intervention_results.json")

        if not results_path.exists():
            print(f"Warning: Missing results for {slug}")
            continue

        with open(results_path) as f:
            data = json.load(f)

        results = data.get("results", data)
        for r in results:
            all_records.append(
                {
                    "prompt_a": a,
                    "prompt_b": b,
                    "prompt_type": pair["type"],
                    "group": r["group"],
                    "delta_logit_constrained": r.get("delta_logit_constrained", 0.0),
                    "delta_prob_constrained": r.get("delta_prob_constrained", 0.0),
                }
            )

    return all_records


def aggregate_and_plot(records):
    # Group -> type -> list of values
    groups_data = defaultdict(
        lambda: {
            "carry": {"dl": [], "dp": []},
            "no_carry": {"dl": [], "dp": []},
            "all": {"dl": [], "dp": []},
        }
    )

    for r in records:
        g = r["group"]
        pt = r["prompt_type"]
        groups_data[g][pt]["dl"].append(r["delta_logit_constrained"])
        groups_data[g][pt]["dp"].append(r["delta_prob_constrained"])
        groups_data[g]["all"]["dl"].append(r["delta_logit_constrained"])
        groups_data[g]["all"]["dp"].append(r["delta_prob_constrained"])

    summary = []
    for g, data in groups_data.items():
        summary.append(
            {
                "group": g,
                "mean_delta_logit_all": np.mean(data["all"]["dl"]),
                "std_delta_logit_all": np.std(data["all"]["dl"]),
                "mean_delta_prob_all": np.mean(data["all"]["dp"]),
                "std_delta_prob_all": np.std(data["all"]["dp"]),
                "mean_delta_logit_carry": np.mean(data["carry"]["dl"])
                if data["carry"]["dl"]
                else 0.0,
                "std_delta_logit_carry": np.std(data["carry"]["dl"])
                if data["carry"]["dl"]
                else 0.0,
                "mean_delta_prob_carry": np.mean(data["carry"]["dp"])
                if data["carry"]["dp"]
                else 0.0,
                "std_delta_prob_carry": np.std(data["carry"]["dp"]) if data["carry"]["dp"] else 0.0,
                "mean_delta_logit_no_carry": np.mean(data["no_carry"]["dl"])
                if data["no_carry"]["dl"]
                else 0.0,
                "std_delta_logit_no_carry": np.std(data["no_carry"]["dl"])
                if data["no_carry"]["dl"]
                else 0.0,
                "mean_delta_prob_no_carry": np.mean(data["no_carry"]["dp"])
                if data["no_carry"]["dp"]
                else 0.0,
                "std_delta_prob_no_carry": np.std(data["no_carry"]["dp"])
                if data["no_carry"]["dp"]
                else 0.0,
            }
        )

    # Standardize order of groups based on general intervention logic
    ordered_groups = []
    for target in ["low_precision_sum", "ones_digit_lookup", "sum_near_X", "say_number_ending_Y"]:
        if any(s["group"] == target for s in summary):
            ordered_groups.append(target)
    for s in summary:
        if s["group"] not in ordered_groups:
            ordered_groups.append(s["group"])

    # Dump JSON
    with open("robustness_results.json", "w") as f:
        json.dump(records, f, indent=2)

    # Dump CSV
    with open("robustness_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print("Saved robustness_results.json and robustness_summary.csv")

    # ---------------------------------------------------------
    # Plots
    # ---------------------------------------------------------

    # 1. Bar chart: mean delta logit per group with error bars (ALL)
    groups = ordered_groups
    means = [next(s["mean_delta_logit_all"] for s in summary if s["group"] == g) for g in groups]
    stds = [next(s["std_delta_logit_all"] for s in summary if s["group"] == g) for g in groups]

    plt.figure(figsize=(10, 6))
    plt.bar(groups, means, yerr=stds, capsize=5, color="skyblue", edgecolor="black")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Robustness: Mean Causal Importance across 40 Prompts", fontsize=14, pad=15)
    plt.ylabel("Mean Δ Logit (Constrained)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("robustness_logit.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Carry vs No-Carry Comparison (Probability)
    means_carry = [
        next(s["mean_delta_prob_carry"] for s in summary if s["group"] == g) for g in groups
    ]
    stds_carry = [
        next(s["std_delta_prob_carry"] for s in summary if s["group"] == g) for g in groups
    ]

    means_nocarry = [
        next(s["mean_delta_prob_no_carry"] for s in summary if s["group"] == g) for g in groups
    ]
    stds_nocarry = [
        next(s["std_delta_prob_no_carry"] for s in summary if s["group"] == g) for g in groups
    ]

    x = np.arange(len(groups))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(
        x - width / 2,
        means_carry,
        width,
        yerr=stds_carry,
        capsize=5,
        label="Carry Prompts",
        color="salmon",
        edgecolor="black",
    )
    plt.bar(
        x + width / 2,
        means_nocarry,
        width,
        yerr=stds_nocarry,
        capsize=5,
        label="No-Carry Prompts",
        color="lightblue",
        edgecolor="black",
    )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Carry vs No-Carry: Impact on Output Probability", fontsize=14, pad=15)
    plt.ylabel("Mean Δ Probability", fontsize=12)
    plt.xticks(x, groups, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig("robustness_probability.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved robustness_logit.png and robustness_probability.png")


def main():
    # print("Generating 20 carry and 20 no-carry prompt pairs...")
    pairs = generate_prompt_pairs(5, 5)

    print("\nRunning extraction and intervention pipeline...")
    run_pipeline(pairs)

    print("\nCollecting cross-prompt data...")
    records = collect_data(pairs)

    if records:
        print("\nAggregating and generating plots...")
        aggregate_and_plot(records)
    else:
        print("\nNo records extracted. Ensure `miq attribute` and `miq intervene` run correctly.")


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    main()
