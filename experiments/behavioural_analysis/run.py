"""Behavioural analysis sweep — main entry point.

Run the full sweep:
    python -m experiments.behavioural_analysis.run

Run a quick smoke test (1 digit, 5 samples, add only):
    python -m experiments.behavioural_analysis.run --quick

Re-plot from an existing CSV (no model needed):
    python -m experiments.behavioural_analysis.run --plot-only --results-csv path/to/results.csv

See experiments/behavioural_analysis/README.md for full documentation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path when run directly
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.behavioural_analysis.dataset import (  # noqa: E402
    Problem,
    build_all_problems,
)
from experiments.behavioural_analysis.evaluate import batched_teacher_force  # noqa: E402
from experiments.behavioural_analysis.prompts import TEMPLATE_IDS, build_prompts  # noqa: E402
from experiments.behavioural_analysis.visualize import run_all_plots  # noqa: E402
from mechinterp_qwen3.utils.config_utils import add_config_args, print_config  # noqa: E402
from mechinterp_qwen3.utils.inference_utils import silence_libraries  # noqa: E402
from mechinterp_qwen3.utils_seed import seed_everything  # noqa: E402

silence_libraries()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("behavioural_analysis.run")


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "operation",
    "template",
    "digit_count",
    "carry_type",
    "seed",
    "problem",
    "ground_truth",
    "correct",
    "per_digit_correct",
    "per_digit_confidence",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_name: str, dtype_str: str):
    """Load a plain HookedTransformer (no transcoders needed for behavioural sweep)."""
    from transformer_lens import HookedTransformer

    from mechinterp_qwen3.utils.model_utils import parse_dtype

    dtype = parse_dtype(dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading model %s (dtype=%s, device=%s) …", model_name, dtype_str, device)
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-problem evaluation
# ---------------------------------------------------------------------------


def evaluate_batch(
    model,
    problems: list[Problem],
    template: str,
    batch_size: int,
) -> list[dict]:
    """Teacher-forcing pass for one batch; returns one result dict per problem."""
    prompts = [build_prompts(p.operation, p.operands, p.expression)[template] for p in problems]
    ground_truths = [p.ground_truth for p in problems]

    tf_results = batched_teacher_force(model, prompts, ground_truths)

    rows = []
    for prob, prompt, (pdc, pdc_conf) in zip(problems, prompts, tf_results, strict=False):
        rows.append(
            {
                "operation": prob.operation,
                "template": template,
                "digit_count": prob.n_digits,
                "carry_type": prob.carry_type,
                "seed": prob.seed,
                "problem": prompt,
                "ground_truth": prob.ground_truth,
                "correct": int(all(pdc)),
                "per_digit_correct": pdc,
                "per_digit_confidence": pdc_conf,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------


def run_sweep(
    model,
    problems: list[Problem],
    out_csv: Path,
    batch_size: int,
    templates: list[str] | None = None,
) -> list[dict]:
    """Iterate over all (template, problem) pairs and write results incrementally."""
    templates_to_run = templates if templates is not None else TEMPLATE_IDS
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    with open(out_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for template in tqdm(templates_to_run, desc="Templates"):
            log.info("Running template %s …", template)

            for i in tqdm(
                range(0, len(problems), batch_size),
                desc=f"  {template} batches",
                leave=False,
            ):
                batch = problems[i : i + batch_size]
                rows = evaluate_batch(model, batch, template, batch_size)
                for row in rows:
                    writer.writerow(
                        {k: str(v) if isinstance(v, list) else v for k, v in row.items()}
                    )
                    all_rows.append(row)
                csvfile.flush()

    log.info("Results written to %s (%d rows)", out_csv, len(all_rows))
    return all_rows


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="experiments.behavioural_analysis.run",
        description="Behavioural analysis sweep for Qwen3-4B on arithmetic tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--model",
        default="Qwen/Qwen3-4B",
        help="HuggingFace model identifier",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Model weight dtype",
    )
    p.add_argument("--seed", type=int, default=42, help="Global random seed")
    p.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Problems per (operation, digit_count, carry_type) cell",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size",
    )
    p.add_argument(
        "--n-digits-min",
        type=int,
        default=1,
        help="Minimum digit count for primary operand",
    )
    p.add_argument(
        "--n-digits-max",
        type=int,
        default=8,
        help="Maximum digit count for primary operand",
    )
    p.add_argument(
        "--templates",
        nargs="+",
        default=None,
        choices=["T1", "T2", "T3", "T4"],
        help="Subset of templates to run (default: all four)",
    )
    p.add_argument(
        "--out-root",
        default="runs/behavioural_analysis",
        help="Root directory for run outputs",
    )
    p.add_argument("--run-id", default=None, help="Custom run id (default: YYYY-MM-DD_HHMM)")

    # Subset flags
    p.add_argument(
        "--operations",
        nargs="+",
        default=None,
        help="Subset of operations to run (default: all)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: n_digits=[1], n_samples=5, addition only",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip inference; load existing CSV and re-generate plots",
    )
    p.add_argument(
        "--results-csv",
        default=None,
        help="Path to existing results CSV for --plot-only",
    )

    add_config_args(p)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print_config(args, title="Behavioural Analysis Sweep Configuration")

    # Output directory
    run_id = args.run_id or datetime.utcnow().strftime("%Y-%m-%d_%H%M")
    out_root = Path(args.out_root) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_root)

    # Save configuration
    with open(out_root / "config.json", "w") as fp:
        json.dump(vars(args), fp, indent=2)

    # ------------------------------------------------------------------
    # Plot-only mode
    # ------------------------------------------------------------------
    if args.plot_only:
        import pandas as pd

        csv_path = Path(args.results_csv) if args.results_csv else out_root / "results.csv"
        if not csv_path.exists():
            log.error("Results CSV not found: %s", csv_path)
            sys.exit(1)
        log.info("Loading results from %s …", csv_path)
        df = pd.read_csv(csv_path)
        log.info("Generating plots …")
        run_all_plots(df, out_root / "plots")
        print(f"\nPlots written to {out_root / 'plots'}")
        return

    # ------------------------------------------------------------------
    # Set seeds
    # ------------------------------------------------------------------
    seed_everything(args.seed)

    # ------------------------------------------------------------------
    # Quick / subset modes
    # ------------------------------------------------------------------
    if args.quick:
        log.info("Quick mode: n_digits=[1], n_samples=5, addition only")
        n_digits_range = range(1, 2)
        n_samples = 5
        operations_filter = {"addition"}
    else:
        n_digits_range = range(args.n_digits_min, args.n_digits_max + 1)
        n_samples = args.n_samples
        operations_filter = set(args.operations) if args.operations else None

    # ------------------------------------------------------------------
    # Generate problems
    # ------------------------------------------------------------------
    log.info("Generating problems …")
    problems = build_all_problems(
        n_digits_range=n_digits_range,
        n_samples=n_samples,
        base_seed=args.seed,
    )

    if operations_filter:
        problems = [p for p in problems if p.operation in operations_filter]

    log.info("Total problems: %d", len(problems))

    # Save problem index (seeds) for downstream experiments
    seeds_path = out_root / "problem_seeds.json"
    with open(seeds_path, "w") as fp:
        json.dump(
            [
                {
                    "operation": p.operation,
                    "n_digits": p.n_digits,
                    "carry_type": p.carry_type,
                    "seed": p.seed,
                    "expression": p.expression,
                    "ground_truth": p.ground_truth,
                }
                for p in problems
            ],
            fp,
            indent=2,
        )
    log.info("Problem seeds saved to %s", seeds_path)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = load_model(args.model, args.dtype)

    # ------------------------------------------------------------------
    # Run sweep
    # ------------------------------------------------------------------
    out_csv = out_root / "results.csv"
    log.info("Starting sweep → %s", out_csv)
    run_sweep(
        model,
        problems,
        out_csv,
        batch_size=args.batch_size,
        templates=args.templates,
    )

    # ------------------------------------------------------------------
    # Visualise
    # ------------------------------------------------------------------
    import pandas as pd

    log.info("Generating plots …")
    df = pd.read_csv(out_csv)
    plots_dir = out_root / "plots"
    run_all_plots(df, plots_dir)
    log.info("Plots written to %s", plots_dir)

    print(f"\n{'=' * 60}")
    print(f"Sweep complete.  Outputs in:\n  {out_root}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
