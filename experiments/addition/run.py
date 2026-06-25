#!/usr/bin/env python3
"""CLI entrypoint for the Anthropic addition case study reproduction.

Run end-to-end:
  python experiments/addition/run.py --config experiments/addition/config.yaml --all

  # Or override individual values:
  python experiments/addition/run.py --config experiments/addition/config.yaml --all --dtype float32

Or run individual phases:
  --make-prompts   Write prompt catalogue to run dir (no model needed)
  --operand-plots  Collect grid activations & plot 100x100 heatmaps
  --all            Run all phases in order

Attribution graph: use miq directly (see README.md step 4).

See experiments/addition/README.md for the full experiment description.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Ensure repo root is on sys.path so relative imports work when run as script
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mechinterp_qwen3.utils.config_utils import (  # noqa: E402
    add_config_args,
    load_config,
    print_config,
    set_parser_defaults_from_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("addition.run")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="addition/run.py",
        description="Anthropic addition case study reproduction on Qwen3-4B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Phase flags
    phases = p.add_argument_group("Phases")
    phases.add_argument("--make-prompts", action="store_true", help="Write prompt catalogue JSON")
    phases.add_argument("--operand-plots", action="store_true", help="100×100 operand heatmaps")
    phases.add_argument("--all", action="store_true", help="Run all phases in order")

    # Model
    model_args = p.add_argument_group("Model")
    model_args.add_argument(
        "--model",
        default=default_model(),
        help="HuggingFace model name",
    )
    model_args.add_argument(
        "--transcoder_set",
        default=default_transcoder_set(),
        help="HuggingFace transcoder set (repo id or local path)",
    )
    model_args.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Model dtype",
    )

    # Output
    out_args = p.add_argument_group("Output")
    out_args.add_argument(
        "--out_root",
        default="runs/addition",
        help="Root directory for run outputs",
    )
    out_args.add_argument(
        "--run_id",
        default=None,
        help="Custom run id (default: YYYY-MM-DD_HHMM)",
    )

    # Operand plots
    plot_args = p.add_argument_group("Operand plots")
    plot_args.add_argument(
        "--top_k_features",
        type=int,
        default=50,
        help="Number of top features to auto-discover for operand plots",
    )

    # Reproducibility
    repr_args = p.add_argument_group("Reproducibility")
    repr_args.add_argument("--seed", type=int, default=42, help="Global random seed")
    repr_args.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Backward-pass batch size for attribution",
    )

    # Config file
    add_config_args(p)

    return p


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _make_run_dir(out_root: str, run_id: str | None) -> Path:
    if run_id is None:
        run_id = datetime.utcnow().strftime("%Y-%m-%d_%H%M")
    run_dir = Path(out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=str(_REPO_ROOT),
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _write_metadata(run_dir: Path, args: argparse.Namespace) -> None:
    meta = {
        "git_commit": _get_git_commit(),
        "seed": args.seed,
        "model_id": args.model,
        "transcoder_id": args.transcoder_set,
        "dtype": args.dtype,
        "config_file": args.config,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": vars(args),
    }
    with open(run_dir / "metadata.json", "w") as fp:
        json.dump(meta, fp, indent=2)
    log.info("Metadata written to %s/metadata.json", run_dir)


def _set_seeds(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    log.info("Seeds set to %d", seed)


def _load_model(args: argparse.Namespace):
    """Load AttributionModel + transcoders."""
    import torch

    from mechinterp_qwen3.attribution_model import AttributionModel
    from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    log.info("Loading transcoder from %r …", args.transcoder_set)
    transcoder, config = load_transcoder_from_hub(
        args.transcoder_set,
        dtype=dtype,
        lazy_encoder=False,
        lazy_decoder=True,
    )

    model_name = args.model or config.get("model_name") or default_model()
    log.info("Loading model %r (dtype=%s) …", model_name, args.dtype)
    model = AttributionModel.from_pretrained_and_transcoders(model_name, transcoder, dtype=dtype)
    return model


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


def phase_make_prompts(run_dir: Path) -> None:
    """Write prompt catalogue to run_dir/prompts.json."""
    from experiments.addition.prompts import (
        CALC_GRID,
        FOCUS_ANSWER,
        FOCUS_PROMPT,
        GENERALIZATION_PROMPTS,
        NL_FOLLOWUP,
        NL_FOLLOWUP_CHAT,
        NL_VARIANT,
        NL_VARIANT_CHAT,
    )

    catalogue = {
        "focus_prompt": FOCUS_PROMPT,
        "focus_answer": FOCUS_ANSWER,
        "nl_variant": NL_VARIANT,
        "nl_variant_chat": NL_VARIANT_CHAT,
        "nl_followup": NL_FOLLOWUP,
        "nl_followup_chat": NL_FOLLOWUP_CHAT,
        "generalization_prompts": list(GENERALIZATION_PROMPTS),
        "calc_grid_size": len(CALC_GRID),
        "calc_grid_sample": CALC_GRID[:5],  # first 5 entries as sanity check
    }
    out = run_dir / "prompts.json"
    with open(out, "w") as fp:
        json.dump(catalogue, fp, indent=2)
    log.info("Prompt catalogue written to %s  (CALC_GRID: %d entries)", out, len(CALC_GRID))
    print(f"\n✓  make-prompts complete → {out}")


def phase_operand_plots(run_dir: Path, model, args: argparse.Namespace) -> None:
    """Collect grid activations and save 100×100 heatmaps."""
    from experiments.addition.operand_plots import run_operand_plots

    out_dir = run_dir / "operand_plots"
    matrices = run_operand_plots(
        model,
        out_dir=out_dir,
        top_k_global=args.top_k_features,
    )
    print(f"\n✓  operand-plots complete → {out_dir}  ({len(matrices)} feature matrices)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()

    pre, _ = parser.parse_known_args()

    # Check for positional config (e.g. "python run.py config.yaml --all")
    pos_config = None
    if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml") and not sys.argv[1].startswith("-"):
        pos_config = sys.argv[1]
        sys.argv.pop(1)

    config_path = pre.config or pos_config
    config = load_config(config_path)

    # Apply config defaults, prioritizing addition_experiment section
    set_parser_defaults_from_config(parser, config, section="addition_experiment")

    args = parser.parse_args()
    # Store the resolved config path so metadata.json captures it
    if args.config is None and config_path:
        args.config = config_path

    # Standardized configuration printing
    print_config(args, title="Effective Addition Experiment Configuration")
    # Determine which phases to run
    run_all = args.all
    do_prompts = run_all or args.make_prompts
    do_plots = run_all or args.operand_plots

    if not any([do_prompts, do_plots]):
        parser.print_help()
        sys.exit(0)

    _set_seeds(args.seed)

    run_dir = _make_run_dir(args.out_root, args.run_id)
    log.info("Run directory: %s", run_dir)
    _write_metadata(run_dir, args)

    model = None
    if do_plots:
        model = _load_model(args)

    if do_prompts:
        phase_make_prompts(run_dir)

    if do_plots:
        assert model is not None
        phase_operand_plots(run_dir, model, args)

    print(f"\n{'=' * 60}")
    print(f"All requested phases complete.  Outputs in:\n  {run_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
