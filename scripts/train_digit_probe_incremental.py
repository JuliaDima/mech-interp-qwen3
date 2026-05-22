#!/usr/bin/env python3
"""Incrementally train a unit-digit probe across layers.

Trains a linear probe to predict the unit digit (ones place) of a+b from
transcoder activations. Iterates through layers 0..k for k=0,1,...,N-1,
caching all activations upfront and reusing them for each probe.

Reports accuracy at each layer prefix and stops when val accuracy exceeds
the target threshold (default 95%).

Example usage:
    # Run with default settings (all layers, grid dataset, 20 epochs)
    python scripts/train_digit_probe_incremental.py

    # Faster run: fewer epochs, early stopping, smaller dataset
    python scripts/train_digit_probe_incremental.py \\
        --n_epochs 10 --early_stopping_patience 3 \\
        --strategy balanced --n_train 2000

    # Sweep only the first 16 layers
    python scripts/train_digit_probe_incremental.py --max_layers 16

    # Use the 'answer' token position instead of the final token
    python scripts/train_digit_probe_incremental.py --token_position answer

    # Change the accuracy target (e.g. 90%)
    python scripts/train_digit_probe_incremental.py --target_accuracy 0.90

    # Save results to a custom directory
    python scripts/train_digit_probe_incremental.py \\
        --output_dir runs/my_digit_sweep --run_id exp1

    # Full grid, L2 regularization, gradient clipping
    python scripts/train_digit_probe_incremental.py \\
        --strategy grid --max_value 99 \\
        --l2_penalty 1e-4 --gradient_clip 1.0 \\
        --learning_rate 1e-3 --n_epochs 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.addition.dataset_generation.generate_dataset_with_predictions import (  # noqa: E402
    TemplateID,
    build_prompt,
)
from mechinterp_qwen3.attribution_model import AttributionModel  # noqa: E402
from mechinterp_qwen3.probe import (  # noqa: E402
    CarryProbe,
    ProbeTrainer,
    compute_unit_digit_label,
    generate_addition_examples,
)
from mechinterp_qwen3.probe.dataset_utils import ProbeDataset  # noqa: E402
from mechinterp_qwen3.utils.config_utils import (  # noqa: E402
    add_config_args,
    load_config,
    set_parser_defaults_from_config,
)
from mechinterp_qwen3.utils.model_utils import get_default_device  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_digit_probe_incremental")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Incrementally train unit-digit probe across layers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_config_args(p)

    model_group = p.add_argument_group("Model")
    model_group.add_argument("--model", default="Qwen/Qwen3-4B")
    model_group.add_argument("--transcoder_set", default="mwhanna/qwen3-4b-transcoders")
    model_group.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"]
    )
    model_group.add_argument("--device", default=None)

    data_group = p.add_argument_group("Dataset")
    data_group.add_argument(
        "--task",
        default="unit_digit",
        choices=["unit_digit", "carry"],
        help="Probe target: 'unit_digit' (10-class) or 'carry' (binary)",
    )
    data_group.add_argument("--max_value", type=int, default=999)
    data_group.add_argument("--strategy", default="grid", choices=["grid", "balanced", "random"])
    data_group.add_argument("--n_train", type=int, default=None)
    data_group.add_argument("--val_split", type=float, default=0.2)
    data_group.add_argument("--seed", type=int, default=42)
    data_group.add_argument("--template", default="T0", choices=["T0", "T1", "T2"])
    data_group.add_argument(
        "--token_position",
        default="final",
        help="Token position: 'final', 'answer', or integer index",
    )

    train_group = p.add_argument_group("Training")
    train_group.add_argument("--n_epochs", type=int, default=20)
    train_group.add_argument("--batch_size", type=int, default=8)
    train_group.add_argument("--learning_rate", type=float, default=5e-3)
    train_group.add_argument("--l2_penalty", type=float, default=1e-3)
    train_group.add_argument(
        "--pca_dim",
        type=int,
        default=128,
        help="PCA dimensionality per layer before concatenating (0 = no PCA)",
    )
    train_group.add_argument("--gradient_clip", type=float, default=None)
    train_group.add_argument("--early_stopping_patience", type=int, default=5)
    train_group.add_argument(
        "--target_accuracy",
        type=float,
        default=0.95,
        help="Stop when val accuracy exceeds this threshold",
    )
    train_group.add_argument(
        "--max_layers",
        type=int,
        default=None,
        help="Maximum number of layers to sweep (default: all)",
    )
    train_group.add_argument(
        "--start_layer",
        type=int,
        default=0,
        help="First layer to include in the incremental sweep (default: 0)",
    )

    output_group = p.add_argument_group("Output")
    output_group.add_argument("--output_dir", default="runs/digit_probe_incremental")
    output_group.add_argument("--run_id", default=None)

    return p


def main():
    parser = build_parser()
    early_args, _ = parser.parse_known_args()
    config = load_config(early_args.config)
    set_parser_defaults_from_config(parser, config)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S") if args.run_id is None else args.run_id
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {output_dir}")

    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = get_default_device() if args.device is None else torch.device(args.device)
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    log.info(f"Loading model: {args.model}")
    model = AttributionModel.from_pretrained(
        model_name=args.model,
        transcoder_set=args.transcoder_set,
        device=device,
        dtype=dtype,
    )
    n_model_layers = model.cfg.n_layers
    d_transcoder = model.transcoders.d_transcoder
    log.info(f"Model loaded. Layers: {n_model_layers}, d_transcoder: {d_transcoder}")

    max_layers = args.max_layers if args.max_layers is not None else n_model_layers
    all_layers = list(range(args.start_layer, max_layers))

    # --- Dataset generation ---
    log.info("Generating addition examples...")
    operands_a, operands_b, carry_labels = generate_addition_examples(
        max_value=args.max_value,
        n_samples=args.n_train,
        strategy=args.strategy,
        seed=args.seed,
    )
    if args.task == "carry":
        labels = carry_labels
        n_classes = 1  # binary (sigmoid + BCE)
        log.info(f"Generated {len(operands_a)} examples (binary carry), label distribution:")
        for v in (0, 1):
            log.info(f"  carry={v}: {sum(1 for lbl in labels if lbl == v)}")
    else:
        labels = [
            compute_unit_digit_label(a, b) for a, b in zip(operands_a, operands_b, strict=False)
        ]
        n_classes = 10
        log.info(f"Generated {len(operands_a)} examples (unit digit), label distribution:")
        for digit in range(10):
            log.info(f"  digit {digit}: {sum(1 for lbl in labels if lbl == digit)}")

    template_id = getattr(TemplateID, args.template)
    prompts = [
        build_prompt(template_id, a, b) for a, b in zip(operands_a, operands_b, strict=False)
    ]

    n_samples = len(prompts)
    n_val = int(n_samples * args.val_split)
    n_train = n_samples - n_val
    log.info(f"Split: {n_train} train, {n_val} val")

    train_prompts, train_labels = prompts[:n_train], labels[:n_train]
    val_prompts, val_labels = prompts[n_train:], labels[n_train:]

    try:
        token_position = int(args.token_position)
    except ValueError:
        token_position = args.token_position

    # --- Cache activations for ALL layers upfront ---
    log.info(f"Caching activations for all {max_layers} layers (this may take a while)...")
    train_dataset = ProbeDataset(
        prompts=train_prompts,
        labels=train_labels,
        model=model,
        layers=all_layers,
        token_position=token_position,
        cache_activations=True,
    )
    val_dataset = ProbeDataset(
        prompts=val_prompts,
        labels=val_labels,
        model=model,
        layers=all_layers,
        token_position=token_position,
        cache_activations=True,
    )
    log.info("Activations cached.")

    # --- Per-layer PCA (fit on train, apply to train+val) ---
    pca_dim = args.pca_dim
    if pca_dim > 0:
        log.info(f"Fitting PCA (d={pca_dim}) per layer on train activations (randomized SVD)...")
        pca_components: dict[int, torch.Tensor] = {}  # layer -> (pca_dim, d_transcoder)
        pca_means: dict[int, torch.Tensor] = {}
        for layer in all_layers:
            X = train_dataset._cached_activations[layer].float()  # (n_train, d_tc)
            mean = X.mean(0)
            X_c = (X - mean).to(device)  # move to GPU for speed

            # Randomized SVD — only computes top-pca_dim components.
            # Sketch: Y = X_c @ Omega, shape (n_train, pca_dim + oversampling)
            # Much faster than full SVD: O(N * d_tc * k) vs O(N * d_tc * min(N,d_tc))
            n_oversampling = 10
            k = pca_dim + n_oversampling
            Omega = torch.randn(X_c.shape[1], k, device=device, dtype=torch.float32)
            Y = X_c @ Omega  # (n_train, k)
            Q, _ = torch.linalg.qr(Y)  # (n_train, k)
            B = Q.T @ X_c  # (k, d_tc)
            _, _, Vt_B = torch.linalg.svd(B, full_matrices=False)  # Vt_B: (k, d_tc)
            components = Vt_B[:pca_dim].cpu()  # (pca_dim, d_tc)

            pca_components[layer] = components.to(dtype=dtype)
            pca_means[layer] = mean.to(dtype=dtype)
            log.info(f"  Layer {layer} PCA done")

        def _apply_pca(cached: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
            return {
                layer: (cached[layer].to(dtype=torch.float32) - pca_means[layer].cpu()).to(
                    dtype=dtype
                )
                @ pca_components[layer].cpu().T
                for layer in all_layers
            }

        train_dataset._cached_activations = _apply_pca(train_dataset._cached_activations)
        val_dataset._cached_activations = _apply_pca(val_dataset._cached_activations)
        d_probe = pca_dim
        log.info(f"PCA done — probe input dim per layer: {d_probe}")
    else:
        d_probe = d_transcoder

    # --- Incremental training sweep ---
    results = []
    header = f"{'Layers':>12}  {'Val Acc':>8}  {'Val Loss':>9}"
    log.info("")
    log.info("=" * 60)
    task_label = "carry (binary)" if args.task == "carry" else "unit digit (10-class)"
    log.info(f"Incremental layer sweep ({task_label} probe)")
    log.info("=" * 60)
    log.info(header)
    log.info("-" * 60)

    target_reached_at = None

    for k in range(len(all_layers)):
        layers_so_far = all_layers[: k + 1]

        probe = CarryProbe(
            layers=layers_so_far,
            d_transcoder=d_probe,
            device=device,
            dtype=dtype,
            n_classes=n_classes,
        )

        trainer = ProbeTrainer(
            probe=probe,
            learning_rate=args.learning_rate,
            l2_penalty=args.l2_penalty,
            gradient_clip=args.gradient_clip,
            device=device,
        )

        trainer.fit(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            early_stopping_patience=args.early_stopping_patience,
            checkpoint_dir=output_dir / f"layer_{k:02d}" / "checkpoints",
            verbose=False,
        )

        val_loss, val_metrics = trainer.evaluate(val_dataset, batch_size=args.batch_size)
        train_loss, train_metrics = trainer.evaluate(train_dataset, batch_size=args.batch_size)

        row = {
            "layers": layers_so_far,
            "n_layers": k + 1,
            "val_accuracy": val_metrics.accuracy,
            "val_loss": val_loss,
            "train_accuracy": train_metrics.accuracy,
            "train_loss": train_loss,
        }
        results.append(row)

        flag = ""
        if val_metrics.accuracy >= args.target_accuracy and target_reached_at is None:
            target_reached_at = k + 1
            flag = f"  <-- TARGET {args.target_accuracy:.0%} REACHED"

        log.info(
            f"  0..{k:2d} ({k + 1:2d} layers):  {val_metrics.accuracy:.4f}  {val_loss:.4f}{flag}"
        )

        # Save per-layer summary
        with open(output_dir / f"layer_{k:02d}" / "result.json", "w") as f:
            json.dump(row, f, indent=2)

        if target_reached_at is not None:
            # Continue to finish reporting but stop early if user set max_layers
            pass  # report all layers up to max_layers

    log.info("=" * 60)
    if target_reached_at is not None:
        log.info(
            f"Target accuracy {args.target_accuracy:.0%} first reached at "
            f"{target_reached_at} layer(s) (layers 0..{target_reached_at - 1})"
        )
    else:
        best = max(results, key=lambda r: r["val_accuracy"])
        log.info(
            f"Target {args.target_accuracy:.0%} not reached. "
            f"Best val accuracy: {best['val_accuracy']:.4f} at {best['n_layers']} layer(s)."
        )

    # Save full results table
    summary = {
        "run_id": run_id,
        "target_accuracy": args.target_accuracy,
        "target_reached_at_n_layers": target_reached_at,
        "results": results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print compact table
    print("\n" + "=" * 60)
    print(f"{'N layers':>10}  {'Val Acc':>8}  {'Train Acc':>10}")
    print("-" * 60)
    for r in results:
        marker = " *" if r["n_layers"] == target_reached_at else "  "
        print(
            f"{r['n_layers']:>10}  {r['val_accuracy']:>8.4f}  {r['train_accuracy']:>10.4f}{marker}"
        )
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
