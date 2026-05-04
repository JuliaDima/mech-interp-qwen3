"""Stage 1a: Train a sequential primitive in isolation — no Qwen.

Pipeline:
    embedding (primitive-specific) → CarryPrimitiveGRU (shared) → head (primitive-specific)

The GRU architecture is identical for every primitive; only the embedding
vocabulary, head output size, and loss function change.  Adding a new
primitive means adding a subclass to primitives.py.

Outputs saved to --out_dir/:
    primitive.pt      – CarryPrimitiveGRU state_dict
    embedding.pt      – primitive embedding state_dict
    head.pt           – prediction head state_dict
    meta.json         – {primitive, n_digits, d_small, seed}
    train_log.json    – per-epoch metrics

Run:
    python -m experiments.hierarchical_module_prototype.train_primitive --primitive carry
    python -m experiments.hierarchical_module_prototype.train_primitive --primitive palindrome \\
        --n_digits 2 --epochs 60 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random as _random
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.hierarchical_module_prototype.model import CarryPrimitiveGRU  # noqa: E402
from experiments.hierarchical_module_prototype.primitives import PRIMITIVES  # noqa: E402
from mechinterp_qwen3.utils.config_utils import (  # noqa: E402
    add_config_args,
    load_config,
    print_config,
    set_parser_defaults_from_config,
)
from mechinterp_qwen3.utils.model_utils import get_default_device  # noqa: E402
from mechinterp_qwen3.utils_seed import seed_everything  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hmp.train_primitive")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 1a: train a sequential primitive in isolation (no Qwen)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_args(p)

    p.add_argument(
        "--primitive",
        default="carry",
        choices=list(PRIMITIVES),
        help="Which primitive to train",
    )
    p.add_argument("--d_small", type=int, default=128, help="Module internal dimension")
    p.add_argument("--n_digits", type=int, default=4, help="Digit count per operand")
    p.add_argument("--n_samples", type=int, default=8000)
    p.add_argument("--held_out_fraction", type=float, default=0.15)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--out_dir", default="runs/hierarchical_module/stage1a", help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=20)

    return p


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_default_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    primitive = PRIMITIVES[args.primitive]
    log.info("Primitive: %s  n_digits=%d  d_small=%d", primitive.name, args.n_digits, args.d_small)

    embedding = primitive.make_embedding(args.d_small).to(device)
    gru = CarryPrimitiveGRU(args.d_small).to(device)
    head = primitive.make_head(args.d_small).to(device)

    params = list(embedding.parameters()) + list(gru.parameters()) + list(head.parameters())
    n_params = sum(p.numel() for p in params)
    log.info("Trainable parameters: %d  (embedding + GRU + head)", n_params)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    train_data = primitive.generate_data(
        args.n_digits,
        args.n_samples,
        args.seed,
        held_out=False,
        held_out_fraction=args.held_out_fraction,
    )
    val_data = primitive.generate_data(
        args.n_digits,
        args.n_samples,
        args.seed,
        held_out=True,
        held_out_fraction=args.held_out_fraction,
    )

    steps_per_epoch = math.ceil(len(train_data) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log.info(
        "Training: %d samples × %d epochs = %d steps  (batch=%d)",
        len(train_data),
        args.epochs,
        total_steps,
        args.batch_size,
    )

    history: list[dict] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        _random.shuffle(train_data)
        embedding.train()
        gru.train()
        head.train()
        epoch_losses: list[float] = []

        for start in range(0, len(train_data), args.batch_size):
            batch = train_data[start : start + args.batch_size]
            if not batch:
                continue

            input_indices, labels = primitive.build_batch(batch, args.n_digits, device)
            x = embedding(input_indices)  # (B, seq, d_small)
            f = gru(x)  # (B, seq, d_small)
            logits = head(f)  # (B, seq, n_out)

            loss = primitive.loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_losses.append(loss.item())
            if global_step % args.log_every == 0:
                log.info(
                    "epoch=%d  step=%d  loss=%.4f  lr=%.2e",
                    epoch,
                    global_step,
                    loss.item(),
                    scheduler.get_last_lr()[0],
                )

        mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))

        embedding.eval()
        gru.eval()
        head.eval()
        with torch.no_grad():
            val_indices, val_labels = primitive.build_batch(val_data, args.n_digits, device)
            val_x = embedding(val_indices)
            val_f = gru(val_x)
            val_logits = head(val_f)
            val_metrics = primitive.metric(val_logits, val_labels)

        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
        log.info("Epoch %d — loss=%.4f  %s", epoch, mean_loss, metrics_str)
        history.append({"epoch": epoch, "mean_loss": mean_loss, **val_metrics})

    # --- Save ---
    torch.save(gru.state_dict(), out_dir / "primitive.pt")
    torch.save(embedding.state_dict(), out_dir / "embedding.pt")
    torch.save(head.state_dict(), out_dir / "head.pt")

    meta = {
        "primitive": args.primitive,
        "n_digits": args.n_digits,
        "d_small": args.d_small,
        "seed": args.seed,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info("Done. Checkpoints written to %s", out_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    pre, _ = parser.parse_known_args()
    config = load_config(pre.config)
    set_parser_defaults_from_config(parser, config)
    args = parser.parse_args()
    print_config(args, title="Primitive Training Configuration")
    train(args)


if __name__ == "__main__":
    main()
