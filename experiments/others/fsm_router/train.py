"""Train the FSM primitive router on synthetic math expressions.

The router learns to detect which arithmetic primitive(s) are present
at each token position, using only predicate-level token abstractions.
Because the input is predicates (not raw token IDs), the router
generalises OOD: any unseen number maps to NUMBER, any unseen operator
maps to the right predicate.

Loss: BCE per (token, primitive) position, averaged over valid positions.

Run:
    python -m experiments.fsm_router.train
    python -m experiments.fsm_router.train --n_samples 10000 --epochs 50
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

from experiments.fsm_router.fsm import PrimitiveRouter  # noqa: E402
from experiments.fsm_router.predicates import N_PREDICATES, tokenize_and_map  # noqa: E402
from experiments.fsm_router.primitives import (  # noqa: E402
    FSM_SPECS,
    PRIMITIVE_DEFS,
    generate_expressions,
)
from mechinterp_qwen3.utils_seed import seed_everything  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fsm_router.train")


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------


def build_batch(
    examples: list[tuple[str, list[list[float]]]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Collate a list of (text, label_matrix) into padded tensors.

    Returns:
        pred_ids: (B, max_T)      long  — predicate IDs, right-padded with OTHER
        labels:   (B, max_T, K)   float — per-token per-primitive targets
        lengths:  list[int]              — actual token counts per example
    """
    K = len(PRIMITIVE_DEFS)
    pred_seqs, label_seqs, lengths = [], [], []

    for text, label_matrix in examples:
        pairs = tokenize_and_map(text)
        pids = [int(p) for _, p in pairs]
        T = len(pids)
        # label_matrix: K × T  →  transpose to T × K
        labs = [[label_matrix[k][t] for k in range(K)] for t in range(T)]
        pred_seqs.append(pids)
        label_seqs.append(labs)
        lengths.append(T)

    max_T = max(lengths)
    B = len(examples)

    pred_ids = torch.zeros(B, max_T, dtype=torch.long, device=device)
    labels = torch.zeros(B, max_T, K, dtype=torch.float, device=device)

    for i, (pids, labs) in enumerate(zip(pred_seqs, label_seqs, strict=False)):
        T = lengths[i]
        pred_ids[i, :T] = torch.tensor(pids, dtype=torch.long, device=device)
        labels[i, :T] = torch.tensor(labs, dtype=torch.float, device=device)

    return pred_ids, labels, lengths


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------


def loss_and_metrics(
    activations: torch.Tensor,  # (B, T, K) — already sigmoid'd
    labels: torch.Tensor,  # (B, T, K)
    lengths: list[int],
) -> tuple[torch.Tensor, dict[str, float]]:
    B, max_T, K = activations.shape
    device = activations.device

    # Boolean mask for valid (non-padded) positions
    mask = torch.zeros(B, max_T, dtype=torch.bool, device=device)
    for i, l in enumerate(lengths):
        mask[i, :l] = True

    # BCE (activations are already in (0,1))
    a = activations.clamp(1e-6, 1 - 1e-6)
    bce = -(labels * a.log() + (1 - labels) * (1 - a).log())  # (B, T, K)
    loss = bce[mask.unsqueeze(-1).expand_as(bce)].mean()

    # Per-primitive accuracy on valid positions
    preds = (activations > 0.5).float()
    metrics: dict[str, float] = {}
    for k, pdef in enumerate(PRIMITIVE_DEFS):
        valid_k = mask  # (B, T)
        acc = (preds[..., k][valid_k] == labels[..., k][valid_k]).float().mean().item()
        metrics[f"{pdef.name}_acc"] = acc

    return loss, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Primitives: %s", [p.name for p in PRIMITIVE_DEFS])

    router = PrimitiveRouter(FSM_SPECS, N_PREDICATES).to(device)
    n_params = sum(p.numel() for p in router.parameters())
    log.info("Router: %d parameters  (%d FSMs)", n_params, len(FSM_SPECS))

    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    all_data = generate_expressions(args.n_samples, seed=args.seed)
    split = int(len(all_data) * (1 - args.held_out_fraction))
    train_data, val_data = all_data[:split], all_data[split:]

    steps_per_epoch = math.ceil(len(train_data) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log.info(
        "Training: %d / %d  (train/val)  epochs=%d  batch=%d",
        len(train_data),
        len(val_data),
        args.epochs,
        args.batch_size,
    )

    history: list[dict] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        _random.shuffle(train_data)
        router.train()
        epoch_losses: list[float] = []

        for start in range(0, len(train_data), args.batch_size):
            batch = train_data[start : start + args.batch_size]
            if not batch:
                continue

            pred_ids, labels, lengths = build_batch(batch, device)
            acts = router(pred_ids)
            loss, _ = loss_and_metrics(acts, labels, lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), args.grad_clip)
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

        router.eval()
        with torch.no_grad():
            vpids, vlabels, vlengths = build_batch(val_data, device)
            vacts = router(vpids)
            _, val_metrics = loss_and_metrics(vacts, vlabels, vlengths)

        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
        log.info("Epoch %d — loss=%.4f  %s", epoch, mean_loss, metrics_str)
        history.append({"epoch": epoch, "mean_loss": mean_loss, **val_metrics})

    torch.save(router.state_dict(), out_dir / "router.pt")
    with open(out_dir / "meta.json", "w") as f:
        json.dump(
            {
                "primitives": [p.name for p in PRIMITIVE_DEFS],
                "fsm_specs": FSM_SPECS,
                "n_predicates": N_PREDICATES,
                "n_samples": args.n_samples,
                "seed": args.seed,
            },
            f,
            indent=2,
        )
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info("Saved to %s", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--n_samples", type=int, default=8000)
    p.add_argument("--held_out_fraction", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--out_dir", default="runs/fsm_router")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=50)
    return p


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
