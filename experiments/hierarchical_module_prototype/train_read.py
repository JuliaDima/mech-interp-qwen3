"""Stage 1b: Train CrossLayerRead interface with frozen BiGRU primitive.

Loads frozen CarryPrimitiveGRU from Stage 1a, then trains CrossLayerRead
(W_read, q, k^(l)) so it maps Qwen residual streams at each token position
to d_small-dimensional vectors that the frozen BiGRU can process.

Loss objective:
    - Extract BiGRU output f at the last prompt position.
    - Apply CarryHead(d_small → n_digits) to predict all n_digits carry labels.
    - Binary cross-entropy per digit position.

The head has only d_small * n_digits + n_digits parameters (~512 for n_digits=4),
so CrossLayerRead's ~422K parameters dominate the learning signal.

Gradient flow:
    loss → CarryHead → f_last → primitive (no .grad on frozen params) → x → CrossLayerRead ✓

One Qwen forward pass per step (collect_residuals, no_grad) +
one module forward with autograd.

Outputs saved to --out_dir/:
    read.pt       – CrossLayerRead state_dict
    meta.json     – Stage 1a metadata + {model, n_layers, d_model, d_vocab, dtype}
    train_log.json

Run:
    python -m experiments.hierarchical_module_prototype.train_read \\
        --stage1a_dir runs/hierarchical_module/stage1a
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.addition.dataset_generation.generate_dataset_with_predictions import (  # noqa: E402
    TEMPLATES,
    TemplateID,
)
from experiments.hierarchical_module_prototype.model import (  # noqa: E402
    CarryHead,
    CarryPrimitiveGRU,
    CrossLayerRead,
    DigitSlotAttention,
)
from experiments.hierarchical_module_prototype.utils import (  # noqa: E402
    build_carry_batch,
    build_prompt_batch,
    collect_residuals,
    generate_hard_regime_pairs,
)
from mechinterp_qwen3.utils.config_utils import (  # noqa: E402
    add_config_args,
    load_config,
    print_config,
    set_parser_defaults_from_config,
)
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype  # noqa: E402
from mechinterp_qwen3.utils_seed import seed_everything  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hmp.train_read")

_ALL_TEMPLATE_IDS = [t.value for t in TemplateID]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 1b: train CrossLayerRead with frozen BiGRU on Qwen residuals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_args(p)

    # Stage 1a checkpoint (required)
    p.add_argument(
        "--stage1a_dir",
        required=True,
        help="Directory from train_primitive.py (contains primitive.pt, meta.json)",
    )

    # Model
    p.add_argument("--model", default="Qwen/Qwen3-4B", help="HuggingFace model name")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])

    # Dataset (n_digits overrides Stage 1a meta if provided)
    p.add_argument(
        "--n_digits",
        type=int,
        default=None,
        help="Digit count for operands (defaults to Stage 1a meta.json)",
    )
    p.add_argument("--n_samples", type=int, default=8000)
    p.add_argument("--held_out_fraction", type=float, default=0.15)
    p.add_argument(
        "--templates",
        nargs="+",
        default=["T0"],
        choices=_ALL_TEMPLATE_IDS,
        help="Template IDs to sample from per batch. Use multiple to force "
        "template-independent digit extraction (e.g. --templates T0 T1 T2)",
    )

    # Optimisation
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="One Qwen forward pass per step — larger batches use more memory",
    )
    p.add_argument(
        "--ref_layer",
        type=int,
        default=0,
        help="Qwen layer whose residual is used as the per-token query in CrossLayerRead",
    )
    p.add_argument(
        "--static_cross_attn",
        action="store_true",
        help="Use a static (token-independent) query in CrossLayerRead instead of input-dependent",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--lr_primitive",
        type=float,
        default=1e-5,
        help="LR for fine-tuning the BiGRU primitive (set to 0 to keep frozen)",
    )
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # I/O
    p.add_argument(
        "--out_dir",
        default="runs/hierarchical_module/stage1b",
        help="Directory for CrossLayerRead checkpoint and logs",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)

    return p


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_default_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage 1a metadata
    stage1a_dir = Path(args.stage1a_dir)
    with open(stage1a_dir / "meta.json") as f:
        meta_a = json.load(f)

    d_small = meta_a["d_small"]
    n_digits = args.n_digits or meta_a["n_digits"]
    dtype = parse_dtype(args.dtype)
    template_strs = [TEMPLATES[TemplateID(t)] for t in args.templates]
    log.info("Templates (%d): %s", len(template_strs), args.templates)

    log.info(
        "Stage 1b — model=%s  n_digits=%d  d_small=%d  ref_layer=%d",
        args.model,
        n_digits,
        d_small,
        args.ref_layer,
    )

    # --- Load frozen Qwen ---
    log.info("Loading frozen Qwen: %s (dtype=%s)", args.model, args.dtype)
    from transformer_lens import HookedTransformer

    qwen = HookedTransformer.from_pretrained(
        args.model,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        dtype=dtype,
    )
    qwen.requires_grad_(False)
    qwen.eval()

    n_layers = qwen.cfg.n_layers
    d_model = qwen.cfg.d_model
    d_vocab = qwen.cfg.d_vocab_out
    log.info("Model loaded: n_layers=%d  d_model=%d  d_vocab=%d", n_layers, d_model, d_vocab)

    # --- Load Stage 1a primitive (optionally fine-tunable) ---
    primitive = CarryPrimitiveGRU(d_small).to(device=device, dtype=dtype)
    primitive.load_state_dict(torch.load(stage1a_dir / "primitive.pt", map_location=device))
    finetune_primitive = args.lr_primitive > 0
    if not finetune_primitive:
        primitive.requires_grad_(False)
    log.info(
        "Primitive loaded from %s  (fine-tune=%s  lr_primitive=%.1e)",
        stage1a_dir,
        finetune_primitive,
        args.lr_primitive,
    )

    # --- Build trainable CrossLayerRead + DigitSlotAttention + CarryHead ---
    input_dependent = not args.static_cross_attn
    log.info("CrossLayerRead mode: %s", "input-dependent" if input_dependent else "static")
    read = CrossLayerRead(
        n_layers,
        d_model,
        d_small,
        ref_layer=args.ref_layer,
        input_dependent=input_dependent,
    ).to(device=device, dtype=dtype)
    slot_attn = DigitSlotAttention(n_digits, d_small).to(device=device, dtype=dtype)
    carry_head = CarryHead(d_small, n_out=1).to(device=device, dtype=dtype)

    n_read = sum(p.numel() for p in read.parameters())
    n_slot = sum(p.numel() for p in slot_attn.parameters())
    n_head = sum(p.numel() for p in carry_head.parameters())
    n_prim = sum(p.numel() for p in primitive.parameters())
    log.info(
        "Trainable Stage 1b parameters: read=%d  slot_attn=%d  head=%d  primitive=%d (lr=%.1e)",
        n_read,
        n_slot,
        n_head,
        n_prim if finetune_primitive else 0,
        args.lr_primitive,
    )

    param_groups = [
        {
            "params": list(read.parameters())
            + list(slot_attn.parameters())
            + list(carry_head.parameters()),
            "lr": args.lr,
        },
    ]
    if finetune_primitive:
        param_groups.append({"params": list(primitive.parameters()), "lr": args.lr_primitive})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    all_trainable = [p for g in param_groups for p in g["params"]]

    train_pairs = generate_hard_regime_pairs(
        n_digits,
        args.n_samples,
        seed=args.seed,
        held_out=False,
        held_out_fraction=args.held_out_fraction,
    )
    val_pairs = generate_hard_regime_pairs(
        n_digits,
        args.n_samples,
        seed=args.seed,
        held_out=True,
        held_out_fraction=args.held_out_fraction,
    )

    steps_per_epoch = math.ceil(len(train_pairs) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log.info(
        "Training Stage 1b: %d pairs × %d epochs = %d steps  (batch=%d)",
        len(train_pairs),
        args.epochs,
        total_steps,
        args.batch_size,
    )
    log.info("NOTE: one Qwen forward pass per step (residual collection, no_grad).")

    history: list[dict] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        import random as _random

        _random.shuffle(train_pairs)

        read.train()
        slot_attn.train()
        carry_head.train()
        primitive.train() if finetune_primitive else primitive.eval()
        epoch_losses: list[float] = []
        epoch_accs: list[float] = []

        for batch_start in range(0, len(train_pairs), args.batch_size):
            batch_pairs = train_pairs[batch_start : batch_start + args.batch_size]
            if not batch_pairs:
                continue

            # Tokenise prompts (no answers needed) — sample template per example
            prompt_tokens, _ = build_prompt_batch(
                batch_pairs, template_strs, qwen.tokenizer, device
            )

            # Carry labels from arithmetic ground truth
            _pair_idx, carry_labels = build_carry_batch(batch_pairs, n_digits, device)

            # ── Collect Qwen residuals (no grad, detached from Qwen graph) ───
            residuals = collect_residuals(qwen, prompt_tokens)  # (n_layers, B, T, d_model)
            residuals = residuals.to(device=device, dtype=dtype)

            # ── CrossLayerRead → DigitSlotAttention → BiGRU (autograd active) ─
            x = read(residuals)  # (B, T, d_small)
            x_slots, _ = slot_attn(x)  # (B, n_digits, d_small) — slots attend over tokens
            f = primitive(x_slots)  # (B, n_digits, d_small)

            logits = carry_head(f).squeeze(-1)  # (B, n_digits)
            loss = F.binary_cross_entropy_with_logits(logits, carry_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_trainable, args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            with torch.no_grad():
                acc = ((logits > 0).float() == carry_labels).float().mean().item()

            epoch_losses.append(loss.item())
            epoch_accs.append(acc)

            if global_step % args.log_every == 0:
                log.info(
                    "epoch=%d  step=%d  loss=%.4f  carry_acc=%.4f  lr=%.2e",
                    epoch,
                    global_step,
                    loss.item(),
                    acc,
                    scheduler.get_last_lr()[0],
                )

        mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        mean_acc = sum(epoch_accs) / max(1, len(epoch_accs))

        # Validation
        read.eval()
        slot_attn.eval()
        carry_head.eval()
        primitive.eval()
        with torch.no_grad():
            val_tokens, _ = build_prompt_batch(
                val_pairs[:64], template_strs, qwen.tokenizer, device
            )
            _val_idx, val_labels = build_carry_batch(val_pairs[:64], n_digits, device)
            val_res = collect_residuals(qwen, val_tokens).to(device=device, dtype=dtype)
            val_x = read(val_res)
            val_x_slots, _ = slot_attn(val_x)
            val_f = primitive(val_x_slots)
            val_logits = carry_head(val_f).squeeze(-1)
            val_acc = ((val_logits > 0).float() == val_labels).float().mean().item()

        history.append(
            {
                "epoch": epoch,
                "mean_loss": mean_loss,
                "train_carry_acc": mean_acc,
                "val_carry_acc": val_acc,
            }
        )
        log.info(
            "Epoch %d complete — loss=%.4f  train_acc=%.4f  val_acc=%.4f",
            epoch,
            mean_loss,
            mean_acc,
            val_acc,
        )

    # --- Save ---
    torch.save(read.state_dict(), out_dir / "read.pt")
    torch.save(slot_attn.state_dict(), out_dir / "slot_attn.pt")
    torch.save(carry_head.state_dict(), out_dir / "carry_head.pt")
    if finetune_primitive:
        torch.save(primitive.state_dict(), out_dir / "primitive.pt")
        log.info("  primitive.pt — fine-tuned BiGRU weights (use in Stage 2 instead of Stage 1a)")

    meta = {
        "model": args.model,
        "n_layers": n_layers,
        "d_model": d_model,
        "d_small": d_small,
        "d_vocab": d_vocab,
        "n_digits": n_digits,
        "dtype": args.dtype,
        "seed": args.seed,
        "ref_layer": args.ref_layer,
        "input_dependent_cross_attn": input_dependent,
        "templates": args.templates,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(out_dir / "train_log.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info("Stage 1b complete. Checkpoint written to %s", out_dir)
    log.info("  read.pt       — CrossLayerRead weights")
    log.info("  slot_attn.pt  — DigitSlotAttention weights")
    log.info("  meta.json     — full architecture metadata for Stage 2")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    pre, _ = parser.parse_known_args()
    config = load_config(pre.config)
    set_parser_defaults_from_config(parser, config)
    args = parser.parse_args()
    print_config(args, title="Stage 1b Training Configuration")
    train(args)


if __name__ == "__main__":
    main()
