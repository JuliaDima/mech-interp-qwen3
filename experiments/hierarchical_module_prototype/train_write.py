"""Stage 2 training: carry-gated, slot-routed write.

Loads frozen CrossLayerRead, DigitSlotAttention, CarryPrimitiveGRU from Stage 1b/1a,
then trains CrossLayerWrite (carry_gate, W_write, layer distribution) against frozen
Qwen on hard-regime addition problems.

Loss:
    L = L_CE + λ_carry · BCE(carry_logits, carry_labels)

The carry supervision forces the per-slot gate to predict actual carries, breaking
the rank-1 steering shortcut: scrambling f now scrambles gating + content together.

Outputs saved to --out_dir/:
    module.pt      – full PrototypeModule state_dict (read+primitive+write)
    meta.json      – architecture metadata (Stage 1b meta + Stage 2 hparams)
    train_log.json – per-epoch metrics

Run:
    python -m experiments.hierarchical_module_prototype.train_write \\
        --stage1a_dir runs/hierarchical_module/stage1a \\
        --stage1b_dir runs/hierarchical_module/stage1b
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
    PrototypeModule,
)
from experiments.hierarchical_module_prototype.utils import (  # noqa: E402
    build_carry_batch,
    build_training_batch,
    collect_residuals,
    compute_ce_on_answer_positions,
    generate_hard_regime_pairs,
    make_write_hooks,
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
log = logging.getLogger("hmp.train_write")

TEMPLATE_STR = TEMPLATES[TemplateID.T0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 2: train scalar gate + cross-layer write on hard-regime addition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_args(p)

    # Stage checkpoints (both required)
    p.add_argument(
        "--stage1a_dir",
        required=True,
        help="Directory from train_primitive.py (contains primitive.pt, meta.json)",
    )
    p.add_argument(
        "--stage1b_dir",
        required=True,
        help="Directory from train_read.py (contains read.pt, meta.json with full arch info)",
    )

    # Model (override meta.json if needed)
    p.add_argument("--model", default=None, help="HuggingFace model name (defaults to meta.json)")
    p.add_argument("--dtype", default=None, choices=["float32", "bfloat16", "float16"])

    # Dataset
    p.add_argument(
        "--n_digits",
        type=int,
        default=None,
        help="Digit count for operands (defaults to meta.json)",
    )
    p.add_argument("--n_samples", type=int, default=8000)
    p.add_argument("--held_out_fraction", type=float, default=0.15)

    # Optimisation
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Use small batch (2–4) — Stage 2 runs two Qwen forward passes per step",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument(
        "--lambda_carry",
        type=float,
        default=1.0,
        help="Carry supervision coefficient. BCE(carry_logits, carry_labels) forces "
        "the per-slot gate to predict actual carries, breaking the rank-1 shortcut",
    )

    # I/O
    p.add_argument(
        "--out_dir",
        default="runs/hierarchical_module/stage2",
        help="Directory for full module checkpoint and logs (module.pt + meta.json)",
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

    # Load Stage 1b metadata (has full architecture info: n_layers, d_model, d_small, …)
    stage1a_dir = Path(args.stage1a_dir)
    stage1b_dir = Path(args.stage1b_dir)
    with open(stage1b_dir / "meta.json") as f:
        meta = json.load(f)

    model_name = args.model or meta["model"]
    dtype_str = args.dtype or meta["dtype"]
    dtype = parse_dtype(dtype_str)
    n_layers = meta["n_layers"]
    d_model = meta["d_model"]
    d_small = meta["d_small"]
    n_digits = args.n_digits or meta["n_digits"]
    ref_layer = meta.get("ref_layer", 0)
    input_dependent = meta.get("input_dependent_cross_attn", True)

    log.info(
        "Stage 2 — model=%s  n_digits=%d  d_small=%d  λ_carry=%.3f",
        model_name,
        n_digits,
        d_small,
        args.lambda_carry,
    )

    # --- Load frozen Qwen ---
    log.info("Loading frozen Qwen: %s (dtype=%s)", model_name, dtype_str)
    from transformer_lens import HookedTransformer

    qwen = HookedTransformer.from_pretrained(
        model_name,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        dtype=dtype,
    )
    qwen.requires_grad_(False)
    qwen.eval()

    # --- Load frozen Stage 1 components ---
    # Prefer fine-tuned primitive from Stage 1b if present (saved when --lr_primitive > 0)
    primitive_path = stage1b_dir / "primitive.pt"
    if not primitive_path.exists():
        primitive_path = stage1a_dir / "primitive.pt"
    primitive = CarryPrimitiveGRU(d_small).to(device=device, dtype=dtype)
    primitive.load_state_dict(torch.load(primitive_path, map_location=device))
    primitive.requires_grad_(False)
    primitive.eval()
    log.info("Frozen primitive loaded from %s", primitive_path)

    read = CrossLayerRead(
        n_layers, d_model, d_small, ref_layer=ref_layer, input_dependent=input_dependent
    ).to(device=device, dtype=dtype)
    read.load_state_dict(torch.load(stage1b_dir / "read.pt", map_location=device))
    read.requires_grad_(False)
    read.eval()
    log.info("Frozen read interface loaded from %s", stage1b_dir)

    slot_attn = DigitSlotAttention(n_digits, d_small).to(device=device, dtype=dtype)
    slot_attn.load_state_dict(torch.load(stage1b_dir / "slot_attn.pt", map_location=device))
    slot_attn.requires_grad_(False)
    slot_attn.eval()
    log.info("Frozen slot attention loaded from %s", stage1b_dir)

    # --- Build full PrototypeModule (write trainable, read+slot_attn+primitive frozen) ---
    module = PrototypeModule(
        n_layers,
        d_model,
        d_small,
        n_digits=n_digits,
        ref_layer=ref_layer,
        input_dependent=input_dependent,
    ).to(device=device, dtype=dtype)
    # Overwrite frozen Stage 1 components
    module.read.load_state_dict(read.state_dict())
    module.slot_attn.load_state_dict(slot_attn.state_dict())
    module.primitive.load_state_dict(primitive.state_dict())
    module.read.requires_grad_(False)
    module.slot_attn.requires_grad_(False)
    module.primitive.requires_grad_(False)

    # Warm-start carry_gate from Stage 1b's carry_head (same Linear(d_small,1) architecture).
    # Zero-weight init causes a gradient deadlock where BCE only moves the bias, not the
    # weight vector. Stage 1b's carry_head already encodes the 93%-accurate carry direction.
    carry_head_path = stage1b_dir / "carry_head.pt"
    if carry_head_path.exists():
        carry_head_ckpt = CarryHead(d_small, n_out=1).to(device=device, dtype=dtype)
        carry_head_ckpt.load_state_dict(torch.load(carry_head_path, map_location=device))
        with torch.no_grad():
            module.write.carry_gate.weight.copy_(carry_head_ckpt.linear.weight)
            module.write.carry_gate.bias.copy_(carry_head_ckpt.linear.bias)
        log.info("carry_gate warm-started from Stage 1b carry_head (%s)", carry_head_path)
    else:
        log.warning(
            "carry_head.pt not found in %s — carry_gate uses zero-weight init "
            "(re-run Stage 1b to generate it)",
            stage1b_dir,
        )

    trainable_params = list(module.write.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    log.info("Trainable Stage 2 parameters: %d  (carry_gate + W_write + layer_dist)", n_trainable)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    train_pairs = generate_hard_regime_pairs(
        n_digits,
        args.n_samples,
        seed=args.seed,
        held_out=False,
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
        "Training Stage 2: %d pairs × %d epochs = %d steps  (batch=%d)",
        len(train_pairs),
        args.epochs,
        total_steps,
        args.batch_size,
    )
    log.info("NOTE: each step runs TWO Qwen forward passes (residual collection + write pass).")

    history: list[dict] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        import random as _random

        _random.shuffle(train_pairs)

        epoch_losses: list[float] = []
        epoch_ce_losses: list[float] = []
        epoch_carry_losses: list[float] = []
        epoch_mean_gates: list[float] = []

        for batch_start in range(0, len(train_pairs), args.batch_size):
            batch_pairs = train_pairs[batch_start : batch_start + args.batch_size]
            if not batch_pairs:
                continue

            # Tokenise
            tokens, prompt_lens, answer_lens = build_training_batch(
                batch_pairs, TEMPLATE_STR, qwen.tokenizer, device
            )

            # ── Pass 1 (no_grad): collect Qwen residuals ──────────────────────
            residuals = collect_residuals(qwen, tokens)  # (n_layers, B, T, d)
            residuals = residuals.to(device=device, dtype=dtype)

            # ── Carry labels from arithmetic ground truth ──────────────────────
            _pair_idx, carry_labels = build_carry_batch(batch_pairs, n_digits, device)
            carry_labels = carry_labels.to(dtype=dtype)  # (B, n_digits)

            # ── Run frozen read + slot_attn + primitive on PROMPT TOKENS ONLY ──
            # Slice to prompt-only so f matches Stage 1b's distribution (slot_attn
            # was trained on prompt-only sequences; answer tokens shift attention
            # distribution and corrupt the warm-started carry_gate weights).
            # At inference time the module also sees only prompt tokens before
            # generating, so this is the correct inductive bias.
            max_prompt_len = max(prompt_lens)
            residuals_prompt = residuals[:, :, :max_prompt_len, :]  # (n_layers, B, P, d)
            with torch.no_grad():
                x = module.read(residuals_prompt)  # (B, P, d_small)
                x_slots, slot_attn_weights = module.slot_attn(
                    x
                )  # (B, n_digits, d_small), (B, n_digits, P)
                f = module.primitive(x_slots)  # (B, n_digits, d_small)
            # f, slot_attn_weights are detached; grads flow through write params only

            # ── Trainable write (carry_gate + W_write + layer dist) ───────────
            # deltas are over prompt positions; pad with zeros for answer positions
            deltas_prompt, carry_logits = module.write(
                f, slot_attn_weights
            )  # (n_layers,B,P,d), (B,n,1)
            T_full = tokens.shape[1]
            if T_full > max_prompt_len:
                pad = torch.zeros(
                    n_layers,
                    len(prompt_lens),
                    T_full - max_prompt_len,
                    d_model,
                    device=device,
                    dtype=dtype,
                )
                deltas = torch.cat([deltas_prompt, pad], dim=2)
            else:
                deltas = deltas_prompt

            # ── Pass 2: Qwen with deltas injected (gradients through deltas) ──
            write_hooks = make_write_hooks(deltas, n_layers)
            logits = qwen.run_with_hooks(tokens, fwd_hooks=write_hooks)  # (B, T, vocab)

            # ── Loss: CE on answers + carry supervision on gates ──────────────
            loss_ce = compute_ce_on_answer_positions(logits, tokens, prompt_lens, answer_lens)
            loss_carry = F.binary_cross_entropy_with_logits(
                carry_logits.squeeze(-1),  # (B, n_digits)
                carry_labels,  # (B, n_digits) — 0/1 per digit slot
            )
            loss = loss_ce + args.lambda_carry * loss_carry

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            mean_gate = carry_logits.detach().sigmoid().mean().item()
            epoch_losses.append(loss.item())
            epoch_ce_losses.append(loss_ce.item())
            epoch_carry_losses.append(loss_carry.item())
            epoch_mean_gates.append(mean_gate)

            if global_step % args.log_every == 0:
                log.info(
                    "epoch=%d  step=%d  loss=%.4f  ce=%.4f  carry_bce=%.4f  mean_gate=%.4f  lr=%.2e",
                    epoch,
                    global_step,
                    loss.item(),
                    loss_ce.item(),
                    loss_carry.item(),
                    mean_gate,
                    scheduler.get_last_lr()[0],
                )

        mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        mean_gate = sum(epoch_mean_gates) / max(1, len(epoch_mean_gates))
        history.append(
            {
                "epoch": epoch,
                "mean_loss": mean_loss,
                "mean_ce_loss": sum(epoch_ce_losses) / max(1, len(epoch_ce_losses)),
                "mean_carry_bce": sum(epoch_carry_losses) / max(1, len(epoch_carry_losses)),
                "mean_gate": mean_gate,
            }
        )
        log.info(
            "Epoch %d complete — loss=%.4f  carry_bce=%.4f  mean_gate=%.4f",
            epoch,
            mean_loss,
            sum(epoch_carry_losses) / max(1, len(epoch_carry_losses)),
            mean_gate,
        )

        # Checkpoint every 5 epochs so a killed job doesn't lose all progress
        if epoch % 5 == 0:
            torch.save(module.state_dict(), out_dir / f"module_ep{epoch:03d}.pt")
            log.info("Checkpoint saved: module_ep%03d.pt", epoch)

    # --- Save full module ---
    torch.save(module.state_dict(), out_dir / "module.pt")

    stage2_meta = {
        **meta,
        "stage2": {
            "lambda_carry": args.lambda_carry,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "n_digits": n_digits,
            "n_samples": args.n_samples,
        },
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(stage2_meta, f, indent=2)

    with open(out_dir / "train_log.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info("Stage 2 complete. Full module saved to %s/module.pt", out_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    pre, _ = parser.parse_known_args()
    config = load_config(pre.config)
    set_parser_defaults_from_config(parser, config)
    args = parser.parse_args()
    print_config(args, title="Stage 2 Training Configuration")
    train(args)


if __name__ == "__main__":
    main()
