"""Stage 2 write diagnostics to find whether the write mechanism uses carry knowledge or a shortcut

Four ablations:

  1. zero_f       – replace BiGRU output f with zeros before gate/write.
                    Since gated = f * g and W_write has no bias, deltas = 0, equivalent to no-module baseline.

  2. scrambled_f  – shuffle f within the batch (real carry values, wrong example).
                    Tests whether the specific carry content of f matters.

  3. template transfer – evaluate full module on templates not seen in Stage 2
                    (Stage 2 trained on T0 only).

  4. n_digits transfer – evaluate on digit counts the slot_attn was not sized for.

Metrics per condition:
  - ce          : cross-entropy on answer positions (lower = better)
  - token_acc   : fraction of answer tokens where argmax == correct token
  - exact_match : fraction of examples where all answer tokens are correct

Run:
    python -m experiments.hierarchical_module_prototype.eval_write \\
        --stage2_dir  runs/hierarchical_module/stage2 \\
        --stage1a_dir runs/hierarchical_module/stage1a \\
        --stage1b_dir runs/hierarchical_module/stage1b_multitemplate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.addition.dataset_generation.generate_dataset_with_predictions import (  # noqa: E402
    TEMPLATES,
    TemplateID,
)
from experiments.hierarchical_module_prototype.model import (  # noqa: E402
    PrototypeModule,
)
from experiments.hierarchical_module_prototype.utils import (  # noqa: E402
    build_training_batch,
    collect_residuals,
    compute_ce_on_answer_positions,
    generate_hard_regime_pairs,
    make_write_hooks,
)
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype  # noqa: E402
from mechinterp_qwen3.utils_seed import seed_everything  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hmp.eval_write")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def compute_metrics(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    prompt_lens: list[int],
    answer_lens: list[int],
) -> dict[str, float]:
    """CE, token accuracy, and exact match on answer positions."""
    ce = compute_ce_on_answer_positions(logits, tokens, prompt_lens, answer_lens).item()

    correct_tokens = 0
    total_tokens = 0
    exact_matches = 0

    B = logits.shape[0]
    for i in range(B):
        p = prompt_lens[i]
        a = answer_lens[i]
        all_correct = True
        for j in range(a):
            logit_pos = p - 1 + j
            target_pos = p + j
            if logit_pos >= logits.shape[1] or target_pos >= tokens.shape[1]:
                break
            pred = int(logits[i, logit_pos].argmax().item())
            tgt = int(tokens[i, target_pos].item())
            if pred == tgt:
                correct_tokens += 1
            else:
                all_correct = False
            total_tokens += 1
        if all_correct:
            exact_matches += 1

    return {
        "ce": ce,
        "token_acc": correct_tokens / max(1, total_tokens),
        "exact_match": exact_matches / B,
    }


@torch.no_grad()
def evaluate_condition(
    module: PrototypeModule,
    qwen,
    pairs: list[tuple[int, int]],
    template_str: str,
    device: torch.device,
    dtype: torch.dtype,
    n_layers: int,
    zero_f: bool = False,
    scramble_f: bool = False,
    bypass_primitive: bool = False,
) -> dict[str, float]:
    """Run one evaluation condition over all pairs (batched)."""
    module.eval()
    all_metrics: list[dict] = []
    batch_size = 32

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        tokens, prompt_lens, answer_lens = build_training_batch(
            batch, template_str, qwen.tokenizer, device
        )

        residuals = collect_residuals(qwen, tokens).to(device=device, dtype=dtype)

        # Slice to prompt-only (matches Stage 1b training distribution)
        max_prompt_len = max(prompt_lens)
        residuals_prompt = residuals[:, :, :max_prompt_len, :]

        x = module.read(residuals_prompt)
        x_slots, slot_attn_weights = module.slot_attn(x)
        f = x_slots if bypass_primitive else module.primitive(x_slots)  # (B, n_digits, d_small)

        if zero_f:
            f = torch.zeros_like(f)
        elif scramble_f:
            perm = torch.randperm(f.shape[0], device=f.device)
            f = f[perm]

        deltas_prompt, _carry_logits = module.write(f, slot_attn_weights)

        # Pad deltas to full sequence length
        T_full = tokens.shape[1]
        if T_full > max_prompt_len:
            pad = torch.zeros(
                n_layers,
                tokens.shape[0],
                T_full - max_prompt_len,
                deltas_prompt.shape[-1],
                device=device,
                dtype=dtype,
            )
            deltas = torch.cat([deltas_prompt, pad], dim=2)
        else:
            deltas = deltas_prompt

        write_hooks = make_write_hooks(deltas, n_layers)
        logits = qwen.run_with_hooks(tokens, fwd_hooks=write_hooks)

        all_metrics.append(compute_metrics(logits, tokens, prompt_lens, answer_lens))

    # Average across batches
    return {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}


@torch.no_grad()
def evaluate_no_module(
    qwen,
    pairs: list[tuple[int, int]],
    template_str: str,
    device: torch.device,
    n_layers: int,
) -> dict[str, float]:
    """Qwen alone, no write hooks."""
    all_metrics: list[dict] = []
    batch_size = 32

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        tokens, prompt_lens, answer_lens = build_training_batch(
            batch, template_str, qwen.tokenizer, device
        )
        logits = qwen(tokens)
        all_metrics.append(compute_metrics(logits, tokens, prompt_lens, answer_lens))

    return {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 2 write diagnostics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage2_dir", required=True)
    p.add_argument("--stage1a_dir", required=True)
    p.add_argument("--stage1b_dir", required=True)
    p.add_argument("--n_eval", type=int, default=256, help="Pairs per condition")
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--out", default=None, help="Optional JSON path for results")
    return p


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = get_default_device()

    stage2_dir = Path(args.stage2_dir)
    stage1b_dir = Path(args.stage1b_dir)

    with open(stage1b_dir / "meta.json") as f:
        meta = json.load(f)

    model_name = meta["model"]
    dtype = parse_dtype(meta["dtype"])
    n_layers = meta["n_layers"]
    d_model = meta["d_model"]
    d_small = meta["d_small"]
    n_digits = meta["n_digits"]
    ref_layer = meta.get("ref_layer", 0)
    input_dependent = meta.get("input_dependent_cross_attn", True)

    log.info("Loading Qwen: %s", model_name)
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

    # Load full module
    module = PrototypeModule(
        n_layers,
        d_model,
        d_small,
        n_digits=n_digits,
        ref_layer=ref_layer,
        input_dependent=input_dependent,
    ).to(device=device, dtype=dtype)
    module.load_state_dict(torch.load(stage2_dir / "module.pt", map_location=device))
    module.requires_grad_(False)
    module.eval()
    log.info("Module loaded from %s", stage2_dir)

    # Held-out pairs (never seen in training, seed differs from train)
    pairs_4 = generate_hard_regime_pairs(n_digits, args.n_eval * 2, seed=args.seed, held_out=True)[
        : args.n_eval
    ]

    T0 = TEMPLATES[TemplateID.T0]
    results: dict[str, dict] = {}

    # ── 1. Baseline: Qwen alone ──────────────────────────────────────────────
    log.info("Evaluating: no_module (Qwen baseline)")
    results["no_module"] = evaluate_no_module(qwen, pairs_4, T0, device, n_layers)

    # ── 2. Full module, T0 (held-out pairs) ──────────────────────────────────
    log.info("Evaluating: full_module / T0")
    results["full_module_T0"] = evaluate_condition(
        module,
        qwen,
        pairs_4,
        T0,
        device,
        dtype,
        n_layers,
    )

    # ── 3. Zero f: deltas collapse to zero → same as no-module ───────────────
    log.info("Evaluating: zero_f")
    results["zero_f"] = evaluate_condition(
        module,
        qwen,
        pairs_4,
        T0,
        device,
        dtype,
        n_layers,
        zero_f=True,
    )

    # ── 4. Scrambled f: correct distribution, wrong carry for this example ───
    log.info("Evaluating: scrambled_f")
    results["scrambled_f"] = evaluate_condition(
        module,
        qwen,
        pairs_4,
        T0,
        device,
        dtype,
        n_layers,
        scramble_f=True,
    )

    # ── 4b. Bypass primitive: skip BiGRU, feed slot-attn output directly ─────
    # If performance matches full_module, the BiGRU is not needed for steering.
    log.info("Evaluating: bypass_primitive (no BiGRU)")
    results["bypass_primitive"] = evaluate_condition(
        module,
        qwen,
        pairs_4,
        T0,
        device,
        dtype,
        n_layers,
        bypass_primitive=True,
    )

    # ── 5. Template transfer (Stage 2 only saw T0) ───────────────────────────
    for tid in [TemplateID.T1, TemplateID.T2, TemplateID.T3]:
        key = f"full_module_{tid.value}"
        log.info("Evaluating: %s", key)
        results[key] = evaluate_condition(
            module,
            qwen,
            pairs_4,
            TEMPLATES[tid],
            device,
            dtype,
            n_layers,
        )

    # ── 6. n_digits transfer ─────────────────────────────────────────────────
    for nd in [3, 5, 6]:
        key = f"full_module_ndigits{nd}"
        log.info("Evaluating: %s  (module trained on %d digits)", key, n_digits)
        pairs_nd = generate_hard_regime_pairs(nd, args.n_eval * 2, seed=args.seed, held_out=True)[
            : args.n_eval
        ]
        results[key] = evaluate_condition(
            module,
            qwen,
            pairs_nd,
            T0,
            device,
            dtype,
            n_layers,
        )
        # Baseline for this digit count
        key_base = f"no_module_ndigits{nd}"
        log.info("Evaluating: %s", key_base)
        results[key_base] = evaluate_no_module(qwen, pairs_nd, T0, device, n_layers)

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Condition':<30}  {'CE':>7}  {'tok_acc':>8}  {'exact_match':>11}")
    print("-" * 72)
    for cond, m in results.items():
        print(f"{cond:<30}  {m['ce']:>7.4f}  {m['token_acc']:>8.4f}  {m['exact_match']:>11.4f}")
    print("=" * 72)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
