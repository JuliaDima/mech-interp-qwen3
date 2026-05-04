"""Generalization assessment for the Stage 1a BiGRU carry primitive.

Tests carry-out prediction accuracy across digit counts 1–15.
The model was trained exclusively on 4-digit operands ([1000, 9999]).

For each digit count d:
  - Generate --n_samples random pairs of d-digit numbers
  - Run PairEmbedding → CarryPrimitiveGRU → CarryHead (all from checkpoint)
  - Report per-position accuracy and overall accuracy

The CarryHead from Stage 1a uses n_out=1 applied per position, so it works
for any sequence length without modification.

Run:
    python -m experiments.hierarchical_module_prototype.eval_primitive_generalization \\
        --stage1a_dir runs/hierarchical_module/stage1a
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.hierarchical_module_prototype.model import (  # noqa: E402
    CarryHead,
    CarryPrimitiveGRU,
    PairEmbedding,
)
from experiments.hierarchical_module_prototype.utils import (  # noqa: E402
    build_carry_batch,
)


def generate_pairs(n_digits: int, n_samples: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    lo = 10 ** (n_digits - 1) if n_digits > 1 else 1
    hi = 10**n_digits - 1
    return [(rng.randint(lo, hi), rng.randint(lo, hi)) for _ in range(n_samples)]


def evaluate_digit_count(
    n_digits: int,
    pairs: list[tuple[int, int]],
    pair_embedding: PairEmbedding,
    primitive: CarryPrimitiveGRU,
    carry_head: CarryHead,
    device: torch.device,
) -> dict:
    pair_indices, carry_labels = build_carry_batch(pairs, n_digits, device)
    # pair_indices: (batch, n_digits), carry_labels: (batch, n_digits)

    with torch.no_grad():
        x = pair_embedding(pair_indices)  # (batch, n_digits, d_small)
        f = primitive(x)  # (batch, n_digits, d_small)
        logits = carry_head(f).squeeze(-1)  # (batch, n_digits)
        preds = (logits > 0).float()

    correct = (preds == carry_labels).float()

    overall_acc = correct.mean().item()
    per_pos_acc = correct.mean(dim=0).tolist()  # accuracy at each digit position

    # Carry frequency (fraction of positions that actually have carry=1)
    carry_freq = carry_labels.mean().item()

    return {
        "n_digits": n_digits,
        "overall_acc": overall_acc,
        "per_pos_acc": per_pos_acc,
        "carry_freq": carry_freq,
        "n_samples": len(pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generalization test: Stage 1a primitive across digit counts 1–15",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage1a_dir",
        required=True,
        help="Stage 1a checkpoint directory (primitive.pt, pair_embedding.pt, carry_head.pt, meta.json)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of random pairs to evaluate per digit count",
    )
    parser.add_argument("--min_digits", type=int, default=1)
    parser.add_argument("--max_digits", type=int, default=15)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--out", default=None, help="Optional path to save results JSON")
    args = parser.parse_args()

    stage1a_dir = Path(args.stage1a_dir)
    with open(stage1a_dir / "meta.json") as f:
        meta = json.load(f)

    d_small = meta["d_small"]
    trained_n_digits = meta["n_digits"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    pair_embedding = PairEmbedding(d_small).to(device)
    pair_embedding.load_state_dict(
        torch.load(stage1a_dir / "pair_embedding.pt", map_location=device)
    )
    pair_embedding.eval()

    primitive = CarryPrimitiveGRU(d_small).to(device)
    primitive.load_state_dict(torch.load(stage1a_dir / "primitive.pt", map_location=device))
    primitive.eval()

    carry_head = CarryHead(d_small, n_out=1).to(device)
    carry_head.load_state_dict(torch.load(stage1a_dir / "carry_head.pt", map_location=device))
    carry_head.eval()

    print(f"\nStage 1a checkpoint: {stage1a_dir}")
    print(f"Trained on: {trained_n_digits}-digit operands  |  d_small={d_small}")
    print(
        f"Evaluating {args.n_samples} pairs per digit count, digits {args.min_digits}–{args.max_digits}\n"
    )
    print(f"{'digits':>7}  {'overall_acc':>12}  {'carry_freq':>11}  {'per-position accuracy'}")
    print("-" * 90)

    all_results = []
    for n_d in range(args.min_digits, args.max_digits + 1):
        pairs = generate_pairs(n_d, args.n_samples, seed=args.seed + n_d)
        result = evaluate_digit_count(n_d, pairs, pair_embedding, primitive, carry_head, device)
        all_results.append(result)

        marker = "  ← trained" if n_d == trained_n_digits else ""
        per_pos_str = "  ".join(f"{a:.3f}" for a in result["per_pos_acc"])
        print(
            f"{n_d:>7}  {result['overall_acc']:>12.4f}  {result['carry_freq']:>11.4f}  [{per_pos_str}]{marker}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
