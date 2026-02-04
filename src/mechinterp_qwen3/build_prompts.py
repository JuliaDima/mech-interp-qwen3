from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from .io import write_jsonl
from .utils_seed import SeedConfig, set_all_seeds


@dataclass(frozen=True)
class GTExample:
    prompt_id: str
    a: int
    b: int
    prompt: str
    expected: str


def make_gt_prompt(a: int, b: int) -> tuple[str, str]:
    # Keep formatting consistent to reduce confounds.
    # Expected answer is "A" if a>b else "B".
    expected = "A" if a > b else "B"
    prompt = (
        "You are solving a simple comparison task.\n"
        "Two numbers are given: A and B.\n"
        "Answer with a single character: 'A' if A is larger, otherwise 'B'.\n\n"
        f"A = {a}\n"
        f"B = {b}\n"
        "Answer: "
    )
    return prompt, expected


def generate_examples(n: int, seed: int, low: int, high: int) -> list[GTExample]:
    set_all_seeds(SeedConfig(seed=seed))
    rng = random.Random(seed)

    exs: list[GTExample] = []
    for i in range(n):
        a = rng.randint(low, high)
        b = rng.randint(low, high)
        # Avoid ties unless you want them; ties add ambiguity.
        while b == a:
            b = rng.randint(low, high)

        prompt, expected = make_gt_prompt(a, b)
        exs.append(GTExample(prompt_id=f"gt_{i:04d}", a=a, b=b, prompt=prompt, expected=expected))
    return exs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="src/mechinterp_qwen3/prompts/greater_than.jsonl")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--low", type=int, default=0)
    ap.add_argument("--high", type=int, default=999)
    args = ap.parse_args()

    out = Path(args.out)
    exs = generate_examples(args.n, args.seed, args.low, args.high)

    rows = []
    for e in exs:
        rows.append(
            {
                "prompt_id": e.prompt_id,
                "behaviour": "greater_than",
                "a": e.a,
                "b": e.b,
                "prompt": e.prompt,
                "expected": e.expected,
            }
        )

    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} prompts -> {out}")


if __name__ == "__main__":
    main()
