"""Print sample pairs from all concept datasets.

Usage:
    python scripts/print_concept_pairs.py                           # stdout
    python scripts/print_concept_pairs.py --output pairs.txt        # save to file
    python scripts/print_concept_pairs.py --concept carry --n_pairs 5
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.concept_localization.pipeline.run_concept_sweep import CONCEPTS


def print_concept_pairs(concept: str, n_pairs: int = 3):
    """Print sample pairs from a concept dataset."""
    try:
        mod = importlib.import_module(f"experiments.concept_localization.concept_datasets.{concept}_dataset")

        # Find generate function
        gen_fn = None
        for attr_name in dir(mod):
            if attr_name.startswith("generate_") and attr_name.endswith("_pairs"):
                gen_fn = getattr(mod, attr_name)
                break

        if not gen_fn:
            print(f"  [No generate function found]")
            return

        pairs = gen_fn(n_per_template=n_pairs, seed=42)

        print(f"\n{concept.upper()}")
        print("=" * 100)
        for i, pair in enumerate(pairs[:n_pairs]):
            print(f"\nPair {i}:")
            print(f"  POS ({pair.label_pos:5s}): {pair.prompt_pos}")
            print(f"  NEG ({pair.label_neg:5s}): {pair.prompt_neg}")
            if pair.meta:
                meta_str = " | ".join(f"{k}={v}" for k, v in sorted(pair.meta.items()))
                print(f"  Meta: {meta_str}")

    except Exception as e:
        print(f"  [ERROR: {str(e)[:60]}]")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--concept", default=None, help="Specific concept (default: all)")
    parser.add_argument("--n_pairs", type=int, default=3, help="Number of pairs to print per concept")
    parser.add_argument("--output", default=None, help="Output file (default: stdout)")
    args = parser.parse_args()

    concepts = [args.concept] if args.concept else CONCEPTS

    # Collect output
    lines = []
    for concept in sorted(concepts):
        # Temporarily redirect print to capture output
        import io
        from contextlib import redirect_stdout

        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            print_concept_pairs(concept, args.n_pairs)
        lines.append(output_buffer.getvalue())

    # Print or save
    output_text = "\n".join(lines) + "\n" + "=" * 100 + f"\nPrinted {args.n_pairs} pairs for each concept\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"Saved to {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
