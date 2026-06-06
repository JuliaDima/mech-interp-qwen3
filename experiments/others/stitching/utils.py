import json
import logging
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.addition.dataset_generation.generate_dataset_with_predictions import (
    TEMPLATES,
    TemplateID,
)

log = logging.getLogger("stitching.utils")


def load_addition_dataset(
    dataset_path: str,
    max_samples: int | None = None,
    num_digits: int = 5,
) -> list[dict[str, Any]]:
    """Load addition dataset from JSONL.  Falls back to generating samples on the fly."""
    path = Path(dataset_path)
    samples: list[dict[str, Any]] = []

    if path.exists():
        with open(path) as f:
            for line in f:
                samples.append(json.loads(line))
                if max_samples and len(samples) >= max_samples:
                    break
        return samples

    # ---- Fallback: generate on the fly ----
    log.warning(
        "Dataset not found at %s — generating %d-digit samples on the fly", dataset_path, num_digits
    )
    num_samples = max_samples if max_samples else 200_000
    max_digits = num_digits
    fallback_n = min(num_samples if num_samples else 50_000, 50_000)  # cap fallback at 50k
    random.seed(42)
    seen: set[tuple[int, int]] = set()
    while len(samples) < fallback_n:
        # Uniformly sample the number of digits to get a balanced mix of difficulties
        d_a = random.randint(1, max_digits)
        d_b = random.randint(1, max_digits)
        a = random.randint(10 ** (d_a - 1) if d_a > 1 else 0, 10**d_a - 1)
        b = random.randint(10 ** (d_b - 1) if d_b > 1 else 0, 10**d_b - 1)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        samples.append(
            {
                "prompt": TEMPLATES[TemplateID.T0].format(a=a, b=b),
                "answer": str(a + b),
                "a": a,
                "b": b,
                "template": "T0",
            }
        )

    log.info("Generated %d samples with mixed digits (max %d)", len(samples), max_digits)
    return samples


def get_small_model_tokenizer(model: Any, max_len: int | None = None):
    """Centralized tokenizer factory for small models.

    Identifies if model is QuantaMaths (15-token vocab) or scratch-trained (18-token).
    Returns a function(text) -> list[int] with consistent padding/truncation.
    """
    _n_ctx = max_len if max_len is not None else model.model.cfg.n_ctx

    # Case 1: QuantaMaths pretrained model (PhilipQuirke/QuantaMaths_*)
    if hasattr(model, "_tokenizer"):
        _tok_fn = model._tokenizer

        def tokenize_qm(text: str, max_l: int = _n_ctx) -> list[int]:
            toks = _tok_fn(text)
            if len(toks) < max_l:
                toks += [0] * (max_l - len(toks))
            return toks[:max_l]

        return tokenize_qm

    # Case 2: Scratch-trained addition model
    _vocab = ["<PAD>", "<BOS>", "<EOS>"] + [str(i) for i in range(10)] + ["+", "=", " "]
    _c2i = {c: i for i, c in enumerate(_vocab)}

    def tokenize_scratch(text: str, max_l: int = _n_ctx) -> list[int]:
        toks = [_c2i.get(c, 0) for c in text]
        if len(toks) < max_l:
            toks += [0] * (max_l - len(toks))
        return toks[:max_l]

    return tokenize_scratch


def identify_cascading_carry_cases(samples: list[dict[str, Any]], threshold: int = 2) -> list[bool]:
    """Identify samples where carry propagates across >= threshold digits."""
    cascading = []
    for sample in samples:
        a = sample.get("a", 0)
        b = sample.get("b", 0)
        carry = 0
        max_consecutive = 0
        current_consecutive = 0

        while a > 0 or b > 0 or carry > 0:
            digit_sum = (a % 10) + (b % 10) + carry
            if digit_sum >= 10:
                carry = 1
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                carry = 0
                current_consecutive = 0
            a //= 10
            b //= 10

        cascading.append(max_consecutive >= threshold)
    return cascading


def plot_stitching_results(
    a_vals: list[int],
    b_vals: list[int],
    prob_before: list[float],
    prob_after: list[float],
    output_dir: Path,
):
    """Plot the teacher-forced probabilities before and after stitching."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate improvement
    improvement = [a - b for a, b in zip(prob_after, prob_before, strict=False)]

    # Log scale a and b to make the scatter plot readable for large numbers
    a_log = np.log10(np.array(a_vals) + 1)
    b_log = np.log10(np.array(b_vals) + 1)

    # 1. Probability Before
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(a_log, b_log, c=prob_before, cmap="viridis", vmin=0, vmax=1, alpha=0.7)
    plt.colorbar(sc, label="P(correct) Before")
    plt.title("Teacher-Forced P(correct) Before Stitching")
    plt.xlabel("log10(a+1)")
    plt.ylabel("log10(b+1)")
    plt.savefig(output_dir / "prob_before.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Probability After
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(a_log, b_log, c=prob_after, cmap="viridis", vmin=0, vmax=1, alpha=0.7)
    plt.colorbar(sc, label="P(correct) After")
    plt.title("Teacher-Forced P(correct) After SAE-mediated Stitching")
    plt.xlabel("log10(a+1)")
    plt.ylabel("log10(b+1)")
    plt.savefig(output_dir / "prob_after.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Improvement
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(a_log, b_log, c=improvement, cmap="RdBu", vmin=-0.5, vmax=0.5, alpha=0.7)
    plt.colorbar(sc, label="Change in P(correct)")
    plt.title("Improvement in P(correct) due to Stitching")
    plt.xlabel("log10(a+1)")
    plt.ylabel("log10(b+1)")
    plt.savefig(output_dir / "improvement.png", dpi=300, bbox_inches="tight")
    plt.close()

    log.info(f"Saved stitching dataset plots to {output_dir}")
