"""Dataset generation for addition prompts with baseline model statistics.

This module generates addition prompts from configurable templates and computes
teacher-forced statistics for mechanistic interpretability experiments.

usage: # will take default values from config.yaml, if not provided
   miq generate-dataset
"""

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from mechinterp_qwen3.utils.inference_utils import batched_greedy_generate, silence_libraries
from mechinterp_qwen3.utils_seed import seed_everything

# Silence Hugging Face Hub downloads and Transformers loading progress
silence_libraries()


class TemplateID(StrEnum):
    """Template identifiers for addition prompts."""

    T0 = "T0"
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"
    T5 = "T5"
    T6 = "T6"


TEMPLATES = {  # see observations.rst for details on why trailing spaces are important
    TemplateID.T0: "calc: {a}+{b}= ",
    TemplateID.T1: "calc: {a} + {b} = ",
    TemplateID.T2: "What is {a}+{b}? Answer: ",
    TemplateID.T3: "calc: {a}+{b}=\n",
    TemplateID.T4: "calc: {a}+{b}=\nAnswer: ",
    TemplateID.T5: "<|im_start|>user\nCalculate {a}+{b}<|im_end|>\n<|im_start|>assistant\n",
    TemplateID.T6: "Answer the following addition problem: {a} + {b} =",
}


class SamplingStrategy(StrEnum):
    """Sampling strategies for (a,b) pairs."""

    GRID = "grid"
    STRATIFIED = "stratified"
    RANDOM = "random"


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""

    model: str
    output_path: Path
    templates: list[TemplateID]
    sampling_strategy: SamplingStrategy
    max_value: int
    n_samples: int | None = None
    batch_size: int = 32
    stratified_n_per_category: int = 100
    stratified_uniform_remainder: int = 100
    top_k: int = 10
    max_gen_tokens: int = 10
    seed: int = 42
    device: str | None = None
    dtype: str = "float32"


def classify_carry_pattern(a: int, b: int) -> Literal["no_carry", "single_carry", "multi_carry"]:
    """Classify addition problem by carry pattern.

    Args:
        a: First operand
        b: Second operand

    Returns:
        Carry pattern classification
    """
    carry_count = 0
    carry = 0

    a_str = str(a)
    b_str = str(b)

    # Pad to same length
    max_len = max(len(a_str), len(b_str))
    a_str = a_str.zfill(max_len)
    b_str = b_str.zfill(max_len)

    # Process right to left
    for i in range(max_len - 1, -1, -1):
        digit_sum = int(a_str[i]) + int(b_str[i]) + carry
        if digit_sum >= 10:
            carry_count += 1
            carry = 1
        else:
            carry = 0

    if carry_count == 0:
        return "no_carry"
    elif carry_count == 1:
        return "single_carry"
    else:
        return "multi_carry"


def generate_pairs(config: DatasetConfig) -> list[tuple[int, int]]:
    """Generate (a, b) pairs according to sampling strategy.

    Args:
        config: Dataset configuration

    Returns:
        List of (a, b) tuples
    """
    if config.sampling_strategy == SamplingStrategy.GRID:
        # Full grid over [0..N] x [0..N]
        pairs = [(a, b) for a in range(config.max_value + 1) for b in range(config.max_value + 1)]

    elif config.sampling_strategy == SamplingStrategy.STRATIFIED:
        pairs = []
        categories = {"no_carry": [], "single_carry": [], "multi_carry": []}

        for a in range(config.max_value + 1):
            for b in range(config.max_value + 1):
                pattern = classify_carry_pattern(a, b)
                categories[pattern].append((a, b))

        import random

        rng = random.Random(config.seed)

        for category_pairs in categories.values():
            if len(category_pairs) <= config.stratified_n_per_category:
                pairs.extend(category_pairs)
            else:
                pairs.extend(rng.sample(category_pairs, config.stratified_n_per_category))

        all_pairs = [
            (a, b) for a in range(config.max_value + 1) for b in range(config.max_value + 1)
        ]
        remaining = [p for p in all_pairs if p not in pairs]
        if len(remaining) <= config.stratified_uniform_remainder:
            pairs.extend(remaining)
        else:
            pairs.extend(rng.sample(remaining, config.stratified_uniform_remainder))

    elif config.sampling_strategy == SamplingStrategy.RANDOM:
        import random

        rng = random.Random(config.seed)
        n_samples = config.n_samples or (config.max_value + 1) ** 2
        pairs = [
            (
                rng.randint(0, config.max_value),
                rng.randint(0, config.max_value),
            )
            for _ in range(n_samples)
        ]

    else:
        raise ValueError(f"Unknown sampling strategy: {config.sampling_strategy}")

    return pairs


def build_prompt(template_id: TemplateID, a: int, b: int) -> str:
    """Build prompt from template.

    Args:
        template_id: Template identifier
        a: First operand
        b: Second operand

    Returns:
        Formatted prompt string
    """
    template = TEMPLATES[template_id]
    return template.format(a=a, b=b)


@dataclass
class PerPositionStats:
    """Per-position statistics for teacher-forced scoring."""

    pos: int
    true_id: int
    true_str: str
    logit_true: float
    prob_true: float
    topk_ids: list[int]
    topk_strs: list[str]
    topk_probs: list[float]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pos": self.pos,
            "true_id": self.true_id,
            "true_str": self.true_str,
            "logit_true": self.logit_true,
            "prob_true": self.prob_true,
            "topk_ids": self.topk_ids,
            "topk_strs": self.topk_strs,
            "topk_probs": self.topk_probs,
        }


@dataclass
class DatasetRecord:
    """Single dataset record with prompt and model statistics."""

    prompt_id: int
    template_id: str
    a: int
    b: int
    prompt_str: str
    true_answer_str: str
    prompt_token_ids: list[int]
    answer_token_ids: list[int]
    answer_token_strs: list[str]
    per_pos: list[PerPositionStats]
    greedy_completion_str: str | None
    metadata: dict

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt_id": self.prompt_id,
            "template_id": self.template_id,
            "a": self.a,
            "b": self.b,
            "prompt_str": self.prompt_str,
            "true_answer_str": self.true_answer_str,
            "prompt_token_ids": self.prompt_token_ids,
            "answer_token_ids": self.answer_token_ids,
            "answer_token_strs": self.answer_token_strs,
            "per_pos": [p.to_dict() for p in self.per_pos],
            "greedy_completion_str": self.greedy_completion_str,
            "metadata": self.metadata,
        }


def score_teacher_forced(
    model: HookedTransformer,
    prompt_str: str,
    true_answer_str: str,
    top_k: int = 10,
) -> tuple[list[int], list[int], list[str], list[PerPositionStats]]:
    """Compute teacher-forced statistics for a prompt-answer pair.

    Args:
        model: HookedTransformer model
        prompt_str: Prompt string
        true_answer_str: True answer string
        top_k: Number of top-k predictions to store

    Returns:
        Tuple of (prompt_token_ids, answer_token_ids, answer_token_strs, per_pos_stats)
    """
    # Tokenize prompt and answer
    if hasattr(model, "tokenize_qwen_input"):
        # tokenize_qwen_input adds the sink token
        prompt_tokens = model.tokenize_qwen_input(prompt_str).cpu()
    else:
        prompt_tokens = model.tokenizer(
            prompt_str, return_tensors="pt", add_special_tokens=False
        ).input_ids.squeeze(0)

    answer_tokens = model.tokenizer(
        true_answer_str, return_tensors="pt", add_special_tokens=False
    ).input_ids.squeeze(0)

    # Combine for teacher forcing
    full_tokens = torch.cat([prompt_tokens, answer_tokens], dim=0)
    full_tokens = full_tokens.unsqueeze(0).to(model.cfg.device)  # Add batch dimension

    with torch.inference_mode():
        logits = model(full_tokens)  # (batch=1, seq_len, vocab_size)

    logits = logits.squeeze(0)  # (seq_len, vocab_size)

    answer_token_strs = [model.tokenizer.decode([token_id.item()]) for token_id in answer_tokens]

    per_pos_stats = []
    prompt_len = len(prompt_tokens)

    for i, true_token_id in enumerate(answer_tokens):
        # Position in the sequence (logit at position i predicts token at i+1)
        # So we use logits from position prompt_len + i - 1 to predict answer token i
        logit_pos = prompt_len - 1 if i == 0 else prompt_len + i - 1

        position_logits = logits[logit_pos]  # (vocab_size,)

        probs = torch.softmax(position_logits, dim=-1)

        true_id = true_token_id.item()
        true_str = model.tokenizer.decode([true_id])
        logit_true = position_logits[true_id].item()
        prob_true = probs[true_id].item()
        topk_probs, topk_ids = torch.topk(probs, min(top_k, len(probs)))
        topk_ids = topk_ids.tolist()
        topk_probs = topk_probs.tolist()
        topk_strs = [model.tokenizer.decode([token_id]) for token_id in topk_ids]

        per_pos_stats.append(
            PerPositionStats(
                pos=i,
                true_id=true_id,
                true_str=true_str,
                logit_true=logit_true,
                prob_true=prob_true,
                topk_ids=topk_ids,
                topk_strs=topk_strs,
                topk_probs=topk_probs,
            )
        )

    return (
        prompt_tokens.tolist(),
        answer_tokens.tolist(),
        answer_token_strs,
        per_pos_stats,
    )


def greedy_generate(
    model: HookedTransformer,
    prompt_str: str,
    max_tokens: int = 10,
) -> str:
    """Wrapper for batched_greedy_generate for a single prompt."""
    return batched_greedy_generate(model, [prompt_str], max_tokens, batch_size=1)[0]


def generate_dataset(config: DatasetConfig) -> tuple[list[DatasetRecord], dict]:
    """Generate complete dataset with all records.

    Args:
        config: Dataset configuration

    Returns:
        Tuple of (records, summary_stats)
    """
    seed_everything(config.seed)

    print(f"Loading model: {config.model}")
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[config.dtype]

    model = HookedTransformer.from_pretrained(
        config.model,
        device=device,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )

    print(f"Generating pairs using {config.sampling_strategy} strategy...")
    pairs = generate_pairs(config)
    print(f"Generated {len(pairs)} unique pairs")
    records = []
    prompt_id = 0

    metadata = {
        "model": config.model,
        "seed": config.seed,
        "dtype": config.dtype,
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
    }

    all_combinations = []
    for template_id in config.templates:
        for a, b in pairs:
            all_combinations.append((template_id, a, b))

    total_items = len(all_combinations)
    print(
        f"Processing {total_items} prompt-template combinations with batch_size={config.batch_size}..."
    )

    with tqdm(total=total_items) as pbar:
        for i in range(0, total_items, config.batch_size):
            batch_items = all_combinations[i : i + config.batch_size]
            batch_prompts = [build_prompt(t_id, a, b) for t_id, a, b in batch_items]
            batch_true_answers = [str(a + b) for _, a, b in batch_items]

            # 1. Batched Greedy Generation
            # This is much faster than the old sequential loop
            batch_greedy_completions = batched_greedy_generate(
                model, batch_prompts, config.max_gen_tokens
            )

            # 2. Sequential Stats (for now, but we can batch this later if needed)
            # score_teacher_forced is still sequential because of handling varied output lengths
            # but it is only one forward pass per prompt, so the bottleneck was actually greedy gen.
            for j, (t_id, a, b) in enumerate(batch_items):
                prompt_str = batch_prompts[j]
                true_answer_str = batch_true_answers[j]
                greedy_completion_str = batch_greedy_completions[j]

                (
                    prompt_token_ids,
                    answer_token_ids,
                    answer_token_strs,
                    per_pos_stats,
                ) = score_teacher_forced(model, prompt_str, true_answer_str, config.top_k)

                record = DatasetRecord(
                    prompt_id=prompt_id,
                    template_id=t_id.value,
                    a=a,
                    b=b,
                    prompt_str=prompt_str,
                    true_answer_str=true_answer_str,
                    prompt_token_ids=prompt_token_ids,
                    answer_token_ids=answer_token_ids,
                    answer_token_strs=answer_token_strs,
                    per_pos=per_pos_stats,
                    greedy_completion_str=greedy_completion_str,
                    metadata=metadata,
                )

                records.append(record)
                prompt_id += 1

            pbar.update(len(batch_items))

    print("Computing summary statistics...")
    summary = compute_summary_stats(records, config)

    return records, summary


def compute_summary_stats(records: list[DatasetRecord], config: DatasetConfig) -> dict:
    """Compute summary statistics for the dataset.

    Args:
        records: List of dataset records
        config: Dataset configuration

    Returns:
        Dictionary of summary statistics
    """
    n_records = len(records)

    answer_lengths = [len(r.answer_token_ids) for r in records]
    length_distribution = dict(Counter(answer_lengths))

    accuracy_per_template = {}
    for template_id in config.templates:
        template_records = [r for r in records if r.template_id == template_id.value]
        correct = sum(
            1
            for r in template_records
            if r.greedy_completion_str is not None
            and r.greedy_completion_str.strip() == r.true_answer_str
        )
        total = len(template_records)
        accuracy_per_template[template_id.value] = correct / total if total > 0 else 0.0

    max_answer_length = max(answer_lengths)
    mean_prob_per_position = []

    for pos in range(max_answer_length):
        probs = [r.per_pos[pos].prob_true for r in records if len(r.per_pos) > pos]
        mean_prob = sum(probs) / len(probs) if probs else 0.0
        mean_prob_per_position.append(mean_prob)

    summary = {
        "n_records": n_records,
        "answer_length_distribution": length_distribution,
        "accuracy_per_template": accuracy_per_template,
        "mean_prob_true_per_position": mean_prob_per_position,
    }

    return summary


def write_dataset(
    records: list[DatasetRecord],
    summary: dict,
    output_path: Path,
) -> None:
    """Write dataset to JSONL file and print summary.

    Args:
        records: List of dataset records
        summary: Summary statistics dictionary
        output_path: Path to output JSONL file
    """
    print(f"Writing dataset to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record.to_dict()) + "\n")

    print(f"Dataset written successfully ({len(records)} records)")

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total records: {summary['n_records']}")
    print("\nAnswer token length distribution:")
    for length, count in sorted(summary["answer_length_distribution"].items()):
        print(f"  {length} tokens: {count} records")

    if summary["accuracy_per_template"]:
        print("\nGreedy generation accuracy per template:")
        for template_id, accuracy in summary["accuracy_per_template"].items():
            print(f"  {template_id}: {accuracy:.2%}")

    print("\nMean prob_true per position:")
    for pos, prob in enumerate(summary["mean_prob_true_per_position"]):
        print(f"  Position {pos}: {prob:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    import sys

    # Import and run the CLI
    try:
        from . import main

        main()
    except ImportError:
        # Fallback: provide helpful error message
        print("ERROR: Could not import dataset_generation module.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Please use the proper CLI entrypoint instead:", file=sys.stderr)
        print("  python -m mechinterp_qwen3.dataset_generation [options]", file=sys.stderr)
        print("", file=sys.stderr)
        print("Or import from this module in your own code.", file=sys.stderr)
        sys.exit(1)
