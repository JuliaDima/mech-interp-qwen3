#!/usr/bin/env python3
"""Quick test script for dataset generation."""

import json
from pathlib import Path

from src.mechinterp_qwen3.dataset_generation import (
    DatasetConfig,
    SamplingStrategy,
    TemplateID,
    generate_dataset,
    write_dataset,
)

# Small test configuration
config = DatasetConfig(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    output_path=Path("test_output.jsonl"),
    templates=[TemplateID.T0],
    sampling_strategy=SamplingStrategy.GRID,
    max_value=5,  # Small grid: 6x6 = 36 pairs
    top_k=5,
    enable_greedy_generation=True,
    max_gen_tokens=5,
    seed=42,
    device=None,  # Auto-detect
    dtype="float32",
)

print("Running small test with grid sampling (0-5)...")
records, summary = generate_dataset(config)
write_dataset(records, summary, config.output_path)

# Verify output
print("\nVerifying output...")
with open(config.output_path) as f:
    lines = f.readlines()
    print(f"Total lines in JSONL: {len(lines)}")

    # Check first record
    first_record = json.loads(lines[0])
    print("\nFirst record keys:", list(first_record.keys()))
    print(f"First prompt: {first_record['prompt_str']}")
    print(f"First answer: {first_record['true_answer_str']}")
    print(f"Per-position stats count: {len(first_record['per_pos'])}")

    # Check a record with multi-token answer
    for line in lines:
        rec = json.loads(line)
        if len(rec["answer_token_ids"]) > 1:
            print(f"\nMulti-token example: {rec['a']} + {rec['b']} = {rec['true_answer_str']}")
            print(f"Answer tokens: {rec['answer_token_strs']}")
            print(f"Answer token IDs: {rec['answer_token_ids']}")
            print(f"Greedy completion: '{rec['greedy_completion_str']}'")
            break

print("\nTest completed successfully!")
