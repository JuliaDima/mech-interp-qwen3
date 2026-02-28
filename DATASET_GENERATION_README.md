# Addition Dataset Generation for Mechanistic Interpretability

## Overview

The dataset generation system provides:

1. **Configurable prompt templates** for addition tasks
2. **Multiple sampling strategies** (grid, stratified by carry patterns, random)
3. **Teacher-forced statistics** (per-token logits, probabilities, top-k predictions)
4. **Optional greedy generation** for accuracy validation
5. **Deterministic runs** with seed control
6. **JSONL output format** for easy downstream processing

## Architecture

### Core Components

- **`dataset_generation.py`**: Main module with all generation logic
  - `generate_pairs()`: Sample (a,b) pairs according to strategy
  - `build_prompt()`: Format prompts from templates
  - `score_teacher_forced()`: Compute per-token statistics under teacher forcing
  - `greedy_generate()`: Optional greedy decoding for validation
  - `generate_dataset()`: Orchestrate full pipeline

- **`dataset_generation.py`**: CLI entrypoint with argparse interface

## Templates

Three configurable templates are provided:

- **T0**: `"calc: {a}+{b}="`
- **T1**: `"calc: {a} + {b} ="` (with spaces)
- **T2**: `"What is {a}+{b}? Answer:"`

## Sampling Strategies

### 1. Grid Sampling (`--sampling_strategy grid`)

Generates all possible (a, b) pairs over [0..N] × [0..N].

**Use case**: Complete coverage for small N (e.g., N ≤ 100)

**Example**:
```bash
python -m mechinterp_qwen3.dataset_generation \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --output_path data/addition_grid_20.jsonl \
  --sampling_strategy grid \
  --max_value 20 \
  --templates T0 T1 T2 \
  --seed 42
```

This generates 21 × 21 × 3 = 1,323 records (3 templates).

### 2. Stratified Sampling (`--sampling_strategy stratified`)

Samples by carry patterns to ensure balanced representation:

- **No-carry**: e.g., 12 + 34 = 46
- **Single-carry**: e.g., 15 + 18 = 33
- **Multi-carry**: e.g., 99 + 99 = 198

**Use case**: Balanced dataset for studying carry mechanisms at larger N

**Example**:
```bash
python -m mechinterp_qwen3.dataset_generation \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --output_path data/addition_stratified_100.jsonl \
  --sampling_strategy stratified \
  --max_value 100 \
  --templates T0 \
  --stratified_n_per_category 200 \
  --stratified_uniform_remainder 100 \
  --seed 42
```

This samples:
- 200 examples from each carry category (no-carry, single-carry, multi-carry)
- 100 additional uniform random samples
- Total: ≤ 700 unique pairs

### 3. Random Sampling (`--sampling_strategy random`)

Pure random sampling with specified sample count.

**Use case**: Quick experiments or very large N

**Example**:
```bash
python -m mechinterp_qwen3.dataset_generation \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --output_path data/addition_random_1000.jsonl \
  --sampling_strategy random \
  --max_value 1000 \
  --n_samples 500 \
  --templates T1 \
  --seed 42
```

## Teacher-Forced Statistics

For each prompt-answer pair, the system computes per-position statistics by running a single forward pass with the concatenated `[prompt, answer]` sequence:

### Per-Position Outputs

For each answer token position `i`:

```json
{
  "pos": 0,
  "true_id": 1234,
  "true_str": "5",
  "logit_true": 12.34,
  "prob_true": 0.987,
  "topk_ids": [1234, 5678, ...],
  "topk_strs": ["5", "6", ...],
  "topk_probs": [0.987, 0.008, ...]
}
```

### Alignment Details

The code correctly aligns prompt and answer positions:
- **First answer token** (position 0): predicted by logits at `prompt_len - 1`
- **Subsequent tokens** (position i > 0): predicted by logits at `prompt_len + i - 1`

This ensures proper teacher-forcing where each token prediction sees all previous tokens.

## Output Format

### JSONL Record Structure

Each line in the output JSONL file contains:

```json
{
  "prompt_id": 0,
  "template_id": "T0",
  "a": 12,
  "b": 34,
  "prompt_str": "calc: 12+34=",
  "true_answer_str": "46",
  "prompt_token_ids": [1234, 5678, ...],
  "answer_token_ids": [9012, 3456],
  "answer_token_strs": ["4", "6"],
  "per_pos": [
    {
      "pos": 0,
      "true_id": 9012,
      "true_str": "4",
      "logit_true": 11.23,
      "prob_true": 0.95,
      "topk_ids": [9012, 1111, 2222, ...],
      "topk_strs": ["4", "5", "3", ...],
      "topk_probs": [0.95, 0.03, 0.01, ...]
    },
    {
      "pos": 1,
      "true_id": 3456,
      "true_str": "6",
      "logit_true": 13.45,
      "prob_true": 0.98,
      "topk_ids": [3456, 7777, 8888, ...],
      "topk_strs": ["6", "5", "7", ...],
      "topk_probs": [0.98, 0.01, 0.005, ...]
    }
  ],
  "greedy_completion_str": "46",
  "metadata": {
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "seed": 42,
    "dtype": "float32",
    "device": "cuda:0",
    "timestamp": "2024-02-28T12:00:00.123456"
  }
}
```

### Summary Statistics

Printed to stdout at completion:

```
==============================================================
DATASET SUMMARY
==============================================================
Total records: 1323

Answer token length distribution:
  1 tokens: 450 records
  2 tokens: 873 records

Greedy generation accuracy per template:
  T0: 98.50%
  T1: 97.80%
  T2: 96.20%

Mean prob_true per position:
  Position 0: 0.9234
  Position 1: 0.9567
==============================================================
```

## CLI Options

### Required Arguments

- `--output_path PATH`: Path to output JSONL file

### Model Configuration

- `--model_name MODEL`: HuggingFace model name (default: `Qwen/Qwen2.5-3B-Instruct`)
  - Also supports: `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen2.5-0.5B-Instruct`, etc.
- `--device DEVICE`: Device to use (`cuda` or `cpu`). Auto-detects if not specified.
- `--dtype DTYPE`: Model dtype (`float32`, `float16`, `bfloat16`). Default: `float32`

### Template Configuration

- `--templates T0 T1 T2`: Which templates to use (can specify multiple). Default: `T0`

### Sampling Configuration

- `--sampling_strategy {grid,stratified,random}`: Sampling strategy. Default: `grid`
- `--max_value N`: Maximum value for a and b. Default: `20`
- `--n_samples N`: Number of samples (required for `random` strategy)
- `--stratified_n_per_category N`: Samples per carry category (stratified only). Default: `100`
- `--stratified_uniform_remainder N`: Additional uniform samples (stratified only). Default: `100`

### Statistics Configuration

- `--top_k K`: Number of top-k tokens to store per position. Default: `10`

### Generation Configuration

- `--enable_greedy_generation`: Enable greedy decoding for validation
- `--max_gen_tokens N`: Max tokens for greedy generation. Default: `10`

### Reproducibility

- `--seed SEED`: Random seed for reproducibility. Default: `42`

## Complete Examples

### Example 1: Small Grid for Quick Testing

```bash
python -m mechinterp_qwen3.dataset_generation \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --output_path data/test_grid.jsonl \
  --sampling_strategy grid \
  --max_value 10 \
  --templates T0 \
  --seed 42
```

**Output**: 11 × 11 = 121 records

### Example 2: Comprehensive Grid with All Templates

```bash
python -m mechinterp_qwen3.dataset_generation \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --output_path data/addition_comprehensive.jsonl \
  --sampling_strategy grid \
  --max_value 50 \
  --templates T0 T1 T2 \
  --top_k 20 \
  --enable_greedy_generation \
  --seed 42
```

**Output**: 51 × 51 × 3 = 7,803 records

### Example 3: Stratified Sampling for Carry Analysis

```bash
python -m mechinterp_qwen3.dataset_generation \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --output_path data/carry_analysis.jsonl \
  --sampling_strategy stratified \
  --max_value 200 \
  --templates T0 T1 \
  --stratified_n_per_category 300 \
  --stratified_uniform_remainder 200 \
  --enable_greedy_generation \
  --seed 42
```

**Output**: ~1,000 balanced records across carry patterns

### Example 4: Large-Scale Random Sampling

```bash
python -m mechinterp_qwen3.dataset_generation \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --output_path data/large_random.jsonl \
  --sampling_strategy random \
  --max_value 9999 \
  --n_samples 5000 \
  --templates T0 \
  --dtype bfloat16 \
  --seed 42
```

**Output**: 5,000 records with 4-digit additions

### Example 5: Using Qwen3-4B Model

```bash
python -m mechinterp_qwen3.dataset_generation \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --output_path data/qwen3_addition.jsonl \
  --sampling_strategy grid \
  --max_value 30 \
  --templates T0 T1 T2 \
  --dtype bfloat16 \
  --enable_greedy_generation \
  --seed 42
```

## Downstream Usage

### Loading the Dataset

```python
import json
from pathlib import Path

def load_dataset(jsonl_path: Path):
    """Load dataset from JSONL file."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

# Load dataset
records = load_dataset(Path("data/addition_grid_20.jsonl"))

# Access statistics
for record in records[:5]:
    print(f"Prompt: {record['prompt_str']}")
    print(f"Answer: {record['true_answer_str']}")
    print(f"First token prob: {record['per_pos'][0]['prob_true']:.4f}")
    print()
```

### Filtering by Carry Pattern

```python
from mechinterp_qwen3.dataset_generation import classify_carry_pattern

# Filter for multi-carry examples
multi_carry = [
    r for r in records
    if classify_carry_pattern(r['a'], r['b']) == "multi_carry"
]

print(f"Multi-carry examples: {len(multi_carry)}")
```

### Computing Accuracy

```python
def compute_accuracy(records):
    """Compute greedy generation accuracy."""
    correct = sum(
        1 for r in records
        if r['greedy_completion_str'] and
           r['greedy_completion_str'].strip() == r['true_answer_str']
    )
    return correct / len(records)

acc = compute_accuracy(records)
print(f"Accuracy: {acc:.2%}")
```

## Integration with SAE Training

This dataset serves as the foundation for SAE (Sparse Autoencoder) training and attribution graph construction:

1. **Activation Collection**: Use prompts to collect MLP activations at selected layers
2. **SAE Training**: Train SAEs on collected activations with proper train/val split
3. **Attribution Graphs**: Build circuit-tracer-style graphs using per-position logits as targets

### Example: Collecting Activations

```python
from mechinterp_qwen3.attribution_model import AttributionModel

# Load model with transcoders
model = AttributionModel.from_pretrained(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    transcoder_set="qwen2.5-3b-transcoders",
    device="cuda",
)

# Get activations for a prompt
prompt = "calc: 12+34="
logits, activations = model.get_activations(prompt, sparse=True)

# activations shape: (n_layers, n_positions, d_transcoder)
```

## Deterministic Runs

All runs are fully deterministic when using the same seed:

- PyTorch random seed
- NumPy random seed
- Python random seed
- CUDA deterministic algorithms
- Controlled sampling order

This ensures perfect reproducibility for scientific experiments.

## Performance Considerations

### Memory

- **Grid sampling**: Memory scales as O(N²) for pairs, O(seq_len × vocab_size) for statistics
- **Greedy generation**: Adds minimal overhead (single forward pass per prompt)
- **Top-k storage**: Default k=10 is sufficient; increase for detailed analysis

### Speed

Approximate throughput (on A100 GPU):

- **Small grid (N=20)**: ~1-2 minutes for 441 pairs × 3 templates
- **Medium grid (N=100)**: ~30-60 minutes for 10,201 pairs × 1 template
- **Stratified (1000 pairs)**: ~5-10 minutes

Use `--dtype bfloat16` for ~2× speedup with minimal accuracy impact.

## Troubleshooting

### Out of Memory

```bash
# Use smaller batch or lower precision
python -m mechinterp_qwen3.dataset_generation \
  --dtype bfloat16 \
  ...
```

### Slow Generation

```bash
# Disable greedy generation if not needed
python -m mechinterp_qwen3.dataset_generation \
  # Omit --enable_greedy_generation
  ...
```

### Token Alignment Issues

The code handles multi-token answers correctly. If you see unexpected behavior:

1. Check that `add_special_tokens=False` is used (it is by default)
2. Verify tokenizer behavior with: `model.tokenizer.tokenize("123")`
3. Inspect `answer_token_strs` field in output
