# Quick Start: Dataset Generation

## Installation

Ensure you have the project installed:

```bash
cd /home/eid23/mechinterp-qwen-3B-Instruct/mechinterp-qwen3
pip install -e .
```

## Basic Usage

### 1. Quick Test (Grid 0-10, Single Template)

```bash
python -m mechinterp_qwen3.dataset_generation \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --output_path data/test.jsonl \
  --sampling_strategy grid \
  --max_value 10 \
  --templates T0 \
  --seed 42
```

**Output**: 121 records, ~1 minute

### 2. Standard Grid (0-20, All Templates)

```bash
python -m mechinterp_qwen3.dataset_generation \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --output_path data/addition_grid_20.jsonl \
  --sampling_strategy grid \
  --max_value 20 \
  --templates T0 T1 T2 \
  --enable_greedy_generation \
  --seed 42
```

**Output**: 1,323 records, ~5 minutes

### 3. Stratified Sampling (Balanced Carry Patterns)

```bash
python -m mechinterp_qwen3.dataset_generation \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --output_path data/carry_balanced.jsonl \
  --sampling_strategy stratified \
  --max_value 100 \
  --templates T0 \
  --stratified_n_per_category 200 \
  --stratified_uniform_remainder 100 \
  --seed 42
```

**Output**: ~700 records, ~3 minutes

### 4. For Qwen3-4B Model (as specified in project)

```bash
python -m mechinterp_qwen3.dataset_generation \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --output_path data/qwen3_addition.jsonl \
  --sampling_strategy grid \
  --max_value 30 \
  --templates T0 T1 \
  --dtype bfloat16 \
  --enable_greedy_generation \
  --seed 42
```

**Output**: 2,883 records

## Verify Output

```bash
# Check line count
wc -l data/addition_grid_20.jsonl

# Inspect first record
head -n 1 data/addition_grid_20.jsonl | python -m json.tool

# View summary (already printed during generation)
```

## Python API Usage

```python
from pathlib import Path
from mechinterp_qwen3.dataset_generation import (
    DatasetConfig,
    SamplingStrategy,
    TemplateID,
    generate_dataset,
    write_dataset,
)

config = DatasetConfig(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    output_path=Path("my_dataset.jsonl"),
    templates=[TemplateID.T0],
    sampling_strategy=SamplingStrategy.GRID,
    max_value=10,
    seed=42,
)

records, summary = generate_dataset(config)
write_dataset(records, summary, config.output_path)
```

## Next Steps

Once you have the dataset:

1. **Load and explore**: See `DATASET_GENERATION_README.md` for loading examples
2. **Train SAEs**: Use the prompts to collect activations at target layers
3. **Build attribution graphs**: Use per-position logits as attribution targets
4. **Validate circuits**: Compare against carry pattern classifications

## Full Documentation

See [DATASET_GENERATION_README.md](DATASET_GENERATION_README.md) for complete documentation.
