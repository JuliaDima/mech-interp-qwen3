Dataset Generation and Visualization
=====================================

This guide covers the complete workflow for generating addition datasets and visualizing model behavior, which forms the foundation for mechanistic interpretability analysis.

Overview
--------

The dataset generation pipeline is a production-quality tool for generating controlled datasets for mechanistic interpretability. It allows researchers to:

1. **Configurable prompt templates** for arithmetic tasks (T0, T1, T2).
2. **Multiple sampling strategies**:
    - **Grid Sampling**: Complete coverage for small ranges (N ≤ 100).
    - **Stratified Sampling**: Balanced representation of carry patterns (no-carry, single-carry, multi-carry).
    - **Random Sampling**: Quick experiments for large ranges.
3. **Teacher-forced statistics**:
    - Per-token logits and probabilities.
    - Top-k predictions (defaults to k=10).
    - Proper position alignment for causal decoding models.
4. **Accuracy Validation**: Mandatory greedy generation to verify model performance on the task.
5. **Batched Processing**: High-throughput generation using optimized batches.

.. important::
   **Teacher forcing is used ONLY for preliminary behavioral analysis and visualization, NOT for circuit discovery.**

   The actual carry discovery experiments (in ``experiments/addition/``) use **causal interventions**
   (activation patching) and **gradient-based attribution**, neither of which require teacher forcing.

   Teacher forcing helps to understand:

   - Which examples are difficult for the model
   - Behavioral patterns across carry types
   - Generate hypotheses before diving into circuit analysis

   But all circuit discovery work uses standard forward passes with attribution/intervention techniques.

Architecture
------------

The system is split into two core modules, now located within the addition experiment directory:

*   **``experiments/addition/dataset_generation/generate_add_dataset.py``**: Contains the core logic for sampling, prompt building, and scoring.
*   **``experiments/addition/dataset_generation/__main__.py``**: The CLI entrypoint for standalone runs.

These are also accessible via the global ``miq`` command:

*   ``miq generate-dataset``: Wrapper for the generation logic.
*   ``miq visualize-dataset``: Wrapper for the visualization suite.

Quick Start
-----------

Generate a basic addition dataset using the ``miq`` CLI:

.. code-block:: bash

   miq generate-dataset \
     --model Qwen/Qwen3-4B \
     --output_path data/addition_grid.jsonl \
     --sampling_strategy grid \
     --max_value 20 \
     --templates T0 T1 T2 \
     --batch_size 32 \
     --seed 42

Visualize the results:

.. code-block:: bash

   miq visualize-dataset \
     data/addition_grid.jsonl \
     --output_dir visualizations/grid \
     --template T0

Prompt Templates
----------------

Three configurable templates are provided for addition tasks:

**T0**: ``"calc: {a}+{b}="``
    Minimal template without spaces

**T1**: ``"calc: {a} + {b} ="``
    Template with spaces around operators

**T2**: ``"What is {a}+{b}? Answer:"``
    Natural language template

Templates allow testing whether model behavior depends on surface-level formatting.

Sampling Strategies
-------------------

Grid Sampling
~~~~~~~~~~~~~

Generates all possible (a, b) pairs over [0..N] × [0..N].

**Use case**: Complete coverage for small N (e.g., N ≤ 100)

.. code-block:: bash

   python -m mechinterp_qwen3.dataset_generation \
     --model Qwen/Qwen3-4B \
     --output_path data/addition_grid_20.jsonl \
     --sampling_strategy grid \
     --max_value 20 \
     --templates T0 \
     --seed 42

**Output**: 21 × 21 = 441 unique (a, b) pairs

Stratified Sampling
~~~~~~~~~~~~~~~~~~~

Samples by carry patterns to ensure balanced representation:

- **No-carry**: e.g., 12 + 34 = 46
- **Single-carry**: e.g., 15 + 18 = 33
- **Multi-carry**: e.g., 99 + 99 = 198

**Use case**: Balanced dataset for studying carry mechanisms at larger N

.. code-block:: bash

   python -m mechinterp_qwen3.dataset_generation \
     --model Qwen/Qwen3-4B \
     --output_path data/addition_stratified.jsonl \
     --sampling_strategy stratified \
     --max_value 100 \
     --templates T0 \
     --stratified_n_per_category 200 \
     --stratified_uniform_remainder 100 \
     --batch_size 32 \
     --seed 42

**Output**: ~700 unique pairs (200 per carry category + 100 random)

Random Sampling
~~~~~~~~~~~~~~~

Pure random sampling with specified sample count.

**Use case**: Quick experiments or very large N

.. code-block:: bash

   python -m mechinterp_qwen3.dataset_generation \
     --model Qwen/Qwen3-4B \
     --output_path data/addition_random.jsonl \
     --sampling_strategy random \
     --max_value 1000 \
     --n_samples 500 \
     --templates T1 \
     --batch_size 64 \
     --seed 42

**Output**: 500 random pairs from [0..1000] × [0..1000]

Teacher-Forced Statistics
-------------------------

.. note::
   **Purpose of Teacher Forcing**: This section describes teacher-forced statistics used
   for **exploratory data analysis only**. These statistics help identify interesting examples
   and behavioral patterns but are **not used in circuit discovery**.

   The circuit discovery experiments (``experiments/addition/``) use gradient-based attribution
   and causal interventions, which do not require teacher forcing.

For each prompt-answer pair, the system computes per-position statistics using a single forward pass:

Process
~~~~~~~

1. Tokenize prompt and answer separately
2. Concatenate: ``[prompt_tokens, answer_tokens]``
3. Run single forward pass (no autoregressive generation)
4. Extract logits at each position
5. Compute statistics for correct token at each position

**Why teacher forcing for visualization?** Teacher forcing allows you to measure the difficulty
of each output position *independently*, isolating whether errors come from a specific digit
position (e.g., tens vs. ones) rather than error accumulation from previous tokens.

Output Format
~~~~~~~~~~~~~

Each JSONL record contains:

.. code-block:: json

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
         "topk_ids": [9012, 1111, ...],
         "topk_strs": ["4", "5", ...],
         "topk_probs": [0.95, 0.03, ...]
       },
       ...
     ],
     "greedy_completion_str": "46",
     "metadata": {
       "model_name": "Qwen/Qwen3-4B",
       "seed": 42,
       "dtype": "float32",
       "device": "cuda:0",
       "timestamp": "2026-02-28T12:00:00"
     }
   }

Key Statistics
~~~~~~~~~~~~~~

For each answer token position:

- **true_id**: The correct token ID
- **true_str**: The correct token string
- **logit_true**: Logit value for correct token
- **prob_true**: Probability assigned to correct token
- **topk_ids/strs/probs**: Top-k predictions (k=10 by default)

Position Alignment
~~~~~~~~~~~~~~~~~~

The code correctly aligns prompt and answer positions:

- **Position 0**: First answer token predicted by logits at ``prompt_len - 1``
- **Position i > 0**: Token i predicted by logits at ``prompt_len + i - 1``

This ensures proper teacher-forcing where each token prediction sees all previous tokens.

Understanding Position 0 Probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may observe low probability for the first answer token (Position 0):

.. code-block:: text

   Position 0: 0.0868 (8.68%)
   Position 1: 0.8284 (82.84%)

**This is expected behavior**, not a bug:

- Position 0 predicts from prompt only (no answer context)
- Subsequent positions use teacher forcing (see previous answer tokens)
- Uncertainty naturally decreases with more context

This is valuable data showing how confidence cascades through the sequence.

Advanced Visualizations
-----------------------

The visualization suite provides 6 analysis types inspired by Anthropic's work:

1. Probability Heatmap
~~~~~~~~~~~~~~~~~~~~~~

**What it shows**: Model's confidence P(correct) for each (a, b) pair as a 2D heatmap.

**Scientific insight**:

- Reveals geometric structure in model's representations
- Tests commutativity (symmetry across diagonal)
- Identifies "easy" vs "hard" regions

.. code-block:: bash

   python -m mechinterp_qwen3.visualize_dataset \
     data/addition_grid.jsonl \
     --output_dir visualizations/grid \
     --template T0

**Output**: ``heatmap_pos0.png``, ``heatmap_pos1.png``, ...

2. Carry Structure Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**What it shows**: Carry patterns (no-carry, single-carry, multi-carry) overlaid with model probability.

**Scientific insight**:

- Tests if model learned explicit carry mechanism
- Shows correlation between carry patterns and confidence
- If probability drops at carry boundaries → model doesn't handle carries well

**Output**: ``carry_structure.png``

3. Diagonal Analysis
~~~~~~~~~~~~~~~~~~~~~

**What it shows**: Mean P(correct) as function of sum (a+b).

**Scientific insight**:

- Diagonals test if model uses sum-based vs operand-based strategies
- Smooth curve → model uses sum in computation
- Noisy/spiky → model uses individual digits

**Output**: ``diagonal_analysis.png``

4. Entropy Map (Novel)
~~~~~~~~~~~~~~~~~~~~~~~

**What it shows**: Prediction entropy (uncertainty) across the grid.

**Scientific insight**:

- Low entropy + low P(correct) → **systematic error** (confidently wrong)
- High entropy + low P(correct) → **genuine uncertainty** (model doesn't know)
- Reveals qualitatively different failure modes

**Output**: ``entropy_map.png``

5. Error Analysis
~~~~~~~~~~~~~~~~~

**What it shows**: Three panels showing where model is correct/incorrect, true digits, and predicted digits.

**Scientific insight**:

- Identifies systematic vs random errors
- Shows if model has consistent biases
- Reveals if errors cluster geometrically

**Output**: ``error_analysis.png``

6. Positional Cascade (Novel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**What it shows**: Side-by-side heatmaps for positions 0, 1, 2... showing confidence increase.

**Scientific insight**:

- Demonstrates teacher forcing effect visually
- Quantifies information flow through answer tokens
- Big jump pos 0→1 → model uses first digit heavily

**Output**: ``positional_cascade.png``

Programmatic Usage
------------------

Python API for custom analysis:

.. code-block:: python

   from pathlib import Path
   from mechinterp_qwen3.visualize_dataset import (
       load_dataset,
       create_probability_heatmap,
       create_carry_structure_plot,
       create_entropy_map,
   )

   # Load data
   records = load_dataset(Path("data/addition_grid.jsonl"))

   # Create specific visualization
   fig, ax = create_probability_heatmap(
       records,
       template_id="T0",
       position=0,
       output_path=Path("my_heatmap.png"),
       figsize=(15, 12)
   )

   # Create carry analysis
   fig, (ax1, ax2) = create_carry_structure_plot(
       records,
       template_id="T0",
       position=0,
       output_path=Path("carry_analysis.png")
   )

Finding Interesting Cases
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   # Find confident-wrong cases (low entropy, low probability)
   confident_wrong = []

   for rec in records:
       if rec["template_id"] != "T0":
           continue

       pos0 = rec["per_pos"][0]
       prob = pos0["prob_true"]

       # Compute entropy
       topk_probs = np.array(pos0["topk_probs"])
       entropy = -np.sum(topk_probs * np.log2(topk_probs + 1e-10))

       # Find systematic errors
       if entropy < 0.5 and prob < 0.3:
           confident_wrong.append((rec["a"], rec["b"], prob, entropy))

   print(f"Found {len(confident_wrong)} confident-wrong cases")
   print("These are good targets for circuit analysis!")

Integration with Attribution Graphs
------------------------------------

The dataset feeds into attribution graph construction:

Workflow
~~~~~~~~

1. **Generate dataset** with teacher-forced statistics
2. **Visualize patterns** to form hypotheses (e.g., "model uses carry mechanism")
3. **Select interesting cases** (high-error vs high-confidence)
4. **Build attribution graphs** for selected cases
5. **Compare graphs** across conditions (carry vs no-carry)

Example: Carry Circuit Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Generate dataset
   python -m mechinterp_qwen3.dataset_generation \
     --output_path data/addition_grid.jsonl \
     --sampling_strategy grid \
     --max_value 20 \
     --templates T0

   # 2. Visualize carry structure
   python -m mechinterp_qwen3.visualize_dataset \
     data/addition_grid.jsonl \
     --output_dir visualizations/carry_analysis

   # 3. Identify carry vs no-carry examples from visualization

   # 4. Build attribution graphs
   miq-build-graph \
     --prompt "calc: 8+9=" \
     --slug carry_example_8_9 \
     --layers 4,12,20

   miq-build-graph \
     --prompt "calc: 1+2=" \
     --slug no_carry_example_1_2 \
     --layers 4,12,20

   # 5. Compare graphs to find carry-specific features

CLI Reference
-------------

Complete options for dataset generation:

.. code-block:: bash

   python -m mechinterp_qwen3.dataset_generation \
     --model Qwen/Qwen3-4B \
     --output_path data/dataset.jsonl \
     --sampling_strategy {grid,stratified,random} \
     --max_value 100 \
     --templates T0 T1 T2 \
     --top_k 10 \
     --batch_size 32 \
     --seed 42 \
     --dtype {float32,float16,bfloat16} \
     --device {cuda,cpu}

Key Parameters
~~~~~~~~~~~~~~

**Model Configuration**:
  - ``--model``: HuggingFace model name (default: Qwen/Qwen3-4B)
  - ``--device``: Device (cuda/cpu, auto-detects if not specified)
  - ``--dtype``: Model precision (float32/float16/bfloat16)

**Sampling**:
  - ``--sampling_strategy``: grid/stratified/random
  - ``--max_value``: Maximum value for a and b
  - ``--n_samples``: Number of samples (required for random)
  - ``--stratified_n_per_category``: Samples per carry category
  - ``--stratified_uniform_remainder``: Additional uniform samples

**Statistics**:
  - ``--top_k``: Number of top-k tokens to store (default: 10)
  - ``--batch_size``: Batch size for generation (default: 32)
  - ``--max_gen_tokens``: Max tokens for greedy generation

**Reproducibility**:
  - ``--seed``: Random seed (default: 42)

Performance Considerations
--------------------------

Memory
~~~~~~

- Grid sampling: O(N²) for pairs
- Teacher-forced stats: O(seq_len × vocab_size)
- Use ``--dtype bfloat16`` for ~2× speedup with minimal accuracy impact

Throughput
~~~~~~~~~~

The system uses batched generation for high throughput. Approximate on A100 GPU:

- Small grid (N=20): <1 minute for 441 pairs × 3 templates
- Medium grid (N=100): 5-10 minutes for 10,201 pairs (with ``batch_size=128``)
- Stratified (1000 pairs): 1-2 minutes

Deterministic Runs
------------------

All runs are fully deterministic with seed control:

- PyTorch, NumPy, Python random seeds
- CUDA deterministic algorithms
- Controlled sampling order

This ensures perfect reproducibility for scientific experiments.

Further Reading
---------------

- `DATASET_GENERATION_README.md` - Comprehensive technical guide
- `QUICKSTART_DATASET.md` - Quick reference with examples
- `VISUALIZATION_GUIDE.md` - Detailed visualization documentation
- `example_visualizations.py` - Python API examples

See Also
--------

- :doc:`usage` - General usage guide
- :doc:`methodology` - Attribution methodology
- :doc:`api/modules` - API reference
