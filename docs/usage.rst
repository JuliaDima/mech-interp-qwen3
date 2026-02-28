Usage Guide
===========

The primary entry point for analyzing the Qwen3-4B-Instruct model is the ``miq-build-graph`` command. This tool automates the process of running forward passes, extracting SAE features, and building an attribution graph.

Getting Started
---------------

A typical analysis job is configured as follows (e.g., in a SLURM script like ``gpu_job_mechinterp.sh``):

.. code-block:: bash

   miq-build-graph \
     --prompt "You are solving a simple comparison task. Two numbers are given: A and B. Answer with a single character: 'A' if A is larger, otherwise 'B'. A = 864, B = 394, Answer: " \
     --slug gt_864_394 \
     --layers 4,12,20 \
     --max_n_logits 10 \
     --feature_to_feature_edges True \
     --node_threshold 0.8 \
     --edge_threshold 0.85

Key Parameters
--------------

*   **--prompt**: The full text input for the model.
*   **--slug**: A unique identifier for the run. Results will be saved in ``graphs/<slug>/``.
*   **--layers**: Comma-separated list of layer IDs to include in the analysis (e.g., ``4,12,20``).
*   **--feature_to_feature_edges**: Set to ``True`` to enable inter-layer feature connectivity (see :doc:`methodology` for details).
*   **--node_threshold**: (0.0 to 1.0) The fraction of total attribution to keep when pruning nodes. Higher means a more dense graph.
*   **--edge_threshold**: (0.0 to 1.0) The fraction of total attribution to keep when pruning edges.

Outputs
-------

After completion, the command produces several files in the output directory:

*   **raw_graph.json**: The full, un-pruned attribution graph.
*   **pruned_graph.json**: The simplified graph based on your thresholds.
*   **metadata.json**: Statistics about the run, including model settings and graph density.

Dataset Generation
------------------

Before building graphs, you can generate synthetic datasets (e.g., addition problems) using the dataset generation pipeline. This tool computes teacher-forced statistics (logits and probabilities) for each answer token, providing the foundation for mechanistic interpretability analysis.

Basic Usage
~~~~~~~~~~~

Generate a grid of addition problems:

.. code-block:: bash

   python -m mechinterp_qwen3.dataset_generation \
     --model_name Qwen/Qwen2.5-3B-Instruct \
     --output_path data/addition_grid.jsonl \
     --sampling_strategy grid \
     --max_value 20 \
     --templates T0 T1 T2 \
     --seed 42

This creates a JSONL file where each line contains:

- Prompt and answer strings
- Token IDs and token strings
- Per-position statistics (logit, probability, top-k predictions)
- Metadata (model name, seed, timestamp)

Sampling Strategies
~~~~~~~~~~~~~~~~~~~

**Grid**: All (a,b) pairs from [0..N] × [0..N]

**Stratified**: Balanced by carry patterns (no-carry, single-carry, multi-carry)

.. code-block:: bash

   python -m mechinterp_qwen3.dataset_generation \
     --sampling_strategy stratified \
     --max_value 100 \
     --stratified_n_per_category 200

**Random**: N random samples from range

.. code-block:: bash

   python -m mechinterp_qwen3.dataset_generation \
     --sampling_strategy random \
     --max_value 1000 \
     --n_samples 500

For complete documentation, see :doc:`dataset_generation`.

Visualizing Results
~~~~~~~~~~~~~~~~~~~

Generate publication-quality visualizations:

.. code-block:: bash

   python -m mechinterp_qwen3.visualize_dataset \
     data/addition_grid.jsonl \
     --output_dir visualizations/grid \
     --template T0

This creates 6 visualization types:

1. **Probability heatmaps** - Model confidence across (a,b) space
2. **Carry structure analysis** - Tests explicit carry mechanism
3. **Diagonal analysis** - Tests sum-based vs digit-based strategies
4. **Entropy maps** - Distinguishes confident errors from uncertainty
5. **Error analysis** - Shows systematic biases
6. **Positional cascade** - Quantifies teacher forcing effect

See :doc:`dataset_generation` for detailed interpretation guide.
