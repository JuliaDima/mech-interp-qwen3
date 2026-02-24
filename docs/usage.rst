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
