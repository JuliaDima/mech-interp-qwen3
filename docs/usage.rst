Usage Guide
===========

The primary entry point for analyzing the Qwen3-4B-Instruct model is the ``miq`` command. This unified CLI provides tools for dataset generation and attribution analysis.

Attribution Analysis
--------------------

The ``miq attribute`` command automates the process of running forward passes, extracting SAE features, and building an attribution graph.

.. code-block:: bash

   miq attribute \
     --prompt "calc: 36+59=" \
     --slug addition_36_59 \
     --transcoder_set mwhanna/qwen3-4b-transcoders \
     --dtype bfloat16 \
     --batch_size 256 \
     --max_feature_nodes 7500 \
     --graph_file_dir graphs/

Key Parameters
~~~~~~~~~~~~~~

*   **--prompt**: The full text input for the model.
*   **--slug**: A unique identifier for the run. Results will be saved in ``graphs/``.
*   **--transcoder_set**: The HuggingFace repository ID containing the transcoders.
*   **--dtype**: Model precision (``float32``, ``float16``, ``bfloat16``).
*   **--batch_size**: Batch size for backward passes.
*   **--max_feature_nodes**: Maximum number of feature nodes to include in the graph.
*   **--graph_file_dir**: Path to save the output JSON graph files.

Outputs
~~~~~~~

After completion, the command produces several files in the output directory:

*   **nodes.json**: The list of transcoder features and logits identified as part of the circuit.
*   **edges.json**: The directed edges showing attribution weights between nodes.
*   **metadata.json**: Statistics about the run, including model settings and graph density.

Dataset Generation
------------------

The ``miq generate-dataset`` command is used to generate synthetic datasets (e.g., addition problems) with ground-truth model statistics.

.. code-block:: bash

   miq generate-dataset \
     --model Qwen/Qwen3-4B \
     --output_path data/addition_grid.jsonl \
     --sampling_strategy grid \
     --max_value 20 \
     --templates T0

Dataset Visualization
---------------------

The ``miq visualize-dataset`` command provides advanced visualizations for behavioral analysis.

.. code-block:: bash

   miq visualize-dataset \
     data/addition_grid.jsonl \
     --output_dir plots/ \
     --template T0

This workflow follows Anthropic's mechanistic interpretability methodology. For a detailed guide on sampling strategies and visualizations, see :doc:`dataset_generation`.

Experiment Modules
------------------

For specialized reproduces (like the Addition Case Study), you can use the experiment-specific runners:

.. code-block:: bash

   # Run everything for the addition reproduction
   python experiments/addition/run.py --all

See :doc:`carry_discovery` for the scientific background and details of this experiment.
