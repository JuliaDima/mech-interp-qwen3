mechinterp-qwen3 documentation
================================

Welcome to the documentation for the **mechinterp-qwen3** pipeline!

This project provides an independent mechanistic interpretability pipeline specifically optimized for the Qwen3-4B-Instruct model.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   usage
   configuration
   dataset_generation
   observations
   methodology
   api/modules

.. toctree::
   :maxdepth: 1
   :caption: Experiments:

   carry_discovery
   accuracy_sweep
   accuracy_sweep_performance
   robustness_experiment
   stitching_experiment

Features
--------

* **Circuit Discovery via Attribution & Intervention**: Gradient-based attribution and causal interventions (activation patching) to discover computational circuits—**no teacher forcing required** for circuit discovery.
* **Dataset Generation for Exploration**: Production-quality tools for generating controlled datasets with teacher-forced statistics (for preliminary behavioral analysis only) and multiple sampling strategies (grid, stratified by carry patterns, random).
* **Advanced Visualizations**: 6 visualization types for hypothesis generation, including novel entropy maps, carry structure analysis, and positional cascades.
* **Linearized Gradient Flow**: Attribution matching the *Attribution Graphs* paper methodology.
* **SAE Integration**: Native support for sparse autoencoders via transcoders.
* **Inter-Layer Circuits**: Ability to compute feature-to-feature connectivity across different model layers.
* **Reproducible Workflows**: Deterministic runs with seed control for scientific reproducibility.

.. important::
   **Teacher Forcing vs. Circuit Discovery**

   This project distinguishes between two separate methodologies:

   1. **Circuit Discovery** (``experiments/addition/``): Uses gradient-based attribution and causal
      interventions. **Does NOT use teacher forcing**. Forward passes use only prompts
      (e.g., ``"calc: 36+59="``), and attribution is computed from output logits.

   2. **Behavioral Analysis** (``experiments/addition/dataset_generation/``): Optional teacher-forced
      statistics for exploratory visualization and hypothesis generation. Helps identify
      interesting examples before circuit analysis.

   See :doc:`carry_discovery` and :doc:`dataset_generation` for details.

.. tip::
   **Best Practice: Verify Model Accuracy First**

   Before circuit discovery, always run an **accuracy sweep** to verify the model actually
   solves the task on your chosen prompt format:

   .. code-block:: bash

      python experiments/addition/accuracy_sweep.py --all --quick

   If accuracy < 80%, you should change the prompt format before analyzing circuits.
   See :doc:`carry_discovery` (Phase 0: Accuracy Sweep) for details.

Installation
------------

To install the dependencies for building this documentation:

.. code-block:: bash

   pip install -e ".[docs]"

Building the Docs
-----------------

.. code-block:: bash

   cd docs
   make html
   # or
   sphinx-build -b html . _build/html
