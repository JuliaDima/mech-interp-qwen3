mechinterp-qwen3 documentation
================================

Welcome to the documentation for the **mechinterp-qwen3** pipeline!

This project provides an independent mechanistic interpretability pipeline specifically optimized for the Qwen3-4B-Instruct model.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   usage
   dataset_generation
   methodology
   api/modules

.. toctree::
   :maxdepth: 1
   :caption: Experiments:

   addition_experiment

Features
--------

* **Dataset Generation**: Production-quality tools for generating controlled datasets with teacher-forced statistics and multiple sampling strategies (grid, stratified by carry patterns, random).
* **Advanced Visualizations**: 6 visualization types for hypothesis generation, including novel entropy maps, carry structure analysis, and positional cascades.
* **Linearized Gradient Flow**: Attribution matching the *Attribution Graphs* paper methodology.
* **SAE Integration**: Native support for sparse autoencoders via transcoders.
* **Inter-Layer Circuits**: Ability to compute feature-to-feature connectivity across different model layers.
* **Reproducible Workflows**: Deterministic runs with seed control for scientific reproducibility.

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
