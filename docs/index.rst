mechinterp-qwen3 documentation
================================

Welcome to the documentation for the **mechinterp-qwen3** pipeline!

This project provides an independent mechanistic interpretability pipeline specifically optimized for the Qwen3-4B-Instruct model.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   usage
   methodology
   api/modules

Features
--------

* **Linearized Gradient Flow**: Attribution matching the *Attribution Graphs* paper methodology.
* **SAE Integration**: Native support for sparse autoencoders via transcoders.
* **Inter-Layer Circuits**: Ability to compute feature-to-feature connectivity across different model layers.

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
