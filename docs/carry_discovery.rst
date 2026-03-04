Carry Discovery (Addition Case Study)
=====================================

This module reproduces the addition mechanistic interpretability case study from Anthropic's "On the Biology of a Large Language Model" (Transformer Circuits Thread, 2025). The goal is to discover and validate the "circuits" that a model uses to perform arithmetic, specifically focusing on how it handles **carries**.

Scientific Context
------------------

Anthropic's research found that while models can explain human-like carry addition in natural language, their internal circuits often resemble **modular arithmetic lookup tables**. This case study reproduction provides the tools to verify these findings on the Qwen3-4B model.

Experimental Protocol
---------------------

The reproduction consists of four distinct phases, orchestrated by the ``experiments/addition/run.py`` script.

1. Prompt Suite Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The suite comprises ~10,000 prompts in the format ``"calc: {a}+{b}="`` for all :math:`a, b \in \{0, \dots, 99\}`. This grid allows for comprehensive mapping of model activations across the entire 2-digit addition space.

.. code-block:: bash

   python experiments/addition/run.py --make-prompts

2. Operand Plots (Behavioral Mapping)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For every feature $f$ in a transcoder, we generate a 100×100 heatmap where the value at $(a, b)$ is the feature's activation at the ``=`` token position.

**Patterns to look for:**

*   **Vertical/Horizontal bands**: Sensitivity to a single operand.
*   **Diagonal stripes**: Encoding the sum :math:`a+b`.
*   **Repeating Mod-10 grids**: Modular arithmetic (ones/tens digit encoding).
*   **Isolated points**: Lookup table entries for specific number pairs.

.. code-block:: bash

   python experiments/addition/run.py --operand-plots

3. Attribution Graph Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We build a pruned computational graph for a specific focus case (default: ``36+59=95``). The graph attributes influence from the output logit back through transcoder features to the input.

**Supernode Proposal**: The system automatically groups features into "supernodes" based on their activation patterns and attribution scores, matching the "ones digit lookup", "tens carry", etc., labels from Anthropic's work.

.. code-block:: bash

   python experiments/addition/run.py --graph

4. Intervention Validation (Constrained Patching)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To prove a feature is truly part of a circuit, we perform **constrained patching**. We clamp the activations to a "perturbed" run (e.g., ``36+60=``) up to an intervention layer, then inhibit specific features and measure the effect on the final logit.

.. code-block:: bash

   python experiments/addition/run.py --intervene

Implementation Details
----------------------

Position Alignment
~~~~~~~~~~~~~~~~~~

The critical activation position is the **last token of the prompt** (the ``=`` token). This is where the model has processed the operands and is preparing to produce the first digit of the answer.

Feature Naming
~~~~~~~~~~~~~~

Features are identified by their layer and local index (e.g., ``L12_F543``). The ``operand_plots_summary.json`` provides a rank-ordered list of features by activation strength.

Reproduction Differences
------------------------

While the protocol is as faithful as possible, there are key differences from the original Anthropic study:

*   **Model**: Qwen3-4B uses a different tokenizer than Anthropic's internal models. One-token vs. multi-token answers are handled in ``expected_answers.py``.
*   **Transcoders**: We use Per-Layer Transcoders (PLTs), resulting in longer attribution paths than the cross-layer transcoders used by Anthropic.

Faithfulness across Formats
---------------------------

The module compares the "structured" circuit (``calc: 36+59=``) with "natural language" variants (``What is 36+59? Answer:``). This tests whether the same internal circuit is reused regardless of the input surface format.

Usage Guide
-----------

For detailed CLI options and log output descriptions, see the :doc:`addition_experiment` section or the project's root ``config.yaml``.
