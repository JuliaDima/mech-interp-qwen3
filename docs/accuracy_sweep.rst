Accuracy Sweep
==============

Before attempting circuit discovery, verify that the model actually solves the task on your chosen prompt format.

Quick Start
-----------

The accuracy sweep is now a standalone script. Run it before any circuit discovery:

.. code-block:: bash

   # Recommended: Run all phases with quick sampling first
   python experiments/addition/accuracy_sweep.py --all --quick

   # Or run a full exhaustive sweep
   python experiments/addition/accuracy_sweep.py --all

Running the Sweep
-----------------

Quick vs. Full Runs
~~~~~~~~~~~~~~~~~~~

Depending on your time and resource constraints, you can choose between a quick validation or a comprehensive sweep.

**Quick Run (Sampling)**
  Tests the model on a random subset of 1,000 prompts (out of 10,000). Useful for rapid iteration on prompt formats.

  .. code-block:: bash

     python experiments/addition/accuracy_sweep.py --all --quick

**Full Sweep (Exhaustive)**
  Tests **every possible prompt** in the 100x100 grid (10,000 prompts). Ensures the "verified prompts" list is complete.

  .. code-block:: bash

     python experiments/addition/accuracy_sweep.py --all

Optimization: Batched Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To speed up both quick and full runs, the script uses **batched inference**. You can tune the batch size based on your GPU's VRAM:

.. code-block:: bash

   # High-performance run (e.g., A100/H100)
   python experiments/addition/accuracy_sweep.py --all --batch_size 256

*   **Default**: 32
*   **Tip**: Increasing the batch size significantly reduces the execution time by saturating GPU utilization.

See :doc:`accuracy_sweep_performance` for detailed benchmarks and tuning tips.

Why This Matters
----------------

**Don't discover circuits in models that don't solve the task!**

Following best practices in mechanistic interpretability:

   *"Choose a prompt distribution where the model genuinely solves the task, and then condition
   your analysis on correct cases."*

If your model doesn't perform carry addition correctly, there's no "carry circuit" to discover—you'd
be analyzing error modes instead of the intended computation.

What the Sweep Does
-------------------

Phase 1: Tokenization Check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Verifies how answers tokenize:

- Single-token: ``"9"`` → ``["9"]``
- Multi-token: ``"95"`` → ``["9", "5"]`` or ``["95"]``?
- Multi-token: ``"100"`` → ``["1", "00"]`` or ``["100"]``?

**Output**: ``tokenization.json``

Phase 2: Accuracy Sweep
~~~~~~~~~~~~~~~~~~~~~~~~

Tests greedy decoding (argmax at ``=`` position) on all prompts:

- Measures: Does ``argmax(logits[-1])`` match the correct first token?
- Reports: Overall accuracy, per-prompt results, failure cases
- **No teacher forcing**: Tests actual generation, not confidence on ground truth.

**Output**: ``accuracy_sweep.json``

Phase 3: Filter Verified Prompts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selects only prompts where:

- Model's prediction is correct (argmax matches ground truth)

**Output**: ``verified_prompts.txt`` (use for downstream analysis!)

Decision Criteria
-----------------

After running the sweep, check the accuracy:

**Accuracy ≥ 80%** ✓
  Proceed with circuit discovery. The model solves the task reliably.

**Accuracy < 80%** ⚠️
  **DO NOT proceed!** Change the prompt format first (e.g., try spacing or natural language).

  - Try spacing: ``"calc: 36 + 59 = "``
  - Try natural language: ``"What is 36+59? Answer:"``
  - Try few-shot: ``"calc: 12+34=46\ncalc: 36+59="``
  - Try explicit delimiter: ``"36+59=<answer>"``

  Then re-run the accuracy sweep until accuracy > 80%.

**Specific focus prompt fails** ⚠️
  Pick a different focus prompt from ``verified_prompts.txt``, or change prompt format.

Output Files
------------

The sweep creates these files in ``runs/addition/accuracy_sweep/``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - ``tokenization.json``
     - How each unique answer tokenizes
   * - ``accuracy_sweep.json``
     - Detailed per-prompt results with probabilities and margins
   * - ``verified_prompts.json``
     - Filtered prompts (with metadata)
   * - ``verified_prompts.txt``
     - Simple text list of verified prompts

Example Output
--------------

.. code-block:: text

   ==========================================================
   TOKENIZATION EXAMPLES
   ==========================================================
      0 → ['0']  (ids: [15])
      1 → ['1']  (ids: [16])
     95 → ['9', '5']  (ids: [24, 20])
    100 → ['100']  (ids: [1567])

   Multi-token answers:
     10 → ['1', '0']
     95 → ['9', '5']
    100 → ['100']

   ==========================================================
   ACCURACY SWEEP RESULTS
   ==========================================================
   Total prompts tested: 10000
   Correct predictions:  8542 (85.42%)
   Incorrect predictions: 1458

   Margin statistics (correct cases):
     Mean:   0.7234
     Median: 0.8012
     Min:    0.0123
     Max:    0.9987

   ✓  accuracy-sweep complete → runs/addition/2025-01-15_1430/accuracy_sweep
      (accuracy: 85.4%, verified: 8542/10000)

Advanced Options
----------------

Standalone Usage
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Just check tokenization
   python experiments/addition/accuracy_sweep.py --tokenization

   # Just run accuracy sweep
   python experiments/addition/accuracy_sweep.py --sweep

   # Just filter (requires sweep already run)
   python experiments/addition/accuracy_sweep.py --filter --min_prob 0.8

Custom Filtering Thresholds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Only keep prompts with P(correct) > 0.7
   python experiments/addition/accuracy_sweep.py --all --min_prob 0.7

   # Only keep prompts with large margins
   python experiments/addition/accuracy_sweep.py --all --min_margin 0.3

Using Verified Prompts
-----------------------

After the sweep, use ``verified_prompts.txt`` to restrict your analysis:

.. code-block:: python

   # Load verified prompts
   with open("runs/addition/accuracy_sweep/verified_prompts.txt") as f:
       verified = [line.strip() for line in f]

   # Filter your analysis to only these prompts
   for prompt in verified:
       # Run operand plots, attribution, etc.
       ...

This ensures you're discovering circuits for **successful computation**, not error modes.

FAQ
---

**Q: Why not use teacher forcing for this?**
  Accuracy sweep tests **actual generation** (greedy decoding), not confidence on ground truth.

**Q: What if my focus prompt (36+59=) fails?**
  Pick a different one from the verified list, or change the prompt format.

**Q: Should I analyze incorrect cases?**
  Only if you're specifically studying error modes. For primary analysis, condition on correct cases.

See Also
--------

- :doc:`carry_discovery` - Full addition case study
- :doc:`methodology` - Theoretical background
- :doc:`usage` - General usage guide
- ``experiments/addition/README_ACCURACY_SWEEP.md`` - Technical details
