Accuracy Sweep Performance
==========================

The accuracy sweep is a critical pre-discovery step. To balance thoroughness with development speed, the system provides several execution modes.

Execution Modes
---------------

Quick Run (Sampling)
~~~~~~~~~~~~~~~~~~~~

The quick run validates the model's behavior on a random subset of 1,000 prompts from the 100x100 grid.

*   **Command**: ``python experiments/addition/accuracy_sweep.py --all --quick``
*   **Best for**: Initial verification of a new prompt format.
*   **Latency**: ~2 minutes (CPU) / ~30 seconds (GPU).
*   **Statistical Coverage**: 10% of the total space.

Full Sweep (Exhaustive)
~~~~~~~~~~~~~~~~~~~~~~~

The full sweep evaluates the model on every possible 2-digit addition prompt (10,000 total).

*   **Command**: ``python experiments/addition/accuracy_sweep.py --all``
*   **Best for**: Final verification before publishing or running large-scale attribution graphs.
*   **Latency**: ~20 minutes (CPU) / ~3-5 minutes (GPU).
*   **Statistical Coverage**: 100% of the total space.

Optimization: Batched Inference
-------------------------------

To prevent high execution times (the original unoptimized implementation took ~11 hours), the script uses **batched inference**.

Mechanism
~~~~~~~~~

Instead of processing one prompt at a time, we group prompts into batches:

1.  **Padding**: Prompts of different lengths (e.g., "0+0" vs "99+99") are padded to the maximum length in the batch.
2.  **Attention Masking**: The model ignores the padding tokens during the forward pass.
3.  **Logit Extraction**: We extract the logits from the last non-padding position for each individual prompt.

Tuning Batch Size
~~~~~~~~~~~~~~~~~

You can adjust the ``--batch_size`` (default: 32) to match your hardware:

*   **CPU**: Keep at 32 for balanced memory usage.
*   **A100/H100 GPU**: Increase to 128 or 256 for maximum throughput.
*   **Low-VRAM GPU**: Decrease to 8 or 16 if you encounter Out-of-Memory (OOM) errors.

.. code-block:: bash

   # Example: High-throughput GPU run
   python experiments/addition/accuracy_sweep.py --all --batch_size 128

Time Estimates
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Mode
     - Device
     - Batch Size
     - Time
   * - Full Sweep
     - CPU
     - 32
     - ~20 mins
   * - Quick Mode
     - CPU
     - 32
     - ~2 mins
   * - Full Sweep
     - GPU
     - 64
     - ~3-5 mins
   * - Quick Mode
     - GPU
     - 64
     - ~30 secs

Decision Guide
--------------

*   **Switching prompt formats?** Use Quick Mode first.
*   **Ready for Attribution?** Use Full Sweep to generate the ``verified_prompts.txt`` list.
*   **Running on a local laptop?** Use Quick Mode with default batch size.
*   **Running on a compute cluster?** Use Full Sweep with high batch size.
