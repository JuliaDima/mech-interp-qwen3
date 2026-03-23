Configuration System
====================

The ``mechinterp-qwen3`` project uses a centralized and hierarchical configuration system based on YAML files. This ensures consistency across different experimental runs, cluster jobs, and CLI tool usage.

The Single Source of Truth
--------------------------

The root of the repository contains a ``config.yaml`` file. This is the **primary configuration source** for the entire project. It defines global defaults (like ``seed`` and ``dtype``) and component-specific sections (like ``generate_dataset`` or ``addition_experiment``).

Priority Chain
--------------

When you run a command, parameters are resolved using the following priority (from lowest to highest):

1.  **Code Defaults**: Hardcoded default values in the Python argument parsers.
2.  **Global Defaults**: Top-level values in the root ``config.yaml``.
3.  **Sectional Defaults**: Values inside a specific section of ``config.yaml`` (e.g., ``addition_experiment:``).
4.  **Custom Config File**: A YAML file provided via the ``--config`` flag or as a positional argument.
5.  **CLI Flags**: Explicitly provided command-line arguments (e.g., ``--dtype float32``).

Hierarchy and Merging
---------------------

The system uses a recursive merging logic (``merge_configs``) to handle overrides efficiently.

Sectional Overrides
~~~~~~~~~~~~~~~~~~~

Components inherit global settings but can override them. For example:

.. code-block:: yaml

   seed: 42
   dtype: "bfloat16"

   addition_experiment:
     dtype: "float32"  # Overrides global bfloat16 for this experiment

When running the addition experiment, the system merges the global dictionary with the ``addition_experiment`` dictionary, ensuring the specific override is applied while keeping the global ``seed``.

Custom Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can provide your own sparse configuration file to override specific values without redefining the entire project:

.. code-block:: bash

   python experiments/addition/run.py my_setup.yaml --all

If ``my_setup.yaml`` only contains a ``seed: 123``, the system will merge it into the existing configuration, keeping all other project-wide defaults intact.

The Universal Sbatch Wrapper
----------------------------

For cluster execution, the ``scripts/sbatch_run.sh`` script acts as a universal wrapper. It preserves the configuration system by forwarding all arguments to the underlying commands.

.. code-block:: bash

   # Uses root config.yaml defaults
   sbatch scripts/sbatch_run.sh python experiments/addition/run.py --all

   # Uses custom overrides
Stitching Experiment
~~~~~~~~~~~~~~~~~~~~

Key settings for the SAE-mediated stitching pipeline:

.. code-block:: yaml

   stitching_experiment:
     hub_model: "PhilipQuirke/QuantaMaths_add_d5_l1_h3_t15K_s372001"
     small_sae_d_transcoder: 4096
     small_sae_lr: 1e-3
     stitch_layer_pairs: [14, 16, 18]
     num_verify_samples: 1000
