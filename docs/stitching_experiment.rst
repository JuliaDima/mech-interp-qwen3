Stitching Experiment
====================

Model stitching experiment for transferring carry computation circuits from a small addition-specialized transformer into Qwen3-4B using **SAE-mediated MLP injection**.

Overview
--------

This experiment implements a refined **affine stitching** methodology (Chen et al., arXiv 2506.06609) to test whether arithmetic circuits learned by a small transformer trained from scratch on integer addition (Quirke & Barez, ICLR 2024) can be successfully transplanted into Qwen3-4B.

Unlike basic residual stream stitching, this version uses **Option B: SAE-mediated MLP injection**. We train a Sparse Autoencoder (SAE) on the small model's MLP outputs to learn a sparse representation, then fit an affine map from these sparse features into the large model's MLP output space.

The experiment consists of seven steps:

1. **Load/Train Small Model**: Use a pretrained QuantaMaths model (PhilipQuirke/QuantaMaths) or train from scratch.
2. **Train Small SAE**: Train a ReLU-based Sparse Autoencoder on the small model's MLP outputs.
3. **Collect Small SAE Outputs**: Reconstruct small MLP outputs through the SAE bottleneck.
4. **Collect Large MLP Outputs**: Extract Qwen3-4B MLP activations on matched (a,b) arithmetic problems.
5. **Fit Affine Stitch Maps**: Learn a linear mapping between the small SAE space and large MLP space.
6. **Inject and Verify**: Patch large model MLP outputs with the mapped small model outputs and measure accuracy.
7. **Compare Attribution Graphs**: Analyze changes in feature reliance before and after stitching.

Research Questions
------------------

This experiment addresses three key questions:

**Circuit Universality**
    Do small and large models learn similar computational strategies for carry propagation, despite differences in scale and training data?

**Representational Alignment**
    Can we find linear mappings between model representations that preserve functional behavior across architectures?

**Feature Interpretability**
    Does the SAE bottleneck clarify which "concepts" from the small model are most relevant for the large model's computation?

Quick Start
-----------

.. code-block:: bash

    # Run full experiment (using pretrained QuantaMaths model)
    python experiments/stitching/run.py --all

    # Use a custom QuantaMaths model from HuggingFace
    python experiments/stitching/run.py --all --hub-model "PhilipQuirke/QuantaMaths_add_d5_l1_h3_t15K_s372001"

    # Train a small model from scratch instead
    python experiments/stitching/run.py --all --hub-model ""

    # Run individual phases
    python experiments/stitching/run.py --train-sae
    python experiments/stitching/run.py --fit-stitch
    python experiments/stitching/run.py --verify

Configuration is loaded from ``config.yaml`` under the ``stitching_experiment`` section.

The Steps
---------

Step 1: Small Addition Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The goal** is to obtain a high-accuracy addition-specialized transformer (1 or 2 layers, 3 heads).

By default, the experiment downloads a pretrained model from the **QuantaMaths** suite (Quirke & Barez 2024). These models achieve >99% accuracy on n-digit addition tasks and use a specialized per-digit tokenizer.

- **Option A**: Load from HuggingFace (``--hub-model``).
- **Option B**: Train from scratch using ``train_small_model()``.

**Output**: ``runs/stitching/small_model.pt``

Step 2: Train Small SAE
~~~~~~~~~~~~~~~~~~~~~~

**The goal** is to train a Sparse Autoencoder on the small model's MLP outputs at the best carry layer (usually layer 1).

- **d_transcoder**: Default 4096 (16x expansion for d_model=256).
- **Activation**: ReLU with Kaiming uniform initialization to avoid dead feature problems.
- **Samples**: Uses 5,000 extraction samples by default.

**Output**: ``runs/stitching/small_sae.safetensors``

Step 3 & 4: Collection (Matched Samples)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The goal** is to cache MLP activations from both models on **identical arithmetic problems**.

1. **Small Model**: Collect reconstructed MLP outputs produced by the SAE (bottlenecked).
2. **Large Model**: Collect `hook_mlp_out` at the same token position using matched `(a, b)` pairs parsed from the small model's QuantaMaths strings.

**Output**: ``runs/stitching/large_mlp_outputs.pt`` and ``runs/stitching/small_sae_outputs.pt``.

Step 5: Fit Affine Stitch Maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The goal** is to learn a linear mapping between the small and large MLP output spaces.

.. math::

    x_{\text{large}} \approx W \cdot x_{\text{small\_SAE}} + b

**Method**: Multi-output Ridge regression with Regularization (α=1e-4).

**Metrics**: R² and CCA scores reported per layer.

Step 6: Inject and Verify
~~~~~~~~~~~~~~~~~~~~~~~~~

**The goal** is to measure the causal effect of stitching small-model activations into Qwen3-4B.

**Optimization**: Use ``--num-verify-samples`` (default 1000) for faster run over the full test set.

**Metrics computed**:

- **Accuracy**: % correct on all test samples.
- **Cascading Carry Accuracy**: % correct on samples where carry propagates ≥2 digits.
- **KL Divergence**: Average KL(before || after) on output distributions.

Visualizing Small Model Predictions
------------------------------------

To understand how the small addition model processes inputs and generates predictions, you can visualize:

1. **Tokenization**: How the input prompt is converted to token IDs
2. **Prediction Grid**: Model confidence for each output token position

**Usage**:

.. code-block:: bash

    # Visualize pretrained QuantaMaths model (default)
    python experiments/stitching/visualize_small_model_predictions.py

    # Customize number of examples
    python experiments/stitching/visualize_small_model_predictions.py --n-examples 20

    # Use a different pretrained model
    python experiments/stitching/visualize_small_model_predictions.py \
        --hub-model PhilipQuirke/QuantaMaths_add_d5_l1_h3_t15K_s372001

    # Train a model from scratch and visualize
    python experiments/stitching/visualize_small_model_predictions.py --train-from-scratch

**Output Files**:

- ``runs/stitching/visualizations/tokenization_example.png``: Shows how a single input is tokenized
- ``runs/stitching/visualizations/predictions_grid.png``: Grid of 10 examples with prediction confidences

Configuration
-------------

All settings are in ``config.yaml`` under ``stitching_experiment``:

.. code-block:: yaml

    stitching_experiment:
      hub_model: "PhilipQuirke/QuantaMaths_add_d5_l1_h3_t15K_s372001"
      small_sae_d_transcoder: 4096
      small_sae_epochs: 500
      stitch_layer_pairs: [14, 16, 18]
      num_verify_samples: 1000
      cascading_carry_threshold: 2

Discussion: Why Fixed-Width Experts Fail
----------------------------------------

Initial results with the **QuantaMaths d5 expert** showed low alignment (R² ≈ 0.10) with Qwen3-4B. Based on our analysis, this is due to **Positional Stiffness**.

.. figure:: ../runs/stitching/shift_sensitivity.png
   :width: 600
   :alt: Shift Sensitivity Plot
   :align: center

   **Positional Stiffness**: The d5 expert fails badly if the input tokens are shifted, even by 1 position.

Probability Improvement
----------------------

Our robust 10-sample analysis (SAE-mediated, 5-digit prompts to match **QuantaMaths d5 expert**) shows a significant improvement in $P(correct)$.

TODO

Sequence Analysis (Teacher-Forced)
----------------------------------

By using **Teacher Forcing**, we evaluate the model's **confidence** in the correct digits at each position.

TODO

**Low R² scores (< 0.10)**
    Often means models are processing different problems or, as shown above, representational strategies are misaligned between absolute and relative position experts.

References
----------

.. [Quirke2024] Quirke, P. & Barez, F. (2024). *Understanding Addition in Transformers*. ICLR 2024.
.. [Chen2025] Chen et al. (2025). *Affine Stitching for Neural Network Interpretability*. arXiv:2506.06609.
See Also
--------

- :doc:`carry_discovery` - Main addition circuit discovery experiment
- :doc:`methodology` - Attribution methodology overview
- :doc:`configuration` - Global configuration reference
