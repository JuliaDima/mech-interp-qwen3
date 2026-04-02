Knowledge Editing Experiment
============================

*Adding an artificial "cortical microcircuit" that performs addition and plugs into the main network.*

This experiment transfers arithmetic knowledge from a small, purpose-trained addition model into Qwen3-4B via a **learned bottleneck** that bridges the two models' SAE feature spaces, without modifying any model weights.

Two injection modes are compared:

- **Replace** — the large model's MLP output at the injection site is replaced entirely by the bottleneck's output.
- **Add** — the bottleneck's output is added on top of the large model's existing MLP output.

.. code-block:: bash

   # Full pipeline
   python experiments/knowledge_editing/run.py --all \
       --small_model_path runs/stitching/rope/small_model.pt \
       --dataset_path data/addition_dataset.jsonl

   # Stages individually
   python experiments/knowledge_editing/run.py --setup
   python experiments/knowledge_editing/run.py --train --mode replace
   python experiments/knowledge_editing/run.py --train --mode add
   python experiments/knowledge_editing/run.py --compare


Formalisation
-------------

Models and Dimensions
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 55 15

   * - Symbol
     - Meaning
     - Value
   * - :math:`\mathcal{M}_s`
     - Small addition transformer (RoPE, 2 layers)
     - —
   * - :math:`\mathcal{M}_B`
     - Qwen3-4B (large model, frozen)
     - —
   * - :math:`\mathcal{A}_s`
     - SAE trained on :math:`\mathcal{M}_s`, last layer :math:`L_s = n_s - 1`
     - —
   * - :math:`\mathcal{A}_B`
     - Qwen3-4B transcoder at inject layer :math:`L_B = 18`
     - —
   * - :math:`d_s`
     - :math:`\mathcal{A}_s` feature dimension
     - 4096
   * - :math:`d_m`
     - :math:`\mathcal{M}_B` residual stream dimension
     - 2560
   * - :math:`d_{mid}`
     - Bottleneck dimension
     - 256
   * - :math:`W^B_{dec}`
     - :math:`\mathcal{A}_B` decoder, :math:`\in \mathbb{R}^{d_{tc} \times d_m}`
     - —


Feature Extraction (Frozen, No Gradient)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given a prompt with the ``=`` token at position :math:`p`, two feature vectors are extracted — both under ``torch.no_grad()``.

**Small model features** — the carry circuit in sparse form:

.. math::

   \mathbf{r}_s = \text{resid\_mid}_{L_s}(x_s)[p] \in \mathbb{R}^{d_s^{model}}

.. math::

   \mathbf{f}_s = \mathcal{A}_s^{\text{enc}}(\mathbf{r}_s) \in \mathbb{R}^{d_s}

where :math:`x_s` is the small-model tokenisation of ``"a+b=answer"`` and :math:`\mathcal{A}_s^{\text{enc}}` applies the encoder and ReLU activation (yielding sparse activations). The last layer (:math:`L_s`) is used because Quirke & Barez (ICLR 2024) show carry computation is maximally active there.

**Big model SAE decoded output** — the large model's expected MLP contribution at this layer:

.. math::

   \mathbf{r}_B = \text{resid\_mid}_{L_B}(x_B)[p] \in \mathbb{R}^{d_m}

.. math::

   \mathbf{f}_B = \mathcal{A}_B^{\text{enc}}(\mathbf{r}_B) \in \mathbb{R}^{d_{tc}}, \qquad
   \hat{\mathbf{y}}_B = \mathbf{f}_B \, W^B_{dec} \in \mathbb{R}^{d_m}

:math:`\hat{\mathbf{y}}_B` is the transcoder's reconstruction of the MLP output — what :math:`\mathcal{M}_B`'s SAE "expects" to happen at this layer for the given input.


Alignment Projection (Precomputed Once)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Collect :math:`\hat{\mathbf{y}}_B^{(i)}` over :math:`N` training samples. Centre the matrix:

.. math::

   X = \bigl[\hat{\mathbf{y}}_B^{(1)}, \ldots, \hat{\mathbf{y}}_B^{(N)}\bigr] - \bar{\mathbf{y}}_B \in \mathbb{R}^{N \times d_m}

Compute truncated SVD :math:`X = U \Sigma V^\top` and take the top :math:`d_{mid}` right singular vectors:

.. math::

   P = V^\top_{1:d_{mid}} \in \mathbb{R}^{d_{mid} \times d_m}

:math:`P` is **fixed after setup** — it captures the principal directions of variation in the large model's SAE outputs across addition problems, providing a :math:`d_{mid}`-dimensional subspace that the bottleneck must operate in.


Learnable Module
~~~~~~~~~~~~~~~~

The only trainable parameters are :math:`\theta` (a two-layer MLP) and :math:`\mathbf{w}_{out}` (a linear map):

.. math::

   \theta: \mathbb{R}^{d_s} \xrightarrow{\text{Linear}} \mathbb{R}^{d_s} \xrightarrow{\text{GELU}} \xrightarrow{\text{Linear}} \mathbb{R}^{d_{mid}}

.. math::

   \mathbf{w}_{out}: \mathbb{R}^{d_{mid}} \to \mathbb{R}^{d_m} \quad \text{(linear, no bias)}

The bottleneck representation and injection vector are:

.. math::

   \tilde{\mathbf{z}} = \theta(\mathbf{f}_s) \in \mathbb{R}^{d_{mid}}, \qquad
   \mathbf{v} = \mathbf{w}_{out}(\tilde{\mathbf{z}}) \in \mathbb{R}^{d_m}


Injection
~~~~~~~~~

At each forward pass during training and evaluation, :math:`\mathbf{v}` is patched into :math:`\mathcal{M}_B` at ``blocks.18.hook_mlp_out`` at position :math:`p`:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Hook modification
   * - **Replace**
     - :math:`\text{MLP-out}_{L_B}[p] \leftarrow \mathbf{v}`
   * - **Add**
     - :math:`\text{MLP-out}_{L_B}[p] \leftarrow \text{MLP-out}_{L_B}[p] + \mathbf{v}`

The patched residual propagates normally through layers :math:`L_B+1, \ldots, 35` to produce logits.


Training Objective
~~~~~~~~~~~~~~~~~~

.. math::

   \mathcal{L} = \mathcal{L}_{CE} + \lambda \, \mathcal{L}_{align}

**Cross-entropy loss** — the large model should predict the correct first answer token after injection:

.. math::

   \mathcal{L}_{CE} = -\log p_{\mathcal{M}_B^{\text{patched}}}(\text{ans}_1 \mid x_B)

**Alignment loss** — the bottleneck should be directionally consistent with the large model's SAE geometry:

.. math::

   \mathcal{L}_{align} = 1 - \cos\!\left(\tilde{\mathbf{z}},\; P\,\hat{\mathbf{y}}_B\right)
   = 1 - \frac{\tilde{\mathbf{z}} \cdot P\hat{\mathbf{y}}_B}{\|\tilde{\mathbf{z}}\|\;\|P\hat{\mathbf{y}}_B\|}

:math:`P\hat{\mathbf{y}}_B \in \mathbb{R}^{d_{mid}}` projects the large-model SAE output into the same space as :math:`\tilde{\mathbf{z}}`. Minimising :math:`\mathcal{L}_{align}` pulls :math:`\theta` toward directions that :math:`\mathcal{M}_B`'s transcoder expects for addition at layer :math:`L_B`.


Gradient Flow
~~~~~~~~~~~~~

All four components :math:`\mathcal{M}_s`, :math:`\mathcal{A}_s`, :math:`\mathcal{M}_B`, :math:`\mathcal{A}_B` are **frozen**. :math:`\mathbf{f}_s` and :math:`\hat{\mathbf{y}}_B` are precomputed without gradient. During training the gradient path is:

.. math::

   \frac{\partial \mathcal{L}_{CE}}{\partial \theta,\, \mathbf{w}_{out}}:\quad
   \mathcal{L}_{CE}
   \;\to\; \text{logits}_{35}
   \;\to\; \cdots
   \;\xrightarrow{\text{residual highway}}\;
   \mathbf{v}
   \;\to\; \mathbf{w}_{out}
   \;\to\; \tilde{\mathbf{z}}
   \;\to\; \theta

MLP outputs at layers :math:`L_B+1, \ldots, 35` are detached from the computation graph via the permanent skip hooks in ``AttributionModel``, so the gradient travels exclusively through the residual stream — exactly the same path used in attribution graph computation.

.. note::

   This means the module is trained to influence the large model's *residual stream* at and above layer 18, not its weight matrices. No model weights are modified at any point.


Relation to SAE Feature Space
------------------------------

The injection operates in **activation space** (``hook_mlp_out``), not directly in SAE feature space. The alignment loss ensures the injected vector :math:`\mathbf{v}` is geometrically consistent with the large model's transcoder geometry, but there is no guarantee that :math:`\mathbf{v}` decomposes cleanly into individual SAE features.

A fully SAE-level edit would instead patch :math:`\mathbf{f}_B` directly (before decoding), activating specific transcoder features by name. That would require first identifying *which* features in :math:`\mathcal{A}_B` correspond to carry — which is precisely what the stitching and carry-discovery experiments aim to establish.


References
----------

- Wang, Peng, Zexi Li, Ningyu Zhang, et al. “WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models.” arXiv:2405.14768. Preprint, arXiv, December 19, 2024. https://doi.org/10.48550/arXiv.2405.14768.
- “MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs.” Accessed March 24, 2026. https://arxiv.org/html/2506.07899.
