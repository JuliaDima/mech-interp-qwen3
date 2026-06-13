Methodology
===========

This project implements a gradient-based attribution pipeline based on the **Attribution Graphs** framework, extended with Sparse Autoencoders (SAEs) and Reconstructive Patching.

.. important::
   **No Teacher Forcing in Circuit Discovery**

   This methodology uses **gradient-based attribution** and **causal interventions** to discover
   circuits. Neither technique requires teacher forcing:

   - Forward passes use only the prompt (e.g., ``"calc: 36+59="``)
   - Attribution is computed from the logit of the correct token
   - Interventions test causality via activation patching

   Teacher forcing (available in ``dataset_generation``) is a separate tool for behavioral
   analysis and is not part of the core circuit discovery methodology.

1. Gradient-Based Attribution
-----------------------------

To ensure that gradients accurately reflect the flow of information through the residue stream, we use integrated gradients or direct attribution from output logits back to input features.

*   **Attribution Target**: We typically use the logit of the correct answer token (or the difference between the correct and incorrect logits) as the attribution target. The model does not generate the answer—we simply measure gradients flowing to the correct token's logit.
*   **Feature Importance**: The importance of a transcoder feature is defined as the magnitude of the gradient of the target with respect to the feature's activation.

2. Reconstructive Patching
--------------------------

Reconstructive patching is used to discover how features in different layers interact.

*   **How it works**: The output of an MLP layer is intercepted and replaced with its SAE reconstruction. This allows gradients to flow through the transcoder bottleneck.
*   **Feature-to-Feature Edges**: Because the reconstruction is now part of the computational graph, we can calculate the attribution of a downstream feature with respect to an upstream feature.
*   **Implementation**: This is handled automatically by the ``AttributionModel`` wrapper, which manages the hooks for transcoder activation and reconstruction injection.

.. note::
   Reconstructive patching operates on **activations**, not on generated tokens. It modifies
   the internal representations during a single forward pass, distinct from teacher forcing
   which would involve providing ground-truth tokens as input.

3. Salient Logit Selection
--------------------------

Instead of attributing from every possible output token, we select the **Salient Logits**—the top $N$ tokens that cover the majority of the probability mass. This focuses the graph construction on the model's actual predictions.

4. Pruning and Visualization
----------------------------

Attribution graphs can be massive. We apply heavy pruning to reveal the core circuit:

*   **Node Pruning**: We filter nodes by their attribution score, keeping only those that exceed a significance threshold (e.g., 5% of total attribution).
*   **Edge Pruning**: We remove edges that represent insignificant contributions to the total information flow.

The resulting ``nodes.json`` and ``edges.json`` represent the most significant pathways used by the model for the given task.

5. Accuracy Verification (Best Practice)
-----------------------------------------

Before attempting circuit discovery, verify that the model actually solves the task on your chosen prompt format.

**Why This Matters**
~~~~~~~~~~~~~~~~~~~~

If the model doesn't perform the computation correctly, there's no "intended circuit" to discover—you'd
be analyzing error modes instead. Following the best practice:

   *"Choose a prompt distribution where the model genuinely solves the task, and then condition your
   analysis on correct cases."*

**Accuracy Sweep Workflow**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For addition tasks (and similar computational tasks), run an accuracy sweep:

1. **Check tokenization**: Verify how answers split into tokens

   - Example: Does "95" tokenize as ["9","5"] or ["95"]?
   - This defines what "first token" means for attribution

2. **Run greedy decoding**: Test model on all prompts with argmax at output position

   - Measure: Does the model's predicted token match ground truth?
   - Report accuracy, margins, and failure cases

3. **Filter verified prompts**: Select only cases where model is correct

   - Condition circuit analysis on successful computations
   - Optionally filter by confidence (P(correct) > threshold)

**Implementation**
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # For addition experiments
   python experiments/addition/accuracy_sweep.py --all --quick

**Decision Criteria**:

- **Accuracy > 80%**: Proceed with circuit discovery
- **Accuracy < 80%**: Change prompt format (spacing, few-shot, delimiters) and re-run sweep
