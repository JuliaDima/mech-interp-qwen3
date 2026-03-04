Methodology
===========

This project implements a gradient-based attribution pipeline based on the **Attribution Graphs** framework, extended with Sparse Autoencoders (SAEs) and Reconstructive Patching.

1. Gradient-Based Attribution
-----------------------------

To ensure that gradients accurately reflect the flow of information through the residue stream, we use integrated gradients or direct attribution from output logits back to input features.

*   **Attribution Target**: We typically use the logit of the correct answer token (or the difference between the correct and incorrect logits) as the attribution target.
*   **Feature Importance**: The importance of a transcoder feature is defined as the magnitude of the gradient of the target with respect to the feature's activation.

2. Reconstructive Patching
--------------------------

Reconstructive patching is used to discover how features in different layers interact.

*   **How it works**: The output of an MLP layer is intercepted and replaced with its SAE reconstruction. This allows gradients to flow through the transcoder bottleneck.
*   **Feature-to-Feature Edges**: Because the reconstruction is now part of the computational graph, we can calculate the attribution of a downstream feature with respect to an upstream feature.
*   **Implementation**: This is handled automatically by the ``AttributionModel`` wrapper, which manages the hooks for transcoder activation and reconstruction injection.

3. Salient Logit Selection
--------------------------

Instead of attributing from every possible output token, we select the **Salient Logits**—the top $N$ tokens that cover the majority of the probability mass. This focuses the graph construction on the model's actual predictions.

4. Pruning and Visualization
----------------------------

Attribution graphs can be massive. We apply heavy pruning to reveal the core circuit:

*   **Node Pruning**: We filter nodes by their attribution score, keeping only those that exceed a significance threshold (e.g., 5% of total attribution).
*   **Edge Pruning**: We remove edges that represent insignificant contributions to the total information flow.

The resulting ``nodes.json`` and ``edges.json`` represent the most significant pathways used by the model for the given task.
