Methodology
===========

This project implements a gradient-based attribution pipeline based on the **Attribution Graphs** framework, extended with Sparse Autoencoders (SAEs) and Reconstructive Patching.

1. Linearized Forward Pass
--------------------------

To ensure that gradients accurately reflect the flow of information through the residue stream, we use a "linearized" forward pass. This involves:

*   **Monkey-patching Attention**: We freeze the attention patterns (Queries and Keys) so that gradients only flow through the Values (OV circuit). This isolates the "content" flow from the "routing" logic.
*   **RMSNorm Linearization**: We treat the normalization scale as a constant during the backward pass to avoid gradient artifacts from the non-linear denominator.

2. Reconstructive Patching
--------------------------

One of the most powerful features of this pipeline is the ability to discover **Inter-Layer Circuits** (connections between features in different layers). This is enabled by the ``--use_patching`` flag.

*   **How it works**: The output of an MLP layer is intercepted and replaced with its SAE reconstruction.
*   **Feature-to-Feature Edges**: Because the reconstruction is now part of the computational graph, we can calculate the gradient of a feature in Layer 12 with respect to a feature in Layer 4.
*   **The Error Trade-off**: Patching introduces error accumulation. Since the SAE is not a perfect reconstruction, small errors at early layers can compound, leading to higher reconstruction errors later in the model. This is a necessary trade-off for cross-layer connectivity.

3. Salient Logit Selection
--------------------------

Instead of attributing from every possible output token, we select the **Salient Logits**—the top $N$ tokens that cover the majority of the probability mass (typically 95%). We use "demeaned" unembedding vectors to ensure the attribution is contrastive (showing why the model chose token A *instead* of others).

4. Pruning Logic
----------------

Attribution graphs can be massive (often $10^5+$ edges). We apply heavy pruning to reveal the core circuit:

*   **Node Pruning**: We sort nodes by their total "influence" (incoming + outgoing attribution) and keep the fewest number of nodes required to explain a specific fraction (e.g., 80%) of the total graph weight.
*   **Edge Pruning**: Similarly, we prune edges that contribute less than a certain threshold to the total attribution.

The results in ``pruned_graph.json`` represent the most significant pathways used by the model for the given prompt.
