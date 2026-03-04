Research Observations
=====================

This section documents critical empirical findings discovered during the development and benchmarking of the ``mechinterp-qwen3`` pipeline. These observations are central to understanding model behavior and ensuring high fidelity in mechanistic interpretability experiments.

Attention Sink Sensitivity in Qwen3-4B
--------------------------------------

During initial benchmarking of the accuracy sweep, it was discovered that **Qwen3-4B-Instruct** is extremely sensitive to the presence of an initial "sink token."

*   **Finding**: Without a proper sink token, first-token prediction accuracy on the addition task (``calc: 36+59=``) dropped to near zero (~1.6%).
*   **Mechanism**: Causal LLMs often use the first token in the sequence as a "global sink" to offload attention when no other tokens are relevant. If this first token is a functional token like ``<|im_start|>`` or ``<|im_end|>``, it may not serve as an effective sink.
*   **Solution**: Standardizing on the **PAD token** (ID ``151643``, which decodes to ``<|endoftext|>`` for Qwen) as the initial token increased target probability from ~0.4% to ~56% for individual prompts, enabling successful circuit discovery.

Format & Instruction Sensitivity
--------------------------------

Research using the ``compare_templates.py`` utility revealed an extreme sensitivity to minor formatting differences, such as whitespace and newlines. The following templates were benchmarked:

.. list-table:: Benchmarked Prompt Templates
   :widths: 10 40 50
   :header-rows: 1

   * - ID
     - Template String
     - Significance
   * - T0
     - ``calc: {a}+{b}=``
     - Baseline (77% accuracy). No spaces.
   * - T1
     - ``calc: {a} + {b} =``
     - Extra spaces (0% accuracy). Causes formatting drift.
   * - T2
     - ``What is {a}+{b}? Answer:``
     - Question format. Tests instruction following.
   * - T3
     - ``calc: {a}+{b}=\n``
     - Baseline with trailing newline.
   * - T4
     - ``calc: {a}+{b}=\nAnswer:``
     - Baseline with explicit answer prefix.
   * - T5
     - ``<|im_start|>user\nCalculate {a}+{b}<|im_end|>\n<|im_start|>assistant\n``
     - Full ChatML format. High fidelity with RLHF training.
   * - T6
     - ``Answer the following addition problem: {a} + {b} =``
     - Descriptive prompt.

*   **Whitespace Impact**: Switching from ``calc: 36+59=`` (Template T0) to ``calc: 36 + 59 =`` (Template T1) caused accuracy to drop from **~77% to 0%** in greedy decoding.
*   **The "Chatty" Failure Mode**: The dominant failure mode for Qwen3-4B on addition prompts is the emission of a preamble or header (e.g., ``? \n\n``) instead of the numerical answer. This suggests an instruction-following "refusal" or "formatting drift" where the model reverts to its chat-training instead of performing pure completion.
*   **Research Implication**: When performing attribution, it is vital to differentiate between **arithmetic errors** (wrong digits) and **formatting errors** (preambles). The latter can often be resolved by adjusting the prompt template until accuracy exceeds 95%, rather than concluding the model "cannot add."

Tokenization Discrepancies
--------------------------

Variations in tokenization logic across different libraries (TransformerLens vs. Transformers) can lead to silent discrepancies in statistics.

*   **Observation**: Different ways of prepending "BOS" tokens can lead to different effective sequences. For example, ``model.to_tokens(prepend_bos=True)`` might prepend a different token than the one the model expects as a sink.
