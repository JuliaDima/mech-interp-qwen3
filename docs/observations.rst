Empirical Findings for Experiment Stability
=========================================

This section documents observations made regarding model behavior and prompt stability during the development of the ``mechinterp-qwen3`` pipeline. These findings inform the choice of default configurations for mechanistic interpretability experiments.

Impact of Initial Sink Tokens
-----------------------------

Qwen3-4B-Instruct shows noticeable sensitivity to the very first token in a sequence, which often serves as an "attention sink."

*   **Observation**: Without an explicit sink token (like the PAD token), first-token accuracy on addition tasks (e.g., ``calc: 36+59=``) was significantly lower (~1.6%).
*   **Sample Diagnostic**: In a test case (prompt: ``calc: 36+59=``), adding the PAD token (ID ``151643``, ``<|endoftext|>``) as the initial token increased the target probability of the first digit from **~0.4% to ~56%**.
*   **Implementation**: To ensure maximum stability during circuit discovery, the ``tokenize_qwen_input`` method always prepends this PAD token to inputs that do not already start with a special token.

Prompt Formatting and Trailing Spaces
-------------------------------------

Benchmarking multiple templates revealed that arithmetic completion is highly sensitive to exact string formatting, particularly whitespace.

*   **Trailing Spaces**: Recent testing indicates that adding a trailing space to certain templates—for example, ``calc: {a}+{b}= ``—can increase first-token accuracy toward **100%**. This suggests the model's training distribution might favor seeing a space before digits in certain contexts.
*   **Template Preference**: Despite the potential for 100% accuracy with more complex prompts or trailing spaces, **Template T0** (``calc: {a}+{b}=``) is retained as the project default. It provides the simplest baseline (approx. 77% greedy accuracy, with over 99.9% arithmetic accuracy once formatting preambles are accounted for) and minimizes unnecessary tokens for circuit analysis.

.. list-table:: Observed Prompt Variations
   :widths: 10 40 50
   :header-rows: 1

   * - ID
     - Template String
     - Observations
   * - T0
     - ``calc: {a}+{b}=``
     - **Default**. Simplest format; high digit accuracy but prone to "chatty" preambles (``?\n\n``).
   * - T1
     - ``calc: {a} + {b} =``
     - Extra internal spaces; led to 0% accuracy in early tests due to formatting drift.
   * - T5
     - ``<|im_start|>user\n...<|im_end|>\n...``
     - Full ChatML format. Higher accuracy but longer and more computationally expensive for circuit analysis.

Handling Tokenization Consistency
---------------------------------

Discrepancies between tokenization logic in different libraries (e.g., how BOS tokens are prepended) can lead to mismatched activations and attribution errors.

*   **Consistency**: A custom ``tokenize_qwen_input`` method is used throughout the project to ensure that the identical token sequence is used in the accuracy sweep, dataset generation, and the final attribution/intervention phases.
