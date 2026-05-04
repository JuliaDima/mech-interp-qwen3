"""Shared helpers for the hierarchical module prototype experiment.

Only contains logic not already present in src/mechinterp_qwen3/.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


# ---------------------------------------------------------------------------
# Residual stream collection via model.run_with_hooks
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_residuals(
    model: HookedTransformer,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Run a frozen Qwen forward pass and collect the post-layer residual stream.

    Uses model.run_with_hooks() with TransformerLens 'hook_resid_post' hooks —
    consistent with the intervention infrastructure in interventions.py.

    Args:
        model: Frozen HookedTransformer (requires_grad=False for all params).
        tokens: (batch, seq) or (seq,) token ids.

    Returns:
        residuals: (n_layers, batch, seq, d_model) CPU tensor — detached from
                   Qwen's computation graph so no gradient flows into Qwen.
    """
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)

    cache: dict[int, torch.Tensor] = {}

    def _make_hook(layer: int):
        def _hook(resid: torch.Tensor, hook) -> torch.Tensor:
            cache[layer] = resid.detach().cpu()
            return resid

        return _hook

    hooks = [
        (f"blocks.{layer}.hook_resid_post", _make_hook(layer))
        for layer in range(model.cfg.n_layers)
    ]
    model.run_with_hooks(tokens, fwd_hooks=hooks)

    return torch.stack([cache[layer] for layer in range(model.cfg.n_layers)])


# ---------------------------------------------------------------------------
# Hard-regime dataset generation (reuses dataset_generation interfaces)
# ---------------------------------------------------------------------------


def generate_hard_regime_pairs(
    n_digits: int,
    n_samples: int,
    seed: int = 42,
    held_out: bool = False,
    held_out_fraction: float = 0.15,
) -> list[tuple[int, int]]:
    """Generate random (a, b) integer pairs where both operands have exactly n_digits digits.

    The held-out split uses the first `held_out_fraction` of the shuffled list;
    the training split uses the remainder.  Splits are disjoint and reproducible.

    Args:
        n_digits: Number of digits per operand (e.g. 4 for 1000–9999).
        n_samples: Total number of pairs to generate (before splitting).
        seed: RNG seed for reproducibility.
        held_out: If True return the held-out evaluation slice; else the train slice.
        held_out_fraction: Fraction of samples reserved for evaluation.

    Returns:
        List of (a, b) integer pairs.
    """
    rng = random.Random(seed)
    lo = 10 ** (n_digits - 1)
    hi = 10**n_digits - 1
    pairs = [(rng.randint(lo, hi), rng.randint(lo, hi)) for _ in range(n_samples)]
    split = max(1, int(len(pairs) * held_out_fraction))
    if held_out:
        return pairs[:split]
    return pairs[split:]


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------


def build_training_batch(
    pairs: list[tuple[int, int]],
    template: str,
    tokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Tokenize a batch of addition problems for teacher-forced training.

    Tokens are prompt + answer concatenated, right-padded to the longest
    sequence in the batch.  Uses add_special_tokens=False (consistent with
    the BOS-token guidance from the project memory).

    Args:
        pairs: List of (a, b) integer pairs.
        template: Format string, e.g. "calc: {a}+{b}= ".
        tokenizer: Model tokenizer.
        device: Target device.

    Returns:
        tokens: (batch, max_seq) long tensor — full prompt+answer for teacher forcing.
        prompt_lengths: Length (in tokens) of each prompt portion.
        answer_lengths: Length (in tokens) of each answer portion.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    prompt_ids_list: list[list[int]] = []
    answer_ids_list: list[list[int]] = []

    for a, b in pairs:
        prompt_str = template.format(a=a, b=b)
        answer_str = str(a + b)
        prompt_ids = tokenizer(prompt_str, return_tensors=None, add_special_tokens=False)[
            "input_ids"
        ]
        answer_ids = tokenizer(answer_str, return_tensors=None, add_special_tokens=False)[
            "input_ids"
        ]
        prompt_ids_list.append(prompt_ids)
        answer_ids_list.append(answer_ids)

    combined = [p + a for p, a in zip(prompt_ids_list, answer_ids_list, strict=False)]
    max_len = max(len(c) for c in combined)

    padded = torch.full((len(combined), max_len), pad_id, dtype=torch.long, device=device)
    for i, seq in enumerate(combined):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

    prompt_lengths = [len(p) for p in prompt_ids_list]
    answer_lengths = [len(a) for a in answer_ids_list]

    return padded, prompt_lengths, answer_lengths


# ---------------------------------------------------------------------------
# Teacher-forced loss on answer digit positions only
# ---------------------------------------------------------------------------


def compute_ce_on_answer_positions(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    prompt_lengths: list[int],
    answer_lengths: list[int],
) -> torch.Tensor:
    """Cross-entropy loss at answer digit positions only (teacher forcing).

    Position mapping:
        logits[i, p-1]   predicts  tokens[i, p]   (first answer token)
        logits[i, p]     predicts  tokens[i, p+1]  (second answer token)  etc.
    where p = prompt_lengths[i].

    Args:
        logits: (batch, seq, vocab) – from model or Stage1Head.
        tokens: (batch, seq) – full prompt+answer token ids.
        prompt_lengths: Prompt lengths per example.
        answer_lengths: Answer lengths per example.

    Returns:
        Scalar CE loss averaged over all answer digit positions in the batch.
    """
    batch_size, seq_len, vocab_size = logits.shape

    logit_slices: list[torch.Tensor] = []
    target_slices: list[torch.Tensor] = []

    for i in range(batch_size):
        p = prompt_lengths[i]
        a = answer_lengths[i]
        for j in range(a):
            logit_pos = p - 1 + j  # logit that predicts answer token j
            target_pos = p + j  # position of answer token j in 'tokens'
            if logit_pos >= seq_len or target_pos >= seq_len:
                break
            logit_slices.append(logits[i, logit_pos])  # (vocab,)
            target_slices.append(tokens[i, target_pos])  # scalar

    if not logit_slices:
        # Nothing to score — return zero with grad_fn if logits has one
        return (logits * 0).sum()

    logit_stack = torch.stack(logit_slices)  # (n_positions, vocab)
    target_stack = torch.stack(target_slices)  # (n_positions,)
    return F.cross_entropy(logit_stack, target_stack)


# ---------------------------------------------------------------------------
# Write hooks for augmented Qwen forward pass
# ---------------------------------------------------------------------------


def make_write_hooks(
    deltas: torch.Tensor,
    n_layers: int,
) -> list[tuple[str, object]]:
    """Build TransformerLens-compatible hook list that injects module deltas.

    Each hook adds delta_l to Qwen's residual stream at layer l.  The delta
    tensors retain their grad_fn so PyTorch propagates gradients from the loss
    back through the deltas into the module parameters.

    Args:
        deltas: (n_layers, batch, seq, d_model) – from CrossLayerWrite.forward.
        n_layers: Number of Qwen layers.

    Returns:
        List of (hook_name, hook_fn) tuples ready for model.run_with_hooks().
    """

    def _make_hook(delta_l: torch.Tensor):
        def _hook(resid: torch.Tensor, hook) -> torch.Tensor:
            return resid + delta_l.to(device=resid.device, dtype=resid.dtype)

        return _hook

    return [
        (f"blocks.{layer_idx}.hook_resid_post", _make_hook(deltas[layer_idx]))
        for layer_idx in range(n_layers)
    ]


# ---------------------------------------------------------------------------
# Greedy decode with module augmentation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Carry label utilities for Stage 1a / 1b
# ---------------------------------------------------------------------------


def compute_carry_labels(a: int, b: int, n_digits: int) -> list[int]:
    """Compute per-position carry-out labels for a + b.

    carry_outs[i] = 1 if the digit pair at position i (MSB=0, LSB=n_digits-1)
    generates a carry into position i-1 (to the left).

    Args:
        a, b: Operands — each must have exactly n_digits digits.
        n_digits: Number of digit positions.

    Returns:
        List of n_digits binary labels (0 or 1).
    """
    carry_outs = [0] * n_digits
    carry = 0
    for i in range(n_digits - 1, -1, -1):
        a_d = (a // 10 ** (n_digits - 1 - i)) % 10
        b_d = (b // 10 ** (n_digits - 1 - i)) % 10
        total = a_d + b_d + carry
        carry_outs[i] = 1 if total >= 10 else 0
        carry = 1 if total >= 10 else 0
    return carry_outs


def build_carry_batch(
    pairs: list[tuple[int, int]],
    n_digits: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build input and target tensors for Stage 1a (BiGRU isolation training).

    Args:
        pairs: List of (a, b) integer pairs.
        n_digits: Digit count per operand.
        device: Target device.

    Returns:
        pair_indices: (batch, n_digits) long tensor, values in [0, 99].
                      pair_indices[i, j] = a_digit_j * 10 + b_digit_j (MSB-first).
        carry_labels: (batch, n_digits) float tensor with 0/1 carry-out per position.
    """
    pair_indices_list: list[list[int]] = []
    carry_labels_list: list[list[int]] = []

    for a, b in pairs:
        indices = [
            ((a // 10 ** (n_digits - 1 - j)) % 10) * 10 + (b // 10 ** (n_digits - 1 - j)) % 10
            for j in range(n_digits)
        ]
        pair_indices_list.append(indices)
        carry_labels_list.append(compute_carry_labels(a, b, n_digits))

    pair_indices = torch.tensor(pair_indices_list, dtype=torch.long, device=device)
    carry_labels = torch.tensor(carry_labels_list, dtype=torch.float, device=device)
    return pair_indices, carry_labels


def build_prompt_batch(
    pairs: list[tuple[int, int]],
    templates: str | list[str],
    tokenizer,
    device: torch.device,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Tokenize prompts only (no answers) for Stage 1b.

    When `templates` is a list, a template is sampled uniformly at random for
    each example in the batch.  This forces DigitSlotAttention to learn
    template-independent digit features rather than memorising token positions.

    Args:
        pairs: List of (a, b) integer pairs.
        templates: A single format string, or a list of format strings.
                   Each string must accept {a} and {b} keyword arguments.
        tokenizer: Model tokenizer.
        device: Target device.
        rng: Optional random.Random for reproducible template sampling.
             If None, uses the module-level random.

    Returns:
        tokens: (batch, max_seq) long tensor — prompt tokens only.
        prompt_lengths: Actual prompt length for each example (before padding).
    """
    import random as _random

    _rng = rng or _random

    if isinstance(templates, str):
        templates = [templates]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    ids_list: list[list[int]] = []
    for a, b in pairs:
        tmpl = _rng.choice(templates)
        prompt_str = tmpl.format(a=a, b=b)
        ids = tokenizer(prompt_str, return_tensors=None, add_special_tokens=False)["input_ids"]
        ids_list.append(ids)

    max_len = max(len(ids) for ids in ids_list)
    padded = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long, device=device)
    for i, ids in enumerate(ids_list):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

    return padded, [len(ids) for ids in ids_list]


# ---------------------------------------------------------------------------
# Greedy decode
# ---------------------------------------------------------------------------


@torch.no_grad()
def greedy_decode_with_module(
    qwen: HookedTransformer,
    module,
    prompt_tokens: list[int],
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> list[int]:
    """Autoregressively generate tokens using the module-augmented Qwen.

    At each step:
      1. Collect residuals from frozen Qwen (no_grad).
      2. Run module (read+primitive+gate+write) to get deltas.
      3. Run second Qwen pass with deltas injected to get next-token logits.

    Args:
        qwen: Frozen HookedTransformer.
        module: Trained PrototypeModule (eval mode, no_grad).
        prompt_tokens: Initial token ids.
        max_new_tokens: Maximum number of tokens to generate.
        device: Compute device.
        dtype: Dtype for module computations.

    Returns:
        List of generated token ids (not including the prompt).
    """
    generated = list(prompt_tokens)
    n_layers = qwen.cfg.n_layers

    for _ in range(max_new_tokens):
        input_t = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)

        # Step 1: collect residuals (already no_grad from decorator)
        residuals = collect_residuals(qwen, input_t)  # (n_layers, 1, seq, d_model)

        # Step 2: module forward (no_grad, module is in eval mode)
        residuals_dev = residuals.to(device=device, dtype=dtype)
        deltas, _f, _g = module(residuals_dev)  # (n_layers, 1, seq, d_model)

        # Step 3: augmented Qwen pass to get logits
        write_hooks = make_write_hooks(deltas, n_layers)
        logits = qwen.run_with_hooks(input_t, fwd_hooks=write_hooks)  # (1, seq, vocab)

        next_token = int(logits[0, -1].argmax().item())
        generated.append(next_token)

        if next_token == qwen.tokenizer.eos_token_id:
            break

    return generated[len(prompt_tokens) :]
