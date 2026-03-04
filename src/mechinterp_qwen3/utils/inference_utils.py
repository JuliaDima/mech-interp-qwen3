"""Utilities for batched inference, tokenization, and generation."""

from dataclasses import dataclass

import torch
from huggingface_hub.utils import disable_progress_bars as disable_hf_progress_bars
from transformer_lens import HookedTransformer
from transformers.utils.logging import disable_progress_bar as disable_transformers_progress_bars


@dataclass
class TokenizationInfo:
    """Information about how an answer tokenizes."""

    answer_str: str
    token_ids: list[int]
    token_strs: list[str]
    n_tokens: int
    is_single_token: bool


def silence_libraries():
    """Disable noisy progress bars from Hugging Face and Transformers."""
    disable_hf_progress_bars()
    disable_transformers_progress_bars()


def tokenize_and_pad(
    model: HookedTransformer,
    prompts: list[str],
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Tokenize a batch of prompts and pad them to the same length.

    Args:
        model: HookedTransformer model
        prompts: List of prompt strings
        device: Device to place tensors on (defaults to model.cfg.device)

    Returns:
        Tuple of (padded_tokens, attention_mask, original_lengths)
    """
    if device is None:
        device = model.cfg.device

    if hasattr(model, "tokenize_qwen_input"):
        tokens_list = [model.tokenize_qwen_input(p).squeeze(0) for p in prompts]
    else:
        tokens_list = [model.to_tokens(p, prepend_bos=True).squeeze(0) for p in prompts]
    lengths = [len(t) for t in tokens_list]
    max_len = max(lengths)

    pad_token_id = model.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.tokenizer.eos_token_id

    padded_tokens = []
    attention_masks = []

    for tokens in tokens_list:
        pad_len = max_len - len(tokens)

        # Create padded sequence
        padded = torch.cat(
            [
                tokens.to(device),
                torch.full((pad_len,), pad_token_id, dtype=tokens.dtype, device=device),
            ]
        )

        # Create attention mask (1 for real tokens, 0 for padding)
        mask = torch.cat(
            [
                torch.ones(len(tokens), device=device, dtype=torch.long),
                torch.zeros(pad_len, device=device, dtype=torch.long),
            ]
        )

        padded_tokens.append(padded)
        attention_masks.append(mask)

    return torch.stack(padded_tokens), torch.stack(attention_masks), lengths


def batched_get_last_logits(
    model: HookedTransformer,
    prompts: list[str],
    batch_size: int = 32,
) -> torch.Tensor:
    """Get logits at the last non-padding position for a batch of prompts.

    Args:
        model: HookedTransformer model
        prompts: List of prompt strings
        batch_size: Batch size for inference

    Returns:
        Tensor of logits (num_prompts, vocab_size)
    """
    all_logits = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        tokens, mask, lengths = tokenize_and_pad(model, batch)

        # Forward pass
        # Transformer-Lens handles the attention mask internally if passed via stop_at_layer or hooks,
        # but for a standard forward pass we usually just pass the tokens.
        # However, to be safe with padding, we should be careful.
        # HookedTransformer's forward doesn't take an attention_mask directly in a way that respects padding
        # in the same way HF does, unless explicitly handled.
        # For simple logit extraction at 'lengths', it usually doesn't matter for causal models.

        logits = model(tokens)  # (batch, seq_len, vocab_size)

        # Extract last non-padding logit for each item in batch
        for j, length in enumerate(lengths):
            all_logits.append(logits[j, length - 1, :])

    return torch.stack(all_logits)


def batched_greedy_generate(
    model: HookedTransformer,
    prompts: list[str],
    max_tokens: int = 10,
    batch_size: int = 32,
) -> list[str]:
    """Perform batched greedy generation using model.generate.

    Args:
        model: HookedTransformer model
        prompts: List of prompt strings
        max_tokens: Maximum tokens to generate
        batch_size: Batch size for generation

    Returns:
        List of generated completion strings (excluding prompt)
    """
    completions = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]

        # Standard generation using Transformer-Lens wrap of model.generate
        # Note: HookedTransformer.generate supports batching if tokens are passed.
        # We'll use the underlying model.generate or HookedTransformer.generate

        # TransformerLens generate usually takes a single prompt or tokens.
        # For batching, we need to pass tokens.
        tokens, _, _ = tokenize_and_pad(model, batch)

        # Generate
        generated_tokens = model.generate(
            tokens,
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy
            verbose=False,
            prepend_bos=False,  # Already handled in tokenize_and_pad
        )

        # Decode and extract completions
        for j, prompt_str in enumerate(batch):
            full_text = model.tokenizer.decode(generated_tokens[j], skip_special_tokens=True)

            # Extract completion (strip prompt)
            if full_text.startswith(prompt_str):
                completion = full_text[len(prompt_str) :].strip()
            else:
                # Fallback
                completion = full_text.replace(prompt_str, "").strip()

            # Clean up: stop at first non-digit if it's an arithmetic result
            final_completion = ""
            for char in completion:
                if char.isdigit():
                    final_completion += char
                elif final_completion:
                    break

            completions.append(final_completion)

    return completions
