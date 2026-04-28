"""Batched inference helpers: tokenization, logit extraction, and greedy generation."""

from dataclasses import dataclass

import torch
from huggingface_hub.utils import disable_progress_bars as disable_hf_progress_bars
from transformer_lens import HookedTransformer
from transformers.utils.logging import disable_progress_bar as disable_transformers_progress_bars

from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input


@dataclass
class TokenizationInfo:
    """Breakdown of how a single answer string tokenizes."""

    answer_str: str
    token_ids: list[int]
    token_strs: list[str]
    n_tokens: int
    is_single_token: bool


def silence_libraries():
    """Suppress progress bar output from HuggingFace Hub and Transformers."""
    disable_hf_progress_bars()
    disable_transformers_progress_bars()


def tokenize_and_pad(
    model: HookedTransformer,
    prompts: list[str],
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Tokenize a list of prompts and right-pad them to a common length.

    Args:
        model: HookedTransformer with an attached tokenizer
        prompts: Input strings
        device: Target device (defaults to model.cfg.device)

    Returns:
        padded_tokens: (n_prompts, max_len) token id tensor
        attention_mask: (n_prompts, max_len) binary mask (1 = real, 0 = pad)
        lengths: original (unpadded) sequence lengths
    """
    if device is None:
        device = model.cfg.device

    if hasattr(model, "tokenize_qwen_input"):
        tokens_list = [
            tokenize_qwen_input(p, model.tokenizer, model.cfg.device).squeeze(0) for p in prompts
        ]
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

        padded = torch.cat(
            [
                tokens.to(device),
                torch.full((pad_len,), pad_token_id, dtype=tokens.dtype, device=device),
            ]
        )

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
    """Extract logits at the last real token position for each prompt.

    Args:
        model: HookedTransformer model
        prompts: Input strings
        batch_size: Number of prompts per forward pass

    Returns:
        (num_prompts, vocab_size) logit tensor
    """
    all_logits = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        tokens, mask, lengths = tokenize_and_pad(model, batch)

        # HookedTransformer doesn't natively use attention_mask, but causal masking
        # means padding after the sequence end doesn't affect earlier positions.
        logits = model(tokens)  # (batch, seq_len, vocab_size)

        for j, length in enumerate(lengths):
            all_logits.append(logits[j, length - 1, :])

    return torch.stack(all_logits)


def batched_greedy_generate(
    model: HookedTransformer,
    prompts: list[str],
    max_tokens: int = 10,
    batch_size: int = 32,
) -> list[str]:
    """Run batched greedy decoding and return the completion strings.

    Args:
        model: HookedTransformer model
        prompts: Input strings
        max_tokens: Maximum tokens to generate per prompt
        batch_size: Number of prompts processed per call to model.generate

    Returns:
        Generated completions with the prompt stripped. For arithmetic outputs,
        only the leading digit characters are retained.
    """
    completions = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]

        tokens, _, _ = tokenize_and_pad(model, batch)

        generated_tokens = model.generate(
            tokens,
            max_new_tokens=max_tokens,
            do_sample=False,  # greedy
            verbose=False,
            prepend_bos=False,  # already handled in tokenize_and_pad
        )

        for j, prompt_str in enumerate(batch):
            full_text = model.tokenizer.decode(generated_tokens[j], skip_special_tokens=True)

            if full_text.startswith(prompt_str):
                completion = full_text[len(prompt_str) :].strip()
            else:
                completion = full_text.replace(prompt_str, "").strip()

            # Keep only leading digits (for arithmetic result extraction)
            final_completion = ""
            for char in completion:
                if char.isdigit():
                    final_completion += char
                elif final_completion:
                    break

            completions.append(final_completion)

    return completions
