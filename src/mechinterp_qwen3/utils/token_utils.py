import warnings

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def tokenize_qwen_input(
    prompt: str | torch.Tensor | list[int],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Convert prompt to 1-D tensor of token ids with proper special token handling (sink token).

    Qwen models often benefit from an initial sink token (like PAD/endoftext) for
    numerical stability in arithmetic tasks. This function prepends a suitable
    special token if the input doesn't already start with one.
    """

    if isinstance(prompt, str):
        tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(
            0
        )
    elif isinstance(prompt, torch.Tensor):
        tokens = prompt.squeeze()
    elif isinstance(prompt, list):
        tokens = torch.tensor(prompt, dtype=torch.long).squeeze()
    else:
        raise TypeError(f"Unsupported prompt type: {type(prompt)}")

    if tokens.ndim > 1:
        raise ValueError(f"Tensor must be 1-D, got shape {tokens.shape}")

    if tokens[0] in tokenizer.all_special_ids:
        return tokens.to(device)

    candidate_bos_token_ids = [
        tokenizer.pad_token_id,  # Prefer PAD as it's the standard attention sink for Qwen
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
    ]
    candidate_bos_token_ids += tokenizer.all_special_ids

    # Find the first not-None candidate
    dummy_bos_token_id = next(filter(lambda x: x is not None, candidate_bos_token_ids), None)

    if dummy_bos_token_id is None:
        warnings.warn(
            "No suitable special token found for BOS token replacement. The first token will be ignored.",
            stacklevel=2,
        )
    else:
        dummy_bos_token_id = int(dummy_bos_token_id)
        tokens = torch.cat([torch.tensor([dummy_bos_token_id], device=tokens.device), tokens])

    return tokens.to(device)
