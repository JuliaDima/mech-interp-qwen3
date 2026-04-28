import warnings

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def tokenize_qwen_input(
    prompt: str | torch.Tensor | list[int],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Convert a prompt to a 1-D token id tensor, prepending an attention-sink token.

    Qwen models use a special token (preferably PAD) at position 0 as an attention
    sink. If the sequence doesn't already start with a special token, one is prepended.
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

    # PAD is the preferred sink token for Qwen; fall back to BOS/EOS/any special
    sink_candidates = [
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
    ]
    sink_candidates += tokenizer.all_special_ids

    sink_token_id = next(filter(lambda x: x is not None, sink_candidates), None)

    if sink_token_id is None:
        warnings.warn(
            "No special token available for use as attention sink. Position 0 will be ignored.",
            stacklevel=2,
        )
    else:
        sink_token_id = int(sink_token_id)
        tokens = torch.cat([torch.tensor([sink_token_id], device=tokens.device), tokens])

    return tokens.to(device)
