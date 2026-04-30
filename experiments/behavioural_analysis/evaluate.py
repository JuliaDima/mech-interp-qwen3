"""Teacher-forcing based correctness and confidence measurement.

No generation is performed.  For each (prompt, ground_truth) pair we run a
single batched forward pass and, at every answer-token position, record:
  - whether the correct token is rank-1 (top logit)  → per_digit_correct
  - the softmax probability of the correct token      → per_digit_confidence

Both lists are indexed left-to-right over the digit characters of the answer
(most-significant digit first).  Overall correctness = all digits are rank-1.
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input

# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------


def _chat_format(tokenizer, prompt: str) -> str:
    """Wrap a raw prompt in the Qwen3 chat template with thinking disabled."""
    kwargs: dict = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Batched teacher-forcing
# ---------------------------------------------------------------------------


@torch.no_grad()
def batched_teacher_force(
    model: HookedTransformer,
    prompt_strs: list[str],
    answer_strs: list[str],
) -> list[tuple[list[bool], list[float]]]:
    """One padded forward pass; returns per-digit (correct, confidence) for each example.

    correct[i]    = True if the model's top-1 logit at that digit position matches
                    the correct token (rank-0 in descending logit order).
    confidence[i] = softmax probability of the correct token at that position.

    Both lists are aligned left-to-right over the digit characters of answer_str
    (most-significant first).  Multi-character tokens share the same value across
    all digits they cover.
    """
    device = model.cfg.device
    tokenizer = model.tokenizer

    prompt_tok_lists: list[torch.Tensor] = []
    answer_tok_lists: list[torch.Tensor] = []

    for ps, ans in zip(prompt_strs, answer_strs, strict=False):
        pt = tokenize_qwen_input(_chat_format(tokenizer, ps), tokenizer, device)
        at = (
            tokenizer(ans, return_tensors="pt", add_special_tokens=False)
            .input_ids.squeeze(0)
            .to(device)
        )
        prompt_tok_lists.append(pt)
        answer_tok_lists.append(at)

    full_seqs = [
        torch.cat([p, a]) for p, a in zip(prompt_tok_lists, answer_tok_lists, strict=False)
    ]
    max_len = max(s.shape[0] for s in full_seqs)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    padded = torch.full((len(full_seqs), max_len), pad_id, dtype=torch.long, device=device)
    for i, seq in enumerate(full_seqs):
        padded[i, : seq.shape[0]] = seq

    logits = model(padded)  # (batch, max_len, vocab)

    results: list[tuple[list[bool], list[float]]] = []
    for i, (p_toks, a_toks, ans) in enumerate(
        zip(prompt_tok_lists, answer_tok_lists, answer_strs, strict=False)
    ):
        p_len = p_toks.shape[0]
        answer_digits = ans.replace(" ", "")

        tok_correct: list[bool] = []
        tok_conf: list[float] = []
        for j, tok_id in enumerate(a_toks.tolist()):
            logit_vec = logits[i, p_len - 1 + j]
            rank = int((logit_vec > logit_vec[tok_id]).sum().item())
            prob = float(torch.softmax(logit_vec, dim=-1)[tok_id].item())
            tok_correct.append(rank == 0)
            tok_conf.append(prob)

        # Expand token-level results to digit character positions
        digit_correct: list[bool] = []
        digit_conf: list[float] = []
        for tok_id, c, p in zip(a_toks.tolist(), tok_correct, tok_conf, strict=False):
            n = max(sum(ch.isdigit() for ch in tokenizer.decode([tok_id])), 1)
            digit_correct.extend([c] * n)
            digit_conf.extend([p] * n)

        n_digits = len(answer_digits)
        pad = n_digits - len(digit_correct)
        if pad > 0:
            digit_correct += [False] * pad
            digit_conf += [0.0] * pad

        results.append((digit_correct[:n_digits], digit_conf[:n_digits]))

    return results
