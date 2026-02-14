"""Salient logit selection using cumulative probability threshold.

Selects the smallest set of top logits whose cumulative probability exceeds
a desired threshold, and returns demeaned unembedding vectors for attribution.
Implements uses the same logic as circuit_tracer, with some minor differences.
"""

import torch


@torch.no_grad()
def compute_salient_logits(
    logits: torch.Tensor,
    unembed_weight: torch.Tensor,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick the smallest logit set whose cumulative prob >= *desired_logit_prob*.

    Args:
        logits: ``(d_vocab,)`` vector (single position).
        unembed_weight: ``(d_vocab, d_model)`` unembedding matrix (model.lm_head.weight).
        max_n_logits: Hard cap on number of logits to consider.
        desired_logit_prob: Cumulative probability threshold.

    Returns:
        tuple of:
            * logit_indices - ``(k,)`` vocabulary ids.
            * logit_probs   - ``(k,)`` softmax probabilities.
            * demeaned_vecs - ``(k, d_model)`` unembedding rows, demeaned.
    """
    probs = torch.softmax(logits, dim=-1)
    top_p, top_idx = torch.topk(probs, max_n_logits)
    cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
    cutoff = min(cutoff, max_n_logits)
    top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

    # unembed_weight is (d_vocab, d_model) for Qwen (lm_head.weight)
    cols = unembed_weight[top_idx]  # (k, d_model)
    demean = unembed_weight.mean(dim=0, keepdim=True)  # (1, d_model)
    demeaned_vecs = cols - demean  # (k, d_model)

    return top_idx, top_p, demeaned_vecs
