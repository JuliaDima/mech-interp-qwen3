"""Primitive registry for the hierarchical module.

Each Primitive defines:
  - make_embedding / make_head  — torch modules wrapping the shared GRU
  - generate_data               — synthetic problem instances
  - build_batch                 — convert problem instances to (input_indices, labels)
  - loss / metric               — task-specific objectives

The shared GRU (CarryPrimitiveGRU) stays identical for all primitives.
Adding a new primitive means subclassing Primitive and registering it in PRIMITIVES.

Input contract for the training loop:
    input_indices : (batch, seq_len)        long — fed to embedding
    labels        : (batch, seq_len, ...)   float or long — fed to loss/metric
    logits        : (batch, seq_len, n_out) float — from head(gru(embedding(input_indices)))
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.hierarchical_module_prototype.model import CarryHead, PairEmbedding
from experiments.hierarchical_module_prototype.utils import (
    build_carry_batch,
    generate_hard_regime_pairs,
)


class SymmetricComparisonHead(nn.Module):
    """Comparison head for BiGRU palindrome detection.

    Splits the BiGRU output into its forward and backward halves, then
    computes explicit comparison features (difference, Hadamard product)
    before classifying. This gives the MLP direct access to symmetry
    signals rather than having to discover them from a flat concatenation.
    """

    def __init__(self, d_small: int) -> None:
        super().__init__()
        half = d_small // 2
        self.mlp = nn.Sequential(
            nn.Linear(4 * half, d_small),
            nn.ReLU(),
            nn.Linear(d_small, 1),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        half = f.shape[-1] // 2
        h_fwd = f[..., :half]
        h_bck = f[..., half:]
        features = torch.cat([h_fwd, h_bck, h_fwd - h_bck, h_fwd * h_bck], dim=-1)
        return self.mlp(features)  # (batch, seq, 1)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Primitive(ABC):
    """Abstract base class for a sequential primitive."""

    name: str

    @abstractmethod
    def make_embedding(self, d_small: int) -> nn.Module:
        """Embedding: (batch, seq_len) long → (batch, seq_len, d_small)."""

    @abstractmethod
    def make_head(self, d_small: int) -> nn.Module:
        """Head: (batch, seq_len, d_small) → (batch, seq_len, n_out)."""

    @abstractmethod
    def generate_data(
        self,
        n_digits: int,
        n_samples: int,
        seed: int,
        held_out: bool,
        held_out_fraction: float,
    ) -> list[Any]:
        """Generate a list of problem instances."""

    @abstractmethod
    def build_batch(
        self,
        problems: list[Any],
        n_digits: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a list of problems to tensors.

        Returns:
            input_indices : (batch, seq_len) long
            labels        : (batch, seq_len) or (batch, seq_len, n_out)
        """

    @abstractmethod
    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Scalar training loss."""

    @abstractmethod
    def metric(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        """Evaluation metrics dict."""


# ---------------------------------------------------------------------------
# Carry primitive (addition carry propagation)
# ---------------------------------------------------------------------------


class CarryPrimitive(Primitive):
    """Predict carry-out at each digit position of a + b.

    Input sequence : n_digits pair-embeddings, one per digit position.
    Labels         : (batch, n_digits) binary float — 1 iff digit i carries.
    Loss           : binary cross-entropy.
    """

    name = "carry"

    def make_embedding(self, d_small: int) -> nn.Module:
        return PairEmbedding(d_small)

    def make_head(self, d_small: int) -> nn.Module:
        return CarryHead(d_small, n_out=1)

    def generate_data(self, n_digits, n_samples, seed, held_out, held_out_fraction):
        return generate_hard_regime_pairs(
            n_digits,
            n_samples,
            seed=seed,
            held_out=held_out,
            held_out_fraction=held_out_fraction,
        )

    def build_batch(self, problems, n_digits, device):
        return build_carry_batch(problems, n_digits, device)
        # input_indices : (B, n_digits) — pair indices 0..99
        # labels        : (B, n_digits) — float 0/1

    def loss(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels)

    def metric(self, logits, labels):
        preds = (logits.squeeze(-1) > 0).float()
        acc = (preds == labels).float().mean().item()
        return {"carry_acc": acc}


# ---------------------------------------------------------------------------
# Multiplication primitive
# ---------------------------------------------------------------------------


def _digits_of(n: int, length: int) -> list[int]:
    """MSB-first digit list of n, zero-padded to `length`."""
    s = str(n).zfill(length)
    return [int(c) for c in s]


class MultiplicationPrimitive(Primitive):
    """Predict each digit of a * b.

    Input sequence : digits of a (MSB first) followed by digits of b — length 2*n_digits.
                     Each digit is an integer 0-9 (vocabulary size 10).
    Labels         : digits of the product, MSB first, zero-padded to 2*n_digits.
    Loss           : cross-entropy over 10 digit classes per position.

    The GRU processes the 2*n_digits input steps and must learn:
      - steps 0..n-1   (a's digits): build a representation of a
      - steps n..2n-1  (b's digits): using a's state + each b digit, produce product digits

    This tests whether the GRU generalizes to operand magnitudes unseen during training.
    """

    name = "multiplication"

    def make_embedding(self, d_small: int) -> nn.Module:
        emb = nn.Embedding(10, d_small)
        nn.init.normal_(emb.weight, std=0.02)
        return emb

    def make_head(self, d_small: int) -> nn.Module:
        head = nn.Linear(d_small, 10, bias=True)
        nn.init.normal_(head.weight, std=0.02)
        nn.init.zeros_(head.bias)
        return head

    def generate_data(self, n_digits, n_samples, seed, held_out, held_out_fraction):
        rng = random.Random(seed)
        lo = 10 ** (n_digits - 1)
        hi = 10**n_digits - 1
        pairs = [(rng.randint(lo, hi), rng.randint(lo, hi)) for _ in range(n_samples)]
        split = max(1, int(len(pairs) * held_out_fraction))
        return pairs[:split] if held_out else pairs[split:]

    def build_batch(self, problems, n_digits, device):
        max_prod_digits = 2 * n_digits
        inputs, labels = [], []
        for a, b in problems:
            inputs.append(_digits_of(a, n_digits) + _digits_of(b, n_digits))
            labels.append(_digits_of(a * b, max_prod_digits))
        return (
            torch.tensor(inputs, dtype=torch.long, device=device),  # (B, 2*n_digits)
            torch.tensor(labels, dtype=torch.long, device=device),  # (B, 2*n_digits)
        )

    def loss(self, logits, labels):
        B, S, C = logits.shape
        return F.cross_entropy(logits.reshape(B * S, C), labels.reshape(B * S))

    def metric(self, logits, labels):
        preds = logits.argmax(-1)  # (B, S)
        token_acc = (preds == labels).float().mean().item()
        exact = (preds == labels).all(dim=-1).float().mean().item()
        return {"token_acc": token_acc, "exact_match": exact}


# ---------------------------------------------------------------------------
# Palindrome detection primitive
# ---------------------------------------------------------------------------


class PalindromeDetectionPrimitive(Primitive):
    """Detect per-position palindrome match in a digit sequence.

    Input sequence : n_digits digits (0-9) — the sequence to inspect.
    Label at pos i : 1.0 if sequence[i] == sequence[n-1-i], else 0.0.
    Loss           : binary cross-entropy per position.

    The BiGRU is ideally suited: at position i the forward state encodes
    sequence[0..i] and the backward state encodes sequence[i..n-1].
    The character at the mirror position n-1-i was processed exactly i steps
    into each direction, giving the BiGRU a natural symmetric structure to
    exploit.  An MLP over a flat input could not generalize to different
    sequence lengths.
    """

    name = "palindrome"

    def make_embedding(self, d_small: int) -> nn.Module:
        emb = nn.Embedding(10, d_small)
        nn.init.normal_(emb.weight, std=0.02)
        return emb

    def make_head(self, d_small: int) -> nn.Module:
        return SymmetricComparisonHead(d_small)

    def generate_data(self, n_digits, n_samples, seed, held_out, held_out_fraction):
        rng = random.Random(seed)
        seqs = []
        for _ in range(n_samples):
            seq = [rng.randint(0, 9) for _ in range(n_digits)]
            # Each mirror pair independently has 50% chance of matching —
            # balances positive/negative labels so the trivial "always no-match"
            # baseline is 50% instead of 90%.
            for i in range(n_digits // 2):
                if rng.random() < 0.5:
                    seq[n_digits - 1 - i] = seq[i]
                else:
                    # Force a non-match by picking a different digit
                    others = [d for d in range(10) if d != seq[i]]
                    seq[n_digits - 1 - i] = rng.choice(others)
            seqs.append(tuple(seq))
        split = max(1, int(n_samples * held_out_fraction))
        return seqs[:split] if held_out else seqs[split:]

    def build_batch(self, problems, n_digits, device):
        inputs, labels = [], []
        for seq in problems:
            n = len(seq)
            inputs.append(list(seq))
            labels.append([float(seq[i] == seq[n - 1 - i]) for i in range(n)])
        return (
            torch.tensor(inputs, dtype=torch.long, device=device),
            torch.tensor(labels, dtype=torch.float, device=device),
        )

    def loss(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels)

    def metric(self, logits, labels):
        preds = (logits.squeeze(-1) > 0).float()
        pos_acc = (preds == labels).float().mean().item()
        # Recall on matching positions (the hard class)
        match_mask = labels > 0.5
        recall = (
            (preds[match_mask] > 0.5).float().mean().item() if match_mask.any() else float("nan")
        )
        full_acc = (preds == labels).all(dim=-1).float().mean().item()
        return {"position_acc": pos_acc, "match_recall": recall, "full_palindrome_acc": full_acc}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ALL: list[Primitive] = [
    CarryPrimitive(),
    MultiplicationPrimitive(),
    PalindromeDetectionPrimitive(),
]

PRIMITIVES: dict[str, Primitive] = {p.name: p for p in _ALL}
