"""Soft learnable FSM for token-level primitive detection.

SoftFSM
-------
State is a soft distribution over n_states.  Transitions are learned per
predicate type.  At each step:

    s_{t+1} = softmax( s_t @ T[p_t] )          (row-vector convention)

where T[p] ∈ R^{S×S} holds the learned transition logits for predicate p.
Initialised with a strong identity bias so the FSM starts as "stay put"
and learns to move only on relevant predicates.

Activation readout:

    A_t = sigmoid( w · s_t + b )

PrimitiveRouter
---------------
K parallel SoftFSMs (one per primitive), sharing the same predicate
vocabulary but with independent transition matrices and readout vectors.

Output: A ∈ R^{B × T × K}
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SoftFSM(nn.Module):
    def __init__(self, n_states: int, n_predicates: int) -> None:
        super().__init__()
        self.n_states = n_states

        # T[predicate, from_state, to_state]
        self.T = nn.Parameter(torch.zeros(n_predicates, n_states, n_states))
        with torch.no_grad():
            # Strong identity bias: start as "stay put", learn to deviate
            eye = torch.eye(n_states) * 3.0
            self.T += eye.unsqueeze(0)

        # Activation readout
        self.w = nn.Parameter(torch.zeros(n_states))
        self.b = nn.Parameter(torch.tensor(-2.0))  # gates start near-closed

    def forward(
        self,
        predicate_ids: torch.Tensor,  # (B, T) long
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            states:      (B, T, n_states) — soft state distribution at each step
            activations: (B, T)           — sigmoid readout
        """
        B, T = predicate_ids.shape
        device = predicate_ids.device

        s = torch.zeros(B, self.n_states, device=device)
        s[:, 0] = 1.0  # start in state 0

        states_list: list[torch.Tensor] = []
        for t in range(T):
            p = predicate_ids[:, t]  # (B,)
            T_p = self.T[p]  # (B, S, S)
            # s_new[j] = Σ_i s[i] * T_p[i,j]  →  s @ T_p
            logits = torch.bmm(s.unsqueeze(1), T_p).squeeze(1)  # (B, S)
            s = torch.softmax(logits, dim=-1)
            states_list.append(s)

        states = torch.stack(states_list, dim=1)  # (B, T, S)
        activations = torch.sigmoid((states * self.w).sum(-1) + self.b)  # (B, T)
        return states, activations


class PrimitiveRouter(nn.Module):
    """K parallel SoftFSMs — one per primitive — over a shared predicate vocab.

    Args:
        specs:        list of (name, n_states) pairs
        n_predicates: shared predicate vocabulary size (from predicates.py)
    """

    def __init__(self, specs: list[tuple[str, int]], n_predicates: int) -> None:
        super().__init__()
        self.names = [name for name, _ in specs]
        self.fsms = nn.ModuleList([SoftFSM(n_states, n_predicates) for _, n_states in specs])

    @property
    def n_primitives(self) -> int:
        return len(self.fsms)

    def forward(
        self,
        predicate_ids: torch.Tensor,  # (B, T) long
    ) -> torch.Tensor:  # (B, T, K)
        return torch.stack(
            [fsm(predicate_ids)[1] for fsm in self.fsms],
            dim=-1,
        )
