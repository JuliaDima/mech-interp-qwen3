"""PrototypeModule architecture for the hierarchical algorithmic module.

Architecture (single primitive, no composition layer):
    Cross-layer read attention → BiGRU primitive → scalar gate → cross-layer write attention

All dimensions follow the spec:
    d_model = 2560  (Qwen3-4B residual stream)
    n_layers = 36
    d_small  = 128  (module internal dimension)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossLayerRead(nn.Module):
    """Attention over all Qwen layers producing module input x_t.

    Two modes controlled by `input_dependent`:

    Static (input_dependent=False):
        α_l = softmax_l( q · k^(l) )          — same weights for every token position
        x_t = W_read · Σ_l α_l · r_t^(l)

    Input-dependent (input_dependent=True):
        α_l^(t) = softmax_l( r_t^(ref) · k^(l) )  — per-token weights from ref_layer
        x_t = W_read · Σ_l α_l^(t) · r_t^(l)
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_small: int,
        ref_layer: int = 0,
        input_dependent: bool = True,
    ) -> None:
        super().__init__()
        self.ref_layer = ref_layer
        self.input_dependent = input_dependent
        self.k = nn.Parameter(torch.randn(n_layers, d_model) * 0.02)
        self.W_read = nn.Linear(d_model, d_small, bias=False)
        nn.init.orthogonal_(self.W_read.weight)
        if not input_dependent:
            # Learned static query; zeros → uniform attention at init
            self.q = nn.Parameter(torch.zeros(d_model))

    def attention_weights(self, residuals: torch.Tensor) -> torch.Tensor:
        """Softmax layer weights.  Shape: (batch, seq, n_layers) or (n_layers,)."""
        if self.input_dependent:
            q_t = residuals[self.ref_layer]  # (batch, seq, d_model)
            scores = q_t @ self.k.T  # (batch, seq, n_layers)
            return torch.softmax(scores, dim=-1)  # (batch, seq, n_layers)
        else:
            scores = self.q @ self.k.T  # (n_layers,)
            return torch.softmax(scores, dim=-1)  # (n_layers,)

    def forward(self, residuals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residuals: (n_layers, batch, seq, d_model) – Qwen residual streams
                       (detached; no gradient flows into Qwen).
        Returns:
            x: (batch, seq, d_small) – module inputs
        """
        r = residuals.permute(1, 2, 0, 3)  # (batch, seq, n_layers, d_model)
        alpha = self.attention_weights(residuals)
        if self.input_dependent:
            # alpha: (batch, seq, n_layers)
            r_weighted = (alpha.unsqueeze(-1) * r).sum(dim=2)  # (batch, seq, d_model)
        else:
            # alpha: (n_layers,) — broadcast over batch and seq
            r_weighted = (alpha.view(1, 1, -1, 1) * r).sum(dim=2)  # (batch, seq, d_model)
        return self.W_read(r_weighted)  # (batch, seq, d_small)


class CarryPrimitiveGRU(nn.Module):
    """Bidirectional GRU carry-propagation primitive.

    h_t^→ = GRU_f( h_{t-1}^→, x_t )
    h_t^← = GRU_b( h_{t+1}^←, x_t )
    f(x_t) = [ h_t^→ ‖ h_t^← ] ∈ R^d_small

    Bidirectionality is essential: carry propagates right-to-left but results
    are consumed left-to-right.
    """

    def __init__(self, d_small: int) -> None:
        super().__init__()
        if d_small % 2 != 0:
            raise ValueError(f"d_small must be even for bidirectional GRU, got {d_small}")
        self.gru = nn.GRU(
            input_size=d_small,
            hidden_size=d_small // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_small)
        Returns:
            f: (batch, seq, d_small) – [h_forward ‖ h_backward]
        """
        output, _ = self.gru(x)
        return output  # (batch, seq, d_small)


class ScalarGate(nn.Module):
    """Scalar gate controlling write magnitude at each token position.

    g_t = σ( v^T · f(x_t) )

    Prevents the module from corrupting non-arithmetic token positions.
    The training regularisation term λ·Σ_t(1−g_t) forces the gate open
    on positions where the write helps.
    """

    def __init__(self, d_small: int) -> None:
        super().__init__()
        # Initialise near zero so gate starts closed; regulariser opens it.
        self.v = nn.Parameter(torch.zeros(d_small))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f: (batch, seq, d_small)
        Returns:
            g: (batch, seq, 1) – gate activations in (0, 1)
        """
        return torch.sigmoid((f * self.v).sum(-1, keepdim=True))


class CrossLayerWrite(nn.Module):
    """Write: routes each slot's carry contribution to token positions.

    Routing uses slot attention weights (which tokens each slot corresponds to) —
    fixed from Stage 1b, no additional attention learned here.  Magnitude per slot
    is controlled by a learned carry gate sigmoid(W_gate · f_i + b), supervised by
    carry labels so it learns to be 1 when digit i has a carry and 0 otherwise.

    write_t = Σ_i slot_attn[i, t] · sigmoid(W_gate · f_i) · W_write(f_i)
    Δr_t^(l) = α_l^w · write_t

    This makes the write carry-content-specific: scrambling f scrambles both the
    gate values and the write content, breaking the rank-1 steering shortcut.
    """

    def __init__(self, n_layers: int, d_model: int, d_small: int) -> None:
        super().__init__()
        # Layer distribution
        self.q_w = nn.Parameter(torch.zeros(d_model))
        self.k_w = nn.Parameter(torch.randn(n_layers, d_model) * 0.02)
        # Per-slot carry gate: scalar logit per slot (sigmoid applied in forward)
        self.carry_gate = nn.Linear(d_small, 1, bias=True)
        nn.init.zeros_(self.carry_gate.weight)
        nn.init.constant_(self.carry_gate.bias, -1.0)  # sigmoid(-1) ≈ 0.27 → starts near-closed
        # Write projection
        self.W_write = nn.Linear(d_small, d_model, bias=False)
        nn.init.zeros_(self.W_write.weight)

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.q_w @ self.k_w.T, dim=0)  # (n_layers,)

    def forward(
        self,
        f: torch.Tensor,
        slot_attn_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f:                 (batch, n_digits, d_small) – BiGRU slot outputs
            slot_attn_weights: (batch, n_digits, seq)     – from DigitSlotAttention (detached)
        Returns:
            deltas:      (n_layers, batch, seq, d_model)
            carry_logits:(batch, n_digits, 1) – pre-sigmoid gate logits (for BCE supervision)
        """
        carry_logits = self.carry_gate(f)  # (batch, n_digits, 1)
        carry_gates = torch.sigmoid(carry_logits)  # (batch, n_digits, 1)
        # Detach gates from CE path: carry_gate trains on BCE only (carry prediction),
        # W_write trains on CE only (write effectiveness). No gradient conflict.
        v = self.W_write(f) * carry_gates.detach()  # (batch, n_digits, d_model)
        write_out = slot_attn_weights.transpose(-2, -1) @ v  # (batch, seq, d_model)
        alpha = self.layer_weights()  # (n_layers,)
        return alpha.view(-1, 1, 1, 1) * write_out.unsqueeze(0), carry_logits


class DigitSlotAttention(nn.Module):
    """Cross-attention with n_digits learned slot queries over the token sequence.

    Each slot attends over the full CrossLayerRead output x and learns to extract
    the digit-token features relevant to that position — slot i naturally learns to
    attend to a's i-th digit and b's i-th digit combined, without any positional
    hardcoding.  Mirrors Stage 1a's pair_embedding structure but derived from Qwen
    residuals.

    Output (batch, n_digits, d_small) is fed directly to CarryPrimitiveGRU.
    """

    def __init__(self, n_digits: int, d_small: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_digits, d_small) * 0.02)
        self.W_k = nn.Linear(d_small, d_small, bias=False)
        self.W_v = nn.Linear(d_small, d_small, bias=False)
        self.scale = d_small**-0.5

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq, d_small) — CrossLayerRead output
        Returns:
            slots:       (batch, n_digits, d_small)
            attn_weights:(batch, n_digits, seq) — used by CrossLayerWrite for routing
        """
        k = self.W_k(x)  # (batch, seq, d_small)
        v = self.W_v(x)  # (batch, seq, d_small)
        q = self.queries.unsqueeze(0)  # (1, n_digits, d_small)
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (batch, n_digits, seq)
        attn = torch.softmax(scores, dim=-1)  # (batch, n_digits, seq)
        return attn @ v, attn  # (batch, n_digits, d_small), (batch, n_digits, seq)


class PairEmbedding(nn.Module):
    """Embedding table mapping (a_digit*10 + b_digit) → R^d_small.

    Input is an integer in [0, 99] encoding one pair of single digits.
    Used exclusively in Stage 1a to give the BiGRU a direct digit-pair signal
    without any Qwen residual streams involved.
    """

    def __init__(self, d_small: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(100, d_small)

    def forward(self, pair_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_indices: (...) long tensor with values in [0, 99]
        Returns:
            x: (..., d_small)
        """
        return self.embedding(pair_indices)


class CarryHead(nn.Module):
    """Linear head predicting carry logits.

    Stage 1a: n_out=1, applied at each digit position → (batch, n_digits, 1).
    Stage 1b: n_out=n_digits, applied at the last prompt position → (batch, n_digits).
    """

    def __init__(self, d_small: int, n_out: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(d_small, n_out, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f: (batch, seq, d_small) or (batch, d_small)
        Returns:
            logits: (batch, seq, n_out) or (batch, n_out) — raw logits before sigmoid
        """
        return self.linear(f)


class Stage1Head(nn.Module):
    """Legacy: temporary linear probe from the original Stage 1 design (kept for reference).

    No longer used by any training script after the Stage 1a/1b split.
    """

    def __init__(self, d_small: int, d_vocab: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_small, d_vocab, bias=True)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f: (batch, seq, d_small)
        Returns:
            logits: (batch, seq, d_vocab)
        """
        return self.linear(f)


class PrototypeModule(nn.Module):
    """Full hierarchical module prototype.

    read → slot_attn → primitive (BiGRU) → write (carry-gated, slot-routed)

    Usage:
        deltas, f, carry_logits = module(residuals)
        # inject deltas into frozen Qwen via write hooks
        # carry_logits used for BCE supervision in Stage 2
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_small: int,
        n_digits: int,
        ref_layer: int = 0,
        input_dependent: bool = True,
    ) -> None:
        super().__init__()
        self.read = CrossLayerRead(
            n_layers, d_model, d_small, ref_layer=ref_layer, input_dependent=input_dependent
        )
        self.slot_attn = DigitSlotAttention(n_digits, d_small)
        self.primitive = CarryPrimitiveGRU(d_small)
        self.write = CrossLayerWrite(n_layers, d_model, d_small)

    def forward(
        self,
        residuals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full module pipeline.

        Args:
            residuals: (n_layers, batch, seq, d_model) – Qwen residual streams
        Returns:
            deltas:      (n_layers, batch, seq, d_model)
            f:           (batch, n_digits, d_small) – BiGRU slot outputs
            carry_logits:(batch, n_digits, 1)       – pre-sigmoid carry gate logits
        """
        x = self.read(residuals)  # (batch, seq, d_small)
        x_slots, slot_attn_weights = self.slot_attn(
            x
        )  # (batch, n_digits, d_small), (batch, n_digits, seq)
        f = self.primitive(x_slots)  # (batch, n_digits, d_small)
        deltas, carry_logits = self.write(
            f, slot_attn_weights
        )  # (n_layers, batch, seq, d_model), (batch, n_digits, 1)
        return deltas, f, carry_logits
