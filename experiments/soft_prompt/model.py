"""Soft-prompt and prefix-tuning modules for addition task conditioning.

Two variants:
  SoftPrompt   — k learnable vectors prepended at the embedding layer only.
                 Cheapest: 1 × k × d_model parameters.

  PrefixTuning — k learnable vectors prepended at the residual stream of every
                 layer (blocks.{L}.hook_resid_pre).  More expressive but costs
                 n_layers × k × d_model parameters.

Both work the same way at inference time:
  1. Prepend k dummy (pad) token ids to the real input ids.
  2. Extend the attention mask with k ones at the front.
  3. Register a forward hook that replaces the first k positions with the
     learned prefix vectors.

This avoids mid-forward sequence-length changes and is compatible with the
existing run_with_hooks API.

Usage (SoftPrompt):
    sp = SoftPrompt(k=10, d_model=2560)
    input_ids, attn_mask = sp.prepare_inputs(real_ids, pad_token_id)
    hooks = sp.hooks(batch_size=1)
    logits = model.run_with_hooks(input_ids, fwd_hooks=hooks,
                                  attention_mask=attn_mask)

Usage (PrefixTuning):
    pt = PrefixTuning(k=10, d_model=2560, n_layers=36)
    input_ids, attn_mask = pt.prepare_inputs(real_ids, pad_token_id)
    hooks = pt.hooks(batch_size=1)
    logits = model.run_with_hooks(input_ids, fwd_hooks=hooks,
                                  attention_mask=attn_mask)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class SoftPrompt(nn.Module):
    """k learnable prefix vectors injected at the embedding layer.

    Parameters
    ----------
    k:
        Number of prefix tokens.
    d_model:
        Residual stream / embedding dimension of the large model.
    init_std:
        Std for random initialisation of the prefix.
    """

    mode = "soft_prompt"

    def __init__(self, k: int = 10, d_model: int = 2560, init_std: float = 0.02) -> None:
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.prefix = nn.Parameter(torch.randn(k, d_model) * init_std)

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        token_ids: torch.Tensor,
        pad_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepend k pad tokens and extend the attention mask.

        Parameters
        ----------
        token_ids:
            1-D tensor of shape (seq_len,) or 2-D (batch, seq_len).
        pad_token_id:
            Token id used as placeholder (will be overwritten by the hook).

        Returns
        -------
        extended_ids:   (batch, k + seq_len)
        attention_mask: (batch, k + seq_len)  — all 1s (prefix is always attended)
        """
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        B, L = token_ids.shape
        device = token_ids.device
        prefix_ids = torch.full((B, self.k), pad_token_id, dtype=torch.long, device=device)
        extended_ids = torch.cat([prefix_ids, token_ids], dim=1)
        attn_mask = torch.ones(B, self.k + L, dtype=torch.long, device=device)
        return extended_ids, attn_mask

    # ------------------------------------------------------------------
    # Hook factory
    # ------------------------------------------------------------------

    def hooks(self, batch_size: int = 1) -> list[tuple[str, object]]:
        """Return a list of (hook_name, hook_fn) tuples for run_with_hooks."""
        k = self.k
        prefix = self.prefix  # (k, d_model)

        def _embed_hook(embeddings: torch.Tensor, hook) -> torch.Tensor:
            # embeddings: (batch, k + seq_len, d_model)
            embeddings = embeddings.clone()
            embeddings[:, :k, :] = prefix.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
            return embeddings

        return [("hook_embed", _embed_hook)]

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mode": self.mode,
                "k": self.k,
                "d_model": self.d_model,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> SoftPrompt:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        obj = cls(k=ckpt["k"], d_model=ckpt["d_model"])
        obj.load_state_dict(ckpt["state_dict"])
        return obj.to(device)


class PrefixTuning(nn.Module):
    """k learnable prefix vectors injected at every layer's residual stream.

    Parameters
    ----------
    k:
        Number of prefix tokens.
    d_model:
        Residual stream dimension.
    n_layers:
        Number of transformer layers.
    init_std:
        Std for random initialisation.
    """

    mode = "prefix_tuning"

    def __init__(
        self,
        k: int = 10,
        d_model: int = 2560,
        n_layers: int = 36,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.n_layers = n_layers
        # One prefix tensor per layer; shape (n_layers, k, d_model)
        self.prefix = nn.Parameter(torch.randn(n_layers, k, d_model) * init_std)

    # ------------------------------------------------------------------
    # Input preparation (identical to SoftPrompt)
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        token_ids: torch.Tensor,
        pad_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        B, L = token_ids.shape
        device = token_ids.device
        prefix_ids = torch.full((B, self.k), pad_token_id, dtype=torch.long, device=device)
        extended_ids = torch.cat([prefix_ids, token_ids], dim=1)
        attn_mask = torch.ones(B, self.k + L, dtype=torch.long, device=device)
        return extended_ids, attn_mask

    # ------------------------------------------------------------------
    # Hook factory
    # ------------------------------------------------------------------

    def hooks(self, batch_size: int = 1) -> list[tuple[str, object]]:
        """Return per-layer hook list for run_with_hooks."""
        k = self.k
        hook_list = []

        for layer in range(self.n_layers):
            layer_prefix = self.prefix[layer]  # (k, d_model)

            def _resid_hook(
                resid: torch.Tensor,
                hook,
                _lp: torch.Tensor = layer_prefix,
                _k: int = k,
            ) -> torch.Tensor:
                resid = resid.clone()
                resid[:, :_k, :] = _lp.unsqueeze(0).expand(resid.shape[0], -1, -1)
                return resid

            hook_list.append((f"blocks.{layer}.hook_resid_pre", _resid_hook))

        return hook_list

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mode": self.mode,
                "k": self.k,
                "d_model": self.d_model,
                "n_layers": self.n_layers,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> PrefixTuning:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        obj = cls(k=ckpt["k"], d_model=ckpt["d_model"], n_layers=ckpt["n_layers"])
        obj.load_state_dict(ckpt["state_dict"])
        return obj.to(device)


def load_prefix(path: str | Path, device: torch.device | str = "cpu") -> SoftPrompt | PrefixTuning:
    """Load either a SoftPrompt or PrefixTuning checkpoint by inspecting the 'mode' key."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt["mode"] == "soft_prompt":
        return SoftPrompt.load(path, device=device)
    elif ckpt["mode"] == "prefix_tuning":
        return PrefixTuning.load(path, device=device)
    raise ValueError(f"Unknown prefix mode: {ckpt['mode']}")
