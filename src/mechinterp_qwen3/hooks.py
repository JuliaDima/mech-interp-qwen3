from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LayerActs:
    mlp_in: torch.Tensor | None = None  # [seq, d_model]
    mlp_out: torch.Tensor | None = None  # [seq, d_model]


class MLPHookManager:
    """
    Collects MLP input/output activations for a small list of layers.
    Intended for *prompt-only* forward passes (no autoregressive loop).
    """

    def __init__(self, model: nn.Module, layer_ids: list[int], detach: bool = True):
        self.model = model
        self.layer_ids = layer_ids
        self.detach = detach  # Whether to detach activations (breaks gradients)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.cache: dict[int, LayerActs] = {i: LayerActs() for i in layer_ids}

    def _get_layers(self) -> list[nn.Module]:
        # Works for many HF causal LMs: model.model.layers
        # Qwen typically follows this layout.
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        raise RuntimeError("Unsupported model layout: expected model.model.layers")

    def install(self) -> None:
        layers = self._get_layers()

        for lid in self.layer_ids:
            block = layers[lid]
            if not hasattr(block, "mlp"):
                raise RuntimeError(f"Layer {lid} has no .mlp module; adjust hook path.")

            mlp = block.mlp

            def pre_hook(module, inputs, lid=lid):
                # inputs is a tuple; first is hidden_states [batch, seq, d_model]
                x = inputs[0]
                # Save batch=0; you can extend to batch later.
                if self.detach:
                    self.cache[lid].mlp_in = x[0].detach().to("cpu")
                else:
                    self.cache[lid].mlp_in = x[0]

            def fwd_hook(module, inputs, output, lid=lid):
                # output should be [batch, seq, d_model]
                y = output
                if self.detach:
                    self.cache[lid].mlp_out = y[0].detach().to("cpu")
                else:
                    # Keep on device and retain gradient for non-leaf tensor
                    mlp_out = y[0]
                    mlp_out.retain_grad()
                    self.cache[lid].mlp_out = mlp_out

            self.handles.append(mlp.register_forward_pre_hook(pre_hook))
            self.handles.append(mlp.register_forward_hook(fwd_hook))

    def clear_cache(self) -> None:
        for lid in self.layer_ids:
            self.cache[lid] = LayerActs()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
