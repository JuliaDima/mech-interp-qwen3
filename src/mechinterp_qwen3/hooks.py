# TODO: Attention is detached, This works fine for CLT, but not really for PLT

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

    Tensor Definitions:
    - mlp_in: The activation immediately preceding the MLP block.
      For Qwen3, this is technically the output of `post_attention_layernorm`.
      Shape: `[batch_size, seq_len, d_model]`, usually with batch_size=1.
      Location: `model.model.layers[{lid}].mlp.input` (conceptually), or
      the output of `model.model.layers[{lid}].post_attention_layernorm`.
    - mlp_out: The activation immediately after the MLP block finishes.
      This corresponds to the residual contribution of the MLP before being added
      to the residual stream.
      Shape: `[batch_size, seq_len, d_model]`, usually with batch_size=1.
      Location: `model.model.layers[{lid}].mlp.output`.
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
                x = inputs[0]  # [1, seq_len, d_model]
                if self.detach:  # TODO: for now, only batch 0, but can extend later
                    self.cache[lid].mlp_in = x[0].detach().to("cpu")
                else:
                    self.cache[lid].mlp_in = x[0]

            def fwd_hook(module, inputs, output, lid=lid):
                # If tuple, first element is [batch, seq, d_model]
                # If plain tensor, it's directly [batch, seq, d_model]
                y = output[0] if isinstance(output, tuple) else output

                # y is [batch, seq, d_model]
                if self.detach:  # TODO: for now, only batch 0, but can extend later
                    self.cache[lid].mlp_out = y[0].detach().to("cpu")
                else:
                    # Keep the FULL tensor (not y[0]) so it remains in the
                    # computation graph.  y[0] creates a view that is NOT an
                    # ancestor of logits, causing autograd.grad to return None.
                    y.retain_grad()
                    self.cache[lid].mlp_out = y

            self.handles.append(mlp.register_forward_pre_hook(pre_hook))
            self.handles.append(mlp.register_forward_hook(fwd_hook))

    def clear_cache(self) -> None:
        for lid in self.layer_ids:
            self.cache[lid] = LayerActs()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


class LinearizedHookManager:
    """Linearized gradient flow hooks matching the Attribution Graphs paper.

    Installs three types of hooks:
    1. Embedding hook: enables gradients on embedding output.
       Tensor: `model.model.embed_tokens.output`.
       Shape: `[batch_size, seq_len, d_model]`.
    2. Attention detach hooks: detach attention outputs so gradients only flow
       through the residual skip connections (currently patched externally).
    3. RMSNorm linearize hooks: treat normalization scale as constant in backward.
       Tensors: `input_layernorm.output`, `post_attention_layernorm.output`,
       and `model.model.norm.output`.
       Shape: `[batch_size, seq_len, d_model]`.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def install(self) -> None:
        # 1. Embedding hook: enable gradients on embedding output
        embed = self.model.model.embed_tokens

        def embed_hook(module, input, output):
            output.requires_grad_(True)
            return output

        self.handles.append(embed.register_forward_hook(embed_hook))

        # 2. Attention detach hooks: detach attention output, re-enable grad
        # REMOVED: Unconditional detachment loop caused zero attributions.
        # Strict attention freezing (Pattern only) is now handled via monkey-patching in `forward_with_sae.py`.

        # 3. RMSNorm linearize hooks: treat scale factor as constant
        norm_modules = []
        for layer in self.model.model.layers:
            norm_modules.append(layer.input_layernorm)
            norm_modules.append(layer.post_attention_layernorm)
        norm_modules.append(self.model.model.norm)

        for norm in norm_modules:
            # Standard RMSNorm forward pass
            # rms_scale = sqrt(mean(x^2) + eps)
            # output = weight * (x / rms_scale)
            # But we linearize the scale factor
            def norm_hook(module, input, output):
                x = input[0]
                with torch.no_grad():  # freeze the denominator
                    rms_scale = (
                        x.float().pow(2).mean(-1, keepdim=True).add(module.variance_epsilon).rsqrt()
                    )
                return module.weight * (x.float() * rms_scale).to(x.dtype)

            self.handles.append(norm.register_forward_hook(norm_hook))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


class ReconstructivePatchingManager:
    """Replaces MLP outputs with SAE reconstructions during forward pass.

    This enables gradients between features across different layers.

    Tensor Definitions:
    - Features: The sparse activations `f(x)` produced by the SAE encoder.
      Shape: `[batch_size, seq_len, d_sae]`.
      Mathematically: `f(x) = ReLU(W_enc(x - b_dec) + b_enc)`.
    - Reconstruction: The reconstructed MLP input `x_hat` produced by the SAE
      decoder, which is patched in place of the true MLP output.
      Shape: `[batch_size, seq_len, d_model]`.
      Mathematically: `x_hat = W_dec(f(x)) + b_dec`.
    """

    def __init__(self, model: nn.Module, transcoders: dict[int, Any]):
        self.model = model
        self.transcoders = transcoders
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.sae_features: dict[int, torch.Tensor] = {}
        self.reconstructions: dict[int, torch.Tensor] = {}

    def _get_layers(self) -> list[nn.Module]:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        raise RuntimeError("Unsupported model layout: expected model.model.layers")

    def install(self) -> None:
        layers = self._get_layers()

        for lid, transcoder in self.transcoders.items():
            block = layers[lid]
            mlp = block.mlp

            def patching_hook(module, inputs, output, lid=lid, transcoder=transcoder):
                # output is [batch, seq, d_model]
                # inputs[0] is mlp_input [batch, seq, d_model]
                mlp_in = inputs[0]

                # Perform SAE reconstruction
                # encode expects [batch, seq, d_model]
                features = transcoder.encode(mlp_in)

                # We save features for later attribution analysis
                # (but they are also part of the graph now!)
                self.sae_features[lid] = features

                # Decode to get reconstruction
                reconstruction = transcoder.decode(features, mlp_in)

                # Cache reconstruction
                self.reconstructions[lid] = reconstruction

                # Return reconstruction instead of original MLP output
                return reconstruction

            self.handles.append(mlp.register_forward_hook(patching_hook))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
