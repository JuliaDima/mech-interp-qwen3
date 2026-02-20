# TODO: Attention is detached, This works fine for CLT, but not really for PLT

from __future__ import annotations

import torch
import torch.nn as nn


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
