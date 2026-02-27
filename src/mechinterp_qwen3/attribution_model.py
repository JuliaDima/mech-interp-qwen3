import contextlib
import warnings
import weakref
from collections.abc import Callable
from functools import partial
from typing import Literal

import numpy as np
import torch
from einops import einsum
from torch import nn
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint

from .transcoder import TranscoderSet
from .transcoder.cross_layer_transcoder import CrossLayerTranscoder
from .utils.hf_utils import load_transcoder_from_hub
from .utils.model_utils import get_default_device


class AttributionMLP(nn.Module):
    """Wrapper for a TransformerLens MLP layer that adds in extra hooks"""

    def __init__(self, old_mlp: nn.Module):
        super().__init__()
        self.old_mlp = old_mlp
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()

    def forward(self, x):
        x = self.hook_in(x)
        mlp_out = self.old_mlp(x)
        return self.hook_out(mlp_out)


class AttributionUnembed(nn.Module):
    """Wrapper for a TransformerLens Unembed layer that adds in extra hooks"""

    def __init__(self, old_unembed: nn.Module):
        super().__init__()
        self.old_unembed = old_unembed
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    @property
    def W_U(self):
        return self.old_unembed.W_U

    @property
    def b_U(self):
        return self.old_unembed.b_U

    def forward(self, x):
        x = self.hook_pre(x)
        x = self.old_unembed(x)
        return self.hook_post(x)


class AttributionModel(HookedTransformer):
    transcoders: TranscoderSet | CrossLayerTranscoder  # Support both types
    feature_input_hook: str
    feature_output_hook: str
    skip_transcoder: bool
    scan: str | list[str] | None
    backend: Literal["transformerlens"]

    @classmethod
    def from_config(
        cls,
        config: HookedTransformerConfig,
        transcoders: TranscoderSet | CrossLayerTranscoder,  # Accept both
        **kwargs,
    ) -> "AttributionModel":
        """Create an AttributionModel from a given HookedTransformerConfig and TranscoderSet

        Args:
            config (HookedTransformerConfig): the config of the HookedTransformer
            transcoders (TranscoderSet): The transcoder set with configuration

        Returns:
            AttributionModel: The loaded AttributionModel
        """
        model = cls(config, **kwargs)
        model._configure_attribution_model(transcoders)
        return model

    @classmethod
    def from_pretrained_and_transcoders(
        cls,
        model_name: str,
        transcoders: TranscoderSet | CrossLayerTranscoder,  # Accept both
        **kwargs,
    ) -> "AttributionModel":
        """Create an AttributionModel from the name of HookedTransformer and TranscoderSet

        Args:
            model_name (str): the name of the pretrained HookedTransformer
            transcoders (TranscoderSet): The transcoder set with configuration

        Returns:
            AttributionModel: The loaded AttributionModel
        """
        model = super().from_pretrained(
            model_name,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            **kwargs,
        )

        model._configure_attribution_model(transcoders)
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        transcoder_set: str,
        device: torch.device | None = None,
        dtype: torch.dtype | None = torch.float32,
        **kwargs,
    ) -> "AttributionModel":
        """Create an AttributionModel from model name and transcoder config

        Args:
            model_name (str): the name of the pretrained HookedTransformer
            transcoder_set (str): Either a predefined transcoder set name, or a config file
            device (torch.device | None): The device to load the model and transcoders on.
                If None, uses the default device. Defaults to None.
            dtype (torch.dtype): The dtype to use for the model and transcoders.
                Defaults to torch.float32.
            **kwargs: Additional keyword arguments passed to HookedTransformer.from_pretrained

        Returns:
            AttributionModel: The loaded AttributionModel
        """
        if device is None:
            device = get_default_device()

        (
            transcoders,
            _,
        ) = load_transcoder_from_hub(transcoder_set, device=device, dtype=dtype)  # type: ignore

        return cls.from_pretrained_and_transcoders(
            model_name,
            transcoders,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    def _configure_attribution_model(self, transcoder_set: TranscoderSet | CrossLayerTranscoder):
        self.backend = "transformerlens"
        transcoder_set.to(self.cfg.device, self.cfg.dtype)

        self.transcoders = transcoder_set
        self.feature_input_hook = transcoder_set.feature_input_hook
        self.original_feature_output_hook = transcoder_set.feature_output_hook
        self.feature_output_hook = transcoder_set.feature_output_hook + ".hook_out_grad"
        self.skip_transcoder = transcoder_set.skip_connection
        self.scan = transcoder_set.scan

        for block in self.blocks:
            block.mlp = AttributionMLP(block.mlp)  # type: ignore

        self.unembed = AttributionUnembed(self.unembed)

        self._configure_gradient_flow()
        self._deduplicate_attention_buffers()
        self.setup()

    def _configure_gradient_flow(self):
        for layer in range(self.cfg.n_layers):
            self._configure_skip_connection(self.blocks[layer], self.transcoders, layer)

        def stop_gradient(acts, hook):
            return acts.detach()

        for block in self.blocks:
            block.attn.hook_pattern.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            block.ln1.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            block.ln2.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            if hasattr(block, "ln1_post"):
                block.ln1_post.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            if hasattr(block, "ln2_post"):
                block.ln2_post.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            self.ln_final.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore

        for param in self.parameters():
            param.requires_grad = False

        def enable_gradient(acts, hook):
            acts.requires_grad = True
            return acts

        self.hook_embed.add_hook(enable_gradient, is_permanent=True)  # type: ignore

    def _configure_skip_connection(
        self, block, transcoders: TranscoderSet | CrossLayerTranscoder, layer: int
    ):
        cached = {}

        def cache_activations(acts, hook):
            cached["acts"] = acts

        def add_skip_connection(acts: torch.Tensor, hook: HookPoint, grad_hook: HookPoint):
            # We add grad_hook because we need a way to hook into the gradients of the output
            # of this function. If we put the backwards hook here at hook, the grads will be 0
            # because we detached acts.
            skip_input_activation = cached.pop("acts")
            if transcoders.skip_connection:
                skip = transcoders.compute_skip(layer, skip_input_activation)
            else:
                skip = skip_input_activation * 0
            return grad_hook(skip + (acts - skip).detach())

        # add feature input hook
        output_hook_parts = self.feature_input_hook.split(".")
        subblock = block
        for part in output_hook_parts:
            subblock = getattr(subblock, part)
        subblock.add_hook(cache_activations, is_permanent=True)

        # add feature output hook and special grad hook
        output_hook_parts = self.original_feature_output_hook.split(".")
        subblock = block
        for part in output_hook_parts:
            subblock = getattr(subblock, part)
        subblock.hook_out_grad = HookPoint()
        subblock.add_hook(
            partial(add_skip_connection, grad_hook=subblock.hook_out_grad),
            is_permanent=True,
        )

    def _deduplicate_attention_buffers(self):
        """
        Share attention buffers across layers to save memory.

        TransformerLens makes separate copies of the same masks and RoPE
        embeddings for each layer - This just keeps one copy
        of each and shares it across all layers.
        """

        attn_masks = {}

        for block in self.blocks:
            attn_masks[block.attn.attn_type] = block.attn.mask  # type: ignore
            if hasattr(block.attn, "rotary_sin"):
                attn_masks["rotary_sin"] = block.attn.rotary_sin  # type: ignore
                attn_masks["rotary_cos"] = block.attn.rotary_cos  # type: ignore

        for block in self.blocks:
            block.attn.mask = attn_masks[block.attn.attn_type]  # type: ignore
            if hasattr(block.attn, "rotary_sin"):
                block.attn.rotary_sin = attn_masks["rotary_sin"]  # type: ignore
                block.attn.rotary_cos = attn_masks["rotary_cos"]  # type: ignore

    def _get_activation_caching_hooks(
        self,
        sparse: bool = False,
        apply_activation_function: bool = True,
        append: bool = False,
    ) -> tuple[list[torch.Tensor], list[tuple[str, Callable]]]:
        activation_matrix = (
            [[] for _ in range(self.cfg.n_layers)] if append else [None] * self.cfg.n_layers
        )

        def cache_activations(acts, hook, layer):
            transcoder_acts = (
                self.transcoders.encode_layer(
                    acts, layer, apply_activation_function=apply_activation_function
                )
                .detach()
                .squeeze(0)
            )

            if not append:
                transcoder_acts[0] = 0

            if sparse:
                transcoder_acts = transcoder_acts.to_sparse()

            if append:
                activation_matrix[layer].append(transcoder_acts)
            else:
                activation_matrix[layer] = transcoder_acts  # type: ignore

        activation_hooks = [
            (
                f"blocks.{layer}.{self.feature_input_hook}",
                partial(cache_activations, layer=layer),
            )
            for layer in range(self.cfg.n_layers)
        ]
        return activation_matrix, activation_hooks  # type: ignore

    def get_activations(
        self,
        inputs: str | torch.Tensor,
        sparse: bool = False,
        apply_activation_function: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the transcoder activations for a given prompt

        Args:
            inputs (str | torch.Tensor): The inputs you want to get activations over
            sparse (bool, optional): Whether to return a sparse tensor of activations.
                Useful if d_transcoder is large. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: the model logits on the inputs and the
                associated activation cache
        """

        activation_cache, activation_hooks = self._get_activation_caching_hooks(
            sparse=sparse,
            apply_activation_function=apply_activation_function,
        )
        with torch.inference_mode(), self.hooks(activation_hooks):  # type: ignore
            logits = self(inputs)
        activation_cache = torch.stack(activation_cache)
        if sparse:
            activation_cache = activation_cache.coalesce()
        return logits, activation_cache

    @contextlib.contextmanager
    def zero_softcap(self):
        current_softcap = self.cfg.output_logits_soft_cap
        try:
            self.cfg.output_logits_soft_cap = 0.0
            yield
        finally:
            self.cfg.output_logits_soft_cap = current_softcap

    def ensure_tokenized(self, prompt: str | torch.Tensor | list[int]) -> torch.Tensor:
        """Convert prompt to 1-D tensor of token ids with proper special token handling."""

        if isinstance(prompt, str):
            tokens = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.squeeze(0)  # type: ignore
        elif isinstance(prompt, torch.Tensor):
            tokens = prompt.squeeze()
        elif isinstance(prompt, list):
            tokens = torch.tensor(prompt, dtype=torch.long).squeeze()
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        if tokens.ndim > 1:
            raise ValueError(f"Tensor must be 1-D, got shape {tokens.shape}")

        if tokens[0] in self.tokenizer.all_special_ids:  # type: ignore
            return tokens.to(self.cfg.device)

        candidate_bos_token_ids = [
            self.tokenizer.bos_token_id,  # type: ignore
            self.tokenizer.pad_token_id,  # type: ignore
            self.tokenizer.eos_token_id,  # type: ignore
        ]
        candidate_bos_token_ids += self.tokenizer.all_special_ids  # type: ignore

        dummy_bos_token_id = next(filter(None, candidate_bos_token_ids))
        if dummy_bos_token_id is None:
            warnings.warn(
                "No suitable special token found for BOS token replacement. The first token will be ignored.",
                stacklevel=2,
            )
        else:
            tokens = torch.cat([torch.tensor([dummy_bos_token_id], device=tokens.device), tokens])

        return tokens.to(self.cfg.device)

    @torch.no_grad()
    def setup_attribution(self, inputs: str | torch.Tensor):
        tokens = self.ensure_tokenized(inputs) if isinstance(inputs, str) else inputs.squeeze()

        assert isinstance(tokens, torch.Tensor), "Tokens must be a tensor"
        assert tokens.ndim == 1, "Tokens must be a 1D tensor"

        mlp_in_cache, mlp_in_caching_hooks, _ = self.get_caching_hooks(
            lambda name: self.feature_input_hook in name
        )

        mlp_out_cache, mlp_out_caching_hooks, _ = self.get_caching_hooks(
            lambda name: self.feature_output_hook in name
        )
        logits = self.run_with_hooks(tokens, fwd_hooks=mlp_in_caching_hooks + mlp_out_caching_hooks)

        mlp_in_cache = torch.cat(list(mlp_in_cache.values()), dim=0)
        mlp_out_cache = torch.cat(list(mlp_out_cache.values()), dim=0)

        attribution_data = self.transcoders.compute_attribution_components(mlp_in_cache)

        error_vectors = mlp_out_cache - attribution_data["reconstruction"]
        error_vectors[:, 0] = 0
        token_vectors = self.W_E[tokens].detach()

        return AttributionContext(
            activation_matrix=attribution_data["activation_matrix"],
            logits=logits,
            error_vectors=error_vectors,
            token_vectors=token_vectors,
            decoder_vecs=attribution_data["decoder_vecs"],
            encoder_vecs=attribution_data["encoder_vecs"],
            encoder_to_decoder_map=attribution_data["encoder_to_decoder_map"],
            decoder_locations=attribution_data["decoder_locations"],
        )


class AttributionContext:
    """Manage hooks for computing attribution rows."""

    def __init__(
        self,
        activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        encoder_vecs: torch.Tensor,
        encoder_to_decoder_map: torch.Tensor,
        decoder_locations: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        n_layers, n_pos, _ = activation_matrix.shape

        self._resid_activations: list[torch.Tensor | None] = [None] * (n_layers + 1)
        self._batch_buffer: torch.Tensor | None = None
        self.n_layers: int = n_layers

        self.logits = logits
        self.activation_matrix = activation_matrix
        self.error_vectors = error_vectors
        self.token_vectors = token_vectors
        self.decoder_vecs = decoder_vecs
        self.encoder_vecs = encoder_vecs

        self.encoder_to_decoder_map = encoder_to_decoder_map
        self.decoder_locations = decoder_locations

        total_active_feats = activation_matrix._nnz()
        self._row_size: int = total_active_feats + (n_layers + 1) * n_pos

    def _caching_hooks(self, feature_input_hook: str) -> list[tuple[str, Callable]]:
        proxy = weakref.proxy(self)

        def _cache(acts: torch.Tensor, hook: HookPoint, *, layer: int) -> torch.Tensor:
            proxy._resid_activations[layer] = acts
            return acts

        hooks = [
            (f"blocks.{layer}.{feature_input_hook}", partial(_cache, layer=layer))
            for layer in range(self.n_layers)
        ]
        hooks.append(("unembed.hook_pre", partial(_cache, layer=self.n_layers)))
        return hooks

    def _compute_score_hook(
        self,
        hook_name: str,
        output_vecs: torch.Tensor,
        write_index: slice,
        read_index: slice | np.ndarray = np.s_[:],
    ) -> tuple[str, Callable]:
        proxy = weakref.proxy(self)

        def _hook_fn(grads: torch.Tensor, hook: HookPoint) -> None:
            proxy._batch_buffer[write_index] += einsum(
                grads.to(output_vecs.dtype)[read_index],
                output_vecs,
                "batch position d_model, position d_model -> position batch",
            )

        return hook_name, _hook_fn

    def _make_attribution_hooks(self, feature_output_hook: str) -> list[tuple[str, Callable]]:
        n_layers, n_pos, _ = self.activation_matrix.shape
        nnz_layers, nnz_positions = self.decoder_locations

        feature_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{feature_output_hook}",
                self.decoder_vecs[layer_mask],
                write_index=self.encoder_to_decoder_map[layer_mask],
                read_index=np.s_[:, nnz_positions[layer_mask]],
            )
            for layer in range(n_layers)
            if (layer_mask := nnz_layers == layer).any()
        ]

        def error_offset(layer: int) -> int:
            return self.activation_matrix._nnz() + layer * n_pos

        error_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{feature_output_hook}",
                self.error_vectors[layer],
                write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
            )
            for layer in range(n_layers)
            if layer < len(self.error_vectors)
        ]

        tok_start = error_offset(n_layers)
        token_hook = [
            self._compute_score_hook(
                "hook_embed",
                self.token_vectors,
                write_index=np.s_[tok_start : tok_start + n_pos],
            )
        ]

        return feature_hooks + error_hooks + token_hook

    @contextlib.contextmanager
    def install_hooks(self, model: "AttributionModel"):
        with model.hooks(
            fwd_hooks=self._caching_hooks(model.feature_input_hook),
            bwd_hooks=self._make_attribution_hooks(model.feature_output_hook),
        ):
            yield

    def compute_batch(
        self,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        batch_size = self._resid_activations[0].shape[0]
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=inject_values.dtype,
            device=inject_values.device,
        )

        batch_idx = torch.arange(len(layers), device=layers.device)

        def _inject(grads, *, batch_indices, pos_indices, values):
            grads_out = grads.clone().to(values.dtype)
            grads_out.index_put_((batch_indices, pos_indices), values)
            return grads_out.to(grads.dtype)

        handles = []
        layers_in_batch = layers.unique().tolist()

        for layer in layers_in_batch:
            mask = layers == layer
            if not mask.any():
                continue
            fn = partial(
                _inject,
                batch_indices=batch_idx[mask],
                pos_indices=positions[mask],
                values=inject_values[mask],
            )
            handles.append(self._resid_activations[int(layer)].register_hook(fn))

        try:
            last_layer = max(layers_in_batch)
            self._resid_activations[last_layer].backward(
                gradient=torch.zeros_like(self._resid_activations[last_layer]),
                retain_graph=retain_graph,
            )
        finally:
            for h in handles:
                h.remove()

        buf, self._batch_buffer = self._batch_buffer, None
        return buf.T[: len(layers)]
