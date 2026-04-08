"""Linear logistic probe for carry detection from transcoder activations."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class CarryProbe(nn.Module):
    """Simple linear logistic probe for carry detection from transcoder activations.

    Optimized for mechanistic interpretability with single-layer, single-token analysis:
        Input: a_l(x) of shape [batch, d_transcoder] from ONE layer at ONE token position
        Linear: w^T * a_l(x) + b -> [batch, 1]
        Output: ŷ = sigmoid(z) -> [batch, 1] probability

    This simple architecture allows:
    - Direct feature importance (one weight per transcoder feature)
    - Layer-wise localization (test each layer independently)
    - Causal intervention (ablate specific features)

    Attributes:
        layer: Single transcoder layer index to use
        d_transcoder: Dimension of transcoder feature space (163,840 for Qwen3-4B)
        linear: Linear layer [d_transcoder] -> [1]
    """

    def __init__(
        self,
        layers: list[int] | None = None,
        d_transcoder: int = 163840,
        max_seq_len: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        n_layers: int | None = None,
        n_classes: int = 1,
    ):
        """Initialize the carry probe.

        Args:
            layers: List of layer indices. If None, defaults to list(range(n_layers))
            d_transcoder: Dimension of the transcoder feature space
            max_seq_len: Ignored (kept for compatibility with existing code)
            device: Device to place parameters on
            dtype: Data type for parameters
            n_layers: Total number of layers. Used if layers is None.
            n_classes: Number of output classes. 1 for binary (sigmoid+BCE),
                >1 for multiclass (softmax+CrossEntropy).
        """
        super().__init__()

        if layers is None:
            if n_layers is None:
                raise ValueError("Must provide either 'layers' or 'n_layers'")
            layers = list(range(n_layers))

        if not layers:
            raise ValueError("layers list cannot be empty")

        self.layers = layers
        self.layer = layers[0] if len(layers) == 1 else -1
        self.d_transcoder = d_transcoder
        self.max_seq_len = max_seq_len  # Kept for compatibility
        self.n_layers = len(layers)
        self.n_classes = n_classes

        out_features = n_classes if n_classes > 1 else 1
        # Linear layer: [d_transcoder * num_layers] -> [out_features]
        self.linear = nn.Linear(
            d_transcoder * self.n_layers, out_features, device=device, dtype=dtype
        )

    def forward(
        self,
        activations: dict[int, torch.Tensor] | torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Compute probe predictions from transcoder activations.

        Args:
            activations: Either:
                - dict mapping layer index to activation tensor [batch, d_transcoder]
                - tensor of shape [batch, d_transcoder * n_layers] or [batch, d_transcoder]
            return_logits: If True, return raw logits instead of probabilities

        Returns:
            Predictions: [batch] tensor of probabilities (or logits)
        """

        if isinstance(activations, dict):
            # Extract and concatenate activations for all layers in order
            tensors = []
            for L in self.layers:
                if L not in activations:
                    raise ValueError(f"Missing activations for layer {L}")
                tensors.append(activations[L])
            x = torch.cat(tensors, dim=-1)  # [batch, d_transcoder * n_layers]
        elif isinstance(activations, torch.Tensor):
            if self.n_layers > 1 and activations.shape[-1] == self.d_transcoder:
                raise ValueError(
                    f"Received single tensor of shape {activations.shape}, but expected activations for {self.n_layers} layers. "
                    f"Pass a dict of layer activations or a pre-concatenated tensor of feature dim {self.d_transcoder * self.n_layers}."
                )
            x = activations  # [batch, d_transcoder * n_layers]
        else:
            raise TypeError(f"activations must be dict or tensor, got {type(activations)}")

        # Validate shape
        if x.ndim != 2:
            raise ValueError(
                f"Expected activations shape [batch, d_transcoder * n_layers={self.d_transcoder * self.n_layers}], "
                f"got shape {x.shape} (ndim={x.ndim}). "
                f"Use token_position='final' or specific index to extract single token."
            )

        if x.shape[1] != self.d_transcoder * self.n_layers:
            raise ValueError(
                f"Expected feature dim {self.d_transcoder * self.n_layers}, got {x.shape[1]}"
            )

        # Convert dtype if needed
        x = x.to(self.linear.weight.dtype)

        if self.n_classes > 1:
            # Multiclass: [batch, d_transcoder * n_layers] -> [batch, n_classes]
            logits = self.linear(x)
            if return_logits:
                return logits
            return torch.softmax(logits, dim=-1)
        else:
            # Binary: [batch, d_transcoder * n_layers] -> [batch]
            logits = self.linear(x).squeeze(-1)
            if return_logits:
                return logits
            return torch.sigmoid(logits)

    def get_layer_weights(self, layer: int | None = None) -> torch.Tensor:
        """Get weight vector for the probe's layer.

        Args:
            layer: Layer index (must be in self.layers, or None to return full weight vector)

        Returns:
            Weight vector of shape [d_transcoder] if layer is specified, or [d_transcoder * n_layers] if None

        Raises:
            ValueError: If layer is not in the probe's layers
        """
        weights = self.linear.weight.squeeze(0)  # [d_transcoder * n_layers]

        if layer is None:
            return weights

        if layer not in self.layers:
            raise ValueError(f"Layer {layer} not found in this probe's layers: {self.layers}")

        idx = self.layers.index(layer)
        start_idx = idx * self.d_transcoder
        end_idx = (idx + 1) * self.d_transcoder

        return weights[start_idx:end_idx]

    def get_top_features(
        self,
        layer: int | None = None,
        k: int = 50,
        by: Literal["abs", "positive", "negative"] = "abs",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get top-k features by weight magnitude.

        Args:
            layer: Layer index (must match self.layer, or None)
            k: Number of top features to return
            by: Selection criterion:
                - 'abs': Largest absolute weight
                - 'positive': Largest positive weight
                - 'negative': Most negative weight

        Returns:
            Tuple of (feature_indices, weights) each of shape [k]

        Raises:
            ValueError: If layer doesn't match the probe's layer
        """
        weights = self.get_layer_weights(layer)

        if by == "abs":
            values, indices = torch.topk(weights.abs(), k)
            return indices, weights[indices]
        elif by == "positive":
            values, indices = torch.topk(weights, k)
            return indices, values
        elif by == "negative":
            values, indices = torch.topk(-weights, k)
            return indices, weights[indices]
        else:
            raise ValueError(f"Unknown selection criterion: {by}")

    def get_feature_rankings(
        self, layer: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get full feature ranking by absolute weight.

        Args:
            layer: Layer index (must match self.layer, or None)

        Returns:
            Tuple of (indices, weights, abs_weights) sorted by descending absolute weight
            Each tensor has shape [d_transcoder]
        """
        weights = self.get_layer_weights(layer)
        abs_weights = weights.abs()
        sorted_indices = torch.argsort(abs_weights, descending=True)

        return sorted_indices, weights[sorted_indices], abs_weights[sorted_indices]

    def to(self, *args, **kwargs):
        """Override to() to ensure all components are moved."""
        super().to(*args, **kwargs)
        return self

    def state_dict_with_metadata(self) -> dict:
        """Get state dict with probe configuration metadata.

        Returns:
            Dictionary containing both parameters and metadata
        """
        return {
            "state_dict": self.state_dict(),
            "layers": self.layers,
            "d_transcoder": self.d_transcoder,
            "max_seq_len": self.max_seq_len,
            "n_classes": self.n_classes,
        }

    @classmethod
    def from_state_dict_with_metadata(
        cls, checkpoint: dict, device: torch.device | None = None
    ) -> CarryProbe:
        """Load probe from state dict with metadata.

        Args:
            checkpoint: Dictionary from state_dict_with_metadata()
            device: Device to load parameters on

        Returns:
            Loaded CarryProbe instance
        """
        probe = cls(
            layers=checkpoint["layers"],
            d_transcoder=checkpoint["d_transcoder"],
            max_seq_len=checkpoint["max_seq_len"],
            n_classes=checkpoint.get("n_classes", 1),
            device=device,
        )
        probe.load_state_dict(checkpoint["state_dict"])
        return probe
