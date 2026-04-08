"""Dataset utilities for extracting transcoder activations online during training."""

from __future__ import annotations

from typing import Literal

import torch
from tqdm import tqdm

from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input


def extract_activations_online(
    model,
    prompts: list[str],
    layers: list[int],
    token_position: int | Literal["final", "answer", "all"] = "all",
    batch_size: int = 32,
    sparse: bool = False,
    show_progress: bool = True,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Extract transcoder activations from prompts online (no precomputation).

    This function runs the model with transcoder hooks to extract activations
    on-the-fly during training. Activations are NOT stored as giant concatenated
    vectors but retrieved per-batch.

    Args:
        model: AttributionModel with transcoders loaded
        prompts: List of text prompts to run through the model
        layers: List of layer indices to extract activations from
        token_position: Which token position to use:
            - 'final': Last token position
            - 'answer': Token after the last '=' (addition-specific)
            - 'all': Extract full sequence for all tokens
            - int: Explicit token index
        batch_size: Batch size for processing
        sparse: Whether to return sparse tensors
        show_progress: Whether to show progress bar

    Returns:
        Tuple of (activations_dict, logits):
            - activations_dict: {layer: [n_samples, d_transcoder]}
            - logits: [n_samples, seq_len, vocab_size]

    Raises:
        ValueError: If invalid layers or token_position specified
    """
    # Validate layers
    n_layers = model.cfg.n_layers
    for layer in layers:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Layer {layer} out of range [0, {n_layers})")

    # Initialize storage
    activations_by_layer = {layer: [] for layer in layers}
    all_logits = []

    # Process in batches
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    iterator = range(n_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting activations")

    for batch_idx in iterator:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]

        # Process batch
        with torch.no_grad():
            # Tokenize
            batch_tokens = [
                tokenize_qwen_input(p, model.tokenizer, model.cfg.device) for p in batch_prompts
            ]

            # Get activations for each prompt individually
            # (batching tokenized sequences requires padding which complicates position selection)
            for tokens in batch_tokens:
                logits, activation_cache = model.get_activations(
                    tokens.unsqueeze(0),  # Add batch dimension
                    sparse=sparse,
                    apply_activation_function=True,
                )

                # activation_cache shape: [n_layers, 1, seq_len, d_transcoder]
                # logits shape: [1, seq_len, vocab_size]

                # Determine token position
                seq_len = tokens.shape[0]
                if token_position == "final":
                    pos = seq_len - 1
                elif token_position == "all":
                    pos = slice(None)
                elif token_position == "answer":
                    # Find last '=' and use next token
                    # This is specific to "calc: a+b= " format
                    token_strs = model.tokenizer.convert_ids_to_tokens(tokens.tolist())
                    try:
                        eq_idx = len(token_strs) - 1 - token_strs[::-1].index("=")
                        pos = min(eq_idx + 1, seq_len - 1)
                    except ValueError:
                        # No '=' found, use final token
                        pos = seq_len - 1
                elif isinstance(token_position, int):
                    if token_position < 0 or token_position >= seq_len:
                        raise ValueError(
                            f"Token position {token_position} out of range for sequence length {seq_len}"
                        )
                    pos = token_position
                else:
                    raise ValueError(f"Unknown token_position: {token_position}")

                # Extract activations at the target position for requested layers
                for layer in layers:
                    # activation_cache[layer] shape: [seq_len, d_transcoder]
                    act = activation_cache[
                        layer, pos, :
                    ]  # [d_transcoder] or [seq_len, d_transcoder]
                    activations_by_layer[layer].append((act.cpu(), seq_len))

                all_logits.append(logits[0].cpu())  # Remove batch dimension

    # Stack or pad activations
    activations_dict = {}
    if token_position == "all":
        import torch.nn.functional as F

        # Determine maximum sequence length across all prompts
        # For our generated additions, they should be fairly short (e.g. 5-15 tokens)
        max_seq_len = max(seq_len for _, seq_len in activations_by_layer[layers[0]])

        # Pad shorter sequences with zeros
        for layer in layers:
            padded_acts = []
            for act, seq_len in activations_by_layer[layer]:
                if seq_len < max_seq_len:
                    # Pad the temporal dimension (dim=0 for [seq_len, d_transcoder])
                    padding = (0, 0, 0, max_seq_len - seq_len)
                    act = F.pad(act, padding, value=0.0)
                padded_acts.append(act)
            activations_dict[layer] = torch.stack(
                padded_acts
            )  # [n_samples, max_seq_len, d_transcoder]
    else:
        for layer in layers:
            activations_dict[layer] = torch.stack(
                [act for act, _ in activations_by_layer[layer]]
            )  # [n_samples, d_transcoder]

    # Stack logits
    # Note: logits have variable sequence lengths, so we store them as a list
    # For now, just return them as-is
    logits_tensor = all_logits  # List of [seq_len, vocab_size]

    return activations_dict, logits_tensor


def pool_activations(
    activations: torch.Tensor,
    strategy: Literal["mean", "max", "final"] = "final",
    positions: list[int] | None = None,
) -> torch.Tensor:
    """Pool activations over token positions.

    Args:
        activations: [batch, seq_len, d_transcoder] or [batch, d_transcoder]
        strategy: Pooling strategy:
            - 'mean': Average over positions
            - 'max': Max over positions
            - 'final': Use final position
        positions: Explicit list of positions to pool over (overrides strategy)

    Returns:
        Pooled activations [batch, d_transcoder]
    """
    if activations.ndim == 2:
        # Already pooled
        return activations

    if positions is not None:
        # Use explicit positions
        activations = activations[:, positions, :]

    if strategy == "mean":
        return activations.mean(dim=1)
    elif strategy == "max":
        return activations.max(dim=1)[0]
    elif strategy == "final":
        return activations[:, -1, :]
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")


class ProbeDataset:
    """Dataset for probe training that extracts activations online.

    This dataset does NOT store activations in memory. Instead, it stores
    prompts and labels, and extracts activations on-demand during training.
    """

    def __init__(
        self,
        prompts: list[str],
        labels: list[int],
        model,
        layers: list[int],
        token_position: int | Literal["final", "answer", "all"] = "all",
        cache_activations: bool = False,
    ):
        """Initialize probe dataset.

        Args:
            prompts: List of text prompts
            labels: List of binary labels (0 or 1)
            model: AttributionModel with transcoders
            layers: List of layer indices to extract activations from
            token_position: Which token position to use
            cache_activations: If True, extract and cache all activations upfront.
                Otherwise, extract on-demand during iteration.
        """
        if len(prompts) != len(labels):
            raise ValueError(f"Mismatched lengths: {len(prompts)} prompts vs {len(labels)} labels")

        self.prompts = prompts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.model = model
        self.layers = layers
        self.token_position = token_position
        self.cache_activations = cache_activations

        self._cached_activations = None

        if cache_activations:
            print("Caching activations...")
            self._cached_activations, _ = extract_activations_online(
                model=model,
                prompts=prompts,
                layers=layers,
                token_position=token_position,
                batch_size=32,
                sparse=False,
                show_progress=True,
            )
            print("Activations cached.")

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Get a single sample.

        Returns:
            Tuple of (activations_dict, label)
        """
        if self._cached_activations is not None:
            # Use cached activations
            activations = {layer: self._cached_activations[layer][idx] for layer in self.layers}
        else:
            # Extract on-demand (expensive!)
            prompt = self.prompts[idx]
            activations, _ = extract_activations_online(
                model=self.model,
                prompts=[prompt],
                layers=self.layers,
                token_position=self.token_position,
                batch_size=1,
                sparse=False,
                show_progress=False,
            )
            # Extract single sample
            activations = {layer: activations[layer][0] for layer in self.layers}

        label = self.labels[idx]
        return activations, label

    def get_batch(self, indices: list[int]) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Get a batch of samples efficiently.

        Args:
            indices: List of sample indices

        Returns:
            Tuple of (activations_dict, labels)
                - activations_dict: {layer: [batch, d_transcoder]}
                - labels: [batch]
        """
        if self._cached_activations is not None:
            # Use cached activations
            if self.token_position == "all":
                import torch.nn.functional as F

                # The cached sequences could have different lengths if max_seq_len was determined locally?
                # Actually during initialization caching, extract_activations_online already padded them
                # to the global max_seq_len across the entire training dataset.
                # So they should all be [max_seq_len, d_transcoder] already!
                activations = {
                    layer: torch.stack([self._cached_activations[layer][i] for i in indices])
                    for layer in self.layers
                }
            else:
                activations = {
                    layer: torch.stack([self._cached_activations[layer][i] for i in indices])
                    for layer in self.layers
                }
        else:
            # Extract batch on-demand
            batch_prompts = [self.prompts[i] for i in indices]
            activations, _ = extract_activations_online(
                model=self.model,
                prompts=batch_prompts,
                layers=self.layers,
                token_position=self.token_position,
                batch_size=len(batch_prompts),
                sparse=False,
                show_progress=False,
            )

            # Note: extract_activations_online pads the batch to local max_seq_len.
            # But the probe evaluates the whole dataset and expects a fixed max_seq_len (d_in).
            # We need to pad to self.max_seq_len if it's smaller, or truncate if it's larger.
            if self.token_position == "all" and hasattr(self, "max_seq_len"):
                import torch.nn.functional as F

                for layer in self.layers:
                    acts = activations[layer]  # [batch, seq_len, d_transcoder]
                    seq_len = acts.shape[1]
                    if seq_len < self.max_seq_len:
                        padding = (0, 0, 0, self.max_seq_len - seq_len)
                        activations[layer] = F.pad(acts, padding, value=0.0)
                    elif seq_len > self.max_seq_len:
                        activations[layer] = acts[:, : self.max_seq_len, :]

        labels = self.labels[indices]
        return activations, labels

    def set_max_seq_len(self, max_seq_len: int):
        """Set the global max sequence length for padding batches."""
        self.max_seq_len = max_seq_len
