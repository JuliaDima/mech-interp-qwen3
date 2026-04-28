from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
from pydantic import BaseModel

from .utils.model_utils import get_default_device


@dataclass
class UnifiedConfig:
    """Config container compatible with both TransformerLens and NNsight naming conventions."""

    n_layers: int
    d_model: int
    d_head: int
    n_heads: int
    d_mlp: int
    d_vocab: int

    tokenizer_name: str
    model_name: str
    original_architecture: str

    n_key_value_heads: int | None = None
    dtype: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "UnifiedConfig":
        """Create from dictionary."""
        return cls(
            n_layers=config_dict["n_layers"],
            d_model=config_dict["d_model"],
            d_head=config_dict["d_head"],
            n_heads=config_dict["n_heads"],
            d_mlp=config_dict["d_mlp"],
            d_vocab=config_dict["d_vocab"],
            tokenizer_name=config_dict["tokenizer_name"],
            model_name=config_dict["model_name"],
            original_architecture=config_dict["original_architecture"],
            n_key_value_heads=config_dict.get("n_key_value_heads"),
            dtype=config_dict.get("dtype"),
        )


def standardize_config(config) -> UnifiedConfig:
    """Normalize any config object into a UnifiedConfig."""

    # Pass-through for already-normalized configs (e.g. loaded from a .pt file)
    if isinstance(config, UnifiedConfig):
        return config

    # Dict path — construct directly
    if isinstance(config, dict):
        return UnifiedConfig.from_dict(config)

    # Object path — convert to dict first, then remap any NNsight-style keys
    config_dict = config.to_dict()

    field_mappings = {
        "num_hidden_layers": "n_layers",
        "hidden_size": "d_model",
        "head_dim": "d_head",
        "num_attention_heads": "n_heads",
        "intermediate_size": "d_mlp",
        "vocab_size": "d_vocab",
        "num_key_value_heads": "n_key_value_heads",
        "torch_dtype": "dtype",
    }

    for nnsight_field, tl_field in field_mappings.items():
        if tl_field not in config_dict and nnsight_field in config_dict:
            config_dict[tl_field] = config_dict[nnsight_field]

    # Fill in any missing metadata fields
    if "original_architecture" not in config_dict:
        architectures = config_dict.get("architectures", [])
        config_dict["original_architecture"] = architectures[0] if architectures else "Unknown"

    if "tokenizer_name" not in config_dict:
        config_dict["tokenizer_name"] = config_dict.get("name_or_path", "Unknown")

    if "model_name" not in config_dict:
        config_dict["model_name"] = config_dict.get("name_or_path", "Unknown")

    return UnifiedConfig.from_dict(config_dict)


class Metadata(BaseModel):
    slug: str
    scan: str
    transcoder_list: list[str]
    prompt_tokens: list[str]
    prompt: str
    node_threshold: float | None = None
    schema_version: int | None = 1


class QParams(BaseModel):
    pinnedIds: list[str]
    supernodes: list[list[str]]
    linkType: str
    clickedId: str
    sg_pos: str


class Node(BaseModel):
    node_id: str
    feature: int
    layer: str
    ctx_idx: int
    feature_type: str
    token_prob: float = 0.0
    is_target_logit: bool = False
    run_idx: int = 0
    reverse_ctx_idx: int = 0
    influence: float | None = None
    activation: float | None = None
    token_str: str | None = None

    @classmethod
    def feature_node(cls, layer, pos, feat_idx, influence=None, activation=None):
        """Create a feature node."""

        def cantor_pairing(x, y):
            return (x + y) * (x + y + 1) // 2 + y

        return cls(
            node_id=f"{layer}_{feat_idx}_{pos}",
            feature=cantor_pairing(layer, feat_idx),
            layer=str(layer),
            ctx_idx=pos,
            feature_type="CLT",
            influence=influence,
            activation=activation,
        )

    @classmethod
    def error_node(cls, layer, pos, influence=None):
        """Create an error node."""
        return cls(
            node_id=f"0_{layer}_{pos}",
            feature=-1,
            layer=str(layer),
            ctx_idx=pos,
            feature_type="mlp_error",
            influence=influence,
        )

    @classmethod
    def token_node(cls, pos, vocab_idx, influence=None):
        """Create a token node."""
        return cls(
            node_id=f"E_{vocab_idx}_{pos}",
            feature=pos,
            layer="E",
            ctx_idx=pos,
            feature_type="embedding",
            influence=influence,
        )

    @classmethod
    def logit_node(
        cls,
        pos,
        vocab_idx,
        token_str: str,
        num_layers,
        target_logit=False,
        token_prob=0.0,
    ):
        """Create a logit node."""
        layer = str(num_layers + 1)
        return cls(
            node_id=f"{layer}_{vocab_idx}_{pos}",
            feature=vocab_idx,
            layer=layer,
            ctx_idx=pos,
            feature_type="logit",
            token_prob=token_prob,
            is_target_logit=target_logit,
            activation=token_prob,
            token_str=token_str,
        )


class Link(BaseModel):
    source: str
    target: str
    weight: float


class Model(BaseModel):
    metadata: Metadata
    qParams: QParams
    nodes: list[Node]
    links: list[dict]


class Graph:
    input_string: str
    input_tokens: torch.Tensor
    logit_tokens: torch.Tensor
    active_features: torch.Tensor
    adjacency_matrix: torch.Tensor
    selected_features: torch.Tensor
    activation_values: torch.Tensor
    logit_probabilities: torch.Tensor
    cfg: UnifiedConfig
    scan: str | list[str] | None

    def __init__(
        self,
        input_string: str,
        input_tokens: torch.Tensor,
        active_features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        cfg,
        logit_tokens: torch.Tensor,
        logit_probabilities: torch.Tensor,
        selected_features: torch.Tensor,
        activation_values: torch.Tensor,
        scan: str | list[str] | None = None,
    ):
        """
        A graph object containing the adjacency matrix describing the direct effect of each
        node on each other. Nodes are either non-zero transcoder features, transcoder errors,
        tokens, or logits. They are stored in the order [active_features[0], ...,
        active_features[n-1], error[layer0][position0], error[layer0][position1], ...,
        error[layer l - 1][position t-1], tokens[0], ..., tokens[t-1], logits[top-1 logit],
        ..., logits[top-k logit]].

        Args:
            input_string (str): The input string attributed.
            input_tokens (List[str]): The input tokens attributed.
            active_features (torch.Tensor): A tensor of shape (n_active_features, 3)
                containing the indices (layer, pos, feature_idx) of the non-zero features
                of the model on the given input string.
            adjacency_matrix (torch.Tensor): The adjacency matrix. Organized as
                [active_features, error_nodes, embed_nodes, logit_nodes], where there are
                model.cfg.n_layers * len(input_tokens) error nodes, len(input_tokens) embed
                nodes, len(logit_tokens) logit nodes. The rows represent target nodes, while
                columns represent source nodes.
            cfg (HookedTransformerConfig): The cfg of the model.
            logit_tokens (List[str]): The logit tokens attributed from.
            logit_probabilities (torch.Tensor): The probabilities of each logit token, given
                the input string.
            scan (Union[str,List[str]] | None, optional): The identifier of the
                transcoders used in the graph. Without a scan, the graph cannot be uploaded
                (since we won't know what transcoders were used). Defaults to None
        """
        self.input_string = input_string
        self.adjacency_matrix = adjacency_matrix
        self.cfg = standardize_config(cfg)
        self.n_pos = len(input_tokens)
        self.active_features = active_features
        self.logit_tokens = logit_tokens
        self.logit_probabilities = logit_probabilities
        self.input_tokens = input_tokens
        if scan is None:
            print("No scan provided — graph cannot be uploaded without one.")
        self.scan = scan
        self.selected_features = selected_features
        self.activation_values = activation_values

    def to(self, device):
        """Send all relevant tensors to the device (cpu, cuda, etc.)

        Args:
            device (_type_): device to send tensors
        """
        self.adjacency_matrix = self.adjacency_matrix.to(device)
        self.active_features = self.active_features.to(device)
        self.logit_tokens = self.logit_tokens.to(device)
        self.logit_probabilities = self.logit_probabilities.to(device)

    def to_pt(self, path: str):
        """Saves the graph at the given path

        Args:
            path (str): The path where the graph will be saved. Should end in .pt
        """
        d = {
            "input_string": self.input_string,
            "adjacency_matrix": self.adjacency_matrix,
            "cfg": self.cfg,
            "active_features": self.active_features,
            "logit_tokens": self.logit_tokens,
            "logit_probabilities": self.logit_probabilities,
            "input_tokens": self.input_tokens,
            "selected_features": self.selected_features,
            "activation_values": self.activation_values,
            "scan": self.scan,
        }
        torch.save(d, path)

    @staticmethod
    def from_pt(path: str, map_location="cpu") -> "Graph":
        """Load a graph (saved using graph.to_pt) from a .pt file at the given path.

        Args:
            path (str): The path of the Graph to load
            map_location (str, optional): the device to load the graph onto.
                Defaults to 'cpu'.

        Returns:
            Graph: the Graph saved at the specified path
        """
        d = torch.load(path, weights_only=False, map_location=map_location)
        return Graph(**d)


def normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    abs_mat = matrix.abs()
    return abs_mat / abs_mat.sum(dim=1, keepdim=True).clamp(min=1e-10)


def compute_influence(A: torch.Tensor, logit_weights: torch.Tensor, max_iter: int = 1000):
    # Neumann series: logit_weights @ (A + A^2 + ...), accumulated via left-multiplication
    step = logit_weights @ A
    total = step.clone()
    for _ in range(max_iter):
        step = step @ A
        if not step.any():
            return total
        total += step
    raise RuntimeError(f"Influence did not converge within {max_iter} iterations")


def find_threshold(scores: torch.Tensor, threshold: float):
    desc = torch.sort(scores, descending=True).values
    cumfrac = torch.cumsum(desc, dim=0) / desc.sum()
    idx = min(int(torch.searchsorted(cumfrac, threshold).item()), len(cumfrac) - 1)
    return desc[idx]


class PruneResult(NamedTuple):
    node_mask: torch.Tensor  # Boolean tensor indicating which nodes to keep
    edge_mask: torch.Tensor  # Boolean tensor indicating which edges to keep
    cumulative_scores: torch.Tensor  # Tensor of cumulative influence scores for each node


def prune_graph(
    graph: Graph, node_threshold: float = 0.8, edge_threshold: float = 0.98
) -> PruneResult:
    """Remove low-influence nodes and edges from the attribution graph.

    Args:
        graph: The graph to prune
        node_threshold: Retain nodes that collectively account for this fraction of influence
        edge_threshold: Retain edges that collectively account for this fraction of influence

    Returns:
        PruneResult with:
        - node_mask: Boolean tensor indicating which nodes survive
        - edge_mask: Boolean tensor indicating which edges survive
        - cumulative_scores: Per-node cumulative influence fraction (for ranking)
    """

    if node_threshold > 1.0 or node_threshold < 0.0:
        raise ValueError("node_threshold must be between 0.0 and 1.0")
    if edge_threshold > 1.0 or edge_threshold < 0.0:
        raise ValueError("edge_threshold must be between 0.0 and 1.0")

    # Extract dimensions
    n_tokens = len(graph.input_tokens)
    n_logits = len(graph.logit_tokens)
    n_features = len(graph.selected_features)

    n_nodes = graph.adjacency_matrix.shape[0]
    logit_weights = torch.zeros(n_nodes, device=graph.adjacency_matrix.device)
    logit_weights[-n_logits:] = graph.logit_probabilities

    norm_adj = normalize_matrix(graph.adjacency_matrix)
    node_influence = compute_influence(norm_adj, logit_weights)
    node_mask = node_influence >= find_threshold(node_influence, node_threshold)
    # Tokens and logits are always retained regardless of influence score
    node_mask[-n_logits - n_tokens :] = True

    # Zero out rows/cols for pruned nodes before computing edge influence
    pruned_matrix = graph.adjacency_matrix.clone()
    pruned_matrix[~node_mask] = 0
    pruned_matrix[:, ~node_mask] = 0

    norm_pruned = normalize_matrix(pruned_matrix)
    all_node_influence = compute_influence(norm_pruned, logit_weights) + logit_weights
    edge_scores = norm_pruned * all_node_influence[:, None]

    edge_mask = edge_scores >= find_threshold(edge_scores.flatten(), edge_threshold)

    old_node_mask = node_mask.clone()
    # Ensure feature and error nodes have outgoing edges
    node_mask[: -n_logits - n_tokens] &= edge_mask[:, : -n_logits - n_tokens].any(0)
    # Ensure feature nodes have incoming edges
    node_mask[:n_features] &= edge_mask[:n_features].any(1)

    # Iterate until the mask stabilises — each round can expose new orphaned nodes.
    # Worst case is O(n_layers) iterations.
    while not torch.all(node_mask == old_node_mask):
        old_node_mask[:] = node_mask
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False

        # Ensure feature and error nodes have outgoing edges
        node_mask[: -n_logits - n_tokens] &= edge_mask[:, : -n_logits - n_tokens].any(0)
        # Ensure feature nodes have incoming edges
        node_mask[:n_features] &= edge_mask[:n_features].any(1)

    sorted_indices = torch.argsort(node_influence, descending=True)
    sorted_vals = node_influence[sorted_indices]
    cumsum = torch.cumsum(sorted_vals, dim=0) / sorted_vals.sum()
    final_scores = torch.empty_like(node_influence)
    final_scores[sorted_indices] = cumsum

    return PruneResult(node_mask, edge_mask, final_scores)


def compute_graph_scores(graph: Graph) -> tuple[float, float]:
    """Score the graph's interpretability: how much computation runs through feature nodes.

    Two complementary metrics:

    - **Replacement score**: fraction of the token→logit influence path that runs through
      transcoder features rather than error nodes. Strict: full-path credit only.

    - **Completeness score**: for each node, the fraction of its incoming influence that
      comes from features or tokens (not errors), weighted by that node's output influence.
      Gives partial credit for nodes that are mostly explained by features.

    Args:
        graph: Attribution graph with features, errors, tokens, and logit nodes.

    Returns:
        (replacement_score, completeness_score), both in [0, 1]. Higher is more interpretable.
    """
    n_logits = len(graph.logit_tokens)
    n_tokens = len(graph.input_tokens)
    n_features = len(graph.selected_features)
    error_start = n_features
    error_end = error_start + n_tokens * graph.cfg.n_layers  # type: ignore
    token_end = error_end + n_tokens

    logit_weights = torch.zeros(
        graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device
    )
    logit_weights[-n_logits:] = graph.logit_probabilities

    norm_adj = normalize_matrix(graph.adjacency_matrix)
    node_influence = compute_influence(norm_adj, logit_weights)
    token_influence = node_influence[error_end:token_end].sum()
    error_influence = node_influence[error_start:error_end].sum()

    replacement_score = token_influence / (token_influence + error_influence)

    non_error_fractions = 1 - norm_adj[:, error_start:error_end].sum(dim=-1)
    output_influence = node_influence + logit_weights
    completeness_score = (non_error_fractions * output_influence).sum() / output_influence.sum()

    return replacement_score.item(), completeness_score.item()


def compute_partial_influences(
    edge_matrix: torch.Tensor,
    logit_p: torch.Tensor,
    row_to_node_index: torch.Tensor,
    max_iter: int = 128,
    device=None,
):
    """Estimate node influence scores via truncated power iteration."""
    device = device or get_default_device()

    normalized_matrix = torch.empty_like(edge_matrix, device=device).copy_(edge_matrix)
    normalized_matrix = normalized_matrix.abs_()
    normalized_matrix /= normalized_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)

    influences = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod[-len(logit_p) :] = logit_p

    for _ in range(max_iter):
        prod = prod[row_to_node_index] @ normalized_matrix
        if not prod.any():
            break
        influences += prod
    else:
        raise RuntimeError("Failed to converge")

    return influences
