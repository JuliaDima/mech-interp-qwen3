"""Attribution graph data structures.

Defines nodes and edges for representing attribution graphs from input tokens
through SAE features to output logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Node:
    """A node in the attribution graph."""

    node_id: str  # Unique identifier
    node_type: str  # "token", "feature", "logit"
    layer: int | None = None  # Layer number (for features)
    feature_id: int | None = None  # Feature ID within layer (for features)
    token_pos: int | None = None  # Token position (for tokens)
    token_str: str | None = None  # Token string (for tokens)
    logit_token_id: int | None = None  # Token ID (for logits)
    logit_token_str: str | None = None  # Token string (for logits)
    activation: float = 0.0  # Activation value
    total_attribution: float = 0.0  # Sum of incoming/outgoing attributions

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.node_id == other.node_id


@dataclass
class Edge:
    """An edge representing attribution between nodes."""

    source: Node
    target: Node
    attribution_score: float

    def __hash__(self) -> int:
        return hash((self.source.node_id, self.target.node_id))


class AttributionGraph:
    """Represents an attribution graph with nodes and edges."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}  # node_id -> Node
        self.edges: list[Edge] = []

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
        # Update total attribution for the source node (outgoing influence)
        # We only count outgoing attribution to avoid "relay inflation" in intermediate features.
        edge.source.total_attribution += abs(edge.attribution_score)

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_edges_from(self, node_id: str) -> list[Edge]:
        """Get all edges originating from a node."""
        return [e for e in self.edges if e.source.node_id == node_id]

    def get_edges_to(self, node_id: str) -> list[Edge]:
        """Get all edges targeting a node."""
        return [e for e in self.edges if e.target.node_id == node_id]

    def prune(self, node_threshold: float = 0.8, edge_threshold: float = 0.98) -> AttributionGraph:
        """

        1. Build a normalized adjacency matrix (rows=targets, cols=sources).
        2. Compute *transitive* node influence via power-iteration:
               node_influence = logit_weights @ A + logit_weights @ A² + ...
           This propagates logit attribution backwards through the full graph,
           giving every node credit for its indirect effect on the output.
        3. Keep nodes whose cumulative influence >= node_threshold of total.
        4. Always keep token and logit nodes.
        5. Compute edge scores on the pruned matrix via the same power-iteration,
           threshold by edge_threshold.
        6. Iteratively remove feature/error nodes that have no surviving
           incoming OR outgoing edges.

        Args:
            node_threshold: Keep nodes covering this fraction of total transitive influence.
            edge_threshold: Keep edges covering this fraction of total edge influence.

        Returns:
            Pruned AttributionGraph.
        """
        import torch

        node_ids = list(self.nodes.keys())
        n = len(node_ids)
        if n == 0:
            return AttributionGraph()

        idx = {nid: i for i, nid in enumerate(node_ids)}

        # Categorize node indices
        logit_indices = [
            i for i, nid in enumerate(node_ids) if self.nodes[nid].node_type == "logit"
        ]
        token_indices = [
            i for i, nid in enumerate(node_ids) if self.nodes[nid].node_type == "token"
        ]
        # error nodes = everything else except token/logit/feature
        fixed_indices = set(logit_indices + token_indices)  # always kept

        # Build raw adjacency matrix  A[target, source] = attribution_score
        A = torch.zeros(n, n)
        for edge in self.edges:
            src = idx.get(edge.source.node_id)
            tgt = idx.get(edge.target.node_id)
            if src is not None and tgt is not None:
                A[tgt, src] += edge.attribution_score

        def normalize(mat: torch.Tensor) -> torch.Tensor:
            """Row-normalize by abs-sum."""
            norms = mat.abs().sum(dim=1, keepdim=True).clamp(min=1e-10)
            return mat.abs() / norms

        def power_influence(
            mat: torch.Tensor, weights: torch.Tensor, max_iter: int = 1000
        ) -> torch.Tensor:
            """Compute logit_weights @ (A + A² + ...) until convergence."""
            current = weights @ mat
            influence = current.clone()
            for _ in range(max_iter):
                if not current.any():
                    break
                current = current @ mat
                influence += current
            return influence

        def find_threshold(scores: torch.Tensor, threshold: float) -> float:
            sorted_s = torch.sort(scores, descending=True).values
            cumsum = torch.cumsum(sorted_s, dim=0) / sorted_s.sum().clamp(min=1e-10)
            idx_t = int(torch.searchsorted(cumsum, threshold).item())
            idx_t = min(idx_t, len(sorted_s) - 1)
            return sorted_s[idx_t].item()

        # For node influence ranking, zero out token source columns.
        # Token nodes are always kept (fixed), but token→logit edges have
        # ~50x larger magnitudes than feature→logit edges, causing row-
        # normalization to wash out all feature influence. Token edges are
        # still used in the edge-scoring step after node selection.
        A_rank = A.clone()
        for ti in token_indices:
            A_rank[:, ti] = 0
        norm_A = normalize(A_rank)

        # Logit weights: logit nodes weighted by their activation (logit probability)
        logit_weights = torch.zeros(n)
        for i in logit_indices:
            logit_weights[i] = self.nodes[node_ids[i]].activation
        if logit_weights.sum() == 0:
            logit_weights[logit_indices] = 1.0 / max(len(logit_indices), 1)

        # 1. Node influence via power-iteration
        node_influence = power_influence(norm_A, logit_weights)

        # 2. Node threshold
        thresh = find_threshold(node_influence, node_threshold)
        node_mask = node_influence >= thresh

        # 3. Always keep tokens and logits
        for i in fixed_indices:
            node_mask[i] = True

        # 4. Edge influence on pruned matrix.
        # Use A_rank (token columns zeroed) consistently to prevent token->logit
        # edges (50x larger magnitudes) from consuming the entire edge budget and
        # crowding out feature->logit edges. Token edges are added back to the
        # final pruned graph because token/logit nodes are always kept.
        pruned_A_rank = A_rank.clone()
        pruned_A_rank[~node_mask] = 0
        pruned_A_rank[:, ~node_mask] = 0

        norm_pruned = normalize(pruned_A_rank)
        edge_influence = power_influence(norm_pruned, logit_weights)
        edge_influence_full = logit_weights.clone()
        edge_influence_full += edge_influence
        edge_scores = norm_pruned * edge_influence_full.unsqueeze(1)

        edge_thresh = find_threshold(edge_scores.flatten(), edge_threshold)
        edge_mask = edge_scores >= edge_thresh

        # 5. Iterative cleanup: drop non-fixed nodes with no surviving outgoing edges.
        # We only require HAS_OUT (node sends to some surviving downstream node).
        # HAS_IN is NOT required because feature nodes in our graph have no upstream
        # (there are no token->feature edges; features are SAE outputs that directly
        # connect to logits). Requiring has_in would eliminate all features.
        old_mask = node_mask.clone()
        while True:
            has_out = edge_mask.any(dim=0)  # col-wise: does source have any outgoing edge?
            for i in range(n):
                if i not in fixed_indices and not has_out[i]:
                    node_mask[i] = False
            if torch.all(node_mask == old_mask):
                break
            old_mask = node_mask.clone()
            edge_mask[~node_mask] = False
            edge_mask[:, ~node_mask] = False

        # 6. Build pruned graph
        kept_ids = {node_ids[i] for i in range(n) if node_mask[i]}
        pruned = AttributionGraph()
        for nid in kept_ids:
            pruned.add_node(self.nodes[nid])

        for edge in self.edges:
            si = idx.get(edge.source.node_id)
            ti = idx.get(edge.target.node_id)
            if (
                si is not None
                and ti is not None
                and node_mask[si]
                and node_mask[ti]
                and edge_mask[ti, si]
            ):
                pruned.add_edge(edge)

        return pruned

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    "layer": n.layer,
                    "feature_id": n.feature_id,
                    "token_pos": n.token_pos,
                    "token_str": n.token_str,
                    "logit_token_id": n.logit_token_id,
                    "logit_token_str": n.logit_token_str,
                    "activation": n.activation,
                    "total_attribution": n.total_attribution,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source.node_id,
                    "target": e.target.node_id,
                    "attribution_score": e.attribution_score,
                }
                for e in self.edges
            ],
        }

    def __repr__(self) -> str:
        return f"AttributionGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"
