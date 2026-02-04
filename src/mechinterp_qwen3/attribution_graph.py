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
        # Update total attribution for nodes
        edge.source.total_attribution += abs(edge.attribution_score)
        edge.target.total_attribution += abs(edge.attribution_score)

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
        Prune the graph by removing low-attribution nodes and edges.

        Args:
            node_threshold: Keep minimum nodes with cumulative attribution >= threshold
            edge_threshold: Keep minimum edges with cumulative attribution >= threshold

        Returns:
            Pruned attribution graph
        """
        # Sort nodes by total attribution (descending)
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.total_attribution, reverse=True)

        # Calculate cumulative attribution
        total_attr = sum(n.total_attribution for n in sorted_nodes)

        # Handle case where no attributions exist
        if total_attr == 0:
            # Keep all nodes if there are no attributions
            nodes_to_keep = set(n.node_id for n in sorted_nodes)
        else:
            cumulative = 0.0
            nodes_to_keep = set()

            for node in sorted_nodes:
                cumulative += node.total_attribution
                nodes_to_keep.add(node.node_id)
                if cumulative / total_attr >= node_threshold:
                    break

        # Sort edges by attribution score (descending)
        sorted_edges = sorted(self.edges, key=lambda e: abs(e.attribution_score), reverse=True)

        # Calculate cumulative edge attribution
        total_edge_attr = sum(abs(e.attribution_score) for e in sorted_edges)

        # Handle case where no edge attributions exist
        if total_edge_attr == 0:
            edges_to_keep = []
        else:
            cumulative = 0.0
            edges_to_keep = []

            for edge in sorted_edges:
                # Only keep edges where both nodes are kept
                if edge.source.node_id in nodes_to_keep and edge.target.node_id in nodes_to_keep:
                    cumulative += abs(edge.attribution_score)
                    edges_to_keep.append(edge)
                    if cumulative / total_edge_attr >= edge_threshold:
                        break

        # Create pruned graph
        pruned = AttributionGraph()
        for node_id in nodes_to_keep:
            pruned.add_node(self.nodes[node_id])
        for edge in edges_to_keep:
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
