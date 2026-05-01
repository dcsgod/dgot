"""
dgot.graph
~~~~~~~~~~
Core ThoughtGraph data structure used throughout the D-GoT pipeline.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import math


@dataclass
class ThoughtNode:
    """A single reasoning unit in the thought graph."""
    id: int
    text: str
    embedding: Optional[List[float]] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return f"ThoughtNode(id={self.id}, score={self.score:.3f}, text='{preview}')"


@dataclass
class ThoughtEdge:
    """A directed logical dependency between two thought nodes."""
    src: int
    dst: int
    weight: float = 1.0
    relation: str = "supports"

    def __repr__(self):
        return f"ThoughtEdge({self.src} --[{self.relation}, w={self.weight:.3f}]--> {self.dst})"


class ThoughtGraph:
    """
    A directed graph of reasoning steps.

    Nodes  = semantic reasoning units (thoughts)
    Edges  = logical dependencies / transitions between thoughts
    """

    def __init__(self):
        self.nodes: List[ThoughtNode] = []
        self.edges: List[ThoughtEdge] = []
        self._node_index: Dict[int, ThoughtNode] = {}

    # ------------------------------------------------------------------ #
    # Construction helpers                                                 #
    # ------------------------------------------------------------------ #

    def add_node(self, text: str, embedding: Optional[List[float]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> ThoughtNode:
        node_id = len(self.nodes)
        node = ThoughtNode(
            id=node_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
        )
        self.nodes.append(node)
        self._node_index[node_id] = node
        return node

    def add_edge(self, src: int, dst: int,
                 weight: float = 1.0, relation: str = "supports") -> ThoughtEdge:
        edge = ThoughtEdge(src=src, dst=dst, weight=weight, relation=relation)
        self.edges.append(edge)
        return edge

    def get_node(self, node_id: int) -> ThoughtNode:
        return self._node_index[node_id]

    # ------------------------------------------------------------------ #
    # Adjacency helpers                                                    #
    # ------------------------------------------------------------------ #

    def adjacency_matrix(self) -> List[List[float]]:
        """Return a dense N×N adjacency matrix of edge weights."""
        n = len(self.nodes)
        mat = [[0.0] * n for _ in range(n)]
        for e in self.edges:
            mat[e.src][e.dst] = e.weight
        return mat

    def neighbors(self, node_id: int) -> List[ThoughtEdge]:
        return [e for e in self.edges if e.src == node_id]

    def in_edges(self, node_id: int) -> List[ThoughtEdge]:
        return [e for e in self.edges if e.dst == node_id]

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {
                    "id": n.id,
                    "text": n.text,
                    "score": n.score,
                    "metadata": n.metadata,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "src": e.src,
                    "dst": e.dst,
                    "weight": e.weight,
                    "relation": e.relation,
                }
                for e in self.edges
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThoughtGraph":
        g = cls()
        for nd in data["nodes"]:
            node = ThoughtNode(
                id=nd["id"],
                text=nd["text"],
                score=nd.get("score", 0.0),
                metadata=nd.get("metadata", {}),
            )
            g.nodes.append(node)
            g._node_index[node.id] = node
        for ed in data["edges"]:
            g.edges.append(ThoughtEdge(
                src=ed["src"],
                dst=ed["dst"],
                weight=ed.get("weight", 1.0),
                relation=ed.get("relation", "supports"),
            ))
        return g

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        return f"ThoughtGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"