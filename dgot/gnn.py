"""
dgot.gnn
~~~~~~~~
Graph Neural Network Evaluator.

Implements differentiable reasoning via attention-based message passing.
This is a pure-Python implementation (no PyTorch / TensorFlow required)
so the package stays lightweight.  It uses simple matrix operations and
softmax attention — enough to produce useful node scores and edge weights.

For research use cases, users can subclass GNNEvaluator and swap in a
PyTorch GNN when full backprop is needed.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .graph import ThoughtGraph


# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python linear algebra helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v)) or 1e-9


def _normalise(v: List[float]) -> List[float]:
    n = _norm(v)
    return [x / n for x in v]


def _softmax(values: List[float]) -> List[float]:
    m = max(values) if values else 0.0
    exps = [math.exp(v - m) for v in values]
    s = sum(exps) or 1e-9
    return [e / s for e in exps]


def _add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def _scale(v: List[float], s: float) -> List[float]:
    return [x * s for x in v]


def _relu(v: List[float]) -> List[float]:
    return [max(0.0, x) for x in v]


def _mean_pool(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    d = len(vectors[0])
    result = [0.0] * d
    for v in vectors:
        for i, x in enumerate(v):
            result[i] += x
    n = len(vectors)
    return [x / n for x in result]


# ─────────────────────────────────────────────────────────────────────────────
# Attention-based message passing
# ─────────────────────────────────────────────────────────────────────────────

class GNNEvaluator:
    """
    Differentiable graph reasoner.

    Each message-passing layer:
      1. Computes attention weights between each node and its in-neighbours
         using scaled dot-product attention on the embeddings.
      2. Aggregates neighbour messages weighted by attention × edge_weight.
      3. Updates the node representation with a residual connection.

    After *layers* passes, each node gets a scalar relevance score
    derived from its final embedding norm (proxy for information content).

    Parameters
    ----------
    layers          : Number of message-passing rounds (default 2).
    attention_temp  : Temperature for attention softmax (lower = sharper).
    residual        : If True, add the original embedding at every layer.
    """

    def __init__(
        self,
        layers: int = 2,
        attention_temp: float = 1.0,
        residual: bool = True,
    ):
        self.layers = layers
        self.attention_temp = attention_temp
        self.residual = residual

    def evaluate(self, graph: ThoughtGraph) -> ThoughtGraph:
        """
        Run message passing on *graph* (in-place).
        Sets .score on every node and updates .weight on every edge.
        Returns the same graph.
        """
        if not graph.nodes:
            return graph

        # Work with mutable copies of embeddings
        h = [list(n.embedding or []) for n in graph.nodes]
        if not h[0]:          # No embeddings available → assign uniform scores
            for node in graph.nodes:
                node.score = 1.0 / len(graph.nodes)
            return graph

        # ── message passing ───────────────────────────────────────────
        for _ in range(self.layers):
            h_new = []
            for node in graph.nodes:
                in_edges = graph.in_edges(node.id)
                if not in_edges:
                    h_new.append(h[node.id])
                    continue

                # Attention scores: query = current node, keys = neighbours
                query = h[node.id]
                keys  = [h[e.src] for e in in_edges]
                raw_attn = [
                    _dot(query, k) / (math.sqrt(len(query)) * self.attention_temp)
                    for k in keys
                ]
                attn_weights = _softmax(raw_attn)

                # Weighted message aggregation (attention × edge weight)
                d = len(query)
                agg = [0.0] * d
                for ew, attn, edge in zip(keys, attn_weights, in_edges):
                    combined_w = attn * edge.weight
                    msg = _scale(ew, combined_w)
                    agg = _add(agg, msg)

                # Residual + ReLU
                if self.residual:
                    updated = _relu(_add(agg, h[node.id]))
                else:
                    updated = _relu(agg)

                h_new.append(updated)

            h = h_new

        # ── update node embeddings and scores ─────────────────────────
        for node, emb in zip(graph.nodes, h):
            node.embedding = emb
            node.score = _norm(emb)          # relevance proxy

        # ── normalise scores to [0, 1] ────────────────────────────────
        max_s = max(n.score for n in graph.nodes) or 1.0
        for node in graph.nodes:
            node.score /= max_s

        # ── recompute edge weights from updated embeddings ────────────
        self._update_edge_weights(graph, h)

        return graph

    # ── helpers ──────────────────────────────────────────────────────

    def _update_edge_weights(
        self, graph: ThoughtGraph, h: List[List[float]]
    ) -> None:
        """Reweight edges using cosine similarity of updated embeddings."""
        for edge in graph.edges:
            src_emb = h[edge.src]
            dst_emb = h[edge.dst]
            if src_emb and dst_emb:
                cos = _dot(_normalise(src_emb), _normalise(dst_emb))
                edge.weight = 0.1 + 0.9 * ((cos + 1.0) / 2.0)

    def graph_embedding(self, graph: ThoughtGraph) -> List[float]:
        """Return a single vector summarising the whole graph (mean pool)."""
        return _mean_pool([n.embedding for n in graph.nodes if n.embedding])