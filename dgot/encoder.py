"""
dgot.encoder
~~~~~~~~~~~~
Graph Encoder.

Converts the text of each ThoughtNode into a dense embedding vector.
Also computes initial edge weights based on semantic similarity.
"""

from __future__ import annotations

import math
from typing import List

from .client import LLMClient
from .graph import ThoughtGraph


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _softmax(values: List[float]) -> List[float]:
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    s = sum(exps)
    return [e / s for e in exps]


class GraphEncoder:
    """
    Encodes every ThoughtNode's text into an embedding and sets initial
    edge weights based on semantic similarity between connected nodes.

    Parameters
    ----------
    client          : LLMClient (used for its .embed() method).
    similarity_init : If True, initialise edge weights from cosine similarity
                      between src/dst embeddings rather than uniform 1.0.
    batch_size      : How many texts to embed in one API call.
    """

    def __init__(
        self,
        client: LLMClient,
        similarity_init: bool = True,
        batch_size: int = 20,
    ):
        self.client = client
        self.similarity_init = similarity_init
        self.batch_size = batch_size

    def encode(self, graph: ThoughtGraph) -> ThoughtGraph:
        """
        In-place: sets .embedding on every node and .weight on every edge.
        Returns the same graph for chaining.
        """
        if not graph.nodes:
            return graph

        # ── batch-embed all node texts ────────────────────────────────
        texts = [n.text for n in graph.nodes]
        embeddings = self._batch_embed(texts)

        for node, emb in zip(graph.nodes, embeddings):
            node.embedding = emb

        # ── set edge weights ──────────────────────────────────────────
        if self.similarity_init:
            for edge in graph.edges:
                src_emb = graph.get_node(edge.src).embedding or []
                dst_emb = graph.get_node(edge.dst).embedding or []
                if src_emb and dst_emb:
                    sim = _cosine(src_emb, dst_emb)
                    # Scale to (0.1, 1.0) so no edge is fully zeroed out
                    edge.weight = 0.1 + 0.9 * ((sim + 1.0) / 2.0)
                else:
                    edge.weight = 1.0

        return graph

    # ── internals ────────────────────────────────────────────────────

    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            results.extend(self.client.embed(batch))
        return results