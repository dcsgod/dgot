"""
dgot.extractor
~~~~~~~~~~~~~~
Path Extractor.

After GNN evaluation and optimization, the extractor identifies the most
salient reasoning trajectory through the ThoughtGraph and compiles it into
a human-readable chain of reasoning.  It also asks the LLM to synthesize
a final answer conditioned on the extracted path.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .client import LLMClient
from .graph import ThoughtGraph, ThoughtNode


# ─────────────────────────────────────────────────────────────────────────────
# Path extraction
# ─────────────────────────────────────────────────────────────────────────────

def _best_path(graph: ThoughtGraph) -> List[ThoughtNode]:
    """
    Find the highest-scoring path through the graph using a greedy
    best-first traversal weighted by node.score × edge.weight.

    Returns an ordered list of ThoughtNodes (the reasoning chain).
    """
    if not graph.nodes:
        return []

    n = len(graph.nodes)

    # ── find root nodes (no incoming edges) ──────────────────────────
    has_incoming = {e.dst for e in graph.edges}
    roots = [nd for nd in graph.nodes if nd.id not in has_incoming]
    if not roots:
        # Fallback: highest-score node
        roots = [max(graph.nodes, key=lambda nd: nd.score)]

    # ── greedy traversal from best root ──────────────────────────────
    start = max(roots, key=lambda nd: nd.score)
    path = [start]
    visited = {start.id}

    current = start
    while True:
        out_edges = [e for e in graph.neighbors(current.id) if e.dst not in visited]
        if not out_edges:
            break
        # Score each candidate: node_score × edge_weight
        best_edge = max(
            out_edges,
            key=lambda e: graph.get_node(e.dst).score * e.weight,
        )
        next_node = graph.get_node(best_edge.dst)
        path.append(next_node)
        visited.add(next_node.id)
        current = next_node

    return path


# ─────────────────────────────────────────────────────────────────────────────
# Synthesizer system prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYNTH_SYSTEM = """\
You are a reasoning synthesizer.  You will receive:
1. The original user question / task.
2. A numbered list of reasoning steps (the extracted thought path).

Your job:
- Walk through the reasoning steps carefully.
- Produce a clear, well-structured final answer.
- Be concise but complete.
- Do NOT say "Based on the reasoning steps…" — just answer directly.
"""


# ─────────────────────────────────────────────────────────────────────────────
# PathExtractor
# ─────────────────────────────────────────────────────────────────────────────

class PathExtractor:
    """
    Extracts the salient reasoning path and synthesizes a final answer.

    Parameters
    ----------
    client      : LLMClient used for answer synthesis.
    temperature : Sampling temperature for synthesis (default 0.3).
    """

    def __init__(self, client: LLMClient, temperature: float = 0.3):
        self.client = client
        self.temperature = temperature

    # ── public ───────────────────────────────────────────────────────

    def extract_path(self, graph: ThoughtGraph) -> List[ThoughtNode]:
        """Return the ordered list of most salient ThoughtNodes."""
        return _best_path(graph)

    def synthesize(
        self,
        original_prompt: str,
        path: List[ThoughtNode],
        max_tokens: int = 1024,
    ) -> str:
        """
        Ask the LLM to produce a final answer given the reasoning path.
        """
        steps_text = "\n".join(
            f"{i+1}. [{node.metadata.get('type', 'inference')}] {node.text}"
            for i, node in enumerate(path)
        )
        user_msg = (
            f"Original question:\n{original_prompt}\n\n"
            f"Reasoning steps:\n{steps_text}\n\n"
            f"Please provide the final answer."
        )
        return self.client.complete(
            prompt=user_msg,
            system=_SYNTH_SYSTEM,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )

    def format_path(self, path: List[ThoughtNode]) -> str:
        """Human-readable string representation of the extracted path."""
        lines = ["=== Extracted Reasoning Path ==="]
        for i, node in enumerate(path):
            node_type = node.metadata.get("type", "?")
            score_bar = "█" * int(node.score * 10) + "░" * (10 - int(node.score * 10))
            lines.append(
                f"Step {i+1:02d} [{node_type:12s}] score={node.score:.3f} [{score_bar}]\n"
                f"         {node.text}"
            )
            if i < len(path) - 1:
                lines.append("         ↓")
        lines.append("=================================")
        return "\n".join(lines)