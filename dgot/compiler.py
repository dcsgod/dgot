"""
dgot.compiler
~~~~~~~~~~~~~
Prompt-to-Graph Compiler.

Takes a natural-language user prompt and asks the LLM to decompose it into
a structured reasoning graph (nodes + edges), which becomes the initialisation
of the latent ThoughtGraph for the D-GoT pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .client import LLMClient
from .graph import ThoughtGraph


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_COMPILER_SYSTEM = """\
You are a reasoning graph compiler.  Given a user's question or task, you must
decompose it into a structured thought graph suitable for step-by-step reasoning.

Return ONLY a valid JSON object with the following schema (no markdown, no prose):

{
  "nodes": [
    {"id": 0, "text": "<one concise reasoning step>", "type": "premise|inference|conclusion|sub-question"},
    ...
  ],
  "edges": [
    {"src": 0, "dst": 1, "relation": "supports|refines|contradicts|leads_to|sub-problem_of"},
    ...
  ],
  "reasoning_goal": "<what the graph is ultimately trying to establish>"
}

Rules:
- Between 4 and 12 nodes.
- Every node must be reachable from at least one other node or be the starting premise.
- The final conclusion node should have no outgoing edges.
- Prefer informative relation labels over generic ones.
- Keep each node text to one sentence.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Compiler
# ─────────────────────────────────────────────────────────────────────────────

class PromptCompiler:
    """
    Compiles a natural-language prompt into an initial ThoughtGraph.

    Parameters
    ----------
    client      : LLMClient instance used to call the LLM.
    temperature : Sampling temperature for graph generation (default 0.4).
    retries     : Number of parse retries on malformed JSON (default 2).
    """

    def __init__(
        self,
        client: LLMClient,
        temperature: float = 0.4,
        retries: int = 2,
    ):
        self.client = client
        self.temperature = temperature
        self.retries = retries

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def compile(self, prompt: str) -> ThoughtGraph:
        """
        Decompose *prompt* into a ThoughtGraph.

        Returns
        -------
        ThoughtGraph
            Initialised with nodes and edges; embeddings are not yet set
            (the Encoder fills those in).
        """
        raw = self._call_llm(prompt)
        data = self._parse(raw)
        return self._build_graph(data)

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _call_llm(self, prompt: str) -> str:
        user_msg = f"Decompose the following into a reasoning graph:\n\n{prompt}"
        for attempt in range(self.retries + 1):
            try:
                return self.client.complete(
                    prompt=user_msg,
                    system=_COMPILER_SYSTEM,
                    temperature=self.temperature,
                    max_tokens=1500,
                )
            except Exception as exc:
                if attempt == self.retries:
                    raise RuntimeError(
                        f"PromptCompiler LLM call failed after {self.retries+1} attempts: {exc}"
                    ) from exc

    def _parse(self, raw: str) -> Dict[str, Any]:
        """Extract and parse JSON from the LLM response."""
        # Strip possible markdown fences
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Try to find the outermost {...}
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError(
            f"PromptCompiler: could not parse JSON from LLM response:\n{raw[:500]}"
        )

    def _build_graph(self, data: Dict[str, Any]) -> ThoughtGraph:
        g = ThoughtGraph()
        # Map original ids → graph node indices (they may not be 0-based)
        id_map: Dict[int, int] = {}

        for nd in data.get("nodes", []):
            node = g.add_node(
                text=nd.get("text", ""),
                metadata={
                    "type": nd.get("type", "inference"),
                    "original_id": nd.get("id"),
                },
            )
            id_map[nd["id"]] = node.id

        for ed in data.get("edges", []):
            src = id_map.get(ed.get("src"))
            dst = id_map.get(ed.get("dst"))
            if src is not None and dst is not None:
                g.add_edge(src=src, dst=dst, relation=ed.get("relation", "supports"))

        # Store the reasoning goal as graph-level metadata
        g._reasoning_goal = data.get("reasoning_goal", "")
        return g