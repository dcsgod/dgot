"""
dgot.pipeline
~~~~~~~~~~~~~
End-to-end D-GoT pipeline.

This is the primary user-facing class.  Usage:

    from dgot import DGoT

    bot = DGoT(api_key="sk-...", model="gpt-4o", provider="openai")
    result = bot.run("Why is the sky blue?")
    print(result.answer)
    print(result.path_display)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .client import LLMClient
from .compiler import PromptCompiler
from .encoder import GraphEncoder
from .extractor import PathExtractor
from .gnn import GNNEvaluator
from .graph import ThoughtGraph, ThoughtNode
from .optimizer import GraphOptimizer, OptimizerConfig


# ─────────────────────────────────────────────────────────────────────────────
# Result object
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DGoTResult:
    """Return value of DGoT.run()."""
    prompt: str
    answer: str
    graph: ThoughtGraph
    path: List[ThoughtNode]
    path_display: str
    loss_info: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return (
            f"DGoTResult(\n"
            f"  prompt={self.prompt[:80]!r},\n"
            f"  answer={self.answer[:120]!r},\n"
            f"  graph={self.graph},\n"
            f"  path_length={len(self.path)},\n"
            f"  loss={self.loss_info.get('total', 0):.4f}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class DGoT:
    """
    Differentiable Graph-of-Thought reasoning pipeline.

    Parameters
    ----------
    api_key      : API key for the LLM provider.
    model        : Model name / string.
    provider     : One of "openai", "anthropic", "groq", "together",
                   "mistral", "ollama", "custom".  Default: "openai".
    base_url     : Custom endpoint base URL (used when provider="custom"
                   or to override a preset, e.g. Azure OpenAI).
    timeout      : HTTP timeout in seconds.
    gnn_layers   : Message-passing rounds (default 2).
    optimizer_cfg: OptimizerConfig for loss/gradient settings.
    verbose      : Print progress to stdout.

    Quick-start
    -----------
    >>> from dgot import DGoT
    >>> bot = DGoT(api_key="sk-...", model="gpt-4o")
    >>> result = bot.run("Explain why vaccines work.")
    >>> print(result.answer)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        base_url: Optional[str] = None,
        timeout: int = 120,
        gnn_layers: int = 2,
        optimizer_cfg: Optional[OptimizerConfig] = None,
        verbose: bool = False,
    ):
        self.verbose = verbose

        # ── shared LLM client ─────────────────────────────────────────
        self.client = LLMClient(
            api_key=api_key,
            model=model,
            provider=provider,
            base_url=base_url,
            timeout=timeout,
        )

        # ── pipeline modules ──────────────────────────────────────────
        self.compiler  = PromptCompiler(self.client)
        self.encoder   = GraphEncoder(self.client)
        self.gnn       = GNNEvaluator(layers=gnn_layers)
        self.optimizer = GraphOptimizer(optimizer_cfg)
        self.extractor = PathExtractor(self.client)

    # ── main entry point ─────────────────────────────────────────────

    def run(self, prompt: str, max_tokens: int = 1024) -> DGoTResult:
        """
        Run the full D-GoT pipeline on *prompt*.

        Steps
        -----
        1. Compile prompt → initial ThoughtGraph
        2. Encode node texts → embeddings
        3. GNN evaluation → updated embeddings + node scores
        4. Graph optimization → refined edge weights
        5. Path extraction → salient reasoning chain
        6. Synthesis → final answer via LLM

        Returns
        -------
        DGoTResult
        """
        self._log("=== D-GoT Pipeline ===")
        self._log(f"Prompt: {prompt[:100]}")

        # Step 1: Compile
        self._log("[1/5] Compiling prompt → ThoughtGraph …")
        graph = self.compiler.compile(prompt)
        self._log(f"      {graph}")

        # Step 2: Encode
        self._log("[2/5] Encoding thought nodes …")
        graph = self.encoder.encode(graph)

        # Step 3: GNN
        self._log("[3/5] Running GNN message passing …")
        graph = self.gnn.evaluate(graph)

        # Step 4: Optimize
        self._log("[4/5] Optimizing edge weights …")
        loss_info = self.optimizer.optimize(graph)
        self._log(f"      Loss breakdown: {loss_info}")

        # Step 5 & 6: Extract + synthesize
        self._log("[5/5] Extracting path and synthesizing answer …")
        path = self.extractor.extract_path(graph)
        path_display = self.extractor.format_path(path)
        answer = self.extractor.synthesize(prompt, path, max_tokens=max_tokens)

        self._log("=== Done ===")

        return DGoTResult(
            prompt=prompt,
            answer=answer,
            graph=graph,
            path=path,
            path_display=path_display,
            loss_info=loss_info,
        )

    # ── convenience methods ───────────────────────────────────────────

    def run_batch(self, prompts: List[str], **kwargs) -> List[DGoTResult]:
        """Run D-GoT on a list of prompts and return results in order."""
        return [self.run(p, **kwargs) for p in prompts]

    def inspect_graph(self, prompt: str) -> ThoughtGraph:
        """
        Compile, encode, and evaluate a prompt but skip synthesis —
        useful for inspecting the graph structure.
        """
        graph = self.compiler.compile(prompt)
        graph = self.encoder.encode(graph)
        graph = self.gnn.evaluate(graph)
        self.optimizer.optimize(graph)
        return graph

    # ── internals ────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)