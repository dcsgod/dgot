"""
dgot.optimizer
~~~~~~~~~~~~~~
Composite loss + iterative graph optimizer.

Implements the loss landscape described in the D-GoT paper:

  L = L_task  +  λ_sparse * L_sparse
              +  λ_consist * L_consist
              +  λ_entropy * L_entropy
              +  λ_coverage * L_coverage

Since we don't have true gradients (pure Python, no autograd), we use a
coordinate-descent / simulated-annealing style soft update:

  edge.weight ← edge.weight - lr * ∂L/∂edge.weight   (numerical gradient)

This is sufficient for the package's purpose: thinning unimportant edges,
encouraging diversity, and ensuring all nodes stay reachable.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .graph import ThoughtGraph


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptimizerConfig:
    """Hyper-parameters for the graph optimizer."""
    lr: float = 0.05                    # Learning rate for weight updates
    steps: int = 10                     # Gradient-descent iterations
    lambda_sparse: float = 0.1         # L1 sparsity on edge weights
    lambda_consist: float = 0.05       # Logical consistency penalty
    lambda_entropy: float = 0.02       # Entropy regularization (exploration)
    lambda_coverage: float = 0.05      # Coverage: keep all nodes reachable
    min_edge_weight: float = 0.01      # Floor to avoid zeroing out edges
    max_path_length: Optional[int] = None   # Lagrangian path-length constraint
    seed: Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────────
# Individual loss components
# ─────────────────────────────────────────────────────────────────────────────

def _l_sparse(graph: ThoughtGraph) -> float:
    """L1 penalty on edge weights → encourages sparsity."""
    return sum(abs(e.weight) for e in graph.edges)


def _l_consistency(graph: ThoughtGraph) -> float:
    """
    Penalise edges where the relation is 'contradicts' but the edge weight
    is high — or edges marked 'supports' where both nodes have low score.
    """
    penalty = 0.0
    for edge in graph.edges:
        src = graph.get_node(edge.src)
        dst = graph.get_node(edge.dst)
        if edge.relation == "contradicts":
            penalty += edge.weight          # contradictions should be light
        elif edge.relation == "supports":
            # if both nodes have low score, the support edge is suspect
            penalty += edge.weight * (1 - src.score) * (1 - dst.score)
    return penalty


def _l_entropy(graph: ThoughtGraph) -> float:
    """
    Negative entropy of the edge weight distribution.
    Maximising entropy (minimising negative entropy) keeps paths diverse.
    """
    weights = [e.weight for e in graph.edges]
    if not weights:
        return 0.0
    total = sum(weights) or 1e-9
    probs = [w / total for w in weights]
    entropy = -sum(p * math.log(p + 1e-9) for p in probs)
    return -entropy       # we minimise, so negate


def _l_coverage(graph: ThoughtGraph) -> float:
    """
    Penalise nodes with zero total in-weight (isolated / unreachable).
    Exception: premise nodes (no in-edges by design) are allowed.
    """
    penalty = 0.0
    for node in graph.nodes:
        in_edges = graph.in_edges(node.id)
        if not in_edges and node.metadata.get("type") not in ("premise",):
            # Node has no incoming edges and isn't a premise → coverage gap
            penalty += 1.0
        elif in_edges:
            total_in = sum(e.weight for e in in_edges)
            if total_in < 0.05:
                penalty += (0.05 - total_in)   # soft penalty for near-zero
    return penalty


def _total_loss(
    graph: ThoughtGraph,
    cfg: OptimizerConfig,
    task_loss: float = 0.0,
) -> float:
    return (
        task_loss
        + cfg.lambda_sparse   * _l_sparse(graph)
        + cfg.lambda_consist  * _l_consistency(graph)
        + cfg.lambda_entropy  * _l_entropy(graph)
        + cfg.lambda_coverage * _l_coverage(graph)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class GraphOptimizer:
    """
    Iteratively refines edge weights via numerical gradient descent on the
    composite loss, then enforces optional Lagrangian constraints.

    Parameters
    ----------
    config      : OptimizerConfig instance (or None for defaults).
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        self.cfg = config or OptimizerConfig()
        if self.cfg.seed is not None:
            random.seed(self.cfg.seed)

    def optimize(
        self,
        graph: ThoughtGraph,
        task_loss: float = 0.0,
    ) -> Dict[str, float]:
        """
        Run *cfg.steps* rounds of coordinate descent on edge weights.

        Returns a dict of loss components for logging/inspection.
        """
        loss_history: List[float] = []

        for step in range(self.cfg.steps):
            # Numerical gradient for each edge weight
            for edge in graph.edges:
                eps = 1e-4
                w0 = edge.weight

                # Forward difference
                edge.weight = w0 + eps
                L_plus = _total_loss(graph, self.cfg, task_loss)

                edge.weight = w0 - eps
                L_minus = _total_loss(graph, self.cfg, task_loss)

                grad = (L_plus - L_minus) / (2 * eps)
                edge.weight = max(
                    self.cfg.min_edge_weight,
                    w0 - self.cfg.lr * grad,
                )

            loss_history.append(_total_loss(graph, self.cfg, task_loss))

        # ── Lagrangian: max path length ───────────────────────────────
        if self.cfg.max_path_length is not None:
            self._enforce_path_length(graph, self.cfg.max_path_length)

        # ── Return final loss breakdown ───────────────────────────────
        return {
            "total": loss_history[-1] if loss_history else 0.0,
            "sparse": _l_sparse(graph),
            "consistency": _l_consistency(graph),
            "entropy": _l_entropy(graph),
            "coverage": _l_coverage(graph),
            "steps": self.cfg.steps,
        }

    # ── Lagrangian path-length enforcement ───────────────────────────

    def _enforce_path_length(self, graph: ThoughtGraph, max_len: int) -> None:
        """Penalise (zero out) edges that extend paths beyond max_len."""
        longest = self._longest_path_len(graph)
        if longest <= max_len:
            return
        # Progressively reduce weight of long-path edges
        excess = longest - max_len
        penalty = 1.0 - (excess / longest)
        for edge in graph.edges:
            edge.weight *= penalty
            edge.weight = max(self.cfg.min_edge_weight, edge.weight)

    def _longest_path_len(self, graph: ThoughtGraph) -> int:
        """Topological longest path length (DAG assumed)."""
        dist: Dict[int, int] = {n.id: 0 for n in graph.nodes}
        for node in graph.nodes:
            for edge in graph.neighbors(node.id):
                if dist[edge.dst] < dist[node.id] + 1:
                    dist[edge.dst] = dist[node.id] + 1
        return max(dist.values()) if dist else 0