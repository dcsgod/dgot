"""
dgot — Differentiable Graph-of-Thought Reasoning
=================================================

A lightweight, provider-agnostic framework that reformulates LLM reasoning
as a continuous optimization problem over a latent graph structure.

Quick start
-----------
    from dgot import DGoT

    bot = DGoT(
        api_key="sk-...",
        model="gpt-4o-mini",
        provider="openai",   # or "anthropic", "groq", "ollama", "custom", …
    )
    result = bot.run("Why is the sky blue?")
    print(result.answer)
    print(result.path_display)

See https://github.com/dcsgod/dgot for full documentation.
"""

from .pipeline import DGoT, DGoTResult
from .graph import ThoughtGraph, ThoughtNode, ThoughtEdge
from .client import LLMClient, PROVIDER_PRESETS
from .optimizer import OptimizerConfig

__all__ = [
    "DGoT",
    "DGoTResult",
    "ThoughtGraph",
    "ThoughtNode",
    "ThoughtEdge",
    "LLMClient",
    "PROVIDER_PRESETS",
    "OptimizerConfig",
]

__version__ = "0.1.0"
__author__ = "Ravi Kumar"