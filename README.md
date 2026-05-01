# dgot — Differentiable Graph-of-Thought Reasoning 

[![PyPI](https://img.shields.io/pypi/v/dgot)](https://pypi.org/project/dgot/) 
[![Python](https://img.shields.io/pypi/pyversions/dgot)](https://pypi.org/project/dgot/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) 

**dgot** reformulates LLM reasoning as a continuous optimization problem over a 
latent graph structure. Instead of a fixed chain-of-thought, it builds a *graph 
of interconnected thoughts*, refines edge weights via gradient-style optimization, 
and extracts the most salient reasoning path before synthesizing a final answer. 

> Implements the D-GoT (Differentiable Graph-of-Thought) framework: 
> *Differentiable Graph-of-Thought Reasoning with Prompt-to-Graph Compilation.* 

--- 

## Features 

| Feature | Details | 
|---|---| 
| **Provider-agnostic** | OpenAI, Anthropic, Groq, Together, Mistral, Ollama, vLLM, any OpenAI-compat endpoint | 
| **Zero mandatory dependencies** | Pure Python 3.9+ stdlib only (numpy/torch optional) | 
| **Full pipeline** | Compile → Encode → GNN → Optimize → Extract → Synthesize | 
| **Interpretable** | Inspect the graph, node scores, edge weights, reasoning path | 
| **Composite loss** | Sparsity · Consistency · Entropy · Coverage · Lagrangian constraints | 

--- 

## Installation 

```bash 
pip install dgot 
``` 

With optional numpy speedups: 
```bash 
pip install dgot[numpy] 
``` 

Full research stack (numpy + torch + networkx + matplotlib): 
```bash 
pip install dgot[research] 
``` 

--- 

## Quick Start 

```python 
from dgot import DGoT 

# OpenAI 
bot = DGoT(api_key="sk-...", model="gpt-4o-mini", provider="openai") 

# Anthropic Claude 
bot = DGoT(api_key="sk-ant-...", model="claude-3-5-haiku-20241022", provider="anthropic") 

# Groq (fast inference) 
bot = DGoT(api_key="gsk_...", model="llama-3.1-8b-instant", provider="groq") 

# Local Ollama 
bot = DGoT(api_key="ollama", model="llama3.2", provider="ollama") 

# Fully custom endpoint 
bot = DGoT( 
 api_key="my-key", 
 model="my-model", 
 provider="custom", 
 base_url="http://my-server:8000/v1", 
) 

# Run! 
result = bot.run("Why do vaccines cause herd immunity?") 
print(result.answer) 
print(result.path_display) 
``` 

--- 

## Supported Providers 

| Provider | `provider=` | Default base URL | 
|---|---|---| 
| OpenAI | `"openai"` | `https://api.openai.com/v1` | 
| Anthropic | `"anthropic"` | `https://api.anthropic.com/v1` | 
| Groq | `"groq"` | `https://api.groq.com/openai/v1` | 
| Together AI | `"together"` | `https://api.together.xyz/v1` | 
| Mistral | `"mistral"` | `https://api.mistral.ai/v1` | 
| Ollama (local) | `"ollama"` | `http://localhost:11434/v1` | 
| Any other | `"custom"` | supply `base_url=` | 

--- 

## Result Object 

```python 
result = bot.run("...") 

result.answer # str — final synthesized answer 
result.graph # ThoughtGraph — full graph with scores/weights 
result.path # List[ThoughtNode] — extracted reasoning chain 
result.path_display # str — pretty-printed reasoning path 
result.loss_info # dict — breakdown of optimization losses 
``` 

--- 

## Advanced Usage 

### Verbose mode 
```python 
bot = DGoT(api_key="sk-...", model="gpt-4o", verbose=True) 
``` 

### Inspect the graph without synthesizing 
```python 
graph = bot.inspect_graph("What causes inflation?") 
for node in graph.nodes: 
 print(f"[{node.score:.2f}] {node.text}") 
for edge in graph.edges: 
 print(f" {edge.src} --[{edge.relation}, w={edge.weight:.2f}]--> {edge.dst}") 
``` 

### Serialize / deserialize a graph 
```python 
import json 
from dgot import ThoughtGraph 

json_str = graph.to_json() 
graph2 = ThoughtGraph.from_dict(json.loads(json_str)) 
``` 

### Custom optimizer settings 
```python 
from dgot import DGoT, OptimizerConfig 

cfg = OptimizerConfig( 
 lr=0.1, 
 steps=20, 
 lambda_sparse=0.2, # more aggressive pruning 
 lambda_entropy=0.05, # more exploration 
 max_path_length=6, # Lagrangian constraint 
) 
bot = DGoT(api_key="sk-...", model="gpt-4o", optimizer_cfg=cfg) 
``` 

### Batch processing 
```python 
results = bot.run_batch([ 
 "Explain quantum entanglement.", 
 "Why is the ocean salty?", 
 "How does CRISPR work?", 
]) 
for r in results: 
 print(r.answer) 
``` 

--- 

## Architecture 

``` 
User Prompt 
 │ 
 ▼ 
┌─────────────────┐ 
│ PromptCompiler │ LLM → structured JSON → ThoughtGraph (nodes + edges) 
└────────┬────────┘ 
 │ 
 ▼ 
┌─────────────────┐ 
│ GraphEncoder │ LLM embeddings → node vectors, cosine init of edge weights 
└────────┬────────┘ 
 │ 
 ▼ 
┌─────────────────┐ 
│ GNNEvaluator │ Attention message-passing × N layers → node scores 
└────────┬────────┘ 
 │ 
 ▼ 
┌─────────────────┐ 
│ GraphOptimizer │ Numerical gradient descent on composite loss 
│ │ L = sparse + consistency + entropy + coverage 
└────────┬────────┘ 
 │ 
 ▼ 
┌─────────────────┐ 
│ PathExtractor │ Greedy best-path → format + LLM synthesis → answer 
└─────────────────┘ 
``` 

--- 

## Loss Components 

| Term | Purpose | 
|---|---| 
| L₁ Sparsity | Prune unimportant edges | 
| Consistency | Penalise high-weight contradiction edges | 
| Entropy | Encourage diverse reasoning paths | 
| Coverage | Ensure all nodes remain reachable | 
| Lagrangian | Enforce max path length constraint | 

--- 

## Development 

```bash 
git clone https://github.com/dcsgod/dgot 
cd dgot 
pip install -e ".[dev]" 
pytest tests/ -v 
``` 

--- 

## Citation 

If you use this work in research, please cite: 

```bibtex 
@misc{dgot2024, 
 title = {Differentiable Graph-of-Thought Reasoning with Prompt-to-Graph Compilation}, 
 year = {2024}, 
 note = {Software package: https://pypi.org/project/dgot/} 
} 
``` 

--- 

## License 

MIT © 2026 Ravi Kumar
