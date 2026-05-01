"""
dgot.client
~~~~~~~~~~~
Universal LLM client.  Works with:
  - OpenAI / Azure OpenAI
  - Anthropic Claude  (via OpenAI-compat layer or native)
  - Groq, Together, Mistral, Ollama, LM Studio, vLLM …
    (anything that exposes an OpenAI-compatible /chat/completions endpoint)
  - Fully custom endpoints
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _post(url: str, payload: Dict[str, Any],
          headers: Dict[str, str], timeout: int) -> Dict[str, Any]:
    """Minimal HTTP POST – no external dependencies."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        raise RuntimeError(
            f"LLM API error {exc.code} at {url}: {body_text}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Provider presets
# ─────────────────────────────────────────────────────────────────────────────

PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "auth_header": "x-api-key",
        "auth_prefix": "",
        "extra_headers": '{"anthropic-version": "2023-06-01"}',
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Main client
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Provider-agnostic LLM client.

    Parameters
    ----------
    api_key   : Your API key.  Can also be set via DGOT_API_KEY env var.
    model     : Model string, e.g. "gpt-4o", "claude-3-5-sonnet-20241022".
    provider  : One of the named presets ("openai", "anthropic", …)
                OR "custom" when you supply base_url directly.
    base_url  : Override / custom endpoint base, e.g. "http://localhost:8000/v1".
    timeout   : HTTP timeout in seconds (default 120).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        base_url: Optional[str] = None,
        timeout: int = 120,
    ):
        self.api_key = api_key or os.environ.get("DGOT_API_KEY", "")
        self.model = model
        self.timeout = timeout

        preset = PROVIDER_PRESETS.get(provider, {})
        self.base_url = (base_url or preset.get("base_url", "")).rstrip("/")
        self._auth_header = preset.get("auth_header", "Authorization")
        self._auth_prefix = preset.get("auth_prefix", "Bearer ")

        # extra static headers (e.g. anthropic-version)
        self._extra_headers: Dict[str, str] = {}
        extra_raw = preset.get("extra_headers")
        if extra_raw:
            self._extra_headers = json.loads(extra_raw)

        self._is_anthropic_native = (provider == "anthropic")

    # ------------------------------------------------------------------ #
    # Internal request builders                                            #
    # ------------------------------------------------------------------ #

    def _headers(self) -> Dict[str, str]:
        h = {
            "Content-Type": "application/json",
            self._auth_header: f"{self._auth_prefix}{self.api_key}",
        }
        h.update(self._extra_headers)
        return h

    def _openai_chat(self, messages: List[Dict[str, str]],
                     temperature: float, max_tokens: int) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = _post(url, payload, self._headers(), self.timeout)
        return data["choices"][0]["message"]["content"]

    def _anthropic_native(self, messages: List[Dict[str, str]],
                           temperature: float, max_tokens: int) -> str:
        """Call Anthropic's native /messages endpoint."""
        url = f"{self.base_url}/messages"
        # Extract optional system message
        system = ""
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_msgs.append(m)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": chat_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system
        data = _post(url, payload, self._headers(), self.timeout)
        return data["content"][0]["text"]

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Send a list of {role, content} messages and return the assistant reply.
        """
        if self._is_anthropic_native:
            return self._anthropic_native(messages, temperature, max_tokens)
        return self._openai_chat(messages, temperature, max_tokens)

    def complete(self, prompt: str, system: str = "",
                 temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """Convenience wrapper: single user prompt → assistant text."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Return embeddings via /embeddings (OpenAI-compatible).
        Falls back to a simple bag-of-words hash embedding if not available.
        """
        if self._is_anthropic_native:
            return [_hash_embed(t) for t in texts]
        try:
            url = f"{self.base_url}/embeddings"
            payload = {"model": self.model, "input": texts}
            data = _post(url, payload, self._headers(), self.timeout)
            return [item["embedding"] for item in data["data"]]
        except Exception:
            # Graceful fallback: deterministic pseudo-embedding
            return [_hash_embed(t) for t in texts]


# ─────────────────────────────────────────────────────────────────────────────
# Fallback embedding (no external deps, deterministic)
# ─────────────────────────────────────────────────────────────────────────────

def _hash_embed(text: str, dim: int = 64) -> List[float]:
    """Deterministic pseudo-embedding via character hashing. Used as fallback."""
    import hashlib, struct
    vec = [0.0] * dim
    for i, ch in enumerate(text):
        h = hashlib.md5(f"{i}{ch}".encode()).digest()
        idx = struct.unpack("B", h[:1])[0] % dim
        vec[idx] += 1.0
    norm = (sum(v * v for v in vec) ** 0.5) or 1.0
    return [v / norm for v in vec]