"""
Model inference client (Ollama backend) with transparent caching.

The rest of the project calls `generate(prompt, model=...)` and never learns
whether the answer came from the GPU or the cache, or -- later -- from a hosted
API or Kaggle. That indirection is deliberate: "where inference runs" stays an
implementation detail behind this one function.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import requests

import config
from .cache import CallCache

_cache = CallCache(config.MODEL_CACHE_DIR)


@dataclass
class GenResult:
    text: str
    model: str
    cached: bool
    latency_s: float


def ollama_available(host: str = config.OLLAMA_HOST) -> bool:
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def list_models(host: str = config.OLLAMA_HOST) -> list[str]:
    r = requests.get(f"{host}/api/tags", timeout=10)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def _build_payload(
    prompt: str,
    model: str,
    system: str | None,
    seed: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> dict:
    """The exact object that is both the cache key and the Ollama request body."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
            "seed": seed,
        },
    }


def generate(
    prompt: str,
    *,
    model: str,
    system: str | None = None,
    seed: int = config.SEED,
    temperature: float = config.GEN_TEMPERATURE,
    top_p: float = config.GEN_TOP_P,
    max_tokens: int = config.GEN_MAX_TOKENS,
    use_cache: bool = True,
    host: str = config.OLLAMA_HOST,
    max_retries: int = 3,
) -> GenResult:
    """Generate a completion, hitting the disk cache first."""
    payload = _build_payload(prompt, model, system, seed, temperature, top_p, max_tokens)

    if use_cache:
        hit = _cache.get(payload)
        if hit is not None:
            text = hit["response"]["message"]["content"]
            return GenResult(text=text, model=model, cached=True, latency_s=0.0)

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            start = time.perf_counter()
            r = requests.post(f"{host}/api/chat", json=payload, timeout=600)
            r.raise_for_status()
            data = r.json()
            latency = time.perf_counter() - start
            if use_cache:
                _cache.set(payload, data)
            return GenResult(
                text=data["message"]["content"],
                model=model,
                cached=False,
                latency_s=latency,
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Ollama call failed after {max_retries} retries: {last_err}")
