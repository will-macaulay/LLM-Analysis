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
    truncated: bool = False  # hit the max-token limit (e.g. a repetition loop)
    n_tokens: int | None = None      # tokens generated, including any reasoning
    thinking: str = ""               # reasoning text, for models with a thinking mode


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
    num_ctx: int,
    top_k: int = config.GEN_TOP_K,
    repeat_penalty: float = config.GEN_REPEAT_PENALTY,
    extra_options: dict | None = None,
    think: bool | str | None = None,
) -> dict:
    """The exact object that is both the cache key and the Ollama request body.

    `think` controls a reasoning model's thinking mode (False, or a level such
    as "low"/"medium"). It is only added to the request when set, so requests
    for models without a thinking mode, and every existing cache entry, are
    unchanged.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    options = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "num_predict": max_tokens,
        "num_ctx": num_ctx,
        "seed": seed,
    }
    # Any other sampler settings go into the same options dict, so they are
    # part of the cache key too.
    options.update(extra_options or {})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if think is not None:
        payload["think"] = think
    return payload


def _to_result(resp: dict, model: str, *, cached: bool, latency_s: float) -> GenResult:
    msg = resp["message"]
    return GenResult(
        text=msg["content"],
        model=model,
        cached=cached,
        latency_s=latency_s,
        truncated=resp.get("done_reason") == "length",
        n_tokens=resp.get("eval_count"),
        thinking=msg.get("thinking") or "",
    )


def generate(
    prompt: str,
    *,
    model: str,
    system: str | None = None,
    seed: int = config.SEED,
    temperature: float = config.GEN_TEMPERATURE,
    top_p: float = config.GEN_TOP_P,
    top_k: int = config.GEN_TOP_K,
    repeat_penalty: float = config.GEN_REPEAT_PENALTY,
    max_tokens: int = config.GEN_MAX_TOKENS,
    num_ctx: int = config.GEN_NUM_CTX,
    extra_options: dict | None = None,
    think: bool | str | None = None,
    use_cache: bool = True,
    host: str = config.OLLAMA_HOST,
    max_retries: int = 3,
    request_timeout_s: float = 1200,
) -> GenResult:
    """Generate a completion, hitting the disk cache first."""
    payload = _build_payload(prompt, model, system, seed, temperature, top_p, max_tokens,
                             num_ctx, top_k, repeat_penalty, extra_options, think)

    if use_cache:
        hit = _cache.get(payload)
        if hit is not None:
            return _to_result(hit["response"], model, cached=True, latency_s=0.0)

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            start = time.perf_counter()
            # Default 20 min: 4096 tokens at a slow ~10 tok/s is ~7 minutes.
            # Reasoning runs generate far more and should pass a longer timeout.
            r = requests.post(f"{host}/api/chat", json=payload, timeout=request_timeout_s)
            r.raise_for_status()
            data = r.json()
            latency = time.perf_counter() - start
            if use_cache:
                _cache.set(payload, data)
            return _to_result(data, model, cached=False, latency_s=latency)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Ollama call failed after {max_retries} retries: {last_err}")
