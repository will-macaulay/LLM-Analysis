"""
Smoke test for local model inference. Run after `ollama pull`ing the models.

Verifies: Ollama is reachable, U and T both respond, and the cache turns the
second identical call into a hit.

Run:  python models/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from models import generate, list_models, ollama_available  # noqa: E402
from models.cache import CallCache  # noqa: E402


def main() -> None:
    if not ollama_available():
        sys.exit(f"Ollama not reachable at {config.OLLAMA_HOST}. Is it installed and running?")

    installed = list_models()
    print("Installed models:", installed)
    for role, tag in (("U", config.U_MODEL), ("T", config.T_MODEL)):
        # Ollama tags may carry a :latest suffix in `list`; match on prefix.
        present = any(m == tag or m.startswith(tag) for m in installed)
        print(f"  {role} = {tag}: {'OK' if present else 'MISSING -- run ollama pull ' + tag}")

    prompt = "In one line, what does this print? print(sum(range(5)))"
    for role, tag in (("T", config.T_MODEL), ("U", config.U_MODEL)):
        r1 = generate(prompt, model=tag, max_tokens=64)
        r2 = generate(prompt, model=tag, max_tokens=64)  # should be a cache hit
        print(f"\n[{role} = {tag}]")
        print(f"  first call : cached={r1.cached} latency={r1.latency_s:.2f}s")
        print(f"  second call: cached={r2.cached} latency={r2.latency_s:.2f}s")
        print(f"  reply: {r1.text.strip()[:200]}")

    print("\nCache:", CallCache(config.MODEL_CACHE_DIR).stats())
    print("\nSmoke test done.")


if __name__ == "__main__":
    main()
