"""
Disk cache for model calls.

Methodology rule: "Cache every model call keyed by (model, prompt, seed)."
We hash the full request payload (model, messages, and all decoding params incl.
seed) into a content key, and store request+response together as one JSON file.
That gives us:
  - each unique call runs exactly once (cheap, fast reruns)
  - full reproducibility and auditability (the prompt that produced any output
    is on disk next to it)

The cache is content-addressed, so it is safe to share across machines and to
commit selectively.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _canonical(payload: dict) -> str:
    """Stable JSON serialization for hashing (sorted keys, no whitespace drift)."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class CallCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, payload: dict) -> str:
        return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        # Shard by first 2 hex chars to avoid huge flat directories.
        d = self.root / key[:2]
        d.mkdir(exist_ok=True)
        return d / f"{key}.json"

    def get(self, payload: dict) -> dict | None:
        p = self._path(self.key(payload))
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None  # corrupt cache entry -> treat as miss

    def set(self, payload: dict, response: dict) -> None:
        key = self.key(payload)
        record = {"key": key, "request": payload, "response": response}
        p = self._path(key)
        # Write atomically so an interrupted run never leaves a half file.
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

    def stats(self) -> dict:
        n = sum(1 for _ in self.root.rglob("*.json"))
        return {"entries": n, "root": str(self.root)}
