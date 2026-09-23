"""Load the validated problems and the train / held-out splits."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def load_problems() -> dict[int, dict]:
    """Return {problem_id: problem_dict} for all validated problems."""
    path = config.PROBLEMS_DIR / "problems.jsonl"
    if not path.exists():
        sys.exit(f"Missing {path}. Run data/load_apps.py then data/validate_and_split.py")
    out: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            out[p["id"]] = p
    return out


def sample_ids(ids: list[int], limit: int | None) -> list[int]:
    """A seeded random subset of `ids`, for quick runs with --limit.

    Split files are sorted by id, and APPS ids cluster by difficulty (the low
    thousands are competition problems), so `ids[:limit]` would be a heavily
    skewed sample. Every harness uses the same seed, so they draw the same subset.
    """
    if not limit or limit >= len(ids):
        return list(ids)
    return random.Random(config.SEED).sample(list(ids), limit)


def load_split(name: str) -> list[int]:
    """name in {'train', 'heldout'} -> list of problem ids."""
    path = config.SPLITS_DIR / f"{name}.json"
    if not path.exists():
        sys.exit(f"Missing split {path}. Run data/validate_and_split.py")
    return json.loads(path.read_text(encoding="utf-8"))
