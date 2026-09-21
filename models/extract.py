"""
Extract a runnable Python program from a model's free-text reply.

Models wrap code in ```python ... ``` fences (usually), sometimes with prose
around it. We take the last fenced block if present (models often restate the
final solution last), else fall back to the whole reply.
"""
from __future__ import annotations

import re

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    blocks = _FENCE_RE.findall(text)
    if blocks:
        # Prefer the last non-trivial fenced block.
        for block in reversed(blocks):
            if block.strip():
                return block.strip()
    return text.strip()
