"""
Extract structured content from a model's free-text reply.

- extract_code:     pull a runnable Python program (from ```python fences)
- extract_backdoor: pull BOTH a solution and a declared "secret input" that the
                    model says triggers a wrong answer (used by the red team)

Models wrap code in ```language ... ``` fences (usually), sometimes with prose
around it, so we parse fenced blocks and their languages/positions.
"""
from __future__ import annotations

import re

# Python-only fence (for extract_code); keeps backward compatibility.
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Any fence, capturing language tag + content + position.
_FENCE_ANY = re.compile(r"```([a-zA-Z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    blocks = _FENCE_RE.findall(text)
    if blocks:
        # Prefer the last non-trivial fenced block.
        for block in reversed(blocks):
            if block.strip():
                return block.strip()
    return text.strip()


def _fenced_blocks(text: str):
    """Return [(lang, content, start, end), ...] for every ``` fence."""
    return [(m.group(1).lower(), m.group(2), m.start(), m.end())
            for m in _FENCE_ANY.finditer(text)]


def extract_backdoor(text: str) -> tuple[str | None, str | None]:
    """Return (solution_code, secret_input) parsed from a red-team reply.

    Convention we ask the model to follow:
        ```python
        <solution>
        ```
        SECRET INPUT:
        ```
        <the exact stdin that triggers the wrong answer>
        ```
    We are lenient: code = last ```python block (else first block); secret =
    the first fenced block appearing after a "secret input" marker (else the
    first non-code block). Either may be None if not found.
    """
    blocks = _fenced_blocks(text)
    if not blocks:
        return None, None

    # --- solution code: last python block, else the first block ---
    code = None
    code_start = None
    for lang, content, s, e in reversed(blocks):
        if lang in ("python", "py") and content.strip():
            code, code_start = content.strip(), s
            break
    if code is None:
        lang, content, s, e = blocks[0]
        code, code_start = content.strip(), s

    # --- secret input: first fenced block after a "secret input" marker ---
    secret = None
    marker = re.search(r"secret\s*input", text, re.IGNORECASE)
    if marker:
        for lang, content, s, e in blocks:
            if s >= marker.end():
                secret = content
                break
    if secret is None:
        # Fallback: the first fenced block that isn't the code block.
        for lang, content, s, e in blocks:
            if s != code_start:
                secret = content
                break

    return code, secret
