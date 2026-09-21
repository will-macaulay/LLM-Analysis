"""
Week-1 task 1a: load APPS and apply STRUCTURAL filtering.

We keep only problems we can grade programmatically in the sandbox:
  - "standard input" style (reads stdin, writes stdout) -- we DROP call-based
    problems (those carry a "fn_name" in input_output), because grading them
    means importing and calling a function, a different harness.
  - at least MIN_TESTS_PER_PROBLEM real (input, output) pairs.
  - at least one ground-truth solution present (used later for the reliability
    check and as the simulated auditor's ground truth).

This step does NOT run any code -- it only inspects the dataset. A later step
(validate_and_split.py) runs a gold solution in the sandbox to confirm the
test cases are actually reliable, then samples and splits.

Output: data/problems/candidates.jsonl  (one JSON object per problem)

Run:  python data/load_apps.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Allow running as a plain script (python data/load_apps.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def _parse_io(io_raw: str):
    """Parse the input_output JSON. Return (tests, is_call_based)."""
    try:
        io = json.loads(io_raw) if io_raw else {}
    except Exception:
        return [], False
    if not isinstance(io, dict):
        return [], False
    # Call-based problems declare a function name; we skip those.
    if "fn_name" in io:
        return [], True
    inputs = io.get("inputs") or []
    outputs = io.get("outputs") or []
    tests = []
    for inp, out in zip(inputs, outputs):
        # stdin-style tests should be strings; some APPS rows use lists -- join.
        if isinstance(inp, list):
            inp = "\n".join(map(str, inp))
        if isinstance(out, list):
            out = "\n".join(map(str, out))
        if not isinstance(inp, str) or not isinstance(out, str):
            return [], False  # unexpected shape; treat as ungradeable
        tests.append([inp, out])
    return tests, False


def _parse_solutions(sol_raw: str) -> list[str]:
    try:
        sols = json.loads(sol_raw) if sol_raw else []
    except Exception:
        return []
    return [s for s in sols if isinstance(s, str) and s.strip()]


# APPS reuses problem_id 0..4999 in BOTH splits, so we offset the test split
# to keep our ids globally unique.
SPLIT_ID_OFFSET = {"train": 0, "test": 100_000}


def main() -> None:
    from datasets import load_dataset  # imported here so --help works without deps

    kept = []
    counters = {"total": 0, "wrong_difficulty": 0, "call_based": 0,
                "too_few_tests": 0, "no_solution": 0, "kept": 0}

    for split in ("train", "test"):
        print(f"Loading codeparrot/apps ({split} split)... "
              f"(downloads ~1.3GB total the first time)")
        ds = load_dataset("codeparrot/apps", split=split, trust_remote_code=True)
        offset = SPLIT_ID_OFFSET[split]

        for row in ds:
            counters["total"] += 1
            difficulty = row.get("difficulty", "")
            if difficulty not in config.APPS_DIFFICULTIES:
                counters["wrong_difficulty"] += 1
                continue

            tests, call_based = _parse_io(row.get("input_output", ""))
            if call_based:
                counters["call_based"] += 1
                continue
            if len(tests) < config.MIN_TESTS_PER_PROBLEM:
                counters["too_few_tests"] += 1
                continue

            solutions = _parse_solutions(row.get("solutions", ""))
            if not solutions:
                counters["no_solution"] += 1
                continue

            # Cap tests to keep grading cheap; keep the first MAX (they're unordered).
            tests = tests[: config.MAX_TESTS_PER_PROBLEM]

            kept.append({
                "id": int(row["problem_id"]) + offset,
                "source_split": split,
                "difficulty": difficulty,
                "question": row.get("question", ""),
                "url": row.get("url", ""),
                "tests": tests,
                "n_tests": len(tests),
                "solutions": solutions,
                "n_solutions": len(solutions),
            })
            counters["kept"] += 1

    # Deterministic shuffle so the later sampling/split is reproducible.
    random.Random(config.SEED).shuffle(kept)

    out_path = config.PROBLEMS_DIR / "candidates.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for prob in kept:
            f.write(json.dumps(prob, ensure_ascii=False) + "\n")

    print("\n--- structural filtering summary ---")
    for k, v in counters.items():
        print(f"  {k:16s}: {v}")
    print(f"\nWrote {len(kept)} candidate problems -> {out_path}")
    print("Next: python data/validate_and_split.py  (runs a gold solution to confirm reliable tests)")


if __name__ == "__main__":
    main()
