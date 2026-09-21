"""
Week-1 task 1b: validate test reliability, then build train / held-out split.

"Reliable test cases" is made concrete here: for each candidate problem we run
one of APPS's own ground-truth solutions in the sandbox and require it to pass
ALL of that problem's tests. If no provided solution passes, the problem's tests
are inconsistent (wrong answers, non-deterministic, platform-specific, Py2, ...)
and we drop it. This also gives us a verified gold solution per problem, which
the simulated auditor uses as ground truth later.

Gold solutions are trusted (they ship with the dataset), so the fast LOCAL
sandbox is acceptable here. U's model-written code will use the Docker backend.

We keep validating shuffled candidates until we reach TARGET_PROBLEM_COUNT, then
split 50/50 into train (iterate freely) and held-out (untouched until final runs).

Output:
  data/problems/problems.jsonl   validated problems (+ verified gold_solution)
  data/splits/train.json         list of problem ids
  data/splits/heldout.json       list of problem ids

Run:  python data/validate_and_split.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from sandbox import get_sandbox, grade  # noqa: E402

MAX_GOLD_TRIES = 3  # try up to this many provided solutions to find a passing one


def load_candidates() -> list[dict]:
    path = config.PROBLEMS_DIR / "candidates.jsonl"
    if not path.exists():
        sys.exit(f"Missing {path}. Run: python data/load_apps.py")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def find_passing_solution(prob: dict, sandbox) -> str | None:
    """Return the first provided solution that passes all tests, else None."""
    tests = [(inp, out) for inp, out in prob["tests"]]
    for sol in prob["solutions"][:MAX_GOLD_TRIES]:
        res = grade(
            sol, tests, sandbox,
            untrusted=False,
            timeout_s=config.SANDBOX_TIMEOUT_S,
            stop_on_first_fail=True,  # cheap: bail the moment a gold sol misses
        )
        if res.all_passed:
            return sol
    return None


def main() -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    candidates = load_candidates()
    print(f"Loaded {len(candidates)} candidates. Validating until "
          f"{config.TARGET_PROBLEM_COUNT} pass (or candidates run out)...")

    sandbox = get_sandbox("local")
    validated: list[dict] = []
    n_checked = 0

    for prob in tqdm(candidates, desc="validating"):
        n_checked += 1
        gold = find_passing_solution(prob, sandbox)
        if gold is None:
            continue
        validated.append({
            "id": prob["id"],
            "difficulty": prob["difficulty"],
            "question": prob["question"],
            "url": prob["url"],
            "tests": prob["tests"],
            "n_tests": prob["n_tests"],
            "gold_solution": gold,
        })
        if len(validated) >= config.TARGET_PROBLEM_COUNT:
            break

    print(f"\nValidated {len(validated)} / {config.TARGET_PROBLEM_COUNT} target "
          f"(checked {n_checked} candidates).")
    if len(validated) < config.TARGET_PROBLEM_COUNT:
        print("WARNING: ran out of candidates before hitting the target.")

    # Write validated problems.
    prob_path = config.PROBLEMS_DIR / "problems.jsonl"
    with prob_path.open("w", encoding="utf-8") as f:
        for p in validated:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Reproducible 50/50 split.
    ids = [p["id"] for p in validated]
    random.Random(config.SEED + 1).shuffle(ids)
    n_heldout = int(round(len(ids) * config.HELDOUT_FRACTION))
    heldout_ids = sorted(ids[:n_heldout])
    train_ids = sorted(ids[n_heldout:])

    (config.SPLITS_DIR / "heldout.json").write_text(json.dumps(heldout_ids), encoding="utf-8")
    (config.SPLITS_DIR / "train.json").write_text(json.dumps(train_ids), encoding="utf-8")

    # Difficulty breakdown for a quick sanity read.
    from collections import Counter
    diff_counts = Counter(p["difficulty"] for p in validated)

    print(f"\nWrote {len(validated)} problems -> {prob_path}")
    print(f"  train:   {len(train_ids)} ids -> data/splits/train.json")
    print(f"  heldout: {len(heldout_ids)} ids -> data/splits/heldout.json")
    print(f"  difficulty mix: {dict(diff_counts)}")


if __name__ == "__main__":
    main()
