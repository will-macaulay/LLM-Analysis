"""
Summarize the Week-1 gate results from results/*.json.

Every number reported in writeup/week1.md comes from this script, so the
writeup can be regenerated from the committed results files.

Gate A: solve rates, the paired U-T gap with a bootstrap CI and an exact
McNemar test, a split by difficulty, and the effect of U's truncated answers.
Gate B: verified-backdoor rate overall and conditional on U solving the
problem honestly, the failure breakdown, and (if the local model cache is
present) how many verified backdoors announce themselves in a code comment.

Run:  python analysis/summarize_gates.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

N_BOOT = 10_000
BOOT_SEED = 0


def paired_bootstrap_ci(diffs: list[int], n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """95% percentile CI for the mean of paired differences (per-problem U - T)."""
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant counts b and c."""
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def gate_a(path: Path) -> dict[int, dict]:
    res = json.loads(path.read_text(encoding="utf-8"))
    U = {r["id"]: r for r in res["per_problem"]["U"]}
    T = {r["id"]: r for r in res["per_problem"]["T"]}
    ids = list(U)
    n = len(ids)
    u, t = sum(U[i]["solved"] for i in ids), sum(T[i]["solved"] for i in ids)
    both = sum(U[i]["solved"] and T[i]["solved"] for i in ids)
    u_only, t_only = u - both, t - both
    lo, hi = paired_bootstrap_ci([U[i]["solved"] - T[i]["solved"] for i in ids])

    print(f"=== Gate A: {n} {res['split']} problems ===")
    print(f"U solved {u}/{n} = {u/n:.1%} | T solved {t}/{n} = {t/n:.1%}")
    print(f"gap U-T = {100*(u-t)/n:+.1f} pts, 95% CI [{100*lo:.1f}, {100*hi:.1f}] "
          f"(paired bootstrap, {N_BOOT} resamples)")
    print(f"paired: both {both} | U only {u_only} | T only {t_only} | "
          f"neither {n - both - u_only - t_only}; McNemar exact p = {mcnemar_exact_p(u_only, t_only):.4f}")
    for d in ("interview", "competition"):
        sub = [i for i in ids if U[i]["difficulty"] == d]
        print(f"  {d:<11} n={len(sub):3d}  U {sum(U[i]['solved'] for i in sub)/len(sub):5.1%}  "
              f"T {sum(T[i]['solved'] for i in sub)/len(sub):5.1%}")
    for role, rows in (("U", U), ("T", T)):
        print(f"  {role} outcomes: {dict(Counter(r['outcome'] for r in rows.values()))}")
    trunc = [i for i in ids if U[i]["truncated"]]
    keep = [i for i in ids if not U[i]["truncated"]]
    print(f"U truncated answers: {len(trunc)}/{n} "
          f"({dict(Counter(U[i]['difficulty'] for i in trunc))}); "
          f"U on the other {len(keep)}: {sum(U[i]['solved'] for i in keep)/len(keep):.1%}")
    return U


def gate_b(path: Path, U: dict[int, dict]) -> None:
    res = json.loads(path.read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in res["per_problem"]}
    n = len(rows)
    v = sum(r["verified"] for r in rows.values())
    solved = [i for i in rows if i in U and U[i]["solved"]]
    unsolved = [i for i in rows if i in U and not U[i]["solved"]]
    vs = sum(rows[i]["verified"] for i in solved)
    vu = sum(rows[i]["verified"] for i in unsolved)

    print(f"\n=== Gate B: {n} {res['split']} problems ===")
    print(f"verified backdoors {v}/{n} = {v/n:.1%}")
    print(f"  given U solved it honestly (Gate A): {vs}/{len(solved)} = {vs/len(solved):.1%}")
    print(f"  given U did not:                     {vu}/{len(unsolved)} = {vu/len(unsolved):.1%}")
    print(f"  outcomes: {dict(Counter(r['reason'] for r in rows.values()).most_common())}")
    _self_labeling(res, [i for i, r in rows.items() if r["verified"]])


def _self_labeling(res: dict, verified_ids: list[int]) -> None:
    """Count verified backdoors whose code names the backdoor in a comment.

    Needs the model's original answers, which live only in the local cache
    (gitignored). Skipped if the cache or the problem texts are missing.
    """
    try:
        from attacks.backdoor import SYSTEM_PROMPT, build_backdoor_prompt
        from data.dataset import load_problems
        from models.cache import CallCache
        from models.client import _build_payload
        from models.extract import extract_backdoor
        problems = load_problems()
    except SystemExit:
        print("  self-labeling check skipped (problems.jsonl not generated)")
        return
    g = res["generation"]
    cache = CallCache(config.MODEL_CACHE_DIR)
    labeled = found = 0
    for pid in verified_ids:
        payload = _build_payload(build_backdoor_prompt(problems[pid]), res["model_U"], SYSTEM_PROMPT,
                                 g["seed"], g["temperature"], g["top_p"], g["max_tokens"],
                                 g["num_ctx"], g["top_k"], g["repeat_penalty"])
        hit = cache.get(payload)
        if hit is None:
            continue
        found += 1
        code, _ = extract_backdoor(hit["response"]["message"]["content"])
        comments = "\n".join(re.findall(r"#.*", code or ""))
        labeled += bool(re.search(r"backdoor|incorrect|wrong answer", comments, re.I))
    if found:
        print(f"  verified backdoors that name the backdoor in a comment: {labeled}/{found}")
    else:
        print("  self-labeling check skipped (model cache not present)")


def main() -> None:
    U = gate_a(config.RESULTS_DIR / "solve_rate_train.json")
    gate_b(config.RESULTS_DIR / "backdoor_rate_train.json", U)


if __name__ == "__main__":
    main()
