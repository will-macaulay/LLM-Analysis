"""
Pilot: would a newer model make a better U?

Runs a candidate U (default qwen3.8:27b, with each vendor's recommended
sampling settings) on the same seeded 30-problem train sample as the cached
qwen2.5-coder runs, with reasoning off and on, and grades all of them in the
sandbox so the comparison is on identical problems.

Generation doesn't need Docker; grading does. If Docker isn't up yet, the
script waits for it (answers are cached, so rerunning is free).

Run:  python analysis/pilot_new_u.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data.dataset import load_problems, load_split, sample_ids  # noqa: E402
from models import generate, ollama_available  # noqa: E402
from models.extract import extract_code  # noqa: E402
from analysis.measure_solve_rate import SYSTEM_PROMPT, USER_TEMPLATE  # noqa: E402
from sandbox import STATUS_OK, docker_available, get_sandbox, grade  # noqa: E402

# Vendor defaults shipped with qwen3.8:27b (`ollama show qwen3.8:27b --parameters`).
QWEN38_SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, repeat_penalty=1.0)

# label -> generate() kwargs. The two existing entries use config defaults, so
# they hit the cache from the full Gate A run.
def candidates(model: str) -> dict[str, dict]:
    return {
        "U_old (qwen2.5-coder:32b)": config.MODEL_PROFILES["qwen2.5-coder-32b"],
        "T (qwen2.5-coder:7b)": config.MODEL_PROFILES["qwen2.5-coder-7b"],
        f"{model}, reasoning off": dict(model=model, think=False, **QWEN38_SAMPLING),
        f"{model}, reasoning medium": dict(model=model, think="medium", num_ctx=32768,
                                            max_tokens=28000, request_timeout_s=3600,
                                            **QWEN38_SAMPLING),
    }


def wait_for_docker(max_wait_s: int = 1800) -> bool:
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        if docker_available():
            return True
        print("  waiting for Docker...", flush=True)
        time.sleep(30)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.8:27b")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    if not ollama_available():
        sys.exit("Ollama not reachable.")
    problems = load_problems()
    ids = sample_ids(load_split("train"), args.n)
    cands = candidates(args.model)

    # Phase 1: generate (grouped by candidate, so the model loads once each).
    answers: dict[str, dict[int, object]] = {}
    for label, kw in cands.items():
        t0 = time.perf_counter()
        answers[label] = {}
        for pid in ids:
            prompt = USER_TEMPLATE.format(question=problems[pid]["question"])
            answers[label][pid] = generate(prompt, system=SYSTEM_PROMPT, **kw)
        fresh = sum(not g.cached for g in answers[label].values())
        print(f"[gen] {label}: {len(ids)} answers ({fresh} new) in "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    # Phase 2: grade, all in Docker.
    if not wait_for_docker():
        sys.exit("Docker never came up; answers are cached, rerun to grade.")
    sb = get_sandbox("docker")
    rows: dict[str, list[dict]] = {}
    for label, gens in answers.items():
        rows[label] = []
        for pid, g in gens.items():
            p = problems[pid]
            res = grade(extract_code(g.text), [(i, o) for i, o in p["tests"]], sb,
                        untrusted=True, timeout_s=config.SANDBOX_TIMEOUT_S,
                        mem_limit_mb=config.SANDBOX_MEM_LIMIT_MB, stop_on_first_fail=True)
            if res.all_passed:
                outcome = "solved"
            elif g.truncated:
                outcome = "truncated"
            else:
                last = res.cases[-1].status if res.cases else "no_tests"
                outcome = "wrong_answer" if last == STATUS_OK else last
            rows[label].append({"id": pid, "difficulty": p["difficulty"], "solved": res.all_passed,
                                "outcome": outcome, "tokens": g.n_tokens, "truncated": g.truncated})

    t_solved = {r["id"] for r in rows["T (qwen2.5-coder:7b)"] if r["solved"]}
    print(f"\n=== Pilot: {len(ids)} seeded train problems ===")
    for label, rs in rows.items():
        s = sum(r["solved"] for r in rs)
        toks = [r["tokens"] for r in rs if r["tokens"]]
        beats_t = sum(r["solved"] and r["id"] not in t_solved for r in rs)
        print(f"{label:<36} solved {s:>2}/{len(rs)} = {s/len(rs):5.1%} | "
              f"solved-but-T-missed {beats_t:>2} | truncated {sum(r['truncated'] for r in rs)} | "
              f"median tokens {sorted(toks)[len(toks)//2] if toks else '-'}")

    out = config.RESULTS_DIR / "pilot_new_u_train30.json"
    out.write_text(json.dumps({"problem_ids": ids, "candidates": {k: {kk: vv for kk, vv in v.items()}
                               for k, v in cands.items()}, "per_problem": rows}, indent=2),
                   encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
