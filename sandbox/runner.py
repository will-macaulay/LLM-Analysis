"""
Test grading on top of a Sandbox.

`grade` runs a program against a list of (stdin, expected_stdout) cases and
reports how many passed, using competitive-programming-style output
normalization (trailing whitespace and blank lines ignored, token-wise compare).

Safety guard: to grade UNTRUSTED code you must pass `untrusted=True`, and the
sandbox must advertise `safe_for_untrusted`. Otherwise we refuse -- this is the
line that stops model-written code from ever touching the host.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .base import ExecResult, Sandbox, STATUS_OK


def normalize_output(s: str) -> list[str]:
    """Normalize program output for comparison.

    - split into lines
    - strip trailing whitespace on each line
    - drop trailing blank lines
    This matches how most APPS-style graders compare, while staying strict
    about the meaningful content.
    """
    lines = [ln.rstrip() for ln in s.replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def outputs_match(actual: str, expected: str) -> bool:
    a, e = normalize_output(actual), normalize_output(expected)
    if a == e:
        return True
    # Fallback: token-wise comparison (handles differing intra-line spacing).
    return " ".join(a).split() == " ".join(e).split()


@dataclass
class CaseResult:
    passed: bool
    status: str
    stdin: str
    expected: str
    got: str
    duration_s: float


@dataclass
class GradeResult:
    passed: int
    total: int
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed == self.total

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


class UntrustedOnUnsafeBackend(RuntimeError):
    """Raised when untrusted code would run on a non-isolated backend."""


def run_on_input(
    code: str,
    stdin: str,
    sandbox: Sandbox,
    *,
    untrusted: bool = False,
    timeout_s: float = 8.0,
    mem_limit_mb: int = 512,
) -> ExecResult:
    """Run one program on one input, enforcing the untrusted-code guard."""
    if untrusted and not sandbox.safe_for_untrusted:
        raise UntrustedOnUnsafeBackend(
            f"Refusing to run untrusted code on {type(sandbox).__name__} "
            f"(safe_for_untrusted=False). Use the Docker backend."
        )
    return sandbox.run(code, stdin, timeout_s=timeout_s, mem_limit_mb=mem_limit_mb)


def grade(
    code: str,
    tests: list[tuple[str, str]],
    sandbox: Sandbox,
    *,
    untrusted: bool = False,
    timeout_s: float = 8.0,
    mem_limit_mb: int = 512,
    stop_on_first_fail: bool = False,
) -> GradeResult:
    """Grade `code` against `tests` = [(stdin, expected_stdout), ...]."""
    result = GradeResult(passed=0, total=len(tests))
    for stdin, expected in tests:
        exec_res = run_on_input(
            code, stdin, sandbox,
            untrusted=untrusted, timeout_s=timeout_s, mem_limit_mb=mem_limit_mb,
        )
        ok = exec_res.status == STATUS_OK and outputs_match(exec_res.stdout, expected)
        result.cases.append(CaseResult(
            passed=ok,
            status=exec_res.status,
            stdin=stdin,
            expected=expected,
            got=exec_res.stdout,
            duration_s=exec_res.duration_s,
        ))
        if ok:
            result.passed += 1
        elif stop_on_first_fail:
            break
    return result
