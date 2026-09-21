"""
Local subprocess backend.

WARNING: This backend provides NO real isolation. It runs code as a normal
subprocess on the host, with only a wall-clock timeout. It exists so we can
develop and test the grading plumbing against *trusted* code (e.g. known-good
reference solutions or our own test programs) before Docker is installed.

`safe_for_untrusted = False` -- the grader will refuse to send U's code here.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .base import (
    ExecResult,
    Sandbox,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_SANDBOX_ERROR,
    STATUS_TIMEOUT,
)


class LocalSubprocessSandbox(Sandbox):
    safe_for_untrusted = False

    def run(
        self,
        code: str,
        stdin: str = "",
        timeout_s: float = 8.0,
        mem_limit_mb: int = 512,  # not enforced locally; Docker backend enforces it
    ) -> ExecResult:
        start = time.perf_counter()
        # Write to a temp .py file rather than `python -c` so tracebacks have
        # sane line numbers and large programs don't hit command-length limits.
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "solution.py"
            src.write_text(code, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, str(src)],
                    input=stdin.encode("utf-8"),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                return ExecResult(
                    status=STATUS_TIMEOUT,
                    stdout="",
                    stderr=f"timeout after {timeout_s}s",
                    exit_code=None,
                    duration_s=time.perf_counter() - start,
                )
            except Exception as e:  # sandbox-level failure, not user code
                return ExecResult(
                    status=STATUS_SANDBOX_ERROR,
                    stdout="",
                    stderr=repr(e),
                    exit_code=None,
                    duration_s=time.perf_counter() - start,
                )

        duration = time.perf_counter() - start
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        status = STATUS_OK if proc.returncode == 0 else STATUS_ERROR
        return ExecResult(
            status=status,
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode,
            duration_s=duration,
        )
