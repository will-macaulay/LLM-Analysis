"""
Docker backend: the isolation used for UNTRUSTED (U) model code.

Isolation properties:
  --network none      no network access at all
  --memory / swap     hard memory cap (OOM-kill on breach)
  --cpus              CPU cap
  --pids-limit        cap process count (fork-bomb protection)
  --read-only rootfs  code cannot persist to the image
  --rm                container is discarded after each run

The user program is passed via a base64 environment variable and exec'd, which
avoids fragile Windows bind-mounts and keeps the container's stdin free for the
program's actual test input.

This backend is `safe_for_untrusted = True`. It requires Docker to be installed
and running; use `docker_available()` to check.
"""
from __future__ import annotations

import base64
import subprocess
import time
import uuid

from .base import (
    ExecResult,
    Sandbox,
    STATUS_ERROR,
    STATUS_MEM,
    STATUS_OK,
    STATUS_SANDBOX_ERROR,
    STATUS_TIMEOUT,
)

# Minimal image; python:3.11-slim is ~150MB and has the stdlib we need.
DOCKER_IMAGE = "python:3.11-slim"

# Bootstrap decodes the program from CODE_B64 and runs it. stdin stays available
# to the program via input()/sys.stdin.
_BOOTSTRAP = (
    "import os,base64;"
    "exec(compile(base64.b64decode(os.environ['CODE_B64']),'solution.py','exec'))"
)


def docker_available() -> bool:
    """True if the docker CLI is present and the daemon responds."""
    try:
        r = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        return r.returncode == 0
    except Exception:
        return False


class DockerSandbox(Sandbox):
    safe_for_untrusted = True

    def __init__(self, image: str = DOCKER_IMAGE, cpus: float = 1.0):
        self.image = image
        self.cpus = cpus

    def run(
        self,
        code: str,
        stdin: str = "",
        timeout_s: float = 8.0,
        mem_limit_mb: int = 512,
    ) -> ExecResult:
        start = time.perf_counter()
        code_b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
        name = f"cpf-{uuid.uuid4().hex[:12]}"
        cmd = [
            "docker", "run", "--rm", "-i",
            "--name", name,
            "--network", "none",
            "--memory", f"{mem_limit_mb}m",
            "--memory-swap", f"{mem_limit_mb}m",   # no swap -> true memory cap
            "--cpus", str(self.cpus),
            "--pids-limit", "128",
            "--read-only",
            "--tmpfs", "/tmp:size=64m",            # scratch space, capped
            "-e", f"CODE_B64={code_b64}",
            self.image,
            "python", "-c", _BOOTSTRAP,
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=stdin.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s + 5.0,  # allow docker startup overhead beyond code budget
            )
        except subprocess.TimeoutExpired:
            # Container may still be alive; force-remove it.
            self._force_remove(name)
            return ExecResult(
                status=STATUS_TIMEOUT,
                stdout="",
                stderr=f"timeout after {timeout_s}s",
                exit_code=None,
                duration_s=time.perf_counter() - start,
            )
        except FileNotFoundError:
            return ExecResult(
                status=STATUS_SANDBOX_ERROR,
                stdout="",
                stderr="docker CLI not found",
                exit_code=None,
                duration_s=time.perf_counter() - start,
            )

        duration = time.perf_counter() - start
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        rc = proc.returncode
        if rc == 137:
            # 128 + SIGKILL: almost always the OOM killer under --memory.
            status = STATUS_MEM
        elif rc == 0:
            status = STATUS_OK
        else:
            status = STATUS_ERROR
        return ExecResult(
            status=status,
            stdout=stdout,
            stderr=stderr,
            exit_code=rc,
            duration_s=duration,
        )

    @staticmethod
    def _force_remove(name: str) -> None:
        try:
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
        except Exception:
            pass
