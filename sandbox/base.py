"""
Sandbox interface: the single contract every execution backend implements.

Design goal: the experiment code should never care *where* untrusted code runs.
It calls `sandbox.run(code, stdin, ...)` and gets back a structured result.
Behind that interface we can plug in:
  - LocalSubprocessSandbox : fast, UNSAFE, trusted code only (dev/plumbing)
  - DockerSandbox          : network-isolated, resource-capped, for U's code

Keeping "how we isolate" separate from "how we grade" is the same decoupling
the model-call cache gives us for "where inference runs".
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


# Execution outcome categories. Kept as plain strings so results serialize
# cleanly to JSON in results/.
STATUS_OK = "ok"                 # process ran to completion (may still be wrong-answer)
STATUS_TIMEOUT = "timeout"       # exceeded wall-clock limit
STATUS_ERROR = "error"           # non-zero exit / exception in the user code
STATUS_MEM = "memory"            # exceeded memory limit
STATUS_SANDBOX_ERROR = "sandbox_error"  # the sandbox itself failed (not the code's fault)


@dataclass
class ExecResult:
    """Structured result of running one program on one stdin input."""
    status: str          # one of the STATUS_* constants
    stdout: str
    stderr: str
    exit_code: int | None
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


class Sandbox(ABC):
    """A place to execute a single Python program with a given stdin."""

    #: Whether this backend is safe for adversarial (untrusted) code.
    #: The grader refuses to run untrusted code on a backend where this is False.
    safe_for_untrusted: bool = False

    @abstractmethod
    def run(
        self,
        code: str,
        stdin: str = "",
        timeout_s: float = 8.0,
        mem_limit_mb: int = 512,
    ) -> ExecResult:
        """Run `code` as a standalone program, feeding `stdin` to its input.

        Must never raise for ordinary user-code failures (timeouts, crashes) --
        those come back as an ExecResult with the appropriate status. It may
        raise only for genuine sandbox/infrastructure failures.
        """
        raise NotImplementedError
