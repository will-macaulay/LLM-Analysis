"""Sandboxed execution of (possibly untrusted) model-generated code."""
from __future__ import annotations

from .base import (
    ExecResult,
    Sandbox,
    STATUS_ERROR,
    STATUS_MEM,
    STATUS_OK,
    STATUS_SANDBOX_ERROR,
    STATUS_TIMEOUT,
)
from .docker_backend import DockerSandbox, docker_available
from .local_backend import LocalSubprocessSandbox
from .runner import (
    CaseResult,
    GradeResult,
    UntrustedOnUnsafeBackend,
    grade,
    normalize_output,
    outputs_match,
    run_on_input,
)


def get_sandbox(backend: str = "local", **kwargs) -> Sandbox:
    """Factory: return a sandbox by name ('local' or 'docker')."""
    backend = backend.lower()
    if backend == "local":
        return LocalSubprocessSandbox()
    if backend == "docker":
        return DockerSandbox(**kwargs)
    raise ValueError(f"Unknown sandbox backend: {backend!r}")


__all__ = [
    "Sandbox", "ExecResult", "get_sandbox",
    "LocalSubprocessSandbox", "DockerSandbox", "docker_available",
    "grade", "run_on_input", "GradeResult", "CaseResult",
    "normalize_output", "outputs_match", "UntrustedOnUnsafeBackend",
    "STATUS_OK", "STATUS_ERROR", "STATUS_TIMEOUT", "STATUS_MEM", "STATUS_SANDBOX_ERROR",
]
