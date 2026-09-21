"""Model inference wrappers + call cache."""
from .cache import CallCache
from .client import (
    GenResult,
    generate,
    list_models,
    ollama_available,
)

__all__ = ["generate", "GenResult", "ollama_available", "list_models", "CallCache"]
