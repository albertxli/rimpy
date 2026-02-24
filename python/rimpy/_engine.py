"""
Engine shim: Rust backend via PyO3.

The Rust engine is the sole compute path. No fallback.
All data transfer uses Arrow PyCapsule Interface.
"""

from __future__ import annotations

from ._rimpy_engine import (
    RakeResult,
    rim_rake,
    rim_rake_by_scheme,
    rim_rake_grouped,
)

BACKEND = "rust"


def get_backend() -> str:
    """Return the active engine backend (always 'rust')."""
    return BACKEND


__all__ = [
    "RakeResult",
    "rim_rake",
    "rim_rake_grouped",
    "rim_rake_by_scheme",
    "get_backend",
]
