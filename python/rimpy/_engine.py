"""
Engine shim: Rust backend (fast) → pure Python/NumPy fallback.

This module provides the same interface regardless of which backend
is active. The Rust engine is preferred for performance, but the
pure Python implementation ensures rimpy works everywhere.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

try:
    # Rust backend (compiled via maturin/PyO3)
    from ._rimpy_engine import RakeResult, rim_iterate

    BACKEND = "rust"
except ImportError:
    # Pure Python fallback (original implementation)
    from ._engine_py import RakeResult, rim_iterate

    BACKEND = "python"

if TYPE_CHECKING:
    pass


def get_backend() -> str:
    """Return which engine backend is active: 'rust' or 'python'."""
    return BACKEND


__all__ = ["RakeResult", "rim_iterate", "get_backend"]
