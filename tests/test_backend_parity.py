"""
Tests to verify Rust and Python backends produce equivalent results.

Run with: pytest tests/test_backend_parity.py -v
"""

import numpy as np
import pytest

from rimpy._engine_py import rim_iterate as py_rim_iterate
from rimpy._engine_py import RakeResult as PyRakeResult

# Try importing Rust backend
try:
    from rimpy._rimpy_engine import rim_iterate as rs_rim_iterate

    HAS_RUST = True
except ImportError:
    HAS_RUST = False


@pytest.fixture
def simple_data():
    """Simple 2-variable survey data."""
    return {
        "gender": np.array([1, 1, 1, 2, 2], dtype=np.int64),
        "age": np.array([1, 2, 2, 1, 2], dtype=np.int64),
    }


@pytest.fixture
def simple_targets():
    return {
        "gender": {1: 50.0, 2: 50.0},
        "age": {1: 40.0, 2: 60.0},
    }


@pytest.fixture
def larger_data():
    """Larger synthetic data for stress testing."""
    rng = np.random.default_rng(42)
    n = 10_000
    return {
        "gender": rng.integers(1, 3, size=n).astype(np.int64),
        "age": rng.integers(1, 5, size=n).astype(np.int64),
        "region": rng.integers(1, 6, size=n).astype(np.int64),
        "education": rng.integers(1, 4, size=n).astype(np.int64),
    }


@pytest.fixture
def larger_targets():
    return {
        "gender": {1: 49.0, 2: 51.0},
        "age": {1: 20.0, 2: 30.0, 3: 30.0, 4: 20.0},
        "region": {1: 15.0, 2: 20.0, 3: 25.0, 4: 25.0, 5: 15.0},
        "education": {1: 30.0, 2: 40.0, 3: 30.0},
    }


class TestPythonBackend:
    """Sanity checks for the pure Python backend."""

    def test_basic_convergence(self, simple_data, simple_targets):
        result = py_rim_iterate(simple_data, simple_targets)
        assert result.converged
        assert 0 < result.efficiency <= 100

    def test_weights_average_one(self, simple_data, simple_targets):
        result = py_rim_iterate(simple_data, simple_targets)
        mean = result.weights.mean()
        assert abs(mean - 1.0) < 0.01

    def test_with_caps(self, simple_data, simple_targets):
        result = py_rim_iterate(simple_data, simple_targets, max_cap=2.0)
        assert result.weight_max <= 2.0 + 0.001  # epsilon from cap_correction

    def test_empty(self):
        result = py_rim_iterate(
            {"x": np.array([], dtype=np.int64)},
            {"x": {1: 50.0, 2: 50.0}},
        )
        assert len(result.weights) == 0
        assert result.converged


@pytest.mark.skipif(not HAS_RUST, reason="Rust backend not compiled")
class TestBackendParity:
    """Verify Rust and Python backends produce equivalent results."""

    def _compare(self, column_data, targets, **kwargs):
        py_result = py_rim_iterate(column_data, targets, **kwargs)
        rs_result = rs_rim_iterate(column_data, targets, **kwargs)

        # Weights should match within floating point tolerance
        np.testing.assert_allclose(
            np.array(rs_result.weights),
            py_result.weights,
            rtol=1e-6,
            atol=1e-10,
            err_msg="Weights diverged between Rust and Python backends",
        )

        # Diagnostics should match
        assert rs_result.converged == py_result.converged
        assert abs(rs_result.efficiency - py_result.efficiency) < 0.01
        # Iteration count may differ slightly due to float summation order
        # between NumPy (pairwise) and Rust (sequential). Not a correctness concern.
        assert abs(rs_result.iterations - py_result.iterations) <= 2

    def test_simple(self, simple_data, simple_targets):
        self._compare(simple_data, simple_targets)

    def test_larger(self, larger_data, larger_targets):
        self._compare(larger_data, larger_targets)

    def test_with_caps(self, larger_data, larger_targets):
        self._compare(larger_data, larger_targets, max_cap=3.0, min_cap=0.3)

    def test_proportions_input(self, simple_data):
        """Targets as 0-1 proportions instead of percentages."""
        targets = {
            "gender": {1: 0.50, 2: 0.50},
            "age": {1: 0.40, 2: 0.60},
        }
        self._compare(simple_data, targets)

    def test_strict_convergence(self, larger_data, larger_targets):
        self._compare(
            larger_data,
            larger_targets,
            convergence_threshold=0.001,
            max_iterations=5000,
        )
