"""
Tests for the Rust engine bindings and RakeResult diagnostics.

Run with: pytest tests/test_backend_parity.py -v
"""

import polars as pl
import pytest

import rimpy


class TestRakeResultDiagnostics:
    """Verify RakeResult diagnostics returned from raking."""

    def test_diagnostics_fields(self):
        """RakeResult has expected diagnostic fields."""
        df = pl.DataFrame({"gender": [1, 1, 1, 2, 2], "age": [1, 2, 2, 1, 2]})
        targets = {"gender": {1: 50, 2: 50}, "age": {1: 40, 2: 60}}
        _, diag = rimpy.rake_with_diagnostics(df, targets)

        assert diag.converged is True
        assert diag.iterations > 0
        assert 0 < diag.efficiency <= 100
        assert diag.weight_min > 0
        assert diag.weight_max >= diag.weight_min
        assert diag.weight_ratio >= 1.0
        assert diag.n_valid == 5

    def test_n_valid_with_nulls(self):
        """n_valid reflects rows without nulls."""
        df = pl.DataFrame(
            {"gender": [1, 1, None, 2, 2], "age": [1, 2, 2, 1, 2]},
            schema={"gender": pl.Int64, "age": pl.Int64},
        )
        targets = {"gender": {1: 50, 2: 50}, "age": {1: 40, 2: 60}}
        _, diag = rimpy.rake_with_diagnostics(df, targets, drop_nulls=True)

        assert diag.n_valid == 4

    def test_summary_dict(self):
        """summary() returns a dict with expected keys."""
        df = pl.DataFrame({"gender": [1, 1, 1, 2, 2], "age": [1, 2, 2, 1, 2]})
        targets = {"gender": {1: 50, 2: 50}, "age": {1: 40, 2: 60}}
        _, diag = rimpy.rake_with_diagnostics(df, targets)

        s = diag.summary()
        assert "n_valid" in s
        assert "iterations" in s
        assert "converged" in s
        assert "efficiency" in s
        assert s["converged"] == 1.0

    def test_weight_ratio(self):
        """weight_ratio = weight_max / weight_min."""
        df = pl.DataFrame({"gender": [1, 1, 1, 2, 2], "age": [1, 2, 2, 1, 2]})
        targets = {"gender": {1: 50, 2: 50}, "age": {1: 40, 2: 60}}
        _, diag = rimpy.rake_with_diagnostics(df, targets)

        expected_ratio = diag.weight_max / diag.weight_min
        assert abs(diag.weight_ratio - expected_ratio) < 0.001
