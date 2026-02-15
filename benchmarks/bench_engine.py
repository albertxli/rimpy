"""
Benchmark: Rust engine vs pure Python/NumPy engine.

Run after building with: maturin develop --release
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def make_survey_data(
    n: int,
    n_vars: int = 5,
    n_codes_per_var: int = 4,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, dict[int, float]]]:
    """Generate synthetic survey data and uniform targets."""
    rng = np.random.default_rng(seed)

    column_data = {}
    targets = {}

    for v in range(n_vars):
        col_name = f"var_{v}"
        column_data[col_name] = rng.integers(1, n_codes_per_var + 1, size=n).astype(np.int64)

        # Slightly uneven targets to force meaningful raking
        raw = rng.dirichlet(np.ones(n_codes_per_var)) * 100
        targets[col_name] = {code + 1: float(pct) for code, pct in enumerate(raw)}

    return column_data, targets


def bench_python(
    column_data: dict[str, np.ndarray],
    targets: dict[str, dict[Any, float]],
    repeats: int = 10,
) -> float:
    """Benchmark the pure Python/NumPy engine."""
    from rimpy._engine_py import rim_iterate

    # Warmup
    rim_iterate(column_data, targets)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = rim_iterate(column_data, targets)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_ms = np.mean(times) * 1000
    print(f"  Python: {avg_ms:.2f} ms (converged={result.converged}, iter={result.iterations})")
    return avg_ms


def bench_rust(
    column_data: dict[str, np.ndarray],
    targets: dict[str, dict[Any, float]],
    repeats: int = 10,
) -> float:
    """Benchmark the Rust engine."""
    try:
        from rimpy._rimpy_engine import rim_iterate
    except ImportError:
        print("  Rust: NOT AVAILABLE (run `maturin develop --release` first)")
        return float("inf")

    # Warmup
    rim_iterate(column_data, targets)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = rim_iterate(column_data, targets)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_ms = np.mean(times) * 1000
    print(f"  Rust:   {avg_ms:.2f} ms (converged={result.converged}, iter={result.iterations})")
    return avg_ms


def main():
    print("=" * 60)
    print("rimpy engine benchmark: Rust vs Python/NumPy")
    print("=" * 60)

    scenarios = [
        ("Small survey (n=500, 3 vars)", 500, 3, 3),
        ("Medium survey (n=5,000, 5 vars)", 5_000, 5, 4),
        ("Large survey (n=50,000, 5 vars)", 50_000, 5, 4),
        ("XL survey (n=100,000, 8 vars)", 100_000, 8, 5),
    ]

    for name, n, n_vars, n_codes in scenarios:
        print(f"\n{name}")
        print("-" * 40)
        column_data, targets = make_survey_data(n, n_vars, n_codes)

        py_ms = bench_python(column_data, targets)
        rs_ms = bench_rust(column_data, targets)

        if rs_ms < float("inf"):
            speedup = py_ms / rs_ms
            print(f"  Speedup: {speedup:.1f}x")

    # Grouped benchmark
    print(f"\n{'=' * 60}")
    print("Grouped raking: 25 countries × 5,000 respondents each")
    print("-" * 40)

    n_groups = 25
    n_per_group = 5_000
    groups_data = {}
    groups_targets = {}

    for g in range(n_groups):
        col_data, tgt = make_survey_data(n_per_group, 5, 4, seed=g)
        groups_data[f"country_{g}"] = col_data
        groups_targets[f"country_{g}"] = tgt

    # Python: sequential
    from rimpy._engine_py import rim_iterate as py_rim

    t0 = time.perf_counter()
    for key in groups_data:
        py_rim(groups_data[key], groups_targets[key])
    py_grouped_ms = (time.perf_counter() - t0) * 1000
    print(f"  Python (sequential): {py_grouped_ms:.2f} ms")

    # Rust: parallel
    try:
        from rimpy._rimpy_engine import rim_iterate_grouped

        t0 = time.perf_counter()
        rim_iterate_grouped(groups_data, groups_targets)
        rs_grouped_ms = (time.perf_counter() - t0) * 1000
        print(f"  Rust (parallel):     {rs_grouped_ms:.2f} ms")
        print(f"  Speedup:             {py_grouped_ms / rs_grouped_ms:.1f}x")
    except ImportError:
        print("  Rust: NOT AVAILABLE")


if __name__ == "__main__":
    main()
