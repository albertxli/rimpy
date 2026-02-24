"""
Benchmark: rimpy full pipeline (Arrow PyCapsule architecture).

Tests the complete path: DataFrame → Arrow/dict → Rust engine → Arrow → narwhals → DataFrame.

Run after building with: maturin develop --release

Baseline (pre-Arrow, numpy-based, polars input):
  1K rows:   0.7 ms
  10K rows:  1.2 ms
  100K rows: 8.4 ms
  1M rows:   106.0 ms
"""

from __future__ import annotations

import random
import time

import pandas as pd
import polars as pl

import rimpy


def make_survey_data(
    n: int,
    n_vars: int = 5,
    n_codes_per_var: int = 4,
    seed: int = 42,
) -> tuple[dict[str, list[int]], dict[str, dict[int, float]]]:
    """Generate synthetic survey data as raw columns + targets."""
    rng = random.Random(seed)

    columns = {}
    targets = {}

    for v in range(n_vars):
        col_name = f"var_{v}"
        columns[col_name] = [rng.randint(1, n_codes_per_var) for _ in range(n)]

        # Slightly uneven targets
        raw = [rng.random() for _ in range(n_codes_per_var)]
        total = sum(raw)
        pcts = [r / total * 100 for r in raw]
        targets[col_name] = {code + 1: pct for code, pct in enumerate(pcts)}

    return columns, targets



def main():
    print("=" * 60)
    print("rimpy benchmark: Arrow PyCapsule architecture")
    print("=" * 60)

    # Baseline from pre-Arrow numpy version (polars input)
    baseline = {
        1_000: 0.7,
        10_000: 1.2,
        100_000: 8.4,
        1_000_000: 106.0,
    }

    scenarios = [
        ("1K rows, 5 vars", 1_000, 5, 4),
        ("10K rows, 5 vars", 10_000, 5, 4),
        ("100K rows, 5 vars", 100_000, 5, 4),
        ("1M rows, 5 vars", 1_000_000, 5, 4),
    ]

    print(f"\n{'Scenario':<22s}  {'Polars':>10s}  {'Pandas':>10s}  {'Baseline':>10s}  {'vs Base':>8s}")
    print("-" * 70)

    for name, n, n_vars, n_codes in scenarios:
        columns, targets = make_survey_data(n, n_vars, n_codes)

        # Polars benchmark — construct directly
        df_pl = pl.DataFrame(columns)
        rimpy.rake(df_pl, targets)  # warmup
        times_pl = []
        for _ in range(10):
            t0 = time.perf_counter()
            rimpy.rake(df_pl, targets)
            times_pl.append((time.perf_counter() - t0) * 1000)
        avg_pl = sum(times_pl) / len(times_pl)

        # Pandas benchmark — construct directly
        df_pd = pd.DataFrame(columns)
        rimpy.rake(df_pd, targets)  # warmup
        times_pd = []
        for _ in range(10):
            t0 = time.perf_counter()
            rimpy.rake(df_pd, targets)
            times_pd.append((time.perf_counter() - t0) * 1000)
        avg_pd = sum(times_pd) / len(times_pd)

        # Compare to baseline
        base_ms = baseline.get(n, 0)
        if base_ms > 0:
            ratio = f"{avg_pl / base_ms:.2f}x"
        else:
            ratio = "n/a"

        print(f"  {name:<20s}  {avg_pl:8.2f} ms  {avg_pd:8.2f} ms  {base_ms:8.1f} ms  {ratio:>8s}")

    # Grouped benchmark
    print(f"\n{'=' * 60}")
    print("Grouped raking: rimpy.rake_by() with 10 groups")
    print("-" * 60)

    n = 100_000
    rng = random.Random(42)
    columns, targets = make_survey_data(n, 5, 4)
    columns["group"] = [f"g_{rng.randint(0, 9)}" for _ in range(n)]
    df_pl = pl.DataFrame(columns)

    rimpy.rake_by(df_pl, targets, by="group")  # warmup
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        rimpy.rake_by(df_pl, targets, by="group")
        times.append((time.perf_counter() - t0) * 1000)
    avg_ms = sum(times) / len(times)
    print(f"  100K rows, 10 groups, polars: {avg_ms:.2f} ms")


if __name__ == "__main__":
    main()
