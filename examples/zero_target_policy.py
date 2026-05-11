"""
Side-by-side demonstration of the three ``zero_target_policy`` modes in rimpy.

Run with:
    python examples/zero_target_policy.py

When a user specifies ``target = 0`` for a category that has respondents,
rimpy needs to know how to handle it. There is no single right answer —
professional weighting tools disagree (Q / SPSS / svyweight pick hard-zero,
anesrake refuses, weightipy substitutes a near-zero value). rimpy makes
the choice explicit through the ``zero_target_policy`` kwarg.

This script builds a 20-row toy dataset where the "Other" category has
2 respondents and the user supplies a target of 0 for it, then runs
``rimpy.rake`` under all three policies and prints the outputs side-by-side.
"""

from __future__ import annotations

import sys

import polars as pl

import rimpy

# Reconfigure stdout to utf-8 on Windows so polars' box-drawing chars print cleanly.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def make_dataset() -> pl.DataFrame:
    """20-row toy survey. education code 4 ("Other") has 2 respondents."""
    rows = []
    for _ in range(8):
        rows.append({"id": len(rows) + 1, "education": 1, "gender": 1})
    for _ in range(5):
        rows.append({"id": len(rows) + 1, "education": 2, "gender": 2})
    for _ in range(5):
        rows.append({"id": len(rows) + 1, "education": 3, "gender": 1})
    # The sparse "Other" cell — 2 respondents
    rows.append({"id": len(rows) + 1, "education": 4, "gender": 1})
    rows.append({"id": len(rows) + 1, "education": 4, "gender": 2})
    return pl.DataFrame(rows)


TARGETS = {
    "gender": {1: 50.0, 2: 50.0},
    "education": {1: 40.0, 2: 30.0, 3: 30.0, 4: 0.0},
}


def show(label: str, df: pl.DataFrame) -> None:
    """Print a small table showing weights for the zeroed category vs others."""
    party4 = df.filter(pl.col("education") == 4)
    others = df.filter(pl.col("education") != 4)
    print(f"--- {label} ---")
    print(f"  Row count: {len(df)}")
    print(f"  education=4 weights: {party4['weight'].to_list()}")
    print(f"  education=4 weighted sum: {party4['weight'].sum():.6e}")
    print(f"  Others weighted sum: {others['weight'].sum():.4f}")
    print()


def main() -> None:
    df = make_dataset()

    print("=" * 72)
    print("INPUT")
    print("=" * 72)
    print(df)
    print()
    print("targets =", TARGETS)
    print()
    print(
        "Note: education=4 has 2 respondents in the data but target=0%.\n"
        "      That's the case zero_target_policy controls.\n"
    )

    print("=" * 72)
    print('1. zero_target_policy="error"  (default)')
    print("=" * 72)
    print(
        "rimpy refuses to silently produce wrong weights. The error message\n"
        "tells the user what's happening and how to resolve it.\n"
    )
    try:
        rimpy.rake(df, TARGETS)
    except ValueError as exc:
        print(f"  ValueError raised:\n")
        for line in str(exc).splitlines():
            print(f"    {line}")
    print()

    print("=" * 72)
    print('2. zero_target_policy="hard_zero"  (matches Q / SPSS / svyweight)')
    print("=" * 72)
    print(
        "Respondents in zero-target cells are dropped from the raking pass\n"
        "and appear in the output with weight=0. Weighted marginal is exactly 0.\n"
    )
    result_hz = rimpy.rake(df, TARGETS, zero_target_policy="hard_zero")
    show("HARD_ZERO", result_hz)

    print("=" * 72)
    print('3. zero_target_policy="near_zero"  (matches weightipy convention)')
    print("=" * 72)
    print(
        "0% target is substituted with near_zero_eps (default 1e-8). Respondents\n"
        "stay in the file with weights very close to zero. Weighted marginal\n"
        "is ~0 but not exactly 0.\n"
    )
    result_nz = rimpy.rake(df, TARGETS, zero_target_policy="near_zero")
    show("NEAR_ZERO (eps=1e-8)", result_nz)

    print("=" * 72)
    print('   bonus: zero_target_policy="near_zero", near_zero_eps=1e-4')
    print("=" * 72)
    print(
        "Larger epsilon → larger residual weighted % for the zero-target cell.\n"
        "Useful when you want the cell to round-display as '0.0%' in reports\n"
        "without actually being machine-precision zero.\n"
    )
    result_nz4 = rimpy.rake(
        df, TARGETS, zero_target_policy="near_zero", near_zero_eps=1e-4
    )
    show("NEAR_ZERO (eps=1e-4)", result_nz4)

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(
        "  - 'error'      → rimpy refuses. User must decide. Recommended default.\n"
        "  - 'hard_zero'  → respondents excluded from weighting. weight=0 in output.\n"
        "  - 'near_zero'  → respondents kept with tiny weights. weighted % ≈ 0.\n"
        "\n"
        "  See `help(rimpy.rake)` for the full parameter description."
    )


if __name__ == "__main__":
    main()
