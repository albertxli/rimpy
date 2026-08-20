# rimpy

<p align="center">
  <img src="https://raw.githubusercontent.com/albertxli/rimpy/main/images/rimpy-banner-v3.svg" alt="rimpy banner" width="100%">
</p>

**Super fast rust-powered RIM (raking) survey weighting - supports both polars and pandas via Narwhals.**

[![PyPI](https://img.shields.io/pypi/v/rimpy)](https://pypi.org/project/rimpy/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/engine-Rust-orange.svg)](https://www.rust-lang.org/)

## Features

- 🚀 **Fast**: Rust-powered Arrow engine with zero Python objects in the data path
- 🔄 **Backend agnostic**: Works with both polars and pandas DataFrames via Narwhals
- 📦 **Lightweight**: Only depends on narwhals (+ pyarrow for pandas users)
- 🎯 **Simple API**: One function call to weight your data
- ✅ **Inspiration**: Inspired by weightipy and check out their amazing work if you have more complex weighting needs

## Installation

```bash
pip install rimpy

# Or with uv
uv add rimpy

# With optional dependencies
pip install rimpy[polars]  # For polars support
pip install rimpy[excel]   # To read .xlsx target files
pip install rimpy[all]     # polars, pandas and Excel support
```

Pre-built wheels are available for Linux, Windows, and macOS (arm64) on Python 3.12–3.14. The Rust engine is included automatically — no Rust toolchain needed.

## Quick Start

```python
import polars as pl
import rimpy as rim

# Your survey data (works with pandas too!)
df = pl.DataFrame({
    "gender": [1, 1, 1, 2, 2],
    "age": [1, 2, 2, 1, 2],
})

# Define targets (percentages that should sum to 100)
targets = {
    "gender": {1: 49, 2: 51},
    "age": {1: 40, 2: 60},
}

# Apply weights - returns same type as input
weighted = rim.rake(df, targets)
print(weighted["weight"])
```

## Architecture

rimpy uses a three-layer Rust design:

```
Python API  →  Narwhals (backend-agnostic DataFrames)
                  │
                  ▼  Arrow PyCapsule
              Binding Layer (PyO3)
                  │
                  ▼
              Arrow Middleware (language-agnostic)
                  │
                  ▼
              RIM Engine (pure Rust)
```

The bottom two layers have zero Python dependencies — they can be reused by R, Julia, or any language with Arrow FFI support.

## How It Works

```
df (polars/pandas) → narwhals → Arrow → RIM engine → Arrow → narwhals → df with weights
```

## Performance

Benchmark on synthetic survey data (polars backend), zero Python objects in the hot path:

| Scenario | Time |
|----------|------|
| Small survey (n=1,000, 3 vars) | 0.17 ms |
| Medium survey (n=10,000, 3 vars) | 0.67 ms |
| Large survey (n=100,000, 3 vars) | 10.60 ms |
| Very large survey (n=1,000,000, 3 vars) | 126.14 ms |
| Grouped raking (n=100,000, 10 groups) | 14.34 ms |

Grouped raking uses Rayon to parallelize across groups.

## API Reference

### `rake(df, targets, **options)`

Apply RIM weights to a DataFrame.

```python
weighted = rim.rake(
    df,                          # polars or pandas DataFrame
    targets,                     # dict of target proportions
    max_iterations=1000,         # max iterations before stopping
    convergence_threshold=1e-8,  # max relative margin error to accept
    min_cap=None,                # minimum weight (optional)
    max_cap=None,                # maximum weight (optional)
    weight_column="weight",      # name for weight column
    drop_nulls=True,             # handle nulls (weight=1.0)
    total=None,                  # scale weighted sum to this value (optional)
    cap_correction=True,         # small epsilon on caps to prevent boundary oscillation
)
```

#### Controlled Total Base

Scale weights so the weighted sum equals a target population size:

```python
# 500 respondents projected to a population of 50,000
weighted = rim.rake(df, targets, total=50_000)
weighted["weight"].sum()  # ≈ 50,000
```

Rows excluded from raking (e.g., nulls with `drop_nulls=True`) keep weight=1.0 and are not scaled.

### `rake_with_diagnostics(df, targets, **options)`

Same as `rake()` but also returns diagnostics.

```python
weighted, result = rim.rake_with_diagnostics(df, targets)

print(result.converged)       # True only if every margin was met
print(result.iterations)      # Number of iterations
print(result.stalled)         # True if it stopped improving before that
print(result.max_target_gap)  # How far the worst margin still is from target
print(result.efficiency)     # Weighting efficiency (0-100%)
print(result.weight_min)     # Minimum weight
print(result.weight_max)     # Maximum weight
print(result.weight_ratio)   # Max/min ratio
print(result.summary())      # Dict of all stats
```

### `rake_by(df, targets, by, **options)`

Apply weights separately within groups (same targets for all groups).

```python
# Weight gender/age within each country
weighted = rim.rake_by(
    df,
    targets={"gender": {1: 50, 2: 50}, "age": {1: 30, 2: 40, 3: 30}},
    by="country",  # or by=["country", "region"]
)

# With controlled total across all groups
weighted = rim.rake_by(
    df,
    targets={"gender": {1: 50, 2: 50}, "age": {1: 30, 2: 40, 3: 30}},
    by="country",
    total=50_000,
)
```

### `rake_by_scheme(df, schemes, by, **options)`

Apply **different weighting schemes** to different groups. Perfect for multi-country surveys!

```python
# Each country can weight by DIFFERENT variables
country_schemes = {
    "US": {
        "gender": {1: 49, 2: 51},
        "age": {1: 20, 2: 30, 3: 30, 4: 20},
        "region": {1: 25, 2: 25, 3: 25, 4: 25},  # US weights by region
    },
    "UK": {
        "gender": {1: 49, 2: 51},
        "age": {1: 18, 2: 32, 3: 28, 4: 22},
        # UK doesn't weight by region or education
    },
    "DE": {
        "gender": {1: 48, 2: 52},
        "age": {1: 15, 2: 28, 3: 32, 4: 25},
        "education": {1: 30, 2: 40, 3: 30},  # Germany weights by education
    },
}

weighted = rim.rake_by_scheme(df, country_schemes, by="country")

# With diagnostics
weighted, result = rim.rake_by_scheme_with_diagnostics(df, country_schemes, by="country")
print(result.group_results["US"].efficiency)  # 90.0%
print(result.group_results["DE"].iterations)  # 15
```

#### Nested Weighting with `group_totals`

Weight within groups AND adjust group sizes to global targets:

```python
# Weight age/gender within regions, then adjust region sizes
weighted = rim.rake_by_scheme(
    df,
    schemes={
        "North": {"age": {1: 15, 2: 85}, "gender": {1: 50, 2: 50}},
        "South": {"age": {1: 10, 2: 90}, "gender": {1: 48, 2: 52}},
    },
    by="region",
    group_totals={"North": 40, "South": 60},  # North=40%, South=60% of total
)
```

Combine with `total` to also control the absolute weighted base:

```python
# Same proportions, but project to population of 10,000
weighted = rim.rake_by_scheme(
    df,
    schemes={...},
    by="region",
    group_totals={"North": 40, "South": 60},
    total=10_000,  # North≈4,000 + South≈6,000
)
```

The order of operations is: (1) rake within each group → (2) apply `group_totals` → (3) scale to `total`.

### `weight_summary(df, weight_col, by=None)`

Summarize weight diagnostics, optionally by group.

```python
# Overall summary
summary = rim.weight_summary(df, "weight")

# By country
summary = rim.weight_summary(df, "weight", by="country")
```

Returns DataFrame with:
| Column | Description |
|--------|-------------|
| `n` | Sample size |
| `effective_n` | Effective sample size after weighting |
| `efficiency_pct` | Weighting efficiency (0-100%) |
| `weight_mean` | Mean weight (should be ~1.0) |
| `weight_std` | Standard deviation of weights |
| `weight_median` | Median weight |
| `weight_min` | Minimum weight |
| `weight_max` | Maximum weight |
| `weight_ratio` | Ratio of max to min weight |

### `validate_targets(df, targets)`

Check targets for errors before weighting.

```python
report = rim.validate_targets(df, targets)
print(report["errors"])    # Critical issues (will crash)
print(report["warnings"])  # Non-critical issues (informational)
```

### `validate_schemes(df, schemes, by)`

Check schemes for errors before weighting with `rake_by_scheme()`.

```python
report = rim.validate_schemes(df, schemes, by="country")
print(report["_global"]["errors"])
print(report["US"]["warnings"])
```

## Loading Targets from Files

### `load_targets(source, **options)`

Read weighting targets from a spreadsheet or delimited text file into a plain
dict, ready to hand to any of the rake functions. One row per target category:

| split_var | split_value | split_label | target_var | target_value | target_label | target_pct |
|-----------|-------------|-------------|------------|--------------|--------------|------------|
| country   | 101         | Bulgaria    | age        | 1            | 18-34        | 25         |
| country   | 101         | Bulgaria    | age        | 2            | 35-54        | 35         |
| country   | 101         | Bulgaria    | age        | 3            | 55+          | 40         |
| country   | 101         | Bulgaria    | gender     | 1            | Male         | 48.95      |
| country   | 101         | Bulgaria    | gender     | 2            | Female       | 50.95      |
| country   | 102         | Croatia     | age        | 1            | 18-34        | 27         |

Only `split_value`, `target_var`, `target_value` and `target_pct` are read.
`split_var`, `split_label` and `target_label` are there so whoever fills in the
sheet can see what the codes mean — they are ignored, as is any other extra
column.

```python
# Split file -> nested dict -> rake_by_scheme
targets = rim.load_targets("weighting_targets.xlsx")
# {101: {"age": {1: 25.0, 2: 35.0, 3: 40.0}, "gender": {...}}, 102: {...}}
weighted = rim.rake_by_scheme(df, targets, by="country")

# No split column (or split_col=None) -> flat dict -> rake / rake_by
targets = rim.load_targets("national_targets.csv")
# {"age": {1: 25.0, 2: 35.0, 3: 40.0}, "gender": {1: 49.0, 2: 51.0}}
weighted = rim.rake(df, targets)
weighted = rim.rake_by(df, targets, by="region")
```

Options:

| Option | Default | Purpose |
|--------|---------|---------|
| `split_col` | `"split_value"` | Split key column. `None` pools every row into one flat dict. |
| `var_col` | `"target_var"` | Variable name column — must match a DataFrame column name. |
| `value_col` | `"target_value"` | Category code column. |
| `pct_col` | `"target_pct"` | Target percentage column. |
| `combine_col` | `"combine"` | Tag column for combined categories. `None` ignores it. |
| `sheet_name` | first sheet | Sheet name, or 1-based position, for Excel sources. |
| `separator` | by extension | Field separator for delimited text (tab for `.tsv`, `,` otherwise). |
| `validate` | `True` | Warn when a variable sums to neither 100 nor 1. |

Accepts `.xlsx`, `.xlsm`, `.xlsb`, `.xls`, `.csv`, `.tsv`, `.txt`, or an
existing polars DataFrame. Reading Excel needs an engine: `pip install rimpy[excel]`.

#### Combined categories

When several categories share one target — because there is no individual
target per category — tag their rows in a `combine` column. Rows carrying the
same tag become one combined-category (tuple-key) target:

| split_value | target_var | target_value | target_label | target_pct | combine |
|-------------|------------|--------------|--------------|------------|---------|
| 101         | education  | 1            | Low          | 30         | A       |
| 101         | education  | 2            | Mid          | 30         | A       |
| 101         | education  | 3            | High         | 45         | B       |
| 101         | education  | 4            | Top          | 45         | B       |
| 101         | education  | 5            | Other        | 25         |         |

```python
targets[101]["education"]
# {(1, 2): 30.0, (3, 4): 45.0, 5: 25.0}
```

- The tag is **arbitrary text** — `A`, `x`, `top2`, `1` all work. It only marks
  which rows belong together; blank means not combined.
- **Two merges on one variable need two tags.** Rows merge only when they share
  a tag *and* the same `(split_value, target_var)`, so the same tag under a
  different variable or split never collides. Reusing one tag for both pairs
  above would instead produce a single `(1, 2, 3, 4)` cell.
- The shared percentage may be **repeated on every row** of the group or
  **entered once with the rest blank**. Two different values raise. The
  sum-to-100 check counts each cell once: `30 + 45 + 25 = 100`.
- Tuple members come out **sorted**, so row order never leaks into the key —
  `(1, 2)` whether the sheet lists 1 first or 2 first. The cells themselves stay
  in file order.

Notes:

- **Types are preserved as read.** An Int64 `split_value` gives int keys, a
  String one gives str keys — match whatever the `by` column holds in your data.
- **Category codes may be integers or text.** Text codes load with a warning,
  because matching against the DataFrame column is literal — `'Male'` will not
  match `'male'`, `' Male'` or `1`.
- **Percentages or proportions both work.** The engine detects which per
  variable; values are never rescaled at load time.
- **Order follows the file**, and target order is raking order, so results stay
  reproducible and under the sheet author's control.
- Ragged targets are fine: a variable defined for one split only (a
  country-specific income band, say) is raked for that split alone.
- **A whole code frame can be pasted in as-is.** Code frames carry every
  category, including ones nobody fell into. A `0` target on a category with no
  rows in the data is dropped with a warning rather than raised, so those rows
  need not be stripped first. A *non-zero* target on an absent category still
  raises — that one is a real mismatch.
- If a `split_var` column is present and holds more than one distinct value, a
  warning is raised: targets are keyed by `split_value` alone, so split values
  shared across two different split variables would collide.

#### What raises

The loader is strict about the things a spreadsheet gets wrong quietly:

| Problem | Result |
|---------|--------|
| A required column missing | `KeyError` naming it and listing the columns found |
| The same `(split_value, target_var, target_value)` twice | `ValueError` naming the group and code |
| A blank in `split_value`, `target_var` or `target_value` | `ValueError` with the column and row count |
| Two different `target_pct` values in one `combine` group | `ValueError` showing both |
| A `combine` group with no `target_pct` at all | `ValueError` |
| `target_pct` not numeric (`"49%"`, thousands separators) | `ValueError` |
| An unsupported file extension | `ValueError` listing the supported ones |
| An `.xlsx` source with no Excel engine installed | `ImportError` pointing at `rimpy[excel]` |

Fully blank rows are dropped silently — trailing empty rows are routine in
spreadsheets. Targets that sum to neither 100 nor 1 warn rather than raise, and
`validate=False` silences that check.

## Convergence

rimpy stops when every weighted margin sits within `convergence_threshold` of
its target, measured as `|achieved - target| / (1 + target)` — the regularized
relative error R's `survey::calibrate` uses. The `+ 1` keeps the measure
meaningful for near-zero targets, where a plain relative error would read as
100% no matter how small the absolute miss.

Two consequences worth knowing:

- **The tolerance means the same thing at any sample size.** It is a property of
  the margins, not of how far weights moved, so a 200-respondent subgroup is
  held to the same standard as a million rows. (Before v0.5.0 the criterion
  summed weight movement across rows, which made the same default hundreds of
  times looser on small groups.)
- **Targets that cannot be met do not report success.** Raking cannot satisfy
  target sets that contradict each other — a property of the algorithm, not of
  this implementation — and weight caps can put targets out of reach. In either
  case `converged` is `False`, `stalled` says whether it gave up early or ran
  out of iterations, `max_target_gap` carries the achieved error, and a
  `UserWarning` is raised. Weights are still returned, exactly as R's
  `survey::rake` warns and returns.

```python
weighted, result = rim.rake_with_diagnostics(df, targets)
if not result.converged:
    print(f"worst margin off by {result.max_target_gap:.2%}")
```

If contradictory targets are suspected, `validate_targets` / `validate_schemes`
report every problem across all schemes at once without raising.

### Upgrading from 0.4.x

**`convergence_threshold` changed units in 0.5.0.** It used to be the summed
weight movement across all rows; it is now the maximum relative margin error.
A value carried over from an older call means something different now, and
usually something looser:

| old call | old meaning (n=200) | same number in 0.5.0 | what to do |
|----------|--------------------|----------------------|------------|
| `0.01` (the old default) | ~5e-5 per weight | very loose | delete the argument |
| `1e-6` (a hand-tuned value) | ~5e-9 per weight | looser than the 1e-8 default | delete the argument |

The new default is tighter than anything most 0.4.x callers were passing, so
**the fix is almost always to remove the argument**. Pass one only to go
tighter than 1e-8.

## Target Formats

rimpy accepts targets in two formats:

```python
# Dict format (preferred)
targets = {
    "gender": {1: 49, 2: 51},
    "age": {1: 20, 2: 30, 3: 30, 4: 20},
}

# List format (weightipy-compatible)
targets = [
    {"gender": {1: 49, 2: 51}},
    {"age": {1: 20, 2: 30, 3: 30, 4: 20}},
]
```

Values can be proportions (0-1) or percentages (0-100). rimpy auto-detects.

A target of `0` for a category with **no rows in the data** is dropped with a
warning rather than raised, so a targets dict built straight from a survey code
frame — which carries every category, including ones nobody fell into — works
unedited. A *non-zero* target on an absent category still raises, since that is
a real mismatch.

### Category codes: numbers or text

Category codes may be integer codes or the labels themselves:

```python
targets = {"gender": {"Male": 49, "Female": 51}}   # String / Categorical / Enum column
targets = {"gender": {1: 49, 2: 51}}               # numeric column
```

Text codes work on polars `String`, `Categorical` and `Enum` columns and on
pandas string and `category` columns, in target columns and in `by` columns
alike. They are coded to integers at the Arrow boundary, so the engine still
rakes on numbers and results are **bit-identical** to recoding the labels to
integers by hand.

Matching is literal: `'Male'` does not match `'male'` or `' Male'`, and an
unknown key raises `ValueError` listing the categories actually present.
Numeric columns still cost less — at 1M rows × 5 variables, integer codes run
~146 ms, `Enum` ~174 ms, `Categorical` ~205 ms and plain `String` ~308 ms — so
recoding to integers is still worth it for very large files.

### Combined categories (tuple keys)

A tuple key merges categories into **one cell** sharing a single target — the
standard way to handle sparse categories without recoding the data upstream:

```python
targets = {
    "gender": {1: 50, 2: 50},
    "education": {1: 33, 2: 24, 3: 33, (4, 5): 10},  # 4 and 5 together = 10%
}
```

Categories 4 and 5 receive one shared raking multiplier; how the 10% splits
between them follows the data (their relative sizes and the other dimensions).
This is exactly equivalent to recoding 4/5 into a single code before raking —
the manual workflow in Q or R's `survey` package — and rimpy produces
bit-identical weights to that manual pre-merge. Each category may appear in at
most one target key; overlapping keys raise `ValueError`.

Tuple members may be text as well, on a text column — `{("Low", "Mid"): 50}` —
but they must not mix labels with numbers, which raises `ValueError`.

### Unknown target keys raise

Any target key that doesn't exist in the data column raises `ValueError`
before raking (regardless of its target value). This catches the classic
Python trap where `{"education": {4-5: 14}}` silently evaluates the dict key
`4-5` to `-1` — the error message lists the column's real categories and
suggests the tuple syntax above. A code that exists in the column but is
missing from one group's slice (partial data) warns instead of raising.

### Converting from weightipy

```python
# weightipy format
weightipy_targets = {
    20230001: [
        {"gender": {1: 49.95, 2: 49.95, 3: 0.1}},
        {"age": {1: 32, 2: 37, 3: 31}},
    ],
}

# Convert to rimpy format
schemes = rim.convert_from_weightipy(weightipy_targets)
weighted = rim.rake_by_scheme(df, schemes, by="country_code")
```

## Special edge cases

RIM weighting has a handful of edge cases where rimpy's behavior is non-obvious
or diverges from professional weighting tools like Q, SPSS, and weightipy.
Notable examples include:

- **`target = 0` on a category that has respondents** — the literal interpretation
  ("weighted % must be 0") is ambiguous in the algorithm and tools disagree on
  how to handle it. rimpy's default refuses with an actionable error; opt-in
  modes (`hard_zero`, `near_zero`) cover the Q / SPSS / weightipy conventions.
- **Unknown target keys** — a key absent from the entire data column raises
  `ValueError` (it's a targets-dict bug, e.g. the `{4-5: 14}` → `{-1: 14}`
  arithmetic trap). A code present in the column but empty within one group's
  slice emits a `UserWarning` instead — that's normal partial data.
- **Combined categories** — tuple keys like `{(4, 5): 10}` merge categories
  into one cell; bit-identical to manually recoding before raking.

See **[`edge_cases.md`](edge_cases.md)** for the full treatment of each case,
the empirical comparison against Q's R-engine output, and recommendations on
which mode to use in production parallel-validation workflows.

## License

MIT
