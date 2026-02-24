# rimpy

<p align="center">
  <img src="images/rimpy-banner-v3.svg" alt="rimpy banner" width="100%">
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
pip install rimpy[all]     # For both polars and pandas
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
    convergence_threshold=0.01,  # convergence criterion
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

print(result.converged)      # True/False
print(result.iterations)     # Number of iterations
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

## Loading Schemes from Files

### `load_schemes(source, **options)`

Load weighting schemes from a **long-format** table.

```python
schemes = rim.load_schemes("targets.xlsx")
weighted = rim.rake_by_scheme(df, schemes, by="country_code")

# Custom column names
schemes = rim.load_schemes(
    "targets.xlsx",
    key_col="country_id",
    var_col="variable",
    code_col="code",
    target_col="pct",
    sheet_name="Wave1",
)
```

Expected input format:

| scheme_key | target_var | target_code | target_pct |
|------------|------------|-------------|------------|
| 20230001   | gender     | 1           | 49.85      |
| 20230001   | gender     | 2           | 49.85      |
| 20230001   | gender     | 3           | 0.3        |
| 20230001   | smoker     | 1           | 21         |
| 20230001   | smoker     | 2           | 79         |

### `load_schemes_wide(source, **options)`

Load weighting schemes from a **wide-format** table.

```python
schemes = rim.load_schemes_wide("targets.xlsx")
weighted = rim.rake_by_scheme(df, schemes, by="country_code")
```

Expected input format:

| target_var | target_code | 20230001 | 20240001 | 20230002 |
|------------|-------------|----------|----------|----------|
| gender     | 1           | 49.85    | 49.9     | 49.9     |
| gender     | 2           | 49.85    | 49.9     | 49.9     |
| gender     | 3           | 0.3      | 0.2      | 0.2      |
| smoker     | 1           | 21       | 9        | 10       |
| smoker     | 2           | 79       | 91       | 90       |

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

## License

MIT
