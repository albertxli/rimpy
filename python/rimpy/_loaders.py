"""
Loader utilities for rimpy weighting schemes.

Functions to load weighting targets from various formats into
the nested dict structure expected by rake_by_scheme().
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl

__all__ = ["load_schemes", "load_schemes_wide"]


def load_schemes(
    source: str | Path | pl.DataFrame,
    *,
    key_col: str = "scheme_key",
    var_col: str = "target_var",
    code_col: str = "target_code",
    target_col: str = "target_pct",
    sheet_name: str | int | None = None,
    validate: bool = True,
) -> dict[Any, dict[str, dict[Any, float]]]:
    """
    Load weighting schemes from long-format table.

    Converts a table of targets into nested dict format for rake_by_scheme().

    Parameters
    ----------
    source
        File path (xlsx, csv) or existing polars DataFrame.
    key_col
        Column containing scheme identifiers (e.g., country codes).
    var_col
        Column containing variable names (e.g., "gender", "smoker").
    code_col
        Column containing category codes (e.g., 1, 2 or "Male", "Female").
    target_col
        Column containing target percentages.
    sheet_name
        Sheet name or index if source is Excel file.
        If None (default), reads the first sheet.
    validate
        If True, warn when targets don't sum to exactly 100 for any (scheme, variable).

    Returns
    -------
    dict
        Nested dict: {scheme_key: {variable: {code: target_pct}}}
        Ready for use with rake_by_scheme(df, schemes=...).

    Examples
    --------
    >>> # From Excel file
    >>> schemes = rimpy.load_schemes("targets.xlsx")
    >>> weighted = rimpy.rake_by_scheme(df, schemes, by="country_code")

    >>> # From existing DataFrame
    >>> targets_df = pl.read_csv("targets.csv")
    >>> schemes = rimpy.load_schemes(targets_df)

    >>> # Custom column names
    >>> schemes = rimpy.load_schemes(
    ...     "targets.xlsx",
    ...     key_col="country_id",
    ...     var_col="variable",
    ...     code_col="code",
    ...     target_col="pct",
    ... )

    >>> # Specific sheet
    >>> schemes = rimpy.load_schemes("targets.xlsx", sheet_name="Wave1")

    Expected input format:

        scheme_key | target_var | target_code | target_pct
        20230001   | gender     | 1           | 49.85
        20230001   | gender     | 2           | 49.85
        20230001   | gender     | 3           | 0.3
        20230001   | smoker     | 1           | 21
        20230001   | smoker     | 2           | 79
        20240001   | gender     | 1           | 49.9
        ...

    Notes
    -----
    - Types are preserved as-is. Ensure key_col values match your `by` column
      and code_col values match your actual data values.
    - Input row order doesn't matter—data is grouped internally.
    """
    import polars as pl

    # Read source into polars DataFrame
    if isinstance(source, (str, Path)):
        source = Path(source)
        if source.suffix in (".xlsx", ".xls"):
            if sheet_name is None:
                df = pl.read_excel(source)
            else:
                df = pl.read_excel(source, sheet_name=sheet_name)
        elif source.suffix == ".csv":
            df = pl.read_csv(source)
        else:
            raise ValueError(f"Unsupported file type: {source.suffix}")
    elif isinstance(source, pl.DataFrame):
        df = source
    else:
        raise TypeError(f"Expected file path or polars DataFrame, got {type(source)}")

    # Validate required columns exist
    required_cols = {key_col, var_col, code_col, target_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Sort for consistent processing
    df = df.sort(key_col, var_col, code_col)

    # Build nested dict
    schemes: dict[Any, dict[str, dict[Any, float]]] = {}

    for row in df.iter_rows(named=True):
        key = row[key_col]
        var = row[var_col]
        code = row[code_col]
        pct = row[target_col]

        schemes.setdefault(key, {}).setdefault(var, {})[code] = float(pct)

    # Validate targets sum to exactly 100
    if validate:
        for scheme_key, variables in schemes.items():
            for var_name, targets in variables.items():
                total = sum(targets.values())
                # Round to 2 decimal places to handle floating point
                if round(total, 2) != 100.0:
                    warnings.warn(
                        f"Scheme '{scheme_key}', variable '{var_name}': "
                        f"targets sum to {total:.2f}%, expected 100%",
                        UserWarning,
                        stacklevel=2,
                    )

    return schemes


def load_schemes_wide(
    source: str | Path | pl.DataFrame,
    *,
    var_col: str = "target_var",
    code_col: str = "target_code",
    sheet_name: str | int | None = None,
    validate: bool = True,
) -> dict[Any, dict[str, dict[Any, float]]]:
    """
    Load weighting schemes from wide-format table.

    Converts a table where scheme keys are columns into nested dict format
    for rake_by_scheme().

    Parameters
    ----------
    source
        File path (xlsx, csv) or existing polars DataFrame.
    var_col
        Column containing variable names (e.g., "gender", "smoker").
    code_col
        Column containing category codes (e.g., 1, 2 or "Male", "Female").
    sheet_name
        Sheet name or index if source is Excel file.
        If None (default), reads the first sheet.
    validate
        If True, warn when targets don't sum to exactly 100 for any (scheme, variable).

    Returns
    -------
    dict
        Nested dict: {scheme_key: {variable: {code: target_pct}}}
        Ready for use with rake_by_scheme(df, schemes=...).

    Examples
    --------
    >>> # From Excel file
    >>> schemes = rimpy.load_schemes_wide("targets.xlsx")
    >>> weighted = rimpy.rake_by_scheme(df, schemes, by="country_code")

    >>> # Custom column names
    >>> schemes = rimpy.load_schemes_wide(
    ...     "targets.xlsx",
    ...     var_col="variable",
    ...     code_col="code",
    ... )

    Expected input format:

        target_var | target_code | 20230001 | 20240001 | 20230002 | ...
        gender     | 1           | 49.85    | 49.9     | 49.9     | ...
        gender     | 2           | 49.85    | 49.9     | 49.9     | ...
        gender     | 3           | 0.3      | 0.2      | 0.2      | ...
        smoker     | 1           | 21       | 9        | 10       | ...
        smoker     | 2           | 79       | 91       | 90       | ...
        ...

    Notes
    -----
    - All columns except var_col and code_col are treated as scheme keys.
    - Column names become scheme keys (type preserved from DataFrame).
    - Types are preserved as-is. Ensure scheme key columns match your `by` column
      and code_col values match your actual data values.
    """
    import polars as pl

    # Read source into polars DataFrame
    if isinstance(source, (str, Path)):
        source = Path(source)
        if source.suffix in (".xlsx", ".xls"):
            if sheet_name is None:
                df = pl.read_excel(source)
            else:
                df = pl.read_excel(source, sheet_name=sheet_name)
        elif source.suffix == ".csv":
            df = pl.read_csv(source)
        else:
            raise ValueError(f"Unsupported file type: {source.suffix}")
    elif isinstance(source, pl.DataFrame):
        df = source
    else:
        raise TypeError(f"Expected file path or polars DataFrame, got {type(source)}")

    # Validate required columns exist
    required_cols = {var_col, code_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Identify scheme key columns (all columns except var_col and code_col)
    scheme_cols = [c for c in df.columns if c not in (var_col, code_col)]

    if not scheme_cols:
        raise ValueError("No scheme key columns found. Expected columns beyond var_col and code_col.")

    # Unpivot to long format
    df_long = df.unpivot(
        index=[var_col, code_col],
        on=scheme_cols,
        variable_name="scheme_key",
        value_name="target_pct",
    )

    # Try to cast scheme_key to numeric if all values are numeric strings
    # This handles the case where Excel column names like "20230001" are read as strings
    try:
        df_long = df_long.with_columns(pl.col("scheme_key").cast(pl.Int64))
    except pl.exceptions.InvalidOperationError:
        # Keep as string if cast fails
        pass

    # Sort for consistent processing
    df_long = df_long.sort("scheme_key", var_col, code_col)

    # Build nested dict
    schemes: dict[Any, dict[str, dict[Any, float]]] = {}

    for row in df_long.iter_rows(named=True):
        key = row["scheme_key"]
        var = row[var_col]
        code = row[code_col]
        pct = row["target_pct"]

        schemes.setdefault(key, {}).setdefault(var, {})[code] = float(pct)

    # Validate targets sum to exactly 100
    if validate:
        for scheme_key, variables in schemes.items():
            for var_name, targets in variables.items():
                total = sum(targets.values())
                # Round to 2 decimal places to handle floating point
                if round(total, 2) != 100.0:
                    warnings.warn(
                        f"Scheme '{scheme_key}', variable '{var_name}': "
                        f"targets sum to {total:.2f}%, expected 100%",
                        UserWarning,
                        stacklevel=2,
                    )

    return schemes
