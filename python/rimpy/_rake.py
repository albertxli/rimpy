"""
Narwhals-based API for RIM weighting.

Supports both polars and pandas DataFrames transparently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import narwhals as nw
import numpy as np
from narwhals.typing import IntoFrameT

from ._engine import RakeResult, rim_iterate

if TYPE_CHECKING:
    from narwhals.typing import IntoFrame

__all__ = [
    "rake",
    "rake_by",
    "rake_by_with_diagnostics",
    "rake_by_scheme",
    "rake_by_scheme_with_diagnostics",
    "rake_with_diagnostics",
    "weight_summary",
    "validate_targets",
    "validate_schemes",
    "convert_from_weightipy",
    "RakeResult",
    "GroupedRakeResult",
]


def _extract_columns_to_numpy(
    df: nw.DataFrame,
    columns: list[str],
) -> dict[str, np.ndarray]:
    """Extract specified columns as numpy arrays."""
    result = {}
    for col in columns:
        # Get the column and convert to numpy
        series = df.get_column(col)
        result[col] = np.ascontiguousarray(series.to_numpy(), dtype=np.int64)
    return result


def _add_weight_column(
    df: nw.DataFrame,
    weights: np.ndarray,
    column_name: str,
) -> nw.DataFrame:
    """Add a numpy array as a new column to a narwhals DataFrame."""
    nw_series = nw.new_series(
        column_name,
        weights,
        dtype=nw.Float64,
        backend=nw.get_native_namespace(df),
    )
    return df.with_columns(nw_series)


def _normalize_targets(
    targets: dict[str, dict[Any, float]] | list[dict[str, dict[Any, float]]],
) -> dict[str, dict[Any, float]]:
    """
    Normalize targets to a consistent format.

    Accepts:
    - Dict: {"gender": {1: 49, 2: 51}, "age": {1: 20, 2: 30, ...}}
    - List of dicts (weightipy style): [{"gender": {1: 49, 2: 51}}, {"age": {...}}]
    """
    if isinstance(targets, list):
        # Weightipy-style list of single-key dicts
        result = {}
        for t in targets:
            result.update(t)
        return result
    return targets


def rake(
    df: IntoFrameT,
    targets: dict[str, dict[Any, float]] | list[dict[str, dict[Any, float]]],
    *,
    max_iterations: int = 1000,
    convergence_threshold: float = 0.01,
    min_cap: float | None = None,
    max_cap: float | None = None,
    weight_column: str = "weight",
    drop_nulls: bool = True,
    total: float | None = None,
    cap_correction: bool = True,
) -> IntoFrameT:
    """
    Apply RIM (raking) weights to a DataFrame.

    Parameters
    ----------
    df
        Input DataFrame (polars or pandas).
    targets
        Target proportions for each variable.
        Can be dict or list of dicts (weightipy-style).
        Values can be proportions (0-1) or percentages (0-100).
        Example: {"gender": {1: 49, 2: 51}, "age": {1: 20, 2: 30, 3: 30, 4: 20}}
    max_iterations
        Maximum iterations before stopping.
    convergence_threshold
        Convergence criterion (lower = stricter).
    min_cap
        Minimum allowed weight (optional).
    max_cap
        Maximum allowed weight (optional).
    weight_column
        Name for the weight column in output.
    drop_nulls
        If True, rows with nulls in target columns get weight=1.0.
    total
        If set, scale weights so the weighted sum of raked rows equals
        this value. Useful for population projection or controlled bases.
    cap_correction
        If True (default), add a small epsilon to caps to prevent
        boundary oscillation. Prevents weights from oscillating when they land exactly on the cap boundary.

    Returns
    -------
    DataFrame
        Original DataFrame with weight column added.
        Same type as input (polars in → polars out).

    Examples
    --------
    >>> import polars as pl
    >>> import rimpy
    >>> df = pl.DataFrame({"gender": [1, 1, 1, 2, 2], "age": [1, 2, 2, 1, 2]})
    >>> targets = {"gender": {1: 50, 2: 50}, "age": {1: 40, 2: 60}}
    >>> weighted = rimpy.rake(df, targets)
    >>> weighted["weight"]

    >>> # With controlled total
    >>> weighted = rimpy.rake(df, targets, total=1000)
    >>> weighted["weight"].sum()  # ≈ 1000
    """
    result_df, _ = rake_with_diagnostics(
        df,
        targets,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        min_cap=min_cap,
        max_cap=max_cap,
        weight_column=weight_column,
        drop_nulls=drop_nulls,
        total=total,
        cap_correction=cap_correction,
    )
    return result_df


def rake_with_diagnostics(
    df: IntoFrameT,
    targets: dict[str, dict[Any, float]] | list[dict[str, dict[Any, float]]],
    *,
    max_iterations: int = 1000,
    convergence_threshold: float = 0.01,
    min_cap: float | None = None,
    max_cap: float | None = None,
    weight_column: str = "weight",
    drop_nulls: bool = True,
    total: float | None = None,
    cap_correction: bool = True,
) -> tuple[IntoFrameT, RakeResult]:
    """
    Apply RIM weights and return diagnostics.

    Same as rake() but also returns RakeResult with diagnostics.

    Parameters
    ----------
    df
        Input DataFrame (polars or pandas).
    targets
        Target proportions for each variable.
    max_iterations
        Maximum iterations before stopping.
    convergence_threshold
        Convergence criterion (lower = stricter).
    min_cap
        Minimum allowed weight (optional).
    max_cap
        Maximum allowed weight (optional).
    weight_column
        Name for the weight column in output.
    drop_nulls
        If True, rows with nulls in target columns get weight=1.0.
    total
        If set, scale weights so the weighted sum of raked rows equals
        this value. Useful for population projection or controlled bases.
    cap_correction
        If True (default), add a small epsilon to caps to prevent
        boundary oscillation. Prevents weights from oscillating when they land exactly on the cap boundary.

    Returns
    -------
    tuple
        (weighted_dataframe, RakeResult)
    """
    # Normalize targets format
    targets_dict = _normalize_targets(targets)
    target_columns = list(targets_dict.keys())

    # Convert to narwhals
    df_nw = nw.from_native(df, eager_only=True)
    n_rows = len(df_nw)

    # Check columns exist
    missing = set(target_columns) - set(df_nw.columns)
    if missing:
        raise KeyError(f"Target columns not found in DataFrame: {missing}")

    # Handle nulls
    if drop_nulls:
        null_mask = df_nw.select(
            nw.any_horizontal(*[nw.col(c).is_null() for c in target_columns], ignore_nulls=True)
        ).to_numpy().flatten()
        valid_mask = ~null_mask
    else:
        valid_mask = np.ones(n_rows, dtype=bool)

    # Extract column data
    if valid_mask.all():
        column_data = _extract_columns_to_numpy(df_nw, target_columns)
        valid_indices = None
    else:
        valid_indices = np.where(valid_mask)[0]
        # Filter to non-null rows using the negated null expression
        df_valid = df_nw.filter(
            ~nw.any_horizontal(*[nw.col(c).is_null() for c in target_columns], ignore_nulls=True)
        )
        column_data = _extract_columns_to_numpy(df_valid, target_columns)

    # Run the core algorithm
    result = rim_iterate(
        column_data=column_data,
        targets=targets_dict,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        min_cap=min_cap,
        max_cap=max_cap,
        cap_correction=cap_correction,
    )

    # Build full weight array
    full_weights = np.ones(n_rows, dtype=np.float64)
    if valid_indices is not None:
        full_weights[valid_indices] = result.weights
    else:
        full_weights = result.weights

    # Scale to controlled total if requested
    if total is not None:
        if total <= 0:
            raise ValueError(f"total must be positive, got {total}")
        if valid_indices is not None:
            raked_sum = full_weights[valid_indices].sum()
            if raked_sum > 0:
                full_weights[valid_indices] *= total / raked_sum
        else:
            current_sum = full_weights.sum()
            if current_sum > 0:
                full_weights *= total / current_sum

    # Add weight column
    result_df = _add_weight_column(df_nw, full_weights, weight_column)

    # Update result with full weights (including null rows)
    result.weights = full_weights

    return nw.to_native(result_df), result


def rake_by(
    df: IntoFrameT,
    targets: dict[str, dict[Any, float]] | list[dict[str, dict[Any, float]]],
    by: str | list[str],
    *,
    max_iterations: int = 1000,
    convergence_threshold: float = 0.01,
    min_cap: float | None = None,
    max_cap: float | None = None,
    weight_column: str = "weight",
    drop_nulls: bool = True,
    total: float | None = None,
    cap_correction: bool = True,
) -> IntoFrameT:
    """
    Apply RIM weights separately within groups.

    Useful for weighting within segments (e.g., by country or region).

    Parameters
    ----------
    df
        Input DataFrame.
    targets
        Target proportions (applied to each group).
    by
        Column(s) to group by.
    **kwargs
        Passed to rake().

    Returns
    -------
    DataFrame
        With weight column, weights computed within each group.

    Examples
    --------
    >>> # Weight by gender/age within each country
    >>> targets = {"gender": {1: 50, 2: 50}, "age": {1: 30, 2: 40, 3: 30}}
    >>> weighted = rimpy.rake_by(df, targets, by="country")
    """
    result_df, _ = rake_by_with_diagnostics(
        df,
        targets,
        by,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        min_cap=min_cap,
        max_cap=max_cap,
        weight_column=weight_column,
        drop_nulls=drop_nulls,
        total=total,
        cap_correction=cap_correction,
    )
    return result_df


def rake_by_with_diagnostics(
    df: IntoFrameT,
    targets: dict[str, dict[Any, float]] | list[dict[str, dict[Any, float]]],
    by: str | list[str],
    *,
    max_iterations: int = 1000,
    convergence_threshold: float = 0.01,
    min_cap: float | None = None,
    max_cap: float | None = None,
    weight_column: str = "weight",
    drop_nulls: bool = True,
    total: float | None = None,
    cap_correction: bool = True,
) -> tuple[IntoFrameT, GroupedRakeResult]:
    """
    Apply RIM weights separately within groups and return diagnostics.

    Same as rake_by() but also returns GroupedRakeResult with per-group diagnostics.

    Returns
    -------
    tuple
        (weighted_dataframe, GroupedRakeResult)

    Examples
    --------
    >>> targets = {"gender": {1: 50, 2: 50}, "age": {1: 30, 2: 40, 3: 30}}
    >>> weighted, result = rimpy.rake_by_with_diagnostics(df, targets, by="country")
    >>> print(result.group_results["US"].efficiency)
    """
    if isinstance(by, str):
        by = [by]

    df_nw = nw.from_native(df, eager_only=True)
    targets_dict = _normalize_targets(targets)
    target_columns = list(targets_dict.keys())

    # Initialize weight column
    full_weights = np.ones(len(df_nw), dtype=np.float64)
    group_results: dict[Any, RakeResult] = {}

    # Add row index
    idx_col = "__rimpy_idx__"
    df_with_idx = df_nw.with_row_index(idx_col)

    # Get unique group combinations
    groups = df_nw.select(by).unique()

    # Process each group
    for row in groups.iter_rows(named=True):
        # Build group key (single value if one column, tuple if multiple)
        group_key = row[by[0]] if len(by) == 1 else tuple(row[c] for c in by)

        # Build filter expression
        filter_expr = nw.lit(True)
        for col, val in row.items():
            if val is None:
                filter_expr = filter_expr & nw.col(col).is_null()
            else:
                filter_expr = filter_expr & (nw.col(col) == val)

        # Filter to group
        df_group = df_with_idx.filter(filter_expr)

        if len(df_group) == 0:
            continue

        # Get original indices
        indices = df_group.get_column(idx_col).to_numpy()

        # Handle nulls in target columns
        if drop_nulls:
            null_mask = df_group.select(
                nw.any_horizontal(*[nw.col(c).is_null() for c in target_columns], ignore_nulls=True)
            ).to_numpy().flatten()
            valid_in_group = ~null_mask
        else:
            valid_in_group = np.ones(len(df_group), dtype=bool)

        if not valid_in_group.any():
            continue

        # Extract data and run
        if valid_in_group.all():
            column_data = _extract_columns_to_numpy(df_group, target_columns)
        else:
            valid_df = df_group.filter(
                ~nw.any_horizontal(*[nw.col(c).is_null() for c in target_columns], ignore_nulls=True)
            )
            column_data = _extract_columns_to_numpy(valid_df, target_columns)
            indices = valid_df.get_column(idx_col).to_numpy()

        result = rim_iterate(
            column_data=column_data,
            targets=targets_dict,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            min_cap=min_cap,
            max_cap=max_cap,
            cap_correction=cap_correction,
        )

        # Store diagnostics
        result.weights = result.weights.copy()
        group_results[group_key] = result

        # Assign weights back - indices are already the valid row indices
        full_weights[indices] = result.weights

    # Scale to controlled total if requested
    if total is not None:
        if total <= 0:
            raise ValueError(f"total must be positive, got {total}")
        current_sum = full_weights.sum()
        if current_sum > 0:
            full_weights *= total / current_sum

    # Add weight column
    result_df = _add_weight_column(df_nw, full_weights, weight_column)

    grouped_result = GroupedRakeResult(
        weights=full_weights,
        group_results=group_results,
    )

    return nw.to_native(result_df), grouped_result


@dataclass
class GroupedRakeResult:
    """Result of grouped raking with per-group diagnostics."""

    weights: np.ndarray
    """Weight factors as numpy array."""

    group_results: dict[Any, RakeResult]
    """Per-group weighting diagnostics."""

    def summary_df(self) -> dict[str, list]:
        """Return summary as dict suitable for DataFrame creation."""
        rows = {
            "group": [],
            "n": [],
            "iterations": [],
            "converged": [],
            "efficiency": [],
            "weight_min": [],
            "weight_max": [],
            "weight_ratio": [],
        }
        for group, result in self.group_results.items():
            rows["group"].append(group)
            rows["n"].append(len(result.weights))
            rows["iterations"].append(result.iterations)
            rows["converged"].append(result.converged)
            rows["efficiency"].append(round(result.efficiency, 2))
            rows["weight_min"].append(round(result.weight_min, 4))
            rows["weight_max"].append(round(result.weight_max, 4))
            rows["weight_ratio"].append(round(result.weight_ratio, 2))
        return rows


def rake_by_scheme(
    df: IntoFrameT,
    schemes: dict[Any, dict[str, dict[Any, float]]],
    by: str,
    *,
    max_iterations: int = 1000,
    convergence_threshold: float = 0.01,
    min_cap: float | None = None,
    max_cap: float | None = None,
    weight_column: str = "weight",
    drop_nulls: bool = True,
    default_scheme: dict[str, dict[Any, float]] | None = None,
    group_totals: dict[Any, float] | None = None,
    total: float | None = None,
    cap_correction: bool = True,
) -> IntoFrameT:
    """
    Apply different weighting schemes to different groups.

    Perfect for multi-country surveys where each country has different
    target variables (e.g., US weights by region, UK doesn't).

    Parameters
    ----------
    df
        Input DataFrame (polars or pandas).
    schemes
        Dict mapping group values to their target schemes.
        Example: {"US": {"gender": {...}, "age": {...}, "region": {...}},
                  "UK": {"gender": {...}, "age": {...}}}
    by
        Column name to group by (e.g., "country").
    max_iterations
        Maximum iterations per group.
    convergence_threshold
        Convergence criterion.
    min_cap
        Minimum weight (optional).
    max_cap
        Maximum weight (optional).
    weight_column
        Name for the weight column.
    drop_nulls
        Handle nulls in target columns (weight=1.0).
    default_scheme
        Optional fallback scheme for groups not in schemes dict.
        If None, groups not in schemes get weight=1.0.
    group_totals
        Optional global proportions for each group.
        If provided, adjusts group sizes after within-group weighting.
        Example: {"North": 40, "South": 60} means North=40%, South=60% of total.

    Returns
    -------
    DataFrame
        With weight column added.

    Examples
    --------
    >>> country_targets = {
    ...     "US": {
    ...         "gender": {1: 49, 2: 51},
    ...         "age": {1: 20, 2: 30, 3: 30, 4: 20},
    ...         "region": {1: 25, 2: 25, 3: 25, 4: 25},
    ...     },
    ...     "UK": {
    ...         "gender": {1: 49, 2: 51},
    ...         "age": {1: 18, 2: 32, 3: 28, 4: 22},
    ...         # No region weighting for UK
    ...     },
    ...     "DE": {
    ...         "gender": {1: 48, 2: 52},
    ...         "age": {1: 15, 2: 28, 3: 32, 4: 25},
    ...         "education": {1: 30, 2: 40, 3: 30},
    ...     },
    ... }
    >>> weighted = rimpy.rake_by_scheme(df, country_targets, by="country")
    
    >>> # With group totals (nested weighting)
    >>> weighted = rimpy.rake_by_scheme(
    ...     df,
    ...     schemes={"North": {"age": {...}, "gender": {...}}, "South": {...}},
    ...     by="region",
    ...     group_totals={"North": 40, "South": 60},
    ... )
    """
    result_df, _ = rake_by_scheme_with_diagnostics(
        df,
        schemes,
        by,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        min_cap=min_cap,
        max_cap=max_cap,
        weight_column=weight_column,
        drop_nulls=drop_nulls,
        default_scheme=default_scheme,
        group_totals=group_totals,
        total=total,
        cap_correction=cap_correction,
    )
    return result_df


def rake_by_scheme_with_diagnostics(
    df: IntoFrameT,
    schemes: dict[Any, dict[str, dict[Any, float]]],
    by: str,
    *,
    max_iterations: int = 1000,
    convergence_threshold: float = 0.01,
    min_cap: float | None = None,
    max_cap: float | None = None,
    weight_column: str = "weight",
    drop_nulls: bool = True,
    default_scheme: dict[str, dict[Any, float]] | None = None,
    group_totals: dict[Any, float] | None = None,
    total: float | None = None,
    cap_correction: bool = True,
) -> tuple[IntoFrameT, GroupedRakeResult]:
    """
    Apply different weighting schemes to different groups with diagnostics.

    Same as rake_by_scheme but returns per-group diagnostics.

    Parameters
    ----------
    group_totals
        Optional global proportions for each group.
        If provided, adjusts group sizes after within-group weighting.
    total
        If set, scale weights so the overall weighted sum equals this value.
        Applied after group_totals adjustment.
    cap_correction
        If True (default), add a small epsilon to caps to prevent
        boundary oscillation. Prevents weights from oscillating when they land exactly on the cap boundary.

    Returns
    -------
    tuple
        (weighted_dataframe, GroupedRakeResult)

    Examples
    --------
    >>> weighted, result = rimpy.rake_by_scheme_with_diagnostics(
    ...     df, country_targets, by="country"
    ... )
    >>> print(result.group_results["US"].efficiency)  # 87.5
    >>> print(result.summary_df())  # Dict for DataFrame
    """
    df_nw = nw.from_native(df, eager_only=True)
    n_rows = len(df_nw)

    if by not in df_nw.columns:
        raise KeyError(f"Grouping column '{by}' not found in DataFrame")

    full_weights = np.ones(n_rows, dtype=np.float64)
    group_results: dict[Any, RakeResult] = {}

    idx_col = "__rimpy_idx__"
    df_with_idx = df_nw.with_row_index(idx_col)

    unique_groups = df_nw.get_column(by).unique().to_numpy()

    for group_value in unique_groups:
        if group_value in schemes:
            targets = schemes[group_value]
        elif default_scheme is not None:
            targets = default_scheme
        else:
            # No scheme - record as skipped
            group_results[group_value] = RakeResult(
                weights=np.array([1.0]),
                iterations=0,
                converged=True,
                efficiency=100.0,
                weight_min=1.0,
                weight_max=1.0,
            )
            continue

        targets = _normalize_targets(targets)
        target_columns = list(targets.keys())

        missing = set(target_columns) - set(df_nw.columns)
        if missing:
            raise KeyError(
                f"Target columns {missing} for group '{group_value}' not found in DataFrame"
            )

        # Filter to group
        if group_value is None:
            df_group = df_with_idx.filter(nw.col(by).is_null())
        else:
            df_group = df_with_idx.filter(nw.col(by) == group_value)

        if len(df_group) == 0:
            continue
            
        # Get original indices
        indices = df_group.get_column(idx_col).to_numpy()

        if drop_nulls:
            null_mask = df_group.select(
                nw.any_horizontal(*[nw.col(c).is_null() for c in target_columns], ignore_nulls=True)
            ).to_numpy().flatten()
            valid_in_group = ~null_mask
        else:
            valid_in_group = np.ones(len(df_group), dtype=bool)

        if not valid_in_group.any():
            continue

        if valid_in_group.all():
            column_data = _extract_columns_to_numpy(df_group, target_columns)
        else:
            valid_df = df_group.filter(
                ~nw.any_horizontal(*[nw.col(c).is_null() for c in target_columns], ignore_nulls=True)
            )
            column_data = _extract_columns_to_numpy(valid_df, target_columns)
            indices = valid_df.get_column(idx_col).to_numpy()

        result = rim_iterate(
            column_data=column_data,
            targets=targets,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            min_cap=min_cap,
            max_cap=max_cap,
            cap_correction=cap_correction,
        )

        # Store diagnostics (copy weights since we reuse result.weights below)
        result.weights = result.weights.copy()
        group_results[group_value] = result

        # Assign weights - indices are already the valid row indices
        full_weights[indices] = result.weights

    # Apply group_totals adjustment if provided
    if group_totals is not None:
        # Normalize group_totals to proportions
        total_pct = sum(group_totals.values())
        if total_pct > 1.5:  # Percentages
            normalized_totals = {k: v / 100.0 for k, v in group_totals.items()}
        else:
            normalized_totals = dict(group_totals)

        # Build index mapping for groups
        idx_col = "__rimpy_idx__"
        df_with_idx = df_nw.with_row_index(idx_col)

        for group_value, target_prop in normalized_totals.items():
            if group_value is None:
                df_group = df_with_idx.filter(nw.col(by).is_null())
            else:
                df_group = df_with_idx.filter(nw.col(by) == group_value)

            if len(df_group) == 0:
                continue

            indices = df_group.get_column(idx_col).to_numpy()

            # Target weighted sum for this group
            target_sum = target_prop * n_rows

            # Current weighted sum for this group
            current_sum = full_weights[indices].sum()

            if current_sum > 0:
                correction = target_sum / current_sum
                full_weights[indices] *= correction

    # Scale to controlled total if requested (applied after group_totals)
    if total is not None:
        if total <= 0:
            raise ValueError(f"total must be positive, got {total}")
        current_sum = full_weights.sum()
        if current_sum > 0:
            full_weights *= total / current_sum

    result_df = _add_weight_column(df_nw, full_weights, weight_column)

    grouped_result = GroupedRakeResult(
        weights=full_weights,
        group_results=group_results,
    )

    return nw.to_native(result_df), grouped_result


def weight_summary(
    df: IntoFrameT,
    weight_col: str = "weight",
    by: str | list[str] | None = None,
) -> IntoFrameT:
    """
    Summarize weight diagnostics, optionally by group.

    Parameters
    ----------
    df
        DataFrame with weight column.
    weight_col
        Name of weight column.
    by
        Column(s) to group by (e.g., "country"). If None, returns overall summary.

    Returns
    -------
    DataFrame
        Summary with n, effective_n, efficiency_pct, weight_mean, weight_std,
        weight_median, weight_min, weight_max, weight_ratio.

    Examples
    --------
    >>> # Overall summary
    >>> rimpy.weight_summary(df, "weight")
    
    >>> # By country
    >>> rimpy.weight_summary(df, "weight", by="S0")
    """
    df_nw = nw.from_native(df, eager_only=True)
    
    w = nw.col(weight_col)
    sum_w = w.sum()
    sum_w_sq = (w ** 2).sum()
    n = nw.len()
    
    agg_exprs = [
        n.alias("n"),
        # sum_w.alias("sum_w"),  # Usually equals n if weights normalized
        ((sum_w ** 2) / sum_w_sq).alias("effective_n"),
        ((sum_w ** 2) / (n * sum_w_sq) * 100).alias("efficiency_pct"),
        w.mean().alias("weight_mean"),
        w.std().alias("weight_std"),
        w.median().alias("weight_median"),
        w.min().alias("weight_min"),
        w.max().alias("weight_max"),
        (w.max() / w.min()).alias("weight_ratio"),
    ]
    
    if by is None:
        # Overall summary - use select with aggregations
        result = df_nw.select(agg_exprs)
    else:
        if isinstance(by, str):
            by = [by]
        result = df_nw.group_by(by).agg(agg_exprs).sort(by)
    
    return nw.to_native(result)


def convert_from_weightipy(
    weightipy_targets: dict[Any, list[dict[str, dict[Any, float]]]],
) -> dict[Any, dict[str, dict[Any, float]]]:
    """
    Convert weightipy-style targets to rimpy scheme format.

    Weightipy uses: {group: [{"var1": {...}}, {"var2": {...}}]}
    rimpy uses:     {group: {"var1": {...}, "var2": {...}}}

    Parameters
    ----------
    weightipy_targets
        Dict mapping group keys to list of single-variable target dicts.

    Returns
    -------
    dict
        Schemes dict suitable for rake_by_scheme().

    Examples
    --------
    >>> weightipy_targets = {
    ...     20230001: [
    ...         {"gender": {1: 49.95, 2: 49.95, 3: 0.1}},
    ...         {"age": {1: 32, 2: 37, 3: 31}},
    ...     ],
    ...     20230002: [
    ...         {"gender": {1: 50, 2: 50}},
    ...         {"age": {1: 30, 2: 40, 3: 30}},
    ...     ],
    ... }
    >>> schemes = rimpy.convert_from_weightipy(weightipy_targets)
    >>> # Now use with rake_by_scheme
    >>> weighted = rimpy.rake_by_scheme(df, schemes, by="country_code")
    """
    schemes: dict[Any, dict[str, dict[Any, float]]] = {}
    for group_key, target_list in weightipy_targets.items():
        schemes[group_key] = {}
        for target_dict in target_list:
            for var_name, var_targets in target_dict.items():
                schemes[group_key][var_name] = var_targets
    return schemes


def validate_targets(
    df: IntoFrame,
    targets: dict[str, dict[Any, float]] | list[dict[str, dict[Any, float]]],
) -> dict[str, list[str]]:
    """
    Validate targets against a DataFrame.

    Checks:
    - All target columns exist (error)
    - All target codes exist in data (warning)
    - All data values have targets (warning)
    - Target proportions sum to exactly 100% (warning)

    Returns
    -------
    dict
        {"errors": [...], "warnings": [...]}
    """
    df_nw = nw.from_native(df, eager_only=True)
    targets_dict = _normalize_targets(targets)

    errors = []
    warnings = []

    for col, props in targets_dict.items():
        # Check column exists (blocker - will crash)
        if col not in df_nw.columns:
            errors.append(f"Column '{col}' not found in DataFrame")
            continue

        # Check all codes exist (informational - only warn if target is non-zero)
        unique_values = set(df_nw.get_column(col).unique().to_numpy())
        for code, target_value in props.items():
            if code not in unique_values and target_value != 0:
                warnings.append(f"Code {code} in targets for '{col}' not found in data")

        # Check codes in data have targets (informational - gets weight=1.0)
        for val in unique_values:
            if val is not None and val not in props:
                warnings.append(f"Value {val} in column '{col}' has no target")

        # Check proportions sum to exactly 100%
        total = sum(props.values())
        if total > 1.5:  # Percentages
            if round(total, 2) != 100:
                warnings.append(
                    f"Targets for '{col}' sum to {total}%, expected 100%"
                )
        else:  # Proportions
            if round(total, 4) != 1.0:
                warnings.append(
                    f"Targets for '{col}' sum to {total}, expected 1.0"
                )

    return {"errors": errors, "warnings": warnings}


def validate_schemes(
    df: IntoFrame,
    schemes: dict[Any, dict[str, dict[Any, float]]],
    by: str,
) -> dict[str, dict[str, list[str]]]:
    """
    Validate weighting schemes against a DataFrame.

    Checks for each group:
    - Grouping column exists (error)
    - Target columns exist (error)
    - Group key exists in data (warning)
    - All target codes exist in the group's data (warning)
    - All data values have targets (warning)
    - Target proportions sum to exactly 100% (warning)

    Parameters
    ----------
    df
        Input DataFrame.
    schemes
        Dict mapping group values to their target schemes.
        Example: {20230001: {"gender": {1: 50, 2: 50}, "age": {...}}, ...}
    by
        Column name to group by (e.g., "country_code").

    Returns
    -------
    dict
        {"_global": {"errors": [...], "warnings": [...]},
         group_key: {"errors": [...], "warnings": [...]}, ...}

    Examples
    --------
    >>> schemes = {
    ...     20230001: {"gender": {1: 50, 2: 50}, "age": {1: 30, 2: 40, 3: 30}},
    ...     20230002: {"gender": {1: 49, 2: 51}, "age": {1: 25, 2: 45, 3: 30}},
    ... }
    >>> report = rimpy.validate_schemes(df, schemes, by="country_code")
    >>> print(report["_global"]["errors"])  # Global issues
    >>> print(report[20230001]["warnings"])  # Group-specific issues
    """
    df_nw = nw.from_native(df, eager_only=True)

    result: dict[str, dict[str, list[str]]] = {
        "_global": {"errors": [], "warnings": []},
    }

    # Check grouping column exists (blocker - will crash)
    if by not in df_nw.columns:
        result["_global"]["errors"].append(
            f"Grouping column '{by}' not found in DataFrame"
        )
        return result

    # Get unique group values in data
    unique_groups = set(df_nw.get_column(by).unique().to_numpy())

    # Check for groups in schemes that don't exist in data (informational)
    for group_key in schemes.keys():
        if group_key not in unique_groups:
            result["_global"]["warnings"].append(
                f"Group '{group_key}' in schemes not found in data"
            )

    # Check for groups in data that don't have schemes (informational)
    for group_val in unique_groups:
        if group_val is not None and group_val not in schemes:
            result["_global"]["warnings"].append(
                f"Group '{group_val}' in data has no scheme"
            )

    # Validate each group's targets
    for group_key, targets in schemes.items():
        group_errors = []
        group_warnings = []

        targets = _normalize_targets(targets)

        # Skip validation if group doesn't exist in data
        if group_key not in unique_groups:
            result[group_key] = {"errors": group_errors, "warnings": group_warnings}
            continue

        # Filter to this group
        if group_key is None:
            df_group = df_nw.filter(nw.col(by).is_null())
        else:
            df_group = df_nw.filter(nw.col(by) == group_key)

        for col, props in targets.items():
            # Check column exists (blocker - will crash)
            if col not in df_nw.columns:
                group_errors.append(f"Column '{col}' not found in DataFrame")
                continue

            # Check all codes exist in this group's data (informational - only warn if target is non-zero)
            unique_values = set(df_group.get_column(col).unique().to_numpy())
            for code, target_value in props.items():
                if code not in unique_values and target_value != 0:
                    group_warnings.append(
                        f"Code {code} in targets for '{col}' not found in group data"
                    )

            # Check codes in group data have targets (informational)
            for val in unique_values:
                if val is not None and val not in props:
                    group_warnings.append(
                        f"Value {val} in column '{col}' has no target"
                    )

            # Check proportions sum to exactly 100%
            total = sum(props.values())
            if total > 1.5:  # Percentages
                if round(total, 2) != 100:
                    group_warnings.append(
                        f"Targets for '{col}' sum to {total}%, expected 100%"
                    )
            else:  # Proportions
                if round(total, 4) != 1.0:
                    group_warnings.append(
                        f"Targets for '{col}' sum to {total}, expected 1.0"
                    )

        result[group_key] = {"errors": group_errors, "warnings": group_warnings}

    return result
