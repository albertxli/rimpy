"""
Narwhals-based API for RIM weighting.

Supports both polars and pandas DataFrames transparently.
All data transfer uses Arrow PyCapsule — no Python lists in the data path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import narwhals as nw
from narwhals.typing import IntoFrameT

from ._engine import (
    RakeResult,
    rim_rake,
    rim_rake_by_scheme,
    rim_rake_grouped,
)

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
        boundary oscillation.

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

    Returns
    -------
    tuple
        (weighted_dataframe, RakeResult)
    """
    targets_dict = _normalize_targets(targets)
    target_columns = list(targets_dict.keys())

    df_nw = nw.from_native(df, eager_only=True)

    # Validate columns exist
    missing = set(target_columns) - set(df_nw.columns)
    if missing:
        raise KeyError(f"Target columns not found in DataFrame: {missing}")

    # Validate total
    if total is not None and total <= 0:
        raise ValueError(f"total must be positive, got {total}")

    # Single Rust call: Arrow in → Arrow out (with weight column appended)
    result_arrow, diagnostics = rim_rake(
        df_nw,
        target_columns,
        targets_dict,
        weight_column=weight_column,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        min_cap=min_cap,
        max_cap=max_cap,
        drop_nulls=drop_nulls,
        total=total,
        cap_correction=cap_correction,
    )

    # Arrow → narwhals → user's native backend
    result_df = nw.from_arrow(result_arrow, backend=nw.get_native_namespace(df_nw))
    return nw.to_native(result_df), diagnostics


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

    Parameters
    ----------
    df
        Input DataFrame.
    targets
        Target proportions (applied to each group).
    by
        Column(s) to group by.

    Returns
    -------
    DataFrame
        With weight column, weights computed within each group.

    Examples
    --------
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

    Returns
    -------
    tuple
        (weighted_dataframe, GroupedRakeResult)
    """
    if isinstance(by, str):
        by = [by]

    df_nw = nw.from_native(df, eager_only=True)
    targets_dict = _normalize_targets(targets)

    # Validate total
    if total is not None and total <= 0:
        raise ValueError(f"total must be positive, got {total}")

    # Single Rust call: full DataFrame + group columns → Arrow with weights
    result_arrow, group_diags_dict = rim_rake_grouped(
        df_nw,
        list(targets_dict.keys()),
        targets_dict,
        group_columns=by,
        weight_column=weight_column,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        min_cap=min_cap,
        max_cap=max_cap,
        drop_nulls=drop_nulls,
        total=total,
        cap_correction=cap_correction,
    )

    result_df = nw.from_arrow(result_arrow, backend=nw.get_native_namespace(df_nw))

    grouped_result = GroupedRakeResult(
        group_results=group_diags_dict,
    )

    return nw.to_native(result_df), grouped_result


@dataclass
class GroupedRakeResult:
    """Result of grouped raking with per-group diagnostics."""

    group_results: dict[Any, RakeResult]
    """Per-group weighting diagnostics."""

    def summary_df(self) -> dict[str, list]:
        """Return summary as dict suitable for DataFrame creation."""
        rows = {
            "group": [],
            "n_valid": [],
            "iterations": [],
            "converged": [],
            "efficiency": [],
            "weight_min": [],
            "weight_max": [],
            "weight_ratio": [],
        }
        for group, result in self.group_results.items():
            rows["group"].append(group)
            rows["n_valid"].append(result.n_valid)
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

    Parameters
    ----------
    df
        Input DataFrame (polars or pandas).
    schemes
        Dict mapping group values to their target schemes.
    by
        Column name to group by.
    default_scheme
        Fallback scheme for groups not in schemes dict.
    group_totals
        Optional global proportions for each group.
    total
        If set, scale weights so overall weighted sum equals this value.

    Returns
    -------
    DataFrame
        With weight column added.

    Examples
    --------
    >>> country_targets = {
    ...     "US": {"gender": {1: 49, 2: 51}, "age": {1: 20, 2: 30, 3: 30, 4: 20}},
    ...     "UK": {"gender": {1: 49, 2: 51}, "age": {1: 18, 2: 32, 3: 28, 4: 22}},
    ... }
    >>> weighted = rimpy.rake_by_scheme(df, country_targets, by="country")
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

    Returns
    -------
    tuple
        (weighted_dataframe, GroupedRakeResult)
    """
    df_nw = nw.from_native(df, eager_only=True)

    if by not in df_nw.columns:
        raise KeyError(f"Grouping column '{by}' not found in DataFrame")

    # Validate total
    if total is not None and total <= 0:
        raise ValueError(f"total must be positive, got {total}")

    # Normalize schemes: convert list-of-dicts targets to flat dicts
    normalized_schemes = {}
    for group_key, group_targets in schemes.items():
        normalized_schemes[group_key] = _normalize_targets(group_targets)

    # Normalize default_scheme
    normalized_default = None
    if default_scheme is not None:
        normalized_default = _normalize_targets(default_scheme)

    # Single Rust call: full DataFrame → Arrow with weights
    result_arrow, group_diags_dict = rim_rake_by_scheme(
        df_nw,
        by,
        normalized_schemes,
        default_scheme=normalized_default,
        weight_column=weight_column,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        min_cap=min_cap,
        max_cap=max_cap,
        drop_nulls=drop_nulls,
        group_totals=group_totals,
        total=total,
        cap_correction=cap_correction,
    )

    result_df = nw.from_arrow(result_arrow, backend=nw.get_native_namespace(df_nw))

    grouped_result = GroupedRakeResult(
        group_results=group_diags_dict,
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
        Column(s) to group by. If None, returns overall summary.

    Returns
    -------
    DataFrame
        Summary with n, effective_n, efficiency_pct, weight_mean, weight_std,
        weight_median, weight_min, weight_max, weight_ratio.
    """
    df_nw = nw.from_native(df, eager_only=True)

    w = nw.col(weight_col)
    sum_w = w.sum()
    sum_w_sq = (w ** 2).sum()
    n = nw.len()

    agg_exprs = [
        n.alias("n"),
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
        if col not in df_nw.columns:
            errors.append(f"Column '{col}' not found in DataFrame")
            continue

        unique_values = set(df_nw.get_column(col).unique().to_list())
        for code, target_value in props.items():
            if code not in unique_values and target_value != 0:
                warnings.append(f"Code {code} in targets for '{col}' not found in data")

        for val in unique_values:
            if val is not None and val not in props:
                warnings.append(f"Value {val} in column '{col}' has no target")

        total = sum(props.values())
        if total > 1.5:
            if round(total, 2) != 100:
                warnings.append(
                    f"Targets for '{col}' sum to {total}%, expected 100%"
                )
        else:
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

    Returns
    -------
    dict
        {"_global": {"errors": [...], "warnings": [...]},
         group_key: {"errors": [...], "warnings": [...]}, ...}
    """
    df_nw = nw.from_native(df, eager_only=True)

    result: dict[str, dict[str, list[str]]] = {
        "_global": {"errors": [], "warnings": []},
    }

    if by not in df_nw.columns:
        result["_global"]["errors"].append(
            f"Grouping column '{by}' not found in DataFrame"
        )
        return result

    unique_groups = set(df_nw.get_column(by).unique().to_list())

    for group_key in schemes.keys():
        if group_key not in unique_groups:
            result["_global"]["warnings"].append(
                f"Group '{group_key}' in schemes not found in data"
            )

    for group_val in unique_groups:
        if group_val is not None and group_val not in schemes:
            result["_global"]["warnings"].append(
                f"Group '{group_val}' in data has no scheme"
            )

    for group_key, targets in schemes.items():
        group_errors = []
        group_warnings = []

        targets = _normalize_targets(targets)

        if group_key not in unique_groups:
            result[group_key] = {"errors": group_errors, "warnings": group_warnings}
            continue

        if group_key is None:
            df_group = df_nw.filter(nw.col(by).is_null())
        else:
            df_group = df_nw.filter(nw.col(by) == group_key)

        for col, props in targets.items():
            if col not in df_nw.columns:
                group_errors.append(f"Column '{col}' not found in DataFrame")
                continue

            unique_values = set(df_group.get_column(col).unique().to_list())
            for code, target_value in props.items():
                if code not in unique_values and target_value != 0:
                    group_warnings.append(
                        f"Code {code} in targets for '{col}' not found in group data"
                    )

            for val in unique_values:
                if val is not None and val not in props:
                    group_warnings.append(
                        f"Value {val} in column '{col}' has no target"
                    )

            total = sum(props.values())
            if total > 1.5:
                if round(total, 2) != 100:
                    group_warnings.append(
                        f"Targets for '{col}' sum to {total}%, expected 100%"
                    )
            else:
                if round(total, 4) != 1.0:
                    group_warnings.append(
                        f"Targets for '{col}' sum to {total}, expected 1.0"
                    )

        result[group_key] = {"errors": group_errors, "warnings": group_warnings}

    return result
