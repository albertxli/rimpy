"""
Narwhals-based API for RIM weighting.

Supports both polars and pandas DataFrames transparently.
All data transfer uses Arrow PyCapsule — no Python lists in the data path.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

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


def _detect_empty_target_categories(
    df_nw: nw.DataFrame,
    targets: dict[str, dict[Any, float]],
) -> list[tuple[str, Any, float]]:
    """Return [(column, code, target_value), ...] for non-zero targets with no data rows.

    Type coercion: uses Python set membership, which treats numerically equal
    int/float as equal (1 in {1.0} is True).
    """
    empty: list[tuple[str, Any, float]] = []
    for col, props in targets.items():
        if col not in df_nw.columns:
            continue
        unique_values = set(df_nw.get_column(col).unique().to_list())
        for code, target_value in props.items():
            if code not in unique_values and target_value != 0:
                empty.append((col, code, target_value))
    return empty


# Valid values for the zero_target_policy kwarg.
_VALID_ZERO_TARGET_POLICIES: frozenset[str] = frozenset({"error", "hard_zero", "near_zero"})


def _detect_zero_target_on_populated_cell(
    df_nw: nw.DataFrame,
    targets: dict[str, dict[Any, float]],
) -> list[tuple[str, Any, int]]:
    """Return [(column, code, n_rows), ...] for target=0 on cells that have rows.

    Complement of _detect_empty_target_categories: that one catches
    "non-zero target + empty cell" (item #2). This one catches
    "zero target + populated cell" (item #3).
    """
    detected: list[tuple[str, Any, int]] = []
    for col, props in targets.items():
        if col not in df_nw.columns:
            continue
        col_series = df_nw.get_column(col)
        for code, target_value in props.items():
            if target_value != 0:
                continue
            if code is None:
                n_rows = int(col_series.is_null().sum())
            else:
                n_rows = int((col_series == code).sum())
            if n_rows > 0:
                detected.append((col, code, n_rows))
    return detected


def _format_zero_target_error(
    detected: list[tuple[str, Any, int]],
    context_label: str = "",
) -> str:
    """Build the actionable ValueError message for zero_target_policy='error'."""
    first_col, first_code, first_n = detected[0]
    prefix = f"{context_label}: " if context_label else ""
    lines = [
        f"{prefix}Target = 0 specified for {first_col} category {first_code!r} "
        f"with {first_n} non-empty respondents.",
        "",
        "rimpy requires explicit handling of zero targets on non-empty cells. Options:",
        "  - To exclude these respondents from weighting entirely:",
        "      drop them from the input DataFrame before calling rake()",
        '  - To force their weighted % to ~0 while keeping them in the file:',
        '      pass zero_target_policy="near_zero"',
        "  - To force weight=0 (matches Q / SPSS behavior):",
        '      pass zero_target_policy="hard_zero"',
        "",
        "See rimpy.rake docstring for the methodological tradeoffs.",
    ]
    if len(detected) > 1:
        lines.append("")
        lines.append("Additional zero-target cells detected:")
        for col, code, n in detected[1:]:
            lines.append(f"  - {col} = {code!r}: {n} respondent(s)")
    return "\n".join(lines)


def _validate_zero_target_kwargs(
    zero_target_policy: str,
    near_zero_eps: float,
) -> None:
    """Validate zero_target_policy and near_zero_eps at entry. Raises ValueError."""
    if zero_target_policy not in _VALID_ZERO_TARGET_POLICIES:
        raise ValueError(
            f"Unknown zero_target_policy={zero_target_policy!r}. "
            f"Expected one of {sorted(_VALID_ZERO_TARGET_POLICIES)}."
        )
    if near_zero_eps <= 0:
        raise ValueError(
            f"near_zero_eps must be > 0 (got {near_zero_eps}). "
            f"Zero or negative values defeat the purpose of zero_target_policy='near_zero'."
        )


def _apply_zero_target_policy(
    df_nw: nw.DataFrame,
    targets_dict: dict[str, dict[Any, float]],
    policy: str,
    near_zero_eps: float,
    *,
    context_label: str = "",
) -> tuple[nw.DataFrame, dict[str, dict[Any, float]], Any]:
    """Apply zero_target_policy. Returns (df_for_engine, modified_targets, keep_mask_or_None).

    - df_for_engine: original df for 'error'/'near_zero', filtered df for 'hard_zero'.
    - modified_targets: original for 'error', 0->eps for 'near_zero', zero entries removed for 'hard_zero'.
    - keep_mask_or_None: narwhals expression for the keep-mask when 'hard_zero' filters,
      else None. Caller uses this to identify dropped rows for reassembly.

    Raises ValueError when policy='error' and any zero-target-on-populated cell is detected.
    """
    detected = _detect_zero_target_on_populated_cell(df_nw, targets_dict)
    if not detected:
        return df_nw, targets_dict, None

    if policy == "error":
        raise ValueError(_format_zero_target_error(detected, context_label))

    if policy == "near_zero":
        new_targets = {col: dict(props) for col, props in targets_dict.items()}
        for col, code, _ in detected:
            # Match the scale of the other targets in this column. The Rust
            # engine (engine.rs:199-215) divides each target by 100 when the
            # column total > 1.5 (percentages → proportions). If the user
            # supplied percentages, express eps in the same units so that the
            # post-normalization effective eps still matches near_zero_eps.
            other_sum = sum(v for k, v in new_targets[col].items() if k != code)
            new_targets[col][code] = (
                near_zero_eps * 100.0 if other_sum > 1.5 else near_zero_eps
            )
        return df_nw, new_targets, None

    # policy == "hard_zero"
    keep_mask = None
    for col, code, _ in detected:
        cond = nw.col(col).is_null() if code is None else (nw.col(col) == code)
        this_keep = ~cond
        keep_mask = this_keep if keep_mask is None else keep_mask & this_keep
    df_filtered = df_nw.filter(keep_mask)
    new_targets = {col: dict(props) for col, props in targets_dict.items()}
    for col, code, _ in detected:
        new_targets[col].pop(code, None)
    return df_filtered, new_targets, keep_mask


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
    zero_target_policy: Literal["error", "hard_zero", "near_zero"] = "error",
    near_zero_eps: float = 1e-8,
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
    zero_target_policy
        How to handle ``target = 0`` when the matching cell has respondents.
        Setting a zero target on a populated cell is ambiguous (the user's
        intent could be exclude / show-as-zero / refuse), so by default rimpy
        forces an explicit decision.

        * ``"error"`` (default) — raise ``ValueError`` with the offending
          cells listed. Matches the R ``anesrake`` convention.
        * ``"hard_zero"`` — drop those respondents from the raking pass; they
          appear in the output with ``weight = 0`` and contribute nothing to
          weighted statistics. Matches Q / SPSS / svyweight.
        * ``"near_zero"`` — substitute ``0 → near_zero_eps`` in the targets
          before raking. Respondents stay in the file with weights very close
          to zero, so the weighted marginal is ~0% but the respondents are not
          discarded. Matches weightipy.
    near_zero_eps
        Epsilon used when ``zero_target_policy="near_zero"``. Default
        ``1e-8``. Must be > 0.

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

    Warnings
    --------
    Emits ``UserWarning`` for every ``(column, code)`` pair where a non-zero
    target proportion is supplied but the data has zero rows with that code
    (after ``drop_nulls``). The rake still runs; the unsatisfiable target is
    silently dropped by the engine.

    Raises
    ------
    ValueError
        When ``zero_target_policy='error'`` (default) and any target of 0 is
        supplied for a category that contains respondents. See the
        ``zero_target_policy`` parameter for resolution options.
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
        zero_target_policy=zero_target_policy,
        near_zero_eps=near_zero_eps,
        _warning_stacklevel=3,
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
    zero_target_policy: Literal["error", "hard_zero", "near_zero"] = "error",
    near_zero_eps: float = 1e-8,
    _warning_stacklevel: int = 2,
) -> tuple[IntoFrameT, RakeResult]:
    """
    Apply RIM weights and return diagnostics.

    Same as rake() but also returns RakeResult with diagnostics.

    Returns
    -------
    tuple
        (weighted_dataframe, RakeResult)

    Warnings
    --------
    Emits ``UserWarning`` for every ``(column, code)`` pair where a non-zero
    target proportion is supplied but the data has zero rows with that code
    (after ``drop_nulls``). The rake still runs; the unsatisfiable target is
    silently dropped by the engine.

    Raises
    ------
    ValueError
        When ``zero_target_policy='error'`` (default) and any target of 0 is
        supplied for a category that contains respondents. The error message
        lists the offending cells and the three resolution options.
    """
    _validate_zero_target_kwargs(zero_target_policy, near_zero_eps)

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

    # Item #2 — detect non-zero targets for codes that have zero rows after
    # null-filtering and warn (engine would silently drop them otherwise).
    df_nw_for_check = (
        df_nw.drop_nulls(subset=target_columns) if drop_nulls else df_nw
    )
    for col, code, val in _detect_empty_target_categories(df_nw_for_check, targets_dict):
        warnings.warn(
            f"Target code {code!r} for column '{col}' has zero rows in data; "
            f"target ({val}) will be silently dropped from raking.",
            UserWarning,
            stacklevel=_warning_stacklevel,
        )

    # Item #3 — apply zero-target policy: refuse, substitute eps, or pre-filter.
    df_for_engine, targets_dict, keep_mask = _apply_zero_target_policy(
        df_nw_for_check,
        targets_dict,
        zero_target_policy,
        near_zero_eps,
    )
    target_columns = list(targets_dict.keys())
    needs_reassembly = keep_mask is not None

    if needs_reassembly:
        # hard_zero filtered rows. Track original positions for reassembly post-rake.
        df_nw = df_nw.with_row_index("_rimpy_row_idx")
        df_for_engine = (
            df_nw.drop_nulls(subset=target_columns).filter(keep_mask)
            if drop_nulls
            else df_nw.filter(keep_mask)
        )

    # Single Rust call: Arrow in → Arrow out (with weight column appended)
    result_arrow, diagnostics = rim_rake(
        df_for_engine if needs_reassembly else df_nw,
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

    if needs_reassembly:
        # Reattach dropped rows with weight=0, restore original row order, strip idx col.
        # Pandas-backend Arrow round-trip introduces __index_level_*__ columns
        # that aren't present in df_nw; drop them to align schemas before concat.
        artifact_cols = [c for c in result_df.columns if c.startswith("__index_level_")]
        if artifact_cols:
            result_df = result_df.drop(*artifact_cols)
        dropped_df = df_nw.filter(~keep_mask).with_columns(
            nw.lit(0.0).alias(weight_column)
        )
        combined = nw.concat([result_df, dropped_df]).sort("_rimpy_row_idx")
        result_df = combined.drop("_rimpy_row_idx")

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
    zero_target_policy: Literal["error", "hard_zero", "near_zero"] = "error",
    near_zero_eps: float = 1e-8,
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
    zero_target_policy
        How to handle ``target = 0`` for a populated cell. See ``rake`` for the
        three options. In ``rake_by`` the targets dict is by construction
        uniform across groups, so any decision is applied globally to all
        groups' slices. Default ``"error"``.
    near_zero_eps
        Epsilon for ``zero_target_policy="near_zero"``. Default ``1e-8``.

    Returns
    -------
    DataFrame
        With weight column, weights computed within each group.

    Examples
    --------
    >>> targets = {"gender": {1: 50, 2: 50}, "age": {1: 30, 2: 40, 3: 30}}
    >>> weighted = rimpy.rake_by(df, targets, by="country")

    Warnings
    --------
    Emits ``UserWarning`` for every group × ``(column, code)`` triple where a
    non-zero target proportion is supplied but that group's data has zero rows
    with that code (after ``drop_nulls``). The rake still runs; the unsatisfiable
    target is silently dropped by the engine. Detection cost is
    ``O(groups × target_columns)`` — for ``by`` columns producing >100 distinct
    group keys, expect a few seconds of overhead on pandas backends.

    Raises
    ------
    ValueError
        When ``zero_target_policy='error'`` (default) and any target of 0 is
        supplied for a category that contains respondents anywhere in the data.
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
        zero_target_policy=zero_target_policy,
        near_zero_eps=near_zero_eps,
        _warning_stacklevel=3,
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
    zero_target_policy: Literal["error", "hard_zero", "near_zero"] = "error",
    near_zero_eps: float = 1e-8,
    _warning_stacklevel: int = 2,
) -> tuple[IntoFrameT, GroupedRakeResult]:
    """
    Apply RIM weights separately within groups and return diagnostics.

    Returns
    -------
    tuple
        (weighted_dataframe, GroupedRakeResult)

    Warnings
    --------
    Emits ``UserWarning`` for every group × ``(column, code)`` triple where a
    non-zero target proportion is supplied but that group's data has zero rows
    with that code (after ``drop_nulls``). Detection cost is
    ``O(groups × target_columns)``.

    Raises
    ------
    ValueError
        When ``zero_target_policy='error'`` (default) and any target of 0 is
        supplied for a category that contains respondents.
    """
    _validate_zero_target_kwargs(zero_target_policy, near_zero_eps)

    if isinstance(by, str):
        by = [by]

    df_nw = nw.from_native(df, eager_only=True)
    targets_dict = _normalize_targets(targets)
    target_col_names = list(targets_dict.keys())

    # Validate total
    if total is not None and total <= 0:
        raise ValueError(f"total must be positive, got {total}")

    # Per-group empty-target detection. A code may exist globally but be absent
    # from a specific group's slice — only per-group filtering catches that.
    group_combos = df_nw.unique(subset=by, maintain_order=True).select(by)
    for combo in group_combos.iter_rows(named=True):
        filter_expr = None
        for c in by:
            v = combo[c]
            cond = nw.col(c).is_null() if v is None else (nw.col(c) == v)
            filter_expr = cond if filter_expr is None else filter_expr & cond

        group_df = df_nw.filter(filter_expr)
        if drop_nulls:
            group_df = group_df.drop_nulls(subset=target_col_names)

        if len(by) == 1:
            group_label_str = repr(combo[by[0]])
        else:
            group_label_str = ", ".join(f"{c}={combo[c]!r}" for c in by)

        for col, code, val in _detect_empty_target_categories(group_df, targets_dict):
            warnings.warn(
                f"Group ({group_label_str}): target code {code!r} for column "
                f"'{col}' has zero rows in this group; target ({val}) will be "
                f"silently dropped.",
                UserWarning,
                stacklevel=_warning_stacklevel,
            )

    # Item #3 — zero-target policy. Targets are uniform across groups in rake_by,
    # so detection runs globally on the full dataframe (after null-filtering for
    # consistency with the engine's drop_nulls path).
    df_nw_for_check = (
        df_nw.drop_nulls(subset=target_col_names) if drop_nulls else df_nw
    )
    df_for_engine, targets_dict, keep_mask = _apply_zero_target_policy(
        df_nw_for_check,
        targets_dict,
        zero_target_policy,
        near_zero_eps,
    )
    target_col_names = list(targets_dict.keys())
    needs_reassembly = keep_mask is not None

    if needs_reassembly:
        df_nw = df_nw.with_row_index("_rimpy_row_idx")
        df_for_engine = (
            df_nw.drop_nulls(subset=target_col_names).filter(keep_mask)
            if drop_nulls
            else df_nw.filter(keep_mask)
        )

    # Single Rust call: full DataFrame + group columns → Arrow with weights
    result_arrow, group_diags_dict = rim_rake_grouped(
        df_for_engine if needs_reassembly else df_nw,
        target_col_names,
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

    if needs_reassembly:
        artifact_cols = [c for c in result_df.columns if c.startswith("__index_level_")]
        if artifact_cols:
            result_df = result_df.drop(*artifact_cols)
        dropped_df = df_nw.filter(~keep_mask).with_columns(
            nw.lit(0.0).alias(weight_column)
        )
        combined = nw.concat([result_df, dropped_df]).sort("_rimpy_row_idx")
        result_df = combined.drop("_rimpy_row_idx")

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
    zero_target_policy: Literal["error", "hard_zero", "near_zero"] = "error",
    near_zero_eps: float = 1e-8,
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
    zero_target_policy
        How to handle ``target = 0`` for a populated cell. See ``rake`` for
        the three options. Detection runs per-scheme — each group's scheme
        (or the ``default_scheme`` if the group lacks an explicit one) is
        checked against that group's data slice independently. Default
        ``"error"``.
    near_zero_eps
        Epsilon for ``zero_target_policy="near_zero"``. Default ``1e-8``.

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

    Warnings
    --------
    Emits ``UserWarning`` for every scheme group × ``(column, code)`` triple
    where a non-zero target proportion is supplied but that group's data has
    zero rows with that code (after ``drop_nulls``). Groups falling back to
    ``default_scheme`` are also checked. Detection cost is
    ``O(groups × target_columns)``.

    Raises
    ------
    ValueError
        When ``zero_target_policy='error'`` (default) and any scheme target
        of 0 is supplied for a category that contains respondents in that
        scheme's group.
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
        zero_target_policy=zero_target_policy,
        near_zero_eps=near_zero_eps,
        _warning_stacklevel=3,
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
    zero_target_policy: Literal["error", "hard_zero", "near_zero"] = "error",
    near_zero_eps: float = 1e-8,
    _warning_stacklevel: int = 2,
) -> tuple[IntoFrameT, GroupedRakeResult]:
    """
    Apply different weighting schemes to different groups with diagnostics.

    Returns
    -------
    tuple
        (weighted_dataframe, GroupedRakeResult)

    Warnings
    --------
    Emits ``UserWarning`` for every scheme group × ``(column, code)`` triple
    where a non-zero target proportion is supplied but that group's data has
    zero rows with that code (after ``drop_nulls``). Groups falling back to
    ``default_scheme`` are also checked.

    Raises
    ------
    ValueError
        When ``zero_target_policy='error'`` (default) and any scheme target
        of 0 is supplied for a category that contains respondents in that
        scheme's group.
    """
    _validate_zero_target_kwargs(zero_target_policy, near_zero_eps)

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

    # Per-group empty-target detection. Covers both groups with explicit schemes
    # and groups falling back to default_scheme.
    unique_groups = set(df_nw.get_column(by).unique().to_list())
    explicit_groups = set(normalized_schemes.keys()) & unique_groups
    fallback_groups = (
        (unique_groups - set(normalized_schemes.keys()))
        if normalized_default is not None
        else set()
    )

    def _check_scheme_group(
        group_key: Any,
        scheme_dict: dict[str, dict[Any, float]],
        scheme_label: str,
    ) -> None:
        if group_key is None:
            group_df = df_nw.filter(nw.col(by).is_null())
        else:
            group_df = df_nw.filter(nw.col(by) == group_key)
        if drop_nulls:
            group_df = group_df.drop_nulls(subset=list(scheme_dict.keys()))
        for col, code, val in _detect_empty_target_categories(group_df, scheme_dict):
            warnings.warn(
                f"{scheme_label} group ({group_key!r}): target code {code!r} for "
                f"column '{col}' has zero rows in this group; target ({val}) "
                f"will be silently dropped.",
                UserWarning,
                stacklevel=_warning_stacklevel,
            )

    for group_key in explicit_groups:
        _check_scheme_group(group_key, normalized_schemes[group_key], "Scheme")
    for group_key in fallback_groups:
        _check_scheme_group(group_key, normalized_default, "Default-scheme")

    # Item #3 — per-scheme zero-target policy. Each scheme is checked against
    # its own group's slice; default_scheme covers fallback groups.
    rows_to_drop_masks: list[Any] = []  # narwhals expressions, OR'd together for hard_zero

    def _apply_policy_to_scheme(
        group_key: Any,
        scheme_dict: dict[str, dict[Any, float]],
        scheme_label: str,
    ) -> dict[str, dict[Any, float]]:
        """Detect + apply policy to one (group, scheme) pair. Returns possibly-modified scheme."""
        if group_key is None:
            group_df = df_nw.filter(nw.col(by).is_null())
        else:
            group_df = df_nw.filter(nw.col(by) == group_key)
        if drop_nulls:
            group_df = group_df.drop_nulls(subset=list(scheme_dict.keys()))

        detected = _detect_zero_target_on_populated_cell(group_df, scheme_dict)
        if not detected:
            return scheme_dict

        if zero_target_policy == "error":
            raise ValueError(
                _format_zero_target_error(
                    detected, context_label=f"{scheme_label} group ({group_key!r})"
                )
            )

        if zero_target_policy == "near_zero":
            new_scheme = {col: dict(props) for col, props in scheme_dict.items()}
            for col, code, _ in detected:
                new_scheme[col][code] = near_zero_eps
            return new_scheme

        # hard_zero: build a row-drop mask for this group's offending codes
        group_match = (
            nw.col(by).is_null() if group_key is None else (nw.col(by) == group_key)
        )
        for col, code, _ in detected:
            code_match = nw.col(col).is_null() if code is None else (nw.col(col) == code)
            rows_to_drop_masks.append(group_match & code_match)
        # Strip zero-target entries from this scheme's targets
        new_scheme = {col: dict(props) for col, props in scheme_dict.items()}
        for col, code, _ in detected:
            new_scheme[col].pop(code, None)
        return new_scheme

    for group_key in explicit_groups:
        normalized_schemes[group_key] = _apply_policy_to_scheme(
            group_key, normalized_schemes[group_key], "Scheme"
        )
    if normalized_default is not None:
        # default_scheme is shared across all fallback groups but may need to
        # be customized per-fallback-group if hard_zero strips different codes.
        # Simpler: substitute/strip once based on the union of fallback detections.
        # For consistency, we still call _apply_policy_to_scheme per fallback group,
        # accumulating any near_zero substitutions into normalized_default.
        for group_key in fallback_groups:
            normalized_default = _apply_policy_to_scheme(
                group_key, normalized_default, "Default-scheme"
            )

    needs_reassembly = bool(rows_to_drop_masks)
    if needs_reassembly:
        # Union the per-group drop masks: a row is dropped if ANY mask matches.
        drop_mask = rows_to_drop_masks[0]
        for m in rows_to_drop_masks[1:]:
            drop_mask = drop_mask | m
        keep_mask = ~drop_mask
        df_nw = df_nw.with_row_index("_rimpy_row_idx")
        df_for_engine = df_nw.filter(keep_mask)
    else:
        df_for_engine = df_nw

    # Single Rust call: full DataFrame → Arrow with weights
    result_arrow, group_diags_dict = rim_rake_by_scheme(
        df_for_engine,
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

    if needs_reassembly:
        artifact_cols = [c for c in result_df.columns if c.startswith("__index_level_")]
        if artifact_cols:
            result_df = result_df.drop(*artifact_cols)
        dropped_df = df_nw.filter(drop_mask).with_columns(
            nw.lit(0.0).alias(weight_column)
        )
        combined = nw.concat([result_df, dropped_df]).sort("_rimpy_row_idx")
        result_df = combined.drop("_rimpy_row_idx")

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
