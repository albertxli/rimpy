"""
Narwhals-based API for RIM weighting.

Supports both polars and pandas DataFrames transparently.
All data transfer uses Arrow PyCapsule — no Python lists in the data path.
"""

from __future__ import annotations

import hashlib
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
    *,
    unique_cache: dict[str, set] | None = None,
) -> list[tuple[str, Any, float]]:
    """Return [(column, code, target_value), ...] for non-zero targets with no data rows.

    Codes absent from the entire raw column raise in _validate_target_keys
    before this runs, so this only catches codes that exist in the column but
    have zero rows in the frame it is given — i.e. after null-dropping or
    within a group slice.

    ``unique_cache`` holds already-computed unique sets per column; only pass
    it when it was computed from the same frame as ``df_nw`` (avoids a second
    O(n) scan per column). Columns missing from the cache are scanned.

    Type coercion: uses Python set membership, which treats numerically equal
    int/float as equal (1 in {1.0} is True).
    """
    empty: list[tuple[str, Any, float]] = []
    for col, props in targets.items():
        if col not in df_nw.columns:
            continue
        if unique_cache is not None and col in unique_cache:
            unique_values = unique_cache[col]
        else:
            unique_values = set(df_nw.get_column(col).unique().to_list())
        for code, target_value in props.items():
            if code not in unique_values and target_value != 0:
                empty.append((col, code, target_value))
    return empty


_TUPLE_HINT = (
    "Hint: a negative key like -1 often comes from Python arithmetic inside a\n"
    "dict literal — {4-5: 14} is evaluated to {-1: 14} before rimpy sees it.\n"
    "To combine categories into one target cell, use tuple syntax: {(4, 5): 14}."
)


def _fmt_categories(values: list[Any]) -> str:
    """Render a column's category values for error messages (sorted, capped at 20)."""
    try:
        vals = sorted(values)
    except TypeError:
        vals = sorted(values, key=repr)
    shown = ", ".join(repr(v) for v in vals[:20])
    if len(vals) > 20:
        return f"[{shown}, ... ({len(vals)} total)]"
    return f"[{shown}]"


def _normalize_tuple_key(col: str, code: tuple) -> list[int]:
    """Validate one tuple key and return its members as ints. Raises ValueError."""
    if len(code) == 0:
        raise ValueError(f"Empty tuple key () in targets for column '{col}'.")
    members: list[int] = []
    for m in code:
        if not isinstance(m, (int, float)) or (
            isinstance(m, float) and not m.is_integer()
        ):
            raise ValueError(
                f"Tuple key {code!r} for column '{col}' contains non-integer "
                f"element {m!r}; tuple members must be integer category codes."
            )
        mi = int(m)
        if mi in members:
            raise ValueError(
                f"Tuple key {code!r} for column '{col}' contains duplicate member {mi}."
            )
        members.append(mi)
    return members


def _validate_target_keys(
    df_nw: nw.DataFrame,
    targets_dict: dict[str, dict[Any, float]],
    *,
    context_label: str = "",
    unique_cache: dict[str, set] | None = None,
) -> list[tuple[str, tuple, int]]:
    """Validate every target key against the full raw column (before null-drop).

    Raises ValueError for: unknown scalar keys (regardless of target value —
    catches the {4-5: 14} → {-1: 14} arithmetic trap), tuples whose members are
    ALL absent, structurally invalid tuples, and overlapping keys. Raises
    KeyError if a column carrying tuple keys is missing (cannot expand).
    Missing columns with scalar-only keys are skipped (the caller's own
    column check handles them).

    ``unique_cache`` (mutated in place) memoizes each column's unique-value
    set so repeated calls — per-scheme validation, and the downstream
    empty-category check when no rows were null-dropped — cost one O(n) scan
    per column total.

    Returns [(col, tuple_key, absent_member), ...] for tuples where some but
    not all members are absent — the caller warns on these.
    """
    prefix = f"{context_label}: " if context_label else ""
    offender_lines: list[str] = []
    add_hint = False
    absent_members: list[tuple[str, tuple, int]] = []

    for col, props in targets_dict.items():
        if col not in df_nw.columns:
            if any(isinstance(c, tuple) for c in props):
                raise KeyError(
                    f"Target column '{col}' (with tuple keys) not found in DataFrame"
                )
            continue
        if unique_cache is not None and col in unique_cache:
            unique_values = unique_cache[col]
        else:
            unique_values = set(df_nw.get_column(col).unique().to_list())
            if unique_cache is not None:
                unique_cache[col] = unique_values
        non_null = [v for v in unique_values if v is not None]
        claimed: dict[int, Any] = {}

        def _claim(code_int: int, key: Any) -> None:
            if code_int in claimed:
                other = claimed[code_int]
                if isinstance(key, tuple) and isinstance(other, tuple):
                    raise ValueError(
                        f"{prefix}Overlapping tuple keys for column '{col}': "
                        f"{other!r} and {key!r} both claim category {code_int}. "
                        f"Each category may appear in at most one target key."
                    )
                scalar, tup = (other, key) if isinstance(key, tuple) else (key, other)
                raise ValueError(
                    f"{prefix}Conflicting target keys for column '{col}': "
                    f"category {code_int} appears both as scalar key {scalar!r} "
                    f"and inside tuple key {tup!r}."
                )
            claimed[code_int] = key

        for code in props:
            if isinstance(code, tuple):
                members = _normalize_tuple_key(col, code)
                for m in members:
                    _claim(m, code)
                present = [m for m in members if m in unique_values]
                if not present:
                    offender_lines.append(
                        f"  - {prefix}column '{col}': tuple key {code!r} — none of "
                        f"its members exist. Existing categories: {_fmt_categories(non_null)}"
                    )
                else:
                    for m in members:
                        if m not in unique_values:
                            absent_members.append((col, code, m))
            else:
                if isinstance(code, (int, float)) and float(code).is_integer():
                    _claim(int(code), code)
                if code not in unique_values:
                    offender_lines.append(
                        f"  - {prefix}column '{col}': key {code!r}. "
                        f"Existing categories: {_fmt_categories(non_null)}"
                    )
                    if (
                        isinstance(code, (int, float))
                        and code < 0
                        and all(isinstance(v, (int, float)) and v >= 0 for v in non_null)
                    ):
                        add_hint = True

    if offender_lines:
        msg = "Unknown target key(s) not present in the data:\n" + "\n".join(
            offender_lines
        )
        if add_hint:
            msg += "\n\n" + _TUPLE_HINT
        raise ValueError(msg)
    return absent_members


def _expand_tuple_targets(
    df_nw: nw.DataFrame,
    targets_dict: dict[str, dict[Any, float]],
    *,
    reusable: set[str] | None = None,
) -> tuple[nw.DataFrame, dict[str, dict[Any, float]], dict[str, tuple[str, dict[int, str]]], list[str]]:
    """Expand tuple keys into a temporary merged column per (column, pattern).

    Assumes _validate_target_keys passed. For each column with multi-member
    tuple keys, adds a recoded column where every tuple's members map to the
    tuple's smallest member (the canonical code — min so that (4,5) and (5,4)
    agree), and rewrites that column's targets to reference the temp column
    with scalar keys, preserving key positions (engine iteration order).
    1-tuples collapse to scalar keys in place. Raking on the temp column is
    bit-identical to raking on a manually pre-merged column.

    ``reusable`` names temp columns created earlier in the same call chain
    (rake_by_scheme expands per scheme); an identical (column, pattern) reuses
    them. Any other pre-existing column with a reserved name raises.

    Returns (df_with_temp_cols, rewritten_targets, merge_labels, new_temp_cols)
    where merge_labels = {temp_col: (orig_col, {canonical: "(4, 5)"})} for
    rendering user-facing messages. No tuples -> inputs unchanged, ({}, []).
    """
    if not any(
        isinstance(code, tuple) for props in targets_dict.values() for code in props
    ):
        return df_nw, targets_dict, {}, []

    reusable = reusable or set()
    merge_labels: dict[str, tuple[str, dict[int, str]]] = {}
    new_temp_cols: list[str] = []
    new_targets: dict[str, dict[Any, float]] = {}

    for col, props in targets_dict.items():
        merges = sorted(
            sorted({int(m) for m in code})
            for code in props
            if isinstance(code, tuple) and len(code) > 1
        )
        if not merges:
            if any(isinstance(c, tuple) for c in props):
                # Only 1-tuples: collapse to scalars, no temp column needed.
                new_targets[col] = {
                    (int(c[0]) if isinstance(c, tuple) else c): v
                    for c, v in props.items()
                }
            else:
                new_targets[col] = props
            continue

        pattern = ";".join(",".join(map(str, members)) for members in merges)
        digest = hashlib.sha1(pattern.encode()).hexdigest()[:8]
        temp_name = f"_rimpy_merged_{col}_{digest}"

        if temp_name in df_nw.columns:
            if temp_name not in reusable:
                raise ValueError(
                    f"Column name '{temp_name}' is reserved by rimpy for internal "
                    f"category merging; rename it in the input DataFrame."
                )
        else:
            expr: Any = nw.col(col)
            for members in merges:
                expr = (
                    nw.when(nw.col(col).is_in(members))
                    .then(nw.lit(members[0]))
                    .otherwise(expr)
                )
            df_nw = df_nw.with_columns(expr.alias(temp_name))
            new_temp_cols.append(temp_name)

        new_props: dict[Any, float] = {}
        code_labels: dict[int, str] = {}
        for code, val in props.items():
            if isinstance(code, tuple) and len(code) > 1:
                canonical = min(int(m) for m in code)
                new_props[canonical] = val
                code_labels[canonical] = repr(code)
            elif isinstance(code, tuple):
                new_props[int(code[0])] = val
            else:
                new_props[code] = val
        new_targets[temp_name] = new_props
        merge_labels[temp_name] = (col, code_labels)

    return df_nw, new_targets, merge_labels, new_temp_cols


def _fmt_target(
    col: str,
    code: Any,
    merge_labels: dict[str, tuple[str, dict[int, str]]],
) -> tuple[str, str]:
    """Map an (engine column, code) pair back to user-facing display names."""
    if col in merge_labels:
        orig_col, code_labels = merge_labels[col]
        return orig_col, code_labels.get(code, repr(code))
    return col, repr(code)


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
    merge_labels: dict[str, tuple[str, dict[int, str]]] | None = None,
) -> str:
    """Build the actionable ValueError message for zero_target_policy='error'."""
    merge_labels = merge_labels or {}
    first_col, first_code, first_n = detected[0]
    first_col, first_code_str = _fmt_target(first_col, first_code, merge_labels)
    prefix = f"{context_label}: " if context_label else ""
    lines = [
        f"{prefix}Target = 0 specified for {first_col} category {first_code_str} "
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
            disp_col, disp_code = _fmt_target(col, code, merge_labels)
            lines.append(f"  - {disp_col} = {disp_code}: {n} respondent(s)")
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
    merge_labels: dict[str, tuple[str, dict[int, str]]] | None = None,
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
        raise ValueError(_format_zero_target_error(detected, context_label, merge_labels))

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

        A tuple key combines categories into ONE merged cell sharing a single
        target: ``{"education": {1: 33, 2: 24, 3: 33, (4, 5): 10}}`` weights
        categories 4 and 5 together to 10% of the total (one shared raking
        multiplier; the 4-vs-5 split inside the 10% follows the data). This is
        exactly equivalent to recoding 4/5 into one code before raking, as
        done manually in Q or R's survey package. Every category may appear
        in at most one key.
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
    target proportion is supplied for a code that exists in the column but has
    zero rows after ``drop_nulls``, and for tuple members with zero rows. The
    rake still runs; the unsatisfiable target is silently dropped by the engine.

    Raises
    ------
    ValueError
        When any target key is absent from the data column entirely (unknown
        keys are dict bugs — e.g. ``{4-5: 14}`` is Python arithmetic and
        reaches rimpy as ``{-1: 14}``; use tuple syntax ``{(4, 5): 14}`` to
        combine categories). Also when tuple keys are malformed or overlap,
        and when ``zero_target_policy='error'`` (default) and any target of 0
        is supplied for a category that contains respondents.
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
    target proportion is supplied for a code that exists in the column but has
    zero rows after ``drop_nulls``, and for tuple members with zero rows. The
    rake still runs; the unsatisfiable target is silently dropped by the engine.

    Raises
    ------
    ValueError
        When any target key is absent from the data column entirely, when
        tuple keys are malformed or overlap, or when
        ``zero_target_policy='error'`` (default) and any target of 0 is
        supplied for a category that contains respondents. The error message
        lists the offending cells and the resolution options.
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

    # Item #4 — reject keys absent from the raw column, then expand tuple keys
    # into a temporary merged column (dropped from the output at the end).
    unique_cache: dict[str, set] = {}
    absent_members = _validate_target_keys(df_nw, targets_dict, unique_cache=unique_cache)
    for col, tup, member in absent_members:
        warnings.warn(
            f"Tuple member {member!r} of key {tup!r} for column '{col}' has "
            f"zero rows in the data; the merged cell is carried by its other member(s).",
            UserWarning,
            stacklevel=_warning_stacklevel,
        )
    df_nw, targets_dict, merge_labels, temp_cols = _expand_tuple_targets(df_nw, targets_dict)
    target_columns = list(targets_dict.keys())

    # Item #2 — detect non-zero targets for codes that have zero rows after
    # null-filtering and warn (engine would silently drop them otherwise).
    df_nw_for_check = (
        df_nw.drop_nulls(subset=target_columns) if drop_nulls else df_nw
    )
    # The cache is only valid for the check frame if null-dropping removed
    # nothing; temp columns aren't cached and fall back to a scan either way.
    cache_for_check = unique_cache if len(df_nw_for_check) == len(df_nw) else None
    for col, code, val in _detect_empty_target_categories(
        df_nw_for_check, targets_dict, unique_cache=cache_for_check
    ):
        disp_col, disp_code = _fmt_target(col, code, merge_labels)
        warnings.warn(
            f"Target code {disp_code} for column '{disp_col}' has zero rows in data; "
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
        merge_labels=merge_labels,
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

    if temp_cols:
        result_df = result_df.drop(*[c for c in temp_cols if c in result_df.columns])

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
        When any target key is absent from the data column entirely (a code
        missing only within one group's slice warns instead), when tuple keys
        are malformed or overlap, or when ``zero_target_policy='error'``
        (default) and any target of 0 is supplied for a category that contains
        respondents anywhere in the data.
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
        When any target key is absent from the data column entirely (a code
        missing only within one group's slice warns instead), when tuple keys
        are malformed or overlap, or when ``zero_target_policy='error'``
        (default) and any target of 0 is supplied for a category that contains
        respondents.
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

    # Item #4 — targets are shared across groups, so key validation and tuple
    # expansion run once against the full column.
    absent_members = _validate_target_keys(df_nw, targets_dict)
    for col, tup, member in absent_members:
        warnings.warn(
            f"Tuple member {member!r} of key {tup!r} for column '{col}' has "
            f"zero rows in the data; the merged cell is carried by its other member(s).",
            UserWarning,
            stacklevel=_warning_stacklevel,
        )
    df_nw, targets_dict, merge_labels, temp_cols = _expand_tuple_targets(df_nw, targets_dict)
    target_col_names = list(targets_dict.keys())

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
            disp_col, disp_code = _fmt_target(col, code, merge_labels)
            warnings.warn(
                f"Group ({group_label_str}): target code {disp_code} for column "
                f"'{disp_col}' has zero rows in this group; target ({val}) will be "
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
        merge_labels=merge_labels,
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

    if temp_cols:
        result_df = result_df.drop(*[c for c in temp_cols if c in result_df.columns])

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
        When any scheme's target key is absent from the data column entirely
        (a code missing only within that scheme's group slice warns instead),
        when tuple keys are malformed or overlap within a scheme, or when
        ``zero_target_policy='error'`` (default) and any scheme target
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
        When any scheme's target key is absent from the data column entirely
        (a code missing only within that scheme's group slice warns instead),
        when tuple keys are malformed or overlap within a scheme, or when
        ``zero_target_policy='error'`` (default) and any scheme target
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

    # Item #4 — per-scheme key validation (against the full column) and tuple
    # expansion. Schemes sharing the same (column, merge pattern) reuse one
    # temp column; distinct patterns get distinct temp columns, each scheme's
    # rewritten targets referencing its own.
    merge_labels: dict[str, tuple[str, dict[int, str]]] = {}
    all_temp_cols: list[str] = []
    unique_cache: dict[str, set] = {}

    def _validate_and_expand(
        scheme_dict: dict[str, dict[Any, float]],
        label: str,
    ) -> dict[str, dict[Any, float]]:
        nonlocal df_nw
        for col, tup, member in _validate_target_keys(
            df_nw, scheme_dict, context_label=label, unique_cache=unique_cache
        ):
            warnings.warn(
                f"{label}: tuple member {member!r} of key {tup!r} for column "
                f"'{col}' has zero rows in the data; the merged cell is carried "
                f"by its other member(s).",
                UserWarning,
                stacklevel=_warning_stacklevel + 1,
            )
        df_nw, expanded, labels, new_cols = _expand_tuple_targets(
            df_nw, scheme_dict, reusable=set(all_temp_cols)
        )
        merge_labels.update(labels)
        all_temp_cols.extend(new_cols)
        return expanded

    for group_key in list(normalized_schemes.keys()):
        normalized_schemes[group_key] = _validate_and_expand(
            normalized_schemes[group_key], f"Scheme group ({group_key!r})"
        )
    if normalized_default is not None:
        normalized_default = _validate_and_expand(normalized_default, "Default-scheme")

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
            disp_col, disp_code = _fmt_target(col, code, merge_labels)
            warnings.warn(
                f"{scheme_label} group ({group_key!r}): target code {disp_code} for "
                f"column '{disp_col}' has zero rows in this group; target ({val}) "
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
                    detected,
                    context_label=f"{scheme_label} group ({group_key!r})",
                    merge_labels=merge_labels,
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

    if all_temp_cols:
        result_df = result_df.drop(
            *[c for c in all_temp_cols if c in result_df.columns]
        )

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
        tuple_covered: set[Any] = set()
        for code, target_value in props.items():
            if isinstance(code, tuple):
                tuple_covered.update(code)
                if (
                    not any(m in unique_values for m in code)
                    and target_value != 0
                ):
                    warnings.append(
                        f"Code {code} in targets for '{col}' not found in data"
                    )
            elif code not in unique_values and target_value != 0:
                warnings.append(f"Code {code} in targets for '{col}' not found in data")

        for val in unique_values:
            if val is not None and val not in props and val not in tuple_covered:
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
            tuple_covered: set[Any] = set()
            for code, target_value in props.items():
                if isinstance(code, tuple):
                    tuple_covered.update(code)
                    if (
                        not any(m in unique_values for m in code)
                        and target_value != 0
                    ):
                        group_warnings.append(
                            f"Code {code} in targets for '{col}' not found in group data"
                        )
                elif code not in unique_values and target_value != 0:
                    group_warnings.append(
                        f"Code {code} in targets for '{col}' not found in group data"
                    )

            for val in unique_values:
                if val is not None and val not in props and val not in tuple_covered:
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
