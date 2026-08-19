"""
Loader utilities for rimpy weighting targets.

A single loader, :func:`load_targets`, reads weighting targets from a
spreadsheet or delimited text file into the dict structure the rake
functions expect.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl

__all__ = ["load_targets"]

# Suffixes handled by pl.read_excel / pl.read_csv.
_EXCEL_SUFFIXES = frozenset({".xlsx", ".xlsm", ".xlsb", ".xls"})
_TEXT_SEPARATORS = {".csv": ",", ".tsv": "\t", ".txt": ","}

# Default split column name. Auto-detection (fall back to flat targets when the
# column is absent) only applies to this default — an explicitly named split
# column that is missing is a mistake, not a flat file.
_DEFAULT_SPLIT_COL = "split_value"

# Default combined-category column name. Same auto-detection rule as the split
# column: absent (or blank throughout) means nothing is combined.
_DEFAULT_COMBINE_COL = "combine"

# Column documenting which variable the split values belong to. Ignored when
# building the dict; only checked for consistency.
_SPLIT_VAR_COL = "split_var"

# Accepted per-variable target sums: percentages or proportions. engine.rs
# detects which by column sum (>1.5 means percentages), so both are valid.
_VALID_TOTALS = (100.0, 1.0)

# Cap on how many offending groups a single warning/error message lists.
_MAX_REPORTED = 10


def _read_source(
    source: str | Path | pl.DataFrame,
    sheet_name: str | int | None,
    separator: str | None,
) -> pl.DataFrame:
    """Read a path or pass through an existing polars DataFrame."""
    import polars as pl

    if isinstance(source, pl.DataFrame):
        return source
    if not isinstance(source, (str, Path)):
        raise TypeError(
            f"Expected a file path or polars DataFrame, got {type(source).__name__}."
        )

    path = Path(source)
    suffix = path.suffix.lower()

    if suffix in _EXCEL_SUFFIXES:
        try:
            if sheet_name is None:
                return pl.read_excel(path)
            if isinstance(sheet_name, int):
                # polars takes a 1-based sheet_id for positional selection.
                return pl.read_excel(path, sheet_id=sheet_name)
            return pl.read_excel(path, sheet_name=sheet_name)
        except ImportError as exc:
            raise ImportError(
                "Reading Excel target files needs an Excel engine. "
                "Install it with: pip install 'rimpy[excel]'"
            ) from exc

    if suffix in _TEXT_SEPARATORS:
        return pl.read_csv(path, separator=separator or _TEXT_SEPARATORS[suffix])

    raise ValueError(
        f"Unsupported file type: '{path.suffix}'. Expected one of "
        f"{sorted(_EXCEL_SUFFIXES | set(_TEXT_SEPARATORS))}."
    )


def _fmt_group(keys: tuple[Any, ...], names: list[str]) -> str:
    """Render one group key tuple for an error or warning message."""
    return ", ".join(f"{name}={key!r}" for name, key in zip(names, keys))


def _cell_key(codes: list[Any]) -> Any:
    """Build the target-dict key for one cell.

    A single code stays a scalar. Several codes become a tuple key (a
    combined-category target). Members are sorted so the key is canonical —
    rimpy treats ``(4, 5)`` and ``(5, 4)`` as the same merge, but Python dicts
    do not, so row order must not leak into the key. Unsortable mixtures fall
    back to file order and are rejected downstream by ``_normalize_tuple_key``.
    """
    if len(codes) == 1:
        return codes[0]
    try:
        return tuple(sorted(codes))
    except TypeError:
        return tuple(codes)


def _cell_id_expr(combine_col: str | None) -> pl.Expr:
    """One id per target cell: shared for combined rows, unique otherwise.

    Rows carrying the same tag fold into one cell; every other row becomes its
    own singleton cell, so combined and plain rows flow through one code path.
    The ``t``/``r`` prefixes keep tags and row ids from ever colliding.
    """
    import polars as pl

    row_id = pl.format("r{}", pl.int_range(pl.len()))
    if combine_col is None:
        return row_id

    tag = pl.col(combine_col).cast(pl.String).str.strip_chars()
    return (
        pl.when(tag.is_null() | (tag.str.len_chars() == 0))
        .then(row_id)
        .otherwise(pl.format("t{}", tag))
    )


def _warn_on_value_dtype(df: pl.DataFrame, value_col: str, stacklevel: int) -> None:
    """Warn when target codes need a second look before raking.

    Text codes rake fine — ``arrow_adapter.rs`` codes them at the Arrow layer —
    but matching against the DataFrame column is literal, so a case or
    whitespace difference silently becomes an unknown-key error later. Non-
    integral floats are truncated by the engine. Codes are always passed
    through unchanged.
    """
    import polars as pl

    dtype = df.schema[value_col]

    if dtype.is_integer():
        return

    if dtype.is_float():
        non_integral = (
            df.filter(pl.col(value_col) % 1 != 0)
            .get_column(value_col)
            .unique()
            .to_list()
        )
        if non_integral:
            shown = sorted(non_integral)[:_MAX_REPORTED]
            warnings.warn(
                f"Column '{value_col}' has non-integer category codes "
                f"({', '.join(repr(c) for c in shown)}). The rimpy engine reads "
                f"category codes as integers, so these will be truncated "
                f"(e.g. 1.5 becomes 1).",
                UserWarning,
                stacklevel=stacklevel,
            )
        return

    warnings.warn(
        f"Category codes in '{value_col}' are {dtype}. Double-check they match "
        f"the values in your DataFrame column exactly — matching is literal, so "
        f"'Male' will not match 'male', ' Male' or 1.",
        UserWarning,
        stacklevel=stacklevel,
    )


def load_targets(
    source: str | Path | pl.DataFrame,
    *,
    split_col: str | None = _DEFAULT_SPLIT_COL,
    var_col: str = "target_var",
    value_col: str = "target_value",
    pct_col: str = "target_pct",
    combine_col: str | None = _DEFAULT_COMBINE_COL,
    sheet_name: str | int | None = None,
    separator: str | None = None,
    validate: bool = True,
) -> dict[Any, dict[str, dict[Any, float]]] | dict[str, dict[Any, float]]:
    """
    Load weighting targets from a spreadsheet or delimited text file.

    Reads one row per target category and returns a plain dict ready for
    :func:`~rimpy.rake`, :func:`~rimpy.rake_by`, or
    :func:`~rimpy.rake_by_scheme`.

    Expected layout (one row per target category):

        split_var | split_value | split_label | target_var | target_value | target_label | target_pct
        country   | 101         | Bulgaria    | age        | 1            | 18-34        | 25
        country   | 101         | Bulgaria    | age        | 2            | 35-54        | 35
        country   | 101         | Bulgaria    | age        | 3            | 55+          | 40
        country   | 101         | Bulgaria    | gender     | 1            | Male         | 48.95
        country   | 102         | Croatia     | age        | 1            | 18-34        | 27
        ...

    Only ``split_value``, ``target_var``, ``target_value`` and ``target_pct``
    are read. ``split_var``, ``split_label`` and ``target_label`` are there so
    whoever fills in the sheet can see what the codes mean; they are ignored,
    as is any other extra column.

    Parameters
    ----------
    source
        File path (``.xlsx``, ``.xlsm``, ``.xlsb``, ``.xls``, ``.csv``,
        ``.tsv``, ``.txt``) or an existing polars DataFrame.
    split_col
        Column holding the split (group) key. Pass ``None`` to ignore any
        split column and pool every row into a single flat target dict.
    var_col
        Column holding the variable name — must match a DataFrame column name.
    value_col
        Column holding the category code.
    pct_col
        Column holding the target percentage.
    combine_col
        Optional column tagging rows that share one target cell. Rows carrying
        the same tag within the same ``(split_value, target_var)`` become a
        single combined-category (tuple-key) target. Blank means not combined.
        Pass ``None`` to ignore the column entirely.
    sheet_name
        Sheet name, or 1-based sheet position, for Excel sources. Defaults to
        the first sheet.
    separator
        Field separator for delimited text. Defaults to ``\\t`` for ``.tsv``
        and ``,`` otherwise.
    validate
        If True (default), warn when a variable's targets sum to neither 100
        (percentages) nor 1 (proportions).

    Returns
    -------
    dict
        **Nested** ``{split_value: {variable: {code: target_pct}}}`` when a
        split column is present — pass to
        ``rake_by_scheme(df, targets, by=...)``.

        **Flat** ``{variable: {code: target_pct}}`` when ``split_col=None`` or
        the file has no split column (or it is entirely blank) — pass to
        ``rake(df, targets)`` or ``rake_by(df, targets, by=...)``.

    Raises
    ------
    KeyError
        A required column is missing.
    ValueError
        Unsupported file type, empty file, nulls in a required column, a
        non-numeric ``target_pct``, or a duplicated (split, variable, code).
    ImportError
        Excel source with no Excel engine installed.

    Examples
    --------
    >>> # Split file -> rake_by_scheme
    >>> targets = rimpy.load_targets("weighting_targets.xlsx")
    >>> targets[101]["gender"]
    {1: 48.95, 2: 50.95, 3: 0.1}
    >>> weighted = rimpy.rake_by_scheme(df, targets, by="country")

    >>> # File with no split column -> rake
    >>> targets = rimpy.load_targets("national_targets.csv")
    >>> weighted = rimpy.rake(df, targets)

    >>> # Force flat, ignoring the split column
    >>> targets = rimpy.load_targets("weighting_targets.xlsx", split_col=None)

    Combined categories: tag the rows that share one target cell. The tag is
    arbitrary text; each distinct tag within a ``(split_value, target_var)``
    becomes one tuple key, so two merges on one variable need two tags::

        target_var  target_value  target_label  target_pct  combine
        education   1             Low           30          A
        education   2             Mid           30          A
        education   3             High          45          B
        education   4             Top           45          B
        education   5             Other         25

    >>> targets["education"]
    {(1, 2): 30.0, (3, 4): 45.0, 5: 25.0}

    The shared percentage may be repeated on every row of the group or entered
    once with the rest blank; two different values raise. Reusing one tag for
    both pairs above would instead produce a single ``(1, 2, 3, 4)`` cell.

    Notes
    -----
    - Types are preserved as polars read them. An Int64 ``split_value`` yields
      int keys, a String one yields str keys — match whatever the ``by``
      column holds in your data.
    - Category codes may be integers or text. Text codes match the DataFrame
      column literally, so they load with a warning to prompt a check for case
      and whitespace differences. Non-integral float codes are truncated by the
      engine.
    - Percentages and proportions both work; the engine detects which per
      variable. Values are never rescaled here.
    - Dict order follows file order, which is the order variables are raked in,
      so results are reproducible and under the sheet author's control.
    """
    import polars as pl

    df = _read_source(source, sheet_name, separator)

    # --- Decide flat vs nested -------------------------------------------
    nested = split_col is not None and split_col in df.columns
    if split_col is not None and split_col not in df.columns:
        # A split column the caller explicitly named must exist; only the
        # default name falls back to flat targets.
        if split_col != _DEFAULT_SPLIT_COL:
            raise KeyError(
                f"Split column '{split_col}' not found. Found columns: "
                f"{list(df.columns)}. Pass split_col=None for targets with no split."
            )
    if nested and df.get_column(split_col).null_count() == df.height:
        nested = False

    required = [var_col, value_col, pct_col]
    if nested:
        required = [split_col, *required]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. Found columns: {list(df.columns)}"
        )

    # --- Decide whether anything is combined ------------------------------
    if (
        combine_col is not None
        and combine_col not in df.columns
        and combine_col != _DEFAULT_COMBINE_COL
    ):
        raise KeyError(
            f"Combine column '{combine_col}' not found. Found columns: "
            f"{list(df.columns)}. Pass combine_col=None to ignore combining."
        )
    combining = combine_col is not None and combine_col in df.columns
    if combining and df.get_column(combine_col).null_count() == df.height:
        combining = False

    # Warn on a file mixing split variables — a nested dict has one split only.
    if nested and _SPLIT_VAR_COL in df.columns:
        split_vars = df.get_column(_SPLIT_VAR_COL).unique().drop_nulls().to_list()
        if len(split_vars) > 1:
            warnings.warn(
                f"Column '{_SPLIT_VAR_COL}' holds more than one split variable "
                f"({', '.join(repr(v) for v in sorted(map(str, split_vars)))}). "
                f"Targets are keyed by '{split_col}' alone, so split values "
                f"shared across variables will collide.",
                UserWarning,
                stacklevel=2,
            )

    df = df.select([*required, combine_col] if combining else required)

    # --- Row hygiene ------------------------------------------------------
    # Trailing blank rows are routine in spreadsheets; anything else is not.
    df = df.filter(~pl.all_horizontal([pl.col(c).is_null() for c in required]))

    if df.height == 0:
        raise ValueError("No target rows found — the source is empty.")

    # target_pct is checked per cell instead: a blank is legal on the trailing
    # rows of a combined group, which share one target with the first row.
    strict = [c for c in required if c != pct_col]
    null_counts = df.select(strict).null_count().row(0)
    nulls = [(c, n) for c, n in zip(strict, null_counts) if n]
    if nulls:
        detail = ", ".join(f"'{c}' ({n} row(s))" for c, n in nulls)
        raise ValueError(
            f"Null value(s) in required column(s): {detail}. Fill them in or "
            f"delete the rows."
        )

    try:
        df = df.with_columns(pl.col(pct_col).cast(pl.Float64, strict=True))
    except pl.exceptions.PolarsError as exc:
        raise ValueError(
            f"Column '{pct_col}' must be numeric, got dtype {df.schema[pct_col]}. "
            f"Remove any '%' signs, thousands separators or text."
        ) from exc

    _warn_on_value_dtype(df, value_col, stacklevel=3)

    # --- Group into cells, then into variables, converting in bulk --------
    # A "cell" is one target: normally one row, or several rows sharing a
    # combine tag. Grouping cells inside (split, var) is what scopes a tag to
    # its own variable. Both stages are vectorized; the only Python loop runs
    # once per cell, not once per row.
    group_keys = [split_col, var_col] if nested else [var_col]

    cells = (
        df.lazy()
        .with_columns(_cell_id_expr(combine_col if combining else None).alias("_cell"))
        .group_by([*group_keys, "_cell"], maintain_order=True)
        .agg(
            pl.col(value_col).alias("_codes"),
            pl.col(pct_col).drop_nulls().unique().alias("_pcts"),
        )
        .collect()
    )

    bad_pct = cells.filter(pl.col("_pcts").list.len() != 1)
    if bad_pct.height:
        shown = []
        for row in bad_pct.head(_MAX_REPORTED).iter_rows(named=True):
            where = _fmt_group(tuple(row[k] for k in group_keys), group_keys)
            found = row["_pcts"]
            detail = (
                f"no {pct_col}"
                if not found
                else f"conflicting {pct_col} values {sorted(found)}"
            )
            shown.append(f"{where}, codes {row['_codes']}: {detail}")
        extra = bad_pct.height - len(shown)
        raise ValueError(
            f"{bad_pct.height} target cell(s) without exactly one "
            f"{pct_col}:\n  "
            + "\n  ".join(shown)
            + (f"\n  ... and {extra} more" if extra > 0 else "")
            + f"\n  Rows sharing a '{combine_col}' tag share one {pct_col}: "
            f"repeat it on every row or enter it once and leave the rest blank."
        )

    variables = (
        cells.lazy()
        .with_columns(pl.col("_pcts").list.first().alias("_pct"))
        .group_by(group_keys, maintain_order=True)
        .agg(pl.col("_codes"), pl.col("_pct"))
        .collect()
    )

    # One bulk Arrow -> Python conversion per column.
    cell_codes = variables.get_column("_codes").to_list()
    cell_pcts = variables.get_column("_pct").to_list()
    var_names = variables.get_column(var_col).to_list()
    split_values = variables.get_column(split_col).to_list() if nested else None

    targets: dict[Any, Any] = {}
    duplicates: list[str] = []

    for i, (var, codes_per_cell, pcts_per_cell) in enumerate(
        zip(var_names, cell_codes, cell_pcts)
    ):
        props: dict[Any, float] = {}
        seen: set[Any] = set()
        dupes: list[Any] = []
        for codes, pct in zip(codes_per_cell, pcts_per_cell):
            for code in codes:
                if code in seen:
                    dupes.append(code)
                seen.add(code)
            props[_cell_key(codes)] = pct

        if dupes:
            keys = (split_values[i], var) if nested else (var,)
            duplicates.append(
                f"{_fmt_group(keys, group_keys)}: "
                f"{', '.join(repr(d) for d in dupes[:_MAX_REPORTED])}"
            )

        if nested:
            targets.setdefault(split_values[i], {})[var] = props
        else:
            targets[var] = props

    if duplicates:
        hint = (
            ""
            if nested
            else f" (targets are pooled across splits — drop split_col=None to key by '{_DEFAULT_SPLIT_COL}')"
        )
        shown = duplicates[:_MAX_REPORTED]
        extra = len(duplicates) - len(shown)
        raise ValueError(
            f"Duplicate category code(s) in {len(duplicates)} group(s){hint}:\n  "
            + "\n  ".join(shown)
            + (f"\n  ... and {extra} more" if extra > 0 else "")
        )

    # --- Validation -------------------------------------------------------
    # Sums over cells, so a combined cell counts once, not once per member.
    if validate:
        totals = variables.select(
            *group_keys, pl.col("_pct").list.sum().round(2).alias("_total")
        ).filter(~pl.col("_total").is_in(_VALID_TOTALS))

        if totals.height:
            shown = [
                f"{_fmt_group(row[:-1], group_keys)}: sums to {row[-1]:g}"
                for row in totals.head(_MAX_REPORTED).rows()
            ]
            extra = totals.height - len(shown)
            warnings.warn(
                f"{totals.height} variable(s) do not sum to 100% (or 1.0):\n  "
                + "\n  ".join(shown)
                + (f"\n  ... and {extra} more" if extra > 0 else ""),
                UserWarning,
                stacklevel=2,
            )

    return targets
