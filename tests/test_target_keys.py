"""
Tests for item #4: target-key validation and tuple keys (combined categories).

Feature A — unknown keys raise: any target key absent from the full raw column
raises ValueError before raking, regardless of target value. Catches the
Python arithmetic-dict-key trap ({4-5: 14} silently becomes {-1: 14}).

Feature B — tuple keys: {"education": {(4, 5): 14}} merges categories 4 and 5
into one cell targeting 14% of the weighted total. Implemented as a temporary
recoded column, so a tuple-key run must be bit-identical to raking on a
manually pre-merged column (the operation R survey / Q users perform by hand).

Run with: pytest tests/test_target_keys.py -v
"""

import warnings

import pandas as pd
import polars as pl
import pytest

import rimpy


# ---------------------------------------------------------------------------
# Fixtures — 320-row education distribution with a sparse tail (codes 4, 5)
# ---------------------------------------------------------------------------


@pytest.fixture
def edu_df_polars():
    """320 rows: education 1-5 (4 and 5 sparse), gender 1-2, country US/UK."""
    cells = [
        # (education, n_us_male, n_us_female, n_uk_male, n_uk_female)
        (1, 28, 27, 25, 25),   # 105
        (2, 23, 22, 21, 21),   #  87
        (3, 27, 28, 25, 25),   # 105
        (4, 4, 3, 3, 2),       #  12
        (5, 3, 3, 3, 2),       #  11
    ]
    rows = []
    for edu, us_m, us_f, uk_m, uk_f in cells:
        rows += [{"country": "US", "gender": 1, "education": edu}] * us_m
        rows += [{"country": "US", "gender": 2, "education": edu}] * us_f
        rows += [{"country": "UK", "gender": 1, "education": edu}] * uk_m
        rows += [{"country": "UK", "gender": 2, "education": edu}] * uk_f
    assert len(rows) == 320
    return pl.DataFrame(rows)


@pytest.fixture
def edu_df_pandas(edu_df_polars):
    return pd.DataFrame(
        {col: edu_df_polars[col].to_list() for col in edu_df_polars.columns}
    )


@pytest.fixture(params=["polars", "pandas"])
def edu_df(request, edu_df_polars, edu_df_pandas):
    return edu_df_polars if request.param == "polars" else edu_df_pandas


TUPLE_TARGETS = {
    "gender": {1: 50.0, 2: 50.0},
    "education": {1: 33.0, 2: 24.0, 3: 33.0, (4, 5): 10.0},
}


def _as_polars(df) -> pl.DataFrame:
    return df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)


def _premerged(df_polars: pl.DataFrame, members: list[int], canonical: int) -> pl.DataFrame:
    """Manually pre-merge education codes, the way an R survey / Q user would."""
    return df_polars.with_columns(
        pl.when(pl.col("education").is_in(members))
        .then(canonical)
        .otherwise(pl.col("education"))
        .alias("edu_merged")
    )


# ---------------------------------------------------------------------------
# Feature A — unknown-key ValueError
# ---------------------------------------------------------------------------


class TestUnknownKeyValidation:
    def test_arithmetic_typo_raises_with_tuple_hint(self, edu_df):
        """{4-5: 14} evaluates to {-1: 14}; the error names the key, lists the
        real categories, and suggests tuple syntax."""
        targets = {"education": {1: 33.0, 2: 24.0, 3: 33.0, 4 - 5: 10.0}}
        with pytest.raises(ValueError, match=r"Unknown target key") as excinfo:
            rimpy.rake(edu_df, targets)
        msg = str(excinfo.value)
        assert "column 'education': key -1" in msg
        assert "[1, 2, 3, 4, 5]" in msg
        assert "tuple syntax: {(4, 5): 14}" in msg

    def test_unknown_key_with_zero_target_is_dropped(self, edu_df_polars):
        """A zero target on an absent category is a satisfied no-op: dropped
        with a warning naming the code, so a typo is still visible."""
        targets = {"education": {1: 33.0, 2: 24.0, 3: 33.0, 4: 5.0, 5: 5.0, -1: 0.0}}
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake(edu_df_polars, targets)
        assert any(
            "Dropped zero target" in str(w.message) and "-1" in str(w.message)
            for w in recs
        )

    def test_hint_absent_when_data_has_negative_codes(self, edu_df_polars):
        """The arithmetic-trap hint only fires when the data has no negative
        codes; a column legitimately containing -1 gets a plain error."""
        df = edu_df_polars.with_columns(
            pl.when(pl.col("education") == 5).then(-1).otherwise(pl.col("education")).alias("education")
        )
        targets = {"education": {1: 33.0, 2: 24.0, 3: 31.0, 4: 5.0, -1: 5.0, -2: 2.0}}
        with pytest.raises(ValueError, match=r"Unknown target key") as excinfo:
            rimpy.rake(df, targets)
        assert "tuple syntax" not in str(excinfo.value)

    def test_multiple_unknown_keys_aggregated_in_one_error(self, edu_df_polars):
        targets = {
            "gender": {1: 45.0, 2: 50.0, 9: 5.0},
            "education": {1: 33.0, 2: 24.0, 3: 31.0, 4: 5.0, 5: 5.0, -1: 2.0},
        }
        with pytest.raises(ValueError) as excinfo:
            rimpy.rake(edu_df_polars, targets)
        msg = str(excinfo.value)
        assert "column 'gender': key 9" in msg
        assert "column 'education': key -1" in msg

    @pytest.mark.parametrize("policy", ["hard_zero", "near_zero"])
    def test_unknown_key_raises_before_zero_target_policy(self, edu_df_polars, policy):
        """Key validation runs before zero_target_policy — an unknown key
        raises instead of being absorbed by the policy."""
        targets = {"education": {1: 33.0, 2: 24.0, 3: 31.0, 4: 5.0, 5: 5.0, -1: 2.0}}
        with pytest.raises(ValueError, match=r"Unknown target key"):
            rimpy.rake(edu_df_polars, targets, zero_target_policy=policy)

    def test_string_key_clean_error_not_pyo3(self, edu_df_polars):
        """A string key against an int column raises the rimpy ValueError, not
        an opaque PyO3 extraction error from the FFI layer."""
        with pytest.raises(ValueError, match=r"Unknown target key"):
            rimpy.rake(edu_df_polars, {"education": {"a": 50.0, 1: 50.0}})

    def test_none_key_valid_iff_column_has_nulls(self, edu_df_polars):
        """{None: 0} works on a column with nulls (existing hard_zero workflow).
        On a null-free column the None key is a zero target for a category with
        no rows, so it is dropped with a warning — which also keeps it away from
        the FFI, where a None key used to crash."""
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake(
                edu_df_polars,
                {"gender": {1: 50.0, 2: 50.0, None: 0.0}},
                zero_target_policy="hard_zero",
            )
        assert any("Dropped zero target" in str(w.message) for w in recs)

        df_with_nulls = edu_df_polars.with_columns(
            pl.when(pl.col("education") == 5).then(None).otherwise(pl.col("gender")).alias("gender")
        )
        result = rimpy.rake(
            df_with_nulls,
            {"gender": {1: 50.0, 2: 50.0, None: 0.0}},
            zero_target_policy="hard_zero",
            drop_nulls=False,
        )
        assert "weight" in result.columns

    def test_float_key_numeric_equality(self, edu_df_polars):
        """{1.0: 50} on an int column passes (hash(1) == hash(1.0));
        a non-integer float key like 4.5 is unknown and raises."""
        result = rimpy.rake(edu_df_polars, {"gender": {1.0: 50.0, 2.0: 50.0}})
        assert "weight" in result.columns

        with pytest.raises(ValueError, match=r"Unknown target key"):
            rimpy.rake(edu_df_polars, {"gender": {1.0: 50.0, 2.0: 40.0, 4.5: 10.0}})

    def test_rake_by_global_absence_raises_group_absence_warns(self, edu_df_polars):
        """Absent from the entire column → raise; absent from one group's
        slice only → per-group warning (item #2 contract preserved)."""
        # education=9 exists nowhere → raise
        with pytest.raises(ValueError, match=r"Unknown target key"):
            rimpy.rake_by(
                edu_df_polars,
                {"education": {1: 33.0, 2: 24.0, 3: 33.0, 9: 10.0}},
                by="country",
            )

        # education=5 removed from UK only → still present globally → warn
        df = edu_df_polars.filter(
            ~((pl.col("country") == "UK") & (pl.col("education") == 5))
        )
        targets = {"education": {1: 33.0, 2: 24.0, 3: 33.0, 4: 5.0, 5: 5.0}}
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            result = rimpy.rake_by(df, targets, by="country")
        msgs = [str(w.message) for w in recs if issubclass(w.category, UserWarning)]
        assert any("Group ('UK')" in m and "code 5" in m for m in msgs)
        assert "weight" in result.columns

    def test_rake_by_scheme_error_identifies_scheme(self, edu_df_polars):
        schemes = {
            "US": {"gender": {1: 50.0, 2: 50.0}},
            "UK": {"gender": {1: 45.0, 2: 50.0, 9: 5.0}},
        }
        with pytest.raises(ValueError, match=r"Scheme group \('UK'\)"):
            rimpy.rake_by_scheme(edu_df_polars, schemes, by="country")

        with pytest.raises(ValueError, match=r"Default-scheme"):
            rimpy.rake_by_scheme(
                edu_df_polars,
                {"US": {"gender": {1: 50.0, 2: 50.0}}},
                by="country",
                default_scheme={"gender": {1: 50.0, 2: 50.0, 9: 10.0}},
            )

    def test_validate_targets_reports_without_raising(self, edu_df_polars):
        """validate_targets is advisory — it reports unknown keys but never raises."""
        report = rimpy.validate_targets(
            edu_df_polars, {"education": {1: 40.0, 2: 30.0, 3: 30.0, 9: 0.0, (4, 5): 0.0}}
        )
        assert isinstance(report, dict)
        # Tuple key (4, 5) is understood: no "not found" noise for it, and
        # values 4/5 count as targeted.
        assert not any("(4, 5)" in w and "not found" in w for w in report["warnings"])
        assert not any("Value 4 " in w or "Value 5 " in w for w in report["warnings"])


# ---------------------------------------------------------------------------
# Feature B — tuple keys (combined categories)
# ---------------------------------------------------------------------------


class TestTupleTargets:
    def test_bit_identical_to_premerged(self, edu_df):
        """The core validation property: tuple-key raking == raking on a
        manually pre-merged column, weight for weight."""
        result_t, diag_t = rimpy.rake_with_diagnostics(edu_df, TUPLE_TARGETS)

        df_pre = _premerged(_as_polars(edu_df), [4, 5], 4)
        result_p, diag_p = rimpy.rake_with_diagnostics(
            df_pre,
            {"gender": {1: 50.0, 2: 50.0}, "edu_merged": {1: 33.0, 2: 24.0, 3: 33.0, 4: 10.0}},
        )
        assert (
            _as_polars(result_t)["weight"].to_list()
            == _as_polars(result_p)["weight"].to_list()
        )
        assert diag_t.iterations == diag_p.iterations
        assert diag_t.efficiency == diag_p.efficiency

        # And the merged cell lands exactly on its combined target
        rp = _as_polars(result_t)
        share = (
            rp.filter(pl.col("education").is_in([4, 5]))["weight"].sum()
            / rp["weight"].sum()
        )
        assert abs(share - 0.10) < 1e-9

    def test_output_schema_and_row_order_unchanged(self, edu_df):
        result = rimpy.rake(edu_df, TUPLE_TARGETS)
        in_cols = (
            edu_df.columns if isinstance(edu_df, pl.DataFrame) else list(edu_df.columns)
        )
        out_cols = (
            result.columns if isinstance(result, pl.DataFrame) else list(result.columns)
        )
        assert out_cols == in_cols + ["weight"]
        rp, ip = _as_polars(result), _as_polars(edu_df)
        assert rp["education"].to_list() == ip["education"].to_list()

    def test_one_member_missing_warns_and_rakes(self, edu_df_polars):
        """Combining a category that came back empty is the standard use for
        tuples — warn once and carry the cell on the remaining member(s)."""
        df_no5 = edu_df_polars.filter(pl.col("education") != 5)
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            result = rimpy.rake(df_no5, TUPLE_TARGETS)
        msgs = [str(w.message) for w in recs if issubclass(w.category, UserWarning)]
        assert any("Tuple member 5 of key (4, 5)" in m for m in msgs)
        assert "weight" in result.columns

        # Parity: identical to targeting only the surviving member
        result_scalar = rimpy.rake(
            df_no5,
            {"gender": {1: 50.0, 2: 50.0}, "education": {1: 33.0, 2: 24.0, 3: 33.0, 4: 10.0}},
        )
        assert result["weight"].to_list() == result_scalar["weight"].to_list()

    def test_all_members_missing_raises(self, edu_df_polars):
        targets = {
            "gender": {1: 50.0, 2: 50.0},
            "education": {1: 40.0, 2: 30.0, 3: 20.0, (8, 9): 10.0},
        }
        with pytest.raises(ValueError, match=r"none of its members exist"):
            rimpy.rake(edu_df_polars, targets)

    def test_all_members_missing_with_zero_target_is_dropped(self, edu_df_polars):
        """Same no-op rule as a scalar: a merged cell targeting 0% whose members
        have no rows is dropped, not raised."""
        targets = {
            "gender": {1: 50.0, 2: 50.0},
            "education": {1: 40.0, 2: 30.0, 3: 30.0, (8, 9): 0.0},
        }
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake(edu_df_polars, targets)
        assert any("Dropped zero target" in str(w.message) for w in recs)

    @pytest.mark.parametrize(
        "bad_props, match",
        [
            ({1: 40.0, (2, 3): 30.0, (3, 4): 30.0}, r"Overlapping tuple keys"),
            ({1: 40.0, 4: 30.0, (4, 5): 30.0}, r"Conflicting target keys"),
            ({1: 40.0, 2: 30.0, (4, 4.0): 30.0}, r"duplicate member"),
            ({1: 40.0, 2: 30.0, (4, "x"): 30.0}, r"mixes category labels"),
            ({1: 40.0, 2: 30.0, (4, None): 30.0}, r"unusable"),
            ({1: 40.0, 2: 30.0, (4, 5.5): 30.0}, r"non-integer"),
            ({1: 40.0, 2: 30.0, (): 30.0}, r"Empty tuple key"),
        ],
    )
    def test_structural_tuple_errors(self, edu_df_polars, bad_props, match):
        with pytest.raises(ValueError, match=match):
            rimpy.rake(edu_df_polars, {"education": bad_props})

    def test_one_tuple_equals_scalar(self, edu_df_polars):
        """(4,) is scalar 4; combining it with scalar 4 is an overlap error."""
        t_tuple = {
            "gender": {1: 50.0, 2: 50.0},
            "education": {1: 33.0, 2: 24.0, 3: 33.0, (4,): 6.0, 5: 4.0},
        }
        t_scalar = {
            "gender": {1: 50.0, 2: 50.0},
            "education": {1: 33.0, 2: 24.0, 3: 33.0, 4: 6.0, 5: 4.0},
        }
        r1 = rimpy.rake(edu_df_polars, t_tuple)
        r2 = rimpy.rake(edu_df_polars, t_scalar)
        assert r1["weight"].to_list() == r2["weight"].to_list()

        with pytest.raises(ValueError, match=r"Conflicting target keys"):
            rimpy.rake(
                edu_df_polars,
                {"education": {1: 33.0, 2: 24.0, 3: 33.0, 4: 5.0, (4,): 5.0}},
            )

    def test_tuple_zero_target_each_policy_mode(self, edu_df_polars):
        """{(4,5): 0} on populated members flows through zero_target_policy;
        messages show the user's column and tuple, never internal temp names."""
        targets = {
            "gender": {1: 50.0, 2: 50.0},
            "education": {1: 40.0, 2: 30.0, 3: 30.0, (4, 5): 0.0},
        }
        # error (default)
        with pytest.raises(ValueError) as excinfo:
            rimpy.rake(edu_df_polars, targets)
        msg = str(excinfo.value)
        assert "education category (4, 5)" in msg
        assert "_rimpy_merged" not in msg

        # hard_zero: all member rows weight 0, length and order preserved
        result = rimpy.rake(edu_df_polars, targets, zero_target_policy="hard_zero")
        assert len(result) == len(edu_df_polars)
        merged_rows = result.filter(pl.col("education").is_in([4, 5]))
        assert (merged_rows["weight"] == 0.0).all()
        assert (result.filter(~pl.col("education").is_in([4, 5]))["weight"] > 0).all()

        # near_zero: merged cell's weighted share ~ 0
        result = rimpy.rake(edu_df_polars, targets, zero_target_policy="near_zero")
        share = (
            result.filter(pl.col("education").is_in([4, 5]))["weight"].sum()
            / result["weight"].sum()
        )
        assert share < 1e-6

    def test_tuple_empty_in_one_group_warning_shows_tuple(self, edu_df_polars):
        """Both members absent from one group only → per-group warning that
        renders the original column name and tuple."""
        df = edu_df_polars.filter(
            ~((pl.col("country") == "UK") & (pl.col("education").is_in([4, 5])))
        )
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake_by(df, TUPLE_TARGETS, by="country")
        msgs = [str(w.message) for w in recs if issubclass(w.category, UserWarning)]
        matching = [m for m in msgs if "Group ('UK')" in m]
        assert len(matching) == 1
        assert "(4, 5)" in matching[0]
        assert "'education'" in matching[0]
        assert "_rimpy_merged" not in matching[0]

    def test_rake_by_parity_with_premerged(self, edu_df):
        result_t, _ = rimpy.rake_by_with_diagnostics(edu_df, TUPLE_TARGETS, by="country")
        df_pre = _premerged(_as_polars(edu_df), [4, 5], 4)
        result_p, _ = rimpy.rake_by_with_diagnostics(
            df_pre,
            {"gender": {1: 50.0, 2: 50.0}, "edu_merged": {1: 33.0, 2: 24.0, 3: 33.0, 4: 10.0}},
            by="country",
        )
        assert (
            _as_polars(result_t)["weight"].to_list()
            == _as_polars(result_p)["weight"].to_list()
        )

    def test_rake_by_scheme_mixed_merge_patterns(self, edu_df_polars):
        """Scheme A merges (4,5), scheme B merges (3,4,5) on the same column —
        distinct temp columns per pattern, each verified against a manual merge."""
        schemes = {
            "US": {"gender": {1: 50.0, 2: 50.0}, "education": {1: 33.0, 2: 24.0, 3: 33.0, (4, 5): 10.0}},
            "UK": {"gender": {1: 50.0, 2: 50.0}, "education": {1: 40.0, 2: 30.0, (3, 4, 5): 30.0}},
        }
        result, _ = rimpy.rake_by_scheme_with_diagnostics(edu_df_polars, schemes, by="country")
        assert result.columns == edu_df_polars.columns + ["weight"]

        df_pre = _premerged(edu_df_polars, [4, 5], 4).with_columns(
            pl.when(pl.col("education").is_in([3, 4, 5]))
            .then(3)
            .otherwise(pl.col("education"))
            .alias("edu_merged_345")
        )
        schemes_pre = {
            "US": {"gender": {1: 50.0, 2: 50.0}, "edu_merged": {1: 33.0, 2: 24.0, 3: 33.0, 4: 10.0}},
            "UK": {"gender": {1: 50.0, 2: 50.0}, "edu_merged_345": {1: 40.0, 2: 30.0, 3: 30.0}},
        }
        result_p, _ = rimpy.rake_by_scheme_with_diagnostics(df_pre, schemes_pre, by="country")
        assert result["weight"].to_list() == result_p["weight"].to_list()

    def test_rake_by_scheme_shared_pattern_reuses_temp_column(self, edu_df_polars):
        """Two schemes with the identical merge pattern share one temp column
        (internal); output is correct and schema clean."""
        shared_edu = {1: 33.0, 2: 24.0, 3: 33.0, (4, 5): 10.0}
        schemes = {
            "US": {"gender": {1: 50.0, 2: 50.0}, "education": dict(shared_edu)},
            "UK": {"gender": {1: 48.0, 2: 52.0}, "education": dict(shared_edu)},
        }
        result, diags = rimpy.rake_by_scheme_with_diagnostics(edu_df_polars, schemes, by="country")
        assert result.columns == edu_df_polars.columns + ["weight"]
        assert set(diags.group_results.keys()) == {"US", "UK"}

    def test_tuple_with_float_column(self, edu_df_polars):
        """Float64 category column + int tuple members: parity vs premerge."""
        df = edu_df_polars.with_columns(pl.col("education").cast(pl.Float64))
        result_t = rimpy.rake(df, TUPLE_TARGETS)
        df_pre = _premerged(df, [4, 5], 4)
        result_p = rimpy.rake(
            df_pre,
            {"gender": {1: 50.0, 2: 50.0}, "edu_merged": {1: 33.0, 2: 24.0, 3: 33.0, 4: 10.0}},
        )
        assert result_t["weight"].to_list() == result_p["weight"].to_list()

    @pytest.mark.parametrize("drop_nulls", [True, False])
    def test_tuple_with_nulls(self, edu_df_polars, drop_nulls):
        """The temp column inherits nulls from its source: same rows dropped
        (or kept at weight 1.0) as a manual premerge."""
        df = edu_df_polars.with_columns(
            pl.when(pl.col("gender") == 2)
            .then(
                pl.when(pl.col("education") == 1).then(None).otherwise(pl.col("education"))
            )
            .otherwise(pl.col("education"))
            .alias("education")
        )
        result_t = rimpy.rake(df, TUPLE_TARGETS, drop_nulls=drop_nulls)
        df_pre = _premerged(df, [4, 5], 4)
        result_p = rimpy.rake(
            df_pre,
            {"gender": {1: 50.0, 2: 50.0}, "edu_merged": {1: 33.0, 2: 24.0, 3: 33.0, 4: 10.0}},
            drop_nulls=drop_nulls,
        )
        assert result_t["weight"].to_list() == result_p["weight"].to_list()

    def test_weightipy_style_list_targets(self, edu_df_polars):
        """Tuple keys survive the list-of-dicts (weightipy style) normalization."""
        list_targets = [
            {"gender": {1: 50.0, 2: 50.0}},
            {"education": {1: 33.0, 2: 24.0, 3: 33.0, (4, 5): 10.0}},
        ]
        r_list = rimpy.rake(edu_df_polars, list_targets)
        r_dict = rimpy.rake(edu_df_polars, TUPLE_TARGETS)
        assert r_list["weight"].to_list() == r_dict["weight"].to_list()

    def test_new_warning_stacklevel_points_at_user_code(self, edu_df_polars):
        """The absent-tuple-member warning points at user code."""
        df_no5 = edu_df_polars.filter(pl.col("education") != 5)
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake(df_no5, TUPLE_TARGETS)
        member_warns = [
            w for w in recs
            if issubclass(w.category, UserWarning) and "Tuple member" in str(w.message)
        ]
        assert len(member_warns) == 1
        assert member_warns[0].filename == __file__
