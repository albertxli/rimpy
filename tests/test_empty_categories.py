"""
Tests for empty-target-category detection in the rimpy API.

Contract (since item #4):
- A target code absent from the ENTIRE raw column raises ``ValueError``
  (unknown key — almost certainly a typo in the targets dict).
- A code present in the column but with zero rows within a group's slice
  (or after null-dropping) emits a ``UserWarning`` — that is normal partial
  data, and the engine silently drops the unsatisfiable target.

These tests cover both severities across all 6 entry points (rake,
rake_with_diagnostics, rake_by, rake_by_with_diagnostics, rake_by_scheme,
rake_by_scheme_with_diagnostics).

Run with: pytest tests/test_empty_categories.py -v
"""

import random
import warnings

import pandas as pd
import polars as pl
import pytest

import rimpy


# ---------------------------------------------------------------------------
# Fixtures — survey data engineered to have specific empty cells
# ---------------------------------------------------------------------------


@pytest.fixture
def survey_df_with_gaps_polars():
    """500-row survey with engineered gaps:

    - gender column has only codes {1, 2} — code 3 absent globally
    - age_group=4 absent in country='UK' specifically (US has all 1-4)
    - (gender=2, region=4) cell is empty by construction (multi-column case)
    """
    rng = random.Random(2024)
    rows = []
    for _ in range(500):
        country = rng.choices(["US", "UK"], weights=[0.6, 0.4])[0]
        gender = rng.choices([1, 2], weights=[0.5, 0.5])[0]  # NO code 3
        if country == "UK":
            age_group = rng.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]  # NO 4
        else:
            age_group = rng.choices([1, 2, 3, 4], weights=[0.3, 0.3, 0.2, 0.2])[0]
        region = rng.choices([1, 2, 3, 4], weights=[0.3, 0.3, 0.3, 0.1])[0]
        if region == 4 and gender == 2:
            region = 3  # forcibly remove (gender=2, region=4) cell
        rows.append(
            {"country": country, "gender": gender, "age_group": age_group, "region": region}
        )
    return pl.DataFrame(rows)


@pytest.fixture
def survey_df_with_gaps_pandas(survey_df_with_gaps_polars):
    return pd.DataFrame(
        {
            col: survey_df_with_gaps_polars[col].to_list()
            for col in survey_df_with_gaps_polars.columns
        }
    )


@pytest.fixture(params=["polars", "pandas"])
def survey_df_with_gaps(
    request, survey_df_with_gaps_polars, survey_df_with_gaps_pandas
):
    """Parametrize tests over polars and pandas backends."""
    return (
        survey_df_with_gaps_polars
        if request.param == "polars"
        else survey_df_with_gaps_pandas
    )


@pytest.fixture
def survey_df_with_null_country_polars(survey_df_with_gaps_polars):
    """Variant where the first 10 rows have country=None and are guaranteed to
    lack age_group=4 (present globally via US rows) — so the None group gets a
    per-group empty-category warning without any globally-unknown key."""
    df = survey_df_with_gaps_polars
    null_mask = pl.Series([True] * 10 + [False] * (len(df) - 10))
    return df.with_columns(
        pl.when(null_mask).then(None).otherwise(pl.col("country")).alias("country"),
        pl.when(null_mask & (pl.col("age_group") == 4))
        .then(1)
        .otherwise(pl.col("age_group"))
        .alias("age_group"),
    )


def _empty_cat_warnings(captured):
    """Filter recorded warnings to just the empty-category ones."""
    return [
        w
        for w in captured
        if issubclass(w.category, UserWarning)
        and ("Target code" in str(w.message) or "target code" in str(w.message))
    ]


# ---------------------------------------------------------------------------
# TestRakeEmptyCategories — single-group rake() / rake_with_diagnostics()
# ---------------------------------------------------------------------------


class TestRakeEmptyCategories:
    def test_raises_on_missing_global_code(self, survey_df_with_gaps):
        """gender=3 has zero rows in the entire column → unknown-key ValueError."""
        targets = {"gender": {1: 40.0, 2: 40.0, 3: 20.0}}
        with pytest.raises(
            ValueError, match=r"Unknown target key\(s\) not present in the data"
        ) as excinfo:
            rimpy.rake(survey_df_with_gaps, targets)
        assert "column 'gender': key 3" in str(excinfo.value)
        assert "[1, 2]" in str(excinfo.value)

    def test_raises_via_with_diagnostics_variant(self, survey_df_with_gaps):
        """The error fires when calling rake_with_diagnostics() directly too."""
        targets = {"gender": {1: 40.0, 2: 40.0, 3: 20.0}}
        with pytest.raises(ValueError, match=r"Unknown target key"):
            rimpy.rake_with_diagnostics(survey_df_with_gaps, targets)

    def test_no_warning_on_complete_data(self, survey_df_with_gaps):
        """All target codes present in data → no empty-category warning."""
        targets = {"gender": {1: 50.0, 2: 50.0}}
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            result = rimpy.rake(survey_df_with_gaps, targets)
        assert len(_empty_cat_warnings(recs)) == 0
        assert "weight" in result.columns

    @pytest.mark.parametrize(
        "fn", [rimpy.rake, rimpy.rake_with_diagnostics]
    )
    def test_warning_stacklevel_points_at_user_code(
        self, survey_df_with_gaps_polars, fn
    ):
        """Warning's filename should match this test file, not _rake.py.
        Verifies stacklevel through both the public and _with_diagnostics paths.

        Uses the null-drop path: region=4 exists in the raw column but all its
        rows have gender nulled out, so after drop_nulls the cell is empty —
        the surviving warning case for the single-group path.
        """
        df = survey_df_with_gaps_polars.with_columns(
            pl.when(pl.col("region") == 4)
            .then(None)
            .otherwise(pl.col("gender"))
            .alias("gender")
        )
        targets = {
            "gender": {1: 50.0, 2: 50.0},
            "region": {1: 30.0, 2: 30.0, 3: 30.0, 4: 10.0},
        }
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            fn(df, targets)
        empty_warns = _empty_cat_warnings(recs)
        assert len(empty_warns) == 1
        # __file__ for this test module
        assert empty_warns[0].filename == __file__, (
            f"expected filename to be {__file__}, "
            f"got {empty_warns[0].filename}"
        )

    def test_no_double_listing_in_error(self, survey_df_with_gaps_polars):
        """One missing code → listed exactly once in the aggregated error."""
        targets = {"gender": {1: 40.0, 2: 40.0, 3: 20.0}}
        with pytest.raises(ValueError) as excinfo:
            rimpy.rake(survey_df_with_gaps_polars, targets)
        assert str(excinfo.value).count("key 3") == 1

    def test_int_target_key_with_float_dataframe_column(
        self, survey_df_with_gaps_polars
    ):
        """Float64 data column + int-keyed targets should not spuriously raise
        for codes that exist (1 ↔ 1.0 numeric equality)."""
        df = survey_df_with_gaps_polars.with_columns(
            pl.col("gender").cast(pl.Float64)
        )
        # Both codes exist in data (as 1.0 and 2.0)
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake(df, {"gender": {1: 50.0, 2: 50.0}})
        assert len(_empty_cat_warnings(recs)) == 0

        # Add a genuinely missing code → unknown-key error
        with pytest.raises(ValueError, match=r"Unknown target key"):
            rimpy.rake(df, {"gender": {1: 40.0, 2: 40.0, 3: 20.0}})

    def test_float_target_key_with_int_dataframe_column(
        self, survey_df_with_gaps_polars
    ):
        """Reverse: Int64 data column + float-keyed targets. Set membership
        relies on hash(1) == hash(1.0); should not asymmetrically raise."""
        df = survey_df_with_gaps_polars  # gender is Int64
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake(df, {"gender": {1.0: 50.0, 2.0: 50.0}})
        assert len(_empty_cat_warnings(recs)) == 0

        # Missing code raises regardless of target value — the != 0 loophole
        # that let {3.0: 0.0} through silently is closed.
        with pytest.raises(ValueError, match=r"Unknown target key"):
            rimpy.rake(df, {"gender": {1.0: 50.0, 2.0: 50.0, 3.0: 0.0}})

        with pytest.raises(ValueError, match=r"Unknown target key"):
            rimpy.rake(df, {"gender": {1.0: 40.0, 2.0: 40.0, 3.0: 20.0}})

    def test_zero_target_for_missing_code_raises(
        self, survey_df_with_gaps_polars
    ):
        """target_value == 0 for a globally missing code raises like any other
        unknown key — an absent code is a dict bug regardless of its value."""
        targets = {"gender": {1: 50.0, 2: 50.0, 3: 0.0}}
        with pytest.raises(ValueError, match=r"Unknown target key"):
            rimpy.rake(survey_df_with_gaps_polars, targets)


# ---------------------------------------------------------------------------
# TestRakeByEmptyCategories — rake_by() / rake_by_with_diagnostics()
# ---------------------------------------------------------------------------


class TestRakeByEmptyCategories:
    def test_warns_on_code_missing_in_one_group_only(self, survey_df_with_gaps):
        """age_group=4 exists in US but not UK; expect warning identifying UK."""
        targets = {"age_group": {1: 25.0, 2: 25.0, 3: 25.0, 4: 25.0}}
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            result = rimpy.rake_by(survey_df_with_gaps, targets, by="country")
        msgs = [str(w.message) for w in _empty_cat_warnings(recs)]
        assert len(msgs) == 1
        assert "Group ('UK')" in msgs[0]
        assert "code 4" in msgs[0]
        assert "weight" in result.columns

    def test_no_warning_when_each_group_has_all_codes(
        self, survey_df_with_gaps
    ):
        """Targets cover only codes present in every group."""
        targets = {"age_group": {1: 33.0, 2: 33.0, 3: 34.0}}
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake_by(survey_df_with_gaps, targets, by="country")
        assert len(_empty_cat_warnings(recs)) == 0

    def test_multi_column_by_per_combo_detection(self, survey_df_with_gaps):
        """by=['country', 'gender']: cell (gender=2, region=4) is empty by
        fixture construction → warnings for (US, 2) and (UK, 2) only."""
        targets = {"region": {1: 25.0, 2: 25.0, 3: 25.0, 4: 25.0}}

        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake_by(
                survey_df_with_gaps, targets, by=["country", "gender"]
            )
        msgs = [str(w.message) for w in _empty_cat_warnings(recs)]
        # Exactly two warnings, one per (US|UK, 2) combination, mentioning code 4
        msgs_with_code4 = [m for m in msgs if "code 4" in m]
        assert len(msgs_with_code4) == 2
        joined = "\n".join(msgs_with_code4)
        assert "country='US', gender=2" in joined
        assert "country='UK', gender=2" in joined
        assert "gender=1" not in joined

        # Reversed by order — labels follow user's input order, not internal sort
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake_by(
                survey_df_with_gaps, targets, by=["gender", "country"]
            )
        msgs = [str(w.message) for w in _empty_cat_warnings(recs) if "code 4" in str(w.message)]
        joined = "\n".join(msgs)
        assert "gender=2, country='US'" in joined
        assert "gender=2, country='UK'" in joined

    def test_null_group_key_warning_format(
        self, survey_df_with_null_country_polars
    ):
        """Rows with country=None form their own group; warnings should
        render cleanly as Group (None): ... and not raise TypeError.

        age_group=4 exists globally (US rows) but the fixture guarantees the
        null-country rows lack it — a per-group warning, not an unknown key.
        """
        df = survey_df_with_null_country_polars
        targets = {"age_group": {1: 25.0, 2: 25.0, 3: 25.0, 4: 25.0}}
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake_by(df, targets, by="country")
        msgs = [str(w.message) for w in _empty_cat_warnings(recs)]
        assert any("Group (None)" in m for m in msgs), (
            f"expected a 'Group (None)' warning, got: {msgs}"
        )
        # Multi-column case with null in one of the by columns
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake_by(df, targets, by=["country", "gender"])
        msgs = [str(w.message) for w in _empty_cat_warnings(recs)]
        assert any("country=None" in m for m in msgs), (
            f"expected a 'country=None' warning, got: {msgs}"
        )

    @pytest.mark.parametrize(
        "fn", [rimpy.rake_by, rimpy.rake_by_with_diagnostics]
    )
    def test_warning_stacklevel_in_grouped_path(
        self, survey_df_with_gaps_polars, fn
    ):
        """Stacklevel correct through both rake_by and rake_by_with_diagnostics."""
        targets = {"age_group": {1: 25.0, 2: 25.0, 3: 25.0, 4: 25.0}}
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            fn(survey_df_with_gaps_polars, targets, by="country")
        empty_warns = _empty_cat_warnings(recs)
        assert len(empty_warns) >= 1
        assert empty_warns[0].filename == __file__


# ---------------------------------------------------------------------------
# TestRakeBySchemeEmptyCategories — rake_by_scheme() / _with_diagnostics()
# ---------------------------------------------------------------------------


class TestRakeBySchemeEmptyCategories:
    def test_warns_when_scheme_references_missing_code(
        self, survey_df_with_gaps
    ):
        """UK scheme requesting age_group=4 (which UK lacks) should warn."""
        # Sanity: scheme excluding age_group=4 for UK should NOT warn
        ok_schemes = {
            "US": {"age_group": {1: 25.0, 2: 25.0, 3: 25.0, 4: 25.0}},
            "UK": {"age_group": {1: 33.0, 2: 33.0, 3: 34.0}},
        }
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake_by_scheme(survey_df_with_gaps, ok_schemes, by="country")
        assert len(_empty_cat_warnings(recs)) == 0

        # Now flip UK to include age_group=4 → expect warning
        bad_schemes = {
            "US": {"age_group": {1: 25.0, 2: 25.0, 3: 25.0, 4: 25.0}},
            "UK": {"age_group": {1: 25.0, 2: 25.0, 3: 25.0, 4: 25.0}},
        }
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            result = rimpy.rake_by_scheme(
                survey_df_with_gaps, bad_schemes, by="country"
            )
        msgs = [str(w.message) for w in _empty_cat_warnings(recs)]
        assert any(
            "Scheme group ('UK')" in m and "code 4" in m for m in msgs
        ), f"expected Scheme group ('UK') warning, got: {msgs}"
        assert "weight" in result.columns

    def test_warns_via_default_scheme_fallback(self, survey_df_with_gaps):
        """UK falls back to default_scheme which requests age_group=4 → warn."""
        schemes = {"US": {"gender": {1: 50.0, 2: 50.0}}}
        default = {"age_group": {1: 25.0, 2: 25.0, 3: 25.0, 4: 25.0}}
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake_by_scheme(
                survey_df_with_gaps,
                schemes,
                by="country",
                default_scheme=default,
            )
        msgs = [str(w.message) for w in _empty_cat_warnings(recs)]
        assert any(
            "Default-scheme group ('UK')" in m and "code 4" in m for m in msgs
        ), f"expected Default-scheme group ('UK') warning, got: {msgs}"

    def test_no_warning_when_all_schemes_complete(self, survey_df_with_gaps):
        """Schemes covering only codes present in each group, no default."""
        schemes = {
            "US": {"gender": {1: 50.0, 2: 50.0}},
            "UK": {"gender": {1: 50.0, 2: 50.0}},
        }
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake_by_scheme(survey_df_with_gaps, schemes, by="country")
        assert len(_empty_cat_warnings(recs)) == 0
