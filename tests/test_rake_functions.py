"""
Tests for the high-level rimpy API: rake(), rake_by(), rake_by_scheme().

Uses realistic mock survey data with demographics (gender, age_group, region, income)
and a country grouping variable for grouped tests.

Run with: pytest tests/test_rake_functions.py -v
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

import rimpy


# ---------------------------------------------------------------------------
# Shared fixtures — realistic survey data
# ---------------------------------------------------------------------------


@pytest.fixture
def survey_df_polars():
    """Realistic survey DataFrame (n=500) with demographic skew, polars."""
    rng = np.random.default_rng(2024)
    n = 500

    # Gender: skewed 60/40 male-heavy (code 1=Male, 2=Female)
    gender = rng.choice([1, 2], size=n, p=[0.60, 0.40]).astype(np.int64)

    # Age group: 1=18-29, 2=30-44, 3=45-59, 4=60+
    age_group = rng.choice([1, 2, 3, 4], size=n, p=[0.30, 0.28, 0.25, 0.17]).astype(
        np.int64
    )

    # Region: 1=Northeast, 2=Midwest, 3=South, 4=West
    region = rng.choice([1, 2, 3, 4], size=n, p=[0.18, 0.21, 0.38, 0.23]).astype(
        np.int64
    )

    # Income: 1=Under 30k, 2=30-60k, 3=60-100k, 4=Over 100k
    income = rng.choice([1, 2, 3, 4], size=n, p=[0.22, 0.30, 0.28, 0.20]).astype(
        np.int64
    )

    # Country: US/UK split (for grouped tests)
    country = rng.choice(["US", "UK"], size=n, p=[0.60, 0.40])

    return pl.DataFrame(
        {
            "gender": gender,
            "age_group": age_group,
            "region": region,
            "income": income,
            "country": country,
        }
    )


@pytest.fixture
def survey_df_pandas(survey_df_polars):
    """Same survey data as pandas DataFrame (constructed directly to avoid pyarrow dep)."""
    return pd.DataFrame(
        {col: survey_df_polars[col].to_list() for col in survey_df_polars.columns}
    )


@pytest.fixture
def demo_targets():
    """Census-like targets for gender and age_group (percentages)."""
    return {
        "gender": {1: 49.0, 2: 51.0},
        "age_group": {1: 22.0, 2: 27.0, 3: 26.0, 4: 25.0},
    }


@pytest.fixture
def full_targets():
    """Targets for all 4 demographic variables."""
    return {
        "gender": {1: 49.0, 2: 51.0},
        "age_group": {1: 22.0, 2: 27.0, 3: 26.0, 4: 25.0},
        "region": {1: 17.0, 2: 21.0, 3: 38.0, 4: 24.0},
        "income": {1: 20.0, 2: 30.0, 3: 28.0, 4: 22.0},
    }


# ---------------------------------------------------------------------------
# TestRake — rake()
# ---------------------------------------------------------------------------


class TestRake:
    """Tests for rimpy.rake() — single-group RIM weighting."""

    def test_basic_convergence(self, survey_df_polars, demo_targets):
        """Weights are produced and column is added."""
        result = rimpy.rake(survey_df_polars, demo_targets)
        assert "weight" in result.columns
        assert len(result) == len(survey_df_polars)
        assert isinstance(result, pl.DataFrame)

    def test_weights_average_one(self, survey_df_polars, demo_targets):
        """Mean weight should be approximately 1.0."""
        result = rimpy.rake(survey_df_polars, demo_targets)
        mean_w = result["weight"].mean()
        assert abs(mean_w - 1.0) < 0.01

    def test_with_max_cap(self, survey_df_polars, demo_targets):
        """Max cap is respected."""
        result = rimpy.rake(survey_df_polars, demo_targets, max_cap=2.0)
        # Cap correction adds 0.0001, so allow small epsilon
        assert result["weight"].max() <= 2.0 + 0.01

    def test_with_min_cap(self, survey_df_polars, demo_targets):
        """Min cap is respected."""
        result = rimpy.rake(survey_df_polars, demo_targets, min_cap=0.5)
        assert result["weight"].min() >= 0.5 - 0.01

    def test_with_both_caps(self, survey_df_polars, demo_targets):
        """Both min and max caps are respected simultaneously."""
        result = rimpy.rake(
            survey_df_polars, demo_targets, min_cap=0.5, max_cap=2.0
        )
        assert result["weight"].min() >= 0.5 - 0.01
        assert result["weight"].max() <= 2.0 + 0.01

    def test_with_controlled_total(self, survey_df_polars, demo_targets):
        """total= scales weight sum to the specified value."""
        result = rimpy.rake(survey_df_polars, demo_targets, total=1000)
        assert abs(result["weight"].sum() - 1000) < 1.0

    def test_drop_nulls(self, survey_df_polars, demo_targets):
        """Rows with nulls in target columns get weight=1.0."""
        # Inject some nulls into gender column (first 10 rows)
        gender_list = survey_df_polars["gender"].to_list()
        for i in range(10):
            gender_list[i] = None
        df = survey_df_polars.with_columns(
            pl.Series("gender", gender_list, dtype=pl.Int64)
        )
        result = rimpy.rake(df, demo_targets, drop_nulls=True)
        # First 10 rows should have weight=1.0
        null_weights = result["weight"][:10].to_numpy()
        assert np.allclose(null_weights, 1.0)

    def test_proportions_and_percentages_match(self, survey_df_polars):
        """Percentage targets and proportion targets produce same weights."""
        targets_pct = {
            "gender": {1: 49.0, 2: 51.0},
            "age_group": {1: 22.0, 2: 27.0, 3: 26.0, 4: 25.0},
        }
        targets_prop = {
            "gender": {1: 0.49, 2: 0.51},
            "age_group": {1: 0.22, 2: 0.27, 3: 0.26, 4: 0.25},
        }
        result_pct = rimpy.rake(survey_df_polars, targets_pct)
        result_prop = rimpy.rake(survey_df_polars, targets_prop)

        np.testing.assert_allclose(
            result_pct["weight"].to_numpy(),
            result_prop["weight"].to_numpy(),
            rtol=1e-6,
        )

    def test_pandas_input_returns_pandas(self, survey_df_pandas, demo_targets):
        """pandas DataFrame in → pandas DataFrame out."""
        result = rimpy.rake(survey_df_pandas, demo_targets)
        assert isinstance(result, pd.DataFrame)
        assert "weight" in result.columns
        mean_w = result["weight"].mean()
        assert abs(mean_w - 1.0) < 0.01

    def test_custom_weight_column_name(self, survey_df_polars, demo_targets):
        """Custom weight column name is respected."""
        result = rimpy.rake(survey_df_polars, demo_targets, weight_column="wt")
        assert "wt" in result.columns
        assert "weight" not in result.columns

    def test_diagnostics(self, survey_df_polars, demo_targets):
        """rake_with_diagnostics returns valid RakeResult."""
        result_df, diag = rimpy.rake_with_diagnostics(survey_df_polars, demo_targets)
        assert diag.converged
        assert 0 < diag.efficiency <= 100
        assert diag.iterations > 0
        assert diag.weight_min > 0
        assert diag.weight_max > diag.weight_min
        assert diag.weight_ratio > 1.0

    def test_four_variables(self, survey_df_polars, full_targets):
        """Raking with 4 demographic variables converges."""
        result_df, diag = rimpy.rake_with_diagnostics(survey_df_polars, full_targets)
        assert diag.converged
        assert abs(result_df["weight"].mean() - 1.0) < 0.01

    def test_weightipy_list_format(self, survey_df_polars):
        """Weightipy-style list of dicts is accepted."""
        targets_list = [
            {"gender": {1: 49.0, 2: 51.0}},
            {"age_group": {1: 22.0, 2: 27.0, 3: 26.0, 4: 25.0}},
        ]
        result = rimpy.rake(survey_df_polars, targets_list)
        assert "weight" in result.columns
        assert abs(result["weight"].mean() - 1.0) < 0.01

    def test_missing_column_raises(self, survey_df_polars):
        """Referencing a non-existent column raises KeyError."""
        bad_targets = {"nonexistent": {1: 50.0, 2: 50.0}}
        with pytest.raises(KeyError):
            rimpy.rake(survey_df_polars, bad_targets)


# ---------------------------------------------------------------------------
# TestRakeBy — rake_by()
# ---------------------------------------------------------------------------


class TestRakeBy:
    """Tests for rimpy.rake_by() — same targets within groups."""

    def test_basic_grouped(self, survey_df_polars, demo_targets):
        """Weights are computed within each country group."""
        result = rimpy.rake_by(survey_df_polars, demo_targets, by="country")
        assert "weight" in result.columns
        assert len(result) == len(survey_df_polars)

    def test_group_weights_average_one(self, survey_df_polars, demo_targets):
        """Within each group, weights average approximately 1.0."""
        result = rimpy.rake_by(survey_df_polars, demo_targets, by="country")

        for country in ["US", "UK"]:
            group_weights = result.filter(pl.col("country") == country)["weight"]
            mean_w = group_weights.mean()
            assert abs(mean_w - 1.0) < 0.05, (
                f"Group {country}: mean weight {mean_w} not ~1.0"
            )

    def test_with_diagnostics(self, survey_df_polars, demo_targets):
        """Per-group RakeResult accessible via GroupedRakeResult."""
        result_df, grouped = rimpy.rake_by_with_diagnostics(
            survey_df_polars, demo_targets, by="country"
        )
        assert "US" in grouped.group_results
        assert "UK" in grouped.group_results

        for country in ["US", "UK"]:
            diag = grouped.group_results[country]
            assert diag.converged
            assert 0 < diag.efficiency <= 100

    def test_summary_df(self, survey_df_polars, demo_targets):
        """GroupedRakeResult.summary_df() returns valid dict."""
        _, grouped = rimpy.rake_by_with_diagnostics(
            survey_df_polars, demo_targets, by="country"
        )
        summary = grouped.summary_df()
        assert "group" in summary
        assert "efficiency" in summary
        assert len(summary["group"]) == 2

    def test_with_caps(self, survey_df_polars, demo_targets):
        """Caps are applied within each group."""
        result = rimpy.rake_by(
            survey_df_polars, demo_targets, by="country", max_cap=2.5
        )
        assert result["weight"].max() <= 2.5 + 0.01

    def test_with_controlled_total(self, survey_df_polars, demo_targets):
        """total= scales overall weight sum across all groups."""
        result = rimpy.rake_by(
            survey_df_polars, demo_targets, by="country", total=5000
        )
        assert abs(result["weight"].sum() - 5000) < 1.0

    def test_pandas_input(self, survey_df_pandas, demo_targets):
        """pandas DataFrame in → pandas DataFrame out for grouped raking."""
        result = rimpy.rake_by(survey_df_pandas, demo_targets, by="country")
        assert isinstance(result, pd.DataFrame)
        assert "weight" in result.columns

    def test_multi_column_groupby(self, survey_df_polars):
        """Group by multiple columns: [country, region]."""
        targets = {"gender": {1: 49.0, 2: 51.0}}
        result = rimpy.rake_by(
            survey_df_polars, targets, by=["country", "region"]
        )
        assert "weight" in result.columns
        assert len(result) == len(survey_df_polars)


# ---------------------------------------------------------------------------
# TestRakeByScheme — rake_by_scheme()
# ---------------------------------------------------------------------------


class TestRakeByScheme:
    """Tests for rimpy.rake_by_scheme() — different targets per group."""

    @pytest.fixture
    def country_schemes(self):
        """Different weighting schemes for US and UK."""
        return {
            "US": {
                "gender": {1: 49.0, 2: 51.0},
                "age_group": {1: 22.0, 2: 27.0, 3: 26.0, 4: 25.0},
                "region": {1: 17.0, 2: 21.0, 3: 38.0, 4: 24.0},
            },
            "UK": {
                "gender": {1: 49.0, 2: 51.0},
                "age_group": {1: 18.0, 2: 26.0, 3: 28.0, 4: 28.0},
                # UK doesn't weight by region
            },
        }

    def test_different_schemes_per_group(self, survey_df_polars, country_schemes):
        """US and UK get different target variables."""
        result = rimpy.rake_by_scheme(
            survey_df_polars, country_schemes, by="country"
        )
        assert "weight" in result.columns
        assert len(result) == len(survey_df_polars)

    def test_diagnostics_per_group(self, survey_df_polars, country_schemes):
        """Per-group diagnostics from scheme-based raking."""
        result_df, grouped = rimpy.rake_by_scheme_with_diagnostics(
            survey_df_polars, country_schemes, by="country"
        )

        for country in ["US", "UK"]:
            diag = grouped.group_results[country]
            assert diag.converged
            assert 0 < diag.efficiency <= 100

    def test_default_scheme_fallback(self, survey_df_polars):
        """Groups not in schemes dict use default_scheme."""
        # Only provide scheme for US, default for everything else
        schemes = {
            "US": {
                "gender": {1: 49.0, 2: 51.0},
                "age_group": {1: 22.0, 2: 27.0, 3: 26.0, 4: 25.0},
            },
        }
        default = {
            "gender": {1: 50.0, 2: 50.0},
        }
        result = rimpy.rake_by_scheme(
            survey_df_polars,
            schemes,
            by="country",
            default_scheme=default,
        )
        assert "weight" in result.columns

        # UK rows should still be weighted (not all 1.0)
        uk_weights = result.filter(pl.col("country") == "UK")["weight"]
        # With default scheme, weights should vary (not all exactly 1.0)
        assert uk_weights.std() > 0.001

    def test_no_scheme_gets_weight_one(self, survey_df_polars):
        """Groups without a scheme and no default get weight=1.0."""
        schemes = {
            "US": {
                "gender": {1: 49.0, 2: 51.0},
            },
        }
        result = rimpy.rake_by_scheme(
            survey_df_polars, schemes, by="country"
        )
        # UK has no scheme → weight=1.0
        uk_weights = result.filter(pl.col("country") == "UK")["weight"]
        assert np.allclose(uk_weights.to_numpy(), 1.0)

    def test_group_totals(self, survey_df_polars, country_schemes):
        """Nested weighting: within-group rake + group proportion adjustment."""
        result = rimpy.rake_by_scheme(
            survey_df_polars,
            country_schemes,
            by="country",
            group_totals={"US": 60, "UK": 40},
        )
        # After group_totals, US should contribute ~60% of total weight
        us_sum = result.filter(pl.col("country") == "US")["weight"].sum()
        total_sum = result["weight"].sum()
        us_pct = us_sum / total_sum * 100
        assert abs(us_pct - 60.0) < 2.0

    def test_group_totals_with_total(self, survey_df_polars, country_schemes):
        """group_totals + total= scales to controlled base."""
        result = rimpy.rake_by_scheme(
            survey_df_polars,
            country_schemes,
            by="country",
            group_totals={"US": 60, "UK": 40},
            total=10000,
        )
        assert abs(result["weight"].sum() - 10000) < 1.0

    def test_with_caps(self, survey_df_polars, country_schemes):
        """Caps are respected in scheme-based raking."""
        result = rimpy.rake_by_scheme(
            survey_df_polars, country_schemes, by="country", max_cap=3.0
        )
        # Weights within raked groups should respect caps
        # (group_totals adjustment may push beyond, but base weights respect caps)
        assert result["weight"].max() <= 3.0 + 0.5  # allow some slack for renorm

    def test_pandas_input(self, survey_df_polars, country_schemes):
        """pandas input → pandas output for scheme-based raking."""
        df_pd = pd.DataFrame(
            {col: survey_df_polars[col].to_list() for col in survey_df_polars.columns}
        )
        result = rimpy.rake_by_scheme(df_pd, country_schemes, by="country")
        assert isinstance(result, pd.DataFrame)
        assert "weight" in result.columns

    def test_missing_group_column_raises(self, survey_df_polars, country_schemes):
        """Referencing a non-existent grouping column raises KeyError."""
        with pytest.raises(KeyError):
            rimpy.rake_by_scheme(
                survey_df_polars, country_schemes, by="nonexistent"
            )


# ---------------------------------------------------------------------------
# TestValidation — validate_targets(), validate_schemes()
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for validation utilities."""

    def test_validate_targets_clean(self, survey_df_polars, demo_targets):
        """Valid targets produce no errors."""
        report = rimpy.validate_targets(survey_df_polars, demo_targets)
        assert len(report["errors"]) == 0

    def test_validate_targets_missing_column(self, survey_df_polars):
        """Missing column is reported as error."""
        bad_targets = {"nonexistent": {1: 50.0, 2: 50.0}}
        report = rimpy.validate_targets(survey_df_polars, bad_targets)
        assert len(report["errors"]) > 0

    def test_validate_targets_bad_sum(self, survey_df_polars):
        """Targets not summing to 100 produce a warning."""
        bad_targets = {"gender": {1: 40.0, 2: 40.0}}  # sum=80
        report = rimpy.validate_targets(survey_df_polars, bad_targets)
        assert len(report["warnings"]) > 0

    def test_validate_schemes(self, survey_df_polars):
        """Scheme validation catches issues per group."""
        schemes = {
            "US": {"gender": {1: 50.0, 2: 50.0}},
            "MISSING_COUNTRY": {"gender": {1: 50.0, 2: 50.0}},
        }
        report = rimpy.validate_schemes(survey_df_polars, schemes, by="country")
        # MISSING_COUNTRY should produce a global warning
        assert len(report["_global"]["warnings"]) > 0


# ---------------------------------------------------------------------------
# TestWeightSummary — weight_summary()
# ---------------------------------------------------------------------------


class TestWeightSummary:
    """Tests for rimpy.weight_summary()."""

    def test_overall_summary(self, survey_df_polars, demo_targets):
        """Overall weight summary produces expected columns."""
        weighted = rimpy.rake(survey_df_polars, demo_targets)
        summary = rimpy.weight_summary(weighted)
        assert isinstance(summary, pl.DataFrame)
        assert "n" in summary.columns
        assert "efficiency_pct" in summary.columns
        assert "effective_n" in summary.columns

    def test_grouped_summary(self, survey_df_polars, demo_targets):
        """Grouped weight summary produces one row per group."""
        weighted = rimpy.rake_by(survey_df_polars, demo_targets, by="country")
        summary = rimpy.weight_summary(weighted, by="country")
        assert len(summary) == 2  # US and UK
