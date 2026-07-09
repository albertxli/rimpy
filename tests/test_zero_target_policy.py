"""
Tests for zero_target_policy on non-empty categories (item #3).

When a user supplies ``target = 0`` for a category that has respondents, rimpy
diverges silently from professional weighting tools. ``zero_target_policy``
forces an explicit decision:

- ``"error"`` (default) raises ValueError with actionable suggestions.
- ``"hard_zero"`` drops those respondents from raking; weight=0 in output.
- ``"near_zero"`` substitutes 0 -> eps in the targets before raking.

Run with: pytest tests/test_zero_target_policy.py -v
"""

import random
import warnings

import pandas as pd
import polars as pl
import pytest

import rimpy


# ---------------------------------------------------------------------------
# Fixtures — pinned 320-row education case mirroring the user's report
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_320_partyaffiliation_polars():
    """320-row survey with pinned cell counts. Tripwire asserts guard drift.

    Cross-tab (Federal=164, State=156, total=320):

        Party code  Federal  State   Marginal
        ----------  -------  -----   --------
            1         77      73      150
            2         38      37       75
            3         31      29       60
            4          1       1        2   <- the sparse trigger cell
            5         17      16       33
        ----------  -------  -----   --------
        Totals       164     156      320
    """
    cells: list[tuple[int, int, int]] = [
        (1, 77, 73),
        (2, 38, 37),
        (3, 31, 29),
        (4, 1, 1),
        (5, 17, 16),
    ]
    rows: list[dict] = []
    for party_code, n_fed, n_state in cells:
        for _ in range(n_fed):
            rows.append({"education": party_code, "gender": 1})
        for _ in range(n_state):
            rows.append({"education": party_code, "gender": 2})

    # Tripwires
    assert len(rows) == 320
    assert sum(1 for r in rows if r["gender"] == 1) == 164
    assert sum(1 for r in rows if r["gender"] == 2) == 156
    assert sum(1 for r in rows if r["education"] == 4) == 2

    rng = random.Random(2024)
    rng.shuffle(rows)
    return pl.DataFrame(rows)


@pytest.fixture
def fixture_320_partyaffiliation_pandas(fixture_320_partyaffiliation_polars):
    return pd.DataFrame(
        {
            col: fixture_320_partyaffiliation_polars[col].to_list()
            for col in fixture_320_partyaffiliation_polars.columns
        }
    )


@pytest.fixture(params=["polars", "pandas"])
def fixture_320_partyaffiliation(
    request, fixture_320_partyaffiliation_polars, fixture_320_partyaffiliation_pandas
):
    return (
        fixture_320_partyaffiliation_polars
        if request.param == "polars"
        else fixture_320_partyaffiliation_pandas
    )


# Standard targets used across most tests: education code 4 is zeroed.
TARGETS_WITH_ZERO_ON_4 = {
    "gender": {1: 50.0, 2: 50.0},
    "education": {1: 33.0, 2: 24.0, 3: 33.0, 4: 0.0, 5: 10.0},
}


def _as_polars(df) -> pl.DataFrame:
    """Normalize result to polars for assertion convenience."""
    return df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)


# ---------------------------------------------------------------------------
# TestZeroTargetPolicyDefault — default "error" mode
# ---------------------------------------------------------------------------


class TestZeroTargetPolicyDefault:
    def test_raises_on_zero_target_with_populated_cell(
        self, fixture_320_partyaffiliation
    ):
        with pytest.raises(
            ValueError,
            match=r"Target = 0 specified for education category 4 with 2 non-empty respondents",
        ):
            rimpy.rake(fixture_320_partyaffiliation, TARGETS_WITH_ZERO_ON_4)

    def test_error_message_lists_all_three_modes(self, fixture_320_partyaffiliation_polars):
        with pytest.raises(ValueError) as excinfo:
            rimpy.rake(fixture_320_partyaffiliation_polars, TARGETS_WITH_ZERO_ON_4)
        msg = str(excinfo.value)
        assert "hard_zero" in msg
        assert "near_zero" in msg
        assert "drop them from the input DataFrame" in msg

    def test_error_message_lists_multiple_zero_targets(
        self, fixture_320_partyaffiliation_polars
    ):
        # Both education=4 and education=5 zeroed
        targets = {
            "gender": {1: 50.0, 2: 50.0},
            "education": {1: 40.0, 2: 30.0, 3: 30.0, 4: 0.0, 5: 0.0},
        }
        with pytest.raises(ValueError) as excinfo:
            rimpy.rake(fixture_320_partyaffiliation_polars, targets)
        msg = str(excinfo.value)
        # Both should be referenced (one as headline, the other in "Additional" list)
        assert "category 4" in msg
        assert "5" in msg
        assert "Additional zero-target cells detected" in msg

    def test_no_error_when_no_zero_targets(self, fixture_320_partyaffiliation_polars):
        targets = {
            "gender": {1: 50.0, 2: 50.0},
            "education": {1: 30.0, 2: 24.0, 3: 30.0, 4: 6.0, 5: 10.0},
        }
        # Should not raise
        result = rimpy.rake(fixture_320_partyaffiliation_polars, targets)
        assert "weight" in result.columns

    def test_zero_target_on_unknown_code_raises_unknown_key(
        self, fixture_320_partyaffiliation_polars
    ):
        # education code 99 doesn't exist in the data. Since item #4, unknown
        # keys raise regardless of target value — before zero_target_policy runs.
        targets = {
            "gender": {1: 50.0, 2: 50.0},
            "education": {1: 33.0, 2: 24.0, 3: 33.0, 4: 1.0, 5: 9.0, 99: 0.0},
        }
        with pytest.raises(ValueError, match=r"Unknown target key"):
            rimpy.rake(fixture_320_partyaffiliation_polars, targets)


# ---------------------------------------------------------------------------
# TestZeroTargetPolicyHardZero — opt-in "hard_zero"
# ---------------------------------------------------------------------------


class TestZeroTargetPolicyHardZero:
    def test_dropped_respondents_get_weight_zero(self, fixture_320_partyaffiliation):
        result = _as_polars(
            rimpy.rake(
                fixture_320_partyaffiliation,
                TARGETS_WITH_ZERO_ON_4,
                zero_target_policy="hard_zero",
            )
        )
        assert len(result) == 320
        zeroed = result.filter(pl.col("education") == 4)
        assert len(zeroed) == 2
        assert all(w == 0.0 for w in zeroed["weight"].to_list())
        # Other respondents must have positive weights
        others = result.filter(pl.col("education") != 4)
        assert all(w > 0.0 for w in others["weight"].to_list())

    def test_row_order_preserved(self, fixture_320_partyaffiliation_polars):
        # Tag input rows with a row index, run hard_zero, confirm order
        tagged = fixture_320_partyaffiliation_polars.with_row_index("_orig_idx")
        result = rimpy.rake(
            tagged,
            TARGETS_WITH_ZERO_ON_4,
            zero_target_policy="hard_zero",
        )
        result_pl = _as_polars(result)
        # _orig_idx should still be in ascending order
        idx_list = result_pl["_orig_idx"].to_list()
        assert idx_list == sorted(idx_list)

    def test_weighted_marginal_for_zeroed_category_is_zero(
        self, fixture_320_partyaffiliation_polars
    ):
        result = rimpy.rake(
            fixture_320_partyaffiliation_polars,
            TARGETS_WITH_ZERO_ON_4,
            zero_target_policy="hard_zero",
        )
        weighted_sum_4 = result.filter(pl.col("education") == 4)[
            "weight"
        ].sum()
        assert weighted_sum_4 == 0.0

    def test_hard_zero_silent_no_warning(self, fixture_320_partyaffiliation_polars):
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake(
                fixture_320_partyaffiliation_polars,
                TARGETS_WITH_ZERO_ON_4,
                zero_target_policy="hard_zero",
            )
        # No empty-target warnings (item #2) since all our targets reference present codes.
        # No zero-target warnings (item #3 hard_zero is silent — opt-in is consent).
        empty_warns = [
            w
            for w in recs
            if issubclass(w.category, UserWarning) and "Target code" in str(w.message)
        ]
        assert len(empty_warns) == 0


# ---------------------------------------------------------------------------
# TestZeroTargetPolicyNearZero — opt-in "near_zero"
# ---------------------------------------------------------------------------


class TestZeroTargetPolicyNearZero:
    def test_near_zero_substitution_produces_small_weights(
        self, fixture_320_partyaffiliation
    ):
        result = _as_polars(
            rimpy.rake(
                fixture_320_partyaffiliation,
                TARGETS_WITH_ZERO_ON_4,
                zero_target_policy="near_zero",
            )
        )
        assert len(result) == 320
        zeroed = result.filter(pl.col("education") == 4)
        for w in zeroed["weight"].to_list():
            # Default eps=1e-8, weights should be very small but positive
            assert 0.0 < w < 1.0

    def test_near_zero_respondents_remain_in_output(
        self, fixture_320_partyaffiliation_polars
    ):
        result = rimpy.rake(
            fixture_320_partyaffiliation_polars,
            TARGETS_WITH_ZERO_ON_4,
            zero_target_policy="near_zero",
        )
        # Same row count, code-4 rows still present
        assert len(result) == 320
        assert len(result.filter(pl.col("education") == 4)) == 2

    def test_custom_near_zero_eps_linear_scaling(
        self, fixture_320_partyaffiliation_polars
    ):
        # With normalized weights (mean=1, sum=n=320), the weighted sum for the
        # zero-target cell tracks eps. Other targets sum to 100%; party-4 gets eps%.
        # weighted_sum_4 ≈ eps * 320 / (1 - eps) ≈ eps * 320 for small eps.
        result_e4 = rimpy.rake(
            fixture_320_partyaffiliation_polars,
            TARGETS_WITH_ZERO_ON_4,
            zero_target_policy="near_zero",
            near_zero_eps=1e-4,
        )
        sum_e4 = result_e4.filter(pl.col("education") == 4)["weight"].sum()
        # Predicted ≈ 1e-4 * 320 = 0.032
        assert 0.025 < sum_e4 < 0.040, (
            f"Expected weighted sum near 0.032 for eps=1e-4, got {sum_e4}"
        )

        result_e2 = rimpy.rake(
            fixture_320_partyaffiliation_polars,
            TARGETS_WITH_ZERO_ON_4,
            zero_target_policy="near_zero",
            near_zero_eps=1e-2,
        )
        sum_e2 = result_e2.filter(pl.col("education") == 4)["weight"].sum()
        # Predicted ≈ 1e-2 * 320 / (1 - 1e-2) ≈ 3.232. Linear scaling check.
        assert 2.5 < sum_e2 < 4.0, (
            f"Expected weighted sum near 3.2 for eps=1e-2, got {sum_e2}"
        )

    def test_near_zero_silent_no_warning(self, fixture_320_partyaffiliation_polars):
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake(
                fixture_320_partyaffiliation_polars,
                TARGETS_WITH_ZERO_ON_4,
                zero_target_policy="near_zero",
            )
        empty_warns = [
            w
            for w in recs
            if issubclass(w.category, UserWarning) and "Target code" in str(w.message)
        ]
        assert len(empty_warns) == 0


# ---------------------------------------------------------------------------
# TestZeroTargetPolicyGroupedRakeBy — rake_by with policy
# ---------------------------------------------------------------------------


@pytest.fixture
def rake_by_df_polars():
    """Survey with country grouping. education=4 has rows in both US and UK."""
    rng = random.Random(2024)
    rows = []
    for _ in range(160):
        rows.append(
            {
                "country": "US",
                "education": rng.choices([1, 2, 3, 4, 5], weights=[5, 3, 4, 1, 2])[0],
            }
        )
    for _ in range(160):
        rows.append(
            {
                "country": "UK",
                "education": rng.choices([1, 2, 3, 4, 5], weights=[4, 4, 4, 1, 2])[0],
            }
        )
    return pl.DataFrame(rows)


class TestZeroTargetPolicyGroupedRakeBy:
    def test_rake_by_error_default_raises(self, rake_by_df_polars):
        targets = {"education": {1: 25.0, 2: 25.0, 3: 25.0, 4: 0.0, 5: 25.0}}
        with pytest.raises(
            ValueError,
            match=r"Target = 0 specified for education category 4",
        ):
            rimpy.rake_by(rake_by_df_polars, targets, by="country")

    def test_rake_by_hard_zero_filters_globally(self, rake_by_df_polars):
        targets = {"education": {1: 25.0, 2: 25.0, 3: 25.0, 4: 0.0, 5: 25.0}}
        result = rimpy.rake_by(
            rake_by_df_polars,
            targets,
            by="country",
            zero_target_policy="hard_zero",
        )
        assert len(result) == 320
        # All party=4 respondents get weight 0 across both countries
        assert result.filter(pl.col("education") == 4)["weight"].sum() == 0.0

    def test_rake_by_near_zero_global_substitution(self, rake_by_df_polars):
        targets = {"education": {1: 25.0, 2: 25.0, 3: 25.0, 4: 0.0, 5: 25.0}}
        result = rimpy.rake_by(
            rake_by_df_polars,
            targets,
            by="country",
            zero_target_policy="near_zero",
        )
        # Weighted sum for party=4 is very small (near eps * n)
        sum_4 = result.filter(pl.col("education") == 4)["weight"].sum()
        assert 0 < sum_4 < 1.0


# ---------------------------------------------------------------------------
# TestZeroTargetPolicyByScheme — rake_by_scheme with policy
# ---------------------------------------------------------------------------


@pytest.fixture
def rake_by_scheme_df_polars():
    """Survey with US/UK/CA groups, all having age categories 1-4."""
    rows = []
    for c in ["US", "UK", "CA"]:
        for _ in range(20):
            rows.append({"country": c, "age": 1})
            rows.append({"country": c, "age": 2})
            rows.append({"country": c, "age": 3})
            rows.append({"country": c, "age": 4})
    return pl.DataFrame(rows)


class TestZeroTargetPolicyByScheme:
    def test_rake_by_scheme_error_identifies_explicit_scheme(
        self, rake_by_scheme_df_polars
    ):
        # US zeros age=4 in its explicit scheme
        schemes = {
            "US": {"age": {1: 25.0, 2: 25.0, 3: 25.0, 4: 0.0}},
            "UK": {"age": {1: 25.0, 2: 25.0, 3: 25.0, 4: 25.0}},
            "CA": {"age": {1: 25.0, 2: 25.0, 3: 25.0, 4: 25.0}},
        }
        with pytest.raises(ValueError) as excinfo:
            rimpy.rake_by_scheme(rake_by_scheme_df_polars, schemes, by="country")
        msg = str(excinfo.value)
        assert "Scheme group ('US')" in msg
        assert "age category 4" in msg

    def test_rake_by_scheme_error_identifies_default_scheme(
        self, rake_by_scheme_df_polars
    ):
        # US has explicit scheme, UK and CA fall back to default which zeros age=4
        schemes = {
            "US": {"age": {1: 25.0, 2: 25.0, 3: 25.0, 4: 25.0}},
        }
        default = {"age": {1: 25.0, 2: 25.0, 3: 25.0, 4: 0.0}}
        with pytest.raises(ValueError) as excinfo:
            rimpy.rake_by_scheme(
                rake_by_scheme_df_polars, schemes, by="country", default_scheme=default
            )
        msg = str(excinfo.value)
        assert "Default-scheme group" in msg
        assert "age category 4" in msg

    def test_rake_by_scheme_hard_zero_per_scheme(self, rake_by_scheme_df_polars):
        schemes = {
            "US": {"age": {1: 25.0, 2: 25.0, 3: 25.0, 4: 0.0}},
            "UK": {"age": {1: 25.0, 2: 25.0, 3: 25.0, 4: 25.0}},
            "CA": {"age": {1: 25.0, 2: 25.0, 3: 25.0, 4: 25.0}},
        }
        result = rimpy.rake_by_scheme(
            rake_by_scheme_df_polars,
            schemes,
            by="country",
            zero_target_policy="hard_zero",
        )
        # US age=4 has weight 0; UK and CA age=4 have positive weights
        us_4 = result.filter((pl.col("country") == "US") & (pl.col("age") == 4))[
            "weight"
        ].sum()
        uk_4 = result.filter((pl.col("country") == "UK") & (pl.col("age") == 4))[
            "weight"
        ].sum()
        ca_4 = result.filter((pl.col("country") == "CA") & (pl.col("age") == 4))[
            "weight"
        ].sum()
        assert us_4 == 0.0
        assert uk_4 > 0.0
        assert ca_4 > 0.0
        assert len(result) == len(rake_by_scheme_df_polars)

    def test_rake_by_scheme_near_zero_per_scheme(self, rake_by_scheme_df_polars):
        schemes = {
            "US": {"age": {1: 25.0, 2: 25.0, 3: 25.0, 4: 0.0}},
            "UK": {"age": {1: 25.0, 2: 25.0, 3: 25.0, 4: 25.0}},
            "CA": {"age": {1: 25.0, 2: 25.0, 3: 25.0, 4: 25.0}},
        }
        result = rimpy.rake_by_scheme(
            rake_by_scheme_df_polars,
            schemes,
            by="country",
            zero_target_policy="near_zero",
        )
        us_4 = result.filter((pl.col("country") == "US") & (pl.col("age") == 4))[
            "weight"
        ].sum()
        uk_4 = result.filter((pl.col("country") == "UK") & (pl.col("age") == 4))[
            "weight"
        ].sum()
        assert 0 < us_4 < 1.0  # near-zero substitution
        assert uk_4 > 0.5  # untouched, weighted to its 25% target


# ---------------------------------------------------------------------------
# TestZeroTargetPolicyValidation — input validation
# ---------------------------------------------------------------------------


class TestZeroTargetPolicyValidation:
    def test_invalid_policy_raises(self, fixture_320_partyaffiliation_polars):
        with pytest.raises(ValueError, match="Unknown zero_target_policy"):
            rimpy.rake(
                fixture_320_partyaffiliation_polars,
                TARGETS_WITH_ZERO_ON_4,
                zero_target_policy="bogus",
            )

    @pytest.mark.parametrize("eps", [-1e-8, 0.0])
    def test_invalid_eps_raises(self, fixture_320_partyaffiliation_polars, eps):
        with pytest.raises(ValueError, match="near_zero_eps must be > 0"):
            rimpy.rake(
                fixture_320_partyaffiliation_polars,
                TARGETS_WITH_ZERO_ON_4,
                zero_target_policy="near_zero",
                near_zero_eps=eps,
            )


# ---------------------------------------------------------------------------
# TestZeroTargetPolicyDoesNotBreakItem2 — regression for item #2
# ---------------------------------------------------------------------------


class TestZeroTargetPolicyDoesNotBreakItem2:
    def test_unknown_key_raises_even_with_nonzero_target(
        self, fixture_320_partyaffiliation_polars
    ):
        # Code 99 absent from the data with a non-zero target. Pre-item-#4 this
        # was the item #2 warning; a globally-unknown key is now a hard error.
        targets = {
            "gender": {1: 50.0, 2: 50.0},
            "education": {
                1: 30.0,
                2: 24.0,
                3: 30.0,
                4: 1.0,
                5: 5.0,
                99: 10.0,
            },
        }
        with pytest.raises(ValueError, match=r"Unknown target key"):
            rimpy.rake(fixture_320_partyaffiliation_polars, targets)

    def test_item_3_hard_zero_still_works_without_unknown_keys(
        self, fixture_320_partyaffiliation_polars
    ):
        # Zero target on a populated cell (item #3) with all keys valid:
        # hard_zero handles it, no warning, no exception.
        targets = {
            "gender": {1: 50.0, 2: 50.0},
            "education": {
                1: 33.0,
                2: 24.0,
                3: 33.0,
                4: 0.0,
                5: 10.0,
            },
        }
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            result = rimpy.rake(
                fixture_320_partyaffiliation_polars,
                targets,
                zero_target_policy="hard_zero",
            )
        assert (
            result.filter(pl.col("education") == 4)["weight"].sum() == 0.0
        )

    def test_unknown_key_raises_before_hard_zero_policy(
        self, fixture_320_partyaffiliation_polars
    ):
        # Ordering proof: an unknown key raises even under hard_zero — key
        # validation runs before zero_target_policy ever sees the targets.
        targets = {
            "gender": {1: 50.0, 2: 50.0},
            "education": {1: 33.0, 2: 24.0, 3: 33.0, 4: 0.0, 5: 10.0, 99: 0.0},
        }
        with pytest.raises(ValueError, match=r"Unknown target key"):
            rimpy.rake(
                fixture_320_partyaffiliation_polars,
                targets,
                zero_target_policy="hard_zero",
            )
