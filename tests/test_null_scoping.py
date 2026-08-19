"""
Tests for null-dropping scope in rake_by_scheme, and for zero targets on
categories that have no rows.

Both fixes come from one real multi-market run (rake_by_scheme by CITY, with
market-specific variables and a code frame carrying every category):

1. `rake_batch_by_scheme` built ONE null mask over the union of every scheme's
   target columns, so a row had to be non-null in every *other* market's
   columns too. With market-specific variables no row qualified: every group
   came back n_valid=0, every weight 1.0, converged=True, and nothing warned.

2. A target of 0.0 on a category absent from the data raised. Survey code
   frames carry every category, so targets built from one legitimately contain
   zeros for categories nobody fell into — a satisfied no-op, not an error.
"""

import warnings

import polars as pl
import pytest

import rimpy

# ---------------------------------------------------------------------------
# Fixtures: two markets, each with its own market-specific variable
# ---------------------------------------------------------------------------

N = 100

MARKET_TARGETS = {
    7: {"D2": {1: 40.0, 2: 60.0}, "XIT": {1: 30.0, 2: 70.0}},
    1: {"D2": {1: 40.0, 2: 60.0}, "XES": {1: 30.0, 2: 70.0}},
}


@pytest.fixture
def market_df():
    """XIT is populated only for city 7, XES only for city 1."""
    return pl.DataFrame(
        {
            "CITY": [7.0] * N + [1.0] * N,
            "D2": [float((i % 2) + 1) for i in range(N)] * 2,
            "XIT": [float((i % 2) + 1) for i in range(N)] + [None] * N,
            "XES": [None] * N + [float((i % 2) + 1) for i in range(N)],
        },
        schema={
            "CITY": pl.Float64,
            "D2": pl.Float64,
            "XIT": pl.Float64,
            "XES": pl.Float64,
        },
    )


def shares(df, col):
    total = df.get_column("weight").sum()
    pairs = df.group_by(col).agg(pl.col("weight").sum()).iter_rows()
    return {k: v / total * 100 for k, v in pairs if k is not None}


def weights(df):
    return df.get_column("weight").to_list()


# ---------------------------------------------------------------------------
# Null-dropping is scoped to each scheme's own columns
# ---------------------------------------------------------------------------


class TestPerSchemeNullScope:
    def test_market_specific_columns_do_not_empty_every_group(self, market_df):
        """The regression: this returned n_valid=0 everywhere and all-1.0 weights."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result, diag = rimpy.rake_by_scheme_with_diagnostics(
                market_df, MARKET_TARGETS, by="CITY"
            )
        assert {k: r.n_valid for k, r in diag.group_results.items()} == {
            "7": N,
            "1": N,
        }
        assert not all(w == 1.0 for w in weights(result))

    def test_each_market_hits_its_own_targets(self, market_df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rimpy.rake_by_scheme(market_df, MARKET_TARGETS, by="CITY")
        for city, col in ((7.0, "XIT"), (1.0, "XES")):
            got = shares(result.filter(pl.col("CITY") == city), col)
            assert abs(got[1.0] - 30.0) < 0.01
            assert abs(got[2.0] - 70.0) < 0.01

    def test_adding_a_scheme_does_not_disturb_other_groups(self, market_df):
        """The invariant the bug broke: one market's weights must not depend on
        which columns *another* market's scheme happens to name."""
        one_market = {7: MARKET_TARGETS[7]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alone = rimpy.rake_by_scheme(market_df, one_market, by="CITY")
            both = rimpy.rake_by_scheme(market_df, MARKET_TARGETS, by="CITY")
        city7 = pl.col("CITY") == 7.0
        assert weights(alone.filter(city7)) == weights(both.filter(city7))

    def test_nulls_inside_a_groups_own_column_are_still_dropped(self, market_df):
        """Scoping the mask must not stop null-dropping where it belongs."""
        holed = market_df.with_columns(
            pl.when((pl.col("CITY") == 7.0) & (pl.int_range(pl.len()) < 10))
            .then(None)
            .otherwise(pl.col("XIT"))
            .alias("XIT")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, diag = rimpy.rake_by_scheme_with_diagnostics(
                holed, MARKET_TARGETS, by="CITY"
            )
        assert diag.group_results["7"].n_valid == N - 10
        assert diag.group_results["1"].n_valid == N

    def test_drop_nulls_false_unchanged(self, market_df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = rimpy.rake_by_scheme(market_df, MARKET_TARGETS, by="CITY")
            b = rimpy.rake_by_scheme(
                market_df, MARKET_TARGETS, by="CITY", drop_nulls=False
            )
        # Every row is non-null in its own market's columns, so both agree.
        assert weights(a) == weights(b)

    def test_shared_columns_still_work(self):
        """Schemes naming the same columns share one mask; nothing changes."""
        df = pl.DataFrame(
            {
                "g": ["A"] * N + ["B"] * N,
                "x": [float((i % 2) + 1) for i in range(N)] * 2,
            }
        )
        t = {"A": {"x": {1: 25.0, 2: 75.0}}, "B": {"x": {1: 60.0, 2: 40.0}}}
        result = rimpy.rake_by_scheme(df, t, by="g")
        assert abs(shares(result.filter(pl.col("g") == "A"), "x")[1.0] - 25.0) < 0.01
        assert abs(shares(result.filter(pl.col("g") == "B"), "x")[1.0] - 60.0) < 0.01


class TestEmptyGroupWarning:
    def test_group_with_every_row_null_warns(self, market_df):
        """A group really can lose all its rows; that must not be silent."""
        # XIT moves to the *other* city: its codes still exist in the column
        # (so key validation passes) but no city-7 row has one.
        blanked = market_df.with_columns(
            pl.when(pl.col("CITY") == 7.0)
            .then(None)
            .otherwise(pl.col("XES"))
            .alias("XIT")
        )
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            _, diag = rimpy.rake_by_scheme_with_diagnostics(
                blanked, MARKET_TARGETS, by="CITY"
            )
        assert diag.group_results["7"].n_valid == 0
        assert any("left unweighted" in str(w.message) for w in recs)

    def test_no_warning_when_every_group_has_rows(self, market_df):
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake_by_scheme(market_df, MARKET_TARGETS, by="CITY")
        assert not any("left unweighted" in str(w.message) for w in recs)


# ---------------------------------------------------------------------------
# Zero targets on categories with no rows
# ---------------------------------------------------------------------------


@pytest.fixture
def gap_df():
    """`code` has categories 2..7 — category 1 exists in the code frame only."""
    return pl.DataFrame(
        {
            "CITY": [7.0] * 180,
            # sex must vary independently of code: cycling both on periods
            # that share a factor makes their targets mutually unsatisfiable.
            "code": [float((i % 6) + 2) for i in range(180)],
            "sex": [float(((i // 6) % 2) + 1) for i in range(180)],
        }
    )


CODEFRAME = {
    7: {
        "code": {1: 0.0, 2: 10.0, 3: 15.0, 4: 17.0, 5: 20.0, 6: 16.0, 7: 22.0},
        "sex": {1: 50.0, 2: 50.0},
    }
}


class TestZeroTargetOnAbsentCategory:
    def test_codeframe_targets_run_unedited(self, gap_df):
        """The user-facing point: targets straight from a code frame work."""
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            result = rimpy.rake_by_scheme(gap_df, CODEFRAME, by="CITY")
        assert any("Dropped zero target" in str(w.message) for w in recs)
        got = shares(result, "code")
        for code, want in ((2.0, 10.0), (3.0, 15.0), (7.0, 22.0)):
            assert abs(got[code] - want) < 0.05

    def test_warning_names_column_and_code(self, gap_df):
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake_by_scheme(gap_df, CODEFRAME, by="CITY")
        msg = next(m for m in (str(w.message) for w in recs) if "Dropped zero" in m)
        assert "'code'" in msg and "1" in msg

    def test_callers_targets_dict_is_not_mutated(self, gap_df):
        """_normalize_targets hands a plain dict straight through, so dropping
        in place would corrupt the caller's variable between runs."""
        targets = {"code": {1: 0.0, 2: 10.0, 3: 15.0, 4: 17.0, 5: 20.0, 6: 16.0, 7: 22.0}}
        before = {c: dict(p) for c, p in targets.items()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rimpy.rake(gap_df, targets)
        assert targets == before

    def test_dropping_every_target_removes_the_column(self):
        df = pl.DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 1, 2]})
        targets = {"a": {1: 50.0, 2: 50.0}, "b": {8: 0.0, 9: 0.0}}
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            result = rimpy.rake(df, targets)
        assert any("no targets left" in str(w.message) for w in recs)
        assert "weight" in result.columns

    def test_still_works_through_rake_by(self, gap_df):
        targets = {"code": {1: 0.0, 2: 10.0, 3: 15.0, 4: 17.0, 5: 20.0, 6: 16.0, 7: 22.0}}
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            result = rimpy.rake_by(gap_df, targets, by="CITY")
        assert any("Dropped zero target" in str(w.message) for w in recs)
        assert abs(shares(result, "code")[2.0] - 10.0) < 0.05

    def test_zero_target_on_populated_category_is_untouched(self, gap_df):
        """Only *absent* categories are dropped; a zero target on a category
        that has respondents still goes through zero_target_policy."""
        targets = {"code": {2: 10.0, 3: 15.0, 4: 17.0, 5: 20.0, 6: 16.0, 7: 22.0, 1: 0.0}}
        bad = targets | {"code": {**targets["code"], 2: 0.0}}
        with pytest.raises(ValueError, match=r"Target = 0 specified"):
            rimpy.rake(gap_df, bad)
