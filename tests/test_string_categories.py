"""
Tests for string category codes in target columns.

Before this, target columns had to be numeric: `arrow_adapter.rs` only decoded
integer/Float64 arrays and `lib.rs` coerced every target key to i64, so
`rake(df, {"gender": {"Male": 49, ...}})` died with
`TypeError: must be real number, not str`.

Strings (and Categorical/Enum/category) are now coded to integers at the Arrow
layer, which keeps `engine.rs` numeric. The parity tests below are the ones
that matter: raking a string column must give *bit-identical* weights to
raking a hand-recoded integer copy, which is what makes this a representation
change rather than an algorithm change.
"""

import random
import warnings

import pandas as pd
import polars as pl
import pytest

import rimpy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EDU = ["Low", "Mid", "High", "Top"]
GENDER = ["Male", "Female"]
REGION = ["North", "South"]

EDU_TARGETS = {"Low": 20.0, "Mid": 30.0, "High": 35.0, "Top": 15.0}
GENDER_TARGETS = {"Male": 48.0, "Female": 52.0}
TARGETS = {"education": EDU_TARGETS, "gender": GENDER_TARGETS}

# Deliberately not first-appearance order, and deliberately including negative
# and non-contiguous codes: parity must hold for *any* bijection, which is what
# proves the internal code assignment cannot affect results.
EDU_MAP = {"Top": 90, "High": 3, "Mid": -7, "Low": 42}
GENDER_MAP = {"Female": 100, "Male": 5}


def _raw(n=600, seed=11):
    rng = random.Random(seed)
    return {
        "education": [rng.choice(EDU) for _ in range(n)],
        "gender": [rng.choice(GENDER) for _ in range(n)],
        "region": [rng.choice(REGION) for _ in range(n)],
    }


@pytest.fixture
def raw():
    return _raw()


@pytest.fixture
def str_df_polars(raw):
    return pl.DataFrame(raw)


@pytest.fixture
def str_df_pandas(raw):
    return pd.DataFrame(raw)


@pytest.fixture(params=["polars", "pandas"])
def str_df(request, str_df_polars, str_df_pandas):
    return str_df_polars if request.param == "polars" else str_df_pandas


@pytest.fixture
def int_df_polars(raw):
    """The same data, hand-recoded to integers."""
    return pl.DataFrame(
        {
            "education": [EDU_MAP[v] for v in raw["education"]],
            "gender": [GENDER_MAP[v] for v in raw["gender"]],
            "region": raw["region"],
        }
    )


INT_TARGETS = {
    "education": {EDU_MAP[k]: v for k, v in EDU_TARGETS.items()},
    "gender": {GENDER_MAP[k]: v for k, v in GENDER_TARGETS.items()},
}


def weights(result, column="weight"):
    if isinstance(result, pl.DataFrame):
        return result.get_column(column).to_list()
    return list(result[column])


def shares(result, col):
    """Weighted percentage per category."""
    if isinstance(result, pl.DataFrame):
        total = result.get_column("weight").sum()
        pairs = result.group_by(col).agg(pl.col("weight").sum()).iter_rows()
        return {k: v / total * 100 for k, v in pairs}
    total = result["weight"].sum()
    grouped = result.groupby(col, observed=True)["weight"].sum()
    return {k: v / total * 100 for k, v in grouped.items()}


# ---------------------------------------------------------------------------
# It works at all
# ---------------------------------------------------------------------------


class TestStringTargetsRake:
    def test_rake_hits_targets(self, str_df):
        result = rimpy.rake(str_df, TARGETS)
        for col, targets in TARGETS.items():
            got = shares(result, col)
            for code, want in targets.items():
                assert abs(got[code] - want) < 0.05, (col, code, got[code])

    def test_rake_by_hits_targets_per_group(self, str_df_polars):
        result = rimpy.rake_by(str_df_polars, TARGETS, by="region")
        for region in REGION:
            group = result.filter(pl.col("region") == region)
            got = shares(group, "education")
            for code, want in EDU_TARGETS.items():
                assert abs(got[code] - want) < 0.05, (region, code)

    def test_rake_by_scheme_hits_targets(self, str_df_polars):
        flat = {"education": dict.fromkeys(EDU, 25.0), "gender": GENDER_TARGETS}
        schemes = {"North": TARGETS, "South": flat}
        result = rimpy.rake_by_scheme(str_df_polars, schemes, by="region")
        south = result.filter(pl.col("region") == "South")
        for code, want in flat["education"].items():
            assert abs(shares(south, "education")[code] - want) < 0.05

    def test_diagnostics_converge(self, str_df_polars):
        _, diag = rimpy.rake_with_diagnostics(str_df_polars, TARGETS)
        assert diag.converged
        assert diag.n_valid == str_df_polars.height


# ---------------------------------------------------------------------------
# Parity — the load-bearing tests
# ---------------------------------------------------------------------------


class TestParityWithManualRecode:
    """String raking must equal hand-recoded integer raking, bit for bit."""

    def test_rake_parity(self, str_df_polars, int_df_polars):
        assert weights(rimpy.rake(str_df_polars, TARGETS)) == weights(
            rimpy.rake(int_df_polars, INT_TARGETS)
        )

    def test_rake_by_parity(self, str_df_polars, int_df_polars):
        a = rimpy.rake_by(str_df_polars, TARGETS, by="region")
        b = rimpy.rake_by(int_df_polars, INT_TARGETS, by="region")
        assert weights(a) == weights(b)

    def test_rake_by_scheme_parity(self, str_df_polars, int_df_polars):
        flat = {"education": dict.fromkeys(EDU, 25.0), "gender": GENDER_TARGETS}
        int_flat = {
            "education": {EDU_MAP[k]: v for k, v in flat["education"].items()},
            "gender": INT_TARGETS["gender"],
        }
        a = rimpy.rake_by_scheme(
            str_df_polars, {"North": TARGETS, "South": flat}, by="region"
        )
        b = rimpy.rake_by_scheme(
            int_df_polars, {"North": INT_TARGETS, "South": int_flat}, by="region"
        )
        assert weights(a) == weights(b)

    def test_caps_parity(self, str_df_polars, int_df_polars):
        kw = {"min_cap": 0.5, "max_cap": 2.0}
        assert weights(rimpy.rake(str_df_polars, TARGETS, **kw)) == weights(
            rimpy.rake(int_df_polars, INT_TARGETS, **kw)
        )

    def test_polars_categorical_parity(self, str_df_polars):
        cat = str_df_polars.with_columns(
            pl.col("education").cast(pl.Categorical), pl.col("gender").cast(pl.Categorical)
        )
        assert weights(rimpy.rake(cat, TARGETS)) == weights(
            rimpy.rake(str_df_polars, TARGETS)
        )

    def test_polars_enum_parity(self, str_df_polars):
        enum = str_df_polars.with_columns(
            pl.col("education").cast(pl.Enum(EDU)),
            pl.col("gender").cast(pl.Enum(GENDER)),
        )
        assert weights(rimpy.rake(enum, TARGETS)) == weights(
            rimpy.rake(str_df_polars, TARGETS)
        )

    def test_pandas_category_parity(self, str_df_pandas):
        cat = str_df_pandas.astype({"education": "category", "gender": "category"})
        assert weights(rimpy.rake(cat, TARGETS)) == weights(
            rimpy.rake(str_df_pandas, TARGETS)
        )


# ---------------------------------------------------------------------------
# Combined categories with string members
# ---------------------------------------------------------------------------


class TestStringTupleKeys:
    TUPLE_TARGETS = {
        "education": {("Low", "Mid"): 50.0, "High": 35.0, "Top": 15.0},
        "gender": GENDER_TARGETS,
    }

    def test_string_tuple_hits_combined_target(self, str_df_polars):
        result = rimpy.rake(str_df_polars, self.TUPLE_TARGETS)
        got = shares(result, "education")
        assert abs(got["Low"] + got["Mid"] - 50.0) < 0.05
        assert abs(got["High"] - 35.0) < 0.05

    def test_matches_manual_pre_merge(self, str_df_polars):
        """A tuple key must equal recoding the members into one category."""
        merged = str_df_polars.with_columns(
            pl.when(pl.col("education").is_in(["Low", "Mid"]))
            .then(pl.lit("LowMid"))
            .otherwise(pl.col("education"))
            .alias("education")
        )
        manual = {
            "education": {"LowMid": 50.0, "High": 35.0, "Top": 15.0},
            "gender": GENDER_TARGETS,
        }
        assert weights(rimpy.rake(str_df_polars, self.TUPLE_TARGETS)) == weights(
            rimpy.rake(merged, manual)
        )

    def test_member_order_does_not_matter(self, str_df_polars):
        flipped = dict(self.TUPLE_TARGETS)
        flipped["education"] = {
            ("Mid", "Low"): 50.0,
            "High": 35.0,
            "Top": 15.0,
        }
        assert weights(rimpy.rake(str_df_polars, self.TUPLE_TARGETS)) == weights(
            rimpy.rake(str_df_polars, flipped)
        )

    def test_categorical_column_with_string_tuple(self, str_df_polars):
        cat = str_df_polars.with_columns(pl.col("education").cast(pl.Categorical))
        assert weights(rimpy.rake(cat, self.TUPLE_TARGETS)) == weights(
            rimpy.rake(str_df_polars, self.TUPLE_TARGETS)
        )

    def test_no_temp_columns_leak(self, str_df_polars):
        result = rimpy.rake(str_df_polars, self.TUPLE_TARGETS)
        assert [c for c in result.columns if c.startswith("_rimpy")] == []
        assert result.columns == [*str_df_polars.columns, "weight"]

    def test_mixed_type_tuple_raises(self, str_df_polars):
        targets = {"education": {("Low", 2): 50.0, "High": 35.0, "Top": 15.0}}
        with pytest.raises(ValueError, match=r"mixes category labels"):
            rimpy.rake(str_df_polars, targets)

    def test_scalar_and_tuple_overlap_raises(self, str_df_polars):
        """The gap that string keys used to slip through unclaimed."""
        targets = {
            "education": {"Low": 20.0, ("Low", "Mid"): 30.0, "High": 35.0, "Top": 15.0}
        }
        with pytest.raises(ValueError, match=r"Conflicting target keys"):
            rimpy.rake(str_df_polars, targets)

    def test_overlapping_string_tuples_raise(self, str_df_polars):
        targets = {"education": {("Low", "Mid"): 50.0, ("Mid", "High"): 50.0}}
        with pytest.raises(ValueError, match=r"Overlapping tuple keys"):
            rimpy.rake(str_df_polars, targets)

    def test_labels_with_separators_do_not_collide(self, str_df_polars):
        """Merge identity must be injective for labels containing , and ;."""
        df = str_df_polars.with_columns(
            pl.col("education").replace({"Low": "a,b", "Mid": "c", "High": "a", "Top": "b,c"})
        )
        targets = {"education": {("a,b", "c"): 50.0, ("a", "b,c"): 50.0}}
        result = rimpy.rake(df, targets)
        got = shares(result, "education")
        assert abs(got["a,b"] + got["c"] - 50.0) < 0.05
        assert abs(got["a"] + got["b,c"] - 50.0) < 0.05


# ---------------------------------------------------------------------------
# Errors, nulls, and the Categorical group column
# ---------------------------------------------------------------------------


class TestStringEdgeCases:
    def test_unknown_string_key_raises(self, str_df_polars):
        targets = {"education": {"Low": 20.0, "Mid": 30.0, "High": 35.0, "Nope": 15.0}}
        with pytest.raises(ValueError, match=r"Unknown target key") as excinfo:
            rimpy.rake(str_df_polars, targets)
        msg = str(excinfo.value)
        assert "'Nope'" in msg
        assert "'Low'" in msg  # existing categories are listed, quoted

    def test_case_mismatch_is_not_silently_accepted(self, str_df_polars):
        targets = {"education": {"low": 20.0, "Mid": 30.0, "High": 35.0, "Top": 15.0}}
        with pytest.raises(ValueError, match=r"Unknown target key"):
            rimpy.rake(str_df_polars, targets)

    def test_categorical_group_column(self, str_df_polars):
        """Categorical `by` used to fail: Unsupported group column type."""
        df = str_df_polars.with_columns(pl.col("region").cast(pl.Categorical))
        result = rimpy.rake_by(df, TARGETS, by="region")
        for region in REGION:
            got = shares(result.filter(pl.col("region") == region), "gender")
            for code, want in GENDER_TARGETS.items():
                assert abs(got[code] - want) < 0.05

    def test_categorical_group_column_parity(self, str_df_polars):
        cat = str_df_polars.with_columns(pl.col("region").cast(pl.Categorical))
        assert weights(rimpy.rake_by(cat, TARGETS, by="region")) == weights(
            rimpy.rake_by(str_df_polars, TARGETS, by="region")
        )

    def test_nulls_dropped_by_default(self, raw):
        raw = dict(raw)
        raw["education"] = [None, *raw["education"][1:]]
        df = pl.DataFrame(raw)
        _, diag = rimpy.rake_with_diagnostics(df, TARGETS)
        assert diag.n_valid == df.height - 1

    def test_nulls_kept_when_drop_nulls_false(self, raw):
        """Null rows must miss every target rather than collide with code 0."""
        raw = dict(raw)
        raw["education"] = [None, *raw["education"][1:]]
        df = pl.DataFrame(raw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rimpy.rake(df, TARGETS, drop_nulls=False)
        ws = weights(result)
        assert len(ws) == df.height
        assert all(w > 0 and w == w for w in ws)  # finite, no NaN

    def test_single_category_column(self):
        df = pl.DataFrame({"g": ["only"] * 10})
        result = rimpy.rake(df, {"g": {"only": 100.0}})
        assert all(abs(w - 1.0) < 1e-9 for w in weights(result))

    def test_numeric_key_on_string_column_errors(self, str_df_polars):
        with pytest.raises((ValueError, TypeError)):
            rimpy.rake(str_df_polars, {"education": {1: 50.0, 2: 50.0}})

    def test_string_key_on_numeric_column_errors(self, int_df_polars):
        with pytest.raises((ValueError, TypeError)):
            rimpy.rake(int_df_polars, {"education": {"Low": 50.0, "Mid": 50.0}})

    def test_weight_summary_works(self, str_df_polars):
        result = rimpy.rake(str_df_polars, TARGETS)
        summary = rimpy.weight_summary(result)
        assert summary.get_column("n").item() == str_df_polars.height

    def test_validate_targets_accepts_strings(self, str_df_polars):
        report = rimpy.validate_targets(str_df_polars, TARGETS)
        assert report["errors"] == []
