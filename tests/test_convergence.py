"""
Tests for the convergence criterion and non-convergence reporting.

Through v0.4.1 rimpy converged on *weight movement*: `sum |w - w_prev| < 0.01`,
inherited from quantipy/weightipy. Two consequences:

1. The metric summed over rows, so the effective per-respondent tolerance was
   `threshold / n` — 5e-5 on a 200-row subgroup, 1e-7 on a million rows. The
   same default was hundreds of times looser on small groups, which is where
   survey work with per-market schemes actually lives.
2. A companion stall check set `converged = true` when progress merely
   plateaued, so mutually contradictory targets were reported as a successful
   run with no warning.

Convergence is now measured on margin misfit — how far each achieved margin is
from its target — in R survey's regularized form `|achieved - target| / (1 +
target)`. That is what the caller actually asked for, it means the same thing at
any n, and targets that cannot be met simply never converge.
"""

import random
import warnings

import polars as pl
import pytest

import rimpy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def independent_df(n, seed=17):
    """Variables drawn independently, so the targets below are satisfiable."""
    rng = random.Random(seed)
    return pl.DataFrame(
        {
            "age": [rng.choice([1, 2, 3]) for _ in range(n)],
            "sex": [rng.choice([1, 2]) for _ in range(n)],
            "region": [rng.choice([1, 2, 3, 4]) for _ in range(n)],
        }
    )


TARGETS = {
    "age": {1: 25.0, 2: 35.0, 3: 40.0},
    "sex": {1: 48.0, 2: 52.0},
    "region": {1: 20.0, 2: 30.0, 3: 30.0, 4: 20.0},
}


@pytest.fixture
def contradictory_df():
    """`code` and `sex` cycle on periods 6 and 2, so they are perfectly
    correlated: codes 2/4/6 are always sex 1. The code targets imply sex 1 is
    43% while the sex targets demand 50%. No weight vector satisfies both."""
    n = 180
    return pl.DataFrame(
        {
            "code": [float((i % 6) + 2) for i in range(n)],
            "sex": [float((i % 2) + 1) for i in range(n)],
        }
    )


CONTRADICTORY_TARGETS = {
    "code": {2: 10.0, 3: 15.0, 4: 17.0, 5: 20.0, 6: 16.0, 7: 22.0},
    "sex": {1: 50.0, 2: 50.0},
}


def margin_error_pp(df, targets, weight="weight"):
    """Worst achieved-vs-target margin error, in percentage points."""
    worst = 0.0
    for var, tgt in targets.items():
        sub = df.filter(pl.col(var).is_not_null())
        total = sub.get_column(weight).sum()
        got = dict(sub.group_by(var).agg(pl.col(weight).sum()).iter_rows())
        for code, want in tgt.items():
            members = code if isinstance(code, tuple) else (code,)
            share = sum(got.get(m, 0.0) for m in members) / total * 100
            worst = max(worst, abs(share - want))
    return worst


# ---------------------------------------------------------------------------
# The tolerance means the same thing at every dataset size
# ---------------------------------------------------------------------------


class TestScaleInvariance:
    @pytest.mark.parametrize("n", [50, 200, 2_000, 20_000])
    def test_margins_hit_regardless_of_size(self, n):
        """The regression: under `sum |Δw|` a 50-row run stopped ~400x looser
        than a 20,000-row one, purely because the metric summed over rows."""
        df = independent_df(n)
        weighted, diag = rimpy.rake_with_diagnostics(df, TARGETS)
        assert diag.converged
        assert margin_error_pp(weighted, TARGETS) < 1e-5, f"n={n}"

    def test_small_group_is_not_looser_than_large(self):
        small = margin_error_pp(rimpy.rake(independent_df(60), TARGETS), TARGETS)
        large = margin_error_pp(rimpy.rake(independent_df(20_000), TARGETS), TARGETS)
        # Both far below any reporting precision; neither is orders of
        # magnitude worse than the other.
        assert small < 1e-5 and large < 1e-5

    def test_tightening_reduces_the_gap(self):
        df = independent_df(2_000)
        gaps = []
        for eps in (1e-2, 1e-6, 1e-10):
            _, d = rimpy.rake_with_diagnostics(df, TARGETS, convergence_threshold=eps)
            gaps.append(d.max_target_gap)
        assert gaps[0] > gaps[1] > gaps[2]


# ---------------------------------------------------------------------------
# Contradictory targets are reported, not hidden
# ---------------------------------------------------------------------------


class TestContradictoryTargets:
    def test_does_not_claim_convergence(self, contradictory_df):
        """v0.4.1 returned converged=True after 2 iterations here, with `code`
        margins 2.77pp off and no warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, diag = rimpy.rake_with_diagnostics(
                contradictory_df, CONTRADICTORY_TARGETS
            )
        assert diag.converged is False
        assert diag.stalled is True
        assert diag.max_target_gap > 0.01

    def test_warns_with_the_gap(self, contradictory_df):
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake(contradictory_df, CONTRADICTORY_TARGETS)
        msgs = [str(w.message) for w in recs]
        assert any("did not converge" in m for m in msgs)
        assert any("off target by" in m for m in msgs)

    def test_weights_are_still_returned(self, contradictory_df):
        """Like R survey, a failure to converge warns but still yields weights."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rimpy.rake(contradictory_df, CONTRADICTORY_TARGETS)
        w = result.get_column("weight")
        assert w.len() == contradictory_df.height
        assert w.min() > 0

    def test_tightening_does_not_rescue_it(self, contradictory_df):
        """The targets are unsatisfiable; no threshold makes them satisfiable."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, a = rimpy.rake_with_diagnostics(
                contradictory_df, CONTRADICTORY_TARGETS, convergence_threshold=1e-4
            )
            _, b = rimpy.rake_with_diagnostics(
                contradictory_df, CONTRADICTORY_TARGETS, convergence_threshold=1e-12
            )
        assert not a.converged and not b.converged

    def test_grouped_warning_names_the_group(self, contradictory_df):
        df = contradictory_df.with_columns(pl.lit("EU").alias("region"))
        schemes = {"EU": CONTRADICTORY_TARGETS}
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.rake_by_scheme(df, schemes, by="region")
        assert any(
            "did not converge" in str(w.message) and "'EU'" in str(w.message)
            for w in recs
        )


# ---------------------------------------------------------------------------
# Well-specified targets stay clean (no false positives)
# ---------------------------------------------------------------------------


class TestNoFalsePositives:
    def test_satisfiable_targets_converge_silently(self):
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            _, diag = rimpy.rake_with_diagnostics(independent_df(500), TARGETS)
        assert diag.converged is True
        assert diag.stalled is False
        assert not any("did not converge" in str(w.message) for w in recs)

    def test_tiny_but_reachable_target(self):
        """A 0.1% target with a couple of respondents in 400 is reachable and
        must not trip the warning — the shape of a real 'other/prefer not to
        say' gender category."""
        rng = random.Random(5)
        df = pl.DataFrame(
            {
                "gender": [3 if i < 2 else rng.choice([1, 2]) for i in range(400)],
                "age": [rng.choice([1, 2, 3]) for _ in range(400)],
            }
        )
        t = {"gender": {1: 48.95, 2: 50.95, 3: 0.1}, "age": {1: 30.0, 2: 35.0, 3: 35.0}}
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            _, diag = rimpy.rake_with_diagnostics(df, t)
        assert diag.converged, f"gap={diag.max_target_gap}"
        assert not any("did not converge" in str(w.message) for w in recs)

    def test_grouped_run_converges(self):
        df = independent_df(600).with_columns(
            (pl.int_range(pl.len()) % 3).alias("country")
        )
        _, diag = rimpy.rake_by_with_diagnostics(df, TARGETS, by="country")
        assert all(r.converged for r in diag.group_results.values())


# ---------------------------------------------------------------------------
# The new diagnostics are exposed
# ---------------------------------------------------------------------------


class TestDiagnosticsSurface:
    def test_fields_present(self):
        _, diag = rimpy.rake_with_diagnostics(independent_df(200), TARGETS)
        assert isinstance(diag.stalled, bool)
        assert isinstance(diag.max_target_gap, float)

    def test_summary_includes_them(self):
        _, diag = rimpy.rake_with_diagnostics(independent_df(200), TARGETS)
        s = diag.summary()
        assert "stalled" in s and "max_target_gap" in s
        assert s["converged"] == 1.0
        assert s["stalled"] == 0.0

    def test_repr_shows_the_gap(self):
        _, diag = rimpy.rake_with_diagnostics(independent_df(200), TARGETS)
        assert "max_target_gap" in repr(diag)

    def test_converged_agrees_with_the_gap(self):
        """converged and max_target_gap must tell the same story — they are
        computed from the same formula."""
        df = independent_df(1_000)
        for eps in (1e-4, 1e-8):
            _, d = rimpy.rake_with_diagnostics(df, TARGETS, convergence_threshold=eps)
            if d.converged:
                assert d.max_target_gap < eps, f"eps={eps} gap={d.max_target_gap}"

    def test_grouped_summary_df_has_the_columns(self):
        df = independent_df(600).with_columns(
            (pl.int_range(pl.len()) % 3).alias("country")
        )
        _, diag = rimpy.rake_by_with_diagnostics(df, TARGETS, by="country")
        cols = diag.summary_df()
        assert "stalled" in cols and "max_target_gap" in cols
