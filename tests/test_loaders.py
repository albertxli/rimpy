"""
Tests for rimpy.load_targets — the file loader for weighting targets.

Layout under test:
    split_var | split_value | split_label | target_var | target_value | target_label | target_pct

CSV is used for most cases so no Excel engine is required; the xlsx round-trip
is gated on fastexcel/xlsxwriter being installed.
"""

import random
import warnings

import polars as pl
import pytest

import rimpy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HEADER = "split_var,split_value,split_label,target_var,target_value,target_label,target_pct"

SPLIT_ROWS = """\
country,101,Bulgaria,age,1,18-34,25
country,101,Bulgaria,age,2,35-54,35
country,101,Bulgaria,age,3,55+,40
country,101,Bulgaria,gender,1,Male,48.95
country,101,Bulgaria,gender,2,Female,50.95
country,101,Bulgaria,gender,3,Other,0.1
country,101,Bulgaria,income_bg,1,Low,60
country,101,Bulgaria,income_bg,2,High,40
country,102,Croatia,age,1,18-34,27
country,102,Croatia,age,2,35-54,32
country,102,Croatia,age,3,55+,41
country,102,Croatia,gender,1,Male,48
country,102,Croatia,gender,2,Female,52
country,102,Croatia,gender,3,Other,0"""

EXPECTED_SPLIT = {
    101: {
        "age": {1: 25.0, 2: 35.0, 3: 40.0},
        "gender": {1: 48.95, 2: 50.95, 3: 0.1},
        "income_bg": {1: 60.0, 2: 40.0},
    },
    102: {
        "age": {1: 27.0, 2: 32.0, 3: 41.0},
        "gender": {1: 48.0, 2: 52.0, 3: 0.0},
    },
}

FLAT_ROWS = """\
target_var,target_value,target_label,target_pct
age,1,18-34,25
age,2,35-54,35
age,3,55+,40
gender,1,Male,49
gender,2,Female,51"""

EXPECTED_FLAT = {
    "age": {1: 25.0, 2: 35.0, 3: 40.0},
    "gender": {1: 49.0, 2: 51.0},
}


def write_csv(tmp_path, text, name="targets.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def split_csv(tmp_path):
    return write_csv(tmp_path, f"{HEADER}\n{SPLIT_ROWS}")


@pytest.fixture
def flat_csv(tmp_path):
    return write_csv(tmp_path, FLAT_ROWS)


def catch(func, *args, **kwargs):
    """Run func, returning (result, [UserWarning messages])."""
    with warnings.catch_warnings(record=True) as recs:
        warnings.simplefilter("always")
        result = func(*args, **kwargs)
    msgs = [str(w.message) for w in recs if issubclass(w.category, UserWarning)]
    return result, msgs


# ---------------------------------------------------------------------------
# Nested (split) targets
# ---------------------------------------------------------------------------


class TestNestedTargets:
    def test_exact_dict_from_csv(self, split_csv):
        targets, msgs = catch(rimpy.load_targets, split_csv)
        assert targets == EXPECTED_SPLIT
        assert msgs == []

    def test_split_keys_are_ints(self, split_csv):
        targets = rimpy.load_targets(split_csv)
        assert all(isinstance(k, int) and not isinstance(k, bool) for k in targets)

    def test_category_codes_are_ints(self, split_csv):
        targets = rimpy.load_targets(split_csv)
        assert all(isinstance(c, int) for c in targets[101]["age"])

    def test_percentages_are_floats(self, split_csv):
        targets = rimpy.load_targets(split_csv)
        assert all(isinstance(p, float) for p in targets[101]["age"].values())

    def test_ragged_variables(self, split_csv):
        """A variable defined for one split only stays under that split."""
        targets = rimpy.load_targets(split_csv)
        assert "income_bg" in targets[101]
        assert "income_bg" not in targets[102]

    def test_label_columns_ignored(self, split_csv):
        targets = rimpy.load_targets(split_csv)
        for variables in targets.values():
            assert "split_label" not in variables
            assert "target_label" not in variables
            assert "split_var" not in variables

    def test_order_follows_file(self, split_csv):
        """Dict order drives raking order, so it must track the file."""
        targets = rimpy.load_targets(split_csv)
        assert list(targets) == [101, 102]
        assert list(targets[101]) == ["age", "gender", "income_bg"]
        assert list(targets[101]["age"]) == [1, 2, 3]

    def test_shuffled_rows_group_correctly(self, tmp_path):
        rows = SPLIT_ROWS.split("\n")
        shuffled = [rows[i] for i in (8, 0, 3, 9, 1, 11, 6, 4, 10, 12, 2, 5, 7, 13)]
        path = write_csv(tmp_path, HEADER + "\n" + "\n".join(shuffled))
        assert rimpy.load_targets(path) == EXPECTED_SPLIT

    def test_polars_dataframe_source(self, split_csv):
        df = pl.read_csv(split_csv)
        assert rimpy.load_targets(df) == EXPECTED_SPLIT

    def test_tsv_source(self, tmp_path):
        text = (HEADER + "\n" + SPLIT_ROWS).replace(",", "\t")
        path = write_csv(tmp_path, text, name="targets.tsv")
        assert rimpy.load_targets(path) == EXPECTED_SPLIT

    def test_explicit_separator(self, tmp_path):
        text = (HEADER + "\n" + SPLIT_ROWS).replace(",", ";")
        path = write_csv(tmp_path, text, name="targets.txt")
        assert rimpy.load_targets(path, separator=";") == EXPECTED_SPLIT

    def test_custom_column_names(self, tmp_path):
        text = (HEADER + "\n" + SPLIT_ROWS).replace(
            HEADER, "sv,country_id,label,variable,code,code_label,pct"
        )
        path = write_csv(tmp_path, text)
        targets = rimpy.load_targets(
            path,
            split_col="country_id",
            var_col="variable",
            value_col="code",
            pct_col="pct",
        )
        assert targets == EXPECTED_SPLIT

    def test_string_split_values_preserved(self, tmp_path):
        text = (HEADER + "\n" + SPLIT_ROWS).replace(",101,", ",BG,").replace(
            ",102,", ",HR,"
        )
        path = write_csv(tmp_path, text)
        targets = rimpy.load_targets(path)
        assert set(targets) == {"BG", "HR"}
        assert targets["BG"]["age"] == EXPECTED_SPLIT[101]["age"]

    def test_extra_columns_ignored(self, tmp_path):
        text = f"{HEADER},notes\n" + "\n".join(
            f"{row},some note" for row in SPLIT_ROWS.split("\n")
        )
        path = write_csv(tmp_path, text)
        assert rimpy.load_targets(path) == EXPECTED_SPLIT

    def test_blank_trailing_rows_dropped(self, tmp_path):
        text = f"{HEADER}\n{SPLIT_ROWS}\n,,,,,,\n,,,,,,"
        path = write_csv(tmp_path, text)
        targets, msgs = catch(rimpy.load_targets, path)
        assert targets == EXPECTED_SPLIT
        assert msgs == []


# ---------------------------------------------------------------------------
# Flat targets
# ---------------------------------------------------------------------------


class TestFlatTargets:
    def test_auto_detect_when_no_split_column(self, flat_csv):
        targets, msgs = catch(rimpy.load_targets, flat_csv)
        assert targets == EXPECTED_FLAT
        assert msgs == []

    def test_auto_detect_when_split_column_all_null(self, tmp_path):
        rows = "\n".join(
            f",,,{r.split(',', 3)[3]}" for r in SPLIT_ROWS.split("\n")[:6]
        )
        path = write_csv(tmp_path, f"{HEADER}\n{rows}")
        targets = rimpy.load_targets(path)
        assert targets == {
            "age": {1: 25.0, 2: 35.0, 3: 40.0},
            "gender": {1: 48.95, 2: 50.95, 3: 0.1},
        }

    def test_explicit_override_pools_single_split(self, tmp_path):
        single = "\n".join(
            r for r in SPLIT_ROWS.split("\n") if r.startswith("country,101")
        )
        path = write_csv(tmp_path, f"{HEADER}\n{single}")
        targets = rimpy.load_targets(path, split_col=None)
        assert targets == EXPECTED_SPLIT[101]

    def test_override_on_multi_split_file_raises(self, split_csv):
        with pytest.raises(ValueError, match=r"Duplicate category code") as excinfo:
            rimpy.load_targets(split_csv, split_col=None)
        assert "split_col=None" in str(excinfo.value)

    def test_missing_explicit_split_column_raises(self, flat_csv):
        with pytest.raises(KeyError, match=r"Split column 'country_id' not found"):
            rimpy.load_targets(flat_csv, split_col="country_id")


# ---------------------------------------------------------------------------
# Type handling
# ---------------------------------------------------------------------------


class TestCodeTypes:
    def test_string_codes_warn_and_pass_through(self, tmp_path):
        text = f"{HEADER}\n" + """\
country,101,Bulgaria,gender,Male,Male,49
country,101,Bulgaria,gender,Female,Female,51"""
        path = write_csv(tmp_path, text)
        targets, msgs = catch(rimpy.load_targets, path)
        assert targets == {101: {"gender": {"Male": 49.0, "Female": 51.0}}}
        assert any("Double-check they match" in m for m in msgs)

    def test_non_integral_float_codes_warn_about_truncation(self, tmp_path):
        text = f"{HEADER}\n" + """\
country,101,Bulgaria,grade,1.5,Low,40
country,101,Bulgaria,grade,2.0,High,60"""
        path = write_csv(tmp_path, text)
        targets, msgs = catch(rimpy.load_targets, path)
        assert targets == {101: {"grade": {1.5: 40.0, 2.0: 60.0}}}
        assert any("truncated" in m for m in msgs)

    def test_integral_float_codes_do_not_warn(self, tmp_path):
        text = f"{HEADER}\n" + """\
country,101,Bulgaria,grade,1.0,Low,40
country,101,Bulgaria,grade,2.0,High,60"""
        path = write_csv(tmp_path, text)
        targets, msgs = catch(rimpy.load_targets, path)
        assert targets == {101: {"grade": {1.0: 40.0, 2.0: 60.0}}}
        assert msgs == []

    def test_integer_codes_do_not_warn(self, split_csv):
        _, msgs = catch(rimpy.load_targets, split_csv)
        assert msgs == []

    def test_warning_points_at_caller(self, tmp_path):
        text = f"{HEADER}\n" + """\
country,101,Bulgaria,gender,Male,Male,49
country,101,Bulgaria,gender,Female,Female,51"""
        path = write_csv(tmp_path, text)
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            rimpy.load_targets(path)
        dtype_warns = [w for w in recs if "Double-check they match" in str(w.message)]
        assert len(dtype_warns) == 1
        assert dtype_warns[0].filename == __file__


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestErrors:
    def test_duplicate_code_raises(self, tmp_path):
        text = f"{HEADER}\n{SPLIT_ROWS}\ncountry,101,Bulgaria,age,1,18-34,26"
        path = write_csv(tmp_path, text)
        with pytest.raises(ValueError, match=r"Duplicate category code") as excinfo:
            rimpy.load_targets(path)
        msg = str(excinfo.value)
        assert "split_value=101" in msg
        assert "target_var='age'" in msg

    def test_missing_required_column_raises(self, tmp_path):
        text = (HEADER + "\n" + SPLIT_ROWS).replace("target_pct", "pct")
        path = write_csv(tmp_path, text)
        with pytest.raises(KeyError, match=r"Missing required column"):
            rimpy.load_targets(path)

    def test_null_in_required_column_raises(self, tmp_path):
        text = f"{HEADER}\n{SPLIT_ROWS}\ncountry,101,Bulgaria,age,,55\\+,10"
        path = write_csv(tmp_path, text)
        with pytest.raises(ValueError, match=r"Null value\(s\) in required column"):
            rimpy.load_targets(path)

    def test_non_numeric_pct_raises(self, tmp_path):
        text = f"{HEADER}\n" + """\
country,101,Bulgaria,gender,1,Male,49%
country,101,Bulgaria,gender,2,Female,51%"""
        path = write_csv(tmp_path, text)
        with pytest.raises(ValueError, match=r"'target_pct' must be numeric"):
            rimpy.load_targets(path)

    def test_empty_file_raises(self, tmp_path):
        path = write_csv(tmp_path, HEADER + "\n")
        with pytest.raises(ValueError, match=r"No target rows found"):
            rimpy.load_targets(path)

    def test_unsupported_suffix_raises(self, tmp_path):
        path = write_csv(tmp_path, HEADER, name="targets.json")
        with pytest.raises(ValueError, match=r"Unsupported file type"):
            rimpy.load_targets(path)

    def test_bad_source_type_raises(self):
        with pytest.raises(TypeError, match=r"file path or polars DataFrame"):
            rimpy.load_targets({"gender": {1: 50, 2: 50}})


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_sum_not_100_warns(self, tmp_path):
        text = (HEADER + "\n" + SPLIT_ROWS).replace(
            "country,102,Croatia,age,3,55+,41", "country,102,Croatia,age,3,55+,45"
        )
        path = write_csv(tmp_path, text)
        _, msgs = catch(rimpy.load_targets, path)
        assert len(msgs) == 1
        assert "do not sum to 100%" in msgs[0]
        assert "split_value=102" in msgs[0]
        assert "target_var='age'" in msgs[0]
        assert "sums to 104" in msgs[0]

    def test_proportions_do_not_warn(self, tmp_path):
        text = f"{HEADER}\n" + """\
country,101,Bulgaria,gender,1,Male,0.49
country,101,Bulgaria,gender,2,Female,0.51"""
        path = write_csv(tmp_path, text)
        targets, msgs = catch(rimpy.load_targets, path)
        assert targets == {101: {"gender": {1: 0.49, 2: 0.51}}}
        assert msgs == []

    def test_validate_false_silences(self, tmp_path):
        text = (HEADER + "\n" + SPLIT_ROWS).replace(
            "country,102,Croatia,age,3,55+,41", "country,102,Croatia,age,3,55+,45"
        )
        path = write_csv(tmp_path, text)
        _, msgs = catch(rimpy.load_targets, path, validate=False)
        assert msgs == []

    def test_mixed_split_var_warns(self, tmp_path):
        text = (HEADER + "\n" + SPLIT_ROWS).replace("country,102", "region,102")
        path = write_csv(tmp_path, text)
        _, msgs = catch(rimpy.load_targets, path)
        assert any("more than one split variable" in m for m in msgs)


# ---------------------------------------------------------------------------
# End to end against the rake functions
# ---------------------------------------------------------------------------


def make_survey_df():
    """Two countries, deliberately skewed away from targets.

    Variables are drawn independently from a fixed seed. That matters: cycling
    two of them on periods that share a factor makes them perfectly correlated,
    and their targets then contradict each other, so no weight vector satisfies
    both and raking legitimately fails to converge.

    Country 101 is sized like the real workbook (400 respondents, 2 of them
    gender 3) because the scheme targets gender 3 at 0.1% — on a 60-row group
    that would be 0.06 weighted people against 2 real respondents.
    """
    rng = random.Random(1234)
    rows = []
    for i in range(400):
        rows.append(
            {
                "country": 101,
                "age": rng.choice([1, 1, 2, 3]),
                "gender": 3 if i < 2 else rng.choice([1, 2]),
                "income_bg": rng.choice([1, 2]),
            }
        )
    for i in range(200):
        rows.append(
            {
                "country": 102,
                "age": rng.choice([1, 1, 2, 3]),
                "gender": rng.choice([1, 2]),
                "income_bg": rng.choice([1, 2]),
            }
        )
    return pl.DataFrame(rows)


def weighted_shares(df, col):
    """Weighted percentage per category of `col`, sorted by category."""
    total = df.get_column("weight").sum()
    got = (
        df.group_by(col)
        .agg(pl.col("weight").sum())
        .sort(col)
        .get_column("weight")
        .to_list()
    )
    return [w / total * 100 for w in got]


class TestEndToEnd:
    def test_nested_feeds_rake_by_scheme(self, split_csv):
        df = make_survey_df()
        targets = rimpy.load_targets(split_csv)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weighted, result = rimpy.rake_by_scheme_with_diagnostics(
                df, targets, by="country"
            )
        assert "weight" in weighted.columns
        assert all(r.converged for r in result.group_results.values())

        # Weighted age margins reproduce the sheet for country 101.
        bg = weighted.filter(pl.col("country") == 101)
        for got, want in zip(weighted_shares(bg, "age"), [25.0, 35.0, 40.0]):
            assert abs(got - want) < 0.1
        hr = weighted.filter(pl.col("country") == 102)
        for got, want in zip(weighted_shares(hr, "age"), [27.0, 32.0, 41.0]):
            assert abs(got - want) < 0.1

    def test_flat_feeds_rake(self, flat_csv):
        # Flat targets cover gender 1/2 only, so drop the sparse gender 3 rows.
        df = make_survey_df().filter(pl.col("gender") != 3)
        targets = rimpy.load_targets(flat_csv)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weighted, result = rimpy.rake_with_diagnostics(df, targets)
        assert result.converged
        for got, want in zip(weighted_shares(weighted, "age"), [25.0, 35.0, 40.0]):
            assert abs(got - want) < 0.1

    def test_flat_feeds_rake_by(self, flat_csv):
        df = make_survey_df().filter(pl.col("gender") != 3)
        targets = rimpy.load_targets(flat_csv)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weighted, result = rimpy.rake_by_with_diagnostics(df, targets, by="country")
        assert all(r.converged for r in result.group_results.values())
        # Every group hits the same flat targets independently.
        for country in (101, 102):
            group = weighted.filter(pl.col("country") == country)
            for got, want in zip(weighted_shares(group, "age"), [25.0, 35.0, 40.0]):
                assert abs(got - want) < 0.1


# ---------------------------------------------------------------------------
# Excel + package surface
# ---------------------------------------------------------------------------


class TestExcel:
    def test_xlsx_matches_csv(self, tmp_path, split_csv):
        pytest.importorskip("fastexcel")
        pytest.importorskip("xlsxwriter")
        path = tmp_path / "targets.xlsx"
        pl.read_csv(split_csv).write_excel(path, worksheet="Wave1")
        assert rimpy.load_targets(path) == EXPECTED_SPLIT
        assert rimpy.load_targets(path, sheet_name="Wave1") == EXPECTED_SPLIT
        assert rimpy.load_targets(path, sheet_name=1) == EXPECTED_SPLIT


class TestPackageSurface:
    def test_load_targets_exported(self):
        assert "load_targets" in rimpy.__all__
        assert callable(rimpy.load_targets)

    @pytest.mark.parametrize("name", ["load_schemes", "load_schemes_wide"])
    def test_old_loaders_removed(self, name):
        assert not hasattr(rimpy, name)
        assert name not in rimpy.__all__


# ---------------------------------------------------------------------------
# Combined categories (the `combine` column)
# ---------------------------------------------------------------------------

COMBINE_HEADER = f"{HEADER},combine"

# education 1..5 with two independent merges: (1,2) and (3,4), 5 left alone.
TWO_MERGE_ROWS = """\
country,101,Bulgaria,education,1,Low,30,A
country,101,Bulgaria,education,2,Mid,30,A
country,101,Bulgaria,education,3,High,45,B
country,101,Bulgaria,education,4,Top,45,B
country,101,Bulgaria,education,5,Other,25,"""


def combine_csv(tmp_path, rows, name="targets.csv"):
    return write_csv(tmp_path, f"{COMBINE_HEADER}\n{rows}", name=name)


class TestCombinedCategories:
    def test_single_merge(self, tmp_path):
        rows = """\
country,101,Bulgaria,education,1,Low,33,
country,101,Bulgaria,education,2,Mid,24,
country,101,Bulgaria,education,3,High,33,
country,101,Bulgaria,education,4,V.High,10,A
country,101,Bulgaria,education,5,Top,10,A"""
        targets, msgs = catch(rimpy.load_targets, combine_csv(tmp_path, rows))
        assert targets == {101: {"education": {1: 33.0, 2: 24.0, 3: 33.0, (4, 5): 10.0}}}
        assert msgs == []

    def test_two_merges_one_variable(self, tmp_path):
        """Two distinct tags -> two tuple keys, and the sum check sees 100."""
        targets, msgs = catch(rimpy.load_targets, combine_csv(tmp_path, TWO_MERGE_ROWS))
        assert targets == {101: {"education": {(1, 2): 30.0, (3, 4): 45.0, 5: 25.0}}}
        assert msgs == []

    def test_one_tag_for_both_pairs_merges_all_four(self, tmp_path):
        """Reusing a tag is a wrong-but-loud merge, not a silent one."""
        rows = TWO_MERGE_ROWS.replace(",B", ",A").replace(",45,", ",30,")
        targets = rimpy.load_targets(combine_csv(tmp_path, rows), validate=False)
        assert targets == {101: {"education": {(1, 2, 3, 4): 30.0, 5: 25.0}}}

    def test_cell_order_follows_first_row(self, tmp_path):
        """Interleaved tags: order is by each cell's first row, not tag name."""
        rows = """\
country,101,Bulgaria,education,1,Low,30,B
country,101,Bulgaria,education,3,High,45,A
country,101,Bulgaria,education,2,Mid,30,B
country,101,Bulgaria,education,4,Top,45,A
country,101,Bulgaria,education,5,Other,25,"""
        targets = rimpy.load_targets(combine_csv(tmp_path, rows))
        assert list(targets[101]["education"]) == [(1, 2), (3, 4), 5]

    def test_tuple_members_are_sorted(self, tmp_path):
        """Row order must not leak into the key: (2,1) != (1,2) as dict keys."""
        rows = """\
country,101,Bulgaria,education,2,Mid,40,A
country,101,Bulgaria,education,1,Low,40,A
country,101,Bulgaria,education,3,High,60,"""
        targets = rimpy.load_targets(combine_csv(tmp_path, rows))
        assert targets == {101: {"education": {(1, 2): 40.0, 3: 60.0}}}

    def test_pct_on_first_row_only(self, tmp_path):
        rows = """\
country,101,Bulgaria,education,1,Low,30,A
country,101,Bulgaria,education,2,Mid,,A
country,101,Bulgaria,education,3,High,70,"""
        targets, msgs = catch(rimpy.load_targets, combine_csv(tmp_path, rows))
        assert targets == {101: {"education": {(1, 2): 30.0, 3: 70.0}}}
        assert msgs == []

    def test_conflicting_pcts_raise(self, tmp_path):
        rows = """\
country,101,Bulgaria,education,1,Low,30,A
country,101,Bulgaria,education,2,Mid,32,A
country,101,Bulgaria,education,3,High,38,"""
        with pytest.raises(ValueError, match=r"without exactly one") as excinfo:
            rimpy.load_targets(combine_csv(tmp_path, rows))
        msg = str(excinfo.value)
        assert "conflicting target_pct values [30.0, 32.0]" in msg
        assert "target_var='education'" in msg

    def test_all_blank_pct_in_group_raises(self, tmp_path):
        rows = """\
country,101,Bulgaria,education,1,Low,,A
country,101,Bulgaria,education,2,Mid,,A
country,101,Bulgaria,education,3,High,100,"""
        with pytest.raises(ValueError, match=r"without exactly one") as excinfo:
            rimpy.load_targets(combine_csv(tmp_path, rows))
        assert "no target_pct" in str(excinfo.value)

    def test_blank_pct_without_combine_raises(self, tmp_path):
        rows = """\
country,101,Bulgaria,education,1,Low,,
country,101,Bulgaria,education,2,Mid,100,"""
        with pytest.raises(ValueError, match=r"without exactly one"):
            rimpy.load_targets(combine_csv(tmp_path, rows))

    def test_tag_scoped_to_variable(self, tmp_path):
        """The same tag under a different variable is a separate cell."""
        rows = """\
country,101,Bulgaria,education,1,Low,40,A
country,101,Bulgaria,education,2,Mid,40,A
country,101,Bulgaria,education,3,High,20,
country,101,Bulgaria,gender,1,Male,100,A
country,101,Bulgaria,gender,2,Female,100,A"""
        targets = rimpy.load_targets(combine_csv(tmp_path, rows), validate=False)
        assert targets[101]["education"] == {(1, 2): 40.0, 3: 20.0}
        assert targets[101]["gender"] == {(1, 2): 100.0}

    def test_tag_scoped_to_split(self, tmp_path):
        rows = """\
country,101,Bulgaria,education,1,Low,40,A
country,101,Bulgaria,education,2,Mid,40,A
country,101,Bulgaria,education,3,High,20,
country,102,Croatia,education,1,Low,35,A
country,102,Croatia,education,2,Mid,35,A
country,102,Croatia,education,3,High,65,"""
        targets = rimpy.load_targets(combine_csv(tmp_path, rows), validate=False)
        assert targets[101]["education"] == {(1, 2): 40.0, 3: 20.0}
        assert targets[102]["education"] == {(1, 2): 35.0, 3: 65.0}

    def test_single_row_group_collapses_to_scalar(self, tmp_path):
        rows = """\
country,101,Bulgaria,education,1,Low,40,A
country,101,Bulgaria,education,2,Mid,60,"""
        targets = rimpy.load_targets(combine_csv(tmp_path, rows))
        assert targets == {101: {"education": {1: 40.0, 2: 60.0}}}

    def test_sum_check_counts_cell_once(self, tmp_path):
        """33+24+33+10 = 100, not 110 - the shared 10 must not double-count."""
        rows = """\
country,101,Bulgaria,education,1,Low,33,
country,101,Bulgaria,education,2,Mid,24,
country,101,Bulgaria,education,3,High,33,
country,101,Bulgaria,education,4,V.High,10,A
country,101,Bulgaria,education,5,Top,10,A"""
        _, msgs = catch(rimpy.load_targets, combine_csv(tmp_path, rows))
        assert msgs == []

    def test_sum_check_still_catches_bad_total(self, tmp_path):
        rows = TWO_MERGE_ROWS.replace(",Other,25,", ",Other,20,")
        _, msgs = catch(rimpy.load_targets, combine_csv(tmp_path, rows))
        assert len(msgs) == 1
        assert "sums to 95" in msgs[0]

    def test_duplicate_inside_group_still_raises(self, tmp_path):
        rows = """\
country,101,Bulgaria,education,1,Low,40,A
country,101,Bulgaria,education,1,Low,40,A
country,101,Bulgaria,education,2,Mid,60,"""
        with pytest.raises(ValueError, match=r"Duplicate category code"):
            rimpy.load_targets(combine_csv(tmp_path, rows))

    def test_whitespace_tag_treated_as_blank(self, tmp_path):
        rows = (
            "country,101,Bulgaria,education,1,Low,40,   \n"
            "country,101,Bulgaria,education,2,Mid,60,   "
        )
        targets = rimpy.load_targets(combine_csv(tmp_path, rows))
        assert targets == {101: {"education": {1: 40.0, 2: 60.0}}}

    def test_combine_column_absent_unchanged(self, split_csv):
        assert rimpy.load_targets(split_csv) == EXPECTED_SPLIT

    def test_combine_column_all_blank_unchanged(self, tmp_path):
        rows = "\n".join(f"{r}," for r in SPLIT_ROWS.split("\n"))
        assert rimpy.load_targets(combine_csv(tmp_path, rows)) == EXPECTED_SPLIT

    def test_combine_col_none_ignores_column(self, tmp_path):
        targets = rimpy.load_targets(
            combine_csv(tmp_path, TWO_MERGE_ROWS), combine_col=None, validate=False
        )
        assert targets == {
            101: {"education": {1: 30.0, 2: 30.0, 3: 45.0, 4: 45.0, 5: 25.0}}
        }

    def test_missing_explicit_combine_column_raises(self, split_csv):
        with pytest.raises(KeyError, match=r"Combine column 'merge' not found"):
            rimpy.load_targets(split_csv, combine_col="merge")

    def test_flat_mode_with_combine(self, tmp_path):
        rows = "\n".join(r.split(",", 3)[3] for r in TWO_MERGE_ROWS.split("\n"))
        path = write_csv(
            tmp_path, "target_var,target_value,target_label,target_pct,combine\n" + rows
        )
        targets, msgs = catch(rimpy.load_targets, path)
        assert targets == {"education": {(1, 2): 30.0, (3, 4): 45.0, 5: 25.0}}
        assert msgs == []

    def test_string_codes_combine_into_string_tuple(self, tmp_path):
        rows = """\
country,101,Bulgaria,education,Low,Low,30,A
country,101,Bulgaria,education,Mid,Mid,30,A
country,101,Bulgaria,education,High,High,70,"""
        targets, msgs = catch(rimpy.load_targets, combine_csv(tmp_path, rows))
        assert targets == {101: {"education": {("Low", "Mid"): 30.0, "High": 70.0}}}
        assert any("Double-check they match" in m for m in msgs)

    def test_end_to_end_both_cells_hit_targets(self, tmp_path):
        """Both merged cells must land on their shared target after raking."""
        path = combine_csv(tmp_path, TWO_MERGE_ROWS)
        targets = rimpy.load_targets(path)
        df = pl.DataFrame(
            {
                "country": [101] * 200,
                "education": [(i % 5) + 1 for i in range(200)],
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weighted, result = rimpy.rake_by_scheme_with_diagnostics(
                df, targets, by="country"
            )
        assert all(r.converged for r in result.group_results.values())
        total = weighted.get_column("weight").sum()
        share = dict(
            weighted.group_by("education").agg(pl.col("weight").sum()).iter_rows()
        )
        assert abs((share[1] + share[2]) / total * 100 - 30.0) < 0.01
        assert abs((share[3] + share[4]) / total * 100 - 45.0) < 0.01
        assert abs(share[5] / total * 100 - 25.0) < 0.01
