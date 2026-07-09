# rimpy edge cases

RIM/raking algorithms have several edge cases where the "right" behavior is
ambiguous or where professional weighting tools disagree with each other. This
document covers those cases, what rimpy does, why, and how to choose the right
behavior for your workflow.

---

## 1. Zero targets on populated categories (`zero_target_policy`)

### The problem

You write a targets dict like this:

```python
targets = {
    "gender":       {1: 50, 2: 50},
    "education": {1: 33, 2: 24, 3: 33, 4: 0, 5: 10},
}
```

`education` category 4 is set to a target of 0 — the user-facing intent is
"category 4 should be 0% of the weighted sample". But category 4 has 2
respondents in the data. What should happen?

There's no universally agreed answer. Professional tools each pick a different
approach:

| Tool | Behavior when `target = 0` on a populated cell |
|------|------------------------------------------------|
| **Q / Quantum / SPSS RIM** | Hard zero. weight=0, weighted % = 0. Documented. |
| **R `svyweight::rakesvy`** | Hard zero (explicit "assigns weight = 0"). |
| **R `anesrake` v0.80** | **Refuses to run**: `stop("you cannot rake any variable category to 0 or a negative number")`. |
| **R `survey::rake`** | Undefined — docs limit convergence guarantee to tables without zero cells. |
| **`weightipy`** | Substitutes `0 → 0.00000001`. Silent. |
| **rimpy < 0.3** (pre-fix) | Silent skip. Weights settle at whatever value cross-dim raking happens to produce (e.g., ~0.45). No warning. **This was a bug.** |

### rimpy's solution

rimpy makes the decision explicit. By default, it refuses with an actionable
error. The user opts into one of two well-defined behaviors via the
`zero_target_policy` kwarg.

```python
rimpy.rake(df, targets, zero_target_policy="error")       # default — raises ValueError
rimpy.rake(df, targets, zero_target_policy="hard_zero")   # drop respondents, weight=0
rimpy.rake(df, targets, zero_target_policy="near_zero")   # substitute 0 → eps in targets
```

The default error message lists all three options inline:

```
Target = 0 specified for education category 4 with 2 non-empty respondents.

rimpy requires explicit handling of zero targets on non-empty cells. Options:
  - To exclude these respondents from weighting entirely:
      drop them from the input DataFrame before calling rake()
  - To force their weighted % to ~0 while keeping them in the file:
      pass zero_target_policy="near_zero"
  - To force weight=0 (matches Q / SPSS behavior):
      pass zero_target_policy="hard_zero"

See rimpy.rake docstring for the methodological tradeoffs.
```

### Mode reference

#### `"error"` (default)

Raises `ValueError` before raking starts. The error message names the
offending column / code / respondent count and lists the three resolution
paths. Matches the R `anesrake` convention.

Best for: production pipelines where unexpected zero targets indicate a bug
in the targets dict (e.g., the `{4-5: 14}` arithmetic-evaluation typo that
silently becomes `{-1: 14}` then maps to nothing).

#### `"hard_zero"`

Filters respondents in zero-target categories out of the raking iteration
entirely. They appear in the output with `weight = 0` and contribute nothing
to weighted statistics. The iteration runs on the reduced sample
(`n − k`, where k is the count of filtered respondents).

Matches: Q / SPSS / svyweight **at the level of weighted percentages and the
final `weight = 0` for zero-target respondents.** Per-respondent weights for
*other* respondents diverge from Q because the iteration runs on a smaller
sample than Q's does internally (see "Q's apparent internal implementation"
below).

#### `"near_zero"`

Substitutes `near_zero_eps` (default `1e-8`) for any declared 0 target in the
targets dict before raking. Respondents stay in the file with weights very
close to zero. Weighted % for the zero-target category is approximately
`eps × n / sum(weights)` — negligible but not exactly 0. The iteration runs
on the full sample.

The substituted value is **scale-aware**: if the column's other targets sum
to a percentage (`> 1.5`), rimpy expresses eps in percentage units (`eps × 100`)
so that the post-normalization effective eps still matches what the user
asked for. This means `near_zero_eps=1e-8` produces the same effective
behavior whether you pass targets as proportions or percentages.

`near_zero_eps` is configurable. Common values:

- `1e-8` (default) — vanishing weights, weighted % is machine-precision noise.
- `1e-4` — weighted % rounds to `0.00%` in 4-decimal reports but is not zero.
- `1e-2` — visible-but-tiny presence in the weighted distribution.

Matches: weightipy's `0 → 0.00000001` substitution. **Empirically closer to
Q's internal behavior than `hard_zero`** for per-respondent weights — see
the next section.

### Q's apparent internal implementation (empirical finding)

We compared rimpy's `hard_zero` and `near_zero` modes against Q's R-engine
weights on a 320-row demographic fixture (2 respondents in `education`
category 4 with target = 0). Q reports those 2 respondents at `weight = 0` and category 4
at weighted % = 0.000000%, matching its documented "hard zero" behavior.

But comparing per-respondent weights across all 320 rows reveals something
unexpected:

| Mode | Respondents agreeing with Q to < 1e-6 (out of 318 non-zero) |
|------|-------------------------------------------------------------|
| `hard_zero` | **0** |
| `near_zero` (eps=1e-8) | **9** |

Under `near_zero`, the 2 category-4 respondents land at ~8e-9 each (right at
the eps scale you'd expect from substituting `target = 1e-8` into the
iteration). Q reports those same respondents as exactly 0.

This is the **opposite** of what the design phase expected. The plan assumed
`hard_zero` would be the "match Q exactly" mode and `near_zero` would be a
softer alternative for the Povaddo near-zero reporting convention.

#### Working hypothesis

The pattern is consistent with Q doing the following internally:

1. **Substitute a small epsilon** (functionally similar to rimpy's `near_zero_eps`)
   for any declared 0 target before raking.
2. **Run the standard raking iteration on all respondents** — no exclusion,
   no filter, no smaller sample.
3. **Post-process the output**: any respondent in a cell with a declared 0
   target gets their weight overwritten to exactly 0 in the reported output.

This explains everything observed:

- The 2 category-4 respondents report as **exactly 0** in Q because of step 3 —
  but the iteration that produced everyone else's weights ran with those
  respondents included at near-zero weight (step 2), so the cross-dimensional
  adjustments propagate the same way as rimpy's `near_zero`.
- rimpy's `near_zero` lands those 2 respondents at ~8e-9 because rimpy
  doesn't do the post-iteration cleanup step that Q does; the eps-scale weight
  is the raw output of the iteration.
- The 9 sub-1e-6 agreements with Q under `near_zero` are explained: same
  iteration path, same cross-dim adjustments, same convergence behavior for
  non-zeroed respondents.
- The **zero** sub-1e-6 agreements under `hard_zero` are explained: rimpy
  drops the 2 respondents, so the iteration runs on n=318 instead of n=320,
  producing a different convergence path for everyone.

#### Testable predictions

If the hypothesis is correct:

1. Decreasing `near_zero_eps` (e.g., `1e-12`) should **increase** the count of
   sub-1e-6 agreements with Q, because rimpy's iteration would more closely
   match Q's effective internal eps.
2. Increasing `near_zero_eps` (e.g., `1e-4` or `1e-2`) should **decrease** the
   count of sub-1e-6 agreements, because rimpy's eps would diverge further
   from Q's internal value.
3. The 2 category-4 respondents' weights under `near_zero` should scale
   linearly with `near_zero_eps` — at `eps=1e-6` they should be ~8e-7, at
   `eps=1e-4` they should be ~8e-5, etc.

All three are cheap to run on the existing fixture. Confirming any
strengthens the hypothesis; falsifying any complicates it.

#### Possible future mode: `near_zero_with_post_zero`

If matching Q's per-respondent output exactly matters for production
parallel-validation work, a fourth mode is conceptually clean:

```
Substitute near_zero_eps before raking (like near_zero), then overwrite
the weights of zero-target respondents to exactly 0 in the output
(like hard_zero's headline result, but without filtering them from
the iteration).
```

This is the hypothesized internal behavior of Q. It would produce:

- **Exactly 0 weights** for zero-target respondents (matches Q output).
- **Iteration path matching Q's full-sample iteration** (matches Q's
  per-respondent weights for other respondents).

This is **not in scope for the current zero_target_policy API** — the
three-mode API is locked. But it's recorded for the backlog as a potential
follow-up if production parallel-validation use cases demand it. The
implementation cost is small: it's essentially `near_zero` plus a zeroing
step in the output reassembly, similar to what `hard_zero` already does for
its own zeroing.

### Choosing a mode

| Goal | Recommended mode |
|------|------------------|
| You want to know about every zero-target gotcha and decide case-by-case | `"error"` (default) |
| Match Q on the *weighted percentage* (= 0%) and have hard zero weights in the output | `"hard_zero"` |
| Match Q's per-respondent weights as closely as the engine allows, for downstream analysis | `"near_zero"` (empirically closer to Q than `hard_zero`) |
| Povaddo near-zero reporting convention (show ~0% in reports without losing respondents) | `"near_zero"` with `near_zero_eps=1e-4` or similar |
| Drop respondents upstream and avoid the question entirely | Pre-filter the DataFrame, then pass to rimpy with the zero entries removed from targets |

### Separate finding: cross-engine numerical residuals

The empirical comparison against Q also surfaced **8 respondents** with weight
deltas in `[1e-3, 5e-2]` that are present under **both** `hard_zero` and
`near_zero`. These are unrelated to zero-target handling and represent
inherent rimpy-vs-Q raking-engine numerical differences. Likely causes:

- Iteration order (does Q traverse dimensions in the same order rimpy does?)
- Convergence tolerance / stopping criterion
- Raking algorithm variant (classic IPF vs. variants — does Q use a different
  update rule?)

This is tracked separately and is **not a bug introduced by item #3**. The
zero-target policy fix solves what it was designed to solve.

### Reference

- Q "Common Errors in Weights": <https://help.qresearchsoftware.com/hc/en-us/articles/4414958629391-Common-Errors-in-Weights>
- R `anesrake` v0.80 source: `R/rakeonvar.{numeric,factor,default}.R` (the
  `stop("you cannot rake any variable category to 0 or a negative number")`
  guard).
- weightipy source: `weightipy/internal/rim.py` `rakeonvar()` (`if target_prop == 0.00: target_prop = 0.00000001`).
- R `svyweight::rakesvy` docs: <https://rdrr.io/cran/svyweight/man/rakesvy.html>

---

## 2. Empty target categories (`UserWarning`)

### The problem

You supply a target proportion for a category code that has zero respondents
in the frame being raked — the engine can't satisfy this target; there's no
one to weight.

### Behavior (since item #4 tightened the contract)

Two severities depending on *where* the code is missing:

- **Absent from the entire raw column** → `ValueError` before raking (see
  §3 — an unknown key is a targets-dict bug, not a data condition).
- **Present in the column but zero rows in the relevant slice** — i.e. within
  one group of `rake_by`/`rake_by_scheme`, or globally but only after
  `drop_nulls` removed its rows — → `UserWarning`, and the rake still runs
  with the unsatisfiable target silently dropped by the engine:

```
UserWarning: Group ('UK'): target code 4 for column 'age' has zero rows
             in this group; target (25.0) will be silently dropped.
```

Group labels are deterministic:

- `rake_by`: `Group ('UK'): ...`
- `rake_by`, multi-column `by`: `Group (country='US', gender=2): ...`
- `rake_by_scheme`, explicit scheme: `Scheme group ('UK'): ...`
- `rake_by_scheme`, default-scheme fallback: `Default-scheme group ('CA'): ...`

This split matches real workflows: a shared code frame where one group lacks
a category (no age=4 respondents in UK this wave) is normal partial data and
shouldn't block the run; a code that exists nowhere in the column is almost
certainly a typo and should.

### Handling a category that came back empty this wave

Remove its key from that wave's targets dict (or combine it into a
neighboring category with tuple syntax, §4):

```python
targets = {"gender": {1: 50, 2: 50}}          # drop the key, or
targets = {"education": {..., (4, 5): 10}}    # fold the empty code into a merge
```

Note the pre-0.4 idiom of writing `{3: 0}` for a globally-absent code now
raises — see §3.

### Reference

- See `to_be_added.md` item #2 (committed as `ebc3a41`) for the original design.


## 3. Unknown target keys (`ValueError`)

### The problem

Two footguns produce silently wrong weights if unknown keys are accepted:

```python
targets = {"education": {1: 33, 2: 20, 3: 33, 4-5: 14}}
```

Python evaluates the dict key `4-5` as arithmetic **before rimpy ever sees
it** — the dict that arrives is `{1: 33, 2: 20, 3: 33, -1: 14}`. Pre-0.4,
rimpy warned on `{-1: 14}` and was completely silent on `{-1: 0}`; the engine
skipped the unmatched key and produced weights that don't satisfy the user's
intended marginal.

### Behavior

Any target key absent from the **entire raw data column** raises `ValueError`
before raking, regardless of its target value. All offenders are aggregated
into one error, and when an offending key is negative while the column's codes
are all non-negative, the message explains the arithmetic trap:

```
ValueError: Unknown target key(s) not present in the data:
  - column 'education': key -1. Existing categories: [1, 2, 3, 4, 5]

Hint: a negative key like -1 often comes from Python arithmetic inside a
dict literal — {4-5: 14} is evaluated to {-1: 14} before rimpy sees it.
To combine categories into one target cell, use tuple syntax: {(4, 5): 14}.
```

Validation runs **before** `zero_target_policy`, so an unknown key raises
under every policy mode. String keys and non-integer float keys (e.g. `4.5`
against an int column) get the same clear error instead of an opaque FFI
failure. `None` is a valid key only when the column actually contains nulls.

`validate_targets()` / `validate_schemes()` remain advisory: they report
unknown keys in their `errors` list without raising.

### What other tools do

Q surfaces empty-categories-with-targets as a hard error in its Diagnostics
Report; R `anesrake` refuses invalid category specs before iterating. rimpy's
raise matches that camp for globally-absent codes while keeping warnings for
group-slice absence (§2).


## 4. Combined-category tuple keys

### The problem

Combining sparse categories before weighting is routine survey work, but
pre-0.4 rimpy's targets dict had no syntax for it — users had to recode the
data column upstream in every notebook.

### Behavior

A tuple key merges its member categories into **one cell** sharing a single
target:

```python
targets = {
    "gender": {1: 50, 2: 50},
    "education": {1: 33, 2: 24, 3: 33, (4, 5): 10},   # 4 and 5 together = 10%
}
```

Every raking iteration applies one shared multiplier to all respondents with
education 4 *or* 5; the split of the 10% between them follows the data (their
relative sizes plus what the other dimensions do). This is the same operation
R `survey` / Q / weightipy users perform by manually recoding — internally
rimpy builds a temporary recoded column, rakes on it, and drops it from the
output, so results are **bit-identical to a manual pre-merge**.

Rules:

- Members must be integer codes (int-valued floats like `4.0` are accepted);
  every category may appear in at most one key — overlapping tuples, or a
  tuple overlapping a scalar key, raise `ValueError`.
- A 1-tuple `(4,)` is equivalent to the scalar key `4`.
- A tuple is valid if **at least one** member exists in the column (combining
  a category that came back empty is the standard use); individually absent
  members emit a `UserWarning`. All members absent → `ValueError` (§3).
- `{(4, 5): 0}` flows through `zero_target_policy` like any zero target on a
  populated cell (§1); error messages render the tuple, not internal names.
- In `rake_by_scheme`, different schemes may merge differently on the same
  column (scheme A `(4, 5)`, scheme B `(3, 4, 5)`) — each pattern gets its own
  internal recode.

Not yet supported in the Excel/CSV scheme loaders — tuple targets are
dict-API-only for now (see `to_be_added.md` item #5).
