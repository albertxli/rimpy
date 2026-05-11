# To Be Added

Backlog of features / fixes / diagnostics to revisit later. Each item should describe the problem, the current behavior, and a sketch of the fix.

---

## 1. Detect & report contradictory weighting targets (silent non-convergence)

### Problem
When target sets are mutually inconsistent — classic case being a single respondent who is the only occupant of two cells whose marginal targets imply different weighted sample sizes for that respondent — there is no weight vector that satisfies all marginals simultaneously. This is a mathematical property of RIM/raking, not an implementation bug; it affects every raking algorithm (Q, R `survey::rake`, weightipy, rimpy).

Q surfaces this as a hard error in its Diagnostics Report. **rimpy currently hides it.**

### Current behavior in rimpy
In `src/engine.rs:299-333`, the iteration loop has two exit conditions:

```rust
// True convergence
if new_diff_error < opts.convergence_threshold {
    converged = true;
    break;
}

// Progress stalled — also reported as converged
if iter > 1 && new_diff_error >= pct_still * diff_error {
    converged = true;
    break;
}
```

The second branch is a **stall detector**. With contradictory targets, weights oscillate between satisfying one target set and the other; `diff_error` flatlines; the stall branch fires and sets `converged = true`. The caller receives:

- `converged: true` (misleading)
- A weight vector that satisfies *neither* contradicting target
- No warning, no diagnostic flag

The conflicted respondent's weight ends up at some mid-oscillation compromise value — **not 1.0**, just whatever the multiplicative updates settled at when the stall fired.

### What's missing
1. **No distinction between true convergence and stall** in `RakeResult`.
2. **No post-hoc verification** that weighted marginals actually match input targets within tolerance.
3. **No pre-flight check** for pathological cells (e.g., n=1 cells appearing in multiple target dimensions).
4. **No diagnostic** identifying *which pair* of target columns conflict (Q's leave-one-out approach).

### Proposed fix (sketch)
Minimum viable improvement — distinguish stall from convergence:

```rust
pub struct RakeResult {
    // ... existing fields
    pub converged: bool,    // true only when diff_error < threshold
    pub stalled: bool,      // true when stall detector fired without true convergence
    pub final_diff_error: f64,
}
```

Stretch goals:
- Add `verify_targets()` helper that recomputes weighted marginals from the result and returns per-target gaps. Surface as `RakeResult.target_gaps: HashMap<String, HashMap<i64, f64>>` or as a method.
- Emit a Python warning when `stalled && max_gap > some_threshold`.
- Diagnostic mode that runs leave-one-out across target columns to identify the conflicting pair (expensive — opt-in only).

### Validation requirements (per CLAUDE.md)
Any algorithm change here must be cross-validated against R `survey::rake()` and Python `weightipy` to ensure detection logic does not produce false positives on edge cases that those libraries handle cleanly (e.g., near-singular but solvable target sets).

### Reference
Q's writeup of the same issue: <https://wiki.q-researchsoftware.com/wiki/Rim_Weighting_does_not_Converge>

---

## 2. Detect & report empty target categories (silent target drop) — ✅ DONE (Python-side, 2026-05-06)

**Resolved by:** Python-side pre-flight detection in `python/rimpy/_rake.py`. New helper `_detect_empty_target_categories()` runs from inside `rake_with_diagnostics`, `rake_by_with_diagnostics`, and `rake_by_scheme_with_diagnostics` (covers all 6 entry points via delegation). Emits `UserWarning` with deterministic group labels — `Group ('UK'):`, `Group (country='US', gender=2):`, `Scheme group ('UK'):`, `Default-scheme group ('CA'):`. Verified by 16 tests (`tests/test_empty_categories.py`, 27 runs across polars/pandas backends), all 68 tests in the suite pass.

**Deferred (still TODO if needed):** Rust-side `RakeResult.empty_targets` field per the original sketch below. The Python-side `drop_nulls(subset=...)` call in front of the detection covers the standard null-filtering path, but a Rust-side detector would catch any future engine-level filtering edge cases language-binding-agnostically. Revisit if a real bug demands it.

---

### Problem
The user supplies a target proportion for a category that has **zero respondents** in the data. It is logically impossible for an empty cell to represent a non-zero share of the weighted sample, so the supplied target cannot be satisfied — but the user expects either an error or, at minimum, a warning that their target was discarded.

Q surfaces this in its Diagnostics Report, naming the offending categories (e.g., "Males, Less than 18 and Females, Less than 18 are empty but have targets assigned"). **rimpy currently hides it.**

### Current behavior in rimpy
In `src/engine.rs:92-114`, `rake_on_variable` silently skips any target code whose index cache entry is empty:

```rust
for (&code, &target_prop) in target_props {
    if let Some(indices) = index_cache.get(&code) {
        if indices.is_empty() {
            continue;     // ← silently drops the target
        }
        ...
```

A second silent-skip path exists in the same function: if `target_count < 1e-10` (line 99), the category is also skipped. Both branches drop the target without telling anyone.

The remaining (populated) categories then rake normally against their targets, but because the dropped category's mass is never enforced, **the resulting weighted marginal will not match the user's requested distribution**. The user sees `converged: true` and a healthy-looking efficiency, but the output is wrong relative to their intent.

### What's missing
1. **No pre-flight detection** that a target code is missing from the data (or has a populated count of zero after null filtering).
2. **No diagnostic** in `RakeResult` listing dropped categories.
3. **No distinction** between "target proportion was 0 by user choice" (legitimate) and "target proportion was non-zero but cell was empty" (error condition).

### Proposed fix (sketch)
Add a pre-flight scan in `rim_iterate` (or upstream in `arrow_adapter::rake_on_batch`) that, for each `(column, code, target_prop)` triple, checks whether `index_cache[col].get(&code)` is non-empty when `target_prop > 0`. Surface results as:

```rust
pub struct EmptyCategory {
    pub column: String,
    pub code: i64,
    pub target_prop: f64,
}

pub struct RakeResult {
    // ... existing fields
    pub empty_targets: Vec<EmptyCategory>,
}
```

Behavior options (decide before implementing):
- **Strict**: return `Err` if any `target_prop > 0` category is empty. Matches Q.
- **Lenient + diagnostic**: continue (current behavior), but populate `empty_targets` so the caller can warn.
- **Configurable**: add `RakeOpts.error_on_empty_target: bool` (default `true`).

The Python layer should emit a warning (or raise) when `empty_targets` is non-empty so users find out before publishing weighted results.

### Validation requirements (per CLAUDE.md)
Cross-check against R `survey::rake()` and weightipy to confirm whether they error or warn on this case. Match the prevailing convention.

### Reference
Q's writeup: <https://help.qresearchsoftware.com/hc/en-us/articles/4414958629391-Common-Errors-in-Weights> — section "Empty Categories".

---

## 3. Decide & document semantics for `target = 0` on non-empty categories

### Problem
When a user supplies `target = 0` for a category that **does** contain respondents, rimpy's output silently disagrees with the user's literal spec and with every other professional weighting tool except weightipy.

Concrete example reported by a user:

```python
targets = {
    "gender":        {1: 50, 2: 50},
    "education":  {1: 33, 2: 24, 3: 33, 4: 0, 5: 10},
}
```

`education = 4` contains 2 respondents and is given a target of 0.

- **rimpy output:** the 2 respondents land at weight ≈ 0.45 each; weighted % for category 4 is ~0.28%, not 0%. The target the user wrote is not satisfied.
- **Q / SPSS / svyweight output:** weight = 0 for those respondents; weighted % = 0.000000%; the target is satisfied. Effective n drops 320 → 318.

The divergence is silent — no warning, no doc note. A user QC'ing rimpy against Q on the same fixture sees different weights and gets no signal why.

### Current behavior in rimpy
The decisive guard is `src/engine.rs:98-99` inside `rake_on_variable`:

```rust
let target_count = target_prop * n;
if target_count < 1e-10 {
    continue;          // ← target=0 → skip this category entirely
}
```

When `target_prop = 0`, the category is silently **skipped** — no multiplier is applied. Respondents in that category keep whatever weight the *other* dimensions assigned them. A secondary code path at `src/engine.rs:339` replaces any exact `0.0` weight with `1.0` post-iteration, so even if the algorithm naturally produced zeros they would be flipped back to non-zero.

The recently-added detection helper at `python/rimpy/_rake.py:61-78` (`_detect_empty_target_categories`, item #2) only fires when `code not in unique_values and target_value != 0`. The asymmetric `!= 0` filter was correct for #2 (we only wanted to surface non-zero targets that couldn't be satisfied by empty cells). But it leaves this case — populated cell + zero target — completely unsurfaced. `validate_targets` at `_rake.py:720` and `validate_schemes` at `_rake.py:804` have the same filter.

### What other tools do

| Library | Zero-target on non-empty cell | Warning / error | Source |
|---|---|---|---|
| Q, Quantum, SPSS RIM | Hard zero (weight=0, weighted %=0) | Documented behavior | Q "Common Errors in Weights" article |
| R `svyweight::rakesvy` | Hard zero (explicit "assigns weight = 0") | Documented behavior | rdrr.io/cran/svyweight/man/rakesvy |
| R `anesrake` v0.80 | **Refuses to run**: `stop("you cannot rake any variable category to 0 or a negative number")` | Hard error before iteration | `R/rakeonvar.numeric.R:19`, `R/rakeonvar.factor.R:18`, `R/rakeonvar.default.R:20` |
| R `survey::rake` | Undefined — docs limit convergence guarantee to tables with no zero cells | Silent | <https://r-survey.r-forge.r-project.org/survey/html/rake.html> |
| `weightipy` | Substitutes `0 → 0.00000001` | Silent | `weightipy/internal/rim.py` `rakeonvar()` |
| **rimpy** | Silent skip via `engine.rs:99` | Silent | this codebase |

Three coherent camps emerge: **hard zero** (Q / SPSS / svyweight), **refuse** (anesrake), and **near-zero silent skip** (weightipy and rimpy — different mechanisms, similar behavior). No industry consensus.

### What's missing
1. **No decision** on which camp rimpy belongs to. The current behavior is the result of a divide-by-zero guard rather than a deliberate contract.
2. **No detection** for the populated-cell + zero-target case. The existing helper's `!= 0` filter actively excludes it.
3. **No warning** when output diverges from the user's literal spec.
4. **No documentation** of the in-house Povaddo workaround (supplying `0.001%` to "show as ~0% without losing respondents" — a workflow convention, not a library contract).

### Design options
- **(a) Hard zero — match Q / SPSS / svyweight.** Drive the weights to 0 in the algorithm. Drop the `target_count < 1e-10` early-continue and the `engine.rs:339` zero-to-one flip for legitimately-zeroed categories. Emit `UserWarning` so existing pipelines don't silently change. **Recommended (user's lean).** Matches the most widely-used professional tools and respects the literal targets dict.
- **(b) Keep current silent-skip but document loudly.** Re-frame as "exclude this dimension's marginal constraint for this category while retaining respondents' cross-dimensional weight." Add `UserWarning` describing divergence from Q. Defensible if documented but harder to communicate to survey researchers familiar with Q.
- **(c) Refuse — match anesrake.** Raise `ValueError` on any zero target with non-empty cells. Forces explicit intent (drop respondents upstream, or supply a near-zero positive target). Most disruptive to existing rimpy users.

### Proposed fix (sketch)
**Recommendation: option (a) — hard zero with warning.**

- Remove `if target_count < 1e-10 { continue; }` at `engine.rs:99`; let the multiplicative update apply (multiplier becomes 0, weights for that category become 0).
- Revisit `engine.rs:339` post-iteration `0.0 → 1.0` flip. This was a numerical safety net; it must not undo legitimate zero weights. Likely: track which categories had `target_prop == 0` explicitly and skip the flip for their rows.
- Add Python-side `_detect_zero_target_on_populated_cell()` helper paralleling `_detect_empty_target_categories`, wired into all 6 entry points the same way (private `_warning_stacklevel` kwarg). Emit `UserWarning` whenever the user supplies `target = 0` for a populated cell, regardless of whether the chosen contract is (a), (b), or (c) — the user should always know they're hitting an edge case.
- Cross-check the resulting weights and weighted marginals against R `svyweight::rakesvy` on the user's fixture before merging.

### Povaddo near-zero convention
The in-house RPM workflow of supplying `0.001%` for "show as ~0% without losing respondents" is a **workflow convention external to rimpy**, not a library contract. Document this explicitly. Open question: codify a constant like `rimpy.NEAR_ZERO = 1e-8` for users who want a sanctioned shortcut, or leave it as user responsibility?

### Validation requirements (per CLAUDE.md)
Any algorithm change here must be cross-checked against R `survey::rake()`, weightipy, and ideally one anesrake run on the same fixture (anesrake will error out — confirm that's the case). Lock the contract before shipping. The user's 320-row education fixture is a good acceptance test.

### Reference
- Q "Common Errors in Weights": <https://help.qresearchsoftware.com/hc/en-us/articles/4414958629391-Common-Errors-in-Weights>
- anesrake v0.80 source at `C:\Users\lipov\SynologyDrive\sandbox\anesrake_0.80\anesrake\R\` (rakeonvar.{numeric,factor,default}.R)
- weightipy source: <https://github.com/kaitumisuuringute-keskus/Weightipy/blob/main/weightipy/internal/rim.py>
- R `survey::rake` docs: <https://r-survey.r-forge.r-project.org/survey/html/rake.html>
- R `svyweight::rakesvy` docs: <https://rdrr.io/cran/svyweight/man/rakesvy.html>

### Out of scope for this item
- Item #1 (contradictory targets / stall vs. converged).
- Item #4 (target-dict key validation / tuple syntax).

---

## 4. Validate target-dict keys + accept tuple keys for combined-category targets

### Problem
Two related target-dict footguns currently produce silently-wrong weights:

**A. Arithmetic-evaluated keys (Python literal trap).** A user writing
```python
"education": {1: 33, 2: 20, 3: 33, 4-5: 14}
```
intends to give categories 4 and 5 a shared target of 14. But Python evaluates `4-5` at dict-construction time, so the dict that reaches rimpy is `{1: 33, 2: 20, 3: 33, -1: 14}`. rimpy silently accepts the `-1` key, fails to map it to any real category, and produces wrong weights.

**B. No first-class combined-category syntax.** Combining sparse categories before weighting is a routine survey operation. Users currently have to merge the underlying data column upstream, since rimpy's target dict has no `(4, 5): 14` form. This pushes data-prep into every notebook and makes the fix for problem A (use tuple syntax) ironic — the suggestion only works if tuple syntax is actually supported.

### Current behavior in rimpy

**Unknown keys (A):**
- `validate_targets` at `python/rimpy/_rake.py:720` and `validate_schemes` at `_rake.py:804` warn about unknown keys, but only when `target_value != 0`. So `{-1: 0}` passes silently, and even `{-1: 14}` only emits a warning if the user explicitly calls `validate_targets()` — the regular `rake()` path never invokes this check.
- The FFI layer at `src/lib.rs:184-204` (`extract_targets`) happily accepts the `-1` key as a valid `i64`; nothing downstream rejects it.

**Tuple keys (B):**
- `src/lib.rs:184-204` (`extract_targets`) only accepts `i64` or `f64` keys (line 193: `code_key.extract::<i64>().or_else(|_| code_key.extract::<f64>().map(|f| f as i64))?`). A tuple key fails extraction with an opaque PyO3 error.
- No Python-side normalization expands tuple keys before the FFI call.

### What's missing
1. **Unknown-key validation fired automatically inside `rake()` and `rake_by()` paths**, not gated on `target_value != 0` and not requiring the user to opt in by calling `validate_targets`.
2. **An actionable error message** that points the user toward the tuple-key fix when an arithmetic-evaluated key looks like a typo (e.g., a non-positive key like `-1` when the column has only positive codes).
3. **First-class tuple-key support** in the rake API.

### Proposed fix (sketch)

**Sub-problem A — unknown-key validation:**
- Add validation that runs from inside the rake pipeline (alongside the existing `_detect_empty_target_categories` invocations in `rake_with_diagnostics`, `rake_by_with_diagnostics`, `rake_by_scheme_with_diagnostics`).
- Validate every target key against the column's unique values **regardless of target value**.
- Decision: warn or raise? Lean: **raise** (`KeyError` or `ValueError`). Unknown keys are unambiguously a bug in the caller's targets dict; they're not a development-time partial-data scenario like #2.
- Error message format: `"Key -1 not found in categories [1, 2, 3, 4, 5] for column 'education'. If you meant to combine categories 4 and 5, use tuple syntax: (4, 5): 14."`

**Sub-problem B — tuple-key combined-category syntax:**
- Accept `{(4, 5): 14}` as "categories 4 and 5 share a combined weighted target of 14".
- **Python-side normalization preferred** — keeps the Rust engine pure and avoids changing the FFI contract. Two implementation paths to choose between:
  - **Split**: expand `{(4, 5): 14}` to `{4: 14, 5: 14}` with shared-target bookkeeping (each respondent in 4 or 5 contributes to the combined marginal of 14).
  - **Virtual code**: synthesize a new internal code (e.g., the smallest unused integer) and remap the data column to it for raking, then map back for the output.
  - Decision deferred until this item is implemented; both are reasonable.
- Open question: how does this interact with `_detect_empty_target_categories` (a tuple is "empty" only if *all* its codes have zero rows) and with the zero-target contract decided in item #3?

### Validation requirements (per CLAUDE.md)
- Cross-check the tuple syntax against weightipy and other RIM libraries — survey research has standard combined-category semantics we shouldn't reinvent. If a popular library uses `{4: 14, 5: 14}` to mean "separately, each at 14%" then `(4, 5): 14` to mean "combined at 14%" is the natural delta.
- Test fixtures must cover: (i) `-1` from `4-5` typo with non-zero target, (ii) `-1` with zero target (catches the existing `!= 0` filter bug), (iii) tuple-key happy path, (iv) tuple-key where one element is missing from the data (interaction with item #2).

### Reference
- Item #2 `_detect_empty_target_categories` at `python/rimpy/_rake.py:61-78` — model for the new validation helper.
- `extract_targets` at `src/lib.rs:184-204` — FFI extraction logic to update if going the FFI route (not recommended).

### Out of scope for this item
- Item #3 (zero-target semantics decision) — different contract decision.
- Item #1 (contradictory targets).
