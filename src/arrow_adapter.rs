//! Language-agnostic Arrow middleware for RIM raking.
//!
//! This module sits between the pure engine (`engine.rs`) and any language
//! binding (`lib.rs` for Python, future R/Julia bindings). It operates on
//! Arrow RecordBatches and has **zero PyO3 dependency**.
//!
//! Three tiers:
//!   1. Column helpers — extract, mask, append, group keys
//!   2. `rake_on_batch` — single-group raking with null handling + total scaling
//!   3. `rake_batch*` — high-level orchestrators returning RecordBatch with weight column

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Float64Type, Int64Type, Schema};
use indexmap::IndexMap;
use rayon::prelude::*;

use crate::engine::{self, RakeOpts, RakeResult};

// ---------------------------------------------------------------------------
// Tier 1: Column helpers
// ---------------------------------------------------------------------------

/// Extract a column from a RecordBatch as `Vec<i64>`.
/// Handles Int8/Int16/Int32/Int64/UInt8/UInt16/UInt32/Float64 types.
pub fn extract_i64_column(batch: &RecordBatch, col_name: &str) -> Result<Vec<i64>, String> {
    let col_idx = batch
        .schema()
        .index_of(col_name)
        .map_err(|_| format!("Column '{col_name}' not found in Arrow data"))?;

    let array = batch.column(col_idx);

    match array.data_type() {
        DataType::Int64 => {
            let arr = array.as_primitive::<Int64Type>();
            Ok(arr.values().to_vec())
        }
        DataType::Float64 => {
            let arr = array.as_primitive::<Float64Type>();
            Ok(arr.values().iter().map(|&f| f as i64).collect())
        }
        DataType::Int32 => {
            let arr = array.as_primitive::<arrow::datatypes::Int32Type>();
            Ok(arr.values().iter().map(|&v| v as i64).collect())
        }
        DataType::Int8 => {
            let arr = array.as_primitive::<arrow::datatypes::Int8Type>();
            Ok(arr.values().iter().map(|&v| v as i64).collect())
        }
        DataType::Int16 => {
            let arr = array.as_primitive::<arrow::datatypes::Int16Type>();
            Ok(arr.values().iter().map(|&v| v as i64).collect())
        }
        DataType::UInt8 => {
            let arr = array.as_primitive::<arrow::datatypes::UInt8Type>();
            Ok(arr.values().iter().map(|&v| v as i64).collect())
        }
        DataType::UInt16 => {
            let arr = array.as_primitive::<arrow::datatypes::UInt16Type>();
            Ok(arr.values().iter().map(|&v| v as i64).collect())
        }
        DataType::UInt32 => {
            let arr = array.as_primitive::<arrow::datatypes::UInt32Type>();
            Ok(arr.values().iter().map(|&v| v as i64).collect())
        }
        dt => Err(format!(
            "Column '{col_name}' has unsupported type {dt:?}, expected integer or float"
        )),
    }
}

/// Code assigned to null slots of a string column.
///
/// `extract_i64_column` leaves raw buffer contents in null slots, which is only
/// harmless because `build_valid_mask` drops those rows — and only when
/// `drop_nulls` is true. With `drop_nulls = false` the code still reaches the
/// engine, so it must be one no real category can be assigned.
const NULL_CODE: i64 = i64::MIN;

/// Category label -> integer code for one string-valued column.
pub type CategoryDict = IndexMap<String, i64>;

/// Per-column category dictionaries; `None` for a numeric column.
pub type ColumnDicts = HashMap<String, Option<CategoryDict>>;

/// A target-dict key as handed over by a language binding: either an integer
/// category code, or a category label for a string-valued column.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TargetKey {
    Int(i64),
    Str(String),
}

/// Decode a string-like Arrow array to owned Strings, or `None` if the array
/// is not string-like.
///
/// Handles Utf8, LargeUtf8 and Utf8View, plus a Dictionary over any of them
/// with any index width (polars Categorical/Enum and pandas category arrive as
/// `Dictionary(UInt32|UInt8|Int8, LargeUtf8|Utf8View)`, so the index type
/// cannot be matched on directly). Everything is cast to LargeUtf8 first so
/// there is a single read path.
fn as_large_utf8(array: &ArrayRef) -> Option<ArrayRef> {
    fn is_stringy(dt: &DataType) -> bool {
        matches!(
            dt,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        )
    }

    let stringy = match array.data_type() {
        dt if is_stringy(dt) => true,
        DataType::Dictionary(_, value_type) => is_stringy(value_type),
        _ => false,
    };
    if !stringy {
        return None;
    }
    arrow::compute::cast(array, &DataType::LargeUtf8).ok()
}

fn decode_string_column(array: &ArrayRef) -> Option<Vec<Option<String>>> {
    let large = as_large_utf8(array)?;
    let arr = large.as_string::<i64>();
    Some(
        (0..arr.len())
            .map(|i| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i).to_string())
                }
            })
            .collect(),
    )
}

/// Extract a target column as integer codes, plus the string dictionary when
/// the column held categories as text.
///
/// Numeric columns behave exactly as before and carry no dictionary. String
/// columns get codes assigned in first-appearance order; the same dictionary
/// must then map that column's target keys (see `resolve_target_keys`), which
/// makes the whole thing equivalent to the caller having recoded the strings to
/// integers by hand. `engine.rs` never sees a string either way.
pub fn extract_codes_column(
    batch: &RecordBatch,
    col_name: &str,
) -> Result<(Vec<i64>, Option<CategoryDict>), String> {
    let col_idx = batch
        .schema()
        .index_of(col_name)
        .map_err(|_| format!("Column '{col_name}' not found in Arrow data"))?;

    let array = batch.column(col_idx);

    // Dictionary columns (polars Categorical/Enum, pandas category) are already
    // encoded: walk the indices and touch each distinct label once, instead of
    // materializing one string per row via a cast.
    if matches!(array.data_type(), DataType::Dictionary(_, _))
        && let Some(values) = as_large_utf8(array.as_any_dictionary().values())
    {
        let dictionary = array.as_any_dictionary();
        let values = values.as_string::<i64>();
        let keys = dictionary.normalized_keys();
        let nulls = array.logical_nulls();

        let mut slot_code: Vec<i64> = vec![-1; values.len()];
        let mut dict = CategoryDict::new();
        let codes = (0..array.len())
            .map(|i| {
                if nulls.as_ref().is_some_and(|n| n.is_null(i)) {
                    return NULL_CODE;
                }
                let slot = keys[i];
                if values.is_null(slot) {
                    return NULL_CODE;
                }
                if slot_code[slot] < 0 {
                    let code = dict.len() as i64;
                    dict.insert(values.value(slot).to_string(), code);
                    slot_code[slot] = code;
                }
                slot_code[slot]
            })
            .collect();
        return Ok((codes, Some(dict)));
    }

    if let Some(large) = as_large_utf8(array) {
        // Look up borrowed &str and only allocate when a category is new, so
        // this stays one allocation per distinct category rather than per row.
        let arr = large.as_string::<i64>();
        let mut dict = CategoryDict::new();
        let codes = (0..arr.len())
            .map(|i| {
                if arr.is_null(i) {
                    return NULL_CODE;
                }
                let label = arr.value(i);
                match dict.get(label) {
                    Some(&code) => code,
                    None => {
                        let code = dict.len() as i64;
                        dict.insert(label.to_string(), code);
                        code
                    }
                }
            })
            .collect();
        return Ok((codes, Some(dict)));
    }

    Ok((extract_i64_column(batch, col_name)?, None))
}

/// Map one column's target keys onto the codes used for its data.
pub fn resolve_target_keys(
    col: &str,
    keys: &HashMap<TargetKey, f64>,
    dict: Option<&CategoryDict>,
) -> Result<HashMap<i64, f64>, String> {
    let mut out = HashMap::with_capacity(keys.len());
    for (key, &value) in keys {
        let code = match (key, dict) {
            (TargetKey::Int(code), None) => *code,
            (TargetKey::Str(label), Some(d)) => *d.get(label.as_str()).ok_or_else(|| {
                format!("Target key {label:?} for column '{col}' is not a category in the data")
            })?,
            (TargetKey::Str(label), None) => {
                return Err(format!(
                    "Target key {label:?} for column '{col}' is text, but that column \
                     holds numeric category codes"
                ));
            }
            (TargetKey::Int(code), Some(_)) => {
                return Err(format!(
                    "Target key {code} for column '{col}' is numeric, but that column \
                     holds text category codes"
                ));
            }
        };
        out.insert(code, value);
    }
    Ok(out)
}

/// Map a whole target set onto data codes, preserving column order.
pub fn resolve_targets(
    targets: &IndexMap<String, HashMap<TargetKey, f64>>,
    dicts: &ColumnDicts,
) -> Result<IndexMap<String, HashMap<i64, f64>>, String> {
    let mut out = IndexMap::with_capacity(targets.len());
    for (col, keys) in targets {
        let dict = dicts.get(col).and_then(|d| d.as_ref());
        out.insert(col.clone(), resolve_target_keys(col, keys, dict)?);
    }
    Ok(out)
}

/// Build a null/valid mask from Arrow null bitmaps for the given columns.
/// Returns a boolean vec: `true` = valid (no nulls in any target column).
pub fn build_valid_mask(
    batch: &RecordBatch,
    target_columns: &[String],
) -> Result<Vec<bool>, String> {
    let n_rows = batch.num_rows();
    let mut valid = vec![true; n_rows];

    for col_name in target_columns {
        let col_idx = batch
            .schema()
            .index_of(col_name)
            .map_err(|_| format!("Column '{col_name}' not found"))?;
        let array = batch.column(col_idx);
        if let Some(nulls) = array.nulls() {
            for (i, v) in valid.iter_mut().enumerate() {
                if !nulls.is_valid(i) {
                    *v = false;
                }
            }
        }
    }

    Ok(valid)
}

/// Append a Float64 weight column to a RecordBatch.
/// Existing columns are Arc-shared (zero-copy). Only the new column is allocated.
pub fn append_weight_column(
    batch: &RecordBatch,
    weights: Vec<f64>,
    column_name: &str,
) -> Result<RecordBatch, String> {
    let weight_array: ArrayRef = Arc::new(Float64Array::from(weights));
    let mut fields: Vec<Arc<Field>> = batch.schema().fields().iter().cloned().collect();
    fields.push(Arc::new(Field::new(column_name, DataType::Float64, false)));
    let new_schema = Arc::new(Schema::new(fields));
    let mut all_columns: Vec<ArrayRef> = batch.columns().to_vec();
    all_columns.push(weight_array);
    RecordBatch::try_new(new_schema, all_columns)
        .map_err(|e| format!("Failed to append weight column: {e}"))
}

/// Extract a group column as String keys.
/// Handles any string-like type (Utf8, LargeUtf8, Utf8View, or a Dictionary
/// over them — i.e. polars Categorical/Enum and pandas category), plus Int64,
/// Int32 and Float64. Null values become `"__null__"`.
pub fn extract_group_keys(batch: &RecordBatch, group_col: &str) -> Result<Vec<String>, String> {
    let col_idx = batch
        .schema()
        .index_of(group_col)
        .map_err(|_| format!("Group column '{group_col}' not found"))?;
    let array = batch.column(col_idx);

    if let Some(values) = decode_string_column(array) {
        return Ok(values
            .into_iter()
            .map(|v| v.unwrap_or_else(|| "__null__".to_string()))
            .collect());
    }

    match array.data_type() {
        DataType::Int64 => {
            let arr = array.as_primitive::<Int64Type>();
            Ok((0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        "__null__".to_string()
                    } else {
                        arr.value(i).to_string()
                    }
                })
                .collect())
        }
        DataType::Int32 => {
            let arr = array.as_primitive::<arrow::datatypes::Int32Type>();
            Ok((0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        "__null__".to_string()
                    } else {
                        arr.value(i).to_string()
                    }
                })
                .collect())
        }
        DataType::Float64 => {
            let arr = array.as_primitive::<Float64Type>();
            Ok((0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        "__null__".to_string()
                    } else {
                        arr.value(i).to_string()
                    }
                })
                .collect())
        }
        dt => Err(format!("Unsupported group column type: {dt:?}")),
    }
}

// ---------------------------------------------------------------------------
// Tier 2: Single-group raking
// ---------------------------------------------------------------------------

/// Run raking on a single batch with null handling and total scaling.
/// Returns `(full_weights, n_valid, RakeResult)`.
pub fn rake_on_batch(
    batch: &RecordBatch,
    target_columns: &[String],
    targets: &IndexMap<String, HashMap<TargetKey, f64>>,
    opts: &RakeOpts,
    drop_nulls: bool,
    total: Option<f64>,
) -> Result<(Vec<f64>, usize, RakeResult), String> {
    let n_rows = batch.num_rows();

    // Extract all target columns, keeping any string dictionary alongside
    let mut columns: HashMap<String, Vec<i64>> = HashMap::new();
    let mut dicts = ColumnDicts::new();
    for col_name in target_columns {
        let (codes, dict) = extract_codes_column(batch, col_name)?;
        columns.insert(col_name.clone(), codes);
        dicts.insert(col_name.clone(), dict);
    }
    let targets = &resolve_targets(targets, &dicts)?;

    // Build valid mask from Arrow null bitmaps
    let valid_mask = if drop_nulls {
        build_valid_mask(batch, target_columns)?
    } else {
        vec![true; n_rows]
    };

    let valid_indices: Vec<usize> = valid_mask
        .iter()
        .enumerate()
        .filter(|&(_, v)| *v)
        .map(|(i, _)| i)
        .collect();

    let n_valid = valid_indices.len();

    // Handle empty case
    if n_valid == 0 {
        let result = RakeResult {
            weights: vec![],
            iterations: 0,
            converged: true,
            efficiency: 100.0,
            weight_min: 1.0,
            weight_max: 1.0,
        };
        return Ok((vec![1.0; n_rows], 0, result));
    }

    // Filter to valid rows (vectorized gather)
    let filtered_columns: HashMap<String, Vec<i64>> = if n_valid == n_rows {
        columns
    } else {
        columns
            .iter()
            .map(|(name, data)| {
                let filtered: Vec<i64> = valid_indices.iter().map(|&i| data[i]).collect();
                (name.clone(), filtered)
            })
            .collect()
    };

    // Run pure Rust engine
    let col_refs: HashMap<String, &[i64]> = filtered_columns
        .iter()
        .map(|(k, v)| (k.clone(), v.as_slice()))
        .collect();

    let result = engine::rim_iterate(&col_refs, targets, opts)?;

    // Scatter weights into full array (1.0 for null/missing rows)
    let mut full_weights = vec![1.0_f64; n_rows];
    if n_valid == n_rows {
        full_weights.copy_from_slice(&result.weights);
    } else {
        for (i, &idx) in valid_indices.iter().enumerate() {
            full_weights[idx] = result.weights[i];
        }
    }

    // Scale to total
    if let Some(target_total) = total
        && target_total > 0.0
    {
        let current_sum: f64 = if n_valid == n_rows {
            full_weights.iter().sum()
        } else {
            valid_indices.iter().map(|&i| full_weights[i]).sum()
        };
        if current_sum > 0.0 {
            let factor = target_total / current_sum;
            if n_valid == n_rows {
                for w in full_weights.iter_mut() {
                    *w *= factor;
                }
            } else {
                for &i in &valid_indices {
                    full_weights[i] *= factor;
                }
            }
        }
    }

    Ok((full_weights, n_valid, result))
}

// ---------------------------------------------------------------------------
// Tier 3: High-level orchestrators
// ---------------------------------------------------------------------------

/// Per-group result returned by grouped operations.
#[derive(Debug, Clone)]
pub struct GroupRakeResult {
    pub group_key: String,
    pub n_valid: usize,
    pub result: RakeResult,
}

/// Single-group raking. Returns RecordBatch with weight column appended + diagnostics.
#[allow(clippy::too_many_arguments)]
pub fn rake_batch(
    batch: &RecordBatch,
    target_columns: &[String],
    targets: &IndexMap<String, HashMap<TargetKey, f64>>,
    weight_column: &str,
    opts: &RakeOpts,
    drop_nulls: bool,
    total: Option<f64>,
) -> Result<(RecordBatch, usize, RakeResult), String> {
    let (full_weights, n_valid, result) =
        rake_on_batch(batch, target_columns, targets, opts, drop_nulls, total)?;
    let result_batch = append_weight_column(batch, full_weights, weight_column)?;
    Ok((result_batch, n_valid, result))
}

/// Grouped raking with same targets for all groups.
///
/// Partitions rows by group column(s), rakes each group in parallel via rayon,
/// assembles a single weight column.
#[allow(clippy::too_many_arguments)]
pub fn rake_batch_grouped(
    batch: &RecordBatch,
    target_columns: &[String],
    targets: &IndexMap<String, HashMap<TargetKey, f64>>,
    group_columns: &[String],
    weight_column: &str,
    opts: &RakeOpts,
    drop_nulls: bool,
    total: Option<f64>,
) -> Result<(RecordBatch, Vec<GroupRakeResult>), String> {
    let n_rows = batch.num_rows();

    // Build composite group keys
    let group_keys = build_composite_keys(batch, group_columns)?;

    // Partition rows by group
    let group_row_indices = partition_by_group(&group_keys);

    // Extract all target columns once, over the whole batch: a string column's
    // dictionary must be shared by every group or codes would disagree.
    let mut all_columns: HashMap<String, Vec<i64>> = HashMap::new();
    let mut dicts = ColumnDicts::new();
    for col_name in target_columns {
        let (codes, dict) = extract_codes_column(batch, col_name)?;
        all_columns.insert(col_name.clone(), codes);
        dicts.insert(col_name.clone(), dict);
    }
    let targets = &resolve_targets(targets, &dicts)?;

    // Build valid mask
    let valid_mask = if drop_nulls {
        build_valid_mask(batch, target_columns)?
    } else {
        vec![true; n_rows]
    };

    // Process groups in parallel
    let group_entries: Vec<(String, Vec<usize>)> = group_row_indices.into_iter().collect();

    let group_results: Vec<(String, Vec<usize>, RakeResult)> = group_entries
        .into_par_iter()
        .map(|(key, row_indices)| {
            let (valid_indices, result) = rake_group(
                &row_indices,
                &all_columns,
                target_columns,
                targets,
                &valid_mask,
                opts,
            )?;
            let _ = valid_indices; // used for weight scattering below
            Ok((key, row_indices, result))
        })
        .collect::<Result<_, String>>()?;

    // Assemble full weight array + collect diagnostics
    let mut full_weights = vec![1.0_f64; n_rows];
    let mut diagnostics: Vec<GroupRakeResult> = Vec::new();

    for (key, row_indices, result) in &group_results {
        let valid_indices: Vec<usize> = row_indices
            .iter()
            .filter(|&&i| valid_mask[i])
            .copied()
            .collect();

        for (i, &idx) in valid_indices.iter().enumerate() {
            if i < result.weights.len() {
                full_weights[idx] = result.weights[i];
            }
        }

        diagnostics.push(GroupRakeResult {
            group_key: key.clone(),
            n_valid: valid_indices.len(),
            result: result.clone(),
        });
    }

    // Scale to total
    if let Some(target_total) = total
        && target_total > 0.0
    {
        let current_sum: f64 = full_weights.iter().sum();
        if current_sum > 0.0 {
            let factor = target_total / current_sum;
            for w in full_weights.iter_mut() {
                *w *= factor;
            }
        }
    }

    let result_batch = append_weight_column(batch, full_weights, weight_column)?;
    Ok((result_batch, diagnostics))
}

/// Per-group scheme raking with different targets per group.
///
/// Handles `group_totals` (nested weighting) and global `total` scaling.
#[allow(clippy::too_many_arguments)]
pub fn rake_batch_by_scheme(
    batch: &RecordBatch,
    group_column: &str,
    schemes: &HashMap<String, IndexMap<String, HashMap<TargetKey, f64>>>,
    default_scheme: Option<&IndexMap<String, HashMap<TargetKey, f64>>>,
    weight_column: &str,
    opts: &RakeOpts,
    drop_nulls: bool,
    group_totals: Option<&HashMap<String, f64>>,
    total: Option<f64>,
) -> Result<(RecordBatch, Vec<GroupRakeResult>), String> {
    let n_rows = batch.num_rows();

    // Extract group column keys
    let group_keys = extract_group_keys(batch, group_column)?;

    // Partition rows by group
    let group_row_indices = partition_by_group(&group_keys);

    // Collect all target columns from all schemes
    let mut all_target_cols: HashSet<String> = HashSet::new();
    for scheme_targets in schemes.values() {
        for col in scheme_targets.keys() {
            all_target_cols.insert(col.clone());
        }
    }
    if let Some(dt) = default_scheme {
        for col in dt.keys() {
            all_target_cols.insert(col.clone());
        }
    }

    // Extract all needed columns once, over the whole batch, so every scheme
    // referencing the same string column shares one code space.
    let mut all_columns: HashMap<String, Vec<i64>> = HashMap::new();
    let mut dicts = ColumnDicts::new();
    for col_name in &all_target_cols {
        if batch.schema().index_of(col_name).is_ok() {
            let (codes, dict) = extract_codes_column(batch, col_name)?;
            all_columns.insert(col_name.clone(), codes);
            dicts.insert(col_name.clone(), dict);
        }
    }

    // Resolve every scheme against those shared dictionaries up front.
    let schemes: HashMap<String, IndexMap<String, HashMap<i64, f64>>> = schemes
        .iter()
        .map(|(k, t)| Ok((k.clone(), resolve_targets(t, &dicts)?)))
        .collect::<Result<_, String>>()?;
    let default_scheme = match default_scheme {
        Some(dt) => Some(resolve_targets(dt, &dicts)?),
        None => None,
    };
    let default_scheme = default_scheme.as_ref();

    // Build valid mask (across ALL target columns that exist)
    let existing_target_cols: Vec<String> = all_target_cols
        .iter()
        .filter(|c| all_columns.contains_key(c.as_str()))
        .cloned()
        .collect();
    let valid_mask = if drop_nulls {
        build_valid_mask(batch, &existing_target_cols)?
    } else {
        vec![true; n_rows]
    };

    // Process each group in parallel
    let group_entries: Vec<(String, Vec<usize>)> = group_row_indices.into_iter().collect();

    let group_results: Vec<(String, Vec<usize>, RakeResult)> = group_entries
        .into_par_iter()
        .map(|(key, row_indices)| {
            // Look up targets for this group
            let group_targets = match schemes.get(&key) {
                Some(t) => t.clone(),
                None => match default_scheme {
                    Some(dt) => dt.clone(),
                    None => {
                        // No scheme — weight = 1.0
                        let result = RakeResult {
                            weights: vec![],
                            iterations: 0,
                            converged: true,
                            efficiency: 100.0,
                            weight_min: 1.0,
                            weight_max: 1.0,
                        };
                        return Ok((key, row_indices, result));
                    }
                },
            };

            let target_columns: Vec<String> = group_targets.keys().cloned().collect();

            let (_, result) = rake_group(
                &row_indices,
                &all_columns,
                &target_columns,
                &group_targets,
                &valid_mask,
                opts,
            )?;

            Ok((key, row_indices, result))
        })
        .collect::<Result<_, String>>()?;

    // Assemble full weight array + collect diagnostics
    let mut full_weights = vec![1.0_f64; n_rows];
    let mut diagnostics: Vec<GroupRakeResult> = Vec::new();

    for (key, row_indices, result) in &group_results {
        let valid_indices: Vec<usize> = row_indices
            .iter()
            .filter(|&&i| valid_mask[i])
            .copied()
            .collect();

        for (i, &idx) in valid_indices.iter().enumerate() {
            if i < result.weights.len() {
                full_weights[idx] = result.weights[i];
            }
        }

        diagnostics.push(GroupRakeResult {
            group_key: key.clone(),
            n_valid: valid_indices.len(),
            result: result.clone(),
        });
    }

    // Apply group_totals correction
    if let Some(gt) = group_totals {
        let total_pct: f64 = gt.values().sum();
        let normalized: HashMap<String, f64> = if total_pct > 1.5 {
            gt.iter().map(|(k, &v)| (k.clone(), v / 100.0)).collect()
        } else {
            gt.clone()
        };

        for (group_key, row_indices, _result) in group_results.iter() {
            if let Some(&target_prop) = normalized.get(group_key) {
                let target_sum = target_prop * n_rows as f64;
                let current_sum: f64 = row_indices.iter().map(|&i| full_weights[i]).sum();
                if current_sum > 0.0 {
                    let factor = target_sum / current_sum;
                    for &i in row_indices {
                        full_weights[i] *= factor;
                    }
                }
            }
        }
    }

    // Scale to global total
    if let Some(target_total) = total
        && target_total > 0.0
    {
        let current_sum: f64 = full_weights.iter().sum();
        if current_sum > 0.0 {
            let factor = target_total / current_sum;
            for w in full_weights.iter_mut() {
                *w *= factor;
            }
        }
    }

    let result_batch = append_weight_column(batch, full_weights, weight_column)?;
    Ok((result_batch, diagnostics))
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Build composite group keys from one or more columns.
fn build_composite_keys(
    batch: &RecordBatch,
    group_columns: &[String],
) -> Result<Vec<String>, String> {
    let n_rows = batch.num_rows();
    if group_columns.len() == 1 {
        extract_group_keys(batch, &group_columns[0])
    } else {
        let per_col: Vec<Vec<String>> = group_columns
            .iter()
            .map(|col| extract_group_keys(batch, col))
            .collect::<Result<_, String>>()?;
        Ok((0..n_rows)
            .map(|i| {
                per_col
                    .iter()
                    .map(|col| col[i].as_str())
                    .collect::<Vec<_>>()
                    .join("||")
            })
            .collect())
    }
}

/// Partition rows by group key, returning (group_key, row_indices) in insertion order.
fn partition_by_group(group_keys: &[String]) -> IndexMap<String, Vec<usize>> {
    let mut map: IndexMap<String, Vec<usize>> = IndexMap::new();
    for (i, key) in group_keys.iter().enumerate() {
        map.entry(key.clone()).or_default().push(i);
    }
    map
}

/// Rake a single group's rows.
///
/// Returns `(valid_indices, RakeResult)` where `valid_indices` are the original
/// row positions within the full batch that were raked.
fn rake_group(
    row_indices: &[usize],
    all_columns: &HashMap<String, Vec<i64>>,
    target_columns: &[String],
    targets: &IndexMap<String, HashMap<i64, f64>>,
    valid_mask: &[bool],
    opts: &RakeOpts,
) -> Result<(Vec<usize>, RakeResult), String> {
    let valid_indices: Vec<usize> = row_indices
        .iter()
        .filter(|&&i| valid_mask[i])
        .copied()
        .collect();

    if valid_indices.is_empty() {
        return Ok((
            valid_indices,
            RakeResult {
                weights: vec![],
                iterations: 0,
                converged: true,
                efficiency: 100.0,
                weight_min: 1.0,
                weight_max: 1.0,
            },
        ));
    }

    let group_column_data: HashMap<String, Vec<i64>> = target_columns
        .iter()
        .filter_map(|name| {
            all_columns.get(name).map(|data| {
                let filtered: Vec<i64> = valid_indices.iter().map(|&i| data[i]).collect();
                (name.clone(), filtered)
            })
        })
        .collect();

    let col_refs: HashMap<String, &[i64]> = group_column_data
        .iter()
        .map(|(k, v)| (k.clone(), v.as_slice()))
        .collect();

    let result = engine::rim_iterate(&col_refs, targets, opts)?;
    Ok((valid_indices, result))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        DictionaryArray, Int64Array, LargeStringArray, StringArray, StringViewArray,
    };
    use arrow::datatypes::{Int8Type, UInt8Type, UInt32Type};

    fn batch(name: &str, array: ArrayRef) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            name,
            array.data_type().clone(),
            true,
        )]));
        RecordBatch::try_new(schema, vec![array]).unwrap()
    }

    fn abc() -> Vec<Option<String>> {
        vec![Some("a".to_string()), None, Some("b".to_string())]
    }

    #[test]
    fn decodes_every_string_layout() {
        let utf8: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None, Some("b")]));
        let large: ArrayRef = Arc::new(LargeStringArray::from(vec![Some("a"), None, Some("b")]));
        let view: ArrayRef = Arc::new(StringViewArray::from(vec![Some("a"), None, Some("b")]));
        for array in [utf8, large, view] {
            assert_eq!(decode_string_column(&array), Some(abc()));
        }
    }

    #[test]
    fn decodes_dictionaries_of_any_index_width() {
        // polars Categorical arrives as UInt32, Enum as UInt8, pandas category
        // as Int8 — the index type cannot be matched on directly.
        let d32: ArrayRef = Arc::new(
            vec![Some("a"), None, Some("b")]
                .into_iter()
                .collect::<DictionaryArray<UInt32Type>>(),
        );
        let d8: ArrayRef = Arc::new(
            vec![Some("a"), None, Some("b")]
                .into_iter()
                .collect::<DictionaryArray<UInt8Type>>(),
        );
        let di8: ArrayRef = Arc::new(
            vec![Some("a"), None, Some("b")]
                .into_iter()
                .collect::<DictionaryArray<Int8Type>>(),
        );
        for array in [d32, d8, di8] {
            assert_eq!(decode_string_column(&array), Some(abc()));
        }
    }

    #[test]
    fn numeric_arrays_are_not_string_like() {
        let array: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        assert_eq!(decode_string_column(&array), None);
    }

    #[test]
    fn numeric_column_extracts_without_a_dictionary() {
        let b = batch("c", Arc::new(Int64Array::from(vec![1, 2, 3])));
        let (codes, dict) = extract_codes_column(&b, "c").unwrap();
        assert_eq!(codes, vec![1, 2, 3]);
        assert!(dict.is_none());
    }

    #[test]
    fn string_column_codes_in_first_appearance_order() {
        let b = batch(
            "c",
            Arc::new(StringArray::from(vec![
                Some("b"),
                Some("a"),
                Some("b"),
                None,
            ])),
        );
        let (codes, dict) = extract_codes_column(&b, "c").unwrap();
        let dict = dict.expect("string column must carry a dictionary");
        assert_eq!(dict["b"], 0);
        assert_eq!(dict["a"], 1);
        assert_eq!(codes, vec![0, 1, 0, NULL_CODE]);
    }

    #[test]
    fn null_code_never_collides_with_a_real_category() {
        let labels: Vec<String> = (0..500).map(|i| format!("cat{i}")).collect();
        let array: ArrayRef = Arc::new(StringArray::from(
            labels.iter().map(|s| Some(s.as_str())).collect::<Vec<_>>(),
        ));
        let b = batch("c", array);
        let (codes, dict) = extract_codes_column(&b, "c").unwrap();
        assert!(dict.unwrap().values().all(|&c| c != NULL_CODE));
        assert!(codes.iter().all(|&c| c != NULL_CODE));
    }

    #[test]
    fn dictionary_column_round_trips_through_extract() {
        let array: ArrayRef = Arc::new(
            vec![Some("x"), Some("y"), Some("x")]
                .into_iter()
                .collect::<DictionaryArray<UInt32Type>>(),
        );
        let (codes, dict) = extract_codes_column(&batch("c", array), "c").unwrap();
        assert_eq!(codes, vec![0, 1, 0]);
        assert_eq!(dict.unwrap().len(), 2);
    }

    fn mf_dict() -> CategoryDict {
        let mut dict = CategoryDict::new();
        dict.insert("M".to_string(), 0);
        dict.insert("F".to_string(), 1);
        dict
    }

    #[test]
    fn resolves_labels_through_the_dictionary() {
        let mut keys = HashMap::new();
        keys.insert(TargetKey::Str("M".to_string()), 40.0);
        keys.insert(TargetKey::Str("F".to_string()), 60.0);

        let resolved = resolve_target_keys("g", &keys, Some(&mf_dict())).unwrap();
        assert_eq!(resolved[&0], 40.0);
        assert_eq!(resolved[&1], 60.0);
    }

    #[test]
    fn resolves_integers_for_numeric_columns() {
        let mut keys = HashMap::new();
        keys.insert(TargetKey::Int(7), 100.0);
        let resolved = resolve_target_keys("g", &keys, None).unwrap();
        assert_eq!(resolved[&7], 100.0);
    }

    #[test]
    fn unknown_label_is_rejected() {
        let mut keys = HashMap::new();
        keys.insert(TargetKey::Str("X".to_string()), 100.0);
        let err = resolve_target_keys("g", &keys, Some(&mf_dict())).unwrap_err();
        assert!(err.contains("not a category"), "{err}");
    }

    #[test]
    fn key_type_mismatches_are_rejected() {
        let mut label = HashMap::new();
        label.insert(TargetKey::Str("M".to_string()), 100.0);
        let err = resolve_target_keys("g", &label, None).unwrap_err();
        assert!(err.contains("holds numeric category codes"), "{err}");

        let mut number = HashMap::new();
        number.insert(TargetKey::Int(1), 100.0);
        let err = resolve_target_keys("g", &number, Some(&mf_dict())).unwrap_err();
        assert!(err.contains("holds text category codes"), "{err}");
    }

    #[test]
    fn resolve_targets_preserves_column_order() {
        let mut targets: IndexMap<String, HashMap<TargetKey, f64>> = IndexMap::new();
        for col in ["z", "a", "m"] {
            let mut keys = HashMap::new();
            keys.insert(TargetKey::Int(1), 100.0);
            targets.insert(col.to_string(), keys);
        }
        let resolved = resolve_targets(&targets, &ColumnDicts::new()).unwrap();
        assert_eq!(resolved.keys().collect::<Vec<_>>(), ["z", "a", "m"]);
    }

    #[test]
    fn group_keys_accept_dictionary_columns() {
        // polars Categorical / pandas category `by` columns used to fail with
        // "Unsupported group column type".
        let array: ArrayRef = Arc::new(
            vec![Some("US"), None, Some("UK")]
                .into_iter()
                .collect::<DictionaryArray<UInt32Type>>(),
        );
        let keys = extract_group_keys(&batch("country", array), "country").unwrap();
        assert_eq!(keys, vec!["US", "__null__", "UK"]);
    }

    #[test]
    fn group_keys_still_accept_plain_strings_and_numbers() {
        let s: ArrayRef = Arc::new(StringArray::from(vec![Some("US"), None]));
        assert_eq!(
            extract_group_keys(&batch("g", s), "g").unwrap(),
            vec!["US", "__null__"]
        );

        let i: ArrayRef = Arc::new(Int64Array::from(vec![101, 102]));
        assert_eq!(
            extract_group_keys(&batch("g", i), "g").unwrap(),
            vec!["101", "102"]
        );
    }

    #[test]
    fn valid_mask_handles_string_columns() {
        let array: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None, Some("b")]));
        let b = batch("c", array);
        let mask = build_valid_mask(&b, &["c".to_string()]).unwrap();
        assert_eq!(mask, vec![true, false, true]);
    }
}
