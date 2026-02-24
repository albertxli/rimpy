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

use arrow::array::{
    Array, ArrayRef, AsArray, Float64Array, RecordBatch, StringArray, StringViewArray,
};
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
/// Handles Utf8, LargeUtf8, Utf8View, Int64, Int32, Float64.
/// Null values become `"__null__"`.
pub fn extract_group_keys(batch: &RecordBatch, group_col: &str) -> Result<Vec<String>, String> {
    let col_idx = batch
        .schema()
        .index_of(group_col)
        .map_err(|_| format!("Group column '{group_col}' not found"))?;
    let array = batch.column(col_idx);

    match array.data_type() {
        DataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
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
        DataType::LargeUtf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow::array::LargeStringArray>()
                .unwrap();
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
        DataType::Utf8View => {
            let arr = array.as_any().downcast_ref::<StringViewArray>().unwrap();
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
    targets: &IndexMap<String, HashMap<i64, f64>>,
    opts: &RakeOpts,
    drop_nulls: bool,
    total: Option<f64>,
) -> Result<(Vec<f64>, usize, RakeResult), String> {
    let n_rows = batch.num_rows();

    // Extract all target columns
    let mut columns: HashMap<String, Vec<i64>> = HashMap::new();
    for col_name in target_columns {
        columns.insert(col_name.clone(), extract_i64_column(batch, col_name)?);
    }

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
    targets: &IndexMap<String, HashMap<i64, f64>>,
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
    targets: &IndexMap<String, HashMap<i64, f64>>,
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

    // Extract all target columns once
    let mut all_columns: HashMap<String, Vec<i64>> = HashMap::new();
    for col_name in target_columns {
        all_columns.insert(col_name.clone(), extract_i64_column(batch, col_name)?);
    }

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
    schemes: &HashMap<String, IndexMap<String, HashMap<i64, f64>>>,
    default_scheme: Option<&IndexMap<String, HashMap<i64, f64>>>,
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

    // Extract all needed columns once
    let mut all_columns: HashMap<String, Vec<i64>> = HashMap::new();
    for col_name in &all_target_cols {
        if batch.schema().index_of(col_name).is_ok() {
            all_columns.insert(col_name.clone(), extract_i64_column(batch, col_name)?);
        }
    }

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
