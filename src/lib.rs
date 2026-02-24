//! PyO3 bindings for rimpy_engine.
//!
//! Unified Arrow PyCapsule architecture:
//!   narwhals DataFrame → __arrow_c_stream__ → Rust → RecordBatch + weight column → __arrow_c_stream__ → narwhals
//!
//! No NumPy. No Python lists in the data path. No backend-specific code.

mod engine;

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, Float64Array, RecordBatch, RecordBatchReader, StringArray,
    StringViewArray,
};
use arrow::datatypes::{DataType, Field, Float64Type, Int64Type, Schema};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow::record_batch::RecordBatchIterator;

use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};

use indexmap::IndexMap;

use engine::RakeOpts;

// ---------------------------------------------------------------------------
// Arrow FFI helpers
// ---------------------------------------------------------------------------

/// Consume a Python object implementing `__arrow_c_stream__` into a RecordBatch.
fn arrow_from_pycapsule(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    let capsule: Bound<'_, PyCapsule> = data
        .call_method1("__arrow_c_stream__", (py.None(),))?
        .cast_into()?;

    // SAFETY: The PyCapsule wraps an FFI_ArrowArrayStream allocated by the producer.
    // We consume it by reading the struct and nulling the release callback on the
    // original, so the PyCapsule destructor becomes a no-op (prevents double-free).
    let capsule_name = CString::new("arrow_array_stream").unwrap();
    let ptr = capsule.pointer_checked(Some(capsule_name.as_c_str()))?;
    let stream_ptr = ptr.as_ptr() as *mut FFI_ArrowArrayStream;
    let stream_owned = unsafe { std::ptr::read(stream_ptr) };
    unsafe {
        (*stream_ptr).release = None;
    }

    let reader = ArrowArrayStreamReader::try_new(stream_owned)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Arrow stream error: {e}")))?;

    let schema = reader.schema();
    let mut batches: Vec<RecordBatch> = Vec::new();
    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Arrow batch error: {e}")))?;
        batches.push(batch);
    }

    if batches.is_empty() {
        return Ok(RecordBatch::new_empty(schema));
    }

    if batches.len() == 1 {
        return Ok(batches.into_iter().next().unwrap());
    }

    arrow::compute::concat_batches(&schema, &batches)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Arrow concat error: {e}")))
}

/// Extract a column from a RecordBatch as Vec<i64>.
/// Handles multiple integer/float types.
fn extract_i64_column(batch: &RecordBatch, col_name: &str) -> PyResult<Vec<i64>> {
    let col_idx = batch
        .schema()
        .index_of(col_name)
        .map_err(|_| pyo3::exceptions::PyKeyError::new_err(format!("Column '{col_name}' not found in Arrow data")))?;

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
        dt => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Column '{col_name}' has unsupported type {dt:?}, expected integer or float"
        ))),
    }
}

/// Build a null/valid mask from Arrow null bitmaps for the given columns.
/// Returns a boolean vec: true = valid (no nulls in any target column).
fn build_valid_mask(batch: &RecordBatch, target_columns: &[String]) -> PyResult<Vec<bool>> {
    let n_rows = batch.num_rows();
    let mut valid = vec![true; n_rows];

    for col_name in target_columns {
        let col_idx = batch
            .schema()
            .index_of(col_name)
            .map_err(|_| pyo3::exceptions::PyKeyError::new_err(format!("Column '{col_name}' not found")))?;
        let array = batch.column(col_idx);
        if let Some(nulls) = array.nulls() {
            for i in 0..n_rows {
                if !nulls.is_valid(i) {
                    valid[i] = false;
                }
            }
        }
    }

    Ok(valid)
}

/// Append a Float64 weight column to a RecordBatch.
/// Existing columns are Arc-shared (zero-copy). Only the new column is allocated.
fn append_weight_column(batch: &RecordBatch, weights: Vec<f64>, column_name: &str) -> PyResult<RecordBatch> {
    let weight_array: ArrayRef = Arc::new(Float64Array::from(weights));
    let mut fields: Vec<Arc<Field>> = batch.schema().fields().iter().cloned().collect();
    fields.push(Arc::new(Field::new(column_name, DataType::Float64, false)));
    let new_schema = Arc::new(Schema::new(fields));
    let mut all_columns: Vec<ArrayRef> = batch.columns().to_vec();
    all_columns.push(weight_array);
    RecordBatch::try_new(new_schema, all_columns)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to append weight column: {e}")))
}

// ---------------------------------------------------------------------------
// PyArrowData: wraps Arrow data for return to Python via PyCapsule
// ---------------------------------------------------------------------------

/// Wraps an Arrow RecordBatch and exports it via the Arrow PyCapsule Interface.
/// Consumers like narwhals/polars call `__arrow_c_stream__` to get zero-copy data.
#[pyclass(name = "_ArrowData")]
struct PyArrowData {
    batch: RecordBatch,
}

#[pymethods]
impl PyArrowData {
    /// Arrow PyCapsule Interface: export as an ArrowArrayStream capsule.
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let _ = requested_schema;
        let schema = self.batch.schema();
        let batches = vec![self.batch.clone()];
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
        let reader: Box<dyn arrow::record_batch::RecordBatchReader + Send> = Box::new(reader);
        let ffi_stream = FFI_ArrowArrayStream::new(reader);

        let capsule_name = CString::new("arrow_array_stream").unwrap();
        PyCapsule::new(py, ffi_stream, Some(capsule_name))
    }

    fn __repr__(&self) -> String {
        format!(
            "_ArrowData(rows={}, cols={})",
            self.batch.num_rows(),
            self.batch.num_columns()
        )
    }
}

// ---------------------------------------------------------------------------
// Python-facing RakeResult (diagnostics only — no weights)
// ---------------------------------------------------------------------------

/// Result of a raking operation (returned to Python).
/// Weights live in the Arrow RecordBatch, not here.
#[pyclass(name = "RakeResult")]
struct PyRakeResult {
    #[pyo3(get)]
    n_valid: usize,

    #[pyo3(get)]
    iterations: usize,

    #[pyo3(get)]
    converged: bool,

    #[pyo3(get)]
    efficiency: f64,

    #[pyo3(get)]
    weight_min: f64,

    #[pyo3(get)]
    weight_max: f64,
}

#[pymethods]
impl PyRakeResult {
    /// Ratio of max to min weight.
    #[getter]
    fn weight_ratio(&self) -> f64 {
        if self.weight_min > 0.0 {
            self.weight_max / self.weight_min
        } else {
            f64::INFINITY
        }
    }

    /// Summary statistics as a Python dict.
    fn summary(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("n_valid".into(), self.n_valid as f64);
        m.insert("iterations".into(), self.iterations as f64);
        m.insert("converged".into(), if self.converged { 1.0 } else { 0.0 });
        m.insert("efficiency".into(), (self.efficiency * 100.0).round() / 100.0);
        m.insert("weight_min".into(), (self.weight_min * 10000.0).round() / 10000.0);
        m.insert("weight_max".into(), (self.weight_max * 10000.0).round() / 10000.0);
        m.insert("weight_ratio".into(), (self.weight_ratio() * 100.0).round() / 100.0);
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "RakeResult(n_valid={}, iterations={}, converged={}, efficiency={:.2}%, weights=[{:.4}..{:.4}])",
            self.n_valid, self.iterations, self.converged, self.efficiency, self.weight_min, self.weight_max,
        )
    }
}

// ---------------------------------------------------------------------------
// Helper: convert Python target dicts to Rust types
// ---------------------------------------------------------------------------

/// Convert Python `dict[str, dict[int, float]]` to Rust IndexMap.
fn extract_targets(targets: &Bound<'_, PyDict>) -> PyResult<IndexMap<String, HashMap<i64, f64>>> {
    let mut result = IndexMap::new();

    for (col_key, props_obj) in targets.iter() {
        let col: String = col_key.extract()?;
        let props: Bound<'_, PyDict> = props_obj.extract()?;
        let mut code_map = HashMap::new();

        for (code_key, val) in props.iter() {
            let code: i64 = code_key
                .extract::<i64>()
                .or_else(|_| code_key.extract::<f64>().map(|f| f as i64))?;
            let target_val: f64 = val.extract()?;
            code_map.insert(code, target_val);
        }

        result.insert(col, code_map);
    }

    Ok(result)
}

/// Internal: run engine on a batch, handling null filtering + weight scatter + total scaling.
/// Returns (full_weights, n_valid, engine::RakeResult).
fn run_rake_on_batch(
    batch: &RecordBatch,
    target_columns: &[String],
    targets: &IndexMap<String, HashMap<i64, f64>>,
    opts: &RakeOpts,
    drop_nulls: bool,
    total: Option<f64>,
) -> PyResult<(Vec<f64>, usize, engine::RakeResult)> {
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
        let result = engine::RakeResult {
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
        // No nulls — reuse original data
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

    let result = engine::rim_iterate(&col_refs, targets, opts)
        .map_err(|e| pyo3::exceptions::PyKeyError::new_err(e))?;

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
    if let Some(target_total) = total {
        if target_total > 0.0 {
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
    }

    Ok((full_weights, n_valid, result))
}

// ---------------------------------------------------------------------------
// rim_rake: single-group raking (Arrow in → Arrow out)
// ---------------------------------------------------------------------------

/// Core RIM raking — single group.
///
/// Accepts any object with `__arrow_c_stream__` (narwhals DataFrame, polars, pyarrow).
/// Returns (Arrow RecordBatch with weight column appended, RakeResult diagnostics).
#[pyfunction]
#[pyo3(signature = (
    data,
    target_columns,
    targets,
    weight_column = "weight",
    max_iterations = 1000,
    convergence_threshold = 0.01,
    min_cap = None,
    max_cap = None,
    drop_nulls = true,
    total = None,
    cap_correction = true,
))]
fn rim_rake(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    target_columns: Vec<String>,
    targets: &Bound<'_, PyDict>,
    weight_column: &str,
    max_iterations: usize,
    convergence_threshold: f64,
    min_cap: Option<f64>,
    max_cap: Option<f64>,
    drop_nulls: bool,
    total: Option<f64>,
    cap_correction: bool,
) -> PyResult<(PyArrowData, PyRakeResult)> {
    let batch = arrow_from_pycapsule(py, data)?;
    let rust_targets = extract_targets(targets)?;
    let opts = RakeOpts {
        max_iterations,
        convergence_threshold,
        min_cap,
        max_cap,
        cap_correction,
    };

    let (full_weights, n_valid, result) =
        run_rake_on_batch(&batch, &target_columns, &rust_targets, &opts, drop_nulls, total)?;

    let result_batch = append_weight_column(&batch, full_weights, weight_column)?;

    Ok((
        PyArrowData { batch: result_batch },
        PyRakeResult {
            n_valid,
            iterations: result.iterations,
            converged: result.converged,
            efficiency: result.efficiency,
            weight_min: result.weight_min,
            weight_max: result.weight_max,
        },
    ))
}

// ---------------------------------------------------------------------------
// rim_rake_grouped: same targets for all groups (Arrow in → Arrow out)
// ---------------------------------------------------------------------------

/// Extract a group column as String keys (handles String, Int64, Float64, etc.)
fn extract_group_keys(batch: &RecordBatch, group_col: &str) -> PyResult<Vec<String>> {
    let col_idx = batch
        .schema()
        .index_of(group_col)
        .map_err(|_| pyo3::exceptions::PyKeyError::new_err(format!("Group column '{group_col}' not found")))?;
    let array = batch.column(col_idx);

    match array.data_type() {
        DataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
            Ok((0..arr.len()).map(|i| {
                if arr.is_null(i) { "__null__".to_string() } else { arr.value(i).to_string() }
            }).collect())
        }
        DataType::LargeUtf8 => {
            let arr = array.as_any().downcast_ref::<arrow::array::LargeStringArray>().unwrap();
            Ok((0..arr.len()).map(|i| {
                if arr.is_null(i) { "__null__".to_string() } else { arr.value(i).to_string() }
            }).collect())
        }
        DataType::Utf8View => {
            let arr = array.as_any().downcast_ref::<StringViewArray>().unwrap();
            Ok((0..arr.len()).map(|i| {
                if arr.is_null(i) { "__null__".to_string() } else { arr.value(i).to_string() }
            }).collect())
        }
        DataType::Int64 => {
            let arr = array.as_primitive::<Int64Type>();
            Ok((0..arr.len()).map(|i| {
                if arr.is_null(i) { "__null__".to_string() } else { arr.value(i).to_string() }
            }).collect())
        }
        DataType::Int32 => {
            let arr = array.as_primitive::<arrow::datatypes::Int32Type>();
            Ok((0..arr.len()).map(|i| {
                if arr.is_null(i) { "__null__".to_string() } else { arr.value(i).to_string() }
            }).collect())
        }
        DataType::Float64 => {
            let arr = array.as_primitive::<Float64Type>();
            Ok((0..arr.len()).map(|i| {
                if arr.is_null(i) { "__null__".to_string() } else { arr.value(i).to_string() }
            }).collect())
        }
        dt => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Unsupported group column type: {dt:?}"
        ))),
    }
}

/// Grouped raking — same targets for all groups.
///
/// Receives the full DataFrame + group column(s). Partitions internally,
/// rakes each group in parallel via rayon, assembles a single weight column.
#[pyfunction]
#[pyo3(signature = (
    data,
    target_columns,
    targets,
    group_columns,
    weight_column = "weight",
    max_iterations = 1000,
    convergence_threshold = 0.01,
    min_cap = None,
    max_cap = None,
    drop_nulls = true,
    total = None,
    cap_correction = true,
))]
fn rim_rake_grouped(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    target_columns: Vec<String>,
    targets: &Bound<'_, PyDict>,
    group_columns: Vec<String>,
    weight_column: &str,
    max_iterations: usize,
    convergence_threshold: f64,
    min_cap: Option<f64>,
    max_cap: Option<f64>,
    drop_nulls: bool,
    total: Option<f64>,
    cap_correction: bool,
) -> PyResult<(PyArrowData, Py<PyDict>)> {
    use rayon::prelude::*;

    let batch = arrow_from_pycapsule(py, data)?;
    let n_rows = batch.num_rows();
    let rust_targets = extract_targets(targets)?;
    let opts = RakeOpts {
        max_iterations,
        convergence_threshold,
        min_cap,
        max_cap,
        cap_correction,
    };

    // Build composite group keys (concatenate multiple columns)
    let group_keys: Vec<String> = if group_columns.len() == 1 {
        extract_group_keys(&batch, &group_columns[0])?
    } else {
        let per_col: Vec<Vec<String>> = group_columns
            .iter()
            .map(|col| extract_group_keys(&batch, col))
            .collect::<PyResult<_>>()?;
        (0..n_rows)
            .map(|i| {
                per_col.iter().map(|col| col[i].as_str()).collect::<Vec<_>>().join("||")
            })
            .collect()
    };

    // Partition rows by group
    let mut group_row_indices: IndexMap<String, Vec<usize>> = IndexMap::new();
    for (i, key) in group_keys.iter().enumerate() {
        group_row_indices.entry(key.clone()).or_default().push(i);
    }

    // Extract all target columns once
    let mut all_columns: HashMap<String, Vec<i64>> = HashMap::new();
    for col_name in &target_columns {
        all_columns.insert(col_name.clone(), extract_i64_column(&batch, col_name)?);
    }

    // Build valid mask
    let valid_mask = if drop_nulls {
        build_valid_mask(&batch, &target_columns)?
    } else {
        vec![true; n_rows]
    };

    // Prepare group data for parallel processing
    let group_entries: Vec<(String, Vec<usize>)> = group_row_indices.into_iter().collect();

    // Process groups in parallel
    let group_results: Vec<(String, Vec<usize>, engine::RakeResult)> = group_entries
        .into_par_iter()
        .map(|(key, row_indices)| {
            // Filter to valid rows within this group
            let valid_indices: Vec<usize> = row_indices
                .iter()
                .filter(|&&i| valid_mask[i])
                .copied()
                .collect();

            if valid_indices.is_empty() {
                let result = engine::RakeResult {
                    weights: vec![],
                    iterations: 0,
                    converged: true,
                    efficiency: 100.0,
                    weight_min: 1.0,
                    weight_max: 1.0,
                };
                return Ok((key, row_indices, result));
            }

            // Extract group's column data
            let group_columns: HashMap<String, Vec<i64>> = all_columns
                .iter()
                .map(|(name, data)| {
                    let filtered: Vec<i64> = valid_indices.iter().map(|&i| data[i]).collect();
                    (name.clone(), filtered)
                })
                .collect();

            let col_refs: HashMap<String, &[i64]> = group_columns
                .iter()
                .map(|(k, v)| (k.clone(), v.as_slice()))
                .collect();

            let result = engine::rim_iterate(&col_refs, &rust_targets, &opts)
                .map_err(|e| pyo3::exceptions::PyKeyError::new_err(e))?;

            Ok((key, row_indices, result))
        })
        .collect::<PyResult<_>>()?;

    // Assemble full weight array
    let mut full_weights = vec![1.0_f64; n_rows];
    let output_dict = PyDict::new(py);

    for (key, row_indices, result) in &group_results {
        // Scatter weights for valid rows in this group
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

        let n_valid = valid_indices.len();
        let py_result = PyRakeResult {
            n_valid,
            iterations: result.iterations,
            converged: result.converged,
            efficiency: result.efficiency,
            weight_min: result.weight_min,
            weight_max: result.weight_max,
        };
        output_dict.set_item(key.as_str(), Py::new(py, py_result)?)?;
    }

    // Scale to total
    if let Some(target_total) = total {
        if target_total > 0.0 {
            let current_sum: f64 = full_weights.iter().sum();
            if current_sum > 0.0 {
                let factor = target_total / current_sum;
                for w in full_weights.iter_mut() {
                    *w *= factor;
                }
            }
        }
    }

    let result_batch = append_weight_column(&batch, full_weights, weight_column)?;

    Ok((
        PyArrowData { batch: result_batch },
        output_dict.unbind(),
    ))
}

// ---------------------------------------------------------------------------
// rim_rake_by_scheme: different targets per group (Arrow in → Arrow out)
// ---------------------------------------------------------------------------

/// Per-group scheme raking — different targets for each group.
///
/// schemes: {group_value: {col: {code: pct}}}
/// Handles group_totals (nested weighting) and global total scaling.
#[pyfunction]
#[pyo3(signature = (
    data,
    group_column,
    schemes,
    default_scheme = None,
    weight_column = "weight",
    max_iterations = 1000,
    convergence_threshold = 0.01,
    min_cap = None,
    max_cap = None,
    drop_nulls = true,
    group_totals = None,
    total = None,
    cap_correction = true,
))]
fn rim_rake_by_scheme(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    group_column: &str,
    schemes: &Bound<'_, PyDict>,
    default_scheme: Option<&Bound<'_, PyDict>>,
    weight_column: &str,
    max_iterations: usize,
    convergence_threshold: f64,
    min_cap: Option<f64>,
    max_cap: Option<f64>,
    drop_nulls: bool,
    group_totals: Option<&Bound<'_, PyDict>>,
    total: Option<f64>,
    cap_correction: bool,
) -> PyResult<(PyArrowData, Py<PyDict>)> {
    use rayon::prelude::*;

    let batch = arrow_from_pycapsule(py, data)?;
    let n_rows = batch.num_rows();
    let opts = RakeOpts {
        max_iterations,
        convergence_threshold,
        min_cap,
        max_cap,
        cap_correction,
    };

    // Parse default scheme
    let default_targets = match default_scheme {
        Some(ds) => Some(extract_targets(ds)?),
        None => None,
    };

    // Parse per-group schemes
    let mut parsed_schemes: HashMap<String, IndexMap<String, HashMap<i64, f64>>> = HashMap::new();
    for (key, value) in schemes.iter() {
        let group_key: String = key.extract::<String>()
            .or_else(|_| key.extract::<i64>().map(|v| v.to_string()))
            .or_else(|_| key.extract::<f64>().map(|v| v.to_string()))?;
        let scheme_dict: Bound<'_, PyDict> = value.extract()?;
        parsed_schemes.insert(group_key, extract_targets(&scheme_dict)?);
    }

    // Parse group_totals
    let parsed_group_totals: Option<HashMap<String, f64>> = match group_totals {
        Some(gt) => {
            let mut m = HashMap::new();
            for (k, v) in gt.iter() {
                let key: String = k.extract::<String>()
                    .or_else(|_| k.extract::<i64>().map(|v| v.to_string()))
                    .or_else(|_| k.extract::<f64>().map(|v| v.to_string()))?;
                let val: f64 = v.extract()?;
                m.insert(key, val);
            }
            Some(m)
        }
        None => None,
    };

    // Extract group column keys
    let group_keys = extract_group_keys(&batch, group_column)?;

    // Partition rows by group
    let mut group_row_indices: IndexMap<String, Vec<usize>> = IndexMap::new();
    for (i, key) in group_keys.iter().enumerate() {
        group_row_indices.entry(key.clone()).or_default().push(i);
    }

    // Collect all target columns from all schemes
    let mut all_target_cols: std::collections::HashSet<String> = std::collections::HashSet::new();
    for scheme_targets in parsed_schemes.values() {
        for col in scheme_targets.keys() {
            all_target_cols.insert(col.clone());
        }
    }
    if let Some(ref dt) = default_targets {
        for col in dt.keys() {
            all_target_cols.insert(col.clone());
        }
    }

    // Extract all needed columns once
    let mut all_columns: HashMap<String, Vec<i64>> = HashMap::new();
    for col_name in &all_target_cols {
        if batch.schema().index_of(col_name).is_ok() {
            all_columns.insert(col_name.clone(), extract_i64_column(&batch, col_name)?);
        }
    }

    // Build valid mask (across ALL target columns that exist)
    let existing_target_cols: Vec<String> = all_target_cols
        .iter()
        .filter(|c| all_columns.contains_key(c.as_str()))
        .cloned()
        .collect();
    let valid_mask = if drop_nulls {
        build_valid_mask(&batch, &existing_target_cols)?
    } else {
        vec![true; n_rows]
    };

    // Prepare group data
    let group_entries: Vec<(String, Vec<usize>)> = group_row_indices.into_iter().collect();

    // Process each group
    let group_results: Vec<(String, Vec<usize>, engine::RakeResult)> = group_entries
        .into_par_iter()
        .map(|(key, row_indices)| {
            // Look up targets for this group
            let group_targets = match parsed_schemes.get(&key) {
                Some(t) => t.clone(),
                None => match &default_targets {
                    Some(dt) => dt.clone(),
                    None => {
                        // No scheme — weight = 1.0
                        let result = engine::RakeResult {
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

            // Filter to valid rows within this group (only for this scheme's columns)
            let valid_indices: Vec<usize> = row_indices
                .iter()
                .filter(|&&i| {
                    target_columns.iter().all(|_col| {
                        if !drop_nulls { return true; }
                        valid_mask[i]
                    })
                })
                .copied()
                .collect();

            if valid_indices.is_empty() {
                let result = engine::RakeResult {
                    weights: vec![],
                    iterations: 0,
                    converged: true,
                    efficiency: 100.0,
                    weight_min: 1.0,
                    weight_max: 1.0,
                };
                return Ok((key, row_indices, result));
            }

            // Extract group's column data
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

            let result = engine::rim_iterate(&col_refs, &group_targets, &opts)
                .map_err(|e| pyo3::exceptions::PyKeyError::new_err(e))?;

            Ok((key, row_indices, result))
        })
        .collect::<PyResult<_>>()?;

    // Assemble full weight array
    let mut full_weights = vec![1.0_f64; n_rows];
    let output_dict = PyDict::new(py);

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

        let n_valid = valid_indices.len();
        let py_result = PyRakeResult {
            n_valid,
            iterations: result.iterations,
            converged: result.converged,
            efficiency: result.efficiency,
            weight_min: result.weight_min,
            weight_max: result.weight_max,
        };
        output_dict.set_item(key.as_str(), Py::new(py, py_result)?)?;
    }

    // Apply group_totals correction
    if let Some(ref gt) = parsed_group_totals {
        // Normalize totals
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
    if let Some(target_total) = total {
        if target_total > 0.0 {
            let current_sum: f64 = full_weights.iter().sum();
            if current_sum > 0.0 {
                let factor = target_total / current_sum;
                for w in full_weights.iter_mut() {
                    *w *= factor;
                }
            }
        }
    }

    let result_batch = append_weight_column(&batch, full_weights, weight_column)?;

    Ok((
        PyArrowData { batch: result_batch },
        output_dict.unbind(),
    ))
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

/// rimpy_engine - Fast RIM/raking engine written in Rust.
#[pymodule]
fn _rimpy_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRakeResult>()?;
    m.add_class::<PyArrowData>()?;
    m.add_function(wrap_pyfunction!(rim_rake, m)?)?;
    m.add_function(wrap_pyfunction!(rim_rake_grouped, m)?)?;
    m.add_function(wrap_pyfunction!(rim_rake_by_scheme, m)?)?;
    Ok(())
}
