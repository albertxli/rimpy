//! PyO3 bindings for rimpy_engine.
//!
//! Three-layer architecture:
//!   Python binding (this file) → Arrow middleware (arrow_adapter.rs) → Pure engine (engine.rs)
//!
//! This file handles ONLY PyO3-specific concerns:
//!   - PyCapsule consumption/export (Arrow FFI)
//!   - PyDict → Rust type conversion
//!   - Python class definitions (PyArrowData, PyRakeResult)
//!   - #[pyfunction] signatures
//!
//! All Arrow data processing lives in arrow_adapter.rs (language-agnostic).

mod arrow_adapter;
mod engine;

use std::collections::HashMap;
use std::ffi::CString;

use arrow::array::RecordBatch;
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow::record_batch::{RecordBatchIterator, RecordBatchReader};

use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};

use indexmap::IndexMap;

use engine::RakeOpts;

// ---------------------------------------------------------------------------
// Arrow FFI: PyCapsule consumption (Python-specific)
// ---------------------------------------------------------------------------

/// Consume a Python object implementing `__arrow_c_stream__` into a RecordBatch.
fn arrow_from_pycapsule(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    let capsule: Bound<'_, PyCapsule> = data
        .call_method1("__arrow_c_stream__", (py.None(),))?
        .cast_into()?;

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

// ---------------------------------------------------------------------------
// PyArrowData: wraps Arrow data for return to Python via PyCapsule
// ---------------------------------------------------------------------------

/// Wraps an Arrow RecordBatch and exports it via the Arrow PyCapsule Interface.
#[pyclass(name = "_ArrowData")]
struct PyArrowData {
    batch: RecordBatch,
}

#[pymethods]
impl PyArrowData {
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
// PyRakeResult: diagnostics only (weights live in the Arrow RecordBatch)
// ---------------------------------------------------------------------------

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
    #[getter]
    fn weight_ratio(&self) -> f64 {
        if self.weight_min > 0.0 {
            self.weight_max / self.weight_min
        } else {
            f64::INFINITY
        }
    }

    fn summary(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("n_valid".into(), self.n_valid as f64);
        m.insert("iterations".into(), self.iterations as f64);
        m.insert(
            "converged".into(),
            if self.converged { 1.0 } else { 0.0 },
        );
        m.insert(
            "efficiency".into(),
            (self.efficiency * 100.0).round() / 100.0,
        );
        m.insert(
            "weight_min".into(),
            (self.weight_min * 10000.0).round() / 10000.0,
        );
        m.insert(
            "weight_max".into(),
            (self.weight_max * 10000.0).round() / 10000.0,
        );
        m.insert(
            "weight_ratio".into(),
            (self.weight_ratio() * 100.0).round() / 100.0,
        );
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

// ---------------------------------------------------------------------------
// rim_rake: single-group raking (thin wrapper)
// ---------------------------------------------------------------------------

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

    let (result_batch, n_valid, result) = arrow_adapter::rake_batch(
        &batch,
        &target_columns,
        &rust_targets,
        weight_column,
        &opts,
        drop_nulls,
        total,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    Ok((
        PyArrowData {
            batch: result_batch,
        },
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
// rim_rake_grouped: same targets for all groups (thin wrapper)
// ---------------------------------------------------------------------------

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
    let batch = arrow_from_pycapsule(py, data)?;
    let rust_targets = extract_targets(targets)?;
    let opts = RakeOpts {
        max_iterations,
        convergence_threshold,
        min_cap,
        max_cap,
        cap_correction,
    };

    let (result_batch, group_results) = arrow_adapter::rake_batch_grouped(
        &batch,
        &target_columns,
        &rust_targets,
        &group_columns,
        weight_column,
        &opts,
        drop_nulls,
        total,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    // Convert Vec<GroupRakeResult> → PyDict of PyRakeResult
    let output_dict = PyDict::new(py);
    for gr in &group_results {
        let py_result = PyRakeResult {
            n_valid: gr.n_valid,
            iterations: gr.result.iterations,
            converged: gr.result.converged,
            efficiency: gr.result.efficiency,
            weight_min: gr.result.weight_min,
            weight_max: gr.result.weight_max,
        };
        output_dict.set_item(gr.group_key.as_str(), Py::new(py, py_result)?)?;
    }

    Ok((
        PyArrowData {
            batch: result_batch,
        },
        output_dict.unbind(),
    ))
}

// ---------------------------------------------------------------------------
// rim_rake_by_scheme: different targets per group (thin wrapper)
// ---------------------------------------------------------------------------

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
    let batch = arrow_from_pycapsule(py, data)?;
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
        let group_key: String = key
            .extract::<String>()
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
                let key: String = k
                    .extract::<String>()
                    .or_else(|_| k.extract::<i64>().map(|v| v.to_string()))
                    .or_else(|_| k.extract::<f64>().map(|v| v.to_string()))?;
                let val: f64 = v.extract()?;
                m.insert(key, val);
            }
            Some(m)
        }
        None => None,
    };

    let (result_batch, group_results) = arrow_adapter::rake_batch_by_scheme(
        &batch,
        group_column,
        &parsed_schemes,
        default_targets.as_ref(),
        weight_column,
        &opts,
        drop_nulls,
        parsed_group_totals.as_ref(),
        total,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    // Convert Vec<GroupRakeResult> → PyDict of PyRakeResult
    let output_dict = PyDict::new(py);
    for gr in &group_results {
        let py_result = PyRakeResult {
            n_valid: gr.n_valid,
            iterations: gr.result.iterations,
            converged: gr.result.converged,
            efficiency: gr.result.efficiency,
            weight_min: gr.result.weight_min,
            weight_max: gr.result.weight_max,
        };
        output_dict.set_item(gr.group_key.as_str(), Py::new(py, py_result)?)?;
    }

    Ok((
        PyArrowData {
            batch: result_batch,
        },
        output_dict.unbind(),
    ))
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

#[pymodule]
fn _rimpy_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRakeResult>()?;
    m.add_class::<PyArrowData>()?;
    m.add_function(wrap_pyfunction!(rim_rake, m)?)?;
    m.add_function(wrap_pyfunction!(rim_rake_grouped, m)?)?;
    m.add_function(wrap_pyfunction!(rim_rake_by_scheme, m)?)?;
    Ok(())
}
