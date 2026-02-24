//! PyO3 bindings for rimpy_engine.
//!
//! Exposes `rim_iterate` and `RakeResult` to Python, accepting NumPy arrays
//! directly with zero-copy reads.

mod engine;

use std::collections::HashMap;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use indexmap::IndexMap;

use engine::{GroupData, RakeOpts};

// ---------------------------------------------------------------------------
// Python-facing RakeResult
// ---------------------------------------------------------------------------

/// Result of a raking operation (returned to Python).
#[pyclass(name = "RakeResult")]
struct PyRakeResult {
    #[pyo3(get, set)]
    weights: Py<PyArray1<f64>>,

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
    #[new]
    #[pyo3(signature = (weights, iterations, converged, efficiency, weight_min, weight_max))]
    fn new(
        weights: Py<PyArray1<f64>>,
        iterations: usize,
        converged: bool,
        efficiency: f64,
        weight_min: f64,
        weight_max: f64,
    ) -> Self {
        PyRakeResult {
            weights,
            iterations,
            converged,
            efficiency,
            weight_min,
            weight_max,
        }
    }

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
            "RakeResult(iterations={}, converged={}, efficiency={:.2}%, weights=[{:.4}..{:.4}])",
            self.iterations, self.converged, self.efficiency, self.weight_min, self.weight_max,
        )
    }
}

// ---------------------------------------------------------------------------
// Helper: convert Python target dicts to Rust types
// ---------------------------------------------------------------------------

/// Convert Python `dict[str, dict[int, float]]` to Rust HashMap.
fn extract_targets(targets: &Bound<'_, PyDict>) -> PyResult<IndexMap<String, HashMap<i64, f64>>> {
    let mut result = IndexMap::new();

    for (col_key, props_obj) in targets.iter() {
        let col: String = col_key.extract()?;
        let props: Bound<'_, PyDict> = props_obj.extract()?;
        let mut code_map = HashMap::new();

        for (code_key, val) in props.iter() {
            // Try int first, then float->i64 for codes like 1.0
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

/// Extract a numpy array as Vec<i64>, handling both i64 and f64 input.
fn extract_i64_array(_py: Python<'_>, val: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    // Try i64 array first
    if let Ok(array) = val.extract::<PyReadonlyArray1<i64>>() {
        return Ok(array.as_slice()?.to_vec());
    }
    // Fall back to f64 array, cast to i64
    let float_arr: PyReadonlyArray1<f64> = val.extract()?;
    Ok(float_arr.as_slice()?.iter().map(|&f| f as i64).collect())
}

// ---------------------------------------------------------------------------
// Main Python-facing function: rim_iterate
// ---------------------------------------------------------------------------

/// Core RIM iteration - drop-in replacement for _engine.rim_iterate.
#[pyfunction]
#[pyo3(signature = (
    column_data,
    targets,
    max_iterations = 1000,
    convergence_threshold = 0.01,
    min_cap = None,
    max_cap = None,
    cap_correction = true,
))]
fn rim_iterate(
    py: Python<'_>,
    column_data: &Bound<'_, PyDict>,
    targets: &Bound<'_, PyDict>,
    max_iterations: usize,
    convergence_threshold: f64,
    min_cap: Option<f64>,
    max_cap: Option<f64>,
    cap_correction: bool,
) -> PyResult<PyRakeResult> {
    // Extract column data
    let mut columns: HashMap<String, Vec<i64>> = HashMap::new();
    for (key, val) in column_data.iter() {
        let col_name: String = key.extract()?;
        columns.insert(col_name, extract_i64_array(py, &val)?);
    }

    // Extract targets
    let rust_targets = extract_targets(targets)?;

    // Build options
    let opts = RakeOpts {
        max_iterations,
        convergence_threshold,
        min_cap,
        max_cap,
        cap_correction,
    };

    // Convert to slice references for the engine
    let col_refs: HashMap<String, &[i64]> = columns
        .iter()
        .map(|(k, v)| (k.clone(), v.as_slice()))
        .collect();

    // Run the engine
    let result = engine::rim_iterate(&col_refs, &rust_targets, &opts)
        .map_err(|e| pyo3::exceptions::PyKeyError::new_err(e))?;

    // Convert weights to NumPy array
    let weights_array = PyArray1::from_vec(py, result.weights).unbind();

    Ok(PyRakeResult {
        weights: weights_array,
        iterations: result.iterations,
        converged: result.converged,
        efficiency: result.efficiency,
        weight_min: result.weight_min,
        weight_max: result.weight_max,
    })
}

// ---------------------------------------------------------------------------
// Grouped raking (parallel)
// ---------------------------------------------------------------------------

/// Rake multiple groups in parallel.
#[pyfunction]
#[pyo3(signature = (
    groups_data,
    groups_targets,
    max_iterations = 1000,
    convergence_threshold = 0.01,
    min_cap = None,
    max_cap = None,
    cap_correction = true,
))]
fn rim_iterate_grouped(
    py: Python<'_>,
    groups_data: &Bound<'_, PyDict>,
    groups_targets: &Bound<'_, PyDict>,
    max_iterations: usize,
    convergence_threshold: f64,
    min_cap: Option<f64>,
    max_cap: Option<f64>,
    cap_correction: bool,
) -> PyResult<Py<PyDict>> {
    let opts = RakeOpts {
        max_iterations,
        convergence_threshold,
        min_cap,
        max_cap,
        cap_correction,
    };

    // Extract all group data upfront
    let mut groups: Vec<(String, GroupData)> = Vec::new();

    for (group_key, group_cols_obj) in groups_data.iter() {
        let key: String = group_key.extract()?;
        let group_cols: Bound<'_, PyDict> = group_cols_obj.extract()?;

        // Extract column data for this group
        let mut column_data = HashMap::new();
        for (col_key, col_val) in group_cols.iter() {
            let col_name: String = col_key.extract()?;
            column_data.insert(col_name, extract_i64_array(py, &col_val)?);
        }

        // Extract targets for this group
        let group_targets_bound: Bound<'_, PyDict> = groups_targets
            .get_item(&key)?
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "No targets for group '{key}'"
                ))
            })?
            .extract()?;

        let targets = extract_targets(&group_targets_bound)?;

        groups.push((
            key,
            GroupData {
                column_data,
                targets,
            },
        ));
    }

    // Compute in parallel
    let results = engine::rim_iterate_grouped(groups, &opts)
        .map_err(|e| pyo3::exceptions::PyKeyError::new_err(e))?;

    // Convert back to Python dict
    let output = PyDict::new(py);
    for (key, result) in results {
        let weights_array = PyArray1::from_vec(py, result.weights).unbind();
        let py_result = PyRakeResult {
            weights: weights_array,
            iterations: result.iterations,
            converged: result.converged,
            efficiency: result.efficiency,
            weight_min: result.weight_min,
            weight_max: result.weight_max,
        };
        output.set_item(key, Py::new(py, py_result)?)?;
    }

    Ok(output.unbind())
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

/// rimpy_engine - Fast RIM/raking engine written in Rust.
#[pymodule]
fn _rimpy_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRakeResult>()?;
    m.add_function(wrap_pyfunction!(rim_iterate, m)?)?;
    m.add_function(wrap_pyfunction!(rim_iterate_grouped, m)?)?;
    Ok(())
}
