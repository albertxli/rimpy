//! Core RIM (Raking / Iterative Proportional Fitting) algorithm.
//!
//! All computation uses plain slices — no ndarray, no Arrow.
//! The compiler auto-vectorizes the inner loops with SIMD.

use std::collections::HashMap;
use indexmap::IndexMap;

/// Result of a single raking operation.
#[derive(Debug, Clone)]
pub struct RakeResult {
    pub weights: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub efficiency: f64,
    pub weight_min: f64,
    pub weight_max: f64,
}

impl RakeResult {
    pub fn weight_ratio(&self) -> f64 {
        if self.weight_min > 0.0 {
            self.weight_max / self.weight_min
        } else {
            f64::INFINITY
        }
    }
}

/// Options for the raking algorithm.
#[derive(Debug, Clone)]
pub struct RakeOpts {
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub min_cap: Option<f64>,
    pub max_cap: Option<f64>,
    pub cap_correction: bool,
}

impl Default for RakeOpts {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            convergence_threshold: 0.01,
            min_cap: None,
            max_cap: None,
            cap_correction: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Index cache: built once per variable, reused across all iterations
// ---------------------------------------------------------------------------

/// Map from category code → row indices where that code appears.
///
/// We use i64 as the key type because survey codes are typically integers.
/// String codes are handled by hashing on the Python side before passing in.
pub type IndexCache = HashMap<i64, Vec<usize>>;

/// Build an index cache in a single pass over the column.
pub fn build_index_cache(column: &[i64], codes: &[i64]) -> IndexCache {
    // Pre-allocate with expected codes
    let mut cache: IndexCache = codes.iter().map(|&c| (c, Vec::new())).collect();

    for (i, &val) in column.iter().enumerate() {
        if let Some(indices) = cache.get_mut(&val) {
            indices.push(i);
        }
    }

    cache
}

// ---------------------------------------------------------------------------
// Single-variable raking adjustment
// ---------------------------------------------------------------------------

/// Adjust weights so the weighted distribution of one variable matches targets.
///
/// Operates entirely in-place — zero allocations.
#[inline]
fn rake_on_variable(
    weights: &mut [f64],
    index_cache: &IndexCache,
    target_props: &HashMap<i64, f64>,
    n: f64,
) {
    for (&code, &target_prop) in target_props {
        if let Some(indices) = index_cache.get(&code) {
            if indices.is_empty() {
                continue;
            }

            let target_count = target_prop * n;
            if target_count < 1e-10 {
                continue;
            }

            // Gather: sum weights at indices
            let current_sum: f64 = indices.iter().map(|&i| weights[i]).sum();

            if current_sum > 0.0 {
                let multiplier = target_count / current_sum;

                // Scatter-multiply: update weights at indices
                for &i in indices {
                    weights[i] *= multiplier;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Weight capping with renormalization
// ---------------------------------------------------------------------------

/// Apply min/max caps and renormalize. In-place, zero allocations.
fn apply_caps(weights: &mut [f64], min_cap: Option<f64>, max_cap: Option<f64>) {
    if min_cap.is_none() && max_cap.is_none() {
        return;
    }

    for _ in 0..100 {
        let mut changed = false;

        if let Some(cap) = max_cap {
            let max_w = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max_w > cap {
                for w in weights.iter_mut() {
                    if *w > cap {
                        *w = cap;
                    }
                }
                renormalize(weights);
                changed = true;
            }
        }

        if let Some(cap) = min_cap {
            let min_w = weights.iter().cloned().fold(f64::INFINITY, f64::min);
            if min_w < cap {
                for w in weights.iter_mut() {
                    if *w < cap {
                        *w = cap;
                    }
                }
                renormalize(weights);
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }
}

/// Renormalize weights so they average to 1.0.
#[inline]
fn renormalize(weights: &mut [f64]) {
    let n = weights.len() as f64;
    if n == 0.0 {
        return;
    }
    let mean: f64 = weights.iter().sum::<f64>() / n;
    if mean > 0.0 {
        for w in weights.iter_mut() {
            *w /= mean;
        }
    }
}

// ---------------------------------------------------------------------------
// Weighting efficiency
// ---------------------------------------------------------------------------

/// Efficiency = (sum(w))^2 / (n * sum(w^2)) * 100
/// Perfect weights (all 1.0) = 100%.
pub fn calculate_efficiency(weights: &[f64]) -> f64 {
    let n = weights.len() as f64;
    if n == 0.0 {
        return 0.0;
    }

    let sum_w: f64 = weights.iter().sum();
    let sum_w_sq: f64 = weights.iter().map(|w| w * w).sum();

    if sum_w_sq == 0.0 {
        return 0.0;
    }

    (sum_w * sum_w) / (n * sum_w_sq) * 100.0
}

// ---------------------------------------------------------------------------
// Target normalization
// ---------------------------------------------------------------------------

/// Normalize targets: if they sum > 1.5, treat as percentages and divide by 100.
fn normalize_targets(
    targets: &IndexMap<String, HashMap<i64, f64>>,
) -> IndexMap<String, HashMap<i64, f64>> {
    targets
        .iter()
        .map(|(col, props)| {
            let total: f64 = props.values().sum();
            let normalized = if total > 1.5 {
                // Percentages → proportions
                props.iter().map(|(&k, &v)| (k, v / 100.0)).collect()
            } else {
                props.clone()
            };
            (col.clone(), normalized)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Main entry point: single-group raking
// ---------------------------------------------------------------------------

/// Core RIM iteration loop.
///
/// `column_data`: maps column name → array of integer codes (i64).
/// `targets`: maps column name → {code: target_proportion_or_pct}.
///
/// This is the function `_rake.py` calls via PyO3.
pub fn rim_iterate(
    column_data: &HashMap<String, &[i64]>,
    targets: &IndexMap<String, HashMap<i64, f64>>,
    opts: &RakeOpts,
) -> RakeResult {
    // Determine n from first column
    let n = match column_data.values().next() {
        Some(col) => col.len(),
        None => {
            return RakeResult {
                weights: vec![],
                iterations: 0,
                converged: true,
                efficiency: 100.0,
                weight_min: 1.0,
                weight_max: 1.0,
            };
        }
    };

    if n == 0 {
        return RakeResult {
            weights: vec![],
            iterations: 0,
            converged: true,
            efficiency: 100.0,
            weight_min: 1.0,
            weight_max: 1.0,
        };
    }

    // Normalize targets
    let normalized = normalize_targets(targets);

    // Validate columns exist
    for col in normalized.keys() {
        if !column_data.contains_key(col) {
            panic!("Target column '{}' not found in data", col);
        }
    }

    // Build index caches (one-time cost, amortized over all iterations)
    let index_caches: HashMap<String, IndexCache> = normalized
        .iter()
        .map(|(col, props)| {
            let codes: Vec<i64> = props.keys().cloned().collect();
            let cache = build_index_cache(column_data[col.as_str()], &codes);
            (col.clone(), cache)
        })
        .collect();

    // Initialize weights to 1.0
    let mut weights = vec![1.0_f64; n];
    let n_f64 = n as f64;

    // Apply cap correction
    let effective_max_cap = opts.max_cap.map(|c| {
        if opts.cap_correction {
            c + 0.0001
        } else {
            c
        }
    });
    let effective_min_cap = opts.min_cap.map(|c| {
        if opts.cap_correction {
            c - 0.0001
        } else {
            c
        }
    });

    // Convergence tracking
    let pct_still = 1.0 - opts.convergence_threshold;
    let mut diff_error = f64::INFINITY;
    let mut converged = false;
    let mut iteration = 0;

    // Pre-allocate old_weights buffer (reused each iteration — no allocation in loop)
    let mut old_weights = vec![0.0_f64; n];

    for iter in 1..=opts.max_iterations {
        iteration = iter;

        // Save current weights (memcpy, no allocation)
        old_weights.copy_from_slice(&weights);

        // Rake on each variable
        for (col, props) in &normalized {
            rake_on_variable(&mut weights, &index_caches[col], props, n_f64);
        }

        // Apply caps
        apply_caps(&mut weights, effective_min_cap, effective_max_cap);

        // Convergence metric: sum of absolute differences
        let new_diff_error: f64 = weights
            .iter()
            .zip(old_weights.iter())
            .map(|(w, o)| (w - o).abs())
            .sum();

        // Converged if error below threshold
        if new_diff_error < opts.convergence_threshold {
            converged = true;
            break;
        }

        // Converged if weights have stabilized (progress stalled)
        if iter > 1 && new_diff_error >= pct_still * diff_error {
            converged = true;
            break;
        }

        diff_error = new_diff_error;
    }

    // Replace zeros with 1.0
    for w in weights.iter_mut() {
        if *w == 0.0 {
            *w = 1.0;
        }
    }

    let efficiency = calculate_efficiency(&weights);
    let weight_min = weights.iter().cloned().fold(f64::INFINITY, f64::min);
    let weight_max = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    RakeResult {
        weights,
        iterations: iteration,
        converged,
        efficiency,
        weight_min,
        weight_max,
    }
}

// ---------------------------------------------------------------------------
// Grouped raking (parallel via Rayon)
// ---------------------------------------------------------------------------

/// Data for a single group, ready for raking.
pub struct GroupData {
    pub column_data: HashMap<String, Vec<i64>>,
    pub targets: IndexMap<String, HashMap<i64, f64>>,
}

/// Rake multiple groups in parallel using Rayon.
///
/// Returns a map of group_key → RakeResult.
pub fn rim_iterate_grouped<K: Send + Sync + Eq + std::hash::Hash + Clone>(
    groups: Vec<(K, GroupData)>,
    opts: &RakeOpts,
) -> Vec<(K, RakeResult)> {
    use rayon::prelude::*;

    groups
        .into_par_iter()
        .map(|(key, group)| {
            // Convert owned Vecs to slices for rim_iterate
            let col_refs: HashMap<String, &[i64]> = group
                .column_data
                .iter()
                .map(|(k, v)| (k.clone(), v.as_slice()))
                .collect();

            let result = rim_iterate(&col_refs, &group.targets, opts);
            (key, result)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_raking() {
        // 5 respondents, 3 male, 2 female → target 50/50
        let gender: Vec<i64> = vec![1, 1, 1, 2, 2];
        let age: Vec<i64> = vec![1, 2, 2, 1, 2];

        let mut column_data = HashMap::new();
        column_data.insert("gender".to_string(), gender.as_slice());
        column_data.insert("age".to_string(), age.as_slice());

        let mut targets = HashMap::new();
        targets.insert(
            "gender".to_string(),
            HashMap::from([(1, 50.0), (2, 50.0)]),
        );
        targets.insert(
            "age".to_string(),
            HashMap::from([(1, 40.0), (2, 60.0)]),
        );

        let opts = RakeOpts::default();
        let result = rim_iterate(&column_data, &targets, &opts);

        assert!(result.converged);
        assert!(result.efficiency > 0.0);
        assert!(result.efficiency <= 100.0);

        // Weights should average ~1.0
        let mean: f64 = result.weights.iter().sum::<f64>() / result.weights.len() as f64;
        assert!((mean - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_efficiency_perfect_weights() {
        let weights = vec![1.0; 100];
        let eff = calculate_efficiency(&weights);
        assert!((eff - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_with_caps() {
        let gender: Vec<i64> = vec![1, 1, 1, 1, 2]; // very skewed
        let mut column_data = HashMap::new();
        column_data.insert("gender".to_string(), gender.as_slice());

        let mut targets = HashMap::new();
        targets.insert(
            "gender".to_string(),
            HashMap::from([(1, 50.0), (2, 50.0)]),
        );

        let opts = RakeOpts {
            max_cap: Some(3.0),
            ..Default::default()
        };

        let result = rim_iterate(&column_data, &targets, &opts);
        // Max weight should respect the cap (with epsilon)
        assert!(result.weight_max <= 3.0 + 0.001);
    }

    #[test]
    fn test_empty_data() {
        let column_data: HashMap<String, &[i64]> = HashMap::new();
        let targets = HashMap::new();
        let opts = RakeOpts::default();

        let result = rim_iterate(&column_data, &targets, &opts);
        assert!(result.weights.is_empty());
        assert!(result.converged);
    }
}
