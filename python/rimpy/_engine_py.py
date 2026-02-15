"""
Pure Python/NumPy RIM engine (fallback when Rust extension isn't available).

This is the original _engine.py, kept as a fallback for platforms
where the Rust extension can't be compiled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class RakeResult:
    """Result of a raking operation."""

    weights: NDArray[np.float64]
    iterations: int
    converged: bool
    efficiency: float
    weight_min: float
    weight_max: float

    @property
    def weight_ratio(self) -> float:
        return self.weight_max / self.weight_min if self.weight_min > 0 else float("inf")

    def summary(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "converged": self.converged,
            "efficiency": round(self.efficiency, 2),
            "weight_min": round(self.weight_min, 4),
            "weight_max": round(self.weight_max, 4),
            "weight_ratio": round(self.weight_ratio, 2),
        }


def _build_index_cache(
    data: NDArray,
    target_codes: list[int | float | str],
) -> dict[int | float | str, NDArray[np.intp]]:
    cache = {}
    for code in target_codes:
        mask = data == code
        cache[code] = np.nonzero(mask)[0]
    return cache


def rake_on_variable(
    weights: NDArray[np.float64],
    index_cache: dict[int | float | str, NDArray[np.intp]],
    target_props: dict[int | float | str, float],
    n_rows: int,
) -> NDArray[np.float64]:
    for code, target_prop in target_props.items():
        indices = index_cache.get(code)
        if indices is None or len(indices) == 0:
            continue
        target_count = target_prop * n_rows
        if target_count < 1e-10:
            target_count = 1e-10
        current_sum = weights[indices].sum()
        if current_sum > 0:
            multiplier = target_count / current_sum
            weights[indices] *= multiplier
    return weights


def apply_caps(
    weights: NDArray[np.float64],
    min_cap: float | None,
    max_cap: float | None,
) -> NDArray[np.float64]:
    if max_cap is None and min_cap is None:
        return weights
    max_iter = 100
    for _ in range(max_iter):
        changed = False
        if max_cap is not None and weights.max() > max_cap:
            weights = np.clip(weights, None, max_cap)
            weights = weights / weights.mean()
            changed = True
        if min_cap is not None and weights.min() < min_cap:
            weights = np.clip(weights, min_cap, None)
            weights = weights / weights.mean()
            changed = True
        if not changed:
            break
    return weights


def calculate_efficiency(weights: NDArray[np.float64]) -> float:
    n = len(weights)
    if n == 0:
        return 0.0
    sum_w = weights.sum()
    sum_w_sq = (weights**2).sum()
    if sum_w_sq == 0:
        return 0.0
    return (sum_w**2 / (n * sum_w_sq)) * 100


def rim_iterate(
    column_data: dict[str, NDArray],
    targets: dict[str, dict[int | float | str, float]],
    max_iterations: int = 1000,
    convergence_threshold: float = 0.01,
    min_cap: float | None = None,
    max_cap: float | None = None,
    cap_correction: bool = True,
) -> RakeResult:
    first_col = next(iter(column_data.values()))
    n_rows = len(first_col)

    if n_rows == 0:
        return RakeResult(
            weights=np.array([], dtype=np.float64),
            iterations=0,
            converged=True,
            efficiency=100.0,
            weight_min=1.0,
            weight_max=1.0,
        )

    normalized_targets: dict[str, dict[int | float | str, float]] = {}
    for col, props in targets.items():
        total = sum(props.values())
        if total > 1.5:
            normalized_targets[col] = {k: v / 100.0 for k, v in props.items()}
        else:
            normalized_targets[col] = dict(props)

    index_caches = {}
    for col, props in normalized_targets.items():
        if col not in column_data:
            raise KeyError(f"Target column '{col}' not found in data")
        index_caches[col] = _build_index_cache(column_data[col], list(props.keys()))

    weights = np.ones(n_rows, dtype=np.float64)

    effective_min_cap = min_cap
    effective_max_cap = max_cap
    if cap_correction:
        if effective_max_cap is not None:
            effective_max_cap += 0.0001
        if effective_min_cap is not None:
            effective_min_cap -= 0.0001

    pct_still = 1 - convergence_threshold
    diff_error = float("inf")
    diff_error_old = float("inf")
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        old_weights = weights.copy()

        if diff_error >= pct_still * diff_error_old and iteration > 1:
            converged = diff_error < convergence_threshold
            break

        for col, props in normalized_targets.items():
            weights = rake_on_variable(weights, index_caches[col], props, n_rows)

        if effective_min_cap is not None or effective_max_cap is not None:
            weights = apply_caps(weights, effective_min_cap, effective_max_cap)

        diff_error_old = diff_error
        diff_error = np.abs(weights - old_weights).sum()
    else:
        converged = False

    weights = np.where(weights == 0, 1.0, weights)

    return RakeResult(
        weights=weights,
        iterations=iteration,
        converged=converged,
        efficiency=calculate_efficiency(weights),
        weight_min=float(weights.min()),
        weight_max=float(weights.max()),
    )
