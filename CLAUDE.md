# rimpy - Project Guide for Claude Code

## What is this?
A Rust-powered RIM (Iterative Proportional Fitting) survey weighting library for Python. Supports both Polars and Pandas via Narwhals.

## Project Structure
```
python/rimpy/
  __init__.py        # Public API exports
  _rake.py           # Main orchestration (rake, rake_by, rake_by_scheme, etc.)
  _engine.py         # Engine shim: imports Rust backend, falls back to Python
  _engine_py.py      # Pure Python/NumPy fallback engine
  _loaders.py        # Scheme file loaders (requires polars)
src/
  lib.rs             # PyO3 bindings (rim_iterate, RakeResult)
  engine.rs          # Core Rust RIM algorithm
tests/
  test_backend_parity.py  # Backend consistency tests
```

## Environment & Build

- **Package manager**: uv (no pip installed in venvs)
- **Build system**: maturin (PyO3) — maturin is NOT on PATH, always use `python -m maturin`
- **Rust extension**: compiled to `.pyd` (Windows) / `.so` (Linux/macOS)

### Common commands
```bash
# Build Rust extension into local venv
python -m maturin develop --release

# Build wheel for distribution
python -m maturin build --release

# Run tests
.venv/Scripts/python -m pytest tests/ -v

# Install test deps
uv pip install pytest polars pandas -p .venv/Scripts/python
```

### Installing into another venv
```bash
# Build wheel then install
python -m maturin build --release
uv pip install target/wheels/rimpy-*.whl --reinstall --python /path/to/other/.venv/Scripts/python
```

## Design Rationale

- **Zero API changes**: `_rake.py` imports `from ._engine import RakeResult, rim_iterate` — the shim handles Rust vs Python transparently. Users see no difference.
- **Graceful fallback**: No Rust compiler? Pure Python still works. Check with `rimpy._engine.get_backend()`.
- **Why raw `Vec<f64>` in Rust**: RIM's core is indexed gather/scatter on a 1D array. Plain slices let LLVM auto-vectorize with SIMD. The hot loop does zero heap allocations (`old_weights` buffer is pre-allocated and reused).
- **Why Rayon for grouped raking**: Each group gets its own weights vector — no shared mutable state. Embarrassingly parallel with near-linear speedup.
- **Data ingress**: NumPy arrays via `pyo3-numpy` (near-zero-copy for contiguous arrays). Future: Arrow FFI for zero-copy from Polars.
- **What stays in Python**: `_rake.py` (orchestration, not a bottleneck), `_loaders.py` (I/O bound), `__init__.py` (public API).

## Known Gotchas

- **Locked .pyd file**: If `maturin develop` fails with "file being used by another process", rename the .pyd first: `mv ...pyd ...pyd.old`, then rebuild.
- **Polars Int32 dtype**: `pl.when().then(int_literal)` produces `Int32`. The `_extract_columns_to_numpy()` in `_rake.py` casts to `np.int64` to handle this.
- **Rust `extract_i64_array`** (lib.rs): Only handles `i64` and `f64` numpy arrays. Any other dtype must be cast on the Python side before calling Rust.
- **PyRakeResult `#[new]`**: The Rust `RakeResult` class has a `#[new]` constructor so Python code can instantiate it directly (needed in `_rake.py` for groups with no scheme).
- **`_loaders.py`** has a top-level `import polars` making it a hard dependency despite polars being listed as optional. This can cause `import rimpy` to fail without polars.
- **Convergence loop**: Stall detection runs AFTER raking and sets `converged=True` when weights stabilize. This is intentional — stabilized weights are converged for practical survey weighting.

## Release Process

1. Bump version in `pyproject.toml`
2. Commit and push
3. Tag: `git tag v0.x.y && git push origin v0.x.y`
4. GitHub Action builds wheels (Linux/Windows/macOS, Python 3.12-3.14) and publishes to PyPI
5. Create GitHub Release: `gh release create v0.x.y --title "v0.x.y" --notes "..."`
