# rimpy — Rust-accelerated RIM engine

## Architecture

```
rimpy/
├── Cargo.toml                    # Rust crate definition
├── pyproject.toml                # Python package (maturin build)
├── src/                          # Rust source
│   ├── lib.rs                    # PyO3 bindings + module exports
│   └── engine.rs                 # Core RIM algorithm (pure Rust)
├── python/                       # Python source (maturin mixed layout)
│   └── rimpy/
│       ├── __init__.py           # Public API (unchanged)
│       ├── _engine.py            # Shim: tries Rust → falls back to Python
│       ├── _engine_py.py         # Pure Python/NumPy fallback
│       ├── _rake.py              # Narwhals orchestration (unchanged)
│       └── _loaders.py           # Scheme loaders (unchanged)
├── tests/
│   └── test_backend_parity.py    # Validates Rust == Python results
└── benchmarks/
    └── bench_engine.py           # Performance comparison
```

### What changed, what didn't

| File | Status | Why |
|------|--------|-----|
| `_rake.py` | **Unchanged** | Orchestration layer, not a bottleneck |
| `_loaders.py` | **Unchanged** | I/O bound, no benefit from Rust |
| `__init__.py` | **Unchanged** | Public API stays the same |
| `_engine.py` | **Now a shim** | Tries `rimpy_engine` (Rust), falls back to `_engine_py` |
| `_engine_py.py` | **Renamed original** | Pure Python fallback for portability |
| `src/engine.rs` | **New** | Core RIM loop in Rust (raw slices, zero alloc) |
| `src/lib.rs` | **New** | PyO3 bindings accepting NumPy arrays |

### Why this design

- **Zero API changes**: `_rake.py` still does `from ._engine import RakeResult, rim_iterate`. Users see no difference.
- **Graceful fallback**: No Rust compiler? No problem — pure Python still works.
- **Check which backend**: `rimpy._engine.get_backend()` returns `"rust"` or `"python"`.

## Building

### Prerequisites

- Rust toolchain: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- uv (already installed)

### Development build

```bash
# uv reads [build-system] → maturin, handles everything
uv pip install -e ".[dev]"

# Verify
uv run python -c "from rimpy._engine import get_backend; print(get_backend())"
# → "rust"
```

### Release build (wheel)

```bash
uv build
# Output: dist/rimpy-0.2.0-cp312-cp312-*.whl
```

### Publish to PyPI

```bash
uv publish
```

### Run benchmarks

```bash
uv run python benchmarks/bench_engine.py
```

### Run tests

```bash
uv run pytest tests/ -v
```

## Rust engine design decisions

### Why raw `Vec<f64>` instead of ndarray or Arrow?

RIM's core operation is **indexed gather/scatter on a 1D array**:
- Gather: sum `weights[indices]` for each category
- Scatter: `weights[indices] *= multiplier`

This is trivially expressed with plain slices. `ndarray` would add abstraction overhead
and allocate on fancy indexing. Arrow is a columnar format, not a compute engine.

Raw slices let `rustc` + LLVM auto-vectorize with SIMD, and the hot loop does **zero
heap allocations** (the `old_weights` buffer is pre-allocated and reused via `copy_from_slice`).

### Why Rayon for grouped raking?

Each country/segment gets its own weights vector — no shared mutable state.
This is embarrassingly parallel. `rayon::par_iter()` gives near-linear speedup
across CPU cores with zero synchronization overhead.

### Data ingress

Currently accepts NumPy arrays via `pyo3-numpy` (near-zero-copy for contiguous arrays).

**Future**: Arrow FFI path for zero-copy from Polars. The engine itself stays the same —
only the ingress layer changes.

## Integration with existing code

Your `_rake.py` and `_loaders.py` remain identical. The only structural change is:

```python
# Before (in _rake.py):
from ._engine import RakeResult, rim_iterate

# After (same import — _engine.py is now a shim):
from ._engine import RakeResult, rim_iterate  # tries Rust, falls back to Python
```

The shim handles everything transparently.
