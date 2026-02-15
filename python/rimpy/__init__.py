"""
rimpy - Fast RIM (raking) survey weighting with narwhals.

Supports both polars and pandas DataFrames.

Example
-------
>>> import polars as pl
>>> import rimpy
>>>
>>> df = pl.DataFrame({
...     "gender": [1, 1, 1, 2, 2],
...     "age": [1, 2, 2, 1, 2],
... })
>>> targets = {
...     "gender": {1: 50, 2: 50},
...     "age": {1: 40, 2: 60},
... }
>>> weighted = rimpy.rake(df, targets)
>>> print(weighted["weight"])
"""

from importlib.metadata import version, PackageNotFoundError

from ._rake import (
    GroupedRakeResult,
    convert_from_weightipy,
    rake,
    rake_by,
    rake_by_scheme,
    rake_by_scheme_with_diagnostics,
    rake_by_with_diagnostics,
    rake_with_diagnostics,
    validate_schemes,
    validate_targets,
    weight_summary,
)
from ._engine import RakeResult
from ._loaders import load_schemes, load_schemes_wide

try:
    __version__ = version("rimpy")
except PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development

__all__ = [
    # Main functions
    "rake",
    "rake_by",
    "rake_by_scheme",
    "rake_with_diagnostics",
    "rake_by_with_diagnostics",
    "rake_by_scheme_with_diagnostics",
    # Utilities
    "weight_summary",
    "validate_targets",
    "validate_schemes",
    "convert_from_weightipy",
    # Loaders
    "load_schemes",
    "load_schemes_wide",
    # Result types
    "RakeResult",
    "GroupedRakeResult",
]
