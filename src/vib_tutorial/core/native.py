"""The Rust kernels (`vib_tutorial._native`, built from native/), or None.

Each kernel speeds up a Python function that stays the reference: the module
picks the kernel where it's built and falls back to the Python otherwise, and
the tests compare the two. VIB_NATIVE=0 runs the Python everywhere.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from types import ModuleType

module: ModuleType | None = None
if os.environ.get("VIB_NATIVE", "1") != "0":
    try:
        # A plain import statement, not importlib: PyInstaller finds what the app bundles by its imports.
        import vib_tutorial._native as _native

        module = _native
    except ImportError:  # not built: the Python reference serves
        pass


def kernel[F: Callable](name: str, reference: F) -> F:
    """The Rust kernel `name`, or `reference` if the module isn't there or lacks it."""
    return getattr(module, name, reference)
