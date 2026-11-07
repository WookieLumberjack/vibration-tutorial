"""The Rust kernels (`vib_tutorial._native`, built from native/), or None.

Each kernel speeds up a Python function that stays the reference: the module
picks the kernel where it's built and falls back to the Python otherwise, and
the tests compare the two. VIB_NATIVE=0 runs the Python everywhere.

vib_tutorial/_native.pyi gives the kernels' types; the classes are re-exported below so mypy checks
their calls (`native.LinearSteps(...)`, where `module` is set).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING

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


# The kernels' classes, typed by vib_tutorial/_native.pyi: use them only where `module` is set. (A module held
# in a variable, as `module` is, has no attributes for mypy, so its calls would go unchecked.)
if TYPE_CHECKING:
    from vib_tutorial._native import FilmSteps, FrictionSteps, LinearSteps, RubSteps
else:
    FilmSteps = getattr(module, "FilmSteps", None)
    FrictionSteps = getattr(module, "FrictionSteps", None)
    LinearSteps = getattr(module, "LinearSteps", None)
    RubSteps = getattr(module, "RubSteps", None)

__all__ = ["FilmSteps", "FrictionSteps", "LinearSteps", "RubSteps", "kernel", "module"]
