"""What the Components tab plots, for the 3D rigid body and the Gyroscope (#409 S-6).

A run's ω, L or τ, as its three components in the axes picked and its size, or the inertia tensor's six
entries: the page works out the vector or the tensor in those axes from its own record, and `series` names and
lays out the curves. The widget is `gui/components_tab.py`.

Working on it: `tests/test_gui_smoke.py`'s Components tests of both pages draw every show in every frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

SHOWS = ("ω, angular velocity", "L, angular momentum", "τ, torque", "I, inertia tensor")
SHOW_W, SHOW_L, SHOW_TAU, SHOW_I = range(4)
UNITS = ("rad/s", "N·m·s", "N·m", "kg·m²")
SPANS = (("the last 2 s", 2.0), ("the last 10 s", 10.0), ("the whole run", None))
TENSOR_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))  # the six entries of a symmetric tensor


@dataclass(frozen=True)
class Series:
    """The tab's curves: `curves` (legend label, values) and the vector's size (None for the tensor)."""

    name: str  # the left axis's: "ω"
    unit: str
    curves: tuple[tuple[str, np.ndarray], ...]
    size: np.ndarray | None
    size_label: str = ""

    @property
    def labels(self) -> tuple[str, ...]:
        """The legend's entries, in order: what changes it."""
        return tuple(label for label, _ in self.curves) + ((self.size_label,) if self.size is not None else ())


def series(show: int, names: Sequence[str], in_axes: np.ndarray) -> Series:
    """The curves for show `show` (SHOW_W …): `in_axes` the vector's components in the axes named `names`, (n, 3),
    or for SHOW_I the tensor in them, (n, 3, 3)."""
    symbol = SHOWS[show][0]
    if show == SHOW_I:
        curves = tuple((f"I {names[i]}{names[j]}", in_axes[:, i, j]) for i, j in TENSOR_PAIRS)
        return Series(symbol, UNITS[show], curves, None)
    curves = tuple((f"{symbol} {names[i]}", in_axes[:, i]) for i in range(3))
    return Series(symbol, UNITS[show], curves, np.linalg.norm(in_axes, axis=1), f"|{symbol}|")
