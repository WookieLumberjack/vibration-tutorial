"""Classical Runge–Kutta 4 for the models that integrate their own equations (#316: the copies in one place).

- `rk4_step(f, y, h)`: one step of y' = f(y) on an array of any shape (3D rigid body's axes, its quaternion).
- `rk4_step_floats(f, y, h, k1)`: one step on a state of Python floats, faster than NumPy for a handful of
  components; `k1` when the caller has worked it out already (the sliding top reads its contact force there).
- `rk4(rhs, z0, h, steps, keep)`: many steps of an autonomous z' = rhs(z), every `keep`-th state; z may hold many
  systems side by side (z[state, case]), or one as floats (Self-excited's sweeps and runs).

The Numerical integration page's RK4 (`core/integration.py`) stays its own: it is what that page teaches, written
out beside the other schemes. Working on it: `tests/test_ode.py`; the models' own tests check their runs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

Rhs = Callable[[Any], Any]  # the state (an array, or a tuple of floats) → its derivative, the same shape


def rk4_step(f: Callable[[np.ndarray], np.ndarray], y: np.ndarray, h: float) -> np.ndarray:
    """One RK4 step of y' = f(y), y an array of any shape."""
    k1 = f(y)
    k2 = f(y + h / 2 * k1)
    k3 = f(y + h / 2 * k2)
    k4 = f(y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_step_floats(
    f: Callable[[Sequence[float]], Sequence[float]],
    y: Sequence[float],
    h: float,
    k1: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """One RK4 step of y' = f(y) on a state of floats; `k1` = f(y) if the caller has it already."""
    half, sixth = 0.5 * h, h / 6.0
    k1 = f(y) if k1 is None else k1
    k2 = f(tuple(a + half * b for a, b in zip(y, k1)))
    k3 = f(tuple(a + half * b for a, b in zip(y, k2)))
    k4 = f(tuple(a + h * b for a, b in zip(y, k3)))
    return tuple(a + sixth * (b1 + 2 * b2 + 2 * b3 + b4) for a, b1, b2, b3, b4 in zip(y, k1, k2, k3, k4))


def rk4(rhs: Rhs, z0: np.ndarray, h: float, steps: int, keep: int = 1) -> np.ndarray:
    """Classical RK4 of z' = rhs(z) (autonomous; z may hold many systems side by side, z[state, case]), every
    `keep`-th state from the first."""
    z0 = np.asarray(z0, dtype=float)
    if z0.ndim == 1:  # one system: its components as floats
        y: tuple[float, ...] = tuple(float(c) for c in z0)
        floats = [y]
        for i in range(1, steps + 1):
            y = rk4_step_floats(rhs, y, h)
            if i % keep == 0:
                floats.append(y)
        return np.array(floats)
    z = z0.copy()
    out = [z.copy()]
    for i in range(1, steps + 1):
        z = rk4_step(lambda v: np.asarray(rhs(v)), z, h)
        if i % keep == 0:
            out.append(z.copy())
    return np.array(out)
