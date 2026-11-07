"""Types for the Rust kernels (native/, built into vib_tutorial._native), so mypy checks the calls into them.

Keep it in step with native/src: `tests/test_native.py` checks every name here is in the built module, and
every name the module has is here. A `PyReadonlyArray`/`PyReadwriteArray` argument takes a float64 NumPy
array only (no conversion); a `Vec` takes any sequence.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Floats = NDArray[np.float64]

ECCENTRICITY_MAX: float

def pi_film(
    k: float, c: float, x: float, y: float, vx: float, vy: float, omega: float
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    """(f_x, f_y, k_xx, k_xy, k_yx, k_yy, c_xx, c_xy, c_yy, Ψ): `fluid_bearing.pi_film_reference`."""

def full_film_damping(k: float, c: float, x: float, y: float) -> tuple[float, float, float]:
    """`fluid_bearing.full_film_damping_reference`."""

def expm(a: Floats) -> Floats:
    """e^A, for the tests to check against SciPy's."""

def foh_discretize(a: Floats, b: Floats, h: float) -> tuple[Floats, Floats, Floats]:
    """(Φ, Γ0, Γ1) as `simulator.foh_discretize` returns them, for the tests to check."""

class FilmSteps:
    """The rotor on oil films (`RotorSimulator._film_steps_reference`)."""

    def __init__(
        self,
        a: Floats,
        b: Floats,
        minv: Floats,
        cf: Floats,
        kf: Floats,
        p: Floats,
        h: float,
        idx: Sequence[int],
        films: Sequence[tuple[float, float, float]],
        drift: float,
        tol: float,
        correction: float,
        iterations: int,
    ) -> None: ...
    def steps(self, z: Floats, u0: Floats, u: Floats, w1: Floats, omega0: float, out: Floats) -> list[float]: ...
    @property
    def takes(self) -> int: ...
    @property
    def taken(self) -> list[tuple[float, float, float]] | None: ...

class FrictionSteps:
    """The chain with Coulomb friction (`Simulator._advance_friction_reference`, `_drive_reference`)."""

    def __init__(
        self,
        aw: Floats,
        p: Floats,
        masses: Floats,
        stiffness: Floats,
        damping: Floats,
        friction: Floats,
        static_friction: Floats,
        base: bool,
        k: int,
        events_per_mass: int,
    ) -> None: ...
    def advance(self, z: Floats, u0: float, u: Floats, h: float, out: Floats, losses: Floats) -> None: ...
    def drive(self, z: Floats, u0: float, u: Floats, h: float, out: Floats) -> None: ...

class LinearSteps:
    """The chain without friction (`Simulator._advance_linear_reference`)."""

    def __init__(self, phi: Floats, g0: Floats, g1: Floats) -> None: ...
    def advance(self, z: Floats, u0: float, u: Floats, out: Floats) -> None: ...

class RubSteps:
    """The disc rubbing on its stator ring (`RotorSimulator._rub_steps_reference`)."""

    def __init__(
        self,
        phi: Floats,
        g0: Floats,
        g1: Floats,
        n_dof: int,
        xd: int,
        yd: int,
        ring: tuple[float, float, float, float, float, float],
        slip: float,
        radius: float,
        tol: float,
        iterations: int,
    ) -> None: ...
    def steps(self, z: Floats, drive: Floats, w1: Floats, omega0: float, out: Floats, rubs: Floats) -> list[float]: ...
