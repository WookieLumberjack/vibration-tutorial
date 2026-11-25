"""Coordinates and bases (#478 Phase 4's new page): two masses between two walls, described in any basis.

The system is wall –k₁– m₁ –k₂– m₂ –k₃– wall, a damper beside each spring (all three c), so

    M = diag(m₁, m₂),   K = [[k₁ + k₂, −k₂], [−k₂, k₂ + k₃]],   C = c [[2, −1], [−1, 2]]

and its configuration is a point x = (x₁, x₂) in a plane. A **basis** is two arrows v₁, v₂ in that plane, the
columns of V, and the coordinates q in it are x = Vq. Put into Mẍ + Cẋ + Kx = 0 and multiplied by Vᵀ:

    (VᵀMV) q̈ + (VᵀCV) q̇ + (VᵀKV) q = 0

The motion is the same; only its description changes, so the frequencies (the pencil's eigenvalues,
`frequencies`) are the same in every basis. The off-diagonal terms are the coupling: dynamic in VᵀMV, static in
VᵀKV. When the arrows lie along the modes, both are diagonal (the modes are M- and K-orthogonal) and each
coordinate moves on its own at its frequency (`modes`).

With one arrow u the motion runs on a **rail**, x = u s, a one-mass oscillator whose ω² is Rayleigh's quotient
uᵀKu / uᵀMu (`rayleigh`): never below ω₁² nor above ω₂², and lowest along mode 1.

`Steps` steps any M, C, K system exactly, the two masses or the rail: free vibration z' = Az is stepped by
Φ = e^{Ah} (`foh.free`, in Rust where it's built), so a stiff spring can't make it unstable, and the energy the
dampers take out over each step is exact, zᵀWz with W from Van Loan's exponential
(`simulator.quadratic_integrals`). It's a `stepping.Stepper`; what a live change keeps is the page's
(`view_models/live_bases.py`).

Working on it: Qt-free (`tests/test_qt_free.py`); `tests/test_bases.py` checks the matrices, the invariance of
the frequencies, the modes' orthogonality, Rayleigh's bounds and the stepper against the closed form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .foh import free
from .simulator import quadratic_integrals
from .stepping import StepClock

TWO_PI = 2.0 * math.pi
STEPS_PER_PERIOD = 100  # of the fastest mode: the plane's path is drawn smooth from the samples
MAX_STEP = 0.005  # s
MAX_STEPS = 20_000  # an advance's most; past it the rest is dropped (`StepClock`)


@dataclass(frozen=True)
class TwoMass:
    """Two masses between two walls (see the module): kg, N/m, N·s/m."""

    m1: float = 1.0
    m2: float = 1.0
    k1: float = 400.0
    k2: float = 100.0  # the coupling spring, between the masses
    k3: float = 400.0
    c: float = 0.0  # each of the three dampers

    def matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(M, C, K)."""
        M = np.diag([self.m1, self.m2])
        K = np.array([[self.k1 + self.k2, -self.k2], [-self.k2, self.k2 + self.k3]])
        C = self.c * np.array([[2.0, -1.0], [-1.0, 2.0]])
        return M, C, K

    def modes(self) -> tuple[np.ndarray, np.ndarray]:
        """The undamped natural frequencies (Hz, ascending) and mode shapes (columns, each of unit length, its x₁
        positive: `unit`)."""
        M, _, K = self.matrices()
        w2, shapes = scipy.linalg.eigh(K, M)
        hz = np.sqrt(np.maximum(w2, 0.0)) / TWO_PI
        return hz, np.column_stack([unit(shapes[:, i]) for i in range(2)])


def unit(v: np.ndarray) -> np.ndarray:
    """`v` scaled to length 1 and signed so its first component is positive (its second, if the first is 0)."""
    v = np.asarray(v, dtype=float)
    n = float(np.hypot(*v)) if v.size == 2 else float(np.linalg.norm(v))
    if n == 0:
        raise ValueError("a zero vector has no direction")
    lead = v[0] if abs(v[0]) > 1e-12 * n else v[1]
    return v / n * (1.0 if lead > 0 else -1.0)


def transformed(V: np.ndarray, A: np.ndarray) -> np.ndarray:
    """VᵀAV: a matrix in the basis whose arrows are V's columns."""
    return V.T @ A @ V


def frequencies(M: np.ndarray, K: np.ndarray) -> np.ndarray:
    """The undamped natural frequencies (Hz, ascending) of M q̈ + K q = 0, in whatever basis M and K are in."""
    w2 = scipy.linalg.eigh(K, M, eigvals_only=True)
    return np.sqrt(np.maximum(w2, 0.0)) / TWO_PI


def rayleigh(u: np.ndarray, M: np.ndarray, K: np.ndarray) -> float:
    """Rayleigh's quotient uᵀKu / uᵀMu: the ω² (rad²/s²) of the motion held to the line along u."""
    return float(u @ K @ u) / float(u @ M @ u)


def rayleigh_hz(u: np.ndarray, M: np.ndarray, K: np.ndarray) -> float:
    return math.sqrt(max(rayleigh(u, M, K), 0.0)) / TWO_PI


def angle(v: np.ndarray) -> float:
    """The direction of the line along `v` (°, in [0, 180)): a line's, so v and −v are the same."""
    a = math.degrees(math.atan2(float(v[1]), float(v[0]))) % 180.0
    return 0.0 if a >= 180.0 - 1e-9 else a


def direction(degrees: float) -> np.ndarray:
    """The unit vector at `degrees` from the x₁ axis."""
    r = math.radians(degrees)
    return np.array([math.cos(r), math.sin(r)])


def state_matrix(M: np.ndarray, C: np.ndarray, K: np.ndarray) -> np.ndarray:
    """A in z' = Az, z = [x, v]."""
    n = M.shape[0]
    Minv = np.linalg.inv(M)
    return np.block([[np.zeros((n, n)), np.eye(n)], [-Minv @ K, -Minv @ C]])


class Steps:
    """M ẍ + C ẋ + K x = 0 stepped exactly from a state (see the module): `t`, `z` = [x, v], `dissipated` (J, the
    dampers' since the start), and `advance(duration)` giving that stretch's samples."""

    def __init__(
        self, M: np.ndarray, C: np.ndarray, K: np.ndarray, z: np.ndarray | None = None, t: float = 0.0
    ) -> None:
        self.M, self.C, self.K = (np.atleast_2d(np.asarray(a, dtype=float)) for a in (M, C, K))
        self.n = self.M.shape[0]
        self.A = state_matrix(self.M, self.C, self.K)
        self.z = np.zeros(2 * self.n) if z is None else np.asarray(z, dtype=float).copy()
        self.t = t
        self.dissipated = 0.0
        self.clock = StepClock()
        w2 = scipy.linalg.eigh(self.K, self.M, eigvals_only=True)
        self.fastest_hz = math.sqrt(max(float(w2.max()), 0.0)) / TWO_PI
        self._loss: dict[float, np.ndarray] = {}  # W a step size

    def step_size(self) -> float:
        """STEPS_PER_PERIOD a period of the fastest mode, at most MAX_STEP, rounded down to a power of two."""
        f = self.fastest_hz
        h = MAX_STEP if f <= 0 else min(MAX_STEP, 1.0 / (STEPS_PER_PERIOD * f))
        return 2.0 ** math.floor(math.log2(h))

    def _loss_form(self, h: float) -> np.ndarray:
        W = self._loss.get(h)
        if W is None:
            Q = np.zeros((2 * self.n, 2 * self.n))
            Q[self.n :, self.n :] = self.C  # the dampers' rate of loss, vᵀCv
            (W,) = quadratic_integrals(self.A, h, [Q])
            self._loss[h] = W
        return W

    def advance(self, duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step on by about `duration` s (whole steps, the rest carried): the samples' times, states (rows) and the
        dampers' running loss (J) at each."""
        h = self.step_size()
        steps = self.clock.steps(duration, h, MAX_STEPS)
        if steps == 0:
            return np.empty(0), np.empty((0, 2 * self.n)), np.empty(0)
        zs = free(self.A, h, steps, self.z)
        if np.any(self.C):
            W = self._loss_form(h)
            lost = self.dissipated + np.cumsum(np.einsum("ij,jk,ik->i", zs[:-1], W, zs[:-1]))
        else:
            lost = np.full(steps, self.dissipated)
        ts = self.t + h * np.arange(1, steps + 1)
        self.z = zs[-1].copy()
        self.t = float(ts[-1])
        self.dissipated = float(lost[-1])
        return ts, zs[1:], lost
