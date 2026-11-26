"""State-space modes (#478 Phase 4, the page after Coordinates & bases): a basis of the state that uncouples any damping.

The two masses' M ẍ + C ẋ + K x = 0 is written first order in the state z = [x, ẋ] (four coordinates):

    ż = A z,   A = [[0, I], [−M⁻¹K, −M⁻¹C]]   (`bases.state_matrix`)

A **basis** of the state is a 4×4 matrix B whose columns are four state vectors, and the coordinates in it are
w = B⁻¹z, so ẇ = (B⁻¹AB) w. Every basis here keeps its columns in **pairs**, one pair a plane the page draws, so
B⁻¹AB is four 2×2 blocks: the diagonal ones each pair's own motion, the off-diagonal ones the coupling between them.
A is not symmetric, so the matrix in a new basis is B⁻¹AB, not BᵀAB: the transpose works for M and K only because
they are symmetric and the basis is used on both sides of the energy.

Three bases (`BASES`), each made by a function:

- `mass_basis`: each mass's own displacement and velocity, its velocity scaled by ω̂ᵢ = √(Kᵢᵢ/Mᵢᵢ) (the mass's
  frequency with the other held) so a pair's swing is round: (x₁, ẋ₁/ω̂₁), (x₂, ẋ₂/ω̂₂). The springs and dampers
  couple the pairs.
- `undamped_basis`: the modes of M and K (the last page's), V's unit columns φᵣ, each with its velocity:
  (qᵣ, q̇ᵣ/ωᵣ). M and K are diagonal in it, so only the dampers (VᵀCV) can still couple the pairs, and they do
  unless the damping is proportional (C = αM + βK, say).
- `state_basis`: A's own eigenvectors. They come in conjugate pairs, yᵣ = [ψᵣ; λᵣψᵣ] with λᵣ = σᵣ + iωᵣ, and the
  real basis takes each pair's real and imaginary parts as its two columns, Re yᵣ and Im yᵣ (`state_modes`). Then

      A [Re y, Im y] = [Re y, Im y] [[σ, ω], [−ω, σ]]

  so B⁻¹AB is block diagonal whatever C is: each pair turns at ω and shrinks at σ on its own, a clean decaying
  spiral, clockwise as a phase plane (x, ẋ/ω) turns. An overdamped mode's pair is two real eigenvalues, its
  block diag(λa, λb): two decays, no swing.

ψᵣ, the eigenvector's displacements, is complex when the damping isn't proportional: the masses don't move in step,
and a mode alone traces an ellipse in the (x₁, x₂) plane, Re ψ cos ωt − Im ψ sin ωt, not a line. Each ψᵣ is
scaled to unit length and turned to be as real as it can be (ψᵀψ real and positive), its larger real part
positive, so with proportional damping it is the undamped mode's φᵣ exactly.

Working on it: Qt-free (`tests/test_qt_free.py`); `tests/test_state_modes.py` checks each basis's blocks, the
eigen pairs against A, the spiral's exact decay, the proportional case against the last page's modes, and the
overdamped pair.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .bases import state_matrix, unit

BASES = ("mass", "undamped", "state")
REAL = 1e-9  # an eigenvalue whose imaginary part is under this of its size is real (an overdamped mode's)


@dataclass(frozen=True)
class StateMode:
    """One mode of the state: a pair of A's eigenvalues and the two real columns of the basis they give."""

    eigenvalues: tuple[complex, complex]  # (λ, λ̄) for a mode that swings; two real decay rates (slower first) if not
    shape: np.ndarray  # ψ, complex, unit length, as real as it turns (see the module); overdamped: the slower's
    columns: np.ndarray  # 4×2, real: Re y and Im y (overdamped: the two real eigenvectors, slower first)

    @property
    def swings(self) -> bool:
        return abs(self.eigenvalues[0].imag) > 0.0

    @property
    def sigma(self) -> float:
        """The decay rate σ (1/s, negative): the pair's real part; overdamped, the slower one's."""
        return float(self.eigenvalues[0].real)

    @property
    def omega(self) -> float:
        """The damped frequency ω (rad/s): 0 if it doesn't swing."""
        return abs(float(self.eigenvalues[0].imag))

    @property
    def natural(self) -> float:
        """|λ|, the undamped-like frequency (rad/s): √(λa λb) for an overdamped pair."""
        a, b = self.eigenvalues
        return math.sqrt(abs((a * b).real))

    @property
    def zeta(self) -> float:
        """The damping ratio: −(λa + λb)/2|λ|, so −σ/|λ| for a pair that swings, above 1 for an overdamped one."""
        a, b = self.eigenvalues
        return -float((a + b).real) / (2.0 * self.natural)

    @property
    def phase(self) -> float:
        """The angle (°, in (−180, 180]) by which m₂ leads m₁ in this mode: 0 or 180 for a real shape."""
        if not self.shape[0]:
            return 0.0
        a = math.degrees(cmath.phase(complex(self.shape[1]) / complex(self.shape[0])))
        return 180.0 if a <= -180.0 + 1e-9 else a


def _as_real(p: np.ndarray) -> complex:
    """The factor that scales the complex displacements p to unit length and turns them as real as they go: pᵀp real
    and positive, the larger of the real parts positive."""
    n = float(np.linalg.norm(p))
    turn = cmath.exp(-0.5j * cmath.phase(complex(p @ p))) / n
    q = p * turn
    lead = int(np.argmax(np.abs(q.real)))
    return turn if q.real[lead] > 0 else -turn


def _real_shape(y: np.ndarray) -> np.ndarray:
    """A real eigenvector scaled so its displacements have unit length, signed as `bases.unit`."""
    d = y[: y.size // 2]
    s = unit(d)
    return y * (s[0] / d[0] if abs(d[0]) > 1e-12 * np.linalg.norm(d) else s[1] / d[1])


def state_modes(M: np.ndarray, C: np.ndarray, K: np.ndarray) -> tuple[StateMode, ...]:
    """A's modes, slowest first (by |λ|), each a pair of eigenvalues and the basis's two columns (see the module)."""
    A = state_matrix(M, C, K)
    lam, Y = np.linalg.eig(A)
    swinging, real = [], []
    for i, value in enumerate(lam):
        if abs(value.imag) > REAL * abs(value):
            if value.imag > 0:
                swinging.append((value, Y[:, i]))
        else:
            real.append((value.real, Y[:, i].real))
    modes: list[StateMode] = []
    n = M.shape[0]
    for value, y in swinging:
        y = y * _as_real(y[:n])
        value = complex(value.real, abs(value.imag))
        modes.append(StateMode((value, value.conjugate()), y[:n], np.column_stack([y.real, y.imag])))
    real.sort(key=lambda r: abs(r[0]))
    for k in range(0, len(real) - 1, 2):  # the real ones two by two, slowest first: an overdamped mode each
        (a, ya), (b, yb) = real[k], real[k + 1]
        ya, yb = _real_shape(ya), _real_shape(yb)
        modes.append(StateMode((complex(a), complex(b)), ya[:n].astype(complex), np.column_stack([ya, yb])))
    modes.sort(key=lambda m: m.natural)
    return tuple(modes)


def state_basis(M: np.ndarray, C: np.ndarray, K: np.ndarray) -> np.ndarray:
    """B whose pairs of columns are A's modes (see the module)."""
    return np.column_stack([m.columns for m in state_modes(M, C, K)])


def mass_basis(M: np.ndarray, K: np.ndarray) -> np.ndarray:
    """B whose pairs are each mass's x and ẋ/ω̂ (see the module)."""
    n = M.shape[0]
    B = np.zeros((2 * n, 2 * n))
    for i in range(n):
        B[i, 2 * i] = 1.0
        B[n + i, 2 * i + 1] = own_frequency(M, K, i)
    return B


def own_frequency(M: np.ndarray, K: np.ndarray, i: int) -> float:
    """ω̂ᵢ = √(Kᵢᵢ/Mᵢᵢ): mass i's frequency with the others held (rad/s)."""
    return math.sqrt(max(float(K[i, i]) / float(M[i, i]), 0.0))


def undamped_modes(M: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The modes of M and K: ω (rad/s, ascending) and the unit shapes φ as columns (`bases.unit`)."""
    w2, shapes = scipy.linalg.eigh(K, M)
    return np.sqrt(np.maximum(w2, 0.0)), np.column_stack([unit(shapes[:, i]) for i in range(shapes.shape[1])])


def undamped_basis(M: np.ndarray, K: np.ndarray) -> np.ndarray:
    """B whose pairs are each undamped mode's q and q̇/ω (see the module)."""
    omega, V = undamped_modes(M, K)
    n = M.shape[0]
    B = np.zeros((2 * n, 2 * n))
    for r in range(n):
        B[:n, 2 * r] = V[:, r]
        B[n:, 2 * r + 1] = omega[r] * V[:, r]
    return B


def basis(name: str, M: np.ndarray, C: np.ndarray, K: np.ndarray) -> np.ndarray:
    """The basis `name` (one of `BASES`)."""
    if name == "mass":
        return mass_basis(M, K)
    if name == "undamped":
        return undamped_basis(M, K)
    if name == "state":
        return state_basis(M, C, K)
    raise ValueError(f"no basis {name!r}: one of {BASES}")


def in_basis(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """B⁻¹AB: A in the basis whose columns are B's."""
    return np.linalg.solve(B, A @ B)


def coordinates(B: np.ndarray, z: np.ndarray) -> np.ndarray:
    """w = B⁻¹z, for one state or a row a state."""
    z = np.asarray(z, dtype=float)
    return np.linalg.solve(B, z.T).T


def coupling(Ab: np.ndarray) -> float:
    """The largest entry of the off-diagonal 2×2 blocks of a matrix in a paired basis, over its largest entry."""
    n = Ab.shape[0] // 2
    off = 0.0
    for r in range(n):
        for s in range(n):
            if r != s:
                off = max(off, float(np.abs(Ab[2 * r : 2 * r + 2, 2 * s : 2 * s + 2]).max()))
    return off / max(float(np.abs(Ab).max()), 1e-300)
