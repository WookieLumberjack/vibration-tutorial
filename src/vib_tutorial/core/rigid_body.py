"""A rigid body tumbling in space: its inertia tensor, principal axes and Euler's equations, with no gravity.

The *3D rigid body* page. A `Body` is a uniform shape (a cuboid of sides a × b × c, a cylinder of radius a
and length c along its z, or a sphere of radius a) of mass m, plus a point mass m_p at `offset` from the
shape's centre, all in the shape's own axes x, y, z (the body's axes). Its centre of mass G is
m_p·offset/(m + m_p) from the shape's centre, and its inertia tensor about G is the shape's about its centre,
moved to G by the parallel-axis theorem, I_G = I_C + m(|d|²1 − d dᵀ) (d from G to the shape's centre),
plus the point mass's own m_p(|p|²1 − p pᵀ) (p from G to it). With the point mass off an axis the tensor in the
shape's axes has products of inertia; its eigenvectors are the principal axes, in which it is diagonal
(`Body.principal`: the moments ascending, the axes as columns, right-handed).

`Tumbler` moves the body about G: its orientation as a unit quaternion q (body to space; `rotation.matrix`)
and its angular momentum about G in space, L. Each step is classical Runge–Kutta on

    q̇ = ½ q ⊗ (0, ω_body),   ω_body = I⁻¹ Bᵀ L,   L̇ = τ,

(B the rotation matrix: Bᵀ L is L in the body's axes), q renormalized after each step. Torque-free, L is fixed
and so is the energy ½ L·ω; RK4's drift in both falls as h⁴ (the *Check* tab shows it). The torque is held in
the body's axes (it turns with the body: a thruster) or in space's (a fixed push), and acts while t <
`torque_until`, so a kick is a torque for a set time. The step is a set fraction of a turn at the body's
current rate (`STEPS_PER_TURN`), capped at `MAX_STEP`.

Working on it: `tests/test_rigid_body.py` checks the tensors against the standard formulas, the principal axes,
the conservation laws, the intermediate-axis flip and a kick's change in L.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

import numpy as np

from . import rotation

STEPS_PER_TURN = 400
MAX_STEP = 5e-3  # s
MAX_STEPS_PER_ADVANCE = 100_000


class Shape(enum.Enum):
    CUBOID = "Cuboid (a box)"
    CYLINDER = "Cylinder"
    SPHERE = "Sphere"


SHAPES = tuple(Shape)


@dataclass(frozen=True)
class Body:
    """A uniform shape plus a point mass; lengths in m, along the shape's axes x, y, z."""

    shape: Shape = Shape.CUBOID
    mass: float = 1.0  # the shape's, kg
    a: float = 0.30  # a cuboid's side along x; a cylinder's or a sphere's radius
    b: float = 0.20  # a cuboid's side along y
    c: float = 0.10  # a cuboid's side along z; a cylinder's length
    point: float = 0.0  # the point mass, kg
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)  # from the shape's centre

    @property
    def total(self) -> float:
        return self.mass + self.point

    def shape_inertia(self) -> np.ndarray:
        """The shape's inertia tensor about its own centre, in its axes: diagonal."""
        m, a, b, c = self.mass, self.a, self.b, self.c
        if self.shape is Shape.CUBOID:
            return np.diag([m * (b * b + c * c), m * (a * a + c * c), m * (a * a + b * b)]) / 12.0
        if self.shape is Shape.CYLINDER:
            across = m * (3.0 * a * a + c * c) / 12.0
            return np.diag([across, across, 0.5 * m * a * a])
        return np.eye(3) * 0.4 * m * a * a

    def centre(self) -> np.ndarray:
        """G, the centre of mass, from the shape's centre."""
        return self.point * np.asarray(self.offset, dtype=float) / self.total

    def inertia(self) -> np.ndarray:
        """The inertia tensor about G in the shape's axes: the shape's moved to G (parallel axes), plus the point
        mass's."""
        G = self.centre()
        d = -G  # G to the shape's centre
        p = np.asarray(self.offset, dtype=float) - G  # G to the point mass
        return (
            self.shape_inertia()
            + self.mass * (d @ d * np.eye(3) - np.outer(d, d))
            + self.point * (p @ p * np.eye(3) - np.outer(p, p))
        )

    def principal(self) -> tuple[np.ndarray, np.ndarray]:
        """The principal moments, ascending, and their axes as the columns of a rotation matrix P (shape's axes to
        principal: I = P diag Pᵀ), each pointing the way of the shape's axis it lies nearest."""
        moments, P = np.linalg.eigh(self.inertia())
        for k in range(3):
            j = int(np.argmax(np.abs(P[:, k])))
            if P[j, k] < 0:
                P[:, k] = -P[:, k]
        if np.linalg.det(P) < 0:
            P[:, 2] = -P[:, 2]
        return moments, P

    def corners(self) -> np.ndarray:
        """A cuboid's eight corners from G, in the shape's axes (the 3D view's box)."""
        half = 0.5 * np.array([self.a, self.b, self.c])
        signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=float)
        return signs * half - self.centre()


class Torque(enum.Enum):
    NONE = "None"
    BODY = "Held in the body's axes (turns with it)"
    SPACE = "Held in space's axes (a fixed push)"


TORQUES = tuple(Torque)


@dataclass
class Samples:
    t: np.ndarray
    q: np.ndarray  # (n, 4)
    momentum: np.ndarray  # L in space (n, 3)
    torque: np.ndarray  # τ in space (n, 3), as it acted over the step ending at each sample


def _empty() -> Samples:
    return Samples(np.zeros(0), np.zeros((0, 4)), np.zeros((0, 3)), np.zeros((0, 3)))


@dataclass
class Tumbler:
    """The body about G, from `omega` (rad/s, in the body's axes) with its axes along space's."""

    body: Body
    omega: tuple[float, float, float]
    torque: tuple[float, float, float] = (0.0, 0.0, 0.0)
    held: Torque = Torque.NONE
    torque_until: float = 0.0  # s: the torque acts while t is below it
    t: float = 0.0
    q: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    L: list[float] = field(default_factory=list)
    _owed: float = 0.0

    def __post_init__(self) -> None:
        self.I = self.body.inertia()
        self.I_inv = np.linalg.inv(self.I)
        if not self.L:
            self.L = [float(v) for v in self.I @ np.asarray(self.omega, dtype=float)]

    def kick(self, duration: float) -> None:
        """Apply the torque from now for `duration` s."""
        self.torque_until = self.t + duration

    def body_rate(self) -> np.ndarray:
        """ω in the body's axes now."""
        B = rotation.matrix(np.asarray(self.q))
        return self.I_inv @ (B.T @ np.asarray(self.L))

    def energy(self) -> float:
        w = self.body_rate()
        return 0.5 * float(w @ self.I @ w)

    def step_size(self) -> float:
        rate = float(np.linalg.norm(self.body_rate()))
        if self.held is not Torque.NONE and self.t < self.torque_until:
            tau = float(np.linalg.norm(self.torque))
            least = float(np.linalg.eigvalsh(self.I)[0])
            rate = max(rate, math.sqrt(tau / least))  # a push from rest turns it at about √(τ/I)
        return MAX_STEP if rate == 0.0 else min(2.0 * math.pi / rate / STEPS_PER_TURN, MAX_STEP)

    def _torque(self, on: bool, q: np.ndarray) -> np.ndarray:
        if self.held is Torque.NONE or not on:
            return np.zeros(3)
        tau = np.asarray(self.torque, dtype=float)
        return rotation.matrix(q) @ tau if self.held is Torque.BODY else tau

    def _rates(self, on: bool, q: np.ndarray, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = self.I_inv @ (rotation.matrix(q).T @ L)
        a, b, c, d = q
        x, y, z = w
        qdot = 0.5 * np.array(
            [-b * x - c * y - d * z, a * x + c * z - d * y, a * y - b * z + d * x, a * z + b * y - c * x]
        )
        return qdot, self._torque(on, q)

    def advance(self, duration: float) -> Samples:
        """Step on by `duration` (whole steps, the part of one left over carried to the next call), a sample a
        step."""
        h = self.step_size()
        owed = self._owed + duration
        n = min(int(owed / h + 1e-9), MAX_STEPS_PER_ADVANCE)
        self._owed = owed - n * h if n < MAX_STEPS_PER_ADVANCE else 0.0
        if n == 0:
            return _empty()
        t_out = np.empty(n)
        q_out = np.empty((n, 4))
        L_out = np.empty((n, 3))
        tau_out = np.empty((n, 3))
        q, L, t = np.asarray(self.q, dtype=float), np.asarray(self.L, dtype=float), self.t
        for i in range(n):
            hh = h
            if t < self.torque_until < t + h:  # the torque stops within this step: end the step there instead,
                hh = self.torque_until - t  # so a kick's impulse is τ times its duration
            on = t < self.torque_until  # the same all step: steps end where the torque stops
            k1q, k1L = self._rates(on, q, L)
            k2q, k2L = self._rates(on, q + hh / 2 * k1q, L + hh / 2 * k1L)
            k3q, k3L = self._rates(on, q + hh / 2 * k2q, L + hh / 2 * k2L)
            k4q, k4L = self._rates(on, q + hh * k3q, L + hh * k3L)
            q = q + hh / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
            q /= np.linalg.norm(q)
            L = L + hh / 6 * (k1L + 2 * k2L + 2 * k3L + k4L)
            t += hh
            t_out[i], q_out[i], L_out[i] = t, q, L
            tau_out[i] = self._torque(on, q)  # what acted over the step that ends here
        self.q, self.L, self.t = [float(v) for v in q], [float(v) for v in L], t
        return Samples(t_out, q_out, L_out, tau_out)
