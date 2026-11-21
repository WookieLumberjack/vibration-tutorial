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

A fast spin needs many steps a second (a kick to 250 rad/s: ~16,000), so `advance` takes at most `max_steps`
and drops the rest of what was asked when it hits it (the page's compute budget: the run slows below real time
rather than owing more each frame), and with `thin` keeps a sample only every `store_stride` steps while no
torque acts (`STORE_PER_TURN` a turn), carried across calls. Every step under a torque is kept, so the work and
impulse summed over the samples (the *Check* tab) stay exact: the steps left out had none.

Working on it: `tests/test_rigid_body.py` checks the tensors against the standard formulas, the principal axes,
the conservation laws, the intermediate-axis flip and a kick's change in L.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

import numpy as np

from . import rotation
from .ode import rk4_step
from .stepping import StepClock

STEPS_PER_TURN = 400
MAX_STEP = 5e-3  # s
MAX_STEPS_PER_ADVANCE = 100_000
STORE_PER_TURN = 50  # samples a turn kept with `thin` (a step is 1/STEPS_PER_TURN of one)


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
    """The body about G, from `omega` (rad/s, in the body's axes), turned by `q` (its axes along space's unless
    given)."""

    body: Body
    omega: tuple[float, float, float]
    torque: tuple[float, float, float] = (0.0, 0.0, 0.0)
    held: Torque = Torque.NONE
    torque_until: float = 0.0  # s: the torque acts while t is below it
    t: float = 0.0
    q: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    L: list[float] = field(default_factory=list)
    _clock: StepClock = field(default_factory=StepClock)
    _skipped: int = 0  # torque-free steps since the last one kept (`thin`)
    last_steps: int = 0  # the steps the last `advance` took
    dropped: float = 0.0  # s of the time asked for that it left out, at `max_steps`
    _lead: bool = False  # a kick came after a step left out: the next thinned advance starts with the state now

    def __post_init__(self) -> None:
        self.I = self.body.inertia()
        self.I_inv = np.linalg.inv(self.I)
        if not self.L:  # L in space: B I ω, ω in the body's axes
            B = rotation.matrix(np.asarray(self.q, dtype=float))
            self.L = [float(v) for v in B @ (self.I @ np.asarray(self.omega, dtype=float))]

    def kick(self, duration: float) -> None:
        """Apply the torque from now for `duration` s."""
        self.torque_until = self.t + duration
        self._lead = self._skipped > 0

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

    def store_stride(self, h: float) -> int:
        """Keep every this many torque-free steps of size h: at most STORE_PER_TURN a turn, none closer than
        MAX_STEP (a slow spin keeps them all)."""
        return max(1, min(int(MAX_STEP / h + 1e-9), STEPS_PER_TURN // STORE_PER_TURN))

    def advance(self, duration: float, max_steps: int = MAX_STEPS_PER_ADVANCE, thin: bool = False) -> Samples:
        """Step on by `duration` (whole steps, the part of one left over carried to the next call), a sample a
        step (with `thin`, a sample every `store_stride` steps while no torque acts). At most `max_steps`: past
        them the rest of `duration` is dropped, not owed."""
        h = self.step_size()
        n = self._clock.steps(duration, h, min(max_steps, MAX_STEPS_PER_ADVANCE))
        self.last_steps, self.dropped = n, self._clock.dropped
        if n == 0:
            return _empty()
        stride = self.store_stride(h) if thin else 1
        lead = int(thin and self._lead)  # the kick's start, so its first step's work sums from there
        self._lead = False
        keep = np.zeros(n + lead, dtype=bool)
        t_out = np.empty(n + lead)
        q_out = np.empty((n + lead, 4))
        L_out = np.empty((n + lead, 3))
        tau_out = np.zeros((n + lead, 3))
        q, L, t = np.asarray(self.q, dtype=float), np.asarray(self.L, dtype=float), self.t
        if lead:
            keep[0], t_out[0], q_out[0], L_out[0] = True, t, q, L  # no torque acted over the step ending here
        for i in range(lead, n + lead):
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
            self._skipped += 1
            if on or self._skipped >= stride:  # a step under a torque is always kept (the Check's sums)
                keep[i], self._skipped = True, 0
        self.q, self.L, self.t = [float(v) for v in q], [float(v) for v in L], t
        if stride == 1 and not lead:
            return Samples(t_out, q_out, L_out, tau_out)
        return Samples(t_out[keep], q_out[keep], L_out[keep], tau_out[keep])


@dataclass
class Drift:
    """The same torque-free run stepped three ways (`compare`), each against a fine reference, at times t: how far
    each orientation has turned from the true one (degrees), how far the matrix has strayed from being a rotation
    (‖BᵀB − 1‖), how far an unnormalized quaternion's length has (| |q| − 1 |), and the Euler angles' 1/sin θ. The
    Euler angles' error is NaN once a step met the lock (an infinite rate)."""

    t: np.ndarray
    matrix: np.ndarray
    euler: np.ndarray
    quaternion: np.ndarray
    orthogonality: np.ndarray
    length: np.ndarray
    lock: np.ndarray


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def _qdot(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    a, b, c, d = q
    x, y, z = w
    return 0.5 * np.array([-b * x - c * y - d * z, a * x + c * z - d * y, a * y - b * z + d * x, a * z + b * y - c * x])


def _apart(B: np.ndarray, R: np.ndarray) -> float:
    """The angle (degrees) of the turn from rotation R to B (B made the nearest rotation first)."""
    U, _, Vt = np.linalg.svd(B)
    M = (U @ Vt).T @ R  # the turn between them: trace 1 + 2 cos χ, and M − Mᵀ = 2 sin χ [n]×
    # atan2 of the two, not acos of the cosine alone: that loses half the digits near χ = 0 (1e-16 → 1e-6°)
    sine = math.hypot(M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1])
    return math.degrees(math.atan2(sine, float(np.trace(M)) - 1.0))


def compare(body: Body, omega: np.ndarray, q0: np.ndarray, duration: float, h: float, fine: int = 8) -> Drift:
    """Step the torque-free body from q0 and ω (in its axes) for `duration` with RK4 at step h, three ways: its
    rotation matrix (Ḃ = B[ω]×, never corrected), its Euler angles (z-x-z, their rates), and its quaternion
    (q̇ = ½q ⊗ ω, made unit after each step; and once not, for |q|'s drift). L stays fixed in space, and ω =
    I⁻¹BᵀL from each one's own orientation. The reference is the quaternion stepped at h/fine."""
    I_inv = np.linalg.inv(body.inertia())
    q0 = np.asarray(q0, dtype=float) / np.linalg.norm(q0)
    B0 = rotation.matrix(q0)
    L = B0 @ body.inertia() @ np.asarray(omega, dtype=float)

    def rate(B: np.ndarray) -> np.ndarray:
        return I_inv @ (B.T @ L)

    n = max(1, round(duration / h))
    out = {k: np.empty(n + 1) for k in ("matrix", "euler", "quaternion", "orthogonality", "length", "lock")}
    B = B0.copy()
    angles = np.array(rotation.to_euler(B0)[:3])
    q, raw, ref = q0.copy(), q0.copy(), q0.copy()
    euler_ok = True

    def euler_rates(y: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(y)):  # a stage already past the lock
            return np.full(3, np.nan)
        Be = rotation.from_euler(*y)
        return np.array(rotation.euler_rates(y[1], y[2], rate(Be)))

    for i in range(n + 1):
        R = rotation.matrix(ref)
        out["matrix"][i] = _apart(B, R)
        out["orthogonality"][i] = float(np.linalg.norm(B.T @ B - np.eye(3)))
        out["quaternion"][i] = _apart(rotation.matrix(q), R)
        out["length"][i] = abs(float(np.linalg.norm(raw)) - 1.0)
        s = math.sin(angles[1])
        out["lock"][i] = 1.0 / abs(s) if abs(s) > rotation.LOCK else math.inf
        out["euler"][i] = _apart(rotation.from_euler(*angles), R) if euler_ok else math.nan
        if i == n:
            break
        B = rk4_step(lambda M: M @ _skew(rate(M)), B, h)  # Ḃ = B[ω]×: the body's axes turn at ω
        if euler_ok:
            with np.errstate(all="ignore"):
                angles = rk4_step(euler_rates, angles, h)
            euler_ok = bool(np.all(np.isfinite(angles)))
        q = rk4_step(lambda p: _qdot(p, rate(rotation.matrix(p / np.linalg.norm(p)))), q, h)
        q /= np.linalg.norm(q)
        raw = rk4_step(lambda p: _qdot(p, rate(rotation.matrix(p / np.linalg.norm(p)))), raw, h)
        for _ in range(fine):
            ref = rk4_step(lambda p: _qdot(p, rate(rotation.matrix(p))), ref, h / fine)
            ref /= np.linalg.norm(ref)
    return Drift(np.arange(n + 1) * h, **out)
