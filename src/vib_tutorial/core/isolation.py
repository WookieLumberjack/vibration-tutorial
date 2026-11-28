"""Vibration isolation: a machine with a rotating unbalance on mounts, and what reaches the floor.

The *Vibration isolation* page. The mounts are sized by their static deflection δ under the machine's
weight (k = Mg/δ, so f_n = √(g/δ)/2π whatever M is) and a damping ratio, as one of three kinds:

- `VISCOUS`: a spring and a damper side by side;
- `RELAXATION`: the damper in series with a spring N k (an elastically supported damper), so the damper can't
  stiffen the mount at high frequency;
- `TWO_STAGE`: the machine on mounts on an intermediate mass (a fraction of the machine's), itself on the same
  mounts again.

The transmissibility (the force on the floor over the force the machine makes, which is also the motion that
reaches the machine over the floor's) is worked out exactly from each kind's dynamic stiffness. `start_up`
runs the machine up to speed through the mount's resonance by the exact first-order-hold update, the
unbalance's force including its tangential part as the speed changes; `let_go` releases it, stopped, from a
pull, to ring at the mounts' natural frequency.

Live (#478, the page rebuilt to run as it's handled), the isolator is stepped as a state space with three inputs,
u = [f, g, ġ]: the unbalance's force on the machine and the floor's motion and velocity under the mounts
(`live_system`, which also gives the force the mounts put on the floor, D z + E u, and where the machine's
velocity is in z). A change while it runs (the mounts, the mass, the kind) keeps the machine's place and carries
its momentum by `carry`'s rules, and a hand holds it with everything else settled round it (`held`).

Working on it
-------------
Isolation sizes the mounts by their sag (k = Mg/δ), and `start_up`'s steady state is the transmissibility (a
test checks). `live_system` is the same isolator driven both ways, and gives the same T from its state
space (`tests/test_live_isolation.py`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import minimize_scalar

from .foh import respond
from .sdof import TWO_PI

G = 9.80665  # m/s²


class Mount(Enum):
    VISCOUS = "Spring + damper"
    RELAXATION = "Relaxation"
    TWO_STAGE = "Two stages"


@dataclass(frozen=True)
class Isolator:
    """A machine of mass M (kg) with an unbalance m_e e (kg·m), running at `rpm`, on mounts."""

    mass: float = 100.0
    rpm: float = 1500.0
    unbalance: float = 0.01  # kg·m
    deflection: float = 0.005  # m, the mounts' static deflection under the machine (each stage's, for two)
    zeta: float = 0.05
    mount: Mount = Mount.VISCOUS
    stiffness_ratio: float = 3.0  # N: the relaxation damper's spring over the mount's
    stage_mass: float = 0.5  # the intermediate mass over the machine's
    block: float = 0.0  # kg of inertia block bolted to the machine (the mounts keep their deflection)

    def __post_init__(self) -> None:
        if not (self.mass > 0 and self.deflection > 0 and self.zeta >= 0 and self.rpm >= 0 and self.block >= 0):
            raise ValueError("the mass and the deflection must be positive, and the rest not negative")
        if self.mount is Mount.RELAXATION and not self.stiffness_ratio > 0:
            raise ValueError("the relaxation spring must be positive")
        if self.mount is Mount.TWO_STAGE and not self.stage_mass > 0:
            raise ValueError("the intermediate mass must be positive")

    @property
    def total(self) -> float:
        """The mass on the mounts: the machine and its inertia block."""
        return self.mass + self.block

    @property
    def stiffness(self) -> float:
        """k = Mg/δ: the mounts under the machine (and block)."""
        return self.total * G / self.deflection

    @property
    def damping(self) -> float:
        return 2.0 * self.zeta * math.sqrt(self.stiffness * self.total)

    @property
    def natural_hz(self) -> float:
        """√(k/M)/2π = √(g/δ)/2π: the single-stage mount's natural frequency."""
        return math.sqrt(G / self.deflection) / TWO_PI

    @property
    def speed_hz(self) -> float:
        return self.rpm / 60.0

    def force_at(self, freq_hz: np.ndarray | float) -> np.ndarray:
        """The unbalance's force amplitude m_e e Ω² at a steady speed."""
        return self.unbalance * (TWO_PI * np.asarray(freq_hz, dtype=float)) ** 2

    def _two_stage(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(M, C, K) of [machine, intermediate]: the lower mounts carry both, at the same deflection."""
        m1, m2 = self.total, self.stage_mass * self.total
        k1, c1 = self.stiffness, self.damping
        k2 = (m1 + m2) * G / self.deflection
        c2 = 2.0 * self.zeta * math.sqrt(k2 * (m1 + m2))
        M = np.diag([m1, m2])
        K = np.array([[k1, -k1], [-k1, k1 + k2]])
        C = np.array([[c1, -c1], [-c1, c1 + c2]])
        return M, C, K

    def held_stage(self, x: float) -> float:
        """Where the intermediate mass settles while the machine is held at x (m): zero but on two stages."""
        if self.mount is not Mount.TWO_STAGE:
            return 0.0
        K = self._two_stage()[2]
        return float(-K[1, 0] * x / K[1, 1])

    def mount_stiffness(self, omega: np.ndarray) -> np.ndarray:
        """A single stage's dynamic stiffness K*(ω): the force on the floor per meter of the machine's motion."""
        w = np.asarray(omega, dtype=float)
        k, c = self.stiffness, self.damping
        if self.mount is Mount.RELAXATION:
            k1 = self.stiffness_ratio * k
            return k + k1 * (1j * w * c) / (k1 + 1j * w * c)
        return k + 1j * w * c

    def response(self, freqs_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(machine's receptance X/F, transmissibility F_floor/F) under a harmonic force F on the machine."""
        w = TWO_PI * np.asarray(freqs_hz, dtype=float)
        if self.mount is Mount.TWO_STAGE:
            M, C, K = self._two_stage()
            Z = K[None] - (w * w)[:, None, None] * M[None] + 1j * w[:, None, None] * C[None]
            H = np.linalg.solve(Z, np.broadcast_to(np.array([[1.0], [0.0]], dtype=complex), (w.size, 2, 1)))[:, :, 0]
            lower = K[1, 1] - K[0, 0] + 1j * w * (C[1, 1] - C[0, 0])  # k2 + iωc2
            return H[:, 0], lower * H[:, 1]
        Ks = self.mount_stiffness(w)
        H = 1.0 / (Ks - w * w * self.total)
        return H, Ks * H

    def transmissibility(self, freqs_hz: np.ndarray) -> np.ndarray:
        return np.abs(self.response(freqs_hz)[1])

    def peak_transmissibility(self) -> float:
        """The transmissibility's largest value at any speed (its resonance's, or 1 if it has none), refined by
        a bounded search round the largest on a grid."""
        f = self.natural_hz * np.geomspace(0.05, 20.0, 2000)
        T = self.transmissibility(f)
        i = int(np.argmax(T))
        if i == 0:
            return max(float(T[0]), 1.0)
        lo, hi = f[i - 1], f[min(i + 1, f.size - 1)]
        best = minimize_scalar(
            lambda x: -float(self.transmissibility(np.array([x]))[0]), bounds=(lo, hi), method="bounded"
        )
        return max(float(-best.fun), float(T[i]))

    def at_speed(self) -> tuple[float, float, float]:
        """(the machine's amplitude, the force on the floor, the transmissibility) at its running speed."""
        f = np.array([self.speed_hz])
        H, T = self.response(f)
        F = float(self.force_at(f)[0])
        return abs(complex(H[0])) * F, abs(complex(T[0])) * F, abs(complex(T[0]))

    def state_space(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(A, B, D) with z' = A z + B f and the force on the floor D z: z = [x, v] (viscous), [x, v, y] with
        y the relaxation spring's end on the damper, or [x1, x2, v1, v2] for two stages."""
        k, c, m = self.stiffness, self.damping, self.total
        if self.mount is Mount.VISCOUS:
            A = np.array([[0.0, 1.0], [-k / m, -c / m]])
            return A, np.array([[0.0], [1.0 / m]]), np.array([k, c])
        if self.mount is Mount.RELAXATION:
            k1 = self.stiffness_ratio * k
            if c == 0:  # no damper: the spring k1 hangs free
                A = np.array([[0.0, 1.0, 0.0], [-k / m, 0.0, 0.0], [0.0, 0.0, 0.0]])
                return A, np.array([[0.0], [1.0 / m], [0.0]]), np.array([k, 0.0, 0.0])
            # machine — k1 — y — c — floor: k1 (x − y) = c y'
            A = np.array([[0.0, 1.0, 0.0], [-(k + k1) / m, 0.0, k1 / m], [k1 / c, 0.0, -k1 / c]])
            return A, np.array([[0.0], [1.0 / m], [0.0]]), np.array([k + k1, 0.0, -k1])
        M, C, K = self._two_stage()
        Minv = np.linalg.inv(M)
        A = np.block([[np.zeros((2, 2)), np.eye(2)], [-Minv @ K, -Minv @ C]])
        B = np.concatenate([np.zeros(2), Minv[:, 0]])[:, None]
        k2, c2 = K[1, 1] - K[0, 0], C[1, 1] - C[0, 0]
        return A, B, np.array([0.0, k2, 0.0, c2])


@dataclass(frozen=True)
class StartUp:
    """A run up from rest (or a `let_go`): time, speed (Hz), the unbalance's force, the machine's displacement
    and the floor's force; the angle the unbalance has turned through (rad, from the top: its force along the
    mount is m_e e Ω² cos φ), and the intermediate mass's displacement (zero but on two stages)."""

    t: np.ndarray
    speed_hz: np.ndarray
    force: np.ndarray
    x: np.ndarray
    floor: np.ndarray
    phase: np.ndarray
    stage: np.ndarray


def start_up(iso: Isolator, ramp: float, hold: float, per_period: int = 40) -> StartUp:
    """Run the machine up from rest to its speed in `ramp` s at a constant acceleration, then hold for `hold` s.

    The unbalance turns through φ(t) = ∫Ω dt; its force along the mount is m_e e (Ω² cos φ + Ω' sin φ). The
    step resolves the running speed and the mount's modes `per_period` times a period.
    """
    w_end = TWO_PI * iso.speed_hz
    A, B, D = iso.state_space()
    fastest = max(iso.speed_hz, float(np.max(np.abs(np.linalg.eigvals(A)))) / TWO_PI, 1e-3)
    h = 1.0 / (per_period * fastest)
    n = math.ceil((ramp + hold) / h) + 1
    t = np.arange(n) * h
    alpha = w_end / ramp if ramp > 0 else 0.0
    w = np.where(t < ramp, alpha * t, w_end)
    phi = np.where(t < ramp, 0.5 * alpha * t * t, 0.5 * alpha * ramp * ramp + w_end * (t - ramp))
    dw = np.where(t < ramp, alpha, 0.0)
    f = iso.unbalance * (w * w * np.cos(phi) + dw * np.sin(phi))
    z = respond(A, B, h, f)
    return StartUp(t, w / TWO_PI, f, z[:, 0], z @ D, phi, _stage(iso, z))


def _stage(iso: Isolator, z: np.ndarray) -> np.ndarray:
    return z[:, 1] if iso.mount is Mount.TWO_STAGE else np.zeros(z.shape[0])


def let_go(iso: Isolator, x0: float, duration: float, per_period: int = 40) -> StartUp:
    """The machine, stopped, held at x0 from rest (long enough for everything else to settle) and let go.

    Held, the relaxation damper has relaxed (its spring's end where the machine is) and the intermediate mass
    has settled where its springs balance; let go, it rings at the mounts' natural frequency and dies away.
    """
    A, B, D = iso.state_space()
    fastest = max(float(np.max(np.abs(np.linalg.eigvals(A)))) / TWO_PI, 1e-3)
    h = 1.0 / (per_period * fastest)
    n = math.ceil(duration / h) + 1
    t = np.arange(n) * h
    if iso.mount is Mount.VISCOUS:
        z0 = np.array([x0, 0.0])
    elif iso.mount is Mount.RELAXATION:
        z0 = np.array([x0, 0.0, x0 if iso.damping > 0 else 0.0])
    else:
        z0 = np.array([x0, iso.held_stage(x0), 0.0, 0.0])
    z = respond(A, B, h, np.zeros(n), z0)
    zero = np.zeros(n)
    return StartUp(t, zero, zero, z[:, 0], z @ D, zero, _stage(iso, z))


def deflection_for(speed_hz: float, isolation: float) -> float:
    """The static deflection (m) undamped mounts need for a fraction `isolation` (0.9 = 90 %) at a speed.

    T = 1/(r² − 1) = 1 − isolation, so r² = 1 + 1/T, f_n = f/r and δ = g/ω_n².
    """
    T = 1.0 - isolation
    if not 0 < T < 1:
        raise ValueError("the isolation must be between 0 and 1")
    fn = speed_hz / math.sqrt(1.0 + 1.0 / T)
    return G / (TWO_PI * fn) ** 2


# ------------------------------------------------------------------ live: stepped as it runs


@dataclass(frozen=True)
class LiveSystem:
    """z' = A z + B u with u = [f, g, ġ] (the unbalance's force, the floor's motion and velocity); the force the
    mounts put on the floor is D z + E u (N, + along x, the machine's up). The machine's displacement is z[0] and
    its velocity z[`velocity`]; on two stages the intermediate mass's is z[1]."""

    A: np.ndarray
    B: np.ndarray
    D: np.ndarray
    E: np.ndarray
    velocity: int


def live_system(iso: Isolator) -> LiveSystem:
    """The isolator as a state space driven by its unbalance and its floor (see `LiveSystem`): z = [x, v]
    (spring + damper), [x, v, y] (relaxation: y the damper's end on its spring Nk) or [x₁, x₂, v₁, v₂] (two
    stages)."""
    k, c, m = iso.stiffness, iso.damping, iso.total
    if iso.mount is Mount.VISCOUS:
        # m ẍ = −k(x − g) − c(ẋ − ġ) + f; on the floor k(x − g) + c(ẋ − ġ)
        A = np.array([[0.0, 1.0], [-k / m, -c / m]])
        B = np.array([[0.0, 0.0, 0.0], [1.0 / m, k / m, c / m]])
        return LiveSystem(A, B, np.array([k, c]), np.array([0.0, -k, -c]), 1)
    if iso.mount is Mount.RELAXATION:
        k1 = iso.stiffness_ratio * k
        if c == 0:  # no damper: the spring Nk hangs free and passes nothing
            A = np.array([[0.0, 1.0, 0.0], [-k / m, 0.0, 0.0], [0.0, 0.0, 0.0]])
            B = np.array([[0.0, 0.0, 0.0], [1.0 / m, k / m, 0.0], [0.0, 0.0, 0.0]])
            return LiveSystem(A, B, np.array([k, 0.0, 0.0]), np.array([0.0, -k, 0.0]), 1)
        # m ẍ = −k(x − g) − k₁(x − y) + f; c(ẏ − ġ) = k₁(x − y); on the floor k(x − g) + k₁(x − y)
        A = np.array([[0.0, 1.0, 0.0], [-(k + k1) / m, 0.0, k1 / m], [k1 / c, 0.0, -k1 / c]])
        B = np.array([[0.0, 0.0, 0.0], [1.0 / m, k / m, 0.0], [0.0, 0.0, 1.0]])
        return LiveSystem(A, B, np.array([k + k1, 0.0, -k1]), np.array([0.0, -k, 0.0]), 1)
    M, C, K = iso._two_stage()
    k2, c2 = K[1, 1] - K[0, 0], C[1, 1] - C[0, 0]
    Minv = np.linalg.inv(M)
    A = np.block([[np.zeros((2, 2)), np.eye(2)], [-Minv @ K, -Minv @ C]])
    inputs = np.array([[1.0, 0.0, 0.0], [0.0, k2, c2]])  # the force on each mass from f, g and ġ
    B = np.vstack([np.zeros((2, 3)), Minv @ inputs])
    return LiveSystem(A, B, np.array([0.0, k2, 0.0, c2]), np.array([0.0, -k2, -c2]), 2)


def _settled_stage(iso: Isolator, x: float, g: float) -> float:
    """Where the intermediate mass's springs balance, the machine at x and the floor at g."""
    K = iso._two_stage()[2]
    k2 = K[1, 1] + K[1, 0]
    return float((k2 * g - K[1, 0] * x) / K[1, 1])


def held(iso: Isolator, x: float, g: float = 0.0) -> np.ndarray:
    """The state with the machine held still at x over the floor at g, long enough for the rest to settle: the
    relaxation damper relaxed (its spring unstretched, y = x), the intermediate mass where its springs balance."""
    if iso.mount is Mount.VISCOUS:
        return np.array([x, 0.0])
    if iso.mount is Mount.RELAXATION:
        return np.array([x, 0.0, x])
    return np.array([x, _settled_stage(iso, x, g), 0.0, 0.0])


def carry(old: Isolator, new: Isolator, z: np.ndarray, g: float = 0.0) -> np.ndarray:
    """The state after a change from `old` to `new` while it runs, at a step's boundary (#478 item 4, as
    `Simulator.set_system`): the machine keeps its place; a heavier machine (or block) captures its extra at rest,
    keeping the momentum, and a lighter one drops a piece at its speed. A relaxation damper kept keeps its spring's
    stretch; one put in starts unstretched. An intermediate mass kept keeps its place, and its momentum as the
    machine's; one put in starts at rest where its springs balance."""
    lo, hi = live_system(old).velocity, live_system(new).velocity
    x, v = float(z[0]), float(z[lo])
    if new.total > old.total:
        v *= old.total / new.total
    if new.mount is Mount.VISCOUS:
        return np.array([x, v])
    if new.mount is Mount.RELAXATION:
        y = float(z[2]) if old.mount is Mount.RELAXATION else x
        return np.array([x, v, y])
    if old.mount is Mount.TWO_STAGE:
        x2, v2 = float(z[1]), float(z[hi + 1])
        m_old, m_new = old.stage_mass * old.total, new.stage_mass * new.total
        if m_new > m_old:
            v2 *= m_old / m_new
        return np.array([x, x2, v, v2])
    return np.array([x, _settled_stage(new, x, g), v, 0.0])
