"""A double pendulum: Lagrange's equations whole, linearized, and against the free-body diagram's.

The *Deriving the equations* page's second system. Two bobs, m₁ on a light rod L₁ from a fixed pivot and m₂ on
a light rod L₂ from m₁, swing in a vertical plane; θ₁ and θ₂ are the rods' angles from the vertical
(`Angles.ABSOLUTE`), or θ₁ and φ = θ₂ − θ₁, the elbow's bend (`Angles.RELATIVE`), (θ₁, θ₂) = T q. With
Δ = θ₁ − θ₂,

    T = ½(m₁ + m₂)L₁²θ̇₁² + m₂L₁L₂ cos Δ θ̇₁θ̇₂ + ½m₂L₂²θ̇₂²,   V = −(m₁ + m₂)gL₁ cos θ₁ − m₂gL₂ cos θ₂,

and Lagrange's equations are

    (m₁ + m₂)L₁²θ̈₁ + m₂L₁L₂(cos Δ θ̈₂ + sin Δ θ̇₂²) + (m₁ + m₂)gL₁ sin θ₁ = 0,
    m₂L₂²θ̈₂ + m₂L₁L₂(cos Δ θ̈₁ − sin Δ θ̇₁²) + m₂gL₂ sin θ₂ = 0

(`accelerations` solves them for θ̈). Small swings make them M q̈ + K q = 0 with M the Hessian of T at rest and K
of V (`matrices`): M = [[(m₁ + m₂)L₁², m₂L₁L₂], [·, m₂L₂²]], K = diag((m₁ + m₂)gL₁, m₂gL₂) in the absolute angles.

The free-body diagram takes each bob along its rod's normal: bob 2 needs no tension there, but bob 1 does (rod
2's, found from bob 2 along its rod). Linearized, its rows (`newton`) are m₁L₁θ̈₁ + (m₁ + m₂)gθ₁ − m₂gθ₂ = 0 and
m₂L₁θ̈₁ + m₂L₂θ̈₂ + m₂gθ₂ = 0: right, but not symmetric. L₁ times their sum (both bobs' moments about the pivot)
and L₂ times the second (bob 2's about the elbow) are Lagrange's (`MIX`). Treating bob 2 as a pendulum on a
fixed elbow drops its m₂L₁θ̈₁ (`slipped`).

`swing` releases it from rest at (θ₁, θ₂): the whole equations by `solve_ivp` (DOP853, tight tolerances: the
energy holds to ~1e-9 of the swing's), beside the linear ones stepped by expm(AΔ), exactly. `divergence` runs
a twin released `TWIN` rad further over and reports how far apart they drift: slowly for small swings,
exponentially once the motion is chaotic.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np
import scipy.integrate
import scipy.linalg

TWO_PI = 2.0 * math.pi
G = 9.81
TWIN = 1e-6  # rad: how much further over the twin starts
RTOL, ATOL = 1e-11, 1e-12  # the whole equations' tolerances


class Angles(enum.Enum):
    ABSOLUTE = "From the vertical: θ1, θ2"
    RELATIVE = "The elbow's bend: θ1, φ = θ2 − θ1"


@dataclass(frozen=True)
class DoublePendulum:
    upper: float = 1.0  # kg, m₁
    lower: float = 1.0  # kg, m₂
    upper_length: float = 0.5  # m, L₁
    lower_length: float = 0.5  # m, L₂
    gravity: float = G

    def positions(self, th1, th2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(x₁, y₁, x₂, y₂) from the pivot, y up."""
        th1, th2 = np.asarray(th1, float), np.asarray(th2, float)
        x1, y1 = self.upper_length * np.sin(th1), -self.upper_length * np.cos(th1)
        return x1, y1, x1 + self.lower_length * np.sin(th2), y1 - self.lower_length * np.cos(th2)

    def energies(self, th1, th2, w1, w2) -> tuple[np.ndarray, np.ndarray]:
        """(T, V) of the whole motion, V from the pivot's height."""
        m1, m2, L1, L2, g = self.upper, self.lower, self.upper_length, self.lower_length, self.gravity
        d = np.asarray(th1) - np.asarray(th2)
        kinetic = 0.5 * (m1 + m2) * L1**2 * w1**2 + m2 * L1 * L2 * np.cos(d) * w1 * w2 + 0.5 * m2 * L2**2 * w2**2
        potential = -(m1 + m2) * g * L1 * np.cos(th1) - m2 * g * L2 * np.cos(th2)
        return kinetic, potential

    def separate_frequencies(self) -> tuple[float, float]:
        """Each rod as a simple pendulum, √(g/L)/2π (Hz)."""
        g = self.gravity
        return math.sqrt(g / self.upper_length) / TWO_PI, math.sqrt(g / self.lower_length) / TWO_PI


def transform(angles: Angles) -> np.ndarray:
    """T: (θ₁, θ₂) = T q."""
    return np.eye(2) if angles is Angles.ABSOLUTE else np.array([[1.0, 0.0], [1.0, 1.0]])


@dataclass(frozen=True)
class Linear:
    M: np.ndarray
    K: np.ndarray


def matrices(p: DoublePendulum, angles: Angles = Angles.ABSOLUTE) -> Linear:
    """Lagrange's M and K for small swings, in the angles."""
    m1, m2, L1, L2, g = p.upper, p.lower, p.upper_length, p.lower_length, p.gravity
    M = np.array([[(m1 + m2) * L1 * L1, m2 * L1 * L2], [m2 * L1 * L2, m2 * L2 * L2]])
    K = np.diag([(m1 + m2) * g * L1, m2 * g * L2])
    T = transform(angles)
    return Linear(T.T @ M @ T, T.T @ K @ T)


def newton(p: DoublePendulum, angles: Angles = Angles.ABSOLUTE) -> Linear:
    """The free-body diagram's rows, linearized: bob 1 and bob 2 each along its rod's normal (per metre)."""
    m1, m2, L1, L2, g = p.upper, p.lower, p.upper_length, p.lower_length, p.gravity
    M = np.array([[m1 * L1, 0.0], [m2 * L1, m2 * L2]])
    K = np.array([[(m1 + m2) * g, -m2 * g], [0.0, m2 * g]])
    T = transform(angles)
    return Linear(M @ T, K @ T)


def mix(p: DoublePendulum, angles: Angles = Angles.ABSOLUTE) -> np.ndarray:
    """R: R times `newton`'s rows is `matrices`: L₁ × (row 1 + row 2), L₂ × row 2, then Tᵀ."""
    R = np.array([[p.upper_length, p.upper_length], [0.0, p.lower_length]])
    return transform(angles).T @ R


def slipped(p: DoublePendulum) -> Linear:
    """Bob 2 taken as a pendulum on a fixed elbow (its m₂L₁θ̈₁ dropped), with Lagrange's first row."""
    m = matrices(p)
    M = m.M.copy()
    M[1, 0] = 0.0
    return Linear(M, m.K)


@dataclass(frozen=True)
class Mode:
    frequency: float  # Hz
    shape: np.ndarray  # in the angles asked for
    ratio: float  # θ₂/θ₁ (absolute), whatever the angles

    @property
    def in_phase(self) -> bool:
        return self.ratio > 0


def modes(p: DoublePendulum, angles: Angles = Angles.ABSOLUTE) -> tuple[Mode, Mode]:
    """The two small-swing modes, slower first, each scaled to θ₁ = 1."""
    m = matrices(p, angles)
    w2, phi = scipy.linalg.eigh(m.K, m.M)
    T = transform(angles)
    out = []
    for i in range(2):
        th = T @ phi[:, i]
        out.append(Mode(math.sqrt(w2[i]) / TWO_PI, phi[:, i] / th[0], th[1] / th[0]))
    return out[0], out[1]


def sweep(p: DoublePendulum, ratios: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies, θ₂/θ₁), each (len(ratios), 2): the modes with the lower bob at ratio × the upper."""
    f, shape = np.empty((ratios.size, 2)), np.empty((ratios.size, 2))
    for i, ratio in enumerate(ratios):
        ms = modes(DoublePendulum(p.upper, ratio * p.upper, p.upper_length, p.lower_length, p.gravity))
        f[i] = [m.frequency for m in ms]
        shape[i] = [m.ratio for m in ms]
    return f, shape


def accelerations(p: DoublePendulum, th1: float, th2: float, w1: float, w2: float) -> tuple[float, float]:
    """θ̈₁ and θ̈₂ from the whole equations."""
    m1, m2, L1, L2, g = p.upper, p.lower, p.upper_length, p.lower_length, p.gravity
    d = th1 - th2
    c, s = math.cos(d), math.sin(d)
    a11, a12, a22 = (m1 + m2) * L1 * L1, m2 * L1 * L2 * c, m2 * L2 * L2
    b1 = -m2 * L1 * L2 * s * w2 * w2 - (m1 + m2) * g * L1 * math.sin(th1)
    b2 = m2 * L1 * L2 * s * w1 * w1 - m2 * g * L2 * math.sin(th2)
    det = a11 * a22 - a12 * a12
    return (a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det


def _whole(p: DoublePendulum, start: tuple[float, float], t: np.ndarray) -> np.ndarray:
    """(n, 4): θ₁, θ₂, θ̇₁, θ̇₂ at t, from rest at `start`."""

    def rhs(_, z):
        return (z[2], z[3], *accelerations(p, *z))

    sol = scipy.integrate.solve_ivp(
        rhs, (0.0, float(t[-1])), [start[0], start[1], 0.0, 0.0], method="DOP853", t_eval=t, rtol=RTOL, atol=ATOL
    )
    return sol.y.T


@dataclass(frozen=True)
class Swing:
    t: np.ndarray
    whole: np.ndarray  # (n, 4): θ₁, θ₂, θ̇₁, θ̇₂
    linear: np.ndarray  # (n, 4)
    energy: np.ndarray  # T + V of the whole motion
    rest: float  # V hanging at rest

    @property
    def drift(self) -> float:
        """The energy's largest change, over the swing's own (from rest at the start to the bottom)."""
        e = self.energy
        return float(np.abs(e - e[0]).max() / max(abs(e[0] - self.rest), 1e-300))


def swing(p: DoublePendulum, start: tuple[float, float], duration: float = 20.0, fs: float = 100.0) -> Swing:
    """Released from rest at `start` (rad, absolute): the whole motion and the linear one."""
    t = np.arange(round(duration * fs) + 1) / fs
    whole = _whole(p, start, t)
    m = matrices(p)
    A = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.linalg.solve(m.M, m.K), np.zeros((2, 2))]])
    Phi = scipy.linalg.expm(A / fs)
    linear = np.empty((t.size, 4))
    linear[0] = (start[0], start[1], 0.0, 0.0)
    for k in range(1, t.size):
        linear[k] = Phi @ linear[k - 1]
    kinetic, potential = p.energies(*whole.T)
    rest = float(sum(p.energies(0.0, 0.0, 0.0, 0.0)))
    return Swing(t, whole, linear, kinetic + potential, rest)


@dataclass(frozen=True)
class Divergence:
    t: np.ndarray
    apart: np.ndarray  # rad: |Δθ₁| + |Δθ₂| between the twins, each within ±π

    def doubling_time(self, lo: float = 1e-5, hi: float = 1e-2) -> float:
        """The separation's doubling time, fitted where it grows from lo to hi (inf if it never gets there)."""
        if not np.any(self.apart >= hi):
            return math.inf
        first = int(np.argmax(self.apart >= hi))
        inside = np.flatnonzero(self.apart[:first] > lo)
        if inside.size < 10:
            return math.inf
        slope = np.polyfit(self.t[inside], np.log(self.apart[inside]), 1)[0]
        return math.log(2.0) / slope if slope > 0 else math.inf


def divergence(p: DoublePendulum, start: tuple[float, float], duration: float = 20.0, fs: float = 50.0) -> Divergence:
    """Two releases TWIN rad apart (the lower rod's start), and how far apart they swing."""
    t = np.arange(round(duration * fs) + 1) / fs
    a = _whole(p, start, t)
    b = _whole(p, (start[0], start[1] + TWIN), t)
    apart = np.abs(np.angle(np.exp(1j * (a[:, :2] - b[:, :2]))))  # each angle's difference, within ±π
    return Divergence(t, apart.sum(axis=1))
