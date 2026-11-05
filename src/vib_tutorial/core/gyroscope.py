"""A spinning top with its tip held in a cup: Euler's equations, Lagrange's top, and its closed forms.

The *Gyroscope* page. The `Top` is a disc of mass m and radius r on a light stem, its centre a height l up the
axis from the tip, which is held at the origin (it can turn there but not slide). About the tip its moments of
inertia are I₁ = m(r²/4 + l²) across the axis and I₃ = mr²/2 along it. In Euler angles (precession φ about the
vertical, tilt θ from it, spin ψ about the axis) Lagrange's function is

    L = ½I₁(θ'² + φ'² sin²θ) + ½I₃(ψ' + φ' cos θ)² − mgl cos θ,

and φ and ψ don't appear in it, so their momenta are conserved: p_ψ = I₃ω₃ (ω₃ = ψ' + φ' cos θ, the spin about
the axis) and p_φ = I₁φ' sin²θ + I₃ω₃ cos θ (the angular momentum about the vertical). With the energy, the tilt
alone is left: with u = cos θ, u'² = f(u) = (α − βu)(1 − u²) − (b − au)², a cubic (a = p_ψ/I₁, b = p_φ/I₁,
β = 2mgl/I₁, α = 2E'/I₁, E' the energy less the spin's ½I₃ω₃²). The axis nods between two of its roots, the
turning points (`Motion.turning`), in a nutation period of 4K(m)/√(β(u₃ − u₁)) (`Motion.nutation_period`), and
precesses at φ' = (b − au)/(1 − u²) as it goes, so the tip of the axis traces waves, cusps or loops
(`Motion.path`). Held at one tilt, it precesses steadily at either root of I₁ cos θ φ'² − I₃ω₃φ' + mgl = 0
(`steady_precession`); upright, it sleeps if I₃²ω₃² > 4I₁mgl (`Top.sleeping_spin`).

`TopSimulator` steps the whole rigid body, not the Euler angles (they are singular upright, where φ and ψ turn
about the same axis): its orientation as a unit quaternion and its angular momentum L in space. Each step is
split (Strang) into gravity's moment, held over half a step (L += ½h τ, τ = −mgl e₃ × ẑ: exact, since τ depends
on the orientation alone), the free top's motion over a whole step (exact: a symmetric body turns about L at
|L|/I₁ and about its own axis at (1/I₃ − 1/I₁)L₃), and gravity's half step again. This is symplectic and second
order; L·ẑ and L·e₃ are conserved to round-off (the moment is across both), and the energy stays within an error
that falls as h². `euler_reference` is the Euler-angle equations by `solve_ivp`, for the tests.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np
import scipy.integrate
import scipy.special

G = 9.81
RPM = 2.0 * math.pi / 60.0  # rad/s per rpm
PER_PERIOD = 200  # steps per period of the fastest motion the steps sample (the axis turning about L)
MAX_STEP = 0.01  # s
MAX_STEPS_PER_ADVANCE = 200_000
STEADY = 1e-9  # turning points closer than this (in cos θ) are steady precession
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(64)


@dataclass(frozen=True)
class Top:
    """A disc on a light stem, its tip held at the origin."""

    mass: float = 0.5  # kg
    radius: float = 0.04  # m, the disc's
    height: float = 0.03  # m, from the tip to the disc's centre (l)
    gravity: float = G

    @property
    def polar(self) -> float:
        """I₃, about the axis."""
        return 0.5 * self.mass * self.radius**2

    @property
    def transverse(self) -> float:
        """I₁, about a line across the axis through the tip (parallel axes)."""
        return self.mass * (0.25 * self.radius**2 + self.height**2)

    @property
    def weight_moment(self) -> float:
        """mgl: gravity's moment on the top lying flat."""
        return self.mass * self.gravity * self.height

    @property
    def sleeping_spin(self) -> float:
        """The least spin (rad/s) that keeps the top upright: I₃ω₃ = 2√(I₁ mgl)."""
        return 2.0 * math.sqrt(self.transverse * self.weight_moment) / self.polar


def steady_precession(top: Top, spin: float, tilt: float) -> tuple[float, float]:
    """The slow and the fast steady precession rates (rad/s) at tilt θ: the roots of
    I₁ cos θ φ'² − I₃ω₃ φ' + mgl = 0, the smaller in size first. NaN where there are none (spinning too slowly);
    the fast one is NaN lying flat (cos θ = 0), where only the slow mgl/I₃ω₃ is left."""
    a2 = top.transverse * math.cos(tilt)
    p = top.polar * spin
    w = top.weight_moment
    if abs(a2) < 1e-12 * top.transverse:
        return (w / p if p else math.nan), math.nan
    disc = p * p - 4.0 * a2 * w
    if disc < -1e-12 * p * p:  # round-off at the least spin is a double root
        return math.nan, math.nan
    root = math.sqrt(max(disc, 0.0))
    one, two = (p - math.copysign(root, p)) / (2 * a2), (p + math.copysign(root, p)) / (2 * a2)
    return (one, two) if abs(one) <= abs(two) else (two, one)


def least_spin(top: Top, tilt: float) -> float:
    """The least spin (rad/s) at which the top can precess steadily at tilt θ: I₃ω₃ = 2√(I₁ mgl cos θ). Upright
    it is the sleeping top's; below the horizontal (cos θ < 0) any spin will do."""
    c = math.cos(tilt)
    return 2.0 * math.sqrt(top.transverse * top.weight_moment * c) / top.polar if c > 1e-12 else 0.0


class Launch(enum.Enum):
    REST = "At rest (φ' = 0)"
    SLOW = "Slow steady precession"
    FAST = "Fast steady precession"
    SET = "Set φ' below"


LAUNCHES = tuple(Launch)


class Path(enum.Enum):
    STEADY = "a circle: steady precession"
    WAVES = "waves: it precesses one way all the time"
    CUSPS = "cusps: it stops at the top of each nod"
    LOOPS = "loops: it runs backward at one turning point"


@dataclass(frozen=True)
class Motion:
    """The top let go at tilt θ₀ (rad) with spin ω₃ (rad/s) about its axis and precession rate φ'₀ (rad/s), its
    tilt not changing (θ'₀ = 0), so θ₀ is one of the turning points."""

    top: Top
    spin: float
    tilt: float
    precession: float = 0.0

    @classmethod
    def launched(cls, top: Top, spin: float, tilt: float, launch: Launch, rate: float = 0.0) -> Motion:
        """Let go as `launch` says: at rest, at either steady rate (at rest where there's none), or at `rate`."""
        if launch is Launch.SET:
            return cls(top, spin, tilt, rate)
        if launch is Launch.REST:
            return cls(top, spin, tilt, 0.0)
        slow, fast = steady_precession(top, spin, tilt)
        chosen = slow if launch is Launch.SLOW else fast
        return cls(top, spin, tilt, chosen if math.isfinite(chosen) else 0.0)

    # ------------------------------------------------------------- what's conserved
    @property
    def spin_momentum(self) -> float:
        """p_ψ = I₃ω₃."""
        return self.top.polar * self.spin

    @property
    def vertical_momentum(self) -> float:
        """p_φ = I₁φ' sin²θ + I₃ω₃ cos θ, the angular momentum about the vertical."""
        s, c = math.sin(self.tilt), math.cos(self.tilt)
        return self.top.transverse * self.precession * s * s + self.spin_momentum * c

    @property
    def energy(self) -> float:
        """E' (J): the energy less the spin's ½I₃ω₃² (constant too)."""
        s = math.sin(self.tilt)
        return 0.5 * self.top.transverse * (self.precession * s) ** 2 + self.top.weight_moment * math.cos(self.tilt)

    @property
    def total_energy(self) -> float:
        return self.energy + 0.5 * self.top.polar * self.spin**2

    def angular_momentum(self) -> np.ndarray:
        """L (space axes, z up) at the start: I₁φ' ẑ + (I₃ω₃ − I₁φ' cos θ) e₃, with the axis e₃ leaning toward −y."""
        s, c = math.sin(self.tilt), math.cos(self.tilt)
        i1, p = self.top.transverse, self.precession
        along = self.spin_momentum - i1 * p * c
        return np.array([0.0, -along * s, i1 * p + along * c])

    # ------------------------------------------------------------- the tilt alone
    @property
    def _abab(self) -> tuple[float, float, float, float]:
        i1 = self.top.transverse
        return (
            self.spin_momentum / i1,
            self.vertical_momentum / i1,
            2 * self.energy / i1,
            2 * self.top.weight_moment / i1,
        )

    def cubic(self) -> tuple[float, float, float, float]:
        """f(u)'s coefficients, highest first: u'² = f(u) = (α − βu)(1 − u²) − (b − au)²."""
        a, b, alpha, beta = self._abab
        return beta, -alpha - a * a, -beta + 2 * a * b, alpha - b * b

    def f(self, u):
        c3, c2, c1, c0 = self.cubic()
        return ((c3 * u + c2) * u + c1) * u + c0

    def roots(self) -> tuple[float, float, float]:
        """f's roots, smallest first: the turning points u₁ ≤ u₂ (cos θ) and the third, u₃ ≥ 1 (inf without
        gravity). cos θ₀ is one of the first two, so it is divided out and the quadratic left solved exactly."""
        c3, c2, c1, _ = self.cubic()
        u0 = math.cos(self.tilt)
        q2, q1 = c3, c2 + u0 * c3
        q0 = c1 + u0 * q1
        if q2 == 0.0:  # no gravity: f is a quadratic
            other = -q0 / q1
            lo, hi = sorted((u0, other))
            return lo, hi, math.inf
        disc = q1 * q1 - 4 * q2 * q0
        root = math.sqrt(max(disc, 0.0))
        r1, r2 = sorted(((-q1 - root) / (2 * q2), (-q1 + root) / (2 * q2)))
        lo, mid, hi = sorted((u0, r1, r2))
        return lo, mid, hi

    def turning(self) -> tuple[float, float]:
        """The tilts (rad) the axis nods between, the uppermost first."""
        u1, u2, _ = self.roots()
        return math.acos(min(1.0, u2)), math.acos(max(-1.0, u1))

    @property
    def steady(self) -> bool:
        u1, u2, _ = self.roots()
        return u2 - u1 < STEADY

    def nutation_period(self) -> float:
        """The time (s) for one nod and back: 4K(m)/√(β(u₃ − u₁)), m = (u₂ − u₁)/(u₃ − u₁). Steady, it is the
        small nod's, 2π/√(β(u₃ − u₂)); without gravity 2π/√(α + a²)."""
        u1, u2, u3 = self.roots()
        c3, c2, _, _ = self.cubic()
        if not math.isfinite(u3):
            return 2.0 * math.pi / math.sqrt(-c2)
        return 4.0 * scipy.special.ellipk((u2 - u1) / (u3 - u1)) / math.sqrt(c3 * (u3 - u1))

    def nutation_hz(self) -> float:
        return 1.0 / self.nutation_period()

    def precession_rate(self, u):
        """φ' = (b − a u)/(1 − u²) at cos θ = u."""
        a, b, _, _ = self._abab
        return (b - a * u) / (1.0 - u * u)

    def mean_precession(self) -> float:
        """The precession's mean rate (rad/s) over a nod: its advance over the nod, 2∫φ' du/√f, over the period."""
        u1, u2, u3 = self.roots()
        if u2 - u1 < STEADY:
            return float(self.precession_rate(0.5 * (u1 + u2)))
        c3, c2, _, _ = self.cubic()
        chi = 0.25 * math.pi * (_NODES + 1.0)  # Gauss–Legendre on [0, π/2]
        u = u1 + (u2 - u1) * np.sin(chi) ** 2
        speed = np.sqrt(c3 * (u3 - u)) if math.isfinite(u3) else math.sqrt(-c2) * np.ones_like(u)
        advance = 2.0 * float(np.sum(_WEIGHTS * 0.25 * math.pi * 2.0 * self.precession_rate(u) / speed))
        return advance / self.nutation_period()

    def path(self) -> Path:
        """What the tip of the axis traces: from φ' at the two turning points."""
        u1, u2, _ = self.roots()
        if u2 - u1 < STEADY:
            return Path.STEADY
        a, b, _, _ = self._abab
        lo, hi = b - a * u1, b - a * u2
        scale = max(abs(a), abs(b), 1e-300)
        if min(abs(lo), abs(hi)) < 1e-9 * scale:
            return Path.CUSPS
        return Path.LOOPS if (lo > 0) != (hi > 0) else Path.WAVES

    def effective_potential(self, theta):
        """U(θ) = ½I₁(b − a cos θ)²/sin²θ + mgl cos θ (J): the tilt moves in it with energy E'."""
        a, b, _, _ = self._abab
        c, s = np.cos(theta), np.sin(theta)
        return 0.5 * self.top.transverse * (b - a * c) ** 2 / (s * s) + self.top.weight_moment * c


# ------------------------------------------------------------------ the simulation
@dataclass
class TopSamples:
    """Each step's end: time, the axis e₃ and the disc's mark (the body's x axis) in space, and the energy (J)."""

    t: np.ndarray
    axis: np.ndarray  # (n, 3)
    mark: np.ndarray  # (n, 3)
    energy: np.ndarray
    momentum: np.ndarray  # (n, 3): L in space


class TopSimulator:
    """The whole rigid body, stepped by the split above: orientation q (a unit quaternion, body to space) and L."""

    def __init__(self, motion: Motion, per_period: int = PER_PERIOD) -> None:
        self.motion = motion
        top = motion.top
        self.i1, self.i3, self.w = top.transverse, top.polar, top.weight_moment
        half = 0.5 * motion.tilt  # a turn by θ about x: the axis leans toward −y
        self.q = [math.cos(half), math.sin(half), 0.0, 0.0]
        self.L = [float(v) for v in motion.angular_momentum()]
        self.t = 0.0
        rate = max(math.sqrt(sum(v * v for v in self.L)) / self.i1, math.sqrt(abs(self.w) / self.i1), 1e-3)
        self.h = min(2.0 * math.pi / rate / per_period, MAX_STEP)

    def axis(self) -> tuple[float, float, float]:
        w, x, y, z = self.q
        return 2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)

    def energy(self) -> float:
        lx, ly, lz = self.L
        ex, ey, ez = self.axis()
        along = lx * ex + ly * ey + lz * ez
        return (
            0.5 * (lx * lx + ly * ly + lz * lz) / self.i1 + 0.5 * (1 / self.i3 - 1 / self.i1) * along**2 + self.w * ez
        )

    def advance(self, duration: float) -> TopSamples:
        """Step on by `duration` (to the nearest whole step, at most MAX_STEPS_PER_ADVANCE)."""
        n = min(round(duration / self.h), MAX_STEPS_PER_ADVANCE)
        out = np.empty((n, 11))
        h, i1, i3, wgt = self.h, self.i1, self.i3, self.w
        kick = 0.5 * h * wgt
        spin_gain = (1.0 / i3 - 1.0 / i1) * h
        w, x, y, z = self.q
        lx, ly, lz = self.L
        t = self.t
        for k in range(n):
            # gravity, half a step: τ = mgl(−e₃y, e₃x, 0)
            ex, ey = 2 * (x * z + w * y), 2 * (y * z - w * x)
            lx -= kick * ey
            ly += kick * ex
            # the free top, a whole step: about its own axis (body z) by (1/I₃ − 1/I₁)L₃h ...
            ez = 1 - 2 * (x * x + y * y)
            along = lx * ex + ly * ey + lz * ez
            c, s = math.cos(0.5 * spin_gain * along), math.sin(0.5 * spin_gain * along)
            w, x, y, z = w * c - z * s, x * c + y * s, y * c - x * s, z * c + w * s
            # ... then about L, in space, by |L|h/I₁
            size = math.sqrt(lx * lx + ly * ly + lz * lz)
            if size > 0:
                c, s = math.cos(0.5 * size * h / i1), math.sin(0.5 * size * h / i1) / size
                ax, ay, az = lx * s, ly * s, lz * s
                w, x, y, z = (
                    c * w - ax * x - ay * y - az * z,
                    c * x + ax * w + ay * z - az * y,
                    c * y - ax * z + ay * w + az * x,
                    c * z + ax * y - ay * x + az * w,
                )
            # gravity's other half
            ex, ey = 2 * (x * z + w * y), 2 * (y * z - w * x)
            lx -= kick * ey
            ly += kick * ex
            t += h
            ez = 1 - 2 * (x * x + y * y)
            along = lx * ex + ly * ey + lz * ez
            out[k] = (
                t,
                ex,
                ey,
                ez,
                1 - 2 * (y * y + z * z),
                2 * (x * y + w * z),
                2 * (x * z - w * y),
                0.5 * (lx * lx + ly * ly + lz * lz) / i1 + 0.5 * (1 / i3 - 1 / i1) * along**2 + wgt * ez,
                lx,
                ly,
                lz,
            )
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        self.q = [w / norm, x / norm, y / norm, z / norm]
        self.L = [lx, ly, lz]
        self.t = t
        return TopSamples(out[:, 0], out[:, 1:4], out[:, 4:7], out[:, 7], out[:, 8:11])


def tilt_and_azimuth(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """θ from the vertical and φ (rad, unwrapped) of axes (n, 3), φ = 0 leaning toward −y as the launch does."""
    theta = np.arccos(np.clip(axis[:, 2], -1.0, 1.0))
    phi = np.unwrap(np.arctan2(axis[:, 0], -axis[:, 1]))
    return theta, phi


def euler_reference(motion: Motion, t: np.ndarray, rtol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """θ(t) and φ(t) from the Euler-angle equations by `solve_ivp`: I₁θ'' = I₁φ'² sin θ cos θ − I₃ω₃φ' sin θ +
    mgl sin θ, φ' = (b − a cos θ)/sin²θ. Singular where sin θ = 0 (the top upright): the tests keep away."""
    a, b, _, _ = motion._abab
    g = motion.top.weight_moment / motion.top.transverse

    def rates(_, y):
        th, dth, _ = y
        s, c = math.sin(th), math.cos(th)
        dphi = (b - a * c) / (s * s)
        return [dth, dphi * dphi * s * c - a * dphi * s + g * s, dphi]

    sol = scipy.integrate.solve_ivp(
        rates, (0.0, float(t[-1])), [motion.tilt, 0.0, 0.0], t_eval=t, rtol=rtol, atol=1e-12, method="DOP853"
    )
    return sol.y[0], sol.y[2]
