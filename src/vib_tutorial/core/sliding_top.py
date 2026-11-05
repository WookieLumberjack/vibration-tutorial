"""A top on a table: a round tip that slides with friction, the rising top and the tippe top.

The *Gyroscope* page's *On a table* settings. A `RoundTop` is a symmetric body of mass m (moments I₃ along its
axis e₃ and I₁ across it, both about its centre of mass G) whose underside is a sphere of radius R centred on
the axis at C = G − a e₃: a > 0 puts G above the sphere's centre (a disc on a stem with a round tip, a = l − R),
a < 0 below it (a tippe top, a ball weighted low). It touches the table at P = C − R ẑ, so G stands at
z = R + a cos θ, and P slips at v_P = v_G + ω × (P − G) (it is never off the table's plane). The table pushes
up with N and drags with Coulomb's friction, F = −μN v_P/|v_P|, smoothed below `SMOOTHING` so that it is a
function of the state: then

    m v_G' = N ẑ + F − mg ẑ,     dL/dt = (P − G) × (N ẑ + F),     L = I₁ω + (I₃ − I₁)ω₃e₃,

with N whatever keeps G at its height: z_G'' = a (e₃·ẑ)'' is linear in N, so N comes out in closed form at
each state. With the energy E = ½m|v_G|² + ½ω·L + mg z_G, friction's power is F·v_P ≤ 0, and

    J = L·(a e₃ + R ẑ) = −L·(P − G)

is conserved whatever the contact force (Jellett's constant: its moment is about P, and the rate of a e₃ + R ẑ
is across L). So the energy falls at a fixed J, and the least energy the top can have at a tilt θ with that J
is (`least_energy`, L taken along I(a e₃ + R ẑ), nothing else moving)

    E_J(θ) = J²/2D(θ) + mg(R + a cos θ),   D(θ) = I₁(a² + R² + 2aR cos θ) + (I₃ − I₁)(a + R cos θ)²,

the curve the top's energy sinks onto: upright E_J(0) = J²/2I₃(a + R)² + mg(R + a), turned over
E_J(π) = J²/2I₃(R − a)² + mg(R − a). Upright is the curve's minimum while J² D'(1) > 2mga D(1)² (D as a
function of cos θ), so a disc on a round tip rises to sleep once it spins faster than `turning_spin`, and a
tippe top turns over once it spins faster than it, if 1 − α < I₁/I₃ < 1 + α (α = −a/R): then turned over is
a minimum and upright isn't.

`SlidingSimulator` steps the state (orientation as a unit quaternion, ω and v_G in space, G's horizontal
position, and the work friction has done) by the classic fourth-order Runge–Kutta step, the quaternion
renormalized after each. Unlike the cup's, nothing here can be split off exactly, the friction dissipates,
and the step must resolve the spin, the free wobble about G and friction's own time μN/(ε × the contact's
mass) where it is smoothed; `RECORD_DT` thins what it keeps. `reference` is the same equations by `solve_ivp`,
for the tests.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np
import scipy.integrate
import scipy.optimize

from .gyroscope import G, Top

SMOOTHING = 0.005  # m/s: friction is smoothed below this slip, μN v/√(v² + ε²)
BALL = 0.4  # a tippe top's I₃ over mR²: a uniform ball's
PER_PERIOD = 64  # steps a period of the fastest turning (the spin, or the free wobble about G)
FRICTION_STEP = 1.0  # the step over the smoothed friction's time constant (RK4 is stable to 2.78)
MAX_STEP = 1e-3  # s
RECORD_DT = 1e-3  # s between the samples kept
MAX_STEPS_PER_ADVANCE = 200_000
COLUMNS = 22  # a sample's: see `_row`


class Event(enum.Enum):
    HOPS = "it leaves the table, which this model can't follow: its tip stays on the table"
    RIM = "its disc's rim touches the table: it falls over"


@dataclass(frozen=True)
class RoundTop:
    """A symmetric body on a table, its underside a sphere of radius R whose centre is a below G on the axis."""

    mass: float
    polar: float  # I₃, about the axis
    central: float  # I₁, across the axis through G
    offset: float  # a (m): G above the sphere's centre; negative below
    radius: float  # R (m), the sphere's
    friction: float = 0.3  # μ
    gravity: float = G
    rim: float = 0.0  # a disc's radius, to see when it touches the table (0: a ball)

    @classmethod
    def disc(cls, top: Top, tip_radius: float, friction: float) -> RoundTop:
        """The cup's disc on a stem, its tip rounded to radius R (its end still l below the disc's centre)."""
        m, r = top.mass, top.radius
        return cls(m, 0.5 * m * r * r, 0.25 * m * r * r, top.height - tip_radius, tip_radius, friction, top.gravity, r)

    @classmethod
    def tippe(
        cls, mass: float, radius: float, low: float, ratio: float, friction: float, gravity: float = G
    ) -> RoundTop:
        """A tippe top: a ball of radius R, its weight `low` below its centre, I₃ = 0.4mR² and I₁ = ratio × I₃
        (its stem left out)."""
        polar = BALL * mass * radius * radius
        return cls(mass, polar, ratio * polar, -low, radius, friction, gravity)

    @property
    def weight(self) -> float:
        return self.mass * self.gravity

    @property
    def alpha(self) -> float:
        """α = −a/R: how far below its centre a ball's weight sits, over its radius."""
        return -self.offset / self.radius

    @property
    def ratio(self) -> float:
        """I₁/I₃."""
        return self.central / self.polar

    @property
    def inverts(self) -> bool:
        """A ball turns over (spun fast enough) if 1 − α < I₁/I₃ < 1 + α."""
        return self.offset < 0 and 1 - self.alpha < self.ratio < 1 + self.alpha

    def height(self, tilt: float) -> float:
        """G's height above the table (m)."""
        return self.radius + self.offset * math.cos(tilt)

    def rim_tilt(self) -> float:
        """The tilt (rad) at which a disc's rim touches the table: R + a cos θ = r sin θ (π for a ball)."""
        if self.rim <= 0:
            return math.pi
        a, R, r = self.offset, self.radius, self.rim
        # r sin θ − a cos θ = R: √(r² + a²) sin(θ − δ) = R, δ = atan2(a, r)
        return math.asin(min(1.0, R / math.hypot(r, a))) + math.atan2(a, r)

    def jellett(self, spin: float, tilt: float) -> float:
        """J (kg·m²/s) let go at tilt θ spinning at ω₃ about the axis and nothing else: I₃ω₃(a + R cos θ)."""
        return self.polar * spin * (self.offset + self.radius * math.cos(tilt))

    def least_energy(self, J: float, tilt):
        """E_J(θ) (J): the least energy at tilt θ with Jellett's J."""
        a, R, i1, i3 = self.offset, self.radius, self.central, self.polar
        c = np.cos(tilt)
        D = i1 * (a * a + R * R + 2 * a * R * c) + (i3 - i1) * (a + R * c) ** 2
        return J * J / (2 * D) + self.weight * (R + a * c)

    def end_spin(self, J: float, upright: bool = True) -> float:
        """The spin (rad/s, about the vertical, seen from above) it would have with Jellett's J ending upright
        (J/I₃(R + a)) or turned over (J/I₃(R − a)); its spin about its own axis is the opposite turned over."""
        arm = self.radius + (self.offset if upright else -self.offset)
        return J / (self.polar * arm) if arm != 0 else math.nan

    def turning_spin(self) -> float:
        """The spin (rad/s, upright) at which upright stops or starts being E_J's minimum: J²D'(1) = 2mgaD(1)²,
        Ω² = mga(a + R)²/R(I₃(a + R) − I₁R). A disc rises to sleep above it; a tippe top turns over above it.
        NaN where there is none (upright is a minimum, or isn't, whatever the spin)."""
        a, R = self.offset, self.radius
        lever = R * (self.polar * (a + R) - self.central * R)
        if lever == 0 or self.weight * a / lever <= 0:
            return math.nan
        return math.sqrt(self.weight * a * (a + R) ** 2 / lever)

    def least_tilt(self, J: float) -> float:
        """The tilt (rad) where E_J is least, between upright and the rim's tilt (or turned over): where friction
        takes the top, if it keeps sliding."""
        hi = self.rim_tilt()
        theta = np.linspace(0.0, hi, 721)
        i = int(np.argmin(self.least_energy(J, theta)))
        if 0 < i < theta.size - 1:
            found = scipy.optimize.minimize_scalar(
                lambda th: float(self.least_energy(J, th)), bounds=(theta[i - 1], theta[i + 1]), method="bounded"
            )
            return float(found.x)
        return float(theta[i])

    def rolling_tilt(self) -> float:
        """A disc's tilt (rad) below which friction can make its tip roll instead of slide, for a fast top:
        rolling, G circles at ω₃R sin θ and the slow precession mga/I₃ω₃, which takes a force μmg at most, so
        sin θ = μI₃/maR (spin drops out). It rises while it slides, so it stalls about here. π/2 if it can roll at
        any tilt; NaN for a ball weighted low."""
        if self.offset <= 0:
            return math.nan
        x = self.friction * self.polar / (self.mass * self.offset * self.radius)
        return math.asin(x) if x < 1 else 0.5 * math.pi

    def contact_rate(self) -> float:
        """Smoothed friction's rate (1/s): μg over ε, times the contact's mass ratio 1 + m(R + |a|)²/I₁."""
        return (
            self.friction
            * self.gravity
            * (1 + self.mass * (self.radius + abs(self.offset)) ** 2 / self.central)
            / SMOOTHING
        )


def launch_state(top: RoundTop, spin: float, tilt: float) -> list[float]:
    """Let go at rest at tilt θ, leaning toward −y, spinning at ω₃ about its axis; G over the origin."""
    s, c = math.sin(tilt), math.cos(tilt)
    return [math.cos(0.5 * tilt), math.sin(0.5 * tilt), 0.0, 0.0, 0.0, -spin * s, spin * c, 0.0, 0.0, 0.0, 0.0, 0.0]


def _field(top: RoundTop):
    """The state's rate, with N and the friction (Fx, Fy). State: q (w, x, y, z), ω, v_G (x, y), G (x, y), work."""
    m, i1, i3, a, R, mu, g = (
        top.mass,
        top.central,
        top.polar,
        top.offset,
        top.radius,
        top.friction,
        top.gravity,
    )
    eps2 = SMOOTHING * SMOOTHING
    k = (i3 - i1) / i1
    j = (i3 - i1) / i3

    def rates(s):
        w, x, y, z, ox, oy, oz, vx, vy, _, _, _ = s
        ex, ey, ez = 2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)
        rx, ry, rz = -a * ex, -a * ey, -a * ez - R  # P − G
        dex, dey, dez = oy * ez - oz * ey, oz * ex - ox * ez, ox * ey - oy * ex  # e₃' = ω × e₃
        sx, sy = vx + oy * rz - oz * ry, vy + oz * rx - ox * rz  # the slip
        size = math.sqrt(sx * sx + sy * sy + eps2)
        fx, fy = -mu * sx / size, -mu * sy / size  # the contact force per N: (fx, fy, 1)
        tx, ty, tz = ry - rz * fy, rz * fx - rx, rx * fy - ry * fx  # its moment about G, per N
        w3 = ox * ex + oy * ey + oz * ez
        t3 = j * (tx * ex + ty * ey + tz * ez)
        # ω' = N A + B: I₁ω' = τ − (I₃ − I₁)(ω₃' e₃ + ω₃ e₃'), I₃ω₃' = τ·e₃
        Ax, Ay, Az = (tx - t3 * ex) / i1, (ty - t3 * ey) / i1, (tz - t3 * ez) / i1
        Bx, By, Bz = -k * w3 * dex, -k * w3 * dey, -k * w3 * dez
        # m z_G'' = N − mg, z_G'' = a((ω' × e₃)·ẑ + (ω × e₃')·ẑ)
        N = m * (g + a * (Bx * ey - By * ex + ox * dey - oy * dex)) / (1 - m * a * (Ax * ey - Ay * ex))
        return (
            [
                0.5 * (-ox * x - oy * y - oz * z),
                0.5 * (ox * w + oy * z - oz * y),
                0.5 * (-ox * z + oy * w + oz * x),
                0.5 * (ox * y - oy * x + oz * w),
                N * Ax + Bx,
                N * Ay + By,
                N * Az + Bz,
                N * fx / m,
                N * fy / m,
                vx,
                vy,
                -N * (fx * sx + fy * sy),  # the work friction has taken out
            ],
            N,
            N * fx,
            N * fy,
        )

    return rates


def _row(top: RoundTop, t: float, s, N: float, Fx: float, Fy: float) -> list[float]:
    """A sample: t, e₃, the mark, G (x, y, z), P (x, y), E, the work, J, N, F (x, y), L, the slip's speed."""
    m, i1, i3, a, R = top.mass, top.central, top.polar, top.offset, top.radius
    w, x, y, z, ox, oy, oz, vx, vy, gx, gy, work = s
    ex, ey, ez = 2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)
    w3 = ox * ex + oy * ey + oz * ez
    Lx, Ly, Lz = i1 * ox + (i3 - i1) * w3 * ex, i1 * oy + (i3 - i1) * w3 * ey, i1 * oz + (i3 - i1) * w3 * ez
    dez = ox * ey - oy * ex
    energy = 0.5 * m * (vx * vx + vy * vy + (a * dez) ** 2) + 0.5 * (Lx * ox + Ly * oy + Lz * oz)
    energy += m * top.gravity * (R + a * ez)
    rx, ry, rz = -a * ex, -a * ey, -a * ez - R
    sx, sy = vx + oy * rz - oz * ry, vy + oz * rx - ox * rz
    return [
        t,
        ex,
        ey,
        ez,
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y),
        gx,
        gy,
        R + a * ez,
        gx + rx,
        gy + ry,
        energy,
        work,
        Lx * a * ex + Ly * a * ey + Lz * (a * ez + R),
        N,
        Fx,
        Fy,
        Lx,
        Ly,
        Lz,
        math.hypot(sx, sy),
    ]


@dataclass
class SlideSamples:
    """The samples kept: time, the axis e₃ and the mark, G, the contact P on the table, the energy and the work
    friction has taken out (J), Jellett's J, the normal force and the friction (N), L about G, and the slip."""

    t: np.ndarray
    axis: np.ndarray  # (n, 3)
    mark: np.ndarray  # (n, 3)
    centre: np.ndarray  # (n, 3): G
    contact: np.ndarray  # (n, 2): P on the table
    energy: np.ndarray
    work: np.ndarray
    jellett: np.ndarray
    normal: np.ndarray
    friction: np.ndarray  # (n, 2)
    momentum: np.ndarray  # (n, 3)
    slip: np.ndarray

    @classmethod
    def of(cls, rows: np.ndarray) -> SlideSamples:
        return cls(
            rows[:, 0],
            rows[:, 1:4],
            rows[:, 4:7],
            rows[:, 7:10],
            rows[:, 10:12],
            rows[:, 12],
            rows[:, 13],
            rows[:, 14],
            rows[:, 15],
            rows[:, 16:18],
            rows[:, 18:21],
            rows[:, 21],
        )


class SlidingSimulator:
    """The top on the table, stepped by RK4 from let go at rest (`launch_state`)."""

    def __init__(self, top: RoundTop, spin: float, tilt: float, per_period: int = PER_PERIOD) -> None:
        self.top = top
        self.state = launch_state(top, spin, tilt)
        self.t = 0.0
        self.event: Event | None = None
        self._rates = _field(top)
        turning = max(abs(spin), abs(spin) * top.polar / top.central, 1.0)
        self.h = min(2 * math.pi / turning / per_period, FRICTION_STEP / max(top.contact_rate(), 1e-9), MAX_STEP)
        self.stride = max(1, round(RECORD_DT / self.h))
        self._count = 0
        self._rim = top.rim_tilt()

    def sample(self) -> SlideSamples:
        """The state now, as one sample."""
        _, N, Fx, Fy = self._rates(self.state)
        return SlideSamples.of(np.array([_row(self.top, self.t, self.state, N, Fx, Fy)]))

    def advance(self, duration: float) -> SlideSamples:
        """Step on by `duration` (to the nearest whole step, at most MAX_STEPS_PER_ADVANCE), keeping every
        `stride`-th step; stops at an `Event`."""
        n = min(round(duration / self.h), MAX_STEPS_PER_ADVANCE)
        rows: list[list[float]] = []
        if self.event is not None:
            return SlideSamples.of(np.zeros((0, COLUMNS)))
        h, rates, s, t = self.h, self._rates, self.state, self.t
        top, rim, stride = self.top, self._rim, self.stride
        rim_cos = math.cos(rim)
        for _ in range(n):
            k1, N, Fx, Fy = rates(s)
            if N < 0:
                self.event = Event.HOPS
                break
            k2 = rates([a + 0.5 * h * b for a, b in zip(s, k1)])[0]
            k3 = rates([a + 0.5 * h * b for a, b in zip(s, k2)])[0]
            k4 = rates([a + h * b for a, b in zip(s, k3)])[0]
            s = [a + h / 6 * (b1 + 2 * (b2 + b3) + b4) for a, b1, b2, b3, b4 in zip(s, k1, k2, k3, k4)]
            norm = math.sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + s[3] * s[3])
            s[0:4] = [v / norm for v in s[0:4]]
            t += h
            self._count += 1
            ez = 1 - 2 * (s[1] * s[1] + s[2] * s[2])
            if ez < rim_cos:
                self.event = Event.RIM
            if self._count % stride == 0 or self.event is not None:
                _, N, Fx, Fy = rates(s)
                rows.append(_row(top, t, s, N, Fx, Fy))
            if self.event is not None:
                break
        self.state, self.t = s, t
        return SlideSamples.of(np.array(rows) if rows else np.zeros((0, COLUMNS)))


def reference(top: RoundTop, spin: float, tilt: float, t: np.ndarray, rtol: float = 1e-11) -> np.ndarray:
    """The state at times t by `solve_ivp` (DOP853) on the same equations: (len(t), 12)."""
    rates = _field(top)
    sol = scipy.integrate.solve_ivp(
        lambda _, y: rates(list(y))[0],
        (0.0, float(t[-1])),
        launch_state(top, spin, tilt),
        t_eval=t,
        rtol=rtol,
        atol=1e-13,
        method="DOP853",
    )
    return sol.y.T
