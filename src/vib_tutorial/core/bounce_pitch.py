"""A car's body in bounce and pitch: one rigid body on two springs, its equations in any two coordinates.

The *Deriving the equations* page. A `Car` is a rigid body of mass m and pitch inertia J = m r² about its
centre of mass G, on a front spring and damper (k_f, c_f) a ahead of G and a rear pair (k_r, c_r) b behind it,
the wheelbase L = a + b; the tyres and wheels are left out (the Quarter-car page has them). G rises x_G and the
body pitches θ, nose up, so the body above the axles rises

    δ_f = x_G + a θ,   δ_r = x_G − b θ,

each spring measured from the static equilibrium, so gravity drops out. The same motion has other coordinates
(`Coordinates`), each q with (x_G, θ) = T q:

* `CENTRE`: q = (x_G, θ), T = I. M = diag(m, J) and K = [[k_f + k_r, k_f a − k_r b], [·, k_f a² + k_r b²]]: the
  springs couple bounce and pitch (static coupling), unless k_f a = k_r b.
* `POINT`: q = (x_P, θ) at a point P a distance e ahead of G, x_G = x_P − e θ. Both M and K couple: M's
  −m e is dynamic coupling. At P = the elastic centre e* = (k_f a − k_r b)/(k_f + k_r) (`elastic_centre`) a
  force pushes the body straight, and K is diagonal.
* `ENDS`: q = (x_f, x_r), the body above each axle: x_G = (b x_f + a x_r)/L, θ = (x_f − x_r)/L. K = diag(k_f,
  k_r), and M couples by (m a b − J)/L², nil when the dynamic index r²/(ab) is 1: the ends then bounce apart.

Lagrange's equations, d/dt ∂T/∂q̇ − ∂T/∂q + ∂D/∂q̇ + ∂V/∂q = Q, with T = ½(m ẋ_G² + J θ̇²), V = ½(k_f δ_f² +
k_r δ_r²) and D = ½(c_f δ̇_f² + c_r δ̇_r²), give M = Tᵀ diag(m, J) T, C = Sᵀ diag(c) S and K = Sᵀ diag(k) S
with δ = S q (S = E T, E = [[1, a], [1, −b]]): symmetric in every set of coordinates (`matrices`). The
free-body diagram's force and moment (about G) balances are the same equations with the rows mixed:
Tᵀ times them is Lagrange's (`newton`). Taken about P they are Lagrange's own rows, if the moment of G's
inertia force, −m e ẍ_G, is kept (`slipped` drops it, as a hurried derivation does).

`modes` solves K φ = ω² M φ; each mode turns the body about a node (`Mode.node`, from G, forward), outside the
wheelbase for a mostly bouncing mode, inside it for a mostly pitching one. `ride` drives the car over a bump
(a 1 − cos bump of `length` and `height` at speed v, the rear wheel L/v after the front) by the exact FOH
update, the road linear between samples.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .simulator import foh_discretize

TWO_PI = 2.0 * math.pi
RIDE_PER_PERIOD = 60  # the ride's samples per period of the faster mode (and per bump's passing, at least)
RIDE_SETTLE = 4.0  # s the ride runs on after the rear wheel leaves the bump


class Coordinates(enum.Enum):
    CENTRE = "Centre of mass: x_G, θ"
    POINT = "A point P: x_P, θ"
    ENDS = "The two ends: x_f, x_r"


@dataclass(frozen=True)
class Car:
    """The body in bounce and pitch (springs and dampers per axle, both wheels together)."""

    mass: float = 1200.0  # kg
    gyration: float = 1.3  # m: J = m r²
    front: float = 1.2  # m, G to the front axle (a)
    rear: float = 1.5  # m, G to the rear axle (b)
    k_front: float = 40e3  # N/m
    k_rear: float = 40e3  # N/m
    c_front: float = 2500.0  # N·s/m
    c_rear: float = 2500.0  # N·s/m

    @property
    def inertia(self) -> float:
        return self.mass * self.gyration**2

    @property
    def wheelbase(self) -> float:
        return self.front + self.rear

    @property
    def dynamic_index(self) -> float:
        """r²/(ab): 1 uncouples the ends."""
        return self.gyration**2 / (self.front * self.rear)

    @property
    def end_masses(self) -> tuple[float, float]:
        """The body's weight on each axle, over g: (m b/L, m a/L)."""
        return self.mass * self.rear / self.wheelbase, self.mass * self.front / self.wheelbase

    def end_frequencies(self) -> tuple[float, float]:
        """Each end on its own spring, √(k/m_end)/2π (Hz): Olley's front and rear ride frequencies."""
        mf, mr = self.end_masses
        return math.sqrt(self.k_front / mf) / TWO_PI, math.sqrt(self.k_rear / mr) / TWO_PI

    def uncoupled_frequencies(self) -> tuple[float, float]:
        """Pure bounce √((k_f + k_r)/m) and pure pitch √((k_f a² + k_r b²)/J), over 2π (Hz)."""
        a, b = self.front, self.rear
        bounce = math.sqrt((self.k_front + self.k_rear) / self.mass)
        pitch = math.sqrt((self.k_front * a * a + self.k_rear * b * b) / self.inertia)
        return bounce / TWO_PI, pitch / TWO_PI

    def elastic_centre(self) -> float:
        """e* (m ahead of G): where a force pushes the body straight, (k_f a − k_r b)/(k_f + k_r)."""
        return (self.k_front * self.front - self.k_rear * self.rear) / (self.k_front + self.k_rear)

    def deflections(self) -> np.ndarray:
        """E: (δ_f, δ_r) = E (x_G, θ)."""
        return np.array([[1.0, self.front], [1.0, -self.rear]])


def transform(car: Car, coords: Coordinates, offset: float = 0.0) -> np.ndarray:
    """T: (x_G, θ) = T q in the coordinates (`offset` is P's e, ahead of G)."""
    if coords is Coordinates.CENTRE:
        return np.eye(2)
    if coords is Coordinates.POINT:
        return np.array([[1.0, -offset], [0.0, 1.0]])
    a, b, L = car.front, car.rear, car.wheelbase
    return np.array([[b / L, a / L], [1.0 / L, -1.0 / L]])


@dataclass(frozen=True)
class Matrices:
    M: np.ndarray
    C: np.ndarray
    K: np.ndarray


def matrices(car: Car, coords: Coordinates, offset: float = 0.0) -> Matrices:
    """Lagrange's M, C and K in the coordinates."""
    T = transform(car, coords, offset)
    S = car.deflections() @ T
    M = T.T @ np.diag([car.mass, car.inertia]) @ T
    C = S.T @ np.diag([car.c_front, car.c_rear]) @ S
    K = S.T @ np.diag([car.k_front, car.k_rear]) @ S
    return Matrices(M, C, K)


def newton(car: Car, coords: Coordinates, offset: float = 0.0) -> Matrices:
    """The free-body diagram's rows: the force balance and the moment balance about G, in q (not symmetric
    unless T is I). Tᵀ times them is `matrices`."""
    T = transform(car, coords, offset)
    E = car.deflections()
    return Matrices(
        np.diag([car.mass, car.inertia]) @ T,
        E.T @ np.diag([car.c_front, car.c_rear]) @ E @ T,
        E.T @ np.diag([car.k_front, car.k_rear]) @ E @ T,
    )


def slipped(car: Car, offset: float) -> Matrices:
    """About P, with the moment balance written J_P θ̈ = ΣM_P, the moment of G's inertia force (−m e ẍ_P) left
    out: wrong, and no longer symmetric."""
    m = matrices(car, Coordinates.POINT, offset)
    M = m.M.copy()
    M[1, 0] = 0.0
    return Matrices(M, m.C, m.K)


def energies(car: Car, coords: Coordinates, q, qdot, offset: float = 0.0) -> tuple[float, float, float]:
    """(T, V, D) straight from the body's motion: what Lagrange's equations differentiate."""
    T = transform(car, coords, offset)
    xg, th = T @ np.asarray(q, float)
    vg, w = T @ np.asarray(qdot, float)
    df, dr = xg + car.front * th, xg - car.rear * th
    vf, vr = vg + car.front * w, vg - car.rear * w
    kinetic = 0.5 * (car.mass * vg * vg + car.inertia * w * w)
    potential = 0.5 * (car.k_front * df * df + car.k_rear * dr * dr)
    dissipation = 0.5 * (car.c_front * vf * vf + car.c_rear * vr * vr)
    return kinetic, potential, dissipation


@dataclass(frozen=True)
class Mode:
    frequency: float  # Hz (undamped)
    shape: np.ndarray  # in the coordinates
    centre: np.ndarray  # (x_G, θ), scaled so the larger end moves 1

    @property
    def node(self) -> float:
        """Where the body stands still, m ahead of G (±inf for pure bounce)."""
        xg, th = self.centre
        return -xg / th if abs(th) > 1e-12 * abs(xg) else math.inf

    def inside(self, car: Car) -> bool:
        """The node between the axles: a mostly pitching mode."""
        return -car.rear < self.node < car.front


def modes(car: Car, coords: Coordinates = Coordinates.CENTRE, offset: float = 0.0) -> tuple[Mode, Mode]:
    """The two undamped modes, slower first."""
    m = matrices(car, coords, offset)
    w2, phi = scipy.linalg.eigh(m.K, m.M)
    T = transform(car, coords, offset)
    out = []
    for i in range(2):
        g = T @ phi[:, i]
        ends = car.deflections() @ g
        scale = ends[np.argmax(np.abs(ends))]
        out.append(Mode(math.sqrt(max(w2[i], 0.0)) / TWO_PI, phi[:, i] / scale, g / scale))
    return out[0], out[1]


def sweep(car: Car, ratios: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies, nodes), each (len(ratios), 2): the modes with the rear spring at ratio × the front's."""
    f, nodes = np.empty((ratios.size, 2)), np.empty((ratios.size, 2))
    for i, ratio in enumerate(ratios):
        ms = modes(Car(**{**car.__dict__, "k_rear": ratio * car.k_front}))
        f[i] = [m.frequency for m in ms]
        nodes[i] = [m.node for m in ms]
    return f, nodes


@dataclass(frozen=True)
class Bump:
    """A speed hump, by default: 75 mm high and 3.7 m long, at 30 km/h."""

    speed: float = 30.0 / 3.6  # m/s
    height: float = 0.075  # m
    length: float = 3.7  # m

    def profile(self, s: np.ndarray) -> np.ndarray:
        """The road's height at s metres past the bump's start."""
        inside = (s >= 0) & (s <= self.length)
        return np.where(inside, 0.5 * self.height * (1.0 - np.cos(TWO_PI * s / self.length)), 0.0)


@dataclass(frozen=True)
class Ride:
    t: np.ndarray
    road: np.ndarray  # (n, 2): the road under the front and rear wheels
    centre: np.ndarray  # (n, 2): x_G, θ
    ends: np.ndarray  # (n, 2): the body above the front and rear axles
    lead: float  # s: when the front wheel reaches the bump
    leaves: float  # s: when the rear wheel leaves the bump

    @property
    def pitch(self) -> np.ndarray:
        return self.centre[:, 1]

    def after(self) -> np.ndarray:
        """The samples once both wheels are past the bump: the car's own motion."""
        return self.t > self.leaves


def ride(car: Car, bump: Bump, lead: float = 0.3) -> Ride:
    """The car over the bump, the front wheel reaching it `lead` s in, by the exact FOH update.

    In (x_G, θ), M q̈ + C q̇ + K q = Eᵀ diag(k) y + Eᵀ diag(c) ẏ, y the road under each wheel. The ẏ term goes
    into the state, w = q̇ − M⁻¹Eᵀ diag(c) y, so the input is y alone, linear over a step.
    """
    m = matrices(car, Coordinates.CENTRE)
    E = car.deflections()
    Mi = np.linalg.inv(m.M)
    Ek, Ec = E.T @ np.diag([car.k_front, car.k_rear]), E.T @ np.diag([car.c_front, car.c_rear])
    A = np.block([[np.zeros((2, 2)), np.eye(2)], [-Mi @ m.K, -Mi @ m.C]])
    B = np.vstack([Mi @ Ec, Mi @ (Ek - m.C @ Mi @ Ec)])
    fast = modes(car)[1].frequency
    h = min(1.0 / (RIDE_PER_PERIOD * fast), bump.length / bump.speed / RIDE_PER_PERIOD)
    leaves = lead + (car.wheelbase + bump.length) / bump.speed
    span = leaves + RIDE_SETTLE
    n = math.ceil(span / h) + 1
    t = np.arange(n) * h
    s = bump.speed * (t - lead)
    y = np.column_stack([bump.profile(s), bump.profile(s - car.wheelbase)])
    Phi, G0, G1 = foh_discretize(A, B, h)
    z = np.zeros((n, 4))
    for k in range(1, n):
        z[k] = Phi @ z[k - 1] + G0 @ y[k - 1] + G1 @ y[k]
    centre = z[:, :2]
    return Ride(t, y, centre, centre @ E.T, lead, leaves)
