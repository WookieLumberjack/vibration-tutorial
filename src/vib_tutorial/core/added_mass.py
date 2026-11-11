"""Added mass: the fluid a body has to push about when it accelerates, and what it does to natural frequencies.

The *Fluid–structure* page. A body moving with velocity U in an ideal, incompressible fluid at rest sets the
fluid moving with it in a potential flow, whose kinetic energy ½m_a U² makes the body behave as if its mass were
m + m_a (the fluid's own pressure on it, ρ ∂φ/∂t, in phase with its acceleration). In closed form:

- a sphere of radius R: m_a = ½ ρ (4/3)πR³, half the fluid it displaces; inside a concentric spherical wall at
  R_o, (2/3)πρR³ (R_o³ + 2R³)/(R_o³ − R³);
- a circular cylinder (per length): m_a = ρπR², all the fluid it displaces; inside a concentric wall,
  ρπR² (R_o² + R²)/(R_o² − R²) (Fritz), growing without bound as the gap closes.

`Body` holds one of them and `flow` its potential, stream function and velocity field (the body moving along x,
the wall still). On a spring, the stiffness stays and the frequency falls to f √(m/(m + m_a)).

A plate's mode pushes the fluid on one side (baffled), and each mode carries its own added mass:
`plate_added_mass` is (ρ/4π²) ∫∫ |Φ̂(k)|²/|k| d²k, Φ̂ the mode shape's Fourier transform, since an incompressible
fluid's pressure on a plane, for a surface acceleration of wavenumber k, is ρ a(k)/|k|. Over the modal mass
ρ_p h ab/4 it gives the mode's AVMI factor β (added virtual mass incremental), and f_wet = f_dry/√(1 + β). Shorter
waves reach less fluid: β falls with the mode's order. The fluid's coupling between modes is left out.

Working on it
-------------
`tests/test_added_mass.py` checks the closed forms against the flow's kinetic energy, the plate's integral
against a direct one in space and a circular piston's (8/3)ρa³, and the frequencies.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .beams_plates import TWO_PI, Plate

WATER = 1000.0  # kg/m³
SEAWATER = 1025.0
AIR = 1.21


class Shape(enum.IntEnum):
    SPHERE = 0
    CYLINDER = 1


SHAPE_NAMES = ("Sphere", "Cylinder (per length)")


@dataclass(frozen=True)
class Body:
    """A sphere or a cylinder of radius R and density ρ_s, in fluid of density ρ_f, inside a still concentric
    wall at `outer` (∞: none)."""

    shape: Shape = Shape.CYLINDER
    radius: float = 0.05  # m
    density: float = 7850.0  # kg/m³
    fluid: float = WATER
    outer: float = math.inf  # m

    @property
    def volume(self) -> float:
        """m³ (per metre of length for a cylinder)."""
        R = self.radius
        return 4 / 3 * math.pi * R**3 if self.shape == Shape.SPHERE else math.pi * R**2

    @property
    def mass(self) -> float:
        return self.density * self.volume

    @property
    def coefficient(self) -> float:
        """C_a = m_a / ρ_f V: ½ for a sphere and 1 for a cylinder in open fluid, more inside a wall."""
        if math.isinf(self.outer):
            return 0.5 if self.shape == Shape.SPHERE else 1.0
        r = self.outer / self.radius
        if self.shape == Shape.SPHERE:
            return 0.5 * (r**3 + 2) / (r**3 - 1)
        return (r**2 + 1) / (r**2 - 1)

    @property
    def added_mass(self) -> float:
        return self.coefficient * self.fluid * self.volume


def wet_frequency(f_dry: float, mass: float, added: float) -> float:
    """The same spring under m + m_a: f √(m/(m + m_a))."""
    return f_dry * math.sqrt(mass / (mass + added))


@dataclass(frozen=True)
class Flow:
    """The fluid's motion about the body (moving at U = 1 along x) on a grid: velocity (lab frame), the stream
    function, and NaN inside the body or beyond the wall. For a sphere, the plane through its axis of motion."""

    x: np.ndarray
    y: np.ndarray
    u: np.ndarray  # (len(y), len(x))
    v: np.ndarray
    stream: np.ndarray
    contours: np.ndarray  # ψ everywhere, for drawing streamlines: the body's own (rigid) inside it, 0 beyond the wall


def _coefficients(body: Body) -> tuple[float, float]:
    """φ = cos θ (A r + B/r^p), p = 1 (cylinder) or 2 (sphere), with ∂φ/∂r = cos θ on the body and 0 at the
    wall: A = −R^{p+1}/(R_o^{p+1} − R^{p+1}), B = A R_o^{p+1}/p (A = 0, B = −R^{p+1}/p without a wall)."""
    p = 2 if body.shape == Shape.SPHERE else 1
    R = body.radius
    if math.isinf(body.outer):
        return 0.0, -(R ** (p + 1)) / p
    Ro = body.outer
    A = -(R ** (p + 1)) / (Ro ** (p + 1) - R ** (p + 1))
    return A, A * Ro ** (p + 1) / p


def potential(body: Body, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    A, B = _coefficients(body)
    p = 2 if body.shape == Shape.SPHERE else 1
    return np.cos(theta) * (A * r + B / r**p)


def flow(body: Body, x: np.ndarray, y: np.ndarray) -> Flow:
    """The velocity ∇φ and the stream function: ψ = sin θ (A r − B/r) for the cylinder, Stokes's
    ψ = sin²θ (A r²/2 − B/r) for the sphere (its meridian plane; u = ∇ × (ψ/ϖ) about the axis)."""
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X, Y)
    th = np.arctan2(Y, X)
    A, B = _coefficients(body)
    with np.errstate(divide="ignore", invalid="ignore"):
        if body.shape == Shape.SPHERE:
            ur = np.cos(th) * (A - 2 * B / r**3)
            ut = -np.sin(th) * (A + B / r**3)
            psi = np.sin(th) ** 2 * (0.5 * A * r**2 - B / r)
        else:
            ur = np.cos(th) * (A - B / r**2)
            ut = -np.sin(th) * (A + B / r**2)
            psi = np.sin(th) * (A * r - B / r)
        u = ur * np.cos(th) - ut * np.sin(th)
        v = ur * np.sin(th) + ut * np.cos(th)
    inside, beyond = r < body.radius, r > body.outer
    rigid = r * np.sin(th) if body.shape == Shape.CYLINDER else 0.5 * (r * np.sin(th)) ** 2  # U y, ½U ϖ²
    contours = np.where(inside, rigid, np.where(beyond, 0.0, psi))  # the wall is the streamline ψ = 0
    for a in (u, v, psi):
        a[inside | beyond] = np.nan
    return Flow(x, y, u, v, psi, contours)


def axis_speed(body: Body, r: np.ndarray) -> np.ndarray:
    """The fluid's speed (over U) ahead of the body along its line of motion: R²/r² for a cylinder, R³/r³ for a
    sphere in open fluid; inside a wall it falls to zero there."""
    A, B = _coefficients(body)
    p = 2 if body.shape == Shape.SPHERE else 1
    return A - p * B / np.asarray(r, dtype=float) ** (p + 1)


# --------------------------------------------------------------------------------------------- a plate
def _shape_power(k: np.ndarray, m: int, a: float) -> np.ndarray:
    """|∫₀^a sin(mπx/a) e^{−ikx} dx|² = (mπ/a)² 2(1 − (−1)^m cos ka) / ((mπ/a)² − k²)², (a/2)² at k = ±mπ/a."""
    km = m * math.pi / a
    den = km * km - k * k
    near = np.abs(den) < 1e-9 * km * km
    with np.errstate(divide="ignore", invalid="ignore"):
        value = km * km * 2 * (1 - (-1) ** m * np.cos(k * a)) / np.where(near, 1.0, den) ** 2
    return np.where(near, 0.25 * a * a, value)


@lru_cache(maxsize=4096)
def _plate_integral(a: float, b: float, m: int, n: int) -> float:
    """∫∫ |Φ̂|²/|k| d²k / 4π² for sin(mπx/a) sin(nπy/b), m³ (times ρ: the added mass on one side). In polar
    wavenumbers the 1/|k| cancels the area's k: Gauss–Legendre over a quarter of the angles (the rest by symmetry)
    and over k in pieces, to 30 times the mode's own wavenumbers (the tail, ∝ 1/K³, is under 10⁻⁴)."""
    top = 30 * max(m * math.pi / a, n * math.pi / b, TWO_PI / min(a, b))
    pieces = int(top * max(a, b) / math.pi) + 4  # 16 nodes to each half oscillation
    xs, ws = np.polynomial.legendre.leggauss(16)
    edges = np.linspace(0.0, top, pieces + 1)
    h = np.diff(edges)
    k = (0.5 * h[:, None] * (xs + 1) + edges[:-1, None]).ravel()
    wk = (0.5 * h[:, None] * ws).ravel()
    xp, wp = np.polynomial.legendre.leggauss(96)
    phi = 0.25 * math.pi * (xp + 1)
    wphi = 0.25 * math.pi * wp
    kk, pp = np.meshgrid(k, phi, indexing="ij")
    f = _shape_power(kk * np.cos(pp), m, a) * _shape_power(kk * np.sin(pp), n, b)
    return float(4 * wk @ f @ wphi) / (4 * math.pi**2)


def plate_added_mass(plate: Plate, m: int, n: int, fluid: float = WATER, sides: int = 1) -> float:
    """Mode (m, n)'s added mass (kg, for its shape's peak amplitude, as the modal mass ρ_p h ab/4), the plate
    baffled with fluid on `sides` sides."""
    return sides * fluid * _plate_integral(plate.a, plate.b, m, n)


def plate_modal_mass(plate: Plate) -> float:
    return plate.surface_mass * plate.a * plate.b / 4


def avmi(plate: Plate, m: int, n: int, fluid: float = WATER, sides: int = 1) -> float:
    """β = m_a / (ρ_p h ab/4): f_wet = f_dry / √(1 + β)."""
    return plate_added_mass(plate, m, n, fluid, sides) / plate_modal_mass(plate)


def plate_dry_frequency(plate: Plate, m: int, n: int) -> float:
    """A simply supported plate's f = (π/2)((m/a)² + (n/b)²)√(D/ρh)."""
    return math.pi**2 * ((m / plate.a) ** 2 + (n / plate.b) ** 2) * plate.scale / TWO_PI


@dataclass(frozen=True)
class PlateModes:
    """The plate's first modes in its dry order, each with β and its wet frequency."""

    m: np.ndarray
    n: np.ndarray
    dry: np.ndarray  # Hz
    beta: np.ndarray
    wet: np.ndarray


def plate_modes(plate: Plate, count: int, fluid: float = WATER, sides: int = 1) -> PlateModes:
    top = math.ceil(math.sqrt(count)) + 3
    pairs = [(m, n) for m in range(1, 3 * top) for n in range(1, 3 * top)]
    pairs.sort(key=lambda mn: plate_dry_frequency(plate, *mn))
    pairs = pairs[:count]
    m = np.array([p[0] for p in pairs])
    n = np.array([p[1] for p in pairs])
    dry = np.array([plate_dry_frequency(plate, *p) for p in pairs])
    beta = np.array([avmi(plate, *p, fluid=fluid, sides=sides) for p in pairs])
    return PlateModes(m, n, dry, beta, dry / np.sqrt(1 + beta))
