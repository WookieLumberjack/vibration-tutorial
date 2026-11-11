"""Sloshing: the free surface of a liquid in a tank, its modes, the spring–mass analogue and a tuned liquid damper.

The *Fluid–structure* page's last two tabs. A rectangular tank of length L (along the motion), breadth B, filled
to depth h, in linear potential flow: φ = cosh(k(z + h)) cos(kx) satisfies Laplace's equation, the walls (x = 0,
L) and the bottom (z = −h) for k_n = nπ/L, and the free surface's condition ∂φ/∂z = ω²φ/g for

    ω_n² = g k_n tanh(k_n h),

√(gh)k in shallow water (a long wave), √(gk) in deep. A sideways motion of the tank excites only the odd modes
(the surface tilts one way and the other); the even ones are symmetric. The force on the walls is that of a
spring–mass analogue: a rigid mass m₀ fixed to the tank and a mass m_n on a spring k_n = m_n ω_n² for each odd
mode,

    m_n = m_L 8 tanh(k_n h) / (n³π³ h/L),  m₀ = m_L − Σ m_n,

m_L = ρLBh the liquid's mass. A shallow tank's liquid nearly all sloshes (Σ 8/n²π² = 1); a deep one's lower layer
moves with the tank. `effective_mass` is the walls' force over the tank's acceleration, m_L at no frequency and m₀
far above the modes.

A tuned liquid damper is such a tank on a structure: its first mode is a tuned mass damper of mass m₁ (screens or
baffles damp it, ζ_d), the rest of the liquid more mass. `damper_response` eliminates the sloshing masses
exactly: each adds a dynamic stiffness −ω²m(k + iωc)/(k + iωc − ω²m) to the structure. Den Hartog's tuning for
the mass ratio μ is f_d = f/(1 + μ), ζ_d = √(3μ/8(1 + μ)³), and the peak √(1 + 2/μ). Real dampers are nonlinear
at large amplitude (the waves break), which this linear model leaves out.

Working on it
-------------
`tests/test_sloshing.py` checks the frequencies and the rigid mass against a finite-volume solution of the tank's
potential flow, the masses' sums, the damper's elimination against a full matrix solve and Den Hartog's peak.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .added_mass import WATER
from .beams_plates import TWO_PI

G = 9.81  # m/s²
TERMS = 2000  # odd modes summed for the rigid mass (the tail goes as 1/n³)


@dataclass(frozen=True)
class Tank:
    """A rectangular tank of length L (along its motion) and breadth B, filled to depth h."""

    length: float = 1.0  # m
    depth: float = 0.5
    breadth: float = 1.0
    fluid: float = WATER  # kg/m³

    @property
    def liquid_mass(self) -> float:
        return self.fluid * self.length * self.breadth * self.depth


def wavenumber(tank: Tank, n: int | np.ndarray) -> np.ndarray:
    return np.asarray(n) * math.pi / tank.length


def slosh_omega(tank: Tank, n: int | np.ndarray) -> np.ndarray:
    """ω_n = √(g k_n tanh(k_n h)), rad/s."""
    k = wavenumber(tank, n)
    return np.sqrt(G * k * np.tanh(k * tank.depth))


def slosh_frequency(tank: Tank, n: int) -> float:
    return float(slosh_omega(tank, n)) / TWO_PI


def slosh_mass(tank: Tank, n: int | np.ndarray) -> np.ndarray:
    """The analogue's mass for mode n: m_L 8 tanh(k_n h)/(n³π³ h/L) for odd n, 0 for even (not excited by a
    sideways motion)."""
    n = np.asarray(n)
    k = wavenumber(tank, n)
    m = tank.liquid_mass * 8 * np.tanh(k * tank.depth) / (n**3 * math.pi**3 * tank.depth / tank.length)
    return np.where(n % 2 == 1, m, 0.0)


def rigid_mass(tank: Tank) -> float:
    """m₀ = m_L − Σ m_n: the liquid that moves with the tank."""
    odd = np.arange(1, 2 * TERMS, 2)
    return tank.liquid_mass - float(np.sum(slosh_mass(tank, odd)))


def effective_mass(tank: Tank, f: np.ndarray, zeta: float = 0.0, modes: int = TERMS) -> np.ndarray:
    """The walls' force over the tank's acceleration (complex, kg), with each mode damped by ζ: m₀ + Σ m_n
    (ω_n² + 2iζω_nω)/(ω_n² − ω² + 2iζω_nω)."""
    odd = np.arange(1, 2 * modes, 2)
    w = TWO_PI * np.asarray(f, dtype=float)[..., None]
    wn = slosh_omega(tank, odd)
    stiff = wn**2 + 2j * zeta * wn * w
    return rigid_mass(tank) + np.sum(slosh_mass(tank, odd) * stiff / (stiff - w**2), axis=-1)


@dataclass(frozen=True)
class TankFlow:
    """Mode n's flow on a grid (x from 0 to L, z from −h to 0) as the surface passes level: its velocity's
    magnitude (over its peak) and stream function; and the surface's shape at its peak, cos(k_n x)."""

    x: np.ndarray
    z: np.ndarray
    speed: np.ndarray  # (len(z), len(x))
    stream: np.ndarray
    surface: np.ndarray  # at x


def tank_flow(tank: Tank, n: int, x: np.ndarray, z: np.ndarray) -> TankFlow:
    """φ = cosh(k(z + h)) cos(kx), so u = −k cosh sin, w = k sinh cos, and ψ = −sinh(k(z + h)) sin(kx) (0 on the
    walls and the bottom)."""
    k = float(wavenumber(tank, n))
    X, Z = np.meshgrid(x, z)
    s = k * (Z + tank.depth)
    kh = k * tank.depth
    grow, fall = np.exp(s - kh), np.exp(-s - kh)  # cosh and sinh over e^{kh}: no overflow in a deep tank
    ch, sh = 0.5 * (grow + fall), 0.5 * (grow - fall)
    speed = np.hypot(ch * np.sin(k * X), sh * np.cos(k * X))
    stream = -sh * np.sin(k * X)
    return TankFlow(x, z, speed / speed.max(), stream / np.abs(stream).max(), np.cos(k * x))


# --------------------------------------------------------------------------------------------- a damper
@dataclass(frozen=True)
class Structure:
    """The structure the damper sits on, as one mode: its mass, natural frequency and damping ratio."""

    mass: float = 10000.0  # kg
    frequency: float = 0.5  # Hz
    zeta: float = 0.01


@dataclass(frozen=True)
class Damper:
    """`count` identical tanks on the structure, their sloshing damped by ζ_d (screens, baffles)."""

    tank: Tank
    count: int = 1
    zeta: float = 0.05
    modes: int = 3  # odd modes kept as springs (the rest are rigid)

    @property
    def masses(self) -> np.ndarray:
        """The sloshing masses of modes 1, 3, 5, …, all the tanks'."""
        return self.count * slosh_mass(self.tank, np.arange(1, 2 * self.modes, 2))

    @property
    def omegas(self) -> np.ndarray:
        return slosh_omega(self.tank, np.arange(1, 2 * self.modes, 2))

    @property
    def rigid(self) -> float:
        """All the liquid but the sloshing masses kept: it moves with the structure."""
        return self.count * self.tank.liquid_mass - float(self.masses.sum())

    @property
    def liquid(self) -> float:
        return self.count * self.tank.liquid_mass


def mass_ratio(structure: Structure, damper: Damper) -> float:
    """μ = m₁ over everything else that moves with the structure."""
    m1 = float(damper.masses[0])
    return m1 / (structure.mass + damper.liquid - m1)


def loaded_frequency(structure: Structure, damper: Damper) -> float:
    """The structure's frequency carrying all but the first sloshing mass rigidly, Hz."""
    m1 = float(damper.masses[0])
    return structure.frequency * math.sqrt(structure.mass / (structure.mass + damper.liquid - m1))


def den_hartog(mu: float) -> tuple[float, float, float]:
    """Den Hartog's tuned mass damper for a force on an undamped structure: the damper's frequency over the
    structure's, its damping ratio and the peak amplification."""
    return 1 / (1 + mu), math.sqrt(3 * mu / (8 * (1 + mu) ** 3)), math.sqrt(1 + 2 / mu)


def damper_response(structure: Structure, damper: Damper | None, f: np.ndarray) -> np.ndarray:
    """The structure's displacement under a force, over its static deflection F/k (complex), with the damper's
    sloshing masses eliminated: each adds −ω²m(k + iωc)/(k + iωc − ω²m) to its dynamic stiffness. None: the
    structure alone."""
    w = TWO_PI * np.asarray(f, dtype=float)
    ws = TWO_PI * structure.frequency
    k = structure.mass * ws**2
    z = k + 2j * structure.zeta * structure.mass * ws * w - w**2 * structure.mass
    if damper is not None:
        z = z - w**2 * damper.rigid
        for m, wn in zip(damper.masses, damper.omegas):
            spring = m * (wn**2 + 2j * damper.zeta * wn * w)  # k + iωc
            z = z - w**2 * m * spring / (spring - w**2 * m)
    return k / z


def frozen_response(structure: Structure, damper: Damper, f: np.ndarray) -> np.ndarray:
    """As if the liquid were frozen: the structure carrying all of it rigidly."""
    w = TWO_PI * np.asarray(f, dtype=float)
    ws = TWO_PI * structure.frequency
    k = structure.mass * ws**2
    return k / (k + 2j * structure.zeta * structure.mass * ws * w - w**2 * (structure.mass + damper.liquid))


def response_band(structure: Structure, points: int = 3000) -> np.ndarray:
    """Frequencies about the structure's, fine enough for a lightly damped damper's peaks."""
    return np.geomspace(structure.frequency / 3, structure.frequency * 2, points)


def peak(structure: Structure, damper: Damper | None, points: int = 3000) -> float:
    return float(np.abs(damper_response(structure, damper, response_band(structure, points))).max())


def peak_against_depth(structure: Structure, damper: Damper, depths: np.ndarray, points: int = 3000) -> np.ndarray:
    """The structure's peak amplification with the tanks filled to each depth."""
    out = np.empty(len(depths))
    f = response_band(structure, points)
    for i, h in enumerate(depths):
        tank = Tank(damper.tank.length, float(h), damper.tank.breadth, damper.tank.fluid)
        d = Damper(tank, damper.count, damper.zeta, damper.modes)
        out[i] = float(np.abs(damper_response(structure, d, f)).max())
    return out
