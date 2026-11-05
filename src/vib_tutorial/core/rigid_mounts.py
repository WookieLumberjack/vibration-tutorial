"""A rigid body on resilient mounts: its six coupled modes, the elastic centre, and decoupling.

The *Rigid body on mounts* page. A `Body` is a uniform box of mass m whose centre of mass is the origin, its edges
along x (length), y (width) and z (height), so its principal moments are I_xx = m(w² + h²)/12 and so on. It
moves as a rigid body: three translations u and three small rotations θ (roll about x, pitch about y, yaw about
z), q = (u, θ). A `Mount` at r (from the centre of mass) is a spring with stiffness k_a along its own axis a and
k_s across it, K_m = k_s I + (k_a − k_s) a aᵀ. Its end moves u + θ × r = T q, T = [I, −[r]×], so

    K = Σ Tᵀ K_m T,   M = diag(m, m, m, I_xx, I_yy, I_zz),

and the modes come from K φ = ω² M φ (`modes`). Each mode's kinetic energy splits among the six coordinates,
M_ii φ_i² over the sum (`Modes.energy`): 100 % in one is a decoupled mode, the measure mount makers quote.

K's off-diagonal block K_tr = −Σ K_m [r]× couples translation to rotation. Taken about another point c, the
mounts sit at r − c and the block changes linearly with c; the point that makes it vanish, where a force
pushes the body straight and a moment only turns it, is the **elastic centre** (`elastic_centre`: in general
no point quite does, so the least-squares one, with what's left; `plane_centres` gives each plane's). If it is the centre of mass, and the box's
principal axes are the mounts' too, M and K are both block-diagonal and every mode is a pure translation or a
pure rotation.

`Layout` is the page's four mounts: at (±span − offset, ±track, −drop) from the centre of mass, each leaning
`tilt` toward the middle in the y–z plane (a V, "focalized"), with k_a and k_s = `shear_ratio` k_a. Below the
centre of mass (drop > 0) the mounts' shear couples sideways motion to roll and fore-and-aft to pitch; off
centre (offset ≠ 0), bounce couples to pitch. Tilting the mounts raises the roll plane's elastic centre to

    z_e = t(k_a − k_s) sin β cos β / (k_a sin²β + k_s cos²β)

above them (t the track, β the tilt; `roll_centre`), so at the tilt `focus_tilt` it reaches the centre of mass
and lateral and roll decouple again: with u = tan β, h u² − t(1 − ρ)u + hρ = 0 (ρ = k_s/k_a, h the drop).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg

DOF = ("x", "y", "z", "roll", "pitch", "yaw")
TWO_PI = 2.0 * math.pi


def cross_matrix(r) -> np.ndarray:
    """[r]×: [r]× v = r × v."""
    x, y, z = r
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


@dataclass(frozen=True)
class Body:
    """A uniform box, its centre of mass at the origin."""

    mass: float = 20.0  # kg
    length: float = 0.6  # m, along x
    width: float = 0.5  # m, along y
    height: float = 0.6  # m, along z

    @property
    def inertia(self) -> tuple[float, float, float]:
        """(I_xx, I_yy, I_zz) about the centre of mass (kg·m²)."""
        m, a, b, c = self.mass, self.length, self.width, self.height
        return m * (b * b + c * c) / 12.0, m * (a * a + c * c) / 12.0, m * (a * a + b * b) / 12.0

    def mass_matrix(self) -> np.ndarray:
        return np.diag([self.mass] * 3 + list(self.inertia))


@dataclass(frozen=True)
class Mount:
    """A spring at `position` (m, from the centre of mass): `axial` along its unit `axis`, `shear` across it."""

    position: tuple[float, float, float]
    axial: float  # N/m
    shear: float  # N/m
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def stiffness(self) -> np.ndarray:
        """K_m = k_s I + (k_a − k_s) a aᵀ (3 × 3)."""
        a = np.array(self.axis)
        return self.shear * np.eye(3) + (self.axial - self.shear) * np.outer(a, a)

    def transform(self, about=(0.0, 0.0, 0.0)) -> np.ndarray:
        """T (3 × 6): the mount's end moves T q, q = (u, θ) about the point `about`."""
        return np.hstack([np.eye(3), -cross_matrix(np.array(self.position) - np.array(about))])


def stiffness_matrix(mounts, about=(0.0, 0.0, 0.0)) -> np.ndarray:
    """K = Σ Tᵀ K_m T (6 × 6), its coordinates about the point `about`."""
    K = np.zeros((6, 6))
    for mount in mounts:
        T = mount.transform(about)
        K += T.T @ mount.stiffness() @ T
    return K


@dataclass(frozen=True)
class Modes:
    """The six modes: frequencies (Hz, ascending), shapes (6 × 6, columns, mass-normalized) and the kinetic
    energy's split among x, y, z, roll, pitch, yaw (6 × 6: rows the coordinates, columns the modes)."""

    frequencies: np.ndarray
    shapes: np.ndarray
    energy: np.ndarray

    def dominant(self, mode: int) -> int:
        """The coordinate with most of a mode's kinetic energy."""
        return int(np.argmax(self.energy[:, mode]))

    def name(self, mode: int) -> str:
        """'roll 97 %, y 3 %': the coordinates with at least 1 % of a mode's kinetic energy, most first."""
        share = self.energy[:, mode]
        order = [i for i in np.argsort(share)[::-1] if share[i] >= 0.01]
        return ", ".join(f"{DOF[i]} {100 * share[i]:.0f} %" for i in order)


def modes(body: Body, mounts) -> Modes:
    """The rigid body's six modes on its mounts."""
    M = body.mass_matrix()
    lam, phi = scipy.linalg.eigh(stiffness_matrix(mounts), M)
    lam = np.maximum(lam, 0.0)
    kinetic = np.diag(M)[:, None] * phi * phi
    return Modes(np.sqrt(lam) / TWO_PI, phi, kinetic / kinetic.sum(axis=0))


def coupling(mounts, about=(0.0, 0.0, 0.0)) -> np.ndarray:
    """K's translation–rotation block (3 × 3) about the point `about`."""
    return stiffness_matrix(mounts, about)[:3, 3:]


def elastic_centre(mounts) -> tuple[np.ndarray, float]:
    """The point about which the coupling block vanishes (least squares, as it is linear in the point), and the
    coupling left there over the coupling about the origin (0 where there is a true elastic centre)."""
    base = coupling(mounts)
    columns = [(coupling(mounts, e) - base).ravel() for e in np.eye(3)]
    A = np.stack(columns, axis=1)
    point = np.linalg.lstsq(A, -base.ravel(), rcond=None)[0]
    left = float(np.linalg.norm(coupling(mounts, point)))
    whole = float(np.linalg.norm(base))
    return point, left / whole if whole > 0 else 0.0


def plane_centres(mounts) -> tuple[float, tuple[float, float]]:
    """Each plane's elastic centre, where a force in the plane neither turns nor is turned by the body: the roll
    plane's height z (K's y–roll entry vanishes about it) and the pitch plane's (x, z) (its x–pitch and z–pitch
    entries vanish). Each entry is linear in the point, so each is a small linear solve. NaN where a plane has
    none (no stiffness to couple)."""

    def entry(point, i, j):
        return stiffness_matrix(mounts, point)[i, j]

    origin = np.zeros(3)
    ez, ex = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    a0 = entry(origin, 1, 3)
    slope = entry(ez, 1, 3) - a0
    roll = -a0 / slope if slope else math.nan
    rows = []
    for i in (0, 2):
        b0 = entry(origin, i, 4)
        rows.append((entry(ex, i, 4) - b0, entry(ez, i, 4) - b0, -b0))
    A = np.array([r[:2] for r in rows])
    b = np.array([r[2] for r in rows])
    if abs(np.linalg.det(A)) < 1e-12 * max(np.abs(A).max(), 1e-300) ** 2:
        return roll, (math.nan, math.nan)
    x, z = np.linalg.solve(A, b)
    return roll, (float(x), float(z))


class Sweep(enum.Enum):
    DROP = "Mounts' drop below the centre of mass"
    TILT = "Mounts' tilt"
    OFFSET = "Centre of mass's offset"
    SHEAR = "Shear over axial stiffness"


SWEEPS = tuple(Sweep)


@dataclass(frozen=True)
class Layout:
    """Four mounts under a box: at (±span − offset, ±track, −drop), each tilted toward the middle in y–z."""

    body: Body = Body()
    span: float = 0.25  # m, half the mounts' spacing along x
    track: float = 0.12  # m, half the spacing along y
    drop: float = 0.0  # m, how far below the centre of mass they sit
    offset: float = 0.0  # m, how far the centre of mass sits ahead (+x) of the mounts' middle
    tilt: float = 0.0  # rad, each mount's axis leaning toward the middle (a V seen from the front)
    stiffness: float = 50e3  # N/m, each mount's along its axis
    shear_ratio: float = 0.25  # k_s/k_a

    def mounts(self) -> tuple[Mount, ...]:
        s, c = math.sin(self.tilt), math.cos(self.tilt)
        out = []
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                out.append(
                    Mount(
                        (sx * self.span - self.offset, sy * self.track, -self.drop),
                        self.stiffness,
                        self.shear_ratio * self.stiffness,
                        (0.0, -sy * s, c),  # up and in: the pair's axes meet above the middle
                    )
                )
        return tuple(out)

    def modes(self) -> Modes:
        return modes(self.body, self.mounts())

    def roll_centre(self) -> float:
        """The roll plane's elastic centre's height (m) above the mounts: t(k_a − k_s) sin β cos β /
        (k_a sin²β + k_s cos²β), where a sideways force through it doesn't roll the body."""
        s, c = math.sin(self.tilt), math.cos(self.tilt)
        ka, ks = self.stiffness, self.shear_ratio * self.stiffness
        return self.track * (ka - ks) * s * c / (ka * s * s + ks * c * c)

    def focus_tilt(self) -> float:
        """The tilt (rad) that puts the roll plane's elastic centre at the centre of mass, decoupling lateral
        from roll: the smaller root of h u² − t(1 − ρ)u + hρ = 0, u = tan β. NaN if there is none (the mounts too
        far below, or too stiff in shear); 0 if they are level with the centre of mass already."""
        h, t, rho = self.drop, self.track, self.shear_ratio
        if h <= 0:
            return 0.0
        disc = (t * (1 - rho)) ** 2 - 4 * h * h * rho
        if disc < 0:
            return math.nan
        return math.atan((t * (1 - rho) - math.sqrt(disc)) / (2 * h))

    def with_(self, sweep: Sweep, value: float) -> Layout:
        """The layout with one quantity swept (lengths in m, the tilt in rad)."""
        field = {Sweep.DROP: "drop", Sweep.TILT: "tilt", Sweep.OFFSET: "offset", Sweep.SHEAR: "shear_ratio"}[sweep]
        return Layout(**{**self.__dict__, field: value})


def sweep(layout: Layout, which: Sweep, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The six frequencies (Hz, len(values) × 6, ascending) and each mode's dominant coordinate as one quantity
    changes."""
    freqs = np.empty((len(values), 6))
    dominant = np.empty((len(values), 6), dtype=int)
    for i, v in enumerate(values):
        m = layout.with_(which, float(v)).modes()
        freqs[i] = m.frequencies
        dominant[i] = [m.dominant(j) for j in range(6)]
    return freqs, dominant
