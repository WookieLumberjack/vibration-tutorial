"""Jeffcott rotor on flexible bearings: model, whirl analysis and exact time stepping.

A massless, uniform Euler–Bernoulli shaft runs through two bearings, A at z = 0
and B at z = L, and carries a rigid disc at z = a: between the bearings
(0 < a < L), or overhung past one of them (a > L past B, a < 0 past A), the
shaft then running on from the bearing to the disc. The shaft bends in
two planes, x (horizontal) and y (vertical). Each plane has four degrees of
freedom:

    the bearing journal A's displacement, the disc's displacement, the disc's
    slope (dx/dz or dy/dz), and the bearing journal B's displacement,

so q = [x_A, x_D, θx, x_B, y_A, y_D, θy, y_B] (m, m, rad, m, ...). θx = dx/dz
and θy = dy/dz are the slopes of the shaft at the disc; with these slopes the
two planes have the same matrices. The bearings are pinned (they take no
moment), so the slopes at A and B are condensed out of the shaft's two beam
elements (A to the disc and the disc to B, or for an overhung disc the span and
the overhang).

Each journal has a mass and is held to ground by a spring and a damper in x and
in y, which may differ (anisotropic supports) and may differ between A and B,
with cross terms (k_xy, k_yx, c_xy, c_yx) too. Or it runs in a plain journal
bearing, an oil film on a rigid housing (core/fluid_bearing.py): its stiffness
and damping then come from the film, under the bearing's share of the weight,
and change with the speed. The film is linearized about where its journal
settles, so with gravity it also adds a constant load that holds the journal
there (`_film_terms`): the rotor rests with its journals on the film's locus.
So in general the stiffness and the damping are K(Ω) and C(Ω), not only K + ΩH.
Or it floats in an active magnetic bearing (core/magnetic_bearing.py): the
magnets' negative stiffness -k_m is in K, and a PID controller behind an
amplifier sets the current that holds the journal. The controllers have states
of their own, x_c, after q and q' in the state (n_states long); the steady
states take the bearings' dynamic stiffness k_i G(s) at each harmonic instead
(support_stiffness), and with integral action a steady load cannot move the
journals at all.
The disc has mass m, polar moment of inertia Ip (about the spin axis) and
diametral moment Id (about a diameter).

The shaft may also have mass (density ρ > 0), which the Jeffcott model leaves
out. It is then a beam finite element model: `elements` Euler–Bernoulli
elements along it (Rayleigh beams: the sections' rotary inertia and their own
gyroscopic moment included), with a consistent or a lumped mass matrix. Every
node has a displacement and a slope in each plane, so the model has many more
DOFs, and its highest modes are far above anything the page shows (tens of
kHz, which the time stepping would have to resolve). So it is reduced, Craig–
Bampton style: the four stations' DOFs are kept as they are, and the rest of
the shaft moves as their static shape (exactly what the massless shaft does)
plus its lowest few modes with the stations held (the fixed-interface modes,
up to MODE_LIMIT). q is then

    [x_A, x_D, θx, x_B, y_A, y_D, θy, y_B, ξ_x1, ..., ξ_xr, ξ_y1, ..., ξ_yr],

the stations first, as before, then the r modal coordinates of each plane. With
a massless shaft r = 0 and the reduction is exactly the condensation above.

The rotor spins at Ω (rad/s) about +z, turning x toward y. A tilted, spinning
disc has angular momentum Ip Ω along its tilted axis; turning that axis takes a
moment, the gyroscopic moment. It enters the equations of motion through the
velocities:

    M q'' + (C + Ω G) q' + (K + Ω H) q = f(t)

G is skew-symmetric (G^T = -G), so q'^T G q' = 0: the gyroscopic forces do no
work and dissipate nothing, though they sit next to the damping matrix. They
couple the two planes' slopes, and make the natural frequencies depend on Ω:
each whirl mode splits into a forward (with the spin) and a backward (against
it) branch, the Campbell diagram.

Two terms can make the rotor unstable. A bearing (or seal) may have
cross-coupled stiffness: a displacement in x pushes the journal in y and vice
versa, k_xy = -k_yx, so K is not symmetric. In a fluid-film bearing that
coupling comes from the oil being dragged round at a fraction λ of the shaft's
speed, so it grows with the speed, as λ c Ω on top of any constant k_xy (c the
bearing's damping). With λ = 1/2, the forward whirl is fed faster than the
bearing damps it once Ω passes twice its frequency: oil whirl and whip. And the
shaft may have internal
(rotating) damping, which resists the bending rate seen by the spinning shaft,
q' - Ω J q in fixed coordinates (J turns x toward y). Part of it is ordinary
damping in C; the rest is Ω H, a skew stiffness growing with the speed, which
also holds the bearings' λ c. Both
skew stiffnesses push a forward whirl along its path, feeding it energy; above
the speed where that outweighs the damping, the whirl grows by itself.

Angles on the rotor are measured from a keyphasor mark on the shaft, at angle
φ in fixed coordinates (φ' = Ω), in the direction of spin. Three sources turn
with the shaft and drive it once per revolution (1X):

- the disc's mass unbalance U = m e (kg·m): its centre of mass is a distance e
  from the shaft's centre at angle φ + ψ (ψ = 0 by default: the heavy spot is at
  the mark). Keeping the centre of mass on its path takes

      f_x = U (Ω² cos φ + Ω' sin φ),   f_y = U (Ω² sin φ - Ω' cos φ)

  on the disc for ψ = 0, or in general f_x + i f_y = U e^{iψ} (Ω² - iΩ') e^{iφ}.
  Balance weights m_k r_k at angles θ_k add to U e^{iψ} the same way. A weight
  at an axial offset h from the disc's centre plane adds a moment w h too; one
  on the shaft at z is a point force there (two-plane balancing).
- the disc's skew τ (rad): its principal axis is tilted from the shaft's by τ,
  toward angle φ + γ. The disc's inertia acts on its own axis, the shaft's slope
  plus the skew, so the skew moves to the right side as a moment on the slopes,
  M_x + i M_y = (I_d - I_p) τ (Ω² - iΩ') e^{i(φ+γ)}: a couple unbalance. A thin
  disc (I_p > I_d) is pushed flat, toward square to the spin axis.
- the shaft's bow: a residual bend, δ at the disc toward angle φ + β, in the
  shape a load at the disc gives with the bearings held (q_bow). The shaft's
  elastic force is K_shaft (q - q_bow(φ)), so the bow is a force K_shaft q_bow
  turning with the shaft, constant in size. It loads only the shaft: its
  forces on the disc and the journals balance, so at rest the shaft takes its
  bent shape without loading the bearings.

The internal damping acts on q - q_bow too, but the bow does not move in the
shaft, so it adds nothing there: C_i q_bow' + Ω H q_bow = 0.

On a horizontal rotor gravity pulls the disc and the journals down (-y), a
constant force f_g. It moves the equilibrium to the static deflection
q_s = (K + Ω H)^{-1} f_g, about which the rotor vibrates exactly as without it
(the model is linear): the static sag, sideways too where K + ΩH is not
symmetric. The tracking and the spectra work on q - q_s, the vibration.

The shaft may be asymmetric (a two-pole generator's rotor, slotted along two
sides): EI(1 + Δ) bending toward its stiff axis, at angle φ + α, and EI(1 - Δ)
across it. Its stiffness then turns with it, K + ΔK(φ) with ΔK changing at 2X,
and the equations have a periodic coefficient. Turning with the shaft on
isotropic supports they are time-invariant, so its modes are exact there
(_rotating_modes); otherwise they are found by Hill's method (_hill_modes).
Either way they are Floquet modes. Its steady states come by harmonic balance
(harmonic_response): a 1X load stays 1X on isotropic supports, and gravity gives
a mean sag and a 2X whirl (gravity_response), largest near half a critical
speed. Between the critical speeds of its two axes it may be unstable.

The shaft may be cracked (`crack`): a straight-fronted transverse crack, its
mouth toward angle φ + β_c, cut to a depth a from the surface. Where it is open it
is a hinge: the shaft's slope jumps across it by c M (M the bending moment
there), c its compliance from fracture mechanics (crack_compliance), large
bending that opens it and small across. The crack is condensed out exactly, a
rank-one loss of stiffness on the stations (_crack_deficits). A breathing crack
is opened and closed by the shaft's weight alone (weight dominance, as in Mayes
and Davies' model): fully open where the weight's bending puts its mouth in
tension, shut half a turn later, open by (1 + cos)/2 between. So its stiffness
is periodic in φ with 1X, 2X and 3X in it, not only the asymmetric shaft's 2X,
and even on isotropic supports it changes in axes turning with the shaft (the
weight does not turn), so its modes come by Hill's method and its steady states
by harmonic balance over every harmonic. Its weight then drives 1X, 2X and 3X,
the last two most near a half and a third of a critical speed. An open crack
(not breathing, a notch) turns with the shaft as the asymmetric shaft does.

A stator ring (`stator`, a seal or a casing) may sit round the disc, a clearance
larger than it. Once the disc's centre moves past the clearance from the ring's
centre, the ring pushes it back and its friction drags the disc's rim against
its slip over the ring (stator_force). That is not linear in anything, so the
ring is in the time response only; every analysis is the rotor's without it.

Time stepping uses the state z = [q, q'], z' = A(Ω) z + B u, with the same exact
first-order-hold matrix-exponential update as the chain's Simulator. A depends
on Ω, so while the speed ramps A is held at the speed at the middle of each short
chunk of steps (CHUNK_TIME), and the unbalance force follows the exact phase.
An asymmetric or cracked shaft's turning stiffness is split off each step as a
kick to the velocities at mid-step, between two exact half steps of the rest.
The oil films' whole force and a stator ring's are loads found by Newton each
step beside the exact update.

Working on it
-------------
The rotor's stiffness depends on speed: K + ΩH, with H from the shaft's internal damping and the bearings'
fluid swirl (`Bearing.swirl`, k_xy growing as λcΩ) (`RotorSystem.circulatory()`, `speed_dependent`), plus an
oil film's K_f(Ω); its damping too, C + C_f(Ω). K itself is not symmetric once a bearing has cross-coupling
(`kxy`, `kyx`). Use `system.stiffness(omega)` and `system.damping(omega)` wherever the speed is known, not
`matrices()` alone (it leaves the films out). `core/rotor_stability.py` finds the onset speed by sweeping and
bisecting the least damped mode's ζ.

The disc may sit between the bearings or overhang one (`position` > 1 or < 0, never on a bearing). The shaft's
nodes come from `RotorSystem._mesh` in axial order, and the 3D view spans `system.extent`, so don't assume the
disc is between 0 and L. Presets live in `core/rotor_presets.py`, and `tests/test_rotor_presets.py` checks the
numbers in their notes.

The shaft may have mass (`density` > 0): it is then a beam FE model (`elements`, consistent or `lumped` mass),
reduced Craig–Bampton style to the stations plus the shaft's fixed-interface modes up to `MODE_LIMIT`
(`RotorSystem._reduced`). So q is `[the 8 station DOFs, ξ_x, ξ_y]`: `XA`…`YB` keep their indices, but q has
`system.n_dof` entries. Size arrays from `n_dof` (not 8 or `N_DOF`), split or build the planes with
`plane_dofs`/`merge`/`plane_load`, and get the shaft's shape anywhere from `deflection`. M is then full
(`mass_inverse`). With density 0 the model is exactly the massless Jeffcott one whatever `elements` is (a test
checks); keep it so. Raising `MODE_LIMIT` costs time: the time step resolves the fastest mode kept.

Everything that turns with the rotor (unbalance, balance weights, disc skew, shaft bow) goes through
`RotorSystem.rotating_load(Ω, Ω')`, as f_x + i f_y = P e^{iφ} on [A, D, θ, B]; both the simulator's `force`
and `synchronous_response` use it, so the steady state and the time response can't drift apart. A new 1X
source belongs there. Two-plane balancing's `plane_weights` (`PlaneWeight`) go there too: on the disc they add
to `mass_unbalance` and `couple_unbalance` (a weight at axial offset h adds w·h to the couple); on the shaft
each is a point force S(z)ᵀw. The Balancing tab keeps its readings at all four stations, and forgets its trial
runs when the planes change, since the influence matrix belongs to them. φ is the keyphasor mark's angle, not
the heavy spot's: every rotor angle (unbalance, weights, bow, skew) is measured from the mark, and the heavy
spot drawn is `mass_unbalance`'s.

An asymmetric shaft (`asymmetry` Δ, EI(1 ± Δ), stiff axis at `asymmetry_angle` from the mark) has a stiffness
that turns with it: `stiffness(Ω)` is its mean, `stiffness_at(Ω, φ)` the whole thing at an instant. Its modes
are Floquet modes (`_rotating_modes`, exact in axes turning with the shaft, on isotropic supports;
`_hill_modes` otherwise), its steady states come by harmonic balance (`harmonic_response`:
`synchronous_response`'s 1X, `gravity_response`'s mean sag and 2X), and the simulator splits each step around
a mid-step kick from the turning part (exact FOH with K held when it stands still). Its criticals are its two
axis rotors' (`axis_system`), and `Onset.end` is where its unstable band ends. A round shaft takes none of
these paths; tests check that Δ → 0 matches it.

A cracked shaft (`crack`, a `Crack`) is a hinge at the crack (compliance from `crack_compliance`, by depth),
condensed exactly onto the stations (`_hinge_loss`). Breathing, the weight opens it (`crack_sign`,
`crack_open(φ)`), so its stiffness has 1X, 2X and 3X; held open it turns with the shaft like the asymmetric
one. Both are one Fourier series, `_periodic_terms` (the crack's mean goes in K, the harmonics E_k turn): test
`system.periodic`, not `asymmetry`, for anything that turns; `spacing` is 1 for a breathing crack (harmonic
balance then couples every harmonic, Hill's method steps by Ω) and `rotating_frame` says when the
rotating-frame modes are exact. `steady_harmonics` gives the whole steady state (gravity and 1X loads
together, 1X–3X), which the Bode plot and the tracker (`vectors3`) compare. A massless shaft's crack does not
depend on `elements` (a test checks).

Gravity (`RotorSystem.gravity`) only moves the rotor's equilibrium to `static_deflection(Ω)`. Anything that
measures vibration (the 1X tracker, peaks, the spectrum, the phase readout) works on `vibration(system, q, Ω)`
(`core/rotor_sweep.py`), the motion about it; only the drawing can show the sag (the page's *Static sag* box).
On an asymmetric shaft gravity also drives 2X: the sag is then the mean of a periodic motion, and the 2X stays
in the vibration.
"""

from __future__ import annotations

import itertools
import math
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import cached_property, lru_cache
from typing import NamedTuple

import numpy as np
import scipy.linalg

from .fluid_bearing import BearingState, JournalBearing
from .magnetic_bearing import MagneticBearing

N_DOF = 8  # the stations' DOFs, the first of q (a shaft with mass adds its modal coordinates after them)
# Index of each station DOF in q.
XA, XD, TX, XB, YA, YD, TY, YB = range(N_DOF)
DOF_NAMES = ("x_A", "x_D", "θx", "x_B", "y_A", "y_D", "θy", "y_B")
STEEL = 7850.0  # kg/m³
MODE_LIMIT = 1000.0  # Hz: the shaft's fixed-interface modes are kept up to this frequency (the top speed: 60,000 rpm)
MAX_MODES = 12  # and at most this many in each plane
MIN_GAP = 1e-3  # of the span: the closest the disc may come to a bearing
POISSON = 0.3  # the shaft's Poisson's ratio, for a crack's compliance
CRACK_DEPTH_MAX = 0.5  # a / D: the strip model's crack compliance holds to half the shaft's diameter
# Rub on a stator ring (RotorSystem.stator), see RotorSimulator._rub_steps:
SLIP = 0.01  # m/s: the rim's friction reaches 0.71 μ N at this slip (s/√(s² + SLIP²), smoothed Coulomb)


def _edge_crack_factor(s: np.ndarray, bending: bool) -> np.ndarray:
    """F(a/h): K = σ √(πa) F for an edge crack of depth a in a strip of height h (Tada's fits, 0.5%)."""
    x = 0.5 * math.pi * s
    root = np.sqrt(np.tan(x) / x) / np.cos(x)
    if bending:
        return root * (0.923 + 0.199 * (1.0 - np.sin(x)) ** 4)
    return root * (0.752 + 2.02 * s + 0.37 * (1.0 - np.sin(x)) ** 3)


# Speeds a system keeps its films' states for (an analysis refresh asks for ~5,000), and their
# terms (whole matrices; a sweep asks for each a few times in a row).
CACHE_STATES = 8192
CACHE_TERMS = 64


def _cached[T](system: RotorSystem, name: str, key: float, make: Callable[[], T], size: int) -> T:
    """`make()` for speed `key`, kept in the system's cache `name`, the least recently used dropped first."""
    cache = system.__dict__.setdefault(name, OrderedDict())
    try:
        cache.move_to_end(key)
        return cache[key]
    except KeyError:
        value = cache[key] = make()
        if len(cache) > size:
            cache.popitem(last=False)
        return value


@lru_cache(maxsize=64)
def crack_compliance(depth: float) -> tuple[float, float]:
    """A straight-fronted crack's bending compliance in a round shaft, per (1 - ν²)/(E R³): (opening, across).

    Its depth is a/D, from the surface (Dimarogonas and Papadopoulos). The crack is
    cut into strips across its front, each an edge crack in a strip as tall as the
    shaft there; the energy released as the crack grows, ∫ (1 - ν²) K²/E dA, gives
    the extra rotation c M across the crack under a bending moment M. Bending that
    pulls the crack's mouth open (the moment along its front) loads every strip in
    bending; across, each strip is in tension or compression by its distance off
    the middle. So the crack gives much more to the first, until it is deep.
    """
    if depth <= 0.0:
        return 0.0, 0.0
    radius, a = 1.0, 2.0 * depth
    half = math.sqrt(a * (2.0 * radius - a))  # half the crack front's length
    inertia = math.pi * radius**4 / 4.0
    x, w = np.polynomial.legendre.leggauss(96)
    xi, wx = half * x, half * w
    chord = np.sqrt(radius**2 - xi**2)
    height = 2.0 * chord  # each strip's height, and its crack's depth
    deep = chord - (radius - a)
    alpha = 0.5 * deep[:, None] * (x[None, :] + 1.0)
    wa = 0.5 * deep[:, None] * w[None, :]
    s = alpha / height[:, None]
    opening = (0.5 * height[:, None] / inertia) ** 2 * math.pi * alpha * _edge_crack_factor(s, True) ** 2
    across = (xi[:, None] / inertia) ** 2 * math.pi * alpha * _edge_crack_factor(s, False) ** 2
    return float(2.0 * wx @ np.sum(wa * opening, axis=1)), float(2.0 * wx @ np.sum(wa * across, axis=1))


@dataclass(frozen=True)
class Crack:
    """A transverse crack in the shaft, at z = position L, its mouth at `angle` from the keyphasor mark.

    Breathing, it is opened and closed by the shaft's weight as it turns; not
    breathing, it is held open (as a sawn notch). `compliance` (rad/(N·m), opening
    and across) overrides the one its depth gives.
    """

    position: float = 0.5  # z / L, as the disc's position (past 0 or 1 on an overhang)
    depth: float = 0.3  # a / D, from the shaft's surface
    angle: float = 0.0  # rad, of the crack's mouth from the keyphasor mark, in the direction of spin
    breathing: bool = True
    compliance: tuple[float, float] | None = None


@dataclass(frozen=True)
class Stator:
    """A rigid ring round the disc (a seal or a casing), which the disc's rim rubs once it reaches it.

    The ring is `clearance` larger in radius than the disc, its centre `offset`
    from the bearings' line toward `offset_angle` (fixed axes, from +x toward +y;
    -90°: down). Where the disc's centre is further than the clearance from the
    ring's centre, by δ, the ring pushes it back with N = k δ + c δ' (never
    pulling), and drags its rim against the slip with μ N, the slip being the rim's
    speed over the ring, R Ω plus the disc's whirl along the ring.

    The drag heats the shaft where it rubs, and a shaft hotter on one side bows
    toward it (the Newkirk effect): with a `thermal_gain` β, the bow at the disc
    settles to β times the rub's heat, in the direction of the hot spot on the
    shaft, lagging it by the `thermal_lag` τ (a first-order lag, τ b' = β H - b).
    """

    clearance: float = 2e-4  # m, radial
    stiffness: float = 2e6  # N/m, the ring's contact stiffness
    damping: float = 100.0  # N·s/m, in contact
    friction: float = 0.1  # μ, the rim on the ring
    offset: float = 0.0  # m, of the ring's centre from the bearings' line
    offset_angle: float = -0.5 * math.pi  # rad, its direction in fixed axes
    thermal_gain: float = 0.0  # m/W: the bow at the disc per watt of rub heat (0: no thermal bow)
    thermal_lag: float = 10.0  # s, the bow's time constant

    @property
    def centre(self) -> tuple[float, float]:
        """The ring's centre (x, y), m."""
        return self.offset * math.cos(self.offset_angle), self.offset * math.sin(self.offset_angle)


def stator_force(
    stator: Stator, radius: float, x: float, y: float, vx: float, vy: float, omega: float
) -> tuple[float, float, float, list[float]]:
    """(f_x, f_y, N, J): the ring's force on the disc at (x, y) moving at (x', y'), the normal force N,
    and the force's derivatives J = d(f_x, f_y)/d(x, y, x', y') (row by row, 8 numbers).

    The friction is Coulomb's smoothed over SLIP: μ N s/√(s² + SLIP²), s = R Ω + v_t,
    the rim's slip along the ring (v_t the disc's velocity along it, in the direction of spin).
    """
    cx, cy = stator.centre
    dx, dy = x - cx, y - cy
    rho = math.hypot(dx, dy)
    pen = rho - stator.clearance
    if pen <= 0.0 or rho == 0.0:
        return 0.0, 0.0, 0.0, [0.0] * 8
    nx, ny = dx / rho, dy / rho  # out from the ring's centre
    tx, ty = -ny, nx  # along the ring, in the direction of spin
    vn, vt = nx * vx + ny * vy, tx * vx + ty * vy
    normal = stator.stiffness * pen + stator.damping * vn
    if normal <= 0.0:  # pulling away faster than the ring springs back: no contact
        return 0.0, 0.0, 0.0, [0.0] * 8
    s = radius * omega + vt
    root = math.sqrt(s * s + SLIP * SLIP)
    sat, dsat = s / root, SLIP * SLIP / root**3
    mu = stator.friction
    gx, gy = nx + mu * sat * tx, ny + mu * sat * ty  # f = -N g
    # n and t turn as the disc moves round the ring: dn/dd = t tᵀ/ρ, dt/dd = -n tᵀ/ρ.
    # dδ'/dd = vᵀ t tᵀ/ρ = (v_t/ρ) tᵀ, dv_t/dd = -(v_n/ρ) tᵀ.
    k, c = stator.stiffness, stator.damping
    dN = (k * nx + c * vt / rho * tx, k * ny + c * vt / rho * ty, c * nx, c * ny)  # dN/d(x, y, x', y')
    ds = (-vn / rho * tx, -vn / rho * ty, tx, ty)  # ds/d(x, y, x', y')
    J: list[float] = []
    for g, n_, t_ in ((gx, nx, tx), (gy, ny, ty)):
        # dg/dd = (t_ - μ σ n_) tᵀ/ρ + μ t_ σ' ds/dd;  dg/dv = μ t_ σ' ds/dv
        turn = (t_ - mu * sat * n_) / rho
        dg = (
            turn * tx + mu * t_ * dsat * ds[0],
            turn * ty + mu * t_ * dsat * ds[1],
            mu * t_ * dsat * ds[2],
            mu * t_ * dsat * ds[3],
        )
        J.extend(-(g * dN[i] + normal * dg[i]) for i in range(4))
    return -normal * gx, -normal * gy, normal, J


@dataclass(frozen=True)
class Bearing:
    """A bearing journal on its support: springs and dampers to ground in x and y, an oil film or a magnet.

    The support pushes back on the journal with f = -K [x, y] - C [x', y'], K =
    [[kx, k_xy], [k_yx, ky]] and C = [[cx, c_xy], [c_yx, cy]]. With a `film` the
    journal runs in a plain journal bearing on a rigid housing instead: K and C
    are then the film's, at the speed and under the bearing's share of the
    rotor's weight (the film's own `load` is not used), and the springs, dampers
    and swirl are left out. With a `magnet` it floats in an active magnetic
    bearing (core/magnetic_bearing.py): K is then the magnets' negative stiffness
    -k_m, and the rest of the push comes from the controller's states.
    """

    kx: float = 5e4  # N/m
    ky: float = 5e4  # N/m
    cx: float = 100.0  # N·s/m
    cy: float = 100.0  # N·s/m
    mass: float = 0.1  # kg, the journal and whatever moves with it
    kxy: float = 0.0  # N/m, cross-coupled stiffness: f_x = -k_xy y
    swirl: float = 0.0  # fluid swirl ratio λ: the cross-coupling grows with speed by λ c Ω (see damping)
    kyx: float | None = None  # N/m: f_y = -k_yx x; None for -k_xy (a pure skew coupling, as a seal's)
    cxy: float = 0.0  # N·s/m: f_x = -c_xy y'
    cyx: float = 0.0  # N·s/m: f_y = -c_yx x'
    film: JournalBearing | None = None  # an oil film in place of the springs and dampers
    magnet: MagneticBearing | None = None  # or an active magnetic bearing

    def __post_init__(self) -> None:
        if self.film is not None and self.magnet is not None:
            raise ValueError("a bearing is an oil film or a magnet, not both")

    @property
    def damping(self) -> float:
        """c (N·s/m), the mean of c_x and c_y: the fluid's drag-round takes k_xy up by λ c Ω."""
        return 0.5 * (self.cx + self.cy)

    @property
    def k_yx(self) -> float:
        return -self.kxy if self.kyx is None else self.kyx

    @property
    def swirling(self) -> bool:
        """True if the springs' cross-coupling grows with the speed (λ c ≠ 0, and no film or magnet)."""
        return self.film is None and self.magnet is None and bool(self.swirl * self.damping)

    def springs(self) -> tuple[np.ndarray, np.ndarray]:
        """(K, C), each 2 x 2: the springs and dampers at rest (none with a film; a magnet's -k_m)."""
        if self.film is not None:
            return np.zeros((2, 2)), np.zeros((2, 2))
        if self.magnet is not None:
            return -self.magnet.negative_stiffness * np.eye(2), np.zeros((2, 2))
        return (
            np.array([[self.kx, self.kxy], [self.k_yx, self.ky]]),
            np.array([[self.cx, self.cxy], [self.cyx, self.cy]]),
        )


class PlaneWeight(NamedTuple):
    """A balance weight in a plane at axial position z (m from bearing A): m r e^{iθ} (kg·m) from the mark.

    On the disc (`disc`), it sits on the rigid disc at an axial offset z - a from its
    centre plane, so it adds w to the disc's unbalance and w (z - a) to its couple.
    On the shaft, it is a point force on the shaft at z.
    """

    z: float
    weight: complex
    disc: bool = False


@dataclass(frozen=True)
class RotorSystem:
    """A Jeffcott rotor: shaft, disc, two bearings and the disc's unbalance. SI units."""

    length: float = 0.5  # m, between the bearings
    diameter: float = 0.01  # m, of the shaft
    youngs: float = 200e9  # Pa
    position: float = 0.5  # disc position a / L (0.5: midspan; > 1 overhung past B, < 0 past A)
    disc_mass: float = 1.0  # kg
    ip: float = 1.25e-3  # kg·m², polar (about the spin axis)
    id: float = 6.25e-4  # kg·m², diametral (about a diameter)
    bearing_a: Bearing = field(default_factory=Bearing)
    bearing_b: Bearing = field(default_factory=Bearing)
    unbalance: float = 20e-6  # kg·m (20 g·mm)
    unbalance_angle: float = 0.0  # rad, ψ: the heavy spot's angle from the keyphasor mark
    internal_damping: float = 0.0  # N·s/m, the shaft's rotating damping, as a damper at the disc
    bow: float = 0.0  # m, the shaft's residual bow at the disc
    bow_angle: float = 0.0  # rad, of the bow from the keyphasor mark (in the direction of spin)
    skew: float = 0.0  # rad, the disc's principal axis's tilt from the shaft's
    skew_angle: float = 0.0  # rad, of the skew's direction from the keyphasor mark
    weights: tuple[complex, ...] = ()  # balance weights on the disc, each m r e^{iθ} (kg·m) from the mark
    plane_weights: tuple[PlaneWeight, ...] = ()  # weights in planes along the rotor (two-plane balancing)
    gravity: float = 0.0  # m/s², pulling in -y (9.81 for a horizontal rotor; 0 for a vertical one)
    density: float = 0.0  # kg/m³ of the shaft: 0 for the Jeffcott model's massless shaft (steel: STEEL)
    elements: int = 8  # beam elements along the shaft, shared between its stretches by length
    lumped: bool = False  # the shaft's mass lumped at the nodes, rather than the consistent mass matrix
    asymmetry: float = 0.0  # Δ: the shaft's EI is EI(1 + Δ) bending toward its stiff axis, EI(1 - Δ) across it
    asymmetry_angle: float = 0.0  # rad, α: the stiff axis's angle from the keyphasor mark
    crack: Crack | None = None  # a transverse crack in the shaft
    nonlinear_films: bool = False  # the time response takes the oil films' whole force, not their linearization
    stator: Stator | None = None  # a ring round the disc that it rubs on (time response only)

    def __post_init__(self) -> None:
        if min(abs(self.position), abs(self.position - 1.0)) < MIN_GAP:
            raise ValueError("the disc cannot sit on a bearing (a/L = 0 or 1)")
        positive = (
            self.length,
            self.diameter,
            self.youngs,
            self.disc_mass,
            self.id,
            self.bearing_a.mass,
            self.bearing_b.mass,
        )
        if min(positive) <= 0.0:
            raise ValueError("lengths, Young's modulus, masses and Id must be positive")
        if self.internal_damping < 0.0:
            raise ValueError("the internal damping cannot be negative")
        if min(self.bearing_a.swirl, self.bearing_b.swirl) < 0.0:
            raise ValueError("the fluid swirls with the shaft: the swirl ratio cannot be negative")
        if self.gravity < 0.0:
            raise ValueError("gravity pulls down (-y): give its size")
        if self.bow < 0.0 or self.skew < 0.0:
            raise ValueError("the bow and the skew are sizes; give their direction as an angle")
        if self.density < 0.0:
            raise ValueError("the shaft's density cannot be negative")
        if not 0.0 <= self.asymmetry < 1.0:
            raise ValueError("the asymmetry Δ is in [0, 1): EI(1 - Δ) about the soft axis must stay positive")
        if self.elements < 2:
            raise ValueError("the shaft needs at least two elements, one each side of the disc")
        lo, hi = self.extent
        if any(not p.disc and not lo <= p.z <= hi for p in self.plane_weights):
            raise ValueError("balance weights go on the shaft, between its ends")
        if self.crack is not None:
            if not 0.0 <= self.crack.depth <= CRACK_DEPTH_MAX:
                raise ValueError(f"the crack's depth a/D is in [0, {CRACK_DEPTH_MAX}]")
            if not lo < self.crack.position * self.length < hi:
                raise ValueError("the crack is in the shaft, between its ends")
        if self.stator is not None:
            st = self.stator
            if st.clearance <= 0.0 or st.stiffness <= 0.0:
                raise ValueError("the stator's clearance and contact stiffness must be positive")
            if min(st.damping, st.friction, st.offset, st.thermal_gain) < 0.0:
                raise ValueError("the stator's damping, friction, offset and thermal gain cannot be negative")
            if st.thermal_lag <= 0.0:
                raise ValueError("the thermal bow's time constant must be positive")
            if self.periodic or self.whole_films:
                raise ValueError("a stator rubs only on a round shaft, with its oil films linearized")

    def with_(self, **changes) -> RotorSystem:
        return replace(self, **changes)

    def on_disc(self, p: PlaneWeight) -> bool:
        """True if the plane weight p is on the disc: put there, or on the shaft at the disc's centre plane."""
        return p.disc or abs(p.z - self.a) <= 1e-9 * self.length

    @property
    def mass_unbalance(self) -> complex:
        """The disc's net mass unbalance (kg·m), the unbalance and the weights: |·| m e, its angle the heavy spot's.

        Plane weights on the disc count, whatever their axial offset.
        """
        return (
            self.unbalance * complex(math.cos(self.unbalance_angle), math.sin(self.unbalance_angle))
            + sum(self.weights, 0j)
            + sum((p.weight for p in self.plane_weights if self.on_disc(p)), 0j)
        )

    @property
    def couple_unbalance(self) -> complex:
        """The disc's net couple unbalance (kg·m²), turning with the shaft: the skew's and the plane weights'.

        Its moment on the disc's slopes is this times (Ω² - iΩ') e^{iφ}.
        """
        return (self.id - self.ip) * self.skew * np.exp(1j * self.skew_angle) + sum(
            (p.weight * (p.z - self.a) for p in self.plane_weights if self.on_disc(p)), 0j
        )

    # ------------------------------------------------------------- the shaft
    @property
    def ei(self) -> float:
        """Bending stiffness EI (N·m²) of the round shaft."""
        return self.youngs * math.pi * self.diameter**4 / 64.0

    @cached_property
    def bow_load(self) -> np.ndarray:
        """One plane's load (4 + r,) of a unit bow at the disc on a round shaft, K_shaft q_bow per metre."""
        return self.shaft_stiffness() @ self.bow_shape()

    @property
    def disc_radius(self) -> float:
        """The disc's radius (m), a uniform thin disc's from its mass and I_p = m R²/2."""
        return math.sqrt(2.0 * self.ip / self.disc_mass)

    @property
    def a(self) -> float:
        """Disc position from bearing A (m)."""
        return self.position * self.length

    @property
    def b(self) -> float:
        """Disc position from bearing B (m), L - a: negative when the disc overhangs B."""
        return self.length - self.a

    @property
    def overhung(self) -> bool:
        """True if the disc sits outside the bearings rather than between them."""
        return not 0.0 < self.position < 1.0

    @property
    def overhang(self) -> float:
        """The disc's distance past the nearer bearing (m), 0 between the bearings."""
        return max(-self.a, self.a - self.length, 0.0)

    @property
    def area(self) -> float:
        """The shaft's cross-section (m²)."""
        return math.pi * self.diameter**2 / 4.0

    @property
    def shaft_mass(self) -> float:
        """The whole shaft's mass (kg), 0 for the Jeffcott model's massless shaft."""
        lo, hi = self.extent
        return self.density * self.area * (hi - lo)

    @cached_property
    def _mesh(self) -> tuple[np.ndarray, tuple[int, int, int]]:
        """The shaft's nodes' z (m) in axial order, and the nodes at bearing A, the disc and bearing B.

        Each stretch between the stations gets elements in proportion to its length, at
        least one. A crack has a node of its own, unless it is next to a station.
        """
        stations = [(0.0, 0), (self.a, 1), (self.length, 2)]
        if self.crack is not None and min(abs(self.crack_z - z) for z, _ in stations) > MIN_GAP * self.length:
            stations.append((self.crack_z, 3))
        stations.sort()
        span = stations[-1][0] - stations[0][0]
        z, node = [stations[0][0]], {stations[0][1]: 0}
        for (z0, _), (z1, n1) in itertools.pairwise(stations):
            k = max(1, round(self.elements * (z1 - z0) / span))
            z.extend(np.linspace(z0, z1, k + 1)[1:])
            node[n1] = len(z) - 1
        return np.array(z), (node[0], node[1], node[2])

    @property
    def crack_z(self) -> float:
        """The crack's axial position (m from bearing A); NaN without one."""
        return math.nan if self.crack is None else self.crack.position * self.length

    @cached_property
    def _assembled(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(K, M, R): one plane's whole shaft, on every node's [w, θ] (see _reduced)."""
        z = self._mesh[0]
        n = 2 * z.size
        K, M, R = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        rho_a, rho_i = self.density * self.area, self.density * math.pi * self.diameter**4 / 64.0
        for e, length in enumerate(np.diff(z)):
            d = slice(2 * e, 2 * e + 4)
            translation, rotation = beam_mass(rho_a, rho_i, length, self.lumped)
            K[d, d] += beam_stiffness(self.ei, length)
            M[d, d] += translation + rotation
            R[d, d] += rotation
        return K, M, R

    @property
    def _station_dofs(self) -> list[int]:
        """The stations' [w_A, w_D, θ_D, w_B] in the whole shaft's DOFs."""
        na, nd, nb = self._mesh[1]
        return [2 * na, 2 * nd, 2 * nd + 1, 2 * nb]

    @property
    def crack_compliances(self) -> tuple[float, float]:
        """The crack's compliance (rad/(N·m)): to bending that opens it, and across; zeros without one."""
        if self.crack is None:
            return 0.0, 0.0
        if self.crack.compliance is not None:
            return self.crack.compliance
        scale = (1.0 - POISSON**2) / (self.youngs * (0.5 * self.diameter) ** 3)
        c_open, c_across = crack_compliance(self.crack.depth)
        return c_open * scale, c_across * scale

    @cached_property
    def _crack_hinge(self) -> tuple[np.ndarray, float]:
        """(b (4,), s): the crack as a hinge, a jump Δθ in the shaft's slope there, condensed onto the stations.

        The shaft's energy on [stations, Δθ], everything else free (statically), is
        ½ [q; Δθ]^T [[K_ss, b], [b^T, s]] [q; Δθ]: b is the bending moment at the
        crack the stations' displacements make (per unit of each), s the shaft's
        stiffness against the jump with the stations held.
        """
        z, _ = self._mesh
        K = self._assembled[0]
        n = K.shape[0]
        node = int(np.argmin(np.abs(z - self.crack_z)))
        # The jump turns the element past the crack at its near end (or, at the shaft's end, the one before it).
        e, j = (node, 1) if node < z.size - 1 else (node - 1, 3)
        ke = beam_stiffness(self.ei, z[e + 1] - z[e])
        g = np.arange(2 * e, 2 * e + 4)
        Kx = np.zeros((n + 1, n + 1))
        Kx[:n, :n] = K
        Kx[n, g] += ke[j]
        Kx[g, n] += ke[:, j]
        Kx[n, n] += ke[j, j]
        kept = self._station_dofs + [n]
        inner = [i for i in range(n + 1) if i not in kept]
        Kc = Kx[np.ix_(kept, kept)] - Kx[np.ix_(kept, inner)] @ np.linalg.solve(
            Kx[np.ix_(inner, inner)], Kx[np.ix_(inner, kept)]
        )
        return Kc[:4, 4].copy(), float(Kc[4, 4])

    def _hinge_loss(self, compliance: float) -> np.ndarray:
        """One plane's stiffness lost (4 + r square) to a hinge of this compliance at the crack.

        Condensing the jump out, the stations' stiffness is K_ss - b b^T c/(1 + c s):
        exact for the massless shaft. With the shaft's mass its fixed-interface modes
        are those of the uncracked shaft (the crack's own inertia is left out).
        """
        loss = np.zeros((4 + self.modes_kept,) * 2)
        if compliance > 0.0:
            b, s = self._crack_hinge
            loss[:4, :4] = np.outer(b, b) * compliance / (1.0 + compliance * s)
        return loss

    @cached_property
    def crack_sign(self) -> int:
        """Where the weight opens a breathing crack: +1 its mouth down (the shaft sags there), -1 up, 0 never.

        The sign of the shaft's curvature at the crack under its weight, the journals
        held (the bending moment of a shaft on two bearings does not depend on them).
        """
        if self.crack is None or not self.gravity:
            return 0
        z, (na, nd, nb) = self._mesh
        K, M, _ = self._assembled
        n = K.shape[0]
        f = -M @ np.tile([1.0, 0.0], z.size)
        f[2 * nd] -= self.disc_mass
        free = [i for i in range(n) if i not in (2 * na, 2 * nb)]
        u = np.zeros(n)
        u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
        node = int(np.argmin(np.abs(z - self.crack_z)))
        e, end = (node, 0.0) if node < z.size - 1 else (node - 1, 1.0)
        length = z[e + 1] - z[e]
        # w'' of the element's cubic at its end, from its [w1, θ1, w2, θ2].
        curve = np.array([12 * end - 6, length * (6 * end - 4), 6 - 12 * end, length * (6 * end - 2)]) / length**2
        w2 = float(curve @ u[2 * e : 2 * e + 4])
        scale = float(np.abs(curve) @ np.abs(u[2 * e : 2 * e + 4])) + 1e-300
        return 0 if abs(w2) <= 1e-9 * scale else (1 if w2 > 0 else -1)

    @property
    def crack_opening(self) -> float:
        """How open the crack is on average over a turn: 1 held open, ½ breathing, 0 shut (or none)."""
        if self.crack is None or (not self.crack.depth and self.crack.compliance is None):
            return 0.0
        if not self.crack.breathing:
            return 1.0
        return 0.5 if self.crack_sign else 0.0

    def crack_open(self, phase: float | np.ndarray) -> np.ndarray:
        """How open the crack is (0 to 1) with the keyphasor mark at angle φ: (1 - s sin θ)/2, θ its mouth's angle."""
        crack = self.crack
        if crack is None or not self.crack_opening:
            return np.zeros(np.shape(phase))
        if not crack.breathing:
            return np.ones(np.shape(phase))
        return 0.5 * (1.0 - self.crack_sign * np.sin(np.asarray(phase, dtype=float) + crack.angle))

    @property
    def nodes(self) -> np.ndarray:
        """z (m) of the shaft's finite element nodes, in axial order."""
        return self._mesh[0].copy()

    @cached_property
    def _reduced(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(T, K, M, R, freqs): one plane's shaft, reduced to the stations and its fixed-interface modes.

        The beam elements are assembled on every node's [w, θ]. The stations' DOFs
        [w_A, w_D, θ_D, w_B] are kept; the others (the slopes at the bearings, and
        every node in between) are the static shape the stations give them,
        -K_ii^{-1} K_ib, plus the modes of the shaft with the stations held, mass
        normalized, up to MODE_LIMIT: full = T @ [stations, ξ]. K, M and R are
        T^T (·) T for the shaft's stiffness, mass and rotary inertia (its
        gyroscopic matrix is 2R: a round section's polar inertia is twice its
        diametral). freqs (Hz) are the modes kept.
        """
        K, M, R = self._assembled
        n = K.shape[0]
        kept = self._station_dofs
        inner = [j for j in range(n) if j not in kept]
        static = -np.linalg.solve(K[np.ix_(inner, inner)], K[np.ix_(inner, kept)])
        if self.density > 0.0:
            w2, shapes = scipy.linalg.eigh(K[np.ix_(inner, inner)], M[np.ix_(inner, inner)])
            r = min(MAX_MODES, int(np.sum(w2 <= (2.0 * math.pi * MODE_LIMIT) ** 2)))
            w2, shapes = w2[:r], shapes[:, :r]
            shapes *= np.sign(shapes[np.argmax(np.abs(shapes), axis=0), np.arange(r)])  # largest entry positive
        else:
            w2, shapes = np.empty(0), np.empty((len(inner), 0))
        T = np.zeros((n, 4 + w2.size))
        T[kept, :4] = np.eye(4)
        T[inner, :4] = static
        T[inner, 4:] = shapes
        return T, T.T @ K @ T, T.T @ M @ T, T.T @ R @ T, np.sqrt(w2) / (2.0 * math.pi)

    @property
    def modes_kept(self) -> int:
        """r, the shaft's fixed-interface modes kept in each plane (0 for a massless shaft)."""
        return self._reduced[0].shape[1] - 4

    @property
    def fixed_interface_freqs(self) -> np.ndarray:
        """The frequencies (Hz) of the shaft's modes kept, with the stations held."""
        return self._reduced[4].copy()

    @property
    def n_dof(self) -> int:
        """The length of q: the 8 stations' DOFs and the shaft's modal coordinates in each plane."""
        return N_DOF + 2 * self.modes_kept

    @cached_property
    def _planes(self) -> tuple[np.ndarray, np.ndarray]:
        r = self.modes_kept
        x = np.r_[0:4, N_DOF : N_DOF + r]
        return x, np.r_[4:8, N_DOF + r : N_DOF + 2 * r]

    @property
    def plane_dofs(self) -> tuple[np.ndarray, np.ndarray]:
        """The indices in q of the x plane's DOFs and the y plane's, each [stations, modal coordinates]."""
        x, y = self._planes
        return x.copy(), y.copy()

    def merge(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """q (..., n_dof) from the two planes' parts (..., 4 + r): the inverse of q[..., plane_dofs]."""
        ix, iy = self._planes
        x, y = np.asarray(x), np.asarray(y)
        out = np.empty(np.broadcast_shapes(x.shape, y.shape)[:-1] + (self.n_dof,), dtype=np.result_type(x, y))
        out[..., ix], out[..., iy] = x, y
        return out

    def _blocks(
        self, xx: np.ndarray, yy: np.ndarray, xy: np.ndarray | None = None, yx: np.ndarray | None = None
    ) -> np.ndarray:
        """An n_dof square matrix from its plane blocks: x on x, y on y, x's row on y's column, and back."""
        ix, iy = self._planes
        A = np.zeros((self.n_dof, self.n_dof), dtype=np.result_type(xx, yy))
        A[np.ix_(ix, ix)], A[np.ix_(iy, iy)] = xx, yy
        if xy is not None:
            A[np.ix_(ix, iy)] = xy
        if yx is not None:
            A[np.ix_(iy, ix)] = yx
        return A

    def shaft_stiffness(self) -> np.ndarray:
        """One plane's shaft stiffness on [w_A, w_D, θ_D, w_B, ξ_1, ..., ξ_r]: 4x4 for a massless shaft.

        The stations' block is the shaft's exact static stiffness (it does not depend
        on the elements); each fixed-interface mode adds its ω² alone on the diagonal.
        """
        return self._reduced[1].copy()

    def plane_mass(self) -> np.ndarray:
        """One plane's mass matrix on [stations, ξ]: the shaft's, the journals', the disc's m and I_d."""
        M = self._reduced[2].copy()
        M[[0, 1, 2, 3], [0, 1, 2, 3]] += [self.bearing_a.mass, self.disc_mass, self.id, self.bearing_b.mass]
        return M

    @property
    def disc_stiffness(self) -> float:
        """The shaft's stiffness at the disc with the bearings held (N/m).

        3EIL/(a²b²) between the bearings; 3EI/(c²(L + c)) for a disc overhung by c.
        """
        k = self.shaft_stiffness()[1:3, 1:3]
        return 1.0 / np.linalg.inv(k)[0, 0]

    @property
    def loss_time(self) -> float:
        """η (s): the internal damping is η K_shaft, so that it is c_i on the disc's own stiffness."""
        return self.internal_damping / self.disc_stiffness

    def bow_shape(self) -> np.ndarray:
        """One plane's bow (4 + r,) on [w_A, w_D, θ_D, w_B, ξ], per metre of bow at the disc.

        The shape a load at the disc gives with the bearings held: no moment at the
        disc, so K_shaft times it is a force at the disc and the bearings' reactions.
        It is a static shape, so the modal coordinates are zero.
        """
        k = self.shaft_stiffness()[1:3, 1:3]
        w, theta = np.linalg.solve(k, [1.0, 0.0])
        shape = np.zeros(4 + self.modes_kept)
        shape[1:3] = 1.0, theta / w
        return shape

    def rotating_load(self, omega: float | np.ndarray, alpha: float | np.ndarray = 0.0) -> np.ndarray:
        """The 1X load on one plane's DOFs as f_x + i f_y = P e^{iφ}: P (..., 4 + r) complex.

        The unbalance and the balance weights on the disc, its couple (the skew's and
        the weights' moment on its slopes), the weights on the shaft (each a point
        force, shared over the DOFs as S(z)^T) and the bow's force K_shaft q_bow, at
        speed Ω and acceleration Ω' = alpha.
        """
        drive = np.asarray(omega, dtype=float) ** 2 - 1j * np.asarray(alpha, dtype=float)
        per_drive, bow = self._load_parts
        return drive[..., None] * per_drive + bow

    @cached_property
    def _load_parts(self) -> tuple[np.ndarray, np.ndarray]:
        """(the load per unit Ω² - iΩ', the bow's constant load), each (4 + r,) complex."""
        per_drive = np.zeros(4 + self.modes_kept, dtype=complex)
        per_drive[1:3] = self.mass_unbalance, self.couple_unbalance
        shaft = [p for p in self.plane_weights if not self.on_disc(p)]
        if shaft:
            per_drive += np.array([p.weight for p in shaft]) @ self.shape_matrix(np.array([p.z for p in shaft]))
        # K_shaft(φ) q_bow(φ): the deviatoric part Δ k e^{2i(φ+α)} conj(·) of the bow turns with it too.
        bow = (
            self.shaft_stiffness()
            @ self.bow_shape()
            * self.bow
            * (
                np.exp(1j * self.bow_angle)
                + self.asymmetry * np.exp(1j * (2.0 * self.asymmetry_angle - self.bow_angle))
            )
        )
        return per_drive, bow

    def plane_load(self, P: np.ndarray) -> np.ndarray:
        """The real load f (..., n_dof) of a turning one f_x + i f_y = P (..., 4 + r): f_x = Re P, f_y = Im P."""
        P = np.asarray(P)
        return self.merge(P.real, P.imag)

    def gravity_load(self, omega: float = 0.0) -> np.ndarray:
        """f_g (n_dof,): the weights of the journals, the disc and the shaft, in -y (N).

        With oil films, also the constant load p(Ω) that holds each journal where its
        film settles it (see _film_terms), so it depends on the speed.
        """
        # The stations moving together, the slope at the disc held, is a rigid shift (modal coordinates 0).
        shift = np.zeros(4 + self.modes_kept)
        shift[[0, 1, 3]] = 1.0
        f = self.merge(np.zeros_like(shift), -self.gravity * self.plane_mass() @ shift)
        return f + self._film_terms(omega)[2] if self.films and self.gravity else f

    def static_deflection(self, omega: float | np.ndarray = 0.0) -> np.ndarray:
        """q_s = (K + ΩH)^{-1} f_g: where gravity holds the rotor, (n_dof,) or (len(omega), n_dof).

        On magnets, from the whole state (static_state): with integral action they hold
        their journals at the centre.

        Unless the stiffness depends on the speed (speed_dependent), neither does this.
        With an asymmetric or cracked shaft the rotor never holds still under
        gravity: q_s is then the mean of its periodic response (gravity_response),
        the sag about which the gravity vibration swings, and it depends on the speed.
        """
        omega = np.asarray(omega, dtype=float)
        n = self.n_dof
        if not self.gravity:
            return np.zeros(omega.shape + (n,))
        if self.periodic:
            from .rotor_modes import gravity_response

            return gravity_response(self, omega.ravel())[0].real.reshape(omega.shape + (n,))
        if self.magnets:  # the magnets' push has states of their own
            if not self.speed_dependent or omega.ndim == 0:
                q = self.static_state(float(omega) if omega.ndim == 0 else 0.0)[:n]
                return q if omega.ndim == 0 else np.broadcast_to(q, omega.shape + (n,)).copy()
            out = np.array([self.static_state(w)[:n] for w in omega.ravel()])
            return out.reshape(omega.shape + (n,))
        if not self.speed_dependent or omega.ndim == 0:
            w = float(omega) if omega.ndim == 0 else 0.0
            q = np.linalg.solve(self.stiffness(w), self.gravity_load(w))
            return q if omega.ndim == 0 else np.broadcast_to(q, omega.shape + (n,)).copy()
        if self.films:
            flat = omega.ravel()
            out = np.array([np.linalg.solve(self.stiffness(w), self.gravity_load(w)) for w in flat])
            return out.reshape(omega.shape + (n,))
        K, H, f = self.matrices()[3], self._circulatory, self.gravity_load()
        return np.linalg.solve(K + omega[..., None, None] * H, np.broadcast_to(f, omega.shape + (n,))[..., None])[
            ..., 0
        ]

    @property
    def sag_varies(self) -> bool:
        """True if the static sag depends on the speed: a speed-dependent stiffness, or a turning one."""
        return bool(self.gravity and (self.speed_dependent or self.periodic))

    def rest_deflection(self, phase: float = 0.0) -> np.ndarray:
        """Where gravity holds the rotor at rest, its keyphasor mark at angle φ (n_dof,).

        As static_deflection(0), except that an asymmetric or cracked shaft at rest
        sags on the stiffness it has at that angle, not on its mean.
        """
        if not self.gravity or not self.periodic:
            return self.static_deflection(0.0)
        return self.static_state(0.0, phase)[: self.n_dof]

    def bow_displacement(self, phase: float) -> np.ndarray:
        """q_bow (n_dof,): the bow at shaft angle φ, as displacements in fixed coordinates."""
        b = self.bow_shape() * self.bow * np.exp(1j * (phase + self.bow_angle))
        return self.merge(b.real, b.imag)

    def shape_matrix(self, z: np.ndarray) -> np.ndarray:
        """S (len(z) x (4 + r)): the shaft's deflection at axial positions z (m) is S @ one plane's DOFs.

        Within each element the shaft bends as a cubic through its nodes' displacements and
        slopes (exact for the massless shaft); beyond the shaft's ends the end element carries on.
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        nodes = self._mesh[0]
        T = self._reduced[0]
        e = np.clip(np.searchsorted(nodes, z, side="right") - 1, 0, nodes.size - 2)
        length = nodes[e + 1] - nodes[e]
        s = (z - nodes[e]) / length
        h = np.column_stack(
            [1 - 3 * s**2 + 2 * s**3, length * (s - 2 * s**2 + s**3), 3 * s**2 - 2 * s**3, length * (s**3 - s**2)]
        )
        rows = 2 * e[:, None] + np.arange(4)
        return np.einsum("zk,zkj->zj", h, T[rows])

    @property
    def extent(self) -> tuple[float, float]:
        """The shaft's ends (m): the bearings, or the disc where it overhangs one."""
        return min(0.0, self.a), max(self.length, self.a)

    def deflection(self, q: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Shaft centre (x, y) at axial positions z for displacements q (..., n_dof): shape (..., len(z), 2)."""
        S = self.shape_matrix(z)
        q = np.asarray(q)
        ix, iy = self._planes
        return np.stack([q[..., ix] @ S.T, q[..., iy] @ S.T], axis=-1)

    # ------------------------------------------------------------- matrices
    def matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(M, C, G, K), each n_dof square on q = [x_A, x_D, θx, x_B, y_A, y_D, θy, y_B, ξ_x, ξ_y].

        The equations of motion are M q'' + (C + Ω G) q' + (K + Ω H) q = f, H from
        circulatory(). K holds the bearings' cross-coupled stiffness (so it is not
        symmetric unless k_xy = 0) and C the shaft's internal damping. An oil film's
        stiffness and damping are not in them: they change with the speed, so use
        stiffness(Ω) and damping(Ω) wherever the speed is known.
        """
        M, C, G, K = self._matrices
        return M.copy(), C.copy(), G.copy(), K.copy()

    @cached_property
    def _matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A, B = self.bearing_a, self.bearing_b
        ks = self.shaft_stiffness()
        mp = self.plane_mass()
        M = self._blocks(mp, mp)
        K = self._blocks(ks, ks)
        C = self._blocks(self.loss_time * ks, self.loss_time * ks)
        for j, bearing in ((XA, A), (XB, B)):
            k, c = bearing.springs()
            K[np.ix_([j, j + 4], [j, j + 4])] += k
            C[np.ix_([j, j + 4], [j, j + 4])] += c
        # Id θx'' + Ω Ip θy' = moment in x-z;  Id θy'' - Ω Ip θx' = moment in y-z. The shaft's own
        # sections do the same, with twice their diametral rotary inertia.
        gp = 2.0 * self._reduced[3]
        gp[2, 2] += self.ip
        G = self._blocks(np.zeros_like(gp), np.zeros_like(gp), gp, -gp)
        return M, C, G, K + self._periodic_terms[0]

    def circulatory(self) -> np.ndarray:
        """H (8x8): the skew stiffness per unit speed, Ω H in the equations.

        The shaft's internal damping η K_shaft acts on the bending rate seen in the
        rotating shaft, q' - Ω J q, J turning x toward y. Its -Ω J part moves to the
        left side as Ω H, with H = η [0 K_shaft; -K_shaft 0]. Each fluid-film
        bearing's cross-coupling λ c Ω adds λ c where K has k_xy.
        """
        return self._circulatory.copy()

    @cached_property
    def _circulatory(self) -> np.ndarray:
        eta_ks = self.loss_time * self.shaft_stiffness()
        zero = np.zeros_like(eta_ks)
        H = self._blocks(zero, zero, eta_ks, -eta_ks)
        for j, bearing in ((XA, self.bearing_a), (XB, self.bearing_b)):
            if bearing.swirling:
                H[j, j + 4] += bearing.swirl * bearing.damping
                H[j + 4, j] -= bearing.swirl * bearing.damping
        return H

    @property
    def films(self) -> bool:
        """True if either bearing is an oil film."""
        return self.bearing_a.film is not None or self.bearing_b.film is not None

    @property
    def whole_films(self) -> bool:
        """True if the time response takes the oil films' whole force (`nonlinear_films`, on a round shaft).

        A turning shaft's split time step keeps them linearized.
        """
        return self.nonlinear_films and self.films and not self.periodic

    @property
    def speed_dependent(self) -> bool:
        """True if the stiffness changes with the speed: internal damping, a swirling bearing or an oil film."""
        return bool(self.internal_damping or self.bearing_a.swirling or self.bearing_b.swirling or self.films)

    def cross_coupling(self, omega: float) -> tuple[float, float]:
        """Each bearing's cross-coupled stiffness k_xy at speed Ω (N/m), A then B: k_xy + λ c Ω, or the film's."""
        states = self.bearing_states(omega)
        a, b = (
            float(st.k[0, 1]) if st is not None else brg.kxy + brg.swirling * brg.swirl * brg.damping * omega
            for brg, st in zip((self.bearing_a, self.bearing_b), states)
        )
        return a, b

    @cached_property
    def bearing_loads(self) -> tuple[float, float]:
        """Each bearing's share of the rotor's weight (N), A then B: the support's upward push on its journal.

        With the journals held, the shaft carries the disc's and its own weight to
        them, and each journal's weight is its own. Negative where the support has
        to pull the journal down (the far bearing of an overhung rotor).
        """
        if not self.gravity:
            return 0.0, 0.0
        ks = self.shaft_stiffness()
        f = -self.gravity * self.plane_mass() @ np.r_[1.0, 1.0, 0.0, 1.0, np.zeros(self.modes_kept)]
        j = [0, 3]
        r = [i for i in range(ks.shape[0]) if i not in j]
        held = np.linalg.solve(ks[np.ix_(r, r)], f[r])
        push = ks[np.ix_(j, r)] @ held - f[j]
        return float(push[0]), float(push[1])

    def bearing_states(self, omega: float) -> tuple[BearingState | None, BearingState | None]:
        """Each oil film's state at speed Ω under its share of the weight, A then B (None: springs)."""
        key = abs(float(omega))

        def make() -> tuple[BearingState | None, BearingState | None]:
            a, b = (
                None if brg.film is None else brg.film.with_load(abs(w)).on_rotor(key)
                for brg, w in zip((self.bearing_a, self.bearing_b), self.bearing_loads)
            )
            return a, b

        return _cached(
            self,
            "_film_states",
            key,
            make,
            CACHE_STATES,
        )

    def _film_terms(self, omega: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(K_f, C_f, p): the films' stiffness and damping at speed Ω, and the load that holds them there.

        A film is linearized about where its journal settles, q_J at its eccentricity
        and attitude, not about the bearing's centre: f = W - K_f (q - q_J) - C_f q',
        W its share of the weight held up. So it adds K_f to the stiffness and the
        constant p = W + K_f q_J to the load (with gravity, beside f_g): the rotor
        then rests with its journals on the film's locus.

        Cached per speed, read-only: a sweep asks for each speed several times.
        """
        return _cached(self, "_film_terms_at", abs(float(omega)), lambda: self._new_film_terms(omega), CACHE_TERMS)

    def _new_film_terms(self, omega: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.n_dof
        Kf, Cf, p = np.zeros((n, n)), np.zeros((n, n)), np.zeros(n)
        for j, b, w, st in zip(
            (XA, XB), (self.bearing_a, self.bearing_b), self.bearing_loads, self.bearing_states(omega)
        ):
            if st is None or b.film is None:
                continue
            d = slice(j, j + 5, 4)  # [x, y] of the journal
            Kf[d, d] = st.k
            Cf[d, d] = st.c
            # An upward pull mirrors the journal through the centre; the coefficients stay the same.
            q_j = math.copysign(b.film.clearance, w) * np.array(st.position())
            p[d] = np.array([0.0, w]) + st.k @ q_j
        for a in (Kf, Cf, p):
            a.flags.writeable = False
        return Kf, Cf, p

    def stiffness(self, omega: float) -> np.ndarray:
        """K + Ω H + the films': the whole stiffness at speed Ω (with an asymmetric shaft, its mean)."""
        K = self._matrices[3]
        if not self.speed_dependent:
            return K.copy()
        K = K + omega * self._circulatory
        return K + self._film_terms(omega)[0] if self.films else K

    def damping(self, omega: float) -> np.ndarray:
        """C + the films': the damping at speed Ω (without the gyroscopic Ω G)."""
        C = self._matrices[1]
        return C + self._film_terms(omega)[1] if self.films else C.copy()

    @property
    def deviatoric(self) -> tuple[np.ndarray, np.ndarray]:
        """(K_c, K_s): an asymmetric shaft's stiffness less its mean is cos 2θ K_c + sin 2θ K_s, θ = φ + α.

        Bending toward the stiff axis (at θ) meets Δ k more than the mean stiffness k
        of the shaft, across it Δ k less: in each plane pair that is Δ k [cos 2θ,
        sin 2θ; sin 2θ, -cos 2θ]. As f_x + i f_y it is Δ k e^{2iθ} (x - iy). The
        shaft's reduced stiffness all scales with EI, so the modal coordinates
        share the same Δ.
        """
        Kc, Ks = self._deviatoric
        return Kc.copy(), Ks.copy()

    @cached_property
    def _deviatoric(self) -> tuple[np.ndarray, np.ndarray]:
        dk = self.asymmetry * self.shaft_stiffness()
        zero = np.zeros_like(dk)
        return self._blocks(dk, -dk), self._blocks(zero, zero, dk, dk)

    @cached_property
    def _periodic_terms(self) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """(the crack's mean loss of stiffness (n_dof square, in K), {k: E_k}): the turning part's harmonics.

        The stiffness less its mean is Σ_k E_k e^{ikφ} + conj, k ≥ 1. The asymmetric
        shaft's deviatoric part is E_2 alone. An open crack takes D_o off the shaft's
        stiffness bending along its mouth's direction θ = φ + β_c, D_a across: in fixed
        axes P + Re(W' e^{2iθ}), P = (D_o + D_a)/2 on both planes and W' the
        deviatoric part of (D_o - D_a)/2, as for the asymmetric shaft. Breathing, it
        is open by g = (1 - s sin θ)/2, so g (P + W e^{2iθ} + W̄ e^{-2iθ}) takes
        harmonics 0 to 3 of θ.
        """
        n = self.n_dof
        mean, E = np.zeros((n, n)), {}
        if self.asymmetry:
            Kc, Ks = self._deviatoric
            E[2] = 0.5 * (Kc - 1j * Ks) * np.exp(2j * self.asymmetry_angle)
        opening = self.crack_opening
        if opening:
            c_open, c_across = self.crack_compliances
            d_open, d_across = self._hinge_loss(c_open), self._hinge_loss(c_across)
            dm, dd = 0.5 * (d_open + d_across), 0.5 * (d_open - d_across)
            zero = np.zeros_like(dm)
            P = self._blocks(dm, dm)
            W = 0.5 * (self._blocks(dd, -dd) - 1j * self._blocks(zero, zero, dd, dd))
            crack = self.crack
            assert crack is not None  # it opens
            turn = np.exp(1j * crack.angle)
            mean -= opening * P
            E[2] = E.get(2, 0.0) - opening * W * turn**2
            if crack.breathing:  # g = ½ + (is/4) e^{iθ} - (is/4) e^{-iθ}
                quarter = 0.25j * self.crack_sign
                E[1] = -quarter * (P - W) * turn
                E[3] = -quarter * W * turn**3
        return mean, {k: E[k] for k in sorted(E) if np.any(E[k])}

    @property
    def periodic(self) -> bool:
        """True if the stiffness changes as the shaft turns: an asymmetric shaft, or an open or breathing crack."""
        return bool(self._periodic_terms[1])

    @property
    def spacing(self) -> int:
        """The harmonics of φ the stiffness changes by are multiples of this: 2 (turning with the shaft) or 1."""
        return math.gcd(*self._periodic_terms[1]) if self.periodic else 2

    @property
    def rotating_frame(self) -> bool:
        """True if the rotor is time-invariant in axes turning with the shaft (see rotating_state_matrix).

        Its stiffness has to turn rigidly with the shaft (only 2X: not a breathing
        crack, opened by the weight that stays put), on isotropic supports.
        """
        return self.spacing == 2 and self.isotropic_supports

    @cached_property
    def _kick(self) -> tuple[np.ndarray, np.ndarray]:
        """(k, S): the turning stiffness's pull on the accelerations is M^{-1} ΔK(φ) q = (cos kφ, sin kφ) S q.

        k (m,) the harmonics and S (2m x n x n) stacked [2 M^{-1} Re E_k, ..., -2 M^{-1} Im E_k, ...].
        """
        E = self._periodic_terms[1]
        Minv = self.mass_inverse
        ks = np.array(list(E), dtype=float)
        S = np.array([2.0 * Minv @ e.real for e in E.values()] + [-2.0 * Minv @ e.imag for e in E.values()])
        return ks, S

    def stiffness_at(self, omega: float, phase: float | np.ndarray) -> np.ndarray:
        """K + Ω H + the deviatoric part at shaft angle φ: the whole stiffness at that instant, (..., n, n)."""
        K = self.stiffness(omega)
        phase = np.asarray(phase, dtype=float)
        K = np.broadcast_to(K, phase.shape + K.shape).copy()
        for k, e in self._periodic_terms[1].items():
            K += 2.0 * (np.exp(1j * k * phase)[..., None, None] * e).real
        return K

    @property
    def isotropic_supports(self) -> bool:
        """True if both bearings have the same stiffness and damping in x as in y, and turn alike.

        Their cross terms then have to be a pure turn (k_yx = -k_xy, no c_xy), and
        no film (whose stiffness depends on the load's direction) nor magnet (whose
        controller's states the rotating frame does not turn).

        Then an asymmetric shaft is time-invariant in coordinates turning with it,
        so a few harmonics give its response and modes exactly (rotating_frame).
        """
        return all(
            b.film is None
            and b.magnet is None
            and b.kx == b.ky
            and b.cx == b.cy
            and b.cxy == b.cyx == 0.0
            and b.k_yx == -b.kxy
            for b in (self.bearing_a, self.bearing_b)
        )

    def axis_system(self, stiff: bool) -> RotorSystem:
        """The same rotor with a round shaft as stiff as this one's stiff (or soft) axis: EI(1 ± Δ).

        Its critical speed is that axis's: an asymmetric shaft has two. A crack's
        axes are along its mouth (soft) and across: in axes turning with the shaft
        it takes ḡ D_o and ḡ D_a off them on average (ḡ = crack_opening), so each
        axis's round shaft has a crack that takes that off both planes. With both an
        asymmetry and a crack, whose axes need not line up, the crack's mean loss
        goes on both axes: approximate.
        """
        f = 1.0 + (self.asymmetry if stiff else -self.asymmetry)
        out = self.with_(youngs=self.youngs * f, asymmetry=0.0)
        opening, crack = self.crack_opening, self.crack
        if not opening or crack is None:
            return out.with_(crack=None) if crack is not None else out
        _, s = self._crack_hinge
        taus = [c / (1.0 + c * s) for c in self.crack_compliances]  # each loss is b b^T τ
        tau = opening * (0.5 * sum(taus) if self.asymmetry else taus[1] if stiff else taus[0])
        # Its hinge's b and s scale with EI, by f: f² τ(c') = τ.
        c = tau / (f * (f - tau * s)) if f > tau * s else 1e12
        return out.with_(crack=replace(crack, breathing=False, compliance=(c, c)))

    @cached_property
    def mass_inverse(self) -> np.ndarray:
        """M^{-1}: diagonal for the massless shaft, full once the shaft's mass couples the DOFs."""
        M = self._matrices[0]
        if np.count_nonzero(M - np.diag(np.diag(M))) == 0:
            return np.diag(1.0 / np.diag(M))
        return np.linalg.inv(M)

    # ------------------------------------------------------------- magnetic bearings
    @property
    def magnets(self) -> bool:
        """True if either bearing is an active magnetic bearing."""
        return self.bearing_a.magnet is not None or self.bearing_b.magnet is not None

    @cached_property
    def _axes(self) -> list[tuple[int, MagneticBearing]]:
        """Each magnet's axes: (the journal's DOF in q, its bearing), x then y, A then B."""
        return [
            (d, b.magnet)
            for j, b in ((XA, self.bearing_a), (XB, self.bearing_b))
            if b.magnet is not None
            for d in (j, j + 4)
        ]

    @property
    def n_control(self) -> int:
        """The controllers' states, after q and q' in the state: 0 without magnets."""
        return sum(m.n_states for _, m in self._axes)

    @property
    def n_states(self) -> int:
        """The length of the state z = [q, q', the controllers' states]."""
        return 2 * self.n_dof + self.n_control

    @cached_property
    def _control(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(A_c, B_c, F_c): the controllers' x_c' = A_c x_c + B_c q, and their push on q, F_c x_c.

        Each axis's journal displacement drives its controller (a, b, c); its current
        pushes the journal back by -k_i c x_c (the magnets' -k_m x is in K).
        """
        n, nc = self.n_dof, self.n_control
        Ac, Bc, Fc = np.zeros((nc, nc)), np.zeros((nc, n)), np.zeros((n, nc))
        k = 0
        for d, m in self._axes:
            a, b, c = m.state_space()
            sl = slice(k, k + b.size)
            Ac[sl, sl], Bc[sl, d], Fc[d, sl] = a, b, -m.current_gain * c
            k += b.size
        return Ac, Bc, Fc

    def support_stiffness(self, s: complex) -> tuple[np.ndarray, list[int]]:
        """(D, pinned): the magnets' controlled push at s (on top of K's -k_m), -D(s) q, n_dof square.

        D(s) = k_i G(s) on each axis. With integral action it is infinite at s = 0: a
        steady load cannot move the journal there, so those DOFs are `pinned`, held
        at 0 (their equation's unknown is then the force the magnet takes).
        """
        n = self.n_dof
        D, pinned = np.zeros((n, n), dtype=complex), []
        for d, m in self._axes:
            if s == 0 and m.integral:
                pinned.append(d)
            else:
                D[d, d] = m.current_gain * m.gain(s)
        return D, pinned

    def dynamic_solve(self, Z: np.ndarray, f: np.ndarray, s: complex) -> np.ndarray:
        """Q from (Z + D(s)) Q = f: Z the rest of the dynamic stiffness at s, D the magnets' (support_stiffness)."""
        if not self.magnets:
            return np.linalg.solve(Z, f)
        D, pinned = self.support_stiffness(s)
        Z = Z + D
        f = np.array(f, dtype=Z.dtype)
        Z[pinned] = 0.0
        Z[pinned, pinned] = 1.0
        f[pinned] = 0.0
        return np.linalg.solve(Z, f)

    def state_matrix(self, omega: float) -> np.ndarray:
        """A(Ω) (n_states square) for z = [q, q', x_c]: z' = A z + B u."""
        G = self._matrices[2]
        Minv = self.mass_inverse
        n = self.n_dof
        A = np.zeros((self.n_states, self.n_states))
        A[:n, n : 2 * n] = np.eye(n)
        A[n : 2 * n, :n] = -Minv @ self.stiffness(omega)
        A[n : 2 * n, n : 2 * n] = -Minv @ (self.damping(omega) + omega * G)
        if self.magnets:
            Ac, Bc, Fc = self._control
            A[n : 2 * n, 2 * n :] = Minv @ Fc
            A[2 * n :, :n], A[2 * n :, 2 * n :] = Bc, Ac
        return A

    def input_matrix(self) -> np.ndarray:
        """B (n_states x n): the inputs u = f (n,), a force or moment on each DOF."""
        n = self.n_dof
        B = np.zeros((self.n_states, n))
        B[n : 2 * n] = self.mass_inverse
        return B

    def static_state(self, omega: float = 0.0, phase: float | None = None) -> np.ndarray:
        """The state z (n_states,) where gravity alone holds the rotor still at speed Ω: A z + B f_g = 0.

        Its q is static_deflection's (rest_deflection's, at shaft angle `phase`); its
        controllers' states hold the weight on the magnets. With an asymmetric or
        cracked shaft turning, q is not still: use static_deflection for its mean.
        """
        if not self.gravity:
            return np.zeros(self.n_states)
        A = self.state_matrix(omega)
        if phase is not None:
            n = self.n_dof
            A[n : 2 * n, :n] = -self.mass_inverse @ self.stiffness_at(omega, phase)
        return np.linalg.solve(A, -self.input_matrix() @ self.gravity_load(omega))

    # ------------------------------------------------------------- checks
    def jeffcott_speed(self) -> float:
        """Critical speed (rad/s) of the textbook Jeffcott rotor with these parts.

        The disc's mass on the shaft's midspan-equivalent stiffness in series with
        the two bearings in parallel, all massless and isotropic (x-direction
        stiffnesses), the disc's tilt ignored: ω² = k_eq / m. Exact for a disc at
        midspan on identical bearings as the journal masses go to zero; poor for an
        overhung disc, whose tilt (and so the gyroscopic moment) matters most.
        """
        a, b, L = self.a, self.b, self.length
        k_shaft = self.disc_stiffness
        # The bearings carry b/L and a/L of the load (one of them negative for an
        # overhung disc: the far bearing is pulled up); the disc moves by the weighted sum.
        flex_bearings = (b / L) ** 2 / self.bearing_a.kx + (a / L) ** 2 / self.bearing_b.kx
        return math.sqrt(1.0 / (1.0 / k_shaft + flex_bearings) / self.disc_mass)


def beam_stiffness(ei: float, length: float) -> np.ndarray:
    """Euler–Bernoulli beam element stiffness on [w1, θ1, w2, θ2], θ = dw/dz."""
    L = length
    return (
        ei
        / L**3
        * np.array(
            [
                [12.0, 6 * L, -12.0, 6 * L],
                [6 * L, 4 * L * L, -6 * L, 2 * L * L],
                [-12.0, -6 * L, 12.0, -6 * L],
                [6 * L, 2 * L * L, -6 * L, 4 * L * L],
            ]
        )
    )


def beam_mass(rho_a: float, rho_i: float, length: float, lumped: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """A beam element's mass on [w1, θ1, w2, θ2]: (translation, rotary inertia), ρA and ρI per length.

    Consistent: the cubic shape functions' ∫ρA N^T N and ∫ρI N'^T N' (a Rayleigh beam).
    Lumped: half of each to each node, on its displacement and its slope.
    """
    L = length
    if lumped:
        half_a, half_i = 0.5 * rho_a * L, 0.5 * rho_i * L
        return np.diag([half_a, 0.0, half_a, 0.0]), np.diag([0.0, half_i, 0.0, half_i])
    translation = (
        rho_a
        * L
        / 420.0
        * np.array(
            [
                [156.0, 22 * L, 54.0, -13 * L],
                [22 * L, 4 * L * L, 13 * L, -3 * L * L],
                [54.0, 13 * L, 156.0, -22 * L],
                [-13 * L, -3 * L * L, -22 * L, 4 * L * L],
            ]
        )
    )
    rotation = (
        rho_i
        / (30.0 * L)
        * np.array(
            [
                [36.0, 3 * L, -36.0, 3 * L],
                [3 * L, 4 * L * L, -3 * L, -L * L],
                [-36.0, -3 * L, 36.0, -3 * L],
                [3 * L, -L * L, -3 * L, 4 * L * L],
            ]
        )
    )
    return translation, rotation
