"""Fluid–structure page: the fluid a body carries with it (added mass), what it does to its frequencies, and a
liquid's sloshing in a tank.

Left: the fluid's density, and the body: a sphere or a cylinder, its radius and density, its natural frequency
on its springs in vacuum, and a concentric wall round it (a tube in a pipe). Centre: the potential flow about the
body as it moves to the right (speed in colour, streamlines), and the fluid's speed ahead of it against distance.
Right: the added-mass coefficient against the wall's distance, the frequency's fall against the body's density;
a plate in water (its own controls: each mode's AVMI factor and its wet frequency); sloshing in a rectangular tank
(its own controls: the frequencies and the spring–mass analogue's masses against depth); a tuned liquid damper on
a structure (its own controls; the tank is the sloshing tab's: its response, and its peak against depth); and the
Theory. While the sloshing or the damper tab shows, the centre shows the tank instead of the body: the mode's
surface at its extremes and its flow as it passes level. The page has its own models (core/added_mass.py,
core/sloshing.py) and no frame loop: a control change redraws at once.

Working on it
-------------
Each tab is worked out only while it shows (`LazyTabs`); the plate's modal added masses are the slow part
(~0.5 s for 24 modes the first time; `added_mass._plate_integral` caches them by the plate's size). The damper's
peak against depth sweeps DEPTHS tanks (~20 ms). Trials' numbers are in `tests/test_tutorial_numbers.py`, the
core's in `tests/test_added_mass.py` and `tests/test_sloshing.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.added_mass import (
    AIR,
    SHAPE_NAMES,
    WATER,
    Body,
    PlateModes,
    Shape,
    avmi,
    axis_speed,
    flow,
    plate_dry_frequency,
    plate_modal_mass,
    plate_modes,
    wet_frequency,
)
from ..core.beams_plates import Material, Plate
from ..core.practice import Problem, Question, Requirement, at_least, at_most, kept, number, pick, rounded
from ..core.sloshing import (
    Damper,
    G,
    Structure,
    Tank,
    damper_response,
    den_hartog,
    frozen_response,
    loaded_frequency,
    mass_ratio,
    peak,
    peak_against_depth,
    response_band,
    rigid_mass,
    slosh_frequency,
    slosh_mass,
    tank_flow,
)
from . import settle_plots
from .layouts import ColumnScroll
from .lesson import (
    Form,
    TrialBox,
    column,
    combo,
    connect_all,
    fmt,
    mark,
    marker,
    note_label,
    plot,
    readout_label,
    set_quietly,
    stack,
    theory,
    three_panes,
    trials_html,
)
from .links import load_notes, page_link
from .style import colors
from .tab_groups import LazyTabs
from .theming import add_legend

TAB_BODY, TAB_PLATE, TAB_SLOSH, TAB_DAMPER = range(4)
GRID = 161  # points across the flow picture
STREAMLINES = 14
PLATE_MODES = 24
RATIOS = np.geomspace(1.02, 20.0, 200)  # R_o/R for the coefficient's curves
DENSITIES = np.geomspace(0.05, 50.0, 200)  # ρ_s/ρ_f
TANK_GRID = 141  # points along the tank's longer side
TANK_STREAMLINES = 10
DEPTH_RATIOS = np.geomspace(0.01, 2.0, 200)  # h/L for the frequencies' and masses' curves
DEPTHS = 120  # depths in the damper's tuning curve


@dataclass(frozen=True)
class FluidTrial:
    name: str
    note: str
    fluid: float = WATER  # kg/m³
    shape: int = Shape.CYLINDER
    radius: float = 25.0  # mm
    density: float = 7850.0  # kg/m³, the body's
    f_dry: float = 20.0  # Hz, on its springs in vacuum
    walled: bool = False
    ratio: float = 1.5  # R_o/R
    # the plate
    a: float = 0.6  # m
    b: float = 0.4
    thickness: float = 3.0  # mm
    plate_rho: float = 7850.0
    E: float = 210.0  # GPa
    sides: int = 1
    m: int = 1
    n: int = 1
    # the tank
    length: float = 1.0  # m, along the motion
    depth: float = 0.5
    breadth: float = 1.0
    slosh: int = 1  # the sloshing mode drawn
    # the damper
    structure_mass: float = 20.0  # t
    structure_f: float = 0.5  # Hz
    structure_zeta: float = 1.0  # %
    tanks: int = 1
    damper_zeta: float = 13.0  # %, the sloshing's
    tab: int = TAB_BODY


TRIALS = (
    FluidTrial(
        "A sphere's added mass",
        "A steel ball 100 mm in radius, in water, on springs that give it 20 Hz in vacuum. As it moves, the water "
        "parts in front, closes behind and flows round its sides (centre): that flow's kinetic energy is "
        "½m<sub>a</sub>U² with m<sub>a</sub> = ½ρ<sub>f</sub>V = 2.09 kg, half the water it displaces. On the "
        "ball's 32.9 kg it lowers the frequency to 19.4 Hz.",
        shape=Shape.SPHERE,
        radius=100.0,
    ),
    FluidTrial(
        "A cylinder carries its own volume",
        "A steel rod 25 mm in radius, moving across its axis: the flow is two-dimensional, and the water it drags "
        "along is all it displaces, m<sub>a</sub> = ρπR² = 1.96 kg a metre (C<sub>a</sub> = 1, twice a "
        "sphere's). The flow ahead falls off as R²/r² against a sphere's R³/r³ (centre, bottom): a cylinder "
        "stirs more of the fluid. 20 Hz falls to 18.8 Hz.",
    ),
    FluidTrial(
        "A light body",
        "The same cylinder at 500 kg/m³, half the water's density (a buoyant float, or a thin tube). Its added "
        "mass is unchanged, 1.96 kg a metre, but now twice its own: the frequency falls to 0.577 of the dry one, "
        "11.5 Hz. The frequency's fall depends on the densities' ratio (right, bottom).",
        density=500.0,
    ),
    FluidTrial(
        "In air",
        "The steel cylinder in air (1.21 kg/m³): the added mass is 2.38 g a metre, 1/6500 of its mass, and the "
        "frequency drops by less than 0.01 %. Added mass matters in water, and for light structures in air "
        "(balloons, membranes, thin panels).",
        fluid=AIR,
    ),
    FluidTrial(
        "A tube in a pipe",
        "The rod inside a concentric pipe of 1.2 times its radius: the water it pushes has to squeeze through "
        "the 5 mm gap, fast, to get round it. C<sub>a</sub> = (R<sub>o</sub>² + R²)/(R<sub>o</sub>² − R²) = 5.55 "
        "(Fritz), so m<sub>a</sub> = 10.9 kg a metre and 20 Hz becomes 15.3 Hz. Heat exchanger tubes in their "
        "baffles, and fuel rods in their channels, are designed with it.",
        walled=True,
        ratio=1.2,
    ),
    FluidTrial(
        "Close the gap",
        "R<sub>o</sub>/R = 1.05: a gap of 1.25 mm. C<sub>a</sub> = 20.5, and the water weighs 2.6 times as much "
        "as the steel: 20 Hz falls to 10.5 Hz. The water beside the rod moves about R/g times as fast as the rod, "
        "here 20.5 times, and its energy goes as the square of that.",
        walled=True,
        ratio=1.05,
    ),
    FluidTrial(
        "A plate in water",
        "The Vibro-acoustics panel, 600 × 400 × 3 mm steel, baffled with water on one side (Plate tab). "
        "Mode (1, 1)'s water weighs 11.9 kg against the plate's modal 1.41 kg: β = 8.43, and 66.6 Hz falls to "
        "21.7 Hz, a third. A thin plate in water is mostly water.",
        tab=TAB_PLATE,
    ),
    FluidTrial(
        "Higher modes carry less",
        "Mode (3, 2), 369 Hz dry. Its cells push and pull side by side, so the water just sloshes between them, "
        "and the fluid it moves reaches only about a cell's size: β = 2.18 against (1, 1)'s 8.43, and it falls to "
        "207 Hz, not a third. The wet modes are closer together than the dry ones (right, bottom).",
        m=3,
        n=2,
        tab=TAB_PLATE,
    ),
    FluidTrial(
        "A thicker plate",
        "The plate 10 mm thick. The water's added mass is the same 11.9 kg, but the plate's modal mass is "
        "3.3 times more: β = 2.53 (it goes as ρ<sub>f</sub>/ρ<sub>p</sub>h), and the fundamental, 222 Hz dry, "
        "falls to 118 Hz, 0.53 of it.",
        thickness=10.0,
        tab=TAB_PLATE,
    ),
    FluidTrial(
        "Sloshing in a tank",
        "A tank 1 m long, filled to 0.5 m (Sloshing tab). Its first mode tilts the surface one way and the "
        "other, at ω² = gk tanh(kh) with k = π/L: 0.846 Hz. Of the 500 kg of water, the spring–mass analogue's "
        "first sloshing mass is 237 kg; 250 kg moves with the tank as if it were solid.",
        tab=TAB_SLOSH,
    ),
    FluidTrial(
        "Shallow water",
        "The same tank filled to 5 cm: a long wave, running at √(gh) from wall to wall and back, 0.349 Hz against "
        "√(gh)/2L = 0.350 Hz. 40.2 kg of the 50 kg sloshes in the first mode, and only 2.71 kg moves with the "
        "tank: in shallow water nearly everything sloshes.",
        depth=0.05,
        tab=TAB_SLOSH,
    ),
    FluidTrial(
        "Deep water",
        "Filled to 2 m. The first mode's flow dies away as e<sup>kz</sup>, within about L/π = 0.32 m of the "
        "surface (the tank's lower part is still), so the frequency stops depending on depth: 0.884 Hz, √(gπ/L)/2π. Four times "
        "the water, but the sloshing mass is only 258 kg, 13 %: the rest, 1729 kg, rides along with the tank.",
        depth=2.0,
        tab=TAB_SLOSH,
    ),
    FluidTrial(
        "A higher mode",
        "Mode 3 of the 0.5 m fill, 1.53 Hz: three half waves, and the flow reaches only a third as deep. Its "
        "analogue mass is 9.55 kg, 2 % of the water (they go as 1/n³): a few modes are all a tank's analogue "
        "needs.",
        slosh=3,
        tab=TAB_SLOSH,
    ),
    FluidTrial(
        "An even mode",
        "Mode 2, 1.25 Hz: the surface rises at both walls and falls in the middle. It's symmetric, so shaking the "
        "tank sideways doesn't excite it, and it has no mass in the analogue (a shake along the tank's length "
        "pushes the walls the same way, not towards each other).",
        slosh=2,
        tab=TAB_SLOSH,
    ),
    FluidTrial(
        "A tuned liquid damper",
        "A 20 t structure at 0.5 Hz with 1 % damping (Damper tab), carrying a tank 2 m long and 2 m wide, "
        "filled to 0.4 m: its first mode is at 0.466 Hz, and screens in it damp the sloshing at 13 %. Frozen, the "
        "1600 kg of water would just add mass, and the structure's peak would be 52.0 times its static "
        "deflection; sloshing, the 1149 kg in the first mode is a tuned mass damper (μ = 0.0562), and the peak is "
        "5.58.",
        length=2.0,
        depth=0.4,
        breadth=2.0,
        tab=TAB_DAMPER,
    ),
    FluidTrial(
        "A bare tank",
        "The same tank with no screens: the water's own damping, ζ<sub>d</sub> = 0.5 %. The structure's peak "
        "splits in two, at 0.426 and 0.540 Hz, each still 34.5 and 32.8 times the static deflection: a damper "
        "needs damping, the more the heavier it is.",
        length=2.0,
        depth=0.4,
        breadth=2.0,
        damper_zeta=0.5,
        tab=TAB_DAMPER,
    ),
    FluidTrial(
        "Mistuned",
        "The tank filled to 0.25 m only: its first mode falls to 0.382 Hz, too far below the structure's, and the "
        "peak rises to 18.2. A liquid damper is tuned by its depth (right, bottom: the peak against it).",
        length=2.0,
        depth=0.25,
        breadth=2.0,
        tab=TAB_DAMPER,
    ),
    FluidTrial(
        "Four tanks",
        "Four of the tanks, filled to 0.3 m: 4.8 t of water, μ = 0.171, and the peak falls to 4.41. More water in "
        "the dampers needs a lower tuning (Den Hartog's f/(1 + μ)), so the four are filled less deep than the "
        "one.",
        length=2.0,
        depth=0.3,
        breadth=2.0,
        tanks=4,
        tab=TAB_DAMPER,
    ),
)

# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


def _sphere(rng: np.random.Generator) -> Question:
    r = pick(rng, (20, 50, 100, 200))  # mm
    rho = pick(rng, (1000, 1025, 850))
    body = Body(Shape.SPHERE, r / 1000, fluid=rho)
    return Question(
        f"A sphere {r:g} mm in radius moves through a liquid of density {rho:g} kg/m³. What is its added mass?",
        body.added_mass,
        f"m<sub>a</sub> = ½ρ<sub>f</sub>(4/3)πR³ = ½ × {rho:g} × (4/3)π × {r / 1000:g}³ = {number(body.added_mass)} "
        "kg: half the liquid it displaces.",
        unit="kg",
        setup=dict(shape=Shape.SPHERE, radius=r, fluid=rho),
    )


def _cylinder(rng: np.random.Generator) -> Question:
    r = pick(rng, (10, 20, 25, 50))
    rho_s = pick(rng, (2700, 7850, 1500))
    f = pick(rng, (10, 20, 50))
    body = Body(Shape.CYLINDER, r / 1000, rho_s, WATER)
    wet = wet_frequency(f, body.mass, body.added_mass)
    return Question(
        f"A solid cylinder {r:g} mm in radius, density {rho_s:g} kg/m³, vibrates across its axis on springs at "
        f"{f:g} Hz in vacuum. What is its natural frequency in water (1000 kg/m³)?",
        wet,
        f"Per metre, m = ρ<sub>s</sub>πR² = {number(body.mass)} kg and m<sub>a</sub> = ρ<sub>f</sub>πR² = "
        f"{number(body.added_mass)} kg, so f = {f:g} √(m/(m + m<sub>a</sub>)) = {f:g} √({rho_s:g}/{rho_s + 1000:g}) "
        f"= {number(wet)} Hz: only the densities matter.",
        unit="Hz",
        setup=dict(radius=r, density=rho_s, f_dry=f),
    )


def _confined(rng: np.random.Generator) -> Question:
    r = pick(rng, (1.1, 1.2, 1.5, 2))
    c = Body(Shape.CYLINDER, 1.0, outer=r).coefficient
    return Question(
        f"A cylinder vibrates inside a concentric pipe whose radius is {r:g} times its own, the gap full of "
        "water. What is its added-mass coefficient C<sub>a</sub> = m<sub>a</sub>/ρπR²?",
        c,
        f"C<sub>a</sub> = (R<sub>o</sub>² + R²)/(R<sub>o</sub>² − R²) = ({r:g}² + 1)/({r:g}² − 1) = {number(c)}.",
        setup=dict(walled=True, ratio=r),
    )


def _plate(rng: np.random.Generator) -> Question:
    f = pick(rng, (40, 66.6, 120, 250))
    beta = pick(rng, (0.5, 2, 4, 8.4))
    wet = f / math.sqrt(1 + beta)
    return Question(
        f"A plate mode at {f:g} Hz in air has an AVMI factor β = {beta:g} in water. What is its frequency in water?",
        wet,
        f"f<sub>wet</sub> = f<sub>dry</sub>/√(1 + β) = {f:g}/√{1 + beta:g} = {number(wet)} Hz.",
        unit="Hz",
        setup=dict(tab=TAB_PLATE),
    )


def _gap(rng: np.random.Generator) -> Question:
    return Question(
        "A tube vibrates inside a pipe full of water. The pipe is made smaller, closing the gap. What happens to "
        "the tube's natural frequency?",
        0,
        "It falls: the water has to squeeze through the gap to get round the tube, faster the narrower it is, and "
        "the added mass, (R<sub>o</sub>² + R²)/(R<sub>o</sub>² − R²) times ρπR², grows without bound as the gap "
        "closes.",
        choices=("It falls", "It rises", "It stays the same", "It rises, then falls"),
        setup=dict(walled=True, ratio=1.05),
        predict=True,
    )


def _thicker(rng: np.random.Generator) -> Question:
    return Question(
        "A steel plate in water is made twice as thick. What happens to its fundamental mode's AVMI factor β?",
        1,
        "It halves: the water's added mass depends on the mode's shape and the plate's size, not its thickness, "
        "while the plate's modal mass ρ<sub>p</sub>hab/4 doubles.",
        choices=("It doubles", "It halves", "It stays the same", "It falls by 8"),
        setup=dict(thickness=6.0, tab=TAB_PLATE),
        predict=True,
    )


def _pipe_spec(rng: np.random.Generator) -> Question:
    f = pick(rng, (20, 30))
    ratio = pick(rng, (1.3, 1.5, 2))  # the design
    body = Body(Shape.CYLINDER, 0.025, 7850.0, WATER, 0.025 * ratio)
    wet = wet_frequency(f, body.mass, body.added_mass)
    need = rounded(wet * 0.995, 3)

    def goal(t: FluidTrial):
        b = Body(
            Shape(t.shape), t.radius / 1000, t.density, t.fluid, t.radius * t.ratio / 1000 if t.walled else math.inf
        )
        return (
            Requirement("Cylinder", SHAPE_NAMES[t.shape], t.shape == Shape.CYLINDER),
            kept("Radius", t.radius, 25.0, "mm"),
            kept("Density", t.density, 7850.0, "kg/m³"),
            kept("Fluid", t.fluid, WATER, "kg/m³"),
            kept("Dry frequency", t.f_dry, f, "Hz"),
            Requirement("In a pipe", "yes" if t.walled else "no", t.walled),
            at_least("Wet frequency", wet_frequency(t.f_dry, b.mass, b.added_mass), need, "Hz"),
        )

    return Question(
        f"A steel rod 25 mm in radius runs inside a water-filled pipe, on supports that give it {f:g} Hz in "
        f"vacuum. Its natural frequency in the pipe must stay at or above {number(need, 3)} Hz. How small can the "
        "pipe be? Set its radius (R<sub>o</sub>/R).",
        0,
        f"The frequency rises as the gap opens and the added mass falls towards the open water's ρπR². With "
        f"R<sub>o</sub>/R = {ratio:g}, C<sub>a</sub> = {number(body.coefficient)} and the rod runs at "
        f"{number(wet)} Hz.",
        setup=dict(walled=True, ratio=1.05, f_dry=float(f)),
        goal=goal,
        design=dict(walled=True, ratio=float(ratio), f_dry=float(f)),
    )


def _slosh(rng: np.random.Generator) -> Question:
    L = pick(rng, (0.5, 1.0, 2.0, 4.0))
    h = pick(rng, (0.1, 0.2, 0.5, 1.0))
    tank = Tank(L, h)
    k = math.pi / L
    f = slosh_frequency(tank, 1)
    return Question(
        f"A rectangular tank {L:g} m long is filled with water to {h:g} m. What is the frequency of its first "
        "sloshing mode, along its length?",
        f,
        f"k = π/L = {number(k)} /m, ω² = gk tanh(kh) = 9.81 × {number(k)} × tanh({number(k * h)}), so f = "
        f"{number(f)} Hz.",
        unit="Hz",
        setup=dict(length=L, depth=h, tab=TAB_SLOSH),
    )


def _slosh_mass(rng: np.random.Generator) -> Question:
    L = pick(rng, (1.0, 2.0, 3.0))
    h = pick(rng, (0.2, 0.5, 1.0))
    B = pick(rng, (0.5, 1.0, 2.0))
    tank = Tank(L, h, B)
    m1 = float(slosh_mass(tank, 1))
    return Question(
        f"A tank {L:g} m long and {B:g} m wide is filled with water to {h:g} m. In the spring–mass analogue for "
        "a sideways motion along its length, what is the first sloshing mass?",
        m1,
        f"m<sub>L</sub> = ρLBh = {number(tank.liquid_mass)} kg, and m<sub>1</sub> = m<sub>L</sub> 8 tanh(πh/L)"
        f"/(π³h/L) = {number(tank.liquid_mass)} × 8 tanh({number(math.pi * h / L)})/(π³ × {number(h / L)}) = "
        f"{number(m1)} kg.",
        unit="kg",
        setup=dict(length=L, depth=h, breadth=B, tab=TAB_SLOSH),
    )


def _tuning(rng: np.random.Generator) -> Question:
    f = pick(rng, (0.2, 0.5, 1.0, 2.0))
    mu = pick(rng, (0.01, 0.02, 0.05, 0.1))
    fd = f / (1 + mu)
    return Question(
        f"A structure's mode at {f:g} Hz gets a tuned liquid damper whose sloshing mass is μ = {mu:g} of the "
        "structure's. By Den Hartog's rule, what should the damper's sloshing frequency be?",
        fd,
        f"f<sub>d</sub> = f/(1 + μ) = {f:g}/{1 + mu:g} = {number(fd)} Hz, a little below the structure's.",
        unit="Hz",
        setup=dict(length=2.0, depth=0.4, breadth=2.0, tab=TAB_DAMPER),
    )


def _deeper(rng: np.random.Generator) -> Question:
    return Question(
        "A tank 1 m long is filled with water to 2 m. More water is added, to 3 m. What happens to its first "
        "sloshing frequency?",
        2,
        "It stays the same: the mode's flow dies away as e<sup>kz</sup> below the surface, within about L/π, and "
        "tanh(kh) is already 1. Only the rigid mass grows.",
        choices=("It rises", "It falls", "It stays the same", "It falls by √(2/3)"),
        setup=dict(depth=3.0, tab=TAB_SLOSH),
        predict=True,
    )


def _even(rng: np.random.Generator) -> Question:
    return Question(
        "A tank is shaken sideways, along its length, at the frequency of its second sloshing mode. How much does "
        "the water slosh?",
        0,
        "Hardly at all: the second mode is symmetric (the surface rises at both walls together), and a sideways "
        "shake pushes the walls the same way, so it does no work on that mode. Its analogue mass is 0.",
        choices=("Hardly at all", "A lot: it's at resonance", "As much as at the first mode", "Twice as much"),
        setup=dict(slosh=2, tab=TAB_SLOSH),
        predict=True,
    )


def _damper_of(t: FluidTrial) -> tuple[Structure, Damper]:
    return (
        Structure(t.structure_mass * 1000, t.structure_f, t.structure_zeta / 100),
        Damper(Tank(t.length, t.depth, t.breadth, t.fluid), t.tanks, t.damper_zeta / 100),
    )


def _damper_spec(rng: np.random.Generator) -> Question:
    zeta = pick(rng, (5.0, 8.0, 10.0))  # the design, %
    design = FluidTrial("", "", length=2.0, depth=0.4, breadth=2.0, damper_zeta=zeta)
    need = rounded(peak(*_damper_of(design)) * 1.01, 3)

    def goal(t: FluidTrial):
        structure, damper = _damper_of(t)
        return (
            kept("Structure's mass", t.structure_mass, 20.0, "t"),
            kept("Structure's frequency", t.structure_f, 0.5, "Hz"),
            kept("Structure's damping", t.structure_zeta, 1.0, "%"),
            kept("Fluid", t.fluid, WATER, "kg/m³"),
            kept("Tank length", t.length, 2.0, "m"),
            kept("Tank breadth", t.breadth, 2.0, "m"),
            kept("Depth", t.depth, 0.4, "m"),
            kept("Tanks", t.tanks, 1),
            at_most("Peak |X|k/F", peak(structure, damper), need),
        )

    return Question(
        "A 20 t structure at 0.5 Hz (1 % damping) carries a tank of water 2 m by 2 m, filled to 0.4 m to tune its "
        f"sloshing. Bare, the water damps itself at 0.5 %. Screens in the tank add damping: how much is needed to "
        f"keep the structure's peak response at or below {number(need, 3)} times its static deflection? Set "
        "ζ<sub>d</sub>.",
        0,
        f"The peak falls as ζ<sub>d</sub> rises towards Den Hartog's optimum, about 13 % here. ζ<sub>d</sub> = "
        f"{zeta:g} % gives {number(peak(*_damper_of(design)))}.",
        setup=dict(length=2.0, depth=0.4, breadth=2.0, damper_zeta=0.5, tab=TAB_DAMPER),
        goal=goal,
        design=dict(length=2.0, depth=0.4, breadth=2.0, damper_zeta=zeta, tab=TAB_DAMPER),
    )


PROBLEMS = tuple(
    Problem(key, title, "fluid_page", f"See {page_link('fluid_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("sphere", "A sphere's added mass", "added-mass", "Added mass", _sphere),
        ("cylinder", "A cylinder in water", "wet", "Wet and dry", _cylinder),
        ("confined", "A cylinder in a pipe", "confined", "Confined fluid", _confined),
        ("plate", "A plate mode in water", "plates", "A plate in water", _plate),
        ("gap", "Predict: a narrower pipe", "confined", "Confined fluid", _gap),
        ("thicker", "Predict: a thicker plate", "plates", "A plate in water", _thicker),
        ("pipe-spec", "Spec: a rod in a pipe", "confined", "Confined fluid", _pipe_spec),
        ("slosh", "A tank's sloshing frequency", "sloshing", "Sloshing", _slosh),
        ("slosh-mass", "A sloshing mass", "analogue", "The spring–mass analogue", _slosh_mass),
        ("tuning", "Tuning a liquid damper", "damper", "A tuned liquid damper", _tuning),
        ("deeper", "Predict: a deeper tank", "sloshing", "Sloshing", _deeper),
        ("even", "Predict: the second mode", "analogue", "The spring–mass analogue", _even),
        ("damper-spec", "Spec: a liquid damper's screens", "damper", "A tuned liquid damper", _damper_spec),
    )
)

TIPS = {
    "fluid": "<p>The fluid's density: water 1000 kg/m³, sea water 1025, oil about 850, air 1.21.</p>",
    "shape": "<p>A sphere, or a long circular cylinder moving across its axis (per metre of length).</p>",
    "radius": "<p>The body's radius R.</p>",
    "density": "<p>The body's density (a solid one; a tube's mass over its outer volume for a hollow one).</p>",
    "f_dry": "<p>The body's natural frequency on its springs in vacuum.</p>",
    "walled": "<p>A still, concentric wall round the body: a spherical shell, or a pipe.</p>",
    "ratio": "<p>The wall's radius over the body's.</p>",
    "a": "<p>The plate's side a.</p>",
    "b": "<p>The plate's side b.</p>",
    "thickness": "<p>The plate's thickness h.</p>",
    "plate_rho": "<p>The plate's density.</p>",
    "E": "<p>The plate's Young's modulus (ν = 0.3).</p>",
    "sides": "<p>Fluid on one side of the plate (a tank's wall, a hull), or both (a plate under water).</p>",
    "m": "<p>The mode's half waves along a.</p>",
    "n": "<p>The mode's half waves along b.</p>",
    "length": "<p>The tank's length L, along the direction it moves in.</p>",
    "depth": "<p>The liquid's depth h.</p>",
    "breadth": "<p>The tank's breadth B, across the motion (it scales the masses, not the frequencies).</p>",
    "slosh": "<p>The sloshing mode drawn: n half waves along the tank's length.</p>",
    "structure_mass": "<p>The structure's mass in the mode the damper is for (its modal mass).</p>",
    "structure_f": "<p>The structure's natural frequency without the damper.</p>",
    "structure_zeta": "<p>The structure's own damping ratio.</p>",
    "tanks": "<p>How many identical tanks the structure carries.</p>",
    "damper_zeta": (
        "<p>The sloshing's damping ratio: about 0.5 % for a bare tank, more with screens or baffles in it.</p>"
    ),
}
FLOW_TIP = (
    "<p>The fluid's motion as the body moves to the right at speed U: its speed over U in colour, and its "
    "streamlines. For a sphere, the plane through its line of motion.</p>"
)
AXIS_TIP = (
    "<p>The fluid's speed ahead of the body, along its line of motion, over the body's: R²/r² for a cylinder and "
    "R³/r³ for a sphere in open fluid (dashed), zero at a wall.</p>"
)
COEFF_TIP = (
    "<p>The added-mass coefficient C<sub>a</sub> = m<sub>a</sub>/ρ<sub>f</sub>V against the wall's radius over "
    "the body's: ½ and 1 far away (dotted), without bound as the gap closes.</p>"
)
RATIO_TIP = (
    "<p>The natural frequency in the fluid over that in vacuum, against the body's density over the fluid's: "
    "1/√(1 + C<sub>a</sub>ρ<sub>f</sub>/ρ<sub>s</sub>).</p>"
)
BETA_TIP = (
    "<p>Each of the plate's first modes' AVMI factor β (its fluid's mass over its own modal mass) against its "
    "frequency in air. Ring: the mode picked.</p>"
)
WETDRY_TIP = (
    "<p>The first modes' frequencies in the fluid against in air: f<sub>dry</sub>/√(1 + β). The dashed line is "
    "f<sub>wet</sub> = f<sub>dry</sub>.</p>"
)

TANK_TIP = (
    "<p>The tank's section in the mode picked: the surface at its two extremes (solid and dashed), and the flow as "
    "it passes level, its speed in colour and its streamlines.</p>"
)
SLOSH_F_TIP = (
    "<p>The first three sloshing modes' frequencies against the liquid's depth, for this tank's length: "
    "√(gh)nπ/L/2π in shallow liquid (dashed), √(gnπ/L)/2π in deep (dotted). Ring: the mode picked.</p>"
)
SLOSH_M_TIP = (
    "<p>The spring–mass analogue's masses over the liquid's, against its depth over the tank's length: the rigid "
    "mass m<sub>0</sub> and the first two sloshing masses (the even modes have none).</p>"
)
RESPONSE_TIP = (
    "<p>The structure's response to a force, over its static deflection, with the liquid sloshing and as if it "
    "were frozen (the same mass, rigid).</p>"
)
TUNE_TIP = (
    "<p>The structure's peak response against the liquid's depth in the tanks (the tanks as set): the depth tunes "
    "the damper. Dotted: Den Hartog's peak √(1 + 2/μ) for this depth's μ.</p>"
)


def _kg(m: float) -> str:
    if m >= 1e4:
        return f"{m / 1000:.3g} t"
    return f"{m:.0f} kg" if m >= 100 else f"{m:.3g} kg"


def _hz(f: float) -> str:
    return f"{f:.3g} Hz"


NOTES_HTML = load_notes("fluid_structure", trials=trials_html(TRIALS))


class FluidStructurePage(QtWidgets.QWidget):
    """Controls on the left, the flow about the body in the centre, the added mass and a plate in water on the
    right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.trials = TrialBox(
            [t.name for t in TRIALS], "<p>Each sets up the body or the plate to show one idea from the Theory tab.</p>"
        )
        self.trials.picked.connect(self.load_trial)

        fluid = Form("Fluid")
        self.fluid = fluid.number("Density ρ<sub>f</sub>:", 0.1, 20000.0, WATER, 2, " kg/m³", TIPS["fluid"])

        body = Form("Body")
        self.shape = combo(SHAPE_NAMES, TIPS["shape"])
        self.shape.setCurrentIndex(Shape.CYLINDER)
        body.add("Shape:", self.shape, TIPS["shape"])
        self.radius = body.number("Radius R:", 1.0, 5000.0, 25.0, 1, " mm", TIPS["radius"])
        self.density = body.number("Density ρ<sub>s</sub>:", 1.0, 30000.0, 7850.0, 0, " kg/m³", TIPS["density"])
        self.f_dry = body.number("In vacuum f:", 0.1, 10000.0, 20.0, 2, " Hz", TIPS["f_dry"])
        self.walled = QtWidgets.QCheckBox("Inside a wall")
        body.add("Wall:", self.walled, TIPS["walled"])
        self.ratio = body.number("Wall radius R<sub>o</sub>/R:", 1.01, 100.0, 1.5, 3, "", TIPS["ratio"])
        self.info = note_label()
        body.row(self.info)
        connect_all(
            (self.fluid, self.shape, self.radius, self.density, self.f_dry, self.walled, self.ratio), self._on_body
        )

        # --- centre
        self.flow_plot = plot("y", "m", "x", "m", height=200)
        self.flow_plot.setToolTip(FLOW_TIP)
        self.flow_plot.setAspectLocked(True)
        self.flow_plot.setMouseEnabled(x=False, y=False)
        self.speed_image = pg.ImageItem(axisOrder="row-major")
        self.flow_plot.addItem(self.speed_image)
        self.streamlines = [pg.IsocurveItem(axisOrder="row-major") for _ in range(STREAMLINES)]
        for item in self.streamlines:
            item.setParentItem(self.speed_image)
        self.body_item = QtWidgets.QGraphicsEllipseItem()
        self.wall_item = QtWidgets.QGraphicsEllipseItem()
        self.arrow = pg.ArrowItem(angle=180, headLen=16, tailLen=26, tailWidth=5)  # the body's motion
        for item in (self.body_item, self.wall_item, self.arrow):
            self.flow_plot.addItem(item)

        self.axis_plot = plot("Fluid speed / U", None, "Distance r/R", None, height=160)
        self.axis_plot.setToolTip(AXIS_TIP)
        add_legend(self.axis_plot.getPlotItem(), offset=(-10, 10))
        self.axis_item = pg.PlotDataItem(name="this body")
        self.axis_refs = [pg.PlotDataItem(name=name) for name in ("cylinder, open: R²/r²", "sphere, open: R³/r³")]
        for item in (*self.axis_refs, self.axis_item):
            self.axis_plot.addItem(item)
        self.readout = readout_label()
        body_view = stack((self.flow_plot, 3), (self.axis_plot, 2))

        # --- right: added mass
        self.coeff_plot = plot("C<sub>a</sub>", None, "Wall R<sub>o</sub>/R", None, log_x=True, log_y=True, height=170)
        self.coeff_plot.setToolTip(COEFF_TIP)
        add_legend(self.coeff_plot.getPlotItem(), offset=(-10, 10))
        self.coeff_items = [pg.PlotDataItem(name=name) for name in ("cylinder", "sphere")]
        self.coeff_open = [pg.InfiniteLine(angle=0, pos=math.log10(c)) for c in (1.0, 0.5)]
        self.coeff_dot = marker(14)
        for item in (*self.coeff_open, *self.coeff_items, self.coeff_dot):
            self.coeff_plot.addItem(item)
        self.ratio_plot = plot(
            "f<sub>wet</sub>/f<sub>dry</sub>", None, "ρ<sub>s</sub>/ρ<sub>f</sub>", None, log_x=True, height=170
        )
        self.ratio_plot.setToolTip(RATIO_TIP)
        add_legend(self.ratio_plot.getPlotItem(), offset=(-10, -10))
        self.ratio_items = [pg.PlotDataItem(name=name) for name in ("cylinder, open", "sphere, open", "this body")]
        self.ratio_dot = marker(14)
        for item in (*self.ratio_items, self.ratio_dot):
            self.ratio_plot.addItem(item)
        self.body_note = note_label()
        self.body_tab = column(self.coeff_plot, self.ratio_plot, self.body_note, stretch=False)

        # --- right: a plate in water
        plate = Form("Plate (simply supported, baffled)")
        self.side_a = plate.number("Side a:", 0.05, 10.0, 0.6, 3, " m", TIPS["a"])
        self.side_b = plate.number("Side b:", 0.05, 10.0, 0.4, 3, " m", TIPS["b"])
        self.thickness = plate.number("Thickness h:", 0.1, 100.0, 3.0, 2, " mm", TIPS["thickness"])
        self.plate_rho = plate.number("Density ρ<sub>p</sub>:", 1.0, 30000.0, 7850.0, 0, " kg/m³", TIPS["plate_rho"])
        self.E = plate.number("Young's modulus E:", 0.01, 2000.0, 210.0, 2, " GPa", TIPS["E"])
        self.sides = plate.number("Wetted sides:", 1, 2, 1, 0, "", TIPS["sides"])
        self.mode_m = plate.number("Half waves m:", 1, 12, 1, 0, "", TIPS["m"])
        self.mode_n = plate.number("Half waves n:", 1, 12, 1, 0, "", TIPS["n"])
        connect_all((self.side_a, self.side_b, self.thickness, self.plate_rho, self.E, self.sides), self._on_plate)
        connect_all((self.mode_m, self.mode_n), self._on_mode)
        self.beta_plot = plot("β", None, "Frequency in air", "Hz", log_x=True, log_y=True, height=160)
        self.beta_plot.setToolTip(BETA_TIP)
        self.beta_items = [marker(7) for _ in range(3)]
        self.beta_dot = marker(15)
        for item in (*self.beta_items, self.beta_dot):
            self.beta_plot.addItem(item)
        self.wetdry_plot = plot("In the fluid", "Hz", "In air", "Hz", log_x=True, log_y=True, height=160)
        self.wetdry_plot.setToolTip(WETDRY_TIP)
        self.wetdry_line = pg.PlotDataItem()
        self.wetdry_items = marker(7)
        self.wetdry_dot = marker(15)
        for item in (self.wetdry_line, self.wetdry_items, self.wetdry_dot):
            self.wetdry_plot.addItem(item)
        self.plate_note = note_label()
        self.plate_tab = column(plate, self.beta_plot, self.wetdry_plot, self.plate_note, stretch=False)
        self.modes: PlateModes | None = None

        # --- right: sloshing
        tank = Form("Tank (rectangular)")
        self.length = tank.number("Length L:", 0.05, 100.0, 1.0, 3, " m", TIPS["length"])
        self.fill_depth = tank.number("Depth h:", 0.005, 100.0, 0.5, 3, " m", TIPS["depth"])
        self.breadth = tank.number("Breadth B:", 0.01, 100.0, 1.0, 3, " m", TIPS["breadth"])
        self.slosh = tank.number("Mode n:", 1, 9, 1, 0, "", TIPS["slosh"])
        connect_all((self.length, self.fill_depth, self.breadth), self._on_tank)
        self.slosh.valueChanged.connect(self._on_slosh)
        self.tank_plot = plot("z", "m", "x", "m", height=200)
        self.tank_plot.setToolTip(TANK_TIP)
        self.tank_plot.setAspectLocked(True)
        self.tank_plot.setMouseEnabled(x=False, y=False)
        self.tank_image = pg.ImageItem(axisOrder="row-major")
        self.tank_plot.addItem(self.tank_image)
        self.tank_lines = [pg.IsocurveItem(axisOrder="row-major") for _ in range(TANK_STREAMLINES)]
        for item in self.tank_lines:
            item.setParentItem(self.tank_image)
        self.surface = pg.PlotDataItem()
        self.surface_back = pg.PlotDataItem()
        self.tank_outline = pg.PlotDataItem()
        for item in (self.surface_back, self.surface, self.tank_outline):
            self.tank_plot.addItem(item)
        self.centre = QtWidgets.QStackedWidget()  # the body's flow, or the tank's
        self.centre.addWidget(body_view)
        self.centre.addWidget(self.tank_plot)
        centre = stack((self.centre, 1), (self.readout, 0))
        self.slosh_f_plot = plot("Frequency", "Hz", "Depth h", "m", log_x=True, height=160)
        self.slosh_f_plot.setToolTip(SLOSH_F_TIP)
        add_legend(self.slosh_f_plot.getPlotItem(), offset=(10, 5), colCount=3)
        self.slosh_f_items = [pg.PlotDataItem(name=f"n = {n}") for n in (1, 2, 3)]
        self.slosh_f_refs = [pg.PlotDataItem(), pg.PlotDataItem()]  # shallow, deep
        self.slosh_f_dot = marker(14)
        for item in (*self.slosh_f_refs, *self.slosh_f_items, self.slosh_f_dot):
            self.slosh_f_plot.addItem(item)
        self.slosh_m_plot = plot("Mass / m<sub>L</sub>", None, "Depth h/L", None, log_x=True, height=160)
        self.slosh_m_plot.setToolTip(SLOSH_M_TIP)
        add_legend(self.slosh_m_plot.getPlotItem(), offset=(10, 5), colCount=3)
        self.slosh_m_items = [pg.PlotDataItem(name=name) for name in ("rigid m₀", "m₁", "m₃")]
        self.slosh_m_line = pg.InfiniteLine(angle=90)
        for item in (*self.slosh_m_items, self.slosh_m_line):
            self.slosh_m_plot.addItem(item)
        self.slosh_tab = column(tank, self.slosh_f_plot, self.slosh_m_plot, stretch=False)

        # --- right: a tuned liquid damper
        damper = Form("Structure and damper")
        self.structure_mass = damper.number("Structure's mass M:", 0.01, 1e6, 20.0, 2, " t", TIPS["structure_mass"])
        self.structure_f = damper.number("Its frequency f:", 0.01, 100.0, 0.5, 3, " Hz", TIPS["structure_f"])
        self.structure_zeta = damper.number("Its damping ζ:", 0.0, 50.0, 1.0, 2, " %", TIPS["structure_zeta"])
        self.tanks = damper.number("Tanks:", 1, 100, 1, 0, "", TIPS["tanks"])
        self.damper_zeta = damper.number(
            "Sloshing damping ζ<sub>d</sub>:", 0.01, 50.0, 13.0, 2, " %", TIPS["damper_zeta"]
        )
        damper.row(note_label("The tank is the Sloshing tab's."))
        connect_all(
            (self.structure_mass, self.structure_f, self.structure_zeta, self.tanks, self.damper_zeta),
            self._on_damper,
        )
        self.response_plot = plot("|X| k/F", None, "Frequency", "Hz", log_y=True, height=170)
        self.response_plot.setToolTip(RESPONSE_TIP)
        add_legend(self.response_plot.getPlotItem(), offset=(-10, 10))
        self.response_items = [pg.PlotDataItem(name=name) for name in ("liquid frozen", "sloshing")]
        for item in self.response_items:
            self.response_plot.addItem(item)
        self.tune_plot = plot("Peak |X| k/F", None, "Depth h", "m", log_y=True, height=170)
        self.tune_plot.setToolTip(TUNE_TIP)
        self.tune_item = pg.PlotDataItem()
        self.tune_ref = pg.InfiniteLine(angle=0)
        self.tune_dot = marker(14)
        for item in (self.tune_ref, self.tune_item, self.tune_dot):
            self.tune_plot.addItem(item)
        self.damper_note = note_label()
        self.damper_tab = column(damper, self.response_plot, self.tune_plot, self.damper_note, stretch=False)

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.body_tab, "Added mass")
        self.tabs.addTab(ColumnScroll(self.plate_tab), "Plate")
        self.tabs.addTab(ColumnScroll(self.slosh_tab), "Sloshing")
        self.tabs.addTab(ColumnScroll(self.damper_tab), "Damper")
        self.tabs.addTab(self.notes, "Theory")
        self.lazy = LazyTabs(
            self.tabs,
            {
                self.body_tab: self._draw_body,
                self.plate_tab: self._draw_plate,
                self.slosh_tab: self._draw_slosh,
                self.damper_tab: self._draw_damper,
            },
        )

        self.tabs.currentChanged.connect(self._on_tab)
        three_panes(self, column(self.trials, fluid, body), centre, self.tabs)
        self.apply_theme()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(FluidTrial("", "", **setup))

    def practice_state(self) -> FluidTrial:
        """The page's settings, as a trial: what a practice spec checks."""
        return FluidTrial(
            "",
            "",
            fluid=self.fluid.value(),
            shape=self.shape.currentIndex(),
            radius=self.radius.value(),
            density=self.density.value(),
            f_dry=self.f_dry.value(),
            walled=self.walled.isChecked(),
            ratio=self.ratio.value(),
            a=self.side_a.value(),
            b=self.side_b.value(),
            thickness=self.thickness.value(),
            plate_rho=self.plate_rho.value(),
            E=self.E.value(),
            sides=int(self.sides.value()),
            m=int(self.mode_m.value()),
            n=int(self.mode_n.value()),
            length=self.length.value(),
            depth=self.fill_depth.value(),
            breadth=self.breadth.value(),
            slosh=int(self.slosh.value()),
            structure_mass=self.structure_mass.value(),
            structure_f=self.structure_f.value(),
            structure_zeta=self.structure_zeta.value(),
            tanks=int(self.tanks.value()),
            damper_zeta=self.damper_zeta.value(),
            tab=self.tabs.currentIndex(),
        )

    def apply_trial(self, t: FluidTrial) -> None:
        for box, value in (
            (self.fluid, t.fluid),
            (self.shape, t.shape),
            (self.radius, t.radius),
            (self.density, t.density),
            (self.f_dry, t.f_dry),
            (self.walled, t.walled),
            (self.ratio, t.ratio),
            (self.side_a, t.a),
            (self.side_b, t.b),
            (self.thickness, t.thickness),
            (self.plate_rho, t.plate_rho),
            (self.E, t.E),
            (self.sides, t.sides),
            (self.mode_m, t.m),
            (self.mode_n, t.n),
            (self.length, t.length),
            (self.fill_depth, t.depth),
            (self.breadth, t.breadth),
            (self.slosh, t.slosh),
            (self.structure_mass, t.structure_mass),
            (self.structure_f, t.structure_f),
            (self.structure_zeta, t.structure_zeta),
            (self.tanks, t.tanks),
            (self.damper_zeta, t.damper_zeta),
        ):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self.modes = None
        self._on_body()

    @property
    def body(self) -> Body:
        R = self.radius.value() / 1000
        outer = R * self.ratio.value() if self.walled.isChecked() else math.inf
        return Body(Shape(self.shape.currentIndex()), R, self.density.value(), self.fluid.value(), outer)

    @property
    def plate(self) -> Plate:
        return Plate(
            self.side_a.value(),
            self.side_b.value(),
            self.thickness.value() / 1000,
            Material(self.E.value() * 1e9, 0.3, self.plate_rho.value()),
        )

    @property
    def tank(self) -> Tank:
        return Tank(self.length.value(), self.fill_depth.value(), self.breadth.value(), self.fluid.value())

    @property
    def structure(self) -> Structure:
        return Structure(
            self.structure_mass.value() * 1000, self.structure_f.value(), self.structure_zeta.value() / 100
        )

    @property
    def damper(self) -> Damper:
        return Damper(self.tank, int(self.tanks.value()), self.damper_zeta.value() / 100)

    @property
    def showing_tank(self) -> bool:
        return self.tabs.currentIndex() in (TAB_SLOSH, TAB_DAMPER)

    def _on_tab(self) -> None:
        tank = self.showing_tank
        if tank != (self.centre.currentIndex() == 1):
            self.centre.setCurrentIndex(1 if tank else 0)
            self._draw_centre()

    def _on_tank(self) -> None:
        self.lazy.stale(self.slosh_tab)
        self.lazy.stale(self.damper_tab)
        if self.showing_tank:
            self._draw_tank()

    def _on_slosh(self) -> None:
        self.lazy.stale(self.slosh_tab)
        if self.showing_tank:
            self._draw_tank()

    def _on_damper(self) -> None:
        self.lazy.stale(self.damper_tab)

    def _on_body(self) -> None:
        self.ratio.setEnabled(self.walled.isChecked())
        self.lazy.stale()  # the fluid's density reaches the plate too
        self._draw_centre()

    def _on_plate(self) -> None:
        self.modes = None
        self.lazy.stale(self.plate_tab)

    def _on_mode(self) -> None:
        self.lazy.stale(self.plate_tab)

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        clear = pg.mkColor(c.mass[0])
        clear.setAlpha(0)  # still fluid shows the grid through it
        self.speed_image.setLookupTable(
            pg.ColorMap([0.0, 1.0], [clear, pg.mkColor(c.mass[0])]).getLookupTable(nPts=256, alpha=True)
        )
        for item in self.streamlines:
            item.setPen(pg.mkPen(c.strong, width=1))
        self.body_item.setPen(pg.mkPen(c.structure, width=2))
        self.body_item.setBrush(QtGui.QBrush(pg.mkColor(c.structure)))
        self.wall_item.setPen(pg.mkPen(c.structure, width=4))
        self.arrow.setStyle(pen=pg.mkPen(c.force), brush=pg.mkBrush(c.force))
        dashed = QtCore.Qt.PenStyle.DashLine
        self.axis_item.setPen(pg.mkPen(c.mass[0], width=3))
        for item, color in zip(self.axis_refs, (c.grey, c.muted)):
            item.setPen(pg.mkPen(color, width=1, style=dashed))
        for item, color in zip(self.coeff_items, (c.mass[0], c.mass[1])):
            item.setPen(pg.mkPen(color, width=2))
        for line in self.coeff_open:
            line.setPen(pg.mkPen(c.faint, width=1, style=QtCore.Qt.PenStyle.DotLine))
        mark(self.coeff_dot, c.force, width=3)
        for item, color, style in zip(
            self.ratio_items,
            (c.mass[0], c.mass[1], c.strong),
            (QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenStyle.SolidLine, dashed),
        ):
            item.setPen(pg.mkPen(color, width=2, style=style))
        mark(self.ratio_dot, c.force, width=3)
        for item, color in zip(self.beta_items, (c.mass[0], c.mass[1], c.mass[3])):
            mark(item, color, filled=True)
        mark(self.beta_dot, c.force, width=3)
        self.wetdry_line.setPen(pg.mkPen(c.grey, width=1, style=dashed))
        mark(self.wetdry_items, c.mass[0], filled=True)
        mark(self.wetdry_dot, c.force, width=3)
        self.tank_image.setLookupTable(
            pg.ColorMap([0.0, 1.0], [clear, pg.mkColor(c.mass[0])]).getLookupTable(nPts=256, alpha=True)
        )
        for item in self.tank_lines:
            item.setPen(pg.mkPen(c.strong, width=1))
        self.surface.setPen(pg.mkPen(c.mass[0], width=3))
        self.surface_back.setPen(pg.mkPen(c.mass[0], width=2, style=dashed))
        self.tank_outline.setPen(pg.mkPen(c.structure, width=4))
        for item, color in zip(self.slosh_f_items, (c.mass[0], c.mass[1], c.mass[3])):
            item.setPen(pg.mkPen(color, width=2))
        self.slosh_f_refs[0].setPen(pg.mkPen(c.grey, width=1, style=dashed))
        self.slosh_f_refs[1].setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DotLine))
        mark(self.slosh_f_dot, c.force, width=3)
        for item, color in zip(self.slosh_m_items, (c.strong, c.mass[0], c.mass[1])):
            item.setPen(pg.mkPen(color, width=2))
        self.slosh_m_line.setPen(pg.mkPen(c.force, width=2))
        self.response_items[0].setPen(pg.mkPen(c.grey, width=2, style=dashed))
        self.response_items[1].setPen(pg.mkPen(c.mass[0], width=3))
        self.tune_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.tune_ref.setPen(pg.mkPen(c.muted, width=1, style=QtCore.Qt.PenStyle.DotLine))
        mark(self.tune_dot, c.force, width=3)
        self._draw_centre()
        self.lazy.stale()

    def _draw_centre(self) -> None:
        if self.showing_tank:
            self._draw_tank()
        else:
            self._draw_flow()

    def _draw_flow(self) -> None:
        body = self.body
        R = body.radius
        half = 1.12 * body.outer if math.isfinite(body.outer) else 4.0 * R
        x = np.linspace(-half, half, GRID)
        f = flow(body, x, x)
        speed = np.hypot(f.u, f.v)
        top = float(np.nanmax(speed))
        self.speed_image.setImage(np.nan_to_num(speed, nan=0.0), levels=(0.0, top), autoLevels=False)
        self.speed_image.setRect(QtCore.QRectF(-half, -half, 2 * half, 2 * half))
        reach = float(np.max(np.abs(f.contours)))
        levels = np.linspace(-reach, reach, STREAMLINES + 2)[1:-1]
        levels = levels[np.abs(levels) > 1e-9 * reach]  # ψ = 0 is the axis (and everything past the wall)
        for item, level in zip(self.streamlines, levels):
            item.setData(f.contours)
            item.setLevel(float(level))
            item.setVisible(True)
        for item in self.streamlines[levels.size :]:
            item.setVisible(False)
        self.body_item.setRect(QtCore.QRectF(-R, -R, 2 * R, 2 * R))
        if math.isfinite(body.outer):
            Ro = body.outer
            self.wall_item.setRect(QtCore.QRectF(-Ro, -Ro, 2 * Ro, 2 * Ro))
        self.wall_item.setVisible(math.isfinite(body.outer))
        self.arrow.setPos(R * 0.8, 0.0)
        self.flow_plot.setXRange(-half, half, padding=0.02)
        self.flow_plot.setYRange(-half, half, padding=0.02)

        far = body.outer / R if math.isfinite(body.outer) else 10.0
        r = np.geomspace(1.0, far, 200)
        self.axis_item.setData(r, axis_speed(body, r * R))
        for item, shape in zip(self.axis_refs, (Shape.CYLINDER, Shape.SPHERE)):
            ref = np.geomspace(1.0, max(far, 10.0), 200)
            item.setData(ref, axis_speed(Body(shape, 1.0), ref))
        self.axis_plot.setXRange(1.0, far, padding=0.02)  # inside a wall, just the gap
        self.axis_plot.setYRange(0.0, 1.0, padding=0.03)

        wet = wet_frequency(self.f_dry.value(), body.mass, body.added_mass)
        per = " a metre" if body.shape == Shape.CYLINDER else ""
        self.info.setText(
            f"m = {fmt(body.mass, 'kg')}{per}, the fluid it displaces {fmt(body.fluid * body.volume, 'kg')}{per}."
        )
        self.readout.setText(
            f"C<sub>a</sub> = {body.coefficient:.4g}: added mass m<sub>a</sub> = {fmt(body.added_mass, 'kg')}{per} on the "
            f"body's {fmt(body.mass, 'kg')}. On its springs, {fmt(self.f_dry.value(), 'Hz')} in vacuum becomes "
            f"{fmt(wet, 'Hz')} in the fluid ({wet / self.f_dry.value():.3g} of it)."
        )
        settle_plots(self)

    def _draw_body(self) -> None:
        body = self.body
        for item, shape in zip(self.coeff_items, (Shape.CYLINDER, Shape.SPHERE)):
            item.setData(RATIOS, [Body(shape, 1.0, outer=r).coefficient for r in RATIOS])
        if math.isfinite(body.outer):
            self.coeff_dot.setData([body.outer / body.radius], [body.coefficient])
        else:
            self.coeff_dot.setData([], [])
        self.coeff_plot.setXRange(math.log10(RATIOS[0]), math.log10(RATIOS[-1]), padding=0.02)
        for item, c in zip(self.ratio_items, (1.0, 0.5, body.coefficient)):
            item.setData(DENSITIES, 1.0 / np.sqrt(1.0 + c / DENSITIES))
        rho = body.density / body.fluid
        self.ratio_dot.setData([rho], [1.0 / math.sqrt(1.0 + body.coefficient / rho)])
        self.ratio_plot.setXRange(math.log10(DENSITIES[0]), math.log10(DENSITIES[-1]), padding=0.02)
        self.ratio_plot.setYRange(0.0, 1.02, padding=0)
        wall = (
            f" The wall at {body.outer / body.radius:.3g} R raises it from the open fluid's "
            f"{Body(body.shape).coefficient:g}."
            if math.isfinite(body.outer)
            else ""
        )
        self.body_note.setText(
            f"C<sub>a</sub> = {body.coefficient:.4g}.{wall} The body is {rho:.3g} times as dense as the fluid, so "
            f"its frequency falls to 1/√(1 + C<sub>a</sub>ρ<sub>f</sub>/ρ<sub>s</sub>) = "
            f"{1 / math.sqrt(1 + body.coefficient / rho):.3g} of the vacuum's."
        )
        settle_plots(self)

    def _plate_modes(self) -> PlateModes:
        if self.modes is None:
            self.modes = plate_modes(self.plate, PLATE_MODES, self.fluid.value(), int(self.sides.value()))
        return self.modes

    def _draw_plate(self) -> None:
        modes = self._plate_modes()
        parity = (modes.m % 2) + (modes.n % 2)
        for item, which in zip(self.beta_items, (2, 1, 0)):
            sel = parity == which
            item.setData(modes.dry[sel], modes.beta[sel])
        p, m, n = self.plate, int(self.mode_m.value()), int(self.mode_n.value())
        fluid, sides = self.fluid.value(), int(self.sides.value())
        beta = avmi(p, m, n, fluid, sides)
        dry = plate_dry_frequency(p, m, n)
        wet = dry / math.sqrt(1 + beta)
        self.beta_dot.setData([dry], [beta])
        self.wetdry_items.setData(modes.dry, modes.wet)
        self.wetdry_dot.setData([dry], [wet])
        lo = float(min(modes.wet.min(), wet)) / 1.3
        hi = float(max(modes.dry.max(), dry)) * 1.3
        self.wetdry_line.setData([lo, hi], [lo, hi])
        for plot_ in (self.beta_plot, self.wetdry_plot):
            plot_.setXRange(math.log10(min(float(modes.dry.min()), dry) / 1.3), math.log10(hi), padding=0)
        self.wetdry_plot.setYRange(math.log10(lo), math.log10(hi), padding=0)
        added = beta * plate_modal_mass(p)
        side = "both sides" if sides == 2 else "one side"
        self.plate_note.setText(
            f"Mode ({m}, {n}), {fmt(dry, 'Hz')} in air, with fluid on {side}: its added mass is "
            f"{fmt(added, 'kg')} against its modal mass ρ<sub>p</sub>hab/4 = {fmt(plate_modal_mass(p), 'kg')}, "
            f"β = {beta:.3g}, and it falls to {fmt(wet, 'Hz')} ({wet / dry:.3g} of it). Dots: the first "
            f"{PLATE_MODES} modes, by the parity of m and n (odd–odd, mixed, even–even). The fluid's coupling "
            "between modes is left out."
        )
        settle_plots(self)

    # ------------------------------------------------------------- sloshing
    def _draw_tank(self) -> None:
        tank = self.tank
        n = int(self.slosh.value())
        L, h = tank.length, tank.depth
        cells = TANK_GRID / max(L, h)
        x = np.linspace(0.0, L, max(int(cells * L), 24))
        z = np.linspace(-h, 0.0, max(int(cells * h), 12))
        f = tank_flow(tank, n, x, z)
        self.tank_image.setImage(f.speed, levels=(0.0, 1.0), autoLevels=False)
        self.tank_image.setRect(QtCore.QRectF(0.0, -h, L, h))
        levels = np.linspace(-1.0, 1.0, TANK_STREAMLINES + 2)[1:-1]
        for item, level in zip(self.tank_lines, levels):
            item.setData(f.stream)
            item.setLevel(float(level))
        amp = 0.15 * min(h, L / n)  # the surface's swing, drawn
        self.surface.setData(x, amp * f.surface)
        self.surface_back.setData(x, -amp * f.surface)
        top = 1.8 * amp
        self.tank_outline.setData([0.0, 0.0, L, L], [top, -h, -h, top])
        # both at once: with the aspect locked, the second of two calls would crop the first
        self.tank_plot.getViewBox().setRange(xRange=(0.0, L), yRange=(-h, top), padding=0.04)
        fn = slosh_frequency(tank, n)
        mL, m0, mn = tank.liquid_mass, rigid_mass(tank), float(slosh_mass(tank, n))
        excited = (
            f"its analogue mass is {_kg(mn)}, {mn / mL:.3g} of the liquid's"
            if n % 2
            else "it's symmetric, so a sideways motion doesn't excite it (no analogue mass)"
        )
        self.readout.setText(
            f"Mode {n} sloshes at {_hz(fn)} (k = nπ/L, ω² = gk tanh(kh)); {excited}. Of the {_kg(mL)} of liquid, "
            f"m<sub>0</sub> = {_kg(m0)} moves with the tank, and the first mode's m<sub>1</sub> = "
            f"{_kg(float(slosh_mass(tank, 1)))} sloshes at {_hz(slosh_frequency(tank, 1))}."
        )
        settle_plots(self)

    def _draw_slosh(self) -> None:
        tank = self.tank
        n = int(self.slosh.value())
        L, h = tank.length, tank.depth
        hs = DEPTH_RATIOS * L
        for item, mode in zip(self.slosh_f_items, (1, 2, 3)):
            item.setData(hs, [slosh_frequency(Tank(L, d), mode) for d in hs])
        self.slosh_f_refs[0].setData(hs, np.sqrt(G * hs) / (2 * L))
        deep = math.sqrt(G * math.pi / L) / (2 * math.pi)
        self.slosh_f_refs[1].setData(hs, np.full_like(hs, deep))
        self.slosh_f_plot.setYRange(0.0, 1.35 * float(slosh_frequency(Tank(L, 10 * L), 3)), padding=0)
        fn = slosh_frequency(tank, n)
        self.slosh_f_dot.setData([h], [fn])
        lo, hi = min(float(hs[0]), h), max(float(hs[-1]), h)
        self.slosh_f_plot.setXRange(math.log10(lo), math.log10(hi), padding=0.02)

        masses = [
            [rigid_mass(Tank(1.0, r)) / Tank(1.0, r).liquid_mass for r in DEPTH_RATIOS],
            [float(slosh_mass(Tank(1.0, r), 1)) / Tank(1.0, r).liquid_mass for r in DEPTH_RATIOS],
            [float(slosh_mass(Tank(1.0, r), 3)) / Tank(1.0, r).liquid_mass for r in DEPTH_RATIOS],
        ]
        for item, values in zip(self.slosh_m_items, masses):
            item.setData(DEPTH_RATIOS, values)
        self.slosh_m_line.setPos(math.log10(h / L))
        ratio_lo, ratio_hi = min(float(DEPTH_RATIOS[0]), h / L), max(float(DEPTH_RATIOS[-1]), h / L)
        self.slosh_m_plot.setXRange(math.log10(ratio_lo), math.log10(ratio_hi), padding=0.02)
        self.slosh_m_plot.setYRange(0.0, 1.25, padding=0)

        settle_plots(self)

    def _draw_damper(self) -> None:
        structure, damper = self.structure, self.damper
        loaded = loaded_frequency(structure, damper)
        f = np.linspace(0.6 * loaded, 1.5 * loaded, 1500)
        sloshing = np.abs(damper_response(structure, damper, f))
        frozen = np.abs(frozen_response(structure, damper, f))
        self.response_items[0].setData(f, frozen)
        self.response_items[1].setData(f, sloshing)
        self.response_plot.setXRange(float(f[0]), float(f[-1]), padding=0)

        tank = damper.tank
        top = max(1.2 * tank.length, 1.1 * tank.depth)
        depths = np.linspace(0.02 * tank.length, top, DEPTHS)
        peaks = peak_against_depth(structure, damper, depths)
        here = peak(structure, damper)
        self.tune_item.setData(depths, peaks)
        self.tune_dot.setData([tank.depth], [here])
        mu = mass_ratio(structure, damper)
        tuning, zeta_opt, dh_peak = den_hartog(mu)
        self.tune_ref.setPos(math.log10(dh_peak))
        self.tune_plot.setXRange(0.0, top, padding=0.02)

        frozen_peak = float(np.abs(frozen_response(structure, damper, response_band(structure))).max())
        m1 = float(damper.masses[0])
        self.damper_note.setText(
            f"Of the {_kg(damper.liquid)} of liquid, the first sloshing mode's {_kg(m1)}, at "
            f"{_hz(slosh_frequency(tank, 1))}, is a tuned mass damper with μ = {mu:.3g}; the rest moves with the "
            f"structure and lowers it to {_hz(loaded)}. Peak: {here:.3g} times the static deflection (frozen, "
            f"{frozen_peak:.3g}). Den Hartog's tuning for this μ: f<sub>d</sub> = {_hz(tuning * loaded)}, "
            f"ζ<sub>d</sub> = {100 * zeta_opt:.3g} %, a peak of {dh_peak:.3g} (with no damping in the structure)."
        )
        settle_plots(self)
