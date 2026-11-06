"""Rotor builder page: build a rotor from shaft sections, discs, bearings and seals, and see it whirl.

Left: Things to try, the selected part's properties (a section's length, diameters, material and elements; a
disc from the library or by its mass and inertias; a bearing's springs and dampers, oil film or magnet; a seal's
stiffness, damping and swirl), the model's settings (beam theory, horizontal or vertical, the speed) and what to
show. Centre: the tools, the side view (click the shaft to split it or to put a part there, drag a part along
it), bending in the mode picked, and the modes at the speed with their damping and whirl. Right: the mode
whirling in 3D, the global matrices with the picked part's share tinted, the part's own matrices, and the
Theory. The page has its own model (core/rotor_builder.py) and swings the mode on its own frame loop while it
shows.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.fluid_bearing import JournalBearing
from ..core.frame import MATERIALS, STEEL
from ..core.magnetic_bearing import MagneticBearing
from ..core.practice import Problem, Question, Requirement, at_least, kept, number, pick
from ..core.rotor_builder import (
    BEARING_KINDS,
    DISC_LIBRARY,
    GRAVITY,
    MAX_ELEMENTS,
    RPM,
    THEORIES,
    TWO_PI,
    Assembly,
    Bearing,
    BearingKind,
    Disc,
    Part,
    Rotor,
    RotorError,
    Seal,
    Section,
    Selection,
    Theory,
    WhirlModes,
    assemble,
    deflection,
    dof_label,
    film_state,
    modes,
    pinned_ends,
    solid_disc,
    uniform,
)
from . import frame_stats, settle_plots, weak
from .frame_builder import matrix_html
from .layouts import FlowLayout, group
from .lesson import (
    Form,
    RunButton,
    TrialBox,
    column,
    combo,
    fmt,
    note_label,
    readout_label,
    set_quietly,
    theory,
    three_panes,
    trials_html,
)
from .links import load_notes, page_link
from .style import colors
from .theming import mute

TAB_WHIRL, TAB_ASSEMBLY, TAB_ELEMENT, TAB_THEORY = range(4)
FRAME_MS = 16
SHOW_HZ = 0.6  # the animation's own pace: every mode whirls at it
SWING = 0.15  # the largest lateral motion drawn, as a fraction of the side view's height
MODE_ROWS = 20  # modes listed at most
PICK_PX = 8  # how near a click has to be to pick a part
GRIDS = (0.005, 0.01, 0.025, 0.05, 0.1)  # m
MATRIX_LIMIT = 40  # DOFs: beyond this the Assembly tab shows the nodes round the picked part
AXIAL = 4.0  # the 3D view's display length of the shaft
WHIRL = 0.45  # display units: the 3D view's largest whirl
NAN2 = np.full(2, np.nan)


class Tool(Enum):
    SELECT = "Select"
    SPLIT = "Split"
    DISC = "Disc"
    BEARING = "Bearing"
    SEAL = "Seal"


TOOLS = tuple(Tool)
TOOL_HINTS = {
    Tool.SELECT: "Click a section or a part to edit it on the left; drag a part along the shaft. Delete removes it.",
    Tool.SPLIT: "Click the shaft to cut its section in two there (each keeps its share of the elements).",
    Tool.DISC: "Pick a disc from the library (or Custom), then click the shaft to put it there.",
    Tool.BEARING: "Pick a kind of bearing, then click the shaft to put one there.",
    Tool.SEAL: "Click the shaft to put a seal there.",
}
PART_OF_TOOL = {Tool.DISC: "disc", Tool.BEARING: "bearing", Tool.SEAL: "seal"}


class Matrix(Enum):
    K = "Stiffness K(Ω)"
    M = "Mass M"
    C = "Damping C(Ω)"
    G = "Gyroscopic G"


MATRICES = tuple(Matrix)
EMPTY = Rotor()


@dataclass(frozen=True)
class RotorTrial:
    name: str
    note: str
    rotor: Rotor = EMPTY
    rpm: float = 0.0
    mode: int = 0
    tab: int = TAB_WHIRL
    picked: Selection | None = None  # the part on the Element tab (and tinted on the Assembly tab)
    matrix: Matrix = Matrix.K
    grid: float = 0.05


# ------------------------------------------------------------------ the trials' rotors
UNIFORM = pinned_ends(Rotor(uniform(1.0, 0.05, 4, 3)))
STUBBY = pinned_ends(Rotor(uniform(0.5, 0.1, 4, 3)))
QUARTER_DISC = dataclasses.replace(UNIFORM, discs=(DISC_LIBRARY[1].at(0.25),))
FREE = Rotor(uniform(1.0, 0.05, 4, 3))
SPRINGS = Bearing(0.0, kxx=2e7, kyy=2e7, cxx=500.0, cyy=500.0)
OVERHUNG = Rotor(
    (Section(0.6, 0.06, 0.0, STEEL, 6), Section(0.2, 0.06, 0.0, STEEL, 2)),
    discs=(DISC_LIBRARY[4].at(0.8),),
    bearings=(SPRINGS, dataclasses.replace(SPRINGS, z=0.6)),
)
SOFT_Y = dataclasses.replace(SPRINGS, kxx=4e6, kyy=1e6, cxx=200.0, cyy=200.0)
ANISOTROPIC = Rotor(
    uniform(1.0, 0.05, 4, 3),
    discs=(DISC_LIBRARY[1].at(0.3),),
    bearings=(SOFT_Y, dataclasses.replace(SOFT_Y, z=1.0)),
)
COMPRESSOR = Rotor(
    (
        Section(0.15, 0.06, 0.0, STEEL, 2),
        Section(0.1, 0.08, 0.0, STEEL, 2),
        Section(0.4, 0.1, 0.04, STEEL, 4),
        Section(0.1, 0.08, 0.0, STEEL, 2),
        Section(0.15, 0.06, 0.0, STEEL, 2),
    ),
    discs=(DISC_LIBRARY[3].at(0.35), DISC_LIBRARY[3].at(0.55), DISC_LIBRARY[5].at(0.9)),
    bearings=(
        Bearing(0.2, kxx=5e7, kyy=5e7, cxx=5e3, cyy=5e3),
        Bearing(0.7, kxx=5e7, kyy=5e7, cxx=5e3, cyy=5e3),
    ),
)
FILM = JournalBearing(length=0.03, diameter=0.05, clearance=50e-6, viscosity=0.02, load=0.0)
FILMS = Rotor(
    uniform(1.0, 0.05, 4, 3),
    discs=(DISC_LIBRARY[1].at(0.5),),
    bearings=(Bearing(0.0, BearingKind.FILM, film=FILM), Bearing(1.0, BearingKind.FILM, film=FILM)),
)
MAGNET = MagneticBearing(gap=0.5e-3, bias=2.0, area=6e-4, turns=150, kp=2e4, ki=2e5, kd=20.0)
MAGNETS = Rotor(
    uniform(1.0, 0.05, 4, 3),
    discs=(DISC_LIBRARY[1].at(0.5),),
    bearings=(Bearing(0.0, BearingKind.MAGNET, magnet=MAGNET), Bearing(1.0, BearingKind.MAGNET, magnet=MAGNET)),
)
SEALED = dataclasses.replace(
    ANISOTROPIC,
    discs=(DISC_LIBRARY[1].at(0.5),),
    bearings=(SPRINGS, dataclasses.replace(SPRINGS, z=1.0)),
    seals=(Seal(0.25, 2e6, 2e3, 0.5),),
)

TRIALS = (
    RotorTrial(
        "A uniform shaft, pinned",
        "A steel shaft 1 m long and 50 mm across, four sections of three Timoshenko beams, on stiff bearings at "
        "its ends (nearly pinned). Every frequency comes twice, 101.25, 401.5 and 891.3 Hz, once whirling forward "
        "and once backward: not spinning, a round shaft can't tell them apart. Euler's pinned beam has 101.56, "
        "406.2 and 914.0 Hz; Euler–Bernoulli beams (Beams, on the left) give 101.48, 405.0 and 908.0. The "
        "sections' turning inertia takes a little off, shear a little more.",
        UNIFORM,
        picked=("section", 0),
    ),
    RotorTrial(
        "A stubby shaft: Timoshenko",
        "Half a metre long and 100 mm across (L/D = 5), pinned: Timoshenko's beams give 776.3, 2784 and 5506 Hz, "
        "Euler–Bernoulli's (with the rotary inertia) 802.5, 3099 and 6610, Euler's formula 812.4, 3250 and 7312. "
        "Shear softens the higher modes most: the third is a quarter below Euler's.",
        STUBBY,
    ),
    RotorTrial(
        "A disc off centre: the gyroscopic split",
        "The pinned shaft with a steel disc 300 × 40 mm (22.2 kg) a quarter along, at 3000 rpm. At rest its pairs "
        "are at 62.28, 265.2 and 501.4 Hz; spinning, each splits, backward below and forward above: 60.83 and "
        "63.70, 261.2 and 268.6, 468.2 and 538.2 Hz. The disc tilts as the shaft bends, most in the third mode, "
        "which splits most. G (the Assembly tab) is where it comes from: I<sub>p</sub> between the disc's slopes.",
        QUARTER_DISC,
        rpm=3000.0,
        picked=("disc", 0),
        matrix=Matrix.G,
    ),
    RotorTrial(
        "An overhung fan",
        "A titanium fan 500 × 20 mm (17.4 kg) 200 mm past the right-hand bearing of a 60 mm steel shaft. At rest "
        "the first pair is at 79.06 Hz; at 3000 rpm it splits to 71.13 (backward) and 86.24 Hz (forward), at 6000 "
        "rpm to 63.10 and 92.30. An overhung disc tilts as it whirls, and a thin one (I<sub>p</sub> ≈ "
        "2I<sub>d</sub>) stiffens its forward whirl strongly.",
        OVERHUNG,
        rpm=3000.0,
        picked=("disc", 0),
    ),
    RotorTrial(
        "Free: no bearings",
        "The 50 mm shaft with nothing holding it. At rest there are 8 roots at 0 (its two translations and two "
        "tilts, each twice: a free body has no stiffness), then pairs at 228.7 and 623.3 Hz (Euler's free–free "
        "beam: 230.2 Hz). At 3000 rpm two of the zero roots become a slow forward whirl at 0.19 Hz: the free "
        "shaft's nutation, I<sub>p</sub>Ω/I<sub>d</sub>, I<sub>p</sub>/I<sub>d</sub> = 1.5D²/L² = 0.00375 of its 50 "
        "rev/s.",
        FREE,
        rpm=3000.0,
    ),
    RotorTrial(
        "Soft bearings, stiffer across",
        "Bearings of 4 MN/m sideways and 1 MN/m up, and the 300 mm disc at 0.3 m. At rest the pairs part, each "
        "into a straight line (whirl ratio 0): 30.58 Hz moving only up and down (the soft way), 45.44 Hz only "
        "sideways. At 6000 rpm the disc's gyroscopic moment couples the two ways into ellipses: mode 3 (shown), "
        "82.29 Hz, whirls backward (−0.20), mode 4, 149.1 Hz, forward (+0.32).",
        ANISOTROPIC,
        rpm=6000.0,
        mode=2,
        picked=("bearing", 0),
    ),
    RotorTrial(
        "A stepped compressor rotor",
        "Five sections, 60, 80 and 100 mm across, the middle one bored out to 40 mm, with two aluminium impellers "
        "350 × 60 mm (15.6 kg each) between the bearings and a steel coupling hub (7.1 kg) overhung. At 6000 rpm: "
        "167.8 and 171.0 Hz, 199.3 and 207.4, 422.5 and 442.9. On the Assembly tab, K with the bored section's "
        "share tinted: each beam's 8 × 8 adds at its two nodes' [x, θx, y, θy], so K is banded.",
        COMPRESSOR,
        rpm=6000.0,
        tab=TAB_ASSEMBLY,
        picked=("section", 2),
        grid=0.025,
    ),
    RotorTrial(
        "Oil-film bearings",
        "Plain journal bearings 50 mm across and 30 mm long, 50 µm clearance, 20 mPa·s oil, each carrying 184.5 N "
        "(half the rotor's 37.6 kg). At 3000 rpm the films' K and C are full, with cross terms (the Element tab): "
        "the rotor bounces on them at 25.3 and 25.6 Hz, damped 24 %, and bends at 50.9 and 51.1 Hz. The oil "
        "swirls at about half the speed: past 5466 rpm, where that nears the bending mode, its forward whirl's ζ "
        "goes negative and it grows by itself (at 6000 rpm, −3.0 %): oil whip.",
        FILMS,
        rpm=3000.0,
        picked=("bearing", 0),
        tab=TAB_ELEMENT,
    ),
    RotorTrial(
        "Magnetic bearings",
        "Active magnetic bearings (0.5 mm gaps, 2 A bias): each magnet's pull is a negative stiffness, k<sub>m</sub> "
        "= 0.543 MN/m on K's diagonal, which the PID beats (k<sub>i</sub>k<sub>p</sub> = 2.71 MN/m). The first pair "
        "is at 37.85 Hz with ζ = 4.57 %, the second at 143.0 with 38.4 %: the derivative damps. The controllers' "
        "12 states join q and q' in the state; four of their roots are real, at −12.7 /s, the integral action "
        "pulling each journal back to the centre.",
        MAGNETS,
        picked=("bearing", 0),
        tab=TAB_ELEMENT,
    ),
    RotorTrial(
        "A seal's swirl",
        "Stiff bearings and an annular seal a quarter along (2 MN/m, 2 kN·s/m, its fluid swirling at half the "
        "shaft's speed). At rest it damps both whirls of the first pair to 4.17 %. At 6000 rpm its cross-coupling, "
        "λΩc = 0.63 MN/m, pushes a forward whirl along and holds a backward one back: forward 0.516 %, backward "
        "7.74 %. Past 6845 rpm the forward one grows: the seal drives it.",
        SEALED,
        rpm=6000.0,
        picked=("seal", 0),
        tab=TAB_ELEMENT,
    ),
)


def first_mode(rotor: Rotor, rpm: float, forward: bool | None = None) -> tuple[float, float]:
    """(f in Hz, ζ) of the rotor's slowest mode at `rpm` (its slowest forward or backward one if asked)."""
    m = modes(assemble(rotor), rpm * RPM)
    pick_ = np.ones(m.hz.size, dtype=bool) if forward is None else (m.whirl > 0) == forward
    i = int(np.argmax(pick_))
    return float(m.hz[i]), float(m.zeta[i])


# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


def _section_inertia(rng: np.random.Generator) -> Question:
    outer = pick(rng, (40.0, 50.0, 60.0, 80.0, 100.0))
    inner = round(pick(rng, (0.0, 0.25, 0.5, 0.6)) * outer)
    s = Section(1.0, outer / 1000, inner / 1000)
    bore = f" with a {inner:g} mm bore" if inner else ", solid"
    return Question(
        f"A shaft section {outer:g} mm across{bore}. What is its diametral second moment of area I, the one in "
        "its bending stiffness EI, in cm⁴?",
        s.inertia * 1e8,
        f"I = π(D⁴ − d⁴)/64 = π({outer / 10:g}⁴ − {inner / 10:g}⁴)/64 = {number(s.inertia * 1e8, 4)} cm⁴ (D and d "
        "in cm). Its polar moment, about the spin axis, is twice that.",
        unit="cm⁴",
    )


def _disc_inertia(rng: np.random.Generator) -> Question:
    diameter = pick(rng, (200.0, 250.0, 300.0, 400.0))
    thickness = pick(rng, (20.0, 30.0, 40.0, 50.0))
    d = solid_disc(0.0, STEEL, diameter / 1000, thickness / 1000)
    return Question(
        f"A solid steel disc (7850 kg/m³) {diameter:g} mm across and {thickness:g} mm thick. What is "
        "its polar moment of inertia I<sub>p</sub>, about the spin axis?",
        d.ip,
        f"m = ρπR²t = {number(d.mass, 4)} kg, and I<sub>p</sub> = mR²/2 = {number(d.ip, 4)} kg·m². Its diametral "
        f"I<sub>d</sub> = m(3R² + t²)/12 = {number(d.id, 4)} kg·m², about half: a thin disc has I<sub>p</sub> ≈ "
        "2I<sub>d</sub>.",
        unit="kg·m²",
    )


def _spin_split(rng: np.random.Generator) -> Question:
    rpm = pick(rng, (3000.0, 6000.0, 10000.0))
    return Question(
        f"The disc a quarter of the way along the pinned shaft (Things to try: <i>A disc off centre</i>) spins "
        f"at {rpm:g} rpm. At rest each mode is a pair at one frequency. What does the spin do to the pair?",
        0,
        "The disc tilts as the shaft bends, and a tilting, spinning disc has a gyroscopic moment I<sub>p</sub>Ω "
        "× its tilt rate: it stiffens the forward whirl (its frequency rises) and softens the backward one (it "
        "falls). The higher modes, which tilt the disc more, split more.",
        choices=(
            "The forward whirl rises, the backward one falls",
            "Both rise",
            "Both fall",
            "Nothing: the spin does no work",
        ),
        setup=dict(rotor=QUARTER_DISC, rpm=rpm, picked=("disc", 0)),
        predict=True,
    )


def _hollow(rng: np.random.Generator) -> Question:
    outer = pick(rng, (40.0, 50.0, 60.0))
    ratio = pick(rng, (0.4, 0.5, 0.6, 0.7))
    inner = round(ratio * outer)
    solid = pinned_ends(Rotor(uniform(1.0, outer / 1000, 4, 3), theory=Theory.RAYLEIGH))
    bored = tuple(dataclasses.replace(s, inner=inner / 1000) for s in solid.sections)
    hollow = dataclasses.replace(solid, sections=bored)
    f0, _ = first_mode(solid, 0.0)
    f1, _ = first_mode(hollow, 0.0)
    return Question(
        f"A steel shaft 1 m long and {outer:g} mm across, pinned at its ends (Euler–Bernoulli beams), has its "
        f"first mode at {number(f0, 4)} Hz. Bore it out to {inner:g} mm (d/D = {ratio:g}). Where is its "
        "first mode now?",
        f1,
        f"f ∝ √(EI/ρA), and I/A = (D² + d²)/16: so f rises by √(1 + (d/D)²) = {number(f1 / f0, 4)}, to "
        f"{number(f1, 4)} Hz. Boring out loses mass faster than stiffness: the material near the centre bends "
        "little.",
        unit="Hz",
        setup=dict(rotor=hollow, picked=("section", 0)),
    )


def _nutation(rng: np.random.Generator) -> Question:
    rpm = pick(rng, (1500.0, 3000.0, 6000.0))
    length = pick(rng, (0.5, 1.0))
    diameter = pick(rng, (50.0, 80.0, 100.0))
    rotor = Rotor(uniform(length, diameter / 1000, 4, 3))
    m = modes(assemble(rotor), rpm * RPM)
    f = float(m.hz[0])
    ratio = 1.5 * (diameter / 1000) ** 2 / length**2
    return Question(
        f"A free steel shaft (no bearings), {length:g} m long and {diameter:g} mm across, spins at "
        f"{rpm:g} rpm. Its slowest mode is a slow forward whirl of the whole shaft, wobbling as a rigid "
        "body: its nutation. At what frequency?",
        f,
        f"A free rigid body's nutation is at I<sub>p</sub>Ω/I<sub>d</sub>. For a slender shaft I<sub>p</sub> = mD²/8 "
        f"and I<sub>d</sub> = mL²/12, so I<sub>p</sub>/I<sub>d</sub> = 1.5D²/L² = {number(ratio, 3)}: at "
        f"{rpm / 60:g} rev/s that is {number(ratio * rpm / 60, 3)} Hz (the model, which bends a little "
        f"too: {number(f, 4)} Hz).",
        unit="Hz",
        tolerance=0.03,
        setup=dict(rotor=rotor, rpm=rpm),
    )


def _spec_stiffen(rng: np.random.Generator) -> Question:
    rpm = 3000.0
    target = pick(rng, (70.0, 75.0, 80.0))
    start = QUARTER_DISC
    disc = start.discs[0]

    def goal(t: RotorTrial) -> tuple[Requirement, ...]:
        try:
            f, _ = first_mode(t.rotor, rpm, forward=None)
        except RotorError:
            f = 0.0
        d = t.rotor.discs[0] if t.rotor.discs else Disc(0.0, 0.0, 0.0, 0.0)
        return (
            at_least("the first mode at 3000 rpm", f, target, "Hz"),
            kept("the shaft's length", t.rotor.length, start.length, "m"),
            kept("the disc's mass", d.mass, disc.mass, "kg"),
            kept("the speed", t.rpm, rpm, "rpm"),
        )

    design = dataclasses.replace(start, sections=uniform(1.0, 0.06, 4, 3))
    return Question(
        f"The pinned shaft with a 300 mm disc a quarter along has its first mode at "
        f"{number(first_mode(start, rpm)[0], 4)} Hz at {rpm:g} rpm. Raise it to {target:g} Hz or more "
        "without changing the shaft's length or the disc (thicken the shaft: Select a section, then its "
        "diameter on the left).",
        0,
        "The frequency goes as √(k/m), and the shaft's k as D⁴ while its mass goes as D²: a 60 mm shaft gives "
        f"{number(first_mode(design, rpm)[0], 4)} Hz.",
        setup=dict(rotor=start, rpm=rpm, picked=("section", 0)),
        goal=goal,
        design=dict(rotor=design, rpm=rpm, picked=("section", 0)),
    )


PROBLEMS = tuple(
    Problem(key, title, "rotor_builder_page", f"See {page_link('rotor_builder_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("section-inertia", "A shaft section's I", "elements", "The shaft's elements", _section_inertia),
        ("disc-inertia", "A disc's polar inertia", "discs", "Discs", _disc_inertia),
        ("spin-split", "Predict: spin a disc", "gyroscopics", "Gyroscopic moments", _spin_split),
        ("hollow", "A hollow shaft", "elements", "The shaft's elements", _hollow),
        ("nutation", "A free shaft's nutation", "gyroscopics", "Gyroscopic moments", _nutation),
        ("stiffen-spec", "Spec: raise the first mode", "modes", "Modes at a speed", _spec_stiffen),
    )
)

TIPS = {
    "length": "<p>The section's length along the shaft. Parts past it move with its end.</p>",
    "outer": "<p>The section's outside diameter: its I grows as D⁴, its mass as D².</p>",
    "inner": "<p>The section's bore (0: solid). A hollow shaft is stiffer for its mass.</p>",
    "material": "<p>Steel (210 GPa, 7850 kg/m³), aluminium (70 GPa, 2700) or titanium (114 GPa, 4430).</p>",
    "elements": "<p>Beam elements in the section (more where a part cuts it): more find the higher modes better.</p>",
    "z": "<p>Where the part sits along the shaft, from its left end.</p>",
    "library": "<p>A disc from the library, by material and size (its mass and inertias worked out from them), or "
    "Custom, by its mass and inertias.</p>",
    "disc_mass": "<p>The disc's mass, on its x and y.</p>",
    "ip": "<p>The polar moment of inertia, about the spin axis: the gyroscopic moment I<sub>p</sub>Ω × the tilt "
    "rate.</p>",
    "id": "<p>The diametral moment of inertia, about a diameter: on the slopes θx and θy.</p>",
    "diameter": "<p>The disc's diameter (drawn; from the library, it sets the inertias).</p>",
    "thickness": "<p>The disc's thickness (drawn; from the library, it sets the inertias).</p>",
    "bearing_kind": "<p>Springs and dampers to the ground; a plain journal bearing's oil film, its stiffness and "
    "damping changing with the speed and its load; or an active magnetic bearing under PID control.</p>",
    "kxx": "<p>The bearing's stiffness along x (sideways).</p>",
    "kyy": "<p>The bearing's stiffness along y (up).</p>",
    "kxy": "<p>Cross-coupled stiffness: a push along x from a displacement in y, f<sub>x</sub> = −k<sub>xy</sub> y. "
    "k<sub>xy</sub> = −k<sub>yx</sub> feeds a forward whirl.</p>",
    "kyx": "<p>Cross-coupled stiffness: f<sub>y</sub> = −k<sub>yx</sub> x.</p>",
    "cxx": "<p>The bearing's damping along x.</p>",
    "cyy": "<p>The bearing's damping along y.</p>",
    "film_length": "<p>The oil film's axial length L.</p>",
    "film_diameter": "<p>The journal's diameter D (the shaft's, where it sits).</p>",
    "clearance": "<p>The radial clearance c between the journal and its bearing.</p>",
    "viscosity": "<p>The oil's viscosity μ.</p>",
    "gap": "<p>Each magnet's air gap with the journal centred.</p>",
    "bias": "<p>The bias current in every coil.</p>",
    "kp": "<p>The PID's proportional gain: k<sub>i</sub>k<sub>p</sub> has to beat the magnets' k<sub>m</sub>.</p>",
    "ki": "<p>The PID's integral gain (0: none).</p>",
    "kd": "<p>The PID's derivative gain: the bearing's damping.</p>",
    "amplifier": "<p>The amplifier's bandwidth: its lag turns the damping negative above it.</p>",
    "seal_k": "<p>The seal's direct stiffness, on x and y.</p>",
    "seal_c": "<p>The seal's direct damping.</p>",
    "swirl": "<p>The fluid in the seal swirls round at λ times the shaft's speed, which gives it a cross-coupled "
    "stiffness k<sub>xy</sub> = λΩc, growing with the speed.</p>",
    "seal_mass": "<p>The fluid's added mass, on x and y.</p>",
    "theory": "<p>Timoshenko's beams shear, and their sections' turning has inertia (and spins, with its "
    "gyroscopic moment); Euler–Bernoulli's here keep the turning inertia (a Rayleigh beam) but don't shear.</p>",
    "gravity": "<p>A horizontal rotor's weight loads its bearings (an oil film's stiffness depends on it); a "
    "vertical one's doesn't, and its films run centred.</p>",
    "rpm": "<p>The speed the modes are worked out at: the gyroscopic matrix, an oil film's coefficients and a "
    "seal's cross-coupling all change with it.</p>",
    "grid": "<p>The spacing parts and cuts snap to.</p>",
    "matrix": "<p>The matrix the Assembly and Element tabs show, at the speed.</p>",
    "animate": "<p>Whirl the rotor in the mode picked (slowed to the same pace whatever its frequency, and "
    "exaggerated). Space.</p>",
}
CANVAS_TIP = (
    "<p>The rotor from the side, its radii stretched to be seen: the shaft's sections, the discs, the bearings "
    "below it (springs, an oil film's housing, a magnet's stator) and the seals' teeth. Pick a tool above. While "
    "it bends in the mode picked, the dotted line is its axis at rest.</p>"
)
TABLE_TIP = (
    "<p>The modes at the speed: each one's damped frequency, its damping ratio and log decrement (negative: it "
    "grows, the rotor is unstable), and whether it whirls with the spin (forward) or against it. Click a row to "
    "show it.</p>"
)
ASSEMBLY_TIP = (
    "<p>The global matrix, node by node [x, θx, y, θy]: tinted, the picked part's share. A dot is an exact zero.</p>"
)
WHIRL_TIP = (
    "<p>The mode whirling in 3D (exaggerated): the shaft's centre line, each node's orbit and the discs. Forward "
    "whirl goes round the way the shaft spins (x toward y); backward, against it.</p>"
)

NOTES_HTML = load_notes("rotor_builder", trials=trials_html(TRIALS))


# ------------------------------------------------------------------ drawing helpers
def _segments(points: list) -> tuple[np.ndarray, np.ndarray]:
    if not points:
        return np.zeros(0), np.zeros(0)
    a = np.array(points, dtype=float).reshape(-1, 2)
    return a[:, 0], a[:, 1]


def _box(z0: float, z1: float, y0: float, y1: float) -> list:
    return [[z0, y0], [z1, y0], [z1, y1], [z0, y1], [z0, y0], NAN2]


def _zigzag(z: float, top: float, bottom: float, width: float, turns: int = 4) -> list:
    """A spring drawn down from `top` to `bottom` at z."""
    s = np.linspace(0.0, 1.0, 2 * turns + 3)
    side = np.zeros_like(s)
    side[1:-1:2], side[2:-1:2] = 1.0, -1.0
    side[-2] = 0.0
    return [[z + width * w, top + (bottom - top) * t] for w, t in zip(side, s)] + [NAN2]


def _ground(z: float, y: float, width: float) -> list:
    out = [[z - width, y], [z + width, y], NAN2]
    for k in range(5):
        x = z - width + k * width / 2
        out += [[x, y], [x - width / 3, y - width / 2.5], NAN2]
    return out


def _whirl_word(w: float) -> str:
    return "forward" if w > 0.5 else "backward" if w < -0.5 else "mixed" if abs(w) > 0.05 else "straight"


class RotorCanvas(pg.PlotWidget):
    """The rotor from the side; builds it with the tool picked. Emits `edited(rotor, selection)` for every change
    and `selected(selection)` when the selection changes."""

    edited = QtCore.Signal(object, object)
    selected = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(CANVAS_TIP)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.hideButtons()
        for side in ("left", "bottom"):
            self.getAxis(side).enableAutoSIPrefix(False)
        self.setLabel("bottom", "z", units="m")
        self.setLabel("left", "r", units="m")
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.rotor = EMPTY
        self.tool = Tool.SELECT
        self.grid = 0.05
        self.selection: Selection | None = None
        self.q: np.ndarray | None = None  # the displacement drawn (real, n_dof)
        self.template: dict[str, Disc | Bearing | Seal] = {
            "disc": DISC_LIBRARY[0].at(0.0),
            "bearing": Bearing(0.0),
            "seal": Seal(0.0),
        }
        self.height_m = 0.2  # the view's half height (m): symbols are sized from it
        self._drag: tuple[str, int] | None = None
        self._dragged: Rotor | None = None
        plot = self.getPlotItem()
        self.axis_line = pg.PlotCurveItem()
        self.ghost = pg.PlotCurveItem(connect="finite")
        self.shaft = pg.PlotCurveItem(connect="finite")
        self.bore = pg.PlotCurveItem(connect="finite")
        self.discs = pg.PlotCurveItem(connect="finite")
        self.supports = pg.PlotCurveItem(connect="finite")
        self.seals = pg.PlotCurveItem(connect="finite")
        self.highlight = pg.PlotCurveItem(connect="finite")
        self.nodes = pg.PlotDataItem(pen=None, symbol="o", symbolSize=4)
        for item in (
            self.axis_line,
            self.ghost,
            self.supports,
            self.seals,
            self.bore,
            self.shaft,
            self.discs,
            self.highlight,
            self.nodes,
        ):
            plot.addItem(item)
        self.apply_theme()

    # --------------------------------------------------------- state
    def set_rotor(self, rotor: Rotor, fit: bool = True) -> None:
        self.rotor = rotor
        if fit:
            self.fit()
        self.draw()

    def set_tool(self, tool: Tool) -> None:
        self.tool = tool
        self.draw()

    def fit(self) -> None:
        r = self.rotor
        length = max(r.length, 4 * self.grid, 1e-3)
        tall = max([s.outer / 2 for s in r.sections] + [d.diameter / 2 for d in r.discs] + [0.02])
        self.height_m = 1.6 * tall
        pad = 0.08 * length
        self.setRange(xRange=(-pad, length + pad), yRange=(-1.5 * tall, 1.3 * tall), padding=0)

    def snap(self, z: float) -> float:
        return float(np.clip(round(z / self.grid) * self.grid, 0.0, self.rotor.length))

    def _pixel(self) -> tuple[float, float]:
        sx, sy = self.getPlotItem().getViewBox().viewPixelSize()
        return sx, sy

    def part_near(self, p: np.ndarray) -> Selection | None:
        """The part (disc, bearing or seal) whose z is nearest p's, within PICK_PX."""
        sx, _ = self._pixel()
        best, reach = None, PICK_PX * sx
        for kind in ("disc", "bearing", "seal"):
            for i, part in enumerate(self.rotor.parts(kind)):
                d = abs(part.z - p[0])
                if kind == "disc":
                    d = max(d - part.thickness / 2, 0.0) if abs(p[1]) <= part.diameter / 2 else math.inf
                elif kind == "bearing" and p[1] > 0:
                    d = math.inf
                if d <= reach:
                    best, reach = (kind, i), d
        return best

    def section_near(self, p: np.ndarray) -> int | None:
        k = self.rotor.section_at(float(p[0]))
        if k is None:
            return None
        _, sy = self._pixel()
        return k if abs(p[1]) <= max(self.rotor.sections[k].outer / 2, PICK_PX * sy) else None

    def select(self, selection: Selection | None) -> None:
        self.selection = selection
        self.draw()
        self.selected.emit(selection)

    # --------------------------------------------------------- editing
    def _commit(self, rotor: Rotor, selection: Selection | None) -> None:
        self.selection = selection
        self.edited.emit(rotor, selection)

    def new_part(self, kind: str, z: float):
        """A new part of `kind` at z, made from its template (a bearing's film fitted to the shaft there)."""
        part = dataclasses.replace(self.template[kind], z=z)
        if isinstance(part, Bearing):
            part = dataclasses.replace(part, film=dataclasses.replace(part.film, diameter=self.rotor.outer_at(z)))
        return part

    def press(self, p: np.ndarray, right: bool = False) -> None:
        """A click at p (view coordinates): what the tool does."""
        if right:
            return
        r = self.rotor
        tool = self.tool
        hit = self.part_near(p)
        if tool is Tool.SELECT:
            if hit is not None:
                self._drag, self._dragged = hit, None
                self.select(hit)
            else:
                k = self.section_near(p)
                self.select(("section", k) if k is not None else None)
            return
        k = r.section_at(float(p[0]))
        if k is None:
            return
        z = self.snap(float(p[0]))
        if tool is Tool.SPLIT:
            k = r.section_at(z)
            if k is not None:
                split = r.split(k, z)
                if split is not r:
                    self._commit(split, ("section", k + 1))
            return
        kind = PART_OF_TOOL[tool]
        there = r.part_at(kind, z, reach=0.5 * self.grid)
        if there is not None:
            self.select((kind, there))
            return
        self._commit(r.with_part(kind, self.new_part(kind, z)), (kind, len(r.parts(kind))))

    def move(self, p: np.ndarray) -> None:
        if self._drag is None:
            return
        kind, i = self._drag
        z = self.snap(float(p[0]))
        part = self.rotor.parts(kind)[i]
        other = self.rotor.part_at(kind, z)
        if other is None or other == i:
            self._dragged = self.rotor.replace_part(kind, i, dataclasses.replace(part, z=z))
            self.draw(self._dragged)

    def release(self) -> None:
        if self._drag is not None and self._dragged is not None:
            self._commit(self._dragged, self._drag)
        self._drag, self._dragged = None, None

    def delete_selected(self) -> None:
        if self.selection is None:
            return
        kind, i = self.selection
        if kind == "section":
            if len(self.rotor.sections) <= 1:
                return
            rotor = self.rotor.without_section(i)
        else:
            rotor = self.rotor.without_part(kind, i)
        self._commit(rotor, None)

    def _view_point(self, event) -> np.ndarray:
        scene = self.mapToScene(event.position().toPoint())
        p = self.getPlotItem().getViewBox().mapSceneToView(scene)
        return np.array([p.x(), p.y()])

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        self.press(self._view_point(event), right=event.button() == QtCore.Qt.MouseButton.RightButton)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        self.move(self._view_point(event))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self.release()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        event.accept()

    def wheelEvent(self, event) -> None:
        event.ignore()  # the page scrolls; the view fits the rotor

    def keyPressEvent(self, event) -> None:
        if event.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            self.delete_selected()
        else:
            super().keyPressEvent(event)

    # --------------------------------------------------------- drawing
    def draw(self, rotor: Rotor | None = None) -> None:
        r = rotor if rotor is not None else self.rotor
        q = self.q if rotor is None and self.q is not None and self.q.size == r.n_dof else None
        h = self.height_m
        if q is not None:
            zc, _, yc = deflection(r, q)
            slope = q[3::4]
        else:
            zc, yc = r.nodes, np.zeros(r.nodes.size)
            slope = np.zeros(r.nodes.size)

        def centre(z: float) -> float:
            return float(np.interp(z, zc, yc)) if zc.size else 0.0

        def tilt(z: float) -> float:
            return float(np.interp(z, r.nodes, slope)) if r.nodes.size else 0.0

        ends = r.ends
        shaft: list = []
        bore: list = []
        ghost: list = []
        for s, a, b in zip(r.sections, ends[:-1], ends[1:]):
            inside = (zc >= a - 1e-12) & (zc <= b + 1e-12)
            zs, ys = zc[inside], yc[inside]
            for sign in (1.0, -1.0):
                shaft += [*np.column_stack([zs, ys + sign * s.outer / 2]), NAN2]
                if s.inner > 0:
                    bore += [*np.column_stack([zs, ys + sign * s.inner / 2]), NAN2]
            for z in (a, b):
                shaft += [[z, centre(z) - s.outer / 2], [z, centre(z) + s.outer / 2], NAN2]
        if q is not None:
            ghost = [[0.0, 0.0], [r.length, 0.0], NAN2]
        discs: list = []
        for d in r.discs:
            R, w = d.diameter / 2, d.thickness / 2
            lean = 0.04 * max(r.length, 0.1) / R  # a big disc's tilt, drawn true, would leave the view
            c, t = centre(d.z), float(np.clip(tilt(d.z), -lean, lean))
            corners = [(-w, -R), (w, -R), (w, R), (-w, R), (-w, -R)]
            discs += [[d.z + dz - t * dy, c + dy] for dz, dy in corners] + [NAN2]
        supports: list = []
        floor = -1.35 * h / 1.6
        for b in r.bearings:
            c = centre(b.z)
            rs = r.outer_at(b.z) / 2
            w = 0.03 * max(r.length, 0.1)
            if b.kind is BearingKind.SPRINGS:
                supports += _zigzag(b.z, c - rs, floor, 0.4 * w)
            elif b.kind is BearingKind.FILM:
                half = max(b.film.length / 2, 0.4 * w)
                supports += _box(b.z - half, b.z + half, -rs - 0.15 * h, rs + 0.15 * h)
                supports += [[b.z, -rs - 0.15 * h], [b.z, floor], NAN2]
            else:
                gap = 0.12 * h
                for sign in (1.0, -1.0):
                    y0 = sign * (rs + gap)
                    supports += _box(b.z - w, b.z + w, y0, y0 + sign * 0.25 * h)
                supports += [[b.z + w, -rs - gap - 0.25 * h], [b.z + w, floor], NAN2]
            supports += _ground(b.z, floor, w)
        seals: list = []
        for seal in r.seals:
            rs = r.outer_at(seal.z) / 2
            w = 0.025 * max(r.length, 0.1)
            for sign in (1.0, -1.0):
                y0, y1 = sign * (rs + 0.04 * h), sign * (rs + 0.22 * h)
                seals += [[seal.z - w, y1], [seal.z + w, y1], NAN2]
                for k in range(4):
                    z = seal.z - w + (k + 0.5) * w / 2
                    seals += [[z, y1], [z, y0], NAN2]
        self.shaft.setData(*_segments(shaft))
        self.bore.setData(*_segments(bore))
        self.ghost.setData(*_segments(ghost))
        self.discs.setData(*_segments(discs))
        self.supports.setData(*_segments(supports))
        self.seals.setData(*_segments(seals))
        if q is None:
            self.axis_line.setData([0.0, r.length], [0.0, 0.0])
        else:
            self.axis_line.setData([], [])
        node_y = yc[np.searchsorted(zc, r.nodes).clip(0, max(zc.size - 1, 0))] if zc.size else np.zeros(0)
        self.nodes.setData(r.nodes, node_y)
        self.highlight.setData(*_segments(self._lit(r, centre)))

    def _lit(self, r: Rotor, centre) -> list:
        if self.selection is None:
            return []
        kind, i = self.selection
        h = self.height_m
        if kind == "section" and i < len(r.sections):
            a, b = r.ends[i], r.ends[i + 1]
            R = r.sections[i].outer / 2
            return _box(a, b, -R, R)
        if kind == "disc" and i < len(r.discs):
            d = r.discs[i]
            c = centre(d.z)
            return _box(d.z - d.thickness / 2, d.z + d.thickness / 2, c - d.diameter / 2, c + d.diameter / 2)
        if kind in ("bearing", "seal") and i < len(r.parts(kind)):
            z = r.parts(kind)[i].z
            w = 0.04 * max(r.length, 0.1)
            top = r.outer_at(z) / 2 + 0.3 * h
            return _box(z - w, z + w, -1.35 * h / 1.6 if kind == "bearing" else -top, top)
        return []

    def apply_theme(self) -> None:
        c = colors
        self.axis_line.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashDotLine))
        self.ghost.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.shaft.setPen(pg.mkPen(c.structure, width=2))
        self.bore.setPen(pg.mkPen(c.structure, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.discs.setPen(pg.mkPen(c.mass[0], width=2.5))
        self.supports.setPen(pg.mkPen(c.structure, width=1.5))
        self.seals.setPen(pg.mkPen(c.mass[1], width=1.5))
        self.highlight.setPen(pg.mkPen(c.force, width=2.5))
        self.nodes.setSymbolBrush(pg.mkBrush(c.structure))
        self.nodes.setSymbolPen(None)
        self.draw()


class WhirlView(pg.PlotWidget):
    """The mode whirling in 3D, orthographically projected: the centre line, the orbits, the discs."""

    YAW, PITCH = math.radians(62.0), math.radians(18.0)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(WHIRL_TIP)
        self.setAspectLocked(True)
        for side in ("left", "bottom"):
            self.hideAxis(side)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.rotor = EMPTY
        self.shape: np.ndarray | None = None  # complex, n_dof, scaled to display units
        plot = self.getPlotItem()
        self.axis_line = pg.PlotCurveItem(connect="finite")
        self.orbits = pg.PlotCurveItem(connect="finite")
        self.supports = pg.PlotCurveItem(connect="finite")
        self.line = pg.PlotCurveItem()
        self.discs = pg.PlotCurveItem(connect="finite")
        self.dots = pg.PlotDataItem(pen=None, symbol="o", symbolSize=5)
        for item in (self.axis_line, self.supports, self.orbits, self.line, self.discs, self.dots):
            plot.addItem(item)
        self.apply_theme()

    def project(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Screen (x, y) of display points p (..., 3) = (lateral x, vertical y, axial)."""
        cy, sy = math.cos(self.YAW), math.sin(self.YAW)
        cp, sp = math.cos(self.PITCH), math.sin(self.PITCH)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        xr = x * cy + z * sy
        zr = -x * sy + z * cy
        return xr, y * cp - zr * sp

    def _lines(self, parts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        gap = np.full((1, 3), np.nan)
        pts = np.vstack([np.vstack([p, gap]) for p in parts]) if parts else np.empty((0, 3))
        return self.project(pts)

    def _axial(self, z: np.ndarray | float) -> np.ndarray:
        L = max(self.rotor.length, 1e-6)
        return (np.asarray(z) / L - 0.5) * AXIAL

    def _radius(self, r: float) -> float:
        tall = max([s.outer / 2 for s in self.rotor.sections] + [d.diameter / 2 for d in self.rotor.discs] + [1e-3])
        return 0.7 * r / tall

    def set_mode(self, rotor: Rotor, shape: np.ndarray | None) -> None:
        """The rotor and its mode (complex, n_dof, its largest translation 1: None for none)."""
        self.rotor = rotor
        self.shape = None if shape is None else shape * WHIRL
        z = self._axial(rotor.nodes)
        zero = np.zeros_like(z)
        self.axis_line.setData(*self._lines([np.column_stack([zero, zero, z])]))
        supports = []
        for b in rotor.bearings:
            u = float(self._axial(b.z))
            supports.append(np.array([[0.0, 0.0, u], [0.0, -1.0, u]]))
            supports.append(np.array([[-0.15, -1.0, u], [0.15, -1.0, u]]))
        self.supports.setData(*self._lines(supports))
        orbits = []
        if self.shape is not None:
            t = np.linspace(0.0, TWO_PI, 49)
            e = np.exp(1j * t)
            X, Y = self.shape[0::4], self.shape[2::4]
            for k in range(z.size):
                orbits.append(np.column_stack([(X[k] * e).real, (Y[k] * e).real, np.full(t.size, z[k])]))
        self.orbits.setData(*self._lines(orbits))
        self.setRange(xRange=(-0.6 * AXIAL, 0.6 * AXIAL), yRange=(-1.4, 1.2), padding=0)
        self.show_phase(0.0)

    def show_phase(self, phase: float) -> None:
        rotor = self.rotor
        if self.shape is not None:
            q = (self.shape * np.exp(1j * phase)).real
            zc, xc, yc = deflection(rotor, q)
        else:
            q = np.zeros(rotor.n_dof)
            zc, xc, yc = rotor.nodes, np.zeros(rotor.nodes.size), np.zeros(rotor.nodes.size)
        u = self._axial(zc)
        self.line.setData(*self.project(np.column_stack([xc, yc, u])))
        self.dots.setData(*self.project(np.column_stack([q[0::4], q[2::4], self._axial(rotor.nodes)])))
        rings = []
        t = np.linspace(0.0, TWO_PI, 41)
        for d in rotor.discs:
            j = rotor.node_at(d.z)
            R = self._radius(d.diameter / 2)
            c = np.array([q[4 * j], q[4 * j + 2], float(self._axial(rotor.nodes[j]))])
            tx, ty = q[4 * j + 1], q[4 * j + 3]  # slopes: the disc's normal is (tx, ty, 1) in metres
            scale = AXIAL / max(rotor.length, 1e-6)
            ring = np.column_stack([R * np.cos(t), R * np.sin(t), np.zeros_like(t)])
            ring[:, 2] = -(tx * ring[:, 0] + ty * ring[:, 1]) / scale  # square to the shaft's tangent
            rings.append(ring + c)
        self.discs.setData(*self._lines(rings))

    def apply_theme(self) -> None:
        c = colors
        self.axis_line.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashDotLine))
        self.orbits.setPen(pg.mkPen(c.grey, width=1))
        self.supports.setPen(pg.mkPen(c.structure, width=1.5))
        self.line.setPen(pg.mkPen(c.strong, width=3))
        self.discs.setPen(pg.mkPen(c.mass[0], width=2))
        self.dots.setSymbolBrush(pg.mkBrush(c.strong))
        self.dots.setSymbolPen(None)


# ------------------------------------------------------------------ the part's matrices as text
def _matrix(part: Part, which: Matrix, omega: float) -> np.ndarray:
    if which is Matrix.K:
        return part.stiffness(omega)
    if which is Matrix.C:
        return part.damping(omega)
    return part.M if which is Matrix.M else part.G


def _global(assembly: Assembly, which: Matrix, omega: float) -> np.ndarray:
    if which is Matrix.K:
        return assembly.stiffness(omega)
    if which is Matrix.C:
        return assembly.damping(omega)
    return assembly.M if which is Matrix.M else assembly.G


def part_html(rotor: Rotor, assembly: Assembly, selection: Selection, which: Matrix, omega: float) -> str:
    """The picked part: what it is, and its own share of the matrix, with the global DOFs it lands on."""
    kind, i = selection
    owned = [p for p in assembly.parts if p.owner == selection]
    if not owned:
        return "<p>Pick a section or a part (the Select tool) to see its matrices here.</p>"
    part = owned[0]
    lines = []
    if kind == "section":
        s = rotor.sections[i]
        lines.append(
            f"<p><b>Section {i + 1}</b>: {fmt(s.length, 'm')} of {s.material.name.lower()}, D = "
            f"{fmt(s.outer, 'm')}"
            + (f", bore d = {fmt(s.inner, 'm')}" if s.inner else ", solid")
            + f": A = {number(s.area * 1e6, 4)} mm², I = {number(s.inertia * 1e12, 4)} mm⁴, EI = "
            f"{fmt(s.ei, 'N·m²')}, {fmt(s.mass, 'kg')}. In {len(owned)} elements; the first, "
            f"{fmt(part.length, 'm')} long, is shown"
        )
        lines.append(
            f" (φ = 12EI/κGAL² = {part.phi:.3g}, κ = {s.kappa:.3f})." if rotor.theory is Theory.TIMOSHENKO else "."
        )
        lines.append(
            " Its G is twice its rotary inertia, coupling the x plane's slopes to the y plane's.</p>"
            if which is Matrix.G
            else "</p>"
        )
    elif kind == "disc":
        d = rotor.discs[i]
        lines.append(
            f"<p><b>Disc {i + 1}</b> ({d.name}) at z = {fmt(d.z, 'm')}: m = {fmt(d.mass, 'kg')}, I<sub>p</sub> = "
            f"{d.ip:.4g} kg·m², I<sub>d</sub> = {d.id:.4g} kg·m² (I<sub>p</sub>/I<sub>d</sub> = "
            f"{d.ip / d.id if d.id else math.inf:.3g}). Rigid: it adds to M on its node, and I<sub>p</sub> to G "
            "between the slopes.</p>"
        )
    elif kind == "bearing":
        b = rotor.bearings[i]
        lines.append(f"<p><b>Bearing {i + 1}</b> at z = {fmt(b.z, 'm')}: {b.kind.value.lower()}")
        if b.kind is BearingKind.FILM:
            w = assembly.loads[i]
            st = film_state(b.film, w, omega)
            lines.append(
                f", L = {fmt(b.film.length, 'm')}, D = {fmt(b.film.diameter, 'm')}, c = {fmt(b.film.clearance, 'm')}, "
                f"μ = {fmt(b.film.viscosity, 'Pa·s')}, under {fmt(w, 'N')} of the rotor's weight. At "
                f"{omega / RPM:.4g} rpm its journal sits at ε = {st.eccentricity:.3f}, {math.degrees(st.attitude):.3g}° "
                "from the load: its K and C are full, and change with the speed.</p>"
            )
        elif b.kind is BearingKind.MAGNET:
            m = b.magnet
            lines.append(
                f": k<sub>m</sub> = {fmt(m.negative_stiffness, 'N/m')} (on K's diagonal, negative), k<sub>i</sub> = "
                f"{fmt(m.current_gain, 'N/A')}; its PID's {m.n_states} states a side follow q and q' in the state: "
                f"k<sub>i</sub>k<sub>p</sub> = {fmt(m.current_gain * m.kp, 'N/m')} holds it.</p>"
            )
        else:
            lines.append(".</p>")
    else:
        seal = rotor.seals[i]
        lines.append(
            f"<p><b>Seal {i + 1}</b> at z = {fmt(seal.z, 'm')}: k = {fmt(seal.k, 'N/m')}, c = {fmt(seal.c, 'N·s/m')}, "
            "swirl "
            f"λ = {seal.swirl:.3g}: at {omega / RPM:.4g} rpm, k<sub>xy</sub> = λΩc = {fmt(seal.swirl * omega * seal.c, 'N/m')} "
            "(and k<sub>yx</sub> = −k<sub>xy</sub>), in K(Ω).</p>"
        )
    own = _matrix(part, which, omega)
    labels = [dof_label(d) for d in part.dofs]
    sym = which.name
    lines.append(f"<p>Its share of {sym}, added at {', '.join(labels)} (tinted on the Assembly tab):</p>")
    lines.append(matrix_html(own, labels, labels, title=sym))
    return "".join(lines)


# ------------------------------------------------------------------ the properties
class Properties(QtWidgets.QGroupBox):
    """The selected item's settings: a section's, a disc's, a bearing's or a seal's (or nothing)."""

    def __init__(self) -> None:
        super().__init__("Selected")
        v = QtWidgets.QVBoxLayout(self)
        self.empty = note_label("Nothing picked: pick the Select tool and click a section or a part.")
        v.addWidget(self.empty)

        self.section = Form("")
        self.section.setFlat(True)
        s = self.section
        self.length = s.number("Length:", 1.0, 10000.0, 250.0, 1, " mm", TIPS["length"])
        self.outer = s.number("Diameter D:", 1.0, 2000.0, 50.0, 1, " mm", TIPS["outer"])
        self.inner = s.number("Bore d:", 0.0, 1999.0, 0.0, 1, " mm", TIPS["inner"])
        self.material = combo([m.name for m in MATERIALS], TIPS["material"])
        s.add("Material:", self.material, TIPS["material"])
        self.elements = QtWidgets.QSpinBox()
        self.elements.setRange(1, MAX_ELEMENTS)
        self.elements.setKeyboardTracking(False)
        s.add("Elements:", self.elements, TIPS["elements"])
        self.section_info = note_label()
        s.row(self.section_info)
        self.add_section = QtWidgets.QPushButton("Add a section after the shaft's end")
        self.add_section.setToolTip("<p>A copy of the last section, added at the right-hand end.</p>")
        s.row(self.add_section)
        v.addWidget(self.section)

        self.disc = Form("")
        self.disc.setFlat(True)
        d = self.disc
        self.disc_z = d.number("At z:", 0.0, 100.0, 0.0, 3, " m", TIPS["z"])
        self.library = combo(["Custom", *[t.name for t in DISC_LIBRARY]], TIPS["library"])
        self.library.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        d.add("Library:", self.library, TIPS["library"])
        self.disc_mass = d.number("Mass m:", 0.0, 1e5, 10.0, 3, " kg", TIPS["disc_mass"])
        self.ip = d.number("Polar Ip:", 0.0, 1e5, 0.05, 5, " kg·m²", TIPS["ip"])
        self.id = d.number("Diametral Id:", 0.0, 1e5, 0.025, 5, " kg·m²", TIPS["id"])
        self.diameter = d.number("Diameter:", 1.0, 5000.0, 200.0, 1, " mm", TIPS["diameter"])
        self.thickness = d.number("Thickness:", 1.0, 2000.0, 30.0, 1, " mm", TIPS["thickness"])
        v.addWidget(self.disc)

        self.bearing = Form("")
        self.bearing.setFlat(True)
        b = self.bearing
        self.bearing_z = b.number("At z:", 0.0, 100.0, 0.0, 3, " m", TIPS["z"])
        self.bearing_kind = combo([k.value for k in BEARING_KINDS], TIPS["bearing_kind"])
        b.add("Kind:", self.bearing_kind, TIPS["bearing_kind"])
        self.kxx = b.number("kxx:", -1e6, 1e6, 50.0, 3, " MN/m", TIPS["kxx"])
        self.kyy = b.number("kyy:", -1e6, 1e6, 50.0, 3, " MN/m", TIPS["kyy"])
        self.kxy = b.number("kxy:", -1e6, 1e6, 0.0, 3, " MN/m", TIPS["kxy"])
        self.kyx = b.number("kyx:", -1e6, 1e6, 0.0, 3, " MN/m", TIPS["kyx"])
        self.cxx = b.number("cxx:", 0.0, 1e6, 2.0, 3, " kN·s/m", TIPS["cxx"])
        self.cyy = b.number("cyy:", 0.0, 1e6, 2.0, 3, " kN·s/m", TIPS["cyy"])
        self.film_length = b.number("Film length L:", 1.0, 1000.0, 25.0, 1, " mm", TIPS["film_length"])
        self.film_diameter = b.number("Journal D:", 1.0, 2000.0, 50.0, 1, " mm", TIPS["film_diameter"])
        self.clearance = b.number("Clearance c:", 1.0, 5000.0, 50.0, 1, " µm", TIPS["clearance"])
        self.viscosity = b.number("Viscosity μ:", 0.1, 10000.0, 30.0, 1, " mPa·s", TIPS["viscosity"])
        self.gap = b.number("Air gap:", 0.05, 20.0, 1.0, 2, " mm", TIPS["gap"])
        self.bias = b.number("Bias current:", 0.01, 100.0, 1.0, 2, " A", TIPS["bias"])
        self.kp = b.number("kp:", 0.0, 1e7, 3500.0, 0, " A/m", TIPS["kp"])
        self.ki = b.number("ki:", 0.0, 1e9, 35000.0, 0, " A/(m·s)", TIPS["ki"])
        self.kd = b.number("kd:", 0.0, 1e5, 5.0, 2, " A·s/m", TIPS["kd"])
        self.amplifier = b.number("Amplifier:", 1.0, 1e5, 1000.0, 0, " Hz", TIPS["amplifier"])
        self.bearing_info = note_label()
        b.row(self.bearing_info)
        v.addWidget(self.bearing)
        self.spring_rows = (self.kxx, self.kyy, self.kxy, self.kyx, self.cxx, self.cyy)
        self.film_rows = (self.film_length, self.film_diameter, self.clearance, self.viscosity)
        self.magnet_rows = (self.gap, self.bias, self.kp, self.ki, self.kd, self.amplifier)

        self.seal = Form("")
        self.seal.setFlat(True)
        e = self.seal
        self.seal_z = e.number("At z:", 0.0, 100.0, 0.0, 3, " m", TIPS["z"])
        self.seal_k = e.number("Stiffness k:", 0.0, 1e6, 2.0, 3, " MN/m", TIPS["seal_k"])
        self.seal_c = e.number("Damping c:", 0.0, 1e6, 2.0, 3, " kN·s/m", TIPS["seal_c"])
        self.swirl = e.number("Swirl λ:", 0.0, 1.0, 0.5, 2, "", TIPS["swirl"])
        self.seal_mass = e.number("Added mass:", 0.0, 1e4, 0.0, 3, " kg", TIPS["seal_mass"])
        v.addWidget(self.seal)

        self.remove = QtWidgets.QPushButton("Delete")
        self.remove.setToolTip("<p>Remove the picked section or part. Del does it too.</p>")
        v.addWidget(self.remove)
        self.forms = (self.empty, self.section, self.disc, self.bearing, self.seal)
        self.show_form(self.empty)

    def show_form(self, form: QtWidgets.QWidget) -> None:
        for w in self.forms:
            w.setVisible(w is form)

    def current(self) -> QtWidgets.QWidget:
        return next(w for w in self.forms if not w.isHidden())

    def show_kind(self, kind: BearingKind) -> None:
        for rows, on in (
            (self.spring_rows, kind is BearingKind.SPRINGS),
            (self.film_rows, kind is BearingKind.FILM),
            (self.magnet_rows, kind is BearingKind.MAGNET),
        ):
            for box in rows:
                self.bearing.form.setRowVisible(box, on)


def _hz(f: float) -> str:
    """Four significant figures, trailing zeros kept (171.0, not 171)."""
    return f"{f:#.4g}".rstrip(".")


def _count(n: int, thing: str) -> str:
    return f"{n} {thing}" + ("" if n == 1 else "s")


def _percent(zeta: float) -> str:
    return f"{100 * zeta:.3g} %"


# ------------------------------------------------------------------ the page
class RotorBuilderPage(QtWidgets.QWidget):
    """Controls on the left, the side view and its modes in the centre, the 3D whirl and the matrices on the
    right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.rotor = EMPTY
        self.assembly: Assembly | None = None
        self.result: WhirlModes | None = None
        self.error = ""
        self.mode = 0
        self.picked: Selection | None = None  # the part on the Element tab
        self.running = True  # a mode whirls on its own
        self._phase = 0.0
        self._history: list[Rotor] = []
        self._future: list[Rotor] = []

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Each loads a rotor, to show one idea from the Theory tab. It's then yours to change.</p>",
        )
        self.trials.picked.connect(self.load_trial)

        self.props = Properties()
        p = self.props
        for box in (p.length, p.outer, p.inner, p.elements):
            box.editingFinished.connect(self._on_section_box)
        p.material.currentIndexChanged.connect(self._on_section_box)
        p.add_section.clicked.connect(self._on_add_section)
        for box in (p.disc_z, p.disc_mass, p.ip, p.id, p.diameter, p.thickness):
            box.editingFinished.connect(self._on_disc_box)
        p.library.currentIndexChanged.connect(self._on_library)
        for box in (p.bearing_z, *p.spring_rows, *p.film_rows, *p.magnet_rows):
            box.editingFinished.connect(self._on_bearing_box)
        p.bearing_kind.currentIndexChanged.connect(self._on_bearing_box)
        for box in (p.seal_z, p.seal_k, p.seal_c, p.swirl, p.seal_mass):
            box.editingFinished.connect(self._on_seal_box)
        p.remove.clicked.connect(weak.slot(self, lambda self: self.canvas.delete_selected()))

        model = Form("Model")
        self.theory = combo([t.value for t in THEORIES], TIPS["theory"])
        model.add("Beams:", self.theory, TIPS["theory"])
        self.gravity = combo(["Horizontal", "Vertical"], TIPS["gravity"])
        model.add("Rotor:", self.gravity, TIPS["gravity"])
        self.rpm = model.number("Speed:", 0.0, 100000.0, 0.0, 0, " rpm", TIPS["rpm"])
        self.grid = combo([f"{1000 * g:g} mm" for g in GRIDS], TIPS["grid"])
        model.add("Grid:", self.grid, TIPS["grid"])
        self.theory.currentIndexChanged.connect(self._on_model)
        self.gravity.currentIndexChanged.connect(self._on_model)
        self.rpm.valueChanged.connect(self._on_speed)
        self.grid.currentIndexChanged.connect(self._on_grid)

        show = Form("Show")
        self.matrix = combo([m.value for m in MATRICES], TIPS["matrix"])
        show.add("Matrix:", self.matrix, TIPS["matrix"])
        self.run_button = RunButton()
        self.run_button.set_running(True)
        self.run_button.setToolTip(TIPS["animate"])
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setToolTip("<p>Back to the start of the whirl.</p>")
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.reset_button)
        show.row(buttons)
        self.matrix.currentIndexChanged.connect(self._draw_tabs)
        self.run_button.toggled.connect(self._on_animate)
        self.reset_button.clicked.connect(self.reset)

        # --- centre: the tools, the side view, the modes
        self.canvas: RotorCanvas = RotorCanvas()
        self.canvas.edited.connect(self._on_edit)
        self.canvas.selected.connect(self._on_selected)
        tools = QtWidgets.QWidget()
        flow = FlowLayout(tools)
        flow.setContentsMargins(0, 0, 0, 0)
        self.tool_buttons = QtWidgets.QButtonGroup(self)
        for k, tool in enumerate(TOOLS):
            b = QtWidgets.QToolButton()
            b.setText(tool.value)
            b.setCheckable(True)
            b.setToolTip(f"<p>{TOOL_HINTS[tool]}</p>")
            self.tool_buttons.addButton(b, k)
            flow.add(b)
        self.tool_buttons.button(0).setChecked(True)
        self.tool_buttons.idClicked.connect(self._on_tool)
        self.disc_kind = combo(["Custom", *[t.name for t in DISC_LIBRARY]], TIPS["library"])
        self.disc_kind.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.disc_kind.setCurrentIndex(1)
        self.disc_kind.currentIndexChanged.connect(self._on_disc_kind)
        self.bearing_tool_kind = combo([k.value for k in BEARING_KINDS], TIPS["bearing_kind"])
        self.bearing_tool_kind.currentIndexChanged.connect(self._on_bearing_tool_kind)
        self.undo_button = QtWidgets.QToolButton()
        self.undo_button.setText("Undo")
        self.undo_button.setToolTip("<p>Undo the last change (Ctrl+Z).</p>")
        self.redo_button = QtWidgets.QToolButton()
        self.redo_button.setText("Redo")
        self.redo_button.setToolTip("<p>Redo it (Ctrl+Shift+Z).</p>")
        self.clear_button = QtWidgets.QToolButton()
        self.clear_button.setText("Clear")
        self.clear_button.setToolTip("<p>Start again from a bare shaft (Undo brings it back).</p>")
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        self.clear_button.clicked.connect(
            weak.slot(self, lambda self: self._set_rotor(Rotor(theory=self.rotor.theory, gravity=self.rotor.gravity)))
        )
        flow.add(group(self.disc_kind), group(self.bearing_tool_kind))
        flow.add(self.undo_button, self.redo_button, self.clear_button)
        for keys, slot in (("Ctrl+Z", self.undo), ("Ctrl+Shift+Z", self.redo), ("Ctrl+Y", self.redo)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(keys), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
        self.hint = note_label(TOOL_HINTS[Tool.SELECT])

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setToolTip(TABLE_TIP)
        self.table.setHorizontalHeaderLabels(["Mode", "f (Hz)", "ζ", "Log dec δ", "Whirl"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._on_row)
        self.readout = readout_label()

        # --- right
        self.whirl = WhirlView()
        self.whirl_note = note_label()
        self.assembly_view = QtWidgets.QTextBrowser()
        self.assembly_view.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        self.assembly_view.setToolTip(ASSEMBLY_TIP)
        self.assembly_note = note_label()
        self.element_view = QtWidgets.QTextBrowser()
        self.notes = theory(NOTES_HTML)

        self.tabs = QtWidgets.QTabWidget()
        whirl_tab = QtWidgets.QWidget()
        wl = QtWidgets.QVBoxLayout(whirl_tab)
        wl.addWidget(self.whirl_note)
        wl.addWidget(self.whirl, 1)
        self.tabs.addTab(whirl_tab, "Whirl")
        self.tabs.addTab(column(self.assembly_note, self.assembly_view, stretch=False), "Assembly")
        self.tabs.addTab(self.element_view, "Element")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._draw_tabs)

        centre = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(centre)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(tools)
        v.addWidget(self.hint)
        v.addWidget(self.canvas, 4)
        v.addWidget(self.table, 0)
        v.addWidget(self.readout, 0)
        three_panes(self, column(self.trials, self.props, model, show), centre, self.tabs)

        self._clock = QtCore.QElapsedTimer()
        self._last = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._frame)
        self._on_tool(0)
        self._on_disc_kind(self.disc_kind.currentIndex())
        self.apply_trial(TRIALS[0], keep_history=False)
        self.apply_theme()

    # ------------------------------------------------------------- trials and practice
    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(RotorTrial("", "", **setup))

    def practice_state(self) -> RotorTrial:
        """The page's settings, as a trial: what a practice spec checks."""
        return RotorTrial(
            "",
            "",
            rotor=self.rotor,
            rpm=self.rpm.value(),
            mode=self.mode,
            tab=self.tabs.currentIndex(),
            picked=self.picked,
            matrix=MATRICES[self.matrix.currentIndex()],
            grid=self.canvas.grid,
        )

    def apply_trial(self, t: RotorTrial, keep_history: bool = True) -> None:
        set_quietly(self.matrix, MATRICES.index(t.matrix))
        set_quietly(self.rpm, t.rpm)
        self.canvas.grid = t.grid
        set_quietly(self.grid, int(np.argmin([abs(g - t.grid) for g in GRIDS])))
        self.mode = t.mode
        self.picked = t.picked
        self.trials.show_note(t.note)
        self._set_rotor(t.rotor, record=keep_history, selection=t.picked)
        self.tabs.setCurrentIndex(t.tab)
        self._phase = 0.0
        self.running = True  # a trial starts it
        self.run_button.set_running(True)
        self._show_mode()
        self._start_timer()

    # ------------------------------------------------------------- editing
    @property
    def omega(self) -> float:
        return self.rpm.value() * RPM

    def _set_rotor(
        self, rotor: Rotor, record: bool = True, selection: Selection | None = None, fit: bool = True
    ) -> None:
        if record and rotor != self.rotor:
            self._history.append(self.rotor)
            self._future.clear()
        self.rotor = rotor
        set_quietly(self.theory, THEORIES.index(rotor.theory))
        set_quietly(self.gravity, 0 if rotor.gravity else 1)
        try:
            self.assembly = assemble(rotor)
            self.error = ""
        except RotorError as error:
            self.assembly, self.error = None, str(error)
        if selection is not None:
            kind, i = selection
            if i >= (len(rotor.sections) if kind == "section" else len(rotor.parts(kind))):
                selection = None
        self.canvas.selection = selection
        self.picked = selection if selection is not None else self.picked
        if self.picked is not None:
            kind, i = self.picked
            if i >= (len(rotor.sections) if kind == "section" else len(rotor.parts(kind))):
                self.picked = None
        self.canvas.set_rotor(rotor, fit=fit)
        self._fill_props()
        self._work()
        self.undo_button.setEnabled(bool(self._history))
        self.redo_button.setEnabled(bool(self._future))

    def _work(self) -> None:
        """The modes at the speed, and everything drawn from them."""
        self.result = None
        if self.assembly is not None:
            try:
                self.result = modes(self.assembly, self.omega)
            except (RotorError, np.linalg.LinAlgError) as error:
                self.error = str(error)
        count = self.result.hz.size if self.result is not None else 0
        self.mode = min(self.mode, max(count - 1, 0))
        self._fill_table()
        self._fill_readout()
        self._draw_tabs()
        self._show_mode()
        self._start_timer()

    def undo(self) -> None:
        if self._history:
            self._future.append(self.rotor)
            self._set_rotor(self._history.pop(), record=False)

    def redo(self) -> None:
        if self._future:
            self._history.append(self.rotor)
            self._set_rotor(self._future.pop(), record=False)

    def _on_edit(self, rotor: Rotor, selection) -> None:
        self._set_rotor(rotor, selection=selection, fit=rotor.length != self.rotor.length)

    def _on_selected(self, selection) -> None:
        if selection is not None:
            self.picked = selection
        self._fill_props()
        self._draw_tabs()

    def _on_tool(self, index: int) -> None:
        tool = TOOLS[index]
        self.canvas.set_tool(tool)
        self.disc_kind.setVisible(tool is Tool.DISC)
        self.bearing_tool_kind.setVisible(tool is Tool.BEARING)
        self.hint.setText(TOOL_HINTS[tool])

    def _on_disc_kind(self, index: int) -> None:
        if index > 0:
            self.canvas.template["disc"] = DISC_LIBRARY[index - 1].at(0.0)

    def _on_bearing_tool_kind(self, index: int) -> None:
        b = self.canvas.template["bearing"]
        assert isinstance(b, Bearing), "the bearing tool's template"
        self.canvas.template["bearing"] = dataclasses.replace(b, kind=BEARING_KINDS[index])

    def _on_grid(self, index: int) -> None:
        self.canvas.grid = GRIDS[index]
        self.canvas.draw()

    def _on_model(self) -> None:
        rotor = dataclasses.replace(
            self.rotor,
            theory=THEORIES[self.theory.currentIndex()],
            gravity=GRAVITY if self.gravity.currentIndex() == 0 else 0.0,
        )
        self._set_rotor(rotor, selection=self.canvas.selection, fit=False)

    def _on_speed(self) -> None:
        self._work()

    def _selected(self, kind: str) -> int | None:
        sel = self.canvas.selection
        return sel[1] if sel is not None and sel[0] == kind else None

    def _on_section_box(self) -> None:
        k = self._selected("section")
        if k is None:
            return
        p = self.props
        outer = p.outer.value() / 1000
        s = Section(
            p.length.value() / 1000,
            outer,
            min(p.inner.value() / 1000, 0.95 * outer),
            MATERIALS[p.material.currentIndex()],
            p.elements.value(),
        )
        if s != self.rotor.sections[k]:
            new = self.rotor.replace_section(k, s)
            self._set_rotor(new, selection=("section", k), fit=new.length != self.rotor.length)

    def _on_add_section(self) -> None:
        self._set_rotor(self.rotor.with_section(), selection=("section", len(self.rotor.sections)))

    def _clamp(self, z: float) -> float:
        return float(np.clip(z, 0.0, self.rotor.length))

    def _on_disc_box(self) -> None:
        i = self._selected("disc")
        if i is None:
            return
        p = self.props
        d = Disc(
            self._clamp(p.disc_z.value()),
            p.disc_mass.value(),
            p.ip.value(),
            p.id.value(),
            p.diameter.value() / 1000,
            p.thickness.value() / 1000,
        )
        old = self.rotor.discs[i]
        if (d.mass, d.ip, d.id, d.diameter, d.thickness) == (old.mass, old.ip, old.id, old.diameter, old.thickness):
            d = dataclasses.replace(d, name=old.name)
        if d != old:
            self.canvas.template["disc"] = d
            self._set_rotor(self.rotor.replace_part("disc", i, d), selection=("disc", i), fit=False)

    def _on_library(self, index: int) -> None:
        i = self._selected("disc")
        if i is None or index == 0:
            return
        old = self.rotor.discs[i]
        d = DISC_LIBRARY[index - 1].at(old.z)
        self.canvas.template["disc"] = d
        self._set_rotor(self.rotor.replace_part("disc", i, d), selection=("disc", i), fit=False)

    def _on_bearing_box(self) -> None:
        i = self._selected("bearing")
        if i is None:
            return
        p = self.props
        old = self.rotor.bearings[i]
        kind = BEARING_KINDS[p.bearing_kind.currentIndex()]
        z = self._clamp(p.bearing_z.value())
        film = JournalBearing(
            p.film_length.value() / 1000,
            p.film_diameter.value() / 1000,
            p.clearance.value() * 1e-6,
            p.viscosity.value() / 1000,
            0.0,
        )
        if kind is BearingKind.FILM and old.kind is not BearingKind.FILM:
            film = dataclasses.replace(film, diameter=self.rotor.outer_at(z))
        try:
            magnet = dataclasses.replace(
                old.magnet,
                gap=p.gap.value() / 1000,
                bias=p.bias.value(),
                kp=p.kp.value(),
                ki=p.ki.value(),
                kd=p.kd.value(),
                amplifier_hz=p.amplifier.value(),
            )
        except ValueError:
            magnet = old.magnet
        b = Bearing(
            z,
            kind,
            p.kxx.value() * 1e6,
            p.kyy.value() * 1e6,
            p.kxy.value() * 1e6,
            p.kyx.value() * 1e6,
            p.cxx.value() * 1e3,
            p.cyy.value() * 1e3,
            film,
            magnet,
        )
        if b != old:
            other = self.rotor.part_at("bearing", z)
            if other is not None and other != i:  # two bearings can't share a place
                b = dataclasses.replace(b, z=old.z)
            self.canvas.template["bearing"] = b
            self._set_rotor(self.rotor.replace_part("bearing", i, b), selection=("bearing", i), fit=False)

    def _on_seal_box(self) -> None:
        i = self._selected("seal")
        if i is None:
            return
        p = self.props
        s = Seal(
            self._clamp(p.seal_z.value()),
            p.seal_k.value() * 1e6,
            p.seal_c.value() * 1e3,
            p.swirl.value(),
            p.seal_mass.value(),
        )
        if s != self.rotor.seals[i]:
            self.canvas.template["seal"] = s
            self._set_rotor(self.rotor.replace_part("seal", i, s), selection=("seal", i), fit=False)

    # ------------------------------------------------------------- filling in
    def _fill_props(self) -> None:
        p = self.props
        sel = self.canvas.selection
        if sel is None:
            p.show_form(p.empty)
            p.remove.setEnabled(False)
            return
        kind, i = sel
        p.remove.setEnabled(kind != "section" or len(self.rotor.sections) > 1)
        top = self.rotor.length
        for at in (p.disc_z, p.bearing_z, p.seal_z):
            at.setMaximum(top)
        if kind == "section":
            s = self.rotor.sections[i]
            for box, value in (
                (p.length, s.length * 1000),
                (p.outer, s.outer * 1000),
                (p.inner, s.inner * 1000),
                (p.material, [m.name for m in MATERIALS].index(s.material.name)),
                (p.elements, s.elements),
            ):
                set_quietly(box, value)
            a = self.rotor.ends[i]
            p.section_info.setText(
                f"From z = {a:.4g} to {a + s.length:.4g} m: A = {number(s.area * 1e6, 4)} mm², I = "
                f"{number(s.inertia * 1e12, 4)} mm⁴, {s.mass:.4g} kg, L/D = {s.length / s.outer:.3g}."
            )
            p.section.setTitle(f"Section {i + 1}")
            p.show_form(p.section)
        elif kind == "disc":
            d = self.rotor.discs[i]
            names = [t.name for t in DISC_LIBRARY]
            for box, value in (
                (p.disc_z, d.z),
                (p.library, names.index(d.name) + 1 if d.name in names else 0),
                (p.disc_mass, d.mass),
                (p.ip, d.ip),
                (p.id, d.id),
                (p.diameter, d.diameter * 1000),
                (p.thickness, d.thickness * 1000),
            ):
                set_quietly(box, value)
            p.disc.setTitle(f"Disc {i + 1}")
            p.show_form(p.disc)
        elif kind == "bearing":
            b = self.rotor.bearings[i]
            for box, value in (
                (p.bearing_z, b.z),
                (p.bearing_kind, BEARING_KINDS.index(b.kind)),
                (p.kxx, b.kxx / 1e6),
                (p.kyy, b.kyy / 1e6),
                (p.kxy, b.kxy / 1e6),
                (p.kyx, b.kyx / 1e6),
                (p.cxx, b.cxx / 1e3),
                (p.cyy, b.cyy / 1e3),
                (p.film_length, b.film.length * 1000),
                (p.film_diameter, b.film.diameter * 1000),
                (p.clearance, b.film.clearance * 1e6),
                (p.viscosity, b.film.viscosity * 1000),
                (p.gap, b.magnet.gap * 1000),
                (p.bias, b.magnet.bias),
                (p.kp, b.magnet.kp),
                (p.ki, b.magnet.ki),
                (p.kd, b.magnet.kd),
                (p.amplifier, b.magnet.amplifier_hz),
            ):
                set_quietly(box, value)
            p.show_kind(b.kind)
            p.bearing_info.setText(self._bearing_info(i))
            p.bearing.setTitle(f"Bearing {i + 1}")
            p.show_form(p.bearing)
        else:
            seal = self.rotor.seals[i]
            for box, value in (
                (p.seal_z, seal.z),
                (p.seal_k, seal.k / 1e6),
                (p.seal_c, seal.c / 1e3),
                (p.swirl, seal.swirl),
                (p.seal_mass, seal.mass),
            ):
                set_quietly(box, value)
            p.seal.setTitle(f"Seal {i + 1}")
            p.show_form(p.seal)

    def _bearing_info(self, i: int) -> str:
        b = self.rotor.bearings[i]
        if b.kind is BearingKind.FILM:
            if self.assembly is None:
                return ""
            w = self.assembly.loads[i]
            st = film_state(b.film, w, self.omega)
            return f"Load {w:.4g} N; at the speed, ε = {st.eccentricity:.3f}."
        if b.kind is BearingKind.MAGNET:
            m = b.magnet
            margin = m.current_gain * m.kp - m.negative_stiffness
            return (
                f"k<sub>m</sub> = {fmt(m.negative_stiffness, 'N/m')}, k<sub>i</sub>k<sub>p</sub> − k<sub>m</sub> = "
                f"{fmt(margin, 'N/m')}" + (" (too weak: it can't hold the journal)." if margin <= 0 else ".")
            )
        return ""

    def _fill_table(self) -> None:
        self.table.blockSignals(True)
        r = self.result
        count = min(r.hz.size, MODE_ROWS) if r is not None else 0
        self.table.setRowCount(count)
        for i in range(count):
            assert r is not None
            texts = [str(i + 1), _hz(r.hz[i]), _percent(r.zeta[i]), f"{r.log_dec[i]:.3g}", _whirl_word(r.whirl[i])]
            for col, text in enumerate(texts):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                if col in (2, 3) and r.zeta[i] < 0:
                    item.setForeground(QtGui.QColor(colors.force))
                self.table.setItem(i, col, item)
        if count:
            self.table.selectRow(min(self.mode, count - 1))
        self.table.blockSignals(False)
        rows = sum(self.table.rowHeight(i) for i in range(min(count, 5))) if count else 0
        self.table.setFixedHeight(self.table.horizontalHeader().height() + rows + 2 * self.table.frameWidth() + 2)

    def _fill_readout(self) -> None:
        r = self.rotor
        text = (
            f"{_count(len(r.sections), 'section')}, {_count(len(r.discs), 'disc')}, "
            f"{_count(len(r.bearings), 'bearing')}, {_count(len(r.seals), 'seal')}: {r.nodes.size} nodes, "
            f"{r.n_dof} DOFs, {r.mass:.4g} kg"
        )
        a = self.assembly
        if a is not None and a.n_control:
            text += f" and {a.n_control} controller states"
        m = self.result
        if m is not None:
            text += f". At {self.rpm.value():.5g} rpm: {m.hz.size} whirl modes"
            if m.rigid:
                text += f", <b>{m.rigid} roots at 0</b> (it isn't held: a rigid body)"
            if m.roots.size:
                text += f", {m.roots.size} real roots"
            unstable = int((m.zeta < 0).sum())
            if unstable:
                text += f". <span style='color:{colors.force}'><b>{unstable} growing: unstable.</b></span>"
            else:
                text += "."
        elif self.error:
            text += f". <span style='color:{colors.force}'>{self.error}</span>"
        self.readout.setText(text)

    def _draw_tabs(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_WHIRL:
            self._draw_whirl()
        elif i == TAB_ASSEMBLY:
            self._draw_assembly()
        elif i == TAB_ELEMENT:
            self._draw_element()

    def _draw_whirl(self) -> None:
        m = self.result
        if m is None or self.mode >= m.hz.size:
            self.whirl.set_mode(self.rotor, None)
            self.whirl_note.setText(self.error or "No whirl modes.")
            return
        w = float(m.whirl[self.mode])
        self.whirl.set_mode(self.rotor, m.shapes[:, self.mode])
        self.whirl.show_phase(self._phase)
        stable = "decays" if m.zeta[self.mode] >= 0 else "<b>grows</b>"
        self.whirl_note.setText(
            f"Mode {self.mode + 1} at {self.rpm.value():.5g} rpm: {m.hz[self.mode]:.4g} Hz, ζ = "
            f"{_percent(m.zeta[self.mode])} (it {stable}), whirling {_whirl_word(w)} (whirl ratio {w:+.2f}: +1 "
            "a circle with the spin, −1 against it, 0 a straight line)."
        )

    def _draw_assembly(self) -> None:
        a = self.assembly
        if a is None:
            self.assembly_view.setHtml(f"<p>{self.error}</p>")
            self.assembly_note.setText("")
            return
        which = MATRICES[self.matrix.currentIndex()]
        A = _global(a, which, self.omega)
        n = A.shape[0]
        tint = np.zeros((n, n), dtype=bool)
        nodes: set[int] = set()
        if self.picked is not None:
            for part in a.parts:
                if part.owner == self.picked:
                    own = _matrix(part, which, self.omega)
                    dofs = list(part.dofs)
                    tint[np.ix_(dofs, dofs)] |= own != 0
                    nodes |= {d // 4 for d in dofs}
        keep = np.arange(n)
        if n > MATRIX_LIMIT:
            around = sorted(nodes) if nodes else [0]
            lo, hi = max(min(around) - 1, 0), min(max(around) + 1, n // 4 - 1)
            hi = max(hi, min(lo + MATRIX_LIMIT // 4 - 1, n // 4 - 1)) if hi - lo < 2 else hi
            keep = np.arange(4 * lo, 4 * hi + 4)
        labels = [dof_label(int(d)) for d in keep]
        idx = np.ix_(keep, keep)
        self.assembly_view.setHtml(matrix_html(A[idx], labels, labels, tint[idx], title=which.name))
        picked = " Pick a section or a part to tint its share."
        if self.picked is not None:
            picked = f" Tinted: {self.picked[0]} {self.picked[1] + 1}'s share."
        unit = {
            Matrix.K: "N/m and N/rad",
            Matrix.M: "kg and kg·m²",
            Matrix.C: "N·s/m",
            Matrix.G: "kg·m² (Ω G in the equations)",
        }[which]
        span = f"nodes {keep[0] // 4 + 1}–{keep[-1] // 4 + 1} of {n // 4}" if keep.size < n else f"all {n} DOFs"
        speed = f" at {self.rpm.value():.5g} rpm" if which in (Matrix.K, Matrix.C) else ""
        self.assembly_note.setText(f"{which.value}{speed} over {span} ({unit}).{picked}")

    def _draw_element(self) -> None:
        if self.assembly is None:
            self.element_view.setHtml(f"<p>{self.error}</p>")
            return
        if self.picked is None:
            self.element_view.setHtml("<p>Pick a section or a part (the Select tool) to see its matrices here.</p>")
            return
        which = MATRICES[self.matrix.currentIndex()]
        self.element_view.setHtml(part_html(self.rotor, self.assembly, self.picked, which, self.omega))

    # ------------------------------------------------------------- the modes
    def _on_row(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.mode = rows[0].row()
            self._show_mode()
            if self.tabs.currentIndex() == TAB_WHIRL:
                self._draw_whirl()

    def _on_animate(self, on: bool) -> None:
        self.running = on
        self._start_timer()

    def reset(self) -> None:
        """The whirl back at its start, still whirling."""
        self._phase = 0.0
        self.running = True
        self.run_button.set_running(True)
        self._show_mode()
        self._start_timer()

    def mode_shape(self) -> np.ndarray | None:
        """The mode picked (complex, n_dof), scaled so its largest translation is SWING of the side view."""
        m = self.result
        if m is None or self.mode >= m.hz.size:
            return None
        return m.shapes[:, self.mode] * (SWING * self.canvas.height_m)

    def _show_mode(self) -> None:
        shape = self.mode_shape()
        self.canvas.q = None if shape is None else (shape * np.exp(1j * self._phase)).real
        self.canvas.draw()
        if self.tabs.currentIndex() == TAB_WHIRL:
            self.whirl.show_phase(self._phase)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._start_timer()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()

    def _start_timer(self) -> None:
        if self.isVisible() and self.running and self.result is not None and not self._timer.isActive():
            self._clock.restart()
            self._last = 0.0
            self._timer.start(FRAME_MS)

    def _frame(self) -> None:
        if not self.running:  # paused since this frame was queued
            return
        began = self._clock.elapsed()
        now = began / 1000.0
        dt = min(now - self._last, 0.1)
        self._last = now
        with frame_stats.frame("Rotor builder", lambda: self._phase, FRAME_MS):
            self.step(dt)
            settle_plots(self)
        if self.isVisible() and self.running and self.result is not None:
            self._timer.start(max(1, FRAME_MS - (self._clock.elapsed() - began)))

    def frame_context(self) -> str:
        r = self.rotor
        return (
            f"{len(r.sections)} sections, {len(r.discs)} discs, {len(r.bearings)} bearings, {r.nodes.size} nodes, "
            f"mode {self.mode + 1} at {self.rpm.value():g} rpm"
        )

    def step(self, dt: float) -> None:
        """Whirl the mode on by dt seconds at the animation's own pace (not while a part is dragged)."""
        if self.canvas._drag is not None:
            return
        self._phase = (self._phase + TWO_PI * SHOW_HZ * dt) % TWO_PI
        self._show_mode()

    def apply_theme(self) -> None:
        self.canvas.apply_theme()
        self.whirl.apply_theme()
        mute(self.hint)
        self._fill_table()
        self._draw_tabs()
