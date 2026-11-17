"""The Rotor builder page's Things to try and practice problems: its trial dataclass and the tabs, matrices and
swings it names, the trials' rotors, and the problems, each worked from its printed numbers with
core/rotor_builder.py (`first_mode`: a rotor's slowest mode at a speed).

`RotorBuilderPage.apply_trial` applies a trial; the page re-exports `TRIALS` and `PROBLEMS`, where the practice
window and the Theory tab look for them. `tests/test_tutorial_numbers.py` checks the trials' numbers,
`tests/test_practice.py` the problems.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..core.fluid_bearing import JournalBearing
from ..core.frame import STEEL
from ..core.magnetic_bearing import MagneticBearing
from ..core.practice import Prediction, Problem, Question, Requirement, at_least, kept, number, pick
from ..core.rotor_builder import (
    DISC_LIBRARY,
    RPM,
    TWO_PI,
    Bearing,
    BearingKind,
    Disc,
    Rotor,
    RotorError,
    Seal,
    Section,
    Selection,
    Theory,
    assemble,
    modes,
    pinned_ends,
    solid_disc,
    uniform,
)
from ..core.rotor_builder_sweep import sweep
from .links import page_link

TAB_WHIRL, TAB_CAMPBELL, TAB_UNBALANCE, TAB_STABILITY, TAB_ASSEMBLY, TAB_ELEMENT, TAB_THEORY = range(7)


class Matrix(Enum):
    K = "Stiffness K(Ω)"
    M = "Mass M"
    C = "Damping C(Ω)"
    G = "Gyroscopic G"


MATRICES = tuple(Matrix)


class Swing(Enum):
    MODE = "The mode picked"
    UNBALANCE = "The unbalance response"


SWINGS = tuple(Swing)
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
    top_rpm: float = 12000.0  # the sweeps' (Campbell, unbalance, stability) top speed
    probe: int | None = None  # the node the Unbalance tab plots (None: the first disc's)
    swing: Swing = Swing.MODE
    predict: Prediction | None = None  # asked before it runs (Things to try)


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
    internal=1e-6,
)
SEALED = dataclasses.replace(
    ANISOTROPIC,
    discs=(DISC_LIBRARY[1].at(0.5),),
    bearings=(SPRINGS, dataclasses.replace(SPRINGS, z=1.0)),
    seals=(Seal(0.25, 2e6, 2e3, 0.5),),
)
DAMPED = dataclasses.replace(SPRINGS, cxx=1e4, cyy=1e4)
INTERNAL = dataclasses.replace(
    UNIFORM,
    discs=(DISC_LIBRARY[1].at(0.5),),
    bearings=(DAMPED, dataclasses.replace(DAMPED, z=1.0)),
    internal=1e-4,
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
        "A disc off centre",
        "The pinned shaft with a steel disc 300 × 40 mm (22.2 kg) a quarter along, at 3000 rpm. At rest its pairs "
        "are at 62.28, 265.2 and 501.4 Hz; spinning, each splits, backward below and forward above: 60.83 and "
        "63.70, 261.2 and 268.6, 468.2 and 538.2 Hz. The disc tilts as the shaft bends, most in the third mode, "
        "which splits most. <m>G</m> (the Assembly tab) is where it comes from: <m>I<sub>p</sub></m> between the "
        "disc's slopes.",
        QUARTER_DISC,
        rpm=3000.0,
        picked=("disc", 0),
        matrix=Matrix.G,
        predict=Prediction(
            "A disc a quarter along the pinned shaft, spinning at 3000 rpm. At rest each mode is a pair at one "
            "frequency. What does the spin do to each pair?",
            (
                "Both fall",
                "Both rise",
                "Nothing: the spin does no work",
                "The forward whirl rises, the backward one falls",
            ),
            3,
        ),
    ),
    RotorTrial(
        "An overhung fan",
        "A titanium fan 500 × 20 mm (17.4 kg) 200 mm past the right-hand bearing of a 60 mm steel shaft. At rest "
        "the first pair is at 79.06 Hz; at 3000 rpm it splits to 71.13 (backward) and 86.24 Hz (forward), at 6000 "
        "rpm to 63.10 and 92.30. An overhung disc tilts as it whirls, and a thin one "
        "(<m>I<sub>p</sub> ≈ 2I<sub>d</sub></m>) stiffens its forward whirl strongly.",
        OVERHUNG,
        rpm=3000.0,
        picked=("disc", 0),
    ),
    RotorTrial(
        "Free: no bearings",
        "The 50 mm shaft with nothing holding it. At rest there are 8 roots at 0 (its two translations and two "
        "tilts, each twice: a free body has no stiffness), then pairs at 228.7 and 623.3 Hz (Euler's free–free "
        "beam: 230.2 Hz). At 3000 rpm two of the zero roots become a slow forward whirl at 0.19 Hz: the free "
        "shaft's nutation, <m>I<sub>p</sub>Ω/I<sub>d</sub></m>, <m>I<sub>p</sub>/I<sub>d</sub> = 1.5D²/L²</m> = 0.00375 of its 50 "
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
        "167.8 and 171.0 Hz, 199.3 and 207.4, 422.5 and 442.9. On the Assembly tab, <m>K</m> with the bored "
        "section's share tinted: each beam's 8 × 8 adds at its two nodes' [<m>x</m>, <m>θ<sub>x</sub></m>, <m>y</m>, "
        "<m>θ<sub>y</sub></m>], so <m>K</m> is banded.",
        COMPRESSOR,
        rpm=6000.0,
        tab=TAB_ASSEMBLY,
        picked=("section", 2),
        grid=0.025,
    ),
    RotorTrial(
        "Oil-film bearings",
        "Plain journal bearings 50 mm across and 30 mm long, 50 µm clearance, 20 mPa·s oil, each carrying 184.5 N "
        "(half the rotor's 37.6 kg). At 3000 rpm the films' <m>K</m> and <m>C</m> are full, with cross terms (the Element tab): "
        "the rotor bounces on them at 25.3 and 25.6 Hz, damped 24 %, and bends at 50.9 and 51.1 Hz. The oil "
        "swirls at about half the speed: past 5466 rpm, where that nears the bending mode, its forward whirl's <m>ζ</m> "
        "goes negative and it grows by itself (at 6000 rpm, −3.0 %): oil whip.",
        FILMS,
        rpm=3000.0,
        picked=("bearing", 0),
        tab=TAB_ELEMENT,
    ),
    RotorTrial(
        "Magnetic bearings",
        "Active magnetic bearings (0.5 mm gaps, 2 A bias): each magnet's pull is a negative stiffness, "
        "<m>k<sub>m</sub></m> = 0.543 MN/m on <m>K</m>'s diagonal, which the PID beats (<m>k<sub>i</sub>k<sub>p</sub></m> = 2.71 MN/m). The first pair "
        "is at 37.85 Hz with <m>ζ</m> = 4.57 %, the second at 143.0 with 38.4 %: the derivative damps. The controllers' "
        "12 states join <m>q</m> and <m>q̇</m> in the state; four of their roots are real, at −12.7 /s, the integral action "
        "pulling each journal back to the centre. Above about 1.2 kHz the controller's lag turns its damping "
        "negative: the shaft's own internal damping (<m>τ<sub>i</sub></m> = 1 µs) keeps its 2.4 kHz bending modes from "
        "growing. Set <m>τ<sub>i</sub></m> to 0 and they do (<m>ζ</m> = −0.61 %).",
        MAGNETS,
        picked=("bearing", 0),
        tab=TAB_ELEMENT,
    ),
    RotorTrial(
        "A seal's swirl",
        "Stiff bearings and an annular seal a quarter along (2 MN/m, 2 kN·s/m, its fluid swirling at half the "
        "shaft's speed). At rest it damps both whirls of the first pair to 4.17 %. At 6000 rpm its cross-coupling, "
        "<m>λΩc</m> = 0.63 MN/m, pushes a forward whirl along and holds a backward one back: forward 0.516 %, backward "
        "7.74 %. Past 6845 rpm the forward one grows: the seal drives it.",
        SEALED,
        rpm=6000.0,
        picked=("seal", 0),
        tab=TAB_ELEMENT,
    ),
    RotorTrial(
        "Campbell diagram: the overhung fan",
        "The fan's branches against the speed: each pair fans out, backward falling and forward rising. Where one "
        "meets 1X (the red line) is a critical speed: 4091 rpm (the first pair's backward branch), 5480 rpm (its "
        "forward one) and 9507 rpm (the second pair's backward). Its amplification factor <m>1/2ζ</m> says how sharply "
        "an unbalance response could peak there: 116 at 5480 rpm, the bearings' 500 N·s/m barely damping it. "
        "Running at 3000 rpm the fan is 36.4 % below the nearest: its separation margin.",
        OVERHUNG,
        rpm=3000.0,
        tab=TAB_CAMPBELL,
        picked=("disc", 0),
    ),
    RotorTrial(
        "The fan's unbalance",
        "The fan's unbalance, 174 g·mm (10 µm of its 17.4 kg). Its orbit peaks at the forward critical, 1003 µm "
        "at 5480 rpm, lagging the heavy spot by 90°; at the backward criticals (4091 and 9507 rpm, grey) it is "
        "12.5 and 13.7 µm, nothing to see. An unbalance turns forward with the shaft, and on round bearings it "
        "can't drive a backward whirl. Past the critical the lag goes to 180°: the disc turns about its centre "
        "of mass, its heavy spot inside the orbit.",
        OVERHUNG,
        rpm=5480.0,
        tab=TAB_UNBALANCE,
        picked=("disc", 0),
        swing=Swing.UNBALANCE,
        predict=Prediction(
            "The overhung fan run up with its unbalance through its criticals: 4091 rpm (backward), 5480 rpm "
            "(forward) and 9507 rpm (backward). How big is its orbit at the backward ones?",
            (
                "As big as at the forward one",
                "Bigger: the backward whirls are less damped",
                "Tiny: an unbalance turns forward, with the shaft",
            ),
            2,
        ),
    ),
    RotorTrial(
        "The compressor's criticals",
        "The stepped compressor rotor, its bearings damped 5 kN·s/m. Its criticals are 9985 rpm (backward), 10308 "
        "(forward), 11758 (backward) and 12748 (forward), with amplification factors from 9.2 to 10.3. At its "
        "6000 rpm it is 66.4 % below the first, and the unbalance (on the Unbalance tab: 156 g·mm on each "
        "impeller, 71 on the hub) whirls the first impeller 2.15 µm; at 10308 rpm, 35.4 µm.",
        COMPRESSOR,
        rpm=6000.0,
        tab=TAB_CAMPBELL,
        grid=0.025,
        top_rpm=15000.0,
    ),
    RotorTrial(
        "Stability map: oil whip",
        "The rotor on oil films, swept to 9000 rpm. The bounce on the films is overdamped (its <m>δ</m> far above 1, its "
        "1X crossings hollow on the Campbell diagram: critically damped). The bending pair's <m>δ</m> are 0.193 "
        "(forward) and 0.0632 (backward) at 3000 rpm; the forward one's crosses 0 at 5466 rpm (the red line) "
        "and at 6000 rpm (shown) it is −0.190: it grows, oil whip. It misses API 684's <m>δ</m> ≥ 0.1 (dashed) well "
        "before.",
        FILMS,
        rpm=6000.0,
        tab=TAB_STABILITY,
        top_rpm=9000.0,
    ),
    RotorTrial(
        "Internal damping",
        "The 300 mm disc at midspan on bearings of 20 MN/m and 10 kN·s/m, and the shaft's internal damping <m>τ<sub>i</sub></m> = "
        "100 µs. At rest it damps the first pair to 2.04 % (0.62 % without it). But it turns with the shaft, "
        "resisting the bending rate the shaft feels: at the forward critical, 2947 rpm, a synchronous whirl "
        "doesn't flex the turning shaft at all, and <m>ζ</m> is back to 0.62 %. Above it the shaft turns faster than it "
        "whirls, and its internal damping drags the whirl on round: unstable from 4245 rpm, 1.44 × the critical. "
        "Double <m>τ<sub>i</sub></m> and that falls to 3596 rpm; more damping in the bearings raises it.",
        INTERNAL,
        rpm=3000.0,
        tab=TAB_STABILITY,
        top_rpm=8000.0,
        predict=Prediction(
            "The shaft's internal damping damps every mode at rest. What does it do to the forward whirl above the "
            "first critical speed?",
            (
                "Damps it more",
                "Nothing",
                "Drives it: past some speed the whirl grows",
            ),
            2,
        ),
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
        why=(
            "",
            "The gyroscopic moment stiffens the forward whirl but softens the backward one: they split, one up and "
            "one down.",
            "The gyroscopic moment stiffens the forward whirl but softens the backward one: they split, one up and "
            "one down.",
            "It does no work, but it couples the tilts: the moment I<sub>p</sub>Ω × tilt rate stiffens one whirl "
            "and softens the other.",
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


def _log_dec(rng: np.random.Generator) -> Question:
    zeta = pick(rng, (0.5, 1.0, 2.0, 5.0, 10.0, 20.0))
    z = zeta / 100
    delta = TWO_PI * z / math.sqrt(1 - z * z)
    return Question(
        f"A rotor's least damped mode has ζ = {zeta:g} %. What is its logarithmic decrement δ, the ln of the ratio "
        "of one swing's peak to the next, which API 684 asks to be 0.1 or more?",
        delta,
        f"δ = 2πζ/√(1 − ζ²) = {number(delta, 4)}, about 2πζ while ζ is small. δ = 0.1 is ζ = 1.6 %.",
        setup=dict(rotor=SEALED, rpm=6000.0, tab=TAB_STABILITY),
    )


def _critical_speed(rng: np.random.Generator) -> Question:
    diameter = pick(rng, (40.0, 50.0, 60.0))
    kind = int(pick(rng, (0, 1, 2)))
    rotor = pinned_ends(Rotor(uniform(1.0, diameter / 1000, 4, 3), discs=(DISC_LIBRARY[kind].at(0.5),)))
    a = assemble(rotor)
    f0 = float(modes(a, 0.0).hz[0])
    top = 1.5 * 60 * f0
    first = next(c for c in sweep(a, top, n_speeds=25).criticals if c.forward)
    return Question(
        f"A {diameter:g} mm steel shaft 1 m long, pinned at its ends, carries a "
        f"{DISC_LIBRARY[kind].name.lower()} at midspan. At rest its first pair of modes is at {number(f0, 4)} Hz. "
        "At what speed is its first forward critical speed, where unbalance drives it?",
        first.rpm,
        f"At a critical the whirl's frequency is the shaft's, the rpm over 60. A disc at midspan doesn't tilt in the "
        f"first mode, so its gyroscopic moment hardly moves it: the critical ≈ 60 × {number(f0, 4)} = {number(60 * f0, 4)} rpm "
        f"(the Campbell diagram's crossing: {number(first.rpm, 4)} rpm, the shaft's own sections tilting a "
        "little).",
        unit="rpm",
        setup=dict(rotor=rotor, rpm=first.rpm, tab=TAB_CAMPBELL, top_rpm=round(top, -2)),
    )


def _backward(rng: np.random.Generator) -> Question:
    return Question(
        "The overhung fan on round bearings has criticals at 4091 rpm (backward), 5480 rpm (forward) and 9507 "
        "rpm (backward). Running up through them with its unbalance, how big is its orbit at the backward ones?",
        1,
        "The unbalance turns with the shaft, forward, so on round bearings it does no work on a backward whirl: "
        "1003 µm at 5480 rpm, but 12.5 and 13.7 µm at 4091 and 9507 rpm, only the forward branch's tail. On "
        "anisotropic bearings the modes are ellipses, part forward and part backward, and a backward critical "
        "does respond.",
        choices=(
            "As big as at the forward one",
            "Tiny: the unbalance can't drive a backward whirl",
            "Infinite: the backward branch has no damping",
            "Zero exactly, at any bearings",
        ),
        why=(
            "The unbalance turns with the shaft, forward: on round bearings it does no work on a backward whirl.",
            "",
            "The unbalance does no work on a backward whirl on round bearings, so it barely moves it: 12.5 and 13.7 "
            "µm, only the forward branch's tail.",
            "On anisotropic bearings the modes are ellipses, part forward, and a backward critical does respond; on "
            "round ones what's left is the forward branch's tail.",
        ),
        setup=dict(rotor=OVERHUNG, rpm=4091.0, tab=TAB_UNBALANCE, picked=("disc", 0), swing=Swing.UNBALANCE),
        predict=True,
    )


def _internal(rng: np.random.Generator) -> Question:
    return Question(
        "A shaft's internal damping (its material's, or a shrink fit's slip) damps every mode of the rotor at "
        "rest. What does it do to the forward whirl above the first critical speed?",
        2,
        "It turns with the shaft and resists the bending rate the shaft feels, the whirl's less the spin's. "
        "Below the critical the whirl outruns the shaft and it damps; at the critical the whirl is synchronous "
        "and it does nothing; above, the shaft outruns the whirl and it drags it on round, feeding it. With "
        "damping in the bearings the onset comes a little above the critical (here 4245 rpm, 1.44 × 2947).",
        choices=(
            "Damps it more, the faster it spins",
            "Nothing: damping can't drive",
            "Drives it: past some speed above the critical the whirl grows",
            "Stiffens it, like the gyroscopic moment",
        ),
        why=(
            "It resists the bending rate the shaft feels, the whirl's less the spin's: above the critical the shaft "
            "outruns the whirl, and it drags the whirl on.",
            "Damping that turns with the shaft can drive a whirl the shaft outruns: past the critical it feeds it.",
            "",
            "The gyroscopic moment stiffens; internal damping is a force on the bending rate, which feeds the whirl "
            "above the critical.",
        ),
        setup=dict(rotor=INTERNAL, rpm=3000.0, tab=TAB_STABILITY, top_rpm=8000.0),
        predict=True,
    )


def _swirl_brake_spec(rng: np.random.Generator) -> Question:
    target = pick(rng, (8000.0, 9000.0, 10000.0))
    start = SEALED
    seal = start.seals[0]

    def goal(t: RotorTrial) -> tuple[Requirement, ...]:
        s = t.rotor.seals[0] if t.rotor.seals else Seal(0.0, 0.0, 0.0, 0.0)
        try:
            onset = sweep(assemble(t.rotor), target, n_speeds=41).onset
        except RotorError:
            onset = 0.0
        return (
            at_least("the speed it stays stable to", target if onset is None else onset, target, "rpm"),
            kept("the seal's stiffness", s.k / 1e6, seal.k / 1e6, "MN/m"),
            kept("the seal's damping", s.c / 1e3, seal.c / 1e3, "kN·s/m"),
        )

    design = dataclasses.replace(start, seals=(dataclasses.replace(seal, swirl=0.1),))
    return Question(
        f"The rotor with a seal (Things to try: <i>A seal's swirl</i>) grows unstable at 6845 rpm, its fluid "
        f"swirling at half the shaft's speed. Keep it stable to {target:g} rpm without changing the seal's "
        "stiffness or damping (select the seal; the Stability tab shows the onset).",
        0,
        "The seal drives the forward whirl through k<sub>xy</sub> = λΩc: cut the swirl and the drive goes with it. "
        "A <b>swirl brake</b> (vanes at the seal's inlet) does that: at λ = 0.1 it is stable throughout. More "
        "damping in the bearings works too.",
        setup=dict(rotor=start, rpm=6000.0, tab=TAB_STABILITY, picked=("seal", 0), top_rpm=target),
        goal=goal,
        design=dict(rotor=design, rpm=6000.0, tab=TAB_STABILITY, picked=("seal", 0), top_rpm=target),
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
        ("log-dec", "The log decrement", "stability", "Stability", _log_dec),
        ("critical", "A critical speed", "campbell", "The Campbell diagram", _critical_speed),
        ("backward", "Predict: a backward critical", "unbalance", "The unbalance response", _backward),
        ("internal", "Predict: internal damping", "internal", "Internal damping", _internal),
        ("swirl-brake", "Spec: a swirl brake", "stability", "Stability", _swirl_brake_spec),
    )
)
