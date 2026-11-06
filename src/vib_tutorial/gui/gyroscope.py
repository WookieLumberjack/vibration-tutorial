"""Gyroscope page: a spinning top, from Euler's equations to precession and nutation, and on a table.

Left: the setting (a disc on a light stem with its tip in a cup, the same disc on a round tip on a table, or a
tippe top on a table), the top (its mass and size, the round tip and the friction on a table, and gravity), how it
is let go (the spin, the tilt, and in the cup the precession: at rest, at either steady rate, or set), and the
playback. Centre: the top in 3D (drag to turn it), with its axis's path and the vectors ω, L and the moment (on a
table the friction), then its tilt against time, beside the turning points worked out in closed form in the cup.
Right: the effective potential the tilt moves in (on a table, the least energy Jellett's constant allows), the
steady precession rates against the spin, the axis's path from above (on a table, the tip's track), the
simulation's check against what is conserved, and the Theory. The page has its own models (core/gyroscope.py,
core/sliding_top.py), stepped on its own frame loop while it shows.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
import scipy.optimize
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.gyroscope import (
    LAUNCHES,
    RPM,
    SHAPES,
    FreeSpin,
    Launch,
    Motion,
    Shape,
    Top,
    TopSimulator,
    body_rates,
    least_spin,
    steady_precession,
    tilt_and_azimuth,
)
from ..core.practice import Problem, Question, Requirement, kept, number, pick, rounded
from ..core.sliding_top import COLUMNS, RoundTop, SlideSamples, SlidingSimulator
from . import frame_stats, settle_plots
from .cms_time import SampleBuffer
from .lesson import (
    Dot,
    Form,
    RunButton,
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
    relabel,
    set_quietly,
    stack,
    theory,
    three_panes,
    trials_html,
)
from .links import load_notes, page_link
from .style import colors
from .theming import add_legend

TAB_POTENTIAL, TAB_PRECESSION, TAB_PATH, TAB_CHECK, TAB_CHAOS = range(5)
FRAME_MS = 16
PLAYBACK = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
REAL_TIME = PLAYBACK.index(1.0)
NODS = 6  # nutation periods the tilt plot spans
TRAIL_POINTS = 1500  # at most, on the 3D view's trail
PATH_POINTS = 4000  # at most, on the path from above
CHECK_POINTS = 2000
CHECK_FRAMES = 30  # the Check tab's text is worked out over the whole run, every this many frames
SECTION_FRAMES = 10  # the Chaos tab is redrawn every this many frames
HISTORY = 400_000  # samples kept
TWIN = 1e-8  # rad: the Chaos tab's second top is let go this much further over
SECTION_POINTS = 8000  # the Poincaré section's, at most
APART_STRIDE = 20  # samples of the two tops' separation, one kept in this many
SKEW_TRAIL, SKEW_TILT, SKEW_PATH = 3.0, 5.0, 10.0  # s of an asymmetric top's trail, tilt and path
SLIDE_HISTORY = 200_000  # samples kept on a table (RECORD_DT apart)
SLIDE_TILT_SPAN = 60.0  # s, at most, of the tilt on a table
TRACK_SPAN = 10.0  # s of the tip's track on the table
TRACK_POINTS = 3000
TABLE_TRAIL = 1.0  # s of the axis's trail and the tip's track in the 3D view, on a table


class Setting(enum.Enum):
    CUP = "Its tip in a cup"
    TABLE = "On a round tip, on a table"
    TIPPE = "A tippe top, on a table"


SETTINGS = tuple(Setting)


@dataclass(frozen=True)
class GyroTrial:
    name: str
    note: str
    mass: float = 0.5
    radius: float = 0.04
    height: float = 0.03
    gravity: float = 9.81
    spin_rpm: float = 1500.0
    tilt: float = 30.0  # degrees from the vertical
    launch: Launch = Launch.REST
    precession: float = 0.0  # rad/s, with Launch.SET
    playback: int = REAL_TIME  # index into PLAYBACK
    tab: int = TAB_PATH
    setting: Setting = Setting.CUP
    tip_radius: float = 0.01  # m, on a table
    friction: float = 0.3  # μ, on a table
    low: float = 0.0045  # m, a tippe top's weight below its centre
    ratio: float = 0.8  # a tippe top's I₁/I₃
    shape: Shape = Shape.DISC  # radius is a square's side, or a rectangle's a
    breadth: float = 0.04  # m, a rectangle's other side b

    @property
    def top(self) -> Top:
        """The disc on its stem (a tippe top's ball: its mass and radius)."""
        return Top(self.mass, self.radius, self.height, self.gravity, self.shape, self.breadth)

    @property
    def round_top(self) -> RoundTop | None:
        """The top on a table, or None in the cup."""
        if self.setting is Setting.TABLE:
            return RoundTop.disc(self.top, self.tip_radius, self.friction)
        if self.setting is Setting.TIPPE:
            return RoundTop.tippe(self.mass, self.radius, self.low, self.ratio, self.friction, self.gravity)
        return None

    @property
    def motion(self) -> Motion:
        """Let go as set; an asymmetric top has no steady precession to start at, so it starts at rest."""
        top = self.top
        launch = self.launch if top.symmetric or self.launch is Launch.SET else Launch.REST
        return Motion.launched(top, self.spin_rpm * RPM, math.radians(self.tilt), launch, self.precession)


TRIALS = (
    GyroTrial(
        "Let go at rest: cusps",
        "A 0.5 kg disc 40 mm in radius, 30 mm up its stem, spun at 1500 rpm and let go at 30° without pushing it "
        "round. It doesn't fall: it drops only to 31.5° and comes back, nodding 14.7 times a second, while the "
        "whole top precesses round at 2.39 rad/s (a turn in 2.63 s). Each nod starts from rest, so the axis's "
        "path (the Path tab) has a cusp at the top of each.",
    ),
    GyroTrial(
        "Steady precession",
        "The same top let go already precessing at the slow steady rate, 2.39 rad/s: the root of "
        "I₁ cos θ φ'² − I₃ω₃ φ' + mgl = 0. It doesn't nod at all, and its axis traces a circle. "
        "mgl/I₃ω₃ = 2.34 rad/s is the fast top's estimate of it.",
        launch=Launch.SLOW,
    ),
    GyroTrial(
        "Fast precession",
        "The quadratic's other root, 109 rad/s (17 turns a second): steady too, near I₃ω₃/I₁ cos θ = 112 rad/s, "
        "the free top's own wobble. Watch it at 0.05× speed.",
        launch=Launch.FAST,
        playback=PLAYBACK.index(0.05),
    ),
    GyroTrial(
        "Pushed back: loops",
        "Let go precessing backward at −2 rad/s: it drops to 32.7°, and the precession turns forward on the way "
        "down and back at the top of each nod, so the path loops. On average it still precesses forward at "
        "2.39 rad/s.",
        launch=Launch.SET,
        precession=-2.0,
    ),
    GyroTrial(
        "Pushed along: waves",
        "Let go precessing forward at 1.2 rad/s, half the steady rate: it drops less, to 30.7°, and never stops "
        "precessing, so the path waves. Faster than the steady rate (try 4 rad/s) it rises instead.",
        launch=Launch.SET,
        precession=1.2,
    ),
    GyroTrial(
        "Spun slowly: deep nods",
        "At 600 rpm the nods are deep, from 30° to 43.9°, and slow, 4.53 a second; the precession is 6.70 rad/s "
        "on average. The fast top's estimates (nods 2I₁mgl sin θ/(I₃ω₃)² = 8.7° deep, at I₃ω₃/2πI₁ = 6.2 Hz) no "
        "longer hold.",
        spin_rpm=600.0,
        playback=PLAYBACK.index(0.25),
    ),
    GyroTrial(
        "A sleeping top",
        "Nearly upright (2°) at 1000 rpm, over the 467 rpm that I₃ω₃ > 2√(I₁mgl) asks: it stays up, nodding "
        "between 2° and 2.26°. Upright, the Euler angles φ and ψ turn about the same axis, so their equations "
        "are singular there; the simulation steps the orientation as a quaternion and doesn't notice.",
        spin_rpm=1000.0,
        tilt=2.0,
    ),
    GyroTrial(
        "Too slow to sleep",
        "At 400 rpm, under 467, upright is unstable: let go at 2°, it falls to 62.2° before it comes back up, "
        "nearly a second a nod. A real top's spin dies away in the friction at its tip, and once under 467 rpm it "
        "starts to wobble.",
        spin_rpm=400.0,
        tilt=2.0,
        tab=TAB_POTENTIAL,
    ),
    GyroTrial(
        "Lying flat",
        "At 90°, as a bicycle wheel held by one end of its axle, steady precession has the one root, "
        "mgl/I₃ω₃ = 2.34 rad/s exactly: a turn in 2.68 s. Gravity's moment mgl turns the angular momentum I₃ω₃ "
        "round without changing its size.",
        tilt=90.0,
        launch=Launch.SLOW,
    ),
    GyroTrial(
        "Spun the other way",
        "At −1500 rpm it does everything in mirror: the precession is −2.39 rad/s. The precession always "
        "turns the angular momentum toward the moment.",
        spin_rpm=-1500.0,
    ),
    GyroTrial(
        "No gravity: nutation is the free wobble",
        "With g = 0 and let go precessing at 5 rad/s, nothing acts on it, so L stays put, 28.5° from the "
        "vertical, and the axis turns round L once every 2πI₁/|L| = 65.0 ms. Seen from the vertical, its tilt "
        "nods between 27.0° and 30°: nutation is this free wobble, which gravity's precession carries round.",
        gravity=0.0,
        launch=Launch.SET,
        precession=5.0,
        playback=PLAYBACK.index(0.05),
        tab=TAB_PATH,
    ),
    GyroTrial(
        "A square top: just like a disc",
        "A 0.5 kg square plate 69.28 mm on a side in place of the disc: I₃ = ms²/6 = 4 kg·cm², the first trial's "
        "disc's mr²/2 (r = s/√3 = 40 mm), and about every line across its axis ms²/12, that disc's mr²/4. So it "
        "does just what the disc did: drops to 31.5°, nods 14.7 times a second and precesses at 2.39 rad/s. Its "
        "corners make no difference; only its moments do.",
        shape=Shape.SQUARE,
        radius=0.06928,
    ),
    GyroTrial(
        "A rectangle: ω₃ isn't kept",
        "A 0.5 kg plate 100 × 40 mm, 30 mm up its stem: I₁ = 5.17, I₂ = 8.67 and I₃ = 4.83 kg·cm², so I₁ ≠ I₂. Its "
        "spin ω₃ now wanders, between 596.7 and 601.1 rpm, so the tilt has no potential of its own: it nods between "
        "28.0° and 40.7°, which no closed form gives. It spins about its smallest moment and stays regular: the Chaos "
        "tab's dots lie on closed curves, and a second top let go 10⁻⁸ rad further over is still only 10⁻⁷ rad away "
        "after 20 s.",
        shape=Shape.RECTANGLE,
        radius=0.1,
        breadth=0.04,
        spin_rpm=600.0,
        playback=PLAYBACK.index(0.25),
        tab=TAB_CHAOS,
    ),
    GyroTrial(
        "No gravity: the tennis-racket flip",
        "A plate 100 × 20 mm, 15 mm up its stem, with g = 0: I₁ = 1.29 &lt; I₃ = 4.33 &lt; I₂ = 5.29 kg·cm², so it "
        "spins about its middle axis, with nothing acting on it. It flips upside down and back, crossing the "
        "horizontal every 0.455 s: half the period 4K(k)/λ = 0.910 s of Euler–Poinsot's solution in Jacobi's "
        "elliptic functions (k² = 1 − 1.2 × 10⁻⁷: near the separatrix, where the period grows without bound). The "
        "Check tab compares ω with it. The free asymmetric top still has a closed form, if a much harder one.",
        shape=Shape.RECTANGLE,
        radius=0.1,
        breadth=0.02,
        height=0.015,
        gravity=0.0,
        spin_rpm=600.0,
        tilt=2.0,
        launch=Launch.SET,
        precession=0.5,
        playback=PLAYBACK.index(0.1),
        tab=TAB_CHECK,
    ),
    GyroTrial(
        "Gravity and the middle axis: chaos",
        "The same plate with gravity, let go at 30°: it tumbles, and no closed form exists. Two tops let go "
        "10⁻⁸ rad apart are 10⁻⁶ rad apart at 1.1 s, 10⁻² at 3.0 s and a whole radian at 5.0 s: the gap multiplies "
        "over and over, the mark of chaos, and the Chaos tab's dots scatter. Spun faster it is no tamer.",
        shape=Shape.RECTANGLE,
        radius=0.1,
        breadth=0.02,
        height=0.015,
        spin_rpm=600.0,
        playback=PLAYBACK.index(0.25),
        tab=TAB_CHAOS,
    ),
    GyroTrial(
        "On a slippery table: it rises",
        "Out of the cup: a disc 20 mm up a stem whose tip is rounded to 10 mm, let go at 20° spinning at 1500 rpm "
        "on a table with μ = 0.01. Its tip slides, and friction raises it: under 10° in 12 s, 5° in 19 s. Jellett's "
        "J = I₃ω₃(a + R cos θ) stays put as friction takes energy out, and the least energy J allows (the Potential "
        "tab) is upright. Below 4.6° (sin θ = μI₃/maR) friction can make its tip roll instead, and the rise stalls "
        "there, spinning at 1457 rpm (upright, J would leave 1455).",
        setting=Setting.TABLE,
        height=0.02,
        friction=0.01,
        tilt=20.0,
        playback=PLAYBACK.index(2.0),
        tab=TAB_POTENTIAL,
    ),
    GyroTrial(
        "On a rough table: it rolls instead",
        "The same top on wood, μ = 0.3: its tip catches and rolls, so friction takes hardly any energy and it "
        "doesn't rise. It drops to 16.2° at once, then rolls round a circle about 1.35 m across (the Path tab), "
        "tilted all the way.",
        setting=Setting.TABLE,
        height=0.02,
        friction=0.3,
        tilt=20.0,
        tab=TAB_PATH,
    ),
    GyroTrial(
        "A tippe top turns over",
        "A 20 g ball 15 mm in radius, weighted 4.5 mm below its centre (α = 0.3), with I₁/I₃ = 0.8, spun at "
        "1200 rpm on its bottom, 5° off upright. 1 − α &lt; I₁/I₃ &lt; 1 + α and it spins faster than 468 rpm, so "
        "turned over is the least energy Jellett's J allows: friction turns it over, past 90° in 1.8 s, and it ends "
        "on its top spinning at J/I₃(R − a) = 643 rpm, the same way seen from above but the other way about its "
        "own axis. Its centre of mass rises 9 mm, and half its energy goes in the friction.",
        setting=Setting.TIPPE,
        mass=0.02,
        radius=0.015,
        spin_rpm=1200.0,
        tilt=5.0,
        playback=PLAYBACK.index(0.25),
        tab=TAB_POTENTIAL,
    ),
    GyroTrial(
        "A tippe top spun too slowly",
        "The same tippe top at 400 rpm, under the 468 rpm at which upright stops being E_J's least: it stays up, "
        "wobbling a few degrees.",
        setting=Setting.TIPPE,
        mass=0.02,
        radius=0.015,
        spin_rpm=400.0,
        tilt=5.0,
        playback=PLAYBACK.index(0.25),
        tab=TAB_POTENTIAL,
    ),
    GyroTrial(
        "A tippe top that can't turn over",
        "With I₁/I₃ = 1.4, over 1 + α = 1.3, turned over is no minimum: E_J is least at 108.6°, and the ball "
        "rolls over past its side and spins there, near 110°.",
        setting=Setting.TIPPE,
        mass=0.02,
        radius=0.015,
        ratio=1.4,
        spin_rpm=1200.0,
        tilt=5.0,
        playback=PLAYBACK.index(0.25),
        tab=TAB_POTENTIAL,
    ),
)


def _disc(rng: np.random.Generator) -> tuple[float, float, float]:
    return pick(rng, (0.2, 0.5, 1.0, 2.0)), pick(rng, (0.03, 0.04, 0.05, 0.06)), pick(rng, (0.02, 0.03, 0.04))


# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


def _square_moments(rng: np.random.Generator) -> Question:
    m, l = pick(rng, (0.2, 0.5, 1.0, 2.0)), pick(rng, (0.02, 0.03, 0.04))
    side = pick(rng, (0.05, 0.06, 0.08, 0.1))
    top = Top(m, side, l, shape=Shape.SQUARE)
    return Question(
        f"A top is a {m:g} kg square plate {side * 1000:g} mm on a side on a light stem, its centre {l * 1000:g} mm "
        "above the tip. What is its moment of inertia I₁ about a line across the axis through the tip, in kg·cm²?",
        top.transverse * 1e4,
        f"About a line through its centre parallel to a side a square plate has ms²/12, and a quarter turn leaves it "
        f"as it was, so it has that about every line across its axis; moved l up the axis it gains ml²: "
        f"I₁ = m(s²/12 + l²) = {m:g} × ({number(side * side / 12)} + {number(l * l)}) m² = "
        f"{number(top.transverse * 1e4)} kg·cm². (Along the axis, I₃ = ms²/6 = {number(top.polar * 1e4)} kg·cm².)",
        unit="kg·cm²",
        setup=dict(mass=m, radius=side, height=l, shape=Shape.SQUARE),
    )


def _square_swap(rng: np.random.Generator) -> Question:
    return Question(
        "A spinning top's 40 mm disc is swapped for a square plate of the same mass, 69.3 mm on a side, at the same "
        "height, spun and let go the same way. How does its motion change?",
        0,
        "Not at all: the square's moments are ms²/6 along its axis and ms²/12 across it, about every line across it, "
        "which for s = √3 × 40 mm are the disc's mr²/2 and mr²/4. The motion depends only on the moments.",
        choices=(
            "Not at all",
            "It wobbles four times a turn, at its corners",
            "It precesses faster",
            "It nods more deeply",
        ),
        setup=dict(shape=Shape.SQUARE, radius=0.06928, tab=TAB_PATH),
        predict=True,
    )


def _rectangle_moments(rng: np.random.Generator) -> Question:
    m, l = pick(rng, (0.2, 0.5, 1.0)), pick(rng, (0.01, 0.02, 0.03))
    a, b = pick(rng, (0.08, 0.1, 0.12)), pick(rng, (0.02, 0.03, 0.04))
    top = Top(m, a, l, shape=Shape.RECTANGLE, breadth=b)
    return Question(
        f"A top is a {m:g} kg rectangular plate {a * 1000:g} × {b * 1000:g} mm on a light stem, its centre "
        f"{l * 1000:g} mm above the tip. What is its moment of inertia about the line through the tip parallel to "
        f"its {b * 1000:g} mm sides, in kg·cm²?",
        top.second * 1e4,
        f"About a line through its centre parallel to the {b * 1000:g} mm sides the plate has ma²/12, a = "
        f"{a * 1000:g} mm being how far it reaches across that line; moved l up the axis it gains ml²: I₂ = m(a²/12 + l²) = "
        f"{number(top.second * 1e4)} kg·cm². (Parallel to the other sides, I₁ = m(b²/12 + l²) = "
        f"{number(top.transverse * 1e4)} kg·cm²: unequal, an asymmetric top.)",
        unit="kg·cm²",
        setup=dict(mass=m, radius=a, breadth=b, height=l, shape=Shape.RECTANGLE),
    )


def _racket(rng: np.random.Generator) -> Question:
    l = pick(rng, (0.005, 0.015, 0.025, 0.035))
    top = Top(0.5, 0.1, l, 0.0, Shape.RECTANGLE, 0.02)
    i1, i2, i3 = top.moments
    middle = min(i1, i2) < i3 < max(i1, i2)
    flips, keeps = "It flips over and back, again and again", "It keeps spinning steadily about its axis"
    return Question(
        f"A plate 100 × 20 mm on a stem, its centre {l * 1000:g} mm above the tip, is spun about its stem with no "
        "gravity, a little off the axis. What does it do?",
        0,
        f"I₁ = m(b²/12 + l²) = {number(i1 * 1e4)}, I₂ = m(a²/12 + l²) = {number(i2 * 1e4)} and I₃ = m(a² + b²)/12 = "
        f"{number(i3 * 1e4)} kg·cm²: I₃ is the "
        + ("middle one, and spin about the middle axis is unstable: the tennis-racket flip." if middle else "")
        + ("" if middle else f"{'largest' if i3 > max(i1, i2) else 'smallest'}, and spin about it is stable."),
        choices=(flips, keeps) if middle else (keeps, flips),
        setup=dict(
            shape=Shape.RECTANGLE,
            radius=0.1,
            breadth=0.02,
            height=l,
            gravity=0.0,
            spin_rpm=600.0,
            tilt=2.0,
            launch=Launch.SET,
            precession=0.5,
            playback=PLAYBACK.index(0.1),
            tab=TAB_CHECK,
        ),
        predict=True,
    )


def _moments(rng: np.random.Generator) -> Question:
    m, r, l = _disc(rng)
    top = Top(m, r, l)
    return Question(
        f"A top is a {m:g} kg disc of radius {r * 1000:g} mm on a light stem, its centre {l * 1000:g} mm above the "
        "tip. What is its moment of inertia I₁ about a line across the axis through the tip, in kg·cm²?",
        top.transverse * 1e4,
        f"About its own diameter a disc has mr²/4; moved l up the axis it gains ml² (parallel axes): "
        f"I₁ = m(r²/4 + l²) = {m:g} × ({number(r * r / 4)} + {number(l * l)}) m² = {number(top.transverse)} kg·m² "
        f"= {number(top.transverse * 1e4)} kg·cm². (Along the axis, I₃ = mr²/2 = {number(top.polar * 1e4)} kg·cm².)",
        unit="kg·cm²",
        setup=dict(mass=m, radius=r, height=l),
    )


def _flat_precession(rng: np.random.Generator) -> Question:
    m, r, l = _disc(rng)
    rpm = pick(rng, (1000, 1500, 2000, 3000))
    top = Top(m, r, l)
    rate = top.weight_moment / (top.polar * rpm * RPM)
    return Question(
        f"A {m:g} kg disc of radius {r * 1000:g} mm spins at {rpm:g} rpm on a light axle, its centre "
        f"{l * 1000:g} mm from the end of the axle, which rests on a pivot with the axle horizontal. How long does "
        "it take to precess once round the pivot?",
        2 * math.pi / rate,
        f"Gravity's moment mgl = {number(top.weight_moment)} N·m turns the angular momentum I₃ω₃ = "
        f"{number(top.polar)} × {number(rpm * RPM)} = {number(top.polar * rpm * RPM)} N·m·s round, at "
        f"Ω = mgl/I₃ω₃ = {number(rate)} rad/s: once in 2π/Ω = {number(2 * math.pi / rate)} s. Lying flat that is "
        "exact (the quadratic for the rate loses its I₁ cos θ term).",
        unit="s",
        setup=dict(mass=m, radius=r, height=l, spin_rpm=rpm, tilt=90.0, launch=Launch.SLOW, tab=TAB_PATH),
    )


def _steady_tilted(rng: np.random.Generator) -> Question:
    m, r, l = _disc(rng)
    tilt = pick(rng, (20, 30, 45, 60))
    top = Top(m, r, l)
    rpm = rounded(pick(rng, (1.5, 2, 3, 4)) * least_spin(top, math.radians(tilt)) / RPM, 2)
    slow, fast = steady_precession(top, rpm * RPM, math.radians(tilt))
    return Question(
        f"A top has I₁ = {number(top.transverse)} kg·m², I₃ = {number(top.polar)} kg·m² and mgl = "
        f"{number(top.weight_moment)} N·m. Spinning at {rpm:g} rpm, at what rate can it precess slowly and steadily "
        f"at {tilt:g}° from the vertical?",
        slow,
        f"Held at θ, the tilt's equation leaves I₁ cos θ φ'² − I₃ω₃ φ' + mgl = 0, with I₃ω₃ = "
        f"{number(top.polar * rpm * RPM)} N·m·s and I₁ cos θ = {number(top.transverse * math.cos(math.radians(tilt)))}. "
        f"Its smaller root is {number(slow)} rad/s (the larger, {number(fast)} rad/s, is the fast precession); the "
        f"fast top's estimate mgl/I₃ω₃ = {number(top.weight_moment / (top.polar * rpm * RPM))} rad/s.",
        unit="rad/s",
        setup=dict(mass=m, radius=r, height=l, spin_rpm=rpm, tilt=float(tilt), launch=Launch.SLOW, tab=TAB_PRECESSION),
    )


def _sleeping(rng: np.random.Generator) -> Question:
    m, r, l = _disc(rng)
    top = Top(m, r, l)
    least = top.sleeping_spin / RPM
    return Question(
        f"A top has I₁ = {number(top.transverse)} kg·m², I₃ = {number(top.polar)} kg·m² and mgl = "
        f"{number(top.weight_moment)} N·m. How fast must it spin to sleep, upright and steady?",
        least,
        f"Upright is stable while I₃²ω₃² &gt; 4I₁mgl: ω₃ &gt; 2√(I₁mgl)/I₃ = {number(top.sleeping_spin)} rad/s = "
        f"{number(least)} rpm.",
        unit="rpm",
        setup=dict(mass=m, radius=r, height=l, spin_rpm=rounded(1.3 * least), tilt=2.0, tab=TAB_POTENTIAL),
    )


def _nutation(rng: np.random.Generator) -> Question:
    m, r, l = _disc(rng)
    rpm = pick(rng, (2000, 3000, 5000))
    top = Top(m, r, l)
    estimate = top.polar * rpm * RPM / top.transverse / (2 * math.pi)
    exact = Motion(top, rpm * RPM, math.radians(30)).nutation_hz()
    return Question(
        f"A top has I₁ = {number(top.transverse)} kg·m² and I₃ = {number(top.polar)} kg·m². Spun fast, at {rpm:g} "
        "rpm, and let go, about how many times a second does its axis nod?",
        estimate,
        f"A fast top nods at the free top's wobble, I₃ω₃/I₁ = {number(top.polar * rpm * RPM / top.transverse)} rad/s: "
        f"{number(estimate)} Hz. (Exactly, let go at 30°, the elliptic integral gives {number(exact)} Hz.)",
        unit="Hz",
        tolerance=0.03,
        setup=dict(mass=m, radius=r, height=l, spin_rpm=rpm, tilt=30.0, playback=PLAYBACK.index(0.05)),
    )


def _faster_spin(rng: np.random.Generator) -> Question:
    n = int(pick(rng, (2, 3, 4)))
    return Question(
        f"A fast top precesses slowly round at 30°. Spun {n} times faster and let go the same way, how does its "
        "precession change?",
        0,
        f"It precesses {n} times more slowly: the rate is about mgl/I₃ω₃. Gravity's moment is the same, and turns "
        f"an angular momentum {n} times larger.",
        choices=(f"{n} times more slowly", f"{n} times faster", "No change", f"{n * n} times more slowly"),
        setup=dict(spin_rpm=1500.0 * n, tilt=30.0, launch=Launch.SLOW, tab=TAB_PRECESSION),
        predict=True,
    )


def _at_rest_path(rng: np.random.Generator) -> Question:
    return Question(
        "A fast top is let go tilted, without being pushed round. What does the tip of its axis trace?",
        0,
        "Cusps: it starts at rest, so at the top of every nod it stops (φ' = 0) before it falls and is turned "
        "round again.",
        choices=("Cusps", "Loops", "Smooth waves", "A circle"),
        setup=dict(tilt=30.0, launch=Launch.REST, tab=TAB_PATH),
        predict=True,
    )


def _pushed_back(rng: np.random.Generator) -> Question:
    return Question(
        "A spinning top is let go tilted with a push the opposite way to its precession. What does the tip of its "
        "axis trace?",
        0,
        "Loops: the precession rate φ' = (b − a cos θ)/sin²θ changes sign between the turning points, so it runs "
        "backward at one of them.",
        choices=("Loops", "Cusps", "Smooth waves", "A circle"),
        setup=dict(tilt=30.0, launch=Launch.SET, precession=-2.0, tab=TAB_PATH),
        predict=True,
    )


def _too_slow(rng: np.random.Generator) -> Question:
    return Question(
        "A top that sleeps at 1000 rpm (it needs 467) is let go at 2° spinning at 400 rpm instead. What does it do?",
        0,
        "It falls far: under 467 rpm upright is unstable, and it nods from 2° down to 62° and back.",
        choices=(
            "It falls far and comes back up",
            "It stays near 2°, nodding a little",
            "It stands upright",
            "It precesses at 2° without nodding",
        ),
        setup=dict(spin_rpm=400.0, tilt=2.0, tab=TAB_POTENTIAL),
        predict=True,
    )


def _reverse(rng: np.random.Generator) -> Question:
    return Question(
        "A precessing top's spin is reversed and it is let go the same way. Which way does it precess?",
        0,
        "The other way: the precession turns L toward gravity's moment, and with L reversed it has to turn the "
        "other way to get there.",
        choices=("The other way", "The same way", "It doesn't precess"),
        setup=dict(spin_rpm=-1500.0, tilt=30.0, launch=Launch.SLOW),
        predict=True,
    )


def _period_spec(rng: np.random.Generator) -> Question:
    m, r, l = _disc(rng)
    tilt = pick(rng, (20, 30, 45))
    top = Top(m, r, l)
    rpm = rounded(pick(rng, (2, 3, 4)) * least_spin(top, math.radians(tilt)) / RPM, 2)
    target = rounded(2 * math.pi / steady_precession(top, rpm * RPM, math.radians(tilt))[0])

    def goal(t: GyroTrial):
        now = t.motion
        rate = abs(now.mean_precession())
        return (
            kept("Mass", t.mass, m, "kg"),
            kept("Radius", t.radius * 1000, r * 1000, "mm"),
            kept("Height", t.height * 1000, l * 1000, "mm"),
            kept("Tilt", t.tilt, tilt, "°"),
            Requirement(
                f"A precession period within ±2 % of {number(target)} s",
                f"{number(2 * math.pi / rate)} s" if rate else "none",
                bool(rate) and abs(2 * math.pi / rate / target - 1) <= 0.02,
            ),
            Requirement(
                "No nodding (turning points within 0.1°)",
                f"{number(math.degrees(now.turning()[1] - now.turning()[0]), 2)}°",
                math.degrees(now.turning()[1] - now.turning()[0]) <= 0.1,
            ),
        )

    return Question(
        f"Spin this {m:g} kg top (radius {r * 1000:g} mm, {l * 1000:g} mm up the stem) and let it go at {tilt:g}° "
        f"so that it precesses steadily, once round in {number(target)} s, without nodding.",
        0,
        f"Steady at θ: I₁ cos θ φ'² − I₃ω₃ φ' + mgl = 0 with φ' = 2π/{number(target)} s = "
        f"{number(2 * math.pi / target)} rad/s, so ω₃ = (I₁ cos θ φ'² + mgl)/I₃φ': about {number(rpm)} rpm. Let "
        "it go at the slow steady rate, or it nods.",
        setup=dict(mass=m, radius=r, height=l, tilt=float(tilt), spin_rpm=rounded(rpm * 0.6), launch=Launch.REST),
        goal=goal,
        design=dict(mass=m, radius=r, height=l, tilt=float(tilt), spin_rpm=rpm, launch=Launch.SLOW),
    )


def _sleep_spec(rng: np.random.Generator) -> Question:
    m, r = pick(rng, (0.2, 0.5, 1.0)), pick(rng, (0.03, 0.04, 0.05))
    rpm = pick(rng, (600, 800, 1000))
    # the highest centre that still sleeps with 20 % to spare (I₃ω₃/1.2 = 2√(I₁ mgl)), down to a tenth of a mm
    highest = scipy.optimize.brentq(lambda h: Top(m, r, h).sleeping_spin - rpm * RPM / 1.2, 1e-4, 1.0)
    best = math.floor(highest * 1e4) / 1e4

    def goal(t: GyroTrial):
        now = t.top
        return (
            kept("Mass", t.mass, m, "kg"),
            kept("Radius", t.radius * 1000, r * 1000, "mm"),
            kept("Spin", t.spin_rpm, rpm, "rpm"),
            Requirement(
                f"It sleeps with 20 % to spare (needs ≤ {number(rpm / 1.2)} rpm)",
                f"needs {number(now.sleeping_spin / RPM)} rpm",
                now.sleeping_spin / RPM <= rpm / 1.2,
            ),
            Requirement(
                f"The disc at least {number(best * 1000 * 0.9)} mm up the stem",
                f"{number(t.height * 1000)} mm",
                t.height >= 0.9 * best,
            ),
        )

    return Question(
        f"A {m:g} kg disc of radius {r * 1000:g} mm will spin at {rpm:g} rpm. Set how far up its stem it sits so it "
        "sleeps with 20 % to spare in spin, as high as that allows (within 10 %).",
        0,
        f"It sleeps if I₃ω₃ &gt; 2√(I₁mgl), with I₁ = m(r²/4 + l²): raising the disc raises both I₁ and mgl, so it "
        f"needs more spin. With 20 % to spare, I₃ω₃/1.2 = 2√(I₁mgl) at about l = {number(best * 1000)} mm.",
        setup=dict(mass=m, radius=r, height=0.1, spin_rpm=rpm, tilt=2.0, tab=TAB_POTENTIAL),
        goal=goal,
        design=dict(mass=m, radius=r, height=best, spin_rpm=rpm, tilt=2.0, tab=TAB_POTENTIAL),
    )


JELLETT = "Jellett's constant: the rising top and the tippe top"


def _rising_end(rng: np.random.Generator) -> Question:
    l, R = pick(rng, (0.02, 0.025)), pick(rng, (0.01, 0.012))
    rpm, tilt = pick(rng, (1000, 1500, 2000)), pick(rng, (25, 30))
    a = l - R
    end = rpm * (a + R * math.cos(math.radians(tilt))) / (a + R)
    return Question(
        f"A disc spins at {rpm:g} rpm on a stem whose tip is rounded to a radius of {R * 1000:g} mm, the disc's centre "
        f"{l * 1000:g} mm above the end of the stem. Let go at {tilt:g}° on a slippery table, it rises. If it rose "
        "all the way upright, how fast would it spin?",
        end,
        f"Friction keeps Jellett's J = I₃ω₃(a + R cos θ), with a = l − R = {number(a * 1000)} mm the disc's centre "
        f"above the tip's. Upright cos θ = 1: ω₃ = {rpm:g} × ({number(a * 1000)} + {R * 1000:g} cos {tilt:g}°)/"
        f"({number(a * 1000)} + {R * 1000:g}) = {number(end)} rpm, a little slower.",
        unit="rpm",
        setup=dict(
            setting=Setting.TABLE,
            radius=0.03,
            height=l,
            tip_radius=R,
            friction=0.01,
            spin_rpm=rpm,
            tilt=tilt,
            playback=PLAYBACK.index(2.0),
            tab=TAB_POTENTIAL,
        ),
    )


def _tippe_end(rng: np.random.Generator) -> Question:
    R, low = pick(rng, (0.012, 0.015, 0.018)), pick(rng, (0.003, 0.004, 0.005))
    rpm, tilt = pick(rng, (800, 1000, 1200, 1500)), pick(rng, (5, 10))
    end = rpm * (R * math.cos(math.radians(tilt)) - low) / (R + low)
    return Question(
        f"A tippe top is a ball {R * 1000:g} mm in radius, its centre of mass {low * 1000:g} mm below the ball's "
        f"centre. Spun at {rpm:g} rpm on its bottom, {tilt:g}° off upright, it turns over. How fast does it spin on "
        "its top, seen from above?",
        end,
        f"Friction keeps Jellett's J = I₃ω₃(a + R cos θ), with a = −{low * 1000:g} mm (the weight below the centre). "
        f"Turned over J = I₃Ω(R − a), so Ω = {rpm:g} × ({R * 1000:g} cos {tilt:g}° − {low * 1000:g})/"
        f"({R * 1000:g} + {low * 1000:g}) = {number(end)} rpm. About its own axis it now spins the other way.",
        unit="rpm",
        setup=dict(
            setting=Setting.TIPPE,
            mass=0.02,
            radius=R,
            low=low,
            ratio=0.9,
            spin_rpm=rpm,
            tilt=tilt,
            playback=PLAYBACK.index(0.25),
            tab=TAB_POTENTIAL,
        ),
    )


def _tippe_turns(rng: np.random.Generator) -> Question:
    ratio = pick(rng, (0.6, 0.8, 1.0, 1.4))
    over, up, side = (
        "It turns over onto its top",
        "It stays upright",
        "It rolls onto its side and spins there",
    )
    if ratio < 0.7:
        choices, why = (up, over, side), "under 1 − α, so upright stays E<sub>J</sub>'s least: it stays up."
    elif ratio < 1.3:
        choices, why = (over, up, side), "between them, so turned over is E<sub>J</sub>'s least: it turns over."
    else:
        choices, why = (
            (side, over, up),
            "over 1 + α, so turned over is no minimum either: E<sub>J</sub> is least past its side, and it ends there.",
        )
    return Question(
        f"A tippe top's weight sits 0.3 of its radius below the ball's centre (α = 0.3), and I₁/I₃ = {ratio:g} about "
        "its centre of mass. Spun fast (1200 rpm) on its bottom on a table, what does it do?",
        0,
        f"1 − α = 0.7 and 1 + α = 1.3; I₁/I₃ = {ratio:g} is {why}",
        choices=choices,
        setup=dict(
            setting=Setting.TIPPE,
            mass=0.02,
            radius=0.015,
            low=0.0045,
            ratio=ratio,
            spin_rpm=1200.0,
            tilt=5.0,
            playback=PLAYBACK.index(0.25),
            tab=TAB_POTENTIAL,
        ),
        predict=True,
    )


PROBLEMS = tuple(
    Problem(key, title, "gyroscope_page", f"See {page_link('gyroscope_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("moments", "A disc on a stem", "rigid-body", "A rigid body turning", _moments),
        ("square-moments", "A square on a stem", "square", "A rigid body turning", _square_moments),
        ("square-swap", "Predict: a square for the disc", "square", "A rigid body turning", _square_swap),
        ("rectangle-moments", "A rectangle on a stem", "asymmetric", "The asymmetric top", _rectangle_moments),
        ("racket", "Predict: the tennis-racket flip", "asymmetric", "The asymmetric top", _racket),
        ("flat-precession", "A wheel on a pivot", "precession", "Steady precession", _flat_precession),
        ("steady-tilted", "Steady precession", "precession", "Steady precession", _steady_tilted),
        ("sleeping", "A sleeping top", "sleeping", "The sleeping top", _sleeping),
        ("nutation", "Nutation", "nutation", "Nutation", _nutation),
        ("faster-spin", "Predict: spin it faster", "precession", "Steady precession", _faster_spin),
        ("at-rest-path", "Predict: let go at rest", "nutation", "Nutation", _at_rest_path),
        ("pushed-back", "Predict: pushed back", "nutation", "Nutation", _pushed_back),
        ("too-slow", "Predict: too slow to sleep", "sleeping", "The sleeping top", _too_slow),
        ("reverse", "Predict: the other way", "precession", "Steady precession", _reverse),
        ("period-spec", "Spec: a steady precession", "precession", "Steady precession", _period_spec),
        ("sleep-spec", "Spec: a top that sleeps", "sleeping", "The sleeping top", _sleep_spec),
        ("rising-end", "Rising to sleep", "jellett", JELLETT, _rising_end),
        ("tippe-end", "A tippe top's spin", "jellett", JELLETT, _tippe_end),
        ("tippe-turns", "Predict: will it turn over?", "jellett", JELLETT, _tippe_turns),
    )
)

TIPS = {
    "mass": "<p>The disc's mass (the stem's is left out).</p>",
    "radius": "<p>The disc's radius: I₃ = mr²/2.</p>",
    "side": "<p>The square's side: I₃ = ms²/6, the same as a disc of radius s/√3.</p>",
    "shape": "<p>A disc, or a square plate: a quarter turn leaves the square as it was, so its moment is the same "
    "about every line across its axis, and it is a symmetric top too. A rectangle isn't (I₁ ≠ I₂): an asymmetric top, "
    "in the cup only.</p>",
    "breadth": "<p>The rectangle's other side b, across its mark: I₁ = m(b²/12 + l²), I₂ = m(a²/12 + l²), "
    "I₃ = m(a² + b²)/12.</p>",
    "side_a": "<p>The rectangle's side a, along its mark (the thick line on it).</p>",
    "height": "<p>From the tip to the disc's centre, l: gravity's arm, and I₁ = m(r²/4 + l²) about the tip.</p>",
    "gravity": "<p>g: 9.81 m/s² on Earth; 0 for a free top.</p>",
    "spin": "<p>ω₃, the spin about the axis (ψ' + φ' cos θ), which stays constant. Negative spins it the other "
    "way.</p>",
    "tilt": "<p>The axis's angle from the vertical when it is let go, its tilt not changing then (a turning "
    "point).</p>",
    "launch": "<p>How it is let go: not precessing (φ' = 0), at the slow or fast steady rate (where there is "
    "one), or at the rate set below.</p>",
    "precession": "<p>φ', the axis's rate round the vertical when it is let go (with <i>Set φ' below</i>).</p>",
    "playback": "<p>Simulated time per real second. The top spins far too fast to see: slow it down.</p>",
    "setting": "<p>The disc (or square) with its tip held in a cup (it can turn there but not slide), the same on a "
    "round tip that slides and rolls on a table, or a tippe top (a ball weighted low) on a table.</p>",
    "tip_radius": "<p>The round tip's radius R: its centre is R up the axis from the end of the stem, and the disc's "
    "centre a = l − R above that.</p>",
    "friction": "<p>μ, Coulomb's friction between the tip and the table: μN against the slip (smoothed below "
    "5 mm/s). 0.01 is a polished, oiled surface; 0.3 is wood.</p>",
    "low": "<p>How far the tippe top's centre of mass sits below the ball's centre, a: α = a/R.</p>",
    "ratio": "<p>I₁/I₃ about the centre of mass (I₃ = 0.4mR², a ball's). It turns over if 1 − α &lt; I₁/I₃ &lt; "
    "1 + α.</p>",
    "ball": "<p>The tippe top's ball's radius R (its stem left out).</p>",
}
VIEW_TIP = (
    "<p>The top on its tip, in 3D: drag to turn the view, double-click to reset it. The line through the disc "
    "is its axis and the tick on the disc a mark that turns with it; the thin line is where the end of the "
    "axis has been, between the dotted circles at the turning points. With <i>Vectors</i> ticked: the "
    "angular momentum L, the angular velocity ω and gravity's moment τ, each scaled to fit.</p>"
)
TILT_TIP = (
    "<p>The axis's tilt from the vertical as it runs, the last few nods. Dashed: the turning points the "
    "energy and the two momenta give, in closed form.</p>"
)
POTENTIAL_TIP = (
    "<p>The tilt moves in U(θ) = (p<sub>φ</sub> − p<sub>ψ</sub> cos θ)²/2I₁ sin²θ + mgl cos θ with energy "
    "E': it nods between where U = E' (dashed). The dot is the tilt now.</p>"
)
PRECESSION_TIP = (
    "<p>The two steady precession rates at this tilt against the spin (solid), and the fast top's estimates "
    "mgl/I₃ω₃ and I₃ω₃/I₁ cos θ (dashed). Under the least spin (dotted) the top can't precess steadily at this "
    "tilt. The dot is this spin's slow rate.</p>"
)
PATH_TIP = "<p>The end of the axis, seen from above, the axis's length 1. Dotted: the turning points' circles.</p>"
ENERGY_TIP = (
    "<p>E<sub>J</sub>(θ), the least energy the top can have at each tilt with its Jellett's J, which friction can't "
    "change (solid), and its energy against its tilt as it runs (thin), down to the dot, now. Friction takes energy "
    "out, so it sinks onto the curve, toward its lowest point.</p>"
)
TRACK_TIP = (
    "<p>The tip's track on the table (solid, mm) and its centre of mass's (dotted), seen from above, the last 10 s.</p>"
)
LEDGER_TIP = (
    "<p>The energy and the work friction has done, together (constant but for the step's error), and Jellett's J "
    "(constant whatever the contact force), each against what it was let go with.</p>"
)
SECTION_TIP = (
    "<p>A Poincaré section: each time the tilt peaks, a dot at the spin angle ψ (from the line of nodes, mod 180°) "
    "and the spin ω₃. A symmetric top keeps ω₃: a line. Regular motion draws closed curves; chaos, a scatter.</p>"
)
APART_TIP = (
    "<p>How far apart (rad, log scale) the axes of two tops let go 10⁻⁸ rad apart have got. Straight up: chaos.</p>"
)
TABLE_TILT_TIP = "<p>The axis's tilt from the vertical as it runs, the last minute.</p>"
CHECK_TIP = (
    "<p>How far the simulation's energy has strayed (over mgl), and its two momenta's (over I₃ω₃: round-off "
    "only). The text compares what it has run with the closed forms.</p>"
)

NOTES_HTML = load_notes("gyroscope", trials=trials_html(TRIALS))


class TopView(pg.PlotWidget):
    """The top in 3D, orthographically projected; drag to turn it."""

    YAW, PITCH = math.radians(25.0), math.radians(20.0)
    RIM = 48  # points round the disc

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(VIEW_TIP)
        self.setAspectLocked(True)
        for side in ("left", "bottom"):
            self.hideAxis(side)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.yaw, self.pitch = self.YAW, self.PITCH
        self._drag: QtCore.QPointF | None = None
        self.top = Top()
        self.round: RoundTop | None = None  # on a table
        self.ball = False
        self.reach = 1.0  # the drawn axis's length (m)
        self.turning: tuple[float, ...] = (0.0, 0.0)
        self.show_vectors = True
        self._last: tuple | None = None

        plot = self.getPlotItem()
        self.table = QtWidgets.QGraphicsPathItem()
        self.vertical = pg.PlotCurveItem()
        self.circles = pg.PlotCurveItem(connect="finite")  # the turning points' circles
        self.trail = pg.PlotCurveItem()
        self.track = pg.PlotCurveItem()  # the tip's, on the table
        self.stem = pg.PlotCurveItem()
        self.tip = pg.PlotCurveItem()  # a round tip's outline
        self.disc = QtWidgets.QGraphicsPathItem()
        self.rim = pg.PlotCurveItem()
        self.spoke = pg.PlotCurveItem()
        self.axis_line = pg.PlotCurveItem()
        self.vectors = [pg.PlotCurveItem() for _ in range(3)]  # L, ω, τ
        self.labels = [pg.TextItem(t, anchor=(0.5, 0.5)) for t in ("L", "ω", "τ")]
        self.end = Dot(7)
        for item in (
            self.table,
            self.vertical,
            self.circles,
            self.trail,
            self.track,
            self.stem,
            self.tip,
            self.disc,
            self.rim,
            self.spoke,
            self.axis_line,
            *self.vectors,
            *self.labels,
        ):
            plot.addItem(item)
        plot.addItem(self.end, ignoreBounds=True)
        self.apply_theme()

    def project(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Screen (x, y) of points p (..., 3), z up: turned by the yaw about z, looked at from the pitch above."""
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        across = x * cy - y * sy
        away = x * sy + y * cy
        return across, z * math.cos(self.pitch) + away * math.sin(self.pitch)

    def set_top(self, top: Top, turning: tuple[float, ...], round_top: RoundTop | None = None) -> None:
        """A new top or launch: its size sets the view's, and its turning points the circles (none on a table,
        where the view follows the top: its centre of mass stays over the middle)."""
        self.top, self.turning, self.round = top, turning, round_top
        self.ball = round_top is not None and round_top.rim == 0
        self.reach = 2.4 * top.radius if self.ball else 1.6 * max(top.height, top.corner)
        self.labels[2].setText("F" if round_top is not None else "τ")
        self._last = None
        self._place_static()

    def _place_static(self) -> None:
        R = self.reach
        ring = np.linspace(0, 2 * math.pi, 97)
        table = np.stack([1.1 * R * np.cos(ring), 1.1 * R * np.sin(ring), np.zeros_like(ring)], axis=1)
        sx, sy = self.project(table)
        path = QtGui.QPainterPath()
        path.moveTo(float(sx[0]), float(sy[0]))
        for a, b in zip(sx[1:], sy[1:]):
            path.lineTo(float(a), float(b))
        self.table.setPath(path)
        vx, vy = self.project(np.array([[0, 0, 0], [0, 0, 1.15 * R]]))
        self.vertical.setData(vx, vy)
        lines = []
        for th in self.turning:
            circle = np.stack(
                [
                    R * math.sin(th) * np.cos(ring),
                    R * math.sin(th) * np.sin(ring),
                    R * math.cos(th) * np.ones_like(ring),
                ],
                axis=1,
            )
            lines.append(circle)
            lines.append(np.full((1, 3), np.nan))
        if lines:
            self.circles.setData(*self.project(np.concatenate(lines)))
        else:
            self.circles.setData([], [])
        span = 1.25 * R
        self.setRange(xRange=(-span, span), yRange=(-0.6 * span, 1.2 * span), padding=0)
        if self._last is not None:
            self.draw(*self._last)

    def _anchor(self, axis: np.ndarray, centre: np.ndarray) -> np.ndarray:
        """Where the drawn axis starts, for axes (..., 3) and centres of mass G: the stem's end (the tip), or a
        ball's centre."""
        rt = self.round
        if rt is None:
            return np.zeros_like(axis)
        hub = centre - rt.offset * axis  # the round tip's centre
        return hub if self.ball else hub - rt.radius * axis

    def _outline(self, item: QtWidgets.QGraphicsPathItem, x: np.ndarray, y: np.ndarray) -> None:
        path = QtGui.QPainterPath()
        path.moveTo(float(x[0]), float(y[0]))
        for a, b in zip(x[1:], y[1:]):
            path.lineTo(float(a), float(b))
        item.setPath(path)

    def draw(
        self,
        axis: np.ndarray,
        mark: np.ndarray,
        momentum: np.ndarray,
        trail: np.ndarray,
        centre: np.ndarray | None = None,
        trail_centres: np.ndarray | None = None,
        track: np.ndarray | None = None,
        force: np.ndarray | None = None,
    ) -> None:
        """The top with its axis along `axis` (a unit vector), the disc's mark toward `mark`, and L; `trail` is
        where the axis has pointed (n, 3). On a table: G at `centre` (drawn over the middle), G when the axis
        pointed along `trail` (n, 3; its height), where the tip has touched the table (`track`, (m, 2)) and the
        friction."""
        self._last = (axis, mark, momentum, trail, centre, trail_centres, track, force)
        top, R = self.top, self.reach
        here = np.zeros(3) if centre is None else np.array([centre[0], centre[1], 0.0])
        G = top.height * axis if centre is None else centre - here
        anchor = self._anchor(axis, G)
        ring = np.linspace(0, 2 * math.pi, self.RIM + 1)[:, None]
        side = np.cross(axis, mark)
        if self.ball:
            rt = self.round
            assert rt is not None  # a ball is only ever on a table
            # its outline is a circle on the screen whatever the view; its equator turns with it
            cx, cy = self.project(anchor)
            self._outline(self.disc, cx + rt.radius * np.cos(ring[:, 0]), cy + rt.radius * np.sin(ring[:, 0]))
            rim = anchor + rt.radius * (np.cos(ring) * mark + np.sin(ring) * side)
            spoke = np.array([anchor, anchor + rt.radius * mark])
            self.stem.setData([], [])
            self.tip.setData([], [])
            start = anchor
        else:
            if top.shape is Shape.DISC:
                rim = G + top.radius * (np.cos(ring) * mark + np.sin(ring) * side)
            else:  # its corners, half a side along the mark and half the other across it
                across = top.breadth if top.shape is Shape.RECTANGLE else top.radius
                k = 0.5 * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]]) * [top.radius, across]
                rim = G + k[:, :1] * mark + k[:, 1:] * side
            self._outline(self.disc, *self.project(rim))
            spoke = np.array([G, G + (1.0 if top.shape is Shape.DISC else 0.5) * top.radius * mark])
            if self.round is None:
                self.tip.setData([], [])
            else:
                hub = anchor + self.round.radius * axis
                hx, hy = self.project(hub)
                self.tip.setData(
                    hx + self.round.radius * np.cos(ring[:, 0]), hy + self.round.radius * np.sin(ring[:, 0])
                )
            tx, ty = self.project(np.array([anchor, G]))
            self.stem.setData(tx, ty)
            start = G
        rx, ry = self.project(rim)
        self.rim.setData(rx, ry)
        sx, sy = self.project(spoke)
        self.spoke.setData(sx, sy)
        end = anchor + R * axis
        ax, ay = self.project(np.array([start, end]))
        self.axis_line.setData(ax, ay)
        ex, ey = self.project(end)
        self.end.setData([ex], [ey])
        if trail.size:
            ends = R * trail
            if trail_centres is not None:  # each over the middle, as the top is drawn: where it pointed, not went
                ends = ends + self._anchor(trail, trail_centres * [0.0, 0.0, 1.0])
            px, py = self.project(ends)
            self.trail.setData(px, py)
        else:
            self.trail.setData([], [])
        if track is not None and track.size:
            flat = np.column_stack([track[:, 0] - here[0], track[:, 1] - here[1], np.zeros(len(track))])
            self.track.setData(*self.project(flat))
        else:
            self.track.setData([], [])
        self._draw_vectors(axis, mark, momentum, np.zeros(3) if centre is None else G, force)

    def _draw_vectors(
        self, axis: np.ndarray, mark: np.ndarray, momentum: np.ndarray, origin: np.ndarray, force: np.ndarray | None
    ) -> None:
        top, R = self.top, self.reach
        size = float(np.linalg.norm(momentum))
        if not self.show_vectors or size == 0:
            for item, label in zip(self.vectors, self.labels):
                item.setData([], [])
                label.hide()
            return
        along = float(momentum @ axis)
        rt = self.round
        i1, i3 = (top.transverse, top.polar) if rt is None else (rt.central, rt.polar)
        omega = momentum / i1 + (1 / i3 - 1 / i1) * along * axis
        if rt is None and not top.symmetric:  # a rectangle's own y axis has I₂
            side = np.cross(axis, mark)
            omega = omega + (1 / top.second - 1 / i1) * float(momentum @ side) * side
        if rt is None:
            third = -top.weight_moment * np.cross(axis, [0.0, 0.0, 1.0])
        else:
            third = np.zeros(3) if force is None else np.array([force[0], force[1], 0.0])
        for item, label, v, length in zip(
            self.vectors, self.labels, (momentum, omega, third), (1.25 * R, 1.05 * R, 0.6 * R)
        ):
            n = float(np.linalg.norm(v))
            if n < 1e-12:
                item.setData([], [])
                label.hide()
                continue
            tip = origin + length * v / n
            x, y = self.project(np.array([origin, tip]))
            item.setData(x, y)
            lx, ly = self.project(origin + 1.1 * (tip - origin))
            label.setPos(float(lx), float(ly))
            label.show()

    def apply_theme(self) -> None:
        c = colors
        grid = QtGui.QColor(c.grey)
        grid.setAlpha(40)
        self.table.setBrush(QtGui.QBrush(grid))
        self.table.setPen(pg.mkPen(c.grey, width=1))
        self.vertical.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.circles.setPen(pg.mkPen(c.fit, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.trail.setPen(pg.mkPen(c.mass[1], width=1.5))
        self.track.setPen(pg.mkPen(c.mass[2], width=1.5))
        self.stem.setPen(pg.mkPen(c.strong, width=3))
        self.tip.setPen(pg.mkPen(c.strong, width=2))
        fill = QtGui.QColor(c.mass[0])
        fill.setAlpha(110)
        self.disc.setBrush(QtGui.QBrush(fill))
        self.disc.setPen(pg.mkPen(None))
        self.rim.setPen(pg.mkPen(c.mass[0], width=2))
        self.spoke.setPen(pg.mkPen(c.strong, width=3))
        self.axis_line.setPen(pg.mkPen(c.strong, width=1.5))
        self.end.setBrush(pg.mkBrush(c.mass[1]))
        self.end.setPen(pg.mkPen(None))
        for item, label, color in zip(self.vectors, self.labels, (c.force, c.mass[2], c.mass[3])):
            item.setPen(pg.mkPen(color, width=2.5))
            label.setColor(color)

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag = event.position()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag is None:
            super().mouseMoveEvent(event)
            return
        d = event.position() - self._drag
        self._drag = event.position()
        self.yaw = (self.yaw - 0.01 * d.x()) % (2 * math.pi)
        self.pitch = float(np.clip(self.pitch + 0.01 * d.y(), -math.pi / 2, math.pi / 2))
        self._place_static()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.yaw, self.pitch = self.YAW, self.PITCH
        self._place_static()
        event.accept()


def _thin(n: int, most: int) -> slice:
    return slice(None, None, max(1, -(-n // most)))


def _section(axis: np.ndarray, mark: np.ndarray, momentum: np.ndarray, top: Top) -> tuple[np.ndarray, np.ndarray]:
    """The Poincaré section's coordinates where the tilt peaks: the spin angle ψ (°, mod 180, a rectangle's
    period), the mark's angle from the line of nodes n = ẑ × e₃, and ω₃ (rpm). With L·ẑ and the energy kept and φ
    left out, that is all the state there is: θ' = 0 there, and θ follows from the energy."""
    nodes = np.stack([-axis[:, 1], axis[:, 0], np.zeros(len(axis))], axis=1)
    nodes /= np.maximum(np.linalg.norm(nodes, axis=1), 1e-300)[:, None]
    psi = np.degrees(np.arctan2(np.sum(mark * np.cross(axis, nodes), 1), np.sum(mark * nodes, 1))) % 180.0
    return psi, np.sum(momentum * axis, 1) / top.polar / RPM


def _columns(s: SlideSamples) -> np.ndarray:
    """A table's samples as the history keeps them: `SlideSamples.of`'s columns after t."""
    return np.column_stack(
        [s.axis, s.mark, s.centre, s.contact, s.energy, s.work, s.jellett, s.normal, s.friction, s.momentum, s.slip]
    )


class GyroscopePage(QtWidgets.QWidget):
    """Controls on the left, the top and its tilt in the centre, the closed forms and the check on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.motion = TRIALS[0].motion
        self.round: RoundTop | None = None  # on a table
        self.sim: TopSimulator | SlidingSimulator = TopSimulator(self.motion)
        self.history = SampleBuffer(10, HISTORY)  # axis, mark, energy, L
        self.slides = SampleBuffer(COLUMNS - 1, 1)  # on a table: a sample's columns after t (made full size there)
        self._track_span: tuple[float, float, float] | None = None  # the Path tab's view on a table: x, y, half
        self._launch_energy = self._launch_jellett = 0.0  # on a table, let go
        self.twin: TopSimulator | None = None  # in the cup: the same top let go TWIN further over
        self.section = SampleBuffer(2, SECTION_POINTS)  # where the tilt peaks: ψ (°, mod 180) and ω₃ (rpm)
        self.apart = SampleBuffer(1, HISTORY // APART_STRIDE)  # how far apart the two tops' axes are
        self._peaks = (math.nan, math.nan)  # the last two samples' e₃·ẑ, to find a peak across two frames
        self.running = False  # held at the launch until started
        self._frames = 0

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Each sets up the top and lets it go, to show one idea from the Theory tab.</p>",
        )
        self.trials.picked.connect(self.load_trial)

        body = Form("Top")
        self.body_form = body.form
        self.setting = combo([setting.value for setting in SETTINGS], TIPS["setting"])
        body.add("Setting:", self.setting, TIPS["setting"])
        self.shape = combo([shape.value for shape in SHAPES], TIPS["shape"])
        body.add("Shape:", self.shape, TIPS["shape"])
        self.mass = body.number("Mass m:", 0.001, 100.0, 0.5, 3, " kg", TIPS["mass"])
        self.radius = body.number("Disc radius r:", 1.0, 500.0, 40.0, 2, " mm", TIPS["radius"])
        self.breadth = body.number("Breadth b:", 1.0, 500.0, 40.0, 2, " mm", TIPS["breadth"])
        self.height = body.number("Height l:", 1.0, 500.0, 30.0, 1, " mm", TIPS["height"])
        self.tip_radius = body.number("Tip radius R:", 0.5, 100.0, 10.0, 1, " mm", TIPS["tip_radius"])
        self.low = body.number("Weight below centre:", 0.0, 100.0, 4.5, 2, " mm", TIPS["low"])
        self.ratio = body.number("I₁/I₃:", 0.1, 3.0, 0.8, 2, "", TIPS["ratio"])
        self.friction = body.number("Friction μ:", 0.0, 2.0, 0.3, 3, "", TIPS["friction"])
        self.gravity = body.number("Gravity g:", 0.0, 100.0, 9.81, 2, " m/s²", TIPS["gravity"])
        self.info = note_label()
        body.row(self.info)

        release = Form("Let go")
        self.release_form = release.form
        self.spin = release.number("Spin ω₃:", -20000.0, 20000.0, 1500.0, 0, " rpm", TIPS["spin"])
        self.tilt = release.number("Tilt θ₀:", 0.1, 179.0, 30.0, 1, "°", TIPS["tilt"])
        self.launch = combo([launch.value for launch in LAUNCHES], TIPS["launch"])
        release.add("Precession:", self.launch, TIPS["launch"])
        self.rate = release.number("φ'₀:", -500.0, 500.0, 0.0, 2, " rad/s", TIPS["precession"])
        self.launch_info = note_label()
        release.row(self.launch_info)

        play = Form("Run")
        self.playback = combo([f"{p:g}×" for p in PLAYBACK], TIPS["playback"])
        self.playback.setCurrentIndex(REAL_TIME)
        play.add("Playback:", self.playback, TIPS["playback"])
        self.pause_button = RunButton()
        self.pause_button.toggled.connect(self._on_pause)
        self.restart_button = QtWidgets.QPushButton("Reset")
        self.restart_button.setToolTip("<p>Back to the launch, held until started.</p>")
        self.restart_button.clicked.connect(self.reset)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.restart_button)
        play.row(buttons)
        self.vectors = QtWidgets.QCheckBox("Vectors L, ω, τ")
        self.vectors.setChecked(True)
        self.vectors.toggled.connect(self._on_vectors)
        play.row(self.vectors)

        connect_all(
            (
                self.setting,
                self.shape,
                self.mass,
                self.radius,
                self.breadth,
                self.height,
                self.tip_radius,
                self.low,
                self.ratio,
                self.friction,
                self.gravity,
                self.spin,
                self.tilt,
                self.launch,
                self.rate,
            ),
            self._on_inputs,
        )

        # --- centre
        self.view = TopView()
        self.tilt_plot = plot("Tilt θ", None, "Time", "s", height=130)
        self.tilt_plot.setToolTip(TILT_TIP)
        self.tilt_item = pg.PlotDataItem()
        self.turn_lines = [pg.InfiniteLine(angle=0), pg.InfiniteLine(angle=0)]
        for item in (*self.turn_lines, self.tilt_item):
            self.tilt_plot.addItem(item)
        self.tilt_plot.setLabel("left", "Tilt θ (°)")
        self.readout = readout_label()

        # --- right
        self.potential_plot = plot("U", "J", "Tilt θ", None, height=200)
        self.potential_plot.setToolTip(POTENTIAL_TIP)
        self.potential_item = pg.PlotDataItem()
        self.energy_line = pg.InfiniteLine(angle=0)
        self.potential_turns = [pg.InfiniteLine(angle=90), pg.InfiniteLine(angle=90)]
        self.potential_now = Dot(10)
        self.potential_trace = pg.PlotDataItem()  # on a table: the energy against the tilt as it runs
        for item in (*self.potential_turns, self.energy_line, self.potential_trace, self.potential_item):
            self.potential_plot.addItem(item)
        self.potential_plot.addItem(self.potential_now, ignoreBounds=True)
        self.potential_plot.setLabel("bottom", "Tilt θ (°)")
        self.potential_note = note_label()

        self.precession_plot = plot("Precession rate", "rad/s", "Spin", "rpm", log_y=True, height=200)
        self.precession_plot.setToolTip(PRECESSION_TIP)
        add_legend(self.precession_plot.getPlotItem(), offset=(-10, 10))
        self.slow_item = pg.PlotDataItem(name="slow")
        self.fast_item = pg.PlotDataItem(name="fast")
        self.slow_guess = pg.PlotDataItem(name="mgl/I₃ω₃")
        self.fast_guess = pg.PlotDataItem(name="I₃ω₃/I₁ cos θ")
        self.least_line = pg.InfiniteLine(angle=90)
        self.precession_now = marker(11)
        for item in (self.least_line, self.slow_guess, self.fast_guess, self.slow_item, self.fast_item):
            self.precession_plot.addItem(item)
        self.precession_plot.addItem(self.precession_now)
        self.precession_note = note_label()

        self.path_plot = plot("", None, "", None, height=260)
        self.path_plot.setToolTip(PATH_TIP)
        self.path_plot.setAspectLocked(True)
        self.path_circles = pg.PlotDataItem(connect="finite")
        self.path_item = pg.PlotDataItem()
        self.path_now = Dot(8)
        for item in (self.path_circles, self.path_item):
            self.path_plot.addItem(item)
        self.path_plot.addItem(self.path_now, ignoreBounds=True)
        self.path_note = note_label()

        self.check_plot = plot("Error", None, "Time", "s", height=200)
        self.check_plot.setToolTip(CHECK_TIP)
        add_legend(self.check_plot.getPlotItem(), offset=(10, 10))
        self.energy_item = pg.PlotDataItem(name="energy, ΔE/mgl")
        self.momentum_item = pg.PlotDataItem(name="momenta, ΔL/I₃ω₃")
        for item in (self.energy_item, self.momentum_item):
            self.check_plot.addItem(item)
        self.check_legend = self.check_plot.getPlotItem().legend
        self.check_note = note_label()

        self.section_plot = plot("ω₃", "rpm", "ψ at each peak of the tilt", None, height=220)
        self.section_plot.setToolTip(SECTION_TIP)
        self.section_plot.setLabel("bottom", "ψ at each peak of the tilt (°, mod 180)")
        self.section_plot.getAxis("left").enableAutoSIPrefix(False)
        self.section_item = marker(3)
        self.section_plot.addItem(self.section_item)
        self.section_plot.setXRange(0, 180, padding=0)
        self.apart_plot = plot("Apart", None, "Time", "s", log_y=True, height=150)
        self.apart_plot.setToolTip(APART_TIP)
        self.apart_item = pg.PlotDataItem()
        self.apart_plot.addItem(self.apart_item)
        self.chaos_note = note_label()

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(column(self.potential_plot, self.potential_note, stretch=False), "Potential")
        self.tabs.addTab(column(self.precession_plot, self.precession_note, stretch=False), "Precession")
        self.tabs.addTab(column(self.path_plot, self.path_note, stretch=False), "Path")
        self.tabs.addTab(column(self.check_plot, self.check_note, stretch=False), "Check")
        self.tabs.addTab(column(self.section_plot, self.apart_plot, self.chaos_note, stretch=False), "Chaos")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(lambda _: self._draw_tab())

        three_panes(
            self,
            column(self.trials, body, release, play),
            stack((self.view, 4), (self.tilt_plot, 2), (self.readout, 0)),
            self.tabs,
        )

        self._clock = QtCore.QElapsedTimer()
        self._last = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._frame)
        self.tabs.setCurrentIndex(TAB_PATH)
        self._on_inputs()
        self.apply_theme()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(GyroTrial("", "", **setup))

    def practice_state(self) -> GyroTrial:
        """The page's settings, as a trial: what a practice spec checks."""
        return GyroTrial(
            "",
            "",
            mass=self.mass.value(),
            radius=self.radius.value() / 1000,
            height=self.height.value() / 1000,
            gravity=self.gravity.value(),
            setting=SETTINGS[self.setting.currentIndex()],
            shape=SHAPES[self.shape.currentIndex()],
            breadth=self.breadth.value() / 1000,
            tip_radius=self.tip_radius.value() / 1000,
            friction=self.friction.value(),
            low=self.low.value() / 1000,
            ratio=self.ratio.value(),
            spin_rpm=self.spin.value(),
            tilt=self.tilt.value(),
            launch=LAUNCHES[self.launch.currentIndex()],
            precession=self.rate.value(),
            playback=self.playback.currentIndex(),
            tab=self.tabs.currentIndex(),
        )

    def apply_trial(self, t: GyroTrial) -> None:
        for box, value in (
            (self.setting, SETTINGS.index(t.setting)),
            (self.shape, SHAPES.index(t.shape)),
            (self.breadth, t.breadth * 1000),
            (self.tip_radius, t.tip_radius * 1000),
            (self.friction, t.friction),
            (self.low, t.low * 1000),
            (self.ratio, t.ratio),
            (self.mass, t.mass),
            (self.radius, t.radius * 1000),
            (self.height, t.height * 1000),
            (self.gravity, t.gravity),
            (self.spin, t.spin_rpm),
            (self.tilt, t.tilt),
            (self.launch, LAUNCHES.index(t.launch)),
            (self.rate, t.precession),
            (self.playback, t.playback),
        ):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self.running = True  # a trial lets the top go
        self.pause_button.set_running(True)
        self._on_inputs()

    def _on_inputs(self) -> None:
        state = self.practice_state()
        if state.setting is not Setting.CUP and state.shape is Shape.RECTANGLE:  # a symmetric top only, on a table
            set_quietly(self.shape, SHAPES.index(Shape.DISC))
            state = self.practice_state()
        self._show_rows(state.setting, state.shape)
        self.rate.setEnabled(state.launch is Launch.SET)
        self.motion = state.motion
        self.round = state.round_top
        if self.round is not None:
            self._on_table(state)
            self.restart()
            return
        top = self.motion.top
        if not top.symmetric:
            self._on_skewed(state)
            self.restart()
            return
        self.info.setText(
            (
                f"I₁ = {top.transverse * 1e4:.4g} kg·cm², I₃ = {top.polar * 1e4:.4g} kg·cm², mgl = "
                f"{fmt(top.weight_moment, 'N·m')}. It sleeps above {top.sleeping_spin / RPM:.4g} rpm."
                if top.weight_moment > 0
                else f"I₁ = {top.transverse * 1e4:.4g} kg·cm², I₃ = {top.polar * 1e4:.4g} kg·cm²; no gravity: a free top."
            )
            + (
                f" A disc of radius s/√3 = {top.radius * 1000 / math.sqrt(3):.4g} mm would move the same."
                if top.shape is Shape.SQUARE
                else ""
            )
        )
        slow, fast = steady_precession(top, self.motion.spin, self.motion.tilt)
        if state.launch in (Launch.SLOW, Launch.FAST) and not math.isfinite(
            slow if state.launch is Launch.SLOW else fast
        ):
            least = least_spin(top, self.motion.tilt) / RPM
            self.launch_info.setText(
                f"No such steady precession at this tilt under {least:.4g} rpm: let go at rest instead."
            )
        else:
            rates = ", ".join(f"{r:.4g}" for r in (slow, fast) if math.isfinite(r))
            self.launch_info.setText(
                f"φ'₀ = {self.motion.precession:.4g} rad/s. Steady rates at this tilt: "
                + (f"{rates} rad/s." if rates else "none.")
            )
        self.restart()

    def _show_rows(self, setting: Setting, shape: Shape) -> None:
        """Show the controls this setting uses."""
        cup, ball = setting is Setting.CUP, setting is Setting.TIPPE
        for box, shown in (
            (self.height, not ball),
            (self.shape, not ball),
            (self.tip_radius, setting is Setting.TABLE),
            (self.low, ball),
            (self.ratio, ball),
            (self.friction, not cup),
        ):
            self.body_form.setRowVisible(box, shown)
        for box in (self.launch, self.rate, self.launch_info):
            self.release_form.setRowVisible(box, cup)
        rectangle = shape is Shape.RECTANGLE and cup
        self.body_form.setRowVisible(self.breadth, rectangle)
        self.shape.model().item(SHAPES.index(Shape.RECTANGLE)).setEnabled(cup)
        label = self.body_form.labelForField(self.radius)
        square = shape is Shape.SQUARE and not ball
        label.setText(
            "Ball radius R:" if ball else "Side s:" if square else "Side a:" if rectangle else "Disc radius r:"
        )
        self.radius.setToolTip(
            TIPS["ball"] if ball else TIPS["side"] if square else TIPS["side_a"] if rectangle else TIPS["radius"]
        )
        label.setToolTip(self.radius.toolTip())
        self.vectors.setText("Vectors L, ω, τ" if cup else "Vectors L, ω, F")
        skewed = rectangle and self.breadth.value() != self.radius.value()
        for tab, on in (
            (TAB_POTENTIAL, not skewed),
            (TAB_PRECESSION, cup and not skewed),
            (TAB_CHAOS, cup),
        ):
            self.tabs.setTabEnabled(tab, on)
        if not self.tabs.isTabEnabled(self.tabs.currentIndex()):
            self.tabs.setCurrentIndex(TAB_PATH)
        for plot_, cup_tip, table_tip in (
            (self.potential_plot, POTENTIAL_TIP, ENERGY_TIP),
            (self.path_plot, PATH_TIP, TRACK_TIP),
            (self.check_plot, CHECK_TIP, LEDGER_TIP),
            (self.tilt_plot, TILT_TIP, TABLE_TILT_TIP),
        ):
            plot_.setToolTip(cup_tip if cup else table_tip)
        relabel(self.potential_plot, "left", "U" if cup else "E", "J")
        self.check_legend.clear()
        self.check_legend.addItem(self.energy_item, "energy, ΔE/mgl" if cup else "energy + friction's work, ΔE/E₀")
        self.check_legend.addItem(
            self.momentum_item,
            "Jellett's J, ΔJ/J₀" if not cup else "L·ẑ, ΔL/I₃ω₃" if skewed else "momenta, ΔL/I₃ω₃",
        )

    def _on_skewed(self, state: GyroTrial) -> None:
        """The info lines for an asymmetric top."""
        top = self.motion.top
        i1, i2, i3 = (v * 1e4 for v in top.moments)
        self.info.setText(
            f"I₁ = {i1:.4g}, I₂ = {i2:.4g}, I₃ = {i3:.4g} kg·cm²"
            + (f", mgl = {fmt(top.weight_moment, 'N·m')}." if top.weight_moment > 0 else "; no gravity: a free top.")
        )
        self.launch_info.setText(
            f"φ'₀ = {self.motion.precession:.4g} rad/s. An asymmetric top has no steady precession to start at"
            + (": let go at rest." if state.launch in (Launch.SLOW, Launch.FAST) else ".")
        )

    def _on_table(self, state: GyroTrial) -> None:
        """The info lines for a top on a table."""
        rt = self.round
        assert rt is not None  # only called on a table
        self.info.setText(
            f"I₁ = {rt.central * 1e4:.4g} kg·cm² (about G), I₃ = {rt.polar * 1e4:.4g} kg·cm²; its centre of mass "
            + (
                f"{rt.offset * 1000:.3g} mm above the round tip's centre."
                if rt.offset >= 0
                else f"{-rt.offset * 1000:.3g} mm below the ball's centre: α = {rt.alpha:.3g}."
            )
        )

    def restart(self) -> None:
        """Let the top go again from the launch set (if it is running; held there if not)."""
        self._track_span = None
        if self.round is not None:
            self.sim = SlidingSimulator(self.round, self.motion.spin, self.motion.tilt)
            if self.slides.t.size < SLIDE_HISTORY:  # first time on a table: its 34 MB only once it's needed
                self.slides = SampleBuffer(COLUMNS - 1, SLIDE_HISTORY)
            self.slides.clear()
            first = self.sim.sample()
            self.slides.extend(first.t, _columns(first))
            self._launch_energy, self._launch_jellett = float(first.energy[0]), float(first.jellett[0])
            self.view.set_top(self.motion.top, (), self.round)
            for line in self.turn_lines:
                line.hide()
            self._show_table_forms()
            self._draw_tab()
            self._draw_now()
            self._update_buttons()
            self._start_timer()
            return
        m = self.motion
        self.sim = TopSimulator(m)
        self.twin = TopSimulator(Motion(m.top, m.spin, m.tilt + TWIN, m.precession))
        self.history.clear()
        self.section.clear()
        self.apart.clear()
        self._peaks = (math.nan, math.nan)
        skewed = not m.top.symmetric
        self.view.set_top(m.top, () if skewed else m.turning())
        for line, th in zip(self.turn_lines, m.turning()):
            line.setPos(math.degrees(th))
            line.setVisible(not skewed)
        if skewed:
            self._show_skewed_forms()
        else:
            self._show_closed_forms()
        self._draw_tab()
        self._draw_now()
        self._update_buttons()
        self._start_timer()

    def _on_vectors(self, on: bool) -> None:
        self.view.show_vectors = on
        self._draw_now()

    def _on_pause(self, on: bool) -> None:
        self.running = on
        self._start_timer()

    def _update_buttons(self) -> None:
        self.pause_button.set_running(self.running)

    def reset(self) -> None:
        """Back to the launch, held there until started."""
        self.running = False
        self.pause_button.rest()
        self.restart()

    # ------------------------------------------------------------- the loop
    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._start_timer()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()

    def _start_timer(self) -> None:
        if self.isVisible() and self.running and not self._timer.isActive():
            self._clock.restart()
            self._last = 0.0
            self._timer.start(FRAME_MS)

    def _frame(self) -> None:
        if not self.running:  # paused or reset since this frame was queued
            return
        began = self._clock.elapsed()
        now = began / 1000.0
        dt = min(now - self._last, 0.1)
        self._last = now
        with frame_stats.frame("Gyroscope", lambda: self.sim.t, FRAME_MS):
            self.step(dt * PLAYBACK[self.playback.currentIndex()])
            settle_plots(self)
        if self.isVisible() and self.running:  # the next frame FRAME_MS after this one began, not after it ended
            self._timer.start(max(1, FRAME_MS - (self._clock.elapsed() - began)))

    def frame_context(self) -> str:
        launch = self.launch.currentText() if self.round is None else self.setting.currentText()
        return f"{self.spin.value():g} rpm, {self.tilt.value():g}°, {launch}"

    def step(self, duration: float) -> None:
        """Run the top on by `duration` s of simulated time and redraw."""
        if isinstance(self.sim, SlidingSimulator):
            stopped = self.sim.event is not None
            slide = self.sim.advance(duration)
            if slide.t.size:
                self.slides.extend(slide.t, _columns(slide))
            if self.sim.event is not None and not stopped:
                self._show_table_forms()
        else:
            s = self.sim.advance(duration)
            if s.t.size:
                self.history.extend(s.t, np.concatenate([s.axis, s.mark, s.energy[:, None], s.momentum], axis=1))
                self._record_chaos(s)
        self._draw_now()
        self._draw_live_tab()

    def _record_chaos(self, s) -> None:
        """The Chaos tab's records: the twin's separation, and the Poincaré section's points (where e₃·ẑ is least:
        the tilt peaks, θ' = 0), each kept whether or not the tab shows."""
        assert self.twin is not None  # set with the cup's simulator
        other = self.twin.advance(float(s.t[-1]) - self.twin.t)
        n = min(len(other.t), len(s.t))
        k = slice(0, n, APART_STRIDE)
        self.apart.extend(s.t[k], np.linalg.norm(s.axis[k] - other.axis[k], axis=1)[:, None])
        ez = np.concatenate([self._peaks, s.axis[:, 2]])
        self._peaks = (float(ez[-2]), float(ez[-1]))
        j = np.flatnonzero((ez[1:-1] < ez[:-2]) & (ez[1:-1] <= ez[2:])) + 1 - 2  # in this chunk
        j = j[j >= 0]
        if j.size:
            psi, spin = _section(s.axis[j], s.mark[j], s.momentum[j], self.motion.top)
            self.section.extend(s.t[j], np.column_stack([psi, spin]))

    def table_samples(self, seconds: float | None = None) -> SlideSamples:
        """The samples kept on a table: the last `seconds` of them, or all."""
        n = self.slides.size
        if seconds is None:
            t, y = self.slides.t[:n], self.slides.y[:n]
        else:
            t, y = self.slides.window(seconds)
        return SlideSamples.of(np.column_stack([t, y]))

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.view.apply_theme()
        self.tilt_item.setPen(pg.mkPen(c.mass[0], width=2))
        for line in (*self.turn_lines, *self.potential_turns):
            line.setPen(pg.mkPen(c.fit, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.potential_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.potential_trace.setPen(pg.mkPen(c.mass[1], width=1))
        self.energy_line.setPen(pg.mkPen(c.force, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.potential_now.setBrush(pg.mkBrush(c.strong))
        self.potential_now.setPen(pg.mkPen(None))
        self.slow_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.fast_item.setPen(pg.mkPen(c.mass[1], width=2))
        self.slow_guess.setPen(pg.mkPen(c.mass[0], width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.fast_guess.setPen(pg.mkPen(c.mass[1], width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.least_line.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DotLine))
        mark(self.precession_now, c.strong, filled=True)
        self.path_circles.setPen(pg.mkPen(c.fit, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.path_item.setPen(pg.mkPen(c.mass[1], width=1.5))
        self.path_now.setBrush(pg.mkBrush(c.strong))
        self.path_now.setPen(pg.mkPen(None))
        self.energy_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.momentum_item.setPen(pg.mkPen(c.grey, width=1.5))
        mark(self.section_item, c.mass[0], filled=True)
        self.apart_item.setPen(pg.mkPen(c.mass[1], width=1.5))
        self._draw_tab()
        self._draw_now()

    def _show_closed_forms(self) -> None:
        m = self.motion
        hi, lo = (math.degrees(v) for v in m.turning())
        mean = m.mean_precession()
        if abs(mean) > 1e-9:
            prec = f"precesses at {mean:.4g} rad/s on average, a turn in {2 * math.pi / abs(mean):.4g} s"
        else:
            prec = "doesn't precess on average"
        if m.steady:
            nod = f"It holds at {hi:.4g}°: steady precession at {mean:.4g} rad/s."
        else:
            nod = (
                f"It nods between {hi:.4g}° and {lo:.4g}°, {m.nutation_hz():.4g} times a second, and {prec}. "
                f"Path: {m.path().value}."
            )
        self.readout.setText(nod)

    def _launch_rates(self) -> np.ndarray:
        """ω in the body's axes when let go: φ'₀ about the vertical, which leans θ₀ from the axis about the body's
        x axis, and ω₃ about the axis."""
        m = self.motion
        return np.array([0.0, m.precession * math.sin(m.tilt), m.spin])

    def _show_skewed_forms(self) -> None:
        """The readout for an asymmetric top: what's lost, and what's left in closed form."""
        m = self.motion
        top = m.top
        i1, i2, i3 = top.moments
        middle = min(i1, i2) < i3 < max(i1, i2)
        parts = [
            "I₁ ≠ I₂: an asymmetric top. ω₃ isn't kept, so there's no effective potential for the tilt, no turning "
            "points and no nutation period in closed form: only the energy and L·ẑ are kept."
        ]
        if top.weight_moment == 0:
            free = FreeSpin.of(top, self._launch_rates())
            parts.append(
                "Without gravity Euler–Poinsot's solution holds: ω in the body goes as Jacobi's cn, sn and dn "
                + (f"(k² = {free.k2:.4g})" if free.k2 < 0.999 else f"(1 − k² = {1 - free.k2:.2g})")
                + f", back every {free.period:.4g} s"
                + (f"; spun about its middle axis, it flips over every {0.5 * free.period:.4g} s." if middle else ".")
            )
        elif middle:
            parts.append("Spun about its middle axis, with gravity, it tumbles: chaos (the Chaos tab).")
        else:
            parts.append("With gravity no closed form is left at all: the Chaos tab shows whether it stays regular.")
        self.readout.setText(" ".join(parts))

    def _show_table_forms(self) -> None:
        """The readout on a table: Jellett's constant and where it takes the top; why it stopped, if it did."""
        rt = self.round
        assert rt is not None  # only on a table
        m = self.motion
        J = rt.jellett(m.spin, m.tilt)
        least = math.degrees(rt.least_tilt(J))
        parts = [
            f"Jellett's J = I₃ω₃(a + R cos θ₀) = {J * 1e4:.3g} kg·cm²/s stays put while friction takes energy out, so "
            f"it heads for E_J's least, at {least:.3g}°."
        ]
        turn = rt.turning_spin()
        if rt.offset > 0:
            if math.isfinite(turn):
                parts.append(
                    f"Upright is least above {turn / RPM:.4g} rpm, where J would leave it sleeping at "
                    f"{rt.end_spin(J) / RPM:.4g} rpm."
                )
            roll = rt.rolling_tilt()
            if roll < 0.5 * math.pi:
                parts.append(
                    f"But its tip can roll instead of slide below about {math.degrees(roll):.3g}° (sin θ = "
                    "μI₃/maR), and it stops rising there."
                )
            else:
                parts.append("Its friction can make its tip roll at any tilt: it travels instead of rising.")
        elif rt.inverts:
            parts.append(
                f"With 1 − α = {1 - rt.alpha:.3g} &lt; I₁/I₃ = {rt.ratio:.3g} &lt; 1 + α = {1 + rt.alpha:.3g}, "
                f"turned over is least above {turn / RPM:.4g} rpm: it would end spinning at "
                f"{rt.end_spin(J, upright=False) / RPM:.4g} rpm, the other way about its own axis."
            )
        elif rt.offset < 0:
            parts.append(
                f"I₁/I₃ = {rt.ratio:.3g} lies outside 1 ± α ({1 - rt.alpha:.3g} to {1 + rt.alpha:.3g}): turned over is "
                "no minimum, so it doesn't turn over."
            )
        sim = self.sim
        if isinstance(sim, SlidingSimulator) and sim.event is not None:
            parts.append(f"<b>Stopped at {sim.t:.3g} s: {sim.event.value}.</b>")
        self.readout.setText(" ".join(parts))
        hi = 180.0 if self.view.ball else min(math.degrees(rt.rim_tilt()) + 2.0, 180.0)
        self.tilt_plot.setYRange(0.0, hi, padding=0)

    def _draw_table_now(self) -> None:
        s = self.table_samples(TRACK_SPAN)
        i0 = int(np.searchsorted(s.t, s.t[-1] - TABLE_TRAIL))
        trail = slice(i0, None, max(1, -(-(s.t.size - i0) // TRAIL_POINTS)))
        self.view.draw(
            s.axis[-1],
            s.mark[-1],
            s.momentum[-1],
            s.axis[trail],
            s.centre[-1],
            s.centre[trail],
            s.contact[trail],
            s.friction[-1],
        )
        t, y = self.slides.window(SLIDE_TILT_SPAN)
        k = _thin(t.size, 2000)
        self.tilt_item.setData(t[k], np.degrees(np.arccos(np.clip(y[k, 2], -1, 1))))
        t1 = max(float(t[-1]), 10.0)
        self.tilt_plot.setXRange(max(0.0, t1 - SLIDE_TILT_SPAN), t1, padding=0)

    def _draw_now(self) -> None:
        if self.round is not None:
            self._draw_table_now()
            return
        n = self.history.size
        if n:
            last = self.history.y[n - 1]
            axis, mark_, momentum = last[0:3], last[3:6], last[7:10]
        else:
            th = self.motion.tilt
            axis = np.array([0.0, -math.sin(th), math.cos(th)])
            mark_ = np.array([1.0, 0.0, 0.0])
            momentum = self.motion.angular_momentum()
        trail = np.zeros((0, 3))
        skewed = not self.motion.top.symmetric
        if n:
            mean = abs(self.motion.mean_precession())
            span = min(2 * math.pi / mean, 20.0) if mean > 1e-9 else 3 * self.motion.nutation_period()
            t, y = self.history.window(SKEW_TRAIL if skewed else span)
            trail = y[_thin(t.size, TRAIL_POINTS), 0:3]
        self.view.draw(axis, mark_, momentum, trail)
        if skewed:
            self._skewed_tilt()
            return
        if n:
            window = min(NODS * self.motion.nutation_period(), 20.0)
            t, y = self.history.window(window)
            k = _thin(t.size, 2000)
            self.tilt_item.setData(t[k], np.degrees(np.arccos(np.clip(y[k, 2], -1, 1))))
            t1 = max(float(t[-1]), window)
            self.tilt_plot.setXRange(t1 - window, t1, padding=0)
        else:
            self.tilt_item.setData([], [])
            self.tilt_plot.setXRange(0, min(NODS * self.motion.nutation_period(), 20.0), padding=0)
        hi, lo = (math.degrees(v) for v in self.motion.turning())
        pad = max(0.15 * (lo - hi), 0.05 * max(hi, 1.0), 0.05)
        self.tilt_plot.setYRange(hi - pad, lo + pad, padding=0)

    def _skewed_tilt(self) -> None:
        """An asymmetric top's tilt, the last few seconds, ranged to what it has done (no turning points)."""
        t, y = self.history.window(SKEW_TILT)
        if not t.size:
            self.tilt_item.setData([], [])
            return
        k = _thin(t.size, 2000)
        theta = np.degrees(np.arccos(np.clip(y[k, 2], -1, 1)))
        self.tilt_item.setData(t[k], theta)
        t1 = max(float(t[-1]), SKEW_TILT)
        self.tilt_plot.setXRange(t1 - SKEW_TILT, t1, padding=0)
        lo, hi = float(theta.min()), float(theta.max())
        pad = max(0.1 * (hi - lo), 0.5)
        self.tilt_plot.setYRange(max(lo - pad, 0.0), min(hi + pad, 180.0), padding=0)

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_CHAOS:
            self._draw_chaos()
            return
        if self.round is not None:
            if i == TAB_POTENTIAL:
                self._draw_energy()
            elif i == TAB_PATH:
                self._draw_track()
            elif i == TAB_CHECK:
                self._draw_ledger()
            return
        if i == TAB_POTENTIAL:
            self._draw_potential()
        elif i == TAB_PRECESSION:
            self._draw_precession()
        elif i == TAB_PATH:
            self._draw_path()
        elif i == TAB_CHECK:
            self._draw_check()

    def _draw_live_tab(self) -> None:
        i = self.tabs.currentIndex()
        if self.round is not None:
            if i == TAB_POTENTIAL:
                self._energy_dot()
            elif i == TAB_PATH:
                self._track_trace()
            elif i == TAB_CHECK:
                self._frames += 1
                self._draw_ledger(text=self._frames % CHECK_FRAMES == 0)
            return
        if i == TAB_CHAOS:
            self._frames += 1
            if self._frames % SECTION_FRAMES == 0:
                self._draw_chaos()
        elif i == TAB_POTENTIAL:
            self._potential_dot()
        elif i == TAB_PATH:
            self._path_trace()
        elif i == TAB_CHECK:
            self._frames += 1
            self._draw_check(text=self._frames % CHECK_FRAMES == 0)

    # ------------------------------------------------------------- on a table
    def _draw_energy(self) -> None:
        rt = self.round
        assert rt is not None  # only on a table
        m = self.motion
        J = rt.jellett(m.spin, m.tilt)
        hi = rt.rim_tilt()
        theta = np.linspace(0.0, hi, 500)
        least = rt.least_energy(J, theta)
        self.potential_item.setData(np.degrees(theta), least)
        for line in (*self.potential_turns, self.energy_line):
            line.hide()
        start = self._launch_energy
        top, base = max(start, float(np.max(least))), float(np.min(least))
        pad = 0.08 * max(top - base, 1e-9 * abs(top))
        self.potential_plot.setXRange(0.0, math.degrees(hi), padding=0)
        self.potential_plot.setYRange(base - pad, top + pad, padding=0)
        where = math.degrees(rt.least_tilt(J))
        self.potential_note.setText(
            f"J = {J * 1e4:.3g} kg·cm²/s. E_J is least at {where:.3g}°: upright it is {fmt(float(least[0]), 'J')}"
            + (f", turned over {fmt(float(least[-1]), 'J')}." if self.view.ball else ".")
            + f" Let go, it has {fmt(start, 'J')}."
        )
        self._energy_dot()

    def _energy_dot(self) -> None:
        n = self.slides.size
        y = self.slides.y[:n]
        k = _thin(n, 2000)
        theta = np.degrees(np.arccos(np.clip(y[k, 2], -1, 1)))
        self.potential_trace.setData(theta, y[k, 11])
        last = y[n - 1]
        self.potential_now.setData([math.degrees(math.acos(min(1.0, max(-1.0, float(last[2])))))], [float(last[11])])

    def _draw_track(self) -> None:
        self._track_span = None
        self.path_circles.setData([], [])
        self.path_note.setText(
            f"From above, in mm, the last {TRACK_SPAN:g} s: where the tip has touched the table (solid) and where its "
            "centre of mass has been (dotted). Sliding, the tip scrawls; rolling, it runs round smooth curves."
        )
        self._track_trace()

    def _track_trace(self) -> None:
        s = self.table_samples(TRACK_SPAN)
        k = _thin(s.t.size, TRACK_POINTS)
        tip, centre = 1000 * s.contact[k], 1000 * s.centre[k, :2]
        self.path_item.setData(tip[:, 0], tip[:, 1])
        self.path_circles.setData(centre[:, 0], centre[:, 1])
        self.path_now.setData([tip[-1, 0]], [tip[-1, 1]])
        both = np.concatenate([tip, centre])
        lo, hi = both.min(axis=0), both.max(axis=0)
        span = self._track_span
        if span is None or np.any(lo < np.array(span[:2]) - span[2]) or np.any(hi > np.array(span[:2]) + span[2]):
            mid = 0.5 * (lo + hi)
            half = max(0.65 * float(np.max(hi - lo)), 5.0)
            self._track_span = (float(mid[0]), float(mid[1]), half)
            self.path_plot.setRange(
                xRange=(mid[0] - half, mid[0] + half), yRange=(mid[1] - half, mid[1] + half), padding=0
            )

    def _draw_ledger(self, text: bool = True) -> None:
        rt = self.round
        assert rt is not None  # only on a table
        s = self.table_samples()
        start, J0 = self._launch_energy, self._launch_jellett
        k = _thin(s.t.size, CHECK_POINTS)
        ledger = (s.energy + s.work - start) / start
        drift = (s.jellett - J0) / (abs(J0) or 1.0)
        self.energy_item.setData(s.t[k], ledger[k])
        self.momentum_item.setData(s.t[k], drift[k])
        if not text:
            return
        sim = self.sim
        assert isinstance(sim, SlidingSimulator)  # on a table
        axis, L = s.axis[-1], s.momentum[-1]
        along = float(L @ axis)
        spin = along / rt.polar
        above = L[2] / rt.central + (1 / rt.polar - 1 / rt.central) * along * axis[2]
        parts = [
            f"h = {sim.h * 1e3:.3g} ms. Jellett's J held to {np.max(np.abs(drift)):.2g} of itself, and the energy "
            f"with friction's work to {np.max(np.abs(ledger)):.2g}. Friction has taken {100 * s.work[-1] / start:.3g} % "
            f"of it. The table pushed up with at least {s.normal.min() / rt.weight:.3g} mg.",
            f"Now at {math.degrees(math.acos(min(1.0, max(-1.0, float(axis[2]))))):.3g}°, slipping at "
            f"{1000 * s.slip[-1]:.3g} mm/s, spinning at {spin / RPM:.4g} rpm about its axis ({above / RPM:.4g} rpm "
            "seen from above).",
            f"Upright, J would leave it {rt.end_spin(J0) / RPM:.4g} rpm"
            + (
                f"; turned over, {rt.end_spin(J0, upright=False) / RPM:.4g} rpm seen from above."
                if self.view.ball
                else "."
            ),
        ]
        if sim.event is not None:
            parts.append(f"Stopped at {sim.t:.3g} s: {sim.event.value}.")
        self.check_note.setText(" ".join(parts))

    # ------------------------------------------------------------- in the cup
    def _draw_potential(self) -> None:
        m = self.motion
        for line in (*self.potential_turns, self.energy_line):
            line.show()
        self.potential_trace.setData([], [])
        hi, lo = m.turning()
        span = max(lo - hi, math.radians(2.0))
        a, b = max(hi - 1.5 * span, math.radians(0.2)), min(lo + 1.5 * span, math.radians(179.8))
        theta = np.linspace(a, b, 600)
        U = m.effective_potential(theta)
        self.potential_item.setData(np.degrees(theta), U)
        self.energy_line.setPos(m.energy)
        for line, th in zip(self.potential_turns, (hi, lo)):
            line.setPos(math.degrees(th))
        inside = U[(theta >= hi - 0.5 * span) & (theta <= lo + 0.5 * span)]
        top = max(float(np.max(inside)) if inside.size else m.energy, m.energy)
        base = float(np.min(U))
        pad = 0.15 * max(top - base, 1e-9 * max(abs(top), 1e-12))
        self.potential_plot.setXRange(math.degrees(a), math.degrees(b), padding=0)
        self.potential_plot.setYRange(base - pad, top + pad, padding=0)
        self._potential_dot()
        top_ = m.top
        if top_.weight_moment > 0:
            sleeps = abs(m.spin) > top_.sleeping_spin
            self.potential_note.setText(
                f"E' = {fmt(m.energy, 'J')}. Upright it {'sleeps' if sleeps else 'would fall'}: "
                f"|ω₃| = {abs(m.spin) / RPM:.4g} rpm against the {top_.sleeping_spin / RPM:.4g} rpm that "
                "I₃ω₃ &gt; 2√(I₁mgl) asks."
            )
        else:
            self.potential_note.setText(f"E' = {fmt(m.energy, 'J')}. No gravity: U is the precession's energy alone.")

    def _potential_dot(self) -> None:
        n = self.history.size
        if not n:
            th = self.motion.tilt
        else:
            th = math.acos(min(1.0, max(-1.0, float(self.history.y[n - 1, 2]))))
        self.potential_now.setData([math.degrees(th)], [float(self.motion.effective_potential(th))])

    def _draw_precession(self) -> None:
        m = self.motion
        top, th = m.top, m.tilt
        here = abs(m.spin) / RPM
        least = least_spin(top, th) / RPM
        hi = max(4.0 * here, 3.0 * least, 100.0)
        rpm = np.geomspace(max(least * 1.0005, hi / 1e3), hi, 400)
        pairs = np.array([steady_precession(top, r * RPM, th) for r in rpm])
        slow, fast = (np.where(np.abs(v) > 0, np.abs(v), np.nan) for v in pairs.T)
        p = top.polar * rpm * RPM
        self.slow_item.setData(rpm, slow, connect="finite")
        self.fast_item.setData(rpm, fast, connect="finite")
        self.slow_guess.setData(rpm, top.weight_moment / p if top.weight_moment > 0 else np.full_like(rpm, np.nan))
        c = math.cos(th)
        self.fast_guess.setData(rpm, p / (top.transverse * c) if c > 1e-9 else np.full_like(rpm, np.nan))
        self.least_line.setPos(least if least > 0 else -1.0)
        self.least_line.setVisible(least > 0)
        s, f = steady_precession(top, m.spin, th)
        on = math.isfinite(s) and s != 0
        self.precession_now.setData([here] if on else [], [abs(s)] if on else [])
        self.precession_plot.setXRange(0, hi, padding=0)
        self.precession_plot.setLabel("bottom", "Spin |ω₃| (rpm)")
        if top.weight_moment <= 0:
            self.precession_note.setText(
                f"No gravity: no slow precession, and the fast one, {abs(f):.4g} rad/s at {here:.4g} rpm, is the free "
                "top's wobble."
                if math.isfinite(f)
                else "No gravity and lying flat: no steady precession but the spin."
            )
        elif least > 0 and here < least:
            self.precession_note.setText(
                f"At {math.degrees(th):.4g}° it needs {least:.4g} rpm to precess steadily; at {here:.4g} rpm it "
                "can only nod."
            )
        else:
            fast_text = f" and {abs(f):.4g}" if math.isfinite(f) else ""
            self.precession_note.setText(
                f"At {math.degrees(th):.4g}° and {here:.4g} rpm: {abs(s):.4g}{fast_text} rad/s (the estimates "
                f"{top.weight_moment / (top.polar * abs(m.spin)):.4g}"
                + (f" and {top.polar * abs(m.spin) / (top.transverse * c):.4g}" if c > 1e-9 else "")
                + ")."
                + (f" Under {least:.4g} rpm (dotted) it can't." if least > 0 else "")
            )

    def _draw_path(self) -> None:
        if not self.motion.top.symmetric:
            self.path_circles.setData([], [])
            self.path_plot.setRange(xRange=(-1.05, 1.05), yRange=(-1.05, 1.05), padding=0)
            self.path_note.setText(
                f"From above, the last {SKEW_PATH:g} s. With no turning points to draw it against, the path no longer "
                "keeps between two circles."
            )
            self._path_trace()
            return
        ring = np.linspace(0, 2 * math.pi, 241)
        xs, ys = [], []
        for th in self.motion.turning():
            xs += list(math.sin(th) * np.cos(ring)) + [np.nan]
            ys += list(math.sin(th) * np.sin(ring)) + [np.nan]
        self.path_circles.setData(xs, ys)
        reach = max(math.sin(th) for th in self.motion.turning())
        if max(self.motion.turning()) > math.pi / 2:
            reach = 1.0
        span = max(1.1 * reach, 0.02)
        self.path_plot.setRange(xRange=(-span, span), yRange=(-span, span), padding=0)
        self.path_note.setText(
            f"From above, the last two turns: {self.motion.path().value}. Let go at the bottom of the plot, it "
            + (
                f"precesses {'counterclockwise' if self.motion.mean_precession() > 0 else 'clockwise'}."
                if abs(self.motion.mean_precession()) > 1e-9
                else "doesn't precess on average: it circles L."
            )
        )
        self._path_trace()

    def _path_trace(self) -> None:
        n = self.history.size
        if not n:
            self.path_item.setData([], [])
            self.path_now.setData([], [])
            return
        mean = abs(self.motion.mean_precession())
        if not self.motion.top.symmetric:
            span = SKEW_PATH
        else:
            span = min(4 * math.pi / mean, 40.0) if mean > 1e-9 else 6 * self.motion.nutation_period()
        _, y = self.history.window(span)
        k = _thin(len(y), PATH_POINTS)
        self.path_item.setData(y[k, 0], y[k, 1])
        self.path_now.setData([y[-1, 0]], [y[-1, 1]])

    def _skewed_check(self, t: np.ndarray, y: np.ndarray, scale_e: float, scale_l: float) -> None:
        """An asymmetric top's check: what's kept, ω₃ not, and without gravity Jacobi's ω."""
        m = self.motion
        top = m.top
        spin = np.sum(y[:, 7:10] * y[:, 0:3], axis=1) / top.polar / RPM
        parts = [
            f"h = {self.sim.h * 1e3:.3g} ms. Largest energy error {np.max(np.abs(y[:, 6] - m.total_energy)) / scale_e:.2g}"
            f"{' mgl' if top.weight_moment > 0 else ' of it'}, L·ẑ's {np.max(np.abs(y[:, 9] - m.vertical_momentum)) / scale_l:.2g} "
            f"of I₃ω₃. ω₃ isn't kept: it has run between {spin.min():.4g} and {spin.max():.4g} rpm."
        ]
        if top.weight_moment == 0:
            omega = body_rates(y[:, 0:3], y[:, 3:6], y[:, 7:10], top)
            exact = FreeSpin.of(top, self._launch_rates()).omega(t)
            error = np.max(np.linalg.norm(omega - exact, axis=1)) / max(abs(m.spin), 1e-12)
            parts.append(f"ω in the body against Euler–Poinsot's (Jacobi's cn, sn, dn): within {error:.2g} of ω₃.")
        self.check_note.setText(" ".join(parts))

    def _draw_chaos(self) -> None:
        """The Poincaré section and the two tops' separation."""
        n = self.section.size
        self.section_item.setData(self.section.y[:n, 0], self.section.y[:n, 1])
        a = self.apart.size
        t, d = self.apart.t[:a], self.apart.y[:a, 0]
        keep = d > 0
        self.apart_item.setData(t[keep], d[keep])
        if a:
            self.apart_plot.setXRange(0.0, max(float(t[-1]), 1.0), padding=0)
        self.apart_plot.setYRange(math.log10(TWIN) - 1, math.log10(2.5), padding=0)
        now = float(d[-1]) if a else TWIN
        symmetric = self.motion.top.symmetric
        self.chaos_note.setText(
            f"{n} peaks of the tilt so far. "
            + (
                "A symmetric top keeps ω₃, so its points lie on one line."
                if symmetric
                else "Closed curves: the motion is regular; a scatter that fills an area: chaos."
            )
            + f" Below, a second top let go {TWIN:g} rad further over: {now:.2g} rad apart now. Regular motion "
            "drifts apart slowly, in proportion to the time; chaos doubles the gap over and over, a straight line on "
            "this log scale, until they're as far apart as two axes can be."
        )

    def _draw_check(self, text: bool = True) -> None:
        n = self.history.size
        m = self.motion
        top = m.top
        scale_e = top.weight_moment if top.weight_moment > 0 else max(abs(m.total_energy), 1e-12)
        scale_l = max(abs(m.spin_momentum), 1e-12)
        if not n:
            self.energy_item.setData([], [])
            self.momentum_item.setData([], [])
            self.check_note.setText("Run it to compare.")
            return
        t, y = self.history.t[:n], self.history.y[:n]
        k = _thin(n, CHECK_POINTS)
        self.energy_item.setData(t[k], (y[k, 6] - m.total_energy) / scale_e)
        along = np.sum(y[k, 7:10] * y[k, 0:3], axis=1)
        drift = np.abs(y[k, 9] - m.vertical_momentum) / scale_l
        if top.symmetric:
            drift = np.maximum(drift, np.abs(along - m.spin_momentum) / scale_l)
        self.momentum_item.setData(t[k], drift)
        if not text:
            return
        if not top.symmetric:
            self._skewed_check(t, y, scale_e, scale_l)
            return
        theta, phi = tilt_and_azimuth(y[:, 0:3])
        parts = [
            f"h = {self.sim.h * 1e3:.3g} ms. Largest energy error {np.max(np.abs(y[:, 6] - m.total_energy)) / scale_e:.2g} mgl."
        ]
        hi, lo = (math.degrees(v) for v in m.turning())
        parts.append(
            f"Tilt run between {math.degrees(theta.min()):.4g}° and {math.degrees(theta.max()):.4g}° "
            f"(closed form {hi:.4g}° and {lo:.4g}°)."
        )
        T = m.nutation_period()
        if not m.steady and t[-1] - t[0] > 2 * T:
            whole = int((t[-1] - t[0]) / T)
            i = min(int(np.searchsorted(t, t[0] + whole * T)), n - 1)
            rate = (phi[i] - phi[0]) / (t[i] - t[0])
            parts.append(
                f"Over {whole} whole nods it precessed at {rate:.4g} rad/s on average (closed form "
                f"{m.mean_precession():.4g})."
            )
        self.check_note.setText(" ".join(parts))
