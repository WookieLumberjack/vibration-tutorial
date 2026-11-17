"""The Gyroscope page's Things to try and practice problems: its trial dataclass, the settings and the tabs it
names, and the problems, each worked from its printed numbers with core/gyroscope.py and core/sliding_top.py.

`GyroscopePage.apply_trial` applies a trial; the page re-exports `TRIALS` and `PROBLEMS`, where the practice window
and the Theory tab look for them. `tests/test_tutorial_numbers.py` checks the trials' numbers, `tests/test_practice.py`
the problems.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np
import scipy.optimize

from ..core.gyroscope import RPM, Launch, Motion, Shape, Top, least_spin, steady_precession
from ..core.practice import Prediction, Problem, Question, Requirement, kept, number, pick, rounded
from ..core.sliding_top import RoundTop
from .links import page_link

TAB_POTENTIAL, TAB_PRECESSION, TAB_PATH, TAB_COMPONENTS, TAB_CHECK, TAB_CHAOS = range(6)
PLAYBACK = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
REAL_TIME = PLAYBACK.index(1.0)


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
    shape: Shape = Shape.DISC  # radius is a square's side, or a rectangle's w
    breadth: float = 0.04  # m, a rectangle's other side b
    predict: Prediction | None = None  # asked before it runs (Things to try)

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
        "Let go at rest",
        "A 0.5 kg disc 40 mm in radius, 30 mm up its stem, spun at 1500 rpm and let go at 30° without pushing it "
        "round. It doesn't fall: it drops only to 31.5° and comes back, nodding 14.7 times a second, while the "
        "whole top precesses round at 2.39 rad/s (a turn in 2.63 s). Each nod starts from rest, so the axis's "
        "path (the Path tab) has a cusp at the top of each.",
        predict=Prediction(
            "Let go at 30° spinning at 1500 rpm, without being pushed round. What does the tip of its axis trace?",
            ("A circle", "Cusps: it stops, dips and goes on", "Loops", "Smooth waves"),
            1,
        ),
    ),
    GyroTrial(
        "Steady precession",
        "The same top let go already precessing at the slow steady rate, 2.39 rad/s: the root of "
        "<m>I₁</m> cos <m>θ φ̇</m>² − <m>I₃ω₃φ̇</m> + <m>mgl</m> = 0. It doesn't nod at all, and its axis traces a circle. "
        "<m>mgl</m>/<m>I₃ω₃</m> = 2.34 rad/s is the fast top's estimate of it.",
        launch=Launch.SLOW,
    ),
    GyroTrial(
        "Fast precession",
        "The quadratic's other root, 109 rad/s (17 turns a second): steady too, near <m>I₃ω₃</m>/<m>I₁</m> cos <m>θ</m> = 112 rad/s, "
        "the free top's own wobble. Watch it at 0.05× speed.",
        launch=Launch.FAST,
        playback=PLAYBACK.index(0.05),
    ),
    GyroTrial(
        "Pushed back",
        "Let go precessing backward at −2 rad/s: it drops to 32.7°, and the precession turns forward on the way "
        "down and back at the top of each nod, so the path loops. On average it still precesses forward at "
        "2.39 rad/s.",
        launch=Launch.SET,
        precession=-2.0,
        predict=Prediction(
            "The same top let go pushed backward, against the way it precesses, at 2 rad/s. What does the tip of its axis trace?",
            ("Cusps", "Loops", "Smooth waves", "A circle, the other way round"),
            1,
        ),
    ),
    GyroTrial(
        "Pushed along",
        "Let go precessing forward at 1.2 rad/s, half the steady rate: it drops less, to 30.7°, and never stops "
        "precessing, so the path waves. Faster than the steady rate (try 4 rad/s) it rises instead.",
        launch=Launch.SET,
        precession=1.2,
        predict=Prediction(
            "Let go pushed forward at 1.2 rad/s, half its steady precession rate. What does the tip of its axis trace?",
            ("Cusps", "Loops", "Smooth waves", "A circle"),
            2,
        ),
    ),
    GyroTrial(
        "Spun slowly: deep nods",
        "At 600 rpm the nods are deep, from 30° to 43.9°, and slow, 4.53 a second; the precession is 6.70 rad/s "
        "on average. The fast top's estimates (nods 2<m>I₁mgl</m> sin <m>θ</m>/(<m>I₃ω₃</m>)² = 8.7° deep, at <m>I₃ω₃</m>/2π<m>I₁</m> = 6.2 Hz) no "
        "longer hold.",
        spin_rpm=600.0,
        playback=PLAYBACK.index(0.25),
    ),
    GyroTrial(
        "A sleeping top",
        "Nearly upright (2°) at 1000 rpm, over the 467 rpm that <m>I₃ω₃</m> &gt; 2√(<m>I₁mgl</m>) asks: it stays up, nodding "
        "between 2° and 2.26°. Upright, the Euler angles <m>φ</m> and <m>ψ</m> turn about the same axis, so their equations "
        "are singular there; the simulation steps the orientation as a quaternion and doesn't notice.",
        spin_rpm=1000.0,
        tilt=2.0,
    ),
    GyroTrial(
        "Nearly upright at 400 rpm",
        "At 400 rpm, under 467, upright is unstable: let go at 2°, it falls to 62.2° before it comes back up, "
        "nearly a second a nod. A real top's spin dies away in the friction at its tip, and once under 467 rpm it "
        "starts to wobble.",
        spin_rpm=400.0,
        tilt=2.0,
        tab=TAB_POTENTIAL,
        predict=Prediction(
            "At 1000 rpm the top sleeps upright; it needs 467. Let go at 2° spinning at only 400 rpm, what does it do?",
            (
                "It stays near 2°, nodding a little",
                "It falls far and comes back up",
                "It stands upright",
                "It precesses at 2° without nodding",
            ),
            1,
        ),
    ),
    GyroTrial(
        "Lying flat",
        "At 90°, as a bicycle wheel held by one end of its axle, steady precession has the one root, "
        "<m>mgl</m>/<m>I₃ω₃</m> = 2.34 rad/s exactly: a turn in 2.68 s. Gravity's moment mgl turns the angular momentum I₃ω₃ "
        "round without changing its size.",
        tilt=90.0,
        launch=Launch.SLOW,
    ),
    GyroTrial(
        "Spun the other way",
        "At −1500 rpm it does everything in mirror: the precession is −2.39 rad/s. The precession always "
        "turns the angular momentum toward the moment.",
        spin_rpm=-1500.0,
        predict=Prediction(
            "The first trial's top, spun the other way (−1500 rpm) and let go the same way. Which way does it precess?",
            ("The same way as before", "The opposite way", "It doesn't precess: it falls"),
            1,
        ),
    ),
    GyroTrial(
        "No gravity: nutation is the free wobble",
        "With g = 0 and let go precessing at 5 rad/s, nothing acts on it, so <b>L</b> stays put, 28.5° from the "
        "vertical, and the axis turns round L once every 2π<m>I₁</m>/|<b>L</b>| = 65.0 ms. Seen from the vertical, its tilt "
        "nods between 27.0° and 30°: nutation is this free wobble, which gravity's precession carries round.",
        gravity=0.0,
        launch=Launch.SET,
        precession=5.0,
        playback=PLAYBACK.index(0.05),
        tab=TAB_PATH,
    ),
    GyroTrial(
        "A square top: just like a disc",
        "A 0.5 kg square plate 69.28 mm on a side in place of the disc: <m>I₃</m> = <m>ms</m>²/6 = 4 kg·cm², the first trial's "
        "disc's <m>mr</m>²/2 (<m>r</m> = <m>s</m>/√3 = 40 mm), and about every line across its axis <m>ms</m>²/12, that disc's <m>mr</m>²/4. So it "
        "does just what the disc did: drops to 31.5°, nods 14.7 times a second and precesses at 2.39 rad/s. Its "
        "corners make no difference; only its moments do.",
        shape=Shape.SQUARE,
        radius=0.06928,
    ),
    GyroTrial(
        "A rectangle: ω₃ isn't kept",
        "A 0.5 kg plate 100 × 40 mm, 30 mm up its stem: <m>I₁</m> = 5.17, <m>I₂</m> = 8.67 and <m>I₃</m> = 4.83 kg·cm², so <m>I₁</m> ≠ <m>I₂</m>. Its "
        "spin <m>ω₃</m> now wanders, between 596.7 and 601.1 rpm, so the tilt has no potential of its own: it nods between "
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
        "No gravity: spun about the middle axis",
        "A plate 100 × 20 mm, 15 mm up its stem, with g = 0: <m>I₁</m> = 1.29 &lt; <m>I₃</m> = 4.33 &lt; <m>I₂</m> = 5.29 kg·cm², so it "
        "spins about its middle axis, with nothing acting on it. It flips upside down and back, crossing the "
        "horizontal every 0.455 s: half the period 4<m>K</m>(<m>k</m>)/<m>λ</m> = 0.910 s of Euler–Poinsot's solution in Jacobi's "
        "elliptic functions (<m>k</m>² = 1 − 1.2 × 10⁻⁷: near the separatrix, where the period grows without bound). The "
        "Check tab compares <m>ω</m> with it. The free asymmetric top still has a closed form, if a much harder one.",
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
        predict=Prediction(
            "A plate 100 × 20 mm spun about its stem, its middle axis, with no gravity and a little off the axis. What does it do?",
            (
                "It keeps spinning steadily about its stem",
                "It flips over and back, again and again",
                "It slowly settles onto its longest axis",
            ),
            1,
        ),
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
        "on a table with <m>μ</m> = 0.01. Its tip slides, and friction raises it: under 10° in 12 s, 5° in 19 s. Jellett's "
        "<m>J</m> = <m>I₃ω₃</m>(<m>a</m> + <m>R</m> cos <m>θ</m>) stays put as friction takes energy out, and the least energy J allows (the Potential "
        "tab) is upright. Below 4.6° (sin <m>θ</m> = <m>μI₃</m>/<m>maR</m>) friction can make its tip roll instead, and the rise stalls "
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
        "The same top on wood, <m>μ</m> = 0.3: its tip catches and rolls, so friction takes hardly any energy and it "
        "doesn't rise. It drops to 16.2° at once, then rolls round a circle about 1.35 m across (the Path tab), "
        "tilted all the way.",
        setting=Setting.TABLE,
        height=0.02,
        friction=0.3,
        tilt=20.0,
        tab=TAB_PATH,
    ),
    GyroTrial(
        "A tippe top spun fast",
        "A 20 g ball 15 mm in radius, weighted 4.5 mm below its centre (<m>α</m> = 0.3), with <m>I₁</m>/<m>I₃</m> = 0.8, spun at "
        "1200 rpm on its bottom, 5° off upright. 1 − <m>α</m> &lt; <m>I₁</m>/<m>I₃</m> &lt; 1 + <m>α</m> and it spins faster than 468 rpm, so "
        "turned over is the least energy Jellett's <m>J</m> allows: friction turns it over, past 90° in 1.8 s, and it ends "
        "on its top spinning at <m>J</m>/<m>I₃</m>(<m>R</m> − <m>a</m>) = 643 rpm, the same way seen from above but the other way about its "
        "own axis. Its centre of mass rises 9 mm, and half its energy goes in the friction.",
        setting=Setting.TIPPE,
        mass=0.02,
        radius=0.015,
        spin_rpm=1200.0,
        tilt=5.0,
        playback=PLAYBACK.index(0.25),
        tab=TAB_POTENTIAL,
        predict=Prediction(
            "A ball weighted below its centre, spun at 1200 rpm on its bottom, 5° off upright, on a table. What does it do?",
            (
                "It stays upright, its weight low",
                "It turns over onto its top, the weight up",
                "It rolls onto its side and spins there",
            ),
            1,
        ),
    ),
    GyroTrial(
        "A tippe top spun too slowly",
        "The same tippe top at 400 rpm, under the 468 rpm at which upright stops being <m>E<sub>J</sub></m>'s least: it stays up, "
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
        "With <m>I₁</m>/<m>I₃</m> = 1.4, over 1 + <m>α</m> = 1.3, turned over is no minimum: <m>E<sub>J</sub></m> is least at 108.6°, and the ball "
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
        why=(
            "",
            "A top's motion depends only on its moments of inertia, and the square's, ms²/6 and ms²/12, equal the "
            "disc's: the corners don't show.",
            "The precession depends on the moments I₃ and I₁, and the square's equal the disc's.",
            "The nod depends on the moments too, and with s = √3 × 40 mm the square's are the disc's.",
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
        f"About a line through its centre parallel to the {b * 1000:g} mm sides the plate has mw²/12, w = "
        f"{a * 1000:g} mm being how far it reaches across that line; moved l up the axis it gains ml²: I₂ = m(w²/12 + l²) = "
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
        f"I₁ = m(b²/12 + l²) = {number(i1 * 1e4)}, I₂ = m(w²/12 + l²) = {number(i2 * 1e4)} and I₃ = m(w² + b²)/12 = "
        f"{number(i3 * 1e4)} kg·cm²: I₃ is the "
        + ("middle one, and spin about the middle axis is unstable: the tennis-racket flip." if middle else "")
        + ("" if middle else f"{'largest' if i3 > max(i1, i2) else 'smallest'}, and spin about it is stable."),
        choices=(flips, keeps) if middle else (keeps, flips),
        why=(
            "",
            (
                "I₃ is the middle moment here, and spin about the middle axis is unstable: a nudge grows into the flip."
                if middle
                else "It flips only about the middle axis; here I₃ is the "
                + ("largest" if i3 > max(i1, i2) else "smallest")
                + ", and a nudge only circles."
            ),
        ),
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
        f"Held at θ, the tilt's equation leaves I₁ cos θ φ̇² − I₃ω₃ φ̇ + mgl = 0, with I₃ω₃ = "
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
        why=(
            "",
            "The rate is about mgl/I₃ω₃: a larger spin is a larger angular momentum for the same moment to turn, "
            "so it's slower.",
            f"Gravity's moment is the same, but it turns an angular momentum {n} times larger: {n} times slower.",
            f"The rate goes as 1/ω₃, not 1/ω₃²: {n} times slower.",
        ),
        setup=dict(spin_rpm=1500.0 * n, tilt=30.0, launch=Launch.SLOW, tab=TAB_PRECESSION),
        predict=True,
    )


def _at_rest_path(rng: np.random.Generator) -> Question:
    return Question(
        "A fast top is let go tilted, without being pushed round. What does the tip of its axis trace?",
        0,
        "Cusps: it starts at rest, so at the top of every nod it stops (φ̇ = 0) before it falls and is turned "
        "round again.",
        choices=("Cusps", "Loops", "Smooth waves", "A circle"),
        why=(
            "",
            "Loops need the precession to run backward at a turning point, which takes a push against it; let go "
            "at rest, φ̇ only falls to 0 there.",
            "Smooth waves need it pushed round to start; let go at rest, it stops at the top of every nod: a cusp.",
            "A circle is steady precession, which needs just the right push round; let go at rest, it nods.",
        ),
        setup=dict(tilt=30.0, launch=Launch.REST, tab=TAB_PATH),
        predict=True,
    )


def _pushed_back(rng: np.random.Generator) -> Question:
    return Question(
        "A spinning top is let go tilted with a push the opposite way to its precession. What does the tip of its "
        "axis trace?",
        0,
        "Loops: the precession rate φ̇ = (a<sub>φ</sub> − a<sub>ψ</sub> cos θ)/sin²θ changes sign between the turning points, so it runs "
        "backward at one of them.",
        choices=("Loops", "Cusps", "Smooth waves", "A circle"),
        why=(
            "",
            "Cusps are a top let go at rest, φ̇ = 0 at the top of each nod. Pushed against its precession, φ̇ "
            "changes sign: loops.",
            "Pushed the opposite way, φ̇ changes sign between the turning points, so the path runs backward: loops.",
            "A circle needs the push that matches steady precession; the opposite one makes it nod and loop.",
        ),
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
        why=(
            "",
            "Under 467 rpm upright is unstable: a 2° lean isn't held but grows, to 62°, before it comes back.",
            "To stand upright it has to spin faster than 467 rpm; at 400 upright is unstable.",
            "Steady precession at 2° needs a fast enough spin; under 467 rpm upright is unstable, and it falls far.",
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
        why=(
            "",
            "Precession turns L towards gravity's moment; with L reversed, it has to turn the other way to get there.",
            "Gravity's moment is still there, and turns the reversed L: it precesses, the other way.",
        ),
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
        f"Steady at θ: I₁ cos θ φ̇² − I₃ω₃ φ̇ + mgl = 0 with φ̇ = 2π/{number(target)} s = "
        f"{number(2 * math.pi / target)} rad/s, so ω₃ = (I₁ cos θ φ̇² + mgl)/I₃φ̇: about {number(rpm)} rpm. Let "
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
    reasons = {
        up: ""
        if ratio < 0.7
        else f"Upright stays E<sub>J</sub>'s least only for I₁/I₃ under 1 − α = 0.7; here it's {ratio:g}.",
        over: ""
        if 0.7 <= ratio < 1.3
        else f"Turned over is E<sub>J</sub>'s least only for I₁/I₃ between 0.7 and 1.3; here it's {ratio:g}.",
        side: "" if ratio >= 1.3 else f"It ends on its side only for I₁/I₃ over 1 + α = 1.3; here it's {ratio:g}.",
    }
    return Question(
        f"A tippe top's weight sits 0.3 of its radius below the ball's centre (α = 0.3), and I₁/I₃ = {ratio:g} about "
        "its centre of mass. Spun fast (1200 rpm) on its bottom on a table, what does it do?",
        0,
        f"1 − α = 0.7 and 1 + α = 1.3; I₁/I₃ = {ratio:g} is {why}",
        choices=choices,
        why=tuple(reasons[c] for c in choices),
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
