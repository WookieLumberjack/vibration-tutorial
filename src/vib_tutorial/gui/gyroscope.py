"""Gyroscope page: a spinning top with its tip in a cup, from Euler's equations to precession and nutation.

Left: the top (a disc on a light stem: its mass, radius and height above the tip, and gravity), how it is let go
(the spin, the tilt, and the precession: at rest, at either steady rate, or set), and the playback. Centre: the
top in 3D (drag to turn it), with its axis's path and the vectors ω, L and the moment, then its tilt against
time beside the turning points worked out in closed form. Right: the effective potential the tilt moves in, the
steady precession rates against the spin, the axis's path from above, the simulation's check against what is
conserved, and the Theory. The page has its own model (core/gyroscope.py), stepped on its own frame loop while it
shows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
import scipy.optimize
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.gyroscope import (
    LAUNCHES,
    RPM,
    Launch,
    Motion,
    Top,
    TopSimulator,
    least_spin,
    steady_precession,
    tilt_and_azimuth,
)
from ..core.practice import Problem, Question, Requirement, kept, number, pick, rounded
from . import frame_stats, settle_plots
from .cms_time import SampleBuffer
from .lesson import (
    Dot,
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
from .links import before, page_link, takeaways
from .style import colors
from .theming import add_legend

TAB_POTENTIAL, TAB_PRECESSION, TAB_PATH, TAB_CHECK = range(4)
FRAME_MS = 16
PLAYBACK = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
REAL_TIME = PLAYBACK.index(1.0)
NODS = 6  # nutation periods the tilt plot spans
TRAIL_POINTS = 1500  # at most, on the 3D view's trail
PATH_POINTS = 4000  # at most, on the path from above
CHECK_POINTS = 2000
CHECK_FRAMES = 30  # the Check tab's text is worked out over the whole run, every this many frames
HISTORY = 400_000  # samples kept


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

    @property
    def top(self) -> Top:
        return Top(self.mass, self.radius, self.height, self.gravity)

    @property
    def motion(self) -> Motion:
        return Motion.launched(self.top, self.spin_rpm * RPM, math.radians(self.tilt), self.launch, self.precession)


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
)


def _disc(rng: np.random.Generator) -> tuple[float, float, float]:
    return pick(rng, (0.2, 0.5, 1.0, 2.0)), pick(rng, (0.03, 0.04, 0.05, 0.06)), pick(rng, (0.02, 0.03, 0.04))


# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


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


PROBLEMS = tuple(
    Problem(key, title, "gyroscope_page", f"See {page_link('gyroscope_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("moments", "A disc on a stem", "rigid-body", "A rigid body turning", _moments),
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
    )
)

TIPS = {
    "mass": "<p>The disc's mass (the stem's is left out).</p>",
    "radius": "<p>The disc's radius: I₃ = mr²/2.</p>",
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
CHECK_TIP = (
    "<p>How far the simulation's energy has strayed (over mgl), and its two momenta's (over I₃ω₃: round-off "
    "only). The text compares what it has run with the closed forms.</p>"
)

NOTES_HTML = """
<!--BEFORE-->
<h3><a name="rigid-body"></a>A rigid body turning</h3>
<p>A body turning about a fixed point has angular momentum L = Iω, with I its inertia tensor about the
point, and gravity's moment τ about the point changes it: dL/dt = τ (in fixed axes). In the body's
<b>principal axes</b> I is diagonal, (I₁, I₂, I₃), but those axes turn, so L's rate in them gains ω × L:</p>
<p align="center">I₁ω₁' + (I₃ − I₂)ω₂ω₃ = τ₁, &nbsp; I₂ω₂' + (I₁ − I₃)ω₃ω₁ = τ₂, &nbsp;
I₃ω₃' + (I₂ − I₁)ω₁ω₂ = τ₃,</p>
<p><b>Euler's equations</b>. A <b>symmetric top</b> has I₁ = I₂, so the third reads I₃ω₃' = τ₃, and gravity's
moment, across the axis, leaves ω₃ (the spin) constant. The top here is a disc of mass m and radius r on a light
stem, its centre a height l above the tip: I₃ = mr²/2 along the axis, and I₁ = m(r²/4 + l²) across it about the
tip (parallel axes, as for the Pendulums page's rod).</p>
<p>Without a moment, L stays put and the axis turns round it at |L|/I₁ (in a cone), while the body turns about
its own axis at (1/I₃ − 1/I₁)L₃ besides: the free top's wobble.</p>

<h3><a name="euler-angles"></a>Euler angles, and why the simulation doesn't use them</h3>
<p>Three angles place the axis and the body round it: the <b>precession</b> φ about the vertical, the
<b>tilt</b> θ from the vertical, and the <b>spin</b> ψ about the axis. Then
ω₃ = ψ' + φ' cos θ, and the axis's own rates are θ' and φ' sin θ. Upright (θ = 0), φ and ψ turn about the same
axis: one of the three angles is lost, the angle rates' equations divide by sin θ, and they blow up (gimbal
lock). A sleeping top sits right there.</p>
<p>A <b>unit quaternion</b> q = (cos ½α, n̂ sin ½α), a turn by α about n̂, has no such place: turns compose by
multiplying quaternions, and nothing divides by zero. The simulation keeps the orientation as one and L in fixed
axes.</p>

<h3><a name="lagrange"></a>Lagrange's top</h3>
<p>The energy in the angles gives Lagrange's function</p>
<p align="center">L = ½I₁(θ'² + φ'² sin²θ) + ½I₃(ψ' + φ' cos θ)² − mgl cos θ.</p>
<p>Neither φ nor ψ appears in it, so their momenta are constant: p<sub>ψ</sub> = I₃ω₃, the angular momentum
about the axis, and p<sub>φ</sub> = I₁φ' sin²θ + I₃ω₃ cos θ, about the vertical (gravity's moment, horizontal,
changes neither). So does the energy. Taking φ' from p<sub>φ</sub>, the tilt is left on its own,</p>
<p align="center">½I₁θ'² + U(θ) = E', &nbsp; U(θ) = (p<sub>φ</sub> − p<sub>ψ</sub> cos θ)²/2I₁ sin²θ + mgl cos θ,</p>
<p>E' the energy less the spin's ½I₃ω₃². It is a one-mass problem in an <b>effective potential</b>, as the
Pendulums page's large swing is in its V(θ), and the tilt moves between the two angles where U = E' (the
<i>Potential</i> tab), the <b>turning points</b>.</p>

<h3><a name="precession"></a>Steady precession</h3>
<p>Held at one tilt (θ'' = 0), the tilt's equation leaves</p>
<p align="center">I₁ cos θ φ'² − I₃ω₃ φ' + mgl = 0,</p>
<p>so a top can precess steadily at either of two rates. For a fast top, the <b>slow</b> one is about
mgl/I₃ω₃: gravity's moment mgl sin θ turns the angular momentum's horizontal part I₃ω₃ sin θ round, at a rate
falling as the spin rises. The <b>fast</b> one is about I₃ω₃/I₁ cos θ, the free wobble, gravity hardly in it.
Lying flat (cos θ = 0) the slow one is exactly mgl/I₃ω₃ and the fast one is gone: the bicycle wheel held by one
end of its axle. The two meet, and then neither is left, at I₃ω₃ = 2√(I₁ mgl cos θ): a slower top can't precess
steadily at that tilt (the <i>Precession</i> tab). The precession always turns L toward the moment, so a top spun
the other way precesses the other way.</p>

<h3><a name="nutation"></a>Nutation</h3>
<p>Let go at any other rate, the tilt nods between the turning points, <b>nutation</b>. With u = cos θ, the
energy is</p>
<p align="center">u'² = f(u) = (α − βu)(1 − u²) − (b − au)²,</p>
<p>with a = p<sub>ψ</sub>/I₁, b = p<sub>φ</sub>/I₁, α = 2E'/I₁ and β = 2mgl/I₁: a cubic, whose roots
u₁ &lt; u₂ in [−1, 1] are the turning points and u₃ &gt; 1 the third. A nod and back takes</p>
<p align="center">T = 4K(m)/√(β(u₃ − u₁)), &nbsp; m = (u₂ − u₁)/(u₃ − u₁),</p>
<p>K the complete elliptic integral, as in the Pendulums page's large swing. A fast top let go at θ₀ nods about
2I₁mgl sin θ₀/(I₃ω₃)² deep at about I₃ω₃/I₁: the faster it spins, the shallower and quicker. As it nods it
precesses at φ' = (b − a cos θ)/sin²θ, so the end of its axis (the <i>Path</i> tab) traces</p>
<ul>
<li><b>cusps</b> if φ' = 0 at a turning point: let go without a push, it stops at the top of each nod;</li>
<li><b>loops</b> if φ' changes sign between them: pushed back, it runs backward at one;</li>
<li><b>waves</b> if it doesn't;</li>
<li>a <b>circle</b> if the two turning points meet: steady precession.</li>
</ul>
<p>Without gravity the top is free: L stays put, and the axis turns round it at |L|/I₁ on a cone, so its tilt
from the vertical nods unless L is vertical. Nutation is that wobble, which gravity's precession carries round.</p>

<h3><a name="sleeping"></a>The sleeping top</h3>
<p>Upright, b = a, and U(θ) ≈ mgl + ½(I₃²ω₃²/4I₁ − mgl)θ² for small θ: a minimum, so upright is stable, if</p>
<p align="center">I₃²ω₃² &gt; 4I₁mgl, &nbsp; ω₃ &gt; 2√(I₁mgl)/I₃.</p>
<p>A top spinning faster stands still upright, <b>asleep</b>; nudged, it nods a little round upright. Slower, U
has a maximum there, and it falls until it comes back up. Friction at the tip slows a real top, which wakes, then
wobbles, then falls. (The cup holds this top's tip; on a table, a round tip rolls and slides, which is what makes
a top rise to sleep, and a tippe top turn over.)</p>

<h3><a name="simulation"></a>The simulation</h3>
<p>The simulation steps the whole body: q and L. Each step of h is split: gravity's moment over half a step
(L += ½h τ, τ = −mgl e₃ × ẑ: exact, since τ depends only on where the axis points, which doesn't change in it),
the free top over the whole step (exact: L fixed, the axis turned round it by |L|h/I₁ and the body about its axis
by (1/I₃ − 1/I₁)L₃h), and gravity's other half. This splitting is <b>symplectic</b>, as the Numerical integration
page's central difference is: the energy's error stays bounded rather than drifting, and falls as h². Both
momenta are kept to round-off, since the moment is at right angles to the vertical and to the axis. The step
resolves the free wobble with 200 steps a turn. The <i>Check</i> tab compares the run with the closed forms.</p>

<h3><a name="to-the-rotor"></a>From the top to the rotor</h3>
<p>A disc on a spinning shaft is a gyroscope too: as the shaft whirls, the disc's axis tilts, and turning its
angular momentum I<sub>p</sub>Ω takes a moment. Its two whirl modes are the top's two precessions: a forward
whirl, stiffened by the spin, its tilting frequency rising toward I<sub>p</sub>Ω/I<sub>d</sub>, the fast
precession's I₃ω₃/I₁; and a backward one, softened, falling as the slow precession does. The Jeffcott rotor
page's Campbell diagram plots them against the speed (see its <i>The gyroscopic effect</i>).</p>

<h3><a name="try-it"></a>Things to try</h3>
<p>Each of these sets up the top and lets it go (Things to try, top left).</p>
<!--TRIALS-->
<!--TAKEAWAYS-->
"""
NOTES_HTML = (
    NOTES_HTML.replace(
        "<!--BEFORE-->",
        before(
            ("pendulum_page", "rotation", "Turning instead of sliding"),
            ("pendulum_page", "large-swings", "Large swings"),
            ("integration_page", "central", "Central difference"),
        ),
    )
    .replace("<!--TRIALS-->", trials_html(TRIALS))
    .replace(
        "<!--TAKEAWAYS-->",
        takeaways(
            "Gravity's moment turns a spinning top's angular momentum round instead of tipping it over: it precesses, "
            "at about mgl/I₃ω₃, slower the faster it spins.",
            "The spin and the angular momentum about the vertical are conserved, so the tilt moves on its own in an "
            "effective potential, nodding (nutation) between two turning points.",
            "A fast top nods at about I₃ω₃/I₁, shallowly; let go at rest its axis traces cusps, pushed back loops, "
            "pushed along waves.",
            "Upright, it sleeps if I₃²ω₃² &gt; 4I₁mgl.",
            "Euler angles lock upright; a quaternion and L, stepped by an exact split, don't.",
        ),
    )
)


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
        self.reach = 1.0  # the drawn axis's length (m)
        self.turning: tuple[float, float] = (0.0, 0.0)
        self.show_vectors = True
        self._last: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._trail = np.zeros((0, 3))

        plot = self.getPlotItem()
        self.table = QtWidgets.QGraphicsPathItem()
        self.vertical = pg.PlotCurveItem()
        self.circles = pg.PlotCurveItem(connect="finite")  # the turning points' circles
        self.trail = pg.PlotCurveItem()
        self.stem = pg.PlotCurveItem()
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
            self.stem,
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

    def set_top(self, top: Top, turning: tuple[float, float]) -> None:
        """A new top or launch: its size sets the view's, and its turning points the circles."""
        self.top, self.turning = top, turning
        self.reach = 1.6 * max(top.height, top.radius)
        self._trail = np.zeros((0, 3))
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
        cx, cy = self.project(np.concatenate(lines))
        self.circles.setData(cx, cy)
        span = 1.25 * R
        self.setRange(xRange=(-span, span), yRange=(-0.6 * span, 1.2 * span), padding=0)
        if self._last is not None:
            self.draw(*self._last, self._trail)

    def draw(self, axis: np.ndarray, mark: np.ndarray, momentum: np.ndarray, trail: np.ndarray) -> None:
        """The top with its axis along `axis` (a unit vector), the disc's mark toward `mark`, and L; `trail` is
        where the axis has pointed (n, 3)."""
        self._last, self._trail = (axis, mark, momentum), trail
        top, R = self.top, self.reach
        centre = top.height * axis
        side = np.cross(axis, mark)
        ring = np.linspace(0, 2 * math.pi, self.RIM + 1)[:, None]
        rim = centre + top.radius * (np.cos(ring) * mark + np.sin(ring) * side)
        rx, ry = self.project(rim)
        path = QtGui.QPainterPath()
        path.moveTo(float(rx[0]), float(ry[0]))
        for a, b in zip(rx[1:], ry[1:]):
            path.lineTo(float(a), float(b))
        self.disc.setPath(path)
        self.rim.setData(rx, ry)
        sx, sy = self.project(np.array([centre, centre + top.radius * mark]))
        self.spoke.setData(sx, sy)
        tx, ty = self.project(np.array([[0.0, 0.0, 0.0], centre]))
        self.stem.setData(tx, ty)
        ax, ay = self.project(np.array([centre, R * axis]))
        self.axis_line.setData(ax, ay)
        ex, ey = self.project(R * axis)
        self.end.setData([ex], [ey])
        if trail.size:
            px, py = self.project(R * trail)
            self.trail.setData(px, py)
        else:
            self.trail.setData([], [])
        self._draw_vectors(axis, momentum)

    def _draw_vectors(self, axis: np.ndarray, momentum: np.ndarray) -> None:
        top, R = self.top, self.reach
        size = float(np.linalg.norm(momentum))
        if not self.show_vectors or size == 0:
            for item, label in zip(self.vectors, self.labels):
                item.setData([], [])
                label.hide()
            return
        along = float(momentum @ axis)
        omega = momentum / top.transverse + (1 / top.polar - 1 / top.transverse) * along * axis
        torque = -top.weight_moment * np.cross(axis, [0.0, 0.0, 1.0])
        for item, label, v, length in zip(
            self.vectors, self.labels, (momentum, omega, torque), (1.25 * R, 1.05 * R, 0.6 * R)
        ):
            n = float(np.linalg.norm(v))
            if n < 1e-12:
                item.setData([], [])
                label.hide()
                continue
            tip = length * v / n
            x, y = self.project(np.array([[0.0, 0.0, 0.0], tip]))
            item.setData(x, y)
            lx, ly = self.project(1.1 * tip)
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
        self.stem.setPen(pg.mkPen(c.strong, width=3))
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


class GyroscopePage(QtWidgets.QWidget):
    """Controls on the left, the top and its tilt in the centre, the closed forms and the check on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.motion = TRIALS[0].motion
        self.sim = TopSimulator(self.motion)
        self.history = SampleBuffer(10, HISTORY)  # axis, mark, energy, L
        self.running = True
        self._frames = 0

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Each sets up the top and lets it go, to show one idea from the Theory tab.</p>",
        )
        self.trials.picked.connect(self.load_trial)

        body = Form("Top")
        self.mass = body.number("Mass m:", 0.01, 100.0, 0.5, 3, " kg", TIPS["mass"])
        self.radius = body.number("Disc radius r:", 1.0, 500.0, 40.0, 1, " mm", TIPS["radius"])
        self.height = body.number("Height l:", 1.0, 500.0, 30.0, 1, " mm", TIPS["height"])
        self.gravity = body.number("Gravity g:", 0.0, 100.0, 9.81, 2, " m/s²", TIPS["gravity"])
        self.info = note_label()
        body.row(self.info)

        release = Form("Let go")
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
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.clicked.connect(self._on_pause)
        self.restart_button = QtWidgets.QPushButton("Let go again")
        self.restart_button.clicked.connect(self.restart)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.restart_button)
        play.row(buttons)
        self.vectors = QtWidgets.QCheckBox("Vectors L, ω, τ")
        self.vectors.setChecked(True)
        self.vectors.toggled.connect(self._on_vectors)
        play.row(self.vectors)

        connect_all(
            (self.mass, self.radius, self.height, self.gravity, self.spin, self.tilt, self.launch, self.rate),
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
        for item in (*self.potential_turns, self.energy_line, self.potential_item):
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
        self.check_note = note_label()

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(column(self.potential_plot, self.potential_note, stretch=False), "Potential")
        self.tabs.addTab(column(self.precession_plot, self.precession_note, stretch=False), "Precession")
        self.tabs.addTab(column(self.path_plot, self.path_note, stretch=False), "Path")
        self.tabs.addTab(column(self.check_plot, self.check_note, stretch=False), "Check")
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
            spin_rpm=self.spin.value(),
            tilt=self.tilt.value(),
            launch=LAUNCHES[self.launch.currentIndex()],
            precession=self.rate.value(),
            playback=self.playback.currentIndex(),
            tab=self.tabs.currentIndex(),
        )

    def apply_trial(self, t: GyroTrial) -> None:
        for box, value in (
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
        self.running = True
        self._on_inputs()

    def _on_inputs(self) -> None:
        state = self.practice_state()
        self.rate.setEnabled(state.launch is Launch.SET)
        self.motion = state.motion
        top = self.motion.top
        self.info.setText(
            f"I₁ = {top.transverse * 1e4:.4g} kg·cm², I₃ = {top.polar * 1e4:.4g} kg·cm², mgl = "
            f"{fmt(top.weight_moment, 'N·m')}. It sleeps above {top.sleeping_spin / RPM:.4g} rpm."
            if top.weight_moment > 0
            else f"I₁ = {top.transverse * 1e4:.4g} kg·cm², I₃ = {top.polar * 1e4:.4g} kg·cm²; no gravity: a free top."
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

    def restart(self) -> None:
        """Let the top go again from the launch set."""
        self.sim = TopSimulator(self.motion)
        self.history.clear()
        self.view.set_top(self.motion.top, self.motion.turning())
        for line, th in zip(self.turn_lines, self.motion.turning()):
            line.setPos(math.degrees(th))
        self._show_closed_forms()
        self._draw_tab()
        self._draw_now()
        self._update_buttons()
        self._start_timer()

    def _on_vectors(self, on: bool) -> None:
        self.view.show_vectors = on
        self._draw_now()

    def _on_pause(self) -> None:
        self.running = not self.running
        self._update_buttons()
        self._start_timer()

    def _update_buttons(self) -> None:
        self.pause_button.setText("Pause" if self.running else "Run")

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
        return f"{self.spin.value():g} rpm, {self.tilt.value():g}°, {self.launch.currentText()}"

    def step(self, duration: float) -> None:
        """Run the top on by `duration` s of simulated time and redraw."""
        s = self.sim.advance(duration)
        if s.t.size:
            self.history.extend(s.t, np.concatenate([s.axis, s.mark, s.energy[:, None], s.momentum], axis=1))
        self._draw_now()
        self._draw_live_tab()

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.view.apply_theme()
        self.tilt_item.setPen(pg.mkPen(c.mass[0], width=2))
        for line in (*self.turn_lines, *self.potential_turns):
            line.setPen(pg.mkPen(c.fit, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.potential_item.setPen(pg.mkPen(c.mass[0], width=2))
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

    def _draw_now(self) -> None:
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
        if n:
            mean = abs(self.motion.mean_precession())
            span = min(2 * math.pi / mean, 20.0) if mean > 1e-9 else 3 * self.motion.nutation_period()
            t, y = self.history.window(span)
            trail = y[_thin(t.size, TRAIL_POINTS), 0:3]
        self.view.draw(axis, mark_, momentum, trail)
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

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
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
        if i == TAB_POTENTIAL:
            self._potential_dot()
        elif i == TAB_PATH:
            self._path_trace()
        elif i == TAB_CHECK:
            self._frames += 1
            self._draw_check(text=self._frames % CHECK_FRAMES == 0)

    def _draw_potential(self) -> None:
        m = self.motion
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
        _, y = self.history.window(min(4 * math.pi / mean, 40.0) if mean > 1e-9 else 6 * self.motion.nutation_period())
        k = _thin(len(y), PATH_POINTS)
        self.path_item.setData(y[k, 0], y[k, 1])
        self.path_now.setData([y[-1, 0]], [y[-1, 1]])

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
        drift = np.maximum(np.abs(y[k, 9] - m.vertical_momentum), np.abs(along - m.spin_momentum)) / scale_l
        self.momentum_item.setData(t[k], drift)
        if not text:
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
