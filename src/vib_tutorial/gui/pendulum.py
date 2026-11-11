"""Pendulums page: oscillators whose stiffness is gravity, and the rotational form of every one-mass formula.

Left: the pendulum (simple, a rod on a pivot, a disc on a wire, or inverted on a spring), its size, the spring,
gravity and damping, and the angle it is let go from. Centre: the swing, whole and linearized, and the restoring
moment against the angle. Right: the period against the amplitude (exact, Duffing's approximation and linear), the
phase plane's energy curves, the rod's period against where it is pivoted, and the Theory. The page has its own
model (core/pendulum.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.pendulum import (
    KINDS,
    Kind,
    Pendulum,
    conjugate_pivot,
    duffing_period,
    measured_period,
    orbit,
    period,
    periods,
    phase_curve,
    rod_period,
    swing,
)
from ..core.practice import Problem, Question, Requirement, kept, number, pick, rounded
from . import pendulum_figures, settle_plots
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
from .theming import add_legend

TAB_PERIOD, TAB_PHASE, TAB_PIVOT = range(3)
SWINGS = 8  # linear periods drawn


@dataclass(frozen=True)
class PendulumTrial:
    name: str
    note: str
    kind: Kind = Kind.SIMPLE
    mass: float = 1.0
    length: float = 1.0
    pivot: float = 0.5
    spring: float = 15.0
    gravity: float = 9.81
    zeta: float = 0.0
    angle: float = 10.0  # degrees
    tab: int = TAB_PERIOD


TRIALS = (
    PendulumTrial(
        "Gravity is the spring",
        "A 1 kg bob on a 1 m string, let go from 10°. Its restoring moment is <m>mgL sin θ ≈ mgLθ</m>, and its "
        "moment of inertia <m>mL²</m>, so <m>ω<sub>n</sub> = √(g/L)</m>: 0.498 Hz, a period of 2.006 s, whatever "
        "the mass (change it). At 10° the whole and the linearized swings can't be told apart.",
        angle=10.0,
    ),
    PendulumTrial(
        "Large swings are slower",
        "The same pendulum from 90°. <m>sin θ</m> falls below <m>θ</m>, so the spring softens as the swing grows: "
        "the period is 2.368 s, 18 % longer than the small swing's, and the linearized swing drifts ahead "
        "(Centre). The Period tab gives the exact period (an elliptic integral), and Duffing's first "
        "approximation, <m>T ≈ T<sub>0</sub>(1 + A²/16)</m>, which holds to 1 % up to about 75°.",
        angle=90.0,
    ),
    PendulumTrial(
        "Nearly over the top",
        "From 170°, the bob nearly balances upside down and lingers there: the period is 2.44 times the small "
        "swing's, and it grows without bound as the start nears 180°. The Phase plane tab shows why: the orbit "
        "runs close to the separatrix, the curve through the upside-down balance that divides swinging from "
        "going round.",
        angle=170.0,
        tab=TAB_PHASE,
    ),
    PendulumTrial(
        "A rod: where to pivot it",
        "A 1 m uniform rod pivoted 0.5 m from its centre (at its end): <m>J = mL²/3</m>, the weight's arm "
        "<m>d</m> = 0.5 m, so it swings as a simple pendulum <m>L<sub>eq</sub></m> = 0.667 m long (1.638 s). "
        "Pivoted 0.289 m from the centre (<m>L/√12</m>, the radius of gyration) it swings fastest, 1.524 s; "
        "pivoted 0.167 m from the centre (<m>k<sub>G</sub>²/d</m>) it swings at 1.638 s again: Kater's "
        "reversible pendulum, which measured <m>g</m>.",
        kind=Kind.ROD,
        angle=10.0,
        tab=TAB_PIVOT,
    ),
    PendulumTrial(
        "A disc on a wire: no softening",
        "A 1 kg disc of radius 0.1 m on a wire of torsional stiffness <m>k<sub>t</sub></m> = 0.5 N·m/rad: "
        "<m>J = mr²/2</m>, <m>f<sub>n</sub> = √(k<sub>t</sub>/J)/2π</m> = 1.59 Hz. The wire's twist is linear, "
        "so the period is the same at 10° as at 170°: the softening above came from gravity's <m>sin θ</m>, not "
        "from turning.",
        kind=Kind.TORSION,
        length=0.1,
        spring=0.5,
        angle=120.0,
    ),
    PendulumTrial(
        "Inverted: gravity as a negative spring",
        "A 1 kg mass on a 1 m arm balanced upright on a 15 N·m/rad spring. Gravity now pushes it over: the "
        "stiffness is <m>k<sub>θ</sub> = k<sub>t</sub> − mgL</m> = 5.19 N·m/rad, <m>f<sub>n</sub></m> = 0.363 Hz. "
        "And it hardens: from 60° the period is 10 % shorter than the small swing's.",
        kind=Kind.INVERTED,
        spring=15.0,
        angle=60.0,
    ),
    PendulumTrial(
        "Inverted: buckling",
        "The spring at 9 N·m/rad, under <m>mgL</m> = 9.81 N·m/rad: the stiffness is negative and upright is "
        "unstable. Let go from 1° with a little damping, it falls over and settles leaning at 40.8°, where the "
        "spring's <m>9θ</m> balances the weight's <m>9.81 sin θ</m>. A column under a load past its buckling "
        "load does the same.",
        kind=Kind.INVERTED,
        spring=9.0,
        angle=1.0,
        zeta=0.1,
    ),
    PendulumTrial(
        "On the Moon",
        "The 1 m pendulum where <m>g</m> = 1.62 m/s²: <m>√(9.81/1.62)</m> = 2.46 times slower, a period of "
        "4.94 s. A pendulum clock is a gravimeter.",
        gravity=1.62,
        angle=10.0,
    ),
)

# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


def _simple_period(rng: np.random.Generator) -> Question:
    length = pick(rng, (0.25, 0.5, 1, 2, 5, 10))
    p = Pendulum(Kind.SIMPLE, length=length)
    return Question(
        f"A small bob hangs on a {length:g} m string (g = 9.81 m/s²). What is its period for small swings?",
        p.period0,
        f"Gravity is the spring: k<sub>θ</sub> = mgL, J = mL², so ω<sub>n</sub> = √(g/L) = "
        f"{number(p.omega0)} rad/s and T = 2π/ω<sub>n</sub> = {number(p.period0)} s, whatever the bob weighs.",
        unit="s",
        setup=dict(kind=Kind.SIMPLE, length=length, angle=5.0),
    )


def _rod_period(rng: np.random.Generator) -> Question:
    length = pick(rng, (0.5, 1, 1.5, 2))
    d = rounded(pick(rng, (0.1, 0.2, 0.3, 0.4, 0.5)) * length)
    p = Pendulum(Kind.ROD, length=length, pivot=d)
    return Question(
        f"A uniform rod {length:g} m long swings about a pivot {number(d)} m from its centre. What is its period "
        "for small swings?",
        p.period0,
        f"J = m(L²/12 + d²) about the pivot (parallel axes) and k<sub>θ</sub> = mgd: T = 2π√((L²/12 + d²)/gd) = "
        f"2π√(({number(length**2 / 12)} + {number(d * d)})/(9.81 × {number(d)})) = {number(p.period0)} s, the mass "
        f"cancelling again. It swings like a simple pendulum {number(p.equivalent_length)} m long.",
        unit="s",
        setup=dict(kind=Kind.ROD, length=length, pivot=d, tab=TAB_PIVOT),
    )


def _large_swing(rng: np.random.Generator) -> Question:
    degrees = pick(rng, (20, 30, 45, 60, 90, 120))
    p = Pendulum(Kind.SIMPLE)
    longer = 100.0 * (period(p, math.radians(degrees)) / p.period0 - 1.0)
    a = math.radians(degrees)
    return Question(
        f"A simple pendulum is let go from {degrees:g}° instead of a small angle. By what percentage is its period "
        "longer?",
        longer,
        f"Exactly, T = T<sub>0</sub>(2/π)K(sin²(A/2)), the energy integral: {number(longer)} % longer. The "
        f"Duffing estimate A²/16 = {number(100 * a * a / 16)} % is good to about 30°, and falls short beyond.",
        unit="%",
        tolerance=0.03,
        setup=dict(kind=Kind.SIMPLE, angle=float(degrees), tab=TAB_PERIOD),
    )


def _inverted_spring(rng: np.random.Generator) -> Question:
    m = pick(rng, (0.5, 1, 2, 5))
    length = pick(rng, (0.2, 0.5, 1))
    least = m * 9.81 * length
    return Question(
        f"A {m:g} kg mass on a light arm {length:g} m long stands upright on a pivot, held up by a torsion spring. "
        "How stiff must the spring be for upright to be stable?",
        least,
        f"Leaning by θ, the weight pushes it over with mgL sin θ ≈ mgLθ, so the stiffness is k<sub>t</sub> − mgL: "
        f"stable only if k<sub>t</sub> &gt; mgL = {m:g} × 9.81 × {length:g} = {number(least)} N·m/rad. A weaker "
        "spring lets it fall to a leaning balance, as a column buckles.",
        unit="N·m/rad",
        setup=dict(kind=Kind.INVERTED, mass=m, length=length, spring=rounded(1.2 * least)),
    )


def _disc_on_wire(rng: np.random.Generator) -> Question:
    m = pick(rng, (0.5, 1, 2, 5))
    r = pick(rng, (0.05, 0.1, 0.2))
    k = pick(rng, (0.5, 1, 2, 5))
    p = Pendulum(Kind.TORSION, mass=m, length=r, spring=k)
    return Question(
        f"A {m:g} kg disc of radius {r:g} m hangs on a wire of torsional stiffness {k:g} N·m/rad. At what frequency "
        "does it twist back and forth?",
        p.natural_hz,
        f"J = mr²/2 = {number(p.inertia)} kg·m², so ω<sub>n</sub> = √(k<sub>t</sub>/J) = {number(p.omega0)} rad/s "
        f"and f<sub>n</sub> = {number(p.natural_hz)} Hz: the single oscillator, turned.",
        unit="Hz",
        setup=dict(kind=Kind.TORSION, mass=m, length=r, spring=k),
    )


def _end_pivot(rng: np.random.Generator) -> Question:
    length = pick(rng, (0.3, 0.6, 0.9, 1.2, 1.5))
    p = Pendulum(Kind.ROD, length=length, pivot=length / 2)
    return Question(
        f"A uniform rod {length:g} m long swings about one end. How long a simple pendulum has the same period?",
        p.equivalent_length,
        f"L<sub>eq</sub> = J/md with J = m(L²/12 + L²/4) = mL²/3 and d = L/2: L<sub>eq</sub> = 2L/3 = "
        f"{number(p.equivalent_length)} m.",
        unit="m",
        setup=dict(kind=Kind.ROD, length=length, pivot=length / 2, tab=TAB_PIVOT),
    )


def _heavier_bob(rng: np.random.Generator) -> Question:
    n = int(pick(rng, (2, 4, 10)))
    return Question(
        f"A simple pendulum's bob is swapped for one {n} times heavier, on the same string. What happens to its "
        "period?",
        0,
        "Nothing: the restoring moment mgL and the inertia mL² grow together, so ω<sub>n</sub> = √(g/L) has no "
        "mass in it.",
        choices=("It stays the same", f"It grows √{n} times", f"It falls to 1/√{n}"),
        setup=dict(kind=Kind.SIMPLE, mass=float(n)),
        predict=True,
    )


def _on_the_moon(rng: np.random.Generator) -> Question:
    return Question(
        "A pendulum clock is taken to the Moon, where gravity is about a sixth of Earth's. What happens to its "
        "pendulum's period?",
        0,
        "It grows √6 ≈ 2.4 times: gravity is the spring, ω<sub>n</sub> = √(g/L). (A disc on a torsion wire "
        "wouldn't change: its spring isn't gravity.)",
        choices=("It grows about 2.4 times", "It grows 6 times", "It stays the same", "It falls to a sixth"),
        setup=dict(kind=Kind.SIMPLE, gravity=1.62),
        predict=True,
    )


def _wire_large_twist(rng: np.random.Generator) -> Question:
    return Question(
        "A disc on a torsion wire is twisted to 90° and let go, instead of 10°. What happens to its period?",
        0,
        "Nothing: a wire's moment is k<sub>t</sub>θ at any twist, so it is linear and its period doesn't depend "
        "on the swing. Only gravity's sin θ makes a pendulum's lengthen.",
        choices=("It stays the same", "It lengthens, as a pendulum's does", "It shortens"),
        setup=dict(kind=Kind.TORSION, angle=90.0, tab=TAB_PERIOD),
        predict=True,
    )


def _pivot_spec(rng: np.random.Generator) -> Question:
    length = pick(rng, (0.5, 1, 1.5, 2))
    d = rounded(pick(rng, (0.12, 0.18, 0.4, 0.45)) * length)
    target = rounded(Pendulum(Kind.ROD, length=length, pivot=d).period0)
    other = conjugate_pivot(length, d)

    def goal(t: PendulumTrial):
        now = Pendulum(t.kind, t.mass, t.length, max(1e-3, t.pivot), t.spring, t.zeta, t.gravity).period0
        return (
            Requirement("A rod on a pivot", t.kind.value, t.kind is Kind.ROD),
            kept("Its length", t.length, length, "m"),
            kept("Gravity", t.gravity, 9.81, "m/s²"),
            Requirement(
                f"Its period within ±0.5 % of {number(target)} s",
                f"{number(now, 4)} s",
                t.kind is Kind.ROD and abs(now / target - 1.0) <= 0.005,
            ),
            Requirement(
                f"The pivot on the rod (≤ {number(length / 2)} m from its centre)",
                f"{number(t.pivot)} m",
                t.pivot <= length / 2,
            ),
        )

    return Question(
        f"A uniform rod {length:g} m long is to be a clock's pendulum with a period of {number(target)} s. Where "
        "along it should the pivot go?",
        0,
        f"T = 2π√((L²/12 + d²)/gd) is least at d = L/√12 = {number(length / math.sqrt(12))} m and longer either "
        f"side, so a longer period is reached at two pivots, d and k<sub>G</sub>²/d: here {number(d)} m and "
        f"{number(other)} m from the centre"
        + ("" if other <= length / 2 else " (off the rod, so only the first)")
        + ". That is Kater's reversible pendulum: the two pivots are L<sub>eq</sub> apart.",
        setup=dict(kind=Kind.ROD, length=length, pivot=rounded(length / math.sqrt(12)), tab=TAB_PIVOT),
        goal=goal,
        design=dict(kind=Kind.ROD, length=length, pivot=d, tab=TAB_PIVOT),
    )


PROBLEMS = tuple(
    Problem(key, title, "pendulum_page", f"See {page_link('pendulum_page', anchor, section)}{more}.", make)
    for key, title, anchor, section, more, make in (
        ("simple-period", "A simple pendulum", "gravity-stiffness", "Gravity as a spring", "", _simple_period),
        ("rod-period", "A rod on a pivot", "gravity-stiffness", "Gravity as a spring", ": parallel axes", _rod_period),
        ("large-swing", "A large swing", "large-swings", "Large swings", "", _large_swing),
        ("inverted-spring", "Hold it upright", "gravity-stiffness", "Gravity as a spring", "", _inverted_spring),
        ("disc-on-wire", "A disc on a wire", "rotation", "Turning instead of sliding", "", _disc_on_wire),
        ("end-pivot", "Equivalent length", "gravity-stiffness", "Gravity as a spring", "", _end_pivot),
        ("heavier-bob", "Predict: a heavier bob", "gravity-stiffness", "Gravity as a spring", "", _heavier_bob),
        ("on-the-moon", "Predict: on the Moon", "gravity-stiffness", "Gravity as a spring", "", _on_the_moon),
        ("wire-large-twist", "Predict: a large twist", "large-swings", "Large swings", "", _wire_large_twist),
        (
            "pivot-spec",
            "Spec: a clock's pendulum",
            "gravity-stiffness",
            "Gravity as a spring",
            ": where is T least?",
            _pivot_spec,
        ),
    )
)

TIPS = {
    "kind": "<p>Simple: a point mass on a massless string. Rod: a uniform rod on a pivot. Disc: a disc twisting on "
    "a wire (no gravity in it). Inverted: a mass on an upright arm held by a torsion spring at its foot.</p>",
    "mass": "<p>The bob's, the rod's or the disc's mass.</p>",
    "length": "<p>The string's length, the rod's length, the disc's radius or the inverted arm.</p>",
    "pivot": "<p>How far from the rod's centre the pivot is (0.5 L: at its end).</p>",
    "spring": "<p>The torsion spring: the wire's twist stiffness, or the inverted pendulum's spring at its foot.</p>",
    "gravity": "<p>g: 9.81 m/s² on Earth, 1.62 on the Moon.</p>",
    "zeta": "<p>A viscous damper at the pivot, as the linearized oscillator's damping ratio.</p>",
    "angle": "<p>The angle it is let go from, at rest.</p>",
}
SWING_TIP = "<p>The angle from release: the whole equation (solid) and the linearized one, sin θ ≈ θ (dashed).</p>"
MOMENT_TIP = (
    "<p>The restoring moment against the angle: the whole one (solid) and its tangent at 0, the "
    "linearized stiffness (dashed). Dotted: the swing's range.</p>"
)
PERIOD_TIP = (
    "<p>The undamped period over the small-swing one against the amplitude: exact (by quadrature), "
    "Duffing's first harmonic balance, and linear (1).</p>"
)
PHASE_TIP = (
    "<p>The undamped orbits (energy curves) in the (θ, θ̇) plane, the separatrix bold where there is one, "
    "and the swing from the release (colour).</p>"
)
PIVOT_TIP = (
    "<p>A uniform rod of this length: its period against the pivot's distance from its centre. The "
    "fastest is at the radius of gyration; every other period is reached at two distances.</p>"
)

NOTES_HTML = load_notes("pendulum", figures=pendulum_figures.FIGURES, trials=trials_html(TRIALS))


class PendulumPage(QtWidgets.QWidget):
    """Controls on the left, the swing and the moment in the centre, period, phase plane and pivot on the
    right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.pendulum = Pendulum()

        self.trials = TrialBox(
            [t.name for t in TRIALS], "<p>Each sets up the pendulum to show one idea from the Theory tab.</p>"
        )
        self.trials.picked.connect(self.load_trial)

        body = Form("Pendulum")
        self.kind = combo([k.value for k in KINDS], TIPS["kind"])
        body.add("Kind:", self.kind, TIPS["kind"])
        self.mass = body.number("Mass m:", 0.001, 1e4, 1.0, 3, " kg", TIPS["mass"])
        self.length = body.number("Length L:", 0.01, 100.0, 1.0, 3, " m", TIPS["length"])
        self.pivot = body.number("Pivot from centre:", 0.001, 50.0, 0.5, 3, " m", TIPS["pivot"])
        self.spring = body.number("Torsion spring:", 0.0, 1e6, 15.0, 3, " N·m/rad", TIPS["spring"])
        self.gravity = body.number("Gravity g:", 0.0, 100.0, 9.81, 2, " m/s²", TIPS["gravity"])
        self.zeta = body.number("Damping ζ:", 0.0, 2.0, 0.0, 3, "", TIPS["zeta"])
        self.info = note_label()
        body.row(self.info)

        release = Form("Release")
        self.angle = release.number("Let go from:", 0.1, 179.0, 10.0, 1, "°", TIPS["angle"])

        connect_all(
            (self.kind, self.mass, self.length, self.pivot, self.spring, self.gravity, self.zeta, self.angle),
            self._on_inputs,
        )

        # --- centre
        self.swing_plot = plot("Angle θ", None, "Time", "s", height=150)
        self.swing_plot.setToolTip(SWING_TIP)
        add_legend(self.swing_plot.getPlotItem(), offset=(-10, 10))
        self.whole_item = pg.PlotDataItem(name="whole (sin θ)")
        self.linear_item = pg.PlotDataItem(name="linearized (θ)")
        for item in (self.linear_item, self.whole_item):
            self.swing_plot.addItem(item)
        self.moment_plot = plot("Restoring moment", "N·m", "Angle θ", None, height=130)
        self.moment_plot.setToolTip(MOMENT_TIP)
        self.moment_item = pg.PlotDataItem()
        self.tangent_item = pg.PlotDataItem()
        self.range_lines = [pg.InfiniteLine(angle=90), pg.InfiniteLine(angle=90)]
        for item in (*self.range_lines, self.tangent_item, self.moment_item):
            self.moment_plot.addItem(item)
        self.readout = readout_label()

        # --- right
        self.period_plot = plot("Period T / T₀", None, "Amplitude", None, height=200)
        self.period_plot.setToolTip(PERIOD_TIP)
        add_legend(self.period_plot.getPlotItem(), offset=(10, 10))
        self.exact_item = pg.PlotDataItem(name="exact")
        self.duffing_item = pg.PlotDataItem(name="Duffing: 1st harmonic")
        self.unit_item = pg.PlotDataItem(name="linear")
        self.period_now = marker(12)
        for item in (self.unit_item, self.duffing_item, self.exact_item, self.period_now):
            self.period_plot.addItem(item)
        self.period_note = note_label()

        self.phase_plot = plot("θ̇", None, "θ", None, height=200)
        self.phase_plot.setToolTip(PHASE_TIP)
        self.orbit_items: list[pg.PlotDataItem] = []
        self.separatrix = pg.PlotDataItem()
        self.trace_item = pg.PlotDataItem()
        self.start_dot = marker(10)
        for item in (self.separatrix, self.trace_item, self.start_dot):
            self.phase_plot.addItem(item)
        self.phase_note = note_label()

        self.pivot_plot = plot("Period", "s", "Pivot from the centre", "m", height=200)
        self.pivot_plot.setToolTip(PIVOT_TIP)
        self.pivot_item = pg.PlotDataItem()
        self.pivot_best = marker(10)
        self.pivot_now = marker(12)
        self.pivot_twin = marker(12, symbol="s")
        self.pivot_level = pg.InfiniteLine(angle=0)
        for item in (self.pivot_level, self.pivot_item, self.pivot_best, self.pivot_twin, self.pivot_now):
            self.pivot_plot.addItem(item)
        self.pivot_note = note_label()

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(column(self.period_plot, self.period_note, stretch=False), "Period")
        self.tabs.addTab(column(self.phase_plot, self.phase_note, stretch=False), "Phase plane")
        self.tabs.addTab(column(self.pivot_plot, self.pivot_note, stretch=False), "Pivot")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._draw_tab)

        three_panes(
            self,
            column(self.trials, body, release),
            stack((self.swing_plot, 3), (self.moment_plot, 2), (self.readout, 0)),
            self.tabs,
        )
        self._on_inputs()
        self.apply_theme()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(PendulumTrial("", "", **setup))

    def practice_state(self) -> PendulumTrial:
        """The page's settings, as a trial: what a practice spec checks."""
        return PendulumTrial(
            "",
            "",
            kind=KINDS[self.kind.currentIndex()],
            mass=self.mass.value(),
            length=self.length.value(),
            pivot=self.pivot.value(),
            spring=self.spring.value(),
            gravity=self.gravity.value(),
            zeta=self.zeta.value(),
            angle=self.angle.value(),
            tab=self.tabs.currentIndex(),
        )

    def apply_trial(self, t: PendulumTrial) -> None:
        for box, value in (
            (self.kind, KINDS.index(t.kind)),
            (self.mass, t.mass),
            (self.length, t.length),
            (self.pivot, t.pivot),
            (self.spring, t.spring),
            (self.gravity, t.gravity),
            (self.zeta, t.zeta),
            (self.angle, t.angle),
        ):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self._on_inputs()

    def _on_inputs(self) -> None:
        kind = KINDS[self.kind.currentIndex()]
        self.pivot.setEnabled(kind is Kind.ROD)
        self.spring.setEnabled(kind in (Kind.TORSION, Kind.INVERTED))
        self.gravity.setEnabled(kind is not Kind.TORSION)
        self.pendulum = Pendulum(
            kind,
            self.mass.value(),
            self.length.value(),
            max(1e-3, self.pivot.value()),
            self.spring.value(),
            self.zeta.value(),
            self.gravity.value(),
        )
        p = self.pendulum
        if p.stable:
            extra = f", L<sub>eq</sub> = {fmt(p.equivalent_length, 'm')}" if kind in (Kind.SIMPLE, Kind.ROD) else ""
            self.info.setText(
                f"J = {p.inertia:.4g} kg·m², k<sub>θ</sub> = {p.stiffness:.4g} N·m/rad: "
                f"f<sub>n</sub> = {p.natural_hz:.4g} Hz, T₀ = {p.period0:.4g} s{extra}."
            )
        elif p.stiffness == 0:
            self.info.setText(f"J = {p.inertia:.4g} kg·m², k<sub>θ</sub> = 0: no restoring moment.")
        else:
            self.info.setText(
                f"J = {p.inertia:.4g} kg·m², k<sub>θ</sub> = {p.stiffness:.4g} N·m/rad: negative, "
                "so it can't stay at θ = 0."
            )
        self.redraw()

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.whole_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.linear_item.setPen(pg.mkPen(c.grey, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.moment_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.tangent_item.setPen(pg.mkPen(c.grey, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        for line in self.range_lines:
            line.setPen(pg.mkPen(c.fit, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.exact_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.duffing_item.setPen(pg.mkPen(c.force, width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.unit_item.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DotLine))
        mark(self.period_now, c.strong, filled=True)
        self.separatrix.setPen(pg.mkPen(c.strong, width=2))
        self.trace_item.setPen(pg.mkPen(c.mass[0], width=2))
        mark(self.start_dot, c.mass[0], filled=True)
        for item in self.orbit_items:
            item.setPen(pg.mkPen(c.grey, width=1))
        self.pivot_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.pivot_level.setPen(pg.mkPen(c.fit, width=1, style=QtCore.Qt.PenStyle.DotLine))
        mark(self.pivot_best, c.grey, filled=True)
        mark(self.pivot_now, c.strong, filled=True)
        mark(self.pivot_twin, c.force)
        self.redraw()

    def redraw(self) -> None:
        self._draw_swing()
        self._draw_tab()
        settle_plots(self)

    @property
    def theta0(self) -> float:
        return math.radians(self.angle.value())

    def _duration(self) -> float:
        p = self.pendulum
        if p.stable:
            return SWINGS * p.period0
        return 10.0 * math.tau / math.sqrt(abs(p.stiffness) / p.inertia) if p.stiffness else 10.0

    def _draw_swing(self) -> None:
        p, a = self.pendulum, self.theta0
        duration = self._duration()
        t, th, tl = swing(p, a, duration, 2400)
        lim = 4 * math.pi
        self.whole_item.setData(t, np.degrees(np.clip(th, -lim, lim)))
        self.linear_item.setData(t, np.degrees(np.clip(tl, -lim, lim)))
        self.swing_plot.setXRange(0.0, duration, padding=0)
        if p.stable:
            top = float(np.nanmax(np.abs(np.degrees(np.clip(th, -lim, lim)))))
            span = max(top, self.angle.value()) * 1.1
            self.swing_plot.setYRange(-span, span, padding=0)
        else:  # the linearized one runs away: range on the whole one
            whole = np.degrees(th[np.isfinite(th)])
            pad = 0.1 * max(float(np.ptp(whole)), 1.0)
            self.swing_plot.setYRange(float(whole.min()) - pad, float(whole.max()) + pad, padding=0)
        self.swing_plot.setLabel("left", "Angle θ (°)")

        reach = max(a, abs(p.leaning())) * 1.3
        reach = min(max(reach, 0.2), math.pi)
        x = np.linspace(-reach, reach, 400)
        self.moment_item.setData(np.degrees(x), p.moment(x))
        self.tangent_item.setData(np.degrees(x), p.stiffness * x)
        for line, v in zip(self.range_lines, (-a, a)):
            line.setPos(math.degrees(v))
        self.moment_plot.setXRange(-math.degrees(reach), math.degrees(reach), padding=0)
        self.moment_plot.setLabel("bottom", "Angle θ (°)")

        if p.stable:
            T = period(p, a) if p.zeta == 0 else math.nan
            measured = measured_period(t, th)
            parts = [f"Small-swing period T₀ = {p.period0:.4g} s."]
            if math.isfinite(T):
                duff = duffing_period(p, a) / p.period0 - 1
                guess = (
                    f" (Duffing's estimate {100 * duff:+.3g} %)"
                    if math.isfinite(duff)
                    else " (Duffing's estimate has no answer this far out)"
                )
                parts.append(
                    f"From {self.angle.value():.3g}° the exact period is {T:.4g} s, "
                    f"{100 * (T / p.period0 - 1):+.3g} %{guess}."
                )
            elif math.isfinite(measured):
                parts.append(
                    f"The swing's zero crossings come {measured:.4g} s apart (damped, and the swing shrinking)."
                )
            self.readout.setText(" ".join(parts))
        elif p.leaning():
            self.readout.setText(
                f"Unstable upright: it falls, and with damping settles leaning at "
                f"±{math.degrees(p.leaning()):.3g}° (k<sub>t</sub> θ = mgd sin θ). The linearized "
                "model just runs away."
            )
        else:
            self.readout.setText("No restoring moment at θ = 0: nothing to oscillate about.")

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_PERIOD:
            self._draw_period()
        elif i == TAB_PHASE:
            self._draw_phase()
        elif i == TAB_PIVOT:
            self._draw_pivot()

    def _draw_period(self) -> None:
        p = self.pendulum
        if not p.stable:
            for item in (self.exact_item, self.duffing_item, self.unit_item, self.period_now):
                item.setData([], [])
            self.period_note.setText("Upright is unstable: there is no swing about it to time.")
            return
        deg = np.linspace(0.5, 179.0, 180)
        a = np.radians(deg)
        exact = periods(p, a) / p.period0
        duff = np.array([duffing_period(p, float(x)) for x in a]) / p.period0
        self.exact_item.setData(deg, exact, connect="finite")
        self.duffing_item.setData(deg, duff, connect="finite")
        self.unit_item.setData([0, 180], [1, 1])
        now = period(p, self.theta0) / p.period0
        self.period_now.setData([self.angle.value()], [now] if math.isfinite(now) else [np.nan])
        finite = np.concatenate([exact[np.isfinite(exact)], duff[np.isfinite(duff)]])
        hi = min(3.0, float(finite.max())) if finite.size else 1.5
        lo = min(1.0, float(finite.min())) if finite.size else 0.5
        self.period_plot.setXRange(0, 180, padding=0)
        self.period_plot.setYRange(lo - 0.05, max(hi, 1.0) + 0.1, padding=0)
        self.period_plot.setLabel("bottom", "Amplitude (°)")
        if p.cubic == 0:
            self.period_note.setText(
                "No gravity in the restoring moment, so it is linear: the period is the same at any amplitude."
            )
        else:
            sense = "softens: larger swings are slower" if p.cubic < 0 else "hardens: larger swings are faster"
            self.period_note.setText(f"It {sense}. At {self.angle.value():.3g}°: T/T₀ = {now:.4g}.")

    def _draw_phase(self) -> None:
        p, a = self.pendulum, self.theta0
        for item in self.orbit_items:
            self.phase_plot.removeItem(item)
        self.orbit_items = []
        hanging = p.weight_moment > 0 and p.torsion == 0
        reach = 1.05 * math.pi if hanging else max(1.3 * a, 1.3 * abs(p.leaning()), 0.3)
        x = np.linspace(-reach, reach, 801)
        V = p.potential(x)
        top = float(p.potential(math.pi)) if hanging else float(np.max(V))
        base = float(np.min(V))
        pen = pg.mkPen(colors.grey, width=1)
        for e in np.linspace(base, top, 8)[1:-1]:
            ox, oy = orbit(p, float(e), x)
            item = pg.PlotDataItem(np.degrees(ox), np.degrees(oy), pen=pen, connect="finite")
            self.phase_plot.addItem(item, ignoreBounds=True)
            self.orbit_items.append(item)
        if hanging:  # the separatrix: the orbit through the upside-down balance
            inside = np.linspace(-math.pi, math.pi, 801)
            y = phase_curve(p, top, inside)
            y[[0, -1]] = 0.0
            self.separatrix.setData(
                np.degrees(np.concatenate([inside, inside[::-1]])), np.degrees(np.concatenate([y, -y[::-1]]))
            )
        else:
            self.separatrix.setData([], [])
        t, th, _ = swing(p, a, self._duration(), 3000)
        dth = np.gradient(th, t)
        keep = np.isfinite(th) & (np.abs(th) < 2 * reach)
        self.trace_item.setData(np.degrees(th[keep]), np.degrees(dth[keep]))
        self.start_dot.setData([math.degrees(a)], [0.0])
        self.phase_plot.setXRange(-math.degrees(reach), math.degrees(reach), padding=0.02)
        self.phase_plot.setLabel("bottom", "θ (°)")
        self.phase_plot.setLabel("left", "θ̇ (°/s)")
        if p.weight_moment > 0 and p.torsion == 0:
            self.phase_note.setText(
                "Grey: undamped orbits of constant energy; bold: the separatrix through the "
                "upside-down balance (±180°). Inside it the pendulum swings; outside it, it "
                "goes round."
            )
        else:
            self.phase_note.setText("Grey: undamped orbits of constant energy. Colour: the swing from release.")

    def _draw_pivot(self) -> None:
        p = self.pendulum
        L, g = p.length, p.gravity if p.gravity > 0 else 9.81
        d = np.linspace(0.02 * L, 0.6 * L, 400)
        T = rod_period(L, d, g)
        self.pivot_item.setData(d, T)
        best = L / math.sqrt(12)
        self.pivot_best.setData([best], [float(rod_period(L, best, g))])
        here = min(max(p.pivot if p.kind is Kind.ROD else 0.5 * L, 0.02 * L), 0.6 * L)
        Th = float(rod_period(L, here, g))
        twin = conjugate_pivot(L, here)
        self.pivot_now.setData([here], [Th])
        self.pivot_twin.setData(
            [twin] if 0.02 * L <= twin <= 0.6 * L else [], [Th] if 0.02 * L <= twin <= 0.6 * L else []
        )
        self.pivot_level.setPos(Th)
        self.pivot_plot.setXRange(0.0, 0.6 * L, padding=0)
        self.pivot_plot.setYRange(0.9 * float(rod_period(L, best, g)), min(float(T.max()), 2.0 * Th), padding=0)
        lead = "" if p.kind is Kind.ROD else f"(A uniform rod {fmt(L, 'm')} long, whatever the kind set.) "
        self.pivot_note.setText(
            f"{lead}Pivoted {fmt(here, 'm')} from the centre: T = {Th:.4g} s. The fastest, "
            f"{float(rod_period(L, best, g)):.4g} s, is at L/√12 = {fmt(best, 'm')} (grey); "
            f"the same period again at k<sub>G</sub>²/d = {fmt(twin, 'm')} (square)."
        )
