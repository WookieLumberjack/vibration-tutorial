"""Numerical integration page: what a time-stepping scheme does to a vibration, and why the app steps exactly.

Left: the oscillator (optionally with a stiff second mode), the time step, the start and the force, and which
schemes to run. Centre: the first mass's motion by each scheme against the closed form, and each one's error.
Right: each scheme's spectral radius, period error and numerical damping against Δt/T, with the step set marked
for each mode. The page has its own model (core/integration.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.integration import (
    SCHEMES,
    STABLE_LIMIT,
    TWO_PI,
    Load,
    Model,
    Scheme,
    exact,
    simulate,
    spectral,
    spectral_curves,
)
from ..core.practice import Problem, Question, Requirement, at_most, kept, number, pick, rounded
from . import integration_figures, settle_plots
from .lesson import (
    Form,
    TrialBox,
    column,
    connect_all,
    fmt,
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

TAB_RADIUS, TAB_PERIOD, TAB_DAMPING = range(3)
MOST_STEPS = 20000  # per scheme per run
RATIOS = np.logspace(-2, 1, 241)  # Δt/T for the spectral plots
POINTS = 2000  # of the closed form drawn


@dataclass(frozen=True)
class IntegrationTrial:
    name: str
    note: str
    per_period: float = 10.0  # Δt = T₁ / this
    periods: float = 10.0
    damping: float = 0.0
    stiff_hz: float = 0.0
    x0: float = 0.01
    force: float = 0.0
    freq_hz: float = 2.0
    schemes: tuple[Scheme, ...] = (Scheme.CENTRAL, Scheme.AVERAGE, Scheme.LINEAR, Scheme.RK4)
    alpha: float = -0.1
    tab: int = TAB_PERIOD
    mass: float = 1.0
    stiffness: float = 400.0


TRIALS = (
    IntegrationTrial(
        "Ten steps a period",
        "1 kg on 400 N/m (3.18 Hz), let go from 10 mm, ten steps a period. No scheme blows up, but each gets the "
        "period wrong: central difference 1.7 % short, Newmark's average acceleration 3.2 % long, linear "
        "acceleration 1.6 % long, RK4 0.11 % long. Over ten periods those errors pile up as phase: the "
        "average-acceleration solution is a third of a cycle behind. RK4's amplitude also shrinks, by "
        "0.065 % damping that nobody put in.",
    ),
    IntegrationTrial(
        "Central difference's limit: Δt < T/π",
        "Central difference is explicit, and stable only for <m>ωΔt ≤ 2</m>, <m>Δt ≤ T/π</m>. At 3.0 steps a period "
        "(<m>ωΔt</m> = 2.09) its error doubles every few steps and the run stops; at 3.2 (<m>ωΔt</m> = 1.96) it is stable "
        "but its period is 29 % short. Newmark's average acceleration runs at any step, its period 30 % long.",
        per_period=3.0,
        schemes=(Scheme.CENTRAL, Scheme.AVERAGE),
        tab=TAB_RADIUS,
    ),
    IntegrationTrial(
        "One step a period",
        "<m>Δt = T</m>. Average acceleration is unconditionally stable: the amplitude holds at 10 mm, but the motion it "
        "computes has a period 2.5 times too long, nothing like the oscillator's. Linear acceleration needs "
        "<m>ωΔt ≤ 2√3</m> (<m>Δt ≤ 0.55T</m>) and blows up. Stable is not the same as accurate.",
        per_period=1.0,
        periods=10.0,
        schemes=(Scheme.AVERAGE, Scheme.LINEAR),
        tab=TAB_RADIUS,
    ),
    IntegrationTrial(
        "RK4's numerical damping",
        "Five steps a period. RK4 is the most accurate here in period (1.1 % long), but it shrinks the "
        "amplitude by 2.2 % a step: numerical damping of <m>ζ̄</m> = 1.8 %, so after ten periods only 33 % is left. "
        "Average acceleration keeps the amplitude exactly, its period 12 % long.",
        per_period=5.0,
        schemes=(Scheme.AVERAGE, Scheme.RK4),
        tab=TAB_DAMPING,
    ),
    IntegrationTrial(
        "Explicit Euler grows",
        "Fifty steps a period, which would be plenty for any other scheme. Explicit Euler steps along the "
        "tangent, which always points outwards from the circle the motion should go round: the amplitude grows "
        "by 0.8 % a step, 48 % a period, 50 times over ten periods. Its spectral radius is above 1 at every "
        "<m>Δt</m> for an undamped oscillator.",
        per_period=50.0,
        schemes=(Scheme.EULER, Scheme.CENTRAL),
        tab=TAB_RADIUS,
    ),
    IntegrationTrial(
        "A stiff mode breaks the explicit schemes",
        "A 0.1 kg mass on a stiff spring hangs from the first: a 1049 Hz mode beside the 3.03 Hz one, as a fine "
        "mesh has. The run takes 50 steps per period of the slow mode, which gives the stiff one <m>ωΔt</m> = 43, a "
        "seventh of a step per period. Central difference and RK4 blow up within 20 ms, although nothing that "
        "matters happens up there. Average acceleration doesn't: what limits an explicit step is the fastest mode "
        "in the model, not the one you want.",
        per_period=50.0,
        periods=5.0,
        stiff_hz=1000.0,
        schemes=(Scheme.CENTRAL, Scheme.RK4, Scheme.AVERAGE),
        tab=TAB_RADIUS,
    ),
    IntegrationTrial(
        "HHT-α damps what it can't resolve",
        "The same stiff mode. It is in the closed form too (let go from rest, <m>m<sub>1</sub></m> sets it ringing), but no step "
        "this long can follow it, so average acceleration turns it into jagged noise that never dies (<m>ρ</m> = 1 at "
        "any step). HHT-α with <m>α</m> = −0.3 has <m>ρ</m> = 0.57 a step at <m>ωΔt</m> = 43: the noise is gone within a few steps, "
        "and the 3.03 Hz mode loses only <m>ζ̄</m> = 0.004 %. That is what finite-element codes use it for.",
        per_period=50.0,
        periods=5.0,
        stiff_hz=1000.0,
        schemes=(Scheme.AVERAGE, Scheme.HHT),
        alpha=-0.3,
        tab=TAB_DAMPING,
    ),
    IntegrationTrial(
        "The exact update and a sine force",
        "A 1 N sine at 2 Hz from rest, ten steps a period. The first-order-hold update every other page uses is "
        "exact for the oscillator at any Δt; its only error comes from treating the force as straight between "
        "samples: 1.3 % of the swing here. RK4's error is 3 %; central difference's and average "
        "acceleration's, 41 and 61 %: the start sets the undamped mass ringing at its natural frequency as well, and their "
        "period errors put that ringing out of phase.",
        x0=0.0,
        force=1.0,
        schemes=(Scheme.FOH, Scheme.CENTRAL, Scheme.AVERAGE, Scheme.RK4),
        tab=TAB_PERIOD,
    ),
)

# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


def _central_limit(rng: np.random.Generator) -> Question:
    m, k = pick(rng, (0.5, 1, 2, 5)), pick(rng, (100, 400, 1000, 2500))
    w = math.sqrt(k / m)
    dt = 2 / w
    return Question(
        f"A {m:g} kg mass on a {k:g} N/m spring, undamped, stepped by central difference. What is the longest time "
        "step that stays stable?",
        1000 * dt,
        f"Central difference is stable while ωΔt ≤ 2. ω = √(k/m) = {number(w)} rad/s, so Δt ≤ 2/ω = "
        f"{number(1000 * dt)} ms: T/π, with T = {number(1000 * TWO_PI / w)} ms.",
        unit="ms",
        setup=dict(mass=m, stiffness=k, per_period=3.2, schemes=(Scheme.CENTRAL,), tab=TAB_RADIUS),
    )


def _average_period(rng: np.random.Generator) -> Question:
    n = int(pick(rng, (4, 5, 8, 10, 20)))
    O = TWO_PI / n
    error = 100 * spectral(Scheme.AVERAGE, O).period_error
    return Question(
        f"Newmark's average acceleration steps an undamped oscillator with {n} steps a period. By how many percent "
        "does the period it computes come out long?",
        error,
        f"Its amplification matrix turns the motion by θ = 2 arctan(Ω/2) a step, against Ω = ωΔt = 2π/{n} = "
        f"{number(O)}: θ = {number(2 * math.atan(O / 2))}. The period is Ω/θ = {number(1 + error / 100, 4)} times "
        f"the true one, {number(error)} % long. (About Ω²/12 = {number(100 * O * O / 12)} % for small steps.)",
        unit="%",
        tolerance=0.02,
        setup=dict(per_period=float(n), schemes=(Scheme.AVERAGE,), tab=TAB_PERIOD),
    )


def _euler_growth(rng: np.random.Generator) -> Question:
    n = int(pick(rng, (20, 50, 100, 200)))
    O = TWO_PI / n
    growth = 100 * ((1 + O * O) ** (n / 2) - 1)
    return Question(
        f"Explicit Euler steps an undamped oscillator with {n} steps a period. By how many percent does the "
        "amplitude grow over one period?",
        growth,
        f"Its amplification matrix I + AΔt has |λ| = ρ = √(1 + Ω²), Ω = 2π/{n} = {number(O)}: ρ = "
        f"{number(math.sqrt(1 + O * O), 5)} a step, ρ<sup>{n}</sup> = {number(1 + growth / 100, 4)} a period, "
        f"{number(growth)} % up.",
        unit="%",
        tolerance=0.02,
        setup=dict(per_period=float(n), schemes=(Scheme.EULER,), tab=TAB_RADIUS),
    )


def _rk4_damping(rng: np.random.Generator) -> Question:
    n = int(pick(rng, (4, 5, 6, 8, 10)))
    O = TWO_PI / n
    s = spectral(Scheme.RK4, O)
    return Question(
        f"Classical Runge–Kutta (RK4) steps an undamped oscillator with {n} steps a period. What damping ratio does "
        "the motion it computes have?",
        100 * s.damping,
        f"Its amplification matrix is the Taylor series of e<sup>AΔt</sup> to Ω⁴/24, Ω = 2π/{n} = {number(O)}; its "
        f"eigenvalues have ρ = {number(s.radius, 5)} a step, and ζ̄ = −ln ρ / |ln λ| = {number(100 * s.damping)} %: "
        "damping nobody put in.",
        unit="%",
        tolerance=0.03,
        setup=dict(per_period=float(n), schemes=(Scheme.RK4,), tab=TAB_DAMPING),
    )


def _stiff_steps(rng: np.random.Generator) -> Question:
    stiff = pick(rng, (500, 1000, 2000))
    f = Model(stiff_hz=stiff).frequencies_hz()
    slow, fast = rounded(f[0]), rounded(f[-1])
    steps = math.pi * fast / slow
    return Question(
        f"A model's modes are at {number(slow)} Hz and {number(fast)} Hz. Stepped by central difference, how many "
        f"steps per period of the {number(slow)} Hz mode does it need at least to stay stable?",
        steps,
        f"The step has to keep ωΔt ≤ 2 for the fastest mode: Δt ≤ 1/(π × {number(fast)}) = "
        f"{number(1000 / (math.pi * fast))} ms. The slow mode's period is {number(1000 / slow)} ms, so that's "
        f"π × {number(fast)}/{number(slow)} = {number(steps)} steps a period, though nothing that matters happens up "
        "there.",
        unit="steps",
        tolerance=0.02,
        setup=dict(stiff_hz=stiff, per_period=50.0, periods=2.0, schemes=(Scheme.CENTRAL,), tab=TAB_RADIUS),
    )


def _halve_step(rng: np.random.Generator) -> Question:
    return Question(
        "Central difference with ten steps a period gets the period 1.7 % short. What happens to that error at twenty "
        "steps a period?",
        0,
        "It falls about four times, to 0.41 %: central difference is second order, its period error ≈ −(ωΔt)²/24.",
        choices=("It falls about four times", "It halves", "It stays about the same", "It falls about sixteen times"),
        setup=dict(per_period=20.0, schemes=(Scheme.CENTRAL,), tab=TAB_PERIOD),
        predict=True,
    )


def _one_step(rng: np.random.Generator) -> Question:
    return Question(
        "Newmark's average acceleration steps an undamped oscillator one step a period (Δt = T). What does the motion "
        "it computes do?",
        0,
        "Its amplitude holds (ρ = 1 at any step: unconditionally stable), but its period is 2.5 times too long, "
        "nothing like the oscillator's. Stable is not accurate.",
        choices=(
            "It keeps its amplitude, with the period far too long",
            "It blows up",
            "It dies away",
            "It follows the true motion closely",
        ),
        setup=dict(per_period=1.0, schemes=(Scheme.AVERAGE,), tab=TAB_RADIUS),
        predict=True,
    )


def _stiff_explicit(rng: np.random.Generator) -> Question:
    return Question(
        "A light mass on a stiff spring adds a 1049 Hz mode beside the 3.03 Hz one. Stepped by central difference "
        "with 50 steps per period of the slow mode, what happens?",
        0,
        "It blows up within a few steps: the stiff mode has ωΔt = 43, far past central difference's 2. What "
        "limits an explicit step is the fastest mode in the model.",
        choices=(
            "It blows up",
            "It runs, the slow mode accurate",
            "It runs, the stiff mode damped out",
            "It runs, the stiff mode's period wrong",
        ),
        setup=dict(per_period=50.0, periods=5.0, stiff_hz=1000.0, schemes=(Scheme.CENTRAL,), tab=TAB_RADIUS),
        predict=True,
    )


def _exact_big_step(rng: np.random.Generator) -> Question:
    return Question(
        "The exact (FOH) update steps a free, damped oscillator with two steps a period. How far off are the values "
        "it computes at the steps?",
        0,
        "Not at all, round-off aside: e<sup>AΔt</sup> is the oscillator's own free motion over a step, at any Δt. "
        "Only a force varying between samples brings in an error.",
        choices=("They're exact", "A few percent", "It's unstable at that step", "Its period comes out long"),
        setup=dict(per_period=2.0, damping=0.8, schemes=(Scheme.FOH, Scheme.AVERAGE), tab=TAB_PERIOD),
        predict=True,
    )


def _step_spec(rng: np.random.Generator) -> Question:
    stiff = pick(rng, (500, 1000, 2000))
    worst = pick(rng, (0.5, 1, 2))  # % period error
    O = TWO_PI / np.arange(1, 1000, 0.1)
    fewest = TWO_PI / O[np.argmax([100 * spectral(Scheme.AVERAGE, o).period_error <= worst for o in O])]
    most = math.ceil(1.25 * fewest)

    def goal(t: IntegrationTrial):
        f = Model(t.mass, t.stiffness, t.damping, t.stiff_hz).frequencies_hz()
        h = 1 / (f[0] * t.per_period)
        run = [s for s in t.schemes]
        rows = [(s, [spectral(s, TWO_PI * v * h, 0.0, t.alpha) for v in f]) for s in run]
        radius = max((r.radius for _, sp in rows for r in sp), default=math.nan)
        errors = [abs(sp[0].period_error) for _, sp in rows]
        error = 100 * max((e if math.isfinite(e) else math.inf for e in errors), default=math.nan)
        return (
            kept("Mass", t.mass, 1.0, "kg"),
            kept("Stiffness", t.stiffness, 400.0, "N/m"),
            kept("Stiff mode", t.stiff_hz, stiff, "Hz"),
            Requirement(
                "A stepping scheme, not the exact update",
                ", ".join(s.value for s in run) or "none",
                bool(run) and Scheme.FOH not in run,
            ),
            Requirement(
                "Every scheme run stable at both modes (ρ ≤ 1)", f"ρ = {number(radius, 4)}", radius <= 1 + 1e-9
            ),
            at_most("The slow mode's period error", error, worst, "%"),
            at_most("Steps a period", t.per_period, most),
        )

    return Question(
        f"The 1 kg, 400 N/m oscillator has a stiff {stiff:g} Hz mode hung from it. Choose the schemes and the step so "
        f"that every scheme run is stable and gets the slow mode's period within {worst:g} %, with no more than "
        f"{most} steps a period (not the exact update: it's a stand-in for a nonlinear model).",
        0,
        f"An explicit scheme needs ωΔt ≤ 2 or so at the stiff mode, hundreds of steps a period, so it has to be "
        f"implicit. Newmark's average acceleration's period error ≈ (ωΔt)²/12 is under {worst:g} % from "
        f"{number(fewest)} steps a period; {most} steps with it alone meet the spec (HHT-α, a little longer, may too).",
        setup=dict(stiff_hz=stiff, per_period=50.0, periods=5.0, schemes=(Scheme.CENTRAL, Scheme.RK4)),
        goal=goal,
        design=dict(stiff_hz=stiff, per_period=float(most), periods=5.0, schemes=(Scheme.AVERAGE,)),
    )


PROBLEMS = tuple(
    Problem(key, title, "integration_page", f"See {page_link('integration_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("central-limit", "Central difference's step", "central", "Central difference", _central_limit),
        ("average-period", "Average acceleration's period", "newmark", "Newmark and HHT-α", _average_period),
        ("euler-growth", "Explicit Euler's growth", "rk4", "Runge–Kutta", _euler_growth),
        ("rk4-damping", "RK4's numerical damping", "rk4", "Runge–Kutta", _rk4_damping),
        ("stiff-steps", "A stiff mode's toll", "central", "Central difference", _stiff_steps),
        ("halve-step", "Predict: half the step", "central", "Central difference", _halve_step),
        ("one-step", "Predict: one step a period", "newmark", "Newmark and HHT-α", _one_step),
        ("stiff-explicit", "Predict: a stiff mode", "central", "Central difference", _stiff_explicit),
        ("exact-big-step", "Predict: the exact update", "exact", "The exact update", _exact_big_step),
        ("step-spec", "Spec: a scheme and a step", "newmark", "Newmark and HHT-α", _step_spec),
    )
)

TIPS = {
    "mass": "<p>The mass m.</p>",
    "stiffness": "<p>The spring k.</p>",
    "damping": "<p>The damper c.</p>",
    "stiff": "<p>A second mass, a tenth of the first, hung from it on a spring stiff enough that alone it would "
    "vibrate at this frequency: a mode far above the first, as a fine mesh has. 0: none.</p>",
    "per_period": "<p>The time step, as steps per period of the first mode: Δt = T₁ / this.</p>",
    "periods": "<p>How long to run, in periods of the first mode.</p>",
    "x0": "<p>The first mass starts here, at rest.</p>",
    "force": "<p>A sine force F sin 2πft on the first mass.</p>",
    "freq": "<p>Its frequency.</p>",
    "alpha": "<p>HHT-α's α, from −1/3 to 0 (0 is the average acceleration scheme): the more negative, the more "
    "it damps motion it can't resolve.</p>",
    "schemes": "<p>Which schemes to run.</p>",
}
TIME_TIP = (
    "<p>The first mass's displacement by each scheme (dots at the steps), against the closed form "
    "(dashed). A scheme that blows up stops.</p>"
)
ERROR_TIP = "<p>Each scheme's error against the closed form.</p>"
RADIUS_TIP = (
    "<p>The spectral radius ρ of each scheme's one-step map of a free, undamped vibration against "
    "Δt/T: above 1 the motion grows every step (unstable); below 1 the scheme damps it. The dotted lines "
    "are the step set, for each mode.</p>"
)
PERIOD_TIP = "<p>The period of the motion each scheme computes, against the true one, against Δt/T.</p>"
DAMPING_TIP = (
    "<p>The damping ratio of the motion each scheme computes for an undamped oscillator (numerical "
    "damping), against Δt/T. Negative: it grows.</p>"
)

NOTES_HTML = load_notes("integration", figures=integration_figures.FIGURES, trials=trials_html(TRIALS))


class IntegrationPage(QtWidgets.QWidget):
    """Controls on the left, the motion and its error in the centre, the schemes' curves on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = Model()
        self.load = Load()
        self.T = self.h = 1.0  # set from the inputs below

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Each sets up the oscillator, the step and the schemes to show one idea from the Theory tab.</p>",
        )
        self.trials.picked.connect(self.load_trial)

        osc = Form("Oscillator")
        self.mass = osc.number("Mass m:", 0.001, 1e4, 1.0, 3, " kg", TIPS["mass"])
        self.stiffness = osc.number("Stiffness k:", 0.01, 1e8, 400.0, 2, " N/m", TIPS["stiffness"])
        self.damping = osc.number("Damping c:", 0.0, 1e6, 0.0, 3, " N·s/m", TIPS["damping"])
        self.stiff = osc.number("Stiff mode:", 0.0, 1e5, 0.0, 1, " Hz", TIPS["stiff"])
        self.osc_info = note_label()
        osc.row(self.osc_info)

        run = Form("Run")
        self.per_period = run.number("Steps a period:", 0.2, 1000.0, 10.0, 2, "", TIPS["per_period"])
        self.periods = run.number("Periods:", 0.5, 100.0, 10.0, 1, "", TIPS["periods"])
        self.x0 = run.number("Start x₀:", -1.0, 1.0, 0.01, 4, " m", TIPS["x0"])
        self.force = run.number("Force F:", 0.0, 1e4, 0.0, 3, " N", TIPS["force"])
        self.freq = run.number("at f:", 0.01, 1e4, 2.0, 3, " Hz", TIPS["freq"])
        self.step_info = note_label()
        run.row(self.step_info)

        pick = Form("Schemes")
        self.checks: dict[Scheme, QtWidgets.QCheckBox] = {}
        for s in SCHEMES:
            box = QtWidgets.QCheckBox(s.value)
            box.setToolTip(TIPS["schemes"])
            self.checks[s] = box
            pick.row(box)
        self.alpha = pick.number("HHT α:", -1 / 3, 0.0, -0.1, 3, "", TIPS["alpha"])
        self.alpha.setSingleStep(0.05)

        connect_all(
            (
                self.mass,
                self.stiffness,
                self.damping,
                self.stiff,
                self.per_period,
                self.periods,
                self.x0,
                self.force,
                self.freq,
                self.alpha,
                *self.checks.values(),
            ),
            self._on_inputs,
        )

        # --- centre
        self.time_plot = plot("Displacement x₁", "m", "Time", "s", height=150)
        self.time_plot.setToolTip(TIME_TIP)
        add_legend(self.time_plot.getPlotItem(), offset=(-10, 10))
        self.exact_item = pg.PlotDataItem(name="closed form")
        self.time_plot.addItem(self.exact_item)
        self.err_plot = plot("Error", "m", "Time", "s", log_y=True, height=120)
        self.err_plot.setToolTip(ERROR_TIP)
        self.err_plot.setXLink(self.time_plot)
        self.run_items: dict[Scheme, pg.PlotDataItem] = {}
        self.err_items: dict[Scheme, pg.PlotDataItem] = {}
        for s in SCHEMES:
            self.run_items[s] = pg.PlotDataItem(name=s.value)
            self.err_items[s] = pg.PlotDataItem()
            self.time_plot.addItem(self.run_items[s])
            self.err_plot.addItem(self.err_items[s])
        self.readout = readout_label()

        # --- right
        def curve_plot(left: str, units: str | None, tip: str) -> tuple[pg.PlotWidget, dict, list]:
            p = plot(left, units, "Δt / T", None, log_x=True, height=200)
            p.setToolTip(tip)
            p.setMouseEnabled(x=False, y=False)
            add_legend(p.getPlotItem(), offset=(10, 10))
            items = {s: pg.PlotDataItem(name=s.value) for s in SCHEMES}
            marks = [pg.InfiniteLine(angle=90), pg.InfiniteLine(angle=90)]
            for item in (*marks, *items.values()):
                p.addItem(item)
            return p, items, marks

        self.radius_plot, self.radius_items, self.radius_marks = curve_plot("Spectral radius ρ", None, RADIUS_TIP)
        self.unit_line = pg.InfiniteLine(angle=0, pos=1.0)
        self.radius_plot.addItem(self.unit_line)
        self.period_plot, self.period_items, self.period_marks = curve_plot("Period error", "%", PERIOD_TIP)
        self.damp_plot, self.damp_items, self.damp_marks = curve_plot("Numerical damping ζ̄", "%", DAMPING_TIP)
        self.curve_note = note_label()

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(column(self.radius_plot, stretch=False), "Spectral radius")
        self.tabs.addTab(column(self.period_plot, stretch=False), "Period error")
        self.tabs.addTab(column(self.damp_plot, stretch=False), "Numerical damping")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._draw_curves)
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.tabs, 1)
        rv.addWidget(self.curve_note)

        three_panes(
            self,
            column(self.trials, osc, run, pick),
            stack((self.time_plot, 3), (self.err_plot, 2), (self.readout, 0)),
            right,
        )
        self.load_trial(0, note=False)
        self.apply_theme()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int, note: bool = True) -> None:
        self.apply_trial(TRIALS[index], note)

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(IntegrationTrial("", "", **setup), note=False)

    def practice_state(self) -> IntegrationTrial:
        """The page's settings, as a trial: what a practice spec checks."""
        return IntegrationTrial(
            "",
            "",
            per_period=self.per_period.value(),
            periods=self.periods.value(),
            damping=self.damping.value(),
            stiff_hz=self.stiff.value(),
            x0=self.x0.value(),
            force=self.force.value(),
            freq_hz=self.freq.value(),
            schemes=tuple(self._chosen()),
            alpha=self.alpha.value(),
            tab=self.tabs.currentIndex(),
            mass=self.mass.value(),
            stiffness=self.stiffness.value(),
        )

    def apply_trial(self, t: IntegrationTrial, note: bool = True) -> None:
        box: QtWidgets.QWidget
        for box, value in (
            (self.mass, t.mass),
            (self.stiffness, t.stiffness),
            (self.damping, t.damping),
            (self.stiff, t.stiff_hz),
            (self.per_period, t.per_period),
            (self.periods, t.periods),
            (self.x0, t.x0),
            (self.force, t.force),
            (self.freq, t.freq_hz),
            (self.alpha, t.alpha),
        ):
            set_quietly(box, value)
        for s, box in self.checks.items():
            set_quietly(box, s in t.schemes)
        if note:
            self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self._on_inputs()

    def _on_inputs(self) -> None:
        self.model = Model(self.mass.value(), self.stiffness.value(), self.damping.value(), self.stiff.value())
        self.load = Load(self.x0.value(), self.force.value(), self.freq.value())
        f = self.model.frequencies_hz()
        self.T = 1.0 / f[0] if f[0] > 0 else 1.0
        self.h = self.T / self.per_period.value()
        self.osc_info.setText(
            "Modes: " + ", ".join(f"{v:.4g} Hz" for v in f) + f"; T₁ = {fmt(self.T, 's')}, ζ = "
            f"{self.model.damping / (2 * math.sqrt(self.model.stiffness * self.model.mass)):.3g}."
        )
        self.step_info.setText(
            f"Δt = {fmt(self.h, 's')}: ωΔt = "
            + ", ".join(f"{TWO_PI * v * self.h:.3g}" for v in f)
            + " (mode by mode); Δt/T = "
            + ", ".join(f"{v * self.h:.3g}" for v in f)
            + "."
        )
        self.redraw()

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.exact_item.setPen(pg.mkPen(c.strong, width=2, style=QtCore.Qt.PenStyle.DashLine))
        for i, s in enumerate(SCHEMES):
            color = c.mass[i % len(c.mass)]
            for item in (self.run_items[s], self.radius_items[s], self.period_items[s], self.damp_items[s]):
                item.setPen(pg.mkPen(color, width=2))
            self.run_items[s].setSymbolBrush(pg.mkBrush(color))
            self.run_items[s].setSymbolPen(None)
            self.err_items[s].setPen(pg.mkPen(color, width=1.5))
        for line in (*self.radius_marks, *self.period_marks, *self.damp_marks):
            line.setPen(pg.mkPen(c.grey, width=1.5, style=QtCore.Qt.PenStyle.DotLine))
        self.unit_line.setPen(pg.mkPen(c.faint, width=1))
        self.redraw()

    def redraw(self) -> None:
        self._draw_runs()
        self._draw_curves()
        settle_plots(self)

    def _chosen(self) -> list[Scheme]:
        return [s for s, box in self.checks.items() if box.isChecked()]

    @staticmethod
    def _legend(p: pg.PlotWidget, entries: list[tuple[pg.PlotDataItem, str]]) -> None:
        legend = p.getPlotItem().legend
        legend.clear()
        for item, name in entries:
            legend.addItem(item, name)

    def _draw_runs(self) -> None:
        h, model, load = self.h, self.model, self.load
        span = self.periods.value() * self.T
        steps = min(MOST_STEPS, max(1, round(span / h)))
        span = steps * h
        tf = np.linspace(0.0, span, POINTS + 1)
        self.exact_item.setData(tf, exact(model, load, tf))
        t = np.arange(steps + 1) * h
        reference = exact(model, load, t)
        swing = float(np.max(np.abs(reference))) or 1.0
        lines = []
        dots = steps <= 200
        for s in SCHEMES:
            item, err = self.run_items[s], self.err_items[s]
            if s not in self._chosen():
                item.setData([], [])
                err.setData([], [])
                continue
            x = simulate(s, model, load, h, steps, self.alpha.value())
            ok = np.isfinite(x)
            item.setData(t[ok], x[ok], symbol="o" if dots else None, symbolSize=5)
            e = np.abs(x - reference)
            good = ok & (e > 0)
            err.setData(t[good], e[good])
            if ok.all():
                lines.append(f"{s.value}: largest error {100 * float(np.max(e)) / swing:.3g} % of the swing")
            else:
                gone = t[np.argmin(ok)]
                lines.append(f"{s.value}: <b>blew up</b> by t = {fmt(float(gone), 's')}")
        chosen = self._chosen()
        self._legend(
            self.time_plot, [(self.exact_item, "closed form")] + [(self.run_items[s], s.value) for s in chosen]
        )
        errors = [
            item.yData
            for s, item in self.err_items.items()
            if s in chosen and item.yData is not None and item.yData.size
        ]
        if errors:
            top = max(float(np.max(e)) for e in errors)
            floor = max(min(float(np.min(e)) for e in errors), 1e-7 * swing)
            self.err_plot.setYRange(math.log10(floor), math.log10(max(top, floor * 10)) + 0.3, padding=0)
        self.time_plot.setXRange(0.0, span, padding=0.01)
        lo, hi = float(np.min(reference)), float(np.max(reference))
        pad = 0.25 * (hi - lo or abs(hi) or 1.0)
        self.time_plot.setYRange(lo - pad, hi + pad, padding=0)
        self.readout.setText(
            f"{steps} steps of {fmt(h, 's')}. " + "; ".join(lines) + "." if lines else "Pick a scheme on the left."
        )

    def _draw_curves(self) -> None:
        i = self.tabs.currentIndex()
        if i > TAB_DAMPING:
            return
        zeta = 0.0
        alpha = self.alpha.value()
        O = TWO_PI * RATIOS
        plot_, items, marks = (
            (self.radius_plot, self.radius_items, self.radius_marks),
            (self.period_plot, self.period_items, self.period_marks),
            (self.damp_plot, self.damp_items, self.damp_marks),
        )[i]
        for s in SCHEMES:
            if s not in self._chosen():
                items[s].setData([], [])
                continue
            radius, period, damping = spectral_curves(s, O, zeta, alpha)
            y = (radius, 100 * period, 100 * damping)[i]
            ok = np.isfinite(y)
            items[s].setData(RATIOS[ok], y[ok], connect="finite")
        self._legend(plot_, [(items[s], s.value) for s in self._chosen()])
        f = self.model.frequencies_hz()
        for line, v in zip(marks, (*f, None, None)):
            line.setVisible(v is not None)
            if v is not None:
                line.setValue(math.log10(max(v * self.h, 1e-6)))
        plot_.setXRange(-2, 1, padding=0)
        if i == TAB_RADIUS:
            plot_.setYRange(0.0, 2.0, padding=0)
        elif i == TAB_PERIOD:
            plot_.setYRange(-50.0, 100.0, padding=0)
        else:
            plot_.setYRange(-10.0, 20.0, padding=0)
        limits = [f"{s.value} ωΔt ≤ {STABLE_LIMIT[s]:.3g}" for s in self._chosen() if 0 < STABLE_LIMIT[s] < math.inf]
        self.curve_note.setText(
            "Undamped oscillator; dotted: this Δt for each mode. "
            + ("Stable while " + ", ".join(limits) + "." if limits else "")
        )
