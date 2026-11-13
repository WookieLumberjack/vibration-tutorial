"""Active control page: velocity feedback on the Simulation page's chain, collocated or not, against a TMD.

Left: the sensor and actuator masses, the gain and the actuator's lag, the disturbance and response masses, and a
tuned mass damper to compare. Centre: the receptance without control, with it, and with the TMD, and each mode's
damping. Right: the root locus as the gain rises, each mode's damping against the gain, skyhook isolation, and the
Theory (its figures: gui/active_control_figures.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.active_control import (
    Controller,
    closed_loop,
    frf,
    max_real,
    modal_damping_estimate,
    oscillating_poles,
    skyhook,
    tmd_for,
    tracked_damping,
    undamped_modes,
)
from ..core.model import ChainSystem
from ..core.practice import Problem, Question, Requirement, at_least, kept, number, pick
from . import active_control_figures, settle_plots
from .lesson import (
    Form,
    TrialBox,
    column,
    connect_all,
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
from .style import MAX_DOF, colors
from .theming import add_legend, mute

TAB_LOCUS, TAB_GAIN, TAB_SKYHOOK = range(3)
GAINS = np.concatenate([[0.0], np.geomspace(0.05, 2000.0, 160)])  # N·s/m, for the locus and the damping plot


@dataclass(frozen=True)
class ControlTrial:
    name: str
    note: str
    sensor: int = 4  # 1-based, as shown
    actuator: int = 4
    gain: float = 20.0
    lag_ms: float = 0.0
    tmd: bool = False
    tab: int = TAB_GAIN
    zeta_passive: float = 0.1
    zeta_sky: float = 0.0
    inp: int = 4  # 1-based, as shown
    out: int = 4
    ratio: float = 0.05
    masses: tuple[float, ...] = ()  # the chain, element by element; none: the uniform four-mass one
    springs: tuple[float, ...] = ()
    dampers: tuple[float, ...] = ()

    def system(self) -> ChainSystem:
        if not self.masses:
            return ChainSystem.uniform(4)
        return ChainSystem(np.array(self.masses), np.array(self.springs), np.array(self.dampers))


TRIALS = (
    ControlTrial(
        "Velocity feedback at the tip",
        "The uniform chain (four 1 kg masses, 400 N/m), a velocity sensor and a force actuator both on m4, "
        "<m>u</m> = −<m>gẋ<sub>4</sub></m> with <m>g</m> = 20 N·s/m: an active damper to the ground. Every mode gains damping, mode 1 most "
        "(<m>ζ</m> from 1.7 % to 72 %), and the tip's resonance peak all but disappears. For light gain each mode gains "
        "Δ<m>ζ</m> = <m>gφ<sub>r</sub></m>(4)²/2<m>ω<sub>r</sub></m>, so the modes that move the tip most are damped most.",
        tab=TAB_GAIN,
    ),
    ControlTrial(
        "Too much gain pins the mass",
        "Collocated feedback is stable at any gain, but more isn't better. Mode 1 turns overdamped at about "
        "26 N·s/m (its curve ends), and the other modes' damping peaks near there and falls again: the actuator "
        "holds m4 nearly still, and the chain vibrates around it as the chain with m4 fixed, three lightly "
        "damped modes (2.4, 4.5 and 5.9 Hz at 200 N·s/m). Each curve has an optimum, as a damper between two "
        "points does.",
        gain=200.0,
        tab=TAB_GAIN,
    ),
    ControlTrial(
        "Non-collocated: spillover",
        "Sensor on <m>m<sub>1</sub></m>, actuator on <m>m<sub>4</sub></m>, <m>g</m> = 20 N·s/m. In mode 1 the two move together, so mode 1 is damped; in "
        "mode 2 they move against each other, so the feedback pushes with the motion: the estimate takes 0.17 "
        "off mode 2's <m>ζ</m> of 0.05, and exactly its <m>ζ</m> is −0.10. The chain is unstable from g = 6.1 N·s/m. Control "
        "designed for one mode spills over into the others.",
        sensor=1,
        actuator=4,
        gain=20.0,
        tab=TAB_LOCUS,
    ),
    ControlTrial(
        "Actuator lag",
        "Back to collocated, but the sensor filter and the actuator each lag by <m>τ</m> = 20 ms. At high frequency the "
        "two lags turn the force by 180°, and the feedback that damped now feeds the motion: the loop is unstable "
        "above 96 N·s/m (with 10 ms, 200). Every real loop has lag, so collocation alone doesn't guarantee "
        "stability.",
        lag_ms=20.0,
        gain=100.0,
        tab=TAB_LOCUS,
    ),
    ControlTrial(
        "Active against a tuned mass damper",
        "A Den Hartog tuned mass damper at m4 with 5 % of mode 1's modal mass there (116 g) cuts the tip's "
        "receptance peak from 0.26 to 0.051 m/N. Collocated velocity feedback does the same with 2.3 N·s/m, and "
        "10 times that does much better and damps every mode, not just the one it's tuned to; but it needs "
        "power, a sensor and a stable loop.",
        gain=2.3,
        tmd=True,
        tab=TAB_GAIN,
    ),
    ControlTrial(
        "Skyhook isolation",
        "A machine on a mount over a vibrating floor (Skyhook tab). A passive damper (<m>ζ</m> = 0.1) trades the "
        "resonance peak (5.1) against isolation above √2 <m>f<sub>n</sub></m> (at 5 <m>f<sub>n</sub></m>: 0.059). A "
        "skyhook damper, pushing against an inertial frame through an actuator that feeds back the mass's "
        "absolute velocity, has neither problem: <m>ζ<sub>sky</sub></m> = 0.5 caps the peak at 1.15 and keeps the "
        "−40 dB/decade roll-off (0.041 at 5 <m>f<sub>n</sub></m>).",
        tab=TAB_SKYHOOK,
        zeta_passive=0.1,
        zeta_sky=0.5,
    ),
)

# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.
TWO_PI = 2.0 * math.pi
UNIFORM = dict(masses=(1.0,) * 4, springs=(400.0,) * 4, dampers=(2.0,) * 4)
SPEC_GAINS = np.geomspace(0.1, 2000.0, 300)


def _least_damped(system: ChainSystem, ctrl: Controller) -> float:
    """The closed loop's least damped oscillating pole's ζ; −inf if the loop is unstable."""
    A = closed_loop(system, ctrl)[0]
    if max_real(A) >= 0:
        return -math.inf
    return min((p.zeta for p in oscillating_poles(A)), default=math.inf)


def _one_mass(rng: np.random.Generator) -> Question:
    m, k = pick(rng, (1, 2, 5, 10)), pick(rng, (100, 400, 1000, 4000))
    g = pick(rng, (2, 5, 10, 20))
    zeta = g / (2 * math.sqrt(k * m))
    return Question(
        f"A {m:g} kg mass on a {k:g} N/m spring, undamped, with a velocity sensor and a force actuator on it: "
        f"u = −g ẋ with g = {g:g} N·s/m. What damping ratio does the loop give it?",
        100 * zeta,
        f"Collocated velocity feedback is a damper of c = g to the ground: ζ = g/2√(km) = {g:g}/(2√({number(k * m)})) "
        f"= {number(100 * zeta)} %.",
        unit="%",
        setup=dict(masses=(m,), springs=(k,), dampers=(0.0,), sensor=1, actuator=1, gain=g, inp=1, out=1, tab=TAB_GAIN),
    )


def _modal_estimate(rng: np.random.Generator) -> Question:
    a = int(pick(rng, (1, 2, 4)))
    r = int(pick(rng, (1, 2, 3, 4)))
    g = pick(rng, (0.5, 1, 2, 5))
    w, phi = undamped_modes(ChainSystem.uniform(4))
    f, shape = round(float(w[r - 1]) / TWO_PI, 3), round(abs(float(phi[a - 1, r - 1])), 3)
    dz = g * shape * shape / (2 * TWO_PI * f)
    return Question(
        f"The uniform chain (four 1 kg masses, 400 N/m): mode {r} is at {f:g} Hz, and its mass-normalized shape is "
        f"{shape:g} at m{a} (in kg<sup>−½</sup>). Collocated velocity feedback at m{a}, g = {g:g} N·s/m: how much "
        f"damping does mode {r} gain, by the light-gain estimate?",
        100 * dz,
        f"Δζ<sub>r</sub> = g φ<sub>r</sub>(a)²/2ω<sub>r</sub> = {g:g} × {shape:g}²/(2 × 2π × {f:g}) = "
        f"{number(100 * dz)} %: the modes that move m{a} most gain most.",
        unit="%",
        setup=dict(**UNIFORM, sensor=a, actuator=a, gain=g, tab=TAB_GAIN),
    )


def _spillover(rng: np.random.Generator) -> Question:
    sensor = int(pick(rng, (1, 2)))
    g = pick(rng, (2, 5, 10, 20))
    w, phi = undamped_modes(ChainSystem.uniform(4))
    f = round(float(w[1]) / TWO_PI, 3)
    ps, pa = round(float(phi[sensor - 1, 1]), 3), round(float(phi[3, 1]), 3)
    loss = -g * ps * pa / (2 * TWO_PI * f)
    return Question(
        f"The uniform chain's mode 2 is at {f:g} Hz, its mass-normalized shape {ps:g} at m{sensor} and {pa:g} at m4. "
        f"Velocity measured at m{sensor} is fed back as a force on m4, g = {g:g} N·s/m. By how much does the light-gain "
        "estimate lower mode 2's damping ratio?",
        100 * loss,
        f"Δζ = g φ(a)φ(s)/2ω = {g:g} × ({pa:g}) × {ps:g}/(2 × 2π × {f:g}) = −{number(100 * loss)} %: in mode 2 the two "
        "masses move against each other, so the force pushes with m4's motion and feeds the mode energy (spillover).",
        unit="%",
        setup=dict(**UNIFORM, sensor=sensor, actuator=4, gain=g, tab=TAB_LOCUS),
    )


def _den_hartog(rng: np.random.Generator) -> Question:
    mu = pick(rng, (0.01, 0.02, 0.05, 0.1))
    zeta = math.sqrt(3 * mu / (8 * (1 + mu) ** 3))
    return Question(
        f"A tuned mass damper with {mu:g} of its mode's modal mass. What damping ratio does Den Hartog's tuning give "
        "it?",
        100 * zeta,
        f"ζ = √(3μ/8(1 + μ)³) = √(3 × {mu:g}/(8 × {number((1 + mu) ** 3, 4)})) = {number(100 * zeta)} %, with the "
        f"absorber tuned to f/(1 + μ) = {number(1 / (1 + mu), 4)} f.",
        unit="%",
        setup=dict(**UNIFORM, tmd=True, ratio=mu, gain=0.0),
    )


def _skyhook(rng: np.random.Generator) -> Question:
    zs, r = pick(rng, (0.2, 0.5, 0.7, 1)), pick(rng, (1, 2, 3, 5))
    t = float(skyhook(np.array([r]), 0.1, zs)[1][0])
    return Question(
        f"A machine on a spring mount, with a skyhook damper of ζ<sub>sky</sub> = {zs:g}. Leaving out its passive damper, what "
        f"is its transmissibility from the floor at {r:g} times its natural frequency?",
        t,
        f"|X/Y| = 1/|1 − r² + 2iζr| = 1/√(({number(1 - r * r)})² + ({number(2 * zs * r)})²) = {number(t)}: the "
        "skyhook damper pushes against the sky, so it never carries the floor's motion through.",
        setup=dict(zeta_passive=0.1, zeta_sky=zs, tab=TAB_SKYHOOK),
    )


def _lag_phase(rng: np.random.Generator) -> Question:
    tau, f = pick(rng, (5, 10, 20)), pick(rng, (5, 10, 20, 50))
    phase = 2 * math.degrees(math.atan(TWO_PI * f * tau / 1000))
    return Question(
        f"A loop has two first-order lags, each τ = {tau:g} ms. How far do they turn the feedback's phase at {f:g} Hz?",
        phase,
        f"Each lag 1/(1 + iωτ) turns it by arctan(ωτ) = arctan(2π × {f:g} × {tau / 1000:g}) = "
        f"{number(phase / 2)}°, so the two by {number(phase)}°. Near 180° the force that should oppose the velocity "
        "helps it.",
        unit="°",
        tolerance=0.01,
        setup=dict(**UNIFORM, lag_ms=tau, gain=100.0, tab=TAB_LOCUS),
    )


def _too_much_gain(rng: np.random.Generator) -> Question:
    return Question(
        "Collocated velocity feedback at the uniform chain's tip: the gain is raised from 20 to 2000 N·s/m. What "
        "happens to the higher modes' damping?",
        0,
        "It falls again: the actuator holds m4 nearly still, and the chain vibrates round it as the chain with m4 "
        "fixed, lightly damped. Each mode's damping has an optimal gain, as a damper's does.",
        choices=("It falls again", "It keeps growing", "It stays where it was at 20 N·s/m", "The loop goes unstable"),
        setup=dict(**UNIFORM, gain=2000.0, tab=TAB_GAIN),
        predict=True,
    )


def _non_collocated(rng: np.random.Generator) -> Question:
    return Question(
        "The uniform chain: velocity measured at m1, the force on m4, g = 20 N·s/m. What does the loop do?",
        0,
        "It goes unstable (from 6.1 N·s/m): in mode 2 m1 and m4 move against each other, so the force feeds mode 2 "
        "energy faster than its damping takes it out.",
        choices=("It goes unstable", "It damps every mode", "It damps mode 1 only, the rest unchanged"),
        setup=dict(**UNIFORM, sensor=1, actuator=4, gain=20.0, tab=TAB_LOCUS),
        predict=True,
    )


def _lagged(rng: np.random.Generator) -> Question:
    return Question(
        "Collocated velocity feedback at the uniform chain's tip, g = 100 N·s/m, with two 20 ms lags in the loop. "
        "Stable?",
        0,
        "No: above 96 N·s/m the two lags turn the force far enough at a high mode that it feeds the motion. "
        "Collocation alone doesn't guarantee stability; lag limits the gain.",
        choices=("No, it's unstable", "Yes: collocated loops are stable at any gain"),
        setup=dict(**UNIFORM, gain=100.0, lag_ms=20.0, tab=TAB_LOCUS),
        predict=True,
    )


def _passive_damping(rng: np.random.Generator) -> Question:
    low, high = skyhook(np.array([5.0]), 0.1, 0.0)[0][0], skyhook(np.array([5.0]), 0.5, 0.0)[0][0]
    return Question(
        "A passive isolator's damper (to the floor) goes from ζ = 0.1 to 0.5. What happens to its transmissibility at "
        "5 times its natural frequency?",
        0,
        f"It gets worse, {number(float(low))} to {number(float(high))}: above √2 f<sub>n</sub> the damper carries the "
        "floor's motion through. A skyhook damper doesn't.",
        choices=("It gets worse", "It gets better", "It stays the same"),
        setup=dict(zeta_passive=0.5, zeta_sky=0.0, tab=TAB_SKYHOOK),
        predict=True,
    )


def _gain_spec(rng: np.random.Generator) -> Question:
    a = int(pick(rng, (2, 4)))
    lag = pick(rng, (0, 5, 10))  # ms
    system = ChainSystem(np.array(UNIFORM["masses"]), np.array(UNIFORM["springs"]), np.array(UNIFORM["dampers"]))
    z = np.array([_least_damped(system, Controller(a - 1, a - 1, float(g), lag / 1000)) for g in SPEC_GAINS])
    best = int(np.argmax(z))
    target = round(0.9 * float(z[best]), 3)
    design = round(float(SPEC_GAINS[best]), 2)
    window = SPEC_GAINS[z >= target]

    def goal(t: ControlTrial):
        system = t.system()
        n = system.n
        ctrl = Controller(min(t.sensor, n) - 1, min(t.actuator, n) - 1, t.gain, t.lag_ms / 1000)
        least = _least_damped(system, ctrl)
        return (
            Requirement(
                "The uniform four-mass chain", f"{n} masses", n == 4 and _same_chain(system, ChainSystem.uniform(4))
            ),
            kept("The sensor's mass", t.sensor, a),
            kept("The actuator's mass", t.actuator, a),
            kept("The lags", t.lag_ms, lag, "ms")
            if lag
            else Requirement("No lag", f"{number(t.lag_ms)} ms", t.lag_ms == 0),
            Requirement("Stable", "yes" if math.isfinite(least) else "no", math.isfinite(least)),
            at_least("The least damped mode's ζ", 100 * least, 100 * target, "%"),
        )

    lagged = f", with two {lag:g} ms lags in the loop" if lag else ""
    return Question(
        f"The uniform chain (four 1 kg masses, 400 N/m, 2 N·s/m dampers), collocated velocity feedback at m{a}{lagged}. "
        f"Choose the gain so that every mode still oscillating has ζ ≥ {number(100 * target)} %.",
        0,
        f"Too little gain leaves mode 1 lightly damped; too much pins m{a}, and the chain vibrates round it, lightly "
        f"damped again. The least damped mode's ζ peaks at {number(100 * float(z[best]))} % near g = {number(design)} "
        f"N·s/m; anything from about {number(float(window.min()))} to {number(float(window.max()))} N·s/m meets it.",
        setup=dict(**UNIFORM, sensor=a, actuator=a, gain=0.5, lag_ms=float(lag), tab=TAB_GAIN),
        goal=goal,
        design=dict(**UNIFORM, sensor=a, actuator=a, gain=design, lag_ms=float(lag), tab=TAB_GAIN),
    )


def _same_chain(a: ChainSystem, b: ChainSystem) -> bool:
    return a.n == b.n and all(
        np.allclose(x, y) for x, y in ((a.masses, b.masses), (a.stiffness, b.stiffness), (a.damping, b.damping))
    )


PROBLEMS = tuple(
    Problem(key, title, "active_control_page", f"See {page_link('active_control_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("one-mass", "An active damper", "feedback", "Feedback", _one_mass),
        ("modal-estimate", "A mode's added damping", "feedback", "Feedback", _modal_estimate),
        ("spillover", "Spillover", "collocation", "Collocation and spillover", _spillover),
        ("den-hartog", "A TMD's damping", "tmd", "Active or passive?", _den_hartog),
        ("skyhook", "Skyhook isolation", "skyhook", "Skyhook", _skyhook),
        ("lag-phase", "Two lags' phase", "lag", "Lag", _lag_phase),
        ("too-much-gain", "Predict: too much gain", "feedback", "Feedback", _too_much_gain),
        (
            "non-collocated",
            "Predict: sensor and actuator apart",
            "collocation",
            "Collocation and spillover",
            _non_collocated,
        ),
        ("lagged", "Predict: a lagging loop", "lag", "Lag", _lagged),
        ("passive-damping", "Predict: more passive damping", "skyhook", "Skyhook", _passive_damping),
        ("gain-spec", "Spec: the gain", "feedback", "Feedback", _gain_spec),
    )
)

TIPS = {
    "dof": "<p>The number of masses: the same chain as the Simulation page.</p>",
    "sensor": "<p>The mass whose velocity is measured.</p>",
    "actuator": "<p>The mass the actuator pushes on (against the ground).</p>",
    "gain": "<p>u = −g ẋ<sub>sensor</sub>: the gain g, in N·s/m (a damper's units).</p>",
    "lag": "<p>The time constant of each of two first-order lags in the loop (sensor filter and actuator). 0: "
    "ideal.</p>",
    "inp": "<p>Where the disturbing force acts, for the receptance plot.</p>",
    "out": "<p>Where the response is measured, for the receptance plot.</p>",
    "tmd": "<p>Hang a Den Hartog tuned mass damper on the actuator's mass, tuned to mode 1, instead of the "
    "control (dashed curve).</p>",
    "ratio": "<p>The absorber's mass, as a fraction of mode 1's modal mass at that mass.</p>",
    "zp": "<p>The passive mount's damping ratio (a damper to the floor).</p>",
    "zs": "<p>The skyhook damper's ratio (a force −c ẋ against an inertial frame).</p>",
}
FRF_TIP = "<p>The receptance from the disturbance to the response: open loop, closed loop, and with the TMD.</p>"
LOCUS_TIP = (
    "<p>The closed-loop poles as the gain rises from 0 (open dots) to 2000 N·s/m; the filled dots are at "
    "the gain set. Right of Re = 0 is unstable.</p>"
)
GAIN_TIP = (
    "<p>Each mode's damping ratio against the gain (followed by continuity); the dotted line is the gain set.</p>"
)
SKY_TIP = "<p>Transmissibility from floor to machine: passive damper, skyhook, and both.</p>"

NOTES_HTML = load_notes("active_control", figures=active_control_figures.FIGURES, trials=trials_html(TRIALS))


class ActiveControlPage(QtWidgets.QWidget):
    """Controls on the left, the receptances in the centre, the locus and damping on the right."""

    dof_requested = QtCore.Signal(int)
    system_requested = QtCore.Signal(object)
    edit_parameters = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system = ChainSystem.uniform(4)
        self._dirty = True

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Each loads the uniform chain into the Simulation page and sets up the loop to "
            "show one idea from the Theory tab.</p>",
        )
        self.trials.picked.connect(self.load_trial)

        ch = Form("Chain")
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.valueChanged.connect(self.dof_requested)
        ch.add("Number of masses:", self.dof, TIPS["dof"])
        link = QtWidgets.QLabel("Masses, springs and dampers are edited on the <a href='#sim'>Simulation page</a>.")
        link.setWordWrap(True)
        mute(link)
        link.linkActivated.connect(self.edit_parameters)
        ch.row(link)

        loop = Form("Feedback loop")
        self.sensor = loop.number("Sensor on mass:", 1, MAX_DOF, 4, 0, "", TIPS["sensor"])
        self.actuator = loop.number("Actuator on mass:", 1, MAX_DOF, 4, 0, "", TIPS["actuator"])
        self.gain = loop.number("Gain g:", 0.0, 5000.0, 20.0, 2, " N·s/m", TIPS["gain"])
        self.lag = loop.number("Lags τ:", 0.0, 500.0, 0.0, 1, " ms", TIPS["lag"])
        self.loop_info = note_label()
        loop.row(self.loop_info)

        plotting = Form("Receptance")
        self.inp = plotting.number("Force on mass:", 1, MAX_DOF, 4, 0, "", TIPS["inp"])
        self.out = plotting.number("Response of mass:", 1, MAX_DOF, 4, 0, "", TIPS["out"])
        self.tmd = QtWidgets.QCheckBox("Compare a tuned mass damper")
        self.tmd.setToolTip(TIPS["tmd"])
        plotting.row(self.tmd)
        self.ratio = plotting.number("Its mass ratio:", 0.005, 0.5, 0.05, 3, "", TIPS["ratio"])

        connect_all(
            (self.sensor, self.actuator, self.gain, self.lag, self.inp, self.out, self.tmd, self.ratio), self.redraw
        )

        # --- centre
        self.frf_plot = plot("|Receptance|", "m/N", "Frequency", "Hz", log_x=True, log_y=True, height=200)
        self.frf_plot.setToolTip(FRF_TIP)
        add_legend(self.frf_plot.getPlotItem(), offset=(-10, 10))
        self.open_item = pg.PlotDataItem(name="no control")
        self.closed_item = pg.PlotDataItem(name="velocity feedback")
        self.tmd_item = pg.PlotDataItem(name="tuned mass damper")
        for item in (self.open_item, self.tmd_item, self.closed_item):
            self.frf_plot.addItem(item)
        self.readout = readout_label()

        # --- right
        self.locus_plot = plot("Im λ / 2π", "Hz", "Re λ", "1/s", height=200)
        self.locus_plot.setToolTip(LOCUS_TIP)
        self.locus_items = marker(4)
        self.locus_open = marker(9)
        self.locus_now = marker(9)
        self.locus_axis = pg.InfiniteLine(angle=90, pos=0)
        for item in (self.locus_axis, self.locus_items, self.locus_open, self.locus_now):
            self.locus_plot.addItem(item)
        self.gain_plot = plot("Damping ratio ζ (%)", None, "Gain g (N·s/m)", None, log_x=True, height=200)
        self.gain_plot.setToolTip(GAIN_TIP)
        add_legend(self.gain_plot.getPlotItem(), offset=(-10, 10))
        self.gain_line = pg.InfiniteLine(angle=90)
        self.gain_zero = pg.InfiniteLine(angle=0, pos=0)
        self.gain_plot.addItem(self.gain_line)
        self.gain_plot.addItem(self.gain_zero)
        self.gain_curves: list[pg.PlotDataItem] = []
        sky = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(sky)
        sf = Form("")
        self.zp = sf.number("Passive ζ:", 0.0, 2.0, 0.1, 3, "", TIPS["zp"])
        self.zs = sf.number("Skyhook ζ:", 0.0, 2.0, 0.5, 3, "", TIPS["zs"])
        connect_all((self.zp, self.zs), self._draw_tab)
        self.sky_plot = plot(
            "Transmissibility |X/Y|", None, "Frequency ratio f/f<sub>n</sub>", None, log_x=True, log_y=True, height=200
        )
        self.sky_plot.setToolTip(SKY_TIP)
        add_legend(self.sky_plot.getPlotItem(), offset=(-10, 10))
        self.sky_items = [
            pg.PlotDataItem(name="passive (damper to the floor)"),
            pg.PlotDataItem(name="skyhook (damper to the sky)"),
            pg.PlotDataItem(name="both"),
        ]
        for item in self.sky_items:
            self.sky_plot.addItem(item)
        sv.addWidget(sf)
        sv.addWidget(self.sky_plot, 1)
        self.tab_note = note_label()

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(column(self.locus_plot, stretch=False), "Root locus")
        self.tabs.addTab(column(self.gain_plot, stretch=False), "Damping vs gain")
        self.tabs.addTab(sky, "Skyhook")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._draw_tab)
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.tabs, 1)
        rv.addWidget(self.tab_note)

        three_panes(self, column(self.trials, ch, loop, plotting), stack((self.frf_plot, 1), (self.readout, 0)), right)
        self.apply_theme()
        self.set_system(self.system)

    # ------------------------------------------------------------- inputs
    def set_system(self, system: ChainSystem, result=None) -> None:
        self.system = system
        n = system.n
        set_quietly(self.dof, n)
        for box in (self.sensor, self.actuator, self.inp, self.out):
            box.blockSignals(True)
            box.setMaximum(n)
            box.blockSignals(False)
        self._gain_built = -1
        self._dirty = True
        if self.isVisible():
            self.redraw()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty:
            self.redraw()

    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note); drawn when shown."""
        self.apply_trial(ControlTrial("", "", **setup), draw=self.isVisible())

    def practice_state(self) -> ControlTrial:
        """The chain and the loop, as a trial: what a practice spec checks."""
        s = self.system
        return ControlTrial(
            "",
            "",
            int(self.sensor.value()),
            int(self.actuator.value()),
            self.gain.value(),
            self.lag.value(),
            self.tmd.isChecked(),
            self.tabs.currentIndex(),
            self.zp.value(),
            self.zs.value(),
            int(self.inp.value()),
            int(self.out.value()),
            self.ratio.value(),
            tuple(float(v) for v in s.masses),
            tuple(float(v) for v in s.stiffness),
            tuple(float(v) for v in s.damping),
        )

    def apply_trial(self, t: ControlTrial, draw: bool = True) -> None:
        system = t.system()
        self.system_requested.emit(system)
        if not _same_chain(self.system, system):
            self.set_system(system)  # nobody passed it back: show it here alone
        for box, value in (
            (self.sensor, t.sensor),
            (self.actuator, t.actuator),
            (self.gain, t.gain),
            (self.lag, t.lag_ms),
            (self.inp, t.inp),
            (self.out, t.out),
            (self.tmd, t.tmd),
            (self.ratio, t.ratio),
            (self.zp, t.zeta_passive),
            (self.zs, t.zeta_sky or 0.5),
        ):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex(t.tab)
        self.tabs.blockSignals(False)
        if draw:
            self.redraw()
        else:
            self._dirty = True

    def controller(self, gain: float | None = None) -> Controller:
        n = self.system.n
        return Controller(
            min(int(self.sensor.value()), n) - 1,
            min(int(self.actuator.value()), n) - 1,
            self.gain.value() if gain is None else gain,
            self.lag.value() / 1000.0,
        )

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.open_item.setPen(pg.mkPen(c.grey, width=2))
        self.closed_item.setPen(pg.mkPen(c.mass[0], width=2.5))
        self.tmd_item.setPen(pg.mkPen(c.mass[1], width=2, style=QtCore.Qt.PenStyle.DashLine))
        mark(self.locus_items, c.mass[0], filled=True)
        mark(self.locus_open, c.grey)
        mark(self.locus_now, c.force, filled=True)
        self.locus_axis.setPen(pg.mkPen(c.strong, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.gain_line.setPen(pg.mkPen(c.grey, width=1.5, style=QtCore.Qt.PenStyle.DotLine))
        self.gain_zero.setPen(pg.mkPen(c.faint, width=1))
        for item, color, style in zip(
            self.sky_items,
            (c.grey, c.mass[0], c.mass[1]),
            (QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenStyle.DashLine),
        ):
            item.setPen(pg.mkPen(color, width=2, style=style))
        self._gain_built = -1
        self._dirty = True
        if self.isVisible():
            self.redraw()

    def redraw(self) -> None:
        self._dirty = False
        s = self.system
        ctrl = self.controller()
        self.ratio.setEnabled(self.tmd.isChecked())
        w, _ = undamped_modes(s)
        top = 2.0 * float(w[-1]) / (2 * np.pi) if w[-1] > 0 else 10.0
        bottom = max(0.2 * float(w[w > 1e-6][0]) / (2 * np.pi), 1e-3) if np.any(w > 1e-6) else 0.05
        f = np.geomspace(bottom, top, 1500)
        n = s.n
        out, inp = min(int(self.out.value()), n) - 1, min(int(self.inp.value()), n) - 1
        A0, B0, C0 = closed_loop(s, None)
        A1, B1, C1 = closed_loop(s, ctrl)
        H0 = np.abs(frf(A0, B0, C0, out, inp, f))
        stable = max_real(A1) < 0
        H1 = np.abs(frf(A1, B1, C1, out, inp, f)) if stable else None
        self.open_item.setData(f, H0)
        self.closed_item.setData(f, H1) if H1 is not None else self.closed_item.setData([], [])
        tmd_text = ""
        if self.tmd.isChecked() and w[0] > 1e-6:
            t = tmd_for(s, ctrl.actuator, self.ratio.value())
            A2, B2, C2 = closed_loop(s, None, t)
            H2 = np.abs(frf(A2, B2, C2, out, inp, f))
            self.tmd_item.setData(f, H2)
            tmd_text = f" TMD ({t[1] * 1e3:.3g} g on m{ctrl.actuator + 1}): peak {H2.max():.3g} m/N."
        else:
            self.tmd_item.setData([], [])
        self.frf_plot.setXRange(np.log10(f[0]), np.log10(f[-1]), padding=0)

        open_poles = oscillating_poles(A0)
        closed = oscillating_poles(A1)
        est = modal_damping_estimate(s, ctrl)
        rows = "".join(
            f"<tr><td>mode {j + 1}</td><td align='right'>{p.freq_hz:.3g} Hz</td>"
            f"<td align='right'>{100 * p.zeta:.3g} %</td><td align='right'>{100 * (p.zeta + est[j]):.3g} %</td></tr>"
            for j, p in enumerate(open_poles)
        )
        closed_rows = ", ".join(f"{p.freq_hz:.3g} Hz at {100 * p.zeta:.3g} %" for p in closed)
        self.readout.setText(
            (
                ""
                if stable
                else "<b>The loop is unstable</b> (a pole in the right half plane); no closed-loop receptance. "
            )
            + f"Peak without control {H0.max():.3g} m/N"
            + (f", with it {H1.max():.3g} m/N." if H1 is not None else ".")
            + tmd_text
            + "<table cellspacing='0' cellpadding='2'><tr><th></th><th>f</th><th>ζ open</th>"
            "<th>ζ + gφ(a)φ(s)/2ω</th></tr>" + rows + "</table>" + f"Closed loop, exactly: {closed_rows}."
        )
        collocated = ctrl.sensor == ctrl.actuator
        self.loop_info.setText(
            "Collocated: a damper to the ground (ideally stable at any gain)."
            if collocated
            else "Not collocated: modes where the two masses move against each other are fed energy."
        )
        self._draw_tab()
        settle_plots(self)

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        s = self.system
        ctrl = self.controller()
        if i == TAB_LOCUS:
            roots = []
            for g in GAINS:
                lam = np.linalg.eigvals(closed_loop(s, self.controller(float(g)))[0])
                roots.append(lam[lam.imag >= 0])
            pts = np.concatenate(roots)
            self.locus_items.setData(pts.real, pts.imag / (2 * np.pi))
            lam0 = np.linalg.eigvals(closed_loop(s, None)[0])
            lam0 = lam0[lam0.imag >= 0]
            self.locus_open.setData(lam0.real, lam0.imag / (2 * np.pi))
            lam = np.linalg.eigvals(closed_loop(s, ctrl)[0])
            lam = lam[lam.imag >= 0]
            self.locus_now.setData(lam.real, lam.imag / (2 * np.pi))
            w = undamped_modes(s)[0]
            fmax = float(w[-1]) / (2 * np.pi) * 1.3
            reals = pts.real[np.abs(pts.real) < 5 * fmax * 2 * np.pi]
            self.locus_plot.setXRange(
                max(float(reals.min()), -4 * fmax * 2 * np.pi * 0.3), max(float(reals.max()), 1.0) * 1.1, padding=0.05
            )
            self.locus_plot.setYRange(0.0, fmax, padding=0.02)
            self.tab_note.setText(
                "Open dots: the open-loop poles; small dots: g from 0 to 2000 N·s/m; red: the gain "
                "set. "
                + (
                    "All stay left of Re λ = 0."
                    if max(pts.real) < 0
                    else "Some cross into the right half plane: unstable."
                )
            )
        elif i == TAB_GAIN:
            n = s.n
            c = colors
            if self._gain_built != n:
                for item in self.gain_curves:
                    self.gain_plot.removeItem(item)
                legend = self.gain_plot.getPlotItem().legend
                legend.clear()
                self.gain_curves = []
                for j in range(n):
                    item = pg.PlotDataItem(pen=pg.mkPen(c.mode[j % len(c.mode)], width=2))
                    self.gain_plot.addItem(item)
                    legend.addItem(item, f"mode {j + 1}")
                    self.gain_curves.append(item)
                self._gain_built = n
            g = GAINS[1:]
            z = tracked_damping(s, ctrl, g)
            for j, item in enumerate(self.gain_curves):
                item.setData(g, 100 * z[:, j], connect="finite")
            self.gain_line.setValue(np.log10(max(ctrl.gain, g[0])))
            self.gain_plot.setXRange(np.log10(g[0]), np.log10(g[-1]), padding=0)
            finite = z[np.isfinite(z)]
            self.gain_plot.setYRange(
                100 * min(-0.05, float(finite.min())), 100 * min(1.05, float(finite.max()) + 0.05), padding=0
            )
            self.tab_note.setText(
                "A curve that ends has stopped oscillating: its pole has reached the real axis (overdamped)."
            )
        elif i == TAB_SKYHOOK:
            r = np.geomspace(0.1, 10.0, 600)
            for item, y in zip(self.sky_items, skyhook(r, self.zp.value(), self.zs.value())):
                item.setData(r, y)
            self.sky_plot.setXRange(-1, 1, padding=0)
            p, sk, _ = skyhook(np.array([1.0, 5.0]), self.zp.value(), self.zs.value())
            self.tab_note.setText(
                f"At f<sub>n</sub>: passive {p[0]:.3g}, skyhook {sk[0]:.3g}. At 5 f<sub>n</sub>: "
                f"passive {p[1]:.3g}, skyhook {sk[1]:.3g}."
            )
        else:
            self.tab_note.setText("")
