"""Duffing & Mathieu page: a mass on a hardening or softening spring, driven or pumped, swept through its resonances.

Left: the oscillator, the drive and the playback. Centre: the amplitude against the
frequency (harmonic balance's branches, backbone and folds, for the drive's first harmonic
and the pump's half harmonic, under each drive cycle's measured ones) and the displacement's
strip chart. Right: the phase plane with its Poincaré section, the Ince–Strutt chart of the
pumped spring's tongues, and the notes (their figures: gui/duffing_figures.py). The page has its own model
(core/duffing.py), not the chain.

What it draws and prints is worked out in its view model (`view_models/duffing.py`, no Qt), which also keeps the
run's record (`Record`: the samples, each drive cycle, the Poincaré section) and the page's settings
(`DuffingSetup`); this module holds the controls, the frame loop and the practice problems, and binds them. Change
a readout or a curve there, and its tests in `tests/test_view_models.py`.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.duffing import (
    PRESETS,
    TWO_PI,
    Duffing,
    DuffingSimulator,
    Sweep,
    parametric_amplitudes,
    response_curve,
)
from ..core.practice import Problem, Question, Requirement, at_least, kept, number, pick, rounded
from ..view_models import duffing as view
from ..view_models.duffing import CHART_DEPTHS, CHART_RATIOS, GROWTH_TOP, LEGEND, LEGS, DuffingSetup
from . import duffing_figures, weak
from .frame_loop import FrameLoop
from .layouts import ColumnScroll, FlowLayout, form_layout, form_row, plot_height
from .lesson import Dot, RunButton
from .links import load_notes, page_link
from .maths import markup
from .notes_view import NotesBrowser
from .panels import spin
from .scrub import Scrub
from .style import colors
from .theming import add_legend, mute

FRAME_BUDGET = 0.008  # s of computing per frame: past it the page runs slower than asked
SPEEDS = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
CUSTOM = "Custom"

TIPS = {
    "mass": "<p>The mass m.</p>",
    "stiffness": "<p>The spring's linear stiffness k: its stiffness at small amplitude.</p>",
    "damping": "<p>The viscous damping c.</p>",
    "cubic": "<p>The spring's cubic term k<sub>3</sub>: its force is k x + k<sub>3</sub> x³. Positive hardens "
    "the spring (a large swing is stiffer, so faster), negative softens it, down to nothing at the "
    "escape amplitude √(k / −k<sub>3</sub>), where the potential has a hilltop.</p>",
    "force": "<p>The drive's amplitude F: the force is F cos φ, with φ advancing at the drive's frequency.</p>",
    "pump": "<p>The pump μ: the linear stiffness pulses as k (1 + μ cos φ), at the drive's frequency "
    "(parametric excitation, the Mathieu equation). Near twice the natural frequency rest turns "
    "unstable and the mass swings at half the pump's frequency.</p>",
    "noise": "<p>A small random force, a fresh value each step: something for a state the pump makes "
    "unstable to grow from. Without it a mass at rest stays at rest.</p>",
    "freq": "<p>The drive's frequency. Held, it stays here; sweeping, this follows the sweep. Typing a frequency "
    "while sweeping stops the sweep there.</p>",
    "range": "<p>The sweep's lower and upper frequencies. A sweep up runs from the first to the second.</p>",
    "time": "<p>How long each leg of a sweep takes, in simulated time. Slower sweeps stay nearer the steady "
    "response and leave a branch nearer its fold.</p>",
    "branch": "<p>Put the mass on a steady response of harmonic balance at the held frequency, at the drive's "
    "phase now: the drive's first harmonic, or with no drive force the pump's half-harmonic swing "
    "(the lower then is rest). Between the folds there are three: the lower and upper are stable, "
    "the middle one is not.</p>",
}
RESPONSE_TIP = (
    "<p>The amplitude of the displacement's first harmonic against the drive's frequency. Solid: harmonic "
    "balance's stable steady states; dashed: its unstable ones (the middle branch); dotted: the backbone, the "
    "free oscillation's frequency at each amplitude; grey: the spring without its cubic term. The arrows are "
    "the jumps from the folds, where a branch ends. The dots are each drive cycle's measured first harmonic, "
    "coloured by the sweep's direction.</p><p>Pumped, the second colour is the half harmonic (Ω/2) the pump "
    "drives, rest included along the axis (dashed inside the tongue, where it is unstable), and the triangles "
    "its measured amplitude.</p>"
)
CHART_TIP = (
    "<p>The Ince–Strutt chart: how fast the pumped linear spring's swing grows (coloured) or dies away "
    "(blank), against the pump's frequency over the natural frequency and its depth μ, at this damping. "
    "Rest is unstable inside the tongues, which reach down toward Ω = 2ω<sub>n</sub>/n: half the pump's "
    "frequency, then the pump's own, then a third... Damping lifts their tips off the axis. The dashed line "
    "is harmonic balance's first tongue; the circle is the page's pump. Click to pump there.</p>"
)
PHASE_TIP = (
    "<p>The phase plane: velocity against displacement over the last few drive cycles. The dots are the "
    "Poincaré section, the state once per drive cycle (where the force peaks): one dot for a steady response "
    "at the drive's frequency, a few for a subharmonic, a cloud that never settles for chaos. Only while the "
    "frequency is held: a sweep's would be a different section each cycle.</p>"
)


def presets_html() -> str:
    """The presets as the Theory tab lists them."""
    return "<ol>" + "".join(f"<li><b>{p.name}.</b> {markup(p.note)}</li>" for p in PRESETS) + "</ol>"


NOTES_HTML = load_notes("duffing", figures=duffing_figures.FIGURES, presets=presets_html())


PRACTICE_SETUP = DuffingSetup  # what a practice problem's setup and the page's practice_state are


# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.
def _backbone(rng: np.random.Generator) -> Question:
    m, k = pick(rng, (0.5, 1, 2)), pick(rng, (200, 400, 1000))
    k3, a = pick(rng, (5e4, 1e5, 2e5)), pick(rng, (10, 20, 30, 40))  # a in mm
    osc = Duffing(m, k, 1.0, k3)
    kap = float(osc.kappa(a / 1000))
    f = float(osc.backbone_hz(a / 1000))
    return Question(
        f"A {m:g} kg mass on a hardening spring, k = {k:g} N/m and k₃ = {k3:g} N/m³, swings freely at {a:g} mm. At "
        "what frequency, by harmonic balance?",
        f,
        f"The spring acts on the first harmonic as κ(a) = k + ¾ k₃ a² = {k:g} + 0.75 × {k3:g} × {a / 1000:g}² = "
        f"{number(kap)} N/m, so f = √(κ/m)/2π = {number(f)} Hz, against {number(osc.natural_hz)} Hz for small "
        "swings.",
        unit="Hz",
        setup=dict(mass=m, stiffness=k, cubic=k3, force=0.5, damping=0.5, start_hz=0.5 * osc.natural_hz, end_hz=2 * f),
    )


def _stiffening(rng: np.random.Generator) -> Question:
    k, k3, a = pick(rng, (200, 400, 1000)), pick(rng, (5e4, 1e5, 2e5, 5e5)), pick(rng, (10, 20, 30))
    added = 100 * 0.75 * k3 * (a / 1000) ** 2 / k
    return Question(
        f"A spring with k = {k:g} N/m and k₃ = {k3:g} N/m³ swings at {a:g} mm. By how many percent does the cubic "
        "stiffen it, as harmonic balance sees it?",
        added,
        f"κ(a) = k + ¾ k₃ a²: the cubic adds ¾ k₃ a²/k = 0.75 × {k3:g} × {a / 1000:g}²/{k:g} = {number(added)} %.",
        unit="%",
    )


def _escape(rng: np.random.Generator) -> Question:
    k, k3 = pick(rng, (100, 400, 1000, 2500)), pick(rng, (-1e4, -2e4, -4e4, -1e5))
    x = math.sqrt(k / -k3)
    return Question(
        f"A softening spring, k = {k:g} N/m and k₃ = {f"{k3:g}".replace("-", "−")} N/m³. Past what amplitude does it push the mass "
        "outwards (its escape amplitude)?",
        1000 * x,
        f"k x + k₃ x³ is back to zero at x = √(k/−k₃) = √({k:g}/{number(-k3)}) = {number(1000 * x)} mm, the "
        "potential's hilltop. (The force peaks at 1/√3 of that.)",
        unit="mm",
        setup=dict(stiffness=k, cubic=k3, force=0.5, damping=0.8, start_hz=0.2, end_hz=2 * math.sqrt(k) / TWO_PI),
    )


def _pump_frequency(rng: np.random.Generator) -> Question:
    m, k = pick(rng, (0.5, 1, 2, 5)), pick(rng, (100, 400, 1000, 4000))
    fn = math.sqrt(k / m) / TWO_PI
    return Question(
        f"A {m:g} kg mass on a {k:g} N/m spring. At what frequency should its stiffness pulse to swing it hardest "
        "from rest (the first tongue's centre)?",
        2 * fn,
        f"f<sub>n</sub> = √(k/m)/2π = {number(fn)} Hz; the principal parametric resonance pumps at twice it, "
        f"{number(2 * fn)} Hz, and the mass swings at {number(fn)} Hz, half the pump's frequency.",
        unit="Hz",
        setup=dict(
            mass=m,
            stiffness=k,
            cubic=0.0,
            force=0.0,
            pump=0.3,
            noise=1e-3,
            freq_hz=rounded(2 * fn, 4),
            mode=Sweep.HOLD,
            start_hz=1.5 * fn,
            end_hz=2.5 * fn,
        ),
    )


def _threshold(rng: np.random.Generator) -> Question:
    m, k, zeta = pick(rng, (0.5, 1, 2)), pick(rng, (100, 400, 1000)), pick(rng, (0.005, 0.01, 0.02, 0.05))
    c = 2 * zeta * math.sqrt(k * m)
    return Question(
        f"A {m:g} kg mass on a {k:g} N/m spring with damping c = {number(c)} N·s/m. Pumped at exactly twice its "
        "natural frequency, how deep must the pump μ be to make rest unstable?",
        4 * zeta,
        f"At Ω = 2ω<sub>n</sub>, (k − mΩ²/4)² + (cΩ/2)² = (μk/2)² leaves c ω<sub>n</sub> = μk/2, so μ = 2cω<sub>n</sub>/k "
        f"= 4ζ, with ζ = c/2√(km) = {number(zeta)}: μ = {number(4 * zeta)}.",
        tolerance=0.02,
    )


def _growth_time(rng: np.random.Generator) -> Question:
    fn, zeta, mu = pick(rng, (2, 3.18, 5, 10)), pick(rng, (0.005, 0.01, 0.02)), pick(rng, (0.1, 0.2, 0.3))
    rate = (mu / 4 - zeta) * TWO_PI * fn
    return Question(
        f"A spring with f<sub>n</sub> = {fn:g} Hz and ζ = {zeta:g} is pumped at 2f<sub>n</sub> with μ = {mu:g}. How "
        "long does a small swing take to grow e times?",
        1 / rate,
        f"σ ≈ (μ/4 − ζ) ω<sub>n</sub> = ({mu:g}/4 − {zeta:g}) × 2π × {fn:g} = {number(rate)} /s, so 1/σ = "
        f"{number(1 / rate)} s. (Floquet's exact rate is within 1 % of this here.)",
        unit="s",
    )


def _settled_swing(rng: np.random.Generator) -> Question:
    m, k, c = 1.0, 400.0, 1.0
    k3, mu = pick(rng, (5e4, 1e5, 2e5)), pick(rng, (0.2, 0.3, 0.4))
    f = pick(rng, (6.4, 6.6, 6.8))
    osc = Duffing(m, k, c, k3, 0.0, mu)
    w = TWO_PI * f
    d = math.sqrt((mu * k / 2) ** 2 - (c * w / 2) ** 2)
    kap = m * w * w / 4 + d
    a = math.sqrt((kap - k) / (0.75 * k3))
    amps, stable = parametric_amplitudes(osc, f)
    assert abs(amps[stable].max() - a) < 1e-9 * a
    return Question(
        f"A 1 kg mass on a hardening spring, k = 400 N/m, k₃ = {k3:g} N/m³, c = 1 N·s/m, its stiffness pumped by "
        f"μ = {mu:g} at {f:g} Hz. What amplitude does its half-harmonic swing settle at?",
        1000 * a,
        f"x = a cos(Ωt/2 − θ): (κ(a) − mΩ²/4)² + (cΩ/2)² = (μk/2)². The stable branch is the one detuned the "
        f"cubic's way: κ(a) = mΩ²/4 + √((μk/2)² − (cΩ/2)²) = {number(m * w * w / 4)} + {number(d)} = {number(kap)} "
        f"N/m, so ¾ k₃ a² = {number(kap - k)} N/m and a = {number(1000 * a)} mm.",
        unit="mm",
        setup=dict(cubic=k3, force=0.0, pump=mu, noise=1e-3, freq_hz=f, mode=Sweep.HOLD, start_hz=5.0, end_hz=8.0),
    )


def _lean(rng: np.random.Generator) -> Question:
    return Question(
        "A hardening spring (k = 400 N/m, so 3.18 Hz for small swings) is swept up in frequency. Where does its "
        "swing jump down?",
        0,
        "Above the linear resonance, at 4.45 Hz: the curve leans right along the backbone, a big swing being a stiff "
        "one, and the upper branch carries the mass past 3.18 Hz to its fold.",
        choices=("Above 3.18 Hz", "At 3.18 Hz", "Below 3.18 Hz"),
        setup=dict(mode=Sweep.UP),
        predict=True,
    )


def _hysteresis(rng: np.random.Generator) -> Question:
    return Question(
        "The same hardening spring is swept up, then back down. Where does it jump up on the way down, against where "
        "it jumped down on the way up?",
        0,
        "Lower: the lower branch's fold is at 3.64 Hz, the upper one's at 4.45 Hz. Between them both branches are "
        "stable, and which the mass is on depends on where it came from.",
        choices=("Lower", "At the same frequency", "Higher"),
        setup=dict(mode=Sweep.UP_DOWN),
        predict=True,
    )


def _softening(rng: np.random.Generator) -> Question:
    return Question(
        "A softening spring (k₃ < 0, 3.18 Hz for small swings) is swept up. Where does its swing jump?",
        0,
        "Up, below 3.18 Hz (at 2.97 Hz): a softening spring's curve leans left, so the lower branch reaches its fold "
        "first and the mass jumps up to the upper one.",
        choices=("Up, below 3.18 Hz", "Down, above 3.18 Hz", "Nowhere: it softens smoothly"),
        setup=dict(damping=0.8, cubic=-2e4, force=1.25, start_hz=1.5, end_hz=4.5, freq_hz=1.5, mode=Sweep.UP),
        predict=True,
    )


def _half_frequency(rng: np.random.Generator) -> Question:
    return Question(
        "No force, but the spring's stiffness pulses ±30 % at 6.37 Hz, twice its natural frequency. At what frequency "
        "does the mass end up swinging?",
        0,
        "At half the pump's, 3.18 Hz: each pulse comes when the stretch adds energy, twice a swing. Rest is unstable "
        "there, and the noise grows into the swing.",
        choices=("3.18 Hz, half the pump's", "6.37 Hz, the pump's", "12.7 Hz, twice the pump's", "It stays at rest"),
        setup=dict(
            cubic=0.0,
            force=0.0,
            pump=0.3,
            noise=1e-3,
            freq_hz=rounded(20.0 / math.pi, 4),
            mode=Sweep.HOLD,
            start_hz=5.0,
            end_hz=8.0,
        ),
        predict=True,
    )


def _has_jump(osc: Duffing) -> bool:
    return bool(response_curve(osc, 3 * osc.natural_hz).folds)


def _jump_spec(rng: np.random.Generator) -> Question:
    F, k3 = pick(rng, (1.0, 2.0, 3.0)), pick(rng, (5e4, 1e5, 2e5))
    base = Duffing(force=F, cubic=k3)
    lo, hi = 0.01, 50.0
    for _ in range(40):  # the damping at which the folds vanish
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if _has_jump(base.with_(damping=mid)) else (lo, mid)
    least = hi
    design = rounded(1.2 * least, 3)
    lowest = math.floor(10_000 * response_curve(base.with_(damping=1.4 * least), 3 * base.natural_hz).peak) / 10

    def goal(t: DuffingSetup):
        osc = t.oscillator()
        peak = response_curve(osc, 3 * osc.natural_hz).peak
        jumps = _has_jump(osc)
        return (
            kept("The mass", t.mass, base.mass, "kg"),
            kept("k", t.stiffness, base.stiffness, "N/m"),
            kept("k₃", t.cubic, k3, "N/m³"),
            kept("The drive F", t.force, F, "N"),
            Requirement("No pump", f"μ = {number(t.pump)}", t.pump == 0),
            Requirement("No jump: one steady amplitude at every frequency", "jumps" if jumps else "no jump", not jumps),
            at_least("The peak swing", 1000 * peak, lowest, "mm"),
        )

    return Question(
        f"A 1 kg mass on a hardening spring (k = 400 N/m, k₃ = {k3:g} N/m³) driven by {F:g} N jumps on a sweep. "
        f"Choose its damping so that it doesn't jump at all, but still swings to {number(lowest)} mm or more at its "
        "peak.",
        0,
        f"More damping lowers the peak, so the curve leans less and stops folding over: the folds meet and vanish at "
        f"c ≈ {number(least)} N·s/m. Above that the peak keeps falling, to {number(lowest)} mm at about "
        f"{number(1.4 * least)} N·s/m: c = {number(design)} N·s/m, say.",
        setup=dict(force=F, cubic=k3, damping=1.0),
        goal=goal,
        design=dict(force=F, cubic=k3, damping=design),
    )


PROBLEMS = tuple(
    Problem(key, title, "duffing_page", f"See {page_link('duffing_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("backbone", "The backbone", "harmonic-balance", "Harmonic balance", _backbone),
        ("stiffening", "The cubic's share", "harmonic-balance", "Harmonic balance", _stiffening),
        ("escape", "The escape amplitude", "duffing", "The Duffing oscillator", _escape),
        ("pump-frequency", "Where to pump", "parametric", "Parametric excitation", _pump_frequency),
        ("threshold", "The pump's threshold", "parametric", "Parametric excitation", _threshold),
        ("growth-time", "How fast it grows", "parametric", "Parametric excitation", _growth_time),
        ("settled-swing", "A pumped swing's size", "parametric", "Parametric excitation", _settled_swing),
        ("lean", "Predict: sweeping up", "jump", "The jump", _lean),
        ("hysteresis", "Predict: sweeping back", "jump", "The jump", _hysteresis),
        ("softening", "Predict: a softening spring", "jump", "The jump", _softening),
        ("half-frequency", "Predict: pumped from rest", "parametric", "Parametric excitation", _half_frequency),
        ("jump-spec", "Spec: damping out the jump", "jump", "The jump", _jump_spec),
    )
)


def _plot(left: str, left_units: str | None, bottom: str, bottom_units: str | None) -> pg.PlotWidget:
    p = pg.PlotWidget()
    p.setMenuEnabled(False)
    p.hideButtons()
    p.showGrid(x=True, y=True, alpha=0.15)
    p.setLabel("left", left, units=left_units)
    p.setLabel("bottom", bottom, units=bottom_units)
    return p


class DuffingPage(QtWidgets.QWidget):
    """Controls on the left, the response and the strip chart in the centre, phase plane and notes on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.sim = DuffingSimulator()
        self.shown: view.Response | None = None  # the response plot's curves, by harmonic balance
        self.running = False  # at rest until started
        self.record = view.Record()  # the run's samples, and its cycles since the last sweep started
        self._y_top = 0.0
        self._loading = False

        # --- left: the oscillator, the drive, the playback
        self.preset = QtWidgets.QComboBox()
        self.preset.setAccessibleName("Preset")
        self.preset.addItems([p.name for p in PRESETS] + [CUSTOM])
        self.preset.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.preset_note = QtWidgets.QLabel()
        self.preset_note.setWordWrap(True)
        mute(self.preset_note)

        osc = QtWidgets.QGroupBox("Oscillator")
        form = form_layout(osc)
        self.mass = spin(0.01, 1000.0, 1.0, 3, " kg")
        self.stiffness = spin(0.0, 1e7, 400.0, 1, " N/m")
        self.damping = spin(0.0, 1e4, 1.0, 3, " N·s/m")
        self.cubic = spin(-1e9, 1e9, 1e5, 1, " N/m³")
        self.force = spin(0.0, 1e4, 2.0, 3, " N")
        self.pump = spin(0.0, 2.0, 0.0, 3)
        self.pump.setSingleStep(0.05)
        self.noise = spin(0.0, 100.0, 0.0, 4, " N")
        for text, box, key in (
            ("Mass m:", self.mass, "mass"),
            ("Stiffness k:", self.stiffness, "stiffness"),
            ("Damping c:", self.damping, "damping"),
            ("Cubic k₃:", self.cubic, "cubic"),
            ("Drive F:", self.force, "force"),
            ("Pump μ:", self.pump, "pump"),
            ("Noise (rms):", self.noise, "noise"),
        ):
            box.valueChanged.connect(self._on_params)
            form_row(form, text, box, TIPS[key])
        self.osc_info = QtWidgets.QLabel()
        self.osc_info.setWordWrap(True)
        mute(self.osc_info)
        form.addRow(self.osc_info)

        drive = QtWidgets.QGroupBox("Drive")
        dv = form_layout(drive)
        self.freq = spin(0.0, 1000.0, 3.0, 3, " Hz")
        self.start_hz = spin(0.0, 1000.0, 2.0, 3, " Hz")
        self.end_hz = spin(0.0, 1000.0, 6.0, 3, " Hz")
        self.sweep_time = spin(1.0, 1e5, 120.0, 0, " s")
        for text, box, key in (
            ("Frequency:", self.freq, "freq"),
            ("Sweep from:", self.start_hz, "range"),
            ("to:", self.end_hz, "range"),
            ("Each leg takes:", self.sweep_time, "time"),
        ):
            form_row(dv, text, box, TIPS[key])
        self.end_hz.setAccessibleName("Sweep to")
        self.freq.valueChanged.connect(self._on_freq)
        for box in (self.start_hz, self.end_hz, self.sweep_time):
            box.valueChanged.connect(self._on_sweep_range)
        self.sweep_buttons: dict[Sweep, QtWidgets.QPushButton] = {}
        row = FlowLayout()
        for mode, text in (
            (Sweep.UP, "Sweep up"),
            (Sweep.DOWN, "Sweep down"),
            (Sweep.UP_DOWN, "Up, then down"),
            (Sweep.DOWN_UP, "Down, then up"),
            (Sweep.HOLD, "Hold"),
        ):
            b = self.sweep_buttons[mode] = QtWidgets.QPushButton(text)
            b.clicked.connect(weak.slot(self, lambda self, _=False, m=mode: self.start_sweep(m)))
            row.add(b)
        dv.addRow(row)
        self.drive_state = QtWidgets.QLabel()
        mute(self.drive_state)
        dv.addRow(self.drive_state)

        branch = QtWidgets.QGroupBox("Start on a branch")
        branch.setToolTip(TIPS["branch"])
        bv = QtWidgets.QVBoxLayout(branch)
        row = FlowLayout()
        self.branch_buttons = []
        for i, text in enumerate(("Lower", "Middle", "Upper")):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(TIPS["branch"])
            b.clicked.connect(weak.slot(self, lambda self, _=False, k=i: self.start_on_branch(k)))
            self.branch_buttons.append(b)
            row.add(b)
        bv.addLayout(row)
        self.branch_note = QtWidgets.QLabel()
        self.branch_note.setWordWrap(True)
        mute(self.branch_note)
        bv.addWidget(self.branch_note)

        play = QtWidgets.QGroupBox("Simulation")
        pv = QtWidgets.QVBoxLayout(play)
        self.sim_speed = QtWidgets.QComboBox()
        self.sim_speed.setAccessibleName("Playback speed")
        for s in SPEEDS:
            self.sim_speed.addItem(f"{s:g}× real time", s)
        self.sim_speed.setCurrentIndex(SPEEDS.index(5.0))
        self.sim_speed.setToolTip("<p>Simulated seconds per second. A sweep of 120 s a leg takes 48 s at 5×.</p>")
        self.run_button = RunButton()
        self.run_button.toggled.connect(self._on_run)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setToolTip(
            "<p>Back to rest at t = 0, the measured points cleared; a sweep starts again from its start.</p>"
        )
        self.reset_button.clicked.connect(self.reset)
        self.clear_button = QtWidgets.QPushButton("Clear points")
        self.clear_button.setToolTip("<p>Forget the measured points (a new sweep does too).</p>")
        self.clear_button.clicked.connect(self.clear_points)
        row = FlowLayout()
        row.add(self.sim_speed, self.run_button, self.reset_button, self.clear_button)
        pv.addLayout(row)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        presets = QtWidgets.QGroupBox("Things to try")
        preset_layout = QtWidgets.QVBoxLayout(presets)
        preset_layout.addWidget(self.preset)
        preset_layout.addWidget(self.preset_note)
        lv.addWidget(presets)
        for pane in (osc, drive, branch, play):
            lv.addWidget(pane)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)

        # --- centre: the response against frequency, the readout, the strip chart
        self.response = _plot("Amplitude (first harmonic)", "m", "Drive frequency", "Hz")
        self.response.setMouseEnabled(x=False, y=False)
        self.response.setToolTip(RESPONSE_TIP)
        self.legend = add_legend(self.response.getPlotItem(), offset=(10, 10), colCount=2)
        self.linear_item = pg.PlotDataItem(name="Linear spring (k₃ = 0)")
        self.backbone_item = pg.PlotDataItem(name="Backbone")
        self.stable_item = pg.PlotDataItem(name="Harmonic balance", connect="finite")
        self.unstable_item = pg.PlotDataItem(name="… unstable", connect="finite")
        self.jump_item = pg.PlotDataItem(connect="pairs")
        self.fold_item = pg.ScatterPlotItem(size=7)
        self.up_item = pg.ScatterPlotItem(size=4, name="Sweeping up")
        self.down_item = pg.ScatterPlotItem(size=4, name="Sweeping down")
        self.hold_item = pg.ScatterPlotItem(size=4, name="Held")
        self.p_stable_item = pg.PlotDataItem(name="Pumped, ½X", connect="finite")
        self.p_unstable_item = pg.PlotDataItem(name="… unstable", connect="finite")
        self.half_items = {
            leg: pg.ScatterPlotItem(size=5, symbol="t", name=f"½X, {text}")
            for leg, text in (("up", "up"), ("down", "down"), ("hold", "held"))
        }
        self.now_item = pg.ScatterPlotItem(size=11)
        self.first_items = {"up": self.up_item, "down": self.down_item, "hold": self.hold_item}
        self.points = (*self.first_items.values(), *self.half_items.values())
        for item in (
            self.linear_item,
            self.backbone_item,
            self.p_unstable_item,
            self.p_stable_item,
            self.unstable_item,
            self.stable_item,
            self.jump_item,
            self.fold_item,
            *self.points,
            self.now_item,
        ):
            self.response.addItem(item)
        plot_height(self.response, 220)
        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)

        self.strip = _plot("Displacement x", "m", "Time", "s")
        self.strip.setMouseEnabled(x=False, y=False)
        self.strip.setToolTip("<p>The mass's displacement over the last 20 s of simulated time.</p>")
        self.x_item = pg.PlotDataItem()
        self.strip.addItem(self.x_item)
        plot_height(self.strip, 140)

        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.response, 3)
        cv.addWidget(self.readout)
        cv.addWidget(self.strip, 2)

        # --- right: the phase plane and the notes
        self.phase = _plot("Velocity", "m/s", "Displacement", "m")
        self.phase.setToolTip(PHASE_TIP)
        self.orbit_item = pg.PlotDataItem()
        self.section_item = pg.ScatterPlotItem(size=3)
        self.phase.addItem(self.orbit_item)
        self.phase.addItem(self.section_item)
        self.scrub = Scrub(self.strip)  # paused: back along the strip chart, the state on the phase plane
        self.scrub.scrubbed.connect(self._on_scrubbed)
        self.scrub_dot = Dot(11)
        self.scrub_dot.hide()
        self.phase.addItem(self.scrub_dot, ignoreBounds=True)
        self.phase_note = QtWidgets.QLabel(
            "The last few drive cycles; the dots are the Poincaré section, the state once per cycle (while the "
            "frequency is held)."
        )
        self.phase_note.setWordWrap(True)
        mute(self.phase_note)
        phase_tab = QtWidgets.QWidget()
        ph = QtWidgets.QVBoxLayout(phase_tab)
        ph.addWidget(self.phase, 1)
        ph.addWidget(self.phase_note)
        self.chart = _plot("Pump depth μ", None, "Pump frequency / natural frequency", None)
        self.chart.setMouseEnabled(x=False, y=False)
        self.chart.setToolTip(CHART_TIP)
        self.chart_image = pg.ImageItem()
        self.chart_zero = pg.IsocurveItem(level=0.0)
        self.chart_zero.setParentItem(self.chart_image)
        self.chart_hb = pg.PlotDataItem(connect="finite")
        self.chart_now = pg.ScatterPlotItem(size=11)
        for item in (self.chart_image, self.chart_hb, self.chart_now):
            self.chart.addItem(item)
        self.chart.setXRange(CHART_RATIOS[0], CHART_RATIOS[-1], padding=0)
        self.chart.setYRange(CHART_DEPTHS[0], CHART_DEPTHS[-1], padding=0)
        self.chart.scene().sigMouseClicked.connect(self._on_chart_click)
        self.chart_note = QtWidgets.QLabel()
        self.chart_note.setWordWrap(True)
        self.chart_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        mute(self.chart_note)
        chart_tab = QtWidgets.QWidget()
        chv = QtWidgets.QVBoxLayout(chart_tab)
        chv.addWidget(self.chart, 1)
        chv.addWidget(self.chart_note)
        self._chart_zeta: float | None = None  # the damping ratio the chart shows
        self.notes = NotesBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(phase_tab, "Phase plane")
        self.chart_tab = chart_tab
        self.tabs.addTab(chart_tab, "Ince–Strutt chart")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._draw_chart)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(center)
        splitter.addWidget(self.tabs)
        splitter.setSizes([330, 820, 550])
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # --- loop
        self._cost = 1e-6  # s of computing per simulated s (see step)
        self._target_t = 0.0
        self.loop = FrameLoop(self, "Duffing & Mathieu", self._tick, self._runs, self._sim_t, self._catch_up)

        self.preset.currentIndexChanged.connect(self._on_preset)
        self.apply_theme()
        self.load_preset(PRESETS[0].name)

    # ------------------------------------------------------------- inputs
    def oscillator(self) -> Duffing:
        """The oscillator the boxes describe (ValueError if they don't describe one)."""
        return Duffing(
            self.mass.value(),
            self.stiffness.value(),
            self.damping.value(),
            self.cubic.value(),
            self.force.value(),
            self.pump.value(),
            self.noise.value(),
        )

    def load_preset(self, name: str) -> None:
        p = next(p for p in PRESETS if p.name == name)
        o = p.osc
        setup = DuffingSetup(
            o.mass, o.stiffness, o.damping, o.cubic, o.force, o.pump, o.noise,
            p.freq_hz, p.start_hz, p.end_hz, p.sweep_time, p.mode, self.tabs.currentIndex(),
        )  # fmt: skip
        self._apply_setup(setup, name, p.note)

    def _pick_preset(self, name: str) -> None:
        """A preset the reader picked: set up and started, as a trial is."""
        self.load_preset(name)
        self._run(True)

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a `DuffingSetup`'s fields), as Custom."""
        self._apply_setup(DuffingSetup(**setup), CUSTOM, "")
        self._run(True)

    def practice_state(self) -> DuffingSetup:
        """The page's oscillator, drive and sweep, as a setup: what a practice spec checks."""
        return DuffingSetup(
            self.mass.value(),
            self.stiffness.value(),
            self.damping.value(),
            self.cubic.value(),
            self.force.value(),
            self.pump.value(),
            self.noise.value(),
            self.freq.value(),
            self.start_hz.value(),
            self.end_hz.value(),
            self.sweep_time.value(),
            self._mode,
            self.tabs.currentIndex(),
        )

    def _apply_setup(self, t: DuffingSetup, name: str, note: str) -> None:
        self._loading = True
        for box, value in (
            (self.mass, t.mass),
            (self.stiffness, t.stiffness),
            (self.damping, t.damping),
            (self.cubic, t.cubic),
            (self.force, t.force),
            (self.pump, t.pump),
            (self.noise, t.noise),
            (self.freq, t.freq_hz),
            (self.start_hz, t.start_hz),
            (self.end_hz, t.end_hz),
            (self.sweep_time, t.sweep_time),
        ):
            box.setValue(value)
        self.preset.blockSignals(True)
        self.preset.setCurrentText(name)
        self.preset.blockSignals(False)
        self.preset_note.setText(markup(note))
        self._loading = False
        d = self.sim.drive
        d.freq_hz, d.start_hz, d.end_hz, d.sweep_time = t.freq_hz, t.start_hz, t.end_hz, t.sweep_time
        self.tabs.blockSignals(True)  # the chart, if it's the tab, is drawn once by _apply_params
        self.tabs.setCurrentIndex(t.tab)
        self.tabs.blockSignals(False)
        self._apply_params()
        self._mode = t.mode
        self.reset()

    def frame_context(self) -> str:
        """For the frame log (frame_stats): the preset showing."""
        return self.preset.currentText()

    def _on_preset(self, index: int) -> None:
        if index < len(PRESETS):
            self._pick_preset(PRESETS[index].name)

    def _on_params(self) -> None:
        if self._loading:
            return
        self._to_custom()
        self._apply_params()

    def _to_custom(self) -> None:
        self.preset.blockSignals(True)
        self.preset.setCurrentText(CUSTOM)
        self.preset.blockSignals(False)
        self.preset_note.setText("")

    def _apply_params(self) -> None:
        try:
            osc = self.oscillator()
        except ValueError as e:
            self.osc_info.setText(f"<b>{e}</b>: the page keeps the last oscillator.")
            return
        self.sim.set_system(osc)
        self.clear_points()
        self._draw_curve()
        self._show_osc_info()
        self._update_branch_buttons()
        self._show_readout()
        self._draw_chart()

    def _on_freq(self, value: float) -> None:
        if self._loading:
            return
        self.sim.drive.freq_hz = value
        if self.sim.drive.legs:  # typing a frequency stops the sweep there
            self.sim.drive.start(Sweep.HOLD)
            self._mode = Sweep.HOLD
        self._update_branch_buttons()
        self._show_drive_state()
        self._draw_chart_point()

    def _on_sweep_range(self) -> None:
        if self._loading:
            return
        d = self.sim.drive
        d.start_hz, d.end_hz, d.sweep_time = self.start_hz.value(), self.end_hz.value(), self.sweep_time.value()
        if d.legs:  # the sweep it was running no longer exists: hold where it is
            d.start(Sweep.HOLD)
            self._mode = Sweep.HOLD
        self._to_custom()
        self.sim.set_system(self.sim.osc)  # the step resolves the new range
        self._draw_curve()
        self._show_drive_state()

    def start_sweep(self, mode: Sweep) -> None:
        """Start a sweep from its start (clearing the measured points), or hold the frequency where it is."""
        self._mode = mode
        self.sim.drive.start(mode)
        if mode is not Sweep.HOLD:
            self.clear_points()
        self._sync_freq()
        self._update_branch_buttons()
        self._show_drive_state()
        if not self.running and not self.sim.escaped:
            self._run(True)

    @property
    def pumped_only(self) -> bool:
        """Pumped with no drive force: the steady states are the pump's half-harmonic swings."""
        return view.pumped_only(self.sim.osc)

    def start_on_branch(self, k: int) -> None:
        """Put the mass on a steady state at the held frequency: 0 the lower, 1 the middle, 2 the upper."""
        a, _ = view.steady_states(self.sim.osc, self.sim.drive.freq_hz)
        if not a.size:
            return
        if self.sim.drive.legs:
            self.sim.drive.start(Sweep.HOLD)
            self._mode = Sweep.HOLD
        self.sim.start_on(float(a[0] if k == 0 else a[1] if k == 1 else a[-1]), half=self.pumped_only)
        self._target_t = self.sim.t
        self._show_drive_state()

    def _update_branch_buttons(self) -> None:
        b = view.branches(self.sim.osc, self.sim.drive.freq_hz)
        for button, on in zip(self.branch_buttons, b.enabled):
            button.setEnabled(on)
        self.branch_note.setText(b.note)

    # ------------------------------------------------------------- run
    def _on_run(self, on: bool) -> None:
        self.running = on
        self.loop.start()
        if on:
            self._hide_scrub()
        else:
            self._show_scrub()

    def _show_scrub(self) -> None:
        """Paused: a cursor at the strip chart's end, to drag back along it."""
        trace = self.record.trace
        if trace.n:
            t = trace.data[0, : trace.n]
            self.scrub.show(float(t[-1]), max(float(t[0]), float(t[-1]) - view.WINDOW))

    def _hide_scrub(self) -> None:
        self.scrub.hide()
        self.scrub_dot.hide()
        self._show_drive_state()

    def _on_scrubbed(self, t: float) -> None:
        """The state at `t` on the phase plane, and the time in the drive's line."""
        trace = self.record.trace
        ts, xs, vs = trace.data[:, : trace.n]
        self.scrub_dot.setData([np.interp(t, ts, xs)], [np.interp(t, ts, vs)])
        self.scrub_dot.show()
        self.drive_state.setText(f"Looking back at t = {t:.2f} s (paused at {self.sim.t:.2f} s).")

    def _run(self, on: bool) -> None:
        self.run_button.set_running(on)
        self._on_run(on)

    def reset(self) -> None:
        """Back to rest at t = 0, waiting for Start; a sweep starts again from its start."""
        self.sim.reset()
        self.sim.drive.start(self._mode)
        if self._mode is Sweep.HOLD:
            self.sim.drive.freq_hz = self.freq.value()
        self._target_t = 0.0
        self.record.clear()
        self.clear_points()
        self._hide_scrub()
        self.running = False
        self.run_button.rest()
        self._sync_freq()
        self._update_branch_buttons()
        self._show_drive_state()
        self._draw_live()

    def clear_points(self) -> None:
        self.record.clear_points()
        self._draw_points()
        self._show_readout()

    # ------------------------------------------------------------- the loop
    def _runs(self) -> bool:
        return self.running

    def _sim_t(self) -> float:
        return self.sim.t

    def _catch_up(self) -> None:
        """As the loop starts or resumes: play on from where the simulation is."""
        self._target_t = self.sim.t

    def _tick(self, dt: float) -> None:
        """A frame: `dt` s of wall-clock time at the playback speed, within the frame's budget."""
        self.step(dt * float(self.sim_speed.currentData()), FRAME_BUDGET)

    def step(self, duration: float, budget: float | None = None) -> None:
        """Advance by `duration` s of simulated time and redraw.

        With a `budget` (s of computing), advance no further than it allows at the cost seen so far,
        so a fast playback falls behind rather than stalling the window.
        """
        self._target_t = min(self._target_t + duration, self.sim.t + 0.1 + duration)
        ahead = self._target_t - self.sim.t
        if budget is not None:
            ahead = min(ahead, budget / self._cost)
        if ahead > 0.5 * self.sim.step_size():
            start = time.perf_counter()
            s = self.sim.advance(ahead)
            if s.t.size:
                self._cost = 0.8 * self._cost + 0.2 * (time.perf_counter() - start) / (s.t[-1] - s.t[0] + 1e-9)
            if self.record.add(s):
                self._draw_points()
                self._show_readout()
                self._update_branch_buttons()
        if self.sim.escaped:
            self._target_t = self.sim.t
            self._run(False)
            self._show_readout()
        self._sync_freq()
        self._show_drive_state()
        self._draw_live()

    def _sync_freq(self) -> None:
        f = round(self.sim.drive.freq_hz, 3)
        if f != self.freq.value():
            self.freq.blockSignals(True)
            self.freq.setValue(f)
            self.freq.blockSignals(False)

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.scrub.apply_theme()
        self.scrub_dot.setPen(pg.mkPen(c.strong, width=1.5))
        self.scrub_dot.setBrush(pg.mkBrush(c.force))
        self.linear_item.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.backbone_item.setPen(pg.mkPen(c.muted, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.stable_item.setPen(pg.mkPen(c.strong, width=2))
        self.unstable_item.setPen(pg.mkPen(c.strong, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.p_stable_item.setPen(pg.mkPen(c.mode[1], width=2))
        self.p_unstable_item.setPen(pg.mkPen(c.mode[1], width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.jump_item.setPen(pg.mkPen(c.force, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.fold_item.setPen(pg.mkPen(c.force, width=1.5))
        self.fold_item.setBrush(pg.mkBrush(c.background))
        for leg, color in (("up", c.mass[0]), ("down", c.mass[1]), ("hold", c.mass[2])):
            dots = {"up": self.up_item, "down": self.down_item, "hold": self.hold_item}[leg]
            dots.setPen(None)
            dots.setBrush(pg.mkBrush(color))
            self.half_items[leg].setPen(pg.mkPen(color, width=1))
            self.half_items[leg].setBrush(None)
        self.now_item.setPen(pg.mkPen(c.force, width=2))
        self.now_item.setBrush(None)
        self.x_item.setPen(pg.mkPen(c.mass[0], width=1))
        self.orbit_item.setPen(pg.mkPen(c.mass[0], width=1))
        self.section_item.setPen(None)
        self.section_item.setBrush(pg.mkBrush(c.force))
        self.chart_zero.setPen(pg.mkPen(c.force, width=1.5))
        self.chart_hb.setPen(pg.mkPen(c.strong, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.chart_now.setPen(pg.mkPen(c.strong, width=2))
        self.chart_now.setBrush(None)
        self.chart_image.setLookupTable(self._chart_lut())
        if self.shown is not None:
            self._draw_curve()
            self._draw_points()
            self._show_readout()

    def _chart_lut(self) -> np.ndarray:
        """Blank where the swing dies away, shading up to the force colour as it grows faster."""
        back = np.array(pg.mkColor(colors.background).getRgb()[:3], dtype=float)
        hot = np.array(pg.mkColor(colors.force).getRgb()[:3], dtype=float)
        n = 256
        zero = round(n * 0.05 / (GROWTH_TOP + 0.05))  # the levels run from -0.05 to GROWTH_TOP
        w = np.clip((np.arange(n) - zero) / (n - 1 - zero), 0.0, 1.0)[:, None]
        w = np.where(w > 0, 0.25 + 0.75 * w, 0.0)  # a step at zero, so the faintest growth shows
        return (back * (1 - w) + hot * w).astype(np.ubyte)

    def _draw_curve(self) -> None:
        osc = self.sim.osc
        self.shown = v = view.response(osc, view.x_range(self.sim.drive))
        self._y_top = v.y_top
        for item, curve in (
            (self.stable_item, v.stable),
            (self.unstable_item, v.unstable),
            (self.p_stable_item, v.p_stable),
            (self.p_unstable_item, v.p_unstable),
            (self.backbone_item, v.backbone),
            (self.jump_item, v.jumps),
        ):
            item.setData(*curve)
        self.linear_item.setData(*v.linear, connect="finite")
        self.fold_item.setData(*v.folds)
        self.response.setXRange(*v.x_range, padding=0)
        self.response.setYRange(0.0, v.y_top, padding=0)
        self.response.setLabel("left", v.y_label, units="m")
        if v.legend_right:  # in the corner the curve leans away from
            self.legend.anchor((1, 0), (1, 0), offset=(-10, 10))
        else:
            self.legend.anchor((0, 0), (0, 0), offset=(10, 10))
        items = {
            "linear": self.linear_item,
            "backbone": self.backbone_item,
            "backbone_2x": self.backbone_item,
            "stable": self.stable_item,
            "unstable": self.unstable_item,
            "p_stable": self.p_stable_item,
            "p_unstable": self.p_unstable_item,
            **self.first_items,
            **{f"half_{leg}": item for leg, item in self.half_items.items()},
        }
        self.legend.clear()
        for key in v.legend:
            self.legend.addItem(items[key], LEGEND[key])

    def _draw_points(self) -> None:
        v = view.points(self.sim.osc, self.record, self._y_top, self.sim.runaway)
        for leg in LEGS:
            self.first_items[leg].setData(pos=v.first[leg])
            self.half_items[leg].setData(pos=v.half[leg])
        self.now_item.setData(pos=v.now)
        if v.y_top is not None:
            self._y_top = v.y_top
            self.response.setYRange(0.0, self._y_top, padding=0)
        self.section_item.setData(pos=v.section)

    def _draw_live(self) -> None:
        v = view.live(self.record.trace, self.sim.t, self.sim.drive.freq_hz)
        self.x_item.setData(*v.strip)
        if v.strip_range is not None:
            self.strip.setXRange(*v.strip_range, padding=0)
        self.orbit_item.setData(*v.orbit)

    def _draw_chart(self) -> None:
        """The Ince–Strutt chart at the page's damping, when it shows (it takes ~0.1 s)."""
        osc = self.sim.osc
        if self.tabs.currentWidget() is not self.chart_tab or osc.stiffness <= 0:
            if osc.stiffness <= 0:
                self.chart_note.setText(view.NO_SPRING)
            return
        if self._chart_zeta != osc.zeta:
            self._chart_zeta = osc.zeta
            c = view.chart(osc.zeta)
            self.chart_image.setImage(c.growth, levels=(-0.05, GROWTH_TOP), autoLevels=False)
            self.chart_image.setRect(QtCore.QRectF(*c.rect))
            self.chart_zero.setData(c.growth)
            self.chart_hb.setData(*c.tongue)
        self._draw_chart_point()

    def _draw_chart_point(self) -> None:
        v = view.chart_point(self.sim.osc, self.sim.drive.freq_hz)
        self.chart_now.setData(pos=v.now)
        self.chart_note.setText(v.note)

    def _on_chart_click(self, event) -> None:
        vb = self.chart.getPlotItem().getViewBox()
        pos = vb.mapSceneToView(event.scenePos())
        r, mu = pos.x(), pos.y()
        osc = self.sim.osc
        if osc.stiffness <= 0 or not (CHART_RATIOS[0] <= r <= CHART_RATIOS[-1] and 0 <= mu <= CHART_DEPTHS[-1]):
            return
        self.pump.setValue(round(mu, 3))  # an edit: Custom, the curves redrawn
        self.freq.setValue(round(r * osc.natural_hz, 3))  # holds there

    def _show_osc_info(self) -> None:
        self.osc_info.setText(view.osc_info(self.sim.osc, self.shown.curve if self.shown else None))

    def _show_drive_state(self) -> None:
        self.drive_state.setText(view.drive_state(self.sim.drive, self.sim.t))

    def _show_readout(self) -> None:
        if self.shown is None:
            return
        v = self.shown
        self.readout.setText(
            view.readout(self.sim.osc, v.curve, v.p_curve, self.record.cycles, self.sim.escaped, self.sim.runaway)
        )
