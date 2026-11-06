"""Nonlinearity & forces page: is the chain linear (stepped sine at several levels, up and down, and the
harmonics it makes), and which force made a measured response (a regularized inverse).

Left: the nonlinear element added to the chain (a cubic spring or a gap, alongside one of its springs; the
chain's own friction counts too), the stepped sine (force and response masses, three force levels, the mode it
sweeps round), and the force identification (the true force, the sensors and what they measure, the noise, and
how the inverse is regularized). Centre: the FRF |X/F| at each level, swept up and down, and the identified
force against the true one. Right: the steady response's waveform and harmonics at one point, the L-curve, the
singular values with the data's coefficients (Picard), and the Theory. The page follows the Simulation page's
chain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import force_id as fid
from ..core import nonlinear_test as nl
from ..core.model import ChainSystem
from ..core.practice import Problem, Question, Requirement, at_most, kept, number, pick
from . import settle_plots
from .layouts import FlowLayout
from .lesson import (
    Form,
    TrialBox,
    column,
    combo,
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

TAB_DISTORTION, TAB_LCURVE, TAB_PICARD, TAB_THEORY = range(4)
BAND = (0.6, 1.6)  # the stepped sine's band, as multiples of the mode's frequency
STEPS = 240
MASSES = (1.0, 1.0, 1.0)
SPRINGS = (1000.0, 1000.0, 1000.0)
DAMPERS = (4.0, 4.0, 4.0)
NONE = (0.0, 0.0, 0.0)


def chain(masses=MASSES, stiffness=SPRINGS, damping=DAMPERS, friction=NONE) -> ChainSystem:
    return ChainSystem(
        np.array(masses, dtype=float),
        np.array(stiffness, dtype=float),
        np.array(damping, dtype=float),
        np.array(friction, dtype=float),
    )


@dataclass(frozen=True)
class NonlinearTrial:
    name: str
    note: str
    kind: nl.Kind = nl.Kind.NONE
    spring: int = 0  # from 0
    k3: float = 2e6  # N/m³
    gap: float = 0.005  # m
    k_gap: float = 5000.0  # N/m
    force: int = 2  # the stepped sine's force, on this mass (from 0)
    out: int = 2  # and its response
    levels: tuple[float, float, float] = (0.05, 0.2, 0.5)  # N
    mode: int = 0  # the band is round this mode
    level: int = 2  # the Distortion tab's level (index)
    hz: float = 2.3  # and frequency
    shape: fid.Shape = fid.Shape.IMPACT
    hit: int = 2  # the identified force is on this mass
    width: float = 0.1  # s
    output: fid.Output = fid.Output.DISPLACEMENT
    sensors: tuple[int, ...] = (0,)
    noise: float = 2.0  # %
    method: fid.Method = fid.Method.TIKHONOV
    log_lam: float = -4.3
    k: int = 40
    tab: int = TAB_DISTORTION
    masses: tuple[float, ...] = MASSES
    stiffness: tuple[float, ...] = SPRINGS
    damping: tuple[float, ...] = DAMPERS
    friction: tuple[float, ...] = NONE

    def system(self) -> ChainSystem:
        return chain(self.masses, self.stiffness, self.damping, self.friction)

    @property
    def element(self) -> nl.Element:
        return nl.Element(self.kind, self.spring, self.k3, self.gap, self.k_gap)


TRIALS = (
    NonlinearTrial(
        "Linear: every level agrees",
        "Three 1 kg masses on 1000 N/m springs with 4 N·s/m dampers, driven on mass 3 by 0.05, 0.2 and 0.5 N, "
        "stepped up and down through mode 1 (2.240 Hz). The six FRFs |X₃/F| lie on one curve, and on the "
        "linear FRF: double the force and the response doubles. That's homogeneity, and the first check of "
        "any test.",
    ),
    NonlinearTrial(
        "A hardening spring",
        "A cubic spring, 2 × 10⁶ N/m³, alongside the first spring. At 0.05 N it hardly shows; at 0.5 N mode 1's "
        "peak has leaned up to 2.319 Hz; at 1 N, swept up, the response rides the leaning branch to its peak at "
        "2.431 Hz and drops off it at 2.450 Hz, while swept down it only jumps up onto it at 2.422 Hz: between "
        "the two the response depends on the way it came. The levels' FRFs differ by 67 % of the peak.",
        kind=nl.Kind.CUBIC,
        levels=(0.05, 0.5, 1.0),
    ),
    NonlinearTrial(
        "A gap: linear, until it closes",
        "A 5000 N/m spring that only touches once the first spring stretches 5 mm either way. At 0.05 and "
        "0.2 N the stretch stays inside the gap and the FRFs agree; at 0.5 N it closes near the peak, which "
        "leans and jumps (swept up at 2.553 Hz, down at 2.356 Hz). A test at the two low levels alone would call "
        "this chain linear.",
        kind=nl.Kind.GAP,
    ),
    NonlinearTrial(
        "Friction: damping that fades",
        "No extra element, but 0.1 N of Coulomb friction on mass 3 (the Simulation page's). At 0.2 N the "
        "friction takes most of the force: the peak FRF is 0.0177 m/N. At 2 N it's a twentieth of the force, "
        "and the peak reaches 0.0456 m/N, near the linear 0.0487. Friction's FRF rises with the level: it "
        "damps a small motion more, as a share.",
        levels=(0.2, 0.5, 2.0),
        friction=(0.0, 0.0, 0.1),
    ),
    NonlinearTrial(
        "Distortion: odd harmonics",
        "The hardening spring at 0.5 N, held at 2.269 Hz just below its leaning peak: the steady response "
        "isn't a sine. Its 3X is 1.9 % of its 1X at mass 3 and its 5X smaller still, and there's no 2X: a "
        "symmetric nonlinearity makes odd harmonics only. Harmonic balance's one harmonic gives 1X within "
        "3 % of the time response's.",
        kind=nl.Kind.CUBIC,
        hz=2.269,
    ),
    NonlinearTrial(
        "Force identification: the plain inverse",
        "Now the other way: a 0.1 s impact on mass 3, measured as mass 1's displacement with 2 % noise. G, "
        "the matrix from the force's samples to the measurement's, is lower triangular with singular values "
        "spanning 20 decades (the chain filters out what's fast), and G⁻¹y divides the noise by the smallest: "
        "the force comes out 73,000 times too large, all noise.",
        method=fid.Method.INVERSE,
        tab=TAB_PICARD,
    ),
    NonlinearTrial(
        "Tikhonov at the L-curve's corner",
        "The same measurement, solved with Tikhonov's λ = 5.4 × 10⁻⁵ from the L-curve's corner: each "
        "direction weighted by s²/(s² + λ²), so those far below λ drop out. The impact comes back within 15 % "
        "(the error's norm against the force's), its peak a little low and spread: the sharpest part of it is "
        "what the chain filtered out.",
        log_lam=math.log10(5.44e-5),
        tab=TAB_LCURVE,
    ),
    NonlinearTrial(
        "Measure the acceleration where the force is",
        "An accelerometer on mass 3, where the impact is: it feels the force at once (a = F/m, before any "
        "spring moves), so G's singular values span only 3 decades, not 20. Tikhonov with λ = 0.024 recovers "
        "the impact within 6 %. Where the sensor is, and what it measures, matters more than the method.",
        output=fid.Output.ACCELERATION,
        sensors=(2,),
        log_lam=math.log10(0.0244),
        tab=TAB_PICARD,
    ),
)


# ------------------------------------------------------------------ practice problems
def _cubic_stiffness(rng: np.random.Generator) -> Question:
    k3 = pick(rng, (1e6, 2e6, 5e6))
    amp = pick(rng, (2.0, 5.0, 10.0))
    k = 0.75 * k3 * (amp / 1000) ** 2
    return Question(
        f"A cubic spring k₃δ³ with k₃ = {k3:.0e} N/m³ swings at {amp:g} mm amplitude. What stiffness does it add "
        "to the first harmonic, in N/m?",
        k,
        f"The first harmonic of k₃(A cos θ)³ is ¾k₃A³ cos θ: a stiffness ¾k₃A² = ¾ × {k3:.0e} × ({amp:g} × 10⁻³)² "
        f"= {number(k)} N/m. It grows with the amplitude's square: the peak leans further at each level.",
        unit="N/m",
    )


def _gap_level(rng: np.random.Generator) -> Question:
    return Question(
        "A chain with a gap (a spring that only touches once a stretch passes a clearance) is tested at a level "
        "so low that the stretch never reaches it. What do its FRFs at two such levels show?",
        0,
        "The same FRF, the linear one: the gap never closes, so nothing nonlinear happens. Homogeneity only "
        "tests the levels it's run at: test at the levels the structure works at.",
        choices=(
            "The same, linear FRF",
            "A peak leaning to higher frequency",
            "Jumps between sweeping up and down",
            "Odd harmonics in the response",
        ),
        setup=dict(kind=nl.Kind.GAP, levels=(0.02, 0.05, 0.1)),
        predict=True,
    )


def _friction_level(rng: np.random.Generator) -> Question:
    return Question(
        "A chain with Coulomb friction is tested at a low and then a high force. How does its peak FRF |X/F| change?",
        0,
        "It rises: friction is a fixed force, so at a higher level it's a smaller share of the motion's forces "
        "and damps less (its describing function is an equivalent damping 4F<sub>c</sub>/(πωX), falling as X "
        "grows).",
        choices=("It rises", "It falls", "It doesn't change", "It leans to a higher frequency"),
        setup=dict(levels=(0.2, 0.5, 2.0), friction=(0.0, 0.0, 0.1)),
        predict=True,
    )


def _filter_factor(rng: np.random.Generator) -> Question:
    s = pick(rng, (1e-3, 1e-4, 1e-5))
    ratio = pick(rng, (0.3, 1.0, 3.0, 10.0))
    lam = s * ratio
    factor = s * s / (s * s + lam * lam)
    return Question(
        f"Tikhonov regularization with λ = {lam:.1e}: by what factor does it keep the direction whose singular "
        f"value is s = {s:.0e}?",
        factor,
        f"s²/(s² + λ²) = 1/(1 + (λ/s)²) = 1/(1 + {ratio:g}²) = {number(factor)}. Directions with s ≫ λ are kept "
        "whole, those with s ≪ λ dropped, with a smooth step between.",
    )


def _identify_spec(rng: np.random.Generator) -> Question:
    target = pick(rng, (8.0, 10.0))

    def goal(t: NonlinearTrial):
        system = t.system()
        h = fid.step_for(system)
        truth = fid.force_history(t.shape, fid.SAMPLES, h, t.width)
        sensors = tuple(s for s in t.sensors if s < system.n)
        if not sensors:
            err = math.inf
        else:
            G = fid.markov(system, t.hit, sensors, t.output, h, fid.SAMPLES)
            p = fid.set_up(G, truth, t.noise / 100)
            err = 100 * fid.error(_solve(p, t.method, 10**t.log_lam, t.k), truth)
        return (
            kept("Noise", t.noise, 2.0, "%"),
            Requirement("The impact on mass 3 kept", "kept" if t.hit == 2 else "moved", t.hit == 2),
            at_most("Sensors", float(len(sensors)), 1.0),
            at_most("The identified force's error", err, target, "%"),
        )

    return Question(
        f"Identify the impact on mass 3 within {target:g} % (the error's norm against the force's), with one "
        "sensor and the noise left at 2 %.",
        0,
        "Measure the acceleration on mass 3 itself (it feels the force at once, so G is well conditioned) and "
        "regularize with Tikhonov near λ = 0.024: about 6 %. A displacement sensor on mass 1 gets about 15 % at "
        "best, whatever λ.",
        setup=dict(method=fid.Method.INVERSE, tab=TAB_LCURVE),
        goal=goal,
        design=dict(
            output=fid.Output.ACCELERATION, sensors=(2,), method=fid.Method.TIKHONOV, log_lam=math.log10(0.0244)
        ),
    )


PROBLEMS = tuple(
    Problem(key, title, "nonlinear_page", f"See {page_link('nonlinear_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("cubic-stiffness", "A cubic spring's stiffness", "describing", "Harmonic balance", _cubic_stiffness),
        ("gap-level", "Predict: a gap at a low level", "homogeneity", "Homogeneity", _gap_level),
        ("friction-level", "Predict: friction at two levels", "describing", "Harmonic balance", _friction_level),
        ("filter-factor", "Tikhonov's filter factor", "regularization", "Regularization", _filter_factor),
        ("identify-spec", "Spec: identify an impact", "regularization", "Regularization", _identify_spec),
    )
)


def _solve(p: fid.Problem, method: fid.Method, lam: float, k: int) -> np.ndarray:
    if method is fid.Method.INVERSE:
        return fid.inverse(p)
    if method is fid.Method.TSVD:
        return fid.tsvd(p, k)
    return fid.tikhonov(p, lam)


TIPS = {
    "dof": "<p>The number of masses: the same chain as the Simulation page. Its friction (set there) is "
    "nonlinear too.</p>",
    "kind": "<p>The nonlinear element the page adds alongside one of the chain's springs: a cubic spring "
    "(hardening, k₃δ³), or a gap (a spring that only touches once the stretch passes a clearance either "
    "way).</p>",
    "spring": "<p>The spring it sits alongside: spring 1 joins mass 1 to the ground, spring i masses i − 1 and i.</p>",
    "k3": "<p>The cubic spring's k₃: its force is k₃δ³.</p>",
    "gap": "<p>The clearance: the gap's spring touches once the stretch passes this, either way.</p>",
    "k_gap": "<p>The gap's spring, once it touches.</p>",
    "force": "<p>The mass the stepped sine drives.</p>",
    "out": "<p>The mass whose response the FRF is.</p>",
    "levels": "<p>Three force amplitudes: each is swept up and then down through the band.</p>",
    "mode": "<p>The band is 0.6 to 1.6 times this mode's frequency.</p>",
    "shape": "<p>The force to identify: an impact (a half-sine), two of them, or a smooth random burst.</p>",
    "hit": "<p>The mass the force acts on.</p>",
    "width": "<p>How long each impact lasts (a burst's smoothing).</p>",
    "output": "<p>What the sensors measure. An accelerometer on the force's own mass feels it at once.</p>",
    "sensors": "<p>The masses with a sensor.</p>",
    "noise": "<p>The sensors' noise, as a share of each one's RMS.</p>",
    "method": "<p>How G f = y is solved: plainly (useless), keeping the first k singular directions, or "
    "Tikhonov's weighting s²/(s² + λ²).</p>",
    "lam": "<p>Tikhonov's λ, as log₁₀ λ.</p>",
    "k": "<p>How many singular directions truncated SVD keeps.</p>",
    "level": "<p>The level and frequency the Distortion tab steps the chain at in time.</p>",
}
FRF_TIP = (
    "<p>The FRF |X/F| at each level: swept up (solid) and down (dashed). Dotted: the linear chain's. A linear "
    "chain's levels all lie on it; the dots mark where a sweep jumped.</p>"
)
FORCE_TIP = "<p>The true force (solid) and the one identified from the noisy measurement (dashed).</p>"

NOTES_HTML = load_notes("nonlinear_forces", trials=trials_html(TRIALS))


class NonlinearForcesPage(QtWidgets.QWidget):
    """Controls on the left, the FRFs and the identified force in the centre, the details on the right."""

    dof_requested = QtCore.Signal(int)  # the user changed N here; the main window owns the system
    system_requested = QtCore.Signal(object)  # a trial's chain, for the Simulation page
    edit_parameters = QtCore.Signal()  # the user wants the Simulation page's parameter panel

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system = chain()
        self._dirty = True
        self.sweeps: list[tuple[nl.Sweep, nl.Sweep]] = []
        self.freqs = np.zeros(0)
        self.problem: fid.Problem | None = None
        self.truth = np.zeros(0)
        self.estimate = np.zeros(0)
        self.curve: fid.LCurve | None = None
        self.h = 0.0
        self.waveform: nl.Waveform | None = None

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Each loads a chain into the Simulation page and sets up the test to show one idea from the Theory "
            "tab.</p>",
        )
        self.trials.picked.connect(self.load_trial)

        ch = Form("Chain")
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setToolTip(TIPS["dof"])
        self.dof.valueChanged.connect(self.dof_requested)
        ch.add("Number of masses:", self.dof, TIPS["dof"])
        link = QtWidgets.QLabel("Masses, springs and friction are edited on the <a href='#sim'>Simulation page</a>.")
        link.setWordWrap(True)
        mute(link)
        link.linkActivated.connect(lambda _: self.edit_parameters.emit())
        ch.row(link)
        self.chain_info = note_label()
        ch.row(self.chain_info)

        element = Form("Nonlinear element")
        self.kind = combo([k.value for k in nl.KINDS], TIPS["kind"])
        element.add("Kind:", self.kind, TIPS["kind"])
        self.spring = element.number("Alongside spring:", 1, MAX_DOF, 1, 0, "", TIPS["spring"])
        self.k3 = element.number("k₃:", -1e12, 1e12, 2e6, 0, " N/m³", TIPS["k3"])
        self.gap = element.number("Gap:", 0.0, 1000.0, 5.0, 2, " mm", TIPS["gap"])
        self.k_gap = element.number("Gap's spring:", 0.0, 1e9, 5000.0, 0, " N/m", TIPS["k_gap"])
        self.element_form = element

        sine = Form("Stepped sine")
        self.force = sine.number("Force on mass:", 1, MAX_DOF, 3, 0, "", TIPS["force"])
        self.out = sine.number("Response of mass:", 1, MAX_DOF, 3, 0, "", TIPS["out"])
        self.levels = [
            sine.number(f"Level {k + 1}:", 1e-6, 1e6, v, 3, " N", TIPS["levels"])
            for k, v in enumerate((0.05, 0.2, 0.5))
        ]
        self.mode = sine.number("Around mode:", 1, MAX_DOF, 1, 0, "", TIPS["mode"])

        ident = Form("Force identification")
        self.shape = combo([s.value for s in fid.SHAPES], TIPS["shape"])
        ident.add("Force:", self.shape, TIPS["shape"])
        self.hit = ident.number("On mass:", 1, MAX_DOF, 3, 0, "", TIPS["hit"])
        self.width = ident.number("Lasting:", 1.0, 2000.0, 100.0, 0, " ms", TIPS["width"])
        self.output = combo([o.value for o in fid.OUTPUTS], TIPS["output"])
        ident.add("Measured:", self.output, TIPS["output"])
        holder = QtWidgets.QWidget()
        flow = FlowLayout(holder)
        flow.setContentsMargins(0, 0, 0, 0)
        self.sensor_boxes: list[QtWidgets.QCheckBox] = []
        for i in range(MAX_DOF):
            box = QtWidgets.QCheckBox(f"m{i + 1}")
            box.setToolTip(TIPS["sensors"])
            self.sensor_boxes.append(box)
            flow.add(box)
        ident.add("Sensors on:", holder, TIPS["sensors"])
        self.noise = ident.number("Noise:", 0.0, 100.0, 2.0, 2, " %", TIPS["noise"])
        self.method = combo([m.value for m in fid.METHODS], TIPS["method"])
        ident.add("Regularized by:", self.method, TIPS["method"])
        self.log_lam = ident.number("log₁₀ λ:", -16.0, 6.0, -4.3, 2, "", TIPS["lam"])
        self.log_lam.setSingleStep(0.1)
        self.k = ident.number("Keep k:", 1, fid.SAMPLES, 40, 0, "", TIPS["k"])
        buttons = QtWidgets.QHBoxLayout()
        self.corner_button = QtWidgets.QPushButton("λ at the corner")
        self.corner_button.setToolTip("<p>Set λ to the L-curve's corner.</p>")
        self.discrepancy_button = QtWidgets.QPushButton("λ from the noise")
        self.discrepancy_button.setToolTip("<p>Set λ by the discrepancy principle: its residual the noise's size.</p>")
        buttons.addWidget(self.corner_button)
        buttons.addWidget(self.discrepancy_button)
        ident.row(buttons)
        self.ident_form = ident
        self.corner_button.clicked.connect(lambda: self._set_lambda(self.curve.corner if self.curve else None))
        self.discrepancy_button.clicked.connect(
            lambda: self._set_lambda(fid.discrepancy(self.problem) if self.problem else None)
        )

        connect_all(
            (self.kind, self.spring, self.k3, self.gap, self.k_gap, self.force, self.out, *self.levels, self.mode),
            self._on_sine,
        )
        connect_all((self.shape, self.hit, self.width, self.output, *self.sensor_boxes, self.noise), self._on_measure)
        connect_all((self.method, self.log_lam, self.k), self._on_solve)

        # --- centre
        self.frf_plot = plot("|X/F|", "m/N", "Frequency", "Hz", log_y=True, height=180)
        self.frf_plot.setToolTip(FRF_TIP)
        self.frf_plot.setMouseEnabled(x=False, y=False)
        self.frf_legend = add_legend(self.frf_plot.getPlotItem(), offset=(-10, 10))
        self.linear_item = pg.PlotDataItem()
        self.frf_plot.addItem(self.linear_item)
        self.up_items = [pg.PlotDataItem() for _ in range(3)]
        self.down_items = [pg.PlotDataItem() for _ in range(3)]
        self.jump_items = [marker(9) for _ in range(3)]
        for item in (*self.up_items, *self.down_items, *self.jump_items):
            self.frf_plot.addItem(item)
        self.dist_line = pg.InfiniteLine(angle=90)
        self.frf_plot.addItem(self.dist_line, ignoreBounds=True)
        self.frf_readout = readout_label()
        self.force_plot = plot("Force", "N", "Time", "s", height=160)
        self.force_plot.setToolTip(FORCE_TIP)
        self.force_plot.setMouseEnabled(x=False, y=False)
        self.force_legend = add_legend(self.force_plot.getPlotItem(), offset=(-10, 10))
        self.truth_item = pg.PlotDataItem()
        self.estimate_item = pg.PlotDataItem()
        self.force_plot.addItem(self.truth_item)
        self.force_plot.addItem(self.estimate_item)
        self.force_legend.addItem(self.truth_item, "true")
        self.force_legend.addItem(self.estimate_item, "identified")
        self.force_readout = readout_label()

        # --- right
        dist = Form("")
        dist.setFlat(True)
        self.dist_level = combo(["Level 1", "Level 2", "Level 3"], TIPS["level"])
        dist.add("At level:", self.dist_level, TIPS["level"])
        self.dist_hz = dist.number("Frequency:", 0.001, 1e4, 2.3, 3, " Hz", TIPS["level"])
        connect_all((self.dist_level, self.dist_hz), self._on_distortion)
        self.wave_plot = plot("Response (1X's amplitude 1)", None, "Drive cycles", None, height=170)
        self.wave_plot.setMouseEnabled(x=False, y=False)
        self.wave_item = pg.PlotDataItem()
        self.sine_item = pg.PlotDataItem()
        self.wave_plot.addItem(self.sine_item)
        self.wave_plot.addItem(self.wave_item)
        self.harm_plot = plot("Of 1X", "%", "Harmonic", None, height=150)
        self.harm_plot.setMouseEnabled(x=False, y=False)
        self.harm_bars = pg.BarGraphItem(x=[], height=[], width=0.6)
        self.harm_plot.addItem(self.harm_bars)
        self.dist_note = note_label()

        self.l_plot = plot("‖f‖", "N", "Residual ‖Gf − y‖", None, log_x=True, log_y=True, height=220)
        self.l_plot.setMouseEnabled(x=False, y=False)
        self.l_item = pg.PlotDataItem()
        self.l_now = marker(12)
        self.l_corner = marker(10, "d")
        for item in (self.l_item, self.l_corner, self.l_now):
            self.l_plot.addItem(item)
        self.l_note = note_label()

        self.picard_plot = plot("", None, "Index i", None, log_y=True, height=240)
        self.picard_plot.setMouseEnabled(x=False, y=False)
        self.picard_legend = add_legend(self.picard_plot.getPlotItem(), offset=(-10, 10))
        self.s_item = pg.PlotDataItem()
        self.beta_item = pg.PlotDataItem()
        self.lam_line = pg.InfiniteLine(angle=0)
        self.picard_plot.addItem(self.s_item)
        self.picard_plot.addItem(self.beta_item)
        self.picard_plot.addItem(self.lam_line, ignoreBounds=True)
        self.picard_legend.addItem(self.s_item, "singular value s<sub>i</sub>")
        self.picard_legend.addItem(self.beta_item, "|u<sub>i</sub>ᵀy|, the data's")
        self.picard_note = note_label()

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(column(dist, self.wave_plot, self.harm_plot, self.dist_note, stretch=False), "Distortion")
        self.tabs.addTab(column(self.l_plot, self.l_note, stretch=False), "L-curve")
        self.tabs.addTab(column(self.picard_plot, self.picard_note, stretch=False), "Singular values")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(lambda _: self._draw_tab())

        three_panes(
            self,
            column(self.trials, ch, element, sine, ident),
            stack((self.frf_plot, 3), (self.frf_readout, 0), (self.force_plot, 3), (self.force_readout, 0)),
            self.tabs,
        )
        self.apply_theme()
        self.set_system(self.system)

    # ------------------------------------------------------------- inputs
    def set_system(self, system: ChainSystem, result=None) -> None:
        """The Simulation page's chain. Worked out now if the page shows, else when it does."""
        self.system = system
        set_quietly(self.dof, system.n)
        for box in (self.spring, self.force, self.out, self.mode, self.hit):
            box.blockSignals(True)
            box.setMaximum(system.n)
            box.blockSignals(False)
        for i, box in enumerate(self.sensor_boxes):
            box.setVisible(i < system.n)
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
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(NonlinearTrial("", "", **setup))

    def practice_state(self) -> NonlinearTrial:
        """The page's settings and the chain it follows, as a trial: what a practice spec checks."""
        s = self.system
        return NonlinearTrial(
            "",
            "",
            kind=nl.KINDS[self.kind.currentIndex()],
            spring=int(self.spring.value()) - 1,
            k3=self.k3.value(),
            gap=self.gap.value() / 1000,
            k_gap=self.k_gap.value(),
            force=int(self.force.value()) - 1,
            out=int(self.out.value()) - 1,
            levels=(self.levels[0].value(), self.levels[1].value(), self.levels[2].value()),
            mode=int(self.mode.value()) - 1,
            level=self.dist_level.currentIndex(),
            hz=self.dist_hz.value(),
            shape=fid.SHAPES[self.shape.currentIndex()],
            hit=int(self.hit.value()) - 1,
            width=self.width.value() / 1000,
            output=fid.OUTPUTS[self.output.currentIndex()],
            sensors=self._sensors(),
            noise=self.noise.value(),
            method=fid.METHODS[self.method.currentIndex()],
            log_lam=self.log_lam.value(),
            k=int(self.k.value()),
            tab=self.tabs.currentIndex(),
            masses=tuple(float(m) for m in s.masses),
            stiffness=tuple(float(k) for k in s.stiffness),
            damping=tuple(float(c) for c in s.damping),
            friction=tuple(float(f) for f in s.friction),
        )

    def apply_trial(self, t: NonlinearTrial) -> None:
        # the controls first, so the chain the Simulation page passes back is worked out once, with them
        system = t.system()
        for box in (self.spring, self.force, self.out, self.mode, self.hit):
            box.blockSignals(True)
            box.setMaximum(system.n)
            box.blockSignals(False)
        for box, value in (
            (self.kind, nl.KINDS.index(t.kind)),
            (self.spring, t.spring + 1),
            (self.k3, t.k3),
            (self.gap, 1000 * t.gap),
            (self.k_gap, t.k_gap),
            (self.force, t.force + 1),
            (self.out, t.out + 1),
            (self.levels[0], t.levels[0]),
            (self.levels[1], t.levels[1]),
            (self.levels[2], t.levels[2]),
            (self.mode, t.mode + 1),
            (self.dist_level, t.level),
            (self.dist_hz, t.hz),
            (self.shape, fid.SHAPES.index(t.shape)),
            (self.hit, t.hit + 1),
            (self.width, 1000 * t.width),
            (self.output, fid.OUTPUTS.index(t.output)),
            (self.noise, t.noise),
            (self.method, fid.METHODS.index(t.method)),
            (self.log_lam, t.log_lam),
            (self.k, t.k),
        ):
            set_quietly(box, value)
        for i, box in enumerate(self.sensor_boxes):
            set_quietly(box, i in t.sensors)
        self.trials.show_note(t.note)
        self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex(t.tab)
        self.tabs.blockSignals(False)
        self._dirty = True
        self.system_requested.emit(system)
        s = self.system
        same = s.n == system.n and all(
            np.allclose(a, b)
            for a, b in (
                (s.masses, system.masses),
                (s.stiffness, system.stiffness),
                (s.damping, system.damping),
                (s.friction, system.friction),
            )
        )
        if not same:
            self.set_system(system)  # nobody passed it back: show it here alone
        if self._dirty:  # set_system didn't redraw (the page is hidden, or the chain came back unchanged)
            self.redraw()

    def _sensors(self) -> tuple[int, ...]:
        return tuple(i for i, box in enumerate(self.sensor_boxes[: self.system.n]) if box.isChecked())

    def _element(self) -> nl.Element:
        return nl.Element(
            nl.KINDS[self.kind.currentIndex()],
            min(int(self.spring.value()), self.system.n) - 1,
            self.k3.value(),
            self.gap.value() / 1000,
            self.k_gap.value(),
        )

    def _set_lambda(self, lam: float | None) -> None:
        if lam is None or lam <= 0:
            return
        set_quietly(self.method, fid.METHODS.index(fid.Method.TIKHONOV))
        self.log_lam.setValue(round(math.log10(lam), 2))
        self._on_solve()

    def _on_sine(self) -> None:
        self._sweep()
        self.waveform = None
        self._draw_frf()
        self._draw_tab()
        settle_plots(self)

    def _on_measure(self) -> None:
        self._measure()
        self._on_solve()

    def _on_solve(self) -> None:
        self._identify()
        self._draw_force()
        self._draw_tab()
        settle_plots(self)

    def _on_distortion(self) -> None:
        self.waveform = None
        self._draw_frf()
        self._draw_tab()
        settle_plots(self)

    # ------------------------------------------------------------- working out
    def redraw(self) -> None:
        self._dirty = False
        s = self.system
        f = s.friction
        self.chain_info.setText(
            "Friction: " + ", ".join(f"{v:.3g} N on mass {i + 1}" for i, v in enumerate(f) if v > 0) + "."
            if np.any(f > 0)
            else "No friction."
        )
        self._sweep()
        self._measure()
        self._identify()
        self.waveform = None
        self._draw_frf()
        self._draw_force()
        self._draw_tab()
        settle_plots(self)

    def _sweep(self) -> None:
        s = self.system
        n = s.n
        kind = nl.KINDS[self.kind.currentIndex()]
        self.element_form.form.setRowVisible(self.k3, kind is nl.Kind.CUBIC)
        self.element_form.form.setRowVisible(self.gap, kind is nl.Kind.GAP)
        self.element_form.form.setRowVisible(self.k_gap, kind is nl.Kind.GAP)
        self.element_form.form.setRowVisible(self.spring, kind is not nl.Kind.NONE)
        M, _, K = s.matrices()
        w = np.sqrt(np.clip(np.sort(np.linalg.eigvals(np.linalg.solve(M, K)).real), 0.0, None))
        self.mode_hz = w / (2 * math.pi)
        r = min(int(self.mode.value()), n) - 1
        f0 = max(float(self.mode_hz[r]), 1e-3)
        self.freqs = np.linspace(BAND[0] * f0, BAND[1] * f0, STEPS)
        force = min(int(self.force.value()), n) - 1
        e = self._element()
        self.sweeps = [
            (
                nl.stepped_sine(s, e, force, F.value(), self.freqs),
                nl.stepped_sine(s, e, force, F.value(), self.freqs, False),
            )
            for F in self.levels
        ]

    def _measure(self) -> None:
        s = self.system
        self.h = fid.step_for(s)
        self.truth = fid.force_history(
            fid.SHAPES[self.shape.currentIndex()], fid.SAMPLES, self.h, self.width.value() / 1000
        )
        sensors = self._sensors()
        if not sensors:
            self.problem = None
            self.curve = None
            return
        G = fid.markov(
            s,
            min(int(self.hit.value()), s.n) - 1,
            sensors,
            fid.OUTPUTS[self.output.currentIndex()],
            self.h,
            fid.SAMPLES,
        )
        self.problem = fid.set_up(G, self.truth, self.noise.value() / 100)
        self.curve = fid.l_curve(self.problem)

    def _identify(self) -> None:
        method = fid.METHODS[self.method.currentIndex()]
        self.ident_form.form.setRowVisible(self.log_lam, method is fid.Method.TIKHONOV)
        self.ident_form.form.setRowVisible(self.k, method is fid.Method.TSVD)
        if self.problem is None:
            self.estimate = np.zeros(0)
            return
        self.estimate = _solve(self.problem, method, 10 ** self.log_lam.value(), int(self.k.value()))

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        dot = QtCore.Qt.PenStyle.DotLine
        dash = QtCore.Qt.PenStyle.DashLine
        self.linear_item.setPen(pg.mkPen(colors.grey, width=1.5, style=dot))
        for k in range(3):
            c = colors.mode[k]
            self.up_items[k].setPen(pg.mkPen(c, width=2))
            self.down_items[k].setPen(pg.mkPen(c, width=2, style=dash))
            mark(self.jump_items[k], c, filled=True)
        self.dist_line.setPen(pg.mkPen(colors.grey, width=1, style=dash))
        self.truth_item.setPen(pg.mkPen(colors.strong, width=2))
        self.estimate_item.setPen(pg.mkPen(colors.fit, width=2, style=dash))
        self.wave_item.setPen(pg.mkPen(colors.strong, width=2))
        self.sine_item.setPen(pg.mkPen(colors.grey, width=1.5, style=dash))
        self.harm_bars.setOpts(brush=pg.mkBrush(colors.mode[0]), pen=None)
        self.l_item.setPen(pg.mkPen(colors.strong, width=2))
        mark(self.l_now, colors.force, filled=True)
        mark(self.l_corner, colors.fit)
        self.s_item.setPen(pg.mkPen(colors.strong, width=2))
        self.beta_item.setPen(None)
        self.beta_item.setSymbol("o")
        self.beta_item.setSymbolSize(4)
        self.beta_item.setSymbolBrush(pg.mkBrush(colors.mode[1]))
        self.beta_item.setSymbolPen(None)
        self.lam_line.setPen(pg.mkPen(colors.force, width=1.5, style=dash))
        if self.freqs.size:
            self.redraw()

    def _draw_frf(self) -> None:
        s = self.system
        n = s.n
        out = min(int(self.out.value()), n) - 1
        force = min(int(self.force.value()), n) - 1
        f = self.freqs
        M, C, K = s.matrices()
        w = 2 * math.pi * f
        e = np.zeros(n)
        e[force] = 1.0
        H = np.linalg.solve(
            K[None] - (w**2)[:, None, None] * M[None] + 1j * w[:, None, None] * C[None],
            e[None, :, None].astype(complex),
        )[:, :, 0]
        self.linear_item.setData(f, np.abs(H[:, out]))
        self.frf_legend.clear()
        self.frf_legend.addItem(self.linear_item, "linear")
        jumps = []
        for k, ((up, down), F) in enumerate(zip(self.sweeps, self.levels, strict=True)):
            F = F.value()
            self.up_items[k].setData(up.hz, np.abs(up.X[:, out]) / F)
            self.down_items[k].setData(down.hz, np.abs(down.X[:, out]) / F)
            points = [(up.hz[j], abs(up.X[j, out]) / F) for j in up.jumps] + [
                (down.hz[j], abs(down.X[j, out]) / F) for j in down.jumps
            ]
            self.jump_items[k].setData([p[0] for p in points], [p[1] for p in points])
            self.frf_legend.addItem(self.up_items[k], f"{F:g} N")
            if up.jumps or down.jumps:
                jumps.append(
                    f"{F:g} N: up at {', '.join(f'{up.hz[j]:.4g}' for j in up.jumps) or '—'} Hz, "
                    f"down at {', '.join(f'{down.hz[j]:.4g}' for j in down.jumps) or '—'} Hz"
                )
        self.frf_plot.setXRange(f[0], f[-1], padding=0)
        self.dist_line.setPos(self.dist_hz.value())
        spread = nl.homogeneity([u for u, _ in self.sweeps], out, [F.value() for F in self.levels])
        apart = max(
            float(np.max(np.abs(np.abs(u.X[:, out]) - np.abs(d.X[::-1, out])))) / F.value()
            for (u, d), F in zip(self.sweeps, self.levels, strict=True)
        )
        top = float(np.max(np.abs(H[:, out])))
        spread = 0.0 if spread < 1e-9 else spread  # round-off
        text = (
            f"Response of mass {out + 1} to a force on mass {force + 1}. The levels' FRFs differ by "
            f"<b>{100 * spread:.3g} %</b> of the peak"
        )
        text += "; up and down agree." if apart < 1e-3 * top else f"; up and down differ by {100 * apart / top:.3g} %."
        if jumps:
            text += " Jumps: " + "; ".join(jumps) + "."
        if spread < 1e-6:
            text += " Linear, as far as these levels can tell."
        self.frf_readout.setText(text)

    def _draw_force(self) -> None:
        t = np.arange(fid.SAMPLES) * self.h
        self.truth_item.setData(t, self.truth)
        p = self.problem
        if p is None or not self.estimate.size:
            self.estimate_item.setData([], [])
            self.force_readout.setText("No sensors: tick at least one mass.")
            return
        self.estimate_item.setData(t, self.estimate)
        top = float(np.abs(self.truth).max())
        big = float(np.abs(self.estimate).max())
        self.force_plot.setXRange(0, t[-1], padding=0)
        if big > 3 * top:
            self.force_plot.setYRange(-1.5 * top, 1.5 * top, padding=0)
        else:
            self.force_plot.enableAutoRange(axis="y")
        method = fid.METHODS[self.method.currentIndex()]
        how = {
            fid.Method.INVERSE: "the plain inverse",
            fid.Method.TSVD: f"truncated SVD, k = {int(self.k.value())}",
            fid.Method.TIKHONOV: f"Tikhonov, λ = {10 ** self.log_lam.value():.3g}",
        }[method]
        err = fid.error(self.estimate, self.truth)
        text = (
            f"Identified by {how}: off by <b>{100 * err:.3g} %</b> (the error's norm against the force's). G's "
            f"singular values span {math.log10(p.s[0] / max(p.s[-1], 1e-300)):.3g} decades."
        )
        if big > 3 * top:
            text += f" Its largest is {big / top:.3g} times the true force's (clipped here)."
        self.force_readout.setText(text)

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_DISTORTION:
            self._draw_distortion()
        elif i == TAB_LCURVE:
            self._draw_lcurve()
        elif i == TAB_PICARD:
            self._draw_picard()

    def _draw_distortion(self) -> None:
        s = self.system
        n = s.n
        out = min(int(self.out.value()), n) - 1
        force = min(int(self.force.value()), n) - 1
        level = self.levels[self.dist_level.currentIndex()].value()
        hz = self.dist_hz.value()
        if self.waveform is None:
            start = nl.harmonic_balance(s, self._element(), force, level, hz)
            self.waveform = nl.steady_response(s, self._element(), force, level, hz, start.X)
        w = self.waveform
        one = float(w.harmonics[0, out])
        if one <= 0:
            return
        cycles = np.concatenate([w.t, w.t + w.t[-1] + (w.t[1] - w.t[0])]) * hz
        x = np.concatenate([w.x[:, out], w.x[:, out]]) / one
        spectrum = np.fft.rfft(w.x[:, out])
        phase = float(np.angle(spectrum[1]))
        self.wave_item.setData(cycles, x)
        self.sine_item.setData(cycles, np.cos(2 * math.pi * cycles + phase))
        self.wave_plot.setXRange(0, 2, padding=0)
        share = 100 * w.harmonics[1:, out] / one
        self.harm_bars.setOpts(x=np.arange(2, nl.HARMONICS + 1), height=share, width=0.6)
        self.harm_plot.setXRange(1.4, nl.HARMONICS + 0.6, padding=0)
        self.harm_plot.getAxis("bottom").setTicks([[(k, f"{k}X") for k in range(2, nl.HARMONICS + 1)]])
        self.harm_plot.setYRange(0, max(float(share.max()) * 1.15, 1e-3), padding=0)
        hb = nl.harmonic_balance(s, self._element(), force, level, hz)
        self.dist_note.setText(
            f"Mass {out + 1}'s steady response to {level:g} N at {hz:.4g} Hz (solid), stepped in time until a "
            f"cycle repeats ({w.cycles} cycles), against a pure sine (dashed). Below, its harmonics against 1X: "
            f"3X is {share[1]:.3g} %, 2X {share[0]:.2g} %. Harmonic balance's 1X is "
            f"{100 * (abs(hb.X[out]) / one - 1):+.2g} % from the time response's."
        )

    def _draw_lcurve(self) -> None:
        c, p = self.curve, self.problem
        if c is None or p is None:
            self.l_item.setData([], [])
            self.l_note.setText("No sensors: tick at least one mass.")
            return
        self.l_item.setData(c.residual, c.size)
        k = int(np.argmin(np.abs(np.log(c.lams) - math.log(c.corner))))
        self.l_corner.setData([c.residual[k]], [c.size[k]])
        res = float(np.linalg.norm(p.G @ self.estimate - p.y)) if self.estimate.size else math.nan
        size = float(np.linalg.norm(self.estimate)) if self.estimate.size else math.nan
        if np.isfinite(res) and res > 0 and size > 0:
            self.l_now.setData([res], [size])
        else:
            self.l_now.setData([], [])
        self.l_note.setText(
            f"Each point a λ: the residual ‖Gf − y‖ against the force's size ‖f‖. The diamond is the corner, λ = "
            f"{c.corner:.3g}; the dot is the solution now. Left of the corner, smaller λ buys almost no residual "
            f"and a much larger force (noise); below it, larger λ smooths the force away. The noise's own norm is "
            f"{p.noise:.3g}."
        )

    def _draw_picard(self) -> None:
        p = self.problem
        if p is None:
            self.s_item.setData([], [])
            self.beta_item.setData([], [])
            self.picard_note.setText("No sensors: tick at least one mass.")
            return
        i = np.arange(1, p.s.size + 1)
        self.s_item.setData(i, np.maximum(p.s, 1e-300))
        self.beta_item.setData(i, np.maximum(np.abs(p.coefficients()), 1e-300))
        method = fid.METHODS[self.method.currentIndex()]
        lam = 10 ** self.log_lam.value() if method is fid.Method.TIKHONOV else None
        if method is fid.Method.TSVD:
            lam = float(p.s[min(int(self.k.value()), p.s.size) - 1])
        self.lam_line.setVisible(lam is not None)
        if lam is not None:
            self.lam_line.setPos(math.log10(lam))
        low = max(float(p.s[-1]), float(p.s[0]) * 1e-22)
        self.picard_plot.setYRange(math.log10(low), math.log10(p.s[0]) + 1, padding=0.02)
        self.picard_note.setText(
            "G's singular values (line) and the data's coefficient on each direction (dots), |u<sub>i</sub>ᵀy|. "
            "Where the dots stop falling with the line, they're noise: dividing them by s<sub>i</sub> is what "
            "blows the plain inverse up. "
            + (
                "The dashed line is λ: Tikhonov keeps the directions above it."
                if method is fid.Method.TIKHONOV
                else "The dashed line is the last s<sub>i</sub> truncated SVD keeps."
                if method is fid.Method.TSVD
                else ""
            )
        )
