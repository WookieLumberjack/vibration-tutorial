"""Damping models page: viscous, hysteretic, Coulomb friction and a viscoelastic (Zener) element on one oscillator.

Left: the oscillator, the damping they're all matched to at f_n, the loop's amplitude and frequency, and
the drive for the response. Centre: the hysteresis loops (force against displacement over a cycle) and the
energy per cycle against frequency. Right: storage stiffness and loss factor against frequency, the
frequency response, the free decay, and the Theory. The page has its own model (core/damping_models.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.damping_models import MODELS, Matched, Model, coulomb_stop
from ..core.practice import Problem, Question, at_most, kept, number, pick, rounded
from ..core.sdof import TWO_PI
from . import damping_figures, settle_plots
from .layouts import ColumnScroll, FlowLayout
from .lesson import (
    Form,
    TrialBox,
    fmt,
    note_label,
    plot,
    readout_label,
    set_quietly,
    trials_html,
)
from .links import load_notes, page_link
from .notes_view import NotesBrowser
from .style import colors
from .theming import add_legend

RATIOS = np.geomspace(0.05, 20.0, 400)  # frequency over f_n on the energy and loss plots
TAB_LOSS, TAB_FRF, TAB_DECAY = range(3)
SHORT = {Model.VISCOUS: "Viscous", Model.HYSTERETIC: "Hysteretic", Model.COULOMB: "Coulomb", Model.ZENER: "Zener"}


@dataclass(frozen=True)
class DampingTrial:
    name: str
    note: str
    zeta: float = 0.05
    amplitude_mm: float = 10.0
    relax: float = 1.0
    ratio: float = 1.0  # the loop's frequency over f_n
    damping_only: bool = True
    force: float = 1.0
    tab: int = TAB_LOSS
    mass: float = 1.0
    stiffness: float = 400.0


def matched(t: DampingTrial) -> Matched:
    """The four models as a trial (or the page's `practice_state`) sets them."""
    return Matched(t.mass, t.stiffness, t.zeta, 1e-3 * t.amplitude_mm, t.relax)


TRIALS = (
    DampingTrial(
        "Four loops, one energy",
        "<m>m</m> = 1 kg, <m>k</m> = 400 N/m (<m>f<sub>n</sub></m> = 3.18 Hz), each model matched to "
        "<m>ζ</m> = 0.05 there. Cycled at <m>f<sub>n</sub></m> through ±10 mm, every loop of damping force "
        "encloses the same <m>πcω<sub>n</sub>X²</m> = 12.6 mJ: that is what matched means. Viscous, hysteretic "
        "and Zener damping make ellipses, their force a quarter cycle ahead of the motion; friction makes a "
        "rectangle, <m>±F<sub>f</sub></m> = ±0.31 N, flipping at each end of the stroke, which doesn't shrink as "
        "the motion does.",
        tab=TAB_LOSS,
    ),
    DampingTrial(
        "Twice the frequency",
        "The same cycle at <m>2f<sub>n</sub></m>. The viscous loop is twice as fat (25.1 mJ: its force grows with "
        "the velocity), the hysteretic and friction loops are unchanged (12.6 mJ), and the Zener element's, "
        "past its loss peak, is thinner (10.7 mJ).",
        ratio=2.0,
    ),
    DampingTrial(
        "With the spring: what a test machine draws",
        "The whole force, spring and damping together, as a test machine measures it on a sample: each loop is "
        "the spring's line <m>kx</m> (dashed) opened out by the damping. The tilt is the stiffness (4 N at 10 mm), "
        "the width the loss (0.4 N): a lightly damped part draws a thin loop.",
        damping_only=False,
    ),
    DampingTrial(
        "Friction can't stop a resonance",
        "1 N at every frequency. The viscous, hysteretic and Zener peaks stop at 25 mm, <m>(F/k)/2ζ</m>. Friction, "
        "matched to them at 10 mm, takes at most <m>4F<sub>f</sub>X</m> a cycle, which grows only as <m>X</m> while the force "
        "puts in <m>πFX</m>: since <m>F</m> = 1 N &gt; <m>4F<sub>f</sub>/π</m> = 0.4 N, the resonance runs away. Set the drive below "
        "0.4 N and friction holds the mass still (to this approximation).",
        tab=TAB_FRF,
    ),
    DampingTrial(
        "A straight-line decay",
        "Released from 10 mm. The viscous swing falls by the same fraction each cycle, <m>e<sup>−δ</sup></m> = 0.73, "
        "for ever. Friction's falls by the same amount, <m>4F<sub>f</sub>/k</m> = 3.1 mm, and stops dead after three "
        "cycles, 0.58 mm off centre, where the spring can no longer beat the friction.",
        tab=TAB_DECAY,
    ),
    DampingTrial(
        "A viscoelastic loss peak",
        "The Zener element: a spring <m>k<sub>0</sub></m> beside a spring <m>k<sub>1</sub></m> and a dashpot in series. Slowly "
        "the dashpot gives and only <m>k<sub>0</sub></m> = 364 N/m holds; fast it locks and "
        "<m>k<sub>0</sub> + k<sub>1</sub></m> = 444 N/m does. In between, at its relaxation frequency (here <m>f<sub>n</sub></m>), the loss "
        "factor peaks at 0.1. Rubber mounts and polymers behave like this.",
        tab=TAB_LOSS,
    ),
    DampingTrial(
        "Relaxing well above resonance",
        "The Zener element's peak moved to <m>10f<sub>n</sub></m>. Below it, its loss grows in proportion to "
        "frequency, as a viscous damper's does: through the oscillator's band it is a viscous damper. It "
        "needs a stiffer arm for the same damping at <m>f<sub>n</sub></m>.",
        relax=10.0,
        tab=TAB_LOSS,
    ),
)

# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


def _viscous_energy(rng: np.random.Generator) -> Question:
    c = pick(rng, (0.5, 1, 2, 5, 10))
    f = pick(rng, (2, 5, 10, 20, 50))
    x = pick(rng, (1, 2, 5, 10))
    energy = math.pi * c * TWO_PI * f * (1e-3 * x) ** 2
    return Question(
        f"A viscous damper of {c:g} N·s/m is cycled at {f:g} Hz through ±{x:g} mm. How much energy does it take "
        "out per cycle?",
        1e3 * energy,
        f"W<sub>d</sub> = πcΩX² with Ω = 2π × {f:g} = {number(TWO_PI * f)} rad/s and X = {number(1e-3 * x)} m: "
        f"π × {c:g} × {number(TWO_PI * f)} × {number(1e-3 * x)}² = {number(1e3 * energy)} mJ.",
        unit="mJ",
    )


def _friction_from_decay(rng: np.random.Generator) -> Question:
    k = pick(rng, (200, 400, 1000, 2500))
    drop = pick(rng, (0.5, 1, 2, 3, 4))
    friction = k * 1e-3 * drop / 4.0
    return Question(
        f"A mass on a spring of {k:g} N/m rings down in a straight line, its swing falling by {drop:g} mm every "
        "cycle. What friction force is it sliding against?",
        friction,
        f"Each half cycle swings about the point F<sub>f</sub>/k off centre on the side friction pushes to, so the "
        f"swing loses 2F<sub>f</sub>/k a half cycle, 4F<sub>f</sub>/k a cycle: F<sub>f</sub> = k × drop/4 = "
        f"{k:g} × {number(1e-3 * drop)}/4 = {number(friction)} N.",
        unit="N",
        setup=dict(stiffness=k, tab=TAB_DECAY),
    )


def _friction_swings(rng: np.random.Generator) -> Question:
    k = pick(rng, (200, 400, 1000))
    friction = pick(rng, (0.2, 0.5, 1, 2))
    d = 1e3 * friction / k  # mm
    n = int(pick(rng, (2, 3, 4, 5, 6)))
    x0 = rounded(d * (2 * n + pick(rng, (-0.6, -0.3, 0.3, 0.6))), 2)
    swings, _ = coulomb_stop(k, friction, 1e-3 * x0)
    return Question(
        f"A mass on a spring of {k:g} N/m slides against a friction force of {friction:g} N. Released from rest "
        f"{number(x0, 2)} mm from centre, how many half cycles (swings from one turn to the next) does it make "
        "before it sticks?",
        swings,
        f"F<sub>f</sub>/k = {number(d)} mm. Each swing takes 2F<sub>f</sub>/k = {number(2 * d)} mm off the distance "
        f"from centre, and it sticks at the first turn within {number(d)} mm of it: {number(x0, 2)} mm takes "
        f"{swings} swings.",
        tolerance=0.01,
    )


def _equivalent_viscous(rng: np.random.Generator) -> Question:
    friction = pick(rng, (0.5, 1, 2, 5))
    f = pick(rng, (2, 5, 10, 20))
    x = pick(rng, (1, 2, 5, 10))
    w = TWO_PI * f
    c = 4.0 * friction / (math.pi * w * 1e-3 * x)
    return Question(
        f"A friction force of {friction:g} N is cycled at {f:g} Hz through ±{x:g} mm. What viscous damper takes the "
        "same energy per cycle?",
        c,
        f"Friction takes 4F<sub>f</sub>X, a viscous damper πcΩX²; equal when c<sub>eq</sub> = 4F<sub>f</sub>/πΩX = "
        f"4 × {friction:g}/(π × {number(w)} × {number(1e-3 * x)}) = {number(c)} N·s/m. Half the amplitude, twice "
        "the c<sub>eq</sub>: friction is heavy damping for small motions.",
        unit="N·s/m",
    )


def _zener_peak(rng: np.random.Generator) -> Question:
    k0 = pick(rng, (100, 200, 400, 1000))
    a = pick(rng, (0.1, 0.25, 0.5, 1, 2, 3))
    k1 = rounded(a * k0)
    a = k1 / k0
    eta = a / (2.0 * math.sqrt(1.0 + a))
    return Question(
        f"A rubber mount behaves as a Zener element: {k0:g} N/m slowly (relaxed), {number(k0 + k1)} N/m fast "
        "(unrelaxed). What is its loss factor at its peak?",
        eta,
        f"k<sub>1</sub> = {number(k1)} N/m, so a = k<sub>1</sub>/k<sub>0</sub> = {number(a)}. η(Ωτ) = "
        f"aΩτ/(1 + (1 + a)Ω²τ²) peaks at Ωτ = 1/√(1 + a), where η = a/(2√(1 + a)) = {number(eta)}. The "
        "further apart the relaxed and unrelaxed stiffnesses, the more it can damp.",
        setup=dict(tab=TAB_LOSS),
    )


def _frequency_doubled(rng: np.random.Generator) -> Question:
    return Question(
        "Four dampers, viscous, hysteretic, Zener (its loss peak at f<sub>n</sub>) and friction, each take the "
        "same energy per cycle at f<sub>n</sub>. Cycled through the same amplitude at 2f<sub>n</sub>, which takes "
        "the most?",
        0,
        "Viscous: its force grows with the velocity, so it takes twice as much. Hysteretic and friction take the "
        "same as before, and the Zener element, past its peak, a little less.",
        choices=("Viscous", "Hysteretic", "Friction", "Zener"),
        setup=dict(ratio=2.0, tab=TAB_LOSS),
        predict=True,
    )


def _amplitude_halved(rng: np.random.Generator) -> Question:
    return Question(
        "A friction damper is cycled through half the amplitude it was, at the same frequency. What happens to "
        "the energy it takes per cycle?",
        0,
        "It halves: W<sub>d</sub> = 4F<sub>f</sub>X, the same force over half the stroke. A viscous or hysteretic "
        "damper's (∝ X²) falls to a quarter; so friction is relatively heavier damping for small motions.",
        choices=("It halves", "It falls to a quarter", "It stays the same", "It doubles"),
        setup=dict(amplitude_mm=5.0, tab=TAB_LOSS),
        predict=True,
    )


def _friction_resonance(rng: np.random.Generator) -> Question:
    return Question(
        "A mass on a spring with friction (no other damping) is driven by a harmonic force at its natural "
        "frequency, the force larger than 4/π times the friction. What becomes of its swing?",
        0,
        "It grows without bound (until something else stops it). At resonance the force puts in πFX a cycle and "
        "friction takes 4F<sub>f</sub>X: both grow as X, so if F &gt; 4F<sub>f</sub>/π the input wins at every "
        "amplitude.",
        choices=("It grows without bound", "It settles where friction balances the force", "It stays stuck"),
        setup=dict(force=1.0, tab=TAB_FRF),
        predict=True,
    )


def _friction_spec(rng: np.random.Generator) -> Question:
    k = pick(rng, (200, 400, 1000))
    x0 = pick(rng, (5, 10, 20))
    swings = int(pick(rng, (3, 4, 5)))
    # friction that stops it after `swings` half cycles with room either side
    d = x0 / (2 * swings + 0.5)  # mm
    z = rounded(2.0 * d / (math.pi * x0), 3)  # F_f = πcω_nX/4 = πζkX/2 on 1 kg
    design = Matched(1.0, k, z, 1e-3 * x0)
    n_d, end = coulomb_stop(k, design.friction, 1e-3 * x0)
    off = rounded(1.25 * 1e3 * abs(end), 2)

    def goal(t: DampingTrial):
        m = matched(t)
        n, stop = coulomb_stop(t.stiffness, m.friction, 1e-3 * t.amplitude_mm)
        return (
            kept("The mass", t.mass, 1.0, "kg"),
            kept("The spring", t.stiffness, k, "N/m"),
            kept("The release, X", t.amplitude_mm, float(x0), "mm"),
            at_most("Swings before it sticks", n, swings),
            at_most("Where it sticks, off centre", 1e3 * abs(stop), off, "mm"),
        )

    return Question(
        f"A 1 kg mass on a spring of {k:g} N/m, released from {x0:g} mm, is to be stopped by friction alone (the "
        f"Free decay tab's Coulomb curve). It must stick within {swings} swings (half cycles), no more than "
        f"{number(off, 2)} mm off centre. The page sets the friction by matching it to ζ at the amplitude X: "
        "choose ζ.",
        0,
        f"More friction stops it sooner, but further off centre (anywhere within F<sub>f</sub>/k), and too much "
        f"never lets it move. Each swing takes 2F<sub>f</sub>/k off, so {swings} swings need about "
        f"2F<sub>f</sub>/k ≈ {x0:g}/{swings} mm. F<sub>f</sub> = πcω<sub>n</sub>X/4 = πζkX/2 (c = 2ζ√(km), m = 1 "
        f"kg): ζ = "
        f"{number(z)} gives F<sub>f</sub> = {number(design.friction)} N, {n_d} swings, stuck "
        f"{number(1e3 * abs(end))} mm off centre. Much more friction does too: with F<sub>f</sub>/k just under X/2 "
        "it makes one swing and sticks near the centre.",
        setup=dict(stiffness=k, amplitude_mm=float(x0), zeta=0.005, tab=TAB_DECAY),
        goal=goal,
        design=dict(stiffness=k, amplitude_mm=float(x0), zeta=z, tab=TAB_DECAY),
    )


PROBLEMS = tuple(
    Problem(key, title, "damping_page", f"See {page_link('damping_page', anchor, section)}{more}.", make)
    for key, title, anchor, section, more, make in (
        ("viscous-energy", "Energy per cycle", "viscous", "Viscous damping", "", _viscous_energy),
        ("friction-decay", "Friction from a decay", "coulomb", "Coulomb friction", "", _friction_from_decay),
        ("friction-swings", "When friction stops it", "coulomb", "Coulomb friction", "", _friction_swings),
        (
            "equivalent-viscous",
            "Equivalent viscous damping",
            "coulomb",
            "Coulomb friction",
            ": equal energies per cycle",
            _equivalent_viscous,
        ),
        ("zener-peak", "A rubber mount's loss peak", "viscoelastic", "The Zener element", "", _zener_peak),
        (
            "frequency-doubled",
            "Predict: twice the frequency",
            "loss-factor",
            "The loss factor, compared",
            "",
            _frequency_doubled,
        ),
        ("amplitude-halved", "Predict: half the amplitude", "coulomb", "Coulomb friction", "", _amplitude_halved),
        (
            "friction-resonance",
            "Predict: friction at resonance",
            "coulomb",
            "Coulomb friction",
            "",
            _friction_resonance,
        ),
        (
            "friction-spec",
            "Spec: stop it with friction",
            "coulomb",
            "Coulomb friction",
            ": how far does each swing fall, and where can it stick?",
            _friction_spec,
        ),
    )
)

TIPS = {
    "zeta": "<p>The damping ratio every model is matched to at f<sub>n</sub>: the same energy per cycle there as "
    "a viscous damper c = 2ζ√(km), and friction at the amplitude X.</p>",
    "amplitude": "<p>The amplitude X of the cycle the loops trace, where friction is matched, and the release "
    "of the free decay.</p>",
    "relax": "<p>Where the Zener element's loss factor peaks, as a multiple of f<sub>n</sub>.</p>",
    "ratio": "<p>The frequency of the cycle the loops trace, over f<sub>n</sub>.</p>",
    "force": "<p>The harmonic force on the Frequency response tab.</p>",
    "damping_only": "<p>Draw only the damping part of the force, without the spring's kx.</p>",
}
LOOP_TIP = (
    "<p>The force the spring and the damping carry, against the displacement, over one steady cycle "
    "x = X sin Ωt. The area inside each loop is the energy it takes out per cycle. The spring alone would be the "
    "straight line kx.</p>"
)
ENERGY_TIP = (
    "<p>The energy each model takes per cycle of amplitude X, against the frequency over f<sub>n</sub>. They "
    "meet at f<sub>n</sub>, where they are matched.</p>"
)
LOSS_TIP = (
    "<p>Top: the storage stiffness K' (the in-phase part of the force per metre) over k. Bottom: the loss "
    "factor η = K''/K', the out-of-phase part over the in-phase part (2ζ at resonance for a viscous "
    "damper).</p>"
)
FRF_TIP = (
    "<p>The steady amplitude under the harmonic force, against frequency, for each model. Friction's is the "
    "describing function's, which can't bound a resonance once F &gt; 4F<sub>f</sub>/π.</p>"
)
DECAY_TIP = (
    "<p>Released from rest at X (held there long, so the Zener element's dashpot is relaxed). Hysteretic "
    "damping has no causal time-domain model: its curve is the viscous one of the same loss at "
    "f<sub>n</sub>.</p>"
)

NOTES_HTML = load_notes("damping_models", figures=damping_figures.FIGURES, trials=trials_html(TRIALS))


class DampingModelsPage(QtWidgets.QWidget):
    """Controls on the left, the loops and energies in the centre, loss, response and decay on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = Matched()

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Each sets up the oscillator and the plots to show one idea from the Theory tab.</p>",
        )
        self.trials.picked.connect(self.load_trial)

        osc = Form("Oscillator and matching")
        self.mass = osc.number("Mass m:", 0.01, 1e4, 1.0, 3, " kg", "<p>The mass m.</p>")
        self.stiffness = osc.number("Stiffness k:", 0.01, 1e8, 400.0, 2, " N/m", "<p>The spring's stiffness k.</p>")
        self.zeta = osc.number("Matched to ζ:", 0.001, 0.5, 0.05, 4, "", TIPS["zeta"])
        self.amplitude = osc.number("Amplitude X:", 0.01, 1000.0, 10.0, 2, " mm", TIPS["amplitude"])
        self.relax = osc.number("Zener peak at:", 0.01, 100.0, 1.0, 3, " × fₙ", TIPS["relax"])
        self.info = note_label()
        osc.row(self.info)
        osc.changed.connect(self._on_model)

        loop = Form("Loops")
        self.ratio = loop.number("Cycle at:", 0.01, 100.0, 1.0, 3, " × fₙ", TIPS["ratio"])
        self.damping_only = QtWidgets.QCheckBox("Damping force only")
        self.damping_only.setChecked(True)
        self.damping_only.setToolTip(TIPS["damping_only"])
        loop.row(self.damping_only)
        self.shown: dict[Model, QtWidgets.QCheckBox] = {}
        row = FlowLayout()
        for m in MODELS:
            box = self.shown[m] = QtWidgets.QCheckBox(SHORT[m])
            box.setChecked(True)
            box.setToolTip(f"<p>Draw the {m.value.lower()} model.</p>")
            box.toggled.connect(self.redraw)
            row.add(box)
        loop.row(row)
        loop.changed.connect(self.redraw)
        self.damping_only.toggled.connect(self.redraw)

        drive = Form("Drive")
        self.force = drive.number("Force F:", 0.0, 1e6, 1.0, 4, " N", TIPS["force"])
        self.drive_note = note_label()
        drive.row(self.drive_note)
        drive.changed.connect(self.redraw)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        for w in (self.trials, osc, loop, drive):
            lv.addWidget(w)
        lv.addStretch(1)

        # --- centre
        self.loop_plot = plot("Force", "N", "Displacement x", "m", height=220)
        self.loop_plot.setToolTip(LOOP_TIP)
        self.loop_legend = add_legend(self.loop_plot.getPlotItem(), offset=(10, 10))
        self.spring_item = pg.PlotDataItem()
        self.loop_items = {m: pg.PlotDataItem() for m in MODELS}
        self.loop_plot.addItem(self.spring_item)
        for item in self.loop_items.values():
            self.loop_plot.addItem(item)
        self.loop_note = readout_label()
        self.energy_plot = plot("Energy per cycle", "J", "Frequency / fₙ", None, log_x=True, log_y=True, height=150)
        self.energy_plot.setToolTip(ENERGY_TIP)
        self.energy_plot.setMouseEnabled(x=False, y=False)
        self.energy_items = {m: pg.PlotDataItem() for m in MODELS}
        for item in self.energy_items.values():
            self.energy_plot.addItem(item)
        self.energy_now = pg.InfiniteLine(angle=90)
        self.energy_plot.addItem(self.energy_now)

        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.loop_plot, 3)
        cv.addWidget(self.loop_note)
        cv.addWidget(self.energy_plot, 2)

        # --- right
        loss = QtWidgets.QWidget()
        lsv = QtWidgets.QVBoxLayout(loss)
        self.storage_plot = plot("K' / k", None, "Frequency / fₙ", None, log_x=True, height=130)
        self.eta_plot = plot("Loss factor η", None, "Frequency / fₙ", None, log_x=True, log_y=True, height=150)
        self.eta_plot.setXLink(self.storage_plot)
        add_legend(self.eta_plot.getPlotItem(), offset=(10, 10))
        for p in (self.storage_plot, self.eta_plot):
            p.setToolTip(LOSS_TIP)
            p.setMouseEnabled(x=False, y=False)
        self.storage_items = {m: pg.PlotDataItem() for m in MODELS}
        self.eta_items = {m: pg.PlotDataItem(name=SHORT[m]) for m in MODELS}
        for m in MODELS:
            self.storage_plot.addItem(self.storage_items[m])
            self.eta_plot.addItem(self.eta_items[m])
        lsv.addWidget(self.storage_plot, 2)
        lsv.addWidget(self.eta_plot, 3)
        self.loss_note = note_label()
        lsv.addWidget(self.loss_note)

        frf = QtWidgets.QWidget()
        frv = QtWidgets.QVBoxLayout(frf)
        self.frf_plot = plot("|X|", "m", "Frequency", "Hz", log_x=True, log_y=True, height=200)
        self.frf_plot.setToolTip(FRF_TIP)
        self.frf_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.frf_plot.getPlotItem(), offset=(-10, 10))
        self.frf_items = {m: pg.PlotDataItem(name=SHORT[m], connect="finite") for m in MODELS}
        for item in self.frf_items.values():
            self.frf_plot.addItem(item)
        frv.addWidget(self.frf_plot, 1)
        self.frf_note = note_label()
        frv.addWidget(self.frf_note)

        decay = QtWidgets.QWidget()
        dv = QtWidgets.QVBoxLayout(decay)
        self.decay_plot = plot("Displacement x", "m", "Time", "s", height=200)
        self.decay_plot.setToolTip(DECAY_TIP)
        add_legend(self.decay_plot.getPlotItem(), offset=(-10, 10))
        self.decay_items = {
            m: pg.PlotDataItem(name=SHORT[m] + (" (viscous equivalent)" if m is Model.HYSTERETIC else ""))
            for m in MODELS
        }
        self.stick_band = pg.LinearRegionItem(orientation="horizontal", movable=False)
        self.decay_plot.addItem(self.stick_band)
        for item in self.decay_items.values():
            self.decay_plot.addItem(item)
        dv.addWidget(self.decay_plot, 1)
        self.decay_note = note_label()
        dv.addWidget(self.decay_note)

        self.notes = NotesBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(loss, "Stiffness && loss")
        self.tabs.addTab(frf, "Frequency response")
        self.tabs.addTab(decay, "Free decay")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._draw_tab)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(ColumnScroll(left))
        splitter.addWidget(center)
        splitter.addWidget(self.tabs)
        splitter.setSizes([320, 760, 560])
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.apply_theme()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(DampingTrial("", "", **setup))

    def practice_state(self) -> DampingTrial:
        """The page's settings, as a trial: what a practice spec checks."""
        return DampingTrial(
            "",
            "",
            zeta=self.zeta.value(),
            amplitude_mm=self.amplitude.value(),
            relax=self.relax.value(),
            ratio=self.ratio.value(),
            damping_only=self.damping_only.isChecked(),
            force=self.force.value(),
            tab=self.tabs.currentIndex(),
            mass=self.mass.value(),
            stiffness=self.stiffness.value(),
        )

    def apply_trial(self, t: DampingTrial) -> None:
        for box, value in (
            (self.mass, t.mass),
            (self.stiffness, t.stiffness),
            (self.zeta, t.zeta),
            (self.amplitude, t.amplitude_mm),
            (self.relax, t.relax),
            (self.ratio, t.ratio),
            (self.damping_only, t.damping_only),
            (self.force, t.force),
        ):
            set_quietly(box, value)
        for box in self.shown.values():
            set_quietly(box, True)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self._on_model()

    def _on_model(self) -> None:
        self.model = Matched(
            self.mass.value(),
            self.stiffness.value(),
            self.zeta.value(),
            1e-3 * self.amplitude.value(),
            self.relax.value(),
        )
        self.redraw()

    # ------------------------------------------------------------- drawing
    def _colors(self) -> dict[Model, str]:
        c = colors
        return {Model.VISCOUS: c.mass[0], Model.HYSTERETIC: c.mode[1], Model.COULOMB: c.force, Model.ZENER: c.mode[3]}

    def apply_theme(self) -> None:
        c = colors
        cols = self._colors()
        for m in MODELS:
            pen = pg.mkPen(cols[m], width=2)
            for items in (
                self.loop_items,
                self.energy_items,
                self.storage_items,
                self.eta_items,
                self.frf_items,
                self.decay_items,
            ):
                items[m].setPen(pen)
        for items in (self.energy_items, self.storage_items, self.eta_items):  # on the hysteretic one there
            items[Model.COULOMB].setPen(pg.mkPen(cols[Model.COULOMB], width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.decay_items[Model.HYSTERETIC].setPen(
            pg.mkPen(cols[Model.HYSTERETIC], width=2, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.spring_item.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.energy_now.setPen(pg.mkPen(c.muted, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.stick_band.setBrush(pg.mkBrush(*pg.mkColor(c.force).getRgb()[:3], 30))
        for line in self.stick_band.lines:
            line.setPen(pg.mkPen(c.force, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.redraw()

    def _visible(self) -> list[Model]:
        return [m for m in MODELS if self.shown[m].isChecked()]

    def redraw(self) -> None:
        mm = self.model
        k0, k1, tau = mm.zener()
        self.info.setText(
            f"f<sub>n</sub> = {mm.natural_hz:.4g} Hz. Matched: c = {mm.c:.4g} N·s/m, η = {mm.eta:.4g}, "
            f"F<sub>f</sub> = {mm.friction:.4g} N, Zener k<sub>0</sub> = {k0:.4g} N/m, "
            f"k<sub>1</sub> = {k1:.4g} N/m, τ = {tau:.4g} s."
        )
        self._draw_loops()
        self._draw_energy()
        self._draw_tab()
        settle_plots(self)

    def _draw_loops(self) -> None:
        mm = self.model
        w = self.ratio.value() * mm.omega_n
        only = self.damping_only.isChecked()
        self.loop_legend.clear()
        X = mm.amplitude
        lines = []
        for m in MODELS:
            item = self.loop_items[m]
            if not self.shown[m].isChecked():
                item.setData([], [])
                continue
            x, total, damp = mm.loop(m, w, 600)
            item.setData(x, damp if only else total)
            e = float(mm.energy_per_cycle(m, np.array([w]))[0])
            self.loop_legend.addItem(item, f"{SHORT[m]}: {fmt(e, 'J')}")
            lines.append(f"{SHORT[m]} {fmt(e, 'J')}")
        if only:
            self.spring_item.setData([], [])
        else:
            self.spring_item.setData([-X, X], [-mm.stiffness * X, mm.stiffness * X])
            self.loop_legend.addItem(self.spring_item, "spring kx")
        self.loop_plot.setXRange(-1.1 * X, 1.1 * X, padding=0)
        self.loop_plot.enableAutoRange(axis="y")
        self.loop_note.setText(
            f"Energy per cycle at {self.ratio.value():.3g} f<sub>n</sub>, X = {fmt(X, 'm')}: "
            + ", ".join(lines)
            + ". (The area inside each loop.)"
        )

    def _draw_energy(self) -> None:
        mm = self.model
        w = RATIOS * mm.omega_n
        for m in MODELS:
            item = self.energy_items[m]
            item.setData(RATIOS, mm.energy_per_cycle(m, w)) if self.shown[m].isChecked() else item.setData([], [])
        self.energy_now.setPos(math.log10(self.ratio.value()))
        self.energy_plot.setXRange(math.log10(RATIOS[0]), math.log10(RATIOS[-1]), padding=0)

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_LOSS:
            self._draw_loss()
        elif i == TAB_FRF:
            self._draw_frf()
        elif i == TAB_DECAY:
            self._draw_decay()

    def _draw_loss(self) -> None:
        mm = self.model
        w = RATIOS * mm.omega_n
        for m in MODELS:
            K = mm.dynamic_stiffness(m, w)
            if self.shown[m].isChecked():
                self.storage_items[m].setData(RATIOS, K.real / mm.stiffness)
                self.eta_items[m].setData(RATIOS, K.imag / K.real)
            else:
                self.storage_items[m].setData([], [])
                self.eta_items[m].setData([], [])
        self.storage_plot.setXRange(math.log10(RATIOS[0]), math.log10(RATIOS[-1]), padding=0)
        k0, k1, tau = mm.zener()
        peak = 1.0 / (tau * math.sqrt(1.0 + k1 / k0)) / mm.omega_n
        self.loss_note.setText(
            f"Zener: relaxed stiffness k<sub>0</sub> = {k0:.4g} N/m, unrelaxed k<sub>0</sub> + k<sub>1</sub> = "
            f"{k0 + k1:.4g} N/m; loss peak {k1 / (2 * math.sqrt(k0 * (k0 + k1))):.3g} at {peak:.3g} f<sub>n</sub>. "
            f"Friction's η = 4F<sub>f</sub>/πkX = {mm.eta:.3g} holds at X = {fmt(mm.amplitude, 'm')} only."
        )

    def _draw_frf(self) -> None:
        mm = self.model
        F = self.force.value()
        f = np.geomspace(0.1 * mm.natural_hz, 10.0 * mm.natural_hz, 1500)
        f = f[np.abs(f / mm.natural_hz - 1.0) > 1e-9]
        bottom = math.inf
        for m in MODELS:
            if not self.shown[m].isChecked() or F <= 0:
                self.frf_items[m].setData([], [])
                continue
            X = np.abs(mm.receptance(m, f, F)) * F
            X = np.where(X > 0, X, np.nan)
            self.frf_items[m].setData(f, X)
            if np.isfinite(X).any():
                bottom = min(bottom, float(np.nanmin(X)))
        top = F / mm.stiffness / mm.eta * 4.0
        if math.isfinite(bottom):
            self.frf_plot.setYRange(math.log10(bottom), math.log10(top), padding=0.02)
        self.frf_plot.setXRange(math.log10(f[0]), math.log10(f[-1]), padding=0)
        threshold = 4.0 * mm.friction / math.pi
        if F > threshold:
            text = (
                f"F = {fmt(F, 'N')} &gt; 4F<sub>f</sub>/π = {fmt(threshold, 'N')}: friction can't stop the "
                "resonance (its curve runs off the top). "
            )
        else:
            text = (
                f"F = {fmt(F, 'N')} ≤ 4F<sub>f</sub>/π = {fmt(threshold, 'N')}: friction holds the mass still "
                "(the describing function's answer; really it sticks and slips). "
            )
        text += f"The others peak at (F/k)/η = {fmt(F / mm.stiffness / mm.eta, 'm')}."
        self.frf_note.setText(text)
        self.drive_note.setText(f"Friction's threshold 4F<sub>f</sub>/π = {fmt(threshold, 'N')}.")

    def _draw_decay(self) -> None:
        mm = self.model
        T = TWO_PI / mm.omega_n
        span = min(max(5.0 / (mm.zeta * mm.omega_n), 3 * T), 60 * T)
        t = np.linspace(0.0, span, 3000)
        for m in MODELS:
            if self.shown[m].isChecked():
                self.decay_items[m].setData(t, mm.free_decay(m, mm.amplitude, t))
            else:
                self.decay_items[m].setData([], [])
        d = mm.friction / mm.stiffness
        self.stick_band.setRegion((-d, d))
        self.stick_band.setVisible(self.shown[Model.COULOMB].isChecked())
        self.decay_plot.setXRange(0, span, padding=0)
        halves, stop = coulomb_stop(mm.stiffness, mm.friction, mm.amplitude)
        self.decay_note.setText(
            f"Viscous: each peak e<sup>−δ</sup> = {math.exp(-2 * math.pi * mm.zeta / math.sqrt(1 - mm.zeta**2)):.3g}"
            f" of the last. Friction: {fmt(4 * d, 'm')} less each cycle; it stops after {halves / 2:g} cycles, "
            f"{fmt(abs(stop), 'm')} off centre, inside the band ±F<sub>f</sub>/k = ±{fmt(d, 'm')}."
        )
