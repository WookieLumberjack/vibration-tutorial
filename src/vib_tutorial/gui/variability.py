"""Variability & localization page: the chain with scattered properties, and a mistuned ring of blades.

Left: the scatter on the Simulation page's chain (each property's coefficient of variation, how many chains), and
the ring (blades, coupling, mistuning, the engine order, the mode). Centre: the chain's receptance at its last mass,
nominal and the band the scattered chains span, and each mode's spread by Monte Carlo and to first order. Right:
the frequencies' histograms, the ring's modes tuned and mistuned, its forced response and the amplification against
mistuning, and the Theory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.model import ChainSystem
from ..core.practice import Problem, Question, at_least, at_most, kept, number, pick, rounded
from ..core.random_vibration import linear
from ..core.variability import (
    Ring,
    Scatter,
    amplification,
    amplification_curve,
    forced,
    monte_carlo,
    natural_hz,
    participation,
    ring_modes,
    sensitivity_sigma,
    sweep_grid,
    tuned_band,
    whitehead,
)
from . import settle_plots
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

TAB_HISTOGRAM, TAB_MODES, TAB_FORCED = range(3)
TWO_PI = 2.0 * math.pi
LEVELS = np.array([0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
SEEDS = 12  # mistuning patterns per level


@dataclass(frozen=True)
class VariabilityTrial:
    name: str
    note: str
    mass: float = 0.0  # CoV, %
    stiffness: float = 5.0
    damping: float = 0.0
    blades: int = 12
    coupling: float = 5.0  # k_c / k, %
    mistuning: float = 0.0  # %
    engine_order: int = 3
    mode: int = 1
    tab: int = TAB_HISTOGRAM
    count: int = 200
    masses: tuple[float, ...] = ()  # a chain element by element (none: the one there is)
    springs: tuple[float, ...] = ()
    dampers: tuple[float, ...] = ()

    def system(self) -> ChainSystem | None:
        if not self.masses:
            return None
        return ChainSystem(np.array(self.masses), np.array(self.springs), np.array(self.dampers))


TRIALS = (
    VariabilityTrial(
        "5 % scatter in the springs",
        "Two hundred copies of the uniform chain, each spring scattered by 5 % (lognormal). Each frequency "
        "scatters by about 1.4 % (mode 1: σ = 0.015 Hz on 1.105 Hz), and the first-order estimate from the "
        "modes' strain energies, σ<sub>f</sub>/f = σ<sub>λ</sub>/2λ, gives 1.44 % for every mode. The receptance's band is narrow "
        "between the peaks and wide at them: a sharp peak moves sideways, so near it the response can be several "
        "times the nominal one.",
    ),
    VariabilityTrial(
        "Masses scatter too",
        "5 % on the masses as well as the springs: the frequencies scatter by about 2 % (mode 1: σ = 0.023 Hz). "
        "Independent sources add in quadrature: √2 times the springs' alone.",
        mass=5.0,
    ),
    VariabilityTrial(
        "Damping scatter only",
        "30 % scatter on the dampers and none on the rest: the frequencies don't move at all, but the peaks "
        "do, by about the damping's scatter. The band only opens at the resonances.",
        stiffness=0.0,
        damping=30.0,
    ),
    VariabilityTrial(
        "A tuned bladed disc",
        "Twelve identical blades, each on a spring to the disc, joined by springs of 5 % of theirs round the "
        "ring (Modes tab). Its modes are waves round the ring: 0 to 6 nodal diameters, all but the first and "
        "last in pairs, in a band from 15.9 to 17.4 Hz. Every mode moves 6 to 12 blades.",
        tab=TAB_MODES,
    ),
    VariabilityTrial(
        "Weak coupling: 1 % mistuning localizes",
        "The coupling cut to 0.5 % and each blade's stiffness scattered by 1 %: the scatter is now twice the "
        "coupling, the waves can't get past the blades that differ, and the lowest mode moves 1.3 blades: "
        "it is one blade's. A mode localized on a few blades puts all the energy there.",
        coupling=0.5,
        mistuning=1.0,
        tab=TAB_MODES,
    ),
    VariabilityTrial(
        "Strong coupling resists",
        "The same 1 % mistuning with a 20 % coupling: every mode still spreads over 6 to 12 blades. What "
        "matters is the mistuning over the coupling, not the mistuning alone.",
        coupling=20.0,
        mistuning=1.0,
        tab=TAB_MODES,
    ),
    VariabilityTrial(
        "Forced response: one blade takes more",
        "Engine order 3 on the ring with a 2 % coupling (Forced tab). Tuned, every blade answers alike. With "
        "this 1 % pattern the worst blade peaks only 1.06 times the tuned one, but over 12 patterns the worst "
        "reaches 1.34 at 1 % and 1.42 at 2 %, where the mistuning matches the coupling: still below "
        "Whitehead's bound (1 + √12)/2 = 2.23. More mistuning helps again (1.28 at 10 %): the blades "
        "decouple.",
        coupling=2.0,
        mistuning=1.0,
        tab=TAB_FORCED,
    ),
)

# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.
BLADE_HZ = Ring().blade_hz  # every ring's blade: 1 kg on 10 kN/m


def _uniform(n: int) -> dict:
    s = ChainSystem.uniform(n)
    return dict(
        masses=tuple(map(float, s.masses)), springs=tuple(map(float, s.stiffness)), dampers=tuple(map(float, s.damping))
    )


def _one_mass(rng: np.random.Generator) -> Question:
    cm, ck = pick(rng, (0, 2, 3, 5)), pick(rng, (2, 4, 5, 10))
    spread = 0.5 * math.hypot(cm, ck)
    return Question(
        f"A single mass on a spring: the mass scatters by {cm:g} % and the spring by {ck:g} % (coefficients of "
        "variation, independent). By how many percent does its natural frequency scatter, to first order?",
        spread,
        f"f ∝ √(k/m), so δf/f = ½(δk/k − δm/m): each source's share is half its own scatter, and independent ones add "
        f"in quadrature: ½√({ck:g}² + {cm:g}²) = {number(spread)} %.",
        unit="%",
        setup=dict(masses=(1.0,), springs=(400.0,), dampers=(2.0,), mass=cm, stiffness=ck),
    )


def _shares(rng: np.random.Generator) -> Question:
    n = int(pick(rng, (3, 4, 5)))
    r = int(rng.integers(n))
    c = pick(rng, (2, 5, 10))
    system = ChainSystem.uniform(n)
    f, phi = natural_hz(system)
    dphi = np.diff(phi, prepend=0.0, axis=0)
    shares = [rounded(float(v), 3) for v in dphi[:, r] ** 2 * system.stiffness / (TWO_PI * f[r]) ** 2]
    spread = 0.5 * c * math.sqrt(sum(v * v for v in shares))
    listed = ", ".join(number(v) for v in shares)
    return Question(
        f"In one mode of a chain, its {n} springs carry these shares of the strain energy: {listed}. Every spring "
        f"scatters by {c:g} %, independently. By how many percent does the mode's frequency scatter, to first order?",
        spread,
        f"A spring's δk moves λ = ω² by its share s<sub>i</sub> of λ, so σ<sub>λ</sub>/λ = c √Σs<sub>i</sub>² = "
        f"{c:g} × √{number(sum(v * v for v in shares))} and σ<sub>f</sub>/f = σ<sub>λ</sub>/2λ = {number(spread)} %. "
        "Less than the springs' own: the mode averages over them.",
        unit="%",
        setup=dict(**_uniform(n), stiffness=c),
    )


def _percentile(rng: np.random.Generator) -> Question:
    c = pick(rng, (5, 10, 20, 30))
    s2 = math.log(1 + (c / 100) ** 2)
    top = 100 * (math.exp(-s2 / 2 + 1.645 * math.sqrt(s2)) - 1)
    return Question(
        f"A spring's stiffness is lognormal with mean at its nominal value and a {c:g} % coefficient of variation. "
        "How far above nominal is its 95th percentile?",
        top,
        f"ln k is normal with σ² = ln(1 + {c / 100:g}²) = {number(s2)} and mean −σ²/2 (so the mean of k is "
        f"nominal): the 95th percentile is exp(−σ²/2 + 1.645σ) = {number(1 + top / 100, 4)}, {number(top)} % above.",
        unit="%",
    )


def _whitehead(rng: np.random.Generator) -> Question:
    n = int(pick(rng, (8, 12, 16, 24, 36, 48)))
    bound = (1 + math.sqrt(n)) / 2
    return Question(
        f"A bladed disc has {n} blades, lightly damped, driven by one engine order. At most how many times the tuned "
        "response can its worst blade see once mistuned?",
        bound,
        f"Whitehead's bound: (1 + √N)/2 = (1 + √{n})/2 = {number(bound)}.",
        tolerance=0.01,
        setup=dict(blades=n, coupling=2.0, mistuning=1.0, tab=TAB_FORCED),
    )


def _band(rng: np.random.Generator) -> Question:
    c = pick(rng, (1, 2, 5, 10, 20))
    fb = rounded(BLADE_HZ)
    top = fb * math.sqrt(1 + 4 * c / 100)
    return Question(
        f"A tuned ring of blades, each alone at {number(fb)} Hz, joined to its neighbours by springs of {c:g} % of "
        "each blade's own. Where is the top of its band of modes?",
        top,
        f"The 0-nodal-diameter mode leaves the coupling springs unstretched, at the blade's {number(fb)} Hz; the "
        f"highest, neighbours in opposition, stretches each by twice the motion: f√(1 + 4k<sub>c</sub>/k) = "
        f"{number(fb)} × √{number(1 + 4 * c / 100)} = {number(top)} Hz.",
        unit="Hz",
        setup=dict(coupling=c, tab=TAB_MODES),
    )


def _participation(rng: np.random.Generator) -> Question:
    shape = [1.0, *(pick(rng, (-0.8, -0.5, -0.2, 0.2, 0.5, 0.8)) for _ in range(3))]
    shape[int(rng.integers(1, 4))] = pick(rng, (0.0, 0.1))
    p2 = sum(v * v for v in shape)
    p4 = sum(v**4 for v in shape)
    value = float(participation(np.array(shape)[:, None])[0])
    return Question(
        f"A mode of a ring of four blades has amplitudes {', '.join(number(v) for v in shape)}. What is its "
        "participation number?",
        value,
        f"(Σφ²)²/Σφ⁴ = {number(p2)}²/{number(p4)} = {number(value)} blades: 4 if all moved alike, 1 if only one did.",
        unit="blades",
    )


def _draws(rng: np.random.Generator) -> Question:
    e = pick(rng, (2, 5, 10))
    count = 1 + 1 / (2 * (e / 100) ** 2)
    return Question(
        f"Monte Carlo estimates a frequency's standard deviation from N scattered chains. About how many chains does "
        f"it take for that estimate's own scatter to be {e:g} %?",
        count,
        f"A sample standard deviation from N normal draws scatters by about 1/√(2(N − 1)): N = 1 + 1/(2 × "
        f"{e / 100:g}²) = {number(count)}.",
        unit="chains",
        tolerance=0.03,
    )


def _damping_only(rng: np.random.Generator) -> Question:
    return Question(
        "Only the dampers scatter, by 30 %; the masses and springs are exact. What do the natural frequencies do?",
        0,
        "They don't move at all: the undamped frequencies depend only on M and K. The peaks' heights scatter, by "
        "about the damping's scatter, and the band opens only at the resonances.",
        choices=("They don't move", "They scatter by about 15 %", "They scatter by about 30 %", "They all drop"),
        setup=dict(**_uniform(4), stiffness=0.0, damping=30.0),
        predict=True,
    )


def _weak_coupling(rng: np.random.Generator) -> Question:
    return Question(
        "Twelve blades with 1 % mistuning, joined by springs of only 0.5 % of their own. How many blades does the "
        "lowest mode move?",
        0,
        "About one (1.3): the mistuning is twice the coupling, the waves can't get past the blades that differ, and "
        "the mode collapses onto one blade. Tuned, it moves all twelve.",
        choices=("About one", "About half of them", "All twelve"),
        setup=dict(coupling=0.5, mistuning=1.0, tab=TAB_MODES),
        predict=True,
    )


def _strong_coupling(rng: np.random.Generator) -> Question:
    return Question(
        "The same 1 % mistuning on twelve blades, but the coupling raised to 20 %. How many blades does the lowest "
        "mode move?",
        0,
        "Nearly all twelve (12.0), as tuned: what localizes a mode is the mistuning over the coupling (here 0.05), "
        "not the mistuning alone.",
        choices=("Nearly all twelve", "About half of them", "About one"),
        setup=dict(coupling=20.0, mistuning=1.0, tab=TAB_MODES),
        predict=True,
    )


def _spread(blades: int, coupling: float, mistuning: float) -> float:
    """The ring's lowest mode's participation number (couplings and mistuning in %)."""
    return float(participation(ring_modes(Ring(blades, coupling=coupling / 100, mistuning=mistuning / 100))[1])[0])


def _ring_spec(rng: np.random.Generator) -> Question:
    blades = int(pick(rng, (8, 12, 16)))
    mistuning = pick(rng, (1, 2, 5))
    ratio = pick(rng, (0.3, 0.4, 0.5))
    coupling = round(mistuning / ratio, 2)  # %, the design
    least = math.floor(10 * _spread(blades, 0.85 * coupling, mistuning)) / 10
    most = rounded(1.25 * coupling, 2)

    def goal(t: VariabilityTrial):
        return (
            kept("Blades", t.blades, blades),
            kept("Mistuning", t.mistuning, mistuning, "%"),
            at_least("Blades the lowest mode moves", _spread(int(t.blades), t.coupling, t.mistuning), least),
            at_most("Coupling", t.coupling, most, "%"),
        )

    return Question(
        f"A ring of {blades} blades with {mistuning:g} % mistuning (the page's pattern). Choose the coupling between "
        f"them so that its lowest mode spreads over at least {number(least, 2)} blades, with the coupling springs no "
        f"more than {number(most)} % of the blades' own.",
        0,
        f"Localization is set by the mistuning over the coupling: the lowest mode moves {number(_spread(blades, coupling, mistuning))} "
        f"blades at a coupling of {number(coupling)} % (ratio {ratio:g}), and fewer as the ratio grows: about "
        f"{number(0.85 * coupling)} % is the least that does it.",
        setup=dict(blades=blades, coupling=0.5, mistuning=mistuning, tab=TAB_MODES),
        goal=goal,
        design=dict(blades=blades, coupling=coupling, mistuning=mistuning, tab=TAB_MODES),
    )


PROBLEMS = tuple(
    Problem(key, title, "variability_page", f"See {page_link('variability_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("one-mass", "One mass's scatter", "scatter", "No two structures are alike", _one_mass),
        ("shares", "A mode's scatter", "scatter", "No two structures are alike", _shares),
        ("percentile", "A lognormal's percentile", "scatter", "No two structures are alike", _percentile),
        ("draws", "How many chains", "scatter", "No two structures are alike", _draws),
        ("band", "A ring's band of modes", "localization", "Mode localization", _band),
        ("participation", "A participation number", "localization", "Mode localization", _participation),
        ("whitehead", "Whitehead's bound", "mistuning", "Forced response of a mistuned disc", _whitehead),
        ("damping-only", "Predict: damping scatter", "scatter", "No two structures are alike", _damping_only),
        ("weak-coupling", "Predict: weak coupling", "localization", "Mode localization", _weak_coupling),
        ("strong-coupling", "Predict: strong coupling", "localization", "Mode localization", _strong_coupling),
        ("ring-spec", "Spec: a coupling that spreads", "localization", "Mode localization", _ring_spec),
    )
)

TIPS = {
    "dof": "<p>The chain's number of masses: the Simulation page's chain is the nominal structure.</p>",
    "cov": "<p>The coefficient of variation (σ over the mean) of each mass, spring or damper, drawn "
    "independently and lognormal.</p>",
    "count": "<p>How many scattered chains to draw.</p>",
    "blades": "<p>The number of blades round the disc.</p>",
    "coupling": "<p>The springs between neighbouring blades, as a share of each blade's own.</p>",
    "mistuning": "<p>The scatter (σ) of the blades' stiffnesses; the pattern is seeded, its mean removed.</p>",
    "eo": "<p>The engine order E: the force on blade j has phase 2πEj/N, a wave E times round the disc.</p>",
    "mode": "<p>Which of the ring's modes to draw, from the lowest.</p>",
}
FRF_TIP = (
    "<p>The receptance at the last mass from a force there: nominal (bold), the 5–95 % band of the "
    "scattered chains (shaded), and a few of them.</p>"
)
HIST_TIP = "<p>Each mode's frequency deviation in the scattered chains, with the first-order normal curve.</p>"
MODES_TIP = (
    "<p>Top: the chosen mode's blade amplitudes, tuned (outline) and mistuned (filled). Bottom: how many "
    "blades each mode moves (participation number).</p>"
)
FORCED_TIP = (
    "<p>Top: each blade's amplitude under the engine-order force, mistuned (thin) and tuned (dashed). "
    "Bottom: the worst blade's peak over the tuned one against the mistuning, mean and worst of the "
    "patterns, and Whitehead's bound.</p>"
)

NOTES_HTML = load_notes("variability", trials=trials_html(TRIALS))


def _fields(t: VariabilityTrial) -> dict:
    """A trial's settings, without its name and note."""
    return {k: v for k, v in t.__dict__.items() if k not in ("name", "note")}


class VariabilityPage(QtWidgets.QWidget):
    """Controls on the left, the chain's band in the centre, histograms and the ring on the right."""

    dof_requested = QtCore.Signal(int)
    system_requested = QtCore.Signal(object)
    edit_parameters = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system = ChainSystem.uniform(4)
        self._dirty = True

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Each sets up the scatter or the ring to show one idea from the Theory tab.</p>",
        )
        self.trials.picked.connect(self.load_trial)

        ch = Form("Chain")
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.valueChanged.connect(self.dof_requested)
        ch.add("Number of masses:", self.dof, TIPS["dof"])
        link = QtWidgets.QLabel(
            "Its nominal masses, springs and dampers are edited on the <a href='#sim'>Simulation page</a>."
        )
        link.setWordWrap(True)
        mute(link)
        link.linkActivated.connect(self.edit_parameters)
        ch.row(link)

        sc = Form("Scatter (CoV)")
        self.cov_m = sc.number("Masses:", 0.0, 30.0, 0.0, 1, " %", TIPS["cov"])
        self.cov_k = sc.number("Springs:", 0.0, 30.0, 5.0, 1, " %", TIPS["cov"])
        self.cov_c = sc.number("Dampers:", 0.0, 100.0, 0.0, 1, " %", TIPS["cov"])
        self.count = sc.number("Chains drawn:", 20, 2000, 200, 0, "", TIPS["count"])

        ring = Form("Ring of blades")
        self.blades = ring.number("Blades:", 2, 48, 12, 0, "", TIPS["blades"])
        self.coupling = ring.number("Coupling:", 0.05, 100.0, 5.0, 2, " %", TIPS["coupling"])
        self.mistuning = ring.number("Mistuning:", 0.0, 20.0, 0.0, 2, " %", TIPS["mistuning"])
        self.eo = ring.number("Engine order:", 0, 24, 3, 0, "", TIPS["eo"])
        self.mode = ring.number("Mode:", 1, 48, 1, 0, "", TIPS["mode"])

        connect_all((self.cov_m, self.cov_k, self.cov_c, self.count), self._on_scatter)
        connect_all((self.blades, self.coupling, self.mistuning, self.eo, self.mode), self._on_ring)

        # --- centre
        self.frf_plot = plot("|Receptance|", "m/N", "Frequency", "Hz", log_x=True, log_y=True, height=200)
        self.frf_plot.setToolTip(FRF_TIP)
        add_legend(self.frf_plot.getPlotItem(), offset=(-10, 10))
        self.band_low, self.band_high = pg.PlotDataItem(), pg.PlotDataItem()
        self.band_fill = pg.FillBetweenItem(self.band_low, self.band_high)
        self.sample_items = [pg.PlotDataItem() for _ in range(5)]
        self.nominal_item = pg.PlotDataItem(name="nominal")
        for item in (self.band_fill, self.band_low, self.band_high, *self.sample_items, self.nominal_item):
            self.frf_plot.addItem(item)
        self.readout = readout_label()

        # --- right: histograms
        self.hist_plot = plot("Chains (density)", None, "Frequency deviation", "%", height=200)
        self.hist_plot.setToolTip(HIST_TIP)
        add_legend(self.hist_plot.getPlotItem(), offset=(-10, 10))
        self.hist_items: list[pg.PlotDataItem] = []
        self.hist_note = note_label()

        # --- right: ring modes
        self.shape_plot = plot("Blade amplitude", None, "Blade", None, height=150)
        self.shape_tuned = pg.BarGraphItem(x=[], height=[], width=0.8)
        self.shape_mis = pg.BarGraphItem(x=[], height=[], width=0.5)
        self.shape_plot.addItem(self.shape_tuned)
        self.shape_plot.addItem(self.shape_mis)
        self.pn_plot = plot("Blades moving", None, "Mode frequency", "Hz", height=150)
        add_legend(self.pn_plot.getPlotItem(), offset=(-10, 10))
        self.pn_tuned = marker(9, name="tuned")
        self.pn_mis = marker(9, "s", name="mistuned")
        self.pn_now = marker(15)
        for item in (self.pn_tuned, self.pn_mis, self.pn_now):
            self.pn_plot.addItem(item)
        for p in (self.shape_plot, self.pn_plot):
            p.setToolTip(MODES_TIP)
        self.modes_note = note_label()

        # --- right: forced ring
        self.sweep_plot = plot("Blade amplitude", "m/N", "Frequency", "Hz", height=150)
        self.sweep_items: list[pg.PlotDataItem] = []
        self.sweep_tuned = pg.PlotDataItem()
        self.sweep_plot.addItem(self.sweep_tuned)
        self.amp_plot = plot("Worst blade / tuned", None, "Mistuning", "%", log_x=True, height=150)
        add_legend(self.amp_plot.getPlotItem(), offset=(10, 10))
        self.amp_mean = pg.PlotDataItem(name="mean", symbol="o", symbolSize=5)
        self.amp_top = pg.PlotDataItem(name="worst pattern", symbol="t", symbolSize=6)
        self.amp_bound = pg.InfiniteLine(angle=0)
        self.amp_now = pg.InfiniteLine(angle=90)
        for item in (self.amp_bound, self.amp_now, self.amp_mean, self.amp_top):
            self.amp_plot.addItem(item)
        for p in (self.sweep_plot, self.amp_plot):
            p.setToolTip(FORCED_TIP)
        self.forced_note = note_label()
        self._curve_key: tuple[int, float, int] | None = None

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(column(self.hist_plot, self.hist_note, stretch=False), "Frequencies")
        self.tabs.addTab(column(self.shape_plot, self.pn_plot, self.modes_note, stretch=False), "Ring modes")
        self.tabs.addTab(column(self.sweep_plot, self.amp_plot, self.forced_note, stretch=False), "Forced ring")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._on_tab)

        three_panes(self, column(self.trials, ch, sc, ring), stack((self.frf_plot, 1), (self.readout, 0)), self.tabs)
        self.apply_theme()
        self.set_system(self.system)

    # ------------------------------------------------------------- inputs
    def set_system(self, system: ChainSystem, result=None) -> None:
        self.system = system
        set_quietly(self.dof, system.n)
        self._dirty = True
        if self.isVisible():
            self.redraw()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty:
            self.redraw()

    def load_trial(self, index: int) -> None:
        t = TRIALS[index]
        if index < 3:  # the chain trials start from the uniform chain
            t = VariabilityTrial(t.name, t.note, **{**_fields(t), **_uniform(4)})
        self.apply_trial(t)

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(VariabilityTrial("", "", **setup))

    def practice_state(self) -> VariabilityTrial:
        """The chain and the page's settings, as a trial: what a practice spec checks."""
        s = self.system
        return VariabilityTrial(
            "",
            "",
            mass=self.cov_m.value(),
            stiffness=self.cov_k.value(),
            damping=self.cov_c.value(),
            blades=int(self.blades.value()),
            coupling=self.coupling.value(),
            mistuning=self.mistuning.value(),
            engine_order=int(self.eo.value()),
            mode=int(self.mode.value()),
            tab=self.tabs.currentIndex(),
            count=int(self.count.value()),
            masses=tuple(float(v) for v in s.masses),
            springs=tuple(float(v) for v in s.stiffness),
            dampers=tuple(float(v) for v in s.damping),
        )

    def apply_trial(self, t: VariabilityTrial) -> None:
        system = t.system()
        if system is not None:
            self.system_requested.emit(system)
            s = self.system
            if s.n != system.n or not all(
                np.allclose(a, b)
                for a, b in ((s.masses, system.masses), (s.stiffness, system.stiffness), (s.damping, system.damping))
            ):
                self.set_system(system)  # nobody passed it back: show it here alone
        self.mode.setMaximum(int(t.blades))
        self.eo.setMaximum(int(t.blades) // 2)
        for box, value in (
            (self.cov_m, t.mass),
            (self.cov_k, t.stiffness),
            (self.cov_c, t.damping),
            (self.count, t.count),
            (self.blades, t.blades),
            (self.coupling, t.coupling),
            (self.mistuning, t.mistuning),
            (self.eo, t.engine_order),
            (self.mode, t.mode),
        ):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self._dirty = True
        if self.isVisible():
            self.redraw()

    @property
    def scatter(self) -> Scatter:
        return Scatter(self.cov_m.value() / 100, self.cov_k.value() / 100, self.cov_c.value() / 100)

    @property
    def ring(self) -> Ring:
        return Ring(
            int(self.blades.value()), coupling=self.coupling.value() / 100, mistuning=self.mistuning.value() / 100
        )

    def _on_tab(self, _index: int) -> None:
        if not self.isVisible():
            self._dirty = True  # drawn when shown
        elif self._dirty:
            self.redraw()
        else:
            self._draw_tab()

    def _on_scatter(self) -> None:
        self._draw_chain()
        if self.tabs.currentIndex() == TAB_HISTOGRAM:
            self._draw_histogram()
        settle_plots(self)

    def _on_ring(self) -> None:
        self.mode.setMaximum(int(self.blades.value()))
        self.eo.setMaximum(int(self.blades.value()) // 2)
        self._draw_tab()
        settle_plots(self)

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.nominal_item.setPen(pg.mkPen(c.strong, width=2.5))
        for item in (self.band_low, self.band_high):
            item.setPen(pg.mkPen(c.mass[0], width=1))
        fill = pg.mkColor(c.mass[0])
        fill.setAlpha(60)
        self.band_fill.setBrush(pg.mkBrush(fill))
        for item in self.sample_items:
            item.setPen(pg.mkPen(c.grey, width=1))
        self.shape_tuned.setOpts(pen=pg.mkPen(c.grey, width=1.5), brush=pg.mkBrush(0, 0, 0, 0))
        self.shape_mis.setOpts(brush=pg.mkBrush(c.mass[0]), pen=None)
        mark(self.pn_tuned, c.grey)
        mark(self.pn_mis, c.mass[0], filled=True)
        mark(self.pn_now, c.force)
        self.sweep_tuned.setPen(pg.mkPen(c.strong, width=2.5, style=QtCore.Qt.PenStyle.DashLine))
        for item, color in ((self.amp_mean, c.mass[0]), (self.amp_top, c.force)):
            item.setPen(pg.mkPen(color, width=2))
            item.setSymbolBrush(pg.mkBrush(color))
            item.setSymbolPen(None)
        self.amp_bound.setPen(pg.mkPen(c.grey, width=1.5, style=QtCore.Qt.PenStyle.DotLine))
        self.amp_now.setPen(pg.mkPen(c.fit, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self._dirty = True
        if self.isVisible():
            self.redraw()

    def redraw(self) -> None:
        self._dirty = False
        self._draw_chain()
        self._draw_tab()
        settle_plots(self)

    def _draw_chain(self) -> None:
        s = self.system
        n = s.n
        f0 = natural_hz(linear(s))[0]
        fpos = f0[f0 > 1e-6]
        lo = 0.3 * float(fpos[0]) if fpos.size else 0.1
        hi = 1.8 * float(f0[-1]) if f0[-1] > 0 else 10.0
        f = np.geomspace(lo, hi, 900)
        self.mc = mc = monte_carlo(s, self.scatter, int(self.count.value()), f, n - 1, n - 1)
        low, high = mc.band()
        self.band_low.setData(f, low)
        self.band_high.setData(f, high)
        for i, item in enumerate(self.sample_items):
            item.setData(f, mc.frf[i] if i < mc.frf.shape[0] else np.full(f.size, np.nan))
        self.nominal_item.setData(f, mc.nominal_frf)
        self.frf_plot.setXRange(np.log10(lo), np.log10(hi), padding=0)
        sigma = sensitivity_sigma(s, self.scatter)
        rows = "".join(
            f"<tr><td>{r + 1}</td><td align='right'>{mc.nominal_freqs[r]:.4g}</td>"
            f"<td align='right'>{mc.freqs[:, r].mean():.4g}</td><td align='right'>{mc.freqs[:, r].std():.3g}</td>"
            f"<td align='right'>{sigma[r]:.3g}</td>"
            f"<td align='right'>{100 * mc.freqs[:, r].std() / mc.nominal_freqs[r]:.3g} %</td></tr>"
            for r in range(n)
        )
        peak = int(np.argmax(mc.nominal_frf))
        self.readout.setText(
            "<table cellspacing='5'><tr><th>mode</th><th>nominal (Hz)</th><th>mean</th><th>σ (Monte Carlo)</th>"
            f"<th>σ (first order)</th><th>σ/f</th></tr>{rows}</table>"
            f"At the nominal peak ({f[peak]:.3g} Hz) the band spans {low[peak] / mc.nominal_frf[peak]:.2g} to "
            f"{high[peak] / mc.nominal_frf[peak]:.2g} times the nominal receptance; {mc.freqs.shape[0]} chains."
        )

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_HISTOGRAM:
            self._draw_histogram()
        elif i == TAB_MODES:
            self._draw_modes()
        elif i == TAB_FORCED:
            self._draw_forced()

    def _draw_histogram(self) -> None:
        mc = self.mc
        for item in self.hist_items:
            self.hist_plot.removeItem(item)
        self.hist_items = []
        legend = self.hist_plot.getPlotItem().legend
        legend.clear()
        dev = 100 * (mc.freqs / mc.nominal_freqs - 1)
        sigma = 100 * sensitivity_sigma(self.system, self.scatter) / mc.nominal_freqs
        span = max(float(np.max(np.abs(dev))), 1e-3) * 1.1
        edges = np.linspace(-span, span, 41)
        x = np.linspace(-span, span, 300)
        for r in range(min(mc.freqs.shape[1], 4)):
            color = colors.mode[r % len(colors.mode)]
            hist, _ = np.histogram(dev[:, r], edges, density=True)
            item = pg.PlotDataItem(edges, hist, stepMode="center", pen=pg.mkPen(color, width=1.5))
            self.hist_plot.addItem(item)
            legend.addItem(item, f"mode {r + 1}")
            self.hist_items.append(item)
            if sigma[r] > 0:
                pdf = np.exp(-0.5 * (x / sigma[r]) ** 2) / (sigma[r] * np.sqrt(2 * np.pi))
                curve = pg.PlotDataItem(x, pdf, pen=pg.mkPen(color, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
                self.hist_plot.addItem(curve)
                self.hist_items.append(curve)
        self.hist_plot.setXRange(-span, span, padding=0)
        self.hist_plot.enableAutoRange(axis="y")
        self.hist_note.setText(
            "Solid: the scattered chains' frequency deviations from nominal; dashed: the "
            "first-order normal curve. A mode with σ = 0 doesn't move."
        )

    def _draw_modes(self) -> None:
        ring = self.ring
        n = ring.blades
        mode = min(int(self.mode.value()), n) - 1
        ft, pt = ring_modes(ring, np.zeros(n))
        fm, pm = ring_modes(ring)
        x = np.arange(1, n + 1)
        shape_t = pt[:, mode] / np.max(np.abs(pt[:, mode]))
        shape_m = pm[:, mode] / np.max(np.abs(pm[:, mode]))
        self.shape_tuned.setOpts(x=x, height=shape_t)
        self.shape_mis.setOpts(x=x, height=shape_m)
        self.shape_plot.setXRange(0.4, n + 0.6, padding=0)
        self.shape_plot.setYRange(-1.1, 1.1, padding=0)
        nt, nm = participation(pt), participation(pm)
        self.pn_tuned.setData(ft, nt)
        self.pn_mis.setData(fm, nm)
        self.pn_now.setData([fm[mode]], [nm[mode]])
        self.pn_plot.setYRange(0, n + 0.5, padding=0)
        lo, hi = tuned_band(ring)
        self.modes_note.setText(
            f"Mode {mode + 1}: tuned {ft[mode]:.4g} Hz, moving {nt[mode]:.2g} blades; mistuned {fm[mode]:.4g} Hz, "
            f"moving {nm[mode]:.2g}. The tuned band runs {lo:.4g} to {hi:.4g} Hz; mistuning "
            f"{100 * ring.mistuning:.3g} % against a coupling of {100 * ring.coupling:.3g} % (ratio "
            f"{ring.mistuning / ring.coupling:.2g})."
        )

    def _draw_forced(self) -> None:
        ring = self.ring
        n = ring.blades
        E = int(self.eo.value())
        f = sweep_grid(ring, 900)
        x = forced(ring, E, f)
        tuned = forced(ring, E, f, np.zeros(n))
        for item in self.sweep_items:
            self.sweep_plot.removeItem(item)
        self.sweep_items = []
        pen = pg.mkPen(colors.mass[0], width=1)
        for j in range(n):
            item = pg.PlotDataItem(f, x[:, j], pen=pen)
            self.sweep_plot.addItem(item)
            self.sweep_items.append(item)
        self.sweep_tuned.setData(f, tuned.max(axis=1))
        self.sweep_plot.setXRange(float(f[0]), float(f[-1]), padding=0)
        key = (n, ring.coupling, E)
        if key != self._curve_key:
            self._curve = amplification_curve(ring, E, LEVELS[1:], SEEDS)
            self._curve_key = key
        mean, top = self._curve
        levels = 100 * LEVELS[1:]
        self.amp_mean.setData(levels, mean)
        self.amp_top.setData(levels, top)
        bound = whitehead(n)
        self.amp_bound.setPos(bound)
        self.amp_now.setPos(np.log10(max(100 * ring.mistuning, levels[0])))
        self.amp_plot.setYRange(0.9, bound + 0.15, padding=0)
        a = amplification(ring, E)
        self.forced_note.setText(
            f"This pattern: the worst blade peaks at {a:.3g} times the tuned ring's. Over {SEEDS} patterns per "
            f"level the worst reaches {float(top.max()):.3g} (at {levels[int(np.argmax(top))]:.3g} %); Whitehead's "
            f"bound for {n} blades is {bound:.3g} (dotted)."
        )
