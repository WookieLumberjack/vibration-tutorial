"""Test planning page: where to put a test's sensors, and what the chain does at one frequency (its ODS).

Left: the sensors (how many modes to target, how many sensors, how they're placed: effective independence,
EfI weighted by the driving-point residue, evenly spaced, or by hand) and the operating point (a force on one
mass at one frequency, and the reference sensor). Centre: the target modes with the sensors on them, and the
operational deflection shape beside the mode most like it. Right: EfI's eliminations step by step, the noise
and spatial aliasing each placement leaves, the ODS's spectrum and each mode's share of it, and the Theory.
The page follows the Simulation page's chain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.model import ChainSystem
from ..core.practice import Problem, Question, Requirement, at_least, at_most, number, pick
from ..core.test_planning import (
    METHODS,
    Method,
    Placement,
    Quality,
    at_phase,
    mac,
    ods,
    ods_parts,
    ods_spectrum,
    place,
    quality,
    shares,
    spectrum_grid,
    undamped_modes,
)
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

TAB_PLACEMENT, TAB_NOISE, TAB_ODS, TAB_THEORY = range(4)
UNIFORM = (1.0,) * 8
SPRINGS = (1000.0,) * 8
DAMPERS = (1.0,) * 8


def chain(masses=UNIFORM, stiffness=SPRINGS, damping=DAMPERS) -> ChainSystem:
    return ChainSystem(np.array(masses, dtype=float), np.array(stiffness, dtype=float), np.array(damping, dtype=float))


@dataclass(frozen=True)
class PlanningTrial:
    name: str
    note: str
    targets: int = 3
    sensors: int = 3
    method: Method = Method.EFI
    hand: tuple[int, ...] = ()  # masses, from 0, for Method.HAND
    force: int = 7  # the mass the operating force is on, from 0
    hz: float = 1.0
    reference: int = 7
    tab: int = TAB_PLACEMENT
    masses: tuple[float, ...] = UNIFORM
    stiffness: tuple[float, ...] = SPRINGS
    damping: tuple[float, ...] = DAMPERS

    def system(self) -> ChainSystem:
        return chain(self.masses, self.stiffness, self.damping)


TRIALS = (
    PlanningTrial(
        "Three sensors by effective independence",
        "Eight 1 kg masses on 1000 N/m springs, its first three modes (0.929, 2.755 and 4.487 Hz) the targets, "
        "and three sensors. EfI starts from all eight and drops, in turn, masses 1, 6, 4, 7 and 3, each the one "
        "that adds least to det(ΦᵀΦ), leaving masses 2, 5 and 8: 10.5 % of all eight's Fisher information. "
        "The modes' noise is 1.61, 1.55 and 1.29 times what eight sensors would give, and no two of them look "
        "alike at the sensors (the auto-MAC's largest off-diagonal is 0.041).",
        hz=0.929,
    ),
    PlanningTrial(
        "Evenly spaced isn't as good",
        "The same chain and targets with three sensors spread evenly, at masses 3, 5 and 8: 7.75 % of the Fisher "
        "information, and modes 2 and 3 alike enough at the sensors that their MAC is 0.188. Effective "
        "independence's 2, 5 and 8 differ by one sensor, moved one mass.",
        method=Method.EVEN,
        tab=TAB_NOISE,
        hz=0.929,
    ),
    PlanningTrial(
        "Sensors bunched at the ground: aliasing",
        "Three sensors on masses 1, 2 and 3, where every mode starts out the same way, as a sine's first quarter. "
        "Modes 1 and 2 look nearly alike there (MAC 0.976): the fit can hardly tell them apart, the Fisher "
        "information is 7.8 × 10⁻⁷ of eight sensors', and mode 1's noise is 240 times as large. That's spatial "
        "aliasing.",
        method=Method.HAND,
        hand=(0, 1, 2),
        tab=TAB_NOISE,
        hz=0.929,
    ),
    PlanningTrial(
        "One sensor more",
        "Effective independence with four sensors for the three targets keeps masses 2, 3, 5 and 8: the Fisher "
        "information nearly doubles, to 19.9 %, and mode 2's noise falls from 1.55 to 1.23 times eight "
        "sensors'. A spare sensor is cheap insurance against one that fails.",
        sensors=4,
        hz=0.929,
    ),
    PlanningTrial(
        "An ODS at a mode is the mode",
        "The chain driven at mode 2's frequency, 2.755 Hz, by a force on mass 8. With 1 N·s/m dampers the mode "
        "is lightly damped, and it is nearly the whole response: its share is 49.5 times the next largest, and "
        "the ODS's MAC with it is 0.999. That's why ODSs are measured at the peaks.",
        hz=2.755,
        tab=TAB_ODS,
    ),
    PlanningTrial(
        "Between two modes: a mix",
        "Halfway between modes 1 and 2, at 1.842 Hz, the ODS is neither: its MAC is 0.734 with mode 1 and 0.249 "
        "with mode 2 (the ODS tab's bars: mode 1's share is 0.86 of the ODS's largest, mode 2's 0.50, in "
        "opposite phase). An ODS is what the structure does, not one of its modes.",
        hz=1.842,
        tab=TAB_ODS,
    ),
    PlanningTrial(
        "Move the force",
        "The same 1.842 Hz, the force now on mass 4: the ODS changes (MAC 0.650 with mode 1, 0.331 with mode 2), "
        "and masses 1 to 4 hardly move, the rest swinging the other way. The modes don't depend on where the "
        "force is; an ODS does.",
        force=3,
        hz=1.842,
        tab=TAB_ODS,
    ),
    PlanningTrial(
        "A reference that hardly moves",
        "The same ODS measured against a reference sensor on mass 1, which moves only 3.2 % as much as mass 8. "
        "Every ratio X<sub>i</sub>/X<sub>ref</sub> is then up to 32 times larger, and in a real test the "
        "reference's noise is magnified by as much. Put the reference where the ODS moves most (mass 8 here).",
        force=3,
        hz=1.842,
        reference=0,
        tab=TAB_ODS,
    ),
)


# ------------------------------------------------------------------ practice problems
def _one_mode_share(rng: np.random.Generator) -> Question:
    shape = tuple(float(pick(rng, (0.2, 0.4, 0.5, 0.7, 0.8, 1.0))) for _ in range(3))
    i = int(rng.integers(3))
    e = float(shares(np.array(shape)[:, None])[i])
    return Question(
        f"A test targets one mode, whose shape at three candidate sensors is ({shape[0]:g}, {shape[1]:g}, "
        f"{shape[2]:g}). What is sensor {i + 1}'s effective independence E<sub>{i + 1}</sub>?",
        e,
        f"With one target, E<sub>i</sub> = φ<sub>i</sub>² / Σ φ<sub>j</sub>²: {shape[i]:g}² / "
        f"{number(sum(x * x for x in shape))} = {number(e)}. The shares add up to 1, the number of targets.",
    )


def _bunched(rng: np.random.Generator) -> Question:
    return Question(
        "Three sensors for a chain's first three modes are put on the three masses nearest the ground. What goes "
        "wrong?",
        0,
        "Near the ground every mode starts out the same way, so at those sensors the modes look alike (the "
        "auto-MAC's off-diagonal near 1): spatial aliasing. The fit can hardly tell them apart, and the noise "
        "in their estimates grows hugely (240 times for mode 1 on the eight-mass chain).",
        choices=(
            "The modes look alike at the sensors: the fit can't tell them apart",
            "Nothing: three sensors are enough for three modes",
            "The sensors overload",
            "Only the highest mode is lost",
        ),
        setup=dict(method=Method.HAND, hand=(0, 1, 2), tab=TAB_NOISE),
        predict=True,
    )


def _ods_at_mode(rng: np.random.Generator) -> Question:
    return Question(
        "A lightly damped chain is driven at exactly its second natural frequency. What does its operational "
        "deflection shape look like?",
        0,
        "Mode 2's shape: at its own frequency a lightly damped mode's response is 1/(2ζ) times its static share, "
        "so it outweighs every other mode by far (MAC 0.999 on the eight-mass chain). Between modes, or with "
        "heavy damping, the ODS is a mix.",
        choices=(
            "Mode 2's shape",
            "Mode 1's shape, the largest",
            "The static deflection under the force",
            "An even mix of all the modes",
        ),
        setup=dict(hz=2.755, tab=TAB_ODS),
        predict=True,
    )


def _reference(rng: np.random.Generator) -> Question:
    force = int(pick(rng, (3, 4, 7)))
    hz = float(pick(rng, (1.2, 1.842, 3.6)))
    system = chain()
    x = ods(system, force, hz)
    best = int(np.argmax(np.abs(x)))
    return Question(
        f"The eight-mass chain (1 kg, 1000 N/m, 1 N·s/m) runs at {hz:g} Hz with a force on mass {force + 1}. On "
        "which mass should the ODS's reference sensor go?",
        best + 1,
        f"Where the ODS moves most, mass {best + 1}: every other sensor is divided by the reference, so a "
        "reference that hardly moves magnifies its own noise into every ratio. (The page's centre plot shows the "
        "ODS: pick its largest entry.)",
        setup=dict(force=force, hz=hz, tab=TAB_ODS),
    )


def _hand_spec(rng: np.random.Generator) -> Question:
    targets = 3
    floor = float(pick(rng, (8.0, 10.0)))

    def goal(t: PlanningTrial):
        system = t.system()
        _, phi = undamped_modes(system)
        sensors = np.array(sorted(set(t.hand)), dtype=int) if t.method is Method.HAND else np.zeros(0, dtype=int)
        q = quality(phi[:, :targets], sensors) if sensors.size else None
        return (
            Requirement("Placed by hand", "yes" if t.method is Method.HAND else "no", t.method is Method.HAND),
            Requirement("The eight-mass chain kept", "kept" if system.n == 8 else "changed", system.n == 8),
            at_most("Sensors", float(sensors.size), 3.0),
            at_least("Fisher information", 100 * q.information if q else 0.0, floor, "%"),
            at_most("Auto-MAC's largest off-diagonal", q.worst if q else 1.0, 0.1),
        )

    return Question(
        f"Place three sensors by hand on the eight-mass chain for its first {targets} modes, with at least {floor:g} "
        "% of all eight's Fisher information and no auto-MAC off-diagonal above 0.1.",
        0,
        "Spread them, with one at the free end, where every mode moves: masses 2, 5 and 8 (effective "
        "independence's pick) give 10.5 % and 0.041. Evenly spaced 3, 5 and 8 fall short: 7.75 % and 0.188.",
        setup=dict(method=Method.HAND, hand=(0, 1, 2), tab=TAB_NOISE),
        goal=goal,
        design=dict(method=Method.HAND, hand=(1, 4, 7), tab=TAB_NOISE),
    )


PROBLEMS = tuple(
    Problem(key, title, "planning_page", f"See {page_link('planning_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("one-mode-share", "EfI with one target", "efi", "Effective independence", _one_mode_share),
        ("bunched", "Predict: bunched sensors", "aliasing", "Noise and spatial aliasing", _bunched),
        ("ods-at-mode", "Predict: an ODS at a mode", "ods", "Operational deflection shapes", _ods_at_mode),
        ("reference", "Where the reference goes", "reference", "Measuring an ODS", _reference),
        ("hand-spec", "Spec: three sensors by hand", "efi", "Effective independence", _hand_spec),
    )
)

TIPS = {
    "dof": "<p>The number of masses: the same chain as the Simulation page, so changing it here changes it there "
    "too. Every mass is a candidate sensor position.</p>",
    "targets": "<p>How many of the lowest modes the test is to identify.</p>",
    "sensors": "<p>How many sensors: at least one per target, or the modes can't be told apart.</p>",
    "method": "<p>Effective independence drops, one at a time, the candidate that adds least to the Fisher "
    "information det(ΦᵀΦ); weighted by the driving-point residue it favours where the response is large; or "
    "spread them evenly, or pick them by hand.</p>",
    "hand": "<p>The masses with a sensor, when they're placed by hand.</p>",
    "force": "<p>The mass the operating force acts on (the machine's unbalance, say).</p>",
    "hz": "<p>The operating frequency: the ODS is the chain's deflection there.</p>",
    "at_mode": "<p>Set the frequency to a natural frequency.</p>",
    "reference": "<p>The reference sensor an ODS is measured against: each ratio X<sub>i</sub>/X<sub>ref</sub>.</p>",
}
SHAPES_TIP = (
    "<p>The target modes (each scaled to largest 1) along the chain, and the sensors on them (the circles, and "
    "the grey lines). A mode with little movement at every sensor is hard to see.</p>"
)
ODS_TIP = (
    "<p>The operational deflection shape: the chain's deflection at the operating frequency under the force, at "
    "the instant it's largest (solid), and a quarter cycle later (dotted: zero for a real shape). Dashed: the "
    "mode most like it. The square is the reference sensor, the triangle the force.</p>"
)

NOTES_HTML = load_notes("test_planning", trials=trials_html(TRIALS))


def _table(head: list[str], rows: list[list[str]]) -> str:
    cell = "style='padding: 2px 8px; white-space: nowrap'"
    out = "<table cellspacing='0' style='border-collapse: collapse'><tr>"
    out += "".join(f"<th {cell}>{h}</th>" for h in head) + "</tr>"
    for row in rows:
        out += "<tr>" + "".join(f"<td align='center' {cell}>{c}</td>" for c in row) + "</tr>"
    return out + "</table>"


def _share(information: float) -> str:
    """A Fisher information share in %, a tiny one as a power of ten."""
    pct = 100 * information
    if pct >= 0.01 or pct == 0:
        return f"{pct:.3g} %"
    exponent = math.floor(math.log10(pct))
    return f"{pct / 10**exponent:.2g} × 10<sup>{exponent}</sup> %"


def _times(x: float) -> str:
    return "∞" if not np.isfinite(x) else f"×{x:.3g}"


class TestPlanningPage(QtWidgets.QWidget):
    """Controls on the left, the shapes in the centre, the methods' detail on the right."""

    __test__ = False  # pytest: not a test class
    dof_requested = QtCore.Signal(int)  # the user changed N here; the main window owns the system
    system_requested = QtCore.Signal(object)  # a trial's chain, for the Simulation page
    edit_parameters = QtCore.Signal()  # the user wants the Simulation page's parameter panel

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system = chain()
        self._dirty = True
        self.hz = np.zeros(0)
        self.phi = np.zeros((0, 0))
        self.placement: Placement | None = None
        self.quality: Quality | None = None
        self.raised = False  # EfI was asked for fewer sensors than targets, and given one a target

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Each loads a chain into the Simulation page and sets up the sensors or the operating point to "
            "show one idea from the Theory tab.</p>",
        )
        self.trials.picked.connect(self.load_trial)

        ch = Form("Chain")
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setToolTip(TIPS["dof"])
        self.dof.valueChanged.connect(self.dof_requested)
        ch.add("Number of masses:", self.dof, TIPS["dof"])
        link = QtWidgets.QLabel("Masses and springs are edited on the <a href='#sim'>Simulation page</a>.")
        link.setWordWrap(True)
        mute(link)
        link.linkActivated.connect(self.edit_parameters)
        ch.row(link)

        sensors = Form("Sensors")
        self.targets = sensors.number("Target modes:", 1, MAX_DOF, 3, 0, "", TIPS["targets"])
        self.count = sensors.number("Sensors:", 1, MAX_DOF, 3, 0, "", TIPS["sensors"])
        self.method = combo([m.value for m in METHODS], TIPS["method"])
        sensors.add("Placed by:", self.method, TIPS["method"])
        holder = QtWidgets.QWidget()
        flow = FlowLayout(holder)
        flow.setContentsMargins(0, 0, 0, 0)
        self.hand: list[QtWidgets.QCheckBox] = []
        for i in range(MAX_DOF):
            box = QtWidgets.QCheckBox(f"m{i + 1}")
            box.setToolTip(TIPS["hand"])
            self.hand.append(box)
            flow.add(box)
        self.hand_row = holder
        sensors.row(holder)
        self.sensors_form = sensors

        operating = Form("Operating point")
        self.force = operating.number("Force on mass:", 1, MAX_DOF, 8, 0, "", TIPS["force"])
        self.freq = operating.number("Frequency:", 0.001, 1e4, 1.0, 3, " Hz", TIPS["hz"])
        self.at_mode = combo(["Pick a mode…"], TIPS["at_mode"])
        operating.add("At mode:", self.at_mode, TIPS["at_mode"])
        self.reference = operating.number("Reference on mass:", 1, MAX_DOF, 8, 0, "", TIPS["reference"])
        self.at_mode.currentIndexChanged.connect(self._on_at_mode)

        connect_all((self.targets, self.count, self.method, *self.hand), self._on_sensors)
        connect_all((self.force, self.freq, self.reference), self._on_operating)

        # --- centre
        self.shapes_plot = plot("Mode (largest 1)", None, "Mass", None, height=160)
        self.shapes_plot.setToolTip(SHAPES_TIP)
        self.shapes_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.shapes_plot.getPlotItem(), offset=(10, 10))
        self.mode_items: list[pg.PlotDataItem] = []
        self.sensor_marks: list[pg.PlotDataItem] = []
        self.sensor_lines: list[pg.InfiniteLine] = []
        self.shapes_readout = readout_label()
        self.ods_plot = plot("ODS (largest 1)", None, "Mass", None, height=160)
        self.ods_plot.setToolTip(ODS_TIP)
        self.ods_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.ods_plot.getPlotItem(), offset=(10, 10))
        self.ods_item = pg.PlotDataItem(name="ODS", symbol="o", symbolSize=8)
        self.ods_quad = pg.PlotDataItem(name="a quarter cycle on")
        self.ods_mode = pg.PlotDataItem(name="the mode most like it")
        self.ods_ref = marker(16, "s")
        self.ods_force = marker(14, "t1")
        for item in (self.ods_mode, self.ods_quad, self.ods_item, self.ods_ref, self.ods_force):
            self.ods_plot.addItem(item)
        self.ods_readout = readout_label()

        # --- right
        self.steps_view = QtWidgets.QTextBrowser()
        self.steps_view.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        self.steps_note = note_label()
        self.noise_view = QtWidgets.QTextBrowser()
        self.noise_view.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        self.noise_note = note_label()
        self.spectrum_plot = plot("|X| per N", "m/N", "Frequency", "Hz", log_x=True, log_y=True, height=200)
        self.spectrum_plot.setMouseEnabled(x=False, y=False)
        self.spectrum_curves: list[pg.PlotDataItem] = []
        self.freq_line = pg.InfiniteLine(angle=90)
        self.spectrum_plot.addItem(self.freq_line, ignoreBounds=True)
        self.parts_plot = plot("Share (of the ODS's largest)", None, "Mode", None, height=160)
        self.parts_plot.setMouseEnabled(x=False, y=False)
        self.parts_bars = pg.BarGraphItem(x=[], height=[], width=0.6)
        self.parts_plot.addItem(self.parts_bars)
        self.parts_note = note_label()

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(column(self.steps_note, self.steps_view, stretch=False), "Placement")
        self.tabs.addTab(column(self.noise_note, self.noise_view, stretch=False), "Noise && aliasing")
        self.tabs.addTab(column(self.spectrum_plot, self.parts_plot, self.parts_note, stretch=False), "ODS")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._draw_tab)

        three_panes(
            self,
            column(self.trials, ch, sensors, operating),
            stack(
                (self.shapes_plot, 3),
                (self.shapes_readout, 0),
                (self.ods_plot, 3),
                (self.ods_readout, 0),
            ),
            self.tabs,
        )
        self.apply_theme()
        self.set_system(self.system)

    # ------------------------------------------------------------- inputs
    def set_system(self, system: ChainSystem, result=None) -> None:
        """The Simulation page's chain. Drawn now if the page shows, else when it does."""
        self.system = system
        set_quietly(self.dof, system.n)
        for box in (self.targets, self.count, self.force, self.reference):
            box.blockSignals(True)
            box.setMaximum(system.n)
            box.blockSignals(False)
        for i, check in enumerate(self.hand):
            check.setVisible(i < system.n)
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
        self.apply_trial(PlanningTrial("", "", **setup))

    def practice_state(self) -> PlanningTrial:
        """The page's settings and the chain it follows, as a trial: what a practice spec checks."""
        s = self.system
        return PlanningTrial(
            "",
            "",
            targets=int(self.targets.value()),
            sensors=int(self.count.value()),
            method=METHODS[self.method.currentIndex()],
            hand=tuple(i for i, box in enumerate(self.hand[: s.n]) if box.isChecked()),
            force=int(self.force.value()) - 1,
            hz=self.freq.value(),
            reference=int(self.reference.value()) - 1,
            tab=self.tabs.currentIndex(),
            masses=tuple(float(m) for m in s.masses),
            stiffness=tuple(float(k) for k in s.stiffness),
            damping=tuple(float(c) for c in s.damping),
        )

    def apply_trial(self, t: PlanningTrial) -> None:
        system = t.system()
        self.system_requested.emit(system)
        s = self.system
        if (
            s.n != system.n
            or not np.allclose(s.masses, system.masses)
            or not np.allclose(s.stiffness, system.stiffness)
            or not np.allclose(s.damping, system.damping)
        ):
            self.set_system(system)  # nobody passed it back: show it here alone
        for box, value in (
            (self.targets, t.targets),
            (self.count, t.sensors),
            (self.method, METHODS.index(t.method)),
            (self.force, t.force + 1),
            (self.freq, t.hz),
            (self.reference, t.reference + 1),
        ):
            set_quietly(box, value)
        for i, check in enumerate(self.hand):
            set_quietly(check, i in t.hand)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self.redraw()

    def _on_sensors(self) -> None:
        self._work_sensors()
        self._draw_shapes()
        self._draw_tab()
        settle_plots(self)

    def _on_operating(self) -> None:
        self._draw_ods()
        self._draw_tab()
        settle_plots(self)

    def _on_at_mode(self, index: int) -> None:
        if index <= 0:
            return
        set_quietly(self.at_mode, 0)
        self.freq.setValue(round(float(self.hz[index - 1]), 3))

    # ------------------------------------------------------------- working out
    def _work_sensors(self) -> None:
        n = self.system.n
        m = min(int(self.targets.value()), n)
        method = METHODS[self.method.currentIndex()]
        self.hand_row.setVisible(method is Method.HAND)
        self.count.setEnabled(method is not Method.HAND)
        keep = (
            max(min(int(self.count.value()), n), m)
            if method in (Method.EFI, Method.EFI_DPR)
            else int(self.count.value())
        )
        hand = tuple(i for i, box in enumerate(self.hand[:n]) if box.isChecked())
        T = self.phi[:, :m]
        self.placement = place(T, self.hz[:m], method, min(keep, n), hand)
        self.quality = quality(T, self.placement.sensors)
        self.raised = method in (Method.EFI, Method.EFI_DPR) and keep > int(self.count.value())

    def redraw(self) -> None:
        self._dirty = False
        self.hz, self.phi = undamped_modes(self.system)
        self.at_mode.blockSignals(True)
        self.at_mode.clear()
        self.at_mode.addItem("Pick a mode…")
        self.at_mode.addItems([f"Mode {r + 1} ({f:.4g} Hz)" for r, f in enumerate(self.hz)])
        self.at_mode.blockSignals(False)
        self._work_sensors()
        self._draw_shapes()
        self._draw_ods()
        self._draw_tab()
        settle_plots(self)

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        dash = QtCore.Qt.PenStyle.DashLine
        self.ods_item.setPen(pg.mkPen(colors.strong, width=2))
        self.ods_item.setSymbolBrush(pg.mkBrush(colors.strong))
        self.ods_item.setSymbolPen(None)
        self.ods_quad.setPen(pg.mkPen(colors.grey, width=1.5, style=QtCore.Qt.PenStyle.DotLine))
        self.ods_mode.setPen(pg.mkPen(colors.fit, width=2, style=dash))
        mark(self.ods_ref, colors.strong)
        mark(self.ods_force, colors.force, filled=True)
        self.freq_line.setPen(pg.mkPen(colors.force, width=1.5, style=dash))
        self.parts_bars.setOpts(brush=pg.mkBrush(colors.mode[0]), pen=None)
        if self.hz.size:
            self.redraw()

    def _draw_shapes(self) -> None:
        p = self.shapes_plot
        n = self.system.n
        idx = np.arange(n + 1)
        m = min(int(self.targets.value()), n)
        legend = p.getPlotItem().legend
        while len(self.mode_items) < MAX_DOF:
            item = pg.PlotDataItem()
            dots = marker(13)
            p.addItem(item)
            p.addItem(dots)
            self.mode_items.append(item)
            self.sensor_marks.append(dots)
        legend.clear()
        sensors = self.placement.sensors if self.placement is not None else np.zeros(0, dtype=int)
        for r, (item, dots) in enumerate(zip(self.mode_items, self.sensor_marks, strict=True)):
            if r < m:
                shape = self.phi[:, r] / np.max(np.abs(self.phi[:, r]))
                colour = colors.mode[r % len(colors.mode)]
                item.setData(idx, np.concatenate([[0.0], shape]))
                item.setPen(pg.mkPen(colour, width=2))
                dots.setData(sensors + 1, shape[sensors])
                mark(dots, colour)
                legend.addItem(item, f"mode {r + 1}, {self.hz[r]:.4g} Hz")
            else:
                item.setData([], [])
                dots.setData([], [])
        for line in self.sensor_lines:
            p.removeItem(line)
        self.sensor_lines = []
        for s in sensors:
            line = pg.InfiniteLine(pos=float(s + 1), angle=90, pen=pg.mkPen(colors.grey, width=1))
            p.addItem(line, ignoreBounds=True)
            self.sensor_lines.append(line)
        p.setXRange(0, n, padding=0.03)
        p.setYRange(-1.1, 1.1, padding=0)
        p.getAxis("bottom").setTicks([[(i, "ground" if i == 0 else f"m{i}") for i in idx]])
        q = self.quality
        if q is None or not sensors.size:
            self.shapes_readout.setText("No sensors: tick some masses.")
            return
        where = ", ".join(str(s + 1) for s in sensors)
        text = f"Sensors on mass{'es' if sensors.size > 1 else ''} {where}. "
        if self.raised:
            text += f"(EfI needs a sensor per target: {sensors.size}.) "
        if q.singular:
            text += (
                f"<b>Too few, or in the wrong places, for {m} target{'s' if m > 1 else ''}</b>: Φ<sub>s</sub>ᵀΦ<sub>s</sub> "
                "is singular, and the modes can't be told apart."
            )
        else:
            text += (
                f"Fisher information {_share(q.information)} of every mass's; the modes' noise "
                + ", ".join(_times(x) for x in q.noise)
                + f"; auto-MAC's largest off-diagonal {q.worst:.3f}."
            )
        self.shapes_readout.setText(text)

    def _draw_ods(self) -> None:
        s = self.system
        n = s.n
        idx = np.arange(n + 1)
        force = min(int(self.force.value()), n) - 1
        ref = min(int(self.reference.value()), n) - 1
        f = self.freq.value()
        x = ods(s, force, f)
        top = float(np.max(np.abs(x)))
        if top <= 0:
            return
        shape = at_phase(x)
        sign = 1.0 if shape[np.argmax(np.abs(shape))] >= 0 else -1.0
        turned = x * np.exp(-0.5j * np.angle(np.sum(x * x))) * sign
        scale = float(np.max(np.abs(turned.real))) or top
        real, quad = turned.real / scale, turned.imag / scale
        self.ods_item.setData(idx, np.concatenate([[0.0], real]))
        self.ods_quad.setData(idx, np.concatenate([[0.0], quad]))
        macs = np.array([mac(x, self.phi[:, r]) for r in range(n)])
        best = int(np.argmax(macs))
        phi = self.phi[:, best] * (1.0 if self.phi[:, best] @ real >= 0 else -1.0)
        self.ods_mode.setData(idx, np.concatenate([[0.0], phi / np.max(np.abs(phi))]))
        legend = self.ods_plot.getPlotItem().legend
        legend.clear()
        legend.addItem(self.ods_item, f"ODS at {f:.4g} Hz")
        legend.addItem(self.ods_quad, "a quarter cycle on")
        legend.addItem(self.ods_mode, f"mode {best + 1} (MAC {macs[best]:.3f})")
        self.ods_ref.setData([ref + 1], [real[ref]])
        self.ods_force.setData([force + 1], [real[force]])
        self.ods_plot.setXRange(0, n, padding=0.03)
        span = max(1.1, float(np.max(np.abs(quad))) * 1.1)
        self.ods_plot.setYRange(-span, span, padding=0)
        self.ods_plot.getAxis("bottom").setTicks([[(i, "ground" if i == 0 else f"m{i}") for i in idx]])
        others = sorted(range(n), key=lambda r: -macs[r])[:3]
        ratio = abs(x[ref]) / top
        text = (
            f"At {f:.4g} Hz, a force on mass {force + 1}: MAC "
            + ", ".join(f"{macs[r]:.3f} with mode {r + 1}" for r in others)
            + f". The reference, mass {ref + 1}, moves {100 * ratio:.3g} % as much as the most"
        )
        text += (
            f": every ratio X<sub>i</sub>/X<sub>ref</sub> is up to {1 / ratio:.3g} times its sensor's motion, its "
            "noise with it."
            if ratio < 0.2
            else "."
        )
        self.ods_readout.setText(text)

    def _draw_tab(self) -> None:
        if self._dirty:  # the modes aren't worked out for this chain yet (a trial turning to a tab): redraw will
            return
        i = self.tabs.currentIndex()
        if i == TAB_PLACEMENT:
            self._draw_steps()
        elif i == TAB_NOISE:
            self._draw_noise()
        elif i == TAB_ODS:
            self._draw_spectrum()

    def _draw_steps(self) -> None:
        p = self.placement
        n = self.system.n
        method = METHODS[self.method.currentIndex()]
        if p is None or not p.steps:
            why = {
                Method.EVEN: "Evenly spaced sensors have no steps.",
                Method.HAND: "Sensors placed by hand have no steps.",
            }.get(method, "Every mass has a sensor: nothing to drop.")
            self.steps_note.setText(why + " Pick effective independence to see it drop candidates one by one.")
            self.steps_view.setHtml("")
            return
        weighted = method is Method.EFI_DPR
        head = ["Step"] + [f"m{i + 1}" for i in range(n)] + ["drops"]
        rows = []
        for k, step in enumerate(p.steps):
            cells = [""] * n
            for c, v in zip(step.candidates, step.score, strict=True):
                text = f"{v:.3g}"
                cells[c] = f"<b style='color:{colors.force}'>{text}</b>" if c == step.dropped else text
            rows.append([str(k + 1), *cells, f"m{step.dropped + 1}"])
        kept = [""] * n
        for c, v in zip(p.sensors, p.final, strict=True):
            kept[c] = f"<b>{v:.3g}</b>"
        rows.append(["kept", *kept, ""])
        self.steps_view.setHtml(_table(head, rows))
        self.steps_note.setText(
            f"Each row: every candidate's E<sub>i</sub>{' × DPR<sub>i</sub>' if weighted else ''} (its share of "
            "the Fisher information), and the one dropped, in red. The shares are worked out again after each drop: "
            "losing one candidate raises the others'."
            + (" (The DPR weights are in s/kg: compare along a row.)" if weighted else "")
        )

    def _draw_noise(self) -> None:
        n = self.system.n
        m = min(int(self.targets.value()), n)
        keep = max(min(int(self.count.value()), n), m)
        T, hz = self.phi[:, :m], self.hz[:m]
        current = self.placement
        options = [("Here", current)]
        for name, method in (("EfI", Method.EFI), ("EfI × DPR", Method.EFI_DPR), ("Evenly", Method.EVEN)):
            options.append((name, place(T, hz, method, keep)))
        head = ["", *[name for name, _ in options]]
        scored = [(p, quality(T, p.sensors)) for _, p in options if p is not None]
        rows = [["Sensors", *[", ".join(f"m{s + 1}" for s in p.sensors) or "none" for p, _ in scored]]]
        rows.append(["Fisher information", *[_share(q.information) for _, q in scored]])
        for r in range(m):
            rows.append([f"Mode {r + 1}'s noise", *[_times(q.noise[r]) for _, q in scored]])
        rows.append(["Worst auto-MAC", *[f"{q.worst:.3f}" for _, q in scored]])
        html = _table(head, rows)
        q = self.quality
        if q is not None and m > 1:
            mac_rows = [[f"mode {r + 1}", *[f"{q.auto_mac[r, s]:.3f}" for s in range(m)]] for r in range(m)]
            html += "<p><b>Auto-MAC at this placement's sensors</b></p>"
            html += _table(["", *[f"mode {s + 1}" for s in range(m)]], mac_rows)
        self.noise_view.setHtml(html)
        self.noise_note.setText(
            f"This placement (Here) beside effective independence, EfI weighted by the driving-point residue, "
            f"and evenly spaced sensors, each with {keep} sensors for {m} target{'s' if m > 1 else ''}. Noise: "
            "each mode's against what a sensor on every mass would give (∞: the modes can't be told apart). Worst "
            "auto-MAC: the largest off-diagonal; near 1 is spatial aliasing."
        )

    def _draw_spectrum(self) -> None:
        s = self.system
        n = s.n
        force = min(int(self.force.value()), n) - 1
        f = self.freq.value()
        freqs = spectrum_grid(self.hz)
        X = np.abs(ods_spectrum(s, force, freqs))
        p = self.spectrum_plot
        while len(self.spectrum_curves) < MAX_DOF:
            curve = pg.PlotDataItem()
            p.addItem(curve)
            self.spectrum_curves.append(curve)
        for i, curve in enumerate(self.spectrum_curves):
            if i < n:
                curve.setData(freqs, np.maximum(X[:, i], 1e-300))
                curve.setPen(pg.mkPen(colors.mass[i % len(colors.mass)], width=1.5))
            else:
                curve.setData([], [])
        p.setXRange(math.log10(freqs[0]), math.log10(freqs[-1]), padding=0)
        self.freq_line.setPos(math.log10(f) if f > 0 else 0.0)
        parts = ods_parts(s, force, f)
        top = float(np.max(np.abs(ods(s, force, f))))
        size = np.max(np.abs(parts), axis=0) / top if top > 0 else np.zeros(n)
        self.parts_bars.setOpts(x=np.arange(1, n + 1), height=size, width=0.6)
        self.parts_plot.setXRange(0.4, n + 0.6, padding=0)
        self.parts_plot.getAxis("bottom").setTicks([[(r + 1, str(r + 1)) for r in range(n)]])
        lead = int(np.argmax(size))
        self.parts_note.setText(
            f"Above: every mass's response to a unit force on mass {force + 1} (a colour per mass), the operating "
            f"frequency dashed. Below: each mode's share of the ODS at {f:.4g} Hz, φ<sub>r</sub>η<sub>r</sub>'s "
            f"largest against the ODS's: mode {lead + 1} leads with {size[lead]:.3g}. They add up (with their "
            "phases) to the ODS exactly."
        )
