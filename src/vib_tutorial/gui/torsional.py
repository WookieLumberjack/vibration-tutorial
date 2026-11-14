"""Torsional vibration page: a drive train of discs, shafts and gear meshes, its torsional criticals and forced
response, and a motor starting it.

Left: the preset, the train (pick a disc or a link to edit it, or click it in the drawing),
the excitation, the speeds and the start-up's motor and load. Centre, two tabs: the
interference (Campbell) diagram, natural frequencies against the orders that excite them,
with the steady dynamic torque in a link against the motor's speed; and the start-up, the
shafts' torque and the motor's speed against time as it runs. Right: the train drawn, its
mode shapes, and the notes. The page has its own model (core/torsional.py and
core/torsional_startup.py), not the chain.

Working on it
-------------
The page is three parts: the drive-train editor and the excitation (`torsional_editor.TrainEditor`), which
hands the page each train it builds (`built`); the start-up, its controls and its tab, on its own frame loop
(`torsional_startup_view.StartUpView`), which the page gives each new train; and this page, which holds the
train and its modes and draws the interference diagram, the response, the drawing and the mode shapes. The
page keeps `start`, `running`, `pause_button` and `loop` for the main window and the tests (its one motion is
the start-up). `test_torsional_page` and `test_torsional_start_up` in `test_gui_smoke.py` check it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.practice import Problem, Question, Requirement, at_least, at_most, kept, number, pick, rounded
from ..core.torsional import (
    FAN,
    PRESETS,
    Disc,
    DriveTrain,
    Mesh,
    Shaft,
    TorsionalMode,
    TorsionalPreset,
    criticals,
    forced_response,
    modes,
    natural_hz,
    order_lines,
    response_speeds,
    separation,
    sources,
    two_inertia_hz,
)
from ..core.torsional_startup import (
    STARTUPS,
    MotorKind,
    crossings,
)
from . import torsional_figures
from .frame_loop import FrameLoop
from .layouts import ColumnScroll, FlowLayout, form_layout, form_row, group, plot_height
from .lesson import RunButton, plot
from .links import load_notes, page_link
from .maths import markup
from .notes_view import NotesBrowser
from .panels import spin
from .style import colors
from .theming import add_legend, mute
from .torsional_editor import TrainEditor
from .torsional_startup_view import StartUpView, fmt_torque

CUSTOM = "Custom"
SHAPES_SHOWN = 6  # modes drawn on the shape plot
MARGIN = 10.0  # % either side of the running speed a critical should stay out of (API 617's 10 %)
TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class TorsionalSetup:
    """A practice problem's setup: a preset to start from, the train instead of its own (None: the preset's), the
    speeds (None: the preset's) and whether the mode shapes show. The page's state comes back as one too."""

    preset: str = PRESETS[0].name
    train: DriveTrain | None = None
    operating_rpm: float | None = None
    top_rpm: float | None = None
    shapes: bool = False

    def drive_train(self) -> DriveTrain:
        return self.train if self.train is not None else _preset(self.preset).train

    def running_rpm(self) -> float:
        return self.operating_rpm if self.operating_rpm is not None else _preset(self.preset).operating_rpm


PRACTICE_SETUP = TorsionalSetup  # what a practice problem's setup and the page's practice_state are


def _preset(name: str) -> TorsionalPreset:
    return next((p for p in PRESETS if p.name == name), PRESETS[0])


# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.
FAN_PRESET, COMPRESSOR_PRESET = PRESETS[0].name, PRESETS[1].name


def _fan(j1: float, j2: float, k: float) -> DriveTrain:
    """The fan preset's train with its inertias and coupling changed."""
    t = FAN.with_disc(0, Disc("Motor", j1)).with_disc(1, Disc("Fan", j2, FAN.discs[1].damping))
    return t.with_link(0, Shaft(k, FAN.links[0].damping, "Coupling"))


def _two_discs(rng: np.random.Generator) -> Question:
    j1, j2 = pick(rng, (0.5, 1, 2, 4)), pick(rng, (5, 10, 20, 50))
    k = pick(rng, (20, 50, 100, 200, 500))  # kN·m/rad
    f = two_inertia_hz(j1, j2, 1000 * k)
    return Question(
        f"A {j1:g} kg·m² motor drives a {j2:g} kg·m² fan through a coupling of {k:g} kN·m/rad. At what frequency "
        "does the train twist?",
        f,
        f"Two discs on one shaft: f = √(k (J₁ + J₂) / J₁J₂) / 2π = √({k:g}×10³ × {j1 + j2:g} / {j1 * j2:g}) / 2π = "
        f"{number(TWO_PI * f)}/2π = {number(f)} Hz. (The train's other mode is the whole of it turning, at 0 Hz.)",
        unit="Hz",
        setup=dict(preset=FAN_PRESET, train=_fan(j1, j2, 1000 * k), shapes=True),
    )


def _shaft_stiffness(rng: np.random.Generator) -> Question:
    d, length = pick(rng, (30, 50, 80, 100)), pick(rng, (0.5, 1, 2))
    J = math.pi * (d / 1000) ** 4 / 32
    k = 80e9 * J / length
    return Question(
        f"A solid steel shaft (G = 80 GPa) {d:g} mm across and {length:g} m long. What is its torsional stiffness?",
        k / 1000,
        f"J = πd⁴/32 = {number(J * 1e8)}×10⁻⁸ m⁴, so k = GJ/L = 80×10⁹ × {number(J * 1e8)}×10⁻⁸ / {length:g} = "
        f"{number(k / 1000)} kN·m/rad. It goes as d⁴: twice as thick is 16 times as stiff.",
        unit="kN·m/rad",
    )


def _referred(rng: np.random.Generator) -> Question:
    z_in, z_out = int(pick(rng, (60, 90, 120))), int(pick(rng, (20, 30, 40)))
    j = pick(rng, (0.05, 0.1, 0.3, 0.5))
    ratio = z_in / z_out
    referred = j * ratio**2
    return Question(
        f"A compressor wheel of {j:g} kg·m² turns on a pinion of {z_out} teeth, driven by a bull gear of {z_in} on "
        "the motor's shaft. What inertia does it add, referred to the motor's shaft?",
        referred,
        f"The pinion turns {z_in}/{z_out} = {number(ratio)} times as fast, so its kinetic energy is that of "
        f"J n² on the motor's shaft: {j:g} × {number(ratio)}² = {number(referred)} kg·m². A small wheel at high speed "
        "weighs a lot on the motor.",
        unit="kg·m²",
        setup=dict(preset=COMPRESSOR_PRESET, shapes=True),
    )


def _critical(rng: np.random.Generator) -> Question:
    f, order = pick(rng, (12, 20, 35, 52.3, 97.3)), pick(rng, (1, 2, 3, 6))
    rpm = 60 * f / order
    return Question(
        f"A drive train's mode is at {f:g} Hz. At what speed does a {order:g}× excitation meet it?",
        rpm,
        f"The order's frequency is {order:g} × rpm/60; it equals {f:g} Hz at 60 × {f:g}/{order:g} = {number(rpm)} rpm.",
        unit="rpm",
    )


def _margin(rng: np.random.Generator) -> Question:
    f, rpm = pick(rng, (45, 50, 52.3, 55, 60)), pick(rng, (1200, 1480, 1500, 1780))
    order = 2.0
    critical = 60 * f / order
    margin = 100 * separation(critical, rpm)
    return Question(
        f"A misaligned coupling excites 2× the running speed of {rpm:g} rpm; the train's mode is at {f:g} Hz. How far "
        "is that critical from the running speed, as a percentage of it?",
        margin,
        f"The 2× line meets {f:g} Hz at 60 × {f:g}/2 = {number(critical)} rpm, {number(margin)} % "
        f"{'above' if critical > rpm else 'below'} {rpm:g} rpm. API 617 asks for 10 %.",
        unit="%",
        tolerance=0.02,
    )


def _tooth_pass(rng: np.random.Generator) -> Question:
    teeth, f = int(pick(rng, (40, 60, 90, 120))), pick(rng, (1500, 2000, 2688, 3000))
    rpm = 60 * f / teeth
    return Question(
        f"A gear of {teeth} teeth turns at the motor's speed, and the mesh's own mode is at {f:g} Hz. At what motor "
        "speed does the tooth-pass meet it?",
        rpm,
        f"The teeth pass at z × rpm/60 = {teeth} × rpm/60 Hz, which is {f:g} Hz at 60 × {f:g}/{teeth} = "
        f"{number(rpm)} rpm.",
        unit="rpm",
        setup=dict(preset=COMPRESSOR_PRESET),
    )


def _node(rng: np.random.Generator) -> Question:
    j1, j2 = pick(rng, (1, 2, 4)), pick(rng, (5, 10, 20))
    share = 100 * j2 / (j1 + j2)
    return Question(
        f"A {j1:g} kg·m² motor and a {j2:g} kg·m² fan on one shaft twist in their mode. How far along the shaft from "
        "the motor is the node, the point that doesn't swing, as a percentage of its length?",
        share,
        f"The two swing opposite, J₁θ₁ = −J₂θ₂ (no net momentum), so θ₁/θ₂ = −{j2:g}/{j1:g}, and the twist falls "
        f"linearly along the shaft: the node is J₂/(J₁ + J₂) = {number(share)} % of the way, near the heavier disc.",
        unit="%",
        tolerance=0.01,
        setup=dict(preset=FAN_PRESET, train=_fan(j1, j2, FAN.links[0].stiffness), shapes=True),
    )


def _softer_coupling(rng: np.random.Generator) -> Question:
    return Question(
        "The motor and fan's coupling is made four times softer. What happens to the train's frequency?",
        0,
        "It halves, 52.3 Hz to 26.2 Hz: f ∝ √k. The 2× critical moves from 1569 to 785 rpm, below the running speed, "
        "passed on the way up.",
        choices=("It halves", "It falls to a quarter", "It falls by about a quarter", "It stays the same"),
        setup=dict(preset=FAN_PRESET, train=_fan(2.0, 10.0, FAN.links[0].stiffness / 4)),
        predict=True,
    )


def _rigid_gears(rng: np.random.Generator) -> Question:
    return Question(
        "The geared compressor's mesh is treated as rigid, the two gears merged into one inertia. What happens to "
        "its two lower modes, 38.9 and 83.1 Hz?",
        0,
        "They barely move (0.05 % and 0.01 %): the mesh is far stiffer than the couplings, so they hardly twist it. "
        "Only the 2688 Hz mode, the gears bouncing on their teeth, is lost.",
        choices=("They barely move", "They rise a lot", "They fall a lot", "They disappear"),
        setup=dict(preset=COMPRESSOR_PRESET, shapes=True),
        predict=True,
    )


def _faster_running(rng: np.random.Generator) -> Question:
    return Question(
        "The motor and fan are run 10 % faster. What happens to the torsional critical speeds?",
        0,
        "Nothing: the train's frequencies don't depend on its speed, so the criticals stay where they are. Only the "
        "band round the running speed moves, here onto the 2× critical at 1569 rpm.",
        choices=(
            "They stay where they are",
            "They rise 10 %",
            "They fall 10 %",
            "They rise as the speed squared",
        ),
        setup=dict(preset=FAN_PRESET, operating_rpm=1630.0),
        predict=True,
    )


def _coupling_spec(rng: np.random.Generator) -> Question:
    j1, j2 = pick(rng, (1, 2, 4)), pick(rng, (5, 10, 20))
    rpm = pick(rng, (1000, 1480, 1780, 2960))
    stiff = j1 * j2 / (j1 + j2)  # k = (2πf)² J₁J₂/(J₁ + J₂)

    def k_at(ratio: float) -> float:
        """The coupling (N·m/rad) whose 2× critical is at `ratio` × the running speed."""
        return (TWO_PI * 2 * ratio * rpm / 60) ** 2 * stiff

    softest, stiffest = rounded(k_at(0.95) / 1000), rounded(k_at(1.35) / 1000)  # kN·m/rad
    design = rounded(k_at(1.2) / 1000)
    start = rounded(k_at(1.0) / 1000)

    def goal(t: TorsionalSetup):
        train = t.drive_train()
        running = t.running_rpm()
        band = [c for c in criticals(train, 1e9) if separation(c.rpm, running) < MARGIN / 100]
        two = train.n == 2 and isinstance(train.links[0], Shaft)
        k = train.links[0].stiffness / 1000 if two else math.nan
        near = min((separation(c.rpm, running) for c in criticals(train, 1e9)), default=math.inf)
        return (
            Requirement("Two discs on a coupling", f"{train.n} discs", two),
            kept("The motor's inertia", train.discs[0].inertia, j1, "kg·m²"),
            kept("The fan's inertia", train.discs[-1].inertia, j2, "kg·m²"),
            kept("The running speed", running, rpm, "rpm"),
            Requirement(
                f"No critical within {MARGIN:g} % of the running speed",
                f"the nearest {number(100 * near)} % away",
                two and not band,
            ),
            at_least("The coupling (it must carry the torque)", k, softest, "kN·m/rad"),
            at_most("The coupling (stiffer ones cost more)", k, stiffest, "kN·m/rad"),
        )

    return Question(
        f"A {j1:g} kg·m² motor drives a {j2:g} kg·m² fan at {rpm:g} rpm, and the coupling's misalignment excites 2×. "
        f"Choose the coupling's stiffness, between {number(softest)} and {number(stiffest)} kN·m/rad, so no torsional "
        f"critical is within {MARGIN:g} % of the running speed.",
        0,
        f"The 2× line meets f = √(k (J₁ + J₂)/J₁J₂)/2π at 30 f rpm, so the critical goes as √k. Within {MARGIN:g} % "
        f"is from {number(k_at(0.9) / 1000)} to {number(k_at(1.1) / 1000)} kN·m/rad; the softest coupling allowed "
        f"is above the lower end, so it has to be stiffer: {number(k_at(1.1) / 1000)} kN·m/rad or more ({design:g} "
        f"puts the critical 20 % above). The 1× critical is twice as fast, far above.",
        setup=dict(preset=FAN_PRESET, train=_fan(j1, j2, 1000 * start), operating_rpm=rpm, top_rpm=2 * rpm),
        goal=goal,
        design=dict(preset=FAN_PRESET, train=_fan(j1, j2, 1000 * design), operating_rpm=rpm, top_rpm=2 * rpm),
    )


PROBLEMS = tuple(
    Problem(key, title, "torsional_page", f"See {page_link('torsional_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("two-discs", "Two discs on a shaft", "torsion", "Torsional vibration", _two_discs),
        ("shaft-stiffness", "A shaft's stiffness", "torsion", "Torsional vibration", _shaft_stiffness),
        ("referred", "An inertia through a gear", "gears", "Gears", _referred),
        ("critical", "A critical speed", "interference", "The interference diagram", _critical),
        ("margin", "The margin to a critical", "interference", "The interference diagram", _margin),
        ("tooth-pass", "Tooth-pass meets the mesh", "gears", "Gears", _tooth_pass),
        ("node", "Where the node is", "torsion", "Torsional vibration", _node),
        ("softer-coupling", "Predict: a softer coupling", "torsion", "Torsional vibration", _softer_coupling),
        ("rigid-gears", "Predict: rigid gears", "gears", "Gears", _rigid_gears),
        ("faster-running", "Predict: running faster", "interference", "The interference diagram", _faster_running),
        ("coupling-spec", "Spec: a coupling's stiffness", "interference", "The interference diagram", _coupling_spec),
    )
)


TIPS = {
    "operating": "<p>The motor's running speed: the interference diagram and the response mark a band of "
    "±10 % round it, where no torsional critical should be.</p>",
    "top": "<p>The highest motor speed the diagram and the response show.</p>",
}
INTERFERENCE_TIP = (
    "<p>The interference (Campbell) diagram. The flat lines are the natural frequencies: they don't change "
    "with speed. Each line through the origin is an order: an excitation at that multiple of the motor's "
    "speed (solid: the train's own excitation and tooth-pass; dashed: 1× and 2× of each shaft's speed, which "
    "a standard asks to check). Where an order crosses a mode is a torsional critical speed; filled red "
    "inside the band round the running speed.</p>"
)
RESPONSE_TIP = (
    "<p>The steady dynamic torque in the chosen link (a mesh: its force times the near gear's base radius), "
    "the amplitude of each order on its own and their sum (the peak when they line up), against the motor's "
    "speed. Each peaks where its order meets a mode the link twists in.</p>"
)


def presets_html() -> str:
    """The presets as the Theory tab lists them, each with its start-up's note where it has one."""
    items = []
    for p in PRESETS:
        start = STARTUPS.get(p.name)
        more = f" <i>Start-up:</i> {markup(start.note)}" if start is not None else ""
        items.append(f"<li><b>{p.name}.</b> {markup(p.note)}{more}</li>")
    return "<ol>" + "".join(items) + "</ol>"


NOTES_HTML = load_notes("torsional", figures=torsional_figures.FIGURES, presets=presets_html())


class TorsionalPage(QtWidgets.QWidget):
    """Controls on the left, the interference diagram and the response in the centre, drawing and shapes on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.train: DriveTrain = PRESETS[0].train
        self._loading = False
        self._modes: list[TorsionalMode] = []
        self._unreached = 0  # modes above every order in the speed range, left off the interference diagram

        # --- left: the preset, the train, the excitation, the speeds
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems([p.name for p in PRESETS] + [CUSTOM])
        self.preset.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.preset_note = QtWidgets.QLabel()
        self.preset_note.setWordWrap(True)
        mute(self.preset_note)

        self.train_editor = TrainEditor()  # the Drive train and Excitation panes

        speeds = QtWidgets.QGroupBox("Speeds")
        sf = form_layout(speeds)
        self.operating = spin(1.0, 1e5, 1480.0, 0, " rpm")
        self.top = spin(10.0, 1e5, 3600.0, 0, " rpm")
        form_row(sf, "Running speed:", self.operating, TIPS["operating"])
        form_row(sf, "Show up to:", self.top, TIPS["top"])
        self.log_scales = QtWidgets.QCheckBox("Log scales on the interference diagram")
        self.log_scales.setToolTip(
            "<p>Both axes logarithmic: every order is then a parallel line, and modes "
            "decades apart (a mesh's and the couplings') all show. On by itself when "
            "a train loads whose modes span more than a decade.</p>"
        )
        sf.addRow(self.log_scales)

        self.startup_view = StartUpView()  # the Start-up pane (its controls) and tab

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        presets = QtWidgets.QGroupBox("Things to try")
        preset_layout = QtWidgets.QVBoxLayout(presets)
        preset_layout.addWidget(self.preset)
        preset_layout.addWidget(self.preset_note)
        lv.addWidget(presets)
        for pane in (self.train_editor, self.train_editor.excitation_box, speeds, self.startup_view.controls):
            lv.addWidget(pane)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)

        # --- centre: the interference diagram, the readout, the response
        self.interference = plot("Frequency", "Hz", "Motor speed", "rpm", log_ready=True)
        self.interference.setToolTip(INTERFERENCE_TIP)
        plot_height(self.interference, 220)
        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.response_link = QtWidgets.QComboBox()
        self.response_link.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.response_link.setMinimumContentsLength(18)
        self.response_link.setToolTip(RESPONSE_TIP)
        self.response = plot("Dynamic torque", "N·m", "Motor speed", "rpm", log_ready=True)
        self.response.setLogMode(x=False, y=True)
        self.response.setToolTip(RESPONSE_TIP)
        plot_height(self.response, 180)
        self.response_legend = add_legend(self.response.getPlotItem(), offset=(-10, 10))  # top right
        self.response_note = QtWidgets.QLabel()
        self.response_note.setWordWrap(True)
        self.response_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        mute(self.response_note)
        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.addWidget(self.interference, 3)
        cv.addWidget(self.readout)
        row = FlowLayout()
        row.add(group("Dynamic torque in:", self.response_link))
        cv.addLayout(row)
        cv.addWidget(self.response, 2)
        cv.addWidget(self.response_note)

        self.views = QtWidgets.QTabWidget()
        self.criticals_tab = center
        self.startup_tab = self.startup_view
        self.views.addTab(center, "Criticals")
        self.views.addTab(self.startup_view, "Start-up")

        # --- right: the train drawn, the mode shapes, the notes
        self.drawing = pg.PlotWidget()
        self.drawing.setMenuEnabled(False)
        self.drawing.hideButtons()
        self.drawing.hideAxis("left")
        self.drawing.hideAxis("bottom")
        self.drawing.setMouseEnabled(x=False, y=False)
        self.drawing.setAspectLocked(True)
        self.drawing.scene().sigMouseClicked.connect(self._on_drawing_click)
        self.drawing.setToolTip(
            "<p>The train, motor on the left: discs (taller the more inertia), shafts, and "
            "gear pairs, the far shaft stepped down. Click a disc or a link to edit it (the list above the editor "
            "names the links).</p>"
        )
        self.drawing_note = QtWidgets.QLabel()
        self.drawing_note.setWordWrap(True)
        mute(self.drawing_note)
        drawing_tab = QtWidgets.QWidget()
        dv = QtWidgets.QVBoxLayout(drawing_tab)
        dv.addWidget(self.drawing, 1)
        dv.addWidget(self.drawing_note)

        self.shapes = plot("Swing (referred to the motor's shaft)", None, "Disc", None, log_ready=True)
        self.shapes.setMouseEnabled(x=False, y=False)
        self.shapes.setToolTip(
            "<p>Each mode's swing at each disc, referred to the motor's shaft (θ / s, so a "
            "rigid turn of the whole train would be flat), largest +1. A link the curve "
            "crosses zero over holds a node, and twists.</p>"
        )
        self.shapes_legend = add_legend(self.shapes.getPlotItem(), offset=(10, 10))
        self.mode_list = QtWidgets.QLabel()
        self.mode_list.setWordWrap(True)
        self.mode_list.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.check = QtWidgets.QLabel()
        self.check.setWordWrap(True)
        self.check.setTextFormat(QtCore.Qt.TextFormat.RichText)
        mute(self.check)
        shapes_tab = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(shapes_tab)
        sv.addWidget(self.shapes, 1)
        sv.addWidget(self.mode_list)
        sv.addWidget(self.check)

        self.notes = NotesBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.drawing_tab = drawing_tab
        self.shapes_tab = shapes_tab
        self.tabs.addTab(drawing_tab, "Drive train")
        self.tabs.addTab(shapes_tab, "Mode shapes")
        self.tabs.addTab(self.notes, "Theory")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self.views)
        splitter.addWidget(self.tabs)
        splitter.setSizes([340, 810, 550])
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # --- wiring
        self.preset.currentIndexChanged.connect(self._on_preset)
        self.train_editor.built.connect(self.set_train)
        self.train_editor.picked.connect(self._on_picked)
        self.startup_view.edited.connect(self._on_startup_edited)
        self.startup_view.shown_wanted.connect(self._show_startup)
        self.operating.valueChanged.connect(self._on_speeds)
        self.top.valueChanged.connect(self._on_speeds)
        self.log_scales.toggled.connect(self._draw_interference)
        self.response_link.currentIndexChanged.connect(self._draw_response)

        self.load_preset(PRESETS[0].name)

    # ------------------------------------------------------------- inputs
    def load_preset(self, name: str) -> None:
        p = next(p for p in PRESETS if p.name == name)
        self._loading = True
        self.operating.setValue(p.operating_rpm)
        self.top.setValue(p.top_rpm)
        self.preset.blockSignals(True)
        self.preset.setCurrentText(name)
        self.preset.blockSignals(False)
        self.preset_note.setText(markup(p.note))
        self._loading = False
        self.train_editor.selected = 0
        self.response_link.blockSignals(True)
        self.response_link.setCurrentIndex(-1)  # set_train picks the slowest mode's link
        self.response_link.blockSignals(False)
        f = natural_hz(p.train)
        self.log_scales.blockSignals(True)
        self.log_scales.setChecked(bool(f.size and f[-1] > 10 * f[0]))
        self.log_scales.blockSignals(False)
        self.startup_view.load(STARTUPS.get(name))
        self.set_train(p.train, custom=False)

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says: the preset, then its train and speeds if given.
        The start-up isn't started."""
        t = TorsionalSetup(**setup)
        self.load_preset(_preset(t.preset).name)
        if t.train is not None:
            self.set_train(t.train)
        if t.operating_rpm is not None or t.top_rpm is not None:
            self._loading = True
            if t.operating_rpm is not None:
                self.operating.setValue(t.operating_rpm)
            if t.top_rpm is not None:
                self.top.setValue(t.top_rpm)
            self._loading = False
            self._to_custom()
            self.redraw()
        self.tabs.setCurrentIndex(1 if t.shapes else 0)

    def practice_state(self) -> TorsionalSetup:
        """The train and the speeds, as a setup: what a practice spec checks."""
        return TorsionalSetup(
            self.preset.currentText(),
            self.train,
            self.operating.value(),
            self.top.value(),
            self.tabs.currentIndex() == 1,
        )

    def frame_context(self) -> str:
        """For the frame log (frame_stats): the preset showing."""
        return self.preset.currentText()

    def _on_preset(self, index: int) -> None:
        if index < len(PRESETS):
            self.load_preset(PRESETS[index].name)

    def _to_custom(self) -> None:
        self.preset.blockSignals(True)
        self.preset.setCurrentText(CUSTOM)
        self.preset.blockSignals(False)
        self.preset_note.setText("")
        self.startup_view.clear_note()

    def set_train(self, train: DriveTrain, custom: bool = True) -> None:
        """Show a new train: every view redrawn, the editors refilled."""
        self.train = train
        self._modes = modes(train)
        if custom:
            self._to_custom()
        self.train_editor.show_train(train)
        self._fill_response_links()
        self.redraw()
        self.startup_view.set_train(train, self._modes)

    def _fill_response_links(self) -> None:
        t = self.train
        shown = self.response_link.currentIndex()
        self.response_link.blockSignals(True)
        self.response_link.clear()
        self.response_link.addItems([link.name for link in t.links])
        if not 0 <= shown < len(t.links):
            # the link the slowest mode twists most, by default
            shown = self._modes[0].link if self._modes else 0
        self.response_link.setCurrentIndex(shown)
        self.response_link.blockSignals(False)

    @property
    def selected(self) -> int:
        """The element being edited: 2i the i-th disc, 2i + 1 the link after it."""
        return self.train_editor.selected

    def select(self, element: int) -> None:
        """Edit element `element`: 2i the i-th disc, 2i + 1 the link after it."""
        self.train_editor.select(element)
        self._draw_train()

    def _on_picked(self, _element: int) -> None:
        self._draw_train()

    # ------------------------------------------------------------- start-up
    def _on_startup_edited(self) -> None:
        self._to_custom()
        self._draw_interference()

    def _show_startup(self) -> None:
        self.views.setCurrentWidget(self.startup_view)

    def start(self) -> None:
        """Start the train from rest, and show it."""
        self.startup_view.start()

    @property
    def running(self) -> bool:
        """The start-up playing (the page's one motion)."""
        return self.startup_view.running

    @property
    def pause_button(self) -> RunButton:
        return self.startup_view.pause_button

    @property
    def loop(self) -> FrameLoop:
        return self.startup_view.loop

    def _on_speeds(self) -> None:
        if self._loading:
            return
        self._to_custom()
        self.redraw()

    # ------------------------------------------------------------- start-up
    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        self.redraw()
        self.startup_view.apply_theme()

    def redraw(self) -> None:
        self._draw_interference()
        self._draw_response()
        self._draw_train()
        self._draw_shapes()

    def _band(self) -> tuple[float, float]:
        op = self.operating.value()
        return op * (1 - MARGIN / 100), op * (1 + MARGIN / 100)

    def _band_item(self) -> pg.LinearRegionItem:
        lo, hi = self._band()
        band = pg.LinearRegionItem((lo, hi), movable=False, brush=pg.mkBrush(*colors.band_shade))
        for line in band.lines:
            line.setPen(pg.mkPen(None))
        return band

    def _draw_interference(self) -> None:
        p = self.interference
        p.clear()
        t, top = self.train, self.top.value()
        log = self.log_scales.isChecked()
        if log != p.getPlotItem().ctrl.logXCheck.isChecked():
            # through a range that reads the same both ways: 0..3600 read as log10 overflows
            p.getPlotItem().getViewBox().setRange(xRange=(0, 1), yRange=(0, 1), padding=0)
            p.setLogMode(x=log, y=log)
        lines = order_lines(t)
        crits = criticals(t, top, lines)
        # up to the fastest mode an order reaches in the speed range (or the slowest, if none does)
        reached = max((c.mode for c in crits), default=0)
        freqs = [md.freq_hz for md in self._modes[: reached + 1]]
        y_top = (2.0 if log else 1.2) * max(freqs) if freqs else 100.0
        y_lo = 0.5 * min(freqs) if log and freqs else 0.0
        x_lo = top / 100.0 if log else 0.0

        def at(x: float, y: float) -> tuple[float, float]:
            """A point in the view's coordinates (log10 of each on log scales), for the items pyqtgraph
            doesn't map itself (text, lines)."""
            return (math.log10(x), math.log10(y)) if log else (x, y)

        lo, hi = self._band()
        band = pg.LinearRegionItem((at(lo, 1)[0], at(hi, 1)[0]), movable=False, brush=pg.mkBrush(*colors.band_shade))
        for line in band.lines:
            line.setPen(pg.mkPen(None))
        p.addItem(band)
        p.addItem(
            pg.InfiniteLine(
                at(self.operating.value(), 1)[0],
                angle=90,
                pen=pg.mkPen(colors.muted, width=1, style=QtCore.Qt.PenStyle.DashLine),
            )
        )
        n_sources = len(sources(t))  # order_lines puts them first
        for j, (label, order) in enumerate(lines):
            own = j < n_sources
            color = colors.mass[j % len(colors.mass)] if own else colors.grey
            pen = pg.mkPen(
                color,
                width=1.5 if own else 1,
                style=QtCore.Qt.PenStyle.SolidLine if own else QtCore.Qt.PenStyle.DashLine,
            )
            # from the bottom (or left) of the plot to its top, or the speed range's end
            start = max(x_lo, 60.0 * y_lo / order)
            end = min(top, 60.0 * y_top / order)
            if end <= start:
                continue
            p.plot([start, end], [order * start / 60.0, order * end / 60.0], pen=pen)
            text = pg.TextItem(label, color=color, anchor=(1, 1) if end < top else (1, 0.5))
            text.setPos(*at(end, order * end / 60.0))
            p.addItem(text)
        motor = self.startup_view.startup.motor
        if self.startup_view.motor_on and motor.kind is MotorKind.SYNCHRONOUS:
            # the poles' 2 s f pulsation, falling from 2f at rest to nothing at synchronism
            xs = np.linspace(max(x_lo, 1e-9), min(top, motor.sync_rpm), 200)
            ys = 2.0 * motor.line_hz * (1.0 - xs / motor.sync_rpm)
            keep = ys > max(y_lo, 1e-9)
            p.plot(xs[keep], ys[keep], pen=pg.mkPen(colors.force, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
            x = 0.45 * min(top, motor.sync_rpm)  # the label ends on the line, below it (it falls to the right)
            text = pg.TextItem("2× slip (start-up)", color=colors.force, anchor=(1, 0))
            text.setPos(*at(x, 2.0 * motor.line_hz * (1.0 - x / motor.sync_rpm)))
            p.addItem(text)
            cs = [c for c in crossings(t, motor) if c.mode < len(freqs) and c.rpm >= x_lo]
            if cs:
                p.plot(
                    [c.rpm for c in cs],
                    [c.freq_hz for c in cs],
                    pen=None,
                    symbol="d",
                    symbolSize=10,
                    symbolPen=pg.mkPen(colors.force, width=1.5),
                    symbolBrush=pg.mkBrush(colors.background),
                )
        for j, f in enumerate(freqs):
            p.plot([max(x_lo, 1e-9), top], [f, f], pen=pg.mkPen(colors.strong, width=1.5))
            text = pg.TextItem(f"Mode {j + 1}, {f:.4g} Hz", color=colors.strong, anchor=(0, 1))
            text.setPos(*at(max(x_lo, 1e-9), f))
            p.addItem(text)
        inside = [c for c in crits if lo <= c.rpm <= hi]
        outside = [c for c in crits if not lo <= c.rpm <= hi and c.rpm >= x_lo]
        if outside:
            p.plot(
                [c.rpm for c in outside],
                [c.freq_hz for c in outside],
                pen=None,
                symbol="o",
                symbolSize=8,
                symbolPen=pg.mkPen(colors.strong, width=1.5),
                symbolBrush=pg.mkBrush(colors.background),
            )
        if inside:
            p.plot(
                [c.rpm for c in inside],
                [c.freq_hz for c in inside],
                pen=None,
                symbol="o",
                symbolSize=10,
                symbolPen=pg.mkPen(colors.force, width=1.5),
                symbolBrush=pg.mkBrush(colors.force),
            )
        p.getPlotItem().getViewBox().setRange(
            xRange=(at(x_lo or 1, 1)[0] if log else 0.0, at(top, 1)[0]),
            yRange=(at(1, y_lo)[1] if log else 0.0, at(1, y_top)[1]),
            padding=0,
        )
        self._unreached = len(self._modes) - len(freqs)
        self._show_readout(crits, inside)

    def _show_readout(self, crits, inside) -> None:
        op = self.operating.value()
        lines = []
        if not self._modes:
            self.readout.setText("No flexible modes.")
            return
        if inside:
            listed = "; ".join(
                f"<b>{c.label} meets mode {c.mode + 1} ({c.freq_hz:.4g} Hz) at {c.rpm:.0f} rpm</b>, "
                f"{100 * separation(c.rpm, op):.1f} % {'above' if c.rpm > op else 'below'}"
                for c in inside
            )
            lines.append(f"Inside ±{MARGIN:g} % of {op:.0f} rpm: {listed}.")
        else:
            near = min(crits, key=lambda c: separation(c.rpm, op), default=None)
            if near is None:
                lines.append(f"No order meets a mode below {self.top.value():.0f} rpm.")
            else:
                lines.append(
                    f"No torsional critical within ±{MARGIN:g} % of {op:.0f} rpm: the nearest, {near.label} "
                    f"on mode {near.mode + 1} ({near.freq_hz:.4g} Hz), is at {near.rpm:.0f} rpm, "
                    f"{100 * separation(near.rpm, op):.1f} % {'above' if near.rpm > op else 'below'}."
                )
        if self._unreached:
            lines.append(
                f"{self._unreached} faster mode{'s' if self._unreached > 1 else ''} no order reaches "
                f"below {self.top.value():.0f} rpm (not drawn)."
            )
        below = [c for c in crits if c.rpm < op * (1 - MARGIN / 100)]
        if below:
            lines.append(
                f"A start-up passes {len(below)} below it, the last {below[-1].label} on mode "
                f"{below[-1].mode + 1} at {below[-1].rpm:.0f} rpm."
            )
        self.readout.setText(" ".join(lines))

    def _draw_response(self) -> None:
        p = self.response
        p.clear()
        self.response_legend.clear()
        t, top = self.train, self.top.value()
        link = self.response_link.currentIndex()
        if not 0 <= link < len(t.links):
            return
        rpm = response_speeds(t, top)
        fr = forced_response(t, rpm)
        if not fr.sources:
            self.response_note.setText(
                "No excitation: pick a disc to put a torque on, or give a mesh a transmission error."
            )
            p.setXRange(0.0, top, padding=0)
            return
        p.addItem(self._band_item())
        legend = self.response_legend
        curves = [
            (src.label, fr.torque[a, link], colors.mass[a % len(colors.mass)], 1.5) for a, src in enumerate(fr.sources)
        ]
        total = fr.total()[link]
        if len(fr.sources) > 1:
            curves.append(("Sum", total, colors.strong, 2.0))
        peak = float(total.max())
        floor = peak * 1e-4 if peak > 0 else 1e-9
        for label, y, color, width in curves:
            item = p.plot(rpm, np.maximum(y, floor), pen=pg.mkPen(color, width=width))
            legend.addItem(item, label)
        p.addItem(
            pg.InfiniteLine(
                self.operating.value(), angle=90, pen=pg.mkPen(colors.muted, width=1, style=QtCore.Qt.PenStyle.DashLine)
            )
        )
        p.setXRange(0.0, top, padding=0)
        y_top = peak * 3.0 if peak > 0 else 1.0
        p.setYRange(math.log10(floor), math.log10(y_top), padding=0)
        op = self.operating.value()
        at_op = float(np.interp(op, rpm, total))
        lo, hi = self._band()
        band = (rpm >= lo) & (rpm <= hi)
        worst = float(total[band].max()) if band.any() else at_op
        i = int(np.argmax(total))
        name = t.links[link].name
        self.response_note.setText(
            f"{name}: ±{fmt_torque(at_op)} at {op:.0f} rpm, up to ±{fmt_torque(worst)} within ±{MARGIN:g} %; "
            f"the most, ±{fmt_torque(peak)}, at {rpm[i]:.0f} rpm."
        )

    def _layout(self) -> tuple[list[tuple[float, float, float]], float]:
        """Each disc's (x, axis height, half height) in the drawing, and the width of a disc."""
        t = self.train
        width = 0.5
        jmax = max(d.inertia for d in t.discs)
        h = [0.25 + 0.75 * (d.inertia / jmax) ** 0.25 for d in t.discs]
        out: list[tuple[float, float, float]] = []
        x, y = 0.0, 0.0
        for i, _d in enumerate(t.discs):
            if i:
                link = t.links[i - 1]
                if isinstance(link, Mesh):
                    # the gears on one line across: sized by their teeth, touching
                    near = out[-1]
                    r_in = near[2]
                    r_out = r_in * link.teeth_out / link.teeth_in
                    h[i] = r_out
                    y = near[1] - r_in - r_out
                    x = near[0]
                else:
                    x += 1.6
            out.append((x, y, h[i]))
        return out, width

    def _draw_train(self) -> None:
        p = self.drawing
        p.clear()
        t = self.train
        spots, width = self._layout()
        structure = colors.structure
        for i in range(t.n - 1):
            (x0, y0, h0), (x1, y1, h1) = spots[i], spots[i + 1]
            chosen = self.selected == 2 * i + 1
            color = colors.force if chosen else structure
            if isinstance(t.links[i], Mesh):
                p.plot(
                    [x0, x0],
                    [y0 - h0, y1 + h1],
                    pen=pg.mkPen(color, width=3 if chosen else 1.5),
                    symbol="o",
                    symbolSize=6,
                    symbolBrush=pg.mkBrush(color),
                    symbolPen=None,
                )
            else:
                p.plot([x0 + width / 2, x1 - width / 2], [y0, y1], pen=pg.mkPen(color, width=4 if chosen else 2.5))
        for i, (x, y, h) in enumerate(spots):
            chosen = self.selected == 2 * i
            rect = QtWidgets.QGraphicsRectItem(x - width / 2, y - h, width, 2 * h)
            rect.setPen(pg.mkPen(colors.force if chosen else structure, width=3 if chosen else 1.5))
            rect.setBrush(pg.mkBrush(colors.mass[i % len(colors.mass)]))
            p.addItem(rect)
            label = pg.TextItem(t.discs[i].name, color=colors.foreground, anchor=(0.5, 1))
            label.setPos(x, y + h + 0.05)
            label.setFont(_small_font())
            p.addItem(label)
        if t.excitation is not None:
            x, y, h = spots[t.excitation.disc]
            arrow = pg.ArrowItem(
                angle=90, tipAngle=40, headLen=14, tailLen=12, tailWidth=3, pen=None, brush=pg.mkBrush(colors.force)
            )
            arrow.setPos(x, y - h)
            p.addItem(arrow)
        xs = [x for x, _, _ in spots]
        ys = [y - h for _, y, h in spots] + [y + h for _, y, h in spots]
        p.getPlotItem().getViewBox().setRange(
            xRange=(min(xs) - 1.0, max(xs) + 1.0), yRange=(min(ys) - 0.6, max(ys) + 0.4), padding=0
        )
        s = t.speeds
        if t.meshes:
            speeds = (
                "Each shaft's speed over the motor's: "
                + ", ".join(
                    f"{d.name} {s[i]:.3g}×" for i, d in enumerate(t.discs) if i and not math.isclose(s[i], s[i - 1])
                )
                + "."
            )
        else:
            speeds = "One shaft: every disc turns at the motor's speed."
        if t.excitation is not None:
            speeds += f" The arrow is the excitation, on the {t.discs[t.excitation.disc].name.lower()}."
        self.drawing_note.setText(speeds)

    def _on_drawing_click(self, event) -> None:
        vb = self.drawing.getPlotItem().getViewBox()
        pos = vb.mapSceneToView(event.scenePos())
        spots, width = self._layout()
        best, dist = None, math.inf
        for i, (x, y, h) in enumerate(spots):
            if abs(pos.x() - x) <= width / 2 and abs(pos.y() - y) <= h:
                best, dist = 2 * i, 0.0
                break
        if best is None:
            for i in range(len(spots) - 1):
                (x0, y0, _), (x1, y1, _) = spots[i], spots[i + 1]
                d = math.hypot(pos.x() - 0.5 * (x0 + x1), pos.y() - 0.5 * (y0 + y1))
                if d < dist:
                    best, dist = 2 * i + 1, d
            if best is None or dist > 1.0:
                return
        self.select(best)

    def _draw_shapes(self) -> None:
        p = self.shapes
        p.clear()
        self.shapes_legend.clear()
        t = self.train
        x = np.arange(t.n)
        p.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        for j, md in enumerate(self._modes[:SHAPES_SHOWN]):
            color = colors.mode[j % len(colors.mode)]
            item = p.plot(
                x,
                md.shape,
                pen=pg.mkPen(color, width=2),
                symbol="o",
                symbolSize=6,
                symbolBrush=pg.mkBrush(color),
                symbolPen=None,
            )
            self.shapes_legend.addItem(item, f"Mode {j + 1}, {md.freq_hz:.4g} Hz")
        p.getAxis("bottom").setTicks([[(i, d.name) for i, d in enumerate(t.discs)]])
        p.setXRange(-0.3, t.n - 0.7, padding=0)
        p.setYRange(-1.1, 1.1, padding=0)
        rows = []
        for j, md in enumerate(self._modes):
            share = md.twist[md.link]
            rows.append(
                f"Mode {j + 1}: {md.freq_hz:.4g} Hz, ζ = {100 * md.zeta:.2g} %, twisting "
                f"{t.links[md.link].name.lower()} ({100 * share:.0f} %)"
            )
        self.mode_list.setText("<br>".join(rows) if rows else "No flexible modes.")
        self._show_check()

    def _show_check(self) -> None:
        t = self.train
        f = natural_hz(t)
        ref = natural_hz(t.referred())
        text = (
            "Referred to the motor's shaft (inertias and stiffnesses × s², each mesh a shaft of "
            "k<sub>m</sub>(r s)²): " + ", ".join(f"{v:.4g}" for v in ref) + " Hz, the same."
        )
        if t.meshes:
            try:
                rigid = natural_hz(t.rigid_gears())
            except ValueError:
                rigid = np.empty(0)
            lost = f.size - rigid.size
            text += (
                " With the gears rigid (the n² reflection): "
                + (", ".join(f"{v:.4g}" for v in rigid) + " Hz" if rigid.size else "no modes")
                + f", {lost} mode{'s' if lost != 1 else ''} lost."
            )
        self.check.setText(text)


def _small_font() -> QtGui.QFont:
    font = QtGui.QFont()
    font.setPointSizeF(font.pointSizeF() * 0.85)
    return font
