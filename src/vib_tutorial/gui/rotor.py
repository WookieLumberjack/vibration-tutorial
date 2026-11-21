"""Jeffcott rotor page: a disc on a flexible shaft in flexible bearings, spun up through its criticals.

Left: the speed controls, the run-up and coast-down sweep, and the rotor's
parameters (`rotor_editor.RotorEditor`: its boxes, and the RotorSystem they describe). Centre: a 3D view of the whirling shaft (projected onto a 2D
pyqtgraph scene, no OpenGL), the orbits at the bearings, the disc and midspan,
and a strip chart. Right: the Campbell diagram, the Bode and polar plots of the
1X response (steady state and tracked sweeps), the waterfall (`rotor_waterfall.WaterfallTab`),
the stability map and full spectrum, the matrices of the equations of motion at the current speed, and the
theory notes. The notes under the boxes and on the Stability tab are `rotor_readouts`' texts.
"""

from __future__ import annotations

import cmath
import math
import time
from collections.abc import Callable
from dataclasses import fields

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..core.fluid_bearing import JournalBearing
from ..core.magnetic_bearing import MagneticBearing
from ..core.rotor import (
    XA,
    XB,
    XD,
    YA,
    YB,
    YD,
    Bearing,
    Crack,
    PlaneWeight,
    RotorSystem,
    Stator,
)
from ..core.rotor_analysis import Analysis, AnalysisWorker, analyse
from ..core.rotor_balance import SteadyMeasurement
from ..core.rotor_modes import Campbell, whirl_modes
from ..core.rotor_presets import rotor_presets
from ..core.rotor_simulator import RPM, RotorSimulator
from ..core.rotor_stability import AsymmetryChart, Onset, full_spectrum, least_damped
from ..core.rotor_sweep import (
    DIRECTIONS,
    STATIONS,
    RunUpCoastDown,
    TrackedSweep,
    phase_degrees,
    probe,
)
from ..view_models.text import number as sig
from . import rotor_practice, weak
from .challenge import ChallengeBox
from .cms_time import SampleBuffer
from .frame_loop import FrameLoop
from .layouts import ColumnScroll, FlowLayout, form_layout, group
from .lesson import RunButton, TrialBox
from .live_label import LiveLabel
from .notes_view import NotesBrowser
from .panels import spin
from .rotor_balance import BalancingPanel
from .rotor_bearing import BearingPanel
from .rotor_bode import SWEEP_COLORS, BodePlots, PolarPlot, SweepPlots, nice_marks
from .rotor_editor import FILMS, MAGNETS, RotorEditor
from .rotor_magnet import MagnetPanel
from .rotor_notes import THEORY_HTML, matrices_html, whirl_name
from .rotor_plots import DAMPED, MIN_PEAK, ORDERS, CampbellPlot, OrbitPlots, TimePlots, critical_speeds
from .rotor_practice import RotorSetup
from .rotor_readouts import fmt_len, instability_causes, onset_text, shaft_note
from .rotor_rub import SPIRAL_POINTS, RubPanel
from .rotor_stability import MEASURES, AsymmetryPlot, FullSpectrum, StabilityMap
from .rotor_tips import (
    HOLD_SWEEPS_TIP,
    ORDER_TIPS,
    REPEAT_TAP_TIP,
    SAG_TIP,
    SPEED_TIP,
    SWEEP_TIP,
    TAP_TIP,
)
from .rotor_view import RotorView
from .rotor_waterfall import WaterfallTab
from .scrub import Scrub
from .style import colors
from .tab_groups import TabGroups
from .theming import mute
from .transport import Transport, speed_text

SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0)  # simulation speed, × real time
TOP_RPM = 10_000.0  # the default speed range
MAX_TOP_RPM = 60_000.0  # the widest it can be set to
PROBLEMS = rotor_practice.PROBLEMS  # the page's practice problems, kept beside it in rotor_practice
TRIALS = tuple(rotor_presets())  # its Things to try are the presets (goals' trial: links count them)
PRACTICE_SETUP = RotorSetup  # what a practice problem's setup and the page's practice_state are
CUSTOM = "Custom"  # the preset box's entry for a rotor that is none of the presets  # other orders of the speed the Campbell diagram can draw
DIRECTION_NAMES = ("x (horizontal probe)", "y (vertical probe)", "Orbit (major axis)")
# Columns of the history: (x, y) at each station, then the unbalance's angle φ, the speed Ω, and a stator
# ring's push on the disc (the largest over the steps each sample stands for).
PHASE, OMEGA, PUSH = 8, 9, 10
STORE_DT = 2.5e-4  # s; samples kept for the plots are at least this far apart
TAP_IMPULSE = 0.005  # N·s, downward on the disc
ORBIT_REVS = 3  # revolutions in each orbit trail
ORBIT_MAX_TIME = 1.0  # s, the longest orbit trail (at low speed)
DASH = QtCore.Qt.PenStyle.DashLine
WORKER = AnalysisWorker()  # shared by every rotor page: one process
JOB_POLL_MS = 30  # how often the page checks for a finished analysis
BUSY_NOTE_S = 0.3  # s an analysis runs before the page says it is working on it
RESTART_S = 0.5  # s a superseded analysis has run past which the worker drops it rather than see it out
RUB = 0.02  # of the span: a whirl this large pauses the simulation (a real rotor would rub)
SAG_SPEEDS = 201  # speeds the static sag is worked out at, with internal damping, and interpolated
SPECTRUM_FRAMES = 6  # the full spectrum is recomputed every this many frames
SPIRAL_DT = 0.05  # s of simulated time between the Newkirk spiral's points
MEASURE_SPEED = 2.0  # × real time, at least, while a balancing measurement runs
FRAME_BUDGET = 0.012  # s of computing per frame: past it (the films' whole force is slow) the page runs slower
WATERFALL_FRAMES = 15  # new spectra are drawn at most every this many frames (each repaint fills every line)
CHART_BUDGET = 0.012  # s per frame spent filling in the asymmetry chart while it shows
CHART_DELTA = 0.5  # the asymmetry chart's top Δ (taller only for a rotor past it)
ORDER_NAMES = ("1X", "2X", "3X")  # what the Bode and polar plots can track


class RotorPage(QtWidgets.QWidget):
    """Parameters and speed on the left; 3D view, orbits and time on the centre; Campbell, matrices, theory."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def transport(self) -> Transport:
        """The run bar drives the simulation's run (it ramps up to the target speed) and its speed."""
        return Transport(self.run_button, self.reset_button, self.sim_speed)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        # First: the analysis process imports NumPy and SciPy (~0.3 s) while the page builds, and the
        # page's first analysis goes to it at the end, rather than hold up the page's opening.
        WORKER.start()
        self.sim = RotorSimulator()
        self.sim.target = 1000.0 / RPM
        self.history = SampleBuffer(11)
        self.campbell: Campbell | None = None
        self.top_rpm = TOP_RPM  # the speed range: Campbell diagram, stability map, Bode plot, speed controls
        self.running = False  # at rest, not spinning, until started
        self._stride = 1
        self._carry = 0  # steps since the last stored sample
        self._matrices_speed = -math.inf  # rpm the matrices were drawn at
        self.run_up: RunUpCoastDown | None = None  # the sweep running, if any
        self.sweeps: list[TrackedSweep] = []  # every leg on the plots, the running one's too
        self._sweep_pairs = 0  # sweeps started since the plots were cleared (each pair's colour)
        self._steady_rpm = np.empty(0)  # the steady-state curve: speeds (rpm)
        self._steady_vectors = np.empty((0, len(STATIONS), 2), dtype=complex)  # and 1X vectors there
        self._steady_orders = [self._steady_vectors] * len(ORDER_NAMES)  # and the 1X, 2X and 3X vectors
        self.asym_chart: AsymmetryChart | None = None  # the stability tab's asymmetry chart, as far as it got
        self.onset: Onset | None = None  # where the rotor goes unstable, if it does up to top_rpm
        self._analysed: RotorSystem | None = None  # the rotor campbell and onset are of (until a new one's arrive)
        self.rubbed = False  # the whirl grew past RUB of the span and the simulation paused
        self.touched: tuple[str, float] | None = None  # or a journal reached its film's clearance: which, how far
        self._sag: tuple[np.ndarray, np.ndarray] | None = None  # see _set_sag
        self._sag_q: tuple[np.ndarray, np.ndarray] | None = None  # and the whole q_s on the same speeds
        self.measurement: SteadyMeasurement | None = None  # a balancing measurement running, if any
        self._measured: Callable[[np.ndarray], None] | None = None  # its callback, given the mean 1X vectors (4, 2)
        self._frames = 0
        self._next_tap = 0.0  # simulated time of the next repeated tap
        self._spiral: tuple[list[complex], list[complex]] = ([], [])  # the 1X vectors and thermal bows recorded
        self._spiral_next = 0.0  # simulated time of the spiral's next point
        self._heat = 0.0  # W: the rub's heat over the last advance

        # The bearing and rub tabs' panels: the editor reads the rotor's films, magnets and stator from them.
        self.bearing_panel = BearingPanel()
        self.bearing_panel.set_top_rpm(self.top_rpm)
        self.bearing_panel.changed.connect(weak.slot(self, lambda self: self._on_bearing_tab(FILMS)))
        self.magnet_panel = MagnetPanel()
        self.magnet_panel.set_top_rpm(self.top_rpm)
        self.magnet_panel.changed.connect(weak.slot(self, lambda self: self._on_bearing_tab(MAGNETS)))
        self.rub_panel = RubPanel()
        self.rub_panel.changed.connect(self._on_params)

        # --- parameters
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(self._preset_box())
        self.challenge = ChallengeBox(self, PROBLEMS)
        lv.addWidget(self.challenge)
        lv.addWidget(self._speed_box())
        self.readout = LiveLabel(wrap=True)  # the speed: rewritten every frame (#409 P-2)
        lv.addWidget(self.readout)
        lv.addWidget(self._sweep_box())
        self.editor = RotorEditor(self.bearing_panel, self.magnet_panel, self.rub_panel)
        self.editor.changed.connect(self._on_params)
        lv.addWidget(self.editor)
        self.crit_label = QtWidgets.QLabel()
        self.crit_label.setWordWrap(True)
        lv.addWidget(self.crit_label)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)

        # --- centre
        self.view = RotorView()
        self.orbits = OrbitPlots()
        self.time = TimePlots()
        self.scrub = Scrub(self.time.x_plot, self.time.w_plot)  # paused: back along the strip chart
        self.scrub.scrubbed.connect(self._on_scrubbed)
        self.station = QtWidgets.QComboBox()
        self.station.addItems(STATIONS)
        self.station.setCurrentIndex(1)
        self.time_window = spin(0.2, 20.0, 3.0, 1, " s")
        self.time_window.setToolTip("Strip chart window")
        self.same_scale = QtWidgets.QCheckBox("Same scale")
        self.same_scale.setToolTip("Draw the orbits at one scale, to compare their sizes")
        self.same_scale.toggled.connect(self._on_same_scale)
        self.show_sag = QtWidgets.QCheckBox("Static sag")
        self.show_sag.setChecked(True)
        self.show_sag.setEnabled(False)
        self.show_sag.setToolTip(SAG_TIP)
        self.show_sag.toggled.connect(self._draw)
        row = FlowLayout()
        row.add(
            QtWidgets.QLabel("Orbits, seen from the B end:"),
            self.same_scale,
            self.show_sag,
            group("Strip chart:", self.station, "Window:", self.time_window),
        )
        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.view, 5)
        cv.addLayout(row)
        cv.addWidget(self.orbits, 3)
        cv.addWidget(self.time, 3)

        # --- right
        self.campbell_plot = CampbellPlot()
        self.campbell_plot.speed_dragged.connect(self.set_target_rpm)
        self.campbell_note = QtWidgets.QLabel()
        self.campbell_note.setWordWrap(True)
        mute(self.campbell_note)
        self.top_speed = spin(1000.0, MAX_TOP_RPM, TOP_RPM, 0, " rpm")
        self.top_speed.setSingleStep(1000.0)
        self.top_speed.setToolTip(
            "<p>The highest speed on the Campbell diagram, the stability map and the Bode plot, and the top "
            "of the speed controls.</p>"
        )
        self.top_speed.valueChanged.connect(self.set_top_rpm)
        self.order_boxes = {}
        for order, name in ORDERS.items():
            box_ = QtWidgets.QCheckBox(name)
            box_.setToolTip(ORDER_TIPS[order])
            box_.toggled.connect(self._on_orders)
            self.order_boxes[order] = box_
        order_row = FlowLayout()
        order_row.add(group("Speeds up to:", self.top_speed), group("Also draw:", *self.order_boxes.values()))
        campbell_tab = QtWidgets.QWidget()
        ct = QtWidgets.QVBoxLayout(campbell_tab)
        ct.addLayout(order_row)
        ct.addWidget(self.campbell_plot, 1)
        ct.addWidget(self.campbell_note)
        self.bode = BodePlots()
        self.polar = PolarPlot()
        self.sweep_plots = SweepPlots(self.bode, self.polar)
        self.sweep_summary = QtWidgets.QLabel()
        self.sweep_summary.setWordWrap(True)
        self.sweep_summary.setTextFormat(QtCore.Qt.TextFormat.RichText)
        bode_tab = QtWidgets.QWidget()
        bt = QtWidgets.QVBoxLayout(bode_tab)
        bt.addWidget(self.bode, 1)
        bt.addWidget(self.sweep_summary)
        self.polar_note = QtWidgets.QLabel(
            "The 1X vector at the chosen station: its length is the amplitude, its angle the phase, counted as "
            "a lead: counter-clockwise from the keyphasor mark at 0°, the way the shaft turns, so a lag turns it "
            "clockwise. Through a lightly damped critical speed the steady-state vector traces a circle, at −90° "
            "(lagging 90°) at its lowest point. With a bowed shaft the curve starts from the "
            "bow (the slow-roll runout), not from the origin. The labels give the speed in rpm."
        )
        self.polar_note.setWordWrap(True)
        mute(self.polar_note)
        polar_tab = QtWidgets.QWidget()
        pt = QtWidgets.QVBoxLayout(polar_tab)
        pt.addWidget(self.polar, 1)
        pt.addWidget(self.polar_note)
        self.stability_map = StabilityMap(DAMPED)
        self.measure = QtWidgets.QComboBox()
        self.measure.addItems(MEASURES)
        self.measure.setToolTip(
            "<p>The log decrement δ = 2πζ/√(1 − ζ²) is the natural log of the ratio of successive peaks of "
            "a decaying vibration; API 684 asks for δ ≥ 0.1 at the running speed.</p>"
        )
        self.asymmetry_plot = AsymmetryPlot()
        self.measure.currentIndexChanged.connect(self._on_measure)
        self.stability_note = QtWidgets.QLabel()
        self.stability_note.setWordWrap(True)
        self.stability_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.spectrum = FullSpectrum()
        self.spectrum_note = QtWidgets.QLabel()
        self.spectrum_note.setWordWrap(True)
        mute(self.spectrum_note)
        stability_tab = QtWidgets.QWidget()
        st = QtWidgets.QVBoxLayout(stability_tab)
        measure_row = QtWidgets.QHBoxLayout()
        measure_row.addWidget(QtWidgets.QLabel("Stability map:"))
        self.measure.setAccessibleName("Stability map")
        measure_row.addWidget(self.measure)
        measure_row.addStretch(1)
        st.addLayout(measure_row)
        self.stability_stack = QtWidgets.QStackedWidget()
        self.stability_stack.addWidget(self.stability_map)
        self.stability_stack.addWidget(self.asymmetry_plot)
        st.addWidget(self.stability_stack, 3)
        st.addWidget(self.stability_note)
        st.addWidget(self.spectrum, 2)
        st.addWidget(self.spectrum_note)
        self.waterfall_tab = WaterfallTab()
        self.balancing = BalancingPanel(self)
        self.matrices = QtWidgets.QTextBrowser()
        self.theory = NotesBrowser()
        self.theory.setHtml(THEORY_HTML)
        self.tabs = TabGroups()
        self.tabs.add_group("Modes", (campbell_tab, "Campbell diagram"), (stability_tab, "Stability"))
        self.tabs.add_group(
            "Response", (bode_tab, "Bode plot"), (polar_tab, "Polar plot"), (self.waterfall_tab, "Waterfall")
        )
        # The panels scroll when the window is short, rather than squeeze their plots or set its height.
        self.tabs.add_group("Balancing", (ColumnScroll(self.balancing), "Balancing"))
        self.tabs.add_group(
            "Bearings",
            (ColumnScroll(self.bearing_panel), "Fluid-film bearing"),
            (ColumnScroll(self.magnet_panel), "Magnetic bearing"),
        )
        self.tabs.add_group("Rub", (ColumnScroll(self.rub_panel), "Rub"))
        self.tabs.add_group("Theory", (self.theory, "Theory"), (self.matrices, "Equations && matrices"))
        self.tabs.changed.connect(self._catch_up_tab)
        self.tabs.changed.connect(self._refresh_matrices)
        self.tabs.changed.connect(weak.slot(self, lambda self: self._draw_waterfall(only_new=True)))
        self.tabs.changed.connect(self._show_bearing_speed)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(center)
        splitter.addWidget(self.tabs)
        splitter.setSizes([450, 650, 600])  # at 1700 px; narrower, each gives way in proportion
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # --- loop
        self._cost = 1e-9  # s of computing per simulated s (see step)
        self._target_t = 0.0
        self.loop = FrameLoop(self, "Jeffcott rotor", self._tick, self._runs, self._sim_t, self._catch_up)
        # Parameter edits are applied at once; the Campbell diagram and the steady states are worked out
        # once they settle, on a worker thread (seconds, for a turning shaft), and shown when they're done.
        self._analysis_timer = QtCore.QTimer(self)
        self._analysis_timer.setSingleShot(True)
        self._analysis_timer.setInterval(150)
        self._analysis_timer.timeout.connect(self._start_analysis)
        self._job: tuple[RotorSystem, float, int, float] | None = None  # (system, top rpm, ticket, started)
        self._busy_noted = False
        self._job_timer = QtCore.QTimer(self)
        self._job_timer.setInterval(JOB_POLL_MS)
        self._job_timer.timeout.connect(self._poll_analysis)

        self._on_params()
        self.crit_label.setText("<b>Critical speeds</b>: working them out…")
        self._start_analysis()  # `_poll_analysis` shows it (tests that need it at once call `_refresh_analysis`)
        self._draw()

    # ------------------------------------------------------------- panels
    def _preset_box(self) -> TrialBox:
        """The presets as Things to try: a preset with a `predict` asks it before it loads."""
        self.presets = rotor_presets()
        self._matched: int | None = None  # the preset the rotor is (`_show_preset`), None: the reader's own
        self.trials = TrialBox(
            [p.name for p in self.presets],
            "<p>Load a rotor that shows one effect clearly, with a speed range to suit it. It starts from "
            "rest and runs up to its target speed. Change any parameter and it becomes your own rotor.</p>",
            predictions=[p.predict for p in self.presets],
        )
        self.trials.picked.connect(self._on_preset)
        return self.trials

    def preset_name(self) -> str:
        """The preset the rotor is, or CUSTOM (not the combo's text: a preset asking its prediction isn't loaded)."""
        return self.presets[self._matched].name if self._matched is not None else CUSTOM

    def _speed_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Speed")
        form = form_layout(box)
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.speed_slider.setAccessibleName("Target speed")
        self.speed_slider.setRange(0, int(TOP_RPM))
        self.speed_slider.setSingleStep(10)
        self.speed_slider.setPageStep(250)
        self.speed_slider.setToolTip(SPEED_TIP)
        self.target_rpm = spin(0.0, TOP_RPM, 1000.0, 0, " rpm")
        self.target_rpm.setToolTip(SPEED_TIP)
        self.speed_slider.setValue(1000)
        self.speed_slider.valueChanged.connect(weak.slot(self, lambda self, v: self.target_rpm.setValue(float(v))))
        self.target_rpm.valueChanged.connect(self._on_target)
        form.addRow("Target speed:", self.target_rpm)
        form.addRow(self.speed_slider)
        self.ramp = spin(10.0, 20_000.0, 500.0, 0, " rpm/s")
        self.ramp.setToolTip("How fast the speed changes toward the target (run-up or coast-down rate)")
        self.ramp.valueChanged.connect(self._on_ramp)
        form.addRow("Ramp rate:", self.ramp)
        self.sim_speed = QtWidgets.QComboBox()
        for s in SPEEDS:
            self.sim_speed.addItem(speed_text(s), s)
        self.sim_speed.setCurrentIndex(SPEEDS.index(0.25))
        self.sim_speed.setToolTip("Slow motion, so the whirl stays visible at speed")
        form.addRow("Simulation:", self.sim_speed)
        buttons = QtWidgets.QHBoxLayout()
        self.run_button = RunButton()
        self.run_button.setToolTip("<p>Start, pause or resume (Space): from rest it ramps up to the target speed.</p>")
        self.run_button.toggled.connect(self._on_run)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setToolTip("Back to rest, not spinning, waiting for Start")
        self.reset_button.clicked.connect(self.reset)
        self.tap_button = QtWidgets.QPushButton("Tap disc")
        self.tap_button.setToolTip(TAP_TIP)
        self.tap_button.clicked.connect(self._on_tap)
        for b in (self.run_button, self.reset_button, self.tap_button):
            buttons.addWidget(b)
        form.addRow(buttons)
        self.repeat_taps = QtWidgets.QCheckBox("Tap the disc every")
        self.repeat_taps.setToolTip(REPEAT_TAP_TIP)
        self.repeat_taps.toggled.connect(self._on_repeat_taps)
        self.tap_every = spin(0.1, 10.0, 0.5, 1, " s")
        self.tap_every.setSingleStep(0.1)
        self.tap_every.setToolTip(REPEAT_TAP_TIP)
        taps = QtWidgets.QHBoxLayout()
        taps.addWidget(self.repeat_taps)
        self.tap_every.setAccessibleName("Tap the disc every")
        taps.addWidget(self.tap_every)
        taps.addWidget(QtWidgets.QLabel("of simulated time"))
        taps.addStretch(1)
        form.addRow(taps)
        return box

    def _sweep_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Run-up and coast-down")
        form = form_layout(box)
        self.sweep_from = spin(60.0, TOP_RPM, 800.0, 0, " rpm")
        self.sweep_to = spin(60.0, TOP_RPM, 2400.0, 0, " rpm")
        for w in (self.sweep_from, self.sweep_to):
            w.setToolTip(SWEEP_TIP)
            w.valueChanged.connect(self._fit_tracking_range)
        speeds = QtWidgets.QHBoxLayout()
        speeds.addWidget(self.sweep_from, 1)
        speeds.addWidget(QtWidgets.QLabel("to"))
        speeds.addWidget(self.sweep_to, 1)
        form.addRow("From:", speeds)
        self.sweep_from.setAccessibleName("Sweep from")
        self.sweep_to.setAccessibleName("Sweep to")
        buttons = QtWidgets.QHBoxLayout()
        self.sweep_button = QtWidgets.QPushButton("Run up and coast down")
        self.sweep_button.setToolTip(SWEEP_TIP)
        self.sweep_button.clicked.connect(self._on_sweep_button)
        self.clear_sweeps_button = QtWidgets.QPushButton("Clear")
        self.clear_sweeps_button.setToolTip("Take every sweep off the Bode and polar plots")
        self.clear_sweeps_button.clicked.connect(self.clear_sweeps)
        buttons.addWidget(self.sweep_button, 2)
        buttons.addWidget(self.clear_sweeps_button, 1)
        form.addRow(buttons)
        self.hold_sweeps = QtWidgets.QCheckBox("Hold earlier sweeps for comparison")
        self.hold_sweeps.setChecked(True)
        self.hold_sweeps.setToolTip(HOLD_SWEEPS_TIP)
        form.addRow(self.hold_sweeps)
        self.probe_station = QtWidgets.QComboBox()
        self.probe_station.addItems(STATIONS)
        self.probe_station.setCurrentIndex(1)
        self.probe_direction = QtWidgets.QComboBox()
        self.probe_direction.addItems(DIRECTION_NAMES)
        self.probe_direction.setToolTip(
            "<p>What the Bode and polar plots show: a probe reading x or y, or the orbit's major axis. "
            "The phase is counted as a lead, the way the shaft turns: minus the angle the shaft turns from the "
            "keyphasor (the mark on the shaft passing +x) to the probe's positive peak, so a lag is negative, and "
            "on a circular orbit the y probe reads 90° less than the x probe. For the orbit, it is the high spot's "
            "angle from the mark, the way the shaft turns.</p>"
        )
        self.probe_order = QtWidgets.QComboBox()
        self.probe_order.addItems(ORDER_NAMES)
        self.probe_order.setToolTip(
            "<p>Track the 1X vector (once per revolution: the unbalance, the bow, the skew), the 2X vector "
            "(twice per revolution) or the 3X. On a round shaft nothing drives 2X or 3X; an asymmetric "
            "shaft's weight drives 2X, most near half its critical speeds, and a breathing crack's drives "
            "2X and 3X, most near a half and a third of them.</p>"
        )
        for c in (self.probe_station, self.probe_direction, self.probe_order):
            c.currentIndexChanged.connect(self._on_probe)
        probe_row = QtWidgets.QHBoxLayout()
        probe_row.addWidget(self.probe_station, 2)
        probe_row.addWidget(self.probe_direction, 3)
        probe_row.addWidget(self.probe_order, 1)
        form.addRow("Plot:", probe_row)
        self.probe_station.setAccessibleName("Plot station")
        self.probe_direction.setAccessibleName("Plot direction")
        self.probe_order.setAccessibleName("Plot order")
        self.show_peaks = QtWidgets.QCheckBox("Per-revolution peaks (dots)")
        self.show_peaks.setChecked(True)
        self.show_peaks.setToolTip(
            "<p>Also draw the largest displacement over each revolution (the largest orbit radius, for the "
            "orbit). Unlike the 1X vector it includes the free vibration at the natural frequency, so "
            "just past the critical speed it shows the beating.</p>"
        )
        self.show_peaks.toggled.connect(self._on_show_peaks)
        form.addRow(self.show_peaks)
        self.sweep_status = QtWidgets.QLabel()
        self.sweep_status.setWordWrap(True)
        mute(self.sweep_status)
        form.addRow(self.sweep_status)
        return box

    # ------------------------------------------------------------- parameters
    def system(self) -> RotorSystem:
        """The rotor the boxes describe (`RotorEditor.system`)."""
        return self.editor.system()

    def set_system(self, system: RotorSystem) -> None:
        """Show and use `system` (for scripts and tests)."""
        self.editor.load(system)
        self._on_params()

    def load_preset(self, name: str) -> None:
        """Load the preset called `name`: its rotor, speed range and target speed, starting from rest."""
        preset = next(p for p in self.presets if p.name == name)
        self.set_top_rpm(preset.top_rpm)
        self.set_system(preset.system)
        self.set_target_rpm(preset.target_rpm)
        self.reset()

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says: a preset with its changes, starting from rest."""
        t = RotorSetup(**setup)
        self.set_top_rpm(t.base.top_rpm if t.top_rpm is None else t.top_rpm)
        self.set_system(t.system())
        self.set_target_rpm(t.base.target_rpm if t.target_rpm is None else t.target_rpm)
        self.reset()
        self._run(True)

    def practice_state(self) -> RotorSetup:
        """The page's rotor and speeds, as a setup: what a practice spec checks."""
        return RotorSetup(
            self.preset_name(), target_rpm=self.target_rpm.value(), top_rpm=self.top_rpm, rotor=self.system()
        )

    def frame_context(self) -> str:
        """For the frame log (frame_stats): the preset showing."""
        return self.preset_name()

    def _on_preset(self, index: int) -> None:
        self.load_preset(self.presets[index].name)
        self._run(True)  # the reader picked it, as a trial

    def _show_preset(self, system: RotorSystem) -> None:
        """Show the preset this rotor is, with its note, or the reader's own rotor."""
        self._matched = next((i for i, p in enumerate(self.presets) if _same_rotor(p.system, system)), None)
        combo = self.trials.combo
        if self._matched is None:
            combo.unpick()
            self.trials.show_note("Your own rotor: pick a preset to start again from one.")
        else:
            combo.blockSignals(True)
            combo.setCurrentIndex(self._matched + combo.first)
            combo.blockSignals(False)
            self.trials.show_note(self.presets[self._matched].note)

    def set_weights(self, weights: tuple[complex, ...], plane_weights: tuple[PlaneWeight, ...] = ()) -> None:
        """Put these balance weights on the disc (kg·m each) and in these planes, in place of any there."""
        self.editor.set_weights(weights, plane_weights)
        self._on_params()

    def hide_unbalance(self, unbalance: float, angle: float, skew: tuple[float, float] | None = None) -> None:
        """Give the disc this unbalance (kg·m at `angle` rad) without showing it: a balancing exercise.

        With `skew` (τ, its angle in rad), hide a couple unbalance too; otherwise the skew stays as set.
        """
        self.editor.hide_unbalance(unbalance, angle, skew)
        self.set_heavy_shown(False)
        self._on_params()

    def reveal_unbalance(self) -> None:
        """Show the hidden unbalance (and couple) in the panel again, and the heavy spot."""
        if self.editor.reveal_unbalance():
            self.set_heavy_shown(True)
            self._on_params()

    def set_heavy_shown(self, shown: bool) -> None:
        self.view.show_heavy = shown
        self.balancing.show_heavy.blockSignals(True)
        self.balancing.show_heavy.setChecked(shown)
        self.balancing.show_heavy.blockSignals(False)
        self._draw()

    def _on_params(self) -> None:
        self._matrices_speed = -math.inf  # redraw the matrices: the numbers changed
        system = self.system()
        if self.sweeps or self.run_up is not None:
            self.clear_sweeps()  # their steady-state curve is about to change
        self.cancel_measurement()
        self.sim.set_system(system)
        self._clear_spiral()  # a new rotor's spiral starts afresh
        self.bearing_panel.set_rotor(system.bearing_loads if system.films else None)
        self.magnet_panel.set_rotor(system if system.magnets else None)
        self.rub_panel.update_info(system)
        self._show_preset(system)
        self.view.set_system(system)
        self.orbits.set_midspan_shown(abs(system.position - 0.5) > 1e-9)
        self._stride = max(1, round(STORE_DT / self.sim.step_size()))
        self._set_sag(system)
        self._new_waterfall()  # its spectra and free whirl frequencies were of the old rotor
        self.editor.shaft_note.setText(shaft_note(system))
        self._analysis_timer.start()

    def _refresh_analysis(self) -> None:
        """The Campbell diagram, critical speeds and steady states for the current parameters, at once."""
        self._show_analysis(self.sim.system, self.top_rpm, analyse(self.sim.system, self.top_rpm))

    def _start_analysis(self) -> None:
        """Hand the analysis to the worker process; `_poll_analysis` shows it. One job at a time."""
        if self._job is not None:
            system, top_rpm, _, started = self._job
            if system is self.sim.system and top_rpm == self.top_rpm:
                return
            if time.perf_counter() - started < RESTART_S:
                return  # nearly done, likely: `_poll_analysis` starts the new one after it
            WORKER.restart()  # a turning shaft's: don't wait seconds for a rotor no longer shown
        system = self.sim.system
        self._job = (system, self.top_rpm, WORKER.submit(system, self.top_rpm), time.perf_counter())
        self._busy_noted = False
        self._job_timer.start()

    def _poll_analysis(self) -> None:
        if self._job is None:
            self._job_timer.stop()
            return
        system, top_rpm, ticket, started = self._job
        if WORKER.lost(ticket):  # another page restarted the worker
            self._job = None
            self._start_analysis()
            return
        try:
            analysis = WORKER.result(ticket)
        except RuntimeError:
            self._job = None  # the next change starts another
            self._job_timer.stop()
            raise
        if analysis is None:
            if time.perf_counter() - started > BUSY_NOTE_S and not self._busy_noted:
                self._busy_noted = True
                self.crit_label.setText("<b>Critical speeds</b>: working them out for the new parameters…")
            return
        self._job = None
        self._job_timer.stop()
        if system is not self.sim.system or top_rpm != self.top_rpm:
            self._start_analysis()  # the parameters changed while it ran: start again on the new ones
            return
        self._show_analysis(system, top_rpm, analysis)

    def _show_analysis(self, system: RotorSystem, top: float, analysis: Analysis) -> None:
        self._analysed = system
        self.campbell = analysis.campbell
        massless = analysis.massless
        self.onset = analysis.onset
        self.bearing_panel.set_onset(self.onset if system.films else None)
        self.magnet_panel.set_onset(self.onset if system.magnets else None)
        iso = _isotropic(system)
        self.campbell_plot.set_result(self.campbell, top, iso, self.onset)
        self.campbell_plot.set_reference(massless)
        self.stability_map.set_result(self.campbell, self.onset, top)
        self._new_asymmetry_chart()
        crits = critical_speeds(self.campbell, iso)
        self.time.set_criticals([w * RPM for w, _ in crits])
        lines = [f"{w * RPM:,.0f} rpm ({sig(w / (2 * math.pi), 4)} Hz), {whirl_name(r)} whirl" for w, r in crits]
        # On oil films, far stiffer than most shafts, the textbook estimate takes the bearings as rigid.
        jeff = (math.sqrt(system.disc_stiffness / system.disc_mass) if system.films else system.jeffcott_speed()) * RPM
        self.crit_label.setText(
            "<b>Critical speeds</b> (where a whirl frequency equals the spin speed, up to "
            f"{top:,.0f} rpm):<br>"
            + ("<br>".join(lines) if lines else "none")
            + (
                "<br><small>The backward whirl crossings are not critical: an isotropic rotor's unbalance "
                "drives only forward whirl.</small>"
                if iso
                else ""
            )
            + "<br><small>Textbook Jeffcott estimate (massless journals, no tilt"
            + (", rigid bearings" if system.films else "")
            + f"): {jeff:,.0f} rpm</small>"
            + (
                "<br><small>An asymmetric or cracked shaft has two critical speeds for each mode: those of a "
                "round shaft as stiff as each of its axes (on average over a turn).</small>"
                if system.periodic
                else ""
            )
            + (
                "<br><small>With a massless shaft (the Jeffcott model, dashed): "
                + (", ".join(f"{w * RPM:,.0f}" for w, _ in critical_speeds(massless, iso)) or "none")
                + " rpm</small>"
                if massless is not None
                else ""
            )
            + f"<br><b>Stability:</b> {onset_text(self.onset, self.top_rpm)}"
        )
        self.stability_note.setText(self._stability_note())
        self.campbell_note.setText(
            "Each dot is a damped natural frequency at that speed, coloured by its whirl direction "
            f"(modes damped more than ζ = {sig(DAMPED, 6)}, here the journals bouncing on their supports, are "
            "hollow). A critical speed (circled) is where a lightly damped whirl frequency meets the 1X "
            "line: the unbalance, turning once per revolution, drives that mode at its natural "
            "frequency. An isotropic rotor's unbalance drives only its forward whirl."
            + (
                " Crosses mark growing modes (ζ < 0), "
                + (
                    "between the dashed lines."
                    if math.isfinite(self.onset.end)
                    else "above the onset of instability (dashed)."
                )
                if self.onset
                else ""
            )
        )
        self._matrices_speed = -math.inf
        self._refresh_matrices()
        self._steady_rpm = analysis.steady_rpm
        self._steady_orders = analysis.steady_orders
        self._steady_vectors = self._steady_orders[0]
        self._draw_tracking(full=True)
        self.balancing.rotor_changed()
        self.balancing.refresh()

    def _stability_note(self) -> str:
        cause = instability_causes(self.sim.system)
        if self.measure.currentIndex() == 2:
            chart = (
                " The chart shades where the rotor is unstable against the shaft's asymmetry Δ and the speed "
                "(the rest of the rotor as it is), between the two axes' critical speeds (dashed)."
            )
            if self.asym_chart is not None and not self.asym_chart.done:
                chart += f" Working it out: {self.asym_chart._next / self.asym_chart.growth.size:.0%}."
        else:
            chart = f" Modes damped more than ζ = {sig(DAMPED, 6)} are left off."
        onset = onset_text(self.onset, self.top_rpm)
        return f"<small>{cause}{onset}{chart} Below: the full spectrum of the strip chart's station.</small>"

    def _on_measure(self, index: int) -> None:
        if index < 2:
            self.stability_map.set_measure(index == 1)
        self.stability_stack.setCurrentIndex(1 if index == 2 else 0)
        self._new_asymmetry_chart()
        self.stability_note.setText(self._stability_note())

    def _new_asymmetry_chart(self) -> None:
        """Start the asymmetry chart afresh if the rotor (other than its asymmetry) or the speed range changed."""
        system = self.sim.system
        top = max(CHART_DELTA, math.ceil(system.asymmetry * 12.0) / 10.0)
        chart = self.asym_chart
        if (
            chart is not None
            and chart.system.with_(asymmetry=0.0) == system.with_(asymmetry=0.0)
            and chart.omegas[-1] == self.top_rpm / RPM
            and chart.asymmetries[-1] == top
        ):
            return
        self.asym_chart = AsymmetryChart(system, self.top_rpm / RPM, top) if self.measure.currentIndex() == 2 else None
        self.asymmetry_plot.set_chart(self.asym_chart)

    def _work_asymmetry_chart(self) -> None:
        """Fill in some more of the asymmetry chart, if it shows, and draw what is new."""
        chart = self.asym_chart
        if (
            chart is None
            or self.measure.currentIndex() != 2
            or self.tabs.current_page() is not self.spectrum.parentWidget()
        ):
            return
        if not chart.done:
            chart.work(CHART_BUDGET)
        elif self.asymmetry_plot.drawn == chart._next:
            return
        self.asymmetry_plot.set_chart(chart)
        if chart.done or self._frames % SPECTRUM_FRAMES == 0:
            self.stability_note.setText(self._stability_note())

    def _on_bearing_tab(self, kind: int) -> None:
        if self.editor.supports.currentIndex() == kind:
            self._on_params()  # the rotor's bearings are the tab's

    def _catch_up_tab(self) -> None:
        """The Balancing and bearing tabs draw only while they show: bring the one showing up to date."""
        tab = self.tabs.current_page()
        if tab in (self.balancing, self.bearing_panel, self.magnet_panel):
            tab.catch_up()
        elif tab is self.rub_panel:
            self._draw_push()

    def _show_bearing_speed(self) -> None:
        if self.tabs.current_page() is self.bearing_panel:
            self.bearing_panel.set_speed(self.sim.omega * RPM)
        elif self.tabs.current_page() is self.magnet_panel:
            self.magnet_panel.set_speed(self.sim.omega * RPM)

    def _refresh_matrices(self) -> None:
        rpm = self.sim.omega * RPM
        if self.tabs.current_page() is not self.matrices or abs(rpm - self._matrices_speed) < 25.0:
            return
        self._matrices_speed = rpm
        scroll = self.matrices.verticalScrollBar().value()
        self.matrices.setHtml(matrices_html(self.sim.system, self.sim.omega, hidden=tuple(self.editor.hidden or ())))
        self.matrices.verticalScrollBar().setValue(scroll)

    def apply_theme(self) -> None:
        self.view.apply_theme()
        self.orbits.apply_theme()
        self.time.apply_theme()
        self.scrub.apply_theme()
        self.campbell_plot.apply_theme()
        self.stability_map.apply_theme()
        self.spectrum.apply_theme()
        self.waterfall_tab.plot.apply_theme()
        self.bode.apply_theme()
        self.polar.apply_theme()
        self.sweep_plots.apply_theme()
        self.balancing.apply_theme()
        self.bearing_panel.apply_theme()
        self.magnet_panel.apply_theme()
        self.rub_panel.apply_theme()
        self._draw_tracking(full=True)
        self._matrices_speed = -math.inf
        self._refresh_matrices()
        self._draw()

    # ------------------------------------------------------------- speed and actions
    def set_top_rpm(self, rpm: float) -> None:
        """Show speeds up to `rpm`: on the plots, and as the top of the speed controls."""
        if self.top_speed.value() != rpm:
            self.top_speed.setValue(rpm)  # comes back here
            return
        self.top_rpm = rpm
        self.speed_slider.setRange(0, int(rpm))
        for w in (self.target_rpm, self.sweep_from, self.sweep_to):
            w.setMaximum(rpm)  # a target above it comes down (and the speed with it)
        self._set_sag(self.sim.system)
        self._new_waterfall()
        self.bearing_panel.set_top_rpm(rpm)
        self.magnet_panel.set_top_rpm(rpm)
        self._analysis_timer.start()

    def _on_orders(self) -> None:
        self.campbell_plot.set_orders({order for order, box_ in self.order_boxes.items() if box_.isChecked()})

    def set_target_rpm(self, rpm: float) -> None:
        self.target_rpm.setValue(min(max(rpm, 0.0), self.top_rpm))

    def _on_target(self, rpm: float) -> None:
        self.stop_sweep(hold_speed=False)  # the user took over the speed
        self.cancel_measurement()
        self.sim.target = rpm / RPM
        self.speed_slider.blockSignals(True)
        self.speed_slider.setValue(round(rpm))
        self.speed_slider.blockSignals(False)
        self._stride = max(1, round(STORE_DT / self.sim.step_size()))

    def _on_ramp(self, rate: float) -> None:
        self.sim.accel = rate / RPM

    def _on_run(self, on: bool) -> None:
        self.running = on
        if self.running:
            self.rubbed = False
            self.loop.start()
            if self.scrub.t is not None:  # the orbits' dots back where the rotor is
                self.scrub.hide()
                self._draw()
            self.scrub.hide()
        else:
            t, _ = self.history.window(self.time_window.value())
            if t.size:
                self.scrub.show(float(t[-1]), float(t[0]))

    def _on_scrubbed(self, t: float) -> None:
        """Each station's dot on its orbit where the record had it at `t`."""
        ts, hist = self.history.window(self.time_window.value())
        if not ts.size:
            return
        i = min(int(np.searchsorted(ts, t)), ts.size - 1)
        row = self._shown(hist[i : i + 1])[0]
        for k, dot in enumerate(self.orbits.dots):
            dot.setData([row[2 * k]], [row[2 * k + 1]])

    def _run(self, on: bool) -> None:
        self.run_button.set_running(on)
        self._on_run(on)

    def _on_same_scale(self, on: bool) -> None:
        self.orbits.same_scale = on
        self._draw()

    def reset(self) -> None:
        self.stop_sweep()
        self.cancel_measurement()
        self.sim.reset()
        self.rubbed = False
        self.history.clear()
        self._clear_spiral()
        self._new_waterfall()
        self._target_t = 0.0
        self._carry = 0
        self._next_tap = 0.0
        self.running = False
        self.run_button.rest()
        self.scrub.hide()
        self._draw()

    def tap(self) -> None:
        self.sim.tap(0.0, -TAP_IMPULSE)

    def _on_tap(self) -> None:
        """The Tap button: a tap, and the rotor started if it waits."""
        self.tap()
        if not self.running:
            self._run(True)

    def _on_repeat_taps(self, on: bool) -> None:
        self.waterfall_tab.tapping = on  # its note says to tap, or not
        if on:
            self._next_tap = self.sim.t  # the first one now

    # ------------------------------------------------------------- balancing measurements
    def start_measurement(self, omega: float, measured: Callable[[np.ndarray], None]) -> None:
        """Run at `omega` rad/s, settle and average the 1X vectors; then call measured(vectors (4, 2))."""
        self.stop_sweep(hold_speed=False)
        self.cancel_measurement()
        if not self.running:
            self._run(True)
        self.measurement = SteadyMeasurement(self.sim, omega)
        self._measured = measured
        self._sync_target()
        self.loop.start()

    def cancel_measurement(self) -> None:
        if self.measurement is None:
            return
        self.measurement = self._measured = None
        self.balancing.measurement_ended()

    def _advance_measurement(self, duration: float):
        m = self.measurement
        assert m is not None, "advanced only while measuring"
        r = m.advance(duration)
        if m.sim.target * RPM != self.target_rpm.value():
            self._sync_target()
        if m.done:
            measured, self.measurement, self._measured = self._measured, None, None
            assert measured is not None and m.result is not None, "set with the measurement, and when done"
            measured(m.result)
            self.balancing.measurement_ended()
        return r

    def measurement_status(self) -> str:
        m = self.measurement
        if m is None:
            return ""
        rpm = m.omega * RPM
        if m.stage == "approach":
            return f"Going to {rpm:,.0f} rpm…"
        if m.stage == "settle":
            return f"Settling at {rpm:,.0f} rpm ({max(m._settle_end - self.sim.t, 0.0):.1f} s)…"
        return f"Measuring at {rpm:,.0f} rpm: {min(m.tracked, m.revs)} of {m.revs} revolutions"

    # ------------------------------------------------------------- run-up and coast-down
    def _on_sweep_button(self) -> None:
        if self.run_up is None:
            self.start_sweep()
        else:
            self.stop_sweep()

    def start_sweep(self) -> None:
        """Ramp to From, settle, run up to To and coast back down, tracking the 1X vector."""
        start, turn = self.sweep_from.value() / RPM, self.sweep_to.value() / RPM
        if start == turn:
            self.sweep_status.setText("Pick two different speeds.")
            return
        self.stop_sweep()
        self.cancel_measurement()
        if not self.hold_sweeps.isChecked():
            self.clear_sweeps()
        self.run_up = RunUpCoastDown(self.sim, start, turn)
        self._sync_target()
        self._sweep_pairs += 1
        self.sweep_button.setText("Stop sweep")
        self.ramp.setEnabled(False)  # each sweep is labelled with its rate
        if not self.running:
            self._run(True)
        if self.tabs.current_page() is not self.polar.parentWidget():
            self.tabs.show_page(self.bode.parentWidget())
        self._fit_tracking_range()
        self._update_sweep_status()
        self.loop.start()

    def stop_sweep(self, hold_speed: bool = True) -> None:
        """Stop the sweep running (its legs so far stay on the plots), holding the speed it got to."""
        if self.run_up is None:
            return
        self.run_up = None
        self.sweep_button.setText("Run up and coast down")
        self.ramp.setEnabled(True)
        if hold_speed:
            self.sim.target = self.sim.omega
            self._sync_target()
        self._update_sweep_status()

    def clear_sweeps(self) -> None:
        self.stop_sweep()
        self.sweeps = []
        self._sweep_pairs = 0
        self.sweep_plots.clear()
        self._update_sweep_status()
        self._draw_tracking(full=True)

    def _sync_target(self) -> None:
        """Show the simulator's target speed (set by the sweep) in the speed box, without acting on it."""
        rpm = self.sim.target * RPM
        for w in (self.target_rpm, self.speed_slider):
            w.blockSignals(True)
        self.target_rpm.setValue(rpm)
        self.speed_slider.setValue(round(rpm))
        for w in (self.target_rpm, self.speed_slider):
            w.blockSignals(False)
        self._stride = max(1, round(STORE_DT / self.sim.step_size()))

    def _advance_sweep(self, duration: float):
        run_up = self.run_up
        assert run_up is not None, "advanced only while sweeping"
        before = len(run_up.sweeps)
        r = run_up.advance(duration)
        for sweep in run_up.sweeps[before:]:
            self.sweeps.append(sweep)
            self.sweep_plots.add(sweep, self._sweep_pairs - 1)
        if run_up.sim.target * RPM != self.target_rpm.value():
            self._sync_target()
        if run_up.done:
            self.run_up = None
            self.sweep_button.setText("Run up and coast down")
            self.ramp.setEnabled(True)
        self._update_sweep_status()
        return r

    def _update_sweep_status(self) -> None:
        r = self.run_up
        if r is None:
            n = len(self.sweeps)
            text = f"{n} sweep{'s' if n != 1 else ''} on the Bode and polar plots." if n else ""
        elif r.stage == "approach":
            text = f"Going to {r.start * RPM:,.0f} rpm…"
        elif r.stage == "settle":
            text = f"Settling at {r.start * RPM:,.0f} rpm…"
        else:
            leg = r.sweeps[-1]
            text = f"{leg.label}: {self.sim.omega * RPM:,.0f} rpm, {len(leg)} revolutions tracked"
        self.sweep_status.setText(text)

    def _on_probe(self) -> None:
        order = self._order()
        self.sweep_plots.order = order
        self.bode.set_order(order)
        self._fit_tracking_range()
        self._draw_tracking(full=True)

    def _on_show_peaks(self, on: bool) -> None:
        self.sweep_plots.set_show_peaks(on)

    def _probe(self) -> tuple[int, str]:
        return self.probe_station.currentIndex(), DIRECTIONS[self.probe_direction.currentIndex()]

    def _order(self) -> int:
        """The harmonic the Bode and polar plots track: 1 (1X), 2 (2X) or 3 (3X)."""
        return self.probe_order.currentIndex() + 1

    def _tracking_range(self) -> tuple[float, float]:
        """Speeds (rpm) the Bode and polar plots show: the sweep's, and those of the sweeps on them."""
        speeds = [self.sweep_from.value(), self.sweep_to.value()]
        for sweep in self.sweeps:
            if len(sweep):
                speeds += [min(sweep.speed) * RPM, max(sweep.speed) * RPM]
        return min(speeds), max(speeds)

    def _fit_tracking_range(self) -> None:
        lo, hi = self._tracking_range()
        pad = 0.04 * max(hi - lo, 1.0)
        self.bode.amp.setXRange(max(lo - pad, 0.0), hi + pad, padding=0)
        self._draw_tracking(full=True)

    def _steady_probe(self) -> tuple[np.ndarray, np.ndarray]:
        """The steady-state 1X (or 2X, 3X) values at the chosen station and direction, and their unwrapped phase."""
        station, direction = self._probe()
        vectors = self._steady_orders[self._order() - 1]
        values = probe(vectors[:, station], direction)
        return values, phase_degrees(values)

    def _draw_tracking(self, full: bool = False) -> None:
        """The Bode and polar plots: the steady-state curve and summary (if `full`), the sweeps, the marker."""
        if not self._steady_rpm.size:
            return
        rpm = self._steady_rpm
        values, phase = self._steady_probe()
        if full:
            self.bode.set_steady(rpm, values, phase)
            lo, hi = self._tracking_range()
            shown = (rpm >= lo) & (rpm <= hi)
            self.polar.set_steady(rpm[shown], values[shown], nice_marks(lo, hi))
        self.sweep_plots.redraw(*self._probe(), rpm, phase, full)
        here = self.sim.omega * RPM
        v = complex(np.interp(here, rpm, values.real), np.interp(here, rpm, values.imag))
        self.bode.set_here(here, abs(v), float(np.interp(here, rpm, phase)))
        self.polar.set_here(v)
        if full or (self.run_up is not None and self.run_up.stage in ("out", "back")):
            text = self._summary_html(rpm, values)
            if text != self.sweep_summary.text():
                self.sweep_summary.setText(text)

    def _summary_html(self, rpm: np.ndarray, values: np.ndarray) -> str:
        """The steady-state peak over the sweeps' speeds, and each sweep's peak against it."""
        station, direction = self._probe()
        lo, hi = self._tracking_range()
        inside = np.flatnonzero((rpm >= lo) & (rpm <= hi))
        if not inside.size:
            return ""
        i = inside[np.argmax(np.abs(values[inside]))]
        ss_amp, ss_rpm = float(abs(values[i])), float(rpm[i])
        wc = ss_rpm / RPM
        where = f"{STATIONS[station].lower()}, {DIRECTION_NAMES[self.probe_direction.currentIndex()].lower()}"
        order = self._order()
        system = self.sim.system
        if order > 1:
            where += f", {order}X"
            if not ss_amp:
                return f"<b>Steady state</b> ({where}): nothing drives {order}X" + (
                    " on a round shaft."
                    if not system.periodic
                    else " on an asymmetric shaft (only a breathing crack's stiffness changes at 3X)."
                    if order == 3 and system.spacing == 2
                    else " without gravity: tick <i>Horizontal rotor</i>."
                )
        head = f"<b>Steady state</b> ({where}): peak {fmt_len(ss_amp)} at {ss_rpm:,.0f} rpm"
        crits = self.campbell.criticals if self.campbell else []
        zetas = [(abs(w - wc), z) for w, _, z in crits]
        near = min(crits, key=lambda c: abs(c[0] - order * wc), default=None)
        if order > 1:
            if near is not None and abs(near[0] - order * wc) < 0.15 * order * wc:
                head += (
                    f", about {'half' if order == 2 else 'a third of'} the critical speed "
                    f"{near[0] * RPM:,.0f} rpm" + (": the <i>gravity critical</i>" if order == 2 else "")
                )
        elif zetas and min(zetas)[0] < 0.05 * wc:
            z = min(zetas)[1]
            head += f"; its mode has ζ = {sig(z)}, ζ² = {sig(z * z, 2)}"
        if not self.sweeps:
            return head + (f"<br><small>Run up and coast down to track the {order}X response over this curve.</small>")
        rows = []
        for c in self.sweep_plots.curves:
            sweep = c.sweep
            amp, w = sweep.peak(station, direction, order)
            if not len(sweep):
                continue
            color = colors.mode[SWEEP_COLORS[c.color % len(SWEEP_COLORS)]]
            line = "━━" if sweep.up else "╍╍"
            alpha = sweep.rate / RPM
            rows.append(
                f"<tr><td><span style='color:{color}'><b>{line}</b></span> {sweep.label}</td>"
                f"<td align='right'>{sig(alpha / wc**2, 2)}</td><td align='right'>{fmt_len(amp)}</td>"
                f"<td align='right'>{100 * amp / ss_amp:.0f}%</td><td align='right'>{w * RPM:,.0f}</td>"
                f"<td align='right'>{(w - wc) * RPM:+,.0f}</td></tr>"
            )
        return (
            head + "<table cellspacing='0' cellpadding='2'><tr><th align='left'>Sweep</th>"
            f"<th>Ω̇/ω<sub>c</sub>²</th><th>{order}X peak</th><th>of steady</th><th>at rpm</th><th>vs steady</th></tr>"
            + "".join(rows)
            + "</table>"
        )

    # ------------------------------------------------------------- loop

    def _runs(self) -> bool:
        return self.running

    def _sim_t(self) -> float:
        return self.sim.t

    def _catch_up(self) -> None:
        """As the loop starts or resumes: play on from where the simulation is."""
        self._target_t = self.sim.t

    def _tick(self, dt: float) -> None:
        speed = float(self.sim_speed.currentData())
        self.step(dt * (max(speed, MEASURE_SPEED) if self.measurement is not None else speed), FRAME_BUDGET)

    def step(self, duration: float, budget: float | None = None) -> None:
        """Advance by `duration` s of simulated time and redraw.

        With a `budget` (s of computing), advance no further than it allows at the
        cost seen so far: where the simulation is slow (the films' whole force),
        the page falls behind real time rather than stalling.
        """
        self._target_t = min(self._target_t + duration, self.sim.t + 0.1 + duration)
        if budget is not None and self._target_t - self.sim.t > budget / self._cost:
            self._target_t = self.sim.t + budget / self._cost
        if self.repeat_taps.isChecked() and self.measurement is None and self.sim.t >= self._next_tap:
            self.tap()
            self._next_tap = self.sim.t + self.tap_every.value()
        start, began = time.perf_counter(), self.sim.t
        if self.run_up is not None:
            r = self._advance_sweep(self._target_t - self.sim.t)
        elif self.measurement is not None:
            r = self._advance_measurement(self._target_t - self.sim.t)
        else:
            r = self.sim.advance(self._target_t - self.sim.t)
        if self.sim.t > began:  # s of computing per simulated s, smoothed
            cost = (time.perf_counter() - start) / (self.sim.t - began)
            self._cost = 0.8 * self._cost + 0.2 * cost if self._cost > 1e-9 else cost
        if r.rub is not None and r.t.size:
            self._heat = float(r.rub[:, 3].mean())
        if r.t.size:
            # Keep every stride-th sample, continuing the stride across advances.
            first = (self._stride - self._carry - 1) % self._stride
            pick = np.arange(first, r.t.size, self._stride)
            self._carry = (r.t.size - 1 - pick[-1]) if pick.size else self._carry + r.t.size
            if pick.size:
                cols = self._columns(r.q[pick], r.phase[pick], r.omega[pick])
                push = np.zeros(pick.size)
                if r.rub is not None:  # the largest push since the last sample kept, so no hit is missed
                    push = np.maximum.reduceat(r.rub[:, 2], np.concatenate([[0], pick[:-1] + 1]))
                cols = np.column_stack([cols, push])
                self.history.extend(r.t[pick], cols)
                vib = self._vibration(cols)
                if self.sim.thermal:  # at a steady speed only: a new speed starts it afresh
                    if self.sim.omega != self.sim.target:
                        self._clear_spiral()
                    elif self.sim.t >= self._spiral_next and self.sim.omega > 0.0:
                        self._record_spiral()
                self.waterfall_tab.feed(r.t[pick], vib[:, 0::2] + 1j * vib[:, 1::2], cols[:, OMEGA])
        touched = self._journal_touched()
        if self.running and (
            touched or np.abs(self.sim.q[[XA, XD, XB, YA, YD, YB]]).max() > RUB * self.sim.system.length
        ):
            self.stop_sweep()
            self.cancel_measurement()
            self._run(False)  # pause
            self.rubbed = True
            self.touched = touched
        self._draw()
        self._refresh_matrices()

    def _journal_touched(self) -> tuple[str, float] | None:
        """The journal ("A" or "B") that has moved its oil film's whole clearance off centre, or onto its
        magnetic bearing's backup bearing, and that clearance."""
        q, s = self.sim.q, self.sim.system
        for name, (jx, jy), b in (("A", (XA, YA), s.bearing_a), ("B", (XB, YB), s.bearing_b)):
            clearance = (
                b.film.clearance if b.film is not None else b.magnet.touchdown if b.magnet is not None else math.inf
            )
            if math.hypot(q[jx], q[jy]) >= clearance:
                return name, clearance
        return None

    def _columns(self, q: np.ndarray, phase: np.ndarray, omega: np.ndarray) -> np.ndarray:
        s = self.sim.system
        mid = s.deflection(q, np.array([0.5 * s.length]))[:, 0, :]
        return np.column_stack([q[:, XA], q[:, YA], q[:, XD], q[:, YD], mid, q[:, XB], q[:, YB], phase, omega])

    def _draw(self) -> None:
        sim = self.sim
        w = abs(sim.omega)
        trail_time = min(ORBIT_MAX_TIME, ORBIT_REVS * 2 * math.pi / w) if w > 0 else ORBIT_MAX_TIME
        _, y = self.history.window(trail_time)
        shown = self._shown(y)
        trails = [shown[:, 2 * i : 2 * i + 2] for i in range(len(STATIONS))]
        keys = [np.empty((0, 2))] * len(STATIONS)
        if y.shape[0] > 1:
            turns = np.floor(y[:, PHASE] / (2 * math.pi))
            idx = np.flatnonzero(np.diff(turns) != 0) + 1
            keys = [t[idx] for t in trails]
        q_vib = sim.q - self._static_sag(sim.omega) if sim.system.gravity else sim.q
        q_shown = sim.q if self.show_sag.isChecked() else q_vib
        here = self._columns(q_shown[None, :], np.array([sim.phase]), np.array([sim.omega]))[0]
        peak = max([float(np.abs(t).max()) for t in trails if t.size] + [float(np.abs(here[:8]).max())])
        ring = self._ring(q_shown)
        self.view.ring = ring
        self.view.update_state(q_shown, sim.phase, trails, peak)
        steady = sim.omega == sim.target and w > 0
        here = self._columns(q_vib[None, :], np.array([sim.phase]), np.array([sim.omega]))[0]
        r_disc = math.hypot(here[2], here[3])
        heavy = self._heavy_angle()
        centre = (0.0, 0.0)
        if self._sag is not None and self.show_sag.isChecked():
            sag = self._sag_at(np.array([sim.omega]))[0]
            centre = (float(sag[2]), float(sag[3]))
        self.orbits.update_orbits(trails, keys, heavy if w > 0 else None, centre, ring)

        t, hist = self.history.window(self.time_window.value())
        st = self.station.currentIndex()
        self.time.update_curves(
            t, self._shown(hist)[:, 2 * st : 2 * st + 2], hist[:, OMEGA] * RPM, self.time_window.value()
        )
        if self.tabs.current_page() is self.rub_panel:
            self._draw_push(t, hist)
        self.campbell_plot.set_speed(sim.omega * RPM)
        self.stability_map.set_speed(sim.omega * RPM)
        self._show_bearing_speed()
        self.asymmetry_plot.set_here(sim.omega * RPM, sim.system.asymmetry)
        self._work_asymmetry_chart()
        self._draw_tracking()
        self.balancing.update_status()
        self._frames += 1
        if self.tabs.current_page() is self.spectrum.parentWidget() and (
            self._frames % SPECTRUM_FRAMES == 0 or not self.running
        ):
            self._draw_spectrum()
        if self._frames % WATERFALL_FRAMES == 0 or not self.running:
            self._draw_waterfall(only_new=True)

        # Readout.
        rpm = sim.omega * RPM
        parts = [f"<b>Speed {rpm:,.0f} rpm</b> ({sig(sim.omega / (2 * math.pi))} Hz)"]
        if not steady and sim.omega != sim.target:
            parts[0] += f" → {sim.target * RPM:,.0f} rpm"
        crits = [c for c, _ in critical_speeds(self.campbell, _isotropic(sim.system))] if self.campbell else []
        if crits and w > 0:
            parts.append(f"Ω / first critical = {sim.omega / crits[0]:.2f}")
        disc = self._vibration(y)[:, 2:4]
        if disc.size:
            text = f"disc whirling up to {fmt_len(float(np.hypot(disc[:, 0], disc[:, 1]).max()))}"
            if sim.system.gravity:
                text += " about its sag"
            if w > 0 and disc.shape[0] > 2:
                # Signed area swept (x dy - y dx): positive turns x toward y, the way the shaft spins.
                swept = float(np.sum(disc[:-1, 0] * np.diff(disc[:, 1]) - disc[:-1, 1] * np.diff(disc[:, 0])))
                text += ", whirling " + ("forward" if swept > 0 else "backward")
            parts.append(text)
        rubbing = sim.system.stator is not None and y.shape[0] and y[:, PUSH].max() > 0.0
        if rubbing:
            parts.append(f"<span style='color:{colors.force}'>rubbing the stator ring</span>")
        # Only with this rotor's own onset: until its analysis arrives the last rotor's is still here, and above
        # it this would work out a turning shaft's Hill modes every frame (12 ms for the cracked one).
        growing = (
            self.onset is not None
            and self._analysed is sim.system
            and sim.omega >= self.onset.omega
            and least_damped(sim.system, w)[0] < 0
        )
        # (Not where a turning shaft's weight drives 2X: the disc's position then says little of the 1X.)
        if (
            heavy is not None
            and steady
            and not growing
            and not rubbing
            and not (sim.system.periodic and sim.system.gravity)
            and r_disc > 1e-3 * max(abs(sim.system.mass_unbalance) / sim.system.disc_mass, MIN_PEAK)
        ):
            lag = math.degrees(heavy - math.atan2(here[3], here[2])) % 360.0
            parts.append(f"heavy spot leads the high spot by {lag:.0f}°")
        if growing:
            zeta, f, _ = least_damped(sim.system, w)
            if f == 0.0:  # a magnetic bearing too weak for its magnets
                parts.append(
                    f"<span style='color:{colors.force}'>unstable: the journals fall toward the magnets</span>"
                )
            elif w > 0:
                grow = -zeta * 2 * math.pi * f / math.sqrt(max(1 - zeta * zeta, 1e-12))
                # With the films' whole force that is only how it starts: oil whirl settles to a limit cycle.
                what = "linearized, the equilibrium is unstable" if sim.system.whole_films else "unstable"
                parts.append(
                    f"<span style='color:{colors.force}'>{what}: a whirl at {sig(f)} Hz "
                    f"({2 * math.pi * f / w:.2f}×) grows ×e every {sig(1 / grow, 2)} s</span>"
                )
        if self.rubbed:
            where = (
                f"journal {self.touched[0]} landed on its backup bearing, {fmt_len(self.touched[1])} off "
                "centre (half the magnets' air gap). A real machine drops onto these when its magnetic "
                "bearings lose control; the model here is linear, and has no contact. "
                if self.touched and sim.system.magnets
                else f"journal {self.touched[0]} reached the wall of its bearing, a clearance of "
                f"{fmt_len(self.touched[1])} off centre. "
                "A real oil film stiffens sharply there and keeps the journal off the wall; the film here is "
                "linearized (about where the journal rests), so it would let it through. Tick <i>Nonlinear "
                "film</i> to step the film's whole force. "
                if self.touched
                else f"the whirl passed {fmt_len(RUB * sim.system.length)} ({RUB:.0%} of the span), "
                f"{fmt_len(RUB * sim.system.length - sim.system.stator.clearance)} past the stator ring's "
                "clearance: its mount would have given way long before, and the machine been wrecked. "
                if sim.system.stator is not None
                else f"the whirl passed {fmt_len(RUB * sim.system.length)} ({RUB:.0%} of the span), where a real "
                "rotor would rub its seals or wreck its bearings. "
                + (
                    "The oil films hold the journals inside their clearance, but not the shaft's bending, so "
                    "oil whip grows on. "
                    if sim.system.whole_films
                    else "The model is linear, so it would grow for ever. "
                )
            )
            parts.append(
                f"<span style='color:{colors.force}'><b>Paused:</b> {where}"
                + (
                    "Set the target outside the unstable range"
                    if self.onset is not None and math.isfinite(self.onset.end)
                    else "Lower the target below the onset"
                )
                + " and Reset.</span>"
            )
        self.readout.setText(" · ".join(parts))

    def _draw_push(self, t: np.ndarray | None = None, hist: np.ndarray | None = None) -> None:
        """The Rub tab's plot of the ring's push, over the strip chart's window (`t`, `hist` from the history),
        and its Newkirk spiral."""
        if t is None or hist is None:
            t, hist = self.history.window(self.time_window.value())
        self.rub_panel.update_contact(t, hist[:, PUSH], self.time_window.value())
        if self.sim.thermal:
            vectors, bows = self._spiral
            self.rub_panel.update_spiral(np.array(vectors, dtype=complex), np.array(bows, dtype=complex), self._heat)

    def _clear_spiral(self) -> None:
        self._spiral = ([], [])
        self._spiral_next = self.sim.t

    def _record_spiral(self) -> None:
        """Add the Newkirk spiral's next point: the disc's 1X vector over the last revolution, and the thermal
        bow, both at the probe's direction."""
        _, y = self.history.window(2.0 * math.pi / abs(self.sim.omega))
        if y.shape[0] < 8:
            return
        self._spiral_next = self.sim.t + SPIRAL_DT
        disc = self._vibration(y)[:, 2:4]
        X = 2.0 * np.mean(disc * np.exp(-1j * y[:, PHASE])[:, None], axis=0)
        direction = self._probe()[1]
        b = self.sim.thermal_bow
        vectors, bows = self._spiral
        vectors.append(complex(probe(X, direction)))
        bows.append(complex(probe(np.array([b, -1j * b]), direction)))  # the bow's runout: x = Re(b e^{iφ})
        if len(vectors) > SPIRAL_POINTS:
            del vectors[1::2], bows[1::2]

    def _ring(self, q: np.ndarray) -> tuple[float, float, float] | None:
        """The stator ring as drawn with the rotor at q (shown with its sag or without): its centre, pushed out
        by the disc where it overlaps it, and its clearance (m)."""
        stator = self.sim.system.stator
        if stator is None:
            return None
        cx, cy = stator.centre
        if not self.show_sag.isChecked() and self.sim.system.gravity:  # drawn about the sag: move the ring too
            sag = self._static_sag(self.sim.omega)
            cx, cy = cx - sag[XD], cy - sag[YD]
        dx, dy = q[XD] - cx, q[YD] - cy
        rho = math.hypot(dx, dy)
        over = rho - stator.clearance
        if over > 0.0:
            cx, cy = cx + over * dx / rho, cy + over * dy / rho
        return cx, cy, stator.clearance

    def _heavy_angle(self) -> float | None:
        """The heavy spot's angle now (rad), or None if it is hidden or there is no unbalance."""
        net = self.sim.system.mass_unbalance
        if not self.view.show_heavy or not net:
            return None
        return self.sim.phase + math.atan2(net.imag, net.real)

    # ------------------------------------------------------------- static sag
    def _set_sag(self, system: RotorSystem) -> None:
        """The static sag at the history's columns, at a grid of speeds (one row if it does not vary)."""
        self.show_sag.setEnabled(system.gravity > 0.0)
        if not system.gravity:
            self._sag = self._sag_q = None
            return
        speeds = np.linspace(0.0, self.top_rpm / RPM, SAG_SPEEDS if system.sag_varies else 1)
        qs = system.static_deflection(speeds)
        self._sag = (speeds, self._columns(qs, speeds, speeds)[:, :8])
        self._sag_q = (speeds, qs)

    def _static_sag(self, omega: float) -> np.ndarray:
        """The static sag q_s (n_dof,) at the speed omega, which each frame draws about.

        An asymmetric or cracked shaft's (its periodic response's mean, a harmonic balance) took ~40 ms to
        work out afresh, a frame's worth, every frame the speed changed: it comes from _set_sag's grid instead,
        as the history's columns do (within 0.05 µm on the cracked preset). Any other rotor's is cached and
        cheap, and exact; so is one past the grid's top speed, or before the grid is set.
        """
        if self._sag_q is None or not self.sim.system.periodic:  # cheap (cached) elsewhere: exact
            return self.sim.system.static_deflection(omega)
        speeds, qs = self._sag_q
        w = abs(omega)
        if speeds.size == 1:
            return qs[0]
        if w > speeds[-1]:
            return self.sim.system.static_deflection(omega)
        i = min(int(np.searchsorted(speeds, w, side="right")) - 1, speeds.size - 2)
        f = (w - speeds[i]) / (speeds[i + 1] - speeds[i])
        return (1.0 - f) * qs[i] + f * qs[i + 1]

    def _sag_at(self, omega: np.ndarray) -> np.ndarray:
        """The static sag (len(omega), 8) in the history's station columns."""
        assert self._sag is not None, "set with the rotor (_set_sag)"
        speeds, cols = self._sag
        if speeds.size == 1:
            return np.broadcast_to(cols[0], (np.size(omega), 8))
        return np.column_stack([np.interp(np.abs(omega), speeds, c) for c in cols.T])

    def _vibration(self, y: np.ndarray) -> np.ndarray:
        """History rows' station columns less the static sag: the vibration."""
        return y[:, :8] if self._sag is None else y[:, :8] - self._sag_at(y[:, OMEGA])

    def _shown(self, y: np.ndarray) -> np.ndarray:
        """History rows' station columns as drawn: with the sag or without, as ticked."""
        return y[:, :8] if self.show_sag.isChecked() else self._vibration(y)

    def _draw_spectrum(self) -> None:
        """The full spectrum at the strip chart's station over the strip chart's window."""
        t, hist = self.history.window(self.time_window.value())
        st = self.station.currentIndex()
        freqs, amps = full_spectrum(t, self._vibration(hist)[:, 2 * st : 2 * st + 2])
        omega = self.sim.omega
        self.spectrum.update_spectrum(
            freqs,
            amps,
            omega,
            whirl_modes(self.sim.system, omega),
            DAMPED,
            top=(3 if self.sim.system.spacing == 1 else 2) if self.sim.system.periodic else 1,
        )
        self.spectrum_note.setText(
            f"Full spectrum of the {STATIONS[st].lower()} orbit over the last {sig(self.time_window.value(), 6)} s "
            "(the strip chart's station and window): forward whirl on the right, backward on the left."
        )

    # ------------------------------------------------------------- waterfall
    def _new_waterfall(self) -> None:
        """Start the spectra afresh: for a new rotor, speed range or block length."""
        self.waterfall_tab.restart(self.sim.system, self.top_rpm)
        self._draw_waterfall()

    def _draw_waterfall(self, only_new: bool = False) -> None:
        """Draw the waterfall if its tab is showing (with `only_new`, only if blocks came in since)."""
        if self.tabs.current_page() is self.waterfall_tab:
            self.waterfall_tab.draw(only_new)


def _isotropic(system: RotorSystem) -> bool:
    """True if the supports act alike in x and y, so that unbalance drives only forward whirl.

    A magnetic bearing does (the same controller on both axes), though its controller's
    states keep the rotor from being time-invariant in the rotating frame (isotropic_supports).
    """
    if not system.magnets:
        return system.isotropic_supports
    plain = [Bearing(mass=b.mass) if b.magnet is not None else b for b in (system.bearing_a, system.bearing_b)]
    return system.with_(bearing_a=plain[0], bearing_b=plain[1]).isotropic_supports


def _same_rotor(a: RotorSystem, b: RotorSystem) -> bool:
    """True if the two rotors are the same up to round-off (the panels' values pass through decimals)."""

    def close(u, v) -> bool:
        if u is None or v is None:
            return u is v
        if isinstance(u, (Bearing, JournalBearing, MagneticBearing, Crack, Stator)):
            # A film's own load is not used on the rotor: each bearing carries its share of the weight.
            return type(u) is type(v) and all(
                close(getattr(u, f.name), getattr(v, f.name))
                for f in fields(u)
                if not (isinstance(u, JournalBearing) and f.name == "load")
            )
        if isinstance(u, tuple):
            return len(u) == len(v) and all(close(x, y) for x, y in zip(u, v))
        return cmath.isclose(u, v, rel_tol=1e-9, abs_tol=1e-15)

    # A massless shaft is the same however it is meshed.
    skip = {"elements", "lumped"} if a.density == b.density == 0.0 else set()
    return all(close(getattr(a, f.name), getattr(b, f.name)) for f in fields(RotorSystem) if f.name not in skip)
