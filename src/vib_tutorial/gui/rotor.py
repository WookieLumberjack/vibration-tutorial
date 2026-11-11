"""Jeffcott rotor page: a disc on a flexible shaft in flexible bearings, spun up through its criticals.

Left: the speed controls, the run-up and coast-down sweep, and the rotor's
parameters. Centre: a 3D view of the whirling shaft (projected onto a 2D
pyqtgraph scene, no OpenGL), the orbits at the bearings, the disc and midspan,
and a strip chart. Right: the Campbell diagram, the Bode and polar plots of the
1X response (steady state and tracked sweeps), the stability map and full
spectrum, the matrices of the equations of motion at the current speed, and the
theory notes.
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
    CRACK_DEPTH_MAX,
    MODE_LIMIT,
    STEEL,
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
    lag_degrees,
    probe,
)
from ..core.rotor_waterfall import Waterfall
from . import frame_stats, rotor_practice, settle_plots, weak
from .cms_time import SampleBuffer
from .layouts import ColumnScroll, FlowLayout, form_layout, group
from .lesson import RunButton
from .notes_view import NotesBrowser
from .panels import spin
from .rotor_balance import BalancingPanel
from .rotor_bearing import BearingPanel
from .rotor_bode import SWEEP_COLORS, BodePlots, PolarPlot, SweepPlots, nice_marks
from .rotor_magnet import MagnetPanel
from .rotor_notes import THEORY_HTML, matrices_html, whirl_name
from .rotor_plots import DAMPED, MIN_PEAK, ORDERS, CampbellPlot, OrbitPlots, TimePlots, critical_speeds
from .rotor_practice import RotorSetup
from .rotor_rub import SPIRAL_POINTS, RubPanel
from .rotor_stability import MEASURES, AsymmetryPlot, FullSpectrum, StabilityMap
from .rotor_tips import (
    ASYMMETRY_TIP,
    BLOCK_TIP,
    BOW_TIP,
    CRACK_TIP,
    ELEMENTS_TIP,
    FILM_TIP,
    GRAVITY_TIP,
    HOLD_SWEEPS_TIP,
    INTERNAL_TIP,
    KXY_TIP,
    MAGNET_TIP,
    NONLINEAR_FILM_TIP,
    ORDER_TIPS,
    REPEAT_TAP_TIP,
    SAG_TIP,
    SHAFT_MASS_TIP,
    SKEW_TIP,
    SPEED_TIP,
    SWEEP_TIP,
    SWIRL_TIP,
    TAP_TIP,
)
from .rotor_view import RotorView
from .rotor_waterfall import AGAINST, SCALES, WaterfallPlot, against_speed, baselines
from .style import colors
from .tab_groups import TabGroups
from .theming import mute

FRAME_MS = 16
SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0)  # simulation speed, × real time
TOP_RPM = 10_000.0  # the default speed range
MAX_TOP_RPM = 60_000.0  # the widest it can be set to
PROBLEMS = rotor_practice.PROBLEMS  # the page's practice problems, kept beside it in rotor_practice
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
SUPPORTS = ("Springs and dampers", "Oil-film journal bearings", "Magnetic bearings")
SPRINGS, FILMS, MAGNETS = range(len(SUPPORTS))
DASH = QtCore.Qt.PenStyle.DashLine
WORKER = AnalysisWorker()  # shared by every rotor page: one process
JOB_POLL_MS = 30  # how often the page checks for a finished analysis
BUSY_NOTE_S = 0.3  # s an analysis runs before the page says it is working on it
RESTART_S = 0.5  # s a superseded analysis has run past which the worker drops it rather than see it out
BEARING_WIDTHS = (56, 56, 50, 50, 46, 56, 40)  # px, the bearing table's columns: kx, ky, cx, cy, m, kxy, λ
BEARING_GAP = 0.02  # of the span: the closest the disc may be put to a bearing
RUB = 0.02  # of the span: a whirl this large pauses the simulation (a real rotor would rub)
GRAVITY = 9.81  # m/s²
SAG_SPEEDS = 201  # speeds the static sag is worked out at, with internal damping, and interpolated
SPECTRUM_FRAMES = 6  # the full spectrum is recomputed every this many frames
SPIRAL_DT = 0.05  # s of simulated time between the Newkirk spiral's points
MEASURE_SPEED = 2.0  # × real time, at least, while a balancing measurement runs
FRAME_BUDGET = 0.012  # s of computing per frame: past it (the films' whole force is slow) the page runs slower
WATERFALL_FRAMES = 15  # new spectra are drawn at most every this many frames (each repaint fills every line)
WATERFALL_RANGE = 2.0  # the waterfall keeps frequencies up to this many times the top speed
CHART_BUDGET = 0.012  # s per frame spent filling in the asymmetry chart while it shows
CHART_DELTA = 0.5  # the asymmetry chart's top Δ (taller only for a rotor past it)
ORDER_NAMES = ("1X", "2X", "3X")  # what the Bode and polar plots can track


class RotorPage(QtWidgets.QWidget):
    """Parameters and speed on the left; 3D view, orbits and time on the centre; Campbell, matrices, theory."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

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
        self._weights: tuple[complex, ...] = ()  # balance weights on the disc (kg·m)
        self._plane_weights: tuple[PlaneWeight, ...] = ()  # weights in balancing planes
        self._hidden: dict[str, float] | None = None  # RotorSystem fields kept out of the panels (SI)
        self.measurement: SteadyMeasurement | None = None  # a balancing measurement running, if any
        self._measured: Callable[[np.ndarray], None] | None = None  # its callback, given the mean 1X vectors (4, 2)
        self._frames = 0
        self._next_tap = 0.0  # simulated time of the next repeated tap
        self.waterfall: Waterfall | None = None  # spectra block by block, made by _new_waterfall
        self._spiral: tuple[list[complex], list[complex]] = ([], [])  # the 1X vectors and thermal bows recorded
        self._spiral_next = 0.0  # simulated time of the spiral's next point
        self._heat = 0.0  # W: the rub's heat over the last advance
        self._waterfall_drawn = False  # the waterfall tab shows the latest blocks

        # --- parameters
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(self._preset_box())
        lv.addWidget(self._speed_box())
        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lv.addWidget(self.readout)
        lv.addWidget(self._sweep_box())
        lv.addWidget(self._rotor_box())
        lv.addWidget(self._bearing_box())
        self.crit_label = QtWidgets.QLabel()
        self.crit_label.setWordWrap(True)
        lv.addWidget(self.crit_label)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)

        # --- centre
        self.view = RotorView()
        self.orbits = OrbitPlots()
        self.time = TimePlots()
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
            "The 1X vector at the chosen station: its length is the amplitude, its angle clockwise from "
            "the right the phase lag. Through a lightly damped critical speed the steady-state vector "
            "traces a circle, lagging 90° at its lowest point. With a bowed shaft the curve starts from the "
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
        self.waterfall_plot = WaterfallPlot()
        self.wf_station = QtWidgets.QComboBox()
        self.wf_station.addItems(STATIONS)
        self.wf_station.setCurrentIndex(1)
        self.wf_against = QtWidgets.QComboBox()
        self.wf_against.addItems(AGAINST)
        self.wf_against.setToolTip(
            "<p>Against speed (a <i>cascade</i>): the latest run-up's or coast-down's spectra, each at its "
            "speed. Against time (a <i>waterfall</i>): every spectrum kept, steady running and taps "
            "included.</p>"
        )
        self.wf_scale = QtWidgets.QComboBox()
        self.wf_scale.addItems(SCALES)
        self.wf_scale.setToolTip(
            "<p>Log: each line's height is its amplitude in decades above a floor 1000 times (60 dB) "
            "below the tallest peak, so small peaks show beside the 1X one. Linear: heights in proportion "
            "to the amplitude, as most cascades are drawn.</p>"
        )
        self.wf_block = spin(0.1, 2.0, 0.5, 2, " s")
        self.wf_block.setSingleStep(0.1)
        self.wf_block.setToolTip(BLOCK_TIP)
        self.wf_block.valueChanged.connect(self._new_waterfall)
        self.wf_marks = QtWidgets.QCheckBox("1X and free whirl")
        self.wf_marks.setChecked(True)
        self.wf_marks.setToolTip("Mark 1X and the free whirl frequencies (from the Campbell diagram) under the lines")
        for c in (self.wf_station, self.wf_against, self.wf_scale):
            c.currentIndexChanged.connect(weak.slot(self, lambda self, _: self._draw_waterfall()))
        self.wf_marks.toggled.connect(weak.slot(self, lambda self, _: self._draw_waterfall()))
        self.wf_clear = QtWidgets.QPushButton("Clear")
        self.wf_clear.setToolTip("Drop every spectrum taken so far")
        self.wf_clear.clicked.connect(self._new_waterfall)
        self.waterfall_note = QtWidgets.QLabel()
        self.waterfall_note.setWordWrap(True)
        mute(self.waterfall_note)
        waterfall_tab = QtWidgets.QWidget()
        wt = QtWidgets.QVBoxLayout(waterfall_tab)
        row = FlowLayout()
        row.add(
            group("Orbit at:", self.wf_station),
            group("Against:", self.wf_against),
            group("Amplitude:", self.wf_scale),
            group("Block:", self.wf_block),
            self.wf_marks,
            self.wf_clear,
        )
        wt.addLayout(row)
        wt.addWidget(self.waterfall_plot, 1)
        wt.addWidget(self.waterfall_note)
        self.balancing = BalancingPanel(self)
        self.bearing_panel = BearingPanel()
        self.bearing_panel.set_top_rpm(self.top_rpm)
        self.bearing_panel.changed.connect(weak.slot(self, lambda self: self._on_bearing_tab(FILMS)))
        self.magnet_panel = MagnetPanel()
        self.magnet_panel.set_top_rpm(self.top_rpm)
        self.magnet_panel.changed.connect(weak.slot(self, lambda self: self._on_bearing_tab(MAGNETS)))
        self.rub_panel = RubPanel()
        self.rub_panel.changed.connect(self._on_params)
        self.matrices = QtWidgets.QTextBrowser()
        self.theory = NotesBrowser()
        self.theory.setHtml(THEORY_HTML)
        self.tabs = TabGroups()
        self.tabs.add_group("Modes", (campbell_tab, "Campbell diagram"), (stability_tab, "Stability"))
        self.tabs.add_group(
            "Response", (bode_tab, "Bode plot"), (polar_tab, "Polar plot"), (waterfall_tab, "Waterfall")
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
        self._clock = QtCore.QElapsedTimer()
        self._last = 0.0
        self._cost = 1e-9  # s of computing per simulated s (see step)
        self._target_t = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._frame)
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
    def _preset_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Things to try")
        v = QtWidgets.QVBoxLayout(box)
        self.presets = rotor_presets()
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems([p.name for p in self.presets] + [CUSTOM])
        self.preset.setToolTip(
            "<p>Load a rotor that shows one effect clearly, with a speed range to suit it. It starts from "
            "rest and runs up to its target speed. Change any parameter and it becomes your own rotor.</p>"
        )
        self.preset.activated.connect(self._on_preset)
        self.preset_note = QtWidgets.QLabel()
        self.preset_note.setWordWrap(True)
        mute(self.preset_note)
        v.addWidget(self.preset)
        v.addWidget(self.preset_note)
        return box

    def _speed_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Speed")
        form = form_layout(box)
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
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
            self.sim_speed.addItem(f"{s:g}× real time", s)
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
            "The phase lag is the angle the shaft turns from the keyphasor (the mark on the shaft passing +x) "
            "to the probe's positive peak, so on a circular orbit the y probe reads 90° more than the x "
            "probe. For the orbit, it is the angle from the mark back to the high spot.</p>"
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

    def _rotor_box(self) -> QtWidgets.QGroupBox:
        s = RotorSystem()
        box = QtWidgets.QGroupBox("Shaft and disc")
        form = form_layout(box)
        self.length = spin(0.05, 5.0, s.length, 3, " m")
        self.length.setToolTip("Bearing span L")
        self.diameter = spin(1.0, 500.0, s.diameter * 1e3, 1, " mm")
        self.diameter.setToolTip("Shaft diameter d; EI = E π d⁴/64")
        self.youngs = spin(1.0, 1000.0, s.youngs / 1e9, 1, " GPa")
        self.youngs.setToolTip("Young's modulus E of the shaft (steel ≈ 200 GPa)")
        self.position = PositionBox()
        self.position.setValue(s.position)
        self.position.setToolTip(
            "<p>Where the disc sits, a/L from bearing A. At 0.5 (midspan) with identical bearings the "
            "disc's translation and tilt are uncoupled, so the spin's gyroscopic moment never shows in "
            "the translational whirl: the classic Jeffcott rotor. Move it off centre and the disc tilts "
            "as it whirls.</p><p>Above 1 the disc overhangs bearing B (below 0, bearing A): an "
            "<i>overhung</i> rotor, like a fan or a pump impeller. The disc then tilts as much as it "
            "moves, and the gyroscopic moment moves even the first critical speed.</p>"
        )
        self.disc_mass = spin(0.01, 1000.0, s.disc_mass, 3, " kg")
        self.ip = spin(1e-6, 100.0, s.ip, 6, " kg·m²")
        self.ip.setToolTip(
            "Polar moment of inertia I<sub>p</sub>, about the spin axis (a thin disc of radius r<sub>d</sub>: m r<sub>d</sub>²/2)"
        )
        self.id = spin(1e-6, 100.0, s.id, 6, " kg·m²")
        self.id.setToolTip(
            "Diametral moment of inertia I<sub>d</sub>, about a diameter (a thin disc: m r<sub>d</sub>²/4 = I<sub>p</sub>/2)"
        )
        self.unbalance = spin(0.0, 1e6, s.unbalance * 1e6, 1, " g·mm")
        self.unbalance.setToolTip(
            "<p>Mass unbalance U = m e: the disc's mass times the distance e of its centre of mass from "
            "the shaft's axis. 20 g·mm on a 1 kg disc is e = 20 µm, about balance grade G6.3 at "
            "3000 rpm (e Ω = 6.3 mm/s).</p>"
        )
        form.addRow("Span L:", self.length)
        form.addRow("Shaft diameter d:", self.diameter)
        form.addRow("Young's modulus:", self.youngs)
        self.shaft_mass = QtWidgets.QCheckBox("Shaft mass, ρ")
        self.shaft_mass.setToolTip(SHAFT_MASS_TIP)
        self.density = spin(100.0, 30000.0, STEEL, 0, " kg/m³")
        self.density.setToolTip("The shaft's density (steel ≈ 7850 kg/m³, aluminium ≈ 2700)")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.shaft_mass)
        row.addWidget(self.density, 1)
        form.addRow("Shaft:", row)
        self.elements = QtWidgets.QSpinBox()
        self.elements.setRange(2, 60)
        self.elements.setValue(s.elements)
        self.elements.setKeyboardTracking(False)
        self.mass_matrix = QtWidgets.QComboBox()
        self.mass_matrix.addItems(["Consistent", "Lumped"])
        for w in (self.elements, self.mass_matrix):
            w.setToolTip(ELEMENTS_TIP)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.elements, 1)
        row.addWidget(self.mass_matrix, 1)
        form.addRow("Beam elements:", row)
        self.shaft_mass.toggled.connect(self._on_params)
        self.mass_matrix.currentIndexChanged.connect(self._on_params)
        form.addRow("Disc position a/L:", self.position)
        form.addRow("Disc mass m:", self.disc_mass)
        form.addRow("Polar inertia I<sub>p</sub>:", self.ip)
        form.addRow("Diametral inertia I<sub>d</sub>:", self.id)
        self.unbalance_angle = spin(-180.0, 180.0, math.degrees(s.unbalance_angle), 0, "°")
        self.unbalance_angle.setToolTip(
            "<p>Where the heavy spot is: its angle from the keyphasor mark on the shaft, in the direction "
            "of spin. The probes' phase is measured from the mark, as on a real machine.</p>"
        )
        self.bow = spin(0.0, 5000.0, s.bow * 1e6, 1, " µm")
        self.bow_angle = spin(-180.0, 180.0, math.degrees(s.bow_angle), 0, "°")
        self.skew = spin(0.0, 10.0, math.degrees(s.skew), 3, "°")
        self.skew_angle = spin(-180.0, 180.0, math.degrees(s.skew_angle), 0, "°")
        self.asymmetry = spin(0.0, 90.0, s.asymmetry * 100, 1, " %")
        self.asymmetry_angle = spin(-180.0, 180.0, math.degrees(s.asymmetry_angle), 0, "°")
        crack = Crack()
        self.crack_depth = spin(0.0, CRACK_DEPTH_MAX * 100, 0.0, 1, " %")
        self.crack_depth.setSingleStep(5.0)
        self.crack_angle = spin(-180.0, 180.0, math.degrees(crack.angle), 0, "°")
        for size, angle, tip, label in (
            (self.unbalance, self.unbalance_angle, None, "Unbalance m·e:"),
            (self.bow, self.bow_angle, BOW_TIP, "Shaft bow δ<sub>b</sub>:"),
            (self.skew, self.skew_angle, SKEW_TIP, "Disc skew τ:"),
            (self.asymmetry, self.asymmetry_angle, ASYMMETRY_TIP, "Asymmetry ±Δ:"),
            (self.crack_depth, self.crack_angle, CRACK_TIP, "Crack depth a<sub>c</sub>/d:"),
        ):
            angle.setWrapping(True)
            angle.setStepType(QtWidgets.QAbstractSpinBox.StepType.DefaultStepType)
            angle.setSingleStep(15.0)
            row = QtWidgets.QHBoxLayout()
            row.addWidget(size, 3)
            row.addWidget(QtWidgets.QLabel("at"))
            row.addWidget(angle, 2)
            if tip:
                size.setToolTip(tip)
                angle.setToolTip(tip)
            form.addRow(label, row)
        self.crack_position = spin(-1.0, 2.0, crack.position, 3, "")
        self.crack_position.setSingleStep(0.05)
        self.crack_breathing = QtWidgets.QCheckBox("Breathing")
        self.crack_breathing.setChecked(crack.breathing)
        self.crack_breathing.toggled.connect(self._on_params)
        self.crack_position.setToolTip(CRACK_TIP)
        self.crack_breathing.setToolTip(CRACK_TIP)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.crack_position, 3)
        row.addWidget(self.crack_breathing, 2)
        form.addRow("Crack at z/L:", row)
        self.internal_damping = spin(0.0, 1e5, s.internal_damping, 1, " N·s/m")
        self.internal_damping.setToolTip(INTERNAL_TIP)
        form.addRow("Internal damping c<sub>i</sub>:", self.internal_damping)
        self.gravity = QtWidgets.QCheckBox(f"Horizontal rotor (g = {GRAVITY} m/s²)")
        self.gravity.setToolTip(GRAVITY_TIP)
        self.gravity.toggled.connect(self._on_params)
        form.addRow("Gravity:", self.gravity)
        self.shaft_note = QtWidgets.QLabel()
        self.shaft_note.setWordWrap(True)
        mute(self.shaft_note)
        form.addRow(self.shaft_note)
        for box_ in (
            self.length,
            self.diameter,
            self.youngs,
            self.position,
            self.disc_mass,
            self.ip,
            self.id,
            self.unbalance,
            self.unbalance_angle,
            self.bow,
            self.bow_angle,
            self.skew,
            self.skew_angle,
            self.asymmetry,
            self.asymmetry_angle,
            self.crack_depth,
            self.crack_angle,
            self.crack_position,
            self.internal_damping,
            self.density,
            self.elements,
        ):
            box_.valueChanged.connect(self._on_params)
        return box

    def _bearing_box(self) -> QtWidgets.QGroupBox:
        b = Bearing()
        box = QtWidgets.QGroupBox("Bearing supports")
        grid = QtWidgets.QGridLayout(box)
        grid.setHorizontalSpacing(4)
        # Units on a second line, so that the headers are no wider than the boxes below them.
        heads = (
            ("k<sub>x</sub><br>[N/m]", "Horizontal support stiffness"),
            ("k<sub>y</sub><br>[N/m]", "Vertical support stiffness"),
            ("c<sub>x</sub><br>[N·s/m]", "Horizontal support damping"),
            ("c<sub>y</sub><br>[N·s/m]", "Vertical support damping"),
            ("m<br>[kg]", "Mass of the journal and whatever moves with it on the support"),
            ("k<sub>xy</sub><br>[N/m]", KXY_TIP),
            ("λ<br>&nbsp;", SWIRL_TIP),
        )
        for c, width in enumerate(BEARING_WIDTHS):
            grid.setColumnStretch(c + 1, width)
        for c, (text, tip) in enumerate(heads):
            label = QtWidgets.QLabel(text)
            label.setToolTip(tip)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(label, 0, c + 1)
        self.bearing_rows: list[list[QtWidgets.QDoubleSpinBox]] = []
        for r, name in enumerate("AB"):
            grid.addWidget(QtWidgets.QLabel(f"<b>{name}</b>"), r + 1, 0)
            row = [
                spin(1e2, 1e10, b.kx, 0),
                spin(1e2, 1e10, b.ky, 0),
                spin(0.0, 1e6, b.cx, 1),
                spin(0.0, 1e6, b.cy, 1),
                spin(0.001, 100.0, b.mass, 3),
                spin(-1e9, 1e9, b.kxy, 0),
                spin(0.0, 1.0, b.swirl, 2),
            ]
            row[5].setToolTip(KXY_TIP)
            row[6].setToolTip(SWIRL_TIP)
            row[6].setSingleStep(0.05)
            for c, w in enumerate(row):
                # No arrows (the wheel and the arrow keys still step): seven columns leave no room for them.
                w.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
                w.setMinimumWidth(BEARING_WIDTHS[c])
                w.valueChanged.connect(self._on_params)
                grid.addWidget(w, r + 1, c + 1)
            self.bearing_rows.append(row)
        self.isotropic = QtWidgets.QCheckBox("Isotropic (ky = kx, cy = cx)")
        self.isotropic.setChecked(True)
        self.isotropic.setToolTip(
            "<p>Untick to give the supports a different stiffness or damping vertically. The single "
            "critical speed then splits in two, the orbits become ellipses, and between the two "
            "criticals the shaft whirls backward, against its spin.</p>"
        )
        self.same_bearings = QtWidgets.QCheckBox("B same as A")
        self.same_bearings.setChecked(True)
        self.same_bearings.setToolTip(
            "<p>Untick to give the two bearings different supports. Like an off-centre disc, unequal "
            "supports couple the disc's translation and tilt, so the gyroscopic moment shows.</p>"
        )
        self.supports = QtWidgets.QComboBox()
        self.supports.addItems(SUPPORTS)
        self.supports.setToolTip(
            "<p>What holds the journals: the springs and dampers in the table, or these bearings at both ends.</p>"
        )
        self.supports.setItemData(FILMS, FILM_TIP, QtCore.Qt.ItemDataRole.ToolTipRole)
        self.supports.setItemData(MAGNETS, MAGNET_TIP, QtCore.Qt.ItemDataRole.ToolTipRole)
        self.supports.currentIndexChanged.connect(self._on_params)
        self.nonlinear_film = QtWidgets.QCheckBox("Nonlinear film")
        self.nonlinear_film.setChecked(True)
        self.nonlinear_film.setToolTip(NONLINEAR_FILM_TIP)
        for check in (self.isotropic, self.same_bearings, self.nonlinear_film):
            check.toggled.connect(self._on_params)
        grid.addWidget(self.isotropic, 3, 0, 1, 8)
        grid.addWidget(self.same_bearings, 4, 0, 1, 8)
        grid.addWidget(self.supports, 5, 0, 1, 5)
        grid.addWidget(self.nonlinear_film, 5, 5, 1, 3)
        return box

    # ------------------------------------------------------------- parameters
    def system(self) -> RotorSystem:
        """The rotor described by the panels, after applying the isotropic and same-as-A ties."""
        iso, same = self.isotropic.isChecked(), self.same_bearings.isChecked()
        kind = self.supports.currentIndex()
        film = kind != SPRINGS  # a film or a magnet holds the journal
        for r, row in enumerate(self.bearing_rows):
            src = self.bearing_rows[0] if same and r == 1 else row
            for c, w in enumerate(row):
                value = src[c].value()
                if iso and c in (1, 3):
                    value = src[c - 1].value()
                if w.value() != value:
                    w.blockSignals(True)
                    w.setValue(value)
                    w.blockSignals(False)
                # A film or a magnet holds the journal in place of the springs: only its mass is left to set.
                w.setEnabled(not (same and r == 1) and not (iso and c in (1, 3)) and not (film and c != 4))
        widget: QtWidgets.QWidget
        for widget in (self.isotropic, self.same_bearings):
            widget.setEnabled(not film)
        # A turning shaft's split time step keeps the films linearized, and has no stator ring.
        round_shaft = not self.asymmetry.value() and not self.crack_depth.value()
        self.nonlinear_film.setEnabled(kind == FILMS and round_shaft)
        self.rub_panel.set_allowed(round_shaft and not (kind == FILMS and self.nonlinear_film.isChecked()))
        self.nonlinear_film.setVisible(kind == FILMS)
        oil = self.bearing_panel.bearing(rotor=False) if kind == FILMS else None
        magnet = self.magnet_panel.bearing() if kind == MAGNETS else None
        bearings = [_bearing(row, oil, magnet) for row in self.bearing_rows]
        for widget in (self.density, self.elements, self.mass_matrix):
            widget.setEnabled(self.shaft_mass.isChecked())
        cracked = self.crack_depth.value() > 0.0
        for widget in (self.crack_angle, self.crack_position, self.crack_breathing):
            widget.setEnabled(cracked)
        return RotorSystem(
            length=self.length.value(),
            diameter=self.diameter.value() / 1e3,
            youngs=self.youngs.value() * 1e9,
            position=_off_bearing(self.position.value()),
            disc_mass=self.disc_mass.value(),
            ip=self.ip.value(),
            id=self.id.value(),
            bearing_a=bearings[0],
            bearing_b=bearings[1],
            unbalance=self.unbalance.value() / 1e6,
            unbalance_angle=math.radians(self.unbalance_angle.value()),
            weights=self._weights,
            plane_weights=self._plane_weights,
            gravity=GRAVITY if self.gravity.isChecked() else 0.0,
            bow=self.bow.value() / 1e6,
            bow_angle=math.radians(self.bow_angle.value()),
            skew=math.radians(self.skew.value()),
            skew_angle=math.radians(self.skew_angle.value()),
            asymmetry=self.asymmetry.value() / 100,
            asymmetry_angle=math.radians(self.asymmetry_angle.value()),
            crack=self._crack() if cracked else None,
            internal_damping=self.internal_damping.value(),
            density=self.density.value() if self.shaft_mass.isChecked() else 0.0,
            elements=self.elements.value(),
            lumped=self.mass_matrix.currentIndex() == 1,
            nonlinear_films=kind == FILMS and self.nonlinear_film.isChecked(),
            stator=self.rub_panel.stator(),
        ).with_(**(self._hidden or {}))

    def _crack(self) -> Crack:
        """The crack the panel describes, kept inside the shaft (and off its ends)."""
        lo, hi = (
            x / self.length.value()
            for x in (
                min(0.0, self.position.value() * self.length.value()),
                max(self.length.value(), self.position.value() * self.length.value()),
            )
        )
        margin = 1e-3 * (hi - lo)
        return Crack(
            position=min(max(self.crack_position.value(), lo + margin), hi - margin),
            depth=self.crack_depth.value() / 100,
            angle=math.radians(self.crack_angle.value()),
            breathing=self.crack_breathing.isChecked(),
        )

    def set_system(self, system: RotorSystem) -> None:
        """Show and use `system` (for scripts and tests)."""
        boxes = (
            self.length,
            self.diameter,
            self.youngs,
            self.position,
            self.disc_mass,
            self.ip,
            self.id,
            self.unbalance,
            self.unbalance_angle,
            self.bow,
            self.bow_angle,
            self.skew,
            self.skew_angle,
            self.asymmetry,
            self.asymmetry_angle,
            self.crack_depth,
            self.crack_angle,
            self.crack_position,
            self.crack_breathing,
            self.internal_damping,
            self.gravity,
            self.isotropic,
            self.same_bearings,
            self.supports,
            self.nonlinear_film,
            self.shaft_mass,
            self.density,
            self.elements,
            self.mass_matrix,
            *(w for row in self.bearing_rows for w in row),
        )
        for w in boxes:
            w.blockSignals(True)
        self.length.setValue(system.length)
        self.diameter.setValue(system.diameter * 1e3)
        self.youngs.setValue(system.youngs / 1e9)
        self.position.setValue(system.position)
        self.disc_mass.setValue(system.disc_mass)
        self.ip.setValue(system.ip)
        self.id.setValue(system.id)
        self._clear_hidden()
        self.unbalance.setValue(system.unbalance * 1e6)
        self.unbalance_angle.setValue(math.degrees(system.unbalance_angle))
        self._weights = system.weights
        self._plane_weights = system.plane_weights
        self.gravity.setChecked(system.gravity > 0.0)
        self.bow.setValue(system.bow * 1e6)
        self.bow_angle.setValue(math.degrees(system.bow_angle))
        self.skew.setValue(math.degrees(system.skew))
        self.skew_angle.setValue(math.degrees(system.skew_angle))
        self.asymmetry.setValue(system.asymmetry * 100)
        self.asymmetry_angle.setValue(math.degrees(system.asymmetry_angle))
        crack = system.crack
        self.crack_depth.setValue(0.0 if crack is None else crack.depth * 100)
        if crack is not None:
            self.crack_angle.setValue(math.degrees(crack.angle))
            self.crack_position.setValue(crack.position)
            self.crack_breathing.setChecked(crack.breathing)
        self.internal_damping.setValue(system.internal_damping)
        self.shaft_mass.setChecked(system.density > 0.0)
        if system.density > 0.0:
            self.density.setValue(system.density)
        self.elements.setValue(system.elements)
        self.mass_matrix.setCurrentIndex(1 if system.lumped else 0)
        A, B = system.bearing_a, system.bearing_b
        self.isotropic.setChecked(all(b.kx == b.ky and b.cx == b.cy for b in (A, B)))
        self.same_bearings.setChecked(A == B)
        for row, b in zip(self.bearing_rows, (A, B)):
            for w, v in zip(row, (b.kx, b.ky, b.cx, b.cy, b.mass, b.kxy, b.swirl)):
                w.setValue(v)
        self.supports.setCurrentIndex(FILMS if A.film is not None else MAGNETS if A.magnet is not None else SPRINGS)
        self.nonlinear_film.setChecked(system.nonlinear_films or A.film is None)
        if A.film is not None:
            self.bearing_panel.set_bearing(A.film, notify=False)
        if A.magnet is not None:
            self.magnet_panel.set_bearing(A.magnet, notify=False)
        self.rub_panel.set_stator(system.stator)
        for w in boxes:
            w.blockSignals(False)
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
            self.preset.currentText(), target_rpm=self.target_rpm.value(), top_rpm=self.top_rpm, rotor=self.system()
        )

    def frame_context(self) -> str:
        """For the frame log (frame_stats): the preset showing."""
        return self.preset.currentText()

    def _on_preset(self, index: int) -> None:
        if index < len(self.presets):
            self.load_preset(self.presets[index].name)
            self._run(True)  # the reader picked it, as a trial
        else:
            self._show_preset(self.sim.system)  # Custom is not a rotor: show what is loaded

    def _show_preset(self, system: RotorSystem) -> None:
        """Select the preset this rotor is, or Custom, and show its note."""
        index = next((i for i, p in enumerate(self.presets) if _same_rotor(p.system, system)), len(self.presets))
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(index)
        self.preset.blockSignals(False)
        self.preset_note.setText(
            self.presets[index].note
            if index < len(self.presets)
            else "Your own rotor: pick a preset to start again from one."
        )

    def set_weights(self, weights: tuple[complex, ...], plane_weights: tuple[PlaneWeight, ...] = ()) -> None:
        """Put these balance weights on the disc (kg·m each) and in these planes, in place of any there."""
        self._weights = tuple(weights)
        self._plane_weights = tuple(plane_weights)
        self._on_params()

    def _hideable(self) -> dict[str, tuple[QtWidgets.QDoubleSpinBox, float]]:
        """The boxes a balancing exercise can hide, by RotorSystem field, with the factor from SI to the box."""
        deg = 180.0 / math.pi
        return {
            "unbalance": (self.unbalance, 1e6),
            "unbalance_angle": (self.unbalance_angle, deg),
            "skew": (self.skew, deg),
            "skew_angle": (self.skew_angle, deg),
        }

    def hide_unbalance(self, unbalance: float, angle: float, skew: tuple[float, float] | None = None) -> None:
        """Give the disc this unbalance (kg·m at `angle` rad) without showing it: a balancing exercise.

        With `skew` (τ, its angle in rad), hide a couple unbalance too; otherwise the skew stays as set.
        """
        self._clear_hidden()
        self._hidden = {"unbalance": unbalance, "unbalance_angle": angle}
        if skew is not None:
            self._hidden.update(skew=skew[0], skew_angle=skew[1])
        boxes = self._hideable()
        for name in self._hidden:
            box = boxes[name][0]
            box.blockSignals(True)
            box.setSpecialValueText("hidden")
            box.setValue(box.minimum())
            box.setEnabled(False)
            box.blockSignals(False)
        self.set_heavy_shown(False)
        self._on_params()

    def reveal_unbalance(self) -> None:
        """Show the hidden unbalance (and couple) in the panel again, and the heavy spot."""
        if self._hidden is None:
            return
        hidden, boxes = self._hidden, self._hideable()
        self._clear_hidden()
        for name, value in hidden.items():
            box, scale = boxes[name]
            box.blockSignals(True)
            box.setValue(value * scale)
            box.blockSignals(False)
        self.set_heavy_shown(True)
        self._on_params()

    def _clear_hidden(self) -> None:
        if self._hidden is not None:
            for name in self._hidden:
                box = self._hideable()[name][0]
                box.setSpecialValueText("")
                box.setEnabled(True)
        self._hidden = None

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
        ks = 48.0 * system.ei / system.length**3
        if system.overhung:
            formula = f" = 3EI/(c²(L + c)), overhung by c = {system.overhang * 1e3:.4g} mm"
        elif abs(system.position - 0.5) < 1e-9:
            formula = " = 48EI/L³"
        else:
            formula = f" (48EI/L³ = {ks:,.0f} N/m at midspan)"
        self.shaft_note.setText(
            f"EI = {system.ei:.4g} N·m²; shaft stiffness at the disc (bearings rigid) "
            f"{system.disc_stiffness:,.0f} N/m"
            + formula
            + self._asymmetry_text(system)
            + self._crack_text(system)
            + self._shaft_mass_text(system)
            + self._sag_text(system)
        )
        self._analysis_timer.start()

    @staticmethod
    def _asymmetry_text(system: RotorSystem) -> str:
        if not system.asymmetry:
            return ""
        d = system.asymmetry
        return (
            f"<br>Asymmetric: EI = {system.ei * (1 + d):.4g} N·m² bending toward the stiff axis "
            f"({math.degrees(system.asymmetry_angle):.0f}° from the mark), {system.ei * (1 - d):.4g} across it"
        )

    @staticmethod
    def _crack_text(system: RotorSystem) -> str:
        crack = system.crack
        if crack is None:
            return ""
        c_open, c_across = system.crack_compliances
        soft, stiff = (system.axis_system(k).stiffness(0.0) for k in (False, True))
        loss = [1.0 - k[XD, XD] / system.with_(crack=None).stiffness(0.0)[XD, XD] for k in (soft, stiff)]
        state = (
            "held open"
            if not crack.breathing
            else "breathing, opened by the weight"
            if system.crack_opening
            else "breathing, but shut: nothing opens it without gravity"
        )
        return (
            f"<br>Crack {crack.depth:.0%} deep at z = {system.crack_z * 1e3:.4g} mm ({state}): compliance "
            f"{c_open:.3g} rad/(N·m) opening it, {c_across:.3g} across; on average over a turn it takes "
            f"{loss[0]:.2%} and {loss[1]:.2%} off the disc's stiffness along and across its mouth"
        )

    @staticmethod
    def _shaft_mass_text(system: RotorSystem) -> str:
        if not system.density:
            return ""
        f = system.fixed_interface_freqs
        held = (
            f"its lowest {f.size} mode{'s' if f.size != 1 else ''} with the stations held "
            f"({', '.join(f'{x:,.0f}' for x in f)} Hz)"
            if f.size
            else f"no modes with the stations held below {MODE_LIMIT:,.0f} Hz"
        )
        return (
            f"<br>Shaft {system.shaft_mass:.3g} kg ({system.shaft_mass / system.disc_mass:.0%} of the disc), "
            f"{system.nodes.size - 1} elements, reduced to the stations and {held}"
        )

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
        lines = [f"{w * RPM:,.0f} rpm ({w / (2 * math.pi):.4g} Hz), {whirl_name(r)} whirl" for w, r in crits]
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
            + f"<br><b>Stability:</b> {self._onset_text()}"
        )
        self.stability_note.setText(self._stability_note())
        self.campbell_note.setText(
            "Each dot is a damped natural frequency at that speed, coloured by its whirl direction "
            f"(modes damped more than ζ = {DAMPED:g}, here the journals bouncing on their supports, are "
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

    def _onset_text(self) -> str:
        o = self.onset
        if o is None:
            return f"stable up to {self.top_rpm:,.0f} rpm."
        whirl = whirl_name(o.whirl)
        if o.omega == 0.0:
            return f"<b>unstable even at rest</b>: the {whirl} whirl at {o.freq_hz:.3g} Hz grows."
        if math.isfinite(o.end):
            return (
                f"<b>unstable from {o.omega * RPM:,.0f} to {o.end * RPM:,.0f} rpm</b>, where the {whirl} whirl "
                f"at {o.freq_hz:.3g} Hz ({o.order:.2f}×) grows; stable again above."
            )
        return (
            f"<b>unstable above {o.omega * RPM:,.0f} rpm</b>, where the {whirl} whirl at "
            f"{o.freq_hz:.3g} Hz ({o.order:.2f}×) starts to grow."
        )

    def _stability_note(self) -> str:
        s = self.sim.system
        bearings = (("A", s.bearing_a), ("B", s.bearing_b))
        coupled = [f"k<sub>xy</sub> = {b.kxy:,.0f} N/m at {n}" for n, b in bearings if b.kxy and b.film is None]
        coupled += [f"swirl λ = {b.swirl:g} at {n}" for n, b in bearings if b.swirling]
        if s.films:
            coupled.append("oil-film bearings (their oil swirling at half the speed)")
        if s.magnets:
            coupled.append("magnetic bearings (their controllers' lag)")
        if s.internal_damping:
            coupled.append(f"internal damping c<sub>i</sub> = {s.internal_damping:g} N·s/m")
        if s.asymmetry:
            coupled.append(f"the shaft's asymmetry Δ = {s.asymmetry:.3g}")
        if s.crack_opening and s.crack is not None:
            coupled.append(f"a {'breathing' if s.crack.breathing else 'open'} crack {s.crack.depth:.0%} deep")
        cause = (
            "With " + " and ".join(coupled) + ": "
            if coupled
            else "No cross-coupling, internal damping, asymmetry or crack: every mode stays damped, however fast it "
            "spins. Set c<sub>i</sub>, k<sub>xy</sub>, λ or Δ to see one go unstable. "
        )
        if self.measure.currentIndex() == 2:
            chart = (
                " The chart shades where the rotor is unstable against the shaft's asymmetry Δ and the speed "
                "(the rest of the rotor as it is), between the two axes' critical speeds (dashed)."
            )
            if self.asym_chart is not None and not self.asym_chart.done:
                chart += f" Working it out: {self.asym_chart._next / self.asym_chart.growth.size:.0%}."
        else:
            chart = f" Modes damped more than ζ = {DAMPED:g} are left off."
        return (
            f"<small>{cause}{self._onset_text()}{chart} Below: the full spectrum of the strip chart's station.</small>"
        )

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
        if self.supports.currentIndex() == kind:
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
        self.matrices.setHtml(matrices_html(self.sim.system, self.sim.omega, hidden=tuple(self._hidden or ())))
        self.matrices.verticalScrollBar().setValue(scroll)

    def apply_theme(self) -> None:
        self.view.apply_theme()
        self.orbits.apply_theme()
        self.time.apply_theme()
        self.campbell_plot.apply_theme()
        self.stability_map.apply_theme()
        self.spectrum.apply_theme()
        self.waterfall_plot.apply_theme()
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
            self._start_timer()

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
        self._draw()

    def tap(self) -> None:
        self.sim.tap(0.0, -TAP_IMPULSE)

    def _on_tap(self) -> None:
        """The Tap button: a tap, and the rotor started if it waits."""
        self.tap()
        if not self.running:
            self._run(True)

    def _on_repeat_taps(self, on: bool) -> None:
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
        self._start_timer()

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
        self._start_timer()

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
        self.polar.getPlotItem().enableAutoRange()

    def _steady_probe(self) -> tuple[np.ndarray, np.ndarray]:
        """The steady-state 1X (or 2X, 3X) values at the chosen station and direction, and their unwrapped lag."""
        station, direction = self._probe()
        vectors = self._steady_orders[self._order() - 1]
        values = probe(vectors[:, station], direction)
        return values, lag_degrees(values)

    def _draw_tracking(self, full: bool = False) -> None:
        """The Bode and polar plots: the steady-state curve and summary (if `full`), the sweeps, the marker."""
        if not self._steady_rpm.size:
            return
        rpm = self._steady_rpm
        values, lag = self._steady_probe()
        if full:
            self.bode.set_steady(rpm, values, lag)
            lo, hi = self._tracking_range()
            shown = (rpm >= lo) & (rpm <= hi)
            self.polar.set_steady(rpm[shown], values[shown], nice_marks(lo, hi))
        self.sweep_plots.redraw(*self._probe(), rpm, lag, full)
        here = self.sim.omega * RPM
        v = complex(np.interp(here, rpm, values.real), np.interp(here, rpm, values.imag))
        self.bode.set_here(here, abs(v), float(np.interp(here, rpm, lag)))
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
        head = f"<b>Steady state</b> ({where}): peak {_fmt_len(ss_amp)} at {ss_rpm:,.0f} rpm"
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
            head += f"; its mode has ζ = {z:.3g}, ζ² = {z * z:.2g}"
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
                f"<td align='right'>{alpha / wc**2:.2g}</td><td align='right'>{_fmt_len(amp)}</td>"
                f"<td align='right'>{100 * amp / ss_amp:.0f}%</td><td align='right'>{w * RPM:,.0f}</td>"
                f"<td align='right'>{(w - wc) * RPM:+,.0f}</td></tr>"
            )
        return (
            head + "<table cellspacing='0' cellpadding='2'><tr><th align='left'>Sweep</th>"
            f"<th>Ω'/ω<sub>c</sub>²</th><th>{order}X peak</th><th>of steady</th><th>at rpm</th><th>vs steady</th></tr>"
            + "".join(rows)
            + "</table>"
        )

    # ------------------------------------------------------------- loop
    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._start_timer()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()

    def _start_timer(self) -> None:
        if self.isVisible() and self.running and not self._timer.isActive():
            self._clock.restart()
            self._last = 0.0
            self._target_t = self.sim.t
            self._timer.start(FRAME_MS)

    def _frame(self) -> None:
        if not self.running:  # paused or reset since this frame was queued
            return
        began = self._clock.elapsed()
        now = began / 1000.0
        dt = min(now - self._last, 0.1)
        self._last = now
        speed = float(self.sim_speed.currentData())
        with frame_stats.frame("Jeffcott rotor", lambda: self.sim.t, FRAME_MS):
            self.step(dt * (max(speed, MEASURE_SPEED) if self.measurement is not None else speed), FRAME_BUDGET)
            settle_plots(self)
        if self.isVisible() and self.running:  # the next frame FRAME_MS after this one began, not after it ended
            self._timer.start(max(1, FRAME_MS - (self._clock.elapsed() - began)))

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
                if self.waterfall is not None and self.waterfall.feed(
                    r.t[pick], vib[:, 0::2] + 1j * vib[:, 1::2], cols[:, OMEGA]
                ):
                    self._waterfall_drawn = False
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
        parts = [f"<b>Speed {rpm:,.0f} rpm</b> ({sim.omega / (2 * math.pi):.3g} Hz)"]
        if not steady and sim.omega != sim.target:
            parts[0] += f" → {sim.target * RPM:,.0f} rpm"
        crits = [c for c, _ in critical_speeds(self.campbell, _isotropic(sim.system))] if self.campbell else []
        if crits and w > 0:
            parts.append(f"Ω / first critical = {sim.omega / crits[0]:.2f}")
        disc = self._vibration(y)[:, 2:4]
        if disc.size:
            text = f"disc whirling up to {_fmt_len(float(np.hypot(disc[:, 0], disc[:, 1]).max()))}"
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
                    f"<span style='color:{colors.force}'>{what}: a whirl at {f:.3g} Hz "
                    f"({2 * math.pi * f / w:.2f}×) grows ×e every {1 / grow:.2g} s</span>"
                )
        if self.rubbed:
            where = (
                f"journal {self.touched[0]} landed on its backup bearing, {_fmt_len(self.touched[1])} off "
                "centre (half the magnets' air gap). A real machine drops onto these when its magnetic "
                "bearings lose control; the model here is linear, and has no contact. "
                if self.touched and sim.system.magnets
                else f"journal {self.touched[0]} reached the wall of its bearing, a clearance of "
                f"{_fmt_len(self.touched[1])} off centre. "
                "A real oil film stiffens sharply there and keeps the journal off the wall; the film here is "
                "linearized (about where the journal rests), so it would let it through. Tick <i>Nonlinear "
                "film</i> to step the film's whole force. "
                if self.touched
                else f"the whirl passed {_fmt_len(RUB * sim.system.length)} ({RUB:.0%} of the span), "
                f"{_fmt_len(RUB * sim.system.length - sim.system.stator.clearance)} past the stator ring's "
                "clearance: its mount would have given way long before, and the machine been wrecked. "
                if sim.system.stator is not None
                else f"the whirl passed {_fmt_len(RUB * sim.system.length)} ({RUB:.0%} of the span), where a real "
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

    def _sag_text(self, system: RotorSystem) -> str:
        if not system.gravity:
            return ""
        q = system.static_deflection(0.0)
        return (
            f"<br>Static sag at rest: disc {_fmt_len(-q[YD])}, journals {_fmt_len(-q[YA])} (A) and "
            f"{_fmt_len(-q[YB])} (B)"
        )

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
            f"Full spectrum of the {STATIONS[st].lower()} orbit over the last {self.time_window.value():g} s "
            "(the strip chart's station and window): forward whirl on the right, backward on the left."
        )

    # ------------------------------------------------------------- waterfall
    def _new_waterfall(self) -> None:
        """Start the spectra afresh: for a new rotor, speed range or block length, or on Clear."""
        f_max = max(WATERFALL_RANGE * self.top_rpm / 60.0, 50.0)
        self.waterfall = Waterfall(self.wf_block.value(), f_max, self.sim.system)
        self._waterfall_drawn = False
        self._draw_waterfall()

    def _draw_waterfall(self, only_new: bool = False) -> None:
        """Draw the waterfall if its tab is showing (with `only_new`, only if blocks came in since)."""
        if self.tabs.current_page() is not self.waterfall_plot.parentWidget():
            return
        if only_new and self._waterfall_drawn:
            return
        wf = self.waterfall
        if wf is None:
            return
        self._waterfall_drawn = True
        mode = self.wf_against.currentIndex()
        blocks = wf.latest_ramp(up=mode == 0) if against_speed(mode) else wf.blocks
        ys, label = baselines(blocks, mode)
        st = self.wf_station.currentIndex()
        log = self.wf_scale.currentIndex() == 0
        tallest = self.waterfall_plot.set_blocks(wf.freqs, blocks, ys, st, log, self.wf_marks.isChecked(), label)
        where = f"the {STATIONS[st].lower()} orbit"
        if tallest is None:
            if against_speed(mode):
                text = (
                    f"No {'run-up' if mode == 0 else 'coast-down'} since the spectra were cleared: change "
                    "the target speed, or run up and coast down, and the spectra are drawn at each speed. "
                    "Spectra taken at a steady speed are shown against time."
                )
            else:
                text = f"The first {wf.block:g} s block is being taken."
            self.waterfall_note.setText(text)
            return
        b = tallest.block
        at = f"{b.omega * RPM:,.0f} rpm" if against_speed(mode) else f"{b.t:.1f} s ({b.omega * RPM:,.0f} rpm)"
        order = f" ({tallest.freq / (b.omega / (2 * math.pi)):+.2f}×)" if b.omega > 0 else ""
        self.waterfall_note.setText(
            f"{len(blocks)} full spectra of {where}, each of a {wf.block:g} s block ({1 / wf.block:.2g} Hz "
            f"resolution), overlapping by half. Tallest peak {_fmt_len(tallest.amp)} at "
            f"{tallest.freq:+.1f} Hz{order}, at {at}"
            + (": heights are in decades above 1/1000 of it (60 dB)." if log else ".")
            + (
                ""
                if self.repeat_taps.isChecked()
                else " Tap the disc, or tick <i>Tap the disc every</i>, to ring the free whirl."
            )
        )


def _bearing(
    row: list[QtWidgets.QDoubleSpinBox], film: JournalBearing | None, magnet: MagneticBearing | None
) -> Bearing:
    """A bearing from its row of boxes (kx, ky, cx, cy, mass, kxy, swirl), on a film or a magnet if given."""
    kx, ky, cx, cy, mass, kxy, swirl = (w.value() for w in row)
    return Bearing(kx, ky, cx, cy, mass, kxy, swirl, film=film, magnet=magnet)


def _isotropic(system: RotorSystem) -> bool:
    """True if the supports act alike in x and y, so that unbalance drives only forward whirl.

    A magnetic bearing does (the same controller on both axes), though its controller's
    states keep the rotor from being time-invariant in the rotating frame (isotropic_supports).
    """
    if not system.magnets:
        return system.isotropic_supports
    plain = [Bearing(mass=b.mass) if b.magnet is not None else b for b in (system.bearing_a, system.bearing_b)]
    return system.with_(bearing_a=plain[0], bearing_b=plain[1]).isotropic_supports


def _fmt_len(meters: float) -> str:
    if meters >= 1e-3:
        return f"{meters * 1e3:.3g} mm"
    if meters >= 1e-6:
        return f"{meters * 1e6:.3g} µm"
    return f"{meters * 1e9:.3g} nm"


class PositionBox(QtWidgets.QDoubleSpinBox):
    """The disc's position a/L: between the bearings or overhung past either, stepping over the bearings."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRange(-1.0, 2.0)
        self.setDecimals(3)
        self.setSingleStep(0.05)
        self.setKeyboardTracking(False)

    def stepBy(self, steps: int) -> None:
        super().stepBy(steps)
        if _off_bearing(self.value()) != self.value():
            super().stepBy(1 if steps > 0 else -1)


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


def _off_bearing(position: float) -> float:
    """a/L moved off a bearing: the disc cannot sit on one (it would have no shaft on one side)."""
    for bearing in (0.0, 1.0):
        if abs(position - bearing) < BEARING_GAP:
            inward = 0.5 - bearing  # toward midspan, if it is right on the bearing
            return bearing + math.copysign(BEARING_GAP, position - bearing if position != bearing else inward)
    return position
