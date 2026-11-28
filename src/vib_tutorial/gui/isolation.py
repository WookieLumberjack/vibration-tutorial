"""Vibration isolation page (#478 Phase 4): a machine with a rotating unbalance on its mounts, live.

The machine is stepped as it runs (`view_models/live_isolation.LiveIsolation`: the isolator's state space, exactly),
not a run worked out in advance and played back. What shakes is the reader's pick: the machine's unbalance, or the
floor under the mounts; both share the one transmissibility. Left: the machine (its mass, speed, unbalance and an
inertia block, and Run up / Run down through resonance), what shakes, the mounts (their kind, sag and damping), then
the Things to try (setups to go back to; their predictions asked once the reader has handled it) and a Challenge.
Center: the machine on its mounts (`isolation_view.IsolationView`: pull it and let go, or drag the grips on the
machine, the spring and the damper as it runs), the forces on it beside it (`free_body_view.FreeBodyView`, the
force on the floor aside), the run's controls, and the time plots, which keep what came before a change with the
change marked. Right: the Transmissibility the reader measures, a point a steady speed (dwell, or Sweep), the
Floor force built as the speed runs up and down, the Design chart, and the Theory.

What the page shows is `view_models/isolation.py`'s and `live_isolation.py`'s; its trials, problems, tips and notes
are `isolation_trials.py`'s (no Qt). The controls are bound to the state (`IsolationTrial`) by `bindings.Bindings`,
so Defaults round-trips by construction.

Working on it: `tests/test_isolation_page.py` (the page live: a dwell landing T, both ways, a run up building the
floor force, the grips, a pull, a change mid-run, predictions after play), `test_live_isolation.py` (the Qt-free
run), `test_view_models.py` (the panels), `test_readiness.py`.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.isolation import Mount, deflection_for
from ..view_models import isolation as view
from ..view_models import live_isolation as li
from ..view_models import recorder as rec
from ..view_models.free_body import ArrowScale
from ..view_models.isolation import CHART_LEVELS, FAMILY, MOUNTS, IsolationTrial, isolator
from ..view_models.live_isolation import COL, FLOOR, GRIPS, LiveIsolation
from ..view_models.text import fmt
from ..view_models.text import number as sig
from . import coalesce
from .bindings import Bindings
from .challenge import ChallengeBox
from .drag import DragLine
from .frame_loop import FrameLoop
from .free_body_view import FreeBodyView
from .isolation_trials import (
    CHART_TIP,
    FORCE_TIP,
    FORCES_TIP,
    NOTES_HTML,
    PROBLEMS,
    SPEED_LINE_TIP,
    STRIP_TIP,
    TIPS,
    TR_TIP,
    TRIALS,
)
from .isolation_view import IsolationView
from .layouts import FlowLayout
from .lesson import ActionToggle, Form, LiveLabel, TrialBox, combo, note_label, plot, set_quietly, three_panes
from .live_parts import ChangeMarks, run_row
from .notes_view import NotesBrowser
from .plots import decimate
from .scrub import Scrub
from .style import colors
from .theming import add_legend
from .transport import Transport

__all__ = ["NOTES_HTML", "PROBLEMS", "TRIALS", "IsolationPage"]

STRIP_SECONDS = 8.0  # the time plots' window
PEAK_SECONDS = 3.0  # the drawing's scale fits the largest swing this recent
MEASURE_EVERY = 0.2  # s of run between the dwell's looks
FORCE_EVERY = 0.1  # s of run between the Floor force tab's redraws
PLAYED_AFTER = 2.0  # s of run after the reader's first change: they've handled it, predictions on
SYSTEM_FIELDS = ("mass", "unbalance", "block", "mount", "deflection_mm", "zeta", "stiffness_ratio", "stage_mass")


class IsolationPage(QtWidgets.QWidget):
    """The machine's controls on the left, the live machine and its time plots in the center, the tabs on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def transport(self) -> Transport:
        return Transport(self.run_button, self.reset_button, self.speed, self.clock)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.live = LiveIsolation()
        self.recorder = rec.Recorder()
        self.sweep: rec.Sweep | None = None
        self.running = False
        self.arrows = ArrowScale(room=1.0)
        self.floor_arrows = ArrowScale(room=1.0)
        self._handled_at: float | None = None
        self._grip_start = 0.0
        self._measured_at = -1.0
        self._forced_at = -1.0
        self._force_since = 0.0  # the Floor force tab's turns start here: the last change of the system or what shakes
        self.bindings = Bindings(IsolationTrial, self)

        # --- left: the machine, what shakes, the mounts, then the Things to try
        machine = Form("Machine")
        self.mass = machine.number("Mass M:", 0.1, 1e6, 100.0, 1, " kg", TIPS["mass"])
        self.rpm = machine.number("Speed:", 0.0, 1e5, 1500.0, 0, " rpm", TIPS["rpm"])
        self.unbalance = machine.number("Unbalance mₑe:", 0.0, 100.0, 0.01, 5, " kg·m", TIPS["unbalance"])
        self.block = machine.number("Inertia block:", 0.0, 1e6, 0.0, 1, " kg", TIPS["block"])
        self.run_up = ActionToggle("Run up", "Running up: click to hold")  # lit while the speed climbs
        self.run_up.setToolTip(TIPS["run_up"])
        self.run_up.toggled.connect(self._on_run_up)
        self.run_down = ActionToggle("Run down", "Running down: click to hold")
        self.run_down.setToolTip(TIPS["run_down"])
        self.run_down.toggled.connect(self._on_run_down)
        self.ramp = machine.number("Run up/down in:", 0.1, 600.0, 2.0, 1, " s", TIPS["ramp"])
        ramps = FlowLayout()
        ramps.add(self.run_up, self.run_down)
        machine.row(ramps)
        self.shaking = combo(li.SHAKING, TIPS["shaking"])
        machine.add("What shakes:", self.shaking, TIPS["shaking"])

        mounts = Form("Mounts")
        self.mount = combo([m.value for m in MOUNTS], TIPS["mount"])
        mounts.add("Kind:", self.mount, TIPS["mount"])
        self.deflection = mounts.number("Static sag δ:", 0.01, 1000.0, 5.0, 2, " mm", TIPS["deflection"])
        self.zeta = mounts.number("Damping ζ:", 0.0, 2.0, 0.05, 3, "", TIPS["zeta"])
        self.stiffness_ratio = mounts.number("Damper's spring N:", 0.01, 1000.0, 3.0, 2, " × k", TIPS["ratio"])
        self.stage_mass = mounts.number("Intermediate mass:", 0.01, 100.0, 0.5, 2, " × M", TIPS["stage"])
        self.mount_info = note_label()
        mounts.row(self.mount_info)
        self.mounts_box = mounts

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Setups to come back to, each showing one idea from the Theory tab. Once you've handled the machine, "
            "some ask you to predict what you'll see first.</p>",
            predictions=[t.predict for t in TRIALS],
            ask_after_play=True,
        )
        self.trials.picked.connect(self.load_trial)
        self.challenge = ChallengeBox(self, PROBLEMS)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        for w in (machine, mounts, self.trials, self.challenge):
            lv.addWidget(w)
        lv.addStretch(1)

        # --- center: the machine, the run, the time plots
        self.body = IsolationView()
        self.body.setMinimumHeight(200)
        self.body.grabbed.connect(self._on_grabbed)
        self.body.pulled.connect(self._on_pulled)
        self.body.released.connect(self._on_let_go)
        self.body.grip_taken.connect(self._on_grip_taken)
        self.body.grip_moved.connect(self._on_grip_moved)
        self.forces = FreeBodyView(name="The forces on the machine", vertical=True)  # up and down, as it moves
        row, self.speed, self.run_button, self.reset_button, self.clock = run_row(self)
        self.run_button.toggled.connect(self._on_run)
        self.reset_button.setToolTip(
            "<p>Back to rest at t = 0, at the Speed box's speed; the measured points are kept.</p>"
        )
        self.reset_button.clicked.connect(self.reset)

        self.strip = plot("Displacement", "m", "Time", "s", height=120)
        self.strip.setToolTip(STRIP_TIP)
        self.strip.setMouseEnabled(x=False, y=False)
        self.g_item = pg.PlotDataItem(name="the floor")
        self.x_item = pg.PlotDataItem(name="the machine")
        for item in (self.g_item, self.x_item):
            self.strip.addItem(item)
        self.force_strip = plot("Force", "N", "Time", "s", height=100)
        self.force_strip.setToolTip(FORCES_TIP)
        self.force_strip.setMouseEnabled(x=False, y=False)
        self.force_strip.setXLink(self.strip)
        add_legend(self.force_strip.getPlotItem(), offset=(10, 5))
        self.unbalance_item = pg.PlotDataItem(name="unbalance (rigid mount)")
        self.floor_item = pg.PlotDataItem(name="on the floor")
        for item in (self.unbalance_item, self.floor_item):
            self.force_strip.addItem(item)
        decimate(self.strip, self.force_strip)
        self.marks = ChangeMarks(self.strip, self.force_strip)
        self.scrub = Scrub(self.strip, self.force_strip)
        self.scrub.scrubbed.connect(self._on_scrubbed)
        self.readout = LiveLabel(wrap=True)
        # room for its longest texts, so a longer one (the floor's, a steady state's) doesn't lay the page out again
        self.readout.setMinimumHeight(3 * self.readout.fontMetrics().lineSpacing())

        centre = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(centre)
        cv.setContentsMargins(0, 0, 0, 0)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.body, 1)
        top.addWidget(self.forces, 1)
        cv.addLayout(top, 3)
        cv.addLayout(row)
        cv.addWidget(self.strip, 2)
        cv.addWidget(self.force_strip, 2)
        cv.addWidget(self.readout)

        # --- right: Transmissibility, Floor force, Design chart, Theory
        tr = QtWidgets.QWidget()
        tv = QtWidgets.QVBoxLayout(tr)
        self.sweep_button = ActionToggle("Sweep", "Sweeping: click to stop")
        self.sweep_button.setToolTip(
            "<p>Step the speed across the mounts' resonance, from 0.4 to 5 f<sub>n</sub>, dwelling at each until its "
            "point lands. Press again to stop.</p>"
        )
        self.sweep_button.toggled.connect(self._on_sweep)
        self.clear_button = QtWidgets.QPushButton("Clear points")
        self.clear_button.setToolTip("<p>Forget the measured points.</p>")
        self.clear_button.clicked.connect(self.clear_points)
        self.theory = QtWidgets.QCheckBox("Show the theory")
        self.theory.setToolTip(TIPS["theory"])
        buttons = FlowLayout()
        buttons.add(self.sweep_button, self.clear_button, self.theory)
        tv.addLayout(buttons)
        self.tr_plot = plot("Transmissibility T", None, "Frequency / fₙ", None, log_x=True, log_y=True, height=200)
        self.tr_plot.setToolTip(TR_TIP)
        self.tr_plot.setMouseEnabled(x=False, y=False)
        self.tr_plot.enableAutoRange(x=False, y=False)
        self.tr_plot.setXRange(math.log10(view.RATIOS[0]), math.log10(view.RATIOS[-1]), padding=0)
        self.tr_plot.setYRange(*view.T_RANGE, padding=0)
        add_legend(self.tr_plot.getPlotItem(), offset=(-10, 10))
        self.family_items = [pg.PlotDataItem() for _ in FAMILY]  # the theory, unnamed: shown only when asked for
        self.tr_item = pg.PlotDataItem()
        self.unit_line = pg.InfiniteLine(angle=0, pos=0.0)
        self.root2_line = pg.InfiniteLine(angle=90, pos=math.log10(math.sqrt(2)))
        # PlotDataItems, not ScatterPlotItems: only they follow the log axes
        self.earlier_item = pg.PlotDataItem(pen=None, symbol="o", symbolSize=6)
        self.machine_points = pg.PlotDataItem(pen=None, symbol="o", symbolSize=9, name="the machine shaking")
        self.floor_points = pg.PlotDataItem(pen=None, symbol="d", symbolSize=11, name="the floor shaking")
        for item in (
            self.unit_line,
            self.root2_line,
            *self.family_items,
            self.tr_item,
            self.earlier_item,
            self.machine_points,
            self.floor_points,
        ):
            self.tr_plot.addItem(item)
        self.speed_line = DragLine(SPEED_LINE_TIP, log=True, bounds=(view.RATIOS[0], view.RATIOS[-1]))
        self.speed_line.dragged.connect(self._on_speed_dragged)
        self.tr_plot.addItem(self.speed_line)
        self.tr_note = note_label()
        tv.addWidget(self.tr_plot, 1)
        tv.addWidget(self.tr_note)

        fv_widget = QtWidgets.QWidget()
        fv = QtWidgets.QVBoxLayout(fv_widget)
        self.force_plot = plot("Force on the floor", "N", "Speed", "rpm", log_y=True, height=200)
        self.force_plot.setToolTip(FORCE_TIP)
        self.force_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.force_plot.getPlotItem(), offset=(10, 10))
        self.rigid_item = pg.PlotDataItem(name="rigid mount (mₑeΩ²)")
        self.steady_item = pg.PlotDataItem()  # the theory, unnamed: shown only when asked for
        self.built_item = pg.PlotDataItem(name="on these mounts")
        self.fn_line = pg.InfiniteLine(angle=90)
        for item in (self.rigid_item, self.steady_item, self.built_item, self.fn_line):
            self.force_plot.addItem(item)
        self.theory_too = QtWidgets.QCheckBox("Show the theory")
        self.theory_too.setToolTip(TIPS["theory"])
        self.theory_too.toggled.connect(self._on_theory_too)
        fbuttons = FlowLayout()
        fbuttons.add(self.theory_too)
        fv.addLayout(fbuttons)
        fv.addWidget(self.force_plot, 1)
        self.force_note = note_label()
        fv.addWidget(self.force_note)

        chart_widget = QtWidgets.QWidget()
        chv = QtWidgets.QVBoxLayout(chart_widget)
        design = Form("Design")
        self.target = design.number("Isolation wanted:", 1.0, 99.99, 90.0, 2, " %", TIPS["target"])
        self.design_note = note_label()
        design.row(self.design_note)
        self.apply_design = QtWidgets.QPushButton("Use that sag")
        self.apply_design.setToolTip(
            "<p>Set the mounts' static deflection to the one the target needs (for undamped single-stage mounts).</p>"
        )
        self.apply_design.clicked.connect(self._on_apply_design)
        design.row(self.apply_design)
        self.chart_plot = plot("Static deflection", "mm", "Speed", "rpm", log_x=True, log_y=True, height=200)
        self.chart_plot.setToolTip(CHART_TIP)
        self.chart_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.chart_plot.getPlotItem(), offset=(-10, 10))
        self.chart_items = [pg.PlotDataItem(name=f"{sig(100 * lv_, 4)} %") for lv_ in CHART_LEVELS]
        self.chart_now = pg.PlotDataItem(pen=None, symbol="o", symbolSize=12)
        for item in (*self.chart_items, self.chart_now):
            self.chart_plot.addItem(item)
        self.chart_note = note_label()
        chv.addWidget(design)
        chv.addWidget(self.chart_plot, 1)
        chv.addWidget(self.chart_note)

        self.notes = NotesBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(tr, "Transmissibility")
        self.tabs.addTab(fv_widget, "Floor force")
        self.tabs.addTab(chart_widget, "Design chart")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._draw_tab)

        three_panes(self, left, centre, self.tabs, sizes=(320, 760, 560))

        self.bindings.bind("mount", self.mount, choices=tuple(MOUNTS))
        for field, widget in (
            ("mass", self.mass),
            ("rpm", self.rpm),
            ("unbalance", self.unbalance),
            ("block", self.block),
            ("ramp", self.ramp),
            ("shaking", self.shaking),
            ("deflection_mm", self.deflection),
            ("zeta", self.zeta),
            ("stiffness_ratio", self.stiffness_ratio),
            ("stage_mass", self.stage_mass),
            ("target", self.target),
            ("theory", self.theory),
            ("tab", self.tabs),
        ):
            self.bindings.bind(field, widget)
        self.bindings.write(IsolationTrial("", ""))
        self.bindings.changed.connect(self._on_field)

        self.loop = FrameLoop(self, "Vibration isolation", self._tick, self._runs, self._sim_t)
        self.apply_theme()
        self._push_system()
        self.reset()

    # ------------------------------------------------------------- the state
    def practice_state(self) -> IsolationTrial:
        """The page's settings, as a trial: what a practice spec checks, and Defaults puts back."""
        return self.bindings.read(IsolationTrial("", ""))

    @property
    def iso(self):
        return isolator(self.practice_state())

    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(IsolationTrial("", "", **setup))

    def apply_trial(self, t: IsolationTrial) -> None:
        """A setup: the machine and its mounts as it says, the run from rest, then set going as `start` says."""
        self.bindings.write(t)
        set_quietly(self.theory_too, t.theory)
        self.recorder.clear()
        self._stop_sweep()
        self.live.set_shaking(t.shaking)
        self._push_system()
        self.reset()
        self._handled_at = None
        self.trials.show_note(t.note)
        if t.start == view.START_RUN:  # at its speed from rest
            self.live.set_speed(t.rpm)
        elif t.start == view.START_RUN_UP:
            self.live.set_speed(t.rpm, t.ramp)
            set_quietly(self.run_up, True)
        self.live.edits.clear()
        self.live.changed_at = t.ramp if t.start == view.START_RUN_UP else 0.0
        if t.start is not None:
            self._run(True)
        self._draw_tab()

    # ------------------------------------------------------------- what the reader changes
    def _on_field(self, field: str) -> None:
        if field in SYSTEM_FIELDS:
            self._handled()
            self._push_system()
        elif field == "rpm":
            self._handled()
            if self.sweep is not None and not self._sweeping_now:
                self.sweep_button.setChecked(False)  # the reader took the speed over
            live = self.live
            if self._sweeping_now or (live.target > 0 and not self.run_down.isChecked()):
                if self.run_up.isChecked():  # climbing: on to the new speed at the same rate
                    live.set_speed(self.rpm.value(), abs(TWO_PI_RPM * self.rpm.value() - live.omega) / live.rate)
                else:
                    live.set_speed(self.rpm.value())  # running: at once, the angle running on
            # stopped (or running down), the box is the speed Run up goes to
            self._draw_speed()
        elif field == "shaking":
            self._handled()
            self.live.set_shaking(self.shaking.currentIndex())
            self._force_since = self.live.t
            self._draw_live()
            self._draw_tab()
        elif field == "theory":
            set_quietly(self.theory_too, self.theory.isChecked())
            self._draw_tab()
        elif field == "target":
            self._draw_chart()
        elif field == "tab":
            self._draw_tab()

    _sweeping_now = False  # the sweep, not the reader, is moving the speed

    def _handled(self) -> None:
        if self._handled_at is None:
            self._handled_at = self.live.t

    def _on_theory_too(self, on: bool) -> None:
        self.theory.setChecked(on)

    def _push_system(self) -> None:
        """The live machine to the controls: a live change by `isolation.carry`'s rules."""
        state = self.practice_state()
        iso = isolator(state)
        for widget, shown in (
            (self.stiffness_ratio, iso.mount is Mount.RELAXATION),
            (self.stage_mass, iso.mount is Mount.TWO_STAGE),
        ):
            widget.setVisible(shown)
            label = self.mounts_box.label(widget)
            if label is not None:
                label.setVisible(shown)
        if self.live.iso.mount is not iso.mount or self.live.iso.block != iso.block:
            self.body.set_layout(iso.mount, iso.block > 0)
        self.live.set_isolator(iso)
        self._force_since = self.live.t
        self.body.set_values(iso.total, state.deflection_mm, state.zeta)
        self.mount_info.setText(view.mount_info(iso))
        self._draw_speed()
        self._draw_chart()
        if not self.running:
            self._draw_live()
            self._draw_tab()

    def _on_run_up(self, on: bool) -> None:
        self._ramp(on, self.run_up, self.run_down, self.rpm.value())

    def _on_run_down(self, on: bool) -> None:
        self._stop_sweep()
        self._ramp(on, self.run_down, self.run_up, 0.0)

    def _ramp(self, on: bool, button: ActionToggle, other: ActionToggle, rpm: float) -> None:
        """Run up or down to `rpm` over the ramp's time, from the speed now (`on`), or hold the speed it's at."""
        self._handled()
        live = self.live
        if not on:  # pressed again: hold the speed it's reached
            live.set_speed(60.0 * live.speed_hz)
            return
        set_quietly(other, False)
        full = max(self.rpm.value(), 60.0 * live.speed_hz, 1e-9)  # the whole ramp's span, in its time
        seconds = self.ramp.value() * abs(rpm - 60.0 * live.speed_hz) / full
        live.set_speed(rpm, seconds)
        if not live.ramping:
            set_quietly(button, False)
        if not self.running:
            self._run(True)

    def _on_speed_dragged(self, ratio: float) -> None:
        self.rpm.setValue(round(60.0 * ratio * self.iso.natural_hz))

    def _on_apply_design(self) -> None:
        s = self.practice_state()
        try:
            d = deflection_for(self.iso.speed_hz, s.target / 100.0)
        except ValueError:
            return
        self.deflection.setValue(round(1e3 * d, 2))

    # ------------------------------------------------------------- the hand
    def _on_grabbed(self) -> None:
        self.scrub.hide()
        self._handled()
        self.live.hold(self.live.x)

    def _on_pulled(self, x: float) -> None:
        self.live.hold(x)
        self._draw_body()

    def _on_let_go(self) -> None:
        self.live.let_go()
        if not self.running:
            self._run(True)  # a pluck sets it going

    def _grip_box(self, name: str) -> QtWidgets.QDoubleSpinBox:
        return {"mass": self.mass, "deflection_mm": self.deflection, "zeta": self.zeta}[name]

    def _on_grip_taken(self, name: str) -> None:
        self._grip_start = self._grip_box(name).value()

    def _on_grip_moved(self, name: str, distance: float) -> None:
        set_quietly(self._grip_box(name), GRIPS[name].dragged(self._grip_start, distance))
        self._handled()
        self._push_system()

    # ------------------------------------------------------------- the run
    def _on_run(self, on: bool) -> None:
        self.running = on
        if on:
            self.scrub.hide()
            self.loop.start()
        else:
            self._show_scrub()

    def _run(self, on: bool) -> None:
        self.run_button.set_running(on)
        self._on_run(on)

    def _runs(self) -> bool:
        return self.running

    def _sim_t(self) -> float:
        return self.live.t

    def reset(self) -> None:
        """Back to rest at t = 0, the machine stopped, waiting for Start or Run up; the measured points kept."""
        self.live.reset()
        self.live.set_speed(0.0)
        for button in (self.run_up, self.run_down):
            set_quietly(button, False)
        self.live.edits.clear()
        self.live.changed_at = 0.0
        self.arrows.reset()
        self.floor_arrows.reset()
        self._measured_at = self._forced_at = -1.0
        self._force_since = 0.0
        self.scrub.hide()
        self.running = False
        self.run_button.rest()
        if self.sweep is not None:
            self.sweep = rec.Sweep(self.sweep.freqs)
            self._set_sweep_speed()
        self._draw_live()
        self._draw_tab()

    def _tick(self, dt: float) -> None:
        self.live.advance(dt * float(self.speed.currentData()))
        self._measure()
        if self._handled_at is not None and self.live.t - self._handled_at >= PLAYED_AFTER:
            self.trials.set_played()
        self._draw_live(frame=True)

    def step(self, duration: float) -> None:
        """Run on by `duration` s of simulated time at once (tests), measuring and drawing as the frames would."""
        end = self.live.t + duration
        while self.live.t < end - 1e-9 and self.live.held_at is None:
            self.live.advance(min(0.05, end - self.live.t) + 1e-12)
            self._measure()
        if self._handled_at is not None and self.live.t - self._handled_at >= PLAYED_AFTER:
            self.trials.set_played()
        self._draw_live()
        self._draw_tab()

    # ------------------------------------------------------------- the dwell
    def _point_key(self) -> tuple:
        return (view.key(self.practice_state()), self.live.shaking)

    def _measure(self) -> None:
        """Land a point if the speed is steady and the swing settled (every MEASURE_EVERY s), and step a sweep on."""
        live = self.live
        if live.t - self._measured_at < MEASURE_EVERY:
            return
        self._measured_at = live.t
        m = li.dwell(live)
        if m is None or self.recorder.land(m, self._point_key()) is None:
            return
        self._draw_tr()
        if self.sweep is not None:
            if self.sweep.landed() is None:
                self._stop_sweep()
            else:
                self._set_sweep_speed()

    def _on_sweep(self, on: bool) -> None:
        if not on:
            self.sweep = None
            return
        self._handled()
        self.sweep = rec.Sweep(view.sweep_speeds(self.practice_state()))
        self._set_sweep_speed()
        if not self.running:
            self._run(True)

    def _set_sweep_speed(self) -> None:
        assert self.sweep is not None, "a sweep sets the speed"
        rpm = self.sweep.frequency  # the sweep's steps are speeds (rpm) here
        if rpm is None:
            return
        self._sweeping_now = True
        for button in (self.run_up, self.run_down):
            set_quietly(button, False)
        self.rpm.setValue(round(rpm, 1))
        self.live.set_speed(self.rpm.value())  # stopped or not, the sweep runs it at this speed
        self._sweeping_now = False

    def _stop_sweep(self) -> None:
        self.sweep = None
        set_quietly(self.sweep_button, False)

    def clear_points(self) -> None:
        self.recorder.clear()
        self._draw_tr()

    # ------------------------------------------------------------- scrubbing
    def _show_scrub(self) -> None:
        r = self.live.record
        if r.size > 1:
            end = float(r.t[r.size - 1])
            self.scrub.show(end, max(float(r.t[0]), end - STRIP_SECONDS))

    def _on_scrubbed(self, t: float) -> None:
        r = self.live.record
        ts = r.t[: r.size]
        at = {
            c: float(np.interp(t, ts, r.y[: r.size, COL[c]]))
            for c in ("x", "stage", "phase", "floor", "unbalance", "g")
        }
        phase = at["phase"] if self.live.shaking != FLOOR else 0.0
        self.body.set_state(at["x"], at["stage"], phase, at["floor"], at["unbalance"], at["g"])
        self.clock.setText(f"t = {sig(t, 4)} s (paused at {sig(self.live.t, 4)} s)")

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.body.apply_theme()
        self.forces.apply_theme()
        self.scrub.apply_theme()
        self.marks.apply_theme()
        self.speed_line.apply_theme()
        for button in (self.run_up, self.run_down, self.sweep_button):
            button.apply_theme()
        dash = QtCore.Qt.PenStyle.DashLine
        # 1 px, the time plots' dense 25 Hz bands: Qt's fast path (wider ones cost 2x and more on Retina: the
        # floor-shaking scenario's two full-height bands painted in 17.7 ms a frame on the Mac, 44.5 fps)
        self.x_item.setPen(pg.mkPen(c.mass[0], width=1))
        self.g_item.setPen(pg.mkPen(c.grey, width=1))
        self.floor_item.setPen(pg.mkPen(c.force, width=1))
        self.unbalance_item.setPen(pg.mkPen(c.grey, width=1))
        for item, z in zip(self.family_items, FAMILY):
            item.setPen(pg.mkPen(c.grey if z else c.faint, width=1))
        self.tr_item.setPen(pg.mkPen(c.mass[0], width=2.5))
        self.unit_line.setPen(pg.mkPen(c.faint, width=1))
        self.root2_line.setPen(pg.mkPen(c.faint, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.machine_points.setSymbolBrush(pg.mkBrush(c.mass[0]))
        self.machine_points.setSymbolPen(pg.mkPen(c.structure, width=1))
        self.floor_points.setSymbolBrush(pg.mkBrush(c.mass[1]))
        self.floor_points.setSymbolPen(pg.mkPen(c.structure, width=1))
        self.earlier_item.setSymbolBrush(pg.mkBrush(c.faint))
        self.earlier_item.setSymbolPen(pg.mkPen(c.faint))
        self.rigid_item.setPen(pg.mkPen(c.grey, width=1.5, style=dash))
        self.steady_item.setPen(pg.mkPen(c.mass[0], width=1.5))
        self.built_item.setPen(pg.mkPen(c.force, width=2.5))
        self.fn_line.setPen(pg.mkPen(c.faint, width=1, style=QtCore.Qt.PenStyle.DotLine))
        for item, color in zip(self.chart_items, c.mass * 2):
            item.setPen(pg.mkPen(color, width=2))
        self.chart_now.setSymbolBrush(pg.mkBrush(c.strong))
        self._draw_live()
        self._draw_tab()

    def _draw_speed(self) -> None:
        self.speed_line.set_value(min(max(self.iso.speed_hz / self.iso.natural_hz, view.RATIOS[0]), view.RATIOS[-1]))

    @coalesce.coalesced
    def _draw_live(self, frame: bool = False) -> None:
        """What moves: the machine and its forces, the time plots, the readout (and, as a `frame` of the run, the
        Floor force tab at its own rate)."""
        live = self.live
        t, y = live.record.window(STRIP_SECONDS)
        end = live.t
        floor_shakes = live.shaking == FLOOR
        self.x_item.setData(t, y[:, COL["x"]])
        self.g_item.setVisible(floor_shakes)
        if floor_shakes:
            self.g_item.setData(t, y[:, COL["g"]])
        self.floor_item.setData(t, y[:, COL["floor"]])
        if floor_shakes:  # stopped: (emptied, not hidden, so its legend's entry stays as it is)
            self.unbalance_item.setData([], [])
        else:
            self.unbalance_item.setData(t, y[:, COL["unbalance"]])
        self.strip.setXRange(max(0.0, end - STRIP_SECONDS), max(end, STRIP_SECONDS), padding=0)
        self.marks.draw(live.edits, end - STRIP_SECONDS)
        recent = t >= end - PEAK_SECONDS
        swing = float(np.abs(y[recent, COL["x"]]).max()) if recent.any() else 0.0
        if floor_shakes:
            swing = max(swing, li.FLOOR_AMPLITUDE)
        if live.held_at is not None:
            swing = max(swing, abs(live.held_at))
        forces = float(np.abs(y[recent][:, [COL["floor"], COL["unbalance"]]]).max()) if recent.any() else 0.0
        self.body.set_scale(max(swing, 1e-6), max(forces, 1e-6))
        self._draw_body()
        way = (" ▲" if live.target > live.omega else " ▼") if live.ramping else ""
        self.clock.setText(f"t = {sig(end, 4)} s, {60 * live.speed_hz:.0f} rpm{way}")
        self._show_readout()
        if not live.ramping:  # the ramp has ended: its button goes out
            for button in (self.run_up, self.run_down):
                if button.isChecked():
                    set_quietly(button, False)
        self.run_up.setEnabled(self.run_up.isChecked() or live.target < TWO_PI_RPM * self.rpm.value() * (1 - 1e-9))
        self.run_down.setEnabled(self.run_down.isChecked() or live.target > 0)
        if frame and self.tabs.currentIndex() == view.TAB_FORCE and end - self._forced_at >= FORCE_EVERY:
            self._draw_force()

    def _draw_body(self) -> None:
        live = self.live
        stage = float(live.z[1]) if live.iso.mount is Mount.TWO_STAGE else 0.0
        f, g, _ = (float(u) for u in live.inputs())
        phase = live.phase if live.shaking != FLOOR else 0.0  # the floor shaking, the unbalance is stopped
        self.body.set_state(live.x, stage, phase, live.floor_force(), f, g)
        self.forces.show(li.free_body(live, self.arrows, self.floor_arrows))

    def _show_readout(self) -> None:
        live = self.live
        if live.held_at is not None:
            self.readout.setText(
                f"Held at x = {fmt(live.held_at, 'm', largest='')}: let go and it rings on its mounts."
            )
            return
        rpm = 60 * live.speed_hz
        if live.ramping:
            way = "up" if live.target > live.omega else "down"
            self.readout.setText(f"Running {way}: {rpm:.0f} rpm; {self._input_text()}.")
            return
        if rpm <= 0:
            self.readout.setText("Stopped. Pull the machine and let go, or Run up.")
            return
        f = live.speed_hz
        points = [p for p in self.recorder.current(self._point_key()) if abs(p.freq_hz - f) <= 1e-6 * f]
        if points:
            iso = dataclasses.replace(self.iso, rpm=rpm)
            self.readout.setText(view.steady_note(iso, abs(points[-1].response), live.shaking))
        else:
            self.readout.setText(
                f"At {rpm:.0f} rpm, {self._input_text()}: waiting for the mounts' own ring to die away…"
            )

    def _input_text(self) -> str:
        """What shakes it now, at the speed now: the unbalance's force, or the floor's motion (fixed)."""
        live = self.live
        if live.shaking == FLOOR:
            return f"the floor moves ±{fmt(li.FLOOR_AMPLITUDE, 'm')} at {sig(live.speed_hz)} Hz"
        return f"the unbalance pulls ±{fmt(float(live.iso.force_at(live.speed_hz)), 'N')} (mₑeΩ²)"

    @coalesce.coalesced
    def _draw_tab(self, *_: object) -> None:
        i = self.tabs.currentIndex()
        if i == view.TAB_T:
            self._draw_tr()
        elif i == view.TAB_FORCE:
            self._draw_force()
        elif i == view.TAB_CHART:
            self._draw_chart()

    @coalesce.coalesced
    def _draw_tr(self, *_: object) -> None:
        state = self.practice_state()
        v = view.t_view(state, self.recorder.points, state.theory)
        self.machine_points.setData(*v.machine)
        self.floor_points.setData(*v.floor)
        self.earlier_item.setData(*v.earlier)
        self.tr_item.setData(*v.isolator)
        for item, curve in zip(self.family_items, v.family or [None] * len(FAMILY)):
            if curve is None:
                item.setData([], [])
            else:
                item.setData(*curve)
        self.tr_note.setText(v.note)

    def _force_turns(self) -> tuple[np.ndarray, np.ndarray]:
        r = self.live.record
        t, y = r.t[: r.size], r.y[: r.size]
        i = int(np.searchsorted(t, self._force_since))
        out = "x" if self.live.shaking == FLOOR else "floor"
        return li.by_speed(t[i:], y[i:, COL["speed"]], y[i:, COL["phase"]], y[i:, COL[out]])

    @coalesce.coalesced
    def _draw_force(self, *_: object) -> None:
        self._forced_at = self.live.t
        state = dataclasses.replace(self.practice_state(), shaking=self.live.shaking)
        rpm, peaks = self._force_turns()
        v = view.force_view(state, rpm, peaks, state.theory)
        self.force_plot.setLabel("left", v.title, units=v.unit)
        self.built_item.setData(*v.built)
        self.rigid_item.setData(*v.rigid)
        self.steady_item.setData(*v.theory)
        self.fn_line.setValue(60 * self.iso.natural_hz)
        self.force_plot.setXRange(*v.rpm_range, padding=0)
        self.force_note.setText(v.note)

    @coalesce.coalesced
    def _draw_chart(self, *_: object) -> None:
        iso = self.iso
        self.design_note.setText(view.design_note(iso, self.target.value() / 100.0))
        ch = view.chart(iso)
        for item, curve in zip(self.chart_items, ch.levels):
            item.setData(*curve)
        self.chart_now.setData(*ch.now)
        self.chart_note.setText(ch.note)


TWO_PI_RPM = 2 * math.pi / 60.0  # rad/s a rpm
