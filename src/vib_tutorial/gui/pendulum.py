"""Pendulums page (#478 Phase 4): oscillators whose stiffness is gravity, swinging live, changed as they swing.

The single oscillator turned: J for m, moments for forces, and gravity's mgd sin θ for the spring. The pendulum is
stepped as it swings (`view_models/live_pendulum.LivePendulum`: the whole equation and its linearized ghost),
not a swing worked out in advance and played back. Left: the pendulum (a bob on a string, a rod on a pivot, a disc
on a wire, or a mass upright on a spring), its size, the spring, gravity and damping, then the Things to try
(setups to go back to; their predictions are asked once the reader has handled the pendulum). Centre: the pendulum
(`pendulum_view.PendulumView`: pluck it, or drag its grips to change L, m, k<sub>t</sub> and the rod's pivot as it
swings, the angular momentum kept) with the moments on it beside it (`free_body_view.FreeBodyView`, the string's
pull under them), the run's controls, the swing against time beside its ghost, which keeps what came before a change
with the change marked, and the restoring moment against the angle. Right: the Period the reader builds swing by
swing (each whole swing lands its period at its amplitude, the theory's curves shown over them on request), the
Phase plane traced live over the orbits of constant energy, the Energy balance through every change, and the Theory.

What the page shows is `view_models/pendulum.py`'s; its trials, problems, tips and notes are `pendulum_trials.py`'s
(no Qt). The controls are bound to the state (`PendulumTrial`) by `bindings.Bindings`, so Defaults round-trips by
construction.

Working on it: `tests/test_pendulum_page.py` (the page live: pluck, grips, a change mid-swing keeping Jθ̇, the
period built by hand, the theory's axes, the moments, predictions after play), `test_live_pendulum.py` (the
Qt-free parts), `test_readiness.py`.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.pendulum import KINDS, Kind, Pendulum, period
from ..core.stepping import FRAME_BUDGET, Budget
from ..view_models import live_pendulum as lp
from ..view_models import pendulum as view
from ..view_models.free_body import ArrowScale
from ..view_models.live_pendulum import COL, LivePendulum
from ..view_models.pendulum import PendulumTrial
from ..view_models.text import number as sig
from . import coalesce
from .bindings import Bindings
from .frame_loop import FrameLoop
from .free_body_view import FreeBodyView
from .layouts import FlowLayout
from .lesson import Dot, Form, LiveLabel, TrialBox, combo, note_label, plot, theory, three_panes
from .live_parts import ChangeMarks, EnergyTab, run_row
from .pendulum_trials import NOTES_HTML, PROBLEMS, TIPS, TRIALS
from .pendulum_view import PendulumView
from .plots import decimate
from .scrub import Scrub
from .style import colors
from .theming import add_legend
from .transport import Transport

__all__ = ["NOTES_HTML", "PROBLEMS", "TRIALS", "PendulumPage"]

STRIP_SECONDS = 20.0  # the swing plot's window: ten swings of the 1 m pendulum
PEAK_SECONDS = 4.0  # the drawing's half or whole plane, and the moment plot's reach, follow the swing this recent
TRAIL_PERIODS = 1.5  # the phase plane's trail
MEASURE_EVERY = 0.2  # s of run between the period recorder's looks
PLAYED_AFTER = 2.0  # s of run after the reader's first change: they've handled it, predictions on
REACH_STEP = math.radians(10.0)  # the phase plane's and moment plot's reach is redrawn in steps of this
SWING_TIP = (
    "<p>The angle as it swings, the last 20 s: the whole equation (colour) and the linearized one, sin θ ≈ θ "
    "(grey), which drifts ahead on a large swing. A dashed line marks a change: the swing carries on through it. "
    "Paused, drag the line at the end back to see the pendulum then.</p>"
)
MOMENT_TIP = (
    "<p>The restoring moment against the angle: the whole one (colour) and its tangent at 0, the linearized "
    "stiffness (grey). The dot is where the pendulum is now. Gravity's sin θ bends a hanging pendulum's below its "
    "tangent: a softening spring.</p>"
)
PERIOD_TIP = (
    "<p>The period, built by hand: each whole swing lands a point, the time it took against how far it swung. "
    "Points from another pendulum (before a change of L, m, g or the spring) are faded. <i>Show the theory</i> "
    "draws the curves they should lie on: exact (the energy integral), Duffing's first harmonic balance, and the "
    "small swing's T₀.</p>"
)
PHASE_TIP = (
    "<p>The swing as a path in the (θ, θ̇) plane, the last swing and a half, over the undamped orbits of constant "
    "energy (grey), with the separatrix (bold) where there is one: the orbit through the upside-down balance, "
    "dividing swinging from going round.</p>"
)
MOMENTS_TIP = (
    "<p>The moments about the pivot now, a row each, the single oscillator's forces turned: the weight's mgd sin θ "
    "(the pendulum's spring), the torsion spring's, the damper's, your hand's while you hold it. The dashed row is "
    "their sum, Jθ̈. Under it, the pull along the string (the rod's pivot, the arm, the wire): it passes through the "
    "pivot, so it has no moment, but it's greatest at the bottom of a swing, where it also turns the bob.</p>"
)
ENERGY_TIP = (
    "<p>Where the energy went: what your hand put in (a pluck), what the damper took out, what each change did "
    "(shortening the string as it swings, a heavier bob caught, gravity changed), and what's stored now. Put in, "
    "less taken out, with the changes, is what's stored, at every instant.</p>"
)


class PendulumPage(QtWidgets.QWidget):
    """The pendulum's controls on the left, the live pendulum and its plots in the centre, the tabs on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def transport(self) -> Transport:
        return Transport(self.run_button, self.reset_button, self.speed, self.clock)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.live = LivePendulum()
        self.recorder = view.PeriodRecorder()
        self.running = False
        self.arrows = ArrowScale(room=1.0)
        self.pulls = ArrowScale(room=1.0)  # the string's pull, on a scale of its own
        self._handled_at: float | None = None
        self._grip_start = 0.0
        self._measured_at = -1.0
        self._budget = Budget()
        self._reach_drawn: tuple | None = None  # what the moment curve was drawn for
        self._phase_reach = math.pi  # the phase plane's half-width (rad)
        self._orbits_for: tuple | None = None  # what its orbits were drawn for
        self.bindings = Bindings(PendulumTrial, self)

        # --- left: the pendulum, then the Things to try
        box = Form("Pendulum")
        self.kind = combo([k.value for k in KINDS], TIPS["kind"])
        box.add("Kind:", self.kind, TIPS["kind"])
        self.mass = box.number("Mass m:", 0.01, 1e4, 1.0, 3, " kg", TIPS["mass"])
        self.length = box.number("Length L:", 0.05, 20.0, 1.0, 3, " m", TIPS["length"])
        self.length_label = box.label(self.length)
        self.pivot = box.number("Pivot from centre:", 0.001, 10.0, 0.5, 3, " m", TIPS["pivot"])
        self.spring = box.number("Torsion spring:", 0.0, 1e5, 15.0, 3, " N·m/rad", TIPS["spring"])
        self.gravity = box.number("Gravity g:", 0.0, 100.0, 9.81, 2, " m/s²", TIPS["gravity"])
        self.zeta = box.number("Damping ζ:", 0.0, 2.0, 0.0, 3, "", TIPS["zeta"])
        self.info = note_label()
        box.row(self.info)
        self.pendulum_box = box

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Setups to come back to, each showing one idea from the Theory tab. Once you've handled the "
            "pendulum, some ask you to predict what you'll see first.</p>",
            predictions=[t.predict for t in TRIALS],
            ask_after_play=True,
        )
        self.trials.picked.connect(self.load_trial)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(box)
        lv.addWidget(self.trials)
        lv.addStretch(1)

        # --- centre: the pendulum and its moments, the run, the swing and the moment curve
        self.view = PendulumView()
        self.view.setMinimumHeight(200)
        self.view.grabbed.connect(self._on_grabbed)
        self.view.turned.connect(self._on_turned)
        self.view.released.connect(self._on_let_go)
        self.view.grip_taken.connect(self._on_grip_taken)
        self.view.grip_moved.connect(self._on_grip_moved)
        self.forces = FreeBodyView(name="The moments on the pendulum", tip=MOMENTS_TIP)
        row, self.speed, self.run_button, self.reset_button, self.clock = run_row(self)
        self.run_button.toggled.connect(self._on_run)
        self.reset_button.setToolTip("<p>Back to rest at t = 0; the measured periods are kept.</p>")
        self.reset_button.clicked.connect(self.reset)

        self.swing_plot = plot("Angle θ", "°", "Time", "s", height=150)
        self.swing_plot.setToolTip(SWING_TIP)
        self.swing_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.swing_plot.getPlotItem(), offset=(-10, 10))
        self.ghost_item = pg.PlotDataItem(name="linearized (θ)")
        self.theta_item = pg.PlotDataItem(name="whole (sin θ)")
        self.swing_plot.addItem(self.ghost_item)
        self.swing_plot.addItem(self.theta_item)
        decimate(self.swing_plot)
        self.marks = ChangeMarks(self.swing_plot)
        self.scrub = Scrub(self.swing_plot)
        self.scrub.scrubbed.connect(self._on_scrubbed)
        self.moment_plot = plot("Moment", "N·m", "Angle θ", "°", height=110)
        self.moment_plot.setToolTip(MOMENT_TIP)
        self.moment_plot.setMouseEnabled(x=False, y=False)
        self.tangent_item = pg.PlotDataItem()
        self.moment_item = pg.PlotDataItem()
        self.moment_plot.addItem(self.tangent_item)
        self.moment_plot.addItem(self.moment_item)
        self.moment_now = Dot(10)
        self.moment_plot.addItem(self.moment_now, ignoreBounds=True)
        self.readout = LiveLabel(wrap=True)  # rewritten as it swings: no layout pass each time

        centre = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(centre)
        cv.setContentsMargins(0, 0, 0, 0)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.view, 3)
        top.addWidget(self.forces, 2)
        cv.addLayout(top, 3)
        cv.addLayout(row)
        cv.addWidget(self.swing_plot, 2)
        cv.addWidget(self.moment_plot, 1)
        cv.addWidget(self.readout)

        # --- right: Period, Phase plane, Energy, Theory
        periods = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(periods)
        self.theory = QtWidgets.QCheckBox("Show the theory")
        self.theory.setToolTip(TIPS["theory"])
        self.clear_button = QtWidgets.QPushButton("Clear points")
        self.clear_button.setToolTip("<p>Forget the measured periods.</p>")
        self.clear_button.clicked.connect(self.clear_points)
        buttons = FlowLayout()
        buttons.add(self.theory, self.clear_button)
        pv.addLayout(buttons)
        self.period_plot = plot("Period", "s", "Amplitude", "°", height=200)
        self.period_plot.setToolTip(PERIOD_TIP)
        self.period_plot.setMouseEnabled(x=False, y=False)
        self.period_plot.enableAutoRange(x=False, y=False)  # set from the view: the same with the theory or not
        add_legend(self.period_plot.getPlotItem(), offset=(10, 10))
        self.exact_item = pg.PlotDataItem(name="exact")
        self.duffing_item = pg.PlotDataItem(name="Duffing: 1st harmonic")
        self.linear_item = pg.PlotDataItem(name="small swing T₀")
        self.earlier_item = pg.PlotDataItem(pen=None, symbol="o", symbolSize=6)
        self.points_item = pg.PlotDataItem(pen=None, symbol="o", symbolSize=8, name="measured")
        for item in (self.linear_item, self.duffing_item, self.exact_item, self.earlier_item, self.points_item):
            self.period_plot.addItem(item)
        self.period_note = note_label()
        pv.addWidget(self.period_plot, 1)
        pv.addWidget(self.period_note)

        phase = QtWidgets.QWidget()
        hv = QtWidgets.QVBoxLayout(phase)
        self.phase_plot = plot("θ̇", "°/s", "θ", "°", height=200)
        self.phase_plot.setToolTip(PHASE_TIP)
        self.phase_plot.setMouseEnabled(x=False, y=False)
        self.orbits_item = pg.PlotDataItem(connect="finite")
        self.separatrix_item = pg.PlotDataItem()
        self.path_item = pg.PlotDataItem()
        for item in (self.orbits_item, self.separatrix_item, self.path_item):
            self.phase_plot.addItem(item)
        self.phase_now = Dot(10)
        self.phase_plot.addItem(self.phase_now, ignoreBounds=True)
        self.phase_note = note_label()
        hv.addWidget(self.phase_plot, 1)
        hv.addWidget(self.phase_note)

        self.energy = EnergyTab(
            balance=lp.energy_view, tip=ENERGY_TIP, empty="Pluck the pendulum to see where the energy goes."
        )
        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(periods, "Period")
        self.tabs.addTab(phase, "Phase plane")
        self.tabs.addTab(self.energy, "Energy")
        self.tabs.addTab(self.notes, "Theory")

        three_panes(self, left, centre, self.tabs, sizes=(320, 760, 560))

        self.bindings.bind("kind", self.kind, choices=KINDS)
        for field, widget in (
            ("mass", self.mass),
            ("length", self.length),
            ("pivot", self.pivot),
            ("spring", self.spring),
            ("gravity", self.gravity),
            ("zeta", self.zeta),
            ("theory", self.theory),
            ("tab", self.tabs),
        ):
            self.bindings.bind(field, widget)
        self.bindings.write(PendulumTrial("", ""))
        self.bindings.changed.connect(self._on_field)

        self.loop = FrameLoop(self, "Pendulums", self._tick, self._runs, self._sim_t)
        self.apply_theme()
        self._push_pendulum()

    # ------------------------------------------------------------- the state
    def practice_state(self) -> PendulumTrial:
        """The page's settings, as a trial: what a practice spec checks, and Defaults puts back."""
        return self.bindings.read(PendulumTrial("", ""))

    @property
    def pendulum(self) -> Pendulum:
        return self.live.pendulum

    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(PendulumTrial("", "", **setup))

    def apply_trial(self, t: PendulumTrial) -> None:
        """A setup: the pendulum as it says, the run from rest, then plucked from its angle."""
        self.bindings.write(t)
        self.recorder.clear()
        self._push_pendulum()
        self.reset()
        self._handled_at = None
        self.trials.show_note(t.note)
        if t.angle:
            self.live.hold(math.radians(t.angle))
            self.live.let_go()
            self._run(True)
        self._draw_tab()

    # ------------------------------------------------------------- what the reader changes
    def _on_field(self, field: str) -> None:
        if field == "theory":
            self._draw_period()
        elif field == "tab":
            self._draw_tab()
        else:
            self._handled()
            self._push_pendulum()

    def _handled(self) -> None:
        if self._handled_at is None:
            self._handled_at = self.live.t

    def _push_pendulum(self) -> None:
        """The live pendulum to the controls: a live change (its angular momentum kept)."""
        state = self.practice_state()
        p = view.pendulum(state)
        for widget, shown in (
            (self.pivot, p.kind is Kind.ROD),
            (self.spring, p.kind in (Kind.TORSION, Kind.INVERTED)),
            (self.gravity, p.kind is not Kind.TORSION),
        ):
            widget.setVisible(shown)
            label = self.pendulum_box.label(widget)
            if label is not None:
                label.setVisible(shown)
        if self.length_label is not None:
            self.length_label.setText("Radius r:" if p.kind is Kind.TORSION else "Length L:")
        self.live.set_pendulum(p)
        self.view.set_pendulum(p)
        self.info.setText(view.info(p))
        self._draw_period()
        if not self.running:
            self._draw_live()  # running, the next frame draws it

    def _box(self, name: str) -> QtWidgets.QDoubleSpinBox:
        return {"length": self.length, "mass": self.mass, "spring": self.spring, "pivot": self.pivot}[name]

    def _on_grip_taken(self, name: str) -> None:
        self._grip_start = self._box(name).value()

    def _on_grip_moved(self, name: str, value: float) -> None:
        start = self._grip_start
        if name == "length":
            g = view.GRIPS["length"]
            new = min(max(start * value, g.low), g.high)
        elif name == "pivot":
            new = view.pivot_dragged(self.length.value(), start, value)
        else:
            new = view.GRIPS[name].dragged(start, value)
        box = self._box(name)
        box.blockSignals(True)
        box.setValue(new)
        box.blockSignals(False)
        self._handled()
        self._push_pendulum()

    def _on_grabbed(self) -> None:
        self.scrub.hide()
        self._handled()
        self.live.hold(self.live.theta)

    def _on_turned(self, theta: float) -> None:
        self.live.hold(theta)
        self._draw_body()

    def _on_let_go(self) -> None:
        self.live.let_go()
        if not self.running:
            self._run(True)  # a pluck sets it going

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
        """Back to rest at t = 0, waiting for Start; the measured periods kept."""
        self.live.reset()
        self.arrows.reset()
        self.pulls.reset()
        self.recorder.restart()
        self._measured_at = -1.0
        self.scrub.hide()
        self.running = False
        self.run_button.rest()
        self._draw_live()

    def _tick(self, dt: float) -> None:
        duration = dt * float(self.speed.currentData())
        duration = min(duration, self._budget.allows(duration, FRAME_BUDGET))
        start = time.perf_counter()
        before = self.live.t
        self.live.advance(duration)
        if self.live.t > before:
            self._budget.spent(time.perf_counter() - start, self.live.t - before)
        self._measure()
        if self._handled_at is not None and self.live.t - self._handled_at >= PLAYED_AFTER:
            self.trials.set_played()
        self._draw_live(frame=True)

    def step(self, duration: float) -> None:
        """Run on by `duration` s of simulated time at once (tests), measuring and drawing as the frames would."""
        end = self.live.t + duration
        while self.live.t < end - 1e-9 and self.live.held is None:
            self.live.advance(min(0.1, end - self.live.t) + 1e-12)
            self._measure()
        if self._handled_at is not None and self.live.t - self._handled_at >= PLAYED_AFTER:
            self.trials.set_played()
        self._draw_live()

    # ------------------------------------------------------------- the period, built by hand
    def _measure(self) -> None:
        """Land the whole swings made since the last look (every MEASURE_EVERY s of run)."""
        live = self.live
        if live.held is not None or live.t - self._measured_at < MEASURE_EVERY:
            return
        self._measured_at = live.t
        t, y = live.since_change()
        if self.recorder.update(t, y[:, COL["theta"]], live.changed_at, view.key(live.pendulum)):
            self._draw_period()

    def clear_points(self) -> None:
        self.recorder.clear()
        self._draw_period()

    @coalesce.coalesced
    def _draw_period(self, *_: object) -> None:
        p, key = self.live.pendulum, view.key(self.live.pendulum)
        v = view.period_view(p, self.recorder.current(key), self.recorder.earlier(key), self.theory.isChecked())
        self.points_item.setData(*v.points)
        self.earlier_item.setData(*v.earlier)
        self.exact_item.setData(*v.exact, connect="finite")
        self.duffing_item.setData(*v.duffing, connect="finite")
        self.linear_item.setData(*v.linear)
        self.period_plot.setXRange(*v.x_span, padding=0)
        self.period_plot.setYRange(*v.y_span, padding=0)
        self.period_note.setText(v.note)

    # ------------------------------------------------------------- scrubbing
    def _show_scrub(self) -> None:
        r = self.live.record
        if r.size > 1:
            end = float(r.t[r.size - 1])
            self.scrub.show(end, max(float(r.t[0]), end - STRIP_SECONDS))

    def _on_scrubbed(self, t: float) -> None:
        r = self.live.record
        ts = r.t[: r.size]
        theta, omega, ghost = (float(np.interp(t, ts, r.y[: r.size, COL[c]])) for c in ("theta", "omega", "ghost"))
        p = self.live.pendulum
        self.view.show(theta, lp.ghost_angle(ghost))
        self.forces.show(view.free_body(p, theta, omega, scale=self.arrows, aside_scale=self.pulls, t=t))
        self.moment_now.setPos(math.degrees(theta), float(p.moment(theta)))
        self.phase_now.setPos(math.degrees(theta), math.degrees(omega))
        self.clock.setText(f"t = {sig(t, 4)} s (paused at {sig(self.live.t, 4)} s)")

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.view.apply_theme()
        self.forces.apply_theme()
        self.scrub.apply_theme()
        self.marks.apply_theme()
        self.energy.apply_theme()
        self.theta_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.ghost_item.setPen(pg.mkPen(c.grey, width=1.5))  # solid: a dashed line restroked each frame is dear
        self.moment_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.tangent_item.setPen(pg.mkPen(c.grey, width=1.5))
        for dot in (self.moment_now, self.phase_now):
            dot.setPen(pg.mkPen(c.background, width=1.5))
            dot.setBrush(pg.mkBrush(c.mass[0]))
        self.exact_item.setPen(pg.mkPen(c.strong, width=1.5))
        self.duffing_item.setPen(pg.mkPen(c.force, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.linear_item.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.points_item.setSymbolPen(None)
        self.points_item.setSymbolBrush(pg.mkBrush(c.mass[0]))
        self.earlier_item.setSymbolPen(pg.mkPen(c.grey, width=1))
        self.earlier_item.setSymbolBrush(None)
        self.orbits_item.setPen(pg.mkPen(c.faint, width=1))
        self.separatrix_item.setPen(pg.mkPen(c.strong, width=2))
        self.path_item.setPen(pg.mkPen(c.mass[0], width=2))
        self._draw_period()
        self._draw_tab()

    @coalesce.coalesced(covers=("_draw_live",))
    def _draw_tab(self, *_: object) -> None:
        self._draw_live()

    def _amplitude(self) -> float:
        """The largest angle of the last PEAK_SECONDS (rad), or where the hand holds it."""
        live = self.live
        if live.held is not None:
            return abs(live.held)
        t, y = live.record.window(PEAK_SECONDS)
        return float(np.abs(y[:, COL["theta"]]).max()) if t.size else abs(live.theta)

    @coalesce.coalesced
    def _draw_live(self, frame: bool = False) -> None:
        """What moves: the pendulum and its moments, the swing, the moment curve, the readout, and the tab showing
        (the energy as a `frame` of the run, at the Energy tab's own rate)."""
        live = self.live
        t, y = live.record.window(STRIP_SECONDS)
        end = live.t
        self.theta_item.setData(t, np.degrees(y[:, COL["theta"]]))
        ghost = np.degrees(y[:, COL["ghost"]])
        ghost[np.abs(ghost) > 720.0] = np.nan  # an unstable linearization runs away
        self.ghost_item.setData(t, ghost, connect="finite")
        self.swing_plot.setXRange(max(0.0, end - STRIP_SECONDS), max(end, STRIP_SECONDS), padding=0)
        self.marks.draw(live.edits, end - STRIP_SECONDS)
        amplitude = self._amplitude()
        self.view.frame(amplitude)
        self._draw_body()
        self._draw_reach(amplitude)
        self.clock.setText(f"t = {sig(end, 4)} s")
        self._show_readout()
        i = self.tabs.currentIndex()
        if i == view.TAB_PHASE:
            self._draw_phase()
        elif i == view.TAB_ENERGY:
            self.energy.draw(live.record, frame)

    def _draw_body(self) -> None:
        live, p = self.live, self.live.pendulum
        theta = live.theta
        self.view.show(theta, None if live.held is not None else lp.ghost_angle(live.ghost))
        held = live.held is not None
        self.forces.show(view.free_body(p, theta, live.omega, held, self.arrows, self.pulls, live.t))
        self.moment_now.setPos(math.degrees(theta), float(p.moment(theta)))
        self.phase_now.setPos(math.degrees(theta), math.degrees(live.omega))

    def _draw_reach(self, amplitude: float) -> None:
        """The moment curve and the phase plane's orbits over the swing's reach, redrawn only when the pendulum or
        the reach (in steps of REACH_STEP) changes."""
        p = self.live.pendulum
        stepped = math.ceil(amplitude / REACH_STEP) * REACH_STEP
        moment_reach = min(max(1.3 * stepped, 1.3 * abs(p.leaning()), 0.2), math.pi)
        phase_reach = view.phase_reach(p, stepped)
        drawn = (view.key(p), phase_reach, moment_reach)
        if drawn == self._reach_drawn:
            return
        self._reach_drawn = drawn
        m = view.moment_view(p, self.live.theta, moment_reach)
        self.moment_item.setData(*m.moment)
        self.tangent_item.setData(*m.tangent)
        self.moment_plot.setXRange(*m.x_span, padding=0)
        self._phase_reach = phase_reach

    def _draw_phase(self) -> None:
        live, p = self.live, self.live.pendulum
        period0 = p.period0 if p.stable else 2.0
        t, y = live.record.window(TRAIL_PERIODS * period0)
        if t.size:
            keep = t >= live.changed_at  # the path since the last change, not across it
            th, om = y[keep, COL["theta"]], y[keep, COL["omega"]]
        else:
            th = om = np.empty(0)
        drawn = (view.key(p), self._phase_reach)
        v = view.phase_view(p, th, om, self._phase_reach) if drawn != self._orbits_for else None
        if v is not None:
            self._orbits_for = drawn
            self.orbits_item.setData(*v.orbits)
            self.separatrix_item.setData(*v.separatrix)
            self.phase_plot.setXRange(*v.x_span, padding=0.02)
            self.phase_note.setText(v.note)
            self.path_item.setData(*v.path)
        else:
            self.path_item.setData(np.degrees(th), np.degrees(om))

    def _show_readout(self) -> None:
        live, p = self.live, self.live.pendulum
        if live.held is not None:
            self.readout.setText(f"Held at θ = {sig(math.degrees(live.held))}°: let go to pluck it.")
            return
        if not p.stable:
            lean = p.leaning()
            self.readout.setText(
                f"Unstable upright: it falls, and with damping settles leaning at ±{sig(math.degrees(lean))}° "
                "(kₜθ = mgd sin θ). The linearized ghost just runs away."
                if lean
                else "No restoring moment at θ = 0: nothing to oscillate about."
            )
            return
        current = self.recorder.current(view.key(p))
        if live.record.size < 2:
            self.readout.setText("Pluck it: drag the bob round and let go.")
            return
        if not current:
            self.readout.setText(f"Small-swing period T₀ = {sig(p.period0, 4)} s. Timing its first swing…")
            return
        last = max(current, key=lambda q: q.t)
        exact = period(replace(p, zeta=0.0), last.amplitude)
        self.readout.setText(
            f"The last swing, ±{sig(math.degrees(last.amplitude))}°, took {sig(last.period, 4)} s "
            f"({view.longer(last.period, p.period0)} on T₀ = {sig(p.period0, 4)} s; "
            f"undamped, exactly {sig(exact, 4)} s)."
        )
