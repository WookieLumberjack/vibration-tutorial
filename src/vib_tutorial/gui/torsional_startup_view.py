"""The Torsional page's start-up: a motor (induction direct on line, or synchronous on its damper cage) starting
the train from rest. Its controls (`controls`, the Start-up pane in the page's left column) and its tab (this
widget: a link's torque and the motor's speed against time, the status and the peak torques).

It runs the train the page gives it (`set_train`, each new one starting a fresh run from rest), on its own
`FrameLoop`: only while its tab shows. `load` sets a preset's motor (or none), `edited` tells the page the
reader changed the motor (the train is then a custom one, and the interference diagram redraws its 2× slip
line), `shown_wanted` asks it to show this tab (Start).

Working on it: the curves keep each one's min and max per pixel column of the run's whole span (`plots.Decimator`,
reset with the run or the link shown), and the ringing link's pen is 1 px: every step drawn, or a wider pen's
joins on its zigzag, made it 20 fps at 2× on a Mac. `test_gui_smoke.py`'s `test_torsional_start_up` checks the
points and extremes.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.torsional import PRESETS, DriveTrain, TorsionalMode
from ..core.torsional_startup import (
    FAN_START,
    Motor,
    MotorKind,
    StartUp,
    StartUpPreset,
    StartUpSimulator,
    crossings,
    running_torques,
    steady_speed,
)
from . import weak
from .frame_loop import FrameLoop
from .layouts import FlowLayout, form_layout, form_row, group, plot_height, show_in_stack
from .lesson import RunButton, connect_all, plot, set_quietly
from .panels import spin
from .plots import Decimator
from .style import colors
from .theming import add_legend, mute

FRAME_BUDGET = 0.008  # s of computing per frame: past it the start-up plays slower than asked
PLAYBACK = (0.05, 0.1, 0.25, 0.5, 1.0)  # × real time
MOTORS = (None, MotorKind.INDUCTION, MotorKind.SYNCHRONOUS)  # the motor combo's items
NO_MOTOR = "The engine drives this train: no motor starts it. Pick one to try anyway."
TIPS = {
    "motor": "<p>The motor on the first disc that starts the train from rest: an induction motor direct on "
    "line, or a synchronous motor started on its damper cage. None: no start-up.</p>",
    "supply": "<p>The line frequency and the motor's pole pairs: it runs at (or, an induction motor, just "
    "below) 60 f / p rpm.</p>",
    "breakdown": "<p>The cage's speed–torque curve (Kloss's): its peak torque, and the slip where it peaks. "
    "It starts the train, and an induction motor runs on it, where it meets the load.</p>",
    "transient": "<p>Switching an induction motor on gives a torque at the line frequency on top of its "
    "curve's, dying away with this time constant: the stator's flux starts with a DC part.</p>",
    "pulsating": "<p>A salient-pole synchronous motor's torque pulses at twice the slip frequency as it starts "
    "(T<sub>p</sub> sin 2δ): from twice the line frequency at rest to nothing at synchronism, "
    "sweeping through the modes below.</p>",
    "field": "<p>The field's pull-out torque (T<sub>f</sub> sin δ), switched on below the slip given, which "
    "pulls the rotor into step.</p>",
    "load": "<p>The load on the last disc, rising as its speed squared (a fan's, a centrifugal "
    "compressor's) to this, on its own shaft, with the motor at its synchronous speed.</p>",
    "duration": "<p>How long the start-up runs, in simulated time.</p>",
}
STARTUP_TIP = (
    "<p>The start-up: the chosen link's torque (a mesh: its force times the near gear's base radius) and the "
    "motor's air-gap torque against time, the dashed line what the link carries running steadily; below, "
    "the motor's speed, with the speeds where a synchronous motor's 2× slip pulsation meets each mode.</p>"
)


def fmt_torque(t: float) -> str:
    return f"{t / 1e3:.3g} kN·m" if abs(t) >= 1e3 else f"{t:.3g} N·m"


class StartUpView(QtWidgets.QWidget):
    """The Start-up tab, its controls (`controls`) and its run."""

    edited = QtCore.Signal()  # the reader changed the motor or the load
    shown_wanted = QtCore.Signal()  # Start: show this tab

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.train: DriveTrain = PRESETS[0].train
        self._modes: list[TorsionalMode] = []
        self.startup: StartUp = FAN_START
        self.motor_on = True
        self.sim: StartUpSimulator | None = None
        self.running = False
        self.note = ""  # the preset's, until something is edited
        self._loading = False

        self.controls = start_box = QtWidgets.QGroupBox("Start-up")
        stf = form_layout(start_box)
        self.motor_kind = QtWidgets.QComboBox()
        self.motor_kind.addItems(["None"] + [k.value for k in MOTORS[1:]])
        self.line_hz = spin(1.0, 400.0, 50.0, 1, " Hz")
        self.pole_pairs = QtWidgets.QSpinBox()
        self.pole_pairs.setRange(1, 50)
        self.pole_pairs.setKeyboardTracking(False)
        self.breakdown = spin(1.0, 1e7, 1000.0, 0, " N·m")
        self.breakdown_slip = spin(0.5, 100.0, 20.0, 1, " %")
        form_row(stf, "Motor:", self.motor_kind, TIPS["motor"])
        form_row(stf, "Supply:", group(self.line_hz, self.pole_pairs, "pole pairs"), TIPS["supply"])
        form_row(stf, "Breakdown:", group(self.breakdown, "at slip", self.breakdown_slip), TIPS["breakdown"])
        self.motor_stack = QtWidgets.QStackedWidget()
        induction_page = QtWidgets.QWidget()
        form = form_layout(induction_page)
        form.setContentsMargins(0, 0, 0, 0)
        self.transient = spin(0.0, 1e7, 0.0, 0, " N·m")
        self.time_constant = spin(1.0, 5000.0, 50.0, 0, " ms")
        form_row(form, "Switch-on torque:", group(self.transient, "decay", self.time_constant), TIPS["transient"])
        self.motor_stack.addWidget(induction_page)
        synchronous_page = QtWidgets.QWidget()
        form = form_layout(synchronous_page)
        form.setContentsMargins(0, 0, 0, 0)
        self.pulsating = spin(0.0, 1e7, 0.0, 0, " N·m")
        self.field = spin(0.0, 1e7, 0.0, 0, " N·m")
        self.field_slip = spin(0.1, 100.0, 5.0, 1, " %")
        form_row(form, "Pulsating (2× slip):", self.pulsating, TIPS["pulsating"])
        form_row(form, "Field:", group(self.field, "below slip", self.field_slip), TIPS["field"])
        self.motor_stack.addWidget(synchronous_page)
        stf.addRow(self.motor_stack)
        self.load_torque = spin(0.0, 1e7, 300.0, 0, " N·m")
        self.duration = spin(0.1, 60.0, 4.0, 1, " s")
        form_row(stf, "Load at full speed:", self.load_torque, TIPS["load"])
        form_row(stf, "Run for:", self.duration, TIPS["duration"])
        row = FlowLayout()
        self.pause_button = RunButton()
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setToolTip("<p>The train back at rest, waiting for Start.</p>")
        self.playback = QtWidgets.QComboBox()
        for x in PLAYBACK:
            self.playback.addItem(f"{x:g}× real time", x)
        self.playback.setCurrentIndex(PLAYBACK.index(0.25))
        self.playback.setToolTip(
            "<p>How fast the start-up plays: its first tenth of a second is what an "
            "induction motor's switch-on rings in.</p>"
        )
        row.add(self.pause_button, self.reset_button, self.playback)
        stf.addRow(row)
        self.startup_info = QtWidgets.QLabel()
        self.startup_info.setWordWrap(True)
        mute(self.startup_info)
        stf.addRow(self.startup_info)

        # the start-up: a link's torque and the motor's speed against time
        self.startup_link = QtWidgets.QComboBox()
        self.startup_link.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.startup_link.setMinimumContentsLength(18)
        self.startup_link.setToolTip(STARTUP_TIP)
        self.startup_torque = plot("Torque", "N·m", "Time", "s", log_ready=True)
        self.startup_torque.setToolTip(STARTUP_TIP)
        plot_height(self.startup_torque, 200)
        self.startup_legend = add_legend(self.startup_torque.getPlotItem(), offset=(-10, 10))
        self.startup_speed = plot("Motor speed", "rpm", "Time", "s", log_ready=True)
        self.startup_speed.setToolTip(STARTUP_TIP)
        self.startup_speed.setXLink(self.startup_torque)
        plot_height(self.startup_speed, 130)
        for p in (self.startup_torque, self.startup_speed):
            p.setClipToView(True)
            p.setDownsampling(auto=True, mode="peak")
        self.motor_curve = self.startup_torque.plot([], [])
        self.link_curve = self.startup_torque.plot([], [])
        self.startup_legend.addItem(self.link_curve, "Link")
        self.startup_legend.addItem(self.motor_curve, "Motor's air-gap torque")
        self.running_line = pg.InfiniteLine(0.0, angle=0)
        self.startup_torque.addItem(self.running_line)
        self.speed_curve = self.startup_speed.plot([], [])
        self._marks: list[tuple[pg.PlotWidget, pg.InfiniteLine]] = []  # the crossings' speeds, when the run passed them
        self.apply_theme()
        self.startup_status = QtWidgets.QLabel()
        self.startup_status.setWordWrap(True)
        mute(self.startup_status)
        self.startup_note = QtWidgets.QLabel()
        self.startup_note.setWordWrap(True)
        self.startup_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        uv = QtWidgets.QVBoxLayout(self)
        row = FlowLayout()
        row.add(group("Torque in:", self.startup_link))
        uv.addLayout(row)
        uv.addWidget(self.startup_torque, 3)
        uv.addWidget(self.startup_speed, 2)
        uv.addWidget(self.startup_status)
        uv.addWidget(self.startup_note)

        self.motor_kind.currentIndexChanged.connect(self._on_startup)
        connect_all(
            (
                self.line_hz,
                self.pole_pairs,
                self.breakdown,
                self.breakdown_slip,
                self.transient,
                self.time_constant,
                self.pulsating,
                self.field,
                self.field_slip,
                self.load_torque,
                self.duration,
            ),
            self._on_startup,
        )
        self.reset_button.clicked.connect(self.reset)
        self.pause_button.toggled.connect(self._on_pause)
        self.startup_link.currentIndexChanged.connect(weak.slot(self, lambda self, _: self._draw_startup(full=True)))

        self._cost = 1e-5  # s of computing per simulated s (see step)
        self.loop = FrameLoop(self, "Torsional vibration", self._tick, self._runs, self._sim_t)

    def load(self, start: StartUpPreset | None) -> None:
        """A preset's start-up (None: the train has no motor), at rest; `set_train` then starts its run."""
        self.motor_on = start is not None
        if start is not None:
            self.startup = start.startup
        self.note = start.note if start is not None else NO_MOTOR
        self.running = False
        self.startup_link.blockSignals(True)
        self.startup_link.setCurrentIndex(-1)  # _new_run picks the link carrying the most
        self.startup_link.blockSignals(False)
        self._fill()

    def set_train(self, train: DriveTrain, modes: list[TorsionalMode]) -> None:
        """A new train (an edit, a preset): a fresh run of it from rest, still playing if it was."""
        self.train, self._modes = train, modes
        self._new_run()

    def clear_note(self) -> None:
        """The preset's note goes once the train or the motor is the reader's own."""
        if self.note:
            self.note = ""
            self._show_info()

    def _fill(self) -> None:
        loading, self._loading = self._loading, True
        m = self.startup.motor
        self.motor_kind.setCurrentIndex(MOTORS.index(m.kind) if self.motor_on else 0)
        for box, value in (
            (self.line_hz, m.line_hz),
            (self.pole_pairs, m.pole_pairs),
            (self.breakdown, m.breakdown),
            (self.breakdown_slip, 100.0 * m.breakdown_slip),
            (self.transient, m.transient),
            (self.time_constant, 1e3 * m.time_constant),
            (self.pulsating, m.pulsating),
            (self.field, m.field),
            (self.field_slip, 100.0 * m.field_slip),
            (self.load_torque, self.startup.load),
            (self.duration, self.startup.duration),
        ):
            set_quietly(box, value)
        page = 0 if m.kind is MotorKind.INDUCTION else 1
        show_in_stack(self.motor_stack, page)
        for w in (
            self.line_hz,
            self.pole_pairs,
            self.breakdown,
            self.breakdown_slip,
            self.motor_stack,
            self.load_torque,
            self.duration,
        ):
            w.setEnabled(self.motor_on)
        self._loading = loading

    def _on_startup(self) -> None:
        if self._loading:
            return
        kind = MOTORS[self.motor_kind.currentIndex()]
        try:
            motor = Motor(
                kind or self.startup.motor.kind,
                self.line_hz.value(),
                self.pole_pairs.value(),
                self.breakdown.value(),
                self.breakdown_slip.value() / 100.0,
                self.transient.value(),
                self.time_constant.value() / 1e3,
                self.pulsating.value(),
                self.field.value(),
                self.field_slip.value() / 100.0,
            )
            startup = StartUp(motor, self.load_torque.value(), self.duration.value())
        except ValueError as e:
            self.startup_info.setText(f"<b>{e}</b>: the start-up keeps the last motor.")
            return
        self.startup, self.motor_on = startup, kind is not None
        self.edited.emit()  # the page: a custom train now, and the interference diagram's 2× slip line
        self._fill()
        self._new_run()

    def _new_run(self) -> None:
        """A fresh start-up of the train as it is now, from rest (still playing if it was)."""
        t, su = self.train, self.startup
        self.sim = StartUpSimulator(t, su) if self.motor_on else None
        cap = round(su.duration / self.sim.h) + 1 if self.sim is not None else 0
        self._k = 0
        self._t = np.empty(cap)
        self._rpm = np.empty(cap)
        self._motor = np.empty(cap)
        self._links = np.empty((t.n - 1, cap))
        self._decimator = Decimator()  # the curves' min and max per pixel column, kept as the run grows
        self._running = running_torques(t, su) if self.motor_on else np.full(t.n - 1, math.nan)
        self._crossings = crossings(t, su.motor) if self.motor_on else []
        self._passed: set[int] = set()
        chosen = self.startup_link.currentIndex()
        self.startup_link.blockSignals(True)
        self.startup_link.clear()
        self.startup_link.addItems([link.name for link in t.links])
        if not 0 <= chosen < len(t.links):
            chosen = int(np.argmax(np.nan_to_num(self._running)))  # the link carrying the most
        self.startup_link.setCurrentIndex(chosen)
        self.startup_link.blockSignals(False)
        for owner, mark in self._marks:
            owner.removeItem(mark)
        self._marks = []
        for c in self._crossings:
            self._add_mark(self.startup_speed, c.rpm, 0, f"Mode {c.mode + 1}, {c.freq_hz:.3g} Hz")
        self.startup_torque.setXRange(0.0, su.duration, padding=0)
        self.startup_torque.enableAutoRange(axis="y")
        self.startup_speed.setYRange(0.0, 1.1 * su.motor.sync_rpm, padding=0)
        if self.sim is None:
            self.running = False
        self._draw_startup(full=True)
        self._show_info()
        self._show_status()
        self._update_buttons()
        self.loop.start()

    def _add_mark(self, plot: pg.PlotWidget, pos: float, angle: int, label: str) -> None:
        line = pg.InfiniteLine(
            pos,
            angle=angle,
            label=label,
            labelOpts={"position": 0.02 if angle == 0 else 0.95, "anchors": [(0, 1), (0, 1)]},
        )
        plot.addItem(line)
        self._marks.append((plot, line))
        self._style_mark(line)

    def _style_mark(self, line: pg.InfiniteLine) -> None:
        line.setPen(pg.mkPen(colors.grey, width=1, style=QtCore.Qt.PenStyle.DotLine))
        line.label.setColor(colors.muted)

    def start(self) -> None:
        """Start the train from rest, and show it."""
        if not self.motor_on:
            return
        self.running = True
        self.shown_wanted.emit()
        self._new_run()

    def reset(self) -> None:
        """The train back at rest, waiting for Start."""
        self.running = False
        self._new_run()

    def _on_pause(self, on: bool) -> None:
        if self.sim is None:
            self._update_buttons()
        elif on and (self.sim.done or self._k == 0):  # from rest (again, if it ran to the end), shown
            self.start()
        else:
            self.running = on
            self._update_buttons()
            self.loop.start()

    def _update_buttons(self) -> None:
        self.reset_button.setEnabled(self.motor_on)
        self.pause_button.setEnabled(self.motor_on)
        if self.sim is None or self.sim.done or (self._k == 0 and not self.running):
            self.pause_button.rest()
        else:
            self.pause_button.set_running(self.running)

    def _runs(self) -> bool:
        return self.running

    def _tick(self, dt: float) -> None:
        """A frame: `dt` s of wall-clock time at the playback speed, within the frame's budget."""
        self.step(dt * float(self.playback.currentData()), FRAME_BUDGET)

    def _sim_t(self) -> float | None:
        return self.sim.t if self.sim is not None else None

    def step(self, duration: float, budget: float | None = None) -> None:
        """Run the start-up on by `duration` s of simulated time and redraw.

        With a `budget` (s of computing), no further than it allows at the cost seen so far, so a fast
        playback falls behind rather than stalling the window.
        """
        sim = self.sim
        if sim is None:
            return
        ahead = duration if budget is None else min(duration, budget / self._cost)
        if ahead >= 0.5 * sim.h and not sim.done:
            start, began = time.perf_counter(), sim.t
            while ahead >= 0.5 * sim.h and not sim.done:
                s = sim.advance(ahead)
                m = s.t.size
                if not m:
                    break
                k = self._k
                self._t[k : k + m] = s.t
                self._rpm[k : k + m] = s.rpm
                self._motor[k : k + m] = s.motor
                self._links[:, k : k + m] = s.links
                self._k += m
                ahead -= m * sim.h
            if sim.t > began:
                self._cost = 0.8 * self._cost + 0.2 * (time.perf_counter() - start) / (sim.t - began)
            self._draw_startup()
            self._show_status()
            if sim.done:
                self.running = False
                self._update_buttons()

    def _draw_startup(self, full: bool = False) -> None:
        k = self._k
        j = self.startup_link.currentIndex()
        if full:
            self.startup_legend.clear()
            name = self.train.links[j].name if 0 <= j < len(self.train.links) else "Link"
            self.startup_legend.addItem(self.link_curve, name)
            self.startup_legend.addItem(self.motor_curve, "Motor's air-gap torque")
            run = self._running[j] if 0 <= j < self._running.size else math.nan
            self.running_line.setVisible(bool(math.isfinite(run)))
            if math.isfinite(run):
                self.running_line.setPos(run)
        t = self._t[:k]
        link = self._links[j] if 0 <= j < self._links.shape[0] else np.zeros_like(self._t)
        # Every step is ~12,000 points a curve by the end, ringing fast: painting them all took 41 ms a frame
        # at 2× on a Mac (20 fps). Keep each curve's min and max per pixel column of the run's whole span.
        width = max(100, int(self.startup_torque.getPlotItem().vb.width()))
        total = max(k, self._t.size)
        if full:
            self._decimator.reset()
        if k > 2 * width:
            self._decimator.max_points = max(2, int(2 * width * k / total))
            picks = self._decimator.index(
                0, k, lambda i: np.column_stack([link[:k][i], self._motor[:k][i], self._rpm[:k][i]]), 3
            )
        else:
            picks = [np.arange(k)] * 3
        for curve, y, rows in zip(
            (self.link_curve, self.motor_curve, self.speed_curve), (link, self._motor, self._rpm), picks
        ):
            curve.setData(t[rows], y[:k][rows])
        for i, c in enumerate(self._crossings):
            if i not in self._passed and k and self._rpm[k - 1] >= c.rpm:
                self._passed.add(i)
                at = self._t[int(np.argmax(self._rpm[:k] >= c.rpm))]
                self._add_mark(self.startup_torque, at, 90, f"Mode {c.mode + 1}")
        self._show_peaks()

    def _show_status(self) -> None:
        sim = self.sim
        if sim is None:
            self.startup_status.setText("No motor on the train: pick one under Start-up to run it up.")
            return
        if self._k == 0:
            self.startup_status.setText("At rest: Start runs the train up.")
            return
        m = sim.startup.motor
        rpm = self._rpm[self._k - 1]
        slip = sim.slip
        text = f"t = {sim.t:.3f} s: the motor at {rpm:.0f} rpm, slip {100 * slip:.1f} %"
        if m.kind is MotorKind.SYNCHRONOUS:
            if sim.field_on and abs(slip) < 1e-3:
                text += ", in step"
            else:
                text += f", its poles pulsing at {m.pulsation_hz(abs(slip)):.3g} Hz"
            text += ", the field on" if sim.field_on else ""
        self.startup_status.setText(text + (". Done." if sim.done else "."))

    def _show_peaks(self) -> None:
        sim = self.sim
        if sim is None or self._k == 0:
            self.startup_note.setText("")
            return
        rows = []
        ratios = np.where(self._running > 0, sim.peaks / np.where(self._running > 0, self._running, 1.0), 0.0)
        worst = int(np.argmax(ratios)) if np.isfinite(ratios).all() else -1
        for i, link in enumerate(self.train.links):
            at = int(np.searchsorted(self._t[: self._k], sim.peak_t[i]))
            rpm = self._rpm[min(at, self._k - 1)]
            text = f"{link.name}: {fmt_torque(sim.peaks[i])} at {sim.peak_t[i]:.3f} s ({rpm:.0f} rpm)"
            if math.isfinite(self._running[i]) and self._running[i] > 0:
                text += f", {ratios[i]:.1f}× its running {fmt_torque(self._running[i])}"
            rows.append(f"<b>{text}</b>" if i == worst else text)
        head = "Peak torques" if sim.done else "Peak torques so far"
        self.startup_note.setText(f"{head}: " + "; ".join(rows) + ".")

    def _show_info(self) -> None:
        if not self.motor_on:
            self.startup_info.setText(self.note or "No motor: pick one to start the train from rest.")
            return
        t, su = self.train, self.startup
        m = su.motor
        w = steady_speed(t, su)
        parts = [self.note] if self.note else []
        if math.isnan(w):
            parts.append(
                f"Synchronous speed {m.sync_rpm:.0f} rpm: the motor can't carry the load there, and "
                f"stalls on the way up."
            )
        elif m.kind is MotorKind.INDUCTION:
            parts.append(
                f"Synchronous speed {m.sync_rpm:.0f} rpm; it runs at {60 * w / (2 * math.pi):.0f} rpm "
                f"(slip {100 * (1 - w / m.sync_speed):.1f} %)."
            )
            near = min(self._modes, key=lambda md: abs(md.freq_hz - m.line_hz), default=None)
            if near is not None:
                j = self._modes.index(near)
                parts.append(
                    f"The switch-on rings at {m.line_hz:g} Hz; mode {j + 1} ({near.freq_hz:.3g} Hz) is "
                    f"{100 * abs(near.freq_hz - m.line_hz) / m.line_hz:.0f} % from it."
                )
        else:
            parts.append(f"It runs at {m.sync_rpm:.0f} rpm.")
            cs = crossings(t, m)
            if cs:
                parts.append(
                    "The poles' 2× slip pulsation meets "
                    + ", ".join(f"mode {c.mode + 1} ({c.freq_hz:.3g} Hz) at {c.rpm:.0f} rpm" for c in cs)
                    + "."
                )
            else:
                parts.append(f"No mode below {2 * m.line_hz:g} Hz for the poles' pulsation to meet.")
        self.startup_info.setText(" ".join(parts))

    def apply_theme(self) -> None:
        self.link_curve.setPen(pg.mkPen(colors.strong, width=1))
        self.motor_curve.setPen(pg.mkPen(colors.force, width=1))
        self.speed_curve.setPen(pg.mkPen(colors.mass[0], width=2))
        self.running_line.setPen(pg.mkPen(colors.muted, width=1, style=QtCore.Qt.PenStyle.DashLine))
        for _, mark in self._marks:
            self._style_mark(mark)
