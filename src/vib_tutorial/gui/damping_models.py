"""Damping models page (#478's pilot, its second page): one live mass with a damping model to pick, as it runs.

The mass runs live (`view_models/live_mass.LiveMass`) with the damping picked (viscous, hysteretic by its stand-in
in time, Coulomb friction, the Zener element), each matched at f<sub>n</sub> to ζ (`view_models/damping.py`), and
the viscous ghost beside it, dashed, plucked and driven as it is. Left: the oscillator, the damping (pick it, or
change ζ, as it runs), the drive, then the Things to try. Centre: the mass with its forces (`live_body`: pluck it,
or drag the grips on the mass, the spring and the damper, which sets ζ), its run, and the time plots with the
ghost's trace and the changes marked. Right: the hysteresis loop the run draws, beside each model's, and the loss
factor against frequency; the Frequency response built by dwelling or sweeping, with the driven state's force
phasors; the Energy balance; the Theory.

Its trials, problems, tips and notes are `damping_trials.py`'s (no Qt); the controls are bound to its state
(`DampingTrial`) by `bindings.Bindings`.

Working on it: `tests/test_damping_page.py` (the page live: a model changed mid-run, the ghost, the loop's
energy, friction sticking, the sweep, Defaults), `test_live_parts.py` (the models' physics).
"""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.damping_models import MODELS, Model
from ..core.stepping import FRAME_BUDGET, Budget
from ..view_models import damping as view
from ..view_models import recorder as rec
from ..view_models.damping import DampingTrial
from ..view_models.free_body import ArrowScale, one_mass
from ..view_models.grips import Grip
from ..view_models.live_mass import COL, LiveMass
from ..view_models.text import fmt
from ..view_models.text import number as sig
from . import coalesce
from .bindings import Bindings
from .damping_trials import NOTES_HTML, PROBLEMS, TIPS, TRIALS
from .frame_loop import FrameLoop
from .layouts import FlowLayout
from .lesson import Form, TrialBox, combo, note_label, plot, readout_label, set_quietly, three_panes
from .live_body import LiveMassView
from .live_parts import ChangeMarks, EnergyTab, run_row
from .notes_view import NotesBrowser
from .plots import decimate
from .scrub import Scrub
from .style import colors, role
from .theming import add_legend

__all__ = ["NOTES_HTML", "PROBLEMS", "TRIALS", "DampingModelsPage"]

STRIP_SECONDS = 10.0
PEAK_SECONDS = 3.0
MEASURE_EVERY = 0.2
PLAYED_AFTER = 2.0
GRIPS = {
    "mass": Grip("mass", 0.01, 1e4, per=0.4),
    "stiffness": Grip("stiffness", 0.01, 1e8, per=0.3),
    "damping": Grip("zeta", 0.001, 0.5, per=0.3),  # the damper's handle sets the ζ every model is matched to
}
LOOP_TIP = (
    "<p>The loop the run draws: the damping force (f − mẍ − kx) against the displacement over its last cycle, the "
    "mass's colour; and faint, each model's steady loop at that frequency and amplitude. The area inside a loop is "
    "the energy it takes out a cycle.</p>"
)
LOSS_TIP = (
    "<p>Each model's loss factor η = K″/K′ against the frequency over f<sub>n</sub>; they meet at f<sub>n</sub>, "
    "where they're matched (2ζ). The line is the drive's frequency.</p>"
)
FRF_TIP = (
    "<p>The frequency response, built by hand: each point a steady swing measured at the drive's frequency, "
    "|X|k/F. <i>Show the theory</i> draws the model's curve (friction's by its describing function) and the "
    "viscous ghost's, dashed.</p>"
)
PHASOR_TIP = (
    "<p>The driven steady state's forces as phasors, turning at the drive's frequency: the spring's kX, the "
    "damping's (a quarter turn ahead of the motion) and the mass's −Ω²mX, head to tail, add up to the drive F. "
    "At resonance the spring's and the mass's cancel, and the damping's alone meets the drive.</p>"
)


class DampingModelsPage(QtWidgets.QWidget):
    """The oscillator and its damping on the left, the live mass and its time plots in the centre, the tabs right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def transport(self):
        from .transport import Transport

        return Transport(self.run_button, self.reset_button, self.speed, self.clock)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        start = DampingTrial("", "")
        self.live = LiveMass(view.params(start))
        self.ghost = LiveMass(view.ghost_params(start))
        self.recorder = rec.Recorder()
        self.sweep: rec.Sweep | None = None
        self.running = False
        self.arrows = ArrowScale(room=1.0)
        self._handled_at: float | None = None
        self._grip_start = 0.0
        self._measured_at = -1.0
        self._sweeping_now = False
        self._budget = Budget()
        self.bindings = Bindings(DampingTrial, self)

        # --- left
        osc = Form("Oscillator")
        self.mass = osc.number("Mass m:", 0.01, 1e4, 1.0, 3, " kg", "<p>The mass m. Drag the handle on its corner.</p>")
        self.stiffness = osc.number(
            "Stiffness k:", 0.01, 1e8, 400.0, 2, " N/m", "<p>The spring's stiffness k. Drag the handle on it.</p>"
        )
        damp = Form("Damping")
        self.model = combo([m.value for m in MODELS], TIPS["model"])
        damp.add("Model:", self.model, TIPS["model"])
        self.zeta = damp.number("Matched to ζ:", 0.001, 0.5, 0.05, 4, "", TIPS["zeta"])
        self.amplitude = damp.number("Friction at X:", 0.01, 1000.0, 10.0, 2, " mm", TIPS["amplitude"])
        self.relax = damp.number("Zener peak at:", 0.01, 100.0, 1.0, 3, " × fₙ", TIPS["relax"])
        self.ghost_on = QtWidgets.QCheckBox("Viscous ghost beside it")
        self.ghost_on.setToolTip(TIPS["ghost"])
        damp.row(self.ghost_on)
        self.info = note_label()
        damp.row(self.info)
        drive = Form("Drive")
        self.drive_on = QtWidgets.QCheckBox("Drive the mass")
        self.drive_on.setToolTip(TIPS["drive"])
        drive.row(self.drive_on)
        self.force = drive.number("Force F:", 0.0, 1e6, 1.0, 4, " N", TIPS["force"])
        self.freq = drive.number("Frequency:", 0.01, 1e4, 3.0, 3, " Hz", TIPS["freq"])
        self.tune = QtWidgets.QPushButton("Tune to fₙ")
        self.tune.clicked.connect(self.tune_to_natural)
        drive.row(self.tune)
        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Setups to come back to, each showing one idea from the Theory tab. Once you've handled the "
            "oscillator, some ask you to predict what you'll see first.</p>",
            predictions=[t.predict for t in TRIALS],
            ask_after_play=True,
        )
        self.trials.picked.connect(self.load_trial)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        for w in (osc, damp, drive, self.trials):
            lv.addWidget(w)
        lv.addStretch(1)

        # --- centre
        self.body = LiveMassView()
        self.body.setMinimumHeight(200)
        self.body.grabbed.connect(self._on_grabbed)
        self.body.pulled.connect(self._on_pulled)
        self.body.let_go.connect(self._on_let_go)
        self.body.grip_taken.connect(self._on_grip_taken)
        self.body.grip_moved.connect(self._on_grip_moved)
        row, self.speed, self.run_button, self.reset_button, self.clock = run_row(self)
        self.run_button.toggled.connect(self._on_run)
        self.reset_button.setToolTip("<p>Back to rest at t = 0; the measured points are kept.</p>")
        self.reset_button.clicked.connect(self.reset)
        self.strip = plot("Displacement x", "m", "Time", "s", height=150)
        self.strip.setMouseEnabled(x=False, y=False)
        self.strip.setToolTip(
            "<p>The mass's displacement, the last 10 s, and the viscous ghost's, dashed. A dashed line marks a "
            "change.</p>"
        )
        add_legend(self.strip.getPlotItem(), offset=(-10, 10))
        self.ghost_item = pg.PlotDataItem(name="viscous ghost")
        self.x_item = pg.PlotDataItem(name="this model")
        self.strip.addItem(self.ghost_item)
        self.strip.addItem(self.x_item)
        self.force_plot = plot("Drive f", "N", "Time", "s", height=80)
        self.force_plot.setXLink(self.strip)
        self.force_plot.setMouseEnabled(x=False, y=False)
        self.f_item = pg.PlotDataItem()
        self.force_plot.addItem(self.f_item)
        decimate(self.strip, self.force_plot)
        self.marks = ChangeMarks(self.strip, self.force_plot)
        self.scrub = Scrub(self.strip, self.force_plot)
        self.scrub.scrubbed.connect(self._on_scrubbed)
        self.readout = readout_label()
        centre = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(centre)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.body, 3)
        cv.addLayout(row)
        cv.addWidget(self.strip, 2)
        cv.addWidget(self.force_plot, 1)
        cv.addWidget(self.readout)

        # --- right: Loop & loss, Frequency response, Energy, Theory
        loop = QtWidgets.QWidget()
        lpv = QtWidgets.QVBoxLayout(loop)
        self.damping_only = QtWidgets.QCheckBox("Damping force only")
        self.damping_only.setToolTip(TIPS["damping_only"])
        lpv.addWidget(self.damping_only)
        self.loop_plot = plot("Force", "N", "Displacement x", "m", height=170)
        self.loop_plot.setMouseEnabled(x=False, y=False)
        self.loop_plot.setToolTip(LOOP_TIP)
        self.theory_loops = {m: pg.PlotDataItem() for m in MODELS}
        self.spring_line = pg.PlotDataItem()
        self.loop_item = pg.PlotDataItem()
        for item in (*self.theory_loops.values(), self.spring_line, self.loop_item):
            self.loop_plot.addItem(item)
        self.loop_note = note_label()
        self.loss_plot = plot("Loss factor η", None, "Frequency / fₙ", None, log_x=True, log_y=True, height=120)
        self.loss_plot.setMouseEnabled(x=False, y=False)
        self.loss_plot.setToolTip(LOSS_TIP)
        add_legend(self.loss_plot.getPlotItem(), offset=(-10, 10))
        self.loss_items = {m: pg.PlotDataItem(name=view.SHORT[m]) for m in MODELS}
        for item in self.loss_items.values():
            self.loss_plot.addItem(item)
        self.loss_now = pg.InfiniteLine(angle=90)
        self.loss_plot.addItem(self.loss_now, ignoreBounds=True)
        lpv.addWidget(self.loop_plot, 3)
        lpv.addWidget(self.loop_note)
        lpv.addWidget(self.loss_plot, 2)

        frf = QtWidgets.QWidget()
        fv = QtWidgets.QVBoxLayout(frf)
        self.sweep_button = QtWidgets.QPushButton("Sweep")
        self.sweep_button.setCheckable(True)
        self.sweep_button.setToolTip(
            "<p>Step the drive through frequencies across the resonance, dwelling at each until its point lands.</p>"
        )
        self.sweep_button.toggled.connect(self._on_sweep)
        self.clear_button = QtWidgets.QPushButton("Clear points")
        self.clear_button.clicked.connect(self.clear_points)
        self.theory = QtWidgets.QCheckBox("Show the theory")
        self.theory.toggled.connect(self._draw_frf)
        buttons = FlowLayout()
        buttons.add(self.sweep_button, self.clear_button, self.theory)
        fv.addLayout(buttons)
        self.mag_plot = plot("Amplification |X|k/F", None, "Frequency", "Hz", log_x=True, log_y=True, height=150)
        self.mag_plot.setMouseEnabled(x=False, y=False)
        self.mag_plot.setToolTip(FRF_TIP)
        self.frf_theory = pg.PlotDataItem()
        self.frf_ghost = pg.PlotDataItem()
        self.frf_earlier = pg.PlotDataItem(pen=None, symbol="o", symbolSize=6)
        self.frf_points = pg.PlotDataItem(pen=None, symbol="o", symbolSize=8)
        self.fn_line = pg.InfiniteLine(angle=90)
        for item in (self.frf_ghost, self.frf_theory, self.fn_line, self.frf_earlier, self.frf_points):
            self.mag_plot.addItem(item)
        self.frf_note = note_label()
        self.phasor_plot = plot("Imaginary", "N", "Real", "N", height=150)
        self.phasor_plot.setAspectLocked(True)
        self.phasor_plot.setMouseEnabled(x=False, y=False)
        self.phasor_plot.setToolTip(PHASOR_TIP)
        add_legend(self.phasor_plot.getPlotItem(), offset=(-10, 10))
        self.phasor_items = {
            name: pg.PlotDataItem(name=label)
            for name, label in (
                ("spring", "spring kX"),
                ("damping", "damping"),
                ("mass", "mass −Ω²mX"),
                ("drive", "drive F"),
            )
        }
        for item in self.phasor_items.values():
            self.phasor_plot.addItem(item)
        fv.addWidget(self.mag_plot, 3)
        fv.addWidget(self.frf_note)
        fv.addWidget(self.phasor_plot, 2)

        self.energy = EnergyTab()
        self.notes = NotesBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(loop, "Loop && loss")
        self.tabs.addTab(frf, "Frequency response")
        self.tabs.addTab(self.energy, "Energy")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._draw_tab)

        three_panes(self, left, centre, self.tabs, sizes=(320, 760, 560))

        self.bindings.bind("model", self.model, choices=MODELS)
        for field, widget in (
            ("mass", self.mass),
            ("stiffness", self.stiffness),
            ("zeta", self.zeta),
            ("amplitude_mm", self.amplitude),
            ("relax", self.relax),
            ("ghost", self.ghost_on),
            ("drive", self.drive_on),
            ("force", self.force),
            ("freq_hz", self.freq),
            ("damping_only", self.damping_only),
            ("tab", self.tabs),
        ):
            self.bindings.bind(field, widget)
        self.bindings.write(start)
        self.bindings.changed.connect(self._on_field)

        self.loop = FrameLoop(self, "Damping models", self._tick, self._runs, self._sim_t)
        self.apply_theme()
        self._push_params()

    # ------------------------------------------------------------- the state
    def practice_state(self) -> DampingTrial:
        """The page's settings, as a trial: what a practice spec checks, and Defaults puts back."""
        return self.bindings.read(DampingTrial("", ""))

    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        self.apply_trial(DampingTrial("", "", **setup))

    def apply_trial(self, t: DampingTrial) -> None:
        self.bindings.write(t)
        if t.tune:
            set_quietly(self.freq, view.drive_freq(t))
        self.recorder.clear()
        self._stop_sweep()
        self._push_params()
        self.reset()
        self._handled_at = None
        self.trials.show_note(t.note)
        if t.x0_mm:
            x0 = 1e-3 * t.x0_mm
            self.body.start_pluck(x0)
            for live in (self.live, self.ghost):
                live.hold(x0)
                live.let_go()
        if t.sweep:
            self.sweep_button.setChecked(True)
        if t.x0_mm or t.drive or t.sweep:
            self._run(True)
        self._draw_all()

    def _state(self) -> DampingTrial:
        return self.practice_state()

    def _key(self) -> tuple:
        """The system the recorder's points belong to: the model and what it's matched with (not the hysteretic
        stand-in's c, which follows the drive's frequency)."""
        t = self._state()
        return (t.model, t.mass, t.stiffness, t.zeta, t.amplitude_mm, t.relax)

    # ------------------------------------------------------------- what the reader changes
    def _on_field(self, field: str) -> None:
        if field in ("damping_only", "tab"):
            self._draw_tab()
            return
        if field == "ghost":
            self._draw_live()
            return
        self._handled()
        if field == "freq_hz" and self.sweep is not None and not self._sweeping_now:
            self.sweep_button.setChecked(False)
        self._push_params()

    def _handled(self) -> None:
        if self._handled_at is None:
            self._handled_at = self.live.t

    def _push_params(self) -> None:
        """The live mass (and the ghost) to the controls: a live change, and the drive."""
        t = self._state()
        self.amplitude.setVisible(t.model is Model.COULOMB)
        self.damping_form_label(self.amplitude).setVisible(t.model is Model.COULOMB)
        self.relax.setVisible(t.model is Model.ZENER)
        self.damping_form_label(self.relax).setVisible(t.model is Model.ZENER)
        d = view.drive(t)
        for live, params in ((self.live, view.params(t)), (self.ghost, view.ghost_params(t))):
            live.set_drive(d)
            live.set_params(params)
        p = self.live.params
        self.body.set_values(
            t.mass, t.stiffness, p.damping, reference=1.0, damper=f"{view.SHORT[t.model]}, ζ = {sig(t.zeta)}"
        )
        self.info.setText(view.info(t))
        self._draw_frf()
        if not self.running:
            self._draw_tab()  # running, the next frame draws it

    def damping_form_label(self, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        form = widget.parentWidget()
        assert isinstance(form, Form), "a damping box sits in its form"
        return form.label(widget)

    def tune_to_natural(self) -> None:
        self.freq.setValue(round(view.matched(self._state()).natural_hz, 6))

    def _on_grabbed(self) -> None:
        self.scrub.hide()
        self._handled()
        self.body.start_pluck(self.live.x)
        for live in (self.live, self.ghost):
            live.hold(self.live.x)

    def _on_pulled(self, x: float) -> None:
        for live in (self.live, self.ghost):
            live.hold(x)
        self._draw_body()

    def _on_let_go(self) -> None:
        for live in (self.live, self.ghost):
            live.let_go()
        if not self.running:
            self._run(True)

    def _grip_box(self, name: str) -> QtWidgets.QDoubleSpinBox:
        return {"mass": self.mass, "stiffness": self.stiffness, "damping": self.zeta}[name]

    def _on_grip_taken(self, name: str) -> None:
        self._grip_start = self._grip_box(name).value()

    def _on_grip_moved(self, name: str, distance: float) -> None:
        set_quietly(self._grip_box(name), GRIPS[name].dragged(self._grip_start, distance))
        self._handled()
        self._push_params()

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
        for live in (self.live, self.ghost):
            live.reset()
        self.arrows.reset()
        self._measured_at = -1.0
        self.scrub.hide()
        self.running = False
        self.run_button.rest()
        if self.sweep is not None:
            self.sweep = rec.Sweep(self.sweep.freqs)
            self._set_sweep_freq()
        self._draw_live()

    def _advance(self, duration: float) -> None:
        self.live.advance(duration)
        if self.ghost_on.isChecked():
            self.ghost.advance(self.live.t - self.ghost.t)

    def _tick(self, dt: float) -> None:
        duration = dt * float(self.speed.currentData())
        duration = min(duration, self._budget.allows(duration, FRAME_BUDGET))
        start = time.perf_counter()
        before = self.live.t
        self._advance(duration)
        if self.live.t > before:
            self._budget.spent(time.perf_counter() - start, self.live.t - before)
        self._measure()
        self._check_played()
        self._draw_live()

    def step(self, duration: float) -> None:
        """Run on by `duration` s of simulated time at once (tests), measuring and drawing as the frames would."""
        end = self.live.t + duration
        while self.live.t < end - 1e-9 and self.live.held is None:
            self._advance(min(0.05, end - self.live.t) + 1e-12)
            self._measure()
        self._check_played()
        self._draw_live()

    def _check_played(self) -> None:
        if self._handled_at is not None and self.live.t - self._handled_at >= PLAYED_AFTER:
            self.trials.set_played()

    def _measure(self) -> None:
        live = self.live
        if not live.drive.on or live.held is not None or live.t - self._measured_at < MEASURE_EVERY:
            return
        self._measured_at = live.t
        t, y = live.record.window(2.5 * rec.PERIODS / live.drive.freq_hz)
        m = rec.measure(t, y[:, COL["x"]], y[:, COL["f"]], live.drive.freq_hz, since=live.changed_at)
        if m is None or self.recorder.land(m, self._key()) is None:
            return
        self._draw_frf()
        if self.sweep is not None:
            if self.sweep.landed() is None:
                self._stop_sweep()
            else:
                self._set_sweep_freq()

    def _on_sweep(self, on: bool) -> None:
        if not on:
            self.sweep = None
            return
        m = view.matched(self._state())
        self.sweep = rec.Sweep(rec.sweep_frequencies(m.natural_hz, m.zeta))
        if not self.drive_on.isChecked():
            set_quietly(self.drive_on, True)
        self._set_sweep_freq()
        if not self.running:
            self._run(True)

    def _set_sweep_freq(self) -> None:
        assert self.sweep is not None, "a sweep sets the drive's frequency"
        f = self.sweep.frequency
        if f is None:
            return
        self._sweeping_now = True
        set_quietly(self.freq, round(f, 3))
        self._sweeping_now = False
        self._push_params()  # the drive, and the hysteretic stand-in's c with it

    def _stop_sweep(self) -> None:
        self.sweep = None
        set_quietly(self.sweep_button, False)

    def clear_points(self) -> None:
        self.recorder.clear()
        self._draw_frf()

    def _show_scrub(self) -> None:
        r = self.live.record
        if r.size > 1:
            end = float(r.t[r.size - 1])
            self.scrub.show(end, max(float(r.t[0]), end - STRIP_SECONDS))

    def _on_scrubbed(self, t: float) -> None:
        r = self.live.record
        ts = r.t[: r.size]
        x, v, f = (float(np.interp(t, ts, r.y[: r.size, COL[c]])) for c in ("x", "v", "f"))
        p = self.live.params
        self.body.show(x, one_mass(p.stiffness, p.damping, x, v, f, scale=self.arrows, t=t))
        self.clock.setText(f"t = {sig(t, 4)} s (paused at {sig(self.live.t, 4)} s)")

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.body.apply_theme()
        self.scrub.apply_theme()
        self.marks.apply_theme()
        self.energy.apply_theme()
        self.x_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.ghost_item.setPen(pg.mkPen(c.grey, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.f_item.setPen(pg.mkPen(c.force, width=1.5))
        for m, item in self.theory_loops.items():
            item.setPen(pg.mkPen(role(view.ROLES[m]), width=1))
        for m, item in self.loss_items.items():
            item.setPen(pg.mkPen(role(view.ROLES[m]), width=2))
        self.loss_now.setPen(pg.mkPen(c.force, width=1))
        self.spring_line.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.loop_item.setPen(pg.mkPen(c.mass[0], width=2.5))
        self.frf_theory.setPen(pg.mkPen(c.strong, width=1.5))
        self.frf_ghost.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.frf_earlier.setSymbolPen(pg.mkPen(c.grey, width=1))
        self.frf_earlier.setSymbolBrush(None)
        self.frf_points.setSymbolPen(None)
        self.frf_points.setSymbolBrush(pg.mkBrush(c.mass[0]))
        self.fn_line.setPen(pg.mkPen(c.muted, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self._draw_all()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._draw_tab()

    def _draw_all(self) -> None:
        self._draw_frf()
        self._draw_tab()  # and what moves

    @coalesce.coalesced(covers=("_draw_live",))
    def _draw_tab(self, *_: object) -> None:
        """The tab showing: the loss curves (they change only with the settings), then what moves on it."""
        t = self._state()
        if self.tabs.currentIndex() == view.TAB_LOOP:
            lv = view.loss_view(t)
            for m, curve in lv.loss:
                self.loss_items[m].setData(*curve)
            self.loss_now.setVisible(lv.now is not None)
            if lv.now is not None:
                self.loss_now.setValue(np.log10(lv.now))
        self._draw_live()

    @coalesce.coalesced
    def _draw_live(self) -> None:
        live = self.live
        r = live.record
        t, y = r.window(STRIP_SECONDS)
        end = live.t
        self.x_item.setData(t, y[:, COL["x"]])
        ghost = self.ghost_on.isChecked()
        self.ghost_item.setVisible(ghost)
        if ghost:
            tg, yg = self.ghost.record.window(STRIP_SECONDS)
            self.ghost_item.setData(tg, yg[:, COL["x"]])
        self.f_item.setData(t, y[:, COL["f"]])
        self.strip.setXRange(max(0.0, end - STRIP_SECONDS), max(end, STRIP_SECONDS), padding=0)
        self.marks.draw(live.edits, end - STRIP_SECONDS)
        recent = y[t >= end - PEAK_SECONDS, COL["x"]] if t.size else np.empty(0)
        peak = float(np.abs(recent).max()) if recent.size else 0.0
        if live.held is not None:
            peak = max(peak, abs(live.held))
        self.body.fit(peak)
        self._draw_body()
        self.clock.setText(f"t = {sig(end, 4)} s")
        self._show_readout()
        i = self.tabs.currentIndex()
        if i == view.TAB_LOOP:
            self._draw_loop()
        elif i == view.TAB_ENERGY:
            self.energy.draw(live)
        elif i == view.TAB_FRF:
            self._draw_phasors()

    def _draw_body(self) -> None:
        live = self.live
        p = live.params
        hand = p.stiffness * live.held if live.held is not None else None
        friction = live.friction_force if p.friction > 0 else None
        arm = live.arm_force if p.has_arm else None
        body = one_mass(
            p.stiffness, p.damping, live.x, live.v, live.force_now, friction, hand, arm, scale=self.arrows, t=live.t
        )
        self.body.show(live.x, body, self.ghost.x if self.ghost_on.isChecked() else None)

    def _show_readout(self) -> None:
        live = self.live
        if live.held is not None:
            self.readout.setText(f"Held at x = {fmt(live.held, 'm', largest='')}: let go to pluck it.")
            return
        if live.drive.on:
            f = live.drive.freq_hz
            points = [p for p in self.recorder.current(self._key()) if abs(p.freq_hz - f) <= 1e-6 * f]
            if points:
                h = points[-1].response
                self.readout.setText(
                    f"Steady at {sig(f)} Hz: x swings ±{fmt(abs(h) * live.drive.force, 'm', largest='')}, lagging "
                    f"the drive by {sig((-np.degrees(np.angle(h))) % 360)}°."
                )
            else:
                self.readout.setText(f"Driven at {sig(f)} Hz: waiting for the swing to settle…")
            return
        t, y = live.since_change()
        still = t.size > 0 and live.v == 0.0 and live.params.friction > 0
        text = view.decay_readout(self._state(), t, y[:, COL["x"]], still) if t.size else ""
        self.readout.setText(text or ("Pluck the mass: drag it sideways and let go." if t.size < 2 else ""))

    def _period(self) -> float | None:
        live = self.live
        if live.drive.on:
            return 1.0 / live.drive.freq_hz
        if live.record.size < 2:
            return None
        return 1.0 / view.matched(self._state()).natural_hz

    def _draw_loop(self) -> None:
        live = self.live
        period = self._period()
        t, y = live.record.window(1.1 * period if period else 1.0)
        lv = view.loop_view(self._state(), t, y[:, COL["x"]], y[:, COL["v"]], y[:, COL["f"]], period)
        self.loop_item.setData(*lv.live)
        drawn = dict(lv.theory)
        for m, item in self.theory_loops.items():
            curve = drawn.get(m)
            item.setData(*(curve if curve is not None else ([], [])))
        self.spring_line.setData(*lv.spring)
        self.loop_note.setText(lv.note)

    def _draw_phasors(self) -> None:
        drawn = {ph.name: ph for ph in view.phasors(self._state(), self.live.t)}
        for name, item in self.phasor_items.items():
            ph = drawn.get(name)
            if ph is None:
                item.setData([], [])
                continue
            item.setData([ph.start[0], ph.end[0]], [ph.start[1], ph.end[1]])
            item.setPen(pg.mkPen(role(ph.role), width=3))

    @coalesce.coalesced
    def _draw_frf(self, *_: object) -> None:
        t = self._state()
        key = self._key()
        fv = view.frf_view(t, self.recorder.current(key), self.recorder.earlier(key), self.theory.isChecked())
        self.frf_points.setData(*fv.points)
        self.frf_earlier.setData(*fv.earlier)
        self.frf_theory.setData(*fv.theory)
        self.frf_ghost.setData(*fv.ghost)
        self.fn_line.setValue(np.log10(fv.natural_hz))
        self.frf_note.setText(fv.note)
        self.mag_plot.setXRange(np.log10(fv.natural_hz / 4), np.log10(fv.natural_hz * 4), padding=0.02)
