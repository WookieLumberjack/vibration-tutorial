"""Forcing an oscillator page (#478 Phase 4): Transient response and Periodic forcing as one page, on the live mass.

The mass is stepped as it runs (`view_models/live_forcing.LiveForcing`), and the reader forces it: *Apply* fires a
one-off force (a hit, a push held until *Let go*, a ramp, a rectangular or half-sine pulse) from that moment, on
whatever the mass is doing, so a second lands on the first's motion (superposition by hand); *Wave* switches a
periodic force on (square, sawtooth, triangle, rectified sine, pulse trains, hammer blows). Left: the oscillator
(m, k, c), the one-off force, the wave, then the Things to try (setups to go back to; their predictions asked once
the reader has handled it). Center: the mass on its spring and damper (the Single oscillator's `live_body`: pluck
it, or drag its grips to change m, k and c as it runs) with the forces on it beside it (`free_body_view`), the
run's controls and the displacement and force against time. Right: the Convolution (Duhamel's integral at an
instant, from the run's record: paused, at the scrubbed time), the Pulse spectrum (built by hand: each pulse
applied from rest lands its peak), the Harmonics (the wave's and the response's lines, from the steady run) and
the Theory.

What the page shows is `view_models/forcing.py`'s and `live_forcing.py`'s; its trials, problems, tips and notes
are `forcing_trials.py`'s (no Qt). The controls are bound to the state (`ForcingTrial`) by `bindings.Bindings`, so
Defaults round-trips by construction.

Working on it: `tests/test_forcing_page.py` (the page live: a hit and a second, a push let go, a pulse landing its
peak, a wave's harmonics, the convolution, the grips, predictions after play, Defaults), `test_live_forcing.py`
(the Qt-free run), `test_view_models.py` (the panels), `test_readiness.py`.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.periodic import Wave
from ..core.transient import PulseKind
from ..view_models import forcing as view
from ..view_models import live_forcing as lf
from ..view_models.forcing import KINDS, SPECTRUM_KINDS, WAVES, ForcingTrial
from ..view_models.free_body import ArrowScale, one_mass
from ..view_models.grips import Grip
from ..view_models.live_forcing import COL, LiveForcing, PulsePeak, Push
from ..view_models.text import number as sig
from . import coalesce
from .bindings import Bindings
from .forcing_trials import CONV_TIP, F_TIP, HARM_TIP, NOTES_HTML, PROBLEMS, PULSE_TIP, TIPS, TRIALS, X_TIP
from .frame_loop import FrameLoop
from .free_body_view import FreeBodyView
from .layouts import FlowLayout
from .lesson import ActionToggle, Form, LiveLabel, TrialBox, combo, note_label, plot, set_quietly, three_panes
from .live_body import LiveMassView
from .live_parts import ChangeMarks, run_row
from .notes_view import NotesBrowser
from .plots import decimate
from .scrub import Scrub
from .style import colors
from .theming import add_legend
from .transport import Transport

__all__ = ["NOTES_HTML", "PROBLEMS", "TRIALS", "ForcingPage"]

STRIP_SECONDS = 8.0  # the time plots' window
PEAK_SECONDS = 3.0  # the body's scale fits the largest swing this recent
MEASURE_EVERY = 0.25  # s of run between the harmonics' and the pulses' looks
TAB_EVERY = 0.1  # s of run between the Convolution tab's redraws
PLAYED_AFTER = 2.0  # s of run after the reader's first change: they've handled it, predictions on
SYSTEM_FIELDS = ("mass", "stiffness", "damping")
WAVE_FIELDS = ("wave", "wave_amplitude", "freq_hz", "duty", "tune")
GRIPS = {  # the Single oscillator's
    "mass": Grip("mass", 0.01, 1e4, per=0.4),
    "stiffness": Grip("stiffness", 0.01, 1e8, per=0.3),
    "damping": Grip("damping", 0.01, 1e6, per=0.3, zero_below=0.02),
}
MARKERS = {"RECTANGLE": "s", "HALF_SINE": "o", "RAMP": "t"}  # each pulse kind's points on the spectrum


class ForcingPage(QtWidgets.QWidget):
    """The oscillator and its forces on the left, the live mass and its time plots in the center, the tabs right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def transport(self) -> Transport:
        return Transport(self.run_button, self.reset_button, self.speed, self.clock)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.live = LiveForcing()
        self.peaks: list[PulsePeak] = []  # the pulses that landed on the spectrum
        self._pending: list[Push] = []  # pulses applied, waiting a period to land
        self.running = False
        self.arrows = ArrowScale(room=1.0)
        self._handled_at: float | None = None
        self._grip_start = 0.0
        self._measured_at = -1.0
        self._drawn_at = -1.0
        self._scrubbed: float | None = None  # the instant the convolution is drawn at, paused and scrubbed
        self._harmonics: lf.Harmonics | None = None
        self.bindings = Bindings(ForcingTrial, self)

        # --- left: the oscillator, the one-off force, the wave, the Things to try
        box = Form("Oscillator")
        self.mass = box.number("Mass m:", 0.01, 1e4, 1.0, 3, " kg", TIPS["mass"])
        self.stiffness = box.number("Stiffness k:", 0.01, 1e8, 400.0, 2, " N/m", TIPS["stiffness"])
        self.damping = box.number("Damping c:", 0.0, 1e6, 2.0, 3, " N·s/m", TIPS["damping"])
        self.info = note_label()
        box.row(self.info)

        force = Form("A force")
        self.kind = combo([k.value for k in KINDS], TIPS["kind"])
        force.add("Kind:", self.kind, TIPS["kind"])
        self.amplitude = force.number("Force F:", -1e6, 1e6, 1.0, 4, " N", TIPS["amplitude"])
        self.amplitude_label = force.label(self.amplitude)
        self.duration = force.number("Length T:", 0.001, 100.0, 0.1, 4, " s", TIPS["duration"])
        self.duration_label = force.label(self.duration)
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.setToolTip(TIPS["apply"])
        self.apply_button.setMinimumHeight(32)
        self.apply_button.clicked.connect(self.apply_force)
        self.let_go_button = QtWidgets.QPushButton("Let go")
        self.let_go_button.setToolTip(TIPS["let_go"])
        self.let_go_button.clicked.connect(self._on_let_go_force)
        buttons = FlowLayout()
        buttons.add(self.apply_button, self.let_go_button)
        force.row(buttons)

        wave = Form("A periodic force")
        self.wave = combo([w.value for w in WAVES], TIPS["wave"])
        wave.add("Wave:", self.wave, TIPS["wave"])
        self.wave_amplitude = wave.number("Height F:", 0.0, 1e6, 1.0, 4, " N", TIPS["wave_amplitude"])
        self.wave_amplitude_label = wave.label(self.wave_amplitude)
        self.freq = wave.number("Frequency f₀:", 0.01, 1e3, 1.0, 4, " Hz", TIPS["freq"])
        self.tune = QtWidgets.QSpinBox()
        self.tune.setRange(0, 9)
        self.tune.setSpecialValueText("off")
        wave.add("f₀ = fₙ /", self.tune, TIPS["tune"])
        self.duty = wave.number("Duty d:", 0.01, 0.99, 0.25, 2, "", TIPS["duty"])
        self.wave_on = ActionToggle("Switch the wave on", "Wave on: click to stop")
        self.wave_on.setToolTip(TIPS["wave_on"])
        self.wave_on.toggled.connect(self._on_wave_on)
        wave.row(self.wave_on)
        self.wave_box = wave

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
        for w in (box, force, wave, self.trials):
            lv.addWidget(w)
        lv.addStretch(1)

        # --- center: the mass, the run, the time plots
        self.body = LiveMassView()
        self.body.setMinimumHeight(180)
        self.body.grabbed.connect(self._on_grabbed)
        self.body.pulled.connect(self._on_pulled)
        self.body.let_go.connect(self._on_let_go)
        self.body.grip_taken.connect(self._on_grip_taken)
        self.body.grip_moved.connect(self._on_grip_moved)
        self.forces = FreeBodyView()
        row, self.speed, self.run_button, self.reset_button, self.clock = run_row(self)
        self.run_button.toggled.connect(self._on_run)
        self.reset_button.setToolTip("<p>Back to rest at t = 0, every force off; the spectrum's points are kept.</p>")
        self.reset_button.clicked.connect(self.reset)

        self.x_plot = plot("Displacement x", "m", "Time", "s", height=130)
        self.x_plot.setToolTip(X_TIP)
        self.x_plot.setMouseEnabled(x=False, y=False)
        self.x_item = pg.PlotDataItem()
        self.x_plot.addItem(self.x_item)
        self.f_plot = plot("Force f", "N", "Time", "s", height=90)
        self.f_plot.setToolTip(F_TIP)
        self.f_plot.setXLink(self.x_plot)
        self.f_plot.setMouseEnabled(x=False, y=False)
        self.f_item = pg.PlotDataItem()
        self.hit_item = pg.PlotDataItem(pen=None, symbol="t1", symbolSize=10)  # a hit: a mark, its force is an instant
        self.f_plot.addItem(self.f_item)
        self.f_plot.addItem(self.hit_item)
        decimate(self.x_plot, self.f_plot)
        self.marks = ChangeMarks(self.x_plot, self.f_plot)
        self.scrub = Scrub(self.x_plot, self.f_plot)
        self.scrub.scrubbed.connect(self._on_scrubbed)
        self.readout = LiveLabel(wrap=True)
        self.readout.setMinimumHeight(2 * self.readout.fontMetrics().lineSpacing())

        centre = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(centre)
        cv.setContentsMargins(0, 0, 0, 0)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.body, 3)
        top.addWidget(self.forces, 2)
        cv.addLayout(top, 3)
        cv.addLayout(row)
        cv.addWidget(self.x_plot, 2)
        cv.addWidget(self.f_plot, 1)
        cv.addWidget(self.readout)

        # --- right: Convolution, Pulse spectrum, Harmonics, Theory
        conv = QtWidgets.QWidget()
        convv = QtWidgets.QVBoxLayout(conv)
        self.conv_plot = plot("f(τ), and h(t − τ) scaled", "N", "τ", "s", height=150)
        self.prod_plot = plot("f(τ) h(t − τ)", "m/s", "τ", "s", height=120)
        self.prod_plot.setXLink(self.conv_plot)
        for p in (self.conv_plot, self.prod_plot):
            p.setToolTip(CONV_TIP)
            p.setMouseEnabled(x=False, y=False)
        add_legend(self.conv_plot.getPlotItem(), offset=(10, 10))
        self.conv_f = pg.PlotDataItem(name="the force f(τ)")
        self.conv_h = pg.PlotDataItem(name="h(t − τ), flipped and slid")
        self.conv_hits = pg.PlotDataItem(pen=None, symbol="t1", symbolSize=10)
        for item in (self.conv_f, self.conv_h, self.conv_hits):
            self.conv_plot.addItem(item)
        self.prod = pg.PlotDataItem(fillLevel=0.0)
        self.prod_plot.addItem(self.prod)
        self.conv_note = note_label()
        convv.addWidget(self.conv_plot, 1)
        convv.addWidget(self.prod_plot, 1)
        convv.addWidget(self.conv_note)

        spec = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(spec)
        self.theory = QtWidgets.QCheckBox("Show the theory")
        self.theory.setToolTip(TIPS["theory"])
        self.clear_button = QtWidgets.QPushButton("Clear points")
        self.clear_button.setToolTip("<p>Forget the landed pulses.</p>")
        self.clear_button.clicked.connect(self.clear_points)
        sbuttons = FlowLayout()
        sbuttons.add(self.clear_button, self.theory)
        sv.addLayout(sbuttons)
        self.pulse_plot = plot("Peak / static", None, "T / Tₙ", None, log_x=True, height=200)
        self.pulse_plot.setToolTip(PULSE_TIP)
        self.pulse_plot.setMouseEnabled(x=False, y=False)
        self.pulse_plot.enableAutoRange(x=False)
        self.pulse_plot.setXRange(np.log10(view.RATIOS[0]), np.log10(view.RATIOS[-1]), padding=0)
        add_legend(self.pulse_plot.getPlotItem(), offset=(-10, 10))
        self.pulse_points = {
            k.name: pg.PlotDataItem(pen=None, symbol=MARKERS[k.name], symbolSize=9, name=k.value)
            for k in SPECTRUM_KINDS
        }
        self.pulse_lines = {k.name: pg.PlotDataItem() for k in SPECTRUM_KINDS}
        for k in SPECTRUM_KINDS:
            self.pulse_plot.addItem(self.pulse_lines[k.name])
            self.pulse_plot.addItem(self.pulse_points[k.name])
        self.pulse_note = note_label()
        sv.addWidget(self.pulse_plot, 1)
        sv.addWidget(self.pulse_note)

        harm = QtWidgets.QWidget()
        hv = QtWidgets.QVBoxLayout(harm)
        self.harmonics = QtWidgets.QSpinBox()
        self.harmonics.setRange(1, 50)
        self.harmonics.setValue(5)
        self.harmonics.setToolTip(TIPS["harmonics"])
        self.harmonics.setAccessibleName("Harmonics shown")
        self.theory_too = QtWidgets.QCheckBox("Show the theory")
        self.theory_too.setToolTip(TIPS["theory"])
        self.theory_too.toggled.connect(self.theory.setChecked)
        hbuttons = FlowLayout()
        hbuttons.add(QtWidgets.QLabel("Harmonics:"), self.harmonics, self.theory_too)
        hv.addLayout(hbuttons)
        self.force_lines = plot("Force's harmonics", "N", "Frequency", "Hz", height=130)
        self.resp_lines = plot("Response's harmonics", "m", "Frequency", "Hz", height=150)
        self.resp_lines.setXLink(self.force_lines)
        for p in (self.force_lines, self.resp_lines):
            p.setToolTip(HARM_TIP)
            p.setMouseEnabled(x=False, y=False)
        self.force_stems = pg.PlotDataItem(connect="pairs")
        self.force_series = pg.PlotDataItem(pen=None, symbol="o", symbolSize=7)
        self.resp_stems = pg.PlotDataItem(connect="pairs")
        self.resp_series = pg.PlotDataItem(pen=None, symbol="o", symbolSize=7)
        self.receptance_item = pg.PlotDataItem()
        self.fn_lines = [pg.InfiniteLine(angle=90) for _ in range(2)]
        for p, items in (
            (self.force_lines, (self.force_stems, self.force_series, self.fn_lines[0])),
            (self.resp_lines, (self.receptance_item, self.resp_stems, self.resp_series, self.fn_lines[1])),
        ):
            for item in items:
                p.addItem(item)
        self.harm_note = note_label()
        hv.addWidget(self.force_lines, 1)
        hv.addWidget(self.resp_lines, 1)
        hv.addWidget(self.harm_note)

        self.notes = NotesBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(conv, "Convolution")
        self.tabs.addTab(spec, "Pulse spectrum")
        self.tabs.addTab(harm, "Harmonics")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._draw_tab)

        three_panes(self, left, centre, self.tabs, sizes=(320, 760, 560))

        self.bindings.bind("kind", self.kind, choices=KINDS)
        self.bindings.bind("wave", self.wave, choices=WAVES)
        for field, widget in (
            ("mass", self.mass),
            ("stiffness", self.stiffness),
            ("damping", self.damping),
            ("amplitude", self.amplitude),
            ("duration", self.duration),
            ("wave_amplitude", self.wave_amplitude),
            ("freq_hz", self.freq),
            ("duty", self.duty),
            ("harmonics", self.harmonics),
            ("tune", self.tune),
            ("tab", self.tabs),
            ("theory", self.theory),
        ):
            self.bindings.bind(field, widget)
        self.bindings.write(ForcingTrial("", ""))
        self.bindings.changed.connect(self._on_field)

        self.loop = FrameLoop(self, "Forcing an oscillator", self._tick, self._runs, self._sim_t)
        self.apply_theme()
        self._show_fields()
        self._push_system()
        self._draw_live()

    # ------------------------------------------------------------- the state
    def practice_state(self) -> ForcingTrial:
        """The page's settings, as a trial: what a practice spec checks, and Defaults puts back."""
        return self.bindings.read(ForcingTrial("", ""))

    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(ForcingTrial("", "", **setup))

    def apply_trial(self, t: ForcingTrial) -> None:
        """A setup: the oscillator and the forces as it says, the run from rest, then the force fired or the wave
        switched on (`start`)."""
        self.bindings.write(t)
        set_quietly(self.theory_too, t.theory)
        set_quietly(self.wave_on, False)
        self.peaks.clear()
        self._show_fields()
        self._push_system()
        self.reset()
        self._handled_at = None
        self.trials.show_note(t.note)
        if t.start == view.APPLY:
            self.apply_force()
        elif t.start == view.WAVE:
            self.wave_on.setChecked(True)
        self._handled_at = None  # the trial's own start isn't the reader's handling
        self._draw_tab()

    # ------------------------------------------------------------- what the reader changes
    def _on_field(self, field: str) -> None:
        if field in SYSTEM_FIELDS:
            self._handled()
            self._push_system()
        elif field in WAVE_FIELDS:
            self._show_fields()
            if field == "tune" or field == "wave":
                self._sync_freq()
            if self.wave_on.isChecked():
                self._handled()
                self.live.set_wave(view.forcing(self.practice_state()))
            self._draw_tab()
        elif field == "kind":
            self._show_fields()
        elif field == "theory":
            set_quietly(self.theory_too, self.theory.isChecked())
            self._draw_tab()
        elif field in ("harmonics", "tab"):
            self._draw_tab()

    def _handled(self) -> None:
        if self._handled_at is None:
            self._handled_at = self.live.t

    def _sync_freq(self) -> None:
        """Tuned, the frequency box shows f<sub>n</sub>/n (and can't be typed into)."""
        state = self.practice_state()
        self.freq.setEnabled(state.tune == 0)
        if state.tune:
            set_quietly(self.freq, view.wave_hz(state))

    def _show_fields(self) -> None:
        state = self.practice_state()
        hit = state.kind is PulseKind.IMPULSE
        self.amplitude_label.setText("Impulse I:" if hit else "Force F:")
        self.amplitude.setSuffix(" N·s" if hit else " N")
        shown = state.kind in (PulseKind.RAMP, PulseKind.RECTANGLE, PulseKind.HALF_SINE)
        self.duration.setVisible(shown)
        self.duration_label.setVisible(shown)
        self.duration_label.setText("Rise time T:" if state.kind is PulseKind.RAMP else "Length T:")
        blows = state.wave is Wave.IMPULSES
        self.wave_amplitude_label.setText("Each blow I:" if blows else "Height F:")
        self.wave_amplitude.setSuffix(" N·s" if blows else " N")
        pulses = state.wave is Wave.PULSES
        self.duty.setVisible(pulses)
        label = self.wave_box.label(self.duty)
        label.setVisible(pulses)
        self._sync_freq()

    def _push_system(self) -> None:
        state = self.practice_state()
        osc = view.oscillator(state)
        self.live.set_params(osc)
        self.body.set_values(state.mass, state.stiffness, state.damping)
        self.info.setText(view.info(osc))
        self._sync_freq()
        if self.wave_on.isChecked():
            self.live.set_wave(view.forcing(self.practice_state()))
        if not self.running:
            self._draw_live()
            self._draw_tab()

    def apply_force(self) -> None:
        """Fire the one-off force from now (see the module)."""
        self._handled()
        self.scrub.hide()
        push = self.live.apply(view.pulse(self.practice_state()))
        if push.pulse.kind is not PulseKind.IMPULSE:
            self._pending.append(push)
        if not self.running:
            self._run(True)
        self._draw_live()

    def _on_let_go_force(self) -> None:
        self._handled()
        self.live.release()
        self._draw_live()

    def _on_wave_on(self, on: bool) -> None:
        self._handled()
        self.scrub.hide()
        self.live.set_wave(view.forcing(self.practice_state()) if on else None)
        self._harmonics = None
        if on and not self.running:
            self._run(True)
        self._draw_tab()

    # ------------------------------------------------------------- the hand
    def _on_grabbed(self) -> None:
        self.scrub.hide()
        self._handled()
        self.body.start_pluck(self.live.x)
        self.live.hold(self.live.x)

    def _on_pulled(self, x: float) -> None:
        self.live.hold(x)
        self._draw_body()

    def _on_let_go(self) -> None:
        self.live.let_go()
        if not self.running:
            self._run(True)  # a pluck sets it going

    def _grip_box(self, name: str) -> QtWidgets.QDoubleSpinBox:
        return {"mass": self.mass, "stiffness": self.stiffness, "damping": self.damping}[name]

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
            self._scrubbed = None
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
        """Back to rest at t = 0, every force off, waiting; the spectrum's points kept."""
        self.live.reset()
        self.live.set_params(view.oscillator(self.practice_state()))
        self.live.since = 0.0
        self.live.edits.clear()
        set_quietly(self.wave_on, False)
        self._pending.clear()
        self._harmonics = None
        self._scrubbed = None
        self.arrows.reset()
        self._measured_at = self._drawn_at = -1.0
        self.scrub.hide()
        self.running = False
        self.run_button.rest()
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
            self._measure(force=True)
        if self._handled_at is not None and self.live.t - self._handled_at >= PLAYED_AFTER:
            self.trials.set_played()
        self._draw_live()
        self._draw_tab()

    def _measure(self, force: bool = False) -> None:
        """Land the pulses a period past their end, and measure the wave's harmonics (every MEASURE_EVERY s)."""
        if not force and self.live.t - self._measured_at < MEASURE_EVERY:
            return
        self._measured_at = self.live.t
        landed = False
        for push in list(self._pending):
            if not (push.quiet and push.alone):
                self._pending.remove(push)
                continue
            peak = lf.pulse_peak(self.live, push)
            if peak is not None:
                self._pending.remove(push)
                if peak.kind in SPECTRUM_KINDS:
                    self.peaks.append(peak)
                    landed = True
        if landed:
            self._draw_pulse()
        if self.live.wave is not None:
            self._harmonics = lf.harmonics(self.live, self.harmonics.value())

    def clear_points(self) -> None:
        self.peaks.clear()
        self._draw_pulse()

    # ------------------------------------------------------------- scrubbing
    def _show_scrub(self) -> None:
        r = self.live.record
        if r.size > 1:
            end = float(r.t[r.size - 1])
            self.scrub.show(end, max(float(r.t[0]), end - STRIP_SECONDS))

    def _on_scrubbed(self, t: float) -> None:
        r = self.live.record
        ts = r.t[: r.size]
        x = float(np.interp(t, ts, r.y[: r.size, COL["x"]]))
        self.body.show(x)
        self._scrubbed = t
        self.clock.setText(f"t = {sig(t, 4)} s (paused at {sig(self.live.t, 4)} s)")
        self._draw_conv()

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.body.apply_theme()
        self.forces.apply_theme()
        self.scrub.apply_theme()
        self.marks.apply_theme()
        for button in (self.wave_on,):
            button.apply_theme()
        self.x_item.setPen(pg.mkPen(c.mass[0], width=1))
        self.f_item.setPen(pg.mkPen(c.force, width=1))
        self.hit_item.setSymbolBrush(pg.mkBrush(c.force))
        self.hit_item.setSymbolPen(pg.mkPen(c.force))
        self.conv_f.setPen(pg.mkPen(c.force, width=2))
        self.conv_h.setPen(pg.mkPen(c.mass[0], width=2))
        self.conv_hits.setSymbolBrush(pg.mkBrush(c.force))
        fill = pg.mkColor(c.mass[1])
        fill.setAlphaF(0.35)
        self.prod.setPen(pg.mkPen(c.mass[1], width=1.5))
        self.prod.setBrush(pg.mkBrush(fill))
        for i, k in enumerate(SPECTRUM_KINDS):
            self.pulse_points[k.name].setSymbolBrush(pg.mkBrush(c.mass[i]))
            self.pulse_points[k.name].setSymbolPen(pg.mkPen(c.structure, width=1))
            self.pulse_lines[k.name].setPen(pg.mkPen(c.mass[i], width=1.5))
        self.force_stems.setPen(pg.mkPen(c.force, width=3))
        self.resp_stems.setPen(pg.mkPen(c.mass[0], width=3))
        for item in (self.force_series, self.resp_series):
            item.setSymbolBrush(pg.mkBrush(c.background))
            item.setSymbolPen(pg.mkPen(c.strong, width=1.5))
        self.receptance_item.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        for line in self.fn_lines:
            line.setPen(pg.mkPen(c.faint, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self._draw_live()
        self._draw_tab()

    @coalesce.coalesced
    def _draw_live(self, frame: bool = False) -> None:
        """What moves: the mass and its forces, the time plots, the readout (and, as a `frame` of the run, the
        Convolution tab at its own rate)."""
        live = self.live
        t, y = live.record.window(STRIP_SECONDS)
        end = live.t
        self.x_item.setData(t, y[:, COL["x"]])
        self.f_item.setData(t, y[:, COL["f"]])
        hits = [(th, i) for th, i in live.hits if th >= end - STRIP_SECONDS]
        self.hit_item.setData([th for th, _ in hits], [0.0 for _ in hits])
        self.x_plot.setXRange(max(0.0, end - STRIP_SECONDS), max(end, STRIP_SECONDS), padding=0)
        self.marks.draw(live.edits, end - STRIP_SECONDS)
        recent = y[t >= end - PEAK_SECONDS, COL["x"]] if t.size else np.empty(0)
        peak = float(np.abs(recent).max()) if recent.size else 0.0
        if live.held_at is not None:
            peak = max(peak, abs(live.held_at))
        self.body.fit(peak)
        self._draw_body()
        self.clock.setText(f"t = {sig(end, 4)} s")
        self.readout.setText(view.drive_note(live))
        self.let_go_button.setEnabled(live.holding())
        if frame and self.tabs.currentIndex() == view.TAB_CONV and end - self._drawn_at >= TAB_EVERY:
            self._draw_conv()
        elif frame and self.tabs.currentIndex() == view.TAB_HARM and end - self._drawn_at >= MEASURE_EVERY:
            self._draw_harm()

    def _draw_body(self) -> None:
        live = self.live
        p = live.osc
        hand = p.stiffness * live.held_at if live.held_at is not None else None
        body = one_mass(
            p.stiffness, p.damping, live.x, live.v, live.force_now(), hand=hand, scale=self.arrows, t=live.t
        )
        self.body.show(live.x)
        self.forces.show(body)

    @coalesce.coalesced
    def _draw_tab(self, *_: object) -> None:
        i = self.tabs.currentIndex()
        if i == view.TAB_CONV:
            self._draw_conv()
        elif i == view.TAB_PULSE:
            self._draw_pulse()
        elif i == view.TAB_HARM:
            self._draw_harm()

    @coalesce.coalesced
    def _draw_conv(self, *_: object) -> None:
        self._drawn_at = self.live.t
        at = self._scrubbed if self._scrubbed is not None and not self.running else None
        v = view.conv_view(lf.convolution(self.live, at), self.live.osc)
        self.conv_f.setData(*v.force)
        self.conv_h.setData(*v.h)
        self.conv_hits.setData(*v.hits)
        self.prod.setData(*v.product)
        self.conv_note.setText(v.note)

    @coalesce.coalesced
    def _draw_pulse(self, *_: object) -> None:
        state = self.practice_state()
        v = view.pulse_view(self.peaks, view.oscillator(state).zeta, state.theory)
        for name, item in self.pulse_points.items():
            item.setData(*v.points[name])
        for name, item in self.pulse_lines.items():
            curve = v.theory.get(name)
            if curve is None:
                item.setData([], [])
            else:
                item.setData(*curve)
        self.pulse_note.setText(v.note)

    @coalesce.coalesced
    def _draw_harm(self, *_: object) -> None:
        self._drawn_at = self.live.t
        state = self.practice_state()
        osc = view.oscillator(state)
        wave = self.live.wave or view.forcing(state)
        v = view.harmonics_view(self._harmonics, osc, wave, state.harmonics, state.theory)
        self.force_stems.setData(*view.stems(v.force.x, v.force.y, 0.0))
        self.resp_stems.setData(*view.stems(v.response.x, v.response.y, 0.0))
        self.force_series.setData(*v.force_series)
        self.resp_series.setData(*v.response_series)
        self.receptance_item.setData(*v.receptance)
        for line in self.fn_lines:
            line.setValue(v.natural_hz)
        self.harm_note.setText(v.note)
