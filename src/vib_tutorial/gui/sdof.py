"""Single oscillator page (#478's pilot): one mass on a spring and a damper, running live, changed as it runs.

The mass is stepped as it runs (`view_models/live_mass.LiveMass`: the chain's exact FOH `Simulator` with one mass),
not a response worked out in advance and played back. Its damper follows a law the reader picks as it runs:
viscous, or hysteretic, Coulomb friction or the Zener element, each matched to c at f<sub>n</sub> (the Damping
models page's, merged here: one example, one page), the viscous ghost beside it, in gray, plucked and driven as it
is. Left: the oscillator's m, k, c and the damper's law, the drive, then the Things to try (setups to go back to;
their predictions are asked once the reader has handled the system). Center: the mass on its spring and damper
(`live_body.LiveMassView`: pluck it, or drag the grips on the mass, the spring and the damper to change m, k and c
as it runs, the motion carrying on by `Simulator.set_system`'s rules) with the forces on it beside it, on the mass
held still (`free_body_view.FreeBodyView`), the run's controls, and the time plots, which keep what came before a
change with the change marked. Right: the Frequency response the reader builds by dwelling or sweeping the drive
(`view_models/recorder.py`) with the driven state's force phasors, the Energy balance through every change, the
Loop & loss (the hysteresis loop the run draws, beside each law's, and their loss factors), the Poles (drag them:
k and c follow) with the phase plane, and the Theory.

What the page shows is `view_models/sdof.py`'s and, for the laws' panels, `view_models/damping.py`'s; its trials,
problems, tips and notes are `sdof_trials.py`'s and `damping_trials.py`'s (no Qt). The controls are bound to the
state (`SdofTrial`) by `bindings.Bindings`, so Defaults round-trips by construction.

Working on it: `tests/test_sdof_page.py` (the page live: pluck, grips, a change mid-run, the recorder, the sweep,
predictions after play, the drawn body's grips, a law changed mid-run, the ghost, the loop's energy, friction
sticking), `test_live_parts.py` (the Qt-free parts, the laws' physics), `test_readiness.py`.
"""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.damping_models import MODELS, Model
from ..core.stepping import FRAME_BUDGET, Budget
from ..view_models import damping as laws
from ..view_models import recorder as rec
from ..view_models import sdof as view
from ..view_models.free_body import ArrowScale, one_mass
from ..view_models.grips import Grip
from ..view_models.live_mass import COL, UNBALANCE, LiveMass, ring_down
from ..view_models.sdof import SdofTrial
from ..view_models.text import fmt
from ..view_models.text import number as sig
from . import coalesce
from .bindings import Bindings
from .drag import DragPoint
from .frame_loop import FrameLoop
from .free_body_view import FreeBodyView
from .layouts import FlowLayout, group
from .lesson import ActionToggle, Dot, Form, LiveLabel, TrialBox, combo, note_label, plot, set_quietly, three_panes
from .live_body import LiveMassView
from .live_parts import ChangeMarks, EnergyTab, LoopTab, run_row
from .notes_view import NotesBrowser
from .panels import spin
from .plots import decimate
from .scrub import Scrub
from .sdof_trials import NOTES_HTML, PROBLEMS, TIPS, TRIALS
from .style import colors, role
from .transport import Transport

__all__ = ["NOTES_HTML", "PROBLEMS", "TRIALS", "SdofPage", "notes_html"]

STRIP_SECONDS = 10.0  # the time plots' window
PHASE_SECONDS = 3.0  # the phase plane's trail
SWEEP_TIP = (
    "<p>The sweep's range, as multiples of f<sub>n</sub> (so it follows f<sub>n</sub> as you change m or k), and "
    "about how many frequencies it dwells at: log-spaced, and with 9 or more, five of them packed round the "
    "resonance. Few points, or a range that misses f<sub>n</sub>, and the peak is missed too. Applies at the next "
    "Sweep; the plots' frequency axis shows the range.</p>"
)
DRIVE_TEXTS = (  # the drive's button, off and on: a force, an unbalance
    ("Apply force", "Force on: click to stop"),
    ("Spin the unbalance", "Unbalance spinning: click to stop"),
)
PEAK_SECONDS = 3.0  # the body's scale fits the largest swing this recent
MEASURE_EVERY = 0.2  # s of run between the recorder's looks
PLAYED_AFTER = 2.0  # s of run after the reader's first change: they've handled it, predictions on
GRIPS = {
    "mass": Grip("mass", 0.01, 1e4, per=0.4),
    "stiffness": Grip("stiffness", 0.01, 1e8, per=0.3),
    "damping": Grip("damping", 0.01, 1e6, per=0.3, zero_below=0.02),
}
POLES_TIP = (
    "<p>The roots of m s² + c s + k in the complex plane: −ζω<sub>n</sub> ± iω<sub>d</sub>. Their distance "
    "from the origin is ω<sub>n</sub>, the angle from the imaginary axis is sin⁻¹ζ. The rays are ζ = 0.05, "
    "0.2, 0.5 and 0.8, the arc the current ω<sub>n</sub>. Drag a pole (or click where it should go) and k and c "
    "follow, the mass kept, while it runs. Drag one onto the real axis for critical damping, then along it to "
    "split the pair.</p>"
)
FRF_TIP = (
    "<p>The frequency response, built by hand: each point is a steady swing the run measured at the drive's "
    "frequency (its amplitude over the static one, |X|k/F, or m|X|/m<sub>e</sub>e for the unbalance, and the "
    "phase by which it lags). Points from before a change of m, k or c are faded. <i>Show the theory</i> draws "
    "the curve they should lie on.</p>"
)
STRIP_TIP = (
    "<p>The mass's displacement as it runs, the last 10 s, and under another damping law the viscous ghost's, "
    "gray and dashed. A dashed line marks where m, k, c or the law changed: the motion carries on through it. "
    "Paused, drag the line at the end back to see the mass then.</p>"
)
PHASOR_TIP = (
    "<p>The driven steady state's forces as phasors, turning at the drive's frequency: the spring's kX, the "
    "damping's (a quarter turn ahead of the motion) and the mass's −Ω²mX, head to tail, add up to the drive F. "
    "At resonance the spring's and the mass's cancel, and the damping's alone meets the drive.</p>"
)


def notes_html() -> str:
    """The notes with the course map's progress as it stands (the reader's problems solved, trials tried)."""
    from . import links
    from . import progress as record
    from .links import course_map
    from .practice import progress

    solved = progress()
    pages = {item.partition("#")[0] for item in record.tried()} & set(links.order())  # a page since renamed: out
    totals = {page: len(links.trials_of(page)) for page in pages}
    tried = record.tried_counts(totals)
    if not solved and not tried:  # nothing done yet: the notes as built
        return NOTES_HTML
    return NOTES_HTML.replace(course_map(), course_map(solved, tried))


class SdofPage(QtWidgets.QWidget):
    """The oscillator's controls on the left, the live mass and its time plots in the center, the tabs on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def transport(self) -> Transport:
        return Transport(self.run_button, self.reset_button, self.speed, self.clock)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.live = LiveMass()
        self.ghost = LiveMass()  # the viscous mass, beside it under another law
        self.recorder = rec.Recorder()
        self.sweep: rec.Sweep | None = None
        self.running = False
        self.arrows = ArrowScale(room=1.0)
        self._handled_at: float | None = None  # the run's time at the reader's first change since the last setup
        self._grip_start = 0.0
        self._measured_at = -1.0
        self._pole_held: complex | None = None  # the pole not being dragged, while one is
        self._budget = Budget()
        self.bindings = Bindings(SdofTrial, self)

        # --- left: the oscillator, the drive, then the Things to try
        box = Form("Oscillator")
        self.mass = box.number("Mass m:", 0.01, 1e4, 1.0, 3, " kg", TIPS["mass"])
        self.stiffness = box.number("Stiffness k:", 0.01, 1e8, 400.0, 2, " N/m", TIPS["stiffness"])
        self.damping = box.number("Damping c:", 0.0, 1e6, 2.0, 3, " N·s/m", TIPS["damping"])
        self.model = combo([m.value for m in MODELS], TIPS["model"])
        box.add("Damper:", self.model, TIPS["model"])
        self.friction_mm = box.number("Friction at X:", 0.01, 1000.0, 10.0, 2, " mm", TIPS["friction_mm"])
        self.relax = box.number("Zener peak at:", 0.01, 100.0, 1.0, 3, " × fₙ", TIPS["relax"])
        self.ghost_on = QtWidgets.QCheckBox("Viscous ghost beside it (gray)")
        self.ghost_on.setToolTip(TIPS["ghost"])
        box.row(self.ghost_on)
        self.info = note_label()
        box.row(self.info)
        self.oscillator_box = box

        drive = Form("Drive")
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["Force F sin Ωt", "Rotating unbalance"])
        drive.add("Input:", self.kind, TIPS["kind"])
        self.amplitude = drive.number("Force F:", 0.0, 1e6, 1.0, 4, " N", TIPS["amplitude"])
        self.amplitude_label = drive.label(self.amplitude)
        self.freq = drive.number("Frequency:", 0.01, 1e4, 3.0, 3, " Hz", TIPS["freq"])
        self.tune = QtWidgets.QPushButton("Tune to fₙ")
        self.tune.setToolTip("<p>Set the drive's frequency to the natural frequency.</p>")
        self.tune.clicked.connect(self.tune_to_natural)
        drive.row(self.tune)
        self.drive_on = ActionToggle(*DRIVE_TEXTS[0])  # set up first, then applied: as the Simulation page's
        self.drive_on.setToolTip(TIPS["drive"])
        drive.row(self.drive_on)

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
        for w in (box, drive, self.trials):
            lv.addWidget(w)
        lv.addStretch(1)

        # --- center: the mass, the run, the time plots
        self.body = LiveMassView()
        self.body.setMinimumHeight(200)
        self.forces = FreeBodyView()
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
        self.strip.setToolTip(STRIP_TIP)
        self.strip.setMouseEnabled(x=False, y=False)
        self.x_item = pg.PlotDataItem(name="x")
        self.ghost_item = pg.PlotDataItem(name="viscous ghost")
        self.peak_item = pg.PlotDataItem(pen=None, symbol="o", symbolSize=6)
        self.strip.addItem(self.ghost_item)
        self.strip.addItem(self.x_item)
        self.strip.addItem(self.peak_item)
        self.force_plot = plot("Drive f", "N", "Time", "s", height=90)
        self.force_plot.setXLink(self.strip)
        self.force_plot.setMouseEnabled(x=False, y=False)
        self.f_item = pg.PlotDataItem()
        self.force_plot.addItem(self.f_item)
        decimate(self.strip, self.force_plot)
        self.marks = ChangeMarks(self.strip, self.force_plot)
        self.scrub = Scrub(self.strip, self.force_plot)
        self.scrub.scrubbed.connect(self._on_scrubbed)
        self.readout = LiveLabel(wrap=True)  # rewritten each frame as the swing decays: no layout pass each time

        centre = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(centre)
        cv.setContentsMargins(0, 0, 0, 0)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.body, 3)
        top.addWidget(self.forces, 2)
        cv.addLayout(top, 3)
        cv.addLayout(row)
        cv.addWidget(self.strip, 2)
        cv.addWidget(self.force_plot, 1)
        cv.addWidget(self.readout)

        # --- right: Energy, Frequency response, Loop & loss, Poles, Theory
        frf = QtWidgets.QWidget()
        fv = QtWidgets.QVBoxLayout(frf)
        self.mag_plot = plot("Amplification |X|k/F", None, "Frequency", "Hz", log_x=True, log_y=True, height=140)
        self.lag_plot = plot("Lag", "°", "Frequency", "Hz", log_x=True, height=100)
        self.lag_plot.setXLink(self.mag_plot)
        for p in (self.mag_plot, self.lag_plot):
            p.enableAutoRange(x=False)  # the frequency axis is the sweep's range (`_draw_frf`), not the curves'
        self.mag_plot.enableAutoRange(y=False)  # and the amplification's is set there too, theory or not
        self.nyq_plot = plot("Im H", "m/N", "Re H", "m/N", height=120)
        self.nyq_plot.setAspectLocked(True)
        for p in (self.mag_plot, self.lag_plot, self.nyq_plot):
            p.setToolTip(FRF_TIP)
            p.setMouseEnabled(x=False, y=False)
        self.band = pg.LinearRegionItem(movable=False)
        self.band.hide()
        self.mag_plot.addItem(self.band)
        # PlotDataItems, not ScatterPlotItems: only they follow the plots' log axes
        self.theory_items = [pg.PlotDataItem() for _ in range(3)]
        self.frf_ghost = pg.PlotDataItem()  # the viscous ghost's, under another law
        self.earlier_items = [pg.PlotDataItem(pen=None, symbol="o", symbolSize=6) for _ in range(2)]
        self.point_items = [pg.PlotDataItem(pen=None, symbol="o", symbolSize=8) for _ in range(3)]
        self.join_items = [pg.PlotDataItem() for _ in range(3)]  # a smooth curve through the points, under them
        self.drive_items = [pg.PlotDataItem(pen=None, symbol="o", symbolSize=12) for _ in range(2)]
        self.fn_lines = [pg.InfiniteLine(angle=90) for _ in range(2)]
        for p, items in (
            (
                self.mag_plot,
                (
                    self.theory_items[0],
                    self.fn_lines[0],
                    self.earlier_items[0],
                    self.join_items[0],
                    self.point_items[0],
                ),
            ),
            (
                self.lag_plot,
                (
                    self.theory_items[1],
                    self.fn_lines[1],
                    self.earlier_items[1],
                    self.join_items[1],
                    self.point_items[1],
                ),
            ),
            (self.nyq_plot, (self.theory_items[2], self.join_items[2], self.point_items[2])),
        ):
            for item in items:
                p.addItem(item)
        self.mag_plot.addItem(self.frf_ghost)
        self.mag_plot.addItem(self.drive_items[0])
        self.lag_plot.addItem(self.drive_items[1])
        self.lag_plot.setYRange(0, 180, padding=0.05)
        self.lag_plot.getAxis("left").setTicks([[(v, f"{v}") for v in (0, 45, 90, 135, 180)]])
        self.sweep_button = ActionToggle("Sweep", "Sweeping: click to stop")
        self.sweep_button.setToolTip(
            "<p>Step the drive through frequencies across the resonance, dwelling at each until its point lands. "
            "Press again to stop.</p>"
        )
        self.sweep_button.toggled.connect(self._on_sweep)
        self.clear_button = QtWidgets.QPushButton("Clear points")
        self.clear_button.setToolTip("<p>Forget the measured points.</p>")
        self.clear_button.clicked.connect(self.clear_points)
        self.theory = QtWidgets.QCheckBox("Show the theory")
        self.theory.setToolTip("<p>Draw the curves the points should lie on, and the half-power band.</p>")
        self.theory.toggled.connect(self._draw_frf)
        self.sweep_from = spin(0.01, 100.0, 0.33, 2, " × fₙ")
        self.sweep_to = spin(0.01, 100.0, 3.0, 2, " × fₙ")
        self.sweep_points = QtWidgets.QSpinBox()
        self.sweep_points.setRange(3, 60)
        self.sweep_points.setValue(15)
        for spinner, name in (
            (self.sweep_from, "Sweep from"),
            (self.sweep_to, "Sweep to"),
            (self.sweep_points, "Sweep points"),
        ):
            spinner.setToolTip(SWEEP_TIP)
            spinner.setAccessibleName(name)
        buttons = FlowLayout()
        buttons.add(
            self.sweep_button,
            group("from", self.sweep_from, "to", self.sweep_to),
            group(self.sweep_points, "points"),
            self.clear_button,
            self.theory,
        )
        fv.addLayout(buttons)
        self.phasor_plot = plot("Imaginary", "N", "Real", "N", height=120)
        self.phasor_plot.setAspectLocked(True)
        self.phasor_plot.setMouseEnabled(x=False, y=False)
        self.phasor_plot.setToolTip(PHASOR_TIP)
        self.phasor_items = {name: pg.PlotDataItem() for name in ("spring", "damping", "mass", "drive")}
        for item in self.phasor_items.values():
            self.phasor_plot.addItem(item)
        planes = QtWidgets.QHBoxLayout()
        planes.addWidget(self.nyq_plot, 1)
        planes.addWidget(self.phasor_plot, 1)
        fv.addWidget(self.mag_plot, 3)
        fv.addWidget(self.lag_plot, 2)
        fv.addLayout(planes, 2)
        self.frf_note = note_label()
        fv.addWidget(self.frf_note)

        self.energy = EnergyTab()
        self.loop_tab = LoopTab(TIPS["damping_only"])
        self.damping_only = self.loop_tab.damping_only

        poles = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(poles)
        self.s_plot = plot("Imaginary part", "rad/s", "Real part", "1/s", height=150)
        self.s_plot.setAspectLocked(True)
        self.s_plot.setMouseEnabled(x=False, y=False)
        self.s_plot.setToolTip(POLES_TIP)
        self.rays = [pg.PlotDataItem() for _ in range(len(view.RAYS))]
        self.arc = pg.PlotDataItem()
        self.pole_dots = [DragPoint("<p>Drag the pole: k and c follow (the mass is kept).</p>") for _ in range(2)]
        self.pole_dots[0].dragged.connect(self._on_upper_dragged)
        self.pole_dots[1].dragged.connect(self._on_lower_dragged)
        for item in (*self.rays, self.arc, *self.pole_dots):
            self.s_plot.addItem(item)
        for dot in self.pole_dots:
            dot.released.connect(self._on_pole_released)
        self.s_plot.scene().sigMouseClicked.connect(self._on_pole_click)
        self.pole_note = note_label()
        self.plane = plot("Velocity / ωₙ", "m", "Displacement", "m", height=150)
        self.plane.setAspectLocked(True)
        self.plane.setMouseEnabled(x=False, y=False)
        self.plane.setToolTip(
            "<p>The motion as a path in the phase plane, the last 3 s: velocity over ω<sub>n</sub> against "
            "displacement. Undamped it is a circle; damping winds it into a spiral, a turn a cycle.</p>"
        )
        self.plane_item = pg.PlotDataItem()
        self.plane.addItem(self.plane_item)
        self.plane_now = Dot(10)
        self.plane.addItem(self.plane_now, ignoreBounds=True)
        sv.addWidget(self.s_plot, 1)
        sv.addWidget(self.pole_note)
        sv.addWidget(self.plane, 1)

        self.notes = NotesBrowser()
        self.notes.setHtml(NOTES_HTML)
        self._notes_html = NOTES_HTML  # redrawn with the course map's progress as the notes show (_refresh_map)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.energy, "Energy")
        self.tabs.addTab(frf, "Frequency response")
        self.tabs.addTab(self.loop_tab, "Loop && loss")
        self.tabs.addTab(poles, "Poles && phase plane")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._on_tab)

        three_panes(self, left, centre, self.tabs, sizes=(320, 760, 560))

        self.bindings.bind("model", self.model, choices=MODELS)
        for field, widget in (
            ("mass", self.mass),
            ("stiffness", self.stiffness),
            ("damping", self.damping),
            ("friction_mm", self.friction_mm),
            ("relax", self.relax),
            ("ghost", self.ghost_on),
            ("drive", self.drive_on),
            ("kind", self.kind),
            ("amplitude", self.amplitude),
            ("freq_hz", self.freq),
            ("damping_only", self.damping_only),
            ("sweep_from", self.sweep_from),
            ("sweep_to", self.sweep_to),
            ("sweep_points", self.sweep_points),
            ("tab", self.tabs),
        ):
            self.bindings.bind(field, widget)
        self.bindings.write(SdofTrial("", ""))
        self.bindings.changed.connect(self._on_field)

        self.loop = FrameLoop(self, "Single oscillator", self._tick, self._runs, self._sim_t)
        self.apply_theme()
        self._on_kind()
        self._push_params()
        self._draw_live()

    # ------------------------------------------------------------- the state
    def practice_state(self) -> SdofTrial:
        """The page's settings, as a trial: what a practice spec checks, and Defaults puts back."""
        return self.bindings.read(SdofTrial("", ""))

    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(SdofTrial("", "", **setup))

    def apply_trial(self, t: SdofTrial) -> None:
        """A setup: the oscillator and the drive as it says, the run from rest, then plucked or driven (or swept)."""
        self.bindings.write(t)
        if t.tune:
            set_quietly(self.freq, view.drive_freq(t))
        self._on_kind()
        self.recorder.clear()
        self._stop_sweep()
        self._push_params()
        self.reset()
        self._handled_at = None
        self.trials.show_note(t.note)
        if t.x0_mm:
            self.body.start_pluck(1e-3 * t.x0_mm)
            for live in (self.live, self.ghost):
                live.hold(1e-3 * t.x0_mm)
                live.let_go()
        if t.sweep:
            self.sweep_button.setChecked(True)  # starts the sweep (and the drive)
        if t.x0_mm or t.drive or t.sweep:
            self._run(True)
        self._draw_tab()

    # ------------------------------------------------------------- what the reader changes
    def _on_field(self, field: str) -> None:
        if field in ("mass", "stiffness", "damping", "model", "friction_mm", "relax"):
            self._handled()
            self._push_params()
        elif field == "ghost":
            self._draw_live()
        elif field == "damping_only":
            self._draw_tab()
        elif field in ("sweep_from", "sweep_to", "sweep_points"):
            self._draw_frf()  # the axes follow the range; a sweep running keeps the one it started with
        elif field in ("drive", "kind", "amplitude", "freq_hz"):
            self._handled()
            if field == "kind":
                self._on_kind()
            if field == "freq_hz" and self.sweep is not None and not self._sweeping_now:
                self.sweep_button.setChecked(False)  # the reader took the frequency over
            self._push_drive()
        elif field == "tab":
            self._on_tab()

    _sweeping_now = False  # the sweep, not the reader, is moving the frequency

    def _handled(self) -> None:
        if self._handled_at is None:
            self._handled_at = self.live.t

    def _on_kind(self) -> None:
        unbalance = self.kind.currentIndex() == UNBALANCE
        self.amplitude_label.setText("Unbalance mₑe:" if unbalance else "Force F:")
        self.amplitude.setSuffix(" kg·m" if unbalance else " N")
        self.amplitude.setDecimals(6 if unbalance else 4)
        self.drive_on.set_texts(*DRIVE_TEXTS[unbalance])

    def _push_params(self) -> None:
        """The live mass (and the ghost) to the controls: a live change, and the drive."""
        state = self.practice_state()
        law = state.model is not Model.VISCOUS
        for widget, shown in (
            (self.friction_mm, state.model is Model.COULOMB),
            (self.relax, state.model is Model.ZENER),
            (self.ghost_on, law),
        ):
            widget.setVisible(shown)
            label = self.oscillator_box.label(widget) if widget is not self.ghost_on else None
            if label is not None:
                label.setVisible(shown)
        self.ghost.set_params(view.viscous(state))
        self.body.set_values(
            state.mass,
            state.stiffness,
            state.damping,
            damper=f"{view.SHORT[state.model]}, c = {fmt(state.damping, 'N·s/m', smallest='m')}" if law else None,
        )
        self.info.setText(view.info(state))
        self._push_drive()  # the drive, and the hysteretic stand-in's c with it
        self._draw_poles()
        if not self.running:
            self._draw_tab()  # running, the next frame draws it

    def _push_drive(self) -> None:
        state = self.practice_state()
        d = view.drive(state)
        for live in (self.live, self.ghost):
            live.set_drive(d)
        self.live.set_params(view.params(state))  # the hysteretic stand-in's c follows the drive
        self._draw_frf()

    def tune_to_natural(self) -> None:
        self.freq.setValue(round(view.oscillator(self.practice_state()).natural_hz, 6))

    def _on_grabbed(self) -> None:
        self.scrub.hide()
        self._handled()
        self.body.start_pluck(self.live.x)
        x = self.live.x
        for live in (self.live, self.ghost):
            live.hold(x)

    def _on_pulled(self, x: float) -> None:
        for live in (self.live, self.ghost):
            live.hold(x)
        self._draw_body()

    def _on_let_go(self) -> None:
        for live in (self.live, self.ghost):
            live.let_go()
        if not self.running:
            self._run(True)  # a pluck sets it going

    def _on_grip_taken(self, name: str) -> None:
        self._grip_start = {"mass": self.mass, "stiffness": self.stiffness, "damping": self.damping}[name].value()

    def _on_grip_moved(self, name: str, distance: float) -> None:
        box = {"mass": self.mass, "stiffness": self.stiffness, "damping": self.damping}[name]
        set_quietly(box, GRIPS[name].dragged(self._grip_start, distance))
        self._handled()
        self._push_params()

    # ------------------------------------------------------------- the poles
    def _set_k_c(self, k: float, c: float) -> None:
        set_quietly(self.stiffness, k)
        set_quietly(self.damping, c)
        self._handled()
        self._push_params()

    def _on_pole_click(self, event) -> None:
        vb = self.s_plot.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        pos = vb.mapSceneToView(event.scenePos())
        sigma, w = pos.x(), abs(pos.y())
        if sigma >= 0 or w <= 0:
            return
        self._set_k_c(*view.pole_pair(self.practice_state().mass, sigma, w))

    def _on_upper_dragged(self, x: float, y: float) -> None:
        self._drag_pole(0, x, y)

    def _on_lower_dragged(self, x: float, y: float) -> None:
        self._drag_pole(1, x, y)

    def _drag_pole(self, i: int, x: float, y: float) -> None:
        vb = self.s_plot.getPlotItem().getViewBox()
        (x0, x1), (y0, y1) = vb.viewRange()
        state = self.practice_state()
        if self._pole_held is None:  # the other pole as the drag began (the dots swap as real poles pass)
            self._pole_held = complex(view.oscillator(state).poles()[1 - i])
        self._set_k_c(*view.pole_drag(view.viscous(state), x, y, self._pole_held, (x1 - x0, y1 - y0)))

    def _on_pole_released(self) -> None:
        self._pole_held = None
        self._draw_poles()  # the range held while dragging, fitted again

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
        """Back to rest at t = 0, waiting for Start; the measured points kept."""
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

    def _tick(self, dt: float) -> None:
        duration = dt * float(self.speed.currentData())
        duration = min(duration, self._budget.allows(duration, FRAME_BUDGET))
        start = time.perf_counter()
        before = self.live.t
        self._advance(duration)
        if self.live.t > before:
            self._budget.spent(time.perf_counter() - start, self.live.t - before)
        self._measure()
        if self._handled_at is not None and self.live.t - self._handled_at >= PLAYED_AFTER:
            self.trials.set_played()
        self._draw_live(frame=True)

    def step(self, duration: float) -> None:
        """Run on by `duration` s of simulated time at once (tests), measuring and drawing as the frames would."""
        end = self.live.t + duration
        while self.live.t < end - 1e-9:
            if self.live.held is not None:
                break
            self._advance(min(0.05, end - self.live.t) + 1e-12)
            self._measure()
        if self._handled_at is not None and self.live.t - self._handled_at >= PLAYED_AFTER:
            self.trials.set_played()
        self._draw_live()

    def _advance(self, duration: float) -> None:
        """The mass on by `duration`, and the ghost to its time (it runs whatever the law: it's cheap, and so
        it's in step when shown)."""
        self.live.advance(duration)
        self.ghost.advance(self.live.t - self.ghost.t)

    # ------------------------------------------------------------- the recorder
    def _measure(self) -> None:
        """Land a point if the driven swing is steady (every MEASURE_EVERY s of run), and step a sweep on."""
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
        o = view.oscillator(self.practice_state())
        s = self.practice_state()
        self.sweep = rec.Sweep(rec.sweep_frequencies(o.natural_hz, o.zeta, s.sweep_points, s.sweep_from, s.sweep_to))
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
        self._push_drive()

    def _stop_sweep(self) -> None:
        self.sweep = None
        set_quietly(self.sweep_button, False)

    def clear_points(self) -> None:
        self.recorder.clear()
        self._draw_frf()

    # ------------------------------------------------------------- scrubbing
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
        g = self.ghost.record
        ghost = float(np.interp(t, g.t[: g.size], g.y[: g.size, COL["x"]])) if self._ghosted() and g.size else None
        recent = r.y[: r.size, COL["x"]][(ts >= t - PEAK_SECONDS) & (ts <= t)]  # the swing then, not at the pause
        self.body.fit(max(float(np.abs(recent).max()) if recent.size else 0.0, abs(x)), at_once=True)
        self.body.show(x, ghost)
        self.forces.show(one_mass(p.stiffness, p.damping, x, v, f, scale=self.arrows, t=t))
        self.plane_now.setPos(x, v / view.oscillator(self.practice_state()).omega_n)
        self.clock.setText(f"t = {sig(t, 4)} s (paused at {sig(self.live.t, 4)} s)")

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.body.apply_theme()
        self.forces.apply_theme()
        self.drive_on.apply_theme()
        self.sweep_button.apply_theme()
        self.scrub.apply_theme()
        self.x_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.ghost_item.setPen(
            pg.mkPen(c.grey, width=1.5)
        )  # solid: a long dashed line, restroked each frame, took 4-6 ms a frame on a Retina Mac
        self.frf_ghost.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.loop_tab.apply_theme()
        self.peak_item.setSymbolPen(None)
        self.peak_item.setSymbolBrush(pg.mkBrush(c.force))
        self.f_item.setPen(pg.mkPen(c.force, width=1.5))
        self.marks.apply_theme()
        for item in self.theory_items:
            item.setPen(pg.mkPen(c.strong, width=1.5))
        for item in self.earlier_items:
            item.setSymbolPen(pg.mkPen(c.grey, width=1))
            item.setSymbolBrush(None)
        for item in self.point_items:
            item.setSymbolPen(None)
            item.setSymbolBrush(pg.mkBrush(c.mass[0]))
        for item in self.join_items:
            item.setPen(pg.mkPen(c.mass[0], width=1.5))
        for item in self.drive_items:
            item.setSymbolPen(pg.mkPen(c.force, width=2))
            item.setSymbolBrush(None)
        self.band.setBrush(pg.mkBrush(*pg.mkColor(c.fit).getRgb()[:3], 45))
        for line in self.band.lines:
            line.setPen(pg.mkPen(c.fit, width=1))
        for line in self.fn_lines:
            line.setPen(pg.mkPen(c.muted, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.energy.apply_theme()
        self.plane_item.setPen(pg.mkPen(c.mass[0], width=1.5))
        self.plane_now.setPen(pg.mkPen(c.background, width=1.5))
        self.plane_now.setBrush(pg.mkBrush(c.mass[0]))
        for ray in self.rays:
            ray.setPen(pg.mkPen(c.faint, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.arc.setPen(pg.mkPen(c.faint, width=1))
        for dot in self.pole_dots:
            dot.apply_theme()
        self._draw_frf()
        self._draw_poles()
        self._draw_tab()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._refresh_map()

    def _on_tab(self, *_: object) -> None:
        self._refresh_map()
        self._draw_tab()

    def _key(self) -> tuple:
        """The system the recorder's points belong to (`view.key`)."""
        return view.key(self.practice_state())

    def _ghosted(self) -> bool:
        return view.ghosted(self.practice_state())

    @coalesce.coalesced(covers=("_draw_live",))
    def _draw_tab(self, *_: object) -> None:
        """The tab showing: the loss curves (they change only with the settings), then what moves on it."""
        if self.tabs.currentIndex() == view.TAB_LOOP:
            self.loop_tab.draw_loss(laws.loss_view(self.practice_state()))
        self._draw_live()

    def _refresh_map(self, *_: object) -> None:
        """The notes again if the course map's progress has changed, at the same place."""
        if self.tabs.currentWidget() is not self.notes:
            return
        html = notes_html()
        if html != self._notes_html:
            bar = self.notes.verticalScrollBar()
            at = bar.value()
            self.notes.setHtml(html)
            bar.setValue(at)
            self._notes_html = html

    @coalesce.coalesced
    def _draw_live(self, frame: bool = False) -> None:
        """What moves: the body and its forces, the time plots, the phase plane, the readout, the energy (as a
        `frame` of the run, at the Energy tab's own rate)."""
        live = self.live
        r = live.record
        t, y = r.window(STRIP_SECONDS)
        end = live.t
        self.x_item.setData(t, y[:, COL["x"]])
        ghost = self._ghosted()
        self.ghost_item.setVisible(ghost)
        if ghost:
            tg, yg = self.ghost.record.window(STRIP_SECONDS)
            self.ghost_item.setData(tg, yg[:, COL["x"]])
        self.f_item.setData(t, y[:, COL["f"]])
        self.strip.setXRange(max(0.0, end - STRIP_SECONDS), max(end, STRIP_SECONDS), padding=0)
        self.marks.draw(self.live.edits, end - STRIP_SECONDS)
        recent = y[t >= end - PEAK_SECONDS, COL["x"]] if t.size else np.empty(0)
        peak = float(np.abs(recent).max()) if recent.size else 0.0
        if live.held is not None:
            peak = max(peak, abs(live.held))
        self.body.fit(peak)
        self._draw_body()
        o = view.oscillator(self.practice_state())
        _, yp = r.window(PHASE_SECONDS)
        curve = view.phase_plane(yp[:, COL["x"]], yp[:, COL["v"]], o.omega_n)
        self.plane_item.setData(curve.x, curve.y)
        self.plane_now.setPos(live.x, live.v / o.omega_n)
        self.clock.setText(f"t = {sig(end, 4)} s")
        self._show_readout()
        i = self.tabs.currentIndex()
        if i == view.TAB_ENERGY:
            self.energy.draw(live.record, frame)
        elif i == view.TAB_LOOP:
            self._draw_loop()
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
        self.body.show(live.x, self.ghost.x if self._ghosted() else None)
        self.forces.show(body)

    def _show_readout(self) -> None:
        live = self.live
        if live.held is not None:
            self.readout.setText(f"Held at x = {fmt(live.held, 'm', largest='')}: let go to pluck it.")
            self.peak_item.setData([], [])
            return
        if live.drive.on:
            self.peak_item.setData([], [])
            f = live.drive.freq_hz
            points = [p for p in self.recorder.current(self._key()) if abs(p.freq_hz - f) <= 1e-6 * f]
            if points:
                h = points[-1].response
                x = abs(h) * live.drive.force
                lag = (-np.degrees(np.angle(h))) % 360
                self.readout.setText(
                    f"Steady at {sig(f)} Hz: x swings ±{fmt(x, 'm', largest='')}, lagging the drive by {sig(lag)}°."
                )
            else:
                self.readout.setText(f"Driven at {sig(f)} Hz: waiting for the swing to settle…")
            return
        t, y = live.since_change()
        state = self.practice_state()
        if state.model is Model.COULOMB and state.damping > 0:
            self.peak_item.setData([], [])
            still = t.size > 0 and live.v == 0.0 and live.params.friction > 0
            text = laws.decay_readout(state, t, y[:, COL["x"]], still) if t.size else ""
            self.readout.setText(text or ("Pluck the mass: drag it sideways and let go." if t.size < 2 else ""))
            return
        decay = ring_down(t, y[:, COL["x"]]) if t.size else None
        if decay is None:
            self.peak_item.setData([], [])
            self.readout.setText("Pluck the mass: drag it sideways and let go." if t.size < 2 else "")
            return
        self.peak_item.setData(decay.times, decay.heights)
        self.readout.setText(decay.text)

    def _period(self) -> float | None:
        live = self.live
        if live.drive.on:
            return 1.0 / live.drive.freq_hz
        if live.record.size < 2:
            return None
        return 1.0 / view.oscillator(self.practice_state()).natural_hz

    def _draw_loop(self) -> None:
        period = self._period()
        t, y = self.live.record.window(1.1 * period if period else 1.0)
        state = self.practice_state()
        self.loop_tab.draw_loop(laws.loop_view(state, t, y[:, COL["x"]], y[:, COL["v"]], y[:, COL["f"]], period))

    def _draw_phasors(self) -> None:
        drawn = {ph.name: ph for ph in laws.phasors(self.practice_state(), self.live.t)}
        for name, item in self.phasor_items.items():
            ph = drawn.get(name)
            if ph is None or not all(np.isfinite((*ph.start, *ph.end))):
                item.setData([], [])
                continue
            item.setData([ph.start[0], ph.end[0]], [ph.start[1], ph.end[1]])
            item.setPen(pg.mkPen(role(ph.role), width=3))

    @coalesce.coalesced
    def _draw_frf(self, *_: object) -> None:
        state = self.practice_state()
        key = view.key(state)
        v = view.frf_view(state, self.recorder.current(key), self.recorder.earlier(key), self.theory.isChecked())
        self.mag_plot.setLabel("left", v.amplification)
        self.point_items[0].setData(*v.points)
        self.point_items[1].setData(*v.lags)
        self.point_items[2].setData(*v.nyquist)
        for item, curve in zip(self.join_items, (v.joined, v.joined_lag, v.joined_nyquist)):
            item.setData(*curve)
        self.earlier_items[0].setData(*v.earlier)
        self.earlier_items[1].setData(*v.earlier_lags)
        self.theory_items[0].setData(*v.theory)
        self.theory_items[1].setData(*v.theory_lag)
        self.theory_items[2].setData(*v.nyquist_theory)
        self.frf_ghost.setData(*v.ghost)
        for item, curve in zip(self.drive_items, (v.drive, v.drive_lag)):
            item.setData(*curve)
        for line in self.fn_lines:
            line.setValue(np.log10(v.natural_hz))
        if v.band is not None:
            self.band.setRegion([np.log10(v.band[0]), np.log10(v.band[1])])
            self.band.show()
        else:
            self.band.hide()
        self.frf_note.setText(v.note)
        self.mag_plot.setXRange(*np.log10(v.x_span), padding=0)  # the lag plot's is linked
        self.mag_plot.setYRange(*np.log10(v.y_span), padding=0)  # the same with the theory shown or not

    @coalesce.coalesced
    def _draw_poles(self) -> None:
        state = self.practice_state()
        pv = view.poles_view(view.viscous(state), state.model)
        for dot, x, y in zip(self.pole_dots, pv.poles.x, pv.poles.y):
            dot.place(float(x), float(y))
        for ray, curve in zip(self.rays, pv.rays):
            ray.setData(*curve)
        self.arc.setData(*pv.arc)
        self.pole_note.setText(pv.note)
        if self._pole_held is None:  # hold the range still under the hand
            reach = 1.15 * float(max(np.abs(pv.poles.x).max(), np.abs(pv.poles.y).max(), 1e-6))
            self.s_plot.setXRange(-reach, 0.1 * reach, padding=0)
            self.s_plot.setYRange(-reach, reach, padding=0)
