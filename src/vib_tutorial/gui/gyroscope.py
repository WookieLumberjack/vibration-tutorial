"""Gyroscope page: a spinning top, from Euler's equations to precession and nutation, and on a table.

Left: the setting (a disc on a light stem with its tip in a cup, the same disc on a round tip on a table, or a
tippe top on a table), the top (its mass and size, the round tip and the friction on a table, and gravity), how it
is let go (the spin, the tilt, and in the cup the precession: at rest, at either steady rate, or set), and the
playback. Centre: the top in 3D (drag to turn it), with its axis's path and the vectors ω, L and the moment (on a
table the friction), then its tilt against time, beside the turning points worked out in closed form in the cup.
Right: the effective potential the tilt moves in (on a table, the least energy Jellett's constant allows), the
steady precession rates against the spin, the axis's path from above (on a table, the tip's track), the
components of ω, L, τ and the inertia tensor in fixed, body or axis-frame axes (core/gyroscope.py's `frame_axes`,
`inertia_in`), with the equations written out at one instant under the plot (`snapshot`, drawn by
gui/equations_snapshot.py), the simulation's check against what is conserved, and the Theory (its figures: gui/gyroscope_figures.py). The page
has its own models (core/gyroscope.py, core/sliding_top.py), stepped on its own frame loop while it shows.

The trials and practice problems are gui/gyroscope_trials.py's (re-exported here, where the practice window and the
Theory tab look for them), and the 3D view is gui/gyroscope_view.py's `TopView`.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.gyroscope import (
    LAUNCHES,
    RPM,
    SHAPES,
    Frame,
    FreeSpin,
    Launch,
    Motion,
    Shape,
    Top,
    TopSimulator,
    angular_velocity,
    body_rates,
    frame_axes,
    gravity_torque,
    in_frame,
    inertia_in,
    least_spin,
    snapshot,
    steady_precession,
    tilt_and_azimuth,
)
from ..core.sliding_top import COLUMNS, RoundTop, SlideSamples, SlidingSimulator, contact_torque
from ..view_models.components import SHOW_I, SHOW_L, SHOW_W, series
from . import coalesce, gyroscope_figures
from .cms_time import SampleBuffer
from .components_tab import ComponentsTab
from .equations_snapshot import snapshot_html
from .frame_loop import FrameLoop
from .gyroscope_trials import (
    PLAYBACK,
    REAL_TIME,
    SETTINGS,
    TAB_CHAOS,
    TAB_CHECK,
    TAB_COMPONENTS,
    TAB_PATH,
    TAB_POTENTIAL,
    TAB_PRECESSION,
    GyroTrial,
    Setting,
)
from .gyroscope_trials import PROBLEMS as PROBLEMS
from .gyroscope_trials import TRIALS as TRIALS
from .gyroscope_view import AXES_SHOWN, TopView
from .lesson import (
    TAB_ROOM,
    Dot,
    Form,
    RunButton,
    TrialBox,
    column,
    combo,
    connect_all,
    fmt,
    mark,
    marker,
    note_label,
    plot,
    readout_label,
    relabel,
    set_quietly,
    stack,
    theory,
    three_panes,
    trials_html,
)
from .links import load_notes
from .live_label import LiveLabel
from .plots import thin
from .style import colors
from .theming import add_legend, mute
from .transport import Transport

NODS = 6  # nutation periods the tilt plot spans
TRAIL_POINTS = 1500  # at most, on the 3D view's trail
PATH_POINTS = 4000  # at most, on the path from above
CHECK_POINTS = 2000
FRAMES = tuple(Frame)
AXIS_NAMES = {Frame.SPACE: ("x", "y", "z"), Frame.BODY: ("1", "2", "3"), Frame.AXIS: ("1′", "2′", "3")}
WHERE = {
    Frame.SPACE: "in fixed axes (x, y, z up)",
    Frame.BODY: "in the body's axes (e₁ the mark, e₃ the axis)",
    Frame.AXIS: "in the axis frame (e₃ the axis, e₁′ horizontal across it)",
}  # the Components tab's snapshot: whose axes
SNAPSHOT_FRAMES = 10  # while it runs the Components tab's snapshot is rewritten every this many frames
CHECK_FRAMES = 30  # the Check tab's text is worked out over the whole run, every this many frames
SECTION_FRAMES = 10  # the Chaos tab is redrawn every this many frames
HISTORY = 400_000  # samples kept
TWIN = 1e-8  # rad: the Chaos tab's second top is let go this much further over
SECTION_POINTS = 8000  # the Poincaré section's, at most
APART_STRIDE = 20  # samples of the two tops' separation, one kept in this many
SKEW_TRAIL, SKEW_TILT, SKEW_PATH = 3.0, 5.0, 10.0  # s of an asymmetric top's trail, tilt and path
SLIDE_HISTORY = 200_000  # samples kept on a table (RECORD_DT apart)
SLIDE_TILT_SPAN = 60.0  # s, at most, of the tilt on a table
TRACK_SPAN = 10.0  # s of the tip's track on the table
TRACK_POINTS = 3000
TABLE_TRAIL = 1.0  # s of the axis's trail and the tip's track in the 3D view, on a table

TIPS = {
    "comp_show": "<p>What to plot as the top runs: its angular velocity ω, its angular momentum L = Iω (about the "
    "tip, or about G on a table), the torque τ that turns L, or the six numbers of its inertia tensor I.</p>",
    "comp_frame": "<p>Whose axes the components are in. <b>Space</b>: fixed, z up. <b>Body</b>: e₁ the mark, e₃ "
    "the axis, turning with the top, spin and all. <b>Axis frame</b>: e₃ the axis, e₁ horizontal across it; it "
    "follows the axis as it precesses and nods, but not the spin about it.</p>",
    "comp_span": "<p>How much of the run to plot.</p>",
    "mass": "<p>The disc's mass (the stem's is left out).</p>",
    "radius": "<p>The disc's radius: I₃ = mr²/2.</p>",
    "side": "<p>The square's side: I₃ = ms²/6, the same as a disc of radius s/√3.</p>",
    "shape": "<p>A disc, or a square plate: a quarter turn leaves the square as it was, so its moment is the same "
    "about every line across its axis, and it is a symmetric top too. A rectangle isn't (I₁ ≠ I₂): an asymmetric top, "
    "in the cup only.</p>",
    "breadth": "<p>The rectangle's other side b, across its mark: I₁ = m(b²/12 + l²), I₂ = m(w²/12 + l²), "
    "I₃ = m(w² + b²)/12.</p>",
    "side_a": "<p>The rectangle's side w, along its mark (the thick line on it).</p>",
    "height": "<p>From the tip to the disc's centre, l: gravity's arm, and I₁ = m(r²/4 + l²) about the tip.</p>",
    "gravity": "<p>g: 9.81 m/s² on Earth; 0 for a free top.</p>",
    "spin": "<p>ω₃, the spin about the axis (ψ̇ + φ̇ cos θ), which stays constant. Negative spins it the other way.</p>",
    "tilt": "<p>The axis's angle from the vertical when it is let go, its tilt not changing then (a turning "
    "point).</p>",
    "launch": "<p>How it is let go: not precessing (φ̇ = 0), at the slow or fast steady rate (where there is "
    "one), or at the rate set below.</p>",
    "precession": "<p>φ̇, the axis's rate round the vertical when it is let go (with <i>Set φ̇ below</i>).</p>",
    "playback": "<p>Simulated time per real second. The top spins far too fast to see: slow it down.</p>",
    "axes": "<p>Draw a frame's axes on the top, at the disc's centre. The <b>body's</b> e₁ (the mark) and e₂ turn "
    "with the top, spin and all; the <b>axis frame's</b> e₁′ (kept horizontal) and e₂′ follow the axis as it "
    "precesses and nods, but not the spin. Both share e₃, the axis, and differ by the spin angle ψ about it. Slow "
    "the playback (0.02×) to see the body's whirl round while the axis frame's only lean and swing.</p>",
    "setting": "<p>The disc (or square) with its tip held in a cup (it can turn there but not slide), the same on a "
    "round tip that slides and rolls on a table, or a tippe top (a ball weighted low) on a table.</p>",
    "tip_radius": "<p>The round tip's radius R: its centre is R up the axis from the end of the stem, and the disc's "
    "centre a = l − R above that.</p>",
    "friction": "<p>μ, Coulomb's friction between the tip and the table: μN against the slip (smoothed below "
    "5 mm/s). 0.01 is a polished, oiled surface; 0.3 is wood.</p>",
    "low": "<p>How far the tippe top's centre of mass sits below the ball's centre, −a (a &lt; 0): α = −a/R.</p>",
    "ratio": "<p>I₁/I₃ about the centre of mass (I₃ = 0.4mR², a ball's). It turns over if 1 − α &lt; I₁/I₃ &lt; "
    "1 + α.</p>",
    "ball": "<p>The tippe top's ball's radius R (its stem left out).</p>",
}
TILT_TIP = (
    "<p>The axis's tilt from the vertical as it runs, the last few nods. Dashed: the turning points the "
    "energy and the two momenta give, in closed form.</p>"
)
POTENTIAL_TIP = (
    "<p>The tilt moves in U(θ) = (p<sub>φ</sub> − p<sub>ψ</sub> cos θ)²/2I₁ sin²θ + mgl cos θ with energy "
    "E': it nods between where U = E' (dashed). The dot is the tilt now.</p>"
)
PRECESSION_TIP = (
    "<p>The two steady precession rates at this tilt against the spin (solid), and the fast top's estimates "
    "mgl/I₃ω₃ and I₃ω₃/I₁ cos θ (dashed). Under the least spin (dotted) the top can't precess steadily at this "
    "tilt. The dot is this spin's slow rate.</p>"
)
PATH_TIP = "<p>The end of the axis, seen from above, the axis's length 1. Dotted: the turning points' circles.</p>"
ENERGY_TIP = (
    "<p>E<sub>J</sub>(θ), the least energy the top can have at each tilt with its Jellett's J, which friction can't "
    "change (solid), and its energy against its tilt as it runs (thin), down to the dot, now. Friction takes energy "
    "out, so it sinks onto the curve, toward its lowest point.</p>"
)
TRACK_TIP = (
    "<p>The tip's track on the table (solid, mm) and its centre of mass's (dotted), seen from above, the last 10 s.</p>"
)
LEDGER_TIP = (
    "<p>The energy and the work friction has done, together (constant but for the step's error), and Jellett's J "
    "(constant whatever the contact force), each against what it was let go with.</p>"
)
SECTION_TIP = (
    "<p>A Poincaré section: each time the tilt peaks, a dot at the spin angle ψ (from the line of nodes, mod 180°) "
    "and the spin ω₃. A symmetric top keeps ω₃: a line. Regular motion draws closed curves; chaos, a scatter.</p>"
)
APART_TIP = (
    "<p>How far apart (rad, log scale) the axes of two tops let go 10⁻⁸ rad apart have got. Straight up: chaos.</p>"
)
TABLE_TILT_TIP = "<p>The axis's tilt from the vertical as it runs, the last minute.</p>"
SNAPSHOT_TIP = (
    "<p>The equations of motion at one instant, in the axes picked, with the numbers: the rotation matrices that "
    "place the body, its inertia tensor in these axes, L = Iω, and dL/dt + Ω × L = τ. While the top runs it is "
    "the latest instant; pause it and drag the line across the plot to see any other.</p>"
)
COMPONENTS_TIP = (
    "<p>The components of ω, L, τ or I in the axes picked, as the top runs; for a vector, its size dashed (the "
    "same in every frame). The text says what to look for.</p>"
)
CHECK_TIP = (
    "<p>How far the simulation's energy has strayed (over mgl), and its two momenta's (over I₃ω₃: round-off "
    "only). The text compares what it has run with the closed forms.</p>"
)

NOTES_HTML = load_notes("gyroscope", figures=gyroscope_figures.FIGURES, trials=trials_html(TRIALS))


def _section(axis: np.ndarray, mark: np.ndarray, momentum: np.ndarray, top: Top) -> tuple[np.ndarray, np.ndarray]:
    """The Poincaré section's coordinates where the tilt peaks: the spin angle ψ (°, mod 180, a rectangle's
    period), the mark's angle from the line of nodes n = ẑ × e₃, and ω₃ (rpm). With L·ẑ and the energy kept and φ
    left out, that is all the state there is: θ̇ = 0 there, and θ follows from the energy."""
    nodes = np.stack([-axis[:, 1], axis[:, 0], np.zeros(len(axis))], axis=1)
    nodes /= np.maximum(np.linalg.norm(nodes, axis=1), 1e-300)[:, None]
    psi = np.degrees(np.arctan2(np.sum(mark * np.cross(axis, nodes), 1), np.sum(mark * nodes, 1))) % 180.0
    return psi, np.sum(momentum * axis, 1) / top.polar / RPM


def _columns(s: SlideSamples) -> np.ndarray:
    """A table's samples as the history keeps them: `SlideSamples.of`'s columns after t."""
    return np.column_stack(
        [s.axis, s.mark, s.centre, s.contact, s.energy, s.work, s.jellett, s.normal, s.friction, s.momentum, s.slip]
    )


class GyroscopePage(QtWidgets.QWidget):
    """Controls on the left, the top and its tilt in the centre, the closed forms and the check on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def transport(self) -> Transport:
        """The run bar drives the top's run and its playback speed."""
        return Transport(self.pause_button, self.restart_button, self.playback)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.motion = TRIALS[0].motion
        self.round: RoundTop | None = None  # on a table
        self.sim: TopSimulator | SlidingSimulator = TopSimulator(self.motion)
        self.history = SampleBuffer(10, HISTORY)  # axis, mark, energy, L
        self.slides = SampleBuffer(COLUMNS - 1, 1)  # on a table: a sample's columns after t (made full size there)
        self._track_span: tuple[float, float, float] | None = None  # the Path tab's view on a table: x, y, half
        self._launch_energy = self._launch_jellett = 0.0  # on a table, let go
        self.twin: TopSimulator | None = None  # in the cup: the same top let go TWIN further over
        self.section = SampleBuffer(2, SECTION_POINTS)  # where the tilt peaks: ψ (°, mod 180) and ω₃ (rpm)
        self.apart = SampleBuffer(1, HISTORY // APART_STRIDE)  # how far apart the two tops' axes are
        self._peaks = (math.nan, math.nan)  # the last two samples' e₃·ẑ, to find a peak across two frames
        self.running = False  # held at the launch until started
        self._frames = 0

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Each sets up the top and lets it go, to show one idea from the Theory tab.</p>",
            predictions=[t.predict for t in TRIALS],
        )
        self.trials.picked.connect(self.load_trial)

        body = Form("Top")
        self.body_form = body.form
        self.setting = combo([setting.value for setting in SETTINGS], TIPS["setting"])
        body.add("Setting:", self.setting, TIPS["setting"])
        self.shape = combo([shape.value for shape in SHAPES], TIPS["shape"])
        body.add("Shape:", self.shape, TIPS["shape"])
        self.mass = body.number("Mass m:", 0.001, 100.0, 0.5, 3, " kg", TIPS["mass"])
        self.radius = body.number("Disc radius r:", 1.0, 500.0, 40.0, 2, " mm", TIPS["radius"])
        self.radius_label = body.label(self.radius)
        self.breadth = body.number("Breadth b:", 1.0, 500.0, 40.0, 2, " mm", TIPS["breadth"])
        self.top_height = body.number("Height l:", 1.0, 500.0, 30.0, 1, " mm", TIPS["height"])
        self.tip_radius = body.number("Tip radius R:", 0.5, 100.0, 10.0, 1, " mm", TIPS["tip_radius"])
        self.low = body.number("Weight below centre:", 0.0, 100.0, 4.5, 2, " mm", TIPS["low"])
        self.ratio = body.number("I₁/I₃:", 0.1, 3.0, 0.8, 2, "", TIPS["ratio"])
        self.friction = body.number("Friction μ:", 0.0, 2.0, 0.3, 3, "", TIPS["friction"])
        self.gravity = body.number("Gravity g:", 0.0, 100.0, 9.81, 2, " m/s²", TIPS["gravity"])
        self.info = note_label()
        body.row(self.info)

        release = Form("Let go")
        self.release_form = release.form
        self.spin = release.number("Spin ω₃:", -20000.0, 20000.0, 1500.0, 0, " rpm", TIPS["spin"])
        self.tilt = release.number("Tilt θ₀:", 0.1, 179.0, 30.0, 1, "°", TIPS["tilt"])
        self.launch = combo([launch.value for launch in LAUNCHES], TIPS["launch"])
        release.add("Precession:", self.launch, TIPS["launch"])
        self.rate = release.number("φ̇₀:", -500.0, 500.0, 0.0, 2, " rad/s", TIPS["precession"])
        self.launch_info = note_label()
        release.row(self.launch_info)

        play = Form("Run")
        self.playback = combo([f"{p:g}×" for p in PLAYBACK], TIPS["playback"])
        self.playback.setCurrentIndex(REAL_TIME)
        play.add("Playback:", self.playback, TIPS["playback"])
        self.pause_button = RunButton()
        self.pause_button.toggled.connect(self._on_pause)
        self.restart_button = QtWidgets.QPushButton("Reset")
        self.restart_button.setToolTip("<p>Back to the launch, held until started.</p>")
        self.restart_button.clicked.connect(self.reset)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.restart_button)
        play.row(buttons)
        self.vectors = QtWidgets.QCheckBox("Vectors L, ω, τ")
        self.vectors.setChecked(True)
        self.vectors.toggled.connect(self._on_vectors)
        play.row(self.vectors)
        self.axes_shown = combo(AXES_SHOWN, TIPS["axes"])
        self.axes_shown.currentIndexChanged.connect(self._on_axes_shown)
        play.add("Axes:", self.axes_shown, TIPS["axes"])

        connect_all(
            (
                self.setting,
                self.shape,
                self.mass,
                self.radius,
                self.breadth,
                self.top_height,
                self.tip_radius,
                self.low,
                self.ratio,
                self.friction,
                self.gravity,
                self.spin,
                self.tilt,
                self.launch,
                self.rate,
            ),
            self._on_inputs,
        )

        # --- centre
        self.view = TopView()
        self.tilt_plot = plot("Tilt θ", None, "Time", "s", height=130)
        self.tilt_plot.setToolTip(TILT_TIP)
        self.tilt_item = pg.PlotDataItem()
        self.turn_lines = [pg.InfiniteLine(angle=0), pg.InfiniteLine(angle=0)]
        for item in (*self.turn_lines, self.tilt_item):
            self.tilt_plot.addItem(item)
        self.tilt_plot.setLabel("left", "Tilt θ (°)")
        self.readout = readout_label()

        # --- right
        self.potential_plot = plot("U", "J", "Tilt θ", None, height=200)
        self.potential_plot.setToolTip(POTENTIAL_TIP)
        self.potential_item = pg.PlotDataItem()
        self.energy_line = pg.InfiniteLine(angle=0)
        self.potential_turns = [pg.InfiniteLine(angle=90), pg.InfiniteLine(angle=90)]
        self.potential_now = Dot(10)
        self.potential_trace = pg.PlotDataItem()  # on a table: the energy against the tilt as it runs
        for item in (*self.potential_turns, self.energy_line, self.potential_trace, self.potential_item):
            self.potential_plot.addItem(item)
        self.potential_plot.addItem(self.potential_now, ignoreBounds=True)
        self.potential_plot.setLabel("bottom", "Tilt θ (°)")
        self.potential_note = note_label()

        self.precession_plot = plot("Precession rate", "rad/s", "Spin", "rpm", log_y=True, height=200)
        self.precession_plot.setToolTip(PRECESSION_TIP)
        add_legend(self.precession_plot.getPlotItem(), offset=(-10, 10))
        self.slow_item = pg.PlotDataItem(name="slow")
        self.fast_item = pg.PlotDataItem(name="fast")
        self.slow_guess = pg.PlotDataItem(name="mgl/I₃ω₃")
        self.fast_guess = pg.PlotDataItem(name="I₃ω₃/I₁ cos θ")
        self.least_line = pg.InfiniteLine(angle=90)
        self.precession_now = marker(11)
        for item in (self.least_line, self.slow_guess, self.fast_guess, self.slow_item, self.fast_item):
            self.precession_plot.addItem(item)
        self.precession_plot.addItem(self.precession_now)
        self.precession_note = note_label()

        self.path_plot = plot("", None, "", None, height=260)
        self.path_plot.setToolTip(PATH_TIP)
        self.path_plot.setAccessibleName("The end of the axis, seen from above")
        self.path_plot.setAspectLocked(True)
        self.path_circles = pg.PlotDataItem(connect="finite")
        self.path_item = pg.PlotDataItem()
        self.path_now = Dot(8)
        for item in (self.path_circles, self.path_item):
            self.path_plot.addItem(item)
        self.path_plot.addItem(self.path_now, ignoreBounds=True)
        self.path_note = note_label()

        self.check_plot = plot("Error", None, "Time", "s", height=200)
        self.check_plot.setToolTip(CHECK_TIP)
        add_legend(self.check_plot.getPlotItem(), offset=(10, 10))
        self.energy_item = pg.PlotDataItem(name="energy, ΔE/mgl")
        self.momentum_item = pg.PlotDataItem(name="momenta, ΔL/I₃ω₃")
        for item in (self.energy_item, self.momentum_item):
            self.check_plot.addItem(item)
        self.check_legend = self.check_plot.getPlotItem().legend
        self.check_note = note_label()

        self.components = ComponentsTab(
            [f.value for f in FRAMES],
            {"show": TIPS["comp_show"], "frame": TIPS["comp_frame"], "span": TIPS["comp_span"]},
            SNAPSHOT_TIP,
            COMPONENTS_TIP,
        )
        self.components.changed.connect(self._draw_components)
        self.components.dragged.connect(self._on_snapshot_dragged)
        self._snap_t: float | None = None  # the instant dragged to; None: the latest

        self.section_plot = plot("ω₃", "rpm", "ψ at each peak of the tilt", None, height=220)
        self.section_plot.setToolTip(SECTION_TIP)
        self.section_plot.setLabel("bottom", "ψ at each peak of the tilt (°, mod 180)")
        self.section_plot.getAxis("left").enableAutoSIPrefix(False)
        self.section_item = marker(3)
        self.section_plot.addItem(self.section_item)
        self.section_plot.setXRange(0, 180, padding=0)
        self.apart_plot = plot("Apart", None, "Time", "s", log_y=True, height=150)
        self.apart_plot.setToolTip(APART_TIP)
        self.apart_item = pg.PlotDataItem()
        self.apart_plot.addItem(self.apart_item)
        self.chaos_note = LiveLabel(wrap=True)  # the count of peaks grows as it runs (#409 P-2)
        mute(self.chaos_note)

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(column(self.potential_plot, self.potential_note, stretch=False), "Potential")
        self.tabs.addTab(column(self.precession_plot, self.precession_note, stretch=False), "Precession")
        self.tabs.addTab(column(self.path_plot, self.path_note, stretch=False), "Path")
        self.tabs.addTab(self.components, "Components")
        self.tabs.addTab(column(self.check_plot, self.check_note, stretch=False), "Check")
        self.tabs.addTab(column(self.section_plot, self.apart_plot, self.chaos_note, stretch=False), "Chaos")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._draw_tab)

        three_panes(
            self,
            column(self.trials, body, release, play),
            stack((self.view, 4), (self.tilt_plot, 2), (self.readout, 0)),
            self.tabs,
            sizes=TAB_ROOM,
        )

        self.loop = FrameLoop(self, "Gyroscope", self._tick, self._runs, self._sim_t)
        self.tabs.setCurrentIndex(TAB_PATH)
        self._on_inputs()
        self.apply_theme()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(GyroTrial("", "", **setup))

    def practice_state(self) -> GyroTrial:
        """The page's settings, as a trial: what a practice spec checks."""
        return GyroTrial(
            "",
            "",
            mass=self.mass.value(),
            radius=self.radius.value() / 1000,
            height=self.top_height.value() / 1000,
            gravity=self.gravity.value(),
            setting=SETTINGS[self.setting.currentIndex()],
            shape=SHAPES[self.shape.currentIndex()],
            breadth=self.breadth.value() / 1000,
            tip_radius=self.tip_radius.value() / 1000,
            friction=self.friction.value(),
            low=self.low.value() / 1000,
            ratio=self.ratio.value(),
            spin_rpm=self.spin.value(),
            tilt=self.tilt.value(),
            launch=LAUNCHES[self.launch.currentIndex()],
            precession=self.rate.value(),
            playback=self.playback.currentIndex(),
            tab=self.tabs.currentIndex(),
        )

    def apply_trial(self, t: GyroTrial) -> None:
        for box, value in (
            (self.setting, SETTINGS.index(t.setting)),
            (self.shape, SHAPES.index(t.shape)),
            (self.breadth, t.breadth * 1000),
            (self.tip_radius, t.tip_radius * 1000),
            (self.friction, t.friction),
            (self.low, t.low * 1000),
            (self.ratio, t.ratio),
            (self.mass, t.mass),
            (self.radius, t.radius * 1000),
            (self.top_height, t.height * 1000),
            (self.gravity, t.gravity),
            (self.spin, t.spin_rpm),
            (self.tilt, t.tilt),
            (self.launch, LAUNCHES.index(t.launch)),
            (self.rate, t.precession),
            (self.playback, t.playback),
        ):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self.running = True  # a trial lets the top go
        self.pause_button.set_running(True)
        self._on_inputs()

    def _on_inputs(self) -> None:
        state = self.practice_state()
        if state.setting is not Setting.CUP and state.shape is Shape.RECTANGLE:  # a symmetric top only, on a table
            set_quietly(self.shape, SHAPES.index(Shape.DISC))
            state = self.practice_state()
        self._show_rows(state.setting, state.shape)
        self.rate.setEnabled(state.launch is Launch.SET)
        self.motion = state.motion
        self.round = state.round_top
        if self.round is not None:
            self._on_table(state)
            self.restart()
            return
        top = self.motion.top
        if not top.symmetric:
            self._on_skewed(state)
            self.restart()
            return
        self.info.setText(
            (
                f"I₁ = {top.transverse * 1e4:.4g} kg·cm², I₃ = {top.polar * 1e4:.4g} kg·cm², mgl = "
                f"{fmt(top.weight_moment, 'N·m')}. It sleeps above {top.sleeping_spin / RPM:.4g} rpm."
                if top.weight_moment > 0
                else f"I₁ = {top.transverse * 1e4:.4g} kg·cm², I₃ = {top.polar * 1e4:.4g} kg·cm²; no gravity: a free top."
            )
            + (
                f" A disc of radius s/√3 = {top.radius * 1000 / math.sqrt(3):.4g} mm would move the same."
                if top.shape is Shape.SQUARE
                else ""
            )
        )
        slow, fast = steady_precession(top, self.motion.spin, self.motion.tilt)
        if state.launch in (Launch.SLOW, Launch.FAST) and not math.isfinite(
            slow if state.launch is Launch.SLOW else fast
        ):
            least = least_spin(top, self.motion.tilt) / RPM
            self.launch_info.setText(
                f"No such steady precession at this tilt under {least:.4g} rpm: let go at rest instead."
            )
        else:
            rates = ", ".join(f"{r:.4g}" for r in (slow, fast) if math.isfinite(r))
            self.launch_info.setText(
                f"φ̇₀ = {self.motion.precession:.4g} rad/s. Steady rates at this tilt: "
                + (f"{rates} rad/s." if rates else "none.")
            )
        self.restart()

    def _show_rows(self, setting: Setting, shape: Shape) -> None:
        """Show the controls this setting uses."""
        cup, ball = setting is Setting.CUP, setting is Setting.TIPPE
        box: QtWidgets.QWidget
        for box, shown in (
            (self.top_height, not ball),
            (self.shape, not ball),
            (self.tip_radius, setting is Setting.TABLE),
            (self.low, ball),
            (self.ratio, ball),
            (self.friction, not cup),
        ):
            self.body_form.setRowVisible(box, shown)
        for box in (self.launch, self.rate, self.launch_info):
            self.release_form.setRowVisible(box, cup)
        rectangle = shape is Shape.RECTANGLE and cup
        self.body_form.setRowVisible(self.breadth, rectangle)
        shapes = self.shape.model()
        assert isinstance(shapes, QtGui.QStandardItemModel), "a combo box's own model"
        shapes.item(SHAPES.index(Shape.RECTANGLE)).setEnabled(cup)
        label = self.radius_label
        square = shape is Shape.SQUARE and not ball
        label.setText(
            "Ball radius R:" if ball else "Side s:" if square else "Side w:" if rectangle else "Disc radius r:"
        )
        self.radius.setToolTip(
            TIPS["ball"] if ball else TIPS["side"] if square else TIPS["side_a"] if rectangle else TIPS["radius"]
        )
        label.setToolTip(self.radius.toolTip())
        self.vectors.setText("Vectors L, ω, τ" if cup else "Vectors L, ω, F")
        skewed = rectangle and self.breadth.value() != self.radius.value()
        for tab, on in (
            (TAB_POTENTIAL, not skewed),
            (TAB_PRECESSION, cup and not skewed),
            (TAB_CHAOS, cup),
        ):
            self.tabs.setTabEnabled(tab, on)
        if not self.tabs.isTabEnabled(self.tabs.currentIndex()):
            self.tabs.setCurrentIndex(TAB_PATH)
        for plot_, cup_tip, table_tip in (
            (self.potential_plot, POTENTIAL_TIP, ENERGY_TIP),
            (self.path_plot, PATH_TIP, TRACK_TIP),
            (self.check_plot, CHECK_TIP, LEDGER_TIP),
            (self.tilt_plot, TILT_TIP, TABLE_TILT_TIP),
        ):
            plot_.setToolTip(cup_tip if cup else table_tip)
        relabel(self.potential_plot, "left", "U" if cup else "E", "J")
        self.check_legend.clear()
        self.check_legend.addItem(self.energy_item, "energy, ΔE/mgl" if cup else "energy + friction's work, ΔE/E₀")
        self.check_legend.addItem(
            self.momentum_item,
            "Jellett's J, ΔJ/J₀" if not cup else "L·ẑ, ΔL/I₃ω₃" if skewed else "momenta, ΔL/I₃ω₃",
        )

    def _on_skewed(self, state: GyroTrial) -> None:
        """The info lines for an asymmetric top."""
        top = self.motion.top
        i1, i2, i3 = (v * 1e4 for v in top.moments)
        self.info.setText(
            f"I₁ = {i1:.4g}, I₂ = {i2:.4g}, I₃ = {i3:.4g} kg·cm²"
            + (f", mgl = {fmt(top.weight_moment, 'N·m')}." if top.weight_moment > 0 else "; no gravity: a free top.")
        )
        self.launch_info.setText(
            f"φ̇₀ = {self.motion.precession:.4g} rad/s. An asymmetric top has no steady precession to start at"
            + (": let go at rest." if state.launch in (Launch.SLOW, Launch.FAST) else ".")
        )

    def _on_table(self, state: GyroTrial) -> None:
        """The info lines for a top on a table."""
        rt = self.round
        assert rt is not None  # only called on a table
        self.info.setText(
            f"I₁ = {rt.central * 1e4:.4g} kg·cm² (about G), I₃ = {rt.polar * 1e4:.4g} kg·cm²; its centre of mass "
            + (
                f"{rt.offset * 1000:.3g} mm above the round tip's centre."
                if rt.offset >= 0
                else f"{-rt.offset * 1000:.3g} mm below the ball's centre: α = {rt.alpha:.3g}."
            )
        )

    def restart(self) -> None:
        """Let the top go again from the launch set (if it is running; held there if not)."""
        self._snap_t = None
        self._track_span = None
        if self.round is not None:
            self.sim = SlidingSimulator(self.round, self.motion.spin, self.motion.tilt)
            if self.slides.t.size < SLIDE_HISTORY:  # first time on a table: its 34 MB only once it's needed
                self.slides = SampleBuffer(COLUMNS - 1, SLIDE_HISTORY)
            self.slides.clear()
            first = self.sim.sample()
            self.slides.extend(first.t, _columns(first))
            self._launch_energy, self._launch_jellett = float(first.energy[0]), float(first.jellett[0])
            self.view.set_top(self.motion.top, (), self.round)
            for line in self.turn_lines:
                line.hide()
            self._show_table_forms()
            self._draw_tab()
            self._draw_now()
            self._update_buttons()
            self.loop.start()
            return
        m = self.motion
        self.sim = TopSimulator(m)
        self.twin = TopSimulator(Motion(m.top, m.spin, m.tilt + TWIN, m.precession))
        self.history.clear()
        self.section.clear()
        self.apart.clear()
        self._peaks = (math.nan, math.nan)
        skewed = not m.top.symmetric
        self.view.set_top(m.top, () if skewed else m.turning())
        for line, th in zip(self.turn_lines, m.turning()):
            line.setPos(math.degrees(th))
            line.setVisible(not skewed)
        if skewed:
            self._show_skewed_forms()
        else:
            self._show_closed_forms()
        self._draw_tab()
        self._draw_now()
        self._update_buttons()
        self.loop.start()

    def _on_vectors(self, on: bool) -> None:
        self.view.show_vectors = on
        self._draw_now()

    def _on_axes_shown(self, index: int) -> None:
        self.view.show_axes = index
        self._draw_now()

    def _on_pause(self, on: bool) -> None:
        self.running = on
        if not on and self.tabs.currentIndex() == TAB_COMPONENTS:
            self._draw_snapshot()  # the instant it stopped at (while running it is rewritten every few frames)
        self.loop.start()

    def _update_buttons(self) -> None:
        self.pause_button.set_running(self.running)

    def reset(self) -> None:
        """Back to the launch, held there until started."""
        self.running = False
        self.pause_button.rest()
        self.restart()

    # ------------------------------------------------------------- the loop

    def _runs(self) -> bool:
        return self.running

    def _sim_t(self) -> float:
        return self.sim.t

    def _tick(self, dt: float) -> None:
        self.step(dt * PLAYBACK[self.playback.currentIndex()])

    def frame_context(self) -> str:
        launch = self.launch.currentText() if self.round is None else self.setting.currentText()
        return f"{self.spin.value():g} rpm, {self.tilt.value():g}°, {launch}"

    def step(self, duration: float) -> None:
        """Run the top on by `duration` s of simulated time and redraw."""
        if isinstance(self.sim, SlidingSimulator):
            stopped = self.sim.event is not None
            slide = self.sim.advance(duration)
            if slide.t.size:
                self.slides.extend(slide.t, _columns(slide))
            if self.sim.event is not None and not stopped:
                self._show_table_forms()
        else:
            s = self.sim.advance(duration)
            if s.t.size:
                self.history.extend(s.t, np.concatenate([s.axis, s.mark, s.energy[:, None], s.momentum], axis=1))
                self._record_chaos(s)
        self._draw_now()
        self._draw_live_tab()

    def _record_chaos(self, s) -> None:
        """The Chaos tab's records: the twin's separation, and the Poincaré section's points (where e₃·ẑ is least:
        the tilt peaks, θ̇ = 0), each kept whether or not the tab shows."""
        assert self.twin is not None  # set with the cup's simulator
        other = self.twin.advance(float(s.t[-1]) - self.twin.t)
        n = min(len(other.t), len(s.t))
        k = slice(0, n, APART_STRIDE)
        self.apart.extend(s.t[k], np.linalg.norm(s.axis[k] - other.axis[k], axis=1)[:, None])
        ez = np.concatenate([self._peaks, s.axis[:, 2]])
        self._peaks = (float(ez[-2]), float(ez[-1]))
        j = np.flatnonzero((ez[1:-1] < ez[:-2]) & (ez[1:-1] <= ez[2:])) + 1 - 2  # in this chunk
        j = j[j >= 0]
        if j.size:
            psi, spin = _section(s.axis[j], s.mark[j], s.momentum[j], self.motion.top)
            self.section.extend(s.t[j], np.column_stack([psi, spin]))

    def table_samples(self, seconds: float | None = None) -> SlideSamples:
        """The samples kept on a table: the last `seconds` of them, or all."""
        n = self.slides.size
        if seconds is None:
            t, y = self.slides.t[:n], self.slides.y[:n]
        else:
            t, y = self.slides.window(seconds)
        return SlideSamples.of(np.column_stack([t, y]))

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.view.apply_theme()
        self.tilt_item.setPen(pg.mkPen(c.mass[0], width=2))
        for line in (*self.turn_lines, *self.potential_turns):
            line.setPen(pg.mkPen(c.fit, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.potential_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.potential_trace.setPen(pg.mkPen(c.mass[1], width=1))
        self.energy_line.setPen(pg.mkPen(c.force, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.potential_now.setBrush(pg.mkBrush(c.strong))
        self.potential_now.setPen(pg.mkPen(None))
        self.slow_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.fast_item.setPen(pg.mkPen(c.mass[1], width=2))
        self.slow_guess.setPen(pg.mkPen(c.mass[0], width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.fast_guess.setPen(pg.mkPen(c.mass[1], width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.least_line.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DotLine))
        mark(self.precession_now, c.strong, filled=True)
        self.path_circles.setPen(pg.mkPen(c.fit, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.path_item.setPen(pg.mkPen(c.mass[1], width=1.5))
        self.path_now.setBrush(pg.mkBrush(c.strong))
        self.path_now.setPen(pg.mkPen(None))
        # 1 px: the errors zigzag at every step, and Qt strokes a wider antialiased zigzag ~100x slower (the Check
        # tab painted in 140 ms a frame at 2 px and 1.5 px, 1.5 ms at 1 px).
        self.energy_item.setPen(pg.mkPen(c.mass[0], width=1))
        self.momentum_item.setPen(pg.mkPen(c.grey, width=1))
        self.components.apply_theme()
        mark(self.section_item, c.mass[0], filled=True)
        self.apart_item.setPen(pg.mkPen(c.mass[1], width=1.5))
        self._draw_tab()
        self._draw_now()

    def _show_closed_forms(self) -> None:
        m = self.motion
        hi, lo = (math.degrees(v) for v in m.turning())
        mean = m.mean_precession()
        if abs(mean) > 1e-9:
            prec = f"precesses at {mean:.4g} rad/s on average, a turn in {2 * math.pi / abs(mean):.4g} s"
        else:
            prec = "doesn't precess on average"
        if m.steady:
            nod = f"It holds at {hi:.4g}°: steady precession at {mean:.4g} rad/s."
        else:
            nod = (
                f"It nods between {hi:.4g}° and {lo:.4g}°, {m.nutation_hz():.4g} times a second, and {prec}. "
                f"Path: {m.path().value}."
            )
        self.readout.setText(nod)

    def _launch_rates(self) -> np.ndarray:
        """ω in the body's axes when let go: φ̇₀ about the vertical, which leans θ₀ from the axis about the body's
        x axis, and ω₃ about the axis."""
        m = self.motion
        return np.array([0.0, m.precession * math.sin(m.tilt), m.spin])

    def _show_skewed_forms(self) -> None:
        """The readout for an asymmetric top: what's lost, and what's left in closed form."""
        m = self.motion
        top = m.top
        i1, i2, i3 = top.moments
        middle = min(i1, i2) < i3 < max(i1, i2)
        parts = [
            "I₁ ≠ I₂: an asymmetric top. ω₃ isn't kept, so there's no effective potential for the tilt, no turning "
            "points and no nutation period in closed form: only the energy and L·ẑ are kept."
        ]
        if top.weight_moment == 0:
            free = FreeSpin.of(top, self._launch_rates())
            parts.append(
                "Without gravity Euler–Poinsot's solution holds: ω in the body goes as Jacobi's cn, sn and dn "
                + (f"(k² = {free.k2:.4g})" if free.k2 < 0.999 else f"(1 − k² = {1 - free.k2:.2g})")
                + f", back every {free.period:.4g} s"
                + (f"; spun about its middle axis, it flips over every {0.5 * free.period:.4g} s." if middle else ".")
            )
        elif middle:
            parts.append("Spun about its middle axis, with gravity, it tumbles: chaos (the Chaos tab).")
        else:
            parts.append("With gravity no closed form is left at all: the Chaos tab shows whether it stays regular.")
        self.readout.setText(" ".join(parts))

    def _show_table_forms(self) -> None:
        """The readout on a table: Jellett's constant and where it takes the top; why it stopped, if it did."""
        rt = self.round
        assert rt is not None  # only on a table
        m = self.motion
        J = rt.jellett(m.spin, m.tilt)
        least = math.degrees(rt.least_tilt(J))
        parts = [
            f"Jellett's J = I₃ω₃(a + R cos θ₀) = {J * 1e4:.3g} kg·cm²/s stays put while friction takes energy out, so "
            f"it heads for E_J's least, at {least:.3g}°."
        ]
        turn = rt.turning_spin()
        if rt.offset > 0:
            if math.isfinite(turn):
                parts.append(
                    f"Upright is least above {turn / RPM:.4g} rpm, where J would leave it sleeping at "
                    f"{rt.end_spin(J) / RPM:.4g} rpm."
                )
            roll = rt.rolling_tilt()
            if roll < 0.5 * math.pi:
                parts.append(
                    f"But its tip can roll instead of slide below about {math.degrees(roll):.3g}° (sin θ = "
                    "μI₃/maR), and it stops rising there."
                )
            else:
                parts.append("Its friction can make its tip roll at any tilt: it travels instead of rising.")
        elif rt.inverts:
            parts.append(
                f"With 1 − α = {1 - rt.alpha:.3g} &lt; I₁/I₃ = {rt.ratio:.3g} &lt; 1 + α = {1 + rt.alpha:.3g}, "
                f"turned over is least above {turn / RPM:.4g} rpm: it would end spinning at "
                f"{rt.end_spin(J, upright=False) / RPM:.4g} rpm, the other way about its own axis."
            )
        elif rt.offset < 0:
            parts.append(
                f"I₁/I₃ = {rt.ratio:.3g} lies outside 1 ± α ({1 - rt.alpha:.3g} to {1 + rt.alpha:.3g}): turned over is "
                "no minimum, so it doesn't turn over."
            )
        sim = self.sim
        if isinstance(sim, SlidingSimulator) and sim.event is not None:
            parts.append(f"<b>Stopped at {sim.t:.3g} s: {sim.event.value}.</b>")
        self.readout.setText(" ".join(parts))
        hi = 180.0 if self.view.ball else min(math.degrees(rt.rim_tilt()) + 2.0, 180.0)
        self.tilt_plot.setYRange(0.0, hi, padding=0)

    def _draw_table_now(self) -> None:
        s = self.table_samples(TRACK_SPAN)
        i0 = int(np.searchsorted(s.t, s.t[-1] - TABLE_TRAIL))
        trail = slice(i0, None, max(1, -(-(s.t.size - i0) // TRAIL_POINTS)))
        self.view.draw(
            s.axis[-1],
            s.mark[-1],
            s.momentum[-1],
            s.axis[trail],
            s.centre[-1],
            s.centre[trail],
            s.contact[trail],
            s.friction[-1],
        )
        t, y = self.slides.window(SLIDE_TILT_SPAN)
        k = thin(t.size, 2000)
        self.tilt_item.setData(t[k], np.degrees(np.arccos(np.clip(y[k, 2], -1, 1))))
        t1 = max(float(t[-1]), 10.0)
        self.tilt_plot.setXRange(max(0.0, t1 - SLIDE_TILT_SPAN), t1, padding=0)

    def _draw_now(self) -> None:
        if self.round is not None:
            self._draw_table_now()
            return
        n = self.history.size
        if n:
            last = self.history.y[n - 1]
            axis, mark_, momentum = last[0:3], last[3:6], last[7:10]
        else:
            th = self.motion.tilt
            axis = np.array([0.0, -math.sin(th), math.cos(th)])
            mark_ = np.array([1.0, 0.0, 0.0])
            momentum = self.motion.angular_momentum()
        trail = np.zeros((0, 3))
        skewed = not self.motion.top.symmetric
        if n:
            mean = abs(self.motion.mean_precession())
            span = min(2 * math.pi / mean, 20.0) if mean > 1e-9 else 3 * self.motion.nutation_period()
            t, y = self.history.window(SKEW_TRAIL if skewed else span)
            trail = y[thin(t.size, TRAIL_POINTS), 0:3]
        self.view.draw(axis, mark_, momentum, trail)
        if skewed:
            self._skewed_tilt()
            return
        if n:
            window = min(NODS * self.motion.nutation_period(), 20.0)
            t, y = self.history.window(window)
            k = thin(t.size, 2000)
            self.tilt_item.setData(t[k], np.degrees(np.arccos(np.clip(y[k, 2], -1, 1))))
            t1 = max(float(t[-1]), window)
            self.tilt_plot.setXRange(t1 - window, t1, padding=0)
        else:
            self.tilt_item.setData([], [])
            self.tilt_plot.setXRange(0, min(NODS * self.motion.nutation_period(), 20.0), padding=0)
        hi, lo = (math.degrees(v) for v in self.motion.turning())
        pad = max(0.15 * (lo - hi), 0.05 * max(hi, 1.0), 0.05)
        self.tilt_plot.setYRange(hi - pad, lo + pad, padding=0)

    def _skewed_tilt(self) -> None:
        """An asymmetric top's tilt, the last few seconds, ranged to what it has done (no turning points)."""
        t, y = self.history.window(SKEW_TILT)
        if not t.size:
            self.tilt_item.setData([], [])
            return
        k = thin(t.size, 2000)
        theta = np.degrees(np.arccos(np.clip(y[k, 2], -1, 1)))
        self.tilt_item.setData(t[k], theta)
        t1 = max(float(t[-1]), SKEW_TILT)
        self.tilt_plot.setXRange(t1 - SKEW_TILT, t1, padding=0)
        lo, hi = float(theta.min()), float(theta.max())
        pad = max(0.1 * (hi - lo), 0.5)
        self.tilt_plot.setYRange(max(lo - pad, 0.0), min(hi + pad, 180.0), padding=0)

    @coalesce.coalesced
    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_CHAOS:
            self._draw_chaos()
            return
        if self.round is not None:
            if i == TAB_POTENTIAL:
                self._draw_energy()
            elif i == TAB_PATH:
                self._draw_track()
            elif i == TAB_CHECK:
                self._draw_ledger()
            elif i == TAB_COMPONENTS:
                self._draw_components()
            return
        if i == TAB_POTENTIAL:
            self._draw_potential()
        elif i == TAB_PRECESSION:
            self._draw_precession()
        elif i == TAB_PATH:
            self._draw_path()
        elif i == TAB_CHECK:
            self._draw_check()
        elif i == TAB_COMPONENTS:
            self._draw_components()

    def _draw_live_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_COMPONENTS:
            self._frames += 1
            self._draw_components()
            return
        if self.round is not None:
            if i == TAB_POTENTIAL:
                self._energy_dot()
            elif i == TAB_PATH:
                self._track_trace()
            elif i == TAB_CHECK:
                self._frames += 1
                self._draw_ledger(text=self._frames % CHECK_FRAMES == 0)
            return
        if i == TAB_CHAOS:
            self._frames += 1
            if self._frames % SECTION_FRAMES == 0:
                self._draw_chaos()
        elif i == TAB_POTENTIAL:
            self._potential_dot()
        elif i == TAB_PATH:
            self._path_trace()
        elif i == TAB_CHECK:
            self._frames += 1
            self._draw_check(text=self._frames % CHECK_FRAMES == 0)

    # ------------------------------------------------------------- components
    def _component_samples(self):
        """(t, axis, mark, L, τ, moments) over the span picked, thinned to CHECK_POINTS; None before a run."""
        span = self.components.span()
        if self.round is None:
            buffer = self.history
        else:
            buffer = self.slides
        n = buffer.size
        if n < 2:
            return None
        t, y = buffer.t[:n], buffer.y[:n]
        first = 0 if span is None else int(np.searchsorted(t, t[-1] - span))
        k = slice(first, n, max(1, -(-(n - first) // CHECK_POINTS)))
        t, y = t[k], y[k]
        axis, mark = y[:, 0:3], y[:, 3:6]
        if self.round is None:
            top = self.motion.top
            return t, axis, mark, y[:, 7:10], gravity_torque(axis, top.weight_moment), top.moments
        rt = self.round
        torque = contact_torque(y[:, 6:9], y[:, 9:11], y[:, 14], y[:, 15:17])
        return t, axis, mark, y[:, 17:20], torque, (rt.central, rt.central, rt.polar)

    def _draw_components(self) -> None:
        tab = self.components
        show = tab.show_box.currentIndex()
        frame = FRAMES[tab.frame_box.currentIndex()]
        data = self._component_samples()
        if data is None:
            tab.clear("Start the top to see its components.")
            return
        t, axis, mark, momentum, torque, moments = data
        axes = frame_axes(axis, mark, frame)
        if show == SHOW_I:
            in_axes = inertia_in(axis, mark, moments, axes)
        else:
            vector = (
                angular_velocity(axis, mark, momentum, moments)
                if show == SHOW_W
                else momentum
                if show == SHOW_L
                else torque
            )
            in_axes = in_frame(vector, axes)
        tab.draw(t, series(show, AXIS_NAMES[frame], in_axes), self._components_note(show, frame))
        if not self.running or self._frames % SNAPSHOT_FRAMES == 0:
            self._draw_snapshot()

    def snapshot_now(self):
        """The `Snapshot` the tab shows: at the instant dragged to (paused), else the latest with two samples after
        it; None before a run. Only the five samples around it are read (the torque is worked out per sample)."""
        buffer = self.history if self.round is None else self.slides
        n = buffer.size
        if n < 2:
            return None
        t = buffer.t[:n]
        if self._snap_t is None or self.running:
            k = max(n - 3, 0)
        else:
            k = int(min(np.searchsorted(t, self._snap_t), n - 1))
        near = slice(max(k - 2, 0), min(k + 3, n))
        t, y = t[near], buffer.y[near]
        axis, mark = y[:, 0:3], y[:, 3:6]
        if self.round is None:
            top = self.motion.top
            momentum, torque, moments = y[:, 7:10], gravity_torque(axis, top.weight_moment), top.moments
        else:
            rt = self.round
            momentum, moments = y[:, 17:20], (rt.central, rt.central, rt.polar)
            torque = contact_torque(y[:, 6:9], y[:, 9:11], y[:, 14], y[:, 15:17])
        frame = FRAMES[self.components.frame_box.currentIndex()]
        return snapshot(t, axis, mark, momentum, torque, moments, frame, k - near.start)

    def _draw_snapshot(self) -> None:
        s = self.snapshot_now()
        if s is None:
            self.components.set_snapshot(None, "")
            return
        self.components.set_snapshot(
            s.t,
            snapshot_html(
                s,
                WHERE[s.kind],
                frame_note="AᵀB is a turn by ψ about e₃.",
                turning_note="precession and nodding, not the spin",
            ),
        )

    def _on_snapshot_dragged(self, t: float) -> None:
        self._snap_t = t
        self._draw_snapshot()

    def _components_note(self, show: int, frame: Frame) -> str:
        """What to look for in this plot."""
        table = self.round is not None
        symmetric = table or self.motion.top.symmetric
        free = not table and self.motion.top.weight_moment == 0
        if show == SHOW_I:
            return {
                Frame.BODY: "Constant: the mass doesn't move in the body's own axes, which is why Euler's equations "
                "are written in them. The products of inertia are zero: these are its principal axes.",
                Frame.SPACE: "In fixed axes I = R I<sub>body</sub> R<sup>T</sup> changes as the body turns, and "
                "products of inertia appear. "
                + (
                    "Here only the axis's tilt and precession change it: spinning about its own axis maps a "
                    "symmetric top onto itself."
                    if symmetric
                    else "The rectangle's spin changes it too: its two moments across the axis differ."
                ),
                Frame.AXIS: "Constant: the axis frame keeps up with the axis, and the spin about it maps a symmetric "
                "top onto itself."
                if symmetric
                else "It swings at twice the spin: the rectangle's two moments across the axis take turns.",
            }[frame]
        where = {
            Frame.SPACE: "In fixed axes, z up.",
            Frame.BODY: "In the body's axes (e₁ the mark, e₃ the axis), turning with the top, spin and all: the "
            "components swing at the spin rate.",
            Frame.AXIS: "In the axis frame (e₃ the axis, e₁ horizontal across it), which follows the axis but not "
            "the spin.",
        }[frame]
        if show == SHOW_W:
            what = (
                "ω = I<sup>−1</sup>L isn't along L, as I₃ ≠ I₁: compare L in the same axes. Its part along the "
                "axis, ω₃, is the spin."
            )
        elif show == SHOW_L:
            what = (
                "No torque: L stays fixed in space, and only its components in turning axes change."
                if free
                else "The table's forces at the contact turn L about G, and friction changes its size."
                if table
                else "dL/dt = τ. Gravity's moment is horizontal, so L's vertical part stays put while the rest turns "
                "with the precession."
            )
        else:
            what = (
                "The normal force and friction at the contact P, about G: (P − G) × F. Gravity acts at G and has none."
                if table
                else "Gravity's moment about the tip, mgl e₃ × (−ẑ): always horizontal and across the axis, so it "
                "has no part along the axis (ω₃ stays) or up (L<sub>z</sub> stays)."
            )
        return f"{what} {where}"

    # ------------------------------------------------------------- on a table
    def _draw_energy(self) -> None:
        rt = self.round
        assert rt is not None  # only on a table
        m = self.motion
        J = rt.jellett(m.spin, m.tilt)
        hi = rt.rim_tilt()
        theta = np.linspace(0.0, hi, 500)
        least = rt.least_energy(J, theta)
        self.potential_item.setData(np.degrees(theta), least)
        for line in (*self.potential_turns, self.energy_line):
            line.hide()
        start = self._launch_energy
        top, base = max(start, float(np.max(least))), float(np.min(least))
        pad = 0.08 * max(top - base, 1e-9 * abs(top))
        self.potential_plot.setXRange(0.0, math.degrees(hi), padding=0)
        self.potential_plot.setYRange(base - pad, top + pad, padding=0)
        where = math.degrees(rt.least_tilt(J))
        self.potential_note.setText(
            f"J = {J * 1e4:.3g} kg·cm²/s. E_J is least at {where:.3g}°: upright it is {fmt(float(least[0]), 'J')}"
            + (f", turned over {fmt(float(least[-1]), 'J')}." if self.view.ball else ".")
            + f" Let go, it has {fmt(start, 'J')}."
        )
        self._energy_dot()

    def _energy_dot(self) -> None:
        n = self.slides.size
        y = self.slides.y[:n]
        k = thin(n, 2000)
        theta = np.degrees(np.arccos(np.clip(y[k, 2], -1, 1)))
        self.potential_trace.setData(theta, y[k, 11])
        last = y[n - 1]
        self.potential_now.setData([math.degrees(math.acos(min(1.0, max(-1.0, float(last[2])))))], [float(last[11])])

    def _draw_track(self) -> None:
        self._track_span = None
        self.path_note.setText(
            f"From above, in mm, the last {TRACK_SPAN:g} s: where the tip has touched the table (solid) and where its "
            "centre of mass has been (dotted). Sliding, the tip scrawls; rolling, it runs round smooth curves."
        )
        self._track_trace()

    def _track_trace(self) -> None:
        s = self.table_samples(TRACK_SPAN)
        k = thin(s.t.size, TRACK_POINTS)
        tip, centre = 1000 * s.contact[k], 1000 * s.centre[k, :2]
        self.path_item.setData(tip[:, 0], tip[:, 1])
        self.path_circles.setData(centre[:, 0], centre[:, 1])
        self.path_now.setData([tip[-1, 0]], [tip[-1, 1]])
        both = np.concatenate([tip, centre])
        lo, hi = both.min(axis=0), both.max(axis=0)
        span = self._track_span
        if span is None or np.any(lo < np.array(span[:2]) - span[2]) or np.any(hi > np.array(span[:2]) + span[2]):
            mid = 0.5 * (lo + hi)
            half = max(0.65 * float(np.max(hi - lo)), 5.0)
            self._track_span = (float(mid[0]), float(mid[1]), half)
            self.path_plot.setRange(
                xRange=(mid[0] - half, mid[0] + half), yRange=(mid[1] - half, mid[1] + half), padding=0
            )

    def _draw_ledger(self, text: bool = True) -> None:
        rt = self.round
        assert rt is not None  # only on a table
        s = self.table_samples()
        start, J0 = self._launch_energy, self._launch_jellett
        k = thin(s.t.size, CHECK_POINTS)
        ledger = (s.energy + s.work - start) / start
        drift = (s.jellett - J0) / (abs(J0) or 1.0)
        self.energy_item.setData(s.t[k], ledger[k])
        self.momentum_item.setData(s.t[k], drift[k])
        if not text:
            return
        sim = self.sim
        assert isinstance(sim, SlidingSimulator)  # on a table
        axis, L = s.axis[-1], s.momentum[-1]
        along = float(L @ axis)
        spin = along / rt.polar
        above = L[2] / rt.central + (1 / rt.polar - 1 / rt.central) * along * axis[2]
        parts = [
            f"h = {sim.h * 1e3:.3g} ms. Jellett's J held to {np.max(np.abs(drift)):.2g} of itself, and the energy "
            f"with friction's work to {np.max(np.abs(ledger)):.2g}. Friction has taken {100 * s.work[-1] / start:.3g} % "
            f"of it. The table pushed up with at least {s.normal.min() / rt.weight:.3g} mg.",
            f"Now at {math.degrees(math.acos(min(1.0, max(-1.0, float(axis[2]))))):.3g}°, slipping at "
            f"{1000 * s.slip[-1]:.3g} mm/s, spinning at {spin / RPM:.4g} rpm about its axis ({above / RPM:.4g} rpm "
            "seen from above).",
            f"Upright, J would leave it {rt.end_spin(J0) / RPM:.4g} rpm"
            + (
                f"; turned over, {rt.end_spin(J0, upright=False) / RPM:.4g} rpm seen from above."
                if self.view.ball
                else "."
            ),
        ]
        if sim.event is not None:
            parts.append(f"Stopped at {sim.t:.3g} s: {sim.event.value}.")
        self.check_note.setText(" ".join(parts))

    # ------------------------------------------------------------- in the cup
    def _draw_potential(self) -> None:
        m = self.motion
        for line in (*self.potential_turns, self.energy_line):
            line.show()
        self.potential_trace.setData([], [])
        hi, lo = m.turning()
        span = max(lo - hi, math.radians(2.0))
        a, b = max(hi - 1.5 * span, math.radians(0.2)), min(lo + 1.5 * span, math.radians(179.8))
        theta = np.linspace(a, b, 600)
        U = m.effective_potential(theta)
        self.potential_item.setData(np.degrees(theta), U)
        self.energy_line.setPos(m.energy)
        for line, th in zip(self.potential_turns, (hi, lo)):
            line.setPos(math.degrees(th))
        inside = U[(theta >= hi - 0.5 * span) & (theta <= lo + 0.5 * span)]
        top = max(float(np.max(inside)) if inside.size else m.energy, m.energy)
        base = float(np.min(U))
        pad = 0.15 * max(top - base, 1e-9 * max(abs(top), 1e-12))
        self.potential_plot.setXRange(math.degrees(a), math.degrees(b), padding=0)
        self.potential_plot.setYRange(base - pad, top + pad, padding=0)
        self._potential_dot()
        top_ = m.top
        if top_.weight_moment > 0:
            sleeps = abs(m.spin) > top_.sleeping_spin
            self.potential_note.setText(
                f"E' = {fmt(m.energy, 'J')}. Upright it {'sleeps' if sleeps else 'would fall'}: "
                f"|ω₃| = {abs(m.spin) / RPM:.4g} rpm against the {top_.sleeping_spin / RPM:.4g} rpm that "
                "I₃ω₃ &gt; 2√(I₁mgl) asks."
            )
        else:
            self.potential_note.setText(f"E' = {fmt(m.energy, 'J')}. No gravity: U is the precession's energy alone.")

    def _potential_dot(self) -> None:
        n = self.history.size
        if not n:
            th = self.motion.tilt
        else:
            th = math.acos(min(1.0, max(-1.0, float(self.history.y[n - 1, 2]))))
        self.potential_now.setData([math.degrees(th)], [float(self.motion.effective_potential(th))])

    def _draw_precession(self) -> None:
        m = self.motion
        top, th = m.top, m.tilt
        here = abs(m.spin) / RPM
        least = least_spin(top, th) / RPM
        hi = max(4.0 * here, 3.0 * least, 100.0)
        rpm = np.geomspace(max(least * 1.0005, hi / 1e3), hi, 400)
        pairs = np.array([steady_precession(top, r * RPM, th) for r in rpm])
        slow, fast = (np.where(np.abs(v) > 0, np.abs(v), np.nan) for v in pairs.T)
        p = top.polar * rpm * RPM
        self.slow_item.setData(rpm, slow, connect="finite")
        self.fast_item.setData(rpm, fast, connect="finite")
        self.slow_guess.setData(rpm, top.weight_moment / p if top.weight_moment > 0 else np.full_like(rpm, np.nan))
        c = math.cos(th)
        self.fast_guess.setData(rpm, p / (top.transverse * c) if c > 1e-9 else np.full_like(rpm, np.nan))
        self.least_line.setPos(least if least > 0 else -1.0)
        self.least_line.setVisible(least > 0)
        s, f = steady_precession(top, m.spin, th)
        on = math.isfinite(s) and s != 0
        self.precession_now.setData([here] if on else [], [abs(s)] if on else [])
        self.precession_plot.setXRange(0, hi, padding=0)
        self.precession_plot.setLabel("bottom", "Spin |ω₃| (rpm)")
        if top.weight_moment <= 0:
            self.precession_note.setText(
                f"No gravity: no slow precession, and the fast one, {abs(f):.4g} rad/s at {here:.4g} rpm, is the free "
                "top's wobble."
                if math.isfinite(f)
                else "No gravity and lying flat: no steady precession but the spin."
            )
        elif least > 0 and here < least:
            self.precession_note.setText(
                f"At {math.degrees(th):.4g}° it needs {least:.4g} rpm to precess steadily; at {here:.4g} rpm it "
                "can only nod."
            )
        else:
            fast_text = f" and {abs(f):.4g}" if math.isfinite(f) else ""
            self.precession_note.setText(
                f"At {math.degrees(th):.4g}° and {here:.4g} rpm: {abs(s):.4g}{fast_text} rad/s (the estimates "
                f"{top.weight_moment / (top.polar * abs(m.spin)):.4g}"
                + (f" and {top.polar * abs(m.spin) / (top.transverse * c):.4g}" if c > 1e-9 else "")
                + ")."
                + (f" Under {least:.4g} rpm (dotted) it can't." if least > 0 else "")
            )

    def _draw_path(self) -> None:
        if not self.motion.top.symmetric:
            self.path_circles.setData([], [])
            self.path_plot.setRange(xRange=(-1.05, 1.05), yRange=(-1.05, 1.05), padding=0)
            self.path_note.setText(
                f"From above, the last {SKEW_PATH:g} s. With no turning points to draw it against, the path no longer "
                "keeps between two circles."
            )
            self._path_trace()
            return
        ring = np.linspace(0, 2 * math.pi, 241)
        xs, ys = [], []
        for th in self.motion.turning():
            xs += list(math.sin(th) * np.cos(ring)) + [np.nan]
            ys += list(math.sin(th) * np.sin(ring)) + [np.nan]
        self.path_circles.setData(xs, ys)
        reach = max(math.sin(th) for th in self.motion.turning())
        if max(self.motion.turning()) > math.pi / 2:
            reach = 1.0
        span = max(1.1 * reach, 0.02)
        self.path_plot.setRange(xRange=(-span, span), yRange=(-span, span), padding=0)
        self.path_note.setText(
            f"From above, the last two turns: {self.motion.path().value}. Let go at the bottom of the plot, it "
            + (
                f"precesses {'counterclockwise' if self.motion.mean_precession() > 0 else 'clockwise'}."
                if abs(self.motion.mean_precession()) > 1e-9
                else "doesn't precess on average: it circles L."
            )
        )
        self._path_trace()

    def _path_trace(self) -> None:
        n = self.history.size
        if not n:
            self.path_item.setData([], [])
            self.path_now.setData([], [])
            return
        mean = abs(self.motion.mean_precession())
        if not self.motion.top.symmetric:
            span = SKEW_PATH
        else:
            span = min(4 * math.pi / mean, 40.0) if mean > 1e-9 else 6 * self.motion.nutation_period()
        _, y = self.history.window(span)
        k = thin(len(y), PATH_POINTS)
        self.path_item.setData(y[k, 0], y[k, 1])
        self.path_now.setData([y[-1, 0]], [y[-1, 1]])

    def _skewed_check(self, t: np.ndarray, y: np.ndarray, scale_e: float, scale_l: float) -> None:
        """An asymmetric top's check: what's kept, ω₃ not, and without gravity Jacobi's ω."""
        m = self.motion
        top = m.top
        spin = np.sum(y[:, 7:10] * y[:, 0:3], axis=1) / top.polar / RPM
        parts = [
            f"h = {self.sim.h * 1e3:.3g} ms. Largest energy error {np.max(np.abs(y[:, 6] - m.total_energy)) / scale_e:.2g}"
            f"{' mgl' if top.weight_moment > 0 else ' of it'}, L·ẑ's {np.max(np.abs(y[:, 9] - m.vertical_momentum)) / scale_l:.2g} "
            f"of I₃ω₃. ω₃ isn't kept: it has run between {spin.min():.4g} and {spin.max():.4g} rpm."
        ]
        if top.weight_moment == 0:
            omega = body_rates(y[:, 0:3], y[:, 3:6], y[:, 7:10], top)
            exact = FreeSpin.of(top, self._launch_rates()).omega(t)
            error = np.max(np.linalg.norm(omega - exact, axis=1)) / max(abs(m.spin), 1e-12)
            parts.append(f"ω in the body against Euler–Poinsot's (Jacobi's cn, sn, dn): within {error:.2g} of ω₃.")
        self.check_note.setText(" ".join(parts))

    def _draw_chaos(self) -> None:
        """The Poincaré section and the two tops' separation."""
        n = self.section.size
        self.section_item.setData(self.section.y[:n, 0], self.section.y[:n, 1])
        a = self.apart.size
        t, d = self.apart.t[:a], self.apart.y[:a, 0]
        keep = d > 0
        self.apart_item.setData(t[keep], d[keep])
        if a:
            self.apart_plot.setXRange(0.0, max(float(t[-1]), 1.0), padding=0)
        self.apart_plot.setYRange(math.log10(TWIN) - 1, math.log10(2.5), padding=0)
        now = float(d[-1]) if a else TWIN
        symmetric = self.motion.top.symmetric
        self.chaos_note.setText(
            f"{n} peaks of the tilt so far. "
            + (
                "A symmetric top keeps ω₃, so its points lie on one line."
                if symmetric
                else "Closed curves: the motion is regular; a scatter that fills an area: chaos."
            )
            + f" Below, a second top let go {TWIN:g} rad further over: {now:.2g} rad apart now. Regular motion "
            "drifts apart slowly, in proportion to the time; chaos doubles the gap over and over, a straight line on "
            "this log scale, until they're as far apart as two axes can be."
        )

    def _draw_check(self, text: bool = True) -> None:
        n = self.history.size
        m = self.motion
        top = m.top
        scale_e = top.weight_moment if top.weight_moment > 0 else max(abs(m.total_energy), 1e-12)
        scale_l = max(abs(m.spin_momentum), 1e-12)
        if not n:
            self.energy_item.setData([], [])
            self.momentum_item.setData([], [])
            self.check_note.setText("Run it to compare.")
            return
        t, y = self.history.t[:n], self.history.y[:n]
        k = thin(n, CHECK_POINTS)
        self.energy_item.setData(t[k], (y[k, 6] - m.total_energy) / scale_e)
        along = np.sum(y[k, 7:10] * y[k, 0:3], axis=1)
        drift = np.abs(y[k, 9] - m.vertical_momentum) / scale_l
        if top.symmetric:
            drift = np.maximum(drift, np.abs(along - m.spin_momentum) / scale_l)
        self.momentum_item.setData(t[k], drift)
        if not text:
            return
        if not top.symmetric:
            self._skewed_check(t, y, scale_e, scale_l)
            return
        theta, phi = tilt_and_azimuth(y[:, 0:3])
        parts = [
            f"h = {self.sim.h * 1e3:.3g} ms. Largest energy error {np.max(np.abs(y[:, 6] - m.total_energy)) / scale_e:.2g} mgl."
        ]
        hi, lo = (math.degrees(v) for v in m.turning())
        parts.append(
            f"Tilt run between {math.degrees(theta.min()):.4g}° and {math.degrees(theta.max()):.4g}° "
            f"(closed form {hi:.4g}° and {lo:.4g}°)."
        )
        T = m.nutation_period()
        if not m.steady and t[-1] - t[0] > 2 * T:
            whole = int((t[-1] - t[0]) / T)
            i = min(int(np.searchsorted(t, t[0] + whole * T)), n - 1)
            rate = (phi[i] - phi[0]) / (t[i] - t[0])
            parts.append(
                f"Over {whole} whole nods it precessed at {rate:.4g} rad/s on average (closed form "
                f"{m.mean_precession():.4g})."
            )
        self.check_note.setText(" ".join(parts))
