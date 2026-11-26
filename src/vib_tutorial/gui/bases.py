"""Coordinates & bases page (#478 Phase 4's new page): two masses live, their motion a point in a plane, described in
any basis the reader drags.

Left: the masses and springs, the basis's switches (one arrow, the modes drawn), then the Things to try (setups to go
back to; their predictions are asked once the reader has handled the masses). Center: the two masses between their
walls (`bases_view.ChainView`: pluck a mass, or drag its grips to change m and k as it runs), the (x₁, x₂) plane
(`bases_plane.Plane`: the point and its path, the arrows v₁ and v₂ to drag, the parallelogram q₁v₁ + q₂v₂), what
the matrices say under it, the run's controls, and the coordinates q against time, the past redescribed whenever
the arrows move. Right: the matrices' steps from M, K and C to VᵀMV, VᵀKV and VᵀCV as labeled tables, their
coupling marked (Matrices, `matrix_label`), the energy along each arrow and the cross term, and their sum in bars (Coupling),
one arrow's frequency against its angle, built by hand angle by angle (One arrow), the energy balance through every
change (Energy), and the Theory.

The masses are stepped as they run (`view_models/live_bases.LiveBases`, exactly); what the page shows is
`view_models/bases.py`'s; its trials, problems, tips and notes are `bases_trials.py`'s (no Qt). The controls are
bound to the state (`BasesTrial`) by `bindings.Bindings`, the arrows kept by the page, so Defaults round-trips by
construction.

Working on it: `tests/test_bases_page.py` (the page live: pluck, grips, arrows that snap and redescribe, the rail
built by hand, predictions after play), `test_live_bases.py` (the Qt-free parts), `test_readiness.py`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.bases import TwoMass, angle, unit
from ..core.stepping import FRAME_BUDGET, Budget
from ..view_models import bases as view
from ..view_models import live_bases as lb
from ..view_models.bases import MM, BasesTrial
from ..view_models.live_bases import COL, LiveBases
from ..view_models.text import number as sig
from . import coalesce
from .bases_plane import Plane
from .bases_trials import NOTES_HTML, PROBLEMS, TIPS, TRIALS
from .bases_view import ChainView
from .bindings import Bindings
from .frame_loop import FrameLoop
from .layouts import FlowLayout
from .lesson import Form, LiveLabel, TrialBox, plot, theory, three_panes
from .live_parts import ChangeMarks, EnergyTab, run_row
from .matrix_label import MatricesLabel, scrolling
from .matrix_table import side_by_side, table_html
from .plots import SteadyRange, decimate
from .scrub import Scrub
from .style import colors, role
from .theming import add_legend, mute
from .transport import Transport

__all__ = ["NOTES_HTML", "PROBLEMS", "TRIALS", "BasesPage"]

STRIP_SECONDS = 10.0  # the coordinates' and the Coupling tab's window: seven beats of the pair as it opens
TRAIL_SECONDS = 3.0  # the plane's path
MATRICES_EVERY = 0.1  # s: the matrices' steps redrawn at most this often while an arrow's dragged as it runs
MEASURE_EVERY = 0.2  # s of run between the rail's looks
PLAYED_AFTER = 2.0  # s of run after the reader's first change: they've handled it, predictions on
SYSTEM_FIELDS = ("m1", "m2", "k1", "k2", "k3", "c")
COORDS_TIP = (
    "<p>The coordinates q₁ and q₂ in the basis the arrows make, the last 10 s: x = q₁v₁ + q₂v₂. With the arrows "
    "along the axes they are the masses' own displacements. Move an arrow and the whole trace is described again "
    "in the new basis: the motion didn't change. Along the modes each is one clean sine. A dashed line marks a "
    "change of the masses, the springs or the one arrow. Paused, drag the line at the end back to see the masses "
    "then.</p>"
)
COUPLING_TIP = (
    "<p>The arrows split the motion into two pieces, one along v₁ and one along v₂ (x = q₁v₁ + q₂v₂: the dashed "
    "parallelogram on the plane). <b>Along v₁</b> is the energy of the first piece alone, <b>along v₂</b> the "
    "second's. The energy of the two together is those two plus a <b>cross term</b>, as "
    "|a + b|² = |a|² + |b|² + 2a·b. The cross term can be negative: as it swings, energy passes between the "
    "pieces. Along the modes it is zero, and each piece keeps its energy.</p>"
)
SUM_TIP = (
    "<p>The energy now, added up. Each bar starts where the last ended: along v₁, then along v₂, then the cross "
    "term, up when it's positive and down when it's negative. Where they end is the stored energy, the last bar. "
    "As it runs the pieces trade energy through the cross term, but the total stays put (less what the dampers "
    "take). Along the modes the cross term is zero and the two pieces hold still.</p>"
)
TRIANGLE_TIP = (
    "<p>The two pieces of the motion as vectors, scaled (by √M for the speeds, by √K for the stretches) so that "
    "half a vector's length squared is its energy. <b>a</b>, the piece along v₁, from the middle; <b>b</b>, along "
    "v₂, from a's tip; the dashed <b>a + b</b> is the whole motion. So, as |a + b|² = |a|² + |b|² + 2a·b, the "
    "cross term is a·b: zero at a right angle. The angle between a and b is set by the arrows alone, and the "
    "motion only stretches them. Turn the arrows onto the modes and both triangles are right-angled.</p>"
)
RAIL_TIP = (
    "<p>One arrow's frequency, built by hand: each angle you try lands its measured frequency here "
    "(0° along x₁, 90° along x₂). Points from other masses or springs are faded. <i>Show the theory</i> draws "
    "Rayleigh's quotient for every angle, the two natural frequencies and the modes' angles.</p>"
)
MATRICES_TIP = (
    "<p>The steps from the masses' own matrices, M, K and C, to the matrices in the basis the arrows make: "
    "VᵀMV, VᵀKV and VᵀCV, V times the matrix times V's columns again. Their off-diagonal terms are the coupling "
    "between the coordinates (in color while they aren't zero). The frequencies, the roots of "
    "det(VᵀKV − ω²VᵀMV) = 0, are the same in every basis. The modes make VᵀMV and VᵀKV diagonal, but VᵀCV only "
    "for some dampers.</p>"
)
ENERGY_TIP = (
    "<p>Where the energy went: what your hand put in (a pluck), what keeping or turning one arrow did, what each change cost or "
    "added (a spring changed at a stretch, mass caught or dropped), what the dampers took out, and what's stored "
    "now. Put in, less taken out, with the changes, is what's stored, at every instant.</p>"
)


class SquarePlot(pg.PlotWidget):
    """A plot as tall as it is wide (an energy triangle's): the height it doesn't need goes to the plots above."""

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return width

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(160, 160)


class BasesPage(QtWidgets.QWidget):
    """The masses' controls on the left, the masses, the plane and the coordinates in the center, the tabs right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def transport(self) -> Transport:
        return Transport(self.run_button, self.reset_button, self.speed, self.clock)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.live = LiveBases()
        self.recorder = view.RailRecorder()
        self.running = False
        self.v1: tuple[float, float] = (1.0, 0.0)
        self.v2: tuple[float, float] = (0.0, 1.0)
        self._handled_at: float | None = None
        self._grip_start = 0.0
        self._measured_at = -1.0
        self.rail_hz: float | None = None  # the rail's frequency, as last measured
        self._budget = Budget()
        self.bindings = Bindings(BasesTrial, self)

        # --- left: the masses, the basis, then the Things to try
        box = Form("Two masses")
        self.m1 = box.number("Mass m₁:", 0.05, 100.0, 1.0, 3, " kg", TIPS["m1"])
        self.m2 = box.number("Mass m₂:", 0.05, 100.0, 1.0, 3, " kg", TIPS["m2"])
        self.k1 = box.number("Spring k₁:", 1.0, 1e5, 400.0, 1, " N/m", TIPS["k1"])
        self.k2 = box.number("Coupling k₂:", 0.0, 1e5, 100.0, 1, " N/m", TIPS["k2"])
        self.k3 = box.number("Spring k₃:", 1.0, 1e5, 400.0, 1, " N/m", TIPS["k3"])
        self.c = box.number("Each damper c:", 0.0, 100.0, 0.0, 3, " N·s/m", TIPS["c"])
        basis_box = QtWidgets.QGroupBox("Basis")
        bl = QtWidgets.QVBoxLayout(basis_box)
        self.rail = QtWidgets.QCheckBox("One arrow: v₁ alone, a reduced basis")
        self.rail.setToolTip(TIPS["rail"])
        self.modes = QtWidgets.QCheckBox("Draw the modes")
        self.modes.setToolTip(TIPS["modes"])
        bl.addWidget(self.rail)
        bl.addWidget(self.modes)

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Setups to come back to, each showing one idea from the Theory tab. Once you've handled the masses, "
            "some ask you to predict what you'll see first.</p>",
            predictions=[t.predict for t in TRIALS],
            ask_after_play=True,
        )
        self.trials.picked.connect(self.load_trial)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(box)
        lv.addWidget(basis_box)
        lv.addWidget(self.trials)
        lv.addStretch(1)

        # --- center: the masses, the plane and its matrices, the run, the coordinates
        self.chain = ChainView()
        self.chain.setMinimumHeight(90)
        self.chain.setMaximumHeight(140)
        self.chain.grabbed.connect(self._on_mass_grabbed)
        self.chain.pulled.connect(self._on_mass_pulled)
        self.chain.released.connect(self._on_let_go)
        self.chain.grip_taken.connect(self._on_grip_taken)
        self.chain.grip_moved.connect(self._on_grip_moved)
        self.plane = Plane()
        self.plane.setMinimumSize(220, 220)
        self.plane.grabbed.connect(self._on_point_grabbed)
        self.plane.pulled.connect(self._on_point_pulled)
        self.plane.released.connect(self._on_let_go)
        self.plane.arrow_moved.connect(self._on_arrow_moved)
        self.plane.arrow_released.connect(self._on_arrow_released)
        # redrawn at every step of an arrow's drag: a QLabel laid the whole page out again each time (10 ms a frame)
        self.matrices = MatricesLabel()
        self.matrices.setToolTip(MATRICES_TIP)
        self.plane_note = LiveLabel(wrap=True)  # rewritten as it runs: a QLabel lays the page out each time
        mute(self.plane_note)
        row, self.speed, self.run_button, self.reset_button, self.clock = run_row(self)
        self.run_button.toggled.connect(self._on_run)
        self.reset_button.setToolTip("<p>Back to rest at t = 0; the angles measured are kept.</p>")
        self.reset_button.clicked.connect(self.reset)

        self.q_plot = plot("Coordinate q", "m", "Time", "s", height=130)
        self.q_plot.setToolTip(COORDS_TIP)
        self.q_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.q_plot.getPlotItem(), offset=(-10, 10))
        self.q_items = [pg.PlotDataItem(name=n) for n in ("q₁, along v₁", "q₂, along v₂")]
        for item in self.q_items:
            self.q_plot.addItem(item)
        decimate(self.q_plot)
        self.marks = ChangeMarks(self.q_plot)
        self.scrub = Scrub(self.q_plot)
        self.scrub.scrubbed.connect(self._on_scrubbed)
        self.readout = LiveLabel(wrap=True)

        centre = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(centre)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.chain, 1)
        cv.addWidget(self.plane, 4)  # the whole width: the matrices have a tab of their own
        cv.addWidget(self.plane_note)
        cv.addLayout(row)
        cv.addWidget(self.q_plot, 2)
        cv.addWidget(self.readout)

        # --- right: Matrices, Coupling, Rail, Energy, Theory
        matrices = QtWidgets.QWidget()
        mv = QtWidgets.QVBoxLayout(matrices)
        mv.addWidget(self.matrices)
        mv.addStretch(1)
        coupling = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(coupling)
        self.shares_plot = plot("Energy", "J", "Time", "s", height=130)
        self.shares_plot.setToolTip(COUPLING_TIP)
        self.shares_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.shares_plot.getPlotItem(), offset=(-10, 10))
        self.share_items = [pg.PlotDataItem(name=n) for n in ("along v₁", "along v₂", "cross term")]
        for item in self.share_items:
            self.shares_plot.addItem(item)
        decimate(self.shares_plot)
        self.shares_note = LiveLabel(wrap=True)  # rewritten as it runs: a QLabel lays the page out each time
        mute(self.shares_note)
        # the energy now, added up: each piece stacked on the last, the stored energy beside them
        self.sum_plot = plot("Energy now", "J", "", None, height=110)
        self.sum_plot.setAccessibleName("The energy now, added up: along each arrow, the cross term, stored")
        self.sum_plot.setToolTip(SUM_TIP)
        self.sum_plot.setMouseEnabled(x=False, y=False)
        self.sum_bars = pg.BarGraphItem(x=[], height=[], width=0.6)
        self.sum_plot.addItem(self.sum_bars)
        self.sum_names: tuple[str, ...] = ()  # the bars' names, as the bottom axis's ticks
        self.sum_y = SteadyRange(self.sum_plot, "y")
        self.sum_plot.setXRange(-0.5, 3.5, padding=0)
        # the energy triangles: the pieces as vectors whose length squared is twice their energy
        self.tri_plots: list[pg.PlotWidget] = []
        self.tri_items: list[list[pg.PlotDataItem]] = []  # the two lines, a, b, a + b in each
        self.tri_notes: list[LiveLabel] = []
        self.tri_span = [0.0, 0.0]  # each triangle's reach as shown
        tri_row = QtWidgets.QHBoxLayout()
        for title in ("Kinetic, through M", "Potential, through K"):
            column = QtWidgets.QVBoxLayout()
            tp = SquarePlot()
            tp.setAccessibleName(f"The energy triangle: {title.lower()}")
            tp.setToolTip(TRIANGLE_TIP)
            tp.setTitle(title)
            tp.setMenuEnabled(False)
            tp.setMouseEnabled(x=False, y=False)
            tp.hideButtons()
            tp.hideAxis("left")
            tp.hideAxis("bottom")
            tp.setAspectLocked(True)
            tp.setMinimumSize(60, 60)  # two side by side mustn't widen the pane (the window fits 1280 px)
            policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
            policy.setHeightForWidth(True)
            tp.setSizePolicy(policy)
            # the scaled arrows' lines (faint, still), then a, b from a's tip, a + b
            items = [pg.PlotDataItem() for _ in range(5)]
            for item in items:
                tp.addItem(item, ignoreBounds=True)
            note = LiveLabel(wrap=True)
            mute(note)
            column.addWidget(tp)
            column.addWidget(note)
            tri_row.addLayout(column, 1)
            self.tri_plots.append(tp)
            self.tri_items.append(items)
            self.tri_notes.append(note)
        pv.addWidget(self.shares_plot, 2)
        pv.addWidget(self.sum_plot, 1)
        pv.addLayout(tri_row)
        pv.addWidget(self.shares_note)

        rails = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(rails)
        self.theory = QtWidgets.QCheckBox("Show the theory")
        self.theory.setToolTip(TIPS["theory"])
        self.clear_button = QtWidgets.QPushButton("Clear points")
        self.clear_button.setToolTip("<p>Forget the angles measured.</p>")
        self.clear_button.clicked.connect(self.clear_points)
        buttons = FlowLayout()
        buttons.add(self.theory, self.clear_button)
        rv.addLayout(buttons)
        self.rail_plot = plot("Frequency", "Hz", "The arrow's angle", "°", height=200)
        self.rail_plot.setToolTip(RAIL_TIP)
        self.rail_plot.setMouseEnabled(x=False, y=False)
        self.rail_plot.enableAutoRange(x=False, y=False)  # set from the view: the same with the theory or not
        add_legend(self.rail_plot.getPlotItem(), offset=(10, -10))
        self.quotient_item = pg.PlotDataItem(name="Rayleigh's quotient")
        self.earlier_item = pg.PlotDataItem(pen=None, symbol="o", symbolSize=6)
        self.points_item = pg.PlotDataItem(pen=None, symbol="o", symbolSize=8, name="measured")
        for item in (self.quotient_item, self.earlier_item, self.points_item):
            self.rail_plot.addItem(item)
        self.levels = [pg.InfiniteLine(angle=0) for _ in range(2)]
        self.mode_marks = [pg.InfiniteLine(angle=90) for _ in range(2)]
        self.now_mark = pg.InfiniteLine(angle=90)
        for line in (*self.levels, *self.mode_marks, self.now_mark):
            self.rail_plot.addItem(line, ignoreBounds=True)
        self.rail_note = LiveLabel(wrap=True)  # rewritten as it runs: a QLabel lays the page out each time
        mute(self.rail_note)
        rv.addWidget(self.rail_plot, 1)
        rv.addWidget(self.rail_note)

        self.energy = EnergyTab(
            balance=lb.energy_view, tip=ENERGY_TIP, empty="Pluck the masses to see where the energy goes."
        )
        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(scrolling(matrices), "Matrices")
        self.tabs.addTab(coupling, "Coupling")
        self.tabs.addTab(rails, "One arrow")
        self.tabs.addTab(self.energy, "Energy")
        self.tabs.addTab(self.notes, "Theory")

        three_panes(self, left, centre, self.tabs, sizes=(300, 780, 560))

        for field in SYSTEM_FIELDS:
            self.bindings.bind(field, getattr(self, field))
        for field, widget in (("rail", self.rail), ("modes", self.modes), ("theory", self.theory), ("tab", self.tabs)):
            self.bindings.bind(field, widget)
        self.bindings.write(BasesTrial("", ""))
        self.bindings.changed.connect(self._on_field)

        self._still_due = False  # an arrow moved since the last frame drew the plane
        self._matrices_at = -1.0  # when the matrices were last drawn (monotonic s)
        self._matrices_late = False  # a drag held them back: drawn when it lets go
        self.loop = FrameLoop(self, "Coordinates & bases", self._tick, self._runs, self._sim_t)
        self.apply_theme()
        self._push_system()

    # ------------------------------------------------------------- the state
    def practice_state(self) -> BasesTrial:
        """The page's settings, as a trial: what a practice spec checks, and Defaults puts back."""
        return self.bindings.read(BasesTrial("", "", v1=self.v1, v2=self.v2))

    @property
    def system(self) -> TwoMass:
        return self.live.system

    @property
    def basis(self) -> np.ndarray:
        return np.column_stack([np.array(self.v1), np.array(self.v2)])

    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(BasesTrial("", "", **setup))

    def apply_trial(self, t: BasesTrial) -> None:
        """A setup: the masses, springs and arrows as it says, the run from rest, then plucked from `pluck`."""
        self.bindings.write(t)
        self.v1, self.v2 = (float(t.v1[0]), float(t.v1[1])), (float(t.v2[0]), float(t.v2[1]))
        self.recorder.clear()
        self.live.set_rail(None)
        self._push_system()
        self.reset()
        self._handled_at = None
        self.trials.show_note(t.note)
        if any(t.pluck):
            self.live.hold(np.array(t.pluck))
            self.live.let_go()
            self._run(True)
        self._draw_tab()

    # ------------------------------------------------------------- what the reader changes
    def _on_field(self, field: str) -> None:
        if field == "theory":
            self._draw_rail()
        elif field == "tab":
            self._draw_tab()
        elif field == "modes":
            self._draw_still()
        else:
            self._handled()
            self._push_system()

    def _handled(self) -> None:
        if self._handled_at is None:
            self._handled_at = self.live.t

    def _push_system(self) -> None:
        """The live run to the controls: the masses and springs (a live change), and the rail."""
        state = self.practice_state()
        self.live.set_system(view.system(state))
        self.live.set_rail(np.array(state.v1) if state.rail else None)
        s = self.live.system
        self.chain.set_values(s.m1, s.m2, s.k1, s.k2, s.k3, s.c)
        self._draw_still()
        self._draw_rail()
        if not self.running:
            self._draw_live()  # running, the next frame draws it

    def _on_grip_taken(self, name: str) -> None:
        self._grip_start = getattr(self, name).value()

    def _on_grip_moved(self, name: str, distance: float) -> None:
        box = getattr(self, name)
        box.blockSignals(True)
        box.setValue(view.GRIPS[name].dragged(self._grip_start, distance))
        box.blockSignals(False)
        self._handled()
        self._push_system()

    def _on_arrow_moved(self, i: int, x1: float, x2: float) -> None:
        v, _ = view.arrow_dragged((x1, x2), view.snap_targets(self.live.system))
        other = self.v2 if i == 0 else self.v1
        if not self.rail.isChecked() and abs(v[0] * other[1] - v[1] * other[0]) < 1e-9:
            return  # onto the other arrow: no basis, so it stays where it was
        if i == 0:
            self.v1 = v
        else:
            self.v2 = v
        self._handled()
        if i == 0 and self.rail.isChecked():
            self.live.set_rail(np.array(v))  # the rail turned, as it runs
        if self.running:
            # drawn in the next frame: a drag's moves come up to 120 a second, and each drawn at once repainted
            # the plane and the matrices between frames as well as in them
            self._still_due = True
            return
        self._draw_still()
        self._draw_live()

    def _on_arrow_released(self, _i: int) -> None:
        self._flush_still()
        self._draw_rail()

    def _flush_still(self) -> None:
        """The arrow a running drag left for the next frame, drawn now (let go, paused); the matrices too, if a
        drag held them back."""
        let_go = self._matrices_late and type(self.plane.held) is not int
        if self._still_due or let_go:
            self._still_due = False
            if let_go:
                self._matrices_at = -1.0  # drawn now, not when they're next due
            self._draw_still()

    def _on_point_grabbed(self) -> None:
        self.scrub.hide()
        self._handled()

    def _on_point_pulled(self, x1: float, x2: float) -> None:
        self.live.hold(np.array([x1, x2]))
        self._draw_body()

    def _on_mass_grabbed(self, _i: int) -> None:
        self._on_point_grabbed()

    def _on_mass_pulled(self, i: int, x: float) -> None:
        now = self.live.x.copy()
        now[i] = x
        self._on_point_pulled(float(now[0]), float(now[1]))

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
            self._flush_still()
            self._show_scrub()

    def _run(self, on: bool) -> None:
        self.run_button.set_running(on)
        self._on_run(on)

    def _runs(self) -> bool:
        return self.running

    def _sim_t(self) -> float:
        return self.live.t

    def reset(self) -> None:
        """Back to rest at t = 0, waiting for Start; the rails measured kept."""
        self.live.reset()
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
        self._played()
        self._flush_still()
        self._draw_live(frame=True)

    def _played(self) -> None:
        if self._handled_at is not None and self.live.t - self._handled_at >= PLAYED_AFTER:
            self.trials.set_played()

    def step(self, duration: float) -> None:
        """Run on by `duration` s of simulated time at once (tests), measuring and drawing as the frames would."""
        end = self.live.t + duration
        while self.live.t < end - 1e-9 and self.live.held is None:
            self.live.advance(min(MEASURE_EVERY, end - self.live.t) + 1e-12)
            self._measure()
        self._played()
        self._flush_still()
        self._draw_live()

    # ------------------------------------------------------------- the rail, built by hand
    def _measure(self) -> None:
        live = self.live
        if live.rail is None or live.held is not None or live.t - self._measured_at < MEASURE_EVERY:
            return
        self._measured_at = live.t
        hz = self.rail_hz = live.rail_hz()
        if hz is not None and self.recorder.update(angle(live.rail), hz, view.key(live.system), live.t):
            self._draw_rail()

    def clear_points(self) -> None:
        self.recorder.clear()
        self._draw_rail()

    @coalesce.coalesced
    def _draw_rail(self, *_: object) -> None:
        s, k = self.live.system, view.key(self.live.system)
        now = angle(self.live.rail) if self.live.rail is not None else None
        v = view.rail_view(s, self.recorder.current(k), self.recorder.earlier(k), self.theory.isChecked(), now)
        self.points_item.setData(*v.points)
        self.earlier_item.setData(*v.earlier)
        self.quotient_item.setData(*v.quotient)
        for line, value in zip(self.levels, (*v.levels, None, None)):
            line.setVisible(value is not None)
            if value is not None:
                line.setValue(value)
        for line, value in zip(self.mode_marks, (*v.mode_angles, None, None)):
            line.setVisible(value is not None)
            if value is not None:
                line.setValue(value)
        self.now_mark.setVisible(v.now is not None)
        if v.now is not None:
            self.now_mark.setValue(v.now)
        self.rail_plot.setXRange(*v.x_span, padding=0)
        self.rail_plot.setYRange(*v.y_span, padding=0)
        self.rail_note.setText(v.note)

    # ------------------------------------------------------------- scrubbing
    def _show_scrub(self) -> None:
        r = self.live.record
        if r.size > 1:
            end = float(r.t[r.size - 1])
            self.scrub.show(end, max(float(r.t[0]), end - STRIP_SECONDS))

    def _on_scrubbed(self, t: float) -> None:
        r = self.live.record
        ts = r.t[: r.size]
        x = np.array([float(np.interp(t, ts, r.y[: r.size, COL[c]])) for c in ("x1", "x2")])
        self.chain.show_at(x)
        V, rail = self.basis, self.rail.isChecked()
        self.plane.show_motion(self._trail(t), x, view.projection(V, rail, x))
        self.clock.setText(f"t = {sig(t, 4)} s (paused at {sig(self.live.t, 4)} s)")

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.chain.apply_theme()
        self.plane.apply_theme()
        self.scrub.apply_theme()
        self.marks.apply_theme()
        self.energy.apply_theme()
        for i, item in enumerate(self.q_items):
            item.setPen(pg.mkPen(c.mass[i], width=2))
        for i, item in enumerate(self.share_items[:2]):
            item.setPen(pg.mkPen(c.mass[i], width=2))
        self.share_items[2].setPen(pg.mkPen(c.grey, width=1.5))
        for line_a, line_b, one, two, whole in self.tri_items:
            line_a.setPen(pg.mkPen(c.mass[0], width=1, style=QtCore.Qt.PenStyle.DotLine))
            line_b.setPen(pg.mkPen(c.mass[1], width=1, style=QtCore.Qt.PenStyle.DotLine))
            one.setPen(pg.mkPen(c.mass[0], width=3))
            two.setPen(pg.mkPen(c.mass[1], width=3))
            whole.setPen(pg.mkPen(c.strong, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.quotient_item.setPen(pg.mkPen(c.strong, width=1.5))
        self.points_item.setSymbolPen(None)
        self.points_item.setSymbolBrush(pg.mkBrush(c.mass[0]))
        self.earlier_item.setSymbolPen(pg.mkPen(c.grey, width=1))
        self.earlier_item.setSymbolBrush(None)
        for line in self.levels:
            line.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DotLine))
        for line in self.mode_marks:
            line.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.now_mark.setPen(pg.mkPen(c.mass[0], width=1))
        self._draw_still()
        self._draw_rail()
        self._draw_tab()

    @coalesce.coalesced
    def _draw_still(self, *_: object) -> None:
        """What moves only when the reader changes something: the plane's arrows, grid and lines, the matrices."""
        s, V, rail = self.live.system, self.basis, self.rail.isChecked()
        p = view.plane_view(s, V, rail, self.modes.isChecked())
        self.plane.set_still(p)
        now = time.monotonic()
        if self._still_due_matrices(now):
            return
        self._matrices_at = now
        m = view.matrices_view(s, V, rail)
        self.plane_note.setText(f"{m.note} {p.note}")  # under the plane: what the Matrices tab says
        self.matrices.set_variants(self._matrices_html(m, True), self._matrices_html(m, False))

    def _still_due_matrices(self, now: float) -> bool:
        """True while a running drag's matrices wait: drawn at most `MATRICES_EVERY` apart then (their rich text
        took 2 ms a frame to set and more to paint, faster than anyone reads), and exactly on letting go."""
        arrow_held = type(self.plane.held) is int  # an arrow's index (False, not held, is a bool)
        if self.running and arrow_held and now - self._matrices_at < MATRICES_EVERY:
            self._matrices_late = True
            return True
        self._matrices_late = False
        return False

    @staticmethod
    def _matrices_html(m: view.MatricesView, wide: bool = True) -> str:
        """The steps: M, K and C, then V and Vᵀ, then VᵀMV, VᵀKV and VᵀCV, then the frequencies. Each matrix a table
        as the Substructuring page's are (`matrix_table`): its rows and columns labeled, its diagonal and its
        coupling tinted apart, a coupling term bold in color. A row's matrices side by side if `wide`, else as many
        to a row as fit two (a narrow pane at 1280 px)."""
        c = colors

        def one(mat: view.Matrix) -> str:
            title = f"{mat.symbol}<br><span style='font-weight:normal'>{mat.unit}</span>" if mat.unit else mat.symbol
            return table_html(
                mat.entries, mat.rows, mat.cols, mat.row_groups, mat.col_groups, title, nowrap=True, marked=mat.marked
            )

        def row(mats: Sequence[view.Matrix]) -> str:
            tables = [one(x) for x in mats]
            if wide or len(tables) <= 2:
                return side_by_side(*tables)
            return side_by_side(*tables[:2]) + "<br>" + side_by_side(*tables[2:])

        def step(text: str) -> str:
            return f"<p style='color:{c.muted}; margin-top:8px; margin-bottom:2px'>{text}</p>"

        v = "v₁" if len(m.basis[0].entries[0]) == 1 else "V"
        parts = [step("1. The system, in x₁ and x₂"), row(m.given)]
        parts += [step(f"2. The arrows as columns: x = {v}q"), row(m.basis)]
        if m.matrices:
            parts += [step(f"3. Multiply by {v}, then by {v}ᵀ: the matrices in q"), row(m.matrices)]
        sums = "".join(f"<br>{line}" for line in m.sums)
        if m.sums:
            parts.append(f"<p>4. The frequencies{sums}</p>")
        parts.append(f"<p style='color:{c.muted}'>{m.note}</p>")
        return "".join(parts)

    @coalesce.coalesced(covers=("_draw_live",))
    def _draw_tab(self, *_: object) -> None:
        self._draw_live()

    def _trail(self, end: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        r = self.live.record
        if not r.size:
            return np.empty(0), np.empty(0)
        t = r.t[: r.size]
        stop = r.size if end is None else int(np.searchsorted(t, end, side="right"))
        start = int(np.searchsorted(t, t[max(stop - 1, 0)] - TRAIL_SECONDS))
        start = max(start, int(np.searchsorted(t, self.live.changed_at if end is None else 0.0)))
        y = r.y[start:stop]
        return y[:, COL["x1"]] * MM, y[:, COL["x2"]] * MM

    @coalesce.coalesced
    def _draw_live(self, frame: bool = False) -> None:
        """What moves: the masses, the point, the coordinates, the readout, and the tab showing."""
        live = self.live
        t, y = live.record.window(STRIP_SECONDS)
        end = live.t
        V, rail = self.basis, self.rail.isChecked()
        x = y[:, COL["x1"] : COL["x2"] + 1]
        q = view.coordinates(V, rail, x)
        for i, item in enumerate(self.q_items):
            if i < q.shape[1]:
                item.setData(t, q[:, i])
            else:
                item.setData([], [])
        self.q_plot.setXRange(max(0.0, end - STRIP_SECONDS), max(end, STRIP_SECONDS), padding=0)
        self.marks.draw(live.edits, end - STRIP_SECONDS)
        self._draw_body()
        self.clock.setText(f"t = {sig(end, 4)} s")
        self._show_readout()
        i = self.tabs.currentIndex()
        if i == view.TAB_COUPLING:
            v = y[:, COL["v1"] : COL["v2"] + 1]
            sh = view.shares(live.system, V, rail, x, v)
            parts = (*sh.own, sh.coupling)
            for k, item in enumerate(self.share_items):
                if k < len(sh.own) or k == 2:
                    item.setData(t, parts[k if k < len(sh.own) else -1])
                else:
                    item.setData([], [])
            self.shares_plot.setXRange(max(0.0, end - STRIP_SECONDS), max(end, STRIP_SECONDS), padding=0)
            self.shares_note.setText(sh.note if t.size else "Pluck the masses to see the energy along each arrow.")
            self._draw_sum(view.energy_sum(sh))
            self._draw_triangles(view.energy_triangles(live.system, V, rail, x, v))
        elif i == view.TAB_ENERGY:
            self.energy.draw(live.record, frame)

    def _draw_triangles(self, triangles: tuple[view.Triangle, view.Triangle]) -> None:
        """The energy triangles (`view.energy_triangles`): a, then b from its tip, and a + b closing them."""
        for k, tri in enumerate(triangles):
            (ax, ay), (bx, by) = tri.a, tri.b
            line_a, line_b, one, two, whole = self.tri_items[k]
            for line, d in zip((line_a, line_b), (*tri.lines, None, None)):
                r = 10 * max(tri.span, self.tri_span[k])  # past the plot's edge either way
                line.setData([-r * d[0], r * d[0]], [-r * d[1], r * d[1]]) if d else line.setData([], [])
            one.setData([0.0, ax], [0.0, ay])
            two.setData([ax, ax + bx], [ay, ay + by])
            whole.setData([0.0, ax + bx], [0.0, ay + by])
            shown = self.tri_span[k]
            if not shown or tri.span > shown or tri.span < 0.5 * shown:  # steady: re-ranged only when it must
                self.tri_span[k] = 1.05 * tri.span
                r = self.tri_span[k]
                self.tri_plots[k].setRange(QtCore.QRectF(-r, -r, 2 * r, 2 * r), padding=0)
            if tri.apart is None:
                text = "One arrow: one piece, no cross term."
            elif abs(tri.apart - 90.0) < 0.05:
                text = "a and b at 90°: no cross term."
            else:
                text = f"a and b {sig(tri.apart, 3)}° apart: a cross term."
            self.tri_notes[k].setText(text)

    def _draw_sum(self, es: view.EnergySum) -> None:
        """The Coupling tab's bars (`view.energy_sum`): a stack of the pieces, and the stored energy beside it."""
        bars = es.bars
        n = len(bars)
        self.sum_bars.setOpts(
            x=list(range(n)),
            y0=[min(b.bottom, b.top) for b in bars],
            height=[abs(b.top - b.bottom) for b in bars],
            brushes=[pg.mkBrush(role(b.role)) for b in bars],
            pen=pg.mkPen(colors.background, width=1),  # a pen a bar trips pyqtgraph 0.14's BarGraphItem
        )
        names = tuple(b.name for b in bars)
        if names != self.sum_names:  # only when the bars change (one arrow or two): ticks re-lay the axis out
            self.sum_names = names
            self.sum_plot.getAxis("bottom").setTicks([list(enumerate(names))])
            self.sum_plot.setXRange(-0.5, max(n, 2) - 0.5, padding=0)
        self.sum_y.fit(0.0, es.y_top)

    def _draw_body(self) -> None:
        x = self.live.x
        self.chain.show_at(x)
        self.plane.show_motion(self._trail(), x, view.projection(self.basis, self.rail.isChecked(), x))

    def _show_readout(self) -> None:
        live = self.live
        x = live.x * MM
        if live.held is not None:
            self.readout.setText(f"Held at x = ({sig(x[0], 3)}, {sig(x[1], 3)}) mm: let go to pluck them.")
            return
        if live.record.size < 2:
            self.readout.setText("Pluck them: drag the point on the plane, or a mass, and let go.")
            return
        if live.rail is not None:
            hz = self.rail_hz if live.t - live.changed_at > MEASURE_EVERY else None
            where = f"One arrow at {sig(angle(live.rail), 3)}°"
            self.readout.setText(f"{where}: {sig(hz, 4)} Hz, measured." if hz else f"{where}: timing its swing…")
            return
        V = self.basis
        if view.singular(V):
            self.readout.setText(f"x = ({sig(x[0], 3)}, {sig(x[1], 3)}) mm: the arrows make no basis.")
            return
        q = np.linalg.solve(V, live.x) * MM
        self.readout.setText(
            f"x = ({sig(x[0], 3)}, {sig(x[1], 3)}) mm = {sig(q[0], 3)} v₁ + {sig(q[1], 3)} v₂ (q in mm)."
        )

    def unit_arrow(self, i: int) -> np.ndarray:
        """Arrow i's direction (tests)."""
        return unit(np.array(self.v1 if i == 0 else self.v2))
