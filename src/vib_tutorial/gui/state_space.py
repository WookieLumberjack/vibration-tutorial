"""State-space modes page (#478 Phase 4, after Coordinates & bases): the same two masses, their state (x, ẋ) four
numbers, shown two at a time on two planes in the basis the reader picks, so the coupling between the pairs, or its
absence, is seen as it runs.

Left: the masses, springs and dampers, the planes' basis (each mass's x and ẋ, the modes of M and K, or A's own
modes), where to start the masses, then the Things to try (setups to go back to; their predictions are asked once
the reader has handled the masses). Center: the two masses between their walls (`bases_view.ChainView`: pluck a
mass, or drag its grips to change m, k and c as it runs), the plane of the masses' displacements with the
state-space modes' shapes (`state_planes.ShapePlane`), the two planes of the state's coordinate pairs
(`state_planes.PairPlane`: drag to set a pair), the run's controls, and the four coordinates against time. Right:
the matrices' steps from M, C, K to T⁻¹AT, its coupling marked (Matrices), each pair's size on a log axis in both
modal bases (Sizes), each mode's shape as phasors (Shapes), and the Theory.

The masses are stepped as they run (`view_models/live_bases.LiveBases`, exactly, the last page's run); what the page
shows is `view_models/state_space.py`'s; its trials, problems, tips and notes are `state_space_trials.py`'s (no
Qt). The controls are bound to the state (`StateTrial`) by `bindings.Bindings`, so Defaults round-trips by
construction.

Working on it: `tests/test_state_space_page.py` (the page live: starts, the planes' drags, switching the basis as
it runs, the grips, predictions after play), `test_state_space_view.py` and `test_state_modes.py` (the Qt-free
parts), `test_readiness.py`.
"""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.bases import TwoMass
from ..core.stepping import FRAME_BUDGET, Budget
from ..view_models import state_space as view
from ..view_models.live_bases import COL, LiveBases
from ..view_models.state_space import MM, StateTrial
from ..view_models.text import number as sig
from . import coalesce
from .bases_view import ChainView
from .bindings import Bindings
from .frame_loop import FrameLoop
from .layouts import FlowLayout
from .lesson import Form, LiveLabel, TrialBox, plot, theory, three_panes
from .live_parts import ChangeMarks, run_row
from .matrix_label import MatricesLabel, scrolling
from .matrix_table import side_by_side, table_html
from .scrub import Scrub
from .state_planes import PairPlane, PlaneRow, ShapePlane
from .state_space_trials import NOTES_HTML, PROBLEMS, TIPS, TRIALS
from .style import colors
from .theming import add_legend, mute
from .transport import Transport

__all__ = ["NOTES_HTML", "PROBLEMS", "TRIALS", "StateSpacePage"]

STRIP_SECONDS = 8.0  # the coordinates' and the Sizes tab's window
TRAIL_SECONDS = 2.0  # the planes' paths
PHASOR_HEIGHT = 250  # px: the Shapes tab's plots
PLAYED_AFTER = 2.0  # s of run after the reader's first change: they've handled it, predictions on
SYSTEM_FIELDS = ("m1", "m2", "k1", "k2", "k3", "c")
COORDS_TIP = (
    "<p>The state's four coordinates in the basis picked, the last 8 s (mm): each pair a thick line and a thin "
    "one, in its plane's color. Switch the basis and the whole trace is described again: the motion didn't change. A dashed line "
    "marks a change of the masses, springs or dampers. Paused, drag the line at the end back to see the masses "
    "then.</p>"
)
MATRICES_TIP = (
    "<p>The steps from the masses' matrices to the state's: M, C and K make A, the state matrix of ż = Az. The basis "
    "T has the basis's state vectors as its columns, two to a plane (the last page's V, for the state): z = Tw, "
    "so Tẇ = ATw, and ẇ alone takes T⁻¹: T⁻¹AT. The last page's Vᵀ was a choice (any matrix on the left keeps "
    "MVq̈ + CVq̇ + KVq = 0 true, and Vᵀ keeps it symmetric); here there's none. Its 2×2 blocks off the diagonal are the coupling between the pairs, in red where it isn't zero. "
    "The modes' λ, frequencies and damping ratios underneath are the same in every basis.</p>"
)
SIZES_TIP = (
    "<p>Each pair's size, √(a² + b²), on a log axis, in two bases at once whatever the planes show: the modes of M "
    "and K (thin) and A's own modes (thick). A state-space mode's size falls as e^(σt), a straight line here; a "
    "mode of M and K started at zero climbs when the dampers couple it to the other.</p>"
)
SHAPES_TIP = (
    "<p>Each mode's shape ψ, a mass's part as an arrow in the complex plane (a phasor), beside the undamped mode's "
    "real shape, dashed. The angle between the masses' arrows is the phase between them: 0° or 180° when the shape "
    "is real, other angles when the dampers make it complex.</p>"
)


def _square(plot_: pg.PlotWidget) -> None:
    """A phasor plot: its aspect locked, a fixed height, and narrow enough that two side by side don't widen the
    pane (the window fits 1280 px)."""
    plot_.setMenuEnabled(False)
    plot_.setMouseEnabled(x=False, y=False)
    plot_.hideButtons()
    plot_.setAspectLocked(True)
    plot_.setMinimumWidth(60)
    plot_.setFixedHeight(PHASOR_HEIGHT)
    plot_.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)


class StateSpacePage(QtWidgets.QWidget):
    """The controls on the left, the masses, the planes and the coordinates in the center, the tabs right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def transport(self) -> Transport:
        return Transport(self.run_button, self.reset_button, self.speed, self.clock)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.live = LiveBases(view.system(StateTrial("", "")))
        self.running = False
        self._handled_at: float | None = None
        self._grip_start = 0.0
        self._w_held = np.zeros(4)  # the coordinates when a pair was taken (mm): the other pair stays there
        self._budget = Budget()
        self.bindings = Bindings(StateTrial, self)

        # --- left: the masses, the basis, the starts, then the Things to try
        box = Form("Two masses")
        self.m1 = box.number("Mass m₁:", 0.05, 100.0, 1.0, 3, " kg", TIPS["m1"])
        self.m2 = box.number("Mass m₂:", 0.05, 100.0, 2.0, 3, " kg", TIPS["m2"])
        self.k1 = box.number("Spring k₁:", 1.0, 1e5, 400.0, 1, " N/m", TIPS["k1"])
        self.k2 = box.number("Coupling k₂:", 0.0, 1e5, 100.0, 1, " N/m", TIPS["k2"])
        self.k3 = box.number("Spring k₃:", 1.0, 1e5, 400.0, 1, " N/m", TIPS["k3"])
        self.c = box.number("Each damper c:", 0.0, 100.0, 1.0, 3, " N·s/m", TIPS["c"])
        basis_box = QtWidgets.QGroupBox("The planes' coordinates")
        bl = QtWidgets.QVBoxLayout(basis_box)
        self.basis = QtWidgets.QComboBox()
        self.basis.addItems(view.BASIS_NAMES)
        self.basis.setToolTip(TIPS["basis"])
        self.basis.setAccessibleName("The planes' coordinates")
        self.basis.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        bl.addWidget(self.basis)
        start_box = QtWidgets.QGroupBox("Start from")
        sl = QtWidgets.QVBoxLayout(start_box)
        buttons = QtWidgets.QWidget()
        starts = FlowLayout(buttons)
        self.start_buttons = []
        for i, name in enumerate(view.STARTS[: view.BUTTONS]):
            b = QtWidgets.QPushButton(name)
            b.clicked.connect(self._on_start_clicked)
            b.setProperty("start", i)
            # Laid out by its whole rect: macOS's layout rect leaves out the bezel's shadow, and the first line's
            # button, with no room above it for that, was cut 4 px short and drawn as a square bevel.
            b.setAttribute(QtCore.Qt.WidgetAttribute.WA_LayoutUsesWidgetRect)
            starts.add(b)
            self.start_buttons.append(b)
        sl.addWidget(buttons)
        self.start_note = QtWidgets.QLabel()  # the last start's initial conditions: rewritten only on a change
        self.start_note.setWordWrap(True)
        mute(self.start_note)
        sl.addWidget(self.start_note)
        self._started: tuple[int, TwoMass] | None = None  # the last start and the system it started on

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
        lv.addWidget(start_box)
        lv.addWidget(self.trials)
        lv.addStretch(1)

        # --- center: the masses, the three planes, the run, the coordinates
        self.chain = ChainView()
        self.chain.setMinimumHeight(90)
        self.chain.setMaximumHeight(240)  # the planes are square: the height they leave is the masses'
        self.chain.grabbed.connect(self._on_mass_grabbed)
        self.chain.pulled.connect(self._on_mass_pulled)
        self.chain.released.connect(self._on_let_go)
        self.chain.grip_taken.connect(self._on_grip_taken)
        self.chain.grip_moved.connect(self._on_grip_moved)
        self.shape_plane = ShapePlane()
        self.shape_plane.grabbed.connect(self._on_grabbed)
        self.shape_plane.pulled.connect(self._on_point_pulled)
        self.shape_plane.released.connect(self._on_let_go)
        self.pair_planes = [PairPlane(i) for i in range(2)]
        for p in self.pair_planes:
            p.grabbed.connect(self._on_pair_grabbed)
            p.pulled.connect(self._on_pair_pulled)
            p.released.connect(self._on_let_go)
        planes = PlaneRow([self.shape_plane, *self.pair_planes])
        self.plane_note = LiveLabel(wrap=True)
        mute(self.plane_note)
        row, self.speed, self.run_button, self.reset_button, self.clock = run_row(self)
        self.run_button.toggled.connect(self._on_run)
        self.reset_button.setToolTip("<p>Back to rest at t = 0.</p>")
        self.reset_button.clicked.connect(self.reset)

        self.w_plot = plot("Coordinates (mm)", None, "Time", "s", height=120)
        self.w_plot.setToolTip(COORDS_TIP)
        self.w_plot.setMouseEnabled(x=False, y=False)
        self.w_legend = add_legend(self.w_plot.getPlotItem(), offset=(-10, 10))
        self.w_items = [pg.PlotDataItem() for _ in range(4)]
        for item in self.w_items:
            self.w_plot.addItem(item)
        self._legend_names: tuple[str, ...] = ()
        self.marks = ChangeMarks(self.w_plot)
        self.scrub = Scrub(self.w_plot)
        self.scrub.scrubbed.connect(self._on_scrubbed)
        self.readout = LiveLabel(wrap=True)

        centre = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(centre)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.chain, 2)
        cv.addWidget(planes)
        cv.addWidget(self.plane_note)
        cv.addLayout(row)
        cv.addWidget(self.w_plot, 2)
        cv.addWidget(self.readout)

        # --- right: Matrices, Sizes, Shapes, Theory
        matrices = QtWidgets.QWidget()
        mv = QtWidgets.QVBoxLayout(matrices)
        self.matrices = MatricesLabel()
        self.matrices.setToolTip(MATRICES_TIP)
        self.matrices_note = LiveLabel(wrap=True)
        mute(self.matrices_note)
        mv.addWidget(self.matrices)
        mv.addWidget(self.matrices_note)
        mv.addStretch(1)

        sizes = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(sizes)
        self.sizes_plot = plot("Size (mm)", None, "Time", "s", log_y=True, height=200)
        self.sizes_plot.setToolTip(SIZES_TIP)
        self.sizes_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.sizes_plot.getPlotItem(), offset=(-10, 10))
        names = ("mode 1 of M and K", "mode 2 of M and K", "state-space mode 1", "state-space mode 2")
        self.size_items = [pg.PlotDataItem(name=n) for n in names]
        for item in self.size_items:
            self.sizes_plot.addItem(item)
        self.sizes_plot.setYRange(np.log10(view.FLOOR * MM), np.log10(2 * view.REACH * MM), padding=0)
        self.sizes_note = LiveLabel(wrap=True)
        mute(self.sizes_note)
        sv.addWidget(self.sizes_plot, 1)
        sv.addWidget(self.sizes_note)

        shapes = QtWidgets.QWidget()
        hv = QtWidgets.QVBoxLayout(shapes)
        shape_row = QtWidgets.QHBoxLayout()
        self.phasor_plots: list[pg.PlotWidget] = []
        self.phasor_items: list[list[pg.PlotDataItem]] = []  # each mass's ψ, then φ, then the unit circle
        self.phasor_labels: list[list[pg.TextItem]] = []
        self.shape_notes: list[LiveLabel] = []
        for r in range(2):
            column = QtWidgets.QVBoxLayout()
            sp = pg.PlotWidget()
            _square(sp)
            sp.setToolTip(SHAPES_TIP)
            sp.setAccessibleName(f"Mode {r + 1}'s shape as phasors, a mass each")
            sp.setTitle(f"Mode {r + 1}")
            sp.setLabel("bottom", "Re")
            sp.setLabel("left", "Im")
            sp.setRange(QtCore.QRectF(-1.1, -1.1, 2.2, 2.2), padding=0)
            items = [pg.PlotDataItem() for _ in range(5)]
            for item in items:
                sp.addItem(item, ignoreBounds=True)
            texts = [pg.TextItem(f"m{'₁₂'[i]}", anchor=(0.0, 1.0)) for i in range(2)]
            for text in texts:
                sp.addItem(text, ignoreBounds=True)
            note = LiveLabel(wrap=True)
            mute(note)
            column.addWidget(sp)
            column.addWidget(note)
            shape_row.addLayout(column, 1)
            self.phasor_plots.append(sp)
            self.phasor_items.append(items)
            self.phasor_labels.append(texts)
            self.shape_notes.append(note)
        hv.addLayout(shape_row)
        hv.addStretch(1)

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(scrolling(matrices), "Matrices")
        self.tabs.addTab(sizes, "Sizes")
        self.tabs.addTab(shapes, "Shapes")
        self.tabs.addTab(self.notes, "Theory")

        three_panes(self, left, centre, self.tabs, sizes=(300, 780, 560))

        for field in SYSTEM_FIELDS:
            self.bindings.bind(field, getattr(self, field))
        for field, widget in (("basis", self.basis), ("tab", self.tabs)):
            self.bindings.bind(field, widget)
        self.bindings.write(StateTrial("", ""))
        self.bindings.changed.connect(self._on_field)

        self.loop = FrameLoop(self, "State-space modes", self._tick, self._runs, self._sim_t)
        self.apply_theme()
        self._push_system()

    # ------------------------------------------------------------- the state
    def practice_state(self) -> StateTrial:
        """The page's settings, as a trial: what a practice spec checks, and Defaults puts back."""
        return self.bindings.read(StateTrial("", ""))

    @property
    def system(self) -> TwoMass:
        return self.live.system

    @property
    def which(self) -> int:
        return self.basis.currentIndex()

    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(StateTrial("", "", **setup))

    def apply_trial(self, t: StateTrial) -> None:
        """A setup: the masses, springs, dampers and basis as it says, the run from rest, then started from `start`."""
        self.bindings.write(t)
        self._push_system()
        self.reset()
        self._handled_at = None
        self.trials.show_note(t.note)
        if t.start is not None:
            self.start(t.start)
        self._draw_tab()

    def start(self, which: int) -> None:
        """Start the masses from `view.STARTS[which]` and let them go."""
        x, v = view.start_state(self.live.system, which)
        self._started = (which, self.live.system)
        self._draw_start()
        self.scrub.hide()
        self.live.hold_state(x, v)
        self._on_let_go()

    # ------------------------------------------------------------- what the reader changes
    def _on_field(self, field: str) -> None:
        if field == "tab":
            self._draw_tab()
            return
        self._handled()
        if field == "basis":
            self._draw_still()
            if not self.running:
                self._draw_live()
        else:
            self._push_system()

    def _handled(self) -> None:
        if self._handled_at is None:
            self._handled_at = self.live.t

    def _push_system(self) -> None:
        """The live run to the controls: the masses, springs and dampers (a live change)."""
        self.live.set_system(view.system(self.practice_state()))
        s = self.live.system
        self.chain.set_values(s.m1, s.m2, s.k1, s.k2, s.k3, s.c)
        self._draw_still()
        if not self.running:
            self._draw_live()  # running, the next frame draws it

    def _on_start_clicked(self) -> None:
        self._handled()
        self.start(int(self.sender().property("start")))

    def _on_grip_taken(self, name: str) -> None:
        self._grip_start = getattr(self, name).value()

    def _on_grip_moved(self, name: str, distance: float) -> None:
        box = getattr(self, name)
        box.blockSignals(True)
        box.setValue(view.GRIPS[name].dragged(self._grip_start, distance))
        box.blockSignals(False)
        self._handled()
        self._push_system()

    def _on_grabbed(self) -> None:
        self.scrub.hide()
        self._handled()
        self._started = None  # a hand's start, not a button's
        self._draw_start()

    def _on_point_pulled(self, x1: float, x2: float) -> None:
        """The (x₁, x₂) plane's hand (mm): the masses held there at rest."""
        self.live.hold(np.array([x1, x2]) / MM)
        self._draw_body()

    def _on_pair_grabbed(self) -> None:
        self._on_grabbed()
        self._w_held = view.pairs(view.basis_of(self.system, self.which), self.live.x, self.live.v)[0]

    def _on_pair_pulled(self, a: float, b: float) -> None:
        plane = self.sender()
        assert isinstance(plane, PairPlane), "pulled by a pair's plane"
        T = view.basis_of(self.system, self.which)
        x, v = view.place(T, self._w_held, plane.index, a, b)
        self.live.hold_state(x, v)
        self._draw_body()

    def _on_mass_grabbed(self, _i: int) -> None:
        self._on_grabbed()

    def _on_mass_pulled(self, i: int, x: float) -> None:
        now = self.live.x.copy()
        now[i] = x
        self.live.hold(now)
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
        """Back to rest at t = 0, waiting for Start."""
        self.live.reset()
        self.scrub.hide()
        self._started = None
        self._draw_start()
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
        self._played()
        self._draw_live(frame=True)

    def _played(self) -> None:
        if self._handled_at is not None and self.live.t - self._handled_at >= PLAYED_AFTER:
            self.trials.set_played()

    def step(self, duration: float) -> None:
        """Run on by `duration` s of simulated time at once (tests), drawing as the frames would."""
        end = self.live.t + duration
        while self.live.t < end - 1e-9 and self.live.held is None:
            self.live.advance(min(0.2, end - self.live.t) + 1e-12)
        self._played()
        self._draw_live()

    # ------------------------------------------------------------- scrubbing
    def _show_scrub(self) -> None:
        r = self.live.record
        if r.size > 1:
            end = float(r.t[r.size - 1])
            self.scrub.show(end, max(float(r.t[0]), end - STRIP_SECONDS))

    def _on_scrubbed(self, t: float) -> None:
        r = self.live.record
        ts = r.t[: r.size]
        z = np.array([float(np.interp(t, ts, r.y[: r.size, COL[c]])) for c in ("x1", "x2", "v1", "v2")])
        self._draw_motion(z[:2], z[2:], end=t)
        self.clock.setText(f"t = {sig(t, 4)} s (paused at {sig(self.live.t, 4)} s)")

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.chain.apply_theme()
        self.shape_plane.apply_theme()
        for p in self.pair_planes:
            p.apply_theme()
        self.scrub.apply_theme()
        self.marks.apply_theme()
        dash = QtCore.Qt.PenStyle.DashLine
        # A pair's second coordinate thin, not dashed: a dash is stroked one at a time, so a long dashed curve
        # cost 12 ms a frame at 2× undamped (#478; mac_timings' "no damping" scenario).
        for k, item in enumerate(self.w_items):
            item.setPen(pg.mkPen(c.mass[k // 2], width=1 if k % 2 else 2.5))
        for k, item in enumerate(self.size_items):
            item.setPen(pg.mkPen(c.mass[k % 2], width=1 if k < 2 else 2.5))
        for items, texts in zip(self.phasor_items, self.phasor_labels):
            for i in range(2):
                items[i].setPen(pg.mkPen(c.mass[i], width=3))
                items[2 + i].setPen(pg.mkPen(c.grey, width=1.5, style=dash))
                texts[i].setColor(c.mass[i])
            items[4].setPen(pg.mkPen(c.faint, width=1))
        self._draw_still()
        self._draw_tab()

    def _draw_start(self) -> None:
        """Under the buttons: the last start's x and ẋ and its place on the planes (nothing once a hand moved them)."""
        started = self._started
        self.start_note.setVisible(started is not None)
        if started is not None:
            which, s = started  # as it was when it started: the masses or springs changed since don't rewrite it
            self.start_note.setText(view.start_note(s, which, self.which))

    @coalesce.coalesced
    def _draw_still(self, *_: object) -> None:
        """What moves only when the reader changes something: the shapes on the plane, the planes' labels, the
        matrices, the Shapes tab."""
        s, which = self.live.system, self.which
        xp = view.x_plane(s)
        self.shape_plane.set_still(xp)
        for plane, labels in zip(self.pair_planes, view.labels(s, which)):
            plane.set_labels(labels)
        m = view.matrices_view(s, which)
        self.plane_note.setText(xp.note)
        for i, button in enumerate(self.start_buttons):
            button.setToolTip(f"{TIPS['start']}<p>{view.STARTS[i]} now: {view.start_conditions(s, i)}</p>")
        self._draw_start()
        self.matrices.set_variants(self._matrices_html(m, True), self._matrices_html(m, False))
        self.matrices_note.setText(m.note)
        self.sizes_note.setText(view.sizes_note(s))
        circle = np.linspace(0.0, 2 * np.pi, 97)
        for items, texts, note, ph in zip(self.phasor_items, self.phasor_labels, self.shape_notes, view.shapes_view(s)):
            for i in range(2):
                a, b = ph.masses[i]
                items[i].setData([0.0, a], [0.0, b])
                items[2 + i].setData([0.0, ph.undamped[i][0]], [0.0, 0.0])
                texts[i].setPos(a * 1.08, b * 1.08 + 0.05)
            items[4].setData(np.cos(circle), np.sin(circle))
            note.setText(ph.note)
        self._name_coordinates()

    def _name_coordinates(self) -> None:
        """The coordinates plot's legend, from the planes' axes."""
        names = tuple(n for p in self.pair_planes for n in (p.labels.x, p.labels.y.split("  ")[0]))
        if names == self._legend_names:
            return
        self._legend_names = names
        self.w_legend.clear()
        for item, name in zip(self.w_items, names):
            self.w_legend.addItem(item, name)

    @staticmethod
    def _matrices_html(m: view.MatricesView, wide: bool = True) -> str:
        """The steps: M, C and K; A; T and T⁻¹AT (side by side if `wide`, else one above the other); the modes. Each matrix a table as the Substructuring page's are
        (`matrix_table`): its rows and columns labeled with their coordinates, its blocks tinted, a coupling term
        bold in color."""
        c = colors

        def one(g: view.Grid) -> str:
            title = f"{g.symbol}<br><span style='font-weight:normal'>{g.unit}</span>" if g.unit else g.symbol
            return table_html(
                g.entries, g.rows, g.cols, g.row_groups, g.col_groups, title, nowrap=True, marked=g.marked
            )

        def step(text: str) -> str:
            return f"<p style='color:{c.muted}; margin-top:8px; margin-bottom:2px'>{text}</p>"

        lines = "".join(f"<br>{line}" for line in m.modes)
        return (
            step("1. The system, in x₁ and x₂")
            + side_by_side(*(one(g) for g in m.given))
            + step("2. The state z = (x₁, x₂, ẋ₁, ẋ₂): ż = Az")
            + one(m.state)
            + step(
                "3. The basis's state vectors as T's columns, z = Tw (the last page's x = Vq, for the state). Then "
                "Tẇ = ATw, and ẇ alone takes T⁻¹: T⁻¹AT"
            )
            + (side_by_side(one(m.basis), one(m.product)) if wide else one(m.basis) + "<br>" + one(m.product))
            + f"<p>4. The modes, the same in every basis{lines}</p>"
        )

    @coalesce.coalesced(covers=("_draw_live",))
    def _draw_tab(self, *_: object) -> None:
        self._draw_live()

    def _since(self, t: np.ndarray, end: float | None) -> int:
        """Where the planes' paths start: TRAIL_SECONDS back, not past the last change."""
        stop_t = t[-1] if end is None else end
        start = int(np.searchsorted(t, stop_t - TRAIL_SECONDS))
        return max(start, int(np.searchsorted(t, self.live.changed_at if end is None else 0.0)))

    @coalesce.coalesced
    def _draw_live(self, frame: bool = False) -> None:
        """What moves: the masses, the planes, the coordinates, the readout, and the tab showing."""
        live = self.live
        t, y = live.record.window(STRIP_SECONDS)
        end = live.t
        x = y[:, COL["x1"] : COL["x2"] + 1]
        v = y[:, COL["v1"] : COL["v2"] + 1]
        T = view.basis_of(live.system, self.which)
        w = view.pairs(T, x, v) if t.size else np.empty((0, 4))
        for k, item in enumerate(self.w_items):
            item.setData(t, w[:, k]) if t.size else item.setData([], [])
        self.w_plot.setXRange(max(0.0, end - STRIP_SECONDS), max(end, STRIP_SECONDS), padding=0)
        self.marks.draw(live.edits, end - STRIP_SECONDS)
        self._draw_motion(live.x, live.v, t=t, y=y, w=w)
        self.clock.setText(f"t = {sig(end, 4)} s")
        self._show_readout()
        if self.tabs.currentIndex() == view.TAB_SIZES:
            sz = view.sizes(live.system, x, v)
            for item, data in zip(self.size_items, (*sz.undamped, *sz.state)):
                item.setData(t, data) if t.size else item.setData([], [])
            self.sizes_plot.setXRange(max(0.0, end - STRIP_SECONDS), max(end, STRIP_SECONDS), padding=0)

    def _draw_motion(
        self,
        x: np.ndarray,
        v: np.ndarray,
        end: float | None = None,
        t: np.ndarray | None = None,
        y: np.ndarray | None = None,
        w: np.ndarray | None = None,
    ) -> None:
        """The masses, the (x₁, x₂) plane and the pairs' planes at (x, v), with their paths up to `end` (now)."""
        live = self.live
        T = view.basis_of(live.system, self.which)
        if t is None or y is None or w is None:
            t, y = live.record.window(STRIP_SECONDS)
            w = view.pairs(T, y[:, COL["x1"] : COL["x2"] + 1], y[:, COL["v1"] : COL["v2"] + 1]) if t.size else None
        self.chain.show_at(x)
        now = view.pairs(T, x, v)[0]
        if t.size and w is not None:
            stop = t.size if end is None else int(np.searchsorted(t, end, side="right"))
            i0 = min(self._since(t[:stop] if stop else t, end), stop)
            trail_x = (y[i0:stop, COL["x1"]] * MM, y[i0:stop, COL["x2"]] * MM)
            trails = [(w[i0:stop, 2 * r], w[i0:stop, 2 * r + 1]) for r in range(2)]
        else:
            trail_x = (np.empty(0), np.empty(0))
            trails = [(np.empty(0), np.empty(0))] * 2
        self.shape_plane.show_motion(trail_x, (float(x[0]) * MM, float(x[1]) * MM))
        for r, plane in enumerate(self.pair_planes):
            plane.show_motion(trails[r], (float(now[2 * r]), float(now[2 * r + 1])))

    def _draw_body(self) -> None:
        self._draw_motion(self.live.x, self.live.v)

    def _show_readout(self) -> None:
        live = self.live
        x = live.x * MM
        if live.held is not None:
            self.readout.setText(f"Held at x = ({sig(x[0], 3)}, {sig(x[1], 3)}) mm: let go to run on from there.")
            return
        if live.record.size < 2:
            self.readout.setText("Start the masses from the left, drag a point on a plane, or pluck a mass.")
            return
        w = view.pairs(view.basis_of(live.system, self.which), live.x, live.v)[0]
        a, b = (float(np.hypot(w[2 * r], w[2 * r + 1])) for r in range(2))
        tiny = 1e-9 * max(a, b, 1e-300)  # a pair started at zero stays at rounding's size: shown as 0
        self.readout.setText(
            f"x = ({sig(x[0], 3)}, {sig(x[1], 3)}) mm; the pairs' sizes {sig(a if a > tiny else 0.0, 3)} and "
            f"{sig(b if b > tiny else 0.0, 3)} mm."
        )
