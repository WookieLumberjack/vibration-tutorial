"""Single- and two-plane balancing exercise for the Jeffcott rotor page.

The steps are those of a field balance: read the slow roll, read the 1X vector
as found at the balancing speed, add a trial weight and read again (in two
planes, a trial in each plane in turn), then put the correction weights on (the
trials taken off) and check. Each reading runs the simulation: the page ramps
to the speed, waits for the transient to die away and averages the 1X vector
over some revolutions. The vectors are kept at every station, so the probes can
be changed afterwards and the arithmetic redone.

Every angle on the tab counts one way, said on it (`CONVENTION`), in its tooltips and in the notes' *Which way
phase is counted*: counter-clockwise seen from beyond bearing B, the way the shaft turns. A reading's phase is a
lead from the keyphasor (arg V, as the solvers count it; a lag is clockwise), and a weight's angle is from the
keyphasor mark, the same way. The polar plot (`PolarGrid`) is the disk as the mark passes the keyphasor, so read at
the x probe a reading points at the high spot, and the weights (squares on its outer ring: the trial T, the
correction W, the heavy spot H while shown) sit where they are on the disk: below the critical, the correction lands
about opposite the reading. Under it, the Bode plot: amplitude and phase against speed, as found and now.

Working on it: `tests/test_rotor_balance.py` checks the arithmetic and that the correction lands opposite the
reading below the critical; `tests/test_gui_smoke.py` balances a mystery rotor through the tab.
"""

from __future__ import annotations

import cmath
import math
from typing import TypeGuard

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.rotor import PlaneWeight
from ..core.rotor_balance import PROBES, corrections, influence_matrix, weight, weight_parts
from ..core.rotor_simulator import RPM
from ..core.rotor_sweep import STATIONS, phase_degrees, probe, steady_vectors
from ..view_models.text import number as sig
from . import weak
from .layouts import FlowLayout, form_layout, group, plot_height, wrapped
from .panels import spin
from .polar_grid import PolarGrid
from .style import colors
from .tab_groups import Deferred
from .theming import add_legend, mute

DASH = QtCore.Qt.PenStyle.DashLine
CURVE_POINTS = 600
BALANCE_RPM = 1200.0
SLOW_RPM = 100.0
MYSTERY_RANGE = (10e-6, 60e-6)  # kg·m, the hidden unbalance's size
MYSTERY_SKEW = (0.05, 0.3)  # degrees, the hidden skew's size (two planes)
PLANE_SPACING = 40.0  # mm between the planes on the disk
SHAFT_PLANES = (200.0, 300.0)  # mm from bearing A, the planes on the shaft
ILL_CONDITIONED = 20.0  # cond(α) above which the planes are hard to tell apart
# The ways to balance: one weight on the disk, two in planes on the disk's faces, or two on the shaft.
MODES = ("One plane: the disk", "Two planes: the disk's faces", "Two planes on the shaft")
# Each reading: its symbol and the color it is drawn in (an attribute of colors, or an index into colors.mode).
READINGS = (
    ("slow", "Vₛ", "grey"),
    ("found", "V₀", 1),
    ("trial", "V₁", 3),
    ("trial2", "V₂", 4),
    ("verify", "V₂", "strong"),
)
# The weights drawn on the polar plot's outer ring: label and color, as READINGS'.
WEIGHT_MARKS: dict[str, tuple[str, str | int]] = {
    "trial": ("T", 3),
    "correction": ("W", "strong"),
    "heavy": ("H", "force"),
}
TRIALS = ("trial", "trial2")  # the readings with a trial weight in plane 1, plane 2
SECOND = "′"  # marks the second probe's readings

STEPS_TIP = (
    "<p>1. Turn slowly and read the runout (slow roll): a bowed shaft reads its bow here, and no unbalance "
    "shows yet.<br>2. Run at the balancing speed and read the 1X vector as found.<br>3. Add a known trial "
    "weight and read again: the change, per unit of weight, is the influence coefficient. With two planes, "
    "do this for each plane in turn (the first trial taken off): each gives a column of the influence "
    "matrix.<br>Last, take the trials off, put on the weights that cancel the readings (less their slow "
    "roll, if ticked), and read again to check.</p><p>Each reading ramps to the speed, waits for the "
    "transient to die away, and averages ten revolutions, at no less than 2× real time.</p>"
)
MYSTERY_TIP = (
    "<p>Give the disk an unbalance of unknown size at an unknown angle, and hide the heavy spot: find it by "
    "balancing. With two planes the disk is skewed too, by a hidden angle: a couple unbalance. "
    "<i>Reveal</i> shows them in the panel again.</p>"
)
CONVENTION = (
    "<b>Angles</b> count counter-clockwise, the way the shaft turns, seen from beyond bearing B. A reading's "
    "phase is a <b>lead</b> from the keyphasor: a lag turns it clockwise. A weight's angle is from the keyphasor "
    "mark on the shaft, the same way. Field instruments mostly count both as lag, clockwise: lag = 360° − lead "
    "(see the notes' <i>Which way phase is counted</i>)."
)
ANGLE_TIP = (
    "<p>Where the trial weight goes: its angle from the keyphasor mark on the shaft, counter-clockwise seen from "
    "beyond bearing B (the way the shaft turns), as every angle on this tab counts.</p>"
)
VECTOR_TIP = (
    "<p>The readings as vectors on a polar plot: the distance from the center is the amplitude (the rings), "
    "the angle the phase, counted as a lead, counter-clockwise from the keyphasor at 0°. The plot is the disk "
    "seen from beyond bearing B as the mark passes the keyphasor, so read at the x probe a vector points at the "
    "high spot, and the squares on the outer ring are weights where they sit on the disk: <b>T</b> the trial, "
    "<b>W</b> the correction, <b>H</b> the heavy spot as found (while shown). Below the critical the high spot is near "
    "the heavy spot, so the correction lands about opposite the reading; above it, the high spot swings round "
    "toward the far side. A y probe, a quarter turn on, reads 90° less.</p><p>The dashed lines from V₀ are "
    "the trial weights' effects. With two probes the second's readings are dashed, their names primed.</p>"
)
MODE_TIP = (
    "<p><b>One plane</b>: one weight on the disk, read at one probe.<br><b>Two planes on the disk</b>: "
    "weights on its two faces, a set distance apart, read at two probes. Together they make a force and "
    "a couple, so they can cancel the disk's unbalance and its skew at once, at every speed.<br><b>Two "
    "planes on the shaft</b>: weights clamped to the shaft wherever you choose. They cancel both readings "
    "at the balancing speed, but on a flexible shaft only there, unless the planes are close to the "
    "unbalance.</p>"
)


def _color(key: str | int) -> str:
    return colors.mode[key] if isinstance(key, int) else getattr(colors, key)


def _lead(v: complex) -> int:
    """A vector's angle in degrees, [0, 360), counter-clockwise: a reading's phase lead, a weight's from the mark."""
    return round(math.degrees(cmath.phase(v))) % 360  # whole degrees: 359.6 shows as 0, not 360


def _vec(v: complex) -> str:
    """A 1X reading as amplitude ∠ phase lead."""
    return f"{sig(abs(v) * 1e6)} µm ∠ {_lead(v):.0f}°"


class BalancingPanel(QtWidgets.QWidget):
    """The balancing tab: settings, the steps, the vectors, and the response before and after."""

    def __init__(self, page, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.page = page
        self.vectors: dict[str, np.ndarray | None] = {key: None for key, _, _ in READINGS}
        self.found_weights: tuple[complex, ...] = ()
        self.found_planes: tuple[PlaneWeight, ...] = ()  # plane weights on as found
        self.found_system = None  # the rotor as found, for the curve
        self.trial: complex | None = None  # the trial weight used (kg·m)
        self.planes_used: tuple[PlaneWeight, ...] = ()  # the planes the trials went in (two planes)
        self.correction: np.ndarray | None = None  # the correction weights put on (kg·m), one per plane
        self._rng = np.random.default_rng()
        self.drawing = Deferred(self, self._draw)  # drawn while the tab shows

        form = form_layout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(MODES)
        self.mode.setToolTip(MODE_TIP)
        self.spacing = spin(1.0, 1000.0, PLANE_SPACING, 0, " mm apart")
        self.spacing.setToolTip(
            "<p>The distance between the disk's two balancing planes, its faces: "
            "the further apart, the smaller the weights a couple needs.</p>"
        )
        self.plane_z = [spin(0.0, 1e4, z, 0, " mm") for z in SHAFT_PLANES]
        for box in self.plane_z:
            box.setToolTip("<p>Where the plane is along the shaft, from bearing A.</p>")
        self.plane_labels = [QtWidgets.QLabel(t) for t in ("at", "and")]
        row = FlowLayout()
        row.add(self.mode, self.spacing, group(*(w for pair in zip(self.plane_labels, self.plane_z) for w in pair)))
        form.addRow("Planes:", row)
        self.mode.setAccessibleName("Planes")
        self.mode.currentIndexChanged.connect(self._on_mode)
        for box in (self.spacing, *self.plane_z):
            box.valueChanged.connect(self._on_planes)

        self.station = QtWidgets.QComboBox()
        self.station.addItems(STATIONS)
        self.direction = QtWidgets.QComboBox()
        self.direction.addItems([f"{d} probe" for d in PROBES])
        self.station2 = QtWidgets.QComboBox()
        self.station2.addItems(STATIONS)
        self.station2.setCurrentIndex(len(STATIONS) - 1)  # bearing B
        self.direction2 = QtWidgets.QComboBox()
        self.direction2.addItems([f"{d} probe" for d in PROBES])
        for c in (self.station, self.direction, self.station2, self.direction2):
            c.currentIndexChanged.connect(self.refresh)
        self.second_label = QtWidgets.QLabel("and")
        self.second_label.setToolTip("The second probe's readings are drawn dashed, their names primed")
        row = FlowLayout()
        row.add(group(self.station, self.direction), group(self.second_label, self.station2, self.direction2))
        form.addRow("Probe:", row)
        self.station.setAccessibleName("Probe")
        self.direction.setAccessibleName("Probe direction")
        self.speed = spin(60.0, 10_000.0, BALANCE_RPM, 0, " rpm")
        self.speed.setToolTip("The speed the rotor is balanced at")
        self.slow_speed = spin(10.0, 10_000.0, SLOW_RPM, 0, " rpm")
        self.slow_speed.setToolTip(
            "<p>Slow roll: slow enough that the unbalance makes no force to speak of. Faster, the reading "
            "holds some unbalance response too, and subtracting it biases the correction.</p>"
        )
        self.subtract = QtWidgets.QCheckBox("Subtract slow roll")
        self.subtract.setChecked(True)
        self.subtract.setToolTip(
            "<p>Work out the correction from the readings less the slow-roll runout, which no weight "
            "removes. Untick to balance the raw reading to zero at the balancing speed instead.</p>"
        )
        self.subtract.toggled.connect(self.refresh)
        row = FlowLayout()
        row.add(group(self.speed, "slow roll", self.slow_speed), self.subtract)
        form.addRow("Balancing at:", row)
        self.speed.setAccessibleName("Balancing at")
        self.trial_mass = spin(0.01, 100.0, 0.5, 2, " g")
        self.trial_radius = spin(1.0, 500.0, 20.0, 0, " mm")
        self.trial_angle = spin(0.0, 359.0, 0.0, 0, "°")
        self.trial_angle.setWrapping(True)
        self.trial_angle.setStepType(QtWidgets.QAbstractSpinBox.StepType.DefaultStepType)
        self.trial_angle.setSingleStep(15.0)
        self.trial_angle.setToolTip(ANGLE_TIP)
        row = FlowLayout()
        row.add(group(self.trial_mass, "at radius", self.trial_radius), group("angle", self.trial_angle))
        form.addRow("Trial weight:", row)
        self.trial_mass.setAccessibleName("Trial weight")

        steps = FlowLayout()
        self._buttons = {key: QtWidgets.QPushButton() for key in ("slow", "found", "trial", "trial2", "verify")}
        actions = {
            "slow": self.measure_slow,
            "found": self.measure_found,
            "trial": weak.slot(self, lambda self: self.measure_trial(0)),
            "trial2": weak.slot(self, lambda self: self.measure_trial(1)),
            "verify": self.measure_verify,
        }
        for key, b in self._buttons.items():
            b.setToolTip(STEPS_TIP)
            b.clicked.connect(actions[key])
            steps.addWidget(b)
        self.mystery = QtWidgets.QPushButton("Mystery unbalance")
        self.mystery.setToolTip(MYSTERY_TIP)
        self.mystery.clicked.connect(self.new_mystery)
        self.reveal = QtWidgets.QPushButton("Reveal")
        self.reveal.setToolTip("Show the hidden unbalance in the panel again, and the heavy spot")
        self.reveal.clicked.connect(page.reveal_unbalance)
        self.remove = QtWidgets.QPushButton("Remove weights")
        self.remove.setToolTip("Take every balance weight off the rotor and start again")
        self.remove.clicked.connect(self.remove_weights)
        self.show_heavy = QtWidgets.QCheckBox("Show heavy spot")
        self.show_heavy.setChecked(True)
        self.show_heavy.setToolTip("Draw the heavy spot on the disk and its line on the disk's orbit")
        self.show_heavy.toggled.connect(page.set_heavy_shown)
        extras = FlowLayout()
        extras.add(self.mystery, self.reveal, self.remove, self.show_heavy)
        self.status = QtWidgets.QLabel()
        mute(self.status)

        self.convention = wrapped(CONVENTION)
        self.vector_plot = pg.PlotWidget()
        vp = self.vector_plot.getPlotItem()
        vp.setMenuEnabled(False)
        self.grid = PolarGrid(vp)
        self.vector_plot.setAccessibleName("Balancing readings and weights, polar plot")
        self.vector_plot.setToolTip(VECTOR_TIP)
        self.arrows, self.tips, self.labels = {}, {}, {}
        for key, sym, _ in READINGS:
            for k in range(2):
                self.arrows[key, k] = vp.plot()
                self.tips[key, k] = pg.ScatterPlotItem(size=8, symbol="o" if k == 0 else "d")
                # The slow roll's label goes under its tip: a bowed rotor balanced reads it again at the check.
                self.labels[key, k] = pg.TextItem(sym + SECOND * k, anchor=(0.5, -0.1) if key == "slow" else (0.5, 1.1))
                vp.addItem(self.tips[key, k])
                vp.addItem(self.labels[key, k])
        self.effects = {(key, k): vp.plot() for key in TRIALS for k in range(2)}
        # The weights where they sit on the disk, on the outer ring: the trial, the corrections, the heavy spot.
        self.weight_marks = {name: pg.ScatterPlotItem(size=11, symbol="s") for name in WEIGHT_MARKS}
        self.weight_labels = {name: pg.TextItem(text, anchor=(0.5, 0.5)) for name, (text, _) in WEIGHT_MARKS.items()}
        for name in WEIGHT_MARKS:
            vp.addItem(self.weight_marks[name])
            vp.addItem(self.weight_labels[name])

        self.curve_plot = pg.GraphicsLayoutWidget()
        cp = self.amp_plot = self.curve_plot.addPlot(row=0, col=0)
        pp = self.phase_plot = self.curve_plot.addPlot(row=1, col=0)
        cp.setLabel("left", "1X", units="m")
        pp.setLabel("left", "Phase, lead", units="°")
        pp.setLabel("bottom", "Speed (rpm)")
        pp.setXLink(cp)
        for plot in (cp, pp):
            plot.setMenuEnabled(False)
            plot.getAxis("bottom").enableAutoSIPrefix(False)
            plot.getAxis("left").enableAutoSIPrefix(plot is cp)
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.getAxis("left").setWidth(52)  # the two plots' speeds line up
        self.curve_plot.ci.layout.setRowStretchFactor(0, 3)
        self.curve_plot.ci.layout.setRowStretchFactor(1, 2)
        add_legend(cp, offset=(-5, 5))
        self.curve_found = [cp.plot(name="As found"), cp.plot()]
        self.curve_now = [cp.plot(name="Now"), cp.plot()]
        self.phase_found = [pp.plot(), pp.plot()]
        self.phase_now = [pp.plot(), pp.plot()]
        self.speed_lines = [pg.InfiniteLine(angle=90) for _ in range(4)]  # balancing, slow roll; on each plot
        for k, line in enumerate(self.speed_lines):
            (cp if k < 2 else pp).addItem(line)
        self.curve_plot.setToolTip(
            "<p>The steady-state 1X response at the probe against speed, as found (at step 2) and with the weights "
            "on now; the second probe's dashed. Top, its amplitude; bottom, its phase, counted as a lead from the "
            "keyphasor, as on the polar plot, so a lag is negative: it falls through −90° at a critical. The lines "
            "mark the balancing and slow-roll speeds.</p>"
        )
        self.results = wrapped()
        self.results.setTextFormat(QtCore.Qt.TextFormat.RichText)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(steps)
        layout.addLayout(extras)
        layout.addWidget(self.status)
        plot_height(self.vector_plot, 260)
        plot_height(self.curve_plot, 300)
        layout.addWidget(self.convention)
        layout.addWidget(self.vector_plot, 3)
        layout.addWidget(self.results)
        layout.addWidget(self.curve_plot, 2)
        for w in (self.speed, self.slow_speed):
            w.valueChanged.connect(self._place_speed_lines)
        self._show_mode()
        self.apply_theme()
        self._place_speed_lines()

    # ------------------------------------------------------------- theme
    def apply_theme(self) -> None:
        self.grid.apply_theme()
        for name, (_, c) in WEIGHT_MARKS.items():
            self.weight_marks[name].setBrush(pg.mkBrush(_color(c)))
            self.weight_marks[name].setPen(pg.mkPen(colors.background, width=1))
            self.weight_labels[name].setColor(_color(c))
        for key, _, c in READINGS:
            color = _color(c)
            for k in range(2):
                self.arrows[key, k].setPen(pg.mkPen(color, width=2.5 if k == 0 else 2, style=DASH if k else None))
                self.tips[key, k].setBrush(pg.mkBrush(color))
                self.tips[key, k].setPen(pg.mkPen(colors.background, width=1))
                self.labels[key, k].setColor(color)
        for (key, _), line in self.effects.items():
            line.setPen(pg.mkPen(_color(dict((r[0], r[2]) for r in READINGS)[key]), width=1.5, style=DASH))
        for k in range(2):
            style = DASH if k else QtCore.Qt.PenStyle.SolidLine
            for curve in (self.curve_found[k], self.phase_found[k]):
                curve.setPen(pg.mkPen(colors.grey, width=2, style=style))
            for curve in (self.curve_now[k], self.phase_now[k]):
                curve.setPen(pg.mkPen(colors.strong, width=2, style=style))
        self.amp_plot.legend.setBrush(colors.legend_brush())
        for k, line in enumerate(self.speed_lines):
            line.setPen(pg.mkPen(colors.force, width=1.5) if k % 2 == 0 else pg.mkPen(colors.grey, width=1, style=DASH))
        self.phase_plot.getAxis("left").setTickSpacing(90.0, 45.0)
        self.refresh()

    # ------------------------------------------------------------- planes and probes
    @property
    def two_planes(self) -> bool:
        return self.mode.currentIndex() > 0

    @property
    def step_buttons(self) -> list[QtWidgets.QPushButton]:
        """The steps' buttons, in order, for the planes chosen."""
        return [b for key, b in self._buttons.items() if key != "trial2" or self.two_planes]

    def planes(self) -> tuple[PlaneWeight, ...]:
        """The two balancing planes chosen (with no weight yet), for the rotor on the page."""
        s = self.page.sim.system
        if self.mode.currentIndex() == 1:
            h = self.spacing.value() / 2e3
            return PlaneWeight(s.a - h, 0j, True), PlaneWeight(s.a + h, 0j, True)
        return tuple(PlaneWeight(box.value() / 1e3, 0j) for box in self.plane_z)

    def probes(self) -> list[tuple[int, str]]:
        """(station, direction) of each probe in use: one for one plane, two for two."""
        pairs = [(self.station, self.direction), (self.station2, self.direction2)][: 2 if self.two_planes else 1]
        return [(s.currentIndex(), PROBES[d.currentIndex()]) for s, d in pairs]

    def _probe_name(self, k: int) -> str:
        station, direction = self.probes()[k]
        return f"{STATIONS[station]}, {direction}"

    def _show_mode(self) -> None:
        mode = self.mode.currentIndex()
        two = mode > 0
        self.spacing.setVisible(mode == 1)
        for w in (*self.plane_labels, *self.plane_z):
            w.setVisible(mode == 2)
        for w in (self.second_label, self.station2, self.direction2):
            w.setVisible(two)
        self._buttons["trial2"].setVisible(two)
        names = (
            ["1. Slow roll", "2. As found", "3. Trial, plane 1", "4. Trial, plane 2", "5. Correct and check"]
            if two
            else ["1. Slow roll", "2. As found", "3. Trial run", "", "4. Correct and check"]
        )
        for b, name in zip(self._buttons.values(), names):
            b.setText(name)
        for k in range(2):
            self.labels["verify", k].setText(("V₃" if two else "V₂") + SECOND * k)
        lo, hi = self.page.sim.system.extent
        for box in self.plane_z:
            box.blockSignals(True)
            box.setRange(lo * 1e3, hi * 1e3)
            box.blockSignals(False)

    def _forget_trials(self) -> None:
        """The trial runs belong to the planes they were made in: forget them when the planes change."""
        self.vectors.update(trial=None, trial2=None, verify=None)
        self.trial, self.planes_used, self.correction = None, (), None

    def _on_mode(self) -> None:
        self._forget_trials()
        self._show_mode()
        self.refresh()

    def _on_planes(self) -> None:
        if self.two_planes:
            self._forget_trials()
        self.refresh()

    def rotor_changed(self) -> None:
        """The rotor's length or disk moved: keep the shaft planes on the shaft."""
        self._show_mode()

    # ------------------------------------------------------------- readings
    def readings(self, key: str) -> np.ndarray | None:
        """A stored reading at each probe in use (complex), or None if not taken."""
        v = self.vectors[key]
        if v is None:
            return None
        return np.array([complex(probe(v[station], direction)) for station, direction in self.probes()])

    def reading(self, key: str, k: int = 0) -> complex | None:
        """A stored reading at probe k, or None if not taken."""
        v = self.readings(key)
        return None if v is None else complex(v[k])

    def _runout(self) -> np.ndarray | complex:
        slow = self.readings("slow")
        return slow if self.subtract.isChecked() and slow is not None else 0j

    def influence(self) -> np.ndarray | None:
        """α (probes x planes) from the trial runs, or None until they are all in."""
        keys = TRIALS if self.two_planes else TRIALS[:1]
        v0 = self.readings("found")
        runs = [r for r in (self.readings(k) for k in keys) if r is not None]
        if v0 is None or len(runs) < len(keys) or self.trial is None:
            return None
        if any(np.array_equal(r, v0) for r in runs):
            return None
        return influence_matrix(v0, np.array(runs), np.full(len(keys), self.trial))

    def correction_weights(self) -> np.ndarray | None:
        """W_c = -α⁻¹ (V₀ - V_s), one weight (kg·m) per plane, or None until the trial runs are in."""
        alpha, v0 = self.influence(), self.readings("found")
        if alpha is None or v0 is None:  # v0 is in once α is
            return None
        try:
            return corrections(v0, alpha, self._runout())
        except ValueError:
            return None

    # ------------------------------------------------------------- steps
    def _measure(self, rpm: float, key: str) -> None:
        def measured(vectors: np.ndarray) -> None:
            self.vectors[key] = vectors
            self.refresh()

        self.page.start_measurement(rpm / RPM, measured)
        self._enable(False)

    def _put_on(self, weights: np.ndarray) -> None:
        """Put these weights on, one per plane, beside the weights on as found."""
        if self.two_planes:
            planes = tuple(p._replace(weight=complex(w)) for p, w in zip(self.planes_used, weights) if w)
            self.page.set_weights(self.found_weights, self.found_planes + planes)
        else:
            self.page.set_weights(self.found_weights + (complex(weights[0]),), self.found_planes)

    def measure_slow(self) -> None:
        self._measure(self.slow_speed.value(), "slow")

    def measure_found(self) -> None:
        """Read as found, with whatever weights are on now; earlier trial and check readings go."""
        system = self.page.sim.system
        self.found_weights, self.found_planes, self.found_system = system.weights, system.plane_weights, system
        self._forget_trials()
        self._measure(self.speed.value(), "found")

    def measure_trial(self, plane: int = 0) -> None:
        """Run with the trial weight in this plane alone (0 or 1)."""
        r = self.trial_radius.value() / 1e3
        if plane == 0:
            self.trial = weight(self.trial_mass.value() / 1e3, r, math.radians(self.trial_angle.value()))
            self.planes_used = self.planes() if self.two_planes else ()
            self.vectors.update(trial2=None, verify=None)
        self.vectors["verify"], self.correction = None, None
        self._put_on(np.eye(2)[plane] * self.trial)
        self._measure(self.speed.value(), TRIALS[plane])

    def measure_verify(self) -> None:
        w = self.correction_weights()
        if w is None:
            return
        self.correction = w
        self._put_on(w)
        self._measure(self.speed.value(), "verify")

    def remove_weights(self) -> None:
        self.page.set_weights((), ())
        self.clear()

    def new_mystery(self) -> None:
        u = float(self._rng.uniform(*MYSTERY_RANGE))
        angle = float(self._rng.uniform(0.0, 2.0 * math.pi))
        skew = None
        if self.two_planes:
            skew = (math.radians(float(self._rng.uniform(*MYSTERY_SKEW))), float(self._rng.uniform(0.0, 2 * math.pi)))
        self.page.set_weights((), ())
        self.page.hide_unbalance(u, angle, skew)
        self.clear()

    def clear(self) -> None:
        """Forget the readings (the slow roll too) and the weights worked out."""
        self.vectors = {key: None for key, _, _ in READINGS}
        self.found_weights, self.found_planes, self.found_system = (), (), None
        self._forget_trials()
        self.refresh()

    def measurement_ended(self) -> None:
        self._enable(True)
        self.refresh()

    def _enable(self, on: bool) -> None:
        for b in (*self._buttons.values(), self.mystery, self.remove, self.mode):
            b.setEnabled(on)
        if on:
            self._buttons["trial"].setEnabled(self.vectors["found"] is not None)
            self._buttons["trial2"].setEnabled(self.vectors["trial"] is not None)
            self._buttons["verify"].setEnabled(self.correction_weights() is not None)

    def update_status(self) -> None:
        text = self.page.measurement_status()
        if text != self.status.text():
            self.status.setText(text)

    # ------------------------------------------------------------- drawing
    def catch_up(self) -> None:
        """Draw what changed while the tab was hidden (a window never shown gets no Show events)."""
        self.drawing.catch_up()

    def _place_speed_lines(self) -> None:
        for k, line in enumerate(self.speed_lines):
            line.setValue(self.speed.value() if k % 2 == 0 else self.slow_speed.value())
        self.refresh()

    def refresh(self) -> None:
        """Redraw the vectors, the results and the curves from the readings kept."""
        if self.page.measurement is None:
            self._enable(True)
        self.drawing.request()

    def _draw(self) -> None:
        values = {key: self.readings(key) for key, _, _ in READINGS}
        shown = {key for key, _, _ in READINGS if key != "trial2" or self.two_planes}
        n = len(self.probes())
        drawn = [abs(v[k]) for key, v in values.items() if v is not None and key in shown for k in range(n)]
        self.grid.fit(max(drawn, default=0.0))
        for key, _, _ in READINGS:
            for k in range(2):
                v = values[key]
                if v is None or key not in shown or k >= n:
                    self.arrows[key, k].setData([], [])
                    self.tips[key, k].setData([], [])
                    self.labels[key, k].setVisible(False)
                else:
                    self.arrows[key, k].setData([0.0, v[k].real], [0.0, v[k].imag])
                    self.tips[key, k].setData([v[k].real], [v[k].imag])
                    self.labels[key, k].setPos(v[k].real, v[k].imag)
                    self.labels[key, k].setVisible(True)
        v0 = values["found"]
        for (key, k), line in self.effects.items():
            v1 = values[key]
            if v0 is not None and v1 is not None and key in shown and k < n:
                line.setData([v0[k].real, v1[k].real], [v0[k].imag, v1[k].imag])
            else:
                line.setData([], [])
        self._draw_weights()
        self.results.setText(self._results_html(values))
        self._draw_curves()

    def weight_angles(self) -> dict[str, list[float]]:
        """The angles (rad, from the mark, counter-clockwise) of the weights drawn on the polar plot, by mark."""
        out: dict[str, list[float]] = {name: [] for name in WEIGHT_MARKS}
        if self.trial is not None:
            out["trial"].append(cmath.phase(self.trial))
        w = self.correction_weights()
        if w is not None:
            out["correction"] += [cmath.phase(complex(x)) for x in w if x]
        system = self.found_system or self.page.sim.system
        net = system.mass_unbalance
        if self.page.view.show_heavy and net:
            out["heavy"].append(cmath.phase(net))
        return out

    def _draw_weights(self) -> None:
        placed: list[float] = []  # the labels' angles so far: one near another steps inward
        for name, angles in self.weight_angles().items():
            points = [self.grid.ring_point(a) for a in angles]
            self.weight_marks[name].setData([x for x, _ in points], [y for _, y in points])
            label = self.weight_labels[name]
            label.setVisible(bool(points))
            if points:
                a = angles[0]
                near = sum(abs(math.remainder(a - b, 2 * math.pi)) < math.radians(12.0) for b in placed)
                r = (0.86 - 0.12 * near) * self.grid.radius
                label.setPos(r * math.cos(a), r * math.sin(a))
                placed.append(a)

    def _results_html(self, values: dict[str, np.ndarray | None]) -> str:
        two = self.two_planes
        runout = self._runout()
        subtract = np.any(runout != 0)
        names = {
            "slow": f"Slow roll ({self.slow_speed.value():,.0f} rpm)",
            "found": "As found",
            "trial": "Trial, plane 1" if two else "Trial run",
            "trial2": "Trial, plane 2",
            "verify": "Check",
        }
        rows = []
        for key, sym, c in READINGS:
            v = values[key]
            if v is None or (key == "trial2" and not two):
                continue
            if key == "verify" and two:
                sym = "V₃"
            cells = [_vec(x) for x in v]
            if subtract:
                cells += ["" if key == "slow" else _vec(x) for x in v - runout]
            rows.append(
                f"<tr><td><span style='color:{_color(c)}'><b>{sym}</b></span> {names[key]}</td>"
                + "".join(f"<td>{cell}</td>" for cell in cells)
                + "</tr>"
            )
        if not rows:
            return (
                "<small>Take the readings in order. The influence coefficients and the correction "
                "weights appear here. Try <i>Mystery unbalance</i> to balance a rotor whose heavy spot "
                "you cannot see.</small>"
            )
        if two:
            heads = [f"{self._probe_name(k)}{SECOND * k}" for k in range(2)]
            heads += [f"less slow roll{SECOND * k}" for k in range(2)] if subtract else []
        else:
            heads = ["1X (∠ lead)", "less slow roll" if subtract else ""]
        out = (
            "<table cellspacing='0' cellpadding='2'><tr><th align='left'>Reading</th>"
            + "".join(f"<th>{h}</th>" for h in heads)
            + "</tr>"
            + "".join(rows)
            + "</table>"
        )
        alpha = self.influence()
        r = self.trial_radius.value() / 1e3
        if alpha is not None:
            assert self.trial is not None, "α comes from the trial runs"
            mass, angle = weight_parts(self.trial, r)
            if two:
                out += self._matrix_html(alpha, mass, angle)
            else:
                a = complex(alpha[0, 0])
                out += (
                    f"Trial {sig(mass * 1e3)} g at {math.degrees(angle):.0f}° from the mark: influence coefficient "
                    f"α = (V₁ − V₀)/T = {_per_weight(a)}<br>"
                )
            w = self.correction_weights()
            less = "(V₀ − V<sub>s</sub>)" if subtract else "V₀"
            if w is None:
                out += "The trial runs cannot tell the planes apart: try planes further apart."
            elif two:
                parts = [f"W<sub>c{j + 1}</sub> = {_weight(x, r)}" for j, x in enumerate(w)]
                out += (
                    f"<b>Correction</b> W<sub>c</sub> = −α<sup>−1</sup>{less}: <b>{'; '.join(parts)}</b> (trials off)"
                )
            else:
                out += f"<b>Correction</b> W<sub>c</sub> = −{less}/α: <b>{_weight(complex(w[0]), r)}</b> from the mark (trial off)"
        v2, v0 = values["verify"], values["found"]
        if v2 is not None and v0_ok(v0, runout):
            share = [100 * abs(a - b) / abs(c - b) for a, b, c in zip(v2, np.broadcast_to(runout, v2.shape), v0)]
            kind = "compensated " if subtract else ""
            if two:
                where = "; ".join(f"{p:.1f}% at {self._probe_name(k)}" for k, p in enumerate(share))
                out += f"<br>Check: {where}, of the {kind}readings as found."
            else:
                out += f"<br>Check: {_vec(v2[0])}, {share[0]:.1f}% of the {kind}reading as found."
        return out

    def _matrix_html(self, alpha: np.ndarray, mass: float, angle: float) -> str:
        """The influence matrix, a column per plane, and how well it tells the planes apart."""
        cells = [[f"{sig(abs(a))} ∠ {_lead(a):.0f}°" for a in row] for row in alpha]
        cond = np.linalg.cond(alpha)
        note = (
            ": the planes are hard to tell apart, so the weights come out large and small errors in the "
            "readings swing them"
            if cond > ILL_CONDITIONED
            else ""
        )
        return (
            f"Trial {sig(mass * 1e3)} g at {math.degrees(angle):.0f}° from the mark in each plane in turn. Influence "
            f"matrix α (µm per g·mm ∠ lead), column j = (V<sub>j</sub> − V₀)/T: "
            f"[{cells[0][0]}, {cells[0][1]}; {cells[1][0]}, {cells[1][1]}], condition number {sig(cond)}{note}.<br>"
        )

    def _draw_curves(self) -> None:
        top = max(4000.0, 2.0 * self.speed.value())
        rpm = np.linspace(top / CURVE_POINTS, top, CURVE_POINTS)
        probes = self.probes()

        def response(system) -> list[np.ndarray]:
            v = steady_vectors(system, rpm / RPM)
            return [probe(v[:, station], direction) for station, direction in probes]

        now = response(self.page.sim.system)
        found: list[np.ndarray] = response(self.found_system) if self.found_system is not None else []
        for k in range(2):
            for values, amp, phase in (
                (now, self.curve_now[k], self.phase_now[k]),
                (found, self.curve_found[k], self.phase_found[k]),
            ):
                if k < len(values):
                    amp.setData(rpm, np.abs(values[k]))
                    phase.setData(rpm, phase_degrees(values[k]))
                else:
                    amp.setData([], [])
                    phase.setData([], [])


def v0_ok(v0: np.ndarray | None, runout) -> TypeGuard[np.ndarray]:
    """True if there is an as-found reading, compensated, to compare the check with."""
    return v0 is not None and bool(np.all(np.abs(v0 - runout) > 0))


def _per_weight(a: complex) -> str:
    """An influence coefficient, in µm per g·mm with its lag."""
    return f"{sig(abs(a))} µm per g·mm ∠ {_lead(a):.0f}°"


def _weight(w: complex, radius: float) -> str:
    """A weight as grams at the radius and an angle from the mark."""
    mass, angle = weight_parts(w, radius)
    return f"{sig(mass * 1e3)} g at {radius * 1e3:.0f} mm, {round(math.degrees(angle)) % 360}°"
