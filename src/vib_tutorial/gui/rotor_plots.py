"""The rotor page's orbit, time and Campbell plots, and the Campbell diagram's critical speeds."""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.rotor_modes import Campbell
from ..core.rotor_simulator import RPM
from ..core.rotor_stability import Onset
from ..core.rotor_sweep import (
    STATIONS,
)
from . import weak
from .lesson import Dot
from .rotor_notes import whirl_name
from .style import colors
from .theming import add_legend

DASH = QtCore.Qt.PenStyle.DashLine

ORDERS = {2.0: "2X", 3.0: "3X", 0.5: "½X"}


MIN_PEAK = 1e-9  # m; never magnify motion smaller than this


RESCALE = 0.04  # an orbit's range follows its peak, but only once it has moved this share


DAMPED = 0.3  # modes with more damping than this are drawn hollow and are not called critical


DOT = QtCore.Qt.PenStyle.DotLine


QWIDGETSIZE_MAX = (1 << 24) - 1  # Qt's "no maximum"


ORBIT_TIP = (
    "<p>The path of the shaft's centre at each station over the last few revolutions, looking from "
    "the B end toward A: x to the right, y up, and the shaft spinning counterclockwise (x toward y). "
    "Each plot has its own scale unless <i>Same scale</i> is ticked.</p>"
    "<p>The small hollow circles are once-per-revolution marks (a keyphasor): where the centre was "
    "each time the mark on the shaft passed +x (the heavy spot is at the mark unless you move it). On "
    "the disc's orbit the red line points at the heavy spot now; the angle from the high spot (where the disc is) to the heavy spot is the phase lag, "
    "0° well below the critical speed, about 90° near it and 180° well above.</p>"
)


def _clip_title(label: pg.LabelItem) -> None:
    """Let a plot's title be cut off when the plot is narrow, rather than set how narrow it can be."""
    update_min = label.updateMin

    def height_only() -> None:  # LabelItem sets its least size to its text's whenever it lays out
        update_min()
        label.setMinimumWidth(0)

    label.updateMin = height_only


class OrbitPlots(pg.GraphicsLayoutWidget):
    """The orbit at each station, seen from the B end; the disc's with its heavy spot."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(ORBIT_TIP)
        self.plots: list[pg.PlotItem] = []
        self.trails: list[pg.PlotDataItem] = []
        self.dots: list[Dot] = []
        self.keys: list[pg.ScatterPlotItem] = []
        self.spans = [MIN_PEAK] * len(STATIONS)
        self._shown_spans = [0.0] * len(STATIONS)  # the ranges set, which move only past RESCALE
        self.same_scale = False
        for i, _name in enumerate(STATIONS):
            p = self.addPlot(row=0, col=i)
            p.setAspectLocked(True)
            p.setMouseEnabled(x=False, y=False)
            p.hideButtons()
            p.setMenuEnabled(False)
            _clip_title(p.titleLabel)
            p.setLabel("bottom", "x", units="m")
            p.setLabel("left", "y", units="m")
            # Not narrowing the axes when the labels do: as the orbits rescale each frame, a
            # narrower label would re-lay out the plots (and paint them again).
            p.getAxis("bottom").setStyle(maxTickLevel=0, autoReduceTextSpace=False)
            p.getAxis("left").setStyle(maxTickLevel=0, autoReduceTextSpace=False)
            for angle in (0, 90):
                line = pg.InfiniteLine(pos=0, angle=angle)
                p.addItem(line)
            self.trails.append(p.plot())
            self.keys.append(pg.ScatterPlotItem(size=6, symbol="o"))
            self.dots.append(Dot(8))
            p.addItem(self.keys[-1])
            p.addItem(self.dots[-1], ignoreBounds=True)
            self.plots.append(p)
            self.ci.layout.setColumnStretchFactor(i, 1)  # equal shares, whatever the width
        self.heavy = self.plots[1].plot()
        self.ring = self.plots[1].plot()  # a stator ring's clearance round the disc
        self.apply_theme()
        self._fit_width()

    def apply_theme(self) -> None:
        for i, p in enumerate(self.plots):
            p.setTitle(f"<span style='color:{colors.mass[i]}'><b>{self._title(i)}</b></span>", size="9pt")
            for item in p.items:
                if isinstance(item, pg.InfiniteLine):
                    item.setPen(pg.mkPen(colors.faint, width=1))
            self.trails[i].setPen(pg.mkPen(colors.mass[i], width=1))
            self.dots[i].setBrush(pg.mkBrush(colors.mass[i]))
            self.dots[i].setPen(pg.mkPen(colors.foreground, width=1))
            self.keys[i].setBrush(pg.mkBrush(None))
            self.keys[i].setPen(pg.mkPen(colors.foreground, width=1))
        self.heavy.setPen(pg.mkPen(colors.force, width=2))
        self.ring.setPen(pg.mkPen(colors.structure, width=2))

    def _title(self, i: int) -> str:
        if i == 1 and not self.plots[2].isVisible():
            return "Disc (midspan)"
        return STATIONS[i]

    def set_midspan_shown(self, shown: bool) -> None:
        """Hide the midspan orbit when the disc is there."""
        if shown == self.plots[2].isVisible():
            return
        self.plots[2].setVisible(shown)
        layout = self.ci.layout
        layout.setColumnStretchFactor(2, 1 if shown else 0)
        # The plot's own width, not the column's: a column is never narrower than what's in it.
        self.plots[2].setMaximumWidth(QWIDGETSIZE_MAX if shown else 0)
        self.apply_theme()
        self._fit_width()

    def _fit_width(self) -> None:
        """At least as wide as the plots can be: a view narrower than its scene's layout cuts the last off."""
        self.ci.layout.invalidate()  # the sizes it holds are from before the change
        least = self.ci.effectiveSizeHint(QtCore.Qt.SizeHint.MinimumSize).width()
        self.setMinimumWidth(math.ceil(least) + 2 * self.frameWidth())

    def update_orbits(
        self,
        trails: list[np.ndarray],
        keys: list[np.ndarray],
        heavy_angle: float | None,
        centre: tuple[float, float] = (0.0, 0.0),
        ring: tuple[float, float, float] | None = None,
    ) -> None:
        """Each station's trail (k, 2), its keyphasor points, and the heavy spot's angle now, drawn from
        `centre` on the disc's orbit (where it whirls about: its sag, if drawn), with a stator `ring`'s
        clearance round the disc (centre x, y and radius)."""
        peaks = [float(np.abs(xy).max()) if xy.size else 0.0 for xy in trails]
        if ring is not None:  # drawn, but not taken into the range: the orbit, not the ring, sets the scale
            t = np.linspace(0.0, 2.0 * math.pi, 97)
            self.ring.setData(ring[0] + ring[2] * np.cos(t), ring[1] + ring[2] * np.sin(t))
        elif self.ring.xData is not None and self.ring.xData.size:
            self.ring.setData([], [])
        if self.same_scale:
            shown = [p for i, p in enumerate(peaks) if self.plots[i].isVisible()]
            peaks = [max(shown)] * len(peaks)
        for i, p in enumerate(self.plots):
            xy = trails[i]
            target = 1.15 * max(peaks[i], MIN_PEAK)
            span = self.spans[i]
            self.spans[i] = span = target if target > span else span + 0.1 * (target - span)
            if xy.size:
                self.trails[i].setData(xy[:, 0], xy[:, 1])
                self.dots[i].setData(xy[-1:, 0], xy[-1:, 1])
            else:
                self.trails[i].setData([], [])
                self.dots[i].setData([], [])
            k = keys[i]
            self.keys[i].setData(k[:, 0], k[:, 1]) if k.size else self.keys[i].setData([], [])
            if abs(span - self._shown_spans[i]) > RESCALE * span:  # a new range re-lays out the axes and repaints
                self._shown_spans[i] = span
                p.setRange(xRange=(-span, span), yRange=(-span, span), padding=0)
        if heavy_angle is None:
            self.heavy.setData([], [])
        else:
            r = 0.9 * self.spans[1]
            x0, y0 = centre
            self.heavy.setData([x0, x0 + r * math.cos(heavy_angle)], [y0, y0 + r * math.sin(heavy_angle)])


class TimePlots(pg.GraphicsLayoutWidget):
    """x and y at one station against time, and the speed."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.x_plot = self.addPlot(row=0, col=0)
        self.x_plot.setLabel("left", "Displacement", units="m")
        add_legend(self.x_plot, offset=(-5, 5), colCount=2)
        self.w_plot = self.addPlot(row=1, col=0)
        self.w_plot.setLabel("left", "Speed", units="rpm")
        self.w_plot.getAxis("left").enableAutoSIPrefix(False)
        self.w_plot.setLabel("bottom", "Time", units="s")
        self.w_plot.setXLink(self.x_plot)
        self.ci.layout.setRowStretchFactor(0, 2)
        self.ci.layout.setRowStretchFactor(1, 1)
        # As the other strip charts: no grid, 1 px pens, clipped and downsampled.
        for p in (self.x_plot, self.w_plot):
            p.setClipToView(True)
            p.setDownsampling(auto=True, mode="peak")
            p.setMouseEnabled(x=False, y=False)
            p.hideButtons()
        self.x_curve = self.x_plot.plot(name="x (horizontal)")
        self.y_curve = self.x_plot.plot(name="y (vertical)")
        self.w_curve = self.w_plot.plot()
        self.crit_lines: list[pg.InfiniteLine] = []
        self.apply_theme()

    def apply_theme(self) -> None:
        self.x_plot.legend.setBrush(colors.legend_brush())
        self.x_curve.setPen(pg.mkPen(colors.mode[1], width=1))
        self.y_curve.setPen(pg.mkPen(colors.mode[3], width=1))
        self.w_curve.setPen(pg.mkPen(colors.strong, width=1))
        for line in self.crit_lines:
            line.setPen(pg.mkPen(colors.force, width=1, style=DOT))

    def set_criticals(self, rpms: list[float]) -> None:
        for line in self.crit_lines:
            self.w_plot.removeItem(line)
        self.crit_lines = [pg.InfiniteLine(pos=r, angle=0) for r in rpms]
        for line in self.crit_lines:
            self.w_plot.addItem(line)
        self.apply_theme()

    def update_curves(self, t: np.ndarray, xy: np.ndarray, rpm: np.ndarray, window: float) -> None:
        self.x_curve.setData(t, xy[:, 0])
        self.y_curve.setData(t, xy[:, 1])
        self.w_curve.setData(t, rpm)
        if t.size:
            x0 = max(t[-1] - window, 0.0) if t[-1] > window else 0.0
            self.x_plot.setXRange(x0, x0 + window, padding=0)
            span = max(float(np.abs(xy).max()), MIN_PEAK)
            self.x_plot.setYRange(-span, span, padding=0.05)
            top = max(float(rpm.max()), *(line.value() for line in self.crit_lines[:1]), 100.0)
            self.w_plot.setYRange(0.0, 1.1 * top, padding=0)


class CampbellPlot(pg.GraphicsLayoutWidget):
    """Whirl frequencies against speed: forward and backward branches, the 1X line, the criticals."""

    speed_dragged = QtCore.Signal(float)  # rpm

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.plot = p = self.addPlot(row=0, col=0)
        p.setLabel("bottom", "Spin speed", units="rpm")
        p.setLabel("left", "Whirl frequency", units="Hz")
        p.getAxis("bottom").enableAutoSIPrefix(False)
        p.showGrid(x=True, y=True, alpha=0.15)
        p.setMenuEnabled(False)
        # The legend sits below the plot: inside, it would hide a branch wherever it went.
        self.legend = pg.LegendItem(colCount=2, brush=colors.legend_brush(), labelTextColor=colors.foreground)
        self.addItem(self.legend, row=1, col=0)
        self.points: dict[str, pg.ScatterPlotItem] = {}
        # Backward drawn first and larger: where the two branches coincide (an axisymmetric
        # rotor not spinning, or a mode the spin cannot split) both still show.
        for key, name, size in (
            ("bw", "backward whirl", 7),
            ("fw", "forward whirl", 3),
            ("planar", "straight-line whirl", 4),
        ):
            self.points[key] = pg.ScatterPlotItem(size=size, name=name)
            p.addItem(self.points[key])
        self.damped = pg.ScatterPlotItem(size=3, name=f"heavily damped (ζ > {DAMPED:g})")
        p.addItem(self.damped)
        self.sync = p.plot(name="1X (synchronous)")
        # The same rotor with a massless shaft, when its shaft has mass: one curve per branch.
        self.reference: list[pg.PlotDataItem] = []
        self.reference_sample = pg.PlotDataItem(name="massless shaft (Jeffcott)")
        # Other orders of the speed, drawn on request: lines but not in the legend, labelled at their ends.
        self.orders = {order: p.plot() for order in ORDERS}
        self.order_labels = {order: pg.TextItem(f"{name}", anchor=(1.1, 1.0)) for order, name in ORDERS.items()}
        for order in ORDERS:
            self.orders[order].setVisible(False)
            self.order_labels[order].setVisible(False)
            p.addItem(self.order_labels[order])
        self.uncrit = pg.ScatterPlotItem(size=11, symbol="o", name="backward crossing, not driven")
        p.addItem(self.uncrit)
        self.crit = pg.ScatterPlotItem(size=11, symbol="o", name="critical speed")
        p.addItem(self.crit)
        self.crit_labels: list[pg.TextItem] = []
        self.unstable = pg.ScatterPlotItem(size=7, symbol="x", name="growing (ζ &lt; 0)")
        p.addItem(self.unstable)
        self.onset_line = pg.InfiniteLine(angle=90)
        self.end_line = pg.InfiniteLine(angle=90)  # where an unstable band ends (an asymmetric shaft's)
        self.onset_label = pg.TextItem(anchor=(1.05, 0.0))
        p.addItem(self.onset_line)
        p.addItem(self.end_line)
        p.addItem(self.onset_label)
        self.speed_line = pg.InfiniteLine(angle=90, movable=True)
        self.speed_line.setToolTip("The speed now. Drag it to set the target speed.")
        self.speed_line.sigPositionChangeFinished.connect(
            weak.slot(self, lambda self, line: self.speed_dragged.emit(max(0.0, line.value())))
        )
        p.addItem(self.speed_line)
        for item in (*self.points.values(), self.damped, self.sync, self.crit, self.uncrit, self.unstable):
            self.legend.addItem(item, item.name())
        # The speed line moves every frame during a run-up, and repainted the thousands of dots under it;
        # cached, they are drawn again only when the analysis or the plot's range changes.
        for item in (*self.points.values(), self.damped):
            item.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.result: Campbell | None = None
        self._reference_shown = False
        self.apply_theme()

    def apply_theme(self) -> None:
        brushes = {"fw": colors.mode[1], "bw": colors.mode[3], "planar": colors.grey}
        for key, item in self.points.items():
            item.setBrush(pg.mkBrush(brushes[key]))
            item.setPen(pg.mkPen(None))
        self.damped.setBrush(pg.mkBrush(None))
        self.damped.setPen(pg.mkPen(colors.grey, width=1))
        self.sync.setPen(pg.mkPen(colors.strong, width=1, style=DASH))
        for line in (*self.reference, self.reference_sample):
            line.setPen(pg.mkPen(colors.muted, width=1.5, style=DASH))
        for order, line in self.orders.items():
            line.setPen(pg.mkPen(colors.muted, width=1, style=DOT))
            self.order_labels[order].setColor(colors.muted)
        self.crit.setBrush(pg.mkBrush(None))
        self.crit.setPen(pg.mkPen(colors.force, width=2))
        self.uncrit.setBrush(pg.mkBrush(None))
        self.uncrit.setPen(pg.mkPen(colors.grey, width=2))
        self.speed_line.setPen(pg.mkPen(colors.force, width=2))
        self.speed_line.setHoverPen(pg.mkPen(colors.force, width=4))
        self.unstable.setBrush(pg.mkBrush(colors.force))
        self.unstable.setPen(pg.mkPen(None))
        self.onset_line.setPen(pg.mkPen(colors.force, width=1.5, style=DASH))
        self.end_line.setPen(pg.mkPen(colors.force, width=1.5, style=DASH))
        self.onset_label.setColor(colors.force)
        for label in self.crit_labels:
            label.setColor(colors.force)

    def set_result(self, result: Campbell, max_rpm: float, isotropic: bool, onset: Onset | None = None) -> None:
        self.result = result
        rpm = np.repeat(result.omega * RPM, result.freq_hz.shape[1])
        f = result.freq_hz.ravel()
        whirl = result.whirl.ravel()
        damped = result.zeta.ravel() > DAMPED
        ok = np.isfinite(f)
        classes = {"fw": whirl > 0.1, "bw": whirl < -0.1}
        classes["planar"] = ~(classes["fw"] | classes["bw"])
        for key, mask in classes.items():
            m = mask & ok & ~damped
            self.points[key].setData(rpm[m], f[m])
        self.damped.setData(rpm[ok & damped], f[ok & damped])
        growing = ok & (result.zeta.ravel() < 0.0)
        self.unstable.setData(rpm[growing], f[growing])
        self.onset_line.setVisible(onset is not None)
        self.end_line.setVisible(onset is not None and math.isfinite(onset.end))
        if onset is not None and math.isfinite(onset.end):
            self.onset_line.setValue(onset.omega * RPM)
            self.end_line.setValue(onset.end * RPM)
            self.onset_label.setText(f"unstable\n{onset.omega * RPM:,.0f}–{onset.end * RPM:,.0f} rpm")
        elif onset is not None:
            self.onset_line.setValue(onset.omega * RPM)
            self.onset_label.setText(f"instability onset\n{onset.omega * RPM:,.0f} rpm")
        else:
            self.onset_label.setText("")
        self.sync.setData([0.0, max_rpm], [0.0, max_rpm / 60.0])
        for order, line in self.orders.items():
            line.setData([0.0, max_rpm], [0.0, order * max_rpm / 60.0])
        p = self.plot
        for label in self.crit_labels:
            p.removeItem(label)
        driven = critical_speeds(result, isotropic)
        crits = [w * RPM for w, _ in driven]
        self.crit.setData(crits, [c / 60.0 for c in crits])
        others = [w * RPM for w, r in lightly_damped(result) if (w, r) not in driven]
        self.uncrit.setData(others, [c / 60.0 for c in others])
        self.crit_labels = []
        groups = _grouped(critical_speeds(result, isotropic))
        for i, (crit_rpm, names) in enumerate(groups):
            # Right of the circle and above it, unless the next one is close: then below it.
            crowded = i + 1 < len(groups) and groups[i + 1][0] - crit_rpm < 0.3 * max_rpm
            label = pg.TextItem(
                f"{crit_rpm:,.0f} rpm ({' + '.join(names)})", anchor=(-0.05, -0.1) if crowded else (-0.05, 1.1)
            )
            label.setPos(crit_rpm, crit_rpm / 60.0)
            p.addItem(label)
            self.crit_labels.append(label)
        top = np.nanmax(f[ok & (f < 2.5 * max_rpm / 60.0)]) if np.any(ok & (f < 2.5 * max_rpm / 60.0)) else 1.0
        y_top = 1.1 * max(top, max_rpm / 60.0)
        p.setRange(xRange=(0.0, max_rpm), yRange=(0.0, y_top), padding=0)
        for order, label in self.order_labels.items():
            end = min(max_rpm, 60.0 * y_top / order)  # where the line leaves the plot
            label.setPos(end, order * end / 60.0)
            label.setAnchor((1.1, 0.0) if end < max_rpm else (1.1, 1.0))  # out the top: hang below
        if onset is not None:
            self.onset_label.setAnchor((1.05, 0.0) if onset.omega * RPM > 0.5 * max_rpm else (-0.05, 0.0))
            self.onset_label.setPos(onset.omega * RPM, 0.97 * y_top)
        self.apply_theme()

    def set_reference(self, result: Campbell | None) -> None:
        """Draw another Campbell diagram's lightly damped branches as dashed curves (None: none)."""
        p = self.plot
        branches = (
            [] if result is None else [j for j in range(result.freq_hz.shape[1]) if np.any(result.zeta[:, j] <= DAMPED)]
        )
        while len(self.reference) < len(branches):
            line = p.plot()
            line.setZValue(-1)
            self.reference.append(line)
        for i, line in enumerate(self.reference):
            if i < len(branches):
                assert result is not None, "branches are only found in a result"
                j = branches[i]
                f = np.where(result.zeta[:, j] <= DAMPED, result.freq_hz[:, j], np.nan)
                line.setData(result.omega * RPM, f, connect="finite")
            line.setVisible(i < len(branches))
        if (result is not None) != self._reference_shown:
            if result is not None:
                self.legend.addItem(self.reference_sample, self.reference_sample.name())
            else:
                self.legend.removeItem(self.reference_sample)
            self._reference_shown = result is not None
        self.apply_theme()

    def set_orders(self, shown: set[float]) -> None:
        """Show the lines of these orders of the speed (keys of ORDERS)."""
        for order in ORDERS:
            self.orders[order].setVisible(order in shown)
            self.order_labels[order].setVisible(order in shown)

    def set_speed(self, rpm: float) -> None:
        if not self.speed_line.moving:
            self.speed_line.setValue(rpm)


def lightly_damped(result: Campbell) -> list[tuple[float, float]]:
    """(speed in rad/s, whirl ratio) of the 1X crossings whose mode is damped less than DAMPED."""
    return [(w, r) for w, r, z in result.criticals if z <= DAMPED]


def critical_speeds(result: Campbell, isotropic: bool) -> list[tuple[float, float]]:
    """The 1X crossings the unbalance drives: lightly damped, and forward if the rotor is isotropic."""
    return [(w, r) for w, r in lightly_damped(result) if not isotropic or r > 0.1]


def _grouped(crits: list[tuple[float, float]]) -> list[tuple[float, list[str]]]:
    """Critical speeds (rpm) with the whirl names of those within 1% of each other merged."""
    out: list[tuple[float, list[str]]] = []
    for w, r in crits:
        rpm = w * RPM
        if out and rpm < 1.01 * out[-1][0]:
            out[-1][1].append(whirl_name(r))
        else:
            out.append((rpm, [whirl_name(r)]))
    return out
