"""Bode and polar plots for the Jeffcott rotor page: the steady-state 1X response and tracked sweeps.

Both plots draw the steady-state 1X response (to the unbalance, skew and bow)
at one station, in one direction, against speed, with a marker at the current
speed, and over it the 1X vectors tracked once per revolution during run-ups
and coast-downs. A sweep pair (its run-up and its coast-down) shares a color:
the run-up is drawn solid, the coast-down dashed. The page's summary table under the plot is
their key, so the legend names only the steady state.

Either can show the 2X vector instead (set_order): an asymmetric shaft's weight
drives it, and its steady state is gravity's 2X response.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.rotor_simulator import RPM
from ..core.rotor_sweep import TrackedSweep, phase_degrees
from .polar_grid import PolarGrid
from .style import colors
from .theming import add_legend

DASH = QtCore.Qt.PenStyle.DashLine
SWEEP_COLORS = (1, 3, 5, 2, 4, 6, 7, 0)  # indices into colors.mode, one per sweep pair in turn

BODE_TIP = (
    "<p>Top: the 1X amplitude at the chosen station and direction against speed; bottom: its phase, counted "
    "as a <b>lead</b>, the way the shaft turns, from the keyphasor (the mark on the shaft passing +x). So a lag "
    "is negative: at −90° the probe's positive peak comes a quarter turn after the mark passes. (Instruments "
    "in the field mostly show the lag, +90° there: see the notes' <i>Which way phase is counted</i>.) The thick line is the steady state, each speed held until the response settles; the red line "
    "and dot are the speed now.</p><p>The thin lines are run-ups (solid) and coast-downs (dashed), "
    "tracked once per revolution like a tracking filter on a real machine. The dots, if shown, are the "
    "largest displacement over each revolution, free vibration included.</p>"
)
POLAR_TIP = (
    "<p>The same 1X vector drawn as a point: its distance from the center is the amplitude (the rings), and "
    "its angle the phase, counted as a <b>lead</b>: counter-clockwise from the keyphasor mark at 0°, the way "
    "the shaft turns (the disk is seen from beyond bearing B). A lag turns it clockwise. As the speed passes a "
    "critical, the steady-state vector traces a loop: very nearly a circle for a single lightly damped "
    "mode, though not exactly one (the mobility, velocity over force, would be). The phase is −90° (a lag "
    "of 90°) at the bottom of the loop, close to the critical speed.</p>"
    "<p>The labeled dots on the steady-state curve mark the speed in rpm.</p>"
)


@dataclass
class SweepCurves:
    """What one tracked leg draws: on the amplitude, phase and polar plots."""

    sweep: TrackedSweep
    color: int  # index into SWEEP_COLORS
    amp: pg.PlotDataItem
    peaks: pg.ScatterPlotItem
    phase: pg.PlotDataItem
    polar: pg.PlotDataItem
    drawn: int = 0  # revolutions drawn so far
    largest: float = 0.0  # its largest amplitude


def _major_grid(plot: pg.PlotItem) -> None:
    """A grid on the major ticks only. The marker moves every frame of a run-up, and each move
    repaints the axes, which draw the grid: the minor lines doubled the plots' paint."""
    plot.showGrid(x=True, y=True, alpha=0.15)
    for name in ("left", "bottom"):
        plot.getAxis(name).setStyle(maxTickLevel=0)


def _pen(sweep: TrackedSweep, color: int, width: float = 1.5) -> QtCore.QObject:
    c = colors.mode[SWEEP_COLORS[color % len(SWEEP_COLORS)]]
    return pg.mkPen(c, width=width, style=QtCore.Qt.PenStyle.SolidLine if sweep.up else DASH)


class BodePlots(pg.GraphicsLayoutWidget):
    """1X amplitude and phase (a lead: a lag is negative) against speed."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(BODE_TIP)
        self.amp = self.addPlot(row=0, col=0)
        self.amp.setLabel("left", "1X amplitude", units="m")
        self.phase = self.addPlot(row=1, col=0)
        self.phase.setLabel("left", "Phase (lead +, lag −)", units="°")
        self.phase.getAxis("left").enableAutoSIPrefix(False)
        self.phase.setLabel("bottom", "Spin speed", units="rpm")
        self.phase.getAxis("bottom").enableAutoSIPrefix(False)
        self.phase.setXLink(self.amp)
        for p in (self.amp, self.phase):
            _major_grid(p)
            p.setAutoVisible(y=True)
            p.setMenuEnabled(False)
        self.ci.layout.setRowStretchFactor(0, 3)
        self.ci.layout.setRowStretchFactor(1, 2)
        self.legend = add_legend(self.amp, offset=(-5, 5))
        self.steady_amp = self.amp.plot(name="Steady state")
        self.steady_phase = self.phase.plot()
        self.here_lines = [pg.InfiniteLine(angle=90) for _ in range(2)]
        self.here_dots = [pg.ScatterPlotItem(size=9) for _ in range(2)]
        self._here: tuple[float, float, float] | None = None
        for p, line, dot in zip((self.amp, self.phase), self.here_lines, self.here_dots):
            p.addItem(line)
            p.addItem(dot)
        self.apply_theme()

    def apply_theme(self) -> None:
        self.legend.setBrush(colors.legend_brush())
        for curve in (self.steady_amp, self.steady_phase):
            curve.setPen(pg.mkPen(colors.strong, width=2.5))
        for line, dot in zip(self.here_lines, self.here_dots):
            line.setPen(pg.mkPen(colors.force, width=1))
            dot.setBrush(pg.mkBrush(colors.force))
            dot.setPen(pg.mkPen(colors.background, width=1))
        self.phase.getAxis("left").setTickSpacing(90.0, 45.0)

    def set_order(self, order: int) -> None:
        """Label the plots for the 1X or the 2X vector."""
        self.amp.setLabel("left", f"{order}X amplitude", units="m")

    def set_steady(self, rpm: np.ndarray, values: np.ndarray, phase: np.ndarray) -> None:
        self.steady_amp.setData(rpm, np.abs(values))
        self.steady_phase.setData(rpm, phase)

    def set_here(self, rpm: float, amp: float, phase: float) -> None:
        if (rpm, amp, phase) == self._here:
            return  # at a steady speed: don't make the plots repaint every frame for nothing
        self._here = (rpm, amp, phase)
        for line in self.here_lines:
            line.setValue(rpm)
        self.here_dots[0].setData([rpm], [amp])
        self.here_dots[1].setData([rpm], [phase])


class PolarPlot(pg.PlotWidget):
    """The 1X vector as a point on a polar grid (`PolarGrid`): amplitude out from the center, phase (a lead)
    counter-clockwise from the keyphasor mark at +x."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(POLAR_TIP)
        p = self.getPlotItem()
        p.setMenuEnabled(False)
        self.grid = PolarGrid(p)
        self.setAccessibleName("Polar plot: the 1X vector, phase as a lead")
        self._steady_r = self._sweep_r = 0.0  # the largest amplitudes drawn: the rings hold both
        self.steady = p.plot()
        self.marks = pg.ScatterPlotItem(size=5)
        p.addItem(self.marks)
        self.mark_labels: list[pg.TextItem] = []
        self.here = pg.ScatterPlotItem(size=9)
        self._here: complex | None = None
        p.addItem(self.here)
        self.here.setZValue(10)
        self.apply_theme()

    def apply_theme(self) -> None:
        self.grid.apply_theme()
        self.steady.setPen(pg.mkPen(colors.strong, width=2.5))
        self.marks.setBrush(pg.mkBrush(colors.strong))
        self.marks.setPen(pg.mkPen(None))
        for label in self.mark_labels:
            label.setColor(colors.muted)
        self.here.setBrush(pg.mkBrush(colors.force))
        self.here.setPen(pg.mkPen(colors.background, width=1))

    @staticmethod
    def xy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Plot coordinates of 1X vectors V = |V| e^{i phase}: the phase (a lead) runs counter-clockwise from +x."""
        return np.real(values), np.imag(values)

    def set_steady(self, rpm: np.ndarray, values: np.ndarray, marks: np.ndarray) -> None:
        """The steady-state curve over the speeds shown, and dots labeled at the speeds `marks` (rpm)."""
        self.steady.setData(*self.xy(values))
        self._steady_r = float(np.abs(values).max()) if values.size else 0.0
        self.fit()
        p = self.getPlotItem()
        for label in self.mark_labels:
            p.removeItem(label)
        self.mark_labels = []
        if not rpm.size or not marks.size:
            self.marks.setData([], [])
            return
        at = np.interp(marks, rpm, values.real) + 1j * np.interp(marks, rpm, values.imag)
        self.marks.setData(*self.xy(at))
        for r, v in zip(marks, at):
            label = pg.TextItem(f"{r:,.0f}", anchor=(-0.1, 1.0))
            label.setPos(float(v.real), float(v.imag))
            p.addItem(label)
            self.mark_labels.append(label)
        self.apply_theme()

    def fit(self, sweep_r: float | None = None) -> None:
        """Rings to hold the steady state and the sweeps (the largest of their amplitudes, `sweep_r` if new)."""
        if sweep_r is not None:
            self._sweep_r = sweep_r
        self.grid.fit(max(self._steady_r, self._sweep_r))

    def set_here(self, value: complex) -> None:
        if value != self._here:  # unchanged at a steady speed: no repaint
            self._here = value
            self.here.setData([value.real], [value.imag])


class SweepPlots:
    """Draws tracked sweeps on a BodePlots and a PolarPlot, adding new revolutions as they arrive."""

    def __init__(self, bode: BodePlots, polar: PolarPlot) -> None:
        self.bode, self.polar = bode, polar
        self.curves: list[SweepCurves] = []
        self.show_peaks = True
        self.order = 1  # the harmonic drawn: 1X or 2X

    def add(self, sweep: TrackedSweep, color: int) -> None:
        c = SweepCurves(
            sweep, color, self.bode.amp.plot(), pg.ScatterPlotItem(size=4), self.bode.phase.plot(), self.polar.plot()
        )
        self.bode.amp.addItem(c.peaks)
        c.peaks.setVisible(self.show_peaks)
        self.curves.append(c)
        self._style(c)

    def clear(self) -> None:
        for c in self.curves:
            for plot, item in (
                (self.bode.amp, c.amp),
                (self.bode.amp, c.peaks),
                (self.bode.phase, c.phase),
                (self.polar.getPlotItem(), c.polar),
            ):
                plot.removeItem(item)
        self.curves = []

    def _style(self, c: SweepCurves) -> None:
        for item in (c.amp, c.phase, c.polar):
            item.setPen(_pen(c.sweep, c.color))
        c.peaks.setBrush(pg.mkBrush(colors.mode[SWEEP_COLORS[c.color % len(SWEEP_COLORS)]]))
        c.peaks.setPen(pg.mkPen(None))

    def apply_theme(self) -> None:
        for c in self.curves:
            self._style(c)

    def set_show_peaks(self, on: bool) -> None:
        self.show_peaks = on
        for c in self.curves:
            c.peaks.setVisible(on)

    def redraw(
        self, station: int, direction: str, steady_rpm: np.ndarray, steady_phase: np.ndarray, full: bool
    ) -> None:
        """Draw each sweep's revolutions; only the sweeps with new ones unless `full`."""
        for c in self.curves:
            if not full and len(c.sweep) == c.drawn:
                continue
            c.drawn = len(c.sweep)
            w, v, peak = c.sweep.trace(station, direction, self.order)
            rpm = w * RPM
            near = np.interp(rpm, steady_rpm, steady_phase) if steady_rpm.size else None
            c.amp.setData(rpm, np.abs(v))
            c.peaks.setData(rpm, peak)
            c.phase.setData(rpm, phase_degrees(v, near))
            c.polar.setData(*PolarPlot.xy(v))
            c.largest = float(np.abs(v).max()) if v.size else 0.0
        largest = max((c.largest for c in self.curves), default=0.0)
        self.polar.fit(largest)


def nice_marks(lo: float, hi: float, count: int = 8) -> np.ndarray:
    """Round speeds (rpm) strictly inside (lo, hi), about `count` of them."""
    if hi <= lo:
        return np.empty(0)
    raw = (hi - lo) / count
    step = 10 ** math.floor(math.log10(raw))
    step *= next(m for m in (1, 2, 2.5, 5, 10) if m * step >= raw)
    first = math.floor(lo / step) + 1
    marks = step * np.arange(first, math.ceil(hi / step))
    return marks[(marks > lo) & (marks < hi)]
