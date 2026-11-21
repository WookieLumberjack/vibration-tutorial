"""Waterfall tab of the Jeffcott rotor page: full spectra stacked against speed or time.

Each line is the full spectrum of one block of the orbit at a station (see
core/rotor_waterfall.py), drawn at a height for its speed (a cascade) or its
time (a waterfall), with its amplitude rising above that baseline. Lines are
filled down to their baseline in the background colour, front to back, so a
tall peak hides the lines behind it as on an analyser's 3D plot. On the base of
each line, the +1X line and the free whirl frequencies at its speed are marked:
where the peaks fall on them, the spectrum is the unbalance or a free mode.

`WaterfallTab` is the tab itself (split from `rotor.py`, #218 C2): its controls, this plot, its note and the
spectra (`core/rotor_waterfall.py`'s `Waterfall`), which the page feeds and draws while the tab shows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.rotor import RotorSystem
from ..core.rotor_simulator import RPM
from ..core.rotor_sweep import STATIONS
from ..core.rotor_waterfall import SpectrumBlock, Waterfall
from ..view_models.text import number as sig
from . import weak
from .layouts import FlowLayout, group
from .panels import spin
from .rotor_readouts import fmt_len
from .rotor_tips import BLOCK_TIP
from .style import colors
from .theming import mute

DOT = QtCore.Qt.PenStyle.DotLine
MAX_LINES = 120  # at most this many spectra are drawn; more are thinned evenly
DECADES = 3.0  # the log scale's range: the floor is this many decades below the tallest line
AGAINST = ("Speed: latest run-up", "Speed: latest coast-down", "Time")
SCALES = ("Log (60 dB)", "Linear")
WATERFALL_RANGE = 2.0  # the waterfall keeps frequencies up to this many times the top speed

WATERFALL_TIP = (
    "<p>Each line is the full spectrum of the orbit over one block of time: forward whirl to the right, "
    "backward to the left, as on the <i>Stability</i> tab. Against speed the lines make a <i>cascade</i>, "
    "against time a <i>waterfall</i>; each rises from its baseline by its amplitude.</p>"
    "<p>The red line on the base is +1X at each line's speed (dotted: −1X), where the unbalance drives. "
    "The dots are the free whirl frequencies at that speed, from the Campbell diagram and coloured as "
    "there, forward and backward. Tap the disc as it runs and peaks show up on them.</p>"
)


@dataclass
class Tallest:
    """The tallest peak drawn: its amplitude (m), frequency (Hz) and line."""

    amp: float
    freq: float
    block: SpectrumBlock


class WaterfallPlot(pg.GraphicsLayoutWidget):
    """Full spectra of a station's orbit stacked against speed or time, with 1X and the free whirl marked."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(WATERFALL_TIP)
        self.plot = p = self.addPlot(row=0, col=0)
        p.setLabel("bottom", "Whirl frequency (− backward, + forward)", units="Hz")
        p.getAxis("bottom").enableAutoSIPrefix(False)
        p.getAxis("left").enableAutoSIPrefix(False)
        p.setMenuEnabled(False)
        p.setMouseEnabled(x=False, y=False)
        p.hideButtons()
        self.legend = pg.LegendItem(colCount=3, brush=colors.legend_brush(), labelTextColor=colors.foreground)
        self.addItem(self.legend, row=1, col=0)
        self.axis = pg.InfiniteLine(pos=0.0, angle=90)
        p.addItem(self.axis)
        self.lines: list[pg.PlotCurveItem] = []
        self.sync = pg.PlotCurveItem()
        self.sync_back = pg.PlotCurveItem()
        self.forward = pg.ScatterPlotItem(size=5, name="free whirl, forward")
        self.backward = pg.ScatterPlotItem(size=5, name="free whirl, backward")
        for z, item in enumerate((self.sync, self.sync_back, self.forward, self.backward)):
            item.setZValue(MAX_LINES + 10 + z)
            p.addItem(item)
        self.sync_sample = pg.PlotDataItem()  # the legend's sample of the 1X line
        self.legend.addItem(self.sync_sample, "1X at each line's speed")
        for item in (self.forward, self.backward):
            self.legend.addItem(item, item.name())
        self.apply_theme()

    def apply_theme(self) -> None:
        self.sync_sample.setPen(pg.mkPen(colors.force, width=2))
        self.axis.setPen(pg.mkPen(colors.faint, width=1))
        for line in self.lines:
            line.setPen(pg.mkPen(colors.strong, width=1))
            line.setBrush(pg.mkBrush(colors.background))
        self.sync.setPen(pg.mkPen(colors.force, width=2))
        self.sync_back.setPen(pg.mkPen(colors.force, width=1.5, style=DOT))
        for item, c in ((self.forward, colors.mode[1]), (self.backward, colors.mode[3])):
            item.setBrush(pg.mkBrush(c))
            item.setPen(pg.mkPen(colors.background, width=0.5))

    def _pool(self, n: int) -> None:
        while len(self.lines) < n:
            line = pg.PlotCurveItem()
            line.setPen(pg.mkPen(colors.strong, width=1))
            line.setBrush(pg.mkBrush(colors.background))
            self.plot.addItem(line)
            self.lines.append(line)

    def set_blocks(
        self,
        freqs: np.ndarray,
        blocks: list[SpectrumBlock],
        ys: np.ndarray,
        station: int,
        log: bool,
        marks: bool,
        y_label: tuple[str, str],
    ) -> Tallest | None:
        """Draw `blocks` (in time order) at baselines `ys`, the spectra at `station`; return the tallest peak."""
        p = self.plot
        p.setLabel("left", *y_label)
        if not blocks:
            for line in self.lines:
                line.setVisible(False)
            for item in (self.sync, self.sync_back):
                item.setData([], [])
            for item in (self.forward, self.backward):
                item.setData([], [])
            return None
        ys = np.asarray(ys, dtype=float)
        f1 = np.array([b.omega for b in blocks]) / (2.0 * math.pi)
        natural = [np.abs(b.natural) for b in blocks if b.natural.size]
        lowest = min((float(n.min()) for n in natural), default=0.0)
        reach = min(max(1.6 * float(f1.max()), 1.3 * lowest, 20.0), float(np.abs(freqs).max()))
        inside = np.abs(freqs) <= reach
        f = freqs[inside]

        pick = np.arange(len(blocks))
        if pick.size > MAX_LINES:
            pick = np.unique(np.round(np.linspace(0, len(blocks) - 1, MAX_LINES)).astype(int))
        amps = np.array([blocks[i].amps[station][inside] for i in pick], dtype=float)
        top = float(amps.max())
        k, j = np.unravel_index(int(np.argmax(amps)), amps.shape)
        tallest = Tallest(top, float(f[j]), blocks[pick[k]])
        if top <= 0.0:
            heights = np.zeros_like(amps)
        elif log:
            floor = top * 10.0**-DECADES
            heights = np.log10(np.maximum(amps, floor) / floor) / DECADES
        else:
            heights = amps / top

        base = ys[pick]
        span = float(np.ptp(base))
        spacing = span / (base.size - 1) if base.size > 1 else 0.0
        scale = max(0.2 * span, 2.5 * spacing) or max(0.05 * abs(float(base[0])), 1.0)
        self._pool(pick.size)
        rank = np.empty(pick.size, dtype=int)
        rank[np.argsort(base, kind="stable")] = np.arange(pick.size)
        for i, line in enumerate(self.lines):
            line.setVisible(i < pick.size)
            if i < pick.size:
                # The lowest line is in front: it hides the ones behind (above) it.
                line.setData(f, base[i] + scale * heights[i], fillLevel=float(base[i]))
                line.setZValue(pick.size - rank[i])

        for item, sign in ((self.sync, 1.0), (self.sync_back, -1.0)):
            item.setData(sign * f1, ys) if marks else item.setData([], [])
        for item, forward in ((self.forward, True), (self.backward, False)):
            pts = [
                (n, base[i])
                for i, b in enumerate(blocks[j] for j in pick)
                for n in b.natural
                if (n > 0) == forward and abs(n) <= reach
            ]
            if marks and pts:
                x, y = np.array(pts).T
                item.setData(x, y)
            else:
                item.setData([], [])
        lo, hi = float(base.min()), float(base.max()) + scale
        pad = 0.03 * (hi - lo)
        p.setRange(xRange=(-reach, reach), yRange=(lo - pad, hi + pad), padding=0)
        return tallest


def against_speed(mode: int) -> bool:
    return mode < 2


def baselines(blocks: list[SpectrumBlock], mode: int) -> tuple[np.ndarray, tuple[str, str]]:
    """The lines' baselines for an AGAINST mode, and the vertical axis's label and units."""
    if against_speed(mode):
        return np.array([b.omega * RPM for b in blocks]), ("Spin speed", "rpm")
    return np.array([b.t for b in blocks]), ("Time", "s")


class WaterfallTab(QtWidgets.QWidget):
    """The Waterfall tab: its controls, the plot and the note under it, and the spectra (`waterfall`).

    The page feeds it the history's samples (`feed`), starts it afresh for a new rotor, speed range or block
    length (`restart`, which Clear and the block box do themselves) and draws it while it shows (`draw`).
    `tapping`: the page's *Tap the disc every* is ticked (the note says to, if not)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.waterfall: Waterfall | None = None  # spectra block by block, made by restart
        self.drawn = False  # the plot shows the latest blocks
        self.tapping = False
        self._rotor: tuple[RotorSystem, float] | None = None  # the rotor and top speed it was last started for
        self.plot = WaterfallPlot()
        self.station = QtWidgets.QComboBox()
        self.station.addItems(STATIONS)
        self.station.setCurrentIndex(1)
        self.against = QtWidgets.QComboBox()
        self.against.addItems(AGAINST)
        self.against.setToolTip(
            "<p>Against speed (a <i>cascade</i>): the latest run-up's or coast-down's spectra, each at its "
            "speed. Against time (a <i>waterfall</i>): every spectrum kept, steady running and taps "
            "included.</p>"
        )
        self.scale = QtWidgets.QComboBox()
        self.scale.addItems(SCALES)
        self.scale.setToolTip(
            "<p>Log: each line's height is its amplitude in decades above a floor 1000 times (60 dB) "
            "below the tallest peak, so small peaks show beside the 1X one. Linear: heights in proportion "
            "to the amplitude, as most cascades are drawn.</p>"
        )
        self.block = spin(0.1, 2.0, 0.5, 2, " s")
        self.block.setSingleStep(0.1)
        self.block.setToolTip(BLOCK_TIP)
        self.block.valueChanged.connect(self._again)
        self.marks = QtWidgets.QCheckBox("1X and free whirl")
        self.marks.setChecked(True)
        self.marks.setToolTip("Mark 1X and the free whirl frequencies (from the Campbell diagram) under the lines")
        for c in (self.station, self.against, self.scale):
            c.currentIndexChanged.connect(weak.slot(self, lambda self, _: self.draw()))
        self.marks.toggled.connect(weak.slot(self, lambda self, _: self.draw()))
        self.clear = QtWidgets.QPushButton("Clear")
        self.clear.setToolTip("Drop every spectrum taken so far")
        self.clear.clicked.connect(self._again)
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        mute(self.note)
        wt = QtWidgets.QVBoxLayout(self)
        row = FlowLayout()
        row.add(
            group("Orbit at:", self.station),
            group("Against:", self.against),
            group("Amplitude:", self.scale),
            group("Block:", self.block),
            self.marks,
            self.clear,
        )
        wt.addLayout(row)
        wt.addWidget(self.plot, 1)
        wt.addWidget(self.note)

    def restart(self, system: RotorSystem, top_rpm: float) -> None:
        """Start the spectra afresh for this rotor, up to WATERFALL_RANGE times this top speed."""
        self._rotor = (system, top_rpm)
        f_max = max(WATERFALL_RANGE * top_rpm / 60.0, 50.0)
        self.waterfall = Waterfall(self.block.value(), f_max, system)
        self.drawn = False

    def _again(self, *_) -> None:
        """Clear, or a new block length: afresh for the same rotor."""
        if self._rotor is not None:
            self.restart(*self._rotor)
            self.draw()

    def feed(self, t: np.ndarray, z: np.ndarray, omega: np.ndarray) -> None:
        """Samples for the spectra (see `core.rotor_waterfall.Waterfall.feed`)."""
        if self.waterfall is not None and self.waterfall.feed(t, z, omega):
            self.drawn = False

    def draw(self, only_new: bool = False) -> None:
        """Draw the spectra (with `only_new`, only if blocks came in since they were last drawn)."""
        if only_new and self.drawn:
            return
        wf = self.waterfall
        if wf is None:
            return
        self.drawn = True
        mode = self.against.currentIndex()
        blocks = wf.latest_ramp(up=mode == 0) if against_speed(mode) else wf.blocks
        ys, label = baselines(blocks, mode)
        st = self.station.currentIndex()
        log = self.scale.currentIndex() == 0
        tallest = self.plot.set_blocks(wf.freqs, blocks, ys, st, log, self.marks.isChecked(), label)
        where = f"the {STATIONS[st].lower()} orbit"
        if tallest is None:
            if against_speed(mode):
                text = (
                    f"No {'run-up' if mode == 0 else 'coast-down'} since the spectra were cleared: change "
                    "the target speed, or run up and coast down, and the spectra are drawn at each speed. "
                    "Spectra taken at a steady speed are shown against time."
                )
            else:
                text = f"The first {sig(wf.block, 6)} s block is being taken."
            self.note.setText(text)
            return
        b = tallest.block
        at = f"{b.omega * RPM:,.0f} rpm" if against_speed(mode) else f"{b.t:.1f} s ({b.omega * RPM:,.0f} rpm)"
        order = f" ({tallest.freq / (b.omega / (2 * math.pi)):+.2f}×)" if b.omega > 0 else ""
        self.note.setText(
            f"{len(blocks)} full spectra of {where}, each of a {sig(wf.block, 6)} s block ({sig(1 / wf.block, 2)} Hz "
            f"resolution), overlapping by half. Tallest peak {fmt_len(tallest.amp)} at "
            f"{tallest.freq:+.1f} Hz{order}, at {at}"
            + (": heights are in decades above 1/1000 of it (60 dB)." if log else ".")
            + ("" if self.tapping else " Tap the disc, or tick <i>Tap the disc every</i>, to ring the free whirl.")
        )
