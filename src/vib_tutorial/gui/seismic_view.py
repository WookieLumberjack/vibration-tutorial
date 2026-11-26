"""Seismic response's building drawn swaying under a record, each story's columns colored by its drift.

The ground at the bottom, then a slab per floor, and between each pair (the ground and floor 1 the first) a
story's two columns, bent in the double curve of a column fixed into both slabs: a story's top sits its drift
over from its bottom. The floors' displacements are relative to the ground (the ground is drawn still).

- `set_history(displacement)`: the floors' displacements (time × floors, m) the motion will play; the sway is
  drawn so the largest of them is `SWAY` of a bay's half-width, and each story's color runs from the
  structure's to the force color as its drift nears the largest any story reaches on the record.
- `set_state(u)`: the floors' displacements at an instant (m).
- `exaggeration`: how many times the sway is drawn larger than it would be with stories `STOREY_M` tall.

Working on it: `tests/test_seismic.py` checks the building follows the playback and the drifting story colors.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets

from . import coalesce
from .style import colors

BAY = 1.0  # a column's x, either side
SLAB = 1.2  # a slab's half-length
SWAY = 0.7  # the largest displacement, drawn as this much of BAY
STOREY_M = 3.0  # a story drawn one unit tall stands for this many meters
SHAPE = np.linspace(0.0, 1.0, 13)
BEND = 3 * SHAPE**2 - 2 * SHAPE**3  # a column fixed at both ends, its top moved over
TIP = (
    "<p>The building on the first record, its sway exaggerated (the ground is drawn still: the floors move "
    "relative to it). A story's columns redden as its drift nears the largest any story reaches on the "
    "record.</p>"
)


def blend(a: QtGui.QColor, b: QtGui.QColor, s: float) -> QtGui.QColor:
    """`s` of the way from color `a` to `b`."""
    s = min(max(s, 0.0), 1.0)
    return QtGui.QColor.fromRgbF(
        a.redF() + s * (b.redF() - a.redF()),
        a.greenF() + s * (b.greenF() - a.greenF()),
        a.blueF() + s * (b.blueF() - a.blueF()),
    )


class BuildingView(pg.PlotWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("The building swaying under the record")
        self.setToolTip(TIP)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.n = 0
        self.gain = 1.0  # display units per meter
        self.peak_drift = 1.0  # m, the largest story drift on the record
        self.u = np.zeros(0)
        self.ground = pg.PlotDataItem()
        self.hatch = pg.PlotDataItem(connect="finite")
        self.slabs = pg.PlotDataItem(connect="finite")
        self.slabs.setZValue(2)
        for item in (self.ground, self.hatch, self.slabs):
            self.addItem(item)
        self.columns: list[pg.PlotDataItem] = []
        self.apply_theme()

    @property
    def exaggeration(self) -> float:
        return self.gain * STOREY_M

    def apply_theme(self) -> None:
        c = colors
        self.ground.setPen(pg.mkPen(c.structure, width=3))
        self.hatch.setPen(pg.mkPen(c.structure, width=1.5))
        self.slabs.setPen(pg.mkPen(c.mass[0], width=7))
        self._draw()

    def set_history(self, displacement: np.ndarray) -> None:
        u = np.asarray(displacement, dtype=float)
        n = u.shape[1]
        if n != self.n:
            for item in self.columns:
                self.removeItem(item)
            self.columns = [pg.PlotDataItem(connect="finite") for _ in range(n)]
            for item in self.columns:
                self.addItem(item)
            self.n = n
            self._fit()
        peak = float(np.max(np.abs(u))) if u.size else 0.0
        self.gain = SWAY * BAY / peak if peak > 0 else 1.0
        drift = np.diff(u, prepend=0.0, axis=1)
        self.peak_drift = float(np.max(np.abs(drift))) if drift.size else 0.0
        self.u = np.zeros(n)
        self._draw()

    def set_state(self, u: np.ndarray) -> None:
        self.u = np.asarray(u, dtype=float)
        self._draw()

    def drifts(self) -> np.ndarray:
        """Each story's drift as a share of the record's largest (what its color shows)."""
        drift = np.abs(np.diff(self.u, prepend=0.0))
        return drift / self.peak_drift if self.peak_drift > 0 else drift

    def _fit(self) -> None:
        self.setXRange(-SLAB - 0.6, SLAB + 0.6, padding=0)
        self.setYRange(-0.35, self.n + 0.4, padding=0)
        x = np.linspace(-SLAB - 0.4, SLAB + 0.4, 16)
        self.ground.setData([x[0], x[-1]], [0.0, 0.0])
        hx = np.repeat(x, 3)
        hy = np.tile([0.0, -0.18, np.nan], x.size)
        hx[1::3] -= 0.15
        self.hatch.setData(hx, hy)

    @coalesce.coalesced(leaf=True)
    def _draw(self) -> None:
        if not self.n or self.u.size != self.n:
            return
        x = self.gain * self.u
        sx, sy = [], []
        for i in range(self.n):
            sx += [x[i] - SLAB, x[i] + SLAB, np.nan]
            sy += [i + 1.0, i + 1.0, np.nan]
        self.slabs.setData(sx, sy)
        share = self.drifts()
        below = np.concatenate(([0.0], x[:-1]))
        c = colors
        for i, item in enumerate(self.columns):
            path = below[i] + (x[i] - below[i]) * BEND
            cx = np.concatenate((path - BAY, [np.nan], path + BAY))
            cy = np.concatenate((i + SHAPE, [np.nan], i + SHAPE))
            item.setData(cx, cy)
            s = float(share[i])
            item.setPen(pg.mkPen(blend(QtGui.QColor(c.structure), QtGui.QColor(c.force), s), width=2 + 3 * s))
