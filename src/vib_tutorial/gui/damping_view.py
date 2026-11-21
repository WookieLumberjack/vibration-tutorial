"""Damping models' four oscillators drawn side by side, each with its own damper, moving with its free decay.

A row for each model (`core.damping_models.Model`), one above the other: a wall, the spring, and below it the
model's damping element (a dashpot for viscous; a dashed one for hysteretic, whose decay is drawn as the
viscous equivalent; a spring and a dashpot in series for the Zener element), and the mass in the model's
colour, its name beside it. The friction row's mass slides on a floor instead, with its stick band ±F<sub>f</sub>/k shaded on it:
where it stops dead. A dashed line marks rest.

- `set_rows(colors, shown, names)` builds and colours them (on every redraw, so a theme change too) (`shown` hides a model's row, as its box does).
- `set_scale(release, band)`: the release amplitude (m) is drawn `SWING` from rest, the band to the same scale.
- `set_positions(x)`: each model's displacement (m).
- Grab any mass and pull it sideways: `grabbed()`, then `pulled(x)` with the displacement (m) under the hand,
  then `released()`. Every mass is released from the same place, so the page sets them all to `x`.

Working on it: `tests/test_playback.py` drags a mass with synthetic mouse events and checks the decay is
redrawn from where it was let go.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.damping_models import MODELS, Model
from . import coalesce
from .animation import damper_path, spring_path
from .drag import Grabbable
from .style import colors

WALL = 0.0
REST = 2.0  # the masses' centres at rest
WIDTH = 0.5  # a mass's width
HEIGHT = 0.6  # its height (rows are 1 apart)
SWING = 0.7  # the release amplitude, drawn
MOST = 0.9  # the farthest a mass can be pulled
SPRING_Y, DAMPER_Y = 0.16, -0.16
FLOOR = -HEIGHT / 2 - 0.04  # the friction row's floor, below its mass
EDGE = REST + WIDTH / 2 + MOST  # the farthest a mass's edge goes
X_RANGE = (-0.2, EDGE + 1.0)  # with room for the names
TIP = (
    "<p>The four models side by side, each matched to the same damping, released together from rest.</p>"
    "<p><b>Pluck:</b> pull a mass sideways and let go: they all start from there. Friction's mass stops dead "
    "inside its shaded band, where the spring can no longer beat the friction.</p>"
)


class DampingView(Grabbable, pg.PlotWidget):
    grabbed = QtCore.Signal()
    pulled = QtCore.Signal(float)  # the displacement under the hand (m)
    released = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("The four damped masses, moving")
        self.setToolTip(TIP)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.setMouseTracking(True)
        self.hideButtons()
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.gain = SWING / 0.01  # display units per metre
        self.held = False
        self._x = {m: 0.0 for m in MODELS}
        self._rows: dict[Model, float] = {}  # each shown model's row (y of its centre)
        self._colors: dict[Model, str] = {}
        self._band = 0.0  # ±F_f/k (m)
        self._wall = pg.PlotDataItem(connect="finite")
        self._hatch = pg.PlotDataItem(connect="finite")
        self._rest = pg.PlotDataItem()
        self._floor = pg.PlotDataItem(connect="finite")
        self._stick = QtWidgets.QGraphicsRectItem()
        for item in (self._wall, self._hatch, self._rest, self._floor, self._stick):
            self.addItem(item)
        self._stick.setZValue(6)  # over the floor's hatching
        self._springs = {m: self._add(pg.PlotDataItem()) for m in MODELS}
        self._dampers = {m: self._add(pg.PlotDataItem(connect="finite")) for m in MODELS}
        self._series = {m: self._add(pg.PlotDataItem()) for m in MODELS}  # the Zener arm's spring
        self._masses = {m: self._add(QtWidgets.QGraphicsRectItem()) for m in MODELS}
        self._labels = {m: self._add(pg.TextItem(anchor=(0, 0.5))) for m in MODELS}
        for m in MODELS:
            self._masses[m].setZValue(5)

    def _add(self, item):
        self.addItem(item)
        return item

    # ------------------------------------------------------------- structure
    def set_rows(self, cols: dict[Model, str], shown: dict[Model, bool], names: dict[Model, str]) -> None:
        """Colour each model's row, and lay out the shown ones from the top."""
        self._colors = dict(cols)
        rows = [m for m in MODELS if shown[m]]
        self._rows = {m: float(len(rows) - 1 - i) for i, m in enumerate(rows)}
        n = max(len(rows), 1)
        pen = pg.mkPen(colors.structure, width=2)
        top, bottom = n - 0.5, -0.5
        self._wall.setData([WALL, WALL], [bottom + 0.1, top - 0.1])
        self._wall.setPen(pg.mkPen(colors.structure, width=4))
        hx: list[float] = []
        hy: list[float] = []
        for y in np.linspace(bottom + 0.15, top - 0.1, 3 * n):
            hx += [WALL, WALL - 0.12, np.nan]
            hy += [y, y - 0.12, np.nan]
        self._hatch.setData(hx, hy)
        self._hatch.setPen(pen)
        self._rest.setData([REST, REST], [bottom + 0.05, top - 0.05])
        self._rest.setPen(pg.mkPen(colors.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        for m in MODELS:
            on = m in self._rows
            for item in (self._springs[m], self._dampers[m], self._series[m], self._masses[m], self._labels[m]):
                item.setVisible(on)
            self._springs[m].setPen(pen)
            dash = QtCore.Qt.PenStyle.DashLine if m is Model.HYSTERETIC else QtCore.Qt.PenStyle.SolidLine
            self._dampers[m].setPen(pg.mkPen(colors.structure, width=2, style=dash))
            self._series[m].setPen(pen)
            self._masses[m].setBrush(pg.mkBrush(cols[m]))
            self._masses[m].setPen(pg.mkPen(colors.structure, width=1.5))
            self._labels[m].setText(names[m])
            self._labels[m].setColor(cols[m])
        friction = Model.COULOMB in self._rows
        self._floor.setVisible(friction)
        self._stick.setVisible(friction)
        if friction:
            floor = self._rows[Model.COULOMB] + FLOOR
            fx: list[float] = [WALL, EDGE]
            fy: list[float] = [floor, floor]
            for tick in np.linspace(WALL + 0.2, EDGE, 14):
                fx += [np.nan, float(tick), float(tick) - 0.1]
                fy += [np.nan, floor, floor - 0.1]
            self._floor.setData(fx, fy)
            self._floor.setPen(pen)
            fill = pg.mkColor(colors.force)
            fill.setAlpha(110)
            self._stick.setBrush(pg.mkBrush(fill))
            self._stick.setPen(pg.mkPen(colors.force, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.setRange(QtCore.QRectF(X_RANGE[0], bottom, X_RANGE[1] - X_RANGE[0], top - bottom), padding=0)
        self._draw()

    def set_scale(self, release: float, band: float) -> None:
        """Draw `release` (m) `SWING` from rest, and friction's stick band ±`band` (m)."""
        if not self.held:  # under the hand the scale holds still
            self.gain = SWING / max(abs(release), 1e-6)
        self._band = band
        self._draw()

    def set_positions(self, x: dict[Model, float]) -> None:
        self._x = dict(x)
        self._draw()

    # ---------------------------------------------------------------- drawing
    @coalesce.coalesced(leaf=True)
    def _draw(self) -> None:
        if Model.COULOMB in self._rows:
            y = self._rows[Model.COULOMB] + FLOOR
            half = min(self.gain * self._band, MOST)
            self._stick.setRect(QtCore.QRectF(REST - half, y - 0.12, 2 * half, 0.12))  # under the floor
        for m, y in self._rows.items():
            xc = REST + float(np.clip(self.gain * self._x.get(m, 0.0), -MOST - 0.3, MOST + 0.3))
            left = xc - WIDTH / 2
            self._masses[m].setRect(QtCore.QRectF(left, y - HEIGHT / 2, WIDTH, HEIGHT))
            self._labels[m].setPos(EDGE + 0.08, y)
            gap = REST - WIDTH / 2 - WALL
            if m is Model.COULOMB:  # the floor takes the friction: the spring alone, centred
                self._springs[m].setData(*spring_path(WALL, left, y, amp=0.09))
                self._dampers[m].setData([], [])
                self._series[m].setData([], [])
                continue
            self._springs[m].setData(*spring_path(WALL, left, y + SPRING_Y, amp=0.07))
            if m is Model.ZENER:  # a spring k1 in series with the dashpot
                mid = WALL + 0.4 * gap
                self._series[m].setData(*spring_path(WALL, mid, y + DAMPER_Y, coils=3, amp=0.07))
                self._dampers[m].setData(*damper_path(mid, left, y + DAMPER_Y, gap - 0.4 * gap, h=0.08))
            else:
                self._series[m].setData([], [])
                self._dampers[m].setData(*damper_path(WALL, left, y + DAMPER_Y, gap, h=0.08))

    # ------------------------------------------------------------------- pluck
    def on_mass(self, pos: QtCore.QPointF) -> bool:
        """Whether a point (view coordinates) is on one of the masses."""
        for m, y in self._rows.items():
            xc = REST + self.gain * self._x.get(m, 0.0)
            if abs(pos.x() - xc) <= WIDTH / 2 + 0.05 and abs(pos.y() - y) <= HEIGHT / 2 + 0.05:
                return True
        return False

    def _pull(self, pos: QtCore.QPointF) -> None:
        offset = float(np.clip(pos.x() - REST, -MOST, MOST))
        self.pulled.emit(offset / self.gain)

    def grab_at(self, pos: QtCore.QPointF) -> object | None:
        return True if self.on_mass(pos) else None

    def _take(self, what: object, pos: QtCore.QPointF) -> None:
        self.grabbed.emit()
        self._pull(pos)

    def _let_go(self, what: object) -> None:
        self.released.emit()
