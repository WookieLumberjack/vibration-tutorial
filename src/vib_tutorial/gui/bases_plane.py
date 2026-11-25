"""The (x₁, x₂) plane (#478 Phase 4, Coordinates & bases): the masses' configuration as a point, and the basis's
arrows, dragged as it runs.

`Plane` draws what `view_models/bases.py`'s `plane_view` and `projection` give (in mm) and forwards what the
hand does:

- **An arrow's tip:** drag it to turn and stretch the arrow: `arrow_taken(i)`, `arrow_moved(i, x₁, x₂)` (m, where
  the tip is), `arrow_released(i)`. The page snaps it (`arrow_dragged`) and draws it back with `set_still`.
- **Anywhere else:** a hand takes the masses there: `grabbed()`, `pulled(x₁, x₂)` (m, within the pluck's reach)
  as it moves, `released()`.

The plane's range is fixed (`REACH`), so the arrows and the path keep their size.

Working on it: `tests/test_bases_page.py` drags an arrow and plucks the point with synthetic mouse events.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..view_models.bases import MM, PLUCK, PlaneView
from .drag import Grabbable
from .lesson import Dot
from .style import colors

TIP_R = 2.2  # mm: an arrow's handle
GRAB = 5.0  # mm: how near an arrow's tip a press takes it
TIP = (
    "<p>The masses' configuration, (x₁, x₂), as one point, and the path it has traced. The two arrows are a basis: "
    "the dashed parallelogram writes the point as q₁v₁ + q₂v₂, and the faint grid is the lines of constant q₁ "
    "and q₂.</p>"
    "<p><b>Drag an arrow's tip</b> to turn or stretch it as it runs: the motion doesn't change, only its "
    "description. Near a mode or an axis it snaps on. <b>Drag anywhere else</b> to take the masses there, and let go "
    "to pluck them.</p>"
)


class Plane(Grabbable, pg.PlotWidget):
    grabbed = QtCore.Signal()
    pulled = QtCore.Signal(float, float)  # m
    released = QtCore.Signal()
    arrow_taken = QtCore.Signal(int)
    arrow_moved = QtCore.Signal(int, float, float)  # the arrow, where its tip is dragged (m)
    arrow_released = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("The plane of the two masses' displacements, with the basis's arrows")
        self.setToolTip(TIP)
        self.setMouseTracking(True)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setAspectLocked(True)
        self.setLabel("bottom", "x₁", units="mm")
        self.setLabel("left", "x₂", units="mm")
        for side in ("bottom", "left"):
            self.getAxis(side).enableAutoSIPrefix(False)
        self.held: object = False  # "point", or an arrow's index
        self.tips: tuple[tuple[float, float], ...] = ()
        self.span = PLUCK * MM

        def add(item, z: float = 0.0):
            item.setZValue(z)
            self.addItem(item, ignoreBounds=True)
            return item

        self.grid = add(pg.PlotDataItem(connect="finite"), -2)
        self.axes_lines = add(pg.PlotDataItem(connect="finite"), -1)
        self.mode_lines = [add(pg.PlotDataItem(), 0) for _ in range(2)]
        self.mode_labels = [add(pg.TextItem(anchor=(0.5, 0.5)), 0) for _ in range(2)]
        self.rail = add(pg.PlotDataItem(), 1)
        self.trail = add(pg.PlotDataItem(), 2)
        self.parallelogram = add(pg.PlotDataItem(connect="finite"), 3)
        self.shafts = [add(pg.PlotDataItem(), 4) for _ in range(2)]
        self.heads = [add(pg.ArrowItem(headLen=16, tipAngle=36, tailLen=0), 5) for _ in range(2)]
        self.handles = [add(QtWidgets.QGraphicsEllipseItem(), 6) for _ in range(2)]
        self.labels = [add(pg.TextItem(anchor=(0.0, 1.0)), 6) for _ in range(2)]
        self.point = Dot(11)
        self.addItem(self.point, ignoreBounds=True)
        self.point.setZValue(7)
        self.apply_theme()

    def apply_theme(self) -> None:
        c = colors
        self.grid.setPen(pg.mkPen(c.faint, width=1))
        self.axes_lines.setPen(pg.mkPen(c.grey, width=1))
        for line in self.mode_lines:
            line.setPen(pg.mkPen(c.grey, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        for label in self.mode_labels:
            label.setColor(c.muted)
        self.rail.setPen(pg.mkPen(c.structure, width=5))
        self.trail.setPen(pg.mkPen(c.strong, width=1.2))
        self.parallelogram.setPen(pg.mkPen(c.muted, width=1, style=QtCore.Qt.PenStyle.DashLine))
        for i in range(2):
            self.shafts[i].setPen(pg.mkPen(c.mass[i], width=3))
            # through setStyle: each setStyle (the angle, as the arrow turns) puts back the style's own pen and brush
            self.heads[i].setStyle(pen=pg.mkPen(c.mass[i], width=1), brush=pg.mkBrush(c.mass[i]))
            self.handles[i].setPen(pg.mkPen(c.mass[i], width=2))
            self.handles[i].setBrush(pg.mkBrush(c.background))
            self.labels[i].setColor(c.mass[i])
        self.point.setPen(pg.mkPen(c.background, width=1.5))
        self.point.setBrush(pg.mkBrush(c.strong))

    # ------------------------------------------------------------- drawing
    def set_still(self, v: PlaneView) -> None:
        """The plane's still parts: the arrows, grid, mode lines and rail."""
        s = v.span
        if s != self.span or not self.tips:
            self.span = s
            self.setRange(QtCore.QRectF(-s, -s, 2 * s, 2 * s), padding=0.02)
            self.axes_lines.setData([-s, s, math.nan, 0, 0], [0, 0, math.nan, -s, s])
        self.grid.setData(*v.grid)
        for i in range(2):
            on = i < len(v.modes)
            self.mode_lines[i].setVisible(on)
            self.mode_labels[i].setVisible(on)
            if on:
                self.mode_lines[i].setData(*v.modes[i])
                self.mode_labels[i].setText(f"mode {i + 1}")
                self.mode_labels[i].setPos(*v.mode_tips[i])
        self.rail.setData(*v.rail)
        self.tips = v.arrows
        for i in range(2):
            on = i < len(v.arrows)
            for item in (self.shafts[i], self.heads[i], self.handles[i], self.labels[i]):
                item.setVisible(on)
            if not on:
                continue
            x, y = v.arrows[i]
            self.shafts[i].setData([0.0, x], [0.0, y])
            n = math.hypot(x, y) or 1.0
            self.heads[i].setPos(x * (1 - TIP_R / n), y * (1 - TIP_R / n))  # its point at the handle's edge
            self.heads[i].setStyle(angle=180.0 - math.degrees(math.atan2(y, x)))
            self.handles[i].setRect(QtCore.QRectF(x - TIP_R, y - TIP_R, 2 * TIP_R, 2 * TIP_R))
            self.labels[i].setText(v.labels[i])
            self.labels[i].setPos(x * 1.08 + 1.5, y * 1.08 + 1.5)

    def show_motion(
        self, trail: tuple[np.ndarray, np.ndarray], x: np.ndarray, box: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """The path (mm), the point now (m) and its parallelogram (mm)."""
        self.trail.setData(*trail)
        self.point.setPos(float(x[0]) * MM, float(x[1]) * MM)
        self.parallelogram.setData(*box)

    # ------------------------------------------------------------- the hand
    def grab_at(self, pos: QtCore.QPointF) -> object | None:
        for i, (x, y) in enumerate(self.tips):
            if math.hypot(pos.x() - x, pos.y() - y) <= GRAB:
                return i
        s = self.span * 1.1
        return "point" if abs(pos.x()) <= s and abs(pos.y()) <= s else None

    def _holding(self) -> bool:
        return self.held is not False

    def _take(self, what: object, pos: QtCore.QPointF) -> None:
        if isinstance(what, int):
            self.arrow_taken.emit(what)
        else:
            self.grabbed.emit()
        self._pull(pos)

    def _pull(self, pos: QtCore.QPointF) -> None:
        what = self.held
        x, y = pos.x() / MM, pos.y() / MM
        if isinstance(what, int):
            self.arrow_moved.emit(what, x, y)
            return
        self.pulled.emit(max(-PLUCK, min(PLUCK, x)), max(-PLUCK, min(PLUCK, y)))

    def _let_go(self, what: object) -> None:
        if isinstance(what, int):
            self.arrow_released.emit(what)
        else:
            self.released.emit()

    def press_point(self, i: int) -> QtCore.QPointF:
        """Where arrow i's tip is, in view coordinates (mm; tests drag it)."""
        x, y = self.tips[i]
        return QtCore.QPointF(x, y)
