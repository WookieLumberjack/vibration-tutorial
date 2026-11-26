"""State-space modes' planes (#478 Phase 4): a pair of the state's coordinates, and the (x₁, x₂) plane with the
modes' shapes.

- `PairPlane`: one pair of the coordinates w = T⁻¹z (mm), its recent path and its point now. Drag anywhere on it to
  put the pair there (`grabbed()`, `pulled(a, b)` mm, within its reach, `released()`): the page sets that pair and
  keeps the other.
- `ShapePlane`: the masses' displacements (x₁, x₂) as a point and its path, with each state-space mode's Re ψ and
  Im ψ as arrows and the ellipse a mode alone traces (`view_models/state_space.x_plane`). Drag to take the masses
  there from rest, and let go to pluck them.

The (x₁, x₂) plane keeps a fixed range (`REACH`), so a path keeps its size as it shrinks. A pair's plane zooms to
its own path (`fit`: at least ±`SMALLEST` mm, grown at once, shrunk only past half its size), so a pair a
thousandth of the other's, a leak just starting, still shows.

Working on it: `tests/test_state_space_page.py` drags both with synthetic mouse events.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..view_models.state_space import MM, REACH, PlaneLabels, XPlane
from .drag import Grabbable
from .lesson import Dot
from .style import colors

MIN_SIDE = 120  # px: a plane's smallest side
SMALLEST = 1.0  # mm: a pair's plane zooms in no further than ±this

PAIR_TIP = (
    "<p>Two of the state's four coordinates, a pair, in the basis picked on the left (mm), and the path they've "
    "traced. In the state-space modes each pair is one mode, and its point spirals in on its own. In the other "
    "bases the pairs are coupled, and a pair started at zero doesn't stay there.</p>"
    "<p><b>Drag</b> to set this pair, the other kept, and let go to run on from there.</p>"
)
SHAPE_TIP = (
    "<p>The masses' displacements, (x₁, x₂), as one point, and its path. The arrows are each state-space mode's "
    "shape, the real part of ψ thick and the imaginary part thin, and the dashed curve is the path the mode alone "
    "would trace: a line when the shape is real, an ellipse when it's complex.</p>"
    "<p><b>Drag</b> to take the masses there from rest, and let go to pluck them.</p>"
)


class PlaneRow(QtWidgets.QWidget):
    """The three planes side by side, as tall as one is wide, so each is square and the height they don't need goes
    to the masses above. A plot in a row doesn't get height-for-width from its layout, and a widget with a layout
    answers it from the layout, so the row places its planes itself (`resizeEvent`) and answers it itself."""

    GAP = 6  # px between the planes

    def __init__(self, planes: list[QtWidgets.QWidget]) -> None:
        super().__init__()
        self.planes = planes
        for p in planes:
            p.setParent(self)
        policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def _side(self, width: int) -> int:
        n = len(self.planes)
        return max((width - self.GAP * (n - 1)) // n, MIN_SIDE)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._side(width)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(3 * 220 + 2 * self.GAP, 220)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(3 * MIN_SIDE + 2 * self.GAP, MIN_SIDE)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        side = min(self._side(self.width()), self.height())
        n = len(self.planes)
        left = (self.width() - n * side - (n - 1) * self.GAP) // 2  # centered, if the row is shorter than square
        for i, p in enumerate(self.planes):
            p.setGeometry(left + i * (side + self.GAP), 0, side, side)


class _Plane(Grabbable, pg.PlotWidget):
    grabbed = QtCore.Signal()
    pulled = QtCore.Signal(float, float)
    released = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        self.setMouseTracking(True)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setAspectLocked(True)
        for side in ("bottom", "left"):
            self.getAxis(side).enableAutoSIPrefix(False)
        s = REACH * MM
        self.span = s
        self.setRange(QtCore.QRectF(-s, -s, 2 * s, 2 * s), padding=0.02)
        self.axes_lines = self._add(pg.PlotDataItem(connect="finite"), -1)
        self.axes_lines.setData([-10 * s, 10 * s, math.nan, 0, 0], [0, 0, math.nan, -10 * s, 10 * s])
        self.trail = self._add(pg.PlotDataItem(), 2)
        self.point = Dot(11)
        self.addItem(self.point, ignoreBounds=True)
        self.point.setZValue(7)

    def _add(self, item, z: float = 0.0):
        item.setZValue(z)
        self.addItem(item, ignoreBounds=True)
        return item

    def apply_theme(self) -> None:
        self.axes_lines.setPen(pg.mkPen(colors.grey, width=1))
        self.point.setPen(pg.mkPen(colors.background, width=1.5))

    def show_motion(self, trail: tuple[np.ndarray, np.ndarray], now: tuple[float, float]) -> None:
        """The path and the point now (mm)."""
        self.trail.setData(*trail)
        self.point.setPos(*now)

    # ------------------------------------------------------------- the hand
    def grab_at(self, pos: QtCore.QPointF) -> object | None:
        s = self.span * 1.1
        return True if abs(pos.x()) <= s and abs(pos.y()) <= s else None

    def _take(self, what: object, pos: QtCore.QPointF) -> None:
        self.grabbed.emit()
        self._pull(pos)

    def _pull(self, pos: QtCore.QPointF) -> None:
        s = self.span
        self.pulled.emit(max(-s, min(s, pos.x())), max(-s, min(s, pos.y())))

    def _let_go(self, what: object) -> None:
        self.released.emit()


class PairPlane(_Plane):
    """One pair of the state's coordinates (see the module)."""

    def __init__(self, index: int, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.index = index
        self.setToolTip(PAIR_TIP)
        self.setAccessibleName(f"The state's coordinate pair {index + 1}")
        self.labels = PlaneLabels("", "", "")
        self.apply_theme()

    def apply_theme(self) -> None:
        super().apply_theme()
        color = colors.mass[self.index]
        self.trail.setPen(pg.mkPen(color, width=1.5))
        self.point.setBrush(pg.mkBrush(color))

    def fit(self, reach: float) -> None:
        """Zoom to show ±`reach` (mm), steadily (see the module)."""
        want = max(reach * 1.15, SMALLEST)
        if want > self.span or want < 0.45 * self.span:
            self.span = want
            s = want
            self.setRange(QtCore.QRectF(-s, -s, 2 * s, 2 * s), padding=0.02)
            self.axes_lines.setData([-10 * s, 10 * s, math.nan, 0, 0], [0, 0, math.nan, -10 * s, 10 * s])

    def show_motion(self, trail: tuple[np.ndarray, np.ndarray], now: tuple[float, float]) -> None:
        super().show_motion(trail, now)
        a, b = trail
        reach = max(
            abs(now[0]),
            abs(now[1]),
            float(np.abs(a).max()) if a.size else 0.0,
            float(np.abs(b).max()) if b.size else 0.0,
        )
        self.fit(reach)

    def grab_at(self, pos: QtCore.QPointF) -> object | None:
        return True  # anywhere on it: a pair can be put anywhere, and the plane zooms out to follow

    def _pull(self, pos: QtCore.QPointF) -> None:
        s = REACH * MM
        self.pulled.emit(max(-s, min(s, pos.x())), max(-s, min(s, pos.y())))

    def set_labels(self, labels: PlaneLabels) -> None:
        if labels == self.labels:
            return
        self.labels = labels
        self.setTitle(labels.title)
        self.setLabel("bottom", labels.x, units="mm")
        self.setLabel("left", labels.y, units="mm")


class ShapePlane(_Plane):
    """The (x₁, x₂) plane with the modes' shapes (see the module)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(SHAPE_TIP)
        self.setAccessibleName("The plane of the two masses' displacements, with the state-space modes' shapes")
        self.setTitle("The masses")
        self.setLabel("bottom", "x₁", units="mm")
        self.setLabel("left", "x₂", units="mm")
        self.ellipses = [self._add(pg.PlotDataItem(), 0) for _ in range(2)]
        self.re_arrows = [self._add(pg.PlotDataItem(), 3) for _ in range(2)]
        self.im_arrows = [self._add(pg.PlotDataItem(), 3) for _ in range(2)]
        self.names = [self._add(pg.TextItem(anchor=(0.0, 1.0)), 4) for _ in range(2)]
        self.apply_theme()

    def apply_theme(self) -> None:
        super().apply_theme()
        c = colors
        self.trail.setPen(pg.mkPen(c.strong, width=1.2))
        self.point.setBrush(pg.mkBrush(c.strong))
        for i in range(2):
            self.ellipses[i].setPen(pg.mkPen(c.mass[i], width=1.2, style=QtCore.Qt.PenStyle.DashLine))
            self.re_arrows[i].setPen(pg.mkPen(c.mass[i], width=3))
            self.im_arrows[i].setPen(pg.mkPen(c.mass[i], width=1.5))
            self.names[i].setColor(c.mass[i])

    def set_still(self, v: XPlane) -> None:
        """Each mode's arrows and ellipse."""
        for i in range(2):
            on = i < len(v.shapes)
            for item in (self.ellipses[i], self.re_arrows[i], self.im_arrows[i], self.names[i]):
                item.setVisible(on)
            if not on:
                continue
            sh = v.shapes[i]
            self.ellipses[i].setData(*sh.ellipse)
            self.re_arrows[i].setData([0.0, sh.re[0]], [0.0, sh.re[1]])
            self.im_arrows[i].setData([0.0, sh.im[0]], [0.0, sh.im[1]])
            self.names[i].setText(f"mode {i + 1}")
            self.names[i].setPos(sh.re[0] * 0.6 + 1.0, sh.re[1] * 0.6 + 1.0)
