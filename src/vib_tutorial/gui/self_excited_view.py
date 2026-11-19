"""Self-excited vibration's body drawn moving: van der Pol's mass on its spring, the galloping square and the
vortex-shedding cylinder in their flow, and the flutter section plunging and pitching.

- `set_body(kind)`: which body (`BLOCK`, `SQUARE`, `CIRCLE`, `WING`), with the flow drawn from the left for the
  last three.
- `set_range(peak)`: the largest displacement the motion reaches, in the body's own units (its side or diameter,
  the wing's semichord), so the view holds it.
- `set_state(y, theta=0)`: the body's displacement (up, in its units) and, for the wing, its pitch (rad, nose
  up).
- `unstable`: the body is filled in the force colour when rest is unstable (a pole right of Re λ = 0).

The bluff bodies and the wing are drawn to scale in their own units: a galloping square's swing of 2 D is two
of its sides. The block, whose x is in metres, is drawn at the scale `set_range` gives.

Working on it: `tests/test_self_excited.py` checks the body follows the playback.
"""

from __future__ import annotations

import enum
import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import coalesce
from .style import colors


class Body(enum.Enum):
    BLOCK = "block"
    SQUARE = "square"
    CIRCLE = "circle"
    WING = "wing"


FLOW_ROWS = 5
WING_T = 0.12  # the section's thickness, of its chord
TIP = (
    "<p>The body moving as the plots show it. The square, the cylinder and the wing are drawn to scale in their "
    "own size (the flow comes from the left); the wing plunges and pitches about its elastic axis (the dot) in "
    "its least stable mode. Filled in the force colour when rest is unstable: a pole right of Re λ = 0.</p>"
)


def _wing_outline() -> np.ndarray:
    """A symmetric section, chord 2 (semichords) from x = −1 (the nose) to 1, as a closed polygon."""
    x = 1 - np.cos(np.linspace(0.0, math.pi, 30))  # 0 … 2, closer at the nose
    t = WING_T * 2 * 5 * (0.2969 * np.sqrt(x / 2) - 0.126 * x / 2 - 0.3516 * (x / 2) ** 2 + 0.2843 * (x / 2) ** 3)
    t -= t[-1] * x / 2  # a closed trailing edge
    upper = np.column_stack((x - 1, t))
    lower = np.column_stack((x[::-1] - 1, -t[::-1]))
    return np.vstack((upper, lower))


WING = _wing_outline()


class SectionView(pg.PlotWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("The self-excited body, moving")
        self.setToolTip(TIP)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setAspectLocked(True)
        self.body = Body.BLOCK
        self.peak = 1.0
        self.axis_x = 0.0  # the wing's elastic axis, semichords from mid-chord
        self.unstable = False
        self.state = (0.0, 0.0)
        self.rest = pg.PlotDataItem()
        self.spring = pg.PlotDataItem()
        self.anchor = pg.PlotDataItem(connect="finite")
        self.flow = pg.PlotDataItem(connect="finite")
        self.shape = QtWidgets.QGraphicsPolygonItem()
        self.pivot = QtWidgets.QGraphicsEllipseItem()
        for item in (self.rest, self.flow, self.anchor, self.spring, self.shape, self.pivot):
            self.addItem(item)
        self.shape.setZValue(5)
        self.pivot.setZValue(6)
        self.apply_theme()

    def apply_theme(self) -> None:
        c = colors
        self.rest.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.spring.setPen(pg.mkPen(c.structure, width=2))
        self.anchor.setPen(pg.mkPen(c.structure, width=2))
        self.flow.setPen(pg.mkPen(c.faint, width=1.5))
        self.shape.setPen(pg.mkPen(c.structure, width=1.5))
        self.pivot.setPen(pg.mkPen(c.structure, width=1))
        self.pivot.setBrush(pg.mkBrush(c.background))
        self._draw()

    def set_body(self, body: Body, axis_x: float = 0.0) -> None:
        self.body, self.axis_x = body, axis_x
        self._fit()

    def set_range(self, peak: float) -> None:
        self.peak = max(float(peak), 1e-12)
        self._fit()

    def set_state(self, y: float, theta: float = 0.0) -> None:
        self.state = (float(y), float(theta))
        self._draw()

    # ------------------------------------------------------------------
    def _half_height(self) -> float:
        """The body's half-height and the swing the view must hold, in display units."""
        swing = 1.2 if self.body is Body.BLOCK else self.peak
        return swing + (0.6 if self.body is Body.WING else 0.5)

    @coalesce.coalesced(leaf=True)
    def _fit(self) -> None:
        reach = self._half_height()
        base = -reach - 1.2  # the spring's foot
        width = max(2.6 * reach, 4.0) if self.body is not Body.WING else max(2.2 * reach, 4.4)
        self.setXRange(-width / 2 - (0.8 if self.body is not Body.BLOCK else 0), width / 2, padding=0.02)
        self.setYRange(base - 0.3, reach + 0.3, padding=0.02)
        self._base = base
        self.rest.setData([-1.4, 1.4], [0.0, 0.0])
        hatch_x = np.linspace(-0.6, 0.6, 7)
        hx = np.repeat(hatch_x, 3)
        hy = np.tile([base, base - 0.25, np.nan], hatch_x.size)
        hx[1::3] -= 0.18
        self.anchor.setData(np.concatenate(([-0.7, 0.7, np.nan], hx)), np.concatenate(([base, base, np.nan], hy)))
        if self.body is Body.BLOCK:
            self.flow.setData([], [])
        else:
            left = -width / 2 - 0.6
            ys = np.linspace(-reach + 0.2, reach - 0.2, FLOW_ROWS)
            fx, fy = [], []
            for y in ys:
                fx += [left, left + 0.9, np.nan, left + 0.9, left + 0.72, np.nan, left + 0.9, left + 0.72, np.nan]
                fy += [y, y, np.nan, y, y + 0.1, np.nan, y, y - 0.1, np.nan]
            self.flow.setData(fx, fy)
        self._draw()

    @coalesce.coalesced(leaf=True)
    def _draw(self) -> None:
        if not hasattr(self, "_base"):
            return
        y, theta = self.state
        if self.body is Body.BLOCK:
            y = y / self.peak * 1.2
        c = colors
        self.shape.setBrush(pg.mkBrush(c.force if self.unstable else c.mass[0]))
        if self.body is Body.WING:
            pts = WING - [self.axis_x, 0.0]  # about the elastic axis
            ca, sa = math.cos(-theta), math.sin(-theta)  # nose (left) up is clockwise
            pts = pts @ np.array([[ca, sa], [-sa, ca]]) + [self.axis_x, y]
            bottom = y
        elif self.body is Body.CIRCLE:
            a = np.linspace(0.0, 2 * math.pi, 40)
            pts = np.column_stack((0.5 * np.cos(a), y + 0.5 * np.sin(a)))
            bottom = y - 0.5
        else:
            pts = np.array([[-0.5, y - 0.5], [0.5, y - 0.5], [0.5, y + 0.5], [-0.5, y + 0.5]])
            bottom = y - 0.5
        self.shape.setPolygon(QtGui.QPolygonF([QtCore.QPointF(float(px), float(py)) for px, py in pts]))
        r = 0.07
        self.pivot.setVisible(self.body is Body.WING)
        self.pivot.setRect(self.axis_x - r, y - r, 2 * r, 2 * r)
        sx = self.axis_x if self.body is Body.WING else 0.0
        top, foot = bottom, self._base
        s = np.linspace(0.0, 1.0, 26)
        zig = np.where((np.arange(26) % 2 == 1) & (s > 0.1) & (s < 0.9), np.where(np.arange(26) % 4 == 1, 1, -1), 0)
        self.spring.setData(sx + 0.15 * zig, foot + (top - foot) * s)
