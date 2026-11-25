"""Coordinates & bases' two masses drawn (#478 items 1, 4): between their walls, moving as they run, with grips.

`ChainView` only draws and forwards what the hand does; what a drag means is the view model's
(`view_models/bases.py`'s `GRIPS`, `live_bases.LiveBases`' `hold`/`let_go`):

- **Pluck:** press on a mass and drag it sideways, the other held where it is: `grabbed(i)`, `pulled(i, x)` (m,
  the mass's displacement) as it moves, `released()`.
- **Grips** (rings): on each mass's corner (up or right: heavier) and each spring (up: stiffer):
  `grip_taken(name)`, `grip_moved(name, distance)` (display units since it was taken, + to grow),
  `grip_released(name)`; the names are `GRIPS`' (m1, m2, k1, k2, k3).

The motion is drawn magnified, `GAIN` display units a metre, the same as the plane's reach always: the plane beside
it is in mm and never rescales, so neither does this.

Working on it: `tests/test_bases_page.py` drags the grips and plucks with synthetic mouse events.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..view_models.bases import PLUCK
from ..view_models.grips import body_size
from ..view_models.text import fmt
from ..view_models.text import number as sig
from .animation import spring_path
from .drag import Grabbable
from .style import colors, text_on

WALL_L, WALL_R = 0.0, 3.6
REST = (1.2, 2.4)  # the masses' centres at rest (display units)
SIDE = 0.4  # the 1 kg mass's side
GAIN = 6.0  # display units a metre: 50 mm is 0.3
GRIP_R = 0.06
TIP = (
    "<p>The two masses between their walls, moving as they run (magnified). The coupling spring k₂ is the one "
    "between them.</p>"
    "<p><b>Pluck:</b> drag a mass sideways and let go (the other stays where it is), or drag the point on the plane "
    "below. <b>Change them as they run:</b> drag the ring on a mass's corner up for a heavier one, or the ring on a "
    "spring up for a stiffer one. The motion carries on from where it is.</p>"
)


class ChainView(Grabbable, pg.PlotWidget):
    grabbed = QtCore.Signal(int)
    pulled = QtCore.Signal(int, float)  # the mass, its displacement (m)
    released = QtCore.Signal()
    grip_taken = QtCore.Signal(str)
    grip_moved = QtCore.Signal(str, float)
    grip_released = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("The two masses between their walls, moving")
        self.setToolTip(TIP)
        self.setMouseTracking(True)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setAspectLocked(True)
        self.setRange(QtCore.QRectF(-0.15, -0.45, WALL_R + 0.3, 0.9), padding=0)
        self.held: object = False  # a mass's index while plucked, a grip's name while it's dragged
        self._grip_from: QtCore.QPointF | None = None
        self._grab_dx = 0.0
        self.x = (0.0, 0.0)
        self.sides = [SIDE, SIDE]
        self.values = {"m1": 1.0, "m2": 1.0, "k1": 400.0, "k2": 100.0, "k3": 400.0}

        def add(item, z: float = 0.0):
            item.setZValue(z)
            self.addItem(item)
            return item

        self.walls = add(pg.PlotDataItem(connect="finite"))
        self.springs = [add(pg.PlotDataItem()) for _ in range(3)]
        self.blocks = [add(QtWidgets.QGraphicsRectItem(), 5) for _ in range(2)]
        self.mass_labels = [add(pg.TextItem(anchor=(0.5, 0.5)), 6) for _ in range(2)]
        self.k_labels = [add(pg.TextItem(anchor=(0.5, 0.0)), 6) for _ in range(3)]
        self.rests = add(pg.PlotDataItem(connect="finite"))
        self.grips = {name: add(QtWidgets.QGraphicsEllipseItem(), 8) for name in ("m1", "m2", "k1", "k2", "k3")}
        self._draw_walls()
        self.apply_theme()
        self.show_at((0.0, 0.0))

    def _draw_walls(self) -> None:
        xs, ys = [WALL_L, WALL_L, float("nan"), WALL_R, WALL_R], [-0.35, 0.35, float("nan"), -0.35, 0.35]
        for wall, side in ((WALL_L, -1.0), (WALL_R, 1.0)):
            for k in range(8):
                y = -0.32 + 0.09 * k
                xs += [float("nan"), wall, wall + side * 0.08]
                ys += [float("nan"), y, y + 0.08]
        self.walls.setData(xs, ys)
        rx, ry = [], []
        for r in REST:
            rx += [r, r, float("nan")]
            ry += [-0.42, -0.34, float("nan")]
        self.rests.setData(rx, ry)

    def apply_theme(self) -> None:
        pen = pg.mkPen(colors.structure, width=2)
        self.walls.setPen(pg.mkPen(colors.structure, width=2))
        for spring in self.springs:
            spring.setPen(pen)
        self.rests.setPen(pg.mkPen(colors.grey, width=1))
        for i, block in enumerate(self.blocks):
            block.setBrush(pg.mkBrush(colors.mass[i]))
            block.setPen(pg.mkPen(colors.structure, width=1.5))
            self.mass_labels[i].setColor(text_on(colors.mass[i]))
        for label in self.k_labels:
            label.setColor(colors.muted)
        for grip in self.grips.values():
            grip.setBrush(pg.mkBrush(colors.background))
            grip.setPen(pg.mkPen(colors.strong, width=2))

    # ------------------------------------------------------------- drawing
    def set_values(self, m1: float, m2: float, k1: float, k2: float, k3: float) -> None:
        self.values = {"m1": m1, "m2": m2, "k1": k1, "k2": k2, "k3": k3}
        self.sides = [SIDE * body_size(m, 1.0) * 0.8 for m in (m1, m2)]
        for i, m in enumerate((m1, m2)):
            self.mass_labels[i].setText(f"{sig(m)} kg")
        for i, k in enumerate((k1, k2, k3)):
            self.k_labels[i].setText(f"k{'₁₂₃'[i]} = {fmt(k, 'N/m', largest='M', smallest='m')}")
        # drawn at the next `show_at`: the page's next draw (a trial's sets each curve once)

    def show_at(self, x: Sequence[float] | np.ndarray) -> None:
        """The masses at displacements x (m)."""
        self.x = (float(x[0]), float(x[1]))
        centres = [REST[i] + GAIN * self.x[i] for i in range(2)]
        edges = [(c - s / 2, c + s / 2) for c, s in zip(centres, self.sides)]
        for i, (left, _) in enumerate(edges):
            s = self.sides[i]
            self.blocks[i].setRect(QtCore.QRectF(left, -s / 2, s, s))
            self.mass_labels[i].setPos(centres[i], 0.0)
        spans = ((WALL_L, edges[0][0]), (edges[0][1], edges[1][0]), (edges[1][1], WALL_R))
        for i, (a, b) in enumerate(spans):
            self.springs[i].setData(*spring_path(a, b, 0.0, coils=5))
            self.k_labels[i].setPos((a + b) / 2, -0.08)
        for name, (gx, gy) in self._grip_spots().items():
            self.grips[name].setRect(QtCore.QRectF(gx - GRIP_R, gy - GRIP_R, 2 * GRIP_R, 2 * GRIP_R))

    def _centre(self, i: int) -> float:
        return REST[i] + GAIN * self.x[i]

    def _grip_spots(self) -> dict[str, tuple[float, float]]:
        c = [self._centre(i) for i in range(2)]
        s = self.sides
        edges = [(c[i] - s[i] / 2, c[i] + s[i] / 2) for i in range(2)]
        return {
            "m1": (edges[0][1], s[0] / 2),
            "m2": (edges[1][1], s[1] / 2),
            "k1": ((WALL_L + edges[0][0]) / 2, 0.0),
            "k2": ((edges[0][1] + edges[1][0]) / 2, 0.0),
            "k3": ((edges[1][1] + WALL_R) / 2, 0.0),
        }

    # ------------------------------------------------------------- the hand
    def grip_at(self, pos: QtCore.QPointF) -> str | None:
        for name, (gx, gy) in self._grip_spots().items():
            if (pos.x() - gx) ** 2 + (pos.y() - gy) ** 2 <= (1.5 * GRIP_R) ** 2:
                return name
        return None

    def grab_at(self, pos: QtCore.QPointF) -> object | None:
        grip = self.grip_at(pos)
        if grip is not None:
            return grip
        for i in range(2):
            s = self.sides[i]
            if abs(pos.x() - self._centre(i)) <= s / 2 and abs(pos.y()) <= s / 2:
                return i
        return None

    def _holding(self) -> bool:
        return self.held is not False

    def _take(self, what: object, pos: QtCore.QPointF) -> None:
        if isinstance(what, str):
            self._grip_from = pos
            self.grip_taken.emit(what)
            return
        assert isinstance(what, int), "a mass's index"
        self._grab_dx = pos.x() - self._centre(what)
        self.grabbed.emit(what)
        self._pull(pos)

    def _pull(self, pos: QtCore.QPointF) -> None:
        what = self.held
        if isinstance(what, str):
            start = self._grip_from
            assert start is not None, "a grip's drag starts where it was taken"
            dx, dy = pos.x() - start.x(), pos.y() - start.y()
            self.grip_moved.emit(what, dx + dy if what.startswith("m") else dy)
            return
        assert isinstance(what, int), "a mass's index"
        x = (pos.x() - self._grab_dx - REST[what]) / GAIN
        self.pulled.emit(what, max(-PLUCK, min(PLUCK, x)))

    def _let_go(self, what: object) -> None:
        if isinstance(what, str):
            self._grip_from = None
            self.grip_released.emit(what)
            return
        self.released.emit()

    def press_point(self, name: str | int) -> QtCore.QPointF:
        """Where a grip (by name) or a mass (by index) is, in view coordinates (tests drag them)."""
        if isinstance(name, int):
            return QtCore.QPointF(self._centre(name), -0.25 * self.sides[name])
        gx, gy = self._grip_spots()[name]
        return QtCore.QPointF(gx, gy)
