"""Vibration isolation's machine drawn on its mounts, moving with the run-up, its unbalance turning, and pulled.

Bottom to top: the floor, with two arrows under it: the force the mounts put on it (solid, in the force
colour) and what a rigid mount would put there at that instant, the unbalance's own force (dashed, grey), to
the same scale. Then the mounts (a spring and a damper; the relaxation mount's damper on a spring of its own),
an intermediate mass and a second set of mounts for two stages, an inertia block when there is one, and the
machine, with its unbalance turning on a crank (up is φ = 0: its force along the mount is m_e e Ω² cos φ).

- `set_layout(mount, block)`: which mounts to draw, and the block.
- `set_scale(peak, force_scale)`: `peak` (m) is drawn `SWING` from rest; `force_scale` (N) is an arrow's length.
- `set_state(x, stage, phase, floor, force)`: the machine's and intermediate mass's displacements (m), the
  unbalance's angle (rad) and the two forces (N).
- Grab the machine and pull it up or down: `grabbed()`, `pulled(x)` with the displacement (m) under the hand,
  then `released()`.

The axes aren't locked to one aspect (the view is wide and short), so the unbalance's disc is sized in
pixels, from the view's pixel size, to stay round.

Working on it: `tests/test_playback.py` checks the machine follows the run-up and a pull lets it go.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.isolation import Mount
from .animation import damper_path, spring_path
from .style import colors

FLOOR_HALF = 1.6
MOUNT_X = 0.6  # the mounts' centres, either side
LEG = 0.11  # a mount's spring and damper either side of its centre
MACHINE_HALF, MACHINE_H = 0.8, 0.65
BLOCK_HALF, BLOCK_H = 1.05, 0.25
PLATE_HALF, PLATE_H = 0.95, 0.14
ONE_STAGE = 1.0  # the mounts' height, one stage
LOWER, UPPER = 0.55, 0.55  # two stages: below and above the intermediate mass
SWING = 0.22  # the largest displacement, drawn
MOST = 0.3  # the farthest the machine can be pulled
ARROW_Y, ARROW_LEN = -0.45, 0.32  # the force arrows' centre and largest half length
DISC_PX, DOT_PX = 17.0, 5.0  # the unbalance's disc and mass (pixels)
Y_BOTTOM = -0.82  # under the force arrows; the top is fitted to the machine (`_fit`)
X_RANGE = (-2.3, 2.3)
TIP = (
    "<p>The machine on its mounts, its unbalance turning, moving with the run-up. Under the floor: the force the "
    "mounts put on it (solid) and what a rigid mount would (dashed), to the same scale.</p>"
    "<p><b>Pull</b> the machine up or down and let go: stopped, it rings on its mounts.</p>"
)


def _vertical(along: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """A horizontal path (`spring_path`, `damper_path`) stood on end: x along it becomes y, across becomes x."""
    xs, ys = along
    return ys, xs


class IsolationView(pg.PlotWidget):
    grabbed = QtCore.Signal()
    pulled = QtCore.Signal(float)  # the machine's displacement under the hand (m)
    released = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("The machine on its mounts, moving")
        self.setToolTip(TIP)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.setMouseTracking(True)
        self.hideButtons()
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.gain = SWING / 1e-4  # display units per metre
        self.force_scale = 1.0
        self.held = False
        self._grip = 0.0
        self.mount = Mount.VISCOUS
        self.block = False
        self.state = (0.0, 0.0, 0.0, 0.0, 0.0)  # x, stage, phase, floor, force
        self.floor_line = self._add(pg.PlotDataItem())
        self.hatch = self._add(pg.PlotDataItem(connect="finite"))
        self.rest = self._add(pg.PlotDataItem())
        self.springs = self._add(pg.PlotDataItem(connect="finite"))
        self.dampers = self._add(pg.PlotDataItem(connect="finite"))
        self.plate = self._add(QtWidgets.QGraphicsRectItem())
        self.block_rect = self._add(QtWidgets.QGraphicsRectItem())
        self.machine = self._add(QtWidgets.QGraphicsRectItem())
        self.disc = self._add(QtWidgets.QGraphicsEllipseItem())
        self.crank = self._add(pg.PlotDataItem())
        self.dot = self._add(QtWidgets.QGraphicsEllipseItem())
        self.floor_arrow = self._add(pg.PlotDataItem())
        self.floor_head = self._add(pg.ArrowItem(headLen=10, tipAngle=40, pen=None))
        self.rigid_arrow = self._add(pg.PlotDataItem())
        self.rigid_head = self._add(pg.ArrowItem(headLen=10, tipAngle=40, pen=None))
        self.floor_text = self._add(pg.TextItem("on the floor", anchor=(1, 0.5)))
        self.rigid_text = self._add(pg.TextItem("rigid mount", anchor=(0, 0.5)))
        for z, item in enumerate((self.machine, self.disc, self.crank, self.dot)):
            item.setZValue(5 + z)
        self._fit()
        self._built = True
        self.apply_theme()

    def _add(self, item):
        self.addItem(item)
        return item

    def apply_theme(self) -> None:
        c = colors
        pen = pg.mkPen(c.structure, width=2)
        self.floor_line.setPen(pg.mkPen(c.structure, width=3))
        for item in (self.hatch, self.springs, self.dampers, self.crank):
            item.setPen(pen)
        self.rest.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.machine.setBrush(pg.mkBrush(c.mass[0]))
        for rect in (self.machine, self.plate, self.block_rect, self.disc, self.dot):
            rect.setPen(pg.mkPen(c.structure, width=1.5))
        self.plate.setBrush(pg.mkBrush(c.mass[1]))
        self.block_rect.setBrush(pg.mkBrush(c.grey))
        self.disc.setBrush(pg.mkBrush(c.background))
        self.dot.setBrush(pg.mkBrush(c.force))
        self.floor_arrow.setPen(pg.mkPen(c.force, width=4))
        self.floor_head.setStyle(brush=c.force)
        self.rigid_arrow.setPen(pg.mkPen(c.grey, width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.rigid_head.setStyle(brush=c.grey)
        self.floor_text.setColor(c.force)
        self.rigid_text.setColor(c.muted)
        self._draw_static()
        self._draw()

    # ------------------------------------------------------------- structure
    def set_layout(self, mount: Mount, block: bool) -> None:
        self.mount, self.block = mount, block
        self._fit()
        self._draw_static()
        self._draw()

    def set_scale(self, peak: float, force_scale: float) -> None:
        if not self.held:  # under the hand the scale holds still
            self.gain = SWING / max(abs(peak), 1e-9)
        self.force_scale = max(force_scale, 1e-12)
        self._draw()

    def set_state(self, x: float, stage: float, phase: float, floor: float, force: float) -> None:
        self.state = (x, stage, phase, floor, force)
        self._draw()

    def _fit(self) -> None:
        """The view from under the arrows to over the machine at the top of its swing."""
        top = self._rest_centre() + MACHINE_H / 2 + MOST + 0.05
        self.setRange(QtCore.QRectF(X_RANGE[0], Y_BOTTOM, X_RANGE[1] - X_RANGE[0], top - Y_BOTTOM), padding=0)

    def _base(self) -> float:
        """The machine's (or its block's) underside at rest."""
        return LOWER + PLATE_H + UPPER if self.mount is Mount.TWO_STAGE else ONE_STAGE

    def _draw_static(self) -> None:
        self.floor_line.setData([-FLOOR_HALF, FLOOR_HALF], [0.0, 0.0])
        hx: list[float] = []
        hy: list[float] = []
        for x in np.linspace(-FLOOR_HALF + 0.1, FLOOR_HALF, 16):
            hx += [float(x), float(x) - 0.12, np.nan]
            hy += [0.0, -0.12, np.nan]
        self.hatch.setData(hx, hy)
        base = self._base() + (BLOCK_H if self.block else 0.0)
        self.rest.setData([MACHINE_HALF + 0.08, MACHINE_HALF + 0.55], [base, base])
        self.plate.setVisible(self.mount is Mount.TWO_STAGE)
        self.block_rect.setVisible(self.block)

    # ---------------------------------------------------------------- drawing
    def _mounts(self, y0: float, y1: float) -> tuple[list[float], list[float], list[float], list[float]]:
        """Both mounts between heights y0 and y1: (spring xs, ys, damper xs, ys), NaN-separated."""
        sx: list[float] = []
        sy: list[float] = []
        dx: list[float] = []
        dy: list[float] = []
        gap = y1 - y0
        for side in (-1.0, 1.0):
            cx = side * MOUNT_X
            a, b = _vertical(spring_path(y0, y1, cx - LEG, coils=5, amp=0.06))
            sx += [*a, np.nan]
            sy += [*b, np.nan]
            if self.mount is Mount.RELAXATION:  # the damper sits on a spring of its own
                mid = y0 + 0.4 * gap
                a, b = _vertical(spring_path(y0, mid, cx + LEG, coils=3, amp=0.05))
                sx += [*a, np.nan]
                sy += [*b, np.nan]
                a, b = _vertical(damper_path(mid, y1, cx + LEG, 0.6 * gap, h=0.06))
            else:
                a, b = _vertical(damper_path(y0, y1, cx + LEG, gap, h=0.06))
            dx += [*a, np.nan]
            dy += [*b, np.nan]
        return sx, sy, dx, dy

    def _draw(self) -> None:
        x, stage, phase, floor, force = self.state
        lift = float(np.clip(self.gain * x, -MOST - 0.1, MOST + 0.1))
        sx: list[float] = []
        sy: list[float] = []
        dx: list[float] = []
        dy: list[float] = []
        if self.mount is Mount.TWO_STAGE:
            up = float(np.clip(self.gain * stage, -MOST, MOST))
            plate = LOWER + up
            self.plate.setRect(QtCore.QRectF(-PLATE_HALF, plate, 2 * PLATE_HALF, PLATE_H))
            for part in (self._mounts(0.0, plate), self._mounts(plate + PLATE_H, self._base() + lift)):
                sx += part[0]
                sy += part[1]
                dx += part[2]
                dy += part[3]
        else:
            sx, sy, dx, dy = self._mounts(0.0, self._base() + lift)
        self.springs.setData(sx, sy)
        self.dampers.setData(dx, dy)
        bottom = self._base() + lift
        if self.block:
            self.block_rect.setRect(QtCore.QRectF(-BLOCK_HALF, bottom, 2 * BLOCK_HALF, BLOCK_H))
            bottom += BLOCK_H
        self.machine.setRect(QtCore.QRectF(-MACHINE_HALF, bottom, 2 * MACHINE_HALF, MACHINE_H))

        # The unbalance: a disc and a mass on a crank, round in pixels.
        px, py = self._pixel()
        cy = bottom + MACHINE_H / 2
        self.disc.setRect(QtCore.QRectF(-DISC_PX * px, cy - DISC_PX * py, 2 * DISC_PX * px, 2 * DISC_PX * py))
        r = 0.7 * DISC_PX
        ex, ey = r * px * math.sin(phase), cy + r * py * math.cos(phase)
        self.crank.setData([0.0, ex], [cy, ey])
        self.dot.setRect(QtCore.QRectF(ex - DOT_PX * px, ey - DOT_PX * py, 2 * DOT_PX * px, 2 * DOT_PX * py))

        # The forces under the floor: up is positive (the mounts pulling the floor up).
        for arrow, head, value, at in (
            (self.floor_arrow, self.floor_head, floor, -0.25),
            (self.rigid_arrow, self.rigid_head, force, 0.25),
        ):
            half = ARROW_LEN * float(np.clip(value / self.force_scale, -1.0, 1.0))
            arrow.setData([at, at], [ARROW_Y - half, ARROW_Y + half])
            head.setPos(at, ARROW_Y + half)
            head.setStyle(angle=90 if half >= 0 else -90)
            head.setVisible(abs(half) > 0.02)
        self.floor_text.setPos(-0.35, ARROW_Y)
        self.rigid_text.setPos(0.35, ARROW_Y)

    def _pixel(self) -> tuple[float, float]:
        """View units per pixel (x, y), or a guess before the view is laid out."""
        size = self.plotItem.vb.viewPixelSize()
        if size[0] > 0 and size[1] > 0 and math.isfinite(size[0]) and math.isfinite(size[1]):
            return float(size[0]), float(size[1])
        return 0.01, 0.02

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, "_built", False):  # (pyqtgraph resizes it while it's built, before its items are)
            self._draw()

    # ------------------------------------------------------------------- pull
    def _view_pos(self, event) -> QtCore.QPointF:
        return self.plotItem.vb.mapSceneToView(self.mapToScene(event.position().toPoint()))

    def on_machine(self, pos: QtCore.QPointF) -> bool:
        rect = self.machine.rect()
        return rect.adjusted(-0.05, -0.05, 0.05, 0.05).contains(pos)

    def _rest_centre(self) -> float:
        return self._base() + (BLOCK_H if self.block else 0.0) + MACHINE_H / 2

    def _pull(self, pos: QtCore.QPointF) -> None:
        offset = float(np.clip(pos.y() - self._grip - self._rest_centre(), -MOST, MOST))
        self.pulled.emit(offset / self.gain)

    def mousePressEvent(self, event) -> None:
        pos = self._view_pos(event)
        if event.button() != QtCore.Qt.MouseButton.LeftButton or not self.on_machine(pos):
            super().mousePressEvent(event)
            return
        self.held = True
        self._grip = pos.y() - self.machine.rect().center().y()  # held where it was taken hold of
        self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        self.grabbed.emit()
        self._pull(pos)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        pos = self._view_pos(event)
        if self.held:
            self._pull(pos)
            event.accept()
            return
        if self.on_machine(pos):
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if not self.held:
            super().mouseReleaseEvent(event)
            return
        self.held = False
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self.released.emit()
        event.accept()
