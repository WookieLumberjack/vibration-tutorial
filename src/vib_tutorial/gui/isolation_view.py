"""Vibration isolation's machine drawn on its mounts, live (#478 Phase 4): its unbalance turning, the floor moving
when it shakes, pulled, and grips that change the machine and its mounts as it runs.

Bottom to top: the floor, with two arrows under it: the force the mounts put on it (solid, in the force
color) and what a rigid mount would put there at that instant, the unbalance's own force (dashed, gray), to
the same scale. Then the mounts (a spring and a damper; the relaxation mount's damper on a spring of its own),
an intermediate mass and a second set of mounts for two stages, an inertia block when there is one, and the
machine, with its unbalance turning on a crank (up is φ = 0: its force along the mount is m_e e Ω² cos φ).

- `set_layout(mount, block)`: which mounts to draw, and the block.
- `set_scale(peak, force_scale)`: `peak` (m) is drawn `SWING` from rest; `force_scale` (N) is an arrow's length.
- `set_state(x, stage, phase, floor, force, g)`: the machine's and intermediate mass's displacements (m), the
  unbalance's angle (rad), the two forces (N) and the floor's displacement (m).
- `set_values(mass, sag_mm, zeta)`: what the grips' labels say.
- Grab the machine and pull it up or down: `grabbed()`, `pulled(x)` with the displacement (m) under the hand,
  then `released()`.
- **Grips** (`view_models/live_isolation.GRIPS`): the handle at the machine's corner (drag up or right: heavier),
  the one on the left mount's spring (down: it sags more, softer) and on its damper (up: more damping):
  `grip_taken(name)`, `grip_moved(name, distance)` (display units since it was taken, + to grow the value),
  `grip_released(name)`.

The axes aren't locked to one aspect (the view is wide and short), so the unbalance's disk is sized in
pixels, from the view's pixel size, to stay round.

Working on it: `tests/test_isolation_page.py` pulls the machine and drags the grips with synthetic mouse events.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.isolation import Mount
from ..view_models.text import number as sig
from . import coalesce
from .animation import damper_path, spring_path
from .drag import Grabbable
from .style import colors

FLOOR_HALF = 1.6
MOUNT_X = 0.6  # the mounts' centers, either side
LEG = 0.11  # a mount's spring and damper either side of its center
MACHINE_HALF, MACHINE_H = 0.8, 0.65
BLOCK_HALF, BLOCK_H = 1.05, 0.25
PLATE_HALF, PLATE_H = 0.95, 0.14
ONE_STAGE = 1.0  # the mounts' height, one stage
LOWER, UPPER = 0.55, 0.55  # two stages: below and above the intermediate mass
SWING = 0.22  # the largest displacement, drawn
MOST = 0.3  # the farthest the machine can be pulled
ARROW_Y, ARROW_LEN = -0.45, 0.32  # the force arrows' center and largest half length
DISC_PX, DOT_PX = 17.0, 5.0  # the unbalance's disk and mass (pixels)
Y_BOTTOM = -0.82  # under the force arrows; the top is fitted to the machine (`_fit`)
X_RANGE = (-2.3, 2.3)
GRIP_PX = 6.0  # a grip's radius (pixels)
GRIP_NAMES = ("mass", "deflection_mm", "zeta")
TIP = (
    "<p>The machine on its mounts, live, its unbalance turning. Under the floor: the force the mounts put on it "
    "(solid) and what a rigid mount would (dashed), to the same scale.</p>"
    "<p><b>Pull</b> the machine up or down and let go: it rings on its mounts. <b>Drag the grips</b> as it runs: the "
    "machine's corner (up or right: heavier; the mounts keep their sag), the spring (down: it sags more, softer) "
    "and the damper (up: more damping).</p>"
)


def _vertical(along: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """A horizontal path (`spring_path`, `damper_path`) stood on end: x along it becomes y, across becomes x."""
    xs, ys = along
    return ys, xs


class IsolationView(Grabbable, pg.PlotWidget):
    grabbed = QtCore.Signal()
    pulled = QtCore.Signal(float)  # the machine's displacement under the hand (m)
    released = QtCore.Signal()
    grip_taken = QtCore.Signal(str)
    grip_moved = QtCore.Signal(str, float)  # the grip, display units dragged since it was taken (+ to grow)
    grip_released = QtCore.Signal(str)

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
        self.gain = SWING / 1e-4  # display units per meter
        self.force_scale = 1.0
        self.held = False
        self._grip = 0.0
        self.mount = Mount.VISCOUS
        self.block = False
        self.state = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # x, stage, phase, floor, force, g
        self._grip_from: QtCore.QPointF | None = None
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
        self.grips = {name: self._add(QtWidgets.QGraphicsEllipseItem()) for name in GRIP_NAMES}
        for grip in self.grips.values():
            grip.setZValue(10)
        self.mass_text = self._add(pg.TextItem(anchor=(0, 1)))
        self.mass_text.setZValue(9)
        self.sag_text = self._add(pg.TextItem(anchor=(1, 0.5)))
        self.zeta_text = self._add(pg.TextItem(anchor=(0, 0.5)))
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
        for grip in self.grips.values():
            grip.setBrush(pg.mkBrush(c.background))
            grip.setPen(pg.mkPen(c.strong, width=2))
        self.mass_text.setColor(c.muted)
        for text in (self.sag_text, self.zeta_text):
            text.setColor(c.muted)
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

    def set_state(self, x: float, stage: float, phase: float, floor: float, force: float, g: float = 0.0) -> None:
        self.state = (x, stage, phase, floor, force, g)
        self._draw()

    def set_values(self, mass: float, sag_mm: float, zeta: float) -> None:
        """What the grips' labels say: the machine's mass (with its block), the mounts' sag and damping."""
        self.mass_text.setText(f"M = {sig(mass)} kg")
        self.sag_text.setText(f"δ = {sig(sag_mm)} mm")
        self.zeta_text.setText(f"ζ = {sig(zeta)}")
        self._draw()

    def _fit(self) -> None:
        """The view from under the arrows to over the machine at the top of its swing."""
        top = self._rest_centre() + MACHINE_H / 2 + MOST + 0.05
        self.setRange(QtCore.QRectF(X_RANGE[0], Y_BOTTOM, X_RANGE[1] - X_RANGE[0], top - Y_BOTTOM), padding=0)

    def _base(self) -> float:
        """The machine's (or its block's) underside at rest."""
        return LOWER + PLATE_H + UPPER if self.mount is Mount.TWO_STAGE else ONE_STAGE

    def _draw_static(self) -> None:
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

    @coalesce.coalesced(leaf=True)
    def _draw(self) -> None:
        x, stage, phase, floor, force, g = self.state
        lift = float(np.clip(self.gain * x, -MOST - 0.1, MOST + 0.1))
        ground = float(np.clip(self.gain * g, -MOST, MOST))  # the floor, shaking
        self.floor_line.setData([-FLOOR_HALF, FLOOR_HALF], [ground, ground])
        hx: list[float] = []
        hy: list[float] = []
        for hxs in np.linspace(-FLOOR_HALF + 0.1, FLOOR_HALF, 16):
            hx += [float(hxs), float(hxs) - 0.12, np.nan]
            hy += [ground, ground - 0.12, np.nan]
        self.hatch.setData(hx, hy)
        sx: list[float] = []
        sy: list[float] = []
        dx: list[float] = []
        dy: list[float] = []
        if self.mount is Mount.TWO_STAGE:
            up = float(np.clip(self.gain * stage, -MOST, MOST))
            plate = LOWER + up
            self.plate.setRect(QtCore.QRectF(-PLATE_HALF, plate, 2 * PLATE_HALF, PLATE_H))
            for part in (self._mounts(ground, plate), self._mounts(plate + PLATE_H, self._base() + lift)):
                sx += part[0]
                sy += part[1]
                dx += part[2]
                dy += part[3]
        else:
            sx, sy, dx, dy = self._mounts(ground, self._base() + lift)
        self.springs.setData(sx, sy)
        self.dampers.setData(dx, dy)
        bottom = self._base() + lift
        if self.block:
            self.block_rect.setRect(QtCore.QRectF(-BLOCK_HALF, bottom, 2 * BLOCK_HALF, BLOCK_H))
            bottom += BLOCK_H
        self.machine.setRect(QtCore.QRectF(-MACHINE_HALF, bottom, 2 * MACHINE_HALF, MACHINE_H))

        # The unbalance: a disk and a mass on a crank, round in pixels.
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

        rect = self.machine.rect()
        self.mass_text.setPos(rect.left(), rect.bottom())  # above the machine (y up)
        spots = self._grip_spots()
        for name, (gx, gy) in spots.items():
            rx, ry = GRIP_PX * px, GRIP_PX * py  # round in pixels, as the disk
            self.grips[name].setRect(QtCore.QRectF(gx - rx, gy - ry, 2 * rx, 2 * ry))
        sx_, sy_ = spots["deflection_mm"]
        self.sag_text.setPos(sx_ - 2.2 * GRIP_PX * px, sy_)
        zx, zy = spots["zeta"]
        self.zeta_text.setPos(zx + 2.2 * GRIP_PX * px, zy)

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
    def on_machine(self, pos: QtCore.QPointF) -> bool:
        rect = self.machine.rect()
        return rect.adjusted(-0.05, -0.05, 0.05, 0.05).contains(pos)

    def _rest_centre(self) -> float:
        return self._base() + (BLOCK_H if self.block else 0.0) + MACHINE_H / 2

    def _pull(self, pos: QtCore.QPointF) -> None:
        what = self.held
        if isinstance(what, str):
            start = self._grip_from
            assert start is not None, "a grip's drag starts where it was taken"
            dx, dy = pos.x() - start.x(), pos.y() - start.y()
            self.grip_moved.emit(what, {"mass": dx + dy, "deflection_mm": -dy}.get(what, dy))
            return
        offset = float(np.clip(pos.y() - self._grip - self._rest_centre(), -MOST, MOST))
        self.pulled.emit(offset / self.gain)

    def _grip_spots(self) -> dict[str, tuple[float, float]]:
        """The grips' places (view units): the machine's top right corner, the middles of the left mount's
        spring and damper (the upper stage's, on two)."""
        rect = self.machine.rect()
        x, stage, *_rest, g = self.state
        lift = float(np.clip(self.gain * x, -MOST - 0.1, MOST + 0.1))
        if self.mount is Mount.TWO_STAGE:
            low = LOWER + float(np.clip(self.gain * stage, -MOST, MOST)) + PLATE_H
        else:
            low = float(np.clip(self.gain * g, -MOST, MOST))
        mid = 0.5 * (low + self._base() + lift)
        return {
            "mass": (rect.right(), rect.top()),
            "deflection_mm": (-MOUNT_X - LEG, mid),
            "zeta": (-MOUNT_X + LEG, mid),
        }

    def grip_at(self, pos: QtCore.QPointF) -> str | None:
        px, py = self._pixel()
        for name, (gx, gy) in self._grip_spots().items():
            if ((pos.x() - gx) / px) ** 2 + ((pos.y() - gy) / py) ** 2 <= (1.6 * GRIP_PX) ** 2:
                return name
        return None

    def press_point(self, name: str) -> QtCore.QPointF:
        """Where a grip is, in view coordinates (tests drag them)."""
        gx, gy = self._grip_spots()[name]
        return QtCore.QPointF(gx, gy)

    def grab_at(self, pos: QtCore.QPointF) -> object | None:
        grip = self.grip_at(pos)
        if grip is not None:
            return grip
        return True if self.on_machine(pos) else None

    def _take(self, what: object, pos: QtCore.QPointF) -> None:
        if isinstance(what, str):
            self._grip_from = pos
            self.grip_taken.emit(what)
            return
        self._grip = pos.y() - self.machine.rect().center().y()  # held where it was taken hold of
        self.grabbed.emit()
        self._pull(pos)

    def _let_go(self, what: object) -> None:
        if isinstance(what, str):
            self._grip_from = None
            self.grip_released.emit(what)
            return
        self.released.emit()
