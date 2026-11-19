"""The Pendulums page's pendulum, drawn at an angle and plucked with the mouse.

Each kind is drawn to scale (in units of its reach R, the pivot to its farthest point): a bob on a string, a rod
on its pivot, a disc on a wire seen from above (θ its twist), or a mass upright on an arm with a torsion spring
at its foot. A dashed line marks rest, and a faint ghost the linearized swing at the same moment.

The reader grabs the body (the bob, the rod's far end, the disc's mark, the upright mass) and turns it about
the pivot: `grabbed()`, then `turned(θ)` (radians, signed) while held, then `released()`. The page sets its
start angle from it and plays the swing from there.

Working on it: the drawing is `set_angle(θ, ghost)` after `set_pendulum(p)`; the angle convention (θ from
straight down for a hanging pendulum, from straight up for an inverted one, the disc's mark from its rest
direction) is `_point` and `_angle_at`, each the other's inverse. `tests/test_playback.py` plucks it.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.pendulum import Kind, Pendulum
from . import coalesce
from .style import colors

PLUCK_TIP = (
    "<p>The pendulum, swinging with the curve below it (drawn to scale). The faint ghost is the linearized "
    "swing at the same moment: watch it drift ahead on a large swing.</p>"
    "<p><b>Pluck it:</b> drag the bob (the rod's end, the disc's mark) round its pivot and let go. It swings "
    "from that angle, from rest.</p>"
)
SPAN = 1.18  # the view's half-width, in reaches
GRAB = 0.3  # how near the handle a press grabs it, in reaches
LIMIT = math.radians(179.0)  # the farthest it can be turned (the start box's limit)


def _hatch(x0: float, x1: float, y: float, down: bool) -> tuple[np.ndarray, np.ndarray]:
    """A support: a line from x0 to x1 at y with hatching on its far side (above it, or below if `down`)."""
    xs, ys = [x0, x1, np.nan], [y, y, np.nan]
    step = (x1 - x0) / 6
    sign = -1.0 if down else 1.0
    for k in range(7):
        x = x0 + k * step
        xs += [x, x - step, np.nan]
        ys += [y, y + sign * abs(step), np.nan]
    return np.array(xs), np.array(ys)


class PendulumView(pg.PlotWidget):
    grabbed = QtCore.Signal()
    turned = QtCore.Signal(float)  # θ (rad), signed
    released = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("The pendulum, swinging")
        self.setToolTip(PLUCK_TIP)
        self.setMouseTracking(True)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setAspectLocked(True)
        self.pendulum = Pendulum()
        self.reach = 1.0
        self.theta = 0.0
        self._ghost_at: float | None = None  # the linearized ghost's angle, or None
        self.held = False
        self.support = pg.PlotDataItem(connect="finite")
        self.rest = pg.PlotDataItem()
        self.ghost = pg.PlotDataItem()
        self.body = pg.PlotDataItem(connect="finite")
        self.disc = QtWidgets.QGraphicsEllipseItem()
        self.bob = QtWidgets.QGraphicsEllipseItem()
        self.ghost_bob = QtWidgets.QGraphicsEllipseItem()
        self.spring = pg.PlotDataItem()
        self.pivot = QtWidgets.QGraphicsEllipseItem()
        self.label = pg.TextItem(anchor=(0.0, 0.0))
        for z, item in enumerate(
            (self.disc, self.support, self.rest, self.spring, self.ghost, self.ghost_bob, self.body, self.bob)
        ):
            item.setZValue(z)
            self.addItem(item)
        self.pivot.setZValue(20)
        self.addItem(self.pivot)
        self.addItem(self.label)
        self.apply_theme()
        self.set_pendulum(self.pendulum)

    def apply_theme(self) -> None:
        c = colors
        self.support.setPen(pg.mkPen(c.structure, width=1.5))
        self.rest.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.ghost.setPen(pg.mkPen(c.faint, width=3))
        self.ghost_bob.setPen(pg.mkPen(c.faint, width=1.5))
        self.ghost_bob.setBrush(pg.mkBrush(None))
        self._body_pen()
        self.bob.setPen(pg.mkPen(c.structure, width=1.5))
        self.bob.setBrush(pg.mkBrush(c.mass[0]))
        fill = pg.mkColor(c.mass[0])
        fill.setAlpha(70)
        self.disc.setPen(pg.mkPen(c.mass[0], width=2))
        self.disc.setBrush(pg.mkBrush(fill))
        self.spring.setPen(pg.mkPen(c.structure, width=1.5))
        self.pivot.setPen(pg.mkPen(c.structure, width=1))
        self.pivot.setBrush(pg.mkBrush(c.background))
        self.label.setColor(c.muted)
        self.set_angle(self.theta)

    def _body_pen(self) -> None:
        """A string or arm thin, a rod thick in the mass's colour, the disc's mark bold."""
        if self.kind is Kind.ROD:
            self.body.setPen(pg.mkPen(colors.mass[0], width=9))
        else:
            self.body.setPen(pg.mkPen(colors.structure, width=3 if self.kind is Kind.TORSION else 2))

    # ------------------------------------------------------------- drawing
    @property
    def kind(self) -> Kind:
        return self.pendulum.kind

    def set_pendulum(self, p: Pendulum, amplitude: float = math.pi) -> None:
        """A new pendulum: its supports, rest line and view, in units of its reach. A swing within ±90°
        (`amplitude`, rad) needs only the half of the view it swings in: that half, drawn larger."""
        self.pendulum = p
        self._body_pen()
        if p.kind is Kind.ROD:
            self.reach = p.pivot + p.length / 2
        else:
            self.reach = p.length
        s = SPAN
        if p.kind is Kind.TORSION or not p.stable or amplitude > math.pi / 2:  # an unstable one falls
            self.setRange(QtCore.QRectF(-s, -s, 2 * s, 2 * s), padding=0)
        elif p.kind is Kind.INVERTED:
            self.setRange(QtCore.QRectF(-s, -0.2, 2 * s, s + 0.2), padding=0)
        else:
            self.setRange(QtCore.QRectF(-s, -s, 2 * s, s + 0.2), padding=0)
        circle = lambda item, x, y, r: item.setRect(QtCore.QRectF(x - r, y - r, 2 * r, 2 * r))  # noqa: E731
        circle(self.pivot, 0.0, 0.0, 0.035)
        torsion = p.kind is Kind.TORSION
        self.disc.setVisible(torsion)
        self.spring.setVisible(p.kind is Kind.INVERTED)
        if torsion:
            circle(self.disc, 0.0, 0.0, 1.0)
            self.support.setData([], [])
            self.rest.setData([0.0, 0.0], [0.0, 1.08])
            self.label.setText("Seen from above:\nthe wire at the centre")
            self.label.setPos(1.15, 1.0)  # beside the disc: the view is wider than it is tall
        elif p.kind is Kind.INVERTED:
            self.support.setData(*_hatch(-0.25, 0.25, -0.06, down=True))
            self.rest.setData([0.0, 0.0], [0.0, 1.1])
            turns = np.linspace(0.0, 4.5 * math.pi, 120)
            r = 0.04 + 0.022 * turns / math.pi
            self.spring.setData(r * np.cos(turns), r * np.sin(turns))
            self.label.setText("")
        else:
            self.support.setData(*_hatch(-0.25, 0.25, 0.0, down=False))
            self.rest.setData([0.0, 0.0], [0.0, -1.1])
            self.label.setText("")
        self.set_angle(self.theta)

    def _point(self, theta: float, r: float = 1.0) -> tuple[float, float]:
        """Where a point r reaches out along the body is, at angle θ."""
        if self.kind is Kind.SIMPLE or self.kind is Kind.ROD:
            return r * math.sin(theta), -r * math.cos(theta)
        return r * math.sin(theta), r * math.cos(theta)  # up, or the disc's mark from its rest direction

    def _angle_at(self, x: float, y: float) -> float:
        if self.kind is Kind.SIMPLE or self.kind is Kind.ROD:
            return math.atan2(x, -y)
        return math.atan2(x, y)

    def _shape(self, theta: float) -> tuple[list[float], list[float], tuple[float, float] | None]:
        """The body's lines at θ, and the centre of its bob (if it has one)."""
        p, R = self.pendulum, self.reach
        if p.kind is Kind.ROD:
            near, far = (p.pivot - p.length / 2) / R, 1.0
            (x0, y0), (x1, y1) = self._point(theta, near), self._point(theta, far)
            return [x0, x1], [y0, y1], None
        if p.kind is Kind.TORSION:
            x, y = self._point(theta)
            return [0.0, x], [0.0, y], None
        x, y = self._point(theta)
        return [0.0, x], [0.0, y], (x, y)

    def set_angle(self, theta: float, ghost: float | None = None) -> None:
        """Draw it at θ (rad), and the linearized ghost at `ghost` (hidden for None)."""
        self.theta, self._ghost_at = theta, ghost
        self._draw_body()

    @coalesce.coalesced(leaf=True)
    def _draw_body(self) -> None:
        theta, ghost = self.theta, self._ghost_at
        r = 0.09
        xs, ys, bob = self._shape(theta)
        self.body.setData(xs, ys)
        self.bob.setVisible(bob is not None)
        if bob is not None:
            self.bob.setRect(QtCore.QRectF(bob[0] - r, bob[1] - r, 2 * r, 2 * r))
        show_ghost = ghost is not None and math.isfinite(ghost) and not self.held
        self.ghost.setVisible(show_ghost)
        self.ghost_bob.setVisible(False)
        if show_ghost:
            assert ghost is not None, "shown only when given"
            gx, gy, gbob = self._shape(ghost)
            self.ghost.setData(gx, gy)
            if gbob is not None:
                self.ghost_bob.setRect(QtCore.QRectF(gbob[0] - r, gbob[1] - r, 2 * r, 2 * r))
                self.ghost_bob.setVisible(True)

    # ------------------------------------------------------------------ pluck
    def _view_pos(self, event) -> QtCore.QPointF:
        return self.plotItem.vb.mapSceneToView(self.mapToScene(event.position().toPoint()))

    def on_handle(self, pos: QtCore.QPointF) -> bool:
        """Whether a point (view coordinates) is on the body's handle: the bob, the rod's end, the disc's mark."""
        hx, hy = self._point(self.theta)
        return math.hypot(pos.x() - hx, pos.y() - hy) <= GRAB

    def _turn_to(self, pos: QtCore.QPointF) -> None:
        theta = max(-LIMIT, min(LIMIT, self._angle_at(pos.x(), pos.y())))
        self.set_angle(theta)
        self.turned.emit(theta)

    def mousePressEvent(self, event) -> None:
        pos = self._view_pos(event)
        if event.button() != QtCore.Qt.MouseButton.LeftButton or not self.on_handle(pos):
            super().mousePressEvent(event)
            return
        self.held = True
        self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        self.grabbed.emit()
        self._turn_to(pos)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        pos = self._view_pos(event)
        if self.held:
            self._turn_to(pos)
            event.accept()
            return
        if self.on_handle(pos):
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
