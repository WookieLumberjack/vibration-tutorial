"""The live pendulum drawn (#478 Phase 4): the Pendulums page's body, plucked and changed by its grips as it swings.

Each kind is drawn to scale in metres: a bob on a string, a rod on its pivot, a disc on a wire seen from above (θ its
twist), or a mass upright on an arm with a torsion spring at its foot. A dashed line marks rest, and a faint ghost
the linearized swing at the same moment. Its forces are drawn beside it, in moments about the pivot
(`free_body_view.FreeBodyView`).

`PendulumView` only draws and forwards what the hand does; what a drag means is the view model's
(`view_models/pendulum.py`'s `GRIPS`, `pivot_dragged`; `live_pendulum.LivePendulum`'s `hold`/`let_go`):

- **Pluck:** take the bob (the rod's far end, the disc's mark, the upright mass) and turn it about the pivot:
  `grabbed()`, `turned(θ)` (rad, signed) while held, `released()`.
- **Grips** (rings): on the string, rod or arm, slid along it to change its length (the disc's on its rim, its
  radius); on the bob's edge (up or out: heavier); on the spring (up: stiffer); on the rod's pivot, slid along the
  rod. `grip_taken(name)`, `grip_moved(name, value)` (the length grip: the pointer's distance from the pivot over
  where the grip was, so L follows the hand; the pivot's: metres slid toward the rod's far end; the others: reaches
  dragged, + to grow), `grip_released(name)`.

The view fits the pendulum's reach, and only half the plane while a hanging (or upright) pendulum swings within
about 80° (`frame`); it holds still while a grip is dragged, so the length is seen to change, and fits again after.

Working on it: the drawing is `show(θ, ghost)` after `set_pendulum(p)`; the angle convention (θ from straight down
for a hanging pendulum, from straight up for an inverted one, the disc's mark from its rest direction) is `_point`
and `_angle_at`, each the other's inverse. `tests/test_pendulum_page.py` plucks it and drags its grips.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.pendulum import Kind, Pendulum
from ..view_models.grips import body_size
from ..view_models.text import fmt
from ..view_models.text import number as sig
from .drag import Grabbable
from .style import colors

TIP = (
    "<p>The pendulum, swinging (drawn to scale). The faint ghost is the linearized swing, sin θ ≈ θ, at the same "
    "moment: watch it drift ahead on a large swing.</p>"
    "<p><b>Pluck it:</b> drag the bob (the rod's end, the disc's mark) round its pivot and let go. <b>Change it as "
    "it swings:</b> slide the ring on the string (rod, arm; the disc's rim) to change its length, drag the ring on "
    "the bob up for a heavier one, the ring on the spring up for a stiffer one, or slide the rod's pivot along it. "
    "The swing carries on: a shorter string swings faster at once, its angular momentum kept.</p>"
)
SPAN = 1.18  # the view's half-width, in reaches
GRAB = 0.3  # how near the handle a press grabs it, in reaches
GRIP_R = 0.045  # a grip's radius, in reaches
GRIP_AT = 0.6  # the length grip's place along the string, rod or arm
LIMIT = math.radians(179.0)  # the farthest it can be turned
FULL_ABOVE, HALF_BELOW = math.radians(80.0), math.radians(60.0)  # the whole plane past this swing, half under that
BOB = 0.09  # the 1 kg bob's radius, in reaches


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


def _circle(item: QtWidgets.QGraphicsEllipseItem, x: float, y: float, r: float) -> None:
    item.setRect(QtCore.QRectF(x - r, y - r, 2 * r, 2 * r))


class PendulumView(Grabbable, pg.PlotWidget):
    grabbed = QtCore.Signal()
    turned = QtCore.Signal(float)  # θ (rad), signed
    released = QtCore.Signal()
    grip_taken = QtCore.Signal(str)
    grip_moved = QtCore.Signal(str, float)  # the grip, what its drag says (see the module)
    grip_released = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("The pendulum, swinging")
        self.setToolTip(TIP)
        self.setMouseTracking(True)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setAspectLocked(True)
        self.pendulum = Pendulum()
        self.reach = 1.0  # m: the pivot to the body's farthest point
        self.theta = 0.0
        self._ghost_at: float | None = None
        self.full = False  # the whole plane shown, not the half it swings in
        self._fitted: tuple | None = None  # what the view's range was fitted to
        self._grip_from: QtCore.QPointF | None = None
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
        self.values = pg.TextItem(anchor=(0.0, 0.0))  # L, m (and k_t, d) in a corner
        self.grips = {name: QtWidgets.QGraphicsEllipseItem() for name in ("length", "mass", "spring", "pivot")}
        for z, item in enumerate(
            (self.disc, self.support, self.rest, self.spring, self.ghost, self.ghost_bob, self.body, self.bob)
        ):
            item.setZValue(z)
            self.addItem(item)
        self.pivot.setZValue(20)
        self.addItem(self.pivot)
        for grip in self.grips.values():
            grip.setZValue(30)
            self.addItem(grip)
        self.addItem(self.label)
        self.addItem(self.values)
        self.set_pendulum(self.pendulum)
        self.apply_theme()  # and drawn

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
        for grip in self.grips.values():
            grip.setBrush(pg.mkBrush(c.background))
            grip.setPen(pg.mkPen(c.strong, width=2))
        self.label.setColor(c.muted)
        self.values.setColor(c.muted)
        self._draw()

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

    @property
    def gripping(self) -> bool:
        return isinstance(self.held, str)

    def set_pendulum(self, p: Pendulum) -> None:
        """The pendulum drawn: its supports, rest line and values; the view fitted to it unless a grip is held."""
        self.pendulum = p
        self._body_pen()
        self.reach = p.pivot + p.length / 2 if p.kind is Kind.ROD else p.length
        R = self.reach
        _circle(self.pivot, 0.0, 0.0, 0.035 * R)
        torsion = p.kind is Kind.TORSION
        self.disc.setVisible(torsion)
        self.spring.setVisible(p.kind is Kind.INVERTED)
        if torsion:
            _circle(self.disc, 0.0, 0.0, R)
            self.support.setData([], [])
            self.rest.setData([0.0, 0.0], [0.0, 1.08 * R])
            self.label.setText("Seen from above:\nthe wire at the centre")
        elif p.kind is Kind.INVERTED:
            self.support.setData(*_hatch(-0.25 * R, 0.25 * R, -0.06 * R, down=True))
            self.rest.setData([0.0, 0.0], [0.0, 1.1 * R])
            turns = np.linspace(0.0, 4.5 * math.pi, 120)
            r = (0.04 + 0.022 * turns / math.pi) * R
            self.spring.setData(r * np.cos(turns), r * np.sin(turns))
            self.label.setText("")
        else:
            self.support.setData(*_hatch(-0.25 * R, 0.25 * R, 0.0, down=False))
            self.rest.setData([0.0, 0.0], [0.0, -1.1 * R])
            self.label.setText("")
        parts = [f"{'r' if torsion else 'L'} = {fmt(p.length, 'm')}", f"m = {sig(p.mass)} kg"]
        if p.kind is Kind.ROD:
            parts.append(f"pivot {fmt(p.pivot, 'm')} from its centre")
        if p.kind in (Kind.TORSION, Kind.INVERTED):
            parts.append(f"kₜ = {sig(p.spring)} N·m/rad")
        self.values.setText("\n".join(parts))
        self._fit()  # drawn at the next `show`

    def frame(self, amplitude: float) -> None:
        """The swing's recent largest angle (rad): the whole plane past about 80°, half of it under about 60°."""
        if not self.full and amplitude > FULL_ABOVE:
            self.full = True
        elif self.full and amplitude < HALF_BELOW:
            self.full = False
        self._fit()

    def _fit(self) -> None:
        if self.gripping:  # held still under the hand
            return
        p, s = self.pendulum, SPAN * self.reach
        whole = p.kind is Kind.TORSION or not p.stable or self.full
        fitted = (s, whole, p.kind is Kind.INVERTED)
        if fitted == self._fitted:
            return
        self._fitted = fitted
        if whole:
            rect = QtCore.QRectF(-s, -s, 2 * s, 2 * s)
        elif p.kind is Kind.INVERTED:
            rect = QtCore.QRectF(-s, -0.2 * self.reach, 2 * s, s + 0.2 * self.reach)
        else:
            rect = QtCore.QRectF(-s, -s, 2 * s, s + 0.2 * self.reach)
        self.setRange(rect, padding=0)
        self.label.setPos(1.05 * self.reach, rect.bottom() + rect.height() - 0.05 * self.reach)
        self.values.setPos(rect.left() + 0.02 * s, rect.bottom() + rect.height() - 0.02 * s)

    def _point(self, theta: float, r: float = 1.0) -> tuple[float, float]:
        """Where a point r reaches out along the body is, at angle θ (metres)."""
        r *= self.reach
        if self.kind is Kind.SIMPLE or self.kind is Kind.ROD:
            return r * math.sin(theta), -r * math.cos(theta)
        return r * math.sin(theta), r * math.cos(theta)  # up, or the disc's mark from its rest direction

    def _angle_at(self, x: float, y: float) -> float:
        if self.kind is Kind.SIMPLE or self.kind is Kind.ROD:
            return math.atan2(x, -y)
        return math.atan2(x, y)

    def _near(self) -> float:
        """The rod's near end, in reaches along it (behind the pivot: negative)."""
        p = self.pendulum
        return (p.pivot - p.length / 2) / self.reach

    def _shape(self, theta: float) -> tuple[list[float], list[float], tuple[float, float] | None]:
        """The body's lines at θ, and the centre of its bob (if it has one)."""
        if self.kind is Kind.ROD:
            (x0, y0), (x1, y1) = self._point(theta, self._near()), self._point(theta, 1.0)
            return [x0, x1], [y0, y1], None
        x, y = self._point(theta)
        if self.kind is Kind.TORSION:
            return [0.0, x], [0.0, y], None
        return [0.0, x], [0.0, y], (x, y)

    def _bob_r(self) -> float:
        return BOB * self.reach * body_size(self.pendulum.mass, 1.0)

    def grip_spots(self) -> dict[str, tuple[float, float]]:
        """Where each grip the kind has is now (metres)."""
        p, th = self.pendulum, self.theta
        spots: dict[str, tuple[float, float]] = {}
        if p.kind is Kind.TORSION:
            spots["length"] = self._point(th + 0.5 * math.pi)  # on the rim, a quarter turn round from the mark
            spots["mass"] = self._point(th - 0.5 * math.pi)
            spots["spring"] = (0.0, -0.25 * self.reach)  # by the wire, at the centre
            return spots
        if p.kind is Kind.ROD:
            spots["length"] = self._point(th, 0.85)
            spots["mass"] = self._point(th, (p.pivot / self.reach))  # at G, the rod's middle
            spots["pivot"] = (0.0, 0.0)
            return spots
        spots["length"] = self._point(th, GRIP_AT)
        bx, by = self._point(th)
        r = self._bob_r()
        spots["mass"] = (bx + 0.7 * r, by + 0.7 * r)  # on the bob's upper right
        if p.kind is Kind.INVERTED:
            spots["spring"] = (0.14 * self.reach, -0.02 * self.reach)
        return spots

    def show(self, theta: float, ghost: float | None = None) -> None:
        """Draw it at θ (rad), and the linearized ghost at `ghost` (hidden for None)."""
        self.theta, self._ghost_at = theta, ghost
        self._draw()

    def _draw(self) -> None:
        theta, ghost = self.theta, self._ghost_at
        r = self._bob_r()
        xs, ys, bob = self._shape(theta)
        self.body.setData(xs, ys)
        self.bob.setVisible(bob is not None)
        if bob is not None:
            _circle(self.bob, bob[0], bob[1], r)
        show_ghost = ghost is not None and math.isfinite(ghost) and self.held is False
        self.ghost.setVisible(show_ghost)
        self.ghost_bob.setVisible(False)
        if show_ghost:
            assert ghost is not None, "shown only when given"
            gx, gy, gbob = self._shape(ghost)
            self.ghost.setData(gx, gy)
            if gbob is not None:
                _circle(self.ghost_bob, gbob[0], gbob[1], r)
                self.ghost_bob.setVisible(True)
        spots = self.grip_spots()
        g = GRIP_R * self.reach
        for name, grip in self.grips.items():
            spot = spots.get(name)
            grip.setVisible(spot is not None)
            if spot is not None:
                _circle(grip, spot[0], spot[1], g)

    # ------------------------------------------------------------------ the hand
    def grip_at(self, pos: QtCore.QPointF) -> str | None:
        near = (1.5 * GRIP_R * self.reach) ** 2
        for name, (gx, gy) in self.grip_spots().items():
            if (pos.x() - gx) ** 2 + (pos.y() - gy) ** 2 <= near:
                return name
        return None

    def on_handle(self, pos: QtCore.QPointF) -> bool:
        """Whether a point is on the body's handle: the bob, the rod's end, the disc's mark."""
        hx, hy = self._point(self.theta)
        return math.hypot(pos.x() - hx, pos.y() - hy) <= GRAB * self.reach

    def grab_at(self, pos: QtCore.QPointF) -> object | None:
        grip = self.grip_at(pos)
        if grip is not None:
            return grip
        return True if self.on_handle(pos) else None

    def _take(self, what: object, pos: QtCore.QPointF) -> None:
        if isinstance(what, str):
            self._grip_from = pos
            self.grip_taken.emit(what)
            return
        self.grabbed.emit()
        self._pull(pos)

    def _pull(self, pos: QtCore.QPointF) -> None:
        what = self.held
        if isinstance(what, str):
            start = self._grip_from
            assert start is not None, "a grip's drag starts where it was taken"
            self.grip_moved.emit(what, self._grip_value(what, start, pos))
            return
        theta = max(-LIMIT, min(LIMIT, self._angle_at(pos.x(), pos.y())))
        self.show(theta)
        self.turned.emit(theta)

    def _grip_value(self, name: str, start: QtCore.QPointF, pos: QtCore.QPointF) -> float:
        if name == "length":  # L follows the hand: the pointer's distance from the pivot over the grip's
            return math.hypot(pos.x(), pos.y()) / max(math.hypot(start.x(), start.y()), 1e-9)
        dx, dy = pos.x() - start.x(), pos.y() - start.y()
        if name == "pivot":  # metres along the rod toward its far end
            ux, uy = self._point(self.theta, 1.0 / self.reach)
            return dx * ux + dy * uy
        return ((dx + dy) if name == "mass" else dy) / self.reach

    def _let_go(self, what: object) -> None:
        if isinstance(what, str):
            self._grip_from = None
            self.grip_released.emit(what)
            self._fit()  # now the hand is off it
            return
        self.released.emit()

    def press_point(self, name: str) -> QtCore.QPointF:
        """Where a grip is, in view coordinates (tests drag them)."""
        gx, gy = self.grip_spots()[name]
        return QtCore.QPointF(gx, gy)

    def handle_point(self) -> QtCore.QPointF:
        """The pluck's handle: the bob, the rod's end, the disc's mark (tests take it)."""
        return QtCore.QPointF(*self._point(self.theta))
