"""What the reader drags on a plot: a line that sets a value, a dot that slides along a curve.

Both look draggable before they're touched: a grip (◀▶ on a line, a ring on a dot), a pointing-hand cursor,
a wider, brighter pen while the mouse is over them, and a tooltip saying what dragging does. Both work in data
units on a log axis too (pyqtgraph's items sit in log10 coordinates there).

- `DragLine`: a vertical (or horizontal) line. `dragged(value)` while the reader drags it, `released(value)` when
  they let go; `set_value` moves it without either (from the page's own box, a trial).
- `DragDot`: a dot on a path given point by point, each with a parameter (the damper along a trade-off curve); a
  drag snaps it to the nearest point on screen and emits that point's parameter.
- `Grabbable`: a mixin for a view whose body the reader takes hold of, pulls and lets go (the pendulum, the four
  dampers' masses, the machine on its mounts, a chain's mass): the open and closed hand, and the press, move and
  release handlers. The view says what's under the pointer (`grab_at`), and what taking it, pulling it and letting
  it go do (`_take`, `_pull`, `_let_go`).
- `DragPoint`: a dot free to go anywhere on the plot (a pole in the s-plane). A drag emits where the pointer is,
  `dragged(x, y)`, and leaves the dot to the page, which works out what that point means, clamps it and puts the
  dot where it ended up (`place`).

The page calls `apply_theme()` from its own (`MainWindow._restyle`), as for any item that keeps pens.

Working on it: drags are signals, not callbacks over `self`, so a page connects a bound method (CLAUDE.md:
a lambda over the page keeps it alive). `tests/test_drag.py` drags both with synthetic mouse events.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui

from .style import colors

HAND = QtCore.Qt.CursorShape.PointingHandCursor


class DragLine(pg.InfiniteLine):
    """A line the reader drags to set a value, with a ◀▶ grip near its top (or right) end."""

    dragged = QtCore.Signal(float)
    released = QtCore.Signal(float)

    def __init__(
        self,
        tip: str,
        angle: float = 90,
        log: bool = False,
        bounds: tuple[float, float] | None = None,
        role: str = "force",
    ) -> None:
        super().__init__(angle=angle, movable=True)
        self.log, self.role = log, role
        self.setToolTip(tip)
        self.setCursor(HAND)
        self.setZValue(20)
        self.addMarker("<|>", position=0.92, size=12)
        if bounds is not None:
            self.setBounds([self._to_view(b) for b in bounds])
        self.sigPositionChanged.connect(self._on_moved)
        self.sigPositionChangeFinished.connect(self._on_finished)
        self.apply_theme()

    def _to_view(self, value: float) -> float:
        return math.log10(value) if self.log else value

    def data_value(self) -> float:
        """The line's value in data units (`value()` is pyqtgraph's, log10 of it on a log axis)."""
        v = float(self.value())
        return 10.0**v if self.log else v

    def set_scale(self, log: bool, bounds: tuple[float, float] | None = None) -> None:
        """Follow the plot's axis to (or from) a log scale, keeping the value; `bounds` in data units."""
        value = self.data_value()
        self.log = log
        if bounds is not None:
            self.setBounds([self._to_view(b) for b in bounds])
        self.set_value(value)

    def set_value(self, value: float) -> None:
        """Move the line without emitting `dragged` or `released`: the page's box or a trial set it."""
        if self.log and not value > 0:
            return
        self.setValue(self._to_view(value))

    def _on_moved(self) -> None:
        if self.moving:  # the reader's drag, not set_value
            self.dragged.emit(self.data_value())

    def _on_finished(self) -> None:
        self.released.emit(self.data_value())

    def apply_theme(self) -> None:
        color = getattr(colors, self.role)
        self.setPen(pg.mkPen(color, width=2))
        self.setHoverPen(pg.mkPen(colors.strong, width=4))
        self.update()


class DragDot(pg.TargetItem):
    """A dot the reader slides along a path: `set_path(x, y, params)`, then `dragged(param)` as it moves."""

    dragged = QtCore.Signal(float)
    released = QtCore.Signal(float)

    def __init__(self, tip: str, size: int = 14, role: str = "force") -> None:
        super().__init__(size=size, symbol="o", movable=True)
        self.role = role
        self.setToolTip(tip)
        self.setCursor(HAND)
        self.setZValue(20)
        self._x = self._y = self._p = np.zeros(0)
        self.param = math.nan
        self.apply_theme()

    def set_path(self, x, y, params) -> None:
        """The points the dot may sit on, in the plot's own coordinates, and each one's parameter."""
        self._x, self._y, self._p = (np.asarray(a, dtype=float) for a in (x, y, params))

    def place(self, x: float, y: float, param: float) -> None:
        """Put the dot here without emitting `dragged` (the page's box or a trial set it)."""
        self.param = param
        self.setPos(x, y)

    def nearest(self, at: QtCore.QPointF) -> int | None:
        """The path's point nearest `at` (a scene position), measured on screen, so both axes count alike."""
        if not self._x.size or (vb := self.getViewBox()) is None:
            return None
        pts = [vb.mapViewToScene(QtCore.QPointF(float(x), float(y))) for x, y in zip(self._x, self._y)]
        d = [(p.x() - at.x()) ** 2 + (p.y() - at.y()) ** 2 for p in pts]
        return int(np.argmin(d))

    def mouseDragEvent(self, ev) -> None:
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        ev.accept()
        if ev.isStart():
            self.moving = True
        if not self.moving:
            return
        i = self.nearest(ev.scenePos())
        if i is not None and self._p[i] != self.param:
            self.param = float(self._p[i])
            self.setPos(float(self._x[i]), float(self._y[i]))
            self.dragged.emit(self.param)
        if ev.isFinish():
            self.moving = False
            self.released.emit(self.param)

    def apply_theme(self) -> None:
        color = getattr(colors, self.role)
        self.setPen(pg.mkPen(colors.strong, width=2))
        self.setBrush(pg.mkBrush(color))
        self.setHoverPen(pg.mkPen(colors.strong, width=3))
        self.setHoverBrush(pg.mkBrush(QtGui.QColor(color).lighter(130)))
        self.update()


class DragPoint(DragDot):
    """A dot the reader drags anywhere: `dragged(x, y)` (view coordinates) as it moves, `released()` at the end.

    The dot doesn't move itself: the page sets what the point means and `place`s it, so a point the page can't
    take (a pole in the right half plane) is clamped where the page puts it.
    """

    dragged = QtCore.Signal(float, float)
    released = QtCore.Signal()

    def place(self, x: float, y: float, param: float = math.nan) -> None:
        super().place(x, y, param)

    def mouseDragEvent(self, ev) -> None:
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        ev.accept()
        if ev.isStart():
            self.moving = True
        if not self.moving or (vb := self.getViewBox()) is None:
            return
        at = vb.mapSceneToView(ev.scenePos())
        self.dragged.emit(float(at.x()), float(at.y()))
        if ev.isFinish():
            self.moving = False
            self.released.emit()


class Grabbable:
    """The grab, pull and release handlers (see the module); a mixin before the view's `pg.PlotWidget`.

    `held` is what's held, or `NOT_HELD` (False for a view with one body, None for the chain, whose held mass's
    index may be 0)."""

    NOT_HELD: object = False
    held: object = False

    def _view_pos(self, event) -> QtCore.QPointF:
        return self.plotItem.vb.mapSceneToView(self.mapToScene(event.position().toPoint()))  # type: ignore[attr-defined]

    def _holding(self) -> bool:
        return self.held is not self.NOT_HELD and self.held is not None and self.held is not False

    def grab_at(self, pos: QtCore.QPointF) -> object | None:
        """What a press at `pos` (view coordinates) takes hold of, or None."""
        raise NotImplementedError

    def _take(self, what: object, pos: QtCore.QPointF) -> None:
        """Taken hold of: the view's signal, then the first pull."""
        self._pull(pos)

    def _pull(self, pos: QtCore.QPointF) -> None:
        raise NotImplementedError

    def _let_go(self, what: object) -> None:
        """Let go of: the view's signal."""

    def mousePressEvent(self, event) -> None:
        pos = self._view_pos(event)
        what = self.grab_at(pos) if event.button() == QtCore.Qt.MouseButton.LeftButton else None
        if what is None:
            super().mousePressEvent(event)  # type: ignore[misc]
            return
        self.held = what
        self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)  # type: ignore[attr-defined]
        self._take(what, pos)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        pos = self._view_pos(event)
        if self._holding():
            self._pull(pos)
            event.accept()
            return
        if self.grab_at(pos) is None:
            self.unsetCursor()  # type: ignore[attr-defined]
        else:
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)  # type: ignore[attr-defined]
        super().mouseMoveEvent(event)  # type: ignore[misc]

    def mouseReleaseEvent(self, event) -> None:
        if not self._holding():
            super().mouseReleaseEvent(event)  # type: ignore[misc]
            return
        what, self.held = self.held, self.NOT_HELD
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)  # type: ignore[attr-defined]
        self._let_go(what)
        event.accept()
