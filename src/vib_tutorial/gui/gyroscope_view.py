"""The Gyroscope page's 3D view of the top: an orthographic projection, turned by dragging, that draws the disc (or
a tippe top's ball) on its stem, the axis's trail and the turning points' circles, the vectors ω, L and τ (or the
friction F on a table), and the body's and the axis frame's axes. The page passes it each frame's state (`draw`).
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.gyroscope import Frame, Shape, Top, frame_axes
from ..core.sliding_top import RoundTop
from .lesson import Dot
from .style import colors

BODY_AXES = 6  # the body's axes' colour, colors.mass[k], apart from the vectors', the trail's and the disc's in every
# theme; the axis frame's are drawn dashed in the axis's own colour, as they go with it
AXES_NONE, AXES_BODY, AXES_HELD, AXES_BOTH = range(4)  # the axes the 3D view draws on the top
AXES_SHOWN = ("None", "Body's (turn with the top)", "Axis frame's (not the spin)", "Both")
VIEW_TIP = (
    "<p>The top on its tip, in 3D: drag to turn the view, double-click to reset it. The line through the disc "
    "is its axis and the tick on the disc a mark that turns with it; the thin line is where the end of the "
    "axis has been, between the dotted circles at the turning points. With <i>Vectors</i> ticked: the "
    "angular momentum L, the angular velocity ω and gravity's moment τ, each scaled to fit. <i>Axes</i> draws the "
    "body's axes e₁, e₂ (solid) and the axis frame's e₁′, e₂′ (dashed) at the disc's centre, about the axis e₃ "
    "they share.</p>"
)


class TopView(pg.PlotWidget):
    """The top in 3D, orthographically projected; drag to turn it."""

    YAW, PITCH = math.radians(25.0), math.radians(20.0)
    RIM = 48  # points round the disc

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("The top in 3D")
        self.setToolTip(VIEW_TIP)
        self.setAspectLocked(True)
        for side in ("left", "bottom"):
            self.hideAxis(side)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.yaw, self.pitch = self.YAW, self.PITCH
        self._drag: QtCore.QPointF | None = None
        self.top = Top()
        self.round: RoundTop | None = None  # on a table
        self.ball = False
        self.reach = 1.0  # the drawn axis's length (m)
        self.turning: tuple[float, ...] = (0.0, 0.0)
        self.show_vectors = True
        self.show_axes = AXES_NONE
        self._last: tuple | None = None

        plot = self.getPlotItem()
        self.table = QtWidgets.QGraphicsPathItem()
        self.vertical = pg.PlotCurveItem()
        self.circles = pg.PlotCurveItem(connect="finite")  # the turning points' circles
        self.trail = pg.PlotCurveItem()
        self.track = pg.PlotCurveItem()  # the tip's, on the table
        self.stem = pg.PlotCurveItem()
        self.tip = pg.PlotCurveItem()  # a round tip's outline
        self.disc = QtWidgets.QGraphicsPathItem()
        self.rim = pg.PlotCurveItem()
        self.spoke = pg.PlotCurveItem()
        self.axis_line = pg.PlotCurveItem()
        self.vectors = [pg.PlotCurveItem() for _ in range(3)]  # L, ω, τ
        self.labels = [pg.TextItem(t, anchor=(0.5, 0.5)) for t in ("L", "ω", "τ")]
        self.end = Dot(7)
        self.frame_lines = [pg.PlotCurveItem() for _ in range(4)]  # e₁, e₂ (the body's), e₁′, e₂′ (the axis frame's)
        # an apostrophe for the prime: the view's font has no ′
        self.frame_labels = [pg.TextItem(t, anchor=(0.5, 0.5)) for t in ("e₁", "e₂", "e₁'", "e₂'", "e₃")]
        bigger = QtGui.QFont()
        bigger.setPointSizeF(bigger.pointSizeF() * 1.25)
        for label in self.frame_labels:
            label.setFont(bigger)
        for item in (
            self.table,
            self.vertical,
            self.circles,
            self.trail,
            self.track,
            self.stem,
            self.tip,
            self.disc,
            self.rim,
            self.spoke,
            self.axis_line,
            *self.frame_lines,
            *self.vectors,
            *self.labels,
            *self.frame_labels,
        ):
            plot.addItem(item)
        plot.addItem(self.end, ignoreBounds=True)
        self.apply_theme()

    def project(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Screen (x, y) of points p (..., 3), z up: turned by the yaw about z, looked at from the pitch above."""
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        across = x * cy - y * sy
        away = x * sy + y * cy
        return across, z * math.cos(self.pitch) + away * math.sin(self.pitch)

    def set_top(self, top: Top, turning: tuple[float, ...], round_top: RoundTop | None = None) -> None:
        """A new top or launch: its size sets the view's, and its turning points the circles (none on a table,
        where the view follows the top: its centre of mass stays over the middle)."""
        self.top, self.turning, self.round = top, turning, round_top
        self.ball = round_top is not None and round_top.rim == 0
        self.reach = 2.4 * top.radius if self.ball else 1.6 * max(top.height, top.corner)
        self.labels[2].setText("F" if round_top is not None else "τ")
        self._last = None
        self._place_static()

    def _place_static(self) -> None:
        R = self.reach
        ring = np.linspace(0, 2 * math.pi, 97)
        table = np.stack([1.1 * R * np.cos(ring), 1.1 * R * np.sin(ring), np.zeros_like(ring)], axis=1)
        sx, sy = self.project(table)
        path = QtGui.QPainterPath()
        path.moveTo(float(sx[0]), float(sy[0]))
        for a, b in zip(sx[1:], sy[1:]):
            path.lineTo(float(a), float(b))
        self.table.setPath(path)
        vx, vy = self.project(np.array([[0, 0, 0], [0, 0, 1.15 * R]]))
        self.vertical.setData(vx, vy)
        lines = []
        for th in self.turning:
            circle = np.stack(
                [
                    R * math.sin(th) * np.cos(ring),
                    R * math.sin(th) * np.sin(ring),
                    R * math.cos(th) * np.ones_like(ring),
                ],
                axis=1,
            )
            lines.append(circle)
            lines.append(np.full((1, 3), np.nan))
        if lines:
            self.circles.setData(*self.project(np.concatenate(lines)))
        else:
            self.circles.setData([], [])
        span = 1.25 * R
        self.setRange(xRange=(-span, span), yRange=(-0.6 * span, 1.2 * span), padding=0)
        if self._last is not None:
            self.draw(*self._last)

    def _anchor(self, axis: np.ndarray, centre: np.ndarray) -> np.ndarray:
        """Where the drawn axis starts, for axes (..., 3) and centres of mass G: the stem's end (the tip), or a
        ball's centre."""
        rt = self.round
        if rt is None:
            return np.zeros_like(axis)
        hub = centre - rt.offset * axis  # the round tip's centre
        return hub if self.ball else hub - rt.radius * axis

    def _outline(self, item: QtWidgets.QGraphicsPathItem, x: np.ndarray, y: np.ndarray) -> None:
        path = QtGui.QPainterPath()
        path.moveTo(float(x[0]), float(y[0]))
        for a, b in zip(x[1:], y[1:]):
            path.lineTo(float(a), float(b))
        item.setPath(path)

    def draw(
        self,
        axis: np.ndarray,
        mark: np.ndarray,
        momentum: np.ndarray,
        trail: np.ndarray,
        centre: np.ndarray | None = None,
        trail_centres: np.ndarray | None = None,
        track: np.ndarray | None = None,
        force: np.ndarray | None = None,
    ) -> None:
        """The top with its axis along `axis` (a unit vector), the disc's mark toward `mark`, and L; `trail` is
        where the axis has pointed (n, 3). On a table: G at `centre` (drawn over the middle), G when the axis
        pointed along `trail` (n, 3; its height), where the tip has touched the table (`track`, (m, 2)) and the
        friction."""
        self._last = (axis, mark, momentum, trail, centre, trail_centres, track, force)
        top, R = self.top, self.reach
        here = np.zeros(3) if centre is None else np.array([centre[0], centre[1], 0.0])
        G = top.height * axis if centre is None else centre - here
        anchor = self._anchor(axis, G)
        ring = np.linspace(0, 2 * math.pi, self.RIM + 1)[:, None]
        side = np.cross(axis, mark)
        if self.ball:
            rt = self.round
            assert rt is not None  # a ball is only ever on a table
            # its outline is a circle on the screen whatever the view; its equator turns with it
            cx, cy = self.project(anchor)
            self._outline(self.disc, cx + rt.radius * np.cos(ring[:, 0]), cy + rt.radius * np.sin(ring[:, 0]))
            rim = anchor + rt.radius * (np.cos(ring) * mark + np.sin(ring) * side)
            spoke = np.array([anchor, anchor + rt.radius * mark])
            self.stem.setData([], [])
            self.tip.setData([], [])
            start = anchor
        else:
            if top.shape is Shape.DISC:
                rim = G + top.radius * (np.cos(ring) * mark + np.sin(ring) * side)
            else:  # its corners, half a side along the mark and half the other across it
                across = top.breadth if top.shape is Shape.RECTANGLE else top.radius
                k = 0.5 * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]]) * [top.radius, across]
                rim = G + k[:, :1] * mark + k[:, 1:] * side
            self._outline(self.disc, *self.project(rim))
            spoke = np.array([G, G + (1.0 if top.shape is Shape.DISC else 0.5) * top.radius * mark])
            if self.round is None:
                self.tip.setData([], [])
            else:
                hub = anchor + self.round.radius * axis
                hx, hy = self.project(hub)
                self.tip.setData(
                    hx + self.round.radius * np.cos(ring[:, 0]), hy + self.round.radius * np.sin(ring[:, 0])
                )
            tx, ty = self.project(np.array([anchor, G]))
            self.stem.setData(tx, ty)
            start = G
        rx, ry = self.project(rim)
        self.rim.setData(rx, ry)
        sx, sy = self.project(spoke)
        self.spoke.setData(sx, sy)
        end = anchor + R * axis
        ax, ay = self.project(np.array([start, end]))
        self.axis_line.setData(ax, ay)
        ex, ey = self.project(end)
        self.end.setData([ex], [ey])
        if trail.size:
            ends = R * trail
            if trail_centres is not None:  # each over the middle, as the top is drawn: where it pointed, not went
                ends = ends + self._anchor(trail, trail_centres * [0.0, 0.0, 1.0])
            px, py = self.project(ends)
            self.trail.setData(px, py)
        else:
            self.trail.setData([], [])
        if track is not None and track.size:
            flat = np.column_stack([track[:, 0] - here[0], track[:, 1] - here[1], np.zeros(len(track))])
            self.track.setData(*self.project(flat))
        else:
            self.track.setData([], [])
        self._draw_vectors(axis, mark, momentum, np.zeros(3) if centre is None else G, force)
        self._draw_axes(axis, mark, G, anchor + R * axis)

    def _draw_axes(self, axis: np.ndarray, mark: np.ndarray, G: np.ndarray, end: np.ndarray) -> None:
        """The body's e₁, e₂ and the axis frame's e₁′, e₂′ from the disc's centre G, as `show_axes` picks, and e₃
        named at the axis's end."""
        body = self.show_axes in (AXES_BODY, AXES_BOTH)
        held = self.show_axes in (AXES_HELD, AXES_BOTH)
        length = 0.75 * self.reach
        b = frame_axes(axis[None], mark[None], Frame.BODY)[0]
        a = frame_axes(axis[None], mark[None], Frame.AXIS)[0]
        for item, label, v, on in zip(
            self.frame_lines, self.frame_labels, (b[0], b[1], a[0], a[1]), (body, body, held, held)
        ):
            if not on:
                item.setData([], [])
                label.hide()
                continue
            x, y = self.project(np.array([G, G + length * v]))
            item.setData(x, y)
            lx, ly = self.project(G + 1.12 * length * v)
            label.setPos(float(lx), float(ly))
            label.show()
        e3 = self.frame_labels[4]
        if body or held:
            ex, ey = self.project(end + 0.1 * self.reach * axis)
            e3.setPos(float(ex), float(ey))
            e3.show()
        else:
            e3.hide()

    def _draw_vectors(
        self, axis: np.ndarray, mark: np.ndarray, momentum: np.ndarray, origin: np.ndarray, force: np.ndarray | None
    ) -> None:
        top, R = self.top, self.reach
        size = float(np.linalg.norm(momentum))
        if not self.show_vectors or size == 0:
            for item, label in zip(self.vectors, self.labels):
                item.setData([], [])
                label.hide()
            return
        along = float(momentum @ axis)
        rt = self.round
        i1, i3 = (top.transverse, top.polar) if rt is None else (rt.central, rt.polar)
        omega = momentum / i1 + (1 / i3 - 1 / i1) * along * axis
        if rt is None and not top.symmetric:  # a rectangle's own y axis has I₂
            side = np.cross(axis, mark)
            omega = omega + (1 / top.second - 1 / i1) * float(momentum @ side) * side
        if rt is None:
            third = -top.weight_moment * np.cross(axis, [0.0, 0.0, 1.0])
        else:
            third = np.zeros(3) if force is None else np.array([force[0], force[1], 0.0])
        for item, label, v, length in zip(
            self.vectors, self.labels, (momentum, omega, third), (1.25 * R, 1.05 * R, 0.6 * R)
        ):
            n = float(np.linalg.norm(v))
            if n < 1e-12:
                item.setData([], [])
                label.hide()
                continue
            tip = origin + length * v / n
            x, y = self.project(np.array([origin, tip]))
            item.setData(x, y)
            lx, ly = self.project(origin + 1.1 * (tip - origin))
            label.setPos(float(lx), float(ly))
            label.show()

    def apply_theme(self) -> None:
        c = colors
        grid = QtGui.QColor(c.grey)
        grid.setAlpha(40)
        self.table.setBrush(QtGui.QBrush(grid))
        self.table.setPen(pg.mkPen(c.grey, width=1))
        self.vertical.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.circles.setPen(pg.mkPen(c.fit, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.trail.setPen(pg.mkPen(c.mass[1], width=1.5))
        self.track.setPen(pg.mkPen(c.mass[2], width=1.5))
        self.stem.setPen(pg.mkPen(c.strong, width=3))
        self.tip.setPen(pg.mkPen(c.strong, width=2))
        fill = QtGui.QColor(c.mass[0])
        fill.setAlpha(110)
        self.disc.setBrush(QtGui.QBrush(fill))
        self.disc.setPen(pg.mkPen(None))
        self.rim.setPen(pg.mkPen(c.mass[0], width=2))
        self.spoke.setPen(pg.mkPen(c.strong, width=3))
        self.axis_line.setPen(pg.mkPen(c.strong, width=1.5))
        self.end.setBrush(pg.mkBrush(c.mass[1]))
        self.end.setPen(pg.mkPen(None))
        for item, label, color in zip(self.vectors, self.labels, (c.force, c.mass[2], c.mass[3])):
            item.setPen(pg.mkPen(color, width=2.5))
            label.setColor(color)
        dash = QtCore.Qt.PenStyle.DashLine
        for k, (item, label) in enumerate(zip(self.frame_lines, self.frame_labels)):
            color = c.mass[BODY_AXES] if k < 2 else c.strong
            item.setPen(pg.mkPen(color, width=2, style=QtCore.Qt.PenStyle.SolidLine if k < 2 else dash))
            label.setColor(color)
        self.frame_labels[4].setColor(c.strong)

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag = event.position()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag is None:
            super().mouseMoveEvent(event)
            return
        d = event.position() - self._drag
        self._drag = event.position()
        self.yaw = (self.yaw + 0.01 * d.x()) % (2 * math.pi)  # the side nearest you follows the cursor
        self.pitch = float(np.clip(self.pitch + 0.01 * d.y(), -math.pi / 2, math.pi / 2))
        self._place_static()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.yaw, self.pitch = self.YAW, self.PITCH
        self._place_static()
        event.accept()
