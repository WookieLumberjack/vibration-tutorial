"""The Frame builder page's canvas, `FrameCanvas`: the frame on its grid, built with the tool picked (place and
drag nodes, join them with members, add springs, dampers and supports), swinging in the mode picked or playing the
load's response, with the load's arrow and the plotted DOF's ring; and the helpers that draw its parts.
"""

from __future__ import annotations

import dataclasses
import math
from enum import Enum

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.frame import MAX_NODES, TWO_PI, Element, Frame, Kind, Support, member_curve
from ..core.frame_response import Load
from .style import colors

NODE_PX, ELEMENT_PX = 10, 7  # how near a click has to be to pick a node or an element


class Tool(Enum):
    SELECT = "Select"
    NODE = "Node"
    BEAM = "Beam"
    BAR = "Bar"
    SPRING = "Spring"
    DAMPER = "Damper"
    TORSION = "Torsion spring"
    MASS = "Mass"
    SUPPORT = "Support"


TOOLS = tuple(Tool)
TOOL_HINTS = {
    Tool.SELECT: "Click a node or an element to edit it on the left; drag a node to move it. Delete removes it.",
    Tool.NODE: "Click the grid to place a node.",
    Tool.BEAM: "Click two nodes (or grid points, which become nodes) to join them with a beam; keep clicking to "
    "go on from the last. Esc or a right click stops.",
    Tool.BAR: "Click two nodes (or grid points) to join them with a bar (pinned ends: it only stretches). Esc or "
    "a right click stops.",
    Tool.SPRING: "Click a node, then another node, or a grid point to anchor it to the ground there.",
    Tool.DAMPER: "Click a node, then another node, or a grid point to anchor it to the ground there.",
    Tool.TORSION: "Click a node to add a torsion spring from its turn to the ground.",
    Tool.MASS: "Click a node to add a point mass there.",
    Tool.SUPPORT: "Pick a support, then click a node to give it that support (click again to free it).",
}

CANVAS_TIP = (
    "<p>The frame on its grid. Pick a tool above. Dotted: the frame at rest, while it swings in the mode picked. "
    "Numbers are the nodes'.</p>"
)


# ------------------------------------------------------------------ drawing helpers
NAN2 = np.full(2, np.nan)


def _zigzag(a: np.ndarray, b: np.ndarray, width: float, turns: int = 5) -> list:
    d = b - a
    length = float(np.hypot(*d))
    if length == 0:
        return [a, b]
    n = np.array([-d[1], d[0]]) / length
    s = np.linspace(0.2, 0.8, 2 * turns + 1)
    wiggle = np.array([0.0] + [(-1) ** i for i in range(1, 2 * turns)] + [0.0])
    return [a, *(a + np.outer(s, d) + np.outer(wiggle * width, n)), b]


def _dashpot(a: np.ndarray, b: np.ndarray, width: float) -> list:
    d = b - a
    length = float(np.hypot(*d))
    if length == 0:
        return [a, b]
    n = np.array([-d[1], d[0]]) / length
    bottom, mouth, plate = a + 0.35 * d, a + 0.65 * d, a + 0.5 * d
    return [
        a,
        bottom,
        NAN2,
        mouth + width * n,
        bottom + width * n,
        bottom - width * n,
        mouth - width * n,
        NAN2,
        plate + 0.75 * width * n,
        plate - 0.75 * width * n,
        NAN2,
        plate,
        b,
    ]


def _ground(at: np.ndarray, toward: np.ndarray, size: float) -> list:
    """A hatched ground line through `at`, across `toward` (a unit vector pointing into the ground)."""
    n = np.array([-toward[1], toward[0]])
    out = [at - 0.8 * size * n, at + 0.8 * size * n, NAN2]
    for s in np.linspace(-0.8, 0.8, 5):
        p = at + s * size * n
        out += [p, p + 0.35 * size * (toward - n), NAN2]
    return out


def _circle(c: np.ndarray, r: float) -> list:
    t = np.linspace(0.0, TWO_PI, 13)
    return [*(c + r * np.column_stack([np.cos(t), np.sin(t)])), NAN2]


def _away(frame: Frame, i: int) -> np.ndarray:
    """The axis direction pointing away from the elements on node i (down if there are none)."""
    p = frame.point(i)
    total = np.zeros(2)
    for e in frame.elements:
        if (e.kind.member or e.kind.link) and i in (e.i, e.j):
            other = frame.point(e.j if e.i == i else e.i) if e.j is not None else np.array(e.anchor)
            d = other - p
            norm = float(np.hypot(*d))
            if norm > 0:
                total += d / norm
    if float(np.hypot(*total)) < 1e-9:
        return np.array([0.0, -1.0])
    axis = int(np.argmax(np.abs(total)))
    out = np.zeros(2)
    out[axis] = -math.copysign(1.0, total[axis])
    return out


def _support(frame: Frame, i: int, at: np.ndarray, h: float) -> list:
    """The node's support symbol at `at`, size h, pointing into the ground."""
    s = frame.nodes[i].support
    if s is Support.FREE:
        return []
    if s is Support.NO_TURN:
        r = 0.45 * h
        return [at + [-r, -r], at + [r, -r], at + [r, r], at + [-r, r], at + [-r, -r], NAN2]
    if s in (Support.PINNED, Support.ROLLER_X, Support.GUIDED_X):
        down = np.array([0.0, -1.0])
    elif s in (Support.ROLLER_Y, Support.GUIDED_Y):
        side = _away(frame, i)
        down = side if side[0] != 0 else np.array([-1.0, 0.0])
    else:
        down = _away(frame, i)
    right = np.array([-down[1], down[0]])

    def p(x: float, y: float) -> np.ndarray:  # local: y negative into the ground
        return at + x * h * right - y * h * down

    if s is Support.CLAMPED:
        return [p(-0.9, 0), p(0.9, 0), NAN2] + _ground(p(0, 0), down, h)[3:]
    if s in (Support.GUIDED_X, Support.GUIDED_Y):
        return (
            [p(-0.7, 0), p(0.7, 0), NAN2]
            + _circle(p(-0.4, -0.22), 0.2 * h)
            + _circle(p(0.4, -0.22), 0.2 * h)
            + _ground(p(0, -0.44), down, h)
        )
    out = [p(0, 0), p(-0.55, -0.9), p(0.55, -0.9), p(0, 0), NAN2]
    if s is Support.PINNED:
        return out + _ground(p(0, -0.9), down, h)
    return out + _circle(p(-0.3, -1.08), 0.17 * h) + _circle(p(0.3, -1.08), 0.17 * h) + _ground(p(0, -1.26), down, h)


def _spiral(at: np.ndarray, h: float) -> list:
    t = np.linspace(0.0, 2.5 * TWO_PI, 60)
    r = h * (0.25 + 0.55 * t / t[-1])
    pts = at + np.column_stack([r * np.cos(t), r * np.sin(t)])
    end = pts[-1]
    return [*pts, end + [0.0, -0.4 * h], NAN2] + _ground(end + [0.0, -0.4 * h], np.array([0.0, -1.0]), 0.6 * h)


def _arrow(at: np.ndarray, dof: int, sign: float, h: float) -> list:
    """A load's arrow ending at `at`: along x or y, or round it for a moment (anticlockwise if sign > 0)."""
    if dof == 2:
        a = np.linspace(-0.6, 0.9, 16) * math.pi * sign
        r = 1.2 * h
        arc = [at + r * np.array([math.cos(t), math.sin(t)]) for t in a]
        end, tangent = arc[-1], np.array([-math.sin(a[-1]), math.cos(a[-1])]) * sign
        tip = [end - 0.5 * h * (tangent + 0.6 * np.array([-tangent[1], tangent[0]]) * k) for k in (1, -1)]
        return [*arc, NAN2, tip[0], end, tip[1], NAN2]
    d = np.zeros(2)
    d[dof] = 1.0 if sign >= 0 else -1.0
    side = np.array([-d[1], d[0]])
    tail = at - 2.5 * h * d
    return [tail, at, NAN2, at - 0.6 * h * d + 0.35 * h * side, at, at - 0.6 * h * d - 0.35 * h * side, NAN2]


def _segments(points: list) -> tuple[np.ndarray, np.ndarray]:
    if not points:
        return np.array([]), np.array([])
    a = np.array([np.asarray(p, dtype=float) for p in points])
    return a[:, 0], a[:, 1]


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    t = float(np.clip((p - a) @ d / max(float(d @ d), 1e-30), 0.0, 1.0))
    return float(np.hypot(*(p - a - t * d)))


# ------------------------------------------------------------------ the canvas
class FrameCanvas(pg.PlotWidget):
    """The frame on its grid; builds it with the tool picked. Emits `edited(frame, picked)` for every change and
    `picked(selection)` when the selection changes: ("node", i), ("element", k) or None."""

    edited = QtCore.Signal(object, object)
    selected = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(CANVAS_TIP)
        self.setAspectLocked(True)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.showGrid(x=True, y=True, alpha=0.25)
        for side in ("left", "bottom"):
            self.getAxis(side).enableAutoSIPrefix(False)
        self.setLabel("bottom", "x", units="m")
        self.setLabel("left", "y", units="m")
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.viewport().setMouseTracking(True)
        self.setMouseTracking(True)
        self.frame = Frame()
        self.tool = Tool.SELECT
        self.support = Support.PINNED
        self.grid = 0.25
        self.selection: tuple[str, int] | None = None
        self.pending: int | None = None  # the first node of a two-click tool
        self.cursor: np.ndarray | None = None
        self.q: np.ndarray | None = None  # the displacement drawn
        self.load: Load | None = None  # drawn as an arrow
        self.probe: int | None = None  # the DOF whose response is plotted: a square round its node
        self.unit = 0.1  # m: symbols' size, from the view's
        self._drag: int | None = None
        self._dragged: Frame | None = None
        self.template = {  # what a new element of each kind is made as: the last one edited
            Kind.BEAM: Element(Kind.BEAM, 0, 1),
            Kind.BAR: Element(Kind.BAR, 0, 1),
            Kind.SPRING: Element(Kind.SPRING, 0, 1, value=100e3),
            Kind.DAMPER: Element(Kind.DAMPER, 0, 1, value=200.0),
            Kind.TORSION_SPRING: Element(Kind.TORSION_SPRING, 0, value=10e3),
            Kind.MASS: Element(Kind.MASS, 0, value=10.0),
        }
        plot = self.getPlotItem()
        self.ghost = pg.PlotCurveItem(connect="finite")
        self.bars = pg.PlotCurveItem(connect="finite")
        self.beams = pg.PlotCurveItem(connect="finite")
        self.links = pg.PlotCurveItem(connect="finite")
        self.supports = pg.PlotCurveItem(connect="finite")
        self.highlight = pg.PlotCurveItem(connect="finite")
        self.rubber = pg.PlotCurveItem()
        self.masses = pg.PlotDataItem(pen=None, symbol="s", symbolSize=15)
        self.nodes = pg.PlotDataItem(pen=None, symbol="o", symbolSize=8)
        self.ring = pg.PlotDataItem(pen=None, symbol="o", symbolSize=16)
        self.arrow = pg.PlotCurveItem(connect="finite")
        self.probe_mark = pg.PlotDataItem(pen=None, symbol="s", symbolSize=18)
        for item in (
            self.ghost,
            self.supports,
            self.links,
            self.bars,
            self.beams,
            self.highlight,
            self.rubber,
            self.masses,
            self.nodes,
            self.ring,
            self.arrow,
            self.probe_mark,
        ):
            plot.addItem(item)
        self.labels: list[pg.TextItem] = []
        self.apply_theme()

    # --------------------------------------------------------- state
    def set_frame(self, frame: Frame, fit: bool = True) -> None:
        self.frame = frame
        if self.pending is not None and self.pending >= len(frame.nodes):
            self.pending = None
        if fit:
            self.fit()
        self.draw()

    def set_tool(self, tool: Tool) -> None:
        self.tool = tool
        self.pending = None
        self.draw()

    def fit(self) -> None:
        x0, x1, y0, y1 = self.frame.extent()
        if not self.frame.nodes:
            x0, x1, y0, y1 = 0.0, 3.0, 0.0, 2.0
        size = max(x1 - x0, y1 - y0, 4 * self.grid)
        pad = max(0.25 * size, 2 * self.grid)
        self.setRange(xRange=(x0 - pad, x1 + pad), yRange=(y0 - pad, y1 + pad), padding=0)
        self.unit = 0.035 * (size + 2 * pad)

    def snap(self, p: np.ndarray) -> np.ndarray:
        g = self.grid
        return np.round(p / g) * g

    def _pixel(self) -> float:
        sx, sy = self.getPlotItem().getViewBox().viewPixelSize()
        return max(sx, sy)

    def node_near(self, p: np.ndarray) -> int | None:
        best, reach = None, NODE_PX * self._pixel()
        for i in range(len(self.frame.nodes)):
            d = float(np.hypot(*(self.frame.point(i) - p)))
            if d <= reach:
                best, reach = i, d
        return best

    def element_near(self, p: np.ndarray) -> int | None:
        best, reach = None, ELEMENT_PX * self._pixel()
        f = self.frame
        for k, e in enumerate(f.elements):
            if not (e.kind.member or e.kind.link):
                continue
            b = f.point(e.j) if e.j is not None else np.array(e.anchor)
            d = _segment_distance(p, f.point(e.i), b)
            if d <= reach:
                best, reach = k, d
        return best

    def select(self, selection: tuple[str, int] | None) -> None:
        self.selection = selection
        self.draw()
        self.selected.emit(selection)

    # --------------------------------------------------------- editing
    def _commit(self, frame: Frame, selection: tuple[str, int] | None) -> None:
        self.selection = selection
        self.edited.emit(frame, selection)

    def _node(self, frame: Frame, p: np.ndarray) -> tuple[Frame, int | None]:
        """The node at p (snapped), made if there isn't one; None if the frame is full."""
        found = self.node_near(p)
        if found is None:
            found = frame.node_at(*p)
        if found is not None:
            return frame, found
        if len(frame.nodes) >= MAX_NODES:
            return frame, None
        return frame.with_node(float(p[0]), float(p[1])), len(frame.nodes)

    def press(self, p: np.ndarray, right: bool = False) -> None:
        """A click at p (view coordinates): what the tool does."""
        if right:
            self.pending = None
            self.draw()
            return
        f = self.frame
        hit = self.node_near(p)
        s = self.snap(p)
        tool = self.tool
        if tool is Tool.SELECT:
            if hit is not None:
                self._drag, self._dragged = hit, None
                self.select(("node", hit))
            else:
                k = self.element_near(p)
                self.select(("element", k) if k is not None else None)
        elif tool is Tool.NODE:
            if hit is not None:
                self.select(("node", hit))
            elif len(f.nodes) < MAX_NODES:
                self._commit(f.with_node(float(s[0]), float(s[1])), ("node", len(f.nodes)))
        elif tool in (Tool.BEAM, Tool.BAR, Tool.SPRING, Tool.DAMPER):
            kind = {Tool.BEAM: Kind.BEAM, Tool.BAR: Kind.BAR, Tool.SPRING: Kind.SPRING, Tool.DAMPER: Kind.DAMPER}[tool]
            if self.pending is None:
                f, i = self._node(f, s if hit is None else f.point(hit))
                if i is None:
                    return
                self.pending = i
                if f is not self.frame:
                    self._commit(f, ("node", i))
                else:
                    self.draw()
                return
            a = self.pending
            if kind.link and hit is None:  # to the ground at s
                if float(np.hypot(*(s - f.point(a)))) > 0:
                    e = dataclasses.replace(self.template[kind], i=a, j=None, anchor=(float(s[0]), float(s[1])))
                    self.pending = None
                    self._commit(f.with_element(e), ("element", len(f.elements)))
                return
            f, b = self._node(f, s if hit is None else f.point(hit))
            if b is None or b == a:
                return
            if any({e.i, e.j} == {a, b} and e.kind is kind for e in f.elements):
                self.pending = b if kind.member else None
                self.draw()
                return
            e = dataclasses.replace(self.template[kind], i=a, j=b, anchor=None)
            self.pending = b if kind.member else None
            self._commit(f.with_element(e), ("element", len(f.elements)))
        elif hit is not None and tool in (Tool.TORSION, Tool.MASS):
            kind = Kind.MASS if tool is Tool.MASS else Kind.TORSION_SPRING
            if not any(e.kind is kind and e.i == hit and e.j is None for e in f.elements):
                f = f.with_element(dataclasses.replace(self.template[kind], i=hit, j=None))
                self._commit(f, ("node", hit))
            else:
                self.select(("node", hit))
        elif hit is not None and tool is Tool.SUPPORT:
            node = f.nodes[hit]
            support = Support.FREE if node.support is self.support else self.support
            self._commit(f.replace_node(hit, dataclasses.replace(node, support=support)), ("node", hit))

    def move(self, p: np.ndarray) -> None:
        self.cursor = p
        if self._drag is not None:
            s = self.snap(p)
            other = self.frame.node_at(*s)
            if other is None or other == self._drag:
                node = self.frame.nodes[self._drag]
                self._dragged = self.frame.replace_node(
                    self._drag, dataclasses.replace(node, x=float(s[0]), y=float(s[1]))
                )
                self.draw(self._dragged)
        elif self.pending is not None:
            self.draw()

    def release(self) -> None:
        if self._drag is not None and self._dragged is not None:
            self._commit(self._dragged, ("node", self._drag))
        self._drag, self._dragged = None, None

    def delete_selected(self) -> None:
        if self.selection is None:
            return
        what, index = self.selection
        f = self.frame.without_node(index) if what == "node" else self.frame.without_element(index)
        self.pending = None
        self._commit(f, None)

    def _view_point(self, event) -> np.ndarray:
        scene = self.mapToScene(event.position().toPoint())
        p = self.getPlotItem().getViewBox().mapSceneToView(scene)
        return np.array([p.x(), p.y()])

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        self.press(self._view_point(event), right=event.button() == QtCore.Qt.MouseButton.RightButton)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        self.move(self._view_point(event))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self.release()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        event.accept()

    def wheelEvent(self, event) -> None:
        event.ignore()  # the page scrolls; the view fits the frame

    def keyPressEvent(self, event) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.pending = None
            self.draw()
        elif event.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            self.delete_selected()
        else:
            super().keyPressEvent(event)

    # --------------------------------------------------------- drawing
    def draw(self, frame: Frame | None = None) -> None:
        f = frame if frame is not None else self.frame
        q = self.q if frame is None and self.q is not None and self.q.size == f.n_dof else None
        h = self.unit

        def at(i: int) -> np.ndarray:
            p = f.point(i)
            return p + q[3 * i : 3 * i + 2] if q is not None else p

        bars: list[np.ndarray] = []
        beams: list[np.ndarray] = []
        links: list[np.ndarray] = []
        ghost: list[np.ndarray] = []
        supports: list[np.ndarray] = []
        for k, e in enumerate(f.elements):
            if e.kind.member and e.j is not None:
                if q is not None:
                    x, y = member_curve(f, k, q)
                    line = [*np.column_stack([x, y]), NAN2]
                    ghost += [f.point(e.i), f.point(e.j), NAN2]
                else:
                    line = [f.point(e.i), f.point(e.j), NAN2]
                (beams if e.kind is Kind.BEAM else bars).extend(line)
            elif e.kind.link:
                a = at(e.i)
                b = at(e.j) if e.j is not None else np.array(e.anchor)
                shape = _zigzag(a, b, 0.3 * h) if e.kind is Kind.SPRING else _dashpot(a, b, 0.3 * h)
                links += [*shape, NAN2]
                if e.j is None:
                    d = b - a
                    links += _ground(b, d / float(np.hypot(*d)), 0.7 * h)
            elif e.kind is Kind.TORSION_SPRING and e.j is None:
                links += _spiral(at(e.i), 0.8 * h)
            elif e.kind is Kind.TORSION_DAMPER and e.j is None:
                links += _circle(at(e.i), 0.6 * h) + _ground(at(e.i) + [0.0, -0.6 * h], np.array([0.0, -1.0]), 0.6 * h)
        for i in range(len(f.nodes)):
            supports += _support(f, i, at(i), h)
        self.bars.setData(*_segments(bars))
        self.beams.setData(*_segments(beams))
        self.links.setData(*_segments(links))
        self.ghost.setData(*_segments(ghost))
        self.supports.setData(*_segments(supports))
        points = np.array([at(i) for i in range(len(f.nodes))]).reshape(-1, 2)
        self.nodes.setData(points[:, 0], points[:, 1])
        heavy = sorted({e.i for e in f.elements if e.kind is Kind.MASS and e.value > 0})
        self.masses.setData(points[heavy, 0] if heavy else [], points[heavy, 1] if heavy else [])
        self._label(points)
        # the selection, and the line from the pending node to the cursor
        lit: list = []
        ring: list[int] = []
        if self.selection is not None:
            what, index = self.selection
            if what == "node" and index < len(f.nodes):
                ring = [index]
            elif what == "element" and index < len(f.elements):
                e = f.elements[index]
                if e.kind.member and e.j is not None:
                    x, y = (
                        member_curve(f, index, q)
                        if q is not None
                        else (np.array([at(e.i)[0], at(e.j)[0]]), np.array([at(e.i)[1], at(e.j)[1]]))
                    )
                    lit = [*np.column_stack([x, y]), NAN2]
                elif e.kind.link:
                    b = at(e.j) if e.j is not None else np.array(e.anchor)
                    lit = [at(e.i), b, NAN2]
        self.highlight.setData(*_segments(lit))
        self.ring.setData([at(i)[0] for i in ring], [at(i)[1] for i in ring])
        load = self.load
        arrow = _arrow(at(load.node), load.dof, load.size, h) if load is not None and load.node < len(f.nodes) else []
        self.arrow.setData(*_segments(arrow))
        probe = self.probe if self.probe is not None and self.probe // 3 < len(f.nodes) else None
        self.probe_mark.setData(*(([at(probe // 3)[0]], [at(probe // 3)[1]]) if probe is not None else ([], [])))
        if self.pending is not None and self.cursor is not None and self.pending < len(f.nodes):
            hit = self.node_near(self.cursor)
            end = f.point(hit) if hit is not None else self.snap(self.cursor)
            a = f.point(self.pending)
            self.rubber.setData([a[0], end[0]], [a[1], end[1]])
        else:
            self.rubber.setData([], [])

    def _label(self, points: np.ndarray) -> None:
        while len(self.labels) < len(points):
            label = pg.TextItem(anchor=(0, 1))
            label.setColor(colors.muted)
            self.getPlotItem().addItem(label, ignoreBounds=True)
            self.labels.append(label)
        offset = 0.25 * self.unit
        for i, label in enumerate(self.labels):
            if i < len(points):
                label.setText(str(i + 1))
                label.setPos(points[i, 0] + offset, points[i, 1] + offset)
                label.show()
            else:
                label.hide()

    def apply_theme(self) -> None:
        c = colors
        self.ghost.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.bars.setPen(pg.mkPen(c.mass[1], width=2))
        self.beams.setPen(pg.mkPen(c.mass[0], width=4))
        self.links.setPen(pg.mkPen(c.structure, width=1.5))
        self.supports.setPen(pg.mkPen(c.structure, width=1.5))
        self.highlight.setPen(pg.mkPen(c.force, width=3))
        self.rubber.setPen(pg.mkPen(c.grey, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.masses.setSymbolBrush(pg.mkBrush(c.structure))
        self.masses.setSymbolPen(None)
        self.nodes.setSymbolBrush(pg.mkBrush(c.background))
        self.nodes.setSymbolPen(pg.mkPen(c.structure, width=1.5))
        self.ring.setSymbolBrush(None)
        self.ring.setSymbolPen(pg.mkPen(c.force, width=2.5))
        self.arrow.setPen(pg.mkPen(c.force, width=2.5))
        self.probe_mark.setSymbolBrush(None)
        self.probe_mark.setSymbolPen(pg.mkPen(c.strong, width=1.5))
        for label in self.labels:
            label.setColor(c.muted)
        self.draw()
