"""The Rotor builder page's views: `RotorCanvas`, the rotor from the side, which builds it with the tool picked
(split the shaft, put a disc, bearing or seal on it, drag a part along it) and bends in the mode or the unbalance
response; and `WhirlView`, the mode whirling in 3D. The page passes them the rotor and each frame's shape.
"""

from __future__ import annotations

import dataclasses
import math
from enum import Enum

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.rotor_builder import DISC_LIBRARY, TWO_PI, Bearing, BearingKind, Disc, Rotor, Seal, Selection, deflection
from .rotor_builder_trials import EMPTY
from .style import colors

PICK_PX = 8  # how near a click has to be to pick a part
AXIAL = 4.0  # the 3D view's display length of the shaft
WHIRL = 0.45  # display units: the 3D view's largest whirl
NAN2 = np.full(2, np.nan)


class Tool(Enum):
    SELECT = "Select"
    SPLIT = "Split"
    DISC = "Disc"
    BEARING = "Bearing"
    SEAL = "Seal"


TOOLS = tuple(Tool)
TOOL_HINTS = {
    Tool.SELECT: "Click a section or a part to edit it on the left; drag a part along the shaft. Delete removes it.",
    Tool.SPLIT: "Click the shaft to cut its section in two there (each keeps its share of the elements).",
    Tool.DISC: "Pick a disc from the library (or Custom), then click the shaft to put it there.",
    Tool.BEARING: "Pick a kind of bearing, then click the shaft to put one there.",
    Tool.SEAL: "Click the shaft to put a seal there.",
}
PART_OF_TOOL = {Tool.DISC: "disc", Tool.BEARING: "bearing", Tool.SEAL: "seal"}
CANVAS_TIP = (
    "<p>The rotor from the side, its radii stretched to be seen: the shaft's sections, the discs, the bearings "
    "below it (springs, an oil film's housing, a magnet's stator) and the seals' teeth. Pick a tool above. While "
    "it bends in the mode picked, the dotted line is its axis at rest.</p>"
)
WHIRL_TIP = (
    "<p>The mode whirling in 3D (exaggerated): the shaft's centre line, each node's orbit and the discs. Forward "
    "whirl goes round the way the shaft spins (x toward y); backward, against it.</p>"
)


# ------------------------------------------------------------------ drawing helpers
def _segments(points: list) -> tuple[np.ndarray, np.ndarray]:
    if not points:
        return np.zeros(0), np.zeros(0)
    a = np.array(points, dtype=float).reshape(-1, 2)
    return a[:, 0], a[:, 1]


def _box(z0: float, z1: float, y0: float, y1: float) -> list:
    return [[z0, y0], [z1, y0], [z1, y1], [z0, y1], [z0, y0], NAN2]


def _zigzag(z: float, top: float, bottom: float, width: float, turns: int = 4) -> list:
    """A spring drawn down from `top` to `bottom` at z."""
    s = np.linspace(0.0, 1.0, 2 * turns + 3)
    side = np.zeros_like(s)
    side[1:-1:2], side[2:-1:2] = 1.0, -1.0
    side[-2] = 0.0
    return [[z + width * w, top + (bottom - top) * t] for w, t in zip(side, s)] + [NAN2]


def _ground(z: float, y: float, width: float) -> list:
    out = [[z - width, y], [z + width, y], NAN2]
    for k in range(5):
        x = z - width + k * width / 2
        out += [[x, y], [x - width / 3, y - width / 2.5], NAN2]
    return out


class RotorCanvas(pg.PlotWidget):
    """The rotor from the side; builds it with the tool picked. Emits `edited(rotor, selection)` for every change
    and `selected(selection)` when the selection changes."""

    edited = QtCore.Signal(object, object)
    selected = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(CANVAS_TIP)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.hideButtons()
        for side in ("left", "bottom"):
            self.getAxis(side).enableAutoSIPrefix(False)
        self.setLabel("bottom", "z", units="m")
        self.setLabel("left", "r", units="m")
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.rotor = EMPTY
        self.tool = Tool.SELECT
        self.grid = 0.05
        self.selection: Selection | None = None
        self.q: np.ndarray | None = None  # the displacement drawn (real, n_dof)
        self.template: dict[str, Disc | Bearing | Seal] = {
            "disc": DISC_LIBRARY[0].at(0.0),
            "bearing": Bearing(0.0),
            "seal": Seal(0.0),
        }
        self.height_m = 0.2  # the view's half height (m): symbols are sized from it
        self._drag: tuple[str, int] | None = None
        self._dragged: Rotor | None = None
        plot = self.getPlotItem()
        self.axis_line = pg.PlotCurveItem()
        self.ghost = pg.PlotCurveItem(connect="finite")
        self.shaft = pg.PlotCurveItem(connect="finite")
        self.bore = pg.PlotCurveItem(connect="finite")
        self.discs = pg.PlotCurveItem(connect="finite")
        self.supports = pg.PlotCurveItem(connect="finite")
        self.seals = pg.PlotCurveItem(connect="finite")
        self.highlight = pg.PlotCurveItem(connect="finite")
        self.nodes = pg.PlotDataItem(pen=None, symbol="o", symbolSize=4)
        for item in (
            self.axis_line,
            self.ghost,
            self.supports,
            self.seals,
            self.bore,
            self.shaft,
            self.discs,
            self.highlight,
            self.nodes,
        ):
            plot.addItem(item)
        self.apply_theme()

    # --------------------------------------------------------- state
    def set_rotor(self, rotor: Rotor, fit: bool = True) -> None:
        self.rotor = rotor
        if fit:
            self.fit()
        self.draw()

    def set_tool(self, tool: Tool) -> None:
        self.tool = tool
        self.draw()

    def fit(self) -> None:
        r = self.rotor
        length = max(r.length, 4 * self.grid, 1e-3)
        tall = max([s.outer / 2 for s in r.sections] + [d.diameter / 2 for d in r.discs] + [0.02])
        self.height_m = 1.6 * tall
        pad = 0.08 * length
        self.setRange(xRange=(-pad, length + pad), yRange=(-1.5 * tall, 1.3 * tall), padding=0)

    def snap(self, z: float) -> float:
        return float(np.clip(round(z / self.grid) * self.grid, 0.0, self.rotor.length))

    def _pixel(self) -> tuple[float, float]:
        sx, sy = self.getPlotItem().getViewBox().viewPixelSize()
        return sx, sy

    def part_near(self, p: np.ndarray) -> Selection | None:
        """The part (disc, bearing or seal) whose z is nearest p's, within PICK_PX."""
        sx, _ = self._pixel()
        best, reach = None, PICK_PX * sx
        for kind in ("disc", "bearing", "seal"):
            for i, part in enumerate(self.rotor.parts(kind)):
                d = abs(part.z - p[0])
                if kind == "disc":
                    d = max(d - part.thickness / 2, 0.0) if abs(p[1]) <= part.diameter / 2 else math.inf
                elif kind == "bearing" and p[1] > 0:
                    d = math.inf
                if d <= reach:
                    best, reach = (kind, i), d
        return best

    def section_near(self, p: np.ndarray) -> int | None:
        k = self.rotor.section_at(float(p[0]))
        if k is None:
            return None
        _, sy = self._pixel()
        return k if abs(p[1]) <= max(self.rotor.sections[k].outer / 2, PICK_PX * sy) else None

    def select(self, selection: Selection | None) -> None:
        self.selection = selection
        self.draw()
        self.selected.emit(selection)

    # --------------------------------------------------------- editing
    def _commit(self, rotor: Rotor, selection: Selection | None) -> None:
        self.selection = selection
        self.edited.emit(rotor, selection)

    def new_part(self, kind: str, z: float):
        """A new part of `kind` at z, made from its template (a bearing's film fitted to the shaft there)."""
        part = dataclasses.replace(self.template[kind], z=z)
        if isinstance(part, Bearing):
            part = dataclasses.replace(part, film=dataclasses.replace(part.film, diameter=self.rotor.outer_at(z)))
        return part

    def press(self, p: np.ndarray, right: bool = False) -> None:
        """A click at p (view coordinates): what the tool does."""
        if right:
            return
        r = self.rotor
        tool = self.tool
        hit = self.part_near(p)
        if tool is Tool.SELECT:
            if hit is not None:
                self._drag, self._dragged = hit, None
                self.select(hit)
            else:
                k = self.section_near(p)
                self.select(("section", k) if k is not None else None)
            return
        k = r.section_at(float(p[0]))
        if k is None:
            return
        z = self.snap(float(p[0]))
        if tool is Tool.SPLIT:
            k = r.section_at(z)
            if k is not None:
                split = r.split(k, z)
                if split is not r:
                    self._commit(split, ("section", k + 1))
            return
        kind = PART_OF_TOOL[tool]
        there = r.part_at(kind, z, reach=0.5 * self.grid)
        if there is not None:
            self.select((kind, there))
            return
        self._commit(r.with_part(kind, self.new_part(kind, z)), (kind, len(r.parts(kind))))

    def drag_to(self, p: np.ndarray) -> None:
        if self._drag is None:
            return
        kind, i = self._drag
        z = self.snap(float(p[0]))
        part = self.rotor.parts(kind)[i]
        other = self.rotor.part_at(kind, z)
        if other is None or other == i:
            self._dragged = self.rotor.replace_part(kind, i, dataclasses.replace(part, z=z))
            self.draw(self._dragged)

    def release(self) -> None:
        if self._drag is not None and self._dragged is not None:
            self._commit(self._dragged, self._drag)
        self._drag, self._dragged = None, None

    def delete_selected(self) -> None:
        if self.selection is None:
            return
        kind, i = self.selection
        if kind == "section":
            if len(self.rotor.sections) <= 1:
                return
            rotor = self.rotor.without_section(i)
        else:
            rotor = self.rotor.without_part(kind, i)
        self._commit(rotor, None)

    def _view_point(self, event) -> np.ndarray:
        scene = self.mapToScene(event.position().toPoint())
        p = self.getPlotItem().getViewBox().mapSceneToView(scene)
        return np.array([p.x(), p.y()])

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        self.press(self._view_point(event), right=event.button() == QtCore.Qt.MouseButton.RightButton)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        self.drag_to(self._view_point(event))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self.release()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        event.accept()

    def wheelEvent(self, event) -> None:
        event.ignore()  # the page scrolls; the view fits the rotor

    def keyPressEvent(self, event) -> None:
        if event.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            self.delete_selected()
        else:
            super().keyPressEvent(event)

    # --------------------------------------------------------- drawing
    def draw(self, rotor: Rotor | None = None) -> None:
        r = rotor if rotor is not None else self.rotor
        q = self.q if rotor is None and self.q is not None and self.q.size == r.n_dof else None
        h = self.height_m
        if q is not None:
            zc, _, yc = deflection(r, q)
            slope = q[3::4]
        else:
            zc, yc = r.nodes, np.zeros(r.nodes.size)
            slope = np.zeros(r.nodes.size)

        def centre(z: float) -> float:
            return float(np.interp(z, zc, yc)) if zc.size else 0.0

        def tilt(z: float) -> float:
            return float(np.interp(z, r.nodes, slope)) if r.nodes.size else 0.0

        ends = r.ends
        shaft: list = []
        bore: list = []
        ghost: list = []
        for s, a, b in zip(r.sections, ends[:-1], ends[1:]):
            inside = (zc >= a - 1e-12) & (zc <= b + 1e-12)
            zs, ys = zc[inside], yc[inside]
            for sign in (1.0, -1.0):
                shaft += [*np.column_stack([zs, ys + sign * s.outer / 2]), NAN2]
                if s.inner > 0:
                    bore += [*np.column_stack([zs, ys + sign * s.inner / 2]), NAN2]
            for z in (a, b):
                shaft += [[z, centre(z) - s.outer / 2], [z, centre(z) + s.outer / 2], NAN2]
        if q is not None:
            ghost = [[0.0, 0.0], [r.length, 0.0], NAN2]
        discs: list = []
        for d in r.discs:
            R, w = d.diameter / 2, d.thickness / 2
            lean = 0.04 * max(r.length, 0.1) / R  # a big disc's tilt, drawn true, would leave the view
            c, t = centre(d.z), float(np.clip(tilt(d.z), -lean, lean))
            corners = [(-w, -R), (w, -R), (w, R), (-w, R), (-w, -R)]
            discs += [[d.z + dz - t * dy, c + dy] for dz, dy in corners] + [NAN2]
        supports: list = []
        floor = -1.35 * h / 1.6
        for b in r.bearings:
            c = centre(b.z)
            rs = r.outer_at(b.z) / 2
            w = 0.03 * max(r.length, 0.1)
            if b.kind is BearingKind.SPRINGS:
                supports += _zigzag(b.z, c - rs, floor, 0.4 * w)
            elif b.kind is BearingKind.FILM:
                half = max(b.film.length / 2, 0.4 * w)
                supports += _box(b.z - half, b.z + half, -rs - 0.15 * h, rs + 0.15 * h)
                supports += [[b.z, -rs - 0.15 * h], [b.z, floor], NAN2]
            else:
                gap = 0.12 * h
                for sign in (1.0, -1.0):
                    y0 = sign * (rs + gap)
                    supports += _box(b.z - w, b.z + w, y0, y0 + sign * 0.25 * h)
                supports += [[b.z + w, -rs - gap - 0.25 * h], [b.z + w, floor], NAN2]
            supports += _ground(b.z, floor, w)
        seals: list = []
        for seal in r.seals:
            rs = r.outer_at(seal.z) / 2
            w = 0.025 * max(r.length, 0.1)
            for sign in (1.0, -1.0):
                y0, y1 = sign * (rs + 0.04 * h), sign * (rs + 0.22 * h)
                seals += [[seal.z - w, y1], [seal.z + w, y1], NAN2]
                for k in range(4):
                    z = seal.z - w + (k + 0.5) * w / 2
                    seals += [[z, y1], [z, y0], NAN2]
        self.shaft.setData(*_segments(shaft))
        self.bore.setData(*_segments(bore))
        self.ghost.setData(*_segments(ghost))
        self.discs.setData(*_segments(discs))
        self.supports.setData(*_segments(supports))
        self.seals.setData(*_segments(seals))
        if q is None:
            self.axis_line.setData([0.0, r.length], [0.0, 0.0])
        else:
            self.axis_line.setData([], [])
        node_y = yc[np.searchsorted(zc, r.nodes).clip(0, max(zc.size - 1, 0))] if zc.size else np.zeros(0)
        self.nodes.setData(r.nodes, node_y)
        self.highlight.setData(*_segments(self._lit(r, centre)))

    def _lit(self, r: Rotor, centre) -> list:
        if self.selection is None:
            return []
        kind, i = self.selection
        h = self.height_m
        if kind == "section" and i < len(r.sections):
            a, b = r.ends[i], r.ends[i + 1]
            R = r.sections[i].outer / 2
            return _box(a, b, -R, R)
        if kind == "disc" and i < len(r.discs):
            d = r.discs[i]
            c = centre(d.z)
            return _box(d.z - d.thickness / 2, d.z + d.thickness / 2, c - d.diameter / 2, c + d.diameter / 2)
        if kind in ("bearing", "seal") and i < len(r.parts(kind)):
            z = r.parts(kind)[i].z
            w = 0.04 * max(r.length, 0.1)
            top = r.outer_at(z) / 2 + 0.3 * h
            return _box(z - w, z + w, -1.35 * h / 1.6 if kind == "bearing" else -top, top)
        return []

    def apply_theme(self) -> None:
        c = colors
        self.axis_line.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashDotLine))
        self.ghost.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.shaft.setPen(pg.mkPen(c.structure, width=2))
        self.bore.setPen(pg.mkPen(c.structure, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.discs.setPen(pg.mkPen(c.mass[0], width=2.5))
        self.supports.setPen(pg.mkPen(c.structure, width=1.5))
        self.seals.setPen(pg.mkPen(c.mass[1], width=1.5))
        self.highlight.setPen(pg.mkPen(c.force, width=2.5))
        self.nodes.setSymbolBrush(pg.mkBrush(c.structure))
        self.nodes.setSymbolPen(None)
        self.draw()


class WhirlView(pg.PlotWidget):
    """The mode whirling in 3D, orthographically projected: the centre line, the orbits, the discs."""

    YAW, PITCH = math.radians(62.0), math.radians(18.0)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("The whirl shape along the shaft")
        self.setToolTip(WHIRL_TIP)
        self.setAspectLocked(True)
        for side in ("left", "bottom"):
            self.hideAxis(side)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.rotor = EMPTY
        self.shape: np.ndarray | None = None  # complex, n_dof, scaled to display units
        plot = self.getPlotItem()
        self.axis_line = pg.PlotCurveItem(connect="finite")
        self.orbits = pg.PlotCurveItem(connect="finite")
        self.supports = pg.PlotCurveItem(connect="finite")
        self.line = pg.PlotCurveItem()
        self.discs = pg.PlotCurveItem(connect="finite")
        self.dots = pg.PlotDataItem(pen=None, symbol="o", symbolSize=5)
        for item in (self.axis_line, self.supports, self.orbits, self.line, self.discs, self.dots):
            plot.addItem(item)
        self.apply_theme()

    def project(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Screen (x, y) of display points p (..., 3) = (lateral x, vertical y, axial)."""
        cy, sy = math.cos(self.YAW), math.sin(self.YAW)
        cp, sp = math.cos(self.PITCH), math.sin(self.PITCH)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        xr = x * cy + z * sy
        zr = -x * sy + z * cy
        return xr, y * cp - zr * sp

    def _lines(self, parts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        gap = np.full((1, 3), np.nan)
        pts = np.vstack([np.vstack([p, gap]) for p in parts]) if parts else np.empty((0, 3))
        return self.project(pts)

    def _axial(self, z: np.ndarray | float) -> np.ndarray:
        L = max(self.rotor.length, 1e-6)
        return (np.asarray(z) / L - 0.5) * AXIAL

    def _radius(self, r: float) -> float:
        tall = max([s.outer / 2 for s in self.rotor.sections] + [d.diameter / 2 for d in self.rotor.discs] + [1e-3])
        return 0.7 * r / tall

    def set_mode(self, rotor: Rotor, shape: np.ndarray | None) -> None:
        """The rotor and its mode (complex, n_dof, its largest translation 1: None for none)."""
        self.rotor = rotor
        self.shape = None if shape is None else shape * WHIRL
        z = self._axial(rotor.nodes)
        zero = np.zeros_like(z)
        self.axis_line.setData(*self._lines([np.column_stack([zero, zero, z])]))
        supports = []
        for b in rotor.bearings:
            u = float(self._axial(b.z))
            supports.append(np.array([[0.0, 0.0, u], [0.0, -1.0, u]]))
            supports.append(np.array([[-0.15, -1.0, u], [0.15, -1.0, u]]))
        self.supports.setData(*self._lines(supports))
        orbits = []
        if self.shape is not None:
            t = np.linspace(0.0, TWO_PI, 49)
            e = np.exp(1j * t)
            X, Y = self.shape[0::4], self.shape[2::4]
            for k in range(z.size):
                orbits.append(np.column_stack([(X[k] * e).real, (Y[k] * e).real, np.full(t.size, z[k])]))
        self.orbits.setData(*self._lines(orbits))
        self.setRange(xRange=(-0.6 * AXIAL, 0.6 * AXIAL), yRange=(-1.4, 1.2), padding=0)
        self.show_phase(0.0)

    def show_phase(self, phase: float) -> None:
        rotor = self.rotor
        if self.shape is not None:
            q = (self.shape * np.exp(1j * phase)).real
            zc, xc, yc = deflection(rotor, q)
        else:
            q = np.zeros(rotor.n_dof)
            zc, xc, yc = rotor.nodes, np.zeros(rotor.nodes.size), np.zeros(rotor.nodes.size)
        u = self._axial(zc)
        self.line.setData(*self.project(np.column_stack([xc, yc, u])))
        self.dots.setData(*self.project(np.column_stack([q[0::4], q[2::4], self._axial(rotor.nodes)])))
        rings = []
        t = np.linspace(0.0, TWO_PI, 41)
        for d in rotor.discs:
            j = rotor.node_at(d.z)
            R = self._radius(d.diameter / 2)
            c = np.array([q[4 * j], q[4 * j + 2], float(self._axial(rotor.nodes[j]))])
            tx, ty = q[4 * j + 1], q[4 * j + 3]  # slopes: the disc's normal is (tx, ty, 1) in metres
            scale = AXIAL / max(rotor.length, 1e-6)
            ring = np.column_stack([R * np.cos(t), R * np.sin(t), np.zeros_like(t)])
            ring[:, 2] = -(tx * ring[:, 0] + ty * ring[:, 1]) / scale  # square to the shaft's tangent
            rings.append(ring + c)
        self.discs.setData(*self._lines(rings))

    def apply_theme(self) -> None:
        c = colors
        self.axis_line.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashDotLine))
        self.orbits.setPen(pg.mkPen(c.grey, width=1))
        self.supports.setPen(pg.mkPen(c.structure, width=1.5))
        self.line.setPen(pg.mkPen(c.strong, width=3))
        self.discs.setPen(pg.mkPen(c.mass[0], width=2))
        self.dots.setSymbolBrush(pg.mkBrush(c.strong))
        self.dots.setSymbolPen(None)
