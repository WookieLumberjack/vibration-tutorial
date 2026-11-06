"""The rotor page's 3D view: the shaft, the disc and the bearings, whirling (magnified)."""

from __future__ import annotations

import cmath
import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.rotor import (
    CRACK_DEPTH_MAX,
    RotorSystem,
)
from ..core.rotor_sweep import (
    STATIONS,
)
from .lesson import Dot
from .rotor_plots import MIN_PEAK
from .style import colors

DASH = QtCore.Qt.PenStyle.DashLine

AXIAL = 4.0  # display length of the shaft in the 3D view


SWING = 0.5  # display units: the largest whirl is magnified to this


SHAFT_DOT = 0.1  # display units: how far off the shaft's centre line a weight on it is drawn


CRACK_TICK = 0.22  # display units: the longest tick a crack is drawn as, from the shaft's centre line


GROUND = 1.1  # display units from the shaft's axis to the supports


HOUSING = 0.16  # display units: an oil-film bearing's housing, drawn round the bearing's axis


STATOR = 0.22  # display units: a magnetic bearing's stator ring, round the bearing's axis


VIEW_TIP = (
    "<p>The shaft's centre line, the disc and the bearings' springs and dampers, with the whirl "
    "magnified (the scale is in the top corner). Drag to turn the view; double-click to reset it.</p>"
    "<p>The red dot on the disc is the heavy spot, where its unbalance is: it turns with the shaft. The "
    "tick across the rim is the keyphasor mark that angles are measured from, and squares are balance "
    "weights. An asymmetric shaft's stiff axis is the dashed diameter across the disc; a crack is the "
    "dark tick on the shaft, pointing at its mouth and thick while it is open. "
    "The thin lines are the orbits of the four stations (bearings, disc, midspan).</p>"
)


class RotorView(pg.PlotWidget):
    """The rotor in 3D, orthographically projected; drag to turn it."""

    YAW, PITCH = math.radians(62.0), math.radians(18.0)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(VIEW_TIP)
        self.setAspectLocked(True)
        for side in ("left", "bottom"):
            self.hideAxis(side)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.yaw, self.pitch = self.YAW, self.PITCH
        self._drag: QtCore.QPointF | None = None
        self.system = RotorSystem()
        self.gain = 1.0  # display units per metre of lateral motion
        self.disc_radius = 0.4  # display units

        plot = self.getPlotItem()
        self.frame = pg.PlotCurveItem(connect="finite")  # base plate, side walls, axis
        self.axis = pg.PlotCurveItem()
        self.supports = pg.PlotCurveItem(connect="finite")  # springs and dampers
        self.trails = [pg.PlotCurveItem() for _ in STATIONS]
        self.shaft = pg.PlotCurveItem()
        self.disc = QtWidgets.QGraphicsPathItem()
        self.spoke = pg.PlotCurveItem()
        self.heavy = Dot(9)
        self.weight_dots = pg.ScatterPlotItem(size=7, symbol="s")  # balance weights, on the disc or the shaft
        self.mark = pg.PlotCurveItem()  # the keyphasor mark: a tick across the disc's rim
        self.stiff_axis = pg.PlotCurveItem()  # an asymmetric shaft's stiff axis: a diameter across the disc
        self.crack_mark = pg.PlotCurveItem()  # a crack: a tick on the shaft toward its mouth
        self.stator_ring = pg.PlotCurveItem(connect="finite")  # a stator ring round the disc, on a pedestal
        self.ring: tuple[float, float, float] | None = None  # its centre (x, y) and clearance (m), as drawn
        self.show_heavy = True
        self.stations = pg.ScatterPlotItem(size=8)
        self.triad = pg.PlotCurveItem(connect="pairs")
        self.triad_labels = [pg.TextItem(t, anchor=(0.5, 0.5)) for t in ("x", "y", "z")]
        self.end_labels = [pg.TextItem(t, anchor=(0.5, 0.0)) for t in ("A", "B")]
        for item in (
            self.frame,
            self.axis,
            self.supports,
            self.stator_ring,
            *self.trails,
            self.shaft,
            self.disc,
            self.spoke,
            self.stiff_axis,
            self.crack_mark,
            self.weight_dots,
            self.mark,
            self.stations,
            self.triad,
            *self.triad_labels,
            *self.end_labels,
        ):
            plot.addItem(item)
        plot.addItem(self.heavy, ignoreBounds=True)
        self.scale_label = pg.TextItem(anchor=(1.0, 0.0))
        plot.addItem(self.scale_label)
        self.apply_theme()
        self._place_static()

    # ------------------------------------------------------------- geometry
    def project(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Screen (x, y) of display points p (..., 3) = (lateral x, vertical y, axial z from midspan)."""
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        xr = x * cy + z * sy
        zr = -x * sy + z * cy  # toward the viewer
        return xr, y * cp - zr * sp

    @property
    def _metres(self) -> float:
        """Metres of shaft per display unit along it: the whole shaft, overhang too, spans AXIAL."""
        z0, z1 = self.system.extent
        return (z1 - z0) / AXIAL

    def _z(self, z_m: float | np.ndarray) -> np.ndarray:
        """Display axial coordinate (from the shaft's middle) of an axial position in metres."""
        z0, z1 = self.system.extent
        return (np.asarray(z_m) - 0.5 * (z0 + z1)) / self._metres

    def set_system(self, system: RotorSystem) -> None:
        self.system = system
        r = math.sqrt(2.0 * system.ip / system.disc_mass) if system.ip > 0 else 0.05
        self.disc_radius = float(np.clip(r / self._metres, 0.2, 0.9))
        self._place_static()

    def _segments(self, parts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Screen coordinates of polylines joined with NaN breaks (for connect='finite')."""
        gap = np.full((1, 3), np.nan)
        pts = np.vstack([np.vstack([p, gap]) for p in parts]) if parts else np.empty((0, 3))
        return self.project(pts)

    def _place_static(self) -> None:
        """The base plate, the side walls, the undeflected axis, the triad and the view's range."""
        h, half = GROUND, AXIAL / 2 + 0.5
        w = 0.8
        plate = np.array([[-w, -h, -half], [w, -h, -half], [w, -h, half], [-w, -h, half], [-w, -h, -half]])
        walls = []
        bearings = self._z(np.array([0.0, self.system.length]))
        for zb in bearings:
            walls.append(
                np.array(
                    [
                        [-h, -0.35, zb - 0.3],
                        [-h, 0.35, zb - 0.3],
                        [-h, 0.35, zb + 0.3],
                        [-h, -0.35, zb + 0.3],
                        [-h, -0.35, zb - 0.3],
                    ]
                )
            )
        self.frame.setData(*self._segments([plate, *walls]))
        self.axis.setData(*self.project(np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])))
        o = np.array([w + 0.3, -h, half - 0.2])
        ends = [o + d for d in (np.array([0.45, 0, 0]), np.array([0, 0.45, 0]), np.array([0, 0, -0.45]))]
        tx, ty = self.project(np.array([v for e in ends for v in (o, e)]))
        self.triad.setData(tx, ty)
        for label, e in zip(self.triad_labels, ends):
            label.setPos(*[float(c) for c in self.project(o + 1.3 * (e - o))])
        for label, zb in zip(self.end_labels, bearings):
            label.setPos(*[float(c) for c in self.project(np.array([0.0, -h - 0.05, zb]))])
        # Range: everything the rotor can reach.
        top = SWING + self.disc_radius + 0.1
        corners = np.array(
            [[sx * (h + 0.2), sy, sz * half] for sx in (-1, 1) for sy in (-h - 0.15, top) for sz in (-1, 1)]
        )
        px, py = self.project(corners)
        self.setRange(xRange=(px.min(), px.max()), yRange=(py.min(), py.max()), padding=0.02)
        vr = self.getViewBox().viewRange()
        self.scale_label.setPos(vr[0][1], vr[1][1])

    def apply_theme(self) -> None:
        self.frame.setPen(pg.mkPen(colors.faint, width=1))
        self.axis.setPen(pg.mkPen(colors.faint, width=1, style=DASH))
        self.supports.setPen(pg.mkPen(colors.structure, width=1))
        self.stator_ring.setPen(pg.mkPen(colors.structure, width=2))
        for curve, i in zip(self.trails, range(len(STATIONS))):
            curve.setPen(pg.mkPen(colors.mass[i], width=1))
        self.shaft.setPen(pg.mkPen(colors.structure, width=3))
        fill = QtGui.QColor(colors.mass[1])
        fill.setAlpha(150)
        self.disc.setBrush(fill)
        pen = QtGui.QPen(QtGui.QColor(colors.mass[1]))
        pen.setCosmetic(True)
        pen.setWidth(2)
        self.disc.setPen(pen)
        self.spoke.setPen(pg.mkPen(colors.force, width=2))
        self.heavy.setBrush(pg.mkBrush(colors.force))
        self.heavy.setPen(pg.mkPen(colors.background, width=1))
        self.weight_dots.setBrush(pg.mkBrush(colors.strong))
        self.weight_dots.setPen(pg.mkPen(colors.background, width=1))
        self.mark.setPen(pg.mkPen(colors.strong, width=2))
        self.stiff_axis.setPen(pg.mkPen(colors.strong, width=1.5, style=DASH))
        self.stations.setPen(pg.mkPen(colors.background, width=1))
        self.triad.setPen(pg.mkPen(colors.muted, width=1))
        for label in (*self.triad_labels, *self.end_labels, self.scale_label):
            label.setColor(colors.muted)

    # ------------------------------------------------------------- drawing
    def update_state(self, q: np.ndarray, phase: float, trails: list[np.ndarray], peak: float) -> None:
        """Draw displacements q (n_dof,) with the keyphasor mark at angle `phase`, and each station's recent (x, y)."""
        s = self.system
        # The disc's own axis: the shaft's slope there plus the skew. Its rim's wobble counts as motion.
        axis = np.array([q[2], q[6]]) + s.skew * np.array(
            [math.cos(phase + s.skew_angle), math.sin(phase + s.skew_angle)]
        )
        peak = max(peak, float(np.hypot(*axis)) * self.disc_radius * self._metres)
        if self.ring is not None:  # the ring in view too
            peak = max(peak, math.hypot(self.ring[0], self.ring[1]) + self.ring[2])
        target = SWING / max(peak, MIN_PEAK)
        # Shrink at once (keep the motion on screen), grow slowly (no flicker as it settles).
        self.gain = target if target < self.gain else self.gain + 0.08 * (target - self.gain)
        g = self.gain
        self.scale_label.setText(f"motion ×{_round_gain(g * self._metres)}")

        # Shaft centre line.
        z = np.union1d(np.linspace(*s.extent, 81), s.nodes)
        defl = s.deflection(q, z) * g
        line = np.column_stack([defl, self._z(z)])
        self.shaft.setData(*self.project(line))

        # Stations and their orbits.
        zs = np.array([0.0, s.a, 0.5 * s.length, s.length])
        here = np.column_stack([s.deflection(q, zs) * g, self._z(zs)])
        # The midspan is drawn only when the disc is not there.
        shown = [i for i in range(len(STATIONS)) if i != 2 or abs(s.position - 0.5) > 1e-9]
        sx, sy = self.project(here[shown])
        self.stations.setData(sx, sy, brush=[pg.mkBrush(colors.mass[i]) for i in shown])
        for i, (curve, xy, zc) in enumerate(zip(self.trails, trails, self._z(zs))):
            if xy.size and i in shown:
                pts = np.column_stack([xy * g, np.full(len(xy), zc)])
                curve.setData(*self.project(pts))
            else:
                curve.setData([], [])

        # Disc: a circle square to its own (magnified) axis.
        scale = g * self._metres  # display slope per radian of real slope
        n = np.array([axis[0] * scale, axis[1] * scale, 1.0])
        n /= np.linalg.norm(n)
        u = np.cross([0.0, 1.0, 0.0], n)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        c = here[1]
        t = np.linspace(0.0, 2 * math.pi, 49)
        rim = c + self.disc_radius * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
        rx, ry = self.project(rim)
        path = QtGui.QPainterPath()
        path.addPolygon(QtGui.QPolygonF([QtCore.QPointF(float(a), float(b)) for a, b in zip(rx, ry)]))
        self.disc.setPath(path)

        def on_disc(angle: float, r: float) -> np.ndarray:
            return c + r * self.disc_radius * (math.cos(angle) * u + math.sin(angle) * v)

        net = s.mass_unbalance
        if self.show_heavy and net:
            spot = on_disc(phase + math.atan2(net.imag, net.real), 0.8)
            self.spoke.setData(*self.project(np.vstack([c, spot])))
            self.heavy.setData(*self.project(spot[None, :]))
        else:
            self.spoke.setData([], [])
            self.heavy.setData([], [])
        dots = [on_disc(phase + cmath.phase(w), 0.92) for w in s.weights]
        unit = n / np.linalg.norm(n)
        for p in s.plane_weights:
            if s.on_disc(p):  # on the disc, at its axial offset
                dots.append(on_disc(phase + cmath.phase(p.weight), 0.92) + unit * (p.z - s.a) / self._metres)
            else:  # on the shaft, just off its centre line
                angle = phase + cmath.phase(p.weight)
                centre = s.deflection(q, [p.z])[0] * g
                dots.append(
                    [
                        centre[0] + SHAFT_DOT * math.cos(angle),
                        centre[1] + SHAFT_DOT * math.sin(angle),
                        float(self._z(p.z)),
                    ]
                )
        if dots:
            self.weight_dots.setData(*self.project(np.array(dots)))
        else:
            self.weight_dots.setData([], [])
        self.mark.setData(*self.project(np.vstack([on_disc(phase, 0.88), on_disc(phase, 1.12)])))
        if s.asymmetry:
            angle = phase + s.asymmetry_angle
            self.stiff_axis.setData(*self.project(np.vstack([on_disc(angle, 0.75), on_disc(angle + math.pi, 0.75)])))
        else:
            self.stiff_axis.setData([], [])
        if s.crack is not None and s.crack.depth:
            angle = phase + s.crack.angle
            centre = s.deflection(q, [s.crack_z])[0] * g
            reach = CRACK_TICK * (0.4 + 0.6 * s.crack.depth / CRACK_DEPTH_MAX)
            tick = np.array(
                [[centre[0], centre[1]], [centre[0] + reach * math.cos(angle), centre[1] + reach * math.sin(angle)]]
            )
            self.crack_mark.setData(*self.project(np.column_stack([tick, np.full(2, self._z(s.crack_z))])))
            self.crack_mark.setPen(pg.mkPen(colors.strong, width=1.5 + 3.0 * float(s.crack_open(phase))))
        else:
            self.crack_mark.setData([], [])

        # Supports: a spring and a damper to ground below (y) and to the side wall (x) at each bearing;
        # or an oil film's housing round the bearing's axis, on a pedestal; or a magnetic bearing's
        # stator ring with its four poles (a magnet pair on each axis), on a pedestal.
        parts = []
        ring = np.linspace(0.0, 2.0 * math.pi, 49)
        for station, bearing in ((0, s.bearing_a), (3, s.bearing_b)):
            j = here[station]
            if bearing.magnet is not None:
                zb = np.full_like(ring, j[2])
                parts.append(np.column_stack([STATOR * np.cos(ring), STATOR * np.sin(ring), zb]))
                for a in np.arange(4) * 0.5 * math.pi:
                    for side in (-0.25, 0.25):  # each pole a pair of lines toward the journal
                        u = np.array([math.cos(a + side * 0.6), math.sin(a + side * 0.6)])
                        parts.append(np.array([[*(STATOR * u), j[2]], [*(0.55 * STATOR * u), j[2]]]))
                parts.append(np.array([[0.0, -STATOR, j[2]], [0.0, -GROUND, j[2]]]))
                continue
            if bearing.film is not None:
                zb = np.full_like(ring, j[2])
                parts.append(np.column_stack([HOUSING * np.cos(ring), HOUSING * np.sin(ring), zb]))
                parts.append(np.array([[0.0, -HOUSING, j[2]], [0.0, -GROUND, j[2]]]))
                continue
            ground_y = np.array([0.0, -GROUND, j[2]])
            ground_x = np.array([-GROUND, 0.0, j[2]])
            for end in (ground_y, ground_x):
                parts += _spring_and_damper(j, end)
        self.supports.setData(*self._segments(parts))
        if self.ring is not None:
            cx, cy, clearance = self.ring
            r = self.disc_radius + clearance * g
            zd = float(self._z(s.a))
            circle = np.column_stack([cx * g + r * np.cos(ring), cy * g + r * np.sin(ring), np.full_like(ring, zd)])
            pedestal = np.array([[cx * g, cy * g - r, zd], [cx * g, -GROUND, zd]])
            self.stator_ring.setData(*self._segments([circle, pedestal]))
        else:
            self.stator_ring.setData([], [])

    # ------------------------------------------------------------- mouse
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
        self.yaw = (self.yaw - 0.01 * d.x()) % (2 * math.pi)
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


def _spring_and_damper(p0: np.ndarray, p1: np.ndarray) -> list[np.ndarray]:
    """Polylines for a spring and a damper side by side from p0 (the journal) to p1 (ground)."""
    d = p1 - p0
    side = np.array([0.0, 0.0, 0.12])  # spring and damper sit either side along the shaft
    s = np.linspace(0.0, 1.0, 15)
    zig = np.zeros_like(s)
    zig[2:-2:2], zig[3:-2:2] = 1.0, -1.0
    off = 0.06 * np.cross(d / np.linalg.norm(d), [0.0, 0.0, 1.0]) if abs(d[2]) < 1e-9 else np.zeros(3)
    spring = p0 - side + s[:, None] * d + zig[:, None] * off
    rod = np.vstack([p0 + side, p0 + side + 0.55 * d])
    w = off * 1.6
    cup = np.vstack(
        [p0 + side + 0.45 * d + w, p0 + side + 0.7 * d + w, p0 + side + 0.7 * d - w, p0 + side + 0.45 * d - w]
    )
    piston = np.vstack([p0 + side + 0.55 * d + 0.8 * w, p0 + side + 0.55 * d - 0.8 * w])
    tail = np.vstack([p0 + side + 0.7 * d, p1 + side])
    bar = np.vstack([p0 - side * 1.5, p0 + side * 1.5])
    return [spring, rod, cup, piston, tail, bar]


def _round_gain(g: float) -> str:
    """A magnification for the label, to two significant figures: 0.52, 9.7, 970, 12,000."""
    g = float(f"{g:.2g}")
    return f"{g:,.0f}" if g >= 10 else f"{g:.2g}"
