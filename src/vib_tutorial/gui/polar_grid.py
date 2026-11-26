"""A polar plot's grid, and a key to which way its angles count, for the rotor page's 1X vectors.

The plot is the disk seen from beyond bearing B (z toward you: x right, y up) as the keyphasor mark passes
+x. Rings of amplitude (labeled toward the bottom), spokes every 30° labeled 0° to 330°
counter-clockwise from the mark, and outside them the key: the keyphasor mark at 0°, an arrow for the
rotation and one for a positive phase (both counter-clockwise: the page counts phase as a lead, in the
direction of spin, as it counts every angle on the rotor), and the x and y probes at 0° and 90°. A 1X
vector V is drawn at its complex value, angle arg V. Read at the x probe, beside the keyphasor, it then
points at the high spot on the disk as drawn, and a weight at θ from the mark sits at θ on the grid
(`ring_point`). A y probe, a quarter turn on, reads 90° less.

With no axes, the plot needs a name for a screen reader: give it one where it's made. `fit(r)` sizes the rings to hold radius r, redrawing them only when the ring step changes (a run-up moves
its vector every frame); the view's range is set to the grid, so it doesn't rescale as the data move.

Working on it: `tests/test_rotor_balance_tab.py` checks the grid's labels and that a vector lands at its
angle; the rotor page's Bode tab and the balancing tab both use it.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg

from ..view_models.text import fmt
from .style import colors

SPOKE_DEG = 30
RING_POINTS = 121
KEY_RADIUS = 1.16  # the rotation's arrow, in units of the outer ring
PHASE_ARC = 0.42  # positive phase's arc, from the 0° spoke
RING_LABEL_DEG = 285.0  # the rings' labels lie along this direction, between two spokes
LABEL_RADIUS = 1.07  # the spokes' labels
ROOM = 1.5  # the view's half-width, in units of the outer ring


def nice_ring(r: float) -> tuple[float, int]:
    """(step, count): rings at step, 2 step, … count steps (2 to 4), the outer at or just beyond r.

    The step is r/4 rounded up to 1, 2, 2.5 or 5 × 10^k.
    """
    if not r > 0.0 or not math.isfinite(r):
        r = 1e-6
    raw = r / 4.0
    mag = 10.0 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1.0, 2.0, 2.5, 5.0, 10.0) if m * mag >= raw * (1.0 - 1e-9))
    return step, max(1, math.ceil(r / step * (1.0 - 1e-9)))


def _arc(radius: float, start: float, end: float, n: int = 24) -> tuple[np.ndarray, np.ndarray]:
    t = np.radians(np.linspace(start, end, n))
    return radius * np.cos(t), radius * np.sin(t)


def _arrow(radius: float, start: float, end: float, head: float) -> tuple[np.ndarray, np.ndarray]:
    """An arc from `start` to `end` (degrees), with a head at `end`, as one path (NaN breaks)."""
    x, y = _arc(radius, start, end)
    t = math.radians(end)
    tip = np.array([radius * math.cos(t), radius * math.sin(t)])
    along = np.sign(end - start) * np.array([-math.sin(t), math.cos(t)])  # the way the arc runs at its end
    out = np.array([math.cos(t), math.sin(t)])
    wings = [tip - head * along + 0.5 * head * out, tip - head * along - 0.5 * head * out]
    hx = [np.nan, wings[0][0], tip[0], wings[1][0]]
    hy = [np.nan, wings[0][1], tip[1], wings[1][1]]
    return np.concatenate([x, hx]), np.concatenate([y, hy])


class PolarGrid:
    """Rings, spokes and the angles' key on `plot`, which it takes over: no axes, a fixed range."""

    def __init__(self, plot: pg.PlotItem) -> None:
        self.plot = plot
        plot.setAspectLocked(True)
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        plot.showGrid(x=False, y=False)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        self.rings = pg.PlotCurveItem(connect="finite")
        self.outer = pg.PlotCurveItem()
        self.spokes = pg.PlotCurveItem(connect="pairs")
        self.key = pg.PlotCurveItem(connect="finite")  # the two arrows
        self.mark = pg.PlotCurveItem()  # the keyphasor mark: a tick across the outer ring at 0°
        self.probes = pg.PlotCurveItem(connect="finite")
        for item in (self.rings, self.outer, self.spokes, self.key, self.mark, self.probes):
            item.setZValue(-10)
            plot.addItem(item)
        self.angle_labels = []
        for deg in range(0, 360, SPOKE_DEG):
            label = pg.TextItem(f"{deg}°", anchor=(0.5, 0.5))
            self.angle_labels.append((deg, label))
        self.ring_labels: list[pg.TextItem] = []
        self.texts = {
            "mark": pg.TextItem(html="keyphasor<br>mark", anchor=(0.0, 1.0)),
            "spin": pg.TextItem("rotation", anchor=(1.0, 1.0)),
            "phase": pg.TextItem("+ phase", anchor=(0.0, 0.5)),
            "x": pg.TextItem("x probe", anchor=(0.0, 0.0)),
            "y": pg.TextItem("y probe", anchor=(0.0, 0.5)),
        }
        for item in [label for _, label in self.angle_labels] + list(self.texts.values()):
            item.setZValue(-5)
            plot.addItem(item)
        self.radius = 0.0  # the outer ring's
        self.apply_theme()
        self.fit(1e-6)

    def apply_theme(self) -> None:
        self.rings.setPen(pg.mkPen(colors.faint, width=1))
        self.outer.setPen(pg.mkPen(colors.grey, width=1))
        self.spokes.setPen(pg.mkPen(colors.faint, width=1))
        self.key.setPen(pg.mkPen(colors.muted, width=1.5))
        self.mark.setPen(pg.mkPen(colors.strong, width=3))
        self.probes.setPen(pg.mkPen(colors.structure, width=4))
        for _, label in self.angle_labels:
            label.setColor(colors.muted)
        for label in self.ring_labels:
            label.setColor(colors.muted)
        for name, text in self.texts.items():
            text.setColor(colors.foreground if name in ("mark", "spin", "phase") else colors.muted)

    def ring_point(self, angle: float) -> tuple[float, float]:
        """The point on the outer ring at `angle` (rad, counter-clockwise from the mark): where a weight goes."""
        return self.radius * math.cos(angle), self.radius * math.sin(angle)

    def fit(self, r: float) -> None:
        """Rings enough to hold radius r (m); redrawn only when the outer ring changes."""
        step, count = nice_ring(r)
        R = step * count
        if R == self.radius:
            return
        self.radius = R
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for k in range(1, count):
            x, y = _arc(k * step, 0.0, 360.0, RING_POINTS)
            xs += [x, np.array([np.nan])]
            ys += [y, np.array([np.nan])]
        self.rings.setData(np.concatenate(xs) if xs else [], np.concatenate(ys) if ys else [])
        self.outer.setData(*_arc(R, 0.0, 360.0, RING_POINTS))
        t = np.radians(np.arange(0, 360, SPOKE_DEG))
        sx = np.column_stack([np.zeros_like(t), R * np.cos(t)]).ravel()
        sy = np.column_stack([np.zeros_like(t), R * np.sin(t)]).ravel()
        self.spokes.setData(sx, sy)
        for deg, label in self.angle_labels:
            a = math.radians(deg)
            label.setPos(LABEL_RADIUS * R * math.cos(a), LABEL_RADIUS * R * math.sin(a))
        for label in self.ring_labels:
            self.plot.removeItem(label)
        self.ring_labels = []
        along = math.radians(RING_LABEL_DEG)
        for k in range(1, count + 1):
            label = pg.TextItem(fmt(k * step, "m"), anchor=(0.5, 0.5))
            r = k * step - 0.07 * R  # just inside its ring, clear of the spokes' labels
            label.setPos(r * math.cos(along), r * math.sin(along))
            label.setZValue(-5)
            self.plot.addItem(label)
            self.ring_labels.append(label)
        # The keyphasor mark: a tick across the outer ring at 0°, named above the 0° spoke's label.
        self.mark.setData([0.94 * R, 1.0 * R], [0.0, 0.0])
        # Positive phase: an angle's arc from the 0° spoke, inside the rings; the rotation: an arc outside them.
        head = 0.06 * R
        phase = _arrow(PHASE_ARC * R, 4.0, 40.0, 0.7 * head)
        spin = _arrow(KEY_RADIUS * R, 105.0, 150.0, head)
        self.key.setData(np.concatenate([spin[0], [np.nan], phase[0]]), np.concatenate([spin[1], [np.nan], phase[1]]))
        # The probes: short bars pointing at the shaft, outside the spokes' labels.
        px = [1.17 * R, 1.29 * R, np.nan, 0.0, 0.0]
        py = [0.0, 0.0, np.nan, 1.17 * R, 1.29 * R]
        self.probes.setData(px, py)
        a = math.radians(22.0)
        self.texts["phase"].setPos(1.05 * PHASE_ARC * R * math.cos(a), 1.05 * PHASE_ARC * R * math.sin(a))
        a = math.radians(140.0)
        self.texts["spin"].setPos(1.2 * R * math.cos(a), 1.2 * R * math.sin(a))
        self.texts["mark"].setPos(1.0 * R, 0.07 * R)
        self.texts["x"].setPos(1.13 * R, -0.04 * R)
        self.texts["y"].setPos(0.05 * R, 1.23 * R)
        self.apply_theme()
        self.plot.getViewBox().setRange(xRange=(-ROOM * R, ROOM * R), yRange=(-1.25 * R, 1.4 * R), padding=0)
