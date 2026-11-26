"""The quarter-car drawn driving over its road: the road scrolling under the tire, the wheel on the tire and
the body on the suspension, each moving as the drive's record has it.

Bottom to top: the road's profile around the car (the car stays put and the road goes by, at the page's speed),
the tire (a spring, kₜ) from the road under the wheel to the wheel (mᵤ), the suspension (a spring kₛ and a
damper cₛ) up to the body (mₛ), and the body's rest level, dashed. Heights are exaggerated, all by the same
factor, so the road, the wheel and the body keep their true proportions to each other: the body floating over
the bumps the wheel follows, or the wheel hopping on its tire.

- `set_record(t, road, body, wheel, speed)`: the drive (s, m, m, m and the speed in m/s) and its exaggeration,
  from the largest height any of them reaches.
- `set_time(t)`: the instant drawn (the page's `Playback`).

Working on it: `tests/test_playback.py` checks the wheel and the body follow the playback.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import coalesce
from .animation import damper_path, spring_path
from .style import colors

AHEAD = 4.0  # m of road drawn either side of the car
TYRE_H, WHEEL_H, SUSP_H, BODY_H = 0.45, 0.3, 0.7, 0.45  # the drawing's heights at rest (its units)
WHEEL_HALF, BODY_HALF = 0.45, 0.9  # half widths (m along the road)
SWING = 0.3  # the largest height drawn, in the drawing's units
Y_RANGE = (-SWING - 0.1, TYRE_H + WHEEL_H + SUSP_H + BODY_H + SWING + 0.1)
LEG = 0.22  # the suspension's spring and damper either side of the middle
TIP = (
    "<p>The car driving over the road, as the Drive tab's record has it. The road goes by under the tire; the "
    "wheel follows it on the tire, and the body rides on the suspension. Heights are exaggerated, all alike: "
    "the body floats over bumps the wheel follows.</p>"
)


def _vertical(along: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """A horizontal path (`spring_path`, `damper_path`) stood on end."""
    xs, ys = along
    return ys, xs


class QuarterCarView(pg.PlotWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("The quarter-car driving over the road")
        self.setToolTip(TIP)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.t = self.road = self.body = self.wheel = np.zeros(1)
        self.speed = 20.0
        self.gain = 1.0
        self.now = 0.0
        self.road_line = self._add(pg.PlotDataItem())
        self.rest = self._add(pg.PlotDataItem())
        self.tyre = self._add(pg.PlotDataItem(connect="finite"))
        self.suspension = self._add(pg.PlotDataItem(connect="finite"))
        self.damper = self._add(pg.PlotDataItem(connect="finite"))
        self.wheel_rect = self._add(QtWidgets.QGraphicsRectItem())
        self.body_rect = self._add(QtWidgets.QGraphicsRectItem())
        self.labels = (self._add(pg.TextItem("mᵤ", anchor=(0.5, 0.5))), self._add(pg.TextItem("mₛ", anchor=(0.5, 0.5))))
        self.setRange(QtCore.QRectF(-AHEAD, Y_RANGE[0], 2 * AHEAD, Y_RANGE[1] - Y_RANGE[0]), padding=0)
        self.apply_theme()

    def _add(self, item):
        self.addItem(item)
        return item

    def apply_theme(self) -> None:
        c = colors
        self.road_line.setPen(pg.mkPen(c.structure, width=3))
        for item in (self.tyre, self.suspension, self.damper):
            item.setPen(pg.mkPen(c.structure, width=2))
        self.rest.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.wheel_rect.setBrush(pg.mkBrush(c.mass[1]))
        self.body_rect.setBrush(pg.mkBrush(c.mass[0]))
        for rect in (self.wheel_rect, self.body_rect):
            rect.setPen(pg.mkPen(c.structure, width=1.5))
        for label in self.labels:
            label.setColor(c.background)
        self._draw()

    def set_record(self, t: np.ndarray, road: np.ndarray, body: np.ndarray, wheel: np.ndarray, speed: float) -> None:
        self.t, self.road, self.body, self.wheel = t, road, body, wheel
        self.speed = max(speed, 1e-6)
        peak = max(float(np.max(np.abs(a))) for a in (road, body, wheel))
        self.gain = SWING / max(peak, 1e-9)
        self._draw()

    def set_time(self, t: float) -> None:
        self.now = t
        self._draw()

    def _at(self, values: np.ndarray, t: float | np.ndarray):
        return np.interp(t, self.t, values)

    @coalesce.coalesced(leaf=True)
    def _draw(self) -> None:
        g, t = self.gain, self.now
        x = np.linspace(-AHEAD, AHEAD, 241)  # m along the road, the car at 0
        self.road_line.setData(x, g * self._at(self.road, t + x / self.speed))
        z, w, b = (float(self._at(a, t)) for a in (self.road, self.wheel, self.body))
        ground = g * z
        wheel = TYRE_H + g * w
        body = TYRE_H + WHEEL_H + SUSP_H + g * b
        self.tyre.setData(*_vertical(spring_path(ground, wheel, 0.0, coils=4, amp=0.12)))
        self.wheel_rect.setRect(QtCore.QRectF(-WHEEL_HALF, wheel, 2 * WHEEL_HALF, WHEEL_H))
        top = wheel + WHEEL_H
        self.suspension.setData(*_vertical(spring_path(top, body, -LEG, coils=5, amp=0.1)))
        self.damper.setData(*_vertical(damper_path(top, body, LEG, SUSP_H, h=0.09)))
        self.body_rect.setRect(QtCore.QRectF(-BODY_HALF, body, 2 * BODY_HALF, BODY_H))
        rest = TYRE_H + WHEEL_H + SUSP_H
        self.rest.setData([BODY_HALF + 0.2, BODY_HALF + 1.6], [rest, rest])
        self.labels[0].setPos(0.0, wheel + WHEEL_H / 2)
        self.labels[1].setPos(0.0, body + BODY_H / 2)
