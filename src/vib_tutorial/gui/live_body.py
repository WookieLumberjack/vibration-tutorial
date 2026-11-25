"""The live mass drawn (#478 items 1, 4): a mass on its spring and damper, and grips on the body that set m, k and c
by dragging. Its forces are drawn beside it, on a mass held still (`free_body_view.FreeBodyView`).

`LiveMassView` only draws and forwards what the hand does; what a drag means is the view model's
(`view_models/grips.py`'s `Grip`, `live_mass.LiveMass`'s `hold`/`let_go`):

- **Pluck:** press on the mass and drag it sideways: `grabbed`, `pulled(x)` (m, from rest) as it moves, `let_go`.
- **Grips:** the corner handle on the mass (drag up or right: heavier), the handle on the spring (up: stiffer) and
  the one on the damper (up: more damping): `grip_taken(name)`, `grip_moved(name, distance)` (display units since
  it was taken, + to grow), `grip_released(name)`. Each value is printed beside its grip.

The motion is drawn magnified (the scale bar under it): the largest recent swing fills `SWING` display units. The
scale holds still under the hand, so a pluck doesn't rescale as it's let go.

Working on it: `tests/test_sdof_page.py` drags the grips and plucks with synthetic mouse events and checks the page
follows. Colours come through `style.role`, so `apply_theme` redraws them.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..view_models.grips import body_size
from ..view_models.text import fmt
from ..view_models.text import number as sig
from .animation import damper_path, spring_path
from .drag import Grabbable
from .style import colors, text_on

REST_X = 1.3  # the mass's centre at rest (display units; the wall at 0)
SIDE = 0.42  # the reference mass's side
SPRING_Y, DAMPER_Y = 0.13, -0.13
SWING = 0.35  # display units the largest recent swing is drawn across
MAX_DRAG = 0.45  # farthest a pluck pulls the mass from rest (display units)
PLUCK_GAIN = SWING / 0.02  # at rest, a pluck's full pull is about 20 mm
MIN_PEAK = 1e-6  # m: never magnify motion smaller than this
GRIP_R = 0.06  # a grip's radius, and how near a press must be
GRIPS = ("mass", "stiffness", "damping")
TIP = (
    "<p>The mass on its spring and damper, moving as it runs (magnified: the bar underneath is to scale). The "
    "forces on it are beside it.</p>"
    "<p><b>Pluck it:</b> drag the mass sideways and let go. <b>Change it as it runs:</b> drag the handle on the "
    "mass's corner (heavier up or right), on the spring (stiffer up) or on the damper (more damping up). The "
    "motion carries on from where it is.</p>"
)


class LiveMassView(Grabbable, pg.PlotWidget):
    grabbed = QtCore.Signal()
    pulled = QtCore.Signal(float)  # m from rest
    let_go = QtCore.Signal()
    grip_taken = QtCore.Signal(str)
    grip_moved = QtCore.Signal(str, float)  # the grip, display units dragged since it was taken
    grip_released = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("The mass on its spring and damper, moving")
        self.setToolTip(TIP)
        self.setMouseTracking(True)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setAspectLocked(True)
        self.setRange(QtCore.QRectF(-0.25, -0.85, 2.95, 1.45), padding=0)  # the largest mass's grip; the scale's text
        self.gain = PLUCK_GAIN  # display units a metre
        self.held: object = False  # True while plucked, a grip's name while it's dragged
        self._grip_from: QtCore.QPointF | None = None
        self._grab_dx = 0.0  # where on the mass it was grabbed, from its centre: the mass moves as the hand does
        self._centre = REST_X
        self._side = SIDE
        self._values = {"mass": 1.0, "stiffness": 400.0, "damping": 2.0}

        def add(item, z: float = 0.0):
            item.setZValue(z)
            self.addItem(item)
            return item

        self.wall = add(pg.PlotDataItem([0, 0], [-0.45, 0.45]))
        hx, hy = [], []
        for yy in np.linspace(-0.4, 0.45, 9):
            hx += [0, -0.1, np.nan]
            hy += [yy, yy - 0.1, np.nan]
        self.hatch = add(pg.PlotDataItem(hx, hy, connect="finite"))
        self.rest = add(pg.PlotDataItem([REST_X, REST_X], [-0.62, -0.52]))
        self.spring = add(pg.PlotDataItem())
        self.damper = add(pg.PlotDataItem(connect="finite"))
        self.block = add(QtWidgets.QGraphicsRectItem(), 5)
        self.ghost = add(QtWidgets.QGraphicsRectItem(), 4)  # another mass beside it, outlined (the viscous ghost)
        self.ghost.hide()
        self.mass_label = add(pg.TextItem(anchor=(0.5, 0.5)), 6)
        self.k_label = add(pg.TextItem(anchor=(0.5, 1.0)), 6)
        self.c_label = add(pg.TextItem(anchor=(0.5, 0.0)), 6)
        self.grips = {name: add(QtWidgets.QGraphicsEllipseItem(), 8) for name in GRIPS}
        self.scale_bar = add(pg.PlotDataItem([0.0, SWING], [-0.68, -0.68]))
        self.scale_text = add(pg.TextItem(anchor=(0, 0.5)))
        self.scale_text.setPos(SWING + 0.05, -0.68)
        self.apply_theme()
        self._place(0.0)

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        pen = pg.mkPen(colors.structure, width=2)
        self.wall.setPen(pg.mkPen(colors.structure, width=4))
        for item in (self.hatch, self.spring, self.damper):
            item.setPen(pen)
        self.rest.setPen(pg.mkPen(colors.grey, width=1))
        self.block.setBrush(pg.mkBrush(colors.mass[0]))
        self.block.setPen(pg.mkPen(colors.structure, width=1.5))
        self.ghost.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
        self.ghost.setPen(pg.mkPen(colors.grey, width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.mass_label.setColor(text_on(colors.mass[0]))
        for label in (self.k_label, self.c_label):
            label.setColor(colors.muted)
        for grip in self.grips.values():
            grip.setBrush(pg.mkBrush(colors.background))
            grip.setPen(pg.mkPen(colors.strong, width=2))
        self.scale_bar.setPen(pg.mkPen(colors.grey, width=2))
        self.scale_text.setColor(colors.muted)

    def set_values(
        self, mass: float, stiffness: float, damping: float, reference: float = 1.0, damper: str | None = None
    ) -> None:
        """The values the grips show, and the mass's size (its side as a block of one material, against
        `reference`); `damper`, what the damper's label says in place of its c (a damping model's)."""
        self._values = {"mass": mass, "stiffness": stiffness, "damping": damping}
        self._side = SIDE * body_size(mass, reference)
        self.mass_label.setText(f"m = {sig(mass)} kg")
        self.k_label.setText(f"k = {fmt(stiffness, 'N/m', largest='M', smallest='m')}")
        c = f"c = {fmt(damping, 'N·s/m', largest='M', smallest='m')}"
        self.c_label.setText(c if damper is None else damper)  # drawn at the next `show`

    def fit(self, peak: float) -> None:
        """Magnify the motion so the largest recent swing, `peak` (m), fills SWING: at once if it grew, slowly as
        it dies away; never under the hand."""
        if self.held is not False:
            return
        target = SWING / max(peak, MIN_PEAK)
        self.gain = target if target < self.gain else self.gain + 0.03 * (target - self.gain)

    def show(self, x: float, ghost: float | None = None) -> None:
        """The mass at displacement x (m); `ghost`, another mass's displacement, outlined."""
        self._place(self.gain * x)
        self.ghost.setVisible(ghost is not None)
        if ghost is not None:
            s = self._side
            g = REST_X + self.gain * ghost
            self.ghost.setRect(QtCore.QRectF(g - s / 2, -s / 2, s, s))
        self.scale_text.setText(f"= {fmt(SWING / self.gain, 'm', largest='', smallest='µ')}")

    def _place(self, offset: float) -> None:
        s = self._side
        self._centre = xc = REST_X + offset
        left = xc - s / 2
        self.block.setRect(QtCore.QRectF(left, -s / 2, s, s))
        self.mass_label.setPos(xc, 0)
        self.spring.setData(*spring_path(0.0, left, SPRING_Y))
        self.damper.setData(*damper_path(0.0, left, DAMPER_Y, REST_X - SIDE / 2))
        self.k_label.setPos(0.45 * left, SPRING_Y + 0.17)
        self.c_label.setPos(0.45 * left, DAMPER_Y - 0.17)
        for name, (gx, gy) in self._grip_spots().items():
            self.grips[name].setRect(QtCore.QRectF(gx - GRIP_R, gy - GRIP_R, 2 * GRIP_R, 2 * GRIP_R))

    def _grip_spots(self) -> dict[str, tuple[float, float]]:
        s, left = self._side, self._centre - self._side / 2
        return {
            "mass": (left + s, s / 2),
            "stiffness": (0.5 * left, SPRING_Y),
            "damping": (0.3 * left, DAMPER_Y),
        }

    # ------------------------------------------------------------- the hand
    def grip_at(self, pos: QtCore.QPointF) -> str | None:
        for name, (gx, gy) in self._grip_spots().items():
            if (pos.x() - gx) ** 2 + (pos.y() - gy) ** 2 <= (1.5 * GRIP_R) ** 2:
                return name
        return None

    def grab_at(self, pos: QtCore.QPointF) -> object | None:
        grip = self.grip_at(pos)
        if grip is not None:
            return grip
        s = self._side
        on_mass = abs(pos.x() - self._centre) <= s / 2 and abs(pos.y()) <= s / 2
        return True if on_mass else None

    def _take(self, what: object, pos: QtCore.QPointF) -> None:
        if isinstance(what, str):
            self._grip_from = pos
            self.grip_taken.emit(what)
            return
        self._grab_dx = pos.x() - self._centre
        self.grabbed.emit()
        self._pull(pos)

    def _pull(self, pos: QtCore.QPointF) -> None:
        what = self.held
        if isinstance(what, str):
            start = self._grip_from
            assert start is not None, "a grip's drag starts where it was taken"
            dx, dy = pos.x() - start.x(), pos.y() - start.y()
            self.grip_moved.emit(what, dx + dy if what == "mass" else dy)
            return
        offset = float(np.clip(pos.x() - self._grab_dx - REST_X, -MAX_DRAG, MAX_DRAG))
        self.pulled.emit(offset / self.gain)

    def _let_go(self, what: object) -> None:
        if isinstance(what, str):
            self._grip_from = None
            self.grip_released.emit(what)
            return
        self.let_go.emit()

    def start_pluck(self, x: float) -> None:
        """Ready the scale for a pluck from x (m): room to pull it a full drag, about 20 mm, or past where it is."""
        self.gain = min(self.gain, PLUCK_GAIN, 0.7 * MAX_DRAG / max(abs(x), 1e-9))

    def press_point(self, name: str) -> QtCore.QPointF:
        """Where a grip (or "mass", the body itself: `body=True`) is, in view coordinates (tests drag them)."""
        gx, gy = self._grip_spots()[name]
        return QtCore.QPointF(gx, gy)

    def body_point(self) -> QtCore.QPointF:
        return QtCore.QPointF(self._centre - 0.25 * self._side, -0.25 * self._side)
