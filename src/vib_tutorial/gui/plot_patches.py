"""Two changes to how every pyqtgraph plot draws, installed once by make_app (`install`).

Grids. pyqtgraph draws a plot's grid as its axes' ticks stretched across the plot, in the
foreground colour at a low alpha, and above the data (the axes sit at z 0.5, over the ViewBox).
Blending those long translucent lines is what a plot that rescales every frame spent its paint
on: on the Duffing page's phase plane and strip chart, ~35 ms a frame, against ~5 ms with the
same lines drawn opaque. So a grid axis draws them in the colour they would have come out over
the background, and goes behind the ViewBox, so that the lines pass under the curves, not over.

Titles. A LabelItem's minimum width is its text's, so a long plot title (the Virtual modal
test's FRF says what each line style means) made its plot wider than a 1280 px window's pane
leaves it, and the layout overflowed its view: the plot's right side and legend were cut off.
A level, centred label (a title, not a legend's entry) now wraps to the width it's given, centred, and only asks for TITLE_MIN_WIDTH.

Scale. The Projector theme draws every line wider and every marker larger (`Theme.line_scale`,
`marker_scale`): `mkPen` (pyqtgraph's own, which its items call too) scales the width it makes, and a
scatter's `setSize` the size it's given (its `addPoints` and `setData` set sizes through it). Pens are made when a page draws or takes a
theme (`apply_theme`), so a theme change rescales them; at 1 nothing changes.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui

from . import style

GRID_Z = -200.0  # below the ViewBox (z -100): the grid goes under the data
AXIS_Z = 0.5  # pyqtgraph's own
TITLE_MIN_WIDTH = 80.0  # px a title asks for at least; narrower than its text, it wraps


def _opaque_grid_specs(self, p: QtGui.QPainter):
    specs = _generate_draw_specs(self, p)
    if specs is None or self.grid is False:
        return specs
    axis_spec, tick_specs, text_specs = specs
    bg = QtGui.QColor(style.colors.background)
    blended: dict[tuple[int, int], QtGui.QPen] = {}  # (pen id, alpha) -> opaque pen; the ticks share a few
    out = []
    for pen, p1, p2 in tick_specs:
        c = pen.color()
        a = c.alphaF()
        if a < 1.0:
            key = (id(pen), c.alpha())
            if key not in blended:
                opaque = QtGui.QPen(pen)
                opaque.setColor(
                    QtGui.QColor.fromRgbF(
                        c.redF() * a + bg.redF() * (1 - a),
                        c.greenF() * a + bg.greenF() * (1 - a),
                        c.blueF() * a + bg.blueF() * (1 - a),
                    )
                )
                blended[key] = opaque
            pen = blended[key]
        out.append((pen, p1, p2))
    return axis_spec, out, text_specs


def _set_grid(self, grid) -> None:
    _set_grid_original(self, grid)
    self.setZValue(AXIS_Z if self.grid is False else GRID_Z)


def _wraps(label) -> bool:
    """A level, centred label: a title. Not a legend's entry (left-justified), which a legend sizes itself to.

    Only the label's own state is read: this runs as items are torn down too, where asking for a parent
    that is being deleted crashed."""
    return getattr(label, "angle", 0) == 0 and label.opts.get("justify", "center") == "center"


def _wrap_to_width(self, ev) -> None:
    if _wraps(self):
        item = self.item
        item.setTextWidth(-1)  # its natural width
        width = self.rect().width()
        if width > 0 and item.boundingRect().width() > width + 0.5:
            item.setTextWidth(max(width, TITLE_MIN_WIDTH))
            option = item.document().defaultTextOption()
            option.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
            item.document().setDefaultTextOption(option)
    _resize_original(self, ev)


def _update_min(self) -> None:
    _update_min_original(self)
    if _wraps(self) and self.minimumWidth() > TITLE_MIN_WIDTH:
        self.setMinimumWidth(TITLE_MIN_WIDTH)
        self._sizeHint[QtCore.Qt.SizeHint.MinimumSize] = (TITLE_MIN_WIDTH, self.minimumHeight())


_generate_draw_specs = pg.AxisItem.generateDrawSpecs
_set_grid_original = pg.AxisItem.setGrid
_resize_original = pg.LabelItem.resizeEvent
_update_min_original = pg.LabelItem.updateMin


_mk_pen = pg.functions.mkPen
_set_size = pg.ScatterPlotItem.setSize


_axis_pen = pg.AxisItem.pen
_unscaled = [0]  # inside an axis's pen: left as it is


def _scaled_pen(*args, **kwargs) -> QtGui.QPen:
    """`mkPen`, its width times the theme's `line_scale` (a cosmetic 0 counts as 1 px). A pen made from a pen
    (an item's `setPen(pen)` copies it) is already scaled, and the axes and their grid keep their width."""
    pen = _mk_pen(*args, **kwargs)
    scale = style.current().line_scale
    if (
        scale != 1.0
        and not _unscaled[0]
        and not (args and isinstance(args[0], QtGui.QPen))
        and pen.style() != QtCore.Qt.PenStyle.NoPen
    ):
        pen.setWidthF(max(pen.widthF(), 1.0) * scale)
    return pen


def _plain_axis_pen(self) -> QtGui.QPen:
    _unscaled[0] += 1
    try:
        return _axis_pen(self)
    finally:
        _unscaled[0] -= 1


def _scaled(size):
    scale = style.current().marker_scale
    if scale == 1.0 or size is None:
        return size
    return np.asarray(size, dtype=float) * scale if np.ndim(size) else float(size) * scale


def _scaled_set_size(self, size, *args, **kargs):
    return _set_size(self, _scaled(size), *args, **kargs)


def install() -> None:
    """Grids opaque and under the data, titles that wrap, pens and markers to the theme's scale; see the module's
    docstring."""
    pg.mkPen = pg.functions.mkPen = _scaled_pen
    pg.AxisItem.pen = _plain_axis_pen
    pg.ScatterPlotItem.setSize = _scaled_set_size
    pg.AxisItem.generateDrawSpecs = _opaque_grid_specs
    pg.AxisItem.setGrid = _set_grid
    pg.LabelItem.resizeEvent = _wrap_to_width
    pg.LabelItem.updateMin = _update_min
