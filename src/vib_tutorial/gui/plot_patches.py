"""Changes to how every pyqtgraph plot draws, installed once by make_app (`install`), and `hold_still`.

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

Still curves (`hold_still(plot)`). Something moving over a plot (a time cursor, a playhead, a body's marker)
dirties the strip it passes, and Qt then paints every curve under it again: the whole path, stroked and only
then clipped. The Single oscillator's three 3,000-point responses took 37 ms a frame that way while they played,
the Frame builder's two dashed ones 16 ms. A held plot's curves (those it has, and those added later: `plot()`,
`addItem`) keep a pixmap of themselves (`DeviceCoordinateCache`), and the strip is copied from it: 2 ms. Hold
only a plot whose curves stay put while something moves: a curve redrawn every frame redraws its pixmap too (the
Simulation page's strip chart, cached, went 4.8 → 7.9 ms). `Playback.add_cursor` holds its plot. The pixmaps
live in Qt's QPixmapCache, 10 MB unless told otherwise, and at a Retina display's 2× each is four times the
size: a 1700 px window's Single oscillator holds 10.6 MB of them, so the cache dropped them as fast as it made
them and every frame drew the curves again, and then copied them too (20 ms a frame on the Mac, 36.8 fps;
#409 N-2). `size_pixmap_cache` gives it `PIXMAP_CACHE_MB`, room for a few pages' held plots at 2× on a large
screen; make_app calls it once the app exists, since making the QApplication sets the limit back to 10 MB.

Wide lines. Qt strokes a polyline wider than 1 px as one outline, and a curve that zigzags (a response with a
stiff mode in it, a noisy signal) makes that outline slow: the Numerical integration page's 2 px responses took
~35 ms a paint, against ~1 ms at 1 px. pyqtgraph draws a solid, opaque, wide pen as separate segments instead,
which is fast, but only without antialiasing (where segments overlap, a translucent pen's joints show darker);
the app antialiases, so a solid opaque pen wider than 1 px is drawn in segments here too (`_segmented`): its
overlaps are the same colour. Only with flat caps, which `mkPen` gives (a polyline's caps are only its two ends):
square or round ones overlap at every joint, and the line came out visibly heavier. A dashed pen still strokes one polyline (its dashes run along the whole curve), so
keep a dense dashed curve at 1 px: the integration page's dashed closed form at 2 px took 220 ms.

Empty legends. A legend with no entries yet (a page that adds them once it has something to show) drew an empty
box over its plot: it draws nothing until it has one (`_paint_legend`).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import style

PIXMAP_CACHE_MB = 96  # QPixmapCache's limit: held curves' pixmaps (see the module's docstring)

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
_plot_add_item = pg.PlotItem.addItem


_mk_pen = pg.functions.mkPen
_set_size = pg.ScatterPlotItem.setSize


_axis_pen = pg.AxisItem.pen
_unscaled = [0]  # inside an axis's pen: left as it is


def _scaled_pen(*args, **kwargs) -> QtGui.QPen:
    """`mkPen`, its width times the theme's `line_scale` (a cosmetic 0 counts as 1 px), with flat caps (a wide
    curve is drawn in segments: `_segmented`). A pen made from a pen (an item's `setPen(pen)` copies it) is
    already scaled, and the axes and their grid keep their width."""
    pen = _mk_pen(*args, **kwargs)
    if not (args and isinstance(args[0], QtGui.QPen)):
        pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
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


def _hold(item) -> None:
    curve = item.curve if isinstance(item, pg.PlotDataItem) else item
    if isinstance(curve, pg.PlotCurveItem):
        curve.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)


def hold_still(plot) -> None:
    """Draw `plot`'s curves from a pixmap of each (see the module's docstring), those added later too."""
    item = plot.getPlotItem() if isinstance(plot, pg.PlotWidget) else plot
    item.held_still = True
    for child in item.items:
        _hold(child)


def _add_item(self, item, *args, **kargs) -> None:
    _plot_add_item(self, item, *args, **kargs)
    if getattr(self, "held_still", False):
        _hold(item)


def _segmented(self, pen: QtGui.QPen) -> bool:
    """Draw the curve as separate segments (fast) for a solid, opaque, flat-capped pen wider than 1 px, antialiased
    or not (square or round caps overlap at each joint, and the line looks heavier: `mkPen`'s are flat)."""
    mode = self.opts["segmentedLineMode"]
    if mode != "auto":
        return mode == "on"
    return (
        pen.widthF() > 1.0
        and pen.capStyle() == QtCore.Qt.PenCapStyle.FlatCap
        and pen.style() == QtCore.Qt.PenStyle.SolidLine
        and pen.isSolid()
        and pen.color().alphaF() == 1.0
    )


_legend_paint = pg.LegendItem.paint


def _paint_legend(self, p, *args) -> None:
    if self.items:  # an empty legend's frame is a box over the data, saying nothing
        _legend_paint(self, p, *args)


def install() -> None:
    """Grids opaque and under the data, titles that wrap, pens and markers to the theme's scale, wide lines in
    segments; see the module's docstring."""
    pg.mkPen = pg.functions.mkPen = _scaled_pen
    pg.AxisItem.pen = _plain_axis_pen
    pg.ScatterPlotItem.setSize = _scaled_set_size
    pg.AxisItem.generateDrawSpecs = _opaque_grid_specs
    pg.AxisItem.setGrid = _set_grid
    pg.LabelItem.resizeEvent = _wrap_to_width
    pg.LabelItem.updateMin = _update_min
    pg.PlotItem.addItem = _add_item
    pg.PlotCurveItem._shouldUseDrawLineSegments = _segmented
    pg.LegendItem.paint = _paint_legend


def size_pixmap_cache() -> None:
    """Room for held curves' pixmaps: called once the QApplication exists, which sets the cache back to its 10 MB
    as it's made (set before it, #417's limit never held)."""
    QtGui.QPixmapCache.setCacheLimit(PIXMAP_CACHE_MB * 1024)
