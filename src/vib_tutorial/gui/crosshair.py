"""A crosshair that follows the pointer over a plot, with the values under it, and the same instant on the plots
that share its x axis (a time cursor across a page's linked time plots).

`install(page)` gives every plot under the page one, as `plot_export.install` gives the menu: plots with a visible
x axis, in views that don't handle the mouse themselves (the drawings: the chain, the rotor, an editing canvas).
Nothing is drawn until the pointer is over a plot; leaving the view hides it.

Over a plot: a faint vertical and horizontal line through the pointer and, by it, the x and y it reads, in the
axes' units (on a log axis, the value, not its log). Every other plot on the page whose x axis is linked to this
one (`setXLink`, either way, or through a third) shows the vertical line at the same x.

Working on it: the lines and the readout take no mouse, so drags and clicks pass to the plot's own items; they're
added with `ignoreBounds` (no plot ranges to them) and put back if a redraw `clear()`ed them. `tests/test_crosshair.py`
moves the pointer over linked plots.
"""

from __future__ import annotations

import math
import weakref

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..view_models.text import number as sig
from . import plot_export
from .style import colors

INSTALLED = "crosshair"  # the view property that marks one done


class _Hair:
    """One plot's lines and readout."""

    def __init__(self, plot: pg.PlotItem) -> None:
        self.plot = plot
        self.v = pg.InfiniteLine(angle=90, movable=False)
        self.h = pg.InfiniteLine(angle=0, movable=False)
        self.label = pg.TextItem(anchor=(0, 1))
        for item in (self.v, self.h, self.label):
            item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
            item.setAcceptHoverEvents(False)
            item.setZValue(30)
            item.hide()

    def _in_plot(self) -> None:
        for item in (self.v, self.h, self.label):
            if item.scene() is None:  # taken out by the page's clear()
                self.plot.addItem(item, ignoreBounds=True)

    def show_at(self, vx: float, vy: float | None) -> None:
        """The lines at the view's (x, y) (log10 on a log axis); a vertical line alone when y is None."""
        self._in_plot()
        pen = pg.mkPen(colors.muted, width=1, style=QtCore.Qt.PenStyle.DashLine)
        self.v.setPen(pen)
        self.v.setPos(vx)
        self.v.show()
        if vy is None:
            self.h.hide()
            self.label.hide()
            return
        self.h.setPen(pen)
        self.h.setPos(vy)
        self.h.show()
        self.label.setColor(colors.foreground)
        fill = pg.mkColor(colors.background)
        fill.setAlpha(colors.legend_alpha)  # legible over the curves, as a legend is
        self.label.fill = pg.mkBrush(fill)
        self.label.setText(f"{_value(self.plot, 'bottom', vx)}, {_value(self.plot, 'left', vy)}")
        self.label.setPos(vx, vy)
        self.label.show()

    def hide(self) -> None:
        for item in (self.v, self.h, self.label):
            item.hide()


def _value(plot: pg.PlotItem, side: str, v: float) -> str:
    """An axis's value at view coordinate v, in its units."""
    log = plot.getAxis(side).logMode
    x = 10.0**v if log else v
    units = plot.getAxis(side).labelUnits or ""
    if not math.isfinite(x):
        return "–"
    return pg.siFormat(x, precision=4, suffix=units) if units else f"{sig(x, 4)}"


def _root(vb: pg.ViewBox) -> pg.ViewBox:
    """The view at the end of a chain of x links (two plots linked to the same one share it)."""
    seen = {id(vb)}
    while (other := vb.linkedView(pg.ViewBox.XAxis)) is not None and id(other) not in seen:
        seen.add(id(other))
        vb = other
    return vb


class _Hook(QtCore.QObject):
    """A view's mouse moves passed on to its page's crosshairs: a child of the view (it goes with it), holding
    the crosshairs weakly and finding the view as its parent, so no closure keeps either alive."""

    def __init__(self, view: pg.GraphicsView, owner: Crosshairs) -> None:
        super().__init__(view)
        self._owner = weakref.ref(owner)
        view.scene().sigMouseMoved.connect(self.moved)

    def moved(self, pos: QtCore.QPointF) -> None:
        owner, view = self._owner(), self.parent()
        if owner is not None and isinstance(view, pg.GraphicsView):
            owner.moved(view, pos)


class Crosshairs(QtCore.QObject):
    """A page's crosshairs (see the module's docstring)."""

    def __init__(self, page: QtWidgets.QWidget) -> None:
        super().__init__(page)
        self.hairs: list[_Hair] = []
        for view in page.findChildren(pg.GraphicsView):
            if view.property(INSTALLED) or plot_export.handles_mouse(view):
                continue
            plots = [p for p in plot_export.plots(view) if p.getAxis("bottom").isVisible()]
            if not plots:
                continue
            view.setProperty(INSTALLED, True)
            self.hairs += [_Hair(p) for p in plots]
            _Hook(view, self)
            view.viewport().installEventFilter(self)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Leave:
            self.hide()
        return False

    def hide(self) -> None:
        for hair in self.hairs:
            hair.hide()

    def moved(self, view: pg.GraphicsView, pos: QtCore.QPointF) -> None:
        """The pointer at scene position `pos` in `view`."""
        over = next(
            (
                h
                for h in self.hairs
                if h.plot.scene() is view.scene() and h.plot.isVisible() and h.plot.vb.sceneBoundingRect().contains(pos)
            ),
            None,
        )
        if over is None:
            self.hide()
            return
        at = over.plot.vb.mapSceneToView(pos)
        root = _root(over.plot.vb)
        for hair in self.hairs:
            if hair is over:
                hair.show_at(at.x(), at.y())
            elif hair.plot.isVisible() and _root(hair.plot.vb) is root:
                hair.show_at(at.x(), None)  # the same x on a plot that shares it
            else:
                hair.hide()


def install(page: QtWidgets.QWidget) -> Crosshairs:
    return Crosshairs(page)
