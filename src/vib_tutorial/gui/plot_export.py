"""A plot's right-click menu: copy or save its curves' data as CSV, save it as a PNG, show all of it (#218 U3), or
hold its curves to compare (#316 B6).

`install(page)` runs when `MainWindow` builds a page. It gives every pyqtgraph
view on the page this menu in place of pyqtgraph's own (which offers a plot's axes and export settings, more
than a reader needs), except a view that handles the mouse itself (`mousePressEvent`: the frame and rotor
builders' canvases, where a right-click edits). A view's curves are its `PlotDataItem`s and `PlotCurveItem`s
with data, in its plots' order; CSV has an x and a y column for each, named after the plot's axes and the
curve's legend name, as the curves needn't share their x.

*Hold these curves* (or H, or the run bar's *Hold curves*, for every plot the page shows: `hold_showing`) copies
every curve the view shows into a grey dashed one behind them (`hold`: a `Held`, a
plain `PlotDataItem` the page doesn't know of, so it stays as the page redraws its own), to compare a change
against: hold, change a control, see both. Holding again adds another; *Clear held curves* (`clear_held`)
takes them all off, as does *Defaults* (`MainWindow.restore_defaults`). Held curves take the theme's grey when it
changes (`restyle`, from `MainWindow._restyle`), go in the CSV named "held: …", and are never held again. A page
whose redraw clears its plot (`PlotItem.clear`, the Continuum's) gets them back: holding gives that plot a
`clear` that puts them back after (`KeepHeld`; its PlotWidget's too, which bound the plot's own when built).

Working on it: `tests/test_plot_export.py` checks the CSV and that every page's plots have the menu.
"""

from __future__ import annotations

import csv
import io
import re
import weakref
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import accessible, weak

MENU = "plot_export"  # the property marking a view that has the menu
FILE_NAME = re.compile(r"[^\w.-]+")


def plots(view: pg.GraphicsView) -> list[pg.PlotItem]:
    """The plots a view shows: a PlotWidget's one, a GraphicsLayoutWidget's every one."""
    if isinstance(view, pg.PlotWidget):
        return [view.getPlotItem()]
    if isinstance(view, pg.GraphicsLayoutWidget):
        return [item for item in view.ci.items if isinstance(item, pg.PlotItem)]
    return []


def _axis(item: pg.PlotItem, side: str) -> str:
    axis = item.getAxis(side)
    text = accessible.plain(axis.labelText or "")
    return f"{text} ({axis.labelUnits})" if text and axis.labelUnits else text


class Held(pg.PlotDataItem):
    """A curve held for comparison: a copy of one the page drew, in grey, behind the page's own."""

    def __init__(self, x: np.ndarray, y: np.ndarray, name: str, line: bool) -> None:
        super().__init__(x, y)
        self.held_name, self.line = name, line
        self.setZValue(-5)
        self.apply_theme()

    def apply_theme(self) -> None:
        from .style import colors

        if self.line:
            self.setPen(pg.mkPen(colors.grey, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
            self.setSymbol(None)
        else:  # markers held as hollow ones
            self.setPen(None)
            self.setSymbol("o")
            self.setSymbolSize(6)
            self.setSymbolPen(pg.mkPen(colors.grey, width=1))
            self.setSymbolBrush(None)


class KeepHeld:
    """A plot's `clear` that keeps its held curves (set on the plot by `hold`; weak, so it holds nothing alive)."""

    def __init__(self, item: pg.PlotItem) -> None:
        self.item = weakref.ref(item)

    def __call__(self) -> None:
        item = self.item()
        if item is None:
            return
        kept = [i for i in item.items if isinstance(i, Held)]
        type(item).clear(item)
        for curve in kept:
            item.addItem(curve)


def _live(item: pg.PlotItem) -> list[pg.PlotDataItem | pg.PlotCurveItem]:
    """A plot's curves the page drew (not held ones), shown, with data."""
    return [
        i
        for i in item.items
        if isinstance(i, pg.PlotDataItem | pg.PlotCurveItem)
        and not isinstance(i, Held)
        and i.isVisible()
        and i.xData is not None
        and i.yData is not None
        and len(i.xData)
    ]


def held(view: pg.GraphicsView) -> list[Held]:
    return [i for item in plots(view) for i in item.items if isinstance(i, Held)]


def hold(view: pg.GraphicsView) -> None:
    """Copy every curve the view shows into a held one (see the module's docstring)."""
    for item in plots(view):
        for k, curve in enumerate(_live(item)):
            x, y = np.array(curve.xData, dtype=float), np.array(curve.yData, dtype=float)
            name = curve.name() if isinstance(curve, pg.PlotDataItem) and curve.name() else f"curve {k + 1}"
            line = not (isinstance(curve, pg.PlotDataItem) and curve.opts["pen"] is None)
            item.addItem(Held(x, y, accessible.plain(name), line))
        if not isinstance(item.__dict__.get("clear"), KeepHeld):
            item.clear = KeepHeld(item)  # a page's redraw clearing the plot keeps them
            if isinstance(view, pg.PlotWidget):
                view.clear = item.clear  # (a PlotWidget binds its plot's methods when built)


def hold_showing(root: QtWidgets.QWidget) -> int:
    """Hold the curves of every plot under `root` that shows (H, the run bar's *Hold curves*: #409 I-11); how many
    views had curves to hold."""
    views = [
        v
        for v in root.findChildren(pg.GraphicsView)
        if v.property(MENU) and v.isVisible() and any(_live(item) for item in plots(v))
    ]
    for view in views:
        hold(view)
    return len(views)


def clear_held(view: pg.GraphicsView) -> None:
    for item in plots(view):
        for curve in [i for i in item.items if isinstance(i, Held)]:
            item.removeItem(curve)


def clear_all_held(root: QtWidgets.QWidget) -> None:
    """Every view's held curves under `root` taken off (Defaults)."""
    for view in root.findChildren(pg.GraphicsView):
        clear_held(view)


def restyle(root: QtWidgets.QWidget) -> None:
    """Held curves in the theme's grey (`MainWindow._restyle`)."""
    for view in root.findChildren(pg.GraphicsView):
        for curve in held(view):
            curve.apply_theme()


def curves(view: pg.GraphicsView) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Each curve's (x name, y name, x, y), finite points only, in the order the plots hold them."""
    out = []
    for p, item in enumerate(plots(view)):
        x_name = _axis(item, "bottom") or "x"
        y_axis = _axis(item, "left") or "y"
        title = accessible.plain(item.titleLabel.text or "")
        for k, curve in enumerate(i for i in item.items if isinstance(i, pg.PlotDataItem | pg.PlotCurveItem)):
            if not curve.isVisible():
                continue
            x, y = curve.xData, curve.yData  # as given: not log-scaled or thinned for the screen
            if x is None or y is None or not len(x):
                continue
            x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            keep = np.isfinite(x) & np.isfinite(y)
            name = curve.name() if isinstance(curve, pg.PlotDataItem) and curve.name() else f"curve {k + 1}"
            if isinstance(curve, Held):
                name = f"held: {curve.held_name}"
            where = "" if len(plots(view)) == 1 else f"{title or f'plot {p + 1}'}: "
            out.append((f"{where}{x_name}", f"{where}{y_axis}, {accessible.plain(name)}", x[keep], y[keep]))
    return out


def csv_text(view: pg.GraphicsView) -> str:
    """The view's curves as CSV: an x and a y column for each, as long as the longest (the rest left blank)."""
    found = curves(view)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([name for x_name, y_name, _, _ in found for name in (x_name, y_name)])
    for r in range(max((len(x) for _, _, x, _ in found), default=0)):
        writer.writerow([f"{v[r]:.10g}" if r < len(v) else "" for _, _, x, y in found for v in (x, y)])
    return buffer.getvalue()


def _suggested(view: pg.GraphicsView, suffix: str) -> str:
    name = FILE_NAME.sub("_", view.accessibleName() or "plot").strip("_")[:60] or "plot"
    return str(
        Path(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.DocumentsLocation))
        / f"{name}{suffix}"
    )


def copy_data(view: pg.GraphicsView) -> None:
    QtGui.QGuiApplication.clipboard().setText(csv_text(view))


def save_data(view: pg.GraphicsView) -> None:
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        view, "Save the plot's data", _suggested(view, ".csv"), "CSV (*.csv)"
    )
    if path:
        Path(path).write_text(csv_text(view), encoding="utf-8")


def save_image(view: pg.GraphicsView) -> None:
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        view, "Save the plot as an image", _suggested(view, ".png"), "PNG (*.png)"
    )
    if path:
        view.grab().save(path, "PNG")


def show_all(view: pg.GraphicsView) -> None:
    for item in plots(view):
        item.enableAutoRange()


def menu(view: pg.GraphicsView) -> QtWidgets.QMenu:
    """The view's menu (its data actions only where it has curves)."""
    m = QtWidgets.QMenu(view)
    has_data = bool(curves(view))
    for text, act, on in (
        ("Copy data as CSV", copy_data, has_data),
        ("Save data as CSV…", save_data, has_data),
        ("Save image as PNG…", save_image, True),
        ("Show all", show_all, bool(plots(view))),
        ("Hold these curves", hold, any(_live(item) for item in plots(view))),
        ("Clear held curves", clear_held, bool(held(view))),
    ):
        action = m.addAction(text)
        action.setEnabled(on)
        action.triggered.connect(weak.slot(view, lambda view, _=False, act=act: act(view)))
    return m


def popup(m: QtWidgets.QMenu, at: QtCore.QPoint) -> None:
    """Show the menu at `at` until it's closed (a test replaces it)."""
    m.exec(at)


def _open(view: pg.GraphicsView, pos: QtCore.QPoint) -> None:
    m = menu(view)
    popup(m, view.mapToGlobal(pos))
    m.deleteLater()


def handles_mouse(view: pg.GraphicsView) -> bool:
    """Whether the view's class handles mouse presses itself (an editing canvas)."""
    return any(
        "mousePressEvent" in vars(cls) for cls in type(view).__mro__ if cls.__module__.startswith("vib_tutorial")
    )


def install(root: QtWidgets.QWidget) -> None:
    """Give every pyqtgraph view under `root` the menu (see the module's docstring)."""
    for view in root.findChildren(pg.GraphicsView):
        if view.property(MENU) or handles_mouse(view):
            continue
        for item in plots(view):
            item.setMenuEnabled(False)
        view.setProperty(MENU, True)
        view.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        view.customContextMenuRequested.connect(weak.slot(view, _open))
