"""`gui/crosshair.py`: the crosshair over the plot under the pointer, its values, and the same x on the plots
linked to it (#316 B10)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
import pyqtgraph as pg  # noqa: E402
from PySide6 import QtCore, QtGui  # noqa: E402

from vib_tutorial.gui import crosshair  # noqa: E402
from vib_tutorial.gui.lesson import plot  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


def hover(view: pg.GraphicsView, scene_pt: QtCore.QPointF, app) -> None:
    """The pointer moved to `scene_pt` over `view`, with no button held."""
    vp = view.viewport()
    local = view.viewportTransform().map(scene_pt)
    ev = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        local,
        vp.mapToGlobal(local),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    limit = pg.getConfigOption("mouseRateLimit")
    pg.setConfigOption("mouseRateLimit", 0)  # these come at once (see test_drag's `drag`)
    try:
        QtWidgets.QApplication.sendEvent(vp, ev)
        app.processEvents()
    finally:
        pg.setConfigOption("mouseRateLimit", limit)


def at(p: pg.PlotWidget, x: float, y: float) -> QtCore.QPointF:
    return p.getPlotItem().vb.mapViewToScene(QtCore.QPointF(x, y))


def hair(hairs: crosshair.Crosshairs, p: pg.PlotWidget) -> crosshair._Hair:
    return next(h for h in hairs.hairs if h.plot is p.getPlotItem())


@pytest.fixture
def page(app):
    """Three plots: two sharing time, one on its own; the last a log axis."""
    w = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(w)
    top = plot("Displacement", "m", "Time", "s")
    below = plot("Force", "N", "Time", "s")
    below.setXLink(top)
    other = plot("Amplitude", "m", "Frequency", "Hz", log_x=True)
    for p in (top, below, other):
        lay.addWidget(p)
    top.plot([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
    below.plot([0.0, 1.0, 2.0], [0.0, -1.0, 0.0])
    other.plot([1.0, 10.0, 100.0], [1.0, 2.0, 1.0])
    for p in (top, below, other):
        p.getPlotItem().vb.autoRange(padding=0)
    w.resize(500, 700)
    w.show()
    app.processEvents()
    return SimpleNamespace(w=w, top=top, below=below, other=other, hairs=crosshair.install(w))


def test_the_plot_under_the_pointer_shows_both_lines_and_its_linked_plot_the_same_time(page, app):
    top, below, other, hairs = page.top, page.below, page.other, page.hairs
    hover(top, at(top, 1.5, 0.25), app)
    over, linked, apart = hair(hairs, top), hair(hairs, below), hair(hairs, other)
    assert over.v.isVisible() and over.h.isVisible() and over.label.isVisible()
    assert over.v.value() == pytest.approx(1.5, abs=0.01) and over.h.value() == pytest.approx(0.25, abs=0.01)
    assert over.label.textItem.toPlainText().replace(" ", " ").startswith("1.5")
    assert linked.v.isVisible() and linked.v.value() == pytest.approx(1.5, abs=0.01)
    assert not linked.h.isVisible() and not linked.label.isVisible()
    assert not apart.v.isVisible()

    hover(below, at(below, 0.5, -0.5), app)  # the link works both ways
    assert hair(hairs, top).v.value() == pytest.approx(0.5, abs=0.01) and not hair(hairs, top).h.isVisible()
    assert hair(hairs, below).h.isVisible()


def test_a_log_axis_reads_its_value_not_its_log(page, app):
    top, other, hairs = page.top, page.other, page.hairs
    hover(other, at(other, 1.0, 1.5), app)  # view x 1.0 is 10 Hz
    text = hair(hairs, other).label.textItem.toPlainText()
    assert text.startswith("10") and "Hz" in text, text
    assert not hair(hairs, top).v.isVisible()


def test_leaving_the_plot_hides_it(page, app):
    top, hairs = page.top, page.hairs
    hover(top, at(top, 1.0, 0.5), app)
    QtWidgets.QApplication.sendEvent(top.viewport(), QtCore.QEvent(QtCore.QEvent.Type.Leave))
    assert not any(item.isVisible() for h in hairs.hairs for item in (h.v, h.h, h.label))


def test_it_comes_back_after_a_redraw_clears_the_plot_and_takes_no_range(page, app):
    top, hairs = page.top, page.hairs
    top.clear()
    top.plot([0.0, 2.0], [0.0, 1.0])
    hover(top, at(top, 1.0, 0.5), app)
    assert hair(hairs, top).v.scene() is top.scene() and hair(hairs, top).v.isVisible()
    before = top.getPlotItem().vb.childrenBounds()
    hover(top, at(top, 1.9, 0.9), app)
    assert top.getPlotItem().vb.childrenBounds() == before


def test_installing_twice_adds_nothing(page):
    assert crosshair.install(page.w).hairs == []


def test_the_window_gives_every_pages_time_plots_one(main_window, app):
    """The Simulation page's linked time plots, and none on the chain's drawing."""
    w = main_window
    views = w.sim_page.findChildren(pg.GraphicsView)
    assert any(v.property(crosshair.INSTALLED) for v in views)
    from vib_tutorial.gui import plot_export

    assert not any(v.property(crosshair.INSTALLED) for v in views if plot_export.handles_mouse(v))
