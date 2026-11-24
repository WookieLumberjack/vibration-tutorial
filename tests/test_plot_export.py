"""`gui/plot_export.py`: a plot's right-click menu copies or saves its data as CSV, saves it as a PNG (#218 U3), or
holds its curves to compare (#316 B6)."""

from __future__ import annotations

import csv
import io
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
import pyqtgraph as pg  # noqa: E402
from PySide6 import QtCore  # noqa: E402

from vib_tutorial.gui import plot_export  # noqa: E402
from vib_tutorial.gui.lesson import plot  # noqa: E402


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


def test_a_plots_curves_as_csv(app, monkeypatch):
    p = plot("Amplitude", "m", "Frequency", "Hz", log_y=True)
    p.addLegend()
    p.plot([1.0, 2.0, 3.0], [0.1, 0.2, np.nan], name="mass 1")  # log-scaled on screen: the CSV has the values
    p.plot([0.5, 1.5], [3.0, 4.0])
    rows = list(csv.reader(io.StringIO(plot_export.csv_text(p))))
    assert rows == [
        ["Frequency (Hz)", "Amplitude (m), mass 1", "Frequency (Hz)", "Amplitude (m), curve 2"],
        ["1", "0.1", "0.5", "3"],
        ["2", "0.2", "1.5", "4"],
    ]
    plot_export.copy_data(p)
    assert QtWidgets.QApplication.clipboard().text() == plot_export.csv_text(p)
    actions = {a.text(): a.isEnabled() for a in plot_export.menu(p).actions()}
    assert actions == {
        "Copy data as CSV": True,
        "Save data as CSV…": True,
        "Save image as PNG…": True,
        "Show all": True,
        "Hold these curves": True,
        "Clear held curves": False,
    }
    shown = []
    box = QtWidgets.QWidget()
    p.setParent(box)
    plot_export.install(box)
    monkeypatch.setattr(plot_export, "popup", lambda menu, _: shown.append([a.text() for a in menu.actions()]))
    p.customContextMenuRequested.emit(QtCore.QPoint(5, 5))  # what a right-click sends
    assert shown == [list(actions)] and not p.getPlotItem().menuEnabled(), "ours, in place of pyqtgraph's"
    empty = plot("x", "m", "t", "s")
    assert not plot_export.menu(empty).actions()[0].isEnabled(), "nothing to copy"
    box.close()
    empty.close()


def test_every_pages_plots_have_the_menu(main_window, app):
    w = main_window
    for i in range(w.pages.count()):
        w.pages.setCurrentIndex(i)  # builds each page
    app.processEvents()
    missing = []
    for i in range(w.pages.count()):
        page = w.pages.widget(i)
        for view in page.findChildren(pg.GraphicsView):
            if plot_export.handles_mouse(view):
                assert view.contextMenuPolicy() != QtCore.Qt.ContextMenuPolicy.CustomContextMenu
                continue
            if not view.property(plot_export.MENU):
                missing.append(f"{w.page_name(page)}: {type(view).__name__} {view.accessibleName()!r}")
    assert not missing, "\n".join(missing)
    assert plot_export.handles_mouse(w.frame_page.canvas), "a right-click edits the frame: no menu"


def test_holding_a_plots_curves_keeps_grey_copies_behind_them(app):
    p = plot("Amplitude", "m", "Frequency", "Hz", log_y=True)
    p.addLegend()
    live = p.plot([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], name="mass 1")
    dots = p.plot([1.5], [0.15], pen=None, symbol="o")
    plot_export.hold(p)
    held = plot_export.held(p)
    assert len(held) == 2 and all(h.zValue() < live.zValue() for h in held)
    assert held[0].line and not held[1].line and held[1].opts["symbol"] == "o"  # a marker held as markers
    assert len(p.getPlotItem().legend.items) == 1  # no legend entry of their own
    live.setData([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])  # the page redraws: the held copy stays as it was
    assert list(held[0].yData) == [0.1, 0.2, 0.3]
    names = next(csv.reader(io.StringIO(plot_export.csv_text(p))))
    assert "Amplitude (m), held: mass 1" in names

    p.clear()  # a page's redraw that clears the plot keeps them
    assert len(plot_export.held(p)) == 2 and live not in p.getPlotItem().items
    live, dots = p.plot([1.0, 2.0], [0.5, 0.6], name="mass 1"), p.plot([1.5], [0.15], pen=None, symbol="o")
    plot_export.hold(p)  # holding again adds the live curves, never the held ones
    assert len(plot_export.held(p)) == 4
    labels = {a.text(): a.isEnabled() for a in plot_export.menu(p).actions()}
    assert labels["Hold these curves"] and labels["Clear held curves"]
    plot_export.clear_held(p)
    assert not plot_export.held(p) and live in p.getPlotItem().items and dots in p.getPlotItem().items
    assert not {a.text(): a.isEnabled() for a in plot_export.menu(p).actions()}["Clear held curves"]


@pytest.mark.parametrize(
    "name", ["isolation_page", "quarter_car_page", "damping_page", "signals_page", "continuum_page"]
)
def test_held_curves_stay_as_the_page_redraws_and_defaults_clears_them(main_window, app, name):
    """The audit's pages (#316 B6): hold every plot after one trial, pick another, and the held curves are still
    there to compare against; Defaults takes them off."""
    w = main_window
    page = getattr(w, name)
    w.pages.setCurrentWidget(page)

    def trial(i: int) -> None:  # from code (a prediction isn't asked), or the Continuum's preset
        if hasattr(page, "load_trial"):
            page.load_trial(i)
        else:
            from vib_tutorial.gui.continuum import PRESETS

            page.load_preset(PRESETS[i].name)

    trial(0)
    for _ in range(3):
        app.processEvents()
    views = [v for v in page.findChildren(pg.GraphicsView) if v.property(plot_export.MENU)]
    for v in views:
        plot_export.hold(v)
    before = {v: [(h, np.array(h.yData)) for h in plot_export.held(v)] for v in views}
    assert sum(len(b) for b in before.values()) > 0
    trial(1)
    for _ in range(3):
        app.processEvents()
    changed = False
    for v in views:
        for h, y in before[v]:
            assert h in plot_export.held(v) and np.array_equal(np.array(h.yData), y, equal_nan=True)
        for item in plot_export.plots(v):
            for curve in plot_export._live(item):
                if not any(
                    len(h.yData) == len(curve.yData) and np.allclose(h.yData, curve.yData, equal_nan=True)
                    for h, _ in before[v]
                ):
                    changed = True
    assert changed  # the trial moved a curve off its held copy
    w.restore_defaults()
    assert not any(plot_export.held(v) for v in views)


def test_h_holds_every_plot_the_page_shows(main_window, app):
    """H (and the run bar's Hold curves) holds the curves of every plot the page shows, not its hidden tabs';
    Shift+H (Clear held) takes them off (#409 I-11)."""
    import pyqtgraph as pg

    w = main_window
    w.show_page("pendulum_page")
    p = w.pendulum_page
    app.processEvents()
    views = [v for v in p.findChildren(pg.GraphicsView) if v.property(plot_export.MENU)]
    showing = [v for v in views if v.isVisible() and plot_export.curves(v)]
    hidden = [v for v in views if not v.isVisible()]
    assert showing and hidden
    w.run_bar.hold.click()
    assert all(plot_export.held(v) for v in showing) and not any(plot_export.held(v) for v in hidden)
    w.run_bar.unhold.click()
    assert not any(plot_export.held(v) for v in views)
    w.hold_curves()  # H
    assert all(plot_export.held(v) for v in showing)
    w.clear_held()  # Shift+H
    assert not any(plot_export.held(v) for v in views)
