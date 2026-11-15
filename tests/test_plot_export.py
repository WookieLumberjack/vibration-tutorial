"""`gui/plot_export.py`: a plot's right-click menu copies or saves its data as CSV, or saves it as a PNG (#218 U3)."""

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


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


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
