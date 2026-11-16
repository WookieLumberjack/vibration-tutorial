"""Every page's long curves follow `plots.py`'s policy (#316 E2): one whose x increases, more than
`PER_PIXEL` points a pixel of its plot, is decimated (`plots.decimate`: each pixel column's min and max)."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from vib_tutorial.gui.main_window import LAZY_PAGES  # noqa: E402

pytestmark = pytest.mark.slow  # builds every page

PER_PIXEL = 10


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


@pytest.mark.parametrize("name", ["sim_page", *(name for name, _ in LAZY_PAGES)])
def test_a_long_curve_whose_x_increases_is_decimated(main_window, app, name):
    import pyqtgraph as pg

    w = main_window
    w.show_page(name)
    page = w.pages.currentWidget()
    app.processEvents()
    for tabs in page.findChildren(QtWidgets.QTabWidget):  # every tab drawn
        for i in range(tabs.count()):
            tabs.setCurrentIndex(i)
            app.processEvents()
    long = []
    for plot in page.findChildren(pg.PlotWidget):
        width = max(plot.getPlotItem().getViewBox().width(), 1.0)
        for item in plot.getPlotItem().listDataItems():
            if not isinstance(item, pg.PlotDataItem):
                continue  # a scatter: points, not a line
            x, _ = item.getOriginalDataset()
            if x is None or len(x) <= PER_PIXEL * width or item.opts["autoDownsample"]:
                continue
            finite = x[np.isfinite(x)]
            if np.all(np.diff(finite) >= 0):
                long.append(f"{plot.getPlotItem().getAxis('left').labelText!r}: {len(x)} points over {width:.0f} px")
    assert long == [], "decimate these plots (plots.decimate)"
