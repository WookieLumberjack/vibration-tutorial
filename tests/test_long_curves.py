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


FRAMES = 60  # a second of a running page's frames, driven by hand
DT = 1 / 60


@pytest.mark.parametrize("name", ["sim_page", *(name for name, _ in LAZY_PAGES)])
def test_a_long_curve_whose_x_increases_is_decimated(main_window, app, name):
    """Each tab drawn, and run (its Start, its frame loops driven a second) where the page runs: a curve that grows
    while it runs is long only then (#409 P-15: the Gyroscope's thinned ones)."""
    import pyqtgraph as pg

    from vib_tutorial.gui import settle_plots
    from vib_tutorial.gui.frame_loop import FrameLoop

    w = main_window
    w.show_page(name)
    page = w.pages.currentWidget()
    app.processEvents()
    long: set[str] = set()
    tab_widgets = page.findChildren(QtWidgets.QTabWidget)
    for tabs in tab_widgets or [None]:
        for i in range(tabs.count() if tabs is not None else 1):
            if tabs is not None:
                tabs.setCurrentIndex(i)
            app.processEvents()
            transport = page.transport() if hasattr(page, "transport") else None
            if transport is not None and not transport.run.isChecked():
                transport.run.click()
            loops = [loop for loop in page.findChildren(FrameLoop) if loop._should_run()]
            for loop in loops:
                loop.enabled = False  # driven here, not by its timer
            for _ in range(FRAMES * bool(loops)):
                for loop in loops:
                    loop._call(loop._tick, DT)
                settle_plots(page)
            long |= set(_long_curves(page, pg))
            for loop in loops:
                loop.enabled = True
            if transport is not None and transport.run.isChecked():
                transport.run.click()
    assert sorted(long) == [], "decimate these plots (plots.decimate)"


def _long_curves(page, pg) -> list[str]:
    found = []
    for plot in page.findChildren(pg.PlotWidget):
        if not plot.isVisible():
            continue
        width = max(plot.getPlotItem().getViewBox().width(), 1.0)
        for item in plot.getPlotItem().listDataItems():
            if not isinstance(item, pg.PlotDataItem):
                continue  # a scatter: points, not a line
            x, _ = item.getOriginalDataset()
            if x is None or len(x) <= PER_PIXEL * width or item.opts["autoDownsample"]:
                continue
            finite = x[np.isfinite(x)]
            if np.all(np.diff(finite) >= 0):
                found.append(f"{plot.getPlotItem().getAxis('left').labelText!r}: {len(x)} points over {width:.0f} px")
    return found
