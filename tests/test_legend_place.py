"""`gui/legend_place.py`: a legend moved to where it hides least of its plot's data."""

from __future__ import annotations

import time

import numpy as np

from vib_tutorial.gui import legend_place as lp


def test_cells_fill_the_lines_between_points_not_only_the_points():
    x, y = np.array([0.0, 99.0]), np.array([0.0, 99.0])  # a diagonal, corner to corner
    joined = lp.cells([x], [y], [True], (100.0, 100.0))
    dots = lp.cells([x], [y], [False], (100.0, 100.0))
    assert dots.sum() == 2
    assert joined.sum() >= 25 and joined[12, 12]  # every cell along the diagonal


def test_cells_skip_gaps_and_points_off_the_plot():
    x = np.array([10.0, np.nan, 50.0, 500.0])
    y = np.array([10.0, 10.0, 10.0, 10.0])
    grid = lp.cells([x], [y], [True], (100.0, 100.0))
    assert grid[2, 2] and grid[2, 12] and not grid[2, 6]  # no line across the NaN
    assert grid[2, 13:].all()  # the line on to the point off the right edge, up to the edge


def test_best_keeps_a_clear_spot_and_leaves_a_covered_one():
    grid = np.zeros((25, 25), bool)  # 100 px square
    grid[:8, :8] = True  # data in the top left
    legend = (30.0, 20.0)
    assert lp.best(grid, legend, (100.0, 100.0), 5.0, now=1) == 1  # top right: clear, it stays
    assert lp.best(grid, legend, (100.0, 100.0), 5.0, now=0) == 1  # top left: covered, the next clear one


def test_best_goes_above_the_plot_only_where_no_spot_is_nearly_clear():
    full = np.ones((25, 25), bool)  # data everywhere
    legend, plot = (30.0, 20.0), (100.0, 100.0)
    assert lp.best(full, legend, plot, 5.0, now=3) == lp.OUTSIDE
    assert lp.best(full, legend, plot, 5.0, now=lp.OUTSIDE) == lp.OUTSIDE  # back in only where clear
    assert lp.best(full, legend, plot, 5.0, now=3, outside=False) == 3  # nowhere better: it stays
    sliver = np.zeros((25, 25), bool)
    sliver[:, 3] = sliver[:, 12] = sliver[:, 22] = True  # a thin line through every spot
    pick = lp.best(sliver, legend, plot, 5.0, now=0)
    assert pick != lp.OUTSIDE and lp.coverage(sliver, lp.spot_rect(pick, legend, plot, 5.0)) <= lp.SLIVER
    clear = np.zeros((25, 25), bool)
    assert lp.best(clear, legend, plot, 5.0, now=lp.OUTSIDE) == 0  # clear again: back in


def test_a_plots_legend_moves_off_its_curve_and_stays_where_dragged(app, free_windows):
    import pyqtgraph as pg

    from vib_tutorial import gui
    from vib_tutorial.gui import theming

    view = pg.PlotWidget()
    view.resize(400, 300)
    plot = view.getPlotItem()
    legend = theming.add_legend(plot, offset=(5, 5))  # top left
    x = np.linspace(0, 1, 200)
    plot.plot(x, 1 - x, name="falling")  # through the top left
    view.show()
    try:
        placer = legend.findChild(lp.Placer)
        assert placer is not None
        for _ in range(3):
            app.processEvents()
        placer.place()
        assert placer.spot == 1  # top right: the line falls from the top left to the bottom right
        began = time.perf_counter()
        for _ in range(20):
            placer.place()
        assert (time.perf_counter() - began) / 20 < 0.005  # a placement, well under a frame
        assert placer.timer.interval() == lp.SLOWEST_MS  # put, again and again: placed ever less often
        placer.timer.stop()
        placer._last = 0.0  # then quiet a while
        placer.soon()
        assert placer.timer.interval() == lp.THROTTLE_MS  # a change after a quiet spell is placed promptly
        placer.timer.stop()
        placer.manual = True  # dragged: it stays there
        spot = placer.spot
        plot.listDataItems()[0].setData(x, x)
        placer.place()
        assert placer.spot == spot
    finally:
        gui.close_graphics(view)  # torn down even when it fails: a view left to gc segfaults


def test_a_legend_is_placed_at_once_and_an_empty_one_draws_nothing(app, free_windows):
    """The first placement doesn't wait for the throttle: a pass or two of the event loop after the data comes, the
    legend is already off the curve. Before it has any entry, a legend draws no box (`plot_patches`)."""
    import pyqtgraph as pg
    from PySide6 import QtGui

    from vib_tutorial import gui
    from vib_tutorial.gui import plot_patches, theming

    plot_patches.install()
    view = pg.PlotWidget()
    view.resize(400, 300)
    plot = view.getPlotItem()
    legend = theming.add_legend(plot, offset=(5, 5))  # top left
    view.show()
    try:
        for _ in range(3):
            app.processEvents()
        assert legend.boundingRect().width() > 0  # its padding: a box, if drawn
        image = QtGui.QImage(200, 100, QtGui.QImage.Format.Format_ARGB32)
        image.fill(0)
        painter = QtGui.QPainter(image)
        legend.paint(painter, None, None)
        painter.end()
        assert not any(image.pixel(x, y) for x in range(0, 200, 4) for y in range(0, 100, 4)), "an empty legend's box"
        x = np.linspace(0, 1, 200)
        plot.plot(x, 1 - x, name="falling")  # through the top left
        placer = legend.findChild(lp.Placer)
        assert placer is not None and placer.spot == 0
        began = time.perf_counter()
        for _ in range(3):
            app.processEvents()
        assert placer.spot == 1  # top right, without waiting
        assert time.perf_counter() - began < lp.THROTTLE_MS / 1000
    finally:
        gui.close_graphics(view)


def test_a_legend_put_above_at_once_is_on_the_plot(app, free_windows):
    """Where no spot inside is clear, the first placement (at once) puts the legend above the plot, in the row
    it makes there: inside the plot's bounds, not above them (it was laid out before the row existed: y < 0)."""
    import pyqtgraph as pg

    from vib_tutorial import gui
    from vib_tutorial.gui import theming

    view = pg.PlotWidget()
    view.resize(400, 300)
    plot = view.getPlotItem()
    legend = theming.add_legend(plot, offset=(5, 5))
    x = np.linspace(0, 1, 400)
    for k in range(3):
        plot.plot(x, 0.5 + 0.5 * np.sin(2 * np.pi * (25 + k) * x), name=f"curve {k}")  # over every spot
    view.show()
    try:
        for _ in range(3):
            app.processEvents()
        placer = legend.findChild(lp.Placer)
        assert placer is not None and placer.spot == lp.OUTSIDE
        assert legend.pos().y() >= 0 and legend.pos().y() + legend.boundingRect().height() <= plot.vb.pos().y() + 1
    finally:
        gui.close_graphics(view)
