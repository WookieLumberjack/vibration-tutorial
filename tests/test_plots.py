"""`gui/plots.py`'s `SteadyRange`: a live plot's axis that moves only when its data outgrows it or shrinks well inside
it (#316 E7); the SI prefixes axes leave off; the rotor strip chart's sparse axes and its ranges at rest."""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from vib_tutorial.gui.lesson import plot  # noqa: E402
from vib_tutorial.gui.plots import SteadyRange  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


def test_a_steady_range_moves_only_when_it_has_to(app):
    p = plot("Displacement", "m", "Time", "s")
    y = SteadyRange(p, "y")
    vb = p.getPlotItem().vb
    assert not vb.autoRangeEnabled()[1]

    y.fit(-1.0, 1.0)
    assert y.shown == pytest.approx((-1.15, 1.15)) and vb.viewRange()[1] == pytest.approx([-1.15, 1.15])
    for lo, hi in ((-0.9, 0.95), (-1.1, 0.6), (-0.7, 0.7)):  # wandering inside, over half the range: kept
        y.fit(lo, hi)
        assert y.shown == pytest.approx((-1.15, 1.15))
    y.fit(-1.0, 2.0)  # out past it: grows
    assert y.shown == pytest.approx((0.5 - 1.725, 0.5 + 1.725))
    y.fit(-0.2, 0.2)  # under half of it: shrinks
    assert y.shown == pytest.approx((-0.23, 0.23))
    y.fit(0.0, 0.0)  # at rest: nothing to fit, kept
    y.fit(float("nan"), 1.0)
    assert y.shown == pytest.approx((-0.23, 0.23))
    x = SteadyRange(p, "x")
    x.fit(2.0, 4.0)
    assert vb.viewRange()[0] == pytest.approx([3.0 - 1.15, 3.0 + 1.15])
    x.reset()
    assert x.shown is None


def test_no_si_prefix_on_percent_degrees_or_decibels(app):
    """An axis in %, °, rpm or dB keeps its numbers: no "m%" or "k°" (`plot_patches._set_label`, #409 U-18).
    Metres still take one, and an axis relabelled from % to metres takes one again."""
    from vib_tutorial.gui import plot_patches

    plot_patches.install()  # as make_app does
    plots = []  # kept till the test's end: one freed with its items in their scene segfaults
    for units, small in (("%", 0.04), ("°", 0.004), ("dB", 0.003), ("dB re 1 µPa", 0.003), ("rpm", 0.002)):
        p = plot("Of 1X", units, "Harmonic", None)
        plots.append(p)
        p.plot([0, 1], [0, small])
        p.getPlotItem().vb.autoRange()
        axis = p.getAxis("left")
        assert axis.labelString().count(units) == 1 and f"m{units}" not in axis.labelString(), axis.labelString()
        assert axis.autoSIPrefixScale == 1.0, units
    p = plot("Lift", "m", "Time", "s")
    p.plot([0, 1], [0, 0.004])
    p.getPlotItem().vb.autoRange()
    axis = p.getAxis("left")
    assert "(mm)" in axis.labelString()
    axis.setLabel("Share", units="%")
    assert "(%)" in axis.labelString() and axis.autoSIPrefixScale == 1.0
    axis.setLabel("Lift", units="m")
    p.getPlotItem().vb.autoRange()
    assert "(mm)" in axis.labelString()


def test_a_sparse_axis_labels_only_what_its_length_has_room_for(app):
    """The rotor's strip chart is ~40 px a plot at 1280 × 800: two labels, not "1500/1000/500/0" (#409 U-10)."""
    from vib_tutorial.gui.rotor_plots import SparseAxis

    axis = SparseAxis("left")

    def labels(lo: float, hi: float, size: float) -> list[float]:
        (_, major), *_ = axis.tickValues(lo, hi, size)
        return [round(v, 9) for v in major]

    assert labels(0.0, 1743.0, 35) == [0.0, 1000.0]  # the speed up past its 1584 rpm critical
    assert labels(-2.28, 2.28, 47) == [-1.0, 1.0]  # x and y running, in µm: odd multiples, where -2, 0, 2 won't fit
    assert labels(-1.05, 1.05, 50) == [-1.0, 1.0]
    assert labels(-1.05, 1.05, 80) == [-1.0, 0.0, 1.0]
    tall = labels(0.0, 1743.0, 200)  # 1700 × 900, or a taller window: more labels, never stacked
    assert 3 <= len(tall) <= int(SparseAxis.FILL * 200 / SparseAxis.LABEL)


def test_the_rotor_strip_chart_shows_its_ranges_before_start(main_window):
    """At rest, before anything's recorded, the strip chart spans its window in s and the speed up past the first
    critical, not pyqtgraph's 0 to 1 (whose ticks stacked round the lone critical line, in ms) (#409 U-10)."""
    w = main_window
    w.show_page("rotor_page")
    page = w.rotor_page
    chart = page.time
    end = time.perf_counter() + 60.0  # the critical speeds come from the analysis, on a worker
    while not chart.crit_lines and time.perf_counter() < end:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.02)
    assert not page.running and not chart._recorded
    (x0, x1), _ = chart.x_plot.vb.viewRange()
    assert (x0, x1) == pytest.approx((0.0, page.time_window.value()))
    _, (lo, hi) = chart.w_plot.vb.viewRange()
    first = chart.crit_lines[0].value()
    assert lo == 0.0 and first < hi < 1.2 * first
    assert chart.w_plot.getAxis("bottom").autoSIPrefixScale == 1.0  # Time in s, as when it runs
