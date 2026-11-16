"""`gui/plots.py`'s `SteadyRange`: a live plot's axis that moves only when its data outgrows it or shrinks well inside
it (#316 E7)."""

from __future__ import annotations

import os

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
