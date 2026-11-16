"""Dragging lines and dots on the plots (`gui/drag.py`), with synthetic mouse events through the plot's view."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore, QtGui  # noqa: E402

Left = QtCore.Qt.MouseButton.LeftButton
NoButton = QtCore.Qt.MouseButton.NoButton


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


def drag(view: QtWidgets.QGraphicsView, start: QtCore.QPointF, end: QtCore.QPointF, app, steps: int = 8) -> None:
    """Press at `start` (scene), move to `end` in steps with the button held, release: what a hand does (QTest's
    mouseMove carries no buttons, so pyqtgraph never sees a drag)."""
    vp = view.viewport()

    def send(kind, scene_pt, button, buttons) -> None:
        local = QtCore.QPointF(view.mapFromScene(scene_pt))
        glob = QtCore.QPointF(vp.mapToGlobal(local.toPoint()))
        ev = QtGui.QMouseEvent(kind, local, glob, button, buttons, QtCore.Qt.KeyboardModifier.NoModifier)
        QtWidgets.QApplication.sendEvent(vp, ev)
        app.processEvents()

    send(QtCore.QEvent.Type.MouseButtonPress, start, Left, Left)
    for k in range(1, steps + 1):
        f = k / steps
        send(QtCore.QEvent.Type.MouseMove, start + (end - start) * f, NoButton, Left)
    send(QtCore.QEvent.Type.MouseButtonRelease, end, Left, NoButton)


def scene_at(plot, x: float, y: float) -> QtCore.QPointF:
    """A point given in the plot's view coordinates (log10 on a log axis), in scene coordinates."""
    return plot.getPlotItem().getViewBox().mapViewToScene(QtCore.QPointF(x, y))


def test_dragging_the_sdof_drive_line_sweeps_the_drive(main_window, app):
    w = main_window
    w.resize(1280, 800)
    p = w.sdof_page
    w.pages.setCurrentWidget(p)
    p.tabs.setCurrentIndex(0)
    for _ in range(5):
        app.processEvents()
    line = p.drive_lines[0]
    assert line.data_value() == pytest.approx(p.freq.value())
    seen = []
    line.dragged.connect(seen.append)
    vb = p.mag_plot.getPlotItem().getViewBox()
    (_, _), (y0, y1) = vb.viewRange()
    y = 0.5 * (y0 + y1)
    drag(p.mag_plot, scene_at(p.mag_plot, line.value(), y), scene_at(p.mag_plot, math.log10(1.0), y), app)
    assert seen and seen[-1] == pytest.approx(1.0, rel=0.03)
    assert p.freq.value() == pytest.approx(1.0, rel=0.03)  # the box follows
    assert p.drive_lines[1].data_value() == pytest.approx(p.freq.value())  # and the phase plot's line
    # the box moves the lines without a drag signal
    seen.clear()
    p.freq.setValue(5.0)
    assert line.data_value() == pytest.approx(5.0) and not seen


def test_dragging_the_sdof_poles_sets_k_and_c(main_window, app):
    """Off the real axis the pair follows the pointer; onto it, critical damping; along it, two real poles."""
    w = main_window
    w.resize(1280, 800)
    p = w.sdof_page
    w.pages.setCurrentWidget(p)
    p.tabs.setCurrentIndex(3)  # Poles
    p.s_plot.grab()
    for _ in range(5):
        app.processEvents()
    m = p.mass.value()
    upper, lower = p.pole_dots
    s = p.osc.poles()
    assert (upper.pos().x(), upper.pos().y()) == pytest.approx((s[0].real, s[0].imag))
    range_before = p.s_plot.getPlotItem().getViewBox().viewRange()

    drag(p.s_plot, scene_at(p.s_plot, s[0].real, s[0].imag), scene_at(p.s_plot, -6.0, 12.0), app)
    s = p.osc.poles()  # σ ± iω_d where it was let go: k = m(σ² + ω²), c = −2mσ
    assert s[0] == pytest.approx(complex(-6.0, 12.0), rel=0.02)
    assert p.stiffness.value() == pytest.approx(m * 180.0, rel=0.03)
    assert p.damping.value() == pytest.approx(12.0 * m, rel=0.03)
    assert (lower.pos().x(), lower.pos().y()) == pytest.approx((s[1].real, s[1].imag), rel=1e-6)
    assert p.free_item.getData()[1] is not None  # the free response beside was redrawn ...
    t, x = p.free_item.getData()
    assert np.ptp(x[t > 1.0]) < 0.01 * np.ptp(x)  # ... decaying at e^(−6t) now

    drag(p.s_plot, scene_at(p.s_plot, -6.0, -12.0), scene_at(p.s_plot, -9.0, 0.0), app)  # the lower one
    assert p.osc.zeta == pytest.approx(1.0, abs=0.01)  # met on the axis: critical
    assert p.osc.omega_n == pytest.approx(9.0, rel=0.02)

    drag(p.s_plot, scene_at(p.s_plot, -9.0, 0.0), scene_at(p.s_plot, -3.0, 0.0), app)
    s = np.sort(p.osc.poles().real)  # one went to −3, the other stayed at −9
    assert s == pytest.approx([-9.0, -3.0], rel=0.03)
    assert p.stiffness.value() == pytest.approx(m * 27.0, rel=0.03)

    drag(p.s_plot, scene_at(p.s_plot, -3.0, 0.0), scene_at(p.s_plot, 5.0, 8.0), app)  # into the right half
    assert p.damping.value() == 0.0 and p.osc.poles()[0].real == pytest.approx(0.0, abs=1e-9)
    assert p.s_plot.getPlotItem().getViewBox().viewRange() != range_before  # refitted once let go


def test_a_drag_dot_snaps_to_its_path_and_emits_the_parameter(app):
    import pyqtgraph as pg

    from vib_tutorial.gui import close_graphics
    from vib_tutorial.gui.drag import DragDot

    w = pg.PlotWidget()
    w.resize(400, 300)
    w.show()
    x = np.linspace(0.0, 10.0, 21)
    y = (x - 5.0) ** 2
    params = np.arange(21) * 100.0
    dot = DragDot("tip")
    w.addItem(dot)
    dot.set_path(x, y, params)
    dot.place(x[4], y[4], params[4])
    w.setRange(xRange=(-1, 11), yRange=(-2, 27), padding=0)
    for _ in range(5):
        app.processEvents()
    got = []
    dot.dragged.connect(got.append)
    done = []
    dot.released.connect(done.append)
    try:
        _drag_dot(w, dot, x, y, params, got, done, app)
    finally:
        close_graphics(w)
        w.close()


def _drag_dot(w, dot, x, y, params, got, done, app) -> None:
    # dragged to (7.1, 3): the nearest path point on screen is x = 7.0 (y = 4)
    drag(w, scene_at(w, x[4], y[4]), scene_at(w, 7.1, 3.0), app)
    assert got and got[-1] == pytest.approx(params[14]) and done == [got[-1]]
    pos = dot.pos()
    assert (pos.x(), pos.y()) == pytest.approx((7.0, 4.0))


def _show(w, page, app) -> None:
    w.resize(1280, 800)
    w.pages.setCurrentWidget(page)
    for _ in range(5):
        app.processEvents()


def test_dragging_the_isolation_speed_line_sets_the_speed(main_window, app):
    p = main_window.isolation_page
    _show(main_window, p, app)
    line = p.speed_line
    r0 = p.iso.speed_hz / p.iso.natural_hz
    assert line.data_value() == pytest.approx(r0)
    (_, _), (y0, y1) = p.tr_plot.getPlotItem().getViewBox().viewRange()
    y = 0.5 * (y0 + y1)
    drag(p.tr_plot, scene_at(p.tr_plot, line.value(), y), scene_at(p.tr_plot, math.log10(0.8), y), app)
    assert p.iso.speed_hz / p.iso.natural_hz == pytest.approx(0.8, rel=0.03)  # below √2: amplified
    assert p.rpm.value() == pytest.approx(0.8 * p.iso.natural_hz * 60.0, rel=0.03)
    assert "amplified" in p.tr_note.text()


def test_dragging_the_quarter_car_dot_changes_the_damper(main_window, app):
    from vib_tutorial.gui.quarter_car import DAMPERS

    p = main_window.quarter_car_page
    _show(main_window, p, app)
    p.tabs.setCurrentIndex(next(k for k in range(p.tabs.count()) if p.tabs.tabText(k).startswith("Trade")))
    for _ in range(5):
        app.processEvents()
    dot = p.trade_now
    before = p.damper.value()
    i = int(np.argmin(np.abs(DAMPERS - 4000.0)))  # a firmer damper, on the car's spring
    start = dot.getViewBox().mapViewToScene(dot.pos())
    end = scene_at(p.trade_plot, float(dot._x[i]), float(dot._y[i]))
    drag(p.trade_plot, start, end, app)
    assert p.damper.value() == pytest.approx(round(DAMPERS[i]), abs=1) and p.damper.value() != before
    p.tabs.setCurrentIndex(next(k for k in range(p.tabs.count()) if p.tabs.tabText(k).startswith("Damper")))
    app.processEvents()
    assert p.damper_now.data_value() == pytest.approx(p.damper.value(), rel=1e-6)  # the sweep's line, once shown


def test_dragging_a_pole_along_the_active_control_locus_sets_the_gain(main_window, app):
    from vib_tutorial.gui.active_control import TAB_LOCUS

    w = main_window
    w.resize(1280, 800)
    p = w.active_control_page
    w.pages.setCurrentWidget(p)
    p.tabs.setCurrentIndex(TAB_LOCUS)
    p.locus_plot.grab()
    for _ in range(5):
        app.processEvents()
    shown = [d for d in p.locus_now if d.isVisible()]
    assert shown and all(d.param == pytest.approx(p.gain.value()) for d in shown)
    dot = shown[0]
    x, y, g = dot._x, dot._y, dot._p
    near = np.flatnonzero(np.abs(np.log(np.maximum(g, 1e-9) / 300.0)) < 0.05)  # each branch's point at ~300 N·s/m
    start = scene_at(p.locus_plot, dot.pos().x(), dot.pos().y())
    far = [(scene_at(p.locus_plot, x[i], y[i]) - start).manhattanLength() for i in near]
    target = int(near[np.argmax(far)])  # the one farthest from the dot on screen
    assert max(far) > 40
    drag(p.locus_plot, start, scene_at(p.locus_plot, x[target], y[target]), app)
    assert p.gain.value() == pytest.approx(g[target], rel=0.02)  # the box follows ...
    assert all(d.param == pytest.approx(p.gain.value()) for d in shown)  # ... and every pole moves to it
