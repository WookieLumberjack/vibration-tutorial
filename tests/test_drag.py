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
        # to the subpixel: mapFromScene rounds to whole pixels, a step of the data on a small plot (Windows')
        local = view.viewportTransform().map(scene_pt)
        glob = vp.mapToGlobal(local)
        ev = QtGui.QMouseEvent(kind, local, glob, button, buttons, QtCore.Qt.KeyboardModifier.NoModifier)
        QtWidgets.QApplication.sendEvent(vp, ev)
        app.processEvents()

    import pyqtgraph as pg

    # pyqtgraph drops moves closer together than its rate limit: these come at once, and only the first (if it
    # cleared the 5 px a drag needs) and the release would count. A hand's moves are spread out: let all through.
    limit = pg.getConfigOption("mouseRateLimit")
    pg.setConfigOption("mouseRateLimit", 0)
    try:
        send(QtCore.QEvent.Type.MouseButtonPress, start, Left, Left)
        for k in range(1, steps + 1):
            f = k / steps
            send(QtCore.QEvent.Type.MouseMove, start + (end - start) * f, NoButton, Left)
        send(QtCore.QEvent.Type.MouseButtonRelease, end, Left, NoButton)
    finally:
        pg.setConfigOption("mouseRateLimit", limit)


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


def test_dragging_a_pole_along_the_complex_modes_sweep_sets_the_element_on_release(main_window, app):
    from vib_tutorial.gui.complex_modes import TAB_SWEEP

    w = main_window
    w.resize(1280, 800)
    p = w.complex_page
    w.pages.setCurrentWidget(p)
    p.load_setup({"damping": (20.0, 0.0, 0.0, 0.0), "tab": TAB_SWEEP})  # c1 = 20 N·s/m, swept
    p.locus.grab()
    for _ in range(5):
        app.processEvents()
    sw = p.sweep_result
    assert sw is not None and sw.element == 0
    shown = [d for d in p.sweep_dots if d.isVisible()]
    assert shown and all(d.param == pytest.approx(20.0, rel=0.1) for d in shown)
    dot = shown[0]
    start = scene_at(p.locus, dot.pos().x(), dot.pos().y())
    j = int(np.argmin(np.abs(sw.values - 200.0)))  # to the top mode's pole at c1 ≈ 200 N·s/m, well away on screen
    lam = sw.poles[j, -1]
    assert np.isfinite(lam)
    seen = []
    dot.dragged.connect(seen.append)
    end = scene_at(p.locus, lam.real, lam.imag / (2 * np.pi))
    assert (end - start).manhattanLength() > 40
    drag(p.locus, start, end, app)
    assert seen and abs(math.log(seen[-1] / sw.values[j])) < 0.5  # a point of that locus near there on screen
    snapped = sw.poles[int(np.argmin(np.abs(sw.values - seen[-1]))), -1]
    assert (scene_at(p.locus, snapped.real, snapped.imag / (2 * np.pi)) - end).manhattanLength() < 6
    # let go: the chain took it (as its box shows it, to 0.001 N·s/m), through the Simulation page
    assert p.system.damping[0] == pytest.approx(seen[-1], abs=5.01e-4)
    assert w.sim_page.sim.system.damping[0] == pytest.approx(seen[-1], abs=5.01e-4)
    for _ in range(3):
        app.processEvents()
    assert all(d.param == pytest.approx(seen[-1], abs=5.01e-4) for d in p.sweep_dots if d.isVisible())  # redrawn


def test_dragging_a_pole_along_the_flutter_locus_sets_the_speed(main_window, app):
    from vib_tutorial.gui.self_excited import TAB_SWEEP, Kind

    w = main_window
    w.resize(1280, 800)
    p = w.self_excited_page
    w.pages.setCurrentWidget(p)
    p.load_setup(
        dict(kind=Kind.FLUTTER, values={"f_speed": 25.0}, tab=TAB_SWEEP)
    )  # past flutter: one pole right of Re = 0
    p.sweep_plot.grab()
    for _ in range(5):
        app.processEvents()
    dots = p.flutter_dots
    assert all(d.isVisible() and d.param == pytest.approx(25.0) for d in dots)
    assert max(d.pos().x() for d in dots) > 0  # unstable
    dot = max(dots, key=lambda d: d.pos().x())
    x, y, u = dot._x, dot._y, dot._p
    start = scene_at(p.sweep_plot, dot.pos().x(), dot.pos().y())
    near = np.flatnonzero(np.abs(u - 10.0) < 0.5)  # each pole at ~10 m/s, below the flutter speed
    far = [(scene_at(p.sweep_plot, x[i], y[i]) - start).manhattanLength() for i in near]
    target = int(near[np.argmax(far)])
    assert max(far) > 40
    drag(p.sweep_plot, start, scene_at(p.sweep_plot, x[target], y[target]), app)
    # The box follows to near the target: the locus there is ~2.5 px per 0.1 m/s, and the release builds
    # land a few m/s off (Windows let go at 7.1 for 9.7), so "near" is the stretch well below flutter.
    assert abs(p.v("f_speed") - u[target]) < 3.0
    assert all(d.isVisible() and d.param == pytest.approx(p.v("f_speed")) for d in dots)  # ... the poles too
    assert max(d.pos().x() for d in dots) < 0  # below flutter: both stable

    p.load_setup(dict(kind=Kind.VDP, values={}, tab=TAB_SWEEP))  # another model's sweep: no locus, no dots
    for _ in range(3):
        app.processEvents()
    assert not any(d.isVisible() for d in dots)


def test_drawing_the_energy_methods_trial_shape_keeps_rayleigh_above_f1(main_window, app):
    """Dragging a mass's point makes the shape the reader's own (the combo says so): Rayleigh's f₁ follows and
    stays above the exact f₁; the drawn shape survives the practice state's round trip."""
    import math

    from vib_tutorial.core.energy_methods import rayleigh
    from vib_tutorial.gui.energy_methods import DRAWN, TWO_PI

    w = main_window
    w.resize(1280, 800)
    p = w.energy_page
    w.pages.setCurrentWidget(p)
    p.shape_plot.grab()
    for _ in range(3):
        app.processEvents()
    g = p.grips[1]
    drag(p.shape_plot, scene_at(p.shape_plot, 2.0, g.pos().y()), scene_at(p.shape_plot, 2.0, -0.8), app)
    assert p.shape.currentText() == DRAWN and p.drawn is not None
    assert p.drawn[1] == pytest.approx(-0.8, abs=0.15)  # to a few pixels and g.pos().y() == pytest.approx(p.drawn[1])
    f_ray = math.sqrt(rayleigh(p.system, p.drawn)) / TWO_PI
    assert f_ray > p.f[0] and f"{f_ray:.4g} Hz" in p.readout.text() and "Your shape" in p.readout.text()

    state = p.practice_state()
    assert state.drawn == tuple(p.drawn)
    p.load_trial(0)
    assert p.shape.currentText() != DRAWN
    p.load_setup({k: v for k, v in vars(state).items() if k not in ("name", "note")})
    assert p.shape.currentText() == DRAWN and np.allclose(p.drawn, state.drawn)


@pytest.mark.parametrize("log", [False, True])
def test_dragging_the_torsional_speed_line_sets_the_running_speed(main_window, app, log):
    p = main_window.torsional_page
    _show(main_window, p, app)
    p.log_scales.setChecked(log)
    app.processEvents()
    line = p.speed_line
    assert line.data_value() == pytest.approx(p.operating.value()) and line.getViewBox() is not None
    lo, hi = p._interference_band.getRegion()  # the band drawn on the scale just switched to
    edges = [10**v for v in (lo, hi)] if log else [lo, hi]
    assert edges == pytest.approx([0.9 * p.operating.value(), 1.1 * p.operating.value()], rel=1e-6)
    (_, _), (y0, y1) = p.interference.getPlotItem().getViewBox().viewRange()
    y = 0.5 * (y0 + y1)
    target = 2000.0
    end = math.log10(target) if log else target
    drag(p.interference, scene_at(p.interference, line.value(), y), scene_at(p.interference, end, y), app)
    assert p.operating.value() == pytest.approx(target, rel=0.02)
    lo, hi = p._interference_band.getRegion()
    edges = [10**v for v in (lo, hi)] if log else [lo, hi]
    assert edges == pytest.approx([0.9 * p.operating.value(), 1.1 * p.operating.value()], rel=1e-6)
    assert line.getViewBox() is not None and line.data_value() == pytest.approx(p.operating.value())


def test_dragging_the_random_band_edges_sets_the_band(main_window, app):
    p = main_window.random_page
    _show(main_window, p, app)
    p.recompute()
    app.processEvents()
    lo_line, hi_line = p.band_lines
    hi0 = p.f_hi.value()
    assert hi_line.data_value() == pytest.approx(hi0) and hi_line.getViewBox() is not None
    (_, _), (y0, y1) = p.input_plot.getPlotItem().getViewBox().viewRange()
    y = 0.5 * (y0 + y1)
    target = 0.5 * hi0
    drag(p.input_plot, scene_at(p.input_plot, hi_line.value(), y), scene_at(p.input_plot, math.log10(target), y), app)
    assert p.f_hi.value() == pytest.approx(target, rel=0.03)
    p.worker.wait()
    app.processEvents()
    assert p.record.input.f_hi == p.f_hi.value()  # made again with the new band, once let go
    assert hi_line.getViewBox() is not None and hi_line.data_value() == pytest.approx(p.f_hi.value())
    # The bottom edge: the band from 0 starts where the plot does; dragged in, it starts there.
    lo = 0.3 * p.f_hi.value()
    drag(p.input_plot, scene_at(p.input_plot, lo_line.value(), y), scene_at(p.input_plot, math.log10(lo), y), app)
    assert p.f_lo.value() == pytest.approx(lo, rel=0.03) and p.f_lo.value() < p.f_hi.value()
    p.worker.wait()
    assert p.record.input.f_lo == p.f_lo.value()


@pytest.mark.parametrize("which", [0, 2])  # the frequency map (the Run-up tab) and the order tracks (Orders)
def test_dragging_the_diagnostics_speed_line_sets_the_shaft_speed(main_window, app, which):
    from vib_tutorial.gui.diagnostics import TAB_ORDERS, TAB_RUNUP

    p = main_window.diagnostics_page
    _show(main_window, p, app)
    p.tabs.setCurrentIndex(TAB_RUNUP if which == 0 else TAB_ORDERS)
    for _ in range(5):
        app.processEvents()
    plot = (p.freq_map, p.order_map_plot, p.orders_plot)[which]
    line = p.speed_lines[which]
    assert line.data_value() == pytest.approx(p.rpm.value())
    (_, _), (y0, y1) = plot.getPlotItem().getViewBox().viewRange()
    y = 0.5 * (y0 + y1)
    drag(plot, scene_at(plot, line.value(), y), scene_at(plot, 3000.0, y), app)
    assert p.rpm.value() == pytest.approx(3000.0, rel=0.02)
    assert p.mesh_lines[0].value() == pytest.approx(p.machine.mesh_hz)  # the record at the new speed
    assert all(other.data_value() == pytest.approx(p.rpm.value()) for other in p.speed_lines)


def _turn_by_hand(view, dx: float, app) -> None:
    """Press in the middle of a 3D view, move `dx` px sideways with the button held, release."""
    vp = view.viewport()
    mid = QtCore.QPointF(vp.width() / 2, vp.height() / 2)
    for kind, at, button, buttons in (
        (QtCore.QEvent.Type.MouseButtonPress, mid, Left, Left),
        (QtCore.QEvent.Type.MouseMove, mid + QtCore.QPointF(dx / 2, 0), NoButton, Left),
        (QtCore.QEvent.Type.MouseMove, mid + QtCore.QPointF(dx, 0), NoButton, Left),
        (QtCore.QEvent.Type.MouseButtonRelease, mid + QtCore.QPointF(dx, 0), Left, NoButton),
    ):
        ev = QtGui.QMouseEvent(kind, at, vp.mapToGlobal(at), button, buttons, QtCore.Qt.KeyboardModifier.NoModifier)
        QtWidgets.QApplication.sendEvent(vp, ev)
    app.processEvents()


@pytest.mark.parametrize(
    "where",
    [
        "rigid_body.BodyView",
        "gyroscope_view.TopView",
        "rigid_mounts.MountView",
        "rotor_view.RotorView",
    ],
)
@pytest.mark.parametrize("pitch", [0.0, 0.5, math.pi / 2])
def test_a_sideways_drag_turns_the_near_side_with_the_cursor(app, where, pitch):
    """In each 3D view, dragging left moves the side nearest you left (#409 N-4: it went right), seen level, from
    a tilt, and from straight above (the near side then at the bottom: the body turns clockwise)."""
    import importlib

    from vib_tutorial import gui

    module, name = where.split(".")
    view = getattr(importlib.import_module(f"vib_tutorial.gui.{module}"), name)()
    view.resize(400, 300)
    view.show()
    try:
        view.pitch = pitch
        around = np.linspace(0, 2 * math.pi, 360, endpoint=False)
        upright = name == "RotorView"  # its vertical is y and it turns about it; the others' is z

        sy, cy = math.sin(view.yaw), math.cos(view.yaw)
        if upright:
            p = np.stack([np.cos(around), np.zeros_like(around), np.sin(around)], axis=-1)
            toward = -p[:, 0] * sy + p[:, 2] * cy
        else:
            p = np.stack([np.cos(around), np.sin(around), np.zeros_like(around)], axis=-1)
            toward = -(p[:, 0] * sy + p[:, 1] * cy)
        near = p[int(np.argmax(toward))]  # the point of the circle nearest you, followed as it turns
        before = float(view.project(near)[0])
        _turn_by_hand(view, -20, app)
        assert float(view.project(near)[0]) < before - 0.05
        _turn_by_hand(view, 40, app)
        assert float(view.project(near)[0]) > before + 0.05
    finally:
        gui.close_graphics(view)


@pytest.mark.parametrize("page", ["pendulum_page", "damping_page", "isolation_page"])
def test_a_pluckable_body_is_taken_pulled_and_let_go_with_the_mouse(main_window, app, page):
    """`drag.Grabbable` (#409 S-11): a press on the body takes it (closed hand, `grabbed`), a move pulls it, the
    release lets it go (`released`); a press off it is no grab."""
    from PySide6.QtTest import QTest

    from vib_tutorial.gui.damping_view import REST

    w = main_window
    w.resize(1280, 800)
    w.show_page(page)
    p = getattr(w, page)
    view = p.view if page == "pendulum_page" else p.body
    for _ in range(5):
        app.processEvents()
    if page == "pendulum_page":
        on = QtCore.QPointF(*view._point(view.theta))
    elif page == "damping_page":
        model, row = next(iter(view._rows.items()))
        on = QtCore.QPointF(REST + view.gain * view._x.get(model, 0.0), row)  # as `on_mass` places it
    else:
        on = view.machine.rect().center()
    assert view.grab_at(on) is True

    def pixel(at: QtCore.QPointF) -> QtCore.QPoint:
        return view.mapFromScene(view.plotItem.vb.mapViewToScene(at))

    seen: list[str] = []
    view.grabbed.connect(lambda: seen.append("grabbed"))
    view.released.connect(lambda: seen.append("released"))
    left = QtCore.Qt.MouseButton.LeftButton
    QTest.mousePress(view.viewport(), left, pos=pixel(on))
    assert view.held is True and seen == ["grabbed"] and view.cursor().shape() == QtCore.Qt.CursorShape.ClosedHandCursor
    QTest.mouseMove(view.viewport(), pixel(on + QtCore.QPointF(0.02, 0.02)))
    QTest.mouseRelease(view.viewport(), left, pos=pixel(on + QtCore.QPointF(0.02, 0.02)))
    assert view.held is False and seen == ["grabbed", "released"]
    view.grabbed.disconnect()
    view.released.disconnect()
