"""`gui/playback.py`: a response walked in time, and the Single oscillator's mass moving with it."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore  # noqa: E402
from test_drag import drag, scene_at  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


def test_playback_walks_the_span_pauses_at_its_end_and_scrubs(app):
    from vib_tutorial.gui import close_graphics
    from vib_tutorial.gui.lesson import plot
    from vib_tutorial.gui.playback import SPEEDS, Playback

    page = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(page)
    p = plot("x", "m", "t", "s")
    pb = Playback(page, "test")
    layout.addWidget(pb.bar())
    layout.addWidget(p)
    cursor = pb.add_cursor(p)
    seen: list[float] = []
    pb.moved.connect(seen.append)
    p.setXRange(0.0, 2.0, padding=0.05)
    page.resize(600, 400)
    page.show()
    for _ in range(10):
        app.processEvents()
    page.grab()  # laid out and painted: the cursor has its place on screen
    pb.loop.enabled = False  # the frames are driven by hand
    pb.speed.setCurrentIndex(next(i for i, (_, s) in enumerate(SPEEDS) if s == 0.5))
    pb.set_span(2.0)
    assert pb.t == 0.0 and pb.run.text() == "Start" and not pb.running

    pb.play()
    assert pb.running and pb.run.text() == "Pause"
    pb._tick(1.0)  # a second of wall clock at ½×
    assert pb.t == pytest.approx(0.5) and cursor.data_value() == pytest.approx(0.5) and pb.clock.text() == "t = 0.50 s"
    pb._tick(10.0)
    assert pb.t == 2.0 and not pb.running and pb.run.text() == "Resume"  # paused at the end
    pb.run.click()  # Resume at the end plays it again
    assert pb.t == 0.0 and pb.running

    pb._tick(1.0)
    pb.set_span(0.3)  # the page redrew with a shorter span: t is kept inside it
    assert pb.t == pytest.approx(0.3) and seen[-1] == pytest.approx(0.3)

    pb.set_span(2.0)
    pb.play()
    drag(p, scene_at(p, 0.0, 0.5), scene_at(p, 1.2, 0.5), app)  # the reader scrubs the cursor
    assert not pb.running and pb.t == pytest.approx(1.2, abs=0.05) and seen[-1] == pb.t

    pb.reset.click()
    assert pb.t == 0.0 and pb.run.text() == "Start" and cursor.data_value() == 0.0
    close_graphics(page)


def test_the_sdof_mass_moves_with_the_response_and_a_pluck_sets_x0(main_window, app):
    from vib_tutorial.core.sdof import free_response, harmonic_from_rest
    from vib_tutorial.gui import sdof

    w = main_window
    w.resize(1280, 800)
    p = w.sdof_page
    w.pages.setCurrentWidget(p)
    for _ in range(5):
        app.processEvents()
    pb = p.playback
    pb.loop.enabled = False
    assert not pb.running and pb.run.text() == "Start" and pb.t == 0.0  # at rest, waiting for Start
    view = p.body
    assert view._centers[0] - 1.0 == pytest.approx(view.gain * 0.010)  # drawn at x0 = 10 mm

    # Played: the mass is where the free vibration is.
    pb.run.click()
    pb._tick(0.4)
    t = pb.t
    x, v = free_response(p.osc, 0.010, 0.0, np.array([t]))
    assert view._centers[0] - 1.0 == pytest.approx(view.gain * x[0])
    assert p.plane_now.pos().x() == pytest.approx(x[0]) and p.plane_now.pos().y() == pytest.approx(v[0] / p.osc.omega_n)
    gain = view.gain
    p.damping.setValue(4.0)  # a control keeps the run state and the time
    assert pb.running and pb.t == t and view.gain == gain  # the same peak: no rescale

    # Pluck: drag the mass right and let go. x0 is where it was let go, v0 = 0, and it runs from t = 0.
    centre = view.plotItem.vb.mapViewToScene(QtCore.QPointF(view._centers[0], 0.0))
    end = view.plotItem.vb.mapViewToScene(QtCore.QPointF(1.0 + 0.4, 0.0))  # 0.4 display units from rest
    drag(view, centre, end, app)
    assert view.held is None and pb.running and pb.t == 0.0
    assert p.v0.value() == 0.0 and p.x0.value() == pytest.approx(1e3 * 0.4 / view.gain, rel=0.05)
    assert view._centers[0] - 1.0 == pytest.approx(view.gain * 1e-3 * p.x0.value(), rel=1e-3)
    assert p.follow.currentIndex() == sdof.BODY_FREE

    # A driven trial: the mass follows the forced response, the force arrow on it, its plot's cursor shown.
    p.load_trial(next(i for i, tr in enumerate(sdof.TRIALS) if tr.name.startswith("Resonance")))
    assert p.follow.currentIndex() == sdof.BODY_DRIVEN and pb.running and pb.t == 0.0
    assert p.cursors[1].isVisible() and not p.cursors[0].isVisible()
    pb._tick(2.0)
    f = p.freq.value()
    x, _ = harmonic_from_rest(p.osc, 1.0, f, np.array([pb.t]))
    assert view._centers[0] - 1.0 == pytest.approx(view.gain * x[0])
    assert pb.span == pytest.approx(p.forced_plot.getPlotItem().getViewBox().viewRange()[0][1], rel=0.02)
    assert view._force_text.toPlainText() == f"F = {np.cos(2 * np.pi * f * pb.t):+.3g} N"

    # Defaults: back to the free vibration, at rest.
    w.restore_defaults()
    assert p.follow.currentIndex() == sdof.BODY_FREE and not pb.running and pb.run.text() == "Start" and pb.t == 0.0


def test_the_pendulum_swings_with_its_curve_and_a_pluck_lets_it_go_from_there(main_window, app):
    import math

    from vib_tutorial.core.pendulum import Kind
    from vib_tutorial.gui import pendulum

    w = main_window
    w.resize(1280, 800)
    p = w.pendulum_page
    w.pages.setCurrentWidget(p)
    for _ in range(5):
        app.processEvents()
    pb, view = p.playback, p.view
    pb.loop.enabled = False
    assert not pb.running and pb.run.text() == "Start" and view.theta == pytest.approx(math.radians(10.0))

    # A trial plays it: the pendulum is where the whole swing is, the ghost where the linearized one is.
    p.load_trial(next(i for i, tr in enumerate(pendulum.TRIALS) if tr.name.startswith("Large swings")))
    assert pb.running and pb.t == 0.0
    pb._tick(0.7)
    t, th, _, tl = (np.asarray(a) for a in p._swing)
    assert view.theta == pytest.approx(np.interp(pb.t, t, th)) and view.ghost.isVisible()
    gx, gy = view.ghost.getData()
    assert math.atan2(gx[1], -gy[1]) == pytest.approx(np.interp(pb.t, t, tl))
    assert p.moment_now.pos().x() == pytest.approx(math.degrees(view.theta))
    x, y = p.whole_item.getData()
    assert np.allclose(np.degrees(np.interp(pb.t, t, th)), np.interp(pb.t, x, y))  # on the curve drawn

    # Pluck it to the left, to 40°: the start box says 40°, the swing is mirrored and plays from t = 0.
    for _ in range(5):
        app.processEvents()
    view.grab()  # laid out: the handle has its place on screen
    pb.seek(0.0)
    vb = view.plotItem.vb
    hand = vb.mapViewToScene(QtCore.QPointF(*view._point(view.theta)))
    to = vb.mapViewToScene(QtCore.QPointF(*view._point(-math.radians(40.0))))
    drag(view, hand, to, app)
    assert not view.held and pb.running and pb.t == 0.0 and p.side == -1.0
    assert p.angle.value() == pytest.approx(40.0, abs=1.5)
    x, y = p.whole_item.getData()
    assert y[0] == pytest.approx(-p.angle.value()) and view.theta == pytest.approx(math.radians(y[0]))

    # A kind drawn its own way: the disc twists about its wire, its mark at θ.
    p.load_trial(next(i for i, tr in enumerate(pendulum.TRIALS) if tr.kind is Kind.TORSION))
    assert p.side == 1.0 and view.disc.isVisible() and not view.bob.isVisible()
    xs, ys = view.body.getData()
    assert math.atan2(xs[1], ys[1]) == pytest.approx(math.radians(120.0))
