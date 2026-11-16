"""`gui/playback.py`: a response walked in time, and the One mass pages' bodies moving with it."""

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


def test_the_damped_masses_move_with_their_decays_and_a_pluck_releases_them_all(main_window, app):
    from vib_tutorial.core.damping_models import MODELS, Model, coulomb_stop
    from vib_tutorial.gui import damping_models
    from vib_tutorial.gui.damping_view import REST, SWING

    w = main_window
    w.resize(1280, 800)
    p = w.damping_page
    w.pages.setCurrentWidget(p)
    pb, view = p.playback, p.body
    pb.loop.enabled = False

    # A trial plays it: each mass is where its own decay is, as drawn.
    p.load_trial(next(i for i, tr in enumerate(damping_models.TRIALS) if tr.name.startswith("A straight-line")))
    for _ in range(5):
        app.processEvents()
    assert pb.running and pb.t == 0.0 and p.tabs.currentIndex() == damping_models.TAB_DECAY
    pb._tick(0.3)
    for m in MODELS:
        t, x = p.decay_items[m].getData()
        assert view._x[m] == pytest.approx(np.interp(pb.t, t, x))
    friction = p.model.friction
    assert view.gain == pytest.approx(SWING / p.model.amplitude)

    # Pull a mass in to 3 mm and let go: all four start from there, friction still matched at X = 10 mm, so it
    # sticks after fewer swings.
    view.grab()  # laid out: the masses have their places on screen
    pb.seek(0.0)
    vb = view.plotItem.vb
    y = view._rows[Model.VISCOUS]
    hand = vb.mapViewToScene(QtCore.QPointF(REST + SWING, y))
    to = vb.mapViewToScene(QtCore.QPointF(REST + 0.3 * SWING, y))
    drag(view, hand, to, app)
    assert not view.held and pb.running and pb.t == 0.0
    assert p.release == pytest.approx(3e-3, rel=0.05) and p.model.friction == friction
    for m in MODELS:
        t, x = p.decay_items[m].getData()
        assert x[0] == pytest.approx(p.release)
    halves, _ = coulomb_stop(p.model.stiffness, friction, p.release)
    assert halves < coulomb_stop(p.model.stiffness, friction, p.model.amplitude)[0]
    assert f"stops after {halves / 2:g} cycles" in p.decay_note.text()

    # A control change (or Defaults) releases them from X again.
    w.restore_defaults()
    assert p.release is None and p.decay_items[Model.VISCOUS].getData()[1][0] == pytest.approx(p.model.amplitude)


def test_the_machine_runs_up_on_its_mounts_and_a_pull_lets_it_ring(main_window, app):
    from vib_tutorial.gui.isolation_view import MOST, SWING
    from vib_tutorial.view_models.isolation import TAB_CHART, TAB_START, TRIALS

    w = main_window
    w.resize(1280, 800)
    p = w.isolation_page
    w.pages.setCurrentWidget(p)
    pb, view = p.playback, p.body
    pb.loop.enabled = False

    # A trial plays the run-up: the machine, its unbalance and the forces are the record's, as drawn.
    p.load_trial(next(i for i, tr in enumerate(TRIALS) if tr.name.startswith("Running up")))
    for _ in range(5):
        app.processEvents()
    assert pb.running and pb.t == 0.0 and p.tabs.currentIndex() == TAB_START and p.pulled_to is None
    pb._tick(0.3)
    v = p.record
    x, stage, phase, floor, force = view.state
    assert force == pytest.approx(np.interp(pb.t, *v.force))
    t, xs = p.sx_item.getData()
    assert x == pytest.approx(np.interp(pb.t, t, xs)) and floor == pytest.approx(np.interp(pb.t, *v.floor))
    assert phase == pytest.approx(np.interp(pb.t, *v.phase)) and phase > 0 and stage == 0.0
    assert view.gain == pytest.approx(SWING / np.max(np.abs(xs)))

    # Pull it down and let go: stopped, it rings on its mounts, from there.
    view.grab()  # laid out: the machine has its place on screen
    pb.seek(0.0)
    vb = view.plotItem.vb
    centre = view.machine.rect().center()
    gain = view.gain  # (let go, it's drawn to the new record's scale)
    hand = vb.mapViewToScene(centre)
    to = vb.mapViewToScene(QtCore.QPointF(centre.x(), centre.y() - 0.5 * MOST))
    drag(view, hand, to, app)
    assert not view.held and pb.running and pb.t == 0.0
    assert p.pulled_to == pytest.approx(-0.5 * MOST / gain, rel=0.05)
    t, s = p.speed_item.getData()
    assert not s.any() and p.sx_item.getData()[1][0] == pytest.approx(p.pulled_to)
    assert "let go, stopped" in p.start_note.text()

    # Another tab and back keeps it; a box runs it up again.
    p.tabs.setCurrentIndex(TAB_CHART)
    p.tabs.setCurrentIndex(TAB_START)
    assert p.pulled_to is not None and not p.speed_item.getData()[1].any()
    p.rpm.setValue(1200.0)
    assert p.pulled_to is None and p.speed_item.getData()[1][-1] == pytest.approx(1200.0)


def test_a_trial_with_a_prediction_asks_it_before_it_runs(main_window, app):
    from vib_tutorial.gui import sdof

    w = main_window
    p = w.sdof_page
    w.pages.setCurrentWidget(p)
    box = p.trials
    i = next(k for k, t in enumerate(sdof.TRIALS) if t.name.startswith("Resonance"))
    prediction = sdof.TRIALS[i].predict
    assert prediction is not None
    before = p.practice_state()
    box.combo.pick(sdof.TRIALS[i].name)
    assert box.asking == i and box.ask.isVisible() and "Predict first" in box.question.text()
    assert p.practice_state() == before and not p.playback.running  # nothing set up yet
    buttons = [box.choices.itemAt(k).widget() for k in range(box.choices.count())]
    assert [b.text() for b in buttons] == list(prediction.choices)

    wrong = next(k for k in range(len(prediction.choices)) if k != prediction.answer)
    buttons[wrong].click()
    assert box.asking is None and not box.ask.isVisible() and "Not quite" in box.verdict.text()
    assert prediction.choices[prediction.answer] in box.verdict.text()
    assert p.freq.value() == pytest.approx(p.osc.natural_hz, abs=1e-3) and p.playback.running  # it ran
    assert box.note.isVisible() and "quality factor" in box.note.text()

    box.combo.pick(sdof.TRIALS[i].name)  # again: asked again
    assert box.ask.isVisible() and not box.verdict.isVisible()
    box.answer(prediction.answer)
    assert "Right" in box.verdict.text()

    box.combo.pick(sdof.TRIALS[i].name)
    box.skip.click()  # Just run it: no verdict
    assert not box.verdict.isVisible() and box.note.isVisible()

    plain = next(k for k, t in enumerate(sdof.TRIALS) if t.predict is None)
    box.combo.pick(sdof.TRIALS[plain].name)  # no prediction: runs at once
    assert not box.ask.isVisible() and box.note.text()

    box.combo.pick(sdof.TRIALS[i].name)
    w.restore_defaults()  # put back as it opened: no question, no verdict
    assert not box.ask.isVisible() and not box.verdict.isVisible() and box.asking is None


@pytest.mark.parametrize(
    ("attr", "module"),
    [
        ("transient_page", "transient"),
        ("periodic_page", "periodic"),
        ("damping_page", "damping_models"),
        ("isolation_page", "isolation"),
    ],
)
def test_the_one_mass_pages_ask_their_predictions_first(main_window, app, attr, module):
    import importlib

    trials = importlib.import_module(f"vib_tutorial.gui.{module}").TRIALS
    w = main_window
    p = getattr(w, attr)
    w.pages.setCurrentWidget(p)
    box = p.trials
    asked = [k for k, t in enumerate(trials) if t.predict is not None]
    assert len(asked) >= len(trials) // 2  # most of them ask
    before = p.practice_state()
    for i in asked:
        box.combo.pick(trials[i].name)
        assert box.asking == i and box.ask.isVisible()
        assert p.practice_state() == before  # nothing set up until it's answered
        box.answer(trials[i].predict.answer)
        assert "Right" in box.verdict.text() and box.note.isVisible()
        w.restore_defaults()
        before = p.practice_state()


def test_a_goals_links_set_up_its_trial_and_ask_its_problem(main_window, app):
    """(conftest points the settings at the worker's own file: the Practice window reads them.)"""
    from PySide6 import QtCore as Core

    w = main_window
    browser = w.pendulum_page.notes
    w._wire_links(w.pendulum_page)
    browser.anchorClicked.emit(Core.QUrl("trial:pendulum_page#2"))
    p = w.pendulum_page
    assert w.pages.currentWidget() is p and p.trials.asking == 1 and p.trials.ask.isVisible()
    browser.anchorClicked.emit(Core.QUrl("practice:pendulum_page#large-swing"))
    pw = w.practice_window
    assert pw is not None and pw.problem is not None and pw.problem.key == "large-swing"
