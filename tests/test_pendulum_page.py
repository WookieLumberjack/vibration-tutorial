"""Pendulums, rebuilt live (#478 Phase 4): the pendulum swings as it's handled, its grips change it mid-swing with
its angular momentum kept, the period is built swing by swing, the moments are drawn beside it, and predictions
come after play."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from test_sdof_page import drag

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore  # noqa: E402

from vib_tutorial.core.pendulum import Kind, Pendulum, period  # noqa: E402
from vib_tutorial.gui import pendulum  # noqa: E402
from vib_tutorial.view_models import pendulum as view  # noqa: E402
from vib_tutorial.view_models.live_pendulum import COL  # noqa: E402


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


@pytest.fixture
def page(main_window, app):
    w = main_window
    w.resize(1280, 800)
    w.show_page("pendulum_page")
    p = w.pendulum_page
    p.loop.enabled = False
    for _ in range(3):
        app.processEvents()
    p.view.grab()  # laid out: its grips have their places on screen
    return p


def points(item) -> int:
    x = item.getData()[0]
    return 0 if x is None else len(x)


def trial(name: str) -> int:
    return next(i for i, t in enumerate(pendulum.TRIALS) if t.name.startswith(name))


def test_it_opens_hanging_at_rest_with_things_to_try_after_its_boxes(page):
    assert not page.running and page.live.t == 0 and page.live.theta == 0
    left = page.trials.parentWidget()
    boxes = [w for w in left.findChildren(QtWidgets.QGroupBox) if w.parentWidget() is left]
    assert [b.title() for b in boxes] == ["Pendulum", "Things to try"]
    assert [page.tabs.tabText(i) for i in range(page.tabs.count())] == ["Period", "Phase plane", "Energy", "Theory"]
    assert page.practice_state() == view.PendulumTrial("", "")
    assert page.pivot.isHidden() and page.spring.isHidden() and not page.gravity.isHidden()
    assert page.period_note.text().startswith("Pluck it") and not points(page.points_item)
    assert 5 <= len(pendulum.TRIALS) <= 8


def test_a_pluck_on_the_bob_holds_it_then_lets_it_swing_and_each_swing_lands_its_period(page, app):
    v = page.view
    to = QtCore.QPointF(*v._point(-math.radians(40.0)))
    drag(v, v.handle_point(), to, app)
    assert page.running and page.live.held is None
    assert page.live.theta == pytest.approx(-math.radians(40.0), abs=math.radians(1.5))
    assert page.live.released > 0  # the hand's work, booked
    page.step(3 * period(Pendulum(), math.radians(40.0)))
    landed = page.recorder.current(view.key(page.pendulum))
    assert len(landed) == 1  # an undamped swing lands one point, again and again
    assert landed[0].period == pytest.approx(period(Pendulum(), landed[0].amplitude), rel=1e-4)
    x, y = page.points_item.getData()
    assert x[0] == pytest.approx(math.degrees(landed[0].amplitude)) and y[0] == landed[0].period
    assert points(page.ghost_item) and page.view.ghost.isVisible()
    assert page.readout.text().startswith("The last swing, ±")


def test_shortening_the_string_by_its_grip_keeps_the_angular_momentum(page, app):
    page.load_trial(trial("A quarter"))  # 0.25 m, from 10°
    page.length.setValue(1.0)  # a 1 m string, swinging on
    page.step(0.47)  # near the bottom
    w, t = page.live.omega, page.live.t
    at = page.view.press_point("length")
    drag(page.view, at, at * 0.5, app)  # the ring slid in to half its distance: half the string
    L = page.length.value()
    assert L == pytest.approx(0.5, rel=0.02) and page.pendulum.length == L and page.live.t == t
    assert page.live.omega == pytest.approx(w / L**2, rel=1e-9)  # J = mL²: a quarter, four times θ̇
    assert page.live.changes > 0 and page.live.edits[-1] == t  # the hand's pull, booked; the change marked
    page.tabs.setCurrentIndex(view.TAB_ENERGY)
    page.step(1.0)
    assert "Changing the pendulum" in page.energy.ledger.text()


def test_the_grips_set_m_k_and_the_rods_pivot_as_it_swings(page, app):
    page.load_trial(trial("Inverted, let go"))
    page.step(0.3)
    v = page.view
    for name, box in (("mass", page.mass), ("spring", page.spring)):
        before = box.value()
        at = v.press_point(name)
        drag(v, at, at + QtCore.QPointF(0.0, 0.3 * v.reach), app)
        assert box.value() == pytest.approx(view.GRIPS[name].dragged(before, 0.3), rel=0.05), name
    assert (page.pendulum.mass, page.pendulum.spring) == (page.mass.value(), page.spring.value())
    page.load_trial(trial("A rod"))
    page.step(0.2)
    before = page.pivot.value()
    ux, uy = v._point(v.theta, 1.0 / v.reach)  # along the rod, a meter
    drag(v, v.press_point("pivot"), QtCore.QPointF(-0.2 * ux, -0.2 * uy), app)  # 0.2 m toward its middle
    assert page.pivot.value() == pytest.approx(before - 0.2, abs=0.01)
    assert page.running and len(page.live.edits) >= 1


def test_the_period_tabs_axes_keep_still_when_the_theory_is_shown(page):
    page.load_trial(trial("Nearly over the top"))
    page.step(40.0)
    assert len(page.recorder.current(view.key(page.pendulum))) > 8  # a point a swing as it dies away
    ranges = page.period_plot.viewRange()
    page.theory.setChecked(True)
    assert np.allclose(page.period_plot.viewRange(), ranges)
    assert points(page.exact_item)
    page.clear_button.click()
    assert not points(page.points_item)


def test_the_moments_beside_it_and_the_strings_pull_under_them(page):
    rows = page.forces.rows()
    assert [s for s, _ in rows] == ["mg", "cθ̇", "Jθ̈", "Tₛ"]
    page.load_trial(trial("A disk"))
    page.forces._write_sizes()
    assert [s for s, _ in page.forces.rows()] == ["kₜθ", "cθ̇", "Jθ̈", "Tₛ"]
    assert all(z.endswith("N·m") for _, z in page.forces.rows()[:3]) and page.forces.rows()[3][1].endswith("N")


def test_predictions_come_after_play(page):
    box = page.trials
    asked = trial("Let go from 90")
    box.combo.pick(pendulum.TRIALS[asked].name)  # before any play: it runs at once
    assert box.asking is None and page.running
    page.load_trial(trial("A quarter"))
    page.length.setValue(0.5)  # the reader handles it ...
    page.step(pendulum.PLAYED_AFTER + 0.1)  # ... and watches
    assert box.played
    unasked = trial("Pump")
    box.combo.pick(pendulum.TRIALS[unasked].name)
    assert box.asking == unasked and box.ask.isVisible()
    box._on_skip()
    assert box.asking is None


def test_an_upright_pendulum_on_a_weak_spring_falls_over_and_leans(page):
    page.load_trial(trial("Inverted, on a weaker"))
    page.step(60.0)
    lean = Pendulum(Kind.INVERTED, spring=9.0, zeta=0.1).leaning()
    assert abs(page.live.theta) == pytest.approx(lean, abs=0.01)
    assert "Unstable upright" in page.readout.text() and page.tabs.currentIndex() == view.TAB_PHASE
    assert points(page.path_item) and not points(page.separatrix_item)


def test_scrubbed_back_the_pendulum_is_drawn_where_it_was(page):
    page.load_trial(trial("Let go from 90"))
    page.step(3.0)
    page._on_run(False)
    r = page.live.record
    t = page.live.t - 1.0
    page._on_scrubbed(t)
    assert page.view.theta == pytest.approx(np.interp(t, r.t[: r.size], r.y[: r.size, COL["theta"]]))
    assert "paused" in page.clock.text()
