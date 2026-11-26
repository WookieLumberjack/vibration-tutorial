"""Coordinates & bases (#478 Phase 4's new page): the two masses run live, a pluck on the plane or a mass sets them
going, the arrows turn the description and snap onto the modes, the grips change m and k as they run, the rail's
frequency is built by hand, and predictions come after play."""

from __future__ import annotations

import os

import numpy as np
import pytest
from test_sdof_page import drag

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore  # noqa: E402

from vib_tutorial.core.bases import TwoMass, angle, direction, rayleigh_hz  # noqa: E402
from vib_tutorial.gui import bases  # noqa: E402
from vib_tutorial.view_models import bases as view  # noqa: E402
from vib_tutorial.view_models.live_bases import COL  # noqa: E402


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


@pytest.fixture
def page(main_window, app):
    w = main_window
    w.resize(1280, 800)
    w.show_page("bases_page")
    p = w.bases_page
    p.loop.enabled = False
    for _ in range(3):
        app.processEvents()
    p.plane.grab()  # laid out: its arrows and the masses have their places on screen
    p.chain.grab()
    return p


def points(item) -> int:
    x = item.getData()[0]
    return 0 if x is None else len(x)


def trial(name: str) -> int:
    return next(i for i, t in enumerate(bases.TRIALS) if t.name.startswith(name))


def test_it_opens_at_rest_along_the_axes_with_things_to_try_after_its_boxes(page):
    assert not page.running and page.live.t == 0 and not page.live.x.any()
    left = page.trials.parentWidget()
    boxes = [w for w in left.findChildren(QtWidgets.QGroupBox) if w.parentWidget() is left]
    assert [b.title() for b in boxes] == ["Two masses", "Basis", "Things to try"]
    assert [page.tabs.tabText(i) for i in range(page.tabs.count())] == [
        "Coupling",
        "One arrow",
        "Energy",
        "Theory",
    ]
    assert page.practice_state() == view.BasesTrial("", "")
    assert "static coupling" in page.plane_note.text() and "3.183 and 3.898 Hz" in page.matrices.text()
    assert 5 <= len(bases.TRIALS) <= 8


def test_a_pluck_on_the_plane_holds_the_masses_there_then_lets_them_run(page, app):
    v = page.plane
    drag(v, QtCore.QPointF(5.0, 5.0), QtCore.QPointF(30.0, -10.0), app)
    assert page.running and page.live.held is None
    assert page.live.x * 1e3 == pytest.approx([30.0, -10.0], abs=0.5)
    assert page.live.books["released"] > 0
    page.step(1.0)
    assert points(page.q_items[0]) and points(page.q_items[1])
    assert page.readout.text().startswith("x = (")


def test_a_mass_dragged_on_the_chain_moves_alone(page, app):
    c = page.chain
    at = c.press_point(1)
    drag(c, at, at + QtCore.QPointF(0.12, 0.0), app)  # 0.12 display units: 20 mm
    assert page.live.x == pytest.approx([0.0, 0.02], abs=2e-3)  # to the pixel


def test_an_arrow_snaps_onto_a_mode_and_the_coordinates_uncouple(page, app):
    page.load_trial(trial("Pluck one mass"))
    page.step(2.0)
    v = page.plane
    # Arrow 1 dragged to 2° off mode 1 (45°): it snaps on; arrow 2 to 2° off mode 2 (135°).
    drag(v, v.press_point(0), QtCore.QPointF(*(view.ARROW * 1e3 * direction(47.0))), app)
    drag(v, v.press_point(1), QtCore.QPointF(*(view.ARROW * 1e3 * direction(133.0))), app)
    assert angle(np.array(page.v1)) == pytest.approx(45.0) and angle(np.array(page.v2)) == pytest.approx(135.0)
    assert page.plane_note.text().count("Both diagonal") == 1
    page.step(0.5)
    # The past redescribed: q is the modes' coordinates over the whole window, each one sine.
    _, q1 = page.q_items[0].getData()
    assert np.ptp(q1) > 0 and page.shares_note.text().startswith("Uncoupled")


def test_the_grips_change_m_k_and_c_as_it_runs(page, app):
    page.load_trial(trial("Pluck one mass"))
    page.c.setValue(1.0)
    page.step(0.3)
    c = page.chain
    for name in ("m2", "k2", "c"):
        before = getattr(page, name).value()
        at = c.press_point(name)
        drag(c, at, at + QtCore.QPointF(0.0, 0.2), app)
        assert getattr(page, name).value() == pytest.approx(view.GRIPS[name].dragged(before, 0.2), rel=0.05), name
    s = page.system
    assert (s.m2, s.k2, s.c) == (page.m2.value(), page.k2.value(), page.c.value())
    assert page.running and page.live.edits


def test_on_a_rail_each_angle_lands_its_frequency(page, app):
    page.load_trial(trial("One arrow along"))
    page.step(1.0)
    k = view.key(page.system)
    assert [p.angle for p in page.recorder.current(k)] == [0.0]
    v = page.plane
    drag(v, v.press_point(0), QtCore.QPointF(*(view.ARROW * 1e3 * direction(30.0))), app)
    assert angle(page.live.rail) == pytest.approx(30.0, abs=0.5)
    page.step(1.5)
    landed = {round(p.angle): p.hz for p in page.recorder.current(k)}
    M, _, K = TwoMass().matrices()
    assert landed[30] == pytest.approx(rayleigh_hz(page.live.rail, M, K), rel=1e-3)
    ranges = page.rail_plot.viewRange()
    page.theory.setChecked(True)
    assert np.allclose(page.rail_plot.viewRange(), ranges) and points(page.quotient_item)
    page.rail.setChecked(False)  # off the rail: it runs on free
    assert page.live.rail is None and page.live.steps.n == 2


def test_predictions_come_after_play(page):
    box = page.trials
    asked = trial("Pluck it along")
    box.combo.pick(bases.TRIALS[asked].name)  # before any play: it runs at once
    assert box.asking is None and page.running
    page.load_trial(trial("Pluck one mass"))
    page.k2.setValue(150.0)  # the reader handles it ...
    page.step(bases.PLAYED_AFTER + 0.1)  # ... and watches
    assert box.played
    unasked = trial("One arrow along")
    box.combo.pick(bases.TRIALS[unasked].name)
    assert box.asking == unasked and box.ask.isVisible()
    box._on_skip()
    assert box.asking is None


def test_scrubbed_back_the_masses_are_drawn_where_they_were(page):
    page.load_trial(trial("Pluck one mass"))
    page.step(2.0)
    page._on_run(False)
    r = page.live.record
    t = page.live.t - 0.7
    page._on_scrubbed(t)
    assert page.chain.x[1] == pytest.approx(np.interp(t, r.t[: r.size], r.y[: r.size, COL["x2"]]))
    assert "paused" in page.clock.text()


def test_a_running_drag_redraws_the_matrices_now_and_then_and_exactly_on_letting_go(page, app):
    """Dragged as it runs, the arrow is drawn in the frames and the matrices' steps at most every
    `MATRICES_EVERY`; letting go draws them for where it stopped."""
    page.load_trial(trial("Pluck one mass"))
    page.step(0.5)
    assert page.running
    page.plane.held = 0  # v₁ in hand
    page._on_arrow_moved(0, *(view.ARROW * direction(30.0)))
    page._tick(1 / 60)  # drawn in the frame: the arrow, and the matrices (none drawn this drag yet)
    before = page.matrices.text()
    page._on_arrow_moved(0, *(view.ARROW * direction(20.0)))
    page._tick(1 / 60)
    assert angle(np.array(page.plane.tips[0])) == pytest.approx(20.0, abs=0.01), "the arrow, every frame"
    assert page.matrices.text() == before, "the matrices wait"
    page.plane.held = False
    page._on_arrow_released(0)
    assert page.matrices.text() != before, "let go: drawn where it stopped"
    assert page.matrices.text() == page._matrices_html(view.matrices_view(page.live.system, page.basis, False))
