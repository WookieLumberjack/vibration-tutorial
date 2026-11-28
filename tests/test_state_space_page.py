"""State-space modes (#478 Phase 4, after Coordinates & bases): the two masses run live, a start sets them going, a
drag on a coordinate plane sets that pair and keeps the other, the basis switches as it runs, the grips change m, k
and c, the matrices mark the coupling, and predictions come after play."""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest
from test_sdof_page import drag

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore  # noqa: E402

from vib_tutorial.gui import state_space  # noqa: E402
from vib_tutorial.view_models import state_space as view  # noqa: E402


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


@pytest.fixture
def page(main_window, app):
    w = main_window
    w.resize(1280, 800)
    w.show_page("state_page")
    p = w.state_page
    p.loop.enabled = False
    for _ in range(3):
        app.processEvents()
    for plane in (p.shape_plane, *p.pair_planes, p.chain):
        plane.grab()  # laid out: the planes and the masses have their places on screen
    return p


def points(item) -> int:
    x = item.getData()[0]
    return 0 if x is None else len(x)


def trial(name: str) -> int:
    return next(i for i, t in enumerate(state_space.TRIALS) if t.name.startswith(name))


def pairs_now(page) -> np.ndarray:
    return view.pairs(page._basis(), page.live.x, page.live.v)[0]


def test_it_opens_at_rest_on_its_system_with_things_to_try_after_the_controls(page):
    assert not page.running and page.live.t == 0 and not page.live.x.any()
    left = page.trials.parentWidget()
    boxes = [w for w in left.findChildren(QtWidgets.QGroupBox) if w.parentWidget() is left]
    assert [b.title() for b in boxes] == ["Two masses", "The planes' coordinates", "Start from", "Things to try"]
    assert [page.tabs.tabText(i) for i in range(page.tabs.count())] == ["Matrices", "Sizes", "Theory"]
    assert page.practice_state() == view.StateTrial("", "")
    assert "dampers still do" in page.matrices_note.text() and "f = 2.421 Hz" in page.matrices.text()
    assert [p.labels.title for p in page.pair_planes] == ["Mode 1 of M and K", "Mode 2 of M and K"]
    assert [b.text() for b in page.start_buttons] == list(view.STARTS[: view.BUTTONS])
    assert page.start_note.isHidden()
    assert 5 <= len(state_space.TRIALS) <= 8


def test_a_start_in_a_state_space_mode_leaves_the_other_pair_still(page):
    page.basis.setCurrentIndex(view.STATE)
    page.start_buttons[3].click()  # state-space mode 2
    assert page.running and page.live.held is None
    page.step(1.0)
    w = pairs_now(page)
    assert np.hypot(w[0], w[1]) < 1e-9 * np.hypot(w[2], w[3])
    assert all(points(item) for item in page.w_items)
    assert page.start_note.text().startswith("State-space mode 2: x = (30, -5.63) mm")
    assert "On the planes: (0, 0) and (30.5, 0) mm." in page.start_note.text()
    page.basis.setCurrentIndex(view.UNDAMPED)  # switched as it runs: the motion is the same, its description not
    assert "On the planes: (-0.0466, 0.991)" in page.start_note.text()
    w = pairs_now(page)
    assert np.hypot(w[0], w[1]) > 1e-3
    assert page.running


def test_dragging_a_pair_sets_it_and_keeps_the_other(page, app):
    page.basis.setCurrentIndex(view.STATE)
    page.start(3)
    page.step(0.2)
    page._run(False)
    before = pairs_now(page)
    plane = page.pair_planes[0]
    hi = plane.viewRange()[0][1]
    drag(plane, QtCore.QPointF(0.0, 0.0), QtCore.QPointF(0.4 * hi, 0.2 * hi), app)
    after = pairs_now(page)
    assert after[:2] == pytest.approx([0.4 * hi, 0.2 * hi], rel=0.05, abs=0.05)
    assert after[2:] == pytest.approx(before[2:], rel=1e-6)
    assert page.running and page.live.held is None and page.live.books["released"] != 0


def test_a_pluck_on_the_masses_plane_holds_them_there_then_lets_them_run(page, app):
    drag(page.shape_plane, QtCore.QPointF(2.0, 2.0), QtCore.QPointF(20.0, -10.0), app)
    assert page.running and page.live.held is None
    assert page.live.x * 1e3 == pytest.approx([20.0, -10.0], abs=0.5)
    page.step(0.5)
    assert page.readout.text().startswith("x = (")


def test_the_grips_change_m_k_and_c_as_it_runs(page, app):
    page.load_trial(trial("The dampers couple the modes"))
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


def test_the_matrices_mark_the_coupling_by_basis(page):
    marked = {}
    for b in (view.MASS, view.UNDAMPED, view.STATE):
        page.basis.setCurrentIndex(b)
        marked[b] = page.matrices.text().count("font-weight:bold")
    # the springs' and dampers' terms in each velocity row; the dampers' alone; none
    assert marked == {view.MASS: 4, view.UNDAMPED: 2, view.STATE: 0}
    page.tabs.setCurrentIndex(view.TAB_MATRICES)
    html = page.matrices.text()
    assert "<th style='white-space:nowrap'>ẋ₁</th>" in html and "<th style='white-space:nowrap'>w₄</th>" in html
    wide, tall = page.matrices._variants
    page.matrices.resize(2000, page.matrices.height())
    assert page.matrices.text() == wide  # room: T and T⁻¹AT side by side
    page.matrices.resize(300, page.matrices.height())
    assert page.matrices.text() == tall  # none: one above the other, nothing cut off
    page.m2.setValue(1.0)  # a symmetric pair: the modes of M and K uncouple the dampers too
    page.basis.setCurrentIndex(view.UNDAMPED)
    assert "font-weight:bold" not in page.matrices.text()
    page.c.setValue(30.0)  # mode 2 overdamped
    page.basis.setCurrentIndex(view.STATE)
    assert page.pair_planes[1].labels.title == "State-space mode 2: overdamped"


def test_the_sizes_tab_shows_the_dampers_coupling(page):
    page.load_trial(trial("The dampers couple the modes"))
    assert page.tabs.currentIndex() == view.TAB_SIZES
    page.step(1.0)
    passed = 10 ** page.size_items[0].getData()[1]  # drawn on a log axis: its data are the logs
    assert passed[0] < 0.01 and passed.max() > 1.0  # mm: mode 1 of M and K climbs from nothing
    assert all(points(item) for item in page.size_items)


def test_the_matrices_tab_and_the_plane_show_complex_shapes(page):
    page.load_trial(trial("Heavier dampers"))
    assert page.tabs.currentIndex() == view.TAB_MATRICES
    assert "Complex: m₂ lags m₁ by 18.8°" in page.own_notes[0].text()
    (_, b), (c, d) = (page.own_items[0][i].getData() for i in range(2))
    assert b[1] == pytest.approx(0.0, abs=1e-12)  # m₁ along +Re
    assert np.degrees(np.arctan2(d[1], c[1])) == pytest.approx(-18.77, abs=0.01)
    assert "ellipse" in page.plane_note.text()


def test_predictions_come_after_play(page):
    box = page.trials
    asked = trial("The dampers couple the modes")
    page.load_trial(trial("Each mass"))
    page.k2.setValue(150.0)  # the reader handles it ...
    page.step(state_space.PLAYED_AFTER + 0.1)  # ... and watches
    assert box.played
    box.combo.pick(state_space.TRIALS[asked].name)
    assert box.asking == asked and box.ask.isVisible()
    box._on_skip()
    assert box.asking is None and page.running


def test_scrubbed_back_the_planes_are_drawn_where_they_were(page):
    page.load_trial(trial("Each mass"))
    page.step(1.0)
    page._run(False)
    page._on_scrubbed(0.5)
    r = page.live.record
    i = int(np.searchsorted(r.t[: r.size], 0.5))
    x1 = r.y[i, 0] * 1e3
    assert page.shape_plane.point.pos().x() == pytest.approx(x1, abs=0.5)
    assert "paused at" in page.clock.text()


def test_reset_and_defaults(page, main_window):
    page.load_trial(trial("Heavier dampers"))
    page.step(0.5)
    page.reset()
    assert page.live.t == 0 and not page.running
    main_window.restore_defaults()
    assert dataclasses.replace(page.practice_state(), tab=0) == view.StateTrial("", "")  # Defaults leaves the tab


def test_your_own_shapes_are_dragged_on_the_matrices_tab_and_snap_onto_As(page, app):
    assert not page.own_box.isHidden() and page.own_reset.isHidden()  # each basis's shapes, read only
    assert not page.own_dots[0].isVisible() and page.own_gauges[0].isHidden()
    page.load_trial(trial("Find A's modes by hand"))
    assert page.which == view.OWN and page.running and not page.own_reset.isHidden()
    page.tabs.setCurrentIndex(view.TAB_MATRICES)
    page.own_plots[0].grab()
    app.processEvents()
    assert np.abs(pairs_now(page)[2:]).max() > 0.1  # only A's mode 1 runs, but your mode 2's plane moves
    s = page.system
    true = view.ratio(view.modes(s)[0])
    start = view.undamped_ratios(s)[0]
    away = start + 1.5j
    drag(page.own_plots[0], QtCore.QPointF(start.real, start.imag), QtCore.QPointF(away.real, away.imag), app)
    assert page.own is not None and page.own[0] == pytest.approx(away, abs=0.05) and page.own_dots[0].pos().y() > 1
    assert "passes motion to mode 2's" in page.own_notes[0].text()
    drag(page.own_plots[0], QtCore.QPointF(away.real, away.imag), QtCore.QPointF(true.real, true.imag), app)
    assert page.own is not None and page.own[0] == true  # snapped on
    assert page.own[1] == view.undamped_ratios(s)[1]
    assert "A's own mode 1" in page.own_notes[0].text() and page.own_gauges[0].value() == 100
    lit = page.matrices.text()
    assert "<th style='white-space:nowrap; color:" in lit  # the dragged pair's columns lit in T and T⁻¹AT
    assert lit.index("T⁻¹AT") < lit.index("Where A comes from")  # the basis first, under its shapes
    page.step(0.5)
    assert np.abs(pairs_now(page)[2:]).max() < 1e-6  # your mode 2's plane falls still
    state = page.practice_state()
    assert view.own_of(state) == page.own
    page.own_reset.click()
    assert page.own is None and "at 0°" in page.own_notes[0].text()
    page.apply_trial(state)  # a setup with shapes puts them back
    assert page.own == view.own_of(state)


def test_the_basis_written_complex_shows_y_and_counts_as_no_handling(page):
    page.basis.setCurrentIndex(view.STATE)
    assert "T⁻¹AT" in page.matrices.text()
    page._handled_at = None
    page.written.setCurrentIndex(view.COMPLEX)
    html = page.matrices.text()
    assert "Y⁻¹AY" in html and "η̄₁" in html and "T⁻¹AT" not in html and "conjugate" in html
    assert page.own_columns[0].text().startswith("Y's η₁ and η̄₁")
    assert page._handled_at is None  # how it's written isn't handling the system
    assert page.practice_state().written == view.COMPLEX
