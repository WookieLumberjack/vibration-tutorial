"""Vibration isolation (#478 Phase 4), live: the machine runs on its mounts, a steady speed lands its T, the floor
shaking lands on the same curve, a run up builds the floor force, the grips and a pull change it as it runs, and
predictions come after play."""

from __future__ import annotations

import os

import numpy as np
import pytest
from test_sdof_page import drag

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore  # noqa: E402

from vib_tutorial.core.isolation import Mount  # noqa: E402
from vib_tutorial.gui import isolation  # noqa: E402
from vib_tutorial.view_models import isolation as view  # noqa: E402
from vib_tutorial.view_models.live_isolation import FLOOR, GRIPS, MACHINE  # noqa: E402


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


@pytest.fixture
def page(main_window, app):
    w = main_window
    w.resize(1280, 800)
    w.show_page("isolation_page")
    p = w.isolation_page
    p.loop.enabled = False
    for _ in range(3):
        app.processEvents()
    p.body.grab()  # laid out: the machine and its grips have their places on screen
    return p


def trial(name: str) -> int:
    return next(i for i, t in enumerate(isolation.TRIALS) if t.name.startswith(name))


def points(item) -> int:
    x = item.getData()[0]
    return 0 if x is None else len(x)


def test_it_opens_at_rest_on_its_system_with_things_to_try_after_the_controls(page):
    assert not page.running and page.live.t == 0 and page.live.x == 0
    left = page.trials.parentWidget()
    boxes = [w for w in left.findChildren(QtWidgets.QGroupBox) if w.parentWidget() is left]
    assert [b.title() for b in boxes] == ["Machine", "Mounts", "Things to try", "Challenge"]
    tabs = [page.tabs.tabText(i) for i in range(page.tabs.count())]
    assert tabs == ["Transmissibility", "Floor force", "Design chart", "Theory"]
    assert page.practice_state() == view.IsolationTrial("", "")
    assert 5 <= len(isolation.TRIALS) <= 8
    assert page.live.speed_hz == 0 and page.run_up.isEnabled() and not page.run_down.isEnabled()  # stopped
    assert page.stiffness_ratio.isHidden() and page.stage_mass.isHidden()  # the plain mount's


def test_a_steady_speed_lands_its_transmissibility(page):
    page.load_trial(trial("Soft mounts"))
    assert page.running and page.tabs.currentIndex() == view.TAB_T
    page.step(1.0)
    assert points(page.machine_points) == 0 and "waiting" in page.readout.text()  # still ringing on its mounts
    assert "the unbalance pulls ±247 N" in page.readout.text()  # what drives it, at this speed
    page.step(4.0)
    r, T = (10 ** float(a[0]) for a in page.machine_points.getData())  # drawn on log axes: its data are the logs
    iso = page.iso
    assert r == pytest.approx(iso.speed_hz / iso.natural_hz) and T == pytest.approx(iso.at_speed()[2], rel=1e-3)
    assert "22.6 N reaches the floor: T = 0.0916" in page.readout.text()
    assert points(page.tr_item) == 0  # the theory waits to be asked for
    page.theory.setChecked(True)
    assert points(page.tr_item) > 100 and page.theory_too.isChecked()


def test_the_floor_shaking_lands_on_the_same_curve(page):
    page.load_trial(trial("Soft mounts"))
    page.step(5.0)
    page.shaking.setCurrentIndex(FLOOR)
    assert page.live.shaking == FLOOR
    page.step(1.0)
    assert "the floor moves ±100 µm at 25 Hz" in page.readout.text()
    page.step(5.0)
    (rm,), (Tm,) = page.machine_points.getData()
    (rf,), (Tf,) = page.floor_points.getData()
    assert rf == pytest.approx(rm) and Tf == pytest.approx(Tm, rel=1e-3)
    assert "fall on one curve" in page.tr_note.text() and "the machine ±9.16 µm" in page.readout.text()
    assert page.g_item.isVisible()


def test_a_run_up_builds_the_floor_force_by_speed(page):
    page.load_trial(trial("Running up"))
    assert page.tabs.currentIndex() == view.TAB_FORCE and page.live.ramping and page.live.speed_hz == 0
    assert page.run_up.isChecked() and "Running up" in page.run_up.text()  # lit while it climbs
    page.step(1.0)
    assert page.clock.text().endswith("rpm ▲")
    page.step(2.0)
    rpm, peak = page.built_item.getData()
    peak = 10**peak  # drawn on a log axis: its data are the logs
    assert peak.max() == pytest.approx(111.0, rel=0.02) and rpm[-1] == pytest.approx(1500.0)
    assert "Largest so far: 111 N" in page.force_note.text()
    assert not page.live.ramping and not page.run_up.isEnabled() and not page.run_up.isChecked()
    page.run_down.click()  # it coasts back down through resonance
    page.step(1.0)
    assert page.live.ramping and "Running down" in page.readout.text() and page.run_up.isEnabled()
    assert page.run_down.isChecked() and page.clock.text().endswith("rpm ▼")
    page.run_down.click()  # pressed again: it holds the speed it's reached
    assert not page.live.ramping and 0 < page.live.speed_hz < 25.0


def test_the_grips_change_the_machine_and_its_mounts_as_it_runs(page, app):
    page.load_trial(trial("Soft mounts"))
    page.step(0.3)
    body = page.body
    for name, box, delta in (
        ("deflection_mm", page.deflection, QtCore.QPointF(0.0, -0.2)),  # pulled down: it sags more
        ("zeta", page.zeta, QtCore.QPointF(0.0, 0.2)),
        ("mass", page.mass, QtCore.QPointF(0.1, 0.1)),
    ):
        before = box.value()
        at = body.press_point(name)
        drag(body, at, at + delta, app)
        assert box.value() > before, name
        distance = -delta.y() if name == "deflection_mm" else delta.x() + delta.y() if name == "mass" else delta.y()
        assert box.value() == pytest.approx(GRIPS[name].dragged(before, distance), rel=0.05), name
    s = page.practice_state()
    assert page.live.iso == view.isolator(s) and page.running and page.live.edits


def test_the_page_opens_stopped_so_a_pull_rings_freely(page, app):
    body = page.body
    centre = body.machine.rect().center()
    drag(body, centre, centre + QtCore.QPointF(0.0, 0.15), app)
    assert page.running and page.live.held_at is None and page.live.x > 0
    page.step(0.5)
    assert np.abs(page.x_item.getData()[1]).max() > 0
    assert page.live.speed_hz == 0 and not page.unbalance_item.getData()[1].any()  # no unbalance turning
    page.rpm.setValue(900.0)  # stopped, the box is where Run up goes
    assert page.live.speed_hz == 0
    page.run_up.click()
    page.step(page.ramp.value() * 900 / 900 + 0.1)
    assert page.live.speed_hz == pytest.approx(15.0) and not page.run_up.isChecked()


def test_a_change_of_kind_shows_its_boxes_and_carries_on(page):
    page.load_trial(trial("Soft mounts"))
    page.step(0.5)
    page.mount.setCurrentIndex(view.MOUNTS.index(Mount.TWO_STAGE))
    assert not page.stage_mass.isHidden() and page.live.iso.mount is Mount.TWO_STAGE and page.running
    page.step(0.5)
    assert page.body.plate.isVisible()


def test_a_sweep_lands_its_speeds_and_stops(page):
    page.load_trial(trial("Soft mounts"))
    page.sweep_button.setChecked(True)
    while page.sweep is not None and page.live.t < 200.0:
        page.step(2.0)
    assert page.sweep is None and not page.sweep_button.isChecked()
    assert points(page.machine_points) == view.SWEEP_POINTS


def test_predictions_come_after_play(page):
    box = page.trials
    page.load_trial(trial("Soft mounts"))
    page.zeta.setValue(0.1)  # the reader handles it ...
    page.step(isolation.PLAYED_AFTER + 0.1)  # ... and watches
    assert box.played
    asked = trial("Two stages")
    box.combo.pick(isolation.TRIALS[asked].name)
    assert box.asking == asked and box.ask.isVisible()
    box._on_skip()
    assert box.asking is None and page.running


def test_scrubbed_back_the_machine_is_drawn_where_it_was(page):
    page.load_trial(trial("Running up"))
    page.step(1.5)
    page._run(False)
    page._on_scrubbed(0.8)
    r = page.live.record
    i = int(np.searchsorted(r.t[: r.size], 0.8))
    assert page.body.state[0] == pytest.approx(r.y[i, 0], rel=0.05, abs=1e-6)
    assert "paused at" in page.clock.text()


def test_reset_and_defaults(page, main_window):
    page.load_trial(trial("The floor shakes"))
    page.step(0.5)
    page.reset()
    assert page.live.t == 0 and not page.running and page.live.shaking == FLOOR
    main_window.restore_defaults()
    assert page.practice_state().shaking == MACHINE and page.live.shaking == MACHINE
