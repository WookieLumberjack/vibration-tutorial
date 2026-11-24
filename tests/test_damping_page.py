"""Damping models, rebuilt live (#478 Phase 3's pilot, its second page): one mass with a damping model picked as it
runs, the viscous ghost beside it, the loop the run draws, the phasors and the frequency response built by hand."""

from __future__ import annotations

import os

import numpy as np
import pytest
from notes_quoting import quoted

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from vib_tutorial.core.damping_models import MODELS, Model  # noqa: E402
from vib_tutorial.gui import damping_models  # noqa: E402
from vib_tutorial.view_models import damping as view  # noqa: E402


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


@pytest.fixture
def page(main_window, app):
    w = main_window
    w.resize(1280, 800)
    w.show_page("damping_page")
    p = w.damping_page
    p.loop.enabled = False
    for _ in range(3):
        app.processEvents()
    return p


def trial(name: str) -> int:
    return next(i for i, t in enumerate(damping_models.TRIALS) if t.name.startswith(name))


def test_it_opens_at_rest_with_things_to_try_after_the_controls(page):
    assert not page.running and page.live.t == 0
    left = page.trials.parentWidget()
    boxes = [w for w in left.findChildren(QtWidgets.QGroupBox) if w.parentWidget() is left]
    assert [b.title() for b in boxes] == ["Oscillator", "Damping", "Drive", "Things to try"]
    assert page.tabs.count() == 4
    assert page.practice_state() == view.DampingTrial("", "")
    assert page.amplitude.isHidden() and page.relax.isHidden()  # friction's X and the Zener peak: their models'


def test_four_loops_one_energy_and_the_readouts_the_worked_example_quotes(page):
    page.load_trial(trial("Four loops"))
    page.step(12.0)
    assert "Steady at 3.18 Hz: x swings ±10 mm" in page.readout.text()
    note = page.loop_note.text()
    assert f"a viscous damper's πcΩX² there is {quoted('damping_models', '12.6')} mJ" in note
    enclosed = float(note.split("encloses ")[1].split(" mJ")[0])
    assert enclosed == pytest.approx(12.57, rel=0.01)  # the run's own loop
    drawn = [m for m, item in page.theory_loops.items() if item.getData()[0] is not None and len(item.getData()[0])]
    assert drawn == list(MODELS)


def test_the_model_changes_as_it_runs_and_the_books_close(page):
    page.load_trial(trial("Four loops"))
    page.step(3.0)
    t = page.live.t
    for i, model in enumerate(MODELS[1:], 1):
        page.model.setCurrentIndex(i)  # a live change: no restart
        assert page.live.t == t and page.practice_state().model is model
        page.step(1.0)
        t = page.live.t
    assert len(page.live.edits) == 3
    from vib_tutorial.view_models.live_mass import energy_view

    r = page.live.record
    books = energy_view(r.t[: r.size], r.y[: r.size])
    assert books is not None and abs(books.residual) < 1e-9 * max(abs(row.value) for row in books.rows)
    page.tabs.setCurrentIndex(view.TAB_ENERGY)
    assert "It closes to" in page.energy.ledger.text()


def test_friction_sticks_off_centre_beside_the_ringing_ghost(page):
    page.load_trial(trial("Released against friction"))
    page.step(4.0)
    text = page.readout.text()
    assert f"Each peak is {quoted('damping_models', '3.14')} mm lower" in text
    assert "Stuck 575 µm off centre" in text  # the worked example's 575 µm
    assert page.live.v == 0.0 and page.ghost.v != 0.0  # the viscous ghost rings on
    assert page.ghost_item.isVisible() and len(page.ghost_item.getData()[0])
    assert page.body.ghost.isVisible()
    assert "Friction F<sub>f</sub> = πcω<sub>n</sub>X/4 = 314 mN" in page.info.text()  # the worked example's


def test_the_zener_info_and_its_live_decay(page):
    page.load_trial(trial("A viscoelastic loss peak"))
    assert f"τ = {quoted('damping_models', '45.2')} ms" in page.info.text()
    assert page.live.params.has_arm and not page.relax.isHidden()
    page.step(2.0)
    assert page.readout.text().startswith("Each peak is")
    names = [a.name for a in page.forces._arrows]
    assert "arm" in names and "damper" not in names


def test_the_hysteretic_stand_in_swept_lands_on_its_curve(page):
    page.load_trial(trial("Hysteretic damping's stand-in"))
    page.step(120.0)
    key = page._key()
    points = page.recorder.current(key)
    assert len(points) >= 14 and page.sweep is None  # the sweep's 15 points, done
    m = view.matched(page.practice_state())
    for p in points:
        h = complex(m.receptance(Model.HYSTERETIC, np.array([p.freq_hz]), 1.0)[0])
        assert abs(p.response) == pytest.approx(abs(h), rel=0.02), p.freq_hz


def test_the_phasors_close_on_the_drive(page):
    page.load_trial(trial("Four loops"))
    page.tabs.setCurrentIndex(view.TAB_FRF)
    page.step(2.0)
    xs, ys = page.phasor_items["drive"].getData()
    assert np.hypot(xs[1], ys[1]) == pytest.approx(0.4)
    mx, my = page.phasor_items["mass"].getData()
    assert (mx[1], my[1]) == pytest.approx((xs[1], ys[1]))


def test_predictions_come_after_play_and_defaults_round_trips(page):
    box = page.trials
    box.combo.pick(damping_models.TRIALS[trial("Twice")].name)
    assert box.asking is None  # not yet played: runs at once
    state = page.practice_state()
    page.load_setup({k: v for k, v in state.__dict__.items() if k not in ("name", "note")})
    assert page.practice_state() == state
    page.zeta.setValue(0.08)
    page.step(damping_models.PLAYED_AFTER + 0.1)
    assert box.played and box.invite.isVisible()
