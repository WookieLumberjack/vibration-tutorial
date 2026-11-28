"""Forcing an oscillator (#478 Phase 4), live: Transient response and Periodic forcing as one page. A hit and a second
add, a push held and let go, a pulse lands its peak on the spectrum, a wave's harmonics land once steady, the
convolution at a scrubbed instant is the run, the grips change m, k and c as it runs, predictions come after play,
and Defaults puts it back."""

from __future__ import annotations

import dataclasses
import os
import re

import numpy as np
import pytest
from test_sdof_page import drag

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore  # noqa: E402

from vib_tutorial.core.sdof import impulse_response  # noqa: E402
from vib_tutorial.core.transient import PulseKind  # noqa: E402
from vib_tutorial.gui import forcing  # noqa: E402
from vib_tutorial.view_models import forcing as view  # noqa: E402


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


@pytest.fixture
def page(main_window, app):
    w = main_window
    w.resize(1280, 800)
    w.show_page("forcing_page")
    p = w.forcing_page
    p.loop.enabled = False
    for _ in range(3):
        app.processEvents()
    p.body.grab()  # laid out: the mass and its grips have their places on screen
    return p


def trial(name: str) -> int:
    return next(i for i, t in enumerate(forcing.TRIALS) if t.name.startswith(name))


def points(item) -> int:
    x = item.getData()[0]
    return 0 if x is None else len(x)


def test_it_opens_at_rest_on_its_system_with_things_to_try_after_the_controls(page):
    assert not page.running and page.live.t == 0 and page.live.x == 0
    left = page.trials.parentWidget()
    boxes = [w for w in left.findChildren(QtWidgets.QGroupBox) if w.parentWidget() is left]
    assert [b.title() for b in boxes] == ["Oscillator", "A force", "A periodic force", "Things to try"]
    tabs = [page.tabs.tabText(i) for i in range(page.tabs.count())]
    assert tabs == ["Convolution", "Pulse spectrum", "Harmonics", "Theory"]
    assert page.practice_state() == view.ForcingTrial("", "")
    assert 5 <= len(forcing.TRIALS) <= 8
    assert not page.let_go_button.isEnabled() and not page.wave_on.isChecked()
    assert page.duration.isHidden()  # a step has no length


def test_a_second_hit_lands_on_the_first_ones_ring(page):
    page.load_trial(trial("A blow"))
    assert page.running and len(page.live.hits) == 1
    page.step(0.4)
    page.apply_force()  # again, as it rings
    page.step(0.6)
    r = page.live.record
    t, x = r.t[: r.size], r.y[: r.size, 0]
    (t1, i1), (t2, i2) = page.live.hits
    both = i1 * impulse_response(page.live.osc, t - t1) + i2 * impulse_response(page.live.osc, t - t2)
    assert np.max(np.abs(x - both)) < 1e-9


def test_a_push_held_then_let_go(page):
    page.load_trial(trial("A step"))
    assert page.let_go_button.isEnabled() and "held" in page.readout.text()
    page.step(1.0)
    page.let_go_button.click()
    assert not page.let_go_button.isEnabled() and not page.live.holding()
    page.step(4.0)
    assert abs(page.live.x) < 1e-4  # rung back down about zero


def test_a_pulse_lands_its_peak_on_the_spectrum(page):
    page.load_trial(trial("A pulse half a period"))
    assert page.tabs.currentIndex() == view.TAB_PULSE
    page.step(0.6)
    (r,), (peak,) = page.pulse_points["RECTANGLE"].getData()
    r = 10**r  # drawn on a log axis: its data are the logs
    assert r == pytest.approx(0.5, abs=0.01) and peak == pytest.approx(1.85, abs=0.005)
    assert "peaked at 1.85" in page.pulse_note.text()
    page.theory.setChecked(True)
    assert points(page.pulse_lines["RECTANGLE"]) == view.RATIOS.size and page.theory_too.isChecked()


def test_a_waves_harmonics_land_once_steady(page):
    page.load_trial(trial("A square wave at a third"))
    assert page.wave_on.isChecked() and page.live.wave is not None and not page.freq.isEnabled()
    page.step(2.0)
    assert "Waiting" in page.harm_note.text()
    page.step(7.0)
    assert page.harm_note.text().startswith("The swing's power by harmonic: 3rd 89.7 %, 1st 10.2 %")
    page.wave_on.setChecked(False)  # off: it rings down
    assert page.live.wave is None


def test_the_convolution_at_a_scrubbed_instant_is_the_run(page):
    page.load_trial(trial("Watch the convolution"))
    page.step(0.5)
    page._run(False)
    page._on_scrubbed(0.3)
    text = page.conv_note.text()
    area, run = re.findall(r"(-?[\d.]+ [µm]?m)\b", text.split("start:")[1])
    assert area == run and "At 0.3 s" in text


def test_the_grips_change_m_k_and_c_as_it_runs(page, app):
    page.load_trial(trial("A blow"))
    page.step(0.2)
    body = page.body
    for name, box in (("mass", page.mass), ("stiffness", page.stiffness), ("damping", page.damping)):
        before = box.value()
        at = body.press_point(name)
        drag(body, at, at + QtCore.QPointF(0.0, 0.2), app)
        assert box.value() != before, name
    s = page.practice_state()
    assert page.live.osc == view.oscillator(s) and page.running and page.live.edits


def test_predictions_come_after_play(page):
    box = page.trials
    page.load_trial(trial("A blow"))
    page.damping.setValue(4.0)  # the reader handles it ...
    page.step(forcing.PLAYED_AFTER + 0.1)  # ... and watches
    assert box.played
    asked = trial("A half-sine")
    box.combo.pick(forcing.TRIALS[asked].name)
    assert box.asking == asked and box.ask.isVisible()
    box._on_skip()
    assert box.asking is None and page.running


def test_a_problems_setup_starts_what_it_sets_up(page):
    page.load_setup(dict(start=view.APPLY, kind=PulseKind.RECTANGLE, amplitude=5.0, duration=0.05))
    assert page.running and page.live.pushes and page.live.pushes[-1].pulse.duration == 0.05
    page.load_setup(dict(start=view.WAVE, tune=2))
    assert page.live.wave is not None and page.live.wave.freq_hz == pytest.approx(
        page.live.osc.natural_hz / 2, abs=1e-3
    )


def test_reset_and_defaults(page, main_window):
    page.load_trial(trial("Hammer blows"))
    page.step(1.0)
    page.reset()
    assert page.live.t == 0 and not page.running and page.live.wave is None and not page.wave_on.isChecked()
    main_window.restore_defaults()
    assert dataclasses.replace(page.practice_state(), tab=0) == view.ForcingTrial("", "")  # Defaults leaves the tab
