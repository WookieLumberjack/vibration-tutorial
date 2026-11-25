"""The Single oscillator, rebuilt live (#478 Phase 3's pilot): the mass runs as it's handled, the frequency
response is built by hand, the energy balance closes through every change, and predictions come after play; its
damper's law is picked as it runs (the Damping models page's, merged here), the viscous ghost beside it, the loop
the run draws, and the phasors."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from notes_quoting import quoted

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore, QtGui  # noqa: E402

from vib_tutorial.core.damping_models import MODELS, Model  # noqa: E402
from vib_tutorial.gui import free_body_view, live_body, sdof  # noqa: E402
from vib_tutorial.view_models import sdof as view  # noqa: E402
from vib_tutorial.view_models.live_mass import COL  # noqa: E402
from vib_tutorial.view_models.text import fmt  # noqa: E402

Left = QtCore.Qt.MouseButton.LeftButton
NoButton = QtCore.Qt.MouseButton.NoButton


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


@pytest.fixture
def page(main_window, app):
    w = main_window
    w.resize(1280, 800)
    w.show_page("sdof_page")
    p = w.sdof_page
    p.loop.enabled = False
    for _ in range(3):
        app.processEvents()
    return p


def trial(name: str) -> int:
    return next(i for i, t in enumerate(sdof.TRIALS) if t.name.startswith(name))


def drag(view_widget, start: QtCore.QPointF, end: QtCore.QPointF, app, steps: int = 8) -> None:
    """Press at `start`, move to `end` with the button held, release (view coordinates of a PlotWidget)."""
    vb = view_widget.getPlotItem().getViewBox()
    vp = view_widget.viewport()

    def send(kind, at: QtCore.QPointF, button, buttons) -> None:
        local = view_widget.mapFromScene(vb.mapViewToScene(at))
        ev = QtGui.QMouseEvent(
            kind, QtCore.QPointF(local), QtCore.QPointF(vp.mapToGlobal(local)), button, buttons,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )  # fmt: skip
        QtWidgets.QApplication.sendEvent(vp, ev)
        app.processEvents()

    send(QtCore.QEvent.Type.MouseButtonPress, start, Left, Left)
    for k in range(1, steps + 1):
        send(QtCore.QEvent.Type.MouseMove, start + (end - start) * (k / steps), NoButton, Left)
    send(QtCore.QEvent.Type.MouseButtonRelease, end, Left, NoButton)


def test_it_opens_at_rest_on_its_system_with_things_to_try_after_the_controls(page):
    assert not page.running and page.live.t == 0 and page.live.x == 0
    left = page.trials.parentWidget()
    boxes = [w for w in left.findChildren(QtWidgets.QGroupBox) if w.parentWidget() is left]
    assert [b.title() for b in boxes] == ["Oscillator", "Drive", "Things to try"]
    assert page.tabs.count() == 5  # four tabs plus the Theory: one over #478 item 6's three, since the merge
    assert page.practice_state() == view.SdofTrial("", "")
    assert page.friction_mm.isHidden() and page.relax.isHidden() and page.ghost_on.isHidden()  # the other laws'


def test_a_pluck_rings_down_and_the_readout_measures_it(page):
    page.load_trial(trial("Ringing down"))
    assert page.running and page.live.x == pytest.approx(0.010)
    page.step(3.0)
    assert page.readout.text().startswith("Each peak is 0.73 of the one before: δ = ln 1.37 = 0.315, so ζ = 0.05")
    times, _ = page.peak_item.getData()
    assert len(times) >= 5


def test_driven_off_resonance_the_readout_and_the_point_say_what_the_worked_example_does(page):
    page.drive_on.setChecked(True)  # 1 N at 3 Hz, from rest
    page.step(12.0)
    assert f"±{quoted('sdof', '17.1')} mm, lagging the drive by {quoted('sdof', '40.1')}°" in page.readout.text()
    points = page.recorder.current(page._key())
    assert len(points) == 1 and points[0].freq_hz == 3.0
    log_amplification = page.point_items[0].getData()[1][0]  # on the log axis, its log
    assert 10**log_amplification == pytest.approx(quoted("sdof", "6.84"), abs=0.005)


def test_a_change_mid_run_carries_the_motion_on_and_is_marked(page):
    page.load_trial(trial("Ringing down"))
    page.step(0.43)
    t, x, v = page.live.t, page.live.x, page.live.v
    page.mass.setValue(4.0)  # the box: a live change, a capture
    assert page.live.t == t and page.live.x == x and page.live.v == pytest.approx(v / 4)
    assert page.live.edits == [t]
    page.step(0.5)
    lines = page.marks.lines
    assert lines and lines[0].isVisible() and lines[0].value() == t
    xs = page.x_item.getData()[0]
    assert xs[0] < t < xs[-1]  # the trace keeps what came before


def test_the_drive_and_the_sweep_are_buttons_lit_while_on(page):
    assert page.drive_on.text() == "Apply force" and not page.drive_on.styleSheet()
    page.drive_on.click()
    assert page.live.drive.on and page.drive_on.text() == "Force on: click to stop"
    assert "background-color" in page.drive_on.styleSheet()  # lit, as the Simulation page's force
    page.kind.setCurrentIndex(view.UNBALANCE)
    assert page.drive_on.text() == "Unbalance spinning: click to stop"
    page.drive_on.click()
    assert not page.live.drive.on and page.drive_on.text() == "Spin the unbalance"
    page.sweep_button.click()  # the sweep drives it too
    assert page.sweep_button.text() == "Sweeping: click to stop"
    assert page.drive_on.isChecked() and page.drive_on.text() == "Unbalance spinning: click to stop"  # set quietly


def test_the_energy_tab_redraws_at_its_own_rate_as_it_runs(page):
    page.load_trial(trial("Ringing down"))
    page.tabs.setCurrentIndex(view.TAB_ENERGY)
    page.step(0.5)  # drawn, as anything but a frame is
    drawn = page.energy.ledger.text()
    assert "Stored now" in drawn
    page.step(0.5)
    drawn = page.energy.ledger.text()
    page._advance(0.5)
    page._draw_live(frame=True)  # the next frame, at once: the same picture, not redrawn
    assert page.energy.ledger.text() == drawn
    page._draw_live()  # anything but a frame (a trial, Reset, a tab, a scrub) draws it now
    assert page.energy.ledger.text() != drawn


def test_scrubbed_back_the_body_is_drawn_to_that_times_scale(page):
    page.load_trial(trial("Ringing down"))
    page.step(12.0)  # rung down to a few µm: the drawing magnified to match
    page._on_run(False)
    page._on_scrubbed(page.live.t - 11.5)  # back to a swing of millimetres
    assert abs(page.body._centre - live_body.REST_X) <= live_body.SWING * 1.01  # on the page, not off it


def test_the_grips_set_m_k_and_c_as_it_runs(page, app):
    page.load_trial(trial("Ringing down"))
    page.step(0.3)
    body = page.body
    for name, box in (("stiffness", page.stiffness), ("damping", page.damping), ("mass", page.mass)):
        before = box.value()
        at = body.press_point(name)
        drag(body, at, at + QtCore.QPointF(0.0, 0.3), app)
        assert box.value() == pytest.approx(sdof.GRIPS[name].dragged(before, 0.3), rel=0.05), name
        assert box.value() > before
    p = page.live.params
    assert (p.mass, p.stiffness, p.damping) == (page.mass.value(), page.stiffness.value(), page.damping.value())
    assert len(page.live.edits) >= 3 and page.running  # each drag a live change, the run going on


def test_a_pluck_on_the_body_holds_the_mass_then_lets_it_go(page, app):
    body = page.body
    start = body.body_point()
    drag(body, start, start + QtCore.QPointF(0.2, 0.0), app)
    assert page.running and page.live.held is None
    assert page.live.x == pytest.approx(0.2 / body.gain, rel=0.05)  # moved as far as the hand
    assert page.live.sim.released > 0  # the hand put the energy in
    page.step(0.2)
    assert page.live.t > 0


def test_the_forces_are_drawn_beside_the_mass_their_words_staying_put_as_they_turn(page):
    forces = page.forces
    page.load_trial(trial("Ringing down"))
    page.step(0.05)
    assert [s for s, _ in forces.rows()] == ["kx", "cẋ", "f", "mẍ"]
    places = [(s.pos(), z.pos()) for s, z in zip(forces.symbols, forces.sizes)]
    directions = set()
    for _ in range(12):  # through a swing: the spring's force turns round
        page.step(0.03)
        assert [(s.pos(), z.pos()) for s, z in zip(forces.symbols, forces.sizes)] == places
        xs = forces.shafts[0].getData()[0]
        directions.add(bool(xs[1] > xs[0]))
        assert abs(xs[0]) == pytest.approx(free_body_view.HALF)  # out of the mass's side
    assert directions == {True, False}
    page.loop.enabled = False
    forces._write_sizes()
    assert forces.rows()[0][1] == fmt(abs(page.live.params.stiffness * page.live.x), "N")


def test_a_sweep_builds_the_frequency_response_point_by_point(page):
    page.load_trial(trial("Building the frequency response"))
    assert page.sweep is not None and page.drive_on.isChecked()
    freqs = page.sweep.freqs
    page.step(120.0)
    points = page.recorder.current(page._key())
    assert [round(p.freq_hz, 3) for p in points] == [round(f, 3) for f in freqs]
    assert page.sweep is None and not page.sweep_button.isChecked()  # done
    note = sdof.TRIALS[trial("Building the frequency response")].note
    assert f"from {freqs[0]:.2f} to {freqs[-1]:.2f} Hz" in note  # as the trial's note says
    page.theory.setChecked(True)
    assert page.band.isVisible() and "Half-power band 3.01–3.33 Hz" in page.frf_note.text()
    page.mass.setValue(2.0)  # the points belong to the old oscillator now: faded
    key = page._key()
    assert page.recorder.current(key) == [] and len(page.recorder.earlier(key)) == len(freqs)
    assert len(page.earlier_items[0].getData()[0]) == len(freqs)


def test_an_undamped_beat_never_lands_a_point(page):
    page.load_trial(trial("Undamped, just off"))
    page.step(15.0)
    assert page.recorder.points == []
    assert "waiting for the swing to settle" in page.readout.text()


def test_the_energy_tab_books_a_capture(page):
    page.load_trial(trial("Ringing down"))
    page.tabs.setCurrentIndex(view.TAB_ENERGY)
    page.step(0.4)
    page.mass.setValue(2.0)
    page.step(0.4)
    text = page.energy.ledger.text()
    assert "Lost capturing mass" in text and "Put in by your hand" in text and "It closes to" in text


def test_predictions_come_after_play_about_a_case_not_yet_tried(page):
    box = page.trials
    asked = trial("Resonance")
    box.combo.pick(sdof.TRIALS[asked].name)  # before any play: it runs at once
    assert box.asking is None and page.drive_on.isChecked()
    assert not box.invite.isVisible()
    page.load_trial(trial("Ringing down"))
    page.stiffness.setValue(500.0)  # the reader handles it ...
    page.step(sdof.PLAYED_AFTER + 0.1)  # ... and watches
    assert box.played and box.invite.isVisible() and "asks what you'll see" in box.invite.text()
    unasked = trial("Undamped, just off")
    box.combo.pick(sdof.TRIALS[unasked].name)
    assert box.asking == unasked and box.ask.isVisible()  # not tried yet: asked
    box._on_skip()  # Just run it: never a gate
    assert box.asking is None


def test_the_poles_drag_sets_k_and_c_live(page, app):
    page.tabs.setCurrentIndex(view.TAB_POLES)
    page.load_trial(trial("Ringing down"))
    page.s_plot.grab()
    for _ in range(3):
        app.processEvents()
    s = page.live.params.oscillator().poles()
    t = page.live.t
    drag(page.s_plot, QtCore.QPointF(s[0].real, s[0].imag), QtCore.QPointF(-6.0, 12.0), app)
    s = page.live.params.oscillator().poles()
    assert s[0] == pytest.approx(complex(-6.0, 12.0), rel=0.03)
    assert page.stiffness.value() == pytest.approx(180.0, rel=0.04) and page.damping.value() == pytest.approx(
        12.0, rel=0.04
    )
    assert page.live.t == t and page.live.edits  # a live change, the run's clock kept


def test_defaults_round_trips_through_the_bindings(page):
    page.load_trial(trial("A rotating unbalance"))
    state = page.practice_state()
    assert state.kind == view.UNBALANCE and state.drive and state.freq_hz == pytest.approx(3.183, abs=1e-3)
    page.load_setup({k: v for k, v in state.__dict__.items() if k not in ("name", "note")})
    assert page.practice_state() == state


def test_the_drive_lag_at_resonance_is_ninety_degrees_on_the_body_and_the_plot(page):
    page.load_trial(trial("Resonance"))
    page.step(15.0)
    points = page.recorder.current(page._key())
    assert points and -math.degrees(np.angle(points[-1].response)) == pytest.approx(90.0, abs=1.0)
    r = page.live.record
    x = r.y[: r.size, COL["x"]]
    assert np.abs(x[-500:]).max() == pytest.approx(0.025, rel=0.02)


# ------------------------------------------------------------------ the damper's law (once Damping models)


def test_four_loops_one_energy_and_the_readouts_the_worked_example_quotes(page):
    page.load_trial(trial("Four loops"))
    page.step(12.0)
    assert "Steady at 3.18 Hz: x swings ±10 mm" in page.readout.text()
    note = page.loop_tab.note.text()
    assert f"a viscous damper's πcΩX² there is {quoted('sdof', '12.6')} mJ" in note
    enclosed = float(note.split("encloses ")[1].split(" mJ")[0])
    assert enclosed == pytest.approx(12.57, rel=0.01)  # the run's own loop
    loops = page.loop_tab.theory_loops
    drawn = [m for m, item in loops.items() if item.getData()[0] is not None and len(item.getData()[0])]
    assert drawn == list(MODELS)


def test_the_law_changes_as_it_runs_and_the_books_close(page):
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
    assert f"Each peak is {quoted('sdof', '3.14')} mm lower" in text
    assert "Stuck 575 µm off centre" in text  # the worked example's 575 µm
    assert page.live.v == 0.0 and page.ghost.v != 0.0  # the viscous ghost rings on
    assert page.ghost_item.isVisible() and len(page.ghost_item.getData()[0])
    assert page.body.ghost.isVisible() and not page.ghost_on.isHidden()
    assert "Friction F<sub>f</sub> = πcω<sub>n</sub>X/4 = 314 mN" in page.info.text()  # the worked example's
    assert not page.friction_mm.isHidden()


def test_the_zener_info_and_its_live_decay(page):
    page.load_trial(trial("A viscoelastic loss peak"))
    assert f"τ<sub>r</sub> = {quoted('sdof', '45.2')} ms" in page.info.text()
    assert page.live.params.has_arm and not page.relax.isHidden()
    page.step(2.0)
    assert page.readout.text().startswith("Each peak is")
    names = [a.name for a in page.forces._arrows]
    assert "arm" in names and "damper" not in names


def test_the_hysteretic_stand_in_swept_lands_on_its_curve(page):
    page.load_trial(trial("Hysteretic damping's stand-in"))
    page.step(120.0)
    points = page.recorder.current(page._key())
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


def test_a_law_trial_round_trips_and_back_to_viscous_the_ghost_goes(page):
    page.load_trial(trial("Twice"))
    state = page.practice_state()
    page.load_setup({k: v for k, v in state.__dict__.items() if k not in ("name", "note")})
    assert page.practice_state() == state
    page.load_trial(trial("Released against friction"))
    page.step(0.5)
    page.model.setCurrentIndex(MODELS.index(Model.VISCOUS))
    page.step(0.2)
    assert not page.ghost_item.isVisible() and not page.body.ghost.isVisible() and page.ghost_on.isHidden()
