"""Scrubbing back through a paused page's record (`gui/scrub.py`): the cursors show only while paused, and the
body is drawn as the record had it at the cursor."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gui


def _advance(page, seconds: float) -> None:
    """Steps for `seconds` of simulated time, recorded in the history (as frames would)."""
    end = page.sim.t + seconds
    while page.sim.t < end - 0.5 * page.sim.step_size():
        page._sim_target = end  # each frame's `_start` takes no more than its budget's worth
        page._start()
        page._collect()


def test_the_simulation_page_scrubs_its_history_while_paused(main_window):
    w = main_window
    p = w.sim_page
    w.pages.setCurrentWidget(p)
    p.force_panel.button.click()  # the harmonic force on
    _advance(p, 3.0)
    p._tick()
    assert p.running and not p.scrub.shown  # running: no cursor

    p.controls.run.click()  # pause
    p._tick()
    assert not p.running and p.scrub.shown and p.scrub.t is None
    t_end = p.history.t[p.history.size - 1]
    assert all(line.data_value() == pytest.approx(t_end) for line in p.scrub.lines)

    # Back to 1.5 s: the chain is drawn as it was then, the clock says so, both cursors there.
    p.scrub.lines[0].dragged.emit(1.5)
    p._tick()
    t = p.history.t[: p.history.size]
    i = int(np.searchsorted(t, 1.5))
    from vib_tutorial.gui.animation import SPACING

    n = p.sim.system.n
    drawn = (p.chain._centers - (np.arange(n) + 1) * SPACING) / p.chain.gain
    assert drawn == pytest.approx(p.history.x[i, :n]) and not np.allclose(drawn, p.sim.displacement)
    assert "looking back" in p.controls.time_label.text() and p.scrub.lines[1].data_value() == pytest.approx(1.5)
    now = p.sim.t
    p.controls.run.click()  # Resume: from where it was paused, the cursors gone
    assert p.sim.t == now
    p._tick()
    assert not p.scrub.shown and p.scrub.t is None


def test_duffing_scrubs_its_strip_chart_onto_the_phase_plane(main_window):
    w = main_window
    p = w.duffing_page
    w.pages.setCurrentWidget(p)
    p.loop.enabled = False
    p.run_button.click()  # Start
    assert p.running and not p.scrub.shown
    p.step(6.0)
    p.run_button.click()  # Pause
    trace = p.record.trace
    ts, xs, vs = trace.data[:, : trace.n]
    assert p.scrub.shown and p.scrub.lines[0].data_value() == pytest.approx(ts[-1])
    p.scrub.lines[0].dragged.emit(3.0)
    dot = p.scrub_dot.pos()
    assert p.scrub_dot.isVisible() and (dot.x(), dot.y()) == pytest.approx(
        (np.interp(3.0, ts, xs), np.interp(3.0, ts, vs))
    )
    assert "Looking back at t = 3.00 s" in p.drive_state.text()
    p.run_button.click()  # Resume: back to now
    assert not p.scrub.shown and not p.scrub_dot.isVisible() and "Looking back" not in p.drive_state.text()


def test_the_torsional_start_up_scrubs_its_run(main_window):
    w = main_window
    p = w.torsional_page
    w.pages.setCurrentWidget(p)
    v = p.startup_view
    p.pause_button.click()  # Start
    v.step(0.5)
    assert not v.scrub.shown
    p.pause_button.click()  # Pause
    k = v._k
    assert v.scrub.shown and v.scrub.lines[1].data_value() == pytest.approx(v._t[k - 1])
    v.scrub.lines[1].dragged.emit(0.2)
    rpm = np.interp(0.2, v._t[:k], v._rpm[:k])
    assert "Looking back at t = 0.200 s" in v.startup_status.text() and f"at {rpm:.0f} rpm" in v.startup_status.text()
    p.pause_button.click()  # Resume
    assert not v.scrub.shown


def test_the_rotor_scrubs_its_strip_chart_onto_the_orbits(main_window):
    w = main_window
    p = w.rotor_page
    w.pages.setCurrentWidget(p)
    p.loop.enabled = False
    p._run(True)
    for _ in range(40):
        p.step(0.02)
    p._run(False)
    t, hist = p.history.window(p.time_window.value())
    assert p.scrub.shown and p.scrub.lines[0].data_value() == pytest.approx(t[-1])
    at = 0.5 * (t[0] + t[-1])
    p.scrub.lines[0].dragged.emit(at)
    i = min(int(np.searchsorted(t, at)), t.size - 1)
    row = p._shown(hist[i : i + 1])[0]
    dot = p.orbits.dots[1].pos()
    assert (dot.x(), dot.y()) == pytest.approx((row[2], row[3]))
    p._run(True)
    assert not p.scrub.shown and p.scrub.t is None
