"""#478 Phase 3's shared live parts, Qt-free: the live mass, its grips, its free-body diagram, the build-the-plot
recorder and the energy balance (`view_models/live_mass.py`, `grips.py`, `free_body.py`, `recorder.py`)."""

import math

import numpy as np
import pytest

from vib_tutorial.core.sdof import TWO_PI, receptance
from vib_tutorial.gui import style
from vib_tutorial.view_models import free_body, grips, recorder
from vib_tutorial.view_models.live_mass import (
    COL,
    UNBALANCE,
    Drive,
    LiveMass,
    Params,
    energy_view,
    ring_down,
)


def _run(live: LiveMass, seconds: float) -> None:
    end = live.t + seconds
    while live.t < end - 1e-9:
        live.advance(min(0.05, end - live.t) + 1e-12)  # the clock carries a frame's part of a step on


def _plucked(x0: float = 0.01, params: Params | None = None) -> LiveMass:
    live = LiveMass(params)
    live.hold(x0)
    live.let_go()
    return live


# ------------------------------------------------------------------ the live mass


def test_a_pluck_rings_down_at_the_oscillators_damping():
    live = _plucked()
    _run(live, 3.0)
    t, y = live.since_change()
    decay = ring_down(t, y[:, COL["x"]])
    assert decay is not None
    osc = live.params.oscillator()
    assert decay.zeta == pytest.approx(osc.zeta, rel=1e-3)
    assert decay.ratio == pytest.approx(math.exp(osc.log_decrement), rel=1e-3)
    assert "Each peak is 0.73 of the one before" in decay.text and "ζ = 0.05" in decay.text


def test_holding_stops_the_clock_and_the_hand_puts_the_energy_in():
    live = LiveMass()
    _run(live, 0.5)
    live.hold(0.02)
    t = live.t
    assert live.advance(0.5) == 0 and live.t == t
    assert live.sim.released == pytest.approx(0.5 * 400.0 * 0.02**2)
    live.let_go()
    assert live.changed_at == t
    _run(live, 0.2)
    assert live.t > t and live.x != 0.02


def test_a_live_change_carries_the_run_on():
    live = _plucked()
    _run(live, 0.4)
    t, x, v = live.t, live.x, live.v
    live.set_params(Params(mass=4.0))
    assert live.t == t and live.x == x  # no restart
    assert live.v == pytest.approx(v / 4.0)  # a capture keeps the momentum
    assert live.edits == [t] and live.changed_at == t
    live.set_params(Params(mass=4.0))  # the same again: no change
    assert live.edits == [t]
    _run(live, 0.3)
    assert live.record.t[0] < t < live.record.t[live.record.size - 1]  # the record kept what came before


def test_the_drives_frequency_changes_without_a_jump_and_an_unbalance_grows_as_the_speed_squared():
    live = LiveMass(drive=Drive(on=True, freq_hz=2.0))
    _run(live, 0.37)
    before = live.force_now
    live.set_drive(Drive(on=True, freq_hz=2.5))
    assert live.force_now == before  # the phase runs on
    assert live.changed_at == live.t
    unbalance = Drive(on=True, kind=UNBALANCE, amplitude=1e-4, freq_hz=5.0)
    assert unbalance.force == pytest.approx(1e-4 * (TWO_PI * 5.0) ** 2)
    live.set_drive(unbalance)
    assert live.sim.force.settings.amplitude == unbalance.force
    at = live.changed_at
    live.set_drive(Drive(on=True, kind=UNBALANCE, amplitude=2e-4, freq_hz=5.0))  # only its size: still steady
    assert live.changed_at == at
    live.set_drive(Drive(on=False))
    assert live.force_now == 0.0


def test_the_energy_balance_closes_through_every_kind_of_change():
    live = LiveMass(drive=Drive(on=True, freq_hz=3.0))
    _run(live, 0.6)
    live.set_params(Params(stiffness=900.0))
    _run(live, 0.3)
    live.set_params(Params(mass=3.0, stiffness=900.0))
    _run(live, 0.3)
    live.set_params(Params(mass=0.5, stiffness=900.0, damping=5.0))
    _run(live, 0.3)
    live.hold(-0.01)
    live.let_go()
    _run(live, 0.3)
    view = energy_view(live.record.t[: live.record.size], live.record.y[: live.record.size])
    assert view is not None
    rows = {r.name: r.value for r in view.rows}
    for name in ("Changing the spring", "Lost capturing mass", "Carried off by mass", "Put in by your hand"):
        assert rows[name] != 0, name
    assert rows["Lost capturing mass"] < 0 and rows["Carried off by mass"] < 0
    assert abs(view.residual) < 1e-9 * max(abs(v) for v in rows.values())
    assert "It closes to" in view.text
    # Each sample's balance closes too, through the changes.
    gap = view.put_in + view.changes - view.taken_out - view.stored
    assert np.abs(gap).max() < 1e-9 * np.abs(view.put_in).max()


def test_reset_goes_back_to_rest_keeping_the_settings():
    live = LiveMass(Params(mass=2.0), Drive(on=True))
    _run(live, 0.3)
    live.reset()
    assert live.t == 0 and live.record.size == 0 and live.x == 0
    assert live.params.mass == 2.0 and live.sim.force.on


# ------------------------------------------------------------------ grips


def test_a_grip_scales_by_the_distance_dragged_within_its_range():
    g = grips.Grip("stiffness", 10.0, 10_000.0, per=0.5)
    assert g.dragged(400.0, 0.5) == pytest.approx(400.0 * math.e)
    assert g.dragged(400.0, -0.5 * math.log(2)) == pytest.approx(200.0)
    assert g.dragged(400.0, 100.0) == 10_000.0 and g.dragged(400.0, -100.0) == 10.0
    assert g.distance(400.0, g.dragged(400.0, 0.3)) == pytest.approx(0.3)


def test_a_damper_grip_snaps_to_nought_and_grows_from_its_low_end():
    g = grips.Grip("damping", 0.01, 500.0, zero_below=0.02)
    assert g.dragged(2.0, -100.0) == 0.0
    assert g.dragged(0.0, 0.1) == 0.0  # from nought, a short drag stays under the snap
    assert g.dragged(0.0, 2.0) == pytest.approx(0.01 * math.exp(4.0))


def test_the_mass_is_drawn_as_a_block_of_one_material():
    assert grips.body_size(8.0, 1.0) == pytest.approx(2.0)
    assert grips.body_size(1.0, 1.0) == 1.0
    assert grips.body_size(1e3, 1.0) == grips.SIZE_RANGE[1] and grips.body_size(1e-3, 1.0) == grips.SIZE_RANGE[0]


# ------------------------------------------------------------------ the free-body diagram


def test_the_forces_add_up_to_the_mass_times_its_acceleration():
    live = LiveMass(drive=Drive(on=True, freq_hz=2.0))
    _run(live, 0.43)
    p = live.params
    body = free_body.one_mass(p.stiffness, p.damping, live.x, live.v, live.force_now)
    a = (live.force_now - p.stiffness * live.x - p.damping * live.v) / p.mass
    assert body.net.value == pytest.approx(p.mass * a)
    assert [arrow.name for arrow in body.arrows] == ["spring", "damper", "drive"]
    assert body.arrows[0].label.startswith("kx = ") and body.arrows[0].label.endswith("N")


def test_held_the_hand_balances_the_spring():
    body = free_body.one_mass(400.0, 2.0, 0.01, 0.0, hand=4.0)
    assert body.net.value == pytest.approx(0.0)
    assert body.arrows[-1].name == "hand"


@pytest.mark.parametrize("theme", list(style.THEMES.values()), ids=lambda t: t.name)
def test_every_role_a_view_model_names_is_a_colour(theme):
    body = free_body.one_mass(400.0, 2.0, 0.01, 0.1, 1.0, friction=0.5, hand=1.0)
    live = _plucked()
    _run(live, 0.1)
    view = energy_view(live.record.t[: live.record.size], live.record.y[: live.record.size])
    assert view is not None
    for name in [a.role for a in (*body.arrows, body.net)] + [r.role for r in view.rows]:
        assert style.role(name, theme).startswith("#"), name
    with pytest.raises((KeyError, AttributeError, ValueError)):
        style.role("energy.nothing", theme)


def test_the_arrow_scale_grows_at_once_and_follows_the_recent_peak_down():
    s = free_body.ArrowScale(room=1.0)
    assert s.update(0.0, 10.0) == pytest.approx(free_body.ROOM / 10.0)
    assert s.update(0.1, 20.0) == pytest.approx(free_body.ROOM / 20.0)  # at once
    assert s.update(0.2, 1.0) == pytest.approx(free_body.ROOM / 20.0)  # small, but the peak is recent
    assert s.update(0.1 + free_body.SHRINK_AFTER - 0.01, 1.0) == pytest.approx(free_body.ROOM / 20.0)
    assert s.update(0.1 + free_body.SHRINK_AFTER + 0.01, 1.0) == pytest.approx(free_body.ROOM / 1.0)
    for k in range(400):  # a steady swing of 5 N: its peak fills the room through every cycle
        t = 10.0 + 0.01 * k
        scale = s.update(t, abs(5.0 * math.sin(2 * math.pi * t)))
        if t >= 10.25:  # its first peak
            assert scale == pytest.approx(free_body.ROOM / 5.0)
    assert s.update(1.0, 2.0) == pytest.approx(free_body.ROOM / 2.0)  # scrubbed back: from there


# ------------------------------------------------------------------ the recorder


def _dwell(live: LiveMass, rec: recorder.Recorder, most: float = 30.0) -> recorder.Point | None:
    end = live.t + most
    while live.t < end:
        _run(live, 0.25)
        t, y = live.since_change()
        m = recorder.measure(t, y[:, COL["x"]], y[:, COL["f"]], live.drive.freq_hz)
        if m is not None and (point := rec.land(m, live.params)) is not None:
            return point
    return None


@pytest.mark.parametrize("freq_hz", [1.5, 3.1831, 6.0])
def test_a_dwell_lands_the_receptance(freq_hz):
    live = LiveMass(drive=Drive(on=True, freq_hz=freq_hz))
    rec = recorder.Recorder()
    point = _dwell(live, rec)
    assert point is not None
    h = receptance(live.params.oscillator(), np.array([freq_hz]))[0]
    assert abs(point.response - h) < 0.01 * abs(h)
    assert rec.current(live.params) == [point] and rec.earlier(live.params) == []


def test_a_swing_still_growing_or_beating_never_lands():
    rec = recorder.Recorder()
    resonant = LiveMass(Params(damping=0.0), Drive(on=True, freq_hz=Params().oscillator().natural_hz))
    assert _dwell(resonant, rec, most=6.0) is None
    beating = LiveMass(Params(damping=0.0), Drive(on=True, freq_hz=3.0))
    assert _dwell(beating, rec, most=6.0) is None
    assert rec.points == []


def test_a_point_from_before_a_change_is_kept_as_earlier_and_a_new_one_replaces_its_frequency():
    rec = recorder.Recorder()
    a, b = Params(), Params(mass=2.0)
    m = recorder.Measurement(2.0, 0.001 + 0j, True)
    rec.land(m, a)
    rec.land(recorder.Measurement(2.0, 0.002 + 0j, True), a)
    assert len(rec.current(a)) == 1 and rec.current(a)[0].response == 0.002
    rec.land(m, b)
    assert len(rec.current(b)) == 1 and len(rec.earlier(b)) == 1
    assert rec.land(recorder.Measurement(3.0, 0.001 + 0j, False), b) is None


def test_a_sweep_visits_its_frequencies_in_order_across_the_resonance():
    freqs = recorder.sweep_frequencies(3.18, 0.05)
    assert freqs == tuple(sorted(freqs)) and freqs[0] == pytest.approx(3.18 / 3) and freqs[-1] == pytest.approx(9.54)
    assert sum(abs(f / 3.18 - 1) <= 0.05 for f in freqs) >= 3  # the half-power band has its points
    sweep = recorder.Sweep(freqs[:3])
    assert sweep.frequency == freqs[0] and sweep.landed() == freqs[1] and sweep.landed() == freqs[2]
    assert sweep.landed() is None and sweep.done


# ------------------------------------------------------------------ the damping laws, live (the Single oscillator)


def _damping(model, **kw):
    from vib_tutorial.view_models import sdof

    return sdof.SdofTrial("", "", model=model, **kw)


def test_the_zener_mass_rings_down_as_the_matched_closed_form_does():
    from vib_tutorial.core.damping_models import Model
    from vib_tutorial.view_models import sdof

    t = _damping(Model.ZENER)
    live = _plucked(0.01, sdof.params(t))
    _run(live, 1.5)
    ts, y = live.since_change()
    expected = sdof.matched(t).free_decay(Model.ZENER, 0.01, np.concatenate([[0.0], ts - live.changed_at]))[1:]
    assert np.abs(y[:, COL["x"]] - expected).max() < 2e-6 * 0.01


def test_friction_stops_where_the_closed_form_says():
    from vib_tutorial.core.damping_models import Model, coulomb_stop
    from vib_tutorial.view_models import sdof

    t = _damping(Model.COULOMB)
    m = sdof.matched(t)
    live = _plucked(0.01, sdof.params(t))
    _run(live, 3.0)
    assert live.x == pytest.approx(coulomb_stop(m.stiffness, m.friction, 0.01)[1], abs=2e-6)
    assert live.v == 0.0
    assert live.friction_force == pytest.approx(m.stiffness * live.x)  # stuck: friction holds the spring's pull


def test_the_books_close_through_changes_to_and_from_a_zener_arm():
    from vib_tutorial.core.damping_models import Model
    from vib_tutorial.view_models import sdof

    live = LiveMass(sdof.params(_damping(Model.VISCOUS)), Drive(on=True, freq_hz=3.0))
    _run(live, 0.5)
    live.set_params(sdof.params(_damping(Model.ZENER)))  # a swap of steppers, the books carried over
    _run(live, 0.5)
    zener = sdof.params(_damping(Model.ZENER, mass=2.0, stiffness=600.0))
    live.set_params(zener)  # a capture and a stiffer spring, the arm's stretch kept
    _run(live, 0.5)
    assert live.arm_force != 0.0
    live.set_params(sdof.params(_damping(Model.COULOMB)))  # and back, the arm's energy taken with it
    _run(live, 0.5)
    live.set_params(sdof.params(_damping(Model.ZENER)))  # friction's loss carried over to the arm's stepper
    _run(live, 0.5)
    view = energy_view(live.record.t[: live.record.size], live.record.y[: live.record.size])
    assert view is not None and len(live.edits) == 4
    gap = view.put_in + view.changes - view.taken_out - view.stored
    assert np.abs(gap).max() < 1e-9 * np.abs(view.put_in).max()


def test_the_hysteretic_stand_in_is_matched_at_the_frequency_it_runs_at():
    from vib_tutorial.core.damping_models import Model
    from vib_tutorial.view_models import sdof

    free = sdof.params(_damping(Model.HYSTERETIC))
    assert free.damping == pytest.approx(sdof.params(_damping(Model.VISCOUS)).damping)  # at f_n, as viscous
    driven = sdof.params(_damping(Model.HYSTERETIC, drive=True, freq_hz=6.0))
    assert driven.damping == pytest.approx(free.damping * sdof.matched(_damping(Model.HYSTERETIC)).natural_hz / 6.0)


def test_a_steady_viscous_loop_encloses_pi_c_omega_x_squared():
    from vib_tutorial.core.damping_models import Model
    from vib_tutorial.view_models import damping, sdof

    t = _damping(Model.VISCOUS, drive=True, freq_hz=2.0)
    live = LiveMass(sdof.params(t), sdof.drive(t))
    _run(live, 12.0)
    ts, y = live.record.window(2.0)
    loop = damping.loop_view(t, ts, y[:, COL["x"]], y[:, COL["v"]], y[:, COL["f"]], 0.5)
    amplitude = np.abs(loop.live.x).max()
    expected = math.pi * sdof.matched(t).c * TWO_PI * 2.0 * amplitude**2
    assert loop.energy == pytest.approx(expected, rel=0.01)
    assert [model for model, _ in loop.theory] == list(Model)


def test_the_phasors_add_up_to_the_drive():
    from vib_tutorial.core.damping_models import Model
    from vib_tutorial.view_models import damping

    for model in (Model.VISCOUS, Model.HYSTERETIC, Model.ZENER):
        t = _damping(model, drive=True, freq_hz=2.5, amplitude=2.0)
        spring, loss, mass, drive = damping.phasors(t, 0.37)
        assert drive.end == pytest.approx(mass.end) and math.hypot(*drive.end) == pytest.approx(2.0)
        assert spring.end == loss.start and loss.end == mass.start
    assert damping.phasors(_damping(Model.VISCOUS), 0.0) == ()
