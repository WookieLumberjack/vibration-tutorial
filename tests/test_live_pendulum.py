"""The live pendulum (#478 Phase 4): a pluck, the live changes' rules, the energy balance, and the Pendulums view
model's panels (`view_models/live_pendulum.py`, `view_models/pendulum.py`)."""

import math
from dataclasses import replace

import numpy as np
import pytest

from vib_tutorial.core.pendulum import Kind, Pendulum, period
from vib_tutorial.core.plain import impure
from vib_tutorial.view_models import pendulum as v
from vib_tutorial.view_models.free_body import ArrowScale
from vib_tutorial.view_models.live_pendulum import COL, LivePendulum, energy, energy_view, ghost_angle


def plucked(p: Pendulum, degrees: float) -> LivePendulum:
    live = LivePendulum(p)
    live.hold(math.radians(degrees))
    live.let_go()
    return live


def run(live: LivePendulum, seconds: float, frame: float = 1 / 60) -> None:
    for _ in range(round(seconds / frame)):
        live.advance(frame)


def closes(live: LivePendulum) -> float:
    view = energy_view(live.record.t[: live.record.size], live.record.y[: live.record.size])
    assert view is not None
    return abs(view.residual)


def test_a_pluck_swings_at_the_exact_period_and_the_hand_put_in_what_it_stores():
    p = Pendulum()
    live = plucked(p, 90)
    assert live.released == pytest.approx(float(p.potential(math.pi / 2)))
    run(live, 3.5 * period(p, math.pi / 2))  # a pluck's first maximum is the release itself, with no sample before it
    assert live.t == pytest.approx(3.5 * period(p, math.pi / 2), abs=0.02), (
        "the frames' time, the part of a step carried"
    )
    t, y = live.since_change()
    swings = v.cycles(t, y[:, COL["theta"]])
    assert len(swings) == 2
    for _, amplitude, measured in swings:
        assert amplitude == pytest.approx(math.pi / 2, abs=1e-6)
        assert measured == pytest.approx(period(p, math.pi / 2), rel=1e-6)
    assert closes(live) < 1e-8, "RK4's drift: under a part in 10⁸ of the 4.9 J stored"


def test_held_it_stands_still_and_the_clock_waits():
    live = LivePendulum()
    live.hold(0.3)
    assert live.advance(1.0) == 0 and live.t == 0.0 and live.acceleration == 0.0


def test_shortening_the_string_keeps_its_angular_momentum_and_books_the_work():
    p = Pendulum()
    live = plucked(p, 30)
    run(live, p.period0 / 4)  # about the bottom, swinging fast
    w, stored = live.omega, live.stored
    shorter = replace(p, length=0.5)
    live.set_pendulum(shorter)
    assert live.omega == pytest.approx(4 * w), "J = mL²: a quarter of the inertia, four times the speed"
    assert live.changes == pytest.approx(live.stored - stored) and live.changes > 0
    assert live.edits == [live.t] and live.changed_at == live.t
    run(live, 2.0)
    assert closes(live) < 1e-8


def test_pumping_a_swing_shortening_at_the_bottom_and_letting_out_at_the_ends_grows_it():
    """A playground swing: pulled in where it's fastest and let out where it stops, each swing grows, and the
    work the hand does is booked as the changes'."""
    long, short = Pendulum(length=1.0), Pendulum(length=0.9)
    live = plucked(long, 10)
    for _ in range(12):  # half swings
        while not abs(live.theta) < 0.01:
            live.advance(0.002)
        live.set_pendulum(short)
        while abs(live.omega) > 0.02:
            live.advance(0.002)
        live.set_pendulum(long)
    assert abs(live.theta) > math.radians(20), "each half swing grows it"
    assert live.changes > 0 and closes(live) < 1e-8


def test_gravity_changed_keeps_the_angle_and_speed_and_books_the_jump():
    p = Pendulum()
    live = plucked(p, 40)
    run(live, 0.3)
    th, w, before = live.theta, live.omega, live.stored
    moon = replace(p, gravity=1.62)
    live.set_pendulum(moon)
    assert (live.theta, live.omega) == (th, w)
    assert live.changes == pytest.approx(energy(moon, th, w) - before)


def test_an_inverted_pendulum_on_a_weak_spring_falls_over_and_settles_leaning():
    p = Pendulum(Kind.INVERTED, spring=9.0, zeta=0.1)
    live = plucked(p, 1)
    run(live, 200.0, 0.25)
    assert abs(live.theta) == pytest.approx(p.leaning(), abs=1e-3)
    assert ghost_angle(live.ghost) is None, "its linearization ran away"
    assert closes(live) < 1e-7


def test_the_free_body_diagrams_moments_add_up_to_j_times_its_angular_acceleration():
    for p in (
        Pendulum(zeta=0.1),
        Pendulum(Kind.ROD, pivot=0.2, zeta=0.1),
        Pendulum(Kind.TORSION, length=0.1, spring=0.5, zeta=0.1),
        Pendulum(Kind.INVERTED, zeta=0.1),
    ):
        live = plucked(p, 50)
        run(live, 0.37)
        body = v.free_body(p, live.theta, live.omega)
        assert body.net.value == pytest.approx(p.inertia * live.acceleration)
        names = [a.name for a in body.arrows]
        assert ("weight" in names) == (p.kind is not Kind.TORSION)
        assert ("spring" in names) == (p.kind in (Kind.TORSION, Kind.INVERTED))
        assert all("N·m" in a.label for a in (*body.arrows, body.net)) and "N" in body.aside[0].label
        held = v.free_body(p, 0.4, 0.0, held=True)
        assert held.net.value == pytest.approx(0.0, abs=1e-12), "the hand balances it"
        assert impure(body) == []


def test_the_strings_tension_is_most_at_the_bottom_and_turns_the_bob():
    p = Pendulum()
    live = plucked(p, 90)
    run(live, period(p, math.pi / 2) / 4)  # at the bottom
    _, _, tension = v.pull(p, live.theta, live.omega)
    assert tension == pytest.approx(3 * p.mass * p.gravity, rel=2e-3), (
        "from 90°: mg + mLθ̇² = 3mg (to the frame nearest the bottom)"
    )
    scale, aside = ArrowScale(1.0), ArrowScale(1.0)
    body = v.free_body(p, live.theta, live.omega, scale=scale, aside_scale=aside, t=live.t)
    assert body.aside_scale * abs(body.aside[0].value) == pytest.approx(0.8), "its own scale fills its room"


def test_each_swing_lands_its_period_and_a_damped_one_traces_the_curve():
    p = Pendulum(zeta=0.01)
    live = plucked(p, 170)
    rec = v.PeriodRecorder()
    for _ in range(40):
        run(live, 1.0)
        t, y = live.since_change()
        rec.update(t, y[:, COL["theta"]], live.changed_at, v.key(p))
    points = rec.current(v.key(p))
    assert len(points) > 10
    amplitudes = [q.amplitude for q in points]
    assert amplitudes == sorted(amplitudes, reverse=True), "a swing a point, as it dies away"
    for q in points:
        assert q.period == pytest.approx(period(replace(p, zeta=0.0), q.amplitude), rel=0.01)


def test_an_undamped_swing_lands_one_point_again_and_another_pendulums_stay_faded():
    p = Pendulum()
    live = plucked(p, 45)
    rec = v.PeriodRecorder()
    for _ in range(4):
        run(live, 2.5)
        t, y = live.since_change()
        rec.update(t, y[:, COL["theta"]], live.changed_at, v.key(p))
    assert len(rec.current(v.key(p))) == 1
    longer = replace(p, length=2.0)
    live.set_pendulum(longer)
    run(live, 8.0)
    t, y = live.since_change()
    rec.update(t, y[:, COL["theta"]], live.changed_at, v.key(longer))
    assert len(rec.current(v.key(longer))) == 1 and len(rec.earlier(v.key(longer))) == 1


def test_the_period_tabs_axes_keep_still_when_the_theory_is_shown():
    p = Pendulum()
    pts = [v.PeriodPoint(math.radians(a), period(p, math.radians(a)), v.key(p), a) for a in (20, 60, 120)]
    off, on = v.period_view(p, pts, [], False), v.period_view(p, pts, [], True)
    assert (off.x_span, off.y_span) == (on.x_span, on.y_span)
    assert off.exact.x.size == 0 and on.exact.x.size > 0
    assert on.y_span[0] < min(on.points.y) and on.y_span[1] > max(on.points.y)
    assert on.note.startswith("The last swing, ±120°")
    unstable = v.period_view(Pendulum(Kind.INVERTED, spring=5.0), [], [], True)
    assert unstable.exact.x.size == 0 and "unstable" in unstable.note
    assert impure(on) == []


def test_the_phase_plane_and_moment_views():
    p = Pendulum()
    th = np.linspace(-1, 1, 50)
    view = v.phase_view(p, th, np.cos(th), v.phase_reach(p, 1.0))
    assert view.separatrix.x.size > 0 and np.nanmax(np.abs(view.orbits.x)) <= 190
    disc = v.phase_view(
        Pendulum(Kind.TORSION, length=0.1, spring=0.5), th, th, v.phase_reach(Pendulum(Kind.TORSION), 1.0)
    )
    assert disc.separatrix.x.size == 0
    m = v.moment_view(p, 0.5, 1.0)
    assert m.now.y[0] == pytest.approx(p.mass * p.gravity * math.sin(0.5))
    assert impure(view) == [] and impure(m) == []


def test_the_grips():
    assert v.GRIPS["length"].dragged(1.0, 0.5) == pytest.approx(math.e)
    assert v.pivot_dragged(1.0, 0.3, 0.5) == 0.5 and v.pivot_dragged(1.0, 0.1, -0.5) == pytest.approx(0.02)
    state = v.PendulumTrial("", "", kind=Kind.ROD, pivot=0.0)
    assert v.pendulum(state).pivot == pytest.approx(0.02)
    assert "T₀ = 2.006 s" in v.info(Pendulum())
