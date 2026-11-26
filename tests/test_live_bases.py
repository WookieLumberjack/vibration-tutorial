"""Coordinates & bases' live run and view model, without Qt (`view_models/live_bases.py`, `view_models/bases.py`):
what a live change keeps, the rail's rules, the balance closing through them, and what each panel shows."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from vib_tutorial.core.bases import TWO_PI, TwoMass, direction, rayleigh_hz
from vib_tutorial.view_models import bases as view
from vib_tutorial.view_models.live_bases import COL, LiveBases, energy_view


def plucked(x=(0.05, 0.0), system: TwoMass | None = None) -> LiveBases:
    live = LiveBases(system)
    live.hold(np.array(x))
    live.let_go()
    return live


def balance(live: LiveBases) -> float:
    r = live.record
    e = energy_view(r.t[: r.size], r.y[: r.size])
    assert e is not None
    return e.residual


def test_a_pluck_books_the_hands_work_and_runs_on_from_rest():
    live = plucked()
    assert live.books["released"] == pytest.approx(0.5 * 500.0 * 0.05**2)  # k₁ + k₂ on x₁
    live.advance(1.0)
    assert live.t > 0.99 and abs(balance(live)) < 1e-12


def test_a_heavier_mass_keeps_its_momentum_and_a_spring_books_its_work():
    live = plucked()
    live.advance(0.37)
    x, v = live.x.copy(), live.v.copy()
    live.set_system(replace(live.system, m1=2.0))
    assert live.v[0] == pytest.approx(v[0] / 2) and live.v[1] == pytest.approx(v[1])
    assert live.books["capture_loss"] == pytest.approx(0.25 * v[0] ** 2)  # ½ m v² Δm / (m + Δm)
    live.set_system(replace(live.system, k2=300.0))
    assert live.books["spring_work"] == pytest.approx(0.5 * 200.0 * (x[0] - x[1]) ** 2, rel=1e-9)
    assert np.allclose(live.x, x)
    live.set_system(replace(live.system, m2=0.5))  # a lighter m₂ drops a piece at its speed
    assert live.v[1] == pytest.approx(v[1]) and live.books["carried_off"] == pytest.approx(0.25 * v[1] ** 2)
    live.advance(1.0)
    assert abs(balance(live)) < 1e-12 and len(live.edits) == 1  # three changes at once: one mark


def test_on_a_rail_it_swings_at_rayleighs_frequency_and_keeps_umv_through_a_turn():
    live = plucked()
    live.advance(0.21)
    M, _, K = live.system.matrices()
    v = live.v.copy()
    u = direction(30.0)
    live.set_rail(u)
    assert np.allclose(live.x, u * float(u @ live.x))  # on the rail
    assert float(u @ M @ live.v) == pytest.approx(float(u @ M @ v))  # the rail's impulse keeps uᵀMv
    s = float(live.s[0])
    live.set_rail(direction(60.0))
    assert float(live.s[0]) == pytest.approx(s)  # carried round at the same distance
    live.advance(2.0)
    assert live.rail_hz() == pytest.approx(rayleigh_hz(direction(60.0), M, K), rel=1e-4)
    live.set_rail(None)  # off the rail: x and v carry on
    x = live.x.copy()
    live.advance(0.5)
    assert live.steps.n == 2 and not np.allclose(live.x, x)
    assert abs(balance(live)) < 1e-12 and live.books["rail_work"] != 0


def test_a_rail_turned_past_upright_keeps_the_point_on_its_side():
    live = plucked((0.0, 0.0))
    live.set_rail(direction(85.0))
    live.hold(np.array([0.0, 0.04]))
    live.set_rail(direction(95.0))  # `unit` flips the direction's sign; the point stays up
    s = 0.04 * math.sin(math.radians(85.0))  # held at its nearest place on the 85° rail
    assert live.x[1] == pytest.approx(s * math.sin(math.radians(95.0)), rel=1e-9) and live.x[1] > 0


def test_held_it_stands_still_and_reset_clears_the_books():
    live = plucked()
    live.hold(np.array([0.01, 0.02]))
    assert live.advance(1.0) == 0 and live.t == 0
    live.let_go()
    live.advance(0.5)
    live.reset()
    assert live.t == 0 and live.record.size == 0 and not any(live.books.values())


# ------------------------------------------------------------------ the view model


def test_snapping_an_arrow_onto_a_mode():
    s = TwoMass()
    tip = view.ARROW * direction(46.5)  # 1.5° off mode 1
    (vx, vy), snapped = view.arrow_dragged(tip, view.snap_targets(s))
    assert snapped == 0 and vx == pytest.approx(vy) and math.hypot(vx, vy) == pytest.approx(1.0)
    _, none = view.arrow_dragged(view.ARROW * direction(52.0), view.snap_targets(s))
    assert none is None
    (vx, vy), _ = view.arrow_dragged((10.0, 0.0))  # far out: the longest arrow
    assert math.hypot(vx, vy) == pytest.approx(view.LENGTHS[1])


def test_the_matrices_say_how_the_coordinates_are_coupled():
    s = TwoMass()
    _, P = s.modes()
    cases = {
        "static coupling": np.eye(2),  # each mass's own displacement: the springs couple them
        "uncoupled": P,  # the modes
        "dynamic coupling": None,  # filled below: VᵀKV diagonal, VᵀMV not
        "both": np.array([[1.0, 0.0], [0.5, 1.0]]),
    }
    # In K's own eigenvectors (perpendicular) with unequal masses, K is diagonal and M isn't.
    a = TwoMass(m1=1.0, m2=3.0)
    _, _, Ka = a.matrices()
    _, E = np.linalg.eigh(Ka)
    cases["dynamic coupling"] = E
    notes = {name: view.matrices_view(a if name == "dynamic coupling" else s, V, False) for name, V in cases.items()}
    assert "(static coupling)" in notes["static coupling"].note
    assert notes["uncoupled"].note.startswith("Both diagonal")
    assert not any(m.coupled for m in notes["uncoupled"].matrices)
    assert "(dynamic coupling)" in notes["dynamic coupling"].note
    assert notes["both"].note.startswith("Both couple")
    hz, _ = s.modes()
    for name in ("static coupling", "uncoupled", "both"):
        assert notes[name].hz == pytest.approx(tuple(hz)), name
    flat = view.matrices_view(s, np.array([[1.0, 2.0], [1.0, 2.0]]), False)
    assert not flat.matrices and "one line" in flat.note
    rail = view.matrices_view(s, np.eye(2), True)
    assert rail.hz == pytest.approx((math.sqrt(500.0) / TWO_PI,)) and "one pure sine" in rail.note


def test_in_the_modes_each_coordinate_keeps_its_energy_and_in_x_it_passes_between_them():
    s = TwoMass()
    live = plucked(system=s)
    live.advance(4.0)
    r = live.record
    x = r.y[: r.size, COL["x1"] : COL["x2"] + 1]
    v = r.y[: r.size, COL["v1"] : COL["v2"] + 1]
    stored = r.y[: r.size, COL["kinetic"]] + r.y[: r.size, COL["potential"]]
    _, P = s.modes()
    for V in (np.eye(2), P, np.array([[1.0, 0.3], [-0.2, 1.0]])):
        sh = view.shares(s, V, False, x, v)
        assert sh.own[0] + sh.own[1] + sh.coupling == pytest.approx(stored, rel=1e-9)
    modal = view.shares(s, P, False, x, v)
    assert np.ptp(modal.own[0]) < 1e-9 * stored[0] and modal.note.startswith("Uncoupled")
    plain = view.shares(s, np.eye(2), False, x, v)
    assert np.ptp(plain.own[1]) > 0.5 * stored[0]  # the beat carries the energy to m₂ and back


def test_the_coordinates_rebuild_x_and_the_parallelogram_closes():
    V = np.array([[1.0, 0.4], [0.3, -1.2]])
    x = np.array([[0.01, 0.02], [-0.03, 0.005]])
    q = view.coordinates(V, False, x)
    assert q @ V.T == pytest.approx(x)
    xs, ys = view.projection(V, False, x[0])
    assert (xs[1], ys[1]) == pytest.approx(tuple(x[0] * view.MM))


def test_the_one_arrow_tab_lands_each_angle_and_keeps_its_axes_with_the_theory():
    s = TwoMass()
    rec = view.RailRecorder()
    k = view.key(s)
    assert rec.update(30.0, 3.4, k, 1.0) and rec.update(30.2, 3.41, k, 2.0)  # within ½°: replaced
    assert len(rec.current(k)) == 1 and rec.update(120.0, 3.8, k, 3.0)
    plain = view.rail_view(s, rec.current(k), [], False, 120.0)
    shown = view.rail_view(s, rec.current(k), [], True, 120.0)
    assert (plain.x_span, plain.y_span) == (shown.x_span, shown.y_span)
    assert shown.mode_angles == pytest.approx((45.0, 135.0)) and shown.quotient[1].min() == pytest.approx(
        s.modes()[0][0], rel=1e-4
    )
    assert "2 angles so far" in plain.note


def test_a_trial_survives_json_and_the_views_are_plain():
    import json

    from vib_tutorial.core.plain import from_plain, impure, to_plain

    t = view.BasesTrial("a", "b", v1=(0.7, 0.7), rail=True, pluck=(0.02, -0.01))
    assert from_plain(view.BasesTrial, json.loads(json.dumps(to_plain(t)))) == t
    s, V = view.system(t), view.basis(t)
    for out in (
        view.plane_view(s, V, False, True),
        view.matrices_view(s, V, False),
        view.rail_view(s, [], [], True, 45.0),
        view.shares(s, V, True, np.zeros((3, 2)), np.zeros((3, 2))),
    ):
        assert impure(out) == [], type(out).__name__
