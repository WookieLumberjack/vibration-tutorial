"""The machine on its mounts live (#478 Phase 4, Vibration isolation): a dwell lands the closed-form transmissibility
for every kind of mount, shaken by the machine or by the floor; a change while it runs keeps the machine's place and
its momentum; a ramp's speed and angle; the floor force built up by speed; the free-body diagram's sum."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vib_tutorial.core.isolation import Isolator, Mount, held, live_system, start_up
from vib_tutorial.core.sdof import TWO_PI
from vib_tutorial.view_models import live_isolation as li
from vib_tutorial.view_models.live_isolation import COL, FLOOR, MACHINE, LiveIsolation

FRAME = 1 / 60


def run(live: LiveIsolation, seconds: float) -> None:
    for _ in range(round(seconds / FRAME)):
        live.advance(FRAME)


def column(live: LiveIsolation, name: str) -> tuple[np.ndarray, np.ndarray]:
    n = live.record.size
    return live.record.t[:n], live.record.y[:n, COL[name]]


@pytest.mark.parametrize("shaking", [MACHINE, FLOOR])
@pytest.mark.parametrize("mount", list(Mount))
def test_the_live_state_space_is_the_closed_forms_both_ways(mount, shaking):
    iso = Isolator(mount=mount)
    s = live_system(iso)
    for f in (3.0, 7.05, 25.0):
        w = TWO_PI * f
        Z = np.linalg.solve(1j * w * np.eye(s.A.shape[0]) - s.A, s.B)  # the state per unit of each input
        if shaking == MACHINE:
            T = s.D @ Z[:, 0] + s.E[0]  # the floor's force over the unbalance's
        else:
            T = Z[0, 1] + 1j * w * Z[0, 2]  # the machine's motion over the floor's (g and ġ = iωg)
        assert abs(T) == pytest.approx(iso.transmissibility(np.array([f]))[0], rel=1e-9)


@pytest.mark.parametrize("shaking", [MACHINE, FLOOR])
@pytest.mark.parametrize("mount", list(Mount))
def test_a_dwell_lands_the_transmissibility_once_the_ring_has_died(mount, shaking):
    iso = Isolator(mount=mount)
    live = LiveIsolation(iso, shaking)
    landed = None
    while live.t < 10.0 and landed is None:
        live.advance(FRAME)
        m = li.dwell(live)
        landed = m if m is not None and m.steady else None
    assert landed is not None and landed.freq_hz == 25.0
    assert abs(landed.response) == pytest.approx(iso.transmissibility(np.array([25.0]))[0], rel=5e-4)
    assert live.t > 1.0  # the start's ring at fₙ had to die away first


def test_an_undamped_mount_never_settles():
    live = LiveIsolation(Isolator(zeta=0.0))
    run(live, 6.0)
    m = li.dwell(live)
    assert m is not None and not m.steady  # the start's ring beats with the speed for good


def test_a_change_keeps_the_machines_place_and_momentum():
    live = LiveIsolation(Isolator())
    run(live, 1.0)
    x, v = live.x, live.v
    live.set_isolator(Isolator(deflection=0.002))  # stiffer mounts: x and v carry on
    assert (live.x, live.v) == (x, v) and live.changed_at == live.t and live.edits == [live.t]
    live.set_isolator(Isolator(deflection=0.002, block=100.0))  # twice the mass: captured at rest
    assert live.x == x and live.v == pytest.approx(v / 2)
    v = live.v
    live.set_isolator(Isolator(deflection=0.002))  # the block let go: the machine keeps its speed
    assert live.v == v
    live.set_isolator(Isolator(deflection=0.002, mount=Mount.RELAXATION))  # its spring starts unstretched
    assert live.z[2] == live.x
    live.set_isolator(Isolator(deflection=0.002, mount=Mount.TWO_STAGE))  # the stage starts at rest, balanced
    stage = held(live.iso, live.x)
    assert live.z[1] == pytest.approx(stage[1]) and live.z[3] == 0.0 and live.v == v
    assert len(live.edits) == 1  # a run of changes is one mark


def test_a_run_up_ramps_the_speed_and_turns_the_unbalance_through_its_integral():
    live = LiveIsolation(Isolator(rpm=0.0))
    live.set_speed(1500.0, 2.0)
    assert live.ramping and live.changed_at == 2.0  # a dwell waits for the ramp's end
    run(live, 1.0)
    assert live.speed_hz == pytest.approx(12.5, rel=1e-3) and li.dwell(live) is None
    run(live, 1.5)
    assert not live.ramping and live.speed_hz == 25.0
    t, phase = column(live, "phase")
    i = int(np.searchsorted(t, 2.0))
    assert phase[i] == pytest.approx(0.5 * (TWO_PI * 25 / 2.0) * t[i] ** 2, rel=1e-6)  # ½αt²
    live.set_speed(600.0)  # at once: the angle runs on, no jump
    before = live.phase
    live.advance(FRAME)
    assert live.speed_hz == 10.0 and 0 < live.phase - before < TWO_PI * 10 * 2 * FRAME


def test_the_floor_force_built_by_speed_matches_the_precomputed_run_up():
    iso = Isolator(rpm=0.0)
    live = LiveIsolation(iso)
    live.set_speed(1500.0, 2.0)
    run(live, 3.0)
    t, floor = column(live, "floor")
    rpm, peak = li.by_speed(t, column(live, "speed")[1], column(live, "phase")[1], floor)
    assert np.all(np.diff(t) > 0) and rpm[0] > 0 and rpm[-1] == pytest.approx(1500.0)
    ref = start_up(Isolator(rpm=1500.0), 2.0, 1.0)
    assert peak.max() == pytest.approx(np.abs(ref.floor).max(), rel=0.01)  # 111 N through resonance
    assert 60 * ref.speed_hz[np.argmax(np.abs(ref.floor))] == pytest.approx(
        rpm[np.argmax(peak)], rel=0.1
    )  # a turn at 10 Hz spans 75 rpm


def test_switched_to_the_floor_it_starts_from_its_place():
    live = LiveIsolation(Isolator())
    run(live, 0.5)
    live.set_shaking(FLOOR)
    assert live.inputs()[1] == pytest.approx(0.0, abs=1e-12) and live.inputs()[0] == 0.0
    live.advance(FRAME)
    _, g = column(live, "g")
    assert np.abs(g).max() <= li.FLOOR_AMPLITUDE and li.output_input(live) == ("x", "g")


@pytest.mark.parametrize("mount", list(Mount))
def test_held_the_clock_stops_and_let_go_it_rings(mount):
    live = LiveIsolation(Isolator(mount=mount, rpm=0.0))
    live.hold(0.002)
    assert live.advance(1.0) == 0 and live.t == 0.0
    body = li.free_body(live)
    assert body.net.value == pytest.approx(0.0, abs=1e-9)  # the hand holds it still
    live.let_go()
    run(live, 0.5)
    assert np.abs(column(live, "x")[1]).max() == pytest.approx(0.002, rel=1e-2)  # from a step after letting go


@pytest.mark.parametrize("mount", list(Mount))
def test_the_free_bodys_forces_add_up_to_mass_times_acceleration(mount):
    live = LiveIsolation(Isolator(mount=mount, block=20.0))
    run(live, 0.7)
    body = li.free_body(live)
    s = live_system(live.iso)
    a = (s.A @ live.z + s.B @ live.inputs())[s.velocity]
    assert body.net.value == pytest.approx(live.iso.total * a, rel=1e-9)
    assert body.aside[0].value == pytest.approx(live.floor_force()) and body.aside[0].name == "floor"
    names = [r.name for r in body.arrows]
    assert names == (["spring", "arm", "unbalance"] if mount is Mount.RELAXATION else ["spring", "damper", "unbalance"])


def test_the_step_takes_100_a_turn_of_the_speed():
    live = LiveIsolation(Isolator(rpm=3000.0, deflection=0.05))  # 50 Hz, the mounts slow
    h = live.step_size()
    assert h <= 1 / (100 * 50) and h > 1 / (2 * 100 * 50) and math.log2(h) == int(math.log2(h))


def test_the_grips_scale_their_values():
    assert li.GRIPS["deflection_mm"].dragged(5.0, 0.4) == pytest.approx(5.0 * math.e)
    assert li.GRIPS["zeta"].dragged(0.01, -1.0) == 0.0  # dragged down past its least, no damping
