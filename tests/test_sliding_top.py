"""A top on a table: the round tip's equations against what they keep, and the table trials' numbers."""

import math

import numpy as np
import pytest

from vib_tutorial.core.gyroscope import RPM, Top
from vib_tutorial.core.sliding_top import Event, RoundTop, SlidingSimulator, reference

DISC = RoundTop.disc(Top(0.5, 0.04, 0.02), 0.01, 0.01)
TIPPE = RoundTop.tippe(0.02, 0.015, 0.0045, 0.8, 0.3)


def run(top: RoundTop, rpm: float, tilt: float, duration: float, per_period: int | None = None):
    sim = SlidingSimulator(top, rpm * RPM, math.radians(tilt), *([per_period] if per_period else []))
    s = sim.advance(duration)
    return sim, s, np.degrees(np.arccos(np.clip(s.axis[:, 2], -1, 1)))


def test_the_round_tops():
    assert DISC.offset == pytest.approx(0.01) and DISC.central == pytest.approx(0.5 * 0.04**2 / 4)
    assert TIPPE.alpha == pytest.approx(0.3) and TIPPE.ratio == pytest.approx(0.8) and TIPPE.inverts
    assert not RoundTop.tippe(0.02, 0.015, 0.0045, 1.4, 0.3).inverts
    th = DISC.rim_tilt()  # the rim just touches
    assert DISC.radius + DISC.offset * math.cos(th) == pytest.approx(DISC.rim * math.sin(th))
    assert TIPPE.rim_tilt() == math.pi


@pytest.mark.parametrize(
    "top, rpm, tilt", [(DISC, 1500, 20), (TIPPE, 1200, 5), (RoundTop.disc(Top(), 0.003, 0.3), 1500, 10)]
)
def test_jellett_and_the_energy_are_kept(top, rpm, tilt):
    """J whatever the friction does, and the energy with friction's work; G at its height; N up."""
    sim, s, theta = run(top, rpm, tilt, 2.5)
    J = top.jellett(rpm * RPM, math.radians(tilt))
    assert s.jellett == pytest.approx(J, rel=3e-6)
    start = sim.sample().energy[0] + sim.sample().work[0]
    assert s.energy + s.work == pytest.approx(start, rel=3e-6)
    assert np.all(np.diff(s.work) >= -1e-15) and sim.event is None and s.normal.min() > 0
    assert s.centre[:, 2] == pytest.approx(top.radius + top.offset * s.axis[:, 2])
    # the energy never goes under the least that J allows at its tilt
    assert np.all(s.energy >= top.least_energy(J, np.radians(theta)) * (1 - 1e-9))


def test_runge_kutta_is_fourth_order():
    top = RoundTop.disc(Top(0.5, 0.04, 0.02), 0.01, 0.3)
    errors = []
    for per in (32, 64):
        sim, _, _ = run(top, 1500, 20, 1.0, per)
        exact = reference(top, 1500 * RPM, math.radians(20), np.array([sim.t]))[0]
        errors.append(np.max(np.abs(np.array(sim.state[:9]) - exact[:9])))
    assert errors[0] / errors[1] == pytest.approx(16, rel=0.25)


def test_without_friction():
    """Nothing pushes G sideways: it stays over where it was let go, and the energy is kept."""
    top = RoundTop.disc(Top(0.5, 0.04, 0.02), 0.01, 0.0)
    sim, s, _ = run(top, 1500, 20, 2.0)
    assert np.abs(s.centre[:, :2]).max() < 1e-12 and np.abs(s.work).max() == 0.0
    assert s.energy == pytest.approx(sim.sample().energy[0], rel=1e-7)


def test_least_energy():
    J = TIPPE.jellett(1200 * RPM, math.radians(5))
    i3, a, R, w = TIPPE.polar, TIPPE.offset, TIPPE.radius, TIPPE.weight
    assert TIPPE.least_energy(J, 0.0) == pytest.approx(J**2 / (2 * i3 * (a + R) ** 2) + w * (R + a))
    assert TIPPE.least_energy(J, math.pi) == pytest.approx(J**2 / (2 * i3 * (R - a) ** 2) + w * (R - a))
    assert TIPPE.least_tilt(J) == math.pi and DISC.least_tilt(DISC.jellett(1500 * RPM, 0.3)) == 0.0
    assert TIPPE.end_spin(J, upright=False) == pytest.approx(J / (i3 * (R - a)))


@pytest.mark.parametrize("top", [DISC, TIPPE])
def test_the_turning_spin(top):
    """Upright is E_J's least on one side of it and not the other: the curvature at upright changes sign."""
    turn = top.turning_spin()

    def bend(spin):  # E_J(θ) − E_J(0) just off upright, for the J it would have upright
        J = top.polar * spin * (top.offset + top.radius)
        return top.least_energy(J, 1e-3) - top.least_energy(J, 0.0)

    above, below = bend(1.1 * turn), bend(0.9 * turn)
    assert (above > 0) == (top.offset > 0) and (below > 0) != (above > 0)


def test_events():
    sim, _, _ = run(RoundTop.disc(Top(0.5, 0.04, 0.03), 0.005, 0.02), 1500, 30, 10.0)
    assert sim.event is Event.HOPS  # it nods hard enough to leave the table
    sim, _, theta = run(RoundTop.disc(Top(0.5, 0.04, 0.02), 0.01, 0.3), 100, 20, 5.0)
    assert sim.event is Event.RIM and theta[-1] == pytest.approx(math.degrees(sim.top.rim_tilt()), abs=0.3)
    assert sim.advance(1.0).t.size == 0  # stopped


def test_trials():
    from vib_tutorial.gui.gyroscope import TRIALS

    by = {t.name: t for t in TRIALS}

    def go(name, duration):
        t = by[name]
        top, m = t.round_top, t.motion
        sim = SlidingSimulator(top, m.spin, m.tilt)
        s = sim.advance(duration)
        assert sim.event is None
        J = top.jellett(m.spin, m.tilt)
        return top, J, s, np.degrees(np.arccos(np.clip(s.axis[:, 2], -1, 1)))

    def first(t, mask):
        return t[np.flatnonzero(mask)[0]]

    top, J, s, theta = go("On a slippery table: it rises", 30.0)
    assert top.least_tilt(J) == 0.0 and math.degrees(top.rolling_tilt()) == pytest.approx(4.6, abs=0.05)
    assert first(s.t, theta < 10) == pytest.approx(12, abs=0.5) and first(s.t, theta < 5) == pytest.approx(19, abs=0.5)
    assert 4.0 < theta[-1] < 4.6
    spin = np.sum(s.momentum * s.axis, axis=1) / top.polar / RPM
    assert spin[-1] == pytest.approx(1457, abs=0.5) and top.end_spin(J) / RPM == pytest.approx(1455, abs=0.5)

    top, J, s, theta = go("On a rough table: it rolls instead", 12.0)
    assert top.rolling_tilt() == 0.5 * math.pi
    assert theta[s.t > 0.5] == pytest.approx(16.2, abs=0.05)
    across = np.max(np.hypot(s.centre[:, 0], s.centre[:, 1]))  # let go on the circle: its diameter
    assert across == pytest.approx(1.35, abs=0.025)

    top, J, s, theta = go("A tippe top turns over", 12.0)
    assert top.turning_spin() / RPM == pytest.approx(468, abs=0.5) and top.inverts
    assert first(s.t, theta > 90) == pytest.approx(1.8, abs=0.05) and theta[-1] > 175
    over = top.end_spin(J, upright=False) / RPM
    assert over == pytest.approx(643, abs=0.5)
    assert np.sum(s.momentum[-1] * s.axis[-1]) / top.polar / RPM == pytest.approx(-over, rel=2e-3)
    assert 1000 * (s.centre[-1, 2] - s.centre[0, 2]) == pytest.approx(9.0, abs=0.05)
    assert s.work[-1] / (s.energy[0] + s.work[0]) == pytest.approx(0.5, abs=0.03)

    top, J, s, theta = go("A tippe top spun too slowly", 10.0)
    assert top.least_tilt(J) == 0.0 and theta.max() < 6

    top, J, s, theta = go("A tippe top that can't turn over", 15.0)
    assert math.degrees(top.least_tilt(J)) == pytest.approx(108.6, abs=0.05)
    assert np.mean(theta[s.t > 5]) == pytest.approx(110, abs=3)
