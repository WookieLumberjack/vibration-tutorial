"""Pendulums: the exact period against the elliptic integral and a stepped swing, the phase plane's orbits."""

import math

import numpy as np
import pytest

from vib_tutorial.core.pendulum import (
    Kind,
    Pendulum,
    duffing_period,
    elliptic_period,
    measured_period,
    orbit,
    period,
    swing,
)


@pytest.mark.parametrize("degrees", [1, 30, 90, 150, 179])
def test_period_matches_the_elliptic_integral(degrees):
    p = Pendulum()
    a = math.radians(degrees)
    assert period(p, a) == pytest.approx(elliptic_period(p, a), rel=1e-9)


@pytest.mark.parametrize(
    "pendulum",
    [
        Pendulum(),
        Pendulum(Kind.ROD, pivot=0.2),
        Pendulum(Kind.INVERTED),
        Pendulum(Kind.TORSION, length=0.1, spring=0.5),
    ],
)
def test_period_matches_the_stepped_swing(pendulum):
    a = math.radians(100)
    t, th, _ = swing(pendulum, a, 8 * pendulum.period0, 20000)
    assert measured_period(t, th) == pytest.approx(period(pendulum, a), rel=1e-4)


def test_small_swings_are_linear():
    p = Pendulum(Kind.ROD, pivot=0.3)
    assert period(p, 1e-4) == pytest.approx(p.period0, rel=1e-8)
    _, th, tl = swing(p, 1e-3, 3 * p.period0)
    assert np.max(np.abs(th - tl)) < 1e-8


def test_duffing_is_second_order_in_the_amplitude():
    p = Pendulum()
    errors = [abs(duffing_period(p, a) - period(p, a)) / p.period0 for a in (0.2, 0.1)]
    assert errors[0] / errors[1] == pytest.approx(16, rel=0.1)  # the next term is O(A⁴)


def test_orbits_keep_their_energy_and_close():
    p = Pendulum()
    x = np.linspace(-math.pi, math.pi, 301)
    e = float(p.potential(1.0))
    ox, oy = orbit(p, e, x)
    keep = np.isfinite(ox)
    energy = 0.5 * p.inertia * oy[keep] ** 2 + p.potential(ox[keep])
    assert np.allclose(energy, e)
    assert np.nanmax(np.abs(ox)) == pytest.approx(1.0) and np.nanmin(np.abs(oy)) == 0.0


@pytest.mark.parametrize(
    "pendulum", [Pendulum(zeta=0.05), Pendulum(Kind.ROD, pivot=0.2), Pendulum(Kind.INVERTED, spring=15.0, zeta=0.02)]
)
def test_the_live_step_follows_the_stepped_swing_and_its_ghost_the_linear_one(pendulum):
    from vib_tutorial.core.pendulum import rk4, step_size

    a = math.radians(60)
    h = step_size(pendulum)
    n = round(3.0 / h)
    y = rk4(pendulum, (a, 0.0, a, 0.0), h, n)
    _, whole, linear = swing(pendulum, a, n * h, n + 1)
    assert np.allclose(y[:, 0], whole[1:], atol=1e-6) and np.allclose(y[:, 2], linear[1:], atol=1e-6)


def test_the_live_step_keeps_an_undamped_swings_energy_and_counts_the_dampers_work():
    from vib_tutorial.core.pendulum import rk4, step_size

    for p in (Pendulum(), Pendulum(zeta=0.1)):
        a, h = math.radians(170), step_size(Pendulum())
        y = rk4(p, (a, 0.0, a, 0.0), h, int(20 / h))
        stored = 0.5 * p.inertia * y[:, 1] ** 2 + p.potential(y[:, 0])
        assert stored + y[:, 4] == pytest.approx(float(p.potential(a)), abs=1e-7)


def test_an_unstable_pendulums_ghost_runs_away_and_stops():
    from vib_tutorial.core.pendulum import rk4, step_size

    p = Pendulum(Kind.INVERTED, spring=9.0, zeta=0.1)
    h = step_size(p)
    y = rk4(p, (0.02, 0.0, 0.02, 0.0), h, int(200 / h))
    assert np.isnan(y[-1, 2]) and abs(y[-1, 0]) == pytest.approx(p.leaning(), abs=1e-3)
