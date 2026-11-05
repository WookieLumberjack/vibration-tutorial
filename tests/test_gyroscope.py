"""The spinning top: the closed forms against the whole body stepped, and the Gyroscope page's trials' numbers."""

import math

import numpy as np
import pytest

from vib_tutorial.core.gyroscope import (
    RPM,
    FreeSpin,
    Launch,
    Motion,
    Path,
    Shape,
    Top,
    TopSimulator,
    body_rates,
    body_reference,
    euler_reference,
    least_spin,
    steady_precession,
    tilt_and_azimuth,
)

TOP = Top()


def run(motion: Motion, duration: float, per_period: int = 200):
    sim = TopSimulator(motion, per_period)
    s = sim.advance(duration)
    return sim, s, *tilt_and_azimuth(s.axis)


def nod_period(t: np.ndarray, theta: np.ndarray) -> float:
    """The mean time between the tilt's maxima."""
    i = np.flatnonzero((theta[1:-1] > theta[:-2]) & (theta[1:-1] >= theta[2:])) + 1
    return float(np.mean(np.diff(t[i])))


def test_moments_of_inertia():
    assert TOP.polar == pytest.approx(0.5 * 0.5 * 0.04**2)
    assert TOP.transverse == pytest.approx(0.5 * (0.04**2 / 4 + 0.03**2))
    assert least_spin(TOP, 0.0) == pytest.approx(TOP.sleeping_spin)
    assert least_spin(TOP, math.radians(90)) == 0.0


def test_a_square_is_a_symmetric_top():
    """Summed over a fine grid, a square plate's moment is ms²/12 about every line across its axis, at any angle,
    and ms²/6 along it: a disc's of radius s/√3, so it moves exactly as that disc does."""
    m, s, n = 0.5, 0.06, 400
    x, y = np.meshgrid((np.arange(n) + 0.5) / n - 0.5, (np.arange(n) + 0.5) / n - 0.5)
    x, y, dm = s * x.ravel(), s * y.ravel(), m / n**2
    square = Top(m, s, 0.03, shape=Shape.SQUARE)
    for beta in np.radians([0.0, 17.0, 45.0, 70.0]):
        across = -x * math.sin(beta) + y * math.cos(beta)  # the distance from the line at angle β
        assert np.sum(across**2) * dm == pytest.approx(square.plate, rel=1e-4)
    assert np.sum(x * x + y * y) * dm == pytest.approx(square.polar, rel=1e-4)
    disc = Top(m, s / math.sqrt(3), 0.03)
    assert (square.polar, square.transverse) == pytest.approx((disc.polar, disc.transverse))
    assert square.corner == pytest.approx(s / math.sqrt(2))


@pytest.mark.parametrize("tilt", [10.0, 30.0, 60.0, 90.0, 120.0])
def test_steady_precession_holds_the_tilt(tilt):
    th, spin = math.radians(tilt), 1500 * RPM
    for rate in steady_precession(TOP, spin, th):
        if not math.isfinite(rate):
            continue
        assert TOP.transverse * math.cos(th) * rate**2 - TOP.polar * spin * rate + TOP.weight_moment == pytest.approx(
            0.0, abs=1e-12
        )
        m = Motion(TOP, spin, th, rate)
        assert m.steady and m.path() is Path.STEADY and m.mean_precession() == pytest.approx(rate)
        _, _, theta, _ = run(m, 0.5)
        assert np.ptp(theta) < 1e-5  # the split's own steady state differs by its O(h²)
    slow, _ = steady_precession(TOP, spin, th)
    assert slow == pytest.approx(TOP.weight_moment / (TOP.polar * spin), rel=0.1)


def test_no_steady_precession_under_the_least_spin():
    th = math.radians(30)
    least = least_spin(TOP, th)
    assert all(math.isfinite(r) for r in steady_precession(TOP, 1.001 * least, th))
    assert not any(math.isfinite(r) for r in steady_precession(TOP, 0.999 * least, th))
    slow, fast = steady_precession(TOP, least, th)
    assert slow == pytest.approx(fast, rel=1e-3)
    assert Motion.launched(TOP, 0.9 * least, th, Launch.SLOW).precession == 0.0  # at rest where there's none


@pytest.mark.parametrize("rate", [-2.0, 0.0, 1.2, 4.0])
def test_the_closed_forms_match_the_simulation(rate):
    m = Motion(TOP, 1500 * RPM, math.radians(30), rate)
    T = m.nutation_period()
    _, s, theta, phi = run(m, 40 * T)
    hi, lo = m.turning()
    assert (theta.min(), theta.max()) == pytest.approx((hi, lo), abs=2e-5)
    assert nod_period(s.t, theta) == pytest.approx(T, rel=1e-4)
    i = int(np.searchsorted(s.t, 40 * T)) - 1
    assert phi[i] / s.t[i] == pytest.approx(m.mean_precession(), rel=1e-3)
    # what's conserved: the momenta to round-off, the energy to the step's error
    assert np.abs(s.momentum[:, 2] - m.vertical_momentum).max() < 1e-12
    assert np.abs(np.sum(s.momentum * s.axis, axis=1) - m.spin_momentum).max() < 1e-12
    assert np.abs(s.energy - m.total_energy).max() < 1e-5 * TOP.weight_moment


def test_the_euler_angle_equations_agree():
    m = Motion(TOP, 600 * RPM, math.radians(30), -1.0)
    _, s, theta, phi = run(m, 2.0)
    k = slice(None, None, 50)
    th, ph = euler_reference(m, s.t[k])
    assert np.abs(th - theta[k]).max() < 5e-5 and np.abs(ph - phi[k]).max() < 1e-5 * np.abs(ph).max()  # second order


def test_the_energy_error_falls_as_h_squared():
    m = Motion(TOP, 600 * RPM, math.radians(30))
    errors = []
    for per in (50, 100):
        _, s, _, _ = run(m, 1.0, per)
        errors.append(np.abs(s.energy - m.total_energy).max())
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.1)


def test_the_paths():
    th, spin = math.radians(30), 1500 * RPM
    assert Motion(TOP, spin, th).path() is Path.CUSPS
    assert Motion(TOP, spin, th, -2.0).path() is Path.LOOPS
    assert Motion(TOP, spin, th, 1.2).path() is Path.WAVES
    assert Motion(TOP, spin, th, 4.0).path() is Path.WAVES
    assert Motion(TOP, -spin, th).mean_precession() == pytest.approx(-Motion(TOP, spin, th).mean_precession())


def test_the_sleeping_top():
    th = math.radians(2)
    awake = Motion(TOP, 1.05 * TOP.sleeping_spin, th)
    falls = Motion(TOP, 0.95 * TOP.sleeping_spin, th)
    assert math.degrees(awake.turning()[1]) < 10 < math.degrees(falls.turning()[1])
    _, _, theta, _ = run(falls, 2 * falls.nutation_period())
    assert theta.max() == pytest.approx(falls.turning()[1], abs=1e-4)
    # nearly upright, through the place the Euler angles lock: the quaternion doesn't mind
    m = Motion(TOP, 1000 * RPM, math.radians(0.5))
    _, _, theta, _ = run(m, 1.0)
    assert theta.max() == pytest.approx(m.turning()[1], abs=1e-5)


def test_the_free_top():
    """Without gravity L stays put and the axis turns round it at |L|/I₁: the nod's period."""
    free = Top(gravity=0.0)
    m = Motion(free, 1500 * RPM, math.radians(30), 5.0)
    L = m.angular_momentum()
    assert m.nutation_period() == pytest.approx(2 * math.pi * free.transverse / np.linalg.norm(L))
    _, s, theta, _ = run(m, 0.5)
    assert np.abs(s.momentum - L).max() < 1e-12
    cone = np.arccos(s.axis @ L / np.linalg.norm(L))
    assert np.ptp(cone) < 1e-9
    assert nod_period(s.t, theta) == pytest.approx(m.nutation_period(), rel=1e-4)


def test_trials():
    from vib_tutorial.gui.gyroscope import PLAYBACK, TRIALS

    by = {t.name: t for t in TRIALS}
    assert all(0 <= t.playback < len(PLAYBACK) for t in TRIALS)
    m = by["Let go at rest: cusps"].motion
    assert math.degrees(m.turning()[1]) == pytest.approx(31.5, abs=0.05)
    assert m.nutation_hz() == pytest.approx(14.7, abs=0.05) and m.mean_precession() == pytest.approx(2.39, abs=0.005)
    assert 2 * math.pi / m.mean_precession() == pytest.approx(2.63, abs=0.005) and m.path() is Path.CUSPS
    m = by["A square top: just like a disc"].motion
    assert m.top.polar * 1e4 == pytest.approx(4.0, abs=5e-4) and m.top.radius / math.sqrt(3) == pytest.approx(
        0.04, abs=5e-6
    )
    assert math.degrees(m.turning()[1]) == pytest.approx(31.5, abs=0.05)
    assert m.nutation_hz() == pytest.approx(14.7, abs=0.05) and m.mean_precession() == pytest.approx(2.39, abs=0.005)
    m = by["Steady precession"].motion
    assert m.precession == pytest.approx(2.39, abs=0.005) and m.steady
    assert TOP.weight_moment / (TOP.polar * 1500 * RPM) == pytest.approx(2.34, abs=0.005)
    m = by["Fast precession"].motion
    assert m.precession == pytest.approx(109, abs=0.5) and m.precession / (2 * math.pi) == pytest.approx(17, abs=0.5)
    assert TOP.polar * 1500 * RPM / (TOP.transverse * math.cos(math.radians(30))) == pytest.approx(112, abs=0.5)
    m = by["Pushed back: loops"].motion
    assert math.degrees(m.turning()[1]) == pytest.approx(32.7, abs=0.05) and m.path() is Path.LOOPS
    assert m.mean_precession() == pytest.approx(2.39, abs=0.005)
    m = by["Pushed along: waves"].motion
    assert math.degrees(m.turning()[1]) == pytest.approx(30.7, abs=0.05) and m.path() is Path.WAVES
    assert Motion(TOP, 1500 * RPM, math.radians(30), 4.0).turning()[0] < math.radians(30)  # it rises
    m = by["Spun slowly: deep nods"].motion
    assert math.degrees(m.turning()[1]) == pytest.approx(43.9, abs=0.05)
    assert (m.nutation_hz(), m.mean_precession()) == pytest.approx((4.53, 6.70), abs=0.005)
    w3 = 600 * RPM
    depth = 2 * TOP.transverse * TOP.weight_moment * math.sin(math.radians(30)) / (TOP.polar * w3) ** 2
    assert math.degrees(depth) == pytest.approx(8.7, abs=0.05)
    assert TOP.polar * w3 / TOP.transverse / (2 * math.pi) == pytest.approx(6.2, abs=0.05)
    m = by["A sleeping top"].motion
    assert TOP.sleeping_spin / RPM == pytest.approx(467, abs=0.5)
    assert math.degrees(m.turning()[1]) == pytest.approx(2.26, abs=0.005)
    m = by["Too slow to sleep"].motion
    assert math.degrees(m.turning()[1]) == pytest.approx(62.2, abs=0.05) and 0.9 < m.nutation_period() < 1.1
    m = by["Lying flat"].motion
    assert m.precession == pytest.approx(2.34, abs=0.005) and 2 * math.pi / m.precession == pytest.approx(
        2.68, abs=5e-3
    )
    m = by["Spun the other way"].motion
    assert m.mean_precession() == pytest.approx(-2.39, abs=0.005)
    m = by["No gravity: nutation is the free wobble"].motion
    L = m.angular_momentum()
    assert math.degrees(math.acos(L[2] / np.linalg.norm(L))) == pytest.approx(28.5, abs=0.05)
    assert m.nutation_period() == pytest.approx(0.065, abs=5e-5)
    assert math.degrees(m.turning()[0]) == pytest.approx(27.0, abs=0.05)


RECTANGLE = Top(0.5, 0.1, 0.06, shape=Shape.RECTANGLE, breadth=0.02)  # spun about its smallest moment: stable


@pytest.mark.parametrize("gravity", [0.0, 9.81])
def test_the_asymmetric_split(gravity):
    """Against Euler's equations by solve_ivp (nothing assumed of the moments): second order, the energy's error
    falling as h², L·ẑ kept to round-off."""
    top = Top(0.5, 0.1, 0.06, gravity, Shape.RECTANGLE, 0.02)
    m = Motion(top, 600 * RPM, math.radians(30), 3.0)
    errors, energy = [], []
    for per in (200, 400):
        sim = TopSimulator(m, per)
        s = sim.advance(1.0)
        axis, omega = body_reference(m, s.t[-1:])
        errors.append(np.abs(axis[0] - s.axis[-1]).max())
        energy.append(np.abs(s.energy - m.total_energy).max())
        assert np.abs(s.momentum[:, 2] - m.vertical_momentum).max() < 1e-12
        assert body_rates(s.axis[-1:], s.mark[-1:], s.momentum[-1:], top)[0] == pytest.approx(omega[0], abs=1e-3)
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.1)
    assert energy[0] / energy[1] == pytest.approx(4.0, rel=0.15)


@pytest.mark.parametrize("height", [0.005, 0.015, 0.06])
def test_euler_poinsot(height):
    """Jacobi's cn, sn, dn against Euler's equations, in each ordering of the moments (z largest, middle,
    smallest: M² above and below 2EI_b)."""
    top = Top(0.5, 0.1, height, 0.0, Shape.RECTANGLE, 0.02)
    m = Motion(top, 600 * RPM, math.radians(30), 3.0)
    t = np.linspace(0.0, 1.0, 41)
    _, omega = body_reference(m, t, rtol=1e-12)
    free = FreeSpin.of(top, np.array([0.0, 3.0 * math.sin(m.tilt), m.spin]))
    assert free.omega(t) == pytest.approx(omega, abs=1e-6 * m.spin)


@pytest.mark.parametrize("height, flips", [(0.005, False), (0.015, True), (0.025, True), (0.035, False)])
def test_the_middle_axis_flips(height, flips):
    """The tennis-racket problem's setups: about the middle axis it flips over, about the others it doesn't."""
    from vib_tutorial.gui.gyroscope import GyroTrial, Launch

    t = GyroTrial(
        "",
        "",
        shape=Shape.RECTANGLE,
        radius=0.1,
        breadth=0.02,
        height=height,
        gravity=0.0,
        spin_rpm=600.0,
        tilt=2.0,
        launch=Launch.SET,
        precession=0.5,
    )
    s = TopSimulator(t.motion).advance(3.0)
    assert (np.degrees(np.arccos(s.axis[:, 2].min())) > 170) == flips


def test_the_section():
    """A symmetric top keeps ω₃: its section is one line; an asymmetric one's isn't."""
    from vib_tutorial.gui.gyroscope import _section

    for top, kept in ((Top(), True), (Top(0.5, 0.1, 0.03, shape=Shape.RECTANGLE, breadth=0.04), False)):
        s = TopSimulator(Motion(top, 600 * RPM, math.radians(30))).advance(5.0)
        ez = s.axis[:, 2]
        j = np.flatnonzero((ez[1:-1] < ez[:-2]) & (ez[1:-1] <= ez[2:])) + 1
        psi, spin = _section(s.axis[j], s.mark[j], s.momentum[j], top)
        assert j.size > 20 and np.all((psi >= 0) & (psi < 180))
        assert (np.ptp(spin) < 1e-6) == kept


def test_asymmetric_trials():
    from vib_tutorial.gui.gyroscope import TRIALS

    by = {t.name: t for t in TRIALS}

    def run(name, duration):
        m = by[name].motion
        a = TopSimulator(m)
        b = TopSimulator(Motion(m.top, m.spin, m.tilt + 1e-8, m.precession))
        s, other = a.advance(duration), b.advance(duration)
        return m, s, np.linalg.norm(s.axis - other.axis, axis=1), np.degrees(np.arccos(np.clip(s.axis[:, 2], -1, 1)))

    m, s, apart, theta = run("A rectangle: ω₃ isn't kept", 20.0)
    assert [round(v * 1e4, 2) for v in m.top.moments] == [5.17, 8.67, 4.83]
    spin = np.sum(s.momentum * s.axis, axis=1) / m.top.polar / RPM
    assert (spin.min(), spin.max()) == pytest.approx((596.7, 601.1), abs=0.05)
    assert (theta.min(), theta.max()) == pytest.approx((28.0, 40.7), abs=0.05)
    assert apart[-1] == pytest.approx(1e-7, rel=0.3)

    m, s, _, _ = run("No gravity: the tennis-racket flip", 3.0)
    i1, i2, i3 = (round(v * 1e4, 2) for v in m.top.moments)
    assert (i1, i3, i2) == (1.29, 4.33, 5.29)
    free = FreeSpin.of(m.top, np.array([0.0, m.precession * math.sin(m.tilt), m.spin]))
    assert free.period == pytest.approx(0.910, abs=5e-4) and 1 - free.k2 == pytest.approx(1.2e-7, rel=0.05)
    crossings = s.t[np.flatnonzero(np.diff(np.sign(s.axis[:, 2])))]
    assert np.diff(crossings) == pytest.approx(0.5 * free.period, abs=1e-3)
    assert 0.5 * free.period == pytest.approx(0.455, abs=5e-4)

    m, s, apart, _ = run("Gravity and the middle axis: chaos", 6.0)

    def first(limit):
        return s.t[np.flatnonzero(apart > limit)[0]]

    assert (first(1e-6), first(1e-2), first(1.0)) == pytest.approx((1.1, 3.0, 5.0), abs=0.05)
