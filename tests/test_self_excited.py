"""The Self-excited vibration page's models (core/self_excited.py)."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from vib_tutorial.core.self_excited import (
    Flutter,
    Galloping,
    VanDerPol,
    Viv,
    galloping_branches,
    rk4,
    settled_amplitude,
)


def test_rk4_matches_solve_ivp_on_van_der_pol():
    m = VanDerPol(eps=1.0)
    t, x, _ = m.simulate(5, 400)
    f = m.rhs(m.eps)
    ref = solve_ivp(lambda _, z: f(z), (0, t[-1]), [m.start, 0.0], t_eval=t, rtol=1e-11, atol=1e-14).y[0]
    assert np.max(np.abs(x - ref)) < 1e-7


def test_rk4_steps_many_systems_at_once():
    m = VanDerPol()
    eps = np.array([0.1, 0.5])
    z = rk4(m.rhs(eps), np.array([[1e-3, 1e-3], [0.0, 0.0]]), 0.01, 200)
    for j, e in enumerate(eps):
        alone = rk4(m.rhs(e), np.array([1e-3, 0.0]), 0.01, 200)
        assert z[:, :, j] == pytest.approx(alone)


@pytest.mark.parametrize(
    ("model", "values", "z0"),
    [
        (VanDerPol(), [0.1, 2.0], [1e-3, 0.0]),
        (Galloping(), [5.0, 9.0], [1e-3, 0.0]),
        (Viv(), [4.0, 5.5], [0.0, 0.0, 2.0, 0.0]),
    ],
)
def test_one_system_stepped_as_floats_is_the_same_as_in_a_sweep(model, values, z0):
    """A single run steps Python floats, a sweep arrays: the same arithmetic in the same order, to the last bit."""
    swept = rk4(model.rhs(np.array(values)), np.array([[c, c] for c in z0]), 1e-3, 500, keep=5)
    for j, value in enumerate(values):
        alone = rk4(model.rhs(value), np.array(z0), 1e-3, 500, keep=5)
        assert np.array_equal(alone, swept[:, :, j])


def test_van_der_pol_limit_cycle_is_2_xr_root_eps():
    for eps in (0.05, 0.2):
        m = VanDerPol(eps=eps)
        assert m.sweep(np.array([eps]), periods=200)[0] == pytest.approx(float(m.theory_amplitude(eps)), rel=0.01)
    assert VanDerPol().sweep(np.array([-0.2]))[0] < 1e-6


def test_galloping_onset_is_where_the_net_damping_changes_sign():
    g = Galloping()
    small = 1e-4 * g.side
    for U, sign in ((0.99 * g.onset, -1), (1.01 * g.onset, 1)):
        assert np.sign(g.net_power(np.array([small]), U)[0]) == sign


def test_galloping_branches_are_where_the_motion_settles():
    g = Galloping()
    for U, start in ((6.0, 0.001), (6.0, 0.12), (12.0, 0.001)):
        _t, y, _ = Galloping(speed=U, start=start).simulate(600)
        amp = settled_amplitude(y)
        stable = [b.amplitude[0] for b in galloping_branches(g, np.array([U])) if b.stable]
        assert min(abs(a - amp) for a in stable) < 0.01 * amp


def test_net_power_averages_the_force():
    g = Galloping()
    a, U = 0.05, 6.0
    theta = np.linspace(0, 2 * np.pi, 20001)[:-1]
    v = -a * g.omega * np.sin(theta)
    from vib_tutorial.core.self_excited import force_coefficient

    F = 0.5 * g.rho * U * U * g.side * force_coefficient(v / U)
    wind = np.mean(F * v) * 2 * np.pi / g.omega
    damper = np.pi * 2 * g.mass * g.zeta * g.omega**2 * a * a
    assert g.net_power(np.array([a]), U)[0] == pytest.approx(wind - damper, rel=1e-9)


def test_viv_locks_in_near_the_strouhal_speed():
    v = Viv()
    amp, freq = v.sweep(np.array([3.0, 5.5, 9.0]))
    assert amp[1] > 5 * max(amp[0], amp[2])
    assert freq[1] == pytest.approx(1.0, abs=0.03) and freq[2] > 1.5


def test_flutter_poles_without_wind_are_the_structures():
    f = Flutter(x_cg=0.0, zeta=0.0)
    lam = f.poles(0.0)
    assert sorted(abs(lam.imag[lam.imag > 0]) / (2 * math.pi)) == pytest.approx([3.0, 6.0])
    assert f.divergence_speed() == pytest.approx(Flutter().divergence_speed())
    Uf = Flutter().flutter_speed()
    assert np.max(Flutter().poles(0.999 * Uf).real) < 0 < np.max(Flutter().poles(1.001 * Uf).real)


def test_steady_lift_flutters_by_coalescence_and_quasi_steady_without():
    """Steady lift is a stiffness: undamped, the two frequencies merge at the flutter speed into a pair of equal and
    opposite damping. Quasi-steady lift's ḣ/U is a damping: the pitch-like mode goes alone, the frequencies apart."""
    undamped = Flutter(zeta=0.0)
    Uf = undamped.flutter_speed()
    below, above = undamped.poles(0.999 * Uf), undamped.poles(1.001 * Uf)
    assert np.allclose(below.real, 0.0, atol=1e-9 * np.abs(below).max())  # no damping anywhere below it
    up = above[above.imag > 0]
    assert up.imag == pytest.approx([up.imag[0]] * 2, rel=1e-9) and up.real.sum() == pytest.approx(0.0, abs=1e-9)
    q = Flutter(quasi_steady=True)
    f, z = q.modes(np.array([0.999 * q.flutter_speed()]))
    assert f[0, 1] - f[0, 0] > 1.0  # Hz apart
    assert z[0, 1] < 1e-3 < 0.05 < z[0, 0]  # the pitch at zero, the plunge damped by the flow


def test_divergence_is_not_flutter():
    """`flutter_speed` counts oscillating poles only, and only below divergence: with the center of mass on the axis
    the steady section only diverges (a real pole crosses at `divergence_speed`)."""
    f = Flutter(x_cg=0.0)
    assert math.isnan(f.flutter_speed())
    Ud = f.divergence_speed()
    lam = f.poles(1.001 * Ud)
    assert np.any((lam.real > 0) & (np.abs(lam.imag) < 1e-6))


def test_flutter_mode_solves_the_sections_equations():
    """λ and (h, θ) satisfy (λ²M + λC + K)(h, θ) = 0, θ real and the larger of |h|/b, |θ| one; the least stable
    mode crosses into the right half plane at the flutter speed."""
    m = Flutter(quasi_steady=True)
    Uf = m.flutter_speed()
    for U in (0.5 * Uf, 1.1 * Uf):
        lam, h, theta = m.mode(U)
        M, C, K = m.matrices(U)
        v = np.array([h, theta])
        assert np.linalg.norm((lam * lam * M + lam * C + K) @ v) < 1e-9 * np.linalg.norm(K) * np.linalg.norm(v)
        assert theta.imag == 0 and max(abs(h) / m.semichord, abs(theta)) == pytest.approx(1.0)
        assert lam.imag > 0 and lam.real == pytest.approx(max(p.real for p in m.poles(U) if p.imag > 1e-9))
        assert (lam.real > 0) == (U > Uf)


def test_the_motion_tab_plays_the_body_and_the_slider_moves_the_poles(main_window):
    """The Motion tab plays the run the center shows (the galloping square at y / D), or flutter's least stable
    mode; dragging the slider moves the poles at once and sets the wind speed only when let go."""
    from PySide6 import QtWidgets

    from vib_tutorial.gui.self_excited import TAB_MOTION, TRIALS, Kind
    from vib_tutorial.gui.self_excited_view import Body

    w = main_window
    page = w.self_excited_page
    w.pages.setCurrentWidget(page)
    QtWidgets.QApplication.processEvents()
    page.load_trial(next(i for i, t in enumerate(TRIALS) if t.name == "Galloping at 12 m/s"))
    assert page.tabs.currentIndex() == TAB_MOTION and page.playback.running  # a trial starts it
    assert page.section.body is Body.SQUARE and page.section.unstable
    t, y, _ = page.motion
    page.playback.seek(float(t[-1]) * 0.8)
    k = int(np.searchsorted(t, page.playback.t))
    assert page.section.state[0] == pytest.approx(y[k])
    assert np.max(np.abs(y)) > 2.0  # 2.7 D, drawn to scale

    speed = page.v("g_speed")
    page.drive.setSliderDown(True)
    page.drive.setValue(10)  # near no wind: the pole moves left of Re λ = 0 ...
    assert page.pole_now.getData()[0][0] < 0 and page.v("g_speed") == speed  # ... but the model isn't rerun
    chosen = page._drive_value()
    page.drive.setSliderDown(False)  # let go: Qt emits sliderReleased
    assert page.v("g_speed") == pytest.approx(chosen, abs=0.01) and page.v("g_speed") < speed
    assert not page.section.unstable

    page.kind.setCurrentIndex(list(Kind).index(Kind.FLUTTER))
    assert page.section.body is Body.WING
    lam, _, _ = page.flutter().mode(page.v("f_speed"))
    assert page.section.unstable == (lam.real > 0)
    _, y, th = page.motion
    assert np.max(np.abs(th)) > 0 and page.time_cursor.isVisible() is False  # flutter's plots are against U
