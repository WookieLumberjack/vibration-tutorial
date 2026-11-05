"""The double pendulum: its whole equations against Lagrange's by finite differences, its linear matrices
against the energies' Hessians and the free-body rows, the modes, the swing and the twins."""

import math

import numpy as np
import pytest

from vib_tutorial.core.double_pendulum import (
    Angles,
    DoublePendulum,
    accelerations,
    divergence,
    matrices,
    mix,
    modes,
    newton,
    slipped,
    swing,
    transform,
)

P = DoublePendulum(upper=1.3, lower=0.7, upper_length=0.6, lower_length=0.45)


def lagrangian(p, q, w):
    kinetic, potential = p.energies(q[0], q[1], w[0], w[1])
    return float(kinetic - potential)


def test_the_whole_equations_are_lagranges():
    """d/dt ∂L/∂q̇ − ∂L/∂q = 0, by finite differences of L along the motion the accelerations give."""
    rng = np.random.default_rng(3)
    h = 1e-5
    for _ in range(5):
        q, w = rng.uniform(-2.5, 2.5, 2), rng.uniform(-3, 3, 2)
        a = np.array(accelerations(P, *q, *w))

        def dl_dw(q, w):
            return np.array([(lagrangian(P, q, w + h * e) - lagrangian(P, q, w - h * e)) / (2 * h) for e in np.eye(2)])

        # along the motion: q → q + w dt, w → w + a dt
        dt = 1e-5
        rate = (dl_dw(q + w * dt, w + a * dt) - dl_dw(q - w * dt, w - a * dt)) / (2 * dt)
        dl_dq = np.array([(lagrangian(P, q + h * e, w) - lagrangian(P, q - h * e, w)) / (2 * h) for e in np.eye(2)])
        np.testing.assert_allclose(rate - dl_dq, 0.0, atol=1e-5)


@pytest.mark.parametrize("angles", list(Angles))
def test_the_matrices_are_the_energies_hessians_at_rest(angles):
    T = transform(angles)
    h = 1e-4

    def hessian(f):
        H = np.empty((2, 2))
        for i in range(2):
            for j in range(2):
                ei, ej = np.eye(2)[i] * h, np.eye(2)[j] * h
                H[i, j] = (f(ei + ej) - f(ei - ej) - f(ej - ei) + f(-ei - ej)) / (4 * h * h)
        return H

    m = matrices(P, angles)
    kinetic = hessian(lambda v: float(P.energies(0.0, 0.0, *(T @ v))[0]))
    potential = hessian(lambda q: float(P.energies(*(T @ q), 0.0, 0.0)[1]))
    np.testing.assert_allclose(kinetic, m.M, rtol=1e-6)
    np.testing.assert_allclose(potential, m.K, rtol=1e-5)


@pytest.mark.parametrize("angles", list(Angles))
def test_the_free_body_rows_mixed_are_lagranges(angles):
    rows, m, R = newton(P, angles), matrices(P, angles), mix(P, angles)
    np.testing.assert_allclose(R @ rows.M, m.M, atol=1e-12)
    np.testing.assert_allclose(R @ rows.K, m.K, atol=1e-12)
    assert not np.allclose(rows.M, rows.M.T)


def test_the_modes_in_either_angles_and_in_closed_form():
    for a, b in zip(modes(P), modes(P, Angles.RELATIVE)):
        assert a.frequency == pytest.approx(b.frequency, rel=1e-12)
        assert a.ratio == pytest.approx(b.ratio, rel=1e-10)
        assert b.shape[1] == pytest.approx(b.ratio - 1.0, rel=1e-10)  # the bend φ = θ₂ − θ₁
    L, g = 0.8, 9.81
    slow, fast = modes(DoublePendulum(2.0, 2.0, L, L))
    w = np.sqrt((2 - np.sqrt(2)) * g / L), np.sqrt((2 + np.sqrt(2)) * g / L)
    assert (slow.frequency, fast.frequency) == pytest.approx((w[0] / (2 * np.pi), w[1] / (2 * np.pi)))
    assert (slow.ratio, fast.ratio) == pytest.approx((np.sqrt(2), -np.sqrt(2)))
    mu = 0.3  # equal rods: θ₂/θ₁ = √((1 + μ)/μ) in the slower mode
    assert modes(DoublePendulum(1.0, mu, L, L))[0].ratio == pytest.approx(np.sqrt((1 + mu) / mu))


def test_the_slip_makes_each_rod_a_simple_pendulum():
    p = DoublePendulum(1.0, 1.0, 0.5, 0.5)
    s = slipped(p)
    f = np.sort(np.sqrt(np.linalg.eigvals(np.linalg.solve(s.M, s.K)).real)) / (2 * np.pi)
    assert f == pytest.approx(p.separate_frequencies())


def test_small_swings_follow_the_linear_model_and_the_energy_holds():
    small = swing(P, (0.01, 0.0), duration=10.0)
    assert np.abs(small.whole[:, :2] - small.linear[:, :2]).max() < 2e-3 * 0.01 * 10
    m = matrices(P)
    w2, phi = np.linalg.eig(np.linalg.solve(m.M, m.K))
    a = np.linalg.solve(phi, [0.01, 0.0])
    exact = (phi * a) @ np.cos(np.sqrt(w2.real)[:, None] * small.t)
    np.testing.assert_allclose(small.linear[:, :2], exact.T, atol=1e-12)
    for start in ((0.7, 1.0), (2.1, 2.1)):
        assert swing(P, start, duration=10.0).drift < 1e-8


def test_high_releases_are_chaotic_and_low_ones_not():
    p = DoublePendulum()
    high = divergence(p, (math.radians(120), math.radians(120)), 15.0)
    low = divergence(p, (math.radians(10), math.radians(14.1)), 15.0)
    assert 0.2 < high.doubling_time() < 1.0 and high.apart[-1] > 1.0
    assert math.isinf(low.doubling_time()) and low.apart.max() < 1e-5
