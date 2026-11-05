"""The bounce-and-pitch car: Lagrange's matrices against the energies, the free-body diagram's rows, the
coordinates, the modes and the ride."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from vib_tutorial.core.bounce_pitch import (
    Bump,
    Car,
    Coordinates,
    energies,
    matrices,
    modes,
    newton,
    ride,
    slipped,
    transform,
)

CAR = Car(mass=1100.0, gyration=1.25, front=1.1, rear=1.6, k_front=35e3, k_rear=46e3, c_front=2100.0, c_rear=2700.0)
CASES = [(Coordinates.CENTRE, 0.0), (Coordinates.POINT, 0.7), (Coordinates.POINT, -0.4), (Coordinates.ENDS, 0.0)]


def hessian(f, n: int = 2, h: float = 1e-4) -> np.ndarray:
    """A function's Hessian at 0 by central differences."""
    H = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            ei, ej = np.eye(n)[i] * h, np.eye(n)[j] * h
            H[i, j] = (f(ei + ej) - f(ei - ej) - f(ej - ei) + f(-ei - ej)) / (4 * h * h)
    return H


@pytest.mark.parametrize(("coords", "e"), CASES)
def test_the_matrices_are_the_energies_hessians(coords, e):
    m = matrices(CAR, coords, e)
    zero = np.zeros(2)
    np.testing.assert_allclose(hessian(lambda v: energies(CAR, coords, zero, v, e)[0]), m.M, rtol=1e-6)
    np.testing.assert_allclose(hessian(lambda q: energies(CAR, coords, q, zero, e)[1]), m.K, rtol=1e-6)
    np.testing.assert_allclose(hessian(lambda v: energies(CAR, coords, zero, v, e)[2]), m.C, rtol=1e-6)
    for A in (m.M, m.C, m.K):
        np.testing.assert_allclose(A, A.T)


@pytest.mark.parametrize(("coords", "e"), CASES)
def test_the_free_body_rows_mixed_by_t_transposed_are_lagranges(coords, e):
    T = transform(CAR, coords, e)
    rows, m = newton(CAR, coords, e), matrices(CAR, coords, e)
    for A, B in ((rows.M, m.M), (rows.C, m.C), (rows.K, m.K)):
        np.testing.assert_allclose(T.T @ A, B, rtol=1e-12, atol=1e-9)


def test_the_closed_forms():
    a, b, L = CAR.front, CAR.rear, CAR.wheelbase
    m, J = CAR.mass, CAR.inertia
    centre = matrices(CAR, Coordinates.CENTRE)
    kf, kr = CAR.k_front, CAR.k_rear
    np.testing.assert_allclose(centre.K, [[kf + kr, kf * a - kr * b], [kf * a - kr * b, kf * a * a + kr * b * b]])
    e = 0.7
    point = matrices(CAR, Coordinates.POINT, e)
    np.testing.assert_allclose(point.M, [[m, -m * e], [-m * e, J + m * e * e]])
    np.testing.assert_allclose(point.K[0, 1], kf * (a - e) - kr * (b + e))
    ends = matrices(CAR, Coordinates.ENDS)
    np.testing.assert_allclose(ends.K, np.diag([kf, kr]), atol=1e-9)
    np.testing.assert_allclose(ends.M[0, 1], (m * a * b - J) / L**2)
    # at the elastic centre K is diagonal; at a dynamic index of 1 the ends' M is
    assert abs(matrices(CAR, Coordinates.POINT, CAR.elastic_centre()).K[0, 1]) < 1e-9
    balanced = Car(gyration=math.sqrt(1.2 * 1.5))
    assert balanced.dynamic_index == pytest.approx(1.0)
    assert abs(matrices(balanced, Coordinates.ENDS).M[0, 1]) < 1e-9
    f = [mode.frequency for mode in modes(balanced, Coordinates.ENDS)]
    assert f == pytest.approx(sorted(balanced.end_frequencies()))


@pytest.mark.parametrize(("coords", "e"), CASES[1:])
def test_every_set_of_coordinates_has_the_same_modes(coords, e):
    reference = modes(CAR)
    for mode, ref in zip(modes(CAR, coords, e), reference):
        assert mode.frequency == pytest.approx(ref.frequency, rel=1e-10)
        np.testing.assert_allclose(mode.centre, ref.centre, rtol=1e-8, atol=1e-12)
        assert mode.node == pytest.approx(ref.node, rel=1e-8)


def test_the_slip_changes_the_frequencies_except_at_g():
    def frequencies(m):
        return np.sort(np.sqrt(np.linalg.eigvals(np.linalg.solve(m.M, m.K)).real)) / (2 * np.pi)

    right = [mode.frequency for mode in modes(CAR)]
    assert frequencies(slipped(CAR, 0.0)) == pytest.approx(right)
    assert not np.allclose(frequencies(slipped(CAR, 1.1)), right, rtol=1e-2)


def test_nodes_lie_where_the_body_stands_still():
    for mode in modes(CAR):
        xg, th = mode.centre
        assert xg + mode.node * th == pytest.approx(0.0, abs=1e-12)
    bounce, pitch = modes(Car(k_rear=40e3 * 1.2 / 1.5))
    assert math.isinf(bounce.node) and pitch.node == pytest.approx(0.0, abs=1e-9)


def test_the_ride_matches_solve_ivp():
    bump = Bump(speed=8.0, height=0.06, length=2.5)
    r = ride(CAR, bump)
    m = matrices(CAR, Coordinates.CENTRE)
    E = CAR.deflections()
    kd, cd = np.diag([CAR.k_front, CAR.k_rear]), np.diag([CAR.c_front, CAR.c_rear])
    Mi = np.linalg.inv(m.M)

    def road(t):
        """The road's height and its rate under each wheel."""
        s = bump.speed * (t - r.lead) - np.array([0.0, CAR.wheelbase])
        inside = (s >= 0) & (s <= bump.length)
        rate = np.pi * bump.height * bump.speed / bump.length * np.sin(2 * np.pi * s / bump.length)
        return bump.profile(s), np.where(inside, rate, 0.0)

    def rhs(t, z):
        y, ydot = road(t)
        q, v = z[:2], z[2:]
        return np.concatenate([v, Mi @ (E.T @ (kd @ y + cd @ ydot) - m.C @ v - m.K @ q)])

    sol = solve_ivp(rhs, (0, r.t[-1]), np.zeros(4), t_eval=r.t, rtol=1e-10, atol=1e-13, max_step=0.002)
    scale = np.abs(sol.y[:2]).max(axis=1)
    np.testing.assert_allclose(r.centre / scale, sol.y[:2].T / scale, atol=2e-3)
    np.testing.assert_allclose(r.ends, r.centre @ E.T)
