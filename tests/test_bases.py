"""Coordinates & bases' core (`core/bases.py`): the matrices in any basis, the frequencies they keep, the modes that
diagonalize them, Rayleigh's quotient and its bounds, and the exact stepper."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vib_tutorial.core.bases import (
    TWO_PI,
    Steps,
    TwoMass,
    angle,
    direction,
    frequencies,
    rayleigh,
    rayleigh_hz,
    transformed,
    unit,
)

ASYMMETRIC = TwoMass(m1=1.0, m2=2.0, k1=400.0, k2=150.0, k3=250.0, c=0.0)


def test_the_matrices_of_two_masses_between_two_walls():
    M, C, K = TwoMass(2.0, 3.0, 100.0, 50.0, 70.0, 0.5).matrices()
    assert np.array_equal(M, np.diag([2.0, 3.0]))
    assert np.array_equal(K, [[150.0, -50.0], [-50.0, 120.0]])
    assert np.array_equal(C, [[1.0, -0.5], [-0.5, 1.0]])


def test_the_symmetric_pairs_modes_are_in_phase_and_opposed():
    hz, shapes = TwoMass().modes()  # 1 kg, 400 N/m to each wall, 100 N/m between
    assert hz == pytest.approx([math.sqrt(400.0) / TWO_PI, math.sqrt(600.0) / TWO_PI])
    assert shapes[:, 0] == pytest.approx([math.sqrt(0.5)] * 2)
    assert shapes[:, 1] == pytest.approx([math.sqrt(0.5), -math.sqrt(0.5)])


@pytest.mark.parametrize("seed", range(5))
def test_any_basis_keeps_the_frequencies(seed):
    rng = np.random.default_rng(seed)
    M, _, K = ASYMMETRIC.matrices()
    V = rng.normal(size=(2, 2))
    hz, _ = ASYMMETRIC.modes()
    assert frequencies(transformed(V, M), transformed(V, K)) == pytest.approx(hz)


def test_the_modes_make_both_matrices_diagonal_but_are_not_perpendicular_with_unequal_masses():
    M, _, K = ASYMMETRIC.matrices()
    hz, P = ASYMMETRIC.modes()
    Mt, Kt = transformed(P, M), transformed(P, K)
    assert abs(Mt[0, 1]) < 1e-12 and abs(Kt[0, 1]) < 1e-9
    assert Kt[0, 0] / Mt[0, 0] == pytest.approx((TWO_PI * hz[0]) ** 2)
    assert abs(float(P[:, 0] @ P[:, 1])) > 0.05  # orthogonal through M, not in the plane


def test_rayleighs_quotient_lies_between_the_frequencies_and_touches_them_along_the_modes():
    M, _, K = ASYMMETRIC.matrices()
    hz, P = ASYMMETRIC.modes()
    f = np.array([rayleigh_hz(direction(a), M, K) for a in np.linspace(0, 180, 721)])
    assert f.min() >= hz[0] * (1 - 1e-9) and f.max() <= hz[1] * (1 + 1e-9)
    assert rayleigh_hz(P[:, 0], M, K) == pytest.approx(hz[0])
    assert rayleigh_hz(P[:, 1], M, K) == pytest.approx(hz[1])
    # Second order: a shape 1 % off the mode errs by about 0.01² in ω².
    off = P[:, 0] + 0.01 * P[:, 1]
    assert rayleigh(off, M, K) / (TWO_PI * hz[0]) ** 2 - 1 < 1e-3


def test_a_lines_angle_and_direction():
    assert angle(np.array([1.0, 1.0])) == pytest.approx(45.0)
    assert angle(np.array([-1.0, -1.0])) == pytest.approx(45.0)  # a line, not an arrow
    assert angle(np.array([1.0, -1.0])) == pytest.approx(135.0)
    assert direction(135.0) == pytest.approx([-math.sqrt(0.5), math.sqrt(0.5)])
    assert unit(np.array([-3.0, 4.0])) == pytest.approx([0.6, -0.8])
    with pytest.raises(ValueError):
        unit(np.zeros(2))


def test_the_stepper_matches_the_closed_form_and_carries_a_part_step():
    s = TwoMass()
    steps = Steps(*s.matrices(), z=np.array([0.05, 0.0, 0.0, 0.0]))
    ts, zs, _ = steps.advance(1.0)
    # From (0.05, 0) at rest: half in each mode, x₁ = 0.025 (cos ω₁t + cos ω₂t).
    w1, w2 = (TWO_PI * f for f in s.modes()[0])
    assert zs[:, 0] == pytest.approx(0.025 * (np.cos(w1 * ts) + np.cos(w2 * ts)), abs=1e-12)
    assert zs[:, 1] == pytest.approx(0.025 * (np.cos(w1 * ts) - np.cos(w2 * ts)), abs=1e-12)
    h = steps.step_size()
    t = steps.t
    steps.advance(0.6 * h)  # under a step: nothing yet ...
    assert steps.t == t
    steps.advance(0.6 * h)  # ... then the step it owed
    assert steps.t == pytest.approx(t + h)


def test_the_dampers_loss_is_exact():
    s = TwoMass(c=0.8)
    M, C, K = s.matrices()
    z0 = np.array([0.04, -0.01, 0.0, 0.2])
    steps = Steps(M, C, K, z=z0)
    _, zs, lost = steps.advance(3.0)
    stored = lambda z: 0.5 * z[2:] @ M @ z[2:] + 0.5 * z[:2] @ K @ z[:2]  # noqa: E731
    assert stored(z0) - stored(zs[-1]) == pytest.approx(lost[-1], rel=1e-10)
    assert np.all(np.diff(lost) >= -1e-15)


def test_a_rail_is_a_one_mass_oscillator_at_rayleighs_frequency():
    s = ASYMMETRIC
    M, C, K = s.matrices()
    u = direction(30.0)[:, None]
    steps = Steps(transformed(u, M), transformed(u, C), transformed(u, K), z=np.array([0.03, 0.0]))
    ts, zs, _ = steps.advance(2.0)
    w = math.sqrt(rayleigh(u[:, 0], M, K))
    assert zs[:, 0] == pytest.approx(0.03 * np.cos(w * ts), abs=1e-12)
