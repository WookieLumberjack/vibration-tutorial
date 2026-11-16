"""`core/ode.py`: the shared RK4 (#316, the copies in one place)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vib_tutorial.core.ode import rk4, rk4_step, rk4_step_floats


def _oscillator(z):
    x, v = z
    return (v, -x)


def test_a_step_is_fourth_order():
    errors = []
    for h in (0.2, 0.1):
        y = np.array([1.0, 0.0])
        for _ in range(round(1.0 / h)):
            y = rk4_step(lambda z: np.array(_oscillator(z)), y, h)
        errors.append(abs(y[0] - math.cos(1.0)))
    assert errors[0] / errors[1] == pytest.approx(16.0, rel=0.1)


def test_floats_arrays_and_many_systems_agree():
    one = rk4(_oscillator, np.array([1.0, 0.0]), 0.05, 40, keep=4)
    assert one.shape == (11, 2) and one[-1, 0] == pytest.approx(math.cos(2.0), abs=1e-6)
    side_by_side = rk4(lambda z: np.array(_oscillator(z)), np.array([[1.0, 2.0], [0.0, 0.0]]), 0.05, 40, keep=4)
    assert side_by_side.shape == (11, 2, 2)
    assert np.array_equal(side_by_side[:, :, 0], one) and np.allclose(
        side_by_side[:, :, 1], 2 * one, rtol=0, atol=1e-15
    )
    y = (1.0, 0.0)
    assert rk4_step_floats(_oscillator, y, 0.05, k1=_oscillator(y)) == rk4_step_floats(_oscillator, y, 0.05)
    assert tuple(rk4_step(lambda z: np.array(_oscillator(z)), np.array(y), 0.05)) == rk4_step_floats(
        _oscillator, y, 0.05
    )
