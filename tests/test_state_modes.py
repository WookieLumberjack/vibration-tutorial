"""State-space modes' core (`core/state_modes.py`): the three paired bases of the state, A's modes and their blocks,
the spiral's exact decay, the proportional case against the undamped modes, and an overdamped pair."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.linalg

from vib_tutorial.core.bases import Steps, TwoMass, state_matrix
from vib_tutorial.core.state_modes import (
    BASES,
    basis,
    coordinates,
    coupling,
    fitted_eigenvalue,
    in_basis,
    mass_basis,
    own_frequency,
    passed,
    ratio,
    shape_basis,
    state_basis,
    state_modes,
    undamped_basis,
    undamped_modes,
)

UNEQUAL = TwoMass(m2=2.0, c=1.0)  # the page's default: dampers the undamped modes can't uncouple
SYMMETRIC = TwoMass(c=1.0)  # equal masses and walls: C is diagonal in the modes too
HEAVY = TwoMass(m2=2.0, c=20.0)  # mode 2 overdamped


@pytest.mark.parametrize("s", [UNEQUAL, SYMMETRIC, HEAVY, TwoMass(m1=3.0, k1=900.0, k2=50.0, k3=100.0, c=4.0)])
def test_the_state_basis_uncouples_any_damping(s):
    M, C, K = s.matrices()
    A = state_matrix(M, C, K)
    Ab = in_basis(state_basis(M, C, K), A)
    assert coupling(Ab) < 1e-9
    for m, r in zip(state_modes(M, C, K), range(2)):
        block = Ab[2 * r : 2 * r + 2, 2 * r : 2 * r + 2]
        if m.swings:
            assert block == pytest.approx(np.array([[m.sigma, m.omega], [-m.omega, m.sigma]]), abs=1e-9)
        else:
            assert block == pytest.approx(np.diag([e.real for e in m.eigenvalues]), abs=1e-9)


def test_the_modes_are_as_eigenvalues_of_a_and_their_columns_its_eigenvectors():
    M, C, K = UNEQUAL.matrices()
    A = state_matrix(M, C, K)
    lam = np.linalg.eigvals(A)
    ms = state_modes(M, C, K)
    got = sorted([e for m in ms for e in m.eigenvalues], key=lambda e: (abs(e), e.imag))
    assert np.array(got) == pytest.approx(np.array(sorted(lam, key=lambda e: (abs(e), e.imag))))
    for m in ms:
        y = m.columns[:, 0] + 1j * m.columns[:, 1]
        assert A @ y == pytest.approx(m.eigenvalues[0] * y)
        assert y[:2] == pytest.approx(m.shape) and np.linalg.norm(m.shape) == pytest.approx(1.0)
        assert (m.shape @ m.shape).imag == pytest.approx(0.0, abs=1e-12)  # turned as real as it goes


def test_the_defaults_modes():
    """The figures the page's notes quote at its defaults (m₂ = 2 kg, c = 1 N·s/m)."""
    one, two = state_modes(*UNEQUAL.matrices())
    assert one.omega / (2 * math.pi) == pytest.approx(2.421, abs=5e-4)
    assert two.omega / (2 * math.pi) == pytest.approx(3.619, abs=5e-4)
    assert one.sigma == pytest.approx(-0.358, abs=5e-4) and two.sigma == pytest.approx(-1.142, abs=5e-4)
    assert one.zeta == pytest.approx(0.0235, abs=5e-5) and two.zeta == pytest.approx(0.0502, abs=5e-5)
    assert one.phase == pytest.approx(-4.51, abs=0.005) and two.phase == pytest.approx(-173.29, abs=0.005)


def test_the_undamped_basis_leaves_only_the_dampers_coupling():
    M, C, K = UNEQUAL.matrices()
    A = state_matrix(M, C, K)
    Ab = in_basis(undamped_basis(M, K), A)
    omega, V = undamped_modes(M, K)
    for r in range(2):  # each pair a harmonic oscillator, (q, q̇/ω), less its own damping
        assert Ab[2 * r, 2 * r + 1] == pytest.approx(omega[r])
        assert Ab[2 * r + 1, 2 * r] == pytest.approx(-omega[r])
        assert Ab[2 * r, 2 * r] == pytest.approx(0.0, abs=1e-9)
    off = Ab[0:2, 2:4]
    assert np.count_nonzero(np.abs(off) > 1e-9) == 1 and abs(off[1, 1]) > 0.1  # the damping's term alone
    Ct, Mt = V.T @ C @ V, V.T @ M @ V
    assert off[1, 1] == pytest.approx(-Ct[0, 1] / Mt[0, 0] * omega[1] / omega[0])
    assert coupling(in_basis(undamped_basis(*SYMMETRIC.matrices()[::2]), state_matrix(*SYMMETRIC.matrices()))) < 1e-9


def test_with_proportional_damping_the_state_modes_are_the_undamped_ones():
    M, C, K = SYMMETRIC.matrices()
    _, V = undamped_modes(M, K)
    for r, m in enumerate(state_modes(M, C, K)):
        assert m.shape.imag == pytest.approx(np.zeros(2), abs=1e-12)
        assert m.shape.real == pytest.approx(V[:, r])
        assert m.phase in (pytest.approx(0.0, abs=1e-9), pytest.approx(180.0, abs=1e-9))


def test_the_mass_basis_is_each_mass_x_and_scaled_velocity():
    M, C, K = UNEQUAL.matrices()
    B = mass_basis(M, K)
    w = coordinates(B, np.array([0.01, 0.02, 0.3, 0.4]))
    assert w == pytest.approx([0.01, 0.3 / own_frequency(M, K, 0), 0.02, 0.4 / own_frequency(M, K, 1)])
    assert own_frequency(M, K, 1) == pytest.approx(math.sqrt(500.0 / 2.0))
    assert coupling(in_basis(B, state_matrix(M, C, K))) > 0.1  # the springs and dampers couple the masses


def test_a_state_mode_alone_spirals_in_exactly_at_its_rate():
    M, C, K = UNEQUAL.matrices()
    B = state_basis(M, C, K)
    two = state_modes(M, C, K)[1]
    steps = Steps(M, C, K, two.columns[:, 0] * 0.03)
    ts, zs, _ = steps.advance(2.0)
    w = coordinates(B, zs)
    assert np.abs(w[:, :2]).max() < 1e-9  # mode 1's pair never moves
    size = np.hypot(w[:, 2], w[:, 3])
    assert size == pytest.approx(0.03 * np.exp(two.sigma * ts), rel=1e-6)
    turned = np.unwrap(np.arctan2(w[:, 3], w[:, 2]))
    assert turned == pytest.approx(-two.omega * ts, abs=1e-6)  # clockwise, at ω


def test_an_overdamped_mode_is_two_real_decays():
    one, two = state_modes(*HEAVY.matrices())
    assert one.swings and not two.swings
    a, b = (e.real for e in two.eigenvalues)
    assert two.eigenvalues[0].imag == 0 and b < a < 0  # the slower first
    assert two.zeta > 1 and two.natural == pytest.approx(math.sqrt(a * b))
    assert two.shape.imag == pytest.approx(np.zeros(2))


def test_every_basis_by_name_and_an_unknown_one():
    M, C, K = UNEQUAL.matrices()
    for name in BASES:
        assert basis(name, M, C, K).shape == (4, 4)
    with pytest.raises(ValueError):
        basis("physical", M, C, K)


def test_the_matrix_in_a_basis_keeps_the_eigenvalues():
    M, C, K = UNEQUAL.matrices()
    A = state_matrix(M, C, K)
    want = np.sort_complex(np.linalg.eigvals(A))
    for name in BASES:
        got = np.sort_complex(scipy.linalg.eigvals(in_basis(basis(name, M, C, K), A)))
        assert got == pytest.approx(want)


@pytest.mark.parametrize("s", [UNEQUAL, SYMMETRIC, TwoMass(m1=3.0, k1=900.0, k2=50.0, k3=100.0, c=4.0)])
def test_a_guessed_shape_uncouples_only_its_own_columns_and_only_at_As_shape(s):
    M, C, K = s.matrices()
    A = state_matrix(M, C, K)
    modes = state_modes(M, C, K)
    true = tuple(ratio(m) for m in modes)
    for m in modes:  # at A's shape the fitted λ is A's, and the pair is A's own, exactly
        assert fitted_eigenvalue(M, C, K, m.shape) == pytest.approx(m.eigenvalues[0])
    assert shape_basis(M, C, K, true) == pytest.approx(state_basis(M, C, K), abs=1e-12)
    _, V = undamped_modes(M, K)
    undamped = (V[1, 0] / V[0, 0], V[1, 1] / V[0, 1])
    proportional = s is SYMMETRIC
    for r in range(2):  # mode r's right shape clears its own columns whatever the other's is
        guess = list(undamped)
        guess[r] = true[r]
        Ab = in_basis(shape_basis(M, C, K, tuple(guess)), A)
        assert passed(Ab, r) < 1e-12
        assert (passed(Ab, 1 - r) < 1e-9) == proportional  # the other, still the undamped mode's, passes motion on
        off = list(true)
        off[r] = true[r] * complex(math.cos(0.05), math.sin(0.05))  # turned 2.9°
        assert passed(in_basis(shape_basis(M, C, K, tuple(off)), A), r) > 1e-3


def test_a_shape_too_damped_to_swing_still_makes_two_columns():
    M, C, K = HEAVY.matrices()
    lam = fitted_eigenvalue(M, C, K, np.array([1.0, -0.2]))
    assert lam.real < 0 and lam.imag > 0
    assert np.linalg.matrix_rank(shape_basis(M, C, K, (2.0, -0.2))) == 4
