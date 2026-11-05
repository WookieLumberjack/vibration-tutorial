"""The Test planning page's model: effective independence against the Fisher information it stands for, the ODS
and its modal parts, and the trials' numbers."""

import itertools

import numpy as np
import pytest

from vib_tutorial.core.model import ChainSystem
from vib_tutorial.core.test_planning import (
    Method,
    at_phase,
    driving_point_residue,
    effective_independence,
    evenly_spaced,
    mac,
    ods,
    ods_parts,
    ods_spectrum,
    place,
    quality,
    shares,
    undamped_modes,
)

UNIFORM = ChainSystem(np.ones(8), np.full(8, 1000.0), np.full(8, 1.0))


def fisher(phi, rows):
    s = phi[list(rows)]
    return np.linalg.det(s.T @ s)


def test_shares_are_a_projector_diagonal():
    _, phi = undamped_modes(UNIFORM)
    for m in (1, 3, 5):
        E = shares(phi[:, :m])
        assert E.sum() == pytest.approx(m)
        assert np.all((E > -1e-12) & (E < 1 + 1e-12))
    one = np.array([0.2, 0.5, 1.0])
    assert shares(one[:, None]) == pytest.approx(one**2 / np.sum(one**2))


def test_dropping_a_dof_scales_the_information_by_one_less_its_share():
    _, phi = undamped_modes(UNIFORM)
    T = phi[:, :3]
    E = shares(T)
    whole = fisher(T, range(8))
    for i in range(8):
        rest = [j for j in range(8) if j != i]
        assert fisher(T, rest) == pytest.approx(whole * (1 - E[i]))


def test_efi_drops_the_least_share_each_step():
    hz, phi = undamped_modes(UNIFORM)
    T = phi[:, :3]
    p = effective_independence(T, 3)
    assert [s.dropped for s in p.steps] == [0, 5, 3, 6, 2]
    for step in p.steps:
        assert step.score == pytest.approx(shares(T[step.candidates]))
        assert step.dropped == step.candidates[np.argmin(step.score)]
    assert list(p.sensors) == [1, 4, 7]
    # greedy, but here as good as the best of all 56 sets of three
    best = max(itertools.combinations(range(8), 3), key=lambda rows: fisher(T, rows))
    assert fisher(T, p.sensors) == pytest.approx(fisher(T, best))
    weighted = effective_independence(T, 3, hz[:3])
    dpr = driving_point_residue(T, hz[:3])
    assert weighted.steps[0].score == pytest.approx(shares(T) * dpr)
    with pytest.raises(ValueError):
        effective_independence(T, 2)


def test_quality():
    _, phi = undamped_modes(UNIFORM)
    T = phi[:, :3]
    q = quality(T, np.arange(8))
    assert q.information == pytest.approx(1.0) and q.noise == pytest.approx(np.ones(3))
    q = quality(T, np.array([0, 1]))
    assert q.singular and np.all(np.isinf(q.noise))
    assert list(evenly_spaced(8, 3)) == [2, 4, 7] and list(evenly_spaced(8, 4)) == [1, 3, 5, 7]
    hand = place(T, np.ones(3), Method.HAND, 3, (7, 1, 1, 4, 12))
    assert list(hand.sensors) == [1, 4, 7]


def test_ods_and_its_parts():
    M, C, K = UNIFORM.matrices()
    for force, hz in ((7, 1.842), (3, 2.755), (0, 7.0)):
        x = ods(UNIFORM, force, hz)
        w = 2 * np.pi * hz
        f = np.zeros(8)
        f[force] = 1.0
        assert x == pytest.approx(np.linalg.solve(K - w * w * M + 1j * w * C, f))
        assert ods_parts(UNIFORM, force, hz).sum(axis=1) == pytest.approx(x, abs=1e-12)
        assert ods_spectrum(UNIFORM, force, np.array([hz]))[0] == pytest.approx(x)
    # with non-proportional damping too
    odd = ChainSystem(np.ones(4), np.full(4, 500.0), np.array([0.0, 30.0, 0.0, 2.0]))
    x = ods(odd, 2, 3.0)
    assert ods_parts(odd, 2, 3.0).sum(axis=1) == pytest.approx(x, abs=1e-12)
    # a real shape's phase comes off whole
    real = np.array([1.0, -2.0, 0.5])
    turned = at_phase(real * np.exp(0.7j))
    assert turned * np.sign(turned[0]) == pytest.approx(real)


# ------------------------------------------------------------------ the trials' numbers
def test_trials():
    from vib_tutorial.gui.test_planning import TRIALS

    by = {t.name: t for t in TRIALS}
    hz, phi = undamped_modes(UNIFORM)
    assert hz[:3] == pytest.approx([0.929, 2.755, 4.487], abs=5e-4)
    T = phi[:, :3]

    def placed(t):
        assert t.system().n == 8
        p = place(T, hz[:3], t.method, t.sensors, t.hand)
        return p, quality(T, p.sensors)

    p, q = placed(by["Three sensors by effective independence"])
    assert [s.dropped + 1 for s in p.steps] == [1, 6, 4, 7, 3] and list(p.sensors + 1) == [2, 5, 8]
    assert 100 * q.information == pytest.approx(10.5, abs=0.05)
    assert q.noise == pytest.approx([1.61, 1.55, 1.29], abs=0.005) and q.worst == pytest.approx(0.041, abs=5e-4)

    p, q = placed(by["Evenly spaced isn't as good"])
    assert list(p.sensors + 1) == [3, 5, 8]
    assert 100 * q.information == pytest.approx(7.75, abs=0.005)
    assert q.auto_mac[1, 2] == pytest.approx(0.188, abs=5e-4) and q.worst == q.auto_mac[1, 2]

    p, q = placed(by["Sensors bunched at the ground: aliasing"])
    assert list(p.sensors + 1) == [1, 2, 3]
    assert q.auto_mac[0, 1] == pytest.approx(0.976, abs=5e-4)
    assert q.information == pytest.approx(7.8e-7, abs=5e-9) and q.noise[0] == pytest.approx(240, abs=0.5)

    t = by["One sensor more"]
    p, q = placed(t)
    assert list(p.sensors + 1) == [2, 3, 5, 8] and 100 * q.information == pytest.approx(19.9, abs=0.05)
    assert q.noise[1] == pytest.approx(1.23, abs=0.005)

    t = by["An ODS at a mode is the mode"]
    assert t.hz == pytest.approx(hz[1], abs=5e-4)
    x = ods(t.system(), t.force, t.hz)
    assert mac(x, phi[:, 1]) == pytest.approx(0.999, abs=5e-4)
    size = np.max(np.abs(ods_parts(t.system(), t.force, t.hz)), axis=0)
    assert size[1] / np.delete(size, 1).max() == pytest.approx(49.5, abs=0.05)

    t = by["Between two modes: a mix"]
    assert t.hz == pytest.approx(0.5 * (hz[0] + hz[1]), abs=5e-4)
    x = ods(t.system(), t.force, t.hz)
    assert (mac(x, phi[:, 0]), mac(x, phi[:, 1])) == pytest.approx((0.734, 0.249), abs=5e-4)
    parts = ods_parts(t.system(), t.force, t.hz)
    size = np.max(np.abs(parts), axis=0) / np.max(np.abs(x))
    assert size[:2] == pytest.approx([0.86, 0.50], abs=0.005)
    assert np.real(parts[7, 0] * np.conj(parts[7, 1])) < 0  # opposite phase

    t = by["Move the force"]
    x = ods(t.system(), t.force, t.hz)
    assert (mac(x, phi[:, 0]), mac(x, phi[:, 1])) == pytest.approx((0.650, 0.331), abs=5e-4)
    shape = at_phase(x)
    shape *= np.sign(shape[7])
    assert np.all(np.abs(shape[:4]) < 0.1 * np.abs(shape).max()) and np.all(shape[4:] > 0)

    t = by["A reference that hardly moves"]
    x = ods(t.system(), t.force, t.hz)
    ratio = abs(x[t.reference]) / np.abs(x).max()
    assert 100 * ratio == pytest.approx(3.2, abs=0.05) and np.abs(x).max() == pytest.approx(abs(x[7]))
    assert 1 / ratio == pytest.approx(32, abs=0.5)
