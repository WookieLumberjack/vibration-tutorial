"""The Frame builder's forced response: the state space against a direct solve, the conventional model against it
when the damping is proportional, the exact stepping against an ODE solver."""

import dataclasses

import numpy as np
import pytest
import scipy.integrate

from vib_tutorial.core.frame import (
    TWO_PI,
    Element,
    Kind,
    assemble,
    cantilever,
    modes,
    portal,
    storeys,
)
from vib_tutorial.core.frame_response import (
    Load,
    LoadKind,
    exact_model,
    formulations,
    frf,
    frf_grid,
    rayleigh,
    record,
    response,
    responses,
)


def floor_damper(c: float = 2000.0):
    """The two-storey frame with a damper from its first floor (node 3) to the ground."""
    return storeys().with_element(Element(Kind.DAMPER, 2, None, value=c, anchor=(-0.5, 1.0)))


def direct_frf(frame, freqs, inp, out):
    """(K − ω²M + iωC)⁻¹ over every free DOF, massless ones included: no condensing, no state space."""
    a = assemble(frame)
    free = a.free_dofs
    M, C, K = (X[np.ix_(free, free)] for X in (a.M, a.C, a.K))
    f = np.zeros(free.size)
    f[list(free).index(inp)] = 1.0
    k = list(free).index(out)
    return np.array([np.linalg.solve(K - w * w * M + 1j * w * C, f)[k] for w in TWO_PI * freqs])


def test_rayleigh_damping_has_its_zeta_at_the_first_two_modes():
    f = dataclasses.replace(portal(), zeta=0.03)
    forms = formulations(f)
    zetas = [m.zeta_conventional for m in forms.modes[:3]]
    assert zetas[:2] == pytest.approx([0.03, 0.03], rel=1e-9)
    assert zetas[2] > 0.03  # beyond mode 2, βω takes over
    alpha, beta = rayleigh(0.03, forms.result.hz)
    w1 = TWO_PI * forms.result.hz[0]
    assert 0.5 * (alpha / w1 + beta * w1) == pytest.approx(0.03)


def test_proportional_damping_the_two_models_agree():
    f = dataclasses.replace(storeys(), zeta=0.02)
    forms = formulations(f)
    assert forms.proportional
    for m in forms.modes:
        if m.pole is not None:
            assert m.hz_exact == pytest.approx(m.hz, rel=1e-9)
            assert m.zeta_exact == pytest.approx(m.zeta_conventional, rel=1e-6, abs=1e-12)
            assert m.measures is not None and m.measures.mpc == pytest.approx(1.0, abs=1e-9)
    for kind in LoadKind:
        r = responses(forms, Load(6, 0, kind, 10.0, 5.0), 18)
        assert np.abs(r.frf_exact - r.frf_conventional).max() < 1e-7 * np.abs(r.frf_exact).max()
        assert np.abs(r.exact - r.conventional).max() < 1e-7 * np.abs(r.exact).max()


def test_a_damper_makes_the_modes_complex_and_the_models_part():
    forms = formulations(floor_damper())
    assert not forms.proportional
    first = forms.modes[0]
    assert first.measures is not None and first.measures.mpc < 0.99
    assert first.hz_exact != pytest.approx(first.hz, rel=1e-3)
    r = responses(forms, Load(2, 0, LoadKind.IMPULSE, 1.0), 6)
    assert np.abs(r.frf_exact - r.frf_conventional).max() > 0.05 * np.abs(r.frf_exact).max()


@pytest.mark.parametrize(
    "frame",
    [
        floor_damper(),
        # a lumped beam's turns have no mass; a torsion damper on one keeps a first-order state
        dataclasses.replace(cantilever(1.0, 3), lumped=True).with_element(Element(Kind.TORSION_DAMPER, 2, value=50.0)),
        # and a damper on a massless node, held by a spring
        cantilever(1.0, 2)
        .with_node(1.0, 0.5)
        .with_element(Element(Kind.SPRING, 2, 3, value=2e5))
        .with_element(Element(Kind.DAMPER, 3, None, value=300.0, anchor=(1.0, 1.0))),
    ],
)
def test_the_state_space_frf_is_the_direct_solve(frame):
    a = assemble(frame)
    forms = formulations(frame, a)
    freqs = frf_grid(forms.result.hz, 120)
    inp = out = 3 * 2 + 1 if frame.nodes[2].y == 0 else 3 * 2
    exact = frf(forms.exact, freqs, inp, out)
    assert exact == pytest.approx(direct_frf(frame, freqs, inp, out), rel=1e-7, abs=1e-12 * np.abs(exact).max())


def test_a_massless_damped_dof_adds_a_relaxation():
    frame = dataclasses.replace(cantilever(1.0, 3), lumped=True).with_element(
        Element(Kind.TORSION_DAMPER, 2, value=50.0)
    )
    a = assemble(frame)
    model = exact_model(frame, a, modes(frame, a))
    forms = formulations(frame, a)
    assert model.size == 2 * forms.conventional.size // 2 + 1
    assert forms.roots.size >= 1


def second_order(frame, load, t):
    """M q'' + C q' + K q = f over the free DOFs (every one with mass), by solve_ivp."""
    a = assemble(frame)
    free = a.free_dofs
    M, C, K = (X[np.ix_(free, free)] for X in (a.M, a.C, a.K))
    n = free.size
    e = np.zeros(n)
    e[list(free).index(load.index)] = 1.0
    Minv = np.linalg.inv(M)

    def force(s):
        if load.kind is LoadKind.HARMONIC:
            return load.size * np.sin(TWO_PI * load.hz * s)
        return load.size

    def rhs(s, z):
        return np.concatenate([z[n:], Minv @ (e * force(s) - C @ z[n:] - K @ z[:n])])

    z0 = np.zeros(2 * n)
    sol = scipy.integrate.solve_ivp(rhs, (0, t[-1]), z0, t_eval=t, method="DOP853", rtol=1e-10, atol=1e-14)
    q = np.zeros((t.size, frame.n_dof))
    q[:, free] = sol.y[:n].T
    return q


@pytest.mark.parametrize("kind", [LoadKind.HARMONIC, LoadKind.STEP])
def test_the_exact_steps_match_an_ode_solver(kind):
    frame = cantilever(1.0, 2).with_element(Element(Kind.DAMPER, 2, None, value=40.0, anchor=(1.0, -0.5)))
    load = Load(2, 1, kind, 100.0, 30.0)
    a = assemble(frame)
    model = exact_model(frame, a, modes(frame, a))
    t = np.linspace(0.0, 0.2, 401)
    q = response(model, load, t)
    ref = second_order(frame, load, t)
    assert np.abs(q - ref).max() < 1e-6 * np.abs(ref).max()


def test_an_impulse_is_the_inverse_transform_of_the_frf():
    """A damped impulse response, in steady harmonic terms: its Fourier transform at f is J·H(f)."""
    frame = dataclasses.replace(cantilever(1.0, 2), zeta=0.05)
    forms = formulations(frame)
    load = Load(2, 1, LoadKind.IMPULSE, 2.0)
    t = np.linspace(0.0, 4.0, 40001)
    q = response(forms.exact, load, t)[:, 7]
    f = forms.result.hz[0]
    transform = np.trapezoid(q * np.exp(-1j * TWO_PI * f * t), t)
    assert transform == pytest.approx(2.0 * frf(forms.exact, np.array([f]), 7, 7)[0], rel=1e-4)


def test_harmonic_at_resonance_grows_without_damping():
    frame = cantilever(1.0, 2)
    forms = formulations(frame)
    f1 = forms.result.hz[0]
    load = Load(2, 1, LoadKind.HARMONIC, 1.0, f1)
    t = record(forms.result.hz, load)
    q = response(forms.exact, load, t)[:, 7]
    assert np.all(np.isfinite(q))
    early, late = np.abs(q[: t.size // 4]).max(), np.abs(q[-t.size // 4 :]).max()
    assert late > 3 * early


def test_a_step_settles_on_the_static_deflection():
    frame = dataclasses.replace(cantilever(1.0, 4), zeta=0.1)
    forms = formulations(frame)
    load = Load(4, 1, LoadKind.STEP, 1000.0)
    t = np.linspace(0.0, 3.0, 3001)
    q = response(forms.exact, load, t)[:, 13]
    e = frame.elements[0]
    static = 1000.0 / (3 * e.material.E * e.section.inertia)  # F L³ / 3EI, L = 1
    assert q[-1] == pytest.approx(static, rel=1e-3)
    assert q.max() == pytest.approx(static * (1 + np.exp(-np.pi * 0.1 / np.sqrt(1 - 0.01))), rel=0.02)


# ------------------------------------------------------------------ the trials' numbers
def peaks(freqs, H):
    a = np.abs(H)
    k = np.flatnonzero((a[1:-1] > a[:-2]) & (a[1:-1] > a[2:])) + 1
    return freqs[k], a[k]


def test_trials():
    from vib_tutorial.gui.frame_builder import TRIALS

    by = {t.name: t for t in TRIALS}
    t = by["Rayleigh damping: one answer"]
    forms = formulations(t.frame)
    assert forms.proportional
    assert [m.hz for m in forms.modes[:3]] == pytest.approx([3.44, 9.14, 92.02], abs=0.005)
    assert [m.zeta_conventional for m in forms.modes[:3]] == pytest.approx([0.02, 0.02, 0.147], abs=5e-4)
    assert all(m.measures.mpc > 0.9995 for m in forms.modes if m.measures is not None)

    t = by["A damper in the first storey"]
    forms = formulations(t.frame)
    one, two = forms.modes[:2]
    assert (one.hz, two.hz) == pytest.approx((3.44, 9.14), abs=0.005)
    assert (one.zeta_conventional, two.zeta_conventional) == pytest.approx((0.239, 0.232), abs=5e-4)
    assert (one.hz_exact, two.hz_exact) == pytest.approx((3.64, 8.65), abs=0.005)
    assert (one.zeta_exact, two.zeta_exact) == pytest.approx((0.246, 0.237), abs=5e-4)
    assert (one.measures.mpc, two.measures.mpc) == pytest.approx((0.796, 0.833), abs=5e-4)
    r = responses(forms, t.load, t.load.index)
    assert np.abs(r.frf_exact - r.frf_conventional).max() / np.abs(r.frf_exact).max() == pytest.approx(0.258, abs=5e-4)

    t = by["A step: twice as far"]
    e = t.frame.elements[0]
    static = 1000.0 / (3 * e.material.E * e.section.inertia)
    r = responses(formulations(t.frame), t.load, t.load.index)
    peak = r.exact[:, t.load.index].max()
    assert (1000 * static, 1000 * peak, peak / static) == pytest.approx((3.05, 6.08, 1.997), abs=0.005)

    t = by["Resonance from rest"]
    forms = formulations(t.frame)
    assert forms.result.hz[0] == pytest.approx(t.load.hz, abs=0.005)
    i = t.load.index
    steady = abs(frf(forms.exact, np.array([t.load.hz]), i, i)[0]) * t.load.size
    still = t.load.size / (3 * e.material.E * e.section.inertia)
    assert (1000 * steady, 1000 * still, steady / still) == pytest.approx((14.8, 0.305, 48.5), abs=0.05)
    r = responses(forms, t.load, i)
    assert r.t[-1] == pytest.approx(0.48, abs=0.005)
    assert 1000 * np.abs(r.exact[:, i]).max() == pytest.approx(10.6, abs=0.05)
    assert 1 / (0.01 * TWO_PI * forms.result.hz[0]) == pytest.approx(0.38, abs=0.005)

    t = by["A tuned mass damper"]
    bare = formulations(cantilever(1.0, 4))
    tip = bare.result.shapes[13, 0]
    assert 1 / tip**2 == pytest.approx(4.91, abs=0.005)
    m, f1 = 0.25, bare.result.hz[0]
    mu = m * tip**2
    assert 100 * mu == pytest.approx(5.1, abs=0.05)
    tuned = f1 / (1 + mu)
    zeta = np.sqrt(3 * mu / (8 * (1 + mu) ** 3))
    assert (tuned, 100 * zeta) == pytest.approx((39.75, 12.8), abs=0.05)
    assert (m, m * (TWO_PI * tuned) ** 2 / 1e3, 2 * zeta * m * TWO_PI * tuned) == pytest.approx(
        (0.25, 15.6, 16.0), abs=0.05
    )
    forms = formulations(t.frame)
    r = responses(forms, t.load, 13)
    f, a = peaks(r.freqs, r.frf_exact)
    assert f[:2] == pytest.approx([37.4, 44.0], abs=0.05)
    assert a[:2] / still * t.load.size == pytest.approx([5.61, 5.53], abs=0.005)
    _, b = peaks(r.freqs, r.frf_conventional)
    assert b[:2] / still * t.load.size == pytest.approx([3.74, 3.86], abs=0.005)
    plain = formulations(dataclasses.replace(cantilever(1.0, 4), zeta=0.01))
    alone = abs(frf(plain.exact, plain.result.hz[:1], 13, 13)[0]) / still * t.load.size
    assert alone == pytest.approx(48.5, abs=0.05)
    assert [mode.measures.mpc for mode in forms.modes[:2]] == pytest.approx([0.833, 0.596], abs=5e-4)
