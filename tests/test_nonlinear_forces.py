"""The Nonlinearity & forces page's models: harmonic balance's describing functions and sweeps against the time
response, the force identification's G against the FOH simulation, and the trials' numbers."""

import math

import numpy as np
import pytest
import scipy.integrate

from vib_tutorial.core import force_id as fid
from vib_tutorial.core import nonlinear_test as nl
from vib_tutorial.core.model import ChainSystem
from vib_tutorial.core.simulator import foh_discretize

CHAIN = ChainSystem(np.ones(3), np.full(3, 1000.0), np.full(3, 4.0))
CUBIC = nl.Element(nl.Kind.CUBIC, 0, k3=2e6)
GAP = nl.Element(nl.Kind.GAP, 0, gap=0.005, k_gap=5000.0)


def first_harmonic(force, amplitude):
    """The first harmonic (cos part) of force(A cos θ) over a cycle, by quadrature."""
    theta = np.linspace(0, 2 * math.pi, 4001)[:-1]
    return 2 * np.mean(force(amplitude * np.cos(theta)) * np.cos(theta))


def test_describing_functions():
    for A in (0.001, 0.004, 0.006, 0.02):
        cubic = first_harmonic(lambda d: 2e6 * d**3, A)
        assert cubic == pytest.approx(0.75 * 2e6 * A**3, rel=1e-9)
        gap = first_harmonic(lambda d: 5000.0 * np.maximum(np.abs(d) - 0.005, 0) * np.sign(d), A)
        assert gap == pytest.approx(float(nl.gap_stiffness(5000.0, 0.005, np.array([A]))[0]) * A, rel=1e-5, abs=1e-12)
    # friction's: F sign(v) has a first harmonic 4F/π, in phase with the velocity
    X = np.array([0.0, 0.0, 0.01 + 0.002j])
    s = ChainSystem(np.ones(3), np.full(3, 1000.0), np.zeros(3), friction=np.array([0, 0, 0.3]))
    N = nl.describing(s, nl.Element(), X, 10.0, 0.01)
    assert abs(N[2]) == pytest.approx(4 * 0.3 / math.pi, rel=1e-6)
    assert np.angle(N[2]) == pytest.approx(np.angle(1j * X[2]), abs=1e-6)


def test_linear_is_homogeneous():
    freqs = np.linspace(1.5, 3.5, 50)
    sweeps = [nl.stepped_sine(CHAIN, nl.Element(), 2, F, freqs) for F in (0.1, 1.0, 10.0)]
    assert nl.homogeneity(sweeps, 2, [0.1, 1.0, 10.0]) < 1e-12
    M, C, K = CHAIN.matrices()
    w = 2 * math.pi * freqs[7]
    assert sweeps[1].X[7] == pytest.approx(np.linalg.solve(K - w * w * M + 1j * w * C, np.array([0, 0, 1.0])))


def residual(system, element, force, F, hz, X):
    M, C, K = system.matrices()
    w = 2 * math.pi * hz
    f = np.zeros(system.n)
    f[force] = F
    scale = float(np.abs(np.linalg.solve(K - w * w * M + 1j * w * C, f)).max())
    return np.linalg.norm((K - w * w * M + 1j * w * C) @ X + nl.describing(system, element, X, w, scale) - f)


@pytest.mark.parametrize(
    "system, element, F",
    [
        (CHAIN, CUBIC, 1.0),
        (CHAIN, GAP, 0.5),
        (
            ChainSystem(np.ones(3), np.full(3, 1000.0), np.full(3, 4.0), friction=np.array([0, 0, 0.1])),
            nl.Element(),
            0.5,
        ),
    ],
)
def test_every_step_is_a_steady_state(system, element, F):
    freqs = np.linspace(0.6 * 2.24, 1.6 * 2.24, 240)
    for up in (True, False):
        sweep = nl.stepped_sine(system, element, 2, F, freqs, up)
        worst = max(residual(system, element, 2, F, f, X) for f, X in zip(sweep.hz, sweep.X, strict=True))
        assert worst < 1e-8 * F


def test_hardening_sweeps_jump_and_disagree():
    freqs = np.linspace(0.6 * 2.24, 1.6 * 2.24, 240)
    up = nl.stepped_sine(CHAIN, CUBIC, 2, 1.0, freqs)
    down = nl.stepped_sine(CHAIN, CUBIC, 2, 1.0, freqs, False)
    assert len(up.jumps) == 1 and len(down.jumps) == 1
    f_up, f_down = up.hz[up.jumps[0]], down.hz[down.jumps[0]]
    assert f_down < f_up
    between = (freqs > f_down + 0.01) & (freqs < f_up - 0.01)
    a_up, a_down = np.abs(up.X[:, 2]), np.abs(down.X[::-1, 2])
    assert np.all(a_up[between] > 1.5 * a_down[between])  # up on the upper branch, down still on the lower
    outside = (freqs < f_down - 0.01) | (freqs > f_up + 0.01)
    assert a_up[outside] == pytest.approx(a_down[outside], rel=1e-6)


def time_reference(system, element, F, hz, x0, v0, cycles):
    """The same chain by solve_ivp from (x0, v0), over `cycles` drive cycles: the last cycle's displacements."""
    M, C, K = system.matrices()
    n = system.n

    def rhs(t, z):
        x, v = z[:n], z[n:]
        g = np.zeros(n)
        d = x[element.spring] - (x[element.spring - 1] if element.spring else 0.0)
        if element.kind is nl.Kind.CUBIC:
            g[element.spring] = element.k3 * d**3
        else:
            g[element.spring] = element.k_gap * max(abs(d) - element.gap, 0) * math.copysign(1, d)
        if element.spring:
            g[element.spring - 1] = -g[element.spring]
        f = np.zeros(n)
        f[2] = F * math.cos(2 * math.pi * hz * t)
        return np.concatenate([v, np.linalg.solve(M, f - K @ x - C @ v - g)])

    T = 1 / hz
    t = (cycles - 1) * T + np.arange(nl.STEPS_PER_CYCLE) * T / nl.STEPS_PER_CYCLE
    sol = scipy.integrate.solve_ivp(
        rhs, (0, cycles * T), np.concatenate([x0, v0]), t_eval=t, method="DOP853", rtol=1e-11, atol=1e-14
    )
    return sol.y[:n].T


# the gap's corner (its force's slope jumps) costs the step its second order there
@pytest.mark.parametrize(
    "element, F, hz, apart, tolerance", [(CUBIC, 0.5, 2.269, 0.04, 2e-3), (GAP, 0.5, 2.3, 0.15, 1e-2)]
)
def test_steady_response_against_an_ode_solver(element, F, hz, apart, tolerance):
    start = nl.harmonic_balance(CHAIN, element, 2, F, hz).X
    w = nl.steady_response(CHAIN, element, 2, F, hz, start)
    x0, v0 = start.real, (2j * math.pi * hz * start).real
    ref = time_reference(CHAIN, element, F, hz, x0, v0, w.cycles)
    assert np.abs(w.x - ref).max() < tolerance * np.abs(ref).max()  # 64 steps a cycle
    # one harmonic is near the whole response (the gap's corner less so), and the rest are odd
    assert abs(start[2]) == pytest.approx(w.harmonics[0, 2], rel=apart)
    assert w.harmonics[1, 2] < 1e-4 * w.harmonics[0, 2] and w.harmonics[2, 2] > 1e-3 * w.harmonics[0, 2]


# ------------------------------------------------------------------ force identification
@pytest.mark.parametrize("output", list(fid.Output))
def test_g_is_the_foh_simulation(output):
    h = fid.step_for(CHAIN)
    f = fid.force_history(fid.Shape.DOUBLE, 120, h, 0.1)
    G = fid.markov(CHAIN, 2, (0, 2), output, h, 120)
    M, C, K = CHAIN.matrices()
    Minv = np.linalg.inv(M)
    A = np.block([[np.zeros((3, 3)), np.eye(3)], [-Minv @ K, -Minv @ C]])
    B = np.concatenate([np.zeros(3), Minv[:, 2]])[:, None]
    Phi, G0, G1 = foh_discretize(A, B, h)
    z = np.zeros(6)
    previous = 0.0
    y = np.zeros((120, 2))
    for k in range(120):
        z = Phi @ z + G0[:, 0] * previous + G1[:, 0] * f[k]
        previous = f[k]
        if output is fid.Output.DISPLACEMENT:
            y[k] = z[[0, 2]]
        else:
            acc = A[3:] @ z + Minv[:, 2] * f[k]
            y[k] = acc[[0, 2]]
    assert G @ f == pytest.approx(np.concatenate([y[:, 0], y[:, 1]]), abs=1e-12 * np.abs(y).max())


def test_regularized_solutions():
    h = fid.step_for(CHAIN)
    f = fid.force_history(fid.Shape.IMPACT, 150, h, 0.1)
    G = fid.markov(CHAIN, 2, (0,), fid.Output.DISPLACEMENT, h, 150)
    p = fid.set_up(G, f, 0.02)
    lam = 1e-4
    normal = np.linalg.solve(G.T @ G + lam * lam * np.eye(150), G.T @ p.y)
    assert fid.tikhonov(p, lam) == pytest.approx(normal, rel=1e-6, abs=1e-9)
    kept = int(np.sum(p.s > p.s[0] * 1e-15))
    assert fid.tsvd(p, kept) == pytest.approx(fid.inverse(p))
    lam = fid.discrepancy(p)
    assert np.linalg.norm(G @ fid.tikhonov(p, lam) - p.y) == pytest.approx(p.noise, rel=1e-6)
    curve = fid.l_curve(p)
    assert curve.size == pytest.approx(np.sort(curve.size)[::-1])  # ‖f‖ falls as λ grows, the residual rises
    assert curve.residual == pytest.approx(np.sort(curve.residual))


# ------------------------------------------------------------------ the trials' numbers
def test_trials():
    from vib_tutorial.gui.nonlinear_forces import BAND, STEPS, TRIALS, _solve

    by = {t.name: t for t in TRIALS}

    def sweeps(t):
        s = t.system()
        M, _, K = s.matrices()
        f0 = math.sqrt(sorted(np.linalg.eigvals(np.linalg.solve(M, K)).real)[t.mode]) / (2 * math.pi)
        freqs = np.linspace(BAND[0] * f0, BAND[1] * f0, STEPS)
        return f0, [
            (nl.stepped_sine(s, t.element, t.force, F, freqs), nl.stepped_sine(s, t.element, t.force, F, freqs, False))
            for F in t.levels
        ]

    def peak(sweep, F, out=2):
        a = np.abs(sweep.X[:, out]) / F
        return sweep.hz[np.argmax(a)], a.max()

    t = by["Linear: every level agrees"]
    f0, runs = sweeps(t)
    assert f0 == pytest.approx(2.240, abs=5e-4)
    assert nl.homogeneity([u for u, _ in runs], 2, list(t.levels)) < 1e-12

    t = by["A hardening spring"]
    _, runs = sweeps(t)
    assert peak(runs[1][0], 0.5)[0] == pytest.approx(2.319, abs=5e-4)
    up, down = runs[2]
    assert peak(up, 1.0)[0] == pytest.approx(2.431, abs=5e-4)
    assert [round(up.hz[k], 3) for k in up.jumps] == [2.450] and [round(down.hz[k], 3) for k in down.jumps] == [2.422]
    assert 100 * nl.homogeneity([u for u, _ in runs], 2, list(t.levels)) == pytest.approx(67, abs=0.5)

    t = by["A gap: linear, until it closes"]
    _, runs = sweeps(t)
    assert nl.homogeneity([runs[0][0], runs[1][0]], 2, list(t.levels[:2])) < 1e-12
    up, down = runs[2]
    assert [round(up.hz[k], 3) for k in up.jumps] == [2.553] and [round(down.hz[k], 3) for k in down.jumps] == [2.356]

    t = by["Friction: damping that fades"]
    _, runs = sweeps(t)
    assert peak(runs[0][0], 0.2)[1] == pytest.approx(0.0177, abs=5e-5)
    assert peak(runs[2][0], 2.0)[1] == pytest.approx(0.0456, abs=5e-5)
    assert peak(sweeps(by["Linear: every level agrees"])[1][0][0], 0.05)[1] == pytest.approx(0.0487, abs=5e-5)
    assert t.friction[2] / t.levels[2] == pytest.approx(1 / 20)

    t = by["Distortion: odd harmonics"]
    F = t.levels[t.level]
    start = nl.harmonic_balance(t.system(), t.element, t.force, F, t.hz).X
    w = nl.steady_response(t.system(), t.element, t.force, F, t.hz, start)
    h = w.harmonics[:, t.out] / w.harmonics[0, t.out]
    assert 100 * h[2] == pytest.approx(1.9, abs=0.05) and h[4] < h[2] and h[1] < 1e-5
    assert 100 * (abs(start[t.out]) / w.harmonics[0, t.out] - 1) == pytest.approx(3, abs=0.5)

    def identified(t):
        s = t.system()
        h = fid.step_for(s)
        truth = fid.force_history(t.shape, fid.SAMPLES, h, t.width)
        p = fid.set_up(fid.markov(s, t.hit, t.sensors, t.output, h, fid.SAMPLES), truth, t.noise / 100)
        return p, truth, _solve(p, t.method, 10**t.log_lam, t.k)

    p, truth, est = identified(by["Force identification: the plain inverse"])
    assert math.log10(p.s[0] / p.s[-1]) == pytest.approx(20, abs=0.5)
    assert np.abs(est).max() / np.abs(truth).max() == pytest.approx(73000, rel=0.01)

    t = by["Tikhonov at the L-curve's corner"]
    p, truth, est = identified(t)
    assert fid.l_curve(p).corner == pytest.approx(10**t.log_lam, rel=0.01)
    assert 10**t.log_lam == pytest.approx(5.4e-5, rel=0.01)
    assert 100 * fid.error(est, truth) == pytest.approx(15, abs=0.5)

    t = by["Measure the acceleration where the force is"]
    p, truth, est = identified(t)
    assert math.log10(p.s[0] / p.s[-1]) == pytest.approx(3, abs=0.05)
    assert 10**t.log_lam == pytest.approx(0.024, rel=0.02) and 100 * fid.error(est, truth) == pytest.approx(
        5.5, abs=0.5
    )
    # the spec's claim: a displacement on mass 1 gets about 15 % at best
    p, truth, _ = identified(by["Tikhonov at the L-curve's corner"])
    best = min(fid.error(fid.tikhonov(p, lam), truth) for lam in np.geomspace(1e-7, 1e-2, 200))
    assert 100 * best == pytest.approx(15, abs=0.5)
