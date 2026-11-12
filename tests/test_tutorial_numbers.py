"""The numbers quoted in the tutorial text (Background, Theory tabs, Try it lists, README).

Each test recomputes what a page tells the student they will see, so a change to a
default, a preset or a solver cannot silently make the text wrong. If one fails, fix
the text (or the change), and quote the new value.
"""

import math

import numpy as np
import pytest
from scipy.signal import find_peaks

from vib_tutorial.core import (
    ChainSystem,
    Simulator,
    compare_coupled,
    compare_modes,
    component_mode_synthesis,
    coupled_model,
    frf,
    modal_analysis,
    modal_energies,
    modal_frf_terms,
    transmissibility,
)
from vib_tutorial.core.presets import presets


def peaks(system: ChainSystem, dof: int, scale: float, lo: float = 0.5, hi: float = 8.0) -> list[float]:
    f = np.linspace(lo, hi, 100001)
    h = np.abs(frf(system, f, dof)[:, dof]) * scale
    return [float(h[i]) for i in find_peaks(h)[0]]


def preset(name: str):
    return next(p for p in presets() if name in p.name)


def with_c1(c1: float) -> ChainSystem:
    s = ChainSystem.uniform(4)
    s.damping[0] = c1
    return s


@pytest.mark.parametrize(
    ("c1", "coupling", "modal", "exact"),
    [
        (5, 0.36, 0.0977, 0.0979),
        (15, 0.71, 0.168, 0.180),
        (50, 0.90, 0.45, 0.98),
    ],
)
def test_background_coupling_table(c1, coupling, modal, exact):
    r = modal_analysis(with_c1(c1))
    assert r.coupling == pytest.approx(coupling, abs=0.005)
    pairs = [(m.zeta_modal, m.damped.zeta) for m in r.modes if m.damped]
    assert any(z_m == pytest.approx(modal, rel=0.01) and z_e == pytest.approx(exact, rel=0.01) for z_m, z_e in pairs)


def test_background_try_it_overdamped_at_c1_200():
    roots = modal_analysis(with_c1(200)).overdamped_roots
    assert len(roots) == 2 and roots[0] > -3 and roots[1] < -100  # one slow, one very fast


def test_background_try_it_energy_moves_to_mode_1():
    for c1, mode1_share in ((15, 0.85), (2, None)):
        s = with_c1(c1)
        r = modal_analysis(s)
        sim = Simulator(s)
        sim.set_state(0.02 * r.modes[2].shape, np.zeros(4))
        for _ in range(20):
            sim.advance(0.05)  # 1 s
        e = modal_energies(s, r, sim.displacement, sim.velocity)
        if mode1_share is None:
            assert e[2] / e.sum() == pytest.approx(1.0, abs=1e-6)  # mode 3 keeps it all
        else:
            assert e[0] / e.sum() > mode1_share  # "within a second most of what is left is in mode 1"


def test_presets_absorber_and_tuned_mass_damper():
    absorber = preset("absorber (undamped)").system
    f = modal_analysis(absorber).modes
    assert (f[1].fn_hz - f[0].fn_hz) / 3.183 == pytest.approx(0.1**0.5, abs=0.01)  # "√0.1 ≈ 32% apart"
    tmd = preset("Den Hartog").system
    assert max(peaks(tmd.resized(1), 0, 400.0)) == pytest.approx(40.0, rel=0.01)  # "40 times"
    assert max(peaks(tmd, 0, 400.0)) < 6.0  # "under 6"
    undamped = ChainSystem(tmd.masses, tmd.stiffness, [0.0, tmd.damping[1]])
    assert peaks(undamped, 0, 400.0) == pytest.approx([6.4, 6.4], rel=0.005)  # "about 6.4", equal
    for c2, count in ((0.0, 2), (5.0, 1)):  # "Set c2 to 0: two sharp peaks. Set it to 5: one"
        assert len(peaks(ChainSystem(tmd.masses, tmd.stiffness, [tmd.damping[0], c2]), 0, 400.0)) == count


def coupling_errors(system, split, mode_a=0, mode_b=0, a_mass="tip"):
    m = coupled_model(system, split, mode_a, mode_b, a_mass)
    return m, [(c.error, c.mac) for c in compare_coupled(m, modal_analysis(system))]


def test_modal_coupling_try_it():
    _, errs = coupling_errors(preset("absorber (undamped)").system, 1)
    assert all(abs(e) < 1e-9 for e, _ in errs)
    tuned = ChainSystem([1, 1, 0.1, 0.1], [400, 400, 40, 40], [2, 2, 0.2, 0.2])
    m, errs = coupling_errors(tuned, 2)
    assert m.mass_ratio == pytest.approx(0.14, abs=0.005)
    assert [e for e, _ in errs] == pytest.approx([0.002, 0.0033], abs=0.0005)  # "about 0.2%"
    _, errs = coupling_errors(tuned, 2, a_mass="effective")
    assert max(abs(e) for e, _ in errs) == pytest.approx(0.03, abs=0.002)  # "about 3%", either sign
    _, errs = coupling_errors(ChainSystem.uniform(8), 6, mode_a=1)
    assert errs[0][0] == pytest.approx(-0.14, abs=0.005)  # "14% too low"


def cms(interfaces, kept, method="craig-bampton"):
    s = ChainSystem.uniform(8)
    return [
        (c.error, c.mac)
        for c in compare_modes(component_mode_synthesis(s, interfaces, kept, method), modal_analysis(s))
        if c.error is not None
    ]


@pytest.mark.parametrize(
    ("interfaces", "kept", "method", "mode", "error", "tol"),
    [
        ([], [0], "craig-bampton", 1, 0.073, 0.001),  # Guyan, +7.3%
        ([], [1], "craig-bampton", 1, 0.0033, 0.0001),
        ([], [1], "craig-bampton", 2, 0.074, 0.001),
        ([], [2], "craig-bampton", 3, 0.044, 0.001),
        ([], [1], "rubin", 2, 0.16, 0.005),
        ([3], [1, 1], "craig-bampton", 4, 0.07, 0.002),
        ([3], [0, 0], "craig-bampton", 2, 0.11, 0.005),
        ([3], [2, 2], "craig-bampton", 6, 0.027, 0.001),
        ([6], [1, 0], "craig-bampton", 3, 0.50, 0.01),
        ([2, 5], [1, 1, 1], "craig-bampton", 4, 0.023, 0.001),
        ([2, 5], [0, 0, 0], "craig-bampton", 3, 0.16, 0.005),
        ([3], [1, 1], "rubin", 4, 0.10, 0.005),
        ([3], [1, 1], "macneal", 2, 0.22, 0.005),
        ([3], [9, 9], "macneal", 6, 0.66, 0.005),
    ],
)
def test_substructuring_try_it(interfaces, kept, method, mode, error, tol):
    assert cms(interfaces, kept, method)[mode - 1][0] == pytest.approx(error, abs=tol)


def test_background_single_mass_numbers():
    s = ChainSystem.uniform(1)
    m, k, c = s.masses[0], s.stiffness[0], s.damping[0]
    wn = (k / m) ** 0.5
    zeta = c / (2 * (k * m) ** 0.5)
    assert (wn, wn / (2 * np.pi), zeta) == pytest.approx((20.0, 3.18, 0.05), rel=0.002)
    assert 1 / (zeta * wn) == pytest.approx(1.0)  # time constant
    assert 2 * np.pi * zeta / (1 - zeta**2) ** 0.5 == pytest.approx(0.31, abs=0.005)  # log decrement
    assert max(peaks(s, 0, k, 1.0, 6.0)) == pytest.approx(1 / (2 * zeta), rel=0.002)  # Q = 10
    assert 2 * zeta * wn / (2 * np.pi) == pytest.approx(0.32, abs=0.005)  # half-power bandwidth, Hz


def test_background_rayleigh_damping_of_the_default_chain():
    zetas = [m.zeta_modal for m in modal_analysis(ChainSystem.uniform(4)).modes]
    assert zetas[0] == pytest.approx(0.017, abs=0.0005) and zetas[3] == pytest.approx(0.094, abs=0.0005)
    for m in modal_analysis(ChainSystem.uniform(4)).modes:
        assert m.zeta_modal == pytest.approx(0.005 * m.omega_n / 2)  # zeta = beta omega / 2


def test_background_isolation_above_root_2_fn():
    for c in (0.5, 2.0, 20.0):
        s = ChainSystem([1.0], [400.0], [c])
        f = np.array([0.99, 1.01]) * 2**0.5 * 20 / (2 * np.pi)
        t = np.abs(transmissibility(s, f))[:, 0]
        assert t[0] > 1 > t[1]
    f = np.linspace(0.01, 20, 20001)
    t3 = np.abs(transmissibility(ChainSystem.uniform(4), f))[:, 2]
    assert f[np.nonzero(t3 > 1)[0][-1]] == pytest.approx(1.65, abs=0.01)  # m3's crossover


def test_frf_matrix_three_forms_and_residual_mass():
    s = ChainSystem.uniform(4)
    r = modal_analysis(s)
    hi = np.array([60.0, 120.0])  # far above the modes: driving-point accelerance tends to 1/m
    w = 2 * np.pi * hi
    acc = np.abs(-(w**2) * frf(s, hi, 3)[:, 3])
    assert acc[1] == pytest.approx(1 / s.masses[3], rel=0.01)
    # Dropping mode 1 leaves a mass-like error -phi phi / w^2 above it.
    f = np.array([4.0, 5.5])
    w = 2 * np.pi * f
    terms = modal_frf_terms(s, r, exact=False)
    without_1 = sum(t.evaluate(f) for t in terms[1:])[:, 3, 3]
    error = frf(s, f, 3)[:, 3] - without_1
    phi = r.modes[0].shape_mass_normalized[3]
    assert error.real == pytest.approx(-(phi**2) / w**2, rel=0.1)


def test_background_rigid_body_root_with_k1_zero():
    s = ChainSystem.uniform(4)
    s.stiffness[0] = 0.0
    roots = modal_analysis(s).overdamped_roots
    assert min(roots, key=abs) == pytest.approx(0.0, abs=1e-6)
    second = sorted(roots, key=abs)[1]
    assert second == pytest.approx(-0.50, abs=0.005)  # close to −c1/(m1 + … + mN)


def test_frf_matrix_antiresonances_of_h22():
    s = ChainSystem.uniform(4)
    s.damping[:] = 0.0
    f = np.linspace(0.5, 8.0, 200001)
    h = np.abs(frf(s, f, 1)[:, 1])
    dips = [float(f[i]) for i in find_peaks(-np.log(h))[0]]
    assert dips == pytest.approx([1.97, 4.50, 5.15], abs=0.01)


def test_frf_matrix_worked_example():
    """The Theory's worked example: H11 of the default chain from the closed-form modes, its static shares, the
    first peak, and the dips as the held 3-mass chain's frequencies, each from the figures it quotes too."""
    import scipy.linalg

    s = ChainSystem.uniform(4)
    M, C, K = s.matrices()
    w2, phi = scipy.linalg.eigh(K, M)
    gamma = np.radians([20, 60, 100, 140])
    p1 = 2 / 3 * np.sin(gamma)
    assert np.abs(phi[0]) == pytest.approx(np.abs(p1))
    assert p1 == pytest.approx([0.2280, 0.5774, 0.6565, 0.4285], abs=0.00005)
    assert p1**2 == pytest.approx([0.0520, 0.3333, 0.4310, 0.1836], abs=0.00005) and np.sum(p1**2) == pytest.approx(1)
    assert w2 == pytest.approx([48.25, 400, 938.9, 1412.8], abs=0.05)
    share = p1**2 / w2
    assert np.sum(share) == pytest.approx(1 / 400) == np.linalg.inv(K)[0, 0]
    assert share * 1e3 == pytest.approx([1.078, 0.833, 0.459, 0.130], abs=0.0005)
    assert 100 * share * 400 == pytest.approx([43.1, 33.3, 18.4, 5.2], abs=0.05)
    assert 100 * share[2:].sum() * 400 == pytest.approx(23.6, abs=0.05)
    assert np.allclose(C, K / 200)
    zeta = np.diag(phi.T @ C @ phi) / (2 * np.sqrt(w2))
    assert zeta == pytest.approx(0.0025 * np.sqrt(w2)) and zeta[0] == pytest.approx(0.0174, abs=0.00005)
    assert zeta[2:] == pytest.approx([0.08, 0.09], abs=0.005)
    peak = 0.0520 / (2 * 0.0174 * 48.25)
    assert peak == pytest.approx(0.0310, abs=0.00005) and peak / 0.0025 == pytest.approx(12.4, abs=0.05)
    h11 = abs(np.linalg.inv(K - w2[0] * M + 1j * np.sqrt(w2[0]) * C)[0, 0])
    assert h11 == pytest.approx(0.0311, abs=0.00005)
    held = 40 * np.sin(np.array([1, 3, 5]) * np.pi / 14)
    assert held == pytest.approx([8.90, 24.94, 36.04], abs=0.005)
    assert held / (2 * np.pi) == pytest.approx([1.417, 3.969, 5.736], abs=0.0005)
    assert held**2 == pytest.approx(scipy.linalg.eigvalsh(K[1:, 1:], M[1:, 1:]))
    f = np.linspace(0.5, 7.0, 130001)
    h = np.abs(frf(s, f, 0)[:, 0])
    dips = [float(f[i]) for i in find_peaks(-np.log(h))[0]]
    assert dips == pytest.approx([1.42, 3.99], abs=0.005)  # the third is washed out
    assert len(find_peaks(np.log(h))[0]) == 3  # and so is the fourth peak
    assert abs(-((2 * np.pi * 50) ** 2) * frf(s, np.array([50.0]), 0)[0, 0]) == pytest.approx(1, abs=0.01)


def test_complex_modes_worked_example():
    """The Theory's worked example: one 20 N·s/m damper at the base of the default chain. The approximation's ζ
    from φ₁ᵣ², the exact poles and shapes, the first-order shape and second-order frequency of mode 1, and the
    locked damper's limit, each from the figures it quotes too."""
    from vib_tutorial.core.complex_modes import compare, complexity, free_decay
    from vib_tutorial.core.modal import modal_analysis
    from vib_tutorial.core.model import ChainSystem

    base = ChainSystem(np.ones(4), np.full(4, 400.0), np.array([20.0, 0, 0, 0]))
    r = modal_analysis(base)
    assert modal_analysis(ChainSystem.uniform(4)).is_proportional
    assert r.coupling == pytest.approx(1.0)  # a rank-one C_m
    phi = np.column_stack([m.shape_mass_normalized for m in r.modes])
    w = np.array([m.omega_n for m in r.modes])
    assert w == pytest.approx([6.946, 20.00, 30.64, 37.59], abs=0.005)
    assert w == pytest.approx(40 * np.sin(np.radians([10, 30, 50, 70])))
    phi2 = np.array([0.0520, 0.3333, 0.4310, 0.1836])
    assert phi[0] ** 2 == pytest.approx(phi2, abs=5e-5)
    cm = phi.T @ base.matrices()[1] @ phi
    assert cm == pytest.approx(20 * np.outer(phi[0], phi[0]))
    rows = compare(base, r)
    approx = [0.0748, 0.167, 0.141, 0.0489]
    assert [c.zeta_approx for c in rows] == pytest.approx(approx, rel=0.004)
    assert 10 * phi2 / np.array([6.946, 20.00, 30.64, 37.59]) == pytest.approx(approx, rel=0.004)
    assert [c.zeta_exact for c in rows] == pytest.approx([0.0726, 0.178, 0.168, 0.0225], rel=0.004)
    assert [c.fn_exact for c in rows] == pytest.approx([1.124, 3.434, 4.569, 5.823], abs=0.0005)
    assert [c.fn_hz for c in rows] == pytest.approx([1.105, 3.183, 4.877, 5.982], abs=0.0005)
    # mode 1, first order: a_s1 = iω₁C_s1/(ω₁² − ω_s²)
    a = np.array([1j * w[0] * cm[s, 0] / (w[0] ** 2 - w[s] ** 2) for s in range(1, 4)])
    assert a == pytest.approx([-0.052j, -0.023j, 0.010j], abs=0.0005)
    guess = phi[:, 0] + phi[:, 1:] @ a
    assert complexity(guess).mpd == pytest.approx(2.97, abs=0.005)
    assert rows[0].measures.mpd == pytest.approx(2.98, abs=0.005)
    terms = [cm[0, s] ** 2 / (w[s] ** 2 - w[0] ** 2) for s in range(1, 4)]
    assert terms == pytest.approx([0.0197, 0.0101, 0.0028], abs=0.00005)
    assert w[0] * math.sqrt(1 + sum(terms)) == pytest.approx(7.058, abs=0.0005)
    assert math.sqrt(6.946**2 * (1 + 0.0197 + 0.0101 + 0.0028)) == pytest.approx(7.058, abs=0.0005)
    assert abs(rows[0].exact) == pytest.approx(7.061, abs=0.0005)
    assert rows[0].exact.real == pytest.approx(-cm[0, 0] / 2, rel=0.02)  # the decay rate kept to this order
    # modes 2 to 4: not small; mode 3 the most complex
    assert abs(1j * w[1] * cm[2, 1] / (w[1] ** 2 - w[2] ** 2)) == pytest.approx(0.28, abs=0.005)
    assert rows[2].measures.mpc == pytest.approx(0.20, abs=0.005)
    assert rows[2].measures.mpd == pytest.approx(28.9, abs=0.05)
    assert min(rows, key=lambda c: c.measures.mpc).index == 3
    assert free_decay(base, r, 1).leak[0] == pytest.approx(0.45, abs=0.005)
    # locked at the base: the 3-mass chain above m1, H11's dips
    held = modal_analysis(ChainSystem(np.ones(3), np.full(3, 400.0), np.zeros(3)))
    assert [m.fn_hz for m in held.modes] == pytest.approx([1.42, 3.97, 5.74], abs=0.005)


def test_continuum_worked_example():
    """The Theory's worked example: the guitar string (the page's default) on 12 elements, its speed and
    fundamental, modes 1 and 6 on each mesh, the 11 lumped modes under the cut-off, and its impedance."""
    from vib_tutorial.core.continuum import PRESETS, FEModel, Scheme, errors, flexible_hz

    p = PRESETS[0]
    c = p.continuum
    assert p.elements == 12 and (c.length, c.tension, c.line_density) == (0.648, 73.0, 6.40e-3)
    assert math.sqrt(73 / 0.0064) == pytest.approx(106.8, abs=0.05) and c.speed == pytest.approx(106.8, abs=0.05)
    assert c.exact_hz(1)[0] == pytest.approx(82.41, abs=0.005)
    assert 106.8 / (2 * 0.648) == pytest.approx(82.41, abs=0.005)
    h = c.length / 12
    assert h == pytest.approx(0.054) and math.pi / 12 == pytest.approx(0.262, abs=0.0005)
    assert (math.pi / 12) ** 2 / 24 == pytest.approx(0.0029, abs=0.00005)
    f = {s: flexible_hz(FEModel(c, 12, s)) for s in Scheme}
    err = {s: errors(FEModel(c, 12, s))[1] for s in Scheme}
    assert f[Scheme.LUMPED][0] == pytest.approx(82.17, abs=0.005)
    assert f[Scheme.CONSISTENT][0] == pytest.approx(82.64, abs=0.005)
    assert f[Scheme.MIXED][0] == pytest.approx(82.41, abs=0.005) and abs(err[Scheme.MIXED][0]) < 0.000015
    assert 100 * err[Scheme.LUMPED][0] == pytest.approx(-0.29, abs=0.005)
    assert 100 * err[Scheme.CONSISTENT][0] == pytest.approx(0.29, abs=0.005)
    assert 6 * 82.407 == pytest.approx(494.4, abs=0.05)
    assert math.sin(math.pi / 4) / (math.pi / 4) == pytest.approx(0.900, abs=0.0005)
    assert 494.4 * 0.900 == pytest.approx(445, abs=0.5) and f[Scheme.LUMPED][5] == pytest.approx(445, abs=0.5)
    assert 100 * err[Scheme.LUMPED][5] == pytest.approx(-10.0, abs=0.05)
    assert 100 * err[Scheme.CONSISTENT][5] == pytest.approx(10.3, abs=0.05)
    assert 100 * err[Scheme.MIXED][5] == pytest.approx(-1.4, abs=0.05)
    assert f[Scheme.LUMPED].size == 11
    assert c.speed / (math.pi * h) == pytest.approx(629.5, abs=0.05)
    assert f[Scheme.LUMPED][10] == pytest.approx(624.2, abs=0.05) and c.exact_hz(11)[10] == pytest.approx(
        906.5, abs=0.05
    )
    assert 100 * err[Scheme.LUMPED][10] == pytest.approx(-31, abs=0.5)
    assert math.sqrt(c.tension * c.line_density) == pytest.approx(0.684, abs=0.0005)
    # the notes' general claims: the cut-off error and a cantilever's ratios
    assert 2 / math.pi - 1 == pytest.approx(-0.36, abs=0.005)
    assert [(b / 1.87510) ** 2 for b in (4.69409, 7.85476)] == pytest.approx(
        [6.27, 17.5], abs=0.05
    )  # roots to 5 figures; 17.55 to 3


def test_beams_plates_worked_example():
    """The Theory's worked example: the page's default beam on mode 1, each theory's frequency, the first-order
    Timoshenko formula, the thickness-shear root and the wave speeds; then the trials' plate, from the figures
    it quotes too."""
    from vib_tutorial.core.beams_plates import (
        TWO_PI,
        Beam,
        Plate,
        Theory,
        beam_frequencies,
        beam_omega,
        rect_modes,
        wave_speeds,
    )
    from vib_tutorial.gui.beams_plates import STEEL, BeamPlateTrial

    t = BeamPlateTrial("", "")
    assert (t.length, t.depth, t.mode, t.E, t.rho, t.nu) == (1.0, 0.1, 1, 210.0, 7850.0, 0.3)
    b = Beam(1.0, 0.1, 0.05, STEEL)
    assert b.area == pytest.approx(0.005) and b.inertia == pytest.approx(4.167e-6, abs=5e-10)
    scale = math.sqrt(210e9 * b.inertia / (7850 * b.area))
    assert scale == pytest.approx(149.3, abs=0.05)
    assert math.pi**2 * 149.3 == pytest.approx(1474, abs=0.5)
    f = {th: float(beam_frequencies(b, 1, th)[0]) for th in Theory}
    assert f[Theory.EULER] == pytest.approx(234.5, abs=0.05) and 1474 / TWO_PI == pytest.approx(234.5, abs=0.1)
    assert (math.pi * 0.1) ** 2 / 12 == pytest.approx(0.00822, abs=0.000005)
    assert f[Theory.RAYLEIGH] == pytest.approx(233.6, abs=0.05)
    G = STEEL.G
    assert G == pytest.approx(80.8e9, rel=0.001) and 5 / 6 * G == pytest.approx(67.3e9, rel=0.001)
    assert 210 / 67.3 == pytest.approx(3.12, abs=0.005)
    assert f[Theory.SHEAR] == pytest.approx(231.6, abs=0.05)
    assert (math.pi * 0.1) ** 2 / 24 == pytest.approx(0.00411, abs=0.000005)
    first = 1 - 0.00411 * 4.12
    assert first == pytest.approx(0.9831, abs=0.00005) and first * 234.5 == pytest.approx(230.5, abs=0.05)
    assert 1 - (math.pi * 0.1) ** 2 / 24 * (1 + 210e9 / (5 / 6 * G)) == pytest.approx(
        f[Theory.TIMOSHENKO] / f[Theory.EULER], abs=1e-3
    )
    assert f[Theory.TIMOSHENKO] == pytest.approx(230.7, abs=0.05)
    assert 100 * (f[Theory.TIMOSHENKO] / f[Theory.EULER] - 1) == pytest.approx(-1.64, abs=0.005)
    assert float(beam_omega(b, math.pi, branch=1)) / TWO_PI == pytest.approx(16.4e3, abs=50)
    assert math.sqrt(5 / 6 * G * b.area / (7850 * b.inertia)) / TWO_PI == pytest.approx(16.1e3, abs=50)
    assert wave_speeds(b) == pytest.approx((5172, 2928), abs=0.5)
    # the bending wave's speed levels off at c_s, the thickness-shear branch's at c_b
    k = 1e4
    assert float(beam_omega(b, k)) / k == pytest.approx(2928, rel=1e-3)
    assert float(beam_omega(b, k, branch=1)) / k == pytest.approx(5172, rel=1e-3)

    p = Plate(0.6, 0.4, 0.003, STEEL)
    assert p.rigidity == pytest.approx(519, abs=0.5) and p.surface_mass == pytest.approx(23.55, abs=0.005)
    assert p.scale == pytest.approx(4.696, abs=0.0005)
    assert math.pi**2 * (1 / 0.36 + 1 / 0.16) * 4.696 == pytest.approx(418.4, abs=0.05)
    assert rect_modes(p, 1)[0].freq_hz == pytest.approx(66.6, abs=0.05)
    assert 0.24 / 2 / 4.696 == pytest.approx(0.026, abs=0.0005) and 0.24 / 2 / p.scale == pytest.approx(
        0.026, abs=0.0005
    )
    assert 1 / (0.24 / 2 / p.scale) == pytest.approx(39, abs=0.5)


def test_integration_worked_example():
    """The Theory's worked example: the page's default (1 kg on 400 N/m from 10 mm, ten steps a period), each
    scheme's step from its closed form, the phase after ten periods, and the readout's largest errors."""
    from vib_tutorial.core.integration import Load, Model, Scheme, exact, simulate, spectral
    from vib_tutorial.gui.integration import IntegrationTrial

    t = IntegrationTrial("", "")
    assert (t.mass, t.stiffness, t.damping, t.x0, t.per_period, t.periods, t.force) == (1, 400, 0, 0.01, 10, 10, 0)
    model, load = Model(), Load()
    w = math.sqrt(400)
    assert w == 20 and w / (2 * math.pi) == pytest.approx(3.18, abs=0.005)
    T = 2 * math.pi / w
    h = T / 10
    assert T == pytest.approx(0.314, abs=0.0005) and h == pytest.approx(0.0314, abs=0.00005)
    O = w * h
    assert O == pytest.approx(0.628, abs=0.0005) and O / 2 == pytest.approx(0.314, abs=0.0005)
    cd, aa = 2 * math.asin(O / 2), 2 * math.atan(O / 2)
    assert cd == pytest.approx(0.639, abs=0.0005) and 0.628 / 0.639 == pytest.approx(0.983, abs=0.0005)
    assert aa == pytest.approx(0.609, abs=0.0005)
    sp = {s: spectral(s, O) for s in (Scheme.CENTRAL, Scheme.AVERAGE, Scheme.LINEAR, Scheme.RK4)}
    assert O / cd - 1 == pytest.approx(sp[Scheme.CENTRAL].period_error) and 100 * sp[
        Scheme.CENTRAL
    ].period_error == pytest.approx(-1.7, abs=0.05)
    assert O / aa - 1 == pytest.approx(sp[Scheme.AVERAGE].period_error) and 100 * sp[
        Scheme.AVERAGE
    ].period_error == pytest.approx(3.2, abs=0.05)
    assert 100 * O**2 / 12 == pytest.approx(3.3, abs=0.05)
    assert 100 * sp[Scheme.LINEAR].period_error == pytest.approx(1.6, abs=0.05)
    assert sp[Scheme.RK4].radius == pytest.approx(0.99959, abs=0.000005)
    assert 100 * sp[Scheme.RK4].period_error == pytest.approx(0.11, abs=0.005)
    assert 100 * sp[Scheme.RK4].damping == pytest.approx(0.065, abs=0.0005)
    rk = sum((1j * O) ** k / math.factorial(k) for k in range(5))
    assert abs(rk) ** 2 == pytest.approx(1 - O**6 / 72 + O**8 / 576)
    assert 1 - 8**3 / 72 + 8**4 / 576 == pytest.approx(1.0)  # ρ = 1 at ωΔt = 2√2
    # ten periods: 100 steps of phase
    assert 100 * aa == pytest.approx(60.9, abs=0.05) and 20 * math.pi == pytest.approx(62.8, abs=0.05)
    assert 20 * math.pi - 100 * aa == pytest.approx(1.95, abs=0.005) and 1.95 / (2 * math.pi) == pytest.approx(
        0.31, abs=0.005
    )
    assert 10 * math.cos(1.95) == pytest.approx(-3.7, abs=0.05)
    assert 100 * cd - 20 * math.pi == pytest.approx(1.08, abs=0.005) and 10 * math.cos(1.08) == pytest.approx(
        4.7, abs=0.05
    )
    assert 20 * math.pi - 100 * float(np.angle(rk)) == pytest.approx(0.07, abs=0.005)
    assert abs(rk) ** 100 == pytest.approx(0.960, abs=0.0005)
    steps = 100
    times = np.arange(steps + 1) * h
    reference = exact(model, load, times)
    largest = {}
    for s in sp:
        x = simulate(s, model, load, h, steps, t.alpha)
        largest[s] = 100 * float(np.max(np.abs(x - reference))) / float(np.max(np.abs(reference)))
        assert 1000 * x[-1] == pytest.approx(
            {Scheme.CENTRAL: 4.7, Scheme.AVERAGE: -3.7, Scheme.LINEAR: 5.5, Scheme.RK4: 9.6}[s], abs=0.05
        )
    assert [round(largest[s], 1) for s in sp] == pytest.approx([98.0, 164.5, 91.9, 7.6], abs=0.06)
    assert f"{largest[Scheme.AVERAGE]:.3g}" == "165" and f"{largest[Scheme.CENTRAL]:.3g}" == "98"
    assert f"{largest[Scheme.LINEAR]:.3g}" == "91.9" and f"{largest[Scheme.RK4]:.3g}" == "7.58"


def test_random_vibration_worked_example():
    """The Theory's worked example: the default base shake on the default chain, Miles mode by mode from one
    common factor, the exact RMS, Rice's rates and the largest peak over the Statistics tab's stretch; then the
    test profile's Grms from its segments, each from the figures it quotes too."""
    from vib_tutorial.core.modal import modal_analysis
    from vib_tutorial.core.model import ChainSystem
    from vib_tutorial.core.random_vibration import Input, Output, Quantity, RandomInput, predict, rice
    from vib_tutorial.gui import random_figures
    from vib_tutorial.gui.random_vibration import RandomSetup

    t = RandomSetup()
    assert (t.kind, t.level, t.f_lo, t.f_hi, t.quantity, t.mass) == (
        Input.BASE,
        0.01,
        0.0,
        10.0,
        Quantity.ACCELERATION,
        -1,
    )
    s = ChainSystem.uniform(4)
    inp = RandomInput()
    assert inp.rms == pytest.approx(0.316, abs=0.0005)
    r = modal_analysis(s)
    for m in r.modes:
        assert m.zeta_modal == pytest.approx(m.omega_n / 400)
        assert m.fn_hz / (2 * m.zeta_modal) == pytest.approx(200 / (2 * math.pi))
    assert 200 / (2 * math.pi) == pytest.approx(31.83, abs=0.005)
    factor = math.sqrt(math.pi / 2 * 31.83 * 0.01)
    assert factor == pytest.approx(0.707, abs=0.0005)
    shares = [abs(m.shape_mass_normalized.sum() * m.shape_mass_normalized[3]) for m in r.modes]
    assert shares == pytest.approx([1.241, 0.333, 0.120, 0.028], abs=0.0005)
    p = predict(s, inp, Output(Quantity.ACCELERATION, 3), r)
    miles = [row[3] for row in p.miles]
    assert miles == pytest.approx([0.878, 0.236, 0.085, 0.020], abs=0.0005)
    assert [0.707 * x for x in (1.241, 0.333, 0.120, 0.028)] == pytest.approx([0.878, 0.236, 0.085, 0.020], abs=0.001)
    assert math.sqrt(0.878**2 + 0.236**2 + 0.085**2 + 0.020**2) == pytest.approx(0.913, abs=0.0005)
    assert p.miles_rms == pytest.approx(0.913, abs=0.0005) and p.rms == pytest.approx(0.911, abs=0.0005)
    assert 100 * (p.miles_rms / p.rms - 1) == pytest.approx(0.2, abs=0.05)
    R = rice(p)
    assert (R.nu0, R.nu_peak, R.alpha, R.delta) == pytest.approx((1.40, 2.47, 0.565, 0.43), abs=0.005)
    assert R.nu0 * 60 == pytest.approx(83.8, abs=0.05)
    assert R.peak_factor(60) == pytest.approx(3.17, abs=0.005) and R.clumped_peak_factor(60) == pytest.approx(
        3.05, abs=0.005
    )
    assert R.peak_factor(60) * R.sigma == pytest.approx(2.89, abs=0.005)
    # Davenport's general figures
    from vib_tutorial.core.random_vibration import davenport

    assert [davenport(c) for c in (100, 3000, 270_000)] == pytest.approx([3.2, 4.1, 5.1], abs=0.05)

    # the test profile: segment areas G1 f1/(p + 1) [(f2/f1)^(p+1) − 1]
    def area(f1, g1, f2, g2):
        p = math.log(g2 / g1) / math.log(f2 / f1)
        return g1 * f1 / (p + 1) * ((f2 / f1) ** (p + 1) - 1)

    prof = random_figures.PROFILE
    areas = [area(*prof[i], *prof[i + 1]) for i in range(3)]
    assert areas == pytest.approx([1.5, 16.8, 15.0])
    assert sum(areas) == pytest.approx(33.3) and math.sqrt(33.3) == pytest.approx(5.77, abs=0.005)
    assert 10 * math.log10(2) == pytest.approx(3.01, abs=0.005)
    f = np.geomspace(20, 2000, 200001)
    g = np.interp(np.log(f), np.log([q[0] for q in prof]), np.log([q[1] for q in prof]))
    assert np.trapezoid(np.exp(g), f) == pytest.approx(33.3, rel=1e-4)


def test_fatigue_worked_example():
    """The Theory's worked example: the default stress on the default welded detail, the narrow-band damage
    rate from its rounded figures, Dirlik's and the record's lives; a notch and a mean stress; the Eurocode
    knee and cut-off; the FDS's time compression."""
    from vib_tutorial.core.fatigue import Curve, Stress, dirlik_damage, narrow_band_damage, rates, record
    from vib_tutorial.gui import fatigue_figures
    from vib_tutorial.gui.fatigue import FatigueTrial

    t = FatigueTrial("", "")
    assert (t.stress, t.b, t.s_ref, t.n_ref, t.duration, t.seed) == (Stress(), 3.0, 80.0, 2e6, 600.0, 1)
    s, c = Stress(), Curve()
    assert (s.f1, s.zeta1, s.sigma1, s.sigma2) == (2.0, 0.02, 20.0, 0.0)
    assert c.K == pytest.approx(1.024e12)
    r = rates(s)
    assert (r.nu0, r.nup, r.alpha2) == pytest.approx((1.99, 2.09, 0.95), abs=0.005)
    assert (2 * math.sqrt(2) * 20) ** 3 == pytest.approx(181_000, rel=0.001)
    assert math.gamma(2.5) == pytest.approx(1.329, abs=0.0005)
    nb = narrow_band_damage(s, c)
    assert nb == pytest.approx(4.68e-7, abs=0.005e-7)
    assert 1.99 * 181_000 * 1.329 / 1.024e12 == pytest.approx(4.68e-7, abs=0.01e-7)
    assert 1 / nb == pytest.approx(2.13e6, abs=0.005e6) and 2.13e6 / 86400 == pytest.approx(24.7, abs=0.05)
    assert 1 / nb / 86400 == pytest.approx(24.7, abs=0.05)
    assert 1 / dirlik_damage(s, c) / 86400 == pytest.approx(25.1, abs=0.05)
    rf = record(s, 600.0, 1).damage_rate(c)
    assert 1 / rf / 86400 == pytest.approx(25.1, abs=0.05) and 100 * (nb / rf - 1) == pytest.approx(1.4, abs=0.05)
    # a notch, and a mean stress
    kf = 1 + 0.8 * (2.5 - 1)
    assert kf == pytest.approx(2.2) and kf**3 == pytest.approx(10.6, abs=0.05)
    assert 1 / nb / 86400 / kf**3 == pytest.approx(2.3, abs=0.05)
    assert 100 / (1 - 200 / 600) == pytest.approx(150) and 100 / (1 - (200 / 600) ** 2) == pytest.approx(112.5)
    assert (fatigue_figures.S_AR, fatigue_figures.S_U, fatigue_figures.EXAMPLE) == (150.0, 600.0, (200.0, 100.0))
    # the Eurocode's knee and cut-off from its slopes
    assert (2 / 5) ** (1 / 3) == pytest.approx(0.737, abs=0.0005) and (5 / 100) ** (1 / 5) == pytest.approx(
        0.549, abs=0.0005
    )
    # time compression with b = 8
    assert 10 ** (2 / 8) == pytest.approx(1.78, abs=0.005) and 10 * math.log10(10 ** (2 / 8)) == pytest.approx(2.5)


def test_modal_coupling_absorber_damping_is_shared():
    r = modal_analysis(preset("absorber").system)
    assert [m.zeta_modal for m in r.modes] == pytest.approx([0.012, 0.012], abs=0.0015)


def test_background_friction_trial_loses_5_mm_a_cycle_and_sticks_after_four():
    from vib_tutorial.gui.sim_trials import TRIALS

    note = next(t.note for t in TRIALS if t.name.startswith("Friction"))
    sim = Simulator(ChainSystem([1.0], [400.0], [0.01], [0.5]))
    sim.set_displacement(np.array([0.02]))
    out = []
    while sim.t < 2.0 - 1e-12:
        out.append(sim.advance(2.0 - sim.t)[:3])
    t, x, v = (np.concatenate(a)[:, 0] if a[0].ndim > 1 else np.concatenate(a) for a in zip(*out))
    turns = np.flatnonzero(v[:-1] * v[1:] < 0) + 1
    lost = 0.02 - np.abs(x[turns[1::2]])  # after each whole cycle
    np.testing.assert_allclose(np.diff(np.concatenate([[0.0], lost]))[:3], 0.005, atol=2e-4)
    still = t[np.flatnonzero(v != 0)[-1]]
    assert still == pytest.approx(1.3, abs=0.05) and abs(x[-1]) < 1.25e-3
    assert "= 5 mm" in note and "four cycles (1.3 s)" in note


def test_background_rayleigh_quotient_and_effective_masses():
    s = ChainSystem.uniform(4)
    M, _, K = s.matrices()
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.sqrt(v @ K @ v / (v @ M @ v)) / (2 * np.pi) == pytest.approx(1.162, abs=5e-4)
    assert modal_analysis(s).modes[0].omega_n / (2 * np.pi) == pytest.approx(1.105, abs=5e-4)
    gamma2 = [(m.shape_mass_normalized @ M @ np.ones(4)) ** 2 for m in modal_analysis(s).modes]
    np.testing.assert_allclose(gamma2, [3.57, 0.33, 0.08, 0.01], atol=0.006)


def test_background_worked_example():
    """The Simulation notes' worked example: the default chain's modes in closed form, from the rounded figures."""
    s = ChainSystem.uniform(4)
    M, C, K = s.matrices()
    np.testing.assert_allclose(M, np.eye(4))
    np.testing.assert_allclose(np.diag(K), [800, 800, 800, 400])
    np.testing.assert_allclose(np.diag(K, 1), [-400, -400, -400])
    np.testing.assert_allclose(C, K / 200)
    r = modal_analysis(s)
    # the closed form: γ_r = (2r − 1)π/(2N + 1), ω_r = 2√(k/m) sin(γ_r/2), φ_r,i = 2 sin(iγ_r)/√((2N + 1)m)
    gammas = [(2 * k - 1) * math.pi / 9 for k in range(1, 5)]
    assert np.degrees(gammas) == pytest.approx([20, 60, 100, 140])
    w = [40 * math.sin(g / 2) for g in gammas]
    assert w == pytest.approx([6.946, 20.00, 30.64, 37.59], abs=0.005)
    assert [m.omega_n for m in r.modes] == pytest.approx(w)
    assert np.array([6.946, 20.00, 30.64, 37.59]) / (2 * math.pi) == pytest.approx(
        [1.105, 3.183, 4.877, 5.982], abs=0.001
    )
    assert [m.fn_hz for m in r.modes] == pytest.approx([1.105, 3.183, 4.877, 5.982], abs=0.0005)
    for k, (g, m) in enumerate(zip(gammas, r.modes)):
        shape = 2 * np.sin(np.arange(1, 5) * g) / 3
        assert np.abs(m.shape_mass_normalized) == pytest.approx(np.abs(shape)), k
    assert 2 / 3 * np.sin(np.radians(20 * np.arange(1, 5))) == pytest.approx([0.228, 0.429, 0.577, 0.657], abs=5e-4)
    # damping: ζ_r = βω_r/2, exact; the damped frequencies
    zetas = 0.0025 * np.array([6.946, 20.00, 30.64, 37.59])
    assert zetas == pytest.approx([0.0174, 0.0500, 0.0766, 0.0940], abs=5e-5)
    assert [m.zeta_modal for m in r.modes] == pytest.approx([m.damped.zeta for m in r.modes])
    assert [m.zeta_modal for m in r.modes] == pytest.approx([0.0174, 0.0500, 0.0766, 0.0940], abs=5e-5)
    fd = np.array([1.105, 3.183, 4.877, 5.982]) * np.sqrt(1 - np.array([0.0174, 0.0500, 0.0766, 0.0940]) ** 2)
    assert fd == pytest.approx([1.105, 3.179, 4.862, 5.956], abs=0.001)
    assert [m.damped.fd_hz for m in r.modes] == pytest.approx([1.105, 3.179, 4.862, 5.956], abs=0.0005)
    # the straight-line guess
    v = np.arange(1.0, 5.0)
    assert v @ K @ v == 1600 and v @ M @ v == 30
    assert math.sqrt(1600 / 30) == pytest.approx(7.303, abs=5e-4)
    assert 7.303 / (2 * math.pi) == pytest.approx(1.162, abs=5e-4)
    assert 1.162 / 1.105 - 1 == pytest.approx(0.05, abs=0.005)
    assert 6.946**2 == pytest.approx(48.25, abs=0.005) and 53.33 / 48.25 - 1 == pytest.approx(0.105, abs=0.0005)
    # the ground: mode 1's participation and effective mass
    assert 0.228 + 0.429 + 0.577 + 0.657 == pytest.approx(1.890, abs=0.0015)
    assert 1.890**2 == pytest.approx(3.57, abs=0.005) and 3.57 / 4 == pytest.approx(0.89, abs=0.005)
    # the tuned mass damper at μ = 5 %
    assert math.sqrt(1 + 2 / 0.05) == pytest.approx(math.sqrt(41)) and math.sqrt(41) == pytest.approx(6.4, abs=0.05)


def test_frf_and_coupling_trials():
    from vib_tutorial.core.modal_coupling import coupled_model as model
    from vib_tutorial.gui.coupling_notes import TRIALS as COUPLING

    s = ChainSystem.uniform(4)
    _, _, K = s.matrices()
    phi = np.array([m.shape_mass_normalized for m in modal_analysis(s).modes]).T
    w2 = np.array([m.omega_n**2 for m in modal_analysis(s).modes])
    share = phi[0] ** 2 / w2 / np.linalg.inv(K)[0, 0]  # m1's static compliance, mode by mode
    assert share[2:].sum() == pytest.approx(0.24, abs=0.005)
    assert (phi[3, 0] ** 2 / w2[0]) / np.linalg.inv(K)[3, 3] == pytest.approx(0.89, abs=0.005)
    assert abs(phi[2, 1]) < 1e-12  # m3 is a node of mode 2
    errors = {}
    for t in COUPLING:
        system = t.system()
        comparisons = compare_coupled(
            model(system, t.split, t.mode_a, t.mode_b, t.a_mass, True), modal_analysis(system)
        )
        errors[t.name] = [100 * c.error for c in comparisons]
    np.testing.assert_allclose(errors["Two equal halves: pushed far apart"][1], 4.9, atol=0.05)
    np.testing.assert_allclose(errors["A light half, tuned: splitting"], [0.2, 0.3], atol=0.05)
    np.testing.assert_allclose(errors["The wrong mass for A"], [2.9, -2.2], atol=0.06)
    assert errors["A mode left out (8 masses)"][0] == pytest.approx(-14, abs=0.5)


# --------------------------------------------------------------------- the One mass and Signals pages' trials
def test_single_oscillator_trials():
    import math

    from vib_tutorial.core.sdof import Oscillator, receptance

    osc = Oscillator()
    assert osc.natural_hz == pytest.approx(3.18, abs=0.005) and osc.zeta == pytest.approx(0.05)
    assert math.exp(osc.log_decrement) == pytest.approx(1.37, abs=0.005)
    assert osc.log_decrement == pytest.approx(0.315, abs=0.0005) and osc.time_constant == pytest.approx(1.0)
    slow = Oscillator(1, 400, 80).poles()
    assert sorted(slow.real) == pytest.approx([-74.6, -5.4], abs=0.05)
    two = Oscillator(1, 400, 80).poles().real.max()  # ζ = 2
    assert -two / osc.omega_n == pytest.approx(0.27, abs=0.005) and -1 / two == pytest.approx(0.19, abs=0.005)
    assert 1 / 20 == pytest.approx(0.05)  # critical: a double root at −ω_n
    assert (1 / 400) / (2 * 0.05) == pytest.approx(0.025)  # 25 mm at resonance, Q = 10
    lo, hi = osc.half_power_hz()
    assert (lo, hi) == pytest.approx((3.01, 3.33), abs=0.005)
    beat = osc.natural_hz - 3.0
    assert beat == pytest.approx(0.18, abs=0.005) and 1 / beat == pytest.approx(5.5, abs=0.05)
    undamped = Oscillator(1, 400, 0)
    assert abs(receptance(undamped, np.array([3.0]))[0]) == pytest.approx(0.022, abs=0.0005)
    assert 1 / (2 * undamped.omega_n) == pytest.approx(0.025)  # F/2mω_n, m/s
    assert (1e-4 / 1) / (2 * 0.05) == pytest.approx(1e-3)
    assert 1 / (2 * osc.omega_n) == pytest.approx(0.025)  # the Nyquist circle's diameter 1/cω_n, m/N


def test_single_oscillator_worked_example_and_springs():
    """The figures Single oscillator's worked example and its springs quote, on the page's defaults."""
    from vib_tutorial.core.sdof import Oscillator, receptance

    osc = Oscillator()
    assert (osc.mass, osc.stiffness, osc.damping) == (1.0, 400.0, 2.0)
    assert osc.omega_n == pytest.approx(20.0) and osc.critical == pytest.approx(40.0)
    assert osc.omega_d == pytest.approx(19.97, abs=0.005)
    assert 1 - osc.omega_d / osc.omega_n == pytest.approx(0.0013, abs=0.00005)
    assert osc.log_decrement == pytest.approx(0.315, abs=0.0005)
    omega = 2 * math.pi * 3.0
    r = omega / osc.omega_n
    assert omega == pytest.approx(18.85, abs=0.005) and r == pytest.approx(0.942, abs=0.0005)
    assert 1 - r**2 == pytest.approx(0.1117, abs=0.00005) and 2 * osc.zeta * r == pytest.approx(0.0942, abs=0.00005)
    assert math.hypot(1 - r**2, 2 * osc.zeta * r) == pytest.approx(0.1462, abs=0.00005)
    assert 2.5 / 0.1462 == pytest.approx(17.1, abs=0.05)  # mm, from the rounded numbers
    h = receptance(osc, np.array([3.0]))[0]
    assert abs(h) * 1e3 == pytest.approx(17.1, abs=0.05)  # mm, for 1 N
    assert abs(h) * osc.stiffness == pytest.approx(6.84, abs=0.005)
    assert -math.degrees(math.atan2(h.imag, h.real)) == pytest.approx(40.1, abs=0.05)
    assert math.degrees(math.atan2(0.0942, 0.1117)) == pytest.approx(40.1, abs=0.05)  # from the rounded numbers too
    assert 2 * osc.zeta * osc.natural_hz == pytest.approx(0.32, abs=0.005)
    # Two of its springs side by side, and end to end.
    assert Oscillator(1, 800, 0).natural_hz == pytest.approx(4.50, abs=0.005)
    assert Oscillator(1, 200, 0).natural_hz == pytest.approx(2.25, abs=0.005)


def test_transient_trials():
    from vib_tutorial.core.sdof import Oscillator
    from vib_tutorial.core.transient import PulseKind, peak_ratio, step_metrics

    osc = Oscillator()
    m = step_metrics(osc)
    assert m.overshoot == pytest.approx(0.85, abs=0.005) and 2.5 * (1 + m.overshoot) == pytest.approx(4.64, abs=0.005)
    assert m.peak_time == pytest.approx(0.157, abs=0.0005) and m.settling == pytest.approx(3.8, abs=0.05)
    assert step_metrics(Oscillator(1, 400, 40)).settling == pytest.approx(0.29, abs=0.005)
    assert 2 * np.sin(np.pi / 20) == pytest.approx(0.31, abs=0.005)
    r = 0.26 / osc.period
    assert peak_ratio(0.0, PulseKind.HALF_SINE, np.array([r]))[0] == pytest.approx(1.77, abs=0.005)
    assert peak_ratio(0.05, PulseKind.HALF_SINE, np.array([r]))[0] == pytest.approx(1.65, abs=0.005)
    r = 0.864 / osc.period
    assert r == pytest.approx(2.75, abs=0.005)
    assert peak_ratio(0.0, PulseKind.RAMP, np.array([r]))[0] - 1 == pytest.approx(0.08, abs=0.005)
    assert peak_ratio(0.05, PulseKind.RAMP, np.array([r]))[0] - 1 == pytest.approx(0.06, abs=0.005)


def test_transient_worked_example_and_pulses():
    """The figures Transient response's worked example quotes, on the page's defaults (a 1 N step), from the
    rounded numbers it shows too; and its pulses' closed forms."""
    from vib_tutorial.core.sdof import Oscillator, impulse_response, step_response
    from vib_tutorial.core.transient import Pulse, PulseKind, convolution, peak_ratio, pulse_response, step_metrics

    osc, pulse = Oscillator(), Pulse()
    assert (osc.mass, osc.stiffness, osc.damping) == (1.0, 400.0, 2.0)
    assert (pulse.kind, pulse.amplitude) == (PulseKind.STEP, 1.0)
    zw, wd = osc.zeta * osc.omega_n, osc.omega_d
    assert osc.natural_hz == pytest.approx(3.18, abs=0.005) and osc.period == pytest.approx(0.314, abs=0.0005)
    assert osc.zeta == pytest.approx(0.05) and zw == pytest.approx(1.0) and wd == pytest.approx(19.97, abs=0.005)
    assert zw / wd == pytest.approx(0.0501, abs=0.00005)
    # the partial fractions: (1/400)(1/s − (s + 2)/((s + 1)² + ω_d²)) is 1/(s(s² + 2s + 400))
    for s in (0.3 + 2j, 5.0, -1 + 7j):
        assert (1 / s - (s + 2) / ((s + 1) ** 2 + wd**2)) / 400 == pytest.approx(1 / (s * (s * s + 2 * s + 400)))
    t = np.linspace(0, 3, 301)
    closed = 2.5e-3 * (1 - np.exp(-t) * (np.cos(wd * t) + zw / wd * np.sin(wd * t)))
    np.testing.assert_allclose(step_response(osc, t), closed, atol=1e-12)
    # the overshoot
    m = step_metrics(osc)
    assert 0.05 * math.pi / math.sqrt(1 - 0.05**2) == pytest.approx(0.1573, abs=0.00005)
    assert math.exp(-0.1573) == pytest.approx(0.854, abs=0.0005) and m.overshoot == pytest.approx(0.854, abs=0.0005)
    assert 2.5 * 1.854 == pytest.approx(4.64, abs=0.005) and 2.5 * (1 + m.overshoot) == pytest.approx(4.64, abs=0.005)
    assert math.pi / 19.97 == pytest.approx(0.157, abs=0.0005) and m.peak_time == pytest.approx(0.157, abs=0.0005)
    assert m.settling_estimate == pytest.approx(4.0) and m.settling == pytest.approx(3.8, abs=0.05)
    assert math.log(50) == pytest.approx(3.9, abs=0.05)  # the 2 % rule: ln 50, rounded to 4
    # at 0.3 s, where the convolution is drawn
    assert wd * 0.3 == pytest.approx(5.9925, abs=0.00005)
    assert math.cos(5.9925) == pytest.approx(0.95805, abs=0.000005)
    assert math.sin(5.9925) == pytest.approx(-0.28661, abs=0.000005)
    assert math.exp(-0.3) == pytest.approx(0.74082, abs=0.000005)
    bracket = 1 - 0.74082 * (0.95805 - 0.0501 * 0.28661)
    assert bracket == pytest.approx(0.3009, abs=0.00005) and 2.5 * bracket == pytest.approx(0.752, abs=0.0005)
    x = float(pulse_response(osc, pulse, np.array([0.0, 0.3]))[-1])
    assert x * 1e3 == pytest.approx(0.752, abs=0.0005)
    assert convolution(osc, pulse, 0.3)[3] * 1e3 == pytest.approx(0.752, abs=0.0005)
    # h and H
    assert osc.time_constant == pytest.approx(1.0)
    assert 2 * osc.zeta * osc.natural_hz == pytest.approx(0.318, abs=0.0005)
    assert 1 / (osc.damping * osc.omega_n) == pytest.approx(0.025)
    # the impulse trial's first swing, 4.6 mm, a little under I/mω_n = 5 mm
    t = np.linspace(0, 0.1, 2001)
    assert 0.1 * impulse_response(osc, t).max() * 1e3 == pytest.approx(4.6, abs=0.05)
    assert 0.1 / osc.omega_n == pytest.approx(0.005)
    # the residual swing of an undamped pulse is |F(ω_n)|/mω_n: a rectangle's (F/k) 2|sin πr|
    undamped = Oscillator(1, 400, 0)
    for r in (0.3, 0.7, 1.0, 1.4):
        T = r * undamped.period
        tau = np.linspace(0, T, 20001)
        y = np.exp(-1j * undamped.omega_n * tau)
        spectrum = abs(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(tau)))  # F = 1 N
        after = undamped.period * np.linspace(r + 0.01, r + 1.01, 2001)
        swing = np.abs(pulse_response(undamped, Pulse(PulseKind.RECTANGLE, 1.0, T), np.concatenate([[0], after])))
        assert swing[1:].max() == pytest.approx(spectrum / undamped.omega_n, rel=1e-3, abs=1e-7)
        assert spectrum / undamped.omega_n == pytest.approx(2 * abs(math.sin(math.pi * r)) / 400, abs=1e-7)
    # the half-sine's worst: 1.77 times static near T = 0.8 T_n
    ratios = np.linspace(0.7, 0.9, 201)
    peaks = peak_ratio(0.0, PulseKind.HALF_SINE, ratios)
    assert peaks.max() == pytest.approx(1.77, abs=0.005) and ratios[peaks.argmax()] == pytest.approx(0.8, abs=0.03)


def test_damping_models_trials():
    from vib_tutorial.core.damping_models import Matched, Model, coulomb_stop

    m = Matched()
    wn = m.omega_n
    assert m.energy_per_cycle(Model.VISCOUS, np.array([wn]))[0] == pytest.approx(0.0126, abs=5e-5)
    assert m.friction == pytest.approx(0.31, abs=0.005)
    assert m.energy_per_cycle(Model.VISCOUS, np.array([2 * wn]))[0] == pytest.approx(0.0251, abs=5e-5)
    assert m.energy_per_cycle(Model.ZENER, np.array([2 * wn]))[0] == pytest.approx(0.0107, abs=5e-5)
    assert 4 * m.friction / np.pi == pytest.approx(0.4, abs=0.005)
    assert 4 * m.friction / m.stiffness == pytest.approx(0.0031, abs=5e-5)
    half_cycles, where = coulomb_stop(m.stiffness, m.friction, 0.01)
    assert half_cycles == 6 and abs(where) == pytest.approx(0.00058, abs=5e-6)
    k0, k1, _ = m.zener()
    assert (k0, k0 + k1) == pytest.approx((364, 444), abs=0.5)


def test_damping_models_worked_example():
    """The figures Damping models' worked example quotes, on the page's defaults, from the rounded numbers it
    shows too; and its equivalent viscous damping and loss factor formulas."""
    from vib_tutorial.core.damping_models import Matched, Model, coulomb_decay, coulomb_stop
    from vib_tutorial.core.sdof import Oscillator

    m = Matched()
    assert (m.mass, m.stiffness, m.zeta, m.amplitude, m.relax) == (1.0, 400.0, 0.05, 0.01, 1.0)
    wn = m.omega_n
    assert wn == pytest.approx(20.0) and m.natural_hz == pytest.approx(3.18, abs=0.005) and m.c == pytest.approx(2.0)
    U = 0.5 * m.stiffness * m.amplitude**2
    assert U == pytest.approx(0.020) and m.eta == pytest.approx(0.1)
    w = np.array([wn])
    for model in Model:
        assert m.energy_per_cycle(model, w)[0] * 1e3 == pytest.approx(12.57, abs=0.005)
        K = m.dynamic_stiffness(model, w)[0]
        assert m.energy_per_cycle(model, w)[0] / (2 * math.pi * 0.5 * K.real * m.amplitude**2) == pytest.approx(
            K.imag / K.real
        )  # η = W_d/2πU = K''/K'
    assert 2 * math.pi * 0.1 * 20 == pytest.approx(12.57, abs=0.005)  # mJ, from the rounded numbers
    assert math.pi * 2 * 20 * 0.01**2 * 1e3 == pytest.approx(12.57, abs=0.005)
    # friction
    assert m.friction == pytest.approx(0.314, abs=0.0005) and 12.57 / 40 == pytest.approx(0.314, abs=0.0005)
    assert m.friction == pytest.approx(0.3142, abs=0.00005) and m.zener()[2] == pytest.approx(0.04525, abs=5e-6)
    assert abs(coulomb_stop(m.stiffness, m.friction, m.amplitude)[1]) * 1e6 == pytest.approx(575, abs=0.5)
    assert 4 * m.friction / m.stiffness * 1e3 == pytest.approx(3.14, abs=0.005)
    assert m.friction / m.stiffness * 1e3 == pytest.approx(0.785, abs=0.0005)
    turns = [10 - 1.57 * n for n in range(1, 7)]  # from the rounded 1.57 mm a half cycle
    assert turns == pytest.approx([8.43, 6.86, 5.29, 3.72, 2.15, 0.58], abs=0.005)
    halves, stop = coulomb_stop(m.stiffness, m.friction, m.amplitude)
    assert halves == 6 and abs(stop) * 1e3 == pytest.approx(0.58, abs=0.005)
    t = np.linspace(0, 2.5 * 2 * math.pi / wn, 50001)  # its turns, every half period: 8.43, 6.86, … mm
    x = coulomb_decay(m.mass, m.stiffness, m.friction, m.amplitude, t)
    half = np.searchsorted(t, [n * math.pi / wn for n in range(1, 5)])
    assert np.abs(x[half]) * 1e3 == pytest.approx([8.43, 6.86, 5.29, 3.72], abs=0.005)
    delta = Oscillator(1, 400, 2).log_decrement
    assert delta == pytest.approx(0.315, abs=0.0005) and math.exp(-delta) == pytest.approx(0.730, abs=0.0005)
    assert math.exp(-0.315) == pytest.approx(0.730, abs=0.0005)
    # driven by 1 N
    assert abs(m.receptance(Model.HYSTERETIC, np.array([m.natural_hz]), 1.0)[0]) * 1e3 == pytest.approx(25.0)
    f = np.linspace(3.0, 3.3, 300001)
    assert np.abs(m.receptance(Model.VISCOUS, f, 1.0)).max() * 1e3 == pytest.approx(25.03, abs=0.005)
    assert 2.5 / (0.1 * math.sqrt(1 - 0.05**2)) == pytest.approx(25.03, abs=0.005)
    assert np.abs(m.receptance(Model.HYSTERETIC, f, 1.0)).max() == pytest.approx(0.025, rel=1e-6)
    assert 4 * m.friction / math.pi == pytest.approx(0.4)
    # the Zener element
    k0, k1, tau = m.zener()
    a = k1 / k0
    assert a == pytest.approx(0.221, abs=0.0005) and a / (2 * math.sqrt(1 + a)) == pytest.approx(0.1)
    assert 0.221 / (2 * math.sqrt(1.221)) == pytest.approx(0.1, abs=0.0005)  # η_max = k1/2√(k0(k0 + k1))
    assert k1 / (2 * math.sqrt(k0 * (k0 + k1))) == pytest.approx(0.1)
    assert k0 == pytest.approx(363.8, abs=0.05) and k1 == pytest.approx(80.4, abs=0.05)
    assert 400 / (1 + 0.221 / 2.221) == pytest.approx(363.8, abs=0.05)
    assert tau * 1e3 == pytest.approx(45.2, abs=0.05) and 1e3 / (20 * math.sqrt(1.221)) == pytest.approx(45.2, abs=0.05)
    assert m.loss_factor(Model.ZENER, w)[0] == pytest.approx(0.1)
    assert m.loss_factor(Model.ZENER, np.geomspace(0.1, 10, 2001) * wn).max() == pytest.approx(0.1, rel=1e-5)
    # equivalent viscous damping: velocity-squared drag a ẋ|ẋ| takes (8/3) a Ω² X³ a cycle
    drag, omega, X = 0.7, 13.0, 0.02
    th = np.linspace(0, 2 * math.pi, 400001)
    v = X * omega * np.cos(th)
    loss = np.trapezoid(drag * v * np.abs(v) * v, th / omega)  # ∮F dx = ∫F ẋ dt
    assert loss == pytest.approx(8 / 3 * drag * omega**2 * X**3, rel=1e-6)
    assert loss / (math.pi * omega * X**2) == pytest.approx(8 * drag * omega * X / (3 * math.pi), rel=1e-6)


def test_isolation_trials():
    from vib_tutorial.core.isolation import Isolator, Mount, deflection_for

    soft = Isolator()
    X, floor, T = soft.at_speed()
    assert soft.natural_hz == pytest.approx(7.05, abs=0.005)
    assert float(soft.force_at(25.0)) == pytest.approx(247, abs=0.5) and floor == pytest.approx(23, abs=0.5)
    assert T == pytest.approx(0.09, abs=0.005) and 20 * np.log10(T) == pytest.approx(-21, abs=0.5)
    assert X == pytest.approx(109e-6, abs=0.5e-6)
    assert deflection_for(25.0, 0.9) == pytest.approx(4.37e-3, abs=5e-6)
    stiff = Isolator(deflection=3e-4)
    assert stiff.natural_hz == pytest.approx(28.8, abs=0.05) and stiff.at_speed()[2] == pytest.approx(3.9, abs=0.05)
    assert Isolator(zeta=0.3).at_speed()[2] == pytest.approx(0.20, abs=0.005)
    assert (soft.natural_hz / soft.speed_hz) ** 2 == pytest.approx(1 / 12, abs=0.005)
    f = np.linspace(1, 30, 20001)
    relax = Isolator(zeta=0.3, mount=Mount.RELAXATION, rpm=3000)
    plain = Isolator(zeta=0.3, rpm=3000)
    assert relax.transmissibility(f).max() == pytest.approx(2.1, abs=0.05)
    assert plain.transmissibility(f).max() == pytest.approx(2.0, abs=0.05)
    assert (relax.at_speed()[2], plain.at_speed()[2]) == pytest.approx((0.070, 0.088), abs=0.0005)
    fast = (Isolator(zeta=0.3, mount=Mount.RELAXATION, rpm=6000), Isolator(zeta=0.3, rpm=6000))
    assert [i.at_speed()[2] for i in fast] == pytest.approx([0.019, 0.043], abs=0.0005)
    assert Isolator(mount=Mount.TWO_STAGE).at_speed()[2] == pytest.approx(0.038, abs=0.0005)
    assert Isolator(mount=Mount.TWO_STAGE, rpm=3000).at_speed()[2] == pytest.approx(0.002, abs=0.0005)
    assert Isolator(rpm=3000).at_speed()[2] == pytest.approx(0.025, abs=0.0005)
    assert Isolator(block=300).at_speed()[0] == pytest.approx(27e-6, abs=0.5e-6)


def test_isolation_worked_example():
    """The figures Vibration isolation's worked example quotes, on the page's defaults, from the rounded
    numbers it shows too; and the base-motion and hysteretic transmissibilities its notes give."""
    from vib_tutorial.core.isolation import G, Isolator, start_up

    iso = Isolator()
    assert (iso.mass, iso.rpm, iso.unbalance, iso.deflection, iso.zeta) == (100.0, 1500.0, 0.01, 0.005, 0.05)
    assert iso.stiffness == pytest.approx(196_100, abs=50) and 100 * 9.807 / 0.005 == pytest.approx(196_100, abs=50)
    assert iso.natural_hz == pytest.approx(7.048, abs=0.0005) and 15.76 / math.sqrt(5) == pytest.approx(
        7.048, abs=0.0005
    )
    assert 60 * iso.natural_hz == pytest.approx(423, abs=0.5) and iso.damping == pytest.approx(442.9, abs=0.05)
    assert math.sqrt(G / 0.001) / (2 * math.pi) == pytest.approx(15.76, abs=0.005)
    assert 2 * math.pi * 25 == pytest.approx(157.08, abs=0.005)
    F = float(iso.force_at(25.0))
    assert F == pytest.approx(246.7, abs=0.05) and 0.01 * 157.08**2 == pytest.approx(246.7, abs=0.05)
    r = 25 / iso.natural_hz
    assert r == pytest.approx(3.547, abs=0.0005) and 25 / 7.048 == pytest.approx(3.547, abs=0.0005)
    assert 3.547**2 == pytest.approx(12.58, abs=0.005) and 0.1 * 3.547 == pytest.approx(0.3547)
    assert math.sqrt(1 + 0.3547**2) == pytest.approx(1.0610, abs=0.00005)
    assert math.hypot(11.58, 0.3547) == pytest.approx(11.585, abs=0.0005)
    assert 1.0610 / 11.585 == pytest.approx(0.0916, abs=0.00005)
    X, floor, T = iso.at_speed()
    assert T == pytest.approx(0.0916, abs=0.00005) and floor == pytest.approx(22.6, abs=0.05)
    assert 246.7 * 0.0916 == pytest.approx(22.6, abs=0.05)
    assert 100 * (1 - T) == pytest.approx(90.8, abs=0.05) and 20 * math.log10(0.0916) == pytest.approx(-20.8, abs=0.05)
    assert X * 1e6 == pytest.approx(109, abs=0.5) and 100 * 12.58 / 11.585 == pytest.approx(109, abs=0.5)
    assert (iso.natural_hz / 25) ** 2 == pytest.approx(0.0795, abs=0.00005)
    assert (7.048 / 25) ** 2 * 246.7 == pytest.approx(19.6, abs=0.05) and 19.6 / 0.1 == pytest.approx(196)
    s = start_up(iso, 2.0, 1.0)
    assert np.abs(s.floor[s.t <= 2.0]).max() == pytest.approx(111, abs=0.5)
    # the static-deflection examples and 90 % isolation
    for mm, hz in ((1, 15.8), (5, 7.0), (25, 3.2)):
        assert Isolator(deflection=mm * 1e-3).natural_hz == pytest.approx(hz, abs=0.05)
    assert 25 / math.sqrt(11) == pytest.approx(7.5, abs=0.05) and (15.76 / 7.54) ** 2 == pytest.approx(4.4, abs=0.05)
    # base motion: M x'' + c x' + k x = k y + c y' passes |X|/|Y| = T
    k, c, M = iso.stiffness, iso.damping, iso.mass
    for f in (2.0, 7.0, 25.0, 80.0):
        w = 2 * math.pi * f
        assert abs((k + 1j * w * c) / (k - w * w * M + 1j * w * c)) == pytest.approx(
            float(iso.transmissibility(np.array([f]))[0])
        )
    # far above resonance: √(1 + (2ζr)²)/r², 2ζ/r once 2ζr ≫ 1; a hysteretic mount keeps falling as 1/r²
    for r in (30.0, 300.0):
        f = r * iso.natural_hz
        assert float(iso.transmissibility(np.array([f]))[0]) == pytest.approx(
            math.sqrt(1 + (0.1 * r) ** 2) / r**2, rel=2e-3
        )
    assert math.sqrt(1 + 30.0**2) / 300**2 == pytest.approx(0.1 / 300, rel=1e-3)
    eta, r = 0.1, 30.0
    hyst = abs((1 + 1j * eta) / (1 - r * r + 1j * eta))
    assert hyst == pytest.approx(math.sqrt(1 + eta**2) / math.hypot(1 - r * r, eta)) and hyst * r * r == pytest.approx(
        1, abs=0.01
    )
    # the machine's motion tends to m_e e/M
    assert Isolator(rpm=60 * 50 * iso.natural_hz).at_speed()[0] == pytest.approx(0.01 / 100, rel=1e-3)


def test_signals_trials():
    from vib_tutorial.core.signals import (
        Kind,
        Source,
        Window,
        mass_loaded,
        nearest_line_reading,
        sample,
        usable_band,
        window_props,
    )
    from vib_tutorial.core.signals import spectrum as averaged

    def reading(source, f, window, block=1024, averages=1, filtered=True):
        n = block + (averages - 1) * block // 2
        _, x = sample(source, 1024.0, n, filtered)
        return nearest_line_reading(averaged(x, 1024.0, block, window, 0.5), f)[1]

    assert reading(Source(f1=50.0, a2=0.0), 50.0, Window.RECTANGULAR) == pytest.approx(1.0, abs=5e-4)
    pair = Source(f1=50.5, f2=70.0, a2=0.001)
    assert reading(pair, 50.5, Window.RECTANGULAR) == pytest.approx(0.64, abs=0.005)
    assert reading(pair, 70.0, Window.RECTANGULAR) == pytest.approx(0.015, abs=0.0005)
    assert reading(pair, 50.5, Window.HANN) == pytest.approx(0.85, abs=0.005)
    assert reading(pair, 70.0, Window.HANN) == pytest.approx(0.0010, abs=0.00005)
    assert reading(pair, 50.5, Window.FLAT_TOP) == pytest.approx(0.999, abs=0.0005)
    flat = window_props(Window.FLAT_TOP)
    assert flat.main_lobe == 5 and flat.enbw == pytest.approx(3.8, abs=0.05)
    alias = Source(f1=900.0, a2=0.0)
    assert reading(alias, 124.0, Window.HANN, filtered=False) == pytest.approx(1.0, abs=5e-4)
    assert reading(alias, 124.0, Window.HANN) == pytest.approx(0.0015, abs=0.00005)
    assert 1024 / 2.56 == 400
    assert 0.5**2 / 512 == pytest.approx(4.9e-4, abs=5e-6)
    hann = window_props(Window.HANN)
    for block, level in ((4096, 0.019), (256, 0.077)):
        assert 2 * 0.5 * np.sqrt(hann.enbw / block) == pytest.approx(level, abs=0.0005)
    assert 0.05 / 0.019 == pytest.approx(2.6, abs=0.05)
    ringing = Source(Kind.RINGING, f1=50.0, zeta=0.01)
    assert 2 * ringing.zeta * ringing.f1 == pytest.approx(1.0) and 1024 / 4096 == 0.25
    assert usable_band(0.01) == pytest.approx(0.22, abs=0.005) and 0.218 * 30000 == pytest.approx(6.5e3, abs=60)
    assert usable_band(0.7, accelerometer=False) == pytest.approx(1.7, abs=0.05)
    assert usable_band(0.7, accelerometer=False) * 4.5 == pytest.approx(7.6, abs=0.05)
    assert mass_loaded(200.0, 0.5, 0.03) / 200 - 1 == pytest.approx(-0.029, abs=0.0005)


def test_periodic_trials():
    import math

    from vib_tutorial.core.periodic import (
        Forcing,
        Wave,
        periodic_steady,
        response_coefficients,
        series,
        series_error,
    )
    from vib_tutorial.core.sdof import Oscillator

    osc = Oscillator()
    t = np.linspace(0.0, 1.0, 20001)
    sq = Forcing(Wave.SQUARE)
    assert np.max(series(sq.coefficients(5), 1.0, t)) - 1 == pytest.approx(0.19, abs=0.005)
    fine = np.linspace(0.0, 0.01, 20001)
    assert np.max(series(sq.coefficients(401), 1.0, fine)) - 1 == pytest.approx(0.18, abs=0.005)
    third = Forcing(Wave.SQUARE, freq_hz=osc.natural_hz / 3)
    assert 4 / (3 * math.pi) == pytest.approx(0.42, abs=0.005) and 1 / (2 * osc.omega_n) == pytest.approx(0.025)
    X = np.abs(response_coefficients(osc, third, 3))
    assert (X[3], X[1]) == pytest.approx((0.0106, 0.0036), abs=5e-5)

    def swing(fo):
        return float(np.max(periodic_steady(osc, fo, t * fo.period)))

    assert swing(third) == pytest.approx(0.0139, abs=5e-5)
    assert swing(Forcing(Wave.SQUARE, freq_hz=osc.natural_hz / 2)) == pytest.approx(0.0050, abs=5e-5)
    blows = periodic_steady(osc, Forcing(Wave.IMPULSES, 0.1), t)
    assert (blows.min(), blows.max()) == pytest.approx((-0.0045, 0.0052), abs=5e-5)
    assert 0.1 / 400 == pytest.approx(0.00025) and osc.time_constant == pytest.approx(1.0)
    assert 2 / math.pi == pytest.approx(0.64, abs=0.005) and 2 / math.pi / 400 == pytest.approx(0.00159, abs=5e-6)
    tri, sq1 = Forcing(Wave.TRIANGLE), Forcing(Wave.SQUARE)
    assert series_error(osc, tri, 3) == pytest.approx(0.019, abs=0.0005)
    assert series_error(osc, sq1, 3) == pytest.approx(0.057, abs=0.0005)
    assert series_error(osc, tri, 9) == pytest.approx(0.0010, abs=0.00005)
    assert series_error(osc, sq1, 9) == pytest.approx(0.0065, abs=0.00005)


def test_periodic_worked_example_and_series():
    """The figures Periodic forcing's worked example and its notes on the series quote, on the page's defaults."""
    import math

    from scipy.special import sici

    from vib_tutorial.core.periodic import Forcing, Wave, periodic_steady, response_coefficients, series
    from vib_tutorial.core.sdof import Oscillator, receptance

    # The notes: the square wave's fundamental carries 8/π² of its power, the first three harmonics 93.3 %;
    # the series overshoots a jump by 8.95 % of it.
    sq = Forcing(Wave.SQUARE)
    C = np.abs(sq.coefficients(20000))
    assert C[1] ** 2 / 2 == pytest.approx(8 / math.pi**2) and 8 / math.pi**2 == pytest.approx(0.811, abs=0.0005)
    assert np.sum(C[1:] ** 2) / 2 == pytest.approx(1.0, abs=1e-4)  # Parseval: the mean square F²
    assert np.sum(C[1:6] ** 2) / 2 == pytest.approx(0.933, abs=0.0005)
    assert sici(math.pi)[0] / math.pi - 0.5 == pytest.approx(0.0895, abs=0.00005)
    t = np.linspace(0.0, 1.0, 40001)
    assert np.mean(sq.force(t[:-1]) * np.sin(2 * math.pi * t[:-1])) == pytest.approx(2 / math.pi, abs=1e-4)
    assert np.mean(sq.force(t[:-1]) * np.sin(4 * math.pi * t[:-1])) == pytest.approx(0.0, abs=1e-4)

    # The worked example.
    osc, fo = Oscillator(), Forcing()
    assert (osc.mass, osc.stiffness, osc.damping) == (1.0, 400.0, 2.0)
    assert (fo.wave, fo.amplitude, fo.freq_hz) == (Wave.SQUARE, 1.0, 1.0)
    assert osc.natural_hz == pytest.approx(3.18, abs=0.005) and osc.zeta == pytest.approx(0.05)
    C = fo.coefficients(5)
    assert np.all(np.abs(np.abs(C[[1, 3, 5]]) - [1.273, 0.4244, 0.255]) <= [5e-4, 5e-5, 5e-4])
    assert np.all(C[[0, 2, 4]] == 0)
    r = 2 * math.pi * np.array([1, 3, 5]) / osc.omega_n
    assert r == pytest.approx([0.314, 0.942, 1.571], abs=5e-4)
    assert 1 - r[0] ** 2 == pytest.approx(0.9013, abs=5e-5) and 2 * osc.zeta * r[0] == pytest.approx(0.0314, abs=5e-5)
    assert 1 - r[2] ** 2 == pytest.approx(-1.4674, abs=5e-5) and 2 * osc.zeta * r[2] == pytest.approx(0.1571, abs=5e-5)
    assert math.hypot(0.9013, 0.0314) == pytest.approx(0.902, abs=5e-4)
    assert math.hypot(1.4674, 0.1571) == pytest.approx(1.476, abs=5e-4)
    H = np.abs(receptance(osc, np.array([1.0, 3.0, 5.0]))) * 1e3  # mm/N
    assert H == pytest.approx([2.77, 17.10, 1.69], abs=0.005)
    assert (2.5 / 0.902, 2.5 / 1.476) == pytest.approx((2.77, 1.69), abs=0.005)  # from the rounded figures
    X = response_coefficients(osc, fo, 5)
    amps = np.abs(X[[1, 3, 5]]) * 1e3
    assert np.all(np.abs(amps - [3.53, 7.26, 0.431]) <= [0.005, 0.005, 0.0005])
    rounded = (1.273 * 2.77, 0.4244 * 17.10, 0.255 * 1.69)
    assert np.all(np.abs(np.array(rounded) - [3.53, 7.26, 0.431]) <= [0.005, 0.005, 0.0005])
    lags = -np.degrees(np.angle(X[[1, 3, 5]] / C[[1, 3, 5]]))
    assert lags == pytest.approx([2.0, 40.1, 173.9], abs=0.05)
    assert amps[1] / amps[0] == pytest.approx(2, abs=0.1)  # "about twice its fundamental"
    for parts in (amps, np.array([3.53, 7.26, 0.431])):
        shares = 100 * parts**2 / np.sum(parts**2)
        assert shares == pytest.approx([19.1, 80.6, 0.3], abs=0.05)
        assert math.sqrt(np.sum(parts**2) / 2) == pytest.approx(5.72, abs=0.005)
        assert np.sum(parts) == pytest.approx(11.2, abs=0.05)
    t = np.linspace(0.0, 1.0, 4000, endpoint=False)
    exact = periodic_steady(osc, fo, t)
    assert np.sqrt(np.mean(exact**2)) * 1e3 == pytest.approx(5.72, abs=0.005)
    assert (exact.min() * 1e3, exact.max() * 1e3) == pytest.approx((-10.0, 10.0), abs=0.05)
    err = np.max(np.abs(series(X, 1.0, t) - exact)) / np.max(np.abs(exact))
    assert 100 * err == pytest.approx(2.02, abs=0.005)


def test_energy_methods_trials():
    from vib_tutorial.core.energy_methods import (
        Basis,
        Shape,
        contaminated,
        dunkerley,
        dunkerley_terms,
        exact,
        holzer,
        holzer_roots,
        iterate,
        rayleigh,
        ritz_table,
        trial_shape,
    )
    from vib_tutorial.gui.energy_methods import chain

    def hz(w2):
        return np.sqrt(w2) / (2 * np.pi)

    s = chain()
    f = exact(s)[0] / (2 * np.pi)
    assert f == pytest.approx([1.105, 3.183, 4.877, 5.982], abs=0.0005)

    def ray(shape, eps=0.0, passes=0, system=s):
        return hz(rayleigh(system, iterate(system, contaminated(system, trial_shape(system, shape), eps), passes)))

    assert ray(Shape.STATIC) == pytest.approx(1.112, abs=0.0005)
    assert ray(Shape.STATIC) / f[0] - 1 == pytest.approx(0.0055, abs=0.00005)
    assert ray(Shape.LINEAR) / f[0] - 1 == pytest.approx(0.051, abs=0.0005)
    assert ray(Shape.LINEAR, passes=1) / f[0] - 1 == pytest.approx(0.00063, abs=0.000005)
    assert ray(Shape.SINE) == pytest.approx(f[0], rel=1e-12)
    assert ray(Shape.SINE, 0.1) / f[0] - 1 == pytest.approx(0.046, abs=0.0005)
    assert ray(Shape.SINE, 0.2) / f[0] - 1 == pytest.approx(0.166, abs=0.0005)
    assert ray(Shape.TIP) == pytest.approx(3.18, abs=0.005) and ray(Shape.TIP) / f[0] - 1 == pytest.approx(
        1.88, abs=0.005
    )
    assert ray(Shape.TIP, passes=1) / f[0] - 1 == pytest.approx(0.051, abs=0.0005)
    assert ray(Shape.TIP, passes=2) / f[0] - 1 == pytest.approx(0.00063, abs=0.000005)
    assert (f[0] / f[1]) ** 2 == pytest.approx(0.12, abs=0.005)
    assert hz(dunkerley_terms(s)) == pytest.approx([3.18, 2.25, 1.84, 1.59], abs=0.005)
    assert hz(dunkerley(s)) == pytest.approx(1.007, abs=0.0005) and hz(dunkerley(s)) / f[0] - 1 == pytest.approx(
        -0.089, abs=0.0005
    )
    tip = chain(masses=(1.0, 1.0, 1.0, 5.0))
    ft = exact(tip)[0] / (2 * np.pi)
    assert hz(dunkerley_terms(tip))[-1] == pytest.approx(0.71, abs=0.005)
    assert (hz(dunkerley(tip)), ft[0]) == pytest.approx((0.624, 0.653), abs=0.0005)
    assert hz(dunkerley(tip)) / ft[0] - 1 == pytest.approx(-0.044, abs=0.0005)
    assert ray(Shape.STATIC, system=tip) / ft[0] - 1 == pytest.approx(0.0025, abs=0.00005)
    table = ritz_table(s, Basis.KRYLOV)
    assert table[1, 0] == pytest.approx(f[0], abs=0.00005)
    assert table[1, 1] == pytest.approx(3.30, abs=0.005) and table[1, 1] / f[1] - 1 == pytest.approx(0.036, abs=0.0005)
    assert (table[2, 1], table[2, 2]) == pytest.approx((3.184, 5.02), abs=0.005)
    stiff = chain(stiffness=(4000.0, 400.0, 400.0, 400.0))
    fs = exact(stiff)[0] / (2 * np.pi)
    powers = ritz_table(stiff, Basis.POWERS)
    assert (powers[1, 0], fs[0]) == pytest.approx((1.63, 1.38), abs=0.005)
    assert powers[1, 0] / fs[0] - 1 == pytest.approx(0.18, abs=0.005)
    assert (powers[1, 1], fs[1]) == pytest.approx((6.04, 3.89), abs=0.005)
    assert ritz_table(stiff, Basis.KRYLOV)[0, 0] / fs[0] - 1 < 0.008
    assert holzer(s, 2 * np.pi * 1.0)[-1].spring == pytest.approx(61, abs=0.5)
    assert holzer_roots(s, 7.0) == pytest.approx(f, abs=0.0005)


def test_energy_methods_worked_example():
    """The Theory's worked example on the page's opening chain: Rayleigh from the sag, Dunkerley, two Krylov
    shapes and a line of Holzer's table, each number from the solver and from the figures it quotes."""
    from vib_tutorial.core.energy_methods import (
        Basis,
        dunkerley,
        dunkerley_terms,
        exact,
        holzer,
        rayleigh,
        ritz_table,
        static_deflection,
    )
    from vib_tutorial.gui.energy_methods import chain

    s = chain()
    M, _, K = s.matrices()
    w, _ = exact(s)
    assert w[0] ** 2 == pytest.approx(48.246, abs=0.0005)
    v = static_deflection(s, s.masses)
    assert 400 * v == pytest.approx([4, 7, 9, 10])
    assert np.diag(np.linalg.inv(K)) * 400 == pytest.approx([1, 2, 3, 4])
    assert 12000 / 246 == pytest.approx(48.78, abs=0.005) == rayleigh(s, v)
    assert math.sqrt(48.78) == pytest.approx(6.984, abs=0.0005)
    assert math.sqrt(rayleigh(s, v)) / (2 * math.pi) == pytest.approx(1.112, abs=0.0005)
    assert math.sqrt(rayleigh(s, v)) / w[0] - 1 == pytest.approx(0.0055, abs=0.00005)
    assert np.sqrt(dunkerley_terms(s)) == pytest.approx([20, 14.14, 11.55, 10], abs=0.005)
    assert dunkerley(s) == pytest.approx(40) and math.sqrt(40) == pytest.approx(6.325, abs=0.0005)
    assert math.sqrt(40) / (2 * math.pi) == pytest.approx(1.007, abs=0.0005)
    assert np.sum(1 / w**2) == pytest.approx(0.025) and 1 / w[0] ** 2 == pytest.approx(0.0207, abs=0.00005)
    # two Krylov shapes, as the notes scale them
    w1 = np.array([4.0, 7, 9, 10])
    w2 = 400 * np.linalg.solve(K, M @ w1)
    assert w2 == pytest.approx([30, 56, 75, 85])
    W = np.column_stack([w1, w2])
    assert W.T @ M @ W == pytest.approx(np.array([[246, 2037], [2037, 16886]]))
    assert W.T @ K @ W == pytest.approx(400 * np.array([[30, 246], [246, 2037]]))
    a, b, c = 246 * 16886 - 2037**2, 400 * (30 * 16886 + 2037 * 246 - 2 * 246 * 2037), 400**2 * (30 * 2037 - 246**2)
    assert (a, b / 1e6, c / 1e7) == pytest.approx((4587, 2.1912, 9.504), abs=0.00005)
    roots = np.sort(np.roots([a, -b, c]))
    assert roots == pytest.approx([48.246, 429.45], abs=0.005)
    ritz = ritz_table(s, Basis.KRYLOV)[1, :2]
    assert ritz**2 * (2 * math.pi) ** 2 == pytest.approx(roots, rel=1e-9)
    assert ritz == pytest.approx([1.105, 3.298], abs=0.0005) and ritz[0] / w[0] * 2 * math.pi - 1 < 1e-5
    assert ritz[1] / (w[1] / (2 * math.pi)) - 1 == pytest.approx(0.036, abs=0.0005)
    # Holzer at 1 Hz
    assert (2 * math.pi) ** 2 == pytest.approx(39.48, abs=0.005)
    rows = holzer(s, 2 * math.pi)
    assert [r.inertia for r in rows] == pytest.approx([39.5, 75.1, 103.2, 121.2], abs=0.05)
    assert [r.spring for r in rows] == pytest.approx([360.5, 285.5, 182.2, 61.0], abs=0.05)
    assert [r.x for r in rows][1:] == pytest.approx([1.901, 2.615, 3.071], abs=0.0005)


def test_integration_trials():
    import math

    from vib_tutorial.core.integration import (
        Load,
        Model,
        Scheme,
        exact,
        simulate,
        spectral,
        stable_limit,
    )

    def at(per_period, scheme, alpha=-0.1):
        return spectral(scheme, 2 * math.pi / per_period, 0.0, alpha)

    assert Model().frequencies_hz()[0] == pytest.approx(3.18, abs=0.005)
    assert at(10, Scheme.CENTRAL).period_error == pytest.approx(-0.017, abs=0.0005)
    assert at(10, Scheme.AVERAGE).period_error == pytest.approx(0.032, abs=0.0005)
    assert at(10, Scheme.LINEAR).period_error == pytest.approx(0.016, abs=0.0005)
    assert at(10, Scheme.RK4).period_error == pytest.approx(0.0011, abs=0.00005)
    assert at(10, Scheme.RK4).damping == pytest.approx(0.00065, abs=0.000005)
    assert 10 * at(10, Scheme.AVERAGE).period_error == pytest.approx(1 / 3, abs=0.02)
    assert 2 * math.pi / 3.0 == pytest.approx(2.09, abs=0.005) and at(3.0, Scheme.CENTRAL).radius > 1
    assert 2 * math.pi / 3.2 == pytest.approx(1.96, abs=0.005)
    assert at(3.2, Scheme.CENTRAL).radius == pytest.approx(1.0)
    assert at(3.2, Scheme.CENTRAL).period_error == pytest.approx(-0.29, abs=0.005)
    assert at(3.0, Scheme.AVERAGE).period_error == pytest.approx(0.30, abs=0.005)
    assert at(1.0, Scheme.AVERAGE).period_error + 1 == pytest.approx(2.5, abs=0.05)
    assert at(1.0, Scheme.AVERAGE).radius == pytest.approx(1.0)
    assert stable_limit(Scheme.LINEAR) == pytest.approx(2 * math.sqrt(3)) and 2 * math.sqrt(3) / (2 * math.pi) == (
        pytest.approx(0.55, abs=0.005)
    )
    assert at(1.0, Scheme.LINEAR).radius > 1
    rk = at(5, Scheme.RK4)
    assert rk.period_error == pytest.approx(0.011, abs=0.0005) and 1 - rk.radius == pytest.approx(0.022, abs=0.0005)
    assert rk.damping == pytest.approx(0.018, abs=0.0005) and rk.radius**50 == pytest.approx(0.33, abs=0.005)
    assert at(5, Scheme.AVERAGE).period_error == pytest.approx(0.12, abs=0.005)
    eu = at(50, Scheme.EULER).radius
    assert eu - 1 == pytest.approx(0.008, abs=0.0005) and eu**50 - 1 == pytest.approx(0.48, abs=0.005)
    assert eu**500 == pytest.approx(50, abs=1)
    stiff = Model(stiff_hz=1000.0)
    f = stiff.frequencies_hz()
    assert f == pytest.approx([3.03, 1049], abs=0.5) and f[0] == pytest.approx(3.03, abs=0.005)
    h = 1 / f[0] / 50
    assert 2 * math.pi * f[1] * h == pytest.approx(43, abs=0.5) and 1 / (f[1] * h) == pytest.approx(1 / 7, abs=0.005)
    for scheme in (Scheme.CENTRAL, Scheme.RK4):
        x = simulate(scheme, stiff, Load(), h, 250)
        assert np.isnan(x[: int(0.020 / h) + 1]).any()
    assert np.all(np.isfinite(simulate(Scheme.AVERAGE, stiff, Load(), h, 250)))
    assert spectral(Scheme.HHT, 2 * math.pi * f[1] * h, 0.0, -0.3).radius == pytest.approx(0.57, abs=0.005)
    assert spectral(Scheme.HHT, 2 * math.pi * f[0] * h, 0.0, -0.3).damping == pytest.approx(0.00004, abs=0.000005)
    m, load = Model(), Load(0.0, 1.0, 2.0)
    h = 1 / m.frequencies_hz()[0] / 10
    t = np.arange(101) * h
    ref = exact(m, load, t)
    errs = {
        s: np.max(np.abs(simulate(s, m, load, h, 100) - ref)) / np.max(np.abs(ref))
        for s in (Scheme.FOH, Scheme.CENTRAL, Scheme.AVERAGE, Scheme.RK4)
    }
    assert errs[Scheme.FOH] == pytest.approx(0.013, abs=0.0005)
    assert [errs[s] for s in (Scheme.RK4, Scheme.CENTRAL, Scheme.AVERAGE)] == pytest.approx(
        [0.03, 0.41, 0.61], abs=0.005
    )


def test_fatigue_trials():
    from vib_tutorial.core.fatigue import (
        Curve,
        Stress,
        dirlik_damage,
        life_text,
        narrow_band_damage,
        rates,
        record,
    )

    c, c5 = Curve(), Curve(b=5.0)
    one = Stress()
    r = record(one, 600.0)
    assert rates(one).alpha2 == pytest.approx(0.95, abs=0.005)
    for est in (narrow_band_damage(one, c), dirlik_damage(one, c)):
        assert abs(est / r.damage_rate(c) - 1) < 0.015
    assert life_text(r.damage_rate(c)) == "25.1 days"
    two = Stress(sigma2=10.0)
    r2 = record(two, 600.0)
    assert rates(two).alpha2 == pytest.approx(0.46, abs=0.005) and rates(two).nu0 == pytest.approx(6.9, abs=0.05)
    assert narrow_band_damage(two, c) / r2.damage_rate(c) == pytest.approx(1.6, abs=0.05)
    assert dirlik_damage(two, c) / r2.damage_rate(c) == pytest.approx(0.92, abs=0.005)
    assert narrow_band_damage(two, c5) / r2.damage_rate(c5) == pytest.approx(1.6, abs=0.05)
    assert dirlik_damage(two, c5) / r2.damage_rate(c5) == pytest.approx(0.82, abs=0.005)
    average = np.mean([record(two, 600.0, k).damage_rate(c5) for k in range(1, 21)])
    assert dirlik_damage(two, c5) / average == pytest.approx(0.85, abs=0.005)
    assert life_text(record(Stress(sigma1=40.0), 600.0).damage_rate(c)) == "3.13 days"
    assert 1.1**3 == pytest.approx(1.33, abs=0.005)
    wide = Stress(zeta1=0.3)
    rw = record(wide, 600.0)
    assert rates(wide).alpha2 == pytest.approx(0.64, abs=0.005)
    assert narrow_band_damage(wide, c) / rw.damage_rate(c) == pytest.approx(1.15, abs=0.005)
    assert abs(dirlik_damage(wide, c) / rw.damage_rate(c) - 1) < 0.02
    short = [record(two, 20.0, k).damage_rate(c5) / r2.damage_rate(c5) for k in range(1, 11)]
    assert (min(short), max(short)) == pytest.approx((0.67, 1.53), abs=0.005)


def test_self_excited_trials():
    from vib_tutorial.core.self_excited import (
        Flutter,
        Galloping,
        VanDerPol,
        Viv,
        dominant_hz,
        galloping_branches,
        settled_amplitude,
    )

    v = VanDerPol()
    t, x, _ = v.simulate(60)
    assert settled_amplitude(x) == pytest.approx(0.0089, abs=0.00005) and -v.eps / 2 == pytest.approx(-0.1)
    t, x, _ = VanDerPol(eps=5.0).simulate(20)
    assert 1 / dominant_hz(x, t[1] - t[0]) == pytest.approx(1.84, abs=0.03)
    g = Galloping()
    assert g.onset == pytest.approx(3.81, abs=0.005)
    U = np.linspace(4.0, 7.5, 351)
    pairs = [b for b in galloping_branches(g, U) if not b.stable]
    band = U[np.isfinite(pairs[0].amplitude)]
    assert (band.min(), band.max()) == pytest.approx((4.7, 7.0), abs=0.05)
    at6 = sorted(b.amplitude[0] / g.side for b in galloping_branches(g, np.array([6.0])) if b.stable)
    assert at6 == pytest.approx([0.51, 1.28], abs=0.005)
    for start, want in ((0.001, 0.51), (0.12, 1.28)):
        _, y, _ = Galloping(speed=6.0, start=start).simulate(600)
        assert settled_amplitude(y) / g.side == pytest.approx(want, abs=0.01)
    _, y, _ = Galloping(speed=12.0).simulate(200)
    assert settled_amplitude(y) / g.side == pytest.approx(2.7, abs=0.05) and 12 / g.onset == pytest.approx(3.1, abs=0.1)
    viv = Viv()
    assert viv.mass_ratio == pytest.approx(10.2, abs=0.05) and viv.strouhal * 5.5 == pytest.approx(1.1)
    amp, freq = viv.sweep(np.array([4.0, 4.5, 5.5, 6.25, 6.5, 9.0]))
    assert amp[2] == pytest.approx(0.21, abs=0.005) and freq[2] == pytest.approx(1.0, abs=0.01)
    assert freq[1:4] == pytest.approx([1.0, 1.0, 1.0], abs=0.04) and abs(freq[4] - 1) > 0.04
    assert abs(freq[0] - 1) > 0.04
    assert amp[5] == pytest.approx(0.019, abs=0.0005) and freq[5] == pytest.approx(1.7, abs=0.02)
    f = Flutter()
    assert f.flutter_speed() == pytest.approx(19.1, abs=0.05) and f.divergence_speed() == pytest.approx(39, abs=0.5)
    fr, _ = f.modes(np.array([0.0, 19.0]))
    assert fr[0] == pytest.approx([2.93, 6.71], abs=0.1) and fr[1] == pytest.approx([3.4, 5.0], abs=0.05)
    assert Flutter(x_cg=-0.2).flutter_speed() == pytest.approx(f.divergence_speed(), abs=0.05)
    assert Flutter(x_cg=0.0).flutter_speed() == pytest.approx(17.7, abs=0.05)


def test_pendulum_trials():
    import math

    from vib_tutorial.core.pendulum import (
        Kind,
        Pendulum,
        conjugate_pivot,
        duffing_period,
        period,
        rod_period,
        swing,
    )

    p = Pendulum()
    assert (p.natural_hz, p.period0) == pytest.approx((0.498, 2.006), abs=5e-4)
    assert Pendulum(mass=7.0).period0 == pytest.approx(p.period0)
    assert period(p, math.radians(90)) == pytest.approx(2.368, abs=5e-4)
    assert period(p, math.radians(30)) / p.period0 - 1 == pytest.approx(0.017, abs=0.0005)
    assert period(p, math.radians(90)) / p.period0 - 1 == pytest.approx(0.18, abs=0.005)
    # Duffing's estimate (either form) within 1 % up to about 75°
    for a, inside in ((70, True), (80, False)):
        T = period(p, math.radians(a))
        for guess in (duffing_period(p, math.radians(a)), p.period0 * (1 + math.radians(a) ** 2 / 16)):
            assert (abs(guess / T - 1) < 0.01) == inside
    assert period(p, math.radians(170)) / p.period0 == pytest.approx(2.44, abs=0.005)
    rod = Pendulum(Kind.ROD)
    assert (rod.inertia, rod.equivalent_length, rod.period0) == pytest.approx((1 / 3, 0.667, 1.638), abs=5e-4)
    assert rod_period(1.0, 1 / math.sqrt(12)) == pytest.approx(1.524, abs=5e-4)
    assert conjugate_pivot(1.0, 0.5) == pytest.approx(0.167, abs=5e-4)
    assert rod_period(1.0, conjugate_pivot(1.0, 0.5)) == pytest.approx(rod.period0)
    disc = Pendulum(Kind.TORSION, length=0.1, spring=0.5)
    assert disc.natural_hz == pytest.approx(1.59, abs=0.005)
    assert period(disc, math.radians(170)) == pytest.approx(disc.period0, rel=1e-9)
    inv = Pendulum(Kind.INVERTED, spring=15.0)
    assert (inv.stiffness, inv.natural_hz) == pytest.approx((5.19, 0.363), abs=5e-4)
    assert 1 - period(inv, math.radians(60)) / inv.period0 == pytest.approx(0.10, abs=0.005)
    lean = Pendulum(Kind.INVERTED, spring=9.0, zeta=0.1)
    assert math.degrees(lean.leaning()) == pytest.approx(40.8, abs=0.05)
    _, th, _ = swing(lean, math.radians(1.0), 70.0)
    assert math.degrees(th[-1]) == pytest.approx(40.8, abs=0.1)
    assert Pendulum(gravity=1.62).period0 == pytest.approx(4.94, abs=0.005)


def test_pendulum_worked_example_and_inertia():
    """The figures Pendulums' worked example and its measured moment of inertia quote, on the page's defaults,
    and from the rounded figures they show."""
    import math

    from vib_tutorial.core.pendulum import Kind, Pendulum, duffing_period, period, swing

    p = Pendulum()
    assert (p.kind, p.mass, p.length, p.gravity, p.zeta) == (Kind.SIMPLE, 1.0, 1.0, 9.81, 0.0)
    assert (p.inertia, p.stiffness) == pytest.approx((1.0, 9.81))
    assert p.omega0 == pytest.approx(3.132, abs=5e-4) and p.natural_hz == pytest.approx(0.4985, abs=5e-5)
    assert p.period0 == pytest.approx(2.006, abs=5e-4)
    assert 3.132 / (2 * math.pi) == pytest.approx(0.4985, abs=5e-5)  # from the rounded ω_n
    assert 2 * math.pi / 3.132 == pytest.approx(2.006, abs=5e-4)
    a = math.radians(10)
    assert a == pytest.approx(0.1745, abs=5e-5)
    assert float(p.moment(a)) == pytest.approx(1.703, abs=5e-4) and p.stiffness * a == pytest.approx(1.712, abs=5e-4)
    assert 9.81 * 0.1745 == pytest.approx(1.712, abs=5e-4)
    assert 1 - 1.703 / 1.712 == pytest.approx(0.005, abs=5e-4)  # 0.5 % less
    assert a * a / 16 == pytest.approx(0.0019, abs=5e-5) and 0.1745**2 / 16 == pytest.approx(0.0019, abs=5e-5)
    exact = period(p, a)
    assert exact == pytest.approx(2.010, abs=5e-4) and 2.006 * 1.0019 == pytest.approx(2.010, abs=5e-4)
    assert 100 * (exact / p.period0 - 1) == pytest.approx(0.191, abs=0.0005)  # the readout's +0.191 %
    assert 100 * (duffing_period(p, a) / p.period0 - 1) == pytest.approx(0.191, abs=0.0005)
    assert 8 * (exact - p.period0) == pytest.approx(0.03, abs=0.005)  # behind after the eight swings drawn
    _, whole, linear = swing(p, a, 8 * p.period0, 4000)
    assert np.max(np.abs(whole - linear)) < 0.1 * a  # too little to see
    # The series for large swings
    for deg in (30, 60, 90):
        A = math.radians(deg)
        series = 1 + A**2 / 16 + 11 * A**4 / 3072
        assert series == pytest.approx(period(p, A) / p.period0, rel=0.012)
    # Measuring J: the 1 m rod hung from its end
    rod = Pendulum(Kind.ROD)
    assert rod.period0 == pytest.approx(1.638, abs=5e-4)
    J = rod.mass * rod.gravity * 0.5 * rod.period0**2 / (4 * math.pi**2)
    assert J == pytest.approx(1 / 3) and J - 0.25 == pytest.approx(1 / 12)
    J_rounded = 1 * 9.81 * 0.5 * 1.638**2 / (4 * math.pi**2)
    assert J_rounded == pytest.approx(0.3334, abs=5e-5) and 0.3334 - 0.25 == pytest.approx(0.0834, abs=5e-5)
    assert J_rounded - 0.25 == pytest.approx(0.0834, abs=5e-5) and 1 / 12 == pytest.approx(0.0833, abs=5e-5)
    assert J / (J - 0.25) == pytest.approx(4.0)
    slow = 1 * 9.81 * 0.5 * (1.001 * rod.period0) ** 2 / (4 * math.pi**2)
    assert slow / J - 1 == pytest.approx(0.002, abs=5e-5)
    assert (slow - 0.25) / (1 / 12) - 1 == pytest.approx(0.008, abs=5e-4)
    # Kater: the two pivots are L_eq apart
    assert 0.4 + (1 / 12) / 0.4 == pytest.approx(Pendulum(Kind.ROD, pivot=0.4).equivalent_length)


def test_beams_plates_trials():
    import math

    from vib_tutorial.core.beams_plates import (
        Beam,
        Plate,
        Theory,
        beam_frequencies,
        beam_omega,
        circular_modes,
        rect_modes,
        wave_speeds,
    )

    b = Beam()
    e = beam_frequencies(b, 5, Theory.EULER)
    assert e[0] == pytest.approx(235, abs=0.5)
    ratio = {th: beam_frequencies(b, 5, th) / e for th in Theory}
    assert 1 / ratio[Theory.TIMOSHENKO][0] - 1 == pytest.approx(0.017, abs=5e-4)
    assert 1 / ratio[Theory.TIMOSHENKO][4] - 1 == pytest.approx(0.33, abs=0.005)
    assert ratio[Theory.TIMOSHENKO][4] == pytest.approx(0.75, abs=0.005)
    assert (ratio[Theory.SHEAR][4], ratio[Theory.RAYLEIGH][4]) == pytest.approx((0.78, 0.91), abs=0.005)
    slim = Beam(depth=0.02)
    r = beam_frequencies(slim, 10) / beam_frequencies(slim, 10, Theory.EULER)
    assert (1 - r[0], 1 - r[9]) == pytest.approx((0.0007, 0.060), abs=5e-4)
    assert wave_speeds(b)[1] == pytest.approx(2900, abs=50)
    cutoff = float(beam_omega(b, 1e-6, branch=1)) / (2 * math.pi)
    assert cutoff == pytest.approx(16e3, rel=0.02)
    assert float(beam_omega(b, math.pi, branch=1)) / (2 * math.pi) == pytest.approx(16.4e3, rel=0.01)
    plate = Plate()
    modes = rect_modes(plate, 6)
    assert [round(m.freq_hz, 1) for m in modes[:4]] == pytest.approx([66.6, 128.1, 204.9, 230.5], abs=0.1)
    assert (modes[5].m, modes[5].n) == (3, 2)
    square = rect_modes(Plate(0.4, 0.4), 6)
    assert (square[4].m, square[4].n, square[5].m, square[5].n) == (1, 3, 3, 1)
    assert square[4].freq_hz == pytest.approx(square[5].freq_hz)
    circle = circular_modes(Plate(0.2, 0.2, circular=True), 4)
    assert [m.root**2 for m in circle] == pytest.approx([10.22, 21.26, 34.88, 39.77], abs=0.005)
    assert [m.freq_hz for m in circle] == pytest.approx([191, 397, 652, 743], abs=0.5)
    per_hz = plate.area / (2 * plate.scale)
    assert (per_hz, 1 / per_hz) == pytest.approx((0.026, 39), rel=0.02)


def test_seismic_trials():
    from vib_tutorial.core.seismic import (
        GROUNDS,
        Design,
        Rule,
        building,
        combine,
        correlation,
        modal,
        suite,
        with_tank,
    )

    d = Design()
    five = building(5)
    m = modal(five, d)
    rho = correlation(m.f, d.zeta)
    assert m.T[0] == pytest.approx(0.99, abs=0.005)
    assert (m.mass_share[0], m.mass_share[:2].sum()) == pytest.approx((0.88, 0.97), abs=0.005)
    assert m.sa[0] / 9.81 == pytest.approx(0.52, abs=0.005)
    shear = {r: float(combine(m.shear, r, rho)[0]) for r in Rule}
    assert (shear[Rule.CQC], shear[Rule.SRSS], shear[Rule.ABS]) == pytest.approx((229e3, 229e3, 277e3), abs=1e3)
    records = suite(five, d).base_shear
    assert records == pytest.approx(235e3, abs=1e3)
    assert shear[Rule.ABS] / records - 1 == pytest.approx(0.18, abs=0.005)
    assert rho[0, 1] == pytest.approx(0.007, abs=5e-4)
    tank = with_tank(five, 0.01)
    mt = modal(tank, d)
    rt = correlation(mt.f, d.zeta)
    assert (mt.f[0], mt.f[1], rt[0, 1]) == pytest.approx((0.96, 1.06, 0.50), abs=0.005)
    vt = suite(tank, d).base_shear
    assert float(combine(mt.shear, Rule.SRSS, rt)[0]) / vt - 1 == pytest.approx(-0.21, abs=0.01)
    assert abs(float(combine(mt.shear, Rule.CQC, rt)[0]) / vt - 1) < 0.11
    swing = suite(tank, d).displacement[-1]
    assert float(combine(mt.displacement, Rule.SRSS, rt)[-1]) / swing - 1 == pytest.approx(0.54, abs=0.01)
    assert abs(float(combine(mt.displacement, Rule.CQC, rt)[-1]) / swing - 1) < 0.11
    soft = modal(five, Design(ground=GROUNDS[3]))
    assert soft.sa[0] / 9.81 == pytest.approx(0.82, abs=0.005)
    assert float(combine(soft.shear, Rule.CQC, correlation(soft.f, 0.05))[0]) == pytest.approx(357e3, abs=1e3)
    eight = modal(building(8), d)
    assert (eight.T[0], eight.mass_share[0]) == pytest.approx((1.52, 0.86), abs=0.005)
    assert eight.sa[0] / 9.81 == pytest.approx(0.34, abs=0.005) and eight.sa[1:4] / 9.81 == pytest.approx(
        0.86, abs=0.005
    )
    total = combine(eight.shear, Rule.CQC, correlation(eight.f, 0.05))
    alone = abs(eight.shear[:, 0]) / total
    assert (alone[0], alone[-1]) == pytest.approx((0.96, 0.74), abs=0.005)
    assert Design(zeta=0.10).eta == pytest.approx(0.82, abs=0.005)


def test_vibroacoustics_trials():
    import math

    import numpy as np

    from vib_tutorial.core.beams_plates import Material, Plate, rect_modes
    from vib_tutorial.core.radiation import (
        AIR,
        band_edges,
        coincidence_dip,
        critical_frequency,
        maidanik,
        mass_law,
        modal_efficiency,
        point_drive,
        radiation_loss_factor,
        transmission_loss,
    )

    def sigma(m, n, f):
        return float(modal_efficiency(p, np.array([m]), np.array([n]), f)[0, 0])

    p = Plate()
    fc = critical_frequency(p)
    assert fc == pytest.approx(3990, abs=5) and AIR.c == 343
    # a corner mode
    f11 = rect_modes(p, 1)[0].freq_hz
    assert (f11, AIR.c / f11) == pytest.approx((66.6, 5.2), abs=0.05)
    assert sigma(1, 1, f11) == pytest.approx(0.037, abs=5e-4)
    f = np.geomspace(100, 5000, 400)
    s11 = np.array([sigma(1, 1, x) for x in f])
    assert f[np.argmax(s11 > 1)] == pytest.approx(450, rel=0.02) and AIR.c / 450 == pytest.approx(0.76, abs=0.005)
    assert (s11.max(), f[s11.argmax()]) == pytest.approx((1.28, 720), rel=0.02)
    # even modes, an edge mode
    modes = rect_modes(p, 10)
    assert (modes[4].m, modes[4].n, modes[4].freq_hz) == pytest.approx((2, 2, 266), abs=0.5)
    assert (sigma(2, 2, 100.0), sigma(1, 1, 100.0)) == pytest.approx((1.4e-5, 0.082), rel=0.03)
    f13 = modes[7].freq_hz
    assert (modes[7].m, modes[7].n, f13) == pytest.approx((1, 3, 435), abs=0.5)
    assert (3 * math.pi / p.b, math.pi / p.a, 2 * math.pi * f13 / AIR.c) == pytest.approx((23.6, 5.2, 7.98), abs=0.05)
    first = [sigma(md.m, md.n, md.freq_hz) for md in modes]
    assert first[7] == pytest.approx(0.059, abs=5e-4) and max(first) == first[7]
    # a point-driven panel
    freq = np.geomspace(20, 2.5 * fc, 700)
    d = point_drive(p, 0.01, (0.21, 0.13), freq)
    assert d.infinite_input == pytest.approx(565e-6, rel=0.002)

    def gap(centre):
        lo, hi = band_edges(np.array([centre]))
        sel = (freq >= lo[0]) & (freq < hi[0])
        return 10 * math.log10(d.input_power[sel].mean() / d.radiated_power[sel].mean())

    assert (gap(1000.0), gap(1000 * 10**0.8)) == pytest.approx((18, 11), abs=0.5)
    eta_rad = radiation_loss_factor(p, 1000.0)
    assert (float(maidanik(p, 1000.0)), eta_rad, 0.01 / eta_rad) == pytest.approx((0.054, 1.5e-4, 66), rel=0.02)
    assert float(maidanik(p, 6300.0)) == pytest.approx(1.7, abs=0.05)
    # the mass law
    assert (p.surface_mass, float(mass_law(p, 500.0))) == pytest.approx((23.6, 39.0), abs=0.051)
    assert transmission_loss(p, 0.01, 500.0)[0] == pytest.approx(33.8, abs=0.05)
    # glass
    glass = Material(70e9, 0.22, 2500.0)
    window = Plate(1.2, 1.0, 0.010, glass)
    assert critical_frequency(window) == pytest.approx(1200, abs=5)
    f_dip, tl_dip = coincidence_dip(window, 0.01)
    assert (f_dip, tl_dip) == pytest.approx((1280, 19.2), rel=0.004)
    assert float(mass_law(window, 500.0)) == pytest.approx(39.5, abs=0.05)
    assert critical_frequency(Plate(1.2, 1.0, 0.004, glass)) == pytest.approx(2990, abs=5)
    assert coincidence_dip(Plate(1.2, 1.0, 0.004, glass), 0.01)[1] == pytest.approx(19.2, abs=0.05)
    # damping at coincidence
    assert (coincidence_dip(p, 0.01)[1], coincidence_dip(p, 0.1)[1]) == pytest.approx((26.1, 37.5), abs=0.05)
    assert (transmission_loss(p, 0.1, 2 * fc)[0], transmission_loss(p, 0.01, 2 * fc)[0]) == pytest.approx(
        (51.0, 41.3), abs=0.05
    )
    at = np.array([f11])  # ⟨v²⟩ at a resonance ∝ 1/η²; over a band ∝ 1/η
    light, damped = (point_drive(p, eta, (0.21, 0.13), at).mean_square_velocity[0] for eta in (0.01, 0.1))
    assert 10 * math.log10(light / damped) == pytest.approx(20, abs=0.2)
    heavy = point_drive(p, 0.1, (0.21, 0.13), freq)
    lo, hi = band_edges(np.array([2000.0]))
    sel = (freq >= lo[0]) & (freq < hi[0])
    ratio = d.mean_square_velocity[sel].mean() / heavy.mean_square_velocity[sel].mean()
    assert 10 * math.log10(ratio) == pytest.approx(10, abs=1)
    assert heavy.input_power[sel].mean() / d.input_power[sel].mean() == pytest.approx(1, abs=0.15)


def test_sea_trials():
    from vib_tutorial.core.beams_plates import Plate
    from vib_tutorial.core.radiation import critical_frequency
    from vib_tutorial.core.sea import (
        Network,
        Oscillators,
        coupling_beta,
        junction_transmission,
        network,
        oscillator_pair,
    )

    # two oscillators share power
    osc = Oscillators()
    r = oscillator_pair(osc, 1.0, 0.5)
    assert (r.E1, r.E2, r.flow) == pytest.approx((0.0396, 0.0366, 0.00593), abs=6e-5)
    assert coupling_beta(osc) == pytest.approx(2.01, abs=0.005)
    # tuned
    tuned = coupling_beta(Oscillators(f2=100.0))
    assert tuned == pytest.approx(26.2, abs=0.05)
    assert tuned / coupling_beta(osc) == pytest.approx(13, abs=0.1)
    assert coupling_beta(Oscillators(f2=150.0)) == pytest.approx(0.070, abs=0.0005)
    assert Oscillators(f2=100.0).deltas == pytest.approx((25.1, 12.6), abs=0.05)
    # strong coupling
    strong = Oscillators(f2=100.0, zeta1=0.002, zeta2=0.002)
    assert coupling_beta(strong) == pytest.approx(196, abs=0.5)
    assert strong.deltas[0] == pytest.approx(2.51, abs=0.005)
    one = oscillator_pair(strong, 1.0, 0.0)
    assert one.E2 / one.E1 == pytest.approx(0.987, abs=0.0005)
    assert one.flow == pytest.approx(0.497, abs=0.0005)

    p1, p2 = Plate(), Plate(0.5, 0.4, 0.002)
    at = np.array([1000.0])

    def band(plate2, loss2=0.01):
        res = network(Network(p1, plate2, 0.01, loss2, 0.4), at)
        e, n = res.energy[0], res.density[0]
        return dict(
            share=(e[1] / n[1]) / (e[0] / n[0]),
            v2=10 * math.log10(res.velocity[0, 1] / res.velocity[0, 0]),
            spl=10 * math.log10(res.pressure[0] / 4e-10),
            by2=res.flow[0, 1, 2] / (res.flow[0, 0, 2] + res.flow[0, 1, 2]),
            eta=res.eta[0],
        )

    # plates at a corner
    assert junction_transmission(p1, p2, 1000.0) == pytest.approx(0.279, abs=0.0005)
    b = band(p2)
    assert (b["eta"][0, 1], b["eta"][1, 0]) == pytest.approx((0.00809, 0.00647), abs=6e-6)
    assert b["v2"] == pytest.approx(-0.57, abs=0.005)
    assert b["share"] == pytest.approx(0.39, abs=0.005)
    assert b["spl"] == pytest.approx(63.1, abs=0.05)
    assert b["by2"] == pytest.approx(0.281, abs=0.0005)
    # equipartition
    e = band(p2, 1e-4)
    assert e["share"] == pytest.approx(0.967, abs=0.0005)
    assert e["v2"] == pytest.approx(3.4, abs=0.05)
    assert e["spl"] == pytest.approx(66.2, abs=0.05)
    assert band(p2, 0.1)["share"] == pytest.approx(0.061, abs=0.0005)
    # a heavy wall
    wall = Plate(0.5, 0.4, 0.010)
    assert junction_transmission(p1, wall, 1000.0) == pytest.approx(0.037, abs=0.0005)
    h = band(wall)
    assert h["v2"] == pytest.approx(-16.2, abs=0.05)
    assert h["by2"] == pytest.approx(0.458, abs=0.0005)
    assert h["spl"] == pytest.approx(65.7, abs=0.05)
    assert critical_frequency(wall) == pytest.approx(1200, abs=5)


def test_fluid_structure_trials():
    from vib_tutorial.core.added_mass import (
        AIR,
        Body,
        Shape,
        avmi,
        plate_added_mass,
        plate_dry_frequency,
        plate_modal_mass,
        wet_frequency,
    )
    from vib_tutorial.core.beams_plates import Plate

    def wet(b):
        return wet_frequency(20.0, b.mass, b.added_mass)

    sphere = Body(Shape.SPHERE, 0.1)
    assert (sphere.added_mass, sphere.mass, wet(sphere)) == pytest.approx((2.09, 32.9, 19.4), rel=0.0025)
    rod = Body(Shape.CYLINDER, 0.025)
    assert (rod.added_mass, wet(rod)) == pytest.approx((1.96, 18.8), rel=0.0025)
    light = Body(Shape.CYLINDER, 0.025, 500.0)
    assert light.added_mass / light.mass == pytest.approx(2.0)
    assert wet(light) / 20 == pytest.approx(0.577, abs=0.0005)
    assert wet(light) == pytest.approx(11.5, abs=0.05)
    air = Body(Shape.CYLINDER, 0.025, fluid=AIR)
    assert air.added_mass == pytest.approx(0.00238, abs=5e-6)
    assert air.mass / air.added_mass == pytest.approx(6500, rel=0.01)
    assert 1 - wet(air) / 20 < 1e-4
    pipe = Body(Shape.CYLINDER, 0.025, outer=0.03)
    assert (pipe.coefficient, pipe.added_mass, wet(pipe)) == pytest.approx((5.55, 10.9, 15.3), rel=0.0025)
    tight = Body(Shape.CYLINDER, 0.025, outer=0.025 * 1.05)
    assert (tight.coefficient, tight.added_mass / tight.mass, wet(tight)) == pytest.approx((20.5, 2.6, 10.5), abs=0.05)
    r = 1.05  # the speed beside the rod: |A + B/R²| = C_a
    assert (r**2 + 1) / (r**2 - 1) == pytest.approx(20.5, abs=0.05)

    plate = Plate()
    assert plate_added_mass(plate, 1, 1) == pytest.approx(11.9, abs=0.05)
    assert plate_modal_mass(plate) == pytest.approx(1.41, abs=0.005)
    beta = avmi(plate, 1, 1)
    f11 = plate_dry_frequency(plate, 1, 1)
    assert (beta, f11, f11 / math.sqrt(1 + beta)) == pytest.approx((8.43, 66.6, 21.7), rel=0.0025)
    beta32 = avmi(plate, 3, 2)
    f32 = plate_dry_frequency(plate, 3, 2)
    assert (beta32, f32, f32 / math.sqrt(1 + beta32)) == pytest.approx((2.18, 369, 207), abs=0.5)
    assert beta32 == pytest.approx(2.18, abs=0.005)
    thick = Plate(thickness=0.010)
    assert plate_modal_mass(thick) / plate_modal_mass(plate) == pytest.approx(3.33, abs=0.005)
    bt, ft = avmi(thick, 1, 1), plate_dry_frequency(thick, 1, 1)
    assert (bt, ft, ft / math.sqrt(1 + bt)) == pytest.approx((2.53, 222, 118), abs=0.5)
    assert bt == pytest.approx(2.53, abs=0.005)
    assert 1 / math.sqrt(1 + bt) == pytest.approx(0.53, abs=0.005)


def test_sloshing_trials():
    from vib_tutorial.core.sloshing import (
        Damper,
        G,
        Structure,
        Tank,
        damper_response,
        frozen_response,
        mass_ratio,
        peak,
        response_band,
        rigid_mass,
        slosh_frequency,
        slosh_mass,
    )

    tank = Tank(1.0, 0.5)
    assert slosh_frequency(tank, 1) == pytest.approx(0.846, abs=5e-4)
    assert (float(slosh_mass(tank, 1)), rigid_mass(tank)) == pytest.approx((237, 250), abs=0.5)
    assert tank.liquid_mass == 500
    shallow = Tank(1.0, 0.05)
    assert (slosh_frequency(shallow, 1), math.sqrt(G * 0.05) / 2) == pytest.approx((0.349, 0.350), abs=5e-4)
    assert float(slosh_mass(shallow, 1)) == pytest.approx(40.2, abs=0.05)
    assert rigid_mass(shallow) == pytest.approx(2.71, abs=0.005)
    deep = Tank(1.0, 2.0)
    assert slosh_frequency(deep, 1) == pytest.approx(0.884, abs=5e-4)
    assert math.sqrt(G * math.pi) / (2 * math.pi) == pytest.approx(0.884, abs=5e-4)
    assert 1 / math.pi == pytest.approx(0.32, abs=0.005)
    assert float(slosh_mass(deep, 1)) == pytest.approx(258, abs=0.5)
    assert float(slosh_mass(deep, 1)) / deep.liquid_mass == pytest.approx(0.13, abs=0.005)
    assert rigid_mass(deep) == pytest.approx(1729, abs=0.5)
    assert slosh_frequency(tank, 3) == pytest.approx(1.53, abs=0.005)
    assert float(slosh_mass(tank, 3)) == pytest.approx(9.55, abs=0.005)
    assert float(slosh_mass(tank, 3)) / 500 == pytest.approx(0.02, abs=0.001)
    assert slosh_frequency(tank, 2) == pytest.approx(1.25, abs=0.005)
    assert float(slosh_mass(tank, 2)) == 0

    structure = Structure(20000, 0.5, 0.01)
    damper = Damper(Tank(2.0, 0.4, 2.0), 1, 0.13)
    assert slosh_frequency(damper.tank, 1) == pytest.approx(0.466, abs=5e-4)
    assert (damper.liquid, float(damper.masses[0])) == pytest.approx((1600, 1149), abs=0.5)
    assert mass_ratio(structure, damper) == pytest.approx(0.0562, abs=5e-5)
    f = response_band(structure)
    assert float(np.abs(frozen_response(structure, damper, f)).max()) == pytest.approx(52.0, abs=0.05)
    assert peak(structure, damper) == pytest.approx(5.58, abs=0.005)
    bare = Damper(Tank(2.0, 0.4, 2.0), 1, 0.005)
    r = np.abs(damper_response(structure, bare, f))
    tops, _ = find_peaks(r)
    assert f[tops] == pytest.approx([0.426, 0.540], abs=5e-4)
    assert r[tops] == pytest.approx([34.5, 32.8], abs=0.05)
    mistuned = Damper(Tank(2.0, 0.25, 2.0), 1, 0.13)
    assert slosh_frequency(mistuned.tank, 1) == pytest.approx(0.382, abs=5e-4)
    assert peak(structure, mistuned) == pytest.approx(18.2, abs=0.05)
    four = Damper(Tank(2.0, 0.3, 2.0), 4, 0.13)
    assert four.liquid == pytest.approx(4800)
    assert (mass_ratio(structure, four), peak(structure, four)) == pytest.approx((0.171, 4.41), abs=5e-3)


def test_quarter_car_trials():
    from dataclasses import replace

    from vib_tutorial.core.quarter_car import (
        Car,
        Output,
        Road,
        best_damper,
        drive,
        response,
    )

    car, road = Car(), Road("C", 20.0)
    r = response(car, road)
    assert car.modes() == pytest.approx((1.24, 11.8), abs=0.01)
    assert (r.rms[Output.BODY], r.load_ratio) == pytest.approx((1.48, 0.24), abs=0.005)
    assert r.rms[Output.TRAVEL] == pytest.approx(0.0107, abs=5e-5)
    rough = response(car, Road("D", 30.0))
    assert (rough.rms[Output.BODY], rough.load_ratio, rough.lift_probability) == pytest.approx(
        (3.63, 0.59, 0.044), abs=0.005
    )
    dampers = np.geomspace(150.0, 10e3, 4000)
    assert best_damper(car, road, Output.BODY, dampers) == pytest.approx(830, abs=10)
    assert best_damper(car, road, Output.TYRE, dampers) == pytest.approx(2340, abs=10)
    soft = replace(car, damper=830.0)
    rs = response(soft, road)
    assert soft.body_zeta() == pytest.approx(0.17, abs=0.005)
    assert (rs.rms[Output.BODY], rs.load_ratio) == pytest.approx((1.36, 0.29), abs=0.005)
    assert rs.rms[Output.TRAVEL] == pytest.approx(0.0144, abs=5e-5)
    assert 1 - rs.rms[Output.BODY] / r.rms[Output.BODY] == pytest.approx(0.08, abs=0.005)
    firm = replace(car, damper=2340.0)
    rf = response(firm, road)
    assert (firm.body_zeta(), rf.load_ratio, rf.rms[Output.BODY]) == pytest.approx((0.48, 0.23, 1.72), abs=0.005)
    springy = replace(car, spring=10e3)
    rk = response(springy, road)
    assert (springy.modes()[0], rk.rms[Output.BODY]) == pytest.approx((0.90, 1.35), abs=0.005)
    assert 1 - rk.rms[Output.BODY] / r.rms[Output.BODY] == pytest.approx(0.09, abs=0.005)
    assert car.sprung * 9.80665 / springy.spring == pytest.approx(0.294, abs=5e-4)
    heavy = replace(car, unsprung=60.0)
    assert (heavy.modes()[1], response(heavy, road).load_ratio) == pytest.approx((9.6, 0.26), abs=0.05)
    d = drive(car, road, 60.0)
    skip = int(2.0 / (d.t[1] - d.t[0]))
    for record, spectral in (
        (d.body, r.rms[Output.BODY]),
        (d.travel, r.rms[Output.TRAVEL]),
        (d.tyre, r.rms[Output.TYRE]),
    ):
        assert np.std(record[skip:]) == pytest.approx(spectral, rel=0.02)


def test_variability_trials():
    from vib_tutorial.core.model import ChainSystem
    from vib_tutorial.core.variability import (
        Ring,
        Scatter,
        amplification,
        amplification_curve,
        monte_carlo,
        participation,
        ring_modes,
        sensitivity_sigma,
        tuned_band,
        whitehead,
    )

    chain = ChainSystem.uniform(4)
    f = np.linspace(0.1, 10.0, 200)
    springs = monte_carlo(chain, Scatter(0.0, 0.05, 0.0), 200, f, 3, 3)
    assert springs.nominal_freqs[0] == pytest.approx(1.105, abs=5e-4)
    assert springs.freqs[:, 0].std() == pytest.approx(0.015, abs=5e-4)
    assert np.mean(springs.freqs.std(axis=0) / springs.nominal_freqs) == pytest.approx(0.014, abs=5e-4)
    assert sensitivity_sigma(chain, Scatter(0.0, 0.05, 0.0)) / springs.nominal_freqs == pytest.approx(0.0144, abs=5e-5)
    both = monte_carlo(chain, Scatter(0.05, 0.05, 0.0), 200, f, 3, 3)
    assert both.freqs[:, 0].std() == pytest.approx(0.023, abs=5e-4)
    assert np.mean(both.freqs.std(axis=0) / both.nominal_freqs) == pytest.approx(0.02, abs=1e-3)
    dampers = monte_carlo(chain, Scatter(0.0, 0.0, 0.3), 200, f, 3, 3)
    assert np.ptp(dampers.freqs, axis=0) == pytest.approx(0.0, abs=1e-12)

    tuned = Ring()
    assert tuned_band(tuned) == pytest.approx((15.9, 17.4), abs=0.05)
    p = participation(ring_modes(tuned)[1])
    assert (p.min(), p.max()) == pytest.approx((6.0, 12.0), abs=0.05)
    weak = Ring(coupling=0.005, mistuning=0.01)
    assert participation(ring_modes(weak)[1])[0] == pytest.approx(1.3, abs=0.05)
    strong = participation(ring_modes(Ring(coupling=0.2, mistuning=0.01))[1])
    assert (strong.min(), strong.max()) == pytest.approx((6.0, 12.0), abs=0.1)

    forced = Ring(coupling=0.02, mistuning=0.01)
    assert amplification(forced, 3) == pytest.approx(1.06, abs=0.005)
    _, top = amplification_curve(forced, 3, np.array([0.01, 0.02, 0.1]), 12)
    assert top == pytest.approx([1.34, 1.42, 1.28], abs=0.005)
    assert whitehead(12) == pytest.approx(2.23, abs=0.005)


def test_diagnostics_trials():
    from dataclasses import replace

    from vib_tutorial.core.diagnostics import (
        BearingFault,
        GearFault,
        Machine,
        RunUp,
        amplitude_spectrum,
        band_pass,
        cepstrum,
        envelope_spectrum,
        kurtosis,
        line_amplitude,
        order_map,
        order_track,
        run_up,
        sidebands,
    )

    band = (2550.0, 3450.0)
    m = Machine(speed=1797 / 60)
    assert m.mesh_hz == pytest.approx(808.65)
    healthy = record_of(m)
    assert kurtosis(healthy.x) == pytest.approx(2.40, abs=0.005)
    _, c = cepstrum(healthy.x, healthy.fs)
    at = round(healthy.fs / m.speed)
    assert c[at - 2 : at + 3].max() == pytest.approx(0.027, abs=5e-4)

    ecc = replace(m, gear=GearFault.ECCENTRIC)
    s = amplitude_spectrum(record_of(ecc).x, healthy.fs)
    _, a = sidebands(s.freqs, s.amplitude, m.mesh_hz, m.speed, 1)
    assert (a[2], a[0], a[1]) == pytest.approx((0.23, 0.08, 1.0), abs=0.005)

    crack = record_of(replace(m, gear=GearFault.CRACK))
    s = amplitude_spectrum(crack.x, crack.fs)
    _, a = sidebands(s.freqs, s.amplitude, m.mesh_hz, m.speed, 6)
    assert np.delete(a, 6) == pytest.approx(0.042, abs=0.006)
    _q, c = cepstrum(crack.x, crack.fs)
    assert 1e3 / m.speed == pytest.approx(33.4, abs=0.05)
    assert c[at - 2 : at + 3].max() == pytest.approx(0.12, abs=0.005)

    noisy = replace(m, noise=1.0)
    f = noisy.bearing.frequencies(m.speed)
    assert (f["BPFO"] / m.speed, f["BPFO"], f["BPFI"], 2 * f["BSF"], f["FTF"]) == pytest.approx(
        (3.585, 107.4, 162.2, 141.2, 11.9), abs=0.05
    )
    outer = replace(noisy, bearing_fault=BearingFault.OUTER)
    r = record_of(outer)
    s = amplitude_spectrum(r.x, r.fs)
    d = outer.defect_hz()
    assert line_amplitude(s.freqs, s.amplitude, d) == pytest.approx(0.049, abs=5e-4)
    assert np.median(s.amplitude[(s.freqs > 50) & (s.freqs < 700)]) == pytest.approx(0.043, abs=5e-4)
    assert (kurtosis(r.x), kurtosis(band_pass(r.x, r.fs, band))) == pytest.approx((3.1, 5.6), abs=0.05)
    e = envelope_spectrum(r.x, r.fs, band)
    assert [line_amplitude(e.freqs, e.amplitude, h * d) for h in (1, 2, 3)] == pytest.approx(
        [0.45, 0.34, 0.23], abs=0.005
    )
    assert np.median(e.amplitude[(e.freqs > 20) & (e.freqs < 500)]) < 0.045
    for fault, (centre, lower, upper) in (
        (BearingFault.INNER, (0.31, 0.16, 0.13)),
        (BearingFault.BALL, (0.29, 0.14, 0.135)),
    ):
        mm = replace(noisy, bearing_fault=fault)
        r = record_of(mm)
        e = envelope_spectrum(r.x, r.fs, band)
        d, fm = mm.defect_hz(), mm.modulation_hz()
        assert [line_amplitude(e.freqs, e.amplitude, v) for v in (d, d - fm, d + fm)] == pytest.approx(
            [centre, lower, upper], abs=0.005
        )

    slow, fast = RunUp(), RunUp(duration=4.0)
    fine = np.linspace(slow.start, slow.end, 2000)
    steady = slow.steady(1, fine)
    assert (steady.max(), 60 * fine[steady.argmax()]) == pytest.approx((211, 2406), abs=1)
    om = order_map(run_up(slow))
    one, two = order_track(om, 1), order_track(om, 2)
    assert (one.max(), 60 * om.speed[one.argmax()]) == pytest.approx((209, 2445), abs=1)
    assert 60 * om.speed[two.argmax()] == pytest.approx(1220, abs=5)
    assert (two.max(), slow.steady(2, fine).max()) == pytest.approx((28.5, 33.3), abs=0.05)
    om = order_map(run_up(fast))
    one = order_track(om, 1)
    assert (one.max(), 60 * om.speed[one.argmax()]) == pytest.approx((179, 2531), abs=1)
    assert 1 - one.max() / steady.max() == pytest.approx(0.15, abs=0.005)
    sweep = 2 * slow.zeta * slow.mode / ((fast.end - fast.start) / fast.duration)
    assert (sweep, sweep * slow.mode) == pytest.approx((0.16, 6.4), abs=0.005)


def record_of(machine):
    from vib_tutorial.core.diagnostics import record

    return record(machine)


def test_deriving_trials():
    from dataclasses import replace

    from vib_tutorial.core.bounce_pitch import Bump, Car, Coordinates, matrices, modes, newton, ride, slipped

    def hz(car, coords=Coordinates.CENTRE, e=0.0):
        return [m.frequency for m in modes(car, coords, e)]

    car = Car()
    m1, m2 = modes(car)
    assert (m1.frequency, m2.frequency) == pytest.approx((1.25, 1.41), abs=0.005)
    assert (m1.node, m2.node) == pytest.approx((-1.92, 0.88), abs=0.005)
    assert not m1.inside(car) and m2.inside(car)
    # no static coupling
    level = replace(car, k_rear=32e3)
    assert level.k_front * level.front == pytest.approx(level.k_rear * level.rear) == pytest.approx(48e3)
    assert hz(level) == pytest.approx((1.23, 1.27), abs=0.005)
    # the elastic centre
    assert car.elastic_centre() == pytest.approx(-0.15)
    at = matrices(car, Coordinates.POINT, -0.15)
    assert abs(at.K[0, 1]) < 1e-9 and at.M[0, 1] == pytest.approx(180.0)
    assert hz(car, Coordinates.POINT, -0.15) == pytest.approx((1.25, 1.41), abs=0.005)
    # moments about the front axle, and the slip
    s = slipped(car, 1.2)
    wrong = np.sort(np.sqrt(np.linalg.eigvals(np.linalg.solve(s.M, s.K)).real)) / (2 * np.pi)
    assert wrong == pytest.approx((0.93, 1.38), abs=0.005)
    # the ends' coordinates
    assert matrices(car, Coordinates.ENDS).M[0, 1] == pytest.approx(18.1, abs=0.05)
    rows = newton(car, Coordinates.ENDS).M
    np.testing.assert_allclose(rows, [[667, 533], [751, -751]], atol=0.5)
    # a dynamic index of 1
    one = replace(car, gyration=1.342)
    assert one.dynamic_index == pytest.approx(1.0, abs=0.002) and math.sqrt(1.2 * 1.5) == pytest.approx(1.34, abs=0.005)
    assert one.end_masses == pytest.approx((666.7, 533.3), abs=0.05)
    assert hz(one) == pytest.approx(one.end_frequencies(), abs=0.002)
    assert one.end_frequencies() == pytest.approx((1.23, 1.38), abs=0.005)
    assert [m.node for m in modes(one)] == pytest.approx((-1.5, 1.2), abs=0.005)
    # over a hump, either way round
    hump = Bump()
    assert car.wheelbase / hump.speed == pytest.approx(0.32, abs=0.005)
    for kf, kr, ends, settled in ((48e3, 30e3, (1.35, 1.19), 1.95), (30e3, 48e3, (1.07, 1.51), 1.52)):
        c = replace(car, k_front=kf, k_rear=kr)
        assert c.end_frequencies() == pytest.approx(ends, abs=0.005)
        r = ride(c, hump)
        assert np.degrees(np.abs(r.pitch[r.after()]).max()) == pytest.approx(settled, abs=0.005)
    assert 1 - 1.068 / 1.51 == pytest.approx(0.29, abs=0.005)


def test_deriving_pendulum_trials():
    from vib_tutorial.core.double_pendulum import (
        Angles,
        DoublePendulum,
        divergence,
        matrices,
        modes,
        newton,
        slipped,
        swing,
    )
    from vib_tutorial.core.pendulum import measured_period

    p = DoublePendulum()
    slow, fast = modes(p)
    assert (slow.frequency, fast.frequency) == pytest.approx((0.540, 1.30), abs=0.005)
    assert (slow.ratio, fast.ratio) == pytest.approx((1.41, -1.41), abs=0.005)
    assert p.separate_frequencies() == pytest.approx((0.705, 0.705), abs=0.0005)
    assert matrices(p).M[0, 1] == pytest.approx(0.25) and matrices(p).K[0, 1] == 0
    # the free-body diagram's rows, and the slip
    np.testing.assert_allclose(newton(p).M, [[0.5, 0], [0.5, 0.5]])
    s = slipped(p)
    wrong = np.sqrt(np.linalg.eigvals(np.linalg.solve(s.M, s.K)).real) / (2 * np.pi)
    assert wrong == pytest.approx((0.705, 0.705), abs=0.0005)
    # the elbow's bend
    bend = matrices(p, Angles.RELATIVE)
    assert (bend.K[0, 1], bend.M[0, 0]) == pytest.approx((4.91, 1.25), abs=0.005)
    assert [m.frequency for m in modes(p, Angles.RELATIVE)] == pytest.approx((0.540, 1.30), abs=0.005)
    assert modes(p, Angles.RELATIVE)[0].shape[1] == pytest.approx(0.41, abs=0.005)
    # released in the slower mode, at 10° and at 40°
    for start, whole, apart in (((10, 14.1), 1.858, 1.8), ((40, 56.6), 1.924, 77)):
        r = swing(p, tuple(np.radians(start)), 20.0)
        assert measured_period(r.t, r.whole[:, 0]) == pytest.approx(whole, abs=0.0005)
        assert measured_period(r.t, r.linear[:, 0]) == pytest.approx(1.853, abs=0.0005)
        assert np.degrees(np.abs(r.whole[:, 0] - r.linear[:, 0]).max()) == pytest.approx(apart, rel=0.03)
    assert 1.924 / 1.853 - 1 == pytest.approx(0.038, abs=0.0005)
    # chaos
    start = (np.radians(120), np.radians(120))
    d = divergence(p, start, 15.0)
    assert d.doubling_time() == pytest.approx(0.49, abs=0.005)
    assert d.apart[np.searchsorted(d.t, 10.0)] > 1.0
    assert swing(p, start).drift < 1e-9


def test_deriving_worked_example():
    """The notes' worked example: the car as the page opens, in (x_G, θ), and the equal double pendulum."""
    from vib_tutorial.core.bounce_pitch import Car, Coordinates, matrices, modes
    from vib_tutorial.core.double_pendulum import DoublePendulum
    from vib_tutorial.core.double_pendulum import modes as pendulum_modes
    from vib_tutorial.gui.deriving import DerivingTrial

    car = DerivingTrial("", "").car
    assert car == Car()
    m, J, a, b, k = 1200.0, 1200 * 1.3**2, 1.2, 1.5, 40e3
    assert J == pytest.approx(2028) and car.inertia == pytest.approx(J)
    K = [[2 * k, k * a - k * b], [k * a - k * b, k * a**2 + k * b**2]]
    assert K == [[80_000, -12_000], [-12_000, 147_600]] and k * a == 48_000 and k * b == 60_000
    mats = matrices(car, Coordinates.CENTRE)
    np.testing.assert_allclose(mats.M, [[m, 0], [0, J]])
    np.testing.assert_allclose(mats.K, K)
    np.testing.assert_allclose(mats.C, [[5000, -750], [-750, 9225]])
    # uncoupled
    assert math.sqrt(80_000 / 1200) == pytest.approx(8.165, abs=5e-4)
    assert math.sqrt(147_600 / 2028) == pytest.approx(8.531, abs=5e-4)
    assert car.uncoupled_frequencies() == pytest.approx((1.30, 1.36), abs=0.005)
    assert (8.165 / (2 * math.pi), 8.531 / (2 * math.pi)) == pytest.approx((1.30, 1.36), abs=0.005)
    # the quadratic, its coefficients as quoted, and its roots from them
    A, B, C = m * J, m * 147_600 + J * 80_000, 80_000 * 147_600 - 12_000**2
    assert (A, B, C) == pytest.approx((2.434e6, 3.394e8, 1.166e10), rel=5e-4)
    roots = np.sort(np.roots([A, -B, C]))
    assert roots == pytest.approx((61.45, 78.00), abs=0.005)
    rounded = np.sort(np.roots([2.434e6, -3.394e8, 1.166e10]))
    assert rounded == pytest.approx((61.45, 78.00), rel=0.01)
    assert np.sqrt([61.45, 78.00]) / (2 * math.pi) == pytest.approx((1.25, 1.41), abs=0.005)
    assert [md.frequency for md in modes(car)] == pytest.approx((1.25, 1.41), abs=0.005)
    assert 1.25 < 1.30 and 1.41 > 1.36  # one below both uncoupled, one above
    # the nodes, from the first row
    assert 61.45 * 1200 == pytest.approx(73_740) and 78.00 * 1200 == pytest.approx(93_600)
    assert 12_000 / (80_000 - 73_740) == pytest.approx(1.92, abs=0.005)
    assert 12_000 / (80_000 - 93_600) == pytest.approx(-0.88, abs=0.005)
    assert [md.node for md in modes(car)] == pytest.approx((-1.92, 0.88), abs=0.005)
    # the other coordinates
    assert -12_000 / 80_000 == car.elastic_centre() == -0.15
    assert 1.69 / (1.2 * 1.5) == pytest.approx(0.939, abs=5e-4) and car.dynamic_index == pytest.approx(0.939, abs=5e-4)
    assert 1200 * 1.5 / 2.7 == pytest.approx(667, abs=0.5) and car.end_masses == pytest.approx((666.7, 533.3), abs=0.05)
    assert car.end_frequencies() == pytest.approx((1.23, 1.38), abs=0.005)
    assert (math.sqrt(40e3 / 667) / (2 * math.pi), math.sqrt(40e3 / 533) / (2 * math.pi)) == pytest.approx(
        (1.23, 1.38), abs=0.005
    )
    # Olley: 30 % more static deflection at the front is a ride frequency 12 % lower
    assert 1 / math.sqrt(1.3) == pytest.approx(0.88, abs=0.005) and 1 - 1 / math.sqrt(1.3) == pytest.approx(
        0.12, abs=0.005
    )
    # the equal double pendulum: (ω²L/g)² − 4(ω²L/g) + 2 = 0
    assert np.sort(np.roots([1, -4, 2])) == pytest.approx((2 - math.sqrt(2), 2 + math.sqrt(2)))
    w2 = (2 - math.sqrt(2)) * 9.807 / 0.5, (2 + math.sqrt(2)) * 9.807 / 0.5
    assert w2 == pytest.approx((11.49, 66.97), abs=0.005)
    p = DoublePendulum()
    assert [md.frequency for md in pendulum_modes(p)] == pytest.approx((0.540, 1.30), abs=0.005)
    assert np.sqrt([11.49, 66.97]) / (2 * math.pi) == pytest.approx((0.540, 1.30), abs=0.005)
    assert p.separate_frequencies() == pytest.approx((0.705, 0.705), abs=0.0005)
