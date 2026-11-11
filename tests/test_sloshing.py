"""Sloshing: the frequencies and the rigid mass against a finite-volume solution of the tank's potential flow, the
analogue's masses, the damper's elimination against a full matrix solve, and Den Hartog's peak."""

import math

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import splu

from vib_tutorial.core.beams_plates import TWO_PI
from vib_tutorial.core.sloshing import (
    Damper,
    G,
    Structure,
    Tank,
    damper_response,
    den_hartog,
    effective_mass,
    frozen_response,
    loaded_frequency,
    mass_ratio,
    peak,
    peak_against_depth,
    rigid_mass,
    slosh_frequency,
    slosh_mass,
    tank_flow,
)


def _energy_matrix(L: float, h: float, nx: int, nz: int) -> sparse.csc_matrix:
    """½∫|∇φ|² = ½ φᵀKφ on nx × nz cells (cell-centred finite volumes, row j of cells at index j·nx + i), with
    nx more unknowns for φ on the surface, half a cell above the top row."""
    dx, dz = L / nx, h / nz
    n = nx * nz
    cell = np.arange(n).reshape(nz, nx)
    pairs = [
        (cell[:, :-1].ravel(), cell[:, 1:].ravel(), dz / dx),  # across x
        (cell[:-1].ravel(), cell[1:].ravel(), dx / dz),  # across z
        (cell[-1], n + np.arange(nx), 2 * dx / dz),  # the top row to the surface
    ]
    rows, cols, vals = [], [], []
    for a, b, c in pairs:
        for r, q, v in ((a, a, c), (b, b, c), (a, b, -c), (b, a, -c)):
            rows.append(r)
            cols.append(q)
            vals.append(np.full(r.size, v))
    return sparse.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n + nx, n + nx)
    )


def _frequencies(tank: Tank, nx: int, nz: int) -> np.ndarray:
    """The surface's eigenproblem ∂φ/∂z = (ω²/g)φ through the flow's Dirichlet-to-Neumann map (the Schur
    complement on the surface's unknowns)."""
    K = _energy_matrix(tank.length, tank.depth, nx, nz)
    n = nx * nz
    A, C, D = K[:n, :n], K[:n, n:].toarray(), K[n:, n:].toarray()
    S = D - C.T @ splu(A).solve(C)
    lam = np.sort(np.linalg.eigvalsh(S / (tank.length / nx)))[1:4]  # the first is the still surface, 0
    return np.sqrt(G * lam) / TWO_PI


def _rigid(tank: Tank, nx: int, nz: int) -> float:
    """The tank moving at U = 1 with its surface held flat (φ = 0): 2 × the flow's kinetic energy."""
    K = _energy_matrix(tank.length, tank.depth, nx, nz)
    n = nx * nz
    dz = tank.depth / nz
    b = np.zeros(n)
    b[nx - 1 :: nx] += dz  # ∂φ/∂n = U on the far wall
    b[::nx] -= dz  # and −U on the near
    phi = splu(K[:n, :n]).solve(b)
    return tank.fluid * tank.breadth * float(phi @ b)


@pytest.mark.parametrize("depth", [0.2, 0.5, 1.2])
def test_frequencies_against_the_flow(depth):
    tank = Tank(1.0, depth)
    coarse, fine = _frequencies(tank, 40, 30), _frequencies(tank, 80, 60)
    extrapolated = (4 * fine - coarse) / 3  # second order
    exact = [slosh_frequency(tank, n) for n in (1, 2, 3)]
    assert extrapolated == pytest.approx(exact, rel=2e-4)


@pytest.mark.parametrize("depth", [0.5, 1.0])
def test_rigid_mass_against_the_flow(depth):
    """About first order in square cells (the corners where the moving wall meets the held surface are singular,
    and a shallow tank's converges more slowly)."""
    tank = Tank(1.0, depth)
    n = round(40 * depth)
    coarse, fine = _rigid(tank, 40, n), _rigid(tank, 80, 2 * n)
    assert 2 * fine - coarse == pytest.approx(rigid_mass(tank), rel=3e-3)


def test_the_analogues_masses():
    for h in (0.01, 0.3, 1.0, 5.0):
        tank = Tank(2.0, h, 1.5)
        odd = np.arange(1, 4000, 2)
        assert rigid_mass(tank) + float(np.sum(slosh_mass(tank, odd))) == pytest.approx(tank.liquid_mass)
        assert rigid_mass(tank) > 0
    assert float(slosh_mass(Tank(), 2)) == 0
    # shallow: nearly all of it sloshes, the first mode 8/π² of it
    shallow = Tank(1.0, 0.001)
    assert float(slosh_mass(shallow, 1)) / shallow.liquid_mass == pytest.approx(8 / math.pi**2, rel=1e-4)
    assert rigid_mass(shallow) / shallow.liquid_mass < 2e-3
    # deep: the first mode is the top 8L/π³ of it
    deep = Tank(1.0, 10.0)
    assert float(slosh_mass(deep, 1)) == pytest.approx(deep.fluid * 8 / math.pi**3)


def test_effective_mass():
    tank = Tank(1.0, 0.5)
    assert abs(effective_mass(tank, np.array([1e-4]))[0]) == pytest.approx(tank.liquid_mass, rel=1e-6)
    assert abs(effective_mass(tank, np.array([1e3]))[0]) == pytest.approx(rigid_mass(tank), rel=1e-3)
    # an even mode's frequency is no resonance
    f2 = slosh_frequency(tank, 2)
    assert np.isfinite(abs(effective_mass(tank, np.array([f2]))[0]))


def test_tank_flow():
    tank = Tank(1.0, 0.4)
    x = np.linspace(0, 1, 41)
    z = np.linspace(-0.4, 0, 21)
    f = tank_flow(tank, 1, x, z)
    assert f.speed.max() == pytest.approx(1.0)
    assert np.allclose(f.stream[:, [0, -1]], 0) and np.allclose(f.stream[0], 0)  # walls and bottom
    assert f.surface[[0, -1]] == pytest.approx([1, -1])
    # a deep tank's flow doesn't overflow
    assert np.all(np.isfinite(tank_flow(Tank(0.1, 50.0), 9, x * 0.1, np.linspace(-50, 0, 21)).speed))


def _full_response(structure: Structure, damper: Damper, f: np.ndarray) -> np.ndarray:
    """The structure and each sloshing mass as degrees of freedom: (K + iωC − ω²M) x = F e₁."""
    m = damper.masses
    k = m * damper.omegas**2
    c = 2 * damper.zeta * m * damper.omegas
    ws = TWO_PI * structure.frequency
    ks, cs = structure.mass * ws**2, 2 * structure.zeta * structure.mass * ws
    n = len(m) + 1
    M = np.diag([structure.mass + damper.rigid, *m])
    K = np.zeros((n, n))
    C = np.zeros((n, n))
    K[0, 0], C[0, 0] = ks, cs
    for j in range(len(m)):
        for mat, v in ((K, k[j]), (C, c[j])):
            mat[0, 0] += v
            mat[j + 1, j + 1] += v
            mat[0, j + 1] -= v
            mat[j + 1, 0] -= v
    out = []
    for w in TWO_PI * f:
        x = np.linalg.solve(K + 1j * w * C - w**2 * M, np.eye(n)[0])
        out.append(x[0] * ks)
    return np.array(out)


def test_damper_against_a_matrix_solve():
    structure = Structure(20000, 0.5, 0.01)
    damper = Damper(Tank(2.0, 0.4, 2.0), 2, 0.08)
    f = np.linspace(0.2, 1.5, 300)
    assert damper_response(structure, damper, f) == pytest.approx(_full_response(structure, damper, f), rel=1e-9)
    # no damper, and a frozen one
    assert abs(damper_response(structure, None, np.array([0.5]))[0]) == pytest.approx(50.0)
    frozen = frozen_response(structure, damper, np.array([0.0]))[0]
    assert frozen == pytest.approx(1.0)


def test_den_hartog():
    """A first-mode damper tuned and damped by Den Hartog's rule on an undamped structure: its peak is close to
    √(1 + 2/μ) (the rule is the fixed points' optimum, not the exact one)."""
    structure = Structure(20000, 0.5, 0.0)
    for count in (1, 3):
        # find the depth that tunes it, for the masses it gives
        depths = np.linspace(0.05, 1.0, 400)
        tuned = None
        for h in depths:
            d = Damper(Tank(2.0, float(h), 2.0), count, 0.1, modes=1)
            ratio, zeta, _ = den_hartog(mass_ratio(structure, d))
            if slosh_frequency(d.tank, 1) >= ratio * loaded_frequency(structure, d):
                tuned = Damper(d.tank, count, zeta, modes=1)
                break
        assert tuned is not None
        _, _, top = den_hartog(mass_ratio(structure, tuned))
        assert peak(structure, tuned, 20000) == pytest.approx(top, rel=0.03)


def test_peak_against_depth_has_a_minimum_where_tuned():
    structure = Structure(20000, 0.5, 0.01)
    damper = Damper(Tank(2.0, 0.4, 2.0), 1, 0.13)
    depths = np.linspace(0.1, 1.0, 46)
    peaks = peak_against_depth(structure, damper, depths)
    best = float(depths[np.argmin(peaks)])
    assert 0.3 < best < 0.5
    assert peaks[0] > 2 * peaks.min() and peaks[-1] > 2 * peaks.min()
