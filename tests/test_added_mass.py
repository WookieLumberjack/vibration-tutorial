"""Added mass: the closed forms against the potential flow's kinetic energy, the plate's wavenumber integral against
a direct one in space and a piston's, and the wet frequencies."""

import itertools
import math

import numpy as np
import pytest
from scipy.special import j1

from vib_tutorial.core.added_mass import (
    WATER,
    Body,
    Shape,
    _plate_integral,
    _shape_power,
    avmi,
    axis_speed,
    flow,
    plate_added_mass,
    plate_dry_frequency,
    plate_modal_mass,
    plate_modes,
    potential,
    wet_frequency,
)
from vib_tutorial.core.beams_plates import Plate


def test_open_fluid_coefficients():
    assert Body(Shape.SPHERE).coefficient == 0.5
    assert Body(Shape.CYLINDER).coefficient == 1.0
    s = Body(Shape.SPHERE, 0.1, fluid=WATER)
    assert s.added_mass == pytest.approx(0.5 * 1000 * 4 / 3 * math.pi * 0.001)
    # a far wall changes nothing
    for shape in Shape:
        assert Body(shape, outer=100.0).coefficient == pytest.approx(Body(shape).coefficient, rel=1e-5)


def _kinetic(body: Body) -> float:
    """½ρ∫|∇φ|² over the fluid (U = 1), on a fine polar grid: 2D per length, or 3D about the axis."""
    R = body.radius
    Ro = body.outer if math.isfinite(body.outer) else 400 * R
    xs, ws = np.polynomial.legendre.leggauss(200)
    # r on a log scale (the flow decays as a power of r), θ over half the plane (the rest by symmetry)
    s = 0.5 * (xs + 1) * math.log(Ro / R)
    r = R * np.exp(s)
    wr = 0.5 * math.log(Ro / R) * ws * r
    th = 0.5 * math.pi * (xs + 1)
    wt = 0.5 * math.pi * ws
    rr, tt = np.meshgrid(r, th, indexing="ij")
    h = 1e-6 * R
    dphi_r = (potential(body, rr + h, tt) - potential(body, rr - h, tt)) / (2 * h)
    dphi_t = (potential(body, rr, tt + 1e-6) - potential(body, rr, tt - 1e-6)) / (2e-6 * rr)
    speed2 = dphi_r**2 + dphi_t**2
    if body.shape == Shape.SPHERE:
        area = 2 * math.pi * rr * np.sin(tt) * rr  # r dθ dr about the axis
        return 0.5 * body.fluid * float(wr @ (speed2 * area) @ wt)
    return 0.5 * body.fluid * 2 * float(wr @ (speed2 * rr) @ wt)


@pytest.mark.parametrize("shape", list(Shape))
@pytest.mark.parametrize("outer", [math.inf, 0.15, 0.06])
def test_added_mass_is_the_flows_kinetic_energy(shape, outer):
    body = Body(shape, 0.05, outer=outer)
    tolerance = 2e-3 if math.isinf(outer) else 1e-5  # open fluid: the grid stops at 400 R
    assert 2 * _kinetic(body) == pytest.approx(body.added_mass, rel=tolerance)


def test_confinement_raises_the_added_mass():
    """Fritz's (R_o² + R²)/(R_o² − R²): 5.55 for a gap a fifth of the radius."""
    assert Body(Shape.CYLINDER, 0.05, outer=0.06).coefficient == pytest.approx(5.545, abs=1e-3)
    c = [Body(Shape.CYLINDER, 0.05, outer=0.05 * r).coefficient for r in (3.0, 2.0, 1.5, 1.1)]
    assert np.all(np.diff(c) > 0)


def test_the_flow_meets_the_body_and_the_wall():
    for shape in Shape:
        body = Body(shape, 0.05, outer=0.1)
        ahead = axis_speed(body, np.array([0.05, 0.1]))
        assert ahead == pytest.approx([1.0, 0.0], abs=1e-12)
        open_ = Body(shape, 0.05)
        p = 3 if shape == Shape.SPHERE else 2
        assert axis_speed(open_, np.array([0.1]))[0] == pytest.approx(0.5**p)
    f = flow(Body(Shape.CYLINDER, 0.05), np.array([0.0, 0.07]), np.array([0.0, 0.07]))
    assert math.isnan(f.u[0, 0])  # inside the body
    # the stream function is constant along the velocity: u·∇ψ = 0, by finite differences
    x = np.linspace(0.06, 0.09, 31)
    for shape in Shape:
        g = flow(Body(shape, 0.05, outer=0.12), x, x)
        dpsi_y, dpsi_x = np.gradient(g.stream, x, x)
        along = g.u * dpsi_x + g.v * dpsi_y
        assert np.nanmax(np.abs(along[2:-2, 2:-2])) < 1e-3 * np.nanmax(np.hypot(dpsi_x, dpsi_y))


def test_wet_frequency():
    assert wet_frequency(10.0, 3.0, 1.0) == pytest.approx(10 * math.sqrt(0.75))


def test_shape_transform():
    """|∫ sin(mπx/a) e^{−ikx} dx|² against direct quadrature, through its removable pole."""
    a = 0.6
    x, w = np.polynomial.legendre.leggauss(200)
    xs = 0.5 * a * (x + 1)
    for m in (1, 2, 3):
        for k in (0.0, 3.0, m * math.pi / a, 40.0):
            direct = abs(np.sum(0.5 * a * w * np.sin(m * math.pi * xs / a) * np.exp(-1j * k * xs))) ** 2
            assert float(_shape_power(np.array([k]), m, a)[0]) == pytest.approx(direct, rel=1e-8, abs=1e-14)


def test_plate_integral_against_space():
    """(1/2π) ∫∫ φ(x)φ(x')/|x − x'|: on a midpoint grid (each cell's own 1/r in closed form), converging as 1/N."""
    a, b = 0.6, 0.4
    for m, n in ((1, 1), (2, 1)):
        values = []
        for N in (40, 80):
            x = (np.arange(N) + 0.5) * a / N
            y = (np.arange(N) + 0.5) * b / N
            X, Y = (g.ravel() for g in np.meshgrid(x, y, indexing="ij"))
            phi = np.sin(m * math.pi * X / a) * np.sin(n * math.pi * Y / b)
            dx, dy = a / N, b / N
            R = np.hypot(X[:, None] - X[None, :], Y[:, None] - Y[None, :])
            np.fill_diagonal(R, 1.0)
            G = dx * dy / R
            np.fill_diagonal(G, 2 * (dx * math.asinh(dy / dx) + dy * math.asinh(dx / dy)))
            values.append(float(phi @ G @ phi) * dx * dy / (2 * math.pi))
        extrapolated = 2 * values[1] - values[0]
        assert _plate_integral(a, b, m, n) == pytest.approx(extrapolated, rel=1e-3)


def test_a_circular_pistons_added_mass():
    """The same wavenumber integral for a baffled piston of radius a, Φ̂ = 2πa J₁(ka)/k, gives (8/3)ρa³."""
    a = 0.1
    x, w = np.polynomial.legendre.leggauss(64)
    edges = np.linspace(0, 4000 / a, 2001)
    k = np.concatenate([0.5 * (e1 - e0) * (x + 1) + e0 for e0, e1 in itertools.pairwise(edges)])
    wk = np.concatenate([0.5 * (e1 - e0) * w for e0, e1 in itertools.pairwise(edges)])
    F2 = (2 * math.pi * a * j1(k * a) / k) ** 2
    integral = float(np.sum(wk * F2 / k * 2 * math.pi * k)) / (4 * math.pi**2)
    assert integral == pytest.approx(8 / 3 * a**3, rel=1e-3)


def test_plate_in_water():
    plate = Plate()  # 600 × 400 × 3 mm steel
    beta = avmi(plate, 1, 1)
    assert beta == pytest.approx(plate_added_mass(plate, 1, 1) / plate_modal_mass(plate))
    assert avmi(plate, 1, 1, sides=2) == pytest.approx(2 * beta)
    assert avmi(Plate(thickness=0.006), 1, 1) == pytest.approx(beta / 2)  # ∝ 1/h
    modes = plate_modes(plate, 12)
    assert modes.wet == pytest.approx(modes.dry / np.sqrt(1 + modes.beta))
    assert modes.dry[0] == pytest.approx(plate_dry_frequency(plate, 1, 1))
    # higher modes carry less fluid
    assert avmi(plate, 3, 3) < avmi(plate, 2, 1) < beta


def test_streamlines_are_continuous_at_the_body_and_the_wall():
    """The body's rigid ψ meets the fluid's at R, and the fluid's is 0 at the wall."""
    for shape in Shape:
        for outer in (math.inf, 0.08):
            body = Body(shape, 0.05, outer=outer)
            th = np.linspace(0.1, 3.0, 7)
            for r in (0.05, 0.08) if math.isfinite(outer) else (0.05,):
                pts = r * np.array([1 - 1e-9, 1 + 1e-9])
                values = [
                    flow(body, np.array([p * math.cos(t)]), np.array([p * math.sin(t)])).contours[0, 0]
                    for t in th
                    for p in pts
                ]
                pairs = np.array(values).reshape(-1, 2)
                assert pairs[:, 0] == pytest.approx(pairs[:, 1], abs=1e-9)
