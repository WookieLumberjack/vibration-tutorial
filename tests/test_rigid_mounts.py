"""A rigid body on mounts: K against closed forms, the elastic centre, focalizing, and the trials' numbers."""

import math

import numpy as np
import pytest

from vib_tutorial.core.rigid_mounts import (
    TWO_PI,
    Body,
    Layout,
    Mount,
    elastic_centre,
    modes,
    plane_centres,
    stiffness_matrix,
)

LEVEL = Layout()


def test_stiffness_is_symmetric_and_the_mount_follows_the_body():
    """K = Σ Tᵀ K_m T: symmetric, and its energy for a small rigid motion is each mount's ½ dᵀ K_m d."""
    layout = Layout(drop=0.08, offset=0.03, tilt=math.radians(20))
    K = stiffness_matrix(layout.mounts())
    assert K == pytest.approx(K.T)
    q = np.array([0.001, -0.002, 0.0005, 0.003, -0.001, 0.002])
    energy = 0.0
    for m in layout.mounts():
        d = q[:3] + np.cross(q[3:], m.position)  # the mount's end
        energy += 0.5 * d @ m.stiffness() @ d
    assert 0.5 * q @ K @ q == pytest.approx(energy)


def test_level_mounts_have_pure_modes_in_closed_form():
    m = LEVEL.modes()
    b, k, ks = LEVEL.body, LEVEL.stiffness, LEVEL.shear_ratio * LEVEL.stiffness
    s, t = LEVEL.span, LEVEL.track
    i_xx, i_yy, i_zz = b.inertia
    expected = {
        0: math.sqrt(4 * ks / b.mass),
        1: math.sqrt(4 * ks / b.mass),
        2: math.sqrt(4 * k / b.mass),
        3: math.sqrt(4 * k * t * t / i_xx),
        4: math.sqrt(4 * k * s * s / i_yy),
        5: math.sqrt(4 * ks * (s * s + t * t) / i_zz),
    }
    assert np.allclose(m.energy.max(axis=0), 1.0)
    for j in range(6):
        assert m.frequencies[j] == pytest.approx(expected[m.dominant(j)] / TWO_PI)
    assert m.energy.sum(axis=0) == pytest.approx(np.ones(6))


def test_a_coupled_pair():
    """Sway and roll alone: the two frequencies' product and sum are the 2 × 2 problem's invariants."""
    layout = Layout(drop=0.08)
    m = layout.modes()
    K = stiffness_matrix(layout.mounts())[np.ix_([1, 3], [1, 3])]
    M = np.diag([layout.body.mass, layout.body.inertia[0]])
    pair = sorted(j for j in range(6) if m.dominant(j) in (1, 3))
    w2 = (TWO_PI * m.frequencies[pair]) ** 2
    assert w2.prod() == pytest.approx(np.linalg.det(K) / np.linalg.det(M))
    assert w2.sum() == pytest.approx(np.trace(np.linalg.solve(M, K)))


def test_the_elastic_centre():
    """Upright mounts alike: in their plane, under their middle; about it the coupling vanishes."""
    for layout, expected in ((Layout(drop=0.08), (0, 0, -0.08)), (Layout(offset=0.05), (-0.05, 0, 0))):
        point, left = elastic_centre(layout.mounts())
        assert point == pytest.approx(expected, abs=1e-12) and left < 1e-12
        assert np.abs(stiffness_matrix(layout.mounts(), point)[:3, 3:]).max() < 1e-9
    # mounts placed anyhow: the least-squares point still clears what can be cleared
    odd = (
        Mount((0.2, 0.1, -0.05), 4e4, 1e4, (0.0, -0.3, 0.954)),
        Mount((-0.25, 0.15, -0.1), 6e4, 2e4),
        Mount((0.1, -0.2, 0.0), 5e4, 1.5e4, (0.2, 0.0, 0.98)),
    )
    point, left = elastic_centre(odd)
    nearby = [elastic_centre_residual(odd, point + d) for d in 0.01 * np.eye(3)]
    assert all(r >= elastic_centre_residual(odd, point) - 1e-12 for r in nearby) and 0 < left < 1


def elastic_centre_residual(mounts, point):
    return float(np.linalg.norm(stiffness_matrix(mounts, point)[:3, 3:]))


@pytest.mark.parametrize("tilt", [0.0, 10.0, 25.0, 40.0])
def test_the_roll_centre_in_closed_form(tilt):
    layout = Layout(drop=0.08, tilt=math.radians(tilt))
    roll, _ = plane_centres(layout.mounts())
    assert roll + layout.drop == pytest.approx(layout.roll_centre(), abs=1e-12)


def test_focusing():
    """At the focal tilt sway and roll are pure again; past h = t(1 − ρ)/2√ρ there is none; a mount as stiff
    across as along can't be focused."""
    layout = Layout(drop=0.08)
    beta = layout.focus_tilt()
    focused = Layout(drop=0.08, tilt=beta)
    assert focused.roll_centre() == pytest.approx(0.08)
    m = focused.modes()
    assert m.energy[1].max() > 0.9999 and m.energy[3].max() > 0.9999
    u = math.tan(beta)
    h, t, rho = 0.08, layout.track, layout.shear_ratio
    assert h * u * u - t * (1 - rho) * u + h * rho == pytest.approx(0.0, abs=1e-12)
    limit = t * (1 - rho) / (2 * math.sqrt(rho))
    assert math.isfinite(Layout(drop=0.99 * limit).focus_tilt())
    assert math.isnan(Layout(drop=1.01 * limit).focus_tilt())
    for tilt in (0.0, 20.0, 40.0):
        stiff = Layout(drop=0.08, shear_ratio=1.0, tilt=math.radians(tilt))
        assert plane_centres(stiff.mounts())[0] == pytest.approx(-0.08, abs=1e-12)


def test_modes_are_m_orthonormal():
    layout = Layout(drop=0.06, offset=0.04, tilt=math.radians(12))
    m = modes(layout.body, layout.mounts())
    M = layout.body.mass_matrix()
    assert m.shapes.T @ M @ m.shapes == pytest.approx(np.eye(6), abs=1e-10)
    K = stiffness_matrix(layout.mounts())
    assert m.shapes.T @ K @ m.shapes == pytest.approx(np.diag((TWO_PI * m.frequencies) ** 2), rel=1e-9, abs=1e-6)


def test_box_inertia():
    b = Body(12.0, 0.3, 0.4, 0.5)
    assert b.inertia == pytest.approx((12 * (0.16 + 0.25) / 12, 12 * (0.09 + 0.25) / 12, 12 * (0.09 + 0.16) / 12))


def test_trials():
    from vib_tutorial.gui.rigid_mounts import FOCUS, TRIALS

    by = {t.name: t for t in TRIALS}

    def table(name):
        m = by[name].layout.modes()
        return {m.name(j): round(float(m.frequencies[j]), 2) for j in range(6)}, m

    level, _ = table("Level with the centre of mass: six pure modes")
    assert sorted(level.values()) == pytest.approx([7.96, 7.96, 8.47, 9.79, 15.92, 16.24], abs=0.006)
    dropped, _ = table("Mounts below the centre of mass: sway and roll couple")
    assert dropped["y 67 %, roll 33 %"] == pytest.approx(6.90) and dropped["roll 67 %, y 33 %"] == pytest.approx(9.77)
    assert dropped["x 99 %"] == pytest.approx(7.83)
    assert elastic_centre(by["Mounts below the centre of mass: sway and roll couple"].layout.mounts())[0][
        2
    ] == pytest.approx(-0.08)
    assert FOCUS == pytest.approx(16.95)
    focused, _ = table("Tilted to the focus: decoupled again")
    assert focused["roll 100 %"] == pytest.approx(7.56) and focused["y 100 %"] == pytest.approx(8.91, abs=0.015)
    offset, _ = table("Centre of mass off-centre: bounce and pitch couple")
    assert offset["z 61 %, pitch 39 %"] == pytest.approx(13.68, abs=0.015)
    assert offset["pitch 61 %, z 39 %"] == pytest.approx(18.89)
    assert offset["y 83 %, yaw 17 %"] == pytest.approx(7.30) and offset["yaw 83 %, y 17 %"] == pytest.approx(10.67)
    stiff = by["Stiff in shear: no focus to be had"].layout
    assert math.isnan(stiff.focus_tilt())
