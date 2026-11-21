"""core/rotation.py: quaternions to rotation matrices, and a snapshot of the equations in any axes."""

import math

import numpy as np
import pytest

from vib_tutorial.core import rotation


def test_a_quaternion_s_matrix():
    """A turn by χ about n is q = (cos χ/2, sin χ/2 n); its matrix is a rotation whose columns are the body's axes."""
    chi = math.radians(90.0)
    q = np.array([[math.cos(chi / 2), 0.0, 0.0, math.sin(chi / 2)]])  # a quarter turn about z
    R = rotation.matrix(q)[0]
    assert np.allclose(R @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), "x goes to y"
    rng = np.random.default_rng(1)
    qs = rng.normal(size=(50, 4))
    qs /= np.linalg.norm(qs, axis=1)[:, None]
    Rs = rotation.matrix(qs)
    assert np.allclose(np.einsum("nij,nkj->nik", Rs, Rs), np.eye(3))
    assert np.allclose(np.linalg.det(Rs), 1.0)
    assert np.allclose(rotation.matrix(-qs), Rs), "q and −q are the same turn"


def test_a_snapshot_in_fixed_axes_is_dl_dt_equals_tau():
    """A body turning steadily about a principal axis under no torque: in space L is fixed (rate 0, τ 0), and in
    the body's axes the transport term is ω × L = 0 too, so both sides are zero."""
    I = np.diag([1.0, 2.0, 3.0])
    w = 2.0
    t = np.linspace(0.0, 0.04, 5)
    body = np.stack(
        [[[math.cos(w * s), math.sin(w * s), 0], [-math.sin(w * s), math.cos(w * s), 0], [0, 0, 1]] for s in t]
    )
    L = np.tile([0.0, 0.0, 3.0 * w], (5, 1))
    tau = np.zeros((5, 3))
    space = np.broadcast_to(np.eye(3), (5, 3, 3))
    fixed = rotation.snapshot(t, body, space, L, tau, I, rotation.Turning.FIXED, 2)
    assert np.allclose(fixed.rate, 0.0) and np.allclose(fixed.omega, [0, 0, w])
    turning = rotation.snapshot(t, body, body, L, tau, I, rotation.Turning.BODY, 2)
    assert np.allclose(turning.inertia, I) and np.allclose(turning.carried, 0.0) and turning.mismatch < 1e-9


def test_euler_angles_there_and_back():
    """z-x-z angles to B and back; at θ = 0 the lock: φ and ψ only count as their sum."""
    for phi, theta, psi in ((0.3, 0.7, -1.2), (-2.5, 2.9, 0.4), (1.0, 0.01, 2.0)):
        B = rotation.from_euler(phi, theta, psi)
        assert np.allclose(B @ B.T, np.eye(3))
        back = rotation.to_euler(B)
        assert not back[3] and np.allclose(back[:3], (phi, theta, psi))
    locked = rotation.to_euler(rotation.from_euler(0.5, 0.0, 0.3))
    assert locked[3] and np.isclose(locked[0], 0.8) and locked[2] == 0.0
    assert np.allclose(rotation.from_euler(0.5, 0.0, 0.3), rotation.from_euler(0.2, 0.0, 0.6)), "only φ + ψ"


def test_euler_rates_match_the_angles_and_blow_up_at_the_lock():
    """The angles' rates from ω in the body's axes against differencing the angles along a turn; and 1/sin θ."""
    rng = np.random.default_rng(3)
    w = rng.normal(size=3)
    phi, theta, psi = 0.4, 1.1, -0.7
    B = rotation.from_euler(phi, theta, psi)
    h = 1e-6
    turn = np.eye(3) + h * np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])  # B(t + h) = B (1 + h[ω]×)
    after = rotation.to_euler(B @ turn)
    numeric = [(a - b) / h for a, b in zip(after[:3], (phi, theta, psi))]
    assert np.allclose(rotation.euler_rates(theta, psi, w), numeric, rtol=1e-4, atol=1e-4)
    small = rotation.euler_rates(1e-3, 0.0, np.array([0.0, 1.0, 0.0]))
    assert small[0] > 900, "φ̇ = ω₂/sin θ: unbounded as θ → 0"


def test_quaternions_and_their_axis_angle():
    rng = np.random.default_rng(4)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        B = rotation.matrix(q)
        back = rotation.quaternion(B)
        assert np.allclose(back, q if q[0] >= 0 else -q)
        chi, n = rotation.axis_angle(q)
        assert np.allclose(B @ n, n), "the axis doesn't move"
        assert np.isclose(np.trace(B), 1 + 2 * math.cos(chi)), "trace 1 + 2 cos χ"
        p = rng.normal(size=4)
        p /= np.linalg.norm(p)
        assert np.allclose(rotation.matrix(rotation.multiply(p, q)), rotation.matrix(p) @ B), "p ⊗ q is q, then p"


def test_a_homogeneous_transform_moves_points_and_turns_directions():
    B = rotation.from_euler(0.3, 0.5, 0.2)
    t = np.array([1.0, -2.0, 0.5])
    T = rotation.homogeneous(B, t)
    p = np.array([0.1, 0.2, 0.3])
    assert np.allclose(T @ np.append(p, 1.0), np.append(B @ p + t, 1.0))
    assert np.allclose(T @ np.append(p, 0.0), np.append(B @ p, 0.0))
    two = rotation.homogeneous(rotation.from_euler(-1.0, 0.2, 0.1), np.array([0.0, 1.0, 0.0]))
    assert np.allclose((two @ T)[:3, 3], two[:3, :3] @ t + two[:3, 3]), "composing is multiplying"


def test_the_views_project_z_up_or_along_a_shaft():
    """`project_z_up` (the top, the mounts, the rigid body) and `project_y_up` (the rotors), which each view had
    a copy of (#409 S-5): at no turn they're plan and side views, and a yaw brings the far side round as N-4 set."""
    import math

    from vib_tutorial.core.rotation import project_y_up, project_z_up

    p = np.array([[1.0, 2.0, 3.0]])
    x, y = project_z_up(p, 0.0, 0.0)
    assert x[0] == 1.0 and y[0] == 3.0  # z up, looking along y
    x, y = project_z_up(p, math.pi / 2, 0.0)
    assert x[0] == pytest.approx(-2.0) and y[0] == pytest.approx(3.0)
    x, y = project_z_up(p, 0.0, math.pi / 2)
    assert y[0] == pytest.approx(2.0), "from above: the depth shows as height"
    x, y = project_y_up(p, 0.0, 0.0)
    assert x[0] == 1.0 and y[0] == 2.0  # y up, the shaft (z) toward the viewer
    x, y = project_y_up(p, math.pi / 2, 0.0)
    assert x[0] == pytest.approx(3.0), "turned a quarter: the shaft runs across"
    x, y = project_y_up(p, 0.0, math.pi / 2)
    assert y[0] == pytest.approx(-3.0)
