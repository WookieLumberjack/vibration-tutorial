"""core/rotation.py: quaternions to rotation matrices, and a snapshot of the equations in any axes."""

import math

import numpy as np

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
