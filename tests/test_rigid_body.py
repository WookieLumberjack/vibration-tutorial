"""core/rigid_body.py: the inertia tensor of a shape and a point mass, its principal axes, and the body tumbling."""

import numpy as np
import pytest

from vib_tutorial.core import rotation
from vib_tutorial.core.rigid_body import Body, Shape, Torque, Tumbler

BOX = Body()  # 0.3 × 0.2 × 0.1 m, 1 kg: three different moments


def _points(body: Body, n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """The body as a cloud of equal point masses (a grid over the shape) plus the point mass: positions and masses."""
    g = (np.arange(n) + 0.5) / n - 0.5
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    if body.shape is Shape.CUBOID:
        P = np.column_stack([X.ravel() * body.a, Y.ravel() * body.b, Z.ravel() * body.c])
    elif body.shape is Shape.CYLINDER:
        P = np.column_stack([2 * body.a * X.ravel(), 2 * body.a * Y.ravel(), body.c * Z.ravel()])
        P = P[np.hypot(P[:, 0], P[:, 1]) <= body.a]
    else:
        P = 2 * body.a * np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        P = P[np.linalg.norm(P, axis=1) <= body.a]
    m = np.full(len(P), body.mass / len(P))
    return np.vstack([P, body.offset]), np.append(m, body.point)


@pytest.mark.parametrize(
    "body",
    [
        BOX,
        Body(Shape.CYLINDER, 2.0, a=0.1, c=0.4),
        Body(Shape.SPHERE, 3.0, a=0.15),
        Body(point=0.3, offset=(0.1, 0.08, -0.03)),
        Body(Shape.CYLINDER, 2.0, a=0.1, c=0.4, point=0.5, offset=(0.1, 0.0, 0.15)),
    ],
)
def test_the_inertia_tensor_is_the_sum_over_the_mass(body):
    """The formulas (the shape's own, parallel axes, the point mass's) against a sum over a grid of point masses:
    Σm(|r|²1 − r rᵀ) about the centre of mass, products of inertia and all."""
    P, m = _points(body)
    G = (m[:, None] * P).sum(axis=0) / m.sum()
    assert np.allclose(body.centre(), G, atol=1e-4)
    r = P - G
    summed = (m[:, None, None] * ((r * r).sum(axis=1)[:, None, None] * np.eye(3) - r[:, :, None] * r[:, None, :])).sum(
        0
    )
    assert np.allclose(body.inertia(), summed, rtol=0.0, atol=3e-3 * np.abs(summed).max())


def test_the_standard_moments():
    m = 2.0
    assert np.allclose(np.diag(Body(Shape.CUBOID, m, 0.3, 0.2, 0.1).inertia()), m * np.array([0.05, 0.1, 0.13]) / 12)
    assert np.allclose(
        np.diag(Body(Shape.CYLINDER, m, a=0.1, c=0.4).inertia()), [m * (0.03 + 0.16) / 12] * 2 + [m * 0.01 / 2]
    )
    assert np.allclose(Body(Shape.SPHERE, m, a=0.1).inertia(), 0.4 * m * 0.01 * np.eye(3))


def test_the_principal_axes():
    """With the point mass off the axes the tensor has products of inertia; P diag Pᵀ gives it back, P is a
    rotation, the moments ascend, and the principal axes have none."""
    body = Body(point=0.3, offset=(0.1, 0.08, -0.03))
    I = body.inertia()
    assert abs(I[0, 1]) > 1e-4, "products of inertia"
    moments, P = body.principal()
    assert np.all(np.diff(moments) > 0)
    assert np.allclose(P @ np.diag(moments) @ P.T, I)
    assert np.allclose(P.T @ P, np.eye(3)) and np.isclose(np.linalg.det(P), 1.0)
    assert np.allclose(P.T @ I @ P, np.diag(moments))


def test_torque_free_l_and_the_energy_are_kept():
    tm = Tumbler(BOX, (1.0, 5.0, 2.0))
    L0, E0 = np.array(tm.L), tm.energy()
    s = tm.advance(10.0)
    assert np.allclose(s.momentum, L0, atol=1e-15), "no torque: L is fixed exactly"
    assert abs(tm.energy() - E0) < 1e-7 * E0
    assert np.allclose(np.linalg.norm(s.q, axis=1), 1.0)


@pytest.mark.parametrize("axis, flips", [(0, False), (1, True), (2, False)])
def test_the_intermediate_axis_flips(axis, flips):
    """Spun about the least or the greatest moment's axis, a nudge stays small; about the middle one it grows,
    and the body turns over (the tennis-racket, or Dzhanibekov, effect) and back."""
    omega = [0.05, 0.05, 0.05]
    omega[axis] = 6.0
    tm = Tumbler(BOX, tuple(omega))
    s = tm.advance(12.0)
    along = rotation.matrix(s.q)[:, axis, axis]  # the spin axis's component along where it started
    assert (along.min() < -0.9) == flips, along.min()


@pytest.mark.parametrize("held", [Torque.BODY, Torque.SPACE])
def test_a_kick_changes_l_by_its_impulse(held):
    """A torque held in space for T changes L by τT exactly; one held in the body about a principal axis it spins
    about does too (it doesn't turn)."""
    tm = Tumbler(BOX, (0.0, 0.0, 3.0), torque=(0.0, 0.0, 0.02), held=held)
    tm.kick(0.5)
    L0 = np.array(tm.L)
    s = tm.advance(1.0)
    assert np.allclose(np.array(tm.L) - L0, [0.0, 0.0, 0.01], atol=1e-12)
    on = s.t <= 0.5 + 1e-12  # a sample's torque is what acted over the step ending at it
    assert np.allclose(s.torque[on], [0.0, 0.0, 0.02]) and np.allclose(s.torque[~on], 0.0)


def test_a_body_held_torque_turns_with_it():
    """A torque held in the body's axes, across a spin, turns with the body, so L wobbles round rather than
    growing steadily one way (as a space-held push does)."""
    pushes = {}
    for held in (Torque.BODY, Torque.SPACE):
        tm = Tumbler(BOX, (0.0, 0.0, 6.0), torque=(0.01, 0.0, 0.0), held=held)
        tm.kick(2.0)
        tm.advance(2.5)  # past the push's end (a run stops a part-step short)
        pushes[held] = abs(tm.L[0])
    assert pushes[Torque.SPACE] == pytest.approx(0.02, rel=1e-9)
    assert pushes[Torque.BODY] < 0.2 * pushes[Torque.SPACE]


def test_a_snapshot_of_the_tumbling_body():
    """In space, in the shape's axes (products of inertia), and in the principal axes (none): the equation holds
    to the step error."""
    body = Body(point=0.3, offset=(0.1, 0.08, -0.03))
    tm = Tumbler(body, (1.0, 4.0, 0.5), torque=(0.0, 0.01, 0.0), held=Torque.BODY)
    tm.kick(5.0)
    s = tm.advance(2.0)
    B = np.transpose(rotation.matrix(s.q), (0, 2, 1))  # rows: the body's axes
    _, P = body.principal()
    I = body.inertia()
    for axes, turning in (
        (np.broadcast_to(np.eye(3), B.shape), rotation.Turning.FIXED),
        (B, rotation.Turning.BODY),
        (np.einsum("ji,njk->nik", P, B), rotation.Turning.BODY),
    ):
        for k in range(3, len(s.t) - 3, 37):
            snap = rotation.snapshot(s.t, B, axes, s.momentum, s.torque, I, turning, k)
            assert snap.mismatch < 1e-4, snap.mismatch
    principal = rotation.snapshot(
        s.t, B, np.einsum("ji,njk->nik", P, B), s.momentum, s.torque, I, rotation.Turning.BODY, 10
    )
    assert np.allclose(principal.inertia, np.diag(np.diag(principal.inertia)), atol=1e-12)


def test_a_start_turned_away_keeps_omega_in_the_body():
    """Started turned by q, ω is still the one asked for in the body's axes, and L = B I ω in space."""
    q = rotation.quaternion(rotation.from_euler(0.4, 1.0, -0.3))
    tm = Tumbler(BOX, (1.0, 2.0, 3.0), q=list(q))
    assert np.allclose(tm.body_rate(), [1.0, 2.0, 3.0])
    assert np.allclose(tm.L, rotation.matrix(q) @ (BOX.inertia() @ [1.0, 2.0, 3.0]))
