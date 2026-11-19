"""A rigid body's orientation and its equations of motion at one instant, in any axes.

Shared by the *Gyroscope* page and the *3D rigid body* page. An orientation is stored as a unit quaternion
q = (w, x, y, z), body to space (`matrix`: its rotation matrix, whose columns are the body's axes in space).
The *3D rigid body* page's Orientation tab writes it four ways: the matrix, Euler's z-x-z angles (`to_euler`,
`from_euler`; their rates `euler_rates`, unbounded at the gimbal lock, sin θ = 0), the quaternion and its axis
and angle (`quaternion`, `axis_angle`; `multiply` composes turns), and a 4 × 4 homogeneous transform with the
body's origin (`homogeneous`).
A frame's axes are given as rows, (n, 3, 3): row i is axis i in space, so a space vector's components in
the frame are `axes @ v`.

`snapshot` writes the equations at one sample: the body's axes in space B, the frame's A, the inertia tensor in
the frame's axes I = (AᵀB) I_body (AᵀB)ᵀ, L = Iω, and the transport theorem dL/dt|axes + Ω × L = τ, Ω the rate
the frame's axes turn: zero for fixed axes, ω for axes fixed in the body (Euler's equations, I constant), and
whatever the page's frame does otherwise. dL/dt is the fourth-order central difference of L's components in the
frame over the samples k ± 2, so `Snapshot.mismatch` is what the simulation's step error leaves.

Working on it: `tests/test_rotation.py`, and the pages' snapshot tests (`test_gyroscope.py`,
`test_rigid_body.py`).
"""

from __future__ import annotations

import enum
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


class Turning(enum.Enum):
    """How a frame's axes turn: fixed in space, fixed in the body (Ω = ω), or some other way."""

    FIXED = "fixed"
    BODY = "body"
    OTHER = "other"


def matrix(q: np.ndarray) -> np.ndarray:
    """The rotation matrices (n, 3, 3) of unit quaternions q (n, 4), body to space: their columns are the body's
    axes in space; one quaternion (4,) gives one (3, 3). One is worked out in floats (the rigid body's step asks
    for it at every RK4 stage: 2 µs, where four stacked arrays took 22; #409 P-18), many filled in place."""
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        w, x, y, z = (float(c) for c in q)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
    qw, qx, qy, qz = (q[..., i] for i in range(4))
    R = np.empty((*q.shape[:-1], 3, 3))
    R[..., 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[..., 0, 1] = 2 * (qx * qy - qw * qz)
    R[..., 0, 2] = 2 * (qx * qz + qw * qy)
    R[..., 1, 0] = 2 * (qx * qy + qw * qz)
    R[..., 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[..., 1, 2] = 2 * (qy * qz - qw * qx)
    R[..., 2, 0] = 2 * (qx * qz - qw * qy)
    R[..., 2, 1] = 2 * (qy * qz + qw * qx)
    R[..., 2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return R


LOCK = 1e-6  # sin θ below this: the Euler angles' φ and ψ turn about the same axis (gimbal lock)


def about(axis: int, angle: float) -> np.ndarray:
    """The rotation matrix of a turn by `angle` about space's axis `axis` (0, 1, 2: x, y, z)."""
    c, s = math.cos(angle), math.sin(angle)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R = np.eye(3)
    R[i, i] = R[j, j] = c
    R[i, j], R[j, i] = -s, s
    return R


def from_euler(phi: float, theta: float, psi: float) -> np.ndarray:
    """B from Euler's angles, z-x-z (the Gyroscope page's): precession φ about z, tilt θ about the line of nodes,
    spin ψ about the body's own axis, B = R_z(φ) R_x(θ) R_z(ψ)."""
    return about(2, phi) @ about(0, theta) @ about(2, psi)


def to_euler(B: np.ndarray) -> tuple[float, float, float, bool]:
    """φ, θ, ψ (rad; θ in [0, π]) from B, and whether they're locked: there sin θ = 0, φ and ψ turn about the same
    axis, only their sum (θ = 0) or difference (θ = π) is set, and ψ is taken as 0."""
    theta = math.acos(max(-1.0, min(1.0, float(B[2, 2]))))
    if math.sin(theta) < LOCK:
        return math.atan2(float(B[1, 0]), float(B[0, 0])), theta, 0.0, True
    phi = math.atan2(float(B[0, 2]), -float(B[1, 2]))
    psi = math.atan2(float(B[2, 0]), float(B[2, 1]))
    return phi, theta, psi, False


def euler_rates(theta: float, psi: float, w: np.ndarray) -> tuple[float, float, float]:
    """φ̇, θ̇, ψ̇ from ω in the body's axes: φ̇ = (ω₁ sin ψ + ω₂ cos ψ)/sin θ, θ̇ = ω₁ cos ψ − ω₂ sin ψ,
    ψ̇ = ω₃ − φ̇ cos θ. φ̇ and ψ̇ grow without bound as sin θ → 0: the lock (inf there)."""
    w1, w2, w3 = (float(v) for v in w)
    s = math.sin(theta)
    across = w1 * math.sin(psi) + w2 * math.cos(psi)
    phi_dot = across / s if abs(s) > LOCK else math.copysign(math.inf, across) if across else 0.0
    return phi_dot, w1 * math.cos(psi) - w2 * math.sin(psi), w3 - phi_dot * math.cos(theta)


def quaternion(B: np.ndarray) -> np.ndarray:
    """The unit quaternion (w, x, y, z), w ≥ 0, of a rotation matrix (Shepperd's: from its largest term)."""
    t = float(np.trace(B))
    terms = [t, float(B[0, 0]), float(B[1, 1]), float(B[2, 2])]
    k = int(np.argmax(terms))
    if k == 0:
        w = 0.5 * math.sqrt(1.0 + t)
        q = [w, (B[2, 1] - B[1, 2]) / (4 * w), (B[0, 2] - B[2, 0]) / (4 * w), (B[1, 0] - B[0, 1]) / (4 * w)]
    else:
        i = k - 1
        j, m = (i + 1) % 3, (i + 2) % 3
        r = 0.5 * math.sqrt(max(1.0 + 2.0 * float(B[i, i]) - t, 0.0))
        q = [0.0] * 4
        q[0] = (B[m, j] - B[j, m]) / (4 * r)
        q[1 + i] = r
        q[1 + j] = (B[j, i] + B[i, j]) / (4 * r)
        q[1 + m] = (B[m, i] + B[i, m]) / (4 * r)
    out = np.array(q, dtype=float)
    return -out if out[0] < 0 else out


def multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton's product p ⊗ q: the turn q, then p (as matrix(p ⊗ q) = matrix(p) matrix(q))."""
    a, b, c, d = p
    e, f, g, h = q
    return np.array(
        [
            a * e - b * f - c * g - d * h,
            a * f + b * e + c * h - d * g,
            a * g - b * h + c * e + d * f,
            a * h + b * g - c * f + d * e,
        ]
    )


def axis_angle(q: np.ndarray) -> tuple[float, np.ndarray]:
    """The angle χ (0 to π, as w ≥ 0 after choosing q or −q) and unit axis n of the turn q = (cos χ/2, sin χ/2 n);
    any axis for no turn (x)."""
    q = -q if q[0] < 0 else q
    half = math.atan2(float(np.linalg.norm(q[1:])), float(q[0]))
    s = math.sin(half)
    return 2.0 * half, (q[1:] / s if s > 1e-12 else np.array([1.0, 0.0, 0.0]))


def homogeneous(B: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The 4 × 4 transform [[B, t], [0 0 0 1]]: a point p (in the body's axes, from its origin) is at B p + t in
    space; written with a 1 as its fourth component it is T (p, 1), and a direction (fourth component 0) only turns."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = B, t
    return T


@dataclass(frozen=True)
class Snapshot:
    """The equations of motion at one instant, in a frame's axes (all its vectors' components in them).

    `orientation` is B, the body's axes in space (its columns), and `frame` A, the frame's axes in space;
    `turn` = AᵀB is the body's axes in the frame's, so I = (AᵀB) I_body (AᵀB)ᵀ. L = Iω, and the rate of L's
    components in the frame's axes (`rate`, from the samples either side) plus Ω × L (`carried`: the axes turning
    under L) is the torque: the transport theorem, which in the body's axes (Ω = ω) is Euler's."""

    t: float
    kind: object  # the page's frame
    turning: Turning
    orientation: np.ndarray  # B (3, 3)
    frame: np.ndarray  # A (3, 3)
    body_inertia: np.ndarray  # in the body's axes
    inertia: np.ndarray  # in the frame's axes
    omega: np.ndarray
    momentum: np.ndarray
    rate: np.ndarray  # d/dt of L's components in these axes
    frame_rate: np.ndarray  # Ω
    torque: np.ndarray

    @property
    def turn(self) -> np.ndarray:
        """AᵀB: the body's axes in the frame's (columns)."""
        return self.frame.T @ self.orientation

    @property
    def carried(self) -> np.ndarray:
        """Ω × L: how much L's components change only because the axes turn under it."""
        return np.cross(self.frame_rate, self.momentum)

    @property
    def mismatch(self) -> float:
        """How far rate + Ω × L is from τ, over the larger of the three (0 is the equation held exactly)."""
        lhs = self.rate + self.carried
        scale = max(np.abs(lhs).max(), np.abs(self.torque).max(), np.abs(self.carried).max(), 1e-300)
        return float(np.abs(lhs - self.torque).max() / scale)


def snapshot(
    t: np.ndarray,
    body: np.ndarray,
    axes: np.ndarray,
    momentum: np.ndarray,
    torque: np.ndarray,
    body_inertia: np.ndarray,
    turning: Turning,
    k: int,
    kind: object = None,
    frame_rate: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Snapshot:
    """The equations at sample `k` of a run's samples (consecutive and evenly spaced): `body` and `axes` the body's
    and the frame's axes as rows (n, 3, 3), L and τ in space (n, 3), I_body in the body's axes. A frame that turns
    neither with space nor with the body gives `frame_rate`, Ω in space (1, 3) from ω in space (1, 3). The rate
    of L's components is the fourth-order central difference over k ± 2 (the body's components swing at the
    spin, and k ± 1 alone missed by ~1 % on the gyroscope's rectangle), k ± 1 or one-sided nearer the ends."""
    n = len(t)
    reach = min(2, k, n - 1 - k)
    lo, hi = (k - reach, k + reach) if reach else (max(k - 1, 0), min(k + 1, n - 1))
    parts = np.einsum("nij,nj->ni", axes[lo : hi + 1], momentum[lo : hi + 1])
    if reach == 2:
        rate = (8.0 * (parts[3] - parts[1]) - (parts[4] - parts[0])) / (3.0 * (t[hi] - t[lo]))
    else:
        rate = (parts[-1] - parts[0]) / (t[hi] - t[lo]) if hi > lo else np.zeros(3)
    B, A = body[k].T, axes[k].T  # columns
    I_body = np.asarray(body_inertia, dtype=float)
    omega = B @ np.linalg.solve(I_body, B.T @ momentum[k])  # in space
    if turning is Turning.FIXED:
        spin = np.zeros(3)
    elif turning is Turning.BODY:
        spin = omega
    else:
        assert frame_rate is not None, "a frame that turns its own way says how"
        spin = frame_rate(omega[None])[0]
    turn = A.T @ B
    return Snapshot(
        t=float(t[k]),
        kind=kind,
        turning=turning,
        orientation=B,
        frame=A,
        body_inertia=I_body,
        inertia=turn @ I_body @ turn.T,
        omega=A.T @ omega,
        momentum=A.T @ momentum[k],
        rate=rate,
        frame_rate=A.T @ spin,
        torque=A.T @ torque[k],
    )
