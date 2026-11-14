"""A rigid body's orientation and its equations of motion at one instant, in any axes.

Shared by the *Gyroscope* page and the *3D rigid body* page. An orientation is stored as a unit quaternion
q = (w, x, y, z), body to space (`matrix`: its rotation matrix, whose columns are the body's axes in space).
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
    axes in space."""
    w, x, y, z = (np.asarray(q, dtype=float)[..., i] for i in range(4))
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


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
