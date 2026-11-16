"""The exact first-order-hold (FOH) update of a linear state space, z' = A z + B f: every linear record's.

With f linear between samples h apart, the update is exact (`foh_discretize`):
z[k] = Φ z[k-1] + Γ₀ f[k-1] + Γ₁ f[k]. Diagonalized, Φ = V diag(μ) V⁻¹, it is one first-order recursive filter
per eigenvalue, η_r[k] = μ_r η_r[k-1] + (V⁻¹Γ₀ f[k-1] + V⁻¹Γ₁ f[k])_r, which `scipy.signal.lfilter` runs in C:
the virtual modal test's chain, the OMA and random-vibration records and the shock page's chain all step this
way. A defective Φ (a free chain's rigid-body λ = 0, at a small h) has no diagonal form to trust: then `FohSteps`
steps z itself.

A record from a given state, `respond` (and `free`, with no input), steps the update directly, in Rust
(`native.LinearSteps`, an input at a time, the states added) where it's built and in Python otherwise
(`respond_reference`): the One mass pages, the bounce–pitch car, the quarter-car's drive, the frame's loads, the
complex modes' free decays and the double pendulum's linear swing.

Working on it: keep `respond` and `respond_reference` the same update, sample for sample (`tests/test_native.py`
compares them), and `FohSteps` the same as stepping it (`tests/test_foh.py`). The chain's `Simulator` keeps its own
cached `LinearSteps`, as it steps a frame at a time with an energy ledger.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

from . import native
from .frf_matrix import MAX_EIGVEC_CONDITION


def foh_discretize(A: np.ndarray, B: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Phi, Gamma0, Gamma1): z1 = Phi z0 + Gamma0 u0 + Gamma1 u1."""
    ns, nu = B.shape
    aug = np.zeros((ns + 2 * nu, ns + 2 * nu))
    aug[:ns, :ns] = A * h
    aug[:ns, ns : ns + nu] = B * h
    aug[ns : ns + nu, ns + nu :] = np.eye(nu)
    E = scipy.linalg.expm(aug)
    Phi = E[:ns, :ns]
    G1 = E[:ns, ns : ns + nu]
    G2 = E[:ns, ns + nu :]
    return Phi, G1 - G2, G2


def _inputs(B: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """B as columns and u as a row per sample, a column per input."""
    B = np.asarray(B, dtype=float)
    B = B[:, None] if B.ndim == 1 else B
    u = np.asarray(u, dtype=float).reshape(len(u), -1)
    if u.shape[1] != B.shape[1]:
        raise ValueError(f"{u.shape[1]} inputs for {B.shape[1]} columns of B")
    return B, u


def respond(A: np.ndarray, B: np.ndarray, h: float, u: np.ndarray, z0: np.ndarray | None = None) -> np.ndarray:
    """The states (len(u), n) of z' = A z + B u under u sampled every h (a row per sample, a column per input:
    a 1-D u is one input) and linear between samples, exactly. z[0] is z0 (rest by default)."""
    B, u = _inputs(B, u)
    Phi, G0, G1 = foh_discretize(A, B, h)
    return _steps(Phi, G0, G1, u, z0)


def free(A: np.ndarray, h: float, steps: int, z0: np.ndarray) -> np.ndarray:
    """The states (steps + 1, n) of z' = A z from z0, h apart: z[k] = expm(A h)ᵏ z0."""
    n = A.shape[0]
    zero = np.zeros((n, 1))
    return _steps(scipy.linalg.expm(A * h), zero, zero, np.zeros((steps + 1, 1)), z0)


def _steps(Phi, G0, G1, u: np.ndarray, z0: np.ndarray | None) -> np.ndarray:
    n = Phi.shape[0]
    z0 = np.zeros(n) if z0 is None else np.asarray(z0, dtype=float)
    if native.module is None or not len(u):
        return respond_reference(Phi, G0, G1, u, z0)
    # One kernel run per input, from z0 on the first and rest on the others: the states add (the update's linear).
    Phi = np.ascontiguousarray(Phi)
    z = np.empty((len(u), n))
    z[0] = z0
    part = np.empty((len(u) - 1, n))
    for j in range(u.shape[1]):
        native.LinearSteps(Phi, np.ascontiguousarray(G0[:, j]), np.ascontiguousarray(G1[:, j])).advance(
            z0 if j == 0 else np.zeros(n), float(u[0, j]), np.ascontiguousarray(u[1:, j]), part
        )
        if j == 0:
            z[1:] = part
        else:
            z[1:] += part
    return z


def respond_reference(Phi, G0, G1, u: np.ndarray, z0: np.ndarray) -> np.ndarray:
    """`respond`'s steps in Python: z[k] = Φ z[k-1] + Γ₀ u[k-1] + Γ₁ u[k], from z[0] = z0."""
    z = np.empty((len(u), Phi.shape[0]))
    if not len(u):
        return z
    drive = u[:-1] @ G0.T + u[1:] @ G1.T
    z[0] = z0
    for k in range(1, len(u)):
        z[k] = Phi @ z[k - 1] + drive[k - 1]
    return z


class FohSteps:
    """The update from rest, resumable: each `run` carries on from the state and the input where the last left
    them. Inputs are a column each (`B`'s), a row per sample."""

    def __init__(self, A: np.ndarray, B: np.ndarray, h: float) -> None:
        self.Phi, self.G0, self.G1 = foh_discretize(A, B, h)
        self.f_prev = np.zeros(B.shape[1])  # the input at the last sample
        self._z = np.zeros(A.shape[0])
        mu, V = np.linalg.eig(self.Phi)
        self.modal = bool(np.linalg.cond(V) < MAX_EIGVEC_CONDITION)
        if self.modal:
            Vinv = np.linalg.inv(V)
            self._mu, self._V, self._Vinv = mu, V, Vinv
            self._b0, self._b1 = Vinv @ self.G0, Vinv @ self.G1
            self._eta = np.zeros(mu.size, dtype=complex)

    @property
    def state(self) -> np.ndarray:
        return (self._V @ self._eta).real if self.modal else self._z

    @state.setter
    def state(self, z: np.ndarray) -> None:
        if self.modal:
            self._eta = self._Vinv @ z
        else:
            self._z = np.array(z, dtype=float)

    def run(self, f: np.ndarray) -> np.ndarray:
        """The states (len(f), A's size) at the samples of f (len(f), inputs); f[k] is the input at the end of
        step k."""
        import scipy.signal  # deferred: scipy.signal adds ~0.9 s to startup

        f = np.asarray(f, dtype=float).reshape(len(f), -1)
        before = np.vstack([self.f_prev[None, :], f[:-1]])
        self.f_prev = f[-1].copy()
        if not self.modal:
            z = np.empty((len(f), self._z.size))
            zk = self._z
            for k in range(len(f)):
                zk = self.Phi @ zk + self.G0 @ before[k] + self.G1 @ f[k]
                z[k] = zk
            self._z = zk
            return z
        drive = before @ self._b0.T + f @ self._b1.T
        eta = np.empty(drive.shape, dtype=complex)
        for r, mu in enumerate(self._mu):
            eta[:, r], _ = scipy.signal.lfilter([1.0], [1.0, -mu], drive[:, r], zi=[mu * self._eta[r]])
        self._eta = eta[-1].copy()
        return (eta @ self._V.T).real
