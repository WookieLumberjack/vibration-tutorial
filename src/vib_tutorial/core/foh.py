"""The exact first-order-hold (FOH) update of a linear state space, z' = A z + B f, for long records.

With f linear between samples h apart, the update is exact (`simulator.foh_discretize`):
z[k] = Φ z[k-1] + Γ₀ f[k-1] + Γ₁ f[k]. Diagonalized, Φ = V diag(μ) V⁻¹, it is one first-order recursive filter
per eigenvalue, η_r[k] = μ_r η_r[k-1] + (V⁻¹Γ₀ f[k-1] + V⁻¹Γ₁ f[k])_r, which `scipy.signal.lfilter` runs in C:
the virtual modal test's chain, the OMA and random-vibration records and the shock page's chain all step this
way. A defective Φ (a free chain's rigid-body λ = 0, at a small h) has no diagonal form to trust: then `FohSteps`
steps z itself.
"""

from __future__ import annotations

import numpy as np

from .frf_matrix import MAX_EIGVEC_CONDITION
from .simulator import foh_discretize


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
