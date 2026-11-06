"""Planning a test on the chain: where to put the sensors, and what the chain does at one frequency.

Sensor placement
----------------
A test measures the response at a few DOFs, the sensors, and fits the modes it wants (the targets, Φ's
columns) to them: y = Φ_s q + noise, so the modal coordinates are q̂ = (Φ_sᵀΦ_s)⁻¹Φ_sᵀ y, and their
covariance σ²(Φ_sᵀΦ_s)⁻¹. Φ_sᵀΦ_s is the Fisher information: the larger its determinant, the less the noise
spoils the fit, and if it is singular (fewer sensors than targets, or sensors that can't tell two modes apart)
the fit fails. Kammer's **effective independence** (EfI) starts from every candidate DOF and drops, one at a
time, the one that adds least to that determinant: each candidate's share is

    E_i = [Φ (ΦᵀΦ)⁻¹ Φᵀ]_ii,  Σ E_i = the number of targets,

the diagonal of the projector onto Φ's columns, and dropping DOF i scales det(ΦᵀΦ) by (1 − E_i). EfI is
greedy, not optimal, but it is quick and usually close. Weighted by the driving-point residue,
DPR_i = Σ_r φ_ir² / ω_r (EfI-DPR), it leans toward where the response is large, so the signals are well above
the noise floor. `quality` scores a placement: the Fisher determinant against all candidates', each mode's
noise against all candidates' (`noise`), and the auto-MAC of the targets at the sensors. A large off-diagonal
there is spatial aliasing: with these sensors two modes look alike.

Operational deflection shapes
-----------------------------
An ODS is how the structure deflects at one frequency under the forces it actually runs with: X(ω) = H(ω) F,
complex, every DOF with its own phase. It is not a mode shape. Near a lightly damped mode that mode dominates
and the ODS looks like it; between modes it is a mix, and it depends on where the forces act, which a mode
shape doesn't. With the undamped modes the response splits exactly into each mode's share,
X = Σ φ_r η_r, η = (Ω² − ω² + iω ΦᵀCΦ)⁻¹ Φᵀ F (`ods_parts`: the modal coordinates solved together, so
non-proportional damping is kept). In a test the forces are unknown, so the ODS is measured against a reference
sensor: T_i = X_i / X_ref, its phase from the cross-spectrum. A reference near a node of the ODS makes every
ratio large and noisy.

Working on it
-------------
`core/test_planning.py` (the *Test planning* page, `gui/test_planning.py`, `planning_page`, after Model
updating; follows the chain): sensor placement for the lowest `targets` undamped modes (`undamped_modes`,
mass-normalized): `effective_independence` (Kammer: drop the least E_i = [Φ(ΦᵀΦ)⁻¹Φᵀ]_ii, worked out again
after each drop; × DPR_i for EfI-DPR; every step kept for the Placement tab), `evenly_spaced`, or by hand
(`place`), scored by `quality` (det(Φ_sᵀΦ_s) against every mass's, each mode's noise against every mass's, the
auto-MAC). The ODS is X = H(ω)F for a unit force on one mass (`ods`); `ods_parts` splits it into φ_r η_r with
the undamped modes' coordinates solved together (ΦᵀCΦ whole), so they add up exactly with any damping.
`tests/test_test_planning.py` checks EfI against the determinant it stands for, the parts' sum, and the
trials' notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy.linalg

from .model import ChainSystem

TWO_PI = 2.0 * np.pi
SINGULAR = 1e-12  # det(Φ_sᵀΦ_s) / det(ΦᵀΦ) below this: the targets can't be told apart


class Method(Enum):
    EFI = "Effective independence"
    EFI_DPR = "EfI × driving-point residue"
    EVEN = "Evenly spaced"
    HAND = "By hand"


METHODS = tuple(Method)


def undamped_modes(system: ChainSystem) -> tuple[np.ndarray, np.ndarray]:
    """(f Hz, Φ mass-normalized, a column per mode), each column's largest entry positive."""
    M, _, K = system.matrices()
    w2, phi = scipy.linalg.eigh(K, M)
    phi *= np.sign(phi[np.argmax(np.abs(phi), axis=0), np.arange(phi.shape[1])])
    return np.sqrt(np.clip(w2, 0.0, None)) / TWO_PI, phi


def shares(phi: np.ndarray) -> np.ndarray:
    """E_i, each row's share of Φ's column space: the projector Φ(ΦᵀΦ)⁻¹Φᵀ's diagonal (they sum to its rank)."""
    G = phi.T @ phi
    return np.einsum("ij,ji->i", phi, np.linalg.solve(G, phi.T))


def driving_point_residue(phi: np.ndarray, hz: np.ndarray) -> np.ndarray:
    """DPR_i = Σ_r φ_ir² / ω_r over the targets."""
    return (phi**2) @ (1.0 / (TWO_PI * hz))


@dataclass(frozen=True)
class Step:
    candidates: np.ndarray  # DOF indices still in, from 0
    score: np.ndarray  # each one's E_i (× DPR_i for EfI-DPR)
    dropped: int  # the DOF dropped at this step


@dataclass(frozen=True)
class Placement:
    sensors: np.ndarray  # sorted DOF indices, from 0
    steps: tuple[Step, ...]  # EfI's eliminations, in order (none for evenly spaced or by hand)
    final: np.ndarray  # the score of each sensor kept, at the end


def effective_independence(phi: np.ndarray, keep: int, hz: np.ndarray | None = None) -> Placement:
    """Kammer's EfI: from every DOF, drop the one with the least E_i (× DPR_i if `hz` is given) until `keep`
    are left. Needs keep ≥ the number of targets (Φ's columns)."""
    n, m = phi.shape
    if not m <= keep <= n:
        raise ValueError(f"EfI keeps between {m} (the targets) and {n} (every DOF) sensors")
    weight = driving_point_residue(phi, hz) if hz is not None else np.ones(n)
    candidates = np.arange(n)
    steps = []
    score = shares(phi) * weight
    while candidates.size > keep:
        k = int(np.argmin(score))
        steps.append(Step(candidates.copy(), score.copy(), int(candidates[k])))
        candidates = np.delete(candidates, k)
        score = shares(phi[candidates]) * weight[candidates]
    return Placement(candidates, tuple(steps), score)


def evenly_spaced(n: int, keep: int) -> np.ndarray:
    """`keep` DOFs spread along the chain, the last at its free end."""
    return np.unique(np.round(np.linspace(n / keep, n, keep)).astype(int) - 1)


def place(phi: np.ndarray, hz: np.ndarray, method: Method, keep: int, hand: tuple[int, ...] = ()) -> Placement:
    n = phi.shape[0]
    if method is Method.EFI:
        return effective_independence(phi, keep)
    if method is Method.EFI_DPR:
        return effective_independence(phi, keep, hz)
    sensors = evenly_spaced(n, keep) if method is Method.EVEN else np.array(sorted({i for i in hand if 0 <= i < n}))
    sensors = sensors.astype(int)
    final = shares(phi[sensors]) if sensors.size and _rank(phi[sensors]) == phi.shape[1] else np.zeros(sensors.size)
    return Placement(sensors, (), final)


def _rank(A: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(A)) if A.size else 0


def mac(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.vdot(a, a).real * np.vdot(b, b).real)
    return float(abs(np.vdot(a, b)) ** 2 / den) if den > 0 else 0.0


def auto_mac(shapes: np.ndarray) -> np.ndarray:
    """MAC between every pair of columns."""
    m = shapes.shape[1]
    return np.array([[mac(shapes[:, r], shapes[:, s]) for s in range(m)] for r in range(m)])


@dataclass(frozen=True)
class Quality:
    information: float  # det(Φ_sᵀΦ_s) / det(ΦᵀΦ) over every DOF: 1 with every sensor, 0 when singular
    noise: np.ndarray  # each target's noise against every DOF's: √[(Φ_sᵀΦ_s)⁻¹]_rr / √[(ΦᵀΦ)⁻¹]_rr (inf if singular)
    auto_mac: np.ndarray  # the targets at the sensors
    worst: float  # its largest off-diagonal (0 for one target)

    @property
    def singular(self) -> bool:
        return self.information < SINGULAR


def quality(phi: np.ndarray, sensors: np.ndarray) -> Quality:
    m = phi.shape[1]
    full = phi.T @ phi
    s = phi[np.asarray(sensors, dtype=int)]
    G = s.T @ s
    info = float(np.linalg.det(G) / np.linalg.det(full)) if s.size else 0.0
    if info < SINGULAR:
        noise = np.full(m, np.inf)
    else:
        noise = np.sqrt(np.diag(np.linalg.inv(G)) / np.diag(np.linalg.inv(full)))
    am = auto_mac(s) if s.size else np.eye(m)
    off = am - np.diag(np.diag(am))
    return Quality(max(info, 0.0), noise, am, float(off.max(initial=0.0)))


# -------------------------------------------------------------------- operational deflection shapes
def ods(system: ChainSystem, force: int, hz: float) -> np.ndarray:
    """X = H(ω) F for a unit force on mass `force` (index from 0): every mass's complex amplitude, m/N."""
    M, C, K = system.matrices()
    w = TWO_PI * hz
    f = np.zeros(system.n)
    f[force] = 1.0
    return np.linalg.solve(K - w * w * M + 1j * w * C, f.astype(complex))


def ods_parts(system: ChainSystem, force: int, hz: float) -> np.ndarray:
    """(n, modes): each undamped mode's share φ_r η_r of the ODS, the modal coordinates solved together (they
    add up to `ods` exactly, with any damping)."""
    _, C, K = system.matrices()
    _, phi = undamped_modes(system)
    w = TWO_PI * hz
    w2 = np.diag(phi.T @ K @ phi)
    eta = np.linalg.solve(np.diag(w2 - w * w) + 1j * w * (phi.T @ C @ phi), phi[force].astype(complex))
    return phi * eta


def ods_spectrum(system: ChainSystem, force: int, freqs: np.ndarray) -> np.ndarray:
    """(len(freqs), n): every mass's response to a unit force on `force`, over frequency."""
    M, C, K = system.matrices()
    w = TWO_PI * np.asarray(freqs, dtype=float)
    Z = K[None] - (w**2)[:, None, None] * M[None] + 1j * w[:, None, None] * C[None]
    f = np.zeros((w.size, system.n, 1), dtype=complex)
    f[:, force] = 1.0
    return np.linalg.solve(Z, f)[:, :, 0]


def at_phase(x: np.ndarray) -> np.ndarray:
    """The complex shape turned to the instant its real part is largest: Re(x e^{−iθ}) with θ the angle that
    maximizes ‖Re‖ (half the angle of Σ x_i², a real shape's own phase)."""
    theta = 0.5 * np.angle(np.sum(x * x))
    return (x * np.exp(-1j * theta)).real


def spectrum_grid(hz: np.ndarray, points: int = 800) -> np.ndarray:
    """Hz for the ODS spectrum: log-spaced over the modes, denser at each, none on a natural frequency."""
    fn = hz[hz > 0]
    base = np.geomspace(0.3 * fn.min(), 1.6 * fn.max(), points)
    peaks = [f * (1 + np.linspace(-0.05, 0.05, 60)) for f in fn]
    return np.unique(np.concatenate([base, *peaks]))
