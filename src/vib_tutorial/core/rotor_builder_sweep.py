"""The Rotor builder's sweeps over the speed: the Campbell diagram and its critical speeds, the stability map, and
the steady response to the discs' unbalance.

The **Campbell diagram** (`sweep`) works the whirl modes out at `N_SPEEDS` speeds from 0 to the top one, each
branch the k-th frequency in order. Where a branch's frequency meets the shaft's own, f = Ω/2π (the 1X line),
an unbalance, which turns once a revolution, drives that mode at its own frequency: a **critical speed**, found
by bisection between the two speeds either side. A forward branch's crossing is the one unbalance drives on
round bearings; a backward one's only shows on anisotropic ones. Each sweep works in a Ritz basis of the
rotor's lowest undamped modes (`Assembly.basis`, remade at each speed where an oil film changes the stiffness),
so it costs about a millisecond a speed whatever the mesh.

The **stability map** is the same sweep's damping: each branch's log decrement δ = 2πζ/√(1 - ζ²) against the
speed, and the speed where the least damped mode of all first grows (`Sweep.onset`), by bisection. API 684 asks
for δ ≥ 0.1 at the running speed.

The **unbalance response** (`unbalance_response`) is the steady state, exactly, in all the DOFs: each disc's
unbalance U = m e at its heavy spot's angle ψ pushes with U Ω² turning with the shaft, f_x + i f_y =
U e^{iψ} Ω² e^{iΩt}, so its complex amplitudes are F_x = U e^{iψ} Ω² and F_y = -i F_x, and

    (K(Ω) - Ω² M + iΩ (C(Ω) + Ω G) + D(iΩ)) Q = F,

D the magnets' controllers' k_i G(iΩ) on their axes. Each node's orbit is an ellipse; its major semi-axis
(`orbit`) is |X + iY|/2 + |X - iY|/2, its forward circle's radius plus its backward one's.

Working on it
-------------
The branches are columns in frequency order, as on the rotor page's Campbell diagram, so two modes that cross
swap columns; a critical is a branch meeting 1X whichever mode it is then. Keep the sweeps in the reduced basis
(a test checks they match the full model's modes) and the steady state in the full one. `tests/test_rotor_builder.py`
checks the criticals against the modes at those speeds, the onset against the full model's, the response
against a closed-form Jeffcott rotor, and the trials' numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .rotor_builder import RPM, TWO_PI, Assembly, Basis, Rotor, WhirlModes, modes

N_SPEEDS = 121
N_BRANCHES = 8  # the Campbell diagram's and the stability map's
BISECT = 40  # halvings for a critical speed or an onset
CRITICALLY_DAMPED = 0.2  # ζ: API 684 doesn't count a crossing whose amplification factor 1/2ζ is under 2.5


@dataclass(frozen=True)
class Critical:
    rpm: float
    whirl: float  # the branch's whirl ratio there: > 0 forward
    zeta: float
    branch: int

    @property
    def hz(self) -> float:
        return self.rpm / 60.0

    @property
    def forward(self) -> bool:
        return self.whirl > 0

    @property
    def amplification(self) -> float:
        """The amplification factor, 1/2ζ: how much the unbalance response peaks there (∞ undamped)."""
        return 1 / (2 * self.zeta) if self.zeta > 0 else math.inf

    @property
    def damped(self) -> bool:
        """Critically damped (API 684): an amplification factor under 2.5, no peak to speak of."""
        return self.zeta >= CRITICALLY_DAMPED


@dataclass(frozen=True, eq=False)
class Sweep:
    """The whirl modes from 0 to the top speed: the Campbell diagram and the stability map."""

    rpm: np.ndarray  # (n_speeds,)
    hz: np.ndarray  # (n_speeds, branches), NaN where there are fewer modes
    zeta: np.ndarray
    whirl: np.ndarray
    least: np.ndarray  # (n_speeds,): the least damping ratio of every mode found at each speed
    criticals: tuple[Critical, ...]
    onset: float | None  # rpm where the least damped mode first grows (None: stable throughout)

    @property
    def log_dec(self) -> np.ndarray:
        z = np.clip(self.zeta, -0.999999, 0.999999)
        return TWO_PI * z / np.sqrt(1 - z * z)


class _Speeds:
    """The modes at any speed in a Ritz basis: one for every speed, or one at each speed with oil films."""

    def __init__(self, assembly: Assembly, top: float) -> None:
        self.assembly = assembly
        self.fixed: Basis | None = None if assembly.films else assembly.basis(0.5 * top)

    def __call__(self, omega: float) -> WhirlModes:
        a = self.assembly
        return modes(a, omega, a.basis(omega) if a.films else self.fixed)


def sweep(assembly: Assembly, top_rpm: float, n_speeds: int = N_SPEEDS, branches: int = N_BRANCHES) -> Sweep:
    """The modes at n_speeds speeds from 0 to top_rpm, the criticals and the onset of instability."""
    rpm = np.linspace(0.0, top_rpm, n_speeds)
    at = _Speeds(assembly, top_rpm * RPM)
    hz = np.full((n_speeds, branches), np.nan)
    zeta, whirl = hz.copy(), np.zeros((n_speeds, branches))
    least = np.full(n_speeds, np.inf)
    for i, r in enumerate(rpm):
        m = at(r * RPM)
        k = min(branches, m.hz.size)
        hz[i, :k], zeta[i, :k], whirl[i, :k] = m.hz[:k], m.zeta[:k], m.whirl[:k]
        if m.zeta.size:
            least[i] = m.zeta.min()
    criticals = []
    for j in range(branches):
        gap = hz[:, j] - rpm / 60.0
        for i in range(n_speeds - 1):
            if np.isfinite(gap[i]) and np.isfinite(gap[i + 1]) and gap[i] > 0.0 >= gap[i + 1]:
                c = _critical(at, j, float(rpm[i]), float(rpm[i + 1]))
                if c is not None:
                    criticals.append(c)
    criticals.sort(key=lambda c: c.rpm)
    onset = None
    unstable = np.flatnonzero(least < 0)
    if unstable.size:
        i = int(unstable[0])
        onset = 0.0 if i == 0 else _bisect(lambda r: _least(at(r * RPM)) < 0, float(rpm[i - 1]), float(rpm[i]))
    return Sweep(rpm, hz, zeta, whirl, least, tuple(criticals), onset)


def _least(m: WhirlModes) -> float:
    return float(m.zeta.min()) if m.zeta.size else math.inf


def _bisect(past, lo: float, hi: float) -> float:
    """The speed (rpm) between lo (not past) and hi (past) where `past` turns true."""
    for _ in range(BISECT):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if past(mid) else (mid, hi)
    return 0.5 * (lo + hi)


def _critical(at: _Speeds, branch: int, lo: float, hi: float) -> Critical | None:
    """Where branch `branch` meets 1X between lo and hi (rpm)."""

    def past(r: float) -> bool:
        f = at(r * RPM).hz
        return branch >= f.size or f[branch] <= r / 60.0

    r = _bisect(past, lo, hi)
    m = at(r * RPM)
    if branch >= m.hz.size:
        return None
    return Critical(r, float(m.whirl[branch]), float(m.zeta[branch]), branch)


# ------------------------------------------------------------------ the unbalance response
def unbalance_load(rotor: Rotor) -> np.ndarray:
    """P (n_dof, complex): the discs' unbalance per Ω², F = Ω² P."""
    P = np.zeros(rotor.n_dof, dtype=complex)
    for d in rotor.discs:
        j = 4 * rotor.node_at(d.z)
        u = d.unbalance * np.exp(1j * d.unbalance_angle)
        P[j] += u
        P[j + 2] += -1j * u
    return P


def unbalance_response(assembly: Assembly, rpm: np.ndarray) -> np.ndarray:
    """Q (len(rpm), n_dof): the steady state at each speed, q = Re(Q e^{iΩt})."""
    a = assembly
    P = unbalance_load(a.rotor)
    out = np.zeros((np.size(rpm), a.n), dtype=complex)
    for i, r in enumerate(np.atleast_1d(rpm)):
        w = float(r) * RPM
        if w <= 0 or not P.any():
            continue
        Z = a.stiffness(w) - w * w * a.M + 1j * w * (a.damping(w) + w * a.G)
        Z = Z.astype(complex)
        for d, m in a.axes:
            Z[d, d] += m.current_gain * m.gain(1j * w)
        out[i] = np.linalg.solve(Z, w * w * P)
    return out


def orbit(Q: np.ndarray, node: int | np.ndarray) -> np.ndarray:
    """The major semi-axis of node `node`'s orbit (or each node's, given several), for each row of Q."""
    X, Y = Q[..., 4 * node], Q[..., 4 * node + 2]
    return 0.5 * (np.abs(X + 1j * Y) + np.abs(X - 1j * Y))


def lag(Q: np.ndarray, node: int, rotor: Rotor) -> np.ndarray:
    """How far (°, 0 to 360) node `node`'s x lags behind the first unbalanced disc's heavy spot."""
    heavy = next((d.unbalance_angle for d in rotor.discs if d.unbalance), 0.0)
    return np.degrees(heavy - np.angle(Q[..., 4 * node])) % 360.0
