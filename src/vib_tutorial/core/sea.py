"""Statistical energy analysis: two coupled oscillators, the L-junction's bending waves, and a network of two
plates and a room.

The SEA tabs of the *Vibro-acoustics & SEA* page. SEA's basis is the two-oscillator result: two oscillators
(blocked frequencies ω₁, ω₂: each with the other held still), coupled by a spring k_c and driven by independent
white-noise forces, exchange power in exact proportion to the difference of their energies E = m⟨v²⟩,

P₁₂ = β (E₁ − E₂), β = (k_c²/m₁m₂)(Δ₁ + Δ₂) / [(ω₁² − ω₂²)² + (Δ₁ + Δ₂)(Δ₁ω₂² + Δ₂ω₁²)], Δ = c/m,

whatever the forces' strengths (`oscillator_pair` works the energies and the flow out exactly, from the state's
covariance by a Lyapunov equation; `coupling_beta` is the formula). Summed over the modes of two subsystems
resonant in a band, it becomes SEA's power balance: ω η_ij E_i − ω η_ji E_j flows from i to j, with the
*coupling loss factors* obeying reciprocity, n_i η_ij = n_j η_ji (n the modal densities).

The coupling between two plates at a right angle (`junction_transmission`) is a bending-wave problem: the corner
line neither moves nor bends (each plate's in-plane stiffness holds the other's bending), so a wave arriving at θ
is reflected and transmitted with a near field on each side, and τ(θ) + |r|² = 1. Its diffuse-field average τ̄
makes the coupling loss factor η₁₂ = c_g₁ L τ̄₁₂ / π ω S₁. A plate couples to a room by its radiation,
η_pr = ρcσ/ωm″ (`radiation.radiation_loss_factor`), and the room back by reciprocity; the room's own loss is its
reverberation time's, 2.2/f T₆₀. `network` solves the three subsystems' balance in each third-octave band.

Working on it
-------------
Keep `oscillator_pair` exact (no weak-coupling approximation): it is what the tab checks `coupling_beta` against.
`tests/test_sea.py` checks the flow's proportionality, the junction's energy balance and reciprocity, and the
network's power balance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from .beams_plates import TWO_PI, Plate
from .radiation import AIR, Fluid, radiation_loss_factor


# --------------------------------------------------------------------------------------------- two oscillators
@dataclass(frozen=True)
class Oscillators:
    """Two oscillators coupled by a spring. Frequencies are the blocked ones (Hz): each with the other held."""

    f1: float = 100.0
    f2: float = 110.0
    zeta1: float = 0.02
    zeta2: float = 0.01
    coupling: float = 0.05  # k_c over oscillator 1's blocked stiffness m₁ω₁²
    m1: float = 1.0
    m2: float = 1.0

    @property
    def omegas(self) -> tuple[float, float]:
        return TWO_PI * self.f1, TWO_PI * self.f2

    @property
    def spring(self) -> float:
        return self.coupling * self.m1 * self.omegas[0] ** 2

    @property
    def deltas(self) -> tuple[float, float]:
        """Δ = c/m = 2ζω, each oscillator's damping rate."""
        w1, w2 = self.omegas
        return 2 * self.zeta1 * w1, 2 * self.zeta2 * w2


@dataclass(frozen=True)
class PairResponse:
    """The pair's steady random response: energies m⟨v²⟩ (J), powers (W)."""

    E1: float
    E2: float
    flow: float  # P₁₂, from 1 to 2 through the spring
    lost1: float  # each oscillator's damper: c⟨v²⟩
    lost2: float


def oscillator_pair(osc: Oscillators, p1: float, p2: float) -> PairResponse:
    """The exact response to independent white-noise forces putting in p1 and p2 (W): the state's covariance
    solves A Σ + Σ Aᵀ + Q = 0, Q's velocity entries 2p/m (a white force puts in its intensity over 2m)."""
    w1, w2 = osc.omegas
    kc = osc.spring
    m1, m2 = osc.m1, osc.m2
    d1, d2 = osc.deltas
    s = w1  # the state is (ω₁x₁, ω₁x₂, v₁, v₂): balanced, so the solve keeps its precision at light damping
    A = np.array(
        [
            [0.0, 0.0, s, 0.0],
            [0.0, 0.0, 0.0, s],
            [-(w1**2) / s, kc / m1 / s, -d1, 0.0],
            [kc / m2 / s, -(w2**2) / s, 0.0, -d2],
        ]
    )
    Q = np.diag([0.0, 0.0, 2 * p1 / m1, 2 * p2 / m2])
    S = solve_continuous_lyapunov(A, -Q)
    v1, v2 = S[2, 2], S[3, 3]
    return PairResponse(m1 * v1, m2 * v2, kc * S[0, 3] / s, m1 * d1 * v1, m2 * d2 * v2)


def coupling_beta(osc: Oscillators) -> float:
    """β in P₁₂ = β(E₁ − E₂): largest when the oscillators are tuned, over a width set by their damping."""
    w1, w2 = osc.omegas
    d1, d2 = osc.deltas
    g = osc.spring**2 / (osc.m1 * osc.m2)
    return g * (d1 + d2) / ((w1**2 - w2**2) ** 2 + (d1 + d2) * (d1 * w2**2 + d2 * w1**2))


# --------------------------------------------------------------------------------------------- the junction
def junction_waves(plate1: Plate, plate2: Plate, f: float, theta: float) -> tuple[float, float]:
    """(τ, |r|²) for a bending wave on plate 1 meeting plate 2 at a right-angle corner, at θ from the normal.

    Along the corner both plates are held from moving (each by the other's in-plane stiffness), turn together,
    and their bending moments balance. Each plate carries its propagating wave and a near field decaying from
    the corner; with the trace k_y = k₁ sin θ shared, plate 2's wave is e^{i k_x2 x}, k_x2 = √(k₂² − k_y²) (none
    if k_y > k₂: then τ = 0)."""
    w = TWO_PI * f
    k1 = math.sqrt(w / plate1.scale)
    k2 = math.sqrt(w / plate2.scale)
    ky = k1 * math.sin(theta)
    kx1 = k1 * math.cos(theta)
    kx2 = complex(np.sqrt(complex(k2 * k2 - ky * ky)))
    if kx2.imag < 0:
        kx2 = -kx2
    mu1 = math.sqrt(k1 * k1 + ky * ky)
    mu2 = math.sqrt(k2 * k2 + ky * ky)
    D1, D2 = plate1.rigidity, plate2.rigidity
    # unknowns: plate 1's reflected wave r and near field n₁ (e^{μ₁x}, x ≤ 0); plate 2's t and n₂ (e^{−μ₂x})
    M = np.array(
        [
            [1, 1, 0, 0],  # plate 1 doesn't move at the corner
            [0, 0, 1, 1],  # nor plate 2
            [-1j * kx1, mu1, -1j * kx2, mu2],  # they turn together
            [-D1 * kx1**2, D1 * mu1**2, D2 * kx2**2, -D2 * mu2**2],  # their moments balance
        ],
        dtype=complex,
    )
    rhs = np.array([-1, 0, -1j * kx1, D1 * kx1**2], dtype=complex)  # the incident wave's part
    r, _, t, _ = np.linalg.solve(M, rhs)
    tau = D2 * k2 * k2 * kx2.real * abs(t) ** 2 / (D1 * k1 * k1 * kx1) if kx2.real > 0 else 0.0
    return float(tau), float(abs(r) ** 2)


def junction_transmission(plate1: Plate, plate2: Plate, f: float, points: int = 96) -> float:
    """τ̄₁₂ = ∫ τ(θ) cos θ dθ over 0 to π/2: the diffuse field's share across the corner (its intensity at θ goes
    as cos θ). Gauss–Legendre in θ, split at plate 2's cut-off angle sin θ = k₂/k₁ where there is one."""
    k1, k2 = math.sqrt(1 / plate1.scale), math.sqrt(1 / plate2.scale)  # × √ω, which cancels
    top = math.pi / 2 if k2 >= k1 else math.asin(k2 / k1)
    x, wts = np.polynomial.legendre.leggauss(points)
    th = 0.5 * top * (x + 1)
    tau = np.array([junction_waves(plate1, plate2, f, t)[0] for t in th])
    return float(0.5 * top * np.sum(wts * tau * np.cos(th)))


def group_speed(plate: Plate, f: float | np.ndarray) -> np.ndarray:
    """A bending wave's energy travels at twice its phase speed: c_g = 2√(ω√(D/ρh))."""
    return 2.0 * np.sqrt(TWO_PI * np.asarray(f, dtype=float) * plate.scale)


def plate_modal_density(plate: Plate) -> float:
    """Modes per Hz, S/2 √(ρh/D) (Beams & plates' asymptote): the same at every frequency."""
    return plate.area / (2 * plate.scale)


def room_modal_density(volume: float, f: float | np.ndarray, fluid: Fluid = AIR) -> np.ndarray:
    """A room's modes per Hz, 4πf²V/c³ (the leading term of Weyl's count)."""
    return 4 * math.pi * np.asarray(f, dtype=float) ** 2 * volume / fluid.c**3


def plate_coupling(plate1: Plate, plate2: Plate, length: float, f: float) -> float:
    """η₁₂ = c_g₁ L τ̄₁₂ / π ω S₁: plate 1's energy crossing a corner of length L in a cycle, over 2π."""
    tau = junction_transmission(plate1, plate2, f)
    return float(group_speed(plate1, f)) * length * tau / (math.pi * TWO_PI * f * plate1.area)


# --------------------------------------------------------------------------------------------- the network
@dataclass(frozen=True)
class Network:
    """Plate 1 (driven) meets plate 2 at a corner of length L; both radiate into a room (one side each)."""

    plate1: Plate
    plate2: Plate
    loss1: float  # each plate's own loss factor
    loss2: float
    length: float  # the corner, m
    volume: float = 50.0  # the room, m³
    reverberation: float = 0.5  # its T₆₀, s
    force: float = 1.0  # N, the point force's amplitude in each band, on plate 1
    fluid: Fluid = AIR


@dataclass(frozen=True)
class NetworkResponse:
    """Each band's loss factors, energies (J) and powers (W); subsystems 0, 1, 2 are plate 1, plate 2, the room."""

    centres: np.ndarray  # Hz
    eta: np.ndarray  # (band, i, j): η_ij for i ≠ j, the damping loss factor on the diagonal
    density: np.ndarray  # (band, i): modes per Hz
    energy: np.ndarray  # (band, i)
    input_power: np.ndarray  # (band,): plate 1's, the infinite plate's F²/16√(Dρh)
    flow: np.ndarray  # (band, i, j): net power from i to j, ω(η_ij E_i − η_ji E_j)
    masses: np.ndarray  # the plates' masses, kg
    pressure: np.ndarray  # the room's ⟨p²⟩ = Eρc²/V, Pa²

    @property
    def velocity(self) -> np.ndarray:
        """⟨v²⟩ of the two plates, E/M."""
        return self.energy[:, :2] / self.masses


def loss_factors(net: Network, f: float) -> tuple[np.ndarray, np.ndarray]:
    """(η, n) at f: the 3 × 3 matrix of loss factors (damping on the diagonal) and the modal densities, the
    room-to-plate ones from reciprocity n_i η_ij = n_j η_ji."""
    p1, p2 = net.plate1, net.plate2
    n = np.array([plate_modal_density(p1), plate_modal_density(p2), float(room_modal_density(net.volume, f))])
    eta = np.zeros((3, 3))
    eta[0, 0], eta[1, 1] = net.loss1, net.loss2
    eta[2, 2] = 2.2 / (f * net.reverberation)  # 60 dB in T₆₀: e^{−ωηT} = 10⁻⁶
    eta[0, 1] = plate_coupling(p1, p2, net.length, f)
    eta[1, 0] = plate_coupling(p2, p1, net.length, f)
    eta[0, 2] = radiation_loss_factor(p1, f, net.fluid)
    eta[1, 2] = radiation_loss_factor(p2, f, net.fluid)
    eta[2, 0] = eta[0, 2] * n[0] / n[2]
    eta[2, 1] = eta[1, 2] * n[1] / n[2]
    return eta, n


def solve_band(eta: np.ndarray, f: float, power: np.ndarray) -> np.ndarray:
    """The energies balancing each subsystem's power: ω(η_i + Σ_j η_ij) E_i − ω Σ_j η_ji E_j = P_i."""
    L = -eta.T.copy()
    np.fill_diagonal(L, eta.sum(axis=1))  # η_ii + Σ_{j≠i} η_ij
    return np.linalg.solve(TWO_PI * f * L, power)


def network(net: Network, centres: np.ndarray, silent: int | None = None) -> NetworkResponse:
    """Every band's energies with plate 1 driven by the point force (the band's input power the infinite
    plate's: Driven panel's band average). `silent` (0 or 1) cuts that plate off from the room, both ways: what
    the room hears by the other plate alone."""
    p1 = net.plate1
    p_in = net.force**2 / (16.0 * math.sqrt(p1.rigidity * p1.surface_mass))
    nb = centres.size
    eta = np.empty((nb, 3, 3))
    dens = np.empty((nb, 3))
    energy = np.empty((nb, 3))
    flow = np.empty((nb, 3, 3))
    for i, f in enumerate(centres):
        eta[i], dens[i] = loss_factors(net, float(f))
        if silent is not None:
            eta[i, silent, 2] = eta[i, 2, silent] = 0.0
        energy[i] = solve_band(eta[i], float(f), np.array([p_in, 0.0, 0.0]))
        off = eta[i] - np.diag(np.diag(eta[i]))
        gross = off * energy[i][:, None]
        flow[i] = TWO_PI * f * (gross - gross.T)
    masses = np.array([p.surface_mass * p.area for p in (p1, net.plate2)])
    pressure = energy[:, 2] * net.fluid.rho * net.fluid.c**2 / net.volume
    return NetworkResponse(centres, eta, dens, energy, np.full(nb, p_in), flow, masses, pressure)
