"""The rotor's whirl: its modes at a speed (Floquet modes for a shaft whose stiffness turns, in axes turning
with it where they're exact, Hill's method otherwise), the Campbell diagram and its criticals, and the steady
states: the synchronous response, harmonic balance, gravity's, every harmonic together. The model is
core.rotor's; the time response is core.rotor_simulator's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .rotor import (
    N_DOF,
    RotorSystem,
)

HARMONICS = 3  # an asymmetric shaft on anisotropic supports: harmonics kept each side of the 1X (or 0X) one
HILL_ORDER = 3  # and harmonics of 2Ω (Ω with a breathing crack) kept each side in Hill's method for its modes
# Nonlinear oil films (RotorSystem.nonlinear_films) in the time step, see RotorSimulator._film_steps:
OVERDAMPED = 0.05  # a root with |Im λ| below this times |Re λ| (ζ > 0.9987) is taken as not oscillating


# ------------------------------------------------------------------ whirl analysis


def whirl_ratio(system: RotorSystem, Q: np.ndarray) -> np.ndarray:
    """Whirl direction of complex amplitudes Q (..., n_dof), x = Re(X e^{iωt}) for ω > 0.

    +1 is a circular forward whirl (with the spin, x toward y), -1 a circular
    backward whirl, 0 a straight line. The motion is weighted by one plane's mass
    matrix, i.e. by kinetic energy: the stations (journals, disc, disc tilt) and,
    with a shaft of some mass, the shaft between them.
    """
    mp = system.plane_mass()
    ix, iy = system.plane_dofs
    X, Y = Q[..., ix], Q[..., iy]
    num = 2.0 * np.imag(np.einsum("...i,ij,...j->...", np.conj(Y), mp, X))
    den = np.real(np.einsum("...i,ij,...j->...", np.conj(X), mp, X) + np.einsum("...i,ij,...j->...", np.conj(Y), mp, Y))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def _whirl_forms(system: RotorSystem) -> tuple[np.ndarray, np.ndarray]:
    """Hermitian (H, D) with v^H H v and v^H D v the numerator and denominator of whirl_ratio."""
    mp = system.plane_mass().astype(complex)
    return system._blocks(0 * mp, 0 * mp, 1j * mp, -1j * mp), system._blocks(mp, mp)


def _circularize(system: RotorSystem, lam: np.ndarray, vec: np.ndarray, tol: float = 1e-7) -> None:
    """Within each group of repeated eigenvalues, pick the purely forward and backward shapes.

    An axisymmetric rotor not spinning (or with no tilt coupling) has each whirl
    frequency twice, and the solver returns any mix of the pair. The combinations
    that extremize the whirl ratio are its circular forward and backward whirls.
    """
    H, D = None, None
    i = 0
    while i < lam.size:
        j = i + 1
        while j < lam.size and abs(lam[j] - lam[i]) <= tol * abs(lam[i]):
            j += 1
        if j - i > 1:
            if H is None:
                H, D = _whirl_forms(system)
            V = vec[:, i:j]
            _, c = scipy.linalg.eigh(V.conj().T @ H @ V, V.conj().T @ D @ V)
            vec[:, i:j] = V @ c
        i = j


@dataclass
class WhirlModes:
    """The rotor's damped modes at one speed, in order of frequency (one per conjugate pair)."""

    omega: float  # spin speed (rad/s)
    freq_hz: np.ndarray  # damped natural frequencies
    zeta: np.ndarray  # damping ratios, -Re λ / |λ|
    whirl: np.ndarray  # whirl ratios (see whirl_ratio): > 0 forward, < 0 backward
    shapes: np.ndarray  # (n, n_dof) complex displacement shapes, largest station entry 1
    drift: float = -math.inf  # 1/s, the fastest growth of a root that does not oscillate (> 0: it diverges)

    @property
    def forward(self) -> np.ndarray:
        return self.whirl >= 0.0


def whirl_modes(system: RotorSystem, omega: float) -> WhirlModes:
    """Eigen-solution of A(Ω): the oscillatory modes (Im λ > 0), lowest first.

    A root that does not oscillate is left out, but the fastest growth among them
    is kept (`drift`): a magnetic bearing too weak for its magnets lets its journal
    fall away without oscillating.

    An asymmetric or cracked shaft's stiffness turns with it, so A depends on
    time and its modes are Floquet's: e^{λt} times a motion periodic in φ. Each is
    listed by the harmonic of it that carries the most kinetic energy (see
    _rotating_modes and _hill_modes): that is its frequency here, and its shape
    and whirl are that harmonic's.
    """
    drift = -math.inf
    if system.periodic:
        if system.rotating_frame:
            lam, vec = _rotating_modes(system, omega)
        else:
            lam, vec, drift = _hill_modes(system, omega)
        order = np.argsort(lam.imag)
        lam, vec = lam[order], vec[:, order]
    else:
        lam, vec = np.linalg.eig(system.state_matrix(omega))
        keep, drift = _oscillation(lam)
        lam, vec = lam[keep], vec[: system.n_dof, keep]
        order = np.argsort(lam.imag)
        lam, vec = lam[order], vec[:, order]
        _circularize(system, lam, vec)
    shapes = vec.T.copy()
    for s in shapes:
        # By a station's largest entry (a shaft mode may barely move them: then by its largest).
        big = s[:N_DOF] if np.abs(s[:N_DOF]).max() > 1e-6 * np.abs(s).max() else s
        s /= big[np.argmax(np.abs(big))]
    return WhirlModes(
        omega=omega,
        freq_hz=lam.imag / (2.0 * math.pi),
        zeta=-lam.real / np.abs(lam),
        whirl=whirl_ratio(system, shapes),
        shapes=shapes,
        drift=drift,
    )


def _oscillation(lam: np.ndarray) -> tuple[np.ndarray, float]:
    """(Im λ > 0 and oscillating, the fastest growth of the roots that do not oscillate (-inf if none)).

    A magnet's controller has repeated slow real roots (its integrals), which the
    solver and the gyroscopic coupling split by a hair: a root this overdamped
    (|Im λ| < OVERDAMPED |Re λ|, ζ > 0.9987) does not oscillate.
    """
    limit = np.maximum(_oscillating(lam), OVERDAMPED * np.abs(lam.real))
    real = np.abs(lam.imag) <= limit
    return lam.imag > limit, float(lam.real[real].max()) if real.any() else -math.inf


def _oscillating(lam: np.ndarray) -> float:
    """The least Im λ of a mode that oscillates (smaller is round-off on a real eigenvalue)."""
    return 1e-9 * max(1.0, float(np.abs(lam).max())) if lam.size else 0.0


def rotating_state_matrix(system: RotorSystem, omega: float) -> np.ndarray:
    """A_r(Ω) (2n x 2n) in coordinates turning with the shaft, q = R(φ) r: constant on isotropic supports.

    R(φ) turns each plane pair by φ, R' = Ω R J. Every matrix but the asymmetric
    shaft's commutes with R when the supports are isotropic, so R^T(...)R leaves
    them, the shaft's deviatoric part is held at φ = 0 (so is an open crack's), and
        M r'' + (C + ΩG + 2ΩMJ) r' + (K + ΩH + ΔK(0) - Ω²M + ΩCJ + Ω²GJ) r = R^T f.
    """
    M, _, G, _ = system._matrices
    C = system.damping(omega)
    n = system.n_dof
    J = _turn(system)
    Kr = system.stiffness_at(omega, 0.0) - omega**2 * M + omega * C @ J + omega**2 * G @ J
    Cr = C + omega * G + 2.0 * omega * M @ J
    Minv = system.mass_inverse
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -Minv @ Kr
    A[n:, n:] = -Minv @ Cr
    return A


def _turn(system: RotorSystem) -> np.ndarray:
    """J (n x n): turns every plane pair x toward y, J [x; y] = [-y; x]."""
    eye = np.eye(system.n_dof // 2)
    return system._blocks(0 * eye, 0 * eye, -eye, eye)


def _rotating_modes(system: RotorSystem, omega: float) -> tuple[np.ndarray, np.ndarray]:
    """An asymmetric shaft's modes on isotropic supports, exactly: (λ (k,), shapes (n_dof, k)), Im λ > 0.

    Turning with the shaft the rotor is time-invariant (rotating_state_matrix).
    A mode r = V e^{μt} there, V = [X; Y] by plane, is seen from the fixed frame
    as a forward circle (X + iY) e^{(μ + iΩ)t} plus a backward one (X - iY)
    e^{(μ - iΩ)t}. The one carrying more kinetic energy names the mode: λ = μ ± iΩ,
    its shape that circle (conjugated if need be, so that Im λ > 0). Its growth
    rate Re λ = Re μ is exact. A mode that stands still turning with the shaft
    (μ real, as between the two critical speeds) is a forward whirl locked to 1X.
    """
    mu, vec = np.linalg.eig(rotating_state_matrix(system, omega))
    # A conjugate pair there is one real motion, and maps to one λ: keep its Im μ ≥ 0 member.
    half = mu.imag >= -_oscillating(mu)
    mu, vec = mu[half], vec[:, half]
    ix, iy = system.plane_dofs
    V = vec[: system.n_dof]
    a, b = V[ix] + 1j * V[iy], V[ix] - 1j * V[iy]
    mp = system.plane_mass()
    fwd = np.real(np.einsum("ik,ij,jk->k", a.conj(), mp, a))
    bwd = np.real(np.einsum("ik,ij,jk->k", b.conj(), mp, b))
    forward = fwd >= bwd
    lam = mu + np.where(forward, 1j, -1j) * omega
    shapes = np.where(forward, system.merge(a.T, -1j * a.T).T, system.merge(b.T, 1j * b.T).T)
    flip = lam.imag < 0
    lam[flip], shapes[:, flip] = lam[flip].conj(), shapes[:, flip].conj()
    keep = lam.imag > _oscillating(lam)
    return lam[keep], shapes[:, keep]


def _hill_modes(system: RotorSystem, omega: float) -> tuple[np.ndarray, np.ndarray, float]:
    """A turning shaft's modes, by Hill's method: (λ, shapes (n_dof, k), drift), Im λ > 0.

    A(t) = A_0 + Σ_k A_k e^{ikφ}, k = ±p, ±2p, ... (p = spacing: 2 for an
    asymmetric shaft, 1 for a breathing crack); a Floquet mode z = e^{λt} Σ c_j
    e^{ijpΩt} has λ c_j = (A_0 - ijpΩ) c_j + Σ_k A_k c_{j-k/p}, an eigenproblem in
    the c_j, |j| ≤ HILL_ORDER. Each
    mode turns up once for every j, shifted by ipΩ (and so does its conjugate);
    the copies kept are those whose kinetic energy is centred on c_0, within half
    a harmonic (the copies near the truncation are the inaccurate ones, and their
    energy is off at the edges). A whirl locked to pX/2 has its energy split
    evenly between two harmonics, so two of its copies sit on that edge; turned
    to Im λ > 0 they are one λ, kept once. Its shape is c_0. drift is as
    whirl_modes's, from the roots that do not oscillate.
    """
    n, p, H = system.n_dof, system.spacing, HILL_ORDER
    A0 = system.state_matrix(omega).astype(complex)
    size, blocks = A0.shape[0], 2 * H + 1
    big = np.zeros((blocks * size, blocks * size), dtype=complex)
    for i in range(blocks):
        d = slice(i * size, (i + 1) * size)
        big[d, d] = A0 - 1j * (i - H) * p * omega * np.eye(size)
    for k, e in system._periodic_terms[1].items():
        Ak = -system.mass_inverse @ e
        for i in range(blocks - k // p):  # A_k takes c_j to c_{j + k/p}, its conjugate back
            lo, hi = slice(i * size, (i + 1) * size), slice((i + k // p) * size, (i + k // p + 1) * size)
            big[hi, lo][n : 2 * n, :n] = Ak
            big[lo, hi][n : 2 * n, :n] = Ak.conj()
    lam, vec = np.linalg.eig(big)
    disp = vec.reshape(blocks, size, -1)[:, :n]  # (harmonic, DOF, mode)
    M = system._matrices[0]
    energy = np.real(np.einsum("hik,ij,hjk->hk", disp.conj(), M, disp))
    centre = np.arange(-H, H + 1) @ energy / energy.sum(axis=0)
    keep = np.abs(centre) <= 0.5 + 1e-6
    lam, shapes = lam[keep], disp[H][:, keep]
    flip = lam.imag < 0
    lam[flip], shapes[:, flip] = lam[flip].conj(), shapes[:, flip].conj()
    keep, drift = _oscillation(lam)
    lam, shapes, flip = lam[keep], shapes[:, keep], flip[keep]
    # A mode and its conjugate (turned) are one: pair each turned copy with an unturned one of the same λ. A
    # double root (a round rotor's two whirls) has two of each, and stays double.
    tol = 1e-7 * max(1.0, float(np.abs(lam).max())) if lam.size else 0.0
    single: list[int] = []
    paired: set[int] = set()
    for i in range(lam.size):
        mate = next((j for j in single if j not in paired and flip[j] != flip[i] and abs(lam[j] - lam[i]) <= tol), None)
        if mate is None:
            single.append(i)
        else:
            paired.add(mate)
    return lam[single], shapes[:, single], drift


@dataclass
class Campbell:
    """Damped natural frequencies against spin speed, and the critical speeds."""

    omega: np.ndarray  # (n_speeds,) rad/s
    freq_hz: np.ndarray  # (n_speeds, n_modes), each column one frequency-ordered branch
    zeta: np.ndarray  # (n_speeds, n_modes)
    whirl: np.ndarray  # (n_speeds, n_modes), > 0 forward
    criticals: list[tuple[float, float, float]]  # (speed in rad/s, whirl ratio, ζ) where a branch meets 1X


def campbell(system: RotorSystem, omega_max: float, n_speeds: int = 241, n_modes: int = 6) -> Campbell:
    """Sweep the speed 0..omega_max and find where each branch crosses the synchronous (1X) line.

    An asymmetric shaft's modes lock onto 1X between its two critical speeds, so
    its branches do not cross the line cleanly. Its critical speeds (and a cracked
    shaft's) are taken as those of a round shaft as stiff as each axis
    (RotorSystem.axis_system): where,
    turning with the shaft, the rotor has no stiffness left along that axis
    against the centrifugal force. With no damping and no gyroscopic coupling
    they are exactly the ends of the band where it is unstable.
    """
    omegas = np.linspace(0.0, omega_max, n_speeds)
    freq = np.full((n_speeds, n_modes), np.nan)
    zeta = np.full((n_speeds, n_modes), np.nan)
    whirl = np.zeros((n_speeds, n_modes))
    for i, w in enumerate(omegas):
        m = whirl_modes(system, w)
        k = min(n_modes, m.freq_hz.size)
        freq[i, :k], zeta[i, :k], whirl[i, :k] = m.freq_hz[:k], m.zeta[:k], m.whirl[:k]
    criticals = []
    if system.periodic:
        for stiff in (False, True):
            criticals += campbell(system.axis_system(stiff), omega_max, n_speeds, n_modes).criticals
        criticals.sort()
        return Campbell(omegas, freq, zeta, whirl, criticals)
    for j in range(n_modes):
        gap = 2.0 * math.pi * freq[:, j] - omegas
        for i in range(n_speeds - 1):
            g0, g1 = gap[i], gap[i + 1]
            if np.isfinite(g0) and np.isfinite(g1) and g0 > 0.0 >= g1:
                guess = float(omegas[i] + (omegas[i + 1] - omegas[i]) * g0 / (g0 - g1))
                wc = _refine_critical(system, j, guess, omegas[i], omegas[i + 1])
                m = whirl_modes(system, wc)
                if j < m.whirl.size:
                    criticals.append((wc, float(m.whirl[j]), float(m.zeta[j])))
    criticals.sort()
    return Campbell(omegas, freq, zeta, whirl, criticals)


def _refine_critical(system: RotorSystem, branch: int, guess: float, lo: float, hi: float) -> float:
    """A few secant steps on 2π f_branch(Ω) = Ω, kept within [lo, hi]."""

    def gap(w: float) -> float:
        f = whirl_modes(system, w).freq_hz
        return 2.0 * math.pi * f[branch] - w if branch < f.size else -w

    x0, x1 = lo, hi
    g0, g1 = gap(x0), gap(x1)
    for _ in range(30):
        if g1 == g0:
            break
        x2 = min(max(x1 - g1 * (x1 - x0) / (g1 - g0), lo), hi)
        x0, g0, x1, g1 = x1, g1, x2, gap(x2)
        if abs(x1 - x0) <= 1e-10 * max(1.0, abs(x1)):
            break
    return x1 if lo <= x1 <= hi else guess


def synchronous_response(system: RotorSystem, omegas: np.ndarray) -> np.ndarray:
    """Steady-state 1X response at constant speeds: Q (len(omegas), n_dof), complex.

    The unbalance, the disc's skew and the shaft's bow together. q(t) = Re(Q e^{iΩt}),
    with the keyphasor mark at angle φ = Ωt (along +x at t = 0). Above
    the stability onset this particular solution still exists, but the free whirl
    grows over it, so the rotor never settles to it.

    With an asymmetric or cracked shaft the response to a 1X load has other
    harmonics in it too (harmonic_response; on isotropic supports none, for the
    asymmetric shaft); this is its 1X part. A breathing crack's weight drives 1X
    as well: that is gravity_response's, and steady_harmonics adds the two.
    """
    M, _, G, _ = system.matrices()
    omegas = np.atleast_1d(np.asarray(omegas, dtype=float))
    out = np.empty((omegas.size, system.n_dof), dtype=complex)
    f = _rotating_force(system, omegas)
    if system.periodic:
        orders = harmonic_orders(system, odd=True)
        F = np.zeros((omegas.size, orders.size, system.n_dof), dtype=complex)
        F[:, _index(orders, 1)], F[:, _index(orders, -1)] = 0.5 * f, 0.5 * f.conj()
        return 2.0 * harmonic_response(system, omegas, orders, F)[:, _index(orders, 1)]
    for i, w in enumerate(omegas):
        Z = system.stiffness(w) - w * w * M + 1j * w * (system.damping(w) + w * G)
        out[i] = system.dynamic_solve(Z, f[i], 1j * w)
    return out


def _rotating_force(system: RotorSystem, omegas: np.ndarray) -> np.ndarray:
    """F (len(omegas), n_dof) complex: the 1X load at each speed is Re(F e^{iφ})."""
    load = system.rotating_load(omegas)
    # f_x + i f_y = P e^{iφ}: f_x = Re(P e^{iφ}), f_y = Re(-iP e^{iφ}).
    return system.merge(load, -1j * load)


def _index(orders: np.ndarray, k: int) -> int:
    return int(np.searchsorted(orders, k))


def harmonic_orders(system: RotorSystem, odd: bool | None) -> np.ndarray:
    """The harmonics of φ a turning shaft's steady state needs: odd (to a 1X load), even (to gravity) or all.

    An asymmetric shaft's deviatoric stiffness shifts a harmonic k to k ± 2, so a
    load at 1X reaches the odd ones and a constant load the even ones. On
    isotropic supports the chain closes at once: a forward whirl at 1X stays 1X
    (in x and y, ±1), and the 0X sag gives only a forward 2X (0, ±2). Otherwise
    HARMONICS each side. A breathing crack's shifts them by 1 too, so every load
    reaches every harmonic: 2 HARMONICS each side, whichever is asked.
    """
    if system.spacing == 1:
        return np.arange(-2 * HARMONICS, 2 * HARMONICS + 1)
    top = 1 if system.rotating_frame else 2 * HARMONICS - 1  # the top odd harmonic, the even one past it
    if odd is None:
        return np.arange(-top - 1, top + 2)
    return np.arange(-top, top + 1, 2) if odd else np.arange(-top - 1, top + 2, 2)


def harmonic_response(
    system: RotorSystem, omega: float | np.ndarray, orders: np.ndarray, loads: np.ndarray
) -> np.ndarray:
    """Harmonic balance at constant speed Ω: the periodic q = Σ Q_k e^{ikφ} under f = Σ F_k e^{ikφ}.

    orders (m,) are the harmonics k, in order; loads (..., m, n_dof) the F_k (F_{-k}
    = conj(F_k) for a real load). Returns the Q_k, (m, n_dof) for one speed or
    (len(omega), m, n_dof) for several (loads then one set for all, or one per
    speed). Row k reads (K(Ω) - k²Ω²M + ikΩ(C(Ω) + ΩG)) Q_k + Σ_j E_j Q_{k-j} =
    F_k, the turning stiffness being Σ_j E_j e^{ijφ} (E_{-j} = conj(E_j));
    harmonics past the ends are taken as zero. Magnetic bearings add their push
    D(ikΩ) to row k (support_stiffness); at k = 0 with integral action their
    journals are held at the centre.
    """
    M, C, G, K = system._matrices
    H = system._circulatory if system.speed_dependent else np.zeros_like(K)
    E = system._periodic_terms[1]
    n, m = system.n_dof, len(orders)
    w = np.atleast_1d(np.asarray(omega, dtype=float))
    F = np.broadcast_to(np.asarray(loads, dtype=complex), (w.size, m, n)).reshape(w.size, m * n)
    out = np.empty((w.size, m * n), dtype=complex)
    chunk = max(1, 2_000_000 // (m * n) ** 2)  # speeds solved at once, to bound the memory
    for c in range(0, w.size, chunk):
        wc = w[c : c + chunk, None, None]
        if system.films:  # the films' stiffness and damping are not linear in Ω
            Kw = np.array([system.stiffness(x) for x in w[c : c + chunk]])
            Cw = np.array([system.damping(x) for x in w[c : c + chunk]])
        else:
            Kw, Cw = K + wc * H, C
        A = np.zeros((wc.shape[0], m * n, m * n), dtype=complex)
        for i, k in enumerate(orders):
            s = k * wc
            d = slice(i * n, (i + 1) * n)
            A[:, d, d] = Kw - s * s * M + 1j * s * (Cw + wc * G)
            for j, kj in enumerate(orders):
                gap = int(k - kj)
                if gap and abs(gap) in E:
                    A[:, d, j * n : (j + 1) * n] = E[gap] if gap > 0 else E[-gap].conj()
        Fc = F[c : c + chunk].copy()
        if system.magnets:
            for i, k in enumerate(orders):
                for j, x in enumerate(wc[:, 0, 0]):
                    D, pinned = system.support_stiffness(1j * k * x)
                    A[j, i * n : (i + 1) * n, i * n : (i + 1) * n] += D
                    rows = [i * n + p for p in pinned]
                    A[j, rows] = 0.0
                    A[j, rows, rows] = 1.0
                    Fc[j, rows] = 0.0
        out[c : c + chunk] = np.linalg.solve(A, Fc[..., None])[..., 0]
    by_order = out.reshape(w.size, m, n)
    return by_order[0] if np.ndim(omega) == 0 else by_order


def gravity_response(system: RotorSystem, omegas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gravity's steady state at constant speeds: (mean sag Q_0, 2X vector Q_2), each (len(omegas), n_dof).

    q = Q_0 + Re(Q_2 e^{2iφ}) + (4X, ...: only on anisotropic supports; and 1X, 3X,
    ... with a breathing crack: see steady_harmonics). A round shaft just sags: Q_0
    = (K + ΩH)^{-1} f_g and Q_2 = 0. An asymmetric shaft's weight acts on a
    stiffness that changes twice per revolution, so it vibrates at 2X, most where
    2Ω meets a natural frequency: the gravity critical, near half the critical speed.
    """
    Q0, V = periodic_steady(system, omegas, gravity=True, rotating=False, top=2)
    return Q0, V[:, 1]


def steady_harmonics(system: RotorSystem, omegas: np.ndarray, top: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """The whole steady state, gravity and the 1X loads together: (Q_0 (len(omegas), n_dof), V (len, top, n_dof)).

    q = Q_0 + Re Σ_k V_k e^{ikφ}, k = 1 ... top (V[:, k - 1]). A round shaft has
    only the sag and the 1X (synchronous_response); a turning one has the others
    from harmonic balance. A breathing crack mixes the two: its weight drives 1X,
    2X and 3X, and the unbalance 2X and the mean too.
    """
    return periodic_steady(system, omegas, gravity=True, rotating=True, top=top)


def periodic_steady(
    system: RotorSystem, omegas: np.ndarray, gravity: bool, rotating: bool, top: int
) -> tuple[np.ndarray, np.ndarray]:
    """As steady_harmonics, from gravity's load, the 1X loads, or both: (Q_0, V_k for k = 1 ... top)."""
    omegas = np.atleast_1d(np.asarray(omegas, dtype=float))
    n = system.n_dof
    Q0 = np.zeros((omegas.size, n), dtype=complex)
    V = np.zeros((omegas.size, top, n), dtype=complex)
    gravity = gravity and bool(system.gravity)
    if not system.periodic:
        if gravity:
            Q0[:] = system.static_deflection(omegas)
        if rotating and top:
            V[:, 0] = synchronous_response(system, omegas)
        return Q0, V
    if not gravity and not rotating:
        return Q0, V
    orders = harmonic_orders(system, None if gravity and rotating else not gravity)
    # Each speed once: a steady run asks for the same one again and again.
    speeds, back = np.unique(omegas, return_inverse=True)
    F = np.zeros((speeds.size, orders.size, n), dtype=complex)
    if gravity:
        F[:, _index(orders, 0)] = [system.gravity_load(w) for w in speeds] if system.films else system.gravity_load()
    if rotating:
        f = _rotating_force(system, speeds)
        F[:, _index(orders, 1)], F[:, _index(orders, -1)] = 0.5 * f, 0.5 * f.conj()
    Q = harmonic_response(system, speeds, orders, F)[back.ravel()]
    if 0 in orders:
        Q0 = Q[:, _index(orders, 0)]
    for k in range(1, top + 1):
        if k in orders:
            V[:, k - 1] = 2.0 * Q[:, _index(orders, k)]
    return Q0, V
