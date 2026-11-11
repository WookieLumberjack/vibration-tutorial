"""Sound radiated by a vibrating plate: radiation efficiency, coincidence, radiated power and the mass law.

The *Vibro-acoustics & SEA* page. The plate is Beams & plates' simply supported rectangle (`beams_plates.Plate`),
set in an infinite rigid baffle with air on one side. A bending wave on it radiates freely only once it outruns
sound: the bending wavenumber k_b = (ω²ρh/D)^¼ meets the acoustic one k = ω/c at the *critical* (coincidence)
frequency f_c = (c²/2π)√(ρh/D). Below it, only the parts of a mode its half waves don't cancel radiate (the corners
and the edges).

Each mode's radiation efficiency σ_mn = W / (ρc S ⟨v²⟩) is Wallace's double integral over the far field's
directions (`modal_efficiency`, all modes at once as a matrix product over a Gauss–Legendre grid), whose
low-frequency limit is 32k²ab/π⁵m²n² for odd m and n. `resonant_average` averages it over the modes resonant in
each band; `maidanik` is the textbook's band average in closed form (Maidanik, as corrected by Crocker and Price),
and `wave_efficiency` the infinite plate's 1/√(1 − f_c/f) above coincidence.

`point_drive` is the plate's response to a harmonic point force, mode by mode (hysteretic loss factor η): the
input power, the mean-square velocity and the radiated power (each mode with its own σ, the modes' mutual
radiation left out), against the infinite plate's input power F²/16√(Dρh) and the energy it settles to, P/ωη.
`transmission_loss` is the infinite plate's between two diffuse fields (incidence up to 78°): the mass law below
coincidence, its dip at f_c, and the damping-controlled rise above.

Working on it
-------------
Keep σ_mn exact for the model (Wallace's integral, enough nodes per oscillation: `_nodes`) and the other curves
its approximations; `tests/test_radiation.py` checks the limits, the band averages against Maidanik, the energy
balance of the point drive and the mass law.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .beams_plates import TWO_PI, Plate

RHO_AIR = 1.21  # kg/m³
C_AIR = 343.0  # m/s
FIELD_LIMIT = math.radians(78.0)  # field incidence: a diffuse field's angles up to 78°
P_REF = 2e-5  # Pa, for levels
W_REF = 1e-12  # W


@dataclass(frozen=True)
class Fluid:
    rho: float = RHO_AIR
    c: float = C_AIR

    @property
    def impedance(self) -> float:
        """ρc, the characteristic impedance (rayl)."""
        return self.rho * self.c


AIR = Fluid()


# --------------------------------------------------------------------------------------------- waves
def critical_frequency(plate: Plate, fluid: Fluid = AIR) -> float:
    """f_c = (c²/2π)√(ρh/D): where the plate's bending wave travels at the speed of sound."""
    return fluid.c**2 / TWO_PI / plate.scale


def bending_wavenumber(plate: Plate, f: float | np.ndarray) -> np.ndarray:
    """k_b = √ω (ρh/D)^¼."""
    return np.sqrt(TWO_PI * np.asarray(f, dtype=float) / plate.scale)


def bending_speed(plate: Plate, f: float | np.ndarray) -> np.ndarray:
    """The bending wave's phase speed ω/k_b = √(ω √(D/ρh)): it grows as √f."""
    return np.sqrt(TWO_PI * np.asarray(f, dtype=float) * plate.scale)


def wave_efficiency(f: float | np.ndarray, fc: float) -> np.ndarray:
    """An infinite plate's free bending wave: no sound below f_c, 1/√(1 − f_c/f) above (∞ at f_c)."""
    f = np.asarray(f, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f > fc, 1.0 / np.sqrt(np.maximum(1.0 - fc / f, 1e-300)), 0.0)


# --------------------------------------------------------------------------------------------- one mode
def _nodes(ka: float) -> int:
    """Gauss–Legendre nodes per angle: enough for the integrand's ka/2π oscillations, six to each."""
    return int(min(400, 24 + 1.2 * ka))


def _factor(alpha: np.ndarray, orders: np.ndarray) -> np.ndarray:
    """[cos or sin](α/2) / ((α/mπ)² − 1) for each m (rows) on the grid (columns): cos for odd m, sin for even.

    At α = mπ it is 0/0; there the limit is −(mπ/4) sin(mπ/2) for odd m and (mπ/4) cos(mπ/2) for even, so a
    node that falls near it takes that."""
    m = orders[:, None].astype(float)
    a = alpha[None, :]
    odd = (orders % 2 == 1)[:, None]
    top = np.where(odd, np.cos(0.5 * a), np.sin(0.5 * a))
    den = (a / (m * math.pi)) ** 2 - 1.0
    near = np.abs(den) < 1e-9
    limit = np.where(
        odd, -0.25 * m * math.pi * np.sin(0.5 * m * math.pi), 0.25 * m * math.pi * np.cos(0.5 * m * math.pi)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(near, limit, top / np.where(near, 1.0, den))
    return out


def modal_efficiency(plate: Plate, ms: np.ndarray, ns: np.ndarray, f: float, fluid: Fluid = AIR) -> np.ndarray:
    """σ_mn at frequency f for every m in `ms` (rows) and n in `ns` (columns): Wallace's (1972) integral

    σ_mn = (64k²ab / π⁶m²n²) ∫∫ [A_m(α) B_n(β)]² sin θ dθ dφ over θ, φ in (0, π/2),

    α = ka sin θ cos φ, β = kb sin θ sin φ, A_m = cos(α/2) (odd m) or sin(α/2) (even m) over ((α/mπ)² − 1)."""
    ms = np.asarray(ms, dtype=int)
    ns = np.asarray(ns, dtype=int)
    k = TWO_PI * f / fluid.c
    a, b = plate.a, plate.b
    n_theta = _nodes(k * math.hypot(a, b))
    x, w = np.polynomial.legendre.leggauss(n_theta)
    t = 0.25 * math.pi * (x + 1.0)  # θ and φ on (0, π/2)
    wt = 0.25 * math.pi * w
    th, ph = np.meshgrid(t, t, indexing="ij")
    weights = (np.outer(wt, wt) * np.sin(th)).ravel()
    alpha = (k * a * np.sin(th) * np.cos(ph)).ravel()
    beta = (k * b * np.sin(th) * np.sin(ph)).ravel()
    A = _factor(alpha, ms) ** 2
    B = _factor(beta, ns) ** 2
    integral = (A * weights) @ B.T
    scale = 64.0 * k * k * a * b / (math.pi**6 * np.outer(ms.astype(float) ** 2, ns.astype(float) ** 2))
    return scale * integral


def low_frequency_efficiency(plate: Plate, m: int, n: int, f: float | np.ndarray, fluid: Fluid = AIR) -> np.ndarray:
    """σ_mn ≈ 32k²ab/π⁵m²n² as k → 0, for odd m and n (an even index cancels it to a higher power of k)."""
    k = TWO_PI * np.asarray(f, dtype=float) / fluid.c
    return 32.0 * k * k * plate.a * plate.b / (math.pi**5 * m * m * n * n)


def classify(plate: Plate, m: int, n: int, f: float, fluid: Fluid = AIR) -> str:
    """Which of Wallace's regions a mode is in at f: its wavenumbers k_x = mπ/a, k_y = nπ/b against k."""
    k = TWO_PI * f / fluid.c
    kx, ky = m * math.pi / plate.a, n * math.pi / plate.b
    if math.hypot(kx, ky) <= k:
        return "a surface mode (above its own coincidence: radiates fully)"
    if kx > k and ky > k:
        return "a corner mode (only its corners radiate)"
    return "an edge mode (the edges along its longer half waves radiate)"


# --------------------------------------------------------------------------------------------- band averages
def third_octaves(f_lo: float, f_hi: float) -> np.ndarray:
    """The preferred third-octave centres (1000 · 10^(i/10)) from f_lo to f_hi."""
    lo = math.ceil(10 * math.log10(f_lo / 1000) - 1e-9)
    hi = math.floor(10 * math.log10(f_hi / 1000) + 1e-9)
    return 1000.0 * 10.0 ** (np.arange(lo, hi + 1) / 10.0)


def band_edges(centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return centres * 10 ** (-0.05), centres * 10**0.05


@dataclass(frozen=True)
class ModeSet:
    """The plate's modes up to a frequency: their indices, frequencies and own radiation efficiencies σ_mn(f_mn)."""

    m: np.ndarray
    n: np.ndarray
    freq: np.ndarray  # Hz
    sigma: np.ndarray  # each at its own resonance


def mode_set(plate: Plate, f_max: float, fluid: Fluid = AIR, count: int = 2000) -> ModeSet:
    """Every mode below f_max (at most `count`), each with σ at its own frequency."""
    s = plate.scale
    m_top = max(1, int(plate.a / math.pi * math.sqrt(TWO_PI * f_max / s)))
    n_top = max(1, int(plate.b / math.pi * math.sqrt(TWO_PI * f_max / s)))
    M, N = np.meshgrid(np.arange(1, m_top + 1), np.arange(1, n_top + 1), indexing="ij")
    freq = math.pi**2 * ((M / plate.a) ** 2 + (N / plate.b) ** 2) * s / TWO_PI
    keep = freq <= f_max
    m, n, fr = M[keep], N[keep], freq[keep]
    order = np.argsort(fr, kind="stable")[:count]
    m, n, fr = m[order], n[order], fr[order]
    sigma = np.empty(fr.size)
    # one integral grid per frequency: group the modes by close frequencies (2 %) and share each group's grid
    i = 0
    while i < fr.size:
        j = i + 1
        while j < fr.size and fr[j] <= fr[i] * 1.02:
            j += 1
        ms, ns = np.unique(m[i:j]), np.unique(n[i:j])
        grid = modal_efficiency(plate, ms, ns, float(np.sqrt(fr[i] * fr[j - 1])), fluid)
        sigma[i:j] = grid[np.searchsorted(ms, m[i:j]), np.searchsorted(ns, n[i:j])]
        i = j
    return ModeSet(m, n, fr, sigma)


def resonant_average(modes: ModeSet, centres: np.ndarray) -> np.ndarray:
    """The mean σ of the modes resonant in each third-octave band (NaN where a band has none)."""
    lo, hi = band_edges(centres)
    out = np.full(centres.size, np.nan)
    for i, (a, b) in enumerate(zip(lo, hi)):
        inside = (modes.freq >= a) & (modes.freq < b)
        if inside.any():
            out[i] = float(modes.sigma[inside].mean())
    return out


def maidanik(plate: Plate, f: float | np.ndarray, fluid: Fluid = AIR) -> np.ndarray:
    """The band-averaged σ of a baffled simply supported plate's resonant modes in closed form:

    below f_c, (λ_c P/S) g₂ + 2(λ_c²/S) g₁ (edge modes, and corner modes below f_c/2), with α = √(f/f_c),
    g₁ = (4/π⁴)(1 − 2α²)/α√(1 − α²) and g₂ = [(1 − α²) ln((1 + α)/(1 − α)) + 2α] / 4π²(1 − α²)^{3/2};
    near f_c, √(a/λ_c) + √(b/λ_c); above, 1/√(1 − f_c/f). Each piece is capped at the peak's value."""
    f = np.asarray(f, dtype=float)
    fc = critical_frequency(plate, fluid)
    lam = fluid.c / fc
    S, P = plate.a * plate.b, 2 * (plate.a + plate.b)
    peak = math.sqrt(plate.a / lam) + math.sqrt(plate.b / lam)
    out = np.empty_like(f)
    for i, fi in np.ndenumerate(f):
        if fi < fc:
            al = math.sqrt(fi / fc)
            g2 = ((1 - al * al) * math.log((1 + al) / (1 - al)) + 2 * al) / (4 * math.pi**2 * (1 - al * al) ** 1.5)
            g1 = (4 / math.pi**4) * (1 - 2 * al * al) / (al * math.sqrt(1 - al * al)) if fi < 0.5 * fc else 0.0
            s = lam * P / S * g2 + 2 * lam * lam / S * g1
        else:
            s = 1.0 / math.sqrt(1.0 - fc / fi) if fi > fc else peak
        out[i] = min(s, peak)
    return out


# --------------------------------------------------------------------------------------------- driven plate
SIGMA_STEP = 2 ** (1 / 8)  # the cached σ_mn(f) grid: eight frequencies an octave, 10 Hz to 40 kHz
SIGMA_GRID = 10.0 * SIGMA_STEP ** np.arange(int(8 * math.log2(4000)) + 1)


@lru_cache(maxsize=1024)
def _sigma_block(a: float, b: float, c: float, m_top: int, n_top: int, f: float) -> np.ndarray:
    """σ_mn(f) for every m ≤ m_top, n ≤ n_top: it depends on the panel's size and the fluid, not its thickness."""
    return modal_efficiency(Plate(a, b), np.arange(1, m_top + 1), np.arange(1, n_top + 1), f, Fluid(c=c))


@dataclass(frozen=True)
class PointDrive:
    """The plate's steady response to F cos ωt at one point, at each frequency: rms values (amplitude² / 2)."""

    freq: np.ndarray  # Hz
    input_power: np.ndarray  # W
    mean_square_velocity: np.ndarray  # ⟨v²⟩, space and time averaged, (m/s)²
    radiated_power: np.ndarray  # W
    radiation_efficiency: np.ndarray  # W / ρcS⟨v²⟩
    infinite_input: float  # W: an infinite plate's F²/16√(Dρh), the band average a finite one's tends to
    infinite_velocity: np.ndarray  # ⟨v²⟩ the infinite plate's input power settles to: P/(ωη M)


def point_drive(
    plate: Plate,
    loss: float,
    point: tuple[float, float],
    freq: np.ndarray,
    force: float = 1.0,
    fluid: Fluid = AIR,
    modes: ModeSet | None = None,
) -> PointDrive:
    """Every mode below 1.5 × the top frequency, each a single oscillator of mass M/4 (its shape's mean square)
    driven by F φ(x₀, y₀), with stiffness ω_mn²(1 + iη). Each mode radiates ρcS σ_mn(f) ⟨v_mn²⟩; σ_mn(f) is
    worked out on `SIGMA_GRID`'s frequencies (cached: it depends only on the panel's size) and interpolated in
    log–log."""
    freq = np.asarray(freq, dtype=float)
    if modes is None:
        modes = mode_set(plate, 1.5 * float(freq.max()), fluid)
    x0, y0 = point
    phi0 = np.sin(modes.m * math.pi * x0 / plate.a) * np.sin(modes.n * math.pi * y0 / plate.b)
    mass = plate.surface_mass * plate.a * plate.b
    w = TWO_PI * freq[:, None]
    wn = TWO_PI * modes.freq[None, :]
    q = force * phi0[None, :] / (0.25 * mass * (wn * wn * (1 + 1j * loss) - w * w))  # modal amplitudes
    v = 1j * w * q
    p_in = 0.5 * force * np.real(np.sum(v * phi0[None, :], axis=1))
    v2 = 0.5 * 0.25 * np.sum(np.abs(v) ** 2, axis=1)  # time ½, space ¼

    # σ_mn(f) on a coarse grid of frequencies (every mode), interpolated
    m_top, n_top = int(modes.m.max()), int(modes.n.max())
    inside = (SIGMA_GRID >= freq.min() / SIGMA_STEP) & (SIGMA_GRID <= freq.max() * SIGMA_STEP)
    coarse = SIGMA_GRID[inside]
    sig = np.array([_sigma_block(plate.a, plate.b, fluid.c, m_top, n_top, float(fc)) for fc in coarse])
    sig_modes = sig[:, modes.m - 1, modes.n - 1]  # (f, mode)
    log_sig = np.log(np.maximum(sig_modes, 1e-300))
    lf, lc = np.log(freq), np.log(coarse)
    sigma_f = np.exp(np.array([np.interp(lf, lc, log_sig[:, j]) for j in range(modes.freq.size)]).T)
    radiated = fluid.impedance * plate.a * plate.b * 0.5 * 0.25 * np.sum(sigma_f * np.abs(v) ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = radiated / (fluid.impedance * plate.a * plate.b * v2)
    p_inf = force * force / (16.0 * math.sqrt(plate.rigidity * plate.surface_mass))
    v2_inf = p_inf / (TWO_PI * freq * loss * mass)
    return PointDrive(freq, p_in, v2, radiated, eff, p_inf, v2_inf)


def radiation_loss_factor(plate: Plate, f: float, fluid: Fluid = AIR) -> float:
    """η_rad = ρcσ/ωm″ (Maidanik's band-averaged σ, one side): the share of a panel's energy it radiates in a
    cycle, over 2π, as its own loss factor η is the share it dissipates. The radiated power over the input is
    then η_rad/(η + η_rad)."""
    return fluid.impedance * float(maidanik(plate, f, fluid)) / (TWO_PI * f * plate.surface_mass)


# --------------------------------------------------------------------------------------------- transmission
def oblique_transmission(plate: Plate, loss: float, f: float, theta: np.ndarray, fluid: Fluid = AIR) -> np.ndarray:
    """τ(θ) = 1 / |1 + Z cos θ / 2ρc| for an infinite plate between two half-spaces of the fluid, its wall
    impedance Z = iωm″ [1 − (f/f_c)² sin⁴θ (1 + iη)]."""
    m2 = plate.surface_mass
    w = TWO_PI * f
    r = f / critical_frequency(plate, fluid)
    s4 = np.sin(theta) ** 4
    Z = 1j * w * m2 * (1.0 - r * r * s4 * (1 + 1j * loss))
    return 1.0 / np.abs(1.0 + Z * np.cos(theta) / (2.0 * fluid.impedance)) ** 2


def transmission_loss(
    plate: Plate, loss: float, f: float | np.ndarray, fluid: Fluid = AIR, limit: float = FIELD_LIMIT
) -> np.ndarray:
    """10 log 1/τ̄, τ̄ = ∫τ(θ) sin θ cos θ dθ / ∫ sin θ cos θ dθ over 0 to `limit` (78°: field incidence).

    The coincidence dip, at the angle where the trace of sound matches the bending wave, is narrow (as narrow
    as η), so the integral is split there."""
    f = np.atleast_1d(np.asarray(f, dtype=float))
    x, w = np.polynomial.legendre.leggauss(64)
    fc = critical_frequency(plate, fluid)
    out = np.empty(f.size)
    for i, fi in enumerate(f):
        cuts = [0.0, limit]
        if fi > fc:
            hit = math.asin(math.sqrt(fc / fi))  # sin²θ = f_c/f: the trace of sound matches the bending wave
            if hit < limit:
                width = max(4 * loss, 1e-3) * math.tan(hit) / 4  # the dip's angular width, about η tan θ / 4
                cuts = sorted({0.0, max(0.0, hit - 8 * width), hit, min(limit, hit + 8 * width), limit})
        total = 0.0
        for lo, hi in itertools.pairwise(cuts):
            th = 0.5 * (hi - lo) * (x + 1) + lo
            total += (
                0.5
                * (hi - lo)
                * float(np.sum(w * oblique_transmission(plate, loss, fi, th, fluid) * np.sin(th) * np.cos(th)))
            )
        mean = total / (0.5 * math.sin(limit) ** 2)
        out[i] = -10.0 * math.log10(mean)
    return out


def mass_law(plate: Plate, f: float | np.ndarray, fluid: Fluid = AIR) -> np.ndarray:
    """Normal incidence: 10 log[1 + (ωm″/2ρc)²] ≈ 20 log(f m″) − 42 dB, 6 dB more for each doubling of f or m″."""
    x = TWO_PI * np.asarray(f, dtype=float) * plate.surface_mass / (2 * fluid.impedance)
    return 10.0 * np.log10(1.0 + x * x)


def coincidence_dip(plate: Plate, loss: float, fluid: Fluid = AIR) -> tuple[float, float]:
    """(f, TL) at the bottom of the diffuse field's coincidence dip, a little above f_c (the field stops at 78°,
    short of grazing): a golden-section search in log f between f_c and 2f_c."""
    fc = critical_frequency(plate, fluid)
    lo, hi = 0.0, math.log(2.0)
    g = (math.sqrt(5) - 1) / 2

    def tl(x: float) -> float:
        return float(transmission_loss(plate, loss, fc * math.exp(x), fluid)[0])

    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    fcv, fdv = tl(c), tl(d)
    for _ in range(40):
        if fcv < fdv:
            hi, d, fdv = d, c, fcv
            c = hi - g * (hi - lo)
            fcv = tl(c)
        else:
            lo, c, fcv = c, d, fdv
            d = lo + g * (hi - lo)
            fdv = tl(d)
    x = 0.5 * (lo + hi)
    return fc * math.exp(x), tl(x)
