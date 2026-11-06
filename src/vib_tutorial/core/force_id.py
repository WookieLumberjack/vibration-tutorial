"""Force identification on the chain: the force that made a measured response, found by a regularized inverse.

The forward problem
-------------------
A force f(t) on one mass, sampled every h and linear between samples, drives the chain through the exact FOH
update, z_k = Φz_{k−1} + Γ₀f_{k−1} + Γ₁f_k, and a sensor reads y_k = c z_k + d f_k (a displacement, or an
acceleration, which feels the force at its own mass at once: d = 1/m there). So every sample is a sum over the
force's samples so far, y = G f, G lower triangular and Toeplitz for each sensor, its entries the Markov
parameters c Φ^{k−j−1}(ΦΓ₁ + Γ₀) (`markov`). Stacking two sensors stacks their G's.

The inverse problem
-------------------
Given y with noise, f = G⁻¹y is useless: G's singular values s_i fall over many decades (the chain filters out
whatever is fast, and a displacement sensor far from the force sees nothing of its last samples), and each
direction's share of the noise is divided by its s_i. Regularization keeps the large ones and damps the small:

* **truncated SVD** keeps the first k directions: f = Σ_{i≤k} (u_iᵀy / s_i) v_i;
* **Tikhonov** minimizes ‖Gf − y‖² + λ²‖f‖², which weights each direction by s_i² / (s_i² + λ²):
  f = Σ s_i / (s_i² + λ²) (u_iᵀy) v_i.

The **L-curve** plots ‖Gf − y‖ against ‖f‖ as λ varies, on log axes: too small a λ and the solution's norm
explodes with the noise, too large and the residual grows as the force is smoothed away; the corner, where the
curvature is largest, balances them (`l_curve`). The **discrepancy principle** picks the λ whose residual is
the noise's own size, when that is known. The **Picard plot** sets |u_iᵀy| beside s_i: where the data's
coefficients stop falling with the singular values, the rest is noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .model import ChainSystem
from .simulator import foh_discretize

TWO_PI = 2.0 * math.pi
SAMPLES = 300
LAMBDAS = 120  # points on the L-curve
MOVES = 0.01  # log-norm: the L-curve's ends, where neither norm has moved this far, aren't searched for its corner


class Shape(Enum):
    IMPACT = "An impact (half-sine)"
    DOUBLE = "Two impacts"
    BURST = "A random burst"


SHAPES = tuple(Shape)


class Output(Enum):
    DISPLACEMENT = "Displacement"
    ACCELERATION = "Acceleration"


OUTPUTS = tuple(Output)


class Method(Enum):
    INVERSE = "None: G⁻¹y"
    TSVD = "Truncated SVD"
    TIKHONOV = "Tikhonov"


METHODS = tuple(Method)


def step_for(system: ChainSystem, samples: int = SAMPLES, periods: float = 4.0) -> float:
    """h: a record of `samples` spanning `periods` of the chain's first mode."""
    M, _, K = system.matrices()
    w = np.sqrt(np.clip(np.linalg.eigvals(np.linalg.solve(M, K)).real, 0.0, None))
    f1 = float(np.min(w[w > 0])) / TWO_PI if np.any(w > 0) else 1.0
    return periods / (f1 * samples)


def force_history(shape: Shape, samples: int, h: float, width: float, seed: int = 1) -> np.ndarray:
    """The true force (N, peak 1 for the impacts, RMS ⅓ for the burst) at each sample."""
    t = np.arange(samples) * h
    if shape is Shape.BURST:
        rng = np.random.default_rng(seed)
        start, end = int(0.1 * samples), int(0.6 * samples)
        f = np.zeros(samples)
        raw = rng.standard_normal(end - start)
        kernel = np.hanning(max(round(width / h), 3))
        smooth = np.convolve(raw, kernel / kernel.sum(), mode="same")
        f[start:end] = smooth / np.sqrt(np.mean(smooth**2)) / 3 * np.hanning(end - start)
        return f
    hits = (0.1,) if shape is Shape.IMPACT else (0.1, 0.45)
    f = np.zeros(samples)
    for at in hits:
        t0 = at * samples * h
        inside = (t >= t0) & (t <= t0 + width)
        f[inside] += np.sin(math.pi * (t[inside] - t0) / width)
    return f


def markov(system: ChainSystem, force: int, sensors: tuple[int, ...], output: Output, h: float, samples: int):
    """G (len(sensors)·samples, samples): every sensor's sample k from the force's sample j, stacked by sensor."""
    n = system.n
    M, C, K = system.matrices()
    Minv = np.linalg.inv(M)
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -Minv @ K
    A[n:, n:] = -Minv @ C
    b = np.zeros((2 * n, 1))
    b[n:, 0] = Minv[:, force]
    Phi, G0, G1 = foh_discretize(A, b, h)
    g0, g1 = G0[:, 0], G1[:, 0]
    rows = np.zeros((len(sensors), 2 * n))
    direct = np.zeros(len(sensors))
    for r, s in enumerate(sensors):
        if output is Output.DISPLACEMENT:
            rows[r, s] = 1.0
        else:
            rows[r] = A[n + s]
            direct[r] = Minv[s, force]
    # the response at sample k to a unit force sample at 0 (and nothing else): z_0 = g1, z_1 = Φg1 + g0, then Φ
    pulse = np.zeros((samples, len(sensors)))
    z = g1.copy()
    pulse[0] = rows @ z + direct
    z = Phi @ z + g0
    for k in range(1, samples):
        pulse[k] = rows @ z
        z = Phi @ z
    G = np.zeros((len(sensors) * samples, samples))
    for r in range(len(sensors)):
        block = G[r * samples : (r + 1) * samples]
        for j in range(samples):
            block[j:, j] = pulse[: samples - j, r]
    return G


@dataclass(frozen=True)
class Problem:
    G: np.ndarray
    U: np.ndarray
    s: np.ndarray
    Vt: np.ndarray
    clean: np.ndarray  # G f, the noise-free measurement
    y: np.ndarray  # with the noise
    noise: float  # ‖noise‖

    def coefficients(self) -> np.ndarray:
        return self.U.T @ self.y


def set_up(G: np.ndarray, f: np.ndarray, noise: float, seed: int = 1) -> Problem:
    """The measurement Gf with white noise of RMS `noise` × each sensor's RMS, and G's SVD."""
    clean = G @ f
    samples = f.size
    rng = np.random.default_rng(seed)
    y = clean.copy()
    for r in range(clean.size // samples):
        part = clean[r * samples : (r + 1) * samples]
        rms = float(np.sqrt(np.mean(part**2)))
        y[r * samples : (r + 1) * samples] += noise * rms * rng.standard_normal(samples)
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    return Problem(G, U, s, Vt, clean, y, float(np.linalg.norm(y - clean)))


def tikhonov(p: Problem, lam: float) -> np.ndarray:
    return p.Vt.T @ (p.s / (p.s**2 + lam * lam) * p.coefficients())


def tsvd(p: Problem, k: int) -> np.ndarray:
    k = max(1, min(int(k), p.s.size))
    return p.Vt[:k].T @ (p.coefficients()[:k] / p.s[:k])


def inverse(p: Problem) -> np.ndarray:
    keep = p.s > p.s[0] * 1e-15  # beyond this G is singular to round-off
    return p.Vt[keep].T @ (p.coefficients()[keep] / p.s[keep])


@dataclass(frozen=True)
class LCurve:
    lams: np.ndarray
    residual: np.ndarray  # ‖Gf − y‖
    size: np.ndarray  # ‖f‖
    corner: float  # λ at the largest curvature


def l_curve(p: Problem, points: int = LAMBDAS) -> LCurve:
    lams = np.geomspace(p.s[0] * 1e-8, p.s[0], points)
    beta = p.coefficients()
    outside = float(np.linalg.norm(p.y) ** 2 - np.sum(beta**2))  # the part of y no f can reach
    filt = p.s[None] ** 2 / (p.s[None] ** 2 + lams[:, None] ** 2)
    residual = np.sqrt(np.sum(((1 - filt) * beta[None]) ** 2, axis=1) + max(outside, 0.0))
    size = np.sqrt(np.sum((filt * beta[None] / p.s[None]) ** 2, axis=1))
    x, y = np.log(residual), np.log(size)
    t = np.log(lams)
    dx, dy = np.gradient(x, t), np.gradient(y, t)
    ddx, ddy = np.gradient(dx, t), np.gradient(dy, t)
    kappa = (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-300)
    # below the smallest singular values that matter both norms stand still, and round-off there makes false
    # corners: look only where the curve moves
    moving = (x > x.min() + MOVES) & (y < y.max() - MOVES)
    moving[:2] = moving[-2:] = False
    inner = np.flatnonzero(moving) if moving.any() else np.arange(2, points - 2)
    corner = float(lams[inner[np.argmax(kappa[inner])]])
    return LCurve(lams, residual, size, corner)


def discrepancy(p: Problem) -> float:
    """λ whose residual ‖Gf − y‖ is the noise's own norm (bisection on log λ)."""
    beta = p.coefficients()
    outside = max(float(np.linalg.norm(p.y) ** 2 - np.sum(beta**2)), 0.0)

    def residual(lam: float) -> float:
        filt = p.s**2 / (p.s**2 + lam * lam)
        return math.sqrt(float(np.sum(((1 - filt) * beta) ** 2)) + outside)

    lo, hi = math.log(p.s[0] * 1e-12), math.log(p.s[0] * 10)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if residual(math.exp(mid)) > p.noise:
            hi = mid
        else:
            lo = mid
    return math.exp(0.5 * (lo + hi))


def error(estimate: np.ndarray, truth: np.ndarray) -> float:
    """‖estimate − truth‖ / ‖truth‖."""
    return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))
