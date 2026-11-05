"""The Deriving the equations page's derivations as rich text: matrices, entries and the frequency equation."""

from __future__ import annotations

import math

import numpy as np

from ..core.practice import number

TWO_PI = 2.0 * math.pi
# The page's systems, tabs and Show items, shared by its modules (deriving.py, deriving_pendulum.py).
CAR, PENDULUM = range(2)
TAB_FBD, TAB_LAGRANGE, TAB_MODES, TAB_BUMP, TAB_PMODES, TAB_SWING, TAB_CHAOS, TAB_THEORY = range(8)
SHOW_MODE1, SHOW_MODE2, SHOW_MOTION = range(3)


def num(value: float) -> str:
    """A matrix entry: whole numbers past 100, four figures below."""
    value = float(value)
    if abs(value) < 1e-9:
        return "0"
    return f"{value:.0f}" if abs(value) >= 100 else f"{value:.4g}"


def matrix_html(name: str, A: np.ndarray) -> str:
    """A 2 × 2 matrix as a bracketed table, its coupling (off-diagonal) entries in bold."""

    def cell(i: int, j: int) -> str:
        text = num(A[i, j]).replace("-", "−")
        return f"<b>{text}</b>" if i != j and text != "0" else text

    bracket = "<td rowspan='2' valign='middle'><span style='font-size:22pt; font-weight:200'>{}</span></td>"
    rows = [
        f"<td align='right'>{cell(i, 0)}</td><td>&nbsp;&nbsp;</td><td align='right'>{cell(i, 1)}</td>" for i in range(2)
    ]
    return (
        f"<table cellspacing='0' cellpadding='1'><tr><td rowspan='2' valign='middle'>{name} =&nbsp;</td>"
        f"{bracket.format('[')}{rows[0]}{bracket.format(']')}</tr><tr>{rows[1]}</tr></table>"
    )


def matrices_html(m) -> str:
    """M, C and K side by side (M and K, if `m` has no C)."""
    named = [("M", m.M), ("C", getattr(m, "C", None)), ("K", m.K)]
    return (
        "<table><tr>"
        + "".join(
            f"<td valign='middle'>{matrix_html(n, A)}</td><td>&nbsp;&nbsp;</td>" for n, A in named if A is not None
        )
        + "</tr></table>"
    )


def dotted(name: str, n: int) -> str:
    """ẋ or ẍ for an x, θ̇ or θ̈ for a θ."""
    if name.startswith("x"):
        return "ẋẍ"[n - 1] + name[1:]
    return name + "̇̈"[n - 1]


def entries(letter: str, symbols: tuple[str, str, str], A: np.ndarray) -> str:
    """M₁₁ = m = 1200, M₁₂ = …: each entry in symbols and in numbers (once, where they're the same)."""
    return ", ".join(
        (f"{letter}{ij} = {s}" + ("" if s == num(v) else f" = {num(v)}")).replace("-", "−")
        for ij, s, v in zip(("₁₁", "₁₂", "₂₂"), symbols, (A[0, 0], A[0, 1], A[1, 1]))
    )


def frequencies_html(m) -> str:
    """det(K − ω²M) = 0 as a quadratic in ω², with the page's numbers, and its roots."""
    M, K = m.M, m.K
    a = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    b = M[0, 0] * K[1, 1] + M[1, 1] * K[0, 0] - M[0, 1] * K[1, 0] - M[1, 0] * K[0, 1]
    c = K[0, 0] * K[1, 1] - K[0, 1] * K[1, 0]
    disc = b * b - 4 * a * c
    roots = sorted(((b - math.sqrt(max(disc, 0.0))) / (2 * a), (b + math.sqrt(max(disc, 0.0))) / (2 * a)))
    f = [math.sqrt(max(w2, 0.0)) / TWO_PI for w2 in roots]
    return (
        "<p>det(K − ω²M) = 0 is a quadratic in ω²: "
        f"<b>{number(a, 4)} ω⁴ − {number(b, 4)} ω² + {number(c, 4)} = 0</b>, so ω² = {number(roots[0], 4)} and "
        f"{number(roots[1], 4)} (rad/s)², "
        f"<b>f = {f[0]:.3g} and {f[1]:.3g} Hz</b>.</p>"
    )
