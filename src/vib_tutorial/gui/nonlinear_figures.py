"""The Nonlinearity & forces page's Theory figures (see `figures`): the force laws of the elements the page adds
(a linear spring, a cubic one, a gap, Coulomb friction), and force identification's y = Gf, G lower triangular
with each column the impulse response shifted down to start at its sample."""

from __future__ import annotations

import math

from . import figures as fig
from .style import colors


def _panel(x0: float, title: str, law, x_label: str) -> str:
    """One force law: force up, stretch (or velocity) across, both through the panel's centre."""
    fg, muted, force = colors.foreground, colors.muted, colors.force
    w, mid, half = 66.0, 78.0, 40.0
    cx = x0 + w / 2
    pts = [(cx + 30 * u, mid - half * law(u)) for u in [i / 40 - 1 for i in range(81)]]
    return "".join(
        [
            fig.line(x0 + 2, mid, x0 + w - 2, mid, muted, 0.8),
            fig.line(cx, mid - half - 4, cx, mid + half + 4, muted, 0.8),
            fig.curve(pts, force, 2.0),
            fig.text(cx, 22, title, fg),
            fig.text(cx, mid + half + 22, x_label, muted),
        ]
    )


def laws_svg() -> str:
    """Force against stretch for a linear spring, a hardening cubic one (kδ + k₃δ³), a gap (k_g past ±g, nothing
    inside), and friction's force against velocity (±F_f, whatever the speed)."""
    gap = 0.4
    panels = [
        ("linear", lambda u: 0.8 * u, "stretch"),
        ("cubic", lambda u: 0.35 * u + 0.6 * u**3, "stretch"),
        ("gap", lambda u: 0.0 if abs(u) <= gap else 1.5 * (u - math.copysign(gap, u)), "stretch"),
        ("friction", lambda u: 0.55 * math.tanh(u / 0.03), "velocity"),
    ]
    parts = [_panel(4 + k * 74, *p) for k, p in enumerate(panels)]
    parts += [
        fig.math_label(4 + 2 * 74 + 33 - 12, 72, "−g", colors.muted, size=12),
        fig.math_label(4 + 2 * 74 + 33 + 12, 94, "g", colors.muted, size=12),
    ]
    return fig.svg(302, 150, "".join(parts))


def markov_svg() -> str:
    """y = G f over 8 samples: G's lower triangle shaded, two of its columns drawn as the impulse response
    starting on the diagonal, and the force sample each column carries."""
    fg, muted, band, force = colors.foreground, colors.muted, colors.mass[0], colors.force
    top, size, n = 22.0, 112.0, 8
    cell = size / n
    yx, gx = 14.0, 60.0
    fx = gx + size + 26
    tri = [(gx, top), (gx + size, top + size), (gx, top + size)]
    parts = [
        fig.area(tri, band, 0.15),
        fig.rect(gx, top, size, size, muted, "none", 1.0),
        fig.rect(yx, top, 14, size, fg, "none", 1.2),
        fig.rect(fx, top, 14, size, fg, "none", 1.2),
        fig.text(yx + 7, top + size + 18, "y", fg, italic=True),
        fig.text(gx + size / 2, top + size + 18, "G", fg, italic=True),
        fig.text(fx + 7, top + size + 18, "f", fg, italic=True),
        fig.text(yx + 30, top + size / 2 + 5, "=", fg),
    ]
    for j in (1, 4):
        x = gx + (j + 0.5) * cell
        pts = []
        for i in range(j, n):
            t = (i - j) / 3.0
            pts.append((x + 0.42 * cell * math.sin(2.2 * t) * math.exp(-0.25 * t) * 2.6, top + (i + 0.5) * cell))
        parts += [
            fig.line(x, top + (j + 0.5) * cell, x, top + size, muted, 0.8, dash="2,2"),
            fig.curve(pts, force, 1.8),
            fig.rect(fx + 1, top + j * cell + 1, 12, cell - 2, force, force, 1.0, opacity=0.6),
        ]
    parts += [
        fig.text(fx + 22, top + 1.5 * cell + 4, "f<sub>2</sub>", force, "start"),
        fig.text(fx + 22, top + 4.5 * cell + 4, "f<sub>5</sub>", force, "start"),
        fig.text(gx + size, top - 6, "zero above the diagonal", muted, "end"),
    ]
    return fig.svg(302, 160, "".join(parts))


FIGURES = {"fig:laws": laws_svg, "fig:markov": markov_svg}
