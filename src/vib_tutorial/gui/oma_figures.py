"""The Operational modal analysis page's Theory figures (see `figures`): the chain shaken by forces nobody
measures, with a sensor on every mass, and SSI-cov's block Toeplitz matrix of correlations, factored into the
observability matrix and its partner."""

from __future__ import annotations

import math

from . import figures as fig
from .style import colors


def _wiggle(x: float, y: float, color: str) -> str:
    """A short random-looking force trace above a mass, and the force's arrow along the chain below it."""
    pts = [(x - 12 + 2 * i, y - 30 + 5 * math.sin(2.7 * i) * math.cos(1.3 * i)) for i in range(13)]
    return fig.curve(pts, color, 1.3) + fig.arrow(x - 12, y - 12, x + 12, y - 12, color, 1.8)


def ambient_svg() -> str:
    """The uniform 4-mass chain from the wall, a random force on every mass (traffic, wind: unmeasured) and an
    accelerometer under each; no force gauge."""
    fg, muted, mass, force = colors.foreground, colors.muted, colors.mass[0], colors.force
    y, wall, gap, w, h = 86.0, 18.0, 36.0, 32.0, 28.0
    parts = [fig.ground(wall, y - 22, y + 22, muted)]
    x = wall
    for i in range(4):
        left = x + gap
        mid = left + w / 2
        parts += [
            fig.spring(x, left, y, fg, coils=3, amplitude=5.0),
            fig.rect(left, y - h / 2, w, h, mass, mass, 2.0, opacity=0.2),
            fig.math_label(mid, y + 6, f"m<sub>{i + 1}</sub>", fg),
            _wiggle(mid, y - h / 2, force),
            fig.rect(mid - 5, y + h / 2, 10, 9, fg, fg, 1.0),
            fig.line(mid, y + h / 2 + 9, mid, y + h / 2 + 24, muted, 1.0),
        ]
        x = left + w
    parts += [
        fig.text(160, 24, "random forces, not measured", muted),
        fig.text(160, 150, "an accelerometer on every mass", muted),
    ]
    return fig.svg(302, 162, "".join(parts))


def toeplitz_svg() -> str:
    """The block Toeplitz matrix of SSI-cov with s = 4 block rows, R_4 … R_7 down its first column, equal to
    the observability matrix (C, CA, CA², CA³) times [A³G A²G AG G]."""
    fg, muted, band = colors.foreground, colors.muted, colors.mass[0]
    cell, top, left = 27.0, 30.0, 8.0
    parts = []
    for i in range(4):
        for j in range(4):
            lag = 4 + i - j
            x, y = left + j * cell, top + i * cell
            parts += [
                fig.rect(x, y, cell, cell, band, band, 1.0, opacity=0.08 + 0.04 * lag),
                fig.math_label(x + cell / 2, y + cell / 2 + 5, f"R<sub>{lag}</sub>", fg, size=12),
            ]
    eq = left + 4 * cell + 10
    parts.append(fig.text(eq + 4, top + 2 * cell + 5, "=", fg))
    ox = eq + 16
    for i, row in enumerate(("C", "CA", "CA<sup>2</sup>", "CA<sup>3</sup>")):
        parts += [
            fig.rect(ox, top + i * cell, 34, cell, colors.mass[1], colors.mass[1], 1.0, opacity=0.15),
            fig.math_label(ox + 17, top + i * cell + cell / 2 + 5, row, fg, size=12),
        ]
    gx = ox + 40
    for j, col in enumerate(("A<sup>3</sup>G", "A<sup>2</sup>G", "AG", "G")):
        parts += [
            fig.rect(gx + j * 29, top, 29, cell, colors.mass[2], colors.mass[2], 1.0, opacity=0.15),
            fig.math_label(gx + j * 29 + 14.5, top + cell / 2 + 5, col, fg, size=12),
        ]
    parts += [
        fig.text(left + 2 * cell, top - 10, "correlations R<sub>i</sub>", muted),
        fig.text(ox + 17, top + 4 * cell + 16, "observability", muted),
        fig.text(gx + 58, top + cell + 16, "its partner", muted),
    ]
    return fig.svg(302, 160, "".join(parts))


FIGURES = {"fig:ambient": ambient_svg, "fig:toeplitz": toeplitz_svg}
