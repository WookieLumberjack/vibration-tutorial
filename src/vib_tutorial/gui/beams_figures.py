"""The Beams & plates page's Theory figures (see `figures`): Euler–Bernoulli's sections against Timoshenko's."""

from __future__ import annotations

import math

from . import figures as fig
from .style import colors

SLOPE = math.radians(20.0)  # the axis's slope, w_x
TURN = math.radians(10.0)  # Timoshenko's section rotation ψ, less than the slope: the rest is shear, γ
HALF = 18.0  # half the depth
SPAN = 60.0  # half the segment's length


def _segment(cx: float, cy: float, section: float, label: str) -> str:
    """A straight piece of bent beam rising at SLOPE, its sections turned `section` from the vertical."""
    fg, muted, mass = colors.foreground, colors.muted, colors.mass[0]
    ux, uy = math.cos(SLOPE), -math.sin(SLOPE)  # along the axis (y is down)
    nx, ny = math.sin(SLOPE), math.cos(SLOPE)  # normal to the axis
    sx, sy = math.sin(section), math.cos(section)  # along a section
    skew = HALF / (nx * sx + ny * sy)  # a section line spans the depth measured along the normal
    parts = []
    for side in (-1, 1):
        x0, y0 = cx - SPAN * ux + side * HALF * nx, cy - SPAN * uy + side * HALF * ny
        parts.append(fig.line(x0, y0, x0 + 2 * SPAN * ux, y0 + 2 * SPAN * uy, fg, 1.5))
    parts.append(fig.line(cx - SPAN * ux, cy - SPAN * uy, cx + SPAN * ux, cy + SPAN * uy, muted, 1.0, "4 3"))
    for t in (-40.0, 0.0, 40.0):
        px, py = cx + t * ux, cy + t * uy
        parts.append(fig.line(px - skew * sx, py - skew * sy, px + skew * sx, py + skew * sy, mass, 2.0))
    parts.append(fig.math_label(cx, 128, label, fg))
    return "".join(parts)


def sections_svg() -> str:
    """Left: Euler–Bernoulli, every section normal to the bent axis. Right: Timoshenko, each section turned by ψ,
    less than the axis's slope w_x; the difference, the shear angle γ = w_x − ψ, is the sections sliding."""
    fg, muted = colors.foreground, colors.muted
    cx, cy = 255.0, 62.0
    nx, ny = math.sin(SLOPE), math.cos(SLOPE)
    sx, sy = math.sin(TURN), math.cos(TURN)
    reach = HALF + 14
    parts = [
        _segment(85.0, cy, SLOPE, "ψ = w<sub>x</sub>"),
        _segment(cx, cy, TURN, "γ = w<sub>x</sub> − ψ"),
        fig.line(cx, cy, cx + reach * nx, cy + reach * ny, muted, 1.0, "3 3"),
        fig.math_label(cx + reach * (nx + sx) / 2 + 9, cy + reach * (ny + sy) / 2 + 6, "γ", fg),
        fig.text(85.0, 14, "Euler–Bernoulli", muted),
        fig.text(cx, 14, "Timoshenko", muted),
    ]
    return fig.svg(340, 140, "".join(parts))


FIGURES = {"fig:sections": sections_svg}
