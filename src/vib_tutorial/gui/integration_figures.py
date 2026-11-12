"""The Numerical integration page's Theory figures (see `figures`): the exact step and explicit Euler's in the
phase plane."""

from __future__ import annotations

import math

from . import figures as fig
from .style import colors

STEP = 0.5  # ωΔt drawn: a bit under 30° a step


def phase_svg() -> str:
    """The free oscillator in the plane of (x, ẋ/ω): the exact motion stays on its circle, each step turning ωΔt.
    Explicit Euler steps along the tangent, which always leads out of the circle: √(1 + ω²Δt²) larger a step."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    exact, euler = colors.mass[0], colors.force
    cx, cy, r = 84.0, 85.0, 38.0
    circle = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in [2 * math.pi * i / 72 for i in range(73)]]
    parts = [
        fig.line(cx - 72, cy, cx + 72, cy, faint, 1.0),
        fig.line(cx, cy - 78, cx, cy + 78, faint, 1.0),
        fig.math_label(cx + 76, cy + 4, "x", muted, "start"),
        fig.math_label(cx + 4, cy - 66, "ẋ/ω", muted, "start"),
        fig.curve(circle, exact, 1.5),
    ]
    for k in range(5):  # the exact steps: on the circle, turning ωΔt each (clockwise, as x = cos, ẋ/ω = −sin)
        a = k * STEP
        parts.append(fig.circle(cx + r * math.cos(a), cy + r * math.sin(a), 3.5, exact, fill=exact))
    x, y = 1.0, 0.0  # explicit Euler on (x, ẋ/ω): (x, y) + ωΔt (y, −x)
    for _ in range(4):
        nx, ny = x + STEP * y, y - STEP * x
        parts.append(fig.arrow(cx + r * x, cy - r * y, cx + r * nx, cy - r * ny, euler, 1.8))
        x, y = nx, ny
    parts += [
        fig.circle(181.0, 40.0, 4.0, exact, fill=exact),
        fig.text(192, 44, "exact: on the circle,", fg, "start"),
        fig.text(192, 60, "turning ωΔt a step", fg, "start"),
        fig.line(174, 100, 188, 100, euler, 1.8),
        fig.text(192, 104, "explicit Euler: along", fg, "start"),
        fig.text(192, 120, "the tangent, outwards", fg, "start"),
    ]
    return fig.svg(340, 170, "".join(parts))


FIGURES = {"fig:phase": phase_svg}
