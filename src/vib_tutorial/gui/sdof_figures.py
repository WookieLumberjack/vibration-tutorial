"""The Single oscillator Theory's figures (see `figures`): the oscillator itself."""

from __future__ import annotations

from . import figures as fig
from .style import colors


def oscillator_svg() -> str:
    """A mass m on a spring k and a damper c from a wall, on rollers, pushed by f(t); x from where it rests."""
    fg, muted = colors.foreground, colors.muted
    mass, force = colors.mass[0], colors.force
    wall, left, right, top, bottom, floor = 24.0, 170.0, 250.0, 50.0, 140.0, 152.0
    parts = [
        fig.ground(wall, 30, floor, muted),
        fig.line(wall, floor, 372, floor, muted, 1.0),
        fig.spring(wall, left, 72, fg),
        fig.math_label((wall + left) / 2, 56, "k", fg),
        fig.damper(wall, left, 118, fg),
        fig.math_label((wall + left) / 2, 145, "c", fg),
        fig.rect(left, top, right - left, bottom - top, mass, mass, 2.0, opacity=0.2),
        fig.math_label((left + right) / 2, (top + bottom) / 2 + 6, "m", fg, size=fig.FONT + 6),
        fig.circle(left + 22, bottom + 6, 6, muted),
        fig.circle(right - 22, bottom + 6, 6, muted),
        fig.arrow(right, 95, right + 75, 95, force, 2.0),
        fig.math_label(right + 42, 85, "f(t)", fg),
        # x: measured from the mass's position at rest, the spring unstretched
        fig.line(left, top - 4, left, 16, muted, 1.0, "4 3"),
        fig.arrow(left, 24, left + 60, 24, fg, 1.5),
        fig.math_label(left + 72, 29, "x", fg, "start"),
    ]
    return fig.svg(380, 175, "".join(parts))


FIGURES = {"fig:oscillator": oscillator_svg}
