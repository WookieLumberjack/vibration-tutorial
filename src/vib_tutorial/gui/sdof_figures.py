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


def springs_svg() -> str:
    """Two springs side by side (in parallel) and end to end (in series), each holding a mass."""
    fg, muted, mass = colors.foreground, colors.muted, colors.mass[0]
    parts = []
    # side by side: both from the wall to the mass, each deflecting as much as the mass moves
    wall, block = 20.0, 130.0
    parts += [
        fig.ground(wall, 22, 118, muted),
        fig.spring(wall, block, 45, fg),
        fig.math_label((wall + block) / 2, 31, "k<tspan baseline-shift='sub' font-size='10'>1</tspan>", fg),
        fig.spring(wall, block, 95, fg),
        fig.math_label((wall + block) / 2, 81, "k<tspan baseline-shift='sub' font-size='10'>2</tspan>", fg),
        fig.rect(block, 30, 40, 80, mass, mass, 2.0, opacity=0.2),
        fig.text(95, 145, "in parallel: forces add", muted),
    ]
    # end to end: one force through both, their stretches adding
    wall, joint, block = 205.0, 262.0, 320.0
    parts += [
        fig.ground(wall, 22, 118, muted),
        fig.spring(wall, joint, 70, fg, coils=4),
        fig.math_label((wall + joint) / 2, 56, "k<tspan baseline-shift='sub' font-size='10'>1</tspan>", fg),
        fig.circle(joint, 70, 2.5, fill=fg),
        fig.spring(joint, block, 70, fg, coils=4),
        fig.math_label((joint + block) / 2, 56, "k<tspan baseline-shift='sub' font-size='10'>2</tspan>", fg),
        fig.rect(block, 30, 40, 80, mass, mass, 2.0, opacity=0.2),
        fig.text(285, 145, "in series: stretches add", muted),
    ]
    return fig.svg(380, 160, "".join(parts))


FIGURES = {"fig:oscillator": oscillator_svg, "fig:springs": springs_svg}
