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
        fig.math_label((wall + block) / 2, 31, "k<sub>1</sub>", fg),
        fig.spring(wall, block, 95, fg),
        fig.math_label((wall + block) / 2, 81, "k<sub>2</sub>", fg),
        fig.rect(block, 30, 40, 80, mass, mass, 2.0, opacity=0.2),
        fig.text(95, 145, "in parallel: forces add", muted),
    ]
    # end to end: one force through both, their stretches adding
    wall, joint, block = 205.0, 262.0, 320.0
    parts += [
        fig.ground(wall, 22, 118, muted),
        fig.spring(wall, joint, 70, fg, coils=4),
        fig.math_label((wall + joint) / 2, 56, "k<sub>1</sub>", fg),
        fig.circle(joint, 70, 2.5, fill=fg),
        fig.spring(joint, block, 70, fg, coils=4),
        fig.math_label((joint + block) / 2, 56, "k<sub>2</sub>", fg),
        fig.rect(block, 30, 40, 80, mass, mass, 2.0, opacity=0.2),
        fig.text(285, 145, "in series: stretches add", muted),
    ]
    return fig.svg(380, 160, "".join(parts))


def unbalance_svg() -> str:
    """A machine on its mount with a rotor turning inside: a small mass m_e at radius e, flung out by m_e eΩ²."""
    import math

    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    mass, force = colors.mass[0], colors.force
    left, right, top, bottom, ground = 70.0, 250.0, 42.0, 128.0, 180.0
    cx, cy, radius, angle = 165.0, 85.0, 28.0, 35.0
    spot_x, spot_y = fig.point(cx, cy, radius, angle)
    out_x, out_y = fig.point(cx, cy, radius + 34, angle)
    parts = [
        fig.floor(40, 300, ground, muted),
        fig.upright(110, bottom, ground, lambda a, b, y: fig.spring(a, b, y, fg, coils=4)),
        fig.math_label(90, 158, "k", fg),
        fig.upright(215, bottom, ground, lambda a, b, y: fig.damper(a, b, y, fg)),
        fig.math_label(238, 158, "c", fg),
        fig.rect(left, top, right - left, bottom - top, mass, mass, 2.0, opacity=0.2),
        fig.math_label(left + 16, bottom - 12, "m", fg),
        fig.circle(cx, cy, radius, faint, dash="4 3"),
        fig.circle(cx, cy, 2.5, fill=fg),
        fig.line(cx, cy, spot_x, spot_y, fg),
        fig.math_label(cx + 4, cy + 16, "e", fg, "start"),
        fig.circle(spot_x, spot_y, 6, fill=force),
        fig.math_label(spot_x - 14, spot_y - 6, "m<sub>e</sub>", fg, "end"),
        fig.arrow(
            spot_x + 8 * math.cos(math.radians(angle)),
            spot_y - 8 * math.sin(math.radians(angle)),
            out_x,
            out_y,
            force,
            2.0,
        ),
        fig.math_label(out_x + 4, out_y + 2, "m<sub>e</sub>eΩ²", fg, "start"),
        fig.arc(cx, cy, radius + 10, 130, 210, fg, 1.5, arrow=True),
        fig.math_label(*fig.point(cx, cy, radius + 24, 170), "Ω", fg),
        # x: the machine's motion, up and down on its mount
        fig.arrow(right + 40, 105, right + 40, 55, fg, 1.5),
        fig.math_label(right + 50, 70, "x", fg, "start"),
        fig.line(right + 4, 105, right + 46, 105, faint, 1.0, "4 3"),
    ]
    return fig.svg(380, 195, "".join(parts))


FIGURES = {"fig:oscillator": oscillator_svg, "fig:springs": springs_svg, "fig:unbalance": unbalance_svg}
