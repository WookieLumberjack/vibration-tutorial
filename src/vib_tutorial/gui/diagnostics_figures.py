"""The Machinery diagnostics page's Theory figure (see `figures`): a ball bearing end on, its n balls of
diameter d on the pitch circle D between the races, and a ball in its grooves cut through, touching the races
along a line at the contact angle α."""

from __future__ import annotations

import math

from . import figures as fig
from .style import colors


def bearing_svg() -> str:
    """Left, the bearing end on: the inner race turning at f_r, the cage carrying 9 balls round the pitch
    circle. Right, one ball cut through, its contact line tilted by α."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    race, ball, rate = colors.mass[0], colors.mass[1], colors.force
    cx, cy = 104.0, 112.0
    pitch, rb = 62.0, 13.0  # px: the pitch circle's radius, a ball's
    parts = [
        fig.circle(cx, cy, pitch + rb + 12, race, race, 1.5).replace("fill=", "fill-opacity='0.15' fill="),
        fig.circle(cx, cy, pitch + rb, race, colors.background, 1.5),
        fig.circle(cx, cy, pitch - rb, race, race, 1.5).replace("fill=", "fill-opacity='0.15' fill="),
        fig.circle(cx, cy, pitch - rb - 12, race, colors.background, 1.5),
        fig.circle(cx, cy, pitch, faint, width=1.0, dash="4,3"),
    ]
    for i in range(9):
        a = 2 * math.pi * i / 9 + math.pi / 2
        parts.append(fig.circle(cx + pitch * math.cos(a), cy - pitch * math.sin(a), rb, ball, ball, 1.2))
    # D across the pitch circle, d across the top ball
    parts += [
        fig.line(
            cx - pitch * math.cos(0.35),
            cy + pitch * math.sin(0.35),
            cx + pitch * math.cos(0.35),
            cy - pitch * math.sin(0.35),
            fg,
            1.0,
        ),
        fig.math_label(cx + 14, cy + 4, "D", fg, "start"),
        fig.line(cx - rb, cy - pitch, cx + rb, cy - pitch, fg, 1.0),
        fig.math_label(cx, cy - pitch - 18, "d", fg),
        fig.arc(cx, cy, 30.0, 200.0, 320.0, rate, 1.5, arrow=True),
        fig.math_label(cx - 4, cy + 22, "f<sub>r</sub>", rate),
        fig.text(cx, 214, "inner race turns, outer still", muted),
    ]
    # the section: a ball between two grooves, the contact line at α from the radial direction
    bx, by, r = 268.0, 108.0, 26.0
    alpha = math.radians(20.0)
    ux, uy = math.sin(alpha), -math.cos(alpha)
    parts += [
        fig.rect(bx - 44, by - r - 22, 88, 14, race, race, 1.5, 0.15),
        fig.rect(bx - 44, by + r + 8, 88, 14, race, race, 1.5, 0.15),
        fig.circle(bx, by, r, ball, ball, 1.5).replace("fill=", "fill-opacity='0.35' fill="),
        fig.line(bx, by - r - 30, bx, by + r + 30, faint, 1.0, "4,3"),
        fig.line(bx - (r + 10) * ux, by - (r + 10) * uy, bx + (r + 10) * ux, by + (r + 10) * uy, fg, 1.5),
        f"<path d='M {bx:.1f} {by - 40:.1f} A 40 40 0 0 1 {bx + 40 * ux:.1f} {by + 40 * uy:.1f}' fill='none' "
        f"stroke='{fg}' stroke-width='1.2'/>",
        fig.math_label(bx + 10, by - 44, "α", fg, "start"),
        fig.text(bx, by - r - 30, "outer race", muted),
        fig.text(bx, by + r + 40, "inner race", muted),
        fig.text(bx, 214, "a ball, cut through", muted),
    ]
    return fig.svg(340, 226, "".join(parts))


FIGURES = {"fig:bearing-geometry": bearing_svg}
