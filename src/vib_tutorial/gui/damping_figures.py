"""The Damping models Theory's figures (see `figures`): a mass sliding on friction, and the Zener element."""

from __future__ import annotations

from . import figures as fig
from .style import colors


def friction_svg() -> str:
    """A mass on a spring sliding on a rough floor: the floor pushes back F_f against its velocity ẋ."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    mass, force = colors.mass[0], colors.force
    wall, left, right, top, floor = 30.0, 170.0, 260.0, 50.0, 120.0
    parts = [
        fig.ground(wall, 40, floor, muted),
        fig.floor(wall, 350, floor, muted),
        fig.spring(wall, left, 85, fg),
        fig.math_label((wall + left) / 2, 70, "k", fg),
        fig.rect(left, top, right - left, floor - top, mass, mass, 2.0, opacity=0.2),
        fig.math_label((left + right) / 2, (top + floor) / 2 + 6, "m", fg, size=fig.FONT + 6),
        # the rough contact: short strokes along the mass's base
        *(fig.line(x, floor - 1, x + 6, floor - 5, faint, 1.0) for x in range(int(left) + 6, int(right) - 6, 10)),
        # velocity to the right, friction on the base to the left
        fig.arrow(right + 10, 70, right + 70, 70, fg, 1.5),
        fig.math_label(right + 80, 75, "ẋ", fg, "start"),
        fig.arrow(right - 8, floor + 18, left + 22, floor + 18, force, 2.0),
        fig.math_label((left + right) / 2, floor + 36, "F<sub>f</sub>", fg),
        # x: from where the spring is unstretched
        fig.line(left, top - 4, left, 16, muted, 1.0, "4 3"),
        fig.arrow(left, 24, left + 50, 24, fg, 1.5),
        fig.math_label(left + 60, 29, "x", fg, "start"),
    ]
    return fig.svg(360, 160, "".join(parts))


def zener_svg() -> str:
    """The standard linear solid: a spring k0 in parallel with a Maxwell arm, a spring k1 and a dashpot c1 in
    series, between a wall and the point that moves x."""
    fg, muted, mass = colors.foreground, colors.muted, colors.mass[0]
    wall, end, upper, lower, joint = 40.0, 280.0, 45.0, 110.0, 150.0
    parts = [
        fig.ground(wall, 25, 130, muted),
        # the bar at the moving end (where the mass hangs on), and both branches from the wall to it
        fig.line(end, upper - 10, end, lower + 10, mass, 3.0),
        fig.spring(wall, end, upper, fg, coils=8),
        fig.math_label((wall + end) / 2, upper - 15, "k<sub>0</sub>", fg),
        fig.spring(wall, joint, lower, fg, coils=4),
        fig.math_label((wall + joint) / 2, lower - 15, "k<sub>1</sub>", fg),
        fig.circle(joint, lower, 2.5, fill=fg),
        fig.damper(joint, end, lower, fg),
        fig.math_label((joint + end) / 2, lower + 26, "c<sub>1</sub>", fg),
        fig.text((wall + joint) / 2, lower + 30, "Maxwell arm", muted),
        # the motion at the end
        fig.arrow(end, 78, end + 50, 78, fg, 1.5),
        fig.math_label(end + 58, 83, "x", fg, "start"),
    ]
    return fig.svg(360, 150, "".join(parts))


FIGURES = {"fig:friction": friction_svg, "fig:zener": zener_svg}
