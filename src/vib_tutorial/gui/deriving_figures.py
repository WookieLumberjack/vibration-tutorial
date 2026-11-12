"""The Deriving the equations Theory's figures (see `figures`): the car's body on its springs, and the double
pendulum with bob 2 cut free."""

from __future__ import annotations

import math
from collections.abc import Callable

from . import figures as fig
from .pendulum_figures import ceiling, hanging, pivot
from .style import colors


def _spring(color: str) -> Callable[[float, float, float], str]:
    return lambda a, b, y: fig.spring(a, b, y, color, coils=4, amplitude=6.0)


def _damper(color: str) -> Callable[[float, float, float], str]:
    return lambda a, b, y: fig.damper(a, b, y, color, half=7.0)


def _across(x1: float, x2: float, y: float, color: str) -> str:
    """A horizontal dimension line from x1 to x2 at y, arrowed both ways."""
    mid = (x1 + x2) / 2
    return fig.arrow(mid, y, x1, y, color, 1.0) + fig.arrow(mid, y, x2, y, color, 1.0)


def car_svg() -> str:
    """The body (m, J about G) on k_f, c_f a ahead of G and k_r, c_r b behind; G rises x_G and the body
    pitches θ, nose (right) up; P is a point e ahead of G."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    mass = colors.mass[0]
    scale = 70.0  # px per metre
    gx, top, bottom, road = 175.0, 55.0, 95.0, 160.0
    front, rear = gx + 1.2 * scale, gx - 1.5 * scale
    px = gx + 0.8 * scale
    parts = [
        fig.rect(rear - 45, top, front - rear + 90, bottom - top, mass, mass, 2.0, opacity=0.2),
        fig.floor(rear - 55, front + 55, road, muted),
    ]
    for x, k, c in ((front, "k<sub>f</sub>", "c<sub>f</sub>"), (rear, "k<sub>r</sub>", "c<sub>r</sub>")):
        parts += [
            fig.upright(x - 11, bottom, road, _spring(fg)),
            fig.upright(x + 11, bottom, road, _damper(fg)),
            fig.math_label(x - 22, 135, k, fg, "end"),
            fig.math_label(x + 24, 135, c, fg, "start"),
            fig.line(x, road, x, road + 20, faint, 1.0, "3 3"),
        ]
    parts += [
        # G, and its rise
        fig.circle(gx, 75, 4, fg, fg),
        fig.math_label(gx - 9, 80, "G", fg, "end"),
        fig.arrow(gx, 71, gx, 18, fg, 1.5),
        fig.math_label(gx + 6, 26, "x<sub>G</sub>", fg, "start"),
        # the pitch, nose up (counterclockwise)
        fig.arc(gx, 75, 26, -35, 35, fg, 1.5, arrow=True),
        fig.math_label(gx + 26, 52, "θ", fg, "start"),
        # P, e ahead of G
        fig.circle(px, 75, 3, fg, colors.background),
        fig.math_label(px + 7, 92, "P", fg, "start"),
        _across(gx, px, 104, muted),
        fig.math_label((gx + px) / 2, 120, "e", fg),
        fig.line(gx, 79, gx, 108, faint, 1.0, "3 3"),
        # a and b, under the road
        _across(rear, gx, road + 16, muted),
        _across(gx, front, road + 16, muted),
        fig.line(gx, road, gx, road + 20, faint, 1.0, "3 3"),
        fig.math_label((rear + gx) / 2, road + 34, "b", fg),
        fig.math_label((gx + front) / 2, road + 34, "a", fg),
        fig.text(front + 40, 48, "front", muted),
    ]
    return fig.svg(360, 205, "".join(parts))


def double_pendulum_svg() -> str:
    """Left: bobs m1, m2 on rods L1, L2 at angles θ1, θ2 from the vertical. Right: bob 2 cut free, its weight
    and rod 2's pull the only forces on it."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    mass, force = colors.mass[0], colors.force
    t1, t2 = 22.0, 40.0
    ox, oy, l1, l2 = 95.0, 24.0, 85.0, 80.0
    x1, y1 = hanging(ox, oy, l1, t1)
    x2, y2 = hanging(x1, y1, l2, t2)
    parts = [
        ceiling(55, 135, oy, muted),
        fig.line(ox, oy, ox, oy + 80, faint, 1.0, "4 3"),
        fig.line(x1, y1, x1, y1 + 62, faint, 1.0, "4 3"),
        fig.line(ox, oy, x1, y1, fg, 2.0),
        fig.line(x1, y1, x2, y2, fg, 2.0),
        pivot(ox, oy),
        fig.arc(ox, oy, 55, -90, -90 + t1, fg, 1.2, arrow=True),
        fig.math_label(ox - 5, oy + 62, "θ<sub>1</sub>", fg, "end"),
        fig.arc(x1, y1, 38, -90, -90 + t2, fg, 1.2, arrow=True),
        fig.math_label(x1 + 12, y1 + 52, "θ<sub>2</sub>", fg, "start"),
        fig.math_label(ox + 0.3 * (x1 - ox) + 8, oy + 0.3 * (y1 - oy) + 4, "L<sub>1</sub>", fg, "start"),
        fig.math_label((x1 + x2) / 2 + 10, (y1 + y2) / 2 - 2, "L<sub>2</sub>", fg, "start"),
        fig.circle(x1, y1, 8, mass, mass),
        fig.math_label(x1 - 13, y1 + 4, "m<sub>1</sub>", fg, "end"),
        fig.circle(x2, y2, 8, mass, mass),
        fig.math_label(x2 + 12, y2 + 14, "m<sub>2</sub>", fg, "start"),
        fig.arrow(22, oy + 30, 22, oy + 80, muted, 1.5),
        fig.math_label(30, oy + 60, "g", fg, "start"),
    ]
    # bob 2, cut free: its weight, and rod 2's pull back along the rod toward bob 1
    bx, by = 285.0, 110.0
    a = math.radians(t2)
    ux, uy = -math.sin(a), -math.cos(a)  # from bob 2 toward bob 1
    parts += [
        fig.line(bx + 60 * ux, by + 60 * uy, bx, by, faint, 1.0, "4 3"),
        fig.circle(bx, by, 8, mass, mass),
        fig.math_label(bx + 14, by - 6, "m<sub>2</sub>", fg, "start"),
        fig.arrow(bx, by + 9, bx, by + 60, force, 2.0),
        fig.math_label(bx + 8, by + 52, "m<sub>2</sub>g", fg, "start"),
        fig.arrow(bx + 9 * ux, by + 9 * uy, bx + 52 * ux, by + 52 * uy, force, 2.0),
        fig.text(bx + 52 * ux - 6, by + 52 * uy - 6, "rod's pull", muted, "end"),
        fig.text(bx, 190, "bob 2, cut free", muted),
    ]
    return fig.svg(360, 205, "".join(parts))


FIGURES = {"fig:car": car_svg, "fig:double-pendulum": double_pendulum_svg}
