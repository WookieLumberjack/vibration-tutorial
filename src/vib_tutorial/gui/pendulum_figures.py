"""The Pendulums Theory's figures (see `figures`): the pendulums' geometry, which no plot on the page draws."""

from __future__ import annotations

import math

from . import figures as fig
from .style import colors

EQ_LABEL = "L<sub>" + fig.UPRIGHT.format("eq") + "</sub>"  # the equivalent length: "eq" is a label, upright


def ceiling(x1: float, x2: float, y: float, color: str) -> str:
    """A fixed ceiling at y from x1 to x2, hatched above."""
    parts = [fig.line(x1, y, x2, y, color, 2.0)]
    x = x1 + 4
    while x < x2:
        parts.append(fig.line(x, y, x + 8, y - 8, color, 1.0))
        x += 9.0
    return "".join(parts)


def pivot(x: float, y: float) -> str:
    """A pin the body turns on."""
    return fig.circle(x, y, 4, colors.foreground, colors.background, 1.5)


def hanging(x: float, y: float, length: float, deg: float) -> tuple[float, float]:
    """The point `length` from (x, y) along a line `deg` from the downward vertical, to the right."""
    a = math.radians(deg)
    return x + length * math.sin(a), y + length * math.cos(a)


def bar(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 16.0) -> str:
    """A rigid body drawn as a wide, translucent bar with round ends."""
    return (
        f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{color}' stroke-opacity='0.25' "
        f"stroke-width='{width}' stroke-linecap='round'/>"
    )


def span(x: float, y1: float, y2: float, color: str) -> str:
    """A vertical dimension line from y1 to y2 at x, arrowed both ways."""
    mid = (y1 + y2) / 2
    return fig.arrow(x, mid, x, y1, color, 1.0) + fig.arrow(x, mid, x, y2, color, 1.0)


def pendulum_svg() -> str:
    """A simple pendulum (a bob m on a string L) and a body on a pivot O (its centre of mass G a distance d
    below), each turned by θ, with the weight mg and its arm d sin θ."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    mass, force = colors.mass[0], colors.force
    theta = 25.0
    parts = []
    # a simple pendulum
    ox, oy, L = 90.0, 30.0, 130.0
    bx, by = hanging(ox, oy, L, theta)
    parts += [
        ceiling(50, 130, oy, muted),
        fig.line(ox, oy, ox, oy + L + 15, faint, 1.0, "4 3"),
        fig.line(ox, oy, bx, by, fg),
        pivot(ox, oy),
        fig.arc(ox, oy, 42, 270, 270 + theta, fg, 1.0),
        fig.math_label(*fig.point(ox, oy + 5, 54, 270 + theta / 2), "θ", fg),
        fig.math_label((ox + bx) / 2 + 12, (oy + by) / 2, "L", fg, "start"),
        fig.circle(bx, by, 10, mass, mass),
        fig.math_label(bx - 16, by + 5, "m", fg, "end"),
        fig.arrow(bx, by + 12, bx, by + 46, force, 2.0),
        fig.math_label(bx + 8, by + 40, "mg", fg, "start"),
        fig.text(80, 222, "a simple pendulum", muted),
    ]
    # a body on a pivot
    ox, oy, d = 265.0, 30.0, 95.0
    gx, gy = hanging(ox, oy, d, theta)
    ex, ey = hanging(ox, oy, 150.0, theta)
    parts += [
        ceiling(225, 305, oy - 18, muted),
        fig.line(ox, oy, ox, oy + 160, faint, 1.0, "4 3"),
        fig.line(ox, oy - 18, ox, oy, fg),
        bar(ox, oy, ex, ey, mass),
        fig.line(ox, oy, gx, gy, fg, 1.0),
        pivot(ox, oy),
        fig.math_label(ox - 10, oy + 18, "O", fg, "end"),
        fig.arc(ox, oy, 36, 270, 270 + theta, fg, 1.0),
        fig.math_label(*fig.point(ox, oy + 5, 48, 270 + theta / 2), "θ", fg),
        fig.math_label((ox + gx) / 2 + 12, (oy + gy) / 2 + 4, "d", fg, "start"),
        fig.circle(gx, gy, 3.5, fill=fg),
        fig.math_label(gx + 10, gy + 2, "G", fg, "start"),
        fig.line(ox, gy, gx, gy, muted, 1.0, "3 2"),
        fig.math_label(ox - 6, gy + 4, "d " + fig.UPRIGHT.format("sin") + " θ", fg, "end"),
        fig.arrow(gx, gy + 6, gx, gy + 50, force, 2.0),
        fig.math_label(gx - 8, gy + 44, "mg", fg, "end"),
        fig.text(285, 222, "a body on a pivot", muted),
    ]
    return fig.svg(380, 235, "".join(parts))


def kinds_svg() -> str:
    """A disc of radius r on a torsion wire k_t, twisted by θ; a mass m on an upright arm L, held by a torsion
    spring k_t at its foot and leaning by θ."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    mass, force = colors.mass[0], colors.force
    parts = []
    # a disc on a wire
    cx, top, cy = 100.0, 26.0, 120.0
    parts += [
        ceiling(60, 140, top, muted),
        fig.line(cx, top, cx, cy, fg, 2.0),
        fig.math_label(cx + 8, 70, "k<sub>t</sub>", fg, "start"),
        f"<ellipse cx='{cx:.1f}' cy='{cy:.1f}' rx='58' ry='14' stroke='{mass}' stroke-width='2' fill='{mass}' "
        "fill-opacity='0.2'/>",
        fig.line(cx, cy, cx + 58, cy, fg, 1.0, "4 3"),
        fig.math_label(cx + 30, cy - 4, "r", fg),
        fig.math_label(cx - 30, cy + 5, "m", fg),
        f"<path d='M {cx - 62:.1f} {cy + 10:.1f} A 64 22 0 0 0 {cx + 50:.1f} {cy + 30:.1f}' fill='none' "
        f"stroke='{fg}' stroke-width='1.5'/>",
        fig.head(cx + 50, cy + 30, math.radians(-12), fg),
        fig.math_label(cx, cy + 46, "θ", fg),
        fig.text(100, 196, "a disc on a wire", muted),
    ]
    # inverted, on a spring
    px, py, L, theta = 285.0, 160.0, 110.0, 20.0
    a = math.radians(theta)
    mx, my = px + L * math.sin(a), py - L * math.cos(a)
    parts += [
        fig.floor(235, 335, py + 6, muted),
        fig.line(px, py, px, py - L - 18, faint, 1.0, "4 3"),
        fig.line(px, py, mx, my, fg, 2.0),
        fig.arc(px, py, 10, 100, 400, fg, 1.0),
        fig.arc(px, py, 16, 120, 380, fg, 1.0),
        pivot(px, py),
        fig.math_label(px - 26, py - 14, "k<sub>t</sub>", fg, "end"),
        fig.arc(px, py, 50, 90 - theta, 90, fg, 1.0),
        fig.math_label(*fig.point(px, py, 62, 90 - theta / 2), "θ", fg),
        fig.math_label((px + mx) / 2 + 10, (py + my) / 2 + 14, "L", fg, "start"),
        fig.circle(mx, my, 11, mass, mass),
        fig.math_label(mx - 16, my - 6, "m", fg, "end"),
        fig.arrow(mx, my + 13, mx, my + 48, force, 2.0),
        fig.math_label(mx + 8, my + 42, "mg", fg, "start"),
        fig.text(285, 196, "inverted, on a spring", muted),
    ]
    return fig.svg(380, 210, "".join(parts))


def kater_svg() -> str:
    """A uniform rod pivoted a distance d above its centre G swings as a simple pendulum L_eq long; pivoted
    k_G²/d below G, on the other side, it swings with the same period: the two pivots are L_eq apart."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    mass = colors.mass[0]
    scale, x = 150.0, 110.0  # px per metre, the rod's line
    top, g = 20.0, 95.0  # the rod's ends are 75 px either side of G: L = 1 m
    d = 0.4
    other = (1 / 12) / d
    p1, p2 = g - d * scale, g + other * scale
    parts = [
        fig.rect(x - 8, top, 16, 2 * (g - top), mass, mass, 1.5, opacity=0.2),
        fig.circle(x, g, 3.5, fill=fg),
        fig.math_label(x + 14, g + 5, "G", fg, "start"),
        pivot(x, p1),
        pivot(x, p2),
        fig.line(x - 12, p1, x - 40, p1, faint, 1.0),
        fig.line(x - 12, g, x - 40, g, faint, 1.0),
        fig.line(x - 12, p2, x - 40, p2, faint, 1.0),
        span(x - 34, p1, g, muted),
        fig.math_label(x - 42, (p1 + g) / 2 + 5, "d", fg, "end"),
        span(x - 34, g, p2, muted),
        fig.math_label(x - 42, (g + p2) / 2 + 5, "k<sub>G</sub>²/d", fg, "end"),
        fig.line(x + 12, p1, x + 52, p1, faint, 1.0),
        fig.line(x + 12, p2, x + 52, p2, faint, 1.0),
        span(x + 46, p1, p2, muted),
        fig.math_label(x + 54, (p1 + p2) / 2 + 22, EQ_LABEL, fg, "start"),
        fig.text(x, 192, "two pivots, one period", muted),
    ]
    # the simple pendulum with the same period
    sx = 285.0
    parts += [
        ceiling(sx - 30, sx + 30, p1, muted),
        fig.line(sx, p1, sx, p2, fg),
        pivot(sx, p1),
        fig.circle(sx, p2, 9, mass, mass),
        span(sx + 30, p1, p2, muted),
        fig.math_label(sx + 38, (p1 + p2) / 2 + 5, EQ_LABEL, fg, "start"),
        fig.text(sx, 192, "the simple pendulum", muted),
    ]
    return fig.svg(380, 205, "".join(parts))


FIGURES = {"fig:pendulum": pendulum_svg, "fig:pendulum-kinds": kinds_svg, "fig:kater": kater_svg}
