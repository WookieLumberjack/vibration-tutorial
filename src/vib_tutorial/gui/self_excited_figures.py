"""The Self-excited vibration page's Theory figures (see `figures`): a square section galloping across a wind, the
wind it sees tilted by its own motion and the force that pushes it on; and the flutter page's typical section, its
quarter chord, elastic axis and centre of mass where the page's defaults put them (`Flutter`), on its plunge and
pitch springs."""

from __future__ import annotations

import math

import numpy as np

from ..core.self_excited import Flutter
from . import figures as fig
from .style import colors


def gallop_svg() -> str:
    """The square moving up at ẏ in a wind U from the left: relative to it the wind comes down at α = ẏ/U, and
    for a square the force across the wind then points up, the way it moves."""
    fg, muted, mass, force = colors.foreground, colors.muted, colors.mass[0], colors.force
    cx, cy, side = 176.0, 92.0, 52.0
    parts = [
        fig.rect(cx - side / 2, cy - side / 2, side, side, mass, mass, 1.5, opacity=0.35),
        fig.math_label(cx, cy + side / 2 + 18, "D", fg),
        fig.line(cx - side / 2, cy + side / 2 + 6, cx + side / 2, cy + side / 2 + 6, fg, 1.0),
    ]
    # the wind U, and the wind it sees: U from the left and ẏ down
    alpha = math.radians(24.0)
    x0, y0 = 40.0, cy
    length = 90.0
    parts += [
        fig.arrow(x0, y0, x0 + length, y0, muted, 1.5),
        fig.math_label(x0 + 10, y0 + 18, "U", muted),
        fig.arrow(x0, y0 - length * math.tan(alpha), x0 + length, y0, colors.strong, 2.0),
        fig.text(x0 - 4, y0 - length * math.tan(alpha) - 8, "the wind it sees", colors.strong, "start"),
        fig.arc(x0 + length, y0, 44.0, 180.0 - math.degrees(alpha), 180.0, fg, 1.2),
        fig.math_label(x0 + length - 52, y0 - 6, "α", fg),
        # its motion and the force
        fig.arrow(cx + side / 2 + 18, cy + 14, cx + side / 2 + 18, cy - 22, mass, 2.0),
        fig.math_label(cx + side / 2 + 26, cy - 2, "ẏ", mass, "start"),
        fig.arrow(cx, cy - side / 2 - 2, cx, cy - side / 2 - 40, force, 2.5),
        fig.math_label(cx + 8, cy - side / 2 - 26, "F<sub>y</sub>", force, "start"),
    ]
    return fig.svg(310, 160, "".join(parts))


def section_svg() -> str:
    """The typical section, leading edge to the left: the lift L at the quarter chord, the elastic axis e
    semichords behind it on its plunge spring k_h and pitch spring k_θ, the centre of mass x_cg semichords
    further back; plunge h down, pitch θ nose up."""
    fg, muted, mass, force = colors.foreground, colors.muted, colors.mass[0], colors.force
    m = Flutter()
    left, cy, b = 34.0, 74.0, 118.0  # the leading edge, the chord line and the semichord (px)
    u = np.linspace(0.0, 1.0, 81)
    thick = 0.12 * 2 * b * 5 * (0.2969 * np.sqrt(u) - 0.126 * u - 0.3516 * u**2 + 0.2843 * u**3 - 0.1036 * u**4)
    top = [(left + 2 * b * float(s), cy - float(t)) for s, t in zip(u, thick, strict=True)]
    bottom = [(left + 2 * b * float(s), cy + float(t)) for s, t in zip(u[::-1], thick[::-1], strict=True)]
    xq = left + 0.5 * b
    xe = xq + m.e * b
    xg = xe + m.x_cg * b
    parts = [
        fig.area(top + bottom, mass, 0.3),
        fig.curve(top + bottom + [top[0]], mass, 1.5),
        fig.line(left - 6, cy, left + 2 * b + 6, cy, muted, 1.0, "3,3"),
        fig.arrow(xq, cy, xq, cy - 50, force, 2.0),
        fig.math_label(xq - 8, cy - 40, "L", force, "end"),
        fig.text(xq - 4, cy + 30, "¼ chord", muted),
        fig.circle(xe, cy, 4.0, fg, "none", 1.8),
        fig.circle(xg, cy, 3.5, fg, fg),
        fig.math_label(xg + 8, cy - 7, "cg", fg, "start", size=fig.FONT),
        fig.upright(xe, cy + 4, cy + 58, lambda a, c, y: fig.spring(a, c, y, fg, coils=4, amplitude=5.0)),
        fig.floor(xe - 16, xe + 16, cy + 58, fg),
        fig.math_label(xe + 10, cy + 48, "k<sub>h</sub>", fg, "start"),
        fig.arc(xe, cy, 13.0, 200.0, 340.0, fg, 1.2),
        fig.math_label(xe + 14, cy + 22, "k<sub>θ</sub>", fg, "start"),
        fig.line(xq, cy - 56, xe, cy - 56, fg, 1.0),
        fig.line(xe, cy - 60, xe, cy - 6, muted, 1.0, "2,2"),
        fig.math_label((xq + xe) / 2 + 4, cy - 62, "eb", fg),
        fig.line(xe, cy - 30, xg, cy - 30, fg, 1.0),
        fig.line(xg, cy - 34, xg, cy - 6, muted, 1.0, "2,2"),
        fig.math_label(xg + 6, cy - 34, "x<sub>cg</sub>b", fg, "start"),
        fig.arrow(left + 2 * b + 16, cy - 12, left + 2 * b + 16, cy + 14, fg, 1.5),
        fig.math_label(left + 2 * b + 22, cy + 4, "h", fg, "start"),
        fig.arc(xe, cy, 0.95 * (xe - left), 168.0, 186.0, fg, 1.2),
        fig.head(*fig.point(xe, cy, 0.95 * (xe - left), 168.0), -math.radians(168.0) + math.pi / 2, fg),
        fig.math_label(left - 2, cy - 22, "θ", fg, "end"),
    ]
    return fig.svg(310, 168, "".join(parts))


FIGURES = {"fig:gallop": gallop_svg, "fig:section": section_svg}
