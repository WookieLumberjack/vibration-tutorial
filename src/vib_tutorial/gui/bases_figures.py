"""Coordinates & bases' Theory figures (see `figures`): the two masses, and a point written in a basis."""

from __future__ import annotations

from . import figures as fig
from .style import colors


def two_masses_svg() -> str:
    """Two masses between two walls: k₁ to the left wall, the coupling k₂ between them, k₃ to the right, a damper c beneath each; x₁, x₂."""
    fg, muted = colors.foreground, colors.muted
    a, b = colors.mass[0], colors.mass[1]
    left_wall, right_wall, top, bottom, floor = 20.0, 360.0, 50.0, 110.0, 122.0
    m1, m2, side = 105.0, 215.0, 60.0
    mid = (top + bottom) / 2
    up, down = mid - 12, mid + 13  # each spring with its damper beneath it
    parts = [
        fig.ground(left_wall, 30, floor, muted),
        fig.ground(right_wall, 30, floor, muted, side=1),
        fig.line(left_wall, floor, right_wall, floor, muted, 1.0),
        fig.spring(left_wall, m1, up, fg, coils=4, amplitude=6.0),
        fig.math_label((left_wall + m1) / 2, up - 10, "k<sub>1</sub>", fg),
        fig.damper(left_wall, m1, down, fg, half=6.0),
        fig.rect(m1, top, side, bottom - top, a, a, 2.0, opacity=0.2),
        fig.math_label(m1 + side / 2, mid + 6, "m<sub>1</sub>", fg),
        fig.spring(m1 + side, m2, up, fg, coils=4, amplitude=6.0),
        fig.math_label((m1 + side + m2) / 2, up - 10, "k<sub>2</sub>", fg),
        fig.damper(m1 + side, m2, down, fg, half=6.0),
        fig.rect(m2, top, side, bottom - top, b, b, 2.0, opacity=0.2),
        fig.math_label(m2 + side / 2, mid + 6, "m<sub>2</sub>", fg),
        fig.spring(m2 + side, right_wall, up, fg, coils=4, amplitude=6.0),
        fig.math_label((m2 + side + right_wall) / 2, up - 10, "k<sub>3</sub>", fg),
        fig.damper(m2 + side, right_wall, down, fg, half=6.0),
    ]
    for x, label in ((m1, "x<sub>1</sub>"), (m2, "x<sub>2</sub>")):
        parts += [
            fig.line(x + side / 2, top - 4, x + side / 2, 16, muted, 1.0, "4 3"),
            fig.arrow(x + side / 2, 24, x + side / 2 + 40, 24, fg, 1.5),
            fig.math_label(x + side / 2 + 50, 29, label, fg, "start"),
        ]
    parts.append(fig.text(190, 145, "each damper c, beneath its spring", muted))
    return fig.svg(380, 155, "".join(parts))


def basis_svg() -> str:
    """The (x₁, x₂) plane: two arrows v₁, v₂ from the origin, and a point x written as q₁v₁ + q₂v₂, its
    parallelogram dashed."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    one, two, point = colors.mass[0], colors.mass[1], colors.strong
    ox, oy = 110.0, 200.0
    v1, v2 = (70.0, -22.0), (18.0, -62.0)  # SVG's y points down
    q1, q2 = 1.6, 1.9
    a = (ox + q1 * v1[0], oy + q1 * v1[1])
    b = (ox + q2 * v2[0], oy + q2 * v2[1])
    x = (a[0] + q2 * v2[0], a[1] + q2 * v2[1])
    parts = [
        fig.arrow(30, oy, 360, oy, muted, 1.0),
        fig.math_label(355, oy + 20, "x<sub>1</sub>", muted, "end"),
        fig.arrow(ox, 225, ox, 20, muted, 1.0),
        fig.math_label(ox - 10, 30, "x<sub>2</sub>", muted, "end"),
        fig.line(*a, *x, faint, 1.5, "5 4"),
        fig.line(*b, *x, faint, 1.5, "5 4"),
        fig.line(ox, oy, *a, one, 1.0, "2 3"),
        fig.line(ox, oy, *b, two, 1.0, "2 3"),
        fig.arrow(ox, oy, ox + v1[0], oy + v1[1], one, 2.5),
        fig.math_label(ox + v1[0] + 4, oy + v1[1] + 18, "v<sub>1</sub>", one, "start"),
        fig.arrow(ox, oy, ox + v2[0], oy + v2[1], two, 2.5),
        fig.math_label(ox + v2[0] - 6, oy + v2[1] - 2, "v<sub>2</sub>", two, "end"),
        fig.circle(*a, 3.0, fill=one),
        fig.math_label(a[0] + 6, a[1] + 18, "q<sub>1</sub>v<sub>1</sub>", one, "start"),
        fig.circle(*b, 3.0, fill=two),
        fig.math_label(b[0] - 8, b[1], "q<sub>2</sub>v<sub>2</sub>", two, "end"),
        fig.circle(*x, 5.0, fill=point),
        fig.math_label(x[0] + 10, x[1] - 6, "x = q<sub>1</sub>v<sub>1</sub> + q<sub>2</sub>v<sub>2</sub>", fg, "start"),
    ]
    return fig.svg(380, 235, "".join(parts))


FIGURES = {"fig:two-masses": two_masses_svg, "fig:basis": basis_svg}
