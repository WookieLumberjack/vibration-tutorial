"""The 3D rigid body page's Theory figures (see `figures`): a box turning about its centre of mass G at ω, one
of its bits of mass m at r from G and that bit's velocity ω × r, the start of L = Iω."""

from __future__ import annotations

import math

from . import figures as fig
from .style import colors


def point_svg() -> str:
    """A box in 3D (oblique), its axes x, y, z from G, ω through G, a bit of mass at r on a corner, and its
    velocity ω × r, across both."""
    fg, muted, mass, body = colors.foreground, colors.muted, colors.mass[0], colors.mass[6]
    spin, speed = colors.mass[2], colors.force
    yaw, pitch = math.radians(30.0), math.radians(20.0)
    scale, ox, oy = 150.0, 150.0, 120.0

    def at(p: tuple[float, float, float]) -> tuple[float, float]:
        x, y, z = p
        across = x * math.cos(yaw) - y * math.sin(yaw)
        away = x * math.sin(yaw) + y * math.cos(yaw)
        return ox + scale * across, oy - scale * (z * math.cos(pitch) + away * math.sin(pitch))

    a, b, c = 0.45, 0.3, 0.18  # half sides
    corners = [(i * a, j * b, k * c) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    parts = [fig.line(*at(corners[i]), *at(corners[j]), mass, 1.5) for i, j in edges]
    gx, gy = at((0.0, 0.0, 0.0))
    for end, name in (((0.75, 0, 0), "x"), ((0, 0.62, 0), "y"), ((0, 0, 0.55), "z")):
        parts.append(fig.arrow(gx, gy, *at(end), body, 1.3))
        lx, ly = at(tuple(1.1 * v for v in end))  # type: ignore[arg-type]
        parts.append(fig.math_label(lx, ly + 5, name, body))
    w = (-0.25, 0.1, 0.62)  # ω, through G
    parts.append(fig.arrow(gx, gy, *at(w), spin, 2.2))
    wx, wy = at(tuple(1.08 * v for v in w))  # type: ignore[arg-type]
    parts.append(fig.math_label(wx + 8, wy, "ω", spin, "start"))
    r = corners[7]  # (a, b, c)
    rx, ry = at(r)
    parts.append(fig.line(gx, gy, rx, ry, fg, 1.2, "4,3"))
    parts.append(fig.math_label((gx + rx) / 2 + 4, (gy + ry) / 2 + 16, "r", fg))
    parts.append(fig.circle(rx, ry, 4.0, fg, fg))
    parts.append(fig.math_label(rx + 8, ry + 14, "m", fg, "start"))
    v = (w[1] * r[2] - w[2] * r[1], w[2] * r[0] - w[0] * r[2], w[0] * r[1] - w[1] * r[0])  # ω × r
    k = 1.1
    vx, vy = at((r[0] + k * v[0], r[1] + k * v[1], r[2] + k * v[2]))
    parts.append(fig.arrow(rx, ry, vx, vy, speed, 2.0))
    parts.append(fig.math_label(vx - 2, vy - 10, "ω × r", speed, "end"))
    parts.append(fig.circle(gx, gy, 3.0, muted, muted))
    parts.append(fig.math_label(gx - 8, gy + 14, "G", muted, "end"))
    return fig.svg(330, 220, "".join(parts))


FIGURES = {"fig:point": point_svg}
