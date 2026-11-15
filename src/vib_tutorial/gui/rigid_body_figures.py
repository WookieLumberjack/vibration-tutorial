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


def euler_svg() -> str:
    """Euler's z-x-z angles: space's X, Y, Z; the line of nodes n̂, turned φ from X about Z; the body's z, tipped θ
    from Z about n̂; and its x, turned ψ from n̂ about its z."""
    fg, muted, body, nodes = colors.foreground, colors.muted, colors.mass[0], colors.strong
    phi, theta, psi = math.radians(40.0), math.radians(40.0), math.radians(65.0)
    yaw, pitch = math.radians(5.0), math.radians(20.0)  # the screen halfway between n̂ and the way z tips
    scale, ox, oy = 105.0, 160.0, 150.0

    def at(p: tuple[float, ...]) -> tuple[float, float]:
        x, y, z = p
        across = x * math.cos(yaw) - y * math.sin(yaw)
        away = x * math.sin(yaw) + y * math.cos(yaw)
        return ox + scale * across, oy - scale * (z * math.cos(pitch) + away * math.sin(pitch))

    def turn(u: tuple[float, ...], v: tuple[float, ...], a: float, r: float = 1.0) -> tuple[float, ...]:
        """r(cos a u + sin a v): from u toward v (perpendicular unit vectors) by a."""
        return tuple(r * (math.cos(a) * p + math.sin(a) * q) for p, q in zip(u, v))

    Z = (0.0, 0.0, 1.0)
    n = (math.cos(phi), math.sin(phi), 0.0)  # the line of nodes
    tip = (math.sin(phi), -math.cos(phi), 0.0)  # the way z tips: horizontal, across n
    z = turn(Z, tip, theta)
    up = (z[1] * n[2] - z[2] * n[1], z[2] * n[0] - z[0] * n[2], z[0] * n[1] - z[1] * n[0])  # z × n
    x = turn(n, up, psi)
    gx, gy = at((0.0, 0.0, 0.0))
    steps = [k / 24 for k in range(25)]
    parts = [
        fig.curve([at(turn((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2 * math.pi * k, 0.8)) for k in steps], muted, 0.8, "2,3"),
        fig.curve([at(turn(n, up, 2 * math.pi * k, 0.8)) for k in steps], body, 0.8, "2,3"),  # the body's x-y plane
    ]
    for end, name, dx in (((1.25, 0, 0), "X", 0), ((0, 1.25, 0), "Y", -16), ((0, 0, 1.25), "Z", 0)):
        parts.append(fig.line(gx, gy, *at(end), muted, 1.0, "3,3"))
        lx, ly = at(tuple(1.1 * v for v in end))
        parts.append(fig.math_label(lx + dx, ly + 5, name, muted))
    parts.append(fig.arrow(gx, gy, *at(n), nodes, 1.6))
    lx, ly = at(tuple(1.15 * v for v in n))
    parts.append(fig.math_label(lx, ly + 5, "n̂", nodes))
    for v, name in ((z, "z"), (x, "x")):
        parts.append(fig.arrow(gx, gy, *at(v), body, 2.0))
        lx, ly = at(tuple(1.13 * c for c in v))
        parts.append(fig.math_label(lx, ly + 5, name, body))
    for u, v, a, r, name in (
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), phi, 0.4, "φ"),
        (Z, tip, theta, 0.62, "θ"),
        (n, up, psi, 0.3, "ψ"),
    ):
        parts.append(fig.curve([at(turn(u, v, a * k, r)) for k in steps], fg, 1.2))
        mx, my = at(turn(u, v, a / 2, r + 0.13))
        parts.append(fig.math_label(mx, my + 5, name, fg))
    return fig.svg(330, 230, "".join(parts))


FIGURES = {"fig:point": point_svg, "fig:euler": euler_svg}
