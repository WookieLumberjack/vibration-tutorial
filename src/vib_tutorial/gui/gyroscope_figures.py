"""The Gyroscope page's Theory figures (see `figures`): the top in its cup with its Euler angles (the tilt θ from
the vertical, the precession φ about it, the spin ψ about the axis) and its sizes r and l; and a top on a table,
its round tip's sphere of radius R, the center of mass G a above the sphere's center, the contact P and the slip
there."""

from __future__ import annotations

import math

from . import figures as fig
from .style import colors


def _ellipse(cx: float, cy: float, rx: float, ry: float, angle: float, stroke: str, fill: str, opacity: float) -> str:
    return (
        f"<ellipse cx='{cx:.1f}' cy='{cy:.1f}' rx='{rx:.1f}' ry='{ry:.1f}' "
        f"transform='rotate({angle:.1f} {cx:.1f} {cy:.1f})' stroke='{stroke}' stroke-width='1.5' fill='{fill}' "
        f"fill-opacity='{opacity}'/>"
    )


def top_svg() -> str:
    """The disk tilted θ = 30° from the vertical on its stem, its center l up from the tip in the cup; the
    precession φ round the vertical, the spin ψ round the axis."""
    fg, muted, mass, strong, force = colors.foreground, colors.muted, colors.mass[0], colors.strong, colors.force
    tx, ty = 150.0, 150.0  # the tip
    theta = math.radians(30.0)
    l_px, r_px = 92.0, 44.0
    ux, uy = math.sin(theta), -math.cos(theta)  # the axis, up and to the right
    cx, cy = tx + l_px * ux, ty + l_px * uy
    parts = [
        fig.line(tx, ty, tx, 18, muted, 1.0, "3,3"),
        fig.math_label(tx - 6, 24, "ẑ", muted, "end"),
        fig.arc(tx, ty + 6, 14.0, 200, 340, fg, 1.5),  # the cup
        fig.line(tx, ty, tx + 1.25 * l_px * ux, ty + 1.25 * l_px * uy, fg, 2.0),
        _ellipse(cx, cy, r_px, 10.0, math.degrees(theta), mass, mass, 0.35),
        fig.circle(tx, ty, 2.5, fg, fg),
        # θ between the vertical and the axis
        f"<path d='M {tx:.1f} {ty - 40:.1f} A 40 40 0 0 1 {tx + 40 * ux:.1f} {ty + 40 * uy:.1f}' fill='none' "
        f"stroke='{fg}' stroke-width='1.2'/>",
        fig.math_label(tx + 9, ty - 44, "θ", fg, "start"),
        # l along the axis, r across it
        fig.math_label(tx + 0.5 * l_px * ux + 12, ty + 0.5 * l_px * uy + 8, "l", fg, "start"),
        fig.line(cx, cy, cx + r_px * math.cos(theta), cy + r_px * math.sin(theta), fg, 1.0),
        fig.math_label(cx + 0.5 * r_px * math.cos(theta) + 2, cy + 0.5 * r_px * math.sin(theta) + 16, "r", fg),
        # the spin ψ round the axis, beyond the disk
        _ellipse(tx + 1.2 * l_px * ux, ty + 1.2 * l_px * uy, 12.0, 4.0, math.degrees(theta), strong, "none", 0.0),
        fig.math_label(tx + 1.2 * l_px * ux + 14, ty + 1.2 * l_px * uy - 4, "ψ", strong, "start"),
        # the precession φ round the vertical, near the top
        _ellipse(tx, 40.0, 30.0, 7.0, 0.0, force, "none", 0.0),
        fig.head(tx + 30, 41, math.pi / 2, force),
        fig.math_label(tx - 36, 44, "φ", force, "end"),
    ]
    return fig.svg(310, 170, "".join(parts))


def tip_svg() -> str:
    """A top tilted on a table: its tip a sphere of radius R round the center C, the center of mass G a further up
    the axis, the contact P under C, and the slip v_P there."""
    fg, muted, mass, force, strong = colors.foreground, colors.muted, colors.mass[0], colors.force, colors.strong
    floor_y = 140.0
    R = 34.0
    a_px = 56.0
    theta = math.radians(28.0)
    ux, uy = math.sin(theta), -math.cos(theta)
    cx, cy = 130.0, floor_y - R  # the sphere's center, R above the table
    gx, gy = cx + a_px * ux, cy + a_px * uy
    px, py = cx, floor_y
    parts = [
        fig.floor(40, 280, floor_y, fg),
        fig.circle(cx, cy, R, mass, mass, 1.5).replace("fill=", "fill-opacity='0.2' fill="),
        fig.line(cx - 1.1 * R * ux, cy - 1.1 * R * uy, cx + 1.9 * a_px * ux, cy + 1.9 * a_px * uy, fg, 2.0),
        _ellipse(gx, gy, 46.0, 9.0, math.degrees(theta), mass, mass, 0.35),
        fig.circle(cx, cy, 2.5, fg, fg),
        fig.circle(gx, gy, 3.5, strong, strong),
        fig.math_label(gx - 8, gy - 4, "G", strong, "end"),
        fig.circle(px, py, 3.0, force, force),
        fig.math_label(px - 6, py - 6, "P", force, "end"),
        fig.line(cx, cy, px, py, muted, 1.0, "3,3"),
        fig.math_label(cx + 6, (cy + py) / 2 + 8, "R", fg, "start"),
        fig.math_label(cx + 0.5 * a_px * ux + 10, cy + 0.5 * a_px * uy + 8, "a", fg, "start"),
        fig.arrow(px, py + 1, px + 40, py + 1, force, 2.0),
        fig.math_label(px + 44, py - 4, "v<sub>P</sub>", force, "start"),
        fig.line(cx, cy, cx, cy - 60, muted, 1.0, "3,3"),
        f"<path d='M {cx:.1f} {cy - 30:.1f} A 30 30 0 0 1 {cx + 30 * ux:.1f} {cy + 30 * uy:.1f}' fill='none' "
        f"stroke='{fg}' stroke-width='1.2'/>",
        fig.math_label(cx + 4, cy - 34, "θ", fg, "start"),
    ]
    return fig.svg(310, 160, "".join(parts))


def frames_svg() -> str:
    """The two frames on the tilted disk, in 3D: the axis frame's e₁′ (horizontal, the line of nodes) and e₂′ (in
    the disk, up the slope), and the body's e₁ (the mark) and e₂, turned ψ from them about the shared e₃."""
    fg, muted, mass = colors.foreground, colors.muted, colors.mass[0]
    held, turned = colors.strong, colors.mass[6]  # the axis frame's, the body's: as the page's 3D view draws them
    theta, phi, psi = math.radians(35.0), math.radians(25.0), math.radians(55.0)
    yaw, pitch = math.radians(20.0), math.radians(15.0)
    e3 = (math.sin(theta) * math.sin(phi), -math.sin(theta) * math.cos(phi), math.cos(theta))
    n = (math.cos(phi), math.sin(phi), 0.0)  # ẑ × e₃, made unit: horizontal
    e2a = (e3[1] * n[2] - e3[2] * n[1], e3[2] * n[0] - e3[0] * n[2], e3[0] * n[1] - e3[1] * n[0])
    e1 = tuple(math.cos(psi) * a + math.sin(psi) * b for a, b in zip(n, e2a))
    e2 = tuple(-math.sin(psi) * a + math.cos(psi) * b for a, b in zip(n, e2a))
    scale, ox, oy = 95.0, 95.0, 188.0  # px per unit; the tip on the page

    def at(p: tuple[float, ...]) -> tuple[float, float]:
        x, y, z = p
        across = x * math.cos(yaw) - y * math.sin(yaw)
        away = x * math.sin(yaw) + y * math.cos(yaw)
        return ox + scale * across, oy - scale * (z * math.cos(pitch) + away * math.sin(pitch))

    def plus(a: tuple[float, ...], b: tuple[float, ...], k: float) -> tuple[float, ...]:
        return tuple(u + k * v for u, v in zip(a, b))

    G = plus((0.0, 0.0, 0.0), e3, 1.0)
    r = 0.62
    rim = [
        at(plus(G, tuple(math.cos(a) * u + math.sin(a) * v for u, v in zip(n, e2a)), r))
        for a in (2 * math.pi * k / 48 for k in range(49))
    ]
    tx, ty = at((0.0, 0.0, 0.0))
    gx, gy = at(G)
    parts = [
        fig.line(tx, ty, *at((0.0, 0.0, 1.7)), muted, 1.0, "3,3"),
        fig.math_label(at((0.0, 0.0, 1.7))[0] - 6, at((0.0, 0.0, 1.7))[1] + 6, "ẑ", muted, "end"),
        fig.area(rim, mass, 0.25),
        fig.curve(rim, mass, 1.2),
        fig.line(tx, ty, gx, gy, held, 2.0),
        fig.circle(tx, ty, 2.5, fg, fg),
        fig.arrow(gx, gy, *at(plus(G, e3, 0.75)), held, 1.8),
        fig.math_label(at(plus(G, e3, 0.8))[0] + 6, at(plus(G, e3, 0.8))[1], "e₃", fg, "start"),
    ]
    for vector, name, color, dash in (
        (n, "e₁′", held, "5,3"),
        (e2a, "e₂′", held, "5,3"),
        (e1, "e₁", turned, ""),
        (e2, "e₂", turned, ""),
    ):
        end = at(plus(G, vector, 0.95))
        parts.append(fig.arrow(gx, gy, *end, color, 1.8, dash))
        lx, ly = at(plus(G, vector, 1.08))
        parts.append(fig.math_label(lx, ly + 5, name, color))
    # ψ: from e₁′ to e₁ in the disk's plane
    arc = [
        at(plus(G, tuple(math.cos(a) * u + math.sin(a) * v for u, v in zip(n, e2a)), 0.42))
        for a in (psi * k / 16 for k in range(17))
    ]
    parts.append(fig.curve(arc, fg, 1.2))
    mx, my = at(plus(G, tuple(math.cos(0.2 * psi) * u + math.sin(0.2 * psi) * v for u, v in zip(n, e2a)), 0.56))
    parts.append(fig.math_label(mx, my + 5, "ψ", fg))
    parts.append(fig.text(300, 30, "axis frame: dashed", held, "end"))
    parts.append(fig.text(300, 48, "body: solid", turned, "end"))
    parts.append(fig.text(300, 66, "e₁′ horizontal", muted, "end"))
    return fig.svg(310, 200, "".join(parts))


FIGURES = {"fig:top": top_svg, "fig:tip": tip_svg, "fig:frames": frames_svg}
