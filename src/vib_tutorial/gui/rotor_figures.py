"""The rotor Theory's figures: SVG drawn from the theme's colours (see `figures`), named in the notes'
HTML as ``<img src="fig:...">``.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.fluid_bearing import attitude
from . import figures as fig
from .style import colors

# ------------------------------------------------------------------ the figures
MARK, LAG = 100.0, 60.0  # degrees: the mark's angle φ (the heavy spot at it), the lag


def spots_svg() -> str:
    """The shaft seen end on: axes, spin, keyphasor mark, heavy spot, high spot and the lag between them."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    shaft, heavy, high = colors.mass[0], colors.force, colors.mass[2]
    ox, oy, whirl, radius = 210.0, 165.0, 34.0, 78.0
    high_deg = MARK - LAG
    cx, cy = fig.point(ox, oy, whirl, high_deg)
    parts = [
        fig.arrow(ox - 150, oy, ox + 180, oy, muted, 1.0),
        fig.text(ox + 180, oy + 18, "x", fg, italic=True),
        fig.arrow(ox, oy + 140, ox, oy - 145, muted, 1.0),
        fig.text(ox + 12, oy - 135, "y", fg, italic=True),
        fig.circle(ox, oy, whirl, faint, dash="4 3"),
        fig.text(ox - 40, oy + 52, "whirl orbit", muted, "end"),
        fig.circle(cx, cy, radius, shaft, width=2.0),
        fig.circle(cx, cy, 3, fill=shaft),
        # the spin, round the disc's centre
        fig.arc(cx, cy, radius + 16, 150, 200, fg, 1.5, arrow=True),
        fig.text(*fig.point(cx, cy, radius + 32, 175), "Ω", fg),
        # the angle φ of the mark from +x, at the disc's centre
        fig.line(cx, cy, cx + radius + 10, cy, faint, 1.0, "4 3"),
        fig.line(cx, cy, *fig.point(cx, cy, radius, MARK), faint, 1.0, "4 3"),
        fig.arc(cx, cy, 22, 0, MARK, muted, 1.2),
        fig.text(*fig.point(cx, cy, 32, 20), "φ", fg, italic=True),
    ]
    # The keyphasor mark: a notch on the rim.
    mx, my = fig.point(cx, cy, radius, MARK)
    a = math.radians(MARK)
    parts.append(fig.head(mx, my, -a + math.pi, fg, 11))
    parts.append(fig.text(*fig.point(cx, cy, radius + 22, MARK - 8), "keyphasor mark", fg, "start"))
    # The heavy spot (at the mark here), and the high spot: the rim's point farthest from the bearings' centre.
    hx, hy = fig.point(cx, cy, radius - 12, MARK)
    parts += [
        fig.circle(hx, hy, 7, fill=heavy),
        fig.text(*fig.point(cx, cy, radius + 14, MARK + 22), "heavy spot", fg, "end"),
    ]
    px, py = fig.point(cx, cy, radius, high_deg)
    parts += [
        fig.line(ox, oy, px, py, high, 1.2, "5 3"),
        fig.circle(px, py, 6, high, colors.background, 2.5),
        fig.text(px + 12, py + 4, "high spot", fg, "start"),
        fig.arc(cx, cy, radius - 30, high_deg, MARK, heavy, 1.5, arrow=False),
        fig.text(*fig.point(cx, cy, radius - 44, (high_deg + MARK) / 2), "lag", fg),
        fig.text(ox + 8, oy + 16, "O", muted, "start"),
    ]
    return fig.svg(440, 330, "".join(parts))


def attitude_svg() -> str:
    """A plain journal bearing: the clearance circle, the locus, and the journal at ε and χ."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    bearing, journal, load = colors.structure, colors.mass[0], colors.force
    ox, oy, big, clear = 170.0, 165.0, 125.0, 42.0  # the sleeve's radius and the (exaggerated) clearance
    eps = 0.6
    chi = math.degrees(attitude(eps))
    deg = -90.0 + chi  # from the load line (down), turned with the spin
    jx, jy = fig.point(ox, oy, eps * clear, deg)
    tx, ty = fig.point(ox, oy, big, deg)
    parts = [
        fig.circle(ox, oy, big, bearing, width=2.5),
        fig.circle(jx, jy, big - clear, journal, width=2.0),
        fig.arc(jx, jy, big - clear - 18, 200, 250, fg, 1.5, arrow=True),
        fig.text(*fig.point(jx, jy, big - clear - 34, 225), "Ω", fg),
        fig.line(ox, oy - 12, ox, oy + big + 8, faint, 1.0, "4 3"),
        fig.line(ox, oy, tx, ty, muted, 1.0, "4 3"),
        fig.circle(ox, oy, clear, faint, dash="3 3"),
    ]
    # The short bearing's locus, ε from 0 (at 90°) to 1 (on the load line).
    pts = []
    for e in np.linspace(0.0, 0.999, 60):
        pts.append(fig.point(ox, oy, e * clear, -90.0 + math.degrees(attitude(e)) if e else 0.0))
    parts.append(
        "<polyline points='"
        + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        + f"' fill='none' stroke='{colors.mass[2]}' stroke-width='2'/>"
    )
    parts += [
        fig.arc(ox, oy, 30, -90.0, deg, fg, 1.2),
        fig.text(*fig.point(ox, oy, 54, (deg - 90.0) / 2), "χ", fg, italic=True),
        fig.circle(ox, oy, 2.5, fill=muted),
        fig.circle(jx, jy, 3.5, fill=journal),
        fig.arrow(jx, jy - 70, jx, jy - 5, load, 2.0),
        fig.text(jx + 8, jy - 56, "W", fg, "start", italic=True),
        fig.text(tx + 6, ty + 14, "thinnest film", fg, "start"),
    ]
    # Labels down the right, each with a short leader.
    lx = ox + big + 30
    legend = [
        (oy - 92, colors.structure, "bearing (sleeve)", "solid"),
        (oy - 68, journal, "journal", "solid"),
        (oy - 44, faint, "clearance circle, radius c", "dash"),
        (oy - 20, colors.mass[2], "locus, ε from 0 to 1", "solid"),
    ]
    for y, color, label, kind in legend:
        parts.append(fig.line(lx, y - 4, lx + 22, y - 4, color, 2.0, "4 3" if kind == "dash" else ""))
        parts.append(fig.text(lx + 30, y, label.replace("radius c", "radius c<sub>r</sub>"), fg, "start"))
    parts += [
        fig.text(lx, oy + 18, f"here ε = {eps} and", muted, "start"),
        fig.text(lx, oy + 36, f"χ = {chi:.0f}° from the load line", muted, "start"),
    ]
    return fig.svg(560, 330, "".join(parts))


def campbell_svg() -> str:
    """A schematic Campbell diagram: forward and backward branches, the 1X line, and where they cross."""
    fg, muted, _faint = colors.foreground, colors.muted, colors.faint
    forward, backward, hot = colors.mass[0], colors.mass[1], colors.force
    x0, y0, w, h = 60.0, 290.0, 420.0, 250.0
    top = 2.6  # speed and frequency, in units of the natural frequency at rest
    p = 0.6  # I_p/I_d: the forward branch rises more slowly than 1X, so it meets it

    def at(s: float, f: float) -> tuple[float, float]:
        return x0 + w * s / top, y0 - h * f / top

    def curve(fn, color: str, width: float = 2.0, dash: str = "") -> str:
        speeds = np.linspace(0.0, top, 80)
        pts = [at(s, fn(s)) for s in speeds if fn(s) <= top]
        d = f" stroke-dasharray='{dash}'" if dash else ""
        return (
            "<polyline points='"
            + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            + f"' fill='none' stroke='{color}' stroke-width='{width}'{d}/>"
        )

    def branch(sign: float):
        return lambda s: (sign * p * s + math.sqrt((p * s) ** 2 + 4.0)) / 2.0

    parts = [
        fig.arrow(x0, y0, x0 + w + 20, y0, muted, 1.0),
        fig.arrow(x0, y0, x0, y0 - h - 20, muted, 1.0),
        fig.text(x0 + w + 16, y0 + 20, "spin speed Ω", fg, "end"),
        fig.text(x0 - 10, y0 - h - 26, "whirl frequency", fg, "start"),
        curve(lambda s: s, fg, 1.5),
        fig.text(*at(2.3, 2.42), "1X", fg),
        curve(branch(1.0), forward),
        fig.text(*at(2.55, branch(1.0)(2.55) + 0.12), "forward", fg, "end"),
        curve(branch(-1.0), backward),
        fig.text(*at(2.55, branch(-1.0)(2.55) - 0.16), "backward", fg, "end"),
        fig.circle(*at(0, 1), 3, fill=muted),
        fig.text(x0 - 8, at(0, 1)[1] + 4, "ω<sub>n</sub>", fg, "end", italic=True),
    ]
    s_f, s_b = 1.0 / math.sqrt(1.0 - p), 1.0 / math.sqrt(1.0 + p)
    fx, fy = at(s_f, s_f)
    bx, by = at(s_b, s_b)
    parts += [
        fig.line(fx, y0, fx, y0 - 7, muted, 1.5),
        fig.circle(fx, fy, 6, fill=hot),
        fig.text(fx + 12, fy + 24, "critical speed", fg, "start"),
        fig.circle(bx, by, 6, hot, colors.background, 2.0),
        fig.text(bx + 12, by + 30, "backward crossing", fg, "start"),
        fig.text(fx, y0 + 18, "ω<sub>c</sub>", fg, italic=True),
    ]
    return fig.svg(540, 330, "".join(parts))


FIGURES = {"fig:spots": spots_svg, "fig:attitude": attitude_svg, "fig:campbell": campbell_svg}
