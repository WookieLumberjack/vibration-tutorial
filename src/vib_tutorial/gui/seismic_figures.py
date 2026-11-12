"""The Seismic response page's Theory figures (see `figures`): the chain as a shear building, and an elastic
against an elastic–plastic oscillator's force and displacement (the equal-displacement rule)."""

from __future__ import annotations

from . import figures as fig
from .style import colors

FLOORS = 4  # drawn; the page's building has five
SWAY = (0.0, 0.34, 0.64, 0.87, 1.0)  # a first mode's shape, ground to roof


def building_svg() -> str:
    """Floors (m) on storeys (k) on moving ground: each floor sways u_j, a storey's drift Δ_j is the difference of
    the floors at its ends, and its shear V_j carries the floors above it."""
    fg, muted, slab = colors.foreground, colors.muted, colors.mass[0]
    x0, w, ground, h, amp = 66.0, 90.0, 162.0, 34.0, 50.0
    parts = [fig.floor(x0 - 30, x0 + w + 40, ground, muted)]

    def left(j: int) -> float:
        return x0 + amp * SWAY[j]

    for j in range(1, FLOORS + 1):
        y, yb = ground - h * j, ground - h * (j - 1)
        xa, xb = left(j), left(j - 1)
        parts += [
            fig.line(xb, yb, xa, y, fg, 1.5),
            fig.line(xb + w, yb, xa + w, y, fg, 1.5),
            fig.rect(xa - 4, y - 7, w + 8, 7, slab, slab, 1.5, opacity=0.3),
        ]
        # the undeformed building, faint
        parts.append(fig.line(x0, yb, x0, y, muted, 1.0, dash="3,3"))
    parts += [
        fig.math_label(left(FLOORS) + w / 2, ground - h * FLOORS - 12, "m", fg),
        fig.math_label(x0 + w / 2 + 6, ground - h / 2 + 5, "k", fg),
        # the roof's sway u
        fig.line(x0, ground - h * FLOORS - 3, left(FLOORS), ground - h * FLOORS - 3, fg, 1.0),
        fig.math_label(x0 - 6, ground - h * FLOORS + 2, "u<sub>4</sub>", fg, "end"),
        # a storey's drift: floor 3 against floor 2
        fig.line(left(2) + w, ground - 2 * h, left(2) + w, ground - 3 * h - 14, muted, 1.0, dash="2,2"),
        fig.line(left(3) + w + 4, ground - 3 * h - 9, left(3) + w + 4, ground - 3 * h - 19, muted, 1.0),
        fig.line(left(2) + w, ground - 3 * h - 14, left(3) + w + 4, ground - 3 * h - 14, fg, 1.2),
        fig.math_label(left(3) + w + 10, ground - 3 * h - 9, "Δ<sub>3</sub>", fg, "start"),
        # the storey shear on storey 1
        fig.arrow(x0 + w + 46, ground - h / 2, x0 + w + 12, ground - h / 2, colors.force, 1.5),
        fig.math_label(x0 + w + 50, ground - h / 2 + 5, "V<sub>1</sub>", fg, "start"),
        # the ground's motion
        fig.arrow(x0 - 10, ground + 12, x0 + 30, ground + 12, colors.force, 1.5),
        fig.math_label(x0 - 14, ground + 17, "a<sub>g</sub>", fg, "end"),
    ]
    return fig.svg(310, 186, "".join(parts))


def ductility_svg() -> str:
    """Force against displacement: the elastic oscillator rises to F_el at u_max; an elastic–plastic one of the same
    stiffness yields at F_y = F_el/q, u_y, and (for a long period) reaches about the same u_max: μ = u_max/u_y ≈ q."""
    fg, muted, el, pl = colors.foreground, colors.muted, colors.mass[1], colors.mass[0]
    ox, oy, sx, sy = 46.0, 150.0, 210.0, 120.0
    q = 3.0

    def p(u: float, f: float) -> tuple[float, float]:
        return ox + sx * u, oy - sy * f

    parts = [
        fig.arrow(ox, oy, ox + sx + 30, oy, muted, 1.2),
        fig.arrow(ox, oy, ox, oy - sy - 18, muted, 1.2),
        fig.math_label(ox + sx + 34, oy + 5, "u", fg, "start"),
        fig.math_label(ox - 6, oy - sy - 16, "F", fg, "end"),
        fig.curve([p(0, 0), p(1, 1)], el, 2.0),
        fig.curve([p(0, 0), p(1 / q, 1 / q), p(1, 1 / q)], pl, 2.5),
        fig.line(*p(1, 0), *p(1, 1), muted, 1.0, dash="3,3"),
        fig.line(*p(1 / q, 0), *p(1 / q, 1 / q), muted, 1.0, dash="3,3"),
        fig.line(*p(0, 1), *p(1, 1), muted, 1.0, dash="3,3"),
        fig.line(*p(0, 1 / q), *p(1 / q, 1 / q), muted, 1.0, dash="3,3"),
        fig.math_label(ox - 6, p(0, 1)[1] + 5, "F<sub>el</sub>", fg, "end"),
        fig.math_label(ox - 6, p(0, 1 / q)[1] + 5, "F<sub>y</sub>", fg, "end"),
        fig.math_label(p(1 / q, 0)[0], oy + 18, "u<sub>y</sub>", fg),
        fig.math_label(p(1, 0)[0], oy + 18, "u<sub>max</sub>", fg),
        fig.text(p(0.62, 0.62)[0] - 8, p(0.62, 0.62)[1] - 6, "elastic", fg, "end"),
        fig.text(p(0.68, 1 / q)[0], p(0.68, 1 / q)[1] - 8, "yields", fg),
    ]
    return fig.svg(310, 178, "".join(parts))


FIGURES = {"fig:building": building_svg, "fig:ductility": ductility_svg}
