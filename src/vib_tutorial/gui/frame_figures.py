"""The Frame builder page's Theory figures (see `figures`): an element at an angle with its global and local
DOFs, the beam element's four Hermite cubics, and two elements' matrices overlapping where they share a node."""

from __future__ import annotations

import math

from . import figures as fig
from .style import colors

GAMMA = 30.0  # degrees: the drawn element's angle to x


def element_svg() -> str:
    """A beam from node 1 to node 2 at γ to x. At node 1 the frame's DOFs (u, v along x and y, θ a turn); at
    node 2 the element's own (u′ along it, v′ across it): T turns one set into the other."""
    fg, muted, glob, own = colors.foreground, colors.muted, colors.mass[0], colors.mass[1]
    g = math.radians(GAMMA)
    x1, y1, length = 62.0, 146.0, 184.0
    x2, y2 = x1 + length * math.cos(g), y1 - length * math.sin(g)
    ax, ay = math.cos(g), -math.sin(g)  # along the element (SVG's y down)
    nx, ny = -math.sin(g), -math.cos(g)  # across it, to its left
    parts = [
        fig.line(x1, y1, x2, y2, fg, 4.0),
        fig.circle(x1, y1, 4.0, fg, colors.background),
        fig.circle(x2, y2, 4.0, fg, colors.background),
        fig.text(x1 + 2, y1 + 40, "node 1", fg),
        fig.text(x2 + 8, y2 + 22, "node 2", fg, "start"),
        # the angle
        fig.line(x1, y1, x1 + 92, y1, muted, 1.0, dash="3,3"),
        fig.arc(x1, y1, 62, 0, GAMMA, muted, 1.0),
        fig.math_label(x1 + 74, y1 - 12, "γ", fg, "start"),
        # node 1: the frame's DOFs
        fig.arrow(x1, y1, x1 + 40, y1, glob, 1.8),
        fig.math_label(x1 + 42, y1 + 18, "u<sub>1</sub>", fg),
        fig.arrow(x1, y1, x1, y1 - 40, glob, 1.8),
        fig.math_label(x1 - 8, y1 - 34, "v<sub>1</sub>", fg, "end"),
        fig.arc(x1, y1, 16, 200, 330, glob, 1.5, arrow=True),
        fig.math_label(x1 - 20, y1 + 6, "θ<sub>1</sub>", fg, "end"),
        # node 2: the element's own
        fig.arrow(x2, y2, x2 + 32 * ax, y2 + 32 * ay, own, 1.8),
        fig.math_label(x2 + 40 * ax + 4, y2 + 40 * ay + 4, "u′<sub>2</sub>", fg, "start"),
        fig.arrow(x2, y2, x2 + 32 * nx, y2 + 32 * ny, own, 1.8),
        fig.math_label(x2 + 36 * nx - 4, y2 + 36 * ny + 2, "v′<sub>2</sub>", fg, "end"),
        fig.math_label(
            x1 + 0.5 * length * math.cos(g) - 16 * nx, y1 - 0.5 * length * math.sin(g) - 16 * ny + 5, "L", fg
        ),
        # the frame's axes
        fig.arrow(256, 182, 282, 182, muted, 1.0),
        fig.arrow(256, 182, 256, 156, muted, 1.0),
        fig.math_label(286, 187, "x", fg, "start"),
        fig.math_label(256, 150, "y", fg),
    ]
    return fig.svg(310, 196, "".join(parts))


def _hermite(n: int, xi: float) -> float:
    """Hermite cubic n (1–4) at ξ; the turns' shapes (2 and 4) per unit L."""
    return (
        1 - 3 * xi**2 + 2 * xi**3,
        xi - 2 * xi**2 + xi**3,
        3 * xi**2 - 2 * xi**3,
        -(xi**2) + xi**3,
    )[n - 1]


def hermite_svg() -> str:
    """The beam element's four shapes, each with one end DOF at 1 and the other three at 0: N₁ (v′₁ = 1), N₂
    (θ₁ = 1), N₃ (v′₂ = 1) and N₄ (θ₂ = 1). Each is a cubic: the beam's exact static shape under end loads."""
    fg, muted, curve = colors.foreground, colors.muted, colors.mass[0]
    w, unit, turn = 116.0, 30.0, 116.0  # px: the element's length, a unit translation, the turns' scale (L)
    panels = (
        (1, 22.0, 50.0, "N<sub>1</sub>: v′<sub>1</sub> = 1"),
        (2, 172.0, 50.0, "N<sub>2</sub>: θ<sub>1</sub> = 1"),
        (3, 22.0, 142.0, "N<sub>3</sub>: v′<sub>2</sub> = 1"),
        (4, 172.0, 142.0, "N<sub>4</sub>: θ<sub>2</sub> = 1"),
    )
    parts = []
    for n, x0, y0, label in panels:
        scale = unit if n in (1, 3) else turn
        pts = [(x0 + w * i / 40, y0 - scale * _hermite(n, i / 40)) for i in range(41)]
        parts += [
            fig.line(x0, y0, x0 + w, y0, muted, 1.0, dash="3,3"),
            fig.curve(pts, curve, 2.2),
            fig.circle(x0, y0 - (scale if n == 1 else 0), 3.0, fg, colors.background),
            fig.circle(x0 + w, y0 - (scale if n == 3 else 0), 3.0, fg, colors.background),
            fig.math_label(x0 + w / 2, y0 + 36, label, fg),
        ]
    return fig.svg(310, 190, "".join(parts))


def assembly_svg() -> str:
    """Two beams, node 1 to 2 and 2 to 3, and the 9 × 9 K they assemble into: beam 1's 6 × 6 lands on rows and
    columns 1–6, beam 2's on 4–9, and the two add on node 2's 3 × 3 block where they overlap."""
    fg, muted, one, two = colors.foreground, colors.muted, colors.mass[0], colors.mass[1]
    cell, gx, gy = 13.0, 88.0, 48.0
    parts = [
        # the beam above
        fig.line(40, 22, 150, 22, one, 4.0),
        fig.line(150, 22, 260, 22, two, 4.0),
        *(fig.circle(x, 22, 4.0, fg, colors.background) for x in (40.0, 150.0, 260.0)),
        fig.text(40, 40, "1", fg),
        fig.text(150, 40, "2", fg),
        fig.text(260, 40, "3", fg),
        fig.text(95, 14, "beam 1", fg),
        fig.text(205, 14, "beam 2", fg),
        fig.rect(gx, gy, 6 * cell, 6 * cell, one, one, 1.5, opacity=0.25),
        fig.rect(gx + 3 * cell, gy + 3 * cell, 6 * cell, 6 * cell, two, two, 1.5, opacity=0.25),
        fig.rect(gx, gy, 9 * cell, 9 * cell, muted, "none", 1.0),
    ]
    for k in range(1, 9):
        parts.append(fig.line(gx + k * cell, gy, gx + k * cell, gy + 9 * cell, colors.faint, 0.6))
        parts.append(fig.line(gx, gy + k * cell, gx + 9 * cell, gy + k * cell, colors.faint, 0.6))
    for node in range(3):
        y = gy + (3 * node + 1.5) * cell + 5
        parts.append(fig.text(gx - 8, y, f"node {node + 1}", fg, "end"))
    parts += [
        fig.text(gx + 9 * cell + 8, gy + 4.5 * cell + 5, "both add", fg, "start"),
        fig.line(gx + 9 * cell + 6, gy + 4.5 * cell, gx + 4.5 * cell + 6, gy + 4.5 * cell, muted, 1.0),
        fig.text(gx + 9 * cell + 8, gy + 8.5 * cell + 5, "0: no element", muted, "start"),
        fig.text(gx + 9 * cell + 8, gy + 8.5 * cell + 21, "joins them", muted, "start"),
        fig.math_label(gx + 4.5 * cell, gy + 9 * cell + 20, "K", fg),
    ]
    return fig.svg(310, 194, "".join(parts))


FIGURES = {"fig:element": element_svg, "fig:hermite": hermite_svg, "fig:assembly": assembly_svg}
