"""The Substructuring page's Theory figure (see `figures`): the chain cut into substructures, with their interior
and boundary DOFs."""

from __future__ import annotations

import math

from . import figures as fig
from .style import colors

BOX = 26.0
HALF = 10.0


def cut_svg() -> str:
    """Six masses cut at m3: A (m1–m3, on the ground) and B (m3–m6), sharing m3. The interface m3 and the loaded tip
    m6 are boundary DOFs x_b; the rest are interior DOFs x_i."""
    fg, muted, a, b = colors.foreground, colors.muted, colors.mass[0], colors.mass[1]
    y, gap = 52.0, 44.0
    x0 = 18.0
    parts = [fig.ground(x0, y - 16, y + 16, muted)]
    xs = [x0 + 20 + gap * j for j in range(6)]
    # the substructures' spans, behind
    parts += [
        fig.rect(x0 + 2, y - 24, xs[2] + BOX + 4 - x0 - 2, 48, a, a, 1.0, opacity=0.08),
        fig.rect(xs[2] - 4, y - 28, xs[5] + BOX + 6 - xs[2] + 4, 56, b, b, 1.0, opacity=0.08),
        fig.text((x0 + xs[2] + BOX) / 2, y - 30, "A", fg),
        fig.text((xs[2] + xs[5] + BOX) / 2 + 8, y - 34, "B", fg),
    ]
    prev = x0
    for j, x in enumerate(xs):
        parts.append(fig.spring(prev, x, y, fg, coils=3, amplitude=5.0))
        boundary = j in (2, 5)
        color = colors.force if boundary else (a if j < 2 else b)
        parts.append(fig.rect(x, y - HALF, BOX, 2 * HALF, color, color, 2.0 if boundary else 1.5, opacity=0.25))
        parts.append(fig.math_label(x + BOX / 2, y + 5, f"m<sub>{j + 1}</sub>", fg))
        parts.append(fig.math_label(x + BOX / 2, y + 30, "x<sub>b</sub>" if boundary else "x<sub>i</sub>", fg))
        prev = x + BOX
    parts += [
        fig.arrow(xs[5] + BOX + 2, y, xs[5] + BOX + 24, y, colors.force, 1.5),
        fig.math_label(xs[5] + BOX + 14, y - 8, "f", fg),
        fig.text(155, y + 58, "interface m3 and the loaded tip: boundary", muted),
    ]
    return fig.svg(310, 120, "".join(parts))


def rails_svg() -> str:
    """The two-mass chain's displacements as points (x₁, x₂) in a plane. Its first mode is a direction in it,
    [0.618, 1]; a one-shape basis is a line through the origin, a rail the model can only move along. The closer
    the rail to the mode, the closer the frequency (each rail's error, as the notes' table)."""
    fg, muted = colors.foreground, colors.muted
    ox, oy, r = 46.0, 168.0, 150.0
    parts = [
        fig.arrow(ox - 8, oy, ox + r + 14, oy, muted, 1.0),
        fig.arrow(ox, oy + 8, ox, oy - r - 10, muted, 1.0),
        fig.math_label(ox + r + 18, oy + 5, "x<sub>1</sub>", fg, "start"),
        fig.math_label(ox - 6, oy - r - 4, "x<sub>2</sub>", fg, "end"),
    ]

    def ray(dx: float, dy: float, length: float) -> tuple[float, float]:
        n = math.hypot(dx, dy)
        return ox + length * dx / n, oy - length * dy / n

    mode = ray(0.618, 1.0, r)
    parts += [fig.line(ox, oy, *mode, colors.mass[0], 3.0), fig.text(mode[0] + 4, mode[1] - 2, "mode 1", fg, "start")]
    for (dx, dy), label in (
        ((1.0, 1.0), "+14 %"),
        ((0.5, 1.0), "+2.3 %"),
    ):
        end = ray(dx, dy, r - 10)
        parts.append(fig.line(ox, oy, *end, fg, 1.2, dash="5,4"))
        anchor = "start" if dx > 0.7 else "end"
        tx = end[0] + (6 if anchor == "start" else -6)
        parts.append(fig.text(tx, end[1] + (14 if dx > 0.7 else 4), label, fg, anchor))
    return fig.svg(310, 186, "".join(parts))


FIGURES = {"fig:cut": cut_svg, "fig:rails": rails_svg}
