"""The Substructuring page's Theory figure (see `figures`): the chain cut into substructures, with their interior
and boundary DOFs."""

from __future__ import annotations

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


FIGURES = {"fig:cut": cut_svg}
