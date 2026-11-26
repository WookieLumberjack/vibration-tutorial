"""State-space modes' Theory figures (see `figures`): a mode's pair spiraling in on its plane, and a complex shape's
ellipse on the (x₁, x₂) plane."""

from __future__ import annotations

import math

from . import figures as fig
from .style import colors


def _spiral(cx: float, cy: float, r0: float, decay: float, turns: float) -> list[tuple[float, float]]:
    """A clockwise spiral (as a phase plane turns), from (cx + r0, cy), shrinking by e^(−decay) a turn (SVG's y
    down, so clockwise on the page is a positive angle here)."""
    n = int(80 * turns)
    out = []
    for i in range(n + 1):
        a = 2 * math.pi * turns * i / n
        r = r0 * math.exp(-decay * a / (2 * math.pi))
        out.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return out


def pairs_svg() -> str:
    """Two planes side by side, each a state-space mode's pair (w₁, w₂) and (w₃, w₄), each point spiraling in on
    its own: the four numbers of the state, two to a plane."""
    fg, muted = colors.foreground, colors.muted
    parts = []
    for k, (cx, label, a, b, decay, turns) in enumerate(
        (
            (95.0, "mode 1", "w<sub>1</sub>", "w<sub>2</sub>", 0.18, 3.0),
            (285.0, "mode 2", "w<sub>3</sub>", "w<sub>4</sub>", 0.35, 2.5),
        )
    ):
        cy, half = 105.0, 72.0
        color = colors.mass[k]
        path = _spiral(cx, cy, 62.0, decay, turns)
        parts += [
            fig.arrow(cx - half, cy, cx + half, cy, muted, 1.0),
            fig.math_label(cx + half - 4, cy + 18, a, muted, "end"),
            fig.arrow(cx, cy + half, cx, cy - half, muted, 1.0),
            fig.math_label(cx - 8, cy - half + 10, b, muted, "end"),
            fig.curve(path, color, 1.8),
            fig.circle(*path[-1], 4.0, fill=color),
            fig.circle(*path[0], 3.0, fill=colors.background),
            fig.text(cx, cy + half + 24, label, fg),
        ]
    parts.append(fig.text(190, 222, "each turns at its ω and shrinks as e^(σt), on its own", muted))
    return fig.svg(380, 232, "".join(parts))


def complex_shape_svg() -> str:
    """The (x₁, x₂) plane: a complex mode's Re ψ and Im ψ as arrows, and the ellipse the mode alone traces,
    Re ψ cos ωt − Im ψ sin ωt, beside the line a real shape would swing along."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    one = colors.mass[0]
    ox, oy, scale = 190.0, 120.0, 95.0
    re, im = (0.55, 0.78), (0.16, -0.12)  # a lag of about 20° between the masses
    pts = []
    for i in range(121):
        t = 2 * math.pi * i / 120
        x = re[0] * math.cos(t) - im[0] * math.sin(t)
        y = re[1] * math.cos(t) - im[1] * math.sin(t)
        pts.append((ox + scale * x, oy - scale * y))
    parts = [
        fig.arrow(40, oy, 345, oy, muted, 1.0),
        fig.math_label(340, oy + 20, "x<sub>1</sub>", muted, "end"),
        fig.arrow(ox, 225, ox, 15, muted, 1.0),
        fig.math_label(ox - 10, 25, "x<sub>2</sub>", muted, "end"),
        fig.line(ox - scale * re[0], oy + scale * re[1], ox + scale * re[0], oy - scale * re[1], faint, 1.5, "5 4"),
        fig.curve(pts, one, 1.8),
        fig.arrow(ox, oy, ox + scale * re[0], oy - scale * re[1], fg, 2.5),
        fig.math_label(ox + scale * re[0] + 6, oy - scale * re[1] + 4, "Re ψ", fg, "start"),
        fig.arrow(ox, oy, ox + scale * im[0] * 2, oy - scale * im[1] * 2, fg, 2.0),
        fig.math_label(ox + scale * im[0] * 2 + 6, oy - scale * im[1] * 2 + 14, "Im ψ (twice)", fg, "start"),
        fig.text(95, 205, "a real shape: a line", muted),
        fig.text(300, 60, "the complex mode: an ellipse", one),
    ]
    return fig.svg(380, 232, "".join(parts))


FIGURES = {"fig:pairs": pairs_svg, "fig:complex-shape": complex_shape_svg}
