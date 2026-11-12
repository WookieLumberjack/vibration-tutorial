"""The Energy methods page's Theory figures (see `figures`): Holzer's step past one mass, and the flexibility
Dunkerley's sum is built from."""

from __future__ import annotations

from . import figures as fig
from .style import colors


def holzer_svg() -> str:
    """Mass i cut free from its springs at a trial ω: spring i pulls it back with N_i, spring i+1 on with N_i+1;
    what is left of N_i after the inertia force m_i ω² x_i goes on to spring i+1."""
    fg, muted, force = colors.foreground, colors.muted, colors.force
    mass = colors.mass[0]
    y, x0, w, h = 62.0, 150.0, 60.0, 44.0
    parts = [
        fig.spring(14, 84, y, muted, coils=3, amplitude=5.0),
        fig.math_label(49, y - 14, "k<sub>i</sub>", fg),
        fig.spring(276, 346, y, muted, coils=3, amplitude=5.0),
        fig.math_label(311, y - 14, "k<sub>i+1</sub>", fg),
        fig.rect(x0, y - h / 2, w, h, mass, mass, 2.0, opacity=0.2),
        fig.math_label(x0 + w / 2, y + 6, "m<sub>i</sub>", fg),
        fig.arrow(x0 - 4, y, x0 - 56, y, force, 2.0),
        fig.math_label(x0 - 30, y - 9, "N<sub>i</sub>", fg),
        fig.arrow(x0 + w + 4, y, x0 + w + 56, y, force, 2.0),
        fig.math_label(x0 + w + 30, y - 9, "N<sub>i+1</sub>", fg),
        fig.line(x0 + w / 2, y - h / 2 - 2, x0 + w / 2, y - h / 2 - 22, colors.faint, 1.0, "3 3"),
        fig.arrow(x0 + w / 2, y - h / 2 - 16, x0 + w / 2 + 24, y - h / 2 - 16, fg, 1.2),
        fig.math_label(x0 + w / 2 + 28, y - h / 2 - 11, "x<sub>i</sub>", fg, "start"),
        fig.math_label(180, y + 50, "N<sub>i+1</sub> = N<sub>i</sub> − m<sub>i</sub>ω²x<sub>i</sub>", fg),
    ]
    return fig.svg(360, 125, "".join(parts))


def flexibility_svg() -> str:
    """The chain without its masses, a 1 N force on point 2: springs 1 and 2 carry it and stretch; the rest
    ride along unstretched. Point 2 moves (K⁻¹)₂₂ = 1/k₁ + 1/k₂."""
    fg, muted, force = colors.foreground, colors.muted, colors.force
    y, wall, gap = 56.0, 20.0, 64.0
    parts = [fig.ground(wall, y - 24, y + 24, muted)]
    x = wall
    for i in range(4):
        nxt = x + gap
        carries = i < 2
        parts += [
            fig.spring(x, nxt - 6, y, fg if carries else muted, coils=4, amplitude=6.0),
            fig.math_label((x + nxt) / 2 - 3, y - 15, f"k<sub>{i + 1}</sub>", fg),
            fig.circle(nxt, y, 6.0, colors.mass[0] if i == 1 else muted, fill=colors.mass[0] if i == 1 else "none"),
        ]
        x = nxt + 6
    p2 = wall + 2 * gap + 6
    parts += [
        fig.line(p2, y + 8, p2, y + 26, force, 2.0),
        fig.arrow(p2, y + 26, p2 + 40, y + 26, force, 2.0),
        fig.text(p2 + 46, y + 30, "1 N", fg, "start"),
        fig.math_label(p2, y + 52, "(K<sup>−1</sup>)<sub>22</sub> = 1/k<sub>1</sub> + 1/k<sub>2</sub>", fg),
    ]
    return fig.svg(360, 125, "".join(parts))


FIGURES = {"fig:holzer": holzer_svg, "fig:flexibility": flexibility_svg}
