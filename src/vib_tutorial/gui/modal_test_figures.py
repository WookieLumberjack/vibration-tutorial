"""The Virtual modal test page's Theory figures (see `figures`): the test set up on the chain (a shaker through a
stinger and force gauge at m4, an accelerometer on every mass), and the shaker's armature between its coil's force
and the structure, which is why the force drops at a resonance."""

from __future__ import annotations

from . import figures as fig
from .style import colors


def setup_svg() -> str:
    """The uniform 4-mass chain from the wall; an accelerometer on each mass, a force gauge on m4 with the stinger
    and shaker to its right. One column of H: the force at m4, every response measured."""
    fg, muted, mass, force = colors.foreground, colors.muted, colors.mass[0], colors.force
    y, wall, gap, w, h = 92.0, 16.0, 22.0, 28.0, 28.0
    parts = [fig.ground(wall, y - 22, y + 22, muted)]
    x = wall
    for i in range(4):
        left = x + gap
        parts += [
            fig.spring(x, left, y, fg, coils=3, amplitude=5.0),
            fig.rect(left, y - h / 2, w, h, mass, mass, 2.0, opacity=0.2),
            fig.math_label(left + w / 2, y + 6, f"m<sub>{i + 1}</sub>", fg),
            # the accelerometer on top, its cable going up
            fig.rect(left + w / 2 - 5, y - h / 2 - 10, 10, 10, fg, fg, 1.0),
            fig.line(left + w / 2, y - h / 2 - 10, left + w / 2, 38, muted, 1.0),
        ]
        x = left + w
    gauge, stinger, body = x, x + 8, x + 30
    parts += [
        fig.text(wall + 2 * gap + 1.5 * w + 13, 30, "an accelerometer on every mass", muted),
        # the force gauge, the stinger and the shaker
        fig.rect(gauge, y - 8, 8, 16, force, force, 1.2, opacity=0.6),
        fig.line(stinger, y, body, y, fg, 1.2),
        fig.rect(body, y - 20, 32, 40, fg, colors.background, 1.5),
        fig.rect(body - 4, y - 7, 8, 14, fg, fg, 1.0, opacity=0.5),
        fig.text(body + 16, y + 36, "shaker", fg),
        fig.text(body + 16, y - 26, "stinger", muted, "end"),
        fig.text(gauge + 4, y + 36, "force gauge", muted, "end"),
        fig.line(gauge + 4, y + 9, gauge + 4, y + 24, muted, 1.0),
        fig.text(
            wall + 2 * gap + 1.5 * w + 13, 150, "the force at m<sub>4</sub>: one column of H, H<sub>j4</sub>", muted
        ),
    ]
    return fig.svg(302, 164, "".join(parts))


def shaker_svg() -> str:
    """The shaker's coil pushes its armature (mass m_sh, on its suspension spring) with F₀; the armature pushes the
    structure through the stinger with F, which the gauge measures. At a resonance the structure moves freely,
    so most of F₀ goes into moving the armature: F drops."""
    fg, muted, mass, force = colors.foreground, colors.muted, colors.mass[1], colors.force
    y = 78.0
    parts = [
        fig.ground(20, y - 24, y + 24, muted),
        fig.spring(20, 74, y, fg, coils=3, amplitude=6.0),
        fig.rect(74, y - 18, 44, 36, mass, mass, 2.0, opacity=0.2),
        fig.math_label(96, y + 6, "m<sub>sh</sub>", fg),
        fig.text(96, y + 36, "armature", muted),
        # the coil's force on the armature
        fig.arrow(80, y - 30, 112, y - 30, force, 2.0),
        fig.math_label(78, y - 25, "F<sub>0</sub>", fg, "end"),
        # the stinger and the gauge
        fig.line(118, y, 176, y, fg, 1.5),
        fig.text(147, y - 8, "stinger", muted),
        fig.rect(176, y - 9, 8, 18, force, force, 1.2, opacity=0.6),
        fig.arrow(160, y + 18, 182, y + 18, force, 1.6),
        fig.math_label(156, y + 23, "F", fg, "end"),
        # the structure
        fig.rect(184, y - 30, 100, 60, fg, colors.background, 1.5),
        fig.text(234, y - 8, "structure", fg),
        fig.math_label(234, y + 18, "A<sub>kk</sub>", fg),
    ]
    return fig.svg(302, 130, "".join(parts))


FIGURES = {"fig:setup": setup_svg, "fig:shaker": shaker_svg}
