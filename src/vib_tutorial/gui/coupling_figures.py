"""The Modal coupling page's Theory figure (see `figures`): the chain split into A and B, and the 2-DOF model one
mode of each becomes."""

from __future__ import annotations

from . import figures as fig
from .style import colors

BOX = 30.0  # a mass's width
HALF = 11.0  # its half height


def _mass(x: float, y: float, label: str, color: str, fg: str) -> str:
    return fig.rect(x, y - HALF, BOX, 2 * HALF, color, color, 1.5, opacity=0.2) + fig.math_label(
        x + BOX / 2, y + 5, label, fg
    )


def split_svg() -> str:
    """Top: A (m₁, m₂ on the ground, its tip free) and B (m₃, m₄, its first spring k₃ tied to the ground: its
    base). Bottom: one mode of each as an oscillator, B's on A's."""
    fg, muted, a, b = colors.foreground, colors.muted, colors.mass[0], colors.mass[1]
    parts: list[str] = []
    y = 34.0
    # A: ground, k1, m1, k2, m2
    parts += [
        fig.ground(14, y - 16, y + 16, muted),
        fig.spring(14, 34, y, fg, coils=3, amplitude=5.0),
        _mass(34, y, "m<sub>1</sub>", a, fg),
        fig.spring(64, 84, y, fg, coils=3, amplitude=5.0),
        _mass(84, y, "m<sub>2</sub>", a, fg),
        fig.text(64, y + 30, "A: tip free", fg),
    ]
    # B: ground (its base), k3, m3, k4, m4
    x0 = 166.0
    parts += [
        fig.ground(x0, y - 16, y + 16, muted),
        fig.spring(x0, x0 + 24, y, fg, coils=3, amplitude=5.0),
        _mass(x0 + 24, y, "m<sub>3</sub>", b, fg),
        fig.spring(x0 + 54, x0 + 74, y, fg, coils=3, amplitude=5.0),
        _mass(x0 + 74, y, "m<sub>4</sub>", b, fg),
        fig.text(x0 + 60, y + 30, "B: base held", fg),
    ]
    # the 2-DOF model
    y2 = 118.0
    parts += [
        fig.line(14, 82, 296, 82, muted, 1.0, dash="3,3"),
        fig.ground(40, y2 - 16, y2 + 16, muted),
        fig.spring(40, 100, y2, fg, coils=4, amplitude=5.0),
        fig.math_label(70, y2 - 12, "k<sub>a</sub>", fg),
        _mass(100, y2, "m<sub>a</sub>", a, fg),
        fig.spring(130, 190, y2, fg, coils=4, amplitude=5.0),
        fig.math_label(160, y2 - 12, "k<sub>b</sub>", fg),
        _mass(190, y2, "m<sub>b</sub>", b, fg),
        fig.text(155, y2 + 32, "one mode of each, B's on A's", fg),
    ]
    return fig.svg(310, 160, "".join(parts))


FIGURES = {"fig:split": split_svg}
