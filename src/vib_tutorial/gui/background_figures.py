"""The Simulation page's Theory figures (see `figures`): the chain, and an absorber on a primary mass."""

from __future__ import annotations

from . import figures as fig
from .style import colors


def _element(x1: float, x2: float, y: float, upper: float = 13.0) -> str:
    """A spring above a damper, side by side between x1 and x2, centred on y."""
    fg = colors.foreground
    return "".join(
        [
            fig.line(x1, y - upper, x1, y + upper, fg),
            fig.line(x2, y - upper, x2, y + upper, fg),
            fig.spring(x1, x2, y - upper, fg, coils=3, amplitude=5.0),
            fig.damper(x1, x2, y + upper, fg, half=6.0),
        ]
    )


def _block(x: float, y: float, w: float, h: float, label: str, index: int = 0) -> str:
    mass = colors.mass[index % len(colors.mass)]
    return fig.rect(x, y - h / 2, w, h, mass, mass, 2.0, opacity=0.2) + fig.math_label(
        x + w / 2, y + 6, label, colors.foreground
    )


def chain_svg() -> str:
    """Four masses in a line, each joined to the next (the first to the wall) by a spring and a damper; each
    moves x_i; a force f on the last; the wall may move x_g."""
    fg, muted, force = colors.foreground, colors.muted, colors.force
    y, w, h, gap = 75.0, 34.0, 44.0, 38.0
    wall = 22.0
    parts = [fig.ground(wall, y - 32, y + 32, muted)]
    x = wall
    for i in range(4):
        left = x + gap
        parts += [
            _element(x, left, y),
            fig.math_label((x + left) / 2, y - 26, f"k<sub>{i + 1}</sub>", fg),
            fig.math_label((x + left) / 2, y + 37, f"c<sub>{i + 1}</sub>", fg),
            _block(left, y, w, h, f"m<sub>{i + 1}</sub>", i),
            fig.line(left + w / 2, y - h / 2 - 2, left + w / 2, y - h / 2 - 22, colors.faint, 1.0, "3 3"),
            fig.arrow(left + w / 2, y - h / 2 - 16, left + w / 2 + 24, y - h / 2 - 16, fg, 1.2),
            fig.math_label(left + w / 2 + 3, y - h / 2 - 22, f"x<sub>{i + 1}</sub>", fg, "start"),
        ]
        x = left + w
    parts += [
        fig.arrow(x + 4, y, x + 30, y, force, 2.0),
        fig.math_label(x + 12, y - 9, "f<sub>4</sub>", fg),
        fig.arrow(wall, y + 50, wall + 24, y + 50, fg, 1.2),
        fig.math_label(wall + 30, y + 55, "x<sub>g</sub>", fg, "start"),
    ]
    return fig.svg(360, 140, "".join(parts))


def absorber_svg() -> str:
    """A primary mass m1 on k1 from the wall, driven by F; an absorber m_a on k_a and c_a hung on it."""
    fg, muted, force = colors.foreground, colors.muted, colors.force
    y, wall = 70.0, 30.0
    p1, w1, pa, wa = 110.0, 60.0, 250.0, 40.0
    parts = [
        fig.ground(wall, y - 35, y + 35, muted),
        fig.spring(wall, p1, y, fg, coils=4, amplitude=6.0),
        fig.math_label((wall + p1) / 2, y - 14, "k<sub>1</sub>", fg),
        _block(p1, y, w1, 56, "m<sub>1</sub>", 0),
        _element(p1 + w1, pa, y, 15),
        fig.math_label((p1 + w1 + pa) / 2, y - 28, "k<sub>a</sub>", fg),
        fig.math_label((p1 + w1 + pa) / 2, y + 42, "c<sub>a</sub>", fg),
        _block(pa, y, wa, 40, "m<sub>a</sub>", 1),
        fig.arrow(p1 + 12, y - 40, p1 + 48, y - 40, force, 2.0),
        fig.math_label(p1 + 54, y - 35, "F", fg, "start"),
        fig.text(p1 + w1 / 2, y + 50, "machine", muted),
        fig.text(pa + wa / 2, y + 50, "absorber", muted),
    ]
    return fig.svg(360, 135, "".join(parts))


FIGURES = {"fig:chain": chain_svg, "fig:absorber": absorber_svg}
