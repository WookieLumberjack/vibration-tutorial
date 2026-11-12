"""The Fatigue page's Theory figures (see `figures`): Haigh's diagram with Goodman's line and Gerber's parabola."""

from __future__ import annotations

from . import figures as fig
from .style import colors

LEFT, RIGHT, TOP, BOTTOM = 40.0, 290.0, 22.0, 140.0
S_AR, S_U = 150.0, 600.0  # the worked example's: 150 MPa fully reversed, 600 MPa ultimate
EXAMPLE = (200.0, 100.0)  # (mean, amplitude) on Goodman's line


def _xy(mean: float, amplitude: float) -> tuple[float, float]:
    return LEFT + (RIGHT - LEFT) * mean / S_U, BOTTOM - (BOTTOM - TOP) * amplitude / (1.1 * S_AR)


def haigh_svg() -> str:
    """Amplitude against mean: the cycles as damaging as a fully reversed S_ar lie on Goodman's straight line to
    S_u, or Gerber's parabola, which gives more; the worked example's cycle, 100 MPa about 200, on Goodman's."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    goodman, gerber = colors.mass[0], colors.force
    parts = [
        fig.line(LEFT, BOTTOM, RIGHT + 10, BOTTOM, faint, 1.0),
        fig.line(LEFT, TOP - 6, LEFT, BOTTOM, faint, 1.0),
        fig.curve([_xy(0, S_AR), _xy(S_U, 0)], goodman, 2.0),
        fig.curve([_xy(m, S_AR * (1 - (m / S_U) ** 2)) for m in [S_U * i / 40 for i in range(41)]], gerber, 2.0),
    ]
    x, y = _xy(*EXAMPLE)
    y0 = _xy(0, S_AR)[1]
    xu = _xy(S_U, 0)[0]
    parts += [
        fig.line(x, y, x, BOTTOM, faint, 1.0, "3 3"),
        fig.line(LEFT, y, x, y, faint, 1.0, "3 3"),
        fig.circle(x, y, 4.0, fg, fill=fg),
        fig.math_label(x, BOTTOM + 16, "S<sub>m</sub>", fg),
        fig.math_label(LEFT - 6, y + 5, "S<sub>a</sub>", fg, "end"),
        fig.math_label(LEFT - 6, y0 + 5, "S<sub>ar</sub>", fg, "end"),
        fig.math_label(xu, BOTTOM + 16, "S<sub>u</sub>", fg),
        fig.text(x + 40, y - 30, "Gerber", gerber, "start"),
        fig.text(x + 22, y + 22, "Goodman", goodman, "start"),
        fig.text((x + xu) / 2 + 20, BOTTOM + 16, "mean", muted),
        fig.text(LEFT + 4, TOP - 10, "amplitude", muted, "start"),
    ]
    return fig.svg(310, 170, "".join(parts))


FIGURES = {"fig:haigh": haigh_svg}
