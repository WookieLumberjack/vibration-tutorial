"""The Quarter-car page's Theory figures (see `figures`): the quarter-car model, and wheelbase filtering."""

from __future__ import annotations

import math

from . import figures as fig
from .style import colors


def _spring(fg: str):
    return lambda a, b, x: fig.spring(a, b, x, fg, coils=4, amplitude=6.0)


def quarter_svg() -> str:
    """The body m_s on the suspension's spring k_s and damper c_s, on the wheel m_u, on the tyre k_t, on the road
    z_r; the body rises x_s and the wheel x_u."""
    fg, muted = colors.foreground, colors.muted
    body, wheel = colors.mass[0], colors.mass[1]
    cx = 130.0
    parts = [
        fig.rect(cx - 50, 14, 100, 34, body, body, 2.0, opacity=0.2),
        fig.math_label(cx, 37, "m<sub>s</sub>", fg),
        fig.upright(cx - 22, 48, 92, lambda a, b, x: fig.spring(a, b, x, fg, coils=4, amplitude=6.0)),
        fig.upright(cx + 22, 48, 92, lambda a, b, x: fig.damper(a, b, x, fg, half=7.0)),
        fig.math_label(cx - 34, 76, "k<sub>s</sub>", fg, "end"),
        fig.math_label(cx + 34, 76, "c<sub>s</sub>", fg, "start"),
        fig.rect(cx - 30, 92, 60, 26, wheel, wheel, 2.0, opacity=0.2),
        fig.math_label(cx, 111, "m<sub>u</sub>", fg),
        fig.upright(cx, 118, 152, lambda a, b, x: fig.spring(a, b, x, fg, coils=3, amplitude=6.0)),
        fig.math_label(cx + 12, 140, "k<sub>t</sub>", fg, "start"),
        fig.curve([(cx - 70 + i * 3.5, 152 + 3 * math.sin(i * 0.5)) for i in range(41)], muted, 1.5),
        fig.math_label(cx + 78, 160, "z<sub>r</sub>", fg, "start"),
    ]
    for y, label in ((31, "x<sub>s</sub>"), (105, "x<sub>u</sub>")):
        parts += [
            fig.arrow(cx + 70, y + 8, cx + 70, y - 12, fg, 1.2),
            fig.math_label(cx + 76, y + 2, label, fg, "start"),
        ]
    parts.append(fig.arrow(cx + 70, 166, cx + 70, 148, fg, 1.2))
    return fig.svg(310, 170, "".join(parts))


def wheelbase_svg() -> str:
    """A car of wheelbase L on a road whose wavelength is 2L: the front wheel on a crest, the rear in a trough, so
    the road drives pitch and no bounce."""
    fg, muted, body = colors.foreground, colors.muted, colors.mass[0]
    x0, base, amp = 25.0, 92.0, 12.0
    rear, front = 95.0, 215.0  # the wheelbase spans half a wavelength: 120 px
    wave = 240.0

    def road(x: float) -> float:
        return base - amp * math.cos(2 * math.pi * (x - front) / wave)

    parts = [fig.curve([(x0 + i * 2.6, road(x0 + i * 2.6)) for i in range(101)], muted, 1.5)]
    yr, yf = road(rear) - 10, road(front) - 10
    parts += [
        fig.circle(rear, yr, 10.0, fg),
        fig.circle(front, yf, 10.0, fg),
        fig.line(rear, yr - 22, front, yf - 22, body, 6.0),
        fig.line(rear, yr - 10, rear, yr - 22, fg, 1.5),
        fig.line(front, yf - 10, front, yf - 22, fg, 1.5),
        fig.line(rear, 112, front, 112, muted, 1.0),
        fig.math_label((rear + front) / 2, 108, "L", fg),
        fig.text(155, 18, "a wavelength of 2L: pitch, no bounce", muted),
    ]
    return fig.svg(310, 120, "".join(parts))


FIGURES = {"fig:quarter": quarter_svg, "fig:wheelbase": wheelbase_svg}
