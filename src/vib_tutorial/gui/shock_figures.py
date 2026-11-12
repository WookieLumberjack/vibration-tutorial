"""The Shock spectrum page's Theory figures (see `figures`): a half-sine compensated for a shaker."""

from __future__ import annotations

import math

from . import figures as fig
from .style import colors

PRE = 1 / 3  # each compensating pulse's peak, over the main one's
LEFT, RIGHT, MID = 20.0, 300.0, 74.0  # x span; the acceleration's zero line
VEL = 150.0  # the velocity's zero line


def _acceleration(s: float) -> float:
    """The compensated pulse on s in [0, 1]: a long negative half-sine before and after the main one, each with
    half its area, so the velocity comes back to zero."""
    main = (0.4, 0.6)  # the main pulse's span
    pre_len = 0.5 * (main[1] - main[0]) / PRE  # half the main pulse's area at PRE of its peak
    if main[0] <= s <= main[1]:
        return math.sin(math.pi * (s - main[0]) / (main[1] - main[0]))
    if main[0] - pre_len <= s < main[0]:
        return -PRE * math.sin(math.pi * (s - main[0] + pre_len) / pre_len)
    if main[1] < s <= main[1] + pre_len:
        return -PRE * math.sin(math.pi * (s - main[1]) / pre_len)
    return 0.0


def compensated_svg() -> str:
    """Above: the half-sine with a pre- and a post-pulse, each of the opposite sign and half its area. Below: the
    table's velocity, which rises with the main pulse but starts and ends at zero."""
    fg, faint = colors.foreground, colors.faint
    acc, vel = colors.force, colors.mass[0]
    n = 400
    s = [i / n for i in range(n + 1)]
    a = [_acceleration(x) for x in s]
    v = [0.0]
    for i in range(1, n + 1):
        v.append(v[-1] + 0.5 * (a[i] + a[i - 1]) / n)
    top = max(abs(x) for x in v)

    def x(t: float) -> float:
        return LEFT + (RIGHT - LEFT) * t

    parts = [
        fig.line(LEFT, MID, RIGHT, MID, faint, 1.0),
        fig.line(LEFT, VEL, RIGHT, VEL, faint, 1.0),
        fig.curve([(x(t), MID - 50 * y) for t, y in zip(s, a, strict=True)], acc, 2.0),
        fig.curve([(x(t), VEL - 18 * y / top) for t, y in zip(s, v, strict=True)], vel, 2.0),
        fig.text(LEFT, 14, "acceleration: pre-pulse, pulse, post-pulse", fg, "start"),
        fig.text(LEFT, 100, "velocity: back to zero at the end", fg, "start"),
    ]
    return fig.svg(310, 170, "".join(parts))


FIGURES = {"fig:compensated": compensated_svg}
