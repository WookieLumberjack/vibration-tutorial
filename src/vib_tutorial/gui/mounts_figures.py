"""The Rigid body on mounts page's Theory figures (see `figures`): a focalized pair of mounts in the roll plane."""

from __future__ import annotations

import math

from . import figures as fig
from .style import colors

TILT = math.radians(40.0)  # drawn tilt β
LENGTH = 40.0  # px, a mount's drawn length


def focus_svg() -> str:
    """The roll plane: a box's section on two mounts t either side of the middle, each leaning β toward it. Their
    lines of action meet above them, at the roll plane's elastic centre, here tilted so it is the centre of mass,
    d above the mounts."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    box, mount = colors.mass[0], colors.force
    cx, base, t = 140.0, 150.0, 55.0
    top = base - LENGTH * math.cos(TILT)  # the mounts' upper ends, on the box's underside
    meet = top - t / math.tan(TILT)  # the lines of action cross here, over the middle
    parts = [
        fig.floor(cx - 115, cx + 115, base, muted),
        fig.rect(cx - 100, meet - 24, 200, top - meet + 24, box, box, 2.0, opacity=0.15),
    ]
    for side in (-1, 1):
        foot = cx + side * (t + LENGTH * math.sin(TILT))
        parts += [
            fig.line(foot, base, cx + side * t, top, mount, 3.5),
            fig.line(cx + side * t, top, cx, meet, faint, 1.0, "4 3"),
        ]
    left_foot = cx - t - LENGTH * math.sin(TILT)
    parts += [
        fig.line(left_foot, base, left_foot, base - 34, faint, 1.0, "2 3"),
        fig.math_label(left_foot + 9, base - 34, "β", fg, "start"),
        fig.circle(cx, meet, 6.0, fg),
        fig.circle(cx, meet, 2.5, fg, fill=fg),
        fig.text(cx + 12, meet + 4, "elastic centre", fg, "start"),
        fig.text(cx + 12, meet + 20, "= centre of mass", fg, "start"),
        fig.line(cx + 108, meet, cx + 108, top, muted, 1.0),
        fig.math_label(cx + 113, (meet + top) / 2 + 5, "d = z<sub>e</sub>", fg, "start"),
        fig.line(cx - t, top + 4, cx - t, base - 2, faint, 1.0, "2 3"),
        fig.line(cx + t, top + 4, cx + t, base - 2, faint, 1.0, "2 3"),
        fig.line(cx - t, top + 10, cx + t, top + 10, muted, 1.0),
        fig.math_label(cx, top + 26, "2t", fg),
    ]
    return fig.svg(310, 170, "".join(parts))


FIGURES = {"fig:focus": focus_svg}
