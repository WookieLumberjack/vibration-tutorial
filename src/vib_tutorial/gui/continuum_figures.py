"""The Continuous systems page's Theory figures (see `figures`): a string cut into beads, and a pulse at a step."""

from __future__ import annotations

import math

from . import figures as fig
from .style import colors

LEFT, RIGHT = 25.0, 315.0


def _ends(y: float) -> str:
    muted = colors.muted
    return fig.ground(LEFT, y - 22, y + 22, muted) + fig.ground(RIGHT, y - 22, y + 22, muted, side=1)


def beads_svg() -> str:
    """Above: a string fixed at both ends, in its first mode, μ per length under T. Below: the same cut into
    N = 6 elements of length h, a bead of μh at each inner node, straight pieces between."""
    fg, muted, mass = colors.foreground, colors.muted, colors.mass[0]
    top, low, amp = 40.0, 112.0, 22.0
    span = RIGHT - LEFT
    curve = [(LEFT + span * i / 60, top - amp * math.sin(math.pi * i / 60)) for i in range(61)]
    parts = [
        _ends(top),
        fig.curve(curve, fg, 2.0),
        fig.math_label(LEFT + span / 2, top + 20, "μ, T", fg),
        _ends(low),
    ]
    nodes = [(LEFT + span * j / 6, low - amp * math.sin(math.pi * j / 6)) for j in range(7)]
    parts.append(fig.curve(nodes, fg, 1.5))
    for x, y in nodes[1:-1]:
        parts.append(fig.circle(x, y, 7.0, mass, fill=mass))
    x1, x2 = nodes[2][0], nodes[3][0]
    parts += [
        fig.line(x1, low + 14, x1, low + 26, muted, 1.0),
        fig.line(x2, low + 14, x2, low + 26, muted, 1.0),
        fig.arrow(x1 + (x2 - x1) / 2, low + 20, x2 - 1, low + 20, muted, 1.0),
        fig.arrow(x1 + (x2 - x1) / 2, low + 20, x1 + 1, low + 20, muted, 1.0),
        fig.math_label((x1 + x2) / 2, low + 36, "h", fg),
        fig.math_label(nodes[4][0] + 10, nodes[4][1] - 12, "μh", fg, "start"),
    ]
    return fig.svg(340, 150, "".join(parts))


def _pulse(centre: float, y: float, width: float, height: float, color: str) -> str:
    points = [
        (centre + width * (i / 40 - 0.5), y - height * 0.5 * (1 + math.cos(2 * math.pi * (i / 40 - 0.5))))
        for i in range(41)
    ]
    return fig.curve(points, color, 2.0)


def step_svg() -> str:
    """A light string (Z1) tied to one with twice the impedance (Z2 = 2Z1). Above: a pulse heading for the joint.
    Below: R = −1/3 of it back, upside down; τ = 2/3 of it on, half as fast and so half as long."""
    fg, muted, force = colors.foreground, colors.muted, colors.force
    joint = 170.0
    top, low = 46.0, 104.0
    parts = []
    for y in (top, low):
        parts += [
            fig.line(LEFT, y, joint, y, fg, 1.2),
            fig.line(joint, y, RIGHT, y, fg, 3.0),
            fig.line(joint, y - 30, joint, y + 12, colors.faint, 1.0, "3 3"),
        ]
    parts += [
        fig.math_label((LEFT + joint) / 2, top + 22, "Z<sub>1</sub>", fg),
        fig.math_label((joint + RIGHT) / 2 + 2, top + 22, "Z<sub>2</sub>", fg, "end"),
        fig.math_label((joint + RIGHT) / 2 + 3, top + 22, "= 2Z<sub>1</sub>", fg, "start"),
        _pulse(90.0, top, 60.0, 30.0, force),
        fig.arrow(125.0, top - 22, 147.0, top - 22, force, 1.5),
        # after: reflected upside down, a third as high; transmitted two thirds, half as long
        fig.curve(
            [
                (90.0 + 60.0 * (i / 40 - 0.5), low + 10.0 * 0.5 * (1 + math.cos(2 * math.pi * (i / 40 - 0.5))))
                for i in range(41)
            ],
            force,
            2.0,
        ),
        fig.arrow(61.0, low + 18, 39.0, low + 18, force, 1.5),
        fig.math_label(90.0, low + 34, "R = −1/3", fg),
        _pulse(240.0, low, 30.0, 20.0, force),
        fig.arrow(260.0, low - 18, 280.0, low - 18, force, 1.5),
        fig.math_label(240.0, low + 34, "τ = 2/3", fg),
        fig.text(6.0, top - 30, "before", muted, "start"),
        fig.text(6.0, low - 30, "after", muted, "start"),
    ]
    return fig.svg(340, 150, "".join(parts))


FIGURES = {"fig:beads": beads_svg, "fig:step": step_svg}
