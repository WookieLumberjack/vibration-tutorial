"""The Periodic forcing Theory's figures (see `figures`): a square wave built from its harmonics, and how a
coefficient is found by averaging the wave times a sine."""

from __future__ import annotations

import math

import numpy as np

from . import figures as fig
from .style import colors

SUB = "<tspan baseline-shift='sub' font-size='10'>{}</tspan>"


def _axes(x0: float, x1: float, y0: float, color: str) -> str:
    """A time axis from x0 to x1 at height y0, with an arrowhead and t."""
    return fig.arrow(x0, y0, x1, y0, color, 1.0) + fig.math_label(x1 - 2, y0 - 6, "t", color, "end")


def harmonics_svg() -> str:
    """A ±F square wave's first three harmonics, (4F/π) sin jω₀t / j for j = 1, 3, 5, and their sum over it."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    left, right = 46.0, 360.0
    span = right - 18 - left  # one period
    t = np.linspace(0.0, 1.0, 241)
    xs = left + span * t
    parts = []
    # top: the three harmonics, each in its own colour
    top_y, top_h = 62.0, 34.0
    parts.append(_axes(left, right, top_y, muted))
    for i, j in enumerate((1, 3, 5)):
        wave = top_y - top_h * np.sin(2 * math.pi * j * t) / j
        parts.append(fig.curve(list(zip(xs, wave)), colors.mass[i], 1.8))
    for i, (j, x, y) in enumerate(((1, 250, 18), (3, 298, 18), (5, 346, 18))):
        parts.append(fig.math_label(x, y, f"j = {j}", colors.mass[i]))
    # bottom: the square wave (grey) and the sum of the three (the force's colour, as on the page)
    bot_y, bot_h = 168.0, 42.0
    parts.append(_axes(left, right, bot_y, muted))
    parts.append(
        fig.curve(
            [(left, bot_y - bot_h), (left + span / 2, bot_y - bot_h), (left + span / 2, bot_y + bot_h)]
            + [(left + span, bot_y + bot_h)],
            colors.grey,
            3.0,
        )
    )
    total = np.sum([4 / math.pi * np.sin(2 * math.pi * j * t) / j for j in (1, 3, 5)], axis=0)
    parts.append(fig.curve(list(zip(xs, bot_y - bot_h * total)), colors.force, 1.8))
    parts += [
        fig.line(left, bot_y - bot_h, left - 5, bot_y - bot_h, faint, 1.0),
        fig.math_label(left - 8, bot_y - bot_h + 5, "F", fg, "end"),
        fig.math_label(left - 8, bot_y + bot_h + 5, "−F", fg, "end"),
        fig.text(left + span * 0.75, bot_y - 22, "their sum", colors.force),
        fig.math_label(left + span, bot_y + bot_h + 18, "T", fg, "middle"),
        fig.line(left + span, bot_y + 3, left + span, bot_y - 3, muted, 1.0),
        fig.math_label(left + span, top_y + 18, "T", fg, "middle"),
        fig.line(left + span, top_y + 3, left + span, top_y - 3, muted, 1.0),
    ]
    return fig.svg(380, 232, "".join(parts))


def _projection(x0: float, j: int, title: str) -> str:
    """One panel: the square wave times sin jω₀t over a period, the area under it shaded (its mean)."""
    fg, muted = colors.foreground, colors.muted
    width, mid, h = 150.0, 92.0, 38.0
    t = np.linspace(0.0, 1.0, 181)
    xs = x0 + width * t
    sine = np.sin(2 * math.pi * j * t)
    product = np.where(t < 0.5, 1.0, -1.0) * sine
    under = [(x0, mid), *zip(xs, mid - h * product), (x0 + width, mid)]
    return "".join(
        [
            fig.area(under, colors.force, 0.25),
            _axes(x0 - 4, x0 + width + 14, mid, muted),
            fig.curve(list(zip(xs, mid - h * sine)), colors.mass[0], 1.2, "4 3"),
            fig.curve(list(zip(xs, mid - h * product)), colors.force, 1.8),
            fig.text(x0 + width / 2, 26, title, fg),
        ]
    )


def coefficient_svg() -> str:
    """The square wave times sin ω₀t (all above the axis: b₁ ≠ 0) and times sin 2ω₀t (the areas cancel: b₂ = 0)."""
    muted = colors.muted
    parts = [
        _projection(22.0, 1, "f × sin ω₀t"),
        _projection(212.0, 2, "f × sin 2ω₀t"),
        fig.text(97.0, 160, "mean 2F/π: b" + SUB.format("1") + " = 4F/π", muted),
        fig.text(287.0, 160, "mean 0: b" + SUB.format("2") + " = 0", muted),
    ]
    return fig.svg(380, 175, "".join(parts))


FIGURES = {"fig:harmonics": harmonics_svg, "fig:coefficient": coefficient_svg}
