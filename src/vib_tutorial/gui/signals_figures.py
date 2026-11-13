"""The Signals & spectra page's Theory figures (see `figures`): a seismic transducer in its case, and the octave and
one-third-octave bands with the page's two default tones in them."""

from __future__ import annotations

import math

from . import figures as fig
from .style import colors


def seismic_svg() -> str:
    """A case fixed to the structure, moving y; inside, a mass m on a spring k and a damper c to the case's floor.
    The sensor reads z, the mass's motion relative to the case."""
    fg, muted, mass = colors.foreground, colors.muted, colors.mass[0]
    left, right, top, floor = 70.0, 210.0, 20.0, 150.0
    my, mh = 48.0, 34.0  # the mass's top and height
    parts = [
        # the structure under the case
        fig.rect(left - 10, floor, right - left + 20, 22, fg, colors.background, 1.5),
        fig.text(right + 18, floor + 16, "structure", fg, "start"),
        # the case
        fig.line(left, floor, left, top, fg, 2.0),
        fig.line(right, floor, right, top, fg, 2.0),
        fig.line(left, top, right, top, fg, 2.0),
        fig.text(left + 6, top + 16, "case", muted, "start"),
        # the mass, its spring and damper down to the case's floor
        fig.rect(110, my, 60, mh, mass, mass, 2.0, opacity=0.25),
        fig.math_label(140, my + 23, "m", fg),
        fig.upright(124, my + mh, floor, lambda a, b, x: fig.spring(a, b, x, fg, coils=4, amplitude=6.0)),
        fig.upright(156, my + mh, floor, lambda a, b, x: fig.damper(a, b, x, fg, half=6.0)),
        fig.math_label(110, 122, "k", fg, "end"),
        fig.math_label(170, 122, "c", fg, "start"),
        # z, relative to the case
        fig.line(172, my + mh / 2, 196, my + mh / 2, muted, 1.0, dash="2,2"),
        fig.arrow(190, my + mh / 2, 190, my - 6, colors.force, 1.5),
        fig.math_label(196, my + 2, "z", fg, "start"),
        # y, the structure's motion
        fig.arrow(46, floor + 10, 46, floor - 26, colors.force, 1.5),
        fig.math_label(40, floor - 10, "y", fg, "end"),
    ]
    return fig.svg(310, 196, "".join(parts))


LOW, HIGH = 35.0, 360.0  # Hz across the figure


def bands_svg() -> str:
    """The octave bands at 63, 125 and 250 Hz and their thirds, on a log axis; the page's tones at 50 Hz (in the
    63 Hz octave, the 50 Hz third) and 120 Hz (the 125 Hz octave and third)."""
    fg, muted, band, tone = colors.foreground, colors.muted, colors.mass[0], colors.force
    x0, x1 = 16.0, 294.0

    def x(f: float) -> float:
        return x0 + (x1 - x0) * math.log(f / LOW) / math.log(HIGH / LOW)

    parts = [fig.text(x0, 18, "octaves", muted, "start"), fig.text(x0, 76, "thirds", muted, "start")]
    for k, name in ((-4, "63"), (-3, "125"), (-2, "250")):  # base ten: f_m = 1000 × 10^(3k/10)
        fm = 1000.0 * 10 ** (0.3 * k)
        lo, hi = fm * 10**-0.15, fm * 10**0.15
        parts += [
            fig.rect(x(lo), 26, x(hi) - x(lo), 26, band, band, 1.2, opacity=0.18),
            fig.text((x(lo) + x(hi)) / 2, 44, name, fg),
        ]
        for j in (-1, 0, 1):
            ft = fm * 10 ** (0.1 * j)
            a, b = ft * 10**-0.05, ft * 10**0.05
            parts.append(fig.rect(x(a), 84, x(b) - x(a), 22, band, band, 1.0, opacity=0.1))
    for ft, name in ((50.0, "50"), (125.0, "125")):
        parts.append(fig.text(x(ft), 100, name, fg))
    for f, label in ((50.0, "50 Hz"), (120.0, "120 Hz")):
        parts += [
            fig.arrow(x(f), 132, x(f), 110, tone, 1.8),
            fig.text(x(f), 148, label, fg),
        ]
    parts.append(fig.arrow(x0, 160, x1, 160, muted, 1.0))
    parts.append(fig.text(x1, 176, "frequency (log)", muted, "end"))
    return fig.svg(310, 184, "".join(parts))


FIGURES = {"fig:seismic": seismic_svg, "fig:bands": bands_svg}
