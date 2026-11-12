"""The Random vibration page's Theory figures (see `figures`): a test profile on log–log axes."""

from __future__ import annotations

import math

from . import figures as fig
from .style import colors

F_LO, F_HI = 10.0, 3000.0  # the frequency axis, Hz
G_LO, G_HI = 0.001, 0.2  # the PSD axis, g²/Hz
LEFT, RIGHT, TOP, BOTTOM = 46.0, 300.0, 16.0, 136.0
PROFILE = ((20.0, 0.01), (80.0, 0.04), (500.0, 0.04), (2000.0, 0.0025))  # the notes' worked example
BAND_DB = 3.0  # the tolerance band drawn either side


def _xy(f: float, g: float) -> tuple[float, float]:
    x = LEFT + (RIGHT - LEFT) * math.log(f / F_LO) / math.log(F_HI / F_LO)
    y = BOTTOM - (BOTTOM - TOP) * math.log(g / G_LO) / math.log(G_HI / G_LO)
    return x, y


def profile_svg() -> str:
    """The worked example's profile: +3 dB/oct from 20 to 80 Hz, flat to 500, −6 dB/oct to 2000; dashed, a band
    of ±3 dB round it, as a controller's tolerance."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    line, band = colors.mass[0], colors.force
    parts = [
        fig.line(LEFT, BOTTOM, RIGHT, BOTTOM, faint, 1.0),
        fig.line(LEFT, TOP, LEFT, BOTTOM, faint, 1.0),
    ]
    for f in (10.0, 100.0, 1000.0):
        x, _ = _xy(f, G_LO)
        parts += [fig.line(x, BOTTOM, x, BOTTOM + 4, muted, 1.0), fig.text(x, BOTTOM + 16, f"{f:g}", muted)]
    for g in (0.001, 0.01, 0.1):
        _, y = _xy(F_LO, g)
        parts += [fig.line(LEFT - 4, y, LEFT, y, muted, 1.0), fig.text(LEFT - 6, y + 4, f"{g:g}", muted, "end")]
    for factor in (10 ** (BAND_DB / 10), 10 ** (-BAND_DB / 10)):
        parts.append(fig.curve([_xy(f, g * factor) for f, g in PROFILE], band, 1.0, "4 3"))
    parts.append(fig.curve([_xy(f, g) for f, g in PROFILE], line, 2.2))
    for f, g in PROFILE:
        x, y = _xy(f, g)
        parts.append(fig.circle(x, y, 3.0, line, fill=line))
    (x1, y1), (x2, y2) = _xy(*PROFILE[0]), _xy(*PROFILE[1])
    parts.append(fig.text((x1 + x2) / 2 - 10, (y1 + y2) / 2 - 18, "+3 dB/oct", fg, "end"))
    (x1, y1), (x2, y2) = _xy(*PROFILE[2]), _xy(*PROFILE[3])
    parts.append(fig.text(RIGHT, 32, "−6 dB/oct", fg, "end"))
    x, y = _xy(200.0, 0.04)
    parts += [
        fig.text(x, y - 18, "0.04 g²/Hz", fg),
        fig.text(RIGHT, BOTTOM + 30, "f (Hz)", muted, "end"),
        fig.text(LEFT + 4, TOP - 4, "G (g²/Hz); dashed ±3 dB", muted, "start"),
    ]
    return fig.svg(310, 170, "".join(parts))


FIGURES = {"fig:profile": profile_svg}
