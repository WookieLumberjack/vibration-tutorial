"""The Vibro-acoustics & SEA page's Theory figures (see `figures`): coincidence, the two coupled oscillators SEA
rests on, and a bending wave meeting a corner between two plates."""

from __future__ import annotations

import math

from . import figures as fig
from .style import colors


def coincidence_svg() -> str:
    """A supersonic bending wave of length λ_b under the plane sound wave it drives: the wavefronts, λ apart, meet
    the plate λ_b apart, so sound leaves at θ from the normal with sin θ = λ/λ_b = k/k_b."""
    fg, muted, wave, sound = colors.foreground, colors.muted, colors.mass[0], colors.mass[1]
    x0, x1, y0 = 20.0, 290.0, 150.0
    lam_b, amp = 90.0, 6.0
    theta = math.radians(40.0)
    parts = [
        fig.curve(
            [(x, y0 + amp * math.sin(2 * math.pi * (x - x0) / lam_b)) for x in range(int(x0), int(x1) + 1, 3)],
            wave,
            2.5,
        )
    ]
    # wavefronts: lines through the crests, tilted θ from the plate's normal (so θ from the vertical)
    dx, dy = math.sin(theta), -math.cos(theta)  # the sound's direction
    for j in range(4):
        cx = x0 + lam_b * (j + 0.25)  # a crest
        length = 110.0
        # a wavefront is perpendicular to the direction: along (cos θ, sin θ)
        ex, ey = math.cos(theta), math.sin(theta)
        parts.append(fig.line(cx, y0 - amp, cx - length * ex, y0 - amp - length * ey, sound, 1.2))
    cx = x0 + lam_b * 1.25
    parts += [
        fig.line(cx + 20, y0 - 8, cx + 20, y0 - 98, muted, 1.0, dash="3,3"),
        fig.arrow(cx + 20, y0 - 8, cx + 20 + 92 * dx, y0 - 8 + 92 * dy, colors.force, 1.5),
        fig.math_label(cx + 27, y0 - 58, "θ", fg, "start"),
        # λ_b under the plate
        fig.line(x0 + lam_b * 0.25, y0 + 16, x0 + lam_b * 1.25, y0 + 16, fg, 1.0),
        fig.line(x0 + lam_b * 0.25, y0 + 11, x0 + lam_b * 0.25, y0 + 21, fg, 1.0),
        fig.line(x0 + lam_b * 1.25, y0 + 11, x0 + lam_b * 1.25, y0 + 21, fg, 1.0),
        fig.math_label(x0 + lam_b * 0.75, y0 + 34, "λ<sub>b</sub>", fg),
        fig.text(x1 - 4, 22, "sound of wavelength λ", fg, "end"),
    ]
    return fig.svg(310, 190, "".join(parts))


def oscillators_svg() -> str:
    """Two oscillators m₁ and m₂ on their own springs and dampers, joined by a spring k_c, each driven by its own
    noise."""
    fg, m1c, m2c = colors.foreground, colors.mass[0], colors.mass[1]
    y, top, ground = 70.0, 30.0, 140.0
    parts = [fig.floor(20, 290, ground, colors.muted)]
    for cx, color, label in ((85.0, m1c, "1"), (225.0, m2c, "2")):
        parts += [
            fig.rect(cx - 30, y - 22, 60, 44, color, color, 2.0, opacity=0.2),
            fig.math_label(cx, y + 6, f"m<sub>{label}</sub>", fg),
            fig.upright(cx - 15, y + 22, ground, lambda a, b, x: fig.spring(a, b, x, fg, coils=3, amplitude=6.0)),
            fig.upright(cx + 15, y + 22, ground, lambda a, b, x: fig.damper(a, b, x, fg, half=6.0)),
            fig.arrow(cx, top - 14, cx, y - 24, colors.force, 1.5),
            fig.math_label(cx + 6, top - 6, f"F<sub>{label}</sub>(t)", fg, "start"),
        ]
    parts += [
        fig.spring(115, 195, y, fg, coils=4, amplitude=6.0),
        fig.math_label(155, y - 12, "k<sub>c</sub>", fg),
    ]
    return fig.svg(310, 156, "".join(parts))


def junction_svg() -> str:
    """Plate 1 meets plate 2 at a right angle: a bending wave arriving on 1 is partly reflected and partly carried
    on into 2 (τ of its energy), the corner turning but not moving."""
    fg, p1, p2 = colors.foreground, colors.mass[0], colors.mass[1]
    cx, cy = 200.0, 50.0
    parts = [
        fig.line(30, cy, cx, cy, p1, 5.0),
        fig.line(cx, cy, cx, 170, p2, 5.0),
        fig.circle(cx, cy, 5.0, fg),
        fig.text(70, cy - 12, "plate 1", fg),
        fig.text(cx + 12, 150, "plate 2", fg, "start"),
        fig.arrow(50, cy + 18, 150, cy + 18, colors.force, 1.8),
        fig.text(100, cy + 36, "incident", fg),
        fig.arrow(150, cy + 52, 70, cy + 52, colors.force, 1.2, dash="4,3"),
        fig.text(110, cy + 70, "reflected, 1 − τ", fg),
        fig.arrow(cx - 18, cy + 40, cx - 18, 160, colors.force, 1.8),
        fig.math_label(cx - 24, 128, "τ", fg, "end"),
        fig.text(cx + 12, cy + 4, "turns,", colors.muted, "start"),
        fig.text(cx + 12, cy + 20, "doesn't move", colors.muted, "start"),
    ]
    return fig.svg(310, 180, "".join(parts))


FIGURES = {"fig:coincidence": coincidence_svg, "fig:oscillators": oscillators_svg, "fig:junction": junction_svg}
