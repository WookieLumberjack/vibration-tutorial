"""The FRF matrix page's Theory figures (see `figures`): where H₂₂'s antiresonances come from."""

from __future__ import annotations

from . import figures as fig
from .style import colors


def _row(y: float, held: bool) -> str:
    """The uniform 4-mass chain on one line: ground, k1, m1, k2, m2, k3, m3, k4, m4. Above, a force on m2;
    below (held), m2 pinned and the two pieces either side free to vibrate on their own."""
    fg, muted, force = colors.foreground, colors.muted, colors.force
    wall, gap, w, h = 24.0, 40.0, 30.0, 30.0
    parts = [fig.ground(wall, y - 20, y + 20, muted)]
    x = wall
    for i in range(4):
        left = x + gap
        piece = 0 if i == 0 else 1  # m1 is one piece, m3 and m4 the other
        tone = colors.mass[piece] if held and i != 1 else colors.mass[0]
        parts.append(fig.spring(x, left, y, fg, coils=3, amplitude=5.0))
        if held and i == 1:
            parts += [
                fig.rect(left, y - h / 2, w, h, muted, muted, 1.5, opacity=0.35),
                fig.text(left + w / 2, y + h / 2 + 16, "held", muted),
            ]
        else:
            parts.append(fig.rect(left, y - h / 2, w, h, tone, tone, 2.0, opacity=0.2))
        parts.append(fig.math_label(left + w / 2, y + 6, f"m<sub>{i + 1}</sub>", fg))
        x = left + w
    if not held:
        m2 = wall + 2 * gap + w + w / 2
        parts += [
            fig.arrow(m2 - 22, y - h / 2 - 12, m2 + 2, y - h / 2 - 12, force, 2.0),
            fig.math_label(m2 - 26, y - h / 2 - 7, "F<sub>2</sub>", fg, "end"),
            fig.text(m2, y + h / 2 + 16, "force and response at m₂", muted),
        ]
    else:
        parts += [
            fig.text(wall + gap + w / 2 + 2, y + h / 2 + 16, "4.50 Hz", colors.mass[0]),
            fig.text(wall + 3 * gap + 3 * w + 20, y + h / 2 + 16, "1.97 and 5.15 Hz", colors.mass[1]),
        ]
    return "".join(parts)


def held_svg() -> str:
    """Above: H₂₂, a force and its response both at m2. Below: m2 held still; m1 between the ground and m2
    vibrates at 4.50 Hz, m3 and m4 hanging off m2 at 1.97 and 5.15 Hz: H₂₂'s antiresonances."""
    return fig.svg(360, 170, _row(52.0, held=False) + _row(132.0, held=True))


FIGURES = {"fig:held": held_svg}
