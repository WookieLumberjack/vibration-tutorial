"""The Complex modes page's Theory figures (see `figures`): proportional damping against a single damper."""

from __future__ import annotations

from . import figures as fig
from .style import colors


def _row(y: float, dampers: tuple[bool, ...], caption: str) -> str:
    """The 4-mass chain on one line, a spring in every gap and a damper below the springs that have one."""
    fg, muted = colors.foreground, colors.muted
    wall, gap, w, h, upper = 22.0, 44.0, 30.0, 34.0, 9.0
    parts = [fig.ground(wall, y - 22, y + 22, muted)]
    x = wall
    for i, damped in enumerate(dampers):
        left = x + gap
        if damped:
            parts += [
                fig.line(x, y - upper, x, y + upper, fg),
                fig.line(left, y - upper, left, y + upper, fg),
                fig.spring(x, left, y - upper, fg, coils=3, amplitude=4.0),
                fig.damper(x, left, y + upper, colors.force, half=5.0),
            ]
        else:
            parts.append(fig.spring(x, left, y, fg, coils=3, amplitude=4.0))
        mass = colors.mass[0]
        parts += [
            fig.rect(left, y - h / 2, w, h, mass, mass, 2.0, opacity=0.2),
            fig.math_label(left + w / 2, y + 6, f"m<sub>{i + 1}</sub>", fg),
        ]
        x = left + w
    parts.append(fig.text(180, y + h / 2 + 17, caption, muted))
    return "".join(parts)


def dampers_svg() -> str:
    """Above: a damper beside every spring, c_i = βk_i, so C = βK: proportional, real modes. Below: one damper,
    at the base: C_m is c1 φ1 φ1ᵀ, every mode coupled to every other, complex modes."""
    return fig.svg(
        360,
        150,
        _row(32.0, (True,) * 4, "c₁ … c₄ = βk₁ … βk₄: proportional, real modes")
        + _row(105.0, (True, False, False, False), "c₁ alone: coupled, complex modes"),
    )


FIGURES = {"fig:dampers": dampers_svg}
