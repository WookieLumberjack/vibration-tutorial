"""The Model updating page's Theory figures (see `figures`): the structure tested and the model with its error,
side by side, and where spring 3 stretches in each of the uniform chain's modes (its share of the strain
energy, which is what each frequency's sensitivity to it is)."""

from __future__ import annotations

import numpy as np

from ..core.modal import modal_analysis
from ..core.model import ChainSystem
from . import figures as fig
from .style import colors


def _chain(y: float, label: str, stiff: bool) -> str:
    """The uniform 4-mass chain on one line; spring 3 drawn heavier and marked × 1.3 if `stiff`."""
    fg, muted, mass, force = colors.foreground, colors.muted, colors.mass[0], colors.force
    wall, gap, w, h = 80.0, 28.0, 26.0, 24.0
    parts = [fig.ground(wall, y - 18, y + 18, muted), fig.text(6, y + 5, label, fg, "start")]
    x = wall
    for i in range(4):
        left = x + gap
        odd = stiff and i == 2
        parts += [
            fig.spring(x, left, y, force if odd else fg, coils=3, amplitude=6.0 if odd else 5.0),
            fig.rect(left, y - h / 2, w, h, mass, mass, 2.0, opacity=0.2),
            fig.math_label(left + w / 2, y + 6, f"m<sub>{i + 1}</sub>", fg, size=13),
        ]
        if odd:
            parts.append(fig.text((x + left) / 2, y - 16, "× 1.3", force))
        x = left + w
    return "".join(parts)


def model_svg() -> str:
    """Above, the structure (the Simulation page's chain), which the virtual test measures; below, the model,
    the same chain with k3 30 % too stiff. Correlation compares their modes; updating changes the model."""
    muted = colors.muted
    parts = [
        _chain(40.0, "structure", stiff=False),
        _chain(128.0, "model", stiff=True),
        fig.arrow(150, 62, 150, 104, muted, 1.2),
        fig.text(158, 88, "correlate, then update", muted, "start"),
    ]
    return fig.svg(302, 158, "".join(parts))


def stretch_svg() -> str:
    """Each mode of the uniform chain, its mass-normalized displacements against position (the wall at 0), with
    spring 3 (between m2 and m3) drawn heavy, and the share of the mode's strain energy it holds."""
    fg, muted, force = colors.foreground, colors.muted, colors.force
    result = modal_analysis(ChainSystem.uniform(4))
    width, top, half = 68.0, 40.0, 26.0
    parts = []
    for r, mode in enumerate(result.modes):
        phi = np.concatenate(([0.0], mode.shape_mass_normalized))
        stretch = np.diff(phi)
        share = 400.0 * stretch[2] ** 2 / mode.omega_n**2
        x0 = 6 + r * (width + 6)
        mid = top + half
        pts = [(x0 + 6 + i * (width - 12) / 4, mid - 1.4 * half * phi[i]) for i in range(5)]
        parts += [
            fig.line(x0 + 4, mid, x0 + width - 4, mid, muted, 0.8, dash="2,2"),
            fig.curve(pts, fg, 1.4),
            fig.line(*pts[2], *pts[3], force, 3.2),
            fig.text(x0 + width / 2, 22, f"mode {r + 1}", fg),
            fig.text(x0 + width / 2, top + 2 * half + 26, f"{100 * share:.0f} %", force),
        ]
        parts += [fig.circle(px, py, 2.2, fg, fg) for px, py in pts[1:]]
    parts.append(fig.text(151, 140, "spring 3's share of the strain energy", muted))
    return fig.svg(302, 150, "".join(parts))


FIGURES = {"fig:model": model_svg, "fig:stretch": stretch_svg}
