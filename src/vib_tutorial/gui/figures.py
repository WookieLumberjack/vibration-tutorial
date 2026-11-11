"""The notes' figures: SVG drawn from the theme's colours when the notes are laid out.

A notes file shows a figure with ``<img src="fig:<name>" width=… height=…>`` in a centred table cell, and a
figure module builds it (``gui/<page>_figures.py``): a function returning SVG text from `colors` (so it follows
the theme, as the plots do), named in that module's ``FIGURES`` dict. The page module passes that dict to the
notes that show it, ``load_notes(<file>, figures=<page>_figures.FIGURES, …)``, which `register`s it, so a
new figure module touches no shared list. `notes_view.NotesBrowser` finds a name (`build`) and draws it
with `render`.

The house style, so the figures read as one set:

- Draw with the primitives here (`line`, `arrow`, `circle`, `arc`, `text`, `spring`, `damper`, `ground`,
  `floor`, `upright` to stand a spring or damper up, `curve` and `area` for a plotted line and the area under it,
  `svg` for the frame), in `colors.foreground` for what the figure is about, `colors.muted` for axes and
  guides, `colors.faint` for construction lines, and a `colors.mass[i]` or `colors.force` where the plots use
  that colour for the same thing (the mass, the force).
- Label symbols with `text(..., italic=True)`: set in the notes' maths font (`maths.FONT`), as ``<m>`` is.
  Words stay in the UI's sans-serif.
- 13 px labels (`FONT`), lines 1.5 px (1 px for guides), no wider than 380 px: the Theory tab is ~407 px
  with the window at 1280 px, less its table cell's padding (the rotor's older figures are wider, and scroll there).

Working on it: `tests/test_figures.py` checks every ``fig:`` name in the notes has a figure, and draws each in
every theme.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from PySide6 import QtCore, QtGui, QtSvg

from .maths import FONT as MATH_FONT
from .style import colors

FONT = 13  # px, the labels' size
SCALE = 2  # rendered at twice the size, for high-density screens


# Every figure loaded, by name: `links.load_notes(name, figures=…)` adds the figures of the notes it loads.
FIGURES: dict[str, Callable[[], str]] = {}


def register(found: dict[str, Callable[[], str]]) -> None:
    """Add a figure module's FIGURES (``fig:`` name → the function that draws it)."""
    for name, draw in found.items():
        if FIGURES.get(name, draw) is not draw:
            raise ValueError(f"two figures are named {name}")
        FIGURES[name] = draw


def all_figures() -> dict[str, Callable[[], str]]:
    """Every figure in the notes: loads every page's notes (which registers their figures) first."""
    from . import links
    from .theory_window import page_html

    for _, pages in links.COURSE:
        for page, _, _ in pages:
            page_html(page)
    return dict(FIGURES)


def build(name: str) -> Callable[[], str] | None:
    """The function that draws figure `name` (``fig:…``), or None. The notes that show it registered it."""
    return FIGURES.get(name)


def render(svg_text: str) -> QtGui.QImage:
    """The SVG as an image at SCALE times its size, marked so it shows at its own size."""
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg_text.encode()))
    size = renderer.defaultSize()
    image = QtGui.QImage(size * SCALE, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    image.setDevicePixelRatio(SCALE)
    return image


# ------------------------------------------------------------------ primitives
def text(x: float, y: float, s: str, color: str, anchor: str = "middle", italic: bool = False, size: int = FONT) -> str:
    style = f" font-style='italic' font-family='{MATH_FONT}'" if italic else ""
    return f"<text x='{x:.1f}' y='{y:.1f}' fill='{color}' font-size='{size}' text-anchor='{anchor}'{style}>{s}</text>"


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1.5, dash: str = "") -> str:
    d = f" stroke-dasharray='{dash}'" if dash else ""
    return f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{color}' stroke-width='{width}'{d}/>"


def head(x: float, y: float, angle: float, color: str, size: float = 9.0) -> str:
    """An arrowhead with its tip at (x, y), pointing along `angle` (radians, SVG axes: y down)."""
    pts = [(x, y)]
    for side in (-1, 1):
        a = angle + math.pi - side * 0.4
        pts.append((x + size * math.cos(a), y + size * math.sin(a)))
    return "<polygon points='" + " ".join(f"{px:.1f},{py:.1f}" for px, py in pts) + f"' fill='{color}'/>"


def arrow(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1.5, dash: str = "") -> str:
    return line(x1, y1, x2, y2, color, width, dash) + head(x2, y2, math.atan2(y2 - y1, x2 - x1), color)


def circle(
    x: float, y: float, r: float, stroke: str = "none", fill: str = "none", width: float = 1.5, dash: str = ""
) -> str:
    d = f" stroke-dasharray='{dash}'" if dash else ""
    return f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r:.1f}' stroke='{stroke}' fill='{fill}' stroke-width='{width}'{d}/>"


def point(cx: float, cy: float, r: float, deg: float) -> tuple[float, float]:
    """The point at `deg` (counterclockwise from +x, as on the page) on a circle; SVG's y points down."""
    a = math.radians(deg)
    return cx + r * math.cos(a), cy - r * math.sin(a)


def arc(
    cx: float, cy: float, r: float, start: float, end: float, color: str, width: float = 1.5, arrow: bool = False
) -> str:
    """An arc counterclockwise from `start` to `end` degrees, with an arrowhead at `end` if asked."""
    x1, y1 = point(cx, cy, r, start)
    x2, y2 = point(cx, cy, r, end)
    large = 1 if (end - start) % 360 > 180 else 0
    path = (
        f"<path d='M {x1:.1f} {y1:.1f} A {r:.1f} {r:.1f} 0 {large} 0 {x2:.1f} {y2:.1f}' fill='none' "
        f"stroke='{color}' stroke-width='{width}'/>"
    )
    if arrow:  # the tangent, counterclockwise on the page, is clockwise in SVG's axes
        path += head(x2, y2, -math.radians(end) - math.pi / 2, color)
    return path


def svg(width: int, height: int, body: str) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}' font-family='sans-serif'>"
        f"<rect x='0.5' y='0.5' width='{width - 1}' height='{height - 1}' fill='{colors.background}' "
        f"stroke='{colors.faint}'/>{body}</svg>"
    )


def rect(
    x: float, y: float, w: float, h: float, stroke: str, fill: str = "none", width: float = 1.5, opacity: float = 1.0
) -> str:
    return (
        f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' stroke='{stroke}' fill='{fill}' "
        f"fill-opacity='{opacity}' stroke-width='{width}'/>"
    )


def spring(x1: float, x2: float, y: float, color: str, coils: int = 6, amplitude: float = 8.0) -> str:
    """A horizontal coil spring from x1 to x2 at height y: straight leads and a zigzag between."""
    lead = 0.15 * (x2 - x1)
    a, b = x1 + lead, x2 - lead
    pts = [(x1, y), (a, y)]
    for i in range(2 * coils):
        pts.append((a + (i + 0.5) * (b - a) / (2 * coils), y + (amplitude if i % 2 == 0 else -amplitude)))
    pts += [(b, y), (x2, y)]
    return (
        "<polyline points='" + " ".join(f"{px:.1f},{py:.1f}" for px, py in pts) + f"' fill='none' stroke='{color}' "
        "stroke-width='1.5' stroke-linejoin='round'/>"
    )


def damper(x1: float, x2: float, y: float, color: str, half: float = 9.0) -> str:
    """A horizontal dashpot from x1 to x2 at height y: a cylinder open to the right, a piston in it."""
    mid = (x1 + x2) / 2
    left, right = mid - 18, mid + 14
    return "".join(
        [
            line(x1, y, left, y, color),
            line(left, y - half, left, y + half, color),
            line(left, y - half, right, y - half, color),
            line(left, y + half, right, y + half, color),
            line(mid + 2, y - half + 3, mid + 2, y + half - 3, color, 2.5),
            line(mid + 2, y, x2, y, color),
        ]
    )


def ground(x: float, y1: float, y2: float, color: str, side: int = -1) -> str:
    """A fixed wall at x from y1 to y2, hatched on the `side` away from the system (-1: its left)."""
    parts = [line(x, y1, x, y2, color, 2.0)]
    step = 9.0
    y = y1 + 4
    while y < y2:
        parts.append(line(x, y, x + side * 8, y + 8, color, 1.0))
        y += step
    return "".join(parts)


def math_label(x: float, y: float, s: str, color: str, anchor: str = "middle", size: int = FONT + 2) -> str:
    """A symbol, in the maths font's italic, a little larger than the words (its x-height is smaller)."""
    return text(x, y, s, color, anchor, italic=True, size=size)


def floor(x1: float, x2: float, y: float, color: str) -> str:
    """A fixed floor at y from x1 to x2, hatched below."""
    parts = [line(x1, y, x2, y, color, 2.0)]
    x = x1 + 4
    while x < x2:
        parts.append(line(x, y, x - 8, y + 8, color, 1.0))
        x += 9.0
    return "".join(parts)


def upright(x: float, y_top: float, y_bottom: float, draw: Callable[[float, float, float], str]) -> str:
    """A horizontal element (`spring`, `damper`: drawn by `draw(x1, x2, y)`) stood upright at x, from y_top
    down to y_bottom."""
    return f"<g transform='translate({x:.1f} {y_top:.1f}) rotate(90)'>{draw(0.0, y_bottom - y_top, 0.0)}</g>"


def curve(points: list[tuple[float, float]], color: str, width: float = 1.5, dash: str = "") -> str:
    """A line through `points` (a plotted curve)."""
    d = f" stroke-dasharray='{dash}'" if dash else ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='{width}' stroke-linejoin='round'{d}/>"


def area(points: list[tuple[float, float]], fill: str, opacity: float = 0.25) -> str:
    """A filled region bounded by `points` (closed back to the first), unstroked: the area under a curve."""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f"<polygon points='{pts}' fill='{fill}' fill-opacity='{opacity}' stroke='none'/>"
