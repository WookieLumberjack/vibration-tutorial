"""The notes' figures and equations drawn by Qt (#316 D7): the only part of the notes that needs it.

`figures` and `maths` make SVG text without Qt (a browser could show it as it is); here it becomes the image a
`notes_view.NotesBrowser` shows: `figure(svg, zoom)`, `equation(latex, px, color)`, each at SCALE times its size
for high-density screens. `install()` (from `make_app`, once a QApplication exists) registers the bundled fonts
with Qt and has `figures` measure its labels with Qt's metrics of the fonts this machine uses (without it, a
figure measures from `font_metrics`' tables).

Working on it: `tests/test_figures.py` and `tests/test_maths.py` draw every figure and equation through here;
the release builds' `--smoke-test` draws one of each.
"""

from __future__ import annotations

import functools
import re

from PySide6 import QtCore, QtGui, QtSvg

from . import figures, maths

SCALE = 2  # drawn at twice their size, for high-density screens


def _image(svg: str, zoom: float = 1.0) -> QtGui.QImage:
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode()))
    size = renderer.defaultSize()
    image = QtGui.QImage(size * (SCALE * zoom), QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    image.setDevicePixelRatio(SCALE)
    return image


def figure(svg: str, zoom: float = 1.0) -> QtGui.QImage:
    """A figure's SVG as an image, marked so it shows at its own size (`zoom` times that)."""
    return _image(svg, zoom)


@functools.lru_cache(maxsize=512)
def equation(latex: str, px: float, color: str) -> QtGui.QImage:
    """`latex` typeset at `px` (the text's pixel size) in `color`, marked to show at its size."""
    return _image(maths.equation_svg(latex, px, color))


def text_width(s: str, italic: bool, size: float) -> float:
    """How wide `s` sets in Qt (its markup dropped), in px: the UI's font, or the maths font in italic."""
    font = QtGui.QFont(maths.FONT) if italic else QtGui.QFont()
    font.setItalic(italic)
    font.setPixelSize(max(1, round(size)))
    return QtGui.QFontMetricsF(font).horizontalAdvance(re.sub(r"<[^>]+>", "", s))


_installed = False


def install() -> None:
    """Register the bundled fonts with Qt and measure the figures' labels with Qt (once a QGuiApplication exists)."""
    global _installed
    if _installed:
        return
    _installed = True
    for file in sorted(maths.FONTS.iterdir(), key=lambda f: f.name):
        if file.name.endswith(".otf"):
            QtGui.QFontDatabase.addApplicationFontFromData(QtCore.QByteArray(file.read_bytes()))
    figures.measure = text_width
