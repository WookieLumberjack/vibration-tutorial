"""The browser every page's Theory notes show in: it draws their equations and figures.

`NotesBrowser` is a QTextBrowser that answers the notes' ``eq:`` images (`maths.render`, at its text's size
and colour) and ``fig:`` images (`figures`), and sets inline maths (``<span class="m">``) a little larger
than the text around it (`maths.INLINE_SIZE`). The document caches the images until its HTML is set again,
so `refresh` sets it again (keeping the place read) after a theme change: `MainWindow._restyle` calls it on
every NotesBrowser it holds, the Theory window on its own. The equations are inked in the theme's own text
colour (`ink`), not the palette's: Qt sends the browser its new palette only after the refresh.

Working on it: `tests/test_notes_markup.py` checks every page's Theory tab is one, and that it redraws in a
new theme.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from . import figures, maths
from .style import colors


def ink() -> str:
    """The equations' colour: the theme's text. Not the browser's palette, which Qt hands it only after
    `MainWindow._restyle` has redrawn the notes, so it would draw them in the previous theme's ink."""
    return colors.text or colors.foreground  # Light keeps the platform's palette: its text is black


class NotesBrowser(QtWidgets.QTextBrowser):
    def __init__(self, html: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._html = ""
        if html:
            self.setHtml(html)

    def text_px(self) -> float:
        """The pixel size of the browser's text, which its maths is sized from."""
        return float(QtGui.QFontInfo(self.font()).pixelSize())

    def setHtml(self, text: str) -> None:
        self._html = text
        size = round(self.text_px() * maths.INLINE_SIZE)
        self.document().setDefaultStyleSheet(f".m {{ font-size: {size}px; }}")
        super().setHtml(text)

    def refresh(self) -> None:
        """Draw the equations and figures again (in a new theme), at the same place."""
        if not self._html:
            return
        bar = self.verticalScrollBar()
        at = bar.value()
        self.setHtml(self._html)
        bar.setValue(at)

    def loadResource(self, kind: int, name: QtCore.QUrl | str):
        if kind == QtGui.QTextDocument.ResourceType.ImageResource:
            key = name.toString() if isinstance(name, QtCore.QUrl) else name
            if key.startswith("eq:") and (latex := maths.EQUATIONS.get(key[3:])) is not None:
                return maths.render(latex, self.text_px(), ink())
            if key.startswith("fig:") and (draw := figures.build(key)) is not None:
                return figures.render(draw())
        return super().loadResource(kind, name)
