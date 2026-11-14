"""The browser every page's Theory notes show in: it draws their equations and figures.

`NotesBrowser` is a QTextBrowser that answers the notes' ``eq:`` images (`maths.render`, at its text's size
and colour) and ``fig:`` images (`figures`), and sets inline maths (``<span class="m">``) a little larger
than the text around it (`maths.INLINE_SIZE`). The document caches the images until its HTML is set again,
so `refresh` sets it again (keeping the place read) after a theme change: `MainWindow._restyle` calls it on
every NotesBrowser it holds, the Theory window on its own. The equations are inked in the theme's own text
colour (`ink`), not the palette's: Qt sends the browser its new palette only after the refresh.

The notes' text size is one for every browser (`zoom`, a factor in `ZOOMS`, kept in the settings): Ctrl+= / -
/ 0 (`add_zoom_keys`, on the main and Theory windows; the browser's own keys too) and Ctrl with the wheel step
it, and `set_zoom` applies it to every NotesBrowser. A browser sets its font to its own size at 1 (`_base`)
times the factor, the equations follow its text and the figures are drawn that much larger, and it keeps the
place read. `test_notes_text_size_is_one_for_all_and_remembered` checks it; the 1280 px width check holds at 1.

Working on it: `tests/test_notes_markup.py` checks every page's Theory tab is one, and that it redraws in a
new theme.
"""

from __future__ import annotations

import re
from functools import partial

from PySide6 import QtCore, QtGui, QtWidgets

from . import figures, maths
from .style import colors
from .theming import settings


def ink() -> str:
    """The equations' colour: the theme's text. Not the browser's palette, which Qt hands it only after
    `MainWindow._restyle` has redrawn the notes, so it would draw them in the previous theme's ink."""
    return colors.text or colors.foreground  # Light keeps the platform's palette: its text is black


ZOOMS = (0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0)  # the notes' text sizes, over the usual
ZOOM_SETTING = "notes/zoom"
WHEEL_STEP = 120  # a mouse wheel's notch (a touchpad sends smaller steps, added up)


def zoom() -> float:
    """The notes' text size over the usual, as kept in the settings (the nearest of ZOOMS)."""
    try:
        factor = float(settings().value(ZOOM_SETTING, 1.0))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        factor = 1.0
    return min(ZOOMS, key=lambda z: abs(z - factor))


def set_zoom(factor: float) -> None:
    """Keep `factor` (the nearest of ZOOMS) and set every NotesBrowser's text to it."""
    factor = min(ZOOMS, key=lambda z: abs(z - factor))
    settings().setValue(ZOOM_SETTING, factor)
    for widget in QtWidgets.QApplication.allWidgets():
        if isinstance(widget, NotesBrowser):
            widget.apply_zoom()


def step_zoom(steps: int) -> None:
    """Larger (`steps` > 0) or smaller text in the notes, by that many of ZOOMS; 0 for the usual size."""
    if steps == 0:
        set_zoom(1.0)
        return
    i = ZOOMS.index(zoom()) + steps
    set_zoom(ZOOMS[max(0, min(len(ZOOMS) - 1, i))])


FIGURE_SIZES = tuple(re.compile(rf'(<img src="fig:[^"]*"[^>]*?\b{a}=")(\d+)(")') for a in ("width", "height"))


def sized(html: str, factor: float) -> str:
    """`html` with its figures' set widths and heights (some notes give them) `factor` times as large."""
    if factor == 1.0:
        return html
    for size in FIGURE_SIZES:
        html = size.sub(lambda m: f"{m.group(1)}{round(int(m.group(2)) * factor)}{m.group(3)}", html)
    return html


ZOOM_KEYS = (("Ctrl+=", 1), ("Ctrl++", 1), ("Ctrl+-", -1), ("Ctrl+0", 0))


def zoom_step_of(event: QtGui.QKeyEvent) -> int | None:
    """The zoom step a key press asks for, or None."""
    Mod = QtCore.Qt.KeyboardModifier
    mods = event.modifiers() & ~(Mod.ShiftModifier | Mod.KeypadModifier)  # + takes Shift on most keyboards
    pressed = QtGui.QKeySequence(QtCore.QKeyCombination(mods, QtCore.Qt.Key(event.key()))).toString()
    return next((steps for key, steps in ZOOM_KEYS if key == pressed), None)


def add_zoom_keys(window: QtWidgets.QWidget) -> None:
    """The text-size keys, wherever the focus is in `window`."""
    for key, steps in ZOOM_KEYS:
        QtGui.QShortcut(QtGui.QKeySequence(key), window, partial(step_zoom, steps))


class NotesBrowser(QtWidgets.QTextBrowser):
    def __init__(self, html: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._html = ""
        self._base = self.font().pointSizeF()  # the text's size at zoom 1
        self._wheel = 0
        self._zoom = 1.0
        self.apply_zoom(redraw=False)
        if html:
            self.setHtml(html)

    def text_px(self) -> float:
        """The pixel size of the browser's text, which its maths is sized from."""
        return float(QtGui.QFontInfo(self.font()).pixelSize())

    def setHtml(self, text: str) -> None:
        self._html = text
        size = round(self.text_px() * maths.INLINE_SIZE)
        self.document().setDefaultStyleSheet(f".m {{ font-size: {size}px; }}")
        super().setHtml(sized(text, self._zoom))

    def refresh(self) -> None:
        """Draw the equations and figures again (in a new theme), at the same place."""
        if not self._html:
            return
        bar = self.verticalScrollBar()
        at = bar.value()
        self.setHtml(self._html)
        bar.setValue(at)

    def apply_zoom(self, redraw: bool = True) -> None:
        """Set the text to the notes' size (`zoom`), its equations and figures with it, at the same place."""
        factor = zoom()
        if factor == self._zoom and redraw:
            return
        self._zoom = factor
        bar = self.verticalScrollBar()
        place = bar.value() / bar.maximum() if bar.maximum() else 0.0
        font = self.font()
        font.setPointSizeF(self._base * factor)
        self.setFont(font)
        if redraw and self._html:
            self.setHtml(self._html)
            bar.setValue(round(place * bar.maximum()))

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        # QTextEdit's own Ctrl+=/- would size only this browser.
        steps = zoom_step_of(event)
        if steps is None:
            super().keyPressEvent(event)
        else:
            step_zoom(steps)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if not event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(event)
            return
        self._wheel += event.angleDelta().y()
        steps = int(self._wheel / WHEEL_STEP)
        if steps:
            self._wheel -= steps * WHEEL_STEP
            step_zoom(steps)
        event.accept()

    def loadResource(self, kind: int, name: QtCore.QUrl | str):
        if kind == QtGui.QTextDocument.ResourceType.ImageResource:
            key = name.toString() if isinstance(name, QtCore.QUrl) else name
            if key.startswith("eq:") and (latex := maths.EQUATIONS.get(key[3:])) is not None:
                return maths.render(latex, self.text_px(), ink())
            if key.startswith("fig:") and (draw := figures.build(key)) is not None:
                return figures.render(draw(), self._zoom)
        return super().loadResource(kind, name)
