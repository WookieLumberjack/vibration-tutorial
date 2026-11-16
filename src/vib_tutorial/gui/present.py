"""Present: the pages with their side panes folded away, for a projector or a shared screen (#316 B8).

Most pages are three panes in a splitter (`lesson.three_panes`): the controls, the main plots, the tabs. `F5`
steps every page through three levels (`LEVELS`), and back:

- 0, *all panes*: as usual;
- 1, *no controls*: the left pane (the parameters, trials and challenges) hidden, so the plots take its room;
- 2, *centre only*: the right pane (the tabs, the notes) hidden too, the main plots the whole width.

A page whose outermost splitter has only two panes loses its left one at level 1 and keeps the other. A hidden
pane gives its minimum width back, so the window's minimum shrinks with the level: a larger *Size* fits a
smaller screen (`display_size.fits` reads the window as it is). The level is kept in the settings (`KEY`), as
a new *Size* needs a restart, and pages built later (`MainWindow._page`) take it as they are built.

Working on it: `tests/test_present.py` checks the levels on every page and the window's minimum with them.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from . import theming

LEVELS = ("all panes", "no controls", "centre only")
KEY = "display/present"


def splitter(page: QtWidgets.QWidget) -> QtWidgets.QSplitter | None:
    """The page's outermost horizontal splitter of two panes or more (its panes), if it has one."""
    for s in page.findChildren(QtWidgets.QSplitter):
        if s.orientation() != QtCore.Qt.Orientation.Horizontal or s.count() < 2:
            continue
        outer = s.parentWidget()
        while outer is not None and outer is not page and not isinstance(outer, QtWidgets.QSplitter):
            outer = outer.parentWidget()
        if outer is page:
            return s
    return None


def apply(page: QtWidgets.QWidget, level: int) -> None:
    """Show or hide the page's side panes for `level`."""
    s = splitter(page)
    if s is None:
        return
    panes = [s.widget(i) for i in range(s.count())]
    first, last = panes[0], panes[-1]
    assert first is not None and last is not None, "a splitter's panes"
    first.setVisible(level < 1)
    if len(panes) >= 3:
        last.setVisible(level < 2)
    # a hidden page's layout keeps its old minimum until it's shown: worked out again now, for the window's
    widget: QtWidgets.QWidget | None = s
    while widget is not None:
        if (layout := widget.layout()) is not None:
            layout.invalidate()
        widget.updateGeometry()
        widget = widget.parentWidget()


def saved() -> int:
    try:
        level = int(str(theming.settings().value(KEY, 0)))
    except (TypeError, ValueError):
        return 0
    return level if 0 <= level < len(LEVELS) else 0


def save(level: int) -> None:
    theming.settings().setValue(KEY, level)
