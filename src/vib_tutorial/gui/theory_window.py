"""The Theory window: every page's Theory notes in a window of their own, beside the main one.

A contents tree (groups, pages, each page's sections) and the notes, with Back and Forward. Links to
another page's notes stay in the window; *Open this page* shows the page in the main window. With
*Follow the page* on, it shows the notes of each page the main window turns to. Each page keeps the
place it was read to, so going back to one carries on where it left off.

The notes are the pages' own module-level HTML, so a page needn't be built to read its notes.

The search box finds a word or a symbol in every page's notes and the glossary (`search`: their plain text,
section by section, indexed the first time); a hit opens its section with the match selected. The *Glossary
of symbols* (`theory/glossary.html`) heads the contents.
"""

from __future__ import annotations

import html as html_lib
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets

from . import links, notes_view, weak
from .notes_view import NotesBrowser
from .theming import settings

HEADING = re.compile(r"<h3>(.*?)</h3>", re.S)
ANCHOR = re.compile(r'<a name="([^"]+)"></a>')
PAGE_ROLE = QtCore.Qt.ItemDataRole.UserRole  # (page attribute, anchor) of a tree item
GLOSSARY = "glossary"  # the glossary's name where a page's would be
GLOSSARY_TITLE = "Glossary of symbols"
INLINE = re.compile(r"</?(?:sub|sup|b|i|a|span|font)\b[^>]*>", re.I)  # tags inside a word or a formula
TAG = re.compile(r"<[^>]+>")
ALT = re.compile(r'<img [^>]*alt="([^"]*)"[^>]*>')
SNIPPET = 50  # characters either side of a hit
MAX_HITS = 200


def title(name: str) -> str:
    """Page `name`'s title, or the glossary's."""
    return GLOSSARY_TITLE if name == GLOSSARY else links.PAGE_TITLES[name].removesuffix(" page")


def page_html(name: str) -> str:
    """Page `name`'s Theory notes (an attribute from links.COURSE), as its Theory tab shows them; or the glossary."""
    if name == GLOSSARY:
        from .notation import glossary_html

        return anchored(links.load_notes("glossary", glossary=glossary_html()))
    if name == "sim_page":
        from .background import BACKGROUND_HTML

        html = BACKGROUND_HTML
    elif name == "sdof_page":
        from .sdof import notes_html

        html = notes_html()  # its course map with the problems solved
    else:
        from .main_window import _page_class

        module = sys.modules[_page_class(name).__module__]
        html = getattr(module, "NOTES_HTML", None) or module.THEORY_HTML
    return anchored(html)


def anchored(html: str) -> str:
    """`html` with an anchor on every <h3> that lacks one (section-1, section-2, …), so the contents reach it."""
    count = 0

    def add(match: re.Match) -> str:
        nonlocal count
        count += 1
        inner = match.group(1)
        return match.group(0) if ANCHOR.search(inner) else f'<h3><a name="section-{count}"></a>{inner}</h3>'

    return HEADING.sub(add, html)


def sections(html: str) -> list[tuple[str, str]]:
    """(anchor, plain title) of each <h3> in `html`, in order."""
    found = []
    for match in HEADING.finditer(html):
        anchor = ANCHOR.search(match.group(1))
        if anchor:
            title = QtGui.QTextDocumentFragment.fromHtml(match.group(1)).toPlainText().strip()
            found.append((anchor.group(1), title))
    return found


def plain(html: str) -> str:
    """The text of some notes' HTML, as read: a formula's sub- and superscripts kept in their word, an
    equation's image as its alt text."""
    text = TAG.sub(" ", INLINE.sub("", ALT.sub(lambda m: " " + m.group(1) + " ", html)))
    return re.sub(r"\s+", " ", html_lib.unescape(text).replace("\u2060", "")).strip()  # maths' word joiners


@dataclass(frozen=True)
class Hit:
    page: str
    anchor: str  # the section the match is in ("" before the first)
    section: str
    snippet: str  # the match with some text either side


_INDEX: list[tuple[str, str, str, str]] | None = None  # (page, anchor, section title, its text)


def index() -> list[tuple[str, str, str, str]]:
    """Every page's notes and the glossary, cut at their anchored headings, in the course's order."""
    global _INDEX
    if _INDEX is None:
        out = []
        for name in [GLOSSARY, *(n for _, pages in links.COURSE for n, _, _ in pages)]:
            html = page_html(name)
            cuts = [(m.start(), m.group(1)) for m in ANCHOR.finditer(html)]
            titles = dict(sections(html))
            starts = [(0, "")] + cuts
            for (start, anchor), end in zip(starts, [c for c, _ in cuts] + [len(html)]):
                text = plain(html[start:end])
                if text:
                    out.append((name, anchor, plain(titles.get(anchor, "")), text))
        _INDEX = out
    return _INDEX


def search(query: str, limit: int = MAX_HITS) -> list[Hit]:
    """The sections whose text has `query` (any case), each with its first match, in the course's order."""
    q = query.strip().lower()
    if not q:
        return []
    hits = []
    for name, anchor, section, text in index():
        at = text.lower().find(q)
        if at < 0:
            continue
        a, b = max(0, at - SNIPPET), min(len(text), at + len(q) + SNIPPET)
        snippet = ("…" if a else "") + text[a:b] + ("…" if b < len(text) else "")
        hits.append(Hit(name, anchor, section, snippet))
        if len(hits) >= limit:
            break
    return hits


class TheoryWindow(QtWidgets.QWidget):
    """The notes of any page, in their own window. `open_page(name)` shows a page in the main window."""

    def __init__(self, open_page: Callable[[str], None], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle("Theory")
        self._open_page = open_page
        self.page: str | None = None
        self._places: dict[str, int] = {}  # page → where it was read to (scroll bar value)
        self._back: list[tuple[str, int]] = []
        self._forward: list[tuple[str, int]] = []

        self.back_button = QtWidgets.QToolButton()
        self.back_button.setText("◀ Back")
        self.back_button.clicked.connect(self.back)
        self.forward_button = QtWidgets.QToolButton()
        self.forward_button.setText("Forward ▶")
        self.forward_button.clicked.connect(self.forward)
        self.follow = QtWidgets.QCheckBox("Follow the page")
        self.follow.setToolTip(
            "<p>Show the notes of each page the main window turns to. Off, the notes stay "
            "where they are as you move between pages.</p>"
        )
        self.open_button = QtWidgets.QToolButton()
        self.open_button.setText("Open this page")
        self.open_button.setToolTip("Show the page these notes belong to in the main window.")
        self.open_button.clicked.connect(weak.slot(self, lambda self: self.page and self._open_page(self.page)))
        self.search_box = QtWidgets.QLineEdit()
        self.search_box.setPlaceholderText("Search the notes (Ctrl+F)")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.setToolTip("<p>Find a word or a symbol in every page's notes and the glossary.</p>")
        self.search_box.textChanged.connect(self._search)
        bar = QtWidgets.QHBoxLayout()
        for w in (self.back_button, self.forward_button, self.follow):
            bar.addWidget(w)
        bar.addWidget(self.search_box, 1)
        bar.addWidget(self.open_button)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self._on_item)
        self.tree.itemActivated.connect(self._on_item)
        self.tree.itemExpanded.connect(self._fill)
        self._items: dict[str, QtWidgets.QTreeWidgetItem] = {}
        glossary = QtWidgets.QTreeWidgetItem([GLOSSARY_TITLE])
        glossary.setData(0, PAGE_ROLE, (GLOSSARY, ""))
        glossary.setToolTip(0, "The symbols the notes use, their meanings and units, and where each comes in")
        glossary.setChildIndicatorPolicy(QtWidgets.QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
        self.tree.addTopLevelItem(glossary)
        self._items[GLOSSARY] = glossary
        number = 1
        for group, pages in links.COURSE:
            top = QtWidgets.QTreeWidgetItem([group])
            font = top.font(0)
            font.setBold(True)
            top.setFont(0, font)
            top.setFlags(top.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
            self.tree.addTopLevelItem(top)
            for name, title, about in pages:
                item = QtWidgets.QTreeWidgetItem([f"{number}. {title.replace('&&', '&')}"])
                item.setData(0, PAGE_ROLE, (name, ""))
                item.setToolTip(0, about)
                item.setChildIndicatorPolicy(QtWidgets.QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
                top.addChild(item)
                self._items[name] = item
                number += 1
            top.setExpanded(True)

        self.browser = NotesBrowser()
        links.wire(self.browser, self.show_page)  # another page's notes stay in this window

        self.hits = QtWidgets.QListWidget()  # the search's hits, in the tree's place while there's a search
        self.hits.setWordWrap(True)
        self.hits.itemClicked.connect(self._on_hit)
        self.hits.itemActivated.connect(self._on_hit)
        self.left = QtWidgets.QStackedWidget()
        self.left.addWidget(self.tree)
        self.left.addWidget(self.hits)
        split = QtWidgets.QSplitter()
        split.addWidget(self.left)
        split.addWidget(self.browser)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 640])
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(split)
        self.resize(950, 800)
        geometry = settings().value("theory_window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        self.follow.setChecked(settings().value("theory_window/follow", "false") == "true")
        self.follow.toggled.connect(lambda on: settings().setValue("theory_window/follow", "true" if on else "false"))
        QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Back), self, self.back)
        QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Find), self, self.search_box.setFocus)
        QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Forward), self, self.forward)
        notes_view.add_zoom_keys(self)
        self._update_buttons()

    # --------------------------------------------------------------- navigation
    def show_page(self, name: str, anchor: str = "", remember: bool = True) -> None:
        """Show page `name`'s notes: at `anchor` if given, else where it was last read to."""
        if name != self.page:
            if remember and self.page is not None:
                self._back.append((self.page, self._scroll()))
                self._forward.clear()
            self._load(name)
        if anchor:
            QtCore.QTimer.singleShot(
                0, self.browser, weak.slot(self, lambda self: self.browser.scrollToAnchor(anchor))
            )  # once it's laid out
        self._update_buttons()

    def follow_page(self, name: str) -> None:
        """The main window turned to page `name`: show its notes if following."""
        if self.follow.isChecked():
            self.show_page(name)

    def back(self) -> None:
        if self._back:
            assert self.page is not None, "a page showed before the one gone back from"
            self._forward.append((self.page, self._scroll()))
            self._step_to(*self._back.pop())

    def forward(self) -> None:
        if self._forward:
            assert self.page is not None, "pages were gone back from"
            self._back.append((self.page, self._scroll()))
            self._step_to(*self._forward.pop())

    def _step_to(self, name: str, place: int) -> None:
        self._load(name)
        self._places[name] = place
        self._restore(place)
        self._update_buttons()

    def _load(self, name: str) -> None:
        if self.page is not None:
            self._places[self.page] = self._scroll()
            self._items[self.page].setExpanded(False)
        self.page = name
        self.browser.setHtml(page_html(name))
        self._restore(self._places.get(name, 0))
        item = self._items[name]
        item.setExpanded(True)  # its sections
        self.tree.setCurrentItem(item)
        self.tree.scrollToItem(item)
        self.setWindowTitle(f"Theory: {title(name)}")

    def _restore(self, place: int) -> None:
        bar = self.browser.verticalScrollBar()
        bar.setValue(place)
        QtCore.QTimer.singleShot(0, bar, lambda: bar.setValue(place))  # again once it's laid out

    def _scroll(self) -> int:
        return self.browser.verticalScrollBar().value()

    def _fill(self, item: QtWidgets.QTreeWidgetItem) -> None:
        """Put a page's sections under it, the first time it's expanded."""
        target = item.data(0, PAGE_ROLE)
        if target and not target[1] and item.childCount() == 0:
            for anchor, title in sections(page_html(target[0])):
                child = QtWidgets.QTreeWidgetItem([title])
                child.setData(0, PAGE_ROLE, (target[0], anchor))
                child.setToolTip(0, title)  # cut short when the tree is narrow
                item.addChild(child)

    def _on_item(self, item: QtWidgets.QTreeWidgetItem) -> None:
        target = item.data(0, PAGE_ROLE)
        if target:
            self.show_page(*target)

    # --------------------------------------------------------------- search
    def _search(self, text: str) -> None:
        """List the sections with `text`, in the tree's place; back to the tree when it's cleared."""
        self.hits.clear()
        if len(text.strip()) < 2:
            self.left.setCurrentWidget(self.tree)
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)  # the first search reads them all
        try:
            found = search(text)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        for hit in found:
            where = title(hit.page) + (f" › {hit.section}" if hit.section else "")
            item = QtWidgets.QListWidgetItem(f"{where}\n{hit.snippet}")
            item.setData(PAGE_ROLE, (hit.page, hit.anchor))
            self.hits.addItem(item)
        if not found:
            self.hits.addItem(QtWidgets.QListWidgetItem("Nothing in the notes has that."))
        self.left.setCurrentWidget(self.hits)

    def _on_hit(self, item: QtWidgets.QListWidgetItem) -> None:
        target = item.data(PAGE_ROLE)
        if not target:
            return
        self.show_page(*target)
        query = self.search_box.text().strip()
        QtCore.QTimer.singleShot(0, self.browser, weak.slot(self, lambda self: self._select(query)))  # after the scroll

    def _select(self, query: str) -> None:
        """Select the first match of `query` from where the notes are scrolled to (the hit's section)."""
        doc = self.browser.document()
        cursor = QtGui.QTextCursor(doc)
        while True:
            cursor = doc.find(query, cursor)
            if cursor.isNull():
                return
            if self.browser.cursorRect(cursor).top() >= -2:
                break
        self.browser.setTextCursor(cursor)
        self.browser.ensureCursorVisible()

    def _update_buttons(self) -> None:
        self.back_button.setEnabled(bool(self._back))
        self.forward_button.setEnabled(bool(self._forward))
        self.open_button.setEnabled(self.page is not None and self.page != GLOSSARY)

    # --------------------------------------------------------------- window
    def apply_theme(self) -> None:
        """Set the notes again: their links' colour and the figures are fixed when the HTML is set."""
        if self.page is not None:
            place = self._scroll()
            self.browser.setHtml(page_html(self.page))
            self._restore(place)

    def save_geometry(self) -> None:
        settings().setValue("theory_window/geometry", self.saveGeometry())

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.save_geometry()
        super().closeEvent(event)
