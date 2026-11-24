"""Every page's Theory notes as HTML, gathered without building a page or loading Qt itself (#478 item 9).

`page_html(name)` is a page's notes as its Theory tab shows them (the Theory window's, the search's and the static
export's source): the page module's `NOTES_HTML` or `THEORY_HTML` (the Simulation page's `BACKGROUND_HTML`, the
Single oscillator's `notes_html()`, with its course map), its headings anchored and its "Next: …" line after them;
or the glossary's. A page's notes are read from its module, so a page whose module is a Qt widget's still brings Qt
in with it; one rebuilt to #478's standard keeps its notes in a Qt-free module (`tests/test_readiness.py` checks
the pages that have joined it). A rebuilt page names that module in `links.NOTES_MODULES`; with Qt loaded (the app),
the page's own module is read instead, as it may add what only Qt can give (the Single oscillator's course map
with the reader's progress, kept in Qt's settings).

Working on it: `tests/test_qt_free.py` imports this module without Qt; `test_links` and `test_notes_markup` read
every page through it.
"""

from __future__ import annotations

import importlib
import re
import sys

from . import links

HEADING = re.compile(r"<h3>(.*?)</h3>", re.S)
ANCHOR = re.compile(r'<a name="([^"]+)"></a>')
GLOSSARY = "glossary"  # the glossary's name where a page's would be


def page_html(name: str) -> str:
    """Page `name`'s Theory notes (an attribute from links.COURSE), as its Theory tab shows them; or the glossary."""
    if name == GLOSSARY:
        from .notation import glossary_html

        return anchored(links.load_notes("glossary", glossary=glossary_html()))
    if name == "sim_page":
        from .background import BACKGROUND_HTML

        html = BACKGROUND_HTML
    elif name in links.NOTES_MODULES and "PySide6" not in sys.modules:  # a rebuilt page's notes, without Qt
        html = importlib.import_module(f"vib_tutorial.gui.{links.NOTES_MODULES[name]}").NOTES_HTML
    elif name == "sdof_page":
        from .sdof import notes_html

        html = notes_html()  # its course map with the problems solved
    else:
        module = links.page_module(name)
        html = getattr(module, "NOTES_HTML", None) or module.THEORY_HTML
    return anchored(html) + links.next_html(name)


def anchored(html: str) -> str:
    """`html` with an anchor on every <h3> that lacks one (section-1, section-2, …), so the contents reach it."""
    count = 0

    def add(match: re.Match) -> str:
        nonlocal count
        count += 1
        inner = match.group(1)
        return match.group(0) if ANCHOR.search(inner) else f'<h3><a name="section-{count}"></a>{inner}</h3>'

    return HEADING.sub(add, html)
