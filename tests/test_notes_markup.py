"""The notes' markup: equations (`maths`), symbols (`notation`), references (`references`), figures (`figures`)
and the browser that draws them (`notes_view`)."""

from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path

import pytest

from vib_tutorial.gui import links, maths, notation, notes_render, references

GUI = Path(links.__file__).parent


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


ORDER = {page: i for i, (page, _, _) in enumerate(p for _, pages in links.COURSE for p in pages)}


def all_notes() -> dict[str, str]:
    """Every page's notes as shown, and the glossary's: loading them fills `maths.EQUATIONS`."""
    from vib_tutorial.gui.theory_window import GLOSSARY, page_html

    return {name: page_html(name) for name in [GLOSSARY, *ORDER]}


def files_of(page: str) -> list[str]:
    """The notes files page `page`'s module loads (`load_notes("name"`)."""
    from vib_tutorial.gui.links import page_module

    if page == "sim_page":
        module = "vib_tutorial.gui.background"
    else:
        module = page_module(page).__name__
    return _notes_in(module)


def _notes_in(module: str) -> list[str]:
    """The notes files `module` loads, or the module it takes its THEORY_HTML or NOTES_HTML from loads (the rotor
    page's are in `rotor_notes`)."""
    import importlib

    source = Path(importlib.import_module(module).__file__ or "").read_text(encoding="utf-8")
    found = re.findall(r'load_notes\(\s*"([a-z_]+)"', source)
    for name in re.findall(r"from \.(\w+) import [^\n]*\b(?:THEORY|NOTES|BACKGROUND)_HTML\b", source):
        found += _notes_in(module.rsplit(".", 1)[0] + "." + name)
    return found


# ------------------------------------------------------------------ inline maths
def test_inline_maths_sets_variables_in_italic():
    html = maths.inline("T = |F<sub>floor</sub>| ≈ cos Ωt/ω<sub>n</sub> + 15 Hz &lt; x<sub>k+1</sub>, <rm>d</rm>x")
    assert html.startswith(f"<span class=\"m\" style=\"font-family:'{maths.FONT}'; white-space:nowrap\">")
    body = html.removeprefix(html[: html.index(">") + 1]).removesuffix("</span>")
    assert "|" + maths.JOINER in body and "/" + maths.JOINER in body, "no line break after | or /"
    assert body.replace(maths.JOINER, "") == (
        "<i>T</i> = |<i>F</i><sub>floor</sub>| ≈ cos Ω<i>t</i>/<i>ω</i><sub><i>n</i></sub> + 15 Hz &lt; "
        "<i>x</i><sub><i>k</i>+1</sub>, d<i>x</i>"
    )
    # A combining dot stays on its letter, inside the italic.
    assert maths.inline("x\u0307").endswith("<i>x\u0307</i></span>")


def test_no_tag_in_the_notes_holds_a_joiner():
    for name, html in all_notes().items():
        assert not re.search(rf"<[^>]*{maths.JOINER}[^>]*>", html), f"{name}: a word joiner inside a tag"


def test_a_leading_minus_is_a_sign():
    assert re.search(r'src="eq:(\w+)"', maths.display(r"s = -\zeta\omega_n")) is not None
    assert "s = {-}\\zeta\\omega_n" in maths.EQUATIONS.values()
    maths.display(r"(1 - r^2)")
    assert "(1 - r^2)" in maths.EQUATIONS.values(), "a minus between two terms stays binary"


def test_plain_text_reads_an_equation():
    assert maths.plain(r"\delta = \ln\frac{x_k}{x_{k+1}} \approx 2\pi\zeta") == "δ = ln xk/(xk+1) ≈ 2πζ"
    assert maths.plain(r"m\ddot{x} + kx = f(t)") == "mẍ + kx = f(t)"
    assert maths.plain(r"\sqrt{1 - \zeta^2}") == "√(1 - ζ²)"


def test_every_equation_is_known_latex_and_reads_as_text():
    all_notes()
    assert maths.EQUATIONS, "the notes have equations"
    for latex in maths.EQUATIONS.values():
        unknown = re.findall(r"<mi>(\\[A-Za-z]+)</mi>", maths.mathml(latex))
        assert not unknown, f"{latex}: LaTeX that ziamath doesn't know: {unknown}"
        text = maths.plain(latex)
        assert "\\" not in text and "{" not in text and "}" not in text, f"{latex} reads as {text!r}"


def test_every_equation_renders_and_fits_the_theory_tab(app):
    """Drawn at the tests' 12 px text, each is at most 360 px wide: the Theory tab is ~400 px at 1280 px, and
    macOS's text is a little larger. Split a longer one over two <eq>s."""

    all_notes()
    for latex in maths.EQUATIONS.values():
        image = notes_render.equation(latex, 12.0, "#000000")
        assert not image.isNull(), latex
        assert image.deviceIndependentSize().width() <= 360, f"{latex}: too wide for the Theory tab"


# ------------------------------------------------------------------ symbols
def notation_keys() -> dict[str, list[str]]:
    """The symbols each page's notes list in their <notation>, by page."""
    out: dict[str, list[str]] = {}
    for page in ORDER:
        for name in files_of(page):
            text = (links.THEORY / f"{name}.html").read_text(encoding="utf-8")
            for found in notation.NOTATION.findall(text):
                out.setdefault(page, []).extend(found.split())
    return out


def test_every_symbol_is_on_a_page_of_the_course():
    assert len(notation.BY_KEY) == len(notation.SYMBOLS), "keys are unique"
    for s in notation.SYMBOLS:
        assert s.page in ORDER, s
        assert re.fullmatch(r"[A-Za-z0-9_]+", s.key), s.key


def test_a_page_lists_only_symbols_known_by_then_and_each_letter_once():
    found = notation_keys()
    assert "sdof_page" in found, "the first page lists its symbols"
    for page, keys in found.items():
        assert len(keys) == len(set(keys)), f"{page} lists a symbol twice"
        later = [k for k in keys if ORDER[notation.BY_KEY[k].page] > ORDER[page]]
        assert not later, f"{page} lists symbols brought in on later pages: {later}"
        written = [letter for k in keys for letter in notation.letters(notation.BY_KEY[k])]
        twice = {w for w in written if written.count(w) > 1}
        assert not twice, f"{page}'s notation gives {sorted(twice)} two meanings"


def test_glossary_lists_every_symbol_by_course_group():
    html = all_notes()["glossary"]
    for group, _ in links.COURSE:
        if any(s.page in {p for p, _, _ in dict(links.COURSE)[group]} for s in notation.SYMBOLS):
            assert f">{group}</h3>" in html
    for s in notation.SYMBOLS:
        target = f"{s.page}#{s.anchor}" if s.anchor else s.page
        assert f'href="page:{target}"' in html, s.key


# ------------------------------------------------------------------ references
def test_every_reference_is_used_and_every_citation_known():
    used: set[str] = set()
    for path in links.THEORY.iterdir():
        if path.name.endswith(".html"):
            text = path.read_text(encoding="utf-8")
            used |= set(references.CITE.findall(text))
            for found in references.READING.findall(text):
                used |= set(found.split())
    assert used <= set(references.BY_KEY), f"unknown references: {sorted(used - set(references.BY_KEY))}"
    assert used == set(references.BY_KEY), f"listed but never used: {sorted(set(references.BY_KEY) - used)}"


def test_further_reading_closes_a_page_and_its_citations_land():
    from vib_tutorial.gui.links import next_html

    html = all_notes()["sdof_page"].removesuffix(next_html("sdof_page"))  # then the link on to the next page
    assert html.rstrip().endswith("</ul>") and '<a name="further-reading"></a>Further reading' in html
    for key in ("inman", "rao", "thomson", "den-hartog"):
        assert f'<a name="ref-{key}"></a>' in html


# ------------------------------------------------------------------ figures
def test_every_figure_named_in_the_notes_exists():
    from vib_tutorial.gui import figures

    named = set()
    for html in all_notes().values():
        named |= set(re.findall(r'src="(fig:[^"]+)"', html))
    assert named == set(figures.all_figures()), "every figure is shown, and every one shown is drawn"


def test_figures_follow_the_theme(app):
    """The figures are SVG drawn from the theme's colours, rendered as the notes are laid out."""
    from PySide6 import QtCore, QtGui

    from vib_tutorial.gui import figures, style
    from vib_tutorial.gui.notes_view import NotesBrowser

    browser = NotesBrowser()
    before = style.THEMES[style.colors.name]
    try:
        for theme in style.THEMES.values():
            style.set_theme(theme)
            for name, build in figures.all_figures().items():
                svg = build()
                assert "baseline-shift" not in svg and " dy=" not in svg, f"{name}: Qt ignores a tspan's shift"
                assert theme.background in svg and theme.foreground in svg
                assert theme.mass[0] in svg or theme.force in svg, f"{name} draws its subject in the plots' colour"
                image = browser.loadResource(QtGui.QTextDocument.ResourceType.ImageResource, QtCore.QUrl(name))
                assert isinstance(image, QtGui.QImage) and not image.isNull()
                assert 300 < image.deviceIndependentSize().width() <= 560
                pixels = {image.pixel(x, y) for x in range(0, image.width(), 7) for y in range(0, image.height(), 7)}
                assert len(pixels) > 5, f"{name} is drawn, not blank"
    finally:
        style.set_theme(before)
        browser.deleteLater()


def test_a_figures_labels_are_measured_without_qt_as_qt_would(app):
    """`font_metrics` has every character a figure's labels use, and measures a label as Qt does with the same
    fonts (#316 D7): within 1 % for the bundled maths font, and for the sans-serif on the machine that generated
    the tables (another machine's UI font may differ: in the app, Qt measures)."""
    from PySide6 import QtGui

    from vib_tutorial.gui import figures, font_metrics, notes_render

    used = set()
    for draw in figures.all_figures().values():
        for text in re.findall(r"<text[^>]*>(.*?)</text>", draw(), re.S):
            used |= set(re.sub(r"<[^>]+>", "", text))
    missing = used - font_metrics.SANS.keys()
    assert not missing, f"rerun scripts/font_metrics.py: {sorted(missing)}"
    notes_render.install()  # as make_app does: the fonts registered, the labels measured by Qt
    assert figures.measure is notes_render.text_width
    for s in ("ω<sub>n</sub>", "k₁ = 400 N/m", "Heavy spot", "x(t)"):
        # at 200 px: at 13 px Windows hints each glyph's advance to a whole pixel (3 % off on a short label)
        assert font_metrics.width(s, True, 200) == pytest.approx(notes_render.text_width(s, True, 200), rel=0.01)
    if QtGui.QFontInfo(QtGui.QFont()).family() == font_metrics.SANS_FAMILY:
        for s in ("Heavy spot", "keyphasor", "whirl orbit"):
            assert font_metrics.width(s, False, 200) == pytest.approx(notes_render.text_width(s, False, 200), rel=0.01)


# ------------------------------------------------------------------ the browser
def test_every_theory_tab_is_a_notes_browser():
    """A browser given a page's notes is a NotesBrowser, or its equations and figures don't draw."""
    notes = re.compile(r"(\S+)\.setHtml\((?:NOTES_HTML|THEORY_HTML|BACKGROUND_HTML|self\._notes_html|notes_html\(\))")
    made = re.compile(r"(\S+) = (QtWidgets\.QTextBrowser|NotesBrowser)\(")
    bad = []
    for path in sorted(GUI.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        kinds = {name: kind for name, kind in made.findall(source)}
        for name in notes.findall(source):
            if kinds.get(name) != "NotesBrowser":
                bad.append(f"{path.name}: {name}")
    assert not bad, "show the notes in a notes_view.NotesBrowser:\n" + "\n".join(bad)


def test_notes_browser_redraws_its_equations_in_a_new_colour(app):
    from PySide6 import QtCore, QtGui, QtWidgets

    from vib_tutorial.gui.notes_view import NotesBrowser

    html = maths.display(r"\omega_n = \sqrt{k/m}") + "<p>" + maths.inline("ω<sub>n</sub>") + "</p>"
    name = QtCore.QUrl(re.search(r'src="([^"]+)"', html).group(1))
    browser = NotesBrowser(html)
    browser.resize(400, 200)
    browser.show()

    def ink() -> set[int]:
        QtWidgets.QApplication.processEvents()
        image = browser.document().resource(QtGui.QTextDocument.ResourceType.ImageResource, name)
        assert isinstance(image, QtGui.QImage) and not image.isNull()
        return {image.pixel(x, y) & 0xFFFFFF for x in range(image.width()) for y in range(image.height())}

    from vib_tutorial.gui import style

    before = style.current()
    try:
        # Ink from the theme, not the palette: Qt sends the browser a new palette after the refresh.
        style.set_theme(style.LIGHT)
        browser.refresh()
        assert 0xFF0000 not in ink()
        style.set_theme(replace(style.DARK, text="#ff0000"))
        browser.refresh()
        assert 0xFF0000 in ink()
        # Inline maths is a little larger than the text round it.
        sheet = browser.document().defaultStyleSheet()
        assert f"font-size: {round(browser.text_px() * maths.INLINE_SIZE)}px" in sheet
    finally:
        style.set_theme(before)
        browser.close()
        browser.deleteLater()


def test_fonts_are_bundled_with_their_licence():
    names = sorted(p.name for p in maths.FONTS.iterdir())
    # STIX Two Text (the maths, OFL.txt) and IBM Plex Sans (the UI, #316 D10), four faces each
    assert {"OFL.txt", "OFL-IBMPlexSans.txt"} <= set(names)
    for family in ("STIXTwoText", "IBMPlexSans"):
        faces = [n for n in names if n.startswith(family + "-") and n.endswith(".otf")]
        assert len(faces) == 4, faces
        assert os.path.getsize(str(maths.FONTS / f"{family}-Regular.otf")) > 100_000


def test_the_app_sets_its_text_in_the_bundled_ui_font(app):
    """#316 D10: IBM Plex Sans at 12 logical px on every platform, so the layouts measure the same everywhere."""
    from PySide6 import QtGui

    from vib_tutorial.gui import notes_render

    notes_render.install()
    font = QtGui.QGuiApplication.font()
    assert QtGui.QFontInfo(font).family() == notes_render.UI_FONT == "IBM Plex Sans"
    assert QtGui.QFontInfo(font).pixelSize() == notes_render.UI_PX
    bold = QtGui.QFont(font)
    bold.setBold(True)
    assert QtGui.QFontInfo(bold).family() == "IBM Plex Sans"  # its own bold face, not a synthesized one
