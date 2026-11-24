"""The notes as static HTML and SVG: every page's Theory, readable in a browser with no Qt (#478 item 9).

`export(out, theme)` writes a page's notes to `<out>/<page>.html` for every page in the course's order, the
glossary to `glossary.html` and the contents to `index.html`, with the bundled fonts in `<out>/fonts`. Each page
is `notes_pages.page_html`'s, made to stand alone by `standalone`:

- an `<eq>`'s image (`eq:` key) becomes its SVG, typeset by `maths.equation_svg` in the theme's text colour;
- a figure (`fig:` name) becomes its SVG, drawn by its figure module in the theme's colours (`style.colors`);
- a `page:` link becomes a link to that page's file, at the same anchor; a goal's *Try it · Practise it*, which
  sets up the running page, goes, since there's no page to set up, and a recap's problem keeps its title as text;
- inline maths keeps its STIX Two Text and the text its IBM Plex Sans, from the bundled fonts.

None of that needs Qt: `maths`, `figures` and `style` are Qt-free, and a figure's labels are measured from
`font_metrics`' tables. A page's notes still come from its module, which for a page not yet rebuilt is a Qt
widget's (see `notes_pages`), so exporting those loads Qt's libraries without using them. `scripts/export_notes.py`
runs it.

Working on it: `tests/test_notes_export.py` exports the glossary and a page and checks what they hold;
`test_qt_free` imports this module without Qt.
"""

from __future__ import annotations

import html as html_lib
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

from . import figures, links, maths, style
from .notes_pages import GLOSSARY, page_html

FONTS = Path(__file__).resolve().parent.parent / "fonts"
TEXT_PX = 16  # the text's size in a browser; the equations are typeset to match it
EQ_IMG = re.compile(r'<img src="eq:([0-9a-f]+)"(?: alt="([^"]*)")?[^>]*>')
FIG_IMG = re.compile(r'<img src="fig:([^"]+)"[^>]*>')
PAGE_LINK = re.compile(r'href="page:([a-z_]+)(#[^"]*)?"')
# A goal's "Try it · Practise it" (`links.goals`) sets up the running page: with no page, the pair goes.
GOAL_ACTIONS = re.compile(r' <a href="trial:[^"]*">[^<]*</a> · <a href="practice:[^"]*">[^<]*</a>')
ACTION_LINK = re.compile(r'<a href="(?:trial|practice):[^"]*">(.*?)</a>', re.S)  # a recap's problem: its title


def file_name(name: str) -> str:
    return f"{name}.html"


def _font_faces() -> str:
    faces = []
    for path in sorted(FONTS.glob("*.otf")):
        family = "STIX Two Text" if path.name.startswith("STIX") else "IBM Plex Sans"
        stem = path.stem.split("-", 1)[1]
        weight = "bold" if "Bold" in stem else "normal"
        slant = "italic" if "Italic" in stem else "normal"
        faces.append(
            f"@font-face {{ font-family: '{family}'; src: url('fonts/{path.name}'); "
            f"font-weight: {weight}; font-style: {slant}; }}"
        )
    return "\n".join(faces)


def _colours(theme: style.Theme) -> tuple[str, str, str]:
    """(text, background, links) of `theme`; Light keeps the platform's palette, so it has none of its own."""
    return theme.text or theme.foreground, theme.base or theme.background, theme.link or theme.mass[0]


def standalone(name: str, html: str, theme: style.Theme) -> str:
    """Page `name`'s notes `html` (`page_html`'s) as a document of its own: see the module."""
    text, background, link = _colours(theme)

    def equation(m: re.Match[str]) -> str:
        latex = maths.EQUATIONS[m.group(1)]
        svg = maths.equation_svg(latex, TEXT_PX, text)
        return f'<span class="eq" role="img" aria-label="{m.group(2) or ""}">{svg}</span>'

    def figure(m: re.Match[str]) -> str:
        draw = figures.build(f"fig:{m.group(1)}")
        if draw is None:
            raise KeyError(f"{name}'s notes show fig:{m.group(1)}, which no figure module registered")
        return f'<span class="fig">{draw()}</span>'

    body = EQ_IMG.sub(equation, html)
    body = FIG_IMG.sub(figure, body)
    body = PAGE_LINK.sub(lambda m: f'href="{file_name(m.group(1))}{m.group(2) or ""}"', body)
    body = ACTION_LINK.sub(r"\1", GOAL_ACTIONS.sub("", body))
    title = "Glossary of symbols" if name == GLOSSARY else links.title(name)
    return _document(title, f'<p class="nav"><a href="index.html">Contents</a></p>\n{body}', text, background, link)


def _document(title: str, body: str, text: str, background: str, link: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_lib.escape(title)}</title>
<style>
{_font_faces()}
body {{ font-family: 'IBM Plex Sans', sans-serif; font-size: {TEXT_PX}px; line-height: 1.5; color: {text};
       background: {background}; max-width: 760px; margin: 0 auto; padding: 16px; }}
a {{ color: {link}; }}
.m {{ font-size: {maths.INLINE_SIZE}em; }}
.eq svg, .fig svg {{ max-width: 100%; height: auto; }}
.fig {{ display: block; text-align: center; }}
table {{ border-collapse: collapse; }}
td, th {{ padding: 2px 8px; vertical-align: top; }}
.nav {{ font-size: 0.9em; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def contents(theme: style.Theme) -> str:
    """The index: the course's groups and pages in order, each linking to its notes, and the glossary."""
    text, background, link = _colours(theme)
    parts = [
        "<h1>Vibration tutorial: the notes</h1>",
        f'<p><a href="{file_name(GLOSSARY)}">Glossary of symbols</a></p>',
    ]
    for group, pages in links.COURSE:
        items = "".join(
            f'<li><a href="{file_name(page)}">{html_lib.escape(title)}</a>: {html_lib.escape(about)}</li>'
            for page, title, about in pages
        )
        parts.append(f"<h2>{html_lib.escape(group)}</h2>\n<ul>{items}</ul>")
    return _document("Vibration tutorial: the notes", "\n".join(parts), text, background, link)


def export(out: Path, theme: str = "Light", pages: Sequence[str] | None = None) -> list[Path]:
    """Write the notes of `pages` (every page and the glossary when None) to `out` as static HTML (see the
    module), drawn in `theme`; returns the files written."""
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(FONTS, out / "fonts", dirs_exist_ok=True)
    names = [GLOSSARY, *links.order()] if pages is None else list(pages)
    before = style.current()
    chosen = style.THEMES[theme]
    style.set_theme(chosen)
    try:
        written = []
        for name in names:
            path = out / file_name(name)
            path.write_text(standalone(name, page_html(name), chosen), encoding="utf-8")
            written.append(path)
        index = out / "index.html"
        index.write_text(contents(chosen), encoding="utf-8")
        return [index, *written]
    finally:
        style.set_theme(before)
