"""The notes' maths: display equations typeset from LaTeX, inline maths set in a maths text font.

A notes file writes a display equation as ``<eq>LaTeX</eq>`` (raw LaTeX: ``&`` and ``<`` as they are) and
inline maths as ``<m>…</m>`` (the notes' usual Unicode and HTML: ``ω<sub>n</sub> = √(k/m)``).
`links.load_notes` turns them into HTML with `display` and `inline`:

- A display equation becomes ``<img src="eq:<key>" alt="<plain text>">`` in a centred paragraph, its
  LaTeX kept in `EQUATIONS`. `notes_view.NotesBrowser` draws it with `render` (ziamath, in STIX Two Math)
  at the size of the browser's text, in its text colour, so it follows the theme. The alt text
  (`plain`) is what the Theory window's search reads.
- Inline maths stays text, kept on one line, set in STIX Two Text (`FONT`, bundled in ``vib_tutorial/fonts`` with its
  licence, registered by `register_fonts` from `make_app`): letters italic as variables, upper-case Greek,
  digits, the functions and units in `UPRIGHT` and a word of two or more letters in a sub- or superscript
  (a label, ``F<sub>floor</sub>``) upright, and so is an ``e`` raised to a power: Euler's number, as ISO 80000-2
  writes it, apart from a variable e (an eccentricity). In ``<eq>`` write it ``\\mathrm{e}``. ``<rm>…</rm>``
  keeps a stretch upright.

Working on it: `tests/test_maths.py` renders every equation in the notes, checks no LaTeX command is left
unknown or in the alt text, and checks the inline rules.
"""

from __future__ import annotations

import functools
import hashlib
import html as html_lib
import re
import unicodedata
from importlib import resources

from PySide6 import QtCore, QtGui, QtSvg

FONT = "STIX Two Text"
FONTS = resources.files("vib_tutorial") / "fonts"
SCALE = 2  # equations rendered at twice their size, for high-density screens
EQ_SIZE = 1.15  # an equation's size over the text's (STIX's x-height is smaller than most UI fonts')
INLINE_SIZE = 1.1  # inline maths' size over the text's, for the same reason

# LaTeX of every display equation loaded, by its key (`eq:<key>` in the notes' HTML).
EQUATIONS: dict[str, str] = {}

UPRIGHT = frozenset(
    "sin cos tan cot sec csc sinh cosh tanh arcsin arccos arctan exp ln log lg Re Im det max min sup inf lim "
    "sgn arg rms erf tr diag Hz kHz dB rpm rad mm cm kg kN MPa GPa kPa Pa ms µs dof DOF FRF MAC RMS PSD SRS "
    "cpm".split()
)
LETTER = (
    r"[^\W\d_\u0391-\u03a9\u00b2\u00b3\u00b9\u2070-\u209f\u00bc-\u00be]"  # a letter: not upper-case Greek, ² ⁿ ₁ or ½
)
WORD = re.compile(rf"&#?\w+;|[\u0391-\u03a9]+|{LETTER}(?:{LETTER}|[\u0300-\u036f])*")
TAG = re.compile(r"(<[^>]+>)")
MATH = re.compile(r"<m>(.*?)</m>", re.S)  # inline maths in the notes, and in the trials' notes
# Qt breaks a line after | and / even in a nowrap span: a word joiner after each keeps the maths in one piece.
JOINER = "\u2060"
JOIN = str.maketrans({"|": "|" + JOINER, "/": "/" + JOINER})
# A minus that starts an operand (after =, a bracket, a comma or a relation) is a sign: TeX spaces it as one
# when it's written {-}, and ziamath otherwise spaces it as a binary minus.
SIGN = re.compile(r"(^|[=(\[,<>]|\\(?:pm|mp|approx|le|ge|lt|gt|quad|qquad|left\(|left\[)\s?)\s*-")


_registered = False


def register_fonts() -> None:
    """Make the bundled maths text font available to the app (once a QGuiApplication exists)."""
    global _registered
    if _registered:
        return
    _registered = True
    for file in sorted(FONTS.iterdir(), key=lambda f: f.name):
        if file.name.endswith(".otf"):
            QtGui.QFontDatabase.addApplicationFontFromData(QtCore.QByteArray(file.read_bytes()))


# ------------------------------------------------------------------ inline maths
def markup(text: str) -> str:
    """`text` with each `<m>…</m>` set as inline maths: the notes do it as they load, and the trials' notes
    (Python strings, shown both in the Theory and under a page's Things to try) as they're shown."""
    return MATH.sub(lambda m: inline(m.group(1)), text)


def inline(fragment: str) -> str:
    """`<m>` maths as HTML: its variables in italic, the whole set in the maths text font."""
    out = []
    script = 0  # depth inside <sub>/<sup>
    upright = 0  # depth inside <rm> or an author's own <i>/<b>
    parts = TAG.split(fragment)
    for at, part in enumerate(parts):
        if part.startswith("<"):
            name = part.strip("</>").split()[0].lower()
            closing = part.startswith("</")
            if name in ("sub", "sup"):
                script += -1 if closing else 1
            elif name in ("rm", "i", "b"):
                upright += -1 if closing else 1
                if name == "rm":
                    continue  # not HTML: drop the tag itself
            out.append(part)
        elif upright:
            out.append(part)
        else:
            text = WORD.sub(functools.partial(_word, in_script=script > 0), part.translate(JOIN))
            if text.endswith("<i>e</i>") and parts[at + 1 : at + 2] == ["<sup>"]:
                text = text.removesuffix("<i>e</i>") + "e"  # e to a power: Euler's number, upright
            out.append(text)
    return f'<span class="m" style="font-family:\'{FONT}\'; white-space:nowrap">{"".join(out)}</span>'


def _word(match: re.Match[str], in_script: bool) -> str:
    word = match.group(0)
    if word.startswith("&") or word in UPRIGHT or "Α" <= word[0] <= "Ω":
        return word  # an entity, a function or unit, upper-case Greek
    letters = re.sub(r"[̀-ͯ]", "", word)
    if in_script and len(letters) > 1:
        return word  # a label: F_floor, δ_mm
    return f"<i>{word}</i>"


# ------------------------------------------------------------------ display equations
def display(latex: str) -> str:
    """An `<eq>`'s HTML: the image the browser draws, centred, with the equation's plain text as its alt."""
    latex = SIGN.sub(lambda m: m.group(1) + " {-}", " ".join(latex.split()))
    key = hashlib.sha1(latex.encode()).hexdigest()[:12]
    EQUATIONS[key] = latex
    alt = html_lib.escape(plain(latex), quote=True)
    return f'<p align="center"><img src="eq:{key}" alt="{alt}"></p>'


@functools.lru_cache(maxsize=512)
def render(latex: str, px: float, color: str) -> QtGui.QImage:
    """`latex` typeset at `px` (the text's pixel size) in `color`, at SCALE times, marked to show at its size."""
    import ziamath

    ziamath.config.svg2 = False  # Qt's SVG renderer doesn't follow SVG 2's <symbol>/<use> glyphs
    svg = ziamath.Latex(latex, size=px * EQ_SIZE, color=color).svg()
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode()))
    size = renderer.defaultSize()
    image = QtGui.QImage(size * SCALE, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    image.setDevicePixelRatio(SCALE)
    return image


def mathml(latex: str) -> str:
    """`latex` as MathML (what ziamath lays out): the tests look in it for commands it didn't know."""
    import latex2mathml.converter

    return latex2mathml.converter.convert(latex)


# ------------------------------------------------------------------ plain text
SPACES = {r"\,", r"\;", r"\:", r"\!", r"\ ", r"\quad", r"\qquad"}
DROP = {r"\left", r"\right", r"\big", r"\Big", r"\bigg", r"\Bigg", r"\displaystyle", r"\limits"}
STYLES = {r"\mathrm", r"\text", r"\mathbf", r"\boldsymbol", r"\mathit", r"\operatorname", r"\mathsf", r"\mathcal"}
ACCENTS = {r"\dot": "̇", r"\ddot": "̈", r"\bar": "̄", r"\hat": "̂", r"\tilde": "̃"}
SUPERSCRIPT = str.maketrans("0123456789+-−()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁽⁾ⁿ")
NAMED = {r"\lt": "<", r"\gt": ">", r"\lbrace": "{", r"\rbrace": "}", r"\ldots": "…", r"\cdots": "⋯"}
TOKEN = re.compile(r"\\[A-Za-z]+|\\.|[{}^_&]|[^\\{}^_&]")


def plain(latex: str) -> str:
    """A rough Unicode reading of `latex` (fractions as a/b, roots as √): for search, not for show."""
    tokens = TOKEN.findall(latex)
    text, _ = _plain(tokens, 0, stop=None)
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", text).strip())  # x and a dot: ẋ, as the notes write it


def _group(tokens: list[str], i: int) -> tuple[str, int]:
    """The plain text of the group or the single token at `i`, and where it ends."""
    while i < len(tokens) and tokens[i] == " ":
        i += 1
    if i >= len(tokens):
        return "", i
    if tokens[i] == "{":
        return _plain(tokens, i + 1, stop="}")
    return _plain(tokens[i : i + 1], 0, stop=None)[0], i + 1


def _wrap(s: str) -> str:
    s = s.strip()
    return s if len(s) <= 1 or re.fullmatch(r"[\wͰ-Ͽ̀-ͯ]+", s) else f"({s})"


def _plain(tokens: list[str], i: int, stop: str | None) -> tuple[str, int]:
    out = []
    while i < len(tokens):
        t = tokens[i]
        i += 1
        if t == stop:
            break
        if t == "{":
            inner, i = _plain(tokens, i, "}")
            out.append(inner)
        elif t == "}":
            continue
        elif t == r"\frac" or t == r"\dfrac" or t == r"\tfrac":
            a, i = _group(tokens, i)
            b, i = _group(tokens, i)
            out.append(f"{_wrap(a)}/{_wrap(b)}")
        elif t == r"\sqrt":
            a, i = _group(tokens, i)
            out.append(f"√{_wrap(a)}")
        elif t in ACCENTS:
            a, i = _group(tokens, i)
            out.append(a + ACCENTS[t])
        elif t in STYLES:
            a, i = _group(tokens, i)
            out.append(a)
        elif t in (r"\begin", r"\end"):
            _, i = _group(tokens, i)
        elif t == "^":
            a, i = _group(tokens, i)
            sup = a.strip()
            out.append(sup.translate(SUPERSCRIPT) if all(c in "0123456789+-−()n" for c in sup) else f"^{_wrap(sup)}")
        elif t == "_":
            a, i = _group(tokens, i)
            out.append(a)
        elif t == "&":
            out.append(" ")
        elif t == r"\\":
            out.append("; ")
        elif t in SPACES:
            out.append(" ")
        elif t in DROP:
            continue
        elif t in NAMED:
            out.append(NAMED[t])
        elif t.startswith("\\") and len(t) > 1:
            out.append(_symbol(t))
        else:
            out.append(t)
    return "".join(out), i


@functools.cache
def _symbol(command: str) -> str:
    """A LaTeX command's Unicode character (latex2mathml's table), or its name if it has none."""
    if len(command) == 2 and not command[1].isalpha():
        return command[1]  # \{ \| \%
    from latex2mathml import symbols_parser

    code = symbols_parser.convert_symbol(command)
    if code is None:
        return f" {command[1:]} "  # an operator name like \sin: its text
    return "".join(chr(int(c, 16)) for c in code.split("-"))
