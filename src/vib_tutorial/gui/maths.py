"""The notes' math: display equations typeset from LaTeX, inline math set in a math text font.

A notes file writes a display equation as ``<eq>LaTeX</eq>`` (raw LaTeX: ``&`` and ``<`` as they are) and
inline maths as ``<m>…</m>`` (the notes' usual Unicode and HTML: ``ω<sub>n</sub> = √(k/m)``).
`links.load_notes` turns them into HTML with `display` and `inline`:

- A display equation becomes ``<img src="eq:<key>" alt="<plain text>">`` in a centered paragraph, its
  LaTeX kept in `EQUATIONS`. `notes_view.NotesBrowser` draws it with `notes_render.equation` (ziamath's SVG, in STIX Two Math)
  at the size of the browser's text, in its text color, so it follows the theme. The alt text
  (`plain`) is what the Theory window's search reads.
- Inline math stays text, kept on one line, set in STIX Two Text (`FONT`, bundled in ``vib_tutorial/fonts`` with its
  license, registered by `notes_render.install` from `make_app`): letters italic as variables, upper-case Greek,
  digits, the functions and units in `UPRIGHT` and a word of two or more letters in a sub- or superscript
  (a label, ``F<sub>floor</sub>``) upright, and so is an ``e`` raised to a power: Euler's number, as ISO 80000-2
  writes it, apart from a variable e (an eccentricity). In ``<eq>`` write it ``\\mathrm{e}``. ``<rm>…</rm>``
  keeps a stretch upright.

Typesetting is ~27 ms an equation (the Jeffcott rotor's notes: 1.4 s the first time their tab showed), so
`equation_svg` keeps each one's SVG on disk (`cache_dir`: the platform's cache folder, or `VIB_CACHE`), typeset in a
placeholder color that's swapped for the theme's, keyed by the LaTeX, the size and ziamath's version: once
typeset, any session and any theme reads it back (#409 P-18). A cache that can't be read or written is skipped.

Working on it: `tests/test_notes_markup.py` renders every equation in the notes, checks no LaTeX command is left
unknown or in the alt text, and checks the inline rules (and the cache; the suite's is a temporary folder,
conftest's `VIB_CACHE`).
"""

from __future__ import annotations

import functools
import hashlib
import html as html_lib
import os
import re
import sys
import unicodedata
from importlib import resources
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

FONT = "STIX Two Text"
FONTS = resources.files("vib_tutorial") / "fonts"
EQ_SIZE = 1.15  # an equation's size over the text's (STIX's x-height is smaller than most UI fonts')
INLINE_SIZE = 1.1  # inline math' size over the text's, for the same reason

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
MATH = re.compile(r"<m>(.*?)</m>", re.S)  # inline math in the notes, and in the trials' notes
# Qt breaks a line after | and / even in a nowrap span: a word joiner after each keeps the math in one piece.
JOINER = "\u2060"
JOIN = str.maketrans({"|": "|" + JOINER, "/": "/" + JOINER})
# A minus that starts an operand (after =, a bracket, a comma or a relation) is a sign: TeX spaces it as one
# when it's written {-}, and ziamath otherwise spaces it as a binary minus.
SIGN = re.compile(r"(^|[=(\[,<>]|\\(?:pm|mp|approx|le|ge|lt|gt|quad|qquad|left\(|left\[)\s?)\s*-")


# ------------------------------------------------------------------ inline math
def markup(text: str) -> str:
    """`text` with each `<m>…</m>` set as inline math: the notes do it as they load, and the trials' notes
    (Python strings, shown both in the Theory and under a page's Things to try) as they're shown."""
    return MATH.sub(lambda m: inline(m.group(1)), text)


def inline(fragment: str) -> str:
    """`<m>` math as HTML: its variables in italic, the whole set in the math text font."""
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
    """An `<eq>`'s HTML: the image the browser draws, centered, with the equation's plain text as its alt."""
    latex = SIGN.sub(lambda m: m.group(1) + " {-}", " ".join(latex.split()))
    key = hashlib.sha1(latex.encode()).hexdigest()[:12]
    EQUATIONS[key] = latex
    alt = html_lib.escape(plain(latex), quote=True)
    return f'<p align="center"><img src="eq:{key}" alt="{alt}"></p>'


CACHE_ENV = "VIB_CACHE"  # a folder for the typeset equations instead of the platform's cache folder
PLACEHOLDER = "#010203"  # the color equations are typeset and kept in, swapped for the theme's


def cache_dir() -> Path:
    """Where typeset equations are kept: `VIB_CACHE`, else the platform's cache folder."""
    if root := os.environ.get(CACHE_ENV):
        return Path(root) / "equations"
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "vib-tutorial" / "equations"


@functools.cache
def _ziamath_version() -> str:
    try:
        return version("ziamath")
    except PackageNotFoundError:  # a frozen app without its metadata: typeset afresh each session
        return f"unknown-{os.getpid()}"


def equation_svg(latex: str, px: float, color: str) -> str:
    """`latex` typeset as SVG at `px` (the text's pixel size) in `color` (`notes_render.equation` draws it): read
    from the cache if it's been typeset before (see the module's docstring)."""
    key = hashlib.sha1(f"{_ziamath_version()}|{EQ_SIZE}|{px!r}|{latex}".encode()).hexdigest()
    path = cache_dir() / f"{key}.svg"
    try:
        svg = path.read_text(encoding="utf-8")
    except OSError:
        svg = _typeset(latex, px)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_name(f"{key}.{os.getpid()}.part")
            part.write_text(svg, encoding="utf-8")
            os.replace(part, path)  # whole or not at all: another window may be reading it
        except OSError:
            pass  # read-only or full: typeset each time
    return svg.replace(f'fill="{PLACEHOLDER}"', f'fill="{color}"')


def _typeset(latex: str, px: float) -> str:
    import ziamath

    ziamath.config.svg2 = False  # Qt's SVG renderer doesn't follow SVG 2's <symbol>/<use> glyphs
    return ziamath.Latex(latex, size=px * EQ_SIZE, color=PLACEHOLDER).svg()


def mathml(latex: str) -> str:
    """`latex` as MathML (what ziamath lays out): the tests look in it for commands it didn't know. ziamath is
    imported first: it teaches latex2mathml the commands it adds (`\\arg`), so without it the answer depended on
    whether anything had typeset an equation yet."""
    import latex2mathml.converter
    import ziamath  # noqa: F401

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
