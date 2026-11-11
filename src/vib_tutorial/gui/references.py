"""The books and papers the notes point to, in one list: each page's *Further reading* is made from it.

A notes file cites a reference in its text with ``<cite>key</cite>`` (written out as a link, "Den Hartog
1956", to the entry below) and lists more to read with ``<reading>key key …</reading>`` (anywhere; it shows
nothing where it stands). `links.load_notes` ends the page with a *Further reading* section: what it cites,
in the order it first does, then the rest of its reading.

Working on it: `tests/test_notation.py` checks every key the notes use is here and every entry here is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Reference:
    key: str
    cite: str  # how the text names it: "Den Hartog 1956"
    entry: str  # HTML: authors, title (in <i>), edition, publisher, year
    note: str = ""  # what it's good for, a few words


REFERENCES = (
    Reference(
        "rao",
        "Rao 2017",
        "S. S. Rao, <i>Mechanical Vibrations</i>, 6th edition (SI), Pearson, 2017",
        "the broadest of the textbooks, with many worked examples",
    ),
    Reference(
        "inman",
        "Inman 2014",
        "D. J. Inman, <i>Engineering Vibration</i>, 4th edition, Pearson, 2014",
        "clear on one mass, modal analysis and measurement",
    ),
    Reference(
        "thomson",
        "Thomson and Dahleh 1998",
        "W. T. Thomson and M. D. Dahleh, <i>Theory of Vibration with Applications</i>, 5th edition, "
        "Prentice Hall, 1998",
        "compact, with the classical results derived",
    ),
    Reference(
        "den-hartog",
        "Den Hartog 1956",
        "J. P. Den Hartog, <i>Mechanical Vibrations</i>, 4th edition, McGraw-Hill, 1956 (Dover reprint, 1985)",
        "the classic: physical explanations, absorbers, rotors",
    ),
    Reference(
        "fourier",
        "Fourier 1822",
        "J. Fourier, <i>Théorie analytique de la chaleur</i>, Firmin Didot, Paris, 1822",
        "where the series were first put to work, on heat flow",
    ),
    Reference(
        "gibbs",
        "Gibbs 1899",
        "J. W. Gibbs, Fourier's series, <i>Nature</i> 59, 200 (1898) and 59, 606 (1899)",
        "two letters; the second describes the overshoot",
    ),
    Reference(
        "hewitt",
        "Hewitt and Hewitt 1979",
        "E. Hewitt and R. E. Hewitt, The Gibbs–Wilbraham phenomenon: an episode in Fourier analysis, "
        "<i>Archive for History of Exact Sciences</i> 21, 129–160, 1979",
        "the overshoot's history, from Wilbraham to Gibbs and after",
    ),
    Reference(
        "kreyszig",
        "Kreyszig 2011",
        "E. Kreyszig, <i>Advanced Engineering Mathematics</i>, 10th edition, Wiley, 2011",
        "Fourier series from the start, with their tables and symmetries",
    ),
    Reference(
        "parseval",
        "Parseval 1806",
        "M.-A. Parseval des Chênes, Mémoire sur les séries et sur l'intégration complète d'une équation aux "
        "différences partielles linéaires du second ordre, à coefficients constants, <i>Mémoires présentés à "
        "l'Institut des Sciences, Lettres et Arts, par divers savans</i> 1, 638–648, 1806 (read 1799)",
        "",
    ),
    Reference(
        "wilbraham",
        "Wilbraham 1848",
        "H. Wilbraham, On a certain periodic function, <i>Cambridge and Dublin Mathematical Journal</i> 3, "
        "198–201, 1848",
        "the overshoot, fifty years before Gibbs",
    ),
)
BY_KEY = {r.key: r for r in REFERENCES}
CITE = re.compile(r"<cite>(.*?)</cite>")
READING = re.compile(r"<reading>(.*?)</reading>", re.S)


def cite(key: str) -> str:
    """A citation in the text: a link to the reference's entry in the page's Further reading."""
    return f'<a href="#ref-{key}">{BY_KEY[key].cite}</a>'


def reading_html(keys: list[str]) -> str:
    """A page's Further reading section: `keys`' entries, in that order."""
    items = []
    for key in keys:
        r = BY_KEY[key]
        note = f": {r.note}" if r.note else ""
        items.append(f'<li><a name="ref-{key}"></a>{r.entry}{note}.</li>')
    return '<h3><a name="further-reading"></a>Further reading</h3><ul>' + "".join(items) + "</ul>"
