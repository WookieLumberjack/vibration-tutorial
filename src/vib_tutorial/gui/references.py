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
        "den-hartog-1931",
        "Den Hartog 1931",
        "J. P. Den Hartog, Forced vibrations with combined Coulomb and viscous friction, <i>Transactions of the "
        "ASME</i> 53 (APM-53-9), 107–115, 1931",
        "the exact sliding motion, and when friction bounds a resonance",
    ),
    Reference(
        "coulomb",
        "Coulomb 1785",
        "C.-A. Coulomb, Théorie des machines simples, en ayant égard au frottement de leurs parties et à la roideur "
        "des cordages, <i>Mémoires de mathématique et de physique présentés à l'Académie royale des sciences</i> "
        "10, 1785 (the Academy's prize of 1781)",
        "dry friction's laws, from experiment",
    ),
    Reference(
        "crandall",
        "Crandall 1970",
        "S. H. Crandall, The role of damping in vibration theory, <i>Journal of Sound and Vibration</i> 11, 3–18, 1970",
        "what each damping model can and can't do, hysteretic damping's lack of a causal form among them",
    ),
    Reference(
        "maxwell",
        "Maxwell 1867",
        "J. C. Maxwell, On the dynamical theory of gases, <i>Philosophical Transactions of the Royal Society of "
        "London</i> 157, 49–88, 1867",
        "the model of a stress that relaxes, a spring and a viscosity in series",
    ),
    Reference(
        "nashif",
        "Nashif, Jones and Henderson 1985",
        "A. D. Nashif, D. I. G. Jones and J. P. Henderson, <i>Vibration Damping</i>, Wiley, 1985",
        "damping materials, their measurement and their use in structures",
    ),
    Reference(
        "rivin",
        "Rivin 2003",
        "E. I. Rivin, <i>Passive Vibration Isolation</i>, ASME Press, 2003",
        "isolation in practice: six degrees of freedom, two stages, nonlinear mounts, waves in isolators",
    ),
    Reference(
        "zener",
        "Zener 1948",
        "C. Zener, <i>Elasticity and Anelasticity of Metals</i>, University of Chicago Press, 1948",
        "the standard linear solid and its relaxation peak",
    ),
    Reference(
        "baker-blackburn",
        "Baker and Blackburn 2005",
        "G. L. Baker and J. A. Blackburn, <i>The Pendulum: A Case Study in Physics</i>, Oxford University Press, 2005",
        "the pendulum in full: large swings, the physical and torsion pendulums, Kater's, chaos",
    ),
    Reference(
        "duffing",
        "Duffing 1918",
        "G. Duffing, <i>Erzwungene Schwingungen bei veränderlicher Eigenfrequenz und ihre technische "
        "Bedeutung</i>, Sammlung Vieweg, Heft 41/42, Vieweg, Braunschweig, 1918",
        "the oscillator with a cubic spring",
    ),
    Reference(
        "kater",
        "Kater 1818",
        "H. Kater, An account of experiments for determining the length of the pendulum vibrating seconds in "
        "the latitude of London, <i>Philosophical Transactions of the Royal Society of London</i> 108, 33–102, "
        "1818",
        "the reversible pendulum",
    ),
    Reference(
        "duhamel",
        "Duhamel 1833",
        "J.-M. C. Duhamel, Sur la méthode générale relative au mouvement de la chaleur dans les corps solides "
        "plongés dans des milieux dont la température varie avec le temps, <i>Journal de l'École polytechnique</i> "
        "14 (22e cahier), 20–77, 1833",
        "the superposition of responses to a changing input, first for heat",
    ),
    Reference(
        "harris",
        "Piersol and Paez 2010",
        "A. G. Piersol and T. L. Paez (eds), <i>Harris' Shock and Vibration Handbook</i>, 6th edition, "
        "McGraw-Hill, 2010",
        "pulses, shock spectra and their tables, for reference",
    ),
    Reference(
        "ogata",
        "Ogata 2010",
        "K. Ogata, <i>Modern Control Engineering</i>, 5th edition, Prentice Hall, 2010",
        "the Laplace transform and transfer functions, step responses' overshoot and settling",
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
        "l'Institut des Sciences, Lettres et Arts, par divers savans</i> 1, 1806 (read 5 April 1799)",
        "",
    ),
    Reference(
        "wilbraham",
        "Wilbraham 1848",
        "H. Wilbraham, On a certain periodic function, <i>Cambridge and Dublin Mathematical Journal</i> 3, "
        "198–201, 1848",
        "the overshoot, fifty years before Gibbs",
    ),
    Reference(
        "lagrange",
        "Lagrange 1788",
        "J.-L. Lagrange, <i>Méchanique analitique</i>, Veuve Desaint, Paris, 1788",
        "mechanics from the energies, without a figure in it",
    ),
    Reference(
        "rayleigh",
        "Rayleigh 1873",
        "J. W. Strutt (Lord Rayleigh), Some general theorems relating to vibrations, <i>Proceedings of the London "
        "Mathematical Society</i> s1-4, 357–368, 1873",
        "where the dissipation function comes in",
    ),
    Reference(
        "milliken",
        "Milliken and Milliken 2002",
        "W. F. Milliken and D. L. Milliken, <i>Chassis Design: Principles and Analysis</i> (from Maurice Olley's "
        "notes), SAE International, 2002",
        "Olley's own ride work, flat ride among it",
    ),
    Reference(
        "gillespie",
        "Gillespie 2021",
        "T. D. Gillespie, <i>Fundamentals of Vehicle Dynamics</i>, revised edition, SAE International, 2021",
        "a car's ride, bounce and pitch",
    ),
    Reference(
        "shinbrot",
        "Shinbrot et al. 1992",
        "T. Shinbrot, C. Grebogi, J. Wisdom and J. A. Yorke, Chaos in a double pendulum, <i>American Journal of "
        "Physics</i> 60, 491–499, 1992",
        "",
    ),
    Reference(
        "rayleigh-sound",
        "Rayleigh 1877",
        "J. W. Strutt (Lord Rayleigh), <i>The Theory of Sound</i>, vol. 1, Macmillan, 1877 (2nd edition 1894)",
        "the quotient and proportional damping, among much else",
    ),
    Reference(
        "ritz",
        "Ritz 1909",
        "W. Ritz, Über eine neue Methode zur Lösung gewisser Variationsprobleme der mathematischen Physik, "
        "<i>Journal für die reine und angewandte Mathematik</i> 135, 1–61, 1909",
        "",
    ),
    Reference(
        "caughey",
        "Caughey and O'Kelly 1965",
        "T. K. Caughey and M. E. J. O'Kelly, Classical normal modes in damped linear dynamic systems, <i>Journal "
        "of Applied Mechanics</i> 32, 583–588, 1965",
        "",
    ),
    Reference(
        "frahm",
        "Frahm 1911",
        "H. Frahm, Device for damping vibrations of bodies, US patent 989,958, 1911",
        "",
    ),
    Reference(
        "ormondroyd",
        "Ormondroyd and Den Hartog 1928",
        "J. Ormondroyd and J. P. Den Hartog, The theory of the dynamic vibration absorber, <i>Transactions of the "
        "ASME</i> 50, 9–22, 1928",
        "",
    ),
    Reference(
        "brock",
        "Brock 1946",
        "J. E. Brock, A note on the damped vibration absorber, <i>Journal of Applied Mechanics</i> 13, A-284, 1946",
        "",
    ),
    Reference(
        "dunkerley",
        "Dunkerley 1894",
        "S. Dunkerley, On the whirling and vibration of shafts, <i>Philosophical Transactions of the Royal Society "
        "A</i> 185, 279–360, 1894",
        "",
    ),
    Reference(
        "holzer",
        "Holzer 1921",
        "H. Holzer, <i>Die Berechnung der Drehschwingungen und ihre Anwendung im Maschinenbau</i>, Springer, Berlin, "
        "1921",
        "",
    ),
    Reference(
        "poincare",
        "Poincaré 1890",
        "H. Poincaré, Sur les équations aux dérivées partielles de la physique mathématique, <i>American Journal "
        "of Mathematics</i> 12, 211–294, 1890",
        "",
    ),
    Reference(
        "wilson-ritz",
        "Wilson, Yuan and Dickens 1982",
        "E. L. Wilson, M.-W. Yuan and J. M. Dickens, Dynamic analysis by direct superposition of Ritz vectors, "
        "<i>Earthquake Engineering and Structural Dynamics</i> 10, 813–821, 1982",
        "",
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
