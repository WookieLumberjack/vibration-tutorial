"""Links in the Theory notes, within a page and between pages.

A note links to another page's Theory with `page:<attribute>#<anchor>`, the attribute as in
`main_window.PAGE_GROUPS` (`page:frf_page#three-forms`); `#anchor` alone stays on its own page,
and http(s) links open in the browser. `before(...)` writes a page's "Before this page" line.
The main window calls `wire` on every Theory browser and handles the `page:` links.

A page's notes live in `theory/<name>.html` (`load_notes`), Qt's rich text but for the tags the
loader writes out: `<before>`, the page: links to read first; `<takeaways>`, one `<li>` each; `<eq>` (a
display equation in LaTeX) and `<m>` (inline maths), see `maths`; `<notation>`, the page's table of
symbols, see `notation`; `<cite>` and `<reading>`, its Further reading, see `references`. Figures are
`<img src="fig:…">`, see `figures`. Show the notes in a `notes_view.NotesBrowser`, which draws them.
"""

from __future__ import annotations

import re
import sys
import weakref
from collections.abc import Callable
from importlib import resources

from PySide6 import QtCore, QtGui, QtWidgets


def page_link(page: str, anchor: str, text: str) -> str:
    return f'<a href="page:{page}#{anchor}">{text}</a>'


def before(*refs: tuple[str, str, str]) -> str:
    """The "Before this page" line: (page attribute, anchor, section title) for each section to read first."""
    items = [page_link(page, anchor, title) + f" ({PAGE_TITLES[page]})" for page, anchor, title in refs]
    text = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    return f"<p><i>Before this page: {text}.</i></p>"


def takeaways(*points: str) -> str:
    """A page's closing list of what to remember."""
    return (
        "<h3><a name=\"takeaways\"></a>What to take away</h3><ul>" + "".join(f"<li>{p}</li>" for p in points) + "</ul>"
    )


THEORY = resources.files("vib_tutorial") / "theory"
BEFORE = re.compile(r"<before>(.*?)</before>", re.S)
TAKEAWAYS = re.compile(r"<takeaways>(.*?)</takeaways>", re.S)
LINK = re.compile(r'<a href="page:([a-z_]+)#([^"]+)">(.*?)</a>', re.S)
ITEM = re.compile(r"<li>(.*?)</li>", re.S)
PART = re.compile(r"<!--([A-Z_]+)-->")
EQ = re.compile(r"<eq>(.*?)</eq>", re.S)  # a display equation, in LaTeX (`maths`)

# The keys, as the first page's notes and the About box list them (macOS shows ⌘ for Ctrl: `keys_html`).
KEYS = (
    ("Space", "start, pause or resume the page's motion (on the Simulation page, its force)"),
    ("F1", "the page's Theory tab"),
    ("Ctrl+F1", "the Theory window: any page's notes beside the page, the search and the glossary"),
    ("Ctrl+F2", "the Practice window: problems with new numbers each time"),
    ("Ctrl+1 … Ctrl+6", "a group of pages, at the page it last showed"),
    ("Ctrl+F", "search the notes, in the Theory window"),
    ("Alt+← / Alt+→", "back and forward in the Theory window"),
    ("Ctrl+Z / Ctrl+Shift+Z", "undo and redo, in the Frame and Rotor builders"),
)
MAC_KEYS = (("Ctrl+", "⌘"), ("Alt+← / Alt+→", "⌘[ / ⌘]"))


def keys_html(mac: bool | None = None) -> str:
    """`KEYS` as a table; with ⌘ for Ctrl on macOS (`mac` None: on this platform)."""
    if mac is None:
        mac = sys.platform == "darwin"
    rows = []
    for keys, what in KEYS:
        if mac:
            for pc, apple in reversed(MAC_KEYS):  # the whole Alt pair first: its ⌘ is no Ctrl+
                keys = keys.replace(pc, apple)
        rows.append(f"<tr><td><b>{keys}</b></td><td>{what}</td></tr>")
    return '<table cellspacing="0" cellpadding="3">' + "".join(rows) + "</table>"


def load_notes(name: str, figures: dict[str, Callable[[], str]] | None = None, **parts: str) -> str:
    """The notes in `theory/<name>.html`, its tags written out (see the module's docstring) and each
    `<!--PART-->` filled by the keyword `part` (what the code works out: the trials, the course map). `figures`
    is the figure module's FIGURES the notes show (`figures.register`)."""
    from . import figures as figure_registry
    from . import maths, notation, references

    if figures:
        figure_registry.register(figures)  # the figures these notes show

    html = (THEORY / f"{name}.html").read_text(encoding="utf-8")
    html = EQ.sub(lambda m: maths.display(m.group(1)), html)  # first: its LaTeX is raw, & and < as they are
    html = BEFORE.sub(lambda m: before(*LINK.findall(m.group(1))), html)
    html = TAKEAWAYS.sub(lambda m: takeaways(*ITEM.findall(m.group(1))), html)
    html = notation.NOTATION.sub(lambda m: notation.notation_html(m.group(1).split()), html)
    reading = list(dict.fromkeys(references.CITE.findall(html)))
    for found in references.READING.findall(html):
        reading += [key for key in found.split() if key not in reading]
    html = references.CITE.sub(lambda m: references.cite(m.group(1)), references.READING.sub("", html))
    if reading:
        html = html.rstrip() + "\n" + references.reading_html(reading) + "\n"
    html = maths.markup(html)
    wanted = set(PART.findall(html))
    given = {key.upper() for key in parts}
    if wanted != given:
        raise ValueError(f"{name}.html wants {sorted(wanted)}, given {sorted(given)}")
    return PART.sub(lambda m: parts[m.group(1).lower()], html)


# The pages in the order they're meant to be read, in groups: (group, ((attribute, tab title, what it's
# about), ...)). Each page uses only ideas from the pages before it. The main window's tabs and the
# Single oscillator page's course map are both made from this.
COURSE = (
    (
        "One mass",
        (
            (
                "sdof_page",
                "Single oscillator",
                "one mass on a spring and a damper: free decay, damping, resonance, half-power, beats, poles",
            ),
            (
                "pendulum_page",
                "Pendulums",
                "gravity as the spring, large swings that soften, and the rotational form of every one-mass formula",
            ),
            (
                "transient_page",
                "Transient response",
                "an impulse, a step or a pulse: the impulse response, convolution, and how long a pulse has to be",
            ),
            (
                "periodic_page",
                "Periodic forcing",
                "a repeating force as a Fourier series: harmonics, Gibbs, resonance through a harmonic",
            ),
            (
                "damping_page",
                "Damping models",
                "viscous, hysteretic, friction and viscoelastic damping: their loops, losses and decays",
            ),
            (
                "isolation_page",
                "Vibration isolation",
                "a machine on mounts: transmissibility, static deflection, damping, two stages, start-up",
            ),
        ),
    ),
    (
        "Foundations",
        (
            (
                "deriving_page",
                "Deriving the equations",
                "free-body diagrams against Lagrange's equations: a car in bounce and pitch, and a double pendulum",
            ),
            (
                "sim_page",
                "Simulation && modal analysis",
                "one mass, then a chain of them: modes, damping, energy, forcing, absorbers, friction",
            ),
            (
                "energy_page",
                "Energy methods",
                "Rayleigh, Dunkerley, Rayleigh–Ritz and Holzer: frequencies from the energies, and how good they are",
            ),
            ("frf_page", "FRF matrix", "every receptance H<sub>jk</sub>, and how the modes add up to it"),
            (
                "complex_page",
                "Complex modes",
                "when damping makes the modes complex, and what the usual approximation misses",
            ),
            (
                "continuum_page",
                "Continuous systems",
                "a string, a bar or a beam cut into a chain: which of its modes to trust",
            ),
            (
                "beams_page",
                "Beams && plates",
                "Timoshenko's thick beam, a plate's modes and nodal lines, and modal density",
            ),
            (
                "integration_page",
                "Numerical integration",
                "central difference, Newmark, HHT-α and Runge–Kutta against the exact update: stability, period error, numerical damping",
            ),
        ),
    ),
    (
        "Responses",
        (
            ("random_page", "Random vibration", "a random force or base: the response's spectrum, RMS and peaks"),
            (
                "fatigue_page",
                "Fatigue",
                "a random stress counted by rainflow: S–N curves, Miner's rule, narrow-band and Dirlik estimates",
            ),
            (
                "quarter_car_page",
                "Quarter-car",
                "a suspension on an ISO 8608 road: comfort, grip and travel from the road's PSD",
            ),
            (
                "mounts_page",
                "Rigid body on mounts",
                "a machine on four mounts in 3D: six coupled modes, the elastic centre, and decoupling by tilting them",
            ),
            ("shock_page", "Shock spectrum", "a short pulse on a bank of oscillators, and on the chain"),
            (
                "seismic_page",
                "Seismic response",
                "the chain as a building: a design spectrum, effective modal mass, and ABS, SRSS and CQC against records",
            ),
            (
                "acoustics_page",
                "Vibro-acoustics && SEA",
                "a panel's sound: radiation efficiency, coincidence, the mass law, and energy flow between parts",
            ),
        ),
    ),
    (
        "Building models",
        (
            ("coupling_page", "Modal coupling", "two parts joined, one mode each: when their modes split"),
            ("cms_page", "Substructuring (CMS)", "a chain reduced to a few modes per part, and back"),
            (
                "frame_page",
                "Frame builder",
                "build a frame or truss from beams, bars and springs: assembly, supports, modes",
            ),
        ),
    ),
    (
        "Testing",
        (
            (
                "signals_page",
                "Signals && spectra",
                "sampling, aliasing, leakage, windows and averaging, and the transducers that make the signals",
            ),
            ("test_page", "Virtual modal test", "measuring the FRF from force and response signals, and fitting modes"),
            ("oma_page", "Operational modal analysis", "the modes from the responses alone, the forces unknown"),
            ("updating_page", "Model updating", "comparing a model with a test, and correcting it"),
            ("planning_page", "Test planning", "where to put the sensors, and what the structure does as it runs"),
            (
                "nonlinear_page",
                "Nonlinearity && forces",
                "is the structure linear (levels, sweeps, harmonics), and which force made a response",
            ),
            (
                "variability_page",
                "Variability && localization",
                "scatter in the chain's properties by Monte Carlo, and modes localized in a mistuned ring of blades",
            ),
        ),
    ),
    (
        "Beyond the chain",
        (
            ("torsional_page", "Torsional vibration", "a drive train of discs, shafts and gears, and its start-up"),
            ("duffing_page", "Duffing && Mathieu", "a nonlinear spring: jumps, chaos, and parametric resonance"),
            (
                "fluid_page",
                "Fluid–structure",
                "a body in water: added mass of spheres, cylinders, tubes in pipes and plates; sloshing in a tank "
                "and a tuned liquid damper",
            ),
            (
                "self_excited_page",
                "Self-excited vibration",
                "negative damping and limit cycles: van der Pol, galloping, vortex-induced vibration and flutter",
            ),
            (
                "active_control_page",
                "Active control",
                "velocity feedback, collocation, spillover and actuator lag, against a tuned mass damper",
            ),
            (
                "gyroscope_page",
                "Gyroscope",
                "a spinning top: Euler's equations, steady precession, nutation and the sleeping top",
            ),
            ("rotor_page", "Jeffcott rotor", "a spinning shaft: critical speeds, balancing, bearings, stability, rub"),
            (
                "rotor_builder_page",
                "Rotor builder",
                "build a rotor from shafts, discs, bearings and seals: its whirl modes, criticals and stability",
            ),
            (
                "diagnostics_page",
                "Machinery diagnostics",
                "order tracking under a run-up, gear-mesh sidebands, bearing defects by the envelope, and the cepstrum",
            ),
        ),
    ),
)


def course_map(progress: dict[str, tuple[int, int]] | None = None) -> str:
    """The pages in order, as linked lists by group; with `progress` (practice.progress(): solved, of how many,
    for each page with one solved), how far each page's problems are done."""
    html, start = [], 1
    first = COURSE[0][1][0][0]  # the page the map is on: its link goes past the map, to its first section
    progress = progress or {}

    def done(name: str) -> str:
        if name not in progress:
            return ""
        solved, total = progress[name]
        return (
            f" <i>(✓ all {total} problems solved)</i>" if solved == total else f" <i>({solved} of {total} solved)</i>"
        )

    for group, pages in COURSE:
        items = "".join(
            f'<li><a href="page:{name}{"#equation" if name == first else ""}">{title.replace("&&", "&amp;")}</a>: '
            f"{about}{done(name)}</li>"
            for name, title, about in pages
        )
        html.append(f'<p><b>{group}</b></p><ol start="{start}">{items}</ol>')
        start += len(pages)
    return "".join(html)


# Shorter ways through the course: (name, who it's for, its pages). Each page in a path has every page it builds
# on (its "Before this page", and theirs) in the path too, so a path reads front to back like the course
# (tests/test_links.py checks it).
PATHS = (
    (
        "Essentials",
        "one mass, isolation, a chain's modes and frequency responses, random vibration, and measuring an FRF",
        (
            "sdof_page",
            "pendulum_page",
            "transient_page",
            "periodic_page",
            "damping_page",
            "isolation_page",
            "deriving_page",
            "sim_page",
            "frf_page",
            "random_page",
            "signals_page",
            "test_page",
        ),
    ),
    (
        "Testing",
        "measuring a structure's modes, with and without the force, and correcting a model with them",
        (
            "sdof_page",
            "pendulum_page",
            "transient_page",
            "damping_page",
            "deriving_page",
            "sim_page",
            "frf_page",
            "random_page",
            "signals_page",
            "test_page",
            "oma_page",
            "updating_page",
            "planning_page",
            "nonlinear_page",
            "variability_page",
        ),
    ),
    (
        "Rotating machinery",
        "drive trains, nonlinear springs, rotors on their bearings, and diagnosing a running machine",
        (
            "sdof_page",
            "pendulum_page",
            "transient_page",
            "damping_page",
            "deriving_page",
            "sim_page",
            "frf_page",
            "complex_page",
            "random_page",
            "signals_page",
            "torsional_page",
            "duffing_page",
            "rotor_page",
            "diagnostics_page",
        ),
    ),
    (
        "Vibration environments",
        "random vibration, fatigue, shock and earthquakes, a car on a road and a machine on its mounts",
        (
            "sdof_page",
            "pendulum_page",
            "damping_page",
            "isolation_page",
            "deriving_page",
            "sim_page",
            "frf_page",
            "random_page",
            "fatigue_page",
            "quarter_car_page",
            "mounts_page",
            "shock_page",
            "seismic_page",
        ),
    ),
)


def paths_html() -> str:
    """The paths, each with its pages linked and numbered by their place in the whole course."""
    number = {name: i for i, name in enumerate((n for _, pages in COURSE for n, _, _ in pages), 1)}
    title = {name: t.replace("&&", "&amp;") for _, pages in COURSE for name, t, _ in pages}
    first = COURSE[0][1][0][0]
    html = []
    for name, about, pages in PATHS:
        items = " · ".join(
            f'{number[p]}&nbsp;<a href="page:{p}{"#equation" if p == first else ""}">{title[p]}</a>' for p in pages
        )
        html.append(f"<p><b>{name}</b> ({len(pages)} pages): {about}.<br>{items}</p>")
    return "".join(html)


# Titles as the links name them (the tab titles without Qt's && escape).
PAGE_TITLES = {
    "sdof_page": "Single oscillator page",
    "pendulum_page": "Pendulums page",
    "transient_page": "Transient response page",
    "periodic_page": "Periodic forcing page",
    "damping_page": "Damping models page",
    "isolation_page": "Vibration isolation page",
    "deriving_page": "Deriving the equations page",
    "sim_page": "Simulation page",
    "energy_page": "Energy methods page",
    "frf_page": "FRF matrix page",
    "complex_page": "Complex modes page",
    "planning_page": "Test planning page",
    "nonlinear_page": "Nonlinearity & forces page",
    "continuum_page": "Continuous systems page",
    "beams_page": "Beams & plates page",
    "integration_page": "Numerical integration page",
    "random_page": "Random vibration page",
    "fatigue_page": "Fatigue page",
    "quarter_car_page": "Quarter-car page",
    "mounts_page": "Rigid body on mounts page",
    "shock_page": "Shock spectrum page",
    "seismic_page": "Seismic response page",
    "acoustics_page": "Vibro-acoustics & SEA page",
    "coupling_page": "Modal coupling page",
    "cms_page": "Substructuring page",
    "frame_page": "Frame builder page",
    "signals_page": "Signals & spectra page",
    "test_page": "Virtual modal test page",
    "oma_page": "Operational modal analysis page",
    "updating_page": "Model updating page",
    "variability_page": "Variability & localization page",
    "torsional_page": "Torsional vibration page",
    "duffing_page": "Duffing & Mathieu page",
    "fluid_page": "Fluid–structure page",
    "self_excited_page": "Self-excited vibration page",
    "active_control_page": "Active control page",
    "gyroscope_page": "Gyroscope page",
    "rotor_page": "Jeffcott rotor page",
    "rotor_builder_page": "Rotor builder page",
    "diagnostics_page": "Machinery diagnostics page",
}


def wire(browser: QtWidgets.QTextBrowser, go: Callable[[str, str], None]) -> None:
    """Have `browser` follow its own #anchors, send page: links to `go(page, anchor)`, open the web outside."""
    if browser.property("links_wired"):
        return
    browser.setProperty("links_wired", True)
    browser.setOpenLinks(False)
    # Neither held strongly: the slot lives as long as the browser, and `go` is often its window's method.
    held = weakref.WeakMethod(go) if hasattr(go, "__self__") else (lambda: go)
    ref = weakref.ref(browser)

    def follow(url: QtCore.QUrl) -> None:
        if url.scheme() == "page":
            if (to := held()) is not None:
                to(url.path(), url.fragment())
        elif url.scheme() in ("http", "https"):
            QtGui.QDesktopServices.openUrl(url)
        elif url.fragment() and (shown := ref()) is not None:
            shown.scrollToAnchor(url.fragment())

    browser.anchorClicked.connect(follow)
