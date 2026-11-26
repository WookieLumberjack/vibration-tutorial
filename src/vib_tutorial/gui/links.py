"""Links in the Theory notes, within a page and between pages.

A note links to another page's Theory with `page:<attribute>#<anchor>`, the attribute as in
`main_window.PAGE_GROUPS` (`page:frf_page#three-forms`); `#anchor` alone stays on its own page,
and http(s) links open in the browser. `before(...)` writes a page's "Before this page" line.
The main window calls `notes_view.wire` on every Theory browser and handles the `page:` links. This module has
no Qt (`tests/test_view_models.py` checks), so the course's order, paths, keys and notes load headless.

A page's notes live in `theory/<name>.html` (`load_notes`), Qt's rich text but for the tags the
loader writes out: `<before>`, the page: links to read first; `<goals page="…">`, what the page sets out
to teach, an `<li trial="n" problem="key">` each, whose links set up its Thing to try number n and ask its
practice problem `key` (`trial:` and `practice:` links, `notes_view.ACTIONS`); `<recap problems="key …">`, a
part's closing points (an `<li>` each) and links to its problems (`load_notes(..., problems=PROBLEMS)`); `<takeaways>`,
one `<li>` each; `<eq>` (a
display equation in LaTeX) and `<m>` (inline math), see `maths`; `<notation>`, the page's table of
symbols, see `notation`; `<cite>` and `<reading>`, its Further reading, see `references`. Figures are
`<img src="fig:…">`, see `figures`. Show the notes in a `notes_view.NotesBrowser`, which draws them.
"""

from __future__ import annotations

import importlib
import re
import sys
from collections.abc import Callable, Sequence
from importlib import resources
from typing import Any


def page_link(page: str, anchor: str, text: str) -> str:
    return f'<a href="page:{page}#{anchor}">{text}</a>'


def before(*refs: tuple[str, str, str]) -> str:
    """The "Before this page" line: (page attribute, anchor, section title) for each section to read first."""
    items = [page_link(page, anchor, title) + f" ({PAGE_TITLES[page]})" for page, anchor, title in refs]
    text = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    return f"<p><i>Before this page: {text}.</i></p>"


def goals(page: str, *items: tuple[str, str, str]) -> str:
    """A page's opening "On this page you'll" list: (trial number, problem key, text) each, with its links."""
    lines = "".join(
        f'<li>{text.strip()} <a href="trial:{page}#{trial}">Try it</a> · '
        f'<a href="practice:{page}#{problem}">Practice it</a></li>'
        for trial, problem, text in items
    )
    return f'<h3><a name="goals"></a>On this page you\'ll</h3><ul>{lines}</ul>'


def takeaways(*points: str) -> str:
    """A page's closing list of what to remember."""
    return (
        "<h3><a name=\"takeaways\"></a>What to take away</h3><ul>" + "".join(f"<li>{p}</li>" for p in points) + "</ul>"
    )


def recap(points: list[str], problems: list[tuple[str, str, str]]) -> str:
    """A part's closing points and its problems (page, key, title): "From this part", with a link to each."""
    html = "<p><b>From this part</b></p><ul>" + "".join(f"<li>{p}</li>" for p in points) + "</ul>"
    if problems:
        links = " · ".join(f'<a href="practice:{page}#{key}">{title}</a>' for page, key, title in problems)
        html += f"<p><i>Practice it:</i> {links}</p>"
    return html


THEORY = resources.files("vib_tutorial") / "theory"
BEFORE = re.compile(r"<before>(.*?)</before>", re.S)
TAKEAWAYS = re.compile(r"<takeaways>(.*?)</takeaways>", re.S)
RECAP = re.compile(r'<recap problems="([a-z0-9 -]*)">(.*?)</recap>', re.S)
GOALS = re.compile(r'<goals page="([a-z_]+)">(.*?)</goals>', re.S)
GOAL = re.compile(r'<li trial="(\d+)" problem="([a-z0-9-]+)">(.*?)</li>', re.S)
LINK = re.compile(r'<a href="page:([a-z_]+)#([^"]+)">(.*?)</a>', re.S)
ITEM = re.compile(r"<li>(.*?)</li>", re.S)
PART = re.compile(r"<!--([A-Z_]+)-->")
EQ = re.compile(r"<eq>(.*?)</eq>", re.S)  # a display equation, in LaTeX (`maths`)

# The keys, as the first page's notes and the About box list them (macOS shows ⌘ for Ctrl: `keys_html`).
KEYS = (
    ("Space", "start, pause or resume the page's motion (on the Simulation page, its force)"),
    ("F1", "the page's Theory tab"),
    ("Ctrl+R", "the page's controls back as it opened (Defaults)"),
    ("H / Shift+H", "hold the curves on the plots showing, in gray, to compare a change against / clear them"),
    ("F5", "Present: the controls folded away, then the right-hand tabs too (the plots the whole width), then back"),
    ("F6 / F7", "a pointer that follows the mouse / ink: drag to draw on the page (F7 again to use the page)"),
    ("Esc", "the ink cleared, the pointer and ink off, and a Thing to try's large note hidden"),
    ("Ctrl+F1", "the Theory window: any page's notes beside the page, the search and the glossary"),
    ("Ctrl+F2", "the Practice window: problems with new numbers each time"),
    ("Ctrl+1 … Ctrl+9", "a group of pages (the first nine), at the page it last showed"),
    ("Ctrl+PgDown / Ctrl+PgUp", "the next or the previous page in the course"),
    (
        "Alt+PgDown / Alt+PgUp",
        "the page's next or previous Thing to try (◂ ▸ beside them); presenting, its note shows large",
    ),
    ("Ctrl+F", "search every page's notes, in the Theory window (from a page, it opens the window there)"),
    ("Alt+← / Alt+→", "back and forward in the Theory window"),
    (
        "Ctrl+= or Ctrl++ / Ctrl+- / Ctrl+0 / Ctrl+wheel",
        "larger, smaller or the usual text in the notes",
    ),
    ("Ctrl+Z / Ctrl+Shift+Z", "undo and redo, in the Frame and Rotor builders"),
)
MAC_KEYS = (("Alt+", "⌥"), ("Ctrl+", "⌘"), ("Alt+← / Alt+→", "⌘[ / ⌘]"))  # applied last to first: Alt is ⌥ there


def key_text(keys: str, mac: bool | None = None) -> str:
    """`keys` (as `KEYS` writes them) as this platform writes them: ⌘ for Ctrl and ⌥ for Alt on macOS (`mac`
    None: on this platform)."""
    if mac is None:
        mac = sys.platform == "darwin"
    if mac:
        for pc, apple in reversed(MAC_KEYS):  # the whole Alt pair first: its ⌘ is no Ctrl+
            keys = keys.replace(pc, apple)
    return keys


def keys_html(mac: bool | None = None) -> str:
    """`KEYS` as a table, as this platform writes them (`key_text`)."""
    rows = [f"<tr><td><b>{key_text(keys, mac)}</b></td><td>{what}</td></tr>" for keys, what in KEYS]
    return '<table cellspacing="0" cellpadding="3">' + "".join(rows) + "</table>"


def load_notes(
    name: str, figures: dict[str, Callable[[], str]] | None = None, problems: Sequence[Any] = (), **parts: str
) -> str:
    """The notes in `theory/<name>.html`, its tags written out (see the module's docstring) and each
    `<!--PART-->` filled by the keyword `part` (what the code works out: the trials, the course map). `figures`
    is the figure module's FIGURES the notes show (`figures.register`); `problems` the page's practice problems,
    which a `<recap problems="key …">` links by their titles."""
    from . import figures as figure_registry
    from . import maths, notation, references

    if figures:
        figure_registry.register(figures)  # the figures these notes show

    html = (THEORY / f"{name}.html").read_text(encoding="utf-8")
    html = EQ.sub(lambda m: maths.display(m.group(1)), html)  # first: its LaTeX is raw, & and < as they are
    html = BEFORE.sub(lambda m: before(*LINK.findall(m.group(1))), html)
    html = TAKEAWAYS.sub(lambda m: takeaways(*ITEM.findall(m.group(1))), html)
    html = GOALS.sub(lambda m: goals(m.group(1), *GOAL.findall(m.group(2))), html)
    by_key = {p.key: p for p in problems}

    def _recap(m: re.Match[str]) -> str:
        keys = m.group(1).split()
        if missing := [k for k in keys if k not in by_key]:
            raise ValueError(f"{name}.html's recap names problems it wasn't given: {missing}")
        return recap(ITEM.findall(m.group(2)), [(by_key[k].page, k, by_key[k].title) for k in keys])

    html = RECAP.sub(_recap, html)
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
        "One oscillator",
        (
            (
                "sdof_page",
                "Single oscillator",
                "one mass on a spring and a damper: free decay, resonance, half-power, beats, poles, and the "
                "damper's law: viscous, hysteretic, friction, viscoelastic",
            ),
            (
                "pendulum_page",
                "Pendulums",
                "gravity as the spring, large swings that soften, and the rotational form of every one-mass formula",
            ),
            (
                "isolation_page",
                "Vibration isolation",
                "a machine on mounts: transmissibility, static deflection, damping, two stages, start-up",
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
        ),
    ),
    (
        "Equations",
        (
            (
                "deriving_page",
                "Deriving the equations",
                "free-body diagrams against Lagrange's equations: a car in bounce and pitch, and a double pendulum",
            ),
            (
                "bases_page",
                "Coordinates && bases",
                "two masses whose motion is a point in a plane: any basis, the coupling it brings, the modes that "
                "remove it, and Rayleigh's quotient from one arrow",
            ),
            (
                "rigid_body_page",
                "3D rigid body",
                "a body tumbling in space: its inertia tensor, principal axes, Euler's equations and the intermediate-axis flip",
            ),
        ),
    ),
    (
        "Modes",
        (
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
                "mounts_page",
                "Rigid body on mounts",
                "a machine on four mounts in 3D: six coupled modes, the elastic center, and decoupling by tilting them",
            ),
        ),
    ),
    (
        "Building models",
        (
            ("coupling_page", "Modal coupling", "two parts joined, one mode each: when their modes split"),
            ("cms_page", "Substructuring (CMS)", "a chain reduced to a few modes per part, and back"),
            (
                "integration_page",
                "Numerical integration",
                "central difference, Newmark, HHT-α and Runge–Kutta against the exact update: stability, period error, numerical damping",
            ),
        ),
    ),
    (
        "Continuous systems",
        (
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
                "frame_page",
                "Frame builder",
                "build a frame or truss from beams, bars and springs: assembly, supports, modes",
            ),
        ),
    ),
    (
        "Signals and random",
        (
            (
                "signals_page",
                "Signals && spectra",
                "sampling, aliasing, leakage, windows and averaging, and the transducers that make the signals",
            ),
            ("random_page", "Random vibration", "a random force or base: the response's spectrum, RMS and peaks"),
        ),
    ),
    (
        "Rotating machines",
        (
            (
                "gyroscope_page",
                "Gyroscope",
                "a spinning top: Euler's equations, steady precession, nutation and the sleeping top",
            ),
            ("torsional_page", "Torsional vibration", "a drive train of disks, shafts and gears, and its start-up"),
            ("rotor_page", "Jeffcott rotor", "a spinning shaft: critical speeds, balancing, bearings, stability, rub"),
            (
                "rotor_builder_page",
                "Rotor builder",
                "build a rotor from shafts, disks, bearings and seals: its whirl modes, criticals and stability",
            ),
            (
                "diagnostics_page",
                "Machinery diagnostics",
                "order tracking under a run-up, gear-mesh sidebands, bearing defects by the envelope, and the cepstrum",
            ),
        ),
    ),
    (
        "Testing",
        (
            ("test_page", "Virtual modal test", "measuring the FRF from force and response signals, and fitting modes"),
            ("updating_page", "Model updating", "comparing a model with a test, and correcting it"),
            ("oma_page", "Operational modal analysis", "the modes from the responses alone, the forces unknown"),
            ("planning_page", "Test planning", "where to put the sensors, and what the structure does as it runs"),
            (
                "variability_page",
                "Variability && localization",
                "scatter in the chain's properties by Monte Carlo, and modes localized in a mistuned ring of blades",
            ),
        ),
    ),
    (
        "Environments",
        (
            ("shock_page", "Shock spectrum", "a short pulse on a bank of oscillators, and on the chain"),
            (
                "quarter_car_page",
                "Quarter-car",
                "a suspension on an ISO 8608 road: comfort, grip and travel from the road's PSD",
            ),
            (
                "fatigue_page",
                "Fatigue",
                "a random stress counted by rainflow: S–N curves, Miner's rule, narrow-band and Dirlik estimates",
            ),
            (
                "acoustics_page",
                "Vibro-acoustics && SEA",
                "a panel's sound: radiation efficiency, coincidence, the mass law, and energy flow between parts",
            ),
        ),
    ),
    (
        "Nonlinear and flow",
        (
            ("duffing_page", "Duffing && Mathieu", "a nonlinear spring: jumps, chaos, and parametric resonance"),
            (
                "fluid_page",
                "Fluid–structure",
                "a body in water: added mass of spheres, cylinders, tubes in pipes and plates; sloshing in a tank and a tuned liquid damper",
            ),
            (
                "self_excited_page",
                "Self-excited vibration",
                "negative damping and limit cycles: van der Pol, galloping, vortex-induced vibration and flutter",
            ),
        ),
    ),
    (
        "Control",
        (
            (
                "active_control_page",
                "Active control",
                "velocity feedback, collocation, spillover and actuator lag, against a tuned mass damper",
            ),
        ),
    ),
    (
        "Appendix",
        (
            (
                "seismic_page",
                "Seismic response",
                "the chain as a building: a design spectrum, effective modal mass, and ABS, SRSS and CQC against records",
            ),
            (
                "nonlinear_page",
                "Nonlinearity && forces",
                "is the structure linear (levels, sweeps, harmonics), and which force made a response",
            ),
        ),
    ),
)


def course_map(
    progress: dict[str, tuple[int, int]] | None = None, tried: dict[str, tuple[int, int]] | None = None
) -> str:
    """The pages in order, as linked lists by group; with `progress` (practice.progress(): solved, of how many,
    for each page with one solved), how far each page's problems are done, and with `tried` (the same for its
    Things to try: `gui/progress.py`), how many of its trials have been tried."""
    html, start = [], 1
    first = COURSE[0][1][0][0]  # the page the map is on: its link goes past the map, to its first section
    progress, tried = progress or {}, tried or {}

    def done(name: str) -> str:
        parts = []
        if name in progress:
            solved, total = progress[name]
            parts.append(f"✓ all {total} problems solved" if solved == total else f"{solved} of {total} solved")
        if name in tried:
            n, total = tried[name]
            parts.append(f"✓ all {total} tried" if n == total else f"{n} of {total} tried")
        return f" <i>({' · '.join(parts)})</i>" if parts else ""

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


def trials_of(attr: str) -> tuple:
    """Page `attr`'s Things to try (its module imported, the page not built)."""
    if attr == "sim_page":
        from .sim_trials import TRIALS

        return TRIALS
    return getattr(page_module(attr), "TRIALS", ())


def order() -> tuple[str, ...]:
    """Every page's attribute, in the course's order."""
    return tuple(attr for _, pages in COURSE for attr, _, _ in pages)


def title(attr: str) -> str:
    return next(name for _, pages in COURSE for a, name, _ in pages if a == attr)


def next_html(attr: str) -> str:
    """The line closing a page's notes: a link on to the next page in the course (none after the last)."""
    pages = order()
    if attr not in pages or attr == pages[-1]:
        return ""
    after = pages[pages.index(attr) + 1]
    return f'<p align="right"><a href="page:{after}"><b>Next: {title(after)} →</b></a></p>'


PATHS = (
    (
        "Essentials",
        "one mass, isolation, a chain's modes and frequency responses, random vibration, and measuring an FRF",
        (
            "sdof_page",
            "pendulum_page",
            "isolation_page",
            "transient_page",
            "periodic_page",
            "deriving_page",
            "sim_page",
            "frf_page",
            "signals_page",
            "random_page",
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
            "deriving_page",
            "sim_page",
            "frf_page",
            "signals_page",
            "random_page",
            "test_page",
            "updating_page",
            "oma_page",
            "planning_page",
            "variability_page",
            "nonlinear_page",
        ),
    ),
    (
        "Rotating machinery",
        "drive trains, rotors on their bearings, and diagnosing a running machine",
        (
            "sdof_page",
            "pendulum_page",
            "transient_page",
            "deriving_page",
            "sim_page",
            "frf_page",
            "complex_page",
            "signals_page",
            "random_page",
            "torsional_page",
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
            "isolation_page",
            "deriving_page",
            "sim_page",
            "frf_page",
            "mounts_page",
            "random_page",
            "shock_page",
            "quarter_car_page",
            "fatigue_page",
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
    "isolation_page": "Vibration isolation page",
    "transient_page": "Transient response page",
    "periodic_page": "Periodic forcing page",
    "deriving_page": "Deriving the equations page",
    "bases_page": "Coordinates & bases page",
    "rigid_body_page": "3D rigid body page",
    "sim_page": "Simulation page",
    "energy_page": "Energy methods page",
    "frf_page": "FRF matrix page",
    "complex_page": "Complex modes page",
    "mounts_page": "Rigid body on mounts page",
    "continuum_page": "Continuous systems page",
    "beams_page": "Beams & plates page",
    "coupling_page": "Modal coupling page",
    "cms_page": "Substructuring page",
    "integration_page": "Numerical integration page",
    "frame_page": "Frame builder page",
    "signals_page": "Signals & spectra page",
    "random_page": "Random vibration page",
    "test_page": "Virtual modal test page",
    "updating_page": "Model updating page",
    "oma_page": "Operational modal analysis page",
    "shock_page": "Shock spectrum page",
    "quarter_car_page": "Quarter-car page",
    "fatigue_page": "Fatigue page",
    "acoustics_page": "Vibro-acoustics & SEA page",
    "duffing_page": "Duffing & Mathieu page",
    "fluid_page": "Fluid–structure page",
    "self_excited_page": "Self-excited vibration page",
    "gyroscope_page": "Gyroscope page",
    "torsional_page": "Torsional vibration page",
    "rotor_page": "Jeffcott rotor page",
    "rotor_builder_page": "Rotor builder page",
    "diagnostics_page": "Machinery diagnostics page",
    "active_control_page": "Active control page",
    "seismic_page": "Seismic response page",
    "planning_page": "Test planning page",
    "nonlinear_page": "Nonlinearity & forces page",
    "variability_page": "Variability & localization page",
}


# Where each page's class lives: its module in gui/ and the class, by the page's attribute in COURSE (the
# Simulation page is built with the main window). A page's module is imported when the page is first built, or when
# its notes or problems are wanted without building it.
PAGE_CLASSES = {
    "sdof_page": "sdof.SdofPage",
    "pendulum_page": "pendulum.PendulumPage",
    "isolation_page": "isolation.IsolationPage",
    "transient_page": "transient.TransientPage",
    "periodic_page": "periodic.PeriodicPage",
    "deriving_page": "deriving.DerivingPage",
    "bases_page": "bases.BasesPage",
    "rigid_body_page": "rigid_body.RigidBodyPage",
    "energy_page": "energy_methods.EnergyMethodsPage",
    "frf_page": "frf_matrix.FrfMatrixPage",
    "complex_page": "complex_modes.ComplexModesPage",
    "mounts_page": "rigid_mounts.MountsPage",
    "coupling_page": "modal_coupling.ModalCouplingPage",
    "cms_page": "substructuring.SubstructuringPage",
    "integration_page": "integration.IntegrationPage",
    "continuum_page": "continuum.ContinuumPage",
    "beams_page": "beams_plates.BeamsPlatesPage",
    "frame_page": "frame_builder.FramePage",
    "signals_page": "signals.SignalsPage",
    "random_page": "random_vibration.RandomVibrationPage",
    "gyroscope_page": "gyroscope.GyroscopePage",
    "torsional_page": "torsional.TorsionalPage",
    "rotor_page": "rotor.RotorPage",
    "rotor_builder_page": "rotor_builder.RotorBuilderPage",
    "diagnostics_page": "diagnostics.DiagnosticsPage",
    "test_page": "modal_test.ModalTestPage",
    "updating_page": "model_updating.ModelUpdatingPage",
    "oma_page": "oma.OmaPage",
    "planning_page": "test_planning.TestPlanningPage",
    "variability_page": "variability.VariabilityPage",
    "shock_page": "shock.ShockPage",
    "quarter_car_page": "quarter_car.QuarterCarPage",
    "fatigue_page": "fatigue.FatiguePage",
    "acoustics_page": "vibroacoustics.VibroAcousticsPage",
    "duffing_page": "duffing.DuffingPage",
    "fluid_page": "fluid_structure.FluidStructurePage",
    "self_excited_page": "self_excited.SelfExcitedPage",
    "active_control_page": "active_control.ActiveControlPage",
    "seismic_page": "seismic.SeismicPage",
    "nonlinear_page": "nonlinear_forces.NonlinearForcesPage",
}


# A page rebuilt to #478's standard keeps its notes (`NOTES_HTML`) in a Qt-free module, named here, so they gather
# and export without Qt (`notes_pages.page_html`; `tests/test_readiness.py`'s `qt_free_notes`).
NOTES_MODULES = {"sdof_page": "sdof_trials", "pendulum_page": "pendulum_trials", "bases_page": "bases_trials"}


def page_module(attr: str):
    """The module of page `attr` (imported now if it isn't yet: it imports Qt)."""
    return importlib.import_module(f"vib_tutorial.gui.{PAGE_CLASSES[attr].split('.')[0]}")


def page_class(attr: str) -> type:
    """The class of page `attr`, its module imported."""
    return getattr(page_module(attr), PAGE_CLASSES[attr].split(".")[1])
