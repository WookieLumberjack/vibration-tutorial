"""What runs without Qt: the numerics (core/) and the course's data (its order, links, trials, tips).

They're imported in a fresh interpreter and must load neither PySide6, shiboken6 nor pyqtgraph. That keeps
them testable and scriptable, and is the seam a front end other than Qt's would build on (a web build over Pyodide,
a Rust one). The view models have their own check (`test_view_models.py`). A module that only needs Qt for a small
part keeps that part elsewhere: `links` has the course's links and `notes_view.wire` the one function that follows
them in a Qt browser.
"""

import json
import pkgutil
import subprocess
import sys

import vib_tutorial.core as core
import vib_tutorial.gui as gui

CORE = [f"vib_tutorial.core.{m.name}" for m in pkgutil.iter_modules(core.__path__)]

# The course's data in gui/: what a front end needs to know about the pages besides drawing them.
COURSE_DATA = [
    f"vib_tutorial.gui.{name}"
    for name in (
        "links",  # COURSE, PATHS, KEYS, load_notes, page_link
        "references",
        "style",  # the themes' colours
        "sim_trials",
        "frame_builder_trials",
        "gyroscope_trials",
        "rotor_builder_trials",
        "rotor_practice",
        "rotor_tips",
        "deriving_text",
        "trials_text",  # the Things to try as the Theory lists them (#409 S-3)
        "matrix_table",  # the notes' matrices (#409 S-13)
        "cms_notes",
        "coupling_notes",
        "rotor_notes",
        "rotor_readouts",
        "equations_snapshot",
        "figures",  # the notes' figures as SVG (#316 D7), measured by font_metrics
        "font_metrics",
        "maths",  # the notes' maths: inline markup, equations as SVG
    )
] + [  # every page's figures
    f"vib_tutorial.gui.{m.name}" for m in pkgutil.iter_modules(gui.__path__) if m.name.endswith("_figures")
]


def qt_loaded_by(modules: list[str]) -> dict[str, list[str]]:
    """The Qt modules each module brought in, importing them in turn in one fresh interpreter. Once one has
    brought Qt in, the later ones would find it loaded: so only the first culprit is named, and fixing it shows
    the next."""
    code = (
        "import sys, importlib, json\n"
        "out = {}\n"
        f"for name in {modules!r}:\n"
        "    before = set(sys.modules)\n"
        "    importlib.import_module(name)\n"
        "    out[name] = sorted(m for m in set(sys.modules) - before\n"
        "                       if m.split('.')[0] in ('PySide6', 'shiboken6', 'pyqtgraph'))\n"
        "print(json.dumps(out))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    return {name: qt for name, qt in json.loads(out).items() if qt}


def test_core_loads_no_qt():
    assert "vib_tutorial.core.simulator" in CORE and len(CORE) > 50
    assert qt_loaded_by(CORE) == {}


def test_the_course_data_loads_no_qt():
    assert qt_loaded_by(COURSE_DATA) == {}


def test_the_notes_figures_and_equations_draw_without_qt():
    """Every figure as SVG, its labels placed by `font_metrics`, and an equation by ziamath, with no Qt loaded
    (#316 D7): what a browser or a handout would show."""
    figure_modules = [name for name in COURSE_DATA if name.endswith("_figures")]
    code = (
        "import sys, importlib\n"
        "from vib_tutorial.gui import maths\n"
        "n = 0\n"
        f"for name in {figure_modules!r}:\n"
        "    for draw in importlib.import_module(name).FIGURES.values():\n"
        "        assert draw().startswith('<svg'), name\n"
        "        n += 1\n"
        "assert maths.equation_svg(r'\\omega_n = \\sqrt{k/m}', 13.0, '#000000').lstrip().startswith('<svg')\n"
        "qt = [m for m in sys.modules if m.split('.')[0] in ('PySide6', 'shiboken6', 'pyqtgraph')]\n"
        "print(n, len(qt))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert int(out[0]) > 60 and out[1] == "0"
