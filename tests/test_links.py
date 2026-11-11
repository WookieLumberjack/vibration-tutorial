"""The Theory notes' links: every page link lands on a section that exists, and "Before this page" only
sends the reader back to pages earlier in the course."""

import re

import pytest

from vib_tutorial.gui.links import COURSE, PAGE_TITLES


def notes() -> dict[str, str]:
    from vib_tutorial.gui import (
        active_control,
        background,
        beams_plates,
        cms_notes,
        complex_modes,
        continuum,
        coupling_notes,
        damping_models,
        deriving,
        diagnostics,
        duffing,
        energy_methods,
        fatigue,
        fluid_structure,
        frame_builder,
        frf_matrix,
        gyroscope,
        integration,
        isolation,
        modal_test,
        model_updating,
        nonlinear_forces,
        oma,
        pendulum,
        periodic,
        quarter_car,
        random_vibration,
        rigid_mounts,
        rotor_builder,
        rotor_notes,
        sdof,
        seismic,
        self_excited,
        shock,
        signals,
        test_planning,
        torsional,
        transient,
        variability,
        vibroacoustics,
    )

    return {
        "sdof_page": sdof.NOTES_HTML,
        "transient_page": transient.NOTES_HTML,
        "damping_page": damping_models.NOTES_HTML,
        "isolation_page": isolation.NOTES_HTML,
        "deriving_page": deriving.NOTES_HTML,
        "sim_page": background.BACKGROUND_HTML,
        "frf_page": frf_matrix.THEORY_HTML,
        "complex_page": complex_modes.NOTES_HTML,
        "continuum_page": continuum.NOTES_HTML,
        "random_page": random_vibration.NOTES_HTML,
        "shock_page": shock.NOTES_HTML,
        "coupling_page": coupling_notes.THEORY_HTML,
        "cms_page": cms_notes.THEORY_HTML,
        "frame_page": frame_builder.NOTES_HTML,
        "planning_page": test_planning.NOTES_HTML,
        "nonlinear_page": nonlinear_forces.NOTES_HTML,
        "signals_page": signals.NOTES_HTML,
        "test_page": modal_test.THEORY_HTML,
        "oma_page": oma.NOTES_HTML,
        "updating_page": model_updating.NOTES_HTML,
        "torsional_page": torsional.NOTES_HTML,
        "duffing_page": duffing.NOTES_HTML,
        "fluid_page": fluid_structure.NOTES_HTML,
        "rotor_page": rotor_notes.THEORY_HTML,
        "rotor_builder_page": rotor_builder.NOTES_HTML,
        "diagnostics_page": diagnostics.NOTES_HTML,
        "variability_page": variability.NOTES_HTML,
        "quarter_car_page": quarter_car.NOTES_HTML,
        "mounts_page": rigid_mounts.NOTES_HTML,
        "seismic_page": seismic.NOTES_HTML,
        "acoustics_page": vibroacoustics.NOTES_HTML,
        "beams_page": beams_plates.NOTES_HTML,
        "pendulum_page": pendulum.NOTES_HTML,
        "active_control_page": active_control.NOTES_HTML,
        "self_excited_page": self_excited.NOTES_HTML,
        "fatigue_page": fatigue.NOTES_HTML,
        "integration_page": integration.NOTES_HTML,
        "energy_page": energy_methods.NOTES_HTML,
        "periodic_page": periodic.NOTES_HTML,
        "gyroscope_page": gyroscope.NOTES_HTML,
    }


ORDER = [name for _, pages in COURSE for name, _, _ in pages]


def test_the_course_lists_every_page_once():
    assert sorted(ORDER) == sorted(PAGE_TITLES) == sorted(notes())


@pytest.mark.parametrize("page", ORDER)
def test_page_links_land_on_a_section(page):
    html = notes()
    anchors = {p: set(re.findall(r'<a name=["\']([^"\']+)["\']', h)) for p, h in html.items()}
    text = html[page]
    for target, anchor in re.findall(r'href="page:([a-z_]+)(?:#([^"]+))?"', text):
        assert target in html, target
        assert not anchor or anchor in anchors[target], f"{page} links to {target}#{anchor}"
    for anchor in re.findall(r'href="#([^"]+)"', text):
        assert anchor in anchors[page], f"{page} links to its own #{anchor}"


@pytest.mark.parametrize("page", ORDER[1:])
def test_before_this_page_points_back(page):
    text = notes()[page]
    line = re.search(r"<p><i>Before this page: (.*?)</i></p>", text)
    assert line, f"{page} has no Before this page line"
    for target in re.findall(r'href="page:([a-z_]+)', line.group(1)):
        assert ORDER.index(target) < ORDER.index(page), f"{page} asks for {target} first"


@pytest.mark.parametrize("page", ORDER)
def test_every_page_ends_with_its_takeaways(page):
    assert '<a name="takeaways"></a>What to take away' in notes()[page]


def test_notes_files_write_out_their_tags(tmp_path, monkeypatch):
    from vib_tutorial.gui import links

    (tmp_path / "page.html").write_text(
        '<before>\n<a href="page:sdof_page#equation">The equation</a>\n</before>\n<h3>Text</h3>\n<!--TRIALS-->\n'
        "<takeaways>\n<li>One.</li>\n<li>Two,\nover two lines.</li>\n</takeaways>\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(links, "THEORY", tmp_path)
    html = links.load_notes("page", trials="<ul><li>A trial.</li></ul>")
    assert html == (
        links.before(("sdof_page", "equation", "The equation"))
        + "\n<h3>Text</h3>\n<ul><li>A trial.</li></ul>\n"
        + links.takeaways("One.", "Two,\nover two lines.")
        + "\n"
    )
    with pytest.raises(ValueError, match="TRIALS"):
        links.load_notes("page")
    with pytest.raises(ValueError, match="COURSE"):
        links.load_notes("page", trials="", course="")


def test_every_notes_file_is_loaded():
    from pathlib import Path

    from vib_tutorial.gui import links

    gui = Path(links.__file__).parent
    loaded = set(
        re.findall(r'load_notes\(\s*"([a-z_]+)"', "".join(p.read_text(encoding="utf-8") for p in gui.glob("*.py")))
    )
    files = {p.stem for p in Path(str(links.THEORY)).glob("*.html")}
    assert files == loaded


def test_the_glossarys_links_land():
    from vib_tutorial.gui.theory_window import page_html

    html = notes()
    anchors = {p: set(re.findall(r'<a name=["\']([^"\']+)["\']', h)) for p, h in html.items()}
    links_ = re.findall(r'href="page:([a-z_]+)(?:#([^"]+))?"', page_html("glossary"))
    assert len(links_) > 30
    for target, anchor in links_:
        assert target in html, target
        assert not anchor or anchor in anchors[target], f"glossary links to {target}#{anchor}"


# What the notes' markup may hold: Qt's rich-text tags the notes use, and the loader's own.
NOTES_TAGS = {
    "a",
    "b",
    "br",
    "h2",
    "h3",
    "h4",
    "i",
    "img",
    "li",
    "ol",
    "p",
    "sub",
    "sup",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}
NOTES_TAGS |= {"before", "takeaways", "eq", "m", "rm", "notation", "cite", "reading"}
ENTITIES = {"amp", "lt", "gt", "nbsp"}


def test_notes_files_parse_strictly():
    """Every `<` in a notes file opens a known tag or a comment, and every `&` an entity: Qt reads "k < k<sub>x"
    as the start of a tag and drops the text up to the next `>`, on the page and in the search index alike."""
    from vib_tutorial.gui.links import THEORY

    bad = []
    for path in sorted(THEORY.iterdir()):
        if not path.name.endswith(".html"):
            continue
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"<eq>.*?</eq>", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)  # raw LaTeX
        for number, line in enumerate(text.splitlines(), 1):
            for m in re.finditer(r"<(?!!--)(/?)([A-Za-z][A-Za-z0-9]*)?", line):
                if m.group(2) is None or m.group(2).lower() not in NOTES_TAGS:
                    bad.append(f"{path.name}:{number}: {line[m.start() : m.start() + 20]!r}")
            for m in re.finditer(r"&(#?\w*);?", line):
                if not m.group(0).endswith(";") or m.group(1) not in ENTITIES:
                    bad.append(f"{path.name}:{number}: {line[m.start() : m.start() + 20]!r}")
    assert not bad, "write < as &lt; and & as &amp;:\n" + "\n".join(bad)


AHEAD = re.compile(r"\blater\b|\bahead\b|\bnext\b", re.I)


def forward_links() -> list[str]:
    """The links in each page's notes (not its "Before this page") to a page later in the course whose sentence
    doesn't say it's later ("later", "ahead", "next")."""
    from vib_tutorial.gui.links import course_map, paths_html
    from vib_tutorial.gui.theory_window import page_html

    order = [name for _, pages in COURSE for name, *_ in pages]
    found = []
    for i, name in enumerate(order):
        html = page_html(name).replace(course_map(), "").replace(paths_html(), "")  # lists of pages
        text = re.sub(r'<a href="page:(\w+)[^"]*">', "\x00\\1\x00", html)  # each link a marker, then no tags
        text = re.sub(r"<[^>]+>", " ", text).replace("&amp;", "&")
        for m in re.finditer("\x00(\\w+)\x00", text):
            if m.group(1) not in order or order.index(m.group(1)) <= i:
                continue
            start = max(text.rfind(c, 0, m.start()) for c in ".!?(") + 1
            ends = [e for e in (text.find(c, m.end()) for c in ".!?)") if e >= 0]
            sentence = re.sub("\x00\\w+\x00", "", text[start : min(ends, default=len(text))])
            if not AHEAD.search(sentence):
                found.append(f"{PAGE_TITLES[name]} → {PAGE_TITLES[m.group(1)]}: {' '.join(sentence.split())}")
    return found


def test_prose_says_when_a_link_goes_ahead():
    """Each page uses only ideas from the pages before it: a link to a later page is a pointer, and says so
    ("later", "ahead", "next"), so that the reader knows the page doesn't lean on it."""
    found = forward_links()
    assert not found, "\n".join(found)


def test_paths_hold_every_page_they_build_on():
    """Each path has, for each of its pages, every page that one's "Before this page" names (and theirs), once
    each and in the course's order, so it reads front to back like the course."""
    from vib_tutorial.gui.links import PATHS
    from vib_tutorial.gui.theory_window import page_html

    order = [name for _, pages in COURSE for name, *_ in pages]
    needs = {}
    for name in order:
        line = re.search(r"Before this page:(.*?)</p>", page_html(name), re.S)
        needs[name] = set(re.findall(r"page:(\w+)", line.group(1))) - {name} if line else set()
    for title, _, pages in PATHS:
        assert list(pages) == sorted(set(pages), key=order.index), f"{title}: out of order or repeated"
        for page in pages:
            missing = needs[page] - set(pages)
            assert not missing, f"{title}: {page} builds on {sorted(missing)}"
