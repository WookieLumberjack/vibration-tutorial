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
        "rotor_page": rotor_notes.THEORY_HTML,
        "diagnostics_page": diagnostics.NOTES_HTML,
        "variability_page": variability.NOTES_HTML,
        "quarter_car_page": quarter_car.NOTES_HTML,
        "mounts_page": rigid_mounts.NOTES_HTML,
        "seismic_page": seismic.NOTES_HTML,
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
