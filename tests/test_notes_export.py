"""The notes exported as static HTML and SVG (`gui/notes_export.py`, `scripts/export_notes.py`)."""

import re
import subprocess
import sys
from pathlib import Path

from vib_tutorial.gui import links, notes_export

ROOT = Path(__file__).resolve().parent.parent
LEFT_BEHIND = re.compile(r'src="(?:eq|fig):|href="(?:page|trial|practice):')


def test_a_page_exports_with_its_equations_figures_and_links(tmp_path):
    written = notes_export.export(tmp_path, "Light", ["glossary", "isolation_page"])
    assert [p.name for p in written] == ["index.html", "glossary.html", "isolation_page.html"]
    assert (tmp_path / "fonts" / "STIXTwoText-Regular.otf").exists()
    page = (tmp_path / "isolation_page.html").read_text(encoding="utf-8")
    assert not LEFT_BEHIND.search(page)  # nothing only the app can show or follow
    assert page.count('<span class="eq"') > 5 and page.count('<span class="fig"><svg') >= 2
    assert "Try it" not in page and "Practise it" not in page  # a goal's links set up the running page
    pages = set(links.order())
    targets = re.findall(r'href="([a-z_]+)\.html(?:#[^"]*)?"', page)
    assert "damping_page" in targets and set(targets) <= pages | {"index", "glossary"}
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert all(f'href="{p}.html"' in index for p in pages)


def test_dark_notes_draw_light_on_dark(tmp_path):
    notes_export.export(tmp_path, "Dark", ["glossary"])
    page = (tmp_path / "glossary.html").read_text(encoding="utf-8")
    assert "background: #" in page and "color: #e4e6e9" in page


def test_the_export_draws_without_qt(tmp_path):
    """The glossary's notes need no page module, so exporting them loads no Qt at all: the drawing needs none."""
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from vib_tutorial.gui import notes_export\n"
        f"notes_export.export(Path({str(tmp_path)!r}), 'Light', ['glossary'])\n"
        "print(sorted({m.split('.')[0] for m in sys.modules} & {'PySide6', 'shiboken6', 'pyqtgraph'}))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
    assert (tmp_path / "glossary.html").read_text(encoding="utf-8").count("<svg") == 0  # no maths: just text


def test_the_script_exports_every_page(tmp_path):
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_notes.py"), str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.startswith(f"{len(links.order()) + 2} files")  # the index, the glossary and every page
    for page in links.order():
        assert not LEFT_BEHIND.search((tmp_path / f"{page}.html").read_text(encoding="utf-8")), page
