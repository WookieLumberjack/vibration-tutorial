"""The README and docs/: every relative link and image lands (on a heading, for an anchor), and each course
group's page has a section for each of its pages, in the course's order."""

import re
from pathlib import Path

import pytest

from vib_tutorial.gui.links import COURSE

ROOT = Path(__file__).resolve().parents[1]
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
FENCE = re.compile(r"^```.*?^```", re.S | re.M)


def headings(path: Path) -> list[str]:
    return re.findall(r"^#{1,6} (.+)$", FENCE.sub("", path.read_text(encoding="utf-8")), re.M)


def anchors(path: Path) -> set[str]:
    """GitHub's anchors for the headings: lower case, punctuation dropped, spaces to hyphens, repeats numbered."""
    seen: dict[str, int] = {}
    out = set()
    for title in headings(path):
        a = re.sub(r"[^\w\- ]", "", re.sub(r"<[^>]+>|[`*]", "", title).lower()).replace(" ", "-")
        n = seen.get(a, 0)
        seen[a] = n + 1
        out.add(a if n == 0 else f"{a}-{n}")
    return out


def group_file(group: str) -> Path:
    return ROOT / "docs" / (re.sub(r"[^a-z0-9]+", "-", group.lower()).strip("-") + ".md")


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_links_land(page):
    text = FENCE.sub("", page.read_text(encoding="utf-8"))
    targets = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'src="([^"]+)"', text)
    for target in targets:
        if re.match(r"[a-z]+:", target):  # http(s), mailto
            continue
        file, _, anchor = target.partition("#")
        path = (page.parent / file).resolve() if file else page
        assert path.exists(), f"{page.name} links to {target}"
        if anchor:
            assert anchor in anchors(path), f"{page.name} links to {target}"


@pytest.mark.parametrize("group, pages", COURSE, ids=[g for g, _ in COURSE])
def test_each_group_has_its_pages_in_order(group, pages):
    path = group_file(group)
    sections = re.findall(r"^## (.+)$", FENCE.sub("", path.read_text(encoding="utf-8")), re.M)
    titles = [tab.replace("&&", "&") for _, tab, _ in pages]
    # each page's section is named by its tab's title (Substructuring's spells out CMS)
    assert [t.split(" (")[0] for t in sections] == [t.split(" (")[0] for t in titles], path.name


def test_the_readme_lists_every_page():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for group, pages in COURSE:
        for _, tab, _ in pages:
            assert f"[{tab.replace('&&', '&')}](docs/{group_file(group).name}#" in text, tab


def test_claude_md_indexes_every_page():
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    for _, pages in COURSE:
        for attr, _, _ in pages:
            line = re.search(rf"^  - .*\(`{attr}`\): `gui/(\w+)\.py`", text, re.M)
            assert line, attr
            assert (ROOT / "src/vib_tutorial/gui" / f"{line.group(1)}.py").exists(), attr
