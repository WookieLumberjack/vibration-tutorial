"""Export every page's Theory notes as static HTML and SVG (#478 item 9): `gui/notes_export.py` says how.

    uv run python scripts/export_notes.py OUT [--theme Light]

Writes OUT/index.html (the contents), a file per page and the glossary, and the fonts. Prints which Qt modules
the run loaded: the drawing needs none, but a page's notes still come from its module until it's rebuilt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vib_tutorial.gui import notes_export, style


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="the folder to write the notes to")
    parser.add_argument("--theme", default="Light", choices=sorted(style.THEMES))
    args = parser.parse_args(argv)
    written = notes_export.export(args.out, args.theme)
    print(f"{len(written)} files in {args.out}")
    qt = sorted({m.split(".")[0] for m in sys.modules if m.split(".")[0] in ("PySide6", "shiboken6", "pyqtgraph")})
    print("Qt loaded by the pages' modules: " + (", ".join(qt) if qt else "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
