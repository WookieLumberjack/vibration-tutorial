"""Background notes on modal analysis and damping, shown in their own tab."""

from __future__ import annotations

from PySide6 import QtWidgets

from .lesson import trials_html
from .links import load_notes
from .notes_view import NotesBrowser
from .sim_trials import TRIALS

COUPLING_TIP = (
    "<p><b>Coupling index</b> = the largest off-diagonal term of the modal damping "
    "matrix C<sub>m</sub> = Φ<sup>T</sup>CΦ, relative to its diagonal:</p>"
    "<p>&nbsp;&nbsp;max<sub>r≠s</sub> |C<sub>m,rs</sub>| / √(C<sub>m,rr</sub> C<sub>m,ss</sub>)</p>"
    "<p>0 means the damping does not couple the undamped modes (proportional damping). "
    "It can never exceed 1, because C<sub>m</sub> is positive semidefinite. "
    "Values near 1 mean the damping force from one mode's motion drives another mode "
    "about as strongly as it damps its own. How much this matters also depends on how "
    "close the modes' frequencies are. See the Theory tab.</p>"
)

BACKGROUND_HTML = load_notes("background", trials=trials_html(TRIALS))


def make_background_view() -> QtWidgets.QTextBrowser:
    view = NotesBrowser()
    view.setHtml(BACKGROUND_HTML)
    view.setOpenExternalLinks(False)
    return view
