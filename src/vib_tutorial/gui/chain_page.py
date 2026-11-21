"""A page that follows the chain hands a trial's chain over to the window, and shows it itself if nobody hands
it back (#409 S-1).

The main window owns the chain (the Simulation page's): a page asks for one with `system_requested`, and the
window sets it on every page that follows the chain, this one included, through `set_system`. A page built
without a window (a test) gets no echo, so `hand_over` shows the chain on the page itself when the page doesn't
have it after asking. Whether it does is `ChainSystem.same_as` (masses, springs, dampers and friction), one test
for every page: some compared only the masses, the springs or the dampers, so a chain that differed elsewhere
was taken as already shown.

Its controls are built here too: `dof_box`, the number of masses (asking the window for a new N), and
`edit_link`, the line that sends the reader to the Simulation page to edit the elements.

Working on it: `tests/test_chain_page.py`; every page's trials run through it in `test_lesson_pages_trials_and_tabs`.
"""

from __future__ import annotations

from typing import Protocol

from PySide6 import QtCore, QtWidgets

from ..core.model import ChainSystem
from .layouts import wrapped
from .style import MAX_DOF

DOF_TIP = "Same chain as the Simulation page; changing it here changes it there too."


class ChainFollower(Protocol):
    system_requested: QtCore.SignalInstance

    @property
    def system(self) -> ChainSystem | None: ...

    def set_system(self, system: ChainSystem) -> None: ...


def dof_box(page: QtCore.QObject, tip: str = DOF_TIP) -> QtWidgets.QSpinBox:
    """The number of masses: changed, it asks the window (`page.dof_requested`) for the chain with that many."""
    box = QtWidgets.QSpinBox()
    box.setRange(1, MAX_DOF)
    box.setToolTip(tip)
    box.valueChanged.connect(page.dof_requested)  # type: ignore[attr-defined]
    return box


def show_count(box: QtWidgets.QSpinBox, n: int) -> None:
    """`box` showing the chain's `n` masses, without asking for it again."""
    box.blockSignals(True)
    box.setValue(n)
    box.blockSignals(False)


def edit_link(page: QtCore.QObject, text: str) -> QtWidgets.QLabel:
    """A muted line whose link (`<a href='#sim'>`) opens the Simulation page's elements (`page.edit_parameters`),
    wrapped: on one line it set its pane's minimum width, leaving the tabs too little on macOS."""
    link = wrapped(text, muted=True)
    link.linkActivated.connect(page.edit_parameters)  # type: ignore[attr-defined]
    return link


def hand_over(page: ChainFollower, system: ChainSystem) -> None:
    """Ask the window for `system`, and show it on `page` if no window handed it back (see the module)."""
    page.system_requested.emit(system)
    if not system.same_as(page.system):
        page.set_system(system)  # nobody passed it back: show it here alone
