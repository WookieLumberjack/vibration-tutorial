"""`gui/chain_page.py`: a page hands a trial's chain to the window, and shows it itself if nobody hands it back."""

from __future__ import annotations

import os

import numpy as np
import pytest

from vib_tutorial.core.model import ChainSystem

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtCore = pytest.importorskip("PySide6.QtCore")

from vib_tutorial.gui import chain_page  # noqa: E402

pytestmark = pytest.mark.gui


def test_a_chain_is_the_same_only_if_every_element_is():
    """`ChainSystem.same_as`: masses, springs, dampers, friction and the static ratio (#409 S-1: pages compared
    one array or three, and took a chain differing elsewhere as already shown)."""
    a = ChainSystem.uniform(4)
    assert a.same_as(a.copy()) and not a.same_as(None) and not a.same_as(ChainSystem.uniform(3))
    for field in ("masses", "stiffness", "damping", "friction"):
        b = a.copy()
        setattr(b, field, getattr(a, field) + np.array([0.0, 0.0, 0.0, 0.5]))
        assert not a.same_as(b), field
    assert not a.same_as(ChainSystem.uniform(4, static_ratio=1.5))
    assert a.same_as(ChainSystem(a.masses * (1 + 1e-12), a.stiffness, a.damping, a.friction)), "to rounding"


class Page(QtCore.QObject):
    system_requested = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.system: ChainSystem | None = ChainSystem.uniform(4)
        self.shown: list[ChainSystem] = []

    def set_system(self, system: ChainSystem) -> None:
        self.system = system
        self.shown.append(system)


def test_a_page_shows_the_chain_itself_only_if_nobody_hands_it_back():
    alone, echoed = Page(), Page()
    echoed.system_requested.connect(echoed.set_system)  # as the main window does
    frictional = ChainSystem.uniform(4, friction=0.2)  # the same masses, springs and dampers
    for page in (alone, echoed):
        chain_page.hand_over(page, frictional)
        assert page.system is frictional and len(page.shown) == 1, "shown once, by the window or by the page"
    chain_page.hand_over(alone, frictional)
    assert len(alone.shown) == 1, "already showing it"
