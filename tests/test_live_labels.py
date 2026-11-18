"""No frame loop rewrites a QLabel every frame (#409 P-2).

A QLabel's `setText` lays the page out again; a text a loop rewrites as it runs is a `LiveLabel`, which repaints
itself alone. Each page that runs is started on each of its tabs and driven a few frames by hand; a QLabel whose
text changed on most of them fails.
"""

import pytest
from PySide6 import QtWidgets

pytestmark = pytest.mark.gui

FRAMES = 8
DT = 1 / 60


def test_no_frame_loop_rewrites_a_qlabel_every_frame(main_window, monkeypatch):
    from vib_tutorial.gui import links, settle_plots
    from vib_tutorial.gui.frame_loop import FrameLoop

    changed: dict[int, tuple[QtWidgets.QLabel, int]] = {}
    set_text = QtWidgets.QLabel.setText

    def counting(self, text):
        if text != self.text():
            changed[id(self)] = (self, changed.get(id(self), (self, 0))[1] + 1)
        set_text(self, text)

    monkeypatch.setattr(QtWidgets.QLabel, "setText", counting)
    w = main_window
    app = QtWidgets.QApplication.instance()
    found = []
    for attr in links.order():
        page = getattr(w, attr)
        w.pages.setCurrentWidget(page)
        tabs = page.findChild(QtWidgets.QTabWidget)
        for k in range(tabs.count() if tabs is not None else 1):
            if tabs is not None:
                tabs.setCurrentIndex(k)
            transport = page.transport() if hasattr(page, "transport") else None
            if transport is not None and not transport.run.isChecked():
                transport.run.click()
            app.processEvents()
            loops = [loop for loop in page.findChildren(FrameLoop) if loop._should_run()]
            for loop in loops:
                loop.enabled = False  # driven here, not by its timer
            changed.clear()
            for _ in range(FRAMES * bool(loops)):
                for loop in loops:
                    loop._call(loop._tick, DT)
                settle_plots(page)
            found += [
                f"{attr} tab {k}: {label.text()[:60]!r}"
                for label, n in changed.values()
                if n >= FRAMES - 2 and label.isVisible()
            ]
            for loop in loops:
                loop.enabled = True
            if transport is not None and transport.run.isChecked():
                transport.run.click()
    assert not found, "QLabels rewritten every frame (make them lesson.LiveLabel):\n" + "\n".join(found)
