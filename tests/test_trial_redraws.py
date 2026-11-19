"""A trial draws each curve once (#409 P-7, `gui/coalesce.py`).

Each page's first and last Things to try are picked as the reader picks them (through its `TrialBox`, so in a
`coalesce.batch()`), with every frame loop held still, and each curve's `setData` is counted: more than once is a
view drawn again in the same setup. `BUDGET` is what each page still does at most, the most times one curve was
set; a page not in it sets each once. Lower a page's entry when it's fixed (`@coalesced` on its redraw), and take
it out at 1. The Simulation page isn't counted (`UNCOUNTED`): its trials start the simulation, and what its views
draw while the setup runs depends on the machine's load (2 to 4).
"""

import collections

import pytest

pytestmark = pytest.mark.gui

BUDGET = {  # 2026-11-18, the mechanism in (#409 P-7); the pages' redraws marked next
    "sdof_page": 2,
    "pendulum_page": 3,
    "transient_page": 2,
    "periodic_page": 2,
    "damping_page": 4,
    "isolation_page": 4,
    "energy_page": 2,
    "frf_page": 3,
    "complex_page": 2,
    "beams_page": 2,
    "rigid_body_page": 3,
    "fatigue_page": 2,
    "quarter_car_page": 5,
    "mounts_page": 2,
    "seismic_page": 6,
    "coupling_page": 4,
    "cms_page": 4,
    "frame_page": 3,
    "signals_page": 2,
    "test_page": 6,
    "planning_page": 3,
    "nonlinear_page": 2,
    "variability_page": 2,
    "fluid_page": 2,
    "self_excited_page": 4,
    "active_control_page": 2,
    "gyroscope_page": 2,
    "rotor_builder_page": 4,
}

UNCOUNTED = {"sim_page"}


def test_each_pages_trials_set_each_curve_once(main_window, monkeypatch):
    import pyqtgraph as pg
    from PySide6 import QtWidgets

    from vib_tutorial.gui import frame_loop, lesson, links
    from vib_tutorial.gui.recompute import Recompute

    counts: collections.Counter[int] = collections.Counter()
    counting = [False]
    set_data, set_curve = pg.PlotDataItem.setData, pg.PlotCurveItem.setData

    def data(self, *args, **kwargs):
        if counting[0]:
            counts[id(self)] += 1
        return set_data(self, *args, **kwargs)

    def curve(self, *args, **kwargs):
        if counting[0] and not isinstance(self.parentItem(), pg.PlotDataItem):  # a PlotDataItem's own: counted there
            counts[id(self)] += 1
        return set_curve(self, *args, **kwargs)

    monkeypatch.setattr(pg.PlotDataItem, "setData", data)
    monkeypatch.setattr(pg.PlotCurveItem, "setData", curve)
    monkeypatch.setattr(frame_loop.FrameLoop, "_should_run", lambda self: False)  # no animation while counting
    w = main_window
    app = QtWidgets.QApplication.instance()
    found = {}
    for attr in links.order():
        if attr in UNCOUNTED:
            continue
        page = getattr(w, attr)
        w.pages.setCurrentWidget(page)
        app.processEvents()
        box = page.findChild(lesson.TrialBox)
        if box is None:
            continue
        n = len(box.combo.names)
        most = 0
        for i in sorted({0, n - 1}):
            counts.clear()
            counting[0] = True
            box._run(i)  # as picked, the prediction (if any) answered
            for worker in page.findChildren(Recompute):
                worker.wait()
            app.processEvents()
            counting[0] = False
            most = max(most, max(counts.values(), default=0))
        if most > 1:
            found[attr] = most
    over = {a: (n, BUDGET.get(a, 1)) for a, n in found.items() if n > BUDGET.get(a, 1)}
    assert not over, "a trial sets a curve more often than its page's BUDGET (n, budget): " + repr(over)
