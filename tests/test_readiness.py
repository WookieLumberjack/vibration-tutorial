"""#478's readiness checks (page standard item 9): what a page rebuilt to the standard must keep to, so a front end
other than Qt (a Rust UI, WebAssembly) could take it over. A page joins by adding its `Ready` entry to `READY`.

- **Its state, trials and problems are plain data:** its state type (its trial dataclass) survives a JSON round
  trip (`core/plain.py`), as do its trials and each practice problem's question over a few seeds, with its
  `setup` and `design` as the state they set up (the page's own `load_setup`'s). A problem's `make` and a spec's
  `goal` are code, found again by the problem's key.
- **Its view model returns plain data:** each panel's function, on the page's opening state and each trial,
  returns only numbers, strings, numeric arrays and dataclasses of them (`plain.impure`): no Qt, no pyqtgraph,
  colors as theme roles' names.
- **Its notes need no Qt** (`qt_free_notes`): `notes_pages.page_html` gathers them in a fresh interpreter without
  loading Qt (so they live in a Qt-free module, not the page's widget's). The glossary is the first; a page joins
  when its rebuild moves its notes out. The static export itself is `test_notes_export`'s.

Isolation and Duffing & Mathieu, the view-model pilots, joined first, as they were: neither's notes are Qt-free
yet, and Duffing's trials are presets, not its state (both are for their rebuilds). The Single oscillator, #478's
pilot, joined whole: its notes are `sdof_trials`'s (`links.NOTES_MODULES`); Damping models, the second, the same,
and Pendulums, Phase 4's first (`pendulum_trials`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pytest

from vib_tutorial.core.plain import from_plain, impure, to_plain
from vib_tutorial.core.practice import Question

SEEDS = range(5)


@dataclass(frozen=True)
class Ready:
    page: str  # its attribute in links.COURSE
    state: Callable[[], Any]  # the page's opening state (as its Defaults puts it back)
    from_setup: Callable[[dict], Any]  # a problem's setup as the state it sets up (the page's load_setup's)
    views: Callable[[Any], Iterable[Any]]  # each panel's view of a state
    trials: Callable[[], tuple] = tuple  # its Things to try, as states
    problems: Callable[[], tuple] = tuple
    qt_free_notes: bool = False


def _isolation_views(state):
    from vib_tutorial.view_models import isolation as v

    iso = v.isolator(state)
    return (
        v.mount_info(iso),
        v.transmissibility(iso),
        v.design_note(iso, 0.9),
        v.chart(iso),
        v.start_up_view(iso, state.ramp),
        v.let_go_view(iso, 0.005),
    )


def _isolation() -> Ready:
    from vib_tutorial.view_models.isolation import TRIALS, IsolationTrial

    def problems():
        from vib_tutorial.gui.isolation import PROBLEMS

        return PROBLEMS

    return Ready(
        "isolation_page",
        lambda: IsolationTrial("", ""),
        lambda setup: IsolationTrial("", "", **setup),
        _isolation_views,
        lambda: TRIALS,
        problems,
    )


def _duffing_views(state):
    from vib_tutorial.core.duffing import Drive
    from vib_tutorial.view_models import duffing as v

    osc = state.oscillator()
    drive = Drive(state.freq_hz, state.start_hz, state.end_hz, state.sweep_time, state.mode)
    return (
        v.response(osc, v.x_range(drive)),
        v.chart_point(osc, state.freq_hz),
        v.osc_info(osc, None),
        v.branches(osc, state.freq_hz),
        v.steady_states(osc, state.freq_hz),
        v.drive_state(drive, 0.0),
    )


def _duffing() -> Ready:
    from vib_tutorial.view_models.duffing import DuffingSetup

    def problems():
        from vib_tutorial.gui.duffing import PROBLEMS

        return PROBLEMS

    return Ready("duffing_page", DuffingSetup, lambda setup: DuffingSetup(**setup), _duffing_views, tuple, problems)


def _sdof_views(state):
    import numpy as np

    from vib_tutorial.view_models import damping as laws
    from vib_tutorial.view_models import sdof as v
    from vib_tutorial.view_models.recorder import Point

    d = v.drive(state)
    measured = [Point(d.freq_hz, complex(1e-3, -5e-4), v.key(state))]  # a point the recorder landed
    t = np.linspace(0.0, 1.0, 400)
    x = 0.01 * np.sin(2 * np.pi * 3.0 * t)
    return (
        v.info(state),
        v.frf_view(state, measured, measured, True),
        v.frf_view(state, [], [], False),
        v.poles_view(v.viscous(state), state.model),
        laws.loop_view(state, t, x, np.gradient(x, t), 0.4 * np.cos(2 * np.pi * 3.0 * t), 1 / 3.0),
        laws.loss_view(state),
        laws.phasors(state, 0.3),
        laws.decay_readout(state, t, x, False),
    )


def _sdof() -> Ready:
    from vib_tutorial.gui import sdof_trials
    from vib_tutorial.view_models.sdof import SdofTrial

    return Ready(
        "sdof_page",
        lambda: SdofTrial("", ""),
        lambda setup: SdofTrial("", "", **setup),
        _sdof_views,
        lambda: sdof_trials.TRIALS,
        lambda: sdof_trials.PROBLEMS,
        qt_free_notes=True,
    )


def _pendulum_views(state):
    import math

    import numpy as np

    from vib_tutorial.view_models import live_pendulum as lp
    from vib_tutorial.view_models import pendulum as v

    p = v.pendulum(state)
    live = lp.LivePendulum(p)
    live.hold(math.radians(30.0))
    live.let_go()
    live.advance(3.0)
    t, y = live.since_change()
    rec = v.PeriodRecorder()
    rec.update(t, y[:, lp.COL["theta"]], live.changed_at, v.key(p))
    th = y[:, lp.COL["theta"]]
    return (
        v.info(p),
        v.free_body(p, live.theta, live.omega),
        v.free_body(p, 0.3, 0.0, held=True),
        v.period_view(p, rec.current(v.key(p)), rec.earlier(v.key(p)), True),
        v.period_view(p, [], [], False),
        v.phase_view(p, th, y[:, lp.COL["omega"]], v.phase_reach(p, float(np.abs(th).max()))),
        v.moment_view(p, live.theta, 1.0),
        lp.energy_view(t, y),
    )


def _pendulum() -> Ready:
    from vib_tutorial.gui import pendulum_trials
    from vib_tutorial.view_models.pendulum import PendulumTrial

    return Ready(
        "pendulum_page",
        lambda: PendulumTrial("", ""),
        lambda setup: PendulumTrial("", "", **setup),
        _pendulum_views,
        lambda: pendulum_trials.TRIALS,
        lambda: pendulum_trials.PROBLEMS,
        qt_free_notes=True,
    )


def _bases_views(state):
    import numpy as np

    from vib_tutorial.view_models import bases as v
    from vib_tutorial.view_models import live_bases as lb

    s, V = v.system(state), v.basis(state)
    live = lb.LiveBases(s)
    live.hold(np.array([0.04, -0.01]))
    live.let_go()
    live.advance(1.0)
    t, y = live.since_change()
    x, xd = y[:, lb.COL["x1"] : lb.COL["x2"] + 1], y[:, lb.COL["v1"] : lb.COL["v2"] + 1]
    rec = v.RailRecorder()
    rec.update(30.0, 3.3, v.key(s), 1.0)
    return (
        v.plane_view(s, V, state.rail, True),
        v.projection(V, False, x[-1]),
        v.matrices_view(s, V, state.rail),
        v.shares(s, V, state.rail, x, xd),
        v.energy_sum(v.shares(s, V, state.rail, x, xd)),
        v.energy_triangles(s, V, state.rail, x, xd),
        v.rail_view(s, rec.current(v.key(s)), rec.earlier(v.key(s)), True, 30.0),
        lb.energy_view(t, y),
    )


def _bases() -> Ready:
    from vib_tutorial.gui import bases_trials
    from vib_tutorial.view_models.bases import BasesTrial

    return Ready(
        "bases_page",
        lambda: BasesTrial("", ""),
        lambda setup: BasesTrial("", "", **setup),
        _bases_views,
        lambda: bases_trials.TRIALS,
        lambda: bases_trials.PROBLEMS,
        qt_free_notes=True,
    )


def _state_views(state):
    import numpy as np

    from vib_tutorial.view_models import live_bases as lb
    from vib_tutorial.view_models import state_space as v

    s = v.system(state)
    live = lb.LiveBases(s)
    x0, v0 = v.start_state(s, 1)
    live.hold_state(x0, v0)
    live.let_go()
    live.advance(1.0)
    _, y = live.since_change()
    x, xd = y[:, lb.COL["x1"] : lb.COL["x2"] + 1], y[:, lb.COL["v1"] : lb.COL["v2"] + 1]
    own = v.own_of(state)
    T = v.basis_of(s, state.basis, own)
    return (
        v.labels(s, state.basis),
        v.pairs(T, x, xd),
        v.place(T, np.zeros(4), 0, 1.0, 2.0),
        v.x_plane(s),
        v.matrices_view(s, state.basis, own),
        v.sizes(s, x, xd),
        v.shapes_view(s),
        v.own_view(s, own),
        lb.energy_view(*live.since_change()),
    )


def _state() -> Ready:
    from vib_tutorial.gui import state_space_trials
    from vib_tutorial.view_models.state_space import StateTrial

    return Ready(
        "state_page",
        lambda: StateTrial("", ""),
        lambda setup: StateTrial("", "", **setup),
        _state_views,
        lambda: state_space_trials.TRIALS,
        lambda: state_space_trials.PROBLEMS,
        qt_free_notes=True,
    )


READY = {r.page: r for r in (_isolation(), _duffing(), _sdof(), _pendulum(), _bases(), _state())}
QT_FREE_NOTES = ["glossary", *(name for name, r in READY.items() if r.qt_free_notes)]


def _through_json(obj: Any) -> Any:
    return from_plain(type(obj), json.loads(json.dumps(to_plain(obj))))


def _states(ready: Ready) -> list:
    return [ready.state(), *ready.trials()]


@pytest.mark.parametrize("page", READY)
def test_a_ready_pages_state_and_trials_survive_json(page):
    for state in _states(READY[page]):
        assert _through_json(state) == state, getattr(state, "name", state)


@pytest.mark.parametrize("page", READY)
def test_a_ready_pages_problems_survive_json(page):
    ready = READY[page]
    problems = ready.problems()
    assert problems, f"{page} has no practice problems to check"
    for problem in problems:
        for seed in SEEDS:
            q = problem.make(np.random.default_rng(seed))
            data = replace(q, goal=None, setup=None, design=None)  # the goal is code; the setups are states
            assert from_plain(Question, json.loads(json.dumps(to_plain(data)))) == data, problem.key
            for setup in (q.setup, q.design):
                if setup is not None:
                    state = ready.from_setup(setup)
                    assert _through_json(state) == state, problem.key


@pytest.mark.parametrize("page", READY)
def test_a_ready_pages_view_model_returns_plain_data(page):
    ready = READY[page]
    for state in _states(ready):
        for view in ready.views(state):
            assert impure(view) == [], (page, getattr(state, "name", ""), type(view).__name__)


@pytest.mark.parametrize("page", QT_FREE_NOTES)
def test_a_ready_pages_notes_gather_without_qt(page):
    code = (
        "import sys\n"
        "from vib_tutorial.gui.notes_pages import page_html\n"
        f"assert page_html({page!r})\n"
        "print(sorted({m.split('.')[0] for m in sys.modules} & {'PySide6', 'shiboken6', 'pyqtgraph'}))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]", f"{page}'s notes load Qt: they belong in a Qt-free module"
