"""Frame timings on this machine's own display, scenario by scenario, as a report to paste into #316.

    uv run python scripts/mac_timings.py                    # every scenario, ~3 minutes
    uv run python scripts/mac_timings.py --seconds 4 rotor  # just the scenarios whose label has "rotor"
    uv run python scripts/mac_timings.py rub                # the full rub, on its opening tabs and on the Rub tab
    uv run python scripts/mac_timings.py "single oscillator" # the live mass, under each damping law
    uv run python scripts/mac_timings.py --app "dist/Vibration Tutorial.app/Contents/MacOS/vib-tutorial"

It opens the window on the real display (not offscreen: the point is the display's own painting, which
on a Retina Mac is most of a frame), runs each scenario through the pages' own controls, and logs every
frame (`VIB_FRAME_STATS`) to <out>/frames.csv. Each scenario gets a second to settle, then `--seconds` of
frames are counted. <out>/report.md has a line per scenario: frames per second, the computing (work) and
drawing (paint) per frame, median and 95th percentile, how many frames were late, and how fast the
simulation ran against what was asked. Under it, *Where the paint goes* splits each scenario's paint by
plot: the milliseconds a frame each plot took to repaint (its items, held pixmaps and the raster at the
display's scale), the most costly first, and the rest of the window (controls, labels, the 3D views) as
"other". The rotor page's opening is timed too, and with `--app` the packaged app's start-up, cold then
warm, as the release workflow measures it.

Each scenario's legend placements are timed too (*Legend placement*: they run on their own timer between
frames, so they show in neither column). *Held curves* counts how often a held curve (`plot_patches.hold_still`)
was stroked again a frame, and the ms that took: about 0 when its pixmap is kept, one a frame when Qt's pixmap
cache drops it (#409 N-2). The Single oscillator runs a second time with its curves not held, to show what holding
gains on this display.

Leave the window alone while it runs: a click or a key press is logged as an input but changes the run.
`scripts/frame_report.py <out>/frames.csv` cuts the same log by what was on screen instead. To share a
run, paste <out>/report.md into the issue as it is (the CSV only if asked: it's large).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vib_tutorial import ONE_THREAD

# BLAS on one thread, as the app runs it (vib_tutorial.main), set before NumPy loads: on OpenBLAS's pool the
# cracked shaft's waterfall blocks took 1.2 s each, not 15 ms, which the app never sees.
os.environ.update({name: os.environ.get(name, "1") for name in ONE_THREAD})

import numpy as np

MARK = "mac_timings: "  # the input rows that start and end a scenario in the log
SETTLE_S = 1.0


@dataclass
class Scenario:
    label: str
    note: str  # what it measures (which #130 or #83 item)
    setup: Callable  # (window) -> (page, asked speed × real time), started and showing
    seconds: float | None = None  # its own length, where shorter than --seconds (a start-up that ends)


def _sim(w, **trial):
    from vib_tutorial.gui.sim_trials import SimTrial

    w.pages.setCurrentWidget(w.sim_page)
    w.sim_page.apply_trial(SimTrial("timing", "", **trial))
    return w.sim_page, w.sim_page.controls.speed_factor


def _stiff(friction: float):
    def setup(w):
        return _sim(
            w,
            masses=(1.0,) * 8,
            springs=(1e8,) * 8,
            dampers=(0.5,) * 8,
            friction=friction,
            freq_hz=50.0,
            force=100.0,
            apply=True,
        )

    return setup


def _sim_trial(w):
    from vib_tutorial.gui.sim_trials import TRIALS

    w.pages.setCurrentWidget(w.sim_page)
    w.sim_page.apply_trial(TRIALS[0])
    return w.sim_page, w.sim_page.controls.speed_factor


def _rotor(preset: str, rpm: float | None = None, speed: float = 1.0, tab: str | None = None):
    """A rotor preset run at `speed`, at `rpm` if given; on its own tabs, or on the page's `tab` (an attribute)."""

    def setup(w):
        from vib_tutorial.gui import rotor

        page = w.rotor_page
        w.pages.setCurrentWidget(page)
        page.load_preset(preset)
        if tab is not None:
            page.tabs.show_page(getattr(page, tab))
        page.sim_speed.setCurrentIndex(rotor.SPEEDS.index(speed))
        if rpm is not None:
            page.ramp.setValue(page.ramp.maximum())
            page.set_target_rpm(rpm)
        page._run(True)
        return page, speed

    return setup


def _duffing(preset: str, speed: float):
    def setup(w):
        from vib_tutorial.gui import duffing

        page = w.duffing_page
        w.pages.setCurrentWidget(page)
        page.load_preset(preset)
        page.sim_speed.setCurrentIndex(duffing.SPEEDS.index(speed))
        page._run(True)
        return page, speed

    return setup


def _startup(preset: str, speed: float = 1.0):
    def setup(w):
        from vib_tutorial.gui import torsional_startup_view

        page = w.torsional_page
        w.pages.setCurrentWidget(page)
        page.load_preset(preset)
        page.startup_view.playback.setCurrentIndex(torsional_startup_view.PLAYBACK.index(speed))
        page.start()
        return page, speed

    return setup


_UNHELD: list = []  # the curves a not-held scenario let go of, held again when it stops


def _cursor_curves(page) -> list:
    """The curves under `page`'s playback cursors: the ones `Playback.add_cursor` held."""
    import pyqtgraph as pg

    curves = []
    for line in page.playback.cursors:
        for item in line.getViewBox().addedItems:
            curve = item.curve if isinstance(item, pg.PlotDataItem) else item
            if isinstance(curve, pg.PlotCurveItem):
                curves.append(curve)
    return curves


def _playback(attribute: str, view: str | None = None, held: bool = True):
    """A small page's closed-form response played back (its `Playback`), at the page's own speed; `view`, the
    page's attribute its playback draws, brought to the front if it's in a tab; `held` False draws the curves
    under its cursors afresh each frame, as before `hold_still`."""

    def setup(w):
        from PySide6 import QtWidgets

        from vib_tutorial.gui import playback

        page = getattr(w, attribute)
        w.pages.setCurrentWidget(page)
        if view is not None:
            for tabs in page.findChildren(QtWidgets.QTabWidget):
                for i in range(tabs.count()):
                    if tabs.widget(i).isAncestorOf(getattr(page, view)):
                        tabs.setCurrentIndex(i)
        if not held:
            for curve in _cursor_curves(page):
                curve.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.NoCache)
                _UNHELD.append(curve)
        page.playback.play()
        return page, playback.SPEEDS[page.playback.speed.currentIndex()][1]

    return setup


def _live_sdof(trial: str):
    """The Single oscillator's trial `trial`, live (its frame loop steps the mass), at 1×."""

    def setup(w):
        from vib_tutorial.gui import sdof

        page = w.sdof_page
        w.pages.setCurrentWidget(page)
        page.load_trial(next(i for i, t in enumerate(sdof.TRIALS) if t.name.startswith(trial)))
        return page, 1.0

    return setup


def _waves(preset: str):
    """A wave preset on the Continuous systems page's Waves tab (it starts the pulse); its speed is the page's."""

    def setup(w):
        page = w.continuum_page
        w.pages.setCurrentWidget(page)
        page.load_preset(preset)
        return page, float("nan")

    return setup


def stop(w, page) -> None:
    """Stop the scenario's motion, whichever kind the page has."""
    from vib_tutorial.gui.playback import Playback

    if isinstance(getattr(page, "playback", None), Playback):
        page.playback.pause()
        from PySide6 import QtWidgets

        while _UNHELD:
            _UNHELD.pop().setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)
    elif page is w.continuum_page:
        page.waves_view.reset()
    elif hasattr(page, "_run"):
        page._run(False)
    elif page is w.sim_page and w.sim_page.force.active:
        w.sim_page.force.switch_off()


def _rigid_body(tab: str):
    def setup(w):
        from vib_tutorial.gui import rigid_body

        page = w.rigid_body_page
        w.pages.setCurrentWidget(page)
        page.trials.combo.pick(rigid_body.TRIALS[0].name)  # starts it
        page.tabs.setCurrentIndex(getattr(rigid_body, tab))  # a TAB_ name
        if not page.running:
            page.pause_button.click()
        return page, rigid_body.PLAYBACK[page.playback.currentIndex()]

    return setup


def _gyroscope(tab: str):
    def setup(w):
        from vib_tutorial.gui import gyroscope

        page = w.gyroscope_page
        w.pages.setCurrentWidget(page)
        page.trials.combo.pick(gyroscope.TRIALS[0].name)  # starts it
        page.tabs.setCurrentIndex(getattr(gyroscope, tab))  # a TAB_ name
        if not page.running:
            page.pause_button.click()
        return page, gyroscope.PLAYBACK[page.playback.currentIndex()]

    return setup


def _sdof_law(model: str, ghost: bool = True, tab: str = "TAB_ENERGY", trial: str = "Ringing down"):
    """The Single oscillator's `trial` under damping law `model` (a `Model` name), the viscous ghost on or off,
    on tab `tab` (a `view_models.sdof` TAB_ name), at 1× (#478: hysteretic reported slow on the Mac)."""

    def setup(w):
        from vib_tutorial.core.damping_models import MODELS, Model
        from vib_tutorial.gui import sdof
        from vib_tutorial.view_models import sdof as view

        page = w.sdof_page
        w.pages.setCurrentWidget(page)
        page.load_trial(next(i for i, t in enumerate(sdof.TRIALS) if t.name.startswith(trial)))
        page.model.setCurrentIndex(MODELS.index(Model[model]))  # a live change: the run carries on
        page.ghost_on.setChecked(ghost)
        page.tabs.setCurrentIndex(getattr(view, tab))
        return page, 1.0

    return setup


def _live_bases(trial: str, tab: str):
    """Coordinates & bases' trial `trial`, live (its frame loop steps the masses), on tab `tab` (a
    `view_models.bases` TAB_ name), at 1× (#478 Phase 4)."""

    def setup(w):
        from vib_tutorial.gui import bases
        from vib_tutorial.view_models import bases as view

        page = w.bases_page
        w.pages.setCurrentWidget(page)
        page.load_trial(next(i for i, t in enumerate(bases.TRIALS) if t.name.startswith(trial)))
        page.tabs.setCurrentIndex(getattr(view, tab))
        return page, 1.0

    return setup


def _turning_arrows(trial: str, tab: str):
    """`_live_bases`, with v₁ held and turned the whole time as a drag turns it (the page's own `_on_arrow_moved`,
    about once a frame), so the plane and the coordinates' whole window are described again each step."""
    base = _live_bases(trial, tab)

    def setup(w):
        from PySide6 import QtCore

        from vib_tutorial.view_models.bases import ARROW

        page, speed = base(w)
        timer = QtCore.QTimer(page)  # the page's: it stops with the window
        turn = {"deg": 0.0}

        def step() -> None:
            turn["deg"] = (turn["deg"] + 0.5) % 90.0  # 0° to 90° and round again, clear of v₂ at 135°
            a = math.radians(turn["deg"])
            page._on_arrow_moved(0, ARROW * math.cos(a), ARROW * math.sin(a))

        timer.timeout.connect(step)
        timer.start(16)
        page.plane.held = 0  # v₁ held, as by a hand: the page draws what a drag draws
        return page, speed

    return setup


def _live_pendulum(trial: str, tab: str):
    """Pendulums' trial `trial`, live (its frame loop steps the pendulum), on tab `tab` (a `view_models.pendulum`
    TAB_ name), at 1× (#478 Phase 4)."""

    def setup(w):
        from vib_tutorial.gui import pendulum
        from vib_tutorial.view_models import pendulum as view

        page = w.pendulum_page
        w.pages.setCurrentWidget(page)
        page.load_trial(next(i for i, t in enumerate(pendulum.TRIALS) if t.name.startswith(trial)))
        page.tabs.setCurrentIndex(getattr(view, tab))
        return page, 1.0

    return setup


SCENARIOS = (
    Scenario("Simulation: 4 masses, resonance building up", "baseline: the first trial", _sim_trial),
    Scenario("Simulation: 8 masses at 1e8 N/m, 50 Hz", "#202 linear steps in Rust, #203 strip chart", _stiff(0.0)),
    Scenario(
        "Simulation: the same with 0.5 N friction", "#92 budget, #95 friction kernel, #96 off thread", _stiff(0.5)
    ),
    Scenario("Rotor: Classic Jeffcott at 1400 rpm, 1×", "rotor paint (#83)", _rotor("Classic Jeffcott rotor", 1400.0)),
    Scenario(
        "Rotor: two-pole shaft at 3000 rpm, 1×",
        "split-step loop (#130, low priority)",
        _rotor("Two-pole rotor (asymmetric shaft)", 3000.0),
    ),
    Scenario(
        "Rotor: cracked shaft at 3000 rpm, 1×",
        "split-step loop (#130, low priority)",
        _rotor("Cracked shaft (breathing crack)", 3000.0),
    ),
    Scenario("Rotor: oil whirl (vertical rotor), 1×", "#94 film kernel", _rotor("Oil whirl (vertical rotor)")),
    Scenario(
        "Rotor: full rub, 1×", "rub kernel; its paint, 12 ms on the Mac (#316 E)", _rotor("Full rub (a tight seal)")
    ),
    Scenario(
        "Rotor: full rub, Rub tab, 1×",
        "the ring's push drawn every frame too (#316 E)",
        _rotor("Full rub (a tight seal)", tab="rub_panel"),
    ),
    Scenario(
        "Duffing: hardening spring, 50×",
        "FOH + Newton stepper (#130, low priority)",
        _duffing("Hardening spring: the jump", 50.0),
    ),
    Scenario(
        "Torsional: geared compressor start-up, 1×",
        "FOH + Newton stepper (#130, low priority)",
        _startup("Geared compressor"),
        seconds=2.5,
    ),
    Scenario(
        "Single oscillator: live, a sweep building its FRF",
        "the live mass, its free-body diagram and the recorder (#478)",
        _live_sdof("Building the frequency response"),
    ),
    *(
        Scenario(
            f"Single oscillator: ringing down, {law.lower()}{'' if ghost else ', no ghost'}, Energy tab",
            "a damping law's run beside its viscous ghost (#478: hysteretic reported slow on the Mac)",
            _sdof_law(law, ghost),
        )
        for law, ghost in (
            ("VISCOUS", True),
            ("HYSTERETIC", True),
            ("HYSTERETIC", False),
            ("COULOMB", True),
            ("ZENER", True),
        )
    ),
    Scenario(
        "Single oscillator: ringing down, hysteretic, Loop & loss tab",
        "the loop redrawn each frame beside each law's (#478)",
        _sdof_law("HYSTERETIC", tab="TAB_LOOP"),
    ),
    Scenario(
        "Single oscillator: driven sweep, hysteretic, Frequency response tab",
        "its c following the drive's frequency (#478)",
        _sdof_law("HYSTERETIC", tab="TAB_FRF", trial="Building the frequency response"),
    ),
    *(
        Scenario(
            f"Pendulums: nearly over the top, dying away, {name} tab",
            "the live pendulum, its moments and the tab's live plot (#478 Phase 4)",
            _live_pendulum("Nearly over the top", tab),
        )
        for name, tab in (("Period", "TAB_PERIOD"), ("Phase plane", "TAB_PHASE"), ("Energy", "TAB_ENERGY"))
    ),
    Scenario(
        "Coordinates & bases: one mass plucked, Coupling tab",
        "the two masses, the plane's path and the coordinates redescribed every frame (#478 Phase 4)",
        _live_bases("Pluck one mass", "TAB_COUPLING"),
    ),
    Scenario(
        "Coordinates & bases: one mass plucked, v₁ turning, Coupling tab",
        "an arrow dragged as it runs: the plane and the whole window redescribed each frame (#478 Phase 4)",
        _turning_arrows("Pluck one mass", "TAB_COUPLING"),
    ),
    Scenario(
        "Coordinates & bases: one arrow along x₁, One arrow tab",
        "one arrow, a reduced basis, its frequency measured as it runs (#478 Phase 4)",
        _live_bases("One arrow along", "TAB_RAIL"),
    ),
    Scenario(
        "Quarter-car: driven over its road",
        "the car's view and its held plots",
        _playback("quarter_car_page", "car_view"),
    ),
    Scenario(
        "Continuous systems: a pulse on 40 beads", "the Waves tab's frame loop", _waves("Waves: a pulse on 40 beads")
    ),
    Scenario("Gyroscope: the first trial, Potential tab", "baseline for the Check tab", _gyroscope("TAB_POTENTIAL")),
    Scenario(
        "Gyroscope: the first trial, Check tab", "the Check plot, reported slow on the Mac", _gyroscope("TAB_CHECK")
    ),
    Scenario(
        "Gyroscope: the first trial, Components tab",
        "its legend kept between frames (#409 P-4: 44 fps)",
        _gyroscope("TAB_COMPONENTS"),
    ),
    Scenario(
        "3D rigid body: the first trial, Components tab",
        "its legend kept between frames (#409 P-4: 38 fps)",
        _rigid_body("TAB_COMPONENTS"),
    ),
)


class PaintSplit:
    """Each plot's repaint time, while `on`: pyqtgraph's views (every plot) timed round their paintEvent."""

    def __init__(self) -> None:
        from pyqtgraph.widgets.GraphicsView import GraphicsView

        self.on = False
        self.ms: dict[str, float] = {}
        self._names: dict[int, str] = {}
        self._cls = GraphicsView
        self._paint = GraphicsView.paintEvent
        split = self

        def paintEvent(view, event):
            if not split.on:
                return split._paint(view, event)
            began = time.perf_counter()
            try:
                return split._paint(view, event)
            finally:
                key = id(view)
                name = split._names.get(key)
                if name is None:
                    name = split._names[key] = _view_name(view)
                split.ms[name] = split.ms.get(name, 0.0) + (time.perf_counter() - began) * 1e3

        GraphicsView.paintEvent = paintEvent

    def take(self) -> dict[str, float]:
        """The times gathered since the last take, and a fresh start (a new page may reuse an id)."""
        ms, self.ms, self._names = self.ms, {}, {}
        return ms

    def uninstall(self) -> None:
        self._cls.paintEvent = self._paint


class CurveStrokes:
    """Each curve's strokes while `on` (`PlotCurveItem.paint`), held ones (a pixmap kept, `DeviceCoordinateCache`)
    apart: Qt calls a held curve's paint only to draw its pixmap again. Installed before the window is built."""

    def __init__(self) -> None:
        import pyqtgraph as pg
        from PySide6 import QtWidgets

        self.on = False
        self.held = [0, 0.0]  # strokes, ms
        self.free = [0, 0.0]
        self._cls = pg.PlotCurveItem
        self._paint = pg.PlotCurveItem.paint
        no_cache = QtWidgets.QGraphicsItem.CacheMode.NoCache
        strokes = self

        def paint(curve, *args):
            if not strokes.on:
                return strokes._paint(curve, *args)
            began = time.perf_counter()
            try:
                return strokes._paint(curve, *args)
            finally:
                tally = strokes.free if curve.cacheMode() == no_cache else strokes.held
                tally[0] += 1
                tally[1] += (time.perf_counter() - began) * 1e3

        pg.PlotCurveItem.paint = paint

    def take(self) -> tuple[list, list]:
        taken, self.held, self.free = (self.held, self.free), [0, 0.0], [0, 0.0]
        return taken

    def uninstall(self) -> None:
        self._cls.paint = self._paint


def stroke_row(label: str, held: list, free: list, frames: int) -> str:
    """A scenario's curve strokes a frame, held and not, with their ms a frame."""
    n = max(frames, 1)
    return f"| {label} | {held[0] / n:.1f} | {held[1] / n:.1f} | {free[0] / n:.1f} | {free[1] / n:.1f} |"


class PlaceTimer:
    """Each legend placement's time (`legend_place.Placer.place`, on its own timer between frames), while `on`.
    Installed before the window is built: a placer's timer holds the method it was given then."""

    def __init__(self) -> None:
        from vib_tutorial.gui import legend_place

        self.on = False
        self.ms: list[float] = []
        self._cls = legend_place.Placer
        self._place = legend_place.Placer.place
        timer = self

        def place(placer):
            if not timer.on:
                return timer._place(placer)
            began = time.perf_counter()
            try:
                return timer._place(placer)
            finally:
                timer.ms.append((time.perf_counter() - began) * 1e3)

        legend_place.Placer.place = place

    def take(self) -> list[float]:
        ms, self.ms = self.ms, []
        return ms

    def uninstall(self) -> None:
        self._cls.place = self._place


def _view_name(view) -> str:
    """A plot's title, its axes' labels, or its accessible name, for the paint split."""
    from vib_tutorial.gui import frame_stats

    name = frame_stats._plot_name(view).removeprefix("plot ")
    if name == type(view).__name__ and view.accessibleName():
        name = view.accessibleName()
    if name == type(view).__name__:
        for item in view.scene().items() if view.scene() is not None else ():
            label = getattr(item, "labelText", "")
            if label:
                return frame_stats._plain(label)[:60]
    return name


def split_rows(label: str, ms: dict[str, float], frames: list[dict[str, str]], top: int = 6) -> list[str]:
    """Where a scenario's paint went, a frame's ms by plot, the costliest `top` and the rest as "other"."""
    n = len(frames)
    if n == 0:
        return []
    paint = sum(float(f["paint_ms"]) for f in frames) / n
    ranked = sorted(ms.items(), key=lambda kv: -kv[1])
    cells = [f"{name} {total / n:.1f}" for name, total in ranked[:top]]
    rest = sum(total for _, total in ranked[top:]) / n
    if rest >= 0.05:
        cells.append(f"{len(ranked) - top} more plots {rest:.1f}")
    other = paint - sum(ms.values()) / n
    cells.append(f"other {max(other, 0.0):.1f}")
    return [f"| {label} | {paint:.1f} | " + "; ".join(cells) + " |"]


def wait(app, seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.001)


def rotor_opening(app, w) -> str:
    """How long the rotor page takes to build, to show, and to show its analysis (#204)."""
    began = time.perf_counter()
    page = w.rotor_page
    built = time.perf_counter() - began
    w.pages.setCurrentWidget(page)
    app.processEvents()
    shown = time.perf_counter() - began
    while page.campbell is None and time.perf_counter() - began < 30.0:
        app.processEvents()
        time.sleep(0.001)
    analysed = time.perf_counter() - began
    return f"built {built * 1e3:.0f} ms, shown {shown * 1e3:.0f} ms, its analysis on screen {analysed * 1e3:.0f} ms"


def startup(app_path: str, out: Path) -> list[str]:
    """The packaged app's start-up, cold then warm, as the release workflow's smoke test records it."""
    rows = [
        "| Run | To first window (s) | Since Python start (s) | MainWindow built (s) |",
        "| --- | --- | --- | --- |",
    ]
    for run in ("cold", "warm"):
        path = out / f"startup-{run}.json"
        path.unlink(missing_ok=True)
        env = dict(os.environ, VIB_TUTORIAL_STARTUP_FILE=str(path), VIB_TUTORIAL_LAUNCHED_AT=f"{time.time():.3f}")
        env.pop("VIB_FRAME_STATS", None)
        subprocess.run([app_path, "--smoke-test"], env=env, timeout=120, check=False)
        if not path.exists():
            rows.append(f"| {run} | (no window painted) | | |")
            continue
        t = json.loads(path.read_text())
        rows.append(f"| {run} | {t.get('launch', float('nan')):.2f} | {t['python']:.2f} | {t['window']:.2f} |")
    return rows


def summarize(frames: list[dict[str, str]], asked: float) -> str:
    if len(frames) < 2:
        return f"| {len(frames)} frames | | | | | |"
    iv = np.array([float(f["interval_ms"]) for f in frames])
    iv = iv[np.isfinite(iv)]
    work = np.array([float(f["work_ms"]) for f in frames])
    paint = np.array([float(f["paint_ms"]) for f in frames])
    fps = 1000.0 / iv.mean() if iv.size else float("nan")
    late = float(np.mean(iv > 25.0)) if iv.size else 0.0
    sims = [(float(f["wall_s"]), float(f["sim_t"])) for f in frames if f["sim_t"]]
    speed = ""
    if len(sims) > 1 and sims[-1][0] > sims[0][0]:
        ran = (sims[-1][1] - sims[0][1]) / (sims[-1][0] - sims[0][0])
        speed = f"{ran:.2f}× of {asked:g}×" if math.isfinite(asked) else f"{ran:.3g}×"
    return (
        f"| {fps:.1f} | {np.median(work):.1f} / {np.percentile(work, 95):.1f} "
        f"| {np.median(paint):.1f} / {np.percentile(paint, 95):.1f} | {late:.0%} | {speed} |"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("only", nargs="*", help="run only the scenarios whose label has one of these (any case)")
    parser.add_argument("--seconds", type=float, default=8.0, help="s of frames counted per scenario")
    parser.add_argument("--out", default="timings", help="folder for frames.csv and report.md")
    parser.add_argument("--theme", default="Light")
    parser.add_argument("--size", default="1440x900", help="the window's size in logical pixels (fits the screen)")
    parser.add_argument("--app", help="a packaged app's executable: time its start-up, cold then warm")
    parser.add_argument("--offscreen", action="store_true", help="no display (to try the script, not to measure)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "frames.csv"
    os.environ["VIB_FRAME_STATS"] = str(csv_path)  # read when the window is built
    if args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from PySide6 import QtGui

    from vib_tutorial import gui
    from vib_tutorial.gui import frame_stats
    from vib_tutorial.gui.main_window import MainWindow

    app = gui.make_app(args.theme)
    split = PaintSplit()  # before any plot is built: PySide looks up a view's paintEvent when it's made
    placing = PlaceTimer()
    strokes = CurveStrokes()
    w = MainWindow()
    width, height = (int(x) for x in args.size.split("x"))
    screen = app.primaryScreen()
    room = screen.availableGeometry()
    w.resize(min(width, room.width()), min(height, room.height() - 40))
    w.show()
    wait(app, 1.0)

    lines = [
        "## Frame timings",
        "",
        f"{platform.platform()}, {platform.machine()}, Python {platform.python_version()}; "
        f"screen {screen.size().width()}×{screen.size().height()} @{screen.devicePixelRatio():g}x, "
        f"{screen.refreshRate():.0f} Hz; window {w.width()}×{w.height()}; theme {args.theme}; "
        f"pixmap cache {QtGui.QPixmapCache.cacheLimit() / 1024:g} MB; git {_git_head()}",
        "",
    ]
    picked = [s for s in SCENARIOS if not args.only or any(o.lower() in s.label.lower() for o in args.only)]
    if any(s.label.startswith("Rotor") for s in picked):
        lines += [f"Rotor page opening (#204): {rotor_opening(app, w)}", ""]

    asked: dict[str, float] = {}
    painted: dict[str, dict[str, float]] = {}
    placed: dict[str, list[float]] = {}
    stroked: dict[str, tuple[list, list]] = {}
    for s in picked:
        print(f"{s.label} …", flush=True)
        try:
            page, asked[s.label] = s.setup(w)
        except Exception as e:  # a page changed under the script: say so, and carry on
            print(f"  failed to set up: {e!r}")
            asked[s.label] = float("nan")
            continue
        wait(app, SETTLE_S)
        stats = frame_stats.STATS
        assert stats is not None, "VIB_FRAME_STATS was set before the window was built"
        split.take()
        placing.take()
        strokes.take()
        stats.note("input", MARK + "start " + s.label)
        split.on = placing.on = strokes.on = True
        counted = min(args.seconds, s.seconds or args.seconds)
        wait(app, counted)
        split.on = placing.on = strokes.on = False
        stats.note("input", MARK + "end " + s.label)
        painted[s.label] = split.take()
        placed[s.label] = [counted, *placing.take()]
        stroked[s.label] = strokes.take()
        stop(w, page)
    split.uninstall()
    placing.uninstall()
    strokes.uninstall()
    gui.close_graphics(w)
    frame_stats.uninstall()

    sys.path.insert(0, str(Path(__file__).parent))  # frame_report, beside this script
    rows = importlib.import_module("frame_report").read(str(csv_path))
    frames: dict[str, list[dict[str, str]]] = {}
    current: str | None = None
    for r in rows:
        if r["kind"] == "input" and r["detail"].startswith(MARK):
            what, _, label = r["detail"][len(MARK) :].partition(" ")
            current = label if what == "start" else None
        elif r["kind"] == "frame" and current is not None:
            frames.setdefault(current, []).append(r)
    lines += [
        "Work and paint: median / 95th percentile per frame (ms); late: frames over 25 ms after the last.",
        "",
        "| Scenario | What it measures | fps | Work | Paint | Late | Sim speed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in picked:
        lines.append(f"| {s.label} | {s.note} " + summarize(frames.get(s.label, []), asked[s.label]))
    lines += [
        "",
        "### Where the paint goes",
        "",
        "A frame's paint (ms, mean), then each plot's repaint, the costliest first; other: the rest of the window.",
        "",
        "| Scenario | Paint | By plot |",
        "| --- | --- | --- |",
    ]
    for s in picked:
        lines += split_rows(s.label, painted.get(s.label, {}), frames.get(s.label, []))
    lines += [
        "",
        "### Legend placement",
        "",
        "Each legend's placement (#316 B16), on its own timer between frames: how often, and ms each.",
        "",
        "| Scenario | Placements a second | ms each (median / max) |",
        "| --- | --- | --- |",
    ]
    for s in picked:
        seconds, *ms = placed.get(s.label, [1.0])
        each = f"{np.median(ms):.2f} / {max(ms):.2f}" if ms else ""
        lines.append(f"| {s.label} | {len(ms) / seconds:.1f} | {each} |")
    lines += [
        "",
        "### Held curves",
        "",
        "Curve strokes a frame (#409 N-2): held curves should be ~0 (drawn from their pixmaps); ms a frame each.",
        "",
        "| Scenario | Held strokes | Held ms | Other strokes | Other ms |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in picked:
        if s.label in stroked:
            lines.append(stroke_row(s.label, *stroked[s.label], len(frames.get(s.label, []))))
    if args.app:
        lines += ["", "### Packaged start-up", "", *startup(args.app, out)]
    report = "\n".join(lines) + "\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Written to {out / 'report.md'}; the frames are in {csv_path}.")
    return 0


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"


if __name__ == "__main__":
    sys.exit(main())
