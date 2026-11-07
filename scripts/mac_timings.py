"""Frame timings on this machine's own display, scenario by scenario, as a report to paste into #130.

    uv run python scripts/mac_timings.py                    # every scenario, ~2 minutes
    uv run python scripts/mac_timings.py --seconds 4 rotor  # just the scenarios whose label has "rotor"
    uv run python scripts/mac_timings.py --app "dist/Vibration Tutorial.app/Contents/MacOS/vib-tutorial"

It opens the window on the real display (not offscreen: the point is the display's own painting, which
on a Retina Mac is most of a frame), runs each scenario through the pages' own controls, and logs every
frame (`VIB_FRAME_STATS`) to <out>/frames.csv. Each scenario gets a second to settle, then `--seconds` of
frames are counted. <out>/report.md has a line per scenario: frames per second, the computing (work) and
drawing (paint) per frame, median and 95th percentile, how many frames were late, and how fast the
simulation ran against what was asked. The rotor page's opening is timed too, and with `--app` the
packaged app's start-up, cold then warm, as the release workflow measures it.

Leave the window alone while it runs: a click or a key press is logged as an input but changes the run.
`scripts/frame_report.py <out>/frames.csv` cuts the same log by what was on screen instead.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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
    w.apply_sim_trial(SimTrial("timing", "", **trial))
    return w.sim_page, w.controls.speed_factor


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
    w.apply_sim_trial(TRIALS[0])
    return w.sim_page, w.controls.speed_factor


def _rotor(preset: str, rpm: float | None = None, speed: float = 1.0):
    def setup(w):
        from vib_tutorial.gui import rotor

        page = w.rotor_page
        w.pages.setCurrentWidget(page)
        page.load_preset(preset)
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
        from vib_tutorial.gui import torsional

        page = w.torsional_page
        w.pages.setCurrentWidget(page)
        page.load_preset(preset)
        page.playback.setCurrentIndex(torsional.PLAYBACK.index(speed))
        page.start()
        return page, speed

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
    Scenario("Rotor: full rub, 1×", "rub kernel", _rotor("Full rub (backward whirl)")),
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
)


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
        speed = f"{ran:.2f}× of {asked:g}×"
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

    from vib_tutorial import gui
    from vib_tutorial.gui import frame_stats
    from vib_tutorial.gui.main_window import MainWindow

    app = gui.make_app(args.theme)
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
        f"git {_git_head()}",
        "",
    ]
    picked = [s for s in SCENARIOS if not args.only or any(o.lower() in s.label.lower() for o in args.only)]
    if any(s.label.startswith("Rotor") for s in picked):
        lines += [f"Rotor page opening (#204): {rotor_opening(app, w)}", ""]

    asked: dict[str, float] = {}
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
        stats.note("input", MARK + "start " + s.label)
        wait(app, min(args.seconds, s.seconds or args.seconds))
        stats.note("input", MARK + "end " + s.label)
        if hasattr(page, "_run"):
            page._run(False)
        elif page is w.sim_page and w.force.active:
            w.force.switch_off()
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
