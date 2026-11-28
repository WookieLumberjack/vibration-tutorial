"""The view models (src/vib_tutorial/view_models/): free of Qt, and what each page prints, as printed.

None of these tests builds a QApplication: a view model is plain functions from a page's settings to what it
shows, so its readouts are checked here against the numbers its notes and trials quote.
"""

import math
import pkgutil
import re
import subprocess
import sys

import numpy as np
import pytest
from notes_quoting import quoted

import vib_tutorial.view_models as view_models
from vib_tutorial.core.isolation import Isolator, Mount, deflection_for
from vib_tutorial.gui.isolation_trials import TRIALS
from vib_tutorial.view_models import duffing
from vib_tutorial.view_models import isolation as view
from vib_tutorial.view_models.isolation import IsolationTrial, isolator
from vib_tutorial.view_models.live_isolation import FLOOR, MACHINE, by_speed
from vib_tutorial.view_models.recorder import Point

MODULES = [f"vib_tutorial.view_models.{m.name}" for m in pkgutil.iter_modules(view_models.__path__)]


def test_view_models_load_no_qt():
    """Each module, imported alone in a fresh interpreter, loads neither Qt, pyqtgraph nor gui/."""
    assert "vib_tutorial.view_models.isolation" in MODULES
    code = (
        "import sys, importlib\n"
        f"for name in {MODULES!r}: importlib.import_module(name)\n"
        "print(' '.join(m for m in sys.modules if m.split('.')[0] in ('PySide6', 'pyqtgraph', 'shiboken6')"
        " or m.startswith('vib_tutorial.gui')))\n"
    )
    loaded = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert loaded.strip() == ""


# ------------------------------------------------------------------ Vibration isolation


def defaults() -> Isolator:
    return isolator(IsolationTrial("", ""))


def at_speed(t: IsolationTrial, shaking: int = MACHINE) -> str:
    """The readout once the trial's speed has landed its T (the theory's: a dwell lands it to 0.05 %)."""
    iso = isolator(t)
    return view.steady_note(iso, iso.at_speed()[2], shaking)


def test_isolation_readouts_print_the_worked_example():
    """The page's readouts on its defaults show the worked example's figures, as the notes round them."""
    iso = defaults()
    info = view.mount_info(iso)
    assert f"c = {quoted('isolation', '442.9')} N·s/m" in info
    assert f"f<sub>n</sub> = {quoted('isolation', '7.048')} Hz" in info
    note = at_speed(IsolationTrial("", ""))
    assert f"{quoted('isolation', '22.6')} N reaches the floor" in note
    assert f"T = {quoted('isolation', '0.0916')}" in note
    assert f"isolation {quoted('isolation', '90.8 %'):g} % (-{quoted('isolation', '20.8')} dB)" in note
    assert "pulls with 247 N" in note and "±109 µm" in note  # the trial's figures (246.7 N in the example)
    assert view.chart(iso).note.startswith("This machine: 1500 rpm on 5 mm mounts, isolating 90.8 %")


def test_isolation_trials_readouts_say_what_their_notes_do():
    """Each trial's note, against the page's readout once the trial's speed has landed its T."""
    notes = {t.name: at_speed(t, t.shaking) for t in TRIALS}
    infos = {t.name: view.mount_info(isolator(t)) for t in TRIALS}
    assert "isolation 90.8 %" in notes["Soft mounts"]  # "91 % isolation"
    assert "f<sub>n</sub> = 28.78 Hz" in infos["Stiff mounts"]  # 28.8 Hz
    assert "amplified 3.86 times" in notes["Stiff mounts"]  # 3.9 times
    assert "T = 0.2," in notes["Soft mounts, more damping"]  # 20 %
    assert "T = 0.0701" in notes["An elastically supported damper"]  # 7.0 %
    assert "T = 0.038," in notes["Two stages"]  # 3.8 %
    assert "±27.1 µm" in notes["An inertia block"]  # ±27 µm instead of ±109 µm
    assert "the machine ±9.16 µm: T = 0.0916" in notes["The floor shakes instead"]
    run = next(t for t in TRIALS if t.name == "Running up through resonance")
    assert "f<sub>n</sub> = 7.048 Hz (423 rpm)" in view.mount_info(isolator(run))  # 420 rpm


def test_isolation_amplifying_mounts_say_so_everywhere():
    """Below √2 f_n the readouts say "amplified", never a negative isolation."""
    stiff = IsolationTrial("", "", deflection_mm=0.3)
    for text in (at_speed(stiff), view.chart(isolator(stiff)).note):
        assert "amplified 3.86 times" in text and not re.search(r"-[\d.]+ %", text)
    assert "isolating" not in view.chart(isolator(stiff)).note


def test_isolation_transmissibility_is_built_from_the_points_and_the_theory_on_request():
    state = IsolationTrial("", "")
    iso = isolator(state)
    fn = iso.natural_hz
    k = view.key(state)
    points = [
        Point(25.0, complex(0.09, 0.01), (k, MACHINE)),
        Point(10.0, complex(0.9, 0.1), (k, MACHINE)),
        Point(25.0, complex(0.0916, 0.0), (k, FLOOR)),
        Point(25.0, complex(0.2, 0.0), (view.key(IsolationTrial("", "", zeta=0.3)), MACHINE)),  # before a change
    ]
    v = view.t_view(state, points, False)
    assert list(v.machine.x) == pytest.approx([10.0 / fn, 25.0 / fn]) and v.machine.y[1] == pytest.approx(
        abs(0.09 + 0.01j)
    )
    assert list(v.floor.x) == pytest.approx([25.0 / fn]) and v.earlier.x.size == 1
    assert v.isolator.x.size == 0 and v.family == ()  # no theory until asked for
    assert "fall on one curve" in v.note and v.speed == pytest.approx(25.0 / fn)
    shown = view.t_view(state, points, True)
    assert len(shown.family) == len(view.FAMILY) and all(c.x is view.RATIOS for c in shown.family)
    for c in shown.family:  # every plain mount passes T = 1 at √2 f_n, whatever its damping
        assert np.interp(math.sqrt(2), c.x, c.y) == pytest.approx(1.0, abs=2e-3)
    peaks = [np.nanmax(c.y) for c in shown.family]  # the resonant peak falls as the damping rises
    assert peaks[0] > 100 and peaks == sorted(peaks, reverse=True)
    assert np.interp(25.0 / fn, *shown.isolator) == pytest.approx(iso.at_speed()[2], rel=1e-3)
    assert "Run the machine" in view.t_view(state, [], False).note
    # (a key that only differs in speed or unbalance is the same mounts)
    assert view.key(IsolationTrial("", "", rpm=900.0, unbalance=0.02)) == k
    bare = view.t_view(IsolationTrial("", "", zeta=0.0), [], True)
    assert np.isfinite(bare.isolator.y).all()  # the resonance is left out


def test_isolation_floor_force_tab():
    state = IsolationTrial("", "")
    iso = isolator(state)
    rpm, peaks = np.array([300.0, 423.0, 600.0]), np.array([10.0, 111.0, 40.0])
    v = view.force_view(state, rpm, peaks, True)
    assert "111 N at 423 rpm" in v.note and v.title == "Force on the floor" and v.unit == "N"
    assert np.allclose(v.rigid.y, iso.force_at(v.rigid.x / 60))
    assert np.allclose(v.theory.y, iso.transmissibility(v.rigid.x / 60) * v.rigid.y)
    assert v.rpm_range[1] >= 1.25 * iso.rpm
    floor = view.force_view(IsolationTrial("", "", shaking=FLOOR), np.empty(0), np.empty(0), False)
    assert floor.title == "Machine's motion" and floor.theory.x.size == 0 and "Run the machine up" in floor.note
    # each whole turn's peak: a steady sine's peak at its speed
    t = np.linspace(0.0, 1.0, 10001)
    phase = 2 * np.pi * 10 * t
    r, pk = by_speed(t, np.full(t.size, 10.0), phase, 3.0 * np.cos(phase))
    assert r == pytest.approx(600.0) and pk == pytest.approx(3.0, rel=1e-4) and r.size == 9


def test_isolation_design_note():
    iso = defaults()
    assert "a sag of 4.37 mm or more" in view.design_note(iso, 0.9)
    assert 1e3 * deflection_for(iso.speed_hz, 0.9) == pytest.approx(4.37, abs=0.005)
    assert view.design_note(iso, 1.0) == ""  # no mount isolates everything


def test_isolation_chart():
    iso = isolator(IsolationTrial("", "", mount=Mount.TWO_STAGE))
    v = view.chart(iso)
    assert len(v.levels) == len(view.CHART_LEVELS)
    # more isolation needs more sag, at every speed; and less at a higher speed
    sags = np.array([c.y for c in v.levels])
    assert (np.diff(sags, axis=0) > 0).all() and (np.diff(sags, axis=1) < 0).all()
    assert (float(v.now.x[0]), float(v.now.y[0])) == pytest.approx((1500.0, 5.0))


# ------------------------------------------------------------------ Duffing & Mathieu
# The page's run without the page: the simulator set up as a preset sets it, its stretches taken into a Record.


def preset_run(name: str):
    from vib_tutorial.core.duffing import PRESETS, DuffingSimulator, Sweep

    p = next(p for p in PRESETS if p.name == name)
    sim = DuffingSimulator(p.osc)
    d = sim.drive
    d.freq_hz, d.start_hz, d.end_hz, d.sweep_time = p.freq_hz, p.start_hz, p.end_hz, p.sweep_time
    d.start(p.mode)
    if p.mode is Sweep.HOLD:
        d.freq_hz = p.freq_hz
    return sim, duffing.Record()


def duffing_readout(sim, record) -> str:
    shown = duffing.response(sim.osc, duffing.x_range(sim.drive))
    return duffing.readout(sim.osc, shown.curve, shown.p_curve, record.cycles, sim.escaped, sim.runaway)


def test_duffing_sweep_readout_and_points():
    """A sweep up and down the hardening spring: harmonic balance's folds as the notes quote them, then the jumps the
    run measured, each leg's points, and the section only once it holds."""
    from vib_tutorial.core.duffing import Sweep

    sim, record = preset_run("Hardening spring: the jump")
    text = duffing_readout(sim, record)
    assert f"between {quoted('duffing', '3.64')} and {quoted('duffing', '4.45')} Hz" in text
    assert f"ends at 4.45 Hz ({quoted('duffing', '71.4')} mm) and drops to {quoted('duffing', '5.23')} mm" in text
    assert f"ends at 3.64 Hz ({quoted('duffing', '24.0')} mm) and jumps to {quoted('duffing', '46.2')} mm" in text
    assert "Measured" not in text
    while sim.drive.mode is not Sweep.HOLD:
        record.add(sim.advance(5.0))
    text = duffing_readout(sim, record)
    assert "Measured: the jump at 4." in text and "and 3.5" in text
    shown = duffing.response(sim.osc, duffing.x_range(sim.drive))
    v = duffing.points(sim.osc, record, shown.y_top, sim.runaway)
    assert len(v.first["up"]) > 100 and len(v.first["down"]) > 100 and not any(len(h) for h in v.half.values())
    assert len(v.section) == sum(c.leg == "hold" for c in record.cycles)  # Poincaré dots only once it holds
    assert v.now.shape == (1, 2)


def test_duffing_live_is_cheap():
    """Every frame: the strip chart decimated to its buckets, the phase plane a view of the trace, not a copy."""
    sim, record = preset_run("Hardening spring: the jump")
    for _ in range(10):
        record.add(sim.advance(5.0))
    v = duffing.live(record.trace, sim.t, sim.drive.freq_hz, 0.07)
    assert v.rest is None
    assert v.strip.x.size <= 2 * duffing.PLOT_POINTS + 2
    assert v.strip_range == pytest.approx((sim.t - duffing.WINDOW, sim.t))
    assert np.shares_memory(v.orbit.x, record.trace.data) and np.shares_memory(v.orbit.y, record.trace.data)
    rest = duffing.live(duffing.Trace(), 0.0, 2.0, 0.07)  # waiting for Start: the run's ranges, not 0 to 1
    assert rest.strip_range == (0.0, duffing.WINDOW)
    assert rest.rest == pytest.approx((0.07, 2 * math.pi * 2.0 * 0.07))


def test_duffing_branches():
    from vib_tutorial.core.duffing import Duffing

    sim, _ = preset_run("Two stable states")
    b = duffing.branches(sim.osc, 4.0)
    assert b.enabled == (True, True, True)
    assert b.note == "At 4 Hz, steady amplitudes: 8.80 mm, 52.0 mm (unstable), 58.3 mm."
    amplitudes = 1e3 * duffing.steady_states(sim.osc, 4.0)[0]  # as the notes print them
    assert amplitudes == pytest.approx(
        [quoted("duffing", "8.80"), quoted("duffing", "52.0"), quoted("duffing", "58.3")], abs=0.05
    )
    assert duffing.branches(sim.osc, 5.0).enabled[:2] == (True, False)  # outside the band: one amplitude
    pumped, _ = preset_run("Parametric resonance, hardening")
    b = duffing.branches(pumped.osc, pumped.drive.freq_hz)
    assert b.enabled == (True, False, True) and "rest (unstable)" in b.note
    shown = duffing.response(pumped.osc, duffing.x_range(pumped.drive))
    assert "p_stable" in shown.legend and "up" not in shown.legend
    assert shown.y_label.startswith("Amplitude (½X")
    soft = Duffing(cubic=-1e5)
    assert duffing.response(soft, (2.0, 6.0)).legend_right


def test_duffing_escapes():
    sim, record = preset_run("Escape from the well")
    while not sim.escaped:
        record.add(sim.advance(5.0))
    assert "ran away" in duffing_readout(sim, record)
    sim, record = preset_run("Parametric resonance")
    while not sim.escaped:
        record.add(sim.advance(5.0))
    text = duffing_readout(sim, record)
    assert "without bound" in text and "km" in text


def test_duffing_chart():
    from vib_tutorial.core.duffing import Duffing

    c = duffing.chart(Duffing().zeta)
    assert c.growth.shape == (186, 61)
    assert c.rect[0] < duffing.CHART_RATIOS[0] and c.rect[0] + c.rect[2] > duffing.CHART_RATIOS[-1]
    pumped = Duffing(pump=0.8)
    at_fn = pumped.natural_hz
    assert "rest is unstable" in duffing.chart_point(pumped, at_fn).note  # the second tongue, at f_n
    assert "rest is stable" in duffing.chart_point(Duffing(pump=0.8, damping=4.0), at_fn).note  # lifted off it
    bare = duffing.chart_point(Duffing(stiffness=0.0, cubic=1e5), 2.0)
    assert bare.now.size == 0 and bare.note == duffing.NO_SPRING


# Where a g or e format code is right: SVG and LaTeX numbers, CSV and the frame log, Qt's scale factor, the
# formatter itself (#409 U-17).
G_FORMAT_KEPT = (
    "figures.py",
    "plot_export.py",
    "frame_stats.py",
    "display_size.py",
    "text.py",
    "maths.py",
    "notes_render.py",
    "font_metrics.py",
)
G_FORMAT = re.compile(r"\{[^{}]*:[#+,]?\.?\d*[ge]\}|:\.\{\w+\}[ge]\}")


def test_no_page_prints_with_a_g_or_e_format_code():
    """Every number the pages show goes through `text.number` (`sig`, `fmt`, …), which never prints e-notation:
    a `{x:.3g}` gives 1.23e+04 or 4.62e-07 once x is large or small enough (#409 U-17). A `float(f"{x:.3g}")`
    rounds rather than prints, and stays."""
    from pathlib import Path

    root = Path(view_models.__file__).parent.parent
    found = []
    for path in sorted([*(root / "gui").glob("*.py"), *(root / "view_models").glob("*.py")]):
        if path.name in G_FORMAT_KEPT or path.name.endswith("_figures.py"):
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if G_FORMAT.search(line) and 'float(f"' not in line:
                found.append(f"{path.name}:{n}: {line.strip()}")
    assert not found, "\n".join(found)


def test_numbers_print_one_way():
    """`view_models/text.py` (#316 D8): figures with trailing zeros dropped unless asked, no −0, SI prefixes in
    a range, and the pages' own formatters are calls to it."""
    from vib_tutorial.gui.animation import _fmt_len
    from vib_tutorial.gui.cms_time import _fmt as cms_length
    from vib_tutorial.gui.continuum import fmt_hz
    from vib_tutorial.gui.continuum_waves_view import _fmt_time
    from vib_tutorial.gui.energy import fmt_energy
    from vib_tutorial.gui.rotor_readouts import fmt_len
    from vib_tutorial.gui.rotor_rub import _fmt_um
    from vib_tutorial.gui.torsional_startup_view import fmt_torque
    from vib_tutorial.view_models.duffing import fmt_length, sig3
    from vib_tutorial.view_models.text import entry, fixed, fmt, number, plain

    assert number(2.5) == "2.5" and number(2.5, zeros=True) == "2.50" and number(-1e-9, 2) == "-1×10⁻⁹"
    assert number(-0.0001, 1) == "-0.0001" and number(float("inf")) == "∞" and number(float("nan")) == "–"
    assert sig3(24.0) == "24.0" and sig3(8.8) == "8.80" and sig3(112.4) == "112"
    assert fmt(0.0251, "m") == "25.1 mm" and fmt(0.0, "m") == "0 m" and fmt(9.996e-4, "m") == "1 mm"
    assert fmt(1.234e7, "N/m", largest="G", smallest="µ") == "12.3 MN/m" and fmt(1.234e7, "N/m") == "12,300 kN/m"
    assert fmt(1e-10, "m") == "0.1 nm"  # past the smallest prefix: kept at it
    assert fmt_length(0.024) == "24.0 mm" and fmt_length(2.0) == "2.00 m" and fmt_length(2500.0) == "2.50 km"
    assert fmt_len(2.5e-5) == "25 µm" and fmt_torque(1500.0) == "1.5 kN·m" and fmt_torque(12.0) == "12 N·m"
    # no e-notation (#409 U-17): separators from 10,000, a power of ten only past 10⁹ or below 10⁻⁵
    assert number(7363, 4) == "7363" and number(56_820, 4) == "56,820" and number(1.0343e5, 5) == "103,430"
    assert number(4.62e-7) == "4.62×10⁻⁷" and number(1.2e12) == "1.2×10¹²" and number(2.5e-5) == "0.000025"
    assert (
        entry(3e-12, scale=100.0) == "0"
        and entry(1.25e8) == "125,000,000"
        and entry(1.25e10) == "1.25×10¹⁰"
        and entry(-0.5) == "-0.5"
    )
    assert fmt_energy(0.0) == "0 J" and fmt_energy(2e-12) == "2 pJ" and fmt_energy(3e-4) == "300 µJ"
    assert (
        plain(0.6612, "g", 4) == "0.6612 g"
        and fixed(-0.04, 1) == "0"
        and fixed(2.0, 1) == "2"
        and fixed(2.25, 2) == "2.25"
    )
    # the pages' own (#409 S-13): calls to fmt
    assert _fmt_um(2.5e-5) == "25 µm" and _fmt_um(0.0012) == "1.2 mm" and cms_length(3e-9) == "3 nm"
    assert _fmt_len(1.5) == "1.5 m" and _fmt_len(0.02) == "20 mm" and _fmt_time(0.004) == "4 ms"
    assert fmt_hz(523.25) == "523.2 Hz" and fmt_hz(12_345.0) == "12.35 kHz"


def test_record_buffers_keep_the_latest():
    """`view_models/record.py` (#409 S-9): full, a buffer drops its oldest half and keeps the newest; the chain's
    `History` numbers every sample it ever stored (`offset`), dropped ones too."""
    from vib_tutorial.view_models.record import ENERGY_COLUMNS, History, SampleBuffer

    b = SampleBuffer(2, capacity=10)
    for start in range(0, 16, 4):
        t = np.arange(start, start + 4, dtype=float)
        b.extend(t, np.column_stack([t, -t]))
    t, y = b.window(100.0)
    assert t[-1] == 15.0 and list(y[-1]) == [15.0, -15.0] and b.size <= 10 and np.all(np.diff(t) == 1.0)
    assert list(b.window(2.0)[0]) == [13.0, 14.0, 15.0]
    h = History(1, capacity=8)
    for start in range(0, 20, 5):
        t = np.arange(start, start + 5, dtype=float)
        z = np.zeros((5, 1))
        h.extend(t, z, z, t, np.zeros((5, len(ENERGY_COLUMNS))), np.zeros((5, 2)))
    assert h.t[h.size - 1] == 19.0 and h.offset + h.size == 20, "every sample numbered"
    h.reset(2)
    assert h.size == 0 and h.offset == 20 and h.x.shape[1] == 2
