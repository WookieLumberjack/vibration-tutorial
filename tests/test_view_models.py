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
from vib_tutorial.view_models import duffing
from vib_tutorial.view_models import isolation as view
from vib_tutorial.view_models.isolation import TRIALS, IsolationTrial, isolator

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


def test_isolation_readouts_print_the_worked_example():
    """The page's readouts on its defaults show the worked example's figures, as the notes round them."""
    iso = defaults()
    info = view.mount_info(iso)
    assert f"c = {quoted('isolation', '442.9')} N·s/m" in info
    assert f"f<sub>n</sub> = {quoted('isolation', '7.048')} Hz" in info
    note = view.transmissibility(iso).note
    assert f"{quoted('isolation', '22.6')} N reaches the floor" in note
    assert f"T = {quoted('isolation', '0.0916')}" in note
    assert f"isolation {quoted('isolation', '90.8 %'):g} % (-{quoted('isolation', '20.8')} dB)" in note
    assert "pulls with 247 N" in note and "±109 µm" in note  # the trial's figures (246.7 N in the example)
    assert view.chart(iso).note.startswith("This machine: 1500 rpm on 5 mm mounts, isolating 90.8 %")


def test_isolation_trials_readouts_say_what_their_notes_do():
    """Each trial's note, against the page's readout once the trial is set up."""
    notes = {t.name: view.transmissibility(isolator(t)).note for t in TRIALS}
    infos = {t.name: view.mount_info(isolator(t)) for t in TRIALS}
    assert "isolation 90.8 %" in notes["Soft mounts"]  # "91 % isolation"
    assert "f<sub>n</sub> = 28.78 Hz" in infos["Stiff mounts"]  # 28.8 Hz
    assert "amplified 3.86 times" in notes["Stiff mounts"]  # 3.9 times
    assert "T = 0.2," in notes["Soft mounts, more damping"]  # 20 %
    assert "T = 0.0701" in notes["An elastically supported damper"]  # 7.0 %
    assert "T = 0.038," in notes["Two stages"]  # 3.8 %
    assert "±27.1 µm" in notes["An inertia block"]  # ±27 µm instead of ±109 µm
    run = next(t for t in TRIALS if t.name == "Running up through resonance")
    assert "Through resonance (423 rpm" in view.start_up_view(isolator(run), run.ramp).note  # 420 rpm


def test_isolation_amplifying_mounts_say_so_everywhere():
    """Below √2 f_n the readouts say "amplified", never a negative isolation."""
    stiff = isolator(IsolationTrial("", "", deflection_mm=0.3))
    for text in (view.transmissibility(stiff).note, view.chart(stiff).note):
        assert "amplified 3.86 times" in text and not re.search(r"-[\d.]+ %", text)
    assert "isolating" not in view.chart(stiff).note


def test_isolation_transmissibility_curves():
    iso = defaults()
    v = view.transmissibility(iso)
    assert len(v.family) == len(view.FAMILY) and all(c.x is view.RATIOS for c in v.family)
    # every plain mount passes T = 1 at √2 f_n, whatever its damping
    for c in v.family:
        assert np.interp(math.sqrt(2), c.x, c.y) == pytest.approx(1.0, abs=2e-3)
    # the resonant peak falls as the damping rises (ζ = 0's misses r = 1 on the grid, so stays finite)
    peaks = [np.nanmax(c.y) for c in v.family]
    assert peaks[0] > 100 and peaks == sorted(peaks, reverse=True)
    r, T = (float(a[0]) for a in v.now)
    assert (r, T) == pytest.approx((iso.speed_hz / iso.natural_hz, iso.at_speed()[2]))
    assert np.interp(r, *v.isolator) == pytest.approx(T, rel=1e-3)
    # the floor force is T times the unbalance's, and the marker sits on it
    assert np.allclose(v.floor.y, iso.transmissibility(v.rigid.x / 60) * v.rigid.y)
    assert v.rpm_range == (v.rigid.x[0], v.rigid.x[-1]) and v.rpm_range[0] < iso.rpm < v.rpm_range[1]
    assert float(v.force_now.y[0]) == pytest.approx(iso.at_speed()[1])


def test_isolation_markers_and_undamped_mounts():
    off = view.transmissibility(isolator(IsolationTrial("", "", rpm=60.0 * 7.048 * 30)))  # r = 30, off the plot
    assert off.now.x.size == 0
    bare = view.transmissibility(isolator(IsolationTrial("", "", zeta=0.0)))
    assert np.isfinite(bare.isolator.y).all() and np.isfinite(bare.floor.y).all()  # the resonance is left out


def test_isolation_design_note():
    iso = defaults()
    assert "a sag of 4.37 mm or more" in view.design_note(iso, 0.9)
    assert 1e3 * deflection_for(iso.speed_hz, 0.9) == pytest.approx(4.37, abs=0.005)
    assert view.design_note(iso, 1.0) == ""  # no mount isolates everything


def test_isolation_start_up_view():
    iso = defaults()
    v = view.start_up_view(iso, 2.0)
    assert v.speed.x.size <= 6001 and v.t_end > 2.0
    assert v.speed.y[-1] == pytest.approx(iso.rpm) and v.resonance_rpm == pytest.approx(60 * iso.natural_hz)
    assert np.allclose(v.rigid.y, iso.force_at(v.speed.y / 60))
    assert "running, 22.6 N and ±109 µm" in v.note


def test_isolation_let_go_view():
    iso = defaults()
    v = view.let_go_view(iso, -0.5e-3)
    assert v.x.y[0] == pytest.approx(-0.5e-3) and not v.speed.y.any() and not v.force.y.any()
    crossings = np.flatnonzero(np.diff(np.sign(v.x.y)) != 0)
    period = 2 * np.mean(np.diff(v.x.x[crossings]))
    assert 1 / period == pytest.approx(iso.natural_hz * math.sqrt(1 - iso.zeta**2), rel=0.01)
    assert "rings on its mounts and dies away, at f<sub>n</sub> = 7.05 Hz" in v.note
    two = isolator(IsolationTrial("", "", mount=Mount.TWO_STAGE))
    assert "one stage of plain mounts would ring" in view.let_go_view(two, 1e-3).note


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
    v = duffing.live(record.trace, sim.t, sim.drive.freq_hz)
    assert v.strip.x.size <= 2 * duffing.PLOT_POINTS + 2
    assert v.strip_range == pytest.approx((sim.t - duffing.WINDOW, sim.t))
    assert np.shares_memory(v.orbit.x, record.trace.data) and np.shares_memory(v.orbit.y, record.trace.data)
    assert duffing.live(duffing.Trace(), 0.0, 2.0).strip_range is None


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


def test_numbers_print_one_way():
    """`view_models/text.py` (#316 D8): figures with trailing zeros dropped unless asked, no −0, SI prefixes in
    a range, and the pages' own formatters are calls to it."""
    from vib_tutorial.gui.energy import fmt_energy
    from vib_tutorial.gui.rotor_readouts import fmt_len
    from vib_tutorial.gui.torsional_startup_view import fmt_torque
    from vib_tutorial.view_models.duffing import fmt_length, sig3
    from vib_tutorial.view_models.text import fixed, fmt, number, plain

    assert number(2.5) == "2.5" and number(2.5, zeros=True) == "2.50" and number(-1e-9, 2) == "-1e-09"
    assert number(-0.0001, 1) == "-0.0001" and number(float("inf")) == "∞" and number(float("nan")) == "–"
    assert sig3(24.0) == "24.0" and sig3(8.8) == "8.80" and sig3(112.4) == "112"
    assert fmt(0.0251, "m") == "25.1 mm" and fmt(0.0, "m") == "0 m" and fmt(9.996e-4, "m") == "1 mm"
    assert fmt(1.234e7, "N/m", largest="G", smallest="µ") == "12.3 MN/m" and fmt(1.234e7, "N/m") == "1.23e+04 kN/m"
    assert fmt(1e-10, "m") == "0.1 nm"  # past the smallest prefix: kept at it
    assert fmt_length(0.024) == "24.0 mm" and fmt_length(2.0) == "2.00 m" and fmt_length(2500.0) == "2.50 km"
    assert fmt_len(2.5e-5) == "25 µm" and fmt_torque(1500.0) == "1.5 kN·m" and fmt_torque(12.0) == "12 N·m"
    assert fmt_energy(0.0) == "0 J" and fmt_energy(2e-12) == "2 pJ" and fmt_energy(3e-4) == "300 µJ"
    assert (
        plain(0.6612, "g", 4) == "0.6612 g"
        and fixed(-0.04, 1) == "0"
        and fixed(2.0, 1) == "2"
        and fixed(2.25, 2) == "2.25"
    )
