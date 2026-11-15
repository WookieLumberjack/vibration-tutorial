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
    assert "isolation 90.8 %" in notes["Soft mounts: isolation"]  # "91 % isolation"
    assert "f<sub>n</sub> = 28.78 Hz" in infos["Stiff mounts: amplification"]  # 28.8 Hz
    assert "amplified 3.86 times" in notes["Stiff mounts: amplification"]  # 3.9 times
    assert "T = 0.2," in notes["Damping costs isolation"]  # 20 %
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


def test_isolation_chart():
    iso = isolator(IsolationTrial("", "", mount=Mount.TWO_STAGE))
    v = view.chart(iso)
    assert len(v.levels) == len(view.CHART_LEVELS)
    # more isolation needs more sag, at every speed; and less at a higher speed
    sags = np.array([c.y for c in v.levels])
    assert (np.diff(sags, axis=0) > 0).all() and (np.diff(sags, axis=1) < 0).all()
    assert (float(v.now.x[0]), float(v.now.y[0])) == pytest.approx((1500.0, 5.0))
