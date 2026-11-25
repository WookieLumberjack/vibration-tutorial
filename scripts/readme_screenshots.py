"""Regenerate the README's and docs/' screenshots in docs/images, each at 1700x900 in the Light theme.

    uv run python scripts/readme_screenshots.py              # all of them
    uv run python scripts/readme_screenshots.py chirp pluck  # just these (the function names below)

Each shot builds a fresh MainWindow, sets up the state its caption (in docs/)
describes and saves window.grab(). The window's frame timer is stopped and the
script advances simulated time itself, so every image is taken at a fixed
moment and reruns give the same pictures. Run it after a UI change, then check
that the captions' numbers still match the images.

It runs offscreen with a temporary settings folder, so it neither opens a
window nor touches the user's saved theme or playback speed. It drives the
widgets directly (some private attributes too, such as MainWindow._tick), so a
UI refactor may need a matching change here: tests/test_scripts.py runs every shot
(without the waits or the PNGs) to catch it. Pick a trial by name with `trial.pick`:
the combo's items are numbered, so `setCurrentText` matches none and loads nothing.
"""

import atexit
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

from vib_tutorial import ONE_THREAD

os.environ.update({name: os.environ.get(name, "1") for name in ONE_THREAD})  # BLAS on one thread, as the app
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="vib-screenshots-")  # before Qt reads settings

from PIL import Image
from PySide6 import QtCore, QtGui
from PySide6.QtTest import QTest

from vib_tutorial import gui
from vib_tutorial.core import ForceKind
from vib_tutorial.core.rotor import RotorSystem
from vib_tutorial.gui.main_window import MainWindow

atexit.register(shutil.rmtree, os.environ["XDG_CONFIG_HOME"], ignore_errors=True)  # the settings folder, on exit

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
COLOURS = 256  # a grab of the flat UI keeps its look in a palette this size, at about a third of the bytes
app = gui.make_app("Light")


def window(theme="Light"):
    w = MainWindow()
    # The script drives time (run), so every shot is at a known moment. Stop the timer before
    # the window shows: frames run while it settles would advance History.offset by a
    # wall-clock-dependent count, and the strip chart's decimation buckets are aligned to it.
    w.sim_page.loop.enabled = False
    w.set_theme(theme)
    w.resize(1700, 900)
    w.show()
    QTest.qWait(200)
    running, w.sim_page.running = w.sim_page.running, False
    w.sim_page._tick()  # draw the chain (its masses' positions) without advancing time
    w.sim_page.running = running
    w.sim_page.controls.reset.click()
    return w


def run(w, seconds):
    """Advance the Simulation page by `seconds` of simulated time (at the chosen speed) through its tick."""
    speed = w.sim_page.controls.speed_factor
    for _ in range(round(seconds / (0.1 * speed))):
        w.sim_page._last_wall = w.sim_page._clock.elapsed() / 1000.0 - 0.1
        w.sim_page._tick()
    app.processEvents()


def settle(ms=300):
    QTest.qWait(ms)


def write_png(grab: QtGui.QPixmap, path: Path) -> None:
    """Save a grab as a COLOURS-colour PNG (the 55 images were 16 MB in full colour, 5.6 MB this way)."""
    image = grab.toImage().convertToFormat(QtGui.QImage.Format.Format_RGB888)
    rgb = Image.frombuffer(
        "RGB", (image.width(), image.height()), bytes(image.constBits()), "raw", "RGB", image.bytesPerLine(), 1
    )
    rgb.quantize(COLOURS).save(path, optimize=True)


def save(w, name):
    settle()
    write_png(w.grab(), OUT / f"{name}.png")
    print("saved", name)
    done(w)


def done(w):
    w.sim_page.loop.enabled = False
    w.close()
    gui.close_graphics(w)
    w.deleteLater()
    app.processEvents()


def set_c1(w, c):
    w.sim_page.params.rows[0][3].setValue(c)


def kind(panel, k):
    panel.kind.setCurrentIndex(panel.kind.findData(k))


def tune(panel, mode):
    """Pick mode `mode` (1-based) in a 'Tune to…' combo."""
    panel.tune.activated.emit(mode)


def coords(w, text):
    w.sim_page.coords_combo.setCurrentIndex(w.sim_page.coords_combo.findText(text, QtCore.Qt.MatchFlag.MatchStartsWith))


def energy_view(w, by_mode):
    w.sim_page.energy.view.setCurrentIndex(1 if by_mode else 0)


def release(w, row):
    w.sim_page.table.selectRow(row)
    w.sim_page._release_mode()


def method(w, state_space):
    w.sim_page.method_combo.setCurrentIndex(1 if state_space else 0)


# ------------------------------------------------------------------ Simulation page
def main_window():
    w = window()
    tune(w.sim_page.force_panel, 2)
    w.sim_page.force_panel.button.click()
    run(w, 6.0)
    w.sim_page.table.selectRow(1)
    save(w, "main_window")


def dark_theme():
    w = window("Dark")
    set_c1(w, 15.0)
    method(w, True)
    coords(w, "Modal")
    tune(w.sim_page.force_panel, 2)
    w.sim_page.force_panel.button.click()
    run(w, 11.0)
    w.sim_page.table.selectRow(2)
    save(w, "dark_theme")


def pluck():
    from vib_tutorial.gui.animation import MAX_DRAG, SPACING

    w = window()
    energy_view(w, True)
    w.sim_page.controls.time_window.setValue(4.0)
    view = w.sim_page.chain

    def pixel(x):
        return view.mapFromScene(view.plotItem.vb.mapViewToScene(QtCore.QPointF(x, 0.0)))

    left = QtCore.Qt.MouseButton.LeftButton
    QTest.mousePress(view.viewport(), left, pos=pixel(view._centers[1]))
    QTest.mouseMove(view.viewport(), pixel(2 * SPACING + 0.8 * MAX_DRAG))
    QTest.mouseRelease(view.viewport(), left, pos=pixel(2 * SPACING + 0.8 * MAX_DRAG))
    run(w, 3.1)
    QTest.mousePress(view.viewport(), left, pos=pixel(view._centers[1]))
    QTest.mouseMove(view.viewport(), pixel(2 * SPACING - 0.5 * MAX_DRAG))
    for _ in range(5):
        w.sim_page._tick()
    save(w, "pluck")


def chirp():
    w = window()
    p = w.sim_page.force_panel
    kind(p, ForceKind.CHIRP)
    p.sweep_start.setValue(2.0)
    p.sweep_end.setValue(7.0)
    p.sweep_time.setValue(40.0)
    w.sim_page.controls.time_window.setValue(40.0)
    coords(w, "Modal")
    w.sim_page.tabs.setCurrentWidget(w.sim_page.frf_tab)
    p.button.click()
    run(w, 36.3)
    save(w, "chirp")


def base_excitation():
    w = window()
    p = w.sim_page.force_panel
    p.input.setCurrentIndex(1)
    p.base_amplitude.setValue(5.0)
    p.freq.setValue(1.6)
    w.sim_page.controls.time_window.setValue(4.0)
    w.sim_page.tabs.setCurrentWidget(w.sim_page.frf_tab)
    p.button.click()
    run(w, 30.3)
    save(w, "base_excitation")


def tuned_mass_damper():
    w = window()
    w.sim_page.params.load_preset(next(pr for pr in w.sim_page.params.presets if "Den Hartog" in pr.name))
    settle()
    w.sim_page.force_panel.button.click()
    run(w, 12.7)
    save(w, "tuned_mass_damper")


def classical_release_nonproportional():
    w = window()
    set_c1(w, 15.0)
    w.sim_page.controls.time_window.setValue(2.05)
    release(w, 2)
    run(w, 1.96)
    save(w, "classical_release_nonproportional")


def state_space_release():
    w = window()
    set_c1(w, 15.0)
    method(w, True)
    w.sim_page.controls.time_window.setValue(2.12)
    release(w, 4)
    run(w, 1.97)
    save(w, "state_space_release")


def modal_coordinates():
    w = window()
    set_c1(w, 15.0)
    coords(w, "Modal")
    w.sim_page.controls.time_window.setValue(2.05)
    release(w, 2)
    run(w, 2.07)
    save(w, "modal_coordinates")


def energy_by_mode():
    w = window()
    set_c1(w, 15.0)
    coords(w, "Modal")
    energy_view(w, True)
    w.sim_page.controls.time_window.setValue(2.05)
    release(w, 2)
    run(w, 0.8)
    save(w, "energy_by_mode")


def energy_history():
    w = window()
    coords(w, "Energy")
    w.sim_page.force_panel.freq.setValue(1.0)
    w.sim_page.force_panel.button.click()
    run(w, 8.3)
    w.sim_page.table.selectRow(0)
    save(w, "energy_history")


def element_forces():
    w = window()
    kind(w.sim_page.force_panel, ForceKind.STEP)
    coords(w, "Element forces")
    w.sim_page.force_panel.button.click()
    run(w, 9.3)
    save(w, "element_forces")


def frequency_response():
    w = window()
    w.sim_page.tabs.setCurrentWidget(w.sim_page.frf_tab)
    tune(w.sim_page.force_panel, 2)
    settle(500)
    write_png(w.sim_page.frf_plot.grab(), OUT / "frequency_response.png")
    print("saved frequency_response")
    done(w)


# ------------------------------------------------------------------ other pages
def frf_matrix():
    w = window()
    page = w.frf_page
    w.pages.setCurrentWidget(page)
    settle()
    page.mode_checks[3].setChecked(False)
    page._on_cell(0, 3, False)
    save(w, "frf_matrix")


def modal_coupling():
    w = window()
    for i in (2, 3):
        w.sim_page.params.rows[i][1].setValue(0.1)
        w.sim_page.params.rows[i][2].setValue(40.0)
        w.sim_page.params.rows[i][3].setValue(0.2)
    w.pages.setCurrentWidget(w.coupling_page)
    save(w, "modal_coupling")


def cms(w, n, interfaces, kept, method_name="Craig"):
    w.sim_page.params.dof.setValue(n)
    page = w.cms_page
    w.pages.setCurrentWidget(page)
    settle()
    page.method.setCurrentIndex(page.method.findText(method_name, QtCore.Qt.MatchFlag.MatchStartsWith))
    page.set_interfaces(interfaces)
    for spin, k in zip(page.kept, kept):
        spin.setValue(k)
    settle()
    return page


def substructuring_three():
    w = window()
    page = cms(w, 8, [2, 5], [1, 1, 1])
    page.tabs.setCurrentWidget(page.basis)
    page.table.selectRow(3)
    save(w, "substructuring_three")


def drive_cms(view, w, seconds):
    view.loop.enabled = False
    view.reset()
    view.kind.setCurrentIndex(view.kind.findData(ForceKind.HARMONIC))
    view.freq.setValue(w.sim_page.modal.modes[3].fn_hz)
    view.button.click()
    for _ in range(round(seconds / 0.05)):
        view.step(0.05)
    view.loop.enabled = False


def substructuring_time():
    w = window()
    page = cms(w, 8, [3], [1, 1])
    page.tabs.setCurrentWidget(page.time)
    settle()
    drive_cms(page.time, w, 9.36)
    save(w, "substructuring_time")


def substructuring_recovery():
    w = window()
    page = cms(w, 8, [3], [2, 2])
    page.tabs.setCurrentWidget(page.recovery)
    settle()
    r = page.recovery
    r.mass_box.setCurrentIndex(r.mass_box.findText("m2", QtCore.Qt.MatchFlag.MatchStartsWith))
    r.spring_box.setCurrentIndex(r.spring_box.findText("Spring k2", QtCore.Qt.MatchFlag.MatchStartsWith))
    drive_cms(r, w, 10.06)
    save(w, "substructuring_recovery")


def substructuring_compare():
    w = window()
    page = cms(w, 8, [3], [1, 1], "Rubin")
    page.tabs.setCurrentWidget(page.compare)
    page.table.selectRow(2)
    save(w, "substructuring_compare")


def test_page(block, force_noise, response_noise, exp_end, tab, poles=None):
    w = window()
    p = w.test_page
    w.pages.setCurrentWidget(p)
    settle()
    p.test_setup.block.setCurrentIndex(p.test_setup.block.findText(str(block)))
    p.test_setup.force_noise.setValue(force_noise)
    p.test_setup.response_noise.setValue(response_noise)
    p.test_setup.exp_end.setValue(exp_end)
    # Random noise: take the first seed whose automatic pick finds `poles` modes, as the caption says.
    for seed in range(40):
        p.seed = seed
        p.restart()
        while not p.acq.done:
            p._measure_some()
        app.processEvents()
        if poles is None or len(p.poles) == poles:
            break
    else:
        print(f"  no seed gave {poles} poles; the image shows {len(p.poles)}")
    p.tabs.setCurrentWidget(getattr(p, tab))
    return w


def virtual_modal_test():
    save(test_page(256, 1.0, 1.0, 2.0, "check"), "virtual_modal_test")


def modal_extraction():
    save(test_page(1024, 0.0, 1.0, 100.0, "stab_plot", poles=3), "modal_extraction")


def oma():
    w = window()
    p = w.oma_page
    w.pages.setCurrentWidget(p)
    settle()
    print(p.readout.text())
    save(w, "oma")


def oma_ssi():
    w = window()
    p = w.oma_page
    w.pages.setCurrentWidget(p)
    settle()
    p.trials.combo.pick("A machine near a mode")
    p.recompute()
    p.method.setCurrentText("SSI-cov")
    p.tabs.setCurrentWidget(p.stab_plot)
    settle()
    print(p.readout.text())
    save(w, "oma_ssi")


# ------------------------------------------------------------------ Jeffcott rotor
def jeffcott_rotor():
    w = window()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    settle()
    page.loop.enabled = False
    page._refresh_analysis()  # at once: it opens with it on its way from the worker
    page.running = False  # the script drives time, as on the Simulation page
    page.ramp.setValue(5000.0)
    page.set_target_rpm(1400.0)
    for _ in range(150):
        page.step(0.02)
    save(w, "jeffcott_rotor")


def jeffcott_run_up():
    w = window()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    settle()
    page.loop.enabled = False
    page._refresh_analysis()  # at once: it opens with it on its way from the worker
    page.running = False
    page.sweep_from.setValue(800.0)
    page.sweep_to.setValue(2400.0)
    for rate in (500.0, 5000.0):  # held side by side
        page.ramp.setValue(rate)
        page.start_sweep()
        while page.run_up is not None:
            page.step(0.02)
    save(w, "jeffcott_run_up")


def jeffcott_bow():
    w = window()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    settle()
    page.loop.enabled = False
    page.running = False
    page.set_system(RotorSystem(bow=10e-6, bow_angle=math.pi))
    page._refresh_analysis()
    page.sweep_from.setValue(400.0)
    page.sweep_to.setValue(2400.0)
    page.ramp.setValue(500.0)
    page.start_sweep()
    while page.run_up is not None:
        page.step(0.02)
    page.tabs.show_page(page.polar.parentWidget())
    page._draw()
    save(w, "jeffcott_bow")


def jeffcott_balancing():
    w = window()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    settle()
    page.loop.enabled = False
    page.running = False
    page.set_system(RotorSystem(bow=20e-6, bow_angle=math.pi / 2))
    page._refresh_analysis()
    page.ramp.setValue(5000.0)
    b = page.balancing
    page.tabs.show_page(b)
    b.station.setCurrentIndex(1)  # the disc
    for button in b.step_buttons:
        button.click()
        while page.measurement is not None:
            page.step(0.02)
    print(b.results.text())
    save(w, "jeffcott_balancing")


def jeffcott_two_plane():
    w = window()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    settle()
    page.loop.enabled = False
    page.running = False
    page.set_system(RotorSystem(position=0.3, skew=math.radians(0.5)))
    page._refresh_analysis()
    page.ramp.setValue(5000.0)
    b = page.balancing
    page.tabs.show_page(b)
    b.mode.setCurrentIndex(1)  # the disc's faces, 40 mm apart; probes at A and B
    for button in b.step_buttons:
        button.click()
        while page.measurement is not None:
            page.step(0.02)
    print(b.results.text())
    save(w, "jeffcott_two_plane")


def jeffcott_stability():
    w = window()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    settle()
    page.loop.enabled = False
    page.running = False
    page.editor.internal_damping.setValue(20.0)
    page._refresh_analysis()
    page.ramp.setValue(5000.0)
    page.set_target_rpm(5000.0)
    for _ in range(60):
        page.step(0.02)
    page.tap()
    for _ in range(60):
        page.step(0.02)
    page.tabs.show_page(page.spectrum.parentWidget())
    page._draw()
    save(w, "jeffcott_stability")


def jeffcott_oil_whirl():
    w = window()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    settle()
    page.loop.enabled = False
    page.running = False
    page.load_preset("Oil whirl (vertical rotor)")
    page._refresh_analysis()
    page.ramp.setValue(5000.0)
    page.set_target_rpm(3500.0)
    for _ in range(500):  # 10 s: the whirl sets in and settles to its limit cycle
        page.step(0.02)
    page.tabs.show_page(page.spectrum.parentWidget())
    page._draw()
    save(w, "jeffcott_oil_whirl")


def jeffcott_magnetic():
    w = window()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    settle()
    page.loop.enabled = False
    page.running = False
    page.load_preset("Magnetic bearings")
    page._refresh_analysis()
    page.ramp.setValue(5000.0)
    page.set_target_rpm(12_000.0)
    for _ in range(130):  # 2.6 s: up to speed
        page.step(0.02)
    page.tap()
    for _ in range(75):  # 1.5 s: the forward tilting mode grows from the tap
        page.step(0.02)
    page.tabs.show_page(page.magnet_panel)
    page._draw()
    print(page.readout.text())
    save(w, "jeffcott_magnetic")


def jeffcott_rub():
    w = window()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    settle()
    page.loop.enabled = False
    page.running = False
    page.load_preset("Partial rub (½X)")
    page._refresh_analysis()
    for _ in range(650):  # 13 s: up to 4400 rpm at 500 rpm/s, and settled into hitting the ring every other turn
        page.step(0.02)
    page.show_sag.setChecked(False)  # the orbit about the sag, the ring's bottom across it
    page.tabs.show_page(page.spectrum.parentWidget())
    for _ in range(50):  # the orbits' scale shrinks gradually to the orbit without the sag
        page._draw()
    print(page.readout.text())
    save(w, "jeffcott_rub")


def jeffcott_newkirk():
    w = window()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    settle()
    page.loop.enabled = False
    page.running = False
    page.load_preset("Newkirk effect (thermal bow)")
    page._refresh_analysis()
    page.tabs.show_page(page.rub_panel)
    for _ in range(900):  # 18 s: up to 1300 rpm, and 15 s of the spiral
        page.step(0.02)
    page._draw()
    print(page.rub_panel.spiral_note.text())
    save(w, "jeffcott_newkirk")


def jeffcott_overhung():
    w = window()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    settle()
    page.loop.enabled = False
    page.running = False
    page.load_preset("Overhung rotor")
    page._refresh_analysis()
    page.order_boxes[0.5].setChecked(True)
    page.ramp.setValue(2000.0)
    page.set_target_rpm(1250.0)
    for _ in range(200):
        page.step(0.02)
    print(page.readout.text())
    save(w, "jeffcott_overhung")


def jeffcott_waterfall():
    w = window()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    settle()
    page.loop.enabled = False
    page.running = False
    page.load_preset("Overhung rotor")
    page._refresh_analysis()
    page.ramp.setValue(200.0)
    page.tabs.show_page(page.waterfall_tab.plot.parentWidget())
    page.repeat_taps.setChecked(True)
    page.set_target_rpm(2500.0)
    while page.sim.omega < page.sim.target:
        page.step(0.02)
    page._draw()
    print(page.waterfall_tab.note.text())
    save(w, "jeffcott_waterfall")


def jeffcott_shaft_mass():
    w = window()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    settle()
    page.loop.enabled = False
    page.running = False
    page.load_preset("Long shaft, with its mass")
    page._refresh_analysis()
    page.ramp.setValue(1000.0)
    page.set_target_rpm(6700.0)
    while page.sim.omega < page.sim.target:
        page.step(0.02)
    for _ in range(25):
        page.step(0.02)
    print(page.readout.text())
    save(w, "jeffcott_shaft_mass")


def jeffcott_asymmetric():
    w = window()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    settle()
    page.loop.enabled = False
    page.running = False
    page.load_preset("Two-pole rotor (asymmetric shaft)")
    page._refresh_analysis()
    page.probe_direction.setCurrentIndex(2)  # the orbit
    page.probe_order.setCurrentIndex(1)  # 2X
    page.sweep_from.setValue(500.0)
    page.sweep_to.setValue(1200.0)
    page.ramp.setValue(200.0)
    page.start_sweep()
    while page.run_up is not None:
        page.step(0.02)
    page.set_target_rpm(785.0)
    for _ in range(100):
        page.step(0.02)
    print(page.sweep_summary.text())
    save(w, "jeffcott_asymmetric")


def jeffcott_asymmetry_chart():
    w = window()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    settle()
    page.loop.enabled = False
    page.running = False
    page.load_preset("Two-pole rotor (asymmetric shaft)")
    page._refresh_analysis()
    page.tabs.show_page(page.spectrum.parentWidget())
    page.measure.setCurrentIndex(2)  # the asymmetry chart
    page.asym_chart.run()
    page.ramp.setValue(2000.0)
    page.set_target_rpm(785.0)
    for _ in range(150):
        page.step(0.02)
    page._draw_spectrum()
    print(page.stability_note.text())
    save(w, "jeffcott_asymmetry_chart")


def jeffcott_cracked():
    w = window()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    settle()
    page.loop.enabled = False
    page.running = False
    page.load_preset("Cracked shaft (breathing crack)")
    page._refresh_analysis()
    page.probe_direction.setCurrentIndex(2)  # the orbit
    page.probe_order.setCurrentIndex(2)  # 3X
    page.sweep_from.setValue(400.0)
    page.sweep_to.setValue(1000.0)
    page.ramp.setValue(100.0)
    page.start_sweep()
    while page.run_up is not None:
        page.step(0.02)
    page.set_target_rpm(522.0)
    for _ in range(100):
        page.step(0.02)
    print(page.sweep_summary.text())
    save(w, "jeffcott_cracked")


def duffing_jump():
    from vib_tutorial.core.duffing import Sweep

    w = window()
    page = w.duffing_page
    w.pages.setCurrentWidget(page)
    settle()
    page.loop.enabled = False
    page.running = False
    while page.sim.drive.mode is not Sweep.HOLD:  # the preset's sweep up, then down
        page.step(1.0)
    print(page.readout.text())
    save(w, "duffing_jump")


def fluid_structure():
    lesson("fluid_page", 1, "fluid_structure")  # a cylinder's dipole flow


def liquid_damper():
    lesson("fluid_page", 14, "liquid_damper")  # a tuned liquid damper on a 20 t structure


def mathieu_sweep():
    from vib_tutorial.core.duffing import Sweep

    w = window()
    page = w.duffing_page
    w.pages.setCurrentWidget(page)
    settle()
    page.loop.enabled = False
    page.running = False
    page.preset.setCurrentText("Parametric sweep: the jump")
    page.tabs.setCurrentWidget(page.chart_tab)
    while page.sim.drive.mode is not Sweep.HOLD:  # down from rest, then back up
        page.step(1.0)
    print(page.readout.text())
    save(w, "mathieu_sweep")


def torsional_compressor():
    w = window()
    page = w.torsional_page
    w.pages.setCurrentWidget(page)
    page.load_preset("Geared compressor")
    page.tabs.setCurrentWidget(page.shapes_tab)
    settle()
    print(page.readout.text())
    save(w, "torsional_compressor")


def torsional_startup():
    w = window()
    page = w.torsional_page
    w.pages.setCurrentWidget(page)
    page.load_preset("Geared compressor")
    page.start()
    page.startup_view.step(page.startup_view.startup.duration)
    settle()
    print(page.startup_view.startup_note.text())
    save(w, "torsional_startup")


def continuum_string():
    w = window()
    page = w.continuum_page
    w.pages.setCurrentWidget(page)
    page.load_preset("Guitar string (low E)")
    page.mode.setValue(6)
    settle()
    print(page.readout.text())
    print(page.mesh_info.text())
    save(w, "continuum_string")


def continuum_beam():
    w = window()
    page = w.continuum_page
    w.pages.setCurrentWidget(page)
    page.load_preset("Steel ruler, clamped–free")
    page.views.setCurrentWidget(page.errors_tab)
    page.mode.setValue(3)
    settle()
    print(page.convergence_note.text())
    print(page.shape_note.text())
    save(w, "continuum_beam")


def continuum_waves():
    w = window()
    page = w.continuum_page
    w.pages.setCurrentWidget(page)
    page.load_preset("Waves: a light string tied to a heavy one")
    page.loop.enabled = False
    page.waves_view.running = False
    page.waves_view.step(120)  # half a crossing: the reflection and the transmitted pulse apart
    settle()
    print(page.waves_view.wave_status.text())
    print(page.waves_view.wave_info.text())
    save(w, "continuum_waves")


def random_vibration():
    w = window()
    page = w.random_page
    w.pages.setCurrentWidget(page)
    page.trials.combo.pick("Base shake")
    page.recompute()
    settle()
    print(page.readout.text())
    print(page.record_info.text())
    save(w, "random_vibration")


def random_statistics():
    w = window()
    page = w.random_page
    w.pages.setCurrentWidget(page)
    page.trials.combo.pick("The base spring's force")
    page.tabs.setCurrentWidget(page.stats_tab)
    page.recompute()
    settle()
    print(page.largest_note.text())
    save(w, "random_statistics")


def complex_modes():
    w = window()
    page = w.complex_page
    w.pages.setCurrentWidget(page)
    page.trials.combo.pick("One damper, at the base")
    page.mode.setCurrentIndex(2)
    page.tabs.setCurrentWidget(page.decay_tab)
    settle()
    print(page.readout.text())
    print(page.decay_note.text())
    save(w, "complex_modes")


def complex_sweep():
    w = window()
    page = w.complex_page
    w.pages.setCurrentWidget(page)
    page.trials.combo.pick("One damper, at the base")
    page.tabs.setCurrentWidget(page.sweep_tab)
    settle()
    print(page.sweep_note.text())
    page.tabs.setCurrentWidget(page.ident_tab)
    settle()
    print(page.ident_note.text())
    page.tabs.setCurrentWidget(page.sweep_tab)
    settle()
    save(w, "complex_sweep")


def model_updating():
    w = window()
    page = w.updating_page
    w.pages.setCurrentWidget(page)
    page.trials.combo.pick("One spring 30 % too stiff")
    page.recompute()
    settle()
    print(page.readout.text())
    print([(p.test + 1, p.model + 1, round(p.mac, 3), f"{100 * p.error:+.2f}") for p in page.corr.pairs])
    save(w, "model_updating")


def model_updating_springs():
    w = window()
    page = w.updating_page
    w.pages.setCurrentWidget(page)
    page.trials.combo.pick("Update: find the stiff spring")
    page.recompute()
    page.tabs.setCurrentWidget(page.update_tab)
    settle()
    print(page.update_note.text())
    save(w, "model_updating_springs")


def test_planning():
    from vib_tutorial.gui.test_planning import TAB_ODS, TRIALS

    w = window()
    page = w.planning_page
    w.pages.setCurrentWidget(page)
    page.load_trial(next(i for i, t in enumerate(TRIALS) if t.name == "Move the force"))
    page.tabs.setCurrentIndex(TAB_ODS)
    settle()
    print(page.shapes_readout.text())
    print(page.ods_readout.text())
    save(w, "test_planning")


def nonlinear_forces():
    from vib_tutorial.gui.nonlinear_forces import TAB_LCURVE, TRIALS

    w = window()
    page = w.nonlinear_page
    w.pages.setCurrentWidget(page)
    page.load_trial(next(i for i, t in enumerate(TRIALS) if t.name == "A hardening spring"))
    page.method.setCurrentIndex(2)  # Tikhonov, at the corner
    page.corner_button.click()
    page.tabs.setCurrentIndex(TAB_LCURVE)
    settle()
    print(page.frf_readout.text())
    print(page.force_readout.text())
    save(w, "nonlinear_forces")


def shock_spectrum():
    w = window()
    page = w.shock_page
    w.pages.setCurrentWidget(page)
    page.trials.combo.pick("Half-sine")
    page.split_time.setChecked(True)
    page.recompute()
    settle()
    print(page.readout.text())
    print(page.osc_note.text())
    save(w, "shock_spectrum")


def shock_chain():
    w = window()
    page = w.shock_page
    w.pages.setCurrentWidget(page)
    page.trials.combo.pick("Modes combined")
    page.recompute()
    settle()
    print(page.chain_note.text())
    save(w, "shock_chain")


def shock_synthesis():
    w = window()
    page = w.shock_page
    w.pages.setCurrentWidget(page)
    page.trials.combo.pick("Wavelets to a spectrum")
    page.recompute()
    page.tabs.setCurrentWidget(page.tri_tab)
    settle()
    print(page.readout.text())
    print(page.tri_note.text())
    save(w, "shock_synthesis")


def lesson(attr, trial, name, tab=None, at=None):
    """A One mass or Signals page with one of its trials loaded (its readouts printed for the caption); `at`, the
    time its moving body is held at (the trial starts it)."""
    w = window()
    page = getattr(w, attr)
    w.pages.setCurrentWidget(page)
    page.load_trial(trial)
    if tab is not None:
        page.tabs.setCurrentIndex(tab)
    if at is not None:
        page.playback.seek(at)
    settle()
    from PySide6 import QtWidgets

    for label in page.findChildren(QtWidgets.QLabel):
        if label.isVisible() and len(label.text()) > 40:
            print(" ", label.text())
    save(w, name)


def single_oscillator():
    """The live mass after a sweep has built its frequency response, the theory's curve shown over the points."""
    from PySide6 import QtWidgets

    from vib_tutorial.gui import sdof

    w = window()
    page = w.sdof_page
    w.pages.setCurrentWidget(page)
    page.loop.enabled = False
    page.load_trial(next(i for i, t in enumerate(sdof.TRIALS) if t.name.startswith("Building the frequency")))
    page.step(120.0)  # the sweep, to its end
    page.theory.setChecked(True)
    settle()
    for label in page.findChildren(QtWidgets.QLabel):
        if label.isVisible() and len(label.text()) > 40:
            print(" ", label.text())
    save(w, "single_oscillator")


def transient_response():
    lesson("transient_page", 3, "transient_response")  # the convolution


def single_oscillator_damping():
    """Four loops, one energy: the viscous mass driven to ±10 mm at f_n, its loop and every law's."""
    from PySide6 import QtWidgets

    from vib_tutorial.gui import sdof

    w = window()
    page = w.sdof_page
    w.pages.setCurrentWidget(page)
    page.loop.enabled = False
    page.load_trial(next(i for i, t in enumerate(sdof.TRIALS) if t.name.startswith("Four loops")))
    page.step(12.0)
    settle()
    for label in page.findChildren(QtWidgets.QLabel):
        if label.isVisible() and len(label.text()) > 40:
            print(" ", label.text())
    save(w, "single_oscillator_damping")


def vibration_isolation():
    lesson("isolation_page", 3, "vibration_isolation", at=0.8)  # through resonance, the machine at its largest swing


def signals_spectra():
    lesson("signals_page", 2, "signals_spectra")  # a Hann window finds the small tone


def pendulums():
    """The live pendulum let go from 175° with a little damping: each swing has landed its period as it died away,
    tracing the curve, the theory's shown over the points."""
    from PySide6 import QtWidgets

    from vib_tutorial.gui import pendulum

    w = window()
    page = w.pendulum_page
    w.pages.setCurrentWidget(page)
    page.loop.enabled = False
    page.load_trial(next(i for i, t in enumerate(pendulum.TRIALS) if t.name.startswith("Nearly over the top")))
    page.step(90.0)
    page.theory.setChecked(True)
    settle()
    for label in page.findChildren(QtWidgets.QLabel):
        if label.isVisible() and len(label.text()) > 40:
            print(" ", label.text())
    save(w, "pendulums")


def coordinates_bases():
    """Two masses of 1 and 2 kg described along their modes: both matrices diagonal, the modes 80° apart, each
    coordinate one sine."""
    from PySide6 import QtWidgets

    from vib_tutorial.gui import bases

    w = window()
    page = w.bases_page
    w.pages.setCurrentWidget(page)
    page.loop.enabled = False
    page.load_trial(next(i for i, t in enumerate(bases.TRIALS) if t.name.startswith("Unequal masses")))
    page.step(6.0)
    settle()
    for label in page.findChildren(QtWidgets.QLabel):
        if label.isVisible() and len(label.text()) > 40:
            print(" ", label.text())
    save(w, "coordinates_bases")


def periodic_forcing():
    lesson("periodic_page", 1, "periodic_forcing")  # resonance through the third harmonic


def energy_methods():
    lesson("energy_page", 6, "energy_methods")  # Rayleigh–Ritz: two shapes, then three


def beams_plates():
    lesson("beams_page", 4, "beams_plates")  # twin modes and Chladni's figures


def numerical_integration():
    lesson("integration_page", 1, "numerical_integration")  # central difference's limit


def fatigue():
    lesson("fatigue_page", 1, "fatigue")  # two modes: the narrow-band formula is pessimistic


def quarter_car():
    lesson("quarter_car_page", 0, "quarter_car")  # a car on an average road


def deriving_equations():
    from vib_tutorial.gui.deriving import TAB_FBD

    w = window()
    page = w.deriving_page
    w.pages.setCurrentWidget(page)
    page.load_trial(3)  # moments about the front axle
    page.tabs.setCurrentIndex(TAB_FBD)
    page.fbd.verticalScrollBar().setValue(page.fbd.verticalScrollBar().maximum())
    settle()
    page._restart()  # the mode at its full swing (t = 0), held there
    page.loop.enabled = False
    print(page.readout.text())
    save(w, "deriving_equations")


def deriving_pendulum():
    from vib_tutorial.gui.deriving import TRIALS

    w = window()
    page = w.deriving_page
    w.pages.setCurrentWidget(page)
    page.load_trial(next(i for i, t in enumerate(TRIALS) if t.name == "Released at 120° and 120°"))
    page.loop.enabled = False
    for _ in range(int(7.5 / 0.05)):  # well into the chaos, its trail looping
        page.step(0.05)
    settle()
    print(page.chaos_note.text())
    save(w, "deriving_pendulum")


def frame_builder():
    from vib_tutorial.gui.frame_builder import TAB_ASSEMBLY, TRIALS

    w = window()
    page = w.frame_page
    w.pages.setCurrentWidget(page)
    page.load_trial(next(i for i, t in enumerate(TRIALS) if t.name == "A portal frame: sway"))
    page.loop.enabled = False
    page.running = False
    page._phase = 0.5 * math.pi
    page.tabs.setCurrentIndex(TAB_ASSEMBLY)
    settle()
    column = next(k for k, e in enumerate(page.frame.elements) if e.i == 0)  # the left column's foot
    page.canvas.select(("element", column))
    page._show_mode()
    settle()
    print(page.canvas.selection, page.assembly_note.text())
    print(page.readout.text())
    save(w, "frame_builder")


def rotor_builder():
    from vib_tutorial.gui.rotor_builder import TAB_WHIRL, TRIALS

    w = window()
    page = w.rotor_builder_page
    w.pages.setCurrentWidget(page)
    page.load_trial(next(i for i, t in enumerate(TRIALS) if t.name == "A stepped compressor rotor"))
    page.loop.enabled = False
    page.running = False
    page.tabs.setCurrentIndex(TAB_WHIRL)
    page.table.selectRow(3)
    page._phase = 0.3 * math.pi
    settle()
    page._show_mode()
    settle()
    print(page.whirl_note.text())
    print(page.readout.text())
    save(w, "rotor_builder")


def rotor_builder_campbell():
    from vib_tutorial.gui.rotor_builder import TRIALS

    w = window()
    page = w.rotor_builder_page
    w.pages.setCurrentWidget(page)
    page.load_trial(next(i for i, t in enumerate(TRIALS) if t.name == "Campbell diagram: the overhung fan"))
    page.loop.enabled = False
    page.running = False
    page._phase = 0.3 * math.pi
    settle()
    page._show_mode()
    settle()
    print(page.campbell_note.text())
    save(w, "rotor_builder_campbell")


def frame_response():
    from vib_tutorial.gui.frame_builder import TAB_RESPONSE, TRIALS

    w = window()
    page = w.frame_page
    w.pages.setCurrentWidget(page)
    page.load_trial(next(i for i, t in enumerate(TRIALS) if t.name == "A tuned mass damper"))
    page.loop.enabled = False
    page.running = False
    page._phase = 0.5 * math.pi
    page.tabs.setCurrentIndex(TAB_RESPONSE)
    settle()
    page._show_mode()
    settle()
    print(page.response_note.text())
    save(w, "frame_response")


def rigid_mounts():
    w = window()
    page = w.mounts_page
    w.pages.setCurrentWidget(page)
    page.load_trial(1)  # mounts below the centre of mass: sway and roll couple
    page.loop.enabled = False
    page.running = False
    page._phase = 0.5 * math.pi
    page._show_mode()
    settle()
    print(page.readout.text())
    save(w, "rigid_mounts")


def rigid_mounts_response():
    from vib_tutorial.gui.rigid_mounts import TRIALS

    w = window()
    page = w.mounts_page
    w.pages.setCurrentWidget(page)
    page.load_trial(next(i for i, t in enumerate(TRIALS) if t.name == "Running at the bounce: amplified"))
    page.loop.enabled = False
    page.running = False
    page._phase = 0.5 * math.pi
    page._show_mode()
    settle()
    print(page.response_note.text())
    save(w, "rigid_mounts_response")


def seismic_response():
    lesson("seismic_page", 2, "seismic_response")  # close modes need CQC


def vibroacoustics():
    lesson("acoustics_page", 3, "vibroacoustics")  # an edge mode, and every mode's σ against Maidanik
    lesson("acoustics_page", 11, "vibroacoustics_sea")  # two plates at a corner and a room


def variability():
    lesson("variability_page", 6, "variability")  # a mistuned disc's forced response


def self_excited():
    lesson("self_excited_page", 8, "self_excited")  # flutter: two modes coalesce


def active_control():
    lesson("active_control_page", 2, "active_control")  # non-collocated: spillover


def gyroscope():
    w = window()
    page = w.gyroscope_page
    w.pages.setCurrentWidget(page)
    page.load_trial(0)  # let go at rest: cusps
    page.loop.enabled = False
    page.running = False
    page.step(2.0)
    settle()
    print(page.readout.text())
    save(w, "gyroscope")


def gyroscope_chaos():
    from vib_tutorial.gui.gyroscope import TRIALS

    w = window()
    page = w.gyroscope_page
    w.pages.setCurrentWidget(page)
    page.load_trial(next(i for i, t in enumerate(TRIALS) if t.name == "Gravity and the middle axis: chaos"))
    page.loop.enabled = False
    page.running = False
    for _ in range(80):
        page.step(0.25)
    page._draw_tab()
    settle()
    print(page.readout.text())
    save(w, "gyroscope_chaos")


def gyroscope_table():
    from vib_tutorial.gui.gyroscope import TRIALS

    w = window()
    page = w.gyroscope_page
    w.pages.setCurrentWidget(page)
    page.load_trial(next(i for i, t in enumerate(TRIALS) if t.name == "A tippe top spun fast"))
    page.loop.enabled = False
    page.running = False
    page.step(4.0)
    settle()
    print(page.readout.text())
    save(w, "gyroscope_table")


def rigid_body():
    """The default box spun 1° off its middle axis, 2.5 s on: turned over, ω's components in its principal axes."""
    w = window()
    page = w.rigid_body_page
    w.pages.setCurrentWidget(page)
    page.load_trial(0)  # spun about the middle axis, it flips
    page.loop.enabled = False
    page.running = False
    page.step(2.5)
    page._draw_tab()
    settle()
    print(page.readout.text())
    save(w, "rigid_body")


def machinery_diagnostics():
    lesson("diagnostics_page", 3, "machinery_diagnostics")  # an outer race defect by its envelope


SHOTS = [
    main_window,
    dark_theme,
    pluck,
    chirp,
    base_excitation,
    tuned_mass_damper,
    classical_release_nonproportional,
    state_space_release,
    modal_coordinates,
    energy_by_mode,
    energy_history,
    element_forces,
    frequency_response,
    frf_matrix,
    modal_coupling,
    substructuring_three,
    substructuring_time,
    substructuring_recovery,
    substructuring_compare,
    virtual_modal_test,
    modal_extraction,
    oma,
    oma_ssi,
    random_vibration,
    random_statistics,
    complex_modes,
    complex_sweep,
    model_updating,
    model_updating_springs,
    test_planning,
    nonlinear_forces,
    shock_spectrum,
    shock_chain,
    shock_synthesis,
    jeffcott_rotor,
    jeffcott_run_up,
    jeffcott_bow,
    jeffcott_balancing,
    jeffcott_two_plane,
    jeffcott_stability,
    jeffcott_oil_whirl,
    jeffcott_magnetic,
    jeffcott_rub,
    jeffcott_newkirk,
    jeffcott_overhung,
    jeffcott_waterfall,
    jeffcott_shaft_mass,
    jeffcott_asymmetric,
    jeffcott_asymmetry_chart,
    jeffcott_cracked,
    duffing_jump,
    mathieu_sweep,
    fluid_structure,
    liquid_damper,
    torsional_compressor,
    torsional_startup,
    continuum_string,
    continuum_beam,
    continuum_waves,
    single_oscillator,
    transient_response,
    single_oscillator_damping,
    vibration_isolation,
    signals_spectra,
    pendulums,
    periodic_forcing,
    deriving_equations,
    deriving_pendulum,
    coordinates_bases,
    rigid_body,
    energy_methods,
    beams_plates,
    numerical_integration,
    fatigue,
    quarter_car,
    rigid_mounts,
    rigid_mounts_response,
    frame_builder,
    frame_response,
    seismic_response,
    vibroacoustics,
    variability,
    self_excited,
    active_control,
    gyroscope,
    gyroscope_chaos,
    gyroscope_table,
    rotor_builder,
    rotor_builder_campbell,
    machinery_diagnostics,
]


def main(names: list[str]) -> None:
    unknown = set(names) - {shot.__name__ for shot in SHOTS}
    if unknown:
        sys.exit(
            f"unknown shot(s): {', '.join(sorted(unknown))}; choose from: " + ", ".join(shot.__name__ for shot in SHOTS)
        )
    for shot in SHOTS:
        if not names or shot.__name__ in names:
            shot()


if __name__ == "__main__":
    main(sys.argv[1:])
