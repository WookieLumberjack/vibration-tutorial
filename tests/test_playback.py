"""`gui/playback.py`: a response walked in time, and the One mass pages' bodies moving with it."""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore  # noqa: E402
from test_drag import drag, scene_at  # noqa: E402


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


def test_playback_walks_the_span_pauses_at_its_end_and_scrubs(app):
    from vib_tutorial.gui import close_graphics
    from vib_tutorial.gui.lesson import plot
    from vib_tutorial.gui.playback import SPEEDS, Playback

    page = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(page)
    p = plot("x", "m", "t", "s")
    pb = Playback(page, "test")
    layout.addWidget(pb.bar())
    layout.addWidget(p)
    cursor = pb.add_cursor(p)
    seen: list[float] = []
    pb.moved.connect(seen.append)
    p.setXRange(0.0, 2.0, padding=0.05)
    page.resize(600, 400)
    page.show()
    for _ in range(10):
        app.processEvents()
    page.grab()  # laid out and painted: the cursor has its place on screen
    pb.loop.enabled = False  # the frames are driven by hand
    pb.speed.setCurrentIndex(next(i for i, (_, s) in enumerate(SPEEDS) if s == 0.5))
    pb.set_span(2.0)
    assert pb.t == 0.0 and pb.run.text() == "Start" and not pb.running

    pb.play()
    assert pb.running and pb.run.text() == "Pause"
    pb._tick(1.0)  # a second of wall clock at ½×
    assert pb.t == pytest.approx(0.5) and cursor.data_value() == pytest.approx(0.5) and pb.clock.text() == "t = 0.50 s"
    pb._tick(10.0)
    assert pb.t == 2.0 and not pb.running and pb.run.text() == "Resume"  # paused at the end
    pb.run.click()  # Resume at the end plays it again
    assert pb.t == 0.0 and pb.running

    pb._tick(1.0)
    pb.set_span(0.3)  # the page redrew with a shorter span: t is kept inside it
    assert pb.t == pytest.approx(0.3) and seen[-1] == pytest.approx(0.3)

    pb.set_span(2.0)
    pb.play()
    drag(p, scene_at(p, 0.0, 0.5), scene_at(p, 1.2, 0.5), app)  # the reader scrubs the cursor
    assert not pb.running and pb.t == pytest.approx(1.2, abs=0.05) and seen[-1] == pb.t

    pb.reset.click()
    assert pb.t == 0.0 and pb.run.text() == "Start" and cursor.data_value() == 0.0
    close_graphics(page)


def test_a_trial_with_a_prediction_asks_it_before_it_runs(main_window, app):
    """On a page that asks before it runs (Transient response; the live pages ask after play)."""
    from vib_tutorial.core.practice import Prediction
    from vib_tutorial.gui.transient import TRIALS

    w = main_window
    p = w.transient_page
    w.pages.setCurrentWidget(p)
    box = p.trials
    i = next(k for k, t in enumerate(TRIALS) if isinstance(t.predict, Prediction))
    prediction = TRIALS[i].predict
    assert isinstance(prediction, Prediction)
    before = p.practice_state()
    box.combo.pick(TRIALS[i].name)
    assert box.asking == i and box.ask.isVisible() and "Predict first" in box.question.text()
    assert p.practice_state() == before  # nothing set up yet
    buttons = [box.choices.itemAt(k).widget() for k in range(box.choices.count())]
    order = prediction.shown_order()  # shuffled, the same each time
    assert [b.text() for b in buttons] == [prediction.choices[k] for k in order]
    assert [int(b.property("choice")) for b in buttons] == list(order)

    wrong = next(k for k in range(len(prediction.choices)) if k != prediction.answer)
    buttons[order.index(wrong)].click()
    assert box.asking is None and not box.ask.isVisible() and "Not quite" in box.verdict.text()
    assert prediction.choices[prediction.answer] in box.verdict.text()
    assert prediction.why[wrong] and prediction.why[wrong][:20] in box.verdict.text()  # why the one picked is wrong
    assert box.note.isVisible() and box.note.text()

    box.combo.pick(TRIALS[i].name)  # again: asked again
    assert box.ask.isVisible() and not box.verdict.isVisible()
    box.answer(prediction.answer)
    assert "Right" in box.verdict.text()

    box.combo.pick(TRIALS[i].name)
    box.skip.click()  # Just run it: no verdict
    assert not box.verdict.isVisible() and box.note.isVisible()

    plain = next(k for k, t in enumerate(TRIALS) if t.predict is None)
    box.combo.pick(TRIALS[plain].name)  # no prediction: runs at once
    assert not box.ask.isVisible() and box.note.text()

    box.combo.pick(TRIALS[i].name)
    w.restore_defaults()  # put back as it opened: no question, no verdict
    assert not box.ask.isVisible() and not box.verdict.isVisible() and box.asking is None


def test_a_placement_is_guessed_on_its_plot(main_window, app):
    """A Placement (#409 I-6): the plot hides its curves and shows the span while the reader drags the line, the
    trial runs on the guess, which stays on the plot, dashed, with the verdict; Defaults takes it away."""
    from vib_tutorial.core.practice import Placement
    from vib_tutorial.gui.transient import TRIALS

    w = main_window
    p = w.transient_page
    w.pages.setCurrentWidget(p)
    box = p.trials
    t = next(t for t in TRIALS if isinstance(t.predict, Placement))
    place = t.predict
    assert isinstance(place, Placement)
    item = getattr(p, place.plot).getPlotItem()
    shown = [c for c in item.listDataItems() if c.isVisible()]
    assert shown
    box.combo.pick(t.name)
    assert box.place.isVisible() and box.line is not None and box.line in item.items
    assert not any(c.isVisible() for c in shown)  # the last trial's curves would give it away
    lo, hi = place.span
    guess = lo + 0.05 * (hi - lo)  # well short
    assert not place.judge(guess)[0]
    box.place_guess(guess)
    assert box.line is None and all(c.isVisible() for c in shown)
    assert "Not quite" in box.verdict.text() and place.low[:20] in box.verdict.text()
    assert box.guessed is not None and box.guessed in item.items
    w.restore_defaults()
    assert box.guessed is None and not box.ask.isVisible()


@pytest.mark.parametrize(
    ("attr", "module"),
    [
        ("transient_page", "transient"),
        ("periodic_page", "periodic"),
    ],
)
def test_the_one_mass_pages_ask_their_predictions_first(main_window, app, attr, module):
    import importlib

    from vib_tutorial.core.practice import Placement

    trials = importlib.import_module(f"vib_tutorial.gui.{module}").TRIALS
    w = main_window
    p = getattr(w, attr)
    w.pages.setCurrentWidget(p)
    box = p.trials
    asked = [k for k, t in enumerate(trials) if t.predict is not None]
    assert len(asked) >= len(trials) // 2  # most of them ask
    before = p.practice_state()
    for i in asked:
        box.combo.pick(trials[i].name)
        assert box.asking == i and box.ask.isVisible()
        assert p.practice_state() == before  # nothing set up until it's answered
        guess = trials[i].predict
        if isinstance(guess, Placement):
            box.place_guess(guess.answer)
            assert "Close" in box.verdict.text() and box.note.isVisible()
        else:
            box.answer(guess.answer)
            assert "Right" in box.verdict.text() and box.note.isVisible()
        w.restore_defaults()
        before = p.practice_state()


def test_a_goals_links_set_up_its_trial_and_ask_its_problem(main_window, app):
    """(conftest points the settings at the worker's own file: the Practice window reads them.)"""
    from PySide6 import QtCore as Core

    w = main_window
    browser = w.transient_page.notes
    w._wire_links(w.transient_page)
    browser.anchorClicked.emit(Core.QUrl("trial:transient_page#1"))
    p = w.transient_page
    assert w.pages.currentWidget() is p and p.trials.asking == 0 and p.trials.ask.isVisible()
    browser.anchorClicked.emit(Core.QUrl("practice:transient_page#impulse-peak"))
    pw = w.practice_window
    assert pw is not None and pw.problem is not None and pw.problem.key == "impulse-peak"


def test_a_cursors_plot_draws_its_curves_from_pixmaps(app):
    """The cursor moves over still curves: `plot_patches.hold_still` caches them, those plotted later too, so a
    frame repaints a strip of each pixmap rather than every curve's whole path (#316 E2)."""
    import pyqtgraph as pg

    from vib_tutorial.gui import close_graphics, plot_patches
    from vib_tutorial.gui.lesson import plot
    from vib_tutorial.gui.playback import Playback

    plot_patches.install()  # as make_app does
    cached = QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache
    page = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(page)
    held, free = plot("x", "m", "t", "s"), plot("y", "m", "t", "s")
    before = held.plot([0.0, 1.0], [0.0, 1.0])
    pb = Playback(page, "test")
    layout.addWidget(held)
    layout.addWidget(free)
    pb.add_cursor(held)
    held.clear()  # a redraw: the curves plotted again
    after, curve = held.plot([0.0, 1.0], [1.0, 0.0]), pg.PlotCurveItem([0.0, 1.0], [0.5, 0.5])
    held.addItem(curve)
    assert before.curve.cacheMode() == cached and after.curve.cacheMode() == cached and curve.cacheMode() == cached
    assert free.plot([0.0, 1.0], [0.0, 1.0]).curve.cacheMode() == QtWidgets.QGraphicsItem.CacheMode.NoCache
    close_graphics(page)


@pytest.mark.gui
def test_make_app_sizes_the_pixmap_cache_once_the_app_is_made():
    """In a fresh interpreter, as the app starts: making the QApplication sets the cache's limit back to 10 MB, so
    a limit set before it never held (#417 did, and the Mac's played-back plot stayed at 37 fps)."""
    import subprocess
    import sys

    from vib_tutorial.gui import plot_patches

    code = (
        "from PySide6 import QtGui; from vib_tutorial import gui; gui.make_app('Light'); "
        "print(QtGui.QPixmapCache.cacheLimit())"
    )
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60, check=True)
    assert int(out.stdout.split()[-1]) == plot_patches.PIXMAP_CACHE_MB * 1024


def test_the_pixmap_cache_holds_a_large_retina_windows_held_curves(main_window):
    """A held curve's pixmap lives in QPixmapCache: at 2× a played-back page's (the Single oscillator's, then
    Pendulums', then Vibration isolation's, now the Quarter-car's), in a 1700 px window, outgrew its
    default 10 MB, so the cache dropped them every frame (20 ms a frame on the Mac, #409 N-2). The limit holds
    a few pages' worth."""
    import pyqtgraph as pg
    from PySide6 import QtGui

    from vib_tutorial.gui import plot_patches

    # make_app sizes the cache (the test above, in a fresh interpreter); this worker's app may not have come from it
    plot_patches.size_pixmap_cache()
    assert QtGui.QPixmapCache.cacheLimit() == plot_patches.PIXMAP_CACHE_MB * 1024
    w = main_window
    w.resize(1700, 1030)  # 1000 px for the pages, below the buttons' row
    p = w.quarter_car_page
    w.pages.setCurrentWidget(p)
    QtWidgets.QApplication.processEvents()
    cached = QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache
    retina = 0  # bytes: each held curve's pixmap over the plot's view, at 2×
    for plot in p.findChildren(pg.PlotWidget):
        for item in plot.getPlotItem().items:
            curve = item.curve if isinstance(item, pg.PlotDataItem) else item
            if isinstance(curve, pg.PlotCurveItem) and curve.cacheMode() == cached:
                view = plot.mapFromScene(curve.mapRectToScene(curve.boundingRect())).boundingRect()
                view = view.intersected(plot.viewport().rect())
                retina += 4 * view.width() * view.height() * 4
    assert retina > 0  # the Single oscillator's 2× pixmaps (> 10 MB) went with its played-back plots (#478)
    assert 3 * retina < plot_patches.PIXMAP_CACHE_MB * 2**20


@pytest.mark.parametrize("wrap", [False, True])
def test_a_live_label_lays_the_page_out_again_only_when_it_needs_more_room(app, wrap):
    """`lesson.LiveLabel`: a frame's new text repaints the label, and asks for a layout pass (a QLabel's would,
    every frame) only when it needs another line, or a line a wider box."""
    from vib_tutorial.gui.lesson import LiveLabel

    page = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(page)
    label = LiveLabel("t = 0.00 s", wrap=wrap)
    layout.addWidget(label)
    layout.addStretch()
    page.resize(200 if wrap else 400, 300)
    page.show()
    app.processEvents()
    asked: list[int] = []

    class Watch(QtCore.QObject):
        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.Type.LayoutRequest:
                asked.append(1)
            return False

    watch = Watch()
    page.installEventFilter(watch)
    for t in (0.25, 0.5, 0.75):
        label.setText(f"t = {t:.2f} s")
        app.processEvents()
    assert asked == [] and label.text() == "t = 0.75 s" and label.accessibleDescription() == "t = 0.75 s"
    label.setText("t = 0.75 s, and a much longer status that needs " + "more room " * 8)
    app.processEvents()
    assert asked, "more room: laid out again"
    assert label.height() >= label.heightForWidth(label.width()) if wrap else label.width() >= label.sizeHint().width()
    page.removeEventFilter(watch)
    page.close()


def test_a_live_line_sized_to_its_text_stops_asking_once_it_is_wide_enough(app):
    """A clock in a row, as wide as its text: its digits differ in width, and sized to each text it grew, was
    laid out narrower, and grew again, a layout pass nearly every frame. It keeps the widest it has been."""
    from vib_tutorial.gui.lesson import LiveLabel

    page = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(page)
    label = LiveLabel("t = 1.111 s")
    row.addWidget(label)
    row.addStretch()
    page.resize(400, 40)
    page.show()
    app.processEvents()
    asked: list[int] = []

    class Watch(QtCore.QObject):
        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.Type.LayoutRequest:
                asked.append(1)
            return False

    texts = ("t = 4.888 s", "t = 1.111 s", "t = 8.044 s", "t = 1.171 s")
    for text in texts:  # once round: each wider than any before may ask
        label.setText(text)
        app.processEvents()
    watch = Watch()
    page.installEventFilter(watch)
    for _ in range(3):
        for text in texts:
            label.setText(text)
            app.processEvents()
    assert asked == [], "every width already met: no layout"
    page.removeEventFilter(watch)
    page.close()


def test_the_quarter_car_drives_over_its_road(main_window, app):
    from vib_tutorial.core.quarter_car import drive
    from vib_tutorial.gui.quarter_car import DRIVE_S, TAB_DRIVE, TRIALS, WINDOW_S
    from vib_tutorial.gui.quarter_car_view import SUSP_H, SWING, TYRE_H, WHEEL_H

    w = main_window
    w.resize(1280, 800)
    p = w.quarter_car_page
    w.pages.setCurrentWidget(p)
    pb, view = p.playback, p.car_view
    pb.loop.enabled = False
    p.tabs.setCurrentIndex(TAB_DRIVE)
    app.processEvents()
    assert not pb.running and pb.span == WINDOW_S  # at rest, waiting for Start

    # A trial on the Drive tab drives it from the start.
    p.load_trial(next(i for i, tr in enumerate(TRIALS) if tr.tab == TAB_DRIVE))
    app.processEvents()
    assert pb.running and pb.t == 0.0
    pb._tick(0.6)
    d = drive(p.car, p.road, DRIVE_S)
    road, wheel, body = (float(np.interp(pb.t, d.t, a)) for a in (d.road, d.wheel_x, d.body_x))
    g = view.gain
    n = int(WINDOW_S / (d.t[1] - d.t[0]))
    assert g == pytest.approx(SWING / max(np.max(np.abs(a[:n])) for a in (d.road, d.body_x, d.wheel_x)))
    # each box's underside (Qt's top(): the smaller y, here the lower edge)
    assert view.wheel_rect.rect().top() == pytest.approx(TYRE_H + g * wheel)
    assert view.body_rect.rect().top() == pytest.approx(TYRE_H + WHEEL_H + SUSP_H + g * body)
    x, y = view.road_line.getData()
    assert np.interp(0.0, x, y) == pytest.approx(g * road, abs=1e-6 * g)  # the road under the tire, now

    # Scrubbing the cursor moves the car with it.
    pb.cursors[0].dragged.emit(3.0)
    assert not pb.running and view.now == pytest.approx(3.0)


# Words that say what a trial shows, which its name mustn't: it shows before the question (#409 C-8). "Twice",
# "static", "impulse" and "negative" name setups too (Dampers twice too strong, The impulse response), so aren't here.
OUTCOME_WORDS = re.compile(
    r"\b(slower|faster|longer|shorter|higher|lower|larger|smaller|bigger|can't|cannot|won't|never|always|costs?|"
    r"quickest|locks?|converges?|amplif\w*|buckl\w*|beats|straight-line|softening|stiffening|overshoot\w*|unstable|"
    r"instability|grows?|falls?|rises?|runs away|stops?|cancels?|no resonance|resonates)\b",
    re.IGNORECASE,
)


def test_every_pages_predictions_are_asked_before_its_trials_run(main_window, app):
    """Every page whose trials carry a `Prediction` asks it through its Things to try first (#316 B3): nothing is
    set up until it's answered, the verdict and the note (the explanation) follow, and a trial's name doesn't give
    the answer away (it shows before the question)."""
    from vib_tutorial.core.practice import Placement
    from vib_tutorial.gui import links, sim_trials

    w = main_window
    checked = 0
    for attr in links.order():
        trials = sim_trials.TRIALS if attr == "sim_page" else getattr(links.page_module(attr), "TRIALS", ())
        asked = [k for k, t in enumerate(trials) if getattr(t, "predict", None) is not None]
        for k in asked:
            p = trials[k].predict
            answer = "" if isinstance(p, Placement) else p.choices[p.answer].split(":")[0].lower()
            if len(answer) >= 6:  # a "Yes" or "No" may well be in a name
                assert answer not in trials[k].name.lower(), f"{trials[k].name!r} gives its answer away"
            # Nor in other words: a name says what's set up, not what happens (#409 C-8).
            said = OUTCOME_WORDS.search(trials[k].name)
            assert said is None, f"{trials[k].name!r} says what happens ({said and said.group(0)!r}): name the setup"
        if not asked:
            continue
        page = getattr(w, attr)
        w.pages.setCurrentWidget(page)
        box = page.trials
        if box.after_play:  # a live page asks once the reader has handled it (`test_sdof_page`): here, at once
            box.after_play, box.played = False, True
        before = page.practice_state()
        for i in asked:
            box.combo.pick(trials[i].name)
            assert box.asking == i and box.ask.isVisible(), f"{attr}: {trials[i].name}"
            assert page.practice_state() == before, f"{attr}: set up before it was answered"
            p = trials[i].predict
            if isinstance(p, Placement):  # guessed on its plot: the guess stays there after it runs
                assert box.place.isVisible() and box.line is not None
                box.place_guess(p.answer)
                assert box.guessed is not None and box.line is None
                assert "Close" in box.verdict.text() and box.note.isVisible()
            else:
                box.answer(p.answer)
                assert "Right" in box.verdict.text() and box.note.isVisible()
            w.restore_defaults()
            before = page.practice_state()
        checked += 1
    assert checked >= 7  # the One mass group and the Gyroscope at least
