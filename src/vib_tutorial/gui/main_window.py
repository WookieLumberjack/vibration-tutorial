"""Main window: the course's pages in their groups (built on first use), and what spans them: the chain the pages
follow (the Simulation page's), Defaults, the Theory and Practice windows, the theme, the present mode and the keys."""

from __future__ import annotations

import time
from dataclasses import fields
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard, runtime_checkable

from PySide6 import QtCore, QtGui, QtWidgets

from ..core import (
    ChainSystem,
    ModalResult,
)
from . import (
    accessible,
    coalesce,
    crosshair,
    display_size,
    errors,
    frame_stats,
    links,
    notes_render,
    notes_view,
    plot_export,
    present,
    theming,
    weak,
)
from .challenge import ChallengeBox
from .ink import InkOverlay
from .ink import Mode as InkMode
from .layouts import ResetButton
from .lesson import RunButton, TabRoom, TrialBox, TrialCombo
from .notes_view import NotesBrowser
from .plot_export import install as install_plot_menus
from .simulation_page import SimulationPage
from .style import SYSTEM, colors
from .tab_groups import PageGroups
from .transport import TransportBar, transport_of
from .trial_overlay import TrialOverlay

if TYPE_CHECKING:
    from .practice import PracticeWindow
    from .theory_window import TheoryWindow


# The pages in reading order, in groups: (group title, ((attribute, tab title), ...)), from links.COURSE.
# All but the Simulation page are imported and built the first time they're opened (MainWindow._page).
LAST_PAGE = "window/page"  # the page showing, kept in the settings: the app reopens on it
PAGE_GROUPS = tuple((group, tuple((name, title) for name, title, _ in pages)) for group, pages in links.COURSE)
# The pages built on first use, in reading order: (attribute, tab title).
LAZY_PAGES = tuple(page for _, pages in PAGE_GROUPS for page in pages if page[0] != "sim_page")


# What the window asks of the pages it builds, besides being QWidgets.
@runtime_checkable
class Themed(Protocol):
    def apply_theme(self) -> None: ...


class ChainPage(Protocol):
    """A page that follows the chain (`_follows_chain`): it asks for the Simulation page's and is given it."""

    dof_requested: QtCore.SignalInstance
    edit_parameters: QtCore.SignalInstance

    def set_system(self, system: ChainSystem, result: ModalResult | None = None) -> None: ...


@runtime_checkable
class PracticePage(Protocol):
    """A page with practice problems: `load_setup` sets one up."""

    def load_setup(self, setup: dict) -> None: ...


@runtime_checkable
class DefaultsPage(Protocol):
    """A page *Defaults* can put back as it opened: `practice_state` read when it's built, `load_setup` given it."""

    def load_setup(self, setup: dict) -> None: ...

    def practice_state(self) -> Any: ...


@runtime_checkable
class SpecPage(Protocol):
    """A page with practice specs: `practice_state` reads its settings back, for the spec to check."""

    def practice_state(self) -> Any: ...


APP_TITLE = "Vibration Tutorial"


def window_title(name: str) -> str:
    """The main window's title while page `name` shows: "Single oscillator — Vibration Tutorial"."""
    return f"{links.PAGE_TITLES[name].removesuffix(' page')} — {APP_TITLE}"


def _follows_chain(page: QtWidgets.QWidget) -> TypeGuard[ChainPage]:
    """Whether a page shows the chain (and is given it); the rotor, Duffing & Mathieu, torsional and continuous
    systems pages have models of their own."""
    return getattr(page, "follows_chain", True)


THEME_TIP = (
    "<p>Colours of the whole app: plots, animation and controls.</p>"
    "<p><b>System</b> follows the desktop's light or dark setting, and switches with it.</p>"
)
SIZE_TIP = (
    "<p>How large the whole app draws: text, lines, plots and figures alike. For a projector or a TV across "
    "the room, try 150 %. The app restarts to change it, back on this page.</p>"
    "<p>The notes' text has a size of its own too (Ctrl+= and Ctrl+-).</p>"
)
DEFAULTS_TIP = (
    "<p>Put this page's controls back as they were when it opened (Ctrl+R): its motion at rest, waiting for "
    "Start. A page that shows the chain puts back the chain it opened with too, on every page.</p>"
)
PRESENT_TIP = (
    "Presenting: the pages' side panes are folded away (F5 steps through the levels). Click to show them all again."
)
THEORY_TIP = (
    "<p>Show this page's <i>Theory</i> tab: what the page shows, the ideas behind it, things to try, "
    "and its symbols. (F1)</p>"
    "<p>Ctrl+1 to Ctrl+6 switch between the groups of pages.</p>"
)
THEORY_WINDOW_TIP = (
    "<p>Every page's Theory in a window of its own, with the pages and their sections listed: it keeps "
    "its place as you move between pages, or follows them. (Ctrl+F1)</p>"
)
PRACTICE_TIP = (
    "<p>Practice problems on the pages, with new numbers every time: work out a number, predict what happens "
    "and then see it on the page, or tune the page until it meets a spec. (Ctrl+F2)</p>"
)
ABOUT_HTML = """<h3>Vibration Tutorial {version}</h3>
<p>An interactive teaching tool for mechanical vibration: a chain of masses, springs and dampers, and the
systems built from it. Start with the first page, <i>One oscillator → Single oscillator</i>, and read each page's
<i>Theory</i> tab (F1); the pages are in the order they are meant to be read. That page's notes say how a
page works.</p>
<p>The keys:</p>
{keys}
<p><b>For educational use only.</b> This program was written with an AI coding assistant (Claude Code).
Although it has an automated test suite, its results are <b>not guaranteed to be technically correct</b>:
the simulations, analyses and theory notes may contain errors. Check any calculation independently before
relying on it, and do not use it for design, certification, safety-related or other engineering
decisions.</p>
<p>Provided "as is", without warranty of any kind. Released under the MIT License.
<a href="https://github.com/WookieLumberjack/vibration-tutorial">github.com/WookieLumberjack/vibration-tutorial</a></p>
"""
INTRODUCED = "window/introduced"  # the pages whose Theory tab their first show opened (introduce_pages)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._introducing = False  # a page's first show opens its Theory tab (introduce_pages): off until run's
        self._arriving = False  # a page shown to be set up (_arrive): its first show leaves the tab be
        notes_render.install()  # the bundled fonts and the UI font, as make_app does (tests build a bare app)
        self.setWindowTitle(APP_TITLE)  # and the page's before it, as each shows (`_on_page_changed`)
        # --- the pages: the Simulation page, then the rest, which follow its chain. An empty placeholder
        # holds each of the rest until it's first opened (see _page).
        self.sim_page = SimulationPage()
        self.sim_page.chain_changed.connect(self._on_chain_changed)
        self.pages = PageGroups()
        self._built: dict[str, QtWidgets.QWidget] = {"sim_page": self.sim_page}
        self._opening: dict[str, Any] = {}  # each page built's practice_state then: what Defaults puts back
        self._opening_chain: dict[str, ChainSystem] = {}  # and the chain then, for a page that follows it
        self._placeholders: dict[str, QtWidgets.QWidget] = {}
        for group_title, pages in PAGE_GROUPS:
            tabs_in_group: list[tuple[QtWidgets.QWidget, str]] = []
            for name, title in pages:
                if name == "sim_page":
                    tabs_in_group.append((self.sim_page, title))
                    continue
                self._placeholders[name] = QtWidgets.QWidget()
                tabs_in_group.append((self._placeholders[name], title))
            self.pages.add_group(group_title, *tabs_in_group)
        # Built, the window shows the Simulation page (the tests and screenshots start there, without building
        # a lazy page); the app then opens on the course's first page (open_first_page).
        self.pages.setCurrentWidget(self.sim_page)
        self.pages.currentChanged.connect(self._on_page_changed)
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(theming.names())
        self.theme_combo.setCurrentText(theming.current_name())
        self.theme_combo.setToolTip(THEME_TIP)
        theme_label = QtWidgets.QLabel("Theme:")
        theme_label.setToolTip(THEME_TIP)
        self.size_combo = QtWidgets.QComboBox()
        self.size_combo.addItems([display_size.label(scale) for scale in display_size.SIZES])
        self.size_combo.setCurrentText(display_size.label(display_size.active()))
        size_label = QtWidgets.QLabel("Size:")
        for widget in (self.size_combo, size_label):
            widget.setToolTip(SIZE_TIP)
        if display_size.from_environment():  # QT_SCALE_FACTOR, set before launch, decides it
            self.size_combo.setEnabled(False)
            self.size_combo.setToolTip(f"Set by {display_size.ENV} in the environment.")
        corner = QtWidgets.QWidget()
        cl = QtWidgets.QHBoxLayout(corner)
        cl.setContentsMargins(6, 2, 6, 2)
        self.theory_button = QtWidgets.QToolButton()
        self.theory_button.setText("Theory")
        self.theory_button.setToolTip(THEORY_TIP)
        self.theory_button.clicked.connect(self.show_theory)
        about = QtWidgets.QToolButton()
        about.setText("About")
        about.setToolTip("Version, disclaimer and licence.")
        about.clicked.connect(self._about)
        self.theory_window_button = QtWidgets.QToolButton()
        self.theory_window_button.setText("Theory window")
        self.theory_window_button.setToolTip(THEORY_WINDOW_TIP)
        self.theory_window_button.clicked.connect(self.open_theory_window)
        self.theory_window: TheoryWindow | None = None  # built when first opened
        self.practice_button = QtWidgets.QToolButton()
        self.practice_button.setText("Practice")
        self.practice_button.setToolTip(PRACTICE_TIP)
        self.practice_button.clicked.connect(self.open_practice_window)
        self.practice_window: PracticeWindow | None = None  # built when first opened
        self.defaults_button = QtWidgets.QToolButton()
        self.defaults_button.setText("Defaults")
        self.defaults_button.setToolTip(DEFAULTS_TIP)
        self.defaults_button.clicked.connect(self.restore_defaults)
        # While presenting, the way back: F5 is easy to miss, and the level is kept for the next launch (#409 U-5)
        self.present_button = QtWidgets.QToolButton()
        self.present_button.setText("Show panes")
        self.present_button.setToolTip(PRESENT_TIP)
        self.present_button.clicked.connect(self.end_present)
        self.present_button.hide()
        cl.addWidget(self.present_button)
        cl.addWidget(self.defaults_button)
        cl.addWidget(self.theory_button)
        cl.addWidget(self.theory_window_button)
        cl.addWidget(self.practice_button)
        cl.addWidget(about)
        cl.addSpacing(8)
        cl.addWidget(theme_label)
        cl.addWidget(self.theme_combo)
        cl.addWidget(size_label)
        cl.addWidget(self.size_combo)
        theme_label.setBuddy(self.theme_combo)  # their names for a screen reader
        size_label.setBuddy(self.size_combo)
        cl.insertStretch(0)  # a row of its own, right-aligned: the groups' tabs need the window's width
        self.run_bar = TransportBar()  # the motion's controls, in one place on every page (#316 B13)
        self.run_bar.hold.clicked.connect(self.hold_curves)
        self.run_bar.unhold.clicked.connect(self.clear_held)
        central = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(central)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        self.error_banner = errors.ErrorBanner()  # what went wrong, once `gui.run()` installs the hook
        column.addWidget(self.error_banner)
        column.addWidget(self.pages, 1)
        # above the pages, but parented after them: the pages come first in the focus chain, so Space isn't Defaults'
        column.insertWidget(column.indexOf(self.pages), corner)
        column.addWidget(self.run_bar)
        self.setCentralWidget(central)
        self.resize(1700, 900)
        self.present_level = 0
        self._adopt("sim_page", self.sim_page)
        self._follow_transport()
        self._opening["sim_page"] = self.sim_page.practice_state()  # it shows as built
        self.setWindowTitle(window_title("sim_page"))
        if (level := present.saved()) != 0:
            self.set_present(level)
        QtGui.QShortcut(QtGui.QKeySequence("F1"), self, self.show_theory)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+F1"), self, self.open_theory_window)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+F"), self, self.find_in_notes)  # ⌘F on macOS
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+F2"), self, self.open_practice_window)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self, self.restore_defaults)
        QtGui.QShortcut(QtGui.QKeySequence("H"), self, self.hold_curves)
        QtGui.QShortcut(QtGui.QKeySequence("Shift+H"), self, self.clear_held)
        QtGui.QShortcut(QtGui.QKeySequence("F5"), self, self.step_present)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+PgDown"), self, self.next_page)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+PgUp"), self, self.previous_page)
        QtGui.QShortcut(QtGui.QKeySequence("Alt+PgDown"), self, self.next_trial)
        QtGui.QShortcut(QtGui.QKeySequence("Alt+PgUp"), self, self.previous_trial)
        self.ink = InkOverlay(self.pages)
        QtGui.QShortcut(QtGui.QKeySequence("F6"), self, self.toggle_pointer)
        QtGui.QShortcut(QtGui.QKeySequence("F7"), self, self.toggle_ink)
        self.trial_overlay = TrialOverlay(self.pages)  # a step's note, large, while presenting
        self.ink_escape = QtGui.QShortcut(QtGui.QKeySequence("Esc"), self, self.clear_ink)
        self.ink_escape.setEnabled(False)  # Esc is the dialogs' and popups' but while there's a pointer, ink or a note
        self.ink.changed.connect(self._on_ink_changed)
        self.trial_overlay.shown.connect(self._on_ink_changed)
        self.pages.currentChanged.connect(self._clear_ink)
        notes_view.add_zoom_keys(self)  # the notes' text size
        for i in range(min(len(PAGE_GROUPS), 9)):  # Ctrl+1 … Ctrl+9: a group, at the page it last showed
            QtGui.QShortcut(
                QtGui.QKeySequence(f"Ctrl+{i + 1}"),
                self,
                weak.slot(self, lambda self, i=i: self.pages.groups.setCurrentIndex(i)),
            )

        self.theme_combo.textActivated.connect(self._on_theme_picked)
        self.size_combo.activated.connect(self._on_size_picked)
        theming.application().styleHints().colorSchemeChanged.connect(self._on_desktop_scheme)
        self.frame_overlay = frame_stats.install(self)  # only with VIB_FRAME_STATS set

    # --------------------------------------------------------------- pages

    def _page(self, name: str) -> QtWidgets.QWidget:
        """The page `name` from LAZY_PAGES, built and put in its tab on first use."""
        page = self._built.get(name)
        if page is not None:
            return page
        page = self._built[name] = links.page_class(name)()
        # Put the page in its placeholder's tab (showing, if that was): only the real page is ever shown.
        holder = self._placeholders.pop(name)
        self.pages.replace(holder, page)
        self._adopt(name, page)
        holder.deleteLater()
        return page

    def _adopt(self, name: str, page: QtWidgets.QWidget) -> None:
        """What the window does for every page it builds: the chain, the plots' menus, crosshairs and names, the notes'
        links and closing line, the present mode, room for its tabs."""
        self._give_chain(page)
        install_plot_menus(page)
        crosshair.install(page)
        self._wire_links(page)
        accessible.name_plots(page)
        present.apply(page, self.present_level)
        if (panes := present.splitter(page)) is not None:
            TabRoom(panes)  # its tabs' names in full, whatever the font
        self._close_notes(name, page)
        for box in page.findChildren(TrialBox):  # what the reader tries there is kept (progress)
            box.page_name = name
            box.show_tried()
        for tabs in page.findChildren(QtWidgets.QTabWidget):  # a tab may hold the motion (Torsional's Start-up)
            tabs.currentChanged.connect(self._follow_transport)

    def _follow_transport(self, *_) -> None:
        """The run bar drives what's runnable on the page and tab showing."""
        page = self.pages.currentWidget()
        self.run_bar.follow(transport_of(page), page)

    def _close_notes(self, name: str, page: QtWidgets.QWidget) -> None:
        """The page's Theory tab ends with a link on to the next page."""
        notes = getattr(page, "notes", None) or getattr(page, "theory", None)
        if isinstance(notes, NotesBrowser):
            notes.footer = links.next_html(name)
            notes.refresh()

    def next_page(self) -> None:
        """Ctrl+PgDown: the next page in the course."""
        self._step_page(1)

    def previous_page(self) -> None:
        """Ctrl+PgUp: the page before."""
        self._step_page(-1)

    def next_trial(self) -> None:
        """Alt+PgDown: the page showing's next Thing to try."""
        self.step_trial(1)

    def previous_trial(self) -> None:
        """Alt+PgUp: the one before."""
        self.step_trial(-1)

    def step_trial(self, by: int) -> None:
        page = self.pages.currentWidget()
        combo = page.findChild(TrialCombo) if page is not None else None
        if combo is not None:
            combo.step(by)
            if self.present_level >= 1:  # its note is folded away with the controls: shown large
                assert page is not None, "a combo found on it"
                self.trial_overlay.show_for(page, combo)

    def _step_page(self, step: int) -> None:
        pages = links.order()
        name = self.page_name(self.pages.currentWidget())
        if name in pages and 0 <= (to := pages.index(name) + step) < len(pages):
            self.show_page(pages[to])

    def toggle_pointer(self) -> None:
        self.ink.toggle(InkMode.POINTER)

    def toggle_ink(self) -> None:
        self.ink.toggle(InkMode.INK)

    def clear_ink(self) -> None:
        """Esc: the ink gone, the pointer and ink off, a trial's large note hidden."""
        self.ink.clear()
        self.trial_overlay.dismiss()

    def _clear_ink(self, _index: int) -> None:
        self.ink.clear_ink()  # it was drawn on the page before
        self.trial_overlay.dismiss()  # and so was the note

    def _on_ink_changed(self, *_) -> None:
        self.ink_escape.setEnabled(self.ink.active or self.trial_overlay.isVisible())

    def step_present(self) -> None:
        """F5: the next of `present.LEVELS` on every page (the side panes folded away, then back)."""
        self.set_present((self.present_level + 1) % len(present.LEVELS))

    def end_present(self) -> None:
        """Every page's panes back (the corner's *Show panes*)."""
        self.set_present(0)

    def set_present(self, level: int) -> None:
        self.present_level = level
        present.save(level)
        self.present_button.setVisible(level != 0)
        self.present_button.setToolTip(f"{PRESENT_TIP} Now: {present.LEVELS[level]}.")
        if level == 0:
            self.trial_overlay.dismiss()  # the box's own note shows again
        for page in self._built.values():
            present.apply(page, level)

    def show_page(self, name: str) -> None:
        """Show page `name` (an attribute in PAGE_GROUPS), building it if it isn't yet."""
        self.pages.setCurrentWidget(self._page(name))

    def open_first_page(self) -> None:
        """Show the course's first page, where the app first opens."""
        self.show_page(PAGE_GROUPS[0][1][0][0])
        # The window was built showing the Simulation page: its group would open on it, not its first page.
        self.pages.rewind()

    def open_start_page(self) -> None:
        """Show the page the reader last had open (kept in the settings), or the course's first page."""
        name = theming.settings().value(LAST_PAGE)
        if name not in {page for _, pages in PAGE_GROUPS for page, _ in pages}:  # none yet, or since renamed
            self.open_first_page()
            return
        self.show_page(str(name))
        self.pages.rewind()  # the other groups open on their first page, as they do on the first run

    def _give_chain(self, page: QtWidgets.QWidget) -> None:
        """Wire a new page that follows the chain to the Simulation page's, and give it the chain."""
        if not _follows_chain(page):
            return
        params = self.sim_page.params
        page.dof_requested.connect(params.dof.setValue)
        page.edit_parameters.connect(weak.slot(self, lambda self: self.pages.setCurrentWidget(self.sim_page)))
        if hasattr(page, "system_requested"):
            page.system_requested.connect(params.set_system)
        page.set_system(self.sim_page.sim.system, self.sim_page.modal)

    def _on_chain_changed(self, system: ChainSystem, modal: ModalResult) -> None:
        for page in self._built.values():  # the rest are given the system when built
            if _follows_chain(page):
                page.set_system(system, modal)

    def restore_defaults(self) -> None:
        """The page showing's controls back as they were when it opened (`_opening`), its motion at rest and
        waiting for Start, its Things to try back at the prompt. A page that follows the chain gets the chain it
        opened with back too (on the Simulation page as well): its settings may make no sense on another."""
        page = self.pages.currentWidget()
        opening = self._opening.get(self.page_name(page) or "")
        if opening is None or not isinstance(page, DefaultsPage):
            return
        now = page.practice_state()
        setup = {f.name: getattr(opening, f.name) for f in fields(opening) if f.name not in ("name", "note")}
        if "tab" in setup:
            setup["tab"] = now.tab  # the tab the reader is on stays
        # The chain as it was first (a page works some settings out from it), and again after if the setup
        # changed it (a setup may describe only the page's own chain).
        chain = self._opening_chain.get(self.page_name(page) or "")
        params = self.sim_page.params
        with coalesce.batch():  # the page's redraws once, with the whole setup loaded
            if chain is not None:
                params.set_system(chain.copy())
            page.load_setup(setup)
            if chain is not None and not chain.same_as(self.sim_page.sim.system):
                params.set_system(chain.copy())
        if any(run.isChecked() or run.started for run in page.findChildren(RunButton)):
            for button in page.findChildren(ResetButton):  # beside its Start: back to t = 0, waiting for Start
                button.click()
        for combo in page.findChildren(TrialCombo):
            combo.unpick()
        for box in page.findChildren(TrialBox):
            box.show_note("")
        for challenge in page.findChildren(ChallengeBox):
            challenge.clear()
        plot_export.clear_all_held(page)

    def page_name(self, page: QtWidgets.QWidget | None) -> str | None:
        """The attribute in PAGE_GROUPS of a page built (the Simulation page is built with the window)."""
        return next((name for name, built in self._built.items() if built is page), None)

    def open_theory_window(self) -> TheoryWindow:
        """Show the Theory window (at the page showing's notes, the first time)."""
        from .theory_window import TheoryWindow

        if self.theory_window is None:
            self.theory_window = TheoryWindow(self.show_page, self, act=self.act)
        if self.theory_window.page is None:
            self.theory_window.show_page(self.page_name(self.pages.currentWidget()) or "sim_page")
        self.theory_window.show()
        self.theory_window.raise_()
        self.theory_window.activateWindow()
        return self.theory_window

    def open_practice_window(self) -> PracticeWindow:
        """Show the Practice window (at the page showing's problems, the first time, if it has any)."""
        from .practice import PracticeWindow

        if self.practice_window is None:
            self.practice_window = PracticeWindow(self.set_up_page, self._read_theory, self._practice_state, self)
        if self.practice_window.problem is None:
            self.practice_window.show_page(self.page_name(self.pages.currentWidget()) or "")
        self.practice_window.show()
        self.practice_window.raise_()
        self.practice_window.activateWindow()
        return self.practice_window

    def set_up_page(self, name: str, setup: dict) -> None:
        """Show page `name` set up as a practice problem's `setup` says."""
        self._arrive(name)
        page = self._page(name)
        assert isinstance(page, PracticePage), "only pages with problems are set up"
        with coalesce.batch():
            page.load_setup(setup)

    def _practice_state(self, name: str):
        """Page `name`'s settings, which a practice spec checks."""
        page = self._page(name)
        assert isinstance(page, SpecPage), "only pages with specs are checked"
        return page.practice_state()

    def _read_theory(self, name: str, anchor: str) -> None:
        """A hint's link: the Theory window at page `name`'s `anchor`."""
        self.open_theory_window().show_page(name, anchor)

    def find_in_notes(self) -> None:
        """The Theory window's search, ready to type (`TheoryWindow.ready_search`): a page has no find of its own, and
        the window's searches every page's notes."""
        self.open_theory_window().ready_search()

    def show_theory(self) -> QtWidgets.QTextBrowser | None:
        """Bring the page showing's Theory tab to the front (and any group of tabs it sits in); its browser."""
        page = self.pages.currentWidget()
        assert page is not None, "a page always shows"
        browser = None
        for tabs in page.findChildren(QtWidgets.QTabWidget):
            for i in range(tabs.count()):
                if tabs.tabText(i) == "Theory":
                    tabs.setCurrentIndex(i)
                    found = tabs.widget(i)
                    if not isinstance(found, QtWidgets.QTextBrowser) and found is not None:
                        found = found.findChild(QtWidgets.QTextBrowser)
                    browser = found or browser
        return browser

    def go_to(self, name: str, anchor: str = "") -> None:
        """Show page `name` (an attribute in PAGE_GROUPS) at its Theory, scrolled to `anchor`."""
        self.show_page(name)
        browser = self.show_theory()
        if browser is not None and anchor:
            QtCore.QTimer.singleShot(0, browser, lambda: browser.scrollToAnchor(anchor))  # once it's laid out

    def act(self, scheme: str, name: str, what: str) -> None:
        """A goal's link (notes_view.ACTIONS): `trial` sets up page `name`'s trial number `what` as its Things to try
        would (asking its prediction first), `practice` asks its problem `what` in the Practice window."""
        if scheme == "trial":
            self._arrive(name)
            page = self.pages.currentWidget()
            assert page is not None, "show_page shows it"
            combo = page.findChild(TrialCombo)
            assert combo is not None, f"{name} has Things to try"
            combo.pick(combo.names[int(what) - 1])
        elif scheme == "practice":
            self.open_practice_window().show_problem(name, what)

    def _wire_links(self, page: QtWidgets.QWidget) -> None:
        for browser in page.findChildren(QtWidgets.QTextBrowser):
            notes_view.wire(browser, self.go_to, self.act)

    def _about(self) -> None:
        from importlib.metadata import PackageNotFoundError, version

        try:
            v = version("vib-tutorial")
        except PackageNotFoundError:
            v = ""
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("About Vibration Tutorial")
        box.setTextFormat(QtCore.Qt.TextFormat.RichText)
        box.setText(ABOUT_HTML.format(version=v, keys=links.keys_html()))
        box.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextBrowserInteraction)
        box.exec()

    def hold_curves(self) -> None:
        """Hold the curves on every plot the page showing shows, in grey, to compare a change against (H)."""
        page = self.pages.currentWidget()
        assert page is not None, "a page always shows"
        plot_export.hold_showing(page)

    def clear_held(self) -> None:
        """Take the page's held curves off (Shift+H)."""
        page = self.pages.currentWidget()
        assert page is not None, "a page always shows"
        plot_export.clear_all_held(page)

    def introduce_pages(self) -> None:
        """From now on, a page's first show (ever: kept in the settings, INTRODUCED) opens its Theory tab at the top,
        where its goals link to its trials and problems; later visits keep the tab the reader left. The first run
        opens on the first page's, its Start here and how a page works. `run` turns it on, after the page it opens
        on shows; a bare MainWindow (tests, screenshots) doesn't, nor do the Theory window, Defaults, a goal's
        trial or a practice setup, which want the page's plots (`_arrive`)."""
        self._introducing = True
        self._introduce(self.page_name(self.pages.currentWidget()))

    def _introduce(self, name: str | None) -> None:
        if not self._introducing or name is None:
            return
        store = theming.settings()
        seen = set(filter(None, str(store.value(INTRODUCED, "")).split(",")))  # a string: a one-item list isn't
        if name in seen:  # kept as a list on every platform
            return
        if not self._arriving:
            browser = self.show_theory()
            if browser is not None:
                browser.verticalScrollBar().setValue(0)
        store.setValue(INTRODUCED, ",".join(sorted(seen | {name})))

    def _arrive(self, name: str) -> None:
        """Show page `name` for something that sets it up (a goal's trial, a practice setup): on the tab the
        setup picks, not its Theory, though this counts as its first show."""
        self._arriving = True
        try:
            self.show_page(name)
        finally:
            self._arriving = False

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        for window in (self.theory_window, self.practice_window):
            if window is not None:
                window.close()  # it closes with this window
        super().closeEvent(event)

    def _on_page_changed(self, index: int) -> None:
        widget = self.pages.widget(index)
        for name, holder in list(self._placeholders.items()):
            if holder is widget:
                # Building a page and working out what it first shows takes up to a second (the rotor's
                # analysis, a first record): say so with the cursor until the event loop is back.
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
                QtCore.QTimer.singleShot(0, QtWidgets.QApplication.restoreOverrideCursor)
                self._page(name)
        page, showing = self.pages.currentWidget(), self.page_name(self.pages.currentWidget())
        if showing is not None and showing not in self._opening and isinstance(page, DefaultsPage):
            # What Defaults puts back: as it first shows (a page may settle its controls in its showEvent).
            self._opening[showing] = page.practice_state()
            if _follows_chain(page):
                self._opening_chain[showing] = self.sim_page.sim.system.copy()
        self._follow_transport()
        self._introduce(showing)
        if showing is not None:
            self.setWindowTitle(window_title(showing))
            theming.settings().setValue(LAST_PAGE, showing)
            for window in (self.theory_window, self.practice_window):
                if window is not None and window.isVisible():
                    window.follow_page(showing)

    # --------------------------------------------------------------- theme
    def set_theme(self, name: str) -> None:
        """Switch every page to a theme (a name from theming.names()) and redraw."""
        theming.apply(name)
        self.theme_combo.setCurrentText(name)
        self._restyle()

    def _on_theme_picked(self, name: str) -> None:
        theming.save_name(name)
        self.set_theme(name)
        scale = display_size.projector_offer(self, name)  # its lines grow, its text doesn't: Size grows both
        if scale is None:
            return
        size = display_size.label(scale)
        answer = QtWidgets.QMessageBox.question(
            self,
            "Size",
            f"The Projector theme draws thicker lines, but its text stays small. Show everything at {size} too? "
            f"The app restarts to change size. (Size, top right, sets it later.)",
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self.size_combo.setCurrentText(size)
            display_size.save(scale)
            display_size.restart()

    def _on_size_picked(self, index: int) -> None:
        """A new size: saved and, if the reader agrees, shown at once by a restart (see `display_size`)."""
        scale = display_size.SIZES[index]
        now = display_size.label(display_size.active())
        if not display_size.fits(self, scale):
            QtWidgets.QMessageBox.information(
                self,
                "Size",
                f"At {display_size.label(scale)} the window needs a screen at least "
                f"{display_size.screen_width_needed(self, scale)} px wide (at 100 %), more than this one has.",
            )
            self.size_combo.setCurrentText(now)
            return
        display_size.save(scale)
        if scale == display_size.active():
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Size",
            f"Restart the app now to show it at {display_size.label(scale)}? (Otherwise it opens at that size "
            "next time.)",
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            display_size.restart()

    def _on_desktop_scheme(self) -> None:
        # Checked once the change (which theming.apply also causes) has settled.
        QtCore.QTimer.singleShot(0, self._follow_desktop)

    def _follow_desktop(self) -> None:
        # Only while following the desktop: an explicit theme sets the scheme itself.
        if theming.current_name() == SYSTEM and colors.name != theming.system_theme().name:
            self.set_theme(SYSTEM)

    def _restyle(self) -> None:
        theming.restyle(self)
        for page in self._built.values():
            assert isinstance(page, Themed), "every page follows the theme"
            page.apply_theme()
        plot_export.restyle(self)
        for window in (self.theory_window, self.practice_window):
            if window is not None:
                window.apply_theme()
        for browser in self.findChildren(NotesBrowser):
            browser.refresh()  # their equations and figures, in the new colours

    # --------------------------------------------------------------- events
    def event(self, event: QtCore.QEvent) -> bool:
        # With frame stats on, time each repaint of the window (every widget in it).
        if frame_stats.STATS is None or event.type() != QtCore.QEvent.Type.UpdateRequest:
            return super().event(event)
        start = time.perf_counter()
        try:
            return super().event(event)
        finally:
            frame_stats.STATS.paint(time.perf_counter() - start)


def _lazy_page(name: str) -> property:
    return property(lambda self: self._page(name), doc=f"The {links.title(name)} page, built on first use.")


# `window.sdof_page` and the rest: each of links.PAGE_CLASSES's pages, built when first asked for.
for _name in links.PAGE_CLASSES:
    setattr(MainWindow, _name, _lazy_page(_name))
