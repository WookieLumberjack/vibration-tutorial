"""Fatigue page: a random stress counted by rainflow, against the narrow-band and Dirlik estimates.

Left: the stress PSD (one or two resonant peaks), the S–N curve and the record. Centre: the PSD and a stretch of
the record with the reversals rainflow counts. Right: the range distributions (the record's rainflow histogram,
Rayleigh's and Dirlik's), where the damage comes from, the S–N curve, and the Theory. The page has its own model
(core/fatigue.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets

from ..core.fatigue import (
    Curve,
    Record,
    Stress,
    dirlik_damage,
    dirlik_range_pdf,
    life_text,
    narrow_band_damage,
    rainflow,
    rates,
    rayleigh_range_pdf,
    record,
    reversals,
)
from ..core.practice import Problem, Question, Requirement, at_least, at_most, kept, number, pick, rounded
from . import settle_plots
from .lesson import (
    Form,
    TrialBox,
    column,
    connect_all,
    note_label,
    plot,
    readout_label,
    set_quietly,
    stack,
    theory,
    three_panes,
    trials_html,
)
from .links import load_notes, page_link
from .style import colors
from .theming import add_legend

TAB_RANGES, TAB_DAMAGE, TAB_SN = range(3)
SHOWN = 4.0  # periods of the slower peak drawn of the record
BINS = 40


@dataclass(frozen=True)
class FatigueTrial:
    name: str
    note: str
    stress: Stress = Stress()  # noqa: RUF009 (frozen, so sharing one is safe)
    b: float = 3.0
    duration: float = 600.0
    seed: int = 1
    tab: int = TAB_RANGES
    s_ref: float = 80.0
    n_ref: float = 2e6


TRIALS = (
    FatigueTrial(
        "One mode: a narrow band",
        "A 2 Hz mode with ζ = 2 %, 20 MPa RMS, on a welded detail (S–N slope 3, 80 MPa range at two million "
        "cycles). The band is narrow (α₂ = 0.95): nearly every up-crossing is one cycle, with Rayleigh-distributed "
        "amplitudes. The narrow-band formula, Dirlik's and Miner's sum over the record's rainflow cycles agree "
        "within 1.5 %: a life of 25 days.",
    ),
    FatigueTrial(
        "Two modes: the narrow-band formula is pessimistic",
        "Add a 15 Hz mode with 10 MPa RMS. The record now has small fast cycles riding on the slow ones, α₂ = "
        "0.46. The narrow-band formula counts every up-crossing (6.9 a second) as a cycle of the whole σ's "
        "Rayleigh, and overestimates the damage 1.6 times; Dirlik, fitted to rainflow counts, is 8 % under the "
        "record's.",
        stress=Stress(sigma2=10.0),
    ),
    FatigueTrial(
        "Rainflow pairs the ripples",
        "The same record, zoomed in. Rainflow treats the stress as rain running down a pagoda roof: each small "
        "ripple closes a small cycle, and the slow swing underneath is still counted at its full range. The "
        "Ranges tab shows the two families, many small ranges and the large ones, which Rayleigh's single hump "
        "can't describe.",
        stress=Stress(sigma2=10.0),
        duration=60.0,
        tab=TAB_RANGES,
    ),
    FatigueTrial(
        "A steeper S–N curve",
        "The two modes, on an S–N slope of 5 (closer to a machined part). Damage goes as S⁵, so the few largest "
        "ranges do nearly all of it (the Damage tab), and the estimates part further: the narrow-band formula is "
        "1.6 times the record's, Dirlik 18 % under (15 % under the average of twenty such records).",
        stress=Stress(sigma2=10.0),
        b=5.0,
        tab=TAB_DAMAGE,
    ),
    FatigueTrial(
        "Double the stress, eight times the damage",
        "One mode at 40 MPa RMS instead of 20. With slope 3 the damage rate goes up 2³ = 8 times: the life "
        "falls from 25 days to 3.1. A load that is 10 % high costs a third of the life (1.1³ = 1.33); the RMS "
        "has to be right before anything else is.",
        stress=Stress(sigma1=40.0),
        tab=TAB_SN,
    ),
    FatigueTrial(
        "A broad peak",
        "One mode but heavily damped, ζ = 30 %: a broad band, α₂ = 0.64. The record's cycles are irregular, "
        "with peaks below zero and troughs above it, and the narrow-band formula is 15 % high, Dirlik within "
        "2 %.",
        stress=Stress(zeta1=0.3),
    ),
    FatigueTrial(
        "A short record scatters",
        "Two modes on slope 5, but only 20 s of record. Rainflow's damage depends on which seed made the record: "
        "over seeds 1 to 10 it runs from 0.67 to 1.53 times the 600 s record's, because a few large cycles do "
        "most of the damage and a short record may or may not catch them. The spectral methods give one number "
        "from the PSD: a long record, or an average over many, is what they stand for.",
        stress=Stress(sigma2=10.0),
        b=5.0,
        duration=20.0,
        seed=1,
    ),
)

# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.
DAY = 86400.0


def _basquin(rng: np.random.Generator) -> Question:
    s_ref, b = pick(rng, (50, 80, 100, 125)), pick(rng, (3, 4, 5))
    S = pick(rng, (0.5, 0.8, 1.5, 2)) * s_ref
    N = float(Curve(b, s_ref).cycles(S)) / 1e6
    return Question(
        f"A detail survives two million cycles of a {s_ref:g} MPa range, on an S–N slope of {b:g}. How many million "
        f"cycles of a {S:g} MPa range does it survive?",
        N,
        f"N = N<sub>ref</sub>(S/S<sub>ref</sub>)<sup>−b</sup> = 2 million × ({S:g}/{s_ref:g})<sup>−{b:g}</sup> = "
        f"{number(N)} million cycles.",
        unit="million cycles",
        setup=dict(b=b, s_ref=s_ref, tab=TAB_SN),
    )


def _miner(rng: np.random.Generator) -> Question:
    curve = Curve()
    ranges = (pick(rng, (40, 60)), pick(rng, (80, 100)), pick(rng, (120, 160)))
    counts = (pick(rng, (1, 2, 5)) * 1e6, pick(rng, (1, 2, 5)) * 1e5, pick(rng, (1, 2, 5)) * 1e4)
    lives = [float(curve.cycles(S)) for S in ranges]
    D = sum(n / N for n, N in zip(counts, lives))
    blocks = ", ".join(f"{n / 1e6:g} million cycles of {S:g} MPa" for S, n in zip(ranges, counts))
    terms = " + ".join(f"{n / 1e6:g}/{number(N / 1e6)}" for n, N in zip(counts, lives))
    return Question(
        f"A welded detail (slope 3, 80 MPa range at two million cycles) takes {blocks}. What is its Miner's sum?",
        D,
        f"Each range's life is N = 2 million × (80/S)³; D = Σ n/N, in millions of cycles: {terms} = {number(D)}"
        + (": past 1, it has failed." if D > 1 else "."),
    )


def _narrow_life(rng: np.random.Generator) -> Question:
    nu0, sigma = pick(rng, (1, 2, 5, 10)), pick(rng, (10, 15, 20, 30))
    curve = Curve()
    rate = nu0 * (2 * math.sqrt(2) * sigma) ** 3 * math.gamma(2.5) / curve.K
    return Question(
        f"A narrow-band stress at {nu0:g} Hz, {sigma:g} MPa RMS, on a welded detail (slope 3, 80 MPa range at two "
        "million cycles). By the narrow-band formula, how many days does it last?",
        1 / rate / DAY,
        f"Each up-crossing is a cycle, its amplitude Rayleigh: the damage a second is ν₀(2√2σ)<sup>b</sup>Γ(1 + b/2)/K "
        f"= {nu0:g} × {number((2 * math.sqrt(2) * sigma) ** 3)} × {number(math.gamma(2.5))}/{number(curve.K)} = "
        f"{number(rate)}, so the life is {number(1 / rate)} s, {number(1 / rate / DAY)} days.",
        unit="days",
        setup=dict(stress=Stress(f1=nu0, sigma1=sigma), tab=TAB_SN),
    )


def _rainflow(rng: np.random.Generator) -> Question:
    top = pick(rng, (80, 100, 120))
    low, high = pick(rng, (20, 30, 40)), pick(rng, (60, 70))
    bottom = pick(rng, (-40, -20, 0))
    x = np.array([0.0, top, low, high, bottom])
    fall = f"{bottom:g}".replace("-", "−")
    ranges, counts = rainflow(x)
    full = float(ranges[counts == 1.0][0])
    halves = sorted(float(r) for r in ranges[counts == 0.5])
    return Question(
        f"Rainflow-count the stress history 0, {top:g}, {low:g}, {high:g}, {fall} MPa. What range has the one "
        "full cycle it finds?",
        full,
        f"The ripple {low:g} → {high:g} is no larger than the swing before it, and the fall to {fall} goes past "
        f"it: it closes a full cycle of {number(full)} MPa and is taken out. What's left, 0, {top:g}, {fall}, is "
        f"two half cycles, {' and '.join(number(h) for h in halves)} MPa: the big swing still counted whole.",
        unit="MPa",
    )


def _overload(rng: np.random.Generator) -> Question:
    high, b = pick(rng, (5, 10, 15, 20)), pick(rng, (3, 4, 5))
    share = 100 / (1 + high / 100) ** b
    return Question(
        f"The stress on a part turns out {high:g} % higher than designed for, its S–N slope {b:g}. What percentage "
        "of the life it was designed for does it have?",
        share,
        f"The damage goes as S<sup>b</sup>, so the life as S<sup>−b</sup>: 1/{number(1 + high / 100)}"
        f"<sup>{b:g}</sup> = {number(share)} %.",
        unit="%",
        tolerance=0.01,
    )


def _second_mode(rng: np.random.Generator) -> Question:
    return Question(
        "A narrow-band stress gains a second, faster mode with half its RMS. Against the record's rainflow count, "
        "the narrow-band formula's damage comes out…",
        0,
        "…too high (1.6 times here): it counts every up-crossing as a cycle of the whole σ's Rayleigh, but the fast "
        "mode's crossings are mostly small ripples on the slow swing.",
        choices=("Too high", "About right", "Too low"),
        setup=dict(stress=Stress(sigma2=10.0), tab=TAB_DAMAGE),
        predict=True,
    )


def _steep_slope(rng: np.random.Generator) -> Question:
    return Question(
        "On a steep S–N curve (slope 5 or more), which of a random stress's cycles do most of the damage?",
        0,
        "The few largest: each cycle's damage goes as S⁵, so a range twice as large does 32 times the damage (the "
        "Damage by range tab).",
        choices=(
            "The few largest",
            "The many small ones",
            "The ones near the RMS, evenly",
        ),
        setup=dict(stress=Stress(sigma2=10.0), b=5.0, tab=TAB_DAMAGE),
        predict=True,
    )


def _short_record(rng: np.random.Generator) -> Question:
    return Question(
        "Only 20 s of the stress is recorded, on a slope of 5. What does the rainflow damage do as the record's seed "
        "changes?",
        0,
        "It scatters widely (0.67 to 1.53 times the long record's over ten seeds): a few large cycles do most of the "
        "damage, and a short record may or may not catch them. The spectral methods give one number.",
        choices=("It scatters widely", "It stays within a few percent", "It's always low", "It's always high"),
        setup=dict(stress=Stress(sigma2=10.0), b=5.0, duration=20.0, seed=2),
        predict=True,
    )


def _damping_spec(rng: np.random.Generator) -> Question:
    base = Stress(sigma2=10.0)  # mode 1 at 20 MPa with ζ = 2 %; mode 2 at 15 Hz, 10 MPa
    zd = pick(rng, (0.04, 0.05, 0.06, 0.08))
    curve = Curve()

    def miles(zeta: float) -> float:
        return base.sigma1 * math.sqrt(base.zeta1 / zeta)

    def life(zeta: float, sigma: float) -> float:
        return 1 / dirlik_damage(Stress(base.f1, zeta, sigma, base.f2, base.zeta2, base.sigma2), curve) / DAY

    sd = round(miles(zd), 1)
    target = rounded(life(zd, sd) / 1.1, 2)
    most = rounded(1.25 * zd, 2)
    lo, hi = base.zeta1, zd
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if life(mid, miles(mid)) < target else (lo, mid)

    def goal(t: FatigueTrial):
        s = t.stress
        follows = abs(s.sigma1 / miles(s.zeta1) - 1) <= 0.015
        return (
            kept("Mode 1's frequency", s.f1, base.f1, "Hz"),
            kept("Mode 2's frequency", s.f2, base.f2, "Hz"),
            kept("Mode 2's ζ", s.zeta2, base.zeta2),
            kept("Mode 2's RMS", s.sigma2, base.sigma2, "MPa"),
            kept("The slope", t.b, curve.b),
            kept("S_ref", t.s_ref, curve.s_ref, "MPa"),
            kept("N_ref", t.n_ref, curve.n_ref, "cycles"),
            Requirement(
                f"Mode 1's RMS follows its damping: 20 √(0.02/ζ₁) = {number(miles(s.zeta1))} MPa",
                f"{number(s.sigma1)} MPa",
                follows,
            ),
            at_most("Mode 1's ζ", s.zeta1, most),
            at_least("Dirlik's life", life(s.zeta1, s.sigma1), target, "days"),
        )

    return Question(
        f"A welded detail (slope 3, 80 MPa at two million cycles) sees a 2 Hz mode at 20 MPa RMS with ζ = 0.02 and a "
        f"15 Hz mode at 10 MPa. Damping added to the 2 Hz mode lowers its RMS as 1/√ζ (Miles), its peak's "
        f"area the same over a wider band. Choose its ζ, no more than {number(most)}, and set its RMS to match, so "
        f"that Dirlik's life is {number(target, 2)} days or more.",
        0,
        f"σ₁ = 20 √(0.02/ζ₁). The damage would fall as ζ<sup>−1.5</sup> for mode 1 alone, but the 15 Hz mode's "
        f"cycles don't change, so the returns diminish: ζ₁ from about {number(hi)} meets it (ζ₁ = {zd:g} gives "
        f"σ₁ = {sd:g} MPa and {number(life(zd, sd))} days).",
        setup=dict(stress=base, tab=TAB_DAMAGE),
        goal=goal,
        design=dict(stress=Stress(base.f1, zd, sd, base.f2, base.zeta2, base.sigma2), tab=TAB_DAMAGE),
    )


PROBLEMS = tuple(
    Problem(key, title, "fatigue_page", f"See {page_link('fatigue_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("basquin", "Cycles to failure", "sn", "The S–N curve", _basquin),
        ("miner", "Miner's sum", "miner", "Miner's rule", _miner),
        ("narrow-life", "A narrow band's life", "spectral", "Damage from the PSD", _narrow_life),
        ("rainflow", "Rainflow by hand", "rainflow", "Rainflow counting", _rainflow),
        ("overload", "A higher stress", "sn", "The S–N curve", _overload),
        ("second-mode", "Predict: a second mode", "spectral", "Damage from the PSD", _second_mode),
        ("steep-slope", "Predict: a steep curve", "sn", "The S–N curve", _steep_slope),
        ("short-record", "Predict: a short record", "record", "The record", _short_record),
        ("damping-spec", "Spec: damping for a life", "spectral", "Damage from the PSD", _damping_spec),
    )
)

TIPS = {
    "f": "<p>A resonance in the stress response: its frequency.</p>",
    "zeta": "<p>Its damping ratio: the band's width.</p>",
    "sigma": "<p>Its RMS stress (0: leave this peak out).</p>",
    "b": "<p>The S–N curve's slope: N ∝ S<sup>−b</sup>. About 3 for welded joints, 5 to 10 for smooth machined "
    "parts.</p>",
    "s_ref": "<p>The stress range the detail survives for N<sub>ref</sub> cycles (a weld detail's category).</p>",
    "n_ref": "<p>The cycles that go with S<sub>ref</sub>.</p>",
    "duration": "<p>How long a record to synthesize and count.</p>",
    "seed": "<p>The random phases' seed.</p>",
}
PSD_TIP = "<p>The stress PSD (one-sided, MPa²/Hz).</p>"
RECORD_TIP = "<p>The start of the synthesized record, with the reversals rainflow counts.</p>"
RANGES_TIP = (
    "<p>The record's rainflow ranges as a density (bars), Rayleigh's (narrow-band) and Dirlik's "
    "range distributions.</p>"
)
DAMAGE_TIP = "<p>Where the damage comes from: each distribution times S<sup>b</sup>, normalized, against the range.</p>"
SN_TIP = "<p>The S–N curve (log–log) and the record's largest rainflow ranges against the cycles they took.</p>"

NOTES_HTML = load_notes("fatigue", trials=trials_html(TRIALS))


class FatiguePage(QtWidgets.QWidget):
    """Controls on the left, the PSD and the record in the centre, the distributions on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.stress = Stress()
        self.curve = Curve()
        self.rec: Record | None = None

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Each sets up the stress, the S–N curve and the record to show one idea from the Theory tab.</p>",
        )
        self.trials.picked.connect(self.load_trial)

        st = Form("Stress PSD")
        self.f1 = st.number("Peak 1 at:", 0.1, 1000.0, 2.0, 2, " Hz", TIPS["f"])
        self.zeta1 = st.number("its ζ:", 0.001, 0.9, 0.02, 3, "", TIPS["zeta"])
        self.sigma1 = st.number("its RMS:", 0.0, 1000.0, 20.0, 1, " MPa", TIPS["sigma"])
        self.f2 = st.number("Peak 2 at:", 0.1, 1000.0, 15.0, 2, " Hz", TIPS["f"])
        self.zeta2 = st.number("its ζ:", 0.001, 0.9, 0.02, 3, "", TIPS["zeta"])
        self.sigma2 = st.number("its RMS:", 0.0, 1000.0, 0.0, 1, " MPa", TIPS["sigma"])
        self.stress_info = note_label()
        st.row(self.stress_info)

        sn = Form("S–N curve")
        self.b = sn.number("Slope b:", 1.0, 20.0, 3.0, 1, "", TIPS["b"])
        self.s_ref = sn.number("Range S_ref:", 1.0, 2000.0, 80.0, 1, " MPa", TIPS["s_ref"])
        self.n_ref = sn.number("at N_ref:", 1e3, 1e9, 2e6, 0, " cycles", TIPS["n_ref"])

        rec = Form("Record")
        self.duration = rec.number("Length:", 5.0, 3600.0, 600.0, 0, " s", TIPS["duration"])
        self.seed = rec.number("Seed:", 1, 9999, 1, 0, "", TIPS["seed"])

        connect_all(
            (self.f1, self.zeta1, self.sigma1, self.f2, self.zeta2, self.sigma2, self.duration, self.seed),
            self._on_stress,
        )
        connect_all((self.b, self.s_ref, self.n_ref), self._on_curve)

        # --- centre
        self.psd_plot = plot("Stress PSD", "MPa²/Hz", "Frequency", "Hz", log_y=True, height=130)
        self.psd_plot.setToolTip(PSD_TIP)
        self.psd_item = pg.PlotDataItem()
        self.psd_plot.addItem(self.psd_item)
        self.rec_plot = plot("Stress", "MPa", "Time", "s", height=150)
        self.rec_plot.setToolTip(RECORD_TIP)
        self.rec_item = pg.PlotDataItem()
        self.rev_item = pg.PlotDataItem(pen=None, symbol="o", symbolSize=5)
        for item in (self.rec_item, self.rev_item):
            self.rec_plot.addItem(item)
        self.readout = readout_label()

        # --- right
        self.ranges_plot = plot("Density (1/MPa)", None, "Stress range S", "MPa", height=200)
        self.ranges_plot.setToolTip(RANGES_TIP)
        add_legend(self.ranges_plot.getPlotItem(), offset=(-10, 10))
        self.hist_item = pg.PlotDataItem(name="record (rainflow)", stepMode="center", fillLevel=0)
        self.ray_item = pg.PlotDataItem(name="Rayleigh (narrow band)")
        self.dirlik_item = pg.PlotDataItem(name="Dirlik")
        for item in (self.hist_item, self.ray_item, self.dirlik_item):
            self.ranges_plot.addItem(item)
        self.dmg_plot = plot("Share of the damage (1/MPa)", None, "Stress range S", "MPa", height=200)
        self.dmg_plot.setToolTip(DAMAGE_TIP)
        add_legend(self.dmg_plot.getPlotItem(), offset=(-10, 10))
        self.dhist_item = pg.PlotDataItem(name="record (rainflow)", stepMode="center", fillLevel=0)
        self.dray_item = pg.PlotDataItem(name="Rayleigh (narrow band)")
        self.ddirlik_item = pg.PlotDataItem(name="Dirlik")
        for item in (self.dhist_item, self.dray_item, self.ddirlik_item):
            self.dmg_plot.addItem(item)
        self.sn_plot = plot("Stress range S", "MPa", "Cycles to failure N", None, log_x=True, log_y=True, height=200)
        self.sn_plot.setToolTip(SN_TIP)
        add_legend(self.sn_plot.getPlotItem(), offset=(-10, 10))
        self.sn_item = pg.PlotDataItem(name="S–N curve")
        self.cycles_item = pg.PlotDataItem(name="the record's cycles (count above S)")
        for item in (self.sn_item, self.cycles_item):
            self.sn_plot.addItem(item)
        self.tab_note = note_label()

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(column(self.ranges_plot, stretch=False), "Ranges")
        self.tabs.addTab(column(self.dmg_plot, stretch=False), "Damage by range")
        self.tabs.addTab(column(self.sn_plot, stretch=False), "S–N curve")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._draw_tab)
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.tabs, 1)
        rv.addWidget(self.tab_note)

        three_panes(
            self,
            column(self.trials, st, sn, rec),
            stack((self.psd_plot, 2), (self.rec_plot, 3), (self.readout, 0)),
            right,
        )
        self.apply_theme()
        self._on_stress()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(FatigueTrial("", "", **setup))

    def practice_state(self) -> FatigueTrial:
        """The page's settings, as a trial: what a practice spec checks."""
        return FatigueTrial(
            "",
            "",
            Stress(
                self.f1.value(),
                self.zeta1.value(),
                self.sigma1.value(),
                self.f2.value(),
                self.zeta2.value(),
                self.sigma2.value(),
            ),
            self.b.value(),
            self.duration.value(),
            int(self.seed.value()),
            self.tabs.currentIndex(),
            self.s_ref.value(),
            self.n_ref.value(),
        )

    def apply_trial(self, t: FatigueTrial) -> None:
        s = t.stress
        for box, value in (
            (self.f1, s.f1),
            (self.zeta1, s.zeta1),
            (self.sigma1, s.sigma1),
            (self.f2, s.f2),
            (self.zeta2, s.zeta2),
            (self.sigma2, s.sigma2),
            (self.b, t.b),
            (self.s_ref, t.s_ref),
            (self.n_ref, t.n_ref),
            (self.duration, t.duration),
            (self.seed, t.seed),
        ):
            set_quietly(box, value)
        if t.note:
            self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self._on_stress()

    def _on_stress(self) -> None:
        self.stress = Stress(
            self.f1.value(),
            self.zeta1.value(),
            self.sigma1.value(),
            self.f2.value(),
            self.zeta2.value(),
            self.sigma2.value(),
        )
        if not self.stress.peaks():
            self.rec = None
        else:
            self.rec = record(self.stress, self.duration.value(), int(self.seed.value()))
        self._on_curve()

    def _on_curve(self) -> None:
        self.curve = Curve(self.b.value(), self.s_ref.value(), self.n_ref.value())
        self.redraw()

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.psd_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.rec_item.setPen(pg.mkPen(c.mass[0], width=1.2))
        self.rev_item.setSymbolBrush(pg.mkBrush(c.force))
        self.rev_item.setSymbolPen(None)
        for hist in (self.hist_item, self.dhist_item):
            hist.setPen(pg.mkPen(c.grey, width=1))
            hist.setBrush(pg.mkBrush(c.band_shade))
        for item, color in (
            (self.ray_item, c.mass[1]),
            (self.dray_item, c.mass[1]),
            (self.dirlik_item, c.mass[2]),
            (self.ddirlik_item, c.mass[2]),
        ):
            item.setPen(pg.mkPen(color, width=2))
        self.sn_item.setPen(pg.mkPen(c.strong, width=2))
        self.cycles_item.setPen(pg.mkPen(c.force, width=2))
        if hasattr(self, "notes"):
            self.redraw()

    def redraw(self) -> None:
        self._draw_main()
        self._draw_tab()
        settle_plots(self)

    def _draw_main(self) -> None:
        s, rec = self.stress, self.rec
        if rec is None:
            for item in (self.psd_item, self.rec_item, self.rev_item):
                item.setData([], [])
            self.readout.setText("Give a peak an RMS stress.")
            self.stress_info.setText("")
            return
        f = s.grid()
        G = s.psd(f)
        keep = G > G.max() * 1e-8
        self.psd_item.setData(f[keep], G[keep])
        r = rates(s)
        self.stress_info.setText(
            f"σ = {r.sigma:.3g} MPa; {r.nu0:.3g} up-crossings and {r.nup:.3g} peaks a second; α₂ = {r.alpha2:.2f}."
        )
        span = min(SHOWN / min(p[0] for p in s.peaks()), rec.duration)
        n = int(np.searchsorted(rec.t, span))
        t, x = rec.t[:n], rec.x[:n]
        self.rec_item.setData(t, x)
        d = np.diff(x)
        idx = np.nonzero(d[:-1] * d[1:] < 0)[0] + 1
        self.rev_item.setData(t[idx], x[idx])
        self.rec_plot.setXRange(0.0, span, padding=0)

        c = self.curve
        nb, dk, rf = narrow_band_damage(s, c), dirlik_damage(s, c), rec.damage_rate(c)
        cycles = float(np.sum(rec.counts))
        self.readout.setText(
            f"Record: {rec.duration:.4g} s, {len(reversals(rec.x))} reversals, {cycles:.4g} rainflow cycles. "
            f"Damage a second, and the life: rainflow {rf:.3g} ({life_text(rf)}); narrow band {nb:.3g} "
            f"({life_text(nb)}, {nb / rf:.2f} × rainflow); Dirlik {dk:.3g} ({life_text(dk)}, {dk / rf:.2f} × "
            "rainflow)."
        )

    def _draw_tab(self) -> None:
        rec, s, c = self.rec, self.stress, self.curve
        i = self.tabs.currentIndex()
        if rec is None or i > TAB_SN:
            return
        sigma = rates(s).sigma
        top = max(float(rec.ranges.max()), 6 * sigma)
        S = np.linspace(0.0, top, 600)
        edges = np.linspace(0.0, top, BINS + 1)
        counts, _ = np.histogram(rec.ranges, edges, weights=rec.counts)
        width = edges[1] - edges[0]
        total = float(np.sum(rec.counts))
        if i == TAB_RANGES:
            self.hist_item.setData(edges, counts / (total * width))
            self.ray_item.setData(S, rayleigh_range_pdf(s, S))
            self.dirlik_item.setData(S, dirlik_range_pdf(s, S))
            self.ranges_plot.setXRange(0.0, top, padding=0)
            self.tab_note.setText(
                "Rayleigh's ranges are one hump at 2σ√2 (the narrow-band assumption); the "
                "record's and Dirlik's have small ranges too, from the ripples."
            )
        elif i == TAB_DAMAGE:
            b = c.b
            mids = 0.5 * (edges[1:] + edges[:-1])
            share = counts * mids**b
            self.dhist_item.setData(edges, share / (np.sum(share) * width) if share.sum() else share)
            for item, pdf in ((self.dray_item, rayleigh_range_pdf(s, S)), (self.ddirlik_item, dirlik_range_pdf(s, S))):
                w = pdf * S**b
                item.setData(S, w / np.trapezoid(w, S))
            self.dmg_plot.setXRange(0.0, top, padding=0)
            biggest = np.sort(rec.ranges[rec.ranges > 0] ** b * rec.counts[rec.ranges > 0])[::-1]
            tenth = max(1, round(0.1 * len(biggest)))
            self.tab_note.setText(
                f"With b = {b:.3g}, the largest tenth of the record's cycles do "
                f"{100 * biggest[:tenth].sum() / biggest.sum():.0f} % of its damage."
            )
        else:
            N = np.logspace(3, 10, 50)
            self.sn_item.setData(N, c.s_ref * (N / c.n_ref) ** (-1 / c.b))
            order = np.argsort(rec.ranges)[::-1]
            above = np.cumsum(rec.counts[order])
            keep = rec.ranges[order] > 0
            # cycles at or above each range, scaled to the life
            life = 1.0 / rec.damage_rate(c) if rec.damage_rate(c) > 0 else rec.duration
            self.cycles_item.setData(above[keep] * life / rec.duration, rec.ranges[order][keep])
            high = float(np.log10(rec.ranges.max()))
            self.sn_plot.setXRange(3, 10, padding=0)
            self.sn_plot.setYRange(
                high - 1.5, max(high, float(np.log10(c.s_ref * (1e3 / c.n_ref) ** (-1 / c.b)))) + 0.1, padding=0
            )
            self.tab_note.setText(
                "Red: how many cycles at or above each range the record would put in over its "
                "whole life (its spectrum of ranges). Where it crosses the S–N curve tells which "
                "ranges matter."
            )
