"""Seismic response spectrum page: the Simulation page's chain as a building, a design spectrum, and the modes'
peaks combined, against the building stepped on records that match the spectrum.

Left: the trials, the challenges (`ChallengeBox`: the damping and base-isolation specs), the building (its floors,
and storey 1's stiffness k₁, the bearings' when it stands on a slab, with the period of the whole building
rigid on it), the design spectrum (peak ground acceleration, ground type, damping) and the combination rule.
Centre: the spectrum with each mode on it, and the modes' table (period, participation, effective mass, base
shear). Right: the building swaying under the first record (`seismic_view.BuildingView`, played by a `Playback`
with a time cursor on the record; a trial starts it), each floor's peak displacement and each storey's shear by
every rule and by the records, the modes' correlation, the ground motions, and the Theory.

The Building, Floors and Records tabs draw from the spectrum's synthesized records (`core.seismic.records`, kept
by ground type and ζ). A new pair takes 0.5–1 s to make, so a spectrum's control has them made on a `Recompute`
worker and the tab drawn when they're in (the spectrum and the table at once); a trial, a new chain, a tab picked
or the page showing make them now.

Working on it: `tests/test_seismic.py` checks the building follows the playback with the drifting storey
reddened, the k₁ box edits the chain, and a new spectrum's records are made off the GUI thread;
`tests/test_challenge.py` the challenges.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.model import ChainSystem
from ..core.practice import Prediction, Problem, Question, Requirement, at_most, kept, number, pick, rounded
from ..core.random_vibration import G
from ..core.seismic import (
    GROUNDS,
    RECORDS,
    RULES,
    Accelerogram,
    Design,
    History,
    Rule,
    Suite,
    building,
    combine,
    correlation,
    design_sa,
    modal,
    records,
    rigid_period,
    time_history,
    with_tank,
)
from ..view_models.text import number as sig
from . import coalesce, seismic_figures, settle_plots
from .challenge import ChallengeBox
from .lesson import (
    Form,
    TrialBox,
    column,
    combo,
    connect_all,
    fmt,
    mark,
    marker,
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
from .playback import Playback
from .recompute import Recompute
from .seismic_view import BuildingView
from .style import MAX_DOF, colors
from .theming import add_legend, mute

TAB_BUILDING, TAB_FLOORS, TAB_CORRELATION, TAB_GROUND = range(4)
PERIODS = np.linspace(0.0, 4.0, 801)[1:]


@dataclass(frozen=True)
class SeismicTrial:
    name: str
    note: str
    storeys: int = 5
    pga: float = 0.3
    ground: int = 2
    zeta: float = 5.0  # %
    tank: float = 0.0  # the rooftop tank's share of mode 1's modal mass there
    rule: Rule = Rule.CQC
    tab: int = TAB_FLOORS
    mass: float = 1e4  # each floor's (kg)
    stiffness: float = 5e6  # each storey's (N/m)
    masses: tuple[float, ...] = ()  # a chain element by element instead (a practice state's), its springs with it
    springs: tuple[float, ...] = ()
    predict: Prediction | None = None  # asked before it runs (Things to try)

    def system(self) -> ChainSystem:
        if self.masses:
            return ChainSystem(np.array(self.masses), np.array(self.springs), np.zeros(len(self.masses)))
        system = building(self.storeys, self.mass, self.stiffness)
        return with_tank(system, self.tank) if self.tank else system

    @property
    def design(self) -> Design:
        return Design(self.pga, GROUNDS[self.ground], self.zeta / 100)


TRIALS = (
    SeismicTrial(
        "A five-storey building",
        "Five 10 t floors on 5 MN/m storeys: <m>T<sub>1</sub></m> = 0.99 s. Mode 1 carries 88 % of the mass and "
        "modes 1 and 2 together 97 % (codes want 90 %), so a few modes are enough. The spectrum gives mode 1 "
        "<m>S<sub>a</sub></m> = 0.52 g, and the modes together a base shear of 230 kN by CQC (229 by SRSS); stepped on "
        "seven records that match the spectrum, the building's base shear averages 235 kN.",
        rule=Rule.CQC,
    ),
    SeismicTrial(
        "Combined by ABS",
        "The same building by ABS: every mode's peak added as if they all arrived together, 277 kN, 18 % over "
        "the records' mean. Modes this far apart (<m>ρ<sub>12</sub></m> = 0.007) peak at different moments, and "
        "SRSS (229 kN) is right.",
        rule=Rule.ABS,
        predict=Prediction(
            "The five-storey building's modes combined by ABS, adding their peaks. Against the mean of seven records,"
            " what does ABS give?",
            (
                "Too little",
                "About the records' mean",
                "Too much",
            ),
            2,
            ordered=True,
            why=(
                "Adding every mode's peak is the most they could give together, never less than what they do.",
                "ABS assumes every mode peaks at the same moment; modes this far apart (ρ₁₂ = 0.007) don't, so it overestimates.",
                '',
            ),
        ),
    ),
    SeismicTrial(
        "A tuned tank on the roof",
        "A tank on the roof, 1 % of mode 1's modal mass there, tuned to mode 1, splits it into two modes "
        "0.96 and 1.06 Hz apart, with <m>ρ</m> = 0.50: they move together much of the time. SRSS treats them as "
        "independent: 21 % low on the base shear, and 54 % high on the tank's own swing, where the two modes "
        "nearly cancel. CQC keeps the cross term, with its sign, and is within 11 % of the records' mean on both.",
        tank=0.01,
        rule=Rule.SRSS,
        predict=Prediction(
            "A tank on the roof, tuned to mode 1, splits it into two close modes (0.96 and 1.06 Hz). Against CQC, "
            "what does SRSS give for the base shear?",
            (
                "More",
                "About the same",
                "Less",
            ),
            2,
            ordered=True,
            why=(
                'SRSS leaves out the positive cross term of two close modes in step, so it comes out under CQC, not over.',
                'Two close modes (ρ = 0.50) move together much of the time; SRSS drops their cross term, which CQC keeps.',
                '',
            ),
        ),
    ),
    SeismicTrial(
        "Soft soil",
        "Ground type D: <m>S</m> = 1.35 and the plateau runs to <m>T<sub>C</sub></m> = 0.8 s, so mode 1 (0.99 s) "
        "sits near it. Its <m>S<sub>a</sub></m> rises from 0.52 g to 0.82 g, and the base shear from 230 to "
        "357 kN: a site's soil matters as much as its seismicity.",
        ground=3,
        predict=Prediction(
            "The five-storey building (T₁ = 0.99 s) moved from ground type C to D, the same a<sub>g</sub>. What does "
            "its base shear do?",
            (
                "It falls",
                "It stays the same",
                "It rises a little, as S does (1.15 to 1.35)",
                "It rises by more than half",
            ),
            3,
            ordered=True,
            why=(
                "Softer ground amplifies the shaking: S rises, and the plateau runs on to T<sub>C</sub> = 0.8 s, close to the building's 0.99 s.",
                "The ground type sets S and the spectrum's corners, and on D both work against this building: S<sub>a</sub> rises from 0.52 g to 0.82 g.",
                'S alone would give 17 %. The plateau also runs on, to T<sub>C</sub> = 0.8 s, close to T₁ = 0.99 s: S<sub>a</sub> rises from 0.52 g to 0.82 g.',
                '',
            ),
        ),
    ),
    SeismicTrial(
        "A taller building",
        "Eight storeys: <m>T<sub>1</sub></m> = 1.52 s, and mode 1 carries 86 % of the mass, under the 90 % codes "
        "want, so mode 2 is needed too. Mode 1 now sits on the falling branch (0.34 g) and modes 2 to 4 on the "
        "plateau (0.86 g): mode 1 alone gives 96 % of the base shear but only 74 % of the top storey's.",
        storeys=8,
    ),
    SeismicTrial(
        "More damping",
        "<m>ζ</m> = 10 % in every mode (dampers added): <m>η</m> = √(10/(5 + 10)) = 0.82 lowers the whole "
        "spectrum by 18 %, and every peak with it.",
        zeta=10.0,
    ),
    SeismicTrial(
        "A soft storey",
        "The first storey a fifth as stiff, 1 MN/m (an open ground floor): <m>T<sub>1</sub></m> = 1.58 s, further "
        "down the spectrum, so the base shear falls to 159 kN. But the first storey drifts 159 mm, against 46 mm, "
        "70 % of the roof's 227 mm: the whole building's sway in one storey, which is where it yields and fails.",
        masses=(1e4,) * 5,
        springs=(1e6,) + (5e6,) * 4,
        tab=TAB_BUILDING,
    ),
    SeismicTrial(
        "Base isolation",
        "The five storeys on a 10 t slab on bearings of 300 kN/m, at 15 % damping: <m>T<sub>b</sub></m> = 2.94 s, "
        "past <m>T<sub>D</sub></m>, with 99.8 % of the mass in that mode. The first storey above the bearings "
        "carries 42 kN, against 230 kN fixed to the ground, and drifts 8 mm, against 46 mm; the bearings move "
        "166 mm. (The page gives every mode the bearings' damping; with 99.8 % of the mass in the first, that "
        "changes little.)",
        zeta=15.0,
        masses=(1e4,) * 6,
        springs=(3e5,) + (5e6,) * 5,
        tab=TAB_BUILDING,
    ),
)

# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


def _base_shear(t: SeismicTrial, rule: Rule = Rule.CQC) -> float:
    m = modal(t.system(), t.design)
    return float(combine(m.shear, rule, correlation(m.f, t.design.zeta))[0])


def _sa(rng: np.random.Generator) -> Question:
    ground = int(pick(rng, range(len(GROUNDS))))
    pga, T = pick(rng, (0.1, 0.2, 0.3, 0.4)), pick(rng, (0.1, 0.3, 0.6, 1.0, 1.5, 2.5))
    d = Design(pga, GROUNDS[ground], 0.05)
    S, TB, TC, TD = d.corners
    sa = float(design_sa(d, np.array([T]))[0]) / G
    if T < TB:
        branch = "below T<sub>B</sub>: a<sub>g</sub>S[1 + T/T<sub>B</sub>(2.5η − 1)]"
    elif T <= TC:
        branch = "on the plateau: 2.5η a<sub>g</sub>S"
    elif T <= TD:
        branch = "between T<sub>C</sub> and T<sub>D</sub>: 2.5η a<sub>g</sub>S T<sub>C</sub>/T"
    else:
        branch = "past T<sub>D</sub>: 2.5η a<sub>g</sub>S T<sub>C</sub>T<sub>D</sub>/T²"
    return Question(
        f"A design spectrum for a<sub>g</sub> = {sig(pga, 6)} g on ground type {GROUNDS[ground]}: S = {sig(S, 6)}, "
        f"T<sub>B</sub> = {sig(TB, 6)} s, T<sub>C</sub> = {sig(TC, 6)} s, T<sub>D</sub> = {sig(TD, 6)} s, ζ = 5 % (η = 1). What is "
        f"S<sub>a</sub> at T = {sig(T, 6)} s?",
        sa,
        f"{sig(T, 6)} s is {branch} = {number(sa)} g.",
        unit="g",
        setup=dict(pga=pga, ground=ground),
    )


def _eta(rng: np.random.Generator) -> Question:
    zeta = pick(rng, (2, 3, 8, 10, 15, 20))
    eta = Design(zeta=zeta / 100).eta
    return Question(
        f"Dampers bring every mode of a building to ζ = {sig(zeta, 6)} %. By what factor η does that scale its design "
        "spectrum, against 5 %?",
        eta,
        f"η = √(10/(5 + ζ%)) = √(10/{sig(5 + zeta, 6)}) = {number(eta)}"
        + (": the spectrum rises." if eta > 1 else f": every peak {number(100 * (1 - eta))} % lower."),
        setup=dict(zeta=float(zeta)),
    )


def _period(rng: np.random.Generator) -> Question:
    n = int(pick(rng, (3, 4, 5, 6, 8)))
    m, k = pick(rng, (5, 10, 20)), pick(rng, (2, 5, 10, 20))
    T = float(modal(building(n, 1000 * m, 1e6 * k), Design()).T[0])
    w = 2 * math.sqrt(k * 1e6 / (m * 1000)) * math.sin(math.pi / (2 * (2 * n + 1)))
    return Question(
        f"A uniform shear building of {n} storeys: {sig(m, 6)} t floors on {sig(k, 6)} MN/m storeys. What is its first mode's "
        "period?",
        T,
        f"A uniform chain fixed at the ground and free at the top has ω<sub>r</sub> = 2√(k/m) sin((2r − 1)π/(2(2n + "
        f"1))): ω₁ = 2 × √({number(k * 1e6 / (m * 1000))}) × sin(π/{2 * (2 * n + 1)}) = {number(w)} rad/s, so T₁ = "
        f"2π/ω₁ = {number(T)} s.",
        unit="s",
        setup=dict(storeys=n, mass=1000 * m, stiffness=1e6 * k),
    )


def _effective_mass(rng: np.random.Generator) -> Question:
    m1, m2 = pick(rng, (10, 15, 20)), pick(rng, (5, 10, 15))
    a = pick(rng, (1.3, 1.5, 1.8, 2.0))
    num, den = m1 + m2 * a, m1 + m2 * a * a
    meff = num * num / den
    return Question(
        f"A two-storey building, {sig(m1, 6)} t on the first floor and {sig(m2, 6)} t on the second, has a mode shaped (1, {sig(a, 6)}). "
        "What is that mode's effective mass?",
        meff,
        f"M<sub>eff</sub> = (φᵀM1)²/φᵀMφ = ({sig(m1, 6)} + {sig(m2, 6)} × {sig(a, 6)})²/({sig(m1, 6)} + {sig(m2, 6)} × {sig(a, 6)}²) = "
        f"{number(num)}²/{number(den)} = {number(meff)} t, {number(100 * meff / (m1 + m2))} % of the "
        f"{sig(m1 + m2, 6)} t.",
        unit="t",
    )


def _modal_shear(rng: np.random.Generator) -> Question:
    meff, sa = pick(rng, (30, 44, 60, 85, 120)), pick(rng, (0.2, 0.35, 0.5, 0.8))
    V = meff * sa * G
    return Question(
        f"A mode with {sig(meff, 6)} t of effective mass sits at S<sub>a</sub> = {sig(sa, 6)} g on the design spectrum. What base "
        "shear does it give?",
        V,
        f"V = M<sub>eff</sub>S<sub>a</sub> = {sig(meff, 6)} t × {sig(sa, 6)} × 9.81 m/s² = {number(V)} kN.",
        unit="kN",
    )


def _srss(rng: np.random.Generator) -> Question:
    v = (pick(rng, (150, 200, 250)), pick(rng, (40, 60, 80)), pick(rng, (10, 20, 30)))
    srss = math.sqrt(sum(x * x for x in v))
    return Question(
        f"Three well-separated modes give base shears of {sig(v[0], 6)}, {sig(v[1], 6)} and {sig(v[2], 6)} kN. What base shear does "
        "SRSS combine them into?",
        srss,
        f"√({sig(v[0], 6)}² + {sig(v[1], 6)}² + {sig(v[2], 6)}²) = {number(srss)} kN (ABS would say {sig(sum(v), 6)} kN; CQC, with the modes "
        "this far apart, about the same as SRSS).",
        unit="kN",
    )


def _rho(rng: np.random.Generator) -> Question:
    ratio, zeta = pick(rng, (1.05, 1.1, 1.2, 1.5, 3.0)), pick(rng, (0.02, 0.05, 0.1))
    rho = float(correlation(np.array([1.0, ratio]), zeta)[0, 1])
    return Question(
        f"Two modes, ζ = {sig(100 * zeta, 6)} % each, their frequencies {sig(ratio, 6)} times apart. What is Der Kiureghian's "
        "correlation ρ between them?",
        rho,
        f"ρ = 8ζ²(1 + r)r<sup>3/2</sup> / ((1 − r²)² + 4ζ²r(1 + r)²) with r = {sig(ratio, 6)}: {number(rho)}"
        + (": well apart, SRSS will do." if rho < 0.1 else ": correlated, so CQC's cross term matters."),
        tolerance=0.02,
    )


def _soft_soil(rng: np.random.Generator) -> Question:
    return Question(
        "The five-storey building (T₁ = 0.99 s) is moved from ground type C to D, the same a<sub>g</sub>. What does "
        "its base shear do?",
        0,
        "It rises by more than half, 229 to 357 kN: S goes from 1.15 to 1.35, and the plateau now runs to "
        "T<sub>C</sub> = 0.8 s, nearer mode 1, whose S<sub>a</sub> rises from 0.52 to 0.82 g.",
        choices=("It rises by more than half", "It rises a little (S: 1.15 to 1.35)", "It stays the same", "It falls"),
        why=(
            "",
            "The soil factor is only part of it: the plateau now runs to T<sub>C</sub> = 0.8 s, nearer mode 1's "
            "0.99 s, and S<sub>a</sub> rises from 0.52 to 0.82 g.",
            "Softer ground raises both the soil factor and the plateau's end, T<sub>C</sub>, towards mode 1.",
            "Softer ground amplifies the shaking (S from 1.15 to 1.35) and stretches the plateau towards mode 1: "
            "the shear rises.",
        ),
        setup=dict(ground=3),
        predict=True,
    )


def _stiffer(rng: np.random.Generator) -> Question:
    return Question(
        "Every storey of the five-storey building (T₁ = 0.99 s, past T<sub>C</sub> = 0.6 s on ground C) is made twice "
        "as stiff. What does its base shear do?",
        0,
        "It rises, 229 to 322 kN: T₁ falls by √2 to 0.70 s, still on the 1/T branch, so S<sub>a</sub> rises by √2. A "
        "stiffer building attracts more force (its drifts still fall).",
        choices=("It rises", "It falls", "It stays the same"),
        why=(
            "",
            "The drifts fall, but the force rises: T₁ moves left on the 1/T branch, where S<sub>a</sub> grows as 1/T.",
            "On the 1/T branch S<sub>a</sub> grows as the period falls: twice the stiffness is 1/√2 of T₁ and √2 "
            "of the force.",
        ),
        setup=dict(stiffness=1e7),
        predict=True,
    )


def _tank_srss(rng: np.random.Generator) -> Question:
    return Question(
        "A tank on the roof, tuned to mode 1, splits it into two close modes (0.96 and 1.06 Hz). Against CQC, what "
        "does SRSS give for the base shear?",
        0,
        "Less, 165 kN against CQC's 200, which is near the records' mean: SRSS treats the two modes as independent, but with ρ = "
        "0.50 they move together much of the time, and CQC keeps the cross term.",
        choices=("Less", "About the same", "More"),
        why=(
            "",
            "Two close modes (ρ = 0.50) move together much of the time; SRSS drops their cross term, which CQC keeps.",
            "SRSS leaves out the positive cross term of two close modes in step, so it comes out under CQC, not over.",
        ),
        setup=dict(tank=0.01, rule=Rule.SRSS),
        predict=True,
    )


def _abs(rng: np.random.Generator) -> Question:
    return Question(
        "Combining the five-storey building's modes by ABS (adding their peaks), against the mean of the records. "
        "What does ABS give?",
        0,
        "Too much, 277 kN against the records' 235: ABS assumes every mode peaks at the same moment, and modes this "
        "far apart (ρ₁₂ = 0.007) don't.",
        choices=("Too much", "About the records' mean", "Too little"),
        why=(
            "",
            "ABS assumes every mode peaks at the same moment; modes this far apart (ρ₁₂ = 0.007) don't, so it "
            "overestimates.",
            "Adding every mode's peak is the most they could give together, never less than what they do.",
        ),
        setup=dict(rule=Rule.ABS),
        predict=True,
    )


def _damping_spec(rng: np.random.Generator) -> Question:
    n = int(pick(rng, (4, 5, 6)))
    ground = int(pick(rng, (1, 2, 3)))
    pga = pick(rng, (0.2, 0.3, 0.4))
    zd = pick(rng, (10, 12, 15, 20))  # %, the design
    start = SeismicTrial("", "", storeys=n, pga=pga, ground=ground)
    design = SeismicTrial("", "", storeys=n, pga=pga, ground=ground, zeta=float(zd))
    limit = rounded(1.04 * _base_shear(design) / 1000)  # kN
    most = float(round(1.3 * zd))
    given = start.system()

    def goal(t: SeismicTrial):
        system = t.system()
        same = (
            system.n == given.n
            and np.allclose(system.masses, given.masses)
            and np.allclose(system.stiffness, given.stiffness)
        )
        return (
            Requirement(f"The {n}-storey building as given", f"{system.n} floors", bool(same)),
            kept("a<sub>g</sub>", t.pga, pga, "g"),
            Requirement("Ground type", GROUNDS[t.ground], t.ground == ground),
            Requirement("Combined by CQC", t.rule.value, t.rule is Rule.CQC),
            at_most("Damping ζ", t.zeta, most, "%"),
            at_most("Base shear", _base_shear(t) / 1000, limit, "kN"),
        )

    lo, hi = 5.0, float(zd)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        trial = SeismicTrial("", "", storeys=n, pga=pga, ground=ground, zeta=mid)
        lo, hi = (mid, hi) if _base_shear(trial) / 1000 > limit else (lo, mid)
    return Question(
        f"The uniform {n}-storey building (10 t floors, 5 MN/m storeys) on ground type {GROUNDS[ground]}, "
        f"a<sub>g</sub> = {sig(pga, 6)} g. Dampers can raise every mode's ζ, to {sig(most, 6)} % at most. Choose ζ so that the "
        f"base shear, combined by CQC, is {number(limit)} kN or less.",
        0,
        f"Every peak scales with η = √(10/(5 + ζ%)), down to 0.55: at 5 % the base shear is "
        f"{number(_base_shear(start) / 1000)} kN, so η must come down to {number(limit * 1000 / _base_shear(start))}, "
        f"ζ ≥ about {number(hi)} % (ζ = {sig(zd, 6)} % gives {number(_base_shear(design) / 1000)} kN).",
        setup=dict(storeys=n, pga=pga, ground=ground),
        goal=goal,
        design=dict(storeys=n, pga=pga, ground=ground, zeta=float(zd)),
    )


def _isolated(t: SeismicTrial) -> tuple[float, float]:
    """The shear in the storey above the bearings and the bearings' own displacement, by CQC (N, m)."""
    m = modal(t.system(), t.design)
    rho = correlation(m.f, t.design.zeta)
    return float(combine(m.shear, Rule.CQC, rho)[1]), float(combine(m.displacement, Rule.CQC, rho)[0])


def _isolation_spec(rng: np.random.Generator) -> Question:
    n = int(pick(rng, (4, 5, 6)))
    ground = int(pick(rng, (1, 2)))
    pga = pick(rng, (0.2, 0.3, 0.4))
    period = pick(rng, (2.0, 2.5, 3.0))  # s, the design's
    masses = (1e4,) * (n + 1)  # the floors on a slab of the same mass
    kb = rounded(sum(masses) * (2 * math.pi / period) ** 2)
    common: dict[str, Any] = dict(storeys=n + 1, pga=pga, ground=ground, zeta=15.0, tab=TAB_BUILDING, masses=masses)
    start = dict(common, springs=(5e6,) * (n + 1))  # the slab fixed to the ground as a storey
    design = dict(common, springs=(kb,) + (5e6,) * n)
    shear, drift = _isolated(SeismicTrial("", "", **design))
    shear_limit = rounded(1.15 * shear / 1000)  # kN
    move_limit = rounded(1.05 * drift * 1000)  # mm

    def goal(t: SeismicTrial):
        system = t.system()
        same = system.n == n + 1 and np.allclose(system.masses, masses) and np.allclose(system.stiffness[1:], 5e6)
        above, move = _isolated(t)
        return (
            Requirement(f"The {n} storeys on their slab as given", f"{system.n} masses", bool(same)),
            kept("a<sub>g</sub>", t.pga, pga, "g"),
            Requirement("Ground type", GROUNDS[t.ground], t.ground == ground),
            kept("Damping ζ", t.zeta, 15.0, "%"),
            at_most("Shear above the bearings", above / 1000, shear_limit, "kN"),
            at_most("The bearings' displacement", move * 1000, move_limit, "mm"),
        )

    return Question(
        f"The uniform {n}-storey building (10 t floors, 5 MN/m storeys) stands on a 10 t slab, on ground type "
        f"{GROUNDS[ground]}, a<sub>g</sub> = {sig(pga, 6)} g, ζ = 15 %. Choose the bearings under the slab (storey 1's "
        f"k<sub>1</sub>) so that the storey above them carries {number(shear_limit)} kN or less, by CQC, while the "
        f"bearings move {number(move_limit)} mm or less.",
        0,
        f"The whole {number(sum(masses) / 1000)} t moves on the bearings at T<sub>b</sub> = 2π√(Σm/k<sub>1</sub>). "
        f"A longer period lowers the shear (S<sub>a</sub> falls), but the bearings move more, up to the spectrum's "
        f"T<sub>D</sub>: a period of about {sig(period, 6)} s, k<sub>1</sub> ≈ {number(kb)} N/m, meets both.",
        setup=start,
        goal=goal,
        design=design,
    )


PROBLEMS = tuple(
    Problem(key, title, "seismic_page", f"See {page_link('seismic_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("design-sa", "Reading the design spectrum", "design-spectrum", "A design spectrum", _sa),
        ("eta", "Damping's correction", "design-spectrum", "A design spectrum", _eta),
        ("period", "A building's period", "participation", "Participation and effective mass", _period),
        ("effective-mass", "Effective mass", "participation", "Participation and effective mass", _effective_mass),
        ("modal-shear", "A mode's base shear", "participation", "Participation and effective mass", _modal_shear),
        ("srss", "SRSS", "combination", "Combining the modes", _srss),
        ("rho", "Two modes' correlation", "combination", "Combining the modes", _rho),
        ("soft-soil", "Predict: softer soil", "design-spectrum", "A design spectrum", _soft_soil),
        ("stiffer", "Predict: a stiffer building", "design-spectrum", "A design spectrum", _stiffer),
        ("tank-srss", "Predict: close modes by SRSS", "combination", "Combining the modes", _tank_srss),
        ("abs", "Predict: ABS", "combination", "Combining the modes", _abs),
        ("damping-spec", "Spec: damping for a base shear", "design-spectrum", "A design spectrum", _damping_spec),
        ("isolation-spec", "Spec: isolate the base", "isolation", "Base isolation", _isolation_spec),
    )
)

TIPS = {
    "dof": "<p>The number of floors: the Simulation page's chain is the building, m1 the first floor.</p>",
    "k1": "<p>Storey 1's stiffness, the Simulation page's k1: soften it for an open ground floor, or put the "
    "building on a slab on bearings and it's the bearings' stiffness. T is the period of the whole building as "
    "a rigid block on it: the base isolation's.</p>",
    "pga": "<p>The design peak ground acceleration a<sub>g</sub> on rock, in g.</p>",
    "ground": "<p>The ground type: soft soil amplifies the motion (S) and widens the plateau (T<sub>C</sub>).</p>",
    "zeta": "<p>The damping ratio of every mode, for the spectrum (η) and the records' response.</p>",
    "rule": "<p>How the modes' peaks are combined for the readout and the bold profile.</p>",
}
SPECTRUM_TIP = (
    "<p>The design spectrum: the peak acceleration of a single oscillator of period T and damping ζ. "
    "Dots: the building's modes.</p>"
)
FLOOR_TIP = (
    "<p>Each floor's peak displacement relative to the ground (left) and each storey's peak shear "
    "(right): by each combination rule, and the mean over the records (dots).</p>"
)
GROUND_TIP = "<p>One of the records (top), and every record's spectrum against the design spectrum (bottom).</p>"


NOTES_HTML = load_notes("seismic", figures=seismic_figures.FIGURES, trials=trials_html(TRIALS))


def _same(a: ChainSystem, b: ChainSystem) -> bool:
    return (
        a.n == b.n
        and np.allclose(a.masses, b.masses)
        and np.allclose(a.stiffness, b.stiffness)
        and np.allclose(a.damping, b.damping)
    )


class SeismicPage(QtWidgets.QWidget):
    """Controls on the left, the spectrum and the modes in the centre, the floors, correlation and records on the
    right."""

    dof_requested = QtCore.Signal(int)
    system_requested = QtCore.Signal(object)
    edit_parameters = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system = ChainSystem.uniform(4)
        self._dirty = True

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Each loads a building into the Simulation page and sets up the spectrum to show "
            "one idea from the Theory tab.</p>",
            predictions=[t.predict for t in TRIALS],
        )
        self.trials.picked.connect(self.load_trial)
        self.challenge = ChallengeBox(self, PROBLEMS)

        ch = Form("Building (the chain)")
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.valueChanged.connect(self.dof_requested)
        ch.add("Floors:", self.dof, TIPS["dof"])
        self.k1 = ch.number("Storey 1 k₁:", 0.0, 1e9, 5e6, 0, " N/m", TIPS["k1"])  # 0: a free chain, not held
        self.k1.valueChanged.connect(self._on_k1)
        self.k1_info = note_label()
        ch.row(self.k1_info)
        link = QtWidgets.QLabel("Masses and storey stiffnesses are edited on the <a href='#sim'>Simulation page</a>.")
        link.setWordWrap(True)
        mute(link)
        link.linkActivated.connect(self.edit_parameters)
        ch.row(link)

        spec = Form("Design spectrum")
        self.pga = spec.number("Peak ground acc.:", 0.01, 2.0, 0.3, 2, " g", TIPS["pga"])
        self.ground = combo(GROUNDS, TIPS["ground"])
        self.ground.setCurrentIndex(2)
        spec.add("Ground type:", self.ground, TIPS["ground"])
        self.zeta = spec.number("Damping ζ:", 0.5, 30.0, 5.0, 1, " %", TIPS["zeta"])
        self.rule = combo([r.value for r in RULES], TIPS["rule"])
        self.rule.setCurrentIndex(RULES.index(Rule.CQC))
        spec.add("Combine by:", self.rule, TIPS["rule"])
        self.spec_info = note_label()
        spec.row(self.spec_info)
        connect_all((self.pga, self.ground, self.zeta, self.rule), self._on_spec)

        # --- centre
        self.spectrum_plot = plot("S<sub>a</sub> (g)", None, "Period T", "s", height=150)
        self.spectrum_plot.setToolTip(SPECTRUM_TIP)
        self.spectrum_item = pg.PlotDataItem()
        self.mode_dots = marker(11)
        self.corner_lines = [pg.InfiniteLine(angle=90) for _ in range(3)]
        for item in (*self.corner_lines, self.spectrum_item, self.mode_dots):
            self.spectrum_plot.addItem(item)
        self.mode_labels: list[pg.TextItem] = []
        self.table = readout_label()
        self.table.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

        # --- right: floors
        floors = QtWidgets.QWidget()
        fl = QtWidgets.QHBoxLayout(floors)
        fl.setContentsMargins(0, 0, 0, 0)
        self.disp_plot = plot("Floor", None, "Peak displacement", "m", height=200)
        self.shear_plot = plot("Storey", None, "Peak shear", "N", height=200)
        self.rule_items: dict[Rule, tuple[pg.PlotDataItem, pg.PlotDataItem]] = {}
        add_legend(self.disp_plot.getPlotItem(), offset=(10, 10))
        for r in RULES:
            self.rule_items[r] = (pg.PlotDataItem(name=r.value), pg.PlotDataItem())
            self.disp_plot.addItem(self.rule_items[r][0])
            self.shear_plot.addItem(self.rule_items[r][1])
        self.th_items = (marker(9, name="records' mean"), marker(9))
        self.disp_plot.addItem(self.th_items[0])
        self.shear_plot.addItem(self.th_items[1])
        for p in (self.disp_plot, self.shear_plot):
            p.setToolTip(FLOOR_TIP)
            p.setMouseEnabled(x=False, y=False)
            p.getAxis("left").setTickSpacing(1, 1)
            fl.addWidget(p)
        self.floor_note = note_label()

        self.rho_label = readout_label()
        self.rho_note = note_label()

        self.record_plot = plot("Ground acc.", "m/s²", "Time", "s", height=120)
        self.record_item = pg.PlotDataItem()
        self.record_plot.addItem(self.record_item)
        self.match_plot = plot("S<sub>a</sub> (g)", None, "Period T", "s", log_x=True, height=150)
        self.match_items: list[pg.PlotDataItem] = []
        self.target_item = pg.PlotDataItem(name="design")
        self.match_mean = pg.PlotDataItem(name="the records' mean")
        add_legend(self.match_plot.getPlotItem(), offset=(-10, 10))
        for item in (self.target_item, self.match_mean):
            self.match_plot.addItem(item)
        for p in (self.record_plot, self.match_plot):
            p.setToolTip(GROUND_TIP)
        self.ground_note = note_label()

        # --- right: the building swaying under the first record
        self.playback = Playback(self, "Seismic response", speed=1.0)
        self.playback.moved.connect(self._place)
        self.building = BuildingView()
        self.building.setMinimumHeight(220)
        self.sway_plot = plot("Ground acc.", "m/s²", "Time", "s", height=110)
        self.sway_plot.setToolTip(GROUND_TIP)
        self.sway_item = pg.PlotDataItem()
        self.sway_plot.addItem(self.sway_item)
        self.sway_cursor = self.playback.add_cursor(self.sway_plot)
        self.building_note = note_label()
        self.history: History | None = None

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(
            column(self.playback.bar(), self.building, self.sway_plot, self.building_note, stretch=False), "Building"
        )
        self.tabs.addTab(column(floors, self.floor_note, stretch=False), "Floors")
        self.tabs.addTab(column(self.rho_label, self.rho_note), "Correlation")
        self.tabs.addTab(column(self.record_plot, self.match_plot, self.ground_note, stretch=False), "Records")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._on_tab)

        three_panes(
            self,
            column(self.trials, self.challenge, ch, spec),
            stack((self.spectrum_plot, 1), (self.table, 0)),
            self.tabs,
        )
        self.worker = Recompute(self._on_records, self)  # a new spectrum's records (0.5–1 s) off the GUI thread
        self.apply_theme()
        self.set_system(self.system)

    # ------------------------------------------------------------- inputs
    def set_system(self, system: ChainSystem, result=None) -> None:
        self.system = system
        set_quietly(self.dof, system.n)
        set_quietly(self.k1, float(system.stiffness[0]))
        T = rigid_period(system)
        self.k1_info.setText(
            f"T = 2π√(Σm/k₁) = {sig(T)} s: the building rigid on storey 1."
            if math.isfinite(T)
            else "k₁ = 0: nothing holds the building to the ground."
        )
        self._dirty = True
        if self.isVisible():
            self.redraw()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty:
            self.redraw()

    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(SeismicTrial("", "", **setup))

    def practice_state(self) -> SeismicTrial:
        """The building and the spectrum's settings, as a trial: what a practice spec checks."""
        s = self.system
        return SeismicTrial(
            "",
            "",
            storeys=s.n,
            pga=self.pga.value(),
            ground=self.ground.currentIndex(),
            zeta=self.zeta.value(),
            rule=RULES[self.rule.currentIndex()],
            tab=self.tabs.currentIndex(),
            masses=tuple(float(v) for v in s.masses),
            springs=tuple(float(v) for v in s.stiffness),
        )

    def apply_trial(self, t: SeismicTrial) -> None:
        system = t.system()
        self.system_requested.emit(system)
        if not _same(self.system, system):
            self.set_system(system)
        for box, value in (
            (self.pga, t.pga),
            (self.ground, t.ground),
            (self.zeta, t.zeta),
            (self.rule, RULES.index(t.rule)),
        ):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self._dirty = True
        if self.isVisible():  # the records are stepped only once it shows
            self.redraw()
        self.playback.play()  # a trial starts the motion

    def _on_k1(self, value: float) -> None:
        """Storey 1's stiffness typed: the chain changes, through the Simulation page."""
        s = self.system
        k = s.stiffness.copy()
        k[0] = value
        system = ChainSystem(s.masses.copy(), k, s.damping.copy(), s.friction.copy(), s.static_ratio)
        self.system_requested.emit(system)
        if not _same(self.system, system):  # no Simulation page (a test)
            self.set_system(system)

    @property
    def design(self) -> Design:
        return Design(self.pga.value(), GROUNDS[self.ground.currentIndex()], self.zeta.value() / 100)

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.spectrum_item.setPen(pg.mkPen(c.strong, width=2))
        mark(self.mode_dots, c.force, filled=True)
        for line in self.corner_lines:
            line.setPen(pg.mkPen(c.faint, width=1, style=QtCore.Qt.PenStyle.DotLine))
        styles = {
            Rule.ABS: (c.grey, QtCore.Qt.PenStyle.DashLine),
            Rule.SRSS: (c.mass[1], QtCore.Qt.PenStyle.SolidLine),
            Rule.NRL: (c.mode[2], QtCore.Qt.PenStyle.DashDotLine),
            Rule.CQC: (c.mass[0], QtCore.Qt.PenStyle.SolidLine),
        }
        for r, items in self.rule_items.items():
            color, style = styles[r]
            for item in items:
                item.setPen(pg.mkPen(color, width=2, style=style))
                item.setSymbol("o")
                item.setSymbolSize(5)
                item.setSymbolBrush(pg.mkBrush(color))
                item.setSymbolPen(None)
        for item in self.th_items:
            mark(item, c.strong)
        self.record_item.setPen(pg.mkPen(c.mass[0], width=1))
        self.target_item.setPen(pg.mkPen(c.strong, width=2.5))
        self.match_mean.setPen(pg.mkPen(c.force, width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.sway_item.setPen(pg.mkPen(c.mass[0], width=1))
        self.building.apply_theme()
        self.playback.apply_theme()
        self._dirty = True
        if self.isVisible():
            self.redraw()

    def _on_spec(self) -> None:
        """A design spectrum's control: drawn at once, but for its records, made on the worker (`_draw_tab`)."""
        if not self.isVisible():
            self._dirty = True
            return
        self.redraw(now=False)

    def _on_tab(self) -> None:
        self._draw_tab()

    @coalesce.coalesced
    def redraw(self, now: bool = True) -> None:
        """The spectrum, the table and the tab showing: the records a tab draws from on the worker unless `now` (a
        trial, a new chain, the page showing: the reader waits for those, and the tests read them at once)."""
        self._dirty = False
        d = self.design
        S, TB, TC, TD = d.corners
        self.spec_info.setText(
            f"S = {sig(S, 6)}, T<sub>B</sub> = {sig(TB, 6)} s, T<sub>C</sub> = {sig(TC, 6)} s, T<sub>D</sub> = "
            f"{sig(TD, 6)} s, η = {sig(d.eta)}: a plateau of {sig(2.5 * d.eta * S * d.pga_g)} g."
        )
        held = math.isfinite(rigid_period(self.system))
        self.tabs.setEnabled(held)  # every tab draws the building's response, and a free one has none
        if not held:
            self._draw_unheld()
            settle_plots(self)
            return
        self.modal = modal(self.system, d)
        self.rho = correlation(self.modal.f, d.zeta)
        self._draw_spectrum()
        self._draw_table()
        self._draw_tab(now)
        settle_plots(self)

    def _draw_unheld(self) -> None:
        """A chain with no first spring (a free one, from another page): the design spectrum alone, and why."""
        d = self.design
        self.spectrum_item.setData(PERIODS, design_sa(d, PERIODS) / G)
        self.mode_dots.setData([], [])
        for label in self.mode_labels:
            self.spectrum_plot.removeItem(label)
        self.mode_labels = []
        self.table.setText(
            "This chain has no spring under storey 1, so nothing holds the building to the ground and the ground "
            "can't shake it. Give storey 1 a stiffness (its k₁ box), or pick a trial."
        )

    def _draw_spectrum(self) -> None:
        d, m = self.design, self.modal
        self.spectrum_item.setData(PERIODS, design_sa(d, PERIODS) / G)
        self.mode_dots.setData(m.T, m.sa / G)
        for line, T in zip(self.corner_lines, d.corners[1:]):
            line.setPos(T)
        for label in self.mode_labels:
            self.spectrum_plot.removeItem(label)
        self.mode_labels = []
        for r in range(min(m.f.size, 4)):
            label = pg.TextItem(str(r + 1), color=colors.force, anchor=(0.5, 1.3))
            label.setPos(float(m.T[r]), float(m.sa[r] / G))
            self.spectrum_plot.addItem(label)
            self.mode_labels.append(label)
        self.spectrum_plot.setXRange(0.0, max(4.0, 1.05 * float(m.T[0])), padding=0)
        self.spectrum_plot.setYRange(0.0, 1.15 * float(np.max(design_sa(d, PERIODS))) / G, padding=0)

    def _draw_table(self) -> None:
        m = self.modal
        rule = RULES[self.rule.currentIndex()]
        cum = np.cumsum(m.mass_share)
        rows = "".join(
            f"<tr><td>{r + 1}</td><td align='right'>{sig(m.f[r])}</td><td align='right'>{sig(m.T[r])}</td>"
            f"<td align='right'>{sig(m.gamma[r] * np.sign(m.phi[-1, r]))}</td>"
            f"<td align='right'>{sig(100 * m.mass_share[r])} %</td><td align='right'>{sig(100 * cum[r])} %</td>"
            f"<td align='right'>{sig(m.sa[r] / G)}</td><td align='right'>{fmt(abs(m.base_shear[r]), 'N')}</td></tr>"
            for r in range(m.f.size)
        )
        totals = " · ".join(
            f"{'<b>' if r is rule else ''}{r.value} {fmt(float(combine(m.shear, r, self.rho)[0]), 'N')}"
            f"{'</b>' if r is rule else ''}"
            for r in RULES
        )
        need = int(np.searchsorted(cum, 0.9 - 1e-9)) + 1
        self.table.setText(
            "<table cellspacing='5'><tr><th>mode</th><th>f (Hz)</th><th>T (s)</th><th>Γ</th><th>M<sub>eff</sub></th>"
            f"<th>Σ M<sub>eff</sub></th><th>S<sub>a</sub> (g)</th><th>V<sub>base</sub></th></tr>{rows}</table>"
            f"Base shear: {totals}. {need} mode{'s' if need > 1 else ''} reach{'' if need > 1 else 'es'} 90 % of "
            f"the mass ({sig(m.total_mass / 1000, 4)} t)."
        )

    def _draw_tab(self, now: bool = True) -> None:
        """The tab showing; the Building, Floors and Records tabs once the spectrum's records are made (`records`
        keeps them by ground type and ζ: a new pair takes 0.5–1 s to synthesize, on the worker unless `now`)."""
        if self._dirty or not self.tabs.isEnabled():  # hidden and not yet drawn, or a free chain's: nothing to draw
            return
        i = self.tabs.currentIndex()
        if i == TAB_CORRELATION:
            self._draw_correlation()
        elif i in (TAB_BUILDING, TAB_FLOORS, TAB_GROUND):
            job = functools.partial(records, self.design)
            if now:
                self.worker.now(job)
            else:
                self.worker.request(job)

    def _on_records(self, recs: tuple[Accelerogram, ...]) -> None:
        """The latest spectrum's records: the tab showing drawn from them."""
        i = self.tabs.currentIndex()
        if i == TAB_BUILDING:
            self._draw_building(recs)
        elif i == TAB_FLOORS:
            self._draw_floors(recs)
        elif i == TAB_GROUND:
            self._draw_ground(recs)
        settle_plots(self)

    def _draw_floors(self, recs: tuple[Accelerogram, ...]) -> None:
        m, rho = self.modal, self.rho
        rule = RULES[self.rule.currentIndex()]
        floors = np.arange(1, m.f.size + 1)
        for r, (di, si) in self.rule_items.items():
            width = 3 if r is rule else 1.5
            for item in (di, si):
                pen = item.opts["pen"]
                pen.setWidthF(width)
                item.setPen(pen)
            di.setData(combine(m.displacement, r, rho), floors)
            si.setData(combine(m.shear, r, rho), floors)
        s = Suite(tuple(time_history(self.system, self.design, g) for g in recs))
        self.th_items[0].setData(s.displacement, floors)
        self.th_items[1].setData(s.shear, floors)
        for p in (self.disp_plot, self.shear_plot):
            p.setYRange(0.5, m.f.size + 0.5, padding=0)
            p.enableAutoRange(axis="x")
        mine = float(combine(m.shear, rule, rho)[0])
        top = float(combine(m.displacement, rule, rho)[-1])
        self.floor_note.setText(
            f"Base shear by {rule.value} {fmt(mine, 'N')}, {sig(100 * (mine / s.base_shear - 1), plus=True)} % against the "
            f"mean of {RECORDS} records ({fmt(s.base_shear, 'N')}); the top floor's displacement "
            f"{fmt(top, 'm')}, {sig(100 * (top / s.displacement[-1] - 1), plus=True)} %."
        )

    def _draw_building(self, recs: tuple[Accelerogram, ...]) -> None:
        ground = recs[0]
        h = time_history(self.system, self.design, ground)
        self.history = h
        self.building.set_history(h.displacement)
        self.sway_item.setData(ground.t, ground.a)
        self.sway_plot.setXRange(0.0, float(h.t[-1]), padding=0)
        self.playback.set_span(float(h.t[-1]))  # places the building

    def _place(self, t: float) -> None:
        h = self.history
        if h is None or self.tabs.currentIndex() != TAB_BUILDING:
            return
        k = min(int(np.searchsorted(h.t, t)), h.t.size - 1)
        self.building.set_state(h.displacement[k])
        drift = np.abs(np.diff(h.displacement[k], prepend=0.0))
        worst = int(np.argmax(drift))
        self.building_note.setText(
            f"Base shear {fmt(abs(float(h.base_shear[k])), 'N')} (its peak on this record "
            f"{fmt(h.peak_base_shear, 'N')}); storey {worst + 1} drifts most now, {fmt(float(drift[worst]), 'm')} "
            f"(its peak {fmt(float(h.peak_drift[worst]), 'm')}). Sway drawn ×{sig(self.building.exaggeration)}, "
            f"storeys as 3 m."
        )

    def _draw_correlation(self) -> None:
        rho = self.rho
        n = rho.shape[0]
        head = "<tr><th></th>" + "".join(f"<th>{j + 1}</th>" for j in range(n)) + "</tr>"
        rows = []
        for i in range(n):
            cells = "".join(
                f"<td align='right'>{'<b>' if i != j and rho[i, j] > 0.1 else ''}{rho[i, j]:.3f}"
                f"{'</b>' if i != j and rho[i, j] > 0.1 else ''}</td>"
                for j in range(n)
            )
            rows.append(f"<tr><th>{i + 1}</th>{cells}</tr>")
        self.rho_label.setText(
            f"<p><b>ρ<sub>ij</sub></b> at ζ = {sig(100 * self.design.zeta)} %</p>"
            f"<table cellspacing='6'>{head}{''.join(rows)}</table>"
        )
        off = rho - np.diag(np.diag(rho))
        i, j = (int(k) for k in np.unravel_index(int(np.argmax(off)), off.shape))
        f = self.modal.f
        self.rho_note.setText(
            f"The most correlated pair: modes {i + 1} and {j + 1} ({sig(f[i])} and {sig(f[j])} Hz, "
            f"ρ = {sig(off[i, j])}). Above about 0.1 SRSS and CQC part; bold."
        )

    def _draw_ground(self, recs: tuple[Accelerogram, ...]) -> None:
        d = self.design
        g = recs[0]
        self.record_item.setData(g.t, g.a)
        self.record_plot.setXRange(0.0, float(g.t[-1]), padding=0)
        for item in self.match_items:
            self.match_plot.removeItem(item)
        self.match_items = []
        pen = pg.mkPen(colors.grey, width=1)
        for rec in recs:
            item = pg.PlotDataItem(1 / rec.f, rec.spectrum / G, pen=pen)
            self.match_plot.addItem(item)
            self.match_items.append(item)
        T = 1 / recs[0].f
        self.target_item.setData(T, design_sa(d, T) / G)
        mean = np.mean([rec.spectrum for rec in recs], axis=0)
        self.match_mean.setData(T, mean / G)
        err = mean / design_sa(d, T) - 1
        pga = np.mean([np.max(np.abs(rec.a)) for rec in recs]) / G
        self.ground_note.setText(
            f"{len(recs)} synthesized records (grey); their mean spectrum is within "
            f"{sig(100 * float(np.max(np.abs(err))), 2)} % of the design one. Their peak ground "
            f"acceleration averages {sig(pga, 2)} g."
        )
