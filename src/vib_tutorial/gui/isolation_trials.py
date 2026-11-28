"""Vibration isolation's Things to try, practice problems, tips and notes, without Qt (#478 item 7).

Its state is `view_models/isolation.py`'s `IsolationTrial`; `gui/isolation.py` binds the page's controls to it and
runs it live. A trial is a setup to go back to: it sets the machine and its mounts, starts it running at its speed
from rest or runs it up from standstill (`start`), and says what to look for. Its prediction is asked only once the
reader has handled the system (`lesson.TrialBox(ask_after_play=True)`), about a case they haven't tried.

Working on it: `tests/test_tutorial_numbers.py` checks the trials' numbers, `tests/test_practice.py` the problems,
`tests/test_qt_free.py` that this loads no Qt.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.isolation import G, Isolator, Mount, deflection_for
from ..core.practice import Placement, Prediction, Problem, Question, at_most, kept, number, pick, rounded
from ..core.sdof import TWO_PI
from ..view_models.isolation import START_RUN, START_RUN_UP, TAB_CHART, TAB_FORCE, TAB_T, IsolationTrial, isolator
from ..view_models.live_isolation import FLOOR
from ..view_models.text import number as sig
from . import isolation_figures
from .links import load_notes, page_link
from .trials_text import trials_html

TRIALS = (
    IsolationTrial(
        "Soft mounts",
        "A 100 kg machine with a 0.01 kg·m unbalance at 1500 rpm (25 Hz) on mounts that sag 5 mm: <m>f<sub>n</sub></m> "
        "= <m>15.76/√5</m> = 7.05 Hz, so the speed is <m>3.5f<sub>n</sub></m>. Started at speed, the machine first "
        "rings on its mounts as well; once that has died away, its point lands on the Transmissibility tab. The "
        "unbalance pulls with 247 N; only 23 N, 9 %, reaches the floor (−21 dB): 91 % isolation.",
        start=START_RUN,
        predict=Prediction(
            "A machine running at 3.5 times its mounts' natural frequency. How much of its unbalance force reaches the floor?",
            ("All of it", "About a tenth", "More than all of it"),
            1,
            why=(
                "Above √2 f<sub>n</sub> a mount passes less than all of it: at 3.5 f<sub>n</sub>, 1/(r² − 1) ≈ 9 %.",
                "",
                "Only below √2 f<sub>n</sub>. At 3.5 f<sub>n</sub> the machine's inertia takes most of the force.",
            ),
        ),
    ),
    IsolationTrial(
        "Stiff mounts",
        "The same machine on mounts that sag only 0.3 mm: <m>f<sub>n</sub></m> = 28.8 Hz, just above the running "
        "speed. The mounts now amplify: 3.9 times the unbalance's force reaches the floor. Below <m>√2 f<sub>n</sub></m> "
        "a mount can only make things worse.",
        deflection_mm=0.3,
        start=START_RUN,
        predict=Placement(
            "The same machine on mounts sagging only 0.3 mm: their natural frequency is just above the speed. What "
            "share of the unbalance's force reaches the floor now? Put the line at that transmissibility.",
            plot="tr_plot",
            answer=3.86,
            span=(0.001, 10.0),
            log=True,
            low="Below √2 f<sub>n</sub> a mount can only pass more than all of it, and 25 Hz is just under f<sub>n</sub> = 28.8 Hz, "
            "near resonance: 3.9 times.",
            high="Near resonance, but not on it: at r = 0.87, with ζ = 0.05, the mounts pass 1/(1 − r²) ≈ 3.9 times "
            "the force.",
            tolerance=0.3,
        ),
    ),
    IsolationTrial(
        "Soft mounts, more damping",
        "Back on the soft mounts, with <m>ζ</m> = 0.3 instead of 0.05. Above <m>√2 f<sub>n</sub></m> the damper passes force "
        "to the floor in proportion to the speed: 20 % now reaches it instead of 9 %. Damping only helps at "
        "resonance, which the machine passes through at start-up.",
        zeta=0.3,
        start=START_RUN,
        predict=Prediction(
            "Back on the soft mounts, with ζ = 0.3 instead of 0.05. Does the floor get more or less?",
            ("Less: damping always helps", "More: about twice as much", "The same"),
            1,
            why=(
                "Only at resonance. Above √2 f<sub>n</sub> the damper's force grows with speed and passes more to the floor: 20 % against 9 %.",
                "",
                "Above √2 f<sub>n</sub> the damper's force, cΩX, adds to what the spring passes: more damping, more force on the floor.",
            ),
        ),
    ),
    IsolationTrial(
        "Running up through resonance",
        "A 2 s run up from standstill to 1500 rpm on the soft, lightly damped mounts. As the speed passes "
        "<m>f<sub>n</sub></m> = 7 Hz (420 rpm) the machine and the floor force swell, then settle to the isolated "
        "running state: the Floor force tab builds each turn's force against the speed, peaking at 111 N. The "
        "unbalance's force grows as the speed squared, so the resonance is passed at a twelfth of the running "
        "force, which is what keeps it tolerable. Try Run down, and a slower run.",
        tab=TAB_FORCE,
        start=START_RUN_UP,
    ),
    IsolationTrial(
        "An elastically supported damper",
        "A relaxation mount at 3000 rpm: the damper sits on a spring 3 times as stiff as the mount's, with "
        "<m>ζ</m> = 0.3. At resonance it passes a little more than the plain mount of the same <m>ζ</m> (2.1 times against "
        "2.0), but high above it the spring in series lets the damper float, and the transmissibility falls as "
        "<m>1/r²</m> again, not <m>1/r</m>: 7.0 % reaches the floor against the plain mount's 8.8 % (at 6000 rpm, 1.9 % "
        "against 4.3 %).",
        zeta=0.3,
        mount=Mount.RELAXATION,
        rpm=3000.0,
        start=START_RUN,
    ),
    IsolationTrial(
        "Two stages",
        "The machine on mounts on a 50 kg intermediate mass, itself on mounts sagging 5 mm under both. Two resonances, but "
        "above them the transmissibility falls as <m>1/r⁴</m> (80 dB a decade) instead of <m>1/r²</m>: 3.8 % reaches the "
        "floor here (one stage: 9 %), and at 3000 rpm the two stages pass 0.2 % against one stage's 2.5 %.",
        mount=Mount.TWO_STAGE,
        start=START_RUN,
        predict=Prediction(
            "One stage passes 9 %. Now two stages, a 50 kg mass between them, each on mounts sagging 5 mm. They pass?",
            ("More than 9 %", "About 4 %", "About 9 %"),
            1,
            why=(
                "Two stages bring two resonances, but well above both the transmissibility falls as 1/r⁴, not 1/r².",
                "",
                "The second stage isolates what the first passes, again: 3.8 %.",
            ),
        ),
    ),
    IsolationTrial(
        "An inertia block",
        "The machine bolted to a 300 kg block, the mounts stiffened to sag the same 5 mm. The transmissibility "
        "doesn't change (<m>f<sub>n</sub></m> depends only on <m>δ</m>), nor does the force on the floor, but the machine "
        "moves a quarter as much (±27 µm instead of ±109 µm): a block steadies the machine, it doesn't "
        "isolate better.",
        block=300.0,
        start=START_RUN,
        predict=Prediction(
            "Bolt the machine to a 300 kg block, the mounts stiffened to sag the same 5 mm. What changes?",
            ("Less force reaches the floor", "The machine moves less; the floor's force stays", "Nothing at all"),
            1,
            why=(
                "The transmissibility depends only on f<sub>n</sub>, which the 5 mm sag sets: the same share of the unbalance gets through.",
                "",
                "The machine's swing is the unbalance over the whole mass: with 300 kg added it's a quarter as large.",
            ),
        ),
    ),
    IsolationTrial(
        "The floor shakes instead",
        "The soft mounts again, the machine's unbalance stopped and the floor under the mounts moving ±0.1 mm at "
        "25 Hz, as from a press nearby. The machine moves ±9.16 µm: T = 0.0916, the same share as the force the "
        "machine sent into the floor. Its point (◆) lands on the machine's (●): one curve answers both questions.",
        shaking=FLOOR,
        start=START_RUN,
        predict=Prediction(
            "The machine's unbalance stops, and the floor under its soft mounts shakes ±0.1 mm at 25 Hz instead. How "
            "much of the floor's motion reaches the machine?",
            ("All of it: the mounts are soft", "About a tenth, as the force did", "Nothing at all"),
            1,
            why=(
                "Soft mounts can't make the machine follow a floor that moves faster than √2 f<sub>n</sub>: its inertia "
                "holds it nearly still.",
                "",
                "Some does: the spring and the damper still push on the machine as the floor moves, 9 % of the way.",
            ),
        ),
    ),
)


# ------------------------------------------------------------------ practice problems
# The page's machine: 100 kg with a 0.01 kg·m unbalance (the trials and the problems keep it).


def _mount_frequency(rng: np.random.Generator) -> Question:
    sag = pick(rng, (0.5, 1, 2, 3, 5, 8, 10, 20))
    fn = Isolator(deflection=1e-3 * sag).natural_hz
    return Question(
        f"A machine sags {sig(sag, 6)} mm on its mounts. What is its natural frequency on them?",
        fn,
        f"k = Mg/δ, so ω<sub>n</sub> = √(k/M) = √(g/δ), whatever the machine weighs: f<sub>n</sub> = "
        f"√(9.81/{number(1e-3 * sag)})/2π = {number(fn)} Hz (15.76/√δ with δ in mm).",
        unit="Hz",
        setup=dict(deflection_mm=sag, tab=TAB_T),
    )


def _transmissibility(rng: np.random.Generator) -> Question:
    sag = pick(rng, (2, 3, 5, 8, 10))
    rpm = pick(rng, (900, 1200, 1500, 1800, 3000))
    zeta = pick(rng, (0.02, 0.05, 0.1, 0.2))
    iso = Isolator(rpm=rpm, deflection=1e-3 * sag, zeta=zeta)
    r = iso.speed_hz / iso.natural_hz
    T = iso.at_speed()[2]
    return Question(
        f"A machine runs at {sig(rpm, 6)} rpm on mounts that sag {sig(sag, 6)} mm, with ζ = {sig(zeta, 6)}. What percentage of its "
        "unbalance's force reaches the floor?",
        100.0 * T,
        f"f<sub>n</sub> = 15.76/√{sig(sag, 6)} = {number(iso.natural_hz)} Hz and the speed is {number(iso.speed_hz)} Hz, "
        f"so r = {number(r)}; T = √(1 + (2ζr)²)/√((1 − r²)² + (2ζr)²) = {number(T)}: {number(100.0 * T)} %.",
        unit="%",
        tolerance=0.03,
        setup=dict(deflection_mm=sag, rpm=rpm, zeta=zeta, tab=TAB_T),
    )


def _sag_for(rng: np.random.Generator) -> Question:
    rpm = pick(rng, (600, 900, 1200, 1500, 1800, 3000))
    isolation = pick(rng, (0.8, 0.9, 0.95))
    sag = deflection_for(rpm / 60.0, isolation)
    T = 1.0 - isolation
    r = math.sqrt(1.0 + 1.0 / T)
    return Question(
        f"Undamped mounts are to isolate {sig(100 * isolation, 6)} % of a machine's unbalance at {sig(rpm, 6)} rpm. How far must "
        "they sag under it?",
        1e3 * sag,
        f"Undamped and above √2 f<sub>n</sub>, T = 1/(r² − 1) = {number(T)}, so r = √(1 + 1/T) = {number(r)}: "
        f"f<sub>n</sub> = {number(rpm / 60.0)}/{number(r)} = {number(rpm / 60.0 / r)} Hz, and δ = "
        f"g/(2πf<sub>n</sub>)² = {number(1e3 * sag)} mm.",
        unit="mm",
        setup=dict(deflection_mm=rounded(1e3 * sag), rpm=rpm, zeta=0.0, tab=TAB_CHART),
    )


def _stiff_or_soft(rng: np.random.Generator) -> Question:
    rpm = pick(rng, (900, 1500, 3000))
    ratio = pick(rng, (0.6, 0.8, 1.1, 2.0, 3.0, 4.0))
    fn = rpm / 60.0 / ratio
    sag = rounded(1e3 * G / (TWO_PI * fn) ** 2, 2)
    iso = Isolator(rpm=rpm, deflection=1e-3 * sag)
    T = iso.at_speed()[2]
    r = iso.speed_hz / iso.natural_hz
    return Question(
        f"A machine running at {sig(rpm, 6)} rpm is put on mounts that sag {number(sag, 2)} mm (ζ = 0.05). Does more or "
        "less force reach the floor than if it were bolted down?",
        0 if T > 1 else 1,
        f"f<sub>n</sub> = 15.76/√{number(sag, 2)} = {number(iso.natural_hz)} Hz, so the speed is r = {number(r)} "
        f"times it, {'below' if r < math.sqrt(2.0) else 'above'} √2: T = {number(T)}. Only above √2 f<sub>n</sub> "
        "does a mount isolate.",
        choices=("More: the mounts amplify it", "Less: the mounts isolate it", "The same"),
        why=(
            "" if T > 1 else f"Mounts amplify only below √2 f<sub>n</sub>, and here r = {number(r)} is above it.",
            "" if T <= 1 else f"Mounts isolate only above √2 f<sub>n</sub>, and here r = {number(r)} is below it.",
            f"The force is the same only at r = √2 exactly; here r = {number(r)}.",
        ),
        setup=dict(deflection_mm=sag, rpm=rpm, tab=TAB_T),
        predict=True,
    )


def _damping_above(rng: np.random.Generator) -> Question:
    zeta = pick(rng, (0.2, 0.3, 0.5))
    before_, after = (Isolator(zeta=z).at_speed()[2] for z in (0.05, zeta))
    return Question(
        f"A machine at 1500 rpm on mounts that sag 5 mm (f<sub>n</sub> = 7 Hz, well below its 25 Hz) has ζ = 0.05. "
        f"The mounts are changed for ones that sag the same but have ζ = {sig(zeta, 6)}. What happens to the force on "
        "the floor while it runs?",
        0,
        f"It rises: T goes from {number(100 * before_)} % to {number(100 * after)} %. Above √2 f<sub>n</sub> the "
        "damper pushes on the floor in proportion to the speed; damping only helps near resonance, at start-up.",
        choices=("It rises", "It falls", "It stays the same"),
        why=(
            "",
            "Damping helps only near resonance, at start-up. Above √2 f<sub>n</sub> the damper's force grows with "
            "the speed and goes straight to the floor, so more of it lets more through.",
            "The damper's force adds to the spring's on the floor: above √2 f<sub>n</sub>, more damping lets more "
            "through.",
        ),
        setup=dict(zeta=zeta, tab=TAB_T),
        predict=True,
    )


def _isolate_spec(rng: np.random.Generator) -> Question:
    rpm = pick(rng, (900, 1200, 1500, 1800, 3000))
    isolation = pick(rng, (0.8, 0.9, 0.95))
    allowed = 100.0 * (1.0 - isolation)
    need = 1e3 * deflection_for(rpm / 60.0, isolation)  # mm, undamped
    design = rounded(1.2 * need, 2)
    most = rounded(design * pick(rng, (1.3, 1.6)), 2)
    T = 100.0 * Isolator(rpm=rpm, deflection=1e-3 * design).at_speed()[2]

    def goal(t: IsolationTrial):
        return (
            kept("The speed", t.rpm, rpm, "rpm"),
            at_most("The force reaching the floor, T", 100.0 * isolator(t).at_speed()[2], allowed, "%"),
            at_most("The mounts' sag", t.deflection_mm, most, "mm"),
        )

    return Question(
        f"The machine runs at {sig(rpm, 6)} rpm and sits on stiff mounts. Choose mounts so no more than {sig(allowed, 6)} % of "
        f"its unbalance's force reaches the floor ({sig(100 * isolation, 6)} % isolation), without letting it sag more "
        f"than {number(most, 2)} mm (any more and its pipes won't line up).",
        0,
        f"Undamped, T = 1/(r² − 1) = {sig(1 - isolation, 6)} needs r = {number(math.sqrt(1 + 1 / (1 - isolation)))}, "
        f"so a sag of at least {number(need)} mm; damping costs a little, so take a little more: {number(design, 2)} "
        f"mm with ζ = 0.05 lets {number(T)} % through. The Design chart's <i>Use that sag</i> gives the undamped "
        "sag for an isolation.",
        setup=dict(rpm=rpm, deflection_mm=rounded(0.3 * need, 2), tab=TAB_CHART),
        goal=goal,
        design=dict(rpm=rpm, deflection_mm=design, tab=TAB_CHART),
    )


def _trade_off_spec(rng: np.random.Generator) -> Question:
    rpm = pick(rng, (1200, 1500, 1800, 3000))
    sag = pick(rng, (3, 5, 8, 10))
    zeta = pick(rng, (0.1, 0.15, 0.2, 0.25))
    iso = Isolator(rpm=rpm, deflection=1e-3 * sag, zeta=zeta)
    peak = rounded(1.1 * iso.peak_transmissibility(), 2)
    allowed = rounded(1.15 * 100.0 * iso.at_speed()[2], 2)

    def goal(t: IsolationTrial):
        now = isolator(t)
        return (
            kept("The speed", t.rpm, rpm, "rpm"),
            at_most("T's peak, at any speed", now.peak_transmissibility(), peak),
            at_most("T at the running speed", 100.0 * now.at_speed()[2], allowed, "%"),
            at_most("The mounts' sag", t.deflection_mm, sag, "mm"),
        )

    return Question(
        f"The machine runs at {sig(rpm, 6)} rpm on mounts that sag {sig(sag, 6)} mm, with hardly any damping (ζ = 0.02): it "
        f"shakes the floor as it runs up through their resonance. Tune the mounts so the transmissibility never "
        f"exceeds {number(peak, 2)} at any speed, while at {sig(rpm, 6)} rpm no more than {number(allowed, 2)} % of the "
        f"force reaches the floor. The mounts may sag {sig(sag, 6)} mm at most.",
        0,
        f"Damping lowers the peak (about 1/2ζ) but raises T at speed (the damper's force grows with the speed), "
        f"so there's a window: ζ = {sig(zeta, 6)} at {sig(sag, 6)} mm gives a peak of {number(iso.peak_transmissibility())} "
        f"and {number(100 * iso.at_speed()[2])} % at speed. Less sag (stiffer mounts) only makes T at speed worse.",
        setup=dict(rpm=rpm, deflection_mm=float(sag), zeta=0.02, tab=TAB_FORCE),
        goal=goal,
        design=dict(rpm=rpm, deflection_mm=float(sag), zeta=zeta, tab=TAB_FORCE),
    )


def _block_spec(rng: np.random.Generator) -> Question:
    rpm = pick(rng, (900, 1200, 1500, 1800))
    sag = pick(rng, (2, 3, 5))
    block = pick(rng, (50, 100, 200, 300))
    iso = Isolator(rpm=rpm, deflection=1e-3 * sag, block=block)
    most = rounded(1.1 * 1e6 * iso.at_speed()[0], 2)
    r2 = (iso.speed_hz / iso.natural_hz) ** 2
    least = 0.01 * r2 / ((r2 - 1.0) * 1e-6 * most) - 100.0

    def goal(t: IsolationTrial):
        return (
            kept("The speed", t.rpm, rpm, "rpm"),
            kept("The machine's mass", t.mass, 100.0, "kg"),
            kept("Its unbalance", t.unbalance, 0.01, "kg·m"),
            kept("The mounts' sag", t.deflection_mm, float(sag), "mm"),
            at_most("The machine's motion", 1e6 * isolator(t).at_speed()[0], most, "µm"),
        )

    return Question(
        f"The 100 kg machine (0.01 kg·m of unbalance) runs at {sig(rpm, 6)} rpm on mounts that sag {sig(sag, 6)} mm. It swings "
        f"too much for the pipes bolted to it: bring its motion down to ±{number(most, 2)} µm, keeping the mounts' "
        "sag (they're stiffened to suit whatever they carry).",
        0,
        f"High above resonance the mounts hardly hold the machine: it swings about its center of mass, X = "
        f"m<sub>e</sub>e r²/((M + B)(r² − 1)) with r² = {number(r2)} fixed by the sag. So it needs M + B ≥ "
        f"0.01 × {number(r2)}/({number(r2 - 1)} × {number(most, 2)}×10<sup>−6</sup>): an inertia block of at least "
        f"{number(max(least, 0.0), 2)} kg. A {sig(block, 6)} kg block gives ±{number(1e6 * iso.at_speed()[0])} µm.",
        setup=dict(rpm=rpm, deflection_mm=float(sag), tab=TAB_T),
        goal=goal,
        design=dict(rpm=rpm, deflection_mm=float(sag), block=float(block), tab=TAB_T),
    )


PROBLEMS = (
    Problem(
        "mount-frequency",
        "Natural frequency from the sag",
        "isolation_page",
        f"See {page_link('isolation_page', 'static-deflection', 'Static deflection')}.",
        _mount_frequency,
    ),
    Problem(
        "transmissibility",
        "Force on the floor",
        "isolation_page",
        f"See {page_link('isolation_page', 'transmissibility', 'Transmissibility')}: first r = f/f<sub>n</sub>.",
        _transmissibility,
    ),
    Problem(
        "sag-for",
        "Mounts for an isolation",
        "isolation_page",
        f"See {page_link('isolation_page', 'static-deflection', 'Static deflection')}: undamped, T = 1/(r² − 1).",
        _sag_for,
    ),
    Problem(
        "stiff-or-soft",
        "Predict: isolate or amplify?",
        "isolation_page",
        f"See {page_link('isolation_page', 'transmissibility', 'Transmissibility')}: where does T cross 1?",
        _stiff_or_soft,
    ),
    Problem(
        "damping-above",
        "Predict: more damping, running",
        "isolation_page",
        f"See {page_link('isolation_page', 'damping-trade-off', 'The damping trade-off')}.",
        _damping_above,
    ),
    Problem(
        "isolate-spec",
        "Spec: isolate the machine",
        "isolation_page",
        f"See {page_link('isolation_page', 'static-deflection', 'Static deflection')}: the softer the mounts, the "
        "better they isolate.",
        _isolate_spec,
    ),
    Problem(
        "trade-off-spec",
        "Spec: through resonance and at speed",
        "isolation_page",
        f"See {page_link('isolation_page', 'damping-trade-off', 'The damping trade-off')}.",
        _trade_off_spec,
    ),
    Problem(
        "block-spec",
        "Spec: steady the machine",
        "isolation_page",
        f"See {page_link('isolation_page', 'inertia-block', 'An inertia block')}.",
        _block_spec,
    ),
)

TIPS = {
    "mass": "<p>The machine's mass M.</p>",
    "rpm": "<p>Its running speed. Running, it changes at once; stopped, it's the speed Run up goes to. The clock "
    "under the plots shows the speed now, with ▲ or ▼ while it climbs or slows.</p>",
    "unbalance": "<p>Its unbalance m<sub>e</sub>e: an off-center mass times its radius. Its force is "
    "m<sub>e</sub>eΩ².</p>",
    "deflection": "<p>How far the mounts sag under the machine's weight. Since k = Mg/δ, the natural frequency "
    "is √(g/δ)/2π = 15.76/√δ<sub>mm</sub> Hz, whatever the mass: the softer the mount, the lower.</p>",
    "zeta": "<p>The mounts' damping ratio.</p>",
    "mount": "<p><b>Spring + damper:</b> side by side. <b>Relaxation:</b> the damper in series with a spring N "
    "times the mount's, so it can't stiffen the mount at high frequency. <b>Two stages:</b> the machine on "
    "mounts on an intermediate mass, itself on mounts.</p>",
    "ratio": "<p>The relaxation damper's spring, over the mount's (N).</p>",
    "stage": "<p>The intermediate mass, over the machine's.</p>",
    "block": "<p>An inertia block bolted to the machine. The mounts are stiffened to keep their sag.</p>",
    "target": "<p>The isolation wanted at the running speed: the share of the unbalance's force kept off the "
    "floor.</p>",
    "ramp": "<p>How long Run up and Run down take to change the speed, at a steady rate.</p>",
    "run_up": "<p>Run the machine up to the Speed box's speed at a steady rate (from standstill in the time "
    "below), through the mounts' resonance; lit while it climbs, and pressed again it holds the speed it's "
    "reached. The Floor force tab builds the force on the floor as it goes. The page opens with the machine "
    "stopped, so a pull and let go rings it freely.</p>",
    "run_down": "<p>Run the machine down to a stop at the same rate: it coasts through resonance the other way. "
    "Lit while it slows; pressed again, it holds the speed.</p>",
    "shaking": "<p><b>The machine:</b> its unbalance shakes it, and the mounts pass some of that force to the floor. "
    "<b>The floor:</b> the unbalance stops and the floor under the mounts moves ±0.1 mm at the speed's frequency, "
    "as from a machine nearby: the same ±0.1 mm at every speed, not growing as the unbalance's force does. The "
    "mounts pass some of that motion to the machine. Both share are the transmissibility, and "
    "their points land on one curve.</p>",
    "theory": "<p>Draw the theory over what's measured: this isolator's transmissibility and plain mounts' at "
    "ζ = 0, 0.1, 0.3 and 0.7; on the Floor force tab, the steady force at each speed.</p>",
}
TR_TIP = (
    "<p>The transmissibility you've measured: at each steady speed, the force on the floor over the unbalance's "
    "(●), or the machine's motion over the floor's (◆), against the speed over f<sub>n</sub>. A point lands once the "
    "mounts' own ring has died away; faded points are from before a change. <i>Show the theory</i> draws this "
    "isolator's curve and plain mounts' at ζ = 0, 0.1, 0.3 and 0.7. It crosses 1 at √2 f<sub>n</sub>. Drag the "
    "speed's line (◀▶) to run the machine faster or slower.</p>"
)
SPEED_LINE_TIP = (
    "<p>The running speed over f<sub>n</sub>. Drag it to run the machine faster or slower: the Speed box follows, "
    "and once the swing settles a point lands. Below √2 the mounts pass on more than the unbalance makes; well "
    "above, less.</p>"
)
FORCE_TIP = (
    "<p>Each whole turn's largest force on the floor against the speed, built as the machine runs up or down "
    "(the floor shaking: the machine's largest motion), beside the unbalance's own force, what a rigid mount would "
    "pass (dashed). <i>Show the theory</i> adds the steady force at each speed.</p>"
)
CHART_TIP = (
    "<p>The static deflection undamped mounts need, against the running speed, for each isolation. The circle is "
    "this machine.</p>"
)
STRIP_TIP = (
    "<p>The machine's displacement (the floor shaking, the floor's too, gray), the last 8 s. A dashed line marks a "
    "change of the machine or its mounts. Paused, drag the line at the end back to see the machine then.</p>"
)
FORCES_TIP = (
    "<p>The force the mounts put on the floor, and the unbalance's own force along them (dashed): what a rigid "
    "mount would pass.</p>"
)


NOTES_HTML = load_notes("isolation", figures=isolation_figures.FIGURES, trials=trials_html(TRIALS))
