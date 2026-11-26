"""Self-excited vibration page: van der Pol, galloping, vortex-induced vibration and flutter.

Left: the model and its parameters. Center: its motion from a small start (time history and phase plane), or for
flutter the two modes' frequencies and damping against the speed. Right: the Motion tab (the body drawn moving,
`self_excited_view.SectionView`, played by a `Playback` over the run, or over flutter's least stable mode,
`Flutter.mode`; a slider for the drive whose linearized poles follow it as it's dragged, the model rerun when
it's let go), the sweep (the steady amplitude against
the parameter that drives it, which shows the Hopf point and any hysteresis; for flutter the root locus, a pole
dragged along it setting the speed), and the Theory (its figures:
gui/self_excited_figures.py). The page has its own models (core/self_excited.py).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field, replace

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.practice import Prediction, Problem, Question, Requirement, at_least, at_most, kept, number, pick, rounded
from ..core.self_excited import (
    PARKINSON,
    TWO_PI,
    Flutter,
    Galloping,
    VanDerPol,
    Viv,
    dominant_hz,
    galloping_branches,
    settled_amplitude,
)
from ..view_models.text import number as sig
from . import coalesce, self_excited_figures, settle_plots
from .drag import DragDot
from .lesson import (
    Form,
    TrialBox,
    column,
    combo,
    connect_all,
    mark,
    marker,
    note_label,
    plot,
    readout_label,
    relabel,
    set_quietly,
    stack,
    theory,
    three_panes,
)
from .links import load_notes, page_link
from .playback import Playback
from .plots import decimate
from .self_excited_view import Body, SectionView
from .style import colors
from .theming import add_legend
from .trials_text import trials_html


class Kind(enum.Enum):
    VDP = "Van der Pol (negative damping)"
    GALLOP = "Galloping (a square section in wind)"
    VIV = "Vortex-induced vibration"
    FLUTTER = "Flutter (a wing section)"


KINDS = tuple(Kind)
LIFTS = ("Steady", "Quasi-steady")  # the tip gives each one's L
TAB_MOTION, TAB_SWEEP = range(2)
VIV_SWEEP = np.linspace(2.0, 14.0, 41)
MM = frozenset({"vdp_ref", "vdp_start", "g_start"})  # shown in mm, as the plots and readouts are; kept in m
DRIVE_STEPS = 200
FLUTTER_CYCLES = 16  # the flutter mode played this many cycles
DENSE_PER_PX = 4  # samples a pixel past which the time plot draws without antialiasing
FLUTTER_START, FLUTTER_CAP = 0.15, 0.6  # its amplitude at the start, and where growth stops being drawn


@dataclass(frozen=True)
class SelfTrial:
    name: str
    note: str
    kind: Kind
    values: dict = field(default_factory=dict)  # control name → value
    tab: int = TAB_MOTION
    quasi_steady: bool = False  # flutter's lift: steady (θ) or quasi-steady (θ + ḣ/U)
    predict: Prediction | None = None  # asked before it runs (Things to try)


TRIALS = (
    SelfTrial(
        "Negative damping at rest",
        "<m>ẍ + ω<sub>n</sub>²x = ω<sub>n</sub>(ε − x²/x<sub>r</sub>²)ẋ</m> with <m>ε</m> = 0.2: at small amplitude the damping is negative (<m>ζ</m> = −0.1), "
        "so a 0.5 mm start grows; at large amplitude it is positive. It settles where the cycle's energy in and out "
        "balance: 8.9 mm, 2<m>x<sub>r</sub></m>√<m>ε</m>. Start it anywhere and it ends on the same closed curve in the phase "
        "plane: a limit cycle.",
        Kind.VDP,
        {"vdp_eps": 0.2},
        predict=Prediction(
            "ε = 0.2: negative damping at small amplitude, positive at large. Started at 0.5 mm, what does the motion"
            " do?",
            (
                "It dies away",
                "It grows without limit",
                "It grows and settles at one amplitude, wherever it starts",
            ),
            2,
            why=(
                'At small amplitude the damping is negative, ζ = −0.1: a small start grows.',
                'At large amplitude the damping turns positive. It settles where the energy in and out balance: 8.9 mm, a limit cycle.',
                '',
            ),
        ),
    ),
    SelfTrial(
        "The Hopf point",
        "<m>ε</m> = −0.1: positive damping everywhere near rest, and the start dies away. The Sweep tab shows the settled "
        "amplitude against <m>ε</m>: zero below <m>ε</m> = 0, then 2<m>x<sub>r</sub></m>√<m>ε</m>, rising steeply from nothing. At <m>ε</m> = 0 the "
        "pair of poles crosses the imaginary axis: a (supercritical) Hopf bifurcation.",
        Kind.VDP,
        {"vdp_eps": -0.1},
        tab=TAB_SWEEP,
    ),
    SelfTrial(
        "Relaxation oscillations",
        "<m>ε</m> = 5: the nonlinearity dominates. The motion creeps, then snaps: a period of 1.84 s against 1 s for small "
        "ε, and a squared-off wave. A dripping tap, a squealing brake and a heartbeat oscillate this way.",
        Kind.VDP,
        {"vdp_eps": 5.0, "vdp_periods": 20.0},
    ),
    SelfTrial(
        "Galloping: Den Hartog's criterion",
        "A 100 mm square section (5 kg/m, 2 Hz, <m>ζ</m> = 0.5 %) across a 4.5 m/s wind. Moving up, it sees the wind "
        "come from above; for a square, the force that angle makes pushes it further up (Den Hartog: d<m>C<sub>L</sub></m>/d<m>α</m> "
        "+ <m>C<sub>D</sub></m> &lt; 0). The wind's negative damping ½<m>ρUDA<sub>1</sub></m> beats the structure's 2<m>mζω<sub>n</sub></m> above "
        "<m>U<sub>g</sub></m> = 3.81 m/s, so the motion grows, slowly this close to the onset.",
        Kind.GALLOP,
        {"g_speed": 4.5, "g_start": 0.001, "g_periods": 300.0},
    ),
    SelfTrial(
        "Galloping's hysteresis",
        "Between 4.7 and 7.0 m/s a square section has two stable motions, at 6 m/s 0.51 <m>D</m> and 1.28 <m>D</m>, with an "
        "unstable one between (Sweep tab: filled and open dots). From a small start it settles on the lower; set "
        "the start to 0.12 m and it settles on the upper. Raise the wind past 7.0 m/s and the lower one is gone: "
        "the motion jumps to the upper, and stays there until the wind drops below 4.7 m/s.",
        Kind.GALLOP,
        {"g_speed": 6.0, "g_start": 0.001, "g_periods": 600.0},
        tab=TAB_SWEEP,
    ),
    SelfTrial(
        "Galloping at 12 m/s",
        "At 12 m/s, three times the onset speed, the section swings 2.7 <m>D</m> from side to side. Unlike flutter or "
        "resonance there is no upper limit: the amplitude keeps growing with the wind. Damping only moves the "
        "onset (<m>U<sub>g</sub></m> ∝ <m>mζ</m>, the Scruton number).",
        Kind.GALLOP,
        {"g_speed": 12.0, "g_start": 0.001, "g_periods": 200.0},
        predict=Prediction(
            'The square section in a 12 m/s wind, three times its galloping onset. Does its motion level off?',
            (
                "Yes, at about one diameter, as vortex shedding does",
                "No: it keeps growing with the wind",
            ),
            1,
            why=(
                "Vortex shedding locks in and limits itself. Galloping draws on the wind's speed, and its amplitude keeps growing with it: 2.7 D here.",
                '',
            ),
        ),
    ),
    SelfTrial(
        "Lock-in",
        "A 50 mm cylinder (20 kg/m, mass ratio 10) on a 1 Hz spring in a current at reduced velocity "
        "<m>U</m>/<m>f<sub>n</sub>D</m> = 5.5. Vortices would shed at St <m>U</m>/<m>D</m> = 1.1 <m>f<sub>n</sub></m>, but the structure's motion "
        "pulls the wake to its own frequency: it swings 0.21 <m>D</m> at <m>f<sub>n</sub></m>. The Sweep tab shows lock-in, the "
        "response within 4 % of <m>f<sub>n</sub></m>, from <m>U<sub>r</sub></m> = 4.5 to 6.25: a band of speeds, not a single "
        "resonance.",
        Kind.VIV,
        {"v_reduced": 5.5},
        tab=TAB_SWEEP,
    ),
    SelfTrial(
        "A reduced velocity of 9",
        "<m>U<sub>r</sub></m> = 9: the wake sheds near its own Strouhal frequency, 1.8 <m>f<sub>n</sub></m>, far above the "
        "structure's (the response is at 1.7 <m>f<sub>n</sub></m>), and it is a forced one: 0.019 <m>D</m>. Vortex-induced vibration is self-limited "
        "(about one diameter at most, in practice), unlike galloping.",
        Kind.VIV,
        {"v_reduced": 9.0},
        predict=Prediction(
            "The cylinder swung 0.21 D at a reduced velocity of 5.5, locked in. The current rises to a reduced "
            "velocity of 9. What does the motion do?",
            (
                "It stays about the same",
                "It keeps growing with the current",
                "It falls to a small forced motion",
            ),
            2,
            why=(
                'Past the lock-in band the wake stops following the structure, so the large swing goes.',
                'Vortex-induced vibration lives in a band of speeds: past lock-in the wake sheds at its own Strouhal frequency, 1.8 f<sub>n</sub>, and no longer drives the structure at resonance.',
                '',
            ),
        ),
    ),
    SelfTrial(
        "A wing section, speeding up",
        "A wing section with plunge at 3 Hz and pitch at 6 Hz, its center of mass behind the elastic axis, in steady "
        "lift. As the speed rises the lift, a stiffness growing as <m>U</m>², couples the two modes: their "
        "frequencies close in (2.93 and 6.71 Hz at rest, 3.6 and 4.7 Hz at 20 m/s) and at 20.7 m/s, nearly met, "
        "the plunge-like mode's damping falls to zero: the flutter speed, well below the 39 m/s at which the pitch "
        "spring alone would diverge. Without structural damping (<m>ζ</m> = 0) they meet exactly, at 4.07 Hz and "
        "20.9 m/s.",
        Kind.FLUTTER,
        {"f_speed": 20.0},
        predict=Prediction(
            "The wing section's plunge and pitch modes are at 2.9 and 6.7 Hz in still air. As the airspeed rises "
            "toward flutter, what do their frequencies do?",
            (
                "Both fall to zero",
                "They stay where they are",
                "They move apart",
                "They close in",
            ),
            3,
            why=(
                'A frequency falling to zero is divergence, the pitch spring giving out; flutter is two modes closing in.',
                'The airflow adds stiffness and coupling: 2.9 and 6.7 Hz move to 3.6 and 4.7 Hz at 20 m/s.',
                'The steady lift couples plunge and pitch, pulling their frequencies together: flutter comes as they close in.',
                '',
            ),
        ),
    ),
    SelfTrial(
        "Mass balance",
        "Move the center of mass onto the elastic axis, <m>x<sub>cg</sub></m> = 0: the inertia no longer couples plunge "
        "and pitch, their frequencies never meet, and the section no longer flutters; it only diverges, at 39 m/s, "
        "where the pitch spring gives out. Mass-balancing control surfaces (lead in the leading edge) does this. "
        "Halfway, at <m>x<sub>cg</sub></m> = 0.1, flutter waits until 22.8 m/s.",
        Kind.FLUTTER,
        {"f_cg": 0.0, "f_speed": 20.0},
        predict=Prediction(
            "The section's center of mass moved onto its elastic axis (x<sub>cg</sub> = 0). What happens to its "
            "flutter?",
            (
                "Nothing changes",
                "It flutters at a lower speed",
                "It no longer flutters; it only diverges, at 39 m/s",
            ),
            2,
            why=(
                "The center of mass's place sets the inertial coupling between plunge and pitch: moved ahead of the axis, it stops the flutter.",
                "Moved ahead of the elastic axis, the center of mass couples plunge and pitch the other way, and they no longer feed each other: that's mass balance.",
                '',
            ),
        ),
    ),
    SelfTrial(
        "Quasi-steady lift: flutter without coalescence",
        "The same section, its lift now following the plunge's <m>ḣ</m>/<m>U</m> too, the angle the plunging "
        "section sees. That term is a damping: it damps the plunge (<m>ζ</m> = 0.1 at 19 m/s) and, through the "
        "lift's moment, feeds the pitch from the plunge's velocity. The pitch-like mode goes unstable on its own, at "
        "19.1 m/s, its frequency still well apart from the plunge's (5.0 and 3.4 Hz), and mass balance no longer "
        "saves it: with <m>x<sub>cg</sub></m> = 0 it flutters at 17.7 m/s. The lift model decides how a wing "
        "flutters: this one leaves out the lag and the pitch-rate terms of unsteady aerodynamics, which damp the "
        "pitch.",
        Kind.FLUTTER,
        {"f_speed": 19.0},
        quasi_steady=True,
    ),
)

DEFAULTS = {
    "vdp_freq": 1.0,
    "vdp_eps": 0.2,
    "vdp_ref": 0.01,
    "vdp_start": 0.0005,
    "vdp_periods": 60.0,
    "g_side": 0.1,
    "g_mass": 5.0,
    "g_fn": 2.0,
    "g_zeta": 0.005,
    "g_speed": 6.0,
    "g_start": 0.001,
    "g_periods": 300.0,
    "v_d": 0.05,
    "v_mass": 20.0,
    "v_fn": 1.0,
    "v_zeta": 0.005,
    "v_reduced": 5.5,
    "f_plunge": 3.0,
    "f_pitch": 6.0,
    "f_cg": 0.2,
    "f_e": 0.3,
    "f_zeta": 0.01,
    "f_speed": 15.0,
}  # every control's value when a trial doesn't set it

# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.
A1 = PARKINSON[0]


def _vdp_amplitude(rng: np.random.Generator) -> Question:
    eps, xr = pick(rng, (0.05, 0.1, 0.2, 0.3, 0.5)), pick(rng, (0.002, 0.005, 0.01, 0.02))
    a = float(VanDerPol(1.0, eps, xr).theory_amplitude(eps))
    return Question(
        f"A van der Pol oscillator, ẍ + ω<sub>n</sub>²x = ω<sub>n</sub>(ε − x²/x<sub>r</sub>²)ẋ, with ε = {sig(eps, 6)} and x<sub>r</sub> = "
        f"{sig(1000 * xr, 6)} mm. How large is its limit cycle's amplitude?",
        1000 * a,
        f"Averaging: over a cycle of x = a cos ωt the energy in, ε⟨ẋ²⟩, balances the energy out, ⟨x²ẋ²⟩/x<sub>r</sub>², "
        f"at a²/4 = εx<sub>r</sub>²: a = 2x<sub>r</sub>√ε = 2 × {sig(1000 * xr, 6)} × √{sig(eps, 6)} = {number(1000 * a)} mm.",
        unit="mm",
        setup=dict(kind=Kind.VDP, values={"vdp_eps": eps, "vdp_ref": xr, "vdp_periods": 120.0}),
    )


def _vdp_growth(rng: np.random.Generator) -> Question:
    eps, f = pick(rng, (0.05, 0.1, 0.2)), pick(rng, (0.5, 1, 2, 5))
    w = TWO_PI * f
    t = 2 * math.log(10) / (eps * w)
    return Question(
        f"A van der Pol oscillator at {sig(f, 6)} Hz with ε = {sig(eps, 6)} starts from a tiny displacement. How long does its "
        "amplitude take to grow ten times, while it's still small?",
        t,
        f"Near rest it is ẍ − εω<sub>n</sub>ẋ + ω<sub>n</sub>²x = 0: poles σ ± iω<sub>d</sub> with σ = εω<sub>n</sub>/2 = {number(eps * w / 2)} /s. The "
        f"envelope grows as e<sup>σt</sup>, so ten times takes ln 10/σ = {number(t)} s.",
        unit="s",
        setup=dict(kind=Kind.VDP, values={"vdp_freq": f, "vdp_eps": eps, "vdp_start": 1e-4}),
    )


def _galloping(rng: np.random.Generator) -> tuple[float, float, float, float]:
    return (
        pick(rng, (0.05, 0.1, 0.2, 0.3)),
        pick(rng, (2, 5, 10, 20)),
        pick(rng, (1, 2, 3, 5)),
        pick(rng, (0.002, 0.005, 0.01)),
    )


def _galloping_onset(rng: np.random.Generator) -> Question:
    side, mass, fn, zeta = _galloping(rng)
    m = Galloping(side, mass, fn, zeta)
    return Question(
        f"A square section {sig(1000 * side, 6)} mm across, {sig(mass, 6)} kg/m, on a spring at {sig(fn, 6)} Hz with ζ = {sig(zeta, 6)}, in "
        f"air (ρ = 1.225 kg/m³). Taking Parkinson's A<sub>1</sub> = {sig(A1, 6)}, at what wind speed does it start to gallop?",
        m.onset,
        f"The wind's damping, −½ρUDA<sub>1</sub>, beats the structure's 2mζω<sub>n</sub> where U<sub>g</sub> = 4mζω<sub>n</sub>/(ρDA<sub>1</sub>) = 4 × {sig(mass, 6)} × "
        f"{sig(zeta, 6)} × {number(m.omega)}/(1.225 × {sig(side, 6)} × {sig(A1, 6)}) = {number(m.onset)} m/s.",
        unit="m/s",
        setup=dict(kind=Kind.GALLOP, values={"g_side": side, "g_mass": mass, "g_fn": fn, "g_zeta": zeta}),
    )


def _galloping_damping(rng: np.random.Generator) -> Question:
    side, mass, fn, _ = _galloping(rng)
    w = TWO_PI * fn
    U = pick(rng, (10, 15, 20, 30))
    zeta = U * 1.225 * side * A1 / (4 * mass * w)
    return Question(
        f"A square section {sig(1000 * side, 6)} mm across, {sig(mass, 6)} kg/m, on a spring at {sig(fn, 6)} Hz, must not gallop below "
        f"{sig(U, 6)} m/s (ρ = 1.225 kg/m³, A<sub>1</sub> = {sig(A1, 6)}). What damping ratio does it need at least?",
        100 * zeta,
        f"U<sub>g</sub> = 4mζω<sub>n</sub>/(ρDA<sub>1</sub>) ≥ {sig(U, 6)} m/s, so ζ ≥ ρDA<sub>1</sub>U/(4mω<sub>n</sub>) = 1.225 × {sig(side, 6)} × {sig(A1, 6)} × {sig(U, 6)}/(4 × "
        f"{sig(mass, 6)} × {number(w)}) = {number(100 * zeta)} %.",
        unit="%",
        setup=dict(kind=Kind.GALLOP, values={"g_side": side, "g_mass": mass, "g_fn": fn, "g_speed": float(U)}),
    )


def _lock_in(rng: np.random.Generator) -> Question:
    d, fn = pick(rng, (0.02, 0.05, 0.1, 0.3)), pick(rng, (0.5, 1, 2, 5))
    m = Viv(d, freq_hz=fn)
    U = m.lock_in_at * fn * d
    return Question(
        f"A cylinder {sig(1000 * d, 6)} mm in diameter on a spring at {sig(fn, 6)} Hz (in water) stands in a current. Taking a "
        "Strouhal number of 0.2, at what current speed does the shedding frequency meet its natural frequency?",
        U,
        f"Vortices shed at St U/D; that is f<sub>n</sub> at U = f<sub>n</sub>D/St = {sig(fn, 6)} × {sig(d, 6)}/0.2 = {number(U)} "
        "m/s, a reduced velocity U/f<sub>n</sub>D of 5. Lock-in spreads over a band of speeds round it.",
        unit="m/s",
        setup=dict(kind=Kind.VIV, values={"v_d": d, "v_fn": fn, "v_reduced": 5.0}),
    )


def _divergence(rng: np.random.Generator) -> Question:
    pitch, e = pick(rng, (4, 6, 8, 10)), pick(rng, (0.1, 0.2, 0.3, 0.4))
    m = Flutter(e=e, pitch_hz=pitch)
    b = m.semichord
    inertia = m.mass * (m.radius * b) ** 2
    kt = inertia * (TWO_PI * pitch) ** 2
    U = m.divergence_speed()
    return Question(
        f"A wing section of semichord b = {sig(b, 6)} m, {sig(m.mass, 6)} kg per meter of span, radius of gyration {sig(m.radius, 6)} b, "
        f"pitches at {sig(pitch, 6)} Hz. Its lift, ½ρU² × 2b × 2π per radian (ρ = 1.225), acts {sig(e, 6)} b ahead of the elastic "
        "axis. At what speed does it diverge?",
        U,
        f"I = m(rb)² = {number(inertia)} kg·m, so k<sub>θ</sub> = I(2πf)² = {number(kt)} N·m/rad. The lift's moment "
        f"about the axis, ½ρU² × 2b × 2π × {sig(e, 6)}b per radian, uses it up at U = √(k<sub>θ</sub>/(2πρb × {sig(e, 6)}b)) = "
        f"{number(U)} m/s.",
        unit="m/s",
        setup=dict(kind=Kind.FLUTTER, values={"f_pitch": pitch, "f_e": e}),
    )


def _double_eps(rng: np.random.Generator) -> Question:
    return Question(
        "A van der Pol oscillator at ε = 0.2 settles on a limit cycle. ε is doubled. What happens to the cycle's "
        "amplitude?",
        0,
        "It grows √2 times, 8.9 mm to 12.6 mm: the amplitude is 2x<sub>r</sub>√ε.",
        choices=("It grows √2 times", "It doubles", "It stays the same", "It grows four times"),
        why=(
            "",
            "The amplitude is 2x<sub>r</sub>√ε: twice the ε is √2 times the swing.",
            "ε sets the cycle's size: 2x<sub>r</sub>√ε grows with it.",
            "The amplitude goes as √ε, not as ε²: √2 times.",
        ),
        setup=dict(kind=Kind.VDP, values={"vdp_eps": 0.4, "vdp_periods": 120.0}),
        predict=True,
    )


def _heavier(rng: np.random.Generator) -> Question:
    return Question(
        "A square section starts to gallop at 3.8 m/s. Its mass per meter is doubled, the frequency and damping kept. "
        "Where does it start to gallop now?",
        0,
        "At twice the speed, 7.6 m/s: U<sub>g</sub> = 4mζω<sub>n</sub>/(ρDA<sub>1</sub>) is proportional to the mass–damping (Scruton) "
        "number mζ.",
        choices=("At twice the speed", "At √2 times the speed", "At the same speed", "At half the speed"),
        why=(
            "",
            "U<sub>g</sub> is in proportion to the mass–damping number mζ itself, not its root: twice.",
            "A heavier section takes more of the wind's energy to move: U<sub>g</sub> grows with mζ.",
            "More mass resists galloping: U<sub>g</sub> = 4mζω<sub>n</sub>/(ρDA<sub>1</sub>) grows with it.",
        ),
        setup=dict(kind=Kind.GALLOP, values={"g_mass": 10.0, "g_speed": 6.0}),
        predict=True,
    )


def _past_lock_in(rng: np.random.Generator) -> Question:
    return Question(
        "A cylinder in a current swings 0.21 D at a reduced velocity of 5.5, locked in. The current rises to a reduced "
        "velocity of 9. What does the motion do?",
        0,
        "It falls to a small forced motion, 0.019 D: the wake sheds near its own Strouhal frequency, 1.8 f<sub>n</sub>, "
        "and no longer follows the structure. Vortex-induced vibration lives in a band of speeds.",
        choices=(
            "It falls to a small forced motion",
            "It keeps growing with the current",
            "It stays about the same",
        ),
        why=(
            "",
            "Vortex-induced vibration lives in a band of speeds: past lock-in the wake sheds at its own Strouhal "
            "frequency, 1.8 f<sub>n</sub>, and no longer drives the structure at resonance.",
            "Past the lock-in band the wake stops following the structure, so the large swing goes.",
        ),
        setup=dict(kind=Kind.VIV, values={"v_reduced": 9.0}),
        predict=True,
    )


def _coalescence(rng: np.random.Generator) -> Question:
    return Question(
        "A wing section's plunge and pitch modes are at 2.9 and 6.7 Hz in still air. As the airspeed rises toward "
        "flutter, what do their frequencies do?",
        0,
        "They close in, 3.6 and 4.7 Hz at 20 m/s, as the steady lift couples them; one mode's damping reaches zero "
        "at 20.7 m/s, the flutter speed.",
        choices=("They close in", "They move apart", "They stay where they are", "Both fall to zero"),
        why=(
            "",
            "The steady lift couples plunge and pitch, pulling their frequencies together: flutter comes as they "
            "close in.",
            "The airflow adds stiffness and coupling: 2.9 and 6.7 Hz move to 3.6 and 4.7 Hz at 20 m/s.",
            "A frequency falling to zero is divergence, the pitch spring giving out; flutter is two modes closing in.",
        ),
        setup=dict(kind=Kind.FLUTTER, values={"f_speed": 20.0}),
        predict=True,
    )


def _mass_balance(rng: np.random.Generator) -> Question:
    return Question(
        "A wing section in steady lift flutters at 21 m/s with its center of mass 0.2 semichords behind the elastic "
        "axis. The center "
        "of mass is moved to 0.2 semichords ahead of it. What happens?",
        0,
        "It no longer flutters: the inertia now couples plunge and pitch the other way, and only divergence is left, "
        "at 39 m/s, where the pitch spring gives out. That is mass balance.",
        choices=(
            "It no longer flutters; it only diverges, at 39 m/s",
            "It flutters at a lower speed",
            "Nothing changes",
        ),
        why=(
            "",
            "Moved ahead of the elastic axis, the center of mass couples plunge and pitch the other way, and they no "
            "longer feed each other: that's mass balance.",
            "The center of mass's place sets the inertial coupling between plunge and pitch: moved ahead of the "
            "axis, it stops the flutter.",
        ),
        setup=dict(kind=Kind.FLUTTER, values={"f_cg": -0.2}),
        predict=True,
    )


def _scruton_spec(rng: np.random.Generator) -> Question:
    side, fn = pick(rng, (0.1, 0.2)), pick(rng, (1, 2, 3))
    times = pick(rng, (3, 4))
    m0, z0 = DEFAULTS["g_mass"], DEFAULTS["g_zeta"]
    start = Galloping(side, m0, fn, z0)
    wind = rounded(times * start.onset, 2)
    stretch = math.sqrt(1.3 * times)
    heaviest, most = rounded(m0 * stretch, 2), rounded(z0 * stretch, 2)
    design = {"g_mass": round(heaviest / 1.05, 2), "g_zeta": round(most / 1.05, 4)}

    def goal(t: SelfTrial):
        v = {**DEFAULTS, **t.values}
        g = Galloping(v["g_side"], v["g_mass"], v["g_fn"], v["g_zeta"])
        return (
            Requirement("Galloping", t.kind.value, t.kind is Kind.GALLOP),
            kept("Side D", v["g_side"], side, "m"),
            kept("f<sub>n</sub>", v["g_fn"], fn, "Hz"),
            at_most("Mass", v["g_mass"], heaviest, "kg/m"),
            at_most("ζ", v["g_zeta"], most),
            at_least("The onset U<sub>g</sub>", g.onset, wind, "m/s"),
        )

    need = wind * 1.225 * side * A1 / (4 * TWO_PI * fn)
    return Question(
        f"A square section {sig(1000 * side, 6)} mm across on a spring at {sig(fn, 6)} Hz gallops from {number(start.onset)} m/s at "
        f"{sig(m0, 6)} kg/m with ζ = {sig(z0, 6)}. It must not gallop below {number(wind, 2)} m/s. Choose its mass, no more than "
        f"{number(heaviest, 2)} kg/m, and its damping, ζ no more than {number(most, 2)}.",
        0,
        f"U<sub>g</sub> = 4mζω<sub>n</sub>/(ρDA<sub>1</sub>), so it needs mζ ≥ ρDA<sub>1</sub>U/(4ω<sub>n</sub>) = {number(need)} kg/m, about {sig(times, 6)} times "
        f"what it has: more than either limit allows alone ({number(heaviest * z0)} or {number(m0 * most)}), so "
        f"both have to rise; {sig(design['g_mass'], 6)} kg/m with ζ = {sig(design['g_zeta'], 6)} gives "
        f"{number(design['g_mass'] * design['g_zeta'])}.",
        setup=dict(kind=Kind.GALLOP, values={"g_side": side, "g_fn": fn, "g_mass": m0, "g_zeta": z0, "g_speed": wind}),
        goal=goal,
        design=dict(kind=Kind.GALLOP, values={"g_side": side, "g_fn": fn, **design, "g_speed": wind}),
    )


PROBLEMS = tuple(
    Problem(key, title, "self_excited_page", f"See {page_link('self_excited_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("vdp-amplitude", "A limit cycle's size", "vdp", "Van der Pol", _vdp_amplitude),
        ("vdp-growth", "Growing from rest", "hopf", "Hopf bifurcation and limit cycles", _vdp_growth),
        ("galloping-onset", "Galloping's onset", "galloping", "Galloping", _galloping_onset),
        ("galloping-damping", "Damping against galloping", "galloping", "Galloping", _galloping_damping),
        ("lock-in", "Where lock-in comes", "viv", "Vortex-induced vibration", _lock_in),
        ("divergence", "Divergence", "flutter", "Flutter", _divergence),
        ("double-eps", "Predict: twice ε", "vdp", "Van der Pol", _double_eps),
        ("heavier", "Predict: a heavier section", "galloping", "Galloping", _heavier),
        ("past-lock-in", "Predict: past lock-in", "viv", "Vortex-induced vibration", _past_lock_in),
        ("coalescence", "Predict: toward flutter", "flutter", "Flutter", _coalescence),
        ("mass-balance", "Predict: mass balance", "flutter", "Flutter", _mass_balance),
        ("scruton-spec", "Spec: mass and damping", "galloping", "Galloping", _scruton_spec),
    )
)

TIPS = {
    "kind": "<p>Which self-excited system.</p>",
    "vdp_freq": "<p>The natural frequency.</p>",
    "vdp_eps": "<p>ε: the damping at small amplitude is −ε/2 of critical (negative for ε &gt; 0).</p>",
    "vdp_ref": "<p>The amplitude scale: the damping changes sign at x = x<sub>r</sub>√ε.</p>",
    "start": "<p>The start: displaced this far, at rest.</p>",
    "periods": "<p>How long to run, in natural periods.</p>",
    "side": "<p>The square's side D.</p>",
    "mass": "<p>Mass per meter of length.</p>",
    "fn": "<p>The natural frequency.</p>",
    "zeta": "<p>The structure's damping ratio.</p>",
    "speed": "<p>The wind speed.</p>",
    "reduced": "<p>The current as a reduced velocity U/f<sub>n</sub>D.</p>",
    "plunge": "<p>The plunge (bending) mode's frequency, without wind.</p>",
    "pitch": "<p>The pitch (torsion) mode's frequency, without wind.</p>",
    "cg": "<p>The center of mass behind the elastic axis, in semichords.</p>",
    "e": "<p>The elastic axis behind the quarter chord (where the lift acts), in semichords.</p>",
    "lift": "<p>Steady, L ∝ θ: the lift follows the pitch alone, a stiffness. Quasi-steady, L ∝ θ + ḣ/U: it also "
    "follows the plunge's ḣ/U, the angle the plunging section sees, which acts as a damping.</p>",
}
DRIVE_TIP = (
    "<p>What drives the model: ε for van der Pol, the wind speed for galloping and flutter, the reduced velocity "
    "for VIV. Drag it and the poles below follow at once; the model is run again when you let go.</p>"
)
POLES_TIP = (
    "<p>The poles of the model at small motion, linearized, at the slider's value (gray: over the slider's "
    "range). A pole right of Re λ = 0 grows: filled in the force color.</p>"
)
SWEEP_TIP = (
    "<p>The settled motion against the parameter that drives it; the dotted line is the value set. For flutter, "
    "the root locus as the speed rises: drag a pole along it to set the speed, and watch one cross Re λ = 0.</p>"
)

NOTES_HTML = load_notes("self_excited", figures=self_excited_figures.FIGURES, trials=trials_html(TRIALS))


class SelfExcitedPage(QtWidgets.QWidget):
    """Controls on the left, the motion in the center, the sweep on the right."""

    follows_chain = False  # MainWindow: this page has its own models, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._viv_cache: tuple | None = None
        self._vdp_cache: tuple | None = None  # (the oscillator and its ε grid, the amplitudes): a tab change redraws
        self._stale = False
        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Each picks a model and sets it up to show one idea from the Theory tab.</p>",
            predictions=[t.predict for t in TRIALS],
        )
        self.trials.picked.connect(self.load_trial)

        top = Form("Model")
        self.kind = combo([k.value for k in KINDS], TIPS["kind"])
        top.add("System:", self.kind, TIPS["kind"])

        self.boxes: dict[str, QtWidgets.QDoubleSpinBox] = {}
        self.pages = QtWidgets.QStackedWidget()
        vdp = Form("Van der Pol")
        self._add(vdp, "vdp_freq", "Frequency:", 0.05, 100.0, 1.0, 3, " Hz", "vdp_freq")
        self._add(vdp, "vdp_eps", "ε:", -2.0, 20.0, 0.2, 3, "", "vdp_eps")
        self._add(vdp, "vdp_ref", "x<sub>r</sub>:", 0.1, 1000.0, 10.0, 2, " mm", "vdp_ref")
        self._add(vdp, "vdp_start", "Start:", 0.0, 1000.0, 0.5, 2, " mm", "start")
        self._add(vdp, "vdp_periods", "Periods:", 2.0, 400.0, 60.0, 0, "", "periods")
        gal = Form("Galloping")
        self._add(gal, "g_side", "Side D:", 0.01, 2.0, 0.1, 3, " m", "side")
        self._add(gal, "g_mass", "Mass:", 0.1, 1000.0, 5.0, 2, " kg/m", "mass")
        self._add(gal, "g_fn", "fₙ:", 0.05, 50.0, 2.0, 3, " Hz", "fn")
        self._add(gal, "g_zeta", "ζ:", 0.0, 0.2, 0.005, 4, "", "zeta")
        self._add(gal, "g_speed", "Wind U:", 0.0, 60.0, 6.0, 2, " m/s", "speed")
        self._add(gal, "g_start", "Start:", 0.0, 2000.0, 1.0, 2, " mm", "start")
        self._add(gal, "g_periods", "Periods:", 10.0, 1000.0, 300.0, 0, "", "periods")
        viv = Form("Vortex-induced vibration (in water)")
        self._add(viv, "v_d", "Diameter D:", 0.005, 2.0, 0.05, 3, " m", "side")
        self._add(viv, "v_mass", "Mass:", 0.1, 5000.0, 20.0, 2, " kg/m", "mass")
        self._add(viv, "v_fn", "fₙ (in water):", 0.05, 50.0, 1.0, 3, " Hz", "fn")
        self._add(viv, "v_zeta", "ζ:", 0.0, 0.2, 0.005, 4, "", "zeta")
        self._add(viv, "v_reduced", "U/fₙD:", 1.0, 20.0, 5.5, 2, "", "reduced")
        fl = Form("Flutter")
        self._add(fl, "f_plunge", "Plunge:", 0.1, 50.0, 3.0, 2, " Hz", "plunge")
        self._add(fl, "f_pitch", "Pitch:", 0.1, 50.0, 6.0, 2, " Hz", "pitch")
        self._add(fl, "f_cg", "x<sub>cg</sub>:", -0.5, 1.0, 0.2, 2, " b", "cg")
        self._add(fl, "f_e", "e:", -0.5, 1.0, 0.3, 2, " b", "e")
        self._add(fl, "f_zeta", "ζ:", 0.0, 0.2, 0.01, 3, "", "zeta")
        self._add(fl, "f_speed", "Speed U:", 0.0, 200.0, 15.0, 1, " m/s", "speed")
        self.lift = combo(LIFTS, TIPS["lift"])
        fl.add("Lift:", self.lift, TIPS["lift"])
        for form in (vdp, gal, viv, fl):
            self.pages.addWidget(form)
        self.info = note_label()
        connect_all((self.kind, self.lift, *self.boxes.values()), self._on_inputs)

        # --- center
        self.top_plot = plot("x", "m", "Time", "s", height=150)
        self.bottom_plot = plot("Velocity", "m/s", "x", "m", height=150)
        add_legend(self.top_plot.getPlotItem(), offset=(-10, 10))
        add_legend(self.bottom_plot.getPlotItem(), offset=(-10, 10))
        self.top_items = [pg.PlotDataItem(), pg.PlotDataItem()]
        # Up to 30,000 steps: decimated, and without antialiasing where they're dense (`_dense`): antialiased, a
        # swing a pixel or two wide took Qt 0.4–0.85 s a paint.
        decimate(self.top_plot)
        self.bottom_items = [pg.PlotDataItem(), pg.PlotDataItem()]
        self.top_line = pg.InfiniteLine(angle=90)
        self.bottom_line = pg.InfiniteLine(angle=90)
        self.bottom_zero = pg.InfiniteLine(angle=0, pos=0)
        for item in (self.top_line, *self.top_items):
            self.top_plot.addItem(item)
        for item in (self.bottom_zero, self.bottom_line, *self.bottom_items):
            self.bottom_plot.addItem(item)
        self.readout = readout_label()

        # --- right
        self.sweep_plot = plot("Amplitude", "m", "Parameter", None, height=200)
        self.sweep_plot.setToolTip(SWEEP_TIP)
        add_legend(self.sweep_plot.getPlotItem(), offset=(-10, 10))
        self.sweep2_plot = plot("Frequency / fₙ", None, "Parameter", None, height=160)
        self.sweep2_plot.setXLink(self.sweep_plot)
        add_legend(self.sweep2_plot.getPlotItem(), offset=(-10, 10))
        self.sweep_curve = pg.PlotDataItem()
        self.sweep_theory = pg.PlotDataItem()
        self.sweep_stable = marker(7)
        self.sweep_unstable = marker(7)
        self.sweep_line = pg.InfiniteLine(angle=90)
        self.flutter_dots = [
            DragDot("<p>Drag the pole along its locus to set the speed.</p>", size=12) for _ in range(2)
        ]
        for item in (
            self.sweep_line,
            self.sweep_theory,
            self.sweep_curve,
            self.sweep_stable,
            self.sweep_unstable,
            *self.flutter_dots,
        ):
            self.sweep_plot.addItem(item)
        for dot in self.flutter_dots:
            dot.setVisible(False)
            dot.dragged.connect(self._on_flutter_dragged)
        self.sweep2_curve = pg.PlotDataItem()
        self.sweep2_ref = pg.PlotDataItem()
        self.sweep2_line = pg.InfiniteLine(angle=90)
        for item in (self.sweep2_line, self.sweep2_ref, self.sweep2_curve):
            self.sweep2_plot.addItem(item)
        self.sweep_note = note_label()

        # --- right: the body moving, and the drive's slider moving the poles
        self.playback = Playback(self, "Self-excited vibration")
        self.playback.moved.connect(self._place)
        self.time_cursor = self.playback.add_cursor(self.top_plot)
        self.section = SectionView()
        self.section.setMinimumHeight(190)
        self.drive = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.drive.setRange(0, DRIVE_STEPS)
        self.drive.setToolTip(DRIVE_TIP)
        self.drive.valueChanged.connect(self._on_drive)
        self.drive.sliderReleased.connect(self._commit_drive)
        self.drive_label = QtWidgets.QLabel()
        self.drive_label.setBuddy(self.drive)
        self.drive.setAccessibleName("Drive")
        self.drive_value = QtWidgets.QLabel()
        self.drive_range = (0.0, 1.0)
        drive = QtWidgets.QWidget()
        dl = QtWidgets.QHBoxLayout(drive)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.addWidget(self.drive_label)
        dl.addWidget(self.drive, 1)
        dl.addWidget(self.drive_value)
        self.pole_plot = plot("Im λ / 2π", "Hz", "Re λ", "1/s", height=150)
        self.pole_plot.setToolTip(POLES_TIP)
        self.pole_plot.getAxis("bottom").enableAutoSIPrefix(False)  # not "m1/s"
        self.pole_axis = pg.InfiniteLine(angle=90, pos=0)
        self.pole_locus = pg.PlotDataItem(pen=None, symbol="o", symbolSize=3)
        self.pole_now = marker(11)
        for item in (self.pole_axis, self.pole_locus, self.pole_now):
            self.pole_plot.addItem(item)
        self.motion_note = note_label()
        self.motion: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None  # t, y, θ

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(
            column(self.playback.bar(), self.section, drive, self.pole_plot, self.motion_note, stretch=False),
            "Motion",
        )
        self.tabs.addTab(column(self.sweep_plot, self.sweep2_plot, self.sweep_note, stretch=False), "Sweep")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._draw_sweep)
        self.tabs.currentChanged.connect(self._draw_motion)

        three_panes(
            self,
            column(self.trials, top, self.pages, self.info),
            stack((self.top_plot, 1), (self.bottom_plot, 1), (self.readout, 0)),
            self.tabs,
        )
        self.apply_theme()
        self._on_inputs()

    def _add(
        self, form: Form, key: str, label: str, lo: float, hi: float, value: float, decimals: int, suffix: str, tip: str
    ) -> None:
        self.boxes[key] = form.number(label, lo, hi, value, decimals, suffix, TIPS[tip])

    def v(self, key: str) -> float:
        """A control's value in SI (the boxes in mm hold `MM` of them)."""
        return float(self.boxes[key].value()) / (1000.0 if key in MM else 1.0)

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note); hidden, it draws
        once shown."""
        self.apply_trial(SelfTrial("", "", **setup))

    def practice_state(self) -> SelfTrial:
        """The page's settings, as a trial: what a practice spec checks."""
        return SelfTrial(
            "",
            "",
            KINDS[self.kind.currentIndex()],
            {k: self.v(k) for k in self.boxes},
            self.tabs.currentIndex(),
            quasi_steady=self.lift.currentIndex() == 1,
        )

    def apply_trial(self, t: SelfTrial) -> None:
        for key, value in {**DEFAULTS, **t.values}.items():
            set_quietly(self.boxes[key], value * (1000.0 if key in MM else 1.0))
        set_quietly(self.kind, KINDS.index(t.kind))
        set_quietly(self.lift, int(t.quasi_steady))
        self.trials.show_note(t.note)
        # a practice setup on a hidden page: the stepping waits until it shows
        self._stale = not (t.name or self.isVisible())
        self.tabs.setCurrentIndex(t.tab)
        if self._stale:
            self.current = t.kind
            self.pages.setCurrentIndex(self.kind.currentIndex())
        else:
            self._on_inputs()
        self.playback.play()  # a trial starts the motion

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._stale:
            self._stale = False
            self._on_inputs()

    def _on_inputs(self) -> None:
        self._stale = False
        self.current = KINDS[self.kind.currentIndex()]
        self.pages.setCurrentIndex(self.kind.currentIndex())
        self.redraw()

    def _on_flutter_dragged(self, speed: float) -> None:
        self.boxes["f_speed"].setValue(speed)  # redraws when it changes ...
        self._place_flutter_dots(self.flutter())  # ... but rounded to the box it may not: the dots at its value

    def _place_flutter_dots(self, flutter: Flutter) -> None:
        """The poles at the speed set, on the locus."""
        now = flutter.poles(self.v("f_speed"))
        now = now[now.imag > 0]
        for k, grip in enumerate(self.flutter_dots):
            grip.setVisible(k < now.size)
            if k < now.size:
                grip.place(float(now[k].real), float(now[k].imag) / (2 * np.pi), self.v("f_speed"))

    # ------------------------------------------------------------- the drive
    def _drive(self) -> tuple[str, str, float, float]:
        """The current model's drive: its box, its label, and the slider's range."""
        if self.current is Kind.VDP:
            return "vdp_eps", "ε:", -0.5, 3.0
        if self.current is Kind.GALLOP:
            onset = self.galloping().onset
            return "g_speed", "Wind U:", 0.0, max(3.0 * onset, 1.2 * self.v("g_speed"))
        if self.current is Kind.VIV:
            return "v_reduced", "U/f<sub>n</sub>D:", 2.0, 14.0
        m = self.flutter()
        Uf, Ud = m.flutter_speed(), m.divergence_speed()
        return "f_speed", "Speed U:", 0.0, 1.3 * (Uf if np.isfinite(Uf) else min(Ud, 100.0))

    def _drive_value(self) -> float:
        lo, hi = self.drive_range
        return lo + (hi - lo) * self.drive.value() / DRIVE_STEPS

    def _set_drive(self) -> None:
        """The slider at the drive box's value, over the model's range, and the poles there."""
        key, label, lo, hi = self._drive()
        self.drive_label.setText(label)
        self.drive_range = (lo, hi)
        value = self.v(key)
        self.drive.blockSignals(True)
        self.drive.setValue(round(DRIVE_STEPS * min(max((value - lo) / (hi - lo), 0.0), 1.0)))
        self.drive.blockSignals(False)
        self.drive_value.setText(f"{sig(value)}")
        values = np.linspace(lo, hi, 61)
        locus = [p for x in values for p in self._poles(float(x))]
        re = np.array([p.real for p in locus] or [0.0])
        hz = np.array([p.imag / TWO_PI for p in locus] or [1.0])
        self.pole_locus.setData(re, hz)
        self.pole_plot.setVisible(self.current is not Kind.VIV)
        self._show_poles(value)
        lo_x, hi_x = min(float(re.min()), 0.0), max(float(re.max()), 0.0)
        pad = 0.08 * max(hi_x - lo_x, 1e-6)
        self.pole_plot.setXRange(lo_x - pad, hi_x + pad, padding=0)
        self.pole_plot.setYRange(0.0, 1.15 * max(float(hz.max()), 1e-6), padding=0)

    def _poles(self, value: float) -> np.ndarray:
        """The linearized poles (Im ≥ 0) at the drive's `value`: none for VIV, whose wake is nonlinear."""
        if self.current is Kind.FLUTTER:
            lam = self.flutter().poles(value)
            return lam[lam.imag >= 0]
        if self.current is Kind.VIV:
            return np.zeros(0, complex)
        if self.current is Kind.VDP:
            w = self.vdp().omega
            a = -value * w  # ẍ − εω ẋ + ω² x at small x
        else:
            g = self.galloping()
            w = g.omega
            a = 2 * g.zeta * w - g.rho * value * g.side * PARKINSON[0] / (2 * g.mass)
        lam = np.roots([1.0, a, w * w])
        return lam[lam.imag >= 0]

    def _show_poles(self, value: float) -> None:
        lam = self._poles(value)
        self.pole_now.setData(lam.real, lam.imag / TWO_PI)
        grows = bool(np.any(lam.real > 1e-9 * np.maximum(np.abs(lam), 1.0)))
        mark(self.pole_now, colors.force if grows else colors.mass[0], filled=True)

    def _on_drive(self, step: int) -> None:
        """The slider moved: the poles follow at once; the model is run again when it's let go (or by a key)."""
        value = self._drive_value()
        self.drive_value.setText(f"{sig(value)}")
        self._show_poles(value)
        if not self.drive.isSliderDown():
            self._commit_drive()

    def _commit_drive(self) -> None:
        key = self._drive()[0]
        self.boxes[key].setValue(self._drive_value())  # redraws when it changes
        self._set_drive()

    def vdp(self) -> VanDerPol:
        return VanDerPol(self.v("vdp_freq"), self.v("vdp_eps"), self.v("vdp_ref"), self.v("vdp_start"))

    def galloping(self) -> Galloping:
        return Galloping(
            self.v("g_side"),
            self.v("g_mass"),
            self.v("g_fn"),
            self.v("g_zeta"),
            self.v("g_speed"),
            start=self.v("g_start"),
        )

    def viv(self) -> Viv:
        return Viv(self.v("v_d"), self.v("v_mass"), self.v("v_fn"), self.v("v_zeta"), self.v("v_reduced"))

    def flutter(self) -> Flutter:
        return Flutter(
            x_cg=self.v("f_cg"),
            e=self.v("f_e"),
            plunge_hz=self.v("f_plunge"),
            pitch_hz=self.v("f_pitch"),
            zeta=self.v("f_zeta"),
            quasi_steady=self.lift.currentIndex() == 1,
        )

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        for item, color in zip(self.top_items, (c.mass[0], c.force)):
            item.setPen(pg.mkPen(color, width=1.5))
        for item, color in zip(self.bottom_items, (c.mass[0], c.mass[1])):
            item.setPen(pg.mkPen(color, width=1.5))
        for grip in self.flutter_dots:
            grip.apply_theme()
        self.section.apply_theme()
        self.playback.apply_theme()
        self.pole_axis.setPen(pg.mkPen(c.faint, width=1.5))
        self.pole_locus.setSymbolBrush(pg.mkBrush(c.grey))
        self.pole_locus.setSymbolPen(None)
        dot = QtCore.Qt.PenStyle.DotLine
        for line in (self.top_line, self.bottom_line, self.sweep_line, self.sweep2_line):
            line.setPen(pg.mkPen(c.grey, width=1.5, style=dot))
        self.bottom_zero.setPen(pg.mkPen(c.faint, width=1))
        self.sweep_curve.setPen(pg.mkPen(c.mass[0], width=2))
        self.sweep_curve.setSymbolBrush(pg.mkBrush(c.mass[0]))
        self.sweep_curve.setSymbolPen(None)
        self.sweep_theory.setPen(pg.mkPen(c.strong, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        mark(self.sweep_stable, c.mass[0], filled=True)
        mark(self.sweep_unstable, c.force)
        self.sweep2_curve.setPen(pg.mkPen(c.mass[0], width=2))
        self.sweep2_ref.setPen(pg.mkPen(c.strong, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        if hasattr(self, "current"):
            self.redraw()

    @coalesce.coalesced(covers=("_draw_sweep", "_draw_motion"))
    def redraw(self) -> None:
        for items in (self.top_items, self.bottom_items):
            for item in items:
                item.setData([], [])
        for p in (self.top_plot, self.bottom_plot):
            p.getPlotItem().legend.clear()
        self.top_line.setVisible(False)
        self.bottom_line.setVisible(False)
        self.bottom_zero.setVisible(False)
        {
            Kind.VDP: self._draw_vdp,
            Kind.GALLOP: self._draw_galloping,
            Kind.VIV: self._draw_viv,
            Kind.FLUTTER: self._draw_flutter,
        }[self.current]()
        self.time_cursor.setVisible(self.current is not Kind.FLUTTER)  # flutter's plots are against the speed
        self._set_drive()
        self._draw_motion()
        self._draw_sweep()
        settle_plots(self)

    def _time_and_phase(
        self, t: np.ndarray, x: np.ndarray, v: np.ndarray, scale: float, unit: str | None, name: str
    ) -> None:
        relabel(self.top_plot, "left", name, unit)
        relabel(self.top_plot, "bottom", "Time", "s")
        relabel(self.bottom_plot, "left", "Velocity", "m/s")
        relabel(self.bottom_plot, "bottom", name, unit)
        self.top_items[0].setData(t, x / scale, antialias=not self._dense(t))
        last = t >= t[-1] - 10.0 * (t[-1] / max(self._periods(), 1.0))  # the phase plane: the last ten periods
        self.bottom_items[0].setData(x[last] / scale, v[last])
        self.top_plot.setXRange(0.0, float(t[-1]), padding=0)
        self.top_plot.enableAutoRange(axis="y")
        self.bottom_plot.enableAutoRange()

    def _dense(self, t: np.ndarray) -> bool:
        """More than a few samples a pixel across the time plot: drawn without antialiasing (see `top_plot`)."""
        return t.size > DENSE_PER_PX * max(self.top_plot.width(), 1)

    def _periods(self) -> float:
        return {Kind.VDP: self.v("vdp_periods"), Kind.GALLOP: self.v("g_periods"), Kind.VIV: 150.0}.get(
            self.current, 1.0
        )

    def _draw_vdp(self) -> None:
        m = self.vdp()
        t, x, v = m.simulate(self.v("vdp_periods"))
        self._time_and_phase(t, x, v, 1.0, "m", "x")
        self._trace = (t, x)
        amp = float(settled_amplitude(x))
        self.info.setText(
            f"Damping at rest ζ = {sig(-m.eps / 2)}; averaging's limit cycle: "
            f"{sig(float(m.theory_amplitude(m.eps)) * 1e3)} mm."
        )
        period = 1.0 / float(dominant_hz(x, t[1] - t[0])) if amp > 1e-9 else float("nan")
        self.readout.setText(
            f"Settled amplitude {sig(amp * 1e3)} mm (2x<sub>r</sub>√ε = "
            f"{sig(float(m.theory_amplitude(m.eps)) * 1e3)} mm); period {sig(period)} s against "
            f"{sig(1 / m.freq_hz)} s at small ε."
            if m.eps > 0
            else f"ε ≤ 0: the start dies away (amplitude at the end {sig(abs(x[-1]) * 1e3)} mm)."
        )

    def _draw_galloping(self) -> None:
        m = self.galloping()
        t, y, v = m.simulate(self.v("g_periods"))
        self._time_and_phase(t, y, v, m.side, None, "y / D")
        self._trace = (t, y / m.side)
        amp = float(settled_amplitude(y))
        self.info.setText(
            f"Onset U<sub>g</sub> = 4mζω<sub>n</sub>/(ρDA<sub>1</sub>) = {sig(m.onset)} m/s; U/U<sub>g</sub> = {sig(m.speed / m.onset)}."
        )
        self.readout.setText(
            f"At {sig(m.speed)} m/s from {sig(m.start * 1e3)} mm: the last quarter swings "
            f"{sig(amp / m.side)} D ({sig(amp * 1e3)} mm)."
        )

    def _draw_viv(self) -> None:
        m = self.viv()
        t, z = m.run(np.array([m.reduced]), periods=150.0)
        y, v, q = z[:, 0, 0], z[:, 1, 0], z[:, 2, 0]
        self._time_and_phase(t, y, v, m.diameter, None, "y / D")
        self._trace = (t, y / m.diameter)
        legend = self.top_plot.getPlotItem().legend
        scale = max(float(np.max(np.abs(y))) / m.diameter, 1e-6) / max(float(np.max(np.abs(q))), 1e-9)
        self.top_items[1].setData(t, q * scale, antialias=not self._dense(t))
        legend.addItem(self.top_items[0], "y / D")
        legend.addItem(self.top_items[1], "wake q (scaled)")
        amp = float(settled_amplitude(y)) / m.diameter
        f = float(dominant_hz(y, t[1] - t[0])) / m.freq_hz
        self.info.setText(
            f"Mass ratio m* = {sig(m.mass_ratio)}; shedding alone at St U/D = "
            f"{sig(m.strouhal * m.reduced)} f<sub>n</sub>; lock-in expected near U<sub>r</sub> = "
            f"{sig(m.lock_in_at)}."
        )
        self.readout.setText(
            f"U<sub>r</sub> = {sig(m.reduced)}: settled amplitude {sig(amp)} D at {sig(f)} "
            f"f<sub>n</sub> (the shedding alone would be at {sig(m.strouhal * m.reduced)} "
            "f<sub>n</sub>)."
        )

    def _draw_flutter(self) -> None:
        m = self.flutter()
        Uf, Ud = m.flutter_speed(), m.divergence_speed()
        top = 1.3 * (Uf if np.isfinite(Uf) else min(Ud, 100.0))
        U = np.linspace(0.0, top, 160)
        f, z = m.modes(U)
        relabel(self.top_plot, "left", "Frequency", "Hz")
        relabel(self.top_plot, "bottom", "Speed U", "m/s")
        relabel(self.bottom_plot, "left", "Damping ratio ζ", None)
        relabel(self.bottom_plot, "bottom", "Speed U", "m/s")
        for j, name in enumerate(("mode 1 (plunge at rest)", "mode 2 (pitch at rest)")):
            self.top_items[j].setData(U, f[:, j], connect="finite")
            self.bottom_items[j].setData(U, z[:, j], connect="finite")
            self.top_plot.getPlotItem().legend.addItem(self.top_items[j], name)
            self.bottom_plot.getPlotItem().legend.addItem(self.bottom_items[j], name)
        self.top_items[1].setPen(pg.mkPen(colors.mass[1], width=1.5))
        self.bottom_zero.setVisible(True)
        for line in (self.top_line, self.bottom_line):
            line.setVisible(True)
            line.setValue(self.v("f_speed"))
        self.top_plot.setXRange(0.0, top, padding=0)
        self.bottom_plot.setXRange(0.0, top, padding=0)
        self.top_plot.enableAutoRange(axis="y")
        finite = z[np.isfinite(z)]
        self.bottom_plot.setYRange(max(-0.1, float(finite.min()) - 0.02), float(finite.max()) + 0.02, padding=0)
        lam = m.poles(self.v("f_speed"))
        up = lam[lam.imag > 0]
        self.info.setText(
            f"Flutter at {sig(Uf)} m/s; the pitch spring alone diverges at {sig(Ud)} m/s."
            if np.isfinite(Uf)
            else f"No flutter: the section only diverges, at {sig(Ud)} m/s, where the pitch spring gives out."
        )
        self.readout.setText(
            f"At {sig(self.v('f_speed'))} m/s: "
            + ", ".join(f"{sig(abs(p) / (2 * np.pi))} Hz with ζ = {sig(-p.real / abs(p))}" for p in sorted(up, key=abs))
            + "."
        )

    @coalesce.coalesced
    def _draw_motion(self) -> None:
        """The body's motion for the Motion tab: the run the center shows, or flutter's least stable mode."""
        if self.tabs.currentIndex() != TAB_MOTION or not hasattr(self, "current") or self._stale:
            return
        axis = 0.0
        if self.current is Kind.FLUTTER:
            m = self.flutter()
            speed = self.v("f_speed")
            lam, h, theta = m.mode(speed)
            body, axis = Body.WING, -0.5 + m.e  # the elastic axis, e semichords behind the quarter chord
            if lam.imag <= 0:
                t = np.array([0.0, 1.0])
                y = th = np.zeros(2)
                self.motion_note.setText(f"At {sig(speed)} m/s neither mode oscillates.")
            else:
                w = lam.imag
                t = np.linspace(0.0, FLUTTER_CYCLES * TWO_PI / w, FLUTTER_CYCLES * 48 + 1)
                amp = np.minimum(FLUTTER_START * np.exp(lam.real * t), FLUTTER_CAP)
                e = np.exp(1j * w * t)
                y = -(h / m.semichord * e).real * amp  # h is down; drawn up
                th = (theta * e).real * amp
                lead = math.degrees(math.atan2((-h).imag, (-h).real))
                per = math.exp(TWO_PI * lam.real / w)
                self.motion_note.setText(
                    f"The least stable mode at {sig(speed)} m/s, {sig(w / TWO_PI)} Hz: the plunge (up) "
                    f"{'leads' if lead >= 0 else 'lags'} the pitch (nose up) by {sig(abs(lead))}°, and the motion "
                    + (
                        f"grows {sig(per)}× a cycle (drawn until it reaches {sig(FLUTTER_CAP, 6)} b)."
                        if per > 1
                        else f"decays to {sig(per)} of itself each cycle."
                    )
                )
            grows = bool(lam.real > 0)
        else:
            t, y = self._trace
            th = np.zeros_like(y)
            body = {Kind.VDP: Body.BLOCK, Kind.GALLOP: Body.SQUARE, Kind.VIV: Body.CIRCLE}[self.current]
            if self.current is Kind.VDP:
                grows = self.v("vdp_eps") > 0
                self.motion_note.setText("The mass on its spring, as the time history shows it.")
            elif self.current is Kind.GALLOP:
                g = self.galloping()
                grows = g.speed > g.onset
                self.motion_note.setText(
                    f"The square to scale, in a {sig(g.speed)} m/s wind from the left (the onset is {sig(g.onset)} m/s)."
                )
            else:
                grows = float(settled_amplitude(y)) > 0.05
                self.motion_note.setText(
                    "The cylinder to scale, in the flow from the left. Its wake is a nonlinear oscillator of its "
                    "own, so there are no poles to show."
                )
        self.section.unstable = grows
        self.section.set_body(body, axis)
        self.section.set_range(max(float(np.max(np.abs(y))), 0.3 if body is not Body.BLOCK else 1e-9))
        self.motion = (t, y, th)
        self.playback.set_span(float(t[-1]))  # places the body

    def _place(self, t: float) -> None:
        if self.motion is None:
            return
        ts, y, th = self.motion
        k = min(int(np.searchsorted(ts, t)), ts.size - 1)
        self.section.set_state(float(y[k]), float(th[k]))

    @coalesce.coalesced
    def _draw_sweep(self) -> None:
        if self.tabs.currentIndex() != TAB_SWEEP or not hasattr(self, "current") or self._stale:
            return
        for item in (
            self.sweep_curve,
            self.sweep_theory,
            self.sweep_stable,
            self.sweep_unstable,
            self.sweep2_curve,
            self.sweep2_ref,
        ):
            item.setData([], [])
        for grip in self.flutter_dots:  # shown on flutter's locus only
            grip.setVisible(False)
        for p in (self.sweep_plot, self.sweep2_plot):
            p.getPlotItem().legend.clear()
        legend = self.sweep_plot.getPlotItem().legend
        self.sweep2_plot.setVisible(self.current in (Kind.VIV, Kind.FLUTTER))
        if self.current is Kind.VDP:
            m = self.vdp()
            eps = np.linspace(-0.5, max(3.0, m.eps * 1.2), 36)
            grid = (replace(m, eps=0.0), float(eps[-1]))  # the sweep is over ε: the oscillator's own doesn't matter
            if self._vdp_cache is None or self._vdp_cache[0] != grid:
                self._vdp_cache = (grid, m.sweep(eps))
            amp = self._vdp_cache[1]
            self.sweep_curve.setData(eps, amp * 1e3, symbol="o", symbolSize=5)
            fine = np.linspace(eps[0], eps[-1], 300)
            self.sweep_theory.setData(fine, m.theory_amplitude(fine) * 1e3)
            legend.addItem(self.sweep_curve, "settled (stepped)")
            legend.addItem(self.sweep_theory, "2x<sub>r</sub>√ε (averaging)")
            relabel(self.sweep_plot, "left", "Amplitude (mm)")
            relabel(self.sweep_plot, "bottom", "ε")
            self.sweep_line.setValue(m.eps)
            self.sweep_note.setText("Zero below ε = 0 (the Hopf point), then growing as √ε: supercritical.")
        elif self.current is Kind.GALLOP:
            gallop = self.galloping()
            U = np.linspace(0.0, max(15.0, 1.3 * gallop.speed), 241)
            stable_x: list[float] = []
            stable_y: list[float] = []
            unstable_x: list[float] = []
            unstable_y: list[float] = []
            for b in galloping_branches(gallop, U):
                ok = np.isfinite(b.amplitude)
                (stable_x if b.stable else unstable_x).extend(b.speed[ok])
                (stable_y if b.stable else unstable_y).extend(b.amplitude[ok] / gallop.side)
            self.sweep_stable.setData(stable_x, stable_y)
            self.sweep_unstable.setData(unstable_x, unstable_y)
            legend.addItem(self.sweep_stable, "stable")
            legend.addItem(self.sweep_unstable, "unstable")
            relabel(self.sweep_plot, "left", "Amplitude / D")
            relabel(self.sweep_plot, "bottom", "Wind speed U (m/s)")
            self.sweep_line.setValue(gallop.speed)
            self.sweep_note.setText(
                f"Averaging (Parkinson's polynomial). Rest is stable below U<sub>g</sub> = "
                f"{sig(gallop.onset)} m/s; above it the amplitude grows with U, on two stable branches "
                "where they overlap."
            )
        elif self.current is Kind.VIV:
            viv = self.viv()
            key = (viv.diameter, viv.mass, viv.freq_hz, viv.zeta)
            if self._viv_cache is None or self._viv_cache[0] != key:
                self._viv_cache = (key, *viv.sweep(VIV_SWEEP))
            _, amp, freq = self._viv_cache
            self.sweep_curve.setData(VIV_SWEEP, amp, symbol="o", symbolSize=5)
            legend.addItem(self.sweep_curve, "settled amplitude")
            self.sweep2_curve.setData(VIV_SWEEP, freq, symbol="o", symbolSize=5)
            self.sweep2_ref.setData(VIV_SWEEP, viv.strouhal * VIV_SWEEP)
            l2 = self.sweep2_plot.getPlotItem().legend
            l2.addItem(self.sweep2_curve, "response")
            l2.addItem(self.sweep2_ref, "shedding alone, St U/D")
            relabel(self.sweep_plot, "left", "Amplitude / D")
            relabel(self.sweep_plot, "bottom", "Reduced velocity U/f<sub>n</sub>D")
            relabel(self.sweep2_plot, "left", "Frequency / f<sub>n</sub>")
            relabel(self.sweep2_plot, "bottom", "Reduced velocity U/f<sub>n</sub>D")
            self.sweep2_plot.setYRange(0.0, 3.0, padding=0)
            self.sweep_line.setValue(viv.reduced)
            self.sweep2_line.setValue(viv.reduced)
            self.sweep_note.setText(
                "Lock-in: where the response frequency stays near f<sub>n</sub> instead of "
                "following the Strouhal line, the amplitude is large."
            )
        else:
            flutter = self.flutter()
            Uf = flutter.flutter_speed()
            top = 1.3 * (Uf if np.isfinite(Uf) else min(flutter.divergence_speed(), 100.0))
            U = np.linspace(0.0, top, 160)
            poles = np.array([flutter.poles(float(u)) for u in U])
            up = np.where(poles.imag > 0, poles, np.nan)
            self.sweep_stable.setData(up.real.ravel(), (up.imag / (2 * np.pi)).ravel())
            on = np.isfinite(up)  # the dots' path: every pole on the locus, its parameter the speed
            speeds = np.broadcast_to(U[:, None], up.shape)[on]
            for grip in self.flutter_dots:
                grip.set_path(up[on].real, up[on].imag / (2 * np.pi), speeds)
            self._place_flutter_dots(flutter)
            legend.addItem(
                self.sweep_stable, f"poles, 0 to 1.3 × the {'flutter' if np.isfinite(Uf) else 'divergence'} speed"
            )
            relabel(self.sweep_plot, "left", "Im λ / 2π (Hz)")
            relabel(self.sweep_plot, "bottom", "Re λ (1/s)")
            self.sweep_line.setValue(0.0)
            self.sweep2_plot.setVisible(False)
            self.sweep_note.setText(
                "The root locus: with quasi-steady lift the plunge's poles move left (the flow damps it) while the "
                "pitch's cross Re λ = 0 (dotted) on their own, at a frequency of their own."
                if flutter.quasi_steady
                else "The root locus: as the speed rises the two modes' poles approach, and where they nearly meet "
                "one crosses Re λ = 0 (dotted), at the flutter speed."
            )
        self.sweep_plot.enableAutoRange()
        self.sweep2_plot.enableAutoRange(axis="x")
