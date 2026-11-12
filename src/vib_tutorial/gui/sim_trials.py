"""Things to try on the Simulation page: each sets up the chain and the controls, and says what to look for.

`MainWindow.apply_sim_trial` applies a trial's settings; the Theory tab lists the same notes.
`tests/test_tutorial_numbers.py` checks their numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core import ChainSystem, ForceKind
from ..core.modal import frf, modal_analysis
from ..core.practice import Problem, Question, Requirement, at_most, kept, number, pick, rounded
from .links import page_link

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class SimTrial:
    name: str
    note: str
    preset: str = ""  # a preset's name (or the start of it); else the uniform 4-mass chain, changed by:
    n: int = 4
    damping: dict[int, float] = field(default_factory=dict)  # element index -> c
    friction: float = 0.0  # N on every mass
    state_space: bool = False
    coords: str = "Physical"  # the Plot coordinates entry it starts with
    by_mode: bool = False  # the energy panel by mode
    speed: float = 1.0
    window: float = 10.0  # s of plot
    release: int | None = None  # the modal table's row to release
    amplitude: float = 20.0  # mm, the release's initial displacement
    animate: bool = False
    tune: int = 0  # mode to tune the harmonic force to (1-based), 0 to leave it
    kind: ForceKind | None = None
    sweep: tuple[float, float, float] | None = None  # chirp start, end (Hz) and time (s)
    base: tuple[float, float] | None = None  # ground motion: amplitude (mm) and frequency (Hz)
    apply: bool = False  # switch the force on
    frf_tab: bool = False  # show the Frequency response tab
    masses: tuple[float, ...] = ()  # each element's value, in place of the uniform chain's (n of them)
    springs: tuple[float, ...] = ()
    dampers: tuple[float, ...] = ()
    target: int | None = None  # the mass the force is on (0-based), else the last
    freq_hz: float | None = None  # the harmonic force's frequency
    force: float | None = None  # its amplitude, N


def chain(t: SimTrial) -> ChainSystem:
    """The chain a trial (not a preset) sets up: the uniform one of `n` masses, or its own elements, then its
    dampers changed and its friction."""
    n = len(t.masses) if t.masses else t.n
    system = ChainSystem.uniform(n)
    for values, given in ((system.masses, t.masses), (system.stiffness, t.springs), (system.damping, t.dampers)):
        if given:
            values[:] = np.asarray(given, dtype=float)
    for i, c in t.damping.items():
        system.damping[i] = c
    system.friction[:] = t.friction
    return system


TRIALS = (
    SimTrial(
        "Resonance building up",
        "A harmonic force on <m>m<sub>4</sub></m>, tuned to mode 1 (1.11 Hz). The swing grows over a few of the "
        "mode's time constants, <m>1/ζω</m> = 8.3 s, to a steady amplitude where the dampers take out what the force puts in. "
        "The Frequency response tab marks the drive (dashed) on the peak.",
        tune=1,
        apply=True,
        window=30.0,
    ),
    SimTrial(
        "Proportional damping: one mode stays one mode",
        "Every damper is 0.005 times its spring (<m>C = βK</m>), the default chain. Mode 3 is released at 0.25× "
        "speed: it decays in one clean shape, only its curve moves in modal coordinates, and it keeps 100% "
        "of the energy (Energy: by mode) while the total decays.",
        coords="Modal",
        by_mode=True,
        speed=0.25,
        window=2.05,
        release=2,
    ),
    SimTrial(
        "Non-proportional damping: modes trade energy",
        "<m>c<sub>1</sub></m> = 15 N·s/m, the other dampers at 2. The note under the modal table says "
        "non-proportional, and <m>ζ</m> modal and <m>ζ</m> exact separate. Mode 3's real shape is no longer an exact mode of the damped chain: "
        "released, it feeds modes 1 and 2, and within a second most of the energy left is in mode 1, which "
        "keeps ringing after mode 3 has died away.",
        damping={0: 15.0},
        coords="Modal",
        by_mode=True,
        speed=0.25,
        window=2.05,
        release=2,
    ),
    SimTrial(
        "A complex mode, released",
        "<m>c<sub>1</sub></m> = 15 again, with Method → State-space: the table lists <m>2N</m> eigenvalues in "
        "conjugate pairs. Mode 3's <m>λ</m> is released with its velocities as well as its displacements, so it stays in one shape as it "
        "decays. The arrows in the complex plane fan out (phases other than 0° and 180°), and the shape never "
        "passes through zero everywhere at once. Release <m>λ*</m> instead: the motion is the same.",
        damping={0: 15.0},
        state_space=True,
        speed=0.25,
        window=2.12,
        release=4,
        animate=True,
    ),
    SimTrial(
        "An overdamped pair (c1 = 200)",
        "A damper so strong that one pair of eigenvalues turns into two real ones, one slow and one very "
        "fast, listed with their time constants instead of a frequency. There are still <m>2N</m> in all.",
        damping={0: 200.0},
        state_space=True,
    ),
    SimTrial(
        "Friction: a straight-line decay",
        "One mass with 0.5 N of friction and a damper too weak to matter (0.01 N·s/m), released from 20 mm. "
        "Each cycle it loses the same "
        "<m>4F<sub>f</sub>/k</m> = 5 mm, not the same fraction: after four cycles (1.3 s) it sticks, at rest. The "
        "modal table's ζ friction column is the equivalent viscous damping at the release amplitude.",
        n=1,
        damping={0: 0.01},
        friction=0.5,
        window=2.0,
        release=0,
    ),
    SimTrial(
        "A chirp through the modes",
        "A sweep from 2 to 7 Hz over 40 s at <m>m<sub>4</sub></m>, in modal coordinates: mode 1 rings from the switch-on and "
        "decays, then modes 2, 3 and 4 swell in turn as the sweep passes them. Sweep faster (Sweep time) and "
        "the peaks come later and lower.",
        coords="Modal",
        kind=ForceKind.CHIRP,
        sweep=(2.0, 7.0, 40.0),
        apply=True,
        window=40.0,
        frf_tab=True,
    ),
    SimTrial(
        "Ground motion: isolation",
        "The ground moves 5 mm at 1.6 Hz instead of a force pushing <m>m<sub>4</sub></m>. The Frequency response tab shows the "
        "transmissibility: 1 at low frequency, peaks at the modes, and below 1 well above them. At 1.6 Hz "
        "<m>m<sub>3</sub></m> already moves less than the ground.",
        base=(5.0, 1.6),
        apply=True,
        window=4.0,
        frf_tab=True,
    ),
    SimTrial(
        "Vibration absorber (undamped)",
        "A small mass tuned to the machine's frequency, driven there. After the start-up transient <m>m<sub>1</sub></m> stands "
        "still and only the absorber moves: in Element forces its spring carries the applied force. Change "
        "the drive frequency a little and <m>m<sub>1</sub></m> starts moving again.",
        preset="Vibration absorber",
        coords="Element forces",
        apply=True,
    ),
    SimTrial(
        "Tuned mass damper (Den Hartog)",
        "The Frequency response tab: the dashed primary alone peaks at 40 times its static deflection, the "
        "damped system at under 6. Set <m>c<sub>2</sub></m> to 0: two sharp peaks. Set it to 5: the damper locks the "
        "absorber to <m>m<sub>1</sub></m> and one tall peak comes back. Every curve passes close to the same two fixed points.",
        preset="Tuned mass damper",
        frf_tab=True,
        apply=True,
    ),
)


# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


def _two_masses(rng: np.random.Generator) -> tuple[float, float, float, float]:
    return (
        pick(rng, (1, 2, 4)),
        pick(rng, (0.5, 1, 2)),
        pick(rng, (200, 400, 800)),
        pick(rng, (100, 200, 400)),
    )


def _two_mass_setup(m1: float, m2: float, k1: float, k2: float) -> dict:
    return dict(masses=(m1, m2), springs=(k1, k2), dampers=(0.2, 0.2), release=0)


def _lowest_frequency(rng: np.random.Generator) -> Question:
    m1, m2, k1, k2 = _two_masses(rng)
    w = modal_analysis(ChainSystem(np.array([m1, m2]), np.array([k1, k2]), np.zeros(2))).modes[0].omega_n
    return Question(
        f"A chain of two masses: m<sub>1</sub> = {m1:g} kg on k<sub>1</sub> = {k1:g} N/m to the ground, and "
        f"m<sub>2</sub> = {m2:g} kg on k<sub>2</sub> = {k2:g} N/m to m<sub>1</sub>. What is its lowest natural "
        "frequency?",
        w / TWO_PI,
        f"K = [[k<sub>1</sub> + k<sub>2</sub>, −k<sub>2</sub>], [−k<sub>2</sub>, k<sub>2</sub>]], M = "
        f"diag(m<sub>1</sub>, m<sub>2</sub>); det(K − ω²M) = 0 is a quadratic in ω²: m<sub>1</sub>m<sub>2</sub>ω⁴ "
        f"− ((k<sub>1</sub> + k<sub>2</sub>)m<sub>2</sub> + k<sub>2</sub>m<sub>1</sub>)ω² + k<sub>1</sub>k<sub>2</sub> "
        f"= 0. Its smaller root is ω² = {number(w * w)}, so f<sub>1</sub> = {number(w)}/2π = {number(w / TWO_PI)} Hz.",
        unit="Hz",
        setup=_two_mass_setup(m1, m2, k1, k2),
    )


def _shape_ratio(rng: np.random.Generator) -> Question:
    m1, m2, k1, k2 = _two_masses(rng)
    w2 = modal_analysis(ChainSystem(np.array([m1, m2]), np.array([k1, k2]), np.zeros(2))).modes[0].omega_n ** 2
    w2 = rounded(w2)
    ratio = (k1 + k2 - w2 * m1) / k2
    return Question(
        f"The same kind of chain: m<sub>1</sub> = {m1:g} kg on k<sub>1</sub> = {k1:g} N/m, m<sub>2</sub> = "
        f"{m2:g} kg on k<sub>2</sub> = {k2:g} N/m. Its first mode has ω² = {number(w2)} (rad/s)². In that mode, "
        "how far does m<sub>2</sub> move for each unit m<sub>1</sub> moves?",
        ratio,
        f"The first row of (K − ω²M)φ = 0: (k<sub>1</sub> + k<sub>2</sub> − ω²m<sub>1</sub>)φ<sub>1</sub> = "
        f"k<sub>2</sub>φ<sub>2</sub>, so φ<sub>2</sub>/φ<sub>1</sub> = ({k1 + k2:g} − {number(w2)} × {m1:g})/{k2:g} = "
        f"{number(ratio)}: positive, the masses move together.",
        setup=_two_mass_setup(m1, m2, k1, k2),
    )


def _rayleigh(rng: np.random.Generator) -> Question:
    m = pick(rng, (1, 2, 5))
    k = pick(rng, (200, 400, 1000))
    guess = pick(rng, (1.5, 2.0, 3.0))
    x = np.array([1.0, guess])
    system = ChainSystem(np.full(2, m), np.full(2, k), np.zeros(2))
    _, _, K = system.matrices()
    w2 = float(x @ K @ x) / (m * float(x @ x))
    exact = modal_analysis(system).modes[0].fn_hz
    return Question(
        f"Two {m:g} kg masses on two {k:g} N/m springs (ground, m<sub>1</sub>, m<sub>2</sub>). Guess the first "
        f"mode's shape as x = (1, {guess:g}). What frequency does Rayleigh's quotient give?",
        math.sqrt(w2) / TWO_PI,
        f"ω² ≈ xᵀKx / xᵀMx: the springs stretch by 1 and {number(guess - 1)}, so xᵀKx = {k:g}(1 + "
        f"{number((guess - 1) ** 2)}) = {number(float(x @ K @ x))}; xᵀMx = {m:g}(1 + {number(guess**2)}) = "
        f"{number(m * float(x @ x))}. ω² = {number(w2)}, f = {number(math.sqrt(w2) / TWO_PI)} Hz, above the exact "
        f"{number(exact)} Hz, as it always is.",
        unit="Hz",
        setup=dict(masses=(m, m), springs=(k, k), dampers=(0.2, 0.2), release=0),
    )


def _proportional_zeta(rng: np.random.Generator) -> Question:
    beta = pick(rng, (0.002, 0.005, 0.01))
    r = int(pick(rng, (2, 3, 4)))
    w = modal_analysis(ChainSystem.uniform(4)).modes[r - 1].omega_n
    w = rounded(w)
    return Question(
        f"The default chain (four 1 kg masses on 400 N/m springs) has every damper β = {beta:g} s times its "
        f"spring. Mode {r} is at ω = {number(w)} rad/s. What is its damping ratio?",
        beta * w / 2.0,
        f"C = βK keeps the modes real, and each mode's ζ = φᵀCφ/2ω = βω²/2ω = βω/2 = {beta:g} × {number(w)}/2 = "
        f"{number(beta * w / 2)}: the higher modes are damped more.",
        setup=dict(damping={i: 400.0 * beta for i in range(4)}, coords="Modal", release=r - 1),
    )


def _absorber_stiffness(rng: np.random.Generator) -> Question:
    m = pick(rng, (1, 2, 5))
    k = pick(rng, (400, 1000, 2000))
    ma = rounded(pick(rng, (0.05, 0.1, 0.2)) * m)
    f = rounded(math.sqrt(k / m) / TWO_PI)
    ka = ma * (TWO_PI * f) ** 2
    return Question(
        f"A {m:g} kg machine on {k:g} N/m is shaken at {number(f)} Hz. A {number(ma)} kg absorber is to be hung "
        "on it to hold it still at that frequency. What spring should the absorber have?",
        ka,
        f"Tuned to the drive, the absorber's spring pushes back with the whole force and the machine stands "
        f"still: k<sub>a</sub> = m<sub>a</sub>(2πf)² = {number(ma)} × {number(TWO_PI * f)}² = {number(ka)} N/m.",
        unit="N/m",
        setup=dict(
            masses=(m, ma),
            springs=(k, rounded(ka, 4)),
            dampers=(0.2, 0.0),
            target=0,
            freq_hz=f,
            apply=True,
            frf_tab=True,
        ),
    )


def _proportional_release(rng: np.random.Generator) -> Question:
    r = int(pick(rng, (2, 3)))
    return Question(
        f"The default chain, its dampers proportional to its springs (C = βK), is released in mode {r}'s shape. "
        f"What do the other modes do as it decays?",
        0,
        "Nothing: proportional damping doesn't couple the modes, so each modal coordinate obeys its own "
        f"equation and only mode {r}'s has started moving. It decays in one clean shape.",
        choices=("They stay still", f"They pick up energy from mode {r}", "They all decay together"),
        setup=dict(coords="Modal", by_mode=True, speed=0.25, window=2.05, release=r - 1),
        predict=True,
    )


def _one_strong_damper(rng: np.random.Generator) -> Question:
    return Question(
        "The default chain with only its first damper made strong (c<sub>1</sub> = 15 N·s/m, the others 2) is "
        "released in mode 3's shape. What do modes 1 and 2 do?",
        0,
        "They pick up energy: the damping is no longer proportional, φᵀCφ has terms off the diagonal, and they "
        "couple the modes. Mode 1, the least damped, ends up with most of what is left.",
        choices=("They pick up energy from mode 3", "They stay still", "They lose energy to mode 3"),
        setup=dict(damping={0: 15.0}, coords="Modal", by_mode=True, speed=0.25, window=2.05, release=2),
        predict=True,
    )


def _absorber_at_tuning(rng: np.random.Generator) -> Question:
    return Question(
        "A machine with an undamped absorber tuned to the drive's frequency is driven at that frequency. What "
        "does the machine do once the start-up has died away?",
        0,
        "It stands still: the absorber swings so that its spring pushes on the machine with exactly the "
        "opposite of the applied force. (The absorber adds two new resonances either side, though.)",
        choices=("It stands still", "It swings at resonance", "It swings half as much as without the absorber"),
        setup=dict(preset="Vibration absorber", apply=True, frf_tab=True),
        predict=True,
    )


def _absorber_spec(rng: np.random.Generator) -> Question:
    m = pick(rng, (1, 2, 5))
    k = pick(rng, (400, 1000, 2000))
    c = rounded(0.04 * math.sqrt(k * m))
    force = pick(rng, (1, 2, 5))
    f = rounded(math.sqrt(k / m) / TWO_PI)
    w2 = (TWO_PI * f) ** 2
    cap = rounded(0.2 * m)
    ma = rounded(0.8 * cap)
    ka = rounded(ma * w2, 5)
    stroke = rounded(1.2e3 * force / ka, 2)  # mm: the absorber's swing ≈ F/k_a
    alone = 1e3 * force * abs(frf(ChainSystem(np.array([m]), np.array([k]), np.array([c])), np.array([f]), 0)[0, 0])
    still = rounded(0.01 * alone, 2)  # mm: a hundredth of its swing without the absorber

    def goal(t: SimTrial):
        system = chain(t)
        driven = t.kind in (None, ForceKind.HARMONIC) and t.base is None
        drive = Requirement(
            f"The drive: {force:g} N at {number(f)} Hz on m<sub>1</sub>",
            f"{number(t.force or 0)} N at {number(t.freq_hz or 0)} Hz on m{(t.target or 0) + 1}",
            driven
            and t.target == 0
            and abs((t.force or 0) - force) <= 0.005 * force
            and abs((t.freq_hz or 0) - f) <= 0.0005 * f,
        )
        x = 1e3 * force * np.abs(frf(system, np.array([f]), 0)[0]) if system.n == 2 else np.full(2, np.inf)
        return (
            kept("The machine's mass m<sub>1</sub>", float(system.masses[0]), m, "kg"),
            kept("Its mount k<sub>1</sub>", float(system.stiffness[0]), k, "N/m"),
            kept("Its damper c<sub>1</sub>", float(system.damping[0]), c, "N·s/m"),
            drive,
            Requirement("One absorber (two masses)", f"{system.n} mass{'es' if system.n > 1 else ''}", system.n == 2),
            at_most("The absorber's mass", float(system.masses[1]) if system.n > 1 else 0.0, cap, "kg"),
            at_most("The machine's swing", float(x[0]), still, "mm"),
            at_most("The absorber's swing", float(x[1]), stroke, "mm"),
        )

    return Question(
        f"A {m:g} kg machine on a {k:g} N/m mount with a {number(c)} N·s/m damper is shaken by {force:g} N at "
        f"{number(f)} Hz, its resonance. Hang an absorber on it (a second mass on its own spring) so the machine "
        f"swings no more than ±{number(still, 2)} mm. The absorber may weigh {number(cap)} kg at most, and has "
        f"room to swing ±{number(stroke, 2)} mm.",
        0,
        f"Tuned to the drive, k<sub>a</sub> = m<sub>a</sub>ω² with ω² = {number(w2)}, the absorber's spring "
        f"cancels the force: the machine stands still and the absorber swings F/k<sub>a</sub>. Room for "
        f"±{number(stroke, 2)} mm needs k<sub>a</sub> ≥ {number(1e3 * force / stroke)} N/m, so m<sub>a</sub> ≥ "
        f"{number(1e3 * force / stroke / w2)} kg. Take {number(ma)} kg on {number(ka, 5)} N/m, with no damper "
        f"(one would let the machine move). Without it the machine swings ±{number(alone)} mm.",
        setup=dict(masses=(m,), springs=(k,), dampers=(c,), target=0, freq_hz=f, force=force, frf_tab=True),
        goal=goal,
        design=dict(masses=(m, ma), springs=(k, ka), dampers=(c, 0.0), target=0, freq_hz=f, force=force, frf_tab=True),
    )


PROBLEMS = tuple(
    Problem(key, title, "sim_page", f"See {page_link('sim_page', anchor, section)}{more}.", make)
    for key, title, anchor, section, more, make in (
        (
            "lowest-frequency",
            "Two masses: the lowest mode",
            "undamped-normal-modes-real",
            "Undamped normal modes",
            ": det(K − ω²M) = 0",
            _lowest_frequency,
        ),
        (
            "shape-ratio",
            "Two masses: the mode's shape",
            "undamped-normal-modes-real",
            "Undamped normal modes",
            "",
            _shape_ratio,
        ),
        ("rayleigh", "Rayleigh's quotient", "rayleigh-quotient", "Estimating a frequency", "", _rayleigh),
        (
            "proportional-zeta",
            "Proportional damping's ζ",
            "proportional-damping",
            "Proportional damping",
            "",
            _proportional_zeta,
        ),
        (
            "absorber-stiffness",
            "Tune an absorber",
            "vibration-absorbers-and-tuned-mass-dampers",
            "Vibration absorbers",
            "",
            _absorber_stiffness,
        ),
        (
            "proportional-release",
            "Predict: a mode released",
            "watching-the-modal-coordinates",
            "Watching the modal coordinates",
            "",
            _proportional_release,
        ),
        (
            "one-strong-damper",
            "Predict: one strong damper",
            "non-proportional-damping-and-coupling",
            "Non-proportional damping",
            "",
            _one_strong_damper,
        ),
        (
            "absorber-at-tuning",
            "Predict: an absorber at its tuning",
            "vibration-absorbers-and-tuned-mass-dampers",
            "Vibration absorbers",
            "",
            _absorber_at_tuning,
        ),
        (
            "absorber-spec",
            "Spec: hold a machine still",
            "vibration-absorbers-and-tuned-mass-dampers",
            "Vibration absorbers",
            ": what sets the absorber's spring, and what its swing?",
            _absorber_spec,
        ),
    )
)
