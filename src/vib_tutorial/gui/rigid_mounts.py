"""Rigid body on mounts page: a box on four mounts in 3D, its six coupled modes, the elastic centre, decoupling.

Left: the body (a uniform box: mass and size), the mounts (where they sit, how far below the centre of mass,
their tilt toward the middle, their stiffness along their axes and across), and which mode to show. Centre: the
box on its mounts in 3D (drag to turn it), swinging in the mode picked, and the six modes' frequencies with
their kinetic energy's split among the six coordinates. Right: the frequencies as one quantity is swept, each
plane's elastic centre beside the mounts' lines of action, and the Theory. The page has its own model
(core/rigid_mounts.py) and animates the mode on its own frame loop while it shows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
import scipy.optimize
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.practice import Problem, Question, Requirement, kept, number, pick
from ..core.rigid_mounts import (
    DOF,
    EXCITATIONS,
    SWEEPS,
    TWO_PI,
    Body,
    Drive,
    Excitation,
    Layout,
    Sweep,
    amplitude,
    elastic_centre,
    mount_forces,
    participation,
    plane_centres,
    response,
    spectrum,
    sweep,
    torque_roll_axis,
)
from . import mounts_figures
from .frame_loop import FrameLoop
from .layouts import form_layout
from .lesson import (
    Dot,
    Form,
    RunButton,
    TrialBox,
    column,
    combo,
    connect_all,
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
from .style import colors
from .theming import add_legend

TAB_SWEEP, TAB_CENTRE, TAB_RESPONSE, TAB_DRIVE = range(4)
FORCED = 6  # the Show combo's item after the six modes: the steady response at the speed
SPEEDS = np.geomspace(60.0, 12000.0, 500)  # rpm, the Response tab's
SHOW_HZ = 0.8  # the animation's own pace: every mode swings at it
SWING = 0.06  # m, how far the farthest point swings in the animation
SPRING = 0.12  # m, a mount's drawn length
COILS = 6
SWEEP_POINTS = 61
SWEEP_RANGES = {  # the swept quantity's range, in the units its box shows
    Sweep.DROP: (0.0, 200.0),
    Sweep.TILT: (0.0, 45.0),
    Sweep.OFFSET: (-150.0, 150.0),
    Sweep.SHEAR: (0.05, 1.0),
}
SWEEP_UNITS = {Sweep.DROP: "mm", Sweep.TILT: "°", Sweep.OFFSET: "mm", Sweep.SHEAR: ""}


@dataclass(frozen=True)
class MountTrial:
    name: str
    note: str
    mass: float = 20.0
    length: float = 0.6
    width: float = 0.5
    height: float = 0.6
    span: float = 0.25
    track: float = 0.12
    drop: float = 0.0
    offset: float = 0.0
    tilt: float = 0.0  # degrees
    stiffness: float = 50e3  # N/m
    shear_ratio: float = 0.25
    mode: int = 0
    sweep: Sweep = Sweep.DROP
    tab: int = TAB_SWEEP
    excitation: Excitation = Excitation.UNBALANCE
    rpm: float = 1500.0
    unbalance: float = 0.002  # kg·m
    at_x: float = 0.1  # m, where the unbalance turns
    at_z: float = 0.1
    torque: float = 10.0  # N·m
    order: int = 2
    crank: float = 0.0  # degrees, the crank axis up from x
    loss: float = 0.1  # η, the mounts' loss factor

    @property
    def drive(self) -> Drive:
        return Drive(
            self.excitation,
            self.rpm,
            self.unbalance,
            (self.at_x, 0.0, self.at_z),
            self.torque,
            self.order,
            math.radians(self.crank),
        )

    @property
    def layout(self) -> Layout:
        body = Body(self.mass, self.length, self.width, self.height)
        return Layout(
            body,
            self.span,
            self.track,
            self.drop,
            self.offset,
            math.radians(self.tilt),
            self.stiffness,
            self.shear_ratio,
        )


FOCUS = round(math.degrees(Layout(drop=0.08).focus_tilt()), 2)  # the focalized trial's tilt, as its box shows it

TRIALS = (
    MountTrial(
        "Level with the centre of mass: six pure modes",
        "A 20 kg box 0.6 × 0.5 × 0.6 m on four mounts of 50 kN/m (12.5 across) level with its centre of mass. "
        "Every mode is pure, all its kinetic energy in one coordinate: <m>x</m> and <m>y</m> at 7.96 Hz (<m>√(4k<sub>s</sub>/m)/2π</m>), "
        "roll 8.47, yaw 9.79, bounce 15.9 (<m>√(4k<sub>a</sub>/m)/2π</m>) and pitch 16.2 Hz. The elastic centre is the "
        "centre of mass.",
        mode=4,
    ),
    MountTrial(
        "Mounts below the centre of mass: sway and roll couple",
        "The same mounts 80 mm below the centre of mass. Pushed sideways at the mounts, the box also rolls, so "
        "sway and roll share two modes: 6.90 Hz (y 67 %, roll 33 %), the box rocking about a point below it, and "
        "9.77 Hz (roll 67 %, y 33 %), rocking about a point above. The elastic centre has dropped to the mounts' "
        "plane. Fore-and-aft and pitch couple too, but hardly (99 %): their frequencies are far apart.",
        drop=0.08,
        mode=0,
    ),
    MountTrial(
        "Tilted to the focus: decoupled again",
        f"Each mount leaning {FOCUS:g}° toward the middle, the pair's lines of action meet above them, and the roll "
        "plane's elastic centre, <m>t(k<sub>a</sub> − k<sub>s</sub>) sin β cos β/(k<sub>a</sub> sin<sup>2</sup>β + k<sub>s</sub> "
        "cos<sup>2</sup>β)</m> above the mounts, rises the 80 mm to the centre of mass: sway and roll are pure again (roll 7.56 Hz, "
        "y 8.91 Hz). That's a focalized mount system. The tilt is in the roll plane only, so the pitch plane's "
        "centre stays at the mounts: no single elastic centre now.",
        drop=0.08,
        tilt=FOCUS,
        sweep=Sweep.TILT,
        tab=TAB_CENTRE,
    ),
    MountTrial(
        "Centre of mass off-centre: bounce and pitch couple",
        "The mounts level again, but the centre of mass 80 mm ahead of their middle: a weight pushes the rear "
        "mounts less than the front, so bounce and pitch couple, 13.7 Hz (z 61 %, pitch 39 %) and 18.9 Hz, and so "
        "do sway and yaw, 7.30 Hz (y 83 %, yaw 17 %) and 10.7 Hz. The elastic centre is 80 mm behind the centre of "
        "mass, under the mounts' middle: the cure is to move the mounts, or stiffen the ones that carry more.",
        offset=0.08,
        mode=4,
        sweep=Sweep.OFFSET,
    ),
    MountTrial(
        "Stiff in shear: no focus to be had",
        "Mounts as stiff across as along (<m>k<sub>s</sub> = k<sub>a</sub></m>) 80 mm below: tilting one changes nothing, "
        "since it is equally stiff every way, and the roll plane's elastic centre stays at the mounts whatever the "
        "tilt. Focalizing needs <m>k<sub>s</sub></m> &lt; <m>k<sub>a</sub></m>, and the softer in shear the higher it can lift "
        "the centre: <m>d ≤ t(1 − ρ<sub>s</sub>)/2√ρ<sub>s</sub></m>.",
        drop=0.08,
        shear_ratio=1.0,
        sweep=Sweep.TILT,
    ),
    MountTrial(
        "An unbalance at 3000 rpm: isolated",
        "The level box with 2 kg·mm of unbalance turning about a crank along x, 100 mm ahead of its centre of mass "
        "and 100 mm up, at 3000 rpm (50 Hz). It pushes <m>m<sub>e</sub>eΩ<sup>2</sup></m> = 197 N, turning; 22.4 N reaches the floor, 11.3 %: "
        "the load turns three times faster than the stiffest mode, and the box's inertia carries it. Above about "
        "1350 rpm the mounts isolate; the Response tab shows where they don't.",
        rpm=3000.0,
        mode=FORCED,
        tab=TAB_RESPONSE,
    ),
    MountTrial(
        "Running at the bounce: amplified",
        "The same unbalance at 955 rpm, where it turns at the bounce mode's 15.9 Hz: 20.0 N of unbalance puts "
        "201 N through the floor, 10.1 times as much (about 1/η), held back only by the rubber's loss factor "
        "<m>η</m> = 0.1.",
        rpm=955.0,
        mode=FORCED,
        tab=TAB_RESPONSE,
    ),
    MountTrial(
        "Lossier rubber: less at resonance, as much isolation",
        "<m>η</m> = 0.3: at 955 rpm the floor gets 70.0 N (3.5 times the unbalance's), not 201; at 3000 rpm 23.2 N, "
        "hardly more than <m>η</m> = 0.1's 22.4. Rubber's hysteretic damping (the Damping models page's) stiffens a mount "
        "by <m>|1 + iη|</m> at every frequency, not more at high ones as a viscous damper's <m>iωc</m> does, so it costs little "
        "isolation.",
        rpm=955.0,
        loss=0.3,
        mode=FORCED,
        tab=TAB_RESPONSE,
    ),
    MountTrial(
        "A firing torque: roll alone",
        "A torque about the crank axis (<m>x</m>), twice a turn at 1500 rpm (50 Hz), on the level mounts: it drives the "
        "roll mode and nothing else (the Drive tab), so the box only rolls. That is what decoupling buys.",
        excitation=Excitation.TORQUE,
        mode=FORCED,
        tab=TAB_DRIVE,
    ),
    MountTrial(
        "Mounts below: the torque drives two modes",
        "The same torque with the mounts 80 mm below the centre of mass: the roll it drives is shared by the two "
        "sway–roll modes, so it drives both (modal forces 1.00 and 0.70), and the box sways as it rolls.",
        excitation=Excitation.TORQUE,
        drop=0.08,
        mode=FORCED,
        tab=TAB_DRIVE,
    ),
    MountTrial(
        "A tilted crank: the torque roll axis",
        "A box 0.8 m long with its crank tilted 20° up from <m>x</m>. A free body under a torque about <m>n</m> turns about "
        "<m>I<sup>−1</sup>n</m>, not <m>n</m>: here 14.0° up (<m>tan γ' = (I<sub>xx</sub>/I<sub>zz</sub>) tan γ</m>), the torque roll axis "
        "(green in the 3D view; dashed, the crank). The torque now drives yaw as well as roll (0.30 of it). Engine "
        "mounts are aimed at this axis, so that the firing torque drives one mode about it.",
        excitation=Excitation.TORQUE,
        length=0.8,
        crank=20.0,
        mode=FORCED,
        tab=TAB_DRIVE,
    ),
)


# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


def _bounce(rng: np.random.Generator) -> Question:
    m, k = pick(rng, (10.0, 20.0, 50.0, 100.0)), pick(rng, (20.0, 50.0, 100.0, 200.0))
    f = math.sqrt(4 * k * 1e3 / m) / TWO_PI
    return Question(
        f"A {m:g} kg machine sits on four mounts of {k:g} kN/m each, level with its centre of mass and evenly "
        "spaced about it. At what frequency does it bounce?",
        f,
        f"Bounce is a pure mode here: the four mounts in parallel, 4k = {number(4 * k)} kN/m, carry m: "
        f"f = √(4k/m)/2π = {number(f)} Hz.",
        unit="Hz",
        setup=dict(mass=m, stiffness=k * 1e3, mode=4),
    )


def _roll(rng: np.random.Generator) -> Question:
    m = pick(rng, (10.0, 20.0, 40.0))
    w, h = pick(rng, (0.4, 0.5, 0.6)), pick(rng, (0.3, 0.4, 0.6))
    t, k = pick(rng, (0.1, 0.12, 0.15)), pick(rng, (20.0, 50.0, 100.0))
    i_xx = m * (w * w + h * h) / 12
    f = math.sqrt(4 * k * 1e3 * t * t / i_xx) / TWO_PI
    return Question(
        f"A {m:g} kg box, {w * 1000:g} mm wide and {h * 1000:g} mm tall, sits on four mounts of {k:g} kN/m along "
        f"their axes, {t * 1000:g} mm either side of its centre and level with its centre of mass. At what "
        "frequency does it roll?",
        f,
        f"Level with the centre of mass, roll is pure: each mount stretches t θ, so the roll stiffness is "
        f"4k t² = {number(4 * k * 1e3 * t * t)} N·m/rad, against I<sub>xx</sub> = m(w² + h²)/12 = {number(i_xx)} kg·m²: "
        f"f = √(4kt²/I<sub>xx</sub>)/2π = {number(f)} Hz.",
        unit="Hz",
        setup=dict(mass=m, width=w, height=h, track=t, stiffness=k * 1e3, mode=2),
    )


def _focus(rng: np.random.Generator) -> Question:
    t, h, rho = pick(rng, (0.12, 0.15, 0.2)), pick(rng, (0.03, 0.05, 0.06)), pick(rng, (0.2, 0.25))
    layout = Layout(track=t, drop=h, shear_ratio=rho)
    beta = math.degrees(layout.focus_tilt())
    return Question(
        f"Mounts {t * 1000:g} mm either side of a machine's middle sit {h * 1000:g} mm below its centre of mass, "
        f"their stiffness across {rho:g} of that along their axes. How far should each lean toward the middle, in "
        "degrees, to decouple sway from roll?",
        beta,
        f"The roll plane's elastic centre must rise h to the centre of mass: t(1 − ρ) sin β cos β/(sin²β + ρ cos²β) "
        f"= h, so with u = tan β, h u² − t(1 − ρ) u + hρ = 0. Its smaller root is u = {number(math.tan(math.radians(beta)))}: "
        f"β = {number(beta)}°.",
        unit="°",
        setup=dict(track=t, drop=h, shear_ratio=rho, tilt=round(beta, 2), sweep=Sweep.TILT, tab=TAB_CENTRE),
    )


def _lowered(rng: np.random.Generator) -> Question:
    return Question(
        "A box on four upright mounts level with its centre of mass has six pure modes. The mounts are moved "
        "down, below the centre of mass. Which modes couple?",
        0,
        "Sway with roll and fore-and-aft with pitch: the mounts' shear now acts below the centre of mass, so a "
        "sideways push also turns it, and turning it also moves the mounts sideways. Bounce and yaw stay pure.",
        choices=(
            "Sway with roll, and fore-and-aft with pitch",
            "Bounce with pitch",
            "Sway with yaw",
            "None: they stay pure",
        ),
        setup=dict(drop=0.08, mode=0),
        predict=True,
    )


def _offset(rng: np.random.Generator) -> Question:
    return Question(
        "A box on four level mounts has its centre of mass moved forward, so the mounts no longer sit evenly "
        "about it. Which modes couple?",
        0,
        "Bounce with pitch, and sway with yaw: a vertical force through the centre of mass now loads the front "
        "and rear mounts unequally and turns the box, and the same goes for a sideways one about the vertical.",
        choices=(
            "Bounce with pitch, and sway with yaw",
            "Sway with roll",
            "Fore-and-aft with pitch",
            "None: they stay pure",
        ),
        setup=dict(offset=0.08, mode=4, sweep=Sweep.OFFSET),
        predict=True,
    )


def _decouple_spec(rng: np.random.Generator) -> Question:
    drop = pick(rng, (0.04, 0.06, 0.08))
    start = MountTrial("", "", drop=drop)
    beta = math.degrees(start.layout.focus_tilt())

    def goal(t: MountTrial):
        modes = t.layout.modes()
        sway = max(modes.energy[1])  # the purest sway mode's share of y
        roll = max(modes.energy[3])
        return (
            kept("Drop", t.drop * 1000, drop * 1000, "mm"),
            kept("Track", t.track * 1000, start.track * 1000, "mm"),
            kept("Shear over axial", t.shear_ratio, start.shear_ratio, ""),
            Requirement("A sway mode at least 99 % sway", f"{number(100 * sway)} %", sway >= 0.99),
            Requirement("A roll mode at least 99 % roll", f"{number(100 * roll)} %", roll >= 0.99),
        )

    return Question(
        f"These mounts sit {drop * 1000:g} mm below the box's centre of mass, so sway and roll couple. Without "
        "moving them or changing their stiffness, make a mode at least 99 % sway and another at least 99 % roll.",
        0,
        f"Lean them toward the middle until the roll plane's elastic centre reaches the centre of mass: "
        f"h u² − t(1 − ρ) u + hρ = 0 with u = tan β gives β = {number(beta)}°.",
        setup=dict(drop=drop, sweep=Sweep.TILT, tab=TAB_CENTRE),
        goal=goal,
        design=dict(drop=drop, tilt=round(beta, 2), sweep=Sweep.TILT, tab=TAB_CENTRE),
    )


def _transmitted(rng: np.random.Generator) -> Question:
    fn, f, eta = pick(rng, (8.0, 10.0, 12.0, 15.0)), pick(rng, (25.0, 40.0, 50.0)), pick(rng, (0.05, 0.1, 0.2))
    r = f / fn
    t = abs(1 + 1j * eta) / abs(1 - r * r + 1j * eta)
    return Question(
        f"A machine bounces on its mounts at {fn:g} Hz, the rubber's loss factor η = {eta:g}. Running at {f:g} Hz, "
        "what fraction of its vertical out-of-balance force reaches the floor, in %?",
        100 * t,
        f"With K(1 + iη), the transmissibility is |1 + iη|/|1 − r² + iη|, r = f/f<sub>n</sub> = {number(r)}: "
        f"{number(100 * t)} %. (Nearly 1/(r² − 1): well above resonance η hardly matters.)",
        unit="%",
    )


def _roll_axis(rng: np.random.Generator) -> Question:
    length, gamma = pick(rng, (0.7, 0.8, 1.0)), pick(rng, (10.0, 15.0, 20.0, 30.0))
    body = Body(20.0, length, 0.5, 0.6)
    i_xx, _, i_zz = body.inertia
    tra = math.degrees(math.atan(i_xx / i_zz * math.tan(math.radians(gamma))))
    return Question(
        f"An engine block (a uniform box {length * 1000:g} mm long, 500 mm wide, 600 mm tall) has its crank tilted "
        f"{gamma:g}° up from its length, in the length–height plane. Its firing torque is about the crank. About "
        "what axis would it turn, free: how many degrees up from its length?",
        tra,
        f"I α = T n, so α ∥ I⁻¹n: with n = (cos γ, 0, sin γ), tan γ' = (I<sub>xx</sub>/I<sub>zz</sub>) tan γ, "
        f"I<sub>xx</sub> = m(w² + h²)/12 and I<sub>zz</sub> = m(l² + w²)/12: γ' = {number(tra)}°. (The mass "
        "cancels.)",
        unit="°",
        setup=dict(excitation=Excitation.TORQUE, length=length, crank=gamma, mode=FORCED, tab=TAB_DRIVE),
    )


def _torque_modes(rng: np.random.Generator) -> Question:
    return Question(
        "An engine's firing torque about its crank (along x, a principal axis) acts on a block whose mounts are "
        "level with its centre of mass and even about it. Which modes does it drive?",
        0,
        "The roll mode alone: the modes are pure, and only the roll mode's shape turns about x, so only its modal "
        "force φᵀQ is not zero.",
        choices=("Roll alone", "Roll and sway", "All six", "Bounce and pitch"),
        setup=dict(excitation=Excitation.TORQUE, mode=FORCED, tab=TAB_DRIVE),
        predict=True,
    )


def _lossier(rng: np.random.Generator) -> Question:
    return Question(
        "A machine runs well above its mounts' modes. Its rubber is swapped for one with three times the loss "
        "factor. What happens to the force reaching the floor at the running speed?",
        0,
        "Almost nothing: hysteretic damping adds iη to the stiffness at every frequency, and well above resonance "
        "|1 + iη|/(r² − 1) hardly changes. It cuts the resonances (passed on start-up) instead.",
        choices=("It hardly changes", "It drops to a third", "It triples", "It drops to a ninth"),
        setup=dict(rpm=3000.0, loss=0.3, mode=FORCED, tab=TAB_RESPONSE),
        predict=True,
    )


def _isolation_spec(rng: np.random.Generator) -> Question:
    rpm = pick(rng, (1200.0, 1500.0, 1800.0))
    start = MountTrial("", "", rpm=rpm)

    def ratio(trial: MountTrial) -> float:
        layout, drive = trial.layout, trial.drive
        q = response(layout.body, layout.mounts(), trial.loss, drive.frequency(), drive.load())
        f = mount_forces(layout.mounts(), trial.loss, q)
        return float(amplitude(f.sum(axis=0)) / amplitude(drive.load()[:3]))

    # the stiffest mounts that pass a tenth, then 10 % softer (the box's boxes show a tenth of a kN/m)
    k = scipy.optimize.brentq(
        lambda log_k: ratio(MountTrial("", "", rpm=rpm, stiffness=math.exp(log_k))) - 0.1, math.log(1e3), math.log(5e4)
    )
    best = math.floor(0.9 * math.exp(k) / 100) * 100

    def goal(t: MountTrial):
        now = ratio(t)
        return (
            kept("Speed", t.rpm, rpm, "rpm"),
            kept("Mass", t.mass, start.mass, "kg"),
            kept("Unbalance", t.unbalance * 1e3, start.unbalance * 1e3, "kg·mm"),
            kept("Drop", t.drop * 1000, start.drop * 1000, "mm"),
            Requirement(
                "At most 10 % of the unbalance's force reaching the floor", f"{number(100 * now)} %", now <= 0.1
            ),
        )

    return Question(
        f"This machine runs at {rpm:g} rpm, and too much of its unbalance reaches the floor. Choose its mounts' "
        "stiffness so that at most 10 % does.",
        0,
        "Softer mounts lower every mode; the force through them falls as (f<sub>n</sub>/f)² above √2 f<sub>n</sub>. "
        f"About {number(best / 1000)} kN/m does it (the Response tab shows the floor's force against the speed).",
        setup=dict(rpm=rpm, mode=FORCED, tab=TAB_RESPONSE),
        goal=goal,
        design=dict(rpm=rpm, stiffness=best, mode=FORCED, tab=TAB_RESPONSE),
    )


PROBLEMS = tuple(
    Problem(key, title, "mounts_page", f"See {page_link('mounts_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("bounce", "Bounce on four mounts", "modes", "Six modes, and how pure they are", _bounce),
        ("roll", "Roll on four mounts", "stiffness", "The stiffness matrix", _roll),
        ("focus", "The focal tilt", "decoupling", "Decoupling", _focus),
        ("lowered", "Predict: mounts below", "decoupling", "Decoupling", _lowered),
        ("offset", "Predict: an offset centre of mass", "elastic-centre", "The elastic centre", _offset),
        ("decouple-spec", "Spec: decouple sway and roll", "decoupling", "Decoupling", _decouple_spec),
        ("transmitted", "Through the mounts", "floor", "What reaches the floor", _transmitted),
        ("roll-axis", "The torque roll axis", "torque-roll-axis", "The torque roll axis", _roll_axis),
        ("torque-modes", "Predict: a firing torque", "forced", "The forced response", _torque_modes),
        ("lossier", "Predict: lossier rubber", "floor", "What reaches the floor", _lossier),
        ("isolation-spec", "Spec: isolate the floor", "floor", "What reaches the floor", _isolation_spec),
    )
)

TIPS = {
    "mass": "<p>The box's mass, its centre of mass at its middle.</p>",
    "length": "<p>The box's size along x (fore and aft).</p>",
    "width": "<p>Its size along y (sideways).</p>",
    "height": "<p>Its size along z (up): I<sub>xx</sub> = m(w² + h²)/12, I<sub>yy</sub> = m(l² + h²)/12, "
    "I<sub>zz</sub> = m(l² + w²)/12.</p>",
    "span": "<p>Half the mounts' spacing fore and aft.</p>",
    "track": "<p>Half their spacing sideways, t.</p>",
    "drop": "<p>How far below the centre of mass the mounts act, h.</p>",
    "offset": "<p>How far the centre of mass sits ahead of the mounts' middle.</p>",
    "tilt": "<p>Each mount's axis leaning toward the middle seen from the front, β: a V (focalized) mount system.</p>",
    "stiffness": "<p>Each mount's stiffness along its axis, k<sub>a</sub>.</p>",
    "shear": "<p>Its stiffness across its axis over along it, ρ = k<sub>s</sub>/k<sub>a</sub> (rubber mounts: about "
    "0.2–0.5).</p>",
    "mode": "<p>The mode the 3D view swings in, slowed to the same pace whatever its frequency.</p>",
    "animate": "<p>Swing the box in the mode picked (on its own), or in the response at the speed (from Start). "
    "Space.</p>",
    "sweep": "<p>The quantity the Sweep tab changes, the rest as set.</p>",
    "excitation": "<p>What drives the machine: an unbalance turning with the shaft (a force mₑeΩ² going round "
    "the crank axis), or a torque about the crank axis at a multiple of the speed (an engine's firing).</p>",
    "rpm": "<p>The machine's speed Ω.</p>",
    "unbalance": "<p>mₑe: the unbalanced mass times its distance from the axis.</p>",
    "at": "<p>Where the unbalance turns, from the centre of mass: off it, its force also turns the body.</p>",
    "torque": "<p>The torque's amplitude about the crank axis.</p>",
    "order": "<p>How many times a turn the torque goes round: a four-stroke four-cylinder engine fires twice a "
    "turn.</p>",
    "crank": "<p>The crank axis's tilt up from x, in the x–z plane.</p>",
    "loss": "<p>The mounts' loss factor η: K(1 + iη), rubber's hysteretic damping (about 0.05 to 0.3).</p>",
}
VIEW_TIP = (
    "<p>The box on its mounts in 3D, swinging in the mode picked (slowed, and exaggerated): drag to turn the view, "
    "double-click to reset it. Dotted: where it rests. The dot is its centre of mass, the cross the elastic "
    "centre.</p>"
)
TABLE_TIP = (
    "<p>The six modes: each one's frequency and how its kinetic energy splits among the three translations and "
    "three rotations about the centre of mass. 100 % in one: a pure (decoupled) mode. Click a row to show it.</p>"
)
SWEEP_TIP = (
    "<p>The six frequencies as one quantity changes, each dot coloured by the coordinate with most of the mode's "
    "energy. Decoupled modes cross freely; coupled ones veer apart. The vertical line is the setting now.</p>"
)
RESPONSE_TIP = (
    "<p>Against the speed: the force each mount puts on the floor, and all four's together (thick); for an "
    "unbalance, dashed, its own force, which a bolted-down machine would pass straight to the floor. Dotted: where "
    "the load turns at a mode's frequency. The vertical line is the speed set.</p>"
)
DRIVE_TIP = "<p>Each mode's modal force |φ<sub>j</sub>ᵀQ|, over the largest: which modes this load drives.</p>"
CENTRE_TIP = (
    "<p>The box seen from the front (the roll plane, left) and from the side (the pitch plane, right): the mounts, "
    "their lines of action (dotted), the centre of mass (dot) and each plane's elastic centre (cross).</p>"
)

NOTES_HTML = load_notes("rigid_mounts", figures=mounts_figures.FIGURES, trials=trials_html(TRIALS))


def _mm(value: float) -> str:
    """A length in m as mm to a tenth, without a −0."""
    text = f"{1000 * value:.1f}"
    return "0" if text in ("0.0", "-0.0") else text.removesuffix(".0")


def _rotate(points: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Points (n, 3) turned by the rotation vector θ (Rodrigues)."""
    angle = float(np.linalg.norm(theta))
    if angle < 1e-15:
        return points
    k = theta / angle
    c, s = math.cos(angle), math.sin(angle)
    return points * c + np.cross(k, points) * s + np.outer(points @ k, k) * (1 - c)


class MountView(pg.PlotWidget):
    """The box on its mounts in 3D, orthographically projected; drag to turn it."""

    YAW, PITCH = math.radians(35.0), math.radians(22.0)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(VIEW_TIP)
        self.setAspectLocked(True)
        for side in ("left", "bottom"):
            self.hideAxis(side)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.yaw, self.pitch = self.YAW, self.PITCH
        self._drag: QtCore.QPointF | None = None
        self.layout_ = Layout()
        self.motion = np.zeros(6)
        self.centre = np.zeros(3)
        plot = self.getPlotItem()
        self.floor = QtWidgets.QGraphicsPathItem()
        self.rest = pg.PlotCurveItem(connect="finite")
        self.edges = pg.PlotCurveItem(connect="finite")
        self.springs = pg.PlotCurveItem(connect="finite")
        self.pads = pg.PlotCurveItem(connect="finite")
        self.cross = pg.PlotCurveItem(connect="finite")
        self.crank = pg.PlotCurveItem()  # the crank axis
        self.roll_axis = pg.PlotCurveItem()  # the torque roll axis
        self.com = Dot(9)
        self.drive: Drive | None = None
        for item in (
            self.floor,
            self.rest,
            self.springs,
            self.pads,
            self.edges,
            self.cross,
            self.crank,
            self.roll_axis,
        ):
            plot.addItem(item)
        plot.addItem(self.com, ignoreBounds=True)
        self.apply_theme()

    def project(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Screen (x, y) of points p (..., 3), z up: turned by the yaw about z, looked at from the pitch above."""
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        across = x * cy - y * sy
        away = x * sy + y * cy
        return across, z * math.cos(self.pitch) + away * math.sin(self.pitch)

    def _corners(self) -> np.ndarray:
        b = self.layout_.body
        return 0.5 * np.array(
            [[sx * b.length, sy * b.width, sz * b.height] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        )

    def _edges(self, corners: np.ndarray) -> np.ndarray:
        """The box's 12 edges as one polyline broken by NaN rows."""
        out = []
        for i in range(8):
            for j in range(i + 1, 8):
                if bin(i ^ j).count("1") == 1:  # corners differing in one coordinate
                    out += [corners[i], corners[j], [np.nan] * 3]
        return np.array(out)

    def set_layout(self, layout: Layout, centre: np.ndarray, drive: Drive | None = None) -> None:
        """A new body or mounts: its size sets the view's; `centre` is the elastic centre's point; with a drive,
        its crank axis (through the unbalance, or the centre of mass) and, for a torque, the torque roll axis."""
        self.layout_, self.centre, self.drive = layout, centre, drive
        self._place_static()

    def _place_static(self) -> None:
        layout = self.layout_
        b = layout.body
        mounts = layout.mounts()
        ground = np.array([np.array(m.position) - SPRING * np.array(m.axis) for m in mounts])
        z0 = float(ground[:, 2].min())
        reach = 0.75 * max(b.length, b.width) + 0.1
        rim = np.array([[sx * reach, sy * reach, z0] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))])
        sx, sy = self.project(rim)
        path = QtGui.QPainterPath()
        path.moveTo(float(sx[0]), float(sy[0]))
        for a, c in zip(sx[1:], sy[1:]):
            path.lineTo(float(a), float(c))
        self.floor.setPath(path)
        self.rest.setData(*self.project(self._edges(self._corners())))
        pads = []
        for g in ground:
            pads += [
                g + [0.03, 0.0, 0.0],
                g - [0.03, 0.0, 0.0],
                [np.nan] * 3,
                g + [0.0, 0.03, 0.0],
                g - [0.0, 0.03, 0.0],
            ]
            pads.append([np.nan] * 3)
        self.pads.setData(*self.project(np.array(pads)))
        c = self.centre
        d = 0.03
        cross = np.array([c - [d, 0, 0], c + [d, 0, 0], [np.nan] * 3, c - [0, d, 0], c + [0, d, 0], [np.nan] * 3])
        cross = np.vstack([cross, [c - [0, 0, d], c + [0, 0, d]]])
        self.cross.setData(*self.project(cross))
        long = 0.6 * max(b.length, b.height)
        if self.drive is None:
            self.crank.setData([], [])
            self.roll_axis.setData([], [])
        else:
            n = self.drive.axis()
            through = np.array(self.drive.at) if self.drive.kind is Excitation.UNBALANCE else np.zeros(3)
            self.crank.setData(*self.project(np.array([through - long * n, through + long * n])))
            if self.drive.kind is Excitation.TORQUE:
                a = torque_roll_axis(b, n)
                self.roll_axis.setData(*self.project(np.array([-long * a, long * a])))
            else:
                self.roll_axis.setData([], [])
        lowest = min(z0, -0.5 * b.height)
        top = 0.5 * b.height + 0.1
        span = max(reach * 1.2, 0.5 * (top - lowest) + 0.15)
        mid = 0.5 * (top + lowest) * math.cos(self.pitch)
        self.setRange(xRange=(-span, span), yRange=(mid - span * 0.75, mid + span * 0.75), padding=0)
        self.draw(self.motion)

    def draw(self, q: np.ndarray) -> None:
        """The box displaced by q = (u, θ) (a large rotation, so the drawing doesn't stretch)."""
        self.motion = q
        u, theta = q[:3], q[3:]
        corners = _rotate(self._corners(), theta) + u
        self.edges.setData(*self.project(self._edges(corners)))
        springs = []
        for m in self.layout_.mounts():
            top = _rotate(np.array([m.position]), theta)[0] + u
            bottom = np.array(m.position) - SPRING * np.array(m.axis)
            springs += list(_coil(bottom, top, np.array(m.axis))) + [[np.nan] * 3]
        self.springs.setData(*self.project(np.array(springs)))
        cx, cy = self.project(u)
        self.com.setData([cx], [cy])

    def apply_theme(self) -> None:
        c = colors
        grid = QtGui.QColor(c.grey)
        grid.setAlpha(40)
        self.floor.setBrush(QtGui.QBrush(grid))
        self.floor.setPen(pg.mkPen(c.grey, width=1))
        self.rest.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.edges.setPen(pg.mkPen(c.mass[0], width=2.5))
        self.springs.setPen(pg.mkPen(c.strong, width=1.5))
        self.pads.setPen(pg.mkPen(c.strong, width=3))
        self.cross.setPen(pg.mkPen(c.force, width=2.5))
        self.crank.setPen(pg.mkPen(c.grey, width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.roll_axis.setPen(pg.mkPen(c.mass[2], width=2))
        self.com.setBrush(pg.mkBrush(c.strong))
        self.com.setPen(pg.mkPen(None))

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag = event.position()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag is None:
            super().mouseMoveEvent(event)
            return
        d = event.position() - self._drag
        self._drag = event.position()
        self.yaw = (self.yaw - 0.01 * d.x()) % (2 * math.pi)
        self.pitch = float(np.clip(self.pitch + 0.01 * d.y(), -math.pi / 2, math.pi / 2))
        self._place_static()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.yaw, self.pitch = self.YAW, self.PITCH
        self._place_static()
        event.accept()


def _coil(bottom: np.ndarray, top: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """A zigzag spring from `bottom` to `top`: straight ends, COILS turns between. It zigzags across `rest`
    (the mount's axis at rest), squared to the spring as it leans: picking the side from the leaning axis
    flipped it mid-swing, where the axis crossed the choice's threshold."""
    axis = top - bottom
    length = float(np.linalg.norm(axis))
    if length == 0:
        return np.array([bottom, top])
    a = axis / length
    side = np.cross(rest, [0.0, 0.0, 1.0] if abs(rest[2]) < 0.9 else [1.0, 0.0, 0.0])
    side -= (side @ a) * a
    side /= np.linalg.norm(side)
    n = 2 * COILS
    s = np.linspace(0.15, 0.85, n + 1)
    wiggle = 0.025 * np.array([0.0] + [(-1) ** i for i in range(1, n)] + [0.0])
    middle = bottom + np.outer(s, axis) + np.outer(wiggle, side)
    return np.vstack([bottom, middle, top])


class MountsPage(QtWidgets.QWidget):
    """Controls on the left, the box in 3D and its modes in the centre, the sweep and the elastic centre right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.layout_ = TRIALS[0].layout
        self.modes = self.layout_.modes()
        self.running = True  # a mode swings on its own; the response waits for Start (_on_mode)
        self._phase = math.pi / 2  # a mode starts at its full swing

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Each sets up the box and its mounts, to show one idea from the Theory tab.</p>",
        )
        self.trials.picked.connect(self.load_trial)

        body = Form("Body")
        self.mass = body.number("Mass m:", 0.1, 10000.0, 20.0, 1, " kg", TIPS["mass"])
        self.length = body.number("Length l:", 10.0, 5000.0, 600.0, 0, " mm", TIPS["length"])
        self.body_width = body.number("Width w:", 10.0, 5000.0, 500.0, 0, " mm", TIPS["width"])
        self.body_height = body.number("Height:", 10.0, 5000.0, 600.0, 0, " mm", TIPS["height"])

        mounts = Form("Mounts")
        self.span = mounts.number("Span (half):", 1.0, 5000.0, 250.0, 0, " mm", TIPS["span"])
        self.track = mounts.number("Track t (half):", 1.0, 5000.0, 120.0, 0, " mm", TIPS["track"])
        self.drop = mounts.number("Drop below CoM h:", -1000.0, 1000.0, 0.0, 1, " mm", TIPS["drop"])
        self.offset = mounts.number("CoM ahead by:", -1000.0, 1000.0, 0.0, 1, " mm", TIPS["offset"])
        self.tilt = mounts.number("Tilt β:", 0.0, 80.0, 0.0, 2, "°", TIPS["tilt"])
        self.stiffness = mounts.number("Axial k_a:", 0.1, 1e5, 50.0, 1, " kN/m", TIPS["stiffness"])
        self.shear = mounts.number("Shear ρ = k_s/k_a:", 0.01, 1.0, 0.25, 2, "", TIPS["shear"])
        self.focus_info = note_label()
        mounts.row(self.focus_info)

        drive = Form("Drive")
        self.drive_form = drive.form
        self.excitation = combo([e.value for e in EXCITATIONS], TIPS["excitation"])
        drive.add("Load:", self.excitation, TIPS["excitation"])
        self.rpm = drive.number("Speed:", 10.0, 50000.0, 1500.0, 0, " rpm", TIPS["rpm"])
        self.unbalance = drive.number("Unbalance mₑe:", 0.0, 1e6, 2.0, 2, " kg·mm", TIPS["unbalance"])
        self.at_x = drive.number("…at x:", -2000.0, 2000.0, 100.0, 0, " mm", TIPS["at"])
        self.at_z = drive.number("…at z:", -2000.0, 2000.0, 100.0, 0, " mm", TIPS["at"])
        self.torque = drive.number("Torque T:", 0.0, 1e6, 10.0, 1, " N·m", TIPS["torque"])
        self.order = drive.number("Order n:", 1.0, 12.0, 2.0, 0, "", TIPS["order"])
        self.crank = drive.number("Crank tilt γ:", -80.0, 80.0, 0.0, 1, "°", TIPS["crank"])
        self.loss = drive.number("Mounts' loss η:", 0.0, 2.0, 0.1, 2, "", TIPS["loss"])

        show = Form("Show")
        self.mode = combo([f"Mode {i + 1}" for i in range(6)] + ["The response at the speed"], TIPS["mode"])
        show.add("Mode:", self.mode, TIPS["mode"])
        self.run_button = RunButton()
        self.run_button.set_running(True)
        self.run_button.setToolTip(TIPS["animate"])
        self.run_button.toggled.connect(self._on_animate)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setToolTip("<p>Back to rest; the response waits for Start.</p>")
        self.reset_button.clicked.connect(self.reset)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.reset_button)
        show.row(buttons)

        self.boxes = (
            self.mass,
            self.length,
            self.body_width,
            self.body_height,
            self.span,
            self.track,
            self.drop,
            self.offset,
            self.tilt,
            self.stiffness,
            self.shear,
            self.excitation,
            self.rpm,
            self.unbalance,
            self.at_x,
            self.at_z,
            self.torque,
            self.order,
            self.crank,
            self.loss,
        )
        connect_all(self.boxes, self._on_inputs)
        self.mode.currentIndexChanged.connect(self._on_mode)

        # --- centre
        self.view = MountView()
        self.table = QtWidgets.QTableWidget(6, 8)
        self.table.setToolTip(TABLE_TIP)
        self.table.setHorizontalHeaderLabels(["Mode", "f (Hz)", *DOF])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._on_row)
        self.readout = readout_label()

        # --- right
        self.sweep_box = combo([s.value for s in SWEEPS], TIPS["sweep"])
        self.sweep_box.currentIndexChanged.connect(self._draw_tab)
        self.sweep_plot = plot("Frequency", "Hz", "", None, height=260)
        self.sweep_plot.setToolTip(SWEEP_TIP)
        self.sweep_plot.getAxis("left").enableAutoSIPrefix(False)
        add_legend(self.sweep_plot.getPlotItem(), offset=(-10, 10), colCount=3)
        self.sweep_lines = pg.PlotDataItem(connect="finite")
        self.sweep_plot.addItem(self.sweep_lines)
        self.sweep_dots = [marker(6, name=name) for name in DOF]
        for item in self.sweep_dots:
            self.sweep_plot.addItem(item)
        self.sweep_now = pg.InfiniteLine(angle=90)
        self.sweep_focus = pg.InfiniteLine(angle=90)
        for item in (self.sweep_now, self.sweep_focus):
            self.sweep_plot.addItem(item)
        self.sweep_note = note_label()
        sweep_row = form_layout()
        sweep_row.addRow("Sweep:", self.sweep_box)
        sweep_tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(sweep_tab)
        v.addLayout(sweep_row)
        v.addWidget(self.sweep_plot)
        v.addWidget(self.sweep_note)
        v.addStretch(1)

        self.planes = []
        for left, bottom in (("z", "y (sideways)"), ("z", "x (fore and aft)")):
            p = plot(left, "mm", bottom, "mm", height=220)
            p.setAspectLocked(True)
            p.setToolTip(CENTRE_TIP)
            for side in ("left", "bottom"):
                p.getAxis(side).enableAutoSIPrefix(False)
            items = {
                "box": pg.PlotDataItem(),
                "lines": pg.PlotDataItem(connect="finite"),
                "mounts": marker(10, "s"),
                "com": marker(10),
                "centre": marker(14, "x"),
            }
            for item in items.values():
                p.addItem(item)
            self.planes.append((p, items))
        self.centre_note = note_label()
        planes = QtWidgets.QWidget()
        h = QtWidgets.QVBoxLayout(planes)
        h.setContentsMargins(0, 0, 0, 0)
        for p, _ in self.planes:
            h.addWidget(p)

        self.response_plot = plot("Force", "N", "Speed", "rpm", log_x=True, log_y=True, height=280)
        self.response_plot.setToolTip(RESPONSE_TIP)
        add_legend(self.response_plot.getPlotItem(), offset=(10, 10))
        self.applied_item = pg.PlotDataItem(name="the load (bolted down)")
        self.total_item = pg.PlotDataItem(name="the floor's, all four")
        self.mount_items = [pg.PlotDataItem(name=f"mount {i + 1}") for i in range(4)]
        self.response_modes = pg.PlotDataItem(connect="finite")
        self.response_now = pg.InfiniteLine(angle=90)
        for item in (self.response_modes, self.applied_item, *self.mount_items, self.total_item, self.response_now):
            self.response_plot.addItem(item)
        self.response_note = note_label()

        self.drive_plot = plot("Modal force", None, "Mode", None, height=200)
        self.drive_plot.setToolTip(DRIVE_TIP)
        self.drive_bars = pg.BarGraphItem(x=np.arange(1, 7), height=np.zeros(6), width=0.6)
        self.drive_plot.addItem(self.drive_bars)
        self.drive_plot.setXRange(0.4, 6.6, padding=0)
        self.drive_plot.setYRange(0.0, 1.1, padding=0)
        self.drive_note = note_label()

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(sweep_tab, "Sweep")
        self.tabs.addTab(column(planes, self.centre_note, stretch=False), "Elastic centre")
        self.tabs.addTab(column(self.response_plot, self.response_note, stretch=False), "Response")
        self.tabs.addTab(column(self.drive_plot, self.drive_note, stretch=False), "Drive")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._draw_tab)

        three_panes(
            self,
            column(self.trials, body, mounts, drive, show),
            stack((self.view, 4), (self.table, 0), (self.readout, 0)),
            self.tabs,
        )

        self.loop = FrameLoop(self, "Rigid body on mounts", self._tick, self._runs, self._sim_t)
        self._on_inputs()
        self.apply_theme()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(MountTrial("", "", **setup))

    def practice_state(self) -> MountTrial:
        """The page's settings, as a trial: what a practice spec checks."""
        return MountTrial(
            "",
            "",
            mass=self.mass.value(),
            length=self.length.value() / 1000,
            width=self.body_width.value() / 1000,
            height=self.body_height.value() / 1000,
            span=self.span.value() / 1000,
            track=self.track.value() / 1000,
            drop=self.drop.value() / 1000,
            offset=self.offset.value() / 1000,
            tilt=self.tilt.value(),
            stiffness=self.stiffness.value() * 1000,
            shear_ratio=self.shear.value(),
            excitation=EXCITATIONS[self.excitation.currentIndex()],
            rpm=self.rpm.value(),
            unbalance=self.unbalance.value() / 1e3,
            at_x=self.at_x.value() / 1000,
            at_z=self.at_z.value() / 1000,
            torque=self.torque.value(),
            order=int(self.order.value()),
            crank=self.crank.value(),
            loss=self.loss.value(),
            mode=self.mode.currentIndex(),
            sweep=SWEEPS[self.sweep_box.currentIndex()],
            tab=self.tabs.currentIndex(),
        )

    def apply_trial(self, t: MountTrial) -> None:
        for box, value in (
            (self.mass, t.mass),
            (self.length, t.length * 1000),
            (self.body_width, t.width * 1000),
            (self.body_height, t.height * 1000),
            (self.span, t.span * 1000),
            (self.track, t.track * 1000),
            (self.drop, t.drop * 1000),
            (self.offset, t.offset * 1000),
            (self.tilt, t.tilt),
            (self.stiffness, t.stiffness / 1000),
            (self.shear, t.shear_ratio),
            (self.excitation, EXCITATIONS.index(t.excitation)),
            (self.rpm, t.rpm),
            (self.unbalance, t.unbalance * 1e3),
            (self.at_x, t.at_x * 1000),
            (self.at_z, t.at_z * 1000),
            (self.torque, t.torque),
            (self.order, t.order),
            (self.crank, t.crank),
            (self.loss, t.loss),
            (self.mode, t.mode),
            (self.sweep_box, SWEEPS.index(t.sweep)),
        ):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self._on_inputs()
        self._phase = 0.0 if self.mode.currentIndex() == FORCED else math.pi / 2
        self.running = True  # a trial starts it
        self.run_button.set_running(True)
        self.loop.start()

    def _on_inputs(self) -> None:
        state = self.practice_state()
        self.layout_ = state.layout
        self.drive, self.loss_factor = state.drive, state.loss
        unbalance = self.drive.kind is Excitation.UNBALANCE
        for box, shown in (
            (self.unbalance, unbalance),
            (self.at_x, unbalance),
            (self.at_z, unbalance),
            (self.torque, not unbalance),
            (self.order, not unbalance),
        ):
            self.drive_form.setRowVisible(box, shown)
        self.modes = self.layout_.modes()
        mounts = self.layout_.mounts()
        point, left = elastic_centre(mounts)
        self.centre, self.left = point, left
        focus = self.layout_.focus_tilt()
        self.focus_info.setText(
            "Level with the centre of mass: no tilt needed."
            if self.layout_.drop <= 0
            else f"Focal tilt: {math.degrees(focus):.4g}°."
            if math.isfinite(focus)
            else "No focal tilt: the mounts are too far below, or too stiff across (h > t(1 − ρ)/2√ρ)."
        )
        for i in range(6):
            self.mode.setItemText(i, f"{i + 1}: {self.modes.frequencies[i]:.3g} Hz, {DOF[self.modes.dominant(i)]}")
        self.view.set_layout(self.layout_, point, self.drive)
        self.forced = self._forced()
        self._fill_table()
        roll, (px, pz) = plane_centres(mounts)
        where = f"({_mm(point[0])}, {_mm(point[1])}, {_mm(point[2])}) mm"
        self.readout.setText(
            (
                f"Elastic centre at {where} from the centre of mass"
                if left < 1e-6
                else f"No single elastic centre: the best point, {where}, leaves {100 * left:.2g} % of the coupling"
            )
            + f". The roll plane's is at z = {_mm(roll)} mm, the pitch plane's at x = {_mm(px)}, z = {_mm(pz)} mm. "
            + self._showing()
        )
        self._draw_tab()
        self._show_mode()
        self.loop.start()

    def _forced(self) -> np.ndarray:
        """The steady response q (6, complex) to the drive at its speed."""
        return response(
            self.layout_.body, self.layout_.mounts(), self.loss_factor, self.drive.frequency(), self.drive.load()
        )

    def _showing(self) -> str:
        """What the 3D view swings in."""
        i = self.mode.currentIndex()
        if i < FORCED:
            return f"Showing mode {i + 1}: {self.modes.name(i)}."
        f = mount_forces(self.layout_.mounts(), self.loss_factor, self.forced)
        hz = self.drive.frequency() / TWO_PI
        return (
            f"Showing the response at {self.drive.rpm:g} rpm ({hz:.3g} Hz): {amplitude(f.sum(axis=0)):.3g} N through "
            f"the floor, the most through one mount {amplitude(f).max():.3g} N."
        )

    def _fill_table(self) -> None:
        self.table.blockSignals(True)
        for i in range(6):
            cells = [str(i + 1), f"{self.modes.frequencies[i]:.3g}"] + [
                f"{100 * self.modes.energy[j, i]:.0f}" for j in range(6)
            ]
            for col, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                if col >= 2:
                    share = float(self.modes.energy[col - 2, i])
                    fill = QtGui.QColor(colors.mass[0])
                    fill.setAlpha(int(150 * share))
                    item.setBackground(QtGui.QBrush(fill))
                self.table.setItem(i, col, item)
        if self.mode.currentIndex() < FORCED:
            self.table.selectRow(self.mode.currentIndex())
        else:
            self.table.clearSelection()
        self.table.blockSignals(False)
        rows = sum(self.table.rowHeight(i) for i in range(6))  # all six rows, no scrolling
        self.table.setFixedHeight(self.table.horizontalHeader().height() + rows + 2 * self.table.frameWidth())

    def _on_row(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows and rows[0].row() != self.mode.currentIndex():
            self.mode.setCurrentIndex(rows[0].row())

    def _on_mode(self) -> None:
        self.reset()
        self.table.blockSignals(True)
        if self.mode.currentIndex() < FORCED:
            self.table.selectRow(self.mode.currentIndex())
        else:
            self.table.clearSelection()
        self.table.blockSignals(False)
        self._on_inputs()

    def _on_animate(self, on: bool) -> None:
        self.running = on
        self._show_mode()
        self.loop.start()

    def reset(self) -> None:
        """A mode back at its full swing, still swinging; the response at rest, waiting for Start."""
        forced = self.mode.currentIndex() == FORCED
        self._phase = 0.0 if forced else math.pi / 2
        self.running = not forced
        if forced:
            self.run_button.rest()
        else:
            self.run_button.set_running(True)
        self._show_mode()
        self.loop.start()

    # ------------------------------------------------------------- the loop

    def _runs(self) -> bool:
        return self.running

    def _sim_t(self) -> float:
        return self._phase

    def _tick(self, dt: float) -> None:
        self.step(dt)

    def frame_context(self) -> str:
        return f"mode {self.mode.currentIndex() + 1}, drop {self.drop.value():g} mm, tilt {self.tilt.value():g}°"

    def step(self, dt: float) -> None:
        """Swing the mode on by dt seconds at the animation's own pace."""
        self._phase = (self._phase + TWO_PI * SHOW_HZ * dt) % TWO_PI
        self._show_mode()

    def mode_shape(self, mode: int) -> np.ndarray:
        """The mode's shape (or, past the six, the forced response, complex), scaled so the farthest corner or
        mount swings SWING."""
        phi = self.modes.shapes[:, mode] if mode < FORCED else self.forced
        points = np.vstack([self.view._corners(), [m.position for m in self.layout_.mounts()]])
        reach = amplitude(phi[:3] + np.cross(phi[3:], points)).max()
        return phi * (SWING / reach if reach > 0 else 0.0)

    def _show_mode(self) -> None:
        i = self.mode.currentIndex()
        if i == FORCED and not self.run_button.started:  # at rest until the drive is started
            self.view.draw(np.zeros(6))
            return
        turn = np.exp(1j * (self._phase - 0.5 * math.pi))  # sin φ for a real shape
        self.view.draw(np.real(self.mode_shape(i) * turn))

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.view.apply_theme()
        self.sweep_lines.setPen(pg.mkPen(c.grey, width=1))
        for i, item in enumerate(self.sweep_dots):
            mark_ = pg.mkBrush(c.mass[i])
            item.setSymbolBrush(mark_)
            item.setSymbolPen(None)
        self.sweep_now.setPen(pg.mkPen(c.strong, width=1.5))
        self.sweep_focus.setPen(pg.mkPen(c.fit, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        for _, items in self.planes:
            items["box"].setPen(pg.mkPen(c.mass[0], width=2))
            items["lines"].setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DotLine))
            items["mounts"].setSymbolBrush(pg.mkBrush(c.strong))
            items["mounts"].setSymbolPen(None)
            items["com"].setSymbolBrush(pg.mkBrush(c.strong))
            items["com"].setSymbolPen(None)
            items["centre"].setSymbolBrush(pg.mkBrush(c.force))
            items["centre"].setSymbolPen(None)
        self.applied_item.setPen(pg.mkPen(c.grey, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.total_item.setPen(pg.mkPen(c.strong, width=2.5))
        for i, item in enumerate(self.mount_items):
            item.setPen(pg.mkPen(c.mass[i], width=1.2))
        self.response_modes.setPen(pg.mkPen(c.fit, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.response_now.setPen(pg.mkPen(c.strong, width=1.5))
        self.drive_bars.setOpts(brush=pg.mkBrush(c.mass[0]), pen=pg.mkPen(None))
        self._fill_table()
        self._draw_tab()
        self._show_mode()

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_SWEEP:
            self._draw_sweep()
        elif i == TAB_CENTRE:
            self._draw_centre()
        elif i == TAB_RESPONSE:
            self._draw_response()
        elif i == TAB_DRIVE:
            self._draw_drive()

    def _draw_response(self) -> None:
        drive = self.drive
        sp = spectrum(self.layout_, self.loss_factor, drive, SPEEDS)
        unbalance = drive.kind is Excitation.UNBALANCE
        self.applied_item.setData(SPEEDS if unbalance else [], sp.applied if unbalance else [])
        self.total_item.setData(SPEEDS, np.maximum(sp.total, 1e-12))
        for i, item in enumerate(self.mount_items):
            item.setData(SPEEDS, np.maximum(sp.mounts[:, i], 1e-12))
        per = 60.0 / (1 if unbalance else drive.order)  # rpm per Hz of the load
        lo, hi = (
            float(np.log10(max(sp.mounts.max(), sp.total.max(), sp.applied.max()) * 1e-4)),
            float(np.log10(max(sp.mounts.max(), sp.total.max(), sp.applied.max()) * 3)),
        )
        xs, ys = [], []
        for f in self.modes.frequencies:
            xs += [f * per, f * per, np.nan]
            ys += [10**lo, 10**hi, np.nan]
        self.response_modes.setData(xs, ys)
        self.response_now.setPos(math.log10(drive.rpm))
        self.response_plot.setXRange(math.log10(SPEEDS[0]), math.log10(SPEEDS[-1]), padding=0)
        self.response_plot.setYRange(lo, hi, padding=0)
        f = mount_forces(self.layout_.mounts(), self.loss_factor, self.forced)
        total, most = float(amplitude(f.sum(axis=0))), float(amplitude(f).max())
        if unbalance:
            applied = float(amplitude(drive.load()[:3]))
            ratio = total / applied if applied else math.nan
            what = f"At {drive.rpm:g} rpm the unbalance pushes {applied:.3g} N; {total:.3g} N reaches the floor, " + (
                f"{100 * ratio:.3g} % of it: isolated." if ratio < 1 else f"{ratio:.3g} times it: amplified."
            )
        else:
            what = (
                f"At {drive.rpm:g} rpm the torque turns at {drive.order}× ({drive.frequency() / TWO_PI:.3g} Hz): the "
                f"mounts push {most:.3g} N at most, the floor {total:.3g} N in all (a torque is a couple: its forces "
                "nearly cancel, but each mount carries its share)."
            )
        self.response_note.setText(
            what + " Dotted: the six modes, at the speeds where the load turns at their frequencies."
        )

    def _draw_drive(self) -> None:
        shares = participation(self.modes, self.drive.load())
        self.drive_bars.setOpts(height=shares)
        names = ", ".join(f"{j + 1} ({self.modes.name(j)})" for j in range(6) if shares[j] >= 0.01)
        text = f"The load drives mode{'s' if (shares >= 0.01).sum() > 1 else ''} {names}."
        if self.drive.kind is Excitation.TORQUE:
            n = self.drive.axis()
            a = torque_roll_axis(self.layout_.body, n)
            text += (
                f" The torque is about the crank, {self.crank.value():.3g}° up from x, but a free body turns about "
                f"I⁻¹n, the torque roll axis, {math.degrees(math.atan2(a[2], a[0])):.3g}° up (the line drawn in "
                "the 3D view): only mounts whose roll mode turns about that axis leave the torque one mode to drive."
            )
        self.drive_note.setText(text)

    def _draw_sweep(self) -> None:
        which = SWEEPS[self.sweep_box.currentIndex()]
        lo, hi = SWEEP_RANGES[which]
        shown = np.linspace(lo, hi, SWEEP_POINTS)
        scale = {Sweep.DROP: 1e-3, Sweep.OFFSET: 1e-3, Sweep.TILT: math.pi / 180, Sweep.SHEAR: 1.0}[which]
        freqs, dominant = sweep(self.layout_, which, shown * scale)
        xs = np.concatenate([np.append(shown, np.nan) for _ in range(6)])
        ys = np.concatenate([np.append(freqs[:, j], np.nan) for j in range(6)])
        self.sweep_lines.setData(xs, ys)
        for d, item in enumerate(self.sweep_dots):
            hit = dominant == d
            item.setData(np.broadcast_to(shown[:, None], hit.shape)[hit], freqs[hit])
        now = {
            Sweep.DROP: self.drop.value(),
            Sweep.TILT: self.tilt.value(),
            Sweep.OFFSET: self.offset.value(),
            Sweep.SHEAR: self.shear.value(),
        }[which]
        self.sweep_now.setPos(now)
        focus = self.layout_.focus_tilt()
        on = which is Sweep.TILT and math.isfinite(focus) and self.layout_.drop > 0
        self.sweep_focus.setVisible(on)
        if on:
            self.sweep_focus.setPos(math.degrees(focus))
        unit = SWEEP_UNITS[which]
        self.sweep_plot.setLabel("bottom", which.value + (f" ({unit})" if unit else ""))
        self.sweep_plot.setXRange(lo, hi, padding=0)
        self.sweep_plot.setYRange(0.0, float(np.nanmax(freqs)) * 1.7, padding=0)  # room above for the legend
        self.sweep_note.setText(
            "Each dot is coloured by the coordinate with most of its mode's energy. Where two modes come close, "
            "coupled ones veer apart and trade characters; decoupled ones cross."
            + (f" Dashed: the focal tilt, {math.degrees(focus):.4g}°." if on else "")
        )

    def _draw_centre(self) -> None:
        layout = self.layout_
        b = layout.body
        mounts = layout.mounts()
        roll, (px, pz) = plane_centres(mounts)
        views = (
            (b.width, lambda m: m.position[1], lambda m: m.axis[1], (0.0, roll)),
            (b.length, lambda m: m.position[0], lambda m: m.axis[0], (px, pz)),
        )
        for (p, items), (size, across, lean, centre) in zip(self.planes, views):
            half, top = 500 * size, 500 * b.height
            items["box"].setData([-half, half, half, -half, -half], [-top, -top, top, top, -top])
            lines, xs, zs = [], [], []
            for m in mounts:
                x0, z0 = 1000 * across(m), 1000 * m.position[2]
                xs.append(x0)
                zs.append(z0)
                a, c = lean(m), m.axis[2]
                lines += [[x0 - 600 * a, z0 - 600 * c], [x0 + 1200 * a, z0 + 1200 * c], [np.nan, np.nan]]
            pts = np.array(lines)
            items["lines"].setData(pts[:, 0], pts[:, 1])
            items["mounts"].setData(xs, zs)
            items["com"].setData([0.0], [0.0])
            cx, cz = centre
            items["centre"].setData([1000 * cx] if math.isfinite(cx) else [], [1000 * cz] if math.isfinite(cz) else [])
            reach = max(half, top, abs(1000 * layout.drop) + 50) * 1.25
            p.setRange(xRange=(-reach, reach), yRange=(-reach, reach), padding=0)
        self.centre_note.setText(
            f"Front (left) and side (right). The roll plane's elastic centre is at z = {_mm(roll)} mm; the pitch "
            f"plane's at x = {_mm(px)}, z = {_mm(pz)} mm, from the centre of mass. Where a plane's is at the centre of "
            "mass (the dot), its modes are pure."
        )
