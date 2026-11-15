"""Vibration isolation's view model: what the page draws and prints, from its settings (`gui/isolation.py` binds it).

The page's settings are an `IsolationTrial` (its trials, `practice_state()` and `load_setup` all use one);
`isolator(trial)` is the machine on its mounts. Each panel has a function from those to what it shows:

- `mount_info(iso)`: the mounts' k, c and f<sub>n</sub>, under their boxes.
- `transmissibility(iso)`: the centre: T against f/f<sub>n</sub> with the plain mounts' family behind it, the
  running point, its readout, and the force on the floor against the speed.
- `design_note(iso, target)`: the sag an isolation needs, under the Design box ("" when none can give it).
- `start_up_view(iso, ramp)`: the Start-up tab's run up through resonance (the costly one: drawn only when it shows).
- `chart(iso)`: the Design chart tab.

Working on it: keep it free of Qt (see the package's docstring). `tests/test_view_models.py` checks the
readouts against the numbers the notes and the trials quote, as printed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ..core.isolation import G, Isolator, Mount, deflection_for, start_up
from ..core.sdof import TWO_PI
from .plot import Curve
from .text import db, fmt

MOUNTS = list(Mount)
FAMILY = (0.0, 0.1, 0.3, 0.7)  # the damping ratios drawn behind the isolator's own curve
RATIOS = np.geomspace(0.1, 20.0, 1500)
CHART_SPEEDS = np.geomspace(5.0, 200.0, 200)  # Hz on the design chart
CHART_LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99)
TAB_START, TAB_CHART = range(2)
T_RANGE = (-4.0, math.log10(30.0))  # the transmissibility plot's y range, as log10


@dataclass(frozen=True)
class IsolationTrial:
    name: str
    note: str
    deflection_mm: float = 5.0
    zeta: float = 0.05
    mount: Mount = Mount.VISCOUS
    stiffness_ratio: float = 3.0
    stage_mass: float = 0.5
    block: float = 0.0
    rpm: float = 1500.0
    ramp: float = 2.0
    tab: int = TAB_START
    mass: float = 100.0
    unbalance: float = 0.01


def isolator(t: IsolationTrial) -> Isolator:
    """The machine on its mounts as a trial (or the page's `practice_state`) sets them."""
    return Isolator(
        t.mass,
        t.rpm,
        t.unbalance,
        1e-3 * t.deflection_mm,
        t.zeta,
        t.mount,
        t.stiffness_ratio,
        t.stage_mass,
        t.block,
    )


TRIALS = (
    IsolationTrial(
        "Soft mounts: isolation",
        "A 100 kg machine with a 0.01 kg·m unbalance at 1500 rpm (25 Hz) on mounts that sag 5 mm: <m>f<sub>n</sub></m> "
        "= <m>15.76/√5</m> = 7.05 Hz, so the speed is <m>3.5f<sub>n</sub></m>. The unbalance pulls with 247 N; only 23 N, 9 %, "
        "reaches the floor (−21 dB): 91 % isolation.",
    ),
    IsolationTrial(
        "Stiff mounts: amplification",
        "The same machine on mounts that sag only 0.3 mm: <m>f<sub>n</sub></m> = 28.8 Hz, just above the running "
        "speed. The mounts now amplify: 3.9 times the unbalance's force reaches the floor. Below <m>√2 f<sub>n</sub></m> "
        "a mount can only make things worse.",
        deflection_mm=0.3,
    ),
    IsolationTrial(
        "Damping costs isolation",
        "Back on the soft mounts, with <m>ζ</m> = 0.3 instead of 0.05. Above <m>√2 f<sub>n</sub></m> the damper passes force "
        "to the floor in proportion to the speed: 20 % now reaches it instead of 9 %. Damping only helps at "
        "resonance, which the machine passes through at start-up.",
        zeta=0.3,
    ),
    IsolationTrial(
        "Running up through resonance",
        "A 2 s run up to 1500 rpm on the soft, lightly damped mounts. As the speed passes <m>f<sub>n</sub></m> = 7 Hz "
        "(420 rpm) the machine and the floor force swell, then settle to the isolated running state. The "
        "unbalance's force grows as the speed squared, so the resonance is passed at a twelfth of the running "
        "force, which is what keeps it tolerable.",
        tab=TAB_START,
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
    ),
    IsolationTrial(
        "Two stages",
        "The machine on mounts on a 50 kg intermediate mass, itself on mounts sagging 5 mm under both. Two resonances, but "
        "above them the transmissibility falls as <m>1/r⁴</m> (80 dB a decade) instead of <m>1/r²</m>: 3.8 % reaches the "
        "floor here (one stage: 9 %), and at 3000 rpm the two stages pass 0.2 % against one stage's 2.5 %.",
        mount=Mount.TWO_STAGE,
    ),
    IsolationTrial(
        "An inertia block",
        "The machine bolted to a 300 kg block, the mounts stiffened to sag the same 5 mm. The transmissibility "
        "doesn't change (<m>f<sub>n</sub></m> depends only on <m>δ</m>), nor does the force on the floor, but the machine "
        "moves a quarter as much (±27 µm instead of ±109 µm): a block steadies the machine, it doesn't "
        "isolate better.",
        block=300.0,
    ),
)


@dataclass(frozen=True)
class Transmissibility:
    family: tuple[Curve, ...]  # plain mounts at each of FAMILY's ζ, against f/f_n (NaN where infinite)
    isolator: Curve  # this isolator, against f/f_n
    now: Curve  # the running point (empty off the plot)
    note: str  # the readout under the plot
    rigid: Curve  # the unbalance's own force against the speed (rpm)
    floor: Curve  # the force on the floor through these mounts
    force_now: Curve
    rpm_range: tuple[float, float]


@dataclass(frozen=True)
class StartUp:
    speed: Curve  # rpm against time
    x: Curve  # the machine's displacement
    floor: Curve  # the force on the floor
    rigid: Curve  # what a rigid mount would pass
    resonance_rpm: float
    t_end: float
    note: str


@dataclass(frozen=True)
class Chart:
    levels: tuple[Curve, ...]  # the sag each of CHART_LEVELS needs against the speed (rpm)
    now: Curve  # this machine
    rpm_range: tuple[float, float]
    note: str


def mount_info(iso: Isolator) -> str:
    fn = iso.natural_hz
    return (
        f"k = {iso.stiffness:.4g} N/m, c = {iso.damping:.4g} N·s/m, f<sub>n</sub> = {fn:.4g} Hz "
        f"({60 * fn:.0f} rpm); running at r = {iso.speed_hz / fn:.3g}."
    )


def transmissibility(iso: Isolator) -> Transmissibility:
    fn = iso.natural_hz
    family = []
    for z in FAMILY:
        T = replace(iso, zeta=z, mount=Mount.VISCOUS, block=0.0).transmissibility(RATIOS * fn)
        family.append(Curve(RATIOS, np.where(np.isfinite(T), T, np.nan)))
    f = RATIOS * fn
    f = f[np.abs(f / fn - 1) > 1e-9] if iso.zeta == 0 else f
    X, Ff, Tn = iso.at_speed()
    r = iso.speed_hz / fn
    F = float(iso.force_at(iso.speed_hz))
    verdict = (
        f"isolation {100 * (1 - Tn):.3g} % ({db(Tn):.3g} dB)"
        if Tn < 1
        else f"amplified {Tn:.3g} times ({db(Tn):+.3g} dB)"
    )
    note = (
        f"At {iso.rpm:.0f} rpm the unbalance pulls with {fmt(F, 'N')}; {fmt(Ff, 'N')} reaches the floor: T = "
        f"{Tn:.3g}, {verdict}. The machine moves ±{fmt(X, 'm')}."
    )
    rpm = np.geomspace(max(iso.rpm / 50.0, 1.0), iso.rpm * 3.0, 800)
    fr = rpm / 60.0
    if iso.zeta == 0:
        fr = fr[np.abs(fr / fn - 1) > 1e-6]
        rpm = fr * 60.0
    force = iso.force_at(fr)
    return Transmissibility(
        family=tuple(family),
        isolator=Curve(f / fn, iso.transmissibility(f)),
        now=Curve.point(r, Tn) if RATIOS[0] <= r <= RATIOS[-1] else Curve.empty(),
        note=note,
        rigid=Curve(rpm, force),
        floor=Curve(rpm, iso.transmissibility(fr) * force),
        force_now=Curve.point(iso.rpm, Ff),
        rpm_range=(float(rpm[0]), float(rpm[-1])),
    )


def design_note(iso: Isolator, target: float) -> str:
    """The sag undamped one-stage mounts need to isolate `target` (a fraction) at the machine's speed."""
    try:
        d = deflection_for(iso.speed_hz, target)
    except ValueError:
        return ""
    return (
        f"{100 * target:.4g} % at {iso.rpm:.0f} rpm needs f<sub>n</sub> ≤ {math.sqrt(G / d) / TWO_PI:.3g} Hz: a sag "
        f"of {1e3 * d:.3g} mm or more (undamped, one stage)."
    )


def start_up_view(iso: Isolator, ramp: float) -> StartUp:
    """A run up to speed in `ramp` seconds, held long enough to settle (decimated to ~6000 points)."""
    hold = max(0.25 * ramp, 1.0, 6.0 / max(iso.zeta * TWO_PI * iso.natural_hz, 1e-3) / 3)
    hold = min(hold, 4.0 * ramp + 2.0)
    s = start_up(iso, ramp, hold)
    k = slice(None, None, max(1, s.t.size // 6000))
    t = s.t[k]
    during = s.t <= ramp
    peak_f = float(np.max(np.abs(s.floor[during]))) if during.any() else 0.0
    peak_x = float(np.max(np.abs(s.x[during]))) if during.any() else 0.0
    X, Ff, _ = iso.at_speed()
    return StartUp(
        speed=Curve(t, 60.0 * s.speed_hz[k]),
        x=Curve(t, s.x[k]),
        floor=Curve(t, s.floor[k]),
        rigid=Curve(t, iso.force_at(s.speed_hz[k])),
        resonance_rpm=60.0 * iso.natural_hz,
        t_end=float(s.t[-1]),
        note=(
            f"Through resonance ({60 * iso.natural_hz:.0f} rpm, dotted) the floor force peaks at {fmt(peak_f, 'N')} "
            f"and the machine at ±{fmt(peak_x, 'm')}; running, {fmt(Ff, 'N')} and ±{fmt(X, 'm')}. Dashed: the "
            "unbalance's own force."
        ),
    )


def chart(iso: Isolator) -> Chart:
    rpm = 60.0 * CHART_SPEEDS
    T = iso.at_speed()[2]
    return Chart(
        levels=tuple(
            Curve(rpm, np.array([1e3 * deflection_for(f, level) for f in CHART_SPEEDS])) for level in CHART_LEVELS
        ),
        now=Curve.point(iso.rpm, 1e3 * iso.deflection),
        rpm_range=(float(rpm[0]), float(rpm[-1])),
        note=(
            f"This machine: {iso.rpm:.0f} rpm on {1e3 * iso.deflection:.3g} mm mounts, "
            + (f"isolating {100 * (1 - T):.3g} %" if T < 1 else f"amplified {T:.3g} times")
            + " with its damping and kind. "
            "Above a line, better isolated than it."
        ),
    )
