"""The machine on its mounts live (#478 Phase 4, Vibration isolation): stepped as it runs, changed as it runs.

The run steps `core/isolation.live_system` exactly (`foh.FohSteps`: the inputs linear between samples, 100 or more
steps a turn of the running speed and 40 a period of the mounts' fastest mode), driven by one of two things (`shaking`):

- `MACHINE`: the unbalance, m<sub>e</sub>e (Ω² cos φ + Ω' sin φ) along the mounts, φ the angle it has turned
  through (from the top), so a speed changing adds its tangential part;
- `FLOOR`: the floor under the mounts moving g = Y cos φ at the same speed (`FLOOR_AMPLITUDE`), the unbalance
  stopped. Switched to the floor, φ starts where g is zero, so the floor starts from its place, moving.

What the reader does, and what each does to the run (the interaction logic, here rather than in Qt handlers):

- `set_isolator(iso)`: a box or a grip changed the mounts, the machine or the kind, a live change by
  `core/isolation.carry`'s rules (the machine keeps its place; a heavier one captures its extra at rest).
- `set_speed(rpm, seconds)`: the speed goes to `rpm` at once (`seconds` 0: the phase runs on, no jump) or ramps
  there over `seconds` at a steady rate (the Run up / Run down buttons): `ramping` while it does.
- `set_shaking(which)`: the machine's unbalance or the floor.
- `hold(x)` then `let_go()`: a hand holds the machine still at x, the rest settled round it (`isolation.held`),
  the clock stopped; let go, it runs on from there: a pluck.
- `advance(duration)`: step on (nothing while held), the samples into `record`. The part of a step left over is
  carried to the next (`StepClock`): it's a `stepping.Stepper`.

`changed_at` is when the present steady state could start: the last change, let-go or switch, or the end of a
ramp. A dwell measures only after it (`dwell`). `edits` keeps the changes' times for the time plots' marks: a
grip's run of changes is one mark, at its last.

The record (`COLUMNS`) keeps each sample's machine x and v, the intermediate mass's x, the floor's g, the unbalance's
force, the force on the floor, the speed (Hz) and the unbalance's angle φ. From it:

- `dwell(live)`: at a steady speed, the transmissibility measured as the recorder measures (the output's
  amplitude and phase over the input's at the speed, from the last `PERIODS` turns and the turns before them,
  steady when the two agree; each a sine fitted by least squares, as the steps needn't make whole turns): the floor's force over the
  unbalance's, or the machine's motion over the floor's. Both are T, and both land on one plot. It lands only once
  the output is one clean sine at the speed (`CLEAN`): the mounts' own ring from the last change has died away.
- `by_speed(t, speed_hz, phase, y)`: each whole turn's largest |y| against the speed (rpm), the floor force (or the
  machine's motion) built up as the speed runs up and down.
- `free_body(live, ...)`: the forces on the machine (the mount's spring and damper, or the relaxation's spring and
  arm, or the upper stage's; the unbalance's; a hand's holding it) and their sum M ẍ, with the force on the floor
  aside: it's on the floor, not the machine, and is what the page is about.
- `GRIPS`: how the drawn machine's and mounts' grips set its mass, the mounts' sag and their damping.

Working on it: Qt-free (the package's docstring); `tests/test_live_isolation.py` checks a dwell lands the
closed-form T for every kind both ways, the rules of a live change, a ramp's speed and phase, and `by_speed`.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..core.foh import FohSteps
from ..core.isolation import Isolator, LiveSystem, Mount, carry, held, live_system
from ..core.sdof import TWO_PI
from ..core.stepping import StepClock
from .free_body import ArrowScale, FreeBody, arrow
from .grips import Grip
from .record import SampleBuffer
from .recorder import PERIODS, TOLERANCE, Measurement

MACHINE, FLOOR = range(2)  # what shakes
SHAKING = ("The machine (its unbalance)", "The floor")
FLOOR_AMPLITUDE = 1e-4  # m: the floor's motion when it shakes
STEPS_PER_PERIOD = 40  # a period of the mounts' fastest mode
STEPS_PER_TURN = 100  # a turn of the speed: the drive, linear between samples, is then within 0.04 % of a sine's
MAX_STEP = 1.0 / 256  # s: the longest step, with nothing fast
MAX_STEPS = 50_000  # in one advance
CLEAN = 0.002  # a dwell's output: what's left after one sine at the speed, at most this of its size
MARK_GAP = 0.25  # s: changes closer than this (a grip's drag) share one mark
COLUMNS = ("x", "v", "stage", "g", "unbalance", "floor", "speed", "phase")
COL = {name: i for i, name in enumerate(COLUMNS)}
GRIPS = {
    "mass": Grip("mass", 1.0, 1e4, per=0.5),  # kg: the machine (its mounts keep their sag, so k grows with it)
    "deflection_mm": Grip("deflection_mm", 0.05, 100.0, per=0.4),  # mm: the mounts' sag, so k = Mg/δ
    "zeta": Grip("zeta", 0.003, 1.0, per=0.4, zero_below=0.005),  # the damping ratio; zero below 0.005
}


class LiveIsolation:
    """The machine on its mounts, stepped as it runs (see the module)."""

    def __init__(self, iso: Isolator | None = None, shaking: int = MACHINE, capacity: int = 200_000) -> None:
        self.iso = iso or Isolator()
        self.shaking = shaking
        self.record = SampleBuffer(len(COLUMNS), capacity)
        self.clock = StepClock()
        self._steps: dict[float, FohSteps] = {}
        self.reset()

    # ------------------------------------------------------------- the state
    def reset(self) -> None:
        """Back to t = 0, at rest and stopped, the speed at the isolator's own."""
        self.t = 0.0
        self.system: LiveSystem = live_system(self.iso)
        self.z = np.zeros(self.system.A.shape[0])
        self.phase = 0.0 if self.shaking == MACHINE else -math.pi / 2
        self.omega = TWO_PI * self.iso.speed_hz
        self.target = self.omega
        self.rate = 0.0  # rad/s² toward the target (0: there)
        self.held_at: float | None = None
        self.changed_at = 0.0
        self.edits: list[float] = []
        self.record.clear()
        self.clock.reset()
        self._steps = {}

    @property
    def x(self) -> float:
        return float(self.z[0])

    @property
    def v(self) -> float:
        return float(self.z[self.system.velocity])

    @property
    def speed_hz(self) -> float:
        return self.omega / TWO_PI

    @property
    def ramping(self) -> bool:
        return self.rate > 0 and self.omega != self.target

    def inputs(self) -> np.ndarray:
        """u = [f, g, ġ] now."""
        return self._inputs(np.array([self.omega]), np.array([self.phase]), np.array([self._accel()]))[0]

    def _accel(self) -> float:
        return math.copysign(self.rate, self.target - self.omega) if self.ramping else 0.0

    def _inputs(self, omega: np.ndarray, phase: np.ndarray, accel: np.ndarray) -> np.ndarray:
        u = np.zeros((omega.size, 3))
        if self.shaking == MACHINE:
            u[:, 0] = self.iso.unbalance * (omega * omega * np.cos(phase) + accel * np.sin(phase))
        else:
            u[:, 1] = FLOOR_AMPLITUDE * np.cos(phase)
            u[:, 2] = -FLOOR_AMPLITUDE * omega * np.sin(phase)
        return u

    def floor_force(self) -> float:
        """The force the mounts put on the floor now (N, + along x)."""
        return float(self.system.D @ self.z + self.system.E @ self.inputs())

    def _changed(self, at: float | None = None) -> None:
        self.changed_at = max(self.t if at is None else at, self.changed_at)
        if self.edits and self.t - self.edits[-1] < MARK_GAP:
            self.edits[-1] = self.t
        else:
            self.edits.append(self.t)

    # ------------------------------------------------------------- what the reader does
    def set_isolator(self, iso: Isolator) -> None:
        """A live change of the mounts, the machine or the kind (see the module); the speed is `set_speed`'s."""
        iso = replace(iso, rpm=self.iso.rpm)  # the speed is set_speed's
        if iso == self.iso:
            return
        g = float(self.inputs()[1])
        self.z = carry(self.iso, iso, self.z, g) if self.held_at is None else held(iso, self.held_at, g)
        self.iso = iso
        self.system = live_system(iso)
        self._steps = {}
        self._changed()

    def set_speed(self, rpm: float, seconds: float = 0.0) -> None:
        """To `rpm` at once (`seconds` 0) or at a steady rate over `seconds` (see the module)."""
        target = TWO_PI * max(rpm, 0.0) / 60.0
        self.target = target
        self.iso = replace(self.iso, rpm=max(rpm, 0.0))
        if seconds <= 0 or target == self.omega:
            self.omega, self.rate = target, 0.0
            self._changed()
        else:
            self.rate = abs(target - self.omega) / seconds
            self._changed(self.t + seconds)
        self._steps = {}

    def set_shaking(self, which: int) -> None:
        if which == self.shaking:
            return
        self.shaking = which
        if which == FLOOR:
            self.phase = -math.pi / 2  # the floor starts from its place: g = Y cos φ = 0
        for steps in self._steps.values():
            steps.f_prev = self.inputs()
        self._changed()

    def hold(self, x: float) -> None:
        """A hand holds the machine still at x, the rest settled round it; the clock stops."""
        self.held_at = x
        self.z = held(self.iso, x, float(self.inputs()[1]))

    def let_go(self) -> None:
        if self.held_at is None:
            return
        self.held_at = None
        self._changed()

    # ------------------------------------------------------------- stepping
    def step_size(self) -> float:
        """STEPS_PER_PERIOD a period of the mounts' fastest mode and STEPS_PER_TURN a turn of the speed (the
        faster of now and where it's going), at most MAX_STEP, rounded down to a power of two (so a few step sizes
        are discretized, and kept)."""
        modes = float(np.max(np.abs(np.linalg.eigvals(self.system.A)))) / TWO_PI
        turns = max(self.omega, self.target) / TWO_PI
        h = min(MAX_STEP, 1.0 / max(STEPS_PER_PERIOD * modes, STEPS_PER_TURN * turns, 1e-9))
        return 2.0 ** math.floor(math.log2(h))

    def _stepper(self, h: float) -> FohSteps:
        steps = self._steps.get(h)
        if steps is None:
            steps = FohSteps(self.system.A, self.system.B, h)
            self._steps = {h: steps}
        steps.state = self.z
        steps.f_prev = self.inputs()
        return steps

    def advance(self, duration: float) -> int:
        """Step on by about `duration` s (nothing while held): the number of samples recorded."""
        if self.held_at is not None:
            return 0
        h = self.step_size()
        n = self.clock.steps(duration, h, MAX_STEPS)
        if n == 0:
            return 0
        tau = h * np.arange(1, n + 1)
        omega, phase, accel = self._run(tau)
        u = self._inputs(omega, phase, accel)
        z = self._stepper(h).run(u)
        times = self.t + tau
        self.t = float(times[-1])
        self.z = z[-1].copy()
        self.omega, self.phase = float(omega[-1]), float(phase[-1])
        if not self.ramping:
            self.rate = 0.0
        floor = z @ self.system.D + u @ self.system.E
        stage = z[:, 1] if self.system.velocity == 2 else np.zeros(n)
        cols = np.column_stack(
            [z[:, 0], z[:, self.system.velocity], stage, u[:, 1], u[:, 0], floor, omega / TWO_PI, phase]
        )
        self.record.extend(times, cols)
        return n

    def _run(self, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The speed, the unbalance's angle and the speed's rate of change at `tau` s from now."""
        w0, phi0 = self.omega, self.phase
        if not self.ramping:
            return np.full(tau.size, w0), phi0 + w0 * tau, np.zeros(tau.size)
        a = self._accel()
        until = abs(self.target - w0) / self.rate  # s to the target
        ramp = np.minimum(tau, until)
        omega = np.where(tau >= until, self.target, w0 + a * ramp)
        phase = phi0 + w0 * tau + a * (0.5 * ramp * ramp + ramp * (tau - ramp))
        return omega, phase, np.where(tau < until, a, 0.0)


# ------------------------------------------------------------------ what's measured


def output_input(live: LiveIsolation) -> tuple[str, str]:
    """The record's columns a dwell divides, output over input: the floor's force over the unbalance's, or the
    machine's motion over the floor's."""
    return ("floor", "unbalance") if live.shaking == MACHINE else ("x", "g")


def dwell(live: LiveIsolation) -> Measurement | None:
    """At a steady speed, T measured from the run since `changed_at` (see the module), or None (ramping, held, or
    too little run yet)."""
    if live.ramping or live.held_at is not None or live.speed_hz <= 0:
        return None
    n = live.record.size
    t, y = live.record.t[:n], live.record.y[:n]
    out, inp = (y[:, COL[c]] for c in output_input(live))
    f = live.speed_hz
    span = PERIODS / f
    if n < 2 or t[-1] - 2 * span < max(live.changed_at, float(t[0])):
        return None
    estimates, clean = [], True
    for end in (float(t[-1]), float(t[-1]) - span):
        i0, i1 = np.searchsorted(t, [end - span, end], side="right")
        if i1 - i0 < 8 * PERIODS:
            return None
        (a, rest_a), (b, _) = (_sine(t[i0:i1], v[i0:i1], f) for v in (out, inp))
        if b == 0:
            return None
        estimates.append(a / b)
        clean = clean and rest_a <= CLEAN * abs(a) / math.sqrt(2)
    now, before = estimates
    return Measurement(f, now, bool(clean and abs(now - before) <= TOLERANCE * abs(now)))


def _sine(t: np.ndarray, y: np.ndarray, freq_hz: float) -> tuple[complex, float]:
    """y's sine at `freq_hz` as a phasor c (y ≈ Re(c e^(iωt)) and a constant), fitted by least squares, and the
    rms of what's left. A fit, not a Fourier sum: the samples needn't make whole turns. What's left shows the
    mounts' own ring not yet died away: the two windows' estimates can agree while it repeats nearly the same in
    each (at 3.5 f<sub>n</sub>, three turns are nearly one of the ring's periods)."""
    w = TWO_PI * freq_hz
    basis = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
    rest = y - basis @ coef
    return complex(coef[0], -coef[1]), float(np.sqrt(np.mean(rest * rest)))


def by_speed(t: np.ndarray, speed_hz: np.ndarray, phase: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each whole turn's largest |y| against its mean speed (rpm): (rpm, peak), turns in order (see the module)."""
    if t.size < 2:
        return np.empty(0), np.empty(0)
    turns = np.floor(np.unwrap(phase) / TWO_PI)
    starts = np.flatnonzero(np.diff(turns) != 0) + 1
    if starts.size < 2:
        return np.empty(0), np.empty(0)
    edges = starts  # whole turns only: from the first new turn to the last one's end
    peak = np.maximum.reduceat(np.abs(y[edges[0] : edges[-1]]), edges[:-1] - edges[0])
    rpm = 60.0 * np.add.reduceat(speed_hz[edges[0] : edges[-1]], edges[:-1] - edges[0]) / np.diff(edges)
    return rpm, peak


# ------------------------------------------------------------------ the free-body diagram


def free_body(live: LiveIsolation, scale: ArrowScale | None = None, floor_scale: ArrowScale | None = None) -> FreeBody:
    """The forces on the machine now, + up (see the module), with the force on the floor aside."""
    iso, z, t = live.iso, live.z, live.t
    _, g, g_dot = (float(u) for u in live.inputs())
    x, v = live.x, live.v
    k, c = iso.stiffness, iso.damping
    rows = []
    if iso.mount is Mount.TWO_STAGE:  # the upper mounts, on the intermediate mass
        x2, v2 = float(z[1]), float(z[3])
        rows.append(arrow("spring", "k(x₁ − x₂)", -k * (x - x2), "energy.potential", short="kx"))
        rows.append(arrow("damper", "c(ẋ₁ − ẋ₂)", -c * (v - v2), "energy.dissipated", short="cẋ"))
    else:
        rows.append(arrow("spring", "k(x − g)" if live.shaking else "kx", -k * (x - g), "energy.potential", short="kx"))
        if iso.mount is Mount.RELAXATION:
            k1 = iso.stiffness_ratio * k
            arm = -k1 * (x - float(z[2])) if c > 0 else 0.0
            rows.append(arrow("arm", "Nk(x − y)", arm, "energy.added", short="arm"))
        else:
            rows.append(
                arrow("damper", "c(ẋ − ġ)" if live.shaking else "cẋ", -c * (v - g_dot), "energy.dissipated", short="cẋ")
            )
    force = float(live.inputs()[0])
    rows.append(arrow("unbalance", "f", force, "force"))
    if live.held_at is not None:
        rows.append(arrow("hand", "hand", -math.fsum(r.value for r in rows), "muted"))
    total = math.fsum(r.value for r in rows)
    net = arrow("net", "Mẍ", total, "strong")
    largest = max(abs(r.value) for r in (*rows, net))
    s = scale.update(t, largest) if scale is not None else (1.0 / largest if largest > 0 else 0.0)
    on_floor = live.floor_force()
    floor = arrow("floor", "on the floor", on_floor, "structure", short="floor")
    a = floor_scale.update(t, abs(on_floor)) if floor_scale is not None else (1.0 / abs(on_floor) if on_floor else 0.0)
    return FreeBody(tuple(rows), net, s, (floor,), a)
