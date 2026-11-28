"""One mass forced live (#478 Phase 4, Forcing an oscillator): hit, pushed, pulsed or driven by a periodic wave as it
runs, changed as it runs.

The run steps the oscillator exactly (`foh.FohSteps` on `Oscillator.matrices`, the force linear between samples,
`STEPS` a period of the oscillator, of the wave and of the shortest pulse acting), driven by the forces the reader
applies. Each acts from the moment it is applied, on whatever the mass is doing, so a second hit lands on the
first's motion: the response is the sum of each force's own (superposition, seen rather than told).

- `apply(pulse)`: a one-off force from now (`core/transient.Pulse`): a hit (an impulse, a jump in momentum I/m,
  applied at a step's boundary), a push (a step) or a ramp held until `release()`, a rectangular or half-sine
  pulse that ends by itself. Each is a `Push` in `pushes`, with its start (and end, once released).
- `set_wave(forcing)`: a periodic force (`core/periodic.Forcing`) from now, its period counted from the moment it
  starts (None stops it); a hammer-blow train is a hit each period.
- `set_params(osc)`: m, k or c changed as it runs, by `Simulator.set_system`'s rules (a heavier mass captures its
  extra at rest, keeping the momentum; a lighter one drops a piece at its speed; a spring keeps x and v).
- `hold(x)` then `let_go()`: a hand holds the mass still at x (the clock stops); let go, it runs on: a pluck.
- `advance(duration)`: step on (nothing while held), the samples into `record` (x, v and the force, impulses
  aside, which are kept as `hits`, their times and sizes). It's a `stepping.Stepper`.

`changed_at` is when the present steady state could start (the last change of the system or the wave, a let-go,
a push or its release); `edits` keeps the changes' times for the time plots' marks.

What is measured from the run (the tabs' raw material):

- `pulse_peak(live, push)`: a pulse's (or a hit's, or a step's) largest response over its static deflection,
  F/k (an impulse's over I ω<sub>n</sub>/k), once a period has passed after it ends: the Pulse spectrum's point at its
  length over the natural period. None if the mass wasn't nearly at rest when it was applied, or another force
  acted meanwhile (superposition would muddle it).
- `harmonics(live, count)`: the wave's and the response's line spectra, each harmonic's amplitude and phase,
  from the last `PERIODS` whole periods by least squares, steady when the periods before them agree (the start's
  free vibration has died away).
- `convolution(live, at)`: Duhamel's integral at the instant `at`, from the record since the last change: the
  force, the impulse response flipped and slid there, h(at − τ), their product, and its area (with each hit's
  I h(at − tᵢ)), which is the motion when it started from rest.

Working on it: Qt-free (the package's docstring); `tests/test_live_forcing.py` checks each force against its closed
form, superposition, a live change, a pulse's peak against `transient.peak_ratio`, the harmonics against
`periodic.response_coefficients` and the convolution against the record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.foh import FohSteps
from ..core.periodic import Forcing, Wave
from ..core.sdof import TWO_PI, Oscillator, impulse_response
from ..core.stepping import StepClock
from ..core.transient import Pulse, PulseKind
from .record import SampleBuffer

STEPS = 100  # steps a period of the oscillator
EDGE_STEPS = 400  # of the wave and of the shortest pulse acting: a jump, smeared over a step, is within ~0.2 %
MAX_STEP = 1.0 / 512  # s
MAX_STEPS = 100_000  # in one advance
PERIODS = 3  # whole periods in each estimate of the harmonics
TOLERANCE = 0.005  # the two estimates' relative difference that counts as steady
QUIET = 0.01  # a pulse lands its peak only if the mass was within this of the pulse's own response, at rest
MARK_GAP = 0.25  # s: changes closer than this share one mark
COLUMNS = ("x", "v", "f")
COL = {name: i for i, name in enumerate(COLUMNS)}


@dataclass
class Push:
    """A one-off force applied at `start` (s); a step or a ramp lasts until released (`end`)."""

    pulse: Pulse
    start: float
    end: float = math.inf  # a step's or ramp's release; a pulse ends by itself
    quiet: bool = True  # the mass was at rest when it was applied
    alone: bool = True  # no other force acted while it did

    @property
    def finish(self) -> float:
        """When its force stops changing the motion's source: the pulse's end, or the release."""
        p = self.pulse
        if p.kind in (PulseKind.RECTANGLE, PulseKind.HALF_SINE):
            return self.start + p.duration
        if p.kind is PulseKind.IMPULSE:
            return self.start
        return self.end

    def force(self, t: np.ndarray) -> np.ndarray:
        f = self.pulse.force(t - self.start) * (t >= self.start)
        return np.where(t < self.end, f, 0.0) if self.end < math.inf else f


@dataclass
class LiveForcing:
    """One mass, forced as it runs (see the module)."""

    osc: Oscillator = field(default_factory=Oscillator)
    capacity: int = 400_000

    def __post_init__(self) -> None:
        self.record = SampleBuffer(len(COLUMNS), self.capacity)
        self.clock = StepClock()
        self.reset()

    def reset(self) -> None:
        """Back to t = 0 at rest, no force acting."""
        self.t = 0.0
        self.z = np.zeros(2)
        self.pushes: list[Push] = []
        self.hits: list[tuple[float, float]] = []  # (t, I): the impulses, applied as jumps in momentum
        self.wave: Forcing | None = None
        self.wave_start = 0.0
        self._next_blow = math.inf  # a hammer train's next blow
        self.held_at: float | None = None
        self.since = 0.0  # the last reset, let-go or change of m, k or c: where Duhamel's integral starts
        self.changed_at = 0.0
        self.edits: list[float] = []
        self.record.clear()
        self.clock.reset()
        self._steps: dict[float, FohSteps] = {}

    @property
    def x(self) -> float:
        return float(self.z[0])

    @property
    def v(self) -> float:
        return float(self.z[1])

    def _changed(self) -> None:
        self.changed_at = self.t
        if self.edits and self.t - self.edits[-1] < MARK_GAP:
            self.edits[-1] = self.t
        else:
            self.edits.append(self.t)

    def force_at(self, t: np.ndarray) -> np.ndarray:
        """Every force acting at the times t but the impulses (N)."""
        t = np.asarray(t, dtype=float)
        f = np.zeros_like(t)
        for p in self.pushes:
            if p.pulse.kind is not PulseKind.IMPULSE:
                f += p.force(t)
        if self.wave is not None and self.wave.wave is not Wave.IMPULSES:
            f += np.where(t >= self.wave_start, self.wave.force(t - self.wave_start), 0.0)
        return f

    def force_now(self) -> float:
        """The force from now on (a push applied now included): what the next step starts from."""
        return float(self.force_at(np.array([self.t + 1e-6 * self.step_size()]))[0])

    def _sampled(self, t: np.ndarray, h: float) -> np.ndarray:
        """The force as the steps take it: at an edge that falls on a sample (a square wave's, a pulse's), the
        mean of its two sides. Taken linear between samples, a jump is centered between the samples either side
        of it; an edge on a sample, given its new value, would be centered half a step early, and a wave whose
        edges fall on samples now and then ran a sixth of a step ahead (2.5 mrad on a square wave's response)."""
        eps = 1e-6 * h
        return 0.5 * (self.force_at(t - eps) + self.force_at(t + eps))

    # ------------------------------------------------------------- what the reader does
    def _acting(self) -> bool:
        """Is any force acting now, or the motion still far from rest?"""
        return any(p.finish > self.t for p in self.pushes) or self.wave is not None

    def _at_rest(self, pulse: Pulse) -> bool:
        k, w = self.osc.stiffness, self.osc.omega_n
        scale = abs(pulse.amplitude) * (w if pulse.kind is PulseKind.IMPULSE else 1.0) / k  # its own response's size
        return math.hypot(self.x, self.v / w) <= QUIET * scale and not self._acting()

    def apply(self, pulse: Pulse) -> Push:
        """A one-off force from now (see the module)."""
        push = Push(pulse, self.t, quiet=self._at_rest(pulse), alone=not self._acting())
        for p in self.pushes:
            if p.finish > self.t:
                p.alone = False
        if self.wave is not None:
            push.alone = False
        self.pushes.append(push)
        if pulse.kind is PulseKind.IMPULSE:
            self._hit(pulse.amplitude)
        self._steps = {}
        self._changed()
        return push

    def release(self) -> None:
        """Let go of a step or a ramp being held."""
        for p in self.pushes:
            if p.end == math.inf and p.pulse.kind in (PulseKind.STEP, PulseKind.RAMP):
                p.end = self.t
        self._steps = {}
        self._changed()

    def holding(self) -> bool:
        """A step or a ramp is being held."""
        return any(p.end == math.inf and p.pulse.kind in (PulseKind.STEP, PulseKind.RAMP) for p in self.pushes)

    def set_wave(self, forcing: Forcing | None) -> None:
        """A periodic force from now, its period counted from now (None stops it)."""
        if forcing == self.wave:
            return
        if forcing is not None and self.wave is not None and forcing.wave is self.wave.wave:
            # the same wave, its size or frequency changed: it runs on from where it is in its period
            phase = ((self.t - self.wave_start) / self.wave.period) % 1.0
            self.wave_start = self.t - phase * forcing.period
        else:
            self.wave_start = self.t
        for p in self.pushes:
            if p.finish > self.t:
                p.alone = False
        self.wave = forcing
        self._next_blow = (
            self.wave_start + math.ceil((self.t - self.wave_start) / forcing.period - 1e-9) * forcing.period
            if forcing is not None and forcing.wave is Wave.IMPULSES
            else math.inf
        )
        self._steps = {}
        self._changed()

    def set_params(self, osc: Oscillator) -> None:
        """m, k or c changed as it runs (see the module)."""
        if osc == self.osc:
            return
        if osc.mass > self.osc.mass and self.held_at is None:
            self.z[1] *= self.osc.mass / osc.mass  # captured at rest: momentum kept
        self.osc = osc
        self._steps = {}
        self.since = self.t
        self._changed()

    def hold(self, x: float) -> None:
        self.held_at = x
        self.z = np.array([x, 0.0])

    def let_go(self) -> None:
        if self.held_at is None:
            return
        self.held_at = None
        self.since = self.t
        self._changed()

    def _hit(self, impulse: float) -> None:
        self.z[1] += impulse / self.osc.mass
        self.hits.append((self.t, impulse))

    # ------------------------------------------------------------- stepping
    def step_size(self) -> float:
        """STEPS a period of the oscillator, EDGE_STEPS of the wave (its harmonics need it) and of the shortest pulse
        acting; at most MAX_STEP; a power of two, so few sizes are discretized."""
        shortest = [1.0 / self.osc.natural_hz]
        if self.wave is not None:
            shortest.append(self.wave.period * (self.wave.duty if self.wave.wave is Wave.PULSES else 1.0))
        for p in self.pushes:
            if p.finish > self.t and p.pulse.has_duration:
                shortest.append(p.pulse.duration)
        h = min(MAX_STEP, shortest[0] / STEPS, min(shortest[1:], default=math.inf) / EDGE_STEPS)
        return 2.0 ** math.floor(math.log2(h))

    def _stepper(self, h: float) -> FohSteps:
        steps = self._steps.get(h)
        if steps is None:  # new, or something changed the force or the system: it starts from the force now
            A, B = self.osc.matrices()
            steps = FohSteps(A, B, h)
            steps.f_prev = np.array([self.force_now()])
            self._steps = {h: steps}
        # (else it carries on from the last run's last sample, as one run: an edge on the boundary between two
        # frames is taken as any other, not restarted from its new value)
        steps.state = self.z
        return steps

    def advance(self, duration: float) -> int:
        """Step on by about `duration` s (nothing while held): the number of samples recorded."""
        if self.held_at is not None:
            return 0
        total = 0
        h = self.step_size()
        n = self.clock.steps(duration, h, MAX_STEPS)
        while n > 0:
            m = n
            if self._next_blow < math.inf:  # up to the next hammer blow's step, where it lands
                m = min(n, max(round((self._next_blow - self.t) / h), 0))
                if m == 0:
                    assert self.wave is not None, "a blow belongs to a hammer train"
                    self._hit(self.wave.amplitude)
                    self._next_blow += self.wave.period
                    continue
            ts = self.t + h * np.arange(1, m + 1)
            f = self._sampled(ts, h)
            z = self._stepper(h).run(f[:, None])
            self.t = float(ts[-1])
            self.z = z[-1].copy()
            self.record.extend(ts, np.column_stack([z, f]))
            total += m
            n -= m
        return total

    def since_change(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.record.size
        t = self.record.t[:n]
        i = int(np.searchsorted(t, self.changed_at))
        return t[i:], self.record.y[i:n]


# ------------------------------------------------------------------ what's measured


@dataclass(frozen=True)
class PulsePeak:
    kind: PulseKind
    ratio: float  # its length over the natural period (0 for a hit or a step)
    peak: float  # its largest response over its static deflection F/k (a hit's over Iω_n/k)
    zeta: float


def pulse_peak(live: LiveForcing, push: Push) -> PulsePeak | None:
    """The push's peak response (see the module), or None: not yet a period after it ended, or the mass not at rest
    when it was applied, or another force acting meanwhile."""
    osc = live.osc
    p = push.pulse
    if not (push.quiet and push.alone) or live.wave is not None:
        return None
    held = p.kind in (PulseKind.STEP, PulseKind.RAMP)
    end = push.start + (p.duration if p.kind is PulseKind.RAMP else 0.0) if held else push.finish
    if live.t < end + 1.05 * osc.period or (held and push.end < end + 1.05 * osc.period):
        return None  # not yet a period on; or a step or ramp let go before its peak
    n = live.record.size
    t, x = live.record.t[:n], live.record.y[:n, COL["x"]]
    i0, i1 = np.searchsorted(t, [push.start, end + 1.05 * osc.period])
    if i1 - i0 < 4:
        return None
    k, w = osc.stiffness, osc.omega_n
    scale = abs(p.amplitude) * (w if p.kind is PulseKind.IMPULSE else 1.0) / k
    ratio = p.duration / osc.period if p.has_duration else 0.0
    return PulsePeak(p.kind, ratio, float(np.max(np.abs(x[i0:i1]))) / scale, osc.zeta)


@dataclass(frozen=True)
class Harmonics:
    freq_hz: float
    force: np.ndarray  # each harmonic's complex amplitude, 0 (the mean) to count
    response: np.ndarray
    steady: bool


def _fit(t: np.ndarray, y: np.ndarray, freq_hz: float, count: int) -> np.ndarray:
    """y's harmonics 0…count at freq_hz by least squares: C_k with y ≈ Re Σ C_k e^(ikωt)."""
    w = TWO_PI * freq_hz * t
    cols = [np.ones_like(t)]
    for k in range(1, count + 1):
        cols += [np.cos(k * w), np.sin(k * w)]
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)
    out = np.empty(count + 1, dtype=complex)
    out[0] = coef[0]
    out[1:] = coef[1::2] - 1j * coef[2::2]
    return out


def harmonics(live: LiveForcing, count: int) -> Harmonics | None:
    """The wave's and the response's harmonics from the steady run (see the module), or None: no wave, or too
    little run since it changed."""
    wave = live.wave
    if wave is None or live.held_at is not None:
        return None
    T = wave.period
    span = PERIODS * T
    n = live.record.size
    t, y = live.record.t[:n], live.record.y[:n]
    if n < 2 or t[-1] - 2 * span < max(live.changed_at, float(t[0])):
        return None
    shifted = t - live.wave_start  # the wave's own time, so its harmonics' phases are the series'
    estimates = []
    for end in (float(t[-1]), float(t[-1]) - span):
        i0, i1 = np.searchsorted(t, [end - span, end], side="right")
        if i1 - i0 < 8 * PERIODS * (count + 1):
            return None
        estimates.append((_fit(shifted[i0:i1], y[i0:i1, COL["f"]], wave.freq_hz, count),
                          _fit(shifted[i0:i1], y[i0:i1, COL["x"]], wave.freq_hz, count)))  # fmt: skip
    (fn, xn), (_, xb) = estimates
    if wave.wave is Wave.IMPULSES:  # the blows aren't in the force column: their series is the train's own
        fn = wave.coefficients(count)
    size = float(np.max(np.abs(xn))) or 1.0
    steady = float(np.max(np.abs(xn - xb))) <= TOLERANCE * size
    return Harmonics(wave.freq_hz, fn, xn, bool(steady))


@dataclass(frozen=True)
class Convolution:
    tau: np.ndarray  # the times before `at` (s)
    force: np.ndarray  # F(τ)
    h: np.ndarray  # h(at − τ), the impulse response flipped and slid to `at`
    product: np.ndarray  # F(τ) h(at − τ)
    hits: tuple[tuple[float, float], ...]  # the impulses (t, I) before `at`, each adding I h(at − t)
    area: float  # ∫ F h dτ + Σ I h: the motion at `at`, from the forces alone
    x: float  # the motion the run recorded there
    since: float  # the integral's start: the last change (a reset, a let-go, a change of the system)


def convolution(live: LiveForcing, at: float | None = None) -> Convolution | None:
    """Duhamel's integral at `at` (now if None) from the record since the last change of the system (see the
    module); None with too little record."""
    osc = live.osc
    n = live.record.size
    t, y = live.record.t[:n], live.record.y[:n]
    since = live.since
    at = live.t if at is None else at
    i0, i1 = np.searchsorted(t, [since, at], side="left")
    i1 = min(i1 + 1, n)
    if i1 - i0 < 2:
        return None
    tau, f = t[i0:i1], y[i0:i1, COL["f"]]
    h = impulse_response(osc, at - tau)
    product = f * h
    hits = tuple((th, imp) for th, imp in live.hits if since <= th <= at)
    area = float(np.trapezoid(product, tau)) + sum(
        imp * float(impulse_response(osc, np.array([at - th]))[0]) for th, imp in hits
    )
    x = float(np.interp(at, t[:n], y[:n, COL["x"]]))
    return Convolution(tau, f, h, product, hits, area, x, since)
