"""Duffing & Mathieu's view model: what the page draws and prints, from its oscillator, drive and run (`gui/duffing.py`
binds it).

The page runs: `core.duffing.DuffingSimulator` steps it, and a `Record` keeps what the run has measured (the last
samples, each drive cycle's harmonics, the Poincaré section). Each panel has a function from those to what it shows:

- `response(osc, x_range)`: harmonic balance's curves for the drive and the pump, the linear spring, the backbone,
  the folds and their jumps, the y range, the axis label and which legend entries show (`LEGEND` keys).
- `points(osc, cycles, y_top, runaway)`: each drive cycle's measured amplitudes by sweep leg, the latest, and a
  taller y range if they outgrew it.
- `live(trace, t, freq_hz)`: the strip chart (min/max per bucket) and the phase plane, every frame.
- `chart(zeta)`, `chart_point(osc, freq_hz)`: the Ince–Strutt chart's growth image and harmonic balance's tongue,
  and the page's pump on it.
- `osc_info`, `drive_state`, `branches`, `readout`: the texts.

Working on it: keep it free of Qt (see the package's docstring), and `live` cheap: it runs every frame, so it takes
views of the trace and allocates only the decimated line. `tests/test_view_models.py` checks the texts against
runs of the simulator, without the page.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.duffing import (
    Cycle,
    Drive,
    Duffing,
    ResponseCurve,
    Samples,
    Sweep,
    amplitudes,
    floquet_growth,
    linear_response,
    measured_jump,
    parametric_amplitudes,
    parametric_curve,
    response_curve,
    tongue_hz,
)
from .plot import Curve
from .text import fmt, number

WINDOW = 20.0  # s of simulated time on the strip chart
PLOT_POINTS = 1200  # the strip chart's buckets (min and max of each)
CAPACITY = 1 << 18  # samples kept for the strip chart and the phase plane
PHASE_CYCLES = 3  # drive cycles of trajectory in the phase plane
SECTION_POINTS = 4000  # Poincaré dots kept
HELD_POINTS = 200  # the held frequency's latest cycles drawn (a sweep's are all drawn)
CHART_RATIOS = np.linspace(0.3, 4.0, 186)  # Omega / omega_n across the Ince–Strutt chart
CHART_DEPTHS = np.linspace(0.0, 1.0, 61)  # mu up it
CHART_STEPS = 300  # RK4 steps per pump period for its Floquet multipliers
GROWTH_TOP = 0.15  # sigma / omega_n where the chart's colour saturates
JUMP_FLOOR = 0.05  # of the curve's peak: a swing smaller than this is rest, to the jump detector
LEGS = ("up", "down", "hold")

# The response plot's legend entries, by key: the page maps each to its item.
LEGEND = {
    "linear": "Linear spring (k₃ = 0)",
    "backbone": "Backbone",
    "backbone_2x": "Backbone, at 2× its frequency",
    "stable": "Harmonic balance",
    "unstable": "… unstable",
    "p_stable": "Pumped, ½X",
    "p_unstable": "… unstable",
    "up": "Sweeping up",
    "down": "Sweeping down",
    "hold": "Held",
    "half_up": "½X, up",
    "half_down": "½X, down",
    "half_hold": "½X, held",
}


@dataclass(frozen=True)
class DuffingSetup:
    """The page's settings: the oscillator, the drive and the sweep, as a preset sets them, and the tab (a practice
    problem's setup, and what `practice_state()` returns)."""

    mass: float = 1.0
    stiffness: float = 400.0
    damping: float = 1.0
    cubic: float = 1e5
    force: float = 2.0
    pump: float = 0.0
    noise: float = 0.0
    freq_hz: float = 2.0
    start_hz: float = 2.0
    end_hz: float = 6.0
    sweep_time: float = 120.0
    mode: Sweep = Sweep.UP_DOWN
    tab: int = 0

    def oscillator(self) -> Duffing:
        return Duffing(self.mass, self.stiffness, self.damping, self.cubic, self.force, self.pump, self.noise)


def sig3(value: float) -> str:
    """Three significant figures, trailing zeros kept as the notes print them: 24.0, 8.80, 5.23, 112."""
    return number(value, 3, zeros=True)


def fmt_length(a: float) -> str:
    """A length in mm below a metre, in km from one, else in m (three significant figures, zeros kept)."""
    return fmt(a, "m", largest="k", smallest="m", zeros=True)


def pumped_only(osc: Duffing) -> bool:
    """Pumped with no drive force: the steady states are the pump's half-harmonic swings."""
    return osc.force == 0 and osc.pump > 0


# ------------------------------------------------------------------ the run's record


class Trace:
    """The last CAPACITY samples of t, x, v: a buffer twice as long, slid back when it fills."""

    def __init__(self) -> None:
        self.data = np.empty((3, 2 * CAPACITY))
        self.n = 0

    def clear(self) -> None:
        self.n = 0

    def add(self, t: np.ndarray, x: np.ndarray, v: np.ndarray) -> None:
        k = t.size
        if k >= CAPACITY:
            t, x, v, k = t[-CAPACITY:], x[-CAPACITY:], v[-CAPACITY:], CAPACITY
        if self.n + k > self.data.shape[1]:
            keep = CAPACITY - k
            self.data[:, :keep] = self.data[:, self.n - keep : self.n]
            self.n = keep
        self.data[:, self.n : self.n + k] = (t, x, v)
        self.n += k

    def since(self, t0: float) -> np.ndarray:
        """(3, n) of the samples from time t0 on (a view, not a copy)."""
        t = self.data[0, : self.n]
        return self.data[:, int(np.searchsorted(t, t0)) : self.n]


class Record:
    """What a run has measured: its last samples, each drive cycle since the points were last cleared, and the
    Poincaré section (the held frequency's cycles' end states)."""

    def __init__(self) -> None:
        self.trace = Trace()
        self.cycles: list[Cycle] = []
        self.section: list[tuple[float, float]] = []

    def add(self, s: Samples) -> bool:
        """Take in a stretch of the run; True if it finished a drive cycle (the points and texts change)."""
        if s.t.size:
            self.trace.add(s.t, s.x, s.v)
        if not s.cycles:
            return False
        self.cycles += s.cycles
        self.section += [(c.x, c.v) for c in s.cycles if c.leg == "hold"]  # at one frequency only
        del self.section[:-SECTION_POINTS]
        return True

    def clear_points(self) -> None:
        self.cycles = []
        self.section = []

    def clear(self) -> None:
        self.trace.clear()
        self.clear_points()


# ------------------------------------------------------------------ the response plot


@dataclass(frozen=True)
class Response:
    curve: ResponseCurve  # the drive's, by harmonic balance
    p_curve: ResponseCurve  # the pump's
    stable: Curve  # pieces joined by NaN gaps
    unstable: Curve
    p_stable: Curve
    p_unstable: Curve
    linear: Curve  # empty with no drive force
    backbone: Curve  # at twice its frequency when only pumped
    jumps: Curve  # pairs: each fold's drop or jump
    folds: Curve
    x_range: tuple[float, float]
    y_top: float
    y_label: str
    legend_right: bool  # in the top right corner (a softening curve leans left), else the top left
    legend: tuple[str, ...]  # LEGEND keys, in order


def x_range(drive: Drive) -> tuple[float, float]:
    """The frequencies shown: the sweep's range and the held frequency, padded."""
    lo = min(drive.start_hz, drive.end_hz, drive.freq_hz)
    hi = max(drive.start_hz, drive.end_hz, drive.freq_hz)
    pad = 0.1 * (hi - lo) if hi > lo else 0.5 * max(hi, 0.1)
    return max(lo - pad, 0.0), hi + pad


def _joined(parts: list[tuple[np.ndarray, np.ndarray]]) -> Curve:
    if not parts:
        return Curve.empty()
    return Curve(
        np.concatenate([np.append(p[0], np.nan) for p in parts]),
        np.concatenate([np.append(p[1], np.nan) for p in parts]),
    )


def response(osc: Duffing, span: tuple[float, float]) -> Response:
    lo, hi = span
    curve, p_curve = response_curve(osc, hi), parametric_curve(osc, hi)
    top = 0.0
    pieces: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        k: [] for k in ("stable", "unstable", "p_stable", "p_unstable")
    }
    for prefix, c in (("", curve), ("p_", p_curve)):
        for f, a, ok in c.pieces:
            pieces[prefix + ("stable" if ok else "unstable")].append((f, a))
            inside = (f >= lo) & (f <= hi) & np.isfinite(a)
            if inside.any():
                top = max(top, float(a[inside].max()))
    y_top = 1.15 * (top or 1e-3)
    f = np.linspace(lo, hi, 600)
    if osc.force > 0:
        lin = linear_response(osc, f)
        linear = Curve(f, np.where(lin <= 2 * y_top, lin, np.nan))
    else:
        linear = Curve.empty()
    a = np.linspace(0.0, 2 * y_top, 300)
    backbone = osc.backbone_hz(a)
    folds = curve.folds + p_curve.folds
    legend: list[str] = []
    if osc.force > 0:
        legend += ["linear", "backbone", "stable", "unstable"]
    elif osc.pump > 0 and osc.cubic != 0:
        legend += ["backbone_2x"]
    if osc.pump > 0:
        legend += ["p_stable", "p_unstable"]
    if osc.force > 0:
        legend += ["up", "down", "hold"]
    if osc.pump > 0:
        legend += ["half_up", "half_down", "half_hold"]
    return Response(
        curve=curve,
        p_curve=p_curve,
        **{key: _joined(parts) for key, parts in pieces.items()},
        linear=linear,
        backbone=Curve(backbone if osc.force > 0 else 2 * backbone, a),
        jumps=Curve(
            np.ravel([(j.freq_hz, j.freq_hz) for j in folds]), np.ravel([(j.amplitude, j.lands_at) for j in folds])
        ),
        folds=Curve(np.array([j.freq_hz for j in folds]), np.array([j.amplitude for j in folds])),
        x_range=(lo, hi),
        y_top=y_top,
        y_label="Amplitude (½X: half harmonic)" if pumped_only(osc) else "Amplitude (first harmonic)",
        legend_right=osc.cubic < 0,
        legend=tuple(legend),
    )


@dataclass(frozen=True)
class Points:
    first: dict[str, np.ndarray]  # by leg: (n, 2) of (Hz, m), each cycle's first harmonic (none undriven)
    half: dict[str, np.ndarray]  # by leg: its half harmonic (none unpumped)
    now: np.ndarray  # (0 or 1, 2): the latest cycle's
    section: np.ndarray  # (n, 2) of (m, m/s)
    y_top: float | None  # a taller range when the points outgrew the plot's, else None


def points(osc: Duffing, record: Record, y_top: float, runaway: float) -> Points:
    cycles = record.cycles
    none = np.empty((0, 2))
    first: dict[str, np.ndarray] = {}
    half: dict[str, np.ndarray] = {}
    top = 0.0
    for leg in LEGS:
        run = [c for c in cycles if c.leg == leg]
        if leg == "hold":
            run = run[-HELD_POINTS:]
        first[leg] = np.array([(c.freq_hz, c.amplitude) for c in run]).reshape(-1, 2) if osc.force > 0 else none
        half[leg] = (
            np.array([(c.freq_hz, c.half) for c in run if math.isfinite(c.half)]).reshape(-1, 2)
            if osc.pump > 0
            else none
        )
        for arr in (first[leg], half[leg]):
            if arr.size:
                top = max(top, float(arr[-400:, 1].max()))
    now = none
    if cycles:
        last = cycles[-1]
        amp = last.half if pumped_only(osc) else last.amplitude
        if math.isfinite(amp):
            now = np.array([(last.freq_hz, amp)])
    taller = 1.15 * top if top > y_top and math.isfinite(top) and top < runaway else None
    return Points(first, half, now, np.array(record.section).reshape(-1, 2), taller)


# ------------------------------------------------------------------ every frame


@dataclass(frozen=True)
class Live:
    strip: Curve  # x against t over the last WINDOW s, min/max per bucket
    strip_range: tuple[float, float] | None  # None before any samples
    orbit: Curve  # v against x over the last PHASE_CYCLES drive cycles


def min_max(t: np.ndarray, y: np.ndarray, buckets: int) -> tuple[np.ndarray, np.ndarray]:
    """Each bucket's min and max, in order, so the line covers what the whole record would."""
    if y.size <= 2 * buckets:
        return t, y
    size = -(-y.size // buckets)
    n = y.size // size * size
    yy = y[:n].reshape(-1, size)
    tt = t[:n:size]
    out = np.empty((2, yy.shape[0]))
    out[0], out[1] = yy.min(axis=1), yy.max(axis=1)
    return np.repeat(tt, 2), out.T.ravel()


def live(trace: Trace, t_now: float, freq_hz: float) -> Live:
    t, x, _ = trace.since(t_now - WINDOW)
    strip = Curve(*min_max(t, x, PLOT_POINTS))
    span = PHASE_CYCLES / freq_hz if freq_hz > 0 else 2.0
    _, x, v = trace.since(t_now - span)
    return Live(strip, (max(t_now - WINDOW, 0.0), max(t_now, WINDOW)) if t.size else None, Curve(x, v))


# ------------------------------------------------------------------ the Ince–Strutt chart


@dataclass(frozen=True)
class Chart:
    growth: np.ndarray  # (ratios, depths): sigma / omega_n of the pumped linear spring's swing
    rect: tuple[float, float, float, float]  # the image's x, y, width, height on the axes
    tongue: Curve  # harmonic balance's first tongue (NaN where it has none)


def chart(zeta: float) -> Chart:
    """The chart at a damping ratio (~0.1 s: the page redraws it only when ζ changes)."""
    growth = floquet_growth(zeta, CHART_RATIOS, CHART_DEPTHS, CHART_STEPS).T
    dr, dm = CHART_RATIOS[1] - CHART_RATIOS[0], CHART_DEPTHS[1] - CHART_DEPTHS[0]
    rect = (
        float(CHART_RATIOS[0] - dr / 2),
        float(CHART_DEPTHS[0] - dm / 2),
        float(CHART_RATIOS[-1] - CHART_RATIOS[0] + dr),
        float(CHART_DEPTHS[-1] - CHART_DEPTHS[0] + dm),
    )
    # harmonic balance's first tongue: (1 - r^2/4)^2 + (zeta r)^2 = (mu/2)^2, a quadratic in r^2
    lo, hi = [], []
    for mu in CHART_DEPTHS:
        roots = np.roots([1 / 16, zeta**2 - 0.5, 1 - mu * mu / 4])
        r2 = np.sort(roots[(np.abs(roots.imag) < 1e-12) & (roots.real > 0)].real)
        lo.append(math.sqrt(r2[0]) if r2.size == 2 else np.nan)
        hi.append(math.sqrt(r2[1]) if r2.size == 2 else np.nan)
    tongue = Curve(np.concatenate([lo[::-1], hi]), np.concatenate([CHART_DEPTHS[::-1], CHART_DEPTHS]))
    return Chart(growth, rect, tongue)


@dataclass(frozen=True)
class ChartPoint:
    now: np.ndarray  # (0 or 1, 2): (Omega / omega_n, mu)
    note: str


NO_SPRING = "No linear spring: no natural frequency to chart the pump against."


def chart_point(osc: Duffing, freq_hz: float) -> ChartPoint:
    if osc.stiffness <= 0:
        return ChartPoint(np.empty((0, 2)), NO_SPRING)
    r = freq_hz / osc.natural_hz
    rate = float(floquet_growth(osc.zeta, np.array([r]), np.array([osc.pump]), CHART_STEPS)[0, 0])
    verdict = (
        f"rest is unstable: a swing grows as e<sup>{rate:.3g} ω<sub>n</sub> t</sup>"
        if rate > 0
        else f"rest is stable: a swing dies away as e<sup>−{-rate:.3g} ω<sub>n</sub> t</sup>"
    )
    return ChartPoint(
        np.array([[r, osc.pump]]),
        f"ζ = {100 * osc.zeta:.2g} %. Pumping at Ω/ω<sub>n</sub> = {r:.3g} with μ = {osc.pump:.3g}, {verdict} "
        "(on the linear spring). Click to pump elsewhere.",
    )


# ------------------------------------------------------------------ the texts


def osc_info(osc: Duffing, curve: ResponseCurve | None) -> str:
    parts = []
    if osc.stiffness > 0:
        parts.append(f"Small swings: f<sub>n</sub> = {osc.natural_hz:.3g} Hz, ζ = {100 * osc.zeta:.2g} %.")
    else:
        parts.append("No linear spring: the frequency depends on the swing alone.")
    if osc.cubic < 0:
        parts.append(f"Escape amplitude {fmt_length(osc.escape)}.")
    elif osc.cubic > 0 and osc.stiffness > 0 and curve is not None and 0 < curve.tip < math.inf:
        ratio = 0.75 * osc.cubic * curve.tip**2 / osc.stiffness
        parts.append(
            f"At the peak ({fmt_length(curve.tip)}) the cubic adds {100 * ratio:.3g} % to the spring's stiffness."
        )
    return " ".join(parts)


def drive_state(drive: Drive, t: float) -> str:
    text = {"up": "Sweeping up", "down": "Sweeping down", "hold": "Holding"}[drive.leg]
    return f"{text} at {drive.freq_hz:.3f} Hz, t = {t:.1f} s"


def steady_states(osc: Duffing, freq_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """The steady amplitudes at a held frequency (smallest first) and their stability: the drive's, or with no
    drive force the pump's (rest first)."""
    return parametric_amplitudes(osc, freq_hz) if pumped_only(osc) else amplitudes(osc, freq_hz)


@dataclass(frozen=True)
class Branches:
    enabled: tuple[bool, bool, bool]  # the Lower, Middle and Upper buttons
    note: str


def branches(osc: Duffing, freq_hz: float) -> Branches:
    a, stable = steady_states(osc, freq_hz)
    listed = ", ".join(f"{fmt_length(x) if x > 0 else 'rest'}{'' if ok else ' (unstable)'}" for x, ok in zip(a, stable))
    what = "swings at half the pump's frequency" if pumped_only(osc) else "steady amplitudes"
    f = freq_hz
    if a.size >= 2:
        note = f"At {f:.3g} Hz, {what}: {listed}."
    elif a.size:
        note = f"At {f:.3g} Hz there is one steady state, {listed}: hold inside the band for more."
    else:
        note = f"At {f:.3g} Hz there is no steady response below the escape amplitude."
    return Branches((a.size > 0, a.size >= 3, a.size >= 2), note)


def _measured(cycles: list[Cycle], p_curve: ResponseCurve, half: bool) -> str | None:
    """'Measured: the jump at ...' for the sweep's legs so far, if any jumped."""
    floor = JUMP_FLOOR * p_curve.tip if half else 0.0
    measured = []
    for leg, word in (("up", "going up"), ("down", "going down")):
        f = measured_jump(cycles, leg, half=half, floor=floor)
        if f is not None:
            measured.append(f"{f:.3g} Hz {word}")
    return "Measured: the jump at " + " and ".join(measured) + "." if measured else None


def readout(
    osc: Duffing, curve: ResponseCurve, p_curve: ResponseCurve, cycles: list[Cycle], escaped: bool, runaway: float
) -> str:
    """The text under the response plot: harmonic balance's jumps, those the run measured, and an escape."""
    lines = []
    if osc.force > 0:
        folds = sorted(curve.folds, key=lambda f: f.freq_hz)
        if len(folds) >= 2:
            lo, hi = folds[0], folds[-1]
            lines.append(
                f"Harmonic balance: two stable amplitudes between {lo.freq_hz:.3g} and {hi.freq_hz:.3g} Hz. "
                f"Sweeping up, the {'upper' if hi.down else 'lower'} branch ends at {hi.freq_hz:.3g} Hz "
                f"({fmt_length(hi.amplitude)}) and {'drops' if hi.down else 'jumps'} to "
                f"{fmt_length(hi.lands_at)}; sweeping down, the {'upper' if lo.down else 'lower'} ends at "
                f"{lo.freq_hz:.3g} Hz ({fmt_length(lo.amplitude)}) and "
                f"{'drops' if lo.down else 'jumps'} to {fmt_length(lo.lands_at)}."
            )
            measured = _measured(cycles, p_curve, half=False)
            if measured:
                lines.append(measured)
        elif folds:
            fold = folds[0]
            lines.append(
                f"Harmonic balance: a branch ends at {fold.freq_hz:.3g} Hz ({fmt_length(fold.amplitude)}) and "
                f"{'drops' if fold.down else 'jumps'} to {fmt_length(fold.lands_at)}; the other runs into the "
                "escape amplitude."
                if osc.cubic < 0
                else f"Harmonic balance: a branch ends at {fold.freq_hz:.3g} Hz; the other runs off the top "
                "(no damping to bound it)."
            )
        else:
            lines.append(
                "Harmonic balance: one steady amplitude at each frequency, so no jump: sweeps up and down agree."
            )
    if osc.pump > 0:
        edges = tongue_hz(osc)
        if edges is None:
            lines.append("Pumped: the pump is too weak for the damping, so rest is stable at every frequency.")
        else:
            text = f"Pumped: rest is unstable between {edges[0]:.3g} and {edges[1]:.3g} Hz (harmonic balance)"
            if osc.cubic == 0:
                text += ", and there a swing at half the pump's frequency grows without bound."
            else:
                jumps = [f for f in p_curve.folds if f.lands_at > f.amplitude]
                text += ", where the swing at half its frequency settles on the branch" + (
                    f"; sweeping down, rest turns unstable at {jumps[0].freq_hz:.3g} Hz and the swing "
                    f"jumps to {fmt_length(jumps[0].lands_at)}."
                    if jumps
                    else "."
                )
            lines.append(text)
            if osc.force == 0 and osc.cubic != 0:
                measured = _measured(cycles, p_curve, half=True)
                if measured:
                    lines.append(measured)
            if osc.force > 0:
                lines.append(
                    "The two curves leave each other out: harmonic balance of the drive alone, and of the pump alone."
                )
    if not lines:
        lines.append("No drive and no pump: the mass only rings down.")
    if escaped:
        lines.append(
            f"<b>The mass went over the potential's hilltop (±{fmt_length(osc.escape)}) and ran away.</b> Reset to "
            "start again."
            if osc.cubic < 0
            else f"<b>The swing grew past {fmt_length(runaway)}, without bound: a linear spring has nothing to stop "
            "it.</b> Reset to start again."
        )
    return "<br>".join(lines)
