"""The build-the-plot recorder (#478 item 3): the reader builds a frequency response from a live run.

Drive a running system at one frequency and dwell there: once its swing is steady, the recorder lands a point,
the response's amplitude and phase over the drive's, at that frequency. *Sweep* steps the drive through a set of
frequencies (`sweep_frequencies`), dwelling at each until its point lands. So the first FRF a reader sees is one
they measured, point by point, the way a test rig's stepped sine does (the theory's curve can be shown over it
afterwards).

`measure(t, out, inp, freq_hz, since)` is one frequency's single-bin Fourier coefficient of the response and of
the drive over the last `PERIODS` whole periods (after `since`, the last change: the drive's frequency or the
system), and the same over the periods before them; `Measurement.steady` is whether the two agree within
`TOLERANCE` in size and phase, so a swing still growing, or beating (undamped, off resonance), never lands.

`Recorder` keeps the points, each with the key of the system it was measured on (the page's m, k, c): a point
from before a change stays, and the page draws it faded (`current`, `earlier`). A frequency already measured on
this system is measured again, the new point in its place.

Working on it: Qt-free (the package's docstring); `tests/test_live_parts.py` checks a dwell lands the theory's
receptance, a growing or beating swing never lands, and a sweep visits its frequencies.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np

PERIODS = 3  # whole periods in each estimate
TOLERANCE = 0.005  # the two estimates' relative difference that counts as steady
SAME_FREQ = 1e-6  # relative: a frequency already measured


@dataclass(frozen=True)
class Measurement:
    freq_hz: float
    response: complex  # the output's amplitude and phase over the input's (X/F: a receptance)
    steady: bool


@dataclass(frozen=True)
class Point:
    freq_hz: float
    response: complex
    key: Hashable  # the system it was measured on


def _coefficient(t: np.ndarray, y: np.ndarray, omega: float) -> complex:
    """The single-bin Fourier coefficient of y over t (whole periods), by the trapezium rule."""
    e = y * np.exp(-1j * omega * t)
    return complex(np.trapezoid(e, t)) * 2.0 / (t[-1] - t[0])


def measure(
    t: np.ndarray, out: np.ndarray, inp: np.ndarray, freq_hz: float, since: float = -math.inf
) -> Measurement | None:
    """The response over the input at `freq_hz` from the last `PERIODS` periods after `since`, steady or not (see
    the module), or None if there's too little run since then, or no input."""
    if freq_hz <= 0 or t.size < 8:
        return None
    period = 1.0 / freq_hz
    end = float(t[-1])
    span = PERIODS * period
    if end - 2 * span < max(since, float(t[0])):
        return None
    omega = 2.0 * math.pi * freq_hz
    estimates = []
    for start in (end - span, end - 2 * span):
        i0, i1 = np.searchsorted(t, [start, start + span])
        if i1 - i0 < 8 * PERIODS:
            return None
        ts = t[i0:i1]
        f = _coefficient(ts, inp[i0:i1], omega)
        if abs(f) == 0:
            return None
        estimates.append(_coefficient(ts, out[i0:i1], omega) / f)
    now, before = estimates
    steady = abs(now - before) <= TOLERANCE * abs(now)
    return Measurement(freq_hz, now, bool(steady))


def sweep_frequencies(
    natural_hz: float, damping_ratio: float, points: int = 15, lo: float = 0.33, hi: float = 3.0
) -> tuple[float, ...]:
    """About `points` frequencies for a sweep from `lo` to `hi` × `natural_hz`, log-spaced; with 9 or more, five
    of them packed round the resonance, within its half-power band (at least ±5 %), where they fall in the range."""
    lo, hi = sorted((lo, hi))
    if points < 9:
        return tuple(float(f) for f in np.geomspace(lo * natural_hz, hi * natural_hz, max(points, 2)))
    wide = np.geomspace(lo * natural_hz, hi * natural_hz, points - 4)
    band = max(damping_ratio, 0.05)
    near = natural_hz * (1.0 + band * np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))
    near = near[(near > wide[0]) & (near < wide[-1])]
    freqs = np.sort(np.concatenate([wide, near]))
    keep = np.concatenate([[True], np.diff(freqs) > 0.02 * freqs[1:]])
    return tuple(float(f) for f in freqs[keep])


class Recorder:
    """The points measured so far (see the module)."""

    def __init__(self) -> None:
        self.points: list[Point] = []

    def land(self, measurement: Measurement, key: Hashable) -> Point | None:
        """Land a steady measurement's point on system `key` (in place of one at its frequency): the point, or None
        if it isn't steady."""
        if not measurement.steady:
            return None
        f = measurement.freq_hz
        self.points = [p for p in self.points if not (p.key == key and abs(p.freq_hz - f) <= SAME_FREQ * f)]
        point = Point(f, measurement.response, key)
        self.points.append(point)
        return point

    def has(self, freq_hz: float, key: Hashable) -> bool:
        return any(p.key == key and abs(p.freq_hz - freq_hz) <= SAME_FREQ * freq_hz for p in self.points)

    def current(self, key: Hashable) -> list[Point]:
        return sorted((p for p in self.points if p.key == key), key=lambda p: p.freq_hz)

    def earlier(self, key: Hashable) -> list[Point]:
        return sorted((p for p in self.points if p.key != key), key=lambda p: p.freq_hz)

    def clear(self) -> None:
        self.points = []


class Sweep:
    """A stepped sweep: the drive dwells at each frequency until its point lands, then moves on."""

    def __init__(self, freqs: tuple[float, ...]) -> None:
        self.freqs = freqs
        self.index = 0

    @property
    def frequency(self) -> float | None:
        """Where the drive should be now, or None once the sweep is done."""
        return self.freqs[self.index] if self.index < len(self.freqs) else None

    def landed(self) -> float | None:
        """This frequency's point landed: the next frequency, or None when done."""
        self.index += 1
        return self.frequency

    @property
    def done(self) -> bool:
        return self.index >= len(self.freqs)
