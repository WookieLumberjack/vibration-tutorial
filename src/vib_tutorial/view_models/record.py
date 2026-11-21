"""What a running page records, in one place (#409 S-9): the samples its time plots and scrubbing read back.

Two ring buffers, Qt-free, both keeping the most recent samples and dropping the oldest half (or more) when
one more stretch won't fit (`_make_room`): amortized O(1) a sample.

- `SampleBuffer(columns, capacity)`: times and a row of columns a sample (the rotor, the top, the rigid body, the
  CMS model in time). `window(seconds)` is the last stretch.
- `History(n, capacity)`: the chain's samples (the Simulation page): displacements, velocities, the force, the
  energies (`ENERGY_COLUMNS`), the element forces and the ground's motion. `offset` numbers the samples: t[i] is
  sample offset + i of every sample ever stored, never restarting (not even on `reset`), so caches keyed by it
  (the strip chart's Decimator) see new data.

Working on it: `tests/test_view_models.py` (`test_record_buffers_keep_the_latest`), and the pages' running tests.
"""

from __future__ import annotations

import numpy as np

# Columns of History.e: stored energy (with the parameters of the time) and the ledger.
ENERGY_COLUMNS = ("kinetic", "potential", "added", "work", "dissipated", "friction")


def _make_room(arrays: tuple[np.ndarray, ...], size: int, capacity: int, k: int) -> tuple[int, int]:
    """Room for `k` more after `size` stored: drop the oldest half (or more) if they won't fit. (kept, dropped)."""
    if size + k <= capacity:
        return size, 0
    keep = min(size, capacity // 2, capacity - k)
    start = size - keep
    for a in arrays:
        a[:keep] = a[start:size]
    return keep, start


class SampleBuffer:
    """Recent samples (t, columns), the oldest half dropped when full (see the module)."""

    def __init__(self, columns: int, capacity: int = 200_000) -> None:
        self.t = np.empty(capacity)
        self.y = np.empty((capacity, columns))
        self.size = 0

    def clear(self) -> None:
        self.size = 0

    def extend(self, ts: np.ndarray, ys: np.ndarray) -> None:
        k = min(ts.size, self.t.size)
        self.size, _ = _make_room((self.t, self.y), self.size, self.t.size, k)
        self.t[self.size : self.size + k] = ts[-k:] if k else ts[:0]
        self.y[self.size : self.size + k] = ys[-k:] if k else ys[:0]
        self.size += k

    def window(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        t = self.t[: self.size]
        i0 = int(np.searchsorted(t, t[-1] - seconds)) if self.size else 0
        return t[i0:], self.y[i0 : self.size]


class History:
    """The chain's recent samples (see the module): t, x, v, f, e (energies), s (n springs' forces, then n
    dampers'), g (the ground's displacement and velocity, 0 unless the input is a ground motion)."""

    def __init__(self, n: int, capacity: int = 300_000) -> None:
        self.capacity = capacity
        self.offset = self.size = 0
        self.reset(n)

    def reset(self, n: int) -> None:
        self.offset += self.size
        self.n = n
        self.t: np.ndarray = np.empty(self.capacity)
        self.x = np.empty((self.capacity, n))
        self.v = np.empty((self.capacity, n))
        self.f = np.empty(self.capacity)
        self.e = np.empty((self.capacity, len(ENERGY_COLUMNS)))
        self.s = np.empty((self.capacity, 2 * n))
        self.g = np.empty((self.capacity, 2))
        self.size = 0

    def extend(
        self,
        ts: np.ndarray,
        xs: np.ndarray,
        vs: np.ndarray,
        fs: np.ndarray,
        es: np.ndarray,
        ss: np.ndarray,
        gs: np.ndarray | None = None,
    ) -> None:
        new: tuple[np.ndarray, ...] = (ts, xs, vs, fs, es, ss, np.zeros((ts.size, 2)) if gs is None else gs)
        k = ts.size
        if k == 0:
            return
        if k >= self.capacity:
            new = tuple(a[-self.capacity :] for a in new)
            self.offset += k - self.capacity  # samples never stored still count
            k = self.capacity
        arrays: tuple[np.ndarray, ...] = (self.t, self.x, self.v, self.f, self.e, self.s, self.g)
        self.size, dropped = _make_room(arrays, self.size, self.capacity, k)
        self.offset += dropped
        s = slice(self.size, self.size + k)
        for a, b in zip(arrays, new):
            a[s] = b
        self.size += k

    def window(self, seconds: float) -> tuple[np.ndarray, ...]:
        """Views (t, x, v, f, e, s, g) of the samples within the last `seconds` of simulated time."""
        i0 = 0
        if self.size:
            i0 = int(np.searchsorted(self.t[: self.size], self.t[self.size - 1] - seconds))
        s = slice(i0, self.size)
        return self.t[s], self.x[s], self.v[s], self.f[s], self.e[s], self.s[s], self.g[s]
