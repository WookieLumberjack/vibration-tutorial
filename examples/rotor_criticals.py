"""A Jeffcott rotor's critical speeds as its bearings stiffen.

Soft bearings let the rotor bounce on them as a rigid body; stiff ones leave the shaft to bend. Sweeping both
bearings' stiffness (`rotor.Bearing`) and finding the forward critical speeds from the Campbell diagram
(`rotor_modes.campbell`) shows the change: the criticals rise with the bearings, then level off at the
shaft's own, pinned-pinned, values.

Run it with `uv run python examples/rotor_criticals.py`.
"""

from __future__ import annotations

import numpy as np

from vib_tutorial.core.rotor import Bearing, RotorSystem
from vib_tutorial.core.rotor_modes import campbell
from vib_tutorial.core.rotor_simulator import RPM

STIFFNESSES = np.logspace(4, 8, 9)  # N/m, each bearing, both directions
TOP_SPEED = 2000.0  # rad/s: the Campbell diagram's range


def forward_criticals(k: float) -> list[float]:
    """The forward critical speeds (rpm) below TOP_SPEED, with both bearings `k` N/m each way."""
    bearing = Bearing(kx=k, ky=k)
    rotor = RotorSystem(bearing_a=bearing, bearing_b=bearing)
    return [w * RPM for w, whirl, _ in campbell(rotor, TOP_SPEED).criticals if whirl > 0]


def main() -> list[tuple[float, list[float]]]:
    print("bearing k (N/m)   forward criticals (rpm)")
    rows = []
    for k in STIFFNESSES:
        criticals = forward_criticals(float(k))
        rows.append((float(k), criticals))
        print(f"{k:15.3g}   " + ", ".join(f"{c:7.0f}" for c in criticals))
    return rows


if __name__ == "__main__":
    main()
