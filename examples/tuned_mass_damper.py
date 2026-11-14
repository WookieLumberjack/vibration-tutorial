"""How much a tuned mass damper calms a chain's first mode, against its mass.

A four-mass chain, lightly damped, is shaken by a force on its last mass. A damper tuned by Den Hartog's rule
(`presets.tuned_mass_damper`) is added there, a fraction mu of mode 1's modal mass, and the peak receptance of
that mass near mode 1 is compared with the chain's own. A heavier damper calms the mode more, with less gain
for each extra kilogram.

Run it with `uv run python examples/tuned_mass_damper.py`.
"""

from __future__ import annotations

import numpy as np

from vib_tutorial.core import ChainSystem, modal_analysis
from vib_tutorial.core.modal import frf
from vib_tutorial.core.presets import den_hartog, modal_mass, tuned_mass_damper

MASS_RATIOS = (0.01, 0.02, 0.05, 0.1)


def peak_receptance(system: ChainSystem, dof: int, freqs_hz: np.ndarray) -> float:
    """The largest |X/F| (m/N) of mass `dof`, for a force on it, over `freqs_hz`."""
    return float(np.abs(frf(system, freqs_hz, dof)[:, dof]).max())


def main() -> list[tuple[float, float, float, float]]:
    chain = ChainSystem([1.0] * 4, [400.0] * 4, [0.5] * 4)
    tip = chain.n - 1
    f1 = modal_analysis(chain).modes[0].fn_hz
    freqs = np.linspace(0.5 * f1, 1.5 * f1, 4001)  # around mode 1
    bare = peak_receptance(chain, tip, freqs)
    print(f"Mode 1 at {f1:.3f} Hz; its modal mass at m4 {modal_mass(chain, 0, tip):.3f} kg")
    print(f"Peak |X4/F4| without a damper: {bare * 1e3:.2f} mm/N\n")
    print("   mu   f_a/f_1   zeta_a   peak (mm/N)   reduction")
    rows = []
    for mu in MASS_RATIOS:
        ratio, zeta = den_hartog(mu)
        peak = peak_receptance(tuned_mass_damper(chain, mu), tip, freqs)
        rows.append((mu, ratio, zeta, peak / bare))
        print(f"{mu:5.2f}   {ratio:7.3f}   {zeta:6.3f}   {peak * 1e3:11.3f}   {bare / peak:7.1f}x")
    return rows


if __name__ == "__main__":
    main()
