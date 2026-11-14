"""The code readers copy: `examples/`' scripts and the Python in docs/code-layout.md's *Scripting the core*.
Each runs here, and the numbers it quotes are checked, so an API change that breaks them fails the suite."""

from __future__ import annotations

import importlib.util
import itertools
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"```python\n(.*?)```", re.S)


def _example(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    assert spec is not None and spec.loader is not None, name
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_example_is_tested():
    assert {p.stem for p in (ROOT / "examples").glob("*.py")} == {"tuned_mass_damper", "rotor_criticals"}


def test_tuned_mass_damper(capsys):
    rows = _example("tuned_mass_damper").main()
    left = [peak for *_, peak in rows]
    assert left[0] < 0.15 and all(b < a for a, b in itertools.pairwise(left)), "a heavier damper calms it more"
    gains = [1 / p for p in left]
    assert gains[1] - gains[0] > (gains[3] - gains[2]) / 5, "less for each extra kilogram"
    assert "reduction" in capsys.readouterr().out


def test_rotor_criticals(capsys):
    rows = _example("rotor_criticals").main()
    first = [criticals[0] for _, criticals in rows]
    assert all(b > a for a, b in itertools.pairwise(first)), "rising with the bearings"
    assert first[-1] == pytest.approx(first[-2], rel=1e-3), "levelled off at the shaft's own"
    assert "forward criticals" in capsys.readouterr().out


def test_the_scripting_example_in_code_layout_runs():
    blocks = FENCE.findall((ROOT / "docs" / "code-layout.md").read_text(encoding="utf-8"))
    assert blocks, "docs/code-layout.md's Python"
    ns: dict = {}
    exec(compile("\n".join(blocks), "code-layout.md", "exec"), ns)
    assert ns["den_hartog"](0.05) == pytest.approx((0.952, 0.127), abs=5e-4)
    assert abs(ns["sim"].energy_added + ns["sim"].work - ns["sim"].dissipated - ns["sim"].stored_energy) < 1e-12
    assert ns["stability_onset"](ns["whip"], 1000.0).order == pytest.approx(0.5, abs=1e-3)
    assert ns["heavy"].n_dof == 12
    W = ns["W"]
    assert np.abs(W) / 0.02 * 1e3 == pytest.approx([7.32, 6.32], abs=5e-3), "grams at 20 mm"
    assert np.degrees(np.angle(W)) % 360 == pytest.approx([180.0, 0.0], abs=0.1)
