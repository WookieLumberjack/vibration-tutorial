"""The numbers quoted in the tutorial text (Background, Theory tabs, Try it lists, README).

Each test recomputes what a page tells the student they will see, so a change to a
default, a preset or a solver cannot silently make the text wrong. If one fails, fix
the text (or the change), and quote the new value.

A test ties a figure to the notes that quote it with `quoted("<notes file>", "1.7 %")` (`notes_quoting.py`), which
fails unless the notes (`theory/<notes file>.html`, outside its `<eq>`s) show that number, and gives its value
(1.7) to check against what the code works out. Any test module may call it: a page's figures that its own tests
already work out are tied there. `test_every_quoted_number_is_checked` reads every module's `quoted` calls and fails
on a decimal or a percentage in the notes that none ties and `notes_unchecked_numbers.txt` doesn't list: the
numbers written before the guard, to tie as their pages are next touched (delete a line's number when it's
tied). So an edited figure (Pendulums' 1.8 %, #220) or a new one needs a test. Run this file as a script to
rewrite the list, only for a number that has nothing to check (a section number, a value taken as given).
"""

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from notes_quoting import THEORY, notes_numbers, quoted
from scipy.signal import find_peaks

from vib_tutorial.core import (
    ChainSystem,
    Simulator,
    compare_coupled,
    compare_modes,
    component_mode_synthesis,
    coupled_model,
    frf,
    modal_analysis,
    modal_energies,
    modal_frf_terms,
    transmissibility,
)
from vib_tutorial.core.presets import presets

UNCHECKED = Path(__file__).with_name("notes_unchecked_numbers.txt")


def tied_numbers() -> dict[str, set[str]]:
    """The numbers each notes file has tied by a `quoted(...)` call in any test module (its literal arguments)."""
    tied: dict[str, set[str]] = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "quoted":
                args = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                where = f"{path.name}, line {node.lineno}"
                assert len(args) == 2 == len(node.args), f"{where}: quoted takes two string literals"
                tied.setdefault(args[0], set()).add(args[1])
    return tied


def unchecked_numbers() -> dict[str, list[str]]:
    """Each notes file's numbers no `quoted` call ties, smallest first."""
    tied = tied_numbers()
    out = {}
    for path in sorted(THEORY.glob("*.html")):
        left = notes_numbers(path.stem) - tied.get(path.stem, set())
        if left:
            out[path.stem] = sorted(left, key=lambda n: (float(n.removesuffix(" %")), n))
    return out


def listing(unchecked: dict[str, list[str]]) -> str:
    head = "# The notes' numbers no test ties yet (see tests/test_tutorial_numbers.py's docstring).\n"
    return head + "".join(f"{notes}: {' | '.join(nums)}\n" for notes, nums in unchecked.items())


def test_every_quoted_number_is_checked():
    listed = {}
    for line in UNCHECKED.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            notes, _, nums = line.partition(": ")
            listed[notes] = nums.split(" | ")
    tied = tied_numbers()
    for notes, nums in tied.items():
        missing = nums - notes_numbers(notes)
        assert not missing, f"{notes}.html doesn't quote {sorted(missing)}"
    now = unchecked_numbers()
    new = {n: sorted(set(v) - set(listed.get(n, []))) for n, v in now.items()}
    new = {n: v for n, v in new.items() if v}
    assert not new, f"numbers in the notes no test checks: tie each with quoted(...): {new}"
    stale = {n: sorted(set(v) - set(now.get(n, []))) for n, v in listed.items()}
    stale = {n: v for n, v in stale.items() if v}
    assert not stale, f"tied or gone: take them off {UNCHECKED.name}: {stale}"


def peaks(system: ChainSystem, dof: int, scale: float, lo: float = 0.5, hi: float = 8.0) -> list[float]:
    f = np.linspace(lo, hi, 100001)
    h = np.abs(frf(system, f, dof)[:, dof]) * scale
    return [float(h[i]) for i in find_peaks(h)[0]]


def preset(name: str):
    return next(p for p in presets() if name in p.name)


def with_c1(c1: float) -> ChainSystem:
    s = ChainSystem.uniform(4)
    s.damping[0] = c1
    return s


@pytest.mark.parametrize(
    ("c1", "coupling", "modal", "exact", "apart"),
    [
        (
            5,
            quoted("background", "0.36"),
            quoted("background", "0.0977"),
            quoted("background", "0.0979"),
            quoted("background", "0.2 %"),
        ),
        (
            15,
            quoted("background", "0.71"),
            quoted("background", "0.168"),
            quoted("background", "0.180"),
            quoted("background", "7 %"),
        ),
        (50, quoted("background", "0.90"), quoted("background", "0.45"), quoted("background", "0.98"), None),
    ],
)
def test_background_coupling_table(c1, coupling, modal, exact, apart):
    r = modal_analysis(with_c1(c1))
    assert r.coupling == pytest.approx(coupling, abs=0.005)
    pairs = [(m.zeta_modal, m.damped.zeta) for m in r.modes if m.damped]
    assert any(z_m == pytest.approx(modal, rel=0.01) and z_e == pytest.approx(exact, rel=0.01) for z_m, z_e in pairs)
    if apart is not None:  # how far apart, as the table rounds it
        assert 100 * (1 - modal / exact) == pytest.approx(apart, abs=0.5 if apart > 1 else 0.05)


def test_background_try_it_overdamped_at_c1_200():
    roots = modal_analysis(with_c1(200)).overdamped_roots
    assert len(roots) == 2 and roots[0] > -3 and roots[1] < -100  # one slow, one very fast


def test_background_try_it_energy_moves_to_mode_1():
    for c1, mode1_share in ((15, 0.85), (2, None)):
        s = with_c1(c1)
        r = modal_analysis(s)
        sim = Simulator(s)
        sim.set_state(0.02 * r.modes[2].shape, np.zeros(4))
        for _ in range(20):
            sim.advance(0.05)  # 1 s
        e = modal_energies(s, r, sim.displacement, sim.velocity)
        if mode1_share is None:
            assert e[2] / e.sum() == pytest.approx(1.0, abs=1e-6)  # mode 3 keeps it all
        else:
            assert e[0] / e.sum() > mode1_share  # "within a second most of what is left is in mode 1"


def test_presets_absorber_and_tuned_mass_damper():
    absorber = preset("absorber (undamped)").system
    f = modal_analysis(absorber).modes
    assert (f[1].fn_hz - f[0].fn_hz) / 3.183 == pytest.approx(0.1**0.5, abs=0.01)  # "√0.1 ≈ 32% apart"
    tmd = preset("Den Hartog").system
    assert max(peaks(tmd.resized(1), 0, 400.0)) == pytest.approx(40.0, rel=0.01)  # "40 times"
    assert max(peaks(tmd, 0, 400.0)) < 6.0  # "under 6"
    undamped = ChainSystem(tmd.masses, tmd.stiffness, [0.0, tmd.damping[1]])
    assert peaks(undamped, 0, 400.0) == pytest.approx(
        [quoted("background", "6.4"), quoted("background", "6.4")], rel=0.005
    )  # "about 6.4", equal
    for c2, count in ((0.0, 2), (5.0, 1)):  # "Set c2 to 0: two sharp peaks. Set it to 5: one"
        assert len(peaks(ChainSystem(tmd.masses, tmd.stiffness, [tmd.damping[0], c2]), 0, 400.0)) == count


def coupling_errors(system, split, mode_a=0, mode_b=0, a_mass="tip"):
    m = coupled_model(system, split, mode_a, mode_b, a_mass)
    return m, [(c.error, c.mac) for c in compare_coupled(m, modal_analysis(system))]


def test_modal_coupling_try_it():
    _, errs = coupling_errors(preset("absorber (undamped)").system, 1)
    assert all(abs(e) < 1e-9 for e, _ in errs)
    tuned = ChainSystem([1, 1, 0.1, 0.1], [400, 400, 40, 40], [2, 2, 0.2, 0.2])
    m, errs = coupling_errors(tuned, 2)
    assert m.mass_ratio == pytest.approx(0.14, abs=0.005)
    assert [e for e, _ in errs] == pytest.approx(
        [quoted("coupling_notes", "0.2 %") / 100, 0.0033], abs=0.0005
    )  # "about 0.2%"
    _, errs = coupling_errors(tuned, 2, a_mass="effective")
    assert max(abs(e) for e, _ in errs) == pytest.approx(
        quoted("coupling_notes", "3 %") / 100, abs=0.002
    )  # "about 3%", either sign
    _, errs = coupling_errors(ChainSystem.uniform(8), 6, mode_a=1)
    assert errs[0][0] == pytest.approx(-0.14, abs=0.005)  # "14% too low"


def test_modal_coupling_worked_example():
    import math

    from vib_tutorial.core.modal_coupling import subsystems

    chain = ChainSystem.uniform(4)
    A, B = subsystems(chain, 2)
    assert A.fn_hz[0] == pytest.approx(quoted("coupling_notes", "1.967"), abs=5e-4)
    assert B.fn_hz[0] == pytest.approx(quoted("coupling_notes", "1.967"), abs=5e-4)
    assert abs(A.Phi[:, 0]) == pytest.approx(
        [quoted("coupling_notes", "0.526"), quoted("coupling_notes", "0.851")], abs=5e-4
    )
    w2 = (2 * math.pi * 1.967) ** 2
    m_b_gamma2 = coupled_model(chain, 2).osc_b.m
    assert m_b_gamma2 == pytest.approx(quoted("coupling_notes", "1.89"), abs=0.005)  # Γ₂, well above 1/φ_tip² = 1.38
    assert (1 / 0.851**2, 1.38 * w2, 0.526 + 0.851, 1.89 * w2) == pytest.approx(
        (quoted("coupling_notes", "1.38"), 211, quoted("coupling_notes", "1.376"), 289), rel=3e-3
    )
    m = coupled_model(chain, 2)
    assert (m.osc_a.m, m.osc_b.m, m.m_a) == pytest.approx(
        (quoted("coupling_notes", "1.382"), quoted("coupling_notes", "1.894"), 1.488), abs=5e-4
    )
    assert m.m_a == pytest.approx(quoted("coupling_notes", "1.49"), abs=0.005)  # the residual mass on A
    assert (m.osc_a.k, m.osc_b.k) == pytest.approx((211, 289), abs=0.5)
    assert 2 - m.osc_b.m == pytest.approx(quoted("coupling_notes", "0.106"), abs=5e-4)
    mu = 1.894 / 1.382
    assert mu == pytest.approx(quoted("coupling_notes", "1.37"), abs=0.005)
    roots = (1 + mu / 2 - math.sqrt(mu + mu * mu / 4), 1 + mu / 2 + math.sqrt(mu + mu * mu / 4))
    assert (1 + mu / 2, math.sqrt(mu + mu * mu / 4)) == pytest.approx((1.685, 1.356), abs=5e-4)
    assert roots == pytest.approx((quoted("coupling_notes", "0.329"), quoted("coupling_notes", "3.04")), abs=0.005)
    f = (1.967 * math.sqrt(0.329), 1.967 * math.sqrt(3.04))
    assert f == pytest.approx((quoted("coupling_notes", "1.13"), quoted("coupling_notes", "3.43")), abs=0.005)
    assert f[1] - f[0] == pytest.approx(quoted("coupling_notes", "2.30"), abs=0.005)
    assert math.sqrt(mu) * 1.967 == pytest.approx(quoted("coupling_notes", "2.30"), abs=0.005)
    bare = coupled_model(chain, 2, residual=False)
    assert bare.fn_hz == pytest.approx((quoted("coupling_notes", "1.13"), quoted("coupling_notes", "3.43")), abs=0.005)
    assert m.mass_ratio == pytest.approx(quoted("coupling_notes", "1.27"), abs=0.005)
    assert m.fn_hz == pytest.approx((quoted("coupling_notes", "1.117"), quoted("coupling_notes", "3.339")), abs=5e-4)
    errs = compare_coupled(m, modal_analysis(chain))
    assert [c.fn_full for c in errs] == pytest.approx(
        [quoted("coupling_notes", "1.105"), quoted("coupling_notes", "3.183")], abs=5e-4
    )
    assert [c.error for c in errs] == pytest.approx(
        [quoted("coupling_notes", "1.1 %") / 100, quoted("coupling_notes", "4.9 %") / 100], abs=5e-4
    )
    assert [c.mac for c in errs] == pytest.approx(
        [quoted("coupling_notes", "0.999"), quoted("coupling_notes", "0.958")], abs=5e-4
    )
    assert [c.error for c in compare_coupled(bare, modal_analysis(chain))] == pytest.approx(
        [quoted("coupling_notes", "2.0 %") / 100, quoted("coupling_notes", "7.8 %") / 100], abs=5e-4
    )
    assert B.fn_hz[1] == pytest.approx(quoted("coupling_notes", "5.15"), abs=0.005)
    assert A.fn_hz[1] == pytest.approx(quoted("coupling_notes", "5.15"), abs=0.005)
    # the "Two equal halves" trial: μ = 1.27, the upper mode 4.9% high, a MAC of 0.96
    assert errs[1].mac == pytest.approx(0.96, abs=0.005)


def cms(interfaces, kept, method="craig-bampton"):
    s = ChainSystem.uniform(8)
    return [
        (c.error, c.mac)
        for c in compare_modes(component_mode_synthesis(s, interfaces, kept, method), modal_analysis(s))
        if c.error is not None
    ]


def test_substructuring_small_example():
    K, M = np.array([[800.0, -400], [-400, 400]]), np.eye(2)
    w2, shapes = np.linalg.eigh(K)
    assert np.sqrt(w2) / (2 * np.pi) == pytest.approx(
        [quoted("cms_notes", "1.967"), quoted("cms_notes", "5.150")], abs=5e-4
    )
    assert shapes[:, 0] / shapes[1, 0] == pytest.approx([quoted("cms_notes", "0.618"), 1], abs=5e-4)
    assert shapes[:, 1] / shapes[0, 1] == pytest.approx([1, -quoted("cms_notes", "0.618")], abs=5e-4)
    V = np.array([[1.0, 1], [1, -1]])
    assert V.T @ K @ V == pytest.approx(np.array([[400, 400], [400, 2000]]))
    assert V.T @ M @ V == pytest.approx(2 * np.eye(2))
    assert 400 * 1**2 + 400 * 2**2 == 2000
    f = np.sqrt(np.linalg.eigvals(np.linalg.solve(V.T @ M @ V, V.T @ K @ V)).real) / (2 * np.pi)
    assert sorted(f) == pytest.approx([quoted("cms_notes", "1.967"), quoted("cms_notes", "5.150")], abs=5e-4)
    v = np.array([1.0, 1])
    assert np.sqrt(v @ K @ v / (v @ v)) / (2 * np.pi) == pytest.approx(quoted("cms_notes", "2.25"), abs=0.005)
    v = np.array([0.5, 1])
    assert (v @ K @ v, v @ v) == pytest.approx((200, quoted("cms_notes", "1.25")))
    assert np.sqrt(200 / 1.25) / (2 * np.pi) == pytest.approx(quoted("cms_notes", "2.01"), abs=0.005)
    # the rails: each one-shape model against mode 1, [0.618, 1]
    mode = shapes[:, 0] / shapes[1, 0]
    for rail, angle, mac, f1 in (
        ([0.5, 1], quoted("cms_notes", "5.2"), quoted("cms_notes", "0.992"), 2.01),
        ([1, 1], 13, quoted("cms_notes", "0.947"), 2.25),
        ([0, 1], 32, quoted("cms_notes", "0.724"), 3.18),
    ):
        v = np.array(rail, dtype=float)
        cos2 = (v @ mode) ** 2 / ((v @ v) * (mode @ mode))
        assert cos2 == pytest.approx(mac, abs=5e-4)
        assert np.degrees(np.arccos(np.sqrt(cos2))) == pytest.approx(angle, abs=0.5 if angle > 10 else 0.05)
        f = np.sqrt(v @ K @ v / (v @ v)) / (2 * np.pi)
        assert f == pytest.approx(f1, abs=0.005) and f > 1.967
    assert (2.01 / 1.967 - 1, 2.25 / 1.967 - 1, 3.18 / 1.967 - 1) == pytest.approx(
        (quoted("cms_notes", "2.3 %") / 100, quoted("cms_notes", "14 %") / 100, quoted("cms_notes", "62 %") / 100),
        abs=0.005,
    )


def test_substructuring_worked_example():
    import math

    chain = ChainSystem.uniform(4)
    w1 = 2 * math.sqrt(400) * math.sin(math.pi / 8)
    assert (math.sin(math.pi / 8), w1, w1 / (2 * math.pi)) == pytest.approx(
        (0.383, 15.3, quoted("cms_notes", "2.44")), rel=4e-3
    )
    K = np.array([[800.0, -400, 0], [-400, 800, -400], [0, -400, 800]])
    w2, phi = np.linalg.eigh(K)
    assert math.sqrt(w2[0]) == pytest.approx(w1)
    assert abs(phi[:, 0]) == pytest.approx(
        [quoted("cms_notes", "0.5"), quoted("cms_notes", "0.707"), quoted("cms_notes", "0.5")], abs=5e-4
    )
    psi = -np.linalg.solve(K, np.array([0.0, 0, -400]))
    assert psi == pytest.approx([quoted("cms_notes", "0.25"), quoted("cms_notes", "0.5"), quoted("cms_notes", "0.75")])
    assert 1 + psi @ psi == pytest.approx(quoted("cms_notes", "1.875"))
    guyan = math.sqrt(100 / 1.875) / (2 * math.pi)
    true = modal_analysis(chain).modes
    assert (guyan, true[0].fn_hz, guyan / true[0].fn_hz - 1) == pytest.approx(
        (quoted("cms_notes", "1.162"), quoted("cms_notes", "1.105"), quoted("cms_notes", "5.1 %") / 100), abs=5e-4
    )
    coupling = 0.5 * 0.25 + 0.707 * 0.5 + 0.5 * 0.75
    assert w1**2 == pytest.approx(234, abs=0.5)
    assert (coupling, abs(phi[:, 0]) @ psi) == pytest.approx(
        (quoted("cms_notes", "0.854"), quoted("cms_notes", "0.854")), abs=6e-4
    )
    # the 2 x 2 model, by hand and as the page builds it
    Kh, Mh = np.diag([234.0, 100.0]), np.array([[1.0, 0.854], [0.854, 1.875]])
    f_hand = np.sqrt(np.linalg.eigvals(np.linalg.solve(Mh, Kh)).real) / (2 * math.pi)
    assert sorted(f_hand) == pytest.approx([quoted("cms_notes", "1.108"), quoted("cms_notes", "3.27")], abs=0.005)
    m = component_mode_synthesis(chain, [], [1])
    c = compare_modes(m, modal_analysis(chain))
    assert (c[0].error, c[1].error) == pytest.approx(
        (quoted("cms_notes", "0.19 %") / 100, quoted("cms_notes", "2.7 %") / 100), abs=5e-4
    )
    assert c[1].mac == pytest.approx(quoted("cms_notes", "0.96"), abs=0.005)
    assert true[1].fn_hz == pytest.approx(quoted("cms_notes", "3.18"), abs=0.005)
    two = compare_modes(component_mode_synthesis(chain, [], [2]), modal_analysis(chain))
    assert (two[0].error, two[1].error) == pytest.approx(
        (quoted("cms_notes", "0.017 %") / 100, quoted("cms_notes", "0.15 %") / 100), abs=5e-5
    )
    assert component_mode_synthesis(chain, [], [0]).fn_hz[0] == pytest.approx(quoted("cms_notes", "1.162"), abs=5e-4)
    assert [component_mode_synthesis(chain, [], [k]).fn_hz[0] for k in (1, 2, 3)] == pytest.approx(
        [1.1076, quoted("cms_notes", "1.1057"), quoted("cms_notes", "1.1055")], abs=5e-5
    )
    assert all(
        abs(x.error) < 1e-9 for x in compare_modes(component_mode_synthesis(chain, [], [3]), modal_analysis(chain))
    )


@pytest.mark.parametrize(
    ("interfaces", "kept", "method", "mode", "error", "tol"),
    [
        ([], [0], "craig-bampton", 1, 0.073, 0.001),  # Guyan, +7.3%
        ([], [1], "craig-bampton", 1, 0.0033, 0.0001),
        ([], [1], "craig-bampton", 2, 0.074, 0.001),
        ([], [2], "craig-bampton", 3, 0.044, 0.001),
        ([], [1], "rubin", 2, 0.16, 0.005),
        ([3], [1, 1], "craig-bampton", 4, 0.07, 0.002),
        ([3], [0, 0], "craig-bampton", 2, 0.11, 0.005),
        ([3], [2, 2], "craig-bampton", 6, 0.027, 0.001),
        ([6], [1, 0], "craig-bampton", 3, 0.50, 0.01),
        ([2, 5], [1, 1, 1], "craig-bampton", 4, 0.023, 0.001),
        ([2, 5], [0, 0, 0], "craig-bampton", 3, 0.16, 0.005),
        ([3], [1, 1], "rubin", 4, 0.10, 0.005),
        ([3], [1, 1], "macneal", 2, 0.22, 0.005),
        ([3], [9, 9], "macneal", 6, 0.66, 0.005),
    ],
)
def test_substructuring_try_it(interfaces, kept, method, mode, error, tol):
    assert cms(interfaces, kept, method)[mode - 1][0] == pytest.approx(error, abs=tol)


def test_background_single_mass_numbers():
    s = ChainSystem.uniform(1)
    m, k, c = s.masses[0], s.stiffness[0], s.damping[0]
    wn = (k / m) ** 0.5
    zeta = c / (2 * (k * m) ** 0.5)
    assert (wn, wn / (2 * np.pi), zeta) == pytest.approx(
        (20.0, quoted("background", "3.18"), quoted("background", "0.05")), rel=0.002
    )
    assert 1 / (zeta * wn) == pytest.approx(1.0)  # time constant
    assert 2 * np.pi * zeta / (1 - zeta**2) ** 0.5 == pytest.approx(
        quoted("background", "0.31"), abs=0.005
    )  # log decrement
    assert max(peaks(s, 0, k, 1.0, 6.0)) == pytest.approx(1 / (2 * zeta), rel=0.002)  # Q = 10
    assert 2 * zeta * wn / (2 * np.pi) == pytest.approx(
        quoted("background", "0.32"), abs=0.005
    )  # half-power bandwidth, Hz


def test_background_rayleigh_damping_of_the_default_chain():
    zetas = [m.zeta_modal for m in modal_analysis(ChainSystem.uniform(4)).modes]
    assert zetas[0] == pytest.approx(0.017, abs=0.0005) and zetas[3] == pytest.approx(0.094, abs=0.0005)
    for m in modal_analysis(ChainSystem.uniform(4)).modes:
        assert m.zeta_modal == pytest.approx(quoted("background", "0.005") * m.omega_n / 2)  # zeta = beta omega / 2


def test_background_isolation_above_root_2_fn():
    for c in (0.5, 2.0, 20.0):
        s = ChainSystem([1.0], [400.0], [c])
        f = np.array([0.99, 1.01]) * 2**0.5 * 20 / (2 * np.pi)
        t = np.abs(transmissibility(s, f))[:, 0]
        assert t[0] > 1 > t[1]
    f = np.linspace(0.01, 20, 20001)
    t3 = np.abs(transmissibility(ChainSystem.uniform(4), f))[:, 2]
    assert f[np.nonzero(t3 > 1)[0][-1]] == pytest.approx(quoted("background", "1.65"), abs=0.01)  # m3's crossover


def test_frf_matrix_three_forms_and_residual_mass():
    s = ChainSystem.uniform(4)
    r = modal_analysis(s)
    hi = np.array([60.0, 120.0])  # far above the modes: driving-point accelerance tends to 1/m
    w = 2 * np.pi * hi
    acc = np.abs(-(w**2) * frf(s, hi, 3)[:, 3])
    assert acc[1] == pytest.approx(1 / s.masses[3], rel=0.01)
    # Dropping mode 1 leaves a mass-like error -phi phi / w^2 above it.
    f = np.array([4.0, 5.5])
    w = 2 * np.pi * f
    terms = modal_frf_terms(s, r, exact=False)
    without_1 = sum(t.evaluate(f) for t in terms[1:])[:, 3, 3]
    error = frf(s, f, 3)[:, 3] - without_1
    phi = r.modes[0].shape_mass_normalized[3]
    assert error.real == pytest.approx(-(phi**2) / w**2, rel=0.1)


def test_background_rigid_body_root_with_k1_zero():
    s = ChainSystem.uniform(4)
    s.stiffness[0] = 0.0
    roots = modal_analysis(s).overdamped_roots
    assert min(roots, key=abs) == pytest.approx(0.0, abs=1e-6)
    second = sorted(roots, key=abs)[1]
    assert second == pytest.approx(-quoted("complex_modes", "0.50"), abs=0.005)  # close to −c1/(m1 + … + mN)


def test_frf_matrix_antiresonances_of_h22():
    s = ChainSystem.uniform(4)
    s.damping[:] = 0.0
    f = np.linspace(0.5, 8.0, 200001)
    h = np.abs(frf(s, f, 1)[:, 1])
    dips = [float(f[i]) for i in find_peaks(-np.log(h))[0]]
    assert dips == pytest.approx(
        [quoted("frf_matrix", "1.97"), quoted("frf_matrix", "4.50"), quoted("frf_matrix", "5.15")], abs=0.01
    )


def test_frf_matrix_worked_example():
    """The Theory's worked example: H11 of the default chain from the closed-form modes, its static shares, the
    first peak, and the dips as the held 3-mass chain's frequencies, each from the figures it quotes too."""
    import scipy.linalg

    s = ChainSystem.uniform(4)
    M, C, K = s.matrices()
    w2, phi = scipy.linalg.eigh(K, M)
    gamma = np.radians([20, 60, 100, 140])
    p1 = 2 / 3 * np.sin(gamma)
    assert np.abs(phi[0]) == pytest.approx(np.abs(p1))
    assert p1 == pytest.approx([0.2280, 0.5774, 0.6565, 0.4285], abs=0.00005)
    assert p1**2 == pytest.approx([quoted("frf_matrix", "0.0520"), 0.3333, 0.4310, 0.1836], abs=0.00005)
    assert np.sum(p1**2) == pytest.approx(1)
    assert w2 == pytest.approx(
        [quoted("frf_matrix", "48.25"), 400, quoted("frf_matrix", "938.9"), quoted("frf_matrix", "1412.8")], abs=0.05
    )
    share = p1**2 / w2
    assert np.sum(share) == pytest.approx(1 / 400) == np.linalg.inv(K)[0, 0]
    assert share * 1e3 == pytest.approx(
        [
            quoted("frf_matrix", "1.078"),
            quoted("frf_matrix", "0.833"),
            quoted("frf_matrix", "0.459"),
            quoted("frf_matrix", "0.130"),
        ],
        abs=0.0005,
    )
    assert 100 * share * 400 == pytest.approx(
        [
            quoted("frf_matrix", "43.1"),
            quoted("frf_matrix", "33.3"),
            quoted("frf_matrix", "18.4"),
            quoted("frf_matrix", "5.2 %"),
        ],
        abs=0.05,
    )
    assert 100 * share[2:].sum() * 400 == pytest.approx(quoted("frf_matrix", "23.6 %"), abs=0.05)
    assert np.allclose(C, K / 200)
    zeta = np.diag(phi.T @ C @ phi) / (2 * np.sqrt(w2))
    assert zeta == pytest.approx(quoted("frf_matrix", "0.0025") * np.sqrt(w2))
    assert zeta[0] == pytest.approx(quoted("frf_matrix", "0.0174"), abs=0.00005)
    assert zeta[2:] == pytest.approx([quoted("frf_matrix", "0.08"), quoted("frf_matrix", "0.09")], abs=0.005)
    peak = 0.0520 / (2 * 0.0174 * 48.25)
    assert peak == pytest.approx(quoted("frf_matrix", "0.0310"), abs=0.00005)
    assert peak / 0.0025 == pytest.approx(quoted("frf_matrix", "12.4"), abs=0.05)
    h11 = abs(np.linalg.inv(K - w2[0] * M + 1j * np.sqrt(w2[0]) * C)[0, 0])
    assert h11 == pytest.approx(quoted("frf_matrix", "0.0311"), abs=0.00005)
    peaks_hz = np.sqrt(w2) / (2 * np.pi)
    shown = [
        quoted("frf_matrix", "1.105"),
        quoted("frf_matrix", "3.183"),
        quoted("frf_matrix", "4.877"),
        quoted("frf_matrix", "5.982"),
    ]
    assert peaks_hz == pytest.approx(shown, abs=0.0005)
    assert peaks_hz[2:] == pytest.approx([quoted("frf_matrix", "4.88"), quoted("frf_matrix", "5.98")], abs=0.005)
    held = 40 * np.sin(np.array([1, 3, 5]) * np.pi / 14)
    assert np.all((peaks_hz[:-1] < held / (2 * np.pi)) & (held / (2 * np.pi) < peaks_hz[1:]))  # they interlace
    assert held == pytest.approx(
        [quoted("frf_matrix", "8.90"), quoted("frf_matrix", "24.94"), quoted("frf_matrix", "36.04")], abs=0.005
    )
    assert held / (2 * np.pi) == pytest.approx(
        [quoted("frf_matrix", "1.417"), quoted("frf_matrix", "3.969"), quoted("frf_matrix", "5.736")], abs=0.0005
    )
    assert held**2 == pytest.approx(scipy.linalg.eigvalsh(K[1:, 1:], M[1:, 1:]))
    f = np.linspace(0.5, 7.0, 130001)
    h = np.abs(frf(s, f, 0)[:, 0])
    dips = [float(f[i]) for i in find_peaks(-np.log(h))[0]]
    assert dips == pytest.approx(
        [quoted("frf_matrix", "1.42"), quoted("frf_matrix", "3.99")], abs=0.005
    )  # the third is washed out
    assert len(find_peaks(np.log(h))[0]) == 3  # and so is the fourth peak
    assert abs(-((2 * np.pi * 50) ** 2) * frf(s, np.array([50.0]), 0)[0, 0]) == pytest.approx(1, abs=0.01)


def test_complex_modes_base_damper_peak():
    """A damper at the base of the default chain, its others taken out: mode 1's ζ peaks at 0.147, near 60 N·s/m."""
    from vib_tutorial.core.complex_modes import Parameter, sweep

    base = ChainSystem(np.ones(4), np.full(4, 400.0), np.array([20.0, 0, 0, 0]))
    value, zeta = sweep(base, Parameter.DAMPER, 0).peak(0)
    assert value == pytest.approx(60, abs=1) and zeta == pytest.approx(quoted("complex_modes", "0.147"), abs=0.0005)


def test_complex_modes_worked_example():
    """The Theory's worked example: one 20 N·s/m damper at the base of the default chain. The approximation's ζ
    from φ₁ᵣ², the exact poles and shapes, the first-order shape and second-order frequency of mode 1, and the
    locked damper's limit, each from the figures it quotes too."""
    from vib_tutorial.core.complex_modes import compare, complexity, free_decay
    from vib_tutorial.core.modal import modal_analysis
    from vib_tutorial.core.model import ChainSystem

    base = ChainSystem(np.ones(4), np.full(4, 400.0), np.array([20.0, 0, 0, 0]))
    r = modal_analysis(base)
    assert modal_analysis(ChainSystem.uniform(4)).is_proportional
    assert r.coupling == pytest.approx(1.0)  # a rank-one C_m
    phi = np.column_stack([m.shape_mass_normalized for m in r.modes])
    w = np.array([m.omega_n for m in r.modes])
    assert w == pytest.approx([quoted("complex_modes", "6.946"), 20.00, 30.64, 37.59], abs=0.005)
    assert w == pytest.approx(40 * np.sin(np.radians([10, 30, 50, 70])))
    phi2 = np.array([0.0520, 0.3333, 0.4310, 0.1836])
    assert phi[0] ** 2 == pytest.approx(phi2, abs=5e-5)
    cm = phi.T @ base.matrices()[1] @ phi
    assert cm == pytest.approx(20 * np.outer(phi[0], phi[0]))
    rows = compare(base, r)
    approx = [0.0748, 0.167, 0.141, 0.0489]
    assert [c.zeta_approx for c in rows] == pytest.approx(approx, rel=0.004)
    assert 10 * phi2 / np.array([6.946, 20.00, 30.64, 37.59]) == pytest.approx(approx, rel=0.004)
    assert [c.zeta_exact for c in rows] == pytest.approx(
        [
            quoted("complex_modes", "0.0726"),
            quoted("complex_modes", "0.178"),
            quoted("complex_modes", "0.168"),
            quoted("complex_modes", "0.0225"),
        ],
        rel=0.004,
    )
    assert [c.fn_exact for c in rows] == pytest.approx(
        [
            quoted("complex_modes", "1.124"),
            quoted("complex_modes", "3.434"),
            quoted("complex_modes", "4.569"),
            quoted("complex_modes", "5.823"),
        ],
        abs=0.0005,
    )
    assert [c.fn_hz for c in rows] == pytest.approx(
        [
            quoted("complex_modes", "1.105"),
            quoted("complex_modes", "3.183"),
            quoted("complex_modes", "4.877"),
            quoted("complex_modes", "5.982"),
        ],
        abs=0.0005,
    )
    # mode 1, first order: a_s1 = iω₁C_s1/(ω₁² − ω_s²)
    a = np.array([1j * w[0] * cm[s, 0] / (w[0] ** 2 - w[s] ** 2) for s in range(1, 4)])
    shown = [-quoted("complex_modes", "0.052"), -quoted("complex_modes", "0.023"), quoted("complex_modes", "0.010")]
    assert a == pytest.approx(1j * np.array(shown), abs=0.0005)
    guess = phi[:, 0] + phi[:, 1:] @ a
    assert complexity(guess).mpd == pytest.approx(quoted("complex_modes", "2.97"), abs=0.005)
    assert rows[0].measures.mpd == pytest.approx(quoted("complex_modes", "2.98"), abs=0.005)
    terms = [cm[0, s] ** 2 / (w[s] ** 2 - w[0] ** 2) for s in range(1, 4)]
    assert terms == pytest.approx([0.0197, 0.0101, 0.0028], abs=0.00005)
    assert w[0] * math.sqrt(1 + sum(terms)) == pytest.approx(7.058, abs=0.0005)
    assert math.sqrt(6.946**2 * (1 + 0.0197 + 0.0101 + 0.0028)) == pytest.approx(7.058, abs=0.0005)
    assert abs(rows[0].exact) == pytest.approx(quoted("complex_modes", "7.061"), abs=0.0005)
    assert rows[0].exact.real == pytest.approx(-cm[0, 0] / 2, rel=0.02)  # the decay rate kept to this order
    # modes 2 to 4: not small; mode 3 the most complex
    assert abs(1j * w[1] * cm[2, 1] / (w[1] ** 2 - w[2] ** 2)) == pytest.approx(
        quoted("complex_modes", "0.28"), abs=0.005
    )
    assert rows[2].measures.mpc == pytest.approx(quoted("complex_modes", "0.20"), abs=0.005)
    assert rows[2].measures.mpd == pytest.approx(quoted("complex_modes", "28.9"), abs=0.05)
    assert min(rows, key=lambda c: c.measures.mpc).index == 3
    assert free_decay(base, r, 1).leak[0] == pytest.approx(quoted("complex_modes", "45 %") / 100, abs=0.005)
    # locked at the base: the 3-mass chain above m1, H11's dips
    held = modal_analysis(ChainSystem(np.ones(3), np.full(3, 400.0), np.zeros(3)))
    assert [m.fn_hz for m in held.modes] == pytest.approx(
        [quoted("complex_modes", "1.42"), quoted("complex_modes", "3.97"), quoted("complex_modes", "5.74")], abs=0.005
    )


def test_continuum_worked_example():
    """The Theory's worked example: the guitar string (the page's default) on 12 elements, its speed and
    fundamental, modes 1 and 6 on each mesh, the 11 lumped modes under the cut-off, and its impedance."""
    from vib_tutorial.core.continuum import PRESETS, FEModel, Scheme, errors, flexible_hz

    p = PRESETS[0]
    c = p.continuum
    assert p.elements == 12 and c.tension == 73.0
    assert (c.length, 1e3 * c.line_density) == pytest.approx(
        (quoted("continuum", "0.648"), quoted("continuum", "6.40"))
    )
    assert math.sqrt(73 / 0.0064) == pytest.approx(106.8, abs=0.05) and c.speed == pytest.approx(106.8, abs=0.05)
    assert c.exact_hz(1)[0] == pytest.approx(quoted("continuum", "82.41"), abs=0.005)
    assert 106.8 / (2 * 0.648) == pytest.approx(quoted("continuum", "82.41"), abs=0.005)
    h = c.length / 12
    assert h == pytest.approx(quoted("continuum", "0.054"))
    assert math.pi / 12 == pytest.approx(quoted("continuum", "0.262"), abs=0.0005)
    assert (math.pi / 12) ** 2 / 24 == pytest.approx(quoted("continuum", "0.29 %") / 100, abs=0.00005)
    f = {s: flexible_hz(FEModel(c, 12, s)) for s in Scheme}
    err = {s: errors(FEModel(c, 12, s))[1] for s in Scheme}
    assert f[Scheme.LUMPED][0] == pytest.approx(quoted("continuum", "82.17"), abs=0.005)
    assert f[Scheme.CONSISTENT][0] == pytest.approx(quoted("continuum", "82.64"), abs=0.005)
    assert f[Scheme.MIXED][0] == pytest.approx(quoted("continuum", "82.41"), abs=0.005)
    assert 100 * abs(err[Scheme.MIXED][0]) == pytest.approx(quoted("continuum", "0.001 %"), abs=0.0005)
    assert 100 * err[Scheme.LUMPED][0] == pytest.approx(-0.29, abs=0.005)
    assert 100 * err[Scheme.CONSISTENT][0] == pytest.approx(0.29, abs=0.005)
    assert 6 * 82.407 == pytest.approx(494.4, abs=0.05)
    assert math.sin(math.pi / 4) / (math.pi / 4) == pytest.approx(0.900, abs=0.0005)
    assert 494.4 * 0.900 == pytest.approx(445, abs=0.5) and f[Scheme.LUMPED][5] == pytest.approx(445, abs=0.5)
    assert 100 * err[Scheme.LUMPED][5] == pytest.approx(-quoted("continuum", "10.0 %"), abs=0.05)
    assert 100 * err[Scheme.CONSISTENT][5] == pytest.approx(quoted("continuum", "10.3 %"), abs=0.05)
    assert 100 * err[Scheme.MIXED][5] == pytest.approx(-quoted("continuum", "1.4 %"), abs=0.05)
    assert f[Scheme.LUMPED].size == 11
    assert c.speed / (math.pi * h) == pytest.approx(quoted("continuum", "629.5"), abs=0.05)
    assert f[Scheme.LUMPED][10] == pytest.approx(quoted("continuum", "624.2"), abs=0.05)
    assert c.exact_hz(11)[10] == pytest.approx(quoted("continuum", "906.5"), abs=0.05)
    assert 100 * err[Scheme.LUMPED][10] == pytest.approx(-quoted("continuum", "31 %"), abs=0.5)
    assert math.sqrt(c.tension * c.line_density) == pytest.approx(quoted("continuum", "0.684"), abs=0.0005)
    # the notes' general claims: the cut-off error and a cantilever's ratios
    assert 2 / math.pi - 1 == pytest.approx(-quoted("continuum", "36 %") / 100, abs=0.005)
    import scipy.optimize

    roots = [scipy.optimize.brentq(lambda b: math.cos(b) * math.cosh(b) + 1, a, a + 2) for a in (1.0, 4.0, 7.0)]
    shown = [quoted("continuum", "1.875"), quoted("continuum", "4.694"), quoted("continuum", "7.855")]
    assert roots == pytest.approx(shown, abs=0.0005)  # a cantilever's βL: cos βL cosh βL = −1
    # ten elements a wavelength, κh ≈ 0.63, keep a string's lumped error, (κh)²/24, under 2 %
    assert 2 * math.pi / 10 == pytest.approx(quoted("continuum", "0.63"), abs=0.005)
    assert 100 * (2 * math.pi / 10) ** 2 / 24 < quoted("continuum", "2 %")
    assert [(b / 1.87510) ** 2 for b in (4.69409, 7.85476)] == pytest.approx(
        [quoted("continuum", "6.27"), quoted("continuum", "17.5")], abs=0.05
    )  # roots to 5 figures; 17.55 to 3


def test_beams_plates_worked_example():
    """The Theory's worked example: the page's default beam on mode 1, each theory's frequency, the first-order
    Timoshenko formula, the thickness-shear root and the wave speeds; then the trials' plate, from the figures
    it quotes too."""
    from vib_tutorial.core.beams_plates import (
        TWO_PI,
        Beam,
        Plate,
        Theory,
        beam_frequencies,
        beam_omega,
        rect_modes,
        wave_speeds,
    )
    from vib_tutorial.gui.beams_plates import STEEL, BeamPlateTrial

    t = BeamPlateTrial("", "")
    assert (t.length, t.depth, t.mode, t.E, t.rho) == (1.0, 0.1, 1, 210.0, 7850.0)
    assert t.nu == quoted("beams_plates", "0.3")
    assert t.length / t.depth == pytest.approx(10)  # L/h = 10
    b = Beam(1.0, 0.1, 0.05, STEEL)
    assert b.area == pytest.approx(quoted("beams_plates", "0.005"))
    assert b.inertia * 1e6 == pytest.approx(quoted("beams_plates", "4.167"), abs=5e-4)
    scale = math.sqrt(210e9 * b.inertia / (7850 * b.area))
    assert scale == pytest.approx(149.3, abs=0.05)
    assert math.pi**2 * 149.3 == pytest.approx(1474, abs=0.5)
    f = {th: float(beam_frequencies(b, 1, th)[0]) for th in Theory}
    assert f[Theory.EULER] == pytest.approx(quoted("beams_plates", "234.5"), abs=0.05)
    assert 1474 / TWO_PI == pytest.approx(quoted("beams_plates", "234.5"), abs=0.1)
    assert (math.pi * 0.1) ** 2 / 12 == pytest.approx(quoted("beams_plates", "0.00822"), abs=0.000005)
    assert f[Theory.RAYLEIGH] == pytest.approx(quoted("beams_plates", "233.6"), abs=0.05)
    G = STEEL.G
    assert G / 1e9 == pytest.approx(quoted("beams_plates", "80.8"), abs=0.05)
    assert 5 / 6 * G / 1e9 == pytest.approx(quoted("beams_plates", "67.3"), abs=0.05)
    assert 210 / 67.3 == pytest.approx(quoted("beams_plates", "3.12"), abs=0.005)
    assert f[Theory.SHEAR] == pytest.approx(quoted("beams_plates", "231.6"), abs=0.05)
    assert (math.pi * 0.1) ** 2 / 24 == pytest.approx(quoted("beams_plates", "0.00411"), abs=0.000005)
    assert 1 + 210 / 67.3 == pytest.approx(quoted("beams_plates", "4.12"), abs=0.005)  # 1 + E/κ_sG
    first = 1 - 0.00411 * 4.12
    assert first == pytest.approx(quoted("beams_plates", "0.9831"), abs=0.00005)
    assert first * 234.5 == pytest.approx(quoted("beams_plates", "230.5"), abs=0.05)
    assert 1 - (math.pi * 0.1) ** 2 / 24 * (1 + 210e9 / (5 / 6 * G)) == pytest.approx(
        f[Theory.TIMOSHENKO] / f[Theory.EULER], abs=1e-3
    )
    assert f[Theory.TIMOSHENKO] == pytest.approx(quoted("beams_plates", "230.7"), abs=0.05)
    assert 100 * (f[Theory.TIMOSHENKO] / f[Theory.EULER] - 1) == pytest.approx(
        -quoted("beams_plates", "1.64 %"), abs=0.005
    )
    assert 100 * (1 - f[Theory.TIMOSHENKO] / f[Theory.EULER]) < quoted("beams_plates", "2 %")  # mode 1 at L/h = 10
    assert float(beam_omega(b, math.pi, branch=1)) / TWO_PI / 1e3 == pytest.approx(
        quoted("beams_plates", "16.4"), abs=0.05
    )
    cut_off = math.sqrt(5 / 6 * G * b.area / (7850 * b.inertia)) / TWO_PI / 1e3
    assert cut_off == pytest.approx(quoted("beams_plates", "16.1"), abs=0.05)
    assert wave_speeds(b) == pytest.approx((5172, 2928), abs=0.5)
    # the bending wave's speed levels off at c_s, the thickness-shear branch's at c_b
    k = 1e4
    assert float(beam_omega(b, k)) / k == pytest.approx(2928, rel=1e-3)
    assert float(beam_omega(b, k, branch=1)) / k == pytest.approx(5172, rel=1e-3)

    p = Plate(0.6, 0.4, 0.003, STEEL)
    assert (p.a**2, p.b**2) == pytest.approx((quoted("beams_plates", "0.36"), quoted("beams_plates", "0.16")))
    assert p.a * p.b == pytest.approx(quoted("beams_plates", "0.24"))
    assert p.rigidity == pytest.approx(519, abs=0.5)
    assert p.surface_mass == pytest.approx(quoted("beams_plates", "23.55"), abs=0.005)
    assert p.scale == pytest.approx(quoted("beams_plates", "4.696"), abs=0.0005)
    assert math.pi**2 * (1 / 0.36 + 1 / 0.16) * 4.696 == pytest.approx(quoted("beams_plates", "418.4"), abs=0.05)
    assert rect_modes(p, 1)[0].freq_hz == pytest.approx(quoted("beams_plates", "66.6"), abs=0.05)
    assert 0.24 / 2 / 4.696 == pytest.approx(quoted("beams_plates", "0.026"), abs=0.0005)
    assert 0.24 / 2 / p.scale == pytest.approx(quoted("beams_plates", "0.026"), abs=0.0005)
    assert 1 / (0.24 / 2 / p.scale) == pytest.approx(39, abs=0.5)


def test_integration_worked_example():
    """The Theory's worked example: the page's default (1 kg on 400 N/m from 10 mm, ten steps a period), each
    scheme's step from its closed form, the phase after ten periods, and the readout's largest errors."""
    from vib_tutorial.core.integration import Load, Model, Scheme, exact, simulate, spectral
    from vib_tutorial.gui.integration import IntegrationTrial

    t = IntegrationTrial("", "")
    assert (t.mass, t.stiffness, t.damping, t.x0, t.per_period, t.periods, t.force) == (1, 400, 0, 0.01, 10, 10, 0)
    model, load = Model(), Load()
    w = math.sqrt(400)
    assert w == 20 and w / (2 * math.pi) == pytest.approx(quoted("integration", "3.18"), abs=0.005)
    T = 2 * math.pi / w
    h = T / 10
    assert T == pytest.approx(quoted("integration", "0.314"), abs=0.0005)
    assert h == pytest.approx(quoted("integration", "0.0314"), abs=0.00005)
    O = w * h
    assert O == pytest.approx(quoted("integration", "0.628"), abs=0.0005)
    assert O / 2 == pytest.approx(quoted("integration", "0.314"), abs=0.0005)
    cd, aa = 2 * math.asin(O / 2), 2 * math.atan(O / 2)
    assert cd == pytest.approx(quoted("integration", "0.639"), abs=0.0005)
    assert 0.628 / 0.639 == pytest.approx(quoted("integration", "0.983"), abs=0.0005)
    assert aa == pytest.approx(quoted("integration", "0.609"), abs=0.0005)
    sp = {s: spectral(s, O) for s in (Scheme.CENTRAL, Scheme.AVERAGE, Scheme.LINEAR, Scheme.RK4)}
    assert O / cd - 1 == pytest.approx(sp[Scheme.CENTRAL].period_error)
    assert 100 * sp[Scheme.CENTRAL].period_error == pytest.approx(-1.7, abs=0.05)
    assert O / aa - 1 == pytest.approx(sp[Scheme.AVERAGE].period_error)
    assert 100 * sp[Scheme.AVERAGE].period_error == pytest.approx(3.2, abs=0.05)
    assert 100 * O**2 / 12 == pytest.approx(quoted("integration", "3.3 %"), abs=0.05)
    assert 100 * sp[Scheme.LINEAR].period_error == pytest.approx(1.6, abs=0.05)
    assert sp[Scheme.RK4].radius == pytest.approx(quoted("integration", "0.99959"), abs=0.000005)
    assert 100 * sp[Scheme.RK4].period_error == pytest.approx(0.11, abs=0.005)
    assert 100 * sp[Scheme.RK4].damping == pytest.approx(0.065, abs=0.0005)
    rk = sum((1j * O) ** k / math.factorial(k) for k in range(5))
    assert abs(rk) ** 2 == pytest.approx(1 - O**6 / 72 + O**8 / 576)
    assert 1 - 8**3 / 72 + 8**4 / 576 == pytest.approx(1.0)  # ρ = 1 at ωΔt = 2√2
    # ten periods: 100 steps of phase
    assert 100 * aa == pytest.approx(quoted("integration", "60.9"), abs=0.05)
    assert 20 * math.pi == pytest.approx(quoted("integration", "62.8"), abs=0.05)
    assert 20 * math.pi - 100 * aa == pytest.approx(quoted("integration", "1.95"), abs=0.005)
    assert 1.95 / (2 * math.pi) == pytest.approx(quoted("integration", "0.31"), abs=0.005)
    assert 10 * math.cos(1.95) == pytest.approx(-quoted("integration", "3.7"), abs=0.05)
    assert 100 * cd - 20 * math.pi == pytest.approx(quoted("integration", "1.08"), abs=0.005)
    assert 10 * math.cos(1.08) == pytest.approx(quoted("integration", "4.7"), abs=0.05)
    assert 20 * math.pi - 100 * float(np.angle(rk)) == pytest.approx(quoted("integration", "0.07"), abs=0.005)
    assert abs(rk) ** 100 == pytest.approx(quoted("integration", "0.960"), abs=0.0005)
    steps = 100
    times = np.arange(steps + 1) * h
    reference = exact(model, load, times)
    largest = {}
    for s in sp:
        x = simulate(s, model, load, h, steps, t.alpha)
        largest[s] = 100 * float(np.max(np.abs(x - reference))) / float(np.max(np.abs(reference)))
        assert 1000 * x[-1] == pytest.approx(
            {
                Scheme.CENTRAL: quoted("integration", "4.7"),
                Scheme.AVERAGE: -quoted("integration", "3.7"),
                Scheme.LINEAR: 5.5,
                Scheme.RK4: quoted("integration", "9.6"),
            }[s],
            abs=0.05,
        )
    assert [round(largest[s], 1) for s in sp] == pytest.approx(
        [98.0, 164.5, quoted("integration", "91.9 %"), 7.6], abs=0.06
    )
    assert float(f"{largest[Scheme.AVERAGE]:.3g}") == quoted("integration", "165 %")  # the readout's 3 figures
    assert float(f"{largest[Scheme.CENTRAL]:.3g}") == quoted("integration", "98 %")
    assert float(f"{largest[Scheme.LINEAR]:.3g}") == quoted("integration", "91.9 %")
    assert float(f"{largest[Scheme.RK4]:.3g}") == quoted("integration", "7.58 %")


def test_random_vibration_worked_example():
    """The Theory's worked example: the default base shake on the default chain, Miles mode by mode from one
    common factor, the exact RMS, Rice's rates and the largest peak over the Statistics tab's stretch; then the
    test profile's Grms from its segments, each from the figures it quotes too."""
    from vib_tutorial.core.modal import modal_analysis
    from vib_tutorial.core.model import ChainSystem
    from vib_tutorial.core.random_vibration import Input, Output, Quantity, RandomInput, predict, rice
    from vib_tutorial.gui import random_figures
    from vib_tutorial.gui.random_vibration import RandomSetup

    t = RandomSetup()
    assert (t.kind, t.f_lo, t.f_hi, t.quantity, t.mass) == (Input.BASE, 0.0, 10.0, Quantity.ACCELERATION, -1)
    assert t.level == quoted("random_vibration", "0.01")
    # the shaker example: 0.01 g²/Hz from 20 to 2000 Hz
    assert 0.01 * (2000 - 20) == pytest.approx(quoted("random_vibration", "19.8"))
    assert math.sqrt(19.8) == pytest.approx(quoted("random_vibration", "4.4"), abs=0.05)
    # the record: 2.5 samples a period of the band's top
    from vib_tutorial.core.random_vibration import sample_rate

    assert sample_rate(RandomInput()) / RandomInput().f_hi == pytest.approx(quoted("random_vibration", "2.5"))
    # a Gaussian's ±σ, ±2σ, ±3σ
    within = [100 * math.erf(n / math.sqrt(2)) for n in (1, 2, 3)]
    assert within[:2] == pytest.approx(
        [quoted("random_vibration", "68.3 %"), quoted("random_vibration", "95.4 %")], abs=0.05
    )
    assert within[2] == pytest.approx(quoted("random_vibration", "99.73 %"), abs=0.005)
    # one block's PSD estimate scatters by 100 %: |X|² of white noise is exponential, its σ its mean
    x = np.random.default_rng(1).standard_normal(1 << 16)
    power = np.abs(np.fft.rfft(x)[1:-1]) ** 2
    assert 100 * power.std() / power.mean() == pytest.approx(quoted("random_vibration", "100 %"), rel=0.03)
    s = ChainSystem.uniform(4)
    inp = RandomInput()
    assert inp.rms == pytest.approx(quoted("random_vibration", "0.316"), abs=0.0005)
    r = modal_analysis(s)
    for m in r.modes:
        assert m.zeta_modal == pytest.approx(m.omega_n / 400)
        assert m.fn_hz / (2 * m.zeta_modal) == pytest.approx(200 / (2 * math.pi))
    assert 200 / (2 * math.pi) == pytest.approx(quoted("random_vibration", "31.83"), abs=0.005)
    factor = math.sqrt(math.pi / 2 * 31.83 * 0.01)
    assert factor == pytest.approx(0.707, abs=0.0005)
    shares = [abs(m.shape_mass_normalized.sum() * m.shape_mass_normalized[3]) for m in r.modes]
    assert shares == pytest.approx(
        [
            quoted("random_vibration", "1.241"),
            quoted("random_vibration", "0.333"),
            quoted("random_vibration", "0.120"),
            quoted("random_vibration", "0.028"),
        ],
        abs=0.0005,
    )
    p = predict(s, inp, Output(Quantity.ACCELERATION, 3), r)
    miles = [row[3] for row in p.miles]
    assert miles == pytest.approx(
        [
            quoted("random_vibration", "0.878"),
            quoted("random_vibration", "0.236"),
            quoted("random_vibration", "0.085"),
            quoted("random_vibration", "0.020"),
        ],
        abs=0.0005,
    )
    assert [0.707 * x for x in (1.241, 0.333, 0.120, 0.028)] == pytest.approx(
        [
            quoted("random_vibration", "0.878"),
            quoted("random_vibration", "0.236"),
            quoted("random_vibration", "0.085"),
            quoted("random_vibration", "0.020"),
        ],
        abs=0.001,
    )
    assert math.sqrt(0.878**2 + 0.236**2 + 0.085**2 + 0.020**2) == pytest.approx(
        quoted("random_vibration", "0.913"), abs=0.0005
    )
    assert p.miles_rms == pytest.approx(quoted("random_vibration", "0.913"), abs=0.0005)
    assert p.rms == pytest.approx(quoted("random_vibration", "0.911"), abs=0.0005)
    assert 100 * (p.miles_rms / p.rms - 1) == pytest.approx(quoted("random_vibration", "0.2 %"), abs=0.05)
    R = rice(p)
    assert (R.nu0, R.nu_peak, R.alpha, R.delta) == pytest.approx(
        (
            quoted("random_vibration", "1.40"),
            quoted("random_vibration", "2.47"),
            quoted("random_vibration", "0.565"),
            quoted("random_vibration", "0.43"),
        ),
        abs=0.005,
    )
    assert R.nu0 * 60 == pytest.approx(quoted("random_vibration", "83.8"), abs=0.05)
    assert R.peak_factor(60) == pytest.approx(quoted("random_vibration", "3.17"), abs=0.005)
    assert R.clumped_peak_factor(60) == pytest.approx(quoted("random_vibration", "3.05"), abs=0.005)
    assert R.peak_factor(60) * R.sigma == pytest.approx(quoted("random_vibration", "2.89"), abs=0.005)
    # Davenport's general figures
    from vib_tutorial.core.random_vibration import davenport

    assert [davenport(c) for c in (100, 3000, 270_000)] == pytest.approx(
        [quoted("random_vibration", "3.2"), quoted("random_vibration", "4.1"), quoted("random_vibration", "5.1")],
        abs=0.05,
    )

    # the test profile: segment areas G1 f1/(p + 1) [(f2/f1)^(p+1) − 1]
    def area(f1, g1, f2, g2):
        p = math.log(g2 / g1) / math.log(f2 / f1)
        return g1 * f1 / (p + 1) * ((f2 / f1) ** (p + 1) - 1)

    prof = random_figures.PROFILE
    assert (prof[0][1], prof[1][1]) == (quoted("random_vibration", "0.01"), quoted("random_vibration", "0.04"))
    areas = [area(*prof[i], *prof[i + 1]) for i in range(3)]
    assert areas == pytest.approx([quoted("random_vibration", "1.5"), quoted("random_vibration", "16.8"), 15.0])
    assert sum(areas) == pytest.approx(quoted("random_vibration", "33.3"))
    assert math.sqrt(33.3) == pytest.approx(quoted("random_vibration", "5.77"), abs=0.005)
    assert 10 * math.log10(2) == pytest.approx(quoted("random_vibration", "3.01"), abs=0.005)
    f = np.geomspace(20, 2000, 200001)
    g = np.interp(np.log(f), np.log([q[0] for q in prof]), np.log([q[1] for q in prof]))
    assert np.trapezoid(np.exp(g), f) == pytest.approx(quoted("random_vibration", "33.3"), rel=1e-4)


def test_fatigue_worked_example():
    """The Theory's worked example: the default stress on the default welded detail, the narrow-band damage
    rate from its rounded figures, Dirlik's and the record's lives; a notch and a mean stress; the Eurocode
    knee and cut-off; the FDS's time compression."""
    from vib_tutorial.core.fatigue import Curve, Stress, dirlik_damage, narrow_band_damage, rates, record
    from vib_tutorial.gui import fatigue_figures
    from vib_tutorial.gui.fatigue import FatigueTrial

    t = FatigueTrial("", "")
    assert (t.stress, t.b, t.s_ref, t.n_ref, t.duration, t.seed) == (Stress(), 3.0, 80.0, 2e6, 600.0, 1)
    s, c = Stress(), Curve()
    assert (s.f1, s.sigma1, s.sigma2) == (2.0, 20.0, 0.0) and 100 * s.zeta1 == pytest.approx(quoted("fatigue", "2 %"))
    assert c.K / 1e12 == pytest.approx(quoted("fatigue", "1.024"))
    r = rates(s)
    assert (r.nu0, r.nup, r.alpha2) == pytest.approx(
        (quoted("fatigue", "1.99"), quoted("fatigue", "2.09"), quoted("fatigue", "0.95")), abs=0.005
    )
    assert (2 * math.sqrt(2) * 20) ** 3 == pytest.approx(181_000, rel=0.001)
    assert math.gamma(2.5) == pytest.approx(1.329, abs=0.0005)
    nb = narrow_band_damage(s, c)
    assert nb * 1e7 == pytest.approx(quoted("fatigue", "4.68"), abs=0.005)
    assert 1.99 * 181_000 * 1.329 / 1.024e12 * 1e7 == pytest.approx(quoted("fatigue", "4.68"), abs=0.01)
    assert 1 / nb / 1e6 == pytest.approx(quoted("fatigue", "2.13"), abs=0.005)
    assert 2.13e6 / 86400 == pytest.approx(quoted("fatigue", "24.7"), abs=0.05)
    assert 1 / nb / 86400 == pytest.approx(quoted("fatigue", "24.7"), abs=0.05)
    assert 1 / dirlik_damage(s, c) / 86400 == pytest.approx(quoted("fatigue", "25.1"), abs=0.05)
    rf = record(s, 600.0, 1).damage_rate(c)
    assert 1 / rf / 86400 == pytest.approx(quoted("fatigue", "25.1"), abs=0.05)
    assert 100 * (nb / rf - 1) == pytest.approx(quoted("fatigue", "1.4 %"), abs=0.05)
    # a notch, and a mean stress
    kf = 1 + quoted("fatigue", "0.8") * (2.5 - 1)
    assert kf == pytest.approx(quoted("fatigue", "2.2")) and kf**3 == pytest.approx(quoted("fatigue", "10.6"), abs=0.05)
    assert 1 / nb / 86400 / kf**3 == pytest.approx(quoted("fatigue", "2.3"), abs=0.05)
    assert 100 / (1 - 200 / 600) == pytest.approx(150)
    assert 100 / (1 - (200 / 600) ** 2) == pytest.approx(quoted("fatigue", "112.5"))
    assert (fatigue_figures.S_AR, fatigue_figures.S_U, fatigue_figures.EXAMPLE) == (150.0, 600.0, (200.0, 100.0))
    # a range 10 % higher does 1.1^b times the damage
    higher = 1 + quoted("fatigue", "10 %") / 100
    assert higher == pytest.approx(quoted("fatigue", "1.1"))
    assert narrow_band_damage(Stress(sigma1=20.0 * higher), c) / nb == pytest.approx(higher**c.b)
    # the Eurocode's knee and cut-off from its slopes
    assert (2 / 5) ** (1 / 3) == pytest.approx(0.737, abs=0.0005)
    assert (5 / 100) ** (1 / 5) == pytest.approx(0.549, abs=0.0005)
    # time compression with b = 8
    assert 10 ** (2 / 8) == pytest.approx(quoted("fatigue", "1.78"), abs=0.005)
    assert 10 * math.log10(10 ** (2 / 8)) == pytest.approx(quoted("fatigue", "2.5"))


def test_quarter_car_worked_example():
    """The Theory's worked example: the default car on a class C road at 72 km/h, the road's white velocity,
    the two modes' estimates, the body's damping, the travel's short answer against the full model, the
    readout's RMS values, and a 2.7 m wheelbase's nulls."""
    from vib_tutorial.core.quarter_car import CLASSES, Car, Output, Road, response
    from vib_tutorial.gui.quarter_car import CarTrial

    t = CarTrial("", "")
    assert (t.grade, t.speed_kmh, t.sprung, t.unsprung, t.spring, t.damper, t.tyre) == ("C", 72, 300, 40, 20, 1500, 200)
    car, road = Car(), Road("C", 20.0)
    assert CLASSES["C"] == pytest.approx(256e-6)
    gv = 4 * math.pi**2 * 256e-6 * 0.1**2 * 20
    assert gv * 1e3 == pytest.approx(quoted("quarter_car", "2.02"), abs=0.005)
    assert t.damper / 1e3 == quoted("quarter_car", "1.5")  # kN·s/m
    f = np.array([0.5, 3.0, 20.0])
    assert (2 * math.pi * f) ** 2 * road.psd(f) == pytest.approx(np.full(3, gv))  # white velocity
    ks = 20e3 * 200e3 / 220e3
    assert ks / 1e3 == pytest.approx(quoted("quarter_car", "18.2"), abs=0.05)
    bounce, hop = car.modes()
    assert math.sqrt(18_200 / 300) / (2 * math.pi) == pytest.approx(quoted("quarter_car", "1.24"), abs=0.005)
    assert bounce == pytest.approx(quoted("quarter_car", "1.24"), abs=0.01)
    assert math.sqrt(220_000 / 40) / (2 * math.pi) == pytest.approx(quoted("quarter_car", "11.8"), abs=0.05)
    assert hop == pytest.approx(quoted("quarter_car", "11.8"), abs=0.1)
    assert car.body_zeta() == pytest.approx(quoted("quarter_car", "0.31"), abs=0.005)
    assert math.sqrt(2.02e-3 * 300 / 6000) * 1e3 == pytest.approx(quoted("quarter_car", "10.1"), abs=0.06)
    assert math.sqrt(gv * 300 / 6000) * 1e3 == pytest.approx(quoted("quarter_car", "10.1"), abs=0.05)
    r = response(car, road)
    assert r.rms[Output.BODY] == pytest.approx(quoted("quarter_car", "1.48"), abs=0.005)
    assert r.rms[Output.TRAVEL] * 1e3 == pytest.approx(quoted("quarter_car", "10.7"), abs=0.05)
    assert 100 * r.load_ratio == pytest.approx(24, abs=0.5) and r.weight == pytest.approx(3334, abs=0.5)
    # the travel's short answer: exact for a rigid wheel, whatever the spring
    rigid = Car(unsprung=40.0, tyre=1e12)
    for spring in (10e3, 20e3, 40e3):
        stiff = Car(spring=spring, tyre=1e12)
        assert response(stiff, road).rms[Output.TRAVEL] == pytest.approx(math.sqrt(gv * 300 / 6000), rel=0.01)
    assert response(rigid, road).rms[Output.TRAVEL] < r.rms[Output.TRAVEL]
    # the best damper for comfort (ζ ≈ 0.2 on the body) and for grip (ζ ≈ 0.5)
    from vib_tutorial.core.quarter_car import best_damper

    dampers = np.geomspace(200, 10_000, 200)
    for output, zeta in ((Output.BODY, quoted("quarter_car", "0.2")), (Output.TYRE, quoted("quarter_car", "0.5"))):
        best = Car(damper=best_damper(car, road, output, dampers))
        assert best.body_zeta() == pytest.approx(zeta, abs=0.05)
    # wheelbase filtering
    tau = quoted("quarter_car", "2.7") / 20
    assert tau == pytest.approx(quoted("quarter_car", "0.135"))
    assert 1 / (2 * tau) == pytest.approx(quoted("quarter_car", "3.7"), abs=0.05)
    assert 1 / tau == pytest.approx(quoted("quarter_car", "7.4"), abs=0.05)


def test_rigid_mounts_worked_example():
    """The Theory's worked example: the default box on level mounts, its inertias and six pure modes from the
    stiffnesses it quotes, and the unbalance at 1500 rpm reaching the floor through bounce alone; the focal tilt
    from its quadratic."""
    from vib_tutorial.core.rigid_mounts import Layout, modes, spectrum
    from vib_tutorial.gui.rigid_mounts import MountTrial

    t = MountTrial("", "")
    assert (t.mass, t.drop, t.offset, t.tilt, t.stiffness, t.rpm) == (20, 0, 0, 0, 50e3, 1500)
    assert (t.length, t.width, t.height) == (
        quoted("rigid_mounts", "0.6"),
        quoted("rigid_mounts", "0.5"),
        quoted("rigid_mounts", "0.6"),
    )
    assert (t.span, t.track) == (quoted("rigid_mounts", "0.25"), quoted("rigid_mounts", "0.12"))
    assert t.stiffness * t.shear_ratio / 1e3 == quoted("rigid_mounts", "12.5")  # k_s, kN/m
    assert (t.unbalance, t.at_x, t.at_z, t.loss) == (
        quoted("rigid_mounts", "0.002"),
        quoted("rigid_mounts", "0.1"),
        quoted("rigid_mounts", "0.1"),
        quoted("rigid_mounts", "0.1"),
    )
    assert t.length**2 == pytest.approx(quoted("rigid_mounts", "0.36"))
    L = t.layout
    assert L.body.inertia == pytest.approx((1.0167, quoted("rigid_mounts", "1.2"), 1.0167), abs=0.0001)
    assert 20 * (0.25 + 0.36) / 12 == pytest.approx(quoted("rigid_mounts", "1.017"), abs=0.0005)
    m = modes(L.body, L.mounts())
    assert m.frequencies == pytest.approx(
        [
            quoted("rigid_mounts", "7.96"),
            quoted("rigid_mounts", "7.96"),
            quoted("rigid_mounts", "8.47"),
            quoted("rigid_mounts", "9.79"),
            quoted("rigid_mounts", "15.9"),
            quoted("rigid_mounts", "16.2"),
        ],
        abs=0.05,
    )
    assert 100 * m.energy.max(axis=1) == pytest.approx(np.full(6, quoted("rigid_mounts", "100 %")))  # every mode pure
    tp = 2 * math.pi
    assert math.sqrt(4 * 12_500 / 20) / tp == pytest.approx(quoted("rigid_mounts", "7.96"), abs=0.005)
    assert 4 * 50_000 * 0.12**2 == pytest.approx(2880)
    assert math.sqrt(2880 / 1.017) / tp == pytest.approx(quoted("rigid_mounts", "8.47"), abs=0.005)
    assert 4 * 12_500 * (0.25**2 + 0.12**2) == pytest.approx(3845)
    assert math.sqrt(3845 / 1.017) / tp == pytest.approx(quoted("rigid_mounts", "9.79"), abs=0.005)
    assert math.sqrt(4 * 50_000 / 20) / tp == pytest.approx(quoted("rigid_mounts", "15.9"), abs=0.05)
    assert 4 * 50_000 * 0.25**2 == pytest.approx(12_500)
    assert math.sqrt(12_500 / 1.2) / tp == pytest.approx(quoted("rigid_mounts", "16.2"), abs=0.05)
    assert 0.002 * (tp * 25) ** 2 == pytest.approx(quoted("rigid_mounts", "49.3"), abs=0.05)
    r = 25 / 15.9155
    assert r == pytest.approx(quoted("rigid_mounts", "1.571"), abs=0.0005)
    assert r * r == pytest.approx(quoted("rigid_mounts", "2.467"), abs=0.0005)
    assert abs(1 + 0.1j) == pytest.approx(quoted("rigid_mounts", "1.005"), abs=0.0005)
    tr = abs(1 + 0.1j) / abs(1 - r * r + 0.1j)
    assert tr == pytest.approx(quoted("rigid_mounts", "0.683"), abs=0.0005)
    assert 1.005 / abs(1 - 2.467 + 0.1j) == pytest.approx(quoted("rigid_mounts", "0.683"), abs=0.001)
    assert abs(1 + 0.1j) / abs(1 - (25 / 7.9577) ** 2 + 0.1j) == pytest.approx(
        quoted("rigid_mounts", "0.113"), abs=0.0005
    )
    assert 25 / 7.96 == pytest.approx(quoted("rigid_mounts", "3.14"), abs=0.005)
    sp = spectrum(L, 0.1, t.drive, np.array([1500.0]))
    assert sp.applied[0] == pytest.approx(quoted("rigid_mounts", "49.3"), abs=0.05)
    assert sp.total[0] == pytest.approx(quoted("rigid_mounts", "33.7"), abs=0.05)
    assert 0.683 * 49.3 == pytest.approx(quoted("rigid_mounts", "33.7"), abs=0.05)
    # the focal tilt is the smaller root of d s² − t(1 − ρ)s + dρ = 0
    lay = Layout(drop=0.08)
    d, tt, rho = 0.08, lay.track, lay.shear_ratio
    s = (tt * (1 - rho) - math.sqrt((tt * (1 - rho)) ** 2 - 4 * d * d * rho)) / (2 * d)
    assert math.atan(s) == pytest.approx(lay.focus_tilt(), rel=1e-6)


def test_shock_worked_example():
    """The Theory's worked example: the default 10 g, 100 ms half-sine at Q = 10, its velocity change, the
    spectrum's impulse end, its peak and its stiff end, and the softest oscillator's stretch."""
    from vib_tutorial.core.shock import G, Pulse, Shape
    from vib_tutorial.gui.shock import ShockSetup, _peak_srs

    t = ShockSetup()
    assert (t.shape, t.peak, t.duration, t.q) == (Shape.HALF_SINE, 10.0, 100.0, 10.0)
    A, T = 10 * G, t.duration / 1e3
    assert T == quoted("shock", "0.1")
    assert A == pytest.approx(quoted("shock", "98.1"), abs=0.05)
    dv = 2 * A * T / math.pi
    assert dv == pytest.approx(quoted("shock", "6.24"), abs=0.005)
    p = Pulse()
    low, top = quoted("shock", "0.2"), quoted("shock", "8.25")
    assert low * T == pytest.approx(quoted("shock", "0.02")) and top * T == pytest.approx(quoted("shock", "0.825"))
    srs = _peak_srs(p, np.array([low, top, 500.0]))
    f = np.linspace(7.5, 9.0, 151)
    assert f[np.argmax(_peak_srs(p, f))] == pytest.approx(top, abs=0.05)  # the spectrum's peak
    assert 2 * math.pi * 0.2 * 6.24 == pytest.approx(quoted("shock", "7.84"), abs=0.005)
    assert 7.84 / G == pytest.approx(quoted("shock", "0.80"), abs=0.005)
    assert srs[0] == pytest.approx(quoted("shock", "0.74"), abs=0.005)
    assert srs[1] == pytest.approx(quoted("shock", "16.5"), abs=0.05)
    assert srs[1] / 10 == pytest.approx(quoted("shock", "1.65"), abs=0.005)
    assert srs[2] == pytest.approx(quoted("shock", "10.0"), abs=0.05)
    assert 6.24 / (2 * math.pi * 0.2) == pytest.approx(quoted("shock", "4.97"), abs=0.005)
    assert srs[0] * G / (2 * math.pi * low) ** 2 == pytest.approx(
        quoted("shock", "4.6"), abs=0.05
    )  # the damped stretch


def test_modal_coupling_a_small_mass_ratio_splits():
    """A mass ratio of only 1 % splits two tuned oscillators' modes by 10 %: Ω² = 1 + μ/2 ± √(μ + μ²/4)."""
    mu = quoted("coupling_notes", "1 %") / 100
    lo, hi = (math.sqrt(1 + mu / 2 + s * math.sqrt(mu + mu * mu / 4)) for s in (-1, 1))
    assert 100 * (hi - lo) == pytest.approx(quoted("coupling_notes", "10 %"), abs=0.05)  # of the tuned frequency


def test_substructuring_a_light_interior():
    """The notes' light interior: 8 masses, m1–m3 0.05 kg, the interface at m4, driven at A's first
    fixed-interface frequency, 10.9 Hz (m1–m3 on k1–k4 with m4 held)."""
    s = ChainSystem.uniform(8)
    s.masses[:3] = quoted("cms_notes", "0.05")
    M, _, K = s.matrices()
    held = np.sqrt(np.sort(np.linalg.eigvals(np.linalg.solve(M[:3, :3], K[:3, :3])).real)) / (2 * math.pi)
    assert held[0] == pytest.approx(quoted("cms_notes", "10.9"), abs=0.05)
    # the steady state there, driven at the tip: the true chain, the coupled reduced model (no modes in A, 1 in
    # B), and each interior re-solved with the coupled boundary motion imposed (the enhanced recovery)
    w = 2 * math.pi * held[0]
    force = np.eye(8)[7]

    def dynamic(M, C, K):
        return K - w * w * M + 1j * w * C

    M, C, K = s.matrices()
    x = np.linalg.solve(dynamic(M, C, K), force)
    model = component_mode_synthesis(s, [3], [0, 1])
    coupled = model.T @ np.linalg.solve(dynamic(model.M, model.C, model.K), model.T.T @ force)

    def enhanced(inside, boundary):
        ii, ib = np.ix_(inside, inside), np.ix_(inside, boundary)
        return -np.linalg.solve(dynamic(M[ii], C[ii], K[ii]), (K[ib] + 1j * w * C[ib]) @ coupled[boundary])

    tip = abs(x[7])
    assert 100 * abs(x[3]) / tip == pytest.approx(quoted("cms_notes", "0.012 %"), abs=5e-4)  # A holds it still
    assert 100 * abs(coupled[3]) / tip == pytest.approx(quoted("cms_notes", "9 %"), abs=0.5)
    assert 100 * (1 - abs(coupled[7]) / tip) == pytest.approx(quoted("cms_notes", "15 %"), abs=0.5)
    in_a, in_b = enhanced([0, 1, 2], [3]), enhanced([4, 5, 6], [3, 7])
    off_enhanced, off_coupled = abs(in_a[0] - x[0]), abs(coupled[0] - x[0])
    assert off_enhanced / off_coupled == pytest.approx(quoted("cms_notes", "4.9"), abs=0.05)
    assert 100 * off_enhanced / tip == pytest.approx(quoted("cms_notes", "11 %"), abs=0.5)
    assert 100 * off_coupled / tip == pytest.approx(quoted("cms_notes", "2.3 %"), abs=0.05)
    assert 100 * abs(in_b[2] - x[6]) / tip == pytest.approx(quoted("cms_notes", "1.7 %"), abs=0.05)
    assert 100 * abs(coupled[6] - x[6]) / tip == pytest.approx(quoted("cms_notes", "34 %"), abs=0.5)


def test_modal_coupling_absorber_damping_is_shared():
    s = preset("absorber").system
    machine = s.damping[0] / (2 * math.sqrt(s.stiffness[0] * s.masses[0]))
    assert machine == pytest.approx(quoted("coupling_notes", "0.025"), abs=5e-4)
    assert s.damping[1] == 0  # the absorber none
    r = modal_analysis(s)
    assert [m.zeta_modal for m in r.modes] == pytest.approx(
        [quoted("coupling_notes", "0.012"), quoted("coupling_notes", "0.012")], abs=0.0015
    )


def test_background_friction_trial_loses_5_mm_a_cycle_and_sticks_after_four():
    from vib_tutorial.gui.sim_trials import TRIALS

    note = next(t.note for t in TRIALS if t.name.startswith("Friction"))
    sim = Simulator(ChainSystem([1.0], [400.0], [0.01], [0.5]))
    sim.set_displacement(np.array([0.02]))
    out = []
    while sim.t < 2.0 - 1e-12:
        out.append(sim.advance(2.0 - sim.t)[:3])
    t, x, v = (np.concatenate(a)[:, 0] if a[0].ndim > 1 else np.concatenate(a) for a in zip(*out))
    turns = np.flatnonzero(v[:-1] * v[1:] < 0) + 1
    lost = 0.02 - np.abs(x[turns[1::2]])  # after each whole cycle
    np.testing.assert_allclose(np.diff(np.concatenate([[0.0], lost]))[:3], 0.005, atol=2e-4)
    still = t[np.flatnonzero(v != 0)[-1]]
    assert still == pytest.approx(1.3, abs=0.05) and abs(x[-1]) < 1.25e-3
    assert "= 5 mm" in note and "four cycles (1.3 s)" in note


def test_background_rayleigh_quotient_and_effective_masses():
    s = ChainSystem.uniform(4)
    M, _, K = s.matrices()
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.sqrt(v @ K @ v / (v @ M @ v)) / (2 * np.pi) == pytest.approx(quoted("background", "1.162"), abs=5e-4)
    assert modal_analysis(s).modes[0].omega_n / (2 * np.pi) == pytest.approx(quoted("background", "1.105"), abs=5e-4)
    gamma2 = [(m.shape_mass_normalized @ M @ np.ones(4)) ** 2 for m in modal_analysis(s).modes]
    quoted_masses = [3.57, quoted("background", "0.33"), quoted("background", "0.08"), quoted("background", "0.01")]
    np.testing.assert_allclose(gamma2, quoted_masses, atol=0.006)


def test_background_worked_example():
    """The Simulation notes' worked example: the default chain's modes in closed form, from the rounded figures."""
    s = ChainSystem.uniform(4)
    M, C, K = s.matrices()
    np.testing.assert_allclose(M, np.eye(4))
    np.testing.assert_allclose(np.diag(K), [800, 800, 800, 400])
    np.testing.assert_allclose(np.diag(K, 1), [-400, -400, -400])
    np.testing.assert_allclose(C, K / 200)
    r = modal_analysis(s)
    # the closed form: γ_r = (2r − 1)π/(2N + 1), ω_r = 2√(k/m) sin(γ_r/2), φ_r,i = 2 sin(iγ_r)/√((2N + 1)m)
    gammas = [(2 * k - 1) * math.pi / 9 for k in range(1, 5)]
    assert np.degrees(gammas) == pytest.approx([20, 60, 100, 140])
    w = [40 * math.sin(g / 2) for g in gammas]
    assert w == pytest.approx(
        [
            quoted("background", "6.946"),
            quoted("background", "20.00"),
            quoted("background", "30.64"),
            quoted("background", "37.59"),
        ],
        abs=0.005,
    )
    assert [m.omega_n for m in r.modes] == pytest.approx(w)
    assert np.array([6.946, 20.00, 30.64, 37.59]) / (2 * math.pi) == pytest.approx(
        [
            quoted("background", "1.105"),
            quoted("background", "3.183"),
            quoted("background", "4.877"),
            quoted("background", "5.982"),
        ],
        abs=0.001,
    )
    assert [m.fn_hz for m in r.modes] == pytest.approx(
        [
            quoted("background", "1.105"),
            quoted("background", "3.183"),
            quoted("background", "4.877"),
            quoted("background", "5.982"),
        ],
        abs=0.0005,
    )
    for k, (g, m) in enumerate(zip(gammas, r.modes)):
        shape = 2 * np.sin(np.arange(1, 5) * g) / 3
        assert np.abs(m.shape_mass_normalized) == pytest.approx(np.abs(shape)), k
    assert 2 / 3 * np.sin(np.radians(20 * np.arange(1, 5))) == pytest.approx(
        [
            quoted("background", "0.228"),
            quoted("background", "0.429"),
            quoted("background", "0.577"),
            quoted("background", "0.657"),
        ],
        abs=5e-4,
    )
    # damping: ζ_r = βω_r/2, exact; the damped frequencies
    assert np.diag(C, 1) == pytest.approx(-2.0) and quoted("background", "0.0025") == pytest.approx(1 / 200 / 2)  # β/2
    zetas = 0.0025 * np.array([6.946, 20.00, 30.64, 37.59])
    assert zetas == pytest.approx(
        [
            quoted("background", "0.0174"),
            quoted("background", "0.0500"),
            quoted("background", "0.0766"),
            quoted("background", "0.0940"),
        ],
        abs=5e-5,
    )
    assert [m.zeta_modal for m in r.modes] == pytest.approx([m.damped.zeta for m in r.modes])
    assert [m.zeta_modal for m in r.modes] == pytest.approx(
        [
            quoted("background", "0.0174"),
            quoted("background", "0.0500"),
            quoted("background", "0.0766"),
            quoted("background", "0.0940"),
        ],
        abs=5e-5,
    )
    fd = np.array([1.105, 3.183, 4.877, 5.982]) * np.sqrt(1 - np.array([0.0174, 0.0500, 0.0766, 0.0940]) ** 2)
    assert fd == pytest.approx(
        [
            quoted("background", "1.105"),
            quoted("background", "3.179"),
            quoted("background", "4.862"),
            quoted("background", "5.956"),
        ],
        abs=0.001,
    )
    assert [m.damped.fd_hz for m in r.modes] == pytest.approx(
        [
            quoted("background", "1.105"),
            quoted("background", "3.179"),
            quoted("background", "4.862"),
            quoted("background", "5.956"),
        ],
        abs=0.0005,
    )
    # the straight-line guess
    v = np.arange(1.0, 5.0)
    assert v @ K @ v == 1600 and v @ M @ v == 30
    assert math.sqrt(1600 / 30) == pytest.approx(quoted("background", "7.303"), abs=5e-4)
    assert 7.303 / (2 * math.pi) == pytest.approx(quoted("background", "1.162"), abs=5e-4)
    assert 100 * (1.162 / 1.105 - 1) == pytest.approx(quoted("background", "5 %"), abs=0.5)
    assert 1600 / 30 == pytest.approx(quoted("background", "53.33"), abs=0.005)
    assert 6.946**2 == pytest.approx(quoted("background", "48.25"), abs=0.005)
    assert 100 * (53.33 / 48.25 - 1) == pytest.approx(quoted("background", "10.5 %"), abs=0.05)
    # the ground: mode 1's participation and effective mass
    assert 0.228 + 0.429 + 0.577 + 0.657 == pytest.approx(quoted("background", "1.890"), abs=0.0015)
    assert 1.890**2 == pytest.approx(quoted("background", "3.57"), abs=0.005)
    assert 100 * 3.57 / 4 == pytest.approx(quoted("background", "89 %"), abs=0.5)
    # the tuned mass damper at μ = 5 %
    mu = quoted("background", "5 %") / 100
    assert math.sqrt(1 + 2 / mu) == pytest.approx(math.sqrt(41))
    assert math.sqrt(41) == pytest.approx(quoted("background", "6.4"), abs=0.05)


def test_frf_and_coupling_trials():
    from vib_tutorial.core.modal_coupling import coupled_model as model
    from vib_tutorial.gui.coupling_notes import TRIALS as COUPLING

    s = ChainSystem.uniform(4)
    _, _, K = s.matrices()
    phi = np.array([m.shape_mass_normalized for m in modal_analysis(s).modes]).T
    w2 = np.array([m.omega_n**2 for m in modal_analysis(s).modes])
    share = phi[0] ** 2 / w2 / np.linalg.inv(K)[0, 0]  # m1's static compliance, mode by mode
    assert share[2:].sum() == pytest.approx(0.24, abs=0.005)
    assert (phi[3, 0] ** 2 / w2[0]) / np.linalg.inv(K)[3, 3] == pytest.approx(0.89, abs=0.005)
    assert abs(phi[2, 1]) < 1e-12  # m3 is a node of mode 2
    errors = {}
    for t in COUPLING:
        system = t.system()
        comparisons = compare_coupled(
            model(system, t.split, t.mode_a, t.mode_b, t.a_mass, True), modal_analysis(system)
        )
        errors[t.name] = [100 * c.error for c in comparisons]
    np.testing.assert_allclose(errors["Two equal halves"][1], 4.9, atol=0.05)
    np.testing.assert_allclose(errors["A light half, tuned"], [0.2, 0.3], atol=0.05)
    np.testing.assert_allclose(errors["The wrong mass for A"], [2.9, -2.2], atol=0.06)
    assert errors["A mode left out (8 masses)"][0] == pytest.approx(-14, abs=0.5)


# --------------------------------------------------------------------- the One mass and Signals pages' trials
def test_single_oscillator_trials():
    import math

    from vib_tutorial.core.sdof import Oscillator, receptance

    osc = Oscillator()
    assert osc.natural_hz == pytest.approx(quoted("sdof", "3.18"), abs=0.005)
    assert osc.zeta == pytest.approx(quoted("sdof", "0.05"))
    assert math.exp(osc.log_decrement) == pytest.approx(quoted("sdof", "1.37"), abs=0.005)
    assert osc.log_decrement == pytest.approx(quoted("sdof", "0.315"), abs=0.0005)
    assert osc.time_constant == pytest.approx(1.0)
    slow = Oscillator(1, 400, 80).poles()
    assert sorted(slow.real) == pytest.approx([-74.6, -5.4], abs=0.05)
    two = Oscillator(1, 400, 80).poles().real.max()  # ζ = 2
    assert -two / osc.omega_n == pytest.approx(0.27, abs=0.005) and -1 / two == pytest.approx(0.19, abs=0.005)
    assert 1 / 20 == pytest.approx(quoted("sdof", "0.05"))  # critical: a double root at −ω_n
    static = 1e3 / osc.stiffness  # mm, for 1 N
    assert static == pytest.approx(quoted("sdof", "2.5")) and 2 * osc.zeta == pytest.approx(quoted("sdof", "0.1"))
    assert static / (2 * osc.zeta) == pytest.approx(25.0)  # mm at resonance, Q = 10
    lo, hi = osc.half_power_hz()
    assert (lo, hi) == pytest.approx((quoted("sdof", "3.01"), quoted("sdof", "3.33")), abs=0.005)
    beat = osc.natural_hz - 3.0
    assert beat == pytest.approx(0.18, abs=0.005) and 1 / beat == pytest.approx(5.5, abs=0.05)
    undamped = Oscillator(1, 400, 0)
    assert abs(receptance(undamped, np.array([3.0]))[0]) == pytest.approx(0.022, abs=0.0005)
    assert 1 / (2 * undamped.omega_n) == pytest.approx(0.025)  # F/2mω_n, m/s
    assert (1e-4 / 1) / (2 * 0.05) == pytest.approx(1e-3)
    assert 1 / (2 * osc.omega_n) == pytest.approx(0.025)  # the Nyquist circle's diameter 1/cω_n, m/N


def test_single_oscillator_worked_example_and_springs():
    """The figures Single oscillator's worked example and its springs quote, on the page's defaults."""
    from vib_tutorial.core.sdof import Oscillator, receptance

    osc = Oscillator()
    assert (osc.mass, osc.stiffness, osc.damping) == (1.0, 400.0, 2.0)
    assert osc.omega_n == pytest.approx(20.0) and osc.critical == pytest.approx(40.0)
    assert 100 * osc.zeta == pytest.approx(quoted("sdof", "5 %"))
    assert osc.omega_d == pytest.approx(quoted("sdof", "19.97"), abs=0.005)
    # 0.13 % below ω_n at ζ = 5 %, the most of the 0.5–5 % of most structures
    assert 100 * (1 - osc.omega_d / osc.omega_n) == pytest.approx(quoted("sdof", "0.13 %"), abs=0.005)
    assert osc.log_decrement == pytest.approx(quoted("sdof", "0.315"), abs=0.0005)
    omega = 2 * math.pi * 3.0
    r = omega / osc.omega_n
    assert omega == pytest.approx(quoted("sdof", "18.85"), abs=0.005)
    assert r == pytest.approx(quoted("sdof", "0.942"), abs=0.0005)
    assert 1 - r**2 == pytest.approx(quoted("sdof", "0.1117"), abs=0.00005)
    assert 2 * osc.zeta * r == pytest.approx(quoted("sdof", "0.0942"), abs=0.00005)
    assert math.hypot(1 - r**2, 2 * osc.zeta * r) == pytest.approx(0.1462, abs=0.00005)
    assert 2.5 / 0.1462 == pytest.approx(17.1, abs=0.05)  # mm, from the rounded numbers
    h = receptance(osc, np.array([3.0]))[0]
    assert abs(h) * 1e3 == pytest.approx(17.1, abs=0.05)  # mm, for 1 N
    assert abs(h) * osc.stiffness == pytest.approx(quoted("sdof", "6.84"), abs=0.005)
    assert -math.degrees(math.atan2(h.imag, h.real)) == pytest.approx(quoted("sdof", "40.1"), abs=0.05)
    assert math.degrees(math.atan2(0.0942, 0.1117)) == pytest.approx(
        quoted("sdof", "40.1"), abs=0.05
    )  # from the rounded numbers too
    assert 2 * osc.zeta * osc.natural_hz == pytest.approx(quoted("sdof", "0.32"), abs=0.005)
    # Two of its springs side by side, and end to end.
    assert Oscillator(1, 800, 0).natural_hz == pytest.approx(quoted("sdof", "4.50"), abs=0.005)
    assert Oscillator(1, 200, 0).natural_hz == pytest.approx(quoted("sdof", "2.25"), abs=0.005)


def test_transient_trials():
    from vib_tutorial.core.sdof import Oscillator
    from vib_tutorial.core.transient import PulseKind, peak_ratio, step_metrics

    osc = Oscillator()
    m = step_metrics(osc)
    assert m.overshoot == pytest.approx(0.85, abs=0.005)
    assert 2.5 * (1 + m.overshoot) == pytest.approx(quoted("transient", "4.64"), abs=0.005)
    assert m.peak_time == pytest.approx(quoted("transient", "0.157"), abs=0.0005)
    assert m.settling == pytest.approx(quoted("transient", "3.8"), abs=0.05)
    assert step_metrics(Oscillator(1, 400, 40)).settling == pytest.approx(0.29, abs=0.005)
    assert 2 * np.sin(np.pi / 20) == pytest.approx(0.31, abs=0.005)
    r = 0.26 / osc.period
    assert peak_ratio(0.0, PulseKind.HALF_SINE, np.array([r]))[0] == pytest.approx(
        quoted("transient", "1.77"), abs=0.005
    )
    assert peak_ratio(0.05, PulseKind.HALF_SINE, np.array([r]))[0] == pytest.approx(1.65, abs=0.005)
    r = 0.864 / osc.period
    assert r == pytest.approx(2.75, abs=0.005)
    assert peak_ratio(0.0, PulseKind.RAMP, np.array([r]))[0] - 1 == pytest.approx(0.08, abs=0.005)
    assert peak_ratio(0.05, PulseKind.RAMP, np.array([r]))[0] - 1 == pytest.approx(0.06, abs=0.005)


def test_transient_worked_example_and_pulses():
    """The figures Transient response's worked example quotes, on the page's defaults (a 1 N step), from the
    rounded numbers it shows too; and its pulses' closed forms."""
    from vib_tutorial.core.sdof import Oscillator, impulse_response, step_response
    from vib_tutorial.core.transient import Pulse, PulseKind, convolution, peak_ratio, pulse_response, step_metrics

    osc, pulse = Oscillator(), Pulse()
    assert (osc.mass, osc.stiffness, osc.damping) == (1.0, 400.0, 2.0)
    assert (pulse.kind, pulse.amplitude) == (PulseKind.STEP, 1.0)
    zw, wd = osc.zeta * osc.omega_n, osc.omega_d
    assert osc.natural_hz == pytest.approx(quoted("transient", "3.18"), abs=0.005)
    assert osc.period == pytest.approx(quoted("transient", "0.314"), abs=0.0005)
    assert osc.zeta == pytest.approx(quoted("transient", "0.05"))
    assert zw == pytest.approx(1.0)
    assert wd == pytest.approx(quoted("transient", "19.97"), abs=0.005)
    assert zw / wd == pytest.approx(quoted("transient", "0.0501"), abs=0.00005)
    # the partial fractions: (1/400)(1/s − (s + 2)/((s + 1)² + ω_d²)) is 1/(s(s² + 2s + 400))
    for s in (0.3 + 2j, 5.0, -1 + 7j):
        assert (1 / s - (s + 2) / ((s + 1) ** 2 + wd**2)) / 400 == pytest.approx(1 / (s * (s * s + 2 * s + 400)))
    t = np.linspace(0, 3, 301)
    closed = 2.5e-3 * (1 - np.exp(-t) * (np.cos(wd * t) + zw / wd * np.sin(wd * t)))
    np.testing.assert_allclose(step_response(osc, t), closed, atol=1e-12)
    # the overshoot
    m = step_metrics(osc)
    assert 0.05 * math.pi / math.sqrt(1 - 0.05**2) == pytest.approx(quoted("transient", "0.1573"), abs=0.00005)
    assert math.exp(-0.1573) == pytest.approx(quoted("transient", "0.854"), abs=0.0005)
    assert m.overshoot == pytest.approx(quoted("transient", "0.854"), abs=0.0005)
    assert 1e3 / osc.stiffness == pytest.approx(quoted("transient", "2.5"))  # mm, the static deflection
    assert 100 * m.overshoot == pytest.approx(quoted("transient", "85.4 %"), abs=0.05)
    assert 1 + 0.854 == pytest.approx(quoted("transient", "1.854"))
    assert 2.5 * 1.854 == pytest.approx(quoted("transient", "4.64"), abs=0.005)
    assert 2.5 * (1 + m.overshoot) == pytest.approx(quoted("transient", "4.64"), abs=0.005)
    assert math.pi / 19.97 == pytest.approx(quoted("transient", "0.157"), abs=0.0005)
    assert m.peak_time == pytest.approx(quoted("transient", "0.157"), abs=0.0005)
    assert m.settling_estimate == pytest.approx(4.0)
    assert m.settling == pytest.approx(quoted("transient", "3.8"), abs=0.05)
    band = quoted("transient", "2 %") / 100  # the settling band, which the envelope reaches at 0.02
    assert band == pytest.approx(quoted("transient", "0.02"))
    assert math.log(1 / band) == pytest.approx(3.9, abs=0.05)  # the 2 % rule: ln 50, rounded to 4
    assert step_metrics(Oscillator(1, 400, 2)).settling == pytest.approx(m.settling)
    # at 0.3 s, where the convolution is drawn
    from vib_tutorial.gui.transient import TransientTrial

    assert TransientTrial.at == quoted("transient", "0.3")  # its default
    assert wd * 0.3 == pytest.approx(quoted("transient", "5.9925"), abs=0.00005)
    assert math.cos(5.9925) == pytest.approx(quoted("transient", "0.95805"), abs=0.000005)
    assert math.sin(5.9925) == pytest.approx(-quoted("transient", "0.28661"), abs=0.000005)
    assert math.exp(-0.3) == pytest.approx(quoted("transient", "0.74082"), abs=0.000005)
    bracket = 1 - 0.74082 * (0.95805 - 0.0501 * 0.28661)
    assert bracket == pytest.approx(quoted("transient", "0.3009"), abs=0.00005)
    assert 2.5 * bracket == pytest.approx(quoted("transient", "0.752"), abs=0.0005)
    x = float(pulse_response(osc, pulse, np.array([0.0, 0.3]))[-1])
    assert x * 1e3 == pytest.approx(quoted("transient", "0.752"), abs=0.0005)
    assert convolution(osc, pulse, 0.3)[3] * 1e3 == pytest.approx(quoted("transient", "0.752"), abs=0.0005)
    # h and H
    assert osc.time_constant == pytest.approx(1.0)
    assert 2 * osc.zeta * osc.natural_hz == pytest.approx(quoted("transient", "0.318"), abs=0.0005)
    assert 1 / (osc.damping * osc.omega_n) == pytest.approx(0.025)
    # the impulse trial's first swing, 4.6 mm, a little under I/mω_n = 5 mm
    t = np.linspace(0, 0.1, 2001)
    assert 0.1 * impulse_response(osc, t).max() * 1e3 == pytest.approx(4.6, abs=0.05)
    assert 0.1 / osc.omega_n == pytest.approx(0.005)
    # the residual swing of an undamped pulse is |F(ω_n)|/mω_n: a rectangle's (F/k) 2|sin πr|
    undamped = Oscillator(1, 400, 0)
    for r in (0.3, 0.7, 1.0, 1.4):
        T = r * undamped.period
        tau = np.linspace(0, T, 20001)
        y = np.exp(-1j * undamped.omega_n * tau)
        spectrum = abs(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(tau)))  # F = 1 N
        after = undamped.period * np.linspace(r + 0.01, r + 1.01, 2001)
        swing = np.abs(pulse_response(undamped, Pulse(PulseKind.RECTANGLE, 1.0, T), np.concatenate([[0], after])))
        assert swing[1:].max() == pytest.approx(spectrum / undamped.omega_n, rel=1e-3, abs=1e-7)
        assert spectrum / undamped.omega_n == pytest.approx(2 * abs(math.sin(math.pi * r)) / 400, abs=1e-7)
    # the half-sine's worst: 1.77 times static near T = 0.8 T_n
    ratios = np.linspace(0.7, 0.9, 201)
    peaks = peak_ratio(0.0, PulseKind.HALF_SINE, ratios)
    assert peaks.max() == pytest.approx(quoted("transient", "1.77"), abs=0.005)
    assert ratios[peaks.argmax()] == pytest.approx(quoted("transient", "0.8"), abs=0.03)


def test_damping_models_trials():
    from vib_tutorial.core.damping_models import Matched, Model, coulomb_stop

    m = Matched()
    wn = m.omega_n
    assert m.energy_per_cycle(Model.VISCOUS, np.array([wn]))[0] == pytest.approx(0.0126, abs=5e-5)
    assert m.friction == pytest.approx(0.31, abs=0.005)
    assert m.energy_per_cycle(Model.VISCOUS, np.array([2 * wn]))[0] == pytest.approx(0.0251, abs=5e-5)
    assert m.energy_per_cycle(Model.ZENER, np.array([2 * wn]))[0] == pytest.approx(0.0107, abs=5e-5)
    assert 4 * m.friction / np.pi == pytest.approx(quoted("sdof", "0.4"), abs=0.005)
    assert 4 * m.friction / m.stiffness == pytest.approx(0.0031, abs=5e-5)
    half_cycles, where = coulomb_stop(m.stiffness, m.friction, 0.01)
    assert half_cycles == 6 and abs(where) == pytest.approx(0.00058, abs=5e-6)
    k0, k1, _ = m.zener()
    assert (k0, k0 + k1) == pytest.approx((364, 444), abs=0.5)


def test_damping_models_worked_example():
    """The figures the Single oscillator's damping laws' worked example quotes, on the page's defaults, from the rounded numbers it
    shows too; and its equivalent viscous damping and loss factor formulas."""
    from vib_tutorial.core.damping_models import Matched, Model, coulomb_decay, coulomb_stop
    from vib_tutorial.core.sdof import Oscillator

    m = Matched()
    assert (m.mass, m.stiffness, m.relax) == (1.0, 400.0, 1.0)
    assert (m.zeta, m.amplitude) == (quoted("sdof", "0.05"), quoted("sdof", "0.01"))
    wn = m.omega_n
    assert wn == pytest.approx(20.0)
    assert m.natural_hz == pytest.approx(quoted("sdof", "3.18"), abs=0.005)
    assert m.c == pytest.approx(2.0)
    U = 0.5 * m.stiffness * m.amplitude**2
    assert U == pytest.approx(0.020) and m.eta == pytest.approx(quoted("sdof", "0.1"))
    w = np.array([wn])
    for model in Model:
        assert m.energy_per_cycle(model, w)[0] * 1e3 == pytest.approx(quoted("sdof", "12.57"), abs=0.005)
        assert m.energy_per_cycle(model, w)[0] * 1e3 == pytest.approx(
            quoted("sdof", "12.6"), abs=0.05
        )  # the loops' readout
        K = m.dynamic_stiffness(model, w)[0]
        assert m.energy_per_cycle(model, w)[0] / (2 * math.pi * 0.5 * K.real * m.amplitude**2) == pytest.approx(
            K.imag / K.real
        )  # η = W_d/2πU = K''/K'
    assert 2 * math.pi * 0.1 * 20 == pytest.approx(quoted("sdof", "12.57"), abs=0.005)  # mJ, from the rounded numbers
    assert math.pi * 2 * 20 * 0.01**2 * 1e3 == pytest.approx(quoted("sdof", "12.57"), abs=0.005)
    # friction
    assert m.friction == pytest.approx(quoted("sdof", "0.314"), abs=0.0005)
    assert 12.57 / 40 == pytest.approx(quoted("sdof", "0.314"), abs=0.0005)
    assert 1e3 * m.friction == pytest.approx(314, abs=0.5)  # the Damping box's readout, 314 mN
    assert 1e3 * m.zener()[2] == pytest.approx(quoted("sdof", "45.2"), abs=0.05)
    assert abs(coulomb_stop(m.stiffness, m.friction, m.amplitude)[1]) * 1e6 == pytest.approx(575, abs=0.5)
    assert 4 * m.friction / m.stiffness * 1e3 == pytest.approx(quoted("sdof", "3.14"), abs=0.005)
    assert m.friction / m.stiffness * 1e3 == pytest.approx(quoted("sdof", "0.785"), abs=0.0005)
    half_cycle = quoted("sdof", "1.57")
    assert 2 * m.friction / m.stiffness * 1e3 == pytest.approx(half_cycle, abs=0.005)
    turns = [10 - half_cycle * n for n in range(1, 7)]  # from the rounded 1.57 mm a half cycle
    assert turns == pytest.approx(
        [
            quoted("sdof", "8.43"),
            quoted("sdof", "6.86"),
            quoted("sdof", "5.29"),
            quoted("sdof", "3.72"),
            quoted("sdof", "2.15"),
            quoted("sdof", "0.58"),
        ],
        abs=0.005,
    )
    halves, stop = coulomb_stop(m.stiffness, m.friction, m.amplitude)
    assert halves == 6 and abs(stop) * 1e3 == pytest.approx(quoted("sdof", "0.58"), abs=0.005)
    t = np.linspace(0, 2.5 * 2 * math.pi / wn, 50001)  # its turns, every half period: 8.43, 6.86, … mm
    x = coulomb_decay(m.mass, m.stiffness, m.friction, m.amplitude, t)
    half = np.searchsorted(t, [n * math.pi / wn for n in range(1, 5)])
    assert np.abs(x[half]) * 1e3 == pytest.approx(
        [
            quoted("sdof", "8.43"),
            quoted("sdof", "6.86"),
            quoted("sdof", "5.29"),
            quoted("sdof", "3.72"),
        ],
        abs=0.005,
    )
    delta = Oscillator(1, 400, 2).log_decrement
    assert delta == pytest.approx(quoted("sdof", "0.315"), abs=0.0005)
    assert math.exp(-delta) == pytest.approx(quoted("sdof", "0.730"), abs=0.0005)
    assert math.exp(-0.315) == pytest.approx(quoted("sdof", "0.730"), abs=0.0005)
    # driven by 1 N
    assert abs(m.receptance(Model.HYSTERETIC, np.array([m.natural_hz]), 1.0)[0]) * 1e3 == pytest.approx(
        quoted("sdof", "25.0")
    )
    f = np.linspace(3.0, 3.3, 300001)
    assert np.abs(m.receptance(Model.VISCOUS, f, 1.0)).max() * 1e3 == pytest.approx(quoted("sdof", "25.03"), abs=0.005)
    assert 2.5 / (0.1 * math.sqrt(1 - 0.05**2)) == pytest.approx(quoted("sdof", "25.03"), abs=0.005)
    assert np.abs(m.receptance(Model.HYSTERETIC, f, 1.0)).max() == pytest.approx(0.025, rel=1e-6)
    assert 4 * m.friction / math.pi == pytest.approx(quoted("sdof", "0.4"))
    # the Zener element
    k0, k1, tau = m.zener()
    a = k1 / k0
    assert a == pytest.approx(quoted("sdof", "0.221"), abs=0.0005)
    assert a / (2 * math.sqrt(1 + a)) == pytest.approx(quoted("sdof", "0.1"))
    assert 1 + a == pytest.approx(quoted("sdof", "1.221"), abs=0.0005)
    assert 0.221 / (2 * math.sqrt(1.221)) == pytest.approx(
        quoted("sdof", "0.1"), abs=0.0005
    )  # η_max = k1/2√(k0(k0 + k1))
    assert k1 / (2 * math.sqrt(k0 * (k0 + k1))) == pytest.approx(quoted("sdof", "0.1"))
    assert k0 == pytest.approx(quoted("sdof", "363.8"), abs=0.05)
    assert k1 == pytest.approx(quoted("sdof", "80.4"), abs=0.05)
    assert 400 / (1 + 0.221 / 2.221) == pytest.approx(quoted("sdof", "363.8"), abs=0.05)
    assert tau * 1e3 == pytest.approx(quoted("sdof", "45.2"), abs=0.05)
    assert 1e3 / (20 * math.sqrt(1.221)) == pytest.approx(quoted("sdof", "45.2"), abs=0.05)
    assert m.loss_factor(Model.ZENER, w)[0] == pytest.approx(quoted("sdof", "0.1"))
    assert m.loss_factor(Model.ZENER, np.geomspace(0.1, 10, 2001) * wn).max() == pytest.approx(
        quoted("sdof", "0.1"), rel=1e-5
    )
    # equivalent viscous damping: velocity-squared drag a ẋ|ẋ| takes (8/3) a Ω² X³ a cycle
    drag, omega, X = 0.7, 13.0, 0.02
    th = np.linspace(0, 2 * math.pi, 400001)
    v = X * omega * np.cos(th)
    loss = np.trapezoid(drag * v * np.abs(v) * v, th / omega)  # ∮F dx = ∫F ẋ dt
    assert loss == pytest.approx(8 / 3 * drag * omega**2 * X**3, rel=1e-6)
    assert loss / (math.pi * omega * X**2) == pytest.approx(8 * drag * omega * X / (3 * math.pi), rel=1e-6)


def test_isolation_trials():
    from vib_tutorial.core.isolation import Isolator, Mount, deflection_for

    soft = Isolator()
    X, floor, T = soft.at_speed()
    assert soft.natural_hz == pytest.approx(7.05, abs=0.005)
    assert float(soft.force_at(25.0)) == pytest.approx(247, abs=0.5) and floor == pytest.approx(23, abs=0.5)
    assert T == pytest.approx(0.09, abs=0.005) and 20 * np.log10(T) == pytest.approx(-21, abs=0.5)
    assert X == pytest.approx(109e-6, abs=0.5e-6)
    assert deflection_for(25.0, 0.9) == pytest.approx(4.37e-3, abs=5e-6)
    stiff = Isolator(deflection=3e-4)
    assert stiff.natural_hz == pytest.approx(28.8, abs=0.05) and stiff.at_speed()[2] == pytest.approx(3.9, abs=0.05)
    assert Isolator(zeta=0.3).at_speed()[2] == pytest.approx(0.20, abs=0.005)
    assert (soft.natural_hz / soft.speed_hz) ** 2 == pytest.approx(1 / 12, abs=0.005)
    f = np.linspace(1, 30, 20001)
    relax = Isolator(zeta=0.3, mount=Mount.RELAXATION, rpm=3000)
    plain = Isolator(zeta=0.3, rpm=3000)
    assert relax.transmissibility(f).max() == pytest.approx(2.1, abs=0.05)
    assert plain.transmissibility(f).max() == pytest.approx(2.0, abs=0.05)
    assert (relax.at_speed()[2], plain.at_speed()[2]) == pytest.approx((0.070, 0.088), abs=0.0005)
    fast = (Isolator(zeta=0.3, mount=Mount.RELAXATION, rpm=6000), Isolator(zeta=0.3, rpm=6000))
    assert [i.at_speed()[2] for i in fast] == pytest.approx([0.019, 0.043], abs=0.0005)
    assert Isolator(mount=Mount.TWO_STAGE).at_speed()[2] == pytest.approx(0.038, abs=0.0005)
    assert Isolator(mount=Mount.TWO_STAGE, rpm=3000).at_speed()[2] == pytest.approx(0.002, abs=0.0005)
    assert Isolator(rpm=3000).at_speed()[2] == pytest.approx(0.025, abs=0.0005)
    assert Isolator(block=300).at_speed()[0] == pytest.approx(27e-6, abs=0.5e-6)
    # The floor shakes instead: ±0.1 mm at 25 Hz moves the machine ±9.16 µm, T = 0.0916 (the force's share)
    from vib_tutorial.view_models.live_isolation import FLOOR_AMPLITUDE

    assert FLOOR_AMPLITUDE == 0.1e-3 and soft.speed_hz == 25.0
    assert FLOOR_AMPLITUDE * T == pytest.approx(9.16e-6, abs=0.005e-6) and T == pytest.approx(0.0916, abs=5e-5)


def test_isolation_worked_example():
    """The figures Vibration isolation's worked example quotes, on the page's defaults, from the rounded
    numbers it shows too; and the base-motion and hysteretic transmissibilities its notes give."""
    from vib_tutorial.core.isolation import G, Isolator, start_up

    iso = Isolator()
    assert (iso.mass, iso.rpm, iso.unbalance) == (100.0, 1500.0, 0.01)
    assert (iso.deflection, iso.zeta) == (quoted("isolation", "0.005"), quoted("isolation", "0.05"))
    assert G == pytest.approx(quoted("isolation", "9.807"), abs=0.0005)
    assert iso.stiffness == pytest.approx(196_100, abs=50) and 100 * 9.807 / 0.005 == pytest.approx(196_100, abs=50)
    assert iso.natural_hz == pytest.approx(quoted("isolation", "7.048"), abs=0.0005)
    assert 15.76 / math.sqrt(5) == pytest.approx(quoted("isolation", "7.048"), abs=0.0005)
    assert 60 * iso.natural_hz == pytest.approx(423, abs=0.5)
    assert iso.damping == pytest.approx(quoted("isolation", "442.9"), abs=0.05)
    assert math.sqrt(G / 0.001) / (2 * math.pi) == pytest.approx(quoted("isolation", "15.76"), abs=0.005)
    assert 2 * math.pi * 25 == pytest.approx(quoted("isolation", "157.08"), abs=0.005)
    F = float(iso.force_at(25.0))
    assert F == pytest.approx(quoted("isolation", "246.7"), abs=0.05)
    assert 0.01 * 157.08**2 == pytest.approx(quoted("isolation", "246.7"), abs=0.05)
    r = 25 / iso.natural_hz
    assert r == pytest.approx(quoted("isolation", "3.547"), abs=0.0005)
    assert 25 / 7.048 == pytest.approx(quoted("isolation", "3.547"), abs=0.0005)
    assert 3.547**2 == pytest.approx(quoted("isolation", "12.58"), abs=0.005)
    assert 0.1 * 3.547 == pytest.approx(quoted("isolation", "0.3547"))
    assert math.sqrt(1 + 0.3547**2) == pytest.approx(1.0610, abs=0.00005)
    assert 1 - r**2 == pytest.approx(-quoted("isolation", "11.58"), abs=0.005)
    assert math.hypot(11.58, 0.3547) == pytest.approx(quoted("isolation", "11.585"), abs=0.0005)
    assert 1.0610 / 11.585 == pytest.approx(quoted("isolation", "0.0916"), abs=0.00005)
    X, floor, T = iso.at_speed()
    assert T == pytest.approx(quoted("isolation", "0.0916"), abs=0.00005)
    assert floor == pytest.approx(quoted("isolation", "22.6"), abs=0.05)
    assert 246.7 * 0.0916 == pytest.approx(quoted("isolation", "22.6"), abs=0.05)
    assert 100 * (1 - T) == pytest.approx(quoted("isolation", "90.8 %"), abs=0.05)
    assert 20 * math.log10(0.0916) == pytest.approx(-quoted("isolation", "20.8"), abs=0.05)
    assert X * 1e6 == pytest.approx(109, abs=0.5) and 100 * 12.58 / 11.585 == pytest.approx(109, abs=0.5)
    assert (iso.natural_hz / 25) ** 2 == pytest.approx(quoted("isolation", "0.0795"), abs=0.00005)
    assert (7.048 / 25) ** 2 * 246.7 == pytest.approx(quoted("isolation", "19.6"), abs=0.05)
    assert 19.6 / 0.1 == pytest.approx(196)
    s = start_up(iso, 2.0, 1.0)
    assert np.abs(s.floor[s.t <= 2.0]).max() == pytest.approx(111, abs=0.5)
    # the static-deflection examples and 90 % isolation
    for mm, hz in ((1, quoted("isolation", "15.8")), (5, 7.0), (25, quoted("isolation", "3.2"))):
        assert Isolator(deflection=mm * 1e-3).natural_hz == pytest.approx(hz, abs=0.05)
    T = 1 - quoted("isolation", "90 %") / 100
    assert T == pytest.approx(quoted("isolation", "0.1")) and 1 + 1 / T == pytest.approx(11)
    assert math.sqrt(11) == pytest.approx(quoted("isolation", "3.3"), abs=0.05)
    assert 25 / math.sqrt(11) == pytest.approx(quoted("isolation", "7.5"), abs=0.05)
    assert (15.76 / 7.54) ** 2 == pytest.approx(quoted("isolation", "4.4"), abs=0.05)
    # base motion: M x'' + c x' + k x = k y + c y' passes |X|/|Y| = T
    k, c, M = iso.stiffness, iso.damping, iso.mass
    for f in (2.0, 7.0, 25.0, 80.0):
        w = 2 * math.pi * f
        assert abs((k + 1j * w * c) / (k - w * w * M + 1j * w * c)) == pytest.approx(
            float(iso.transmissibility(np.array([f]))[0])
        )
    # far above resonance: √(1 + (2ζr)²)/r², 2ζ/r once 2ζr ≫ 1; a hysteretic mount keeps falling as 1/r²
    # 2ζ/r from r ≫ 1/2ζ: about 10 f_n up at ζ = 0.05, 2 f_n at ζ = 0.3
    assert 1 / (2 * iso.zeta) == pytest.approx(10) and 1 / (2 * quoted("isolation", "0.3")) == pytest.approx(2, abs=0.4)
    for r in (30.0, 300.0):
        f = r * iso.natural_hz
        assert float(iso.transmissibility(np.array([f]))[0]) == pytest.approx(
            math.sqrt(1 + (2 * iso.zeta * r) ** 2) / r**2, rel=2e-3
        )
    assert math.sqrt(1 + 30.0**2) / 300**2 == pytest.approx(0.1 / 300, rel=1e-3)
    eta, r = 0.1, 30.0
    hyst = abs((1 + 1j * eta) / (1 - r * r + 1j * eta))
    assert hyst == pytest.approx(math.sqrt(1 + eta**2) / math.hypot(1 - r * r, eta))
    assert hyst * r * r == pytest.approx(1, abs=0.01)
    # the machine's motion tends to m_e e/M
    assert Isolator(rpm=60 * 50 * iso.natural_hz).at_speed()[0] == pytest.approx(
        quoted("isolation", "0.01") / 100, rel=1e-3
    )


def test_signals_trials():
    from vib_tutorial.core.signals import (
        Kind,
        Source,
        Window,
        mass_loaded,
        nearest_line_reading,
        sample,
        usable_band,
        window_props,
    )
    from vib_tutorial.core.signals import spectrum as averaged

    def reading(source, f, window, block=1024, averages=1, filtered=True):
        n = block + (averages - 1) * block // 2
        _, x = sample(source, 1024.0, n, filtered)
        return nearest_line_reading(averaged(x, 1024.0, block, window, quoted("signals", "50 %") / 100), f)[1]

    assert reading(Source(f1=50.0, a2=0.0), 50.0, Window.RECTANGULAR) == pytest.approx(
        quoted("signals", "1.000"), abs=5e-4
    )
    pair = Source(f1=50.5, f2=70.0, a2=0.001)
    assert reading(pair, 50.5, Window.RECTANGULAR) == pytest.approx(0.64, abs=0.005)
    assert reading(pair, 70.0, Window.RECTANGULAR) == pytest.approx(0.015, abs=0.0005)
    assert reading(pair, 50.5, Window.HANN) == pytest.approx(0.85, abs=0.005)
    assert reading(pair, 70.0, Window.HANN) == pytest.approx(0.0010, abs=0.00005)
    assert reading(pair, 50.5, Window.FLAT_TOP) == pytest.approx(0.999, abs=0.0005)
    flat = window_props(Window.FLAT_TOP)
    assert flat.main_lobe == 5 and flat.enbw == pytest.approx(3.8, abs=0.05)
    alias = Source(f1=900.0, a2=0.0)
    assert reading(alias, 124.0, Window.HANN, filtered=False) == pytest.approx(quoted("signals", "1.000"), abs=5e-4)
    assert reading(alias, 124.0, Window.HANN) == pytest.approx(0.0015, abs=0.00005)
    assert 1024 / 2.56 == 400
    assert 0.5**2 / 512 == pytest.approx(4.9e-4, abs=5e-6)
    hann = window_props(Window.HANN)
    for block, level in ((4096, 0.019), (256, 0.077)):
        assert 2 * 0.5 * np.sqrt(hann.enbw / block) == pytest.approx(level, abs=0.0005)
    assert 0.05 / 0.019 == pytest.approx(2.6, abs=0.05)
    ringing = Source(Kind.RINGING, f1=50.0, zeta=0.01)
    assert 2 * ringing.zeta * ringing.f1 == pytest.approx(1.0) and 1024 / 4096 == 0.25
    assert usable_band(0.01) == pytest.approx(quoted("signals", "0.22"), abs=0.005)
    assert 0.218 * 30000 == pytest.approx(6.5e3, abs=60)
    assert usable_band(0.7, accelerometer=False) == pytest.approx(quoted("signals", "1.7"), abs=0.05)
    assert usable_band(0.7, accelerometer=False) * quoted("signals", "4.5") == pytest.approx(
        7.6, abs=0.05
    )  # a geophone's
    assert usable_band(0.7) == pytest.approx(quoted("signals", "0.6"), abs=0.01)
    assert quoted("signals", "0.7") == 0.7  # ζ ≈ 0.7: the flat band
    assert usable_band(0.01, quoted("signals", "5 %") / 100) == usable_band(0.01)  # within 5 %
    assert 100 * (1 - mass_loaded(200.0, 0.5, 0.03) / 200) == pytest.approx(quoted("signals", "2.9 %"), abs=0.05)
    assert -window_props(Window.RECTANGULAR).scalloping_db == pytest.approx(quoted("signals", "3.9"), abs=0.05)
    # one block's line power of noise scatters by 100 %
    noise = np.abs(np.fft.rfft(np.random.default_rng(2).standard_normal(1 << 16))[1:-1]) ** 2
    assert 100 * noise.std() / noise.mean() == pytest.approx(quoted("signals", "100 %"), rel=0.03)


def test_periodic_trials():
    import math

    from vib_tutorial.core.periodic import (
        Forcing,
        Wave,
        periodic_steady,
        response_coefficients,
        series,
        series_error,
    )
    from vib_tutorial.core.sdof import Oscillator

    osc = Oscillator()
    t = np.linspace(0.0, 1.0, 20001)
    sq = Forcing(Wave.SQUARE)
    assert np.max(series(sq.coefficients(5), 1.0, t)) - 1 == pytest.approx(0.19, abs=0.005)
    fine = np.linspace(0.0, 0.01, 20001)
    assert np.max(series(sq.coefficients(401), 1.0, fine)) - 1 == pytest.approx(quoted("periodic", "0.18"), abs=0.005)
    third = Forcing(Wave.SQUARE, freq_hz=osc.natural_hz / 3)
    assert 4 / (3 * math.pi) == pytest.approx(0.42, abs=0.005) and 1 / (2 * osc.omega_n) == pytest.approx(0.025)
    X = np.abs(response_coefficients(osc, third, 3))
    assert (X[3], X[1]) == pytest.approx((0.0106, 0.0036), abs=5e-5)

    def swing(fo):
        return float(np.max(periodic_steady(osc, fo, t * fo.period)))

    assert swing(third) == pytest.approx(0.0139, abs=5e-5)
    assert swing(Forcing(Wave.SQUARE, freq_hz=osc.natural_hz / 2)) == pytest.approx(0.0050, abs=5e-5)
    blows = periodic_steady(osc, Forcing(Wave.IMPULSES, 0.1), t)
    assert (blows.min(), blows.max()) == pytest.approx((-0.0045, 0.0052), abs=5e-5)
    assert 0.1 / 400 == pytest.approx(0.00025) and osc.time_constant == pytest.approx(1.0)
    assert 2 / math.pi == pytest.approx(0.64, abs=0.005) and 2 / math.pi / 400 == pytest.approx(0.00159, abs=5e-6)
    tri, sq1 = Forcing(Wave.TRIANGLE), Forcing(Wave.SQUARE)
    assert series_error(osc, tri, 3) == pytest.approx(0.019, abs=0.0005)
    assert series_error(osc, sq1, 3) == pytest.approx(0.057, abs=0.0005)
    assert series_error(osc, tri, 9) == pytest.approx(0.0010, abs=0.00005)
    assert series_error(osc, sq1, 9) == pytest.approx(0.0065, abs=0.00005)


def test_periodic_worked_example_and_series():
    """The figures Periodic forcing's worked example and its notes on the series quote, on the page's defaults."""
    import math

    from scipy.special import sici

    from vib_tutorial.core.periodic import Forcing, Wave, periodic_steady, response_coefficients, series
    from vib_tutorial.core.sdof import Oscillator, receptance

    # The notes: the square wave's fundamental carries 8/π² of its power, the first three harmonics 93.3 %;
    # the series overshoots a jump by 8.95 % of it.
    sq = Forcing(Wave.SQUARE)
    C = np.abs(sq.coefficients(20000))
    assert C[1] ** 2 / 2 == pytest.approx(8 / math.pi**2)
    assert 100 * 8 / math.pi**2 == pytest.approx(quoted("periodic", "81.1 %"), abs=0.05)
    assert np.sum(C[1:] ** 2) / 2 == pytest.approx(1.0, abs=1e-4)  # Parseval: the mean square F²
    assert 100 * np.sum(C[1:6] ** 2) / 2 == pytest.approx(quoted("periodic", "93.3 %"), abs=0.05)
    gibbs = 100 * (sici(math.pi)[0] / math.pi - 0.5)
    assert gibbs == pytest.approx(quoted("periodic", "8.95 %"), abs=0.005)
    assert gibbs == pytest.approx(quoted("periodic", "9 %"), abs=0.5)  # the Background's round figure
    t = np.linspace(0.0, 1.0, 40001)
    assert np.mean(sq.force(t[:-1]) * np.sin(2 * math.pi * t[:-1])) == pytest.approx(2 / math.pi, abs=1e-4)
    assert np.mean(sq.force(t[:-1]) * np.sin(4 * math.pi * t[:-1])) == pytest.approx(0.0, abs=1e-4)

    # The worked example.
    osc, fo = Oscillator(), Forcing()
    assert (osc.mass, osc.stiffness, osc.damping) == (1.0, 400.0, 2.0)
    assert (fo.wave, fo.amplitude, fo.freq_hz) == (Wave.SQUARE, 1.0, 1.0)
    assert osc.natural_hz == pytest.approx(quoted("periodic", "3.18"), abs=0.005)
    assert osc.zeta == pytest.approx(quoted("periodic", "0.05"))
    C = fo.coefficients(5)
    harmonics = [quoted("periodic", "1.273"), quoted("periodic", "0.4244"), quoted("periodic", "0.255")]
    assert np.all(np.abs(np.abs(C[[1, 3, 5]]) - harmonics) <= [5e-4, 5e-5, 5e-4])
    assert 1e3 / osc.stiffness == pytest.approx(quoted("periodic", "2.5"))  # mm/N, the static
    assert np.all(C[[0, 2, 4]] == 0)
    r = 2 * math.pi * np.array([1, 3, 5]) / osc.omega_n
    assert r == pytest.approx(
        [quoted("periodic", "0.314"), quoted("periodic", "0.942"), quoted("periodic", "1.571")], abs=5e-4
    )
    assert 1 - r[0] ** 2 == pytest.approx(quoted("periodic", "0.9013"), abs=5e-5)
    assert 2 * osc.zeta * r[0] == pytest.approx(quoted("periodic", "0.0314"), abs=5e-5)
    assert 1 - r[2] ** 2 == pytest.approx(-quoted("periodic", "1.4674"), abs=5e-5)
    assert 2 * osc.zeta * r[2] == pytest.approx(quoted("periodic", "0.1571"), abs=5e-5)
    assert math.hypot(0.9013, 0.0314) == pytest.approx(quoted("periodic", "0.902"), abs=5e-4)
    assert math.hypot(1.4674, 0.1571) == pytest.approx(quoted("periodic", "1.476"), abs=5e-4)
    H = np.abs(receptance(osc, np.array([1.0, 3.0, 5.0]))) * 1e3  # mm/N
    assert H == pytest.approx(
        [quoted("periodic", "2.77"), quoted("periodic", "17.10"), quoted("periodic", "1.69")], abs=0.005
    )
    assert (2.5 / 0.902, 2.5 / 1.476) == pytest.approx(
        (quoted("periodic", "2.77"), quoted("periodic", "1.69")), abs=0.005
    )  # from the rounded figures
    X = response_coefficients(osc, fo, 5)
    amps = np.abs(X[[1, 3, 5]]) * 1e3
    shown = np.array([quoted("periodic", "3.53"), quoted("periodic", "7.26"), quoted("periodic", "0.431")])
    assert np.all(np.abs(amps - shown) <= [0.005, 0.005, 0.0005])
    rounded = (1.273 * 2.77, 0.4244 * 17.10, 0.255 * 1.69)
    assert np.all(np.abs(np.array(rounded) - shown) <= [0.005, 0.005, 0.0005])
    lags = -np.degrees(np.angle(X[[1, 3, 5]] / C[[1, 3, 5]]))
    assert lags == pytest.approx(
        [quoted("periodic", "2.0"), quoted("periodic", "40.1"), quoted("periodic", "173.9")], abs=0.05
    )
    assert amps[1] / amps[0] == pytest.approx(2, abs=0.1)  # "about twice its fundamental"
    for parts in (amps, shown):
        shares = 100 * parts**2 / np.sum(parts**2)
        assert shares == pytest.approx(
            [quoted("periodic", "19.1 %"), quoted("periodic", "80.6 %"), quoted("periodic", "0.3 %")], abs=0.05
        )
        assert math.sqrt(np.sum(parts**2) / 2) == pytest.approx(5.72, abs=0.005)
        assert np.sum(parts) == pytest.approx(quoted("periodic", "11.2"), abs=0.05)
    t = np.linspace(0.0, 1.0, 4000, endpoint=False)
    exact = periodic_steady(osc, fo, t)
    assert np.sqrt(np.mean(exact**2)) * 1e3 == pytest.approx(5.72, abs=0.005)
    assert (exact.min() * 1e3, exact.max() * 1e3) == pytest.approx((-10.0, 10.0), abs=0.05)
    err = np.max(np.abs(series(X, 1.0, t) - exact)) / np.max(np.abs(exact))
    assert 100 * err == pytest.approx(quoted("periodic", "2.02 %"), abs=0.005)


def test_energy_methods_trials():
    from vib_tutorial.core.energy_methods import (
        Basis,
        Shape,
        contaminated,
        dunkerley,
        dunkerley_terms,
        exact,
        holzer,
        holzer_roots,
        iterate,
        rayleigh,
        ritz_table,
        trial_shape,
    )
    from vib_tutorial.gui.energy_methods import chain

    def hz(w2):
        return np.sqrt(w2) / (2 * np.pi)

    s = chain()
    f = exact(s)[0] / (2 * np.pi)
    assert f == pytest.approx(
        [
            quoted("energy_methods", "1.105"),
            quoted("energy_methods", "3.183"),
            quoted("energy_methods", "4.877"),
            quoted("energy_methods", "5.982"),
        ],
        abs=0.0005,
    )

    def ray(shape, eps=0.0, passes=0, system=s):
        return hz(rayleigh(system, iterate(system, contaminated(system, trial_shape(system, shape), eps), passes)))

    assert ray(Shape.STATIC) == pytest.approx(quoted("energy_methods", "1.112"), abs=0.0005)
    assert ray(Shape.STATIC) / f[0] - 1 == pytest.approx(quoted("energy_methods", "0.55 %") / 100, abs=0.00005)
    assert ray(Shape.LINEAR) / f[0] - 1 == pytest.approx(0.051, abs=0.0005)
    assert ray(Shape.LINEAR, passes=1) / f[0] - 1 == pytest.approx(0.00063, abs=0.000005)
    assert ray(Shape.SINE) == pytest.approx(f[0], rel=1e-12)
    assert ray(Shape.SINE, 0.1) / f[0] - 1 == pytest.approx(0.046, abs=0.0005)
    # 10 % of mode 2 (ε = 0.1) puts ω² about ε² = 1 % times (ω₂/ω₁)² − 1 high
    eps = quoted("energy_methods", "0.1")
    assert 100 * eps == quoted("energy_methods", "10 %")
    assert 100 * eps**2 == pytest.approx(quoted("energy_methods", "1 %"))
    assert (ray(Shape.SINE, eps) / f[0]) ** 2 - 1 == pytest.approx(eps**2 * ((f[1] / f[0]) ** 2 - 1), rel=0.3)
    assert ray(Shape.SINE, 0.2) / f[0] - 1 == pytest.approx(0.166, abs=0.0005)
    assert ray(Shape.TIP) == pytest.approx(quoted("energy_methods", "3.18"), abs=0.005)
    assert ray(Shape.TIP) / f[0] - 1 == pytest.approx(1.88, abs=0.005)
    assert ray(Shape.TIP, passes=1) / f[0] - 1 == pytest.approx(0.051, abs=0.0005)
    assert ray(Shape.TIP, passes=2) / f[0] - 1 == pytest.approx(0.00063, abs=0.000005)
    assert (f[0] / f[1]) ** 2 == pytest.approx(0.12, abs=0.005)
    assert hz(dunkerley_terms(s)) == pytest.approx(
        [
            quoted("energy_methods", "3.18"),
            quoted("energy_methods", "2.25"),
            quoted("energy_methods", "1.84"),
            quoted("energy_methods", "1.59"),
        ],
        abs=0.005,
    )
    assert hz(dunkerley(s)) == pytest.approx(quoted("energy_methods", "1.007"), abs=0.0005)
    assert hz(dunkerley(s)) / f[0] - 1 == pytest.approx(-quoted("energy_methods", "8.9 %") / 100, abs=0.0005)
    tip = chain(masses=(1.0, 1.0, 1.0, 5.0))
    ft = exact(tip)[0] / (2 * np.pi)
    assert hz(dunkerley_terms(tip))[-1] == pytest.approx(0.71, abs=0.005)
    assert (hz(dunkerley(tip)), ft[0]) == pytest.approx((0.624, 0.653), abs=0.0005)
    assert hz(dunkerley(tip)) / ft[0] - 1 == pytest.approx(-0.044, abs=0.0005)
    assert ray(Shape.STATIC, system=tip) / ft[0] - 1 == pytest.approx(0.0025, abs=0.00005)
    table = ritz_table(s, Basis.KRYLOV)
    assert table[1, 0] == pytest.approx(f[0], abs=0.00005)
    assert table[1, 1] == pytest.approx(3.30, abs=0.005)
    assert table[1, 1] / f[1] - 1 == pytest.approx(quoted("energy_methods", "3.6 %") / 100, abs=0.0005)
    assert (table[2, 1], table[2, 2]) == pytest.approx((3.184, 5.02), abs=0.005)
    stiff = chain(stiffness=(4000.0, 400.0, 400.0, 400.0))
    fs = exact(stiff)[0] / (2 * np.pi)
    powers = ritz_table(stiff, Basis.POWERS)
    assert (powers[1, 0], fs[0]) == pytest.approx((1.63, 1.38), abs=0.005)
    assert powers[1, 0] / fs[0] - 1 == pytest.approx(0.18, abs=0.005)
    assert (powers[1, 1], fs[1]) == pytest.approx((6.04, 3.89), abs=0.005)
    assert ritz_table(stiff, Basis.KRYLOV)[0, 0] / fs[0] - 1 < 0.008
    assert holzer(s, 2 * np.pi * 1.0)[-1].spring == pytest.approx(61, abs=0.5)
    assert holzer_roots(s, 7.0) == pytest.approx(f, abs=0.0005)


def test_energy_methods_worked_example():
    """The Theory's worked example on the page's opening chain: Rayleigh from the sag, Dunkerley, two Krylov
    shapes and a line of Holzer's table, each number from the solver and from the figures it quotes."""
    from vib_tutorial.core.energy_methods import (
        Basis,
        dunkerley,
        dunkerley_terms,
        exact,
        holzer,
        rayleigh,
        ritz_table,
        static_deflection,
    )
    from vib_tutorial.gui.energy_methods import chain

    s = chain()
    M, _, K = s.matrices()
    w, _ = exact(s)
    assert w[0] ** 2 == pytest.approx(quoted("energy_methods", "48.246"), abs=0.0005)
    v = static_deflection(s, s.masses)
    assert 400 * v == pytest.approx([4, 7, 9, 10])
    assert np.diag(np.linalg.inv(K)) * 400 == pytest.approx([1, 2, 3, 4])
    assert 12000 / 246 == pytest.approx(48.78, abs=0.005) == rayleigh(s, v)
    assert math.sqrt(48.78) == pytest.approx(quoted("energy_methods", "6.984"), abs=0.0005)
    assert math.sqrt(rayleigh(s, v)) / (2 * math.pi) == pytest.approx(quoted("energy_methods", "1.112"), abs=0.0005)
    assert math.sqrt(rayleigh(s, v)) / w[0] - 1 == pytest.approx(quoted("energy_methods", "0.55 %") / 100, abs=0.00005)
    assert np.sqrt(dunkerley_terms(s)) == pytest.approx(
        [20, quoted("energy_methods", "14.14"), quoted("energy_methods", "11.55"), 10], abs=0.005
    )
    assert dunkerley(s) == pytest.approx(40)
    assert math.sqrt(40) == pytest.approx(quoted("energy_methods", "6.325"), abs=0.0005)
    assert math.sqrt(40) / (2 * math.pi) == pytest.approx(quoted("energy_methods", "1.007"), abs=0.0005)
    assert np.sum(1 / w**2) == pytest.approx(quoted("energy_methods", "0.025"))
    assert 1 / w[0] ** 2 == pytest.approx(quoted("energy_methods", "0.0207"), abs=0.00005)
    # two Krylov shapes, as the notes scale them
    w1 = np.array([4.0, 7, 9, 10])
    w2 = 400 * np.linalg.solve(K, M @ w1)
    assert w2 == pytest.approx([30, 56, 75, 85])
    W = np.column_stack([w1, w2])
    assert W.T @ M @ W == pytest.approx(np.array([[246, 2037], [2037, 16886]]))
    assert W.T @ K @ W == pytest.approx(400 * np.array([[30, 246], [246, 2037]]))
    a, b, c = 246 * 16886 - 2037**2, 400 * (30 * 16886 + 2037 * 246 - 2 * 246 * 2037), 400**2 * (30 * 2037 - 246**2)
    assert (a, b / 1e6, c / 1e7) == pytest.approx(
        (4587, quoted("energy_methods", "2.1912"), quoted("energy_methods", "9.504")), abs=0.00005
    )
    roots = np.sort(np.roots([a, -b, c]))
    assert roots == pytest.approx([quoted("energy_methods", "48.246"), quoted("energy_methods", "429.45")], abs=0.005)
    ritz = ritz_table(s, Basis.KRYLOV)[1, :2]
    assert ritz**2 * (2 * math.pi) ** 2 == pytest.approx(roots, rel=1e-9)
    assert ritz == pytest.approx([quoted("energy_methods", "1.105"), quoted("energy_methods", "3.298")], abs=0.0005)
    assert ritz[0] / w[0] * 2 * math.pi - 1 < 1e-5
    assert ritz[1] / (w[1] / (2 * math.pi)) - 1 == pytest.approx(quoted("energy_methods", "3.6 %") / 100, abs=0.0005)
    # Holzer at 1 Hz, the tab's start
    from vib_tutorial.gui.energy_methods import EnergyTrial

    assert EnergyTrial.holzer_hz == quoted("energy_methods", "1.00")
    assert (2 * math.pi) ** 2 == pytest.approx(quoted("energy_methods", "39.48"), abs=0.005)
    rows = holzer(s, 2 * math.pi)
    assert [r.inertia for r in rows] == pytest.approx(
        [
            quoted("energy_methods", "39.5"),
            quoted("energy_methods", "75.1"),
            quoted("energy_methods", "103.2"),
            quoted("energy_methods", "121.2"),
        ],
        abs=0.05,
    )
    assert [r.spring for r in rows] == pytest.approx(
        [
            quoted("energy_methods", "360.5"),
            quoted("energy_methods", "285.5"),
            quoted("energy_methods", "182.2"),
            quoted("energy_methods", "61.0"),
        ],
        abs=0.05,
    )
    assert [r.x for r in rows][1:] == pytest.approx(
        [quoted("energy_methods", "1.901"), quoted("energy_methods", "2.615"), quoted("energy_methods", "3.071")],
        abs=0.0005,
    )


def test_integration_trials():
    import math

    from vib_tutorial.core.integration import (
        Load,
        Model,
        Scheme,
        exact,
        simulate,
        spectral,
        stable_limit,
    )

    def at(per_period, scheme, alpha=-0.1):
        return spectral(scheme, 2 * math.pi / per_period, 0.0, alpha)

    assert Model().frequencies_hz()[0] == pytest.approx(quoted("integration", "3.18"), abs=0.005)
    assert at(10, Scheme.CENTRAL).period_error == pytest.approx(-quoted("integration", "1.7 %") / 100, abs=0.0005)
    assert at(10, Scheme.AVERAGE).period_error == pytest.approx(quoted("integration", "3.2 %") / 100, abs=0.0005)
    assert at(10, Scheme.LINEAR).period_error == pytest.approx(quoted("integration", "1.6 %") / 100, abs=0.0005)
    assert at(10, Scheme.RK4).period_error == pytest.approx(quoted("integration", "0.11 %") / 100, abs=0.00005)
    assert at(10, Scheme.RK4).damping == pytest.approx(quoted("integration", "0.065 %") / 100, abs=0.000005)
    assert 10 * at(10, Scheme.AVERAGE).period_error == pytest.approx(1 / 3, abs=0.02)
    assert 2 * math.pi / 3.0 == pytest.approx(2.09, abs=0.005) and at(3.0, Scheme.CENTRAL).radius > 1
    assert 2 * math.pi / 3.2 == pytest.approx(1.96, abs=0.005)
    assert at(3.2, Scheme.CENTRAL).radius == pytest.approx(1.0)
    assert at(3.2, Scheme.CENTRAL).period_error == pytest.approx(-0.29, abs=0.005)
    assert at(3.0, Scheme.AVERAGE).period_error == pytest.approx(0.30, abs=0.005)
    assert at(1.0, Scheme.AVERAGE).period_error + 1 == pytest.approx(2.5, abs=0.05)
    assert at(1.0, Scheme.AVERAGE).radius == pytest.approx(1.0)
    assert stable_limit(Scheme.LINEAR) == pytest.approx(2 * math.sqrt(3))
    assert 2 * math.sqrt(3) / (2 * math.pi) == (pytest.approx(0.55, abs=0.005))
    assert at(1.0, Scheme.LINEAR).radius > 1
    rk = at(5, Scheme.RK4)
    assert rk.period_error == pytest.approx(0.011, abs=0.0005) and 1 - rk.radius == pytest.approx(0.022, abs=0.0005)
    assert rk.damping == pytest.approx(0.018, abs=0.0005) and rk.radius**50 == pytest.approx(0.33, abs=0.005)
    assert at(5, Scheme.AVERAGE).period_error == pytest.approx(0.12, abs=0.005)
    eu = at(50, Scheme.EULER).radius
    assert eu - 1 == pytest.approx(0.008, abs=0.0005) and eu**50 - 1 == pytest.approx(0.48, abs=0.005)
    assert eu**500 == pytest.approx(50, abs=1)
    stiff = Model(stiff_hz=1000.0)
    f = stiff.frequencies_hz()
    assert f == pytest.approx([3.03, 1049], abs=0.5) and f[0] == pytest.approx(3.03, abs=0.005)
    h = 1 / f[0] / 50
    assert 2 * math.pi * f[1] * h == pytest.approx(43, abs=0.5) and 1 / (f[1] * h) == pytest.approx(1 / 7, abs=0.005)
    for scheme in (Scheme.CENTRAL, Scheme.RK4):
        x = simulate(scheme, stiff, Load(), h, 250)
        assert np.isnan(x[: int(0.020 / h) + 1]).any()
    assert np.all(np.isfinite(simulate(Scheme.AVERAGE, stiff, Load(), h, 250)))
    assert spectral(Scheme.HHT, 2 * math.pi * f[1] * h, 0.0, -0.3).radius == pytest.approx(0.57, abs=0.005)
    assert spectral(Scheme.HHT, 2 * math.pi * f[0] * h, 0.0, -0.3).damping == pytest.approx(0.00004, abs=0.000005)
    m, load = Model(), Load(0.0, 1.0, 2.0)
    h = 1 / m.frequencies_hz()[0] / 10
    t = np.arange(101) * h
    ref = exact(m, load, t)
    errs = {
        s: np.max(np.abs(simulate(s, m, load, h, 100) - ref)) / np.max(np.abs(ref))
        for s in (Scheme.FOH, Scheme.CENTRAL, Scheme.AVERAGE, Scheme.RK4)
    }
    assert errs[Scheme.FOH] == pytest.approx(0.013, abs=0.0005)
    assert [errs[s] for s in (Scheme.RK4, Scheme.CENTRAL, Scheme.AVERAGE)] == pytest.approx(
        [0.03, 0.41, 0.61], abs=0.005
    )


def test_fatigue_trials():
    from vib_tutorial.core.fatigue import (
        Curve,
        Stress,
        dirlik_damage,
        life_text,
        narrow_band_damage,
        rates,
        record,
    )

    c, c5 = Curve(), Curve(b=5.0)
    one = Stress()
    r = record(one, 600.0)
    assert rates(one).alpha2 == pytest.approx(quoted("fatigue", "0.95"), abs=0.005)
    for est in (narrow_band_damage(one, c), dirlik_damage(one, c)):
        assert abs(est / r.damage_rate(c) - 1) < 0.015
    assert life_text(r.damage_rate(c)) == "25.1 days"
    two = Stress(sigma2=10.0)
    r2 = record(two, 600.0)
    assert rates(two).alpha2 == pytest.approx(0.46, abs=0.005) and rates(two).nu0 == pytest.approx(6.9, abs=0.05)
    assert narrow_band_damage(two, c) / r2.damage_rate(c) == pytest.approx(1.6, abs=0.05)
    assert dirlik_damage(two, c) / r2.damage_rate(c) == pytest.approx(0.92, abs=0.005)
    assert narrow_band_damage(two, c5) / r2.damage_rate(c5) == pytest.approx(1.6, abs=0.05)
    assert dirlik_damage(two, c5) / r2.damage_rate(c5) == pytest.approx(0.82, abs=0.005)
    average = np.mean([record(two, 600.0, k).damage_rate(c5) for k in range(1, 21)])
    assert dirlik_damage(two, c5) / average == pytest.approx(0.85, abs=0.005)
    assert life_text(record(Stress(sigma1=40.0), 600.0).damage_rate(c)) == "3.13 days"
    assert 1.1**3 == pytest.approx(1.33, abs=0.005)  # the damage rises by a third…
    assert 1 / 1.1**3 == pytest.approx(0.75, abs=0.005)  # …so a quarter of the life goes (#409 C-1)
    wide = Stress(zeta1=0.3)
    rw = record(wide, 600.0)
    assert rates(wide).alpha2 == pytest.approx(0.64, abs=0.005)
    assert narrow_band_damage(wide, c) / rw.damage_rate(c) == pytest.approx(1.15, abs=0.005)
    assert abs(dirlik_damage(wide, c) / rw.damage_rate(c) - 1) < 0.02
    short = [record(two, 20.0, k).damage_rate(c5) / r2.damage_rate(c5) for k in range(1, 11)]
    assert (min(short), max(short)) == pytest.approx((0.67, 1.53), abs=0.005)


def test_self_excited_trials():
    from vib_tutorial.core.self_excited import (
        Flutter,
        Galloping,
        VanDerPol,
        Viv,
        dominant_hz,
        galloping_branches,
        settled_amplitude,
    )

    v = VanDerPol()
    t, x, _ = v.simulate(60)
    assert settled_amplitude(x) == pytest.approx(0.0089, abs=0.00005)
    assert -v.eps / 2 == pytest.approx(-quoted("self_excited", "0.1"))
    t, x, _ = VanDerPol(eps=5.0).simulate(20)
    assert 1 / dominant_hz(x, t[1] - t[0]) == pytest.approx(1.84, abs=0.03)
    g = Galloping()
    assert g.onset == pytest.approx(quoted("self_excited", "3.81"), abs=0.005)
    U = np.linspace(4.0, 7.5, 351)
    pairs = [b for b in galloping_branches(g, U) if not b.stable]
    band = U[np.isfinite(pairs[0].amplitude)]
    assert (band.min(), band.max()) == pytest.approx((4.7, 7.0), abs=0.05)
    at6 = sorted(b.amplitude[0] / g.side for b in galloping_branches(g, np.array([6.0])) if b.stable)
    assert at6 == pytest.approx([quoted("self_excited", "0.51"), quoted("self_excited", "1.28")], abs=0.005)
    for start, want in ((0.001, 0.51), (0.12, 1.28)):
        _, y, _ = Galloping(speed=6.0, start=start).simulate(600)
        assert settled_amplitude(y) / g.side == pytest.approx(want, abs=0.01)
    _, y, _ = Galloping(speed=12.0).simulate(200)
    assert settled_amplitude(y) / g.side == pytest.approx(2.7, abs=0.05) and 12 / g.onset == pytest.approx(3.1, abs=0.1)
    viv = Viv()
    assert viv.mass_ratio == pytest.approx(quoted("self_excited", "10.2"), abs=0.05)
    assert viv.strouhal * 5.5 == pytest.approx(quoted("self_excited", "1.1"))
    amp, freq = viv.sweep(np.array([4.0, 4.5, 5.5, 6.25, 6.5, 9.0]))
    assert amp[2] == pytest.approx(0.21, abs=0.005) and freq[2] == pytest.approx(1.0, abs=0.01)
    assert freq[1:4] == pytest.approx([1.0, 1.0, 1.0], abs=0.04) and abs(freq[4] - 1) > 0.04
    assert abs(freq[0] - 1) > 0.04
    assert amp[5] == pytest.approx(0.019, abs=0.0005) and freq[5] == pytest.approx(1.7, abs=0.02)
    # steady lift: coalescence, mass balance removes it
    f = Flutter()
    assert f.flutter_speed() == pytest.approx(quoted("self_excited", "20.7"), abs=0.05)
    assert f.divergence_speed() == pytest.approx(39, abs=0.5)
    fr, _ = f.modes(np.array([0.0, 20.0]))
    assert fr[0] == pytest.approx([quoted("self_excited", "2.93"), quoted("self_excited", "6.71")], abs=0.1)
    assert fr[1] == pytest.approx([3.6, 4.7], abs=0.05)
    undamped = Flutter(zeta=0.0)
    U0 = undamped.flutter_speed()
    assert U0 == pytest.approx(quoted("self_excited", "20.9"), abs=0.05)
    fr, _ = undamped.modes(np.array([1.0001 * U0]))
    assert fr[0] == pytest.approx([quoted("self_excited", "4.07")] * 2, abs=0.01)  # merged
    assert math.isnan(Flutter(x_cg=0.0).flutter_speed()) and math.isnan(Flutter(x_cg=-0.2).flutter_speed())
    assert Flutter(x_cg=0.1).flutter_speed() == pytest.approx(22.8, abs=0.05)
    # quasi-steady lift: the pitch-like mode alone, the frequencies apart; mass balance doesn't save it
    q = Flutter(quasi_steady=True)
    assert q.flutter_speed() == pytest.approx(quoted("self_excited", "19.1"), abs=0.05)
    fr, z = q.modes(np.array([19.0]))
    assert fr[0] == pytest.approx([quoted("self_excited", "3.4"), quoted("self_excited", "5.0")], abs=0.05)
    assert z[0, 0] == pytest.approx(0.1, abs=0.01) and z[0, 1] < 0.002  # the plunge damped, the pitch near zero
    assert q.at(x_cg=0.0).flutter_speed() == pytest.approx(quoted("self_excited", "17.7"), abs=0.05)


def test_pendulum_trials():
    import math

    from vib_tutorial.core.pendulum import (
        Kind,
        Pendulum,
        conjugate_pivot,
        duffing_period,
        period,
        rod_period,
        swing,
    )

    p = Pendulum()
    assert (p.natural_hz, p.period0) == pytest.approx((0.498, 2.006), abs=5e-4)
    assert Pendulum(mass=7.0).period0 == pytest.approx(p.period0)
    assert period(p, math.radians(90)) == pytest.approx(2.368, abs=5e-4)
    assert 100 * (period(p, math.radians(30)) / p.period0 - 1) == pytest.approx(quoted("pendulum", "1.7 %"), abs=0.05)
    assert 100 * (period(p, math.radians(90)) / p.period0 - 1) == pytest.approx(quoted("pendulum", "18 %"), abs=0.5)
    # Duffing's estimate (either form) within 1 % up to about 75°
    for a, inside in ((70, True), (80, False)):
        T = period(p, math.radians(a))
        for guess in (duffing_period(p, math.radians(a)), p.period0 * (1 + math.radians(a) ** 2 / 16)):
            assert (100 * abs(guess / T - 1) < quoted("pendulum", "1 %")) == inside
    quarter = Pendulum(length=0.25)  # A quarter of the string
    assert (quarter.period0, period(quarter, math.radians(10))) == pytest.approx((1.003, 1.005), abs=5e-4)
    from vib_tutorial.view_models import pendulum as view
    from vib_tutorial.view_models.live_pendulum import COL, LivePendulum

    live = LivePendulum(Pendulum(zeta=0.003))  # Nearly over the top, dying away: its first whole swing
    live.hold(math.radians(175))
    live.let_go()
    live.advance(10.0)
    t, y = live.since_change()
    _, first, took = view.cycles(t, y[:, COL["theta"]])[0]
    assert math.degrees(first) == pytest.approx(150, abs=1.5) and took == pytest.approx(3.5, abs=0.05)
    assert took / p.period0 == pytest.approx(1.76, abs=0.005)
    rod = Pendulum(Kind.ROD)
    assert (rod.inertia, rod.equivalent_length, rod.period0) == pytest.approx((1 / 3, 0.667, 1.638), abs=5e-4)
    assert rod_period(1.0, 1 / math.sqrt(12)) == pytest.approx(1.524, abs=5e-4)
    assert conjugate_pivot(1.0, 0.5) == pytest.approx(0.167, abs=5e-4)
    assert rod_period(1.0, conjugate_pivot(1.0, 0.5)) == pytest.approx(rod.period0)
    disc = Pendulum(Kind.TORSION, length=0.1, spring=0.5)
    assert disc.natural_hz == pytest.approx(1.59, abs=0.005)
    assert period(disc, math.radians(170)) == pytest.approx(disc.period0, rel=1e-9)
    assert disc.period0 == pytest.approx(0.628, abs=5e-4)
    assert Pendulum(Kind.TORSION, length=0.2, spring=0.5).inertia == pytest.approx(4 * disc.inertia)  # twice r
    inv = Pendulum(Kind.INVERTED, spring=15.0)
    assert (inv.stiffness, inv.natural_hz) == pytest.approx((5.19, 0.363), abs=5e-4)
    assert 1 - period(inv, math.radians(60)) / inv.period0 == pytest.approx(0.10, abs=0.005)
    lean = Pendulum(Kind.INVERTED, spring=9.0, zeta=0.1)
    assert math.degrees(lean.leaning()) == pytest.approx(40.8, abs=0.05)
    _, th, _ = swing(lean, math.radians(1.0), 70.0)
    assert math.degrees(th[-1]) == pytest.approx(40.8, abs=0.1)


def test_pendulum_worked_example_and_inertia():
    """The figures Pendulums' worked example and its measured moment of inertia quote, on the page's defaults,
    and from the rounded figures they show."""
    import math

    from vib_tutorial.core.pendulum import Kind, Pendulum, duffing_period, period, swing

    p = Pendulum()
    assert (p.kind, p.mass, p.length, p.gravity, p.zeta) == (Kind.SIMPLE, 1.0, 1.0, quoted("pendulum", "9.81"), 0.0)
    assert (p.inertia, p.stiffness) == pytest.approx((1.0, 9.81))
    w0, f0, T0 = quoted("pendulum", "3.132"), quoted("pendulum", "0.4985"), quoted("pendulum", "2.006")
    assert p.omega0 == pytest.approx(w0, abs=5e-4) and p.natural_hz == pytest.approx(f0, abs=5e-5)
    assert p.period0 == pytest.approx(T0, abs=5e-4)
    assert w0 / (2 * math.pi) == pytest.approx(f0, abs=5e-5)  # from the rounded ω_n
    assert 2 * math.pi / w0 == pytest.approx(T0, abs=5e-4)
    a = math.radians(10)
    a_, moment, linear_moment, g = (
        quoted("pendulum", "0.1745"),
        quoted("pendulum", "1.703"),
        quoted("pendulum", "1.712"),
        quoted("pendulum", "9.81"),
    )
    assert a == pytest.approx(a_, abs=5e-5)
    assert float(p.moment(a)) == pytest.approx(moment, abs=5e-4)
    assert p.stiffness * a == pytest.approx(linear_moment, abs=5e-4)
    assert g * a_ == pytest.approx(linear_moment, abs=5e-4)
    assert 100 * (1 - moment / linear_moment) == pytest.approx(quoted("pendulum", "0.5 %"), abs=0.05)  # less
    grow = quoted("pendulum", "0.0019")
    assert a * a / 16 == pytest.approx(grow, abs=5e-5) and a_**2 / 16 == pytest.approx(grow, abs=5e-5)
    assert 100 * grow == pytest.approx(quoted("pendulum", "0.19 %"))
    assert 1 + grow == pytest.approx(quoted("pendulum", "1.0019"))
    exact = period(p, a)
    T = quoted("pendulum", "2.010")
    assert exact == pytest.approx(T, abs=5e-4) and T0 * 1.0019 == pytest.approx(T, abs=5e-4)
    longer = quoted("pendulum", "0.191 %")  # the readout's +0.191 %
    assert 100 * (exact / p.period0 - 1) == pytest.approx(longer, abs=0.0005)
    assert 100 * (duffing_period(p, a) / p.period0 - 1) == pytest.approx(longer, abs=0.0005)
    assert 8 * (exact - p.period0) == pytest.approx(
        quoted("pendulum", "0.03"), abs=0.005
    )  # behind after the eight swings drawn
    _, whole, linear = swing(p, a, 8 * p.period0, 4000)
    assert np.max(np.abs(whole - linear)) < 0.1 * a  # too little to see
    # Duffing's estimate keeps the period within 0.14 % up to 45°
    off = 100 * (1 - p.period0 * (1 + math.radians(45) ** 2 / 16) / period(p, math.radians(45)))
    assert off == pytest.approx(quoted("pendulum", "0.14 %"), abs=0.005)
    # From 90°, the string pulls three times the bob's weight at the bottom
    from vib_tutorial.view_models.live_pendulum import LivePendulum
    from vib_tutorial.view_models.pendulum import pull

    live = LivePendulum(p)
    live.hold(math.pi / 2)
    live.let_go()
    live.advance(period(p, math.pi / 2) / 4)
    assert pull(p, live.theta, live.omega)[2] == pytest.approx(3 * p.mass * p.gravity, rel=2e-3)
    # The series for large swings
    for deg in (30, 60, 90):
        A = math.radians(deg)
        series = 1 + A**2 / 16 + 11 * A**4 / 3072
        assert series == pytest.approx(period(p, A) / p.period0, rel=0.012)
    # Measuring J: the 1 m rod hung from its end
    rod = Pendulum(Kind.ROD)
    Tr, d, mdd = (
        quoted("pendulum", "1.638"),
        quoted("pendulum", "0.5"),
        quoted("pendulum", "0.25"),
    )  # its period, G's distance from the pivot, md²
    assert rod.period0 == pytest.approx(Tr, abs=5e-4) and d * d * rod.mass == mdd
    J = rod.mass * rod.gravity * d * rod.period0**2 / (4 * math.pi**2)
    assert J == pytest.approx(1 / 3) and J - mdd == pytest.approx(1 / 12)
    J_rounded, JG = 1 * 9.81 * d * Tr**2 / (4 * math.pi**2), quoted("pendulum", "0.0834")
    assert J_rounded == pytest.approx(quoted("pendulum", "0.3334"), abs=5e-5)
    assert quoted("pendulum", "0.3334") - mdd == pytest.approx(JG, abs=5e-5)
    assert J_rounded - mdd == pytest.approx(JG, abs=5e-5)
    assert 1 / 12 == pytest.approx(quoted("pendulum", "0.0833"), abs=5e-5)
    assert J / (J - 0.25) == pytest.approx(4.0)
    slow = 1 * 9.81 * 0.5 * ((1 + quoted("pendulum", "0.1 %") / 100) * rod.period0) ** 2 / (4 * math.pi**2)
    assert 100 * (slow / J - 1) == pytest.approx(quoted("pendulum", "0.2 %"), abs=5e-3)  # from 0.1 % slower
    assert 100 * ((slow - 0.25) / (1 / 12) - 1) == pytest.approx(quoted("pendulum", "0.8 %"), abs=0.05)
    # Kater: the two pivots are L_eq apart
    assert 0.4 + (1 / 12) / 0.4 == pytest.approx(Pendulum(Kind.ROD, pivot=0.4).equivalent_length)


def test_beams_plates_trials():
    import math

    from vib_tutorial.core.beams_plates import (
        Beam,
        Plate,
        Theory,
        beam_frequencies,
        beam_omega,
        circular_modes,
        rect_modes,
        wave_speeds,
    )

    b = Beam()
    e = beam_frequencies(b, 5, Theory.EULER)
    assert e[0] == pytest.approx(235, abs=0.5)
    ratio = {th: beam_frequencies(b, 5, th) / e for th in Theory}
    assert 1 / ratio[Theory.TIMOSHENKO][0] - 1 == pytest.approx(0.017, abs=5e-4)
    assert 1 / ratio[Theory.TIMOSHENKO][4] - 1 == pytest.approx(0.33, abs=0.005)
    assert ratio[Theory.TIMOSHENKO][4] == pytest.approx(0.75, abs=0.005)
    assert (ratio[Theory.SHEAR][4], ratio[Theory.RAYLEIGH][4]) == pytest.approx((0.78, 0.91), abs=0.005)
    slim = Beam(depth=0.02)
    r = beam_frequencies(slim, 10) / beam_frequencies(slim, 10, Theory.EULER)
    assert (1 - r[0], 1 - r[9]) == pytest.approx((0.0007, 0.060), abs=5e-4)
    assert wave_speeds(b)[1] == pytest.approx(2900, abs=50)
    cutoff = float(beam_omega(b, 1e-6, branch=1)) / (2 * math.pi)
    assert cutoff == pytest.approx(16e3, rel=0.02)
    assert float(beam_omega(b, math.pi, branch=1)) / (2 * math.pi) == pytest.approx(16.4e3, rel=0.01)
    plate = Plate()
    modes = rect_modes(plate, 6)
    assert [round(m.freq_hz, 1) for m in modes[:4]] == pytest.approx(
        [quoted("beams_plates", "66.6"), 128.1, 204.9, quoted("beams_plates", "230.5")], abs=0.1
    )
    assert (modes[5].m, modes[5].n) == (3, 2)
    square = rect_modes(Plate(0.4, 0.4), 6)
    assert (square[4].m, square[4].n, square[5].m, square[5].n) == (1, 3, 3, 1)
    assert square[4].freq_hz == pytest.approx(square[5].freq_hz)
    circle = circular_modes(Plate(0.2, 0.2, circular=True), 4)
    assert [m.root**2 for m in circle] == pytest.approx([10.22, 21.26, 34.88, 39.77], abs=0.005)
    assert [m.freq_hz for m in circle] == pytest.approx([191, 397, 652, 743], abs=0.5)
    per_hz = plate.area / (2 * plate.scale)
    assert (per_hz, 1 / per_hz) == pytest.approx((quoted("beams_plates", "0.026"), 39), rel=0.02)


def test_seismic_trials():
    from vib_tutorial.core.seismic import (
        GROUNDS,
        Design,
        Rule,
        building,
        combine,
        correlation,
        modal,
        suite,
        with_tank,
    )

    d = Design()
    five = building(5)
    m = modal(five, d)
    rho = correlation(m.f, d.zeta)
    assert m.T[0] == pytest.approx(0.99, abs=0.005)
    assert (m.mass_share[0], m.mass_share[:2].sum()) == pytest.approx(
        (quoted("seismic", "88.0 %") / 100, 0.97), abs=0.005
    )
    assert m.sa[0] / 9.81 == pytest.approx(0.52, abs=0.005)
    shear = {r: float(combine(m.shear, r, rho)[0]) for r in Rule}
    assert (shear[Rule.CQC], shear[Rule.SRSS], shear[Rule.ABS]) == pytest.approx((230e3, 229e3, 277e3), abs=500)
    records = suite(five, d).base_shear
    assert records == pytest.approx(235e3, abs=1e3)
    assert shear[Rule.ABS] / records - 1 == pytest.approx(0.18, abs=0.005)
    assert rho[0, 1] == pytest.approx(0.007, abs=5e-4)
    tank = with_tank(five, 0.01)
    mt = modal(tank, d)
    rt = correlation(mt.f, d.zeta)
    assert (mt.f[0], mt.f[1], rt[0, 1]) == pytest.approx((0.96, 1.06, 0.50), abs=0.005)
    vt = suite(tank, d).base_shear
    assert float(combine(mt.shear, Rule.SRSS, rt)[0]) / vt - 1 == pytest.approx(-0.21, abs=0.01)
    assert abs(float(combine(mt.shear, Rule.CQC, rt)[0]) / vt - 1) < 0.11
    swing = suite(tank, d).displacement[-1]
    assert float(combine(mt.displacement, Rule.SRSS, rt)[-1]) / swing - 1 == pytest.approx(0.54, abs=0.01)
    assert abs(float(combine(mt.displacement, Rule.CQC, rt)[-1]) / swing - 1) < 0.11
    soft = modal(five, Design(ground=GROUNDS[3]))
    assert soft.sa[0] / 9.81 == pytest.approx(0.82, abs=0.005)
    assert float(combine(soft.shear, Rule.CQC, correlation(soft.f, 0.05))[0]) == pytest.approx(357e3, abs=1e3)
    eight = modal(building(8), d)
    assert (eight.T[0], eight.mass_share[0]) == pytest.approx((1.52, 0.86), abs=0.005)
    assert eight.sa[0] / 9.81 == pytest.approx(0.34, abs=0.005)
    assert eight.sa[1:4] / 9.81 == pytest.approx(0.86, abs=0.005)
    total = combine(eight.shear, Rule.CQC, correlation(eight.f, 0.05))
    alone = abs(eight.shear[:, 0]) / total
    assert (alone[0], alone[-1]) == pytest.approx((0.96, 0.74), abs=0.005)
    assert Design(zeta=0.10).eta == pytest.approx(0.82, abs=0.005)


def test_seismic_worked_example():
    import math

    import numpy as np

    from vib_tutorial.core.model import ChainSystem
    from vib_tutorial.core.seismic import Design, G, Rule, building, combine, correlation, design_sa, modal, suite

    d = Design()
    S, TB, TC, TD = d.corners
    assert (S, TB, TC) == (quoted("seismic", "1.15"), quoted("seismic", "0.2"), quoted("seismic", "0.6"))
    assert (TD, d.eta) == (2.0, 1.0)
    assert 100 * d.zeta == quoted("seismic", "5 %")
    assert d.pga_g == quoted("seismic", "0.3")
    plateau = quoted("seismic", "2.5") * d.eta * d.pga_g * S
    assert plateau == pytest.approx(quoted("seismic", "0.8625"), abs=1e-12)
    assert float(f"{2.5 * d.eta * S * d.pga_g:.3g}") == quoted("seismic", "0.862")  # the readout's, as it prints it
    assert float(design_sa(d, np.array([0.4]))[0]) / G == pytest.approx(plateau)  # on the plateau
    m = modal(building(5), d)
    w1 = 2 * math.sqrt(500) * math.sin(math.pi / 22)
    assert w1 == pytest.approx(6.36, abs=0.005) and 2 * math.pi / w1 == pytest.approx(m.T[0], rel=1e-9)
    assert m.T[0] == pytest.approx(quoted("seismic", "0.987"), abs=5e-4)
    assert 0.8625 * 0.6 / 0.987 == pytest.approx(quoted("seismic", "0.524"), abs=5e-4)
    assert m.sa[0] / 9.81 == pytest.approx(quoted("seismic", "0.524"), abs=5e-4)
    assert m.mass_share[0] == pytest.approx(quoted("seismic", "88.0 %") / 100, abs=5e-4)
    assert m.mass_share[0] * 50 == pytest.approx(quoted("seismic", "44.0"), abs=0.05)  # t, of the 50 t
    assert 44.0e3 * 0.524 * 9.81 == pytest.approx(226e3, abs=500)
    assert abs(m.base_shear[0]) == pytest.approx(226e3, abs=500)
    assert (m.T[1], m.mass_share[1], abs(m.base_shear[1])) == pytest.approx(
        (quoted("seismic", "0.338"), quoted("seismic", "8.7 %") / 100, 37e3), rel=0.01
    )
    rho = correlation(m.f, d.zeta)
    assert float(combine(m.shear, Rule.CQC, rho)[0]) == pytest.approx(230e3, abs=500)
    assert float(combine(m.shear, Rule.ABS, rho)[0]) == pytest.approx(277e3, abs=500)
    assert suite(building(5), d).base_shear == pytest.approx(235e3, abs=500)
    sd = m.sa[0] / (2 * math.pi * m.f[0]) ** 2
    assert (sd, m.gamma[0] * m.phi[-1, 0]) == pytest.approx((0.127, quoted("seismic", "1.25")), abs=5e-3)
    drift = combine(m.drift, Rule.CQC, rho)
    assert float(combine(m.displacement, Rule.CQC, rho)[-1]) == pytest.approx(0.159, abs=5e-4)
    assert drift[0] == pytest.approx(0.046, abs=5e-4)
    assert drift[0] / 3 == pytest.approx(quoted("seismic", "1.5 %") / 100, abs=5e-4)
    nu = quoted("seismic", "0.5")
    assert nu * drift[0] / 3 == pytest.approx(quoted("seismic", "0.77 %") / 100, abs=5e-5)
    # within the limit for partitions that don't feel it (0.010h), not for brittle (0.005h) or ductile (0.0075h) ones
    assert quoted("seismic", "0.005") < quoted("seismic", "0.0075") < nu * drift[0] / 3 < quoted("seismic", "0.010")
    assert 230 / 3 == pytest.approx(77, abs=0.5)
    # a soft story
    soft = ChainSystem(np.full(5, 1e4), np.array([1e6] + [5e6] * 4), np.zeros(5))
    ms = modal(soft, d)
    rs = correlation(ms.f, d.zeta)
    ds = combine(ms.drift, Rule.CQC, rs)
    top = float(combine(ms.displacement, Rule.CQC, rs)[-1])
    assert (ms.T[0], float(combine(ms.shear, Rule.CQC, rs)[0])) == pytest.approx((1.58, 159e3), rel=3e-3)
    assert (ds[0], top, ds[0] / top) == pytest.approx((0.159, 0.227, 0.70), abs=5e-3)
    # base isolation
    iso = ChainSystem(np.full(6, 1e4), np.array([3e5] + [5e6] * 5), np.zeros(6))
    di = Design(zeta=quoted("seismic", "15 %") / 100)
    mi = modal(iso, di)
    ri = correlation(mi.f, di.zeta)
    assert (mi.T[0], di.eta, mi.mass_share[0]) == pytest.approx(
        (quoted("seismic", "2.94"), quoted("seismic", "0.71"), 0.998), abs=5e-3
    )
    assert mi.T[0] > TD
    shear = combine(mi.shear, Rule.CQC, ri)
    drift_i = combine(mi.drift, Rule.CQC, ri)
    assert shear[1] == pytest.approx(42e3, abs=500) and max(drift_i[1:]) == pytest.approx(0.008, abs=5e-4)
    assert float(combine(mi.displacement, Rule.CQC, ri)[0]) == pytest.approx(0.166, abs=5e-4)


def test_vibroacoustics_trials():
    import math

    import numpy as np

    from vib_tutorial.core.beams_plates import Material, Plate, rect_modes
    from vib_tutorial.core.radiation import (
        AIR,
        band_edges,
        coincidence_dip,
        critical_frequency,
        maidanik,
        mass_law,
        modal_efficiency,
        point_drive,
        radiation_loss_factor,
        transmission_loss,
    )

    def sigma(m, n, f):
        return float(modal_efficiency(p, np.array([m]), np.array([n]), f)[0, 0])

    p = Plate()
    fc = critical_frequency(p)
    assert fc == pytest.approx(3990, abs=5) and AIR.c == 343
    # a corner mode
    f11 = rect_modes(p, 1)[0].freq_hz
    assert (f11, AIR.c / f11) == pytest.approx(
        (quoted("vibroacoustics", "66.6"), quoted("vibroacoustics", "5.2")), abs=0.05
    )
    assert sigma(1, 1, f11) == pytest.approx(0.037, abs=5e-4)
    f = np.geomspace(100, 5000, 400)
    s11 = np.array([sigma(1, 1, x) for x in f])
    assert f[np.argmax(s11 > 1)] == pytest.approx(450, rel=0.02) and AIR.c / 450 == pytest.approx(0.76, abs=0.005)
    assert (s11.max(), f[s11.argmax()]) == pytest.approx((1.28, 720), rel=0.02)
    # even modes, an edge mode
    modes = rect_modes(p, 10)
    assert (modes[4].m, modes[4].n, modes[4].freq_hz) == pytest.approx((2, 2, 266), abs=0.5)
    assert (sigma(2, 2, 100.0), sigma(1, 1, 100.0)) == pytest.approx((1.4e-5, 0.082), rel=0.03)
    f13 = modes[7].freq_hz
    assert (modes[7].m, modes[7].n, f13) == pytest.approx((1, 3, 435), abs=0.5)
    assert (3 * math.pi / p.b, math.pi / p.a, 2 * math.pi * f13 / AIR.c) == pytest.approx(
        (quoted("vibroacoustics", "23.6"), quoted("vibroacoustics", "5.2"), 7.98), abs=0.05
    )
    first = [sigma(md.m, md.n, md.freq_hz) for md in modes]
    assert first[7] == pytest.approx(0.059, abs=5e-4) and max(first) == first[7]
    # a point-driven panel
    freq = np.geomspace(20, 2.5 * fc, 700)
    d = point_drive(p, 0.01, (0.21, 0.13), freq)
    assert d.infinite_input == pytest.approx(565e-6, rel=0.002)

    def gap(centre):
        lo, hi = band_edges(np.array([centre]))
        sel = (freq >= lo[0]) & (freq < hi[0])
        return 10 * math.log10(d.input_power[sel].mean() / d.radiated_power[sel].mean())

    assert (gap(1000.0), gap(1000 * 10**0.8)) == pytest.approx((18, 11), abs=0.5)
    eta_rad = radiation_loss_factor(p, 1000.0)
    assert (float(maidanik(p, 1000.0)), eta_rad, 0.01 / eta_rad) == pytest.approx(
        (quoted("vibroacoustics", "0.054"), 1.5e-4, 66), rel=0.02
    )
    assert float(maidanik(p, 6300.0)) == pytest.approx(1.7, abs=0.05)
    # the mass law
    assert (p.surface_mass, float(mass_law(p, 500.0))) == pytest.approx(
        (quoted("vibroacoustics", "23.6"), quoted("vibroacoustics", "39.0")), abs=0.051
    )
    assert transmission_loss(p, 0.01, 500.0)[0] == pytest.approx(quoted("vibroacoustics", "33.8"), abs=0.05)
    # glass
    glass = Material(70e9, 0.22, 2500.0)
    window = Plate(1.2, 1.0, 0.010, glass)
    assert critical_frequency(window) == pytest.approx(1200, abs=5)
    f_dip, tl_dip = coincidence_dip(window, 0.01)
    assert (f_dip, tl_dip) == pytest.approx((1280, 19.2), rel=0.004)
    assert float(mass_law(window, 500.0)) == pytest.approx(39.5, abs=0.05)
    assert critical_frequency(Plate(1.2, 1.0, 0.004, glass)) == pytest.approx(2990, abs=5)
    assert coincidence_dip(Plate(1.2, 1.0, 0.004, glass), 0.01)[1] == pytest.approx(19.2, abs=0.05)
    # damping at coincidence
    assert (coincidence_dip(p, 0.01)[1], coincidence_dip(p, 0.1)[1]) == pytest.approx((26.1, 37.5), abs=0.05)
    assert (transmission_loss(p, 0.1, 2 * fc)[0], transmission_loss(p, 0.01, 2 * fc)[0]) == pytest.approx(
        (51.0, 41.3), abs=0.05
    )
    at = np.array([f11])  # ⟨v²⟩ at a resonance ∝ 1/η²; over a band ∝ 1/η
    light, damped = (point_drive(p, eta, (0.21, 0.13), at).mean_square_velocity[0] for eta in (0.01, 0.1))
    assert 10 * math.log10(light / damped) == pytest.approx(20, abs=0.2)
    heavy = point_drive(p, 0.1, (0.21, 0.13), freq)
    lo, hi = band_edges(np.array([2000.0]))
    sel = (freq >= lo[0]) & (freq < hi[0])
    ratio = d.mean_square_velocity[sel].mean() / heavy.mean_square_velocity[sel].mean()
    assert 10 * math.log10(ratio) == pytest.approx(10, abs=1)
    assert heavy.input_power[sel].mean() / d.input_power[sel].mean() == pytest.approx(1, abs=0.15)


def test_vibroacoustics_worked_example():
    import math

    import numpy as np

    from vib_tutorial.core.beams_plates import Plate, rect_modes
    from vib_tutorial.core.radiation import (
        AIR,
        band_edges,
        critical_frequency,
        maidanik,
        mass_law,
        radiation_loss_factor,
        transmission_loss,
    )
    from vib_tutorial.core.sea import plate_modal_density, room_modal_density
    from vib_tutorial.gui.vibroacoustics import AcousticsTrial

    t = AcousticsTrial("", "")
    assert (t.thickness / 1e3, t.nu, t.loss) == (
        quoted("vibroacoustics", "0.003"),
        quoted("vibroacoustics", "0.3"),
        quoted("vibroacoustics", "0.01"),
    )
    assert t.t60 == quoted("vibroacoustics", "0.5") and t.a * t.b == pytest.approx(quoted("vibroacoustics", "0.24"))
    p = Plate()
    assert (p.rigidity, p.surface_mass) == pytest.approx((519, quoted("vibroacoustics", "23.55")), abs=0.5)
    assert 210e9 * 0.003**3 / (12 * (1 - 0.3**2)) == pytest.approx(519, abs=0.5)
    fc = critical_frequency(p)
    assert 343**2 / (2 * math.pi) * math.sqrt(23.55 / 519) == pytest.approx(3990, abs=5)
    assert fc == pytest.approx(3990, abs=5) and 343 / fc == pytest.approx(0.086, abs=5e-4)
    assert rect_modes(p, 1)[0].freq_hz == pytest.approx(quoted("vibroacoustics", "66.6"), abs=0.05)
    n = plate_modal_density(p)
    lo, hi = band_edges(np.array([1000.0]))
    assert (lo[0], hi[0]) == pytest.approx((891, 1122), abs=0.5)
    assert 100 * (hi[0] - lo[0]) / 1000 == pytest.approx(
        quoted("vibroacoustics", "23 %"), abs=0.5
    )  # a third-octave's width
    assert n == pytest.approx(quoted("vibroacoustics", "0.0256"), abs=5e-5)
    assert n * (hi[0] - lo[0]) == pytest.approx(quoted("vibroacoustics", "5.9"), abs=0.05)
    assert sum(1 for m in rect_modes(p, 100) if lo[0] <= m.freq_hz < hi[0]) == 5
    p_in = 1 / (16 * math.sqrt(p.rigidity * p.surface_mass))
    assert p_in == pytest.approx(565e-6, abs=0.5e-6)
    assert 10 * math.log10(p_in / 1e-12) == pytest.approx(quoted("vibroacoustics", "87.5"), abs=0.05)
    mass = p.surface_mass * p.a * p.b
    energy = 565e-6 / (2 * math.pi * 1000 * 0.01)
    v2 = energy / mass
    assert (energy * 1e6, mass, v2 * 1e6, math.sqrt(v2) * 1e3) == pytest.approx(
        (
            quoted("vibroacoustics", "9.0"),
            quoted("vibroacoustics", "5.65"),
            quoted("vibroacoustics", "1.59"),
            quoted("vibroacoustics", "1.26"),
        ),
        rel=4e-3,
    )
    sigma = float(maidanik(p, 1000.0))
    assert sigma == pytest.approx(quoted("vibroacoustics", "0.054"), abs=5e-4)
    assert AIR.impedance == pytest.approx(415, abs=0.5)
    w = 415 * 0.24 * 0.054 * 1.59e-6
    assert w * 1e6 == pytest.approx(quoted("vibroacoustics", "8.6"), abs=0.05)
    assert 10 * math.log10(w / 1e-12) == pytest.approx(quoted("vibroacoustics", "69.3"), abs=0.05)
    assert 10 * math.log10(565e-6 / w) == pytest.approx(18, abs=0.5)
    eta_rad = radiation_loss_factor(p, 1000.0)
    assert (415 * 0.054 / (2 * math.pi * 1000 * 23.55), eta_rad) == pytest.approx(
        (quoted("vibroacoustics", "1.5") * 1e-4, quoted("vibroacoustics", "1.5") * 1e-4), abs=0.05e-4
    )
    assert 0.01 / eta_rad == pytest.approx(66, abs=0.5)
    x = 2 * math.pi * 500 * 23.55 / 830
    assert 10 * math.log10(1 + x * x) == pytest.approx(quoted("vibroacoustics", "39.0"), abs=0.05)
    assert float(mass_law(p, 500.0)) == pytest.approx(quoted("vibroacoustics", "39.0"), abs=0.05)
    assert 20 * math.log10(500 * p.surface_mass) - 42 == pytest.approx(quoted("vibroacoustics", "39.4"), abs=0.05)
    assert transmission_loss(p, 0.01, 500.0)[0] == pytest.approx(quoted("vibroacoustics", "33.8"), abs=0.05)
    # the room
    assert quoted("vibroacoustics", "0.161") == pytest.approx(24 * math.log(10) / AIR.c, abs=5e-4)  # Sabine's
    assert (0.161 * 50 / 0.5, 2.2 / (1000 * 0.5)) == pytest.approx(
        (quoted("vibroacoustics", "16.1"), quoted("vibroacoustics", "0.0044")), abs=1e-9
    )
    rooms = float(room_modal_density(50.0, 1000.0))
    assert rooms == pytest.approx(quoted("vibroacoustics", "15.6"), abs=0.05)
    assert rooms / n == pytest.approx(600, rel=0.02)
    # a double wall: two 10 kg/m² leaves 100 mm apart
    f0 = math.sqrt(AIR.rho * AIR.c**2 / 0.1 * (1 / 10 + 1 / 10)) / (2 * math.pi)
    assert f0 == pytest.approx(85, abs=0.5)


def test_sea_trials():
    from vib_tutorial.core.beams_plates import Plate
    from vib_tutorial.core.radiation import critical_frequency
    from vib_tutorial.core.sea import (
        Network,
        Oscillators,
        coupling_beta,
        junction_transmission,
        network,
        oscillator_pair,
    )

    # two oscillators share power
    osc = Oscillators()
    r = oscillator_pair(osc, 1.0, 0.5)
    assert (r.E1, r.E2, r.flow) == pytest.approx((0.0396, 0.0366, 0.00593), abs=6e-5)
    assert coupling_beta(osc) == pytest.approx(2.01, abs=0.005)
    # tuned
    tuned = coupling_beta(Oscillators(f2=100.0))
    assert tuned == pytest.approx(26.2, abs=0.05)
    assert tuned / coupling_beta(osc) == pytest.approx(13, abs=0.1)
    assert coupling_beta(Oscillators(f2=150.0)) == pytest.approx(0.070, abs=0.0005)
    assert Oscillators(f2=100.0).deltas == pytest.approx((25.1, 12.6), abs=0.05)
    # strong coupling
    strong = Oscillators(f2=100.0, zeta1=0.002, zeta2=0.002)
    assert coupling_beta(strong) == pytest.approx(196, abs=0.5)
    assert strong.deltas[0] == pytest.approx(2.51, abs=0.005)
    one = oscillator_pair(strong, 1.0, 0.0)
    assert one.E2 / one.E1 == pytest.approx(0.987, abs=0.0005)
    assert one.flow == pytest.approx(0.497, abs=0.0005)

    p1, p2 = Plate(), Plate(0.5, 0.4, 0.002)
    at = np.array([1000.0])

    def band(plate2, loss2=0.01):
        res = network(Network(p1, plate2, 0.01, loss2, 0.4), at)
        e, n = res.energy[0], res.density[0]
        return dict(
            share=(e[1] / n[1]) / (e[0] / n[0]),
            v2=10 * math.log10(res.velocity[0, 1] / res.velocity[0, 0]),
            spl=10 * math.log10(res.pressure[0] / 4e-10),
            by2=res.flow[0, 1, 2] / (res.flow[0, 0, 2] + res.flow[0, 1, 2]),
            eta=res.eta[0],
        )

    # plates at a corner
    assert junction_transmission(p1, p2, 1000.0) == pytest.approx(0.279, abs=0.0005)
    b = band(p2)
    assert (b["eta"][0, 1], b["eta"][1, 0]) == pytest.approx((0.00809, 0.00647), abs=6e-6)
    assert b["v2"] == pytest.approx(-0.57, abs=0.005)
    assert b["share"] == pytest.approx(0.39, abs=0.005)
    assert b["spl"] == pytest.approx(63.1, abs=0.05)
    assert b["by2"] == pytest.approx(0.281, abs=0.0005)
    # equipartition
    e = band(p2, 1e-4)
    assert e["share"] == pytest.approx(0.967, abs=0.0005)
    assert e["v2"] == pytest.approx(3.4, abs=0.05)
    assert e["spl"] == pytest.approx(66.2, abs=0.05)
    assert band(p2, 0.1)["share"] == pytest.approx(0.061, abs=0.0005)
    # a heavy wall
    wall = Plate(0.5, 0.4, 0.010)
    assert junction_transmission(p1, wall, 1000.0) == pytest.approx(0.037, abs=0.0005)
    h = band(wall)
    assert h["v2"] == pytest.approx(-16.2, abs=0.05)
    assert h["by2"] == pytest.approx(0.458, abs=0.0005)
    assert h["spl"] == pytest.approx(65.7, abs=0.05)
    assert critical_frequency(wall) == pytest.approx(1200, abs=5)


def test_fluid_structure_trials():
    from vib_tutorial.core.added_mass import (
        AIR,
        Body,
        Shape,
        avmi,
        plate_added_mass,
        plate_dry_frequency,
        plate_modal_mass,
        wet_frequency,
    )
    from vib_tutorial.core.beams_plates import Plate

    def wet(b):
        return wet_frequency(20.0, b.mass, b.added_mass)

    sphere = Body(Shape.SPHERE, 0.1)
    assert (sphere.added_mass, sphere.mass, wet(sphere)) == pytest.approx((2.09, 32.9, 19.4), rel=0.0025)
    rod = Body(Shape.CYLINDER, 0.025)
    assert (rod.added_mass, wet(rod)) == pytest.approx((1.96, 18.8), rel=0.0025)
    light = Body(Shape.CYLINDER, 0.025, 500.0)
    assert light.added_mass / light.mass == pytest.approx(2.0)
    assert wet(light) / 20 == pytest.approx(0.577, abs=0.0005)
    assert wet(light) == pytest.approx(11.5, abs=0.05)
    air = Body(Shape.CYLINDER, 0.025, fluid=AIR)
    assert air.added_mass == pytest.approx(0.00238, abs=5e-6)
    assert air.mass / air.added_mass == pytest.approx(6500, rel=0.01)
    assert 1 - wet(air) / 20 < 1e-4
    pipe = Body(Shape.CYLINDER, 0.025, outer=0.03)
    assert (pipe.coefficient, pipe.added_mass, wet(pipe)) == pytest.approx(
        (quoted("fluid_structure", "5.55"), 10.9, 15.3), rel=0.0025
    )
    tight = Body(Shape.CYLINDER, 0.025, outer=0.025 * 1.05)
    assert (tight.coefficient, tight.added_mass / tight.mass, wet(tight)) == pytest.approx((20.5, 2.6, 10.5), abs=0.05)
    r = 1.05  # the speed beside the rod: |A + B/R²| = C_a
    assert (r**2 + 1) / (r**2 - 1) == pytest.approx(20.5, abs=0.05)

    plate = Plate()
    assert plate_added_mass(plate, 1, 1) == pytest.approx(quoted("fluid_structure", "11.9"), abs=0.05)
    assert plate_modal_mass(plate) == pytest.approx(1.41, abs=0.005)
    beta = avmi(plate, 1, 1)
    f11 = plate_dry_frequency(plate, 1, 1)
    assert (beta, f11, f11 / math.sqrt(1 + beta)) == pytest.approx(
        (quoted("fluid_structure", "8.43"), quoted("fluid_structure", "66.6"), quoted("fluid_structure", "21.7")),
        rel=0.0025,
    )
    beta32 = avmi(plate, 3, 2)
    f32 = plate_dry_frequency(plate, 3, 2)
    assert (beta32, f32, f32 / math.sqrt(1 + beta32)) == pytest.approx((2.18, 369, 207), abs=0.5)
    assert beta32 == pytest.approx(2.18, abs=0.005)
    thick = Plate(thickness=0.010)
    assert plate_modal_mass(thick) / plate_modal_mass(plate) == pytest.approx(3.33, abs=0.005)
    bt, ft = avmi(thick, 1, 1), plate_dry_frequency(thick, 1, 1)
    assert (bt, ft, ft / math.sqrt(1 + bt)) == pytest.approx((2.53, 222, 118), abs=0.5)
    assert bt == pytest.approx(2.53, abs=0.005)
    assert 1 / math.sqrt(1 + bt) == pytest.approx(0.53, abs=0.005)


def test_sloshing_trials():
    from vib_tutorial.core.sloshing import (
        Damper,
        G,
        Structure,
        Tank,
        damper_response,
        frozen_response,
        mass_ratio,
        peak,
        response_band,
        rigid_mass,
        slosh_frequency,
        slosh_mass,
    )

    tank = Tank(1.0, 0.5)
    assert slosh_frequency(tank, 1) == pytest.approx(quoted("fluid_structure", "0.846"), abs=5e-4)
    assert (float(slosh_mass(tank, 1)), rigid_mass(tank)) == pytest.approx((237, 250), abs=0.5)
    assert tank.liquid_mass == 500
    shallow = Tank(1.0, 0.05)
    assert (slosh_frequency(shallow, 1), math.sqrt(G * 0.05) / 2) == pytest.approx((0.349, 0.350), abs=5e-4)
    assert float(slosh_mass(shallow, 1)) == pytest.approx(40.2, abs=0.05)
    assert rigid_mass(shallow) == pytest.approx(2.71, abs=0.005)
    deep = Tank(1.0, 2.0)
    assert slosh_frequency(deep, 1) == pytest.approx(0.884, abs=5e-4)
    assert math.sqrt(G * math.pi) / (2 * math.pi) == pytest.approx(0.884, abs=5e-4)
    assert 1 / math.pi == pytest.approx(0.32, abs=0.005)
    assert float(slosh_mass(deep, 1)) == pytest.approx(258, abs=0.5)
    assert float(slosh_mass(deep, 1)) / deep.liquid_mass == pytest.approx(0.13, abs=0.005)
    assert rigid_mass(deep) == pytest.approx(1729, abs=0.5)
    assert slosh_frequency(tank, 3) == pytest.approx(1.53, abs=0.005)
    assert float(slosh_mass(tank, 3)) == pytest.approx(quoted("fluid_structure", "9.55"), abs=0.005)
    assert float(slosh_mass(tank, 3)) / 500 == pytest.approx(0.02, abs=0.001)
    assert slosh_frequency(tank, 2) == pytest.approx(1.25, abs=0.005)
    assert float(slosh_mass(tank, 2)) == 0

    structure = Structure(20000, 0.5, 0.01)
    damper = Damper(Tank(2.0, 0.4, 2.0), 1, 0.13)
    assert slosh_frequency(damper.tank, 1) == pytest.approx(quoted("fluid_structure", "0.466"), abs=5e-4)
    assert (damper.liquid, float(damper.masses[0])) == pytest.approx((1600, 1149), abs=0.5)
    assert mass_ratio(structure, damper) == pytest.approx(quoted("fluid_structure", "0.0562"), abs=5e-5)
    f = response_band(structure)
    assert float(np.abs(frozen_response(structure, damper, f)).max()) == pytest.approx(52.0, abs=0.05)
    assert peak(structure, damper) == pytest.approx(quoted("fluid_structure", "5.58"), abs=0.005)
    bare = Damper(Tank(2.0, 0.4, 2.0), 1, 0.005)
    r = np.abs(damper_response(structure, bare, f))
    tops, _ = find_peaks(r)
    assert f[tops] == pytest.approx([0.426, 0.540], abs=5e-4)
    assert r[tops] == pytest.approx([34.5, 32.8], abs=0.05)
    mistuned = Damper(Tank(2.0, 0.25, 2.0), 1, 0.13)
    assert slosh_frequency(mistuned.tank, 1) == pytest.approx(0.382, abs=5e-4)
    assert peak(structure, mistuned) == pytest.approx(18.2, abs=0.05)
    four = Damper(Tank(2.0, 0.3, 2.0), 4, 0.13)
    assert four.liquid == pytest.approx(4800)
    assert (mass_ratio(structure, four), peak(structure, four)) == pytest.approx((0.171, 4.41), abs=5e-3)


def test_quarter_car_trials():
    from dataclasses import replace

    from vib_tutorial.core.quarter_car import (
        Car,
        Output,
        Road,
        best_damper,
        drive,
        response,
    )

    car, road = Car(), Road("C", 20.0)
    r = response(car, road)
    assert car.modes() == pytest.approx((quoted("quarter_car", "1.24"), quoted("quarter_car", "11.8")), abs=0.01)
    assert (r.rms[Output.BODY], r.load_ratio) == pytest.approx(
        (quoted("quarter_car", "1.48"), quoted("quarter_car", "24 %") / 100), abs=0.005
    )
    assert r.rms[Output.TRAVEL] == pytest.approx(0.0107, abs=5e-5)
    rough = response(car, Road("D", 30.0))
    assert (rough.rms[Output.BODY], rough.load_ratio, rough.lift_probability) == pytest.approx(
        (3.63, 0.59, 0.044), abs=0.005
    )
    dampers = np.geomspace(150.0, 10e3, 4000)
    assert best_damper(car, road, Output.BODY, dampers) == pytest.approx(830, abs=10)
    assert best_damper(car, road, Output.TYRE, dampers) == pytest.approx(2340, abs=10)
    soft = replace(car, damper=830.0)
    rs = response(soft, road)
    assert soft.body_zeta() == pytest.approx(0.17, abs=0.005)
    assert (rs.rms[Output.BODY], rs.load_ratio) == pytest.approx((1.36, 0.29), abs=0.005)
    assert rs.rms[Output.TRAVEL] == pytest.approx(0.0144, abs=5e-5)
    assert 1 - rs.rms[Output.BODY] / r.rms[Output.BODY] == pytest.approx(0.08, abs=0.005)
    firm = replace(car, damper=2340.0)
    rf = response(firm, road)
    assert (firm.body_zeta(), rf.load_ratio, rf.rms[Output.BODY]) == pytest.approx((0.48, 0.23, 1.72), abs=0.005)
    springy = replace(car, spring=10e3)
    rk = response(springy, road)
    assert (springy.modes()[0], rk.rms[Output.BODY]) == pytest.approx((0.90, 1.35), abs=0.005)
    assert 1 - rk.rms[Output.BODY] / r.rms[Output.BODY] == pytest.approx(0.09, abs=0.005)
    assert car.sprung * 9.80665 / springy.spring == pytest.approx(0.294, abs=5e-4)
    heavy = replace(car, unsprung=60.0)
    assert (heavy.modes()[1], response(heavy, road).load_ratio) == pytest.approx((9.6, 0.26), abs=0.05)
    d = drive(car, road, 60.0)
    skip = int(2.0 / (d.t[1] - d.t[0]))
    for record, spectral in (
        (d.body, r.rms[Output.BODY]),
        (d.travel, r.rms[Output.TRAVEL]),
        (d.tyre, r.rms[Output.TYRE]),
    ):
        assert np.std(record[skip:]) == pytest.approx(spectral, rel=0.02)


def test_variability_trials():
    from vib_tutorial.core.model import ChainSystem
    from vib_tutorial.core.variability import (
        Ring,
        Scatter,
        amplification,
        amplification_curve,
        monte_carlo,
        participation,
        ring_modes,
        sensitivity_sigma,
        tuned_band,
        whitehead,
    )

    chain = ChainSystem.uniform(4)
    f = np.linspace(0.1, 10.0, 200)
    springs = monte_carlo(chain, Scatter(0.0, 0.05, 0.0), 200, f, 3, 3)
    assert springs.nominal_freqs[0] == pytest.approx(quoted("variability", "1.105"), abs=5e-4)
    assert springs.freqs[:, 0].std() == pytest.approx(0.015, abs=5e-4)
    assert np.mean(springs.freqs.std(axis=0) / springs.nominal_freqs) == pytest.approx(0.014, abs=5e-4)
    assert sensitivity_sigma(chain, Scatter(0.0, 0.05, 0.0)) / springs.nominal_freqs == pytest.approx(0.0144, abs=5e-5)
    both = monte_carlo(chain, Scatter(0.05, 0.05, 0.0), 200, f, 3, 3)
    assert both.freqs[:, 0].std() == pytest.approx(0.023, abs=5e-4)
    assert np.mean(both.freqs.std(axis=0) / both.nominal_freqs) == pytest.approx(0.02, abs=1e-3)
    dampers = monte_carlo(chain, Scatter(0.0, 0.0, 0.3), 200, f, 3, 3)
    assert np.ptp(dampers.freqs, axis=0) == pytest.approx(0.0, abs=1e-12)

    tuned = Ring()
    assert tuned_band(tuned) == pytest.approx((15.9, 17.4), abs=0.05)
    p = participation(ring_modes(tuned)[1])
    assert (p.min(), p.max()) == pytest.approx((6.0, 12.0), abs=0.05)
    weak = Ring(coupling=0.005, mistuning=0.01)
    assert participation(ring_modes(weak)[1])[0] == pytest.approx(1.3, abs=0.05)
    strong = participation(ring_modes(Ring(coupling=0.2, mistuning=0.01))[1])
    assert (strong.min(), strong.max()) == pytest.approx((6.0, 12.0), abs=0.1)

    forced = Ring(coupling=0.02, mistuning=0.01)
    assert amplification(forced, 3) == pytest.approx(1.06, abs=0.005)
    _, top = amplification_curve(forced, 3, np.array([0.01, 0.02, 0.1]), 12)
    assert top == pytest.approx([1.34, 1.42, 1.28], abs=0.005)
    assert whitehead(12) == pytest.approx(quoted("variability", "2.23"), abs=0.005)


def test_variability_worked_example():
    """The Variability & localization page's worked example: the lognormal factors, mode 1's strain-energy shares
    and first-order spread against Monte Carlo, the band at the peak, the tuned ring and its engine-order peak, two
    blades sharing a mode, and Whitehead's bound."""
    import math

    import scipy.stats

    from vib_tutorial.core.model import ChainSystem
    from vib_tutorial.core.variability import (
        Ring,
        Scatter,
        forced,
        monte_carlo,
        natural_hz,
        participation,
        ring_modes,
        sensitivity_sigma,
        sweep_grid,
        whitehead,
    )
    from vib_tutorial.gui.variability_figures import pair_modes

    vk = quoted("variability", "5 %") / 100  # the springs' scatter
    s2 = math.log(1 + vk**2)
    assert s2 == pytest.approx(quoted("variability", "0.002497"), abs=5e-7)
    assert math.sqrt(s2) == pytest.approx(quoted("variability", "0.0500"), abs=5e-5)
    assert -s2 / 2 == pytest.approx(-quoted("variability", "0.00125"), abs=5e-6)
    assert scipy.stats.norm.ppf(0.95) == pytest.approx(quoted("variability", "1.645"), abs=5e-4)
    assert math.exp(-0.00125 + 1.645 * 0.05) == pytest.approx(quoted("variability", "1.084"), abs=5e-4)

    chain = ChainSystem.uniform(4)
    f, phi = natural_hz(chain)
    assert f == pytest.approx(
        [1.105, quoted("variability", "3.183"), quoted("variability", "4.877"), quoted("variability", "5.982")],
        abs=5e-4,
    )
    w2 = (2 * math.pi * f) ** 2
    shares = np.diff(phi, prepend=0.0, axis=0) ** 2 * chain.stiffness[:, None] / w2
    assert shares[:, 0] == pytest.approx(
        [
            quoted("variability", "0.431"),
            quoted("variability", "0.333"),
            quoted("variability", "0.184"),
            quoted("variability", "0.052"),
        ],
        abs=5e-4,
    )
    assert (shares**2).sum(axis=0) == pytest.approx([1 / 3] * 4, abs=5e-4)
    assert 0.5 * 5 * math.sqrt(0.333) == pytest.approx(quoted("variability", "1.44"), abs=5e-3)
    scatter = Scatter(0.0, vk, 0.0)
    sigma = sensitivity_sigma(chain, scatter)
    assert sigma[0] == pytest.approx(quoted("variability", "0.0160"), abs=5e-5)
    assert 100 * sigma / f == pytest.approx([quoted("variability", "1.44 %")] * 4, abs=5e-3)

    grid = np.geomspace(0.3 * f[0], 1.8 * f[-1], 900)  # the page's
    mc = monte_carlo(chain, scatter, 200, grid, 3, 3)
    spread = mc.freqs.std(axis=0)
    assert spread[0] == pytest.approx(quoted("variability", "0.0155"), abs=5e-5)
    assert 100 * spread[0] / f[0] == pytest.approx(quoted("variability", "1.40 %"), abs=5e-3)
    rel = 100 * spread / f
    assert (rel.min(), rel.max()) == pytest.approx(
        (quoted("variability", "1.36"), quoted("variability", "1.45 %")), abs=5e-3
    )
    assert 1 / math.sqrt(398) == pytest.approx(quoted("variability", "0.05"), abs=5e-4)
    assert 100 / math.sqrt(398) == pytest.approx(
        quoted("variability", "5.0 %"), abs=0.05
    )  # 200 chains: σ_f to about 5 %
    assert 0.05 * 0.0160 == pytest.approx(quoted("variability", "0.0008"), abs=5e-5)

    _, C, _ = chain.matrices()
    zeta = np.diag(phi.T @ C @ phi) / (2 * np.sqrt(w2))
    assert zeta[0] == pytest.approx(quoted("variability", "0.0174"), abs=5e-5)
    assert 100 * zeta[0] == pytest.approx(quoted("variability", "1.74"), abs=5e-3)
    assert scipy.stats.norm.ppf(0.5 + quoted("variability", "95 %") / 200) == pytest.approx(
        quoted("variability", "1.96"), abs=5e-3
    )
    assert 1.96 * 1.44 == pytest.approx(quoted("variability", "2.82"), abs=5e-3)
    assert 1.96 * 1.44 == pytest.approx(quoted("variability", "2.82 %"), abs=5e-3)
    assert 1 / math.sqrt(1 + (2.82 / 1.74) ** 2) == pytest.approx(quoted("variability", "0.53"), abs=5e-3)
    low, high = mc.band()
    peak = int(np.argmax(mc.nominal_frf))
    assert low[peak] / mc.nominal_frf[peak] == pytest.approx(quoted("variability", "0.54"), abs=5e-3)
    assert high[peak] / mc.nominal_frf[peak] == pytest.approx(quoted("variability", "1.00"), abs=5e-3)

    ring = Ring()
    assert ring.blade_hz == pytest.approx(quoted("variability", "15.92"), abs=5e-3)
    tuned_hz, _ = ring_modes(ring)
    n = np.arange(7)
    assert 100 * ring.coupling == quoted("variability", "5 %")
    assert ring.zeta == quoted("variability", "0.003")
    assert 1 + 4 * ring.coupling == pytest.approx(quoted("variability", "1.2"))
    bands = ring.blade_hz * np.sqrt(1 + 4 * ring.coupling * np.sin(np.pi * n / 12) ** 2)
    assert bands == pytest.approx(
        [
            quoted("variability", "15.92"),
            quoted("variability", "16.02"),
            quoted("variability", "16.31"),
            quoted("variability", "16.69"),
            quoted("variability", "17.07"),
            quoted("variability", "17.34"),
            quoted("variability", "17.43"),
        ],
        abs=5e-3,
    )
    assert np.unique(tuned_hz.round(6)) == pytest.approx(bands, abs=1e-6)
    wb, w3 = 2 * math.pi * ring.blade_hz, 2 * math.pi * bands[3]
    assert (wb, w3) == pytest.approx((quoted("variability", "100.0"), quoted("variability", "104.9")), abs=0.05)
    assert 1 / (2 * 0.003 * 100.0 * 104.9) == pytest.approx(quoted("variability", "0.0159"), abs=5e-5)
    swept = forced(ring, 3, sweep_grid(ring))
    assert swept.max() == pytest.approx(quoted("variability", "0.0159"), abs=5e-5)
    assert np.ptp(swept.max(axis=0)) == pytest.approx(0.0, abs=1e-9)

    (a0, b0, p0), (a1, b1, p1), (a5, b5, p5) = pair_modes()
    from vib_tutorial.gui.variability_figures import RATIOS

    assert RATIOS[1] == 1.0 and 100 * 0.01 == quoted("variability", "1 %")  # pair_modes' k_c = 0.01 k: blades 1 % apart
    assert (a0, b0, p0) == pytest.approx((0.707, 0.707, 2.00), abs=5e-3)
    assert math.degrees(0.5 * math.atan(2.0)) == pytest.approx(quoted("variability", "31.7"), abs=0.05)
    assert (a1, b1, p1) == pytest.approx(
        (quoted("variability", "0.85"), quoted("variability", "0.53"), quoted("variability", "1.67")), abs=5e-3
    )
    assert math.degrees(0.5 * math.atan(2 / 5)) == pytest.approx(quoted("variability", "10.9"), abs=0.05)
    assert (a5, b5, p5) == pytest.approx(
        (quoted("variability", "0.98"), quoted("variability", "0.19"), quoted("variability", "1.07")), abs=5e-3
    )
    assert participation(np.array([[math.cos(0.5 * math.atan(2.0))], [math.sin(0.5 * math.atan(2.0))]]))[0] == (
        pytest.approx(quoted("variability", "1.67"), abs=5e-3)
    )

    assert whitehead(12) == pytest.approx(quoted("variability", "2.23"), abs=5e-3)
    assert 2.23 * 0.0159 == pytest.approx(quoted("variability", "0.0355"), abs=5e-5)


def test_torsional_worked_example():
    """The Torsional vibration page's worked example: the motor and fan's mode, node and Holzer check, its damping,
    the 2× critical and the torque there and at the running speed, a softer coupling, the motor's start and
    running point; and the geared compressor's mesh mode."""
    import math

    from vib_tutorial.core.torsional import (
        COMPRESSOR,
        FAN,
        criticals,
        forced_response,
        modes,
        natural_hz,
        response_speeds,
        separation,
    )
    from vib_tutorial.core.torsional_startup import FAN_START, running_torques, steady_speed
    from vib_tutorial.gui.torsional_figures import node_fraction

    w2 = 180_000 * 12 / 20
    assert w2 == pytest.approx(108_000)
    assert math.sqrt(w2) == pytest.approx(quoted("torsional", "328.6"), abs=0.05)
    mode = modes(FAN)[0]
    assert mode.freq_hz == pytest.approx(quoted("torsional", "52.3"), abs=0.05)
    assert mode.shape == pytest.approx([1.0, -quoted("torsional", "0.2")])
    assert node_fraction() == pytest.approx(10 / 12)
    assert 100 * node_fraction() == pytest.approx(quoted("torsional", "83 %"), abs=0.5)
    assert 2 * w2 == pytest.approx(216_000) and 216_000 / 180_000 == pytest.approx(quoted("torsional", "1.2"))
    assert 2 * w2 + 10 * w2 * (1 - 1.2) == pytest.approx(0.0, abs=1e-9)

    assert mode.zeta == pytest.approx(quoted("torsional", "0.0183"), abs=5e-5)
    assert 1 / (2 * mode.zeta) == pytest.approx(quoted("torsional", "27.3"), abs=0.05)
    crit = criticals(FAN, 3600)[0]
    assert (crit.order, crit.rpm) == pytest.approx((2.0, 1569), abs=0.5)
    assert 60 * 52.3 / 2 == pytest.approx(1569, abs=0.5)
    assert 100 * separation(crit.rpm, 1480) == pytest.approx(quoted("torsional", "6.0 %"), abs=0.05)
    from vib_tutorial.gui.torsional import MARGIN

    assert MARGIN == quoted("torsional", "10 %")
    assert 100 * separation(crit.rpm, 1480) < MARGIN  # API 617's band: inside it

    assert 50 * 2 / 12 == pytest.approx(quoted("torsional", "8.33"), abs=5e-3)
    rpm = response_speeds(FAN, 3600)
    peak = forced_response(FAN, rpm).total()[0].max()
    assert peak == pytest.approx(228, abs=0.5)
    assert 8.33 * 27.3 == pytest.approx(227, abs=0.5)
    r = 2 * 1480 / 60 / mode.freq_hz
    assert 2 * 1480 / 60 == pytest.approx(quoted("torsional", "49.3"), abs=0.05)
    assert r == pytest.approx(quoted("torsional", "0.943"), abs=5e-4)
    assert 8.33 / math.sqrt((1 - 0.943**2) ** 2 + (2 * 0.0183 * 0.943) ** 2) == pytest.approx(
        quoted("torsional", "71.8"), abs=0.05
    )
    assert forced_response(FAN, np.array([1480.0])).total()[0, 0] == pytest.approx(
        quoted("torsional", "72.1"), abs=0.05
    )
    assert 1 - MARGIN / 100 == pytest.approx(quoted("torsional", "0.9"))
    assert 0.9 * 49.3 == pytest.approx(quoted("torsional", "44.4"), abs=0.05)
    assert (2 * math.pi * 44.4) ** 2 * 20 / 12 / 1000 == pytest.approx(130, abs=0.5)

    motor = FAN_START.motor
    assert motor.sync_rpm == pytest.approx(1500)
    assert motor.cage(1.0)[0] == pytest.approx(615, abs=0.5)
    assert 1 + motor.breakdown_slip**2 == pytest.approx(quoted("torsional", "1.04"))
    assert 2 * 1600 * 0.2 / 1.04 == pytest.approx(615, abs=0.5)
    w = steady_speed(FAN, FAN_START)
    assert 60 * w / (2 * math.pi) == pytest.approx(1444, abs=0.5)
    assert 100 * (1 - w / motor.sync_speed) == pytest.approx(quoted("torsional", "3.75 %"), abs=5e-3)
    assert running_torques(FAN, FAN_START)[0] == pytest.approx(580, abs=0.5)

    mesh = COMPRESSOR.links[1]
    assert COMPRESSOR.discs[2].inertia == quoted("torsional", "0.05") and mesh.base_radius == quoted("torsional", "0.3")
    assert mesh.base_radius * mesh.teeth_out / mesh.teeth_in == pytest.approx(
        quoted("torsional", "0.075")
    )  # the pinion's
    assert 3 / 4**2 == pytest.approx(quoted("torsional", "0.1875"))
    assert 2e9 * 0.075**2 / 1e7 == pytest.approx(quoted("torsional", "1.125"))
    assert 0.05 + 0.1875 == pytest.approx(quoted("torsional", "0.2375"))
    w2_mesh = 1.125e7 * 0.2375 / (0.05 * 0.1875)
    assert w2_mesh / 1e8 == pytest.approx(quoted("torsional", "2.85"), rel=5e-3)
    assert math.sqrt(w2_mesh) / (2 * math.pi) == pytest.approx(2687, abs=0.5)
    assert natural_hz(COMPRESSOR)[2] == pytest.approx(2688, abs=0.5)
    tooth = next(c for c in criticals(COMPRESSOR, 1800) if c.mode == 2)
    assert (tooth.order, tooth.rpm) == pytest.approx((120, 1344), abs=0.5)


def test_duffing_worked_example():
    """The Duffing & Mathieu page's worked example: the opening hardening spring and the parametric presets."""
    from vib_tutorial.core import duffing as d

    osc = d.PRESETS[0].osc
    assert (osc.mass, osc.stiffness, osc.damping, osc.cubic, osc.force) == (1.0, 400.0, 1.0, 1e5, 2.0)
    # small swings: f_n 3.18 Hz, ζ 0.025, 2m/c = 2 s
    assert osc.natural_hz == pytest.approx(quoted("duffing", "3.18"), abs=0.005)
    assert osc.zeta == pytest.approx(quoted("duffing", "0.025"))
    assert 100 * osc.zeta == pytest.approx(quoted("duffing", "2.5 %"))
    assert 2 * osc.mass / osc.damping == 2.0
    # the backbone at 50 mm: κ 587.5 N/m, 3.86 Hz
    assert osc.stiffness + quoted("duffing", "0.75") * osc.cubic * quoted("duffing", "0.05") ** 2 == pytest.approx(
        quoted("duffing", "587.5")
    )
    assert osc.kappa(0.05) == pytest.approx(quoted("duffing", "587.5"))
    assert float(osc.backbone_hz(0.05)) == pytest.approx(quoted("duffing", "3.86"), abs=0.005)
    # the peak: 71.5 mm on the backbone, κ 783.4, Ω 27.99 rad/s (4.45 Hz), F/cΩ = 71.5 mm, the cubic +96 %
    curve = d.response_curve(osc, 6.0)
    assert curve.tip == pytest.approx(quoted("duffing", "0.0715"), abs=0.00005)
    assert curve.tip * 1e3 == pytest.approx(quoted("duffing", "71.5"), abs=0.05)
    assert osc.kappa(0.0715) == pytest.approx(quoted("duffing", "783.4"), abs=0.05)
    assert osc.kappa(0.0715) - osc.stiffness == pytest.approx(quoted("duffing", "383.4"), abs=0.05)
    assert math.sqrt(783.4) == pytest.approx(quoted("duffing", "27.99"), abs=0.005)
    assert math.sqrt(783.4) / (2 * math.pi) == pytest.approx(quoted("duffing", "4.45"), abs=0.005)
    assert 2 / 27.99 == pytest.approx(quoted("duffing", "0.0715"), abs=0.00005)
    assert 100 * 383.4 / 400 == pytest.approx(quoted("duffing", "96 %"), abs=0.5)
    assert 100 * 0.75 * osc.cubic * curve.tip**2 / osc.stiffness == pytest.approx(
        quoted("duffing", "95.8 %"), abs=0.05
    )  # the readout
    assert osc.force / (osc.damping * 20.0) == pytest.approx(quoted("duffing", "0.1"))
    # the folds: 4.45 Hz (71.4 mm drops to 5.23 mm) and 3.64 Hz (24.0 mm jumps to 46.2 mm)
    hi, lo = sorted(curve.folds, key=lambda f: -f.freq_hz)
    assert (hi.freq_hz, hi.amplitude * 1e3, hi.lands_at * 1e3) == pytest.approx(
        (quoted("duffing", "4.45"), quoted("duffing", "71.4"), quoted("duffing", "5.23")), abs=0.005, rel=2e-3
    )
    assert hi.down and not lo.down
    assert lo.freq_hz == pytest.approx(quoted("duffing", "3.64"), abs=0.005)
    assert lo.amplitude * 1e3 == pytest.approx(quoted("duffing", "24.0"), abs=0.05)
    assert lo.lands_at * 1e3 == pytest.approx(quoted("duffing", "46.2"), abs=0.05)
    # inside the band at 4 Hz: 8.80, 52.0 (unstable), 58.3 mm; the upper checks to 1.99 N
    a, stable = d.amplitudes(osc, 4.0)
    assert a * 1e3 == pytest.approx(
        [quoted("duffing", "8.80"), quoted("duffing", "52.0"), quoted("duffing", "58.3")], abs=0.05
    )
    assert list(stable) == [True, False, True]
    # the simulated dots sit on the curve to about 1 %, on the upper branch too (κ 1.6 times k)
    for start in (a[0], a[2]):
        sim = d.DuffingSimulator(osc, d.Drive(freq_hz=4.0))
        sim.start_on(start)
        cycles = []
        while sim.t < 40.0 - 1e-9:
            cycles += sim.advance(min(5.0, 40.0 - sim.t)).cycles
        settled = np.mean([c.amplitude for c in cycles[-10:]])
        assert 100 * abs(settled / start - 1) < quoted("duffing", "1 %")
    w = 2 * math.pi * 4.0
    assert w == pytest.approx(quoted("duffing", "25.13"), abs=0.005)
    assert w * w == pytest.approx(quoted("duffing", "631.7"), abs=0.05)
    assert a[2] == pytest.approx(quoted("duffing", "0.0583"), abs=5e-5)
    assert osc.kappa(0.0583) == pytest.approx(quoted("duffing", "654.9"), abs=0.05)
    assert 654.9 - 631.7 == pytest.approx(quoted("duffing", "23.2"))
    assert 0.0583 * math.hypot(23.2, 25.13) == pytest.approx(quoted("duffing", "1.99"), abs=0.005)
    # the pump: μ 0.3 at 6.37 Hz, threshold 4ζ = 0.1, growth 1.0 /s against Floquet's 0.0498 ω_n = 0.996 /s
    lin = next(p.osc for p in d.PRESETS if p.name == "Parametric resonance")
    hard = next(p for p in d.PRESETS if p.name == "Parametric resonance, hardening")
    assert lin.pump == quoted("duffing", "0.3") and hard.osc.pump == 0.3
    assert hard.freq_hz == pytest.approx(quoted("duffing", "6.37"), abs=0.005)
    assert 4 * lin.zeta == pytest.approx(quoted("duffing", "0.1"))
    assert (0.3 / 4 - 0.025) * 20 == pytest.approx(quoted("duffing", "1.0"))
    growth = float(d.floquet_growth(lin.zeta, np.array([2.0]), np.array([0.3]))[0, 0])
    assert growth == pytest.approx(quoted("duffing", "0.0498"), abs=0.00005)
    assert 0.0498 * 20 == pytest.approx(quoted("duffing", "0.996"))
    # the tongue: undamped 5.87 to 6.83 Hz, the page's 5.89 to 6.80
    assert (1 - lin.pump / 2, 1 + lin.pump / 2) == pytest.approx(
        (quoted("duffing", "0.85"), quoted("duffing", "1.15"))
    )  # 1 ± μ/2
    assert 6.37 * math.sqrt(0.85) == pytest.approx(quoted("duffing", "5.87"), abs=0.005)
    assert 6.37 * math.sqrt(1.15) == pytest.approx(quoted("duffing", "6.83"), abs=0.005)
    assert d.tongue_hz(lin) == pytest.approx((quoted("duffing", "5.89"), quoted("duffing", "6.80")), abs=0.005)
    # the hardening spring's settled swing: mΩ²/4 = 400, √(60² − 20²) = 56.6, κ 456.6, a = 27.5 mm
    w = 2 * math.pi * hard.freq_hz
    assert w * w / 4 == pytest.approx(400.0)
    assert math.sqrt(60.0**2 - 20.0**2) == pytest.approx(quoted("duffing", "56.6"), abs=0.05)
    assert 400 + 56.6 == pytest.approx(quoted("duffing", "456.6"))
    assert math.sqrt(56.6 / 75_000) * 1e3 == pytest.approx(quoted("duffing", "27.5"), abs=0.05)
    swing = d.parametric_amplitudes(hard.osc, hard.freq_hz)[0].max()
    assert swing * 1e3 == pytest.approx(quoted("duffing", "27.5"), abs=0.05)
    ueda = next(p.osc for p in d.PRESETS if p.name == "Chaos (Ueda)")
    assert (ueda.stiffness, ueda.damping, ueda.cubic, ueda.force) == (
        0.0,
        quoted("duffing", "0.05"),
        1.0,
        quoted("duffing", "7.5"),
    )


def test_fluid_structure_worked_example():
    """The Fluid–structure page's worked example: its defaults (rod, plate, tank, structure) and the damper trial."""
    from vib_tutorial.core import added_mass as am
    from vib_tutorial.core import sloshing as sl
    from vib_tutorial.core.beams_plates import Material, Plate
    from vib_tutorial.gui.fluid_structure import TRIALS, FluidTrial

    t = FluidTrial("", "")
    assert (t.radius, t.density, t.f_dry, t.fluid) == (25.0, 7850.0, 20.0, 1000.0)
    assert t.ratio == quoted("fluid_structure", "1.5")
    assert t.radius / 1e3 == quoted("fluid_structure", "0.025")
    assert t.ratio**2 == pytest.approx(quoted("fluid_structure", "2.25"))
    assert t.fluid / t.density == pytest.approx(quoted("fluid_structure", "0.1274"), abs=5e-5)
    # Fritz's C_a: 1.67 with the pipe at twice the radius, 5.55 at 1.2 times
    assert am.Body(am.Shape.CYLINDER, 0.025, outer=0.05).coefficient == pytest.approx(
        quoted("fluid_structure", "1.67"), abs=0.005
    )
    assert am.Body(
        am.Shape.CYLINDER, 0.025, outer=0.025 * quoted("fluid_structure", "1.2")
    ).coefficient == pytest.approx(5.55, abs=0.005)
    # the rod: 15.41 kg and 1.963 kg a meter, 18.84 Hz
    rod = am.Body(am.Shape.CYLINDER, 0.025, 7850.0)
    assert rod.mass == pytest.approx(quoted("fluid_structure", "15.41"), abs=0.005)
    assert rod.added_mass == pytest.approx(quoted("fluid_structure", "1.963"), abs=0.0005)
    assert am.wet_frequency(20.0, rod.mass, rod.added_mass) == pytest.approx(
        quoted("fluid_structure", "18.84"), abs=0.005
    )
    assert 20 / math.sqrt(1 + 1000 / 7850) == pytest.approx(quoted("fluid_structure", "18.84"), abs=0.005)
    # in a pipe at 1.5R: C_a 2.6, 5.11 kg, 17.33 Hz
    piped = am.Body(am.Shape.CYLINDER, 0.025, 7850.0, outer=0.0375)
    assert piped.coefficient == pytest.approx(quoted("fluid_structure", "2.6"))
    assert piped.added_mass == pytest.approx(quoted("fluid_structure", "5.11"), abs=0.005)
    assert 20 / math.sqrt(1 + 2.6 * 0.1274) == pytest.approx(quoted("fluid_structure", "17.33"), abs=0.005)
    assert am.wet_frequency(20.0, piped.mass, piped.added_mass) == pytest.approx(
        quoted("fluid_structure", "17.33"), abs=0.005
    )
    # the plate: D 519.2, ρh 23.55, (1, 1) at 66.6 Hz; modal 1.413 kg, 11.9 kg of water, β 8.43, 21.7 Hz
    assert (t.plate_rho, t.E, t.sides) == (7850.0, 210.0, 1)
    assert (t.a, t.b, t.thickness / 1e3) == (
        quoted("fluid_structure", "0.6"),
        quoted("fluid_structure", "0.4"),
        quoted("fluid_structure", "0.003"),
    )
    assert 1 - 0.3**2 == pytest.approx(quoted("fluid_structure", "0.91"))
    plate = Plate(0.6, 0.4, 0.003, Material(210e9, 0.3, 7850.0))
    assert plate.rigidity == pytest.approx(quoted("fluid_structure", "519.2"), abs=0.05)
    assert plate.surface_mass == pytest.approx(quoted("fluid_structure", "23.55"))
    assert am.plate_dry_frequency(plate, 1, 1) == pytest.approx(quoted("fluid_structure", "66.6"), abs=0.05)
    assert math.pi / 2 * (1 / 0.6**2 + 1 / 0.4**2) * math.sqrt(519.2 / 23.55) == pytest.approx(
        quoted("fluid_structure", "66.6"), abs=0.05
    )
    assert am.plate_modal_mass(plate) == pytest.approx(quoted("fluid_structure", "1.413"), abs=0.0005)
    assert am.plate_added_mass(plate, 1, 1) == pytest.approx(quoted("fluid_structure", "11.9"), abs=0.05)
    assert am.avmi(plate, 1, 1) == pytest.approx(quoted("fluid_structure", "8.43"), abs=0.005)
    assert 1 + 8.43 == pytest.approx(quoted("fluid_structure", "9.43"))
    assert 66.6 / math.sqrt(9.43) == pytest.approx(quoted("fluid_structure", "21.7"), abs=0.05)
    assert am.avmi(plate, 1, 1, sides=2) == pytest.approx(quoted("fluid_structure", "16.86"), abs=0.005)
    assert 66.6 / math.sqrt(17.86) == pytest.approx(quoted("fluid_structure", "15.8"), abs=0.05)
    # the tank: tanh(π/2) 0.91715, ω² 28.27, 0.846 Hz; m_L 500, m₁ 236.6, m₃ 9.55, m₀ 250
    assert (t.length, t.depth, t.breadth) == (1.0, quoted("fluid_structure", "0.5"), 1.0)
    tank = sl.Tank(1.0, 0.5, 1.0)
    assert math.tanh(math.pi * 0.5) == pytest.approx(quoted("fluid_structure", "0.91715"), abs=0.000005)
    assert sl.G == pytest.approx(quoted("fluid_structure", "9.81"))
    assert math.sqrt(28.27) == pytest.approx(quoted("fluid_structure", "5.317"), abs=5e-4)
    assert 9.81 * math.pi * 0.91715 == pytest.approx(quoted("fluid_structure", "28.27"), abs=0.005)
    assert math.sqrt(28.27) / (2 * math.pi) == pytest.approx(quoted("fluid_structure", "0.846"), abs=0.0005)
    assert sl.slosh_frequency(tank, 1) == pytest.approx(quoted("fluid_structure", "0.846"), abs=0.0005)
    assert tank.liquid_mass == pytest.approx(500.0)
    assert 500 * 8 * 0.91715 / (math.pi**3 * 0.5) == pytest.approx(quoted("fluid_structure", "236.6"), abs=0.05)
    assert float(sl.slosh_mass(tank, 1)) == pytest.approx(quoted("fluid_structure", "236.6"), abs=0.05)
    assert float(sl.slosh_mass(tank, 3)) == pytest.approx(quoted("fluid_structure", "9.55"), abs=0.005)
    assert sl.rigid_mass(tank) == pytest.approx(250.0, abs=0.5)
    # as a damper on the default structure: μ 0.0117, f′ 0.497, Den Hartog 0.491 Hz; peak 50.0 → 47.7
    assert (t.structure_mass, t.structure_f) == (20.0, 0.5)
    assert (t.structure_zeta, t.damper_zeta) == (
        quoted("fluid_structure", "1 %"),
        quoted("fluid_structure", "13 %"),
    )  # %
    structure = sl.Structure(20000.0, 0.5, 0.01)
    damper = sl.Damper(tank, 1, 0.13)
    assert sl.mass_ratio(structure, damper) == pytest.approx(quoted("fluid_structure", "0.0117"), abs=0.00005)
    assert 236.6 / (20000 + 500 - 236.6) == pytest.approx(quoted("fluid_structure", "0.0117"), abs=0.00005)
    assert 0.5 * math.sqrt(20000 / 20263) == pytest.approx(quoted("fluid_structure", "0.497"), abs=0.0005)
    assert sl.loaded_frequency(structure, damper) == pytest.approx(quoted("fluid_structure", "0.497"), abs=0.0005)
    assert 1 + 0.0117 == pytest.approx(quoted("fluid_structure", "1.0117"))
    assert 0.497 / 1.0117 == pytest.approx(quoted("fluid_structure", "0.491"), abs=0.0005)
    assert sl.peak(structure, None) == pytest.approx(quoted("fluid_structure", "50.0"), abs=0.05)
    assert sl.peak(structure, damper) == pytest.approx(quoted("fluid_structure", "47.7"), abs=0.05)
    # tuned: the trial's 2 × 2 m tank at 0.4 m
    trial = next(tr for tr in TRIALS if tr.name == "A tuned liquid damper")
    assert (trial.length, trial.depth, trial.breadth) == (2.0, 0.4, 2.0)
    assert trial.depth / trial.length == pytest.approx(quoted("fluid_structure", "0.2"))  # κ₁h = (π/L)h = 0.2π
    big = sl.Tank(2.0, 0.4, 2.0)
    assert big.liquid_mass == pytest.approx(1600.0)
    assert math.tanh(0.2 * math.pi) == pytest.approx(quoted("fluid_structure", "0.5569"), abs=0.00005)
    assert math.sqrt(9.81 * math.pi / 2 * 0.5569) / (2 * math.pi) == pytest.approx(
        quoted("fluid_structure", "0.466"), abs=0.0005
    )
    assert 1600 * 8 * 0.5569 / (math.pi**3 * 0.2) == pytest.approx(1149, abs=0.5)
    assert float(sl.slosh_mass(big, 1)) == pytest.approx(1149, abs=0.5)
    tuned = sl.Damper(big, 1, 0.13)
    assert 1149 / 20451 == pytest.approx(quoted("fluid_structure", "0.0562"), abs=0.00005)
    assert sl.mass_ratio(structure, tuned) == pytest.approx(quoted("fluid_structure", "0.0562"), abs=0.00005)
    assert 0.5 * math.sqrt(20000 / 20451) == pytest.approx(quoted("fluid_structure", "0.4945"), abs=0.00005)
    assert 0.4945 / 1.0562 == pytest.approx(quoted("fluid_structure", "0.468"), abs=0.0005)
    ratio, zd, top = sl.den_hartog(0.0562)
    assert ratio == pytest.approx(1 / quoted("fluid_structure", "1.0562"))
    assert 100 * zd == pytest.approx(quoted("fluid_structure", "13.4 %"), abs=0.05)
    assert top == pytest.approx(quoted("fluid_structure", "6.05"), abs=0.005)
    assert sl.peak(structure, tuned) == pytest.approx(quoted("fluid_structure", "5.58"), abs=0.005)


def test_self_excited_worked_example():
    """The Self-excited vibration page's worked example: each model on the page's defaults."""
    from vib_tutorial.core import self_excited as se
    from vib_tutorial.gui.self_excited import DEFAULTS

    d = DEFAULTS
    # van der Pol: ζ −0.1 at rest, 2 × 10 × √0.2 = 8.94 mm
    assert (d["vdp_freq"], d["vdp_eps"], d["vdp_ref"]) == (1.0, quoted("self_excited", "0.2"), 0.01)
    vdp = se.VanDerPol(1.0, 0.2, 0.01)
    assert -vdp.eps / 2 == pytest.approx(-quoted("self_excited", "0.1"))
    assert float(vdp.theory_amplitude(0.2)) * 1e3 == pytest.approx(quoted("self_excited", "8.94"), abs=0.005)
    # galloping: 0.628 against the wind's 0.989 at 6 m/s; U_g 3.81 m/s; 0.51 D, 0.91 D (unstable), 1.28 D
    assert (d["g_side"], d["g_mass"], d["g_fn"], d["g_speed"]) == (0.1, 5.0, 2.0, 6.0)
    assert d["g_zeta"] == quoted("self_excited", "0.005")
    assert 100 * d["g_zeta"] == pytest.approx(quoted("self_excited", "0.5 %"))
    g = se.Galloping(0.1, 5.0, 2.0, 0.005, 6.0)
    assert g.rho == quoted("self_excited", "1.225") and se.PARKINSON[0] == quoted("self_excited", "2.69")
    assert 2 * math.pi * 2 == pytest.approx(quoted("self_excited", "12.57"), abs=0.005)
    assert 2 * 5 * 0.005 * 4 * math.pi == pytest.approx(quoted("self_excited", "0.628"), abs=0.0005)
    assert 4 * 5 * 0.005 * 12.57 / (1.225 * 0.1 * 2.69) == pytest.approx(quoted("self_excited", "3.81"), abs=0.005)
    assert g.onset == pytest.approx(quoted("self_excited", "3.81"), abs=0.005)
    assert 0.5 * 1.225 * 6 * 0.1 * 2.69 == pytest.approx(quoted("self_excited", "0.989"), abs=0.0005)
    branches = se.galloping_branches(g, np.array([6.0]))
    found = sorted((float(b.amplitude[0]) / g.side, b.stable) for b in branches)
    assert [a for a, _ in found] == pytest.approx(
        [quoted("self_excited", "0.51"), quoted("self_excited", "0.91"), quoted("self_excited", "1.28")], abs=0.005
    )
    assert [s for _, s in found] == [True, False, True]
    # vortex shedding: m* 10.2, m_a 1.96 kg/m, 0.275 m/s, 1.1 f_n, lock-in near 5
    assert (d["v_d"], d["v_mass"], d["v_fn"], d["v_reduced"]) == (
        quoted("self_excited", "0.05"),
        20.0,
        1.0,
        quoted("self_excited", "5.5"),
    )
    v = se.Viv(0.05, 20.0, 1.0)
    assert v.strouhal == quoted("self_excited", "0.2") and v.gamma == quoted("self_excited", "0.8")
    assert v.mass_ratio == pytest.approx(quoted("self_excited", "10.2"), abs=0.05)
    assert v.added_mass == pytest.approx(quoted("self_excited", "1.96"), abs=0.005)
    assert 5.5 * 1 * 0.05 == pytest.approx(quoted("self_excited", "0.275"))
    assert v.strouhal * 5.5 == pytest.approx(quoted("self_excited", "1.1"))
    assert v.lock_in_at == pytest.approx(5.0)
    # flutter at rest: S 0.3, I 0.05625, k_h 3553, k_θ 79.94; 0.4725ω⁴ − 999.3ω² + 284 000 = 0 → 2.93, 6.71 Hz
    assert (d["f_plunge"], d["f_pitch"], d["f_cg"], d["f_e"]) == (3.0, 6.0, 0.2, 0.3)
    f = se.Flutter()
    assert (f.semichord, f.mass, f.radius) == (quoted("self_excited", "0.15"), 10.0, quoted("self_excited", "0.5"))
    M, _, K = f.matrices(0.0)
    assert M[0, 1] == pytest.approx(quoted("self_excited", "0.3"))
    assert M[1, 1] == pytest.approx(quoted("self_excited", "0.05625"))
    assert K[0, 0] == pytest.approx(3553, abs=0.5)
    assert K[1, 1] == pytest.approx(quoted("self_excited", "79.94"), abs=0.005)
    assert 10 * 0.05625 - 0.3**2 == pytest.approx(quoted("self_excited", "0.4725"))
    assert 3553 * 0.05625 + 79.94 * 10 == pytest.approx(quoted("self_excited", "999.3"), abs=0.05)
    assert 3553 * 79.94 == pytest.approx(284_000, abs=500)
    roots = np.sort(np.roots([0.4725, -999.3, 284_000.0]))
    assert np.sqrt(roots) / (2 * math.pi) == pytest.approx(
        [quoted("self_excited", "2.93"), quoted("self_excited", "6.71")], abs=0.005
    )
    fr, _ = f.modes(np.array([0.0]))
    assert fr[0] == pytest.approx([quoted("self_excited", "2.93"), quoted("self_excited", "6.71")], abs=0.005)
    # divergence 39.2 m/s; steady flutter 20.7 m/s at 3.8 and 4.4 Hz (none with the cg on the axis); quasi-steady
    # 19.1 m/s at 3.4 and 5.0 Hz, 17.7 with the cg on the axis
    assert math.sqrt(79.94 / (2 * math.pi * 1.225 * 0.15**2 * 0.3)) == pytest.approx(
        quoted("self_excited", "39.2"), abs=0.05
    )
    assert f.divergence_speed() == pytest.approx(quoted("self_excited", "39.2"), abs=0.05)
    Uf = f.flutter_speed()
    assert Uf == pytest.approx(quoted("self_excited", "20.7"), abs=0.05)
    fr, _ = f.modes(np.array([0.999 * Uf]))
    assert fr[0] == pytest.approx([quoted("self_excited", "3.8"), quoted("self_excited", "4.4")], abs=0.05)
    assert math.isnan(f.at(x_cg=0.0).flutter_speed())
    q = f.at(quasi_steady=True)
    assert q.flutter_speed() == pytest.approx(quoted("self_excited", "19.1"), abs=0.05)
    fr, _ = q.modes(np.array([0.999 * q.flutter_speed()]))
    assert fr[0] == pytest.approx([3.4, 5.0], abs=0.05)
    assert q.at(x_cg=0.0).flutter_speed() == pytest.approx(quoted("self_excited", "17.7"), abs=0.05)


def test_active_control_worked_example():
    """The Active control page's worked example: the uniform chain under velocity feedback, spillover, lag, a TMD,
    skyhook, and the state feedback, observer, positive position feedback and shunt examples."""
    from vib_tutorial.core import active_control as ac
    from vib_tutorial.core.model import ChainSystem
    from vib_tutorial.gui.active_control import TRIALS

    chain = ChainSystem.uniform(4)
    for got, want in zip(TRIALS[0].system().matrices(), chain.matrices(), strict=True):
        assert np.allclose(got, want)
    assert np.allclose(chain.damping, 2.0)
    # the modes: 1.105 … 5.982 Hz, 1.74 … 9.40 %; shapes at m4 0.657, −0.577, 0.429, −0.228 (m1 positive)
    w, phi = ac.undamped_modes(chain)
    assert w / (2 * math.pi) == pytest.approx(
        [
            quoted("active_control", "1.105"),
            quoted("active_control", "3.183"),
            quoted("active_control", "4.877"),
            quoted("active_control", "5.982"),
        ],
        abs=0.0005,
    )
    open_loop = ac.oscillating_poles(ac.closed_loop(chain, None)[0])
    assert [p.zeta for p in open_loop] == pytest.approx(
        [0.0174, quoted("active_control", "0.05"), 0.0766, 0.0940], abs=0.00005
    )
    shown = [
        quoted("active_control", "1.74"),
        quoted("active_control", "5.00"),
        quoted("active_control", "7.66"),
        quoted("active_control", "9.40 %"),
    ]
    assert [100 * p.zeta for p in open_loop] == pytest.approx(shown, abs=0.005)
    assert np.all(phi[0] > 0)  # undamped_modes' convention, the same on every platform
    assert phi[3] == pytest.approx(
        [
            quoted("active_control", "0.657"),
            -quoted("active_control", "0.577"),
            quoted("active_control", "0.429"),
            -quoted("active_control", "0.228"),
        ],
        abs=0.0005,
    )
    assert w[1] == pytest.approx(20.0)
    assert phi[3, 1] ** 2 == pytest.approx(1 / 3)
    # velocity feedback at the tip, g = 20: mode 2 0.167 → 0.217 (page 0.214); mode 1 0.622 → 0.639 (page 0.719)
    tip = ac.Controller(3, 3, 20.0)
    est = ac.modal_damping_estimate(chain, tip)
    assert 20 * (1 / 3) / (2 * 20) == pytest.approx(quoted("active_control", "0.167"), abs=0.0005)
    assert est[1] == pytest.approx(quoted("active_control", "0.167"), abs=0.0005)
    assert 0.05 + est[1] == pytest.approx(quoted("active_control", "0.217"), abs=0.0005)  # with its own damping
    assert 20 * 0.657**2 / (2 * 2 * math.pi * 1.105) == pytest.approx(quoted("active_control", "0.622"), abs=0.0005)
    assert 0.0174 + 0.622 == pytest.approx(quoted("active_control", "0.639"), abs=0.0005)
    closed = ac.oscillating_poles(ac.closed_loop(chain, tip)[0])
    assert closed[0].zeta == pytest.approx(quoted("active_control", "0.719"), abs=0.0005)
    assert closed[1].zeta == pytest.approx(quoted("active_control", "0.214"), abs=0.0005)
    g, z = ac.best_gain(chain, tip)
    assert (g, z) == pytest.approx((25, quoted("active_control", "0.98")), abs=0.5)
    assert z == pytest.approx(quoted("active_control", "0.98"), abs=0.005)
    # spillover: −0.167 → −0.117 by the estimate, −0.10 exactly
    spill = ac.Controller(0, 3, 20.0)
    assert ac.modal_damping_estimate(chain, spill)[1] == pytest.approx(-quoted("active_control", "0.167"), abs=0.0005)
    assert 0.05 - 0.167 == pytest.approx(-quoted("active_control", "0.117"))
    poles = np.linalg.eigvals(ac.closed_loop(chain, spill)[0])
    mode2 = poles[np.argmin(np.abs(np.abs(poles) - 20.0) + (poles.imag <= 0) * 1e9)]
    assert -mode2.real / abs(mode2) == pytest.approx(-quoted("active_control", "0.10"), abs=0.005)
    # lag: 1/τ = 50 rad/s = 7.96 Hz
    assert 1 / 0.02 / (2 * math.pi) == pytest.approx(quoted("active_control", "7.96"), abs=0.005)
    # the TMD: 2.32 kg modal, 0.116 kg, 1.053 Hz, ζ 0.127
    assert 1 / 0.657**2 == pytest.approx(quoted("active_control", "2.32"), abs=0.005)
    mu = quoted("active_control", "5 %") / 100
    j, mt, kt, ct = ac.tmd_for(chain, 3, mu)
    assert (1 + mu, 3 * mu) == pytest.approx(
        (quoted("active_control", "1.05"), quoted("active_control", "0.15"))
    )  # Den Hartog's f ratio and 3μ
    assert w[0] / (2 * math.pi) == pytest.approx(quoted("active_control", "1.1055"), abs=5e-5)
    assert j == 3
    assert mt == pytest.approx(quoted("active_control", "0.116"), abs=0.0005)
    assert math.sqrt(kt / mt) / (2 * math.pi) == pytest.approx(quoted("active_control", "1.053"), abs=0.0005)
    assert 1.1055 / 1.05 == pytest.approx(quoted("active_control", "1.053"), abs=0.0005)
    assert ct / (2 * math.sqrt(kt * mt)) == pytest.approx(quoted("active_control", "0.127"), abs=0.0005)
    # skyhook: peak 1.155, 0.0408 against the passive 0.0589 at 5 f_n
    sky_zeta, passive_zeta = quoted("active_control", "0.5"), quoted("active_control", "0.1")
    assert 1 - sky_zeta**2 == pytest.approx(quoted("active_control", "0.75"))
    assert 1 / (2 * 0.5 * math.sqrt(0.75)) == pytest.approx(quoted("active_control", "1.155"), abs=0.0005)
    r = np.linspace(0.01, 3, 30001)
    assert ac.skyhook(r, passive_zeta, sky_zeta)[1].max() == pytest.approx(
        quoted("active_control", "1.155"), abs=0.0005
    )
    passive, sky, _ = ac.skyhook(np.array([5.0]), passive_zeta, sky_zeta)
    assert sky[0] == pytest.approx(quoted("active_control", "0.0408"), abs=0.00005)
    assert passive[0] == pytest.approx(quoted("active_control", "0.0589"), abs=0.00005)
    # state feedback on 1 kg, 400 N/m, 2 N·s/m to 30 rad/s, ζ 0.7: k_p 500, k_v 40
    kp, kv = 1 * 30**2 - 400, 2 * 1 * 0.7 * 30 - 2
    assert (kp, kv) == pytest.approx((500, 40))
    lam = np.roots([1.0, 2 + kv, 400 + kp])
    assert np.abs(lam) == pytest.approx([30, 30])
    assert -lam[0].real / abs(lam[0]) == pytest.approx(quoted("active_control", "0.7"))
    # its observer at 100 rad/s, 0.7: l1 138, l2 9324
    l1, l2 = 140 - 2, 10_000 - 400 - 2 * 138
    assert (l1, l2) == (138, 9324)
    A = np.array([[0.0, 1.0], [-400.0, -2.0]])
    err = np.linalg.eigvals(A - np.outer([l1, l2], [1.0, 0.0]))
    assert np.abs(err) == pytest.approx([100, 100])
    assert -err[0].real / abs(err[0]) == pytest.approx(quoted("active_control", "0.7"))

    # positive position feedback: stable for γ < 1 (here on mode 2, ζ 0.05, the filter at its frequency, ζ_f 0.3)
    def ppf_max_real(gamma: float) -> float:
        wr, zr, wf, zf = 20.0, 0.05, 20.0, 0.3
        M = np.array(
            [
                [0, 1, 0, 0],
                [-(wr**2), -2 * zr * wr, gamma * wr**2, 0],
                [0, 0, 0, 1],
                [wf**2, 0, -(wf**2), -2 * zf * wf],
            ]
        )
        return float(np.linalg.eigvals(M).real.max())

    assert ppf_max_real(0.9) < 0 < ppf_max_real(1.1)
    # a shunt: 100 nF at 50 Hz needs 101 H
    assert 1 / ((2 * math.pi * 50) ** 2 * 1e-7) == pytest.approx(101, abs=0.5)


def test_gyroscope_worked_example():
    """The Gyroscope page's worked example: the opening top (0.5 kg disk, r 40 mm, l 30 mm, 1500 rpm, let go at
    rest at 30°)."""
    import scipy.special

    from vib_tutorial.core import gyroscope as gy
    from vib_tutorial.gui.gyroscope import TRIALS

    trial = TRIALS[0]
    assert (trial.mass, trial.radius, trial.height) == (
        quoted("gyroscope", "0.5"),
        quoted("gyroscope", "0.04"),
        quoted("gyroscope", "0.03"),
    )
    assert (trial.spin_rpm, trial.tilt) == (1500.0, 30.0) and trial.gravity == quoted("gyroscope", "9.81")
    from vib_tutorial.gui.gyroscope import PLAYBACK

    assert quoted("gyroscope", "0.02") in PLAYBACK  # the slowest playback
    top, motion = trial.top, trial.motion
    # the top: I₃ 4.0, I₁ 6.5 kg·cm², mgl 0.14715 N·m, ω₃ 157.1 rad/s
    i1, _, i3 = top.moments
    assert (i3 * 1e4, i1 * 1e4) == pytest.approx((quoted("gyroscope", "4.0"), quoted("gyroscope", "6.5")))
    assert top.weight_moment == pytest.approx(quoted("gyroscope", "0.14715"))
    assert 1500 * 2 * math.pi / 60 == pytest.approx(quoted("gyroscope", "157.08"), abs=0.005)
    # sleeps above 48.9 rad/s, 467 rpm
    assert 2 * math.sqrt(6.5e-4 * 0.14715) / 4.0e-4 == pytest.approx(quoted("gyroscope", "48.9"), abs=0.05)
    assert top.sleeping_spin == pytest.approx(quoted("gyroscope", "48.9"), abs=0.05)
    assert top.sleeping_spin / gy.RPM == pytest.approx(467, abs=0.5)
    # steady precession: 2.39 and 109 rad/s; estimates 2.34 and 112
    assert 4.0e-4 * 157.08 == pytest.approx(quoted("gyroscope", "0.06283"), abs=0.000005)
    assert 6.5e-4 * math.cos(math.radians(30)) * 1e4 == pytest.approx(quoted("gyroscope", "5.629"), abs=5e-4)
    slow, fast = gy.steady_precession(top, 1500 * gy.RPM, math.radians(30))
    assert (slow, fast) == pytest.approx((quoted("gyroscope", "2.39"), 109), abs=0.5)
    assert slow == pytest.approx(quoted("gyroscope", "2.39"), abs=0.005)
    assert np.sort(np.roots([5.629e-4, -0.06283, 0.14715])) == pytest.approx(
        [quoted("gyroscope", "2.39"), 109], abs=0.5
    )
    assert 0.14715 / 0.06283 == pytest.approx(quoted("gyroscope", "2.34"), abs=0.005)
    assert 0.06283 / 5.629e-4 == pytest.approx(112, abs=0.5)
    # nutation: a_ψ 96.7, a_φ 83.7, γ 452.8; roots 0.8528, 0.8660, 19.78; 30° to 31.48°; k² 6.98e-4, K 1.5711, 67.9 ms
    assert 0.06283 / 6.5e-4 == pytest.approx(quoted("gyroscope", "96.7"), abs=0.05)
    assert 96.66 * math.cos(math.radians(30)) == pytest.approx(quoted("gyroscope", "83.7"), abs=0.05)
    assert 2 * 0.14715 / 6.5e-4 == pytest.approx(quoted("gyroscope", "452.8"), abs=0.05)
    u1, u2, u3 = motion.roots()
    assert (u1, u2, u3) == pytest.approx(
        (quoted("gyroscope", "0.8528"), quoted("gyroscope", "0.8660"), quoted("gyroscope", "19.78")), abs=0.005
    )
    assert u1 == pytest.approx(quoted("gyroscope", "0.8528"), abs=0.00005)
    assert u2 == pytest.approx(quoted("gyroscope", "0.8660"), abs=0.00005)
    lo, hi = (math.degrees(x) for x in motion.turning())
    assert (lo, hi) == pytest.approx((30.0, quoted("gyroscope", "31.48")), abs=0.005)
    assert 19.78 - 0.8528 == pytest.approx(quoted("gyroscope", "18.93"), abs=0.005)
    assert u2 - u1 == pytest.approx(quoted("gyroscope", "0.01321"), abs=0.000005)
    assert 0.01321 / 18.93 * 1e4 == pytest.approx(quoted("gyroscope", "6.98"), abs=0.005)
    assert scipy.special.ellipk(6.98e-4) == pytest.approx(quoted("gyroscope", "1.5711"), abs=0.00005)
    assert 4 * 1.5711 / math.sqrt(452.8 * 18.93) * 1e3 == pytest.approx(quoted("gyroscope", "67.9"), abs=0.05)
    assert motion.nutation_period() == pytest.approx(0.0679, abs=0.00005)
    assert motion.nutation_hz() == pytest.approx(quoted("gyroscope", "14.7"), abs=0.05)
    # the fast top's estimates: 0.0242 rad = 1.39° deep, 15.4 Hz
    assert 2 * 6.5e-4 * 0.14715 * 0.5 / 0.06283**2 == pytest.approx(quoted("gyroscope", "0.0242"), abs=0.00005)
    assert math.degrees(0.0242) == pytest.approx(quoted("gyroscope", "1.39"), abs=0.005)
    assert hi - lo == pytest.approx(quoted("gyroscope", "1.48"), abs=0.005)
    assert 0.06283 / (2 * math.pi * 6.5e-4) == pytest.approx(quoted("gyroscope", "15.4"), abs=0.05)
    # the path: cusps; 2.39 rad/s on average, a turn in 2.63 s
    assert motion.path() is gy.Path.CUSPS
    assert motion.mean_precession() == pytest.approx(quoted("gyroscope", "2.39"), abs=0.005)
    assert 2 * math.pi / 2.39 == pytest.approx(quoted("gyroscope", "2.63"), abs=0.005)


def test_diagnostics_trials():
    from dataclasses import replace

    from vib_tutorial.core.diagnostics import (
        BearingFault,
        GearFault,
        Machine,
        RunUp,
        amplitude_spectrum,
        band_pass,
        cepstrum,
        envelope_spectrum,
        kurtosis,
        line_amplitude,
        order_map,
        order_track,
        run_up,
        sidebands,
    )

    band = (2550.0, 3450.0)
    m = Machine(speed=1797 / 60)
    assert m.mesh_hz == pytest.approx(quoted("diagnostics", "808.65"))
    healthy = record_of(m)
    assert kurtosis(healthy.x) == pytest.approx(quoted("diagnostics", "2.40"), abs=0.005)
    _, c = cepstrum(healthy.x, healthy.fs)
    at = round(healthy.fs / m.speed)
    assert c[at - 2 : at + 3].max() == pytest.approx(0.027, abs=5e-4)

    ecc = replace(m, gear=GearFault.ECCENTRIC)
    s = amplitude_spectrum(record_of(ecc).x, healthy.fs)
    _, a = sidebands(s.freqs, s.amplitude, m.mesh_hz, m.speed, 1)
    assert (a[2], a[0], a[1]) == pytest.approx((0.23, 0.08, quoted("diagnostics", "1.0")), abs=0.005)

    crack = record_of(replace(m, gear=GearFault.CRACK))
    s = amplitude_spectrum(crack.x, crack.fs)
    _, a = sidebands(s.freqs, s.amplitude, m.mesh_hz, m.speed, 6)
    assert np.delete(a, 6) == pytest.approx(0.042, abs=0.006)
    _q, c = cepstrum(crack.x, crack.fs)
    assert 1e3 / m.speed == pytest.approx(33.4, abs=0.05)
    assert c[at - 2 : at + 3].max() == pytest.approx(0.12, abs=0.005)

    noisy = replace(m, noise=1.0)
    f = noisy.bearing.frequencies(m.speed)
    assert (f["BPFO"] / m.speed, f["BPFO"], f["BPFI"], 2 * f["BSF"], f["FTF"]) == pytest.approx(
        (
            quoted("diagnostics", "3.585"),
            quoted("diagnostics", "107.4"),
            quoted("diagnostics", "162.2"),
            quoted("diagnostics", "141.2"),
            11.9,
        ),
        abs=0.05,
    )
    outer = replace(noisy, bearing_fault=BearingFault.OUTER)
    r = record_of(outer)
    s = amplitude_spectrum(r.x, r.fs)
    d = outer.defect_hz()
    assert line_amplitude(s.freqs, s.amplitude, d) == pytest.approx(0.049, abs=5e-4)
    assert np.median(s.amplitude[(s.freqs > 50) & (s.freqs < 700)]) == pytest.approx(0.043, abs=5e-4)
    assert (kurtosis(r.x), kurtosis(band_pass(r.x, r.fs, band))) == pytest.approx((3.1, 5.6), abs=0.05)
    e = envelope_spectrum(r.x, r.fs, band)
    assert [line_amplitude(e.freqs, e.amplitude, h * d) for h in (1, 2, 3)] == pytest.approx(
        [0.45, 0.34, 0.23], abs=0.005
    )
    assert np.median(e.amplitude[(e.freqs > 20) & (e.freqs < 500)]) < 0.045
    for fault, (centre, lower, upper) in (
        (BearingFault.INNER, (0.31, 0.16, 0.13)),
        (BearingFault.BALL, (0.29, 0.14, 0.135)),
    ):
        mm = replace(noisy, bearing_fault=fault)
        r = record_of(mm)
        e = envelope_spectrum(r.x, r.fs, band)
        d, fm = mm.defect_hz(), mm.modulation_hz()
        assert [line_amplitude(e.freqs, e.amplitude, v) for v in (d, d - fm, d + fm)] == pytest.approx(
            [centre, lower, upper], abs=0.005
        )

    slow, fast = RunUp(), RunUp(duration=4.0)
    fine = np.linspace(slow.start, slow.end, 2000)
    steady = slow.steady(1, fine)
    assert (steady.max(), 60 * fine[steady.argmax()]) == pytest.approx((211, 2406), abs=1)
    om = order_map(run_up(slow))
    one, two = order_track(om, 1), order_track(om, 2)
    assert (one.max(), 60 * om.speed[one.argmax()]) == pytest.approx((209, 2445), abs=1)
    assert 60 * om.speed[two.argmax()] == pytest.approx(1220, abs=5)
    assert (two.max(), slow.steady(2, fine).max()) == pytest.approx((28.5, 33.3), abs=0.05)
    om = order_map(run_up(fast))
    one = order_track(om, 1)
    assert (one.max(), 60 * om.speed[one.argmax()]) == pytest.approx((179, 2531), abs=1)
    assert 1 - one.max() / steady.max() == pytest.approx(quoted("diagnostics", "0.15"), abs=0.005)
    sweep = 2 * slow.zeta * slow.mode / ((fast.end - fast.start) / fast.duration)
    assert (sweep, sweep * slow.mode) == pytest.approx(
        (quoted("diagnostics", "0.16"), quoted("diagnostics", "6.4")), abs=0.005
    )


def record_of(machine):
    from vib_tutorial.core.diagnostics import record

    return record(machine)


def test_deriving_trials():
    from dataclasses import replace

    from vib_tutorial.core.bounce_pitch import Bump, Car, Coordinates, matrices, modes, newton, ride, slipped

    def hz(car, coords=Coordinates.CENTRE, e=0.0):
        return [m.frequency for m in modes(car, coords, e)]

    car = Car()
    m1, m2 = modes(car)
    assert (m1.frequency, m2.frequency) == pytest.approx(
        (quoted("deriving", "1.25"), quoted("deriving", "1.41")), abs=0.005
    )
    assert (m1.node, m2.node) == pytest.approx((-quoted("deriving", "1.92"), quoted("deriving", "0.88")), abs=0.005)
    assert not m1.inside(car) and m2.inside(car)
    # no static coupling
    level = replace(car, k_rear=32e3)
    assert level.k_front * level.front == pytest.approx(level.k_rear * level.rear) == pytest.approx(48e3)
    assert hz(level) == pytest.approx((quoted("deriving", "1.23"), 1.27), abs=0.005)
    # the elastic center
    assert car.elastic_centre() == pytest.approx(-quoted("deriving", "0.15"))
    at = matrices(car, Coordinates.POINT, -0.15)
    assert abs(at.K[0, 1]) < 1e-9 and at.M[0, 1] == pytest.approx(180.0)
    assert hz(car, Coordinates.POINT, -0.15) == pytest.approx(
        (quoted("deriving", "1.25"), quoted("deriving", "1.41")), abs=0.005
    )
    # moments about the front axle, and the slip
    s = slipped(car, 1.2)
    wrong = np.sort(np.sqrt(np.linalg.eigvals(np.linalg.solve(s.M, s.K)).real)) / (2 * np.pi)
    assert wrong == pytest.approx((0.93, quoted("deriving", "1.38")), abs=0.005)
    # the ends' coordinates
    assert matrices(car, Coordinates.ENDS).M[0, 1] == pytest.approx(18.1, abs=0.05)
    rows = newton(car, Coordinates.ENDS).M
    np.testing.assert_allclose(rows, [[667, 533], [751, -751]], atol=0.5)
    # a dynamic index of 1
    one = replace(car, gyration=1.342)
    assert one.dynamic_index == pytest.approx(1.0, abs=0.002) and math.sqrt(1.2 * 1.5) == pytest.approx(1.34, abs=0.005)
    assert one.end_masses == pytest.approx((666.7, 533.3), abs=0.05)
    assert hz(one) == pytest.approx(one.end_frequencies(), abs=0.002)
    assert one.end_frequencies() == pytest.approx((quoted("deriving", "1.23"), quoted("deriving", "1.38")), abs=0.005)
    assert [m.node for m in modes(one)] == pytest.approx(
        (-quoted("deriving", "1.5"), quoted("deriving", "1.2")), abs=0.005
    )
    # over a hump, either way round
    hump = Bump()
    assert car.wheelbase / hump.speed == pytest.approx(0.32, abs=0.005)
    for kf, kr, ends, settled in ((48e3, 30e3, (1.35, 1.19), 1.95), (30e3, 48e3, (1.07, 1.51), 1.52)):
        c = replace(car, k_front=kf, k_rear=kr)
        assert c.end_frequencies() == pytest.approx(ends, abs=0.005)
        r = ride(c, hump)
        assert np.degrees(np.abs(r.pitch[r.after()]).max()) == pytest.approx(settled, abs=0.005)
    assert 1 - 1.068 / 1.51 == pytest.approx(0.29, abs=0.005)


def test_deriving_pendulum_trials():
    from vib_tutorial.core.double_pendulum import (
        Angles,
        DoublePendulum,
        divergence,
        matrices,
        modes,
        newton,
        slipped,
        swing,
    )
    from vib_tutorial.core.pendulum import measured_period

    p = DoublePendulum()
    slow, fast = modes(p)
    assert (slow.frequency, fast.frequency) == pytest.approx(
        (quoted("deriving", "0.540"), quoted("deriving", "1.30")), abs=0.005
    )
    assert (slow.ratio, fast.ratio) == pytest.approx(
        (quoted("deriving", "1.41"), -quoted("deriving", "1.41")), abs=0.005
    )
    assert p.separate_frequencies() == pytest.approx(
        (quoted("deriving", "0.705"), quoted("deriving", "0.705")), abs=0.0005
    )
    assert matrices(p).M[0, 1] == pytest.approx(0.25) and matrices(p).K[0, 1] == 0
    # the free-body diagram's rows, and the slip
    np.testing.assert_allclose(newton(p).M, [[0.5, 0], [0.5, 0.5]])
    s = slipped(p)
    wrong = np.sqrt(np.linalg.eigvals(np.linalg.solve(s.M, s.K)).real) / (2 * np.pi)
    assert wrong == pytest.approx((quoted("deriving", "0.705"), quoted("deriving", "0.705")), abs=0.0005)
    # the elbow's bend
    bend = matrices(p, Angles.RELATIVE)
    assert (bend.K[0, 1], bend.M[0, 0]) == pytest.approx((4.91, 1.25), abs=0.005)
    assert [m.frequency for m in modes(p, Angles.RELATIVE)] == pytest.approx(
        (quoted("deriving", "0.540"), quoted("deriving", "1.30")), abs=0.005
    )
    assert modes(p, Angles.RELATIVE)[0].shape[1] == pytest.approx(0.41, abs=0.005)
    # released in the slower mode, at 10° and at 40°
    for start, whole, apart in (((10, 14.1), 1.858, 1.8), ((40, 56.6), 1.924, 77)):
        r = swing(p, tuple(np.radians(start)), 20.0)
        assert measured_period(r.t, r.whole[:, 0]) == pytest.approx(whole, abs=0.0005)
        assert measured_period(r.t, r.linear[:, 0]) == pytest.approx(1.853, abs=0.0005)
        assert np.degrees(np.abs(r.whole[:, 0] - r.linear[:, 0]).max()) == pytest.approx(apart, rel=0.03)
    assert 1.924 / 1.853 - 1 == pytest.approx(0.038, abs=0.0005)
    assert 100 * (1.924 / 1.853 - 1) == pytest.approx(quoted("deriving_pendulum", "4 %"), abs=0.5)  # "about 4 %"
    # chaos
    start = (np.radians(120), np.radians(120))
    d = divergence(p, start, 15.0)
    assert d.doubling_time() == pytest.approx(0.49, abs=0.005)
    assert d.apart[np.searchsorted(d.t, 10.0)] > 1.0
    assert swing(p, start).drift < 1e-9


def test_deriving_worked_example():
    """The notes' worked example: the car as the page opens, in (x_G, θ), and the equal double pendulum."""
    from vib_tutorial.core.bounce_pitch import Car, Coordinates, matrices, modes
    from vib_tutorial.core.double_pendulum import DoublePendulum
    from vib_tutorial.core.double_pendulum import modes as pendulum_modes
    from vib_tutorial.gui.deriving import DerivingTrial

    car = DerivingTrial("", "").car
    assert car == Car()
    assert car.gyration == quoted("deriving", "1.3") and car.gyration**2 == pytest.approx(quoted("deriving", "1.69"))
    assert car.front + car.rear == pytest.approx(quoted("deriving", "2.7"))
    m, J, a, b, k = 1200.0, 1200 * 1.3**2, 1.2, 1.5, 40e3
    assert J == pytest.approx(2028) and car.inertia == pytest.approx(J)
    K = [[2 * k, k * a - k * b], [k * a - k * b, k * a**2 + k * b**2]]
    assert K == [[80_000, -12_000], [-12_000, 147_600]] and k * a == 48_000 and k * b == 60_000
    mats = matrices(car, Coordinates.CENTRE)
    np.testing.assert_allclose(mats.M, [[m, 0], [0, J]])
    np.testing.assert_allclose(mats.K, K)
    np.testing.assert_allclose(mats.C, [[5000, -750], [-750, 9225]])
    # uncoupled
    assert math.sqrt(80_000 / 1200) == pytest.approx(quoted("deriving", "8.165"), abs=5e-4)
    assert math.sqrt(147_600 / 2028) == pytest.approx(quoted("deriving", "8.531"), abs=5e-4)
    assert car.uncoupled_frequencies() == pytest.approx(
        (quoted("deriving", "1.30"), quoted("deriving", "1.36")), abs=0.005
    )
    assert (8.165 / (2 * math.pi), 8.531 / (2 * math.pi)) == pytest.approx(
        (quoted("deriving", "1.30"), quoted("deriving", "1.36")), abs=0.005
    )
    # the quadratic, its coefficients as quoted, and its roots from them
    A, B, C = m * J, m * 147_600 + J * 80_000, 80_000 * 147_600 - 12_000**2
    assert (A, B, C) == pytest.approx((2.434e6, 3.394e8, 1.166e10), rel=5e-4)
    roots = np.sort(np.roots([A, -B, C]))
    assert roots == pytest.approx((quoted("deriving", "61.45"), quoted("deriving", "78.00")), abs=0.005)
    rounded = np.sort(np.roots([2.434e6, -3.394e8, 1.166e10]))
    assert rounded == pytest.approx((quoted("deriving", "61.45"), quoted("deriving", "78.00")), rel=0.01)
    assert np.sqrt([61.45, 78.00]) / (2 * math.pi) == pytest.approx(
        (quoted("deriving", "1.25"), quoted("deriving", "1.41")), abs=0.005
    )
    assert [md.frequency for md in modes(car)] == pytest.approx(
        (quoted("deriving", "1.25"), quoted("deriving", "1.41")), abs=0.005
    )
    assert 1.25 < 1.30 and 1.41 > 1.36  # one below both uncoupled, one above
    # the nodes, from the first row
    assert 61.45 * 1200 == pytest.approx(73_740) and 78.00 * 1200 == pytest.approx(93_600)
    assert 12_000 / (80_000 - 73_740) == pytest.approx(quoted("deriving", "1.92"), abs=0.005)
    assert 12_000 / (80_000 - 93_600) == pytest.approx(-quoted("deriving", "0.88"), abs=0.005)
    assert [md.node for md in modes(car)] == pytest.approx(
        (-quoted("deriving", "1.92"), quoted("deriving", "0.88")), abs=0.005
    )
    # the other coordinates
    assert -12_000 / 80_000 == car.elastic_centre() == -0.15
    assert 1.69 / (1.2 * 1.5) == pytest.approx(quoted("deriving", "0.939"), abs=5e-4)
    assert car.dynamic_index == pytest.approx(quoted("deriving", "0.939"), abs=5e-4)
    assert 1200 * 1.5 / 2.7 == pytest.approx(667, abs=0.5) and car.end_masses == pytest.approx((666.7, 533.3), abs=0.05)
    assert car.end_frequencies() == pytest.approx((quoted("deriving", "1.23"), quoted("deriving", "1.38")), abs=0.005)
    assert (math.sqrt(40e3 / 667) / (2 * math.pi), math.sqrt(40e3 / 533) / (2 * math.pi)) == pytest.approx(
        (quoted("deriving", "1.23"), quoted("deriving", "1.38")), abs=0.005
    )
    # Olley: 30 % more static deflection at the front is a ride frequency 12 % lower
    assert 1 / math.sqrt(1.3) == pytest.approx(quoted("deriving", "0.88"), abs=0.005)
    assert 100 * (1 - 1 / math.sqrt(1.3)) == pytest.approx(quoted("deriving", "12 %"), abs=0.5)
    assert 1 + quoted("deriving", "30 %") / 100 == pytest.approx(quoted("deriving", "1.3"))
    # the equal double pendulum: (ω²L/g)² − 4(ω²L/g) + 2 = 0
    assert np.sort(np.roots([1, -4, 2])) == pytest.approx((2 - math.sqrt(2), 2 + math.sqrt(2)))
    p = DoublePendulum()
    assert p.upper_length == p.lower_length == quoted("deriving", "0.5")
    assert p.gravity == quoted("deriving", "9.81")
    w2 = (2 - math.sqrt(2)) * 9.81 / 0.5, (2 + math.sqrt(2)) * 9.81 / 0.5
    assert w2 == pytest.approx((quoted("deriving", "11.49"), quoted("deriving", "66.99")), abs=0.005)
    assert [md.frequency for md in pendulum_modes(p)] == pytest.approx(
        (quoted("deriving", "0.540"), quoted("deriving", "1.30")), abs=0.005
    )
    assert np.sqrt([11.49, 66.99]) / (2 * math.pi) == pytest.approx(
        (quoted("deriving", "0.540"), quoted("deriving", "1.30")), abs=0.005
    )
    assert p.separate_frequencies() == pytest.approx(
        (quoted("deriving", "0.705"), quoted("deriving", "0.705")), abs=0.0005
    )


def test_frame_builder_worked_example():
    import scipy.optimize

    from vib_tutorial.core.frame import STEEL, assemble, cantilever, modes

    frame = cantilever(1.0, 4)
    sec = frame.elements[0].section
    E, rho, A, I = STEEL.E, STEEL.rho, sec.area, sec.inertia
    L = frame.nodes[1].x - frame.nodes[0].x
    assert L == quoted("frame_builder", "0.25")
    assert (A * 1e6, I * 1e12 / 1e5, E * A / 1e6, E * I / 1e3, rho * A) == pytest.approx(
        (
            2500,
            quoted("frame_builder", "5.208"),
            525,
            quoted("frame_builder", "109.4"),
            quoted("frame_builder", "19.63"),
        ),
        abs=0.05,
    )
    assert (E * A / L / 1e9, 12 * E * I / L**3 / 1e6, 6 * E * I / L**2 / 1e6) == pytest.approx(
        (quoted("frame_builder", "2.10"), quoted("frame_builder", "84.0"), quoted("frame_builder", "10.5"))
    )
    assert (4 * E * I / L / 1e6, 2 * E * I / L / 1e6) == pytest.approx(
        (quoted("frame_builder", "1.75"), quoted("frame_builder", "0.875"))
    )
    a = assemble(frame)
    assert (a.K[3, 3] / 1e9, a.K[4, 4] / 1e6, a.K[5, 5] / 1e6, a.K[4, 5]) == pytest.approx(
        (quoted("frame_builder", "4.20"), 168, quoted("frame_builder", "3.50"), 0)
    )
    assert (156 / 420, rho * A * L, a.M[4, 4]) == pytest.approx(
        (quoted("frame_builder", "0.3714"), quoted("frame_builder", "4.906"), quoted("frame_builder", "3.645")),
        abs=5e-4,
    )
    assert int(a.free.sum()) == 12
    _, _, K = a.reduced()
    q = np.linalg.solve(K, 1000.0 * np.eye(12)[:, 10])  # 1000 N up at the tip, v5
    assert (1000 * q[10], 1000 * q[11]) == pytest.approx(
        (quoted("frame_builder", "3.048"), quoted("frame_builder", "4.571")), abs=5e-4
    )
    assert (1000 * 1000 / (3 * E * I), 1000 * 1000 / (2 * E * I)) == pytest.approx(
        (quoted("frame_builder", "3.048"), quoted("frame_builder", "4.571")), abs=5e-4
    )
    c = math.sqrt(E * I / (rho * A))
    roots = [scipy.optimize.brentq(lambda b: math.cos(b) * math.cosh(b) + 1, a, a + 2) for a in (1.0, 4.0, 7.0)]
    assert roots == pytest.approx(
        [quoted("frame_builder", "1.875"), quoted("frame_builder", "4.694"), quoted("frame_builder", "7.855")],
        abs=0.0005,
    )
    euler = np.array([b * b * c / (2 * math.pi) for b in roots])
    assert c == pytest.approx(quoted("frame_builder", "74.65"), abs=0.005)
    assert euler == pytest.approx(
        [quoted("frame_builder", "41.78"), quoted("frame_builder", "261.8"), quoted("frame_builder", "733.1")], abs=0.05
    )
    hz = modes(frame).hz
    assert hz[:4] == pytest.approx(
        [quoted("frame_builder", "41.78"), quoted("frame_builder", "262.1"), quoted("frame_builder", "738.7"), 1301],
        abs=0.5,
    )
    assert 100 * (hz[:3] / euler - 1) == pytest.approx(
        [quoted("frame_builder", "0.003 %"), quoted("frame_builder", "0.12 %"), quoted("frame_builder", "0.77 %")],
        abs=0.005,
    )
    bar = math.sqrt(E / rho) / 4
    assert (math.sqrt(E / rho), bar, 100 * (hz[3] / bar - 1)) == pytest.approx(
        (5172, 1293, quoted("frame_builder", "0.6 %")), abs=0.5
    )
    table = {
        1: (quoted("frame_builder", "41.97"), quoted("frame_builder", "413.6"), 1426),
        2: (quoted("frame_builder", "41.80"), quoted("frame_builder", "264.0"), quoted("frame_builder", "893.0")),
        4: (41.78, 262.1, 738.7),
        8: (41.78, 261.8, quoted("frame_builder", "733.5")),
    }
    for parts, row in table.items():
        assert modes(cantilever(1.0, parts)).hz[:3] == pytest.approx(row, rel=6e-4)


def test_signals_worked_example():
    from vib_tutorial.core.signals import Source, Window, sample, spectrum, window_props

    fs, n = 1024.0, 1024
    source = Source()  # the page's opening tones: 1 at 50 Hz, 0.1 at 120 Hz
    assert (n / fs, fs / n, fs / 2, fs / 2.56) == (1.0, 1.0, 512.0, 400.0)
    from vib_tutorial.core.signals import FILTER_RATIO

    assert FILTER_RATIO == quoted("signals", "2.56")
    assert 2.56 * 120 / fs == pytest.approx(quoted("signals", "0.3"))
    assert 0.3**16 * 1e9 == pytest.approx(quoted("signals", "4.3"), abs=0.05)
    assert 2.56 * 624 / fs == pytest.approx(quoted("signals", "1.56"))
    gains = source.gain(np.array([120.0, 400.0, 624.0]), fs, True)
    assert gains[0] == pytest.approx(quoted("signals", "1.000"), abs=1e-8)
    assert gains[1] == pytest.approx(1 / math.sqrt(2))
    assert (1.56**16, gains[2], 20 * math.log10(gains[2])) == pytest.approx(
        (1230, quoted("signals", "0.0285"), -quoted("signals", "30.9")), rel=2e-3
    )
    _, x = sample(source, fs, n, True)
    spec = spectrum(x, fs, n, Window.HANN)
    props = window_props(Window.HANN)
    assert (props.coherent_gain, props.enbw) == (0.5, 1.5)
    readings = [
        quoted("signals", "0.500"),
        quoted("signals", "1.000"),
        quoted("signals", "0.500"),
        quoted("signals", "0.050"),
        quoted("signals", "0.100"),
        quoted("signals", "0.050"),
    ]
    assert spec.amplitude[[49, 50, 51, 119, 120, 121]] == pytest.approx(readings, abs=5e-4)
    assert 20 * math.log10(spec.amplitude[50]) == pytest.approx(quoted("signals", "0.00"), abs=0.005)  # +0.00 dB
    assert props.coherent_gain == quoted("signals", "0.5")
    assert spec.amplitude[49] == pytest.approx(quoted("signals", "0.5"), abs=5e-4)
    assert spec.amplitude[119] == pytest.approx(quoted("signals", "0.05"), abs=5e-4)
    assert spec.amplitude[120] == pytest.approx(quoted("signals", "0.1"), abs=5e-4)
    assert float(np.mean(x**2)) == pytest.approx(quoted("signals", "0.505"))
    assert float(np.sum(spec.psd) * spec.df) == pytest.approx(quoted("signals", "0.505"))
    assert (
        spec.psd[50]
        == pytest.approx(quoted("signals", "0.5") / quoted("signals", "1.5"))
        == pytest.approx(quoted("signals", "0.333"), abs=5e-4)
    )
    assert 1 / math.sqrt(2) == pytest.approx(quoted("signals", "0.707"), abs=5e-4)
    level = 20 * math.log10(1 / math.sqrt(2) / 1e-6)
    assert (level, level - 20) == pytest.approx((quoted("signals", "117.0"), quoted("signals", "97.0")), abs=0.05)
    third = 1000 * 10 ** np.array([-1.3, -0.9])
    edges = np.array([third * 10**-0.05, third * 10**0.05])
    assert edges.T.ravel() == pytest.approx(
        [quoted("signals", "44.7"), quoted("signals", "56.2"), quoted("signals", "112.2"), quoted("signals", "141.3")],
        abs=0.05,
    )
    w50, w120 = 2 * math.pi * 50, 2 * math.pi * 120
    assert (1000 / w50, 1e6 / w50**2, 100 / w120, 1e5 / w120**2) == pytest.approx(
        (quoted("signals", "3.18"), quoted("signals", "10.1"), quoted("signals", "0.133"), quoted("signals", "0.176")),
        rel=4e-3,
    )
    assert ((1 / w50**2) / (0.1 / w120**2), (120 / 50) ** 2) == pytest.approx(
        (57.6, quoted("signals", "5.76")), rel=1e-3
    )
    # the notes' other numbers: a drift, octave and third ratios
    drift = quoted("signals", "0.01")
    assert (drift * 10, 0.5 * drift * 10**2) == pytest.approx((quoted("signals", "0.1"), quoted("signals", "0.5")))
    assert (10**0.3, 10**0.1, 20 * math.log10(2)) == pytest.approx(
        (2.0, quoted("signals", "1.26"), quoted("signals", "6.02")), abs=0.005
    )


def test_modal_test_worked_example():
    from vib_tutorial.core import ChainSystem, frf, impact_spectrum, modal_analysis
    from vib_tutorial.gui.modal_test import suggested_fs

    chain = ChainSystem.uniform(4)  # 1 kg, 400 N/m, 2 N·s/m
    modes = [m.damped for m in modal_analysis(chain).modes]
    w = np.array([40 * math.sin(math.radians(a)) for a in (10, 30, 50, 70)])
    assert w == pytest.approx(
        [
            quoted("modal_test", "6.95"),
            quoted("modal_test", "20.0"),
            quoted("modal_test", "30.6"),
            quoted("modal_test", "37.6"),
        ],
        abs=0.05,
    )
    fn = np.array([d.fn_hz for d in modes])
    assert fn == pytest.approx(w / (2 * math.pi))
    assert fn == pytest.approx(
        [
            quoted("modal_test", "1.105"),
            quoted("modal_test", "3.183"),
            quoted("modal_test", "4.877"),
            quoted("modal_test", "5.982"),
        ],
        abs=5e-4,
    )
    zeta = np.array([d.zeta for d in modes])
    assert zeta == pytest.approx(w / 400)
    assert zeta == pytest.approx(
        [
            quoted("modal_test", "0.0174"),
            quoted("modal_test", "0.0500"),
            quoted("modal_test", "0.0766"),
            quoted("modal_test", "0.0940"),
        ],
        abs=5e-5,
    )
    assert [d.fd_hz for d in modes] == pytest.approx(
        [
            quoted("modal_test", "1.105"),
            quoted("modal_test", "3.179"),
            quoted("modal_test", "4.862"),
            quoted("modal_test", "5.956"),
        ],
        abs=5e-4,
    )
    from vib_tutorial.core.measurement import AA_CUTOFF, Processing

    assert 100 * AA_CUTOFF == quoted("modal_test", "80 %")
    assert AA_CUTOFF == quoted("modal_test", "0.8")
    assert 2 / AA_CUTOFF == quoted("modal_test", "2.5")  # fs = 2.5 f_max
    assert 100 / quoted("modal_test", "2.56") * 2 == pytest.approx(
        quoted("modal_test", "78 %"), abs=0.5
    )  # the analyzers' 2.56
    assert 100 * Processing().exp_end == quoted("modal_test", "5 %")
    fs = suggested_fs(modal_analysis(chain))
    assert fs >= quoted("modal_test", "1.25") * 5.956 * 2 / 0.8
    assert (1.25 * 5.956 * 2 / 0.8, fs, fs / 2, 0.8 * fs / 2) == pytest.approx(
        (quoted("modal_test", "18.6"), 19.0, quoted("modal_test", "9.5"), quoted("modal_test", "7.6")), abs=0.05
    )
    T = 1024 / fs
    assert (T, 1 / T, 10 * T / 60) == pytest.approx(
        (quoted("modal_test", "53.9"), quoted("modal_test", "0.0186"), 9.0), abs=0.05
    )
    widths = [2 * d.zeta * d.fn_hz for d in modes]
    assert widths == pytest.approx(
        [
            quoted("modal_test", "0.0384"),
            quoted("modal_test", "0.318"),
            quoted("modal_test", "0.747"),
            quoted("modal_test", "1.12"),
        ],
        rel=4e-3,
    )
    assert [b * T for b in widths] == pytest.approx(
        [
            quoted("modal_test", "2.07"),
            quoted("modal_test", "17.2"),
            quoted("modal_test", "40.3"),
            quoted("modal_test", "60.6"),
        ],
        rel=3e-3,
    )
    assert widths[0] * 2048 / fs == pytest.approx(quoted("modal_test", "4.1"), abs=0.05)
    decay = modes[0].zeta * w[0]
    assert math.exp(-decay * T) * 100 < quoted("modal_test", "1 %")  # no leakage
    assert (decay, decay * T, math.exp(-decay * T) * 100) == pytest.approx(
        (quoted("modal_test", "0.121"), quoted("modal_test", "6.50"), quoted("modal_test", "0.15 %")), abs=5e-3
    )
    tau_w = T / math.log(20)
    assert (tau_w, 1 / (tau_w * w[0])) == pytest.approx(
        (quoted("modal_test", "18.0"), quoted("modal_test", "0.0080")), abs=0.05
    )
    assert 1 / (tau_w * w[0]) == pytest.approx(quoted("modal_test", "0.0080"), abs=5e-5)
    pulse = quoted("modal_test", "0.05")  # the hammer's 50 ms
    levels = 20 * np.log10(impact_spectrum(pulse, np.array([d.fd_hz for d in modes])))
    assert levels[3] == pytest.approx(-quoted("modal_test", "0.73"), abs=0.005)
    assert [round(v) + 0 for v in levels] == [0, 0, 0, -1]
    assert impact_spectrum(pulse, np.array([quoted("modal_test", "1.5") / pulse]))[0] == pytest.approx(
        0.0, abs=1e-12
    )  # its first zero
    result = modal_analysis(chain)
    assert np.abs(result.modes[1].shape) == pytest.approx([1, 1, 0, 1], abs=1e-9)
    assert min(abs(m.shape[3]) for m in result.modes) > 0.3  # every mode moves at m4
    shape = np.array([quoted("modal_test", "0.347"), quoted("modal_test", "0.653"), quoted("modal_test", "0.879"), 1.0])
    assert np.abs(result.modes[0].shape) == pytest.approx(shape, abs=5e-4)
    phi2 = 1 / float(np.sum(shape**2))
    assert (phi2, phi2 / (2 * 0.0174)) == pytest.approx(
        (quoted("modal_test", "0.431"), quoted("modal_test", "12.4")), abs=0.05
    )
    assert 2 * 0.0174 == pytest.approx(quoted("modal_test", "0.0347"), abs=5e-4)
    assert 0.431 / 0.0347 == pytest.approx(quoted("modal_test", "12.4"), abs=0.05)
    accel = abs(-((2 * math.pi * modes[0].fn_hz) ** 2) * frf(chain, np.array([modes[0].fn_hz]), 3)[0, 3])
    assert accel == pytest.approx(quoted("modal_test", "12.4"), abs=0.05)
    armature = quoted("modal_test", "0.1")
    assert armature * accel == pytest.approx(quoted("modal_test", "1.24"), abs=0.005)
    assert 100 * (1 - abs(1 / (1 + 1.24j))) == pytest.approx(quoted("modal_test", "37 %"), abs=0.5)
    assert (abs(1 / (1 + 1.24j)), abs(1 / (1 + armature * accel * 1j))) == pytest.approx(
        (quoted("modal_test", "0.63"), quoted("modal_test", "0.63")), abs=5e-3
    )


def test_oma_worked_example():
    from types import SimpleNamespace

    from vib_tutorial.core import ChainSystem, modal_analysis
    from vib_tutorial.core.oma import cross_spectra
    from vib_tutorial.gui.oma import CYCLES, OmaPage

    chain = ChainSystem.uniform(4)
    result = modal_analysis(chain)
    f1, z1 = result.modes[0].fn_hz, result.modes[0].zeta_modal
    shown = [quoted("oma", "1.105"), quoted("oma", "3.183"), quoted("oma", "4.877"), quoted("oma", "5.982")]
    assert [m.fn_hz for m in result.modes] == pytest.approx(shown, abs=5e-4)
    shown = [quoted("oma", "0.0174"), quoted("oma", "0.0500"), quoted("oma", "0.0766"), quoted("oma", "0.0940")]
    assert [m.zeta_modal for m in result.modes] == pytest.approx(shown, abs=5e-5)
    from vib_tutorial.core.oma import HARMONIC_ZETA, MAC_MIN

    assert MAC_MIN == quoted("oma", "0.8") and 100 * HARMONIC_ZETA == quoted("oma", "1 %")
    from vib_tutorial.gui.oma import OmaSetup

    assert OmaSetup().noise == quoted("oma", "2 %")  # %, of each channel
    fs, duration, block = OmaPage._suggested(SimpleNamespace(result=result))  # the page's own choice
    assert (fs, duration, block) == (19.0, 1809, 2048)
    assert (CYCLES / f1, duration * fs) == pytest.approx((1809, 34371), abs=0.5)
    assert z1 * f1 / 2 == pytest.approx(quoted("oma", "0.0096"), abs=5e-5)
    assert (fs / (0.5 * z1 * f1), 19 / 0.0096) == pytest.approx((1979, 1979), abs=1)
    assert (2 * z1 * f1, 2 * z1 * f1 * block / fs) == pytest.approx(
        (quoted("oma", "0.0384"), quoted("oma", "4.1")), abs=0.05
    )
    assert 0.0384 / 0.00928 == pytest.approx(quoted("oma", "4.1"), abs=0.05)
    assert fs / block == pytest.approx(quoted("oma", "0.00928"), abs=5e-6)
    spectra = cross_spectra(np.zeros((34371, 1)), fs, block)
    assert spectra.blocks == 32 == (34371 - 2048) // 1024 + 1
    assert 100 / math.sqrt(32) == pytest.approx(quoted("oma", "18 %"), abs=0.5)
    rows = round(2 * fs / f1)
    assert (2 * fs / f1, rows, 2 * rows - 1, (2 * rows - 1) / fs) == pytest.approx(
        (quoted("oma", "34.4"), 34, 67, quoted("oma", "3.5")), abs=0.05
    )
    assert (4 * rows, 2 * 4, min(15 * 4, 120)) == (136, 8, 60)
    f2 = result.modes[1].fn_hz
    assert ((f1 + f2) / 2, (1 + quoted("oma", "2 %") / 100) * f2) == pytest.approx(
        (quoted("oma", "2.144"), quoted("oma", "3.247")), abs=5e-4
    )
    assert (3 / 8) / (1 / 4) == 1.5
    t = np.linspace(0, 2 * math.pi, 100001)
    assert np.mean(np.sin(t) ** 4) / np.mean(np.sin(t) ** 2) ** 2 == pytest.approx(quoted("oma", "1.5"), abs=1e-3)
    # Scaling by 0.1 kg on m4
    heavier = modal_analysis(ChainSystem(np.array([1, 1, 1, 1.1]), chain.stiffness, chain.damping, chain.friction))
    g0, g1 = f1, heavier.modes[0].fn_hz
    assert (g0, g1, 100 * (g1 / g0 - 1)) == pytest.approx(
        (quoted("oma", "1.1055"), quoted("oma", "1.0823"), -quoted("oma", "2.1 %")), abs=5e-3
    )
    added = quoted("oma", "0.1")  # kg on m4
    assert (1.1055**2 - 1.0823**2) / (1.0823**2 * added) == pytest.approx(quoted("oma", "0.433"), abs=5e-4)
    psi = np.abs(result.modes[0].shape)
    assert psi[3] == 1.0 and 1 / float(np.sum(psi**2)) == pytest.approx(quoted("oma", "0.431"), abs=5e-4)
    alpha2 = (g0**2 - g1**2) / (g1**2 * 0.1)
    assert 100 * (alpha2 * float(np.sum(psi**2)) - 1) == pytest.approx(quoted("oma", "0.6 %"), abs=0.05)
    shape4 = np.abs(result.modes[3].shape)
    h4, k4 = result.modes[3].fn_hz, heavier.modes[3].fn_hz
    a4 = (h4**2 - k4**2) / (k4**2 * 0.1 * shape4[3] ** 2)
    assert shape4[3] == pytest.approx(quoted("oma", "0.347"), abs=5e-4)
    assert 100 * (1 - k4 / h4) == pytest.approx(quoted("oma", "0.23 %"), abs=5e-3)
    assert 100 * (1 - a4 * float(np.sum(shape4**2))) == pytest.approx(quoted("oma", "13 %"), abs=0.5)
    assert sorted(psi) == pytest.approx([0.347, quoted("oma", "0.653"), quoted("oma", "0.879"), 1.0], abs=5e-4)


def test_model_updating_worked_example():
    from vib_tutorial.core import ChainSystem, modal_analysis
    from vib_tutorial.core.model_updating import Kind, Parameter, frequency_sensitivity

    chain = ChainSystem.uniform(4)
    exact = modal_analysis(chain)
    stiff = chain.stiffness.copy()
    from vib_tutorial.gui.model_updating import TRIALS as UPDATING

    factor = quoted("model_updating", "1.3")
    assert UPDATING[0].changes == (("k", 2, factor),)  # the first trial, the page's opening error
    assert 100 * (factor - 1) == pytest.approx(quoted("model_updating", "30 %"))
    stiff[2] *= factor
    model = modal_analysis(ChainSystem(chain.masses, stiff, chain.damping))
    f0 = np.array([m.fn_hz for m in exact.modes])
    f1 = np.array([m.fn_hz for m in model.modes])
    assert f1 == pytest.approx(
        [
            quoted("model_updating", "1.129"),
            quoted("model_updating", "3.312"),
            quoted("model_updating", "4.906"),
            quoted("model_updating", "6.379"),
        ],
        abs=5e-4,
    )
    assert 100 * (f1 / f0 - 1) == pytest.approx(
        [
            quoted("model_updating", "2.2"),
            quoted("model_updating", "4.1"),
            quoted("model_updating", "0.6"),
            quoted("model_updating", "6.6"),
        ],
        abs=0.05,
    )
    assert 100 * (f1[3] / f0[3] - 1) == pytest.approx(quoted("model_updating", "6.6 %"), abs=0.05)
    assert [m.fn_hz for m in exact.modes] == pytest.approx(
        [
            quoted("model_updating", "1.105"),
            quoted("model_updating", "3.183"),
            quoted("model_updating", "4.877"),
            quoted("model_updating", "5.982"),
        ],
        abs=5e-4,
    )
    macs = [
        float((a.shape @ b.shape) ** 2 / ((a.shape @ a.shape) * (b.shape @ b.shape)))
        for a, b in zip(exact.modes, model.modes)
    ]
    assert macs == pytest.approx(
        [
            quoted("model_updating", "0.999"),
            quoted("model_updating", "0.993"),
            quoted("model_updating", "0.992"),
            quoted("model_updating", "0.988"),
        ],
        abs=5e-4,
    )
    springs = frequency_sensitivity(chain, exact, [Parameter(Kind.SPRING, j) for j in range(4)])
    masses = frequency_sensitivity(chain, exact, [Parameter(Kind.MASS, j) for j in range(4)])
    phi = exact.modes[0].shape_mass_normalized
    assert (exact.modes[0].omega_n, *phi) == pytest.approx(
        (
            quoted("model_updating", "6.946"),
            quoted("model_updating", "0.2280"),
            quoted("model_updating", "0.4285"),
            quoted("model_updating", "0.5774"),
            quoted("model_updating", "0.6565"),
        ),
        abs=1e-4,
    )
    assert 0.5774 - 0.4285 == pytest.approx(quoted("model_updating", "0.149"), abs=5e-4)
    assert 400 * 0.149**2 / (2 * 6.946**2) == pytest.approx(quoted("model_updating", "0.092"), abs=5e-4)
    assert springs == pytest.approx(
        np.array(
            [
                [
                    quoted("model_updating", "0.216"),
                    quoted("model_updating", "0.167"),
                    quoted("model_updating", "0.092"),
                    quoted("model_updating", "0.026"),
                ],
                [
                    quoted("model_updating", "0.167"),
                    0,
                    quoted("model_updating", "0.167"),
                    quoted("model_updating", "0.167"),
                ],
                [
                    quoted("model_updating", "0.092"),
                    quoted("model_updating", "0.167"),
                    quoted("model_updating", "0.026"),
                    quoted("model_updating", "0.216"),
                ],
                [
                    quoted("model_updating", "0.026"),
                    quoted("model_updating", "0.167"),
                    quoted("model_updating", "0.216"),
                    quoted("model_updating", "0.092"),
                ],
            ]
        ).T.T,
        abs=5e-4,
    )
    assert min(macs) > quoted("model_updating", "0.98")  # every pair above 0.98
    assert min(macs) == pytest.approx(quoted("model_updating", "0.99"), abs=0.005)  # one spring 30 % off: near 0.99
    scaled = ChainSystem(chain.masses * 1.1, chain.stiffness * 1.1, chain.damping)
    assert quoted("model_updating", "1.1") == 1.1  # every mass and spring by 1.1: no frequency moves
    assert [m.fn_hz for m in modal_analysis(scaled).modes] == pytest.approx(f0)
    assert min(macs) > quoted("model_updating", "0.9")  # a good correlation
    assert np.sum(springs, axis=1) == pytest.approx(quoted("model_updating", "0.5"))
    assert np.sum(masses, axis=1) == pytest.approx(-quoted("model_updating", "0.5"))
    share = 2 * springs[:, 2]
    assert 100 * share == pytest.approx([18, 33, 5, 43], abs=0.5)
    linear = 100 * (1.3 ** springs[:, 2] - 1)
    assert linear == pytest.approx(
        [
            quoted("model_updating", "2.4"),
            quoted("model_updating", "4.5"),
            quoted("model_updating", "0.7"),
            quoted("model_updating", "5.8 %"),
        ],
        abs=0.05,
    )
    y = np.log(f1 / f0)
    s = springs[:, 2]
    assert (float(s @ y), float(s @ s)) == pytest.approx((0.0226, 0.0833), abs=5e-5)
    theta = float(s @ y) / float(s @ s)
    assert (theta, math.exp(theta), 0.0226 / 0.0833) == pytest.approx(
        (quoted("model_updating", "0.271"), quoted("model_updating", "1.31"), quoted("model_updating", "0.271")),
        abs=5e-3,
    )
    scaled = modal_analysis(ChainSystem(1.1 * chain.masses, 1.1 * chain.stiffness, chain.damping))
    assert [m.fn_hz for m in scaled.modes] == pytest.approx(list(f0))


def test_test_planning_worked_example():
    from vib_tutorial.core.modal import mac
    from vib_tutorial.core.test_planning import (
        Method,
        driving_point_residue,
        ods,
        ods_parts,
        place,
        quality,
        shares,
        undamped_modes,
    )
    from vib_tutorial.gui.planning_figures import ellipse_axes

    chain = ChainSystem.uniform(4)
    assert (chain.masses[0], chain.stiffness[0], chain.damping[0]) == (1.0, 400.0, 2.0)
    hz, phi = undamped_modes(chain)
    assert hz == pytest.approx(
        [
            quoted("test_planning", "1.105"),
            quoted("test_planning", "3.183"),
            quoted("test_planning", "4.877"),
            quoted("test_planning", "5.982"),
        ],
        abs=5e-4,
    )
    T = phi[:, :3]
    four = phi[:, 3] * np.sign(phi[1, 3])
    assert four == pytest.approx(
        [
            -quoted("test_planning", "0.4285"),
            quoted("test_planning", "0.6565"),
            -quoted("test_planning", "0.5774"),
            quoted("test_planning", "0.2280"),
        ],
        abs=5e-5,
    )
    E = shares(T)
    assert E == pytest.approx(1 - four**2) and E.sum() == pytest.approx(3)
    assert E == pytest.approx(
        [
            quoted("test_planning", "0.816"),
            quoted("test_planning", "0.569"),
            quoted("test_planning", "0.667"),
            quoted("test_planning", "0.948"),
        ],
        abs=5e-4,
    )
    for method in (Method.EFI, Method.EFI_DPR, Method.EVEN):
        p = place(T, hz[:3], method, 3)
        assert list(p.sensors + 1) == [1, 3, 4]
    assert [s.dropped for s in place(T, hz[:3], Method.EFI, 3).steps] == [1]
    assert driving_point_residue(T, hz[:3]).shape == (4,)
    q = quality(T, np.array([0, 2, 3]))
    assert q.information == pytest.approx(1 - E[1])
    assert 100 * q.information == pytest.approx(quoted("test_planning", "43.1 %"), abs=0.05)
    a = T[1] * np.sign(T[1, 0])
    assert a == pytest.approx(
        [quoted("test_planning", "0.4285"), quoted("test_planning", "0.5774"), -quoted("test_planning", "0.2280")],
        abs=5e-5,
    )
    assert 1 - 0.569 == pytest.approx(quoted("test_planning", "0.431"))
    assert 1 + a**2 / 0.431 == pytest.approx(
        [quoted("test_planning", "1.43"), quoted("test_planning", "1.77"), quoted("test_planning", "1.12")], abs=0.005
    )
    assert q.noise == pytest.approx(
        [quoted("test_planning", "1.19"), quoted("test_planning", "1.33"), quoted("test_planning", "1.06")], abs=0.005
    )
    assert int(np.argmax(q.noise)) == 1 and int(np.argmax(a**2)) == 1
    assert (0.2280 * 0.5774 - 0.6565 * 0.5774, 0.2280**2 + 0.5774**2 + 0.6565**2) == pytest.approx(
        (quoted("test_planning", "0.1316") - quoted("test_planning", "0.3790"), quoted("test_planning", "0.8164")),
        abs=2e-4,
    )
    assert 0.5774**2 + 0.5774**2 == pytest.approx(quoted("test_planning", "0.6667"), abs=1e-4)
    assert (0.1316 - 0.3790) ** 2 / (0.8164 * 0.6667) == pytest.approx(quoted("test_planning", "0.112"), abs=5e-4)
    assert q.auto_mac[0, 1] == pytest.approx(quoted("test_planning", "0.112"), abs=5e-4) and q.worst == q.auto_mac[0, 1]
    assert quality(T, np.array([0, 1, 2])).information == pytest.approx(quoted("test_planning", "0.052"), abs=5e-4)
    # the figure's ellipses: a circle, a little larger, long and thin
    sizes = [np.sqrt(np.linalg.eigvalsh(c)) for c in ellipse_axes()]
    assert sizes[0] == pytest.approx([1, 1]) and 1 < sizes[1].max() < 2 < 3 < sizes[2].max()
    # where to excite: mass 3 is mode 2's node, mass 4 has every target
    assert abs(T[2, 1]) < 1e-12
    assert np.abs(T[3]) == pytest.approx(
        [quoted("test_planning", "0.6565"), quoted("test_planning", "0.5774"), quoted("test_planning", "0.4285")],
        abs=5e-5,
    )
    # the ODS at 1 Hz, force and reference on mass 4
    w1, w = 2 * np.pi * hz[0], 2 * np.pi
    assert (w1, w, w1**2, w**2) == pytest.approx(
        (
            quoted("test_planning", "6.946"),
            quoted("test_planning", "6.283"),
            quoted("test_planning", "48.25"),
            quoted("test_planning", "39.48"),
        ),
        abs=5e-3,
    )
    assert chain.damping[0] / chain.stiffness[0] == quoted("test_planning", "0.005")
    assert w1**2 - w**2 == pytest.approx(quoted("test_planning", "8.77"), abs=0.005)
    assert w * 0.005 * w1**2 == pytest.approx(quoted("test_planning", "1.52"), abs=5e-3)
    assert 0.6565**2 == pytest.approx(quoted("test_planning", "0.4310"), abs=5e-5)
    x = ods(chain, 3, 1.0)
    one = 0.4310 / (48.25 - 39.48 + 1j * 1.52)
    assert abs(one) == pytest.approx(quoted("test_planning", "0.0484"), abs=5e-5)
    assert abs(x[3]) == pytest.approx(quoted("test_planning", "0.0496"), abs=5e-5)
    parts = ods_parts(chain, 3, 1.0)
    assert abs(parts[3, 0]) == pytest.approx(quoted("test_planning", "0.0484"), abs=5e-5)
    share = np.max(np.abs(parts), axis=0) / np.max(np.abs(x))
    assert share[0] == pytest.approx(quoted("test_planning", "0.977"), abs=5e-4)
    assert mac(x, phi[:, 0]) == pytest.approx(quoted("test_planning", "0.999"), abs=5e-4)
    assert np.abs(x) / np.abs(x).max() == pytest.approx(
        [quoted("test_planning", "0.326"), quoted("test_planning", "0.619"), quoted("test_planning", "0.852"), 1],
        abs=5e-4,
    )
    assert int(np.argmax(np.abs(x))) == 3


def test_nonlinear_forces_worked_example():
    from vib_tutorial.core import force_id as fid
    from vib_tutorial.core import nonlinear_test as nl
    from vib_tutorial.core.test_planning import undamped_modes
    from vib_tutorial.gui.nonlinear_forces import BAND, STEPS, TRIALS, chain

    s = chain()
    assert (s.masses[0], s.stiffness[0], s.damping[0]) == (1.0, 1000.0, 4.0)
    hz, phi = undamped_modes(s)
    w1 = 2 * np.pi * hz[0]
    assert (hz[0], w1) == pytest.approx(
        (quoted("nonlinear_forces", "2.240"), quoted("nonlinear_forces", "14.07")), abs=5e-3
    )
    assert w1**2 == pytest.approx(quoted("nonlinear_forces", "198.1"), abs=0.05)
    assert phi[:, 0] == pytest.approx(
        [
            quoted("nonlinear_forces", "0.3280"),
            quoted("nonlinear_forces", "0.5910"),
            quoted("nonlinear_forces", "0.7370"),
        ],
        abs=5e-5,
    )
    assert s.damping[0] / s.stiffness[0] == quoted("nonlinear_forces", "0.004")
    assert 0.004 * 14.07 / 2 == pytest.approx(quoted("nonlinear_forces", "0.02814"), abs=5e-6)
    assert 0.7370**2 / (2 * 0.02814 * 198.1) == pytest.approx(quoted("nonlinear_forces", "0.0487"), abs=5e-5)
    freqs = np.linspace(BAND[0] * hz[0], BAND[1] * hz[0], STEPS)
    assert freqs[1] - freqs[0] == pytest.approx(quoted("nonlinear_forces", "0.0094"), abs=5e-5)
    linear = nl.stepped_sine(s, nl.Element(), 2, 1.0, freqs)
    assert np.abs(linear.X[:, 2]).max() == pytest.approx(quoted("nonlinear_forces", "0.0487"), abs=5e-5)
    # the cubic spring's lean at 0.5 N
    cubic = TRIALS[1].element
    assert cubic.k3 == 2e6 and cubic.spring == 0
    sweep = nl.stepped_sine(s, cubic, 2, quoted("nonlinear_forces", "0.5"), freqs)
    i = int(np.argmax(np.abs(sweep.X[:, 2])))
    assert sweep.hz[i] == pytest.approx(quoted("nonlinear_forces", "2.319"), abs=5e-4)
    assert 1000 * abs(sweep.X[i, 0]) == pytest.approx(quoted("nonlinear_forces", "9.82"), abs=5e-3)
    assert abs(sweep.X[i, 2]) / 0.5 == pytest.approx(quoted("nonlinear_forces", "0.0483"), abs=5e-5)
    assert 9.82 / 1000 == pytest.approx(quoted("nonlinear_forces", "0.00982"))
    assert 0.75 * 2e6 * 0.00982**2 == pytest.approx(quoted("nonlinear_forces", "144.6"), abs=0.05)
    assert phi[0, 0] ** 2 == pytest.approx(quoted("nonlinear_forces", "0.1076"), abs=5e-5)
    assert 198.1 + 144.6 * 0.1076 == pytest.approx(quoted("nonlinear_forces", "213.7"), abs=0.05)
    assert (np.sqrt(213.7), np.sqrt(213.7) / (2 * np.pi)) == pytest.approx(
        (quoted("nonlinear_forces", "14.62"), quoted("nonlinear_forces", "2.326")), abs=5e-3
    )
    # friction's fade
    t = TRIALS[3]
    assert t.friction[2] == quoted("nonlinear_forces", "0.1")
    assert t.levels[0] == quoted("nonlinear_forces", "0.2")
    assert t.levels[2] == 2.0
    for F, X3, ceq, tol, added, peak in (
        (
            0.2,
            quoted("nonlinear_forces", "3.55"),
            quoted("nonlinear_forces", "2.55"),
            5e-3,
            quoted("nonlinear_forces", "1.39"),
            quoted("nonlinear_forces", "0.0177"),
        ),
        (
            2.0,
            quoted("nonlinear_forces", "91.2"),
            quoted("nonlinear_forces", "0.099"),
            5e-4,
            quoted("nonlinear_forces", "0.054"),
            quoted("nonlinear_forces", "0.0456"),
        ),
    ):
        sw = nl.stepped_sine(t.system(), t.element, 2, F, freqs)
        i = int(np.argmax(np.abs(sw.X[:, 2])))
        assert sw.hz[i] == pytest.approx(quoted("nonlinear_forces", "2.234"), abs=5e-4)
        assert 2 * np.pi * sw.hz[i] == pytest.approx(quoted("nonlinear_forces", "14.04"), abs=5e-3)
        assert 1000 * abs(sw.X[i, 2]) == pytest.approx(X3, abs=0.05)
        assert 4 * 0.1 / (np.pi * 14.04 * X3 / 1000) == pytest.approx(ceq, abs=tol)
        assert F > 1 or X3 / 1000 == pytest.approx(quoted("nonlinear_forces", "0.00355"))
        assert ceq * 0.7370**2 == pytest.approx(added, abs=tol)
        total = 0.004 * 198.1 + ceq * 0.7370**2
        assert 0.7370**2 / (14.07 * total) == pytest.approx(peak, abs=5e-5)
        assert abs(sw.X[i, 2]) / F == pytest.approx(peak, abs=5e-5)
    assert (0.004 * 198.1, 0.7370**2, 0.79 + 1.39) == pytest.approx(
        (quoted("nonlinear_forces", "0.79"), quoted("nonlinear_forces", "0.5432"), quoted("nonlinear_forces", "2.18")),
        abs=5e-3,
    )
    # force identification
    t = TRIALS[6]
    h = fid.step_for(s)
    assert (fid.SAMPLES, h * 1000, h * fid.SAMPLES, 4 / hz[0]) == pytest.approx(
        (
            300,
            quoted("nonlinear_forces", "5.95"),
            quoted("nonlinear_forces", "1.79"),
            quoted("nonlinear_forces", "1.79"),
        ),
        abs=5e-3,
    )
    truth = fid.force_history(t.shape, fid.SAMPLES, h, t.width)
    G = fid.markov(s, t.hit, t.sensors, t.output, h, fid.SAMPLES)
    assert t.sensors == (0,)
    assert t.output is fid.Output.DISPLACEMENT
    assert t.noise == quoted("nonlinear_forces", "2 %")  # %
    p = fid.set_up(G, truth, t.noise / 100)
    assert (p.s[0] * 1e3, p.s[-3] * 1e10) == pytest.approx((7.7, quoted("nonlinear_forces", "1.4")), rel=0.02)
    assert p.s[-2] < 1e-17
    assert np.log10(p.s[0] / p.s[-3]) == pytest.approx(quoted("nonlinear_forces", "7.7"), abs=0.05)
    assert p.noise * 1e4 == pytest.approx(quoted("nonlinear_forces", "1.70"), abs=5e-3)
    lam = fid.l_curve(p).corner
    assert lam * 1e5 == pytest.approx(quoted("nonlinear_forces", "5.44"), abs=5e-3)
    assert 10**t.log_lam == pytest.approx(lam, abs=5e-8)
    assert int((p.s > lam).sum()) == 37
    f = fid.tikhonov(p, lam)
    assert np.linalg.norm(G @ f - p.y) * 1e4 == pytest.approx(quoted("nonlinear_forces", "1.63"), abs=5e-3)
    assert 100 * fid.error(f, truth) == pytest.approx(quoted("nonlinear_forces", "15 %"), abs=0.5)
    assert (f.max(), truth.max()) == pytest.approx(
        (quoted("nonlinear_forces", "0.95"), quoted("nonlinear_forces", "0.997")), abs=5e-3
    )
    ld = fid.discrepancy(p)
    fd = fid.tikhonov(p, ld)
    assert ld * 1e4 == pytest.approx(quoted("nonlinear_forces", "1.27"), abs=5e-3) and int((p.s > ld).sum()) == 34
    assert np.linalg.norm(G @ fd - p.y) == pytest.approx(p.noise, rel=1e-3)
    assert 100 * fid.error(fd, truth) == pytest.approx(quoted("nonlinear_forces", "18 %"), abs=0.5)
    t = TRIALS[7]
    G = fid.markov(s, t.hit, t.sensors, t.output, h, fid.SAMPLES)
    p = fid.set_up(G, truth, t.noise / 100)
    assert G[0, 0] == pytest.approx(quoted("nonlinear_forces", "0.98"), abs=5e-3)
    assert np.log10(p.s[0] / p.s[-1]) == pytest.approx(quoted("nonlinear_forces", "3.0"), abs=0.05)
    assert 10**t.log_lam == pytest.approx(quoted("nonlinear_forces", "0.024"), abs=5e-4)
    assert 100 * fid.error(fid.tikhonov(p, 10**t.log_lam), truth) < quoted("nonlinear_forces", "6 %")


def test_rotor_equations_of_motion():
    """Jeffcott rotor's part 2: the gyroscopic split, K_shaft, the beam elements and the eigenproblem."""
    from vib_tutorial.core.rotor import POISSON, STEEL, Bearing, RotorSystem, beam_mass, beam_stiffness
    from vib_tutorial.core.rotor_modes import whirl_modes

    s = RotorSystem()
    d, L = s.diameter, s.length
    inertia, area = math.pi * d**4 / 64, math.pi * d**2 / 4
    ei = s.youngs * inertia
    assert round(ei, 2) == quoted("rotor_notes", "98.17")
    assert (s.ip, s.id) == (1.25e-3, quoted("rotor_notes", "6.25") * 1e-4)
    # The disk's own tilt on rigid bearings: k_θ = 12EI/L, the θx entry of K_shaft.
    k_theta = 12 * ei / L
    ks = s.shaft_stiffness()
    assert round(k_theta) == 2356 and ks[2, 2] == pytest.approx(k_theta)
    assert round(math.sqrt(k_theta / s.id) / (2 * math.pi), 1) == quoted("rotor_notes", "309.0")
    rpm = 2 * math.pi / 60
    w = 6000 * rpm
    assert round(s.ip * w, 3) == quoted("rotor_notes", "0.785")

    def split(ip: float) -> tuple[float, float]:
        a = ip * w
        r = math.sqrt(a * a + 4 * s.id * k_theta)
        return (a + r) / (2 * s.id) / (2 * math.pi), (r - a) / (2 * s.id) / (2 * math.pi)

    rigid = Bearing(kx=1e10, ky=1e10, cx=100.0, cy=100.0, mass=0.1)
    for ip, fw, bw in (
        (s.ip, quoted("rotor_notes", "424.8"), quoted("rotor_notes", "224.8")),
        (s.id, quoted("rotor_notes", "363.0"), quoted("rotor_notes", "263.0")),
    ):
        assert [round(f, 1) for f in split(ip)] == [fw, bw]
        m = whirl_modes(s.with_(bearing_a=rigid, bearing_b=rigid, ip=ip), w)
        tilt = sorted((round(f, 1), wr > 0) for f, wr in zip(m.freq_hz, m.whirl, strict=True) if 100 < f < 1000)
        assert tilt == [(bw, False), (fw, True)]
        assert round(fw - bw) == round(ip / s.id * w / (2 * math.pi))  # 200 Hz, then 100 Hz
    # K_shaft on the defaults, in kN/m, kN/rad and kN·m/rad; a rigid shift takes no force.
    assert [round(ks[i, j] / 1e3, 3) for i, j in ((0, 0), (1, 1), (0, 2), (2, 2))] == [18.85, 37.699, 4.712, 2.356]
    assert round(ks[1, 1]) == round(48 * ei / L**3) == 37699
    assert round(ks[1, 1] / 1e3, 2) == quoted("rotor_notes", "37.70")  # on x_D, in kN/m
    # the ramp's gyroscopic term I_p Ω̇ θ left out, against the disk's tilt stiffness, at 5000 rpm/s
    assert round(100 * s.ip * 5000 * rpm / k_theta, 2) == quoted("rotor_notes", "0.03 %")
    assert np.allclose(ks @ [1, 1, 0, 1], 0.0, atol=1e-9)
    # The beam elements on the defaults with shaft mass.
    sm = s.with_(density=STEEL)
    le = L / sm.elements
    assert (sm.elements, 1e3 * le) == (8, quoted("rotor_notes", "62.5"))
    assert round(12 * ei / le**3 / 1e6, 2) == quoted("rotor_notes", "4.83")
    rho_a, rho_i = STEEL * area, STEEL * inertia
    assert round(rho_a, 4) == quoted("rotor_notes", "0.6165")
    mt, mr = beam_mass(rho_a, rho_i, le)
    shift = np.array([1.0, 0.0, 1.0, 0.0])
    assert round(shift @ mt @ shift, 4) == quoted("rotor_notes", "0.0385")
    assert round(8 * shift @ mt @ shift, 3) == quoted("rotor_notes", "0.308")
    assert np.allclose(beam_stiffness(ei, le) @ shift, 0.0)
    assert round(100 * mr[0, 0] / mt[0, 0], 2) == quoted("rotor_notes", "0.52 %")
    assert round(100 * mr[1, 1] / mt[1, 1], 1) == quoted("rotor_notes", "2.2 %")
    assert round(2 * rho_i * L * 1e6, 2) == quoted("rotor_notes", "3.85")
    assert round(100 * 2 * rho_i * L / s.ip, 1) == quoted("rotor_notes", "0.3 %")
    kappa = 6 * (1 + POISSON) / (7 + 6 * POISSON)
    assert round(kappa, 3) == quoted("rotor_notes", "0.886")

    def phi(length: float) -> float:
        return 1.5 * (1 + POISSON) / kappa * (d / length) ** 2

    shear = s.youngs / (2 * (1 + POISSON))
    assert phi(le) == pytest.approx(12 * ei / (kappa * shear * area * le**2))
    assert round(phi(le), 3) == quoted("rotor_notes", "0.056") and round(phi(L), 5) == quoted("rotor_notes", "0.00088")
    assert 1 - 1 / math.sqrt(1 + phi(L)) < quoted("rotor_notes", "0.05 %") / 100  # shear: the critical's change
    assert (2 * 2 * (sm.elements + 1), sm.n_dof, sm.n_states) == (36, 12, 24)
    # The eigenproblem at 3000 rpm: the first pair, the conical mode, and the spin reversed.
    w = 3000 * rpm
    lam = np.linalg.eigvals(s.state_matrix(w))
    first = lam[np.argmin(np.abs(lam))]
    assert (round(first.real, 3), round(abs(first.imag), 1)) == (
        -quoted("rotor_notes", "7.714"),
        quoted("rotor_notes", "165.9"),
    )
    assert round(abs(first.imag) / (2 * math.pi), 2) == quoted("rotor_notes", "26.40")
    assert round(-first.real / abs(first), 4) == quoted("rotor_notes", "0.0464")
    assert s.n_states == 16
    for sign, back in ((1, 271.5), (-1, 368.0)):
        m = whirl_modes(s, sign * w)
        conical = {round(f, 1): wr > 0 for f, wr in zip(m.freq_hz, m.whirl, strict=True) if f > 200}
        assert conical == {quoted("rotor_notes", "271.5"): sign < 0, quoted("rotor_notes", "368.0"): sign > 0}, back
    # Left eigenvectors at Ω are the right ones at −Ω (M, C, K symmetric; G, H skew).
    M, C, G, K = s.matrices()
    H = s.circulatory()
    lm, vm = np.linalg.eig(s.state_matrix(-w))
    for j in range(lm.size):
        chi = vm[: s.n_dof, j]
        D = lm[j] ** 2 * M + lm[j] * (C + w * G) + K + w * H
        assert np.linalg.norm(chi @ D) <= 1e-9 * np.linalg.norm(D) * np.linalg.norm(chi)


def test_rotor_worked_example():
    """Jeffcott rotor's worked example on the defaults, and the squeeze-film and rolling-bearing figures."""
    from vib_tutorial.core.rotor import XD, RotorSystem
    from vib_tutorial.core.rotor_modes import campbell, synchronous_response, whirl_modes

    s = RotorSystem()
    rpm = 60 / (2 * math.pi)
    ei = s.youngs * math.pi * s.diameter**4 / 64
    ks = 48 * ei / s.length**3
    assert round(ei, 2) == quoted("rotor_notes", "98.17") and round(ks) == 37699 == round(s.disc_stiffness)
    kb = s.bearing_a.kx
    keq = 1 / (1 / ks + 1 / (2 * kb))
    wc = math.sqrt(keq / s.disc_mass)
    assert (kb, round(keq), round(wc, 2), round(wc * rpm)) == (50000.0, 27378, quoted("rotor_notes", "165.46"), 1580)
    assert round(wc, 1) == quoted("rotor_notes", "165.5")
    assert s.bearing_a.mass == s.bearing_b.mass == quoted("rotor_notes", "0.1")  # the journals' 0.1 kg
    crit = campbell(s, 3000 / rpm, n_speeds=121, n_modes=4).criticals
    assert {round(w * rpm) for w, _, _ in crit} == {1584}
    m = whirl_modes(s, 0.0)
    assert [round(f, 2) for f in m.freq_hz[:2]] == [
        quoted("rotor_notes", "26.40"),
        quoted("rotor_notes", "26.40"),
    ]
    assert round(m.zeta[0], 4) == quoted("rotor_notes", "0.0464")
    zeta = 0.0464
    assert round(2 * math.pi * zeta / math.sqrt(1 - zeta**2), 3) == quoted("rotor_notes", "0.292")
    e = 20.0  # µm
    assert s.unbalance == pytest.approx(e * 1e-6)
    assert round(s.disc_mass * s.unbalance * (3000 / rpm) ** 2, 2) == quoted("rotor_notes", "1.97")
    for speed, r, amp, lag, page_amp, page_lag in (
        (
            3000,
            quoted("rotor_notes", "1.894"),
            quoted("rotor_notes", "27.7"),
            quoted("rotor_notes", "176.1"),
            quoted("rotor_notes", "28.0"),
            quoted("rotor_notes", "175.9"),
        ),
        (
            500,
            quoted("rotor_notes", "0.3157"),
            quoted("rotor_notes", "2.21"),
            quoted("rotor_notes", "1.9"),
            2.22,
            quoted("rotor_notes", "1.8"),
        ),
    ):
        assert round(speed / 1584, 4 if r < 1 else 3) == r
        assert round(e * r * r / math.sqrt((1 - r * r) ** 2 + (2 * zeta * r) ** 2), 2 if amp < 10 else 1) == amp
        assert round(math.degrees(math.atan2(2 * zeta * r, 1 - r * r)), 1) == lag
        X = synchronous_response(s, np.array([speed / rpm]))[0, XD]
        assert round(abs(X) * 1e6, 2 if page_amp < 10 else 1) == page_amp
        if speed == 500:
            assert round(abs(X) * 1e6, 1) == quoted("rotor_notes", "2.2")  # "the disk whirls 2.2 µm at 500 rpm"
        assert round(-math.degrees(np.angle(X)), 1) == page_lag
    assert round(e / (2 * zeta)) == 216
    X = synchronous_response(s, np.array([1584 / rpm]))[0, XD]
    assert round(abs(X) * 1e6) == 217 and round(-math.degrees(np.angle(X))) == 88
    assert abs(216 / 217 - 1) < quoted("rotor_notes", "1 %") / 100  # e/2ζ, within 1 % of the peak
    first = np.linalg.eigvals(s.state_matrix(3000 / rpm))
    first = first[np.argmin(np.abs(first))]
    assert (round(first.real, 3), round(abs(first.imag), 1)) == (
        -quoted("rotor_notes", "7.714"),
        quoted("rotor_notes", "165.9"),
    )
    # A squeeze-film damper: πηRL³/(2c³(1 − ε²)^1.5).
    eta, radius, length, clearance = 0.02, 0.05, 0.02, 0.2e-3

    def squeeze(eps: float) -> float:
        return math.pi * eta * radius * length**3 / (2 * clearance**3 * (1 - eps**2) ** 1.5)

    assert (round(squeeze(0.0)), round(squeeze(quoted("rotor_notes", "0.5")))) == (1571, 2418)
    assert round(squeeze(0.0) / 100) == 16
    # Rolling bearings: stiffness ∝ load^(1/3) for balls, ^0.1 for rollers.
    assert round(100 * (2 ** (1 / 3) - 1)) == quoted("rotor_notes", "26 %")
    assert round(100 * (2**0.1 - 1)) == quoted("rotor_notes", "7 %")


def test_rotor_damping_and_measuring():
    """Jeffcott rotor's support damping trial, the run-up's ramp against 2ζ², and the overhung rotor's waterfall."""
    from vib_tutorial.core.rotor import Bearing, RotorSystem
    from vib_tutorial.core.rotor_modes import whirl_modes
    from vib_tutorial.core.rotor_presets import rotor_presets
    from vib_tutorial.core.rotor_simulator import RPM

    def first_zeta(c: float) -> float:
        b = Bearing(cx=c, cy=c)
        return float(whirl_modes(RotorSystem(bearing_a=b, bearing_b=b), 0.0).zeta[0])

    zeta = first_zeta(100.0)
    assert round(zeta, 3) == quoted("rotor_notes", "0.046")
    assert round(first_zeta(400.0), 3) == quoted("rotor_notes", "0.089")
    assert round(first_zeta(10_000.0), 3) == quoted("rotor_notes", "0.005")  # the journals locked
    assert round(2 * zeta**2, 4) == quoted("rotor_notes", "0.0043")
    wc = 2 * math.pi * whirl_modes(RotorSystem(), 0.0).freq_hz[0]
    for rate, ratio, digits in (
        (100, quoted("rotor_notes", "0.0004"), 4),
        (500, quoted("rotor_notes", "0.0019"), 4),
        (5000, quoted("rotor_notes", "0.019"), 3),
    ):
        assert round((rate / RPM) / wc**2, digits) == ratio  # Ω̇/ω_c²
    # The overhung rotor's first mode from rest to 2,500 rpm, as the taps trace it.
    overhung = next(p for p in rotor_presets() if p.name == "Overhung rotor").system
    back, forward = whirl_modes(overhung, 2500.0 / RPM).freq_hz[:2]
    assert (round(back, 1), round(forward, 1)) == (quoted("rotor_notes", "5.6"), quoted("rotor_notes", "25.5"))
    assert round(200 * 0.5 / 60, 1) == quoted("rotor_notes", "1.7")  # Ω̇T_b: 100 rpm over a 0.5 s block


def test_rotor_builder_worked_example():
    """Rotor builder's worked example (the opening shaft) and its mesh convergence."""
    import dataclasses

    from vib_tutorial.core import rotor_builder as rb
    from vib_tutorial.gui.rotor_builder import TRIALS

    rotor = TRIALS[0].rotor
    s = rotor.sections[0]
    mat = s.material
    assert (mat.E, mat.nu, mat.rho, s.outer, s.inner, len(rotor.sections), s.elements) == (
        210e9,
        quoted("rotor_builder", "0.3"),
        7850.0,
        0.05,
        0.0,
        4,
        3,
    )
    asm = rb.assemble(rotor)
    assert rotor.nodes.size == 13 and asm.M.shape == (52, 52)
    assert round(s.area * 1e6, 1) == quoted("rotor_builder", "1963.5") and round(s.inertia * 1e7, 3) == quoted(
        "rotor_builder", "3.068"
    )
    ei, rho_a, length = mat.E * s.inertia, mat.rho * s.area, 1.0
    assert round(ei / 1e3, 2) == quoted("rotor_builder", "64.43") and round(rho_a, 2) == quoted(
        "rotor_builder", "15.41"
    )
    hz = rb.modes(asm, 0.0).hz
    from vib_tutorial.core.rotor_builder_sweep import CRITICALLY_DAMPED
    from vib_tutorial.gui.rotor_builder import API_LOG_DEC

    assert API_LOG_DEC == quoted("rotor_builder", "0.1") and 100 * CRITICALLY_DAMPED == quoted(
        "rotor_builder", "20 %"
    )  # API 684's
    assert 1 / (2 * CRITICALLY_DAMPED) == quoted("rotor_builder", "2.5")  # the amplification factor there
    assert [round(f, 2) for f in hz[:6]] == [101.25, 101.25, 401.5, 401.5, 891.33, 891.33]
    eb = rb.modes(rb.assemble(dataclasses.replace(rotor, theory=rb.Theory.RAYLEIGH)), 0.0).hz
    assert [round(eb[0], 2), round(eb[4], 1)] == [101.48, 908.0]
    euler = [n * n * math.pi / 2 * math.sqrt(ei / (rho_a * length**4)) for n in (1, 2, 3)]
    assert [round(euler[0], 2), round(euler[1], 1), round(euler[2], 1)] == [101.56, 406.2, 914.0]
    rg = s.outer / 4
    assert rg == pytest.approx(math.sqrt(s.inertia / s.area)) and rg == 0.0125
    rot = [0.5 * (n * math.pi * rg / length) ** 2 for n in (1, 3)]
    assert [round(100 * rot[0], 3), round(100 * rot[1], 2)] == [
        quoted("rotor_builder", "0.077 %"),
        quoted("rotor_builder", "0.69 %"),
    ]
    assert [round(euler[0] * (1 - rot[0]), 2), round(euler[2] * (1 - rot[1]), 1)] == [
        101.48,
        quoted("rotor_builder", "907.7"),
    ]
    assert round(s.kappa, 3) == quoted("rotor_builder", "0.886") and round(mat.E / mat.G, 1) == quoted(
        "rotor_builder", "2.6"
    )
    factor = 1 + mat.E / (s.kappa * mat.G)
    assert round(factor, 3) == quoted("rotor_builder", "3.933")
    assert [round(100 * rot[0] * factor, 2), round(100 * rot[1] * factor, 1)] == [
        quoted("rotor_builder", "0.30 %"),
        quoted("rotor_builder", "2.7 %"),
    ]
    assert [round(euler[0] * (1 - rot[0] * factor), 2), round(euler[2] * (1 - rot[1] * factor), 1)] == [
        101.25,
        quoted("rotor_builder", "889.1"),
    ]

    def phi(span: float) -> float:
        return 12 * ei / (s.kappa * mat.G * s.area * span**2)

    assert round(length / 12 * 1e3, 1) == quoted("rotor_builder", "83.3") and round(phi(length / 12), 3) == quoted(
        "rotor_builder", "0.792"
    )
    assert round(phi(length), 4) == quoted("rotor_builder", "0.0055") and phi(length / 3) == pytest.approx(
        9 * phi(length)
    )

    # Timoshenko's exact pinned frequencies (the 2 × 2 determinant), and the mesh converging on them.
    def exact(n: int) -> float:
        k, kg = n * math.pi / length, s.kappa * mat.G * s.area
        a, rho_i = rho_a * mat.rho * s.inertia, mat.rho * s.inertia
        b = -(rho_a * (ei * k * k + kg) + rho_i * kg * k * k)
        c = kg * k * k * ei * k * k
        return math.sqrt((-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)) / (2 * math.pi)

    assert (round(exact(1), 2), round(exact(3), 1)) == (101.25, quoted("rotor_builder", "890.3"))
    thirds = []
    for n in (1, 2, 3, 6):
        mesh = rb.pinned_ends(rb.Rotor(rb.uniform(1.0, 0.05, 4, n)))
        f = rb.modes(rb.assemble(mesh), 0.0).hz
        thirds.append(round(f[4], 1))
        assert f[4] > exact(3) and f[0] > exact(1) - 1e-6
        if n == 1:
            assert abs(f[0] - exact(1)) < 0.045
    assert thirds == [
        quoted("rotor_builder", "912.7"),
        quoted("rotor_builder", "893.2"),
        891.3,
        quoted("rotor_builder", "890.5"),
    ]
    assert round(100 * (hz[4] / exact(3) - 1), 2) == quoted("rotor_builder", "0.12 %")


def test_diagnostics_worked_example():
    """Machinery diagnostics: the healthy gearbox, the bearing's frequencies, the sidebands, severity and TSA."""
    from vib_tutorial.core import diagnostics as dg

    m = dg.Machine()
    assert (m.speed, m.teeth, round(m.mesh_hz, 2)) == (quoted("diagnostics", "29.95"), 27, 808.65)
    assert (round(2 * m.mesh_hz, 1), round(3 * m.mesh_hz, 2)) == (
        quoted("diagnostics", "1617.3"),
        quoted("diagnostics", "2425.95"),
    )
    assert (dg.FS, dg.DURATION, dg.FS / dg.BLOCK) == (20480.0, 4.0, quoted("diagnostics", "2.5"))
    rec = dg.record(m)
    assert dg.SHAFT == (quoted("diagnostics", "0.3"), quoted("diagnostics", "0.1")) and dg.MESH_HARMONICS == (
        1.0,
        quoted("diagnostics", "0.4"),
        0.15,
    )  # the model's
    assert m.noise == quoted("diagnostics", "0.2") and m.zeta == quoted("diagnostics", "0.05")
    sp = dg.amplitude_spectrum(rec.x, rec.fs)
    read = [round(dg.line_amplitude(sp.freqs, sp.amplitude, f), 3) for f in (m.speed, 2 * m.speed)]
    read += [round(dg.line_amplitude(sp.freqs, sp.amplitude, h * m.mesh_hz), 3) for h in (1, 2, 3)]
    assert read == [
        quoted("diagnostics", "0.301"),
        quoted("diagnostics", "0.102"),
        quoted("diagnostics", "1.001"),
        quoted("diagnostics", "0.399"),
        quoted("diagnostics", "0.149"),
    ]
    assert round(dg.kurtosis(rec.x), 2) == quoted("diagnostics", "2.4")
    # The 6205's frequencies, from rolling without slip.
    b = m.bearing
    rb = b.ball / b.pitch
    assert (b.balls, b.contact_deg, round(rb, 4)) == (9, 0.0, quoted("diagnostics", "0.2034"))
    assert (b.ball * 1e3, b.pitch * 1e3) == pytest.approx(
        (quoted("diagnostics", "7.94"), quoted("diagnostics", "39.04"))
    )
    f = b.frequencies(m.speed)
    assert (round(f["FTF"], 2), round(f["BPFO"], 1), round(f["BPFI"], 1), round(2 * f["BSF"], 1)) == (
        quoted("diagnostics", "11.93"),
        107.4,
        162.2,
        141.2,
    )
    assert (round(f["BPFO"] / m.speed, 3), round(f["BPFI"] / m.speed, 3)) == (3.585, quoted("diagnostics", "5.415"))
    assert f["BPFO"] + f["BPFI"] == pytest.approx(b.balls * m.speed)
    # An eccentric pinion: sidebands (s ± β)/2 beside the unit mesh line.
    s, beta = m.gear_severity, 0.5 * m.gear_severity
    assert (s, beta) == (quoted("diagnostics", "0.3"), 0.15) and (
        round((s + beta) / 2, 3),
        round((s - beta) / 2, 3),
    ) == (quoted("diagnostics", "0.225"), quoted("diagnostics", "0.075"))
    e = dg.Machine(gear=dg.GearFault.ECCENTRIC)
    r2 = dg.record(e)
    s2 = dg.amplitude_spectrum(r2.x, r2.fs)
    lines = [round(dg.line_amplitude(s2.freqs, s2.amplitude, e.mesh_hz + k * e.speed), 3) for k in (-1, 0, 1)]
    assert lines == [quoted("diagnostics", "0.076"), quoted("diagnostics", "0.995"), quoted("diagnostics", "0.226")]
    # The envelope's band: the resonance's half-power band, and about 3ζ either side.
    assert (round(2 * m.zeta * m.resonance), round(3 * m.zeta * m.resonance)) == (300, 450)
    # Severity: each line's RMS velocity, the noise's over 10–1000 Hz, and the total.
    rms = [a / (2 * math.pi * fq) / math.sqrt(2) * 1e3 for a, fq in ((0.3, m.speed), (0.1, 2 * m.speed))]
    rms += [a / (2 * math.pi * h * m.mesh_hz) / math.sqrt(2) * 1e3 for h, a in zip((1, 2, 3), dg.MESH_HARMONICS)]
    assert round(0.3 / (2 * math.pi * m.speed) * 1e3, 3) == quoted("diagnostics", "1.594")
    assert [round(v, 3) for v in rms] == [
        quoted("diagnostics", "1.127"),
        quoted("diagnostics", "0.188"),
        quoted("diagnostics", "0.139"),
        quoted("diagnostics", "0.028"),
        quoted("diagnostics", "0.007"),
    ]
    noise = math.sqrt(m.noise**2 / (dg.FS / 2) / (4 * math.pi**2) * (1 / 10 - 1 / 1000)) * 1e3
    assert round(noise, 3) == quoted("diagnostics", "0.099")
    assert round(math.sqrt(sum(v * v for v in rms) + noise**2), 2) == quoted("diagnostics", "1.16")
    assert 1.16 < quoted("diagnostics", "1.4")  # zone A (ISO 20816-3's A/B boundary on a rigid support)
    assert round(0.25 * quoted("diagnostics", "2.8"), 1) == quoted("diagnostics", "0.7")  # a significant change
    t = np.linspace(0, 1, 100_000, endpoint=False)
    assert dg.kurtosis(np.sin(2 * math.pi * 50 * t)) == pytest.approx(
        quoted("diagnostics", "1.5"), abs=1e-3
    )  # a sine's
    # Time-synchronous averaging over the record's whole revolutions.
    revs = int(dg.DURATION * m.speed)
    assert (revs, round(math.sqrt(revs), 1), round(m.noise / math.sqrt(revs), 3)) == (
        119,
        quoted("diagnostics", "10.9"),
        quoted("diagnostics", "0.018"),
    )
    # The run-up: Q, the half-power band, and how many of the mode's cycles the 1X spends crossing it.
    run = dg.RunUp()
    assert (run.mode, run.zeta, round(1 / (2 * run.zeta), 1), round(2 * run.zeta * run.mode, 1)) == (
        40.0,
        quoted("diagnostics", "0.03"),
        quoted("diagnostics", "16.7"),
        quoted("diagnostics", "2.4"),
    )
    for seconds, crossing, cycles in ((20.0, quoted("diagnostics", "0.8"), 32.0), (4.0, 0.16, 6.4)):
        rate = (run.end - run.start) / seconds
        assert (
            round(quoted("diagnostics", "2.4") / rate, 2),
            round(quoted("diagnostics", "2.4") / rate * run.mode, 1),
        ) == (crossing, cycles)
    assert (dg.PER_REV // 2, dg.REVS) == (16, 8)


def test_rigid_body_notes():
    """The 3D rigid body notes' numbers: the default box's moments, the flip's growth rate and time, and the
    wobble about the greatest axis; and the trials' (a 0.02 N·m push for 1 s)."""
    from vib_tutorial.core.rigid_body import Body, Torque, Tumbler
    from vib_tutorial.gui import rigid_body as page

    box = Body()
    I1, I2, I3 = box.principal()[0]
    assert [round(v * 1e3, 3 if v < 0.01 else 2) for v in (I1, I2, I3)] == [
        quoted("rigid_body", "4.167"),
        quoted("rigid_body", "8.333"),
        quoted("rigid_body", "10.83"),
    ]
    lam = 6 * math.sqrt((I2 - I1) * (I3 - I2) / (I1 * I3))
    assert round(lam, 2) == quoted("rigid_body", "2.88")
    assert round(math.radians(1.0), 5) == quoted("rigid_body", "0.01745")
    assert round(1 / math.radians(1.0), 1) == quoted("rigid_body", "57.3")
    assert round(math.log(57.3) / 2.88, 1) == quoted("rigid_body", "1.4")
    assert round(6 * math.sqrt((I3 - I1) * (I3 - I2) / (I1 * I2)), 2) == quoted("rigid_body", "4.16")
    # The flip takes about that long: the middle axis first points the other way near 1.4 s plus a little.
    trial = page.TRIALS[0]
    tm = Tumbler(box, tuple(page.start_rate(box, trial.spin_axis, trial.rate, trial.nudge)))
    s = tm.advance(3.0)
    from vib_tutorial.core import rotation

    along = rotation.matrix(s.q)[:, 1, 1]
    assert 1.4 < s.t[np.argmax(along < 0)] < 2.2
    push = page.TRIALS[5]
    assert (push.held, push.torque[2], push.duration) == (Torque.SPACE, 0.02, 1.0)


def test_rigid_body_euler_rates_trial():
    """The trial's z passes 1.7° from Z, where φ̇ reaches about 67 rad/s, 33 times its 2 rad/s."""
    from vib_tutorial.core import rotation
    from vib_tutorial.core.rigid_body import Tumbler
    from vib_tutorial.gui import rigid_body as page

    t = next(t for t in page.TRIALS if t.name == "Euler's rates blow up at the lock")
    body = t.body
    q = rotation.quaternion(rotation.from_euler(*np.radians(t.pose)))
    s = Tumbler(body, tuple(page.start_rate(body, t.spin_axis, t.rate, t.nudge)), q=list(q)).advance(1.0)
    least, fastest = 180.0, 0.0
    for qq, L in zip(s.q, s.momentum):
        B = rotation.matrix(qq)
        _, theta, psi, _ = rotation.to_euler(B)
        w = np.linalg.solve(body.inertia(), B.T @ L)
        least = min(least, math.degrees(theta))
        fastest = max(fastest, abs(rotation.euler_rates(theta, psi, w)[0]))
    assert round(least, 1) == quoted("rigid_body", "1.7") and round(fastest) == 67 and round(fastest / t.rate) == 33


def test_rigid_body_worked_example():
    """The worked example: the start's ω, L, |L|, energy and their angles (the readout's), and the off-center mass's
    G, I_xx, I_xy (the Inertia tab's) and principal moments and axis."""
    from vib_tutorial.core.rigid_body import Body
    from vib_tutorial.gui import rigid_body as page

    box = Body()
    sides = (quoted("rigid_body", "0.3"), quoted("rigid_body", "0.2"), quoted("rigid_body", "0.1"))
    assert (box.mass, box.a, box.b, box.c) == (1.0, *sides)
    w = page.start_rate(box, 1, 6.0, 1.0)
    L = box.inertia() @ w
    assert [round(v, 4) for v in w] == [0.0, 5.9991, quoted("rigid_body", "0.1047")]
    assert round(w[1], 3) == quoted("rigid_body", "5.999")
    assert (round(L[1], 5), round(L[2], 6), round(float(np.linalg.norm(L)), 5)) == (
        quoted("rigid_body", "0.04999"),
        quoted("rigid_body", "0.001134"),
        quoted("rigid_body", "0.05001"),
    )
    assert round(0.5 * float(w @ L), 4) == quoted("rigid_body", "0.1500")
    Ix, Iy = quoted("rigid_body", "8.333"), quoted("rigid_body", "10.83")
    tilt = math.degrees(math.atan(Iy / Ix * math.tan(math.radians(1.0))))
    assert round(tilt, 2) == quoted("rigid_body", "1.30")
    apart = math.degrees(math.acos(float(w @ L) / float(np.linalg.norm(w) * np.linalg.norm(L))))
    assert round(apart, 2) == quoted("rigid_body", "0.30")
    mp, x, y = quoted("rigid_body", "0.3"), quoted("rigid_body", "0.1"), quoted("rigid_body", "0.08")
    off = Body(point=mp, offset=(x, y, 0.0))
    assert 1 + mp == quoted("rigid_body", "1.3")
    G = off.centre()
    gx, gy = quoted("rigid_body", "0.02308"), quoted("rigid_body", "0.01846")
    assert [round(v * 1000, 1) for v in G[:2]] == [
        quoted("rigid_body", "23.1"),
        quoted("rigid_body", "18.5"),
    ]
    assert round(mp / 1.3 * y, 5) == gy
    assert round(mp / 1.3 * x, 5) == gx
    I = off.inertia()
    Ixx = round(I[0, 0] * 1e3, 3)
    assert (
        round((quoted("rigid_body", "4.167") * 1e-3 + gy**2 + mp * (y - gy) ** 2) * 1e3, 3)
        == Ixx
        == quoted("rigid_body", "5.644")
    )
    Ixy = round(-I[0, 1] * 1e3, 3)
    assert round((gx * gy + mp * (x - gx) * (y - gy)) * 1e3, 3) == Ixy == quoted("rigid_body", "1.846")
    assert round(mp / 1.3 * x * y, 6) == round(Ixy * 1e-3, 6)
    assert round(Ixy * 1e-3, 5) == quoted("rigid_body", "0.00185")
    moments, P = off.principal()
    assert [round(v * 1e3, 3 if v < 0.01 else 2) for v in moments] == [
        quoted("rigid_body", "5.036"),
        quoted("rigid_body", "11.25"),
        quoted("rigid_body", "14.62"),
    ]
    assert round(math.degrees(math.atan2(P[1, 0], P[0, 0])), 1) == quoted("rigid_body", "18.2")


def test_rigid_body_drift_numbers():
    """The notes' and the trial's drift numbers: the last trial at 1/20 of a turn for 10 s, and the rates trial's
    start, which passes the lock."""
    from vib_tutorial.core import rotation
    from vib_tutorial.core.rigid_body import compare
    from vib_tutorial.gui import rigid_body as page

    names = [t.name for t in page.TRIALS]
    ends = {}
    for name in ("Step it three ways", "Euler's rates blow up at the lock"):
        t = page.TRIALS[names.index(name)]
        w = page.start_rate(t.body, t.spin_axis, t.rate, t.nudge)
        q0 = rotation.quaternion(rotation.from_euler(*np.radians(t.pose)))
        ends[name] = compare(t.body, w, q0, 10.0, 2 * math.pi / np.linalg.norm(w) / 20)
    d = ends["Step it three ways"]
    assert (round(d.quaternion[-1], 3), round(d.matrix[-1], 2), round(d.orthogonality[-1], 4)) == (
        quoted("rigid_body", "0.047"),
        quoted("rigid_body", "0.29"),
        quoted("rigid_body", "0.0038"),
    )
    assert round(d.euler[-1]) == 13 and d.euler[-1] > d.matrix[-1] > d.quaternion[-1]
    d = ends["Euler's rates blow up at the lock"]
    assert round(d.quaternion[-1], 3) == quoted("rigid_body", "0.006")
    # past the lock the Euler angles amplify rounding: 156–177° across the release builds' machines
    assert d.euler[-1] > 150  # the notes: "more than 150°"


if __name__ == "__main__":
    UNCHECKED.write_text(listing(unchecked_numbers()), encoding="utf-8")
    print(f"wrote {UNCHECKED}", file=sys.stderr)


def test_bases_worked_example():
    """The figures Coordinates & bases' worked example quotes, on the page's defaults, and from the rounded figures
    they show."""
    import math

    from vib_tutorial.core.bases import TWO_PI, TwoMass, direction, rayleigh_hz
    from vib_tutorial.view_models.bases import BasesTrial, system

    s = system(BasesTrial("", ""))
    assert s == TwoMass(1.0, 1.0, 400.0, 100.0, 400.0, 0.0)
    M, _, K = s.matrices()
    assert K.tolist() == [[500.0, -100.0], [-100.0, 500.0]]
    hz, _ = s.modes()
    f1, f2 = quoted("bases", "3.183"), quoted("bases", "3.898")
    assert hz == pytest.approx([f1, f2], abs=5e-4)
    assert math.sqrt(400.0) / TWO_PI == pytest.approx(f1, abs=5e-4)
    beat = quoted("bases", "1.398")
    assert 1 / (f2 - f1) == pytest.approx(beat, abs=2e-3) and 1 / (hz[1] - hz[0]) == pytest.approx(beat, abs=5e-4)
    assert beat / 2 == pytest.approx(quoted("bases", "0.699"), abs=5e-4)
    assert rayleigh_hz(direction(0.0), M, K) == pytest.approx(quoted("bases", "3.559"), abs=5e-4)
    assert rayleigh_hz(direction(45.0), M, K) == pytest.approx(f1, abs=5e-4)
    _, U = TwoMass(m2=2.0).modes()
    apart = math.degrees(math.acos(abs(float(U[:, 0] @ U[:, 1]))))
    assert apart == pytest.approx(quoted("bases", "80.1"), abs=0.05)
    # One arrow: 11.8 % high along x₁; v = (1, 0.8), 38.7°, 6.3° off mode 1, only 0.3 % high.
    assert quoted("bases", "3.559") / f1 - 1 == pytest.approx(quoted("bases", "11.8 %") / 100, abs=5e-4)
    v = np.array([1.0, quoted("bases", "0.8")])
    kv, mv = float(v @ K @ v), float(v @ M @ v)
    assert (kv, mv) == pytest.approx((660.0, quoted("bases", "1.64")))  # 500 − 160 + 320, as the notes add it
    assert 500 - 2 * 100 * 0.8 + 500 * 0.8**2 == pytest.approx(kv)
    assert kv / mv == pytest.approx(quoted("bases", "402.4"), abs=0.05)
    assert math.sqrt(402.4) / TWO_PI == pytest.approx(quoted("bases", "3.193"), abs=5e-4)
    assert rayleigh_hz(v, M, K) == pytest.approx(3.193, abs=5e-4)
    assert math.degrees(math.atan(0.8)) == pytest.approx(quoted("bases", "38.7"), abs=0.05)
    assert 45 - 38.7 == pytest.approx(quoted("bases", "6.3"), abs=1e-9)
    assert 3.193 / 3.183 - 1 == pytest.approx(quoted("bases", "0.3 %") / 100, abs=5e-4)


def test_bases_trials():
    """The numbers Coordinates & bases' Things to try quote."""
    import math

    from vib_tutorial.core.bases import TwoMass, angle, direction, rayleigh_hz, transformed
    from vib_tutorial.gui.bases_trials import TRIALS
    from vib_tutorial.view_models.bases import basis, system

    def note(name: str) -> str:
        return next(t for t in TRIALS if t.name.startswith(name)).note

    def says(name: str, *numbers: str) -> None:
        for n in numbers:
            assert n in note(name), (name, n)

    says("Pluck one mass", "0.699 s", "1.398 s")
    weak = TwoMass(k2=10.0).modes()[0]
    assert f"{weak[1]:.3f}" == "3.262" and f"{1 / (weak[1] - weak[0]):.1f}" == "12.7"
    t = next(t for t in TRIALS if t.name.startswith("The arrows along"))
    M, _, K = system(t).matrices()
    V = basis(t)
    assert transformed(V, M) == pytest.approx(np.eye(2), abs=1e-6)
    assert transformed(V, K) == pytest.approx(np.diag([400.0, 600.0]), abs=1e-4)
    says("The arrows along", "3.183 Hz", "3.898 Hz", "400 and 600 N/m")
    for name in ("Unequal masses", "A stiffer coupling"):
        s = system(next(t for t in TRIALS if t.name.startswith(name)))
        f, P = s.modes()
        says(name, f"{f[0]:.3f}", f"{f[1]:.3f}", f"{angle(P[:, 0]):.1f}°")
    _, P = TwoMass(m2=2.0).modes()
    says("Unequal masses", f"{angle(P[:, 1]):.1f}°", f"{math.degrees(math.acos(abs(float(P[:, 0] @ P[:, 1])))):.1f}°")
    t = next(t for t in TRIALS if t.name.startswith("Coupled through the masses"))
    M, _, K = system(t).matrices()
    V = basis(t)
    assert transformed(V, K) == pytest.approx(np.diag([400.0, 600.0]), abs=1e-4)
    assert transformed(V, M)[0, 1] == pytest.approx(-0.5, abs=1e-6)
    says("Coupled through the masses", "69.6°", "169.5°", "−0.5 kg")
    s = TwoMass()
    M, _, K = s.matrices()
    assert rayleigh_hz(direction(0.0), M, K) == pytest.approx(3.559, abs=5e-4)
    says("One arrow along x₁", "500 N/m", "3.559 Hz")
    says("One arrow along x₁", "11.8 %", "3.183 Hz", "0.3 % at 38.7°")
    assert rayleigh_hz(direction(38.7), M, K) / rayleigh_hz(direction(45.0), M, K) - 1 == pytest.approx(0.003, abs=5e-4)


def test_state_space_worked_example():
    """The figures State-space modes' notes quote, on the page's defaults, and from the rounded figures they show."""
    import math

    from vib_tutorial.core.bases import TWO_PI, Steps, TwoMass, state_matrix
    from vib_tutorial.core.state_modes import in_basis, own_frequency, state_modes, undamped_basis, undamped_modes
    from vib_tutorial.view_models import state_space as v

    s = v.system(v.StateTrial("", ""))
    assert s == TwoMass(1.0, 2.0, 400.0, 100.0, 400.0, 1.0)
    M, C, K = s.matrices()
    A = state_matrix(M, C, K)
    assert A[2:, :2].tolist() == [[-500.0, 100.0], [50.0, -250.0]]
    assert A[2:, 2:].tolist() == [[-2.0, 1.0], [quoted("state_space", "0.5"), -1.0]]
    assert own_frequency(M, K, 0) == pytest.approx(quoted("state_space", "22.4"), abs=0.05)
    assert own_frequency(M, K, 1) == pytest.approx(quoted("state_space", "15.8"), abs=0.05)
    omega, V = undamped_modes(M, K)
    assert omega / TWO_PI == pytest.approx([quoted("state_space", "2.421"), quoted("state_space", "3.624")], abs=5e-4)
    assert (V.T @ C @ V)[0, 1] == pytest.approx(-quoted("state_space", "0.51"), abs=5e-3)
    Ab = in_basis(undamped_basis(M, K), A)
    assert Ab[1, 3] == pytest.approx(quoted("state_space", "0.41"), abs=5e-3)
    assert Ab[3, 1] == pytest.approx(quoted("state_space", "0.333"), abs=5e-4)
    one, two = state_modes(M, C, K)
    assert one.sigma == pytest.approx(-quoted("state_space", "0.358"), abs=5e-4)
    assert one.omega == pytest.approx(quoted("state_space", "15.21"), abs=5e-3)
    assert two.sigma == pytest.approx(-quoted("state_space", "1.14"), abs=5e-3)
    assert two.omega == pytest.approx(quoted("state_space", "22.74"), abs=5e-3)
    assert two.omega / TWO_PI == pytest.approx(quoted("state_space", "3.619"), abs=5e-4)
    assert 100 * one.zeta == pytest.approx(quoted("state_space", "2.35 %"), abs=5e-3)
    assert 100 * two.zeta == pytest.approx(quoted("state_space", "5.02 %"), abs=5e-3)
    assert -one.phase == pytest.approx(quoted("state_space", "4.51"), abs=5e-3)
    assert -two.phase == pytest.approx(quoted("state_space", "173.29"), abs=5e-3)
    assert 180 - 173.29 == pytest.approx(quoted("state_space", "6.71"), abs=1e-9)
    assert two.sigma == pytest.approx(-quoted("state_space", "1.142"), abs=5e-4)
    assert math.log(2) / 1.142 == pytest.approx(quoted("state_space", "0.607"), abs=5e-4)
    assert math.log(2) / 0.358 == pytest.approx(quoted("state_space", "1.94"), abs=5e-3)
    # The coupling: started 30 mm out in mode 2 of M and K, mode 1's pair climbs to 1.48 mm in 0.41 s.
    x, xd = v.start_state(s, 1)
    assert np.abs(x).max() * 1e3 == pytest.approx(30.0)
    ts, zs, _ = Steps(M, C, K, np.concatenate([x, xd])).advance(2.0)
    passed = v.sizes(s, zs[:, :2], zs[:, 2:]).undamped[0]
    i = int(np.argmax(passed))
    assert passed[i] == pytest.approx(quoted("state_space", "1.48"), abs=5e-3)
    assert ts[i] == pytest.approx(0.41, abs=5e-3)
    # The symmetric pair's dampers are proportional, C = −3M + 0.01K; the default's aren't, φ₂ᵀCφ₁ = −0.51.
    Ms, Cs, Ks = TwoMass(c=1.0).matrices()
    assert Cs == pytest.approx(-3 * Ms + quoted("state_space", "0.01") * Ks)
    assert not np.allclose(C @ np.linalg.solve(M, K), K @ np.linalg.solve(M, C))  # Caughey and O'Kelly's test fails
    # Mode 2 overdamped past 18.8 N·s/m on each damper (m₂ = 2 kg).
    assert all(m.swings for m in state_modes(*TwoMass(m2=2.0, c=18.8).matrices()))
    assert not state_modes(*TwoMass(m2=2.0, c=18.9).matrices())[1].swings
    assert quoted("state_space", "18.8") == 18.8


def test_state_space_trials():
    """The numbers State-space modes' Things to try quote."""
    from vib_tutorial.core.bases import TWO_PI, Steps
    from vib_tutorial.core.state_modes import state_modes
    from vib_tutorial.gui.state_space_trials import TRIALS
    from vib_tutorial.view_models import state_space as v

    def trial(name: str):
        return next(t for t in TRIALS if t.name.startswith(name))

    def says(name: str, *numbers: str) -> None:
        for n in numbers:
            assert n in trial(name).note, (name, n)

    def passed_peak(c: float) -> tuple[float, float]:
        s = v.system(v.StateTrial("", "", c=c))
        M, C, K = s.matrices()
        x, xd = v.start_state(s, 1)
        ts, zs, _ = Steps(M, C, K, np.concatenate([x, xd])).advance(2.0)
        passed = v.sizes(s, zs[:, :2], zs[:, 2:]).undamped[0]
        return float(passed.max()), float(ts[int(np.argmax(passed))])

    peak, at = passed_peak(1.0)
    says("The dampers couple the modes of M and K", f"{peak:.2f} mm", f"{at:.2f} s")
    assert (
        f"{passed_peak(2.0)[0]:.2f}" == "2.28"
        and "2.28 mm" in trial("The dampers couple the modes of M and K").predict.why[1]
    )
    s = v.system(trial("The same start"))
    x, xd = v.start_state(s, 1)
    sz = v.sizes(s, x[None, :], xd[None, :])
    says("The same start", f"{sz.state[0][0]:.2f} mm", "−0.358 /s", "−1.14 /s")
    one, two = state_modes(*v.system(trial("One state-space mode")).matrices())
    says("One state-space mode", f"{two.omega / TWO_PI:.3f} Hz", "−1.14 /s")
    one, _ = state_modes(*v.system(trial("Heavier dampers")).matrices())
    says("Heavier dampers", f"{-one.phase:.1f}°")
    one, two = state_modes(*v.system(trial("Mode 2 overdamped")).matrices())
    a, b = (e.real for e in two.eigenvalues)
    says("Mode 2 overdamped", f"−{-a:.1f}", f"−{-b:.1f} /s", f"{one.omega / TWO_PI:.3f} Hz", "18.8 N·s/m")
