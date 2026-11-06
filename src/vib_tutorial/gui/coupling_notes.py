"""Text for the Modal coupling page: the theory notes, the notation, and the live step-by-step matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import ChainSystem, ModalResult
from ..core.modal_coupling import CoupledModel
from ..core.presets import presets
from .cms_notes import _side_by_side, fmt, matrix_html
from .lesson import trials_html
from .links import load_notes
from .style import colors


@dataclass(frozen=True)
class CouplingTrial:
    """A chain for the Simulation page, and how this page splits and couples it."""

    name: str
    note: str
    chain: str  # "absorber", "uniform", "light" or "eight"
    split: int  # A is masses 1..split
    mode_a: int = 0  # 0-based
    mode_b: int = 0
    a_mass: str = "tip"
    tab: int = 0  # the right-hand tab: veering, shapes, FRF

    def system(self) -> ChainSystem:
        if self.chain == "absorber":
            return next(p.system for p in presets() if p.name.startswith("Vibration absorber")).copy()
        if self.chain == "eight":
            return ChainSystem.uniform(8)
        system = ChainSystem.uniform(4)
        if self.chain == "light":  # B a tenth of A, tuned to it, still C = K/200
            system.masses[2:] = 0.1
            system.stiffness[2:] = 40.0
            system.damping[2:] = 0.2
        return system


TRIALS = (
    CouplingTrial(
        "An absorber is already a 2-DOF model",
        "The Vibration absorber chain, split after m1: A is the machine, B the absorber, tuned to it with "
        "μ = 0.1. The 2-DOF model is the chain itself, so the errors are 0. The two modes sit "
        "√0.1 ≈ 32% of f<sub>A</sub> apart, one either side.",
        "absorber",
        split=1,
    ),
    CouplingTrial(
        "Two equal halves: pushed far apart",
        "The uniform chain split after m2. Both halves have the same mode 1, so f<sub>B</sub> = "
        "f<sub>A</sub>, but μ = 1.37 is large: the modes are pushed far apart. The upper one is 4.9% too "
        "high, with a MAC of 0.96, because the halves' second modes aren't far off and are left out.",
        "uniform",
        split=2,
    ),
    CouplingTrial(
        "A light half, tuned: splitting",
        "m3 and m4 at 0.1 kg on 40 N/m springs: B is tuned to A with μ = 0.14, and the 2-DOF model is "
        "within 0.2% and 0.3%, both above the chain (the Rayleigh–Ritz bound). The Veering tab sweeps B's "
        "frequency: the two curves veer apart instead of crossing. Switch it to the mass ratio to see the "
        "splitting shrink as B gets lighter.",
        "light",
        split=2,
    ),
    CouplingTrial(
        "The wrong mass for A",
        "The same light half, with A's oscillator given its effective mass instead of its modal mass at the "
        "interface. The errors grow to 2.9% and −2.2%: no longer Rayleigh–Ritz, so they can have either sign.",
        "light",
        split=2,
        a_mass="effective",
    ),
    CouplingTrial(
        "A mode left out (8 masses)",
        "Eight masses split after m6, A's mode 2 coupled with B's mode 1. The pair now stands for the "
        "chain's modes 2 and 3, but the lower is 14% too low with a MAC of 0.63: A's mode 1 moves the tip "
        "too and couples to B's just as much, but it is left out. The bound still holds by mode number: "
        "the 2-DOF mode 1 is above the chain's mode 1. The Mode shapes tab shows the mismatch.",
        "eight",
        split=6,
        mode_a=1,
        tab=1,
    ),
)

NOTATION_HTML = load_notes("coupling_notation")

THEORY_HTML = load_notes("coupling_notes", notation=NOTATION_HTML, trials=trials_html(TRIALS))


# ------------------------------------------------------------------ step by step (live numbers)


def _vec_table(
    headers: list[str],
    columns: list[np.ndarray],
    rows: list[str],
    total: list[str] | None = None,
    bold_row: int | None = None,
) -> str:
    """Term-by-term table: one row per DOF, one column per quantity, and an optional totals row."""
    scale = [float(np.abs(c).max()) if np.size(c) else 1.0 for c in columns]
    out = [
        "<table border='1' cellspacing='0' cellpadding='3'><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    ]
    for r, label in enumerate(rows):
        style = f" bgcolor='{colors.block_tints['b', 'q']}'" if r == bold_row else ""
        cells = "".join(f"<td align='right'{style}>{fmt(c[r], s)}</td>" for c, s in zip(columns, scale))
        out.append(f"<tr><th>{label}</th>{cells}</tr>")
    if total is not None:
        out.append("<tr><th>Σ</th>" + "".join(f"<td align='right'><b>{t}</b></td>" for t in total) + "</tr>")
    out.append("</table>")
    return "".join(out)


def _mat(A: np.ndarray, labels: list[str], group: str, title: str) -> str:
    groups = [group] * len(labels)
    return matrix_html(A, labels, labels, groups, groups, title)


def _g(v: float) -> str:
    return f"{v:.4g}"


def matrices_html(model: CoupledModel, full: ModalResult) -> str:
    """Every step from the chain to the 2-DOF model, with the current numbers."""
    system, j, n = model.system, model.split, model.system.n
    A, B = model.A, model.B
    ra, rb = model.mode_a, model.mode_b
    M, C, K = system.matrices()
    labels = [f"x{i + 1}" for i in range(n)]
    la, lb = labels[:j], labels[j:]
    tints = colors.block_tints
    parts = [
        "<p><i>Every number below follows the current parameters and choices. Symbols are defined in "
        "the <a href='#notation'>Notation</a> table at the end.</i></p>",
        f"<p><small>Cell colours: <span style='background:{tints['i', 'i']}'>&nbsp;A&nbsp;</span> "
        f"<span style='background:{tints['b', 'b']}'>&nbsp;B&nbsp;</span> "
        f"<span style='background:{tints['b', 'i']}'>&nbsp;A–B coupling&nbsp;</span> "
        f"<span style='background:{tints['q', 'q']}'>&nbsp;2-DOF model&nbsp;</span></small></p>",
    ]

    # 1. The chain and the split
    groups = ["i"] * j + ["b"] * (n - j)
    kj, cj = system.stiffness[j], system.damping[j]
    parts.append(f"<h3>1. The chain, split after m{j}</h3>")
    parts.append(
        f"<p>K [N/m], C [N·s/m] and M [kg] of the whole chain. A is {', '.join(la)}; B is {', '.join(lb)}. "
        f"They are joined by k<sub>{j + 1}</sub> = {_g(kj)} N/m and c<sub>{j + 1}</sub> = {_g(cj)} N·s/m, "
        "the off-diagonal (coupling) entries.</p>"
    )
    parts.append(
        _side_by_side(*(matrix_html(X, labels, labels, groups, groups, t) for X, t in ((K, "K"), (C, "C"), (M, "M"))))
    )
    parts.append(
        f"<p><b>A on its own</b> is the top-left block with k<sub>{j + 1}</sub> taken off its tip entry: "
        f"K<sub>A,{j}{j}</sub> = {_g(K[j - 1, j - 1])} − {_g(kj)} = {_g(A.K[-1, -1])}. "
        f"<b>B on its own</b>, with k<sub>{j + 1}</sub>, c<sub>{j + 1}</sub> tied to ground, is exactly the "
        f"bottom-right block: its first entry k<sub>{j + 1}</sub>"
        + (f" + k<sub>{j + 2}</sub>" if n - j > 1 else "")
        + " is unchanged, only the spring's other end is now the ground (B's <i>base</i>). "
        "M<sub>A</sub> and M<sub>B</sub> are the diagonals of each side's own masses.</p>"
    )
    parts.append(
        _side_by_side(
            _mat(A.K, la, "i", "K<sub>A</sub>"),
            _mat(A.M, la, "i", "M<sub>A</sub>"),
            _mat(B.K, lb, "b", "K<sub>B</sub>"),
            _mat(B.M, lb, "b", "M<sub>B</sub>"),
        )
    )

    # 2. Modes of each
    parts.append("<h3>2. The modes of each, on its own</h3>")
    parts.append(
        "<p>Kψ = ω²Mψ for each subsystem. The eigenproblem fixes each mode's <i>shape</i> "
        "but not its size: any multiple of ψ is the same mode.</p>"
    )
    for sub, r in ((A, ra), (B, rb)):
        freqs = ", ".join(
            (f"<b>{sub.name}{k + 1}: {_g(f)} Hz</b>" if k == r else f"{sub.name}{k + 1}: {_g(f)} Hz")
            for k, f in enumerate(sub.fn_hz)
        )
        parts.append(f"<p>{sub.name}: {freqs}. Coupled here: {sub.name}{r + 1} (ω = {_g(sub.omegas[r])} rad/s).</p>")

    # 3. Scaling: generalized mass and mass normalization
    phi_a, phi_b = A.Phi[:, ra], B.Phi[:, rb]
    ma_diag, mb_diag = np.diag(A.M), np.diag(B.M)
    tip = phi_a[-1]
    psi_a = phi_a / tip if abs(tip) > 1e-12 else phi_a / np.abs(phi_a).max()
    gm_a = float(psi_a @ A.M @ psi_a)
    peak = int(np.argmax(np.abs(phi_b)))
    psi_b = phi_b / phi_b[peak]
    gm_b = float(psi_b @ B.M @ psi_b)
    parts.append("<h3>3. Giving each mode a size: generalized mass and mass normalization</h3>")
    parts.append(
        "<p>Pick a scale, ψ, dimensionless. Its <b>generalized mass</b> ψ<sup>T</sup>Mψ = Σ m<sub>i</sub>ψ<sub>i</sub>² "
        "is in kg, and depends on the scale. The <b>mass-normalized</b> mode divides the shape by its square root, "
        "φ = ψ/√(ψ<sup>T</sup>Mψ), so φ<sup>T</sup>Mφ = 1 whatever ψ was, and φ is in <b>1/√kg</b>.</p>"
    )
    parts.append(f"<p><b>A{ra + 1}</b>, scaled to 1 at its tip x{j} (the interface):</p>")
    parts.append(
        _vec_table(
            [
                "",
                "m<sub>i</sub> [kg]",
                "ψ<sub>i</sub>",
                "m<sub>i</sub>ψ<sub>i</sub>² [kg]",
                "φ<sub>i</sub> = ψ<sub>i</sub>/√(ψ<sup>T</sup>Mψ) [1/√kg]",
            ],
            [ma_diag, psi_a, ma_diag * psi_a**2, phi_a],
            la,
            ["", "", f"ψ<sup>T</sup>Mψ = {_g(gm_a)}", ""],
            bold_row=j - 1,
        )
    )
    parts.append(f"<p><b>B{rb + 1}</b>, scaled to 1 at its largest entry, {lb[peak]}:</p>")
    parts.append(
        _vec_table(
            ["", "m<sub>i</sub> [kg]", "ψ<sub>i</sub>", "m<sub>i</sub>ψ<sub>i</sub>² [kg]", "φ<sub>i</sub> [1/√kg]"],
            [mb_diag, psi_b, mb_diag * psi_b**2, phi_b],
            lb,
            ["", "", f"ψ<sup>T</sup>Mψ = {_g(gm_b)}", ""],
        )
    )

    # 4. How the joint loads each subsystem
    parts.append("<h3>4. How the joint loads each subsystem</h3>")
    parts.append(
        "<p>A mode's mass depends on <i>where and how</i> it is pushed. Write the load on the subsystem as a "
        "vector times one number, and project the mode on that vector: that is the mode's "
        "<b>participation</b>. The two sides of this joint are loaded in different ways.</p>"
    )
    e_tip = np.zeros(j)
    e_tip[-1] = 1.0
    parts.append(
        f"<p><b>A is loaded at one point.</b> B pushes on A with the interface force F, on A's tip x{j} only. "
        f"The load is F·<b>e</b><sub>tip</sub>, with <b>e</b><sub>tip</sub> = [{', '.join('1' if v else '0' for v in e_tip)}]"
        "<sup>T</sup>: a <i>point load vector</i>, 1 at the loaded DOF and 0 elsewhere.</p>"
    )
    rhs = np.zeros(n - j)
    rhs[0] = kj
    if kj > 0:
        r_vec = np.linalg.solve(B.K, rhs)
        r_text = (
            f"K<sub>B</sub><b>r</b> = k<sub>{j + 1}</sub><b>e</b><sub>1</sub> = "
            f"[{', '.join(_g(v) for v in rhs)}]<sup>T</sup> gives <b>r</b> = "
            f"[{', '.join(_g(v) for v in r_vec)}]<sup>T</sup>"
        )
    else:
        r_vec = np.ones(n - j)
        r_text = (
            f"k<sub>{j + 1}</sub> = 0, so B floats and any base motion leaves it where it is; its rigid "
            "motion is still <b>r</b> = <b>1</b>"
        )
    parts.append(
        "<p><b>B is loaded by moving its base.</b> When the base moves by u, B's masses follow it by <b>r</b>u "
        "plus the mode's own motion. <b>r</b> is the <b>influence (direction) vector</b>: each DOF's static "
        "displacement for a unit base motion in the direction of excitation. The base pulls on B only through "
        f"k<sub>{j + 1}</sub> on its first mass, so {r_text}: every mass moves with the base, as a rigid body. "
        "In a 3-D model each node would have x, y and z DOFs, and <b>r</b> for a shake along x would have 1 "
        "at the x DOFs and 0 at the y and z DOFs (and lever arms for rotations). The chain has only x DOFs, "
        "so there is only the one direction and <b>r</b> = <b>1</b>. "
        "Moving the base accelerates every mass, so the load on B is inertial, −M<sub>B</sub><b>r</b>ü: "
        "spread over <i>every</i> mass, in proportion to its mass.</p>"
    )

    # 5. Participation
    part_b = mb_diag * r_vec * phi_b
    gamma = float(part_b.sum())
    parts.append("<h3>5. Participation: project the mode on the load</h3>")
    parts.append(
        _side_by_side(
            f"<p><b>A{ra + 1}</b>: φ<sup>T</sup><b>e</b><sub>tip</sub></p>"
            + _vec_table(
                ["", "e<sub>tip,i</sub>", "φ<sub>i</sub>", "e<sub>tip,i</sub>φ<sub>i</sub>"],
                [e_tip, phi_a, e_tip * phi_a],
                la,
                ["", "", f"φ<sub>tip</sub> = {_g(tip)}"],
                bold_row=j - 1,
            ),
            f"<p><b>B{rb + 1}</b>: Γ = φ<sup>T</sup>M<sub>B</sub><b>r</b></p>"
            + _vec_table(
                ["", "m<sub>i</sub>", "r<sub>i</sub>", "φ<sub>i</sub>", "m<sub>i</sub>r<sub>i</sub>φ<sub>i</sub>"],
                [mb_diag, r_vec, phi_b, part_b],
                lb,
                ["", "", "", f"Γ = {_g(gamma)}"],
            ),
        )
    )
    parts.append(
        "<p>This is why only <b>one entry</b> of φ counts for A: the point load vector is zero at every DOF but "
        "the tip, so the projection keeps φ<sub>tip</sub> alone. For B the load is at every mass, so every "
        "entry counts, weighted by its mass: Γ = Σ m<sub>i</sub>r<sub>i</sub>φ<sub>i</sub>. Units: "
        "φ<sub>tip</sub> in 1/√kg; Γ in (kg)(1)(1/√kg) = √kg.</p>"
    )

    # 6. From participation to the oscillator's mass
    m_tip = float(A.tip_masses()[ra])
    parts.append("<h3>6. From participation to the oscillator's mass</h3>")
    parts.append(
        f"<p><b>A: m<sub>a</sub> = 1/φ<sub>tip</sub>²</b>. With the modal coordinate q (x = φq), the mode obeys "
        f"q̈ + ω²q = φ<sub>tip</sub>F, and the joint sees the tip move by u<sub>a</sub> = φ<sub>tip</sub>q. "
        f"Multiply by φ<sub>tip</sub>: ü<sub>a</sub> + ω²u<sub>a</sub> = φ<sub>tip</sub>²F, which is an oscillator "
        f"of mass 1/φ<sub>tip</sub>². The participation appears <i>twice</i> (in how F drives q, and in how q moves "
        f"the tip) and <i>divides</i>: a large tip motion per unit q makes the mode look light.</p>"
        f"<p>&nbsp;&nbsp;m<sub>tip</sub> = 1/φ<sub>tip</sub>² = 1/{_g(tip)}² = <b>{_g(m_tip)} kg</b> = "
        f"ψ<sup>T</sup>Mψ of step 3 ({_g(gm_a)} kg): the tip-scaled shape's generalized mass. Normalizing put the "
        "whole mode's mass into the tip entry, and squaring and inverting it gets that mass back.</p>"
    )
    parts.append(
        f"<p><b>B: m<sub>b</sub> = Γ²</b>. The base motion drives the mode, q̈ + ω²q = −Γü, and the mode pushes "
        f"back on the base with Γq̈ (on top of m<sub>B</sub>ü for B moving rigidly). With y = q/Γ this is "
        f"ÿ + ω²y = −ü and a base force Γ²ÿ + m<sub>B</sub>ü: an oscillator of mass Γ² on the base. The "
        f"participation again appears twice (driving and pushing back) and here <i>multiplies</i>.</p>"
        f"<p>&nbsp;&nbsp;m<sub>eff</sub> = Γ² = {_g(gamma)}² = <b>{_g(gamma**2)} kg</b>; scale-free, "
        f"(ψ<sup>T</sup>M<b>r</b>)²/(ψ<sup>T</sup>Mψ) = {_g(float(psi_b @ B.M @ r_vec))}²/{_g(gm_b)} = "
        f"{_g(float(psi_b @ B.M @ r_vec) ** 2 / gm_b)} kg.</p>"
    )
    eff = B.effective_masses
    parts.append(
        "<p>Every mode of B, with its share. They add up to <b>r</b><sup>T</sup>M<sub>B</sub><b>r</b> = "
        f"m<sub>B</sub> = {_g(B.total_mass)} kg:</p>"
    )
    parts.append(
        _vec_table(
            ["Mode", "f [Hz]", "Γ [√kg]", "Γ² [kg]", "share"],
            [B.fn_hz, B.participation, eff, eff / B.total_mass],
            [f"B{k + 1}" for k in range(eff.size)],
            ["", "", f"{_g(eff.sum())}", f"{eff.sum() / B.total_mass:.3g}"],
            bold_row=rb,
        )
    )
    parts.append(
        f"<p>Residual mass: m<sub>B</sub> − Γ² = {_g(B.total_mass)} − {_g(gamma**2)} = "
        f"<b>{_g(model.m_residual)} kg</b>, B's other modes moving with the base (A's tip)"
        + (", added to m<sub>a</sub>." if model.residual else "; not added here (the check box is off).")
        + "</p>"
    )
    gamma_a = float(A.participation[ra])
    parts.append(
        f"<p><b>For comparison, A's effective mass</b> uses A's own base motion, <b>r</b><sub>A</sub> = <b>1</b>: "
        f"Γ<sub>A</sub> = Σ m<sub>i</sub>φ<sub>i</sub> = {_g(gamma_a)} √kg, Γ<sub>A</sub>² = <b>{_g(gamma_a**2)} kg</b>, "
        f"against m<sub>tip</sub> = {_g(m_tip)} kg. It projects A's mode on a load at every mass, but the joint "
        "loads A at its tip only"
        + (
            " (and it is the mass in use now: <i>Mass of A's oscillator</i> is set to the effective mass)."
            if model.a_mass == "effective"
            else "."
        )
        + "</p>"
    )

    # 7. The oscillators
    a, b = model.osc_a, model.osc_b
    ca, cb = A.modal_damping[ra], B.modal_damping[rb]
    parts.append("<h3>7. The two oscillators</h3>")
    parts.append(
        "<p>Each keeps its mode's frequency and modal damping: k = mω², c = m·φ<sup>T</sup>Cφ (= 2ζωm).</p>"
        f"<p>&nbsp;&nbsp;A{ra + 1}: m = {_g(a.m)} kg, k<sub>a</sub> = {_g(a.m)} × {_g(A.omegas[ra])}² = "
        f"<b>{_g(a.k)} N/m</b>, φ<sup>T</sup>C<sub>A</sub>φ = {_g(ca)} 1/s, c<sub>a</sub> = <b>{_g(a.c)} N·s/m</b>"
        + (
            f"; m<sub>a</sub> = {_g(a.m)} + {_g(model.m_residual)} (residual) = <b>{_g(model.m_a)} kg</b>"
            if model.residual
            else f"; m<sub>a</sub> = <b>{_g(model.m_a)} kg</b>"
        )
        + f"<br>&nbsp;&nbsp;B{rb + 1}: m<sub>b</sub> = <b>{_g(b.m)} kg</b>, k<sub>b</sub> = {_g(b.m)} × "
        f"{_g(B.omegas[rb])}² = <b>{_g(b.k)} N/m</b>, φ<sup>T</sup>C<sub>B</sub>φ = {_g(cb)} 1/s, "
        f"c<sub>b</sub> = <b>{_g(b.c)} N·s/m</b></p>"
    )

    # 8. The 2-DOF model
    l2 = ["u<sub>a</sub>", "u<sub>b</sub>"]
    parts.append("<h3>8. The 2-DOF model</h3>")
    parts.append(
        "<p>ground — k<sub>a</sub>, c<sub>a</sub> — m<sub>a</sub> — k<sub>b</sub>, c<sub>b</sub> — "
        "m<sub>b</sub>, assembled like a 2-mass chain:</p>"
    )
    parts.append(_side_by_side(_mat(model.K, l2, "q", "K"), _mat(model.C, l2, "q", "C"), _mat(model.M, l2, "q", "M")))
    zs = ", ".join("overdamped" if z is None else f"{z:.4f}" for z in model.zetas)
    parts.append(
        f"<p>K η = ω² M η gives <b>{_g(model.fn_hz[0])}</b> and <b>{_g(model.fn_hz[1])} Hz</b> (ζ = {zs}, exact, "
        f"from the damped eigenvalues). The chain's lowest two: {_g(full.modes[0].fn_hz)}"
        + (f" and {_g(full.modes[1].fn_hz)}" if len(full.modes) > 1 else "")
        + " Hz. μ = m<sub>b</sub>/m<sub>a</sub> "
        f"= {_g(model.mass_ratio)}.</p>"
    )

    # 9. Rayleigh-Ritz check
    T = np.zeros((n, 2))
    T[:j, 0] = psi_a if abs(tip) > 1e-12 else 0.0
    rel = B.participation[rb] * phi_b
    T[j:, 0] = r_vec - rel
    T[j:, 1] = rel
    parts.append("<h3>9. Check: the 2-DOF model is a Rayleigh–Ritz model of the chain</h3>")
    parts.append(
        "<p>The chain restricted to two shapes, x = T[u<sub>a</sub>; u<sub>b</sub>]: column 1 is A's mode scaled "
        "to 1 at the tip with B carried rigidly (<b>r</b>) minus B's mode, column 2 is B's mode relative to its "
        "base, Γφ<sub>B</sub>. Projecting the chain's matrices, T<sup>T</sup>MT and T<sup>T</sup>KT:</p>"
    )
    TM, TK = T.T @ M @ T, T.T @ K @ T
    parts.append(
        _side_by_side(
            matrix_html(T, labels, l2, groups, ["q", "q"], "T"),
            _mat(TK, l2, "q", "T<sup>T</sup>KT"),
            _mat(TM, l2, "q", "T<sup>T</sup>MT"),
        )
    )
    same = np.allclose(TM, model.M, atol=1e-9 * max(1.0, M.max())) and np.allclose(
        TK, model.K, rtol=1e-9, atol=1e-9 * K.max()
    )
    if same:
        parts.append(
            "<p>These are the 2-DOF K and M of step 8 exactly: the oscillator model <i>is</i> this "
            "Rayleigh–Ritz model, so its frequencies are upper bounds of the chain's lowest two. (T<sup>T</sup>MT "
            "is diagonal: B's mode, relative to its base, does not share inertia with the rigid motion, since "
            "Γ·Γ − Γ²·φ<sup>T</sup>Mφ = 0.)</p>"
        )
    else:
        parts.append(
            f"<p style='color:{colors.poor}'>These differ from the 2-DOF matrices of step 8: with "
            + ("no residual mass" if not model.residual else "A's effective mass")
            + " the oscillator model is not this Rayleigh–Ritz model, so it gives no bound.</p>"
        )
    parts.append("<h3><a name='notation'></a>Notation</h3>" + NOTATION_HTML)
    return "".join(parts)
