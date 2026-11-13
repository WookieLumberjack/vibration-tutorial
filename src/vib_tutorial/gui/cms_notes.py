"""Text for the Substructuring page: the theory notes and the live matrix walkthrough."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import (
    ChainSystem,
    CMSModel,
    ForceKind,
    ModalResult,
    Substructure,
    compare_modes,
    substructure_damping,
)
from ..core.modal import TWO_PI
from . import cms_figures
from .lesson import trials_html
from .links import load_notes
from .style import colors


@dataclass(frozen=True)
class CmsTrial:
    """A uniform chain (8 masses unless `n` says) for the Simulation page, and how this page cuts and reduces it."""

    name: str
    note: str
    interfaces: tuple[int, ...] = ()  # 0-based interface masses
    kept: tuple[int, ...] = (1,)  # modes kept per substructure, the last repeated
    method: str = "craig-bampton"
    tab: int = 0  # Modes & FRF, Time response, Back expansion, Compare methods, Basis, Matrices
    force: ForceKind | None = None  # applied on the tab's tip force (Time response or Back expansion)
    tune: int = 0  # harmonic force at this mode (1-based) of the full chain
    floating: bool = False  # k1 = k2 = 0
    mode: int = 0  # the comparison table's row to select (1-based), 0 to leave it
    n: int = 8  # masses in the uniform chain
    changed: bool = False  # the chain isn't this uniform one (a practice spec's state: edited on the Simulation page)

    def system(self) -> ChainSystem:
        system = ChainSystem.uniform(self.n)
        if self.floating:
            system.stiffness[:2] = 0.0
        return system


STEP, HARMONIC = ForceKind.STEP, ForceKind.HARMONIC
TRIALS = (
    CmsTrial(
        "One substructure: Guyan",
        "No cut and no modes kept: only <m>x<sub>8</sub></m> is left, one DOF and one frequency, 0.630 Hz against "
        "the true 0.587 Hz (+7.3%). The Basis tab shows why: the one constraint mode is a straight line "
        "from the ground to the tip, so this is Rayleigh's quotient with a linear shape.",
        kept=(0,),
        tab=4,
    ),
    CmsTrial(
        "One substructure: one mode kept",
        "The first mode with the tip held (1.24 Hz) joins the basis. Mode 1 falls to 0.33% and mode 2 is "
        "7.4% high. Keep 2 and mode 3 is 4.4% high; keep all 7 (All modes) and every error is zero.",
        kept=(1,),
    ),
    CmsTrial(
        "Guyan, in time",
        "A step force at the tip on the Guyan model. It can only move in its one straight-line shape, so it "
        "rings at the wrong frequency and drifts in and out of phase with the full chain: the error swings "
        "between a small fraction and most of the tip motion. Raise the modes kept and watch the two "
        "chains lock together.",
        kept=(0,),
        tab=1,
        force=STEP,
    ),
    CmsTrial(
        "One substructure: Rubin",
        "Rubin with 1 mode. The free-interface modes of the only substructure are the chain's own modes, so "
        "mode 1 is exact, but mode 2 is 16% high: the residual flexibility gets the statics right, not the "
        "next mode.",
        kept=(1,),
        method="rubin",
    ),
    CmsTrial(
        "Cut at m4, one mode each",
        "4 DOFs (<m>q<sub>A1</sub>, q<sub>B1</sub>, x<sub>4</sub>, x<sub>8</sub></m>) instead of 8. Mode 1 is "
        "within 0.03%, mode 4 is 7% high with MAC 0.84, and modes 5 to 8 aren't in the model at all. "
        "Under <i>Substructures on their own</i>, A and B are identical three-mass pieces with a clamped "
        "mode at 2.44 Hz each; coupled, they split into modes 3 (2.84 Hz) and 4 (3.84 Hz), each carrying "
        "about half its energy.",
        interfaces=(3,),
    ),
    CmsTrial(
        "Cut at m4: Guyan",
        "Only the two boundary DOFs are left: mode 1 is still within 2%, mode 2 is 11% high. The reduced "
        "FRF matches at low frequency and drifts above mode 1. Click <i>All modes (exact)</i>: every error "
        "drops to zero, although the matrices look nothing like M and K.",
        interfaces=(3,),
        kept=(0,),
    ),
    CmsTrial(
        "Cut at m4, two modes each",
        "6 DOFs. Every frequency stays above the true one. The kept fixed-interface modes reach 4.5 Hz "
        "(Matrices tab, step 3), and modes 1 to 3 (up to 2.8 Hz) are within 0.03%; mode 6 is still 2.7% "
        "high.",
        interfaces=(3,),
        kept=(2,),
        tab=5,
    ),
    CmsTrial(
        "The interface at m7: the rule of thumb",
        "B is just the last spring, with no interior, so all the reduction happens in A. Here A keeps 3 "
        "modes (up to 3.97 Hz) and modes 1 to 4 are within 0.2%. Set A's modes kept to 1: 3 DOFs, and mode "
        "3 is 50% too high (MAC 0.27).",
        interfaces=(6,),
        kept=(3, 1),
    ),
    CmsTrial(
        "Driving at mode 4",
        "Cut at m4, 1 mode each, a harmonic force at the full chain's mode 4 (3.84 Hz). The Craig–Bampton "
        "model resonates at 4.11 Hz instead, so its tip moves 30% less and lags behind: the error is about "
        "half the motion. Keep 2 modes in each and it drops to 9%; keep 3 and the two chains move as one. "
        "(Tuned to mode 1, they stay within 1%.)",
        interfaces=(3,),
        tab=1,
        force=HARMONIC,
        tune=4,
    ),
    CmsTrial(
        "Three substructures",
        "Cut at m3 and m6, 1 mode in each of A, B and C: 6 DOFs, as with one cut at m4 and 2 modes each, "
        "but 3 of them are interfaces and the tip, and mode 4 is 2.3% high (one cut, 2 modes each: within "
        "0.71%). Click <i>Guyan</i>: 3 physical DOFs, and mode 3 is 16% high.",
        interfaces=(2, 5),
    ),
    CmsTrial(
        "Free interface: Rubin",
        "Cut at m4, 1 mode each, Rubin. B's one kept mode is now its rigid-body mode (B1 at 0 Hz, carrying "
        "83% of coupled mode 1's kinetic energy). Mode 1 is within 0.003% (Craig–Bampton: 0.02%) and mode 2 "
        "within 0.9%, but mode 4 is 10% high against Craig–Bampton's 7%. In the Basis tab the boundary "
        "columns are residual attachment modes, which already bend like the higher modes.",
        interfaces=(3,),
        method="rubin",
    ),
    CmsTrial(
        "Free interface: MacNeal",
        "Still 4 coordinates, but the boundary DOFs are massless, so there are only 2 modes: mode 1 is 0.6% "
        "high and mode 2 22%. Click <i>All modes</i>: 6 modes now, still not exact (mode 6 is 66% high), "
        "because the mass of B's discarded modes is gone. Drive it at mode 1 (Time response): the 0.6% "
        "matters on a mode this lightly damped, and the tip error grows to about half the motion.",
        interfaces=(3,),
        method="macneal",
    ),
    CmsTrial(
        "Three methods compared",
        "Mode 3's error against the model's size, for each method. Every curve falls as modes are added; "
        "Craig–Bampton and Rubin reach the exact answer with all modes (drawn at the floor), MacNeal stays "
        "0.8% high, and for mode 3 Craig–Bampton and Rubin take turns in the lead. Select mode 1: Rubin is "
        "ahead at every size.",
        interfaces=(3,),
        tab=3,
        mode=3,
    ),
    CmsTrial(
        "Parts that float",
        "<m>k<sub>1</sub> = k<sub>2</sub></m> = 0, so m1 and m2 aren't held by anything but the interface. "
        "Craig–Bampton fails (m1 and m2 float when the boundary is held: switch to it and see), but the "
        "free-interface methods simply find more rigid-body modes in A.",
        interfaces=(3,),
        method="rubin",
        floating=True,
    ),
    CmsTrial(
        "Back expansion after a step",
        "Cut at m4, 1 mode each, a step force; m2 and spring k2 plotted. The hollow masses are recovered, the "
        "solid ones come straight from the coupled solve. Once the ringing dies away every recovery agrees "
        "with the truth (the constraint modes are exact static shapes). While it rings, x2 is about 2% off "
        "from the coupled solution and 4% from the boundary alone: the gap is the q part. With Guyan the two "
        "recoveries are the same, both about 28% off.",
        interfaces=(3,),
        tab=2,
        force=STEP,
    ),
    CmsTrial(
        "Recovering a force",
        "2 modes each, driven at mode 4. <m>x<sub>b</sub></m> is within about 9%, the coupled recovery of x2 within "
        "15%, but its force in k2 is off by nearly 40%: a force is a difference of displacements, which "
        "brings out the truncated higher modes. The enhanced recovery puts them back (about 12%). Keep 3 "
        "(all) modes and every recovery but the boundary-only one is exact.",
        interfaces=(3,),
        kept=(2,),
        tab=2,
        force=HARMONIC,
        tune=4,
    ),
)


def legend_html() -> str:
    """The key to the cell tints (Theme.block_tints: one per partition block)."""
    tints = colors.block_tints
    return (
        "<p><small>Cell colors: "
        f"<span style='background:{tints['i', 'i']}'>&nbsp;interior–interior (ii)&nbsp;</span> "
        f"<span style='background:{tints['b', 'i']}'>&nbsp;interior–boundary (ib, bi)&nbsp;</span> "
        f"<span style='background:{tints['b', 'b']}'>&nbsp;boundary–boundary (bb)&nbsp;</span> "
        f"<span style='background:{tints['q', 'q']}'>&nbsp;modal–modal (qq)&nbsp;</span> "
        f"<span style='background:{tints['b', 'q']}'>&nbsp;modal–boundary (qb)&nbsp;</span>"
        "</small></p>"
    )


THEORY_HTML = load_notes("cms_notes", figures=cms_figures.FIGURES, trials=trials_html(TRIALS))


def fmt(v: float, scale: float) -> str:
    """Four significant figures; entries negligible relative to the matrix are shown as 0."""
    if abs(v) <= 1e-10 * scale:
        return "0"
    return f"{v:.4g}"


def matrix_html(
    A: np.ndarray,
    rows: list[str],
    cols: list[str],
    row_groups: list[str],
    col_groups: list[str],
    title: str = "",
    dim_cols: int | None = None,
) -> str:
    """An HTML table with labeled rows/columns and cells tinted by partition block.

    Columns from dim_cols on are greyed out (e.g. discarded modes).
    """
    scale = float(np.abs(A).max()) if A.size else 1.0
    head = "".join(f"<th>{c}</th>" for c in cols)
    out = [f"<table border='1' cellspacing='0' cellpadding='3'><tr><th>{title}</th>{head}</tr>"]
    for r, (label, rg) in enumerate(zip(rows, row_groups)):
        cells = []
        for c, cg in enumerate(col_groups):
            dim = dim_cols is not None and c >= dim_cols
            tint = colors.dim_tint if dim else colors.block_tints[tuple(sorted((rg, cg)))]
            text = fmt(A[r, c], scale)
            color = f" style='color:{colors.grey}'" if text == "0" or dim else ""
            cells.append(f"<td align='right' bgcolor='{tint}'{color}>{text}</td>")
        out.append(f"<tr><th>{label}</th>{''.join(cells)}</tr>")
    out.append("</table>")
    return "".join(out)


def _dof_labels(dofs: np.ndarray) -> list[str]:
    return [f"x{d + 1}" for d in dofs]


def _local(sub: Substructure, mat: np.ndarray, title: str) -> str:
    groups = ["i"] * sub.ni + ["b"] * sub.nb
    labels = _dof_labels(sub.dofs)
    return matrix_html(mat, labels, labels, groups, groups, title)


def _reduced_labels(sub: Substructure) -> tuple[list[str], list[str]]:
    labels = [f"q<sub>{sub.name}{k + 1}</sub>" for k in range(sub.n_kept)] + _dof_labels(sub.boundary)
    return labels, ["q"] * sub.n_kept + ["b"] * sub.nb


def _side_by_side(*tables: str) -> str:
    cells = "".join(f"<td valign='top' style='padding-right:14px'>{t}</td>" for t in tables)
    return f"<table cellspacing='0' cellpadding='0'><tr>{cells}</tr></table>"


def _join(items: list[str]) -> str:
    return ", ".join(items) if items else "none"


def matrices_html(model: CMSModel, full: ModalResult | None = None) -> str:
    """Every step of the reduction, with the current numbers."""
    system = model.system
    n = system.n
    M, C, K = system.matrices()
    subs = model.substructures
    bnd = {int(b) for b in model.boundary}
    groups = ["b" if d in bnd else "i" for d in range(n)]
    labels = _dof_labels(np.arange(n))
    interfaces = [int(b) for b in model.boundary[:-1]]
    parts = [legend_html()]

    parts.append("<h3>1. The full model</h3>")
    parts.append(
        f"<p>N = {n} physical DOFs. Boundary (master) DOFs: <b>{_join(_dof_labels(model.boundary))}</b> "
        + (
            "(the interface" + ("s" if len(interfaces) > 1 else "") + " and the loaded tip). "
            if interfaces
            else "(only the loaded tip: with no cut the whole chain is one substructure). "
        )
        + f"Interior: {_join([l for l, g in zip(labels, groups) if g == 'i'])}. "
        "K [N/m], C [N·s/m] and M [kg]. C is assembled from the dampers exactly like K from the "
        "springs, so it has the same tridiagonal pattern.</p>"
    )
    parts.append(
        _side_by_side(
            matrix_html(K, labels, labels, groups, groups, "K"),
            matrix_html(C, labels, labels, groups, groups, "C"),
            matrix_html(M, labels, labels, groups, groups, "M"),
        )
    )

    parts.append("<h3>2. Cut into substructures</h3>")
    for sub in subs:
        springs = ", ".join(f"k<sub>{e + 1}</sub>, c<sub>{e + 1}</sub>" for e in sub.elements)
        parts.append(
            f"<p><b>Substructure {sub.name}</b>: springs and dampers {springs}; interior "
            f"{_join(_dof_labels(sub.interior))}; boundary {_join(_dof_labels(sub.boundary))}. "
            "Reordered [interior | boundary]:</p>"
        )
        parts.append(
            _side_by_side(
                _local(sub, sub.K, f"K<sup>({sub.name})</sup>"),
                _local(sub, sub.C, f"C<sup>({sub.name})</sup>"),
                _local(sub, sub.M, f"M<sup>({sub.name})</sup>"),
            )
        )
    for j in interfaces:
        left = next(s for s in subs if j in s.boundary and j + 1 not in s.dofs)
        right = next(s for s in subs if j in s.boundary and s is not left)
        parts.append(
            f"<p>The interface entry of the full model, K<sub>{j + 1},{j + 1}</sub> = k<sub>{j + 1}</sub> + "
            f"k<sub>{j + 2}</sub> = {fmt(K[j, j], 1)}, is split: {fmt(system.stiffness[j], 1)} goes to "
            f"{left.name} and {fmt(system.stiffness[j + 1], 1)} to {right.name}. C<sub>{j + 1},{j + 1}</sub> "
            f"= c<sub>{j + 1}</sub> + c<sub>{j + 2}</sub> = {fmt(C[j, j], 1)} splits the same way "
            f"({fmt(system.damping[j], 1)} + {fmt(system.damping[j + 1], 1)}). The interface mass "
            f"m<sub>{j + 1}</sub> = {fmt(system.masses[j], 1)} "
            + (
                f"is split half and half, {fmt(system.masses[j] / 2, 1)} to each, so that both have mass "
                "at every DOF their free modes can move (any split works: assembly adds them back). "
                if model.free
                else f"goes to {left.name} (any split works: assembly adds them back). "
            )
            + "Every other entry belongs to one substructure only, so "
            "K = Σ L<sub>s</sub><sup>T</sup>K<sup>(s)</sup>L<sub>s</sub> exactly, and the same for C and M.</p>"
        )
    if model.free:
        parts += _free_steps(model)
    else:
        parts += _fixed_steps(model)
    parts += _assembly_steps(model, full)
    return "".join(parts)


def _fixed_steps(model: CMSModel) -> list[str]:
    """Craig-Bampton steps 3 to 6: fixed-interface modes, constraint modes, T, reduced matrices."""
    subs = model.substructures
    parts = []

    parts.append("<h3>3. Fixed-interface normal modes</h3>")
    parts.append(
        "<p>Boundary clamped (x<sub>b</sub> = 0): K<sub>ii</sub>φ = ω²M<sub>ii</sub>φ, "
        "mass-normalized so Φ<sup>T</sup>M<sub>ii</sub>Φ = I. These are <i>undamped</i> modes: "
        "C plays no part in choosing the basis (steps 3 to 5). It is only projected onto that "
        "basis in step 6.</p>"
    )
    for sub in subs:
        if not sub.ni:
            parts.append(
                f"<p><b>{sub.name}</b> has no interior DOFs, so it has no fixed-interface modes: "
                "it is represented by its boundary DOFs alone.</p>"
            )
            continue
        freqs = ", ".join(
            (f"<b>{f:.4g}</b>" if r < sub.n_kept else f"<span style='color:{colors.grey}'>{f:.4g}</span>")
            for r, f in enumerate(sub.omegas / TWO_PI)
        )
        parts.append(
            f"<p><b>{sub.name}</b>: f = {freqs} Hz (<b>bold</b>: kept, {sub.n_kept} of {sub.ni}; grey: discarded).</p>"
        )
        cols = [f"φ<sub>{r + 1}</sub>" for r in range(sub.ni)]
        parts.append(
            matrix_html(
                sub.Phi,
                _dof_labels(sub.interior),
                cols,
                ["i"] * sub.ni,
                ["q"] * sub.ni,
                f"Φ<sup>({sub.name})</sup>",
                dim_cols=sub.n_kept,
            )
        )

    parts.append("<h3>4. Constraint modes</h3>")
    parts.append(
        "<p>Ψ = −K<sub>ii</sub><sup>−1</sup>K<sub>ib</sub>: column j is the static shape of the "
        "interior when boundary DOF j moves 1 m and the other boundary DOFs are held. Along a "
        "chain the interior simply interpolates between its boundary masses, weighted by the "
        "spring flexibilities.</p>"
    )
    tables = []
    for sub in subs:
        if sub.ni:
            tables.append(
                matrix_html(
                    sub.Psi,
                    _dof_labels(sub.interior),
                    _dof_labels(sub.boundary),
                    ["i"] * sub.ni,
                    ["b"] * sub.nb,
                    f"Ψ<sup>({sub.name})</sup>",
                )
            )
    parts.append(_side_by_side(*tables) if tables else "<p>No interior DOFs.</p>")

    parts.append("<h3>5. Transformation</h3>")
    parts.append(
        "<p>[x<sub>i</sub>; x<sub>b</sub>] = T [q; x<sub>b</sub>] with "
        "T = [ Φ<sub>k</sub> Ψ ; 0 I ]. Only the kept modes appear, so T is tall: that is "
        "the reduction.</p>"
    )
    tables = []
    for sub in subs:
        red_labels, red_groups = _reduced_labels(sub)
        tables.append(
            matrix_html(
                sub.T,
                _dof_labels(sub.dofs),
                red_labels,
                ["i"] * sub.ni + ["b"] * sub.nb,
                red_groups,
                f"T<sup>({sub.name})</sup>",
            )
        )
    parts.append(_side_by_side(*tables))

    parts.append("<h3>6. Reduced substructure matrices</h3>")
    parts.append(
        "<p>K̂ = T<sup>T</sup>KT is block diagonal: the kept ω² (in rad²/s²) for q, and the "
        "boundary stiffness K̂<sub>bb</sub> = K<sub>bb</sub> − K<sub>bi</sub>K<sub>ii</sub><sup>−1</sup>"
        "K<sub>ib</sub>. M̂ = T<sup>T</sup>MT has I for q and the coupling M̂<sub>qb</sub>. "
        "The damping is projected onto the same basis, Ĉ = T<sup>T</sup>CT:</p>"
        "<p>&nbsp;&nbsp;Ĉ<sub>qq</sub> = Φ<sub>k</sub><sup>T</sup>C<sub>ii</sub>Φ<sub>k</sub>, &nbsp; "
        "Ĉ<sub>qb</sub> = Φ<sub>k</sub><sup>T</sup>(C<sub>ii</sub>Ψ + C<sub>ib</sub>), &nbsp; "
        "Ĉ<sub>bb</sub> = C<sub>bb</sub> + C<sub>bi</sub>Ψ + Ψ<sup>T</sup>C<sub>ib</sub> + "
        "Ψ<sup>T</sup>C<sub>ii</sub>Ψ</p>"
        "<p>Unlike K̂, Ĉ is <b>not</b> block diagonal in general. Ψ is a static solution for K, "
        "not for C, so Ĉ<sub>qb</sub> ≠ 0, and the undamped Φ<sub>k</sub> only diagonalize "
        "Ĉ<sub>qq</sub> when the damping is proportional. The exception is damping proportional "
        "to stiffness inside the substructure (C<sup>(s)</sup> = βK<sup>(s)</sup>, e.g. every "
        "c<sub>i</sub>/k<sub>i</sub> equal): then Ĉ = βK̂ and it inherits K̂'s block-diagonal form. "
        "The diagonal of Ĉ<sub>qq</sub> gives each kept fixed-interface mode a damping ratio "
        "ζ<sub>r</sub> = Ĉ<sub>rr</sub> / 2ω<sub>r</sub>.</p>"
    )
    for sub in subs:
        red_labels, red_groups = _reduced_labels(sub)
        parts.append(
            _side_by_side(
                matrix_html(sub.K_red, red_labels, red_labels, red_groups, red_groups, f"K̂<sup>({sub.name})</sup>"),
                matrix_html(sub.C_red, red_labels, red_labels, red_groups, red_groups, f"Ĉ<sup>({sub.name})</sup>"),
                matrix_html(sub.M_red, red_labels, red_labels, red_groups, red_groups, f"M̂<sup>({sub.name})</sup>"),
            )
        )
        parts.append(_damping_note(sub))
    return parts


def _free_steps(model: CMSModel) -> list[str]:
    """Rubin / MacNeal steps 3 to 6: free-interface modes, residual flexibility, T, reduced matrices."""
    subs = model.substructures
    parts = [
        "<h3>3. Free-interface normal modes</h3>",
        "<p>Boundary free: Kφ = ω²Mφ over <i>all</i> the substructure's DOFs, mass-normalized so "
        "Φ<sup>T</sup>MΦ = I. Every mode moves the boundary too. Rigid-body modes (ω = 0) come first "
        "and are always kept. These are <i>undamped</i> modes: C plays no part in choosing the "
        "basis.</p>",
    ]
    for sub in subs:
        if not sub.ni:
            parts.append(
                f"<p><b>{sub.name}</b> has no interior DOFs, so there is nothing to reduce: it is "
                "kept as its physical boundary DOFs.</p>"
            )
            continue
        freqs = ", ".join(
            (f"<b>{f:.4g}</b>" if r < sub.n_kept else f"<span style='color:{colors.grey}'>{f:.4g}</span>")
            for r, f in enumerate(np.where(sub.omegas > 1e-9, sub.omegas, 0.0) / TWO_PI)
        )
        rigid = f" {sub.n_rigid} rigid-body mode{'s' if sub.n_rigid > 1 else ''} at 0 Hz." if sub.n_rigid else ""
        parts.append(
            f"<p><b>{sub.name}</b>: f = {freqs} Hz (<b>bold</b>: kept, {sub.n_kept} of "
            f"{sub.omegas.size}; grey: discarded).{rigid}</p>"
        )
        cols = [f"φ<sub>{r + 1}</sub>" for r in range(sub.omegas.size)]
        parts.append(
            matrix_html(
                sub.Phi,
                _dof_labels(sub.dofs),
                cols,
                ["i"] * sub.ni + ["b"] * sub.nb,
                ["q"] * sub.omegas.size,
                f"Φ<sup>({sub.name})</sup>",
                dim_cols=sub.n_kept,
            )
        )

    parts.append("<h3>4. Residual flexibility</h3>")
    parts.append(
        "<p>G<sub>d</sub> = Φ<sub>d</sub>Λ<sub>d</sub><sup>−1</sup>Φ<sub>d</sub><sup>T</sup> "
        "[m/N]: the static flexibility of the <i>discarded</i> modes, which respond almost "
        "statically to the interface forces because they are high in frequency. Without "
        "rigid-body modes it equals K<sup>−1</sup> − Φ<sub>k</sub>Λ<sub>k</sub><sup>−1</sup>"
        "Φ<sub>k</sub><sup>T</sup>, so in practice the discarded modes are never computed. "
        "Only its boundary columns are used: G<sub>d</sub>E<sub>b</sub> are the residual "
        "attachment modes. Scaled to a unit boundary displacement, "
        "R = G<sub>ib</sub>G<sub>bb</sub><sup>−1</sup>.</p>"
    )
    for sub in subs:
        if not sub.ni:
            continue
        groups = ["i"] * sub.ni + ["b"] * sub.nb
        labels = _dof_labels(sub.dofs)
        assert sub.G is not None, "a free-interface reduction keeps G"
        parts.append(
            _side_by_side(
                matrix_html(sub.G, labels, labels, groups, groups, f"G<sub>d</sub><sup>({sub.name})</sup>"),
                matrix_html(
                    sub.Psi,
                    _dof_labels(sub.interior),
                    _dof_labels(sub.boundary),
                    ["i"] * sub.ni,
                    ["b"] * sub.nb,
                    f"R<sup>({sub.name})</sup>",
                ),
            )
        )

    parts.append("<h3>5. Transformation</h3>")
    parts.append(
        "<p>x = Φ<sub>k</sub>q + G<sub>d</sub>E<sub>b</sub>f<sub>b</sub>. Its boundary rows "
        "give f<sub>b</sub> = G<sub>bb</sub><sup>−1</sup>(x<sub>b</sub> − Φ<sub>bk</sub>q), and "
        "substituting: [x<sub>i</sub>; x<sub>b</sub>] = T [q; x<sub>b</sub>] with "
        "T = [ Φ<sub>ik</sub> − RΦ<sub>bk</sub> &nbsp;R ; 0 &nbsp;I ]. The same shape as "
        "Craig–Bampton's T, so x<sub>b</sub> is physical and assembly is unchanged. The modal "
        "columns are the kept free modes with their boundary motion taken out.</p>"
    )
    tables = []
    for sub in subs:
        red_labels, red_groups = _reduced_labels(sub)
        tables.append(
            matrix_html(
                sub.T,
                _dof_labels(sub.dofs),
                red_labels,
                ["i"] * sub.ni + ["b"] * sub.nb,
                red_groups,
                f"T<sup>({sub.name})</sup>",
            )
        )
    parts.append(_side_by_side(*tables))

    parts.append("<h3>6. Reduced substructure matrices</h3>")
    parts.append(
        "<p>K̂ = T<sup>T</sup>KT = [ Λ<sub>k</sub> + Φ<sub>bk</sub><sup>T</sup>G<sub>bb</sub><sup>−1</sup>"
        "Φ<sub>bk</sub> &nbsp;−Φ<sub>bk</sub><sup>T</sup>G<sub>bb</sub><sup>−1</sup> ; "
        "−G<sub>bb</sub><sup>−1</sup>Φ<sub>bk</sub> &nbsp;G<sub>bb</sub><sup>−1</sup> ]: unlike "
        "Craig–Bampton, the modes are coupled to the boundary by stiffness. G<sub>bb</sub><sup>−1</sup> "
        "is the residual stiffness at the boundary.</p>"
    )
    if model.method == "rubin":
        parts.append(
            "<p><b>Rubin</b>: M̂ = T<sup>T</sup>MT, a Rayleigh–Ritz projection. The residual "
            "attachment modes carry mass (the <i>residual mass</i>), so the modes are coupled to "
            "the boundary by mass as well. The damping is projected the same way, "
            "Ĉ = T<sup>T</sup>CT.</p>"
        )
    else:
        parts.append(
            "<p><b>MacNeal</b>: the residual flexibility is a massless spring. Only the kept "
            "modes carry inertia: M̂ = [ I 0 ; 0 0 ], so the boundary rows of M̂ are zero. The "
            "damping is projected like the stiffness, Ĉ = T<sup>T</sup>CT (damping was not part "
            "of MacNeal's method; this way it differs from Rubin's only in the mass).</p>"
        )
    for sub in subs:
        red_labels, red_groups = _reduced_labels(sub)
        parts.append(
            _side_by_side(
                matrix_html(sub.K_red, red_labels, red_labels, red_groups, red_groups, f"K̂<sup>({sub.name})</sup>"),
                matrix_html(sub.C_red, red_labels, red_labels, red_groups, red_groups, f"Ĉ<sup>({sub.name})</sup>"),
                matrix_html(sub.M_red, red_labels, red_labels, red_groups, red_groups, f"M̂<sup>({sub.name})</sup>"),
            )
        )
        if sub.ni and sub.n_kept:
            Pk = sub.Phi[:, : sub.n_kept]
            modal = np.diag(Pk.T @ sub.C @ Pk)
            zetas = ", ".join(
                "rigid" if w < 1e-9 else f"{c / (2 * w):.4f}" for c, w in zip(modal, sub.omegas[: sub.n_kept])
            )
            parts.append(
                f"<p><b>{sub.name}</b>: kept free-interface ζ = φ<sub>r</sub><sup>T</sup>Cφ<sub>r</sub>"
                f" / 2ω<sub>r</sub> = {zetas}.</p>"
            )
    return parts


def _assembly_steps(model: CMSModel, full: ModalResult | None) -> list[str]:
    """Steps 7 to 10, the same for every method: assemble, solve and recover, damping, back expansion."""
    system = model.system
    n = system.n
    subs = model.substructures
    bnd = {int(b) for b in model.boundary}
    groups = ["b" if d in bnd else "i" for d in range(n)]
    labels = _dof_labels(np.arange(n))
    interfaces = [int(b) for b in model.boundary[:-1]]
    parts = []
    parts.append("<h3>7. Assemble the reduced model</h3>")
    shared = _join(_dof_labels(np.array(interfaces)))
    if interfaces:
        parts.append(
            f"<p>Coordinates [q, x<sub>b</sub>]: {model.n_red} DOFs instead of {n} "
            f"({model.n_modal} modal + {model.boundary.size} boundary). The interface DOF{'s' if len(interfaces) > 1 else ''} "
            f"{shared} {'are' if len(interfaces) > 1 else 'is'} shared, so the entries of the substructures on "
            "either side add on its row and column, just like element matrices in step 1. "
            "The q of different substructures never touch.</p>"
        )
    else:
        parts.append(
            f"<p>Coordinates [q, x<sub>b</sub>]: {model.n_red} DOFs instead of {n} "
            f"({model.n_modal} modal + 1 boundary). With no cut there is only substructure A, so there is "
            "nothing to join: the reduced model is A's own reduced matrices.</p>"
        )
    glabels = [f"q<sub>{l[2:]}</sub>" if l.startswith("q_") else l for l in model.labels]
    ggroups = ["q" if l.startswith("q_") else "b" for l in model.labels]
    for j in interfaces:
        g = model.labels.index(f"x{j + 1}")
        pieces = []
        for attr, sym, total_mat in (("K_red", "K̂", model.K), ("C_red", "Ĉ", model.C), ("M_red", "M̂", model.M)):
            terms = []
            for sub in subs:
                if j in sub.boundary:
                    loc = sub.n_kept + int(np.searchsorted(sub.boundary, j))
                    terms.append(fmt(getattr(sub, attr)[loc, loc], 1) + f" ({sub.name})")
            total = total_mat[g, g]
            pieces.append(f"{sym}<sub>x{j + 1},x{j + 1}</sub> = {' + '.join(terms)} = {fmt(total, 1)}")
        parts.append(f"<p>At the interface: {'; '.join(pieces)}.</p>")
    parts.append(
        _side_by_side(
            matrix_html(model.K, glabels, glabels, ggroups, ggroups, "K̂"),
            matrix_html(model.C, glabels, glabels, ggroups, ggroups, "Ĉ"),
            matrix_html(model.M, glabels, glabels, ggroups, ggroups, "M̂"),
        )
    )

    parts.append("<h3>8. Solve and recover</h3>")
    parts.append(
        "<p>K̂η = ω²M̂η gives the reduced model's modes; the physical shapes are x = Tη, with "
        "the global T below (rows: physical DOFs, columns: reduced coordinates). The comparison "
        "table and the Modes &amp; FRF tab compare them with the full model.</p>"
    )
    if model.method == "macneal":
        parts.append(
            "<p>MacNeal's M̂ is zero on the boundary rows, so the boundary DOFs have no inertia: "
            "they follow the modes statically. They are condensed out first, "
            "K̂<sub>c</sub> = K̂<sub>qq</sub> − K̂<sub>qb</sub>K̂<sub>bb</sub><sup>−1</sup>K̂<sub>bq</sub>, "
            f"which leaves {model.omegas.size} modes, and recovered as "
            "x<sub>b</sub> = −K̂<sub>bb</sub><sup>−1</sup>K̂<sub>bq</sub>q.</p>"
        )
    parts.append(matrix_html(model.T, labels, glabels, groups, ggroups, "T"))
    freqs = ", ".join(f"{f:.4g}" for f in model.fn_hz)
    parts.append(f"<p>Reduced-model natural frequencies: {freqs} Hz.</p>")
    parts.append(_damped_comparison(model, full))
    parts.append(_recovery_step(model))
    return parts


def _recovery_step(model: CMSModel) -> str:
    """Step 10: each substructure's output (OTM) and loads (LTM) transformation matrices."""
    out = [
        "<h3>10. Back expansion (data recovery)</h3>",
        "<p>The coupled solve gives r = [q, x<sub>b</sub>] only. Each substructure recovers its "
        "interior from its own q and boundary DOFs with its rows of T (the output transformation "
        "matrix, OTM), and its spring forces k<sub>e</sub>(x<sub>e</sub> − x<sub>e−1</sub>) with "
        "the loads transformation matrix (LTM). The <i>Back expansion</i> tab applies them in "
        "time.</p>",
    ]
    K = model.system.matrices()[2]
    k = model.system.stiffness
    for sub in model.substructures:
        if not sub.ni:
            out.append(f"<p><b>{sub.name}</b> has no interior masses: nothing to recover.</p>")
            continue
        cols, cgroups = _reduced_labels(sub)
        rows = _dof_labels(sub.interior)
        otm = sub.T[: sub.ni]
        tables = [matrix_html(otm, rows, cols, ["i"] * sub.ni, cgroups, f"OTM<sup>({sub.name})</sup>")]
        # Spring e joins node e-1 (or the ground) to node e; both are DOFs of this substructure.
        local = {int(d): j for j, d in enumerate(sub.dofs)}
        ltm = np.zeros((sub.elements.size, sub.T.shape[1]))
        for r, e in enumerate(sub.elements):
            ltm[r] = k[e] * (sub.T[local[int(e)]] - (sub.T[local[int(e) - 1]] if e > 0 else 0.0))
        springs = [f"k<sub>{e + 1}</sub>" for e in sub.elements]
        tables.append(matrix_html(ltm, springs, cols, ["i"] * len(springs), cgroups, f"LTM<sup>({sub.name})</sup> [N]"))
        text = f"<p><b>{sub.name}</b>: x<sub>i</sub> = OTM [q; x<sub>b</sub>], " + (
            "Φ<sub>k</sub> in the q columns and Ψ in the boundary columns. "
            "Boundary only (static) keeps just the Ψ columns."
            if not sub.free
            else "Φ<sub>ik</sub> − RΦ<sub>bk</sub> in the q columns and R in the boundary columns. "
            "Boundary only (static) uses −K<sub>ii</sub><sup>−1</sup>K<sub>ib</sub> instead, "
            "the exact static shape, shown on the right."
        )
        i, b = sub.interior, sub.boundary
        Kii = K[np.ix_(i, i)]
        if sub.free and np.linalg.cond(Kii) < 1e12:
            guyan = -np.linalg.solve(Kii, K[np.ix_(i, b)])
            bl = _dof_labels(b)
            tables.append(
                matrix_html(
                    guyan, rows, bl, ["i"] * sub.ni, ["b"] * sub.nb, "−K<sub>ii</sub><sup>−1</sup>K<sub>ib</sub>"
                )
            )
        out.append(text + "</p>")
        out.append(_side_by_side(*tables))
    return "".join(out)


def _damping_note(sub: Substructure) -> str:
    if not sub.n_kept:
        return (
            f"<p><b>{sub.name}</b> keeps no modes, so Ĉ<sup>({sub.name})</sup> is just the damping "
            "seen from the boundary (Ĉ<sub>bb</sub>).</p>"
        )
    modal, boundary = substructure_damping(sub)
    zetas = ", ".join(f"{sub.C_red[r, r] / (2 * w):.4f}" for r, w in enumerate(sub.omegas[: sub.n_kept]))
    if max(modal, boundary) < 1e-6:
        verdict = (
            "Ĉ has the same block-diagonal form as K̂: the damping in this substructure is "
            "proportional to its stiffness, so it couples nothing."
        )
    else:
        found = [
            f"{name} {v:.2f}" for name, v in (("between kept modes", modal), ("mode–boundary", boundary)) if v >= 1e-6
        ]
        verdict = (
            f"Ĉ couples what K̂ keeps apart. Largest relative coupling {', '.join(found)} "
            "(0 = none, 1 = as large as the diagonal terms)."
        )
    return f"<p><b>{sub.name}</b>: fixed-interface ζ = {zetas}. {verdict}</p>"


def _damped_comparison(model: CMSModel, full: ModalResult | None) -> str:
    """Step 9: exact damping ratios of the reduced and the full model, mode by mode."""
    out = [
        "<h3>9. Damping in the reduced model</h3>",
        "<p>The frequencies above ignore damping. The reduced model does include it: Ĉ is used "
        "for the FRF on the Modes &amp; FRF tab, and the damped eigenvalues of "
        "M̂η̈ + Ĉη̇ + K̂η = 0 give its damping ratios. Both columns are exact (state-space) "
        "values, so any difference comes from the reduction alone. The comparison table on the left shows the same two columns.</p>",
    ]
    if full is None:
        return "".join(out)
    rows = [
        "<table border='1' cellspacing='0' cellpadding='3'>"
        f"<tr><th>Mode</th><th>ζ true</th><th>ζ {model.short_name}</th><th>ζ error</th></tr>"
    ]
    for c in compare_modes(model, full):
        cells = [str(c.index), "overdamped" if c.zeta_true is None else f"{c.zeta_true:.4f}"]
        if c.fn_red is None:
            cells += [f"<span style='color:{colors.grey}'>not in model</span>", "—"]
        elif c.zeta_red is None:
            cells += ["overdamped", "—"]
        else:
            err = c.zeta_error
            color = (
                colors.foreground
                if err is None
                else colors.good
                if abs(err) < 1e-3
                else colors.fair
                if abs(err) < 0.05
                else colors.poor
            )
            cells += [
                f"{c.zeta_red:.4f}",
                "—" if err is None else f"<span style='color:{color}'>{100 * err:+.3g}%</span>",
            ]
        rows.append("<tr>" + "".join(f"<td align='right'>{x}</td>" for x in cells) + "</tr>")
    rows.append("</table>")
    out.append("".join(rows))
    out.append(
        "<p>With stiffness-proportional damping (the default), ζ<sub>r</sub> = βω<sub>r</sub>/2, "
        "so ζ CB errs exactly as much as the frequency does. With non-proportional damping "
        "(try c<sub>1</sub> = 15 on the Simulation page), the coupling terms of Ĉ matter, and "
        "truncating modes also loses the damping they carried: the higher modes' ζ can be off "
        "by much more than their frequency.</p>"
    )
    return "".join(out)
