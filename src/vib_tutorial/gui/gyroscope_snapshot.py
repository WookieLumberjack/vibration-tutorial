"""The Gyroscope page's *Components* tab snapshot: the equations of motion at one instant, written out with numbers.

`snapshot_html` takes a `core.gyroscope.Snapshot` and writes, in the axes picked: the rotation matrices that place
the body (B, its axes in space; in the axis frame also A, the frame's, and AᵀB, the turn by ψ about e₃ between
them), the inertia tensor I = (AᵀB) I_body (AᵀB)ᵀ, L = Iω, and the transport theorem dL/dt + Ω × L = τ (Euler's
equations in the body's axes, dL/dt = τ in space), each vector and matrix as a bracketed table of numbers and how
closely the two sides agree. Each bracket has its own power of ten, so a 10⁻⁴ kg·m² inertia reads as 2.083.

Working on it: `tests/test_gyroscope.py`'s `test_a_snapshot_of_the_equations` checks the numbers, and
`tests/test_gui_smoke.py`'s `test_gyroscope_snapshot` the tab.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.gyroscope import Frame, Snapshot

SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
BRACKETS = {3: ("⎡⎢⎣", "⎤⎥⎦")}  # rows: top, middle, bottom
WHERE = {
    Frame.SPACE: "in fixed axes (x, y, z up)",
    Frame.BODY: "in the body's axes (e₁ the mark, e₃ the axis)",
    Frame.AXIS: "in the axis frame (e₃ the axis, e₁′ horizontal across it)",
}


def _power(values: np.ndarray) -> int:
    """The power of ten a bracket is written in: 0 for numbers between 0.01 and 10 000."""
    top = float(np.abs(values).max())
    if top == 0.0 or 1e-2 <= top < 1e4:
        return 0
    return math.floor(math.log10(top))


def _entry(value: float, scale: float, top: float) -> str:
    if abs(value) <= 1e-9 * top:
        return "0"
    v = value / scale
    digits = max(0, 3 - math.floor(math.log10(max(top / scale, 1e-300))))
    return f"{v:.{digits}f}".replace("-", "−")


def bracket(values: np.ndarray, bold_off_diagonal: bool = False) -> str:
    """A 3-vector (a column) or a 3 × 3 matrix in square brackets, with its power of ten in front if it needs one.
    A matrix's non-zero off-diagonal entries are bold when asked (the products of inertia)."""
    A = np.atleast_2d(np.asarray(values, dtype=float))
    if A.shape[0] == 1:
        A = A.T
    power = _power(A)
    scale = 10.0**power
    top = float(np.abs(A).max())
    left, right = BRACKETS[3]
    rows = []
    for i in range(3):
        cells = []
        for j in range(A.shape[1]):
            text = _entry(A[i, j], scale, top)
            if bold_off_diagonal and i != j and text != "0":
                text = f"<b>{text}</b>"
            cells.append(f"<td align='right'>&nbsp;{text}&nbsp;</td>")
        rows.append(f"<tr><td>{left[i]}</td>{''.join(cells)}<td>{right[i]}</td></tr>")
    table = f"<table cellspacing='0' cellpadding='0'>{''.join(rows)}</table>"
    if power == 0:
        return table
    factor = f"10{str(power).translate(SUPERSCRIPT)}&nbsp;×"
    return f"<table cellspacing='0' cellpadding='0'><tr><td valign='middle'>{factor}</td><td>{table}</td></tr></table>"


def _diag(I: np.ndarray) -> str:
    """diag(I₁, I₂, I₃) in one power of ten: 10⁻⁴ × diag(6.5, 6.5, 4)."""
    moments = np.diag(I)
    power = _power(moments)
    text = f"diag({', '.join(f'{v / 10.0**power:.4g}' for v in moments)})"
    return text if power == 0 else f"10{str(power).translate(SUPERSCRIPT)} × {text}"


def equation(*parts: str) -> str:
    """Brackets and the words between them, side by side, centred on one line."""
    cells = "".join(
        f"<td valign='middle'>{p}</td>" if p.startswith("<table") else f"<td valign='middle'>&nbsp;{p}&nbsp;</td>"
        for p in parts
    )
    return f"<table cellspacing='0' cellpadding='0'><tr>{cells}</tr></table>"


def snapshot_html(s: Snapshot) -> str:
    """The equations at `s`'s instant, in its axes (see the module's docstring)."""
    frame = s.kind
    out = [f"<p><b>At t = {s.t:.3f} s</b>, {WHERE[frame]}:</p>"]
    if frame is Frame.AXIS:
        out += [
            "<p>Where the body is: A, the axis frame's axes in space (its columns), and AᵀB, the body's axes in "
            "the frame's, a turn by ψ about e₃. B = A·(AᵀB) is the body's axes in space.</p>",
            equation("A =", bracket(s.frame), "AᵀB =", bracket(s.turn)),
        ]
        inertia = "I = (AᵀB) I<sub>body</sub> (AᵀB)ᵀ ="
    else:
        out += [
            "<p>Where the body is: B, its axes e₁, e₂, e₃ in space, as columns.</p>",
            equation("B =", bracket(s.orientation)),
        ]
        inertia = "I = B I<sub>body</sub> Bᵀ =" if frame is Frame.SPACE else "I = I<sub>body</sub> ="
    out += [
        f"<p>The inertia tensor, kg·m² (products of inertia in bold), from I<sub>body</sub> = {_diag(s.body_inertia)}:"
        "</p>",
        equation(inertia, bracket(s.inertia, bold_off_diagonal=True)),
        "<p>L = Iω: angular momentum (N·m·s) from the angular velocity (rad/s).</p>",
        equation("L =", bracket(s.momentum), "=", bracket(s.inertia, bold_off_diagonal=True), bracket(s.omega)),
    ]
    if frame is Frame.SPACE:
        out += [
            "<p>dL/dt = τ (N·m): fixed axes don't turn (Ω = 0), so L's components change only by the torque.</p>",
            equation("dL/dt", bracket(s.rate), "=", "τ", bracket(s.torque)),
        ]
    else:
        name = "ω" if frame is Frame.BODY else "Ω"
        lead = (
            "Euler's equations: the axes turn with the body (Ω = ω), and I is constant in them, so dL/dt = Iω̇."
            if frame is Frame.BODY
            else "The axes turn at Ω (precession and nodding, not the spin), so L's components change by the "
            "torque less Ω × L."
        )
        out += [
            f"<p>{lead} dL/dt + {name} × L = τ (N·m), with {name} = "
            f"({', '.join(f'{v:.4g}' for v in s.frame_rate)}) rad/s:</p>".replace("-", "−"),
            equation("dL/dt", bracket(s.rate), "+", f"{name} × L", bracket(s.carried), "=", "τ", bracket(s.torque)),
        ]
    out.append(
        f"<p>The two sides agree to {100 * s.mismatch:.2g} % (dL/dt is from the samples either side: "
        "the rest is the simulation's step error).</p>"
    )
    return "".join(out)
