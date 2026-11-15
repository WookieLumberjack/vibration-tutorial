"""The equations of motion at one instant, written out with numbers (the *Gyroscope* and *3D rigid body* pages).

`snapshot_html` takes a `core.rotation.Snapshot` and writes, in the axes picked: the rotation matrices that place
the body (B, its axes in space; in a frame of its own also A, the frame's, and AᵀB, the body in it), the inertia
tensor I = (AᵀB) I_body (AᵀB)ᵀ, L = Iω, and the transport theorem dL/dt + Ω × L = τ (Euler's equations in axes
fixed in the body, dL/dt = τ in space), each vector and matrix as a bracketed table of numbers and how closely
the two sides agree. Each bracket has its own power of ten, so a 10⁻⁴ kg·m² inertia reads as 2.083. The page
says in words where its axes are and, for a frame of its own, what A and AᵀB are and how the frame turns.

Working on it: `tests/test_gyroscope.py`'s `test_a_snapshot_of_the_equations` and `tests/test_rotation.py` check
the numbers, and `tests/test_gui_smoke.py`'s `test_gyroscope_snapshot` and the rigid body page's test the tabs.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.rotation import Snapshot, Turning

SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


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
    """A vector (a column) or a matrix in square brackets, with its power of ten in front if it needs one. A
    matrix's non-zero off-diagonal entries are bold when asked (the products of inertia)."""
    A = np.atleast_2d(np.asarray(values, dtype=float))
    if A.shape[0] == 1:
        A = A.T
    power = _power(A)
    scale = 10.0**power
    top = float(np.abs(A).max())
    n = A.shape[0]
    left = "⎡" + "⎢" * (n - 2) + "⎣"
    right = "⎤" + "⎥" * (n - 2) + "⎦"
    rows = []
    for i in range(n):
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


def snapshot_html(s: Snapshot, where: str, frame_note: str = "", turning_note: str = "") -> str:
    """The equations at `s`'s instant, in its axes, `where` saying which ("in fixed axes (x, y, z up)"). For a
    frame that isn't space's or the body's own, `frame_note` says what A and AᵀB are; for one that turns its own
    way, `turning_note` how (see the module's docstring)."""
    out = [f"<p><b>At t = {s.t:.3f} s</b>, {where}:</p>"]
    own = s.turning is Turning.FIXED or np.allclose(s.frame, s.orientation)
    if own:
        out += [
            "<p>Where the body is: B, its axes in space, as columns.</p>",
            equation("B =", bracket(s.orientation)),
        ]
        if s.turning is Turning.FIXED:
            inertia = "I = B I<sub>body</sub> Bᵀ ="
        else:
            inertia = "I = I<sub>body</sub> ="
    else:
        out += [
            f"<p>Where the body is: A, the frame's axes in space (its columns), and AᵀB, the body's axes in the "
            f"frame's. {frame_note} B = A·(AᵀB) is the body's axes in space.</p>",
            equation("A =", bracket(s.frame), "AᵀB =", bracket(s.turn)),
        ]
        inertia = "I = (AᵀB) I<sub>body</sub> (AᵀB)ᵀ ="
    diagonal = np.allclose(s.body_inertia, np.diag(np.diag(s.body_inertia)))
    source = _diag(s.body_inertia) if diagonal else "the tensor in the body's own axes (the Inertia tab)"
    out += [
        f"<p>The inertia tensor, kg·m² (products of inertia in bold), from I<sub>body</sub> = {source}:</p>",
        equation(inertia, bracket(s.inertia, bold_off_diagonal=True)),
        "<p>L = Iω: angular momentum (N·m·s) from the angular velocity (rad/s).</p>",
        equation("L =", bracket(s.momentum), "=", bracket(s.inertia, bold_off_diagonal=True), bracket(s.omega)),
    ]
    if s.turning is Turning.FIXED:
        out += [
            "<p>dL/dt = τ (N·m): fixed axes don't turn (Ω = 0), so L's components change only by the torque.</p>",
            equation("dL/dt", bracket(s.rate), "=", "τ", bracket(s.torque)),
        ]
    else:
        name = "ω" if s.turning is Turning.BODY else "Ω"
        lead = (
            "Euler's equations: the axes turn with the body (Ω = ω), and I is constant in them, so dL/dt = Iω̇."
            if s.turning is Turning.BODY
            else f"The axes turn at Ω ({turning_note}), so L's components change by the torque less Ω × L."
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
