"""A plane frame or truss built element by element: the *Frame builder* page's model.

Each node has three DOFs, (u, v, θ): x, y and a turn anticlockwise, numbered 3i, 3i + 1, 3i + 2. The elements
(`Element`, SI throughout):

* a **bar**: axial stiffness only, EA/L along it; mass ρAL, consistent ρAL/6 [[2, 1], [1, 2]] along it and across
  it (so a rigid translation in any direction carries ρAL), or lumped, half to each end;
* a **beam**: the bar's axial part plus bending, Euler–Bernoulli's Hermite cubics, or Timoshenko's (shear
  deformation and the sections' rotary inertia, through φ = 12EI/κGAL², Friedman and Kosmatka's interdependent
  interpolation: exact statically, and its matrices are Euler–Bernoulli's at φ = 0). Lumped, a beam's mass goes
  half to each end's translations (Timoshenko's rotary inertia ρIL/2 to its turns);
* a **spring** or **damper** between two nodes, along the line joining them, or from a node to a fixed anchor,
  along the line to it; a **torsion spring** or **damper** on a node's turn, to ground or to another node's;
* a **point mass** m with a rotary inertia J at a node.

Every element's local matrices k', m', c' act on its local DOFs d = T q_e, q_e its nodes' global DOFs (`Local`):
T turns the global axes to the element's (a bar's or a spring's T keeps only the displacement along it). Its share
of the global matrices is Tᵀk'T, added at its DOFs (`assemble`). A node's support fixes some of its DOFs
(`Support`); a DOF no element touches (a truss node's turn) is *unused*. Both are struck out, leaving the free DOFs.
A free DOF with no mass (a lumped Euler–Bernoulli beam's turns, or a node with only springs) is condensed out
exactly, statically, before the eigenproblem K φ = ω² M φ (`modes`). A mechanism shows as modes at 0 Hz.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy.linalg

TWO_PI = 2.0 * math.pi
DOF_NAMES = ("u", "v", "θ")
MAX_NODES = 40
RIGID = 1e-10  # K scaled to a unit diagonal: an eigenvalue below this is a mechanism (a mode at 0 Hz)


class Support(Enum):
    """Which of (u, v, θ) a node's support fixes."""

    FREE = "Free"
    PINNED = "Pinned"
    ROLLER_X = "Roller, slides along x"
    ROLLER_Y = "Roller, slides along y"
    CLAMPED = "Clamped"
    GUIDED_X = "Guided, slides along x"
    GUIDED_Y = "Guided, slides along y"
    NO_TURN = "Held from turning"

    @property
    def fixed(self) -> tuple[bool, bool, bool]:
        return _FIXED[self]


_FIXED = {
    Support.FREE: (False, False, False),
    Support.PINNED: (True, True, False),
    Support.ROLLER_X: (False, True, False),
    Support.ROLLER_Y: (True, False, False),
    Support.CLAMPED: (True, True, True),
    Support.GUIDED_X: (False, True, True),
    Support.GUIDED_Y: (True, False, True),
    Support.NO_TURN: (False, False, True),
}
SUPPORTS = tuple(Support)


class Kind(Enum):
    BAR = "Bar"
    BEAM = "Beam"
    SPRING = "Spring"
    DAMPER = "Damper"
    TORSION_SPRING = "Torsion spring"
    TORSION_DAMPER = "Torsion damper"
    MASS = "Point mass"

    @property
    def member(self) -> bool:
        return self in (Kind.BAR, Kind.BEAM)

    @property
    def link(self) -> bool:
        """A spring or damper along a line (between nodes, or to an anchor)."""
        return self in (Kind.SPRING, Kind.DAMPER)

    @property
    def torsion(self) -> bool:
        return self in (Kind.TORSION_SPRING, Kind.TORSION_DAMPER)

    @property
    def damper(self) -> bool:
        return self in (Kind.DAMPER, Kind.TORSION_DAMPER)


class Theory(Enum):
    EULER = "Euler–Bernoulli"
    TIMOSHENKO = "Timoshenko (shear, rotary inertia)"


THEORIES = tuple(Theory)


@dataclass(frozen=True)
class Material:
    name: str
    E: float  # Pa
    nu: float
    rho: float  # kg/m³

    @property
    def G(self) -> float:
        return self.E / (2 * (1 + self.nu))


STEEL = Material("Steel", 210e9, 0.3, 7850.0)
ALUMINIUM = Material("Aluminium", 70e9, 0.33, 2700.0)
TITANIUM = Material("Titanium", 114e9, 0.34, 4430.0)
MATERIALS = (STEEL, ALUMINIUM, TITANIUM)


class Shape(Enum):
    RECTANGLE = "Rectangle"
    TUBE = "Round tube"  # solid when the wall is half the diameter (or more)


SHAPES = tuple(Shape)


@dataclass(frozen=True)
class Section:
    """A rectangle `width` × `depth` (depth in the frame's plane, so it bends about the width), or a round tube of
    outer diameter `depth` and wall `wall` (solid at depth/2)."""

    shape: Shape = Shape.RECTANGLE
    depth: float = 0.05
    width: float = 0.05
    wall: float = 0.025

    @property
    def inner(self) -> float:
        """A tube's inner diameter (0 if solid)."""
        return max(self.depth - 2 * self.wall, 0.0)

    @property
    def area(self) -> float:
        if self.shape is Shape.RECTANGLE:
            return self.width * self.depth
        return math.pi / 4 * (self.depth**2 - self.inner**2)

    @property
    def inertia(self) -> float:
        if self.shape is Shape.RECTANGLE:
            return self.width * self.depth**3 / 12
        return math.pi / 64 * (self.depth**4 - self.inner**4)

    def kappa(self, nu: float) -> float:
        """The shear coefficient: Timoshenko's 5/6 for a rectangle, Cowper's for a round tube."""
        if self.shape is Shape.RECTANGLE:
            return 5.0 / 6.0
        m2 = (self.inner / self.depth) ** 2
        return 6 * (1 + nu) * (1 + m2) ** 2 / ((7 + 6 * nu) * (1 + m2) ** 2 + (20 + 12 * nu) * m2)


SQUARE_50 = Section()  # 50 × 50 mm


@dataclass(frozen=True)
class Node:
    x: float
    y: float
    support: Support = Support.FREE


@dataclass(frozen=True)
class Element:
    """One element; which fields count depends on its kind.

    A member (bar or beam) joins `i` and `j` and has a `material` and a `section`. A spring or damper joins `i` to
    `j`, or with j None to `anchor`, a fixed point; `value` is its k (N/m) or c (N·s/m). A torsion spring or
    damper turns `i` against `j`'s turn, or against the ground with j None; `value` in N·m/rad or N·m·s/rad. A point
    mass sits at `i`: `value` is its m (kg), `inertia` its J (kg·m²)."""

    kind: Kind
    i: int
    j: int | None = None
    material: Material = STEEL
    section: Section = SQUARE_50
    value: float = 0.0
    inertia: float = 0.0
    anchor: tuple[float, float] | None = None


@dataclass(frozen=True)
class Frame:
    nodes: tuple[Node, ...] = ()
    elements: tuple[Element, ...] = ()
    lumped: bool = False
    theory: Theory = Theory.EULER

    @property
    def n_dof(self) -> int:
        return 3 * len(self.nodes)

    def point(self, i: int) -> np.ndarray:
        n = self.nodes[i]
        return np.array([n.x, n.y])

    def length(self, e: Element) -> float:
        """A member's or a line spring's length (to its anchor, if grounded)."""
        a = self.point(e.i)
        b = self.point(e.j) if e.j is not None else np.array(e.anchor if e.anchor is not None else a)
        return float(np.hypot(*(b - a)))

    def direction(self, e: Element) -> np.ndarray:
        """The unit vector from i toward j (or the anchor)."""
        a = self.point(e.i)
        b = self.point(e.j) if e.j is not None else np.array(e.anchor if e.anchor is not None else a)
        d = b - a
        return d / float(np.hypot(*d))

    def extent(self) -> tuple[float, float, float, float]:
        """(x0, x1, y0, y1) round the nodes and anchors."""
        pts = [(n.x, n.y) for n in self.nodes] + [e.anchor for e in self.elements if e.anchor is not None]
        if not pts:
            return 0.0, 0.0, 0.0, 0.0
        xs, ys = zip(*pts)
        return min(xs), max(xs), min(ys), max(ys)

    # ---------------------------------------------------------------- editing (each gives a new frame)
    def with_node(self, x: float, y: float, support: Support = Support.FREE) -> Frame:
        return dataclasses.replace(self, nodes=self.nodes + (Node(x, y, support),))

    def with_element(self, e: Element) -> Frame:
        return dataclasses.replace(self, elements=self.elements + (e,))

    def replace_node(self, i: int, node: Node) -> Frame:
        nodes = list(self.nodes)
        nodes[i] = node
        return dataclasses.replace(self, nodes=tuple(nodes))

    def replace_element(self, k: int, e: Element) -> Frame:
        elements = list(self.elements)
        elements[k] = e
        return dataclasses.replace(self, elements=tuple(elements))

    def without_element(self, k: int) -> Frame:
        return dataclasses.replace(self, elements=self.elements[:k] + self.elements[k + 1 :])

    def without_node(self, i: int) -> Frame:
        """The node gone, with every element on it, the later nodes renumbered."""

        def shift(n: int) -> int:
            return n if n < i else n - 1

        kept = tuple(
            dataclasses.replace(e, i=shift(e.i), j=None if e.j is None else shift(e.j))
            for e in self.elements
            if i not in (e.i, e.j)
        )
        return dataclasses.replace(self, nodes=self.nodes[:i] + self.nodes[i + 1 :], elements=kept)

    def node_at(self, x: float, y: float, tolerance: float = 1e-9) -> int | None:
        for k, n in enumerate(self.nodes):
            if abs(n.x - x) <= tolerance and abs(n.y - y) <= tolerance:
                return k
        return None

    def divide(self, k: int, parts: int) -> Frame:
        """Member k cut into `parts` equal members, the new nodes between free."""
        e = self.elements[k]
        assert e.kind.member and e.j is not None and parts >= 1
        if parts == 1:
            return self
        a, b = self.point(e.i), self.point(e.j)
        frame = self.without_element(k)
        ends = [e.i]
        for s in range(1, parts):
            x, y = a + (b - a) * s / parts
            frame = frame.with_node(float(x), float(y))
            ends.append(len(frame.nodes) - 1)
        ends.append(e.j)
        for p, q in itertools.pairwise(ends):
            frame = frame.with_element(dataclasses.replace(e, i=p, j=q))
        return frame


def dof_label(dof: int) -> str:
    """'u3' for node 3's x DOF (nodes counted from 1)."""
    return f"{DOF_NAMES[dof % 3]}{dof // 3 + 1}"


# -------------------------------------------------------------------- element matrices
@dataclass(frozen=True)
class Local:
    """One element's matrices in its own axes, and how they reach the global ones: d = T q[dofs]."""

    dofs: tuple[int, ...]  # the global DOFs it touches
    local: tuple[str, ...]  # its local DOFs' names
    T: np.ndarray  # (len(local), len(dofs))
    k: np.ndarray
    m: np.ndarray
    c: np.ndarray
    length: float = 0.0
    angle: float = 0.0  # rad, the element's axis from x
    phi: float = 0.0  # Timoshenko's shear parameter 12EI/κGAL²
    k_axial: np.ndarray | None = None  # a beam's stiffness split: axial, the rest bending
    k_bending: np.ndarray | None = None

    def stiffness(self) -> np.ndarray:
        return self.T.T @ self.k @ self.T

    def mass(self) -> np.ndarray:
        return self.T.T @ self.m @ self.T

    def damping(self) -> np.ndarray:
        return self.T.T @ self.c @ self.T


def _rotation(c: float, s: float) -> np.ndarray:
    """Global (u, v, θ) at both ends to local (axial, transverse, θ)."""
    r = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return scipy.linalg.block_diag(r, r)


def beam_matrices(
    E: float, G: float, rho: float, A: float, I: float, kappa: float, L: float, theory: Theory, lumped: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """A plane beam's (k', m', k_axial, φ) in its own axes (u1, v1, θ1, u2, v2, θ2)."""
    phi = 12 * E * I / (kappa * G * A * L * L) if theory is Theory.TIMOSHENKO else 0.0
    axial = np.zeros((6, 6))
    ea = E * A / L
    axial[np.ix_([0, 3], [0, 3])] = ea * np.array([[1.0, -1.0], [-1.0, 1.0]])
    b = E * I / ((1 + phi) * L**3)
    bend = b * np.array(
        [
            [12, 6 * L, -12, 6 * L],
            [6 * L, (4 + phi) * L * L, -6 * L, (2 - phi) * L * L],
            [-12, -6 * L, 12, -6 * L],
            [6 * L, (2 - phi) * L * L, -6 * L, (4 + phi) * L * L],
        ]
    )
    k = axial.copy()
    vt = [1, 2, 4, 5]
    k[np.ix_(vt, vt)] += bend
    m = np.zeros((6, 6))
    total = rho * A * L
    if lumped:
        for n in (0, 1, 3, 4):
            m[n, n] = total / 2
        if theory is Theory.TIMOSHENKO:
            m[2, 2] = m[5, 5] = rho * I * L / 2
        return k, m, axial, phi
    m[np.ix_([0, 3], [0, 3])] = total / 6 * np.array([[2.0, 1.0], [1.0, 2.0]])
    p = phi
    a = 13 / 35 + 7 * p / 10 + p * p / 3
    bb = (11 / 210 + 11 * p / 120 + p * p / 24) * L
    c = 9 / 70 + 3 * p / 10 + p * p / 6
    d = (13 / 420 + 3 * p / 40 + p * p / 24) * L
    e = (1 / 105 + p / 60 + p * p / 120) * L * L
    f = (1 / 140 + p / 60 + p * p / 120) * L * L
    trans = total / (1 + p) ** 2 * np.array([[a, bb, c, -d], [bb, e, d, -f], [c, d, a, -bb], [-d, -f, -bb, e]])
    m[np.ix_(vt, vt)] += trans
    if theory is Theory.TIMOSHENKO:
        g = (1 / 10 - p / 2) * L
        h = (2 / 15 + p / 6 + p * p / 3) * L * L
        j = (1 / 30 + p / 6 - p * p / 6) * L * L
        rot = (
            rho
            * I
            / ((1 + p) ** 2 * L)
            * np.array([[6 / 5, g, -6 / 5, g], [g, h, -g, -j], [-6 / 5, -g, 6 / 5, -g], [g, -j, -g, h]])
        )
        m[np.ix_(vt, vt)] += rot
    return k, m, axial, phi


def local(frame: Frame, k: int) -> Local:
    """Element k's matrices in its own axes, its T and its DOFs."""
    e = frame.elements[k]
    z1 = np.zeros((1, 1))
    if e.kind is Kind.MASS:
        dofs = tuple(range(3 * e.i, 3 * e.i + 3))
        return Local(
            dofs, ("u", "v", "θ"), np.eye(3), np.zeros((3, 3)), np.diag([e.value, e.value, e.inertia]), np.zeros((3, 3))
        )
    if e.kind.torsion:
        one = np.ones((1, 1)) * e.value
        if e.j is None:
            dofs, T = (3 * e.i + 2,), np.eye(1)
            names: tuple[str, ...] = ("θ",)
        else:  # the twist θ_j − θ_i
            dofs, T, names = (3 * e.i + 2, 3 * e.j + 2), np.array([[-1.0, 1.0]]), ("θj − θi",)
        k_, c_ = (z1, one) if e.kind.damper else (one, z1)
        return Local(dofs, names, T, k_, z1, c_)
    L = frame.length(e)
    c, s = frame.direction(e)
    angle = math.atan2(s, c)
    if e.kind.link:
        one = np.ones((1, 1)) * e.value
        k_, c_ = (z1, one) if e.kind.damper else (one, z1)
        if e.j is None:  # stretched as node i moves away from the anchor: −(c u + s v)
            dofs, T, names = (3 * e.i, 3 * e.i + 1), np.array([[-c, -s]]), ("δ",)
        else:  # the stretch along i → j
            dofs = (3 * e.i, 3 * e.i + 1, 3 * e.j, 3 * e.j + 1)
            T, names = np.array([[-c, -s, c, s]]), ("δ",)
        return Local(dofs, names, T, k_, z1, c_, L, angle)
    assert e.j is not None
    mat, sec = e.material, e.section
    A, I = sec.area, sec.inertia
    if e.kind is Kind.BAR:
        dofs = (3 * e.i, 3 * e.i + 1, 3 * e.j, 3 * e.j + 1)
        r = np.array([[c, s], [-s, c]])
        T = scipy.linalg.block_diag(r, r)
        kb = np.zeros((4, 4))
        kb[np.ix_([0, 2], [0, 2])] = mat.E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
        total = mat.rho * A * L
        if frame.lumped:
            mb = total / 2 * np.eye(4)
        else:  # ρAL/6 [[2, 1], [1, 2]] along and across
            mb = total / 6 * np.kron(np.array([[2.0, 1.0], [1.0, 2.0]]), np.eye(2))
        return Local(dofs, ("a1", "t1", "a2", "t2"), T, kb, mb, np.zeros((4, 4)), L, angle, k_axial=kb)
    dofs = tuple(range(3 * e.i, 3 * e.i + 3)) + tuple(range(3 * e.j, 3 * e.j + 3))
    kl, ml, axial, phi = beam_matrices(mat.E, mat.G, mat.rho, A, I, sec.kappa(mat.nu), L, frame.theory, frame.lumped)
    return Local(
        dofs,
        ("a1", "t1", "θ1", "a2", "t2", "θ2"),
        _rotation(c, s),
        kl,
        ml,
        np.zeros((6, 6)),
        L,
        angle,
        phi,
        k_axial=axial,
        k_bending=kl - axial,
    )


# -------------------------------------------------------------------- assembly
@dataclass(frozen=True)
class Assembly:
    """The global M, C, K over every DOF, and which DOFs are fixed, unused and free."""

    M: np.ndarray
    C: np.ndarray
    K: np.ndarray
    fixed: np.ndarray  # bool per DOF: a support holds it
    unused: np.ndarray  # bool per DOF: no element touches it
    locals: tuple[Local, ...]

    @property
    def free(self) -> np.ndarray:
        return ~(self.fixed | self.unused)

    @property
    def free_dofs(self) -> np.ndarray:
        return np.flatnonzero(self.free)

    def reduced(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """M, C, K over the free DOFs."""
        f = self.free_dofs
        return self.M[np.ix_(f, f)], self.C[np.ix_(f, f)], self.K[np.ix_(f, f)]

    @property
    def massless(self) -> np.ndarray:
        """bool per DOF: free, but with no mass (condensed out)."""
        return self.free & ~np.any(self.M != 0, axis=1)


def assemble(frame: Frame) -> Assembly:
    n = frame.n_dof
    M, C, K = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    touched = np.zeros(n, dtype=bool)
    parts = []
    for k in range(len(frame.elements)):
        part = local(frame, k)
        idx = np.ix_(part.dofs, part.dofs)
        K[idx] += part.stiffness()
        M[idx] += part.mass()
        C[idx] += part.damping()
        touched[list(part.dofs)] = True
        parts.append(part)
    fixed = np.array([f for node in frame.nodes for f in node.support.fixed], dtype=bool)
    if frame.nodes:  # a DOF the elements leave alone has nothing to vibrate
        unused = ~touched | (~np.any(K != 0, axis=1) & ~np.any(M != 0, axis=1) & ~np.any(C != 0, axis=1))
    else:
        unused = np.zeros(0, dtype=bool)
    return Assembly(M, C, K, fixed, unused & ~fixed, tuple(parts))


class FrameError(ValueError):
    """The model can't be solved (said why, in words for the page)."""


LOOSE = "A massless node can move with nothing to stop it: add a mass, an element or a support."


@dataclass(frozen=True)
class Modes:
    hz: np.ndarray  # ascending; mechanisms at 0
    shapes: np.ndarray  # (n_dof, n_modes), every DOF (0 where fixed or unused), mass-normalized
    energy: np.ndarray  # (3, n_modes): each mode's strain energy share in axial, bending, springs
    mechanisms: int
    condensed: int  # massless DOFs condensed out


def condense(K: np.ndarray, M: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Guyan's reduction onto `keep` (bool): exact for DOFs with no mass. (K_r, M_r, R), q = R q_keep."""
    drop = ~keep
    Kss = K[np.ix_(drop, drop)]
    try:
        G = -np.linalg.solve(Kss, K[np.ix_(drop, keep)])
    except np.linalg.LinAlgError:
        raise FrameError(LOOSE) from None
    if not np.all(np.isfinite(G)) or np.linalg.cond(Kss) > 1e14:
        raise FrameError(LOOSE)
    R = np.zeros((len(keep), int(keep.sum())))
    R[keep] = np.eye(int(keep.sum()))
    R[drop] = G
    return R.T @ K @ R, R.T @ M @ R, R


def modes(frame: Frame, assembly: Assembly | None = None) -> Modes:
    """The undamped modes, K φ = ω² M φ over the free DOFs, massless ones condensed out."""
    a = assembly if assembly is not None else assemble(frame)
    free = a.free_dofs
    if free.size == 0:
        raise FrameError("No free DOFs: add nodes and elements, or free a support.")
    _, _, K = a.reduced()
    M = a.M[np.ix_(free, free)]
    has_mass = np.any(M != 0, axis=1)
    if not has_mass.any():
        raise FrameError("Nothing has mass: add a bar, a beam or a point mass.")
    if has_mass.all():
        Kr, Mr, R = K, M, np.eye(free.size)
    else:
        Kr, Mr, R = condense(K, M, has_mass)
    Kr, Mr = 0.5 * (Kr + Kr.T), 0.5 * (Mr + Mr.T)
    w2, X = scipy.linalg.eigh(Kr, Mr)
    rigid = np.arange(w2.size) < mechanisms(Kr)
    w2 = np.where(rigid, 0.0, np.maximum(w2, 0.0))
    shapes = np.zeros((frame.n_dof, w2.size))
    shapes[free] = R @ X
    # each shape's sign: its largest entry positive, so redraws don't flip it
    big = np.argmax(np.abs(shapes), axis=0)
    shapes *= np.where(shapes[big, np.arange(w2.size)] < 0, -1.0, 1.0)
    return Modes(
        np.sqrt(w2) / TWO_PI,
        shapes,
        strain_shares(a, shapes),
        int(rigid.sum()),
        int((~has_mass).sum()),
    )


def mechanisms(K: np.ndarray) -> int:
    """How many independent ways the structure moves with no strain: K's null space, found with K scaled to a
    unit diagonal (a slender beam's bending is many orders softer than its stretch, so ω² alone can't tell)."""
    d = np.diag(K).copy()
    loose = d <= 0
    scale = 1.0 / np.sqrt(np.where(loose, 1.0, d))
    Ks = K * np.outer(scale, scale)
    Ks[loose, :] = Ks[:, loose] = 0.0
    return int(np.sum(np.linalg.eigvalsh(0.5 * (Ks + Ks.T)) < RIGID))


def strain_shares(a: Assembly, shapes: np.ndarray) -> np.ndarray:
    """(3, n_modes): each mode's strain energy split among the members' axial stretch, their bending (with
    shear), and the springs. A mechanism's column is 0."""
    out = np.zeros((3, shapes.shape[1]))
    for part in a.locals:
        d = part.T @ shapes[list(part.dofs)]  # local displacements, one column per mode
        if part.k_axial is not None:
            out[0] += np.einsum("im,ij,jm->m", d, part.k_axial, d)
            if part.k_bending is not None:
                out[1] += np.einsum("im,ij,jm->m", d, part.k_bending, d)
        else:
            out[2] += np.einsum("im,ij,jm->m", d, part.k, d)
    total = out.sum(axis=0)
    return np.divide(out, total, out=np.zeros_like(out), where=total > 0)


def member_curve(frame: Frame, k: int, q: np.ndarray, points: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Member k's displaced line under q (every DOF): x, y at `points` stations. A beam bends by its Hermite
    cubics (Euler–Bernoulli's: Timoshenko's shear adds a linear part to the deflection the ends already give),
    a bar stays straight."""
    e = frame.elements[k]
    assert e.j is not None
    a, b = frame.point(e.i), frame.point(e.j)
    s = np.linspace(0.0, 1.0, points)
    straight = a + np.outer(s, b - a)
    ua, ub = q[3 * e.i : 3 * e.i + 2], q[3 * e.j : 3 * e.j + 2]
    moved = straight + np.outer(1 - s, ua) + np.outer(s, ub)
    if e.kind is Kind.BEAM:
        L = frame.length(e)
        c, sn = frame.direction(e)
        across = np.array([-sn, c])
        t1, t2 = q[3 * e.i + 2], q[3 * e.j + 2]
        # the bending beyond the chord: the slopes' Hermite terms less the chord's rotation (which the straight
        # interpolation already has)
        chord = ((ub - ua) @ across) / L
        h2 = L * (s - 2 * s * s + s**3)
        h4 = L * (-s * s + s**3)
        moved = moved + np.outer(h2 * (t1 - chord) + h4 * (t2 - chord), across)
    return moved[:, 0], moved[:, 1]


# -------------------------------------------------------------------- presets
COLUMN = Section(depth=0.02, width=0.05)
FLOOR = Section(depth=0.1, width=0.02)


def cantilever(length: float = 1.0, parts: int = 4, section: Section = SQUARE_50, material: Material = STEEL) -> Frame:
    """A horizontal beam clamped at its left end, cut into `parts` members."""
    nodes = tuple(Node(length * i / parts, 0.0, Support.CLAMPED if i == 0 else Support.FREE) for i in range(parts + 1))
    elements = tuple(Element(Kind.BEAM, i, i + 1, material, section) for i in range(parts))
    return Frame(nodes, elements)


def simply_supported(
    length: float = 1.0, parts: int = 4, section: Section = SQUARE_50, material: Material = STEEL
) -> Frame:
    """A horizontal beam, pinned at its left end and on a roller at its right, cut into `parts` members."""

    def support(i: int) -> Support:
        return Support.PINNED if i == 0 else Support.ROLLER_X if i == parts else Support.FREE

    nodes = tuple(Node(length * i / parts, 0.0, support(i)) for i in range(parts + 1))
    elements = tuple(Element(Kind.BEAM, i, i + 1, material, section) for i in range(parts))
    return Frame(nodes, elements)


def portal(
    span: float = 2.0,
    height: float = 1.5,
    parts: int = 1,
    column: Section = SQUARE_50,
    girder: Section = SQUARE_50,
    base: Support = Support.CLAMPED,
) -> Frame:
    """Two columns and a girder, each cut into `parts` members; the bases `base`. The nodes are numbered along
    the frame, up the left column, across and down (so K is banded): the corners are 0, parts, 2 parts and
    3 parts."""
    corners = np.array([[0.0, 0.0], [0.0, height], [span, height], [span, 0.0]])
    points = [corners[0]] + [
        a + (b - a) * s / parts for a, b in itertools.pairwise(corners) for s in range(1, parts + 1)
    ]
    last = len(points) - 1
    nodes = tuple(Node(float(x), float(y), base if k in (0, last) else Support.FREE) for k, (x, y) in enumerate(points))
    elements = tuple(
        Element(Kind.BEAM, k, k + 1, section=girder if parts <= k < 2 * parts else column) for k in range(last)
    )
    return Frame(nodes, elements)


def warren(bays: int = 4, bay: float = 1.0, height: float = 0.8, section: Section = SQUARE_50) -> Frame:
    """A Warren truss of bars: `bays` along the bottom chord, pinned at the left, a roller at the right."""
    bottom = [Node(bay * i, 0.0) for i in range(bays + 1)]
    bottom[0] = Node(0.0, 0.0, Support.PINNED)
    bottom[-1] = Node(bay * bays, 0.0, Support.ROLLER_X)
    top = [Node(bay * (i + 0.5), height) for i in range(bays)]
    nodes = tuple(bottom + top)
    elements = []
    n = bays + 1
    for i in range(bays):
        elements.append(Element(Kind.BAR, i, i + 1, section=section))  # bottom chord
        elements.append(Element(Kind.BAR, i, n + i, section=section))  # up diagonal
        elements.append(Element(Kind.BAR, n + i, i + 1, section=section))  # down diagonal
        if i + 1 < bays:
            elements.append(Element(Kind.BAR, n + i, n + i + 1, section=section))  # top chord
    return Frame(nodes, tuple(elements))


def storeys(
    count: int = 2,
    span: float = 1.5,
    height: float = 1.0,
    column: Section = COLUMN,
    floor: Section = FLOOR,
    floor_mass: float = 100.0,
    parts: int = 2,
) -> Frame:
    """A frame `count` storeys high: clamped columns (each cut into `parts` members), a deep floor beam at each
    level, and `floor_mass` on each floor, half at either end."""
    frame = Frame((Node(0.0, 0.0, Support.CLAMPED), Node(span, 0.0, Support.CLAMPED)))
    below = (0, 1)
    for level in range(1, count + 1):
        frame = frame.with_node(0.0, height * level).with_node(span, height * level)
        left, right = len(frame.nodes) - 2, len(frame.nodes) - 1
        for end, top in zip(below, (left, right)):
            frame = frame.with_element(Element(Kind.BEAM, end, top, section=column))
            frame = frame.divide(len(frame.elements) - 1, parts)
        frame = frame.with_element(Element(Kind.BEAM, left, right, section=floor))
        for top in (left, right):
            frame = frame.with_element(Element(Kind.MASS, top, value=floor_mass / 2))
        below = (left, right)
    return frame
