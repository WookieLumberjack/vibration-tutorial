"""A rotor built part by part: the *Rotor builder* page's model.

The shaft runs along z from 0, a row of `Section`s, each a round tube (solid when its bore is 0) of a material
from the library (`frame.MATERIALS`: steel, aluminium, titanium), cut into beam elements. On it sit, each at an
axial position z:

* **discs** (`Disc`): rigid, a mass m, a polar moment of inertia I_p about the spin axis and a diametral I_d
  about a diameter, from the library (`DISC_LIBRARY`, by material and size: `solid_disc`) or given;
* **bearings** (`Bearing`): springs and dampers from the shaft to the ground in x and y, which may differ and
  be cross-coupled, or a plain journal bearing's oil film (core/fluid_bearing.py, under the bearing's share of
  the weight, stiffness and damping changing with the speed), or an active magnetic bearing
  (core/magnetic_bearing.py, its negative stiffness and its PID controller's states);
* **seals** (`Seal`): an annular seal's direct stiffness and damping, an added mass, and the cross-coupled
  stiffness k_xy = λΩc that the fluid's swirl at a fraction λ of the shaft's speed brings, growing with it.

The mesh (`Rotor.nodes`) puts a node at every section's ends and every part, then shares each stretch's
elements out by length, so a part always sits on a node. Every node has four DOFs: x, the slope θx = dx/dz, y
and θy = dy/dz, numbered 4i to 4i + 3 (`dof_label`). With slopes in both planes the two planes have the same
beam matrices.

Each shaft element is a rotating Timoshenko beam (Nelson's: shear through φ = 12EI/κGAL², κ Cowper's for a
tube, and the sections' rotary inertia), or a Rayleigh beam (Euler–Bernoulli with the rotary inertia: φ = 0),
the matrices `frame.bending_matrices`' on [w1, θ1, w2, θ2] in each plane. A spinning section's angular
momentum gives it a gyroscopic matrix too: its sections' polar inertia is twice their diametral one, so its
share of G is twice its rotary inertia, coupling the x-plane slopes to the y-plane's. A disc adds m to x and y,
I_d to θx and θy, and I_p to G. So (`Assembly`)

    M q'' + (C(Ω) + Ω G) q' + K(Ω) q = f,

G skew (it does no work), K(Ω) = K + Ω H + K_f(Ω) and C(Ω) = C + C_f(Ω): H the seals' skew λc (and the shaft's
internal damping's: β K_shaft acts on the bending rate seen in the turning shaft, q' - ΩJq, so it is ordinary
damping βK_shaft in C and a skew βK_shaft between the planes in H, which feeds a forward whirl slower than the
shaft), K_f and C_f the
oil films' (each a full 2 × 2 with cross terms, linearized where its journal settles under its load). A film's
load is its bearing's share of the rotor's weight, with every bearing held (`Assembly.loads`): on a vertical
rotor (no gravity) it carries none and runs centred. A magnetic bearing puts -k_m in K and its controllers'
states after q and q' in the state z = [q, q', x_c], so the modes come from the state matrix A(Ω)
(`state_matrix`), as the rotor page's do. Its eigenvalues λ = -ζω ± iω_d are the whirl modes at that speed
(`modes`): each one's whirl ratio says whether it whirls with the spin (forward, +1) or against it (backward,
-1), weighted by kinetic energy. Repeated pairs (a round rotor at rest) are made circular, as on the rotor page.

Working on it
-------------
`core/rotor_builder.py` (the *Rotor builder* page, `gui/rotor_builder.py`, after the Jeffcott rotor; its own
model, not the chain or `core/rotor.py`'s): a frozen `Rotor` of `Section`s and parts (`Disc`, `Bearing`, `Seal`)
at absolute z; editing returns a new rotor (`split`, `resized`, `without_section` move the parts past the change
with it), so the page's undo is a list of rotors. `assemble` gives the speed-free matrices and each element's
share (`Part`, tinted on the Assembly tab); `Assembly.stiffness(Ω)` and `damping(Ω)` add the seals', the
films' and the internal damping's terms, so use them, not `K` and `C`, wherever the speed is known. The modes
table works in the full model; the sweeps (core/rotor_builder_sweep.py) in `Assembly.basis`'s Ritz basis, which
`modes` and `state_matrix` take too (a test checks the two agree on every trial). The beam matrices are the Frame
builder's (`frame.bending_matrices`): its tests check Timoshenko's against Euler–Bernoulli's at φ = 0, and
`tests/test_rotor_builder.py` checks this model against the closed forms (a pinned shaft, a free one, a disc's
gyroscopic split) and against `core/rotor.py` for a Jeffcott rotor with a massive shaft, and the trials' notes.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import Any

import numpy as np
import scipy.linalg

from .fluid_bearing import BearingState, JournalBearing
from .frame import MATERIALS, STEEL, Material, Shape, bending_matrices
from .frame import Section as FrameSection
from .magnetic_bearing import MagneticBearing

TWO_PI = 2.0 * math.pi
RPM = TWO_PI / 60.0  # rad/s per rpm
DOF_NAMES = ("x", "θx", "y", "θy")
MAX_NODES = 61
MAX_ELEMENTS = 12  # in one section
GRAVITY = 9.81  # m/s²: a horizontal rotor's
STILL = TWO_PI * 0.01  # rad/s: a root slower than this (0.01 Hz) is a rigid-body one, the rotor not held
OVERDAMPED = 0.05  # a root with |Im λ| below this times |Re λ| is taken as not oscillating
REPEATED = 1e-7  # eigenvalues this close (relative) are a pair to make circular
NEUTRAL = 1e-9  # a damping ratio smaller than this is round-off: an undamped rotor's is 0, not growing
GAP = 1e-9  # m: two positions this close are the same node
ECCENTRICITY = 10e-6  # m: a library disc's centre of mass off the shaft's (its unbalance m e; ISO G2.5 at ~2400 rpm)
REDUCED = 32  # DOFs: a sweep over the speed works on the lowest this many undamped modes (see `Basis`)


class RotorError(ValueError):
    """The rotor can't be worked out as it stands (a part off the shaft, an oil film with nothing to rest on)."""


class Theory(Enum):
    TIMOSHENKO = "Timoshenko"  # shear and rotary inertia
    RAYLEIGH = "Euler–Bernoulli"  # with rotary inertia: a Rayleigh beam


THEORIES = tuple(Theory)


class BearingKind(Enum):
    SPRINGS = "Springs and dampers"
    FILM = "Oil film (plain journal)"
    MAGNET = "Magnetic (PID)"


BEARING_KINDS = tuple(BearingKind)


@dataclass(frozen=True)
class Section:
    """A stretch of shaft: a round tube `outer` across with a bore `inner` (0: solid), `elements` beams long."""

    length: float = 0.25  # m
    outer: float = 0.05  # m
    inner: float = 0.0  # m
    material: Material = STEEL
    elements: int = 2

    @property
    def area(self) -> float:
        return math.pi / 4 * (self.outer**2 - self.inner**2)

    @property
    def inertia(self) -> float:
        """The diametral second moment of area, π(D⁴ - d⁴)/64; the polar one is twice it."""
        return math.pi / 64 * (self.outer**4 - self.inner**4)

    @property
    def kappa(self) -> float:
        """Cowper's shear coefficient for a round tube."""
        return FrameSection(Shape.TUBE, self.outer, wall=(self.outer - self.inner) / 2).kappa(self.material.nu)

    @property
    def mass(self) -> float:
        return self.material.rho * self.area * self.length

    @property
    def ei(self) -> float:
        return self.material.E * self.inertia


@dataclass(frozen=True)
class Disc:
    """A rigid disc at z: mass, polar and diametral inertia; `diameter` and `thickness` for drawing it."""

    z: float = 0.5  # m
    mass: float = 10.0  # kg
    ip: float = 0.05  # kg·m², about the spin axis
    id: float = 0.025  # kg·m², about a diameter through its centre
    diameter: float = 0.2  # m
    thickness: float = 0.04  # m
    name: str = "Custom"  # the library's entry it came from
    unbalance: float = 0.0  # kg·m, m e: its centre of mass off the shaft's by e
    unbalance_angle: float = 0.0  # rad, the heavy spot's angle from the shaft's keyphasor mark


def solid_disc(
    z: float, material: Material, diameter: float, thickness: float, bore: float = 0.0, name: str = "Custom"
) -> Disc:
    """A disc cut from `material`: an annulus `diameter` across with a `bore`, `thickness` thick.

    m = ρπ(D² - d²)t/4, I_p = m(R² + r²)/2, I_d = m(3(R² + r²) + t²)/12.
    """
    R, r = diameter / 2, bore / 2
    m = material.rho * math.pi * (R * R - r * r) * thickness
    ip, id_ = m * (R * R + r * r) / 2, m * (3 * (R * R + r * r) + thickness**2) / 12
    return Disc(z, m, ip, id_, diameter, thickness, name, m * ECCENTRICITY)


@dataclass(frozen=True)
class DiscType:
    name: str
    material: Material
    diameter: float  # m
    thickness: float  # m
    bore: float = 0.0  # m: 0 for a disc shrunk onto the shaft, the shaft's own mass where it sits counting

    def at(self, z: float) -> Disc:
        return solid_disc(z, self.material, self.diameter, self.thickness, self.bore, self.name)


DISC_LIBRARY = (
    DiscType("Steel disc Ø200 × 30 mm", STEEL, 0.2, 0.03),
    DiscType("Steel disc Ø300 × 40 mm", STEEL, 0.3, 0.04),
    DiscType("Steel flywheel Ø400 × 80 mm", STEEL, 0.4, 0.08),
    DiscType("Aluminium impeller Ø350 × 60 mm", MATERIALS[1], 0.35, 0.06),
    DiscType("Titanium fan Ø500 × 20 mm", MATERIALS[2], 0.5, 0.02),
    DiscType("Steel coupling hub Ø120 × 80 mm", STEEL, 0.12, 0.08),
)


@dataclass(frozen=True)
class Bearing:
    """A bearing at z: springs and dampers to the ground (f = -K [x, y] - C [x', y']), an oil film or a magnet."""

    z: float = 0.0  # m
    kind: BearingKind = BearingKind.SPRINGS
    kxx: float = 5e7  # N/m
    kyy: float = 5e7  # N/m
    kxy: float = 0.0  # N/m: f_x = -k_xy y
    kyx: float = 0.0  # N/m: f_y = -k_yx x
    cxx: float = 2e3  # N·s/m
    cyy: float = 2e3  # N·s/m
    film: JournalBearing = field(default_factory=lambda: JournalBearing(0.025, 0.05, 50e-6, 0.03, 0.0))
    magnet: MagneticBearing = field(default_factory=MagneticBearing)

    def springs(self) -> tuple[np.ndarray, np.ndarray]:
        """(K, C) on [x, y], speed-free: a film's are none (they come with the speed), a magnet's -k_m."""
        if self.kind is BearingKind.FILM:
            return np.zeros((2, 2)), np.zeros((2, 2))
        if self.kind is BearingKind.MAGNET:
            return -self.magnet.negative_stiffness * np.eye(2), np.zeros((2, 2))
        return np.array([[self.kxx, self.kxy], [self.kyx, self.kyy]]), np.diag([self.cxx, self.cyy])


@dataclass(frozen=True)
class Seal:
    """An annular seal at z: direct k and c, an added mass, and the swirl λ that makes k_xy = λΩc."""

    z: float = 0.5  # m
    k: float = 2e6  # N/m
    c: float = 2e3  # N·s/m
    swirl: float = 0.5  # λ: the fluid turns at λΩ
    mass: float = 0.0  # kg, added to x and y


PART_KINDS = ("disc", "bearing", "seal")
type Selection = tuple[str, int]  # ("section", k), ("disc", i), ("bearing", i) or ("seal", i)


@dataclass(frozen=True)
class Rotor:
    sections: tuple[Section, ...] = (Section(0.5),)
    discs: tuple[Disc, ...] = ()
    bearings: tuple[Bearing, ...] = ()
    seals: tuple[Seal, ...] = ()
    theory: Theory = Theory.TIMOSHENKO
    gravity: float = GRAVITY  # m/s², pulling in -y; 0 for a vertical rotor
    internal: float = 0.0  # s, β: the shaft's internal damping, β K_shaft on its bending rate in the shaft

    # --------------------------------------------------------- geometry
    @property
    def length(self) -> float:
        return float(sum(s.length for s in self.sections))

    @property
    def ends(self) -> np.ndarray:
        """The sections' ends, 0 first: len(sections) + 1 positions."""
        return np.concatenate([[0.0], np.cumsum([s.length for s in self.sections])])

    def section_at(self, z: float) -> int | None:
        """The section z is in (the left one at a joint), None off the shaft."""
        ends = self.ends
        if not self.sections or z < -GAP or z > ends[-1] + GAP:
            return None
        i = int(np.searchsorted(ends, z - GAP)) - 1
        return min(max(i, 0), len(self.sections) - 1)

    def outer_at(self, z: float) -> float:
        """The shaft's diameter at z (the larger one at a step), 0 off it."""
        ends = self.ends
        found = [s.outer for s, a, b in zip(self.sections, ends[:-1], ends[1:]) if a - GAP <= z <= b + GAP]
        return max(found, default=0.0)

    def parts(self, kind: str) -> tuple:
        return {"disc": self.discs, "bearing": self.bearings, "seal": self.seals}[kind]

    @property
    def mass(self) -> float:
        return float(sum(s.mass for s in self.sections) + sum(d.mass for d in self.discs))

    # --------------------------------------------------------- editing
    def _with(self, kind: str, items: tuple) -> Rotor:
        changes: dict[str, Any] = {kind + "s": items}
        return dataclasses.replace(self, **changes)

    def with_part(self, kind: str, part) -> Rotor:
        return self._with(kind, (*self.parts(kind), part))

    def replace_part(self, kind: str, i: int, part) -> Rotor:
        items = list(self.parts(kind))
        items[i] = part
        return self._with(kind, tuple(items))

    def without_part(self, kind: str, i: int) -> Rotor:
        items = self.parts(kind)
        return self._with(kind, items[:i] + items[i + 1 :])

    def part_at(self, kind: str, z: float, reach: float = GAP) -> int | None:
        best, near = None, reach
        for i, p in enumerate(self.parts(kind)):
            if abs(p.z - z) <= near:
                best, near = i, abs(p.z - z)
        return best

    def _moved(self, move) -> Rotor:
        """Every part at move(z), those it gives None dropped."""
        out: dict[str, Any] = {}
        for kind in PART_KINDS:
            kept = []
            for p in self.parts(kind):
                z = move(p.z)
                if z is not None:
                    kept.append(dataclasses.replace(p, z=float(z)))
            out[kind + "s"] = tuple(kept)
        return dataclasses.replace(self, **out)

    def replace_section(self, k: int, section: Section) -> Rotor:
        """Section k swapped for `section`; if its length changes, the parts on it stretch with it and those past
        it move along."""
        a, b = self.ends[k], self.ends[k + 1]
        grow = section.length - (b - a)
        scale = section.length / (b - a)
        sections = self.sections[:k] + (section,) + self.sections[k + 1 :]
        rotor = dataclasses.replace(self, sections=sections)
        if grow == 0:
            return rotor
        return rotor._moved(lambda z: z + grow if z > b + GAP else a + (z - a) * scale if z > a + GAP else z)

    def with_section(self, section: Section | None = None) -> Rotor:
        """A section added at the right-hand end (a copy of the last one by default)."""
        new = section if section is not None else (self.sections[-1] if self.sections else Section())
        return dataclasses.replace(self, sections=(*self.sections, new))

    def without_section(self, k: int) -> Rotor:
        """Section k taken out: the parts on it go with it, those past it move back."""
        a, b = self.ends[k], self.ends[k + 1]
        sections = self.sections[:k] + self.sections[k + 1 :]
        rotor = dataclasses.replace(self, sections=sections)
        last = len(self.sections) - 1

        def move(z: float) -> float | None:
            if z >= b - GAP and k < last:  # at or past its far end
                return z - (b - a)
            if z <= a + GAP and k > 0:  # at or before its near end
                return z
            return None  # on it, or on an end that goes with it

        return rotor._moved(move)

    def split(self, k: int, z: float) -> Rotor:
        """Section k cut in two at z (its elements shared out by length)."""
        a, b = self.ends[k], self.ends[k + 1]
        if not a + GAP < z < b - GAP:
            return self
        s = self.sections[k]
        left = max(1, round(s.elements * (z - a) / (b - a)))
        right = max(1, s.elements - left)
        parts = (
            dataclasses.replace(s, length=z - a, elements=left),
            dataclasses.replace(s, length=b - z, elements=right),
        )
        return dataclasses.replace(self, sections=self.sections[:k] + parts + self.sections[k + 1 :])

    # --------------------------------------------------------- the mesh
    @cached_property
    def _mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """(z of every node, the section each element is in)."""
        if not self.sections:
            return np.zeros(0), np.zeros(0, dtype=int)
        ends = self.ends
        stops = sorted({p.z for kind in PART_KINDS for p in self.parts(kind)})
        z, owner = [0.0], []
        for k, (s, a, b) in enumerate(zip(self.sections, ends[:-1], ends[1:])):
            cuts = [a] + [w for w in stops if a + GAP < w < b - GAP] + [b]
            for z0, z1 in itertools.pairwise(cuts):
                n = max(1, round(s.elements * (z1 - z0) / (b - a)))
                z.extend(np.linspace(z0, z1, n + 1)[1:])
                owner.extend([k] * n)
        return np.array(z), np.array(owner, dtype=int)

    @property
    def nodes(self) -> np.ndarray:
        return self._mesh[0]

    @property
    def element_sections(self) -> np.ndarray:
        """The section each element is in."""
        return self._mesh[1]

    @property
    def n_dof(self) -> int:
        return 4 * self.nodes.size

    def node_at(self, z: float) -> int:
        """The node at z (the nearest)."""
        return int(np.argmin(np.abs(self.nodes - z)))

    def check(self) -> None:
        """Raise RotorError if it can't be worked out."""
        if not self.sections:
            raise RotorError("There's no shaft: add a section.")
        for s in self.sections:
            if s.length <= 0 or s.outer <= 0 or not 0 <= s.inner < s.outer:
                raise RotorError("Every section needs a length, and a bore smaller than its diameter.")
        top = self.length
        for kind in PART_KINDS:
            for p in self.parts(kind):
                if not -GAP <= p.z <= top + GAP:
                    raise RotorError(f"A {kind} is off the shaft (at z = {p.z:.4g} m; the shaft is {top:.4g} m).")
        zs = [b.z for b in self.bearings]
        if len(set(np.round(zs, 9))) < len(zs):
            raise RotorError("Two bearings are at the same place: take one away.")
        if self.nodes.size > MAX_NODES:
            raise RotorError(f"The mesh has {self.nodes.size} nodes, more than {MAX_NODES}: use fewer elements.")


def dof_label(dof: int) -> str:
    return f"{DOF_NAMES[dof % 4]}{dof // 4 + 1}"


def plane_dofs(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """The x plane's DOFs [x, θx] node by node, and the y plane's [y, θy] beside them."""
    i = np.arange(n_nodes)
    x = np.column_stack([4 * i, 4 * i + 1]).ravel()
    return x, x + 2


# ------------------------------------------------------------------ assembly
@dataclass(frozen=True)
class Part:
    """One element's share of the matrices, on its `dofs`: a shaft element, a disc, a bearing or a seal."""

    owner: Selection
    label: str
    dofs: tuple[int, ...]
    M: np.ndarray
    K: np.ndarray
    C: np.ndarray
    G: np.ndarray
    H: np.ndarray  # the skew stiffness per unit speed (a seal's λc)
    film: tuple[JournalBearing, float] | None = None  # an oil film and its load (N): K and C come with the speed
    phi: float = 0.0  # a Timoshenko element's φ
    length: float = 0.0  # a shaft element's

    def stiffness(self, omega: float) -> np.ndarray:
        if self.film is not None:
            return self.K + film_state(*self.film, omega).k
        return self.K + omega * self.H

    def damping(self, omega: float) -> np.ndarray:
        if self.film is not None:
            return self.C + film_state(*self.film, omega).c
        return self.C


def film_state(film: JournalBearing, load: float, omega: float) -> BearingState:
    """The film's state at speed Ω under `load` (its sign only mirrors where the journal sits)."""
    return film.with_load(abs(load)).on_rotor(omega)


@dataclass(frozen=True, eq=False)
class Assembly:
    """The rotor's matrices without the speed's terms, on every node's [x, θx, y, θy]: see stiffness(Ω)."""

    rotor: Rotor
    M: np.ndarray
    K: np.ndarray  # the shaft, the bearings' springs, the seals' direct k and the magnets' -k_m
    C: np.ndarray  # the bearings' and seals' dampers
    G: np.ndarray  # gyroscopic, per unit speed: Ω G
    H: np.ndarray  # the seals' skew stiffness per unit speed: Ω H
    parts: tuple[Part, ...]
    loads: tuple[float, ...]  # each bearing's share of the weight (N, up)
    films: tuple[tuple[int, JournalBearing, float], ...]  # (the journal's x DOF, the film, its load)
    axes: tuple[tuple[int, MagneticBearing], ...]  # each magnet's axes: (DOF, its bearing)

    @property
    def n(self) -> int:
        return self.M.shape[0]

    def stiffness(self, omega: float) -> np.ndarray:
        """K + Ω H + the films' K_f(Ω)."""
        K = self.K + omega * self.H
        for d, film, load in self.films:
            K[np.ix_([d, d + 2], [d, d + 2])] += film_state(film, load, omega).k
        return K

    def damping(self, omega: float) -> np.ndarray:
        """C + the films' C_f(Ω) (without the gyroscopic Ω G)."""
        C = self.C.copy()
        for d, film, load in self.films:
            C[np.ix_([d, d + 2], [d, d + 2])] += film_state(film, load, omega).c
        return C

    @property
    def n_control(self) -> int:
        return sum(m.n_states for _, m in self.axes)

    @property
    def n_states(self) -> int:
        return 2 * self.n + self.n_control

    @cached_property
    def mass_inverse(self) -> np.ndarray:
        return np.linalg.inv(self.M)

    def state_matrix(self, omega: float, basis: Basis | None = None) -> np.ndarray:
        """A(Ω) for z = [q, q', x_c], or with a basis T for z = [p, p', x_c], q = T p."""
        T = basis.T if basis is not None else None
        M, K, C = self.M, self.stiffness(omega), self.damping(omega) + omega * self.G
        if T is not None:
            M, K, C = T.T @ M @ T, T.T @ K @ T, T.T @ C @ T
        Minv = self.mass_inverse if T is None else np.linalg.inv(M)
        n = M.shape[0]
        A = np.zeros((2 * n + self.n_control, 2 * n + self.n_control))
        A[:n, n : 2 * n] = np.eye(n)
        A[n : 2 * n, :n] = -Minv @ K
        A[n : 2 * n, n : 2 * n] = -Minv @ C
        k = 2 * n
        for d, m in self.axes:
            a, b, c = m.state_space()
            sl = slice(k, k + b.size)
            row = T[d] if T is not None else np.eye(n)[d]  # the journal's displacement in the coordinates
            A[sl, sl] = a
            A[sl, :n] = np.outer(b, row)
            A[n : 2 * n, sl] += np.outer(Minv @ row, -m.current_gain * c)
            k += b.size
        return A

    def basis(self, omega: float = 0.0, size: int = REDUCED) -> Basis | None:
        """The lowest `size` undamped modes of a symmetric stand-in, as a Ritz basis (None: too few DOFs to need
        one).

        The stand-in is K's symmetric part with the films' direct stiffness at speed Ω and each magnet's
        controller as its proportional spring k_i k_p: near enough the rotor's for its lowest modes to span the
        lowest whirl modes at any speed nearby, with G, C, the films' cross terms and the controllers all kept in
        the projection. Rayleigh–Ritz: the reduced model's frequencies are close to the full one's low ones.
        """
        if self.n <= size:
            return None
        K = 0.5 * (self.K + self.K.T)
        for d, film, load in self.films:
            k = film_state(film, load, omega).k
            K[np.ix_([d, d + 2], [d, d + 2])] += 0.5 * (k + k.T)
        for d, m in self.axes:
            K[d, d] += m.current_gain * m.kp
        _, T = scipy.linalg.eigh(K, self.M, subset_by_index=(0, size - 1))
        return Basis(T)


@dataclass(frozen=True, eq=False)
class Basis:
    """A Ritz basis for the displacements, q = T p (n_dof × size): see `Assembly.basis`."""

    T: np.ndarray


def _place(A: np.ndarray, dofs: list[int], block: np.ndarray) -> None:
    A[np.ix_(dofs, dofs)] += block


def assemble(rotor: Rotor) -> Assembly:
    """The matrices, element by element (raises RotorError if the rotor can't be worked out)."""
    rotor.check()
    z = rotor.nodes
    n = 4 * z.size
    M, K, C, G, H = (np.zeros((n, n)) for _ in range(5))
    parts: list[Part] = []
    timoshenko = rotor.theory is Theory.TIMOSHENKO
    for e, k in enumerate(rotor.element_sections):
        s = rotor.sections[k]
        L = float(z[e + 1] - z[e])
        mat = s.material
        kb, mt, mr, phi = bending_matrices(mat.E, mat.G, mat.rho, s.area, s.inertia, s.kappa, L, timoshenko, True)
        x = [4 * e, 4 * e + 1, 4 * e + 4, 4 * e + 5]
        y = [d + 2 for d in x]
        dofs = sorted(x + y)
        local = {name: np.zeros((8, 8)) for name in ("M", "K", "G")}
        lx, ly = [dofs.index(d) for d in x], [dofs.index(d) for d in y]
        for own in (lx, ly):
            local["K"][np.ix_(own, own)] += kb
            local["M"][np.ix_(own, own)] += mt + mr
        local["G"][np.ix_(lx, ly)] += 2 * mr  # the sections' polar inertia is twice their diametral
        local["G"][np.ix_(ly, lx)] -= 2 * mr
        # Internal damping turns with the shaft: -β K (q' - Ω J q), J turning x toward y. Its -Ω J part is a skew
        # stiffness Ω H, H = β [0 K; -K 0] between the planes.
        damp, skew = rotor.internal * local["K"], np.zeros((8, 8))
        skew[np.ix_(lx, ly)] += rotor.internal * kb
        skew[np.ix_(ly, lx)] -= rotor.internal * kb
        parts.append(
            Part(
                ("section", int(k)),
                f"Element {e + 1}",
                tuple(dofs),
                local["M"],
                local["K"],
                damp,
                local["G"],
                skew,
                phi=phi,
                length=L,
            )
        )
    for i, d in enumerate(rotor.discs):
        j = rotor.node_at(d.z)
        dofs = [4 * j + q for q in range(4)]
        m = np.diag([d.mass, d.id, d.mass, d.id])
        g = np.zeros((4, 4))
        g[1, 3], g[3, 1] = d.ip, -d.ip
        zero = np.zeros((4, 4))
        parts.append(Part(("disc", i), f"Disc {i + 1}", tuple(dofs), m, zero, zero, g, zero))
    for i, s in enumerate(rotor.seals):
        j = rotor.node_at(s.z)
        dofs = [4 * j, 4 * j + 2]
        h = np.array([[0.0, s.swirl * s.c], [-s.swirl * s.c, 0.0]])
        parts.append(
            Part(
                ("seal", i),
                f"Seal {i + 1}",
                tuple(dofs),
                s.mass * np.eye(2),
                s.k * np.eye(2),
                s.c * np.eye(2),
                np.zeros((2, 2)),
                h,
            )
        )
    for part in parts:
        for A, a in ((M, part.M), (K, part.K), (C, part.C), (G, part.G), (H, part.H)):
            _place(A, list(part.dofs), a)
    loads = (
        _loads(rotor, M, K) if any(b.kind is BearingKind.FILM for b in rotor.bearings) else (0.0,) * len(rotor.bearings)
    )
    films, axes = [], []
    for i, (b, w) in enumerate(zip(rotor.bearings, loads)):
        j = rotor.node_at(b.z)
        dofs = [4 * j, 4 * j + 2]
        kb, cb = b.springs()
        film = (b.film, w) if b.kind is BearingKind.FILM else None
        parts.append(
            Part(
                ("bearing", i),
                f"Bearing {i + 1}",
                tuple(dofs),
                np.zeros((2, 2)),
                kb,
                cb,
                np.zeros((2, 2)),
                np.zeros((2, 2)),
                film=film,
            )
        )
        _place(K, dofs, kb)
        _place(C, dofs, cb)
        if b.kind is BearingKind.FILM:
            films.append((4 * j, b.film, w))
        elif b.kind is BearingKind.MAGNET:
            axes += [(4 * j, b.magnet), (4 * j + 2, b.magnet)]
    return Assembly(rotor, M, K, C, G, H, tuple(parts), loads, tuple(films), tuple(axes))


def _loads(rotor: Rotor, M: np.ndarray, K: np.ndarray) -> tuple[float, ...]:
    """Each bearing's share of the weight (N, up) with every bearing held: the shaft and seals carry the rest."""
    if not rotor.gravity:
        return (0.0,) * len(rotor.bearings)
    if len(rotor.bearings) < 2:
        raise RotorError("An oil film needs the rotor to rest on two bearings or more, to know its load.")
    n = M.shape[0]
    up = np.zeros(n)
    up[2::4] = 1.0
    f = -rotor.gravity * M @ up
    held = [4 * rotor.node_at(b.z) + 2 for b in rotor.bearings]
    free = [d for d in range(n) if d not in held]
    q = np.zeros(n)
    q[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
    push = K[held] @ q - f[held]
    return tuple(float(p) for p in push)


# ------------------------------------------------------------------ modes at a speed
@dataclass(frozen=True, eq=False)
class WhirlModes:
    """The rotor's damped modes at one speed, slowest first (one per conjugate pair)."""

    omega: float  # rad/s
    hz: np.ndarray  # damped natural frequencies
    zeta: np.ndarray  # -Re λ / |λ|
    whirl: np.ndarray  # > 0 forward, < 0 backward (see whirl_ratio)
    shapes: np.ndarray  # n_dof × modes, complex: q = Re(shape e^{iωt}), its largest translation 1
    rigid: int  # roots at 0: the rotor isn't held (each free motion counts twice)
    roots: np.ndarray  # the other real roots (rad/s, < 0 if they decay): overdamped, or a controller's

    @property
    def log_dec(self) -> np.ndarray:
        """δ = 2πζ/√(1 - ζ²): the logarithmic decrement."""
        z = np.clip(self.zeta, -0.999999, 0.999999)
        return TWO_PI * z / np.sqrt(1 - z * z)


def whirl_ratio(assembly: Assembly, Q: np.ndarray) -> np.ndarray:
    """+1 a circular forward whirl (with the spin, x toward y), -1 backward, 0 a straight line; Q (n, modes).

    Weighted by one plane's mass matrix (kinetic energy), as on the rotor page.
    """
    ix, iy = plane_dofs(assembly.n // 4)
    mp = assembly.M[np.ix_(ix, ix)]
    X, Y = Q[ix], Q[iy]
    num = 2.0 * np.imag(np.einsum("im,ij,jm->m", np.conj(Y), mp, X))
    den = np.real(np.einsum("im,ij,jm->m", np.conj(X), mp, X) + np.einsum("im,ij,jm->m", np.conj(Y), mp, Y))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def _circularize(assembly: Assembly, lam: np.ndarray, vec: np.ndarray) -> None:
    """Within each group of repeated eigenvalues, the purely forward and backward combinations (vec: the
    displacements, n_dof × modes)."""
    n = assembly.n
    ix, iy = plane_dofs(n // 4)
    mp = assembly.M[np.ix_(ix, ix)].astype(complex)
    Hf, Df = np.zeros((n, n), complex), np.zeros((n, n), complex)
    Hf[np.ix_(iy, ix)], Hf[np.ix_(ix, iy)] = 1j * mp, -1j * mp
    Df[np.ix_(ix, ix)], Df[np.ix_(iy, iy)] = mp, mp
    i = 0
    while i < lam.size:
        j = i + 1
        while j < lam.size and abs(lam[j] - lam[i]) <= REPEATED * abs(lam[i]):
            j += 1
        if j - i > 1:
            V = vec[:, i:j]
            _, c = scipy.linalg.eigh(V.conj().T @ Hf @ V, V.conj().T @ Df @ V)
            vec[:, i:j] = V @ c
        i = j


def modes(assembly: Assembly, omega: float, basis: Basis | None = None) -> WhirlModes:
    """The whirl modes at speed Ω (rad/s): exactly, or in a Ritz basis (`Assembly.basis`), much faster."""
    n = assembly.n
    lam, vec = scipy.linalg.eig(assembly.state_matrix(omega, basis))
    still = np.abs(lam) < STILL
    real = ~still & (np.abs(lam.imag) < OVERDAMPED * np.abs(lam.real))
    up = ~still & ~real & (lam.imag > 0)
    order = np.argsort(lam[up].imag)
    lam_up = lam[up][order]
    Q = vec[:, up][:, order]
    Q = Q[:n] if basis is None else basis.T @ Q[: basis.T.shape[1]]
    _circularize(assembly, lam_up, Q)
    if Q.size:
        trans = np.abs(Q[np.r_[0:n:4, 2:n:4]]).max(axis=0)
        top = np.argmax(np.abs(Q), axis=0)
        size = np.where(trans > 0, trans, np.abs(Q).max(axis=0))
        Q = Q * np.exp(-1j * np.angle(Q[top, np.arange(Q.shape[1])])) / size
    w = np.abs(lam_up)
    zeta = -lam_up.real / np.where(w > 0, w, 1.0)
    return WhirlModes(
        omega,
        lam_up.imag / TWO_PI,
        np.where(np.abs(zeta) < NEUTRAL, 0.0, zeta),
        whirl_ratio(assembly, Q),
        Q,
        int(still.sum()),
        np.sort(lam[real].real)[::-1],
    )


def deflection(rotor: Rotor, q: np.ndarray, points: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(z, x, y): the shaft's centre line drawn through q (real, n_dof) by each element's Hermite cubics."""
    z = rotor.nodes
    zs: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    s = np.linspace(0.0, 1.0, points + 1)
    h1, h2 = 1 - 3 * s**2 + 2 * s**3, s - 2 * s**2 + s**3
    h3, h4 = 3 * s**2 - 2 * s**3, -(s**2) + s**3
    for e in range(z.size - 1):
        L = z[e + 1] - z[e]
        first = 0 if e == 0 else 1
        zs.append((z[e] + s * L)[first:])
        for out, d in ((xs, 0), (ys, 2)):
            a, b = 4 * e + d, 4 * e + 4 + d
            out.append((h1 * q[a] + h2 * L * q[a + 1] + h3 * q[b] + h4 * L * q[b + 1])[first:])
    if not zs:
        return z.copy(), q[0::4].copy(), q[2::4].copy()
    return np.concatenate(zs), np.concatenate(xs), np.concatenate(ys)


# ------------------------------------------------------------------ ready-made rotors
def uniform(
    length: float = 1.0, diameter: float = 0.05, parts: int = 4, elements: int = 2, material: Material = STEEL
) -> tuple[Section, ...]:
    """A shaft of `parts` equal sections."""
    return tuple(Section(length / parts, diameter, 0.0, material, elements) for _ in range(parts))


def pinned_ends(rotor: Rotor, stiffness: float = 1e12) -> Rotor:
    """The rotor on very stiff, undamped bearings at its two ends (close to pinned)."""
    ends = (
        Bearing(0.0, kxx=stiffness, kyy=stiffness, cxx=0.0, cyy=0.0),
        Bearing(rotor.length, kxx=stiffness, kyy=stiffness, cxx=0.0, cyy=0.0),
    )
    return dataclasses.replace(rotor, bearings=ends)
