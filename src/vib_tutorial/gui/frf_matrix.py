"""FRF matrix page: every receptance H_jk, and how modes and inputs build it up.

The centre is an N x N grid of small plots, one per term of the receptance
matrix H(w) (row j = response x_j, column k = force at m_k). The controls
choose which modal terms are summed, and which forces act together; the
selected cell is drawn large on the right with its individual terms, their
sum, and the full solution.
"""

from __future__ import annotations

import enum
import itertools
import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import ChainSystem, ModalResult, ModalTerm, frf_matrix, modal_analysis, modal_frf_terms
from ..core.practice import Problem, Question, Requirement, at_most, number, pick
from . import weak
from .layouts import ColumnScroll, form_layout
from .lesson import TrialCombo, trials_html
from .links import load_notes, page_link
from .plots import frequency_grid
from .style import MAX_DOF, colors
from .theming import add_legend, mute

# Colours: the full solution in colors.strong; the sum of the modal terms, the
# selected cells and the worst band in colors.force.
FREQ_POINTS = 600  # per curve; the grid draws up to N^2 (2 + modes) curves
WORST_FRACTION = 0.5  # the shaded band is where the difference is at least this share of its largest


class Quantity(enum.Enum):
    MAGNITUDE = "Magnitude |H|"
    PHASE = "Phase ∠H"
    REAL = "Real part Re H"
    IMAG = "Imaginary part Im H"

    @property
    def partner(self) -> Quantity:
        """The quantity drawn with this one in the large plot."""
        return {
            Quantity.MAGNITUDE: Quantity.PHASE,
            Quantity.PHASE: Quantity.MAGNITUDE,
            Quantity.REAL: Quantity.IMAG,
            Quantity.IMAG: Quantity.REAL,
        }[self]

    @property
    def axis_label(self) -> str:
        return {
            Quantity.MAGNITUDE: "|H|  [m/N]",
            Quantity.PHASE: "∠H  [deg]",
            Quantity.REAL: "Re H  [m/N]",
            Quantity.IMAG: "Im H  [m/N]",
        }[self]


class Expansion(enum.Enum):
    EXACT = "Exact (complex modes)"
    CLASSICAL = "Classical (real modes, ζ modal)"


def part(h: np.ndarray, q: Quantity) -> np.ndarray:
    """The plotted real quantity of complex FRF values (along axis 0)."""
    if q is Quantity.MAGNITUDE:
        m = np.abs(h)
        return np.where(m > 0, m, np.nan)  # log axis
    if q is Quantity.PHASE:
        # Unwrapped, so a phase lag that builds up over several resonances stays visible.
        return np.degrees(np.unwrap(np.angle(h), axis=0))
    return h.real.copy() if q is Quantity.REAL else h.imag.copy()


def y_range(q: Quantity, values: list[np.ndarray]) -> tuple[float, float]:
    """Shared y range (log10 units for magnitude) covering `values`."""
    finite = np.concatenate([v[np.isfinite(v)].ravel() for v in values])
    if q is Quantity.PHASE:
        if not finite.size:
            return -195.0, 195.0
        return 90.0 * math.floor(finite.min() / 90.0) - 15.0, 90.0 * math.ceil(finite.max() / 90.0) + 15.0
    if q is Quantity.MAGNITUDE:
        finite = finite[finite > 0]
        if not finite.size:
            return -6.0, 0.0
        return math.log10(finite.min()) - 0.3, math.log10(finite.max()) + 0.3
    m = float(np.abs(finite).max()) if finite.size else 1.0
    return -1.1 * m, 1.1 * m


def worst_band(f: np.ndarray, full: np.ndarray, partial: np.ndarray) -> tuple[float, float, float] | None:
    """(low, high, peak) frequencies of the band around the largest |partial - full|.

    The band extends from the peak for as long as the difference stays above
    WORST_FRACTION of its largest value. None if the two agree to rounding.
    """
    d = np.abs(partial - full)
    i = int(np.argmax(d))
    if d[i] <= 1e-9 * max(np.abs(full).max(), 1e-300):
        return None
    above = d >= WORST_FRACTION * d[i]
    lo = i
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = i
    while hi < d.size - 1 and above[hi + 1]:
        hi += 1
    return float(f[max(lo - 1, 0)]), float(f[min(hi + 1, d.size - 1)]), float(f[i])


def set_phase_ticks(axis: pg.AxisItem, phase: bool) -> None:
    """Degrees on multiples of 90 (labels every 180 when the range is wide); otherwise automatic."""
    if phase:
        axis.setTickSpacing(180.0, 90.0)
    else:
        axis.setTickSpacing()


def log_ticks(lo: float, hi: float, mantissas: tuple[int, ...]) -> list[list[tuple[float, str]]]:
    """Labelled ticks at mantissa x 10^e and unlabelled minor ticks, in log10 view units."""
    major: list[tuple[float, str]] = []
    minor: list[tuple[float, str]] = []
    for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for m in range(1, 10):
            v = m * 10.0**e
            if lo <= v <= hi:
                (major if m in mantissas else minor).append((math.log10(v), f"{v:g}" if m in mantissas else ""))
    return [major, minor]


def term_color(term: ModalTerm, roots_seen: int) -> str:
    if term.mode is not None:
        return colors.mode[(term.mode - 1) % len(colors.mode)]
    return colors.roots[roots_seen % len(colors.roots)]


def swatch(color: str) -> QtGui.QIcon:
    pix = QtGui.QPixmap(12, 12)
    pix.fill(QtGui.QColor(color))
    return QtGui.QIcon(pix)


class FrfGrid(pg.GraphicsLayoutWidget):
    """N x N small plots of H_jk: row j = response x_j, column k = force at m_k."""

    cell_clicked = QtCore.Signal(int, int, bool)  # output j, input k, add to inputs (Ctrl held)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.n = 0
        self.cells: list[list[pg.PlotItem]] = []
        self.full: list[list[pg.PlotDataItem]] = []
        self.total: list[list[pg.PlotDataItem]] = []
        self.terms: list[list[list[pg.PlotDataItem]]] = []
        self.left_axes: list[pg.AxisItem] = []
        self.bottom_axes: list[pg.AxisItem] = []
        self.corner: pg.LabelItem | None = None
        self.scene().sigMouseClicked.connect(self._on_click)

    def build(self, n: int, n_terms: int, force: bool = False) -> None:
        """Lay out the grid for n masses and n_terms modal terms per cell (again if `force`)."""
        if not force and n == self.n and self.terms and len(self.terms[0][0]) == n_terms:
            return
        self.clear()
        self.n = n
        font = QtGui.QFont()
        font.setPointSize(7)
        self.corner = pg.LabelItem("", size="8pt")
        self.addItem(self.corner, row=0, col=0, colspan=2)
        for k in range(n):
            self.addItem(pg.LabelItem(f"<b>F at m{k + 1}</b>", size="9pt", color=colors.mass[k]), row=0, col=k + 2)
        self.cells, self.full, self.total, self.terms, self.left_axes = [], [], [], [], []
        thin = pg.mkPen(colors.cell_border, width=1)
        for j in range(n):
            self.addItem(pg.LabelItem(f"<b>x{j + 1}</b>", size="9pt", color=colors.mass[j]), row=j + 1, col=0)
            cells: list[pg.PlotItem] = []
            full: list[pg.PlotDataItem] = []
            total: list[pg.PlotDataItem] = []
            terms: list[list[pg.PlotDataItem]] = []
            for k in range(n):
                p = pg.PlotItem()
                p.hideAxis("left")
                p.hideAxis("bottom")
                p.hideButtons()
                p.setMenuEnabled(False)
                p.setMouseEnabled(x=False, y=False)
                p.setLogMode(x=True, y=False)
                vb = p.getViewBox()
                vb.setBorder(thin)
                if j == k:
                    vb.setBackgroundColor(colors.diagonal_tint)
                if cells or self.cells:
                    p.setXLink(self.cells[0][0] if self.cells else cells[0])
                    p.setYLink(self.cells[0][0] if self.cells else cells[0])
                self.addItem(p, row=j + 1, col=k + 2)
                # 1 px pens: up to a few hundred curves are drawn at once.
                terms.append([p.plot(pen=pg.mkPen(colors.grey, width=1)) for _ in range(n_terms)])
                full.append(p.plot(pen=pg.mkPen(colors.strong, width=1)))
                total.append(p.plot(pen=pg.mkPen(colors.force, width=1, style=QtCore.Qt.PenStyle.DashLine)))
                cells.append(p)
            self.cells.append(cells)
            self.full.append(full)
            self.total.append(total)
            self.terms.append(terms)
            axis = pg.AxisItem("left", linkView=cells[0].getViewBox())
            axis.setStyle(tickFont=font, tickTextOffset=2)
            axis.setWidth(44)
            axis.enableAutoSIPrefix(False)
            self.addItem(axis, row=j + 1, col=1)
            self.left_axes.append(axis)
        self.bottom_axes = []
        for k in range(n):
            axis = pg.AxisItem("bottom", linkView=self.cells[0][k].getViewBox())
            axis.setStyle(tickFont=font, tickTextOffset=2)
            axis.setLogMode(x=True, y=False)
            self.addItem(axis, row=n + 1, col=k + 2)
            self.bottom_axes.append(axis)
        self.addItem(pg.LabelItem("Frequency [Hz]", size="8pt"), row=n + 2, col=2, colspan=n)

    def set_curves(
        self,
        f: np.ndarray,
        q: Quantity,
        full: np.ndarray,
        total: np.ndarray,
        terms: list[tuple[str, np.ndarray] | None],
    ) -> None:
        """Real (F, n, n) arrays of the plotted quantity; `terms` entries are (color, values) or None (hidden)."""
        log_y = q is Quantity.MAGNITUDE
        for row in self.cells:
            for p in row:
                p.setLogMode(x=True, y=log_y)
        for axis in self.left_axes:
            axis.setLogMode(x=False, y=log_y)
            set_phase_ticks(axis, q is Quantity.PHASE)
        ticks = log_ticks(f[0], f[-1], (1, 2, 5) if self.n <= 4 else (1,))
        for axis in self.bottom_axes:
            axis.setTicks(ticks)
        for j in range(self.n):
            for k in range(self.n):
                self.full[j][k].setData(f, full[:, j, k], connect="finite")
                self.total[j][k].setData(f, total[:, j, k], connect="finite")
                for curve, term in zip(self.terms[j][k], terms):
                    curve.setVisible(term is not None)
                    if term is not None:
                        color, values = term
                        curve.setPen(pg.mkPen(color, width=1))
                        curve.setData(f, values[:, j, k], connect="finite")
        lo, hi = y_range(q, [full, total])
        self.cells[0][0].setXRange(math.log10(f[0]), math.log10(f[-1]), padding=0)
        self.cells[0][0].setYRange(lo, hi, padding=0)
        assert self.corner is not None, "laid out by build"
        self.corner.setText(f"<i>{q.axis_label}</i>", size="8pt")

    def set_selection(self, output: int, inputs: set[int]) -> None:
        thin = pg.mkPen(colors.cell_border, width=1)
        bold = pg.mkPen(colors.force, width=3)
        for j, row in enumerate(self.cells):
            for k, p in enumerate(row):
                p.getViewBox().setBorder(bold if j == output and k in inputs else thin)

    def _on_click(self, event) -> None:
        pos = event.scenePos()
        for j, row in enumerate(self.cells):
            for k, p in enumerate(row):
                if p.sceneBoundingRect().contains(pos):
                    add = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
                    self.cell_clicked.emit(j, k, add)
                    return


class FrfDetail(QtWidgets.QWidget):
    """The selected term large: its parts, their sum, and the full solution."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.header = QtWidgets.QLabel()
        self.header.setWordWrap(True)
        self.header.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.plots = pg.GraphicsLayoutWidget()
        self.top = self.plots.addPlot(row=0, col=0)
        self.bottom = self.plots.addPlot(row=1, col=0)
        self.bottom.setXLink(self.top)
        self.bottom.setLabel("bottom", "Frequency", units="Hz")
        for p in (self.top, self.bottom):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis("left").enableAutoSIPrefix(False)
        self.legend = add_legend(self.top, offset=(-5, 5))
        self.plots.ci.layout.setRowStretchFactor(0, 3)
        self.plots.ci.layout.setRowStretchFactor(1, 2)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addWidget(self.plots, 1)

    def set_data(
        self,
        f: np.ndarray,
        q: Quantity,
        full: np.ndarray,
        total: np.ndarray,
        terms: list[tuple[str, str, np.ndarray]],
        lines: list[tuple[float, str]],
        header: str,
        worst: tuple[float, float, float] | None = None,
    ) -> None:
        """Complex (F,) curves; `terms` are (name, color, values); `lines` are (frequency, color).

        `worst` = (low, high, peak) frequencies of the band to shade, where the sum
        differs most from the full solution.
        """
        self.header.setText(header)
        first = q if q in (Quantity.MAGNITUDE, Quantity.REAL) else q.partner
        self.legend.clear()
        for p, pq in ((self.top, first), (self.bottom, first.partner)):
            p.clear()
            log_y = pq is Quantity.MAGNITUDE
            p.setLogMode(x=True, y=log_y)
            p.setLabel("left", pq.axis_label)
            set_phase_ticks(p.getAxis("left"), pq is Quantity.PHASE)
            for fn, color in lines:
                p.addItem(
                    pg.InfiniteLine(
                        pos=math.log10(fn),
                        angle=90,
                        pen=pg.mkPen(color, width=1, style=QtCore.Qt.PenStyle.DotLine),
                    )
                )
            if worst is not None:
                lo_f, hi_f, peak_f = worst
                brush = pg.mkColor(colors.force)
                brush.setAlpha(40)
                band = pg.LinearRegionItem(
                    (math.log10(lo_f), math.log10(hi_f)),
                    movable=False,
                    brush=brush,
                    pen=pg.mkPen(None),
                )
                band.setZValue(-10)
                p.addItem(band)
                p.addItem(
                    pg.InfiniteLine(
                        pos=math.log10(peak_f),
                        angle=90,
                        pen=pg.mkPen(colors.force, width=1.5, style=QtCore.Qt.PenStyle.DashLine),
                    )
                )
            for name, color, values in terms:
                p.plot(f, part(values, pq), pen=pg.mkPen(color, width=1.5), name=name, connect="finite")
            p.plot(f, part(full, pq), pen=pg.mkPen(colors.strong, width=2.5), name="Full solution", connect="finite")
            p.plot(
                f,
                part(total, pq),
                pen=pg.mkPen(colors.force, width=2, style=QtCore.Qt.PenStyle.DashLine),
                name="Sum of the terms shown",
                connect="finite",
            )
            lo, hi = y_range(pq, [part(full, pq), part(total, pq)])
            p.setYRange(lo, hi, padding=0)
        ticks = log_ticks(f[0], f[-1], (1, 2, 5))
        for p in (self.top, self.bottom):
            p.getAxis("bottom").setTicks(ticks)
        self.top.setXRange(math.log10(f[0]), math.log10(f[-1]), padding=0)


@dataclass(frozen=True)
class FrfTrial:
    """A chain (the uniform 4-mass one, its dampers changed by `damping`) and what to show of its H."""

    name: str
    note: str
    output: int  # mass index of the response
    inputs: tuple[int, ...]  # mass indices of the forces
    quantity: Quantity = Quantity.MAGNITUDE
    unticked: tuple[int, ...] = ()  # modes (1-based) left out of the sum
    damping: tuple[float, ...] = ()  # c_i, else the default 2 N·s/m each
    expansion: Expansion = Expansion.EXACT

    def system(self) -> ChainSystem:
        system = ChainSystem.uniform(4)
        if self.damping:
            system.damping[:] = self.damping
        return system


TRIALS = (
    FrfTrial(
        "Antiresonances at a driving point (H22)",
        "Force and response both at m2. Peaks and dips alternate, and the phase falls 180° at each peak and "
        "climbs back at each dip. Undamped, the dips would be at 1.97 Hz and 5.15 Hz (m3 and m4 vibrating on "
        "their own with m2 held) and 4.50 Hz (m1 between the ground and m2). With the dampers the one at "
        "5.15 Hz, squeezed between the peaks at 4.88 and 5.98 Hz, is washed out to a shoulder.",
        output=1,
        inputs=(1,),
    ),
    FrfTrial(
        "A force at a node (m3 in mode 2)",
        "Mode 2 has a node at m3 (its shape is 0.58, 0.58, 0, −0.58). Pushing there can't excite it, and "
        "measuring there can't see it, so H33, and the whole of row 3 and column 3, have no peak at 3.18 Hz. "
        "Click another cell in row 3: still no mode 2.",
        output=2,
        inputs=(2,),
    ),
    FrfTrial(
        "End to end (H14): no antiresonances",
        "Force at m4, response at m1. Between m1 and m4 there is no chain left outside, so there are no "
        "dips, and the phase keeps falling by 180° at every peak, 720° in all. Compare H22's phase, which "
        "swings back up at each dip.",
        output=0,
        inputs=(3,),
        quantity=Quantity.PHASE,
    ),
    FrfTrial(
        "Truncate the high modes (H11)",
        "Modes 3 and 4 unticked, the real part shown. Near modes 1 and 2 the sum is close, but below them it "
        "sits low: modes 3 and 4 hold 24% of m1's static compliance (their residual flexibility), and the "
        "header lists each mode's share. Tick mode 3 back and the offset mostly closes.",
        output=0,
        inputs=(0,),
        quantity=Quantity.REAL,
        unticked=(3, 4),
    ),
    FrfTrial(
        "Leave out mode 1 (H44)",
        "Only mode 1 unticked. Above 1.1 Hz the missing term is mass-like, −φφᵀ/ω², so the sum is off by "
        "an amount that grows towards low frequency as 1/ω²: the residual mass. At ω = 0 the error is mode "
        "1's 89% share of the static compliance.",
        output=3,
        inputs=(3,),
        unticked=(1,),
    ),
    FrfTrial(
        "Two forces at once (m3 and m4)",
        "Unit forces at m3 and m4, in phase, the response at m4: the sum of H43 and H44. At mode 1 (1.11 Hz) "
        "m3 and m4 move together in the mode, the forces help each other and m4 moves almost twice as far. "
        "At mode 3 (4.88 Hz) they move in opposite directions, the forces fight, and the sum is smaller than "
        "either term alone.",
        output=3,
        inputs=(2, 3),
    ),
    FrfTrial(
        "Non-proportional damping: the classical sum misses",
        "c1 = 15 N·s/m, the other dampers at 2, with Modal terms → Classical. Even with every mode ticked the "
        "sum misses the full solution, most where two modes' peaks overlap, because the classical terms "
        "drop ΦᵀCΦ's coupling. Switch to Exact (complex modes): the sum lies on the full solution again.",
        output=0,
        inputs=(0,),
        damping=(15.0, 2.0, 2.0, 2.0),
        expansion=Expansion.CLASSICAL,
    ),
)


# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


def static_shares(j: int, k: int) -> np.ndarray:
    """Each mode's share of the uniform 4-mass chain's static compliance H_jk(0) = (K⁻¹)_jk, in %."""
    system = ChainSystem.uniform(4)
    flexibility = np.linalg.inv(system.matrices()[2])[j, k]
    return np.array(
        [
            100.0 * m.shape_mass_normalized[j] * m.shape_mass_normalized[k] / m.omega_n**2 / flexibility
            for m in modal_analysis(system).modes
        ]
    )


def _static_compliance(rng: np.random.Generator) -> Question:
    j, k = sorted(int(v) for v in rng.integers(0, 4, 2))
    h = (j + 1) / 400.0
    return Question(
        f"The default chain: four 1 kg masses on four 400 N/m springs, the first to the ground. A steady force at "
        f"m{k + 1}: how far does m{j + 1} move per newton?",
        1e3 * h,
        f"Statically only the springs between the ground and the force stretch, each by the whole force; m{j + 1} "
        f"sits above {j + 1} of them: (K⁻¹)<sub>{j + 1}{k + 1}</sub> = min(j, k)/k = {j + 1}/400 = "
        f"{number(1e3 * h)} mm/N (Maxwell: the same either way round).",
        unit="mm/N",
        setup=dict(output=j, inputs=(k,), quantity=Quantity.REAL),
    )


def _mode_share(rng: np.random.Generator) -> Question:
    r = int(pick(rng, (1, 2, 3)))
    j = int(pick(rng, (0, 3)))
    share = float(static_shares(j, j)[r - 1])
    return Question(
        f"In the default four-mass chain, what share of m{j + 1}'s static compliance H<sub>{j + 1}{j + 1}</sub>(0) "
        f"does mode {r} hold?",
        share,
        f"H<sub>jj</sub>(0) = Σ φ<sub>jr</sub>²/ω<sub>r</sub>², so mode {r}'s share is "
        f"φ<sub>{j + 1}{r}</sub>²/ω<sub>{r}</sub>² over (K⁻¹)<sub>{j + 1}{j + 1}</sub> = {j + 1}/400: "
        f"{number(share)} %. The large plot's header lists every mode's.",
        unit="%",
        setup=dict(output=j, inputs=(j,), quantity=Quantity.REAL, expansion=Expansion.CLASSICAL),
    )


def _accelerance_limit(rng: np.random.Generator) -> Question:
    m = pick(rng, (0.5, 1, 2, 5, 10))
    return Question(
        f"A force pushes a {m:g} kg mass in a chain of springs and masses. What does its driving-point accelerance "
        "level off at, far above every mode?",
        1.0 / m,
        f"Far above the modes the springs can't keep up and the mass is pushed alone: ẍ = F/m, so −ω²H<sub>jj</sub> "
        f"→ 1/m = {number(1 / m)} 1/kg.",
        unit="1/kg",
    )


def _antiresonance(rng: np.random.Generator) -> Question:
    m2 = pick(rng, (0.5, 1, 2))
    k2 = pick(rng, (100, 200, 400, 800))
    f = math.sqrt(k2 / m2) / (2 * math.pi)
    return Question(
        f"Two masses: m<sub>1</sub> on a spring to the ground, and m<sub>2</sub> = {m2:g} kg on k<sub>2</sub> = "
        f"{k2:g} N/m to m<sub>1</sub>. At what frequency does the driving-point FRF H<sub>11</sub> have its "
        "antiresonance?",
        f,
        f"At an antiresonance m<sub>1</sub> stands still under the force: the chain beyond it vibrates on its own, "
        f"m<sub>2</sub> on k<sub>2</sub> with m<sub>1</sub> held, √(k<sub>2</sub>/m<sub>2</sub>)/2π = {number(f)} "
        "Hz. It's an absorber, whatever m<sub>1</sub> and k<sub>1</sub> are.",
        unit="Hz",
    )


def _count_dips(rng: np.random.Generator) -> Question:
    n = int(pick(rng, (3, 4, 5, 6)))
    while True:
        j, k = sorted(int(v) for v in rng.integers(1, n + 1, 2))
        if (j - 1) + (n - k) > 0:
            break
    count = (j - 1) + (n - k)
    return Question(
        f"An undamped chain of {n} masses, the first on a spring to the ground. How many antiresonances does "
        f"H<sub>{j}{k}</sub> have?",
        count,
        f"One for each mass outside the stretch from m{j} to m{k}: the pieces of chain beyond it, with m{j} and m{k} "
        f"held, have their own natural frequencies. ({j} − 1) + ({n} − {k}) = {count}.",
        tolerance=0.01,
        setup=dict(output=j - 1, inputs=(k - 1,)) if n == 4 else None,
    )


def _mobility(rng: np.random.Generator) -> Question:
    h = pick(rng, (0.1, 0.2, 0.5, 1, 2))
    f = pick(rng, (2, 5, 10, 20))
    v = 2 * math.pi * f * 1e-3 * h
    return Question(
        f"A receptance measures {h:g} mm/N at {f:g} Hz. What is the mobility there?",
        v,
        f"For harmonic motion velocity is iω times displacement: |iωH| = 2π × {f:g} × {number(1e-3 * h)} = "
        f"{number(v)} m/(N·s).",
        unit="m/(N·s)",
    )


def _reciprocity(rng: np.random.Generator) -> Question:
    j, k = sorted(int(v) for v in rng.choice(4, 2, replace=False))
    return Question(
        f"m{k + 1} is pushed and m{j + 1} measured, then m{j + 1} pushed and m{k + 1} measured. How do the two FRFs "
        "compare?",
        0,
        "They are identical, magnitude and phase (Maxwell–Betti): M, C and K are symmetric, so H is.",
        choices=("Identical", "The same magnitude, phases opposite", "Different"),
        setup=dict(output=j, inputs=(k,)),
        predict=True,
    )


def _truncated_static(rng: np.random.Generator) -> Question:
    return Question(
        "A driving-point FRF is summed from every mode but the highest. At low frequency, how does the sum compare "
        "with the full FRF?",
        0,
        "Too small: at a driving point every mode's term starts positive (φ<sub>jr</sub>²/ω<sub>r</sub>²), so leaving "
        "one out loses its share of the static compliance, its residual flexibility.",
        choices=("Too small", "Too large", "The same"),
        setup=dict(output=0, inputs=(0,), quantity=Quantity.REAL, unticked=(4,), expansion=Expansion.CLASSICAL),
        predict=True,
    )


def _driving_point_dips(rng: np.random.Generator) -> Question:
    return Question(
        "Between two neighbouring resonances of a lightly damped driving-point FRF, what is there?",
        0,
        "Always an antiresonance: both modes' constants φ<sub>jr</sub>² are positive, and between the peaks the "
        "lower is already mass-like and the upper still spring-like, so they cancel somewhere.",
        choices=("Always an antiresonance", "Sometimes one, sometimes not", "Never one"),
        setup=dict(output=1, inputs=(1,)),
        predict=True,
    )


def _truncation_spec(rng: np.random.Generator) -> Question:
    cells = ((0, 0), (1, 1), (0, 1), (1, 3), (2, 2), (3, 3), (0, 3))
    while True:
        j, k = cells[int(rng.integers(len(cells)))]
        most = pick(rng, (2, 5, 10))
        shares = static_shares(j, k)
        best = min(
            (subset for size in range(1, 5) for subset in itertools.combinations(range(4), size)),
            key=lambda c: (abs(100.0 - shares[list(c)].sum()) > most, len(c)),
        )
        if len(best) < 4:
            break

    def goal(t: FrfTrial):
        kept_modes = [r for r in range(4) if r + 1 not in t.unticked]
        error = abs(100.0 - float(shares[kept_modes].sum()))
        return (
            Requirement(
                f"The cell H<sub>{j + 1}{k + 1}</sub>",
                f"H{t.output + 1}{'+'.join(str(i + 1) for i in t.inputs)}",
                t.output == j and t.inputs == (k,),
            ),
            at_most("Modes ticked", len(kept_modes), len(best)),
            at_most("The static compliance's error", error, most, "%"),
        )

    kept_text = ", ".join(str(r + 1) for r in best)
    return Question(
        f"Show H<sub>{j + 1}{k + 1}</sub> of the default chain and untick as many modes as you can while the ticked "
        f"ones' sum still gets its static compliance within {most:g} % (the large plot's header gives the error).",
        0,
        "Each mode holds φ<sub>jr</sub>φ<sub>kr</sub>/ω<sub>r</sub>² of it: "
        + ", ".join(f"mode {r + 1} {number(v)} %" for r, v in enumerate(shares))
        + f". Keeping {len(best)} (mode{'s' if len(best) > 1 else ''} {kept_text}) is the fewest that does it"
        + (": off the diagonal the shares have signs, and some cancel." if j != k else "."),
        setup=dict(output=j, inputs=(k,), quantity=Quantity.REAL, expansion=Expansion.CLASSICAL),
        goal=goal,
        design=dict(
            output=j,
            inputs=(k,),
            quantity=Quantity.REAL,
            expansion=Expansion.CLASSICAL,
            unticked=tuple(r + 1 for r in range(4) if r not in best),
        ),
    )


PROBLEMS = tuple(
    Problem(key, title, "frf_page", f"See {page_link('frf_page', anchor, section)}{more}.", make)
    for key, title, anchor, section, more, make in (
        (
            "static-compliance",
            "Static compliance",
            "truncation",
            "Truncation",
            ": the static compliance",
            _static_compliance,
        ),
        ("mode-share", "A mode's share", "truncation", "Truncation", "", _mode_share),
        ("accelerance-limit", "Accelerance far above", "three-forms", "Three forms", "", _accelerance_limit),
        ("antiresonance", "An antiresonance", "sum-of-modes", "Every term is a sum of modes", "", _antiresonance),
        ("count-dips", "Counting antiresonances", "sum-of-modes", "Every term is a sum of modes", "", _count_dips),
        ("mobility", "Mobility from receptance", "three-forms", "Three forms", "", _mobility),
        ("reciprocity", "Predict: swap force and sensor", "receptance-matrix", "The FRF matrix", "", _reciprocity),
        ("truncated-static", "Predict: the highest mode left out", "truncation", "Truncation", "", _truncated_static),
        (
            "driving-point-dips",
            "Predict: between two peaks",
            "sum-of-modes",
            "Every term is a sum of modes",
            "",
            _driving_point_dips,
        ),
        (
            "truncation-spec",
            "Spec: as few modes as will do",
            "truncation",
            "Truncation",
            ": each mode's share of the static compliance",
            _truncation_spec,
        ),
    )
)


class FrfMatrixPage(QtWidgets.QWidget):
    """Controls on the left, the N x N grid in the centre, the selected term and theory on the right."""

    dof_requested = QtCore.Signal(int)  # the user changed N here; the main window owns the system
    system_requested = QtCore.Signal(object)  # a trial's chain, for the Simulation page
    edit_parameters = QtCore.Signal()  # the user wants the Simulation page's parameter panel

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system: ChainSystem | None = None
        self.result: ModalResult | None = None
        self.freqs = np.zeros(0)
        self.H = np.zeros((0, 0, 0), dtype=complex)  # (F, n, n) full solution
        self.terms: list[ModalTerm] = []
        self.term_H = np.zeros((0, 0, 0, 0), dtype=complex)  # (R, F, n, n)
        self.static: np.ndarray | None = None  # K^-1, the static compliance; None if the chain can move as a rigid body
        self.term_static = np.zeros((0, 0, 0))  # (R, n, n) each term at w = 0
        self.term_colors: list[str] = []
        self.expansion_note = ""
        self.output = 0
        self.inputs: set[int] = {0}
        self._dirty = False
        self._unchecked: set[str] = set()  # remembered by label across recomputes
        self._rebuild_grid = False  # the theme changed: build the grid again

        # --- controls
        trials = QtWidgets.QGroupBox("Things to try")
        tv = QtWidgets.QVBoxLayout(trials)
        self.trial = TrialCombo(
            [t.name for t in TRIALS],
            "<p>Each sets up the chain and the plots to show one idea from the Theory tab. "
            "Loading one replaces the Simulation page's chain.</p>",
        )
        self.trial.picked.connect(self.load_trial)
        tv.addWidget(self.trial)
        self.trial_note = QtWidgets.QLabel()
        self.trial_note.setWordWrap(True)
        mute(self.trial_note)
        self.trial_note.setVisible(False)
        tv.addWidget(self.trial_note)

        box = QtWidgets.QGroupBox("FRF matrix H(ω) = (K − ω²M + iωC)⁻¹")
        form = form_layout(box)
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setToolTip("Same chain as the Simulation page; changing it here changes it there too.")
        self.dof.valueChanged.connect(self.dof_requested)
        form.addRow("Number of masses:", self.dof)
        self.quantity = QtWidgets.QComboBox()
        for q in Quantity:
            self.quantity.addItem(q.value, q)
        self.quantity.setToolTip(
            "<p>What each small plot shows. The large plot on the right shows the pair: "
            "magnitude with phase, or real with imaginary part.</p>"
            "<p>The modal terms add as complex numbers, so their <b>real and imaginary parts add "
            "directly</b>; their magnitudes and phases do not.</p>"
        )
        self.quantity.currentIndexChanged.connect(self.redraw)
        form.addRow("Show:", self.quantity)
        self.expansion = QtWidgets.QComboBox()
        for e in Expansion:
            self.expansion.addItem(e.value, e)
        self.expansion.setToolTip(
            "<p><b>Exact:</b> each eigenvalue pair of the state-space model contributes "
            "R/(iω − λ) + R*/(iω − λ*). The sum of all of them is the full solution for any "
            "damping.</p>"
            "<p><b>Classical:</b> each undamped mode contributes φφᵀ/(ω<sub>r</sub>² − ω² + "
            "2iζ<sub>r</sub>ω<sub>r</sub>ω). With non-proportional damping even the sum of all modes "
            "misses the full solution, because the coupling terms of ΦᵀCΦ are dropped.</p>"
        )
        self.expansion.currentIndexChanged.connect(self._recompute_terms)
        form.addRow("Modal terms:", self.expansion)
        link = QtWidgets.QLabel("Masses, springs and dampers are edited on the <a href='#sim'>Simulation page</a>.")
        mute(link)
        link.setWordWrap(True)
        link.linkActivated.connect(self.edit_parameters)
        form.addRow(link)

        self.mode_box = QtWidgets.QGroupBox("Modes included in the sum")
        mv = QtWidgets.QVBoxLayout(self.mode_box)
        self.mode_list = QtWidgets.QVBoxLayout()
        mv.addLayout(self.mode_list)
        buttons = QtWidgets.QHBoxLayout()
        self.all_button = QtWidgets.QPushButton("All")
        self.none_button = QtWidgets.QPushButton("None")
        self.all_button.clicked.connect(weak.slot(self, lambda self: self._check_all(True)))
        self.none_button.clicked.connect(weak.slot(self, lambda self: self._check_all(False)))
        buttons.addWidget(self.all_button)
        buttons.addWidget(self.none_button)
        mv.addLayout(buttons)
        self.show_terms = QtWidgets.QCheckBox("Draw each mode's term in the grid too")
        self.show_terms.setToolTip("The large plot always draws them. In the grid they can crowd the small plots.")
        self.show_terms.toggled.connect(self.redraw)
        mv.addWidget(self.show_terms)
        self.mode_note = QtWidgets.QLabel()
        self.mode_note.setWordWrap(True)
        mute(self.mode_note)
        mv.addWidget(self.mode_note)
        self.mode_checks: list[QtWidgets.QCheckBox] = []

        io_box = QtWidgets.QGroupBox("Response and forces")
        iv = QtWidgets.QVBoxLayout(io_box)
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(QtWidgets.QLabel("Response of:"))
        self.output_combo = QtWidgets.QComboBox()
        self.output_combo.currentIndexChanged.connect(self._on_output)
        out_row.addWidget(self.output_combo, 1)
        iv.addLayout(out_row)
        iv.addWidget(QtWidgets.QLabel("Unit forces (in phase) at:"))
        self.input_grid = QtWidgets.QGridLayout()
        iv.addLayout(self.input_grid)
        self.input_checks: list[QtWidgets.QCheckBox] = []
        io_note = QtWidgets.QLabel(
            "Click a plot to select it. Ctrl+click more plots in the same row, or tick more "
            "forces, to apply several forces at once: x<sub>j</sub> = Σ<sub>k</sub> H<sub>jk</sub>F<sub>k</sub>, "
            "the sum along the row."
        )
        io_note.setWordWrap(True)
        mute(io_note)
        iv.addWidget(io_note)

        for combo in (self.trial, self.quantity, self.expansion, self.output_combo):
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(trials)
        lv.addWidget(box)
        lv.addWidget(self.mode_box)
        lv.addWidget(io_box)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # --- centre: the grid
        self.grid = FrfGrid()
        self.grid.cell_clicked.connect(self._on_cell)
        self.grid_title = QtWidgets.QLabel()
        self.grid_title.setWordWrap(True)
        centre = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(centre)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.grid_title)
        cv.addWidget(self.grid, 1)

        # --- right: the selected term, and theory
        self.detail = FrfDetail()
        self.theory = QtWidgets.QTextBrowser()
        self.theory.setHtml(THEORY_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.detail, "Selected")
        self.tabs.addTab(self.theory, "Theory")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(centre)
        splitter.addWidget(self.tabs)
        splitter.setSizes([350, 800, 550])
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def apply_theme(self) -> None:
        """Swatches now; the grid, its curves and the large plot are redrawn when the page shows."""
        for k, c in enumerate(self.input_checks):
            c.setIcon(swatch(colors.mass[k]))
        self._rebuild_grid = True
        if self.system is not None:
            self._dirty = True
            if self.isVisible():
                self.refresh()

    # ------------------------------------------------------------ inputs
    def set_system(self, system: ChainSystem, result: ModalResult | None = None) -> None:
        """New chain parameters. Recomputed now if visible, else when the page is shown."""
        self.system = system
        self.result = result
        self._dirty = True
        if self.isVisible():
            self.refresh()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty:
            self.refresh()

    def load_trial(self, index: int) -> None:
        """Load TRIALS[index]: its chain (through the Simulation page), cells, quantity and modes."""
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(FrfTrial("", "", **setup))

    def practice_state(self) -> FrfTrial:
        """The page's cell, quantity, expansion and the modes left out, as a trial: what a practice spec checks."""
        assert self.system is not None, "the chain comes before any practice"
        return FrfTrial(
            "",
            "",
            self.output,
            tuple(sorted(self.inputs)),
            self.quantity.currentData(),
            tuple(sorted({t.mode for t in self.terms if t.label in self._unchecked and t.mode is not None})),
            tuple(float(c) for c in self.system.damping),
            self.expansion.currentData(),
        )

    def apply_trial(self, trial: FrfTrial) -> None:
        self.trial_note.setText(trial.note)
        self.trial_note.setVisible(True)
        self._unchecked = set()
        system = trial.system()
        self.system_requested.emit(system)
        if self.system is None or self.system.n != system.n or not np.allclose(self.system.damping, system.damping):
            self.set_system(system)  # nobody passed it back: show it here alone
        if self._dirty:
            self.refresh()
        self._unchecked = {t.label for t in self.terms if t.mode in trial.unticked}
        for widget, value in ((self.quantity, trial.quantity), (self.expansion, trial.expansion)):
            widget.setCurrentIndex(widget.findData(value))
        for t, c in zip(self.terms, self.mode_checks):
            c.blockSignals(True)
            c.setChecked(t.label not in self._unchecked)
            c.blockSignals(False)
        self.output, self.inputs = trial.output, set(trial.inputs)
        self._sync_io()
        self.redraw()
        self.tabs.setCurrentWidget(self.detail)

    def _on_cell(self, j: int, k: int, add: bool) -> None:
        if add and j == self.output:
            inputs = self.inputs ^ {k}
            self.inputs = inputs or {k}
        else:
            self.output, self.inputs = j, {k}
        self._sync_io()
        self.redraw()

    def _on_output(self, j: int) -> None:
        if j >= 0 and j != self.output:
            self.output = j
            self.redraw()

    def _on_input_toggled(self) -> None:
        inputs = {k for k, c in enumerate(self.input_checks) if c.isChecked()}
        if not inputs:  # keep at least one force
            self._sync_io()
            return
        self.inputs = inputs
        self.redraw()

    def _on_mode_toggled(self) -> None:
        self._unchecked = {t.label for t, c in zip(self.terms, self.mode_checks) if not c.isChecked()}
        self.redraw()

    def _check_all(self, on: bool) -> None:
        for c in self.mode_checks:
            c.blockSignals(True)
            c.setChecked(on)
            c.blockSignals(False)
        self._on_mode_toggled()

    # ----------------------------------------------------------- compute
    def refresh(self) -> None:
        if self.system is None:
            return
        self._dirty = False
        system = self.system
        n = system.n
        if self.result is None:
            self.result = modal_analysis(system)
        self.dof.blockSignals(True)
        self.dof.setValue(n)
        self.dof.blockSignals(False)
        if len(self.input_checks) != n:
            self._build_io(n)
        self.freqs = frequency_grid(self.result, FREQ_POINTS)
        self.H = frf_matrix(system, self.freqs)
        K = system.matrices()[2]
        self.static = np.linalg.inv(K) if np.linalg.cond(K) < 1e12 else None
        self._recompute_terms()

    def _build_io(self, n: int) -> None:
        self.output = n - 1
        self.inputs = {n - 1}
        self.output_combo.blockSignals(True)
        self.output_combo.clear()
        self.output_combo.addItems([f"m{j + 1}  (x{j + 1})" for j in range(n)])
        self.output_combo.blockSignals(False)
        for c in self.input_checks:
            self.input_grid.removeWidget(c)
            c.setParent(None)
            c.deleteLater()
        self.input_checks = []
        for k in range(n):
            c = QtWidgets.QCheckBox(f"m{k + 1}")
            c.setIcon(swatch(colors.mass[k]))
            c.toggled.connect(self._on_input_toggled)
            self.input_grid.addWidget(c, k // 4, k % 4)
            self.input_checks.append(c)
        self._sync_io()

    def _sync_io(self) -> None:
        self.output_combo.blockSignals(True)
        self.output_combo.setCurrentIndex(self.output)
        self.output_combo.blockSignals(False)
        for k, c in enumerate(self.input_checks):
            c.blockSignals(True)
            c.setChecked(k in self.inputs)
            c.blockSignals(False)

    def _recompute_terms(self) -> None:
        if self.system is None or self.result is None or not self.freqs.size:
            return
        exact = self.expansion.currentData() is Expansion.EXACT
        self.expansion_note = ""
        try:
            terms = modal_frf_terms(self.system, self.result, exact)
        except ValueError:
            terms = modal_frf_terms(self.system, self.result, exact=False)
            self.expansion_note = (
                "<b>The exact expansion is not available:</b> the chain has a repeated eigenvalue "
                "(a free, unsprung chain moves as a rigid body), which has no pole-residue form. "
                "Showing the classical expansion instead."
            )
        self.terms = terms
        self.term_H = np.stack([t.evaluate(self.freqs) for t in terms])
        with np.errstate(divide="ignore", invalid="ignore"):
            # Imaginary parts cancel at w = 0 (a pair's two terms are conjugates).
            self.term_static = np.stack([t.evaluate(np.zeros(1))[0].real for t in terms])
        self.term_colors = []
        roots = 0
        for t in terms:
            self.term_colors.append(term_color(t, roots))
            roots += t.mode is None
        self._build_mode_checks()
        self.grid.build(self.system.n, len(terms), force=self._rebuild_grid)
        self._rebuild_grid = False
        self.redraw()

    def _build_mode_checks(self) -> None:
        for c in self.mode_checks:
            self.mode_list.removeWidget(c)
            c.setParent(None)
            c.deleteLater()
        self.mode_checks = []
        for t, color in zip(self.terms, self.term_colors):
            detail = f"{t.fn_hz:.3g} Hz" if t.fn_hz > 0 else "non-oscillatory"
            if math.isfinite(t.zeta):
                detail += f", ζ {t.zeta:.3g}"
            c = QtWidgets.QCheckBox(f"{t.label}   ({detail})")
            c.setIcon(swatch(color))
            c.setChecked(t.label not in self._unchecked)
            c.toggled.connect(self._on_mode_toggled)
            self.mode_list.addWidget(c)
            self.mode_checks.append(c)

    def selected_terms(self) -> np.ndarray:
        return np.array([c.isChecked() for c in self.mode_checks], dtype=bool)

    def redraw(self) -> None:
        if self.system is None or not self.term_H.size:
            return
        n = self.system.n
        q = self.quantity.currentData()
        sel = self.selected_terms()
        total = self.term_H[sel].sum(axis=0) if sel.any() else np.zeros_like(self.H)
        grid_terms = [
            (color, part(h, q)) if (on and self.show_terms.isChecked()) else None
            for h, color, on in zip(self.term_H, self.term_colors, sel)
        ]
        self.grid.set_curves(self.freqs, q, part(self.H, q), part(total, q), grid_terms)
        self.grid.set_selection(self.output, self.inputs)
        self.grid_title.setText(self._grid_title(n, sel))

        j, inputs = self.output, sorted(self.inputs)
        full = self.H[:, j, inputs].sum(axis=1)
        partial = total[:, j, inputs].sum(axis=1)
        if len(inputs) == 1:
            k = inputs[0]
            parts = [
                (t.label, color, h[:, j, k])
                for t, h, color, on in zip(self.terms, self.term_H, self.term_colors, sel)
                if on
            ]
        else:
            parts = [(f"H{j + 1}{k + 1}: force at m{k + 1}", colors.mass[k], total[:, j, k]) for k in inputs]
        lines = [(t.fn_hz, color) for t, color in zip(self.terms, self.term_colors) if t.fn_hz >= self.freqs[0]]
        worst = worst_band(self.freqs, full, partial) if sel.any() else None
        header = self._detail_header(j, inputs, sel, full, partial, worst)
        self.detail.set_data(self.freqs, q, full, partial, parts, lines, header, worst)

    # -------------------------------------------------------------- text
    def _grid_title(self, n: int, sel: np.ndarray) -> str:
        used = int(sel.sum())
        return (
            f"<b>All {n}×{n} = {n * n} receptances H<sub>jk</sub> = x<sub>j</sub>/F<sub>k</sub></b> "
            f"(row: response, column: force). <span style='color:{colors.strong}'>{self._strong_name()}: "
            f"full solution</span>; <span style='color:{colors.force}'>dashed red: sum of the {used} of {len(sel)} modal "
            "terms ticked</span>. Shaded: driving point (j = k). The matrix is symmetric, "
            "H<sub>jk</sub> = H<sub>kj</sub>."
        )

    @staticmethod
    def _strong_name() -> str:
        return "White" if colors.dark else "Black"

    def _detail_header(
        self,
        j: int,
        inputs: list[int],
        sel: np.ndarray,
        full: np.ndarray,
        partial: np.ndarray,
        worst: tuple[float, float, float] | None,
    ) -> str:
        if len(inputs) == 1:
            k = inputs[0]
            kind = "driving point" if j == k else "transfer"
            title = (
                f"<b>H<sub>{j + 1}{k + 1}</sub> = x<sub>{j + 1}</sub> / F<sub>{k + 1}</sub></b> "
                f"({kind}): response of m{j + 1} to a force at m{k + 1}."
            )
            parts = (
                f"Coloured: each ticked mode's term. <span style='color:{colors.force}'>Dashed red</span> = "
                f"Σ over the ticked modes r of mode r's ({j + 1},{k + 1}) entry only; the other cells "
                "are responses to other forces."
            )
            assert self.system is not None, "drawn only once there is a chain"
            zeros = (j if j <= k else k) + (self.system.n - 1 - max(j, k))
            parts += (
                " Undamped, it has no antiresonances (see Theory)."
                if zeros == 0
                else f" Undamped, it has {zeros} antiresonance{'s' if zeros > 1 else ''}"
                f"{', counting any that coincide' if zeros > 1 else ''} (see Theory)."
            )
        else:
            names = ", ".join(f"m{k + 1}" for k in inputs)
            terms = " + ".join(f"H<sub>{j + 1}{k + 1}</sub>" for k in inputs)
            title = f"<b>x<sub>{j + 1}</sub> for unit in-phase forces at {names}</b> = {terms}."
            cols = ", ".join(f"({j + 1},{k + 1})" for k in inputs)
            parts = (
                "Coloured: each force's term (from the ticked modes). "
                f"<span style='color:{colors.force}'>Dashed red</span> = Σ over these forces and the "
                f"ticked modes r of mode r's entries {cols}: the sum along row {j + 1}."
            )
        err = float(np.abs(partial - full).max() / max(np.abs(full).max(), 1e-300))
        color = colors.good if err < 1e-6 else colors.fair if err < 0.05 else colors.poor
        amount = "none (equal to rounding error)" if err < 1e-9 else f"{100 * err:.3g}%"
        verdict = f"Largest difference from the full solution: <b style='color:{color}'>{amount}</b> of the peak"
        if worst is not None:
            verdict += (
                f", at <b>{worst[2]:.3g} Hz</b> (dashed red line). <span style='color:{colors.force}'>"
                f"Shaded: {worst[0]:.3g}–{worst[1]:.3g} Hz</span>, where the difference is at "
                f"least {WORST_FRACTION:.0%} of that."
            )
        else:
            verdict += "."
        if int(sel.sum()) == 0:
            verdict = "No modes ticked: the sum is zero."
        note = f"<br><span style='color:{colors.muted}'>{self.expansion_note}</span>" if self.expansion_note else ""
        return f"{title} {parts}<br>{verdict}<br>{self._static_text(j, inputs, sel)}{note}"

    def _static_text(self, j: int, inputs: list[int], sel: np.ndarray) -> str:
        """Static compliance (w = 0): the ticked modes' sum vs K^-1, and each mode's share."""
        if self.static is None:
            return (
                "<b>Static compliance</b> (ω = 0): none. K is singular, so the chain can move "
                "as a rigid body and a steady force has no static deflection."
            )
        full = float(self.static[j, inputs].sum())
        each = self.term_static[:, j, inputs].sum(axis=1)
        partial = float(each[sel].sum())
        err = (partial - full) / full if full else 0.0
        color = colors.good if abs(err) < 1e-6 else colors.fair if abs(err) < 0.05 else colors.poor
        amount = "0% (exact)" if abs(err) < 1e-9 else f"{100 * err:+.3g}%"
        shares = []
        for t, c, value, on in zip(self.terms, self.term_colors, each, sel):
            share = (
                "—"
                if not (full and np.isfinite(value))
                else "0%"
                if abs(value / full) < 1e-9
                else (f"{100 * value / full:.3g}%")
            )  # a node's share is round-off
            style = "" if on else f" style='color:{colors.grey}'"
            shares.append(f"<span style='color:{c}'>■</span><span{style}>{t.label} {share}</span>")
        return (
            f"<b>Static compliance</b> (ω = 0): K<sup>−1</sup> gives {full:.4g} m/N; the ticked "
            f"modes give {partial:.4g} m/N, error <b style='color:{color}'>{amount}</b>. "
            f"<span style='color:{colors.muted}'>Each mode's share (grey: not ticked):</span> {', '.join(shares)}."
        )


THEORY_HTML = load_notes("frf_matrix", trials=trials_html(TRIALS))
