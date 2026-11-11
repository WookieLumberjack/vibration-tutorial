"""Vibro-acoustics & SEA page: how a vibrating panel makes sound, and how much of it gets through.

Left: the panel (a simply supported rectangle in a baffle, from Beams & plates), its material and loss factor, the
mode to look at and where a point force drives it. Centre: the mode's shape with the drive point, and the
wavenumbers: the bending wave's k_b against sound's k = ω/c, crossing at the critical frequency, with the mode's own
k_x and k_y. Right: the radiation efficiency (the mode's against frequency; every mode's at its resonance with
their band average and Maidanik's), the point-driven panel (its input and radiated power, ⟨v²⟩ against the
infinite plate's), the transmission loss between two rooms (the mass law and the coincidence dip), and the
Theory. The page has its own model (core/radiation.py).

Working on it
-------------
Each tab is worked out only while it shows (`LazyTabs`); the driven panel's modal sum is the slow part (~0.5 s
for the default panel; `radiation._sigma_block` caches its σ_mn(f), which depends only on the panel's size). Trials' numbers are in
`tests/test_tutorial_numbers.py`, the core's in `tests/test_radiation.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.beams_plates import TWO_PI, Material, Plate, PlateMode, rect_shape
from ..core.practice import Problem, Question, Requirement, at_least, kept, number, pick, rounded
from ..core.radiation import (
    AIR,
    ModeSet,
    PointDrive,
    band_edges,
    bending_wavenumber,
    classify,
    coincidence_dip,
    critical_frequency,
    low_frequency_efficiency,
    maidanik,
    mass_law,
    modal_efficiency,
    mode_set,
    point_drive,
    radiation_loss_factor,
    resonant_average,
    third_octaves,
    transmission_loss,
    wave_efficiency,
)
from . import settle_plots
from .lesson import (
    Form,
    TrialBox,
    column,
    connect_all,
    fmt,
    mark,
    marker,
    note_label,
    plot,
    readout_label,
    set_quietly,
    stack,
    theory,
    three_panes,
    trials_html,
)
from .links import load_notes, page_link
from .style import colors
from .tab_groups import LazyTabs
from .theming import add_legend

TAB_RADIATION, TAB_DRIVEN, TAB_TRANSMISSION = range(3)
F_LOW = 20.0  # Hz, the plots' lowest frequency
F_TOP = 20e3  # and their highest
MAX_MODES = 600  # modes the driven panel sums (its top frequency comes down to keep it to these)
CURVE_POINTS = 120  # frequencies σ_mn(f) of one mode is worked out at
DRIVE_POINTS = 700  # frequencies the driven panel is worked out at
GRID = 121  # image points across the panel


@dataclass(frozen=True)
class AcousticsTrial:
    name: str
    note: str
    a: float = 0.6  # m
    b: float = 0.4  # m
    thickness: float = 3.0  # mm
    E: float = 210.0  # GPa
    rho: float = 7850.0  # kg/m³
    nu: float = 0.3
    loss: float = 0.01  # η
    m: int = 1
    n: int = 1
    drive_x: float = 0.35  # the drive point, as fractions of a and b
    drive_y: float = 0.33
    tab: int = TAB_RADIATION


GLASS: dict[str, float] = dict(E=70.0, rho=2500.0, nu=0.22)  # a practice setup's material

TRIALS = (
    AcousticsTrial(
        "Coincidence",
        "A steel panel 600 × 400 mm, 3 mm thick. Its bending waves travel at √ω (D/ρh)<sup>¼</sup>, slower at low "
        "frequency than sound's 343 m/s and faster at high: the two meet at the critical frequency "
        "f<sub>c</sub> = (c²/2π)√(ρh/D) = 3.99 kHz (centre, bottom: k<sub>b</sub> crosses k). Above it, a bending "
        "wave sends sound off at the angle where its trace matches, sin θ = k/k<sub>b</sub>; below it, its sound "
        "cancels, all but at the edges and corners.",
    ),
    AcousticsTrial(
        "A corner mode",
        "Mode (1, 1) at 66.6 Hz. Sound's wavelength there, 5.2 m, dwarfs the panel: the whole panel pumps in "
        "phase, like a small piston, and σ<sub>11</sub> = 0.037. Far below its resonance it follows "
        "32k²ab/π⁵ (dashed), so 20 dB less for each tenfold drop in frequency. Above, it passes 1 at 450 Hz, where sound's "
        "wavelength (0.76 m) is about the panel's diagonal, peaks at 1.28 near 720 Hz and settles to 1.",
    ),
    AcousticsTrial(
        "Even modes cancel",
        "Mode (2, 2), 266 Hz: four cells, each pushing the air its neighbours pull. At 100 Hz σ<sub>22</sub> is "
        "1.4 × 10<sup>−5</sup>, against (1, 1)'s 0.082 at the same frequency: an even index cancels to two more "
        "powers of k, and here both are even. The modes' dots (right, bottom) fall in three bands: odd–odd "
        "highest, even–even lowest.",
        m=2,
        n=2,
    ),
    AcousticsTrial(
        "An edge mode",
        "Mode (1, 3), 435 Hz: its half waves across b (k<sub>y</sub> = 3π/b = 23.6 rad/m) are shorter than "
        "sound's (k = 7.98 rad/m), its one across a (k<sub>x</sub> = 5.2) longer. Along a nothing cancels, so the "
        "two edges along a radiate: σ<sub>13</sub> = 0.059, the most of the first ten modes. The averages "
        "(right, bottom) are mostly edge modes from a few hundred hertz to f<sub>c</sub>, which is why "
        "Maidanik's formula grows with the perimeter.",
        m=1,
        n=3,
    ),
    AcousticsTrial(
        "A point-driven panel",
        "A 1 N force at (0.21 m, 0.13 m), η = 0.01 (Driven panel tab). At each resonance the input power peaks; "
        "averaged over a band it settles to the infinite plate's F²/16√(Dρh), here 565 µW, from a few hundred "
        "hertz up, and ⟨v²⟩ to the energy that input sustains, P/ωηM. The radiated power stays far below: 18 dB "
        "in the 1 kHz band, where σ ≈ 0.054 makes the radiation's own loss factor ρcσ/ωm″ = 1.5 × 10<sup>−4</sup>, "
        "a 66th of η; 11 dB at 6.3 kHz, above f<sub>c</sub>, where σ ≈ 1.7. Sound is a small leak from a panel's "
        "energy.",
        tab=TAB_DRIVEN,
    ),
    AcousticsTrial(
        "The mass law",
        "Between two rooms (Transmission loss tab), the panel's mass holds the sound back: at normal incidence "
        "TL = 10 log[1 + (ωm″/2ρc)²], 39.0 dB at 500 Hz for 23.6 kg/m². Sound from every direction (field "
        "incidence, to 78°) gets through more easily: 33.8 dB. Double the mass or the frequency and the TL rises "
        "6 dB, until the coincidence dip at f<sub>c</sub>.",
        tab=TAB_TRANSMISSION,
    ),
    AcousticsTrial(
        "Thicker glass, a dip in the speech band",
        "A window of 10 mm glass, 1.2 × 1 m: f<sub>c</sub> = 1.20 kHz, in the middle of speech, and its TL "
        "dips to 19.2 dB at 1.28 kHz, against 39.5 dB on the mass law at 500 Hz. Thinner glass puts the dip "
        "higher (4 mm: f<sub>c</sub> = 2.99 kHz) but has less mass. f<sub>c</sub>m″ is the material's own "
        "constant, so the dip is 19.2 dB deep at any thickness of glass with η = 0.01: laminated glass, with a "
        "damping layer, fills it in.",
        a=1.2,
        b=1.0,
        thickness=10.0,
        tab=TAB_TRANSMISSION,
        E=70.0,
        rho=2500.0,
        nu=0.22,
    ),
    AcousticsTrial(
        "Damping at coincidence",
        "The steel panel with η = 0.1 instead of 0.01. Below f<sub>c</sub> the TL hardly changes (mass-controlled); "
        "at and above it, only damping holds the sound back: the dip's bottom rises from 26.1 to 37.5 dB, and at "
        "2f<sub>c</sub> the TL is 51.0 dB against 41.3 dB, 10 log 10 = 10 dB more. On the Driven panel tab, "
        "damping lowers ⟨v²⟩'s resonant peaks by 20 dB but its band average only by 10 dB, and leaves the band's "
        "input power as it was.",
        loss=0.1,
        tab=TAB_TRANSMISSION,
    ),
)

# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.
STEEL = Material()
GLASS_MATERIAL = Material(70e9, 0.22, 2500.0)
ALUMINIUM = Material(70e9, 0.33, 2700.0)
MATERIALS = (
    ("steel", STEEL, "E = 210 GPa, ν = 0.3, ρ = 7850 kg/m³"),
    ("aluminium", ALUMINIUM, "E = 70 GPa, ν = 0.33, ρ = 2700 kg/m³"),
    ("glass", GLASS_MATERIAL, "E = 70 GPa, ν = 0.22, ρ = 2500 kg/m³"),
)


def _material_setup(mat: Material) -> dict:
    return dict(E=mat.E / 1e9, rho=mat.rho, nu=mat.nu)


def _critical(rng: np.random.Generator) -> Question:
    name, mat, text = MATERIALS[int(rng.integers(len(MATERIALS)))]
    h = pick(rng, (1, 2, 3, 4, 6, 8, 10))
    plate = Plate(0.6, 0.4, h / 1000, mat)
    fc = critical_frequency(plate)
    return Question(
        f"A {name} panel ({text}) {h:g} mm thick, in air (c = 343 m/s). At what frequency do its bending waves "
        "travel at the speed of sound?",
        fc,
        f"D = Eh³/12(1 − ν²) = {number(plate.rigidity)} N·m and ρh = {number(plate.surface_mass)} kg/m², so "
        f"f<sub>c</sub> = (c²/2π)√(ρh/D) = (343²/2π) × {number(1 / plate.scale)} = {number(fc)} Hz.",
        unit="Hz",
        setup=dict(thickness=float(h), **_material_setup(mat)),
    )


def _mass_law(rng: np.random.Generator) -> Question:
    h = pick(rng, (1, 2, 3, 5))
    f = pick(rng, (125, 250, 500, 1000))
    plate = Plate(0.6, 0.4, h / 1000, STEEL)
    tl = float(mass_law(plate, f))
    x = TWO_PI * f * plate.surface_mass / (2 * AIR.impedance)
    return Question(
        f"A steel sheet {h:g} mm thick (ρ = 7850 kg/m³) between two rooms of air (ρc = {number(AIR.impedance)} "
        f"rayl). By the mass law, what is its transmission loss at normal incidence at {f:g} Hz?",
        tl,
        f"m″ = ρh = {number(plate.surface_mass)} kg/m², ωm″/2ρc = {number(x)}, and TL = 10 log(1 + {number(x)}²) "
        f"= {number(tl)} dB.",
        unit="dB",
        tolerance=0.01,
        setup=dict(thickness=float(h), tab=TAB_TRANSMISSION),
    )


def _radiated(rng: np.random.Generator) -> Question:
    a, b = pick(rng, (0.5, 0.6, 1)), pick(rng, (0.4, 0.5))
    v = pick(rng, (0.5, 1, 2, 5))  # mm/s rms
    sigma = pick(rng, (0.01, 0.05, 0.2, 1))
    w = AIR.impedance * a * b * sigma * (v / 1000) ** 2
    return Question(
        f"A panel {a:g} × {b:g} m vibrates at {v:g} mm/s rms, averaged over it, with a radiation efficiency of "
        f"{sigma:g}. What sound power does it radiate into air (ρc = {number(AIR.impedance)} rayl)?",
        w * 1e6,
        f"W = ρc S σ ⟨v²⟩ = {number(AIR.impedance)} × {number(a * b)} × {sigma:g} × ({v:g} × 10⁻³)² = "
        f"{number(w)} W, {number(w * 1e6)} µW.",
        unit="µW",
    )


def _corner(rng: np.random.Generator) -> Question:
    a, b = pick(rng, (0.5, 0.6, 1)), pick(rng, (0.4, 0.5))
    m, n = ((1, 1), (1, 3), (3, 1))[int(rng.integers(3))]
    f = pick(rng, (20, 30, 40))
    plate = Plate(a, b, 0.010, STEEL)
    sigma = float(low_frequency_efficiency(plate, m, n, f))
    k = TWO_PI * f / AIR.c
    return Question(
        f"A baffled steel panel {a:g} × {b:g} m, 10 mm thick, vibrates in its mode ({m}, {n}) at {f:g} Hz, well "
        "below the mode's own resonance. From the low-frequency limit for odd m and n, what is its radiation "
        "efficiency, in thousandths (σ × 1000)?",
        1000 * sigma,
        f"k = ω/c = 2π × {f:g}/343 = {number(k)} rad/m, and σ<sub>mn</sub> ≈ 32k²ab/π⁵m²n² = 32 × {number(k * k)} × "
        f"{number(a * b)}/(π⁵ × {m * m * n * n}) = {number(sigma)}, {number(1000 * sigma)} thousandths: the panel "
        "pumps air like a small piston, and most of it sloshes round its edge.",
        unit="× 10⁻³",
        setup=dict(a=a, b=b, thickness=10.0, m=m, n=n),
    )


def _thicker(rng: np.random.Generator) -> Question:
    return Question(
        "A panel is made twice as thick, the same material. What happens to its critical frequency?",
        1,
        "It halves: f<sub>c</sub> = (c²/2π)√(ρh/D), and √(ρh/D) ∝ √(h/h³) = 1/h. A thicker panel's bending waves "
        "travel faster, so they catch up with sound at a lower frequency.",
        choices=("It doubles", "It halves", "It stays the same", "It falls by √8"),
        setup=dict(thickness=6.0),
        predict=True,
    )


def _double_mass(rng: np.random.Generator) -> Question:
    return Question(
        "Below coincidence, how much does doubling a panel's mass per area add to its transmission loss?",
        0,
        "6 dB: the mass law's TL ≈ 20 log(ωm″/2ρc), and 20 log 2 = 6.0 dB. (Doubling the thickness doubles the "
        "mass but also halves f<sub>c</sub>, bringing the dip down.)",
        choices=("6 dB", "3 dB", "12 dB", "Nothing: it's set by stiffness"),
        setup=dict(thickness=6.0, tab=TAB_TRANSMISSION),
        predict=True,
    )


def _even_mode(rng: np.random.Generator) -> Question:
    return Question(
        "At a low frequency, which of a baffled panel's modes radiates best: (1, 1), (2, 1) or (2, 2)?",
        0,
        "(1, 1): the whole panel moves in phase, a piston. In (2, 1) the halves push and pull, and their sound "
        "cancels to a dipole's; in (2, 2), to a quadrupole's. Each even index costs two more powers of k.",
        choices=("(1, 1)", "(2, 1)", "(2, 2)", "All the same"),
        setup=dict(m=2, n=2),
        predict=True,
    )


def _window_spec(rng: np.random.Generator) -> Question:
    f = pick(rng, (250, 500))
    h = pick(rng, (5, 6, 8))  # mm: the design
    plate = Plate(1.2, 1.0, h / 1000, GLASS_MATERIAL)
    tl = float(transmission_loss(plate, 0.01, f)[0])
    fc = critical_frequency(plate)
    need_tl = rounded(tl - 1.5, 2)
    need_fc = rounded(fc / 1.25, 2)

    def goal(t: AcousticsTrial):
        p = Plate(t.a, t.b, t.thickness / 1000, Material(t.E * 1e9, t.nu, t.rho))
        glass = abs(t.E - 70) <= 1 and abs(t.rho - 2500) <= 10 and abs(t.nu - 0.22) <= 0.005
        return (
            Requirement("Glass", f"{number(t.E)} GPa, {number(t.rho, 4)} kg/m³", glass),
            kept("Width a", t.a, 1.2, "m"),
            kept("Height b", t.b, 1.0, "m"),
            kept("Loss factor η", t.loss, 0.01),
            at_least(f"TL at {f:g} Hz", float(transmission_loss(p, t.loss, f)[0]), need_tl, "dB"),
            at_least("Critical frequency", critical_frequency(p), need_fc, "Hz"),
        )

    return Question(
        f"A window 1.2 × 1 m of single glass (η = 0.01) must give at least {number(need_tl, 2)} dB at {f:g} Hz "
        f"(field incidence), and keep its coincidence dip above {number(need_fc, 2)} Hz. Choose the thickness.",
        0,
        f"The TL below coincidence grows 6 dB per doubling of the thickness (the mass law), while f<sub>c</sub> "
        f"falls in proportion to 1/h: a window between the two. {h:g} mm gives {number(tl)} dB and "
        f"f<sub>c</sub> = {number(fc)} Hz.",
        setup=dict(a=1.2, b=1.0, thickness=3.0, tab=TAB_TRANSMISSION, **GLASS),
        goal=goal,
        design=dict(a=1.2, b=1.0, thickness=float(h), tab=TAB_TRANSMISSION, **GLASS),
    )


PROBLEMS = tuple(
    Problem(key, title, "acoustics_page", f"See {page_link('acoustics_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("critical", "The critical frequency", "coincidence", "Coincidence", _critical),
        ("mass-law", "The mass law", "transmission", "Between two rooms", _mass_law),
        ("radiated", "Radiated power", "efficiency", "Radiation efficiency", _radiated),
        ("corner", "A mode far below resonance", "modes", "A mode's radiation", _corner),
        ("thicker-fc", "Predict: a thicker panel", "coincidence", "Coincidence", _thicker),
        ("double-mass", "Predict: twice the mass", "transmission", "Between two rooms", _double_mass),
        ("even-mode", "Predict: which mode radiates", "modes", "A mode's radiation", _even_mode),
        ("window-spec", "Spec: a window's glass", "transmission", "Between two rooms", _window_spec),
    )
)

TIPS = {
    "a": "<p>The panel's side a (along x).</p>",
    "b": "<p>The panel's side b (along y).</p>",
    "thickness": "<p>The panel's thickness h.</p>",
    "E": "<p>Young's modulus.</p>",
    "rho": "<p>Density.</p>",
    "nu": "<p>Poisson's ratio.</p>",
    "loss": "<p>The panel's loss factor η (its stiffness is D(1 + iη)): 0.001 to 0.01 for bare metal or glass, "
    "0.1 or more with a damping layer.</p>",
    "m": "<p>The mode's half waves along a.</p>",
    "n": "<p>The mode's half waves along b.</p>",
    "drive": "<p>Where the point force (1 N) drives the panel, as a fraction of each side.</p>",
}
WAVE_TIP = (
    "<p>Wavenumbers against frequency: the panel's free bending wave k<sub>b</sub> ∝ √f, sound's k = ω/c ∝ f, "
    "crossing at the critical frequency. Dotted: the mode's k<sub>x</sub> = mπ/a and k<sub>y</sub> = nπ/b. "
    "Where k is below both the mode is a corner mode, between them an edge mode.</p>"
)
SHAPE_TIP = "<p>The mode's shape (colour: up and down) and the drive point (Driven panel tab).</p>"
ONE_TIP = (
    "<p>The mode's radiation efficiency σ<sub>mn</sub> = W/ρcS⟨v²⟩ against frequency, by Wallace's integral, "
    "with its low-frequency limit (odd m and n). The dot is at the mode's own resonance.</p>"
)
ALL_TIP = (
    "<p>Every mode's σ at its own resonance (dots, by the parity of m and n), their average in each third "
    "octave, Maidanik's formula for that average, and the infinite plate's 1/√(1 − f<sub>c</sub>/f).</p>"
)
POWER_TIP = (
    "<p>The power a 1 N point force puts in, and the sound power the panel radiates, against the infinite "
    "plate's input power F²/16√(Dρh) (dashed). Steps: third-octave averages.</p>"
)
VELOCITY_TIP = (
    "<p>The panel's mean-square velocity, space and time averaged, against the energy the infinite plate's "
    "input power sustains: P/ωηM (dashed).</p>"
)
TL_TIP = (
    "<p>The transmission loss of the panel (as an infinite plate) between two diffuse fields, sound arriving "
    "from every angle up to 78°, against the mass law at normal incidence.</p>"
)

NOTES_HTML = load_notes("vibroacoustics", trials=trials_html(TRIALS))


class VibroAcousticsPage(QtWidgets.QWidget):
    """Controls on the left, the mode and the wavenumbers in the centre, the radiation, the driven panel and the
    transmission loss on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.trials = TrialBox(
            [t.name for t in TRIALS], "<p>Each sets up the panel to show one idea from the Theory tab.</p>"
        )
        self.trials.picked.connect(self.load_trial)

        panel = Form("Panel (simply supported, baffled)")
        self.side_a = panel.number("Side a:", 0.05, 10.0, 0.6, 3, " m", TIPS["a"])
        self.side_b = panel.number("Side b:", 0.05, 10.0, 0.4, 3, " m", TIPS["b"])
        self.thickness = panel.number("Thickness h:", 0.1, 100.0, 3.0, 2, " mm", TIPS["thickness"])
        self.loss = panel.number("Loss factor η:", 0.0001, 0.5, 0.01, 4, "", TIPS["loss"])
        self.info = note_label()
        panel.row(self.info)

        material = Form("Material")
        self.E = material.number("Young's modulus E:", 0.01, 2000.0, 210.0, 2, " GPa", TIPS["E"])
        self.rho = material.number("Density ρ:", 1.0, 30000.0, 7850.0, 0, " kg/m³", TIPS["rho"])
        self.nu = material.number("Poisson's ratio ν:", 0.0, 0.49, 0.3, 2, "", TIPS["nu"])

        mode = Form("Mode and drive")
        self.mode_m = mode.number("Half waves m:", 1, 40, 1, 0, "", TIPS["m"])
        self.mode_n = mode.number("Half waves n:", 1, 40, 1, 0, "", TIPS["n"])
        self.drive_x = mode.number("Drive at x/a:", 0.01, 0.99, 0.35, 2, "", TIPS["drive"])
        self.drive_y = mode.number("Drive at y/b:", 0.01, 0.99, 0.33, 2, "", TIPS["drive"])

        connect_all(
            (self.side_a, self.side_b, self.thickness, self.loss, self.E, self.rho, self.nu),
            self._on_panel,
        )
        connect_all((self.mode_m, self.mode_n), self._on_mode)
        connect_all((self.drive_x, self.drive_y), self._on_drive)

        # --- centre
        self.shape_plot = plot("y", "m", "x", "m", height=180)
        self.shape_plot.setToolTip(SHAPE_TIP)
        self.shape_plot.setAspectLocked(True)
        self.shape_plot.setMouseEnabled(x=False, y=False)
        self.shape_image = pg.ImageItem(axisOrder="row-major")
        self.nodal = pg.IsocurveItem(level=0.0, axisOrder="row-major")  # as the image
        self.nodal.setParentItem(self.shape_image)
        self.outline = pg.PlotDataItem()
        self.drive_dot = marker(12, "x")
        for item in (self.shape_image, self.outline, self.drive_dot):
            self.shape_plot.addItem(item)

        self.wave_plot = plot("Wavenumber", "rad/m", "Frequency", "Hz", log_x=True, log_y=True, height=180)
        self.wave_plot.setToolTip(WAVE_TIP)
        add_legend(self.wave_plot.getPlotItem(), offset=(10, 10))
        self.kb_item = pg.PlotDataItem(name="bending wave k<sub>b</sub>")
        self.k_item = pg.PlotDataItem(name="sound k = ω/c")
        self.kx_line = pg.InfiniteLine(angle=0, label="mπ/a", labelOpts={"position": 0.97})
        self.ky_line = pg.InfiniteLine(angle=0, label="nπ/b", labelOpts={"position": 0.88})
        self.fc_line = pg.InfiniteLine(angle=90, label="coincidence", labelOpts={"position": 0.08})
        self.mode_dot = marker(10)
        for item in (self.kx_line, self.ky_line, self.fc_line, self.kb_item, self.k_item, self.mode_dot):
            self.wave_plot.addItem(item)
        self.readout = readout_label()
        centre = stack((self.shape_plot, 1), (self.wave_plot, 1), (self.readout, 0))

        # --- right: radiation efficiency
        self.one_plot = plot("σ<sub>mn</sub>", None, "Frequency", "Hz", log_x=True, log_y=True, height=150)
        self.one_plot.setToolTip(ONE_TIP)
        add_legend(self.one_plot.getPlotItem(), offset=(-10, -10))
        self.one_item = pg.PlotDataItem(name="Wallace's integral")
        self.one_low = pg.PlotDataItem(name="32k²ab/π⁵m²n²")
        self.one_dot = marker(10)
        self.one_fc = pg.InfiniteLine(angle=90)
        for item in (self.one_fc, self.one_low, self.one_item, self.one_dot):
            self.one_plot.addItem(item)
        self.all_plot = plot("σ", None, "Frequency", "Hz", log_x=True, log_y=True, height=150)
        self.all_plot.setToolTip(ALL_TIP)
        add_legend(self.all_plot.getPlotItem(), offset=(10, 10))
        self.parity_items = [marker(5, name=label) for label in ("odd–odd", "odd–even", "even–even")]
        self.band_item = pg.PlotDataItem(name="third-octave average", connect="pairs")
        self.maidanik_item = pg.PlotDataItem(name="Maidanik")
        self.wave_item = pg.PlotDataItem(name="infinite plate")
        self.all_fc = pg.InfiniteLine(angle=90)
        for item in (self.all_fc, *self.parity_items, self.wave_item, self.maidanik_item, self.band_item):
            self.all_plot.addItem(item)
        self.radiation_note = note_label()
        self.radiation_tab = column(self.one_plot, self.all_plot, self.radiation_note, stretch=False)

        # --- right: the driven panel
        self.power_plot = plot("Power", "W", "Frequency", "Hz", log_x=True, log_y=True, height=150)
        self.power_plot.setToolTip(POWER_TIP)
        add_legend(self.power_plot.getPlotItem(), offset=(-10, -10))
        self.input_item = pg.PlotDataItem(name="input")
        self.radiated_item = pg.PlotDataItem(name="radiated")
        self.input_band = pg.PlotDataItem(connect="pairs")
        self.radiated_band = pg.PlotDataItem(connect="pairs")
        self.input_inf = pg.PlotDataItem(name="infinite plate's input")
        self.power_fc = pg.InfiniteLine(angle=90)
        for item in (
            self.power_fc,
            self.input_item,
            self.radiated_item,
            self.input_band,
            self.radiated_band,
            self.input_inf,
        ):
            self.power_plot.addItem(item)
        self.velocity_plot = plot("⟨v²⟩", "m²/s²", "Frequency", "Hz", log_x=True, log_y=True, height=150)
        self.velocity_plot.setToolTip(VELOCITY_TIP)
        add_legend(self.velocity_plot.getPlotItem(), offset=(-10, -10))
        self.v2_item = pg.PlotDataItem(name="panel")
        self.v2_band = pg.PlotDataItem(connect="pairs")
        self.v2_inf = pg.PlotDataItem(name="P∞/ωηM")
        for item in (self.v2_item, self.v2_band, self.v2_inf):
            self.velocity_plot.addItem(item)
        self.driven_note = note_label()
        self.driven_tab = column(self.power_plot, self.velocity_plot, self.driven_note, stretch=False)

        # --- right: transmission loss
        self.tl_plot = plot("Transmission loss", "dB", "Frequency", "Hz", log_x=True, height=220)
        self.tl_plot.setToolTip(TL_TIP)
        add_legend(self.tl_plot.getPlotItem(), offset=(10, 10))
        self.tl_item = pg.PlotDataItem(name="field incidence (to 78°)")
        self.mass_item = pg.PlotDataItem(name="mass law, normal incidence")
        self.tl_fc = pg.InfiniteLine(angle=90, label="coincidence", labelOpts={"position": 0.04})
        for item in (self.tl_fc, self.mass_item, self.tl_item):
            self.tl_plot.addItem(item)
        self.tl_note = note_label()
        self.transmission_tab = column(self.tl_plot, self.tl_note, stretch=False)

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.radiation_tab, "Radiation")
        self.tabs.addTab(self.driven_tab, "Driven panel")
        self.tabs.addTab(self.transmission_tab, "Transmission loss")
        self.tabs.addTab(self.notes, "Theory")
        self.lazy = LazyTabs(
            self.tabs,
            {
                self.radiation_tab: self._draw_radiation,
                self.driven_tab: self._draw_driven,
                self.transmission_tab: self._draw_transmission,
            },
        )

        three_panes(self, column(self.trials, panel, material, mode), centre, self.tabs)
        self.modes: ModeSet | None = None
        self.drive: PointDrive | None = None
        self._on_panel()
        self.apply_theme()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(AcousticsTrial("", "", **setup))

    def practice_state(self) -> AcousticsTrial:
        """The page's settings, as a trial: what a practice spec checks."""
        return AcousticsTrial(
            "",
            "",
            a=self.side_a.value(),
            b=self.side_b.value(),
            thickness=self.thickness.value(),
            E=self.E.value(),
            rho=self.rho.value(),
            nu=self.nu.value(),
            loss=self.loss.value(),
            m=int(self.mode_m.value()),
            n=int(self.mode_n.value()),
            drive_x=self.drive_x.value(),
            drive_y=self.drive_y.value(),
            tab=self.tabs.currentIndex(),
        )

    def apply_trial(self, t: AcousticsTrial) -> None:
        for box, value in (
            (self.side_a, t.a),
            (self.side_b, t.b),
            (self.thickness, t.thickness),
            (self.E, t.E),
            (self.rho, t.rho),
            (self.nu, t.nu),
            (self.loss, t.loss),
            (self.mode_m, t.m),
            (self.mode_n, t.n),
            (self.drive_x, t.drive_x),
            (self.drive_y, t.drive_y),
        ):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self._on_panel()

    @property
    def plate(self) -> Plate:
        return Plate(
            self.side_a.value(),
            self.side_b.value(),
            self.thickness.value() / 1000,
            Material(self.E.value() * 1e9, self.nu.value(), self.rho.value()),
        )

    def _on_panel(self) -> None:
        p = self.plate
        self.fc = critical_frequency(p)
        self.info.setText(
            f"D = {p.rigidity:.4g} N·m, m″ = ρh = {p.surface_mass:.4g} kg/m²; f<sub>c</sub> = {fmt(self.fc, 'Hz')}."
        )
        self.modes = None  # worked out when a tab wants them
        self.drive = None
        self.lazy.stale()
        self._draw_centre()

    def _on_mode(self) -> None:
        self._draw_centre()
        self.lazy.stale(self.radiation_tab)

    def _on_drive(self) -> None:
        self.drive = None
        self._draw_shape()
        self.lazy.stale(self.driven_tab)
        settle_plots(self)

    @property
    def f_top(self) -> float:
        """The plots' top frequency: 2.5 f_c, at most 20 kHz."""
        return min(F_TOP, max(2.5 * self.fc, 400.0))

    def _modes(self) -> ModeSet:
        """Every mode up to the top frequency, each with σ at its own resonance (worked out once per panel)."""
        if self.modes is None:
            self.modes = mode_set(self.plate, self.f_top, count=4 * MAX_MODES)
        return self.modes

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.shape_image.setLookupTable(
            pg.ColorMap(
                [0.0, 0.5, 1.0], [pg.mkColor(c.mass[0]), pg.mkColor(c.background), pg.mkColor(c.force)]
            ).getLookupTable(nPts=256)
        )
        self.nodal.setPen(pg.mkPen(c.strong, width=2))
        self.outline.setPen(pg.mkPen(c.structure, width=2))
        mark(self.drive_dot, c.strong, width=3)
        dotted = pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DotLine)
        dashed = QtCore.Qt.PenStyle.DashLine
        self.kb_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.k_item.setPen(pg.mkPen(c.force, width=2))
        for line in (self.kx_line, self.ky_line):
            line.setPen(dotted)
            line.label.setColor(c.muted)
        for line in (self.fc_line, self.one_fc, self.all_fc, self.power_fc, self.tl_fc):
            line.setPen(pg.mkPen(c.faint, width=1, style=dashed))
        self.fc_line.label.setColor(c.muted)
        self.tl_fc.label.setColor(c.muted)
        mark(self.mode_dot, c.strong, filled=True)
        self.one_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.one_low.setPen(pg.mkPen(c.grey, width=2, style=dashed))
        mark(self.one_dot, c.strong, filled=True)
        for item, color in zip(self.parity_items, (c.mass[0], c.mass[1], c.mass[3])):
            mark(item, color, filled=True)
        self.band_item.setPen(pg.mkPen(c.strong, width=2))
        self.maidanik_item.setPen(pg.mkPen(c.force, width=2, style=dashed))
        self.wave_item.setPen(pg.mkPen(c.grey, width=1, style=dashed))
        self.input_item.setPen(pg.mkPen(c.mass[0], width=1))
        self.radiated_item.setPen(pg.mkPen(c.mass[1], width=1))
        self.input_band.setPen(pg.mkPen(c.mass[0], width=3))
        self.radiated_band.setPen(pg.mkPen(c.mass[1], width=3))
        self.input_inf.setPen(pg.mkPen(c.strong, width=2, style=dashed))
        self.v2_item.setPen(pg.mkPen(c.mass[0], width=1))
        self.v2_band.setPen(pg.mkPen(c.mass[0], width=3))
        self.v2_inf.setPen(pg.mkPen(c.strong, width=2, style=dashed))
        self.tl_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.mass_item.setPen(pg.mkPen(c.grey, width=2, style=dashed))
        self._draw_centre()
        self.lazy.stale()

    def _draw_centre(self) -> None:
        self._draw_shape()
        self._draw_waves()
        settle_plots(self)

    def _draw_shape(self) -> None:
        p = self.plate
        m, n = int(self.mode_m.value()), int(self.mode_n.value())
        x = np.linspace(0, p.a, GRID)
        y = np.linspace(0, p.b, max(3, int(GRID * p.b / p.a)))
        w = rect_shape(PlateMode(m, n, 0.0), p, x, y)
        self.shape_image.setImage(w, levels=(-1, 1), autoLevels=False)
        rect = QtCore.QRectF(0, 0, p.a, p.b)
        self.shape_image.setRect(rect)
        self.nodal.setData(w)
        self.outline.setData([0, p.a, p.a, 0, 0], [0, 0, p.b, p.b, 0])
        self.drive_dot.setData([self.drive_x.value() * p.a], [self.drive_y.value() * p.b])
        self.shape_plot.setXRange(0, p.a, padding=0.04)
        self.shape_plot.setYRange(0, p.b, padding=0.04)

    def _mode_freq(self, m: int, n: int) -> float:
        p = self.plate
        return math.pi**2 * ((m / p.a) ** 2 + (n / p.b) ** 2) * p.scale / TWO_PI

    def _draw_waves(self) -> None:
        p = self.plate
        m, n = int(self.mode_m.value()), int(self.mode_n.value())
        f = np.geomspace(F_LOW, self.f_top, 200)
        self.kb_item.setData(f, bending_wavenumber(p, f))
        self.k_item.setData(f, TWO_PI * f / AIR.c)
        kx, ky = m * math.pi / p.a, n * math.pi / p.b
        self.kx_line.setPos(math.log10(kx))
        self.ky_line.setPos(math.log10(ky))
        self.fc_line.setPos(math.log10(self.fc))
        fmn = self._mode_freq(m, n)
        self.mode_dot.setData([fmn], [math.hypot(kx, ky)])
        self.wave_plot.setXRange(math.log10(F_LOW), math.log10(self.f_top), padding=0)
        k_lo = min(TWO_PI * F_LOW / AIR.c, float(bending_wavenumber(p, F_LOW)))
        k_hi = max(TWO_PI * self.f_top / AIR.c, float(bending_wavenumber(p, self.f_top)), kx, ky)
        self.wave_plot.setYRange(math.log10(k_lo), math.log10(k_hi), padding=0.05)
        sigma = float(modal_efficiency(p, np.array([m]), np.array([n]), fmn)[0, 0])
        self.readout.setText(
            f"Mode ({m}, {n}) at {fmt(fmn, 'Hz')}: k<sub>x</sub> = {kx:.3g}, k<sub>y</sub> = {ky:.3g} rad/m against "
            f"sound's k = {TWO_PI * fmn / AIR.c:.3g}; {classify(p, m, n, fmn)}, σ = {sigma:.3g}. "
            f"Critical frequency f<sub>c</sub> = {fmt(self.fc, 'Hz')}."
        )

    def _draw_radiation(self) -> None:
        p = self.plate
        m, n = int(self.mode_m.value()), int(self.mode_n.value())
        top = self.f_top
        f = np.geomspace(F_LOW, top, CURVE_POINTS)
        one = np.array([modal_efficiency(p, np.array([m]), np.array([n]), fi)[0, 0] for fi in f])
        self.one_item.setData(f, one)
        if m % 2 and n % 2:
            low = low_frequency_efficiency(p, m, n, f)
            keep = low < 3.0
            self.one_low.setData(f[keep], low[keep])
        else:
            self.one_low.setData([], [])
        fmn = self._mode_freq(m, n)
        if F_LOW <= fmn <= top:
            self.one_dot.setData([fmn], [float(modal_efficiency(p, np.array([m]), np.array([n]), fmn)[0, 0])])
        else:
            self.one_dot.setData([], [])
        for line in (self.one_fc, self.all_fc):
            line.setPos(math.log10(self.fc))
        lo = max(float(one.min()), 1e-9)
        self.one_plot.setXRange(math.log10(F_LOW), math.log10(top), padding=0)
        self.one_plot.setYRange(math.log10(lo), math.log10(max(3.0, float(one.max()))), padding=0.05)

        modes = self._modes()
        parity = (modes.m % 2) + (modes.n % 2)  # 2 odd–odd, 1 mixed, 0 even–even
        for item, which in zip(self.parity_items, (2, 1, 0)):
            sel = parity == which
            item.setData(modes.freq[sel], modes.sigma[sel])
        centres = third_octaves(max(F_LOW, 25.0), top)
        avg = resonant_average(modes, centres)
        lo_e, hi_e = band_edges(centres)
        xs, ys = _steps(lo_e, hi_e, avg)
        self.band_item.setData(xs, ys)
        fm = np.geomspace(F_LOW, top, 300)
        self.maidanik_item.setData(fm, maidanik(p, fm))
        above = fm[fm > self.fc * 1.02]
        self.wave_item.setData(above, wave_efficiency(above, self.fc))
        self.all_plot.setXRange(math.log10(F_LOW), math.log10(top), padding=0)
        floor = max(float(np.min(modes.sigma)) if modes.sigma.size else 1e-6, 1e-7)
        self.all_plot.setYRange(math.log10(floor), math.log10(5.0), padding=0.05)
        mid = (modes.freq > 0.25 * self.fc) & (modes.freq < 0.5 * self.fc)
        between = (
            f"From f<sub>c</sub>/4 to f<sub>c</sub>/2 the {int(mid.sum())} modes' mean σ is "
            f"{float(modes.sigma[mid].mean()):.3g}, Maidanik's {float(maidanik(p, 0.5**1.5 * self.fc)):.3g} at "
            "the middle. "
            if mid.any()
            else ""
        )
        self.radiation_note.setText(
            f"{modes.freq.size} modes below {fmt(top, 'Hz')}, {int((modes.freq < self.fc).sum())} of them below "
            f"f<sub>c</sub>. {between}Above f<sub>c</sub> every mode radiates fully: σ → 1."
        )
        settle_plots(self)

    def _point_drive(self) -> PointDrive:
        if self.drive is None:
            p = self.plate
            modes = self._modes()
            # keep to MAX_MODES: the top frequency comes down if there are more modes than that
            top = self.f_top
            if modes.freq.size > MAX_MODES:
                top = min(top, float(modes.freq[MAX_MODES - 1]) / 1.5)
            sub = ModeSet(*(arr[modes.freq <= 1.5 * top] for arr in (modes.m, modes.n, modes.freq, modes.sigma)))
            f = np.geomspace(F_LOW, top, DRIVE_POINTS)
            self.drive_top = top
            self.drive = point_drive(
                p, self.loss.value(), (self.drive_x.value() * p.a, self.drive_y.value() * p.b), f, modes=sub
            )
        return self.drive

    def _draw_driven(self) -> None:
        d = self._point_drive()
        f = d.freq
        self.input_item.setData(f, np.maximum(d.input_power, 1e-30))
        self.radiated_item.setData(f, np.maximum(d.radiated_power, 1e-30))
        self.input_inf.setData([f[0], f[-1]], [d.infinite_input, d.infinite_input])
        self.v2_item.setData(f, d.mean_square_velocity)
        self.v2_inf.setData(f, d.infinite_velocity)
        centres = third_octaves(max(F_LOW * 1.12, 25.0), f[-1] / 1.12)
        lo, hi = band_edges(centres)
        means = [_band_means(f, values, lo, hi) for values in (d.input_power, d.radiated_power, d.mean_square_velocity)]
        for item, values in zip((self.input_band, self.radiated_band, self.v2_band), means):
            xs, ys = _steps(lo, hi, values)
            item.setData(xs, ys)
        self.power_fc.setPos(math.log10(self.fc))
        top = math.log10(f[-1])
        self.power_plot.setXRange(math.log10(F_LOW), top, padding=0)
        self.velocity_plot.setXRange(math.log10(F_LOW), top, padding=0)
        hi_p = float(np.max(d.input_power))
        self.power_plot.setYRange(math.log10(hi_p) - 8, math.log10(hi_p) + 0.3, padding=0)
        v2 = d.mean_square_velocity
        self.velocity_plot.setYRange(math.log10(float(v2.min())), math.log10(float(v2.max())), padding=0.05)
        p, eta = self.plate, self.loss.value()
        parts = []
        for at in (0.25 * self.fc, 1.5 * self.fc):
            if at <= f[-1]:
                rad = radiation_loss_factor(p, at)
                parts.append(
                    f"{rad:.2g} at {fmt(at, 'Hz')}, the sound {10 * math.log10(1 + eta / rad):.3g} dB below the input"
                )
        cut = f" (to {fmt(f[-1], 'Hz')}: {MAX_MODES} modes)" if self.drive_top < self.f_top else ""
        self.driven_note.setText(
            f"The infinite plate takes F²/16√(Dρh) = {fmt(d.infinite_input, 'W')} from 1 N. The radiation's own loss "
            f"factor ρcσ/ωm″ (Maidanik's σ) against η = {eta:.3g}: "
            + "; ".join(parts)
            + f". The modes' mutual radiation is left out{cut}."
        )
        settle_plots(self)

    def _draw_transmission(self) -> None:
        p, eta = self.plate, self.loss.value()
        f_dip = coincidence_dip(p, eta)[0]
        f = np.sort(np.append(np.geomspace(F_LOW * 2.5, F_TOP, 260), f_dip))  # the dip's bottom, too
        tl = transmission_loss(p, eta, f)
        self.tl_item.setData(f, tl)
        self.mass_item.setData(f, mass_law(p, f))
        self.tl_fc.setPos(math.log10(self.fc))
        self.tl_plot.setXRange(math.log10(f[0]), math.log10(f[-1]), padding=0)
        self.tl_plot.setYRange(max(0.0, float(tl.min()) - 5), float(mass_law(p, f[-1])) + 5, padding=0)
        f_dip, tl_dip = coincidence_dip(p, eta)
        at500 = float(transmission_loss(p, eta, 500.0)[0])
        self.tl_note.setText(
            f"m″ = {p.surface_mass:.3g} kg/m². At 500 Hz: {at500:.3g} dB from every angle, "
            f"{float(mass_law(p, 500.0)):.3g} dB at normal incidence. f<sub>c</sub> = {fmt(self.fc, 'Hz')}, and the "
            f"coincidence dip's bottom {tl_dip:.3g} dB at {fmt(f_dip, 'Hz')}; above it the TL rises about 9 dB an "
            "octave, more with more damping."
        )
        settle_plots(self)


def _steps(lo: np.ndarray, hi: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A level line across each band (draw with connect="pairs"); bands without a value are left out."""
    ok = np.isfinite(values)
    xs = np.column_stack((lo[ok], hi[ok])).ravel()
    return xs, np.repeat(values[ok], 2)


def _band_means(f: np.ndarray, values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Each band's mean over the frequencies in it (NaN for a band with none)."""
    out = np.full(lo.size, np.nan)
    for i, (a, b) in enumerate(zip(lo, hi)):
        sel = (f >= a) & (f < b)
        if sel.any():
            out[i] = float(values[sel].mean())
    return out
