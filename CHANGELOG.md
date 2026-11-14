# Changelog

What changed in each release, in brief. The pull requests have the detail, and each
[release](https://github.com/WookieLumberjack/vibration-tutorial/releases) lists them.

## Unreleased

- **New pages:**
  - Gyroscope;
  - Rigid body on mounts;
  - Deriving the equations;
  - Frame builder;
  - Test planning;
  - Nonlinearity & forces;
  - Rotor builder;
  - Vibro-acoustics & SEA;
  - Fluid–structure.
- **Practice window:** problems with new numbers each time on most pages, predict-then-run, and
  tune-to-a-spec problems.
- **Notes:**
  - every page's notes brought to one quality bar: typeset equations, figures drawn in the
    theme, a symbols table, worked examples checked by the tests, and further reading with
    checked editions;
  - the Theory window, with search across the notes and a glossary of symbols;
  - one text size for every Theory view (Ctrl+= / Ctrl+-).
- **Guidance:**
  - a first-run *How a page works*;
  - a course map with progress and four shorter paths;
  - the app reopens on the page last shown;
  - one way to start a page's motion (Start / Pause, Space).
- **Presenter mode:** a *Size* box scales the whole app to 125–200 % for a projector.
- **Speed:** more Rust kernels, a shared frame loop on every page, and the rotor's analysis off
  the UI thread.
- **Code:** ruff and mypy over the whole package, with a pre-commit hook. The large pages are
  split into smaller modules. The GUI tests free their windows. The README is split into
  `docs/` pages, with screenshots that a test keeps working.
- **Fixed:** errors the course audit found in the notes; a deadlock in the rotor's analysis
  worker that hung the macOS builds.

## 0.1.4 (2026-10-28)

- The app opens on the course's first page, and the Theory notes can be popped out into a
  window of their own.

## 0.1.3 (2026-10-27)

- **New pages:**
  - Operational modal analysis;
  - Continuous systems;
  - Random vibration;
  - Complex modes;
  - Model updating;
  - Shock spectrum;
  - Duffing & Mathieu;
  - Torsional vibration;
  - the One mass group and Signals & spectra;
  - twelve more, from the first course audit.
- **The pages grouped into a course,** with links between their notes and one-click
  *Things to try*.
- **Rotor:** a rotor–stator rub and the Newkirk effect.
- **Speed:** the first Rust kernels (oil film, friction, rub), the frame-timing overlay
  (`VIB_FRAME_STATS`), and plots that paint once a frame.
- The window fits a 1280 px screen.

## 0.1.2 (2026-10-12)

- **The Jeffcott rotor page:**
  - Bode, polar and waterfall plots;
  - stability;
  - bow;
  - balancing in one and two planes;
  - an overhung disc;
  - fluid-film and magnetic bearings;
  - an asymmetric and a cracked shaft;
  - its notes audited and extended.
- **Coulomb friction on the masses:** stick-slip, and a describing-function FRF.
- Acceleration measured on the virtual modal test.
- Pages built when first opened, and the strip chart decimated incrementally.

## 0.1.1 (2026-10-01)

- Fixed an intermittent macOS crash in the GUI tests; macOS builds again.

## 0.1.0 (2026-09-27)

The first release: the chain simulator with classical and state-space modal analysis, the
FRF matrix, substructuring (Craig–Bampton and free-interface), modal coupling, the virtual
modal test with modal parameter extraction, the presets, colour themes, and standalone builds
for Linux, Windows and macOS.
