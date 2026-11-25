[Contents](../README.md#contents)

# The window

| Area | What it holds |
|---|---|
| **Left** | System parameters (m, k, c for each element and friction F<sub>f</sub> for each mass, 1 to 8 masses), the excitation (a force on a mass, or motion of the ground), and simulation controls (run/pause, speed, plot window, auto-scale) |
| **Centre** | Animation of the chain (with a scale bar for the real displacement) and live energy bars, above time histories of the applied force and of the motion (in physical or modal coordinates), of the energy, or of the force in each spring and damper |
| **Right** | Tabs: *Modal analysis* (table, mode shapes, release), *Frequency response*, and *Theory* (notes written for students) |
| **Top** | The pages, in 11 groups, and the *Theory*, *Theory window*, *About*, *Theme* and *Size* controls |

The pages are grouped in the order they are meant to be read, each using only ideas from the pages
before it (Ctrl+1 to Ctrl+9 jump to the first nine groups, Ctrl+PgDown and Ctrl+PgUp to the next or previous page, and
every page's notes end with a link on to the next). Each page opens on its Theory tab the first time it's shown, where its goals
(*On this page you'll*) link to its Things to try and its problems; after that it opens on the tab you left. So
the first run opens on the first page, *Single oscillator*, at its Theory: *Start here*, the map of the course (which counts each page's practice problems solved), four shorter *paths*
through it (Essentials, Testing, Rotating machinery, Vibration environments: each with every page its pages
build on), then *How a page works* and the keys (the *About* box lists the keys too). Ctrl+= and Ctrl+− (or Ctrl and the mouse wheel) make the notes' text, equations and figures larger or smaller in every Theory tab and the Theory window, and Ctrl+0 puts them back; the size is remembered. The *Theory window* (Ctrl+F1) shows any page's Theory in a
window of its own, with the pages and their sections listed, a *Glossary of symbols* at the top, and a search
box (Ctrl+F, ⌘F on a Mac, from the window or from a page) that finds a word or a symbol in every page's notes: it keeps its place as you move between pages,
or follows them. Paused, a page that simulates as it runs (this one, Duffing & Mathieu, the torsional start-up and the Jeffcott rotor) shows a cursor on its time plots: drag it back to see the motion as it was then (the chain drawn, the state on the phase plane, the readings, each station's place on its orbit); Start carries on from where it was paused.

The groups (each one's page here describes its pages): the first six are the course's spine, read in order;
the next five are branches into application areas; the Appendix holds pages that are to be folded into others.

1. **[One oscillator](one-oscillator.md).** Single oscillator, Pendulums, Vibration isolation, Transient response and Periodic forcing.
2. **[Equations](equations.md).** Deriving the equations and 3D rigid body.
3. **[Modes](modes.md).** *Simulation & modal analysis* (the chain above), Energy methods, FRF matrix, Complex modes and Rigid body on mounts.
4. **[Building models](building-models.md).** Modal coupling, Substructuring (CMS) and Numerical integration.
5. **[Continuous systems](continuous-systems.md).** Continuous systems, Beams & plates and Frame builder.
6. **[Signals and random](signals-and-random.md).** Signals & spectra and Random vibration.
7. **[Rotating machines](rotating-machines.md).** Gyroscope, Torsional vibration, Jeffcott rotor, Rotor builder and Machinery diagnostics.
8. **[Testing](testing.md).** Virtual modal test, Model updating, Operational modal analysis, Test planning and Variability & localization.
9. **[Environments](environments.md).** Shock spectrum, Quarter-car, Fatigue and Vibro-acoustics & SEA.
10. **[Nonlinear and flow](nonlinear-and-flow.md).** Duffing & Mathieu, Fluid–structure and Self-excited vibration.
11. **[Control](control.md).** Active control.
12. **[Appendix](appendix.md).** Seismic response and Nonlinearity & forces.

Every page has a *Theory* tab (the *Theory* button or F1 opens it) that starts with what to read first,
links to the sections it builds on, and ends with *Things to try*, *What to take away*, a table of the
page's symbols and books to read further (the pages are being brought to this form one by one). Most pages'
*Things to try* are one click: pick one from the box at the top left and the page sets itself up. On the
One mass pages (so far) some ask you to predict first: the question and its choices show in the box, and the page
sets up only once you've picked one (or pressed *Just run it*), with the verdict above the note that
explains it. Their notes also open with *On this page you'll*, two or three goals, each with a link
that sets up its Thing to try and one that asks its practice problem. All of them are described below.

Vibration isolation, Quarter-car, Active control, the Frame builder and the Jeffcott rotor also have a
*Challenge* box under their Things to try: a spec to meet with the page's own controls (an isolation at a
speed, a damper for comfort and grip). Pick one and the page sets up where it starts; as you tune, each
requirement is ticked or crossed with what the page has now, and once all are met it counts as solved,
as in the Practice window. *Start over* sets it up again, *New numbers* asks it afresh.

The *Practice* window (Ctrl+F2) asks every page's problems with new numbers each time. The box above its list
filters it by page or problem, *Next unsolved* goes on to the next one you haven't solved in the course's
order, a wrong choice says why it's wrong (on the pages whose problems give their reasons so far), and some
problems show a figure of the situation above the question.

## A plot's data

Right-click a plot for its menu: **Copy data as CSV** puts its curves on the clipboard, to paste into a
spreadsheet; **Save data as CSV…** writes them to a file, an x and a y column for each curve, named after
the plot's axes and the curve's legend; **Save image as PNG…** saves the plot as it shows; **Show all**
fits the plot to its data again after a zoom. **Hold these curves** keeps a grey dashed copy of what the plot
shows behind it: change a control and compare the new curves with the held ones (hold again to add more).
**Clear held curves** takes them off, as *Defaults* does. **Hold curves** on the run bar (H) holds every
plot the page shows at once, and **Clear held** (Shift+H) takes them all off. (The frame and rotor builders' canvases keep their
own right-click, which edits them.)

A line with a ◆ grip, or a dot that grows under the mouse, can be dragged: the drive's frequency on the
Single oscillator's Bode plot, the running speed on Vibration isolation's transmissibility, the damper on
Quarter-car's trade-off and damper sweep, the speed on the Jeffcott rotor's Campbell diagram, the trial
frequency in Energy methods' Holzer tab, the oscillator picked on the Shock spectrum, and the instant on the
Gyroscope's and 3D rigid body's component plots. The page's own box follows the drag, and the drag the box.

The **run bar** along the foot of the window holds the motion's controls on every page: **Start** /
**Pause**, **Reset**, the speed and the time, and **Hold curves** / **Clear held**. It drives whatever the page and tab showing can run (a page with
nothing that plays in time says so). Space does the same.

The Single oscillator draws the mass itself above its time plots, moving with the free vibration or the
driven response: **Start** plays it (Space; at ¼ speed to begin with, as a 3 Hz swing is a blur), **Reset**
takes it back to t = 0, and the time cursor on the plot it follows can be dragged to scrub. Drag the mass
sideways and let go to pluck it: it vibrates freely from where you let go. Pendulums does the same with
the pendulum (a bob, a rod, a disc on its wire seen from above, an inverted arm), swinging beside a faint
ghost of the linearized swing: turn it about its pivot and let go to start it from that angle, either side.
Vibration isolation's *Start-up* tab draws
the machine on its mounts, running up with its unbalance turning: pull it and let go and, stopped, it rings
on its mounts.

## Themes

*Theme* (top right) switches the whole app between **Light**, **Dark**, **Solarized**,
**Nord** and **Projector**, live, without resetting the simulation. **Projector** is Light with darker
greys, every line twice as wide and every marker 40 % larger, which a projector's washed-out picture
needs; its text stays at its usual size, so the first time it's picked the app offers to show everything
at 125 % too (*Size*, below). **System** (the default) follows the
desktop's light or dark setting, and changes with it. Each theme has its own palettes for
the masses, modes and substructures, chosen so every curve, label and table entry stays
legible on its background. The choice is remembered.

![Dark theme: a harmonic force tuned to mode 2, plotted in modal coordinates, with non-proportional damping](images/dark_theme.png)

## Size, for presenting

*Size* (top right, beside *Theme*) draws the whole app larger, for a projector or a TV across the room:
**125 %**, **150 %** or **200 %**. Text, lines, markers, plots and figures all grow together, as on a
high-resolution screen, so every page looks as it does at 100 %, only larger. The app restarts to change
it, back on the page it showed; the choice is remembered. At 150 % the window needs a screen about
1,700 px wide (a 1080p projector or TV), and a size the screen can't hold is refused. If the app opens on
a smaller screen than the size was picked for, it goes back to 100 %. The notes' text has its own size as
well (Ctrl+= and Ctrl+-), for reading closely at any size.

**F5** (*Present*) folds every page's side panes away: once for the controls (the plots take their room), again
for the right-hand tabs too (the main plots the whole width), and once more for all of them back. The window
then needs less room: about 760 px wide without the controls and 740 without either side, against 1,184, so a
larger *Size* fits a smaller screen (its height, about 660 px, is still the plots'). The choice is remembered,
across the restart a new *Size* takes; while it's on, *Show panes* in the top-right corner brings every pane back.
With the controls folded away, **Alt+PgDown** and **Alt+PgUp** still step through the page's *Things to try*,
and each step's note shows large over the top of the page, so the room can read it; a step that asks a
prediction shows its question and choices there, to answer from the front. The × or **Esc** hides it.

**F6** turns on a pointer, a large dot in the force colour that follows the mouse, for pointing at a curve
across the room. **F7** is ink: drag to draw on the page (circle a peak, sketch the curve you expect before
showing it); F7 again gives the mouse back to the page with the ink left on it. **Esc** clears the ink and
turns both off, and changing page clears the ink.

## Defaults

*Defaults* (top right, or Ctrl+R) puts the page showing back as it first opened: every control as it was,
*Things to try* back at *Pick one…*, the motion at rest waiting for *Start* (or a mode shape swinging, on
a page that opens showing one). A page that shows the chain puts back the chain it opened with too, on the
Simulation page and every other page that shows it. The tab you're on stays.

---

[Contents](../README.md#contents)
