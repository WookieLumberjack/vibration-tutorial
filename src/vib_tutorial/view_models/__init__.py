"""What a page shows, worked out without Qt: its curves, markers, axis ranges and readout texts.

A page's widget does three jobs: it reads its controls, works out what to show, and draws it. A view
model is the middle job on its own, as plain functions from the page's settings (its trial dataclass, the
same one `practice_state()` returns and `load_setup` takes) to frozen dataclasses of NumPy arrays and
strings. The widget then only binds: controls → trial → view → `setData`/`setText`, with the colours
picked from the theme by what each curve is.

So a page's readouts can be tested as the reader sees them, without a QApplication, and its plots drawn
somewhere else (a practice problem's "read the plot", a handout).

Working on it:

- No Qt here, nor pyqtgraph, nor anything from `gui/` (`tests/test_view_models.py` imports every module
  in a fresh interpreter and checks none of them is loaded).
- One function per panel or tab, so a page still works out only what shows (the start-up's run, say).
- A view holds what's drawn, not how: no pens or colours (those are the theme's, at draw time).
- A page that runs keeps what the run measured in a plain object here too (`duffing.Record`), which the simulator's
  stretches go into; the frame loop, its budget and the Start/Pause stay the page's. A function the frame calls
  (`duffing.live`) takes views of the record and allocates only what it draws.
- A page gets one when it's next worked on, not all at once: Isolation (`isolation.py`) and Duffing & Mathieu
  (`duffing.py`) so far, and the parts two pages share: the Components tab's (`components.py`, the Gyroscope and
  the 3D rigid body) and Vibro-acoustics' slow tabs' work (`vibroacoustics.py`). #478's live pages share
  `live_mass.py` (one mass stepped and changed as it runs, its energy balance), `grips.py` (a drag on the body to
  m, k, c), `free_body.py` (the forces on the body as arrows) and `recorder.py` (a frequency response built from
  the live run); their colours are theme roles' names, `style.role`'s. `record.py` holds what a running
  page records (`SampleBuffer`, the chain's `History`); `plot.Curve` is a line's or a marker's points; `text`
  formats numbers as the pages do (`number`, `fmt`, `entry`: never e-notation).
"""
