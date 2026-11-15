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
- A page gets one when it's next worked on, not all at once. Isolation is the first (`isolation.py`).
"""
