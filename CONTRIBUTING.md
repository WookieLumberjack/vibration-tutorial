# Contributing

Thanks for helping. This is a teaching app, so a change is judged first on whether it's
correct and whether a reader learning alone can follow it.

[CLAUDE.md](CLAUDE.md) is the detailed guide to the code: how the pages, notes, tests and
builds fit together, and the rules that keep them working. Each module's docstring says how it
works and, under *Working on it*, which tests check it. Read those before changing a module.

## Setting up

You need [uv](https://docs.astral.sh/uv/) and a [Rust toolchain](https://rustup.rs/) (uv
builds the Rust kernels in `native/` on the first sync).

```bash
uv sync                 # .venv, the dependencies and the Rust kernels
uv run vib-tutorial     # the app
uvx pre-commit install  # ruff and mypy before each commit
```

## Before you open a pull request

- `uv run ruff format` and `uv run ruff check`: the CI's *Lint* workflow checks both.
- `uv run mypy`: the whole package is clean, and stays so.
- `uv run pytest`: the whole suite, about four minutes on four workers. The CI doesn't run
  it on pull requests, so run it yourself. `uv run pytest -m "not gui and not slow"` is the
  quick loop (~30 s) while you work.
- After a visible change, rerun the affected shots in `scripts/readme_screenshots.py` and
  check that the captions' numbers in `docs/` still hold.

One change per pull request, based on `main`. Say in the description what changed, why, and
how you checked it.

## Notes and numbers

A page's notes live in `src/vib_tutorial/theory/<module>.html`. A number the notes quote
is checked in `tests/test_tutorial_numbers.py`. A result named after someone gets a
citation from `gui/references.py`. A new symbol goes in `gui/notation.py`'s `SYMBOLS`.
CLAUDE.md lists the full quality bar a page's notes are held to.

## Adding a page

A new page touches several places. Each is checked by a test, but here is the list:

1. `src/vib_tutorial/core/<topic>.py`: the model, in NumPy/SciPy with no Qt, and
   `tests/test_<topic>.py`.
2. `src/vib_tutorial/gui/<page>.py`: the page, built from `gui/lesson.py` (`plot`, `Form`,
   `TrialCombo`, `RunButton`), with a `TRIALS` tuple. A page that animates holds a
   `gui/frame_loop.py` `FrameLoop`.
3. `src/vib_tutorial/theory/<page>.html`: its notes, with `<before>`, `<takeaways>`,
   `<notation>` and *Further reading*. Add a figure module, `gui/<page>_figures.py`, if the
   notes need figures.
4. `gui/links.py`'s `COURSE`: where it sits in the course. It may use only ideas from the
   pages before it.
5. The main window's protocols in `gui/main_window.py`, if the window calls something new
   on it.
6. `tests/test_gui_smoke.py`'s `test_lesson_pages_trials_and_tabs` list, and the notes'
   numbers in `tests/test_tutorial_numbers.py`.
7. Optional: `PROBLEMS` for the Practice window, with `load_setup`.
8. A `##` section in its group's `docs/<group>.md`, and a line in the README's Contents.
9. The page in CLAUDE.md's *Pages* list.

## Licence

By contributing you agree your work is released under the project's [MIT License](LICENSE).
