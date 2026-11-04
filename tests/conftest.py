"""Test-wide setup, run before any test module imports NumPy."""

import os

from vib_tutorial import ONE_THREAD

# BLAS on one thread, as the app runs it (vib_tutorial.main): the matrices are small, and
# OpenBLAS's pool makes their solves several times slower (worst on Windows CI).
for name in ONE_THREAD:
    os.environ.setdefault(name, "1")

# Two malloc arenas, not glibc's eight per core: the xdist workers inherit it, and their arenas' reserved address
# space (not memory used) put the worker running the GUI smoke tests over an 8 GB `ulimit -v`.
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
