//! vib_tutorial's numerical kernels in plain Rust: the hot loops the Python app's `vib_tutorial._native`
//! binds (`native/src`), with no Python in them, so another front end (a WebAssembly build, an egui app) can
//! call them as they are.
//!
//! Each kernel mirrors Python code that stays the reference implementation (`vib_tutorial.core.native`
//! picks between them), and the Python tests compare the two. Inputs are `linalg::Mat`s and slices;
//! outputs are written into the caller's rows (`linalg::RowsMut`); a shape that doesn't fit is an `Err`
//! with what didn't, which the binding raises as Python's ValueError.
//!
//! Working on it: keep it free of PyO3 and numpy (the binding converts), and keep each kernel the same
//! arithmetic as its Python reference, step for step (`tests/test_native.py`).

pub mod film;
pub mod friction;
pub mod linalg;
pub mod linear;
pub mod rub;

/// A kernel's result: a shape mismatch is an error naming it.
pub type Result<T> = std::result::Result<T, String>;
