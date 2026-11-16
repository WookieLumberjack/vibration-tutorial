//! The Python binding of vib_tutorial's Rust kernels (`vib-kernels`, native/kernels), imported as
//! `vib_tutorial._native`.
//!
//! Each class and function here converts NumPy arrays to the kernels' matrices and slices, runs the kernel
//! without the GIL (so the caller's other threads run meanwhile) and raises a shape the kernel refuses as a
//! ValueError. The numerics are all in the kernels; `src/vib_tutorial/_native.pyi` types what's here.
//!
//! Each kernel mirrors Python code that stays the reference implementation (`vib_tutorial.core.native`
//! picks between them), and the tests compare the two.

use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vib_kernels::linalg::{Mat, RowsMut, expm, expm_foh};
use vib_kernels::{film, friction, linear, newton, rub};

fn mat(a: &PyReadonlyArray2<f64>) -> Mat {
    let a = a.as_array();
    let (rows, cols) = a.dim();
    Mat::from_rows(rows, cols, a.iter().copied().collect())
}

fn vec1(a: &PyReadonlyArray1<f64>) -> Vec<f64> {
    a.as_array().to_vec()
}

/// `f` given `out`'s rows to write: in place where it's C-contiguous (as `np.empty` makes), else through a copy.
fn with_rows<R>(out: &mut PyReadwriteArray2<f64>, f: impl FnOnce(RowsMut) -> R) -> R {
    let mut a = out.as_array_mut();
    let (rows, cols) = a.dim();
    if let Some(data) = a.as_slice_mut() {
        return f(RowsMut { data, rows, cols });
    }
    let mut copy = vec![0.0; rows * cols];
    let result = f(RowsMut {
        data: &mut copy,
        rows,
        cols,
    });
    for j in 0..rows {
        for i in 0..cols {
            a[[j, i]] = copy[j * cols + i];
        }
    }
    result
}

fn raise<T>(r: vib_kernels::Result<T>) -> PyResult<T> {
    r.map_err(PyValueError::new_err)
}

/// One step size's update for one input: z₁ = Φ z₀ + g₀ u₀ + g₁ u₁ (`linear::LinearSteps`).
#[pyclass(module = "vib_tutorial._native")]
struct LinearSteps(linear::LinearSteps);

#[pymethods]
impl LinearSteps {
    #[new]
    fn new(phi: PyReadonlyArray2<f64>, g0: PyReadonlyArray1<f64>, g1: PyReadonlyArray1<f64>) -> PyResult<Self> {
        raise(linear::LinearSteps::new(mat(&phi), vec1(&g0), vec1(&g1))).map(LinearSteps)
    }

    /// From the state z with the input u0, through the inputs `u` (one per step, at its end): each step's
    /// state into `out` (steps × len(z)).
    fn advance(
        &self,
        py: Python<'_>,
        z: PyReadonlyArray1<f64>,
        u0: f64,
        u: PyReadonlyArray1<f64>,
        mut out: PyReadwriteArray2<f64>,
    ) -> PyResult<()> {
        let (z, u) = (vec1(&z), vec1(&u));
        raise(with_rows(&mut out, |rows| {
            py.detach(|| self.0.advance(&z, u0, &u, rows))
        }))
    }
}

/// The chain with friction, for one input (`friction::FrictionSteps`).
#[pyclass(module = "vib_tutorial._native")]
struct FrictionSteps(friction::FrictionSteps);

#[pymethods]
impl FrictionSteps {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (aw, p, masses, stiffness, damping, friction, static_friction, base, k, events_per_mass))]
    fn new(
        aw: PyReadonlyArray2<f64>,
        p: PyReadonlyArray2<f64>,
        masses: PyReadonlyArray1<f64>,
        stiffness: PyReadonlyArray1<f64>,
        damping: PyReadonlyArray1<f64>,
        friction: PyReadonlyArray1<f64>,
        static_friction: PyReadonlyArray1<f64>,
        base: bool,
        k: usize,
        events_per_mass: usize,
    ) -> PyResult<Self> {
        raise(friction::FrictionSteps::new(
            mat(&aw),
            mat(&p),
            vec1(&masses),
            vec1(&stiffness),
            vec1(&damping),
            vec1(&friction),
            vec1(&static_friction),
            base,
            k,
            events_per_mass,
        ))
        .map(FrictionSteps)
    }

    /// `_advance_friction`'s steps: each step's state into `out` and its losses into `losses`.
    #[allow(clippy::too_many_arguments)]
    fn advance(
        &mut self,
        py: Python<'_>,
        z: PyReadonlyArray1<f64>,
        u0: f64,
        u: PyReadonlyArray1<f64>,
        h: f64,
        mut out: PyReadwriteArray2<f64>,
        mut losses: PyReadwriteArray2<f64>,
    ) -> PyResult<()> {
        let (z, u) = (vec1(&z), vec1(&u));
        let kernel = &mut self.0;
        raise(with_rows(&mut out, |out| {
            with_rows(&mut losses, |losses| {
                py.detach(|| kernel.advance(&z, u0, &u, h, out, losses))
            })
        }))
    }

    /// `drive`'s steps, without the ledger: each state into `out`.
    fn drive(
        &mut self,
        py: Python<'_>,
        z: PyReadonlyArray1<f64>,
        u0: f64,
        u: PyReadonlyArray1<f64>,
        h: f64,
        mut out: PyReadwriteArray2<f64>,
    ) -> PyResult<()> {
        let (z, u) = (vec1(&z), vec1(&u));
        let kernel = &mut self.0;
        raise(with_rows(&mut out, |out| {
            py.detach(|| kernel.drive(&z, u0, &u, h, out))
        }))
    }
}

/// The π-film's whole force on the journal, its derivatives and its dissipation (`film::pi_film`).
#[pyfunction]
fn pi_film(k: f64, c: f64, x: f64, y: f64, vx: f64, vy: f64, omega: f64) -> film::Film {
    film::pi_film(k, c, x, y, vx, vy, omega)
}

/// (c_xx, c_xy, c_yy): the whole uncavitated film's damping at (x, y) (`film::full_film_damping`).
#[pyfunction]
fn full_film_damping(k: f64, c: f64, x: f64, y: f64) -> (f64, f64, f64) {
    film::full_film_damping(k, c, x, y)
}

/// The rotor's nonlinear film steps at one held speed and step length (`film::FilmSteps`).
#[pyclass(module = "vib_tutorial._native")]
struct FilmSteps(film::FilmSteps);

#[pymethods]
impl FilmSteps {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (a, b, minv, cf, kf, p, h, idx, films, drift, tol, correction, iterations))]
    fn new(
        a: PyReadonlyArray2<f64>,
        b: PyReadonlyArray2<f64>,
        minv: PyReadonlyArray2<f64>,
        cf: PyReadonlyArray2<f64>,
        kf: PyReadonlyArray2<f64>,
        p: PyReadonlyArray1<f64>,
        h: f64,
        idx: Vec<usize>,
        films: Vec<(f64, f64, f64)>,
        drift: f64,
        tol: f64,
        correction: f64,
        iterations: usize,
    ) -> PyResult<Self> {
        raise(film::FilmSteps::new(
            mat(&a),
            mat(&b),
            mat(&minv),
            mat(&cf),
            mat(&kf),
            vec1(&p),
            h,
            idx,
            films,
            drift,
            tol,
            correction,
            iterations,
        ))
        .map(FilmSteps)
    }

    /// `_film_steps_reference`'s loop, q into `out`'s rows; returns the last state.
    #[allow(clippy::too_many_arguments)]
    fn steps(
        &mut self,
        py: Python<'_>,
        z: PyReadonlyArray1<f64>,
        u0: PyReadonlyArray1<f64>,
        u: PyReadonlyArray2<f64>,
        w1: PyReadonlyArray1<f64>,
        omega0: f64,
        mut out: PyReadwriteArray2<f64>,
    ) -> PyResult<Vec<f64>> {
        let (z, u0, u, w1) = (vec1(&z), vec1(&u0), mat(&u), vec1(&w1));
        let kernel = &mut self.0;
        raise(with_rows(&mut out, |out| {
            py.detach(|| kernel.steps(&z, &u0, &u, &w1, omega0, out))
        }))
    }

    /// How many times the damping bound has been taken.
    #[getter]
    fn takes(&self) -> usize {
        self.0.takes()
    }

    /// Each film's damping bound (c_xx, c_xy, c_yy) as last taken, or None before the first step.
    #[getter]
    fn taken(&self) -> Option<Vec<(f64, f64, f64)>> {
        self.0.taken()
    }
}

/// The disc rubbing on its stator ring (`rub::RubSteps`).
#[pyclass(module = "vib_tutorial._native")]
struct RubSteps(rub::RubSteps);

#[pymethods]
impl RubSteps {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (phi, g0, g1, n_dof, xd, yd, ring, slip, radius, tol, iterations))]
    fn new(
        phi: PyReadonlyArray2<f64>,
        g0: PyReadonlyArray2<f64>,
        g1: PyReadonlyArray2<f64>,
        n_dof: usize,
        xd: usize,
        yd: usize,
        ring: rub::Ring,
        slip: f64,
        radius: f64,
        tol: f64,
        iterations: usize,
    ) -> PyResult<Self> {
        raise(rub::RubSteps::new(
            mat(&phi),
            mat(&g0),
            mat(&g1),
            n_dof,
            xd,
            yd,
            ring,
            slip,
            radius,
            tol,
            iterations,
        ))
        .map(RubSteps)
    }

    /// `_rub_steps_reference`'s loop: q into `out`'s rows, the ring's force, push and heat into `rubs`'.
    #[allow(clippy::too_many_arguments)]
    fn steps(
        &self,
        py: Python<'_>,
        z: PyReadonlyArray1<f64>,
        drive: PyReadonlyArray2<f64>,
        w1: PyReadonlyArray1<f64>,
        omega0: f64,
        mut out: PyReadwriteArray2<f64>,
        mut rubs: PyReadwriteArray2<f64>,
    ) -> PyResult<Vec<f64>> {
        let (z, drive, w1) = (vec1(&z), mat(&drive), vec1(&w1));
        raise(with_rows(&mut out, |out| {
            with_rows(&mut rubs, |rubs| {
                py.detach(|| self.0.steps(&z, &drive, &w1, omega0, out, rubs))
            })
        }))
    }
}

/// A linear update under force laws, Newton at each step's end (`newton::NewtonSteps`). Each law kind is a
/// keyword: `cubic`, a (col, i, k3, k1) per spring.
#[pyclass(module = "vib_tutorial._native")]
struct NewtonSteps(newton::NewtonSteps);

#[pymethods]
impl NewtonSteps {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (phi, g0, g1, tol, iterations, watch=None, cubic=vec![]))]
    fn new(
        phi: PyReadonlyArray2<f64>,
        g0: PyReadonlyArray2<f64>,
        g1: PyReadonlyArray2<f64>,
        tol: f64,
        iterations: usize,
        watch: Option<(usize, f64)>,
        cubic: Vec<(usize, usize, f64, f64)>,
    ) -> PyResult<Self> {
        let laws = cubic
            .into_iter()
            .map(|(col, i, k3, k1)| newton::Law::Cubic { col, i, k3, k1 })
            .collect();
        raise(newton::NewtonSteps::new(
            mat(&phi),
            mat(&g0),
            mat(&g1),
            laws,
            tol,
            iterations,
            watch,
        ))
        .map(NewtonSteps)
    }

    /// From z with the inputs u0, a step per row of `applied`: states into `out`, inputs into `inputs`;
    /// returns how many steps it took.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        py: Python<'_>,
        z: PyReadonlyArray1<f64>,
        u0: PyReadonlyArray1<f64>,
        applied: PyReadonlyArray2<f64>,
        modulation: PyReadonlyArray1<f64>,
        mut out: PyReadwriteArray2<f64>,
        mut inputs: PyReadwriteArray2<f64>,
    ) -> PyResult<usize> {
        let (z, u0, applied, modulation) = (vec1(&z), vec1(&u0), mat(&applied), vec1(&modulation));
        raise(with_rows(&mut out, |out| {
            with_rows(&mut inputs, |inputs| {
                py.detach(|| self.0.run(&z, &u0, &applied, &modulation, out, inputs))
            })
        }))
    }
}

type Array<'py> = Bound<'py, numpy::PyArray2<f64>>;

fn to_numpy<'py>(py: Python<'py>, m: Mat) -> PyResult<Array<'py>> {
    numpy::PyArray1::from_vec(py, m.data).reshape([m.rows, m.cols])
}

/// e^A, for the tests to check against SciPy's.
#[pyfunction(name = "expm")]
fn expm_py<'py>(py: Python<'py>, a: PyReadonlyArray2<'py, f64>) -> PyResult<Array<'py>> {
    let m = mat(&a);
    if m.rows != m.cols {
        return Err(PyValueError::new_err("expm needs a square matrix"));
    }
    to_numpy(py, expm(&m))
}

/// (Φ, Γ0, Γ1) as `foh.foh_discretize(A, B, h)` returns them, for the tests to check.
#[pyfunction(name = "foh_discretize")]
fn foh_discretize_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
    h: f64,
) -> PyResult<(Array<'py>, Array<'py>, Array<'py>)> {
    let (a, b) = (mat(&a), mat(&b));
    if a.rows != a.cols || b.rows != a.rows {
        return Err(PyValueError::new_err("A must be square, and B have its rows"));
    }
    let (phi, g1, g2) = expm_foh(&a, &b, h);
    Ok((
        to_numpy(py, phi)?,
        to_numpy(py, g1.combine(1.0, &g2, -1.0))?,
        to_numpy(py, g2)?,
    ))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ECCENTRICITY_MAX", film::ECCENTRICITY_MAX)?;
    m.add_function(wrap_pyfunction!(pi_film, m)?)?;
    m.add_function(wrap_pyfunction!(full_film_damping, m)?)?;
    m.add_class::<FilmSteps>()?;
    m.add_class::<FrictionSteps>()?;
    m.add_class::<RubSteps>()?;
    m.add_class::<LinearSteps>()?;
    m.add_class::<NewtonSteps>()?;
    m.add_function(wrap_pyfunction!(expm_py, m)?)?;
    m.add_function(wrap_pyfunction!(foh_discretize_py, m)?)?;
    Ok(())
}
