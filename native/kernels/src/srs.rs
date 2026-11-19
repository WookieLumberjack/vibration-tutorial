//! The shock response spectrum's oscillators (`shock.srs_peaks_reference`): each one a complex pole λ stepped
//! through the base's acceleration by the exact FOH update, η_{k+1} = μ η_k + c0 u_k + c1 u_{k+1}, its absolute
//! acceleration 2 Re(λ²η) and relative displacement 2 Re(η) at every sample.
//!
//! The Python keeps every oscillator's whole history, (f, samples) complex, and then finds the peaks; this
//! steps oscillators a few at a time and keeps only their largest values as they pass, each with its
//! neighbours, refined by the same parabola (`_refined`): the first sample where a value is largest wins, as
//! `argmax`. Then the residual, the free decay from the last state, 2 Re(λ^p η e^{λτ}) at `RESIDUAL_POINTS` a
//! damped period (`_free`), where its largest values lie.
//!
//! Working on it: keep the update and the refinement the reference's, step for step (`tests/test_native.py`).

use std::f64::consts::TAU;

/// Points a damped period where the residual's peak is looked for (`shock.RESIDUAL_POINTS`; a test checks
/// they agree).
pub const RESIDUAL_POINTS: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    const fn new(re: f64, im: f64) -> Self {
        C { re, im }
    }
    fn add(self, o: C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: C) -> C {
        C::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: C) -> C {
        C::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
    fn scale(self, s: f64) -> C {
        C::new(self.re * s, self.im * s)
    }
    fn div(self, o: C) -> C {
        let d = o.re * o.re + o.im * o.im;
        C::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
    fn exp(self) -> C {
        let m = self.re.exp();
        C::new(m * self.im.cos(), m * self.im.sin())
    }
}

/// A sequence's largest value and its two neighbours, as the samples pass.
struct Peak {
    best: f64,
    at: usize,
    before: f64,
    after: Option<f64>,
}

impl Peak {
    fn new() -> Self {
        Peak {
            best: f64::NEG_INFINITY,
            at: 0,
            before: f64::NAN,
            after: None,
        }
    }

    fn see(&mut self, k: usize, v: f64, prev: f64) {
        if v > self.best {
            (self.best, self.at, self.before, self.after) = (v, k, prev, None);
        } else if k == self.at + 1 {
            self.after = Some(v);
        }
    }

    /// The largest value, refined by a parabola through it and its neighbours where it's inside and a crest.
    fn refined(&self) -> f64 {
        let (a, peak) = (self.before, self.best);
        match self.after {
            Some(c) if self.at > 0 => {
                let curve = a - 2.0 * peak + c;
                if curve < 0.0 {
                    let p = 0.5 * (a - c) / curve;
                    peak - 0.25 * (a - c) * p
                } else {
                    peak
                }
            }
            _ => peak,
        }
    }
}

/// One oscillator's FOH update and its acceleration's factor: η ← μ η + c0 u_prev + c1 u, y = 2 Re(λ² η).
#[derive(Clone, Copy)]
struct Osc {
    mu: C,
    c0: C,
    c1: C,
    lam: C,
    lam2: C,
}

impl Osc {
    /// `shock._foh` for the pole of an oscillator at `fi` Hz, damping `zeta`, the input `h` s a step.
    fn new(fi: f64, zeta: f64, h: f64) -> Self {
        let w = TAU * fi;
        let lam = C::new(-zeta * w, w * (1.0 - zeta * zeta).sqrt());
        let beta = C::new(0.0, 1.0 / (2.0 * lam.im)); // −1/(2i Im λ)
        let mu = lam.scale(h).exp();
        let one = C::new(1.0, 0.0);
        let i0 = mu.sub(one).div(lam);
        let i1 = mu.sub(one).sub(lam.scale(h)).div(lam.mul(lam).scale(h));
        Osc {
            mu,
            c0: beta.mul(i0.sub(i1)),
            c1: beta.mul(i1),
            lam,
            lam2: lam.mul(lam),
        }
    }
}

/// Oscillators stepped side by side through each sample: each one's update waits on its last, so one alone
/// leaves the processor idle between steps (5 ns a step); several interleave. Each lane's arithmetic is the same.
const LANES: usize = 8;

/// The free decay from `eta` over one damped period: the largest absolute acceleration, the largest of its
/// negative, and the largest |relative displacement|.
fn residual(o: &Osc, eta: C) -> (f64, f64, f64) {
    let period = TAU / o.lam.im;
    let (ay, az) = (o.lam2.mul(eta), eta);
    let (mut up, mut down, mut z) = (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for j in 0..=RESIDUAL_POINTS {
        let e = o.lam.scale(j as f64 / RESIDUAL_POINTS as f64 * period).exp();
        let y = 2.0 * ay.mul(e).re;
        (up, down) = (up.max(y), down.max(-y));
        z = z.max((2.0 * az.mul(e).re).abs());
    }
    (up, down, z)
}

/// For each oscillator at `f` (Hz), damping `zeta`, under the base acceleration `a` sampled `h` apart, a row
/// of `f.len()` each: while the record lasts, the refined largest absolute acceleration and the largest of its
/// negative; after it, the same of the residual; and the largest |relative displacement| over both.
pub fn srs_peaks(a: &[f64], h: f64, f: &[f64], zeta: f64) -> crate::Result<[Vec<f64>; 5]> {
    if a.is_empty() {
        return Err("the record has no samples".into());
    }
    if !(0.0..1.0).contains(&zeta) {
        return Err("the oscillators' damping must be below critical".into());
    }
    let mut out: [Vec<f64>; 5] = Default::default();
    for row in out.iter_mut() {
        row.reserve(f.len());
    }
    for block in f.chunks(LANES) {
        // A short last block repeats its last oscillator in the spare lanes, whose results are dropped.
        let osc: [Osc; LANES] = std::array::from_fn(|l| Osc::new(block[l.min(block.len() - 1)], zeta, h));
        let mut up: [Peak; LANES] = std::array::from_fn(|_| Peak::new());
        let mut down: [Peak; LANES] = std::array::from_fn(|_| Peak::new());
        let mut z: [Peak; LANES] = std::array::from_fn(|_| Peak::new());
        let (mut prev_y, mut prev_z) = ([f64::NAN; LANES], [f64::NAN; LANES]);
        let mut eta = [C::new(0.0, 0.0); LANES];
        let mut u_prev = 0.0;
        for (k, &u) in a.iter().enumerate() {
            for l in 0..LANES {
                let o = &osc[l];
                // lfilter([c1, c0], [1, −μ]): η_k = c1 u_k + (c0 u_{k−1} + μ η_{k−1})
                eta[l] = o.c1.scale(u).add(o.c0.scale(u_prev).add(o.mu.mul(eta[l])));
                let y = 2.0 * o.lam2.mul(eta[l]).re;
                let zk = (2.0 * eta[l].re).abs();
                up[l].see(k, y, prev_y[l]);
                down[l].see(k, -y, -prev_y[l]);
                z[l].see(k, zk, prev_z[l]);
                (prev_y[l], prev_z[l]) = (y, zk);
            }
            u_prev = u;
        }
        for l in 0..block.len() {
            let (r_up, r_down, r_z) = residual(&osc[l], eta[l]);
            out[0].push(up[l].refined());
            out[1].push(down[l].refined());
            out[2].push(r_up);
            out[3].push(r_down);
            out[4].push(z[l].refined().max(r_z));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_step_drives_an_undamped_oscillator_to_twice_its_static_response() {
        // A base step of 1 m/s² (its first sample a ramp up): the relative displacement peaks at 2/ω².
        let f = 10.0;
        let w = TAU * f;
        let h = 1e-4;
        let a: Vec<f64> = (0..2000).map(|k| if k == 0 { 0.0 } else { 1.0 }).collect();
        let [_, _, _, _, z] = srs_peaks(&a, h, &[f], 0.0).unwrap();
        assert!((z[0] * w * w - 2.0).abs() < 1e-3, "{}", z[0] * w * w);
    }

    #[test]
    fn the_first_of_equal_peaks_wins_and_an_end_peak_is_not_refined() {
        let mut p = Peak::new();
        for (k, v) in [0.0, 1.0, 0.5, 1.0, 0.0].into_iter().enumerate() {
            p.see(k, v, if k == 0 { f64::NAN } else { [0.0, 1.0, 0.5, 1.0][k - 1] });
        }
        assert_eq!(p.at, 1);
        let mut q = Peak::new();
        for (k, v) in [0.0, 1.0, 2.0].into_iter().enumerate() {
            q.see(k, v, k as f64 - 1.0);
        }
        assert_eq!(q.refined(), 2.0);
    }

    #[test]
    fn bad_damping_or_an_empty_record_is_an_error() {
        assert!(srs_peaks(&[1.0], 1e-3, &[1.0], 1.0).is_err());
        assert!(srs_peaks(&[], 1e-3, &[1.0], 0.05).is_err());
    }
}
