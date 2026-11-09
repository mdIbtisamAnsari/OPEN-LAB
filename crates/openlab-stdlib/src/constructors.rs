use openlab_core::{CoreError, MxArray, Shape};

// -----------------------------------------------------------------------------
// Clean-Room Pseudorandom Number Generator (Xorshift128+)
// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Prng {
    s: [u64; 2],
}

impl Default for Prng {
    fn default() -> Self {
        Self {
            s: [0x123456789abcdef0, 0xfedcba9876543210],
        }
    }
}

impl Prng {
    pub fn new(seed: u64) -> Self {
        let s0 = if seed == 0 { 0x54321 } else { seed };
        let s1 = s0.wrapping_mul(6364136223846793005).wrapping_add(1);
        Self { s: [s0, s1] }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut s1 = self.s[0];
        let s0 = self.s[1];
        self.s[0] = s0;
        s1 ^= s1 << 23;
        self.s[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
        self.s[1].wrapping_add(s0)
    }

    /// Uniform floating-point in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        let val = self.next_u64() >> 11;
        (val as f64) * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal N(0, 1) via Box-Muller transform
    pub fn next_randn(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

use std::sync::Mutex;

// Global thread-safe PRNG instance protected by Mutex
static GLOBAL_PRNG: Mutex<Prng> = Mutex::new(Prng {
    s: [0x123456789abcdef0, 0xfedcba9876543210],
});

fn with_prng<F, R>(f: F) -> R
where
    F: FnOnce(&mut Prng) -> R,
{
    let mut guard = GLOBAL_PRNG.lock().unwrap();
    f(&mut guard)
}

// -----------------------------------------------------------------------------
// Matrix Constructors
// -----------------------------------------------------------------------------

pub fn zeros(dims: &[usize]) -> MxArray {
    MxArray::zeros_nd(Shape::new(dims))
}

pub fn ones(dims: &[usize]) -> MxArray {
    MxArray::ones_nd(Shape::new(dims))
}

pub fn eye(m: usize, n: usize) -> MxArray {
    MxArray::eye_mn(m, n)
}

/// Uniform random matrix in [0, 1)
pub fn rand(dims: &[usize]) -> MxArray {
    let shape = Shape::new(dims);
    let n = shape.numel();
    let mut data = Vec::with_capacity(n);
    with_prng(|rng| {
        for _ in 0..n {
            data.push(rng.next_f64());
        }
    });
    MxArray::from_f64_vec(shape, data).unwrap()
}

/// Standard normal N(0, 1) random matrix
pub fn randn(dims: &[usize]) -> MxArray {
    let shape = Shape::new(dims);
    let n = shape.numel();
    let mut data = Vec::with_capacity(n);
    with_prng(|rng| {
        for _ in 0..n {
            data.push(rng.next_randn());
        }
    });
    MxArray::from_f64_vec(shape, data).unwrap()
}

/// Linearly spaced vector: linspace(x1, x2, n)
pub fn linspace(x1: f64, x2: f64, n: usize) -> Result<MxArray, CoreError> {
    if n == 0 {
        return Ok(MxArray::zeros(1, 0));
    }
    if n == 1 {
        return Ok(MxArray::from_scalar(x2));
    }

    let step = (x2 - x1) / ((n - 1) as f64);
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        if i == n - 1 {
            data.push(x2);
        } else {
            data.push(x1 + (i as f64) * step);
        }
    }

    MxArray::from_f64_vec(Shape::row_vector(n), data)
}

/// Logarithmically spaced vector: logspace(a, b, n)
/// Generates n points logarithmically spaced between 10^a and 10^b.
pub fn logspace(a: f64, b: f64, n: usize) -> Result<MxArray, CoreError> {
    let linear = linspace(a, b, n)?;
    crate::elmath::apply_unary_f64(&linear, |x| 10.0_f64.powf(x))
}

/// Extracts diagonal or builds diagonal matrix:
/// - If v is a vector, diag(v, k) creates a diagonal matrix with v on the k-th diagonal.
/// - If v is a 2D matrix, diag(v, k) extracts the k-th diagonal as a column vector.
pub fn diag(v: &MxArray, k: i32) -> Result<MxArray, CoreError> {
    if !v.shape().is_2d() {
        return Err(CoreError::InvalidOperation(
            "diag requires a 1D vector or 2D matrix.".into(),
        ));
    }

    if v.is_vector() {
        let len = v.numel();
        let k_abs = k.unsigned_abs() as usize;
        let n = len + k_abs;
        let mut mat = MxArray::zeros(n, n);

        let (row_start, col_start) = if k >= 0 {
            (1, 1 + k_abs)
        } else {
            (1 + k_abs, 1)
        };

        for i in 0..len {
            let val = v.get_linear_0based(i)?;
            mat.set_2d(row_start + i, col_start + i, val)?;
        }

        Ok(mat)
    } else {
        // Extract diagonal from 2D matrix
        let m = v.rows();
        let n = v.cols();
        let mut diag_vals = Vec::new();

        let (mut r, mut c) = if k >= 0 {
            (1, 1 + (k as usize))
        } else {
            (1 + (k.unsigned_abs() as usize), 1)
        };

        while r <= m && c <= n {
            diag_vals.push(v.get_2d(r, c)?);
            r += 1;
            c += 1;
        }

        let out_len = diag_vals.len();
        MxArray::from_f64_vec(Shape::col_vector(out_len), diag_vals)
    }
}

/// 2D grid generation: [X, Y] = meshgrid(x, y)
/// In MATLAB:
/// If x has length N and y has length M:
/// X is M-by-N, where each row is a copy of x.
/// Y is M-by-N, where each column is a copy of y.
pub fn meshgrid(x: &MxArray, y: &MxArray) -> Result<(MxArray, MxArray), CoreError> {
    let nx = x.numel();
    let ny = y.numel();

    let mut x_mat = MxArray::zeros(ny, nx);
    let mut y_mat = MxArray::zeros(ny, nx);

    for r in 1..=ny {
        let y_val = y.get_linear_0based(r - 1)?;
        for c in 1..=nx {
            let x_val = x.get_linear_0based(c - 1)?;
            x_mat.set_2d(r, c, x_val)?;
            y_mat.set_2d(r, c, y_val)?;
        }
    }

    Ok((x_mat, y_mat))
}
