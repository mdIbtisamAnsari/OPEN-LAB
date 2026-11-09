use openlab_core::{Complex64, CoreError, MxArray, MxData, Shape};

/// Matrix multiplication C = A * B.
///
/// Features:
/// 1. Scalar expansion: if either A or B is 1x1, performs scalar multiplication.
/// 2. Cache-optimized column-major GEMM loop ordering: (j, k, i).
/// 3. Complex matrix multiplication: supports real x complex, complex x real, complex x complex.
/// 4. Handles empty matrix dimensions correctly according to linear algebra rules:
///    - (M x 0) * (0 x N) -> (M x N) zeros.
///    - (0 x K) * (K x N) -> (0 x N) empty.
///    - (M x K) * (K x 0) -> (M x 0) empty.
pub fn matmul(a: &MxArray, b: &MxArray) -> Result<MxArray, CoreError> {
    if !a.shape().is_2d() || !b.shape().is_2d() {
        return Err(CoreError::InvalidOperation(
            "Matrix multiplication is only defined for 2D matrices.".into(),
        ));
    }

    // Scalar expansion rule
    if a.is_scalar() {
        if a.is_complex() {
            let s = a.get_complex_2d(1, 1)?;
            return scalar_mul_complex(b, s);
        } else {
            let s = a.to_scalar()?;
            return b.scalar_mul(s);
        }
    }
    if b.is_scalar() {
        if b.is_complex() {
            let s = b.get_complex_2d(1, 1)?;
            return scalar_mul_complex(a, s);
        } else {
            let s = b.to_scalar()?;
            return a.scalar_mul(s);
        }
    }

    let m = a.rows();
    let k_a = a.cols();
    let k_b = b.rows();
    let n = b.cols();

    if k_a != k_b {
        return Err(CoreError::DimensionMismatch {
            expected: vec![m, k_a],
            actual: vec![k_b, n],
        });
    }

    let is_complex = a.is_complex() || b.is_complex();

    if is_complex {
        matmul_complex(a, b, m, k_a, n)
    } else {
        matmul_real(a, b, m, k_a, n)
    }
}

/// Real double GEMM
fn matmul_real(
    a: &MxArray,
    b: &MxArray,
    m: usize,
    k_dim: usize,
    n: usize,
) -> Result<MxArray, CoreError> {
    let out_shape = Shape::matrix(m, n);
    if m == 0 || n == 0 {
        return Ok(MxArray::zeros(m, n));
    }

    let a_slice = match a.as_slice_f64() {
        Some(s) => s,
        None => {
            return Err(CoreError::IncompatibleTypes {
                expected: "double",
                actual: a.class_name(),
            })
        }
    };

    let b_slice = match b.as_slice_f64() {
        Some(s) => s,
        None => {
            return Err(CoreError::IncompatibleTypes {
                expected: "double",
                actual: b.class_name(),
            })
        }
    };

    let mut c_data = vec![0.0; m * n];

    // If inner dimension is 0, sum_{k=0}^{-1} is empty sum = 0
    if k_dim == 0 {
        return MxArray::from_f64_vec(out_shape, c_data);
    }

    // Cache-optimized column-major GEMM: (j, k, i)
    for j in 0..n {
        let b_col_offset = j * k_dim;
        let c_col_offset = j * m;

        for k in 0..k_dim {
            let b_val = b_slice[k + b_col_offset];
            let a_col_offset = k * m;

            // Inner loop along rows i has stride 1
            for i in 0..m {
                c_data[i + c_col_offset] += a_slice[i + a_col_offset] * b_val;
            }
        }
    }

    MxArray::from_f64_vec(out_shape, c_data)
}

/// Complex double GEMM
fn matmul_complex(
    a: &MxArray,
    b: &MxArray,
    m: usize,
    k_dim: usize,
    n: usize,
) -> Result<MxArray, CoreError> {
    let out_shape = Shape::matrix(m, n);
    if m == 0 || n == 0 {
        return Ok(MxArray::zeros(m, n));
    }

    let a_complex = to_complex_vec(a)?;
    let b_complex = to_complex_vec(b)?;

    let mut c_data = vec![Complex64::zero(); m * n];

    if k_dim == 0 {
        return MxArray::from_complex_vec(out_shape, c_data);
    }

    for j in 0..n {
        let b_col_offset = j * k_dim;
        let c_col_offset = j * m;

        for k in 0..k_dim {
            let b_val = b_complex[k + b_col_offset];
            let a_col_offset = k * m;

            for i in 0..m {
                c_data[i + c_col_offset] =
                    c_data[i + c_col_offset] + a_complex[i + a_col_offset] * b_val;
            }
        }
    }

    MxArray::from_complex_vec(out_shape, c_data)
}

fn to_complex_vec(arr: &MxArray) -> Result<Vec<Complex64>, CoreError> {
    match arr.data() {
        MxData::Double(v) => Ok(v.iter().map(|&x| Complex64::real(x)).collect()),
        MxData::ComplexDouble(v) => Ok((**v).clone()),
        _ => Err(CoreError::IncompatibleTypes {
            expected: "numeric double",
            actual: arr.class_name(),
        }),
    }
}

fn scalar_mul_complex(arr: &MxArray, s: Complex64) -> Result<MxArray, CoreError> {
    let mut vec = to_complex_vec(arr)?;
    for elem in vec.iter_mut() {
        *elem = *elem * s;
    }
    MxArray::from_complex_vec(arr.shape().clone(), vec)
}

/// Matrix-vector product y = A * x.
pub fn matvec(a: &MxArray, x: &MxArray) -> Result<MxArray, CoreError> {
    matmul(a, x)
}

/// Matrix power A^p for a square matrix A and non-negative integer p.
pub fn matpow(a: &MxArray, p: usize) -> Result<MxArray, CoreError> {
    if !a.shape().is_2d() || a.rows() != a.cols() {
        return Err(CoreError::NonSquareMatrix {
            rows: a.rows(),
            cols: a.cols(),
        });
    }

    let n = a.rows();
    if p == 0 {
        return Ok(MxArray::eye(n));
    }
    if p == 1 {
        return Ok(a.clone());
    }

    let mut base = a.clone();
    let mut result = MxArray::eye(n);
    let mut exp = p;

    while exp > 0 {
        if exp % 2 == 1 {
            result = matmul(&result, &base)?;
        }
        if exp > 1 {
            base = matmul(&base, &base)?;
        }
        exp /= 2;
    }

    Ok(result)
}
