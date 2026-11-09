use openlab_core::{CoreError, MxArray};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    One,       // 1-norm: max column sum
    Two,       // 2-norm: max singular value
    Inf,       // Inf-norm: max row sum
    Frobenius, // Frobenius norm
}

/// Backslash operator A \ b (mldivide).
///
/// Polyalgorithmic decision tree matching MATLAB:
/// 1. If A is empty -> returns empty.
/// 2. If A is square:
///    - If symmetric positive definite -> Cholesky
///    - Otherwise -> LU with partial pivoting
///    - If singular -> SVD pseudoinverse least squares
/// 3. If A is rectangular:
///    - If M > N (overdetermined) -> Householder QR least squares
///    - If M < N (underdetermined) -> SVD minimum-norm solution
pub fn mldivide(a: &MxArray, b: &MxArray) -> Result<MxArray, CoreError> {
    if !a.shape().is_2d() || !b.shape().is_2d() {
        return Err(CoreError::InvalidOperation(
            "mldivide is only defined for 2D matrices.".into(),
        ));
    }

    let m = a.rows();
    let n = a.cols();

    if b.rows() != m {
        return Err(CoreError::DimensionMismatch {
            expected: vec![m, 1],
            actual: vec![b.rows(), b.cols()],
        });
    }

    if m == 0 || n == 0 {
        return Ok(MxArray::zeros(n, b.cols()));
    }

    if m == n {
        // Attempt Cholesky if symmetric
        if is_symmetric(a) {
            if let Ok(chol) = crate::cholesky::decompose(a) {
                if let Ok(x) = chol.solve(b) {
                    return Ok(x);
                }
            }
        }

        // Use LU decomposition with partial pivoting
        let lu = crate::lu::decompose(a)?;
        match lu.solve(b) {
            Ok(x) => Ok(x),
            Err(_) => {
                // Singular or ill-conditioned: fall back to SVD pseudoinverse
                let svd = crate::svd::decompose(a, true)?;
                let pinv_a = svd.pinv(None)?;
                crate::gemm::matmul(&pinv_a, b)
            }
        }
    } else if m > n {
        // Overdetermined system: QR least-squares
        let qr = crate::qr::decompose(a, true)?;
        match qr.solve_least_squares(b) {
            Ok(x) => Ok(x),
            Err(_) => {
                let svd = crate::svd::decompose(a, true)?;
                let pinv_a = svd.pinv(None)?;
                crate::gemm::matmul(&pinv_a, b)
            }
        }
    } else {
        // Underdetermined system (M < N): minimum norm solution via SVD
        let svd = crate::svd::decompose(a, true)?;
        let pinv_a = svd.pinv(None)?;
        crate::gemm::matmul(&pinv_a, b)
    }
}

/// Slash operator b / A (mrdivide).
/// Defined as: b / A = (A^T \ b^T)^T.
pub fn mrdivide(b: &MxArray, a: &MxArray) -> Result<MxArray, CoreError> {
    let at = a.transpose()?;
    let bt = b.transpose()?;
    let xt = mldivide(&at, &bt)?;
    xt.transpose()
}

/// Matrix inversion inv(A) = A \ I.
pub fn inv(a: &MxArray) -> Result<MxArray, CoreError> {
    if !a.shape().is_2d() || a.rows() != a.cols() {
        return Err(CoreError::NonSquareMatrix {
            rows: a.rows(),
            cols: a.cols(),
        });
    }

    let n = a.rows();
    if n == 0 {
        return Ok(MxArray::zeros(0, 0));
    }

    let eye = MxArray::eye(n);
    mldivide(a, &eye)
}

/// Matrix determinant det(A) via LU decomposition.
pub fn det(a: &MxArray) -> Result<f64, CoreError> {
    if !a.shape().is_2d() || a.rows() != a.cols() {
        return Err(CoreError::NonSquareMatrix {
            rows: a.rows(),
            cols: a.cols(),
        });
    }

    let n = a.rows();
    if n == 0 {
        return Ok(1.0); // det([]) = 1 in MATLAB
    }

    let lu = crate::lu::decompose(a)?;
    lu.det()
}

/// Matrix trace: sum of diagonal entries sum(diag(A)).
pub fn trace(a: &MxArray) -> Result<f64, CoreError> {
    if !a.shape().is_2d() {
        return Err(CoreError::InvalidOperation(
            "trace is only defined for 2D matrices.".into(),
        ));
    }

    let min_dim = std::cmp::min(a.rows(), a.cols());
    let mut sum = 0.0;
    for i in 1..=min_dim {
        sum += a.get_2d(i, i)?;
    }
    Ok(sum)
}

/// Matrix norm for various p-norms.
pub fn norm(a: &MxArray, p: NormType) -> Result<f64, CoreError> {
    if !a.shape().is_2d() {
        return Err(CoreError::InvalidOperation(
            "norm is only defined for 2D matrices.".into(),
        ));
    }

    let m = a.rows();
    let n = a.cols();
    if m == 0 || n == 0 {
        return Ok(0.0);
    }

    match p {
        NormType::One => {
            // Maximum absolute column sum
            let mut max_sum = 0.0;
            for j in 1..=n {
                let mut col_sum = 0.0;
                for i in 1..=m {
                    col_sum += a.get_2d(i, j)?.abs();
                }
                if col_sum > max_sum {
                    max_sum = col_sum;
                }
            }
            Ok(max_sum)
        }
        NormType::Inf => {
            // Maximum absolute row sum
            let mut max_sum = 0.0;
            for i in 1..=m {
                let mut row_sum = 0.0;
                for j in 1..=n {
                    row_sum += a.get_2d(i, j)?.abs();
                }
                if row_sum > max_sum {
                    max_sum = row_sum;
                }
            }
            Ok(max_sum)
        }
        NormType::Frobenius => {
            // Square root of sum of squared elements
            let slice = match a.as_slice_f64() {
                Some(s) => s,
                None => {
                    return Err(CoreError::IncompatibleTypes {
                        expected: "double",
                        actual: a.class_name(),
                    })
                }
            };
            let mut sum_sq = 0.0;
            for &val in slice {
                sum_sq += val * val;
            }
            Ok(sum_sq.sqrt())
        }
        NormType::Two => {
            // Maximum singular value
            let svd = crate::svd::decompose(a, true)?;
            Ok(svd.singular_values.first().copied().unwrap_or(0.0))
        }
    }
}

/// Matrix condition number cond(A) = s_max / s_min.
pub fn cond(a: &MxArray) -> Result<f64, CoreError> {
    let svd = crate::svd::decompose(a, true)?;
    Ok(svd.cond())
}

/// Matrix rank estimation via SVD.
pub fn rank(a: &MxArray, tol: Option<f64>) -> Result<usize, CoreError> {
    let svd = crate::svd::decompose(a, true)?;
    Ok(svd.rank(tol))
}

/// Moore-Penrose pseudoinverse pinv(A).
pub fn pinv(a: &MxArray, tol: Option<f64>) -> Result<MxArray, CoreError> {
    let svd = crate::svd::decompose(a, true)?;
    svd.pinv(tol)
}

fn is_symmetric(a: &MxArray) -> bool {
    let m = a.rows();
    let n = a.cols();
    if m != n {
        return false;
    }
    for i in 1..=m {
        for j in (i + 1)..=n {
            let v1 = a.get_2d(i, j).unwrap_or(0.0);
            let v2 = a.get_2d(j, i).unwrap_or(0.0);
            if (v1 - v2).abs() > 1e-12 {
                return false;
            }
        }
    }
    true
}
