use openlab_core::{CoreError, MxArray, Shape};

#[derive(Debug, Clone)]
pub struct CholeskyDecomposition {
    pub l: MxArray, // Lower triangular L such that A = L * L^T
    pub r: MxArray, // Upper triangular R such that A = R^T * R (MATLAB default chol(A))
}

impl CholeskyDecomposition {
    /// Solves A * x = b for symmetric positive definite matrix A using Cholesky factors.
    pub fn solve(&self, b: &MxArray) -> Result<MxArray, CoreError> {
        let n = self.l.rows();
        if b.rows() != n {
            return Err(CoreError::DimensionMismatch {
                expected: vec![n, 1],
                actual: vec![b.rows(), b.cols()],
            });
        }

        let b_cols = b.cols();
        let mut x_out = vec![0.0; n * b_cols];

        for col in 1..=b_cols {
            // Forward substitution: L * y = b
            let mut y = vec![0.0; n];
            for i in 1..=n {
                let mut sum = 0.0;
                for j in 1..i {
                    sum += self.l.get_2d(i, j)? * y[j - 1];
                }
                let diag = self.l.get_2d(i, i)?;
                y[i - 1] = (b.get_2d(i, col)? - sum) / diag;
            }

            // Back substitution: L^T * x = y  (or R * x = y)
            let mut x = vec![0.0; n];
            for i in (1..=n).rev() {
                let mut sum = 0.0;
                for j in (i + 1)..=n {
                    sum += self.r.get_2d(i, j)? * x[j - 1];
                }
                let diag = self.r.get_2d(i, i)?;
                x[i - 1] = (y[i - 1] - sum) / diag;
            }

            let col_offset = (col - 1) * n;
            for i in 0..n {
                x_out[i + col_offset] = x[i];
            }
        }

        MxArray::from_f64_vec(Shape::matrix(n, b_cols), x_out)
    }
}

/// Computes the Cholesky decomposition of a symmetric positive-definite matrix A.
/// Returns L (lower) and R (upper) such that A = L * L^T = R^T * R.
pub fn decompose(a: &MxArray) -> Result<CholeskyDecomposition, CoreError> {
    if !a.shape().is_2d() || a.rows() != a.cols() {
        return Err(CoreError::NonSquareMatrix {
            rows: a.rows(),
            cols: a.cols(),
        });
    }

    let n = a.rows();
    if n == 0 {
        return Ok(CholeskyDecomposition {
            l: MxArray::zeros(0, 0),
            r: MxArray::zeros(0, 0),
        });
    }

    // Verify symmetry within numerical precision
    for i in 1..=n {
        for j in (i + 1)..=n {
            let diff = (a.get_2d(i, j)? - a.get_2d(j, i)?).abs();
            if diff > 1e-12 {
                return Err(CoreError::InvalidOperation(
                    "Matrix must be symmetric for Cholesky decomposition.".into(),
                ));
            }
        }
    }

    let mut l = MxArray::zeros(n, n);

    for j in 1..=n {
        let mut d_sum = 0.0;
        for k in 1..j {
            let l_jk = l.get_2d(j, k)?;
            d_sum += l_jk * l_jk;
        }

        let a_jj = a.get_2d(j, j)?;
        let s = a_jj - d_sum;
        if s <= 0.0 {
            return Err(CoreError::InvalidOperation(
                "Matrix must be positive definite.".into(),
            ));
        }

        let l_jj = s.sqrt();
        l.set_2d(j, j, l_jj)?;

        for i in (j + 1)..=n {
            let mut off_sum = 0.0;
            for k in 1..j {
                off_sum += l.get_2d(i, k)? * l.get_2d(j, k)?;
            }
            let a_ij = a.get_2d(i, j)?;
            let l_ij = (a_ij - off_sum) / l_jj;
            l.set_2d(i, j, l_ij)?;
        }
    }

    let r = l.transpose()?;

    Ok(CholeskyDecomposition { l, r })
}
