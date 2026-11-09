use openlab_core::{CoreError, MxArray, Shape};

#[derive(Debug, Clone)]
pub struct LuDecomposition {
    pub l: MxArray,
    pub u: MxArray,
    pub p: MxArray,
    pub pivots: Vec<usize>,
    pub num_swaps: usize,
}

impl LuDecomposition {
    /// Computes the determinant from the LU decomposition of a square matrix:
    /// det(A) = (-1)^swaps * prod(diag(U))
    pub fn det(&self) -> Result<f64, CoreError> {
        let n = self.u.rows();
        if self.u.cols() != n {
            return Err(CoreError::NonSquareMatrix {
                rows: n,
                cols: self.u.cols(),
            });
        }

        let mut d = if self.num_swaps % 2 == 1 { -1.0 } else { 1.0 };
        for i in 1..=n {
            d *= self.u.get_2d(i, i)?;
        }
        Ok(d)
    }

    /// Solves the linear system A * x = b using forward and back substitution.
    pub fn solve(&self, b: &MxArray) -> Result<MxArray, CoreError> {
        let n = self.l.rows();
        if self.u.cols() != n {
            return Err(CoreError::NonSquareMatrix {
                rows: n,
                cols: self.u.cols(),
            });
        }
        if b.rows() != n {
            return Err(CoreError::DimensionMismatch {
                expected: vec![n, 1],
                actual: vec![b.rows(), b.cols()],
            });
        }

        let b_cols = b.cols();
        let mut x_out = vec![0.0; n * b_cols];

        for col in 1..=b_cols {
            // Step 1: Apply permutation b_p = P * b
            let mut y = vec![0.0; n];
            for i in 0..n {
                let orig_row = self.pivots[i] + 1; // 1-based
                y[i] = b.get_2d(orig_row, col)?;
            }

            // Step 2: Forward substitution L * y = b_p (L has 1.0 on diagonal)
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..i {
                    sum += self.l.get_2d(i + 1, j + 1)? * y[j];
                }
                y[i] -= sum;
            }

            // Step 3: Back substitution U * x = y
            let mut x = vec![0.0; n];
            for i in (0..n).rev() {
                let diag = self.u.get_2d(i + 1, i + 1)?;
                if diag.abs() < 1e-15 {
                    return Err(CoreError::InvalidOperation(
                        "Matrix is singular to working precision.".into(),
                    ));
                }

                let mut sum = 0.0;
                for j in (i + 1)..n {
                    sum += self.u.get_2d(i + 1, j + 1)? * x[j];
                }
                x[i] = (y[i] - sum) / diag;
            }

            // Store in output column-major
            let out_col_offset = (col - 1) * n;
            for i in 0..n {
                x_out[i + out_col_offset] = x[i];
            }
        }

        MxArray::from_f64_vec(Shape::matrix(n, b_cols), x_out)
    }
}

/// Computes the LU decomposition with partial pivoting: P * A = L * U.
pub fn decompose(a: &MxArray) -> Result<LuDecomposition, CoreError> {
    if !a.shape().is_2d() {
        return Err(CoreError::InvalidOperation(
            "LU decomposition is only defined for 2D matrices.".into(),
        ));
    }

    let m = a.rows();
    let n = a.cols();
    let k_min = std::cmp::min(m, n);

    if m == 0 || n == 0 {
        return Ok(LuDecomposition {
            l: MxArray::zeros(m, k_min),
            u: MxArray::zeros(k_min, n),
            p: MxArray::eye(m),
            pivots: (0..m).collect(),
            num_swaps: 0,
        });
    }

    // Work on mutable copy
    let mut a_work = a.clone();
    let mut pivots: Vec<usize> = (0..m).collect();
    let mut num_swaps = 0;

    for k in 0..k_min {
        // Find pivot in column k, starting from row k
        let mut max_val = 0.0;
        let mut pivot_row = k;

        for i in k..m {
            let val = a_work.get_2d(i + 1, k + 1)?.abs();
            if val > max_val {
                max_val = val;
                pivot_row = i;
            }
        }

        // Swap rows if needed
        if pivot_row != k {
            for j in 0..n {
                let v1 = a_work.get_2d(k + 1, j + 1)?;
                let v2 = a_work.get_2d(pivot_row + 1, j + 1)?;
                a_work.set_2d(k + 1, j + 1, v2)?;
                a_work.set_2d(pivot_row + 1, j + 1, v1)?;
            }
            pivots.swap(k, pivot_row);
            num_swaps += 1;
        }

        let pivot = a_work.get_2d(k + 1, k + 1)?;
        if pivot.abs() > 1e-15 {
            for i in (k + 1)..m {
                let factor = a_work.get_2d(i + 1, k + 1)? / pivot;
                a_work.set_2d(i + 1, k + 1, factor)?;

                for j in (k + 1)..n {
                    let old_val = a_work.get_2d(i + 1, j + 1)?;
                    let sub_val = factor * a_work.get_2d(k + 1, j + 1)?;
                    a_work.set_2d(i + 1, j + 1, old_val - sub_val)?;
                }
            }
        }
    }

    // Extract L (M x K_min) and U (K_min x N)
    let mut l = MxArray::zeros(m, k_min);
    let mut u = MxArray::zeros(k_min, n);

    for i in 1..=m {
        for j in 1..=k_min {
            if i > j {
                l.set_2d(i, j, a_work.get_2d(i, j)?)?;
            } else if i == j {
                l.set_2d(i, j, 1.0)?;
            }
        }
    }

    for i in 1..=k_min {
        for j in 1..=n {
            if i <= j {
                u.set_2d(i, j, a_work.get_2d(i, j)?)?;
            }
        }
    }

    // Build permutation matrix P (M x M)
    let mut p = MxArray::zeros(m, m);
    for (row, &orig_row) in pivots.iter().enumerate() {
        p.set_2d(row + 1, orig_row + 1, 1.0)?;
    }

    Ok(LuDecomposition {
        l,
        u,
        p,
        pivots,
        num_swaps,
    })
}

