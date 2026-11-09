use openlab_core::{CoreError, MxArray, Shape};

#[derive(Debug, Clone)]
pub struct QrDecomposition {
    pub q: MxArray,
    pub r: MxArray,
}

impl QrDecomposition {
    /// Solves the linear system or least-squares problem A * x = b.
    /// For M >= N, solves the overdetermined least-squares problem min ||A*x - b||_2.
    pub fn solve_least_squares(&self, b: &MxArray) -> Result<MxArray, CoreError> {
        let m = self.q.rows();
        let n = self.r.cols();
        if b.rows() != m {
            return Err(CoreError::DimensionMismatch {
                expected: vec![m, 1],
                actual: vec![b.rows(), b.cols()],
            });
        }

        let b_cols = b.cols();
        // Compute d = Q^T * b
        let qt = self.q.transpose()?;
        let d = crate::gemm::matmul(&qt, b)?;

        // Solve upper triangular R(1..N, 1..N) * x = d(1..N, :)
        let mut x_out = vec![0.0; n * b_cols];

        for col in 1..=b_cols {
            let mut x = vec![0.0; n];
            for i in (1..=n).rev() {
                let diag = self.r.get_2d(i, i)?;
                if diag.abs() < 1e-15 {
                    return Err(CoreError::InvalidOperation(
                        "Rank-deficient matrix in QR solve.".into(),
                    ));
                }

                let mut sum = 0.0;
                for j in (i + 1)..=n {
                    sum += self.r.get_2d(i, j)? * x[j - 1];
                }
                x[i - 1] = (d.get_2d(i, col)? - sum) / diag;
            }

            let col_offset = (col - 1) * n;
            for i in 0..n {
                x_out[i + col_offset] = x[i];
            }
        }

        MxArray::from_f64_vec(Shape::matrix(n, b_cols), x_out)
    }
}

/// Computes the QR decomposition of matrix A using Householder reflections.
/// If `economy` is true, returns Q (M x K) and R (K x N) where K = min(M, N).
pub fn decompose(a: &MxArray, economy: bool) -> Result<QrDecomposition, CoreError> {
    if !a.shape().is_2d() {
        return Err(CoreError::InvalidOperation(
            "QR decomposition is only defined for 2D matrices.".into(),
        ));
    }

    let m = a.rows();
    let n = a.cols();
    let k_min = std::cmp::min(m, n);

    if m == 0 || n == 0 {
        return Ok(QrDecomposition {
            q: MxArray::zeros(m, if economy { k_min } else { m }),
            r: MxArray::zeros(if economy { k_min } else { m }, n),
        });
    }

    let mut r = a.clone();
    let mut q = MxArray::eye(m);

    for k in 1..=k_min {
        // Extract vector x = R(k..m, k)
        let sub_len = m - k + 1;
        let mut x = vec![0.0; sub_len];
        let mut norm_sq = 0.0;
        for i in 0..sub_len {
            let val = r.get_2d(k + i, k)?;
            x[i] = val;
            norm_sq += val * val;
        }

        let norm_x = norm_sq.sqrt();
        if norm_x > 1e-15 {
            let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
            let alpha = -sign * norm_x;

            let mut v = x;
            v[0] -= alpha;

            let mut v_norm_sq = 0.0;
            for &vi in &v {
                v_norm_sq += vi * vi;
            }
            let v_norm = v_norm_sq.sqrt();

            if v_norm > 1e-15 {
                for vi in &mut v {
                    *vi /= v_norm;
                }

                // Apply Householder reflector H = I - 2 * v * v^T to R(k..m, k..n)
                for j in k..=n {
                    let mut dot = 0.0;
                    for i in 0..sub_len {
                        dot += v[i] * r.get_2d(k + i, j)?;
                    }
                    let factor = 2.0 * dot;
                    for i in 0..sub_len {
                        let old = r.get_2d(k + i, j)?;
                        r.set_2d(k + i, j, old - factor * v[i])?;
                    }
                }

                // Apply Householder reflector to Q: Q = Q * H
                // Equivalent to: Q(1..m, k..m) = Q(1..m, k..m) - 2 * (Q * v) * v^T
                for i in 1..=m {
                    let mut dot = 0.0;
                    for l in 0..sub_len {
                        dot += q.get_2d(i, k + l)? * v[l];
                    }
                    let factor = 2.0 * dot;
                    for l in 0..sub_len {
                        let old = q.get_2d(i, k + l)?;
                        q.set_2d(i, k + l, old - factor * v[l])?;
                    }
                }
            }
        }
    }

    // Clean up numerical noise below diagonal of R
    for i in 1..=m {
        for j in 1..=n {
            if i > j {
                let _ = r.set_2d(i, j, 0.0);
            }
        }
    }

    if economy && m > n {
        let q_econ = q.submatrix(1, m, 1, n)?;
        let r_econ = r.submatrix(1, n, 1, n)?;
        Ok(QrDecomposition {
            q: q_econ,
            r: r_econ,
        })
    } else {
        Ok(QrDecomposition { q, r })
    }
}

