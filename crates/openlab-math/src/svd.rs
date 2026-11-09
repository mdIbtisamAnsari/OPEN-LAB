use openlab_core::{CoreError, MxArray};

#[derive(Debug, Clone)]
pub struct SvdDecomposition {
    pub u: MxArray, // M x M (or M x K if economy)
    pub s: MxArray, // M x N (or K x K if economy)
    pub v: MxArray, // N x N (or N x K if economy)
    pub singular_values: Vec<f64>,
}

impl SvdDecomposition {
    /// Estimates matrix rank based on singular values and tolerance:
    /// rank = count(s > tol), where default tol = max(M, N) * eps(max(s))
    pub fn rank(&self, tol: Option<f64>) -> usize {
        if self.singular_values.is_empty() {
            return 0;
        }
        let max_s = self.singular_values[0];
        let m = self.u.rows();
        let n = self.v.rows();
        let default_tol = (std::cmp::max(m, n) as f64) * max_s * f64::EPSILON;
        let threshold = tol.unwrap_or(default_tol);

        self.singular_values.iter().filter(|&&s| s > threshold).count()
    }

    /// Condition number cond(A) = s_max / s_min.
    pub fn cond(&self) -> f64 {
        if self.singular_values.is_empty() {
            return 0.0;
        }
        let s_max = self.singular_values[0];
        let s_min = *self.singular_values.last().unwrap();
        if s_min == 0.0 {
            f64::INFINITY
        } else {
            s_max / s_min
        }
    }

    /// Moore-Penrose Pseudoinverse A^+ = V * S^+ * U^T
    pub fn pinv(&self, tol: Option<f64>) -> Result<MxArray, CoreError> {
        let m = self.u.rows();
        let n = self.v.rows();
        let k = self.singular_values.len();

        let max_s = if k > 0 { self.singular_values[0] } else { 0.0 };
        let default_tol = (std::cmp::max(m, n) as f64) * max_s * f64::EPSILON;
        let threshold = tol.unwrap_or(default_tol);

        // Construct S^+ (N x M)
        let mut s_plus = MxArray::zeros(n, m);
        for i in 0..k {
            let s_val = self.singular_values[i];
            if s_val > threshold {
                s_plus.set_2d(i + 1, i + 1, 1.0 / s_val)?;
            }
        }

        let ut = self.u.transpose()?;
        let vs = crate::gemm::matmul(&self.v, &s_plus)?;
        crate::gemm::matmul(&vs, &ut)
    }
}

/// Computes the Singular Value Decomposition of matrix A: A = U * S * V^T.
pub fn decompose(a: &MxArray, economy: bool) -> Result<SvdDecomposition, CoreError> {
    if !a.shape().is_2d() {
        return Err(CoreError::InvalidOperation(
            "SVD is only defined for 2D matrices.".into(),
        ));
    }

    let m = a.rows();
    let n = a.cols();

    if m == 0 || n == 0 {
        return Ok(SvdDecomposition {
            u: MxArray::zeros(m, if economy { 0 } else { m }),
            s: MxArray::zeros(if economy { 0 } else { m }, n),
            v: MxArray::zeros(n, if economy { 0 } else { n }),
            singular_values: Vec::new(),
        });
    }

    // If M < N, compute SVD of A^T and swap U and V
    if m < n {
        let at = a.transpose()?;
        let svd_t = decompose(&at, economy)?;
        let u = svd_t.v;
        let v = svd_t.u;
        let s = svd_t.s.transpose()?;
        return Ok(SvdDecomposition {
            u,
            s,
            v,
            singular_values: svd_t.singular_values,
        });
    }

    // Here M >= N
    // Phase 1: Golub-Kahan Bidiagonalization
    let mut u = MxArray::eye(m);
    let mut v = MxArray::eye(n);
    let mut b = a.clone();

    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n]; // e[i] is superdiagonal B(i, i+1)

    for i in 0..n {
        // Left Householder transformation (zero out column i below diagonal)
        let sub_len = m - i;
        let mut col_x = vec![0.0; sub_len];
        let mut norm_sq = 0.0;
        for k in 0..sub_len {
            let val = b.get_2d(i + 1 + k, i + 1)?;
            col_x[k] = val;
            norm_sq += val * val;
        }

        let norm_x = norm_sq.sqrt();
        if norm_x > 1e-15 {
            let sign = if col_x[0] >= 0.0 { 1.0 } else { -1.0 };
            let alpha = -sign * norm_x;
            col_x[0] -= alpha;

            let mut v_norm_sq = 0.0;
            for &x in &col_x {
                v_norm_sq += x * x;
            }
            let v_norm = v_norm_sq.sqrt();
            for x in &mut col_x {
                *x /= v_norm;
            }

            // Apply to B(i..m, i..n) from left
            for col in i..n {
                let mut dot = 0.0;
                for k in 0..sub_len {
                    dot += col_x[k] * b.get_2d(i + 1 + k, col + 1)?;
                }
                let factor = 2.0 * dot;
                for k in 0..sub_len {
                    let old = b.get_2d(i + 1 + k, col + 1)?;
                    b.set_2d(i + 1 + k, col + 1, old - factor * col_x[k])?;
                }
            }

            // Accumulate into U
            for row in 0..m {
                let mut dot = 0.0;
                for k in 0..sub_len {
                    dot += u.get_2d(row + 1, i + 1 + k)? * col_x[k];
                }
                let factor = 2.0 * dot;
                for k in 0..sub_len {
                    let old = u.get_2d(row + 1, i + 1 + k)?;
                    u.set_2d(row + 1, i + 1 + k, old - factor * col_x[k])?;
                }
            }
        }
        d[i] = b.get_2d(i + 1, i + 1)?;

        // Right Householder transformation (zero out row i beyond column i+1)
        if i + 2 < n {
            let sub_row_len = n - (i + 1);
            let mut row_x = vec![0.0; sub_row_len];
            let mut norm_row_sq = 0.0;
            for k in 0..sub_row_len {
                let val = b.get_2d(i + 1, i + 2 + k)?;
                row_x[k] = val;
                norm_row_sq += val * val;
            }

            let norm_row = norm_row_sq.sqrt();
            if norm_row > 1e-15 {
                let sign = if row_x[0] >= 0.0 { 1.0 } else { -1.0 };
                let alpha = -sign * norm_row;
                row_x[0] -= alpha;

                let mut v_norm_sq = 0.0;
                for &x in &row_x {
                    v_norm_sq += x * x;
                }
                let v_norm = v_norm_sq.sqrt();
                for x in &mut row_x {
                    *x /= v_norm;
                }

                // Apply to B(i..m, i+1..n) from right
                for row in i..m {
                    let mut dot = 0.0;
                    for k in 0..sub_row_len {
                        dot += b.get_2d(row + 1, i + 2 + k)? * row_x[k];
                    }
                    let factor = 2.0 * dot;
                    for k in 0..sub_row_len {
                        let old = b.get_2d(row + 1, i + 2 + k)?;
                        b.set_2d(row + 1, i + 2 + k, old - factor * row_x[k])?;
                    }
                }

                // Accumulate into V
                for row in 0..n {
                    let mut dot = 0.0;
                    for k in 0..sub_row_len {
                        dot += v.get_2d(row + 1, i + 2 + k)? * row_x[k];
                    }
                    let factor = 2.0 * dot;
                    for k in 0..sub_row_len {
                        let old = v.get_2d(row + 1, i + 2 + k)?;
                        v.set_2d(row + 1, i + 2 + k, old - factor * row_x[k])?;
                    }
                }
            }
            e[i] = b.get_2d(i + 1, i + 2)?;
        } else if i + 1 < n {
            e[i] = b.get_2d(i + 1, i + 2)?;
        }
    }

    // Phase 2: Implicitly Shifted QR Algorithm on Bidiagonal Matrix
    let mut p = n;
    let max_iter = 100 * n;
    let mut iter = 0;

    while p > 1 && iter < max_iter {
        iter += 1;

        // Find split: test if e[q-1] is negligible
        let mut q = p - 1;
        while q > 0 {
            let thresh = f64::EPSILON * (d[q - 1].abs() + d[q].abs());
            if e[q - 1].abs() <= thresh.max(1e-15) {
                e[q - 1] = 0.0;
                break;
            }
            q -= 1;
        }

        if q == p - 1 {
            // Deflation at bottom
            p -= 1;
            continue;
        }

        // Wilkinson shift from trailing 2x2 submatrix of B^T * B
        let d_m1 = d[p - 2];
        let d_m = d[p - 1];
        let e_m1 = e[p - 2];
        let e_m2 = if p >= 3 { e[p - 3] } else { 0.0 };

        let t1 = d_m1 * d_m1 + e_m2 * e_m2;
        let t2 = d_m * d_m + e_m1 * e_m1;
        let b12 = d_m1 * e_m1;

        let delta = (t1 - t2) / 2.0;
        let sign_delta = if delta >= 0.0 { 1.0 } else { -1.0 };
        let mu = if b12.abs() < 1e-15 {
            t2
        } else {
            let denom = delta + sign_delta * (delta * delta + b12 * b12).sqrt();
            if denom.abs() < 1e-15 {
                t2
            } else {
                t2 - (b12 * b12) / denom
            }
        };

        // Check for zero diagonal elements in the active block
        let mut zero_diag = false;
        for k in (q..p).rev() {
            let thresh = f64::EPSILON * (if k > 0 { e[k - 1].abs() } else { 0.0 } + if k < p - 1 { e[k].abs() } else { 0.0 });
            if d[k].abs() <= thresh.max(1e-15) {
                if k == p - 1 && k > 0 {
                    let (c, s, r) = givens(d[k - 1], e[k - 1]);
                    d[k - 1] = r;
                    e[k - 1] = 0.0;
                    for i in 0..n {
                        let v1 = v.get_2d(i + 1, k)?;
                        let v2 = v.get_2d(i + 1, k + 1)?;
                        v.set_2d(i + 1, k, c * v1 + s * v2)?;
                        v.set_2d(i + 1, k + 1, -s * v1 + c * v2)?;
                    }
                }
                zero_diag = true;
                break;
            }
        }
        if zero_diag {
            continue;
        }

        let mut y = d[q] * d[q] - mu;
        let mut z = d[q] * e[q];

        for k in q..(p - 1) {
            // Givens rotation on the right (columns k, k+1)
            let (c, s, r) = givens(y, z);
            if k > q {
                e[k - 1] = r;
            }

            let d_k = d[k];
            let e_k = e[k];
            let d_kp1 = d[k + 1];

            d[k] = c * d_k + s * e_k;
            e[k] = -s * d_k + c * e_k;
            let temp = s * d_kp1;
            d[k + 1] = c * d_kp1;

            // Accumulate into V
            for i in 0..n {
                let v_ik = v.get_2d(i + 1, k + 1)?;
                let v_ikp1 = v.get_2d(i + 1, k + 2)?;
                v.set_2d(i + 1, k + 1, c * v_ik + s * v_ikp1)?;
                v.set_2d(i + 1, k + 2, -s * v_ik + c * v_ikp1)?;
            }

            // Givens rotation on the left (rows k, k+1)
            y = d[k];
            z = temp;
            let (c_l, s_l, r_l) = givens(y, z);
            d[k] = r_l;

            let e_k_new = c_l * e[k] + s_l * d[k + 1];
            d[k + 1] = -s_l * e[k] + c_l * d[k + 1];
            e[k] = e_k_new;

            if k + 2 < p {
                z = s_l * e[k + 1];
                e[k + 1] = c_l * e[k + 1];
                y = e[k];
            }

            // Accumulate into U
            for i in 0..m {
                let u_ik = u.get_2d(i + 1, k + 1)?;
                let u_ikp1 = u.get_2d(i + 1, k + 2)?;
                u.set_2d(i + 1, k + 1, c_l * u_ik + s_l * u_ikp1)?;
                u.set_2d(i + 1, k + 2, -s_l * u_ik + c_l * u_ikp1)?;
            }
        }
    }

    // Phase 3: Make singular values non-negative and sort descending
    for i in 0..n {
        if d[i] < 0.0 {
            d[i] = -d[i];
            for row in 1..=m {
                let val = u.get_2d(row, i + 1)?;
                u.set_2d(row, i + 1, -val)?;
            }
        }
    }

    // Sort singular values descending with permutation of columns of U and V
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| d[b].partial_cmp(&d[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut sorted_d = vec![0.0; n];
    let mut sorted_u = MxArray::zeros(m, if economy { n } else { m });
    let mut sorted_v = MxArray::zeros(n, if economy { n } else { n });

    for (new_idx, &old_idx) in indices.iter().enumerate() {
        sorted_d[new_idx] = d[old_idx];
        for row in 1..=m {
            sorted_u.set_2d(row, new_idx + 1, u.get_2d(row, old_idx + 1)?)?;
        }
        for row in 1..=n {
            sorted_v.set_2d(row, new_idx + 1, v.get_2d(row, old_idx + 1)?)?;
        }
    }

    // If full U requested (and not economy), copy remaining columns
    if !economy && m > n {
        for col in (n + 1)..=m {
            for row in 1..=m {
                sorted_u.set_2d(row, col, u.get_2d(row, col)?)?;
            }
        }
    }

    // Build diagonal matrix S
    let s_rows = if economy { n } else { m };
    let s_cols = n;
    let mut s_mat = MxArray::zeros(s_rows, s_cols);
    for i in 0..n {
        s_mat.set_2d(i + 1, i + 1, sorted_d[i])?;
    }

    Ok(SvdDecomposition {
        u: sorted_u,
        s: s_mat,
        v: sorted_v,
        singular_values: sorted_d,
    })
}

/// Helper function to compute stable Givens rotation parameters (c, s, r)
/// such that [c, s; -s, c] * [a; b] = [r; 0]
fn givens(a: f64, b: f64) -> (f64, f64, f64) {
    if b == 0.0 {
        let c = if a >= 0.0 { 1.0 } else { -1.0 };
        (c, 0.0, a.abs())
    } else if a == 0.0 {
        let s = if b >= 0.0 { 1.0 } else { -1.0 };
        (0.0, s, b.abs())
    } else if a.abs() > b.abs() {
        let t = b / a;
        let u = (1.0 + t * t).sqrt() * if a >= 0.0 { 1.0 } else { -1.0 };
        let c = 1.0 / u;
        let s = t * c;
        let r = a * u;
        (c, s, r)
    } else {
        let t = a / b;
        let u = (1.0 + t * t).sqrt() * if b >= 0.0 { 1.0 } else { -1.0 };
        let s = 1.0 / u;
        let c = t * s;
        let r = b * u;
        (c, s, r)
    }
}
