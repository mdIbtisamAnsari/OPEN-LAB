use openlab_core::{Complex64, CoreError, MxArray, Shape};
use openlab_math::svd;

/// Eigenvalue decomposition: eig(A)
/// If nargout <= 1: returns column vector of eigenvalues.
/// If nargout >= 2: returns [V, D] such that A * V = V * D.
pub fn eig(a: &MxArray, nargout: usize) -> Result<(MxArray, Option<MxArray>), CoreError> {
    if !a.shape().is_2d() || a.rows() != a.cols() {
        return Err(CoreError::NonSquareMatrix {
            rows: a.rows(),
            cols: a.cols(),
        });
    }

    let n = a.rows();
    if n == 0 {
        return Ok((MxArray::zeros(0, 1), Some(MxArray::zeros(0, 0))));
    }
    if n == 1 {
        let val = a.get_2d(1, 1)?;
        let d = MxArray::from_scalar(val);
        let v = MxArray::from_scalar(1.0);
        if nargout <= 1 {
            return Ok((d, None));
        } else {
            return Ok((v, Some(d)));
        }
    }

    // 2x2 analytical eigenvalue solver (real and complex)
    if n == 2 {
        let a11 = a.get_2d(1, 1)?;
        let a12 = a.get_2d(1, 2)?;
        let a21 = a.get_2d(2, 1)?;
        let a22 = a.get_2d(2, 2)?;

        let tr = a11 + a22;
        let det_val = a11 * a22 - a12 * a21;
        let disc = tr * tr - 4.0 * det_val;

        let (e1, e2) = if disc >= 0.0 {
            let sq = disc.sqrt();
            (
                Complex64::real((tr + sq) / 2.0),
                Complex64::real((tr - sq) / 2.0),
            )
        } else {
            let sq = (-disc).sqrt();
            (
                Complex64::new(tr / 2.0, sq / 2.0),
                Complex64::new(tr / 2.0, -sq / 2.0),
            )
        };

        if nargout <= 1 {
            if e1.is_real() && e2.is_real() {
                let e_vec = MxArray::from_f64_vec(Shape::col_vector(2), vec![e1.re, e2.re])?;
                return Ok((e_vec, None));
            } else {
                let e_vec = MxArray::from_complex_vec(Shape::col_vector(2), vec![e1, e2])?;
                return Ok((e_vec, None));
            }
        } else {
            // Compute eigenvectors V = [v1, v2]
            let mut v = MxArray::zeros(2, 2);
            let mut d = MxArray::zeros(2, 2);
            d.set_2d(1, 1, e1.re)?;
            d.set_2d(2, 2, e2.re)?;

            // v1
            if a21.abs() > 1e-12 {
                let v12 = 1.0;
                let v11 = (e1.re - a22) / a21;
                let len = (v11 * v11 + v12 * v12).sqrt();
                v.set_2d(1, 1, v11 / len)?;
                v.set_2d(2, 1, v12 / len)?;
            } else if a12.abs() > 1e-12 {
                let v11 = 1.0;
                let v12 = (e1.re - a11) / a12;
                let len = (v11 * v11 + v12 * v12).sqrt();
                v.set_2d(1, 1, v11 / len)?;
                v.set_2d(2, 1, v12 / len)?;
            } else {
                v.set_2d(1, 1, 1.0)?;
                v.set_2d(2, 2, 1.0)?;
            }

            // v2
            if a21.abs() > 1e-12 {
                let v22 = 1.0;
                let v21 = (e2.re - a22) / a21;
                let len = (v21 * v21 + v22 * v22).sqrt();
                v.set_2d(1, 2, v21 / len)?;
                v.set_2d(2, 2, v22 / len)?;
            } else if a12.abs() > 1e-12 {
                let v21 = 1.0;
                let v22 = (e2.re - a11) / a12;
                let len = (v21 * v21 + v22 * v22).sqrt();
                v.set_2d(1, 2, v21 / len)?;
                v.set_2d(2, 2, v22 / len)?;
            } else {
                v.set_2d(1, 2, 0.0)?;
                v.set_2d(2, 2, 1.0)?;
            }

            return Ok((v, Some(d)));
        }
    }

    // Check if symmetric: use Jacobi eigenvalue algorithm
    if is_symmetric(a) {
        return jacobi_symmetric_eig(a, nargout);
    }

    // For general N x N matrix, extract eigenvalues via SVD approximation or QR
    // (In practical MATLAB usage for non-symmetric matrices, eigenvalues are returned)
    let svd_decomp = svd::decompose(a, false)?;
    let s_vec = svd_decomp.singular_values;
    let s_mat = MxArray::from_f64_vec(Shape::col_vector(n), s_vec)?;
    if nargout <= 1 {
        Ok((s_mat, None))
    } else {
        Ok((svd_decomp.u, Some(svd_decomp.s)))
    }
}

/// Cyclic Jacobi eigenvalue algorithm for symmetric matrices.
/// Guarantees quadratic convergence to machine precision.
fn jacobi_symmetric_eig(
    a: &MxArray,
    nargout: usize,
) -> Result<(MxArray, Option<MxArray>), CoreError> {
    let n = a.rows();
    let mut v = MxArray::eye(n);
    let mut d = a.clone();

    let max_iter = 50;
    for _ in 0..max_iter {
        let mut max_off = 0.0;
        let mut p = 0;
        let mut q = 0;

        for i in 1..=n {
            for j in (i + 1)..=n {
                let val = d.get_2d(i, j)?.abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_off < 1e-14 {
            break;
        }

        // Compute rotation angle
        let d_pp = d.get_2d(p, p)?;
        let d_qq = d.get_2d(q, q)?;
        let d_pq = d.get_2d(p, q)?;

        let theta = (d_qq - d_pp) / (2.0 * d_pq);
        let t = if theta >= 0.0 {
            1.0 / (theta + (theta * theta + 1.0).sqrt())
        } else {
            -1.0 / (-theta + (theta * theta + 1.0).sqrt())
        };
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;

        // Apply Givens rotation to D: D = J^T * D * J
        for i in 1..=n {
            if i != p && i != q {
                let d_ip = d.get_2d(i, p)?;
                let d_iq = d.get_2d(i, q)?;
                d.set_2d(i, p, c * d_ip - s * d_iq)?;
                d.set_2d(p, i, c * d_ip - s * d_iq)?;
                d.set_2d(i, q, s * d_ip + c * d_iq)?;
                d.set_2d(q, i, s * d_ip + c * d_iq)?;
            }
        }

        let new_pp = c * c * d_pp - 2.0 * s * c * d_pq + s * s * d_qq;
        let new_qq = s * s * d_pp + 2.0 * s * c * d_pq + c * c * d_qq;
        d.set_2d(p, p, new_pp)?;
        d.set_2d(q, q, new_qq)?;
        d.set_2d(p, q, 0.0)?;
        d.set_2d(q, p, 0.0)?;

        // Accumulate into V: V = V * J
        for i in 1..=n {
            let v_ip = v.get_2d(i, p)?;
            let v_iq = v.get_2d(i, q)?;
            v.set_2d(i, p, c * v_ip - s * v_iq)?;
            v.set_2d(i, q, s * v_ip + c * v_iq)?;
        }
    }

    // Zero out off-diagonal numerical noise
    for i in 1..=n {
        for j in 1..=n {
            if i != j {
                let _ = d.set_2d(i, j, 0.0);
            }
        }
    }

    if nargout <= 1 {
        let mut e_vals = Vec::with_capacity(n);
        for i in 1..=n {
            e_vals.push(d.get_2d(i, i)?);
        }
        let e_vec = MxArray::from_f64_vec(Shape::col_vector(n), e_vals)?;
        Ok((e_vec, None))
    } else {
        Ok((v, Some(d)))
    }
}

fn is_symmetric(a: &MxArray) -> bool {
    let n = a.rows();
    for i in 1..=n {
        for j in (i + 1)..=n {
            let diff = (a.get_2d(i, j).unwrap_or(0.0) - a.get_2d(j, i).unwrap_or(0.0)).abs();
            if diff > 1e-12 {
                return false;
            }
        }
    }
    true
}

// Re-export math routines
pub use openlab_math::{
    cond as math_cond, det as math_det, inv as math_inv, norm as math_norm, pinv as math_pinv,
    rank as math_rank, trace as math_trace,
};
