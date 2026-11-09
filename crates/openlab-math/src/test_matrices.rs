use openlab_core::{CoreError, MxArray};

/// Generates an N-by-N Hilbert matrix: H(i, j) = 1 / (i + j - 1).
///
/// Famously ill-conditioned matrix used to benchmark numerical solvers.
pub fn hilb(n: usize) -> Result<MxArray, CoreError> {
    let mut mat = MxArray::zeros(n, n);
    for i in 1..=n {
        for j in 1..=n {
            let val = 1.0 / ((i + j - 1) as f64);
            mat.set_2d(i, j, val)?;
        }
    }
    Ok(mat)
}

/// Generates an N-by-N exact inverse Hilbert matrix.
///
/// H^(-1)(i, j) = (-1)^(i+j) * (i+j-1) * nCr(n+i-1, n-j) * nCr(n+j-1, n-i) * (nCr(i+j-2, i-1))^2
pub fn invhilb(n: usize) -> Result<MxArray, CoreError> {
    let mut mat = MxArray::zeros(n, n);
    for i in 1..=n {
        for j in 1..=n {
            let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            let term1 = (i + j - 1) as f64;
            let term2 = n_cr(n + i - 1, n - j);
            let term3 = n_cr(n + j - 1, n - i);
            let term4 = n_cr(i + j - 2, i - 1);
            let val = sign * term1 * term2 * term3 * term4 * term4;
            mat.set_2d(i, j, val)?;
        }
    }
    Ok(mat)
}

/// Generates an N-by-N symmetric Pascal matrix: P(i, j) = nCr(i+j-2, j-1).
pub fn pascal(n: usize) -> Result<MxArray, CoreError> {
    let mut mat = MxArray::zeros(n, n);
    for i in 1..=n {
        for j in 1..=n {
            let val = n_cr(i + j - 2, j - 1);
            mat.set_2d(i, j, val)?;
        }
    }
    Ok(mat)
}

/// Generates an N-by-N Moler test matrix:
/// M(i, j) = min(i, j) - 2 for i != j, and M(i, i) = i.
/// Symmetric positive definite with one very small eigenvalue.
pub fn moler(n: usize) -> Result<MxArray, CoreError> {
    let mut mat = MxArray::zeros(n, n);
    for i in 1..=n {
        for j in 1..=n {
            let val = if i == j {
                i as f64
            } else {
                (std::cmp::min(i, j) as f64) - 2.0
            };
            mat.set_2d(i, j, val)?;
        }
    }
    Ok(mat)
}

/// Generates an N-by-N Magic Square: all rows, columns and main diagonals sum to n*(n^2+1)/2.
pub fn magic(n: usize) -> Result<MxArray, CoreError> {
    if n == 0 {
        return Ok(MxArray::zeros(0, 0));
    }
    if n == 2 {
        return Err(CoreError::InvalidOperation(
            "Magic square of order 2 does not exist.".into(),
        ));
    }

    if n % 2 == 1 {
        // Odd order: Siamese method
        magic_odd(n)
    } else if n % 4 == 0 {
        // Doubly-even order
        magic_doubly_even(n)
    } else {
        // Singly-even order (n % 4 == 2): Strachey method
        magic_singly_even(n)
    }
}

fn magic_odd(n: usize) -> Result<MxArray, CoreError> {
    let mut mat = MxArray::zeros(n, n);
    let mut r = 0;
    let mut c = n / 2;

    for num in 1..=(n * n) {
        mat.set_2d(r + 1, c + 1, num as f64)?;
        let next_r = (r + n - 1) % n;
        let next_c = (c + 1) % n;

        if mat.get_2d(next_r + 1, next_c + 1)? != 0.0 {
            r = (r + 1) % n;
        } else {
            r = next_r;
            c = next_c;
        }
    }

    Ok(mat)
}

fn magic_doubly_even(n: usize) -> Result<MxArray, CoreError> {
    let mut mat = MxArray::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let num = i * n + j + 1;
            let in_diag_block = ((i % 4 == 0 || i % 4 == 3) && (j % 4 == 0 || j % 4 == 3))
                || ((i % 4 == 1 || i % 4 == 2) && (j % 4 == 1 || j % 4 == 2));

            if in_diag_block {
                mat.set_2d(i + 1, j + 1, (n * n + 1 - num) as f64)?;
            } else {
                mat.set_2d(i + 1, j + 1, num as f64)?;
            }
        }
    }
    Ok(mat)
}

fn magic_singly_even(n: usize) -> Result<MxArray, CoreError> {
    let half_n = n / 2;
    let sub = magic_odd(half_n)?;
    let mut mat = MxArray::zeros(n, n);
    let shift = (half_n * half_n) as f64;

    // Subgrids A (top-left), B (bottom-right), C (top-right), D (bottom-left)
    for i in 1..=half_n {
        for j in 1..=half_n {
            let a_val = sub.get_2d(i, j)?;
            mat.set_2d(i, j, a_val)?; // A
            mat.set_2d(i + half_n, j + half_n, a_val + shift)?; // B
            mat.set_2d(i, j + half_n, a_val + 2.0 * shift)?; // C
            mat.set_2d(i + half_n, j, a_val + 3.0 * shift)?; // D
        }
    }

    let k = (n - 2) / 4;
    // Swap columns in left subgrid
    for i in 1..=half_n {
        for j in 1..=k {
            let row_idx = if i == (half_n / 2) + 1 { j + 1 } else { j };
            let top = mat.get_2d(i, row_idx)?;
            let bot = mat.get_2d(i + half_n, row_idx)?;
            mat.set_2d(i, row_idx, bot)?;
            mat.set_2d(i + half_n, row_idx, top)?;
        }
    }

    // Swap columns in right subgrid if k > 1
    if k > 1 {
        for i in 1..=half_n {
            for j in (n - k + 2)..=n {
                let top = mat.get_2d(i, j)?;
                let bot = mat.get_2d(i + half_n, j)?;
                mat.set_2d(i, j, bot)?;
                mat.set_2d(i + half_n, j, top)?;
            }
        }
    }

    Ok(mat)
}

fn n_cr(n: usize, r: usize) -> f64 {
    if r > n {
        return 0.0;
    }
    if r == 0 || r == n {
        return 1.0;
    }
    let k = std::cmp::min(r, n - r);
    let mut res = 1.0;
    for i in 1..=k {
        res = res * ((n - k + i) as f64) / (i as f64);
    }
    res.round()
}
