use openlab_core::{Complex64, MxArray};
use openlab_math::{
    cholesky, cond, default_backend, det, gemm::matpow, hilb, inv, invhilb, lu, magic,
    matmul, mldivide, moler, mrdivide, norm, pascal, qr, svd, trace, NormType,
};

fn matrix_diff_norm(a: &MxArray, b: &MxArray) -> f64 {
    let diff = a.sub(b).expect("subtraction should succeed");
    norm(&diff, NormType::Inf).expect("norm should succeed")
}

// =========================================================================
// 1. GEMM (Matrix Multiplication) Tests
// =========================================================================

#[test]
fn test_gemm_real_standard() {
    // 2x3 * 3x2 -> 2x2
    // A = [1, 2, 3; 4, 5, 6]
    // B = [7, 8; 9, 1; 2, 3]
    // C = [1*7 + 2*9 + 3*2, 1*8 + 2*1 + 3*3; 4*7 + 5*9 + 6*2, 4*8 + 5*1 + 6*3]
    //   = [7 + 18 + 6, 8 + 2 + 9; 28 + 45 + 12, 32 + 5 + 18]
    //   = [31, 19; 85, 55]
    let a = MxArray::from_2d_slice(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
    let b = MxArray::from_2d_slice(&[&[7.0, 8.0], &[9.0, 1.0], &[2.0, 3.0]]);
    let c = matmul(&a, &b).unwrap();

    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 2);
    assert_eq!(c.get_2d(1, 1).unwrap(), 31.0);
    assert_eq!(c.get_2d(1, 2).unwrap(), 19.0);
    assert_eq!(c.get_2d(2, 1).unwrap(), 85.0);
    assert_eq!(c.get_2d(2, 2).unwrap(), 55.0);
}

#[test]
fn test_gemm_complex() {
    // A = [1+2i, 3; 4, 5-1i]
    // B = [2, -i; 1+i, 2]
    // A(1, 1)*B(1, 1) + A(1, 2)*B(2, 1) = (1+2i)*2 + 3*(1+i) = (2+4i) + (3+3i) = 5+7i
    let mut a = MxArray::zeros(2, 2);
    a.set_complex_2d(1, 1, Complex64::new(1.0, 2.0)).unwrap();
    a.set_complex_2d(1, 2, Complex64::new(3.0, 0.0)).unwrap();
    a.set_complex_2d(2, 1, Complex64::new(4.0, 0.0)).unwrap();
    a.set_complex_2d(2, 2, Complex64::new(5.0, -1.0)).unwrap();

    let mut b = MxArray::zeros(2, 2);
    b.set_complex_2d(1, 1, Complex64::new(2.0, 0.0)).unwrap();
    b.set_complex_2d(1, 2, Complex64::new(0.0, -1.0)).unwrap();
    b.set_complex_2d(2, 1, Complex64::new(1.0, 1.0)).unwrap();
    b.set_complex_2d(2, 2, Complex64::new(2.0, 0.0)).unwrap();

    let c = matmul(&a, &b).unwrap();
    let c11 = c.get_complex_2d(1, 1).unwrap();
    assert_eq!(c11, Complex64::new(5.0, 7.0));
}

#[test]
fn test_gemm_scalar_expansion() {
    let a = MxArray::from_2d_slice(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let s = MxArray::from_scalar(3.0);

    // s * A
    let res1 = matmul(&s, &a).unwrap();
    assert_eq!(res1.get_2d(1, 1).unwrap(), 3.0);
    assert_eq!(res1.get_2d(2, 2).unwrap(), 12.0);

    // A * s
    let res2 = matmul(&a, &s).unwrap();
    assert_eq!(res2.get_2d(1, 1).unwrap(), 3.0);
    assert_eq!(res2.get_2d(2, 2).unwrap(), 12.0);
}

#[test]
fn test_gemm_empty_dimensions() {
    // In MATLAB:
    // zeros(2, 0) * zeros(0, 3) -> zeros(2, 3)
    let a = MxArray::zeros(2, 0);
    let b = MxArray::zeros(0, 3);
    let c = matmul(&a, &b).unwrap();
    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 3);
    assert_eq!(c.numel(), 6);
    assert_eq!(c.get_2d(1, 1).unwrap(), 0.0);
    assert_eq!(c.get_2d(2, 3).unwrap(), 0.0);

    // zeros(0, 2) * zeros(2, 3) -> zeros(0, 3)
    let d = MxArray::zeros(0, 2);
    let e = MxArray::zeros(2, 3);
    let f = matmul(&d, &e).unwrap();
    assert_eq!(f.rows(), 0);
    assert_eq!(f.cols(), 3);
    assert_eq!(f.numel(), 0);
}

#[test]
fn test_matpow() {
    let a = MxArray::from_2d_slice(&[&[1.0, 1.0], &[1.0, 0.0]]);
    // Fibonacci transition matrix: A^0 = I
    let a0 = matpow(&a, 0).unwrap();
    assert_eq!(a0, MxArray::eye(2));

    // A^1 = A
    let a1 = matpow(&a, 1).unwrap();
    assert_eq!(a1, a);

    // A^5 = [8, 5; 5, 3]
    let a5 = matpow(&a, 5).unwrap();
    assert_eq!(a5.get_2d(1, 1).unwrap(), 8.0);
    assert_eq!(a5.get_2d(1, 2).unwrap(), 5.0);
    assert_eq!(a5.get_2d(2, 1).unwrap(), 5.0);
    assert_eq!(a5.get_2d(2, 2).unwrap(), 3.0);
}

// =========================================================================
// 2. LU Decomposition Tests
// =========================================================================

#[test]
fn test_lu_decomposition_invariant() {
    // 3x3 matrix:
    let a = MxArray::from_2d_slice(&[
        &[2.0, 1.0, 1.0],
        &[4.0, -6.0, 0.0],
        &[-2.0, 7.0, 2.0],
    ]);

    let lu_decomp = lu::decompose(&a).unwrap();

    // Verify P * A = L * U
    let pa = matmul(&lu_decomp.p, &a).unwrap();
    let lu_prod = matmul(&lu_decomp.l, &lu_decomp.u).unwrap();
    let error = matrix_diff_norm(&pa, &lu_prod);
    assert!(error < 1e-14, "P*A must equal L*U, got error {}", error);

    // Verify L is unit lower triangular
    for i in 1..=3 {
        assert_eq!(lu_decomp.l.get_2d(i, i).unwrap(), 1.0);
        for j in (i + 1)..=3 {
            assert_eq!(lu_decomp.l.get_2d(i, j).unwrap(), 0.0);
        }
    }

    // Verify U is upper triangular
    for i in 1..=3 {
        for j in 1..i {
            assert_eq!(lu_decomp.u.get_2d(i, j).unwrap(), 0.0);
        }
    }

    // Verify determinant
    let det_val = lu_decomp.det().unwrap();
    // det(A) = 2*(-12 - 0) - 1*(8 - 0) + 1*(28 - 12) = -24 - 8 + 16 = -16
    assert!((det_val - (-16.0)).abs() < 1e-12);
}

#[test]
fn test_lu_solve() {
    let a = MxArray::from_2d_slice(&[
        &[1.0, 2.0, 3.0],
        &[2.0, 5.0, 3.0],
        &[1.0, 0.0, 8.0],
    ]);
    let b = MxArray::from_2d_slice(&[&[5.0], &[3.0], &[17.0]]);

    let lu_decomp = lu::decompose(&a).unwrap();
    let x = lu_decomp.solve(&b).unwrap();

    // Check A * x = b
    let ax = matmul(&a, &x).unwrap();
    let err = matrix_diff_norm(&ax, &b);
    assert!(err < 1e-13, "A*x must equal b, error: {}", err);
}

// =========================================================================
// 3. Cholesky Decomposition Tests
// =========================================================================

#[test]
fn test_cholesky_spd() {
    // Symmetric positive-definite matrix
    // A = [4, 12, -16; 12, 37, -43; -16, -43, 98]
    let a = MxArray::from_2d_slice(&[
        &[4.0, 12.0, -16.0],
        &[12.0, 37.0, -43.0],
        &[-16.0, -43.0, 98.0],
    ]);

    let chol = cholesky::decompose(&a).unwrap();

    // Verify A = L * L^T
    let lt = chol.l.transpose().unwrap();
    let l_lt = matmul(&chol.l, &lt).unwrap();
    let err = matrix_diff_norm(&a, &l_lt);
    assert!(err < 1e-13, "A must equal L*L^T, error: {}", err);

    // Verify A = R^T * R
    let rt = chol.r.transpose().unwrap();
    let rt_r = matmul(&rt, &chol.r).unwrap();
    let err_r = matrix_diff_norm(&a, &rt_r);
    assert!(err_r < 1e-13, "A must equal R^T*R, error: {}", err_r);

    // Verify L diagonal entries are strictly positive
    for i in 1..=3 {
        assert!(chol.l.get_2d(i, i).unwrap() > 0.0);
    }
}

#[test]
fn test_cholesky_non_positive_definite_failure() {
    // Indefinite matrix [1, 2; 2, 1] has eigenvalues 3 and -1 -> not positive definite
    let a = MxArray::from_2d_slice(&[&[1.0, 2.0], &[2.0, 1.0]]);
    let res = cholesky::decompose(&a);
    assert!(res.is_err(), "Must reject non-positive-definite matrix");
}

#[test]
fn test_cholesky_non_symmetric_failure() {
    let a = MxArray::from_2d_slice(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let res = cholesky::decompose(&a);
    assert!(res.is_err(), "Must reject non-symmetric matrix");
}

// =========================================================================
// 4. QR Decomposition Tests
// =========================================================================

#[test]
fn test_qr_square_and_orthogonal() {
    let a = MxArray::from_2d_slice(&[
        &[12.0, -51.0, 4.0],
        &[6.0, 167.0, -68.0],
        &[-4.0, 24.0, -41.0],
    ]);

    let qr_decomp = qr::decompose(&a, false).unwrap();

    // Verify A = Q * R
    let qr_prod = matmul(&qr_decomp.q, &qr_decomp.r).unwrap();
    let err = matrix_diff_norm(&a, &qr_prod);
    assert!(err < 1e-13, "A must equal Q*R, error: {}", err);

    // Verify Q is orthogonal: Q^T * Q = I
    let qt = qr_decomp.q.transpose().unwrap();
    let qt_q = matmul(&qt, &qr_decomp.q).unwrap();
    let eye = MxArray::eye(3);
    let err_orth = matrix_diff_norm(&qt_q, &eye);
    assert!(err_orth < 1e-13, "Q must be orthogonal, error: {}", err_orth);
}

#[test]
fn test_qr_overdetermined_least_squares() {
    // 4x2 matrix (overdetermined system A*x ~= b)
    let a = MxArray::from_2d_slice(&[
        &[1.0, 1.0],
        &[1.0, 2.0],
        &[1.0, 3.0],
        &[1.0, 4.0],
    ]);
    let b = MxArray::from_2d_slice(&[&[6.0], &[5.0], &[7.0], &[10.0]]);

    let qr_decomp = qr::decompose(&a, true).unwrap();
    let x = qr_decomp.solve_least_squares(&b).unwrap();

    assert_eq!(x.rows(), 2);
    assert_eq!(x.cols(), 1);

    // Normal equations condition: A^T * (A*x - b) must be zero!
    let ax = matmul(&a, &x).unwrap();
    let residual = ax.sub(&b).unwrap();
    let at = a.transpose().unwrap();
    let normal_eq = matmul(&at, &residual).unwrap();
    let err = norm(&normal_eq, NormType::Inf).unwrap();
    assert!(err < 1e-12, "Least squares normal equations must hold: {}", err);
}

// =========================================================================
// 5. SVD (Singular Value Decomposition) Tests
// =========================================================================

#[test]
fn test_svd_reconstruction() {
    let a = MxArray::from_2d_slice(&[
        &[1.0, 2.0],
        &[3.0, 4.0],
        &[5.0, 6.0],
    ]);

    let svd_decomp = svd::decompose(&a, false).unwrap();

    // Verify A = U * S * V^T
    let us = matmul(&svd_decomp.u, &svd_decomp.s).unwrap();
    let vt = svd_decomp.v.transpose().unwrap();
    let usv = matmul(&us, &vt).unwrap();
    let err = matrix_diff_norm(&a, &usv);
    assert!(err < 1e-13, "A must equal U*S*V^T, error: {}", err);

    // Verify U is orthogonal: U^T * U = I (3x3)
    let ut = svd_decomp.u.transpose().unwrap();
    let ut_u = matmul(&ut, &svd_decomp.u).unwrap();
    let err_u = matrix_diff_norm(&ut_u, &MxArray::eye(3));
    assert!(err_u < 1e-13, "U must be orthogonal: {}", err_u);

    // Verify V is orthogonal: V^T * V = I (2x2)
    let vt_v = matmul(&vt, &svd_decomp.v).unwrap();
    let err_v = matrix_diff_norm(&vt_v, &MxArray::eye(2));
    assert!(err_v < 1e-13, "V must be orthogonal: {}", err_v);

    // Verify singular values are descending
    let s = &svd_decomp.singular_values;
    assert!(s[0] >= s[1]);
    assert!(s[1] >= 0.0);
}

#[test]
fn test_svd_rank_deficient() {
    // Rank 1 matrix: row 2 = 2 * row 1
    let a = MxArray::from_2d_slice(&[
        &[1.0, 2.0],
        &[2.0, 4.0],
    ]);

    let svd_decomp = svd::decompose(&a, false).unwrap();
    assert_eq!(svd_decomp.rank(None), 1, "Rank of degenerate matrix must be 1");
}

// =========================================================================
// 6. Solvers (mldivide, mrdivide, inv, pinv)
// =========================================================================

#[test]
fn test_mldivide_and_mrdivide() {
    let a = MxArray::from_2d_slice(&[&[3.0, 1.0], &[1.0, 2.0]]);
    let b = MxArray::from_2d_slice(&[&[9.0], &[8.0]]);

    // Solve A * x = b -> x = A \ b
    let x = mldivide(&a, &b).unwrap();
    assert!((x.get_2d(1, 1).unwrap() - 2.0).abs() < 1e-12);
    assert!((x.get_2d(2, 1).unwrap() - 3.0).abs() < 1e-12);

    // Test mrdivide: y * A = b_row -> y = b_row / A
    let b_row = b.transpose().unwrap();
    let y = mrdivide(&b_row, &a).unwrap();
    let ya = matmul(&y, &a).unwrap();
    let err = matrix_diff_norm(&ya, &b_row);
    assert!(err < 1e-12, "y*A must equal b_row: {}", err);
}

#[test]
fn test_inv_and_identity() {
    let a = MxArray::from_2d_slice(&[
        &[4.0, 7.0],
        &[2.0, 6.0],
    ]);

    let a_inv = inv(&a).unwrap();

    // A * A^(-1) = I
    let prod1 = matmul(&a, &a_inv).unwrap();
    let err1 = matrix_diff_norm(&prod1, &MxArray::eye(2));
    assert!(err1 < 1e-13, "A * inv(A) must equal I: {}", err1);

    // A^(-1) * A = I
    let prod2 = matmul(&a_inv, &a).unwrap();
    let err2 = matrix_diff_norm(&prod2, &MxArray::eye(2));
    assert!(err2 < 1e-13, "inv(A) * A must equal I: {}", err2);
}

// =========================================================================
// 7. Canonical MATLAB Test Matrices (Moler, Hilbert, Pascal, Magic)
// =========================================================================

#[test]
fn test_hilbert_and_invhilb_parity() {
    let n = 4;
    let h = hilb(n).unwrap();
    let h_inv = invhilb(n).unwrap();

    // Exact inverse multiplication: H * H^(-1) = I
    let prod = matmul(&h, &h_inv).unwrap();
    let err = matrix_diff_norm(&prod, &MxArray::eye(n));
    assert!(err < 1e-11, "H * invhilb(H) must equal I within precision: {}", err);

    // Check condition number of Hilbert(4) is ~ 15514
    let c = cond(&h).unwrap();
    assert!(c > 15000.0 && c < 16000.0, "Condition number of hilb(4) is ~15514, got {}", c);
}

#[test]
fn test_pascal_matrix_determinant() {
    // The determinant of ANY Pascal matrix is exactly 1!
    for n in 1..=5 {
        let p = pascal(n).unwrap();
        let d = det(&p).unwrap();
        assert!((d - 1.0).abs() < 1e-10, "det(pascal({})) must be 1, got {}", n, d);
    }
}

#[test]
fn test_magic_square_properties() {
    // Test magic squares of odd, doubly-even, and singly-even orders
    for &n in &[3, 4, 5, 6] {
        let m = magic(n).unwrap();
        let magic_sum = (n * (n * n + 1) / 2) as f64;

        // Check each row sum
        for i in 1..=n {
            let mut r_sum = 0.0;
            for j in 1..=n {
                r_sum += m.get_2d(i, j).unwrap();
            }
            assert_eq!(r_sum, magic_sum, "Row {} sum in magic({})", i, n);
        }

        // Check each col sum
        for j in 1..=n {
            let mut c_sum = 0.0;
            for i in 1..=n {
                c_sum += m.get_2d(i, j).unwrap();
            }
            assert_eq!(c_sum, magic_sum, "Col {} sum in magic({})", j, n);
        }

        // Check main diagonal
        let tr = trace(&m).unwrap();
        assert_eq!(tr, magic_sum, "Trace of magic({}) must equal magic sum", n);
    }
}

#[test]
fn test_moler_matrix_symmetry_and_positive_definiteness() {
    let m = moler(5).unwrap();
    // Must be symmetric
    let mt = m.transpose().unwrap();
    assert_eq!(m, mt);

    // Cholesky should succeed because Moler is positive definite
    let chol = cholesky::decompose(&m);
    assert!(chol.is_ok(), "Moler matrix must be positive definite");
}

#[test]
fn test_matrix_norms() {
    let a = MxArray::from_2d_slice(&[
        &[1.0, -2.0, 3.0],
        &[-4.0, 5.0, -6.0],
    ]);

    // 1-norm: max column sum = max(1+4, 2+5, 3+6) = 9
    assert_eq!(norm(&a, NormType::One).unwrap(), 9.0);

    // Inf-norm: max row sum = max(1+2+3, 4+5+6) = 15
    assert_eq!(norm(&a, NormType::Inf).unwrap(), 15.0);

    // Frobenius norm: sqrt(1 + 4 + 9 + 16 + 25 + 36) = sqrt(91) ~= 9.53939
    let fro = norm(&a, NormType::Frobenius).unwrap();
    assert!((fro - 91.0_f64.sqrt()).abs() < 1e-12);
}

#[test]
fn test_backend_interface() {
    let backend = default_backend();
    assert!(backend.name().contains("CPU Reference"));

    let a = MxArray::eye(3);
    let b = MxArray::from_scalar(2.0);
    let c = backend.gemm(&a, &b).unwrap();
    assert_eq!(c.get_2d(1, 1).unwrap(), 2.0);
}
