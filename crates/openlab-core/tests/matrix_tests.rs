use openlab_core::{
    Complex64, CoreError, MatrixFormatter, MxArray, Shape,
};

#[test]
fn test_shape_semantics() {
    // MATLAB guarantees at least 2 dimensions
    let s_scalar = Shape::scalar();
    assert_eq!(s_scalar.dims(), &[1, 1]);
    assert_eq!(s_scalar.numel(), 1);
    assert!(s_scalar.is_scalar());
    assert!(!s_scalar.is_vector());

    let s_row = Shape::row_vector(5);
    assert_eq!(s_row.dims(), &[1, 5]);
    assert_eq!(s_row.numel(), 5);
    assert!(s_row.is_vector());

    let s_col = Shape::col_vector(4);
    assert_eq!(s_col.dims(), &[4, 1]);
    assert_eq!(s_col.numel(), 4);
    assert!(s_col.is_vector());

    // Trailing singleton dimension stripping for ndims > 2
    let s_3d_strip = Shape::new(&[2, 3, 1]);
    assert_eq!(s_3d_strip.dims(), &[2, 3]);

    let s_3d = Shape::new(&[2, 3, 4]);
    assert_eq!(s_3d.dims(), &[2, 3, 4]);
    assert_eq!(s_3d.numel(), 24);
    assert_eq!(s_3d.strides(), vec![1, 2, 6]);
}

#[test]
fn test_column_major_memory_layout() {
    // Create a 2x3 matrix:
    // [ 1.0, 2.0, 3.0 ]
    // [ 4.0, 5.0, 6.0 ]
    let mat = MxArray::from_2d_slice(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

    assert_eq!(mat.rows(), 2);
    assert_eq!(mat.cols(), 3);
    assert_eq!(mat.numel(), 6);

    // In Fortran column-major storage, elements must be laid out as:
    // col 0: 1.0, 4.0
    // col 1: 2.0, 5.0
    // col 2: 3.0, 6.0
    let slice = mat.as_slice_f64().expect("should be double");
    assert_eq!(slice, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    // Test 1-based indexing
    assert_eq!(mat.get_2d(1, 1).unwrap(), 1.0);
    assert_eq!(mat.get_2d(2, 1).unwrap(), 4.0);
    assert_eq!(mat.get_2d(1, 2).unwrap(), 2.0);
    assert_eq!(mat.get_2d(2, 2).unwrap(), 5.0);
    assert_eq!(mat.get_2d(1, 3).unwrap(), 3.0);
    assert_eq!(mat.get_2d(2, 3).unwrap(), 6.0);

    // Test linear 1-based indexing (MATLAB A(4) is column 1, row 2 -> 5.0)
    assert_eq!(mat.get_linear_1based(1).unwrap(), 1.0);
    assert_eq!(mat.get_linear_1based(2).unwrap(), 4.0);
    assert_eq!(mat.get_linear_1based(3).unwrap(), 2.0);
    assert_eq!(mat.get_linear_1based(4).unwrap(), 5.0);
    assert_eq!(mat.get_linear_1based(5).unwrap(), 3.0);
    assert_eq!(mat.get_linear_1based(6).unwrap(), 6.0);
}

#[test]
fn test_index_out_of_bounds() {
    let mat = MxArray::zeros(2, 2);
    // Index 0 in 1-based indexing is illegal
    assert!(mat.get_2d(0, 1).is_err());
    assert!(mat.get_2d(1, 0).is_err());
    // Exceeding bounds
    let err = mat.get_2d(3, 1).unwrap_err();
    match err {
        CoreError::IndexOutOfBounds { dim, index, bound } => {
            assert_eq!(dim, 1);
            assert_eq!(index, 3);
            assert_eq!(bound, 2);
        }
        _ => panic!("Expected IndexOutOfBounds error"),
    }
}

#[test]
fn test_copy_on_write_semantics() {
    let mut a = MxArray::from_2d_slice(&[&[10.0, 20.0], &[30.0, 40.0]]);
    let b = a.clone();

    // Both a and b share the exact same underlying buffer pointer initially
    let ptr_a = a.as_slice_f64().unwrap().as_ptr();
    let ptr_b = b.as_slice_f64().unwrap().as_ptr();
    assert_eq!(ptr_a, ptr_b, "Cloned array must share memory buffer (zero copy)");

    // Modify a: should trigger Copy-on-Write
    a.set_2d(1, 1, 99.0).unwrap();

    // Verify a was updated
    assert_eq!(a.get_2d(1, 1).unwrap(), 99.0);
    // Verify b was NOT affected
    assert_eq!(b.get_2d(1, 1).unwrap(), 10.0);

    // Pointers must now be distinct
    let new_ptr_a = a.as_slice_f64().unwrap().as_ptr();
    assert_ne!(new_ptr_a, ptr_b, "Modified array must have isolated buffer");
}

#[test]
fn test_transpose() {
    let a = MxArray::from_2d_slice(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
    let at = a.transpose().unwrap();

    assert_eq!(at.rows(), 3);
    assert_eq!(at.cols(), 2);
    assert_eq!(at.get_2d(1, 1).unwrap(), 1.0);
    assert_eq!(at.get_2d(1, 2).unwrap(), 4.0);
    assert_eq!(at.get_2d(2, 1).unwrap(), 2.0);
    assert_eq!(at.get_2d(2, 2).unwrap(), 5.0);
    assert_eq!(at.get_2d(3, 1).unwrap(), 3.0);
    assert_eq!(at.get_2d(3, 2).unwrap(), 6.0);
}

#[test]
fn test_complex_numbers_and_hermitian_transpose() {
    let c1 = Complex64::new(1.0, 2.0);
    let c2 = Complex64::new(3.0, -4.0);
    let c_sum = c1 + c2;
    assert_eq!(c_sum.re, 4.0);
    assert_eq!(c_sum.im, -2.0);

    let mut mat = MxArray::zeros(2, 1);
    mat.set_complex_2d(1, 1, Complex64::new(1.0, 5.0)).unwrap();
    mat.set_complex_2d(2, 1, Complex64::new(2.0, -3.0)).unwrap();

    // Hermitian transpose A'
    let hermitian = mat.ctranspose().unwrap();
    assert_eq!(hermitian.rows(), 1);
    assert_eq!(hermitian.cols(), 2);
    assert_eq!(
        hermitian.get_complex_2d(1, 1).unwrap(),
        Complex64::new(1.0, -5.0)
    );
    assert_eq!(
        hermitian.get_complex_2d(1, 2).unwrap(),
        Complex64::new(2.0, 3.0)
    );
}

#[test]
fn test_submatrix() {
    // 3x3 matrix:
    // [ 1, 2, 3 ]
    // [ 4, 5, 6 ]
    // [ 7, 8, 9 ]
    let a = MxArray::from_2d_slice(&[
        &[1.0, 2.0, 3.0],
        &[4.0, 5.0, 6.0],
        &[7.0, 8.0, 9.0],
    ]);

    // Submatrix rows 2..=3, cols 1..=2:
    // [ 4, 5 ]
    // [ 7, 8 ]
    let sub = a.submatrix(2, 3, 1, 2).unwrap();
    assert_eq!(sub.rows(), 2);
    assert_eq!(sub.cols(), 2);
    assert_eq!(sub.get_2d(1, 1).unwrap(), 4.0);
    assert_eq!(sub.get_2d(1, 2).unwrap(), 5.0);
    assert_eq!(sub.get_2d(2, 1).unwrap(), 7.0);
    assert_eq!(sub.get_2d(2, 2).unwrap(), 8.0);
}

#[test]
fn test_broadcasting_arithmetic() {
    // 1. Matrix + Scalar
    let a = MxArray::from_2d_slice(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let s = MxArray::from_scalar(10.0);
    let res = a.add(&s).unwrap();
    assert_eq!(res.get_2d(1, 1).unwrap(), 11.0);
    assert_eq!(res.get_2d(1, 2).unwrap(), 12.0);
    assert_eq!(res.get_2d(2, 1).unwrap(), 13.0);
    assert_eq!(res.get_2d(2, 2).unwrap(), 14.0);

    // 2. Column vector (2x1) + Row vector (1x2) broadcasting to (2x2)
    // [10]        [1, 2]     [11, 12]
    // [20]    +           =  [21, 22]
    let col = MxArray::from_2d_slice(&[&[10.0], &[20.0]]);
    let row = MxArray::from_2d_slice(&[&[1.0, 2.0]]);
    let res2 = col.add(&row).unwrap();
    assert_eq!(res2.rows(), 2);
    assert_eq!(res2.cols(), 2);
    assert_eq!(res2.get_2d(1, 1).unwrap(), 11.0);
    assert_eq!(res2.get_2d(1, 2).unwrap(), 12.0);
    assert_eq!(res2.get_2d(2, 1).unwrap(), 21.0);
    assert_eq!(res2.get_2d(2, 2).unwrap(), 22.0);

    // 3. Element-wise multiplication (.*)
    let prod = col.elementwise_mul(&row).unwrap();
    assert_eq!(prod.get_2d(1, 1).unwrap(), 10.0);
    assert_eq!(prod.get_2d(1, 2).unwrap(), 20.0);
    assert_eq!(prod.get_2d(2, 1).unwrap(), 20.0);
    assert_eq!(prod.get_2d(2, 2).unwrap(), 40.0);
}

#[test]
fn test_reductions() {
    let a = MxArray::from_2d_slice(&[&[1.0, 2.0], &[3.0, 4.0]]);
    assert_eq!(a.sum_all().unwrap(), 10.0);
    assert_eq!(a.prod_all().unwrap(), 24.0);
    assert_eq!(a.mean_all().unwrap(), 2.5);
    assert_eq!(a.min_all().unwrap(), 1.0);
    assert_eq!(a.max_all().unwrap(), 4.0);
}

#[test]
fn test_matlab_indexing_subtleties() {
    let a = MxArray::from_2d_slice(&[&[1.0, 2.0], &[3.0, 4.0]]); // 2x2 matrix

    // 1. Single subscript linear indexing
    assert_eq!(a.get_linear_1based(1).unwrap(), 1.0);
    assert_eq!(a.get_linear_1based(2).unwrap(), 3.0);
    assert_eq!(a.get_linear_1based(3).unwrap(), 2.0);
    assert_eq!(a.get_linear_1based(4).unwrap(), 4.0);

    // 2. Trailing singleton dimension access: A(1, 2, 1) == A(1, 2)
    assert_eq!(a.get_nd(&[1, 2, 1]).unwrap(), 2.0);
    assert_eq!(a.get_nd(&[1, 2, 1, 1]).unwrap(), 2.0);

    // 3. Trailing non-singleton dimension access must fail: A(1, 2, 2) out of bounds
    assert!(a.get_nd(&[1, 2, 2]).is_err());
}

#[test]
fn test_matrix_display_formatting() {
    let eye = MxArray::eye(2);
    let output = format!("{}", MatrixFormatter::new(&eye).with_var_name("I"));
    assert!(output.contains("I ="));
    assert!(output.contains("1"));
}
