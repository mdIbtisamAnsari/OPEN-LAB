use openlab_core::MxArray;
use openlab_stdlib::{
    diag, disp, eig, eye, fliplr, flipud, horzcat, length, linspace, logspace, meshgrid,
    numel, ones, rand, randn, repmat, rot90, size, sprintf, tril, triu, vertcat, zeros,
    BuiltinRegistry, DimSpec,
};

#[test]
fn test_constructors() {
    let z = zeros(&[2, 3]);
    assert_eq!(z.rows(), 2);
    assert_eq!(z.cols(), 3);
    assert_eq!(z.get_2d(1, 1).unwrap(), 0.0);

    let o = ones(&[3, 2]);
    assert_eq!(o.rows(), 3);
    assert_eq!(o.cols(), 2);
    assert_eq!(o.get_2d(1, 1).unwrap(), 1.0);

    let e = eye(3, 3);
    assert_eq!(e.get_2d(1, 1).unwrap(), 1.0);
    assert_eq!(e.get_2d(1, 2).unwrap(), 0.0);

    let r = rand(&[4, 4]);
    assert_eq!(r.numel(), 16);
    let rn = randn(&[4, 4]);
    assert_eq!(rn.numel(), 16);

    let ls = linspace(0.0, 10.0, 5).unwrap();
    assert_eq!(ls.numel(), 5);
    assert_eq!(ls.get_linear_0based(0).unwrap(), 0.0);
    assert_eq!(ls.get_linear_0based(4).unwrap(), 10.0);

    let logs = logspace(1.0, 3.0, 3).unwrap();
    assert_eq!(logs.numel(), 3);
    assert_eq!(logs.get_linear_0based(0).unwrap(), 10.0);
    assert_eq!(logs.get_linear_0based(1).unwrap(), 100.0);
    assert_eq!(logs.get_linear_0based(2).unwrap(), 1000.0);
}

#[test]
fn test_diag_and_meshgrid() {
    // Vector -> Diagonal matrix
    let v = linspace(1.0, 3.0, 3).unwrap();
    let d = diag(&v, 0).unwrap();
    assert_eq!(d.rows(), 3);
    assert_eq!(d.cols(), 3);
    assert_eq!(d.get_2d(1, 1).unwrap(), 1.0);
    assert_eq!(d.get_2d(2, 2).unwrap(), 2.0);
    assert_eq!(d.get_2d(3, 3).unwrap(), 3.0);
    assert_eq!(d.get_2d(1, 2).unwrap(), 0.0);

    // Diagonal matrix -> Vector
    let extracted = diag(&d, 0).unwrap();
    assert_eq!(extracted.numel(), 3);
    assert_eq!(extracted.get_linear_0based(0).unwrap(), 1.0);
    assert_eq!(extracted.get_linear_0based(1).unwrap(), 2.0);
    assert_eq!(extracted.get_linear_0based(2).unwrap(), 3.0);

    // Meshgrid
    let x = linspace(1.0, 3.0, 3).unwrap();
    let y = linspace(10.0, 20.0, 2).unwrap();
    let (xm, ym) = meshgrid(&x, &y).unwrap();
    assert_eq!(xm.rows(), 2);
    assert_eq!(xm.cols(), 3);
    assert_eq!(xm.get_2d(1, 1).unwrap(), 1.0);
    assert_eq!(xm.get_2d(1, 3).unwrap(), 3.0);
    assert_eq!(ym.get_2d(1, 1).unwrap(), 10.0);
    assert_eq!(ym.get_2d(2, 1).unwrap(), 20.0);
}

#[test]
fn test_manipulation() {
    let a = MxArray::from_2d_slice(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

    assert_eq!(size(&a, None).unwrap(), vec![2, 3]);
    assert_eq!(size(&a, Some(1)).unwrap(), vec![2]);
    assert_eq!(size(&a, Some(2)).unwrap(), vec![3]);
    assert_eq!(length(&a), 3);
    assert_eq!(numel(&a), 6);

    // repmat
    let rep = repmat(&a, 2, 2).unwrap();
    assert_eq!(rep.rows(), 4);
    assert_eq!(rep.cols(), 6);

    // horzcat & vertcat
    let b = MxArray::from_2d_slice(&[&[7.0, 8.0], &[9.0, 10.0]]);
    let h = horzcat(&[a.clone(), b]).unwrap();
    assert_eq!(h.rows(), 2);
    assert_eq!(h.cols(), 5);

    let v_cat = vertcat(&[a.clone(), a.clone()]).unwrap();
    assert_eq!(v_cat.rows(), 4);
    assert_eq!(v_cat.cols(), 3);

    // fliplr & flipud
    let flr = fliplr(&a).unwrap();
    assert_eq!(flr.get_2d(1, 1).unwrap(), 3.0);
    let fud = flipud(&a).unwrap();
    assert_eq!(fud.get_2d(1, 1).unwrap(), 4.0);

    // rot90
    let r90 = rot90(&a, 1).unwrap();
    assert_eq!(r90.rows(), 3);
    assert_eq!(r90.cols(), 2);

    // triu & tril
    let sq = MxArray::from_2d_slice(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let u = triu(&sq, 0).unwrap();
    assert_eq!(u.get_2d(2, 1).unwrap(), 0.0);
    assert_eq!(u.get_2d(1, 2).unwrap(), 2.0);

    let l = tril(&sq, 0).unwrap();
    assert_eq!(l.get_2d(1, 2).unwrap(), 0.0);
    assert_eq!(l.get_2d(2, 1).unwrap(), 3.0);
}

#[test]
fn test_reductions() {
    let a = MxArray::from_2d_slice(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

    // Default 2D sum down columns (dim = 1) -> 1x3 row vector [5, 7, 9]
    let s_col = openlab_stdlib::sum(&a, DimSpec::FirstNonSingleton).unwrap();
    assert_eq!(s_col.rows(), 1);
    assert_eq!(s_col.cols(), 3);
    assert_eq!(s_col.get_2d(1, 1).unwrap(), 5.0);
    assert_eq!(s_col.get_2d(1, 2).unwrap(), 7.0);
    assert_eq!(s_col.get_2d(1, 3).unwrap(), 9.0);

    // sum across rows (dim = 2) -> 2x1 col vector [6; 15]
    let s_row = openlab_stdlib::sum(&a, DimSpec::Dim(2)).unwrap();
    assert_eq!(s_row.rows(), 2);
    assert_eq!(s_row.cols(), 1);
    assert_eq!(s_row.get_2d(1, 1).unwrap(), 6.0);
    assert_eq!(s_row.get_2d(2, 1).unwrap(), 15.0);

    // sum(A, 'all') -> scalar 21
    let s_all = openlab_stdlib::sum(&a, DimSpec::All).unwrap();
    assert_eq!(s_all.to_scalar().unwrap(), 21.0);

    // mean
    let m_all = openlab_stdlib::mean(&a, DimSpec::All).unwrap();
    assert_eq!(m_all.to_scalar().unwrap(), 3.5);

    // min / max
    let min_all = openlab_stdlib::min(&a, DimSpec::All).unwrap();
    assert_eq!(min_all.to_scalar().unwrap(), 1.0);
    let max_all = openlab_stdlib::max(&a, DimSpec::All).unwrap();
    assert_eq!(max_all.to_scalar().unwrap(), 6.0);
}

#[test]
fn test_eigenvalues() {
    // Symmetric matrix [2, 1; 1, 2] has eigenvalues 3 and 1
    let a = MxArray::from_2d_slice(&[&[2.0, 1.0], &[1.0, 2.0]]);
    let (e, _) = eig(&a, 1).unwrap();
    assert_eq!(e.numel(), 2);
    let e1 = e.get_linear_0based(0).unwrap();
    let e2 = e.get_linear_0based(1).unwrap();
    assert!((e1 - 3.0).abs() < 1e-10 || (e1 - 1.0).abs() < 1e-10);
    assert!((e2 - 3.0).abs() < 1e-10 || (e2 - 1.0).abs() < 1e-10);

    // Full decomposition [V, D] = eig(A)
    let (v, d_opt) = eig(&a, 2).unwrap();
    let d = d_opt.expect("D must be present when nargout >= 2");
    // Verify A * V = V * D
    let av = openlab_math::matmul(&a, &v).unwrap();
    let vd = openlab_math::matmul(&v, &d).unwrap();
    let diff = av.sub(&vd).unwrap();
    let err = openlab_math::norm(&diff, openlab_math::NormType::Inf).unwrap();
    assert!(err < 1e-12, "A*V must equal V*D, error: {}", err);
}

#[test]
fn test_io_functions() {
    let s = sprintf("Matrix size: %d x %d", &[MxArray::from_scalar(3.0), MxArray::from_scalar(4.0)]).unwrap();
    assert_eq!(s, "Matrix size: 3 x 4");

    let a = MxArray::from_2d_slice(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let d = disp(&a).unwrap();
    assert!(d.contains("1"));
    assert!(d.contains("4"));
}

#[test]
fn test_builtin_registry() {
    let reg = BuiltinRegistry::new();
    assert!(reg.contains("zeros"));
    assert!(reg.contains("sin"));
    assert!(reg.contains("eig"));
    assert!(reg.contains("lu"));

    // Call size with multiple return values
    let a = MxArray::zeros(4, 5);
    let sz_out = reg.call("size", &[a], 2).unwrap();
    assert_eq!(sz_out.len(), 2);
    assert_eq!(sz_out[0].to_scalar().unwrap(), 4.0);
    assert_eq!(sz_out[1].to_scalar().unwrap(), 5.0);
}
