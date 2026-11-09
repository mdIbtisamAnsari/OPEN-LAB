use openlab_core::{CoreError, MxArray, Shape};

pub enum DimSpec {
    FirstNonSingleton,
    Dim(usize),
    All,
}

impl DimSpec {
    pub fn from_opt(opt: Option<usize>) -> Self {
        match opt {
            Some(d) => DimSpec::Dim(d),
            None => DimSpec::FirstNonSingleton,
        }
    }
}

pub fn sum(arr: &MxArray, dim: DimSpec) -> Result<MxArray, CoreError> {
    reduce_2d(arr, dim, 0.0, |acc, val| acc + val)
}

pub fn prod(arr: &MxArray, dim: DimSpec) -> Result<MxArray, CoreError> {
    reduce_2d(arr, dim, 1.0, |acc, val| acc * val)
}

pub fn mean(arr: &MxArray, dim: DimSpec) -> Result<MxArray, CoreError> {
    let s = sum(arr, match dim {
        DimSpec::All => DimSpec::All,
        DimSpec::Dim(d) => DimSpec::Dim(d),
        DimSpec::FirstNonSingleton => DimSpec::FirstNonSingleton,
    })?;

    match dim {
        DimSpec::All => {
            let n = arr.numel() as f64;
            s.scalar_div(n)
        }
        DimSpec::Dim(d) => {
            let count = if d == 1 { arr.rows() } else if d == 2 { arr.cols() } else { 1 } as f64;
            s.scalar_div(count)
        }
        DimSpec::FirstNonSingleton => {
            if arr.is_vector() {
                s.scalar_div(arr.numel() as f64)
            } else {
                s.scalar_div(arr.rows() as f64)
            }
        }
    }
}

pub fn min(arr: &MxArray, dim: DimSpec) -> Result<MxArray, CoreError> {
    reduce_2d(arr, dim, f64::INFINITY, |acc, val| if val < acc { val } else { acc })
}

pub fn max(arr: &MxArray, dim: DimSpec) -> Result<MxArray, CoreError> {
    reduce_2d(arr, dim, f64::NEG_INFINITY, |acc, val| if val > acc { val } else { acc })
}

pub fn any(arr: &MxArray, dim: DimSpec) -> Result<MxArray, CoreError> {
    let res = reduce_2d(arr, dim, 0.0, |acc, val| if val != 0.0 || acc != 0.0 { 1.0 } else { 0.0 })?;
    to_logical(&res)
}

pub fn all(arr: &MxArray, dim: DimSpec) -> Result<MxArray, CoreError> {
    let res = reduce_2d(arr, dim, 1.0, |acc, val| if val == 0.0 || acc == 0.0 { 0.0 } else { 1.0 })?;
    to_logical(&res)
}

/// Cumulative sum: cumsum(A, dim)
pub fn cumsum(arr: &MxArray, dim: Option<usize>) -> Result<MxArray, CoreError> {
    if !arr.shape().is_2d() {
        return Err(CoreError::InvalidOperation("cumsum currently supports 2D arrays.".into()));
    }
    let d = dim.unwrap_or(if arr.rows() > 1 { 1 } else { 2 });
    let m = arr.rows();
    let n = arr.cols();
    let mut out = MxArray::zeros(m, n);

    if d == 1 {
        // down columns
        for c in 1..=n {
            let mut acc = 0.0;
            for r in 1..=m {
                acc += arr.get_2d(r, c)?;
                out.set_2d(r, c, acc)?;
            }
        }
    } else {
        // across rows
        for r in 1..=m {
            let mut acc = 0.0;
            for c in 1..=n {
                acc += arr.get_2d(r, c)?;
                out.set_2d(r, c, acc)?;
            }
        }
    }

    Ok(out)
}

/// Difference and approximate derivative: diff(A)
pub fn diff(arr: &MxArray) -> Result<MxArray, CoreError> {
    if !arr.shape().is_2d() {
        return Err(CoreError::InvalidOperation("diff requires a 1D vector or 2D matrix.".into()));
    }

    if arr.is_vector() {
        let n = arr.numel();
        if n <= 1 {
            return Ok(MxArray::zeros(0, 0));
        }
        let mut out = Vec::with_capacity(n - 1);
        for i in 1..n {
            let v1 = arr.get_linear_0based(i - 1)?;
            let v2 = arr.get_linear_0based(i)?;
            out.push(v2 - v1);
        }
        let shape = if arr.rows() == 1 {
            Shape::row_vector(n - 1)
        } else {
            Shape::col_vector(n - 1)
        };
        MxArray::from_f64_vec(shape, out)
    } else {
        // Default operates down columns (first non-singleton)
        let m = arr.rows();
        let n = arr.cols();
        if m <= 1 {
            return Ok(MxArray::zeros(0, n));
        }
        let mut out = MxArray::zeros(m - 1, n);
        for c in 1..=n {
            for r in 1..(m) {
                let v1 = arr.get_2d(r, c)?;
                let v2 = arr.get_2d(r + 1, c)?;
                out.set_2d(r, c, v2 - v1)?;
            }
        }
        Ok(out)
    }
}

// -----------------------------------------------------------------------------
// Internal Reducer Engine
// -----------------------------------------------------------------------------

fn reduce_2d<F>(
    arr: &MxArray,
    dim: DimSpec,
    init: f64,
    op: F,
) -> Result<MxArray, CoreError>
where
    F: Fn(f64, f64) -> f64,
{
    if arr.is_empty() {
        return Ok(MxArray::zeros(0, 0));
    }

    match dim {
        DimSpec::All => {
            let n = arr.numel();
            let mut acc = init;
            for i in 0..n {
                acc = op(acc, arr.get_linear_0based(i)?);
            }
            Ok(MxArray::from_scalar(acc))
        }
        DimSpec::Dim(1) => {
            // Down columns: (M x N) -> (1 x N)
            let m = arr.rows();
            let n = arr.cols();
            let mut out = Vec::with_capacity(n);
            for c in 1..=n {
                let mut acc = init;
                for r in 1..=m {
                    acc = op(acc, arr.get_2d(r, c)?);
                }
                out.push(acc);
            }
            MxArray::from_f64_vec(Shape::row_vector(n), out)
        }
        DimSpec::Dim(2) => {
            // Across rows: (M x N) -> (M x 1)
            let m = arr.rows();
            let n = arr.cols();
            let mut out = Vec::with_capacity(m);
            for r in 1..=m {
                let mut acc = init;
                for c in 1..=n {
                    acc = op(acc, arr.get_2d(r, c)?);
                }
                out.push(acc);
            }
            MxArray::from_f64_vec(Shape::col_vector(m), out)
        }
        DimSpec::Dim(_) => {
            // Dimension beyond 2 on 2D matrix returns self
            Ok(arr.clone())
        }
        DimSpec::FirstNonSingleton => {
            if arr.is_vector() {
                // Vector reduces to scalar
                let n = arr.numel();
                let mut acc = init;
                for i in 0..n {
                    acc = op(acc, arr.get_linear_0based(i)?);
                }
                Ok(MxArray::from_scalar(acc))
            } else if arr.rows() > 1 {
                reduce_2d(arr, DimSpec::Dim(1), init, op)
            } else {
                reduce_2d(arr, DimSpec::Dim(2), init, op)
            }
        }
    }
}

fn to_logical(arr: &MxArray) -> Result<MxArray, CoreError> {
    let n = arr.numel();
    let mut bools = Vec::with_capacity(n);
    for i in 0..n {
        bools.push(arr.get_linear_0based(i)? != 0.0);
    }
    MxArray::from_bool_vec(arr.shape().clone(), bools)
}
