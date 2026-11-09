use openlab_core::{CoreError, MxArray, Shape};

pub fn size(arr: &MxArray, dim: Option<usize>) -> Result<Vec<usize>, CoreError> {
    let dims = arr.shape().dims();
    if let Some(d) = dim {
        if d == 0 {
            return Err(CoreError::InvalidOperation("Dimension must be >= 1".into()));
        }
        if d <= dims.len() {
            Ok(vec![dims[d - 1]])
        } else {
            Ok(vec![1])
        }
    } else {
        Ok(dims.to_vec())
    }
}

pub fn length(arr: &MxArray) -> usize {
    if arr.is_empty() {
        0
    } else {
        arr.shape().dims().iter().copied().max().unwrap_or(0)
    }
}

pub fn numel(arr: &MxArray) -> usize {
    arr.numel()
}

pub fn ndims(arr: &MxArray) -> usize {
    arr.ndims()
}

pub fn isempty(arr: &MxArray) -> bool {
    arr.is_empty()
}

pub fn isvector(arr: &MxArray) -> bool {
    arr.is_vector()
}

pub fn isscalar(arr: &MxArray) -> bool {
    arr.is_scalar()
}

pub fn ismatrix(arr: &MxArray) -> bool {
    arr.ndims() == 2
}

pub fn reshape(arr: &MxArray, new_shape: &[usize]) -> Result<MxArray, CoreError> {
    arr.reshape(Shape::new(new_shape))
}

pub fn squeeze(arr: &MxArray) -> Result<MxArray, CoreError> {
    let dims = arr.shape().dims();
    let mut non_singletons: Vec<usize> = dims.iter().copied().filter(|&d| d != 1).collect();
    if non_singletons.len() < 2 {
        while non_singletons.len() < 2 {
            non_singletons.push(1);
        }
    }
    arr.reshape(Shape::new(&non_singletons))
}

/// Replicates and tiles an array: repmat(A, m, n)
pub fn repmat(arr: &MxArray, m_tiles: usize, n_tiles: usize) -> Result<MxArray, CoreError> {
    if !arr.shape().is_2d() {
        return Err(CoreError::InvalidOperation(
            "repmat currently supports 2D arrays.".into(),
        ));
    }

    let m = arr.rows();
    let n = arr.cols();
    let out_m = m * m_tiles;
    let out_n = n * n_tiles;

    let mut out = MxArray::zeros(out_m, out_n);
    for tr in 0..m_tiles {
        for tc in 0..n_tiles {
            let r_offset = tr * m;
            let c_offset = tc * n;
            for r in 1..=m {
                for c in 1..=n {
                    let val = arr.get_2d(r, c)?;
                    out.set_2d(r_offset + r, c_offset + c, val)?;
                }
            }
        }
    }

    Ok(out)
}

/// Horizontal concatenation: [A, B, C]
pub fn horzcat(arrays: &[MxArray]) -> Result<MxArray, CoreError> {
    if arrays.is_empty() {
        return Ok(MxArray::empty_double());
    }

    let first_rows = arrays[0].rows();
    let mut total_cols = 0;

    for arr in arrays {
        if !arr.is_empty() && arr.rows() != first_rows {
            return Err(CoreError::DimensionMismatch {
                expected: vec![first_rows],
                actual: vec![arr.rows()],
            });
        }
        total_cols += arr.cols();
    }

    let mut out = MxArray::zeros(first_rows, total_cols);
    let mut col_cursor = 1;

    for arr in arrays {
        let cols = arr.cols();
        for c in 1..=cols {
            for r in 1..=first_rows {
                let val = arr.get_2d(r, c)?;
                out.set_2d(r, col_cursor, val)?;
            }
            col_cursor += 1;
        }
    }

    Ok(out)
}

/// Vertical concatenation: [A; B; C]
pub fn vertcat(arrays: &[MxArray]) -> Result<MxArray, CoreError> {
    if arrays.is_empty() {
        return Ok(MxArray::empty_double());
    }

    let first_cols = arrays[0].cols();
    let mut total_rows = 0;

    for arr in arrays {
        if !arr.is_empty() && arr.cols() != first_cols {
            return Err(CoreError::DimensionMismatch {
                expected: vec![first_cols],
                actual: vec![arr.cols()],
            });
        }
        total_rows += arr.rows();
    }

    let mut out = MxArray::zeros(total_rows, first_cols);
    let mut row_cursor = 1;

    for arr in arrays {
        let rows = arr.rows();
        for r in 1..=rows {
            for c in 1..=first_cols {
                let val = arr.get_2d(r, c)?;
                out.set_2d(row_cursor, c, val)?;
            }
            row_cursor += 1;
        }
    }

    Ok(out)
}

/// General N-D array concatenation along dimension dim: cat(dim, A, B)
pub fn cat(dim: usize, arrays: &[MxArray]) -> Result<MxArray, CoreError> {
    if dim == 1 {
        vertcat(arrays)
    } else if dim == 2 {
        horzcat(arrays)
    } else {
        Err(CoreError::InvalidOperation(
            "Higher-dimensional cat not yet supported.".into(),
        ))
    }
}

pub fn fliplr(arr: &MxArray) -> Result<MxArray, CoreError> {
    if !arr.shape().is_2d() {
        return Err(CoreError::InvalidOperation("fliplr requires a 2D matrix.".into()));
    }
    let m = arr.rows();
    let n = arr.cols();
    let mut out = MxArray::zeros(m, n);
    for r in 1..=m {
        for c in 1..=n {
            let val = arr.get_2d(r, c)?;
            out.set_2d(r, n - c + 1, val)?;
        }
    }
    Ok(out)
}

pub fn flipud(arr: &MxArray) -> Result<MxArray, CoreError> {
    if !arr.shape().is_2d() {
        return Err(CoreError::InvalidOperation("flipud requires a 2D matrix.".into()));
    }
    let m = arr.rows();
    let n = arr.cols();
    let mut out = MxArray::zeros(m, n);
    for r in 1..=m {
        for c in 1..=n {
            let val = arr.get_2d(r, c)?;
            out.set_2d(m - r + 1, c, val)?;
        }
    }
    Ok(out)
}

/// Rotate matrix 90 degrees counterclockwise k times: rot90(A, k)
pub fn rot90(arr: &MxArray, k: i32) -> Result<MxArray, CoreError> {
    if !arr.shape().is_2d() {
        return Err(CoreError::InvalidOperation("rot90 requires a 2D matrix.".into()));
    }
    let rot = ((k % 4) + 4) % 4;
    match rot {
        0 => Ok(arr.clone()),
        1 => {
            // 90 deg CCW: transpose then flipud
            let t = arr.transpose()?;
            flipud(&t)
        }
        2 => {
            // 180 deg
            let f = fliplr(arr)?;
            flipud(&f)
        }
        3 => {
            // 270 deg CCW: transpose then fliplr
            let t = arr.transpose()?;
            fliplr(&t)
        }
        _ => unreachable!(),
    }
}

/// Upper triangular part of matrix: triu(A, k)
pub fn triu(arr: &MxArray, k: i32) -> Result<MxArray, CoreError> {
    if !arr.shape().is_2d() {
        return Err(CoreError::InvalidOperation("triu requires a 2D matrix.".into()));
    }
    let m = arr.rows();
    let n = arr.cols();
    let mut out = MxArray::zeros(m, n);
    for r in 1..=m {
        for c in 1..=n {
            if (c as i64) - (r as i64) >= (k as i64) {
                out.set_2d(r, c, arr.get_2d(r, c)?)?;
            }
        }
    }
    Ok(out)
}

/// Lower triangular part of matrix: tril(A, k)
pub fn tril(arr: &MxArray, k: i32) -> Result<MxArray, CoreError> {
    if !arr.shape().is_2d() {
        return Err(CoreError::InvalidOperation("tril requires a 2D matrix.".into()));
    }
    let m = arr.rows();
    let n = arr.cols();
    let mut out = MxArray::zeros(m, n);
    for r in 1..=m {
        for c in 1..=n {
            if (c as i64) - (r as i64) <= (k as i64) {
                out.set_2d(r, c, arr.get_2d(r, c)?)?;
            }
        }
    }
    Ok(out)
}

