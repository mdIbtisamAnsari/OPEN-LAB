use crate::complex::Complex64;
use crate::data::MxData;
use crate::error::CoreError;
use crate::shape::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct MxArray {
    shape: Shape,
    data: MxData,
}

impl MxArray {
    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    /// Creates a 2D matrix of zeros (type: double).
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::zeros_nd(Shape::matrix(rows, cols))
    }

    /// Creates an N-dimensional array of zeros (type: double).
    pub fn zeros_nd(shape: Shape) -> Self {
        let n = shape.numel();
        Self {
            shape,
            data: MxData::Double(Arc::new(vec![0.0; n])),
        }
    }

    /// Creates a 2D matrix of ones (type: double).
    pub fn ones(rows: usize, cols: usize) -> Self {
        Self::ones_nd(Shape::matrix(rows, cols))
    }

    /// Creates an N-dimensional array of ones (type: double).
    pub fn ones_nd(shape: Shape) -> Self {
        let n = shape.numel();
        Self {
            shape,
            data: MxData::Double(Arc::new(vec![1.0; n])),
        }
    }

    /// Creates an N-by-N identity matrix.
    pub fn eye(n: usize) -> Self {
        Self::eye_mn(n, n)
    }

    /// Creates an M-by-N matrix with ones on the main diagonal and zeros elsewhere.
    pub fn eye_mn(rows: usize, cols: usize) -> Self {
        let mut mat = Self::zeros(rows, cols);
        let min_dim = std::cmp::min(rows, cols);
        for i in 1..=min_dim {
            let _ = mat.set_2d(i, i, 1.0);
        }
        mat
    }

    /// Creates a 1x1 scalar (type: double).
    pub fn from_scalar(val: f64) -> Self {
        Self {
            shape: Shape::scalar(),
            data: MxData::Double(Arc::new(vec![val])),
        }
    }

    /// Creates a 1x1 complex scalar (type: double complex).
    pub fn from_scalar_complex(val: Complex64) -> Self {
        Self {
            shape: Shape::scalar(),
            data: MxData::ComplexDouble(Arc::new(vec![val])),
        }
    }

    /// Creates a 1x1 logical scalar (type: logical).
    pub fn from_scalar_bool(val: bool) -> Self {
        Self {
            shape: Shape::scalar(),
            data: MxData::Logical(Arc::new(vec![val])),
        }
    }

    /// Creates an array from a Shape and raw column-major f64 vector.
    pub fn from_f64_vec(shape: Shape, data: Vec<f64>) -> Result<Self, CoreError> {
        if data.len() != shape.numel() {
            return Err(CoreError::DimensionMismatch {
                expected: vec![shape.numel()],
                actual: vec![data.len()],
            });
        }
        Ok(Self {
            shape,
            data: MxData::Double(Arc::new(data)),
        })
    }

    /// Creates an array from a Shape and raw column-major Complex64 vector.
    pub fn from_complex_vec(shape: Shape, data: Vec<Complex64>) -> Result<Self, CoreError> {
        if data.len() != shape.numel() {
            return Err(CoreError::DimensionMismatch {
                expected: vec![shape.numel()],
                actual: vec![data.len()],
            });
        }
        Ok(Self {
            shape,
            data: MxData::ComplexDouble(Arc::new(data)),
        })
    }

    /// Creates a 2D matrix from row-wise nested slices, converting to column-major storage.
    ///
    /// Example: `MxArray::from_2d_slice(&[&[1.0, 2.0], &[3.0, 4.0]])`
    pub fn from_2d_slice(rows: &[&[f64]]) -> Self {
        if rows.is_empty() {
            return Self::empty_double();
        }
        let m = rows.len();
        let n = rows[0].len();
        let mut col_major = vec![0.0; m * n];

        for (r, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), n, "All rows must have the same length");
            for (c, &val) in row.iter().enumerate() {
                // Column-major linear index: r + c * m
                col_major[r + c * m] = val;
            }
        }

        Self {
            shape: Shape::matrix(m, n),
            data: MxData::Double(Arc::new(col_major)),
        }
    }

    /// Creates a char matrix (1xN row vector) from a string.
    pub fn from_string(s: &str) -> Self {
        let chars: Vec<char> = s.chars().collect();
        let len = chars.len();
        Self {
            shape: Shape::row_vector(len),
            data: MxData::Char(Arc::new(chars)),
        }
    }

    /// Creates a logical array from a boolean vector and shape.
    pub fn from_bool_vec(shape: Shape, data: Vec<bool>) -> Result<Self, CoreError> {
        if data.len() != shape.numel() {
            return Err(CoreError::DimensionMismatch {
                expected: vec![shape.numel()],
                actual: vec![data.len()],
            });
        }
        Ok(Self {
            shape,
            data: MxData::Logical(Arc::new(data)),
        })
    }

    /// Creates an empty double array (`[]`, shape: 0x0).
    pub fn empty_double() -> Self {
        Self {
            shape: Shape::matrix(0, 0),
            data: MxData::Double(Arc::new(Vec::new())),
        }
    }

    // -------------------------------------------------------------------------
    // Metadata & Introspection
    // -------------------------------------------------------------------------

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn data(&self) -> &MxData {
        &self.data
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn ndims(&self) -> usize {
        self.shape.ndims()
    }

    pub fn rows(&self) -> usize {
        self.shape.rows()
    }

    pub fn cols(&self) -> usize {
        self.shape.cols()
    }

    pub fn class_name(&self) -> &'static str {
        self.data.class_name()
    }

    pub fn is_empty(&self) -> bool {
        self.shape.is_empty()
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.is_scalar()
    }

    pub fn is_vector(&self) -> bool {
        self.shape.is_vector()
    }

    pub fn is_numeric(&self) -> bool {
        self.data.is_numeric()
    }

    pub fn is_complex(&self) -> bool {
        self.data.is_complex()
    }

    pub fn is_logical(&self) -> bool {
        self.data.is_logical()
    }

    pub fn total_bytes(&self) -> usize {
        self.data.total_bytes()
    }

    /// Extracts scalar f64 if this array is 1x1 double.
    pub fn to_scalar(&self) -> Result<f64, CoreError> {
        if !self.is_scalar() {
            return Err(CoreError::InvalidOperation(
                "Array is not a scalar.".into(),
            ));
        }
        match &self.data {
            MxData::Double(v) => Ok(v[0]),
            _ => Err(CoreError::IncompatibleTypes {
                expected: "double",
                actual: self.class_name(),
            }),
        }
    }

    // -------------------------------------------------------------------------
    // Element Access & Mutating (with CoW)
    // -------------------------------------------------------------------------

    /// 0-based linear index read for double.
    pub fn get_linear_0based(&self, idx: usize) -> Result<f64, CoreError> {
        if idx >= self.numel() {
            return Err(CoreError::LinearIndexOutOfBounds {
                index: idx + 1,
                bound: self.numel(),
            });
        }
        match &self.data {
            MxData::Double(v) => Ok(v[idx]),
            MxData::Logical(v) => Ok(if v[idx] { 1.0 } else { 0.0 }),
            _ => Err(CoreError::IncompatibleTypes {
                expected: "double",
                actual: self.class_name(),
            }),
        }
    }

    /// 1-based linear index read for double (MATLAB standard: `A(5)`).
    pub fn get_linear_1based(&self, idx_1based: usize) -> Result<f64, CoreError> {
        if idx_1based == 0 {
            return Err(CoreError::LinearIndexOutOfBounds {
                index: 0,
                bound: self.numel(),
            });
        }
        self.get_linear_0based(idx_1based - 1)
    }

    /// 1-based 2D matrix read (`A(r, c)`).
    pub fn get_2d(&self, row_1based: usize, col_1based: usize) -> Result<f64, CoreError> {
        let linear = self.shape.linear_index_1based(&[row_1based, col_1based])?;
        self.get_linear_0based(linear)
    }

    /// 1-based N-dimensional read (`A(i, j, k, ...)`).
    pub fn get_nd(&self, coords_1based: &[usize]) -> Result<f64, CoreError> {
        let linear = self.shape.linear_index_1based(coords_1based)?;
        self.get_linear_0based(linear)
    }

    /// 0-based linear index write with Copy-on-Write (CoW).
    pub fn set_linear_0based(&mut self, idx: usize, val: f64) -> Result<(), CoreError> {
        let total = self.numel();
        if idx >= total {
            return Err(CoreError::LinearIndexOutOfBounds {
                index: idx + 1,
                bound: total,
            });
        }
        match &mut self.data {
            MxData::Double(ref mut arc) => {
                let vec = Arc::make_mut(arc);
                vec[idx] = val;
                Ok(())
            }
            _ => Err(CoreError::IncompatibleTypes {
                expected: "double",
                actual: self.class_name(),
            }),
        }
    }

    /// 1-based linear index write with CoW.
    pub fn set_linear_1based(&mut self, idx_1based: usize, val: f64) -> Result<(), CoreError> {
        if idx_1based == 0 {
            return Err(CoreError::LinearIndexOutOfBounds {
                index: 0,
                bound: self.numel(),
            });
        }
        self.set_linear_0based(idx_1based - 1, val)
    }

    /// 1-based 2D matrix write (`A(r, c) = val`).
    pub fn set_2d(&mut self, row_1based: usize, col_1based: usize, val: f64) -> Result<(), CoreError> {
        let linear = self.shape.linear_index_1based(&[row_1based, col_1based])?;
        self.set_linear_0based(linear, val)
    }

    /// 1-based N-dimensional write.
    pub fn set_nd(&mut self, coords_1based: &[usize], val: f64) -> Result<(), CoreError> {
        let linear = self.shape.linear_index_1based(coords_1based)?;
        self.set_linear_0based(linear, val)
    }

    /// 1-based 2D read for Complex64.
    pub fn get_complex_2d(&self, row_1based: usize, col_1based: usize) -> Result<Complex64, CoreError> {
        let linear = self.shape.linear_index_1based(&[row_1based, col_1based])?;
        match &self.data {
            MxData::ComplexDouble(v) => Ok(v[linear]),
            MxData::Double(v) => Ok(Complex64::real(v[linear])),
            _ => Err(CoreError::IncompatibleTypes {
                expected: "complex double",
                actual: self.class_name(),
            }),
        }
    }

    /// 1-based 2D write for Complex64.
    pub fn set_complex_2d(
        &mut self,
        row_1based: usize,
        col_1based: usize,
        val: Complex64,
    ) -> Result<(), CoreError> {
        let linear = self.shape.linear_index_1based(&[row_1based, col_1based])?;
        // If currently real double and val is complex, promote to complex double
        if let MxData::Double(ref arc) = self.data {
            if !val.is_real() {
                let complex_vec: Vec<Complex64> = arc.iter().map(|&x| Complex64::real(x)).collect();
                self.data = MxData::ComplexDouble(Arc::new(complex_vec));
            }
        }

        match &mut self.data {
            MxData::ComplexDouble(ref mut arc) => {
                let vec = Arc::make_mut(arc);
                vec[linear] = val;
                Ok(())
            }
            MxData::Double(ref mut arc) => {
                let vec = Arc::make_mut(arc);
                vec[linear] = val.re;
                Ok(())
            }
            _ => Err(CoreError::IncompatibleTypes {
                expected: "complex double",
                actual: self.class_name(),
            }),
        }
    }

    /// Borrow contiguous f64 slice if data is Double.
    pub fn as_slice_f64(&self) -> Option<&[f64]> {
        match &self.data {
            MxData::Double(arc) => Some(arc.as_slice()),
            _ => None,
        }
    }

    /// Borrow mutable f64 slice (triggers Copy-on-Write).
    pub fn as_mut_slice_f64(&mut self) -> Option<&mut [f64]> {
        match &mut self.data {
            MxData::Double(ref mut arc) => Some(Arc::make_mut(arc).as_mut_slice()),
            _ => None,
        }
    }

    // -------------------------------------------------------------------------
    // Matrix Transformations
    // -------------------------------------------------------------------------

    /// Reshapes array into a new shape having the exact same number of elements.
    pub fn reshape(&self, new_shape: Shape) -> Result<MxArray, CoreError> {
        if self.numel() != new_shape.numel() {
            return Err(CoreError::InvalidReshape {
                from: self.numel(),
                to: new_shape.numel(),
            });
        }
        Ok(MxArray {
            shape: new_shape,
            data: self.data.clone(), // CoW zero allocation!
        })
    }

    /// 2D Matrix transpose (`A.'`).
    pub fn transpose(&self) -> Result<MxArray, CoreError> {
        if !self.shape.is_2d() {
            return Err(CoreError::InvalidOperation(
                "Transpose is only defined for 2D arrays.".into(),
            ));
        }
        let m = self.rows();
        let n = self.cols();
        let new_shape = Shape::matrix(n, m);

        match &self.data {
            MxData::Double(src) => {
                let mut out = vec![0.0; m * n];
                for r in 0..m {
                    for c in 0..n {
                        // Original (r, c) is at r + c * m
                        // Transposed (c, r) is at c + r * n
                        out[c + r * n] = src[r + c * m];
                    }
                }
                Ok(MxArray {
                    shape: new_shape,
                    data: MxData::Double(Arc::new(out)),
                })
            }
            MxData::ComplexDouble(src) => {
                let mut out = vec![Complex64::zero(); m * n];
                for r in 0..m {
                    for c in 0..n {
                        out[c + r * n] = src[r + c * m];
                    }
                }
                Ok(MxArray {
                    shape: new_shape,
                    data: MxData::ComplexDouble(Arc::new(out)),
                })
            }
            _ => Err(CoreError::InvalidOperation(
                "Transpose not implemented for this data type.".into(),
            )),
        }
    }

    /// 2D Complex Conjugate Transpose (Hermitian transpose `A'`).
    pub fn ctranspose(&self) -> Result<MxArray, CoreError> {
        if !self.shape.is_2d() {
            return Err(CoreError::InvalidOperation(
                "Complex conjugate transpose is only defined for 2D arrays.".into(),
            ));
        }
        let m = self.rows();
        let n = self.cols();
        let new_shape = Shape::matrix(n, m);

        match &self.data {
            MxData::Double(_) => self.transpose(),
            MxData::ComplexDouble(src) => {
                let mut out = vec![Complex64::zero(); m * n];
                for r in 0..m {
                    for c in 0..n {
                        out[c + r * n] = src[r + c * m].conj();
                    }
                }
                Ok(MxArray {
                    shape: new_shape,
                    data: MxData::ComplexDouble(Arc::new(out)),
                })
            }
            _ => Err(CoreError::InvalidOperation(
                "Conjugate transpose not implemented for this data type.".into(),
            )),
        }
    }

    /// Extracts a 2D submatrix (1-based, inclusive): `A(r_start..=r_end, c_start..=c_end)`.
    pub fn submatrix(
        &self,
        r_start: usize,
        r_end: usize,
        c_start: usize,
        c_end: usize,
    ) -> Result<MxArray, CoreError> {
        if !self.shape.is_2d() {
            return Err(CoreError::InvalidOperation(
                "Submatrix is only defined for 2D arrays.".into(),
            ));
        }
        if r_start == 0 || r_end > self.rows() || r_start > r_end {
            return Err(CoreError::IndexOutOfBounds {
                dim: 1,
                index: r_end,
                bound: self.rows(),
            });
        }
        if c_start == 0 || c_end > self.cols() || c_start > c_end {
            return Err(CoreError::IndexOutOfBounds {
                dim: 2,
                index: c_end,
                bound: self.cols(),
            });
        }

        let out_rows = r_end - r_start + 1;
        let out_cols = c_end - c_start + 1;
        let out_shape = Shape::matrix(out_rows, out_cols);
        let mut out_data = vec![0.0; out_rows * out_cols];

        for (out_c, c) in (c_start..=c_end).enumerate() {
            for (out_r, r) in (r_start..=r_end).enumerate() {
                let val = self.get_2d(r, c)?;
                out_data[out_r + out_c * out_rows] = val;
            }
        }

        Ok(MxArray {
            shape: out_shape,
            data: MxData::Double(Arc::new(out_data)),
        })
    }

    // -------------------------------------------------------------------------
    // Element-Wise Operations (with Broadcasting)
    // -------------------------------------------------------------------------

    /// Element-wise binary operation helper with broadcasting.
    fn binary_elementwise_f64<F>(&self, other: &MxArray, op: F) -> Result<MxArray, CoreError>
    where
        F: Fn(f64, f64) -> f64,
    {
        let out_shape = self.shape.broadcast_shape(&other.shape)?;
        let n = out_shape.numel();
        let mut out_vec = Vec::with_capacity(n);

        let strides_out = out_shape.strides();
        let strides_a = self.shape.strides();
        let strides_b = other.shape.strides();

        let nd = out_shape.ndims();

        let slice_a = match &self.data {
            MxData::Double(v) => v.as_slice(),
            _ => {
                return Err(CoreError::IncompatibleTypes {
                    expected: "double",
                    actual: self.class_name(),
                })
            }
        };
        let slice_b = match &other.data {
            MxData::Double(v) => v.as_slice(),
            _ => {
                return Err(CoreError::IncompatibleTypes {
                    expected: "double",
                    actual: other.class_name(),
                })
            }
        };

        // If shapes are identical, fast path 1D iteration!
        if self.shape == other.shape {
            for i in 0..n {
                out_vec.push(op(slice_a[i], slice_b[i]));
            }
            return Ok(MxArray {
                shape: out_shape,
                data: MxData::Double(Arc::new(out_vec)),
            });
        }

        // Broadcasting iteration
        for lin_out in 0..n {
            // Deconstruct lin_out into coords
            let mut lin_a = 0;
            let mut lin_b = 0;
            let mut temp = lin_out;

            for d in (0..nd).rev() {
                let coord_out = temp / strides_out[d];
                temp %= strides_out[d];

                let dim_a = self.shape.dims().get(d).copied().unwrap_or(1);
                let dim_b = other.shape.dims().get(d).copied().unwrap_or(1);

                let coord_a = if dim_a == 1 { 0 } else { coord_out };
                let coord_b = if dim_b == 1 { 0 } else { coord_out };

                if d < strides_a.len() {
                    lin_a += coord_a * strides_a[d];
                }
                if d < strides_b.len() {
                    lin_b += coord_b * strides_b[d];
                }
            }

            out_vec.push(op(slice_a[lin_a], slice_b[lin_b]));
        }

        Ok(MxArray {
            shape: out_shape,
            data: MxData::Double(Arc::new(out_vec)),
        })
    }

    /// Element-wise addition (`A + B`) with broadcasting.
    pub fn add(&self, other: &MxArray) -> Result<MxArray, CoreError> {
        self.binary_elementwise_f64(other, |a, b| a + b)
    }

    /// Element-wise subtraction (`A - B`) with broadcasting.
    pub fn sub(&self, other: &MxArray) -> Result<MxArray, CoreError> {
        self.binary_elementwise_f64(other, |a, b| a - b)
    }

    /// Element-wise multiplication (`A .* B`) with broadcasting.
    pub fn elementwise_mul(&self, other: &MxArray) -> Result<MxArray, CoreError> {
        self.binary_elementwise_f64(other, |a, b| a * b)
    }

    /// Element-wise right division (`A ./ B`) with broadcasting.
    pub fn elementwise_div(&self, other: &MxArray) -> Result<MxArray, CoreError> {
        self.binary_elementwise_f64(other, |a, b| a / b)
    }

    /// Adds a scalar to all elements: `A + s`.
    pub fn scalar_add(&self, s: f64) -> Result<MxArray, CoreError> {
        match &self.data {
            MxData::Double(v) => {
                let out: Vec<f64> = v.iter().map(|&x| x + s).collect();
                Ok(MxArray {
                    shape: self.shape.clone(),
                    data: MxData::Double(Arc::new(out)),
                })
            }
            _ => Err(CoreError::IncompatibleTypes {
                expected: "double",
                actual: self.class_name(),
            }),
        }
    }

    /// Subtracts a scalar from all elements: `A - s`.
    pub fn scalar_sub(&self, s: f64) -> Result<MxArray, CoreError> {
        self.scalar_add(-s)
    }

    /// Multiplies all elements by a scalar: `A * s`.
    pub fn scalar_mul(&self, s: f64) -> Result<MxArray, CoreError> {
        match &self.data {
            MxData::Double(v) => {
                let out: Vec<f64> = v.iter().map(|&x| x * s).collect();
                Ok(MxArray {
                    shape: self.shape.clone(),
                    data: MxData::Double(Arc::new(out)),
                })
            }
            _ => Err(CoreError::IncompatibleTypes {
                expected: "double",
                actual: self.class_name(),
            }),
        }
    }

    /// Divides all elements by a scalar: `A / s`.
    pub fn scalar_div(&self, s: f64) -> Result<MxArray, CoreError> {
        if s == 0.0 {
            // In MATLAB, division by 0 yields Inf or NaN, but we can also check
        }
        self.scalar_mul(1.0 / s)
    }

    /// Negates all elements: `-A`.
    pub fn negate(&self) -> Result<MxArray, CoreError> {
        self.scalar_mul(-1.0)
    }

    // -------------------------------------------------------------------------
    // Reductions
    // -------------------------------------------------------------------------

    /// Sums all elements in the array.
    pub fn sum_all(&self) -> Result<f64, CoreError> {
        match &self.data {
            MxData::Double(v) => Ok(v.iter().sum()),
            _ => Err(CoreError::IncompatibleTypes {
                expected: "double",
                actual: self.class_name(),
            }),
        }
    }

    /// Product of all elements in the array.
    pub fn prod_all(&self) -> Result<f64, CoreError> {
        match &self.data {
            MxData::Double(v) => Ok(v.iter().product()),
            _ => Err(CoreError::IncompatibleTypes {
                expected: "double",
                actual: self.class_name(),
            }),
        }
    }

    /// Mean of all elements in the array.
    pub fn mean_all(&self) -> Result<f64, CoreError> {
        let n = self.numel();
        if n == 0 {
            return Err(CoreError::EmptyArray);
        }
        let sum = self.sum_all()?;
        Ok(sum / (n as f64))
    }

    /// Minimum element in the array.
    pub fn min_all(&self) -> Result<f64, CoreError> {
        match &self.data {
            MxData::Double(v) => {
                if v.is_empty() {
                    return Err(CoreError::EmptyArray);
                }
                let mut min_val = f64::INFINITY;
                for &val in v.iter() {
                    if val < min_val {
                        min_val = val;
                    }
                }
                Ok(min_val)
            }
            _ => Err(CoreError::IncompatibleTypes {
                expected: "double",
                actual: self.class_name(),
            }),
        }
    }

    /// Maximum element in the array.
    pub fn max_all(&self) -> Result<f64, CoreError> {
        match &self.data {
            MxData::Double(v) => {
                if v.is_empty() {
                    return Err(CoreError::EmptyArray);
                }
                let mut max_val = f64::NEG_INFINITY;
                for &val in v.iter() {
                    if val > max_val {
                        max_val = val;
                    }
                }
                Ok(max_val)
            }
            _ => Err(CoreError::IncompatibleTypes {
                expected: "double",
                actual: self.class_name(),
            }),
        }
    }
}
