use crate::error::CoreError;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Creates a new Shape, guaranteeing at least 2 dimensions,
    /// and stripping trailing dimensions of size 1 beyond dimension 2.
    pub fn new(dims: &[usize]) -> Self {
        if dims.is_empty() {
            return Self { dims: vec![0, 0] };
        }
        if dims.len() == 1 {
            return Self {
                dims: vec![dims[0], 1],
            };
        }
        let mut d = dims.to_vec();
        while d.len() > 2 && d.last() == Some(&1) {
            d.pop();
        }
        Self { dims: d }
    }

    /// Creates a 2D matrix shape [rows, cols].
    pub fn matrix(rows: usize, cols: usize) -> Self {
        Self {
            dims: vec![rows, cols],
        }
    }

    /// Creates a 1x1 scalar shape.
    pub fn scalar() -> Self {
        Self { dims: vec![1, 1] }
    }

    /// Creates a 1xN row vector shape.
    pub fn row_vector(len: usize) -> Self {
        Self { dims: vec![1, len] }
    }

    /// Creates an Nx1 column vector shape.
    pub fn col_vector(len: usize) -> Self {
        Self { dims: vec![len, 1] }
    }

    /// Total number of elements in the array.
    pub fn numel(&self) -> usize {
        if self.dims.contains(&0) {
            0
        } else {
            self.dims.iter().product()
        }
    }

    /// Number of dimensions (always >= 2).
    pub fn ndims(&self) -> usize {
        self.dims.len()
    }

    /// Slice of dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Number of rows (dimension 1).
    pub fn rows(&self) -> usize {
        self.dims.first().copied().unwrap_or(0)
    }

    /// Number of columns (dimension 2).
    pub fn cols(&self) -> usize {
        self.dims.get(1).copied().unwrap_or(0)
    }

    /// Whether this shape represents an empty array (numel == 0).
    pub fn is_empty(&self) -> bool {
        self.numel() == 0
    }

    /// Whether this shape is a 1x1 scalar.
    pub fn is_scalar(&self) -> bool {
        self.dims.len() == 2 && self.dims[0] == 1 && self.dims[1] == 1
    }

    /// Whether this shape is a row or column vector.
    pub fn is_vector(&self) -> bool {
        if self.dims.len() != 2 {
            return false;
        }
        (self.dims[0] == 1 && self.dims[1] > 1) || (self.dims[0] > 1 && self.dims[1] == 1)
    }

    /// Whether this shape is strictly 2-dimensional.
    pub fn is_2d(&self) -> bool {
        self.dims.len() == 2
    }

    /// Computes column-major strides for each dimension.
    pub fn strides(&self) -> Vec<usize> {
        let mut strides = Vec::with_capacity(self.dims.len());
        let mut current_stride = 1;
        for &dim in &self.dims {
            strides.push(current_stride);
            current_stride *= dim;
        }
        strides
    }

    /// Computes the 0-based linear index from 0-based multi-dimensional coordinates in column-major order.
    /// Supports:
    /// 1. Exact coordinate matching: coords.len() == dims.len()
    /// 2. Single-subscript linear indexing: coords.len() == 1 (A(k))
    /// 3. Trailing singleton subscripts: coords.len() > dims.len() where extra coords == 0
    /// 4. Partial linear indexing: coords.len() < dims.len() where last coord spans remaining dims
    pub fn linear_index_0based(&self, coords: &[usize]) -> Result<usize, CoreError> {
        if coords.is_empty() {
            return Err(CoreError::InvalidShape("Empty coordinate list.".into()));
        }

        // Single subscript linear indexing
        if coords.len() == 1 {
            let idx = coords[0];
            if idx >= self.numel() {
                return Err(CoreError::LinearIndexOutOfBounds {
                    index: idx + 1,
                    bound: self.numel(),
                });
            }
            return Ok(idx);
        }

        // If extra trailing coordinates are supplied (e.g. A(1, 2, 1)),
        // they must be 0 (in 0-based) otherwise index exceeds bounds.
        if coords.len() > self.dims.len() {
            for (extra_dim, &c) in coords[self.dims.len()..].iter().enumerate() {
                if c != 0 {
                    return Err(CoreError::IndexOutOfBounds {
                        dim: self.dims.len() + extra_dim + 1,
                        index: c + 1,
                        bound: 1,
                    });
                }
            }
        }

        let compare_len = std::cmp::min(coords.len(), self.dims.len());
        let mut linear = 0;
        let mut stride = 1;

        for i in 0..compare_len {
            let coord = coords[i];
            let is_last = i == coords.len() - 1 && coords.len() < self.dims.len();

            if is_last {
                // Partial linear indexing: remaining capacity is product of remaining dimensions
                let remaining_bound: usize = self.dims[i..].iter().product();
                if coord >= remaining_bound {
                    return Err(CoreError::IndexOutOfBounds {
                        dim: i + 1,
                        index: coord + 1,
                        bound: remaining_bound,
                    });
                }
                linear += coord * stride;
            } else {
                let dim = self.dims[i];
                if coord >= dim {
                    return Err(CoreError::IndexOutOfBounds {
                        dim: i + 1,
                        index: coord + 1,
                        bound: dim,
                    });
                }
                linear += coord * stride;
                stride *= dim;
            }
        }

        Ok(linear)
    }

    /// Computes the 0-based linear index from 1-based MATLAB coordinates.
    pub fn linear_index_1based(&self, coords_1based: &[usize]) -> Result<usize, CoreError> {
        for (i, &c) in coords_1based.iter().enumerate() {
            if c == 0 {
                return Err(CoreError::IndexOutOfBounds {
                    dim: i + 1,
                    index: 0,
                    bound: self.dims.get(i).copied().unwrap_or(0),
                });
            }
        }
        let coords_0based: Vec<usize> = coords_1based.iter().map(|&c| c - 1).collect();
        self.linear_index_0based(&coords_0based)
    }

    /// Computes 0-based multi-dimensional coordinates from a 0-based linear index.
    pub fn coords_from_linear_0based(&self, mut linear: usize) -> Vec<usize> {
        let mut coords = vec![0; self.dims.len()];
        let strides = self.strides();
        for i in (0..self.dims.len()).rev() {
            let s = strides[i];
            coords[i] = linear / s;
            linear %= s;
        }
        coords
    }

    /// Computes 1-based multi-dimensional coordinates from a 0-based linear index.
    pub fn coords_from_linear_1based(&self, linear: usize) -> Vec<usize> {
        self.coords_from_linear_0based(linear)
            .into_iter()
            .map(|c| c + 1)
            .collect()
    }

    /// Transposes a 2D shape [rows, cols] -> [cols, rows].
    pub fn transpose_2d(&self) -> Result<Self, CoreError> {
        if !self.is_2d() {
            return Err(CoreError::InvalidOperation(
                "Transpose is only defined for 2D arrays.".into(),
            ));
        }
        Ok(Self::matrix(self.cols(), self.rows()))
    }

    /// Checks if this shape can broadcast with another shape under MATLAB broadcasting rules.
    pub fn can_broadcast_with(&self, other: &Shape) -> bool {
        self.broadcast_shape(other).is_ok()
    }

    /// Computes the broadcasted shape resulting from combining self and other.
    pub fn broadcast_shape(&self, other: &Shape) -> Result<Shape, CoreError> {
        let max_ndims = std::cmp::max(self.dims.len(), other.dims.len());
        let mut result_dims = Vec::with_capacity(max_ndims);

        for i in 0..max_ndims {
            let dim_a = self.dims.get(i).copied().unwrap_or(1);
            let dim_b = other.dims.get(i).copied().unwrap_or(1);

            if dim_a == dim_b {
                result_dims.push(dim_a);
            } else if dim_a == 1 {
                result_dims.push(dim_b);
            } else if dim_b == 1 {
                result_dims.push(dim_a);
            } else {
                return Err(CoreError::CannotBroadcast {
                    shape_a: self.clone(),
                    shape_b: other.clone(),
                });
            }
        }

        Ok(Shape::new(&result_dims))
    }
}

