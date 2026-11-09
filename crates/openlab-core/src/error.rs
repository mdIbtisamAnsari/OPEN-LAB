use crate::shape::Shape;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum CoreError {
    IndexOutOfBounds {
        dim: usize,
        index: usize,
        bound: usize,
    },
    LinearIndexOutOfBounds {
        index: usize,
        bound: usize,
    },
    DimensionMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    InvalidShape(String),
    IncompatibleTypes {
        expected: &'static str,
        actual: &'static str,
    },
    InvalidReshape {
        from: usize,
        to: usize,
    },
    EmptyArray,
    NonSquareMatrix {
        rows: usize,
        cols: usize,
    },
    CannotBroadcast {
        shape_a: Shape,
        shape_b: Shape,
    },
    DivisionByZero,
    InvalidOperation(String),
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreError::IndexOutOfBounds { dim, index, bound } => {
                write!(
                    f,
                    "Index in position {} exceeds array bounds (index {} exceeds bound {}).",
                    dim, index, bound
                )
            }
            CoreError::LinearIndexOutOfBounds { index, bound } => {
                write!(
                    f,
                    "Index exceeds array bounds (index {} exceeds total elements {}).",
                    index, bound
                )
            }
            CoreError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Dimension mismatch: expected {:?}, got {:?}.",
                    expected, actual
                )
            }
            CoreError::InvalidShape(msg) => write!(f, "Invalid shape: {}", msg),
            CoreError::IncompatibleTypes { expected, actual } => {
                write!(
                    f,
                    "Incompatible array types: expected {}, got {}.",
                    expected, actual
                )
            }
            CoreError::InvalidReshape { from, to } => {
                write!(
                    f,
                    "To RESHAPE the number of elements must not change (from {} to {}).",
                    from, to
                )
            }
            CoreError::EmptyArray => write!(f, "Operation not supported on empty array."),
            CoreError::NonSquareMatrix { rows, cols } => {
                write!(
                    f,
                    "Matrix must be square, but is {}x{}.",
                    rows, cols
                )
            }
            CoreError::CannotBroadcast { shape_a, shape_b } => {
                write!(
                    f,
                    "Arrays have incompatible sizes for this operation: {:?} and {:?}.",
                    shape_a.dims(),
                    shape_b.dims()
                )
            }
            CoreError::DivisionByZero => write!(f, "Division by zero encountered."),
            CoreError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
        }
    }
}

impl std::error::Error for CoreError {}

