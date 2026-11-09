pub mod array;
pub mod complex;
pub mod data;
pub mod error;
pub mod format;
pub mod shape;

pub use array::MxArray;
pub use complex::Complex64;
pub use data::MxData;
pub use error::CoreError;
pub use format::{DisplayFormat, MatrixFormatter};
pub use shape::Shape;
