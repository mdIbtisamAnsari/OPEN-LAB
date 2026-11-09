pub mod constructors;
pub mod elmath;
pub mod io;
pub mod linear_algebra;
pub mod manipulation;
pub mod reduction;
pub mod registry;

pub use constructors::{diag, eye, linspace, logspace, meshgrid, ones, rand, randn, zeros};
pub use elmath::*;
pub use io::{disp, fprintf, sprintf};
pub use linear_algebra::eig;
pub use manipulation::*;
pub use reduction::*;
pub use registry::{BuiltinFn, BuiltinRegistry};

