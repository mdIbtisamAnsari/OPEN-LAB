pub mod backend;
pub mod cholesky;
pub mod gemm;
pub mod lu;
pub mod qr;
pub mod solve;
pub mod svd;
pub mod test_matrices;

pub use backend::{default_backend, CpuBackend, MathBackend};
pub use cholesky::CholeskyDecomposition;
pub use gemm::{matmul, matpow, matvec};
pub use lu::LuDecomposition;
pub use qr::QrDecomposition;
pub use solve::{cond, det, inv, mldivide, mrdivide, norm, pinv, rank, trace, NormType};
pub use svd::SvdDecomposition;
pub use test_matrices::{hilb, invhilb, magic, moler, pascal};
