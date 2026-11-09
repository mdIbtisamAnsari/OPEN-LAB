use crate::cholesky::CholeskyDecomposition;
use crate::lu::LuDecomposition;
use crate::qr::QrDecomposition;
use crate::svd::SvdDecomposition;
use openlab_core::{CoreError, MxArray};

/// Pluggable mathematical execution backend.
///
/// Designed to support future CPU (SIMD/threaded), BLAS/LAPACK,
/// OpenCL/CUDA, or WebGPU backends without changing the top-level API.
pub trait MathBackend: Send + Sync {
    fn name(&self) -> &'static str;

    /// Dense matrix multiplication: C = A * B
    fn gemm(&self, a: &MxArray, b: &MxArray) -> Result<MxArray, CoreError>;

    /// LU decomposition with partial pivoting: P * A = L * U
    fn lu(&self, a: &MxArray) -> Result<LuDecomposition, CoreError>;

    /// Cholesky decomposition for symmetric positive definite matrices: A = L * L^T
    fn cholesky(&self, a: &MxArray) -> Result<CholeskyDecomposition, CoreError>;

    /// QR decomposition via Householder reflectors: A = Q * R
    fn qr(&self, a: &MxArray, economy: bool) -> Result<QrDecomposition, CoreError>;

    /// Singular Value Decomposition: A = U * S * V^T
    fn svd(&self, a: &MxArray, economy: bool) -> Result<SvdDecomposition, CoreError>;

    /// Polyalgorithmic linear system solve: A \ b (mldivide)
    fn solve(&self, a: &MxArray, b: &MxArray) -> Result<MxArray, CoreError>;

    /// Matrix inverse: A^(-1)
    fn inv(&self, a: &MxArray) -> Result<MxArray, CoreError>;
}

/// Pure-Rust reference CPU backend.
///
/// Features:
/// - Cache-optimized column-major GEMM
/// - Unconditionally stable partial-pivoting LU
/// - Numerically stable Householder QR
/// - Golub-Kahan-Reinsch Bidiagonal SVD
/// - 100% clean-room with zero external dependencies
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl MathBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "CPU Reference Backend (Pure Rust Clean-Room)"
    }

    fn gemm(&self, a: &MxArray, b: &MxArray) -> Result<MxArray, CoreError> {
        crate::gemm::matmul(a, b)
    }

    fn lu(&self, a: &MxArray) -> Result<LuDecomposition, CoreError> {
        crate::lu::decompose(a)
    }

    fn cholesky(&self, a: &MxArray) -> Result<CholeskyDecomposition, CoreError> {
        crate::cholesky::decompose(a)
    }

    fn qr(&self, a: &MxArray, economy: bool) -> Result<QrDecomposition, CoreError> {
        crate::qr::decompose(a, economy)
    }

    fn svd(&self, a: &MxArray, economy: bool) -> Result<SvdDecomposition, CoreError> {
        crate::svd::decompose(a, economy)
    }

    fn solve(&self, a: &MxArray, b: &MxArray) -> Result<MxArray, CoreError> {
        crate::solve::mldivide(a, b)
    }

    fn inv(&self, a: &MxArray) -> Result<MxArray, CoreError> {
        crate::solve::inv(a)
    }
}

static DEFAULT_BACKEND: CpuBackend = CpuBackend;

/// Returns the currently active mathematical backend.
pub fn default_backend() -> &'static dyn MathBackend {
    &DEFAULT_BACKEND
}

