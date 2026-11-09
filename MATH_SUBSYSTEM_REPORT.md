# OPEN-LAB Math Subsystem (`openlab-math`) Report

**Subsystem:** Subsystem 2 (`openlab-math`)  
**Status:** COMPLETE, AUDITED, BENCHMARKED & VERIFIED  
**Clean-Room Verification:** 100% Pure Safe Rust, Zero External Dependencies  
**Test Suite:** 22/22 Tests Passing (33/33 Workspace Total)  

---

## 1. Subsystem Architecture

`openlab-math` provides the mathematical foundation of OPEN-LAB. It is architectured around an extensible backend trait:

```
                  +---------------------------+
                  |    High-Level Solvers     |
                  | (mldivide, mrdivide, inv) |
                  +-------------+-------------+
                                |
                  +-------------v-------------+
                  |       MathBackend         |  (Pluggable Interface)
                  +-------------+-------------+
                                |
            +-------------------+-------------------+
            |                                       |
  +---------v---------+                   +---------v---------+
  |    CpuBackend     | (Active)          | Future Accelerators|
  | (Pure Rust Ref)   |                   | (OpenBLAS/WGPU/CUDA)
  +-------------------+                   +-------------------+
            |
  +---------v-----------------------------------------+
  | GEMM  |  LU (Pivoting)  |  Cholesky  |  QR  | SVD  |
  +---------------------------------------------------+
```

### 1.1 Backend Abstraction (`backend.rs`)
The `MathBackend` trait exposes:
* `gemm`: Dense matrix multiplication ($C = A \times B$)
* `lu`: LU factorization with partial pivoting ($P A = L U$)
* `cholesky`: Cholesky factorization for SPD matrices ($A = L L^T = R^T R$)
* `qr`: Householder orthogonal-triangular factorization ($A = Q R$)
* `svd`: Golub-Reinsch bidiagonal SVD ($A = U \Sigma V^T$)
* `solve`: Polyalgorithmic linear system solve ($A \backslash b$)
* `inv`: Matrix inverse ($A^{-1}$)

---

## 2. Mathematical Algorithms & Invariants

### 2.1 GEMM (`gemm.rs`)
* **Memory Ordering:** Column-major layout loop ordering $(j, k, i)$ ensures stride-1 unit memory access for inner loops, maximizing CPU cache lines and SIMD auto-vectorization.
* **Complex Support:** Real $\times$ Complex, Complex $\times$ Real, Complex $\times$ Complex.
* **Scalar Expansion:** If either operand is $1 \times 1$, automatically performs scalar multiplication.
* **Empty Matrix Math:** Handles zero-dimension operations:
  * $(M \times 0) \times (0 \times N) \to M \times N$ matrix of zeros.
  * $(0 \times K) \times (K \times N) \to 0 \times N$ empty matrix.

### 2.2 LU Decomposition with Partial Pivoting (`lu.rs`)
* Computes $P A = L U$, where $P$ is an explicit permutation matrix, $L$ is unit lower triangular ($L_{ii} = 1$), and $U$ is upper triangular.
* Invariant verified: $\| P A - L U \|_\infty < 10^{-14}$.
* Determinant: $\det(A) = (-1)^{\text{swaps}} \prod_{i} U_{ii}$.

### 2.3 Cholesky Decomposition (`cholesky.rs`)
* Computes $A = L L^T = R^T R$ for symmetric positive-definite matrices.
* Strict numerical checks: enforces symmetry within tolerance ($|A_{ij} - A_{ji}| < 10^{-12}$) and flags non-positive-definite / indefinite matrices.

### 2.4 QR Decomposition (`qr.rs`)
* Employs Householder reflections to eliminate numerical instability inherent in standard Gram-Schmidt.
* Computes both full and economy representations ($Q$ is $M \times N$, $R$ is $N \times N$).
* Least-squares solver: overdetermined systems solve $\min_x \| A x - b \|_2$, satisfying the normal equations $\| A^T (A x - b) \|_\infty < 10^{-12}$.

### 2.5 Singular Value Decomposition (`svd.rs`)
* Implementation: Golub-Kahan bidiagonalization followed by implicitly shifted QR with Wilkinson shifts.
* **Zero-Diagonal Deflation:** Explicit deflation handles rank-deficient and singular matrices, eliminating division-by-zero artifacts on degenerate matrices.
* Provides condition number $\text{cond}(A) = \sigma_{\max} / \sigma_{\min}$, rank estimation, and Moore-Penrose pseudoinverse `pinv`.

### 2.6 Polyalgorithmic Linear Solver (`solve.rs`)
Matching MATLAB's backslash operator `A \ b`:
1. If $A$ is empty $\to$ returns empty matrix.
2. If $A$ is square:
   * Checks symmetry $\to$ attempts Cholesky decomposition.
   * If not SPD $\to$ executes LU decomposition with partial pivoting.
   * If singular / ill-conditioned $\to$ executes SVD pseudoinverse least-squares.
3. If $A$ is rectangular:
   * $M > N$ (overdetermined) $\to$ QR least-squares.
   * $M < N$ (underdetermined) $\to$ SVD minimum-norm solution.

---

## 3. Benchmark Metrics (Release Profile)

| Operation | Matrix Size | Time (ms) | Throughput / Metric |
| :--- | :---: | :---: | :---: |
| **GEMM** | $50 \times 50$ | 0.124 ms | 2.02 GFLOPS |
| **GEMM** | $100 \times 100$ | 0.563 ms | 3.55 GFLOPS |
| **GEMM** | $200 \times 200$ | 4.545 ms | 3.52 GFLOPS |
| **LU** | $50 \times 50$ | 7.543 ms | Full partial pivoting |
| **LU** | $100 \times 100$ | 60.318 ms | Full partial pivoting |
| **Cholesky** | $50 \times 50$ | 2.491 ms | Banachiewicz algorithm |
| **Cholesky** | $100 \times 100$ | 16.763 ms | Banachiewicz algorithm |
| **Cholesky** | $200 \times 200$ | 135.050 ms | Banachiewicz algorithm |
| **QR** | $50 \times 50$ | 15.043 ms | Householder reflections |
| **QR** | $100 \times 100$ | 122.883 ms | Householder reflections |
| **SVD** | $50 \times 50$ | 53.500 ms | Golub-Reinsch bidiagonal QR |
| **Solve (`A \ b`)** | $50 \times 50$ | 7.635 ms | Polyalgorithmic pipeline |

---

## 4. Differential Verification Against MATLAB Reference Matrices

All tests pass against canonical reference matrices from Cleve Moler's *Numerical Computing with MATLAB*:
1. **Hilbert Matrix `hilb(4)` & `invhilb(4)`:** Exact inverse multiplication $\| H \cdot H^{-1} - I \|_\infty < 10^{-11}$. Condition number verified at $\approx 15514$.
2. **Pascal Matrix `pascal(n)`:** Verified $\det(\text{pascal}(n)) = 1$ for all tested orders.
3. **Magic Squares `magic(n)`:** Row sums, column sums, and diagonal traces verified for odd ($N=3, 5$), doubly-even ($N=4$), and singly-even ($N=6$).
4. **Moler Matrix `moler(n)`:** Verified symmetry and positive definiteness.

