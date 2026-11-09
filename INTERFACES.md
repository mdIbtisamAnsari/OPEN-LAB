# OPEN-LAB System Interfaces Specification

**Subsystems:** `openlab-core`, `openlab-math`, `openlab-syntax`, `openlab-stdlib`, `openlab-vm`  
**Document Version:** 1.0.0  
**Status:** Canonical Interface Definition  

---

## 1. Architectural Interaction Graph

```
                            +--------------------+
                            |   openlab-syntax   |
                            | (AST, Lexer, Span) |
                            +---------+----------+
                                      |
                            +---------v----------+
                            |     openlab-vm     |  <---+ (Interpreter, Scopes)
                            +---------+----------+      |
                                      |                 |
                   +------------------+-----------------+
                   |                                    |
         +---------v----------+               +---------v----------+
         |   openlab-stdlib   |               |    openlab-math    |
         | (Builtin Registry) |-------------->| (BLAS/LAPACK Core) |
         +---------+----------+               +---------+----------+
                   |                                    |
                   +------------------+-----------------+
                                      |
                            +---------v----------+
                            |    openlab-core    |
                            | (MxArray, Storage) |
                            +--------------------+
```

---

## 2. Cross-Crate Data Contracts

### 2.1 `openlab-core` Contract
* **Primary Currency:** `MxArray` (representing scalars, vectors, 2D/ND matrices, logicals, characters, cells, and complex arrays).
* **Memory Invariant:** Column-major Fortran contiguous layouts with Copy-on-Write reference counting (`Arc<Vec<T>>`).
* **Error Model:** `CoreError` defining boundary diagnostics, dimension mismatches, and mathematical invalidities.

### 2.2 `openlab-math` Contract
* **Backend Trait:** `MathBackend` supplying pure-Rust and accelerator-ready algorithms for GEMM, LU, Cholesky, QR, SVD, and linear solvers.
* **Solvers:** Exposes high-level numerical solvers (`mldivide`, `mrdivide`, `inv`, `det`, `rank`, `cond`, `norm`, `pinv`) operating purely on `MxArray`.

### 2.3 `openlab-syntax` Contract
* **Output:** `Program` and `Statement` Abstract Syntax Trees.
* **Contextual Features:** Semicolon output suppression tags (`print_result: bool`), Pratt-parsed operator precedence, matrix/cell literals, and multi-variable assignment targets (`AssignTarget::MultiVar`).

### 2.4 `openlab-stdlib` Contract
* **Function Signature:** Standardized function handler:
  ```rust
  pub type BuiltinFn = fn(&[MxArray], usize /* nargout */) -> Result<Vec<MxArray>, CoreError>;
  ```
* **Registry:** `BuiltinRegistry` provides $O(1)$ dispatch of built-in names to mathematical, structural, and string routines.

### 2.5 `openlab-vm` Contract
* **Execution Model:** AST Tree-walk interpreter with stack-allocated call frames (`CallFrame`), lexical workspace scopes (`Workspace`), and persistent variable storage.
* **Variable vs Function Disambiguation:** Resolves identifiers by checking:
  1. Local frame workspace
  2. Persistent variables
  3. Global variables
  4. Built-in functions (`BuiltinRegistry`)
  5. User-defined functions

