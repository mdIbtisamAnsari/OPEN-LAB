# OPEN-LAB Rust Architectural Specification

**Project:** OPEN-LAB  
**Architecture Version:** 1.0.0  
**Language:** Rust (2021/2024 edition, stable)  
**Target:** High-performance, memory-safe numerical computing engine

---

## 1. System Overview

OPEN-LAB is architected as a modular, high-performance Rust workspace. The design favors clean separation of concerns, zero-cost abstractions, data cache locality via column-major contiguous memory buffers, and safe concurrency.

```
                      +-------------------+
                      |    openlab-cli    |  (Executable Binary)
                      +---------+---------+
                                |
                      +---------v---------+
                      |   openlab-repl    |  (Interactive Environment)
                      +---------+---------+
                                |
             +------------------+------------------+
             |                                     |
   +---------v---------+                 +---------v---------+
   |   openlab-matio   |                 |    openlab-vm     | (Bytecode & Interpreter)
   | (MAT-File v5/v4)  |                 +----+---------+----+
   +---------+---------+                      |         |
             |                   +------------+         +------------+
             |                   |                                   |
             |         +---------v---------+               +---------v---------+
             |         |  openlab-syntax   |               |  openlab-stdlib   |
             |         |  (Lexer & Parser) |               |  (Builtin Library)|
             |         +---------+---------+               +---------+---------+
             |                   |                                   |
             +-------------------+-----------------+                 |
                                                   |                 |
                                         +---------v---------+       |
                                         |   openlab-math    |<------+
                                         | (Linear Algebra)  |
                                         +---------+---------+
                                                   |
                                         +---------v---------+
                                         |   openlab-core    |
                                         | (Array & Types)   |
                                         +-------------------+
```

---

## 2. Workspace Crate Architecture

| Crate | Responsibility | Dependencies |
| :--- | :--- | :--- |
| **`openlab-core`** | Fundamental types, `MxArray`, `Shape`, column-major storage, slices, COW storage, complex numbers. | None (pure Rust standard library) |
| **`openlab-math`** | Linear algebra, BLAS/LAPACK bindings, matrix factorizations (LU, Cholesky, QR, SVD), element-wise kernels. | `openlab-core` |
| **`openlab-syntax`** | Lexical scanner, token stream, AST, Pratt parser for expressions, source span diagnostics. | `openlab-core` |
| **`openlab-stdlib`** | Standard built-in numerical functions, matrix constructors, array transformations. | `openlab-core`, `openlab-math` |
| **`openlab-vm`** | Scope manager, workspace environments, bytecode/AST evaluation engine, function call frame. | `openlab-core`, `openlab-syntax`, `openlab-stdlib`, `openlab-math` |
| **`openlab-matio`** | MAT-file (Level 5 / Level 4) binary reader and writer with zlib deflation. | `openlab-core` |
| **`openlab-repl`** | Read-Eval-Print-Loop with command line editing, syntax highlighting, workspace inspector. | `openlab-vm`, `openlab-core` |
| **`openlab-cli`** | Top-level command line entry point (`openlab`), batch script executor, flag parser. | `openlab-repl`, `openlab-vm` |

---

## 3. Core Memory Model (`openlab-core`)

### 3.1 Memory Layout: Fortran Column-Major
In MATLAB, multi-dimensional arrays are stored contiguously in column-major order. For a matrix of dimension $M \times N$:
$$\text{Linear Offset}(i, j) = i + j \times M \quad (0 \le i < M, 0 \le j < N)$$

For general N-dimensional arrays with shape $(d_0, d_1, \dots, d_{k-1})$:
$$\text{Offset}(i_0, i_1, \dots, i_{k-1}) = \sum_{r=0}^{k-1} i_r \times \prod_{m=0}^{r-1} d_m$$

### 3.2 Dynamic Type Representation: `MxArray`
`MxArray` represents any MATLAB entity. To guarantee zero allocation overhead for common operations and reference-counted copy-on-write sharing, `MxArray` is implemented as:

```rust
pub struct MxArray {
    shape: Shape,
    data: MxData,
}

pub enum MxData {
    Double(Arc<Vec<f64>>),
    ComplexDouble(Arc<Vec<Complex64>>),
    Single(Arc<Vec<f32>>),
    ComplexSingle(Arc<Vec<Complex32>>),
    Int8(Arc<Vec<i8>>),
    Uint8(Arc<Vec<u8>>),
    Int16(Arc<Vec<i16>>),
    Uint16(Arc<Vec<u16>>),
    Int32(Arc<Vec<i32>>),
    Uint32(Arc<Vec<u32>>),
    Int64(Arc<Vec<i64>>),
    Uint64(Arc<Vec<u64>>),
    Logical(Arc<Vec<bool>>),
    Char(Arc<Vec<char>>),
    Cell(Arc<Vec<MxArray>>),
    Struct(Arc<StructArray>),
    Sparse(Arc<SparseMatrix>),
}
```

### 3.3 Copy-On-Write (COW) Semantics
All buffer variants use `Arc<Vec<T>>`. Slicing and variable copies are $O(1)$ pointer increments. If an element or slice is modified, `Arc::make_mut` triggers an allocation only when the reference count is greater than 1, exactly matching MATLAB's COW runtime semantics.

### 3.4 1-Based Indexing vs 0-Based Internal Storage
User-facing syntax is 1-based (MATLAB standard: `A(1, 1)` is the first element). The parser and VM translate 1-based index operations to 0-based internal coordinates, with bounds checking throwing MATLAB-compatible error messages:
`Index in position 1 exceeds array bounds (must not exceed M).`

