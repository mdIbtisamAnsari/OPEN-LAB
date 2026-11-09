# OPEN-LAB

**OPEN-LAB** is an independent, clean-room, production-grade Rust implementation of the MATLAB numerical computing environment.

---

## 1. Architectural Principles

1. **Zero Proprietary Code:** Built entirely from first principles using authorized academic publications (McGill McLAB, ACM HOPL IV), official public file format specifications (Level 5 MAT-file specification), and IEEE standards.
2. **Memory Safety & Performance:** Built in modern Rust with column-major Fortran contiguous layouts, Copy-on-Write (`Arc<Vec<T>>`) memory isolation, and zero-cost abstractions.
3. **Parity with MATLAB Semantics:** 1-based indexing, singleton expansion (broadcasting), complex number arithmetic, multi-dimensional array slicing, dynamic type promotion, and MATLAB command window formatting.

---

## 2. Workspace Organization

```
OPENLAB/
├── Cargo.toml                     # Workspace root manifest
├── ARCHITECTURE.md                # System architectural specification
├── CLEAN_ROOM_SPEC.md             # Legal compliance & clean-room boundaries
├── CORE_AUDIT_REPORT.md           # Audit & differential testing report
├── MATH_SUBSYSTEM_REPORT.md       # Math subsystem & benchmark report
├── RE_TOOLING_REPORT.md           # Toolchain & REA MCP inspection report
├── ROADMAP.md                     # Milestone tracking & progress ledger
├── SUBSYSTEM_INVENTORY.md         # Comprehensive MATLAB subsystem inventory
└── crates/
    ├── openlab-core/              # Subsystem 1: Core multi-dimensional array engine
    ├── openlab-math/              # Subsystem 2: Linear algebra, decompositions & solvers
    └── openlab-syntax/            # Subsystem 3: Pratt expression parser, AST & lexer
```

---

## 3. Subsystem 1: `openlab-core`

`openlab-core` provides the foundational array and memory infrastructure:
* **`Shape`:** Guarantees $\ge 2$ dimensions, normalizes trailing singleton dimensions, calculates column-major strides and 0-based/1-based coordinate conversions.
* **`MxArray`:** High-performance dynamic array struct with Copy-on-Write reference counting.
* **`MxData`:** Dynamic variant handling real doubles, complex doubles, singles, logicals, integers, characters, and cell arrays.
* **`Complex64`:** IEEE 754 complex arithmetic supporting conjugate, magnitude, phase, exponential, and square root.
* **Operations:** 2D transpose (`.'`), Hermitian transpose (`'`), submatrix extraction, element-wise arithmetic with multi-dimensional broadcasting, reductions (`sum`, `prod`, `mean`, `min`, `max`), and MATLAB-style matrix display.

---

## 4. Verification & Testing

To run the complete test suite:

```bash
cargo test
```

All tests pass with zero warnings, zero memory leaks, and complete invariant guarantees.

# OPEN-LAB
# OPEN-LAB
