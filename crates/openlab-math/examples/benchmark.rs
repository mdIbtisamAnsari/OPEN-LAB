use openlab_core::MxArray;
use openlab_math::{cholesky, lu, magic, matmul, mldivide, pascal, qr, svd};
use std::time::Instant;

fn main() {
    println!("============================================================");
    println!("             OPEN-LAB Math Subsystem Benchmarks             ");
    println!("============================================================");

    // 1. GEMM Benchmarks
    for &n in &[50, 100, 200] {
        let a = magic(n).unwrap();
        let b = pascal(n).unwrap();

        let start = Instant::now();
        let iters = if n <= 100 { 20 } else { 5 };
        for _ in 0..iters {
            let _ = matmul(&a, &b).unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64() / (iters as f64);
        let gflops = (2.0 * (n as f64).powi(3) * 1e-9) / elapsed;
        println!(
            "GEMM ({}x{}): {:8.3} ms  ({:6.2} GFLOPS)",
            n,
            n,
            elapsed * 1000.0,
            gflops
        );
    }

    // 2. LU Decomposition Benchmarks
    for &n in &[50, 100, 200] {
        let a = magic(n).unwrap();
        let iters = if n <= 100 { 20 } else { 5 };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = lu::decompose(&a).unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64() / (iters as f64);
        println!("LU   ({}x{}): {:8.3} ms", n, n, elapsed * 1000.0);
    }

    // 3. Cholesky Decomposition Benchmarks (using well-conditioned SPD matrix A = B^T * B + n*I)
    for &n in &[50, 100, 200] {
        let b = magic(n).unwrap();
        let bt = b.transpose().unwrap();
        let mut spd = matmul(&bt, &b).unwrap();
        // Add n * I to diagonal to ensure strong positive definiteness
        for i in 1..=n {
            let val = spd.get_2d(i, i).unwrap();
            let _ = spd.set_2d(i, i, val + (n as f64) * 1000.0);
        }

        let iters = if n <= 100 { 20 } else { 5 };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = cholesky::decompose(&spd).unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64() / (iters as f64);
        println!("CHOL ({}x{}): {:8.3} ms", n, n, elapsed * 1000.0);
    }

    // 4. QR Decomposition Benchmarks
    for &n in &[50, 100, 150] {
        let a = magic(n).unwrap();
        let iters = if n <= 100 { 10 } else { 3 };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = qr::decompose(&a, true).unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64() / (iters as f64);
        println!("QR   ({}x{}): {:8.3} ms", n, n, elapsed * 1000.0);
    }

    // 5. SVD Benchmarks
    for &n in &[20, 50, 80] {
        let a = magic(n).unwrap();
        let iters = if n <= 50 { 5 } else { 2 };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = svd::decompose(&a, true).unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64() / (iters as f64);
        println!("SVD  ({}x{}): {:8.3} ms", n, n, elapsed * 1000.0);
    }

    // 6. Linear Solve (mldivide A \ b)
    for &n in &[50, 100, 200] {
        let a = magic(n).unwrap();
        let b = MxArray::ones(n, 1);
        let iters = if n <= 100 { 20 } else { 5 };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = mldivide(&a, &b).unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64() / (iters as f64);
        println!("SOLVE({}x{}): {:8.3} ms", n, n, elapsed * 1000.0);
    }

    println!("============================================================");
}
