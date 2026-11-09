use openlab_core::{Complex64, CoreError, MxArray, MxData};

/// Applies an element-wise real-to-real unary function over an MxArray.
pub fn apply_unary_f64<F>(arr: &MxArray, f: F) -> Result<MxArray, CoreError>
where
    F: Fn(f64) -> f64,
{
    match arr.data() {
        MxData::Double(v) => {
            let out: Vec<f64> = v.iter().map(|&x| f(x)).collect();
            MxArray::from_f64_vec(arr.shape().clone(), out)
        }
        MxData::Logical(v) => {
            let out: Vec<f64> = v.iter().map(|&x| f(if x { 1.0 } else { 0.0 })).collect();
            MxArray::from_f64_vec(arr.shape().clone(), out)
        }
        _ => Err(CoreError::IncompatibleTypes {
            expected: "double or logical",
            actual: arr.class_name(),
        }),
    }
}

// -----------------------------------------------------------------------------
// Trigonometric Functions
// -----------------------------------------------------------------------------

pub fn sin(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, f64::sin)
}

pub fn cos(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, f64::cos)
}

pub fn tan(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, f64::tan)
}

pub fn asin(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, f64::asin)
}

pub fn acos(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, f64::acos)
}

pub fn atan(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, f64::atan)
}

pub fn atan2(y: &MxArray, x: &MxArray) -> Result<MxArray, CoreError> {
    let out_shape = y.shape().broadcast_shape(x.shape())?;
    let n = out_shape.numel();
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let y_val = y.get_linear_0based(i % y.numel())?;
        let x_val = x.get_linear_0based(i % x.numel())?;
        out.push(y_val.atan2(x_val));
    }

    MxArray::from_f64_vec(out_shape, out)
}

// Degrees versions
pub fn sind(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, |x| (x * std::f64::consts::PI / 180.0).sin())
}

pub fn cosd(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, |x| (x * std::f64::consts::PI / 180.0).cos())
}

pub fn tand(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, |x| (x * std::f64::consts::PI / 180.0).tan())
}

// -----------------------------------------------------------------------------
// Hyperbolic Functions
// -----------------------------------------------------------------------------

pub fn sinh(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, f64::sinh)
}

pub fn cosh(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, f64::cosh)
}

pub fn tanh(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, f64::tanh)
}

// -----------------------------------------------------------------------------
// Exponential, Logarithmic & Powers
// -----------------------------------------------------------------------------

pub fn exp(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, f64::exp)
}

pub fn log(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, f64::ln)
}

pub fn log10(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, f64::log10)
}

pub fn log2(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, f64::log2)
}

pub fn sqrt(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, f64::sqrt)
}

pub fn cbrt(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, f64::cbrt)
}

// -----------------------------------------------------------------------------
// Rounding & Truncation
// -----------------------------------------------------------------------------

pub fn floor(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, f64::floor)
}

pub fn ceil(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, f64::ceil)
}

pub fn round(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, f64::round)
}

/// fix: round towards zero (truncate)
pub fn fix(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, f64::trunc)
}

pub fn abs(arr: &MxArray) -> Result<MxArray, CoreError> {
    match arr.data() {
        MxData::Double(v) => {
            let out: Vec<f64> = v.iter().map(|&x| x.abs()).collect();
            MxArray::from_f64_vec(arr.shape().clone(), out)
        }
        MxData::ComplexDouble(v) => {
            let out: Vec<f64> = v.iter().map(|&c| c.abs()).collect();
            MxArray::from_f64_vec(arr.shape().clone(), out)
        }
        _ => Err(CoreError::IncompatibleTypes {
            expected: "numeric double",
            actual: arr.class_name(),
        }),
    }
}

pub fn sign(arr: &MxArray) -> Result<MxArray, CoreError> {
    apply_unary_f64(arr, |x| {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    })
}

pub fn rem(x: &MxArray, y: &MxArray) -> Result<MxArray, CoreError> {
    let out_shape = x.shape().broadcast_shape(y.shape())?;
    let n = out_shape.numel();
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let vx = x.get_linear_0based(i % x.numel())?;
        let vy = y.get_linear_0based(i % y.numel())?;
        out.push(vx % vy);
    }

    MxArray::from_f64_vec(out_shape, out)
}

pub fn mod_op(x: &MxArray, y: &MxArray) -> Result<MxArray, CoreError> {
    let out_shape = x.shape().broadcast_shape(y.shape())?;
    let n = out_shape.numel();
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let vx = x.get_linear_0based(i % x.numel())?;
        let vy = y.get_linear_0based(i % y.numel())?;
        let m = ((vx % vy) + vy) % vy;
        out.push(m);
    }

    MxArray::from_f64_vec(out_shape, out)
}

// -----------------------------------------------------------------------------
// Complex Functions
// -----------------------------------------------------------------------------

pub fn real(arr: &MxArray) -> Result<MxArray, CoreError> {
    match arr.data() {
        MxData::Double(_) => Ok(arr.clone()),
        MxData::ComplexDouble(v) => {
            let out: Vec<f64> = v.iter().map(|&c| c.re).collect();
            MxArray::from_f64_vec(arr.shape().clone(), out)
        }
        _ => Err(CoreError::IncompatibleTypes {
            expected: "numeric double",
            actual: arr.class_name(),
        }),
    }
}

pub fn imag(arr: &MxArray) -> Result<MxArray, CoreError> {
    match arr.data() {
        MxData::Double(_) => Ok(MxArray::zeros_nd(arr.shape().clone())),
        MxData::ComplexDouble(v) => {
            let out: Vec<f64> = v.iter().map(|&c| c.im).collect();
            MxArray::from_f64_vec(arr.shape().clone(), out)
        }
        _ => Err(CoreError::IncompatibleTypes {
            expected: "numeric double",
            actual: arr.class_name(),
        }),
    }
}

pub fn conj(arr: &MxArray) -> Result<MxArray, CoreError> {
    match arr.data() {
        MxData::Double(_) => Ok(arr.clone()),
        MxData::ComplexDouble(v) => {
            let out: Vec<Complex64> = v.iter().map(|&c| c.conj()).collect();
            MxArray::from_complex_vec(arr.shape().clone(), out)
        }
        _ => Err(CoreError::IncompatibleTypes {
            expected: "numeric double",
            actual: arr.class_name(),
        }),
    }
}

pub fn angle(arr: &MxArray) -> Result<MxArray, CoreError> {
    match arr.data() {
        MxData::Double(v) => {
            let out: Vec<f64> = v
                .iter()
                .map(|&x| if x >= 0.0 { 0.0 } else { std::f64::consts::PI })
                .collect();
            MxArray::from_f64_vec(arr.shape().clone(), out)
        }
        MxData::ComplexDouble(v) => {
            let out: Vec<f64> = v.iter().map(|&c| c.arg()).collect();
            MxArray::from_f64_vec(arr.shape().clone(), out)
        }
        _ => Err(CoreError::IncompatibleTypes {
            expected: "numeric double",
            actual: arr.class_name(),
        }),
    }
}
