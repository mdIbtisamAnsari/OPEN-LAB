use openlab_core::{CoreError, MxArray, Shape};
use openlab_syntax::Expr;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Matrix(MxArray),
    FunctionHandle(String),
    AnonymousFn {
        params: Vec<String>,
        body: Expr,
        closure_env: HashMap<String, Value>,
    },
    Cell(Vec<Vec<Value>>),
}

impl Value {
    pub fn from_f64(v: f64) -> Self {
        Value::Matrix(MxArray::from_scalar(v))
    }

    pub fn from_bool(v: bool) -> Self {
        Value::Matrix(MxArray::from_scalar_bool(v))
    }

    pub fn from_string(s: &str) -> Self {
        Value::Matrix(MxArray::from_string(s))
    }

    pub fn to_matrix(&self) -> Result<MxArray, CoreError> {
        match self {
            Value::Matrix(mat) => Ok(mat.clone()),
            _ => Err(CoreError::IncompatibleTypes {
                expected: "matrix",
                actual: "non-matrix value",
            }),
        }
    }

    pub fn to_scalar(&self) -> Result<f64, CoreError> {
        match self {
            Value::Matrix(mat) => mat.to_scalar(),
            _ => Err(CoreError::IncompatibleTypes {
                expected: "scalar",
                actual: "non-matrix value",
            }),
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Matrix(mat) => {
                if mat.is_empty() {
                    false
                } else if mat.is_scalar() {
                    mat.to_scalar().unwrap_or(0.0) != 0.0
                } else {
                    // In MATLAB, if condition on matrix is true iff ALL elements are non-zero!
                    for i in 0..mat.numel() {
                        if mat.get_linear_0based(i).unwrap_or(0.0) == 0.0 {
                            return false;
                        }
                    }
                    true
                }
            }
            _ => true,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Matrix(m) => write!(f, "{}", m),
            Value::FunctionHandle(h) => write!(f, "    @{}", h),
            Value::AnonymousFn { params, .. } => {
                write!(f, "    @({}) ...", params.join(", "))
            }
            Value::Cell(rows) => {
                writeln!(f, "    {}x{} cell array", rows.len(), rows.first().map(|r| r.len()).unwrap_or(0))
            }
        }
    }
}

