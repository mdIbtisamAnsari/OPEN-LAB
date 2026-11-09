use openlab_core::CoreError;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum VmError {
    UndefinedVariable(String),
    UndefinedFunction(String),
    Core(CoreError),
    Syntax(String),
    IndexOutOfBounds(String),
    TypeError(String),
    ExecutionError(String),
    NotEnoughInputs { func: String, expected: usize, got: usize },
    TooManyOutputs { func: String, max: usize, got: usize },
}

impl fmt::Display for VmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VmError::UndefinedVariable(v) => write!(f, "Unrecognized function or variable '{}'.", v),
            VmError::UndefinedFunction(fn_name) => {
                write!(f, "Undefined function or variable '{}'.", fn_name)
            }
            VmError::Core(err) => write!(f, "{}", err),
            VmError::Syntax(msg) => write!(f, "Syntax error: {}", msg),
            VmError::IndexOutOfBounds(msg) => write!(f, "Index out of bounds: {}", msg),
            VmError::TypeError(msg) => write!(f, "Type error: {}", msg),
            VmError::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
            VmError::NotEnoughInputs { func, expected, got } => {
                write!(
                    f,
                    "Not enough input arguments for function '{}': expected {}, got {}.",
                    func, expected, got
                )
            }
            VmError::TooManyOutputs { func, max, got } => {
                write!(
                    f,
                    "Too many output arguments for function '{}': max {}, requested {}.",
                    func, max, got
                )
            }
        }
    }
}

impl std::error::Error for VmError {}

impl From<CoreError> for VmError {
    fn from(err: CoreError) -> Self {
        VmError::Core(err)
    }
}

impl From<String> for VmError {
    fn from(msg: String) -> Self {
        VmError::ExecutionError(msg)
    }
}

