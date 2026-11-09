use crate::{constructors, elmath, io, linear_algebra, manipulation, reduction};
use openlab_core::{CoreError, MxArray, Shape};
use std::collections::HashMap;

pub type BuiltinFn = fn(&[MxArray], usize) -> Result<Vec<MxArray>, CoreError>;

#[derive(Default)]
pub struct BuiltinRegistry {
    functions: HashMap<String, BuiltinFn>,
}

impl BuiltinRegistry {
    pub fn new() -> Self {
        let mut reg = Self {
            functions: HashMap::new(),
        };
        reg.register_defaults();
        reg
    }

    pub fn register(&mut self, name: &str, func: BuiltinFn) {
        self.functions.insert(name.to_lowercase(), func);
    }

    pub fn get(&self, name: &str) -> Option<&BuiltinFn> {
        self.functions.get(&name.to_lowercase())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.functions.contains_key(&name.to_lowercase())
    }

    pub fn call(
        &self,
        name: &str,
        args: &[MxArray],
        nargout: usize,
    ) -> Result<Vec<MxArray>, CoreError> {
        match self.get(name) {
            Some(func) => func(args, nargout),
            None => Err(CoreError::InvalidOperation(format!(
                "Unrecognized function or variable '{}'.",
                name
            ))),
        }
    }

    fn register_defaults(&mut self) {
        // ---------------------------------------------------------------------
        // Matrix Constructors
        // ---------------------------------------------------------------------
        self.register("zeros", |args, _| {
            let dims = parse_dims(args)?;
            Ok(vec![constructors::zeros(&dims)])
        });

        self.register("ones", |args, _| {
            let dims = parse_dims(args)?;
            Ok(vec![constructors::ones(&dims)])
        });

        self.register("eye", |args, _| {
            if args.is_empty() {
                Ok(vec![constructors::eye(1, 1)])
            } else if args.len() == 1 {
                let n = args[0].to_scalar()? as usize;
                Ok(vec![constructors::eye(n, n)])
            } else {
                let m = args[0].to_scalar()? as usize;
                let n = args[1].to_scalar()? as usize;
                Ok(vec![constructors::eye(m, n)])
            }
        });

        self.register("rand", |args, _| {
            let dims = parse_dims(args)?;
            Ok(vec![constructors::rand(&dims)])
        });

        self.register("randn", |args, _| {
            let dims = parse_dims(args)?;
            Ok(vec![constructors::randn(&dims)])
        });

        self.register("linspace", |args, _| {
            if args.len() < 2 {
                return Err(CoreError::InvalidOperation(
                    "Not enough input arguments for linspace.".into(),
                ));
            }
            let x1 = args[0].to_scalar()?;
            let x2 = args[1].to_scalar()?;
            let n = if args.len() >= 3 {
                args[2].to_scalar()? as usize
            } else {
                100
            };
            Ok(vec![constructors::linspace(x1, x2, n)?])
        });

        self.register("logspace", |args, _| {
            if args.len() < 2 {
                return Err(CoreError::InvalidOperation(
                    "Not enough input arguments for logspace.".into(),
                ));
            }
            let a = args[0].to_scalar()?;
            let b = args[1].to_scalar()?;
            let n = if args.len() >= 3 {
                args[2].to_scalar()? as usize
            } else {
                50
            };
            Ok(vec![constructors::logspace(a, b, n)?])
        });

        self.register("diag", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let k = if args.len() >= 2 {
                args[1].to_scalar()? as i32
            } else {
                0
            };
            Ok(vec![constructors::diag(&args[0], k)?])
        });

        self.register("meshgrid", |args, _| {
            if args.len() < 2 {
                return Err(CoreError::InvalidOperation(
                    "meshgrid requires at least 2 input arguments.".into(),
                ));
            }
            let (x, y) = constructors::meshgrid(&args[0], &args[1])?;
            Ok(vec![x, y])
        });

        self.register("magic", |args, _| {
            let n = if args.is_empty() { 3 } else { args[0].to_scalar()? as usize };
            Ok(vec![openlab_math::magic(n)?])
        });

        self.register("pascal", |args, _| {
            let n = if args.is_empty() { 3 } else { args[0].to_scalar()? as usize };
            Ok(vec![openlab_math::pascal(n)?])
        });

        self.register("hilb", |args, _| {
            let n = if args.is_empty() { 3 } else { args[0].to_scalar()? as usize };
            Ok(vec![openlab_math::hilb(n)?])
        });

        self.register("invhilb", |args, _| {
            let n = if args.is_empty() { 3 } else { args[0].to_scalar()? as usize };
            Ok(vec![openlab_math::invhilb(n)?])
        });

        self.register("moler", |args, _| {
            let n = if args.is_empty() { 3 } else { args[0].to_scalar()? as usize };
            Ok(vec![openlab_math::moler(n)?])
        });

        // ---------------------------------------------------------------------
        // Shape & Inspection
        // ---------------------------------------------------------------------
        self.register("size", |args, nargout| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let dim = if args.len() >= 2 {
                Some(args[1].to_scalar()? as usize)
            } else {
                None
            };
            let sz = manipulation::size(&args[0], dim)?;

            if nargout >= 2 && dim.is_none() {
                // [m, n] = size(A)
                let mut out = Vec::new();
                for &d in &sz {
                    out.push(MxArray::from_scalar(d as f64));
                }
                while out.len() < nargout {
                    out.push(MxArray::from_scalar(1.0));
                }
                Ok(out)
            } else {
                // s = size(A) returns [m, n] row vector
                let f_vec: Vec<f64> = sz.iter().map(|&x| x as f64).collect();
                let len = f_vec.len();
                Ok(vec![MxArray::from_f64_vec(Shape::row_vector(len), f_vec)?])
            }
        });

        self.register("length", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            Ok(vec![MxArray::from_scalar(manipulation::length(&args[0]) as f64)])
        });

        self.register("numel", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            Ok(vec![MxArray::from_scalar(manipulation::numel(&args[0]) as f64)])
        });

        self.register("ndims", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            Ok(vec![MxArray::from_scalar(manipulation::ndims(&args[0]) as f64)])
        });

        self.register("isempty", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            Ok(vec![MxArray::from_scalar_bool(manipulation::isempty(&args[0]))])
        });

        self.register("isvector", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            Ok(vec![MxArray::from_scalar_bool(manipulation::isvector(&args[0]))])
        });

        self.register("isscalar", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            Ok(vec![MxArray::from_scalar_bool(manipulation::isscalar(&args[0]))])
        });

        self.register("ismatrix", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            Ok(vec![MxArray::from_scalar_bool(manipulation::ismatrix(&args[0]))])
        });

        self.register("reshape", |args, _| {
            if args.len() < 2 {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let dims = parse_dims(&args[1..])?;
            Ok(vec![manipulation::reshape(&args[0], &dims)?])
        });

        self.register("squeeze", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            Ok(vec![manipulation::squeeze(&args[0])?])
        });

        self.register("repmat", |args, _| {
            if args.len() < 2 {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let m = args[1].to_scalar()? as usize;
            let n = if args.len() >= 3 {
                args[2].to_scalar()? as usize
            } else {
                m
            };
            Ok(vec![manipulation::repmat(&args[0], m, n)?])
        });

        self.register("transpose", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            Ok(vec![args[0].transpose()?])
        });

        self.register("ctranspose", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            Ok(vec![args[0].ctranspose()?])
        });

        self.register("triu", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let k = if args.len() >= 2 { args[1].to_scalar()? as i32 } else { 0 };
            Ok(vec![manipulation::triu(&args[0], k)?])
        });

        self.register("tril", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let k = if args.len() >= 2 { args[1].to_scalar()? as i32 } else { 0 };
            Ok(vec![manipulation::tril(&args[0], k)?])
        });

        self.register("fliplr", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            Ok(vec![manipulation::fliplr(&args[0])?])
        });

        self.register("flipud", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            Ok(vec![manipulation::flipud(&args[0])?])
        });

        self.register("rot90", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let k = if args.len() >= 2 { args[1].to_scalar()? as i32 } else { 1 };
            Ok(vec![manipulation::rot90(&args[0], k)?])
        });

        // ---------------------------------------------------------------------
        // Elementary Math
        // ---------------------------------------------------------------------
        self.register("sin", |args, _| unary_helper(args, elmath::sin));
        self.register("cos", |args, _| unary_helper(args, elmath::cos));
        self.register("tan", |args, _| unary_helper(args, elmath::tan));
        self.register("asin", |args, _| unary_helper(args, elmath::asin));
        self.register("acos", |args, _| unary_helper(args, elmath::acos));
        self.register("atan", |args, _| unary_helper(args, elmath::atan));
        self.register("atan2", |args, _| {
            if args.len() < 2 {
                return Err(CoreError::InvalidOperation("atan2 requires 2 inputs.".into()));
            }
            Ok(vec![elmath::atan2(&args[0], &args[1])?])
        });
        self.register("sind", |args, _| unary_helper(args, elmath::sind));
        self.register("cosd", |args, _| unary_helper(args, elmath::cosd));
        self.register("tand", |args, _| unary_helper(args, elmath::tand));
        self.register("sinh", |args, _| unary_helper(args, elmath::sinh));
        self.register("cosh", |args, _| unary_helper(args, elmath::cosh));
        self.register("tanh", |args, _| unary_helper(args, elmath::tanh));

        self.register("exp", |args, _| unary_helper(args, elmath::exp));
        self.register("log", |args, _| unary_helper(args, elmath::log));
        self.register("log10", |args, _| unary_helper(args, elmath::log10));
        self.register("log2", |args, _| unary_helper(args, elmath::log2));
        self.register("sqrt", |args, _| unary_helper(args, elmath::sqrt));
        self.register("cbrt", |args, _| unary_helper(args, elmath::cbrt));
        self.register("abs", |args, _| unary_helper(args, elmath::abs));
        self.register("sign", |args, _| unary_helper(args, elmath::sign));
        self.register("floor", |args, _| unary_helper(args, elmath::floor));
        self.register("ceil", |args, _| unary_helper(args, elmath::ceil));
        self.register("round", |args, _| unary_helper(args, elmath::round));
        self.register("fix", |args, _| unary_helper(args, elmath::fix));
        self.register("rem", |args, _| {
            if args.len() < 2 {
                return Err(CoreError::InvalidOperation("rem requires 2 inputs.".into()));
            }
            Ok(vec![elmath::rem(&args[0], &args[1])?])
        });
        self.register("mod", |args, _| {
            if args.len() < 2 {
                return Err(CoreError::InvalidOperation("mod requires 2 inputs.".into()));
            }
            Ok(vec![elmath::mod_op(&args[0], &args[1])?])
        });

        self.register("real", |args, _| unary_helper(args, elmath::real));
        self.register("imag", |args, _| unary_helper(args, elmath::imag));
        self.register("conj", |args, _| unary_helper(args, elmath::conj));
        self.register("angle", |args, _| unary_helper(args, elmath::angle));

        // ---------------------------------------------------------------------
        // Reductions
        // ---------------------------------------------------------------------
        self.register("sum", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let dim = parse_dim_spec(args)?;
            Ok(vec![reduction::sum(&args[0], dim)?])
        });

        self.register("prod", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let dim = parse_dim_spec(args)?;
            Ok(vec![reduction::prod(&args[0], dim)?])
        });

        self.register("mean", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let dim = parse_dim_spec(args)?;
            Ok(vec![reduction::mean(&args[0], dim)?])
        });

        self.register("min", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let dim = parse_dim_spec(args)?;
            Ok(vec![reduction::min(&args[0], dim)?])
        });

        self.register("max", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let dim = parse_dim_spec(args)?;
            Ok(vec![reduction::max(&args[0], dim)?])
        });

        self.register("any", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let dim = parse_dim_spec(args)?;
            Ok(vec![reduction::any(&args[0], dim)?])
        });

        self.register("all", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let dim = parse_dim_spec(args)?;
            Ok(vec![reduction::all(&args[0], dim)?])
        });

        self.register("cumsum", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let dim = if args.len() >= 2 { Some(args[1].to_scalar()? as usize) } else { None };
            Ok(vec![reduction::cumsum(&args[0], dim)?])
        });

        self.register("diff", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            Ok(vec![reduction::diff(&args[0])?])
        });

        // ---------------------------------------------------------------------
        // Linear Algebra Built-ins
        // ---------------------------------------------------------------------
        self.register("inv", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            Ok(vec![linear_algebra::math_inv(&args[0])?])
        });

        self.register("pinv", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            Ok(vec![linear_algebra::math_pinv(&args[0], None)?])
        });

        self.register("det", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            Ok(vec![MxArray::from_scalar(linear_algebra::math_det(&args[0])?)])
        });

        self.register("rank", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            Ok(vec![MxArray::from_scalar(linear_algebra::math_rank(&args[0], None)? as f64)])
        });

        self.register("trace", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            Ok(vec![MxArray::from_scalar(linear_algebra::math_trace(&args[0])?)])
        });

        self.register("cond", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            Ok(vec![MxArray::from_scalar(linear_algebra::math_cond(&args[0])?)])
        });

        self.register("norm", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let p = openlab_math::NormType::Two;
            Ok(vec![MxArray::from_scalar(linear_algebra::math_norm(&args[0], p)?)])
        });

        self.register("lu", |args, nargout| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let decomp = openlab_math::lu::decompose(&args[0])?;
            if nargout >= 3 {
                Ok(vec![decomp.l, decomp.u, decomp.p])
            } else if nargout == 2 {
                Ok(vec![decomp.l, decomp.u])
            } else {
                Ok(vec![decomp.u])
            }
        });

        self.register("chol", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let decomp = openlab_math::cholesky::decompose(&args[0])?;
            Ok(vec![decomp.r]) // MATLAB default chol(A) is upper triangular R
        });

        self.register("qr", |args, nargout| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let decomp = openlab_math::qr::decompose(&args[0], false)?;
            if nargout >= 2 {
                Ok(vec![decomp.q, decomp.r])
            } else {
                Ok(vec![decomp.r])
            }
        });

        self.register("svd", |args, nargout| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let decomp = openlab_math::svd::decompose(&args[0], false)?;
            if nargout >= 3 {
                Ok(vec![decomp.u, decomp.s, decomp.v])
            } else {
                let s_vec = decomp.singular_values;
                let n = s_vec.len();
                Ok(vec![MxArray::from_f64_vec(Shape::col_vector(n), s_vec)?])
            }
        });

        self.register("eig", |args, nargout| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let (v, d_opt) = linear_algebra::eig(&args[0], nargout)?;
            if let Some(d) = d_opt {
                Ok(vec![v, d])
            } else {
                Ok(vec![v])
            }
        });

        // ---------------------------------------------------------------------
        // I/O Built-ins
        // ---------------------------------------------------------------------
        self.register("disp", |args, _| {
            if args.is_empty() {
                println!();
            } else {
                let s = io::disp(&args[0])?;
                println!("{}", s);
            }
            Ok(Vec::new())
        });

        self.register("sprintf", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let fmt_str = if let openlab_core::MxData::Char(c) = args[0].data() {
                let s: String = c.iter().collect();
                s
            } else {
                return Err(CoreError::InvalidOperation("Format must be a character vector.".into()));
            };
            let formatted = io::sprintf(&fmt_str, &args[1..])?;
            Ok(vec![MxArray::from_string(&formatted)])
        });

        self.register("fprintf", |args, _| {
            if args.is_empty() {
                return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
            }
            let fmt_str = if let openlab_core::MxData::Char(c) = args[0].data() {
                let s: String = c.iter().collect();
                s
            } else {
                return Err(CoreError::InvalidOperation("Format must be a character vector.".into()));
            };
            let formatted = io::sprintf(&fmt_str, &args[1..])?;
            print!("{}", formatted);
            Ok(Vec::new())
        });
    }
}

fn unary_helper<F>(args: &[MxArray], f: F) -> Result<Vec<MxArray>, CoreError>
where
    F: Fn(&MxArray) -> Result<MxArray, CoreError>,
{
    if args.is_empty() {
        return Err(CoreError::InvalidOperation("Not enough input arguments.".into()));
    }
    Ok(vec![f(&args[0])?])
}

fn parse_dims(args: &[MxArray]) -> Result<Vec<usize>, CoreError> {
    if args.is_empty() {
        return Ok(vec![1, 1]);
    }
    if args.len() == 1 {
        // zeros(N) -> N-by-N square matrix
        if args[0].is_scalar() {
            let n = args[0].to_scalar()? as usize;
            return Ok(vec![n, n]);
        }
        // zeros([M, N, ...]) -> from row vector
        let n = args[0].numel();
        let mut dims = Vec::with_capacity(n);
        for i in 0..n {
            dims.push(args[0].get_linear_0based(i)? as usize);
        }
        return Ok(dims);
    }
    let mut dims = Vec::with_capacity(args.len());
    for arg in args {
        dims.push(arg.to_scalar()? as usize);
    }
    Ok(dims)
}

fn parse_dim_spec(args: &[MxArray]) -> Result<reduction::DimSpec, CoreError> {
    if args.len() < 2 {
        return Ok(reduction::DimSpec::FirstNonSingleton);
    }
    if let Ok(s) = args[1].to_scalar() {
        Ok(reduction::DimSpec::Dim(s as usize))
    } else {
        // e.g. sum(A, 'all')
        if let openlab_core::MxData::Char(c) = args[1].data() {
            let str_val: String = c.iter().collect();
            if str_val == "all" {
                return Ok(reduction::DimSpec::All);
            }
        }
        Ok(reduction::DimSpec::FirstNonSingleton)
    }
}

