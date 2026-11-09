use openlab_core::Complex64;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Expr {
        expr: Expr,
        print_result: bool,
    },
    Assign {
        target: AssignTarget,
        value: Expr,
        print_result: bool,
    },
    If {
        branches: Vec<(Expr, Vec<Statement>)>, // (condition, body) for if / elseif
        else_branch: Option<Vec<Statement>>,
    },
    For {
        var: String,
        iter: Expr,
        body: Vec<Statement>,
    },
    While {
        condition: Expr,
        body: Vec<Statement>,
    },
    Switch {
        expr: Expr,
        cases: Vec<(Expr, Vec<Statement>)>,
        otherwise: Option<Vec<Statement>>,
    },
    TryCatch {
        try_body: Vec<Statement>,
        err_var: Option<String>,
        catch_body: Vec<Statement>,
    },
    FunctionDef {
        name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
        body: Vec<Statement>,
    },
    Return,
    Break,
    Continue,
    Global(Vec<String>),
    Command {
        name: String,
        args: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssignTarget {
    Var(String),
    MultiVar(Vec<String>), // [a, b] = f()
    Subscript {
        target: Box<Expr>,
        indices: Vec<Expr>,
    },
    CellSubscript {
        target: Box<Expr>,
        indices: Vec<Expr>,
    },
    Field {
        target: Box<Expr>,
        field: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Number(f64),
    Complex(Complex64),
    StringLit(String),
    CharLit(String),
    Identifier(String),
    MatrixLit(Vec<Vec<Expr>>), // rows of columns
    CellLit(Vec<Vec<Expr>>),
    ColonRange {
        start: Box<Expr>,
        step: Option<Box<Expr>>,
        end: Box<Expr>,
    },
    ColonWildcard, // ':'
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    CallOrIndex {
        target: Box<Expr>,
        args: Vec<Expr>,
    },
    CellIndex {
        target: Box<Expr>,
        args: Vec<Expr>,
    },
    DotField {
        target: Box<Expr>,
        field: String,
    },
    Transpose(Box<Expr>),
    CTranspose(Box<Expr>),
    End,
    AnonymousFn {
        params: Vec<String>,
        body: Box<Expr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Plus,
    Minus,
    Not, // ~
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    LeftDiv, // \ (mldivide)
    Pow,     // ^

    // Element-Wise
    DotMul,     // .*
    DotDiv,     // ./
    DotLeftDiv, // .\
    DotPow,     // .^

    // Relational
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,

    // Logical
    And,    // &
    AndAnd, // &&
    Or,     // |
    OrOr,   // ||
}

