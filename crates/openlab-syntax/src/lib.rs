pub mod ast;
pub mod lexer;
pub mod parser;
pub mod span;
pub mod token;

pub use ast::{AssignTarget, BinaryOp, Expr, Program, Statement, UnaryOp};
pub use lexer::Lexer;
pub use parser::Parser;
pub use span::Span;
pub use token::{Token, TokenKind};

/// Convenient helper to parse a MATLAB script from a source string.
pub fn parse_script(source: &str) -> Result<Program, String> {
    let mut parser = Parser::from_source(source)?;
    parser.parse_program()
}

/// Convenient helper to parse a single MATLAB expression from a source string.
pub fn parse_expr(source: &str) -> Result<Expr, String> {
    let mut parser = Parser::from_source(source)?;
    parser.parse_expr(0)
}

