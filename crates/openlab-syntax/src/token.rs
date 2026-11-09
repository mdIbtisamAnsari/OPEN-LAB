use crate::span::Span;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords
    Function,
    End,
    If,
    ElseIf,
    Else,
    For,
    While,
    Switch,
    Case,
    Otherwise,
    Try,
    Catch,
    Return,
    Break,
    Continue,
    Global,
    Persistent,

    // Literals
    Number(f64),
    Imaginary(f64),
    CharLit(String),
    StringLit(String),

    // Identifiers
    Ident(String),

    // Arithmetic Operators
    Plus,          // +
    Minus,         // -
    Star,          // *
    Slash,         // /
    Backslash,     // \
    Caret,         // ^

    // Element-Wise Operators
    DotStar,       // .*
    DotSlash,      // ./
    DotBackslash,  // .\
    DotCaret,      // .^

    // Transpose
    Transpose,     // .'
    CTranspose,    // '

    // Relational Operators
    EqEq,          // ==
    NotEq,         // ~=
    Lt,            // <
    LtEq,          // <=
    Gt,            // >
    GtEq,          // >=

    // Logical Operators
    And,           // &
    AndAnd,        // &&
    Or,            // |
    OrOr,          // ||
    Tilde,         // ~

    // Assignment
    Equal,         // =

    // Delimiters & Punctuation
    Colon,         // :
    Comma,         // ,
    Semicolon,     // ;
    Dot,           // .
    At,            // @
    OpenParen,     // (
    CloseParen,    // )
    OpenBracket,   // [
    CloseBracket,  // ]
    OpenBrace,     // {
    CloseBrace,    // }

    // Line / Stream
    Newline,
    Eof,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::Function => write!(f, "function"),
            TokenKind::End => write!(f, "end"),
            TokenKind::If => write!(f, "if"),
            TokenKind::ElseIf => write!(f, "elseif"),
            TokenKind::Else => write!(f, "else"),
            TokenKind::For => write!(f, "for"),
            TokenKind::While => write!(f, "while"),
            TokenKind::Switch => write!(f, "switch"),
            TokenKind::Case => write!(f, "case"),
            TokenKind::Otherwise => write!(f, "otherwise"),
            TokenKind::Try => write!(f, "try"),
            TokenKind::Catch => write!(f, "catch"),
            TokenKind::Return => write!(f, "return"),
            TokenKind::Break => write!(f, "break"),
            TokenKind::Continue => write!(f, "continue"),
            TokenKind::Global => write!(f, "global"),
            TokenKind::Persistent => write!(f, "persistent"),
            TokenKind::Number(n) => write!(f, "{}", n),
            TokenKind::Imaginary(n) => write!(f, "{}i", n),
            TokenKind::CharLit(s) => write!(f, "'{}'", s),
            TokenKind::StringLit(s) => write!(f, "\"{}\"", s),
            TokenKind::Ident(s) => write!(f, "{}", s),
            TokenKind::Plus => write!(f, "+"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::Star => write!(f, "*"),
            TokenKind::Slash => write!(f, "/"),
            TokenKind::Backslash => write!(f, "\\"),
            TokenKind::Caret => write!(f, "^"),
            TokenKind::DotStar => write!(f, ".*"),
            TokenKind::DotSlash => write!(f, "./"),
            TokenKind::DotBackslash => write!(f, ".\\"),
            TokenKind::DotCaret => write!(f, ".^"),
            TokenKind::Transpose => write!(f, ".'"),
            TokenKind::CTranspose => write!(f, "'"),
            TokenKind::EqEq => write!(f, "=="),
            TokenKind::NotEq => write!(f, "~="),
            TokenKind::Lt => write!(f, "<"),
            TokenKind::LtEq => write!(f, "<="),
            TokenKind::Gt => write!(f, ">"),
            TokenKind::GtEq => write!(f, ">="),
            TokenKind::And => write!(f, "&"),
            TokenKind::AndAnd => write!(f, "&&"),
            TokenKind::Or => write!(f, "|"),
            TokenKind::OrOr => write!(f, "||"),
            TokenKind::Tilde => write!(f, "~"),
            TokenKind::Equal => write!(f, "="),
            TokenKind::Colon => write!(f, ":"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Semicolon => write!(f, ";"),
            TokenKind::Dot => write!(f, "."),
            TokenKind::At => write!(f, "@"),
            TokenKind::OpenParen => write!(f, "("),
            TokenKind::CloseParen => write!(f, ")"),
            TokenKind::OpenBracket => write!(f, "["),
            TokenKind::CloseBracket => write!(f, "]"),
            TokenKind::OpenBrace => write!(f, "{{"),
            TokenKind::CloseBrace => write!(f, "}}"),
            TokenKind::Newline => write!(f, "<newline>"),
            TokenKind::Eof => write!(f, "<EOF>"),
        }
    }
}

