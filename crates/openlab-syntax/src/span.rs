use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub line: usize,
    pub col: usize,
    pub offset: usize,
    pub len: usize,
}

impl Span {
    pub fn new(line: usize, col: usize, offset: usize, len: usize) -> Self {
        Self {
            line,
            col,
            offset,
            len,
        }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

