use crate::span::Span;
use crate::token::{Token, TokenKind};

#[derive(Debug, Clone)]
pub struct Lexer<'a> {
    source: &'a str,
    chars: Vec<(usize, char)>, // (byte_offset, char)
    cursor: usize,
    line: usize,
    col: usize,
    prev_token_kind: Option<TokenKind>,
    bracket_depth: usize,
    brace_depth: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        let chars: Vec<(usize, char)> = source.char_indices().collect();
        Self {
            source,
            chars,
            cursor: 0,
            line: 1,
            col: 1,
            prev_token_kind: None,
            bracket_depth: 0,
            brace_depth: 0,
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token()?;
            let is_eof = tok.kind == TokenKind::Eof;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }
        Ok(tokens)
    }

    pub fn next_token(&mut self) -> Result<Token, String> {
        self.skip_whitespace_and_comments();

        if self.is_at_end() {
            let span = Span::new(self.line, self.col, self.current_offset(), 0);
            return Ok(Token::new(TokenKind::Eof, span));
        }

        let start_line = self.line;
        let start_col = self.col;
        let start_offset = self.current_offset();

        let ch = self.peek();

        // 1. Newlines
        if ch == '\n' || ch == '\r' {
            self.advance();
            if ch == '\r' && self.peek() == '\n' {
                self.advance();
            }
            self.line += 1;
            self.col = 1;

            let span = Span::new(start_line, start_col, start_offset, 1);
            let tok = Token::new(TokenKind::Newline, span);
            self.prev_token_kind = Some(TokenKind::Newline);
            return Ok(tok);
        }

        // 2. Identifiers or Keywords
        if ch.is_alphabetic() || ch == '_' {
            let ident = self.consume_ident();
            let span = Span::new(
                start_line,
                start_col,
                start_offset,
                self.current_offset() - start_offset,
            );
            let kind = match ident.as_str() {
                "function" => TokenKind::Function,
                "end" => TokenKind::End,
                "if" => TokenKind::If,
                "elseif" => TokenKind::ElseIf,
                "else" => TokenKind::Else,
                "for" => TokenKind::For,
                "while" => TokenKind::While,
                "switch" => TokenKind::Switch,
                "case" => TokenKind::Case,
                "otherwise" => TokenKind::Otherwise,
                "try" => TokenKind::Try,
                "catch" => TokenKind::Catch,
                "return" => TokenKind::Return,
                "break" => TokenKind::Break,
                "continue" => TokenKind::Continue,
                "global" => TokenKind::Global,
                "persistent" => TokenKind::Persistent,
                _ => TokenKind::Ident(ident),
            };
            self.prev_token_kind = Some(kind.clone());
            return Ok(Token::new(kind, span));
        }

        // 3. Numbers (or leading dot followed by digit, e.g. .5)
        if ch.is_ascii_digit() || (ch == '.' && self.peek_next().is_ascii_digit()) {
            return self.consume_number(start_line, start_col, start_offset);
        }

        // 4. Strings: double quotes "..."
        if ch == '"' {
            return self.consume_double_quoted_string(start_line, start_col, start_offset);
        }

        // 5. Transpose vs Single-quoted char literal '...'
        if ch == '\'' {
            if self.is_transpose_context() {
                self.advance();
                let span = Span::new(start_line, start_col, start_offset, 1);
                let tok = Token::new(TokenKind::CTranspose, span);
                self.prev_token_kind = Some(TokenKind::CTranspose);
                return Ok(tok);
            } else {
                return self.consume_single_quoted_char_lit(start_line, start_col, start_offset);
            }
        }

        // 6. Dot operators: .', .*, ./, .\, .^, or Dot alone
        if ch == '.' {
            self.advance();
            let next = self.peek();
            let kind = match next {
                '\'' => {
                    self.advance();
                    TokenKind::Transpose
                }
                '*' => {
                    self.advance();
                    TokenKind::DotStar
                }
                '/' => {
                    self.advance();
                    TokenKind::DotSlash
                }
                '\\' => {
                    self.advance();
                    TokenKind::DotBackslash
                }
                '^' => {
                    self.advance();
                    TokenKind::DotCaret
                }
                _ => TokenKind::Dot,
            };
            let span = Span::new(
                start_line,
                start_col,
                start_offset,
                self.current_offset() - start_offset,
            );
            self.prev_token_kind = Some(kind.clone());
            return Ok(Token::new(kind, span));
        }

        // 7. Punctuation, Delimiters, Operators
        self.advance();
        let kind = match ch {
            '+' => TokenKind::Plus,
            '-' => TokenKind::Minus,
            '*' => TokenKind::Star,
            '/' => TokenKind::Slash,
            '\\' => TokenKind::Backslash,
            '^' => TokenKind::Caret,
            ':' => TokenKind::Colon,
            ',' => TokenKind::Comma,
            ';' => TokenKind::Semicolon,
            '@' => TokenKind::At,
            '(' => TokenKind::OpenParen,
            ')' => TokenKind::CloseParen,
            '[' => {
                self.bracket_depth += 1;
                TokenKind::OpenBracket
            }
            ']' => {
                if self.bracket_depth > 0 {
                    self.bracket_depth -= 1;
                }
                TokenKind::CloseBracket
            }
            '{' => {
                self.brace_depth += 1;
                TokenKind::OpenBrace
            }
            '}' => {
                if self.brace_depth > 0 {
                    self.brace_depth -= 1;
                }
                TokenKind::CloseBrace
            }
            '=' => {
                if self.peek() == '=' {
                    self.advance();
                    TokenKind::EqEq
                } else {
                    TokenKind::Equal
                }
            }
            '~' => {
                if self.peek() == '=' {
                    self.advance();
                    TokenKind::NotEq
                } else {
                    TokenKind::Tilde
                }
            }
            '<' => {
                if self.peek() == '=' {
                    self.advance();
                    TokenKind::LtEq
                } else {
                    TokenKind::Lt
                }
            }
            '>' => {
                if self.peek() == '=' {
                    self.advance();
                    TokenKind::GtEq
                } else {
                    TokenKind::Gt
                }
            }
            '&' => {
                if self.peek() == '&' {
                    self.advance();
                    TokenKind::AndAnd
                } else {
                    TokenKind::And
                }
            }
            '|' => {
                if self.peek() == '|' {
                    self.advance();
                    TokenKind::OrOr
                } else {
                    TokenKind::Or
                }
            }
            _ => {
                return Err(format!(
                    "Unexpected character '{}' at {}:{}",
                    ch, start_line, start_col
                ));
            }
        };

        let span = Span::new(
            start_line,
            start_col,
            start_offset,
            self.current_offset() - start_offset,
        );
        self.prev_token_kind = Some(kind.clone());
        Ok(Token::new(kind, span))
    }

    fn is_transpose_context(&self) -> bool {
        match self.prev_token_kind {
            Some(TokenKind::Ident(_))
            | Some(TokenKind::Number(_))
            | Some(TokenKind::Imaginary(_))
            | Some(TokenKind::CloseParen)
            | Some(TokenKind::CloseBracket)
            | Some(TokenKind::CloseBrace)
            | Some(TokenKind::CTranspose)
            | Some(TokenKind::Transpose)
            | Some(TokenKind::End) => true,
            _ => false,
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        while !self.is_at_end() {
            let ch = self.peek();

            // Line continuation: ... followed by optional whitespace and newline
            if ch == '.' && self.peek_next() == '.' && self.peek_at(2) == '.' {
                self.advance();
                self.advance();
                self.advance();
                // skip until end of line
                while !self.is_at_end() && self.peek() != '\n' && self.peek() != '\r' {
                    self.advance();
                }
                if !self.is_at_end() {
                    let nl = self.peek();
                    self.advance();
                    if nl == '\r' && self.peek() == '\n' {
                        self.advance();
                    }
                    self.line += 1;
                    self.col = 1;
                }
                continue;
            }

            // Line Comments %
            if ch == '%' {
                // Check for block comment %{ ... %}
                if self.peek_next() == '{' {
                    self.advance(); // %
                    self.advance(); // {
                    while !self.is_at_end() {
                        if self.peek() == '%' && self.peek_next() == '}' {
                            self.advance();
                            self.advance();
                            break;
                        }
                        if self.peek() == '\n' {
                            self.line += 1;
                            self.col = 0;
                        }
                        self.advance();
                    }
                    continue;
                } else {
                    // Regular line comment
                    while !self.is_at_end() && self.peek() != '\n' && self.peek() != '\r' {
                        self.advance();
                    }
                    continue;
                }
            }

            // Regular spaces and tabs (not newlines)
            if ch == ' ' || ch == '\t' {
                self.advance();
                continue;
            }

            break;
        }
    }

    fn consume_ident(&mut self) -> String {
        let mut s = String::new();
        while !self.is_at_end() && (self.peek().is_alphanumeric() || self.peek() == '_') {
            s.push(self.peek());
            self.advance();
        }
        s
    }

    fn consume_number(
        &mut self,
        start_line: usize,
        start_col: usize,
        start_offset: usize,
    ) -> Result<Token, String> {
        let mut num_str = String::new();
        let mut has_dot = false;
        let mut has_exp = false;

        while !self.is_at_end() {
            let ch = self.peek();
            if ch.is_ascii_digit() {
                num_str.push(ch);
                self.advance();
            } else if ch == '.' && !has_dot && !has_exp && self.peek_next() != '*' && self.peek_next() != '/' && self.peek_next() != '\\' && self.peek_next() != '^' && self.peek_next() != '\'' {
                has_dot = true;
                num_str.push(ch);
                self.advance();
            } else if (ch == 'e' || ch == 'E') && !has_exp {
                has_exp = true;
                num_str.push(ch);
                self.advance();
                if self.peek() == '+' || self.peek() == '-' {
                    num_str.push(self.peek());
                    self.advance();
                }
            } else {
                break;
            }
        }

        // Check for imaginary unit 'i' or 'j'
        let is_imag = if !self.is_at_end() && (self.peek() == 'i' || self.peek() == 'j') {
            self.advance();
            true
        } else {
            false
        };

        let val: f64 = num_str
            .parse()
            .map_err(|e| format!("Invalid number literal '{}': {}", num_str, e))?;

        let span = Span::new(
            start_line,
            start_col,
            start_offset,
            self.current_offset() - start_offset,
        );

        let kind = if is_imag {
            TokenKind::Imaginary(val)
        } else {
            TokenKind::Number(val)
        };

        self.prev_token_kind = Some(kind.clone());
        Ok(Token::new(kind, span))
    }

    fn consume_single_quoted_char_lit(
        &mut self,
        start_line: usize,
        start_col: usize,
        start_offset: usize,
    ) -> Result<Token, String> {
        self.advance(); // opening '
        let mut s = String::new();

        while !self.is_at_end() {
            let ch = self.peek();
            if ch == '\'' {
                if self.peek_next() == '\'' {
                    // Escaped single quote ''
                    s.push('\'');
                    self.advance();
                    self.advance();
                } else {
                    // End of string
                    self.advance();
                    let span = Span::new(
                        start_line,
                        start_col,
                        start_offset,
                        self.current_offset() - start_offset,
                    );
                    let kind = TokenKind::CharLit(s);
                    self.prev_token_kind = Some(kind.clone());
                    return Ok(Token::new(kind, span));
                }
            } else if ch == '\n' || ch == '\r' {
                return Err(format!(
                    "Unterminated character vector literal at {}:{}",
                    start_line, start_col
                ));
            } else {
                s.push(ch);
                self.advance();
            }
        }

        Err(format!(
            "Unterminated character vector literal at {}:{}",
            start_line, start_col
        ))
    }

    fn consume_double_quoted_string(
        &mut self,
        start_line: usize,
        start_col: usize,
        start_offset: usize,
    ) -> Result<Token, String> {
        self.advance(); // opening "
        let mut s = String::new();

        while !self.is_at_end() {
            let ch = self.peek();
            if ch == '"' {
                self.advance();
                let span = Span::new(
                    start_line,
                    start_col,
                    start_offset,
                    self.current_offset() - start_offset,
                );
                let kind = TokenKind::StringLit(s);
                self.prev_token_kind = Some(kind.clone());
                return Ok(Token::new(kind, span));
            } else if ch == '\\' {
                self.advance();
                if !self.is_at_end() {
                    let esc = self.peek();
                    s.push(match esc {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        '\\' => '\\',
                        '"' => '"',
                        _ => esc,
                    });
                    self.advance();
                }
            } else {
                s.push(ch);
                self.advance();
            }
        }

        Err(format!(
            "Unterminated string literal at {}:{}",
            start_line, start_col
        ))
    }

    fn is_at_end(&self) -> bool {
        self.cursor >= self.chars.len()
    }

    fn peek(&self) -> char {
        if self.is_at_end() {
            '\0'
        } else {
            self.chars[self.cursor].1
        }
    }

    fn peek_next(&self) -> char {
        if self.cursor + 1 >= self.chars.len() {
            '\0'
        } else {
            self.chars[self.cursor + 1].1
        }
    }

    fn peek_at(&self, offset: usize) -> char {
        if self.cursor + offset >= self.chars.len() {
            '\0'
        } else {
            self.chars[self.cursor + offset].1
        }
    }

    fn advance(&mut self) -> char {
        let ch = self.peek();
        self.cursor += 1;
        self.col += 1;
        ch
    }

    fn current_offset(&self) -> usize {
        if self.cursor < self.chars.len() {
            self.chars[self.cursor].0
        } else {
            self.source.len()
        }
    }
}

