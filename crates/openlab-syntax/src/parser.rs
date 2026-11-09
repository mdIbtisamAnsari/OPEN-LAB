use crate::ast::{
    AssignTarget, BinaryOp, Expr, Program, Statement, UnaryOp,
};
use crate::lexer::Lexer;
use crate::token::{Token, TokenKind};
use openlab_core::Complex64;

#[derive(Debug, Clone)]
pub struct Parser {
    tokens: Vec<Token>,
    cursor: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, cursor: 0 }
    }

    pub fn from_source(source: &str) -> Result<Self, String> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize()?;
        Ok(Self::new(tokens))
    }

    pub fn parse_program(&mut self) -> Result<Program, String> {
        let mut statements = Vec::new();
        self.skip_newlines();

        while !self.is_at_end() {
            let stmt = self.parse_statement()?;
            statements.push(stmt);
            self.skip_newlines_and_separators();
        }

        Ok(Program { statements })
    }

    // -------------------------------------------------------------------------
    // Statement Parsing
    // -------------------------------------------------------------------------

    pub fn parse_statement(&mut self) -> Result<Statement, String> {
        self.skip_newlines();

        // 1. Control flow keywords
        if self.match_token(&TokenKind::If) {
            return self.parse_if();
        }
        if self.match_token(&TokenKind::For) {
            return self.parse_for();
        }
        if self.match_token(&TokenKind::While) {
            return self.parse_while();
        }
        if self.match_token(&TokenKind::Switch) {
            return self.parse_switch();
        }
        if self.match_token(&TokenKind::Try) {
            return self.parse_try_catch();
        }
        if self.match_token(&TokenKind::Function) {
            return self.parse_function();
        }
        if self.match_token(&TokenKind::Return) {
            return Ok(Statement::Return);
        }
        if self.match_token(&TokenKind::Break) {
            return Ok(Statement::Break);
        }
        if self.match_token(&TokenKind::Continue) {
            return Ok(Statement::Continue);
        }
        if self.match_token(&TokenKind::Global) {
            let mut vars = Vec::new();
            while let TokenKind::Ident(name) = self.peek_kind() {
                vars.push(name.clone());
                self.advance();
            }
            return Ok(Statement::Global(vars));
        }

        // 2. Multi-variable assignment: [a, b] = f()
        if self.peek_kind() == &TokenKind::OpenBracket {
            if let Some(stmt) = self.try_parse_multivar_assign()? {
                return Ok(stmt);
            }
        }

        // 3. Regular Expression or Assignment
        let expr = self.parse_expr(0)?;

        // Check if followed by '=' -> Assignment
        if self.match_token(&TokenKind::Equal) {
            let val = self.parse_expr(0)?;
            let print_result = self.consume_statement_terminator();
            let target = match expr {
                Expr::Identifier(name) => AssignTarget::Var(name),
                Expr::CallOrIndex { target, args } => AssignTarget::Subscript {
                    target,
                    indices: args,
                },
                Expr::CellIndex { target, args } => AssignTarget::CellSubscript {
                    target,
                    indices: args,
                },
                Expr::DotField { target, field } => AssignTarget::Field { target, field },
                _ => return Err("Invalid assignment target.".into()),
            };
            return Ok(Statement::Assign {
                target,
                value: val,
                print_result,
            });
        }

        let print_result = self.consume_statement_terminator();
        Ok(Statement::Expr { expr, print_result })
    }

    fn try_parse_multivar_assign(&mut self) -> Result<Option<Statement>, String> {
        let saved_cursor = self.cursor;
        self.advance(); // [

        let mut vars = Vec::new();
        while self.peek_kind() != &TokenKind::CloseBracket && !self.is_at_end() {
            if let TokenKind::Ident(name) = self.peek_kind() {
                vars.push(name.clone());
                self.advance();
                if self.peek_kind() == &TokenKind::Comma {
                    self.advance();
                }
            } else {
                self.cursor = saved_cursor;
                return Ok(None);
            }
        }

        if !self.match_token(&TokenKind::CloseBracket) {
            self.cursor = saved_cursor;
            return Ok(None);
        }

        if !self.match_token(&TokenKind::Equal) {
            self.cursor = saved_cursor;
            return Ok(None);
        }

        let value = self.parse_expr(0)?;
        let print_result = self.consume_statement_terminator();
        Ok(Some(Statement::Assign {
            target: AssignTarget::MultiVar(vars),
            value,
            print_result,
        }))
    }

    fn parse_if(&mut self) -> Result<Statement, String> {
        let mut branches = Vec::new();

        // 1. Initial if condition
        let cond = self.parse_expr(0)?;
        self.skip_newlines_and_separators();
        let body = self.parse_block(&[TokenKind::ElseIf, TokenKind::Else, TokenKind::End])?;
        branches.push((cond, body));

        // 2. elseif branches
        while self.match_token(&TokenKind::ElseIf) {
            let cond = self.parse_expr(0)?;
            self.skip_newlines_and_separators();
            let body = self.parse_block(&[TokenKind::ElseIf, TokenKind::Else, TokenKind::End])?;
            branches.push((cond, body));
        }

        // 3. Optional else branch
        let else_branch = if self.match_token(&TokenKind::Else) {
            self.skip_newlines_and_separators();
            let body = self.parse_block(&[TokenKind::End])?;
            Some(body)
        } else {
            None
        };

        self.expect(&TokenKind::End, "Expected 'end' to close 'if' statement")?;
        Ok(Statement::If {
            branches,
            else_branch,
        })
    }

    fn parse_for(&mut self) -> Result<Statement, String> {
        let var = match self.peek_kind() {
            TokenKind::Ident(name) => {
                let s = name.clone();
                self.advance();
                s
            }
            _ => return Err("Expected loop variable in 'for' statement.".into()),
        };

        self.expect(&TokenKind::Equal, "Expected '=' in 'for' loop header")?;
        let iter = self.parse_expr(0)?;
        self.skip_newlines_and_separators();

        let body = self.parse_block(&[TokenKind::End])?;
        self.expect(&TokenKind::End, "Expected 'end' to close 'for' loop")?;

        Ok(Statement::For { var, iter, body })
    }

    fn parse_while(&mut self) -> Result<Statement, String> {
        let condition = self.parse_expr(0)?;
        self.skip_newlines_and_separators();

        let body = self.parse_block(&[TokenKind::End])?;
        self.expect(&TokenKind::End, "Expected 'end' to close 'while' loop")?;

        Ok(Statement::While { condition, body })
    }

    fn parse_switch(&mut self) -> Result<Statement, String> {
        let expr = self.parse_expr(0)?;
        self.skip_newlines_and_separators();

        let mut cases = Vec::new();
        let mut otherwise = None;

        while !self.is_at_end() && self.peek_kind() != &TokenKind::End {
            if self.match_token(&TokenKind::Case) {
                let case_expr = self.parse_expr(0)?;
                self.skip_newlines_and_separators();
                let body = self.parse_block(&[
                    TokenKind::Case,
                    TokenKind::Otherwise,
                    TokenKind::End,
                ])?;
                cases.push((case_expr, body));
            } else if self.match_token(&TokenKind::Otherwise) {
                self.skip_newlines_and_separators();
                let body = self.parse_block(&[TokenKind::End])?;
                otherwise = Some(body);
            } else {
                break;
            }
        }

        self.expect(&TokenKind::End, "Expected 'end' to close 'switch' block")?;
        Ok(Statement::Switch {
            expr,
            cases,
            otherwise,
        })
    }

    fn parse_try_catch(&mut self) -> Result<Statement, String> {
        self.skip_newlines_and_separators();
        let try_body = self.parse_block(&[TokenKind::Catch, TokenKind::End])?;

        let mut err_var = None;
        let mut catch_body = Vec::new();

        if self.match_token(&TokenKind::Catch) {
            if let TokenKind::Ident(var) = self.peek_kind() {
                err_var = Some(var.clone());
                self.advance();
            }
            self.skip_newlines_and_separators();
            catch_body = self.parse_block(&[TokenKind::End])?;
        }

        self.expect(&TokenKind::End, "Expected 'end' to close 'try' block")?;
        Ok(Statement::TryCatch {
            try_body,
            err_var,
            catch_body,
        })
    }

    fn parse_function(&mut self) -> Result<Statement, String> {
        // Syntax variants:
        // function out = fname(in1, in2)
        // function [out1, out2] = fname(in1, in2)
        // function fname(in1, in2)
        let mut outputs = Vec::new();

        if self.peek_kind() == &TokenKind::OpenBracket {
            self.advance();
            while self.peek_kind() != &TokenKind::CloseBracket && !self.is_at_end() {
                if let TokenKind::Ident(name) = self.peek_kind() {
                    outputs.push(name.clone());
                    self.advance();
                    if self.peek_kind() == &TokenKind::Comma {
                        self.advance();
                    }
                } else {
                    break;
                }
            }
            self.expect(&TokenKind::CloseBracket, "Expected ']' in function outputs")?;
            self.expect(&TokenKind::Equal, "Expected '=' after function outputs")?;
        }

        let first_ident = match self.peek_kind() {
            TokenKind::Ident(name) => {
                let s = name.clone();
                self.advance();
                s
            }
            _ => return Err("Expected function name or output variable.".into()),
        };

        let fname = if outputs.is_empty() && self.match_token(&TokenKind::Equal) {
            // first_ident was single output variable: function out = fname(...)
            outputs.push(first_ident);
            match self.peek_kind() {
                TokenKind::Ident(name) => {
                    let s = name.clone();
                    self.advance();
                    s
                }
                _ => return Err("Expected function name after '='.".into()),
            }
        } else {
            first_ident
        };

        // Inputs (in1, in2, ...)
        let mut inputs = Vec::new();
        if self.match_token(&TokenKind::OpenParen) {
            while self.peek_kind() != &TokenKind::CloseParen && !self.is_at_end() {
                if let TokenKind::Ident(name) = self.peek_kind() {
                    inputs.push(name.clone());
                    self.advance();
                    if self.peek_kind() == &TokenKind::Comma {
                        self.advance();
                    }
                } else {
                    break;
                }
            }
            self.expect(&TokenKind::CloseParen, "Expected ')' after function inputs")?;
        }

        self.skip_newlines_and_separators();
        let body = self.parse_block(&[TokenKind::End])?;
        self.expect(&TokenKind::End, "Expected 'end' to close function definition")?;

        Ok(Statement::FunctionDef {
            name: fname,
            inputs,
            outputs,
            body,
        })
    }

    fn parse_block(&mut self, stop_tokens: &[TokenKind]) -> Result<Vec<Statement>, String> {
        let mut stmts = Vec::new();
        while !self.is_at_end() && !stop_tokens.contains(self.peek_kind()) {
            let stmt = self.parse_statement()?;
            stmts.push(stmt);
            self.skip_newlines_and_separators();
        }
        Ok(stmts)
    }

    fn consume_statement_terminator(&mut self) -> bool {
        // Semicolon suppresses output (print_result = false)
        if self.match_token(&TokenKind::Semicolon) {
            while self.match_token(&TokenKind::Semicolon) || self.match_token(&TokenKind::Comma) {}
            false
        } else if self.match_token(&TokenKind::Comma) {
            while self.match_token(&TokenKind::Comma) {}
            true
        } else {
            true
        }
    }

    // -------------------------------------------------------------------------
    // Expression Parsing (Pratt Precedence Climbing)
    // -------------------------------------------------------------------------

    pub fn parse_expr(&mut self, min_bp: u8) -> Result<Expr, String> {
        let mut left = self.parse_primary()?;

        loop {
            // Postfix operators: transpose (' or .'), call (), cell access {}, field access .
            if self.peek_kind() == &TokenKind::CTranspose {
                self.advance();
                left = Expr::CTranspose(Box::new(left));
                continue;
            }
            if self.peek_kind() == &TokenKind::Transpose {
                self.advance();
                left = Expr::Transpose(Box::new(left));
                continue;
            }
            if self.peek_kind() == &TokenKind::OpenParen {
                self.advance();
                let args = self.parse_expr_list(TokenKind::CloseParen)?;
                left = Expr::CallOrIndex {
                    target: Box::new(left),
                    args,
                };
                continue;
            }
            if self.peek_kind() == &TokenKind::OpenBrace {
                self.advance();
                let args = self.parse_expr_list(TokenKind::CloseBrace)?;
                left = Expr::CellIndex {
                    target: Box::new(left),
                    args,
                };
                continue;
            }
            if self.peek_kind() == &TokenKind::Dot && matches!(self.peek_next_kind(), Some(TokenKind::Ident(_))) {
                self.advance(); // .
                if let TokenKind::Ident(field) = self.peek_kind().clone() {
                    self.advance();
                    left = Expr::DotField {
                        target: Box::new(left),
                        field,
                    };
                    continue;
                }
            }

            // Colon Range: start : end or start : step : end
            if self.peek_kind() == &TokenKind::Colon && min_bp <= 10 {
                self.advance();
                let next_expr = self.parse_expr(11)?;
                if self.peek_kind() == &TokenKind::Colon {
                    self.advance();
                    let end_expr = self.parse_expr(11)?;
                    left = Expr::ColonRange {
                        start: Box::new(left),
                        step: Some(Box::new(next_expr)),
                        end: Box::new(end_expr),
                    };
                } else {
                    left = Expr::ColonRange {
                        start: Box::new(left),
                        step: None,
                        end: Box::new(next_expr),
                    };
                }
                continue;
            }

            // Infix binary operators
            let (op, l_bp, r_bp) = match self.peek_kind() {
                // Precedence 20: Or
                TokenKind::Or => (BinaryOp::Or, 20, 21),
                TokenKind::OrOr => (BinaryOp::OrOr, 20, 21),
                // Precedence 30: And
                TokenKind::And => (BinaryOp::And, 30, 31),
                TokenKind::AndAnd => (BinaryOp::AndAnd, 30, 31),
                // Precedence 40: Relational
                TokenKind::EqEq => (BinaryOp::Eq, 40, 41),
                TokenKind::NotEq => (BinaryOp::NotEq, 40, 41),
                TokenKind::Lt => (BinaryOp::Lt, 40, 41),
                TokenKind::LtEq => (BinaryOp::LtEq, 40, 41),
                TokenKind::Gt => (BinaryOp::Gt, 40, 41),
                TokenKind::GtEq => (BinaryOp::GtEq, 40, 41),
                // Precedence 50: Add/Sub
                TokenKind::Plus => (BinaryOp::Add, 50, 51),
                TokenKind::Minus => (BinaryOp::Sub, 50, 51),
                // Precedence 60: Mul/Div
                TokenKind::Star => (BinaryOp::Mul, 60, 61),
                TokenKind::Slash => (BinaryOp::Div, 60, 61),
                TokenKind::Backslash => (BinaryOp::LeftDiv, 60, 61),
                TokenKind::DotStar => (BinaryOp::DotMul, 60, 61),
                TokenKind::DotSlash => (BinaryOp::DotDiv, 60, 61),
                TokenKind::DotBackslash => (BinaryOp::DotLeftDiv, 60, 61),
                // Precedence 80: Power (right associative)
                TokenKind::Caret => (BinaryOp::Pow, 81, 80),
                TokenKind::DotCaret => (BinaryOp::DotPow, 81, 80),
                _ => break,
            };

            if l_bp < min_bp {
                break;
            }

            self.advance();
            let right = self.parse_expr(r_bp)?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        // Unary operators: +, -, ~
        if self.match_token(&TokenKind::Plus) {
            let expr = self.parse_expr(70)?;
            return Ok(Expr::Unary {
                op: UnaryOp::Plus,
                expr: Box::new(expr),
            });
        }
        if self.match_token(&TokenKind::Minus) {
            let expr = self.parse_expr(70)?;
            return Ok(Expr::Unary {
                op: UnaryOp::Minus,
                expr: Box::new(expr),
            });
        }
        if self.match_token(&TokenKind::Tilde) {
            let expr = self.parse_expr(70)?;
            return Ok(Expr::Unary {
                op: UnaryOp::Not,
                expr: Box::new(expr),
            });
        }

        // Colon Wildcard ':'
        if self.match_token(&TokenKind::Colon) {
            return Ok(Expr::ColonWildcard);
        }

        // Number
        if let TokenKind::Number(n) = self.peek_kind() {
            let val = *n;
            self.advance();
            return Ok(Expr::Number(val));
        }

        // Imaginary
        if let TokenKind::Imaginary(n) = self.peek_kind() {
            let val = *n;
            self.advance();
            return Ok(Expr::Complex(Complex64::imag(val)));
        }

        // String / Char
        if let TokenKind::CharLit(s) = self.peek_kind() {
            let val = s.clone();
            self.advance();
            return Ok(Expr::CharLit(val));
        }
        if let TokenKind::StringLit(s) = self.peek_kind() {
            let val = s.clone();
            self.advance();
            return Ok(Expr::StringLit(val));
        }

        // Keyword 'end' in index position
        if self.match_token(&TokenKind::End) {
            return Ok(Expr::End);
        }

        // Identifier
        if let TokenKind::Ident(name) = self.peek_kind() {
            let s = name.clone();
            self.advance();
            return Ok(Expr::Identifier(s));
        }

        // Grouping: ( expr )
        if self.match_token(&TokenKind::OpenParen) {
            let expr = self.parse_expr(0)?;
            self.expect(&TokenKind::CloseParen, "Expected ')'")?;
            return Ok(expr);
        }

        // Matrix literal: [1, 2; 3, 4]
        if self.match_token(&TokenKind::OpenBracket) {
            return self.parse_matrix_literal();
        }

        // Cell literal: {1, 'two'}
        if self.match_token(&TokenKind::OpenBrace) {
            return self.parse_cell_literal();
        }

        // Anonymous function: @(x, y) expr
        if self.match_token(&TokenKind::At) {
            return self.parse_anonymous_function();
        }

        Err(format!(
            "Unexpected token '{}' at position.",
            self.peek_kind()
        ))
    }

    fn parse_matrix_literal(&mut self) -> Result<Expr, String> {
        let mut rows = Vec::new();
        let mut current_row = Vec::new();

        self.skip_newlines();
        while self.peek_kind() != &TokenKind::CloseBracket && !self.is_at_end() {
            if self.match_token(&TokenKind::Semicolon) || self.match_token(&TokenKind::Newline) {
                if !current_row.is_empty() {
                    rows.push(current_row);
                    current_row = Vec::new();
                }
                self.skip_newlines();
                continue;
            }

            let elem = self.parse_expr(0)?;
            current_row.push(elem);

            // Optional comma between elements
            if self.peek_kind() == &TokenKind::Comma {
                self.advance();
            }
        }

        if !current_row.is_empty() {
            rows.push(current_row);
        }

        self.expect(&TokenKind::CloseBracket, "Expected ']' at end of matrix")?;
        Ok(Expr::MatrixLit(rows))
    }

    fn parse_cell_literal(&mut self) -> Result<Expr, String> {
        let mut rows = Vec::new();
        let mut current_row = Vec::new();

        self.skip_newlines();
        while self.peek_kind() != &TokenKind::CloseBrace && !self.is_at_end() {
            if self.match_token(&TokenKind::Semicolon) || self.match_token(&TokenKind::Newline) {
                if !current_row.is_empty() {
                    rows.push(current_row);
                    current_row = Vec::new();
                }
                self.skip_newlines();
                continue;
            }

            let elem = self.parse_expr(0)?;
            current_row.push(elem);

            if self.peek_kind() == &TokenKind::Comma {
                self.advance();
            }
        }

        if !current_row.is_empty() {
            rows.push(current_row);
        }

        self.expect(&TokenKind::CloseBrace, "Expected '}' at end of cell array")?;
        Ok(Expr::CellLit(rows))
    }

    fn parse_anonymous_function(&mut self) -> Result<Expr, String> {
        self.expect(&TokenKind::OpenParen, "Expected '(' in anonymous function parameter list")?;
        let mut params = Vec::new();
        while self.peek_kind() != &TokenKind::CloseParen && !self.is_at_end() {
            if let TokenKind::Ident(p) = self.peek_kind() {
                params.push(p.clone());
                self.advance();
                if self.peek_kind() == &TokenKind::Comma {
                    self.advance();
                }
            } else {
                break;
            }
        }
        self.expect(&TokenKind::CloseParen, "Expected ')' after parameter list")?;
        let body = self.parse_expr(0)?;
        Ok(Expr::AnonymousFn {
            params,
            body: Box::new(body),
        })
    }

    fn parse_expr_list(&mut self, closing_delim: TokenKind) -> Result<Vec<Expr>, String> {
        let mut args = Vec::new();
        if self.peek_kind() == &closing_delim {
            self.advance();
            return Ok(args);
        }

        loop {
            let expr = self.parse_expr(0)?;
            args.push(expr);

            if self.peek_kind() == &TokenKind::Comma {
                self.advance();
            } else if self.peek_kind() == &closing_delim {
                self.advance();
                break;
            } else {
                return Err(format!("Expected ',' or '{}'", closing_delim));
            }
        }

        Ok(args)
    }

    // -------------------------------------------------------------------------
    // Helper Methods
    // -------------------------------------------------------------------------

    fn is_at_end(&self) -> bool {
        self.cursor >= self.tokens.len() || self.tokens[self.cursor].kind == TokenKind::Eof
    }

    fn peek_kind(&self) -> &TokenKind {
        if self.cursor < self.tokens.len() {
            &self.tokens[self.cursor].kind
        } else {
            &TokenKind::Eof
        }
    }

    fn peek_next_kind(&self) -> Option<&TokenKind> {
        if self.cursor + 1 < self.tokens.len() {
            Some(&self.tokens[self.cursor + 1].kind)
        } else {
            None
        }
    }

    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.cursor += 1;
        }
        &self.tokens[self.cursor - 1]
    }

    fn match_token(&mut self, kind: &TokenKind) -> bool {
        if self.peek_kind() == kind {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: &TokenKind, err: &str) -> Result<(), String> {
        if self.peek_kind() == kind {
            self.advance();
            Ok(())
        } else {
            Err(format!("{} (got '{}')", err, self.peek_kind()))
        }
    }

    fn skip_newlines(&mut self) {
        while self.peek_kind() == &TokenKind::Newline {
            self.advance();
        }
    }

    fn skip_newlines_and_separators(&mut self) {
        while self.peek_kind() == &TokenKind::Newline
            || self.peek_kind() == &TokenKind::Comma
            || self.peek_kind() == &TokenKind::Semicolon
        {
            self.advance();
        }
    }
}

