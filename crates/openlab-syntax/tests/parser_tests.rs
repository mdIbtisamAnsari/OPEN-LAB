use openlab_core::Complex64;
use openlab_syntax::{
    ast::{AssignTarget, BinaryOp, Expr, Statement},
    parse_expr, parse_script,
};

#[test]
fn test_parse_arithmetic_precedence() {
    // 1 + 2 * 3 -> 1 + (2 * 3)
    let expr = parse_expr("1 + 2 * 3").unwrap();
    match expr {
        Expr::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Add);
            assert_eq!(*left, Expr::Number(1.0));
            match *right {
                Expr::Binary {
                    op: op_r,
                    left: left_r,
                    right: right_r,
                } => {
                    assert_eq!(op_r, BinaryOp::Mul);
                    assert_eq!(*left_r, Expr::Number(2.0));
                    assert_eq!(*right_r, Expr::Number(3.0));
                }
                _ => panic!("Expected multiplication on right"),
            }
        }
        _ => panic!("Expected binary addition"),
    }
}

#[test]
fn test_parse_matrix_literal() {
    let expr = parse_expr("[1, 2, 3; 4, 5, 6]").unwrap();
    match expr {
        Expr::MatrixLit(rows) => {
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].len(), 3);
            assert_eq!(rows[1].len(), 3);
            assert_eq!(rows[0][0], Expr::Number(1.0));
            assert_eq!(rows[0][2], Expr::Number(3.0));
            assert_eq!(rows[1][0], Expr::Number(4.0));
            assert_eq!(rows[1][2], Expr::Number(6.0));
        }
        _ => panic!("Expected matrix literal"),
    }
}

#[test]
fn test_parse_cell_literal() {
    let expr = parse_expr("{1, 'hello', 3.14}").unwrap();
    match expr {
        Expr::CellLit(rows) => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].len(), 3);
            assert_eq!(rows[0][0], Expr::Number(1.0));
            assert_eq!(rows[0][1], Expr::CharLit("hello".into()));
        }
        _ => panic!("Expected cell literal"),
    }
}

#[test]
fn test_parse_colon_range() {
    // Binary range: 1:10
    let r1 = parse_expr("1:10").unwrap();
    match r1 {
        Expr::ColonRange { start, step, end } => {
            assert_eq!(*start, Expr::Number(1.0));
            assert!(step.is_none());
            assert_eq!(*end, Expr::Number(10.0));
        }
        _ => panic!("Expected binary colon range"),
    }

    // Ternary range: 1:2:10
    let r2 = parse_expr("1:2:10").unwrap();
    match r2 {
        Expr::ColonRange { start, step, end } => {
            assert_eq!(*start, Expr::Number(1.0));
            assert_eq!(*step.unwrap(), Expr::Number(2.0));
            assert_eq!(*end, Expr::Number(10.0));
        }
        _ => panic!("Expected ternary colon range"),
    }
}

#[test]
fn test_parse_transpose_vs_char_lit() {
    // Transpose
    let t1 = parse_expr("A'").unwrap();
    match t1 {
        Expr::CTranspose(inner) => assert_eq!(*inner, Expr::Identifier("A".into())),
        _ => panic!("Expected CTranspose"),
    }

    let t2 = parse_expr("A.'").unwrap();
    match t2 {
        Expr::Transpose(inner) => assert_eq!(*inner, Expr::Identifier("A".into())),
        _ => panic!("Expected Transpose"),
    }

    let t3 = parse_expr("[1, 2]'").unwrap();
    match t3 {
        Expr::CTranspose(inner) => match *inner {
            Expr::MatrixLit(_) => {}
            _ => panic!("Expected MatrixLit inside CTranspose"),
        },
        _ => panic!("Expected CTranspose"),
    }

    // Char literal
    let s = parse_expr("'world'").unwrap();
    assert_eq!(s, Expr::CharLit("world".into()));
}

#[test]
fn test_parse_imaginary_numbers() {
    let expr = parse_expr("3.5i").unwrap();
    assert_eq!(expr, Expr::Complex(Complex64::imag(3.5)));

    let expr2 = parse_expr("4j").unwrap();
    assert_eq!(expr2, Expr::Complex(Complex64::imag(4.0)));
}

#[test]
fn test_parse_anonymous_function() {
    let expr = parse_expr("@(x, y) x.^2 + y.^2").unwrap();
    match expr {
        Expr::AnonymousFn { params, body } => {
            assert_eq!(params, vec!["x", "y"]);
            match *body {
                Expr::Binary { op, .. } => assert_eq!(op, BinaryOp::Add),
                _ => panic!("Expected addition body"),
            }
        }
        _ => panic!("Expected anonymous function"),
    }
}

#[test]
fn test_parse_semicolon_suppression() {
    let script = parse_script("x = 5;\ny = 10\n").unwrap();
    assert_eq!(script.statements.len(), 2);

    match &script.statements[0] {
        Statement::Assign { print_result, .. } => assert_eq!(*print_result, false),
        _ => panic!("Expected Assign"),
    }
    match &script.statements[1] {
        Statement::Assign { print_result, .. } => assert_eq!(*print_result, true),
        _ => panic!("Expected Assign"),
    }
}

#[test]
fn test_parse_control_flow() {
    let source = r#"
        if x > 0
            y = 1;
        elseif x == 0
            y = 0;
        else
            y = -1;
        end

        for i = 1:5
            sum = sum + i;
        end

        while count > 0
            count = count - 1;
        end
    "#;

    let program = parse_script(source).unwrap();
    assert_eq!(program.statements.len(), 3);
    assert!(matches!(program.statements[0], Statement::If { .. }));
    assert!(matches!(program.statements[1], Statement::For { .. }));
    assert!(matches!(program.statements[2], Statement::While { .. }));
}

#[test]
fn test_parse_function_definition() {
    let source = r#"
        function [v, d] = my_eig(A)
            v = 1;
            d = 2;
        end
    "#;

    let program = parse_script(source).unwrap();
    assert_eq!(program.statements.len(), 1);
    match &program.statements[0] {
        Statement::FunctionDef {
            name,
            inputs,
            outputs,
            body,
        } => {
            assert_eq!(name, "my_eig");
            assert_eq!(inputs, &vec!["A"]);
            assert_eq!(outputs, &vec!["v", "d"]);
            assert_eq!(body.len(), 2);
        }
        _ => panic!("Expected FunctionDef"),
    }
}

#[test]
fn test_parse_multivar_assign() {
    let program = parse_script("[V, D] = eig(A);").unwrap();
    assert_eq!(program.statements.len(), 1);
    match &program.statements[0] {
        Statement::Assign {
            target: AssignTarget::MultiVar(vars),
            print_result,
            ..
        } => {
            assert_eq!(vars, &vec!["V", "D"]);
            assert_eq!(*print_result, false);
        }
        _ => panic!("Expected MultiVar assignment"),
    }
}

#[test]
fn test_comments_and_continuation() {
    let source = r#"
        % Initial line comment
        x = [1, ...
             2; ...
             3, 4]; % trailing comment
        %{
           Multi-line block comment
           which spans multiple lines
        %}
        y = x';
    "#;

    let program = parse_script(source).unwrap();
    assert_eq!(program.statements.len(), 2);
}
