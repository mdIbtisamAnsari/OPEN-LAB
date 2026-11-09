use openlab_core::{CoreError, MxArray, MxData};

/// disp(X) displays array X without printing the variable name.
pub fn disp(arr: &MxArray) -> Result<String, CoreError> {
    if let MxData::Char(chars) = arr.data() {
        let s: String = chars.iter().collect();
        return Ok(s);
    }
    Ok(format!("{}", arr))
}

/// Formats a string with arguments: sprintf(fmt, args...)
pub fn sprintf(fmt_str: &str, args: &[MxArray]) -> Result<String, CoreError> {
    let mut out = String::new();
    let mut chars = fmt_str.chars().peekable();
    let mut arg_idx = 0;

    while let Some(ch) = chars.next() {
        if ch == '%' {
            if let Some(&next_ch) = chars.peek() {
                if next_ch == '%' {
                    chars.next();
                    out.push('%');
                    continue;
                }
                if next_ch == 's' {
                    chars.next();
                    if arg_idx < args.len() {
                        let arg = &args[arg_idx];
                        arg_idx += 1;
                        if let MxData::Char(c_vec) = arg.data() {
                            let s: String = c_vec.iter().collect();
                            out.push_str(&s);
                        } else {
                            out.push_str(&format!("{}", arg));
                        }
                    }
                    continue;
                }
                if next_ch == 'd' || next_ch == 'i' {
                    chars.next();
                    if arg_idx < args.len() {
                        let val = args[arg_idx].to_scalar().unwrap_or(0.0) as i64;
                        arg_idx += 1;
                        out.push_str(&format!("{}", val));
                    }
                    continue;
                }
                if next_ch == 'f' {
                    chars.next();
                    if arg_idx < args.len() {
                        let val = args[arg_idx].to_scalar().unwrap_or(0.0);
                        arg_idx += 1;
                        out.push_str(&format!("{:.6}", val));
                    }
                    continue;
                }
                if next_ch == 'g' || next_ch == 'e' {
                    chars.next();
                    if arg_idx < args.len() {
                        let val = args[arg_idx].to_scalar().unwrap_or(0.0);
                        arg_idx += 1;
                        out.push_str(&format!("{:e}", val));
                    }
                    continue;
                }
            }
        }
        if ch == '\\' {
            if let Some(&next_ch) = chars.peek() {
                if next_ch == 'n' {
                    chars.next();
                    out.push('\n');
                    continue;
                } else if next_ch == 't' {
                    chars.next();
                    out.push('\t');
                    continue;
                } else if next_ch == '\\' {
                    chars.next();
                    out.push('\\');
                    continue;
                }
            }
        }
        out.push(ch);
    }

    Ok(out)
}

/// fprintf(fmt, args...) formats and prints to standard output.
pub fn fprintf(fmt_str: &str, args: &[MxArray]) -> Result<(), CoreError> {
    let s = sprintf(fmt_str, args)?;
    print!("{}", s);
    Ok(())
}
