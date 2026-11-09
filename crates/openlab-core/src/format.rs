use crate::array::MxArray;
use crate::data::MxData;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayFormat {
    Short,      // 4 decimal digits
    Long,       // 15 decimal digits
    ShortE,     // exponential
    LongE,      // exponential
}

pub struct MatrixFormatter<'a> {
    array: &'a MxArray,
    var_name: Option<&'a str>,
    format: DisplayFormat,
}

impl<'a> MatrixFormatter<'a> {
    pub fn new(array: &'a MxArray) -> Self {
        Self {
            array,
            var_name: None,
            format: DisplayFormat::Short,
        }
    }

    pub fn with_var_name(mut self, name: &'a str) -> Self {
        self.var_name = Some(name);
        self
    }

    pub fn with_format(mut self, format: DisplayFormat) -> Self {
        self.format = format;
        self
    }
}

impl<'a> fmt::Display for MatrixFormatter<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Variable header if provided
        if let Some(name) = self.var_name {
            writeln!(f, "{} =", name)?;
            writeln!(f)?;
        }

        if self.array.is_empty() {
            let dims = self.array.shape().dims();
            if dims == [0, 0] {
                return writeln!(f, "     []");
            } else {
                let dim_str: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
                return writeln!(f, "     Empty matrix: {}", dim_str.join("-by-"));
            }
        }

        if let MxData::Char(chars) = self.array.data() {
            let s: String = chars.iter().collect();
            return writeln!(f, "    '{}'", s);
        }

        if self.array.is_logical() && self.array.ndims() == 2 {
            return self.format_logical_2d(f);
        }

        if self.array.ndims() == 2 {
            self.format_2d(f)
        } else {
            self.format_nd(f)
        }
    }
}

impl<'a> MatrixFormatter<'a> {
    fn format_logical_2d(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let m = self.array.rows();
        let n = self.array.cols();
        for r in 1..=m {
            write!(f, "  ")?;
            for c in 1..=n {
                let val = self.array.get_2d(r, c).unwrap_or(0.0) != 0.0;
                write!(f, "   {}", if val { 1 } else { 0 })?;
            }
            writeln!(f)?;
        }
        Ok(())
    }

    fn format_value(&self, val: f64) -> String {
        if val.is_nan() {
            return "   NaN".to_string();
        }
        if val.is_infinite() {
            return if val > 0.0 { "   Inf".to_string() } else { "  -Inf".to_string() };
        }
        match self.format {
            DisplayFormat::Short => {
                if val.fract() == 0.0 && val.abs() < 1e5 {
                    format!("{:8.0}", val)
                } else {
                    format!("{:10.4}", val)
                }
            }
            DisplayFormat::Long => {
                format!("{:20.15}", val)
            }
            DisplayFormat::ShortE => {
                format!("{:12.4e}", val)
            }
            DisplayFormat::LongE => {
                format!("{:24.15e}", val)
            }
        }
    }

    fn format_2d(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let m = self.array.rows();
        let n = self.array.cols();

        for r in 1..=m {
            write!(f, "  ")?;
            for c in 1..=n {
                if self.array.is_complex() {
                    let c_val = self.array.get_complex_2d(r, c).unwrap();
                    write!(f, "  {:>16}", format!("{}", c_val))?;
                } else {
                    let val = self.array.get_2d(r, c).unwrap();
                    write!(f, "{}", self.format_value(val))?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }

    fn format_nd(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = self.array.shape();
        let dims = shape.dims();
        let m = dims[0];
        let n = dims[1];

        let num_pages: usize = dims[2..].iter().product();

        for page in 0..num_pages {
            // Deconstruct page into trailing indices
            let mut page_indices = Vec::with_capacity(dims.len() - 2);
            let mut temp = page;
            for &d in &dims[2..] {
                page_indices.push((temp % d) + 1);
                temp /= d;
            }

            let page_header: Vec<String> = page_indices.iter().map(|idx| idx.to_string()).collect();
            writeln!(f, "(:,:,{}) =", page_header.join(","))?;
            writeln!(f)?;

            for r in 1..=m {
                write!(f, "  ")?;
                for c in 1..=n {
                    let mut full_coords = vec![r, c];
                    full_coords.extend_from_slice(&page_indices);
                    let val = self.array.get_nd(&full_coords).unwrap();
                    write!(f, "{}", self.format_value(val))?;
                }
                writeln!(f)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl fmt::Display for MxArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        MatrixFormatter::new(self).fmt(f)
    }
}
