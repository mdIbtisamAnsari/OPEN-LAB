use crate::array::MxArray;
use crate::complex::Complex64;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub enum MxData {
    Double(Arc<Vec<f64>>),
    ComplexDouble(Arc<Vec<Complex64>>),
    Single(Arc<Vec<f32>>),
    Int8(Arc<Vec<i8>>),
    Uint8(Arc<Vec<u8>>),
    Int16(Arc<Vec<i16>>),
    Uint16(Arc<Vec<u16>>),
    Int32(Arc<Vec<i32>>),
    Uint32(Arc<Vec<u32>>),
    Int64(Arc<Vec<i64>>),
    Uint64(Arc<Vec<u64>>),
    Logical(Arc<Vec<bool>>),
    Char(Arc<Vec<char>>),
    Cell(Arc<Vec<MxArray>>),
}

impl MxData {
    pub fn class_name(&self) -> &'static str {
        match self {
            MxData::Double(_) => "double",
            MxData::ComplexDouble(_) => "double",
            MxData::Single(_) => "single",
            MxData::Int8(_) => "int8",
            MxData::Uint8(_) => "uint8",
            MxData::Int16(_) => "int16",
            MxData::Uint16(_) => "uint16",
            MxData::Int32(_) => "int32",
            MxData::Uint32(_) => "uint32",
            MxData::Int64(_) => "int64",
            MxData::Uint64(_) => "uint64",
            MxData::Logical(_) => "logical",
            MxData::Char(_) => "char",
            MxData::Cell(_) => "cell",
        }
    }

    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            MxData::Double(_)
                | MxData::ComplexDouble(_)
                | MxData::Single(_)
                | MxData::Int8(_)
                | MxData::Uint8(_)
                | MxData::Int16(_)
                | MxData::Uint16(_)
                | MxData::Int32(_)
                | MxData::Uint32(_)
                | MxData::Int64(_)
                | MxData::Uint64(_)
        )
    }

    pub fn is_complex(&self) -> bool {
        matches!(self, MxData::ComplexDouble(_))
    }

    pub fn is_logical(&self) -> bool {
        matches!(self, MxData::Logical(_))
    }

    pub fn is_char(&self) -> bool {
        matches!(self, MxData::Char(_))
    }

    pub fn is_cell(&self) -> bool {
        matches!(self, MxData::Cell(_))
    }

    pub fn element_byte_size(&self) -> usize {
        match self {
            MxData::Double(_) => std::mem::size_of::<f64>(),
            MxData::ComplexDouble(_) => std::mem::size_of::<Complex64>(),
            MxData::Single(_) => std::mem::size_of::<f32>(),
            MxData::Int8(_) | MxData::Uint8(_) | MxData::Logical(_) => 1,
            MxData::Int16(_) | MxData::Uint16(_) => 2,
            MxData::Int32(_) | MxData::Uint32(_) => 4,
            MxData::Int64(_) | MxData::Uint64(_) => 8,
            MxData::Char(_) => 2, // MATLAB char is UTF-16 (2 bytes)
            MxData::Cell(_) => std::mem::size_of::<usize>(),
        }
    }

    pub fn total_bytes(&self) -> usize {
        match self {
            MxData::Double(v) => v.len() * std::mem::size_of::<f64>(),
            MxData::ComplexDouble(v) => v.len() * std::mem::size_of::<Complex64>(),
            MxData::Single(v) => v.len() * std::mem::size_of::<f32>(),
            MxData::Int8(v) => v.len(),
            MxData::Uint8(v) => v.len(),
            MxData::Int16(v) => v.len() * 2,
            MxData::Uint16(v) => v.len() * 2,
            MxData::Int32(v) => v.len() * 4,
            MxData::Uint32(v) => v.len() * 4,
            MxData::Int64(v) => v.len() * 8,
            MxData::Uint64(v) => v.len() * 8,
            MxData::Logical(v) => v.len(),
            MxData::Char(v) => v.len() * 2,
            MxData::Cell(v) => v.iter().map(|item| item.total_bytes()).sum(),
        }
    }
}

