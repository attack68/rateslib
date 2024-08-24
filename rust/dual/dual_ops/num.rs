use crate::dual::dual::{Dual, Dual2, Number};
use num_traits::Num;

impl Num for Dual {
    // PartialEq + Zero + One + NumOps (Add + Sub + Mul + Div + Rem)
    type FromStrRadixErr = String;
    fn from_str_radix(_src: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Err("No implementation for sting radix for Dual".to_string())
    }
}

impl Num for Dual2 {
    type FromStrRadixErr = String;
    fn from_str_radix(_src: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Err("No implementation for sting radix for Dual2".to_string())
    }
}

impl Num for Number {
    type FromStrRadixErr = String;
    fn from_str_radix(_src: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Err("No implementation for sting radix for Number".to_string())
    }
}
