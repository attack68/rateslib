use ndarray::{Array1, Array, arr1};
use num_traits;
use num_traits::Pow;
use std::sync::Arc;


pub mod dual1;
use crate::dual::dual1::Dual;


#[derive(Debug, Clone, PartialEq)]
pub enum Duals {
    Float(f64),
    Dual(Dual),
}

// impl_op_ex!(+ |a: &Duals, b: &Duals| -> Duals {
//     use Duals::*;
//     match a {
//         Float(mt)=>{
//             match b {
//                 Float(mto) => Float(mt + mto),
//                 Dual(mto) => Dual(mt + mto)
//             }
//         },
//         Dual(mt) => {
//             match b {
//                 Float(mto) => Dual(mt + mto),
//                 Dual(mto) => Dual(mt + mto),
//             }
//         }
//     }
// });

// impl std::ops::Add for &Duals {
//     type Output = Duals;
//     fn add(self, other: &Duals) -> Duals {
//         use Duals::*;
//         match self {
//             Float(MT)=>{
//                 match other {
//                     Float(MTO) => Float(MT + MTO),
//                     Dual(MTO) => Dual(MT + MTO)
//                 }
//             },
//             Dual(MT) => {
//                 match other {
//                     Float(MTO) => Dual(MT + MTO),
//                     Dual(MTO) => Dual(MT + MTO),
//                 }
//             }
//         }
//     }
// }
//
// impl std::ops::Add for Duals {
//     type Output = Duals;
//     fn add(self, other: Duals) -> Duals {
//         use Duals::*;
//         match self {
//             Float(MT)=>{
//                 match other {
//                     Float(MTO) => Float(MT + MTO),
//                     Dual(MTO) => Dual(MT + MTO)
//                 }
//             },
//             Dual(MT) => {
//                 match other {
//                     Float(MTO) => Dual(MT + MTO),
//                     Dual(MTO) => Dual(MT + MTO),
//                 }
//             }
//         }
//     }
// }

