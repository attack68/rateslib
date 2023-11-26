use ndarray::{Array1, Array, arr1};
// use ndarray_einsum_beta::*;
use num_traits;
use num_traits::Pow;
// use std::collections::HashSet;
use std::sync::Arc;

// use indexmap::indexset;
use indexmap::set::IndexSet;
use auto_ops::{impl_op, impl_op_commutative, impl_op_ex, impl_op_ex_commutative};

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

