use derive_more::*;
use std::ops::*;
use llml_simd_proc::{Assign, assign_targets, assign_rhs};
use super::*;

macro_rules! impl_scal_arith {
    ($target:ident, $ty:ident, $($trait:ident, $fun:ident),+) => {
        $(
            impl $trait<$ty> for $target {
                type Output = Self;
    
                #[inline(always)]
                fn $fun (self, rhs: $ty) -> Self::Output {
                    self.$fun(Into::<Self>::into(rhs))
                }
            }

            impl $trait<$target> for $ty {
                type Output = $target;
    
                #[inline(always)]
                fn $fun (self, rhs: $target) -> Self::Output {
                    Into::<$target>::into(self).$fun(rhs)
                }
            }
        )*
    };
}

impl_composite!(
    (f32x4 => 4, f32x2 => 2) as f32x6: f32,
    (f32x4 => 4, f32x4 => 4) as f32x8: f32,
    (f32x4 => 4, f32x6 => 6) as f32x10: f32,
    (f32x6 => 6, f32x6 => 6) as f32x12: f32,
    (f32x6 => 6, f32x8 => 8) as f32x14: f32,
    (f32x8 => 8, f32x8 => 8) as f32x16: f32
);

impl_composite!(
    (f64x2 => 2, f64x2 => 2) as f64x4: f64,
    (f64x2 => 2, f64x4 => 4) as f64x6: f64,
    (f64x4 => 4, f64x4 => 4) as f64x8: f64,
    (f64x4 => 4, f64x6 => 6) as f64x10: f64,
    (f64x6 => 6, f64x6 => 6) as f64x12: f64,
    (f64x4 => 6, f64x8 => 8) as f64x14: f64,
    (f64x4 => 8, f64x8 => 8) as f64x16: f64
);