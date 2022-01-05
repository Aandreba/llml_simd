use crate::{Simd};
use_x86!();

impl Simd<__m128d> {
    /// Sums all the values inside
    #[inline(always)]
    pub fn sum (self) -> f64 {
        self[0] + self[1]
    }

    /// Sums all the values inside
    #[inline(always)]
    pub fn prod (self) -> f64 {
        self[0] * self[1]
    }
}