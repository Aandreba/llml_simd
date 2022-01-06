use crate::{Simd, Simdt};
use_x86!();

impl Simdt for Simd<__m128d> {
    type Item = f64;

    /// Sums all the values inside
    #[inline(always)]
    fn sum (self) -> f64 {
        self[0] + self[1]
    }

    /// Sums all the values inside
    #[inline(always)]
    fn prod (self) -> f64 {
        self[0] * self[1]
    }

    /// Finds the smallest value inside
    #[inline(always)]
    fn min (self) -> Self::Item {
        self[0].min(self[1])
    }

    /// Finds the biggest value inside
    #[inline(always)]
    fn max (self) -> Self::Item {
        self[0].max(self[1])
    }
}