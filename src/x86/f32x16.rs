use crate::{Simd, Simdt};
use_x86!();

/*
    I DON'T HACE A CPU WITH AVX512, SO RIGHT NOW I'M JUST ASSUMING THIS WORKS.
    THIS MUST BE CHECKED IN THE FUTURE
*/

#[cfg(any(feature = "force-avx512", target_feature = "avx512f"))]
impl Simdt for Simd<__m512> {
    type Item = f32;
    
    /// Sums all the values inside
    /// [See](https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction)
    #[inline(always)]
    fn sum (self) -> f32 {
        unsafe {
            let vlow  = _mm512_castps512_ps256(self.0);
            let vhigh1 = _mm512_extractf32x4_ps(self.0, 2);
            let vhigh2 = _mm512_extractf32x4_ps(self.0, 3);
            let vhigh = _mm256_set_m128(vhigh2, vhigh1);
            let vlow = _mm256_add_ps(vlow, vhigh);
            Simd(vlow).sum()
        }
    }

    /// Multiplies all the values inside
    #[inline(always)]
    fn prod (self) -> f32 {
        unsafe {
            let vlow  = _mm512_castps512_ps256(self.0);
            let vhigh1 = _mm512_extractf32x4_ps(self.0, 2);
            let vhigh2 = _mm512_extractf32x4_ps(self.0, 3);
            let vhigh = _mm256_set_m128(vhigh2, vhigh1);
            let vlow = _mm256_mul_ps(vlow, vhigh);
            Simd(vlow).prod()
        }
    }

    #[inline(always)]
    /// Finds the smallest value inside
    fn min (self) -> f32 {
        unsafe {
            let vlow  = _mm512_castps512_ps256(self.0);
            let vhigh1 = _mm512_extractf32x4_ps(self.0, 2);
            let vhigh2 = _mm512_extractf32x4_ps(self.0, 3);
            let vhigh = _mm256_set_m128(vhigh2, vhigh1);
            let vlow = _mm256_min_ps(vlow, vhigh);
            Simd(vlow).min()
        }
    }

    #[inline(always)]
    /// Finds the biggest value inside
    fn max (self) -> f32 {
        unsafe {
            let vlow  = _mm512_castps512_ps256(self.0);
            let vhigh1 = _mm512_extractf32x4_ps(self.0, 2);
            let vhigh2 = _mm512_extractf32x4_ps(self.0, 3);
            let vhigh = _mm256_set_m128(vhigh2, vhigh1);
            let vlow = _mm256_max_ps(vlow, vhigh);
            Simd(vlow).max()
        }
    }
}