use crate::{Simd, Simdt};
use_x86!();

#[cfg(any(feature = "force-avx", target_feature = "avx"))]
impl Simdt for Simd<__m256> {
    type Item = f32;
    
    /// Sums all the values inside
    /// [See](https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction)
    #[inline(always)]
    fn sum (self) -> f32 {
        unsafe {
            let vlow  = _mm256_castps256_ps128(self.0);
            let vhigh = _mm256_extractf128_ps(self.0, 1);
            let vlow = _mm_add_ps(vlow, vhigh);
            Simd(vlow).sum()
        }
    }

    /// Multiplies all the values inside
    /// [See](https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction)
    #[inline(always)]
    fn prod (self) -> f32 {
        unsafe {
            let vlow  = _mm256_castps256_ps128(self.0);
            let vhigh = _mm256_extractf128_ps(self.0, 1);
            let vlow = _mm_mul_ps(vlow, vhigh);
            Simd(vlow).prod()
        }
    }
}