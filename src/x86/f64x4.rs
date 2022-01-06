use crate::{Simd, Simdt};
use_x86!();

impl Simdt for Simd<__m256d> {
    type Item = f64;
    
    /// Sums all the values inside
    #[inline(always)]
    fn sum (self) -> f64 {
        unsafe {
            let vlow = _mm256_castpd256_pd128(self.0);
            let vhigh = _mm256_extractf128_pd(self.0, 1);
            let vlow = _mm_add_pd(vlow, vhigh);
            Simd(vlow).sum()
        }
    }

    /// Multiplies all the values inside
    #[inline(always)]
    fn prod (self) -> f64 {
        unsafe {
            let vlow = _mm256_castpd256_pd128(self.0);
            let vhigh = _mm256_extractf128_pd(self.0, 1);
            let vlow = _mm_mul_pd(vlow, vhigh);
            Simd(vlow).prod()
        }
    }

    /// Multiplies all the values inside
    #[inline(always)]
    fn min (self) -> f64 {
        unsafe {
            let vlow = _mm256_castpd256_pd128(self.0);
            let vhigh = _mm256_extractf128_pd(self.0, 1);
            let vlow = _mm_min_pd(vlow, vhigh);
            Simd(vlow).min()
        }
    }

    /// Multiplies all the values inside
    #[inline(always)]
    fn max (self) -> f64 {
        unsafe {
            let vlow = _mm256_castpd256_pd128(self.0);
            let vhigh = _mm256_extractf128_pd(self.0, 1);
            let vlow = _mm_max_pd(vlow, vhigh);
            Simd(vlow).max()
        }
    }
}