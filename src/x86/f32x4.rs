use cfg_if::cfg_if;
use crate::{Simd};
use_x86!();

impl Simd<__m128> {
    cfg_if! {
        if #[cfg(any(feature = "force-sse3", target_feature = "sse3"))] {
            /// Sums all the values inside
            /// [See](https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction)
            #[inline(always)]
            pub fn sum (self) -> f32 {
                unsafe {
                    let shuf = _mm_movehdup_ps(self.0);
                    let sums = _mm_add_ps(self.0, shuf);
                    let shuf = _mm_movehl_ps(shuf, sums);
                    let sums = _mm_add_ss(sums, shuf);
                    _mm_cvtss_f32(sums)
                }
            }

            /// Multiplies all the values inside
            /// [See](https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction)
            #[inline(always)]
            pub fn prod (self) -> f32 {
                unsafe {
                    let shuf = _mm_movehdup_ps(self.0);
                    let sums = _mm_mul_ps(self.0, shuf);
                    let shuf = _mm_movehl_ps(shuf, sums);
                    let sums = _mm_mul_ss(sums, shuf);
                    _mm_cvtss_f32(sums)
                }
            }
        } else {
            /// Sums all the values inside
            /// [See](https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction)
            #[inline(always)]
            pub fn sum (self) -> f32 {
                unsafe {
                    let shuf = _mm_shuffle_ps(self.0, self.0, _MM_SHUFFLE(2, 3, 0, 1));
                    let sums = _mm_add_ps(self.0, shuf);
                    let shuf = _mm_movehl_ps(shuf, sums);
                    let sums = _mm_add_ss(sums, shuf);
                    _mm_cvtss_f32(sums)
                }
            }

            /// Multiplies all the values inside
            /// [See](https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction)
            #[inline(always)]
            pub fn prod (self) -> f32 {
                unsafe {
                    let shuf = _mm_shuffle_ps(self.0, self.0, _MM_SHUFFLE(2, 3, 0, 1));
                    let sums = _mm_mul_ps(self.0, shuf);
                    let shuf = _mm_movehl_ps(shuf, sums);
                    let sums = _mm_mul_ss(sums, shuf);
                    _mm_cvtss_f32(sums)
                }
            }
        }
    }
}