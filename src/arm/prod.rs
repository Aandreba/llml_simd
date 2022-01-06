macro_rules! impl_prod2 {
    ($($name:ident, $ty:ident),+) => {
        $(
            impl Simd<concat_idents!($name, x2_t)> {
                #[inline(always)]
                fn _prod (self) -> $ty {
                    self[0] * self[1]
                }
            }
        )*
    };
}

macro_rules! impl_prod4 {
    ($($name:ident, $ty:ident),+) => {
        $(
            impl Simd<concat_idents!($name, x4_t)> {
                #[inline(always)]
                fn _prod (self) -> $ty {
                    unsafe {
                        let ptr = &self as *const Self as *const $ty;
                        let alpha : concat_idents!($name, x2_t) = *(ptr as *const _);
                        let beta : concat_idents!($name, x2_t) = *(ptr.add(2) as *const _);
                        (Simd(alpha) * Simd(beta)).prod()
                    }
                }
            }
        )*
    };
}

/*
    let shuf = _mm_shuffle_ps(self.0, self.0, _MM_SHUFFLE(2, 3, 0, 1));
    let sums = _mm_add_ps(self.0, shuf);
    let shuf = _mm_movehl_ps(shuf, sums);
    let sums = _mm_add_ss(sums, shuf);
    _mm_cvtss_f32(sums)
*/