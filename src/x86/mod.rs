use std::ops::{Add, Sub, Mul, Div, Neg, Index, IndexMut};
use cfg_if::cfg_if;
use crate::{Simd, Simdt, SimdType, SimdTypeX86};
use crate::Simdable;

macro_rules! use_x86 {
    () => {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;

        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;
    };
}

use_x86!();

macro_rules! simd_map {
    ($($name:ident, $type:ident, $sub:ident, $post:ident),+) => {
        $(
            simd_map!($name, $type);
            simd_map_arith!(
                $name, $type, $sub, $post,
                Add, add,
                Sub, sub,
                Mul, mul,
                Div, div
            );

            impl Neg for Simd<$name> {
                type Output = Self;

                #[inline(always)]
                fn neg (self) -> Self {
                    unsafe { Simd(concat_idents!(_, $sub, _sub_, $post)(concat_idents!(_, $sub, _, setzero, _, $post)(), self.0)) }
                }
            }
        )*
    };

    ($name:ident, $type:ident) => {
        impl Index<usize> for Simd<$name> {
            type Output = $type;

            #[inline(always)]
            fn index (&self, idx: usize) -> &Self::Output {
                unsafe { &*(self as *const Self as *const $type).add(idx) }
            }
        }

        impl IndexMut<usize> for Simd<$name> {
            #[inline(always)]
            fn index_mut (&mut self, idx: usize) -> &mut Self::Output {
                unsafe { &mut *(self as *mut Self as *mut $type).add(idx) }
            }
        }
    };
}

macro_rules! simd_map_arith {
    ($target:ident, $type:ident, $sub:ident, $post:ident, $($trait:ident, $fun:ident),+) => {
        $(
            impl $trait for Simd<$target> {
                type Output = Self;

                #[inline(always)]
                fn $fun (self, rhs: Self) -> Self::Output {
                    unsafe { Simd(concat_idents!(_, $sub, _, $fun, _, $post)(self.0, rhs.0)) }
                }
            }

            impl $trait<$type> for Simd<$target> {
                type Output = Self;

                #[inline(always)]
                fn $fun (self, rhs: $type) -> Self::Output {
                    unsafe {
                        Simd(concat_idents!(_, $sub, _, $fun, _, $post)(
                            self.0,
                            concat_idents!(_, $sub, _set1_, $post)(rhs)
                        )) 
                    }
                }
            }

            impl $trait<Simd<$target>> for Simd<$type> {
                type Output = Simd<$target>;

                #[inline(always)]
                fn $fun (self, rhs: Self::Output) -> Self::Output {
                    unsafe {
                        Simd(concat_idents!(_, $sub, _, $fun, _, $post)(
                            concat_idents!(_, $sub, _set1_, $post)(self.0),
                            rhs.0
                        )) 
                    }
                }
            }
        )*
    };
}

flat_mod!(f32x2, f32x4, f32x8);
simd_map!(__m128, f32, mm, ps);

#[inline(always)]
pub fn f32x2 (a: [f32;2]) -> Simd<__m64> {
    __m64::new(a[1], a[0])
}

#[inline(always)]
pub fn f32x4 (a: [f32;4]) -> Simd<__m128> {
    unsafe { Simd(_mm_set_ps(a[3], a[2], a[1], a[0])) }
}

cfg_if! {
    if #[cfg(target_feature = "sse2")] {
        simd_map!(__m128d, f64, mm, pd);
        flat_mod!(f64x2);

        #[inline(always)]
        pub fn f64x2 (a: [f64;2]) -> Simd<__m128d> {
            unsafe { Simd(_mm_set_pd(a[1], a[0])) }
        }
    }
}

cfg_if! {
    if #[cfg(any(feature = "force-avx", target_feature = "avx"))] {
        simd_map!(
            __m256, f32, mm256, ps,
            __m256d, f64, mm256, pd
        );

        flat_mod!(f64x4);

        #[inline(always)]
        pub fn f32x8 (a: [f32;8]) -> Simd<__m256> {
            unsafe { Simd(_mm256_set_ps(a[7], a[6], a[5], a[4], a[3], a[2], a[1], a[0])) }
        }

        #[inline(always)]
        pub fn f64x4 (a: [f64;4]) -> Simd<__m256d> {
            unsafe { Simd(_mm256_set_pd(a[3], a[2], a[1], a[0])) }
        }
    } else {
        #[inline(always)]
        pub fn f32x8 (a: [f32;8]) -> Simd<[Simd<__m128>;2]> {
            unsafe {
                Simd([
                    Simd(_mm_set_ps(a[3], a[2], a[1], a[0])),
                    Simd(_mm_set_ps(a[7], a[6], a[5], a[4])) 
                ]) 
            }
        }

        #[cfg(target_feature = "sse2")]
        #[inline(always)]
        pub fn f64x4 (a: [f64;4]) -> Simd<[Simd<__m128d>;2]> {
            unsafe { Simd([Simd(_mm_set_pd(a[1], a[0])), Simd(_mm_set_pd(a[3], a[2]))]) }
        }
    }
}

cfg_if! {
    if #[cfg(any(feature = "force-avx512", target_feature = "avx512f"))] {
        simd_map!(
            __m512, f32, mm512, ps,
            __m512d, f64, mm512, pd
        );

        flat_mod!(f32x16, f64x8);

        #[inline(always)]
        pub fn f32x16 (a: [f32;16]) -> Simd<__m512> {
            unsafe { 
                Simd(_mm512_set_ps(
                    a[15], a[14], a[13], a[12], 
                    a[11], a[10], a[9], a[8],
                    a[7], a[6], a[5], a[4], 
                    a[3], a[2], a[1], a[0]
                )) 
            }
        }

        #[inline(always)]
        pub fn f64x8 (a: [f64;8]) -> Simd<[Simd<__m128>;2]> {
            unsafe { 
                Simd(_mm512_set_pd(
                    a[7], a[6], a[5], a[4], 
                    a[3], a[2], a[1], a[0]
                )) 
            }
        }

    } else if #[cfg(any(feature = "force-avx", target_feature = "avx"))] {
        #[inline(always)]
        pub fn f32x16 (a: [f32;16]) -> Simd<[Simd<__m256>;2]> {
            unsafe { 
                Simd([
                    Simd(_mm256_set_ps(a[7], a[6], a[5], a[4], a[3], a[2], a[1], a[0])),
                    Simd(_mm256_set_ps(a[15], a[14], a[13], a[12], a[11], a[10], a[9], a[8]))
                ]) 
            }
        }

        #[inline(always)]
        pub fn f64x8 (a: [f64;8]) -> Simd<[Simd<__m256d>;2]> {
            unsafe { 
                Simd([
                    Simd(_mm256_set_pd(a[3], a[2], a[1], a[0])),
                    Simd(_mm256_set_pd(a[7], a[6], a[5], a[4]))
                ])
            }
        }
    } else {
        #[inline(always)]
        pub fn f32x16 (a: [f32;16]) -> Simd<[Simd<__m128>; 4]> {
            unsafe { 
                Simd([
                    Simd(_mm_set_ps(a[3], a[2], a[1], a[0])),
                    Simd(_mm_set_ps(a[7], a[6], a[5], a[4])),
                    Simd(_mm_set_ps(a[11], a[10], a[9], a[8])),
                    Simd(_mm_set_ps(a[15], a[14], a[13], a[12]))
                ]) 
            }
        }

        #[cfg(target_feature = "sse2")]
        #[inline(always)]
        pub fn f64x8 (a: [f64;8]) -> Simd<[Simd<__m128d>; 4]> {
            unsafe { 
                Simd([
                    Simd(_mm_set_pd(a[1], a[0])),
                    Simd(_mm_set_pd(a[3], a[2])),
                    Simd(_mm_set_pd(a[5], a[4])),
                    Simd(_mm_set_pd(a[7], a[6]))
                ])
            }
        }
    }
}


// REMAINING CONFIG
cfg_if! {
    if #[cfg(any(feature = "force-avx512", target_feature = "avx512f"))] {
        simdable!(f32, __m64, __m128, __m256, __m512, [Simd<__m512d>;2]);
        simdable!(f64, __m128d, __m256d, __m512d, [Simd<__m512d>;2], [Simd<__m512d>;4]);

        // TODO X32, X64
    } else if #[cfg(any(feature = "force-avx", target_feature = "avx"))] {
        simdable!(f32, __m64, __m128, __m256, [Simd<__m256>;2], [Simd<__m256>;4]);
        simdable!(f64, __m128d, __m256d, [Simd<__m256d>;2], [Simd<__m256d>;4], [Simd<__m256d>;8]);

        #[inline(always)]
        pub fn f32x32 (a: [f32;32]) -> Simd<[Simd<__m256>;4]> {
            Simd([
                f32x8(a[..8].try_into().unwrap()),
                f32x8(a[8..16].try_into().unwrap()),
                f32x8(a[16..24].try_into().unwrap()),
                f32x8(a[24..].try_into().unwrap()),
            ])
        }

        #[inline(always)]
        pub fn f32x64 (a: [f32;64]) -> Simd<[Simd<__m256>;8]> {
            Simd([
                f32x8(a[..8].try_into().unwrap()),
                f32x8(a[8..16].try_into().unwrap()),
                f32x8(a[16..24].try_into().unwrap()),
                f32x8(a[24..32].try_into().unwrap()),
                f32x8(a[32..40].try_into().unwrap()),
                f32x8(a[40..48].try_into().unwrap()),
                f32x8(a[48..56].try_into().unwrap()),
                f32x8(a[56..].try_into().unwrap())
            ])
        }

        #[inline(always)]
        pub fn f64x16 (a: [f64;16]) -> Simd<[Simd<__m256d>;4]> {
            Simd([
                f64x4(a[..4].try_into().unwrap()),
                f64x4(a[4..8].try_into().unwrap()),
                f64x4(a[8..12].try_into().unwrap()),
                f64x4(a[12..].try_into().unwrap()),
            ])
        }

        #[inline(always)]
        pub fn f64x32 (a: [f64;32]) -> Simd<[Simd<__m256d>;8]> {
            Simd([
                f64x4(a[..4].try_into().unwrap()),
                f64x4(a[4..8].try_into().unwrap()),
                f64x4(a[8..12].try_into().unwrap()),
                f64x4(a[12..16].try_into().unwrap()),
                f64x4(a[16..20].try_into().unwrap()),
                f64x4(a[20..24].try_into().unwrap()),
                f64x4(a[24..28].try_into().unwrap()),
                f64x4(a[28..].try_into().unwrap()),
            ])
        }
    } else {
        simdable!(f32, __m64, __m128, [Simd<__m128>;2], [Simd<__m128>;4], [Simd<__m128>;8]);

        #[inline(always)]
        pub fn f32x32 (a: [f32;32]) -> Simd<[Simd<__m128>;8]> {
            Simd([
                f32x4(a[..4].try_into().unwrap()),
                f32x4(a[4..8].try_into().unwrap()),
                f32x4(a[8..12].try_into().unwrap()),
                f32x4(a[12..16].try_into().unwrap()),
                f32x4(a[16..20].try_into().unwrap()),
                f32x4(a[20..24].try_into().unwrap()),
                f32x4(a[24..28].try_into().unwrap()),
                f32x4(a[28..].try_into().unwrap()),
            ])
        }

        #[inline(always)]
        pub fn f32x64 (a: [f32;64]) -> Simd<[Simd<__m128>;16]> {
            Simd([
                f32x4(a[..4].try_into().unwrap()),
                f32x4(a[4..8].try_into().unwrap()),
                f32x4(a[8..12].try_into().unwrap()),
                f32x4(a[12..16].try_into().unwrap()),
                f32x4(a[16..20].try_into().unwrap()),
                f32x4(a[20..24].try_into().unwrap()),
                f32x4(a[24..28].try_into().unwrap()),
                f32x4(a[28..32].try_into().unwrap()),
                f32x4(a[32..36].try_into().unwrap()),
                f32x4(a[36..40].try_into().unwrap()),
                f32x4(a[40..44].try_into().unwrap()),
                f32x4(a[44..48].try_into().unwrap()),
                f32x4(a[48..52].try_into().unwrap()),
                f32x4(a[52..56].try_into().unwrap()),
                f32x4(a[56..60].try_into().unwrap()),
                f32x4(a[60..].try_into().unwrap()),
            ])
        }

        cfg_if! {
            if #[cfg(target_feature = "sse2")] {
                simdable!(f64, __m128d, [Simd<__m128d>;2], [Simd<__m128d>;4], [Simd<__m128d>;8], [Simd<__m128d>;16]);

                #[inline(always)]
                pub fn f64x16 (a: [f64;16]) -> Simd<[Simd<__m128d>;8]> {
                    Simd([
                        f64x2(a[..2].try_into().unwrap()),
                        f64x2(a[2..4].try_into().unwrap()),
                        f64x2(a[4..6].try_into().unwrap()),
                        f64x2(a[6..8].try_into().unwrap()),
                        f64x2(a[8..10].try_into().unwrap()),
                        f64x2(a[10..12].try_into().unwrap()),
                        f64x2(a[12..14].try_into().unwrap()),
                        f64x2(a[14..].try_into().unwrap()),
                    ])
                }

                #[inline(always)]
                pub fn f64x32 (a: [f64;32]) -> Simd<[Simd<__m128d>;16]> {
                    Simd([
                        f64x2(a[..2].try_into().unwrap()),
                        f64x2(a[2..4].try_into().unwrap()),
                        f64x2(a[4..6].try_into().unwrap()),
                        f64x2(a[6..8].try_into().unwrap()),
                        f64x2(a[8..10].try_into().unwrap()),
                        f64x2(a[10..12].try_into().unwrap()),
                        f64x2(a[12..14].try_into().unwrap()),
                        f64x2(a[14..16].try_into().unwrap()),
                        f64x2(a[16..18].try_into().unwrap()),
                        f64x2(a[18..20].try_into().unwrap()),
                        f64x2(a[20..22].try_into().unwrap()),
                        f64x2(a[22..24].try_into().unwrap()),
                        f64x2(a[24..26].try_into().unwrap()),
                        f64x2(a[26..28].try_into().unwrap()),
                        f64x2(a[28..30].try_into().unwrap()),
                        f64x2(a[30..].try_into().unwrap()),
                    ])
                }
            }
        }
    }
}

cfg_if! {
    if #[cfg(any(feature = "force-avx512", target_feature = "avx512f"))] {
        /// Returns the newest architecture feature available to the compiler
        #[inline(always)]
        pub fn get_top_feature () -> SimdType {
            SimdType::X86(SimdTypeX86::AVX512)
        }
    } else if #[cfg(any(feature = "force-avx2", target_feature = "avx2"))] {
        /// Returns the newest architecture feature available to the compiler
        #[inline(always)]
        pub fn get_top_feature () -> SimdType {
            SimdType::X86(SimdTypeX86::AVX2)
        }
    } else if #[cfg(any(feature = "force-avx", target_feature = "avx"))] {
        /// Returns the newest architecture feature available to the compiler
        #[inline(always)]
        pub fn get_top_feature () -> SimdType {
            SimdType::X86(SimdTypeX86::AVX)
        }
    } else if #[cfg(any(feature = "force-sse3", target_feature = "sse3"))] {
        /// Returns the newest architecture feature available to the compiler
        #[inline(always)]
        pub fn get_top_feature () -> SimdType {
            SimdType::X86(SimdTypeX86::SSE3)
        }
    } else if #[cfg(target_feature = "sse2")] {
        /// Returns the newest architecture feature available to the compiler
        #[inline(always)]
        pub fn get_top_feature () -> SimdType {
            SimdType::X86(SimdTypeX86::SSE2)
        }
    } else {
        /// Returns the newest architecture feature available to the compiler
        #[inline(always)]
        pub fn get_top_feature () -> SimdType {
            SimdType::X86(SimdTypeX86::SSE)
        }
    }
}