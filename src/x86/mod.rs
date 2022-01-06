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
        )*
    };
}

flat_mod!(f32x4, f32x8);
simd_map!(__m128, f32, mm, ps);
simdable!(f32);

#[inline(always)]
pub fn f32x2 (x: f32, y: f32) -> Simd<__m128> {
    unsafe { Simd(_mm_set_ps(x, y, 0., 0.)) }
}

#[inline(always)]
pub fn f32x4 (x: f32, y: f32, z: f32, w: f32) -> Simd<__m128> {
    unsafe { Simd(_mm_set_ps(x, y, z, w)) }
}

cfg_if! {
    if #[cfg(target_feature = "sse2")] {
        simd_map!(__m128d, f64, mm, pd);
        simdable!(f64);
        flat_mod!(f64x2);

        #[inline(always)]
        pub fn f64x2 (x: f64, y: f64) -> Simd<__m128d> {
            unsafe { Simd(_mm_set_pd(x, y)) }
        }
    }
}

cfg_if! {
    if #[cfg(any(feature = "force-avx", target_feature = "avx"))] {
        simd_map!(
            __m256, f32, mm256, ps,
            __m256d, f64, mm256, pd
        );

        #[inline(always)]
        pub fn f32x8 (
            x: f32, y: f32, z: f32, w: f32,
            a: f32, b: f32, c: f32, d: f32
        ) -> Simd<__m256> {
            unsafe { Simd(_mm256_set_ps(x, y, z, w, a, b, c, d)) }
        }

        #[inline(always)]
        pub fn f64x4 (x: f64, y: f64, z: f64, w: f64) -> Simd<__m256d> {
            unsafe { Simd(_mm256_set_pd(x, y, z, w)) }
        }
    } else {
        #[inline(always)]
        pub fn f32x8 (
            x: f32, y: f32, z: f32, w: f32,
            a: f32, b: f32, c: f32, d: f32
        ) -> Simd<[Simd<__m128>;2]> {
            unsafe { Simd([_mm_set_ps(x, y, z, w), _mm_set_ps(a, b, c, d)]) }
        }
    }
}

cfg_if! {
    if #[cfg(any(feature = "force-avx512", target_feature = "avx512f"))] {
        simd_map!(
            __m512, f32, mm512, ps,
            __m512d, f64, mm512, pd
        );

        #[inline(always)]
        pub fn f32x16 (
            x: f32, y: f32, z: f32, w: f32,
            a: f32, b: f32, c: f32, d: f32,
            x1: f32, y1: f32, z1: f32, w1: f32,
            a1: f32, b1: f32, c1: f32, d1: f32
        ) -> Simd<__m512> {
            unsafe { 
                Simd(_mm512_set_ps(
                    x, y, z, w, a, b, c, d,
                    x1, y1, z1, w1, a1, b1, c1, d1
                )) 
            }
        }

        #[inline(always)]
        pub fn f64x8 (
            x: f64, y: f64, z: f64, w: f64,
            a: f64, b: f64, c: f64, d: f64
        ) -> Simd<__m512d> {
            unsafe { Simd(_mm512_set_pd(x, y, z, w, a, b, c, d)) }
        }
    }
}

cfg_if! {
    if #[cfg(any(feature = "force-avx", target_feature = "avx"))] {
        /// Returns the newest architecture feature available to the compiler
        pub fn get_top_feature () -> SimdType {
            SimdType::X86(SimdTypeX86::AVX)
        }
    } else if #[cfg(any(feature = "force-sse3", target_feature = "sse3"))] {
        /// Returns the newest architecture feature available to the compiler
        pub fn get_top_feature () -> SimdType {
            SimdType::x86(SimdTypeX86::SSE3)
        }
    } else {
        /// Returns the newest architecture feature available to the compiler
        pub fn get_top_feature () -> SimdType {
            SimdType::x86(SimdTypeX86::SSE)
        }
    }
}