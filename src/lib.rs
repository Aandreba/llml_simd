#![feature(repr_simd, concat_idents, stdsimd, associated_type_bounds, core_intrinsics, generic_const_exprs)]

use std::ops::{Deref, Add, Sub, Mul, Div};
use cfg_if::cfg_if;
use utils::Ordx;
pub mod utils;

macro_rules! flat_mod {
    ($($i:ident),+) => {
        $(
            mod $i;
            pub use $i::*;
        )*
    };
}

macro_rules! simdable {
    ($i:ident, $x2:ty, $x4:ty, $x8:ty, $x16:ty, $x32:ty) => {
        impl Simdable for $i {
            type X2 = Simd<$x2>;
            type X4 = Simd<$x4>;
            type X8 = Simd<$x8>;
            type X16 = Simd<$x16>;
            type X32 = Simd<$x32>;

            #[inline(always)]
            fn x2 (a: [Self;2]) -> Self::X2 {
                concat_idents!($i, x2)(a)
            }

            #[inline(always)]
            fn x4 (a: [Self;4]) -> Self::X4 {
                concat_idents!($i, x4)(a)
            }

            #[inline(always)]
            fn x8 (a: [Self;8]) -> Self::X8 {
                concat_idents!($i, x8)(a)
            }

            #[inline(always)]
            fn x16 (a: [Self;16]) -> Self::X16 {
                concat_idents!($i, x16)(a)
            }

            #[inline(always)]
            fn x32 (a: [Self;32]) -> Self::X32 {
                concat_idents!($i, x32)(a)
            }
        }
    };
}

flat_mod!(simd);
cfg_if! {
    if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        flat_mod!(x86);
    } else if #[cfg(any(target_arch = "arm", target_arch = "aarch64"))] {
        flat_mod!(arm);
    }
}

// SIMD Wrapper
#[derive(Debug, Clone, Copy)]
pub struct Simd<T> (pub T);

// COMPILE FEATURES
#[derive(Debug)]
pub enum SimdType {
    X86 (SimdTypeX86),
    ARM
}

#[derive(Debug)]
#[repr(u8)]
pub enum SimdTypeX86 {
    SSE,
    SSE2,
    SSE3,
    SSSE3,
    AVX,
    AVX2,
    AVX512
}

// SIMDABLE PRIMITIVES
pub trait Simdable where
    Self:
        Sized + Ordx +
        Add<Output = Self> +
        Sub<Output = Self> +
        Mul<Output = Self> +
        Div<Output = Self> +
{
    type X2: Simdt<Item = Self>;
    type X4: Simdt<Item = Self>;
    type X8: Simdt<Item = Self>;
    type X16: Simdt<Item = Self>;
    type X32: Simdt<Item = Self>;

    fn x2 (a: [Self;2]) -> Self::X2;
    fn x4 (a: [Self;4]) -> Self::X4;
    fn x8 (a: [Self;8]) -> Self::X8;
    fn x16 (a: [Self;16]) -> Self::X16;
    fn x32 (a: [Self;32]) -> Self::X32;
}

// Packed singles
pub fn f32x1 (x: f32) -> Simd<f32> {
    Simd(x)
}

pub fn f64x1 (x: f64) -> Simd<f64> {
    Simd(x)
}