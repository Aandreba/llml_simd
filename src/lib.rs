#![feature(repr_simd, concat_idents, stdsimd, associated_type_bounds, core_intrinsics)]
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
    ($($i:ident),+) => {
        $(
            impl Simdable for $i {}
        )*
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
pub struct Simd<T> (pub(crate) T);

impl<T> Deref for Simd<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
} 

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
{}

// Packed singles
pub fn f32x1 (x: f32) -> Simd<f32> {
    Simd(x)
}

pub fn f64x1 (x: f64) -> Simd<f64> {
    Simd(x)
}