#![feature(repr_simd, concat_idents, stdsimd, stdarch)]
use std::ops::Deref;
use cfg_if::cfg_if;

macro_rules! flat_mod {
    ($($i:ident),+) => {
        $(
            mod $i;
            pub use $i::*;
        )*
    };
}

flat_mod!(simd);
cfg_if! {
    if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        flat_mod!(x86);
    }   
}

#[derive(Debug, Clone, Copy)]
pub struct Simd<T> (pub(crate) T);

impl<T> Deref for Simd<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
} 

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