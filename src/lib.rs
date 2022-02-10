#![feature(concat_idents, exclusive_range_pattern)]
#![cfg_attr(target_feature = "sse", feature(stdarch, stdsimd))]
#![cfg_attr(target_arch = "wasm32", feature(simd_wasm64))]
#![no_std]

use cfg_if::cfg_if;
macro_rules! flat_mod {
    ($($i:ident),+) => {
        $(
            mod $i;
            pub use $i::*;
        )*
    };
}

macro_rules! import {
    ($($i:ident),+) => {
        cfg_if::cfg_if! {
            if #[cfg(target = "force_naive")] {
                $(pub use crate::naive::$i;)*
            } else if #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse"))] {
                $(pub use crate::x86::$i;)*
            } else if #[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"))] {
                $(pub use crate::arm::$i;)*
            } else if #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))] {
                $(pub use crate::wasm::$i;)*
            } else {
                $(pub use crate::wasm::$i;)*
                //$(pub use crate::naive::$i;)*
            }
        }
    };
}

macro_rules! impl_clone {
    ($($target:ident, $ty:ident, $len:literal),+) => {
        $(
            impl Clone for $target {
                #[inline(always)]
                fn clone(&self) -> Self {
                    unsafe { Self::load(self as *const Self as *const $ty) }
                }
            }
        )*
    };
}

include!("composite.rs");

cfg_if! {
    if #[cfg(target = "force_naive")] {
        mod naive;
    } else if #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse"))] {
        mod x86;
        flat_mod!(generics);
    } else if #[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"))] {
        mod arm;
        flat_mod!(generics);
    } else if #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))] {
        mod wasm;
        flat_mod!(generics);
    } else {
        mod wasm;
        flat_mod!(generics);
        //mod naive;
    }
}

/// Floating-point vectors
pub mod float {
    /// Single-precision floating point vectors
    pub mod single {
        import!(f32x2, f32x4, f32x6, f32x8, f32x10, f32x12, f32x14, f32x16);
    }

    /// Double-precision floating point vectors
    pub mod double {
        import!(f64x2, f64x4, f64x6, f64x8, f64x10, f64x12, f64x14, f64x16);
    }
}

/// Check current implementation
pub enum LlmlImpl {
    /// x86/x86_64 SSE (128-bit) implementation
    SSE,

    /// x86/x86_64 AVX (256-bit) implementation
    AVX,

    /// arm/aarch64 NEON (64-bit to 128-bit) implementation
    NEON,

    /// WASM32 SIMD128 proposal (128-bit) implementation
    WASM,

    /// Naive implementation with arrays. Useful as a backup if no other method is available
    NAIVE
}

impl LlmlImpl {
    pub const CURRENT : Self = current_impl();

    #[inline]
    pub const fn is_64bit (&self) -> bool {
        matches!(self, NEON)
    }

    #[inline]
    pub const fn is_128bit (&self) -> bool {
        match self {
            NAIVE => false,
            _ => true
        }
    }

    #[inline]
    pub const fn is_256bit (&self) -> bool {
        matches!(self, AVX)
    }
}

#[inline]
pub const fn current_impl () -> LlmlImpl {
    cfg_if! {
        if #[cfg(target = "force_naive")] {
            LlmlImpl::NAIVE
        } else if #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse"))] {
            #[cfg(all(feature = "use_avx", target_feature = "avx"))]
            LlmlImpl::AVX
            #[cfg(not(all(feature = "use_avx", target_feature = "avx")))]
            LlmlImpl::SSE
        } else if #[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"))] {
            LlmlImpl::NEON
        } else if #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))] {
            LlmlImpl::WASM
        } else {
            LlmlImpl::NAIVE
        }
    }
}