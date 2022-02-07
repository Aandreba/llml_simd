#![feature(concat_idents, exclusive_range_pattern)]
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
            if #[cfg(target = "use_naive")] {
                $(pub use crate::naive::$i;)*
            } else if #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse"))] {
                $(pub use crate::x86::$i;)*
            } else if #[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"))] {
                $(pub use crate::arm::$i;)*
            } else if #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))] {
                $(pub use crate::wasm::$i;)*
            } else {
                $(pub use crate::naive::$i;)*
            }
        }
    };
}

cfg_if! {
    if #[cfg(target = "use_naive")] {
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
        mod naive;
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