macro_rules! arch_use {
    () => {
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "x86")] {
                use core::arch::x86::*;
            } else {
                use core::arch::x86_64::*;
            }
        }
    };
}

use cfg_if::cfg_if;
mod sse;
pub use self::sse::{f32x2, f32x4, f64x2};

cfg_if! {
    if #[cfg(all(feature = "use_avx", target_feature = "avx"))] {
        mod avx;
        pub use self::avx::*;
    } else {
        mod avx;
        pub use self::avx::*;
        //pub use self::sse::*;
    }
}