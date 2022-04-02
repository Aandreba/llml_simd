use cfg_if::cfg_if;
mod sse;
pub use self::sse::{f32x2};

cfg_if! {
    if #[cfg(all(feature = "use_avx", target_feature = "avx"))] {
        mod avx; 
        pub use self::avx::*;
    }
}