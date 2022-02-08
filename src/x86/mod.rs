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

macro_rules! impl_scal_arith {
    ($target:ident, $ty:ident, $($trait:ident, $fun:ident),+) => {
        $(
            impl $trait<$ty> for $target {
                type Output = Self;
    
                #[inline(always)]
                fn $fun (self, rhs: $ty) -> Self::Output {
                    self.$fun(Into::<$target>::into(rhs))
                }
            }

            impl $trait<$target> for $ty {
                type Output = $target;
    
                #[inline(always)]
                fn $fun (self, rhs: $target) -> Self::Output {
                    Into::<$target>::into(self).$fun(rhs)
                }
            }
        )*
    };
}

use cfg_if::cfg_if;
mod sse;

/*cfg_if! {
    if #[cfg(all(feature = "use_avx", target_feature = "avx"))] {
        flat_mod!(avx);
        pub use self::sse::{f32x2, f32x4, f64x2};
    } else {
        pub use self::sse::*;
    }
}*/

flat_mod!(avx);
pub use self::sse::{f32x2, f32x4, f64x2};