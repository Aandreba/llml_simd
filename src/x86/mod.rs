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

macro_rules! abs_mask {
    (f32, 128) => {
        [i32::MAX, i32::MAX, i32::MAX, i32::MAX]
    };

    (f64, 128) => {
        [i64::MAX, i64::MAX]
    };
}

macro_rules! impl_hoz_fns_straight {
    ($ty:ident, $($fun:ident $(as $name:ident)? with $tag:ident: $docs:expr),+) => {
        $(
            impl_hoz_fns_straight!(1, $fun $(,$name)?, $ty, $docs, $tag);
        )*
    };

    ($ty:ident, $($fun:ident $(as $name:ident)?: $docs:expr),+) => {
        $(
            impl_hoz_fns_straight!(1, $fun $(,$name)?, $ty, $docs, m);
        )*
    };

    (1, $fun:ident, $name:ident, f32, $docs:expr, $tag:ident) => {
        #[doc=$docs]
        #[inline(always)]
        pub fn $name (self) -> f32 {
            unsafe {
                #[cfg(target_feature = "sse3")]
                let shuf = _mm_conat!(movehdup, $ty, $tag)(self.0);
                #[cfg(not(target_feature = "sse3"))]
                let shuf = _mm_concat!(shuffle, f32, $tag)(self.0, self.0, _MM_SHUFFLE(2, 3, 0, 1));

                let sums = _mm_concat!($fun, f32, $tag)(self.0, shuf);
                let shuf = _mm_concat!(movehl, f32, $tag)(shuf, sums);
                let sums = _mm_concat!($fun, f32, $tag)(sums, shuf);

                impl_hoz_fns_straight!(@cvt f32, $tag)(sums)
            }
        }
    };

    (1, $fun:ident, $name:ident, f64, $docs:expr, $($tag:ident)?) => {
        #[doc=$docs]
        #[inline(always)]
        pub fn $name (self) -> f64 {
            self[0].$fun(self[1])
        }
    };

    (1, $fun:ident, $ty:ident, $docs:expr, $($tag:ident)?) => {
        impl_hoz_fns_straight!(1, $fun, $fun, $ty, $docs, $($tag)?);
    };

    (@cvt f32, $tag:ident) => {
        concat_idents!(_m, $tag, _cvtss_f32)
    };

    (@cvt f64, $tag:ident) => {
        concat_idents!(_m, $tag, _cvtsd_f64)
    };
}

macro_rules! impl_other_fns_straight {
    ($ty:ident, $($fun:ident $(as $name:ident)? $(with $tag:ident)?: $docs:expr),+) => {
        $(
            impl_other_fns_straight!(1, $fun $(,$name)?, $ty, $docs, $($tag)?);
        )*
    };

    (1, $fun:ident, $name:ident, $ty:ident, $docs:expr, $($tag:ident)?) => {
        #[doc=concat!("Returns a vector with the ", $docs, " of each lane")]
        #[inline(always)]
        pub fn $name (self, rhs: Self) -> Self {
            unsafe { Self(_mm_concat!($fun, $ty, $($tag)?)(self.0, rhs.0)) }
        }
    };

    (1, $fun:ident, $ty:ident, $docs:expr, $($tag:ident)?) => {
        impl_other_fns!(1, $fun, $fun, $ty, $docs, $($tag)?)
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