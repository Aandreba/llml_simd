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

macro_rules! _mm_concat {
    ($fun:ident, f32,) => {
        _mm_concat!(@internal $fun, s, m)
    };

    ($fun:ident, f32, $tag:ident) => {
        _mm_concat!(@internal $fun, s, $tag)
    };

    ($fun:ident, f64,) => {
        _mm_concat!(@internal $fun, d, m)
    };

    ($fun:ident, f64, $tag:ident) => {
        _mm_concat!(@internal $fun, d, $tag)
    };

    (@internal $fun:ident, $label:ident, $tag:ident) => {
        concat_idents!(_m, $tag, _, $fun, _p, $label)
    }
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

macro_rules! impl_straight {
    (@arith $target:ident, $ty:ident, $($trait:ident, $fun:ident, $($tag:ident)?),+) => {
        $(
            impl $trait for $target {
                type Output = Self;

                #[inline(always)]
                fn $fun (self, rhs: Self) -> Self::Output {
                    unsafe { Self(_mm_concat!($fun, $ty, $($tag)?)(self.0, rhs.0)) }
                }
            }

            impl_scal_arith!($target, $ty, $trait, $fun);
        )*
    };

    ($($og:ident as $target:ident $(with $tag:ident)? => [$ty:ident;$len:literal]),+) => {
        $(
            #[allow(non_camel_case_types)]
            #[repr(transparent)]
            #[derive(Copy, Assign)]
            #[assign_targets(Add, Sub, Mul, Div)]
            #[assign_rhs(Self, $ty)]
            pub struct $target($og);
            impl_straight!(
                @arith $target, $ty,
                Add, add, $($tag)?,
                Sub, sub, $($tag)?,
                Mul, mul, $($tag)?,
                Div, div, $($tag)?
            );

            impl Neg for $target {
                type Output = Self;

                #[inline(always)]
                fn neg (self) -> Self::Output {
                    unsafe { Self(_mm_concat!(sub, $ty, $($tag)?)(_mm_concat!(setzero, $ty, $($tag)?)(), self.0)) }
                }
            }

            impl PartialEq for $target {
                #[inline(always)]
                fn eq (&self, rhs: &Self) -> bool {
                    let cmp : u128 = unsafe { transmute(_mm_concat!(cmpeq, $ty, $($tag)?)(self.0, rhs.0)) };
                    cmp == u128::MAX
                }
            }

            impl $target {
                const ABS_MASK : $og = unsafe { transmute(abs_mask!($ty, 128)) };

                /// Loads values from the pointer into the SIMD vector
                #[inline(always)]
                pub unsafe fn load (ptr: *const $ty) -> Self {
                    let reverse = arr![|i| *ptr.add($len - i); $len];
                    Self(_mm_concat!(loadu, $ty, $($tag)?)(addr_of!(reverse).cast()))
                }

                #[doc=concat!("Returns a vector with the absolute values of the original vector")]
                #[inline(always)]
                pub fn abs (self) -> Self {
                    unsafe { Self(_mm_concat!(and, $ty, $($tag)?)(Self::ABS_MASK, self.0)) }
                }

                #[doc=concat!("Returns a vector with the absolute values of the original vector")]
                #[inline(always)]
                pub fn sqrt (self) -> Self {
                    unsafe { Self(_mm_concat!(sqrt, $ty, $($tag)?)(self.0)) }
                }

                impl_other_fns_straight!(
                    $ty,
                    min as vmin $(with $tag)?: "smallest/minimum value",
                    max as vmax $(with $tag)?: "biggest/maximum value"
                );

                impl_hoz_fns_straight!(
                    $ty,
                    min $(with $tag)?: "Gets the smallest/minimum value of the vector",
                    max $(with $tag)?: "Gets the biggest/maximum value of the vector",
                    add as sum $(with $tag)?: "Sums up all the values inside the vector"
                );
            }

            impl From<$ty> for $target {
                #[inline(always)]
                fn from (x: $ty) -> Self {
                    unsafe { Self(_mm_concat!(set1, $ty, $($tag)?)(x)) }
                }
            }
        )*
    }
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