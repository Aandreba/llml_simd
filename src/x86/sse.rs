use core::ops::*;
use core::mem::transmute;
use core::ptr::addr_of;
use llml_simd_proc::*;
use derive_more::*;
arch_use!();

macro_rules! _mm_concat {
    ($fun:ident, f32) => {
        _mm_concat!(@internal $fun, s)
    };

    ($fun:ident, f64) => {
        _mm_concat!(@internal $fun, d)
    };

    (@internal $fun:ident, $label:ident) => {
        concat_idents!(_mm_, $fun, _p, $label)
    }
}

macro_rules! abs_mask {
    (f32) => {
        [i32::MAX, i32::MAX, i32::MAX, i32::MAX]
    };

    (f64) => {
        [i64::MAX, i64::MAX]
    };
}

macro_rules! impl_hoz_fns_straight {
    ($ty:ident, $($fun:ident: $docs:expr),+) => {
        $(
            impl_hoz_fns_straight!(1, $fun, $fun, $ty, $docs);
        )*
    };
    
    ($ty:ident, $($fun:ident as $name:ident: $docs:expr),+) => {
        $(
            impl_hoz_fns_straight!(1, $fun, $name, $ty, $docs);
        )*
    };

    (1, $fun:ident, $name:ident, f32, $docs:expr) => {
        #[doc=$docs]
        #[inline(always)]
        pub fn $name (self) -> f32 {
            unsafe {
                #[cfg(target_feature = "sse3")]
                let shuf = _mm_concat!(movehdup, f32)(self.0);
                #[cfg(not(target_feature = "sse3"))]
                let shuf = _mm_concat!(shuffle, f32)(self.0, self.0, _MM_SHUFFLE(2, 3, 0, 1));

                let sums = _mm_concat!($fun, f32)(self.0, shuf);
                let shuf = _mm_concat!(movehl, f32)(shuf, sums);
                let sums = _mm_concat!($fun, f32)(sums, shuf);

                _mm_cvtss_f32(sums)
            }
        }
    };

    (1, $fun:ident, $name:ident, f64, $docs:expr) => {
        #[doc=$docs]
        #[inline(always)]
        pub fn $name (self) -> f64 {
            self[0].$fun(self[1])
        }
    };

    (1, $fun:ident, $ty:ident, $docs:expr, $($tag:ident)?) => {
        impl_hoz_fns_straight!(1, $fun, $fun, $ty, $docs);
    };
}

macro_rules! impl_other_fns_straight {
    ($ty:ident, $($fun:ident $(as $name:ident)?: $docs:expr),+) => {
        $(
            impl_other_fns_straight!(1, $fun $(,$name)?, $ty, $docs);
        )*
    };

    (1, $fun:ident, $name:ident, $ty:ident, $docs:expr) => {
        #[doc=concat!("Returns a vector with the ", $docs, " of each lane")]
        #[inline(always)]
        pub fn $name (self, rhs: Self) -> Self {
            unsafe { Self(_mm_concat!($fun, $ty)(self.0, rhs.0)) }
        }
    };

    (1, $fun:ident, $ty:ident, $docs:expr) => {
        impl_other_fns!(1, $fun, $fun, $ty, $docs, $($tag)?)
    };
}

macro_rules! impl_straight {
    (@arith $target:ident, $ty:ident, $($trait:ident, $fun:ident),+) => {
        $(
            impl $trait for $target {
                type Output = Self;

                #[inline(always)]
                fn $fun (self, rhs: Self) -> Self::Output {
                    unsafe { Self(_mm_concat!($fun, $ty)(self.0, rhs.0)) }
                }
            }

            impl_scal_arith!($target, $ty, $trait, $fun);
        )*
    };

    ($($og:ident as $target:ident => [$ty:ident;$len:literal]),+) => {
        $(
            #[allow(non_camel_case_types)]
            #[repr(transparent)]
            #[derive(Copy, Assign)]
            #[assign_targets(Add, Sub, Mul, Div)]
            #[assign_rhs(Self, $ty)]
            pub struct $target(pub(crate) $og);
            impl_straight!(
                @arith $target, $ty,
                Add, add,
                Sub, sub,
                Mul, mul,
                Div, div
            );

            impl Neg for $target {
                type Output = Self;

                #[inline(always)]
                fn neg (self) -> Self::Output {
                    unsafe { Self(_mm_concat!(sub, $ty)(_mm_concat!(setzero, $ty)(), self.0)) }
                }
            }

            impl PartialEq for $target {
                #[inline(always)]
                fn eq (&self, rhs: &Self) -> bool {
                    let cmp : u128 = unsafe { transmute(_mm_concat!(cmpeq, $ty)(self.0, rhs.0)) };
                    cmp == u128::MAX
                }
            }

            impl $target {
                const ABS_MASK : $og = unsafe { transmute(abs_mask!($ty)) };

                /// Loads values from the pointer into the SIMD vector
                #[inline(always)]
                pub unsafe fn load (ptr: *const $ty) -> Self {
                    let reverse = arr![|i| *ptr.add($len - 1 - i); $len];
                    Self(_mm_concat!(loadu, $ty)(addr_of!(reverse).cast()))
                }

                #[doc=concat!("Returns a vector with the absolute values of the original vector")]
                #[inline(always)]
                pub fn abs (self) -> Self {
                    unsafe { Self(_mm_concat!(and, $ty)(Self::ABS_MASK, self.0)) }
                }

                #[doc=concat!("Returns a vector with the absolute values of the original vector")]
                #[inline(always)]
                pub fn sqrt (self) -> Self {
                    unsafe { Self(_mm_concat!(sqrt, $ty)(self.0)) }
                }

                impl_other_fns_straight!(
                    $ty,
                    min as vmin: "smallest/minimum value",
                    max as vmax: "biggest/maximum value"
                );

                impl_hoz_fns_straight!(
                    $ty,
                    min: "Gets the smallest/minimum value of the vector",
                    max: "Gets the biggest/maximum value of the vector"
                );

                impl_hoz_fns_straight!(
                    $ty,
                    add as sum: "Sums up all the values inside the vector"
                );
            }

            impl From<$ty> for $target {
                #[inline(always)]
                fn from (x: $ty) -> Self {
                    unsafe { Self(_mm_concat!(set1, $ty)(x)) }
                }
            }
        )*
    }
}

impl_straight!(
    __m128 as f32x2 => [f32;2],
    __m128 as f32x4 => [f32;4],
    __m128d as f64x2 => [f64;2]
);

impl_composite!(
    (f32x4 => 4, f32x2 => 2) as f32x6: f32,
    (f32x4 => 4, f32x4 => 4) as f32x8: f32,
    (f64x2 => 2, f64x2 => 2) as f64x4: f64,
    (f64x4 => 4, f64x2 => 2) as f64x6: f64,
    (f64x4 => 4, f64x4 => 4) as f64x8: f64,
    (f64x6 => 6, f64x4 => 4) as f64x10: f64,
    (f64x6 => 6, f64x6 => 6) as f64x12: f64,
    (f64x8 => 8, f64x6 => 6) as f64x14: f64,
    (f64x8 => 8, f64x8 => 8) as f64x16: f64
);

impl_composite!(
    (f32x4 => 4, f32x4 => 4, f32x2 => 2) as f32x10: f32,
    (f32x4 => 4, f32x4 => 4, f32x4 => 4) as f32x12: f32
);

impl_composite!(
    (f32x4 => 4, f32x4 => 4, f32x4 => 4, f32x2 => 2) as f32x14: f32,
    (f32x4 => 4, f32x4 => 4, f32x4 => 4, f32x4 => 4) as f32x16: f32
);