use core::ops::*;
use core::mem::transmute;
use core::ptr::addr_of;
use llml_simd_proc::*;
use crate::float::single::*;
use crate::float::double::*;
use derive_more::Neg;
arch_use!();

macro_rules! _mm_concat {
    ($fun:ident, f32) => {
        _mm_concat!(@internal $fun, s)
    };

    ($fun:ident, f64) => {
        _mm_concat!(@internal $fun, d)
    };

    (@internal $fun:ident, $label:ident) => {
        concat_idents!(_mm256_, $fun, _p, $label)
    }
}

macro_rules! abs_mask {
    (f32) => {
        [i32::MAX;8]
    };

    (f64) => {
        [i64::MAX;4]
    };
}

macro_rules! impl_hoz_fns_straight {
    (f32, $($fun:ident as $name:ident: $docs:expr),+) => {
        $(
            #[doc=$docs]
            #[inline(always)]
            pub fn $name (self) -> f32 {
                unsafe {
                    let vlow  = _mm256_castps256_ps128(self.0);
                    let vhigh = _mm256_extractf128_ps(self.0, 1); // high 128
                    let v = f32x4(concat_idents!(_mm_, $fun, _ps)(vlow, vhigh));
                    return v.$name();
                }
            }
        )*
    };

    (f64, $($fun:ident as $name:ident: $docs:expr),+) => {
        $(
            #[doc=$docs]
            #[inline(always)]
            pub fn $name (self) -> f64 {
                unsafe {
                    let shuf = _mm256_shuffle_pd(self.0, self.0, _MM_SHUFFLE(2, 3, 0, 1));
                    let sums = _mm_concat!($fun, f64)(self.0, shuf);
                    let shuff = _mm256_shuffle_pd(shuf, sums, _MM_SHUFFLE(0, 1, 4, 5));
                    let sums = _mm_concat!($fun, f64)(sums, shuf);
                    _mm256_cvtsd_f64(sums)
                }
            }
        )*
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
    (@arith $target:ident, $ty:ident, $($trait:ident, $fun:ident, $($tag:ident)?),+) => {
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

    ($($og:ident as $target:ident $(with $tag:ident)? => [$ty:ident;$len:literal]),+) => {
        $(
            #[allow(non_camel_case_types)]
            #[repr(transparent)]
            #[derive(Copy, Assign)]
            #[assign_targets(Add, Sub, Mul, Div)]
            #[assign_rhs(Self, $ty)]
            pub struct $target(pub(crate) $og);
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
                    unsafe { Self(_mm_concat!(sub, $ty)(_mm_concat!(setzero, $ty)(), self.0)) }
                }
            }

            impl PartialEq for $target {
                #[inline(always)]
                fn eq (&self, rhs: &Self) -> bool {
                    let cmp : [u128;2] = unsafe { transmute(_mm_concat!(cmp, $ty)(self.0, rhs.0, _MM_CMPINT_EQ)) };
                    cmp[0] == u128::MAX && cmp[1] == u128::MAX
                }
            }

            impl $target {
                const ABS_MASK : $og = unsafe { transmute(abs_mask!($ty)) };

                /// Loads values from the pointer into the SIMD vector
                #[inline(always)]
                pub unsafe fn load (ptr: *const $ty) -> Self {
                    Self(_mm_concat!(loadu, $ty)(ptr))
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
                    min as min: "Gets the smallest/minimum value of the vector",
                    max as max: "Gets the biggest/maximum value of the vector",
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
    __m256 as f32x8 => [f32;8],
    __m256d as f64x4 => [f64;4]
);

impl_clone!(
    f32x8, f32, 8,
    f64x4, f64, 4
);

impl_composite!(
    (f32x8 => 8, f32x2 => 2) as f32x10: f32,
    (f32x8 => 8, f32x4 => 4) as f32x12: f32,
    (f32x8 => 8, f32x6 => 6) as f32x14: f32,
    (f32x8 => 8, f32x8 => 6) as f32x16: f32
);

impl_composite!(
    (f64x4 => 4, f64x2 => 2) as f64x6: f64,
    (f64x4 => 4, f64x4 => 4) as f64x8: f64
);

impl_composite!(
    (f64x4 => 4, f64x4 => 4, f64x2 => 2) as f64x10: f64,
    (f64x4 => 4, f64x4 => 4, f64x4 => 4) as f64x12: f64
);

impl_composite!(
    (f64x4 => 4, f64x4 => 4, f64x4 => 4, f64x2 => 2) as f64x14: f64,
    (f64x4 => 4, f64x4 => 4, f64x4 => 4, f64x4 => 4) as f64x16: f64
);