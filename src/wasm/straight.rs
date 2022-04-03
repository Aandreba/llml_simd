use core::ops::*;
use derive_more::Neg;
use llml_simd_proc::*;
use core::mem::{transmute};
use core::ptr::addr_of;
use core::arch::wasm32::*;

macro_rules! impl_arith {
    ($target:ident, $ty:ident, $($trait:ident, $fun:ident),+) => {
        $(
            impl $trait for $target {
                type Output = Self;
    
                #[inline(always)]
                fn $fun (self, rhs: Self) -> Self::Output {
                    unsafe { Self(concat_idents!($target, _, $fun)(self.0, rhs.0)) }
                }
            }

            impl_scal_arith!($target, $ty, $trait, $fun);
        )*
    };
}

macro_rules! f32x4_hoz {
    ($($fun:ident $(as $name:ident)?: $docs:expr),+) => {
        $(
            f32x4_hoz!(1, $fun $(,$name)?, $docs);
        )*
    };

    (1, $fun:ident, $name:ident, $docs:expr) => {
        #[doc=$docs]
        #[inline(always)]
        pub fn $name (self) -> f32 {
            let shuf = u32x4_shuffle::<1, 0, 3, 2>(self.0, self.0);
            let sums = concat_idents!(f32x4_, $fun)(self.0, shuf);
            let shuf = u32x4_shuffle::<0, 1, 4, 5>(shuf, sums);
            let sums = concat_idents!(f32x4_, $fun)(sums, shuf);
            f32x4_extract_lane::<3>(sums)
        }
    };

    (1, $fun:ident, $docs:expr) => {
        f32x4_hoz!(1, $fun, $fun, $docs);
    };
}

macro_rules! impl_self_fns_stright {
    ($target:ident, $ty:ident, $($fun:ident $(as $name:ident)? $(with $tag:ident)?: $docs:expr),+) => {
        $(
            impl_self_fns_stright!(1, $target, $fun $(,$name)?, $ty, $docs, $($tag)?);
        )*
    };

    (1, $target:ident, $fun:ident, $name:ident, $ty:ident, $docs:expr, $($tag:ident)?) => {
        #[doc=concat!("Returns a vector with the ", $docs, " of the original vector")]
        #[inline(always)]
        pub fn $name (self) -> Self {
            unsafe { Self(concat_idents!($target, _, $fun)(self.0)) }
        }
    };

    (1, $target:ident, $fun:ident, $ty:ident, $docs:expr, $($tag:ident)?) => {
        impl_self_fns_stright!(1, $target, $fun, $fun, $ty, $docs, $($tag)?);
    };
}

macro_rules! impl_other_fns_straight {
    ($target:ident, $ty:ident, $($fun:ident $(as $name:ident)? $(with $tag:ident)?: $docs:expr),+) => {
        $(
            impl_other_fns_straight!(1, $target, $fun $(,$name)?, $ty, $docs, $($tag)?);
        )*
    };

    (1, $target:ident, $fun:ident, $name:ident, $ty:ident, $docs:expr, $($tag:ident)?) => {
        #[doc=concat!("Returns a vector with the ", $docs, " of each lane")]
        #[inline(always)]
        pub fn $name (self, rhs: Self) -> Self {
            unsafe { Self(concat_idents!($target, _, $fun)(self.0, rhs.0)) }
        }
    };

    (1, $target:ident, $fun:ident, $ty:ident, $docs:expr, $($tag:ident)?) => {
        impl_other_fns_straight!(1, $fun, $fun, $ty, $docs, $($tag)?);
    };
}

macro_rules! impl_straight {
    ($([$ty:ident;$len:literal] as $name:ident),+) => {
        $(
            #[allow(non_camel_case_types)]
            #[repr(transparent)]
            #[derive(Copy, Assign)]
            #[assign_targets(Add, Sub, Mul, Div)]
            #[assign_rhs(Self, $ty)]
            pub struct $name(v128);

            impl_arith!(
                $name, $ty,
                Add, add,
                Sub, sub,
                Mul, mul,
                Div, div
            );

            impl Neg for $name {
                type Output = Self;

                #[inline(always)]
                fn neg(self) -> Self::Output {
                    unsafe { Self(concat_idents!($name, _neg)(self.0)) }
                }
            }

            impl $name {                
                /// Loads values from the pointer into the SIMD vector
                #[inline(always)]
                pub unsafe fn load (ptr: *const $ty) -> Self {
                    Self(v128_load(ptr.cast()))
                }
                
                impl_self_fns_stright!(
                    $name, $ty,
                    abs: "absolute values",
                    sqrt: "square roots"
                );

                impl_other_fns_straight!(
                    $name, $ty,
                    pmin as vmin: "smallest/minimum value",
                    pmax as vmax: "biggest/maximum value"
                );
            }

            impl From<$ty> for $name {
                #[inline(always)]
                fn from(x: $ty) -> Self {
                    unsafe { Self(impl_straight!(@splat $ty)(addr_of!(x).cast())) }
                }
            }

            impl PartialEq for $name { 
                #[inline(always)]
                fn eq (&self, rhs: &Self) -> bool {
                    unsafe { transmute::<v128, u128>(concat_idents!($name, _eq)(self.0, rhs.0)) == u128::MAX }
                }
            }
        )*
    };

    (@splat f32) => { v128_load32_splat };
    (@splat f64) => { v128_load64_splat };
}

impl f32x4 { 
    f32x4_hoz!(
        pmin as min: "Gets the smallest/minimum value of the vector",
        pmax as max: "Gets the biggest/maximum value of the vector", 
        add as sum: "Sums up all the values inside the vector",
        mul as prod: "Multiplies all the values inside the vector"
    ); 

    /// Interleaves elements of both vectors into one
    #[inline(always)]
    pub fn zip (self, rhs: Self) -> Self {
        unsafe { Self(u32x4_shuffle::<0, 4, 1, 5>(self.0, rhs.0)) }
    }
}

impl f64x2 {
    /// Gets the smallest/minimum value of the vector
    #[inline(always)]
    pub fn min (self) -> f64 {
        unsafe { 
            let ptr = addr_of!(self) as *const f64;
            (*ptr).min(*ptr.add(1))
        }
    }

    /// Gets the biggest/maximum value of the vector
    #[inline(always)]
    pub fn max (self) -> f64 {
        unsafe { 
            let ptr = addr_of!(self) as *const f64;
            (*ptr).max(*ptr.add(1))
        }
    }

    /// Sums up all the values inside the vector
    #[inline(always)]
    pub fn sum (self) -> f64 {
        unsafe { 
            let ptr = addr_of!(self) as *const f64;
            (*ptr).add(*ptr.add(1))
        }
    }

    /// Multiplies all the values inside the vector
    #[inline(always)]
    pub fn prod (self) -> f64 {
        unsafe { 
            let ptr = addr_of!(self) as *const f64;
            (*ptr).mul(*ptr.add(1))
        }
    }

    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding error.
    /// # Compatibility
    /// The fused multiply-add operation is only available on arm/aarch64 and x86/x86-64 with the target feature ```fma```.
    /// For the rest of targets, a regular multiplication and addition are performed
    #[inline(always)]
    pub fn mul_add (self, rhs: Self, add: Self) -> Self {
        (self * rhs) + add
    }

    /// Interleaves elements of both vectors into one
    #[inline(always)]
    pub fn zip (self, rhs: Self) -> Self {
        unsafe { Self(u64x2_shuffle::<0, 2>(self.0, rhs.0)) }
    }
}

use super::f32x2;

impl_straight!(
    [f32;4] as f32x4,
    [f64;2] as f64x2
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