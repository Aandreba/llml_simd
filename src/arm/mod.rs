use std::{ops::{Add, Sub, Mul, Div, Neg, Index, IndexMut}, intrinsics::transmute};
use crate::{Simd, Simdt, Simdable, SimdType};

include!("prod.rs");

macro_rules! arm_use {
    () => {
        #[cfg(target_arch = "arm")]
        use std::arch::arm::*;

        #[cfg(target_arch = "aarch64")]
        use std::arch::aarch64::*;
    };
}

macro_rules! simd_map {
    ($name:ident, $type:ident, $len:literal) => {
        simd_map!(
            $name, $len, $type,
            Add, add,
            Sub, sub,
            Mul, mul,
            Div, div
        );

        simd_map!(
            bool, $name, $len, $type,
            Neg, neg
        );

        impl Simdt for Simd<concat_idents!($name, _t)> {
            type Item = $type;

            /// Sums all the values inside
            #[inline(always)]
            fn sum (self) -> $type {
                unsafe { concat_idents!(vaddv, _, $type)(self.0) }
            }

            /// Multiplies all the values inside
            #[inline(always)]
            fn prod (self) -> Self::Item {
                Self::_prod(self)
            }

            /// Finds the smallest value inside
            #[inline(always)]
            fn min (self) -> Self::Item {
                unsafe { concat_idents!(vminv, _, $type)(self.0) }
            }

            /// Finds the biggest value inside
            #[inline(always)]
            fn max (self) -> Self::Item {
                unsafe { concat_idents!(vmaxv, _, $type)(self.0) }
            }
        }

        impl Index<usize> for Simd<concat_idents!($name, _t)> {
            type Output = $type;

            #[inline(always)]
            fn index (&self, idx: usize) -> &Self::Output {
                unsafe { &*(self as *const Self as *const $type).add(idx) }
            }
        }

        impl IndexMut<usize> for Simd<concat_idents!($name, _t)> {
            #[inline(always)]
            fn index_mut (&mut self, idx: usize) -> &mut <Self as Index<usize>>::Output {
                unsafe { &mut *(self as *mut Self as *mut $type).add(idx) }
            }
        }
    };

    (bool, $name:ident, $len:literal, $ty:ident, $($trait:ident, $fun:ident),+) => {
        $(
            impl $trait for Simd<concat_idents!($name, _t)> {
                type Output = Self;

                #[inline(always)]
                fn $fun (self) -> Self::Output {
                    unsafe { Simd(concat_idents!(v, $fun, _, $ty)(self.0)) }
                }
            }
        )*
    };

    ($name:ident, $len:literal, $ty:ident, $($trait:ident, $fun:ident),+) => {
        $(
            impl $trait for Simd<concat_idents!($name, _t)> {
                type Output = Self;

                #[inline(always)]
                fn $fun (self, rhs: Self) -> Self::Output {
                    unsafe { Simd(concat_idents!(v, $fun, _, $ty)(self.0, rhs.0)) }
                }
            }

            impl $trait<$ty> for Simd<concat_idents!($name, _t)> {
                type Output = Self;

                #[inline(always)]
                fn $fun (self, rhs: $ty) -> Self::Output {
                    unsafe { 
                        Simd(concat_idents!(v, $fun, _, $ty)(
                            self.0, 
                            transmute([rhs; $len])
                        ))
                    }
                }
            }

            impl $trait<Simd<concat_idents!($name, _t)>> for Simd<$ty> {
                type Output = Simd<concat_idents!($name, _t)>;

                #[inline(always)]
                fn $fun (self, rhs: Self::Output) -> Self::Output {
                    unsafe { 
                        Simd(concat_idents!(v, $fun, _, $ty)(
                            transmute([self;$len]),
                            rhs.0
                        ))
                    }
                }
            }
        )*
    };
}

macro_rules! simd_map_tag {
    ($name:ident, $type:ident, $len:literal, $tag:ident) => {
        simd_map_tag!(
            $name, $len, $type, $tag,
            Add, add,
            Sub, sub,
            Mul, mul,
            Div, div
        );

        simd_map_tag!(
            bool, $name, $len, $type, $tag,
            Neg, neg
        );

        impl Simdt for Simd<concat_idents!($name, _t)> {
            type Item = $type;

            /// Sums all the values inside
            #[inline(always)]
            fn sum (self) -> $type {
                unsafe { concat_idents!(vaddv, $tag, _, $type)(self.0) }
            }

            /// Multiplies all the values inside
            #[inline(always)]
            fn prod (self) -> Self::Item {
                Self::_prod(self)
            }

            /// Finds the smallest value inside
            #[inline(always)]
            fn min (self) -> Self::Item {
                unsafe { concat_idents!(vminv, $tag, _, $type)(self.0) }
            }

            /// Finds the biggest value inside
            #[inline(always)]
            fn max (self) -> Self::Item {
                unsafe { concat_idents!(vmaxv, $tag, _, $type)(self.0) }
            }
        }

        impl Index<usize> for Simd<concat_idents!($name, _t)> {
            type Output = $type;

            #[inline(always)]
            fn index (&self, idx: usize) -> &Self::Output {
                unsafe { &*(self as *const Self as *const $type).add(idx) }
            }
        }

        impl IndexMut<usize> for Simd<concat_idents!($name, _t)> {
            #[inline(always)]
            fn index_mut (&mut self, idx: usize) -> &mut <Self as Index<usize>>::Output {
                unsafe { &mut *(self as *mut Self as *mut $type).add(idx) }
            }
        }
    };

    (bool, $name:ident, $len:literal, $ty:ident, $tag:ident, $($trait:ident, $fun:ident),+) => {
        $(
            impl $trait for Simd<concat_idents!($name, _t)> {
                type Output = Self;

                #[inline(always)]
                fn $fun (self) -> Self::Output {
                    unsafe { Simd(concat_idents!(v, $fun, $tag, _, $ty)(self.0)) }
                }
            }
        )*
    };

    ($name:ident, $len:literal, $ty:ident, $tag:ident, $($trait:ident, $fun:ident),+) => {
        $(
            impl $trait for Simd<concat_idents!($name, _t)> {
                type Output = Self;

                #[inline(always)]
                fn $fun (self, rhs: Self) -> Self::Output {
                    unsafe { Simd(concat_idents!(v, $fun, $tag, _, $ty)(self.0, rhs.0)) }
                }
            }

            impl $trait<$ty> for Simd<concat_idents!($name, _t)> {
                type Output = Self;

                #[inline(always)]
                fn $fun (self, rhs: $ty) -> Self::Output {
                    unsafe { 
                        Simd(concat_idents!(v, $fun, $tag, _, $ty)(
                            self.0, 
                            transmute([rhs;$len])
                        ))
                    }
                }
            }

            impl $trait<Simd<concat_idents!($name, _t)>> for Simd<$ty> {
                type Output = Simd<concat_idents!($name, _t)>;

                #[inline(always)]
                fn $fun (self, rhs: Self::Output) -> Self::Output {
                    unsafe { 
                        Simd(concat_idents!(v, $fun, $tag, _, $ty)(
                            transmute([self;$len]),
                            rhs.0
                        ))
                    }
                }
            }
        )*
    };
} 

flat_mod!(prod);
simdable!(f32, f64);
arm_use!();

simd_map!(float32x2, f32, 2);
simd_map_tag!(float32x4, f32, 4, q);
simd_map_tag!(float64x2, f64, 2, q);

impl_prod2!(
    float32, f32,
    float64, f64
);

impl_prod4!(
    float32, f32
);

pub fn f32x2 (x: f32, y: f32) -> Simd<float32x2_t> {
    unsafe { Simd(transmute([x, y])) }
}

pub fn f32x4 (x: f32, y: f32, z: f32, w: f32) -> Simd<float32x4_t> {
    unsafe {
        Simd(transmute([x, y, z, w]))
    }
}

pub fn f64x2 (x: f64, y: f64) -> Simd<float64x2_t> {
    unsafe { Simd(transmute([x, y])) }
}

/// Returns the newest architecture feature available to the compiler
pub fn get_top_feature () -> SimdType {
    SimdType::ARM
}