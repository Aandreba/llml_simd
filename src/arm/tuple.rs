use std::{ops::{Add, Sub, Mul, Div, Neg, Index, IndexMut}};
use crate::{Simd, Simdt};
arm_use!();

macro_rules! simd_map {
    ($name:ident, $ty:ident) => {
        simd_map!(
            $name, $ty,
            Add, add, +,
            Sub, sub, -,
            Mul, mul, *,
            Div, div, /
        );
        
        impl Simdt for Simd<$name> {
            type Item = $ty;

            /// Sums all the values inside
            #[inline(always)]
            fn sum (self) -> Self::Item {
                Simd(self.0.0).sum() + Simd(self.0.1).sum()
            }

            /// Multiplies all the values inside
            #[inline(always)]
            fn prod (self) -> Self::Item {
                Simd(self.0.0).prod() * Simd(self.0.1).prod()
            }

            /// Finds the smallest value inside
            #[inline(always)]
            fn min (self) -> Self::Item {
                Simd(self.0.0).min().min(Simd(self.0.1).min())
            }

            /// Finds the biggest value inside
            #[inline(always)]
            fn max (self) -> Self::Item {
                Simd(self.0.0).min().min(Simd(self.0.1).min())
            }
        }

        impl Neg for Simd<$name> {
            type Output = Self;

            #[inline(always)]
            fn neg (self) -> Self::Output {
                Simd::zip(-Simd(self.0.0), -Simd(self.0.1))
            }
        }

        impl Index<usize> for Simd<$name> {
            type Output = $ty;

            #[inline(always)]
            fn index (&self, idx: usize) -> &Self::Output {
                unsafe { &*(self as *const Self as *const $ty).add(idx) }
            }
        }

        impl IndexMut<usize> for Simd<$name> {
            #[inline(always)]
            fn index_mut (&mut self, idx: usize) -> &mut <Self as Index<usize>>::Output {
                unsafe { &mut *(self as *mut Self as *mut $ty).add(idx) }
            }
        }

        impl Simd<$name> {
            #[inline(always)]
            fn zip (a: Simd<float32x4_t>, b: Simd<float32x4_t>) -> Self {
                unsafe { Simd(vzipq_f32(a.0, b.0)) }
            }
        }
    };

    ($name:ident, $ty:ident, $($trait:ident, $fun:ident, $sy:tt),+) => {
        $(
            impl $trait for Simd<$name> {
                type Output = Self;

                #[inline(always)]
                fn $fun (self, rhs: Self) -> Self::Output {
                    Self::zip(Simd(self.0.0) $sy Simd(rhs.0.0), Simd(self.0.1) $sy Simd(rhs.0.1))
                }
            }

            impl $trait<$ty> for Simd<$name> {
                type Output = Self;

                #[inline(always)]
                fn $fun (self, rhs: $ty) -> Self::Output {
                    Self::zip(Simd(self.0.0) $sy rhs, Simd(self.0.1) $sy rhs)
                }
            }

            impl $trait<Simd<$name>> for Simd<$ty> {
                type Output = Simd<$name>;

                #[inline(always)]
                fn $fun (self, rhs: Simd<$name>) -> Self::Output {
                    Simd::<$name>::zip(self $sy Simd(rhs.0.0), self $sy Simd(rhs.0.1))
                }
            }
        )*
    };
}

simd_map!(float32x4x2_t, f32);