use std::{ops::{Add, Sub, Mul, Div, Neg, Index, IndexMut}};
use crate::{Simd, Simdt};
use_x86!();

macro_rules! map_traits {
    () => {
        map_traits!(
            Add, add, +,
            Sub, sub, -,
            Mul, mul, *,
            Div, div, /
        );
    };

    ($($trait:ident, $fun:ident, $sy:tt),+) => {
        $(
            impl $trait for Simd<__m64> {
                type Output = Self;

                #[inline(always)]
                fn $fun (self, rhs: Self) -> Self::Output {
                    __m64::from_simd(self.0.0 $sy rhs.0.0)
                }
            }

            impl $trait<f32> for Simd<__m64> {
                type Output = Self;

                #[inline(always)]
                fn $fun (self, rhs: f32) -> Self::Output {
                    unsafe { __m64::from_simd(self.0.0 $sy Simd(_mm_set_ps(0., 0., rhs, rhs))) }
                }
            }

            impl $trait<Simd<__m64>> for Simd<f32> {
                type Output = Simd<__m64>;

                #[inline(always)]
                fn $fun (self, rhs: Simd<__m64>) -> Self::Output {
                    unsafe { __m64::from_simd(Simd(_mm_set_ps(0., 0., self.0, self.0)) $sy rhs.0.0) }
                }
            }
        )*
    };
}

#[derive(Debug, Clone, Copy)]
pub struct __m64 (Simd<__m128>);
map_traits!();

impl __m64 {
    #[inline(always)]
    pub(crate) fn new (x: f32, y: f32) -> Simd<Self> {
        unsafe { Simd(__m64(Simd(_mm_set_ps(0., 0., y, x)))) }
    }

    #[inline(always)]
    pub(crate) fn from_simd (x: Simd<__m128>) -> Simd<Self> {
        Simd(__m64(x))
    }
}

impl Neg for Simd<__m64> {
    type Output = Self;

    fn neg (self) -> Self {
        __m64::from_simd(Simd(0.) - self.0.0)
    }
}

impl Index<usize> for Simd<__m64> {
    type Output = f32;

    #[inline(always)]
    fn index (&self, index: usize) -> &Self::Output {
        &self.0.0[index]
    }
}

impl IndexMut<usize> for Simd<__m64> {
    #[inline(always)]
    fn index_mut (&mut self, index: usize) -> &mut Self::Output {
        &mut self.0.0[index]
    }
}

impl Simdt for Simd<__m64> {
    type Item = f32;

    fn sum (self) -> Self::Item {
        self[0] + self[1]
    }

    fn prod (self) -> Self::Item {
        self[0] * self[1]
    }

    fn min (self) -> Self::Item {
        self[0].min(self[1])
    }

    fn max (self) -> Self::Item {
        self[0].max(self[1])
    }
}