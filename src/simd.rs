use std::{ops::{Add, Sub, Mul, Div, Index, IndexMut, Neg}, intrinsics::transmute};
use crate::{Simd, Simdable};
use crate::utils::Ordx;

pub trait Simdt: Sized where
    Self: 
        Add<Self, Output = Self> + Add<Self::Item, Output = Self> +
        Sub<Self, Output = Self> + Sub<Self::Item, Output = Self> +
        Mul<Self, Output = Self> + Mul<Self::Item, Output = Self> +
        Div<Self, Output = Self> + Div<Self::Item, Output = Self> + 
        Neg<Output = Self> +
        Index<usize, Output = Self::Item> + IndexMut<usize>
{
    type Item: Simdable;

    fn sum (self) -> Self::Item;
    fn prod (self) -> Self::Item;

    fn min (self) -> Self::Item;
    fn max (self) -> Self::Item;
}

macro_rules! impl_array_2 {
    () => {
        impl_array_2!(
            Add, add, +,
            Sub, sub, -,
            Mul, mul, *,
            Div, div, /
        );
    };

    ($($trait:ident, $fun:ident, $sy:tt),+) => {
        $(
            impl<A: Simdt + Copy> $trait for Simd<[A;2]> where A::Item: Simdable {
                type Output = Self;
                
                #[inline(always)]
                fn $fun (self, rhs: Self) -> Self::Output {
                    Self([self.0[0] $sy rhs.0[0], self.0[1] $sy rhs.0[1]])
                } 
            }

            impl<T: Simdable + Copy, A: Simdt<Item = T> + Copy> $trait<T> for Simd<[A; 2]> {
                type Output = Self;
                
                #[inline(always)]
                fn $fun (self, rhs: T) -> Self::Output {
                    Self([self.0[0] $sy rhs, self.0[1] $sy rhs])
                } 
            }
        )*
    };
}

impl<A: Simdt> Index<usize> for Simd<[A; 2]> {
    type Output = A::Item;

    #[inline(always)]
    fn index (&self, idx: usize) -> &Self::Output {
        unsafe { &*(self as *const Self as *const A::Item).add(idx) }
    } 
}

impl<A: Simdt> IndexMut<usize> for Simd<[A; 2]> {
    #[inline(always)]
    fn index_mut (&mut self, idx: usize) -> &mut Self::Output {
        unsafe { &mut *(self as *mut Self as *mut A::Item).add(idx) }
    } 
}

impl_array_2!();
impl<A: Simdt + Copy> Neg for Simd<[A;2]> where A::Item: Simdable {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Simd([-self.0[0], -self.0[1]])
    }
}

impl<A: Simdt + Copy> Simdt for Simd<[A;2]> where A::Item: Copy {
    type Item = A::Item;

    /// Sums all the values inside
    #[inline(always)]
    fn sum (self) -> Self::Item {
        self.0[0].sum() + self.0[1].sum()
    }

    /// Multiplies all the values inside
    #[inline(always)]
    fn prod (self) -> Self::Item {
        self.0[0].prod() * self.0[1].prod()
    }

    /// Finds the smallest value inside
    #[inline(always)]
    fn min (self) -> Self::Item {
        self.0[0].min().min(self.0[1].min())
    }

    /// Finds the biggest value inside
    #[inline(always)]
    fn max (self) -> Self::Item {
        self.0[0].max().max(self.0[1].max())
    }
}