use std::{ops::{Add, Sub, Mul, Div, Index, IndexMut}};
use crate::{Simd, Simdable};

pub trait Simdt: Sized where
    Self: 
        Add<Self, Output = Self> + Add<Self::Item, Output = Self> +
        Sub<Self, Output = Self> + Sub<Self::Item, Output = Self> +
        Mul<Self, Output = Self> + Mul<Self::Item, Output = Self> +
        Div<Self, Output = Self> + Div<Self::Item, Output = Self> + 
        Index<usize, Output = Self::Item> + IndexMut<usize>
{
    type Item: Simdable;

    fn sum (self) -> Self::Item;
    fn prod (self) -> Self::Item;
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
                    Self([self.first() $sy rhs.first(), self.last() $sy rhs.last()])
                } 
            }

            impl<T: Simdable + Copy, A: Simdt<Item = T> + Copy> $trait<T> for Simd<[A; 2]> {
                type Output = Self;
                
                #[inline(always)]
                fn $fun (self, rhs: T) -> Self::Output {
                    Self([self.first() $sy rhs, self.last() $sy rhs])
                } 
            }
        )*
    };
}

impl<A: Simdt + Copy> Simd<[A;2]> {
    #[inline(always)]
    pub fn first (&self) -> A {
        unsafe { *(self as *const Self as *const A) }
    }

    #[inline(always)]
    pub fn last (&self) -> A {
        unsafe { *(self as *const Self as *const A).add(1) }
    }
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
impl<A: Simdt + Copy> Simdt for Simd<[A;2]> where A::Item: Copy {
    type Item = A::Item;

    /// Sums all the values inside
    #[inline(always)]
    fn sum (self) -> Self::Item {
        self.first().sum() + self.last().sum()
    }

    /// Multiplies all the values inside
    #[inline(always)]
    fn prod (self) -> Self::Item {
        self.first().prod() * self.last().prod()
    }
}