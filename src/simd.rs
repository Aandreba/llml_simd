use std::{ops::{Add, Sub, Mul, Div, Index, IndexMut, Neg}, intrinsics::transmute, os::windows::prelude::MetadataExt};
use crate::{Simd, Simdable};
use crate::utils::Ordx;
use crate::{f32x4, f64x4};

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

macro_rules! impl_array_4 {
    () => {
        impl_array_4!(
            Add, add, +,
            Sub, sub, -,
            Mul, mul, *,
            Div, div, /
        );
    };

    ($($trait:ident, $fun:ident, $sy:tt),+) => {
        $(
            impl<A: Simdt + Copy> $trait for Simd<[A;4]> where A::Item: Simdable {
                type Output = Self;
                
                #[inline(always)]
                fn $fun (self, rhs: Self) -> Self::Output {
                    Self([
                        self.0[0] $sy rhs.0[0], 
                        self.0[1] $sy rhs.0[1],
                        self.0[2] $sy rhs.0[2], 
                        self.0[3] $sy rhs.0[3]
                    ])
                } 
            }

            impl<T: Simdable + Copy, A: Simdt<Item = T> + Copy> $trait<T> for Simd<[A;4]> {
                type Output = Self;
                
                #[inline(always)]
                fn $fun (self, rhs: T) -> Self::Output {
                    Self([
                        self.0[0] $sy rhs, 
                        self.0[1] $sy rhs,
                        self.0[2] $sy rhs,
                        self.0[3] $sy rhs
                    ])
                } 
            }
        )*
    };
}

macro_rules! impl_array_8 {
    () => {
        impl_array_8!(
            Add, add, +,
            Sub, sub, -,
            Mul, mul, *,
            Div, div, /
        );
    };

    ($($trait:ident, $fun:ident, $sy:tt),+) => {
        $(
            impl<A: Simdt + Copy> $trait for Simd<[A;8]> where A::Item: Simdable {
                type Output = Self;
                
                #[inline(always)]
                fn $fun (self, rhs: Self) -> Self::Output {
                    Self([
                        self.0[0] $sy rhs.0[0], 
                        self.0[1] $sy rhs.0[1],
                        self.0[2] $sy rhs.0[2], 
                        self.0[3] $sy rhs.0[3],
                        self.0[4] $sy rhs.0[4], 
                        self.0[5] $sy rhs.0[5],
                        self.0[6] $sy rhs.0[6], 
                        self.0[7] $sy rhs.0[7]
                    ])
                } 
            }

            impl<T: Simdable + Copy, A: Simdt<Item = T> + Copy> $trait<T> for Simd<[A;8]> {
                type Output = Self;
                
                #[inline(always)]
                fn $fun (self, rhs: T) -> Self::Output {
                    Self([
                        self.0[0] $sy rhs, 
                        self.0[1] $sy rhs,
                        self.0[2] $sy rhs,
                        self.0[3] $sy rhs,
                        self.0[4] $sy rhs, 
                        self.0[5] $sy rhs,
                        self.0[6] $sy rhs,
                        self.0[7] $sy rhs
                    ])
                } 
            }
        )*
    };
}

macro_rules! impl_array_16 {
    () => {
        impl_array_16!(
            Add, add, +,
            Sub, sub, -,
            Mul, mul, *,
            Div, div, /
        );
    };

    ($($trait:ident, $fun:ident, $sy:tt),+) => {
        $(
            impl<A: Simdt + Copy> $trait for Simd<[A;16]> where A::Item: Simdable {
                type Output = Self;
                
                #[inline(always)]
                fn $fun (self, rhs: Self) -> Self::Output {
                    Self([
                        self.0[0] $sy rhs.0[0], 
                        self.0[1] $sy rhs.0[1],
                        self.0[2] $sy rhs.0[2], 
                        self.0[3] $sy rhs.0[3],
                        self.0[4] $sy rhs.0[4], 
                        self.0[5] $sy rhs.0[5],
                        self.0[6] $sy rhs.0[6], 
                        self.0[7] $sy rhs.0[7],

                        self.0[8] $sy rhs.0[8], 
                        self.0[9] $sy rhs.0[9],
                        self.0[10] $sy rhs.0[10], 
                        self.0[11] $sy rhs.0[11],
                        self.0[12] $sy rhs.0[12], 
                        self.0[13] $sy rhs.0[13],
                        self.0[14] $sy rhs.0[14], 
                        self.0[15] $sy rhs.0[15]
                    ])
                } 
            }

            impl<T: Simdable + Copy, A: Simdt<Item = T> + Copy> $trait<T> for Simd<[A;16]> {
                type Output = Self;
                
                #[inline(always)]
                fn $fun (self, rhs: T) -> Self::Output {
                    Self([
                        self.0[0] $sy rhs, 
                        self.0[1] $sy rhs,
                        self.0[2] $sy rhs,
                        self.0[3] $sy rhs,
                        self.0[4] $sy rhs, 
                        self.0[5] $sy rhs,
                        self.0[6] $sy rhs,
                        self.0[7] $sy rhs,

                        self.0[8] $sy rhs, 
                        self.0[9] $sy rhs,
                        self.0[10] $sy rhs,
                        self.0[11] $sy rhs,
                        self.0[12] $sy rhs, 
                        self.0[13] $sy rhs,
                        self.0[14] $sy rhs,
                        self.0[15] $sy rhs
                    ])
                } 
            }
        )*
    };
}

macro_rules! impl_array {
    () => {
        impl_array!(2, 4, 8, 16);
    };

    ($($len:literal),+) => {
        $(
            impl<A: Simdt> Index<usize> for Simd<[A;$len]> {
                type Output = A::Item;
            
                #[inline(always)]
                fn index (&self, idx: usize) -> &Self::Output {
                    unsafe { &*(self as *const Self as *const A::Item).add(idx) }
                } 
            }
            
            impl<A: Simdt> IndexMut<usize> for Simd<[A;$len]> {
                #[inline(always)]
                fn index_mut (&mut self, idx: usize) -> &mut Self::Output {
                    unsafe { &mut *(self as *mut Self as *mut A::Item).add(idx) }
                } 
            }
        )*
    };
}

impl_array_2!();
impl_array_4!();
impl_array_8!();
impl_array_16!();
impl_array!();

impl<A: Simdt + Copy> Neg for Simd<[A;2]> where A::Item: Simdable {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Simd([-self.0[0], -self.0[1]])
    }
}

impl<A: Simdt + Copy> Neg for Simd<[A;4]> where A::Item: Simdable {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Simd([
            -self.0[0], 
            -self.0[1],
            -self.0[2],
            -self.0[3]
        ])
    }
}

impl<A: Simdt + Copy> Neg for Simd<[A;8]> where A::Item: Simdable {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Simd([
            -self.0[0], 
            -self.0[1],
            -self.0[2],
            -self.0[3],
            -self.0[4], 
            -self.0[5],
            -self.0[6],
            -self.0[7]
        ])
    }
}

impl<A: Simdt + Copy> Neg for Simd<[A;16]> where A::Item: Simdable {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Simd([
            -self.0[0], 
            -self.0[1],
            -self.0[2],
            -self.0[3],
            -self.0[4], 
            -self.0[5],
            -self.0[6],
            -self.0[7],

            -self.0[8], 
            -self.0[9],
            -self.0[10],
            -self.0[11],
            -self.0[12], 
            -self.0[13],
            -self.0[14],
            -self.0[15]
        ])
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

impl<A: Simdt + Copy> Simdt for Simd<[A;4]> where A::Item: Simdable + Copy {
    type Item = A::Item;

    /// Sums all the values inside
    #[inline(always)]
    fn sum (self) -> Self::Item {
        Self::Item::x4([
            self.0[0].sum(),
            self.0[1].sum(),
            self.0[2].sum(),
            self.0[3].sum()
        ]).sum()
    }

    /// Multiplies all the values inside
    #[inline(always)]
    fn prod (self) -> Self::Item {
        Self::Item::x4([
            self.0[0].prod(),
            self.0[1].prod(),
            self.0[2].prod(),
            self.0[3].prod()
        ]).prod()
    }

    /// Finds the smallest value inside
    #[inline(always)]
    fn min (self) -> Self::Item {
        Self::Item::x4([
            self.0[0].min(),
            self.0[1].min(),
            self.0[2].min(),
            self.0[3].min()
        ]).min()
    }

    /// Finds the biggest value inside
    #[inline(always)]
    fn max (self) -> Self::Item {
        Self::Item::x4([
            self.0[0].max(),
            self.0[1].max(),
            self.0[2].max(),
            self.0[3].max()
        ]).max()
    }
}

impl<A: Simdt + Copy> Simdt for Simd<[A;8]> where A::Item: Simdable + Copy {
    type Item = A::Item;

    /// Sums all the values inside
    #[inline(always)]
    fn sum (self) -> Self::Item {
        Self::Item::x8([
            self.0[0].sum(),
            self.0[1].sum(),
            self.0[2].sum(),
            self.0[3].sum(),
            self.0[4].sum(),
            self.0[5].sum(),
            self.0[6].sum(),
            self.0[7].sum()
        ]).sum()
    }

    /// Multiplies all the values inside
    #[inline(always)]
    fn prod (self) -> Self::Item {
        Self::Item::x8([
            self.0[0].prod(),
            self.0[1].prod(),
            self.0[2].prod(),
            self.0[3].prod(),
            self.0[4].prod(),
            self.0[5].prod(),
            self.0[6].prod(),
            self.0[7].prod()
        ]).prod()
    }

    /// Finds the smallest value inside
    #[inline(always)]
    fn min (self) -> Self::Item {
        Self::Item::x8([
            self.0[0].min(),
            self.0[1].min(),
            self.0[2].min(),
            self.0[3].min(),
            self.0[4].min(),
            self.0[5].min(),
            self.0[6].min(),
            self.0[7].min()
        ]).min()
    }

    /// Finds the biggest value inside
    #[inline(always)]
    fn max (self) -> Self::Item {
        Self::Item::x8([
            self.0[0].max(),
            self.0[1].max(),
            self.0[2].max(),
            self.0[3].max(),
            self.0[4].max(),
            self.0[5].max(),
            self.0[6].max(),
            self.0[7].max()
        ]).max()
    }
}

impl<A: Simdt + Copy> Simdt for Simd<[A;16]> where A::Item: Simdable + Copy {
    type Item = A::Item;

    /// Sums all the values inside
    #[inline(always)]
    fn sum (self) -> Self::Item {
        Self::Item::x16([
            self.0[0].sum(),
            self.0[1].sum(),
            self.0[2].sum(),
            self.0[3].sum(),
            self.0[4].sum(),
            self.0[5].sum(),
            self.0[6].sum(),
            self.0[7].sum(),
            self.0[8].sum(),
            self.0[9].sum(),
            self.0[10].sum(),
            self.0[11].sum(),
            self.0[12].sum(),
            self.0[13].sum(),
            self.0[14].sum(),
            self.0[15].sum()
        ]).sum()
    }

    /// Multiplies all the values inside
    #[inline(always)]
    fn prod (self) -> Self::Item {
        Self::Item::x16([
            self.0[0].prod(),
            self.0[1].prod(),
            self.0[2].prod(),
            self.0[3].prod(),
            self.0[4].prod(),
            self.0[5].prod(),
            self.0[6].prod(),
            self.0[7].prod(),
            self.0[8].prod(),
            self.0[9].prod(),
            self.0[10].prod(),
            self.0[11].prod(),
            self.0[12].prod(),
            self.0[13].prod(),
            self.0[14].prod(),
            self.0[15].prod()
        ]).prod()
    }

    /// Finds the smallest value inside
    #[inline(always)]
    fn min (self) -> Self::Item {
        Self::Item::x16([
            self.0[0].min(),
            self.0[1].min(),
            self.0[2].min(),
            self.0[3].min(),
            self.0[4].min(),
            self.0[5].min(),
            self.0[6].min(),
            self.0[7].min(),
            self.0[8].min(),
            self.0[9].min(),
            self.0[10].min(),
            self.0[11].min(),
            self.0[12].min(),
            self.0[13].min(),
            self.0[14].min(),
            self.0[15].min()
        ]).min()
    }

    /// Finds the biggest value inside
    #[inline(always)]
    fn max (self) -> Self::Item {
        Self::Item::x16([
            self.0[0].max(),
            self.0[1].max(),
            self.0[2].max(),
            self.0[3].max(),
            self.0[4].max(),
            self.0[5].max(),
            self.0[6].max(),
            self.0[7].max(),
            self.0[8].max(),
            self.0[9].max(),
            self.0[10].max(),
            self.0[11].max(),
            self.0[12].max(),
            self.0[13].max(),
            self.0[14].max(),
            self.0[15].max()
        ]).max()
    }
}