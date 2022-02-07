use core::fmt::Debug;
use crate::float::single::*;
use crate::float::double::*;
use core::ptr::addr_of;
use core::ops::*;

macro_rules! impl_generic {
    ($($target:ident, $ty:ident, $len:literal),+) => {
        $(
            impl $target {
                #[inline(always)]
                pub fn new (a: [$ty;$len]) -> Self {
                    Self::from(a)
                }
            
                /// Creates a new vector with all lines filled with the provided value
                #[inline(always)]
                pub fn filled_with (a: $ty) -> Self {
                    Self::from(a)
                }
            }

            impl Debug for $target {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    let array : [$ty;$len] = self.clone().into();
                    f.debug_list().entries(array).finish()
                }
            }

            impl From<[$ty;$len]> for $target {
                #[inline(always)]
                fn from(x: [$ty;$len]) -> Self {
                    unsafe { Self::load(addr_of!(x).cast()) }
                }
            }

            impl Into<[$ty;$len]> for $target {
                #[inline(always)]
                fn into(self) -> [$ty;$len] {
                    let ptr : *const [$ty;$len] = addr_of!(self).cast();
                    unsafe { *ptr }
                }
            }
        )*
    };
}

macro_rules! impl_index {
    ($($target:ident, $ty:ident, $len:literal),+) => {
        $(
            impl $target {
                /// Returns a reference to the value in the specified lane without checking if it's within range
                #[inline(always)]
                pub unsafe fn index_unchecked (&self, idx: usize) -> &$ty {
                    let ptr = self as *const Self as *const $ty;
                    &*ptr.add(idx)
                }

                /// Returns a mutable reference to the value in the specified lane without checking if it's within range
                #[inline(always)]
                pub unsafe fn index_mut_unchecked (&mut self, idx: usize) -> &mut $ty {
                    let ptr = self as *mut Self as *mut $ty;
                    &mut *ptr.add(idx)
                }
            }

            impl Index<usize> for $target {
                type Output = $ty;

                #[inline(always)]
                fn index (&self, idx: usize) -> &Self::Output {
                    match idx {
                        0..$len => unsafe { self.index_unchecked(idx) },
                        _ => panic!("Index out of bounds")
                    }
                }
            }

            impl IndexMut<usize> for $target {
                #[inline(always)]
                fn index_mut (&mut self, idx: usize) -> &mut Self::Output {
                    match idx {
                        0..$len => unsafe { self.index_mut_unchecked(idx) },
                        _ => panic!("Index out of bounds")
                    }
                }
            }
        )*  
    };
}

impl_generic!(
    f32x2, f32, 2,
    f32x4, f32, 4,
    f32x6, f32, 6,
    f32x8, f32, 8,
    f32x10, f32, 10,
    f32x12, f32, 12,
    f32x14, f32, 14,
    f32x16, f32, 16,

    f64x2, f64, 2,
    f64x4, f64, 4,
    f64x6, f64, 6,
    f64x8, f64, 8,
    f64x10, f64, 10,
    f64x12, f64, 12,
    f64x14, f64, 14,
    f64x16, f64, 16
);

impl_clone!(
    f32x2, f32, 2,
    f32x4, f32, 4,
    f64x2, f64, 2
);

impl_index!(
    f32x2, f32, 2,
    f32x4, f32, 4,
    f32x6, f32, 6,
    f32x8, f32, 8,
    f32x10, f32, 10,
    f32x12, f32, 12,
    f32x14, f32, 14,
    f32x16, f32, 16,

    f64x2, f64, 2,
    f64x4, f64, 4,
    f64x6, f64, 6,
    f64x8, f64, 8,
    f64x10, f64, 10,
    f64x12, f64, 12,
    f64x14, f64, 14,
    f64x16, f64, 16
);