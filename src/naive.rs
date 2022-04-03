use llml_simd_proc::*;
use core::ops::*;
use std::mem::MaybeUninit;

#[inline]
fn array<T, F: Fn(usize) -> T, const N: usize> (f: F) -> [T; N] {
    let mut array = MaybeUninit::<[T;N]>::uninit();
    let ptr : *mut T = array.as_mut_ptr().cast();

    for i in 0..N {
        unsafe { ptr.add(i).write(f(i)); }
    }

    unsafe { array.assume_init() }
}

macro_rules! impl_self_fns {
    ([$ty:ident;$len:literal], $($fun:ident $(as $name:ident)?: $docs:expr),+) => {
        $(
            impl_self_fns!(1, $fun $(,$name)?, $ty, $len, $docs);
        )*
    };

    (1, $fun:ident, $ty:ident, $len:literal, $docs:expr) => {
        #[cfg(feature = "use_std")]
        #[doc=concat!("Returns a vector with the ", $docs, " of the original vector")]
        #[inline(always)]
        pub fn $fun (self) -> Self {
            Self(array(|i| (self[i] as $ty).$fun()))
        }
    };

    (1, $fun:ident, $name:ident, $ty:ident, $len:literal, $docs:expr) => {
        #[cfg(feature = "use_std")]
        #[doc=concat!("Returns a vector with the ", $docs, " of the original vector")]
        #[inline(always)]
        pub fn $name (self) -> Self {
            Self(array(|i| (self[i] as $ty).$fun()))
        }
    };
}

macro_rules! impl_other_fns {
    ([$ty:ident;$len:literal], $($fun:ident $(as $name:ident)?: $docs:expr),+) => {
        $(
            impl_other_fns!(1, $fun $(, $name)?, $ty, $len, $docs);
        )*
    };

    (1, $fun:ident, $ty:ident, $len:literal, $docs:expr) => {
        #[doc=concat!("Returns a vector with the ", $docs, " of each lane")]
        #[inline(always)]
        pub fn $fun (self, rhs: Self) -> Self {
            Self(array(|i| (self[i] as $ty).$fun(rhs[i] as $ty)))
        }
    };

    (1, $fun:ident, $name:ident, $ty:ident, $len:literal, $docs:expr) => {
        #[doc=concat!("Returns a vector with the ", $docs, " of each lane")]
        #[inline(always)]
        pub fn $name (self, rhs: Self) -> Self {
            Self(array(|i| (self[i] as $ty).$fun(rhs[i] as $ty)))
        }
    };
}

macro_rules! impl_naive {
    (@arith $target:ident, $ty:ident, $len:literal, $($trait:ident, $fun:ident),+) => {
        $(
            impl $trait for $target {
                type Output = Self;

                #[inline(always)]
                fn $fun (self, rhs: Self) -> Self::Output {
                    let arr = array(|i| (self.0[i] as $ty).$fun(rhs.0[i]));
                    Self(arr)
                }
            }

            impl $trait<$ty> for $target {
                type Output = Self;

                #[inline(always)]
                fn $fun (self, rhs: $ty) -> Self::Output {
                    Self(array(|i| (self.0[i] as $ty).$fun(rhs)))
                }
            }

            impl $trait<$target> for $ty {
                type Output = $target;

                #[inline(always)]
                fn $fun (self, rhs: $target) -> Self::Output {
                    $target(array(|i| self.$fun(rhs.0[i])))
                }
            }
        )*
    };

    ($([$ty:ident;$len:literal] as $target:ident),+) => {
        $(
            #[allow(non_camel_case_types)]
            #[repr(transparent)]
            #[derive(Debug, Clone, Copy, Assign)]
            #[assign_targets(Add, Sub, Mul, Div)]
            #[assign_rhs(Self, $ty)]
            pub struct $target([$ty;$len]);
            impl_naive!(
                @arith $target, $ty, $len,
                Add, add,
                Sub, sub,
                Mul, mul,
                Div, div
            );

            impl Neg for $target {
                type Output = Self;

                fn neg (self) -> Self::Output {
                    Self(array(|i| self[i].neg()))
                }
            }

            impl $target {
                #[inline(always)]
                pub fn new (a: [$ty;$len]) -> Self {
                    Self(a)
                }
            
                /// Creates a new vector with all lines filled with the provided value
                #[inline(always)]
                pub fn filled_with (a: $ty) -> Self {
                    Self([a;$len])
                }
                
                /// Loads values from the pointer into the SIMD vector
                #[inline(always)]
                pub unsafe fn load (ptr: *const $ty) -> Self {
                    Self(*(ptr as *const [$ty;$len]))
                }

                /// Returns a reference to the value in the specified lane without checking if it’s within range
                #[inline(always)]
                pub unsafe fn index_unchecked (&self, idx: usize) -> &$ty {
                    self.index(idx)
                }

                /// Returns a mutable reference to the value in the specified lane without checking if it’s within range
                #[inline(always)]
                pub unsafe fn index_mut_unchecked (&mut self, idx: usize) -> &mut $ty {
                    self.index_mut(idx)
                }

                impl_self_fns!(
                    [$ty;$len],
                    abs: "absolute values",
                    sqrt: "square roots"
                );

                /// Gets the smallest/minimum value of the vector
                #[inline(always)]
                pub fn min (self) -> $ty {
                    *self.0.iter()
                        .reduce(|x, y| if x <= y { x } else { y })
                        .unwrap()
                }

                /// Gets the biggest/maximum value of the vector
                #[inline(always)]
                pub fn max (self) -> $ty {
                    *self.0.iter()
                        .reduce(|x, y| if x >= y { x } else { y })
                        .unwrap()
                }

                /// Sums up all the values inside the vector
                #[inline(always)]
                pub fn sum (self) -> $ty {
                    self.0.iter().sum::<$ty>()
                }

                /// Multiplies all the values inside the vector
                #[inline(always)]
                pub fn prod (self) -> $ty {
                    self.0.iter().product::<$ty>()
                }

                impl_other_fns!(
                    [$ty;$len],
                    min as vmin: "smallest/minimum value",
                    max as vmax: "biggest/maximum value"
                );

                /// Fused multiply-add. Computes `(self * a) + b` with only one rounding error.
                #[inline(always)]
                pub fn mul_add (self, rhs: Self, add: Self) -> Self {
                    Self(array(|i| self[i].mul_add(rhs[i], add[i])))
                }

                /// Interleaves elements of both vectors into one
                #[inline(always)]
                pub fn zip (self, rhs: Self) -> Self {
                    let self_ptr : *const $ty = core::ptr::addr_of!(self).cast();
                    let rhs_ptr : *const $ty = core::ptr::addr_of!(rhs).cast();

                    let mut result = MaybeUninit::<[$ty;$len]>::uninit();
                    let ptr : *mut $ty = result.as_mut_ptr().cast();

                    let mut i = 0;
                    let mut j = 0;

                    while (i < $len) {
                        unsafe {
                            ptr.add(i).write(self_ptr.add(j).read());
                            ptr.add(i+1).write(rhs_ptr.add(j).read());
                        }

                        j += 1;
                        i += 2;
                    }

                    unsafe { Self(result.assume_init()) }
                }
            }

            impl Index<usize> for $target {
                type Output = $ty;

                #[inline(always)]
                fn index (&self, idx: usize) -> &$ty {
                    self.0.index(idx)
                }
            }

            impl IndexMut<usize> for $target {
                #[inline(always)]
                fn index_mut (&mut self, idx: usize) -> &mut $ty {
                    self.0.index_mut(idx)
                }
            }

            impl PartialEq for $target {
                #[inline]
                fn eq (&self, other: &Self) -> bool {
                    self.0.iter().enumerate().all(|(i, x)| *x == other[i])
                }

                #[inline]
                fn ne (&self, other: &Self) -> bool {
                    self.0.iter().enumerate().any(|(i, x)| *x != other[i])
                }
            }

            impl From<[$ty;$len]> for $target {
                #[inline(always)]
                fn from (x: [$ty;$len]) -> Self {
                    Self(x)
                }
            }

            impl From<$ty> for $target {
                #[inline(always)]
                fn from (x: $ty) -> Self {
                    Self::filled_with(x)
                }
            }

            impl Into<[$ty;$len]> for $target {
                #[inline(always)]
                fn into (self) -> [$ty;$len] {
                    self.0
                }
            }
        )*
    };
}

impl_naive!(
    [f32;2] as f32x2,
    [f32;3] as f32x3,
    [f32;4] as f32x4,
    [f32;6] as f32x6,
    [f32;8] as f32x8,
    [f32;10] as f32x10,
    [f32;12] as f32x12,
    [f32;14] as f32x14,
    [f32;16] as f32x16,

    [f64;2] as f64x2,
    [f64;3] as f64x3,
    [f64;4] as f64x4,
    [f64;6] as f64x6,
    [f64;8] as f64x8,
    [f64;10] as f64x10,
    [f64;12] as f64x12,
    [f64;14] as f64x14,
    [f64;16] as f64x16
);