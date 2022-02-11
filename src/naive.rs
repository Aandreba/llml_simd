use core::ops::*;
use llml_simd_proc::*;

macro_rules! impl_self_fns {
    ([$ty:ident;$len:literal], $($fun:ident $(as $name:ident)?: $docs:expr),+) => {
        $(
            impl_self_fns!(1, $fun $(,$name)?, $ty, $len, $docs);
        )*
    };

    (1, $fun:ident, $ty:ident, $len:literal, $docs:expr) => {
        #[doc=concat!("Returns a vector with the ", $docs, " of the original vector")]
        #[inline(always)]
        pub fn $fun (self) -> Self {
            Self(arr![|i| (self[i] as $ty).$fun();$len])
        }
    };

    (1, $fun:ident, $name:ident, $ty:ident, $len:literal, $docs:expr) => {
        #[doc=concat!("Returns a vector with the ", $docs, " of the original vector")]
        #[inline(always)]
        pub fn $name (self) -> Self {
            Self(arr![|i| (self[i] as $ty).$fun();$len])
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
            Self(arr![|i| (self[i] as $ty).$fun(rhs[i] as $ty);$len])
        }
    };

    (1, $fun:ident, $name:ident, $ty:ident, $len:literal, $docs:expr) => {
        #[doc=concat!("Returns a vector with the ", $docs, " of each lane")]
        #[inline(always)]
        pub fn $name (self, rhs: Self) -> Self {
            Self(arr![|i| (self[i] as $ty).$fun(rhs[i] as $ty);$len])
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
                    let arr = arr![|i| (self.0[i] as $ty).$fun(rhs.0[i]); $len];
                    Self(arr)
                }
            }

            impl $trait<$ty> for $target {
                type Output = Self;

                #[inline(always)]
                fn $fun (self, rhs: $ty) -> Self::Output {
                    Self(arr![|i| (self.0[i] as $ty).$fun(rhs); $len])
                }
            }

            impl $trait<$target> for $ty {
                type Output = $target;

                #[inline(always)]
                fn $fun (self, rhs: $target) -> Self::Output {
                    $target(arr![|i| self.$fun(rhs.0[i]); $len])
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
                    Self(arr![|i| self[i].neg();$len])
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

                impl_other_fns!(
                    [$ty;$len],
                    min as vmin: "smallest/minimum value",
                    max as vmax: "biggest/maximum value"
                );
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

            impl Into<[$ty;$len]> for $target {
                #[inline(always)]
                fn into (self) -> [$ty;$len] {
                    self.0
                }
            }
        )*
    };
}

cfg_if::cfg_if! {
    if #[cfg(not(feature = "use_std"))] {
        use core::mem::transmute;

        trait FloatExt {
            fn abs(self) -> Self;
            fn sqrt(self) -> Self;
        }
        
        impl FloatExt for f32 {
            #[inline(always)]
            fn abs (self) -> f32 {
                unsafe { transmute::<i32,f32>(transmute::<f32,i32>(self) & i32::MAX) }
            }
        
            #[inline(always)]
            fn sqrt (self) -> f32 {
                todo!()
            }
        }

        impl FloatExt for f64 {
            #[inline(always)]
            fn abs (self) -> f64 {
                unsafe { transmute::<i64,f64>(transmute::<f64,i64>(self) & i64::MAX) }
            }
        
            #[inline(always)]
            fn sqrt (self) -> f64 {
                todo!()
            }
        }
    }
}

impl_naive!(
    [f32;2] as f32x2,
    [f32;4] as f32x4,
    [f32;6] as f32x6,
    [f32;8] as f32x8,
    [f32;10] as f32x10,
    [f32;12] as f32x12,
    [f32;14] as f32x14,
    [f32;16] as f32x16,

    [f64;2] as f64x2,
    [f64;4] as f64x4,
    [f64;6] as f64x6,
    [f64;8] as f64x8,
    [f64;10] as f64x10,
    [f64;12] as f64x12,
    [f64;14] as f64x14,
    [f64;16] as f64x16
);