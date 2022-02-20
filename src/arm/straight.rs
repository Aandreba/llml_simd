use core::ops::*;
use llml_simd_proc::{assign_targets, assign_rhs, Assign};
use core::mem::transmute;
arch_use!();

macro_rules! impl_arith {
    ($target:ident, $ty:ident, $($trait:ident, $fun:ident $(with $tag:ident)?),+) => {
        $(
            impl $trait for $target {
                type Output = Self;
    
                #[inline(always)]
                fn $fun (self, rhs: Self) -> Self::Output {
                    unsafe { Self(concat_idents!(v, $fun, $($tag,)? _, $ty)(self.0, rhs.0)) }
                }
            }
    
            impl $trait<$ty> for $target {
                type Output = Self;
    
                #[inline(always)]
                fn $fun (self, rhs: $ty) -> Self::Output {
                    self.$fun(Into::<$target>::into(rhs))
                }
            }

            impl $trait<$target> for $ty {
                type Output = $target;
    
                #[inline(always)]
                fn $fun (self, rhs: $target) -> Self::Output {
                    Into::<$target>::into(self).$fun(rhs)
                }
            }
        )*
    };
}

macro_rules! impl_hoz_fns {
    ($ty:ident, $($fun:ident $(as $name:ident)? $(with $tag:ident)?: $docs:expr),+) => {
        $(
            impl_hoz_fns!(1, $fun $(,$name)?, $ty, $docs ,$($tag)?);
        )*
    };

    (1, $fun:ident, $ty:ident, $docs:expr, $($tag:ident)?) => {
        #[doc=$docs]
        #[inline(always)]
        pub fn $fun (self) -> $ty {
            unsafe { concat_idents!(v, $fun, v, $($tag,)? _, $ty)(self.0) }
        }
    };

    (1, $fun:ident, $name:ident, $ty:ident, $docs:expr, $($tag:ident)?) => {
        #[doc=$docs]
        #[inline(always)]
        pub fn $name (self) -> $ty {
            unsafe { concat_idents!(v, $fun, v, $($tag,)? _, $ty)(self.0) }
        }
    };
}

macro_rules! impl_self_fns {
    ($ty:ident, $($fun:ident $(as $name:ident)? $(with $tag:ident)?: $docs:expr),+) => {
        $(
            impl_self_fns!(1, $fun $(, $name)?, $ty, $docs, $($tag)?);
        )*
    };

    (1, $fun:ident, $ty:ident, $docs:expr, $($tag:ident)?) => {
        #[doc=concat!("Returns a vector with the ", $docs, " of the original vector")]
        #[inline(always)]
        pub fn $fun (self) -> Self {
            unsafe { Self(concat_idents!(v, $fun, $($tag,)? _, $ty)(self.0)) }
        }
    };

    (1, $fun:ident, $name:ident, $ty:ident, $docs:expr, $($tag:ident)?) => {
        #[doc=concat!("Returns a vector with the ", $docs, " of the original vector")]
        #[inline(always)]
        pub fn $name (self) -> Self {
            unsafe { Self(concat_idents!(v, $fun, $($tag,)? _, $ty)(self.0)) }
        }
    };
}

macro_rules! impl_other_fns {
    ($ty:ident, $($fun:ident $(as $name:ident)? $(with $tag:ident)?: $docs:expr),+) => {
        $(
            impl_other_fns!(1, $fun $(, $name)?, $ty, $docs, $($tag)?);
        )*
    };

    (1, $fun:ident, $ty:ident, $docs:expr, $($tag:ident)?) => {
        #[doc=concat!("Returns a vector with the ", $docs, " of each lane")]
        #[inline(always)]
        pub fn $fun (self, rhs: Self) -> Self {
            unsafe { Self(concat_idents!(v, $fun, $($tag,)? _, $ty)(self.0, rhs.0)) }
        }
    };

    (1, $fun:ident, $name:ident, $ty:ident, $docs:expr, $($tag:ident)?) => {
        #[doc=concat!("Returns a vector with the ", $docs, " of each lane")]
        #[inline(always)]
        pub fn $name (self, rhs: Self) -> Self {
            unsafe { Self(concat_idents!(v, $fun, $($tag,)? _, $ty)(self.0, rhs.0)) }
        }
    };
}

macro_rules! impl_straight {
    ($(($og:ident => $og_mask:ident) as $name:ident, [$ty:ident => $mask:ident;$len:literal] $(with $tag:ident)?),+) => {
        $(
            #[allow(non_camel_case_types)]
            #[repr(transparent)]
            #[derive(Copy, Assign)]
            #[assign_targets(Add, Sub, Mul, Div)]
            #[assign_rhs(Self, $ty)]
            pub struct $name(pub(crate) $og);

            impl_arith!(
                $name, $ty,
                Add, add $(with $tag)?,
                Sub, sub $(with $tag)?,
                Mul, mul $(with $tag)?,
                Div, div $(with $tag)?
            );

            impl Neg for $name {
                type Output = Self;

                #[inline(always)]
                fn neg(self) -> Self::Output {
                    unsafe { Self(concat_idents!(vneg, $($tag,)? _, $ty)(self.0)) }
                }
            }

            impl $name {
                const CMP_MASK : $og_mask = unsafe { transmute([$mask::MAX;$len]) };

                /// Loads values from the pointer into the SIMD vector
                #[inline(always)]
                pub unsafe fn load (ptr: *const $ty) -> Self {
                    Self(concat_idents!(vld1, $($tag,)? _, $ty)(ptr))
                }

                impl_self_fns!(
                    $ty,
                    abs $(with $tag)?: "absolute values",
                    sqrt $(with $tag)?: "square roots"
                );

                impl_hoz_fns!(
                    $ty,
                    min $(with $tag)?: "Gets the smallest/minimum value of the vector",
                    max $(with $tag)?: "Gets the biggest/maximum value of the vector",
                    add as sum $(with $tag)?: "Sums up all the values inside the vector"
                );

                impl_other_fns!(
                    $ty,
                    min as vmin $(with $tag)?: "smallest/minimum value",
                    max as vmax $(with $tag)?: "biggest/maximum value"
                );

                /// Interleaves elements of both vectors into one
                #[inline(always)]
                pub fn zip (self, rhs: Self) -> Self {
                    unsafe { Self(concat_idents!(vzip1, $($tag,)? _, $ty)(self.0, rhs.0)) }
                }
            }

            impl From<$ty> for $name {
                #[inline(always)]
                fn from(x: $ty) -> Self {
                    unsafe { Self(concat_idents!(vld1, $($tag,)? _dup_, $ty)(&x)) }
                }
            }

            impl PartialEq for $name {
                #[inline(always)]
                fn eq (&self, rhs: &Self) -> bool {
                    unsafe {
                        let cmp : $og_mask = transmute(concat_idents!(vceq, $($tag,)? _, $ty)(self.0, rhs.0));
                        cmp == Self::CMP_MASK
                    }
                }
            }
        )*
    };
}

impl_straight!(
    (float32x2_t => u64) as f32x2, [f32 => u32; 2],
    (float32x4_t => u128) as f32x4, [f32 => u32; 4] with q,
    (float64x2_t => u128) as f64x2, [f64 => u64; 2] with q
);