macro_rules! impl_scal_arith {
    ($target:ident, $ty:ident, $($trait:ident, $fun:ident),+) => {
        $(
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
    (2, $ty:ident, $($fun:ident as $name:ident, $docs:expr),+) => {
        $(
            #[doc=$docs]
            #[inline(always)]
            pub fn $name (self) -> $ty {
                self.0.$name().$fun(self.1.$name())
            }
        )*
    };

    (3, f32, $($fun:ident, $docs:expr),+) => {
        $(
            #[doc=$docs]
            #[inline(always)]
            pub fn $fun (self) -> f32 {
                let array = [self.0.$fun(), self.1.$fun(), self.2.$fun(), 0.];
                f32x4::from(array).$fun()
            }
        )*
    };

    (4, f32, $($fun:ident, $docs:expr),+) => {
        $(
            #[doc=$docs]
            #[inline(always)]
            pub fn $fun (self) -> f32 {
                let array = [self.0.$fun(), self.1.$fun(), self.2.$fun(), self.3.$fun()];
                f32x4::from(array).$fun()
            }
        )*
    };

    (3, f64, $($fun:ident, $docs:expr),+) => {
        $(
            #[doc=$docs]
            #[inline(always)]
            pub fn $fun (self) -> f64 {
                let array = [self.0.$fun(), self.1.$fun(), self.2.$fun(), 0.];
                f64x4::from(array).$fun()
            }
        )*
    };

    (4, f64, $($fun:ident, $docs:expr),+) => {
        $(
            #[doc=$docs]
            #[inline(always)]
            pub fn $fun (self) -> f64 {
                let array = [self.0.$fun(), self.1.$fun(), self.2.$fun(), self.3.$fun()];
                f64x4::from(array).$fun()
            }
        )*
    };
}

macro_rules! impl_self_fns {
    (2, $ty:ident, $($fun:ident $(with $tag:ident)?: $docs:expr),+) => {
        $(
            #[doc=concat!("Returns a vector with the ", $docs, " of the original vector")]
            #[inline(always)]
            pub fn $fun (self) -> Self {
                Self (
                    self.0.$fun(),
                    self.1.$fun()
                )
            }
        )*
    };

    (3, $ty:ident, $($fun:ident $(with $tag:ident)?: $docs:expr),+) => {
        $(
            #[doc=concat!("Returns a vector with the ", $docs, " of the original vector")]
            #[inline(always)]
            pub fn $fun (self) -> Self {
                Self (
                    self.0.$fun(),
                    self.1.$fun(),
                    self.2.$fun()
                )
            }
        )*
    };

    (4, $ty:ident, $($fun:ident $(with $tag:ident)?: $docs:expr),+) => {
        $(
            #[doc=concat!("Returns a vector with the ", $docs, " of the original vector")]
            #[inline(always)]
            pub fn $fun (self) -> Self {
                Self (
                    self.0.$fun(),
                    self.1.$fun(),
                    self.2.$fun(),
                    self.3.$fun()
                )
            }
        )*
    }
}

macro_rules! impl_other_fns {
    (2, $($fun:ident, $docs:expr),+) => {
        $(
            #[doc=concat!("Returns a vector with the ", $docs, " of each lane")]
            #[inline(always)]
            pub fn $fun (self, rhs: Self) -> Self {
                Self (
                    self.0.$fun(rhs.0),
                    self.1.$fun(rhs.1)
                )
            }
        )*
    };

    (3, $($fun:ident, $docs:expr),+) => {
        $(
            #[doc=concat!("Returns a vector with the ", $docs, " of each lane")]
            #[inline(always)]
            pub fn $fun (self, rhs: Self) -> Self {
                Self (
                    self.0.$fun(rhs.0),
                    self.1.$fun(rhs.1),
                    self.2.$fun(rhs.2)
                )
            }
        )*
    };

    (4, $($fun:ident, $docs:expr),+) => {
        $(
            #[doc=concat!("Returns a vector with the ", $docs, " of each lane")]
            #[inline(always)]
            pub fn $fun (self, rhs: Self) -> Self {
                Self (
                    self.0.$fun(rhs.0),
                    self.1.$fun(rhs.1),
                    self.2.$fun(rhs.2),
                    self.3.$fun(rhs.3)
                )
            }
        )*
    };
}

macro_rules! impl_composite {
    (@arith2 $target:ident, $ty:ident, $($trait:ident, $fun:ident),+) => {
        $(
            impl $trait for $target {
                type Output = Self;
    
                #[inline(always)]
                fn $fun (self, rhs: Self) -> Self::Output {
                    Self(
                        self.0.$fun(rhs.0),
                        self.1.$fun(rhs.1),
                    )
                }
            }

            impl_scal_arith!($target, $ty, $trait, $fun);
        )*
    };

    (@arith3 $target:ident, $ty:ident, $($trait:ident, $fun:ident),+) => {
        $(
            impl $trait for $target {
                type Output = Self;
    
                #[inline(always)]
                fn $fun (self, rhs: Self) -> Self::Output {
                    Self(
                        self.0.$fun(rhs.0),
                        self.1.$fun(rhs.1),
                        self.2.$fun(rhs.2),
                    )
                }
            }

            impl_scal_arith!($target, $ty, $trait, $fun);
        )*
    };

    (@arith4 $target:ident, $ty:ident, $($trait:ident, $fun:ident),+) => {
        $(
            impl $trait for $target {
                type Output = Self;
    
                #[inline(always)]
                fn $fun (self, rhs: Self) -> Self::Output {
                    Self(
                        self.0.$fun(rhs.0),
                        self.1.$fun(rhs.1),
                        self.2.$fun(rhs.2),
                        self.3.$fun(rhs.3),
                    )
                }
            }

            impl_scal_arith!($target, $ty, $trait, $fun);
        )*
    };

    ($(($x:ident => $lx:literal, $y:ident => $ly:literal) as $name:ident: $ty:ident),*) => {
        $(
            #[allow(non_camel_case_types)]
            #[repr(C)]
            #[derive(Clone, Copy, Assign, Neg, PartialEq)]
            #[assign_targets(Add, Sub, Mul, Div)]
            #[assign_rhs(Self, $ty)]
            pub struct $name(pub(crate) $x, pub(crate) $y);
    
            impl_composite!(
                @arith2 $name, $ty,
                Add, add,
                Sub, sub,
                Mul, mul,
                Div, div
            );

            impl $name {
                /// Loads values from the pointer into the SIMD vector
                #[inline(always)]
                pub unsafe fn load (ptr: *const $ty) -> Self {
                    Self (
                        <$x>::load(ptr),
                        <$y>::load(ptr.add($lx))
                    )
                }

                impl_self_fns!(
                    2, $ty,
                    abs: "absolute values",
                    sqrt: "square roots"
                );

                impl_hoz_fns!(
                    2, $ty,
                    min as min, "Gets the smallest/minimum value of the vector",
                    max as max, "Gets the biggest/maximum value of the vector",
                    add as sum, "Sums up all the values inside the vector"
                );

                impl_other_fns!(
                    2,
                    vmin, "smallest/minimum value",
                    vmax, "biggest/maximum value"
                );
            }
    
            impl From<$ty> for $name {
                #[inline(always)]
                fn from(x: $ty) -> Self {
                    Self(Into::<$x>::into(x), Into::<$y>::into(x))
                }
            }
        )*
    };

    ($(($x:ident => $lx:literal, $y:ident => $ly:literal, $z:ident => $lz:literal) as $name:ident: $ty:ident),*) => {
        $(
            #[allow(non_camel_case_types)]
            #[derive(Clone, Copy, Assign, Neg, PartialEq)]
            #[assign_targets(Add, Sub, Mul, Div)]
            #[assign_rhs(Self, $ty)]
            pub struct $name(pub(crate) $x, pub(crate) $y, pub(crate) $z);
    
            impl_composite!(
                @arith3 $name, $ty,
                Add, add,
                Sub, sub,
                Mul, mul,
                Div, div
            );

            impl $name {
                /// Loads values from the pointer into the SIMD vector
                #[inline(always)]
                pub unsafe fn load (ptr: *const $ty) -> Self {
                    Self (
                        <$x>::load(ptr),
                        <$y>::load(ptr.add($lx)),
                        <$z>::load(ptr.add($lx + $ly))
                    )
                }

                impl_self_fns!(
                    3, $ty,
                    abs: "absolute values",
                    sqrt: "square roots"
                );

                #[doc="Gets the smallest/minimum value of the vector"]
                #[inline(always)]
                pub fn min (self) -> $ty {
                    let array = [self.0.min(), self.1.min(), self.2.min(), $ty::MAX];
                    <concat_idents!($ty, x4)>::from(array).min()
                }

                #[doc="Gets the biggest/maximum value of the vector"]
                #[inline(always)]
                pub fn max (self) -> $ty {
                    let array = [self.0.max(), self.1.max(), self.2.max(), $ty::MIN];
                    <concat_idents!($ty, x4)>::from(array).max()
                }

                #[doc="Sums up all the values inside the vector"]
                #[inline(always)]
                pub fn sum (self) -> $ty {
                    let array = [self.0.sum(), self.1.sum(), self.2.sum(), 0.];
                    <concat_idents!($ty, x4)>::from(array).sum()
                }

                impl_other_fns!(
                    3,
                    vmin, "smallest/minimum value",
                    vmax, "biggest/maximum value"
                );
            }
    
            impl From<$ty> for $name {
                #[inline(always)]
                fn from(x: $ty) -> Self {
                    Self(x.into(), x.into(), x.into())
                }
            }
        )*
    };

    ($(($x:ident => $lx:literal, $y:ident => $ly:literal, $z:ident => $lz:literal, $w:ident => $lw:literal) as $name:ident: $ty:ident),*) => {
        $(
            #[allow(non_camel_case_types)]
            #[derive(Clone, Copy, Assign, Neg, PartialEq)]
            #[assign_targets(Add, Sub, Mul, Div)]
            #[assign_rhs(Self, $ty)]
            pub struct $name(pub(crate) $x, pub(crate) $y, pub(crate) $z, pub(crate) $w);
    
            impl_composite!(
                @arith4 $name, $ty,
                Add, add,
                Sub, sub,
                Mul, mul,
                Div, div
            );

            impl $name {
                /// Loads values from the pointer into the SIMD vector
                #[inline(always)]
                pub unsafe fn load (ptr: *const $ty) -> Self {
                    Self (
                        <$x>::load(ptr),
                        <$y>::load(ptr.add($lx)),
                        <$z>::load(ptr.add($lx + $ly)),
                        <$w>::load(ptr.add($lx + $ly + $lz)),
                    )
                }

                impl_self_fns!(
                    4, $ty,
                    abs: "absolute values",
                    sqrt: "square roots"
                );

                impl_hoz_fns!(
                    4, $ty,
                    min, "Gets the smallest/minimum value of the vector",
                    max, "Gets the biggest/maximum value of the vector",
                    sum, "Sums up all the values inside the vector"
                );

                impl_other_fns!(
                    4,
                    vmin, "smallest/minimum value",
                    vmax, "biggest/maximum value"
                );
            }
    
            impl From<$ty> for $name {
                #[inline(always)]
                fn from(x: $ty) -> Self {
                    Self(x.into(), x.into(), x.into(), x.into())
                }
            }
        )*
    };
}