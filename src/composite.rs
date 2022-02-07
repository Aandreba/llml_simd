macro_rules! impl_hoz_fns {
    ($ty:ident, $($fun:ident $(as $name:ident)?: $docs:expr),+) => {
        $(
            impl_hoz_fns!(1, $fun $(,$name)?, $ty, $docs);
        )*
    };

    (1, $fun:ident, $ty:ident, $docs:expr) => {
        #[doc=$docs]
        #[inline(always)]
        pub fn $fun (self) -> $ty {
            self.0.$fun().$fun(self.1.$fun())
        }
    };

    (1, $fun:ident, $name:ident, $ty:ident, $docs:expr) => {
        #[doc=$docs]
        #[inline(always)]
        pub fn $name (self) -> $ty {
            self.0.$name().$fun(self.1.$name())
        }
    };
}

macro_rules! impl_self_fns {
    ($ty:ident, $($fun:ident $(with $tag:ident)?: $docs:expr),+) => {
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
    }
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
            Self (
                self.0.$fun(rhs.0),
                self.1.$fun(rhs.1)
            )
        }
    };

    (1, $fun:ident, $name:ident, $ty:ident, $docs:expr, $($tag:ident)?) => {
        #[doc=concat!("Returns a vector with the ", $docs, " of each lane")]
        #[inline(always)]
        pub fn $name (self, rhs: Self) -> Self {
            Self (
                self.0.$name(rhs.0),
                self.1.$name(rhs.1)
            )
        }
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
            #[derive(Clone, Copy, Assign, Neg, PartialEq)]
            #[assign_targets(Add, Sub, Mul, Div)]
            #[assign_rhs(Self, $ty)]
            pub struct $name($x, $y);
    
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
                    $ty,
                    abs: "absolute values",
                    sqrt: "square roots"
                );

                impl_hoz_fns!(
                    $ty,
                    min: "Gets the smallest/minimum value of the vector",
                    max: "Gets the biggest/maximum value of the vector",
                    add as sum: "Sums up all the values inside the vector"
                );

                impl_other_fns!(
                    $ty,
                    min as vmin: "smallest/minimum value",
                    max as vmax: "biggest/maximum value"
                );
            }
    
            impl From<$ty> for $name {
                #[inline(always)]
                fn from(x: $ty) -> Self {
                    Self(x.into(), x.into())
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
            pub struct $name($x, $y, $z);
    
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

                /*impl_self_fns!(
                    $ty,
                    abs: "absolute values",
                    sqrt: "square roots"
                );

                impl_hoz_fns!(
                    $ty,
                    min: "Gets the smallest/minimum value of the vector",
                    max: "Gets the biggest/maximum value of the vector",
                    add as sum: "Sums up all the values inside the vector"
                );

                impl_other_fns!(
                    $ty,
                    min as vmin: "smallest/minimum value",
                    max as vmax: "biggest/maximum value"
                );*/
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
            pub struct $name($x, $y, $z, $w);
    
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
                    let step = $lx + $ly;
                    Self (
                        <$x>::load(ptr),
                        <$y>::load(ptr.add($lx)),
                        <$z>::load(ptr.add(step)),
                        <$w>::load(ptr.add(step + $lz)),
                    )
                }

                /*impl_self_fns!(
                    $ty,
                    abs: "absolute values",
                    sqrt: "square roots"
                );

                impl_hoz_fns!(
                    $ty,
                    min: "Gets the smallest/minimum value of the vector",
                    max: "Gets the biggest/maximum value of the vector",
                    add as sum: "Sums up all the values inside the vector"
                );

                impl_other_fns!(
                    $ty,
                    min as vmin: "smallest/minimum value",
                    max as vmax: "biggest/maximum value"
                );*/
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