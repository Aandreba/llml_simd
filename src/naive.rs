macro_rules! impl_naive {
    ($([$ty:ident;$len:literal] as $target:ident),+) => {
        $(
            #[allow(non_camel_case_types)]
            #[repr(transparent)]
            #[derive(Clone, Copy, Add, Sub, Mul, Div, Assign)]
            #[assign_targets(Add, Sub, Mul, Div)]
            #[assign_rhs(Self, $ty)]
            pub struct $name($og);
        )*
    };
}

impl_naive!(
    f32x2
);