use crate::float::single::*;
use crate::float::double::*;
use rand::{prelude::Distribution, distributions::Standard};

macro_rules! impl_rand {
    ($([$ty:ident;$len:literal] as $target:ident),+) => {
        $(
            impl Distribution<$target> for Standard {
                #[inline(always)]
                fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> $target {
                    <[$ty;$len] as Into<$target>>::into(self.sample(rng))
                }
            }
        )*
    };
}

impl_rand!(
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