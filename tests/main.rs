use llml_simd::float::single::*;
use llml_simd::float::double::*;
use llml_simd_proc::arr;
use core::ops::*;
use rand::random;

macro_rules! test_other {
    ($($fun:ident),+) => {
        $(
            #[test]
            pub fn $fun () {
                test_other!($fun,
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
            }
        )*
    };

    ($fun:ident, $([$ty:ident;$len:literal] as $target:ident),+) => {
        $(
            let alpha : [$ty;$len] = random();
            let beta : [$ty;$len] = random();
            let naive = alpha.into_iter()
                .zip(beta.into_iter())
                .map(|(x, y)| x.$fun(y));

            let alpha = <$target>::new(alpha);
            let beta = <$target>::new(beta);
            let simd = alpha.$fun(beta);

            Into::<[$ty;$len]>::into(simd)
                .into_iter()
                .zip(naive)
                .for_each(|(simd, naive)| assert_eq!(simd, naive, concat!("'", stringify!($fun), "' failed for '", stringify!($target), "'")));
        )*
    };
}

test_other!(add, sub, mul, div);