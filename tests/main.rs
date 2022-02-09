use llml_simd::float::single::*;
use llml_simd::float::double::*;
use core::ops::*;
use rand::random;

macro_rules! test_other {
    ($($fun:ident $(as $name:ident)?),+) => {
        $(
            test_other!($fun, $($name,)?
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
        )*
    };

    ($fun:ident, $name:ident, $([$ty:ident;$len:literal] as $target:ident),+) => {
        #[test]
        pub fn $name() {
            $(
                let alpha : [$ty;$len] = random();
                let beta : [$ty;$len] = random();
                let naive = alpha.into_iter()
                    .zip(beta.into_iter())
                    .map(|(x, y)| x.$fun(y));
    
                let alpha = <$target>::new(alpha);
                let beta = <$target>::new(beta);
                let simd = alpha.$name(beta);
    
                Into::<[$ty;$len]>::into(simd)
                    .into_iter()
                    .zip(naive)
                    .for_each(|(simd, naive)| assert!((simd - naive).abs() <= $ty::EPSILON, stringify!($fun)));
            )*
        }
    };

    ($fun:ident, $([$ty:ident;$len:literal] as $target:ident),+) => {
        test_other!($fun, $fun $(,[$ty;$len] as $target)*);
    };
}

macro_rules! test_mappings {
    ($($fun:ident $(as $name:ident)?),+) => {
        $(
            test_mappings!($fun, $($name,)?
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
        )*
    };

    ($fun:ident, $name:ident, $([$ty:ident;$len:literal] as $target:ident),+) => {
        #[test]
        pub fn $name() {
            $(
                let alpha : [$ty;$len] = random();
                let naive = alpha.into_iter()
                    .map(|x| x.$fun());
    
                let alpha = <$target>::new(alpha);
                let simd = alpha.$name();
    
                Into::<[$ty;$len]>::into(simd)
                    .into_iter()
                    .zip(naive)
                    .for_each(|(simd, naive)| assert!((simd - naive).abs() <= $ty::EPSILON, stringify!($target)));
            )*
        }
    };

    ($fun:ident, $([$ty:ident;$len:literal] as $target:ident),+) => {
        test_mappings!($fun, $fun $(,[$ty;$len] as $target)*);
    };
}

macro_rules! test_horiz {
    ($($fun:ident $(as $name:ident)?),+) => {
        $(
            test_horiz!($fun, $($name,)?
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
        )*
    };

    ($fun:ident, $name:ident, $([$ty:ident;$len:literal] as $target:ident),+) => {
        #[test]
        pub fn $name () {
            $(
                let alpha : [$ty;$len] = random();
                let naive = alpha.into_iter()
                    .reduce(|x, y| x.$fun(y)).unwrap();

                let alpha = <$target>::new(alpha);
                let simd = alpha.$name();

                let diff = (naive - simd).abs();
                assert!(diff <= $ty::EPSILON * ($len as $ty), concat!("Comparison filed for '", stringify!($target), "'"));
            )*
        }
    };

    ($fun:ident, $([$ty:ident;$len:literal] as $target:ident),+) => {
        test_horiz!($fun, $fun $(,[$ty;$len] as $target)*);
    };
}

macro_rules! test_index {
    ($([$ty:ident;$len:literal] as $target:ident),+) => {
        $(
            let array : [$ty;$len] = random();
            let simd = $target::from(array);

            for i in 0..$len {
                assert_eq!(array[i], simd[i], concat!("Assertion failed for '", stringify!($target), "'"));
            }
        )*
    }
}

test_other!(
    add, sub, mul, div, 
    min as vmin, max as vmax
);

test_horiz!(
    min, max, add as sum
);

test_mappings!(
    neg, abs, sqrt
);

#[test]
pub fn index () {
    test_index!(
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