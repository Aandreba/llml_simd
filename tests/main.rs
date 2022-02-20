#![feature(concat_idents)]
use llml_simd::float::single::*;
use llml_simd::float::double::*;
use core::ops::*;
use rand::random;

macro_rules! test_other {
    ($($fun:ident $(as $name:ident)?),+) => {
        $(
            test_other!($fun, $($name,)?
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
                    .for_each(|(simd, naive)| assert!((simd - naive).abs() <= $ty::EPSILON * 2., stringify!($fun)));
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
                [f32;3] as f32x3,
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
                    .for_each(|(simd, naive)| assert!((simd - naive).abs() <= $ty::EPSILON * 2., stringify!($target)));
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
                [f32;3] as f32x3,
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
                if diff > ($ty::EPSILON + 1.) * ($len as $ty) {
                    panic!("{0} error for '{1}': expected {simd}, got {naive}", stringify!($fun), stringify!($target));
                }
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

macro_rules! test_rand {
    ($([$ty:ident;$len:literal] as $target:ident),+) => {
        $(
            let array : $target = random();
            assert_eq!(array, array);
        )*
    }
}

macro_rules! test_serde {
    ($([$ty:ident;$len:literal] as $target:ident),+) => {
        $(
            let test : $target = random::<[$ty;$len]>().into();
            let ser = serde_json::to_string(&test).unwrap();
            let de : $target = serde_json::from_str(&ser).unwrap();
            assert!((test - de).abs().sum() <= $ty::EPSILON * ($len as $ty));
        )*
    }
}

macro_rules! test_into {
    ($([$ty:ident;$len:literal] as $target:ident),+) => {
        $(
            let array : [$ty;$len] = random();
            let into : $target = array.into();
            let from : [$ty;$len] = into.into();
            assert_eq!(array, from);
        )*
    }
}

macro_rules! test_from {
    ($([$ty:ident;$len:literal] as $target:ident),+) => {
        $(
            let scalar : $ty = random();
            let vec : $target = scalar.into();

            for i in 0..$len {
                assert_eq!(vec[i], scalar);
            }
        )*
    }
}

macro_rules! test_clone {
    ($([$ty:ident;$len:literal] as $target:ident),+) => {
        $(
            let array : [$ty;$len] = random();
            let alpha : $target = array.into();
            let beta = alpha.clone();
            assert_eq!(alpha, beta, stringify!($target));
        )*
    }
}

macro_rules! test_eq {
    ($([$ty:ident;$len:literal] as $target:ident),+) => {
        $(
            let first_array : [$ty;$len] = random();
            let alpha : $target = first_array.into();
            let beta = alpha.clone();

            let mut last_array : [$ty;$len];
            loop {
                last_array = random();
                if !last_array.iter().enumerate().all(|(i, x)| *x == first_array[i]) { break }
            }
            let gamma : $target = last_array.into();

            assert_eq!(alpha, beta, stringify!($target));
            assert_ne!(alpha, gamma);
        )*
    }
}

macro_rules! test_transpose {
    ($([$ty:ident;$len:literal] as $target:ident),+) => {
        $(
            let alpha_array : [$ty;$len] = random();
            let alpha : $target = alpha_array.into();

            let beta_array : [$ty;$len] = random();
            let beta : $target = beta_array.into();

            let mut result = [(0 as $ty);$len];
            for i in 0..($len/2) {
                result[2 * i] = alpha[i];
                result[2 * i + 1] = beta[i];
            }

            assert_eq!(Into::<[$ty;$len]>::into(alpha.zip(beta)), result);
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
}

#[test]
pub fn eq () {
    test_eq!(
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
}

#[cfg(feature = "random")]
#[test]
pub fn rnd () {
    test_rand!(
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
}

#[cfg(feature = "serialize")]
#[test]
pub fn serialize () {
    test_serde!(
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
}

#[test]
pub fn clone () {
    test_clone!(
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
}

#[test]
pub fn into () {
    test_into!(
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
}

#[test]
pub fn from () {
    test_from!(
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
}

#[test]
pub fn transp () {
    test_transpose!(
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

    // f32x3
    let alpha_array : [f32;3] = random();
    let alpha : f32x3 = alpha_array.into();

    let beta_array : [f32;3] = random();
    let beta : f32x3 = beta_array.into();

    let result = [alpha[0], beta[0], alpha[1]];
    assert_eq!(Into::<[f32;3]>::into(alpha.zip(beta)), result);

    // f64x3
    let alpha_array : [f64;3] = random();
    let alpha : f64x3 = alpha_array.into();

    let beta_array : [f64;3] = random();
    let beta : f64x3 = beta_array.into();

    let result = [alpha[0], beta[0], alpha[1]];
    assert_eq!(Into::<[f64;3]>::into(alpha.zip(beta)), result);
}

// vtrn1q_f32