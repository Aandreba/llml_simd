use criterion::{criterion_group, criterion_main, Criterion};
use llml_simd::float::{single::*, double::*};
use llml_simd_proc::*;
use core::ops::*;
use rand::random;

macro_rules! bench_other {
    ($($fun:ident),+) => {
        $(
            bench_other!($fun,
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

    ($fun:ident, $([$ty:ident;$len:literal] as $target:ident),+) => {
        pub fn $fun(c: &mut Criterion) {
            $(
                c.bench_function(concat!(stringify!($fun), " for ", stringify!($target)), |b| {
                    let alpha : $target = random();
                    let beta : $target = random();
                    b.iter(|| arr![|i| alpha.$fun(beta); 1000])
                });
            )*
        }
    };
}

macro_rules! bench_horiz {
    ($($fun:ident),+) => {
        $(
            bench_horiz!($fun,
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

    ($fun:ident, $([$ty:ident;$len:literal] as $target:ident),+) => {
        pub fn $fun (c: &mut Criterion) {
            $(
                c.bench_function(concat!(stringify!($fun), " for ", stringify!($target)), |b| {
                    let alpha : $target = random();
                    b.iter(|| arr![|i| alpha.$fun(); 1000])
                });
            )*
        }
    };
}

macro_rules! bench_fma {
    () => {
        bench_fma!(
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
    };

    ($([$ty:ident;$len:literal] as $target:ident),+) => {
        pub fn mul_add(c: &mut Criterion) {
            $(
                c.bench_function(concat!("mul_add for ", stringify!($target)), |b| {
                    let alpha : $target = random();
                    let beta : $target = random();
                    let gamma : $target = random();
                    b.iter(|| arr![|i| alpha.mul_add(beta, gamma); 1000])
                });
            )*
        }
    };
}

bench_other!(add, sub, mul, div, vmin, vmax);
bench_horiz!(min, max, sum, prod);
bench_fma!();

criterion_group!(benches, 
    add, sub, mul, div, vmin, vmax,
    min, max, sum, prod, mul_add
);

criterion_main!(benches);