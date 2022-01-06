use std::intrinsics::transmute;
use llml_simd::Simdt;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use llml_simd::f32x8;
use rand::random;

struct Vec8 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let alpha = f32x8(
        random(), random(), random(), random(),
        random(), random(), random(), random()
    );

    let beta : Vec8 = unsafe { transmute(alpha) };

    c.bench_function("naive sum f32x8", |b| 
        b.iter(|| beta.x + beta.y + beta.z + beta.w + beta.a + beta.b + beta.c + beta.d)
    );

    c.bench_function("simd sum f32x8", |b| 
        b.iter(|| alpha.sum())
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);