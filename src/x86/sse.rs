use core::ops::*;
use core::mem::transmute;
use core::ptr::addr_of;
use llml_simd_proc::*;
use derive_more::*;
arch_use!();

impl_straight!(
    __m128 as f32x2 => [f32;2],
    __m128 as f32x4 => [f32;4],
    __m128d as f64x2 => [f64;2]
);

impl_composite!(
    (f32x4 => 4, f32x2 => 2) as f32x6: f32,
    (f32x4 => 4, f32x4 => 4) as f32x8: f32,
    (f64x2 => 2, f64x2 => 2) as f64x4: f64,
    (f64x4 => 4, f64x2 => 2) as f64x6: f64,
    (f64x4 => 4, f64x4 => 4) as f64x8: f64,
    (f64x6 => 6, f64x4 => 4) as f64x10: f64,
    (f64x6 => 6, f64x6 => 6) as f64x12: f64,
    (f64x8 => 8, f64x6 => 6) as f64x14: f64,
    (f64x8 => 8, f64x8 => 8) as f64x16: f64
);

impl_composite!(
    (f32x4 => 4, f32x4 => 4, f32x2 => 2) as f32x10: f32,
    (f32x4 => 4, f32x4 => 4, f32x4 => 4) as f32x12: f32
);

impl_composite!(
    (f32x4 => 4, f32x4 => 4, f32x4 => 4, f32x2 => 2) as f32x14: f32,
    (f32x4 => 4, f32x4 => 4, f32x4 => 4, f32x4 => 4) as f32x16: f32
);