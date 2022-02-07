use core::ops::*;
use core::mem::transmute;
use core::ptr::addr_of;
use llml_simd_proc::*;
use crate::float::single::*;
use crate::float::double::*;
use derive_more::Neg;
arch_use!();

impl_straight!(
    __m256 as f32x6 with m256 => [f32;6],
    __m256 as f32x8 with m256 => [f32;8],
    __m256d as f64x4 with m256d => [f64;4]
);

impl_clone!(
    f32x6, f32, 6,
    f32x8, f32, 8,
    f64x4, f64, 4
);

impl_composite!(
    (f32x8 => 8, f32x2 => 2) as f32x10: f32,
    (f32x8 => 8, f32x4 => 4) as f32x12: f32,
    (f32x8 => 8, f32x6 => 6) as f32x14: f32,
    (f32x8 => 8, f32x8 => 6) as f32x16: f32
);