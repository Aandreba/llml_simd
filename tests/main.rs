use std::f32::consts::{PI, E};

use llml_simd::{get_top_feature, Simdt, f32x8, f32x4};

#[test]
pub fn me () {
    let target = get_top_feature();
    println!("{:?}", target);
    
    let alpha = f32x8(1., 20., -2., 4., PI, -E, f32::EPSILON, -0.);
    let sum = alpha.min();
    let prod = alpha.max();
    println!("{:?} {:?}", sum, prod);
}