use std::f32::consts::{PI, E};
use llml_simd::{get_top_feature, Simdt, f32x32};
use rand::random;

#[test]
pub fn me () {
    let target = get_top_feature();
    println!("{:?}", target);
    
    let alpha = f32x32(random());
    println!("{:?} {:?}", alpha.min(), alpha.max());
    println!("{:?}", alpha)
}