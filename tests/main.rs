use llml_simd::{get_top_feature, Simdt, f32x2, f32x4};

#[test]
pub fn me () {
    let target = get_top_feature();
    println!("{:?}", target);
    
    let alpha = f32x4(1., 2., 3., 4.);
    let sum = alpha.sum();
    let prod = alpha.prod();
    println!("{:?} {:?}", sum, prod);
}