use llml_simd::{get_top_feature, Simdt, f32x8};

#[test]
pub fn me () {
    let target = get_top_feature();
    println!("{:?}", target);
    
    let alpha = f32x8(1., 2., 3., 4., 5., 6., 7., 8.);
    let sum = alpha.sum();
    let prod = alpha.prod();
    println!("{:?} {:?}", sum, prod);
}