use llml_simd::{f32x4, get_top_feature};

#[test]
pub fn me () {
    let target = get_top_feature();
    println!("{:?}", target);
    
    let alpha = f32x4(1., 2., 3., 4.);
    println!("{:?} {:?}", alpha.sum(), alpha.prod())
}