use llml_simd::{get_top_feature, f32x8, Simdt};

#[test]
pub fn me () {
    let target = get_top_feature();
    println!("{:?}", target);
    
    let alpha = f32x8(
        1., 2., 3., 4.,
        5., 6., 7., 8.
    );
    
    println!("{:?} {:?}", alpha.sum(), alpha.prod())
}