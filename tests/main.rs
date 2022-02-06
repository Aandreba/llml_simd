use llml_simd::float::single::f32x14;

#[test]
pub fn me () {
    let mut alex = f32x14::new([1., 2., 3., 4.]);
    alex += 2.;
    println!("{alex:?}")
}