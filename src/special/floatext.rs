pub trait NoStdMath {
    fn abs (self) -> Self;
    fn sqrt (self) -> Self;
}

const ABS_MASK_F32 : u32 = i32::MAX as u32;
const ABS_MASK_F64 : u64 = i64::MAX as u64;

impl NoStdMath for f32 {
    #[inline]
    fn abs (self) -> Self {
        f32::from_bits(self.to_bits() & ABS_MASK_F32)
    }

    #[inline]
    fn sqrt (self) -> Self {
        sqrt_recip_f32(self).recip()
    }
}

impl NoStdMath for f64 {
    #[inline]
    fn abs (self) -> Self {
        f64::from_bits(self.to_bits() & ABS_MASK_F64)
    }

    #[inline]
    fn sqrt (self) -> Self {
        sqrt_recip_f64(self).recip()
    }
}

pub fn sqrt_recip_f32 (x: f32) -> f32 {
    let x2 = x / 2.;
    let i = x.to_bits();
    let i = 0x5f3759df - ( i >> 1 );

    let y = f32::from_bits(i);
    let y = y * ( 1.5 - ( x2 * y * y ) );
    y * ( 1.5 - ( x2 * y * y ) )
}

pub fn sqrt_recip_f64 (x: f64) -> f64 {
    let x2 = x / 2.;
    let i = (x as f32).to_bits();
    let i = 0x5f3759df - ( i >> 1 );

    let y = f32::from_bits(i) as f64;
    let y = y * ( 1.5 - ( x2 * y * y ) );
    let y = y * ( 1.5 - ( x2 * y * y ) );
    let y = y * ( 1.5 - ( x2 * y * y ) );
    y * ( 1.5 - ( x2 * y * y ) )
}