use llml_simd_proc::*;
use core::ops::*;
use core::intrinsics::transmute;
use core::ptr::addr_of;
use crate::float::single::f32x4;
arch_use!();

// FLOAT VECTOR 3
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Copy, Assign)]
#[assign_targets(Add, Sub, Mul, Div)]
#[assign_rhs(Self, f32)]
pub struct f32x3 (pub(crate) float32x4_t);

impl f32x3 {
    const DIV_SELECT_MASK : uint32x4_t = unsafe { transmute([0, 0, 0, u32::MAX]) };
    const DIV_BIT_MASK : float32x4_t = unsafe { transmute(0u128) };

    const SELECT_MASK : uint32x4_t = unsafe { transmute([u32::MAX, 0, 0, 0]) };
    const MIN_MASK : float32x4_t = unsafe { transmute([f32::MAX, f32::MAX, f32::MAX, f32::MAX]) };
    const MAX_MASK : float32x4_t = unsafe { transmute([f32::MIN, f32::MIN, f32::MIN, f32::MIN]) };

    /// Loads values from the pointer into the SIMD vector
    #[inline(always)]
    pub unsafe fn load (ptr: *const f32) -> Self {
        let ptr = [*ptr, *ptr.add(1), *ptr.add(2), 0.];
        Self(vld1q_f32(addr_of!(ptr).cast()))
    }

    /// Returns a vector with the absolute values of the original vector
    #[inline(always)]
    pub fn abs (self) -> Self {
        unsafe { Self(vabsq_f32(self.0)) }
    }

    /// Returns a vector with the absolute values of the original vector
    #[inline(always)]
    pub fn sqrt (self) -> Self {
        unsafe { Self(vsqrtq_f32(self.0)) }
    }

    /// Gets the smallest/minimum value of the vector
    #[inline(always)]
    pub fn min (self) -> f32 {
        unsafe { 
            let masked = vbslq_f32(Self::SELECT_MASK, Self::MIN_MASK, self.0);
            f32x4(masked).min() 
        }
    }

    /// Gets the biggest/maximum value of the vector
    #[inline(always)]
    pub fn max (self) -> f32 {
        unsafe { 
            let masked = vbslq_f32(Self::SELECT_MASK, Self::MAX_MASK, self.0);
            f32x4(masked).max()
        }
    }

    /// Sums up all the values inside the vector
    #[inline(always)]
    pub fn sum (self) -> f32 {
        f32x4(self.0).sum()
    }

    /// Returns a vector with the smallest/minimum value of each lane
    #[inline(always)]
    pub fn vmin (self, rhs: Self) -> Self {
        unsafe { Self(vminq_f32(self.0, rhs.0)) }
    }

    /// Returns a vector with the absolute values of the original vector
    #[inline(always)]
    pub fn vmax (self, rhs: Self) -> Self {
        unsafe { Self(vmaxq_f32(self.0, rhs.0)) }
    }

    /// Interleaves elements of both vectors into one
    #[inline(always)]
    pub fn zip (self, rhs: Self) -> Self {
        unsafe { 
            let zip = vzip1q_f32(self.0, rhs.0);
            Self(vbslq_f32(Self::DIV_SELECT_MASK, Self::DIV_BIT_MASK, zip))
        }
    }
}

impl Add for f32x3 {
    type Output = Self;

    #[inline(always)]
    fn add (self, rhs: Self) -> Self::Output {
        unsafe { Self(vaddq_f32(self.0, rhs.0)) }
    }
}

impl Sub for f32x3 {
    type Output = Self;

    #[inline(always)]
    fn sub (self, rhs: Self) -> Self::Output {
        unsafe { Self(vsubq_f32(self.0, rhs.0)) }
    }
}

impl Mul for f32x3 {
    type Output = Self;

    #[inline(always)]
    fn mul (self, rhs: Self) -> Self::Output {
        unsafe { Self(vmulq_f32(self.0, rhs.0)) }
    }
}

impl Div for f32x3 {
    type Output = Self;

    #[inline(always)]
    fn div (self, rhs: Self) -> Self::Output {
        unsafe {
            let div = vdivq_f32(self.0, rhs.0);
            Self(vbslq_f32(Self::DIV_SELECT_MASK, Self::DIV_BIT_MASK, div))
        }
    }
}

impl Neg for f32x3 {
    type Output = Self;

    #[inline(always)]
    fn neg (self) -> Self::Output {
        unsafe { Self(vnegq_f32(self.0)) }
    }
}

impl PartialEq for f32x3 {
    #[inline(always)]
    fn eq (&self, other: &Self) -> bool {
        unsafe {
            let cmp = vceqq_f32(self.0, other.0);
            transmute::<uint32x4_t, u128>(cmp) == u128::MAX
        }
    }
}

impl From<f32> for f32x3 {
    #[inline(always)]
    fn from(x: f32) -> Self {
        Self::new([x,x,x])
    }
}

impl_scal_arith!(
    f32x3, f32,
    Add, add,
    Sub, sub,
    Mul, mul,
    Div, div
);