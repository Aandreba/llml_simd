use core::ops::*;
use core::mem::transmute;
use core::ptr::addr_of;
use llml_simd_proc::*;
use crate::float::single::f32x4;
arch_use!();

#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Copy, Assign)]
#[assign_targets(Add, Sub, Mul, Div)]
#[assign_rhs(Self, f32)]
pub struct f32x2 (pub(crate) __m128);

impl f32x2 {
    const DIV_MASK : __m128 = unsafe { transmute([u32::MAX, u32::MAX, 0, 0]) };
    const ABS_MASK : __m128 = unsafe { transmute([i32::MAX, i32::MAX, 0, 0]) };

    /// Loads values from the pointer into the SIMD vector
    #[inline(always)]
    pub unsafe fn load (ptr: *const f32) -> Self {
        Self(_mm_set_ps(0., 0., *ptr.add(1), *ptr))
    }

    /// Returns a vector with the absolute values of the original vector
    #[inline(always)]
    pub fn abs (self) -> Self {
        unsafe { Self(_mm_and_ps(Self::ABS_MASK, self.0)) }
    }

    /// Returns a vector with the absolute values of the original vector
    #[inline(always)]
    pub fn sqrt (self) -> Self {
        unsafe { Self(_mm_sqrt_ps(self.0)) }
    }

    /// Gets the smallest/minimum value of the vector
    #[inline(always)]
    pub fn min (self) -> f32 {
        unsafe {
            let ptr = addr_of!(self) as *const f32;
            (*ptr).min(*ptr.add(1))
        }
    }

    /// Gets the biggest/maximum value of the vector
    #[inline(always)]
    pub fn max (self) -> f32 {
        unsafe {
            let ptr = addr_of!(self) as *const f32;
            (*ptr).max(*ptr.add(1))
        }
    }

    /// Sums up all the values inside the vector
    #[inline(always)]
    pub fn sum (self) -> f32 {
        unsafe {
            let ptr = addr_of!(self) as *const f32;
            *ptr + *ptr.add(1)
        }
    }

    /// Multiplies all the values inside the vector
    #[inline(always)]
    pub fn prod (self) -> f32 {
        unsafe {
            let ptr = addr_of!(self) as *const f32;
            *ptr * *ptr.add(1)
        }
    }

    /// Returns a vector with the smallest/minimum value of each lane
    #[inline(always)]
    pub fn vmin (self, rhs: Self) -> Self {
        unsafe { Self(_mm_min_ps(self.0, rhs.0)) }
    }

    /// Returns a vector with the absolute values of the original vector
    #[inline(always)]
    pub fn vmax (self, rhs: Self) -> Self {
        unsafe { Self(_mm_max_ps(self.0, rhs.0)) }
    }

    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding error.
    #[inline(always)]
    pub fn mul_add (self, rhs: Self, add: Self) -> Self {
        Self(f32x4(self.0).mul_add(f32x4(rhs.0), f32x4(add.0)).0)
    }

    /// Interleaves elements of both vectors into one
    #[inline(always)]
    pub fn zip (self, rhs: Self) -> Self {
        unsafe { Self(_mm_set_ps(0., 0., *addr_of!(rhs).cast(), *addr_of!(self).cast())) }
    }
}

impl Add for f32x2 {
    type Output = Self;

    #[inline(always)]
    fn add (self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm_add_ps(self.0, rhs.0)) }
    }
}

impl Sub for f32x2 {
    type Output = Self;

    #[inline(always)]
    fn sub (self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm_sub_ps(self.0, rhs.0)) }
    }
}

impl Mul for f32x2 {
    type Output = Self;

    #[inline(always)]
    fn mul (self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm_mul_ps(self.0, rhs.0)) }
    }
}

impl Div for f32x2 {
    type Output = Self;

    #[inline(always)]
    fn div (self, rhs: Self) -> Self::Output {
        unsafe {
            let div = _mm_div_ps(self.0, rhs.0);
            Self(_mm_and_ps(Self::DIV_MASK, div))
        }
    }
}

impl Neg for f32x2 {
    type Output = Self;

    #[inline(always)]
    fn neg (self) -> Self::Output {
        unsafe { Self(_mm_sub_ps(_mm_setzero_ps(), self.0)) }
    }
}

impl PartialEq for f32x2 {
    #[inline(always)]
    fn eq (&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm_cmpeq_ps(self.0, other.0);
            *(&cmp as *const __m128 as *const u64) == u64::MAX
        }
    }
}

impl From<f32> for f32x2 {
    #[inline(always)]
    fn from(x: f32) -> Self {
        Self::new([x,x])
    }
}

impl_scal_arith!(
    f32x2, f32,
    Add, add,
    Sub, sub,
    Mul, mul,
    Div, div
);