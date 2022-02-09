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
pub struct f32x6 (pub(crate) __m256);

impl f32x6 {
    const DIV_MASK : __m256 = unsafe { transmute([u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, 0, 0]) };
    const ABS_MASK : __m256 = unsafe { transmute([i32::MAX, i32::MAX, i32::MAX, i32::MAX, i32::MAX, i32::MAX, 0, 0]) };
    
    const MIN_MASK : __m128 = unsafe { transmute([0., 0., f32::MAX, f32::MAX]) };
    const MAX_MASK : __m128 = unsafe { transmute([0., 0., f32::MIN, f32::MIN]) };

    /// Loads values from the pointer into the SIMD vector
    #[inline(always)]
    pub unsafe fn load (ptr: *const f32) -> Self {
        Self(_mm256_set_ps(0., 0., *ptr.add(5), *ptr.add(4), *ptr.add(3), *ptr.add(2), *ptr.add(1), *ptr))
    }

    /// Returns a vector with the absolute values of the original vector
    #[inline(always)]
    pub fn abs (self) -> Self {
        unsafe { Self(_mm256_and_ps(Self::ABS_MASK, self.0)) }
    }

    /// Returns a vector with the absolute values of the original vector
    #[inline(always)]
    pub fn sqrt (self) -> Self {
        unsafe { Self(_mm256_and_ps(Self::DIV_MASK, _mm256_sqrt_ps(self.0))) }
    }

    /// Gets the smallest/minimum value of the vector
    #[inline(always)]
    pub fn min (self) -> f32 {
        unsafe {
            let vlow  = _mm256_castps256_ps128(self.0);
            let vhigh = _mm_or_ps(Self::MIN_MASK, _mm256_extractf128_ps(self.0, 1)); // high 128
            let v = f32x4(_mm_min_ps(vlow, vhigh));
            return v.min();
        }
    }

    /// Gets the biggest/maximum value of the vector
    #[inline(always)]
    pub fn max (self) -> f32 {
        unsafe {
            let vlow  = _mm256_castps256_ps128(self.0);
            let vhigh = _mm_or_ps(Self::MAX_MASK, _mm256_extractf128_ps(self.0, 1)); // high 128
            let v = f32x4(_mm_max_ps(vlow, vhigh));
            return v.max();
        }
    }

    /// Sums up all the values inside the vector
    #[inline(always)]
    pub fn sum (self) -> f32 {
        unsafe {
            let vlow  = _mm256_castps256_ps128(self.0);
            let vhigh = _mm256_extractf128_ps(self.0, 1); // high 128
            let v = f32x4(_mm_add_ps(vlow, vhigh));
            return v.sum();
        }
    }

    /// Returns a vector with the smallest/minimum value of each lane
    #[inline(always)]
    pub fn vmin (self, rhs: Self) -> Self {
        unsafe { Self(_mm256_min_ps(self.0, rhs.0)) }
    }

    /// Returns a vector with the absolute values of the original vector
    #[inline(always)]
    pub fn vmax (self, rhs: Self) -> Self {
        unsafe { Self(_mm256_max_ps(self.0, rhs.0)) }
    }
}

impl Add for f32x6 {
    type Output = Self;

    #[inline(always)]
    fn add (self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm256_add_ps(self.0, rhs.0)) }
    }
}

impl Sub for f32x6 {
    type Output = Self;

    #[inline(always)]
    fn sub (self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm256_sub_ps(self.0, rhs.0)) }
    }
}

impl Mul for f32x6 {
    type Output = Self;

    #[inline(always)]
    fn mul (self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm256_mul_ps(self.0, rhs.0)) }
    }
}

impl Div for f32x6 {
    type Output = Self;

    #[inline(always)]
    fn div (self, rhs: Self) -> Self::Output {
        unsafe {
            let div = _mm256_div_ps(self.0, rhs.0);
            Self(_mm256_and_ps(Self::DIV_MASK, div))
        }
    }
}

impl Neg for f32x6 {
    type Output = Self;

    #[inline(always)]
    fn neg (self) -> Self::Output {
        unsafe { Self(_mm256_sub_ps(_mm256_setzero_ps(), self.0)) }
    }
}

impl PartialEq for f32x6 {
    #[inline(always)]
    fn eq (&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmp_ps(self.0, other.0, _MM_CMPINT_EQ);
            let cmp : (u128, u64) = *addr_of!(cmp).cast();
            cmp.0 == u128::MAX && cmp.1 == u64::MAX
        }
    }
}

impl From<f32> for f32x6 {
    #[inline(always)]
    fn from(x: f32) -> Self {
        Self::new([x;6])
    }
}

impl_clone!(f32x6, f32, 6);
impl_scal_arith!(
    f32x6, f32,
    Add, add,
    Sub, sub,
    Mul, mul,
    Div, div
);