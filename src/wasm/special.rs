use core::ops::*;
use core::mem::transmute;
use core::ptr::addr_of;
use llml_simd_proc::*;
use core::arch::wasm::*;

#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Copy, Assign)]
#[assign_targets(Add, Sub, Mul, Div)]
#[assign_rhs(Self, f32)]
pub struct f32x2 (pub(crate) v128);

impl f32x2 {
    const DIV_MASK : v128 = unsafe { transmute([u32::MAX, u32::MAX, 0, 0]) };

    /// Loads values from the pointer into the SIMD vector
    #[inline(always)]
    pub unsafe fn load (ptr: *const f32) -> Self {
        Self(core::arch::wasm::f32x4(*ptr, *ptr.add(1), 0., 0.))
    }

    /// Returns a vector with the absolute values of the original vector
    #[inline(always)]
    pub fn abs (self) -> Self {
        unsafe { Self(f32x4_abs(self.0)) }
    }

    /// Returns a vector with the absolute values of the original vector
    #[inline(always)]
    pub fn sqrt (self) -> Self {
        unsafe { Self(v128_and(Self::DIV_MASK, f32x4_sqrt(self.0))) }
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

    /// Multiplies the vector by a scalar
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
        unsafe { Self(f32x4_pmin(self.0, rhs.0)) }
    }

    /// Returns a vector with the absolute values of the original vector
    #[inline(always)]
    pub fn vmax (self, rhs: Self) -> Self {
        unsafe { Self(f32x4_pmax(self.0, rhs.0)) }
    }

    /// Interleaves elements of both vectors into one
    #[inline(always)]
    pub fn zip (self, rhs: Self) -> Self {
        unsafe { Self(u32x4_shuffle::<0, 4, 2, 2>(self.0, rhs.0)) }
    }
}

impl Add for f32x2 {
    type Output = Self;

    #[inline(always)]
    fn add (self, rhs: Self) -> Self::Output {
        unsafe { Self(f32x4_add(self.0, rhs.0)) }
    }
}

impl Sub for f32x2 {
    type Output = Self;

    #[inline(always)]
    fn sub (self, rhs: Self) -> Self::Output {
        unsafe { Self(f32x4_sub(self.0, rhs.0)) }
    }
}

impl Mul for f32x2 {
    type Output = Self;

    #[inline(always)]
    fn mul (self, rhs: Self) -> Self::Output {
        unsafe { Self(f32x4_mul(self.0, rhs.0)) }
    }
}

impl Div for f32x2 {
    type Output = Self;

    #[inline(always)]
    fn div (self, rhs: Self) -> Self::Output {
        unsafe {
            let div = f32x4_div(self.0, rhs.0);
            Self(v128_and(Self::DIV_MASK, div))
        }
    }
}

impl Neg for f32x2 {
    type Output = Self;

    #[inline(always)]
    fn neg (self) -> Self::Output {
        unsafe { Self(f32x4_neg(self.0)) }
    }
}

impl PartialEq for f32x2 {
    #[inline(always)]
    fn eq (&self, other: &Self) -> bool {
        unsafe {
            let cmp = f32x4_eq(self.0, other.0);
            *(&cmp as *const v128 as *const u64) == u64::MAX
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