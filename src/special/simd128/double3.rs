#[cfg(not(feature = "use_std"))]
use crate::special::NoStdMath;
use core::ops::*;
use std::ptr::addr_of;
use llml_simd_proc::*;
use crate::float::double::f64x2;

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Copy, Assign)]
#[assign_targets(Add, Sub, Mul, Div)]
#[assign_rhs(Self, f64)]
pub struct f64x3 (pub(crate) f64x2, pub(crate) f64);

impl f64x3 {
    /// Loads values from the pointer into the SIMD vector
    #[inline(always)]
    pub unsafe fn load (ptr: *const f64) -> Self {
        Self(f64x2::load(ptr), *ptr.add(2))
    }

    /// Returns a vector with the absolute values of the original vector
    #[inline(always)]
    pub fn abs (self) -> Self {
        Self(self.0.abs(), self.1.abs())
    }

    /// Returns a vector with the absolute values of the original vector
    #[inline(always)]
    pub fn sqrt (self) -> Self {
        Self(self.0.sqrt(), self.1.sqrt())
    }

    /// Gets the smallest/minimum value of the vector
    #[inline(always)]
    pub fn min (self) -> f64 {
        self.0.min().min(self.1)
    }

    /// Gets the biggest/maximum value of the vector
    #[inline(always)]
    pub fn max (self) -> f64 {
        self.0.max().max(self.1)
    }

    /// Sums up all the values inside the vector
    #[inline(always)]
    pub fn sum (self) -> f64 {
        self.0.sum() + self.1
    }

    /// Returns a vector with the smallest/minimum value of each lane
    #[inline(always)]
    pub fn vmin (self, rhs: Self) -> Self {
        Self(self.0.vmin(rhs.0), self.1.min(rhs.1))
    }

    /// Returns a vector with the absolute values of the original vector
    #[inline(always)]
    pub fn vmax (self, rhs: Self) -> Self {
        Self(self.0.vmax(rhs.0), self.1.max(rhs.1))
    }

    /// Interleaves elements of both vectors into one
    #[inline(always)]
    pub fn zip (self, rhs: Self) -> Self {
        unsafe { 
            let alpha = addr_of!(self) as *const f64;
            let beta = addr_of!(rhs) as *const f64;
            Self::new([*alpha, *beta, *alpha.add(1)])
        }
    }
}

impl Add for f64x3 {
    type Output = Self;

    #[inline(always)]
    fn add (self, rhs: Self) -> Self::Output {
        Self(self.0.add(rhs.0), self.1.add(rhs.1))
    }
}

impl Sub for f64x3 {
    type Output = Self;

    #[inline(always)]
    fn sub (self, rhs: Self) -> Self::Output {
        Self(self.0.sub(rhs.0), self.1.sub(rhs.1))
    }
}

impl Mul for f64x3 {
    type Output = Self;

    #[inline(always)]
    fn mul (self, rhs: Self) -> Self::Output {
        Self(self.0.mul(rhs.0), self.1.mul(rhs.1))
    }
}

impl Div for f64x3 {
    type Output = Self;

    #[inline(always)]
    fn div (self, rhs: Self) -> Self::Output {
        Self(self.0.div(rhs.0), self.1.div(rhs.1))
    }
}

impl Neg for f64x3 {
    type Output = Self;

    #[inline(always)]
    fn neg (self) -> Self::Output {
        Self(self.0.neg(), self.1.neg())
    }
}

impl PartialEq for f64x3 {
    #[inline(always)]
    fn eq (&self, other: &Self) -> bool {
        self.0.eq(&other.0) && self.1.eq(&other.1)
    }
}

impl From<f64> for f64x3 {
    #[inline(always)]
    fn from(x: f64) -> Self {
        Self(f64x2::from(x), x)
    }
}

impl_scal_arith!(
    f64x3, f64,
    Add, add,
    Sub, sub,
    Mul, mul,
    Div, div
);