use std::ops::*;
use llml_simd_proc::{Assign, assign_targets, assign_rhs};

macro_rules! impl_arith {
    ($($trait:ident, $fun:ident),+) => {
        $(
            impl $trait for TestStruct {
                type Output = Self;
            
                fn $fun(self, rhs: Self) -> Self::Output {
                    Self {
                        first: self.first.$fun(rhs.first),
                        last: self.last.$fun(rhs.last),
                    }
                }
            }
        )*
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Assign)]
#[assign_targets(Add, Sub, Mul, Div)]
#[assign_rhs(Self)]
struct TestStruct {
    first: i16,
    last: f64
}
impl_arith!(
    Add, add,
    Sub, sub,
    Mul, mul,
    Div, div
);

#[test]
fn add () {
    let mut test = TestStruct { first: 1, last: 2. };
    test += TestStruct { first: 2, last: 1. };
    assert_eq!(test, TestStruct { first: 3, last: 3. })
}

#[test]
fn sub () {
    let mut test = TestStruct { first: 1, last: 2. };
    test -= TestStruct { first: 2, last: 1. };
    assert_eq!(test, TestStruct { first: -1, last: 1. })
}

#[test]
fn mul () {
    let mut test = TestStruct { first: 1, last: 2. };
    test *= TestStruct { first: 2, last: 1. };
    assert_eq!(test, TestStruct { first: 2, last: 2. })
}

#[test]
fn div () {
    let mut test = TestStruct { first: 1, last: 2. };
    test /= TestStruct { first: 2, last: 1. };
    assert_eq!(test, TestStruct { first: 0, last: 2. })
}