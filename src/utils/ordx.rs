pub trait Ordx: PartialOrd {
    fn min (self, rhs: Self) -> Self;
    fn max (self, rhs: Self) -> Self;
}

impl<T> Ordx for T where T: PartialOrd {
    fn min (self, rhs: Self) -> Self {
        if self <= rhs { self } else { rhs }
    }

    fn max (self, rhs: Self) -> Self {
        if self >= rhs { self } else { rhs }
    }
}