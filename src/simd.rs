use std::{ops::{Add, Sub, Mul, Div, Index, IndexMut}};

pub trait Simdt<T>: Sized where
    Self: 
        Add<Self, Output = Self> + Add<T, Output = Self> +
        Sub<Self, Output = Self> + Sub<T, Output = Self> +
        Mul<Self, Output = Self> + Mul<T, Output = Self> +
        Div<Self, Output = Self> + Div<T, Output = Self> + 
        Index<usize, Output = T> + IndexMut<usize>
{}