use serde::Deserialize;
use serde::Serialize;
use serde::de::Visitor;
use serde::ser::SerializeSeq;
use crate::float::single::*;
use crate::float::double::*;

macro_rules! impl_ser {
    ($([$ty:ident;$len:literal] as $target:ident),+) => {
        $(
            impl Serialize for $target {
                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: serde::Serializer {
                    let mut seq = serializer.serialize_seq(Some($len))?;
                    for i in 0..$len {
                        unsafe { seq.serialize_element(self.index_unchecked(i))?; }
                    }
                    seq.end()
                }
            }
        )*
    };
}

macro_rules! impl_de {
    ($([$ty:ident;$len:literal] as $target:ident),+) => {
        $(
            impl<'de> Deserialize<'de> for $target {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: serde::Deserializer<'de> {
                    let array = <[$ty;$len] as Deserialize<'de>>::deserialize(deserializer)?;
                    Ok(Self::from(array))
                }
            }
        )*
    };
}

macro_rules! impl_serde {
    ($([$ty:ident;$len:literal] as $target:ident),+) => {
        $(
            impl_ser!([$ty;$len] as $target);
            impl_de!([$ty;$len] as $target);
        )*
    };
}

impl_serde!(
    [f32;2] as f32x2,
    [f32;4] as f32x4,
    [f32;6] as f32x6,
    [f32;8] as f32x8,
    [f32;10] as f32x10,
    [f32;12] as f32x12,
    [f32;14] as f32x14,
    [f32;16] as f32x16,

    [f64;2] as f64x2,
    [f64;4] as f64x4,
    [f64;6] as f64x6,
    [f64;8] as f64x8,
    [f64;10] as f64x10,
    [f64;12] as f64x12,
    [f64;14] as f64x14,
    [f64;16] as f64x16
);