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
    () => {
        
    };
}

macro_rules! impl_serde {
    () => {
        
    };
}

impl<'de> Deserialize for f32x2 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: serde::Deserializer<'de> {
        struct LocalVisitor;

        impl<'a> Visitor<'a> for LocalVisitor {
            type Value = f32x2;

            fn expecting(&self, formatter: &mut alloc::fmt::Formatter) -> alloc::fmt::Result {
                todo!()
            }
        }

        deserializer.deserialize_seq(visitor)
    }
}