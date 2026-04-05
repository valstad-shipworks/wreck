use alloc::vec::Vec;
use serde::de::{Deserialize, Deserializer, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeSeq, Serializer};

pub(crate) mod array_as_seq {
    use super::*;

    pub fn serialize<S, T, const N: usize>(arr: &[T; N], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize,
    {
        let mut seq = serializer.serialize_seq(Some(N))?;
        for item in arr {
            seq.serialize_element(item)?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D, T, const N: usize>(deserializer: D) -> Result<[T; N], D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        struct ArrayVisitor<T, const N: usize>(core::marker::PhantomData<T>);

        impl<'de, T, const N: usize> Visitor<'de> for ArrayVisitor<T, N>
        where
            T: Deserialize<'de>,
        {
            type Value = [T; N];

            fn expecting(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "an array of length {N}")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut vec = Vec::with_capacity(N);
                while let Some(item) = seq.next_element()? {
                    vec.push(item);
                }
                vec.try_into().map_err(|v: Vec<T>| {
                    serde::de::Error::invalid_length(v.len(), &self)
                })
            }
        }

        deserializer.deserialize_seq(ArrayVisitor::<T, N>(core::marker::PhantomData))
    }
}
