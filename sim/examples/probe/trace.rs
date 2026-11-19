//! The bytes a probe world records: every number as its bits, little-endian.

#[derive(Default, PartialEq, Eq, Debug)]
pub struct Trace(pub Vec<u8>);

impl Trace {
    pub fn f32(&mut self, x: f32) {
        self.0.extend_from_slice(&x.to_bits().to_le_bytes());
    }

    pub fn f32s(&mut self, xs: &[f32]) {
        for &x in xs {
            self.f32(x);
        }
    }

    pub fn u32(&mut self, x: u32) {
        self.0.extend_from_slice(&x.to_le_bytes());
    }
}

/// Sorted pairs of body positions in contact, from (collider, collider) pairs.
pub fn contact_rows(mut pairs: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    for pair in &mut pairs {
        if pair.0 > pair.1 {
            *pair = (pair.1, pair.0);
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}
