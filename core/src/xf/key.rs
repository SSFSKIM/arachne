//! Content keys: BLAKE2b with a 16-byte digest over a shape's or paint table's canonical bytes.
//!
//! Canonical bytes are the kind code, the number of arrays, then per array its dtype code
//! (1 = f64, 2 = u32, 3 = u8), its number of dimensions, each dimension as a little-endian u64, and
//! its data, little-endian. Python's `hashlib.blake2b(canonical, digest_size=16)` gives the same key.
//! Keys find duplicates only; no drawer derives geometry from a key.

use super::{PAINT_TABLE_KIND, ShapeData, ShapeKind};

/// A 128-bit content key (a ShapeKey or a PaintKey).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Key(pub [u8; 16]);

impl Key {
    pub fn hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

const DTYPE_F64: u8 = 1;
const DTYPE_U32: u8 = 2;
const DTYPE_U8: u8 = 3;

/// Canonical bytes, built array by array.
struct Canonical(Vec<u8>);

impl Canonical {
    fn new(kind: u8, arrays: u8) -> Canonical {
        Canonical(vec![kind, arrays])
    }

    fn header(&mut self, dtype: u8, dims: &[usize]) {
        self.0.push(dtype);
        self.0.push(dims.len() as u8);
        for d in dims {
            self.0.extend_from_slice(&(*d as u64).to_le_bytes());
        }
    }

    fn f64s<const N: usize>(&mut self, rows: &[[f64; N]]) {
        self.header(DTYPE_F64, &[rows.len(), N]);
        for row in rows {
            for v in row {
                self.0.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    fn u32s<const N: usize>(&mut self, rows: &[[u32; N]]) {
        self.header(DTYPE_U32, &[rows.len(), N]);
        for row in rows {
            for v in row {
                self.0.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    fn u8s(&mut self, dims: &[usize], data: &[u8]) {
        self.header(DTYPE_U8, dims);
        self.0.extend_from_slice(data);
    }

    fn key(&self) -> Key {
        Key(blake2b_128(&self.0))
    }
}

/// The key of a shape.
pub fn shape_key(data: &ShapeData) -> Key {
    match data {
        ShapeData::Path { points } => {
            let mut c = Canonical::new(ShapeKind::Path as u8, 1);
            c.f64s(points);
            c.key()
        }
        ShapeData::Mesh {
            positions,
            triangles,
            normals,
        } => {
            let mut c = Canonical::new(ShapeKind::Mesh as u8, 2 + normals.is_some() as u8);
            c.f64s(positions);
            c.u32s(triangles);
            if let Some(n) = normals {
                c.f64s(n);
            }
            c.key()
        }
        ShapeData::Image {
            width,
            height,
            rgba,
        } => {
            let mut c = Canonical::new(ShapeKind::Image as u8, 1);
            c.u8s(&[*height as usize, *width as usize, 4], rgba);
            c.key()
        }
        ShapeData::Reserved { code, payload } => {
            let mut c = Canonical::new(*code, 1);
            c.u8s(&[payload.len()], payload);
            c.key()
        }
    }
}

/// The key of a paint table.
pub fn paint_key(rows: &[[f64; 4]]) -> Key {
    let mut c = Canonical::new(PAINT_TABLE_KIND, 1);
    c.f64s(rows);
    c.key()
}

/// BLAKE2b (RFC 7693) with a 16-byte digest, no key, salt or personalization.
pub fn blake2b_128(data: &[u8]) -> [u8; 16] {
    let digest = blake2b(data, 16);
    let mut out = [0u8; 16];
    out.copy_from_slice(&digest[..16]);
    out
}

const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

/// The compression function F over one 128-byte block.
fn compress(h: &mut [u64; 8], block: &[u8; 128], counter: u128, last: bool) {
    let mut m = [0u64; 16];
    for (i, word) in m.iter_mut().enumerate() {
        *word = u64::from_le_bytes(block[i * 8..i * 8 + 8].try_into().unwrap());
    }
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= counter as u64;
    v[13] ^= (counter >> 64) as u64;
    if last {
        v[14] = !v[14];
    }
    fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
        v[d] = (v[d] ^ v[a]).rotate_right(32);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(24);
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
        v[d] = (v[d] ^ v[a]).rotate_right(16);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(63);
    }
    for round in 0..12 {
        let s = &SIGMA[round % 10];
        g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// BLAKE2b (RFC 7693) of `data` with an `out_len`-byte digest (1..=64), unkeyed.
pub fn blake2b(data: &[u8], out_len: usize) -> Vec<u8> {
    assert!(
        (1..=64).contains(&out_len),
        "BLAKE2b digests are 1 to 64 bytes"
    );
    let mut h = IV;
    // Parameter block: digest length, no key, fanout 1, depth 1.
    h[0] ^= 0x0101_0000 ^ out_len as u64;
    let mut counter: u128 = 0;
    let mut chunks = data.chunks(128).peekable();
    if chunks.peek().is_none() {
        compress(&mut h, &[0u8; 128], 0, true);
    }
    while let Some(chunk) = chunks.next() {
        let mut block = [0u8; 128];
        block[..chunk.len()].copy_from_slice(chunk);
        counter += chunk.len() as u128;
        compress(&mut h, &block, counter, chunks.peek().is_none());
    }
    h.iter()
        .flat_map(|w| w.to_le_bytes())
        .take(out_len)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    // Expected values from Python's hashlib.blake2b.
    #[test]
    fn blake2b_matches_the_rfc_and_hashlib() {
        assert_eq!(
            hex(&blake2b(b"abc", 64)),
            "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1\
             7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923"
        );
        assert_eq!(hex(&blake2b_128(b"")), "cae66941d9efbd404e4d88758ea67670");
        assert_eq!(
            hex(&blake2b_128(b"abc")),
            "cf4ab791c62b8d2b2109c90275287816"
        );
        // Exactly one block, and a final partial block after full ones.
        assert_eq!(
            hex(&blake2b_128(&[0u8; 128])),
            "40d697295c01d2312035c1fa7d884b8d"
        );
        let data: Vec<u8> = (0..=255u8).chain(0..44u8).collect();
        assert_eq!(hex(&blake2b_128(&data)), "15f53a13900056879b859fc1bcea544d");
    }

    #[test]
    fn keys_hash_the_documented_canonical_bytes() {
        let path = ShapeData::Path {
            points: vec![[0.0, 1.5, -2.0], [3.25, 0.0, 1e-300]],
        };
        assert_eq!(shape_key(&path).hex(), "69386af91d5f42748e4fb36ee1ac459d");
        assert_eq!(
            paint_key(&[[1.0, 0.5, 0.25, 1.0]]).hex(),
            "0933b2b6afe2fda4d64218b75ca2a80a"
        );
    }

    #[test]
    fn a_path_and_a_paint_with_the_same_numbers_differ() {
        let row = [1.0, 0.5, 0.25, 1.0];
        let as_paint = paint_key(&[row]);
        let mut c = Canonical::new(ShapeKind::Path as u8, 1);
        c.f64s(&[row]);
        assert_ne!(as_paint, c.key());
    }
}
