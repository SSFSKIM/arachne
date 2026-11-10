//! Small readers the tests need and the oracle does not: SHA-256, JSON, and the hex text the XF 1.5
//! fixtures are committed as. Written from their standards (FIPS 180-4, RFC 8259), with no crate.
#![allow(dead_code)]

/// SHA-256 of `data`, as lowercase hex.
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_be_bytes());
    for block in msg.chunks(64) {
        let mut w = [0u32; 64];
        for (i, c) in block.chunks(4).enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (a, b) in h.iter_mut().zip(v) {
            *a = a.wrapping_add(b);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

/// The bytes of a fixture's hex text, under its rules: two lowercase hex digits per byte, 64
/// digits to a line, the last line holding the rest (an even number, 64 or fewer, none padded),
/// every line ending with one LF; nothing else. `Err` names the first rule broken.
pub fn decode_hex_text(text: &[u8]) -> Result<Vec<u8>, String> {
    if text.last() != Some(&b'\n') {
        return Err("the text does not end with a newline".into());
    }
    let lines: Vec<&[u8]> = text[..text.len() - 1].split(|&b| b == b'\n').collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let last = i + 1 == lines.len();
        if (!last && line.len() != 64) || (last && (line.is_empty() || line.len() > 64)) {
            return Err(format!("line {} holds {} digits", i + 1, line.len()));
        }
        if line.len() % 2 != 0 {
            return Err(format!("line {} holds an odd number of digits", i + 1));
        }
        for pair in line.chunks(2) {
            let digit = |c: u8| match c {
                b'0'..=b'9' => Ok(c - b'0'),
                b'a'..=b'f' => Ok(c - b'a' + 10),
                _ => Err(format!("line {} holds {:?}", i + 1, c as char)),
            };
            out.push(digit(pair[0])? * 16 + digit(pair[1])?);
        }
    }
    Ok(out)
}

/// A JSON value (RFC 8259); numbers kept as their text's `f64`.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(a) => a,
            _ => &[],
        }
    }
    pub fn str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn num(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }
    pub fn u64(&self) -> Option<u64> {
        self.num()
            .filter(|n| *n >= 0.0 && n.fract() == 0.0)
            .map(|n| n as u64)
    }
}

pub fn parse_json(text: &str) -> Result<Json, String> {
    let b = text.as_bytes();
    let mut at = 0;
    let v = value(b, &mut at)?;
    ws(b, &mut at);
    if at != b.len() {
        return Err(format!("trailing text at {at}"));
    }
    Ok(v)
}

fn ws(b: &[u8], at: &mut usize) {
    while *at < b.len() && matches!(b[*at], b' ' | b'\t' | b'\n' | b'\r') {
        *at += 1;
    }
}

fn value(b: &[u8], at: &mut usize) -> Result<Json, String> {
    ws(b, at);
    let c = *b.get(*at).ok_or("unexpected end")?;
    match c {
        b'{' => {
            *at += 1;
            let mut kv = Vec::new();
            ws(b, at);
            if b.get(*at) == Some(&b'}') {
                *at += 1;
                return Ok(Json::Obj(kv));
            }
            loop {
                ws(b, at);
                let Json::Str(k) = value(b, at)? else {
                    return Err(format!("a key that is not a string at {at}"));
                };
                ws(b, at);
                if b.get(*at) != Some(&b':') {
                    return Err(format!("no colon at {at}"));
                }
                *at += 1;
                kv.push((k, value(b, at)?));
                ws(b, at);
                match b.get(*at) {
                    Some(b',') => *at += 1,
                    Some(b'}') => {
                        *at += 1;
                        return Ok(Json::Obj(kv));
                    }
                    _ => return Err(format!("an object not closed at {at}")),
                }
            }
        }
        b'[' => {
            *at += 1;
            let mut a = Vec::new();
            ws(b, at);
            if b.get(*at) == Some(&b']') {
                *at += 1;
                return Ok(Json::Arr(a));
            }
            loop {
                a.push(value(b, at)?);
                ws(b, at);
                match b.get(*at) {
                    Some(b',') => *at += 1,
                    Some(b']') => {
                        *at += 1;
                        return Ok(Json::Arr(a));
                    }
                    _ => return Err(format!("an array not closed at {at}")),
                }
            }
        }
        b'"' => {
            *at += 1;
            let mut s: Vec<u8> = Vec::new();
            loop {
                let c = *b.get(*at).ok_or("a string not closed")?;
                *at += 1;
                match c {
                    b'"' => break,
                    b'\\' => {
                        let e = *b.get(*at).ok_or("an escape not closed")?;
                        *at += 1;
                        match e {
                            b'"' | b'\\' | b'/' => s.push(e),
                            b'b' => s.push(8),
                            b'f' => s.push(12),
                            b'n' => s.push(b'\n'),
                            b'r' => s.push(b'\r'),
                            b't' => s.push(b'\t'),
                            b'u' => {
                                let hex = |at: &mut usize| -> Result<u32, String> {
                                    let h = b.get(*at..*at + 4).ok_or("a short \\u")?;
                                    *at += 4;
                                    u32::from_str_radix(std::str::from_utf8(h).unwrap(), 16)
                                        .map_err(|e| e.to_string())
                                };
                                let mut cp = hex(at)?;
                                if (0xd800..0xdc00).contains(&cp) {
                                    if b.get(*at..*at + 2) != Some(b"\\u") {
                                        return Err("a lone high surrogate".into());
                                    }
                                    *at += 2;
                                    let lo = hex(at)?;
                                    cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                                }
                                let ch = char::from_u32(cp).ok_or("a bad code point")?;
                                let mut buf = [0u8; 4];
                                s.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                            }
                            _ => return Err(format!("a bad escape at {at}")),
                        }
                    }
                    _ => s.push(c),
                }
            }
            String::from_utf8(s)
                .map(Json::Str)
                .map_err(|e| e.to_string())
        }
        b't' if b[*at..].starts_with(b"true") => {
            *at += 4;
            Ok(Json::Bool(true))
        }
        b'f' if b[*at..].starts_with(b"false") => {
            *at += 5;
            Ok(Json::Bool(false))
        }
        b'n' if b[*at..].starts_with(b"null") => {
            *at += 4;
            Ok(Json::Null)
        }
        _ => {
            let start = *at;
            while *at < b.len() && matches!(b[*at], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
            {
                *at += 1;
            }
            std::str::from_utf8(&b[start..*at])
                .unwrap()
                .parse::<f64>()
                .map(Json::Num)
                .map_err(|e| format!("a bad number at {start}: {e}"))
        }
    }
}

#[test]
fn sha256_of_known_strings() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let long = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    assert_eq!(
        sha256_hex(long),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}
