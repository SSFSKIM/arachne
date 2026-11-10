//! The oracle's decisions for one frame and their file, `ORACLE_FORMAT` 1 (`FORMAT.md`).
//!
//! Little-endian fixed-size records in sections a directory locates, so each section reads with
//! `numpy.fromfile` at its offset. The file carries decisions (and, once depth lands, depths and
//! bands), never bytes. `read.py` beside the crate is the one reader for other programs;
//! [`read`] here exists so the crate's tests prove the two agree.

use std::fmt::Write as _;

pub const ORACLE_FORMAT: u16 = 1;
pub const MAGIC: &[u8; 4] = b"AROR";
/// The fixed header's size, before the section directory.
pub const HEADER_SIZE: usize = 88;
pub const DIRECTORY_ENTRY: usize = 24;
pub const FRAGMENT_SIZE: usize = 64;
pub const COVER_SIZE: usize = 456;
pub const PIXEL_SIZE: usize = 12;
pub const REFUSAL_SIZE: usize = 32;
/// Written where a field holds nothing.
pub const NONE_U32: u32 = u32::MAX;

/// A sample's coverage verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Cover {
    Absent = 0,
    Yes = 1,
    Either = 2,
}

/// A fragment flag: its screen shift is infinite (contract §6, "Unbounded keys"), so it is
/// `truth_unable` at every sample of the window. Such a fragment carries no cover rows.
pub const FLAG_UNBOUNDED: u16 = 1;

/// One fragment the frame draws: one layer of one path, one mesh triangle, or one point block's
/// pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct Fragment {
    pub id: u32,
    /// Its order entry's index.
    pub entry: u32,
    pub slot: u32,
    pub object: u32,
    pub part: u32,
    /// Its index within its layer, as the core's report numbers it (`sub`).
    pub sub: u32,
    pub layer: u8,
    /// Its record's kind code.
    pub kind: u8,
    pub flags: u16,
    /// The drawn path its layers share (`own_layers@1`'s group), else [`NONE_U32`].
    pub group: u32,
    /// Its registered plane identity, else [`NONE_U32`] (written with depth).
    pub plane: u32,
    /// Its screen shift `ε`, in pixels.
    pub epsilon: f64,
    /// The pixel box its rows lie in: `[x0, y0, x1, y1)`.
    pub bbox: [i32; 4],
}

/// One fragment at one pixel: its verdict and areas at every sample.
#[derive(Clone, Debug, PartialEq)]
pub struct CoverRow {
    pub x: u32,
    pub y: u32,
    pub fragment: u32,
    pub cover: [u8; 16],
    /// The admitted cell bytes at each sample, `[lo, hi]`.
    pub cell_lo: [u8; 16],
    pub cell_hi: [u8; 16],
    /// The admitted whole-pixel coverage bytes.
    pub pixel_lo: u8,
    pub pixel_hi: u8,
    /// A path layer's exact pixel area (NaN for a mask).
    pub pixel_area: f64,
    /// A path layer's exact area of each sample's cell (NaN for a mask).
    pub cell_area: [f64; 16],
    /// The fragment's key at each sample (NaN until depth is written).
    pub depth: [f64; 16],
    /// Its band at each sample (NaN until depth is written).
    pub band: [f64; 16],
}

/// A refusal as the file holds it.
#[derive(Clone, Debug, PartialEq)]
pub struct Refused {
    pub reason: u8,
    /// The record kind code, 0xff where none.
    pub kind: u8,
    pub entry: u32,
    pub slot: u32,
    pub object: u32,
    pub bits: u64,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub xf: (u16, u16),
    pub width: u32,
    pub height: u32,
    /// The pixels decided: `[x0, y0, x1, y1)`.
    pub window: [u32; 4],
    /// 0: decided; 1: `truth_unable` for the whole frame.
    pub status: u8,
    pub seq: u64,
    /// The 16 sample offsets in sixteenths of a pixel from its centre.
    pub offsets: [[i8; 2]; 16],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Decisions {
    pub header: Header,
    pub fragments: Vec<Fragment>,
    /// Sorted by `(y, x, fragment)`.
    pub covers: Vec<CoverRow>,
    pub refused: Vec<Refused>,
}

pub const STATUS_DECIDED: u8 = 0;
pub const STATUS_UNABLE: u8 = 1;

struct Section {
    tag: [u8; 4],
    record_size: u32,
    count: u64,
    bytes: Vec<u8>,
}

fn u32s(b: &mut Vec<u8>, vs: &[u32]) {
    for v in vs {
        b.extend(v.to_le_bytes());
    }
}

fn f64s(b: &mut Vec<u8>, vs: &[f64]) {
    for v in vs {
        b.extend(v.to_le_bytes());
    }
}

/// The pixel index: per pixel of the window, row by row, its first cover row and their count.
fn pixel_index(d: &Decisions) -> Vec<u8> {
    let [x0, y0, x1, y1] = d.header.window;
    let w = (x1 - x0) as usize;
    let mut first = vec![0u64; w * (y1 - y0) as usize];
    let mut count = vec![0u32; first.len()];
    for (i, c) in d.covers.iter().enumerate().rev() {
        let k = (c.y - y0) as usize * w + (c.x - x0) as usize;
        first[k] = i as u64;
        count[k] += 1;
    }
    let mut b = Vec::with_capacity(first.len() * PIXEL_SIZE);
    for (f, c) in first.iter().zip(&count) {
        b.extend(if *c == 0 { 0 } else { *f }.to_le_bytes());
        b.extend(c.to_le_bytes());
    }
    b
}

/// The file's bytes.
pub fn write(d: &Decisions) -> Vec<u8> {
    let mut frag = Vec::with_capacity(d.fragments.len() * FRAGMENT_SIZE);
    for f in &d.fragments {
        u32s(&mut frag, &[f.id, f.entry, f.slot, f.object, f.part, f.sub]);
        frag.extend([f.layer, f.kind]);
        frag.extend(f.flags.to_le_bytes());
        u32s(&mut frag, &[f.group]);
        f64s(&mut frag, &[f.epsilon]);
        for v in f.bbox {
            frag.extend(v.to_le_bytes());
        }
        u32s(&mut frag, &[f.plane, 0]);
    }
    let mut cov = Vec::with_capacity(d.covers.len() * COVER_SIZE);
    for c in &d.covers {
        u32s(&mut cov, &[c.x, c.y, c.fragment]);
        cov.extend(c.cover);
        cov.extend(c.cell_lo);
        cov.extend(c.cell_hi);
        cov.extend([c.pixel_lo, c.pixel_hi, 0, 0]);
        f64s(&mut cov, &[c.pixel_area]);
        f64s(&mut cov, &c.cell_area);
        f64s(&mut cov, &c.depth);
        f64s(&mut cov, &c.band);
    }
    let mut text = Vec::new();
    let mut refu = Vec::with_capacity(d.refused.len() * REFUSAL_SIZE);
    for r in &d.refused {
        refu.extend([r.reason, r.kind, 0, 0]);
        u32s(&mut refu, &[r.entry, r.slot, r.object]);
        refu.extend(r.bits.to_le_bytes());
        u32s(&mut refu, &[text.len() as u32, r.detail.len() as u32]);
        text.extend(r.detail.as_bytes());
    }
    let pixl = pixel_index(d);
    let sections = [
        Section {
            tag: *b"FRAG",
            record_size: FRAGMENT_SIZE as u32,
            count: d.fragments.len() as u64,
            bytes: frag,
        },
        Section {
            tag: *b"COVR",
            record_size: COVER_SIZE as u32,
            count: d.covers.len() as u64,
            bytes: cov,
        },
        Section {
            tag: *b"PIXL",
            record_size: PIXEL_SIZE as u32,
            count: (pixl.len() / PIXEL_SIZE) as u64,
            bytes: pixl,
        },
        Section {
            tag: *b"REFU",
            record_size: REFUSAL_SIZE as u32,
            count: d.refused.len() as u64,
            bytes: refu,
        },
        Section {
            tag: *b"TEXT",
            record_size: 1,
            count: text.len() as u64,
            bytes: text,
        },
    ];
    let h = &d.header;
    let mut b = Vec::new();
    b.extend(MAGIC);
    b.extend(ORACLE_FORMAT.to_le_bytes());
    b.extend((HEADER_SIZE as u16).to_le_bytes());
    b.extend(h.xf.0.to_le_bytes());
    b.extend(h.xf.1.to_le_bytes());
    u32s(&mut b, &[h.width, h.height]);
    u32s(&mut b, &h.window);
    b.extend([h.status, 0, 0, 0]);
    b.extend(h.seq.to_le_bytes());
    for o in h.offsets {
        b.extend([o[0] as u8, o[1] as u8]);
    }
    u32s(&mut b, &[sections.len() as u32, 0]);
    debug_assert_eq!(b.len(), HEADER_SIZE);
    // Each section starts on an 8-byte boundary after the directory.
    let mut at = HEADER_SIZE + sections.len() * DIRECTORY_ENTRY;
    let mut offsets = Vec::new();
    for s in &sections {
        at = at.div_ceil(8) * 8;
        offsets.push(at);
        at += s.bytes.len();
    }
    for (s, off) in sections.iter().zip(&offsets) {
        b.extend(s.tag);
        b.extend(s.record_size.to_le_bytes());
        b.extend((*off as u64).to_le_bytes());
        b.extend(s.count.to_le_bytes());
    }
    for (s, off) in sections.iter().zip(&offsets) {
        b.resize(*off, 0);
        b.extend(&s.bytes);
    }
    b
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let s = self
            .b
            .get(self.at..self.at + N)
            .ok_or("the file ends inside a record")?;
        self.at += N;
        Ok(s.try_into().unwrap())
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take::<1>()?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take()?))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take()?))
    }
    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.take()?))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take()?))
    }
    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.take()?))
    }
    fn f64s(&mut self) -> Result<[f64; 16], String> {
        let mut out = [0.0; 16];
        for v in &mut out {
            *v = self.f64()?;
        }
        Ok(out)
    }
}

/// The file read back (sections it does not know are skipped).
pub fn read(b: &[u8]) -> Result<Decisions, String> {
    let mut r = Reader { b, at: 0 };
    if &r.take::<4>()? != MAGIC {
        return Err("not an oracle decisions file".into());
    }
    let format = r.u16()?;
    if format != ORACLE_FORMAT {
        return Err(format!(
            "ORACLE_FORMAT {format} is not one this reader knows"
        ));
    }
    let header_size = r.u16()? as usize;
    let xf = (r.u16()?, r.u16()?);
    let (width, height) = (r.u32()?, r.u32()?);
    let window = [r.u32()?, r.u32()?, r.u32()?, r.u32()?];
    let status = r.u8()?;
    r.at += 3;
    let seq = r.u64()?;
    let mut offsets = [[0i8; 2]; 16];
    for o in &mut offsets {
        *o = [r.u8()? as i8, r.u8()? as i8];
    }
    let n = r.u32()? as usize;
    r.at = header_size;
    let mut dir = Vec::new();
    for _ in 0..n {
        let tag = r.take::<4>()?;
        let size = r.u32()? as usize;
        let off = r.u64()? as usize;
        let count = r.u64()? as usize;
        dir.push((tag, size, off, count));
    }
    let section = |tag: &[u8; 4]| dir.iter().find(|d| &d.0 == tag).copied();
    let mut d = Decisions {
        header: Header {
            xf,
            width,
            height,
            window,
            status,
            seq,
            offsets,
        },
        fragments: Vec::new(),
        covers: Vec::new(),
        refused: Vec::new(),
    };
    if let Some((_, size, off, count)) = section(b"FRAG") {
        for i in 0..count {
            let mut r = Reader {
                b,
                at: off + i * size,
            };
            let f = Fragment {
                id: r.u32()?,
                entry: r.u32()?,
                slot: r.u32()?,
                object: r.u32()?,
                part: r.u32()?,
                sub: r.u32()?,
                layer: r.u8()?,
                kind: r.u8()?,
                flags: r.u16()?,
                group: r.u32()?,
                epsilon: r.f64()?,
                bbox: [r.i32()?, r.i32()?, r.i32()?, r.i32()?],
                plane: r.u32()?,
            };
            d.fragments.push(f);
        }
    }
    if let Some((_, size, off, count)) = section(b"COVR") {
        for i in 0..count {
            let mut r = Reader {
                b,
                at: off + i * size,
            };
            let (x, y, fragment) = (r.u32()?, r.u32()?, r.u32()?);
            let (cover, cell_lo, cell_hi) = (r.take::<16>()?, r.take::<16>()?, r.take::<16>()?);
            let (pixel_lo, pixel_hi) = (r.u8()?, r.u8()?);
            r.at += 2;
            d.covers.push(CoverRow {
                x,
                y,
                fragment,
                cover,
                cell_lo,
                cell_hi,
                pixel_lo,
                pixel_hi,
                pixel_area: r.f64()?,
                cell_area: r.f64s()?,
                depth: r.f64s()?,
                band: r.f64s()?,
            });
        }
    }
    let text = section(b"TEXT").map_or(&[][..], |(_, _, off, count)| &b[off..off + count]);
    if let Some((_, size, off, count)) = section(b"REFU") {
        for i in 0..count {
            let mut r = Reader {
                b,
                at: off + i * size,
            };
            let (reason, kind) = (r.u8()?, r.u8()?);
            r.at += 2;
            let (entry, slot, object, bits) = (r.u32()?, r.u32()?, r.u32()?, r.u64()?);
            let (start, len) = (r.u32()? as usize, r.u32()? as usize);
            let detail = text
                .get(start..start + len)
                .ok_or("a refusal's text lies outside TEXT")?;
            d.refused.push(Refused {
                reason,
                kind,
                entry,
                slot,
                object,
                bits,
                detail: String::from_utf8_lossy(detail).into_owned(),
            });
        }
    }
    Ok(d)
}

fn hex(v: f64) -> String {
    format!("{:016x}", v.to_bits())
}

fn list<T: std::fmt::Display>(vs: impl IntoIterator<Item = T>) -> String {
    vs.into_iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// The decisions as canonical text, one line per item, floats as their bits in hex: what
/// `read.py --canonical` prints for the same file.
pub fn canonical(d: &Decisions) -> String {
    let h = &d.header;
    let mut s = String::new();
    let status = if h.status == STATUS_DECIDED {
        "decided"
    } else {
        "truth_unable"
    };
    let _ = writeln!(
        s,
        "header format={ORACLE_FORMAT} xf={}.{} size={}x{} window={} status={status} seq={}",
        h.xf.0,
        h.xf.1,
        h.width,
        h.height,
        list(h.window),
        h.seq
    );
    let _ = writeln!(
        s,
        "offsets {}",
        h.offsets
            .iter()
            .map(|o| format!("{},{}", o[0], o[1]))
            .collect::<Vec<_>>()
            .join(" ")
    );
    for f in &d.fragments {
        let _ = writeln!(
            s,
            "fragment id={} entry={} slot={} object={} part={} sub={} layer={} kind={} flags={} \
             group={} plane={} epsilon={} bbox={}",
            f.id,
            f.entry,
            f.slot,
            f.object,
            f.part,
            f.sub,
            f.layer,
            f.kind,
            f.flags,
            f.group,
            f.plane,
            hex(f.epsilon),
            list(f.bbox)
        );
    }
    for c in &d.covers {
        let _ = writeln!(
            s,
            "cover x={} y={} fragment={} cover={} cell={} pixel={}-{} area={} cells={} depth={} \
             band={}",
            c.x,
            c.y,
            c.fragment,
            list(c.cover),
            (0..16)
                .map(|k| format!("{}-{}", c.cell_lo[k], c.cell_hi[k]))
                .collect::<Vec<_>>()
                .join(","),
            c.pixel_lo,
            c.pixel_hi,
            hex(c.pixel_area),
            list(c.cell_area.iter().map(|v| hex(*v))),
            list(c.depth.iter().map(|v| hex(*v))),
            list(c.band.iter().map(|v| hex(*v)))
        );
    }
    for r in &d.refused {
        let _ = writeln!(
            s,
            "refused reason={} kind={} entry={} slot={} object={} bits={} detail={}",
            r.reason, r.kind, r.entry, r.slot, r.object, r.bits, r.detail
        );
    }
    s
}
