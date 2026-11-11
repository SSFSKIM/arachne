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
pub const REFUSAL_SIZE: usize = 40;
pub const BAND_SIZE: usize = 544;
pub const PAIR_SIZE: usize = 24;
pub const COLOUR_SIZE: usize = 12;
pub const PIXEL_COLOUR_SIZE: usize = 24;
pub const LABEL_SIZE: usize = 20;
/// A fragment's colour kind: not computed (a gradient), one solid, or per pixel (`COLP`).
pub const COLOUR_GRADIENT: u8 = 0;
pub const COLOUR_SOLID: u8 = 1;
pub const COLOUR_PER_PIXEL: u8 = 2;
/// A pair's label where a read is not bounded: `truth_unable`, every edge.
pub const LABEL_UNABLE: u8 = 0;
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
/// A fragment flag: a mesh triangle whose alpha byte is 0 (contract §1, `alpha_round@1`). The
/// contract makes it not present, so it has no rows and no pairs; it is written so a report that
/// carries it is named `zero_alpha_fragment`. Its `bbox` is the pixels it would cover.
pub const FLAG_ZERO_ALPHA: u16 = 2;

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
    /// The camera-facing label it belongs to (contract §10), an index the frame's `LABL` rows
    /// name, else [`NONE_U32`].
    pub label: u32,
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
    /// The fragment's own key `o` at each sample (adjusted on its own fill); NaN where absent or
    /// not bounded.
    pub depth: [f64; 16],
    /// Its arithmetic bound `R` there; NaN where absent or not bounded.
    pub band: [f64; 16],
}

/// One fragment's band at one pixel, on its own row (`which` 0) or its row on its record's plane
/// (`which` 1): `I = [o − S⁻ − 3R, o + S⁺ + 3R]` at each sample.
#[derive(Clone, Debug, PartialEq)]
pub struct BandRow {
    pub x: u32,
    pub y: u32,
    pub fragment: u32,
    pub which: u8,
    pub o: [f64; 16],
    pub r: [f64; 16],
    pub s_minus: [f64; 16],
    pub s_plus: [f64; 16],
    /// 0 absent, 1 bounded, 2 `band_unbounded`, 3 `curve_recut`.
    pub status: [u8; 16],
}

/// One fragment's colour (contract §9): its kind and, for a solid, its premultiplied bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct ColourRow {
    pub fragment: u32,
    pub kind: u8,
    pub rgba: [u8; 4],
}

/// A per-pixel fragment's colour at one pixel: the least and most premultiplied bytes admitted.
#[derive(Clone, Debug, PartialEq)]
pub struct PixelColour {
    pub x: u32,
    pub y: u32,
    pub fragment: u32,
    pub lo: [u8; 4],
    pub hi: [u8; 4],
}

/// One pair of fragments present at one sample: its admitted edges and its label.
#[derive(Clone, Debug, PartialEq)]
pub struct PairRow {
    pub x: u32,
    pub y: u32,
    /// The pair, `a < b` by fragment id.
    pub a: u32,
    pub b: u32,
    pub sample: u8,
    /// Bit 0 weak (paint order), bit 1 strong `a` in front, bit 2 strong `b` in front.
    pub edges: u8,
    /// 0 `truth_unable`, 1 `Depth(a)`, 2 `Depth(b)`, 3 `TieBand`, 4 `ExactTie`, 5 `own_layers@1`,
    /// 6 `one_plane@1`, 7 `lines@1`, 8 `near_parallel@1`.
    pub label: u8,
    /// How each strong edge is read ([`crate::order::Via`]'s code): `a` in front's in bits 0–1,
    /// `b` in front's in bits 2–3.
    pub provenance: u8,
    /// The pair's edges, in `edges`' bits, in the sets built with the records that paint nothing
    /// offered to the registry too (`invisible_registry_offer`, contract §5).
    pub offered: u8,
}

/// A label's role toward a fragment at one sample (`LABL`): the label's placement facts (contract
/// §10, `label_placement@2`).
pub const ROLE_FRONT: u8 = 0;
/// A qualifying sheet that holds the sample's order point: the label goes immediately behind the
/// rearmost such fragment in the order of the others.
pub const ROLE_ANCHOR: u8 = 1;
/// An open pair or an open class: the sample is `label_rule_open` under the row's causes.
pub const ROLE_OPEN: u8 = 2;
/// `label_kind_unknown`: an anchor in one reading, not in the other; both orders are admitted.
pub const ROLE_KIND_UNKNOWN: u8 = 3;
/// `label_alpha_straddle`: the fill's admitted alpha holds 0 and a positive byte; oracle-unable.
pub const ROLE_ALPHA_STRADDLE: u8 = 4;

/// One label and one other fragment, both present at one sample.
#[derive(Clone, Debug, PartialEq)]
pub struct LabelRow {
    pub x: u32,
    pub y: u32,
    /// The label (`Fragment::label`).
    pub label: u32,
    /// The other fragment, of no label.
    pub fragment: u32,
    pub sample: u8,
    pub role: u8,
    /// 0 none, 1 `interval`, 2 `folded`, 3 `disconnected`, 4 `mesh`, 5 `label`.
    pub pair_cause: u8,
    /// 0 none, 1 `unmapped`, 2 `generic_only`, 3 `paths_disagree`, 4 `no_source`,
    /// 5 `cylinder_grouped`, 6 `point_cloud`.
    pub class_cause: u8,
}

/// A refusal as the file holds it.
#[derive(Clone, Debug, PartialEq)]
pub struct Refused {
    pub reason: u8,
    /// The record kind code, 0xff where none.
    pub kind: u8,
    /// For `stream_refused`, the refused delta's source-row code (`StreamCause`); else 0.
    pub cause: u8,
    pub entry: u32,
    pub slot: u32,
    pub object: u32,
    pub bits: u64,
    /// For `stream_refused`, the seq of the delta that refused the stream; else `u64::MAX`.
    pub at_seq: u64,
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
    /// Sorted as their cover rows, own row before plane row.
    pub bands: Vec<BandRow>,
    /// Sorted by `(y, x, sample, a, b)`.
    pub pairs: Vec<PairRow>,
    /// One per fragment, by id.
    pub colours: Vec<ColourRow>,
    /// One per cover row of a per-pixel fragment, in the cover rows' order.
    pub pixel_colours: Vec<PixelColour>,
    /// Sorted by `(y, x, sample, label, fragment)`.
    pub labels: Vec<LabelRow>,
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
        u32s(&mut frag, &[f.plane, f.label]);
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
        refu.extend([r.reason, r.kind, r.cause, 0]);
        u32s(&mut refu, &[r.entry, r.slot, r.object]);
        refu.extend(r.bits.to_le_bytes());
        refu.extend(r.at_seq.to_le_bytes());
        u32s(&mut refu, &[text.len() as u32, r.detail.len() as u32]);
        text.extend(r.detail.as_bytes());
    }
    let mut band = Vec::with_capacity(d.bands.len() * BAND_SIZE);
    for r in &d.bands {
        u32s(&mut band, &[r.x, r.y, r.fragment]);
        band.extend([r.which, 0, 0, 0]);
        for v in [&r.o, &r.r, &r.s_minus, &r.s_plus] {
            f64s(&mut band, v);
        }
        band.extend(r.status);
    }
    let mut pair = Vec::with_capacity(d.pairs.len() * PAIR_SIZE);
    for r in &d.pairs {
        u32s(&mut pair, &[r.x, r.y, r.a, r.b]);
        pair.extend([r.sample, r.edges, r.label, r.provenance]);
        pair.extend([r.offered, 0, 0, 0]);
    }
    let mut colr = Vec::with_capacity(d.colours.len() * COLOUR_SIZE);
    for r in &d.colours {
        u32s(&mut colr, &[r.fragment]);
        colr.push(r.kind);
        colr.extend(r.rgba);
        colr.extend([0, 0, 0]);
    }
    let mut colp = Vec::with_capacity(d.pixel_colours.len() * PIXEL_COLOUR_SIZE);
    for r in &d.pixel_colours {
        u32s(&mut colp, &[r.x, r.y, r.fragment]);
        colp.extend(r.lo);
        colp.extend(r.hi);
        u32s(&mut colp, &[0]);
    }
    let mut labl = Vec::with_capacity(d.labels.len() * LABEL_SIZE);
    for r in &d.labels {
        u32s(&mut labl, &[r.x, r.y, r.label, r.fragment]);
        labl.extend([r.sample, r.role, r.pair_cause, r.class_cause]);
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
            tag: *b"BAND",
            record_size: BAND_SIZE as u32,
            count: d.bands.len() as u64,
            bytes: band,
        },
        Section {
            tag: *b"PAIR",
            record_size: PAIR_SIZE as u32,
            count: d.pairs.len() as u64,
            bytes: pair,
        },
        Section {
            tag: *b"COLR",
            record_size: COLOUR_SIZE as u32,
            count: d.colours.len() as u64,
            bytes: colr,
        },
        Section {
            tag: *b"COLP",
            record_size: PIXEL_COLOUR_SIZE as u32,
            count: d.pixel_colours.len() as u64,
            bytes: colp,
        },
        Section {
            tag: *b"LABL",
            record_size: LABEL_SIZE as u32,
            count: d.labels.len() as u64,
            bytes: labl,
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
        bands: Vec::new(),
        pairs: Vec::new(),
        colours: Vec::new(),
        pixel_colours: Vec::new(),
        labels: Vec::new(),
        refused: Vec::new(),
    };
    if let Some((_, size, off, count)) = section(b"LABL") {
        for i in 0..count {
            let mut r = Reader {
                b,
                at: off + i * size,
            };
            d.labels.push(LabelRow {
                x: r.u32()?,
                y: r.u32()?,
                label: r.u32()?,
                fragment: r.u32()?,
                sample: r.u8()?,
                role: r.u8()?,
                pair_cause: r.u8()?,
                class_cause: r.u8()?,
            });
        }
    }
    if let Some((_, size, off, count)) = section(b"COLR") {
        for i in 0..count {
            let mut r = Reader {
                b,
                at: off + i * size,
            };
            d.colours.push(ColourRow {
                fragment: r.u32()?,
                kind: r.u8()?,
                rgba: r.take::<4>()?,
            });
        }
    }
    if let Some((_, size, off, count)) = section(b"COLP") {
        for i in 0..count {
            let mut r = Reader {
                b,
                at: off + i * size,
            };
            d.pixel_colours.push(PixelColour {
                x: r.u32()?,
                y: r.u32()?,
                fragment: r.u32()?,
                lo: r.take::<4>()?,
                hi: r.take::<4>()?,
            });
        }
    }
    if let Some((_, size, off, count)) = section(b"BAND") {
        for i in 0..count {
            let mut r = Reader {
                b,
                at: off + i * size,
            };
            let (x, y, fragment) = (r.u32()?, r.u32()?, r.u32()?);
            let which = r.u8()?;
            r.at += 3;
            d.bands.push(BandRow {
                x,
                y,
                fragment,
                which,
                o: r.f64s()?,
                r: r.f64s()?,
                s_minus: r.f64s()?,
                s_plus: r.f64s()?,
                status: r.take::<16>()?,
            });
        }
    }
    if let Some((_, size, off, count)) = section(b"PAIR") {
        for i in 0..count {
            let mut r = Reader {
                b,
                at: off + i * size,
            };
            d.pairs.push(PairRow {
                x: r.u32()?,
                y: r.u32()?,
                a: r.u32()?,
                b: r.u32()?,
                sample: r.u8()?,
                edges: r.u8()?,
                label: r.u8()?,
                provenance: r.u8()?,
                offered: r.u8()?,
            });
        }
    }
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
                label: r.u32()?,
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
            let (reason, kind, cause) = (r.u8()?, r.u8()?, r.u8()?);
            r.at += 1;
            let (entry, slot, object, bits) = (r.u32()?, r.u32()?, r.u32()?, r.u64()?);
            let at_seq = r.u64()?;
            let (start, len) = (r.u32()? as usize, r.u32()? as usize);
            let detail = text
                .get(start..start + len)
                .ok_or("a refusal's text lies outside TEXT")?;
            d.refused.push(Refused {
                reason,
                kind,
                cause,
                entry,
                slot,
                object,
                bits,
                at_seq,
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
             group={} plane={} label={} epsilon={} bbox={}",
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
            f.label,
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
    for r in &d.bands {
        let _ = writeln!(
            s,
            "band x={} y={} fragment={} which={} o={} r={} s_minus={} s_plus={} status={}",
            r.x,
            r.y,
            r.fragment,
            r.which,
            list(r.o.iter().map(|v| hex(*v))),
            list(r.r.iter().map(|v| hex(*v))),
            list(r.s_minus.iter().map(|v| hex(*v))),
            list(r.s_plus.iter().map(|v| hex(*v))),
            list(r.status)
        );
    }
    for r in &d.pairs {
        let _ = writeln!(
            s,
            "pair x={} y={} sample={} a={} b={} edges={} label={} provenance={} offered={}",
            r.x, r.y, r.sample, r.a, r.b, r.edges, r.label, r.provenance, r.offered
        );
    }
    for r in &d.labels {
        let _ = writeln!(
            s,
            "label x={} y={} sample={} label={} fragment={} role={} pair_cause={} class_cause={}",
            r.x, r.y, r.sample, r.label, r.fragment, r.role, r.pair_cause, r.class_cause
        );
    }
    for r in &d.colours {
        let _ = writeln!(
            s,
            "colour fragment={} kind={} rgba={}",
            r.fragment,
            r.kind,
            list(r.rgba)
        );
    }
    for r in &d.pixel_colours {
        let _ = writeln!(
            s,
            "pixel_colour x={} y={} fragment={} lo={} hi={}",
            r.x,
            r.y,
            r.fragment,
            list(r.lo),
            list(r.hi)
        );
    }
    for r in &d.refused {
        let _ = writeln!(
            s,
            "refused reason={} kind={} cause={} entry={} slot={} object={} bits={} at_seq={} detail={}",
            r.reason, r.kind, r.cause, r.entry, r.slot, r.object, r.bits, r.at_seq, r.detail
        );
    }
    s
}
