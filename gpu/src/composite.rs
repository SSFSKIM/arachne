//! The reader of a proposed Frame IR record kind, `composite`: entries drawn into a picture of
//! their own, with the effects the compositor applies to a layer.
//!
//! The layout is a proposal: it is not yet in `core/XF.md`, and its kind code and the minor
//! version that carries it are not yet allocated. A `composite` record begins with the four
//! fields every record begins with (`u8 kind  u64 features  u32 slot  u32 object`), which the
//! stream's own reader reads; this reads what follows, in place of the geometry and paint a drawn
//! record carries, in the binary form of `core/XF.md` (little-endian, `count` a u32, `item` a u32
//! byte length and its bytes, `bool` a u8 0 or 1):
//!
//! ```text
//! u8 flags (1 blur, 2 mask, 4 depth, 8 focus, 16 window)
//! [f64 sigma]  [f64 depth]  [f64 focus_depth  f64 aperture]
//! [item(view)  f64 x12 corners  u8 resampling]                    (window)
//! count (u8 layer  u32 slot  u32 part)*                           (the entries drawn into it)
//! [count (u8 layer  u32 slot  u32 part)*  f64 feather  bool invert]   (the mask's entries)
//! ```
//!
//! Bytes after these belong to a newer minor and are skipped. A flag this reader does not know
//! makes the record unreadable, since what follows depends on it.

use std::fmt;

use crate::frame::Focus;

pub const BLUR: u8 = 1;
pub const MASK: u8 = 2;
pub const DEPTH: u8 = 4;
pub const FOCUS: u8 = 8;
pub const WINDOW: u8 = 16;
const KNOWN: u8 = BLUR | MASK | DEPTH | FOCUS | WINDOW;

/// One entry of a draw order: the order's layer, a record's slot, and the part of it drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    pub layer: u8,
    pub slot: u32,
    pub part: u32,
}

/// A window's view and its four corners, in CE's image order, as the image record's sub-frame
/// form holds them.
#[derive(Clone, Debug, PartialEq)]
pub struct CompositeWindow {
    /// The view's bytes, unread: the stream's reader reads a view.
    pub view: Vec<u8>,
    pub corners: [[f64; 3]; 4],
    pub resampling: u8,
}

/// The entries drawn into a soft mask's picture, its feather and whether it is inverted.
#[derive(Clone, Debug, PartialEq)]
pub struct CompositeMask {
    pub entries: Vec<Entry>,
    pub feather: f64,
    pub invert: bool,
}

/// The body of a `composite` record.
#[derive(Clone, Debug, PartialEq)]
pub struct Composite {
    pub sigma: Option<f64>,
    pub depth: Option<f64>,
    pub focus: Option<Focus>,
    pub window: Option<CompositeWindow>,
    /// The entries drawn into this picture, and nowhere else.
    pub entries: Vec<Entry>,
    pub mask: Option<CompositeMask>,
}

/// Why a body could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompositeError(pub String);

impl fmt::Display for CompositeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "a composite record: {}", self.0)
    }
}

impl std::error::Error for CompositeError {}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], CompositeError> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&end| end <= self.bytes.len());
        let Some(end) = end else {
            return Err(CompositeError(format!(
                "ends at byte {} inside {what}",
                self.bytes.len()
            )));
        };
        let taken = &self.bytes[self.at..end];
        self.at = end;
        Ok(taken)
    }

    fn u8(&mut self, what: &str) -> Result<u8, CompositeError> {
        Ok(self.take(1, what)?[0])
    }

    fn u32(&mut self, what: &str) -> Result<u32, CompositeError> {
        let bytes = self.take(4, what)?;
        Ok(u32::from_le_bytes(bytes.try_into().expect("4 bytes")))
    }

    fn f64(&mut self, what: &str) -> Result<f64, CompositeError> {
        let bytes = self.take(8, what)?;
        Ok(f64::from_le_bytes(bytes.try_into().expect("8 bytes")))
    }

    fn bool(&mut self, what: &str) -> Result<bool, CompositeError> {
        match self.u8(what)? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(CompositeError(format!("{what} is {other}, not a bool"))),
        }
    }

    fn entries(&mut self, what: &str) -> Result<Vec<Entry>, CompositeError> {
        let count = self.u32(what)? as usize;
        // Each entry is 9 bytes: a count past what is left cannot be read.
        if count > (self.bytes.len() - self.at) / 9 {
            return Err(CompositeError(format!(
                "{what} counts {count} entries past its end"
            )));
        }
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            entries.push(Entry {
                layer: self.u8(what)?,
                slot: self.u32(what)?,
                part: self.u32(what)?,
            });
        }
        Ok(entries)
    }
}

/// Reads a `composite` record's body: the bytes after its four header fields.
pub fn read_composite(body: &[u8]) -> Result<Composite, CompositeError> {
    let mut r = Reader { bytes: body, at: 0 };
    let flags = r.u8("the flags")?;
    if flags & !KNOWN != 0 {
        return Err(CompositeError(format!(
            "flags {flags:#04x} hold bits this reader does not know"
        )));
    }
    let has = |flag: u8| flags & flag != 0;
    let sigma = if has(BLUR) {
        Some(r.f64("sigma")?)
    } else {
        None
    };
    let depth = if has(DEPTH) {
        Some(r.f64("depth")?)
    } else {
        None
    };
    let focus = if has(FOCUS) {
        Some(Focus {
            depth: r.f64("the focus depth")?,
            aperture: r.f64("the aperture")?,
        })
    } else {
        None
    };
    let window = if has(WINDOW) {
        let length = r.u32("the window's view")? as usize;
        let view = r.take(length, "the window's view")?.to_vec();
        let mut corners = [[0.0; 3]; 4];
        for value in corners.iter_mut().flatten() {
            *value = r.f64("the window's corners")?;
        }
        let resampling = r.u8("the window's resampling")?;
        Some(CompositeWindow {
            view,
            corners,
            resampling,
        })
    } else {
        None
    };
    let entries = r.entries("the entries")?;
    let mask = if has(MASK) {
        Some(CompositeMask {
            entries: r.entries("the mask's entries")?,
            feather: r.f64("the mask's feather")?,
            invert: r.bool("the mask's invert")?,
        })
    } else {
        None
    };
    Ok(Composite {
        sigma,
        depth,
        focus,
        window,
        entries,
        mask,
    })
}
