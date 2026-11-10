//! XF 1.5's source section, read by the oracle itself from `XF.md`'s binary form (the shared XF
//! reader decodes everything before it and skips it as a newer minor's).
//!
//! A file is split into its deltas here, each read as its own item: the file is refused only for
//! its magic or its major, and a delta that does not read is refused alone (contract §11). Each
//! delta's fields up to its order are decoded by the shared reader; this module walks the same
//! bytes by their framing alone to find where the order ends, then reads the section that follows
//! (XF 1.5: class names, then rows) and makes the reader's six section checks in the order the
//! wire gives them. `apply`'s seventh, `slot_not_held`, is the stream's (`stream.rs`).
//!
//! The object table's `stable_id` and `source` strings are never read here: they are opaque.

use arachne_core::xf::Delta;
use arachne_core::xf::wire::decode_delta;

use crate::stream::StreamCause;

/// One node of a source path: the writer's token for an object, and its class's index into the
/// section's class names.
pub type Node = (u32, u32);

/// A row: the slot it is keyed by, its facing group (0: none), and its paths of containment,
/// each the record's owner first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub slot: u32,
    pub facing_group: u32,
    pub paths: Vec<Vec<Node>>,
}

/// A delta's source section.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Section {
    pub names: Vec<String>,
    pub rows: Vec<Row>,
}

impl Section {
    /// A row's paths with each class index replaced by its name.
    pub fn named(&self, row: &Row) -> Vec<Vec<(u32, String)>> {
        row.paths
            .iter()
            .map(|p| {
                p.iter()
                    .map(|&(token, class)| (token, self.names[class as usize].clone()))
                    .collect()
            })
            .collect()
    }
}

/// Why a delta does not read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unread {
    /// An ordinary refusal of the reader (`invalid_delta`): the name of the wire reader's error
    /// (`Truncated`, `Invalid`, `Version`, `KeyMismatch`), and its words.
    Ordinary { error: String, detail: String },
    /// One of the section's six reader codes: the stream is refused from this delta on.
    Code(StreamCause),
}

/// One delta of a file as read: its stated version and seq (read from its fixed header where the
/// rest does not read), and what it holds.
#[derive(Clone, Debug)]
pub struct WireDelta {
    pub version: (u16, u16),
    pub seq: u64,
    pub read: Result<(Delta, Option<Section>), Unread>,
}

impl WireDelta {
    /// A delta built in memory, with no section.
    pub fn of(d: Delta) -> WireDelta {
        WireDelta {
            version: d.version,
            seq: d.seq,
            read: Ok((d, None)),
        }
    }
}

/// An XF version, (major, minor).
pub type Version = (u16, u16);

/// Why a file does not read at all: its magic, its major, or its framing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileError(pub String);

struct Bytes<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Bytes<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&e| e <= self.b.len())
            .ok_or_else(|| format!("{n} bytes wanted at {} of {}", self.at, self.b.len()))?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn bool(&mut self) -> Result<bool, String> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            v => Err(format!("a bool of {v}")),
        }
    }
    fn item(&mut self) -> Result<&'a [u8], String> {
        let n = self.u32()? as usize;
        self.take(n)
    }
    /// Skips `count` items, or `count` fixed records of `size` bytes.
    fn skip_items(&mut self) -> Result<(), String> {
        let n = self.u32()?;
        for _ in 0..n {
            self.item()?;
        }
        Ok(())
    }
    fn skip_fixed(&mut self, size: usize) -> Result<(), String> {
        let n = self.u32()? as usize;
        self.take(n.checked_mul(size).ok_or("a count that overflows")?)?;
        Ok(())
    }
}

/// Splits an XF file into its deltas' bytes (`XF.md`, Binary form: the magic, the major and
/// minor, a count, then each delta as an item).
pub fn split_file(bytes: &[u8]) -> Result<(Version, Vec<&[u8]>), FileError> {
    let mut r = Bytes { b: bytes, at: 0 };
    let bad = |e: String| FileError(e);
    if r.take(4).map_err(bad)? != b"ARXF" {
        return Err(FileError("not an XF file (its magic)".into()));
    }
    let version = (r.u16().map_err(bad)?, r.u16().map_err(bad)?);
    if version.0 != crate::stream::KNOWN_XF.0 {
        return Err(FileError(format!("XF major {}", version.0)));
    }
    let n = r.u32().map_err(bad)?;
    let mut deltas = Vec::new();
    for _ in 0..n {
        deltas.push(r.item().map_err(bad)?);
    }
    Ok((version, deltas))
}

/// The wire reader's error, by its variant's name, and its words.
fn ordinary(e: &arachne_core::xf::wire::XfError) -> Unread {
    let words = format!("{e:?}");
    let error = words
        .split(|c: char| !c.is_ascii_alphanumeric())
        .next()
        .unwrap_or("")
        .to_string();
    Unread::Ordinary {
        error,
        detail: words,
    }
}

/// Reads one delta's bytes: the shared reader's fields, then, for a delta of minor 5 or later,
/// its source section with the reader's six checks.
pub fn read_delta(bytes: &[u8]) -> WireDelta {
    // The fixed header, read where the rest may not read.
    let mut h = Bytes { b: bytes, at: 0 };
    let version = (h.u16().unwrap_or(0), h.u16().unwrap_or(0));
    let seq = {
        let mut s = Bytes { b: bytes, at: 20 };
        s.u64().unwrap_or(u64::MAX)
    };
    let read = (|| {
        let d = decode_delta(bytes).map_err(|e| ordinary(&e))?;
        if version.1 < 5 {
            return Ok((d, None));
        }
        let at = order_end(bytes).map_err(|e| Unread::Ordinary {
            error: "Truncated".into(),
            detail: e,
        })?;
        let section = read_section(&bytes[at..], &d)?;
        Ok((d, Some(section)))
    })();
    WireDelta { version, seq, read }
}

/// Where a delta's order ends: its fields walked by their framing alone.
fn order_end(bytes: &[u8]) -> Result<usize, String> {
    let mut r = Bytes { b: bytes, at: 28 };
    if r.bool()? {
        r.item()?; // the view
    }
    r.skip_items()?; // objects
    r.skip_items()?; // shapes
    r.skip_items()?; // paint tables
    r.skip_fixed(16)?; // released keys
    r.skip_items()?; // records
    r.skip_fixed(4)?; // removed slots
    if r.bool()? {
        r.skip_fixed(9)?; // the order: u8 layer, u32 slot, u32 part
    }
    Ok(r.at)
}

/// The section after a delta's order, and the reader's checks in the wire's order: the class
/// names first (`class_named_twice`), then each row as written, at the first that offends the
/// first of `slot_named_twice`, `slot_removed`, `row_empty`, `path_empty`, `class_out_of_range`.
/// Bytes that end inside the section are the delta's ordinary refusal. Whatever follows the rows
/// belongs to a newer minor and is skipped.
fn read_section(bytes: &[u8], d: &Delta) -> Result<Section, Unread> {
    let mut r = Bytes { b: bytes, at: 0 };
    let truncated = |e: String| Unread::Ordinary {
        error: "Truncated".into(),
        detail: format!("the source section: {e}"),
    };
    let n = r.u32().map_err(truncated)?;
    let mut names = Vec::new();
    for _ in 0..n {
        let s = r.item().map_err(truncated)?;
        let name = std::str::from_utf8(s).map_err(|_| Unread::Ordinary {
            error: "Invalid".into(),
            detail: "a class name that is not UTF-8".into(),
        })?;
        names.push(name.to_string());
    }
    let mut seen = std::collections::HashSet::new();
    if !names.iter().all(|n| seen.insert(n.as_str())) {
        return Err(Unread::Code(StreamCause::ClassNamedTwice));
    }
    let n = r.u32().map_err(truncated)?;
    let mut rows: Vec<Row> = Vec::new();
    for _ in 0..n {
        let slot = r.u32().map_err(truncated)?;
        let facing_group = r.u32().map_err(truncated)?;
        let paths_n = r.u32().map_err(truncated)?;
        let mut paths = Vec::new();
        for _ in 0..paths_n {
            let nodes = r.u32().map_err(truncated)?;
            let mut path = Vec::new();
            for _ in 0..nodes {
                path.push((r.u32().map_err(truncated)?, r.u32().map_err(truncated)?));
            }
            paths.push(path);
        }
        let row = Row {
            slot,
            facing_group,
            paths,
        };
        let code = if rows.iter().any(|earlier| earlier.slot == slot) {
            Some(StreamCause::SlotNamedTwice)
        } else if d.remove.contains(&slot) {
            Some(StreamCause::SlotRemoved)
        } else if facing_group == 0 && row.paths.is_empty() {
            Some(StreamCause::RowEmpty)
        } else if row.paths.iter().any(Vec::is_empty) {
            Some(StreamCause::PathEmpty)
        } else if row
            .paths
            .iter()
            .flatten()
            .any(|&(_, c)| c as usize >= names.len())
        {
            Some(StreamCause::ClassOutOfRange)
        } else {
            None
        };
        if let Some(code) = code {
            return Err(Unread::Code(code));
        }
        rows.push(row);
    }
    Ok(Section { names, rows })
}

/// Reads a file: its version and each delta as read.
pub fn read_file_bytes(bytes: &[u8]) -> Result<(Version, Vec<WireDelta>), FileError> {
    let (version, items) = split_file(bytes)?;
    Ok((version, items.into_iter().map(read_delta).collect()))
}

/// Reads the file at `path`.
pub fn read_file(path: &std::path::Path) -> Result<(Version, Vec<WireDelta>), FileError> {
    let bytes = std::fs::read(path).map_err(|e| FileError(format!("{}: {e}", path.display())))?;
    read_file_bytes(&bytes)
}
