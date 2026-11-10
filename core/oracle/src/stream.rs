//! A stream's state across deltas, applied as `XF.md` states a delta's meaning, and the frame
//! refusals of contract §11.
//!
//! This is the oracle's own reading of XF's apply semantics, not the core's stream: a delta is
//! checked whole against what it would leave, then applied in XF's order (the view, objects,
//! uploads, records set, records removed, keys released, the order). A delta that is not valid
//! XF changes nothing, and a frame drawn at it is refused (`invalid_delta`). Whatever the oracle
//! cannot judge refuses the whole frame as `truth_unable`, never part of it.

use std::collections::{BTreeMap, HashMap, HashSet};

use arachne_core::xf::{
    Background, Body, Camera, Delta, FEATURE_BACKGROUND_IMAGE, FEATURE_PRE_PROJECTED, ImageSource,
    Key, LightModel, ObjectInfo, OrderEntry, Record, RecordKind, Shape, ShapeData, Space, Stops,
    View,
};

use crate::draw;

/// The XF version the oracle knows: major 1, minor up to 4. A delta of a newer minor carries
/// fields whose meaning it cannot know.
pub const KNOWN_XF: (u16, u16) = (1, 4);

/// The feature bits the oracle knows (contract §11): bit 0, a background-image fill (Cairo's), and
/// bit 1, a pre-projected tail.
pub const KNOWN_BITS: u64 = FEATURE_BACKGROUND_IMAGE | FEATURE_PRE_PROJECTED;

/// Why a frame is `truth_unable`. Codes are `ORACLE_FORMAT`'s (`FORMAT.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Reason {
    /// A record kind the oracle does not know.
    UnknownKind = 1,
    /// A reserved kind: volume, field or emission.
    ReservedKind = 2,
    /// A feature bit the oracle does not know.
    UnknownFeature = 3,
    /// A view whose light model the oracle does not know.
    UnknownLightModel = 4,
    /// A delta of a newer XF minor than the oracle reads, at or before the frame.
    XfNewer = 5,
    /// The frame's delta is not valid XF: the core applies none of it.
    InvalidDelta = 6,
    /// No view has been set.
    NoView = 7,
    /// The core refuses the frame (contract §11's list): there is nothing to judge.
    CoreRefuses = 8,
    /// A delta at or before the frame refused the stream by a source-row code (XF 1.5): that
    /// delta's frame and every later one are `truth_unable` under it, and no later delta clears it.
    /// The refusal carries the delta's seq and its [`StreamCause`].
    StreamRefused = 9,
}

/// Why a stream is refused from a delta on: the delta's source-row refusal code (`XF.md`,
/// "Refusals, each by its code"). Codes are `ORACLE_FORMAT`'s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StreamCause {
    ClassNamedTwice = 1,
    SlotNamedTwice = 2,
    SlotRemoved = 3,
    RowEmpty = 4,
    PathEmpty = 5,
    ClassOutOfRange = 6,
    SlotNotHeld = 7,
}

impl StreamCause {
    pub fn name(self) -> &'static str {
        match self {
            StreamCause::ClassNamedTwice => "class_named_twice",
            StreamCause::SlotNamedTwice => "slot_named_twice",
            StreamCause::SlotRemoved => "slot_removed",
            StreamCause::RowEmpty => "row_empty",
            StreamCause::PathEmpty => "path_empty",
            StreamCause::ClassOutOfRange => "class_out_of_range",
            StreamCause::SlotNotHeld => "slot_not_held",
        }
    }
}

impl Reason {
    pub fn code(self) -> u8 {
        self as u8
    }

    pub fn name(self) -> &'static str {
        match self {
            Reason::UnknownKind => "unknown_kind",
            Reason::ReservedKind => "reserved_kind",
            Reason::UnknownFeature => "unknown_feature",
            Reason::UnknownLightModel => "unknown_light_model",
            Reason::XfNewer => "xf_newer",
            Reason::InvalidDelta => "invalid_delta",
            Reason::NoView => "no_view",
            Reason::CoreRefuses => "core_refuses",
            Reason::StreamRefused => "stream_refused",
        }
    }

    /// The reason a file's code names, if it names one.
    pub fn from_code(code: u8) -> Option<Reason> {
        [
            Reason::UnknownKind,
            Reason::ReservedKind,
            Reason::UnknownFeature,
            Reason::UnknownLightModel,
            Reason::XfNewer,
            Reason::InvalidDelta,
            Reason::NoView,
            Reason::CoreRefuses,
            Reason::StreamRefused,
        ]
        .into_iter()
        .find(|r| r.code() == code)
    }
}

/// One refusal: what the oracle cannot judge, and where.
#[derive(Clone, Debug, PartialEq)]
pub struct Refusal {
    pub reason: Reason,
    /// The order entry's index, where an entry is refused.
    pub entry: Option<u32>,
    pub slot: Option<u32>,
    pub object: Option<u32>,
    /// The record's kind code, where a record is refused.
    pub kind: Option<u8>,
    /// The feature bits refused (unknown ones), else 0.
    pub bits: u64,
    /// For [`Reason::StreamRefused`], the refused delta's seq and code.
    pub stream: Option<(u64, StreamCause)>,
    /// What exactly, in words.
    pub detail: String,
}

impl Refusal {
    fn new(reason: Reason, detail: impl Into<String>) -> Refusal {
        Refusal {
            reason,
            entry: None,
            slot: None,
            object: None,
            kind: None,
            bits: 0,
            stream: None,
            detail: detail.into(),
        }
    }

    fn of(reason: Reason, entry: u32, rec: &Record, detail: impl Into<String>) -> Refusal {
        Refusal {
            entry: Some(entry),
            slot: Some(rec.slot),
            object: Some(rec.object),
            kind: Some(rec.kind.code()),
            ..Refusal::new(reason, detail)
        }
    }
}

/// Whether a record is the overlay's: Cairo draws it over the core's frame, and its order entry
/// is neither drawn nor refused (contract §1).
pub fn is_overlay(rec: &Record) -> bool {
    rec.depth.space == Space::Overlay
        || rec.kind == RecordKind::Image
        || rec.features & FEATURE_BACKGROUND_IMAGE != 0
}

/// What a drawer holds across frames.
#[derive(Clone, Debug, Default)]
pub struct State {
    pub view: Option<View>,
    pub shapes: HashMap<Key, Shape>,
    pub paints: HashMap<Key, Stops>,
    pub objects: BTreeMap<u32, ObjectInfo>,
    pub records: BTreeMap<u32, Record>,
    pub order: Vec<OrderEntry>,
    /// The first delta of a newer XF minor applied, if any: every frame from it on is refused.
    pub newer: Option<(u16, u16)>,
}

impl State {
    pub fn new() -> State {
        State::default()
    }

    /// Applies `d`, or refuses it whole and changes nothing.
    pub fn apply(&mut self, d: &Delta) -> Result<(), Refusal> {
        self.check(d)?;
        if let Some(view) = &d.view {
            self.view = Some(view.clone());
        }
        for o in &d.objects {
            self.objects.insert(o.object, o.clone());
        }
        for s in &d.upload {
            self.shapes.entry(s.key()).or_insert_with(|| s.clone());
        }
        for t in &d.upload_paints {
            self.paints.entry(t.key()).or_insert_with(|| t.clone());
        }
        for r in &d.set {
            self.records.insert(r.slot, r.clone());
        }
        for slot in &d.remove {
            self.records.remove(slot);
        }
        for key in &d.release {
            self.shapes.remove(key);
            self.paints.remove(key);
        }
        if let Some(order) = &d.order {
            self.order = order.clone();
        }
        if d.version.1 > KNOWN_XF.1 && self.newer.is_none() {
            self.newer = Some(d.version);
        }
        Ok(())
    }

    /// `d` checked whole against the stream and what it would leave, in the order `XF.md` gives
    /// apply's checks ("Where each check is made", Apply), so that a delta faulty twice reports the
    /// first: the major; a key uploaded twice; each uploaded shape, then paint table; a release of
    /// a key neither held nor uploaded; a removal of a slot neither held nor set; a release of a key
    /// still in use; the orders of the displays left, when records are set or removed; each record
    /// set; the delta's view; the order. (The reader's checks come first and are the wire reader's;
    /// the source rows are XF 1.5's, which the oracle does not read.)
    fn check(&self, d: &Delta) -> Result<(), Refusal> {
        let bad = |what: String| Err(Refusal::new(Reason::InvalidDelta, what));
        // 1. The major.
        if d.version.0 != KNOWN_XF.0 {
            return bad(format!(
                "XF major {} is not one the oracle reads",
                d.version.0
            ));
        }
        // 3. A key uploaded twice in the delta, as two shapes, two tables or one of each.
        let mut uploaded = HashSet::new();
        for k in d
            .upload
            .iter()
            .map(Shape::key)
            .chain(d.upload_paints.iter().map(Stops::key))
        {
            if !uploaded.insert(k) {
                return bad(format!("key {} uploaded twice in one delta", k.hex()));
            }
        }
        // 4. Each uploaded shape: its own lists, and its key against what the stream holds.
        for s in &d.upload {
            let k = s.key();
            if let ShapeData::Mesh {
                positions,
                triangles,
                normals,
            } = s.data()
            {
                if triangles
                    .iter()
                    .flatten()
                    .any(|&v| v as usize >= positions.len())
                {
                    return bad(format!("mesh {} names a vertex it lacks", k.hex()));
                }
                if normals.as_ref().is_some_and(|n| n.len() != positions.len()) {
                    return bad(format!("mesh {} has normals not one per vertex", k.hex()));
                }
            }
            if let ShapeData::Image {
                width,
                height,
                rgba,
            } = s.data()
                && rgba.len() as u128 != *width as u128 * *height as u128 * 4
            {
                return bad(format!(
                    "image {} holds {} bytes, not {width} x {height} x 4",
                    k.hex(),
                    rgba.len()
                ));
            }
            if self.paints.contains_key(&k) {
                return bad(format!(
                    "shape {} uploaded under a key the stream holds as a paint table",
                    k.hex()
                ));
            }
            if self.shapes.get(&k).is_some_and(|held| held != s) {
                return bad(format!(
                    "shape {} uploaded under a key the stream holds with other content",
                    k.hex()
                ));
            }
        }
        // 5. Each uploaded paint table: its key against the stream's shapes.
        for t in &d.upload_paints {
            if self.shapes.contains_key(&t.key()) {
                return bad(format!(
                    "paint table {} uploaded under a key the stream holds as a shape",
                    t.key().hex()
                ));
            }
        }
        // 6. A release of a key the stream does not hold and the delta does not upload.
        for k in &d.release {
            if !(self.shapes.contains_key(k) || self.paints.contains_key(k) || uploaded.contains(k))
            {
                return bad(format!(
                    "the delta releases {}, which is neither held nor uploaded",
                    k.hex()
                ));
            }
        }
        // 7. A removal of a slot the stream does not hold and the delta does not set.
        for slot in &d.remove {
            if !(self.records.contains_key(slot) || d.set.iter().any(|r| r.slot == *slot)) {
                return bad(format!(
                    "the delta removes slot {slot}, which is neither held nor set"
                ));
            }
        }
        let released: HashSet<Key> = d.release.iter().copied().collect();
        let removed: HashSet<u32> = d.remove.iter().copied().collect();
        // The records the delta leaves: the last set in each slot, unless removed.
        let mut records: BTreeMap<u32, &Record> = self
            .records
            .iter()
            .filter(|(slot, _)| !removed.contains(slot))
            .map(|(slot, r)| (*slot, r))
            .collect();
        for r in &d.set {
            if !removed.contains(&r.slot) {
                records.insert(r.slot, r);
            }
        }
        let view = d.view.as_ref().or(self.view.as_ref());
        // 8. A release of a key a record the delta leaves, or the view, still uses (a record's
        // keys include its display's background).
        for k in &d.release {
            if let Some(r) = records
                .values()
                .find(|r| r.shape_keys().contains(k) || r.paint_keys().contains(k))
            {
                return bad(format!(
                    "the delta releases {}, which slot {} uses",
                    k.hex(),
                    r.slot
                ));
            }
            if view.is_some_and(|v| v.background == Background::Image(*k)) {
                return bad(format!(
                    "the delta releases {}, the view's background image",
                    k.hex()
                ));
            }
        }
        let shape = |k: &Key| -> Option<&Shape> {
            if released.contains(k) {
                return None;
            }
            d.upload
                .iter()
                .find(|s| s.key() == *k)
                .or_else(|| self.shapes.get(k))
        };
        let table = |k: &Key| -> Option<&Stops> {
            if released.contains(k) {
                return None;
            }
            d.upload_paints
                .iter()
                .find(|t| t.key() == *k)
                .or_else(|| self.paints.get(k))
        };
        // 9. When the delta sets or removes a record, the order of every display it leaves (the
        // last set in each slot; 10 reads every display set).
        if !(d.set.is_empty() && d.remove.is_empty()) {
            for r in records.values() {
                if let Body::Image(ImageSource::SubFrame { order, .. }) = &r.body
                    && let Some(what) = order_problem(order, &records)
                {
                    return bad(format!("slot {}'s display's order: {what}", r.slot));
                }
            }
        }
        // 10. Each record the delta sets and does not remove, but one of a kind the oracle does
        // not know.
        for r in d
            .set
            .iter()
            .filter(|r| !removed.contains(&r.slot) && !matches!(r.kind, RecordKind::Unknown(_)))
        {
            if let Some(what) = record_problem(r, &shape, &table, &records) {
                return bad(format!("slot {}: {what}", r.slot));
            }
        }
        // 11. The delta's view.
        if let Some(v) = &d.view
            && let Some(what) = view_problem("the view", v, &shape)
        {
            return bad(what);
        }
        // 12. The order, the delta's or the stream's.
        if let Some(what) = order_problem(d.order.as_ref().unwrap_or(&self.order), &records) {
            return bad(format!("the view's order: {what}"));
        }
        Ok(())
    }

    /// Every refusal of the frame the state draws now (contract §11, and §1's background), in
    /// order; empty when the oracle judges it.
    pub fn refusals(&self) -> Vec<Refusal> {
        let mut out = Vec::new();
        if let Some(v) = self.newer {
            out.push(Refusal::new(
                Reason::XfNewer,
                format!("a delta of XF {}.{} was applied", v.0, v.1),
            ));
        }
        let Some(view) = &self.view else {
            out.push(Refusal::new(Reason::NoView, "no view is set"));
            return out;
        };
        if let LightModel::Other { code, .. } = &view.light {
            out.push(Refusal::new(
                Reason::UnknownLightModel,
                format!("light model {code}"),
            ));
        }
        // Contract §1: the background is an image's top-left W x H corner; a smaller image is
        // refused.
        if let Background::Image(k) = &view.background
            && let Some(ShapeData::Image { width, height, .. }) =
                self.shapes.get(k).map(Shape::data)
            && (*width < view.pixel_width || *height < view.pixel_height)
        {
            out.push(Refusal::new(
                Reason::CoreRefuses,
                format!(
                    "the background image is {width} x {height}, smaller than the {} x {} frame",
                    view.pixel_width, view.pixel_height
                ),
            ));
        }
        let shading = matches!(&view.camera, Camera::ThreeD(c) if c.should_apply_shading);
        let ce_light = view.light == LightModel::Ce;
        for (i, e) in self.order.iter().enumerate() {
            let rec = &self.records[&e.slot];
            if is_overlay(rec) {
                continue;
            }
            if let Some(r) = self.record_refusal(i as u32, rec, shading, ce_light) {
                out.push(r);
            }
        }
        out
    }

    /// Why a drawn record refuses its frame, if it does.
    fn record_refusal(
        &self,
        entry: u32,
        rec: &Record,
        shading: bool,
        ce_light: bool,
    ) -> Option<Refusal> {
        let unknown = rec.features & !KNOWN_BITS;
        if unknown != 0 {
            return Some(Refusal {
                bits: unknown,
                ..Refusal::of(
                    Reason::UnknownFeature,
                    entry,
                    rec,
                    format!("feature bits {unknown:#x}"),
                )
            });
        }
        if let RecordKind::Unknown(code) = rec.kind {
            return Some(Refusal::of(
                Reason::UnknownKind,
                entry,
                rec,
                format!("record kind {code}"),
            ));
        }
        if rec.kind.is_reserved() {
            return Some(Refusal::of(
                Reason::ReservedKind,
                entry,
                rec,
                rec.kind.name(),
            ));
        }
        let refuses = |what: String| Some(Refusal::of(Reason::CoreRefuses, entry, rec, what));
        let prototype = rec
            .geometry
            .shape_a
            .and_then(|k| self.shapes.get(&k))
            .map(Shape::data);
        let mesh = matches!(prototype, Some(ShapeData::Mesh { .. }));
        if let Some(what) = self.paint_refusal(rec, mesh, shading, ce_light) {
            return refuses(what);
        }
        match (&rec.body, prototype) {
            (Body::None, _) if rec.kind == RecordKind::Path => {
                if draw::path_points(rec, &self.shapes).is_none() {
                    return refuses(
                        "a path whose points cannot be drawn: no path shape, or a \
                                    blend of shapes of different lengths"
                            .into(),
                    );
                }
            }
            (Body::FaceGroup(f), Some(ShapeData::Path { points })) => {
                if f.uv.is_empty() || (f.face_ends.is_empty() && points.len() % f.uv.len() != 0) {
                    return refuses(
                        "a face group whose points do not divide into its faces".into(),
                    );
                }
            }
            (Body::Instances(body), Some(data)) => {
                let per = match data {
                    ShapeData::Mesh { triangles, .. } => triangles.len() as u64,
                    _ => 3,
                };
                if body.placements.len() as u64 * per > u32::MAX as u64 {
                    return refuses("an instances record of more parts than 2^32".into());
                }
            }
            (Body::Points(body), Some(ShapeData::Path { points: own })) => {
                let positions = match &rec.pre_projected {
                    Some(tail) => match self.shapes.get(&tail.points).map(Shape::data) {
                        Some(ShapeData::Path { points }) => points.len(),
                        _ => 0,
                    },
                    None => own.len(),
                };
                if rec.pre_projected.is_some()
                    && self.paints.get(&body.colors).map(|t| t.rows().len()) != Some(positions)
                {
                    return refuses(
                        "a pre-projected points record placing other than one point per colour"
                            .into(),
                    );
                }
                let side = draw::block_side(body.radius) as u128;
                if side * side * positions as u128 > u32::MAX as u128 + 1 {
                    return refuses("a points record of more block writes than 2^32".into());
                }
            }
            _ => {}
        }
        None
    }

    /// Contract §11's paint and geometry refusals, read from the paint the record draws with.
    fn paint_refusal(
        &self,
        rec: &Record,
        mesh: bool,
        shading: bool,
        ce_light: bool,
    ) -> Option<String> {
        let tail = rec.pre_projected.as_ref();
        let shaded = tail.is_none() && shading && rec.depth.shade_in_3d;
        let shaped = rec.geometry.blend.is_some()
            || rec.geometry.reveal.is_some()
            || rec.paint.paint_b.is_some();
        if shaped && rec.kind != RecordKind::Path {
            return Some("a blend, reveal or paint blend on a record that is not a path".into());
        }
        if shaped && tail.is_some() {
            return Some("a blend, reveal or paint blend on a pre-projected path".into());
        }
        if rec.clip.is_some() {
            return Some("a clip".into());
        }
        if rec.kind == RecordKind::Points {
            return None;
        }
        let mut p = match draw::drawn_paint(&rec.paint, &self.paints) {
            Ok(p) => p,
            Err(what) => return Some(what.into()),
        };
        let mut ends = rec.paint.gradient;
        if let Some(t) = tail {
            p.fill = draw::rows(&t.fill, &self.paints);
            p.stroke = draw::rows(&t.stroke, &self.paints);
            p.background_stroke = draw::rows(&t.background_stroke, &self.paints);
            ends = t.gradient;
        }
        let drawn = |rows: &Option<Vec<[f64; 4]>>, width: Option<f64>| {
            rows.is_some() && width.is_none_or(|w| w > 0.0)
        };
        let gradient = |rows: &Option<Vec<[f64; 4]>>| rows.as_ref().is_some_and(|r| r.len() > 1);
        let strokes = [
            (&p.stroke, p.stroke_width),
            (&p.background_stroke, p.background_stroke_width),
        ];
        if mesh {
            if strokes.iter().any(|(rows, w)| drawn(rows, Some(*w))) {
                return Some("a mesh with a stroke".into());
            }
            if gradient(&p.fill) {
                return Some("a mesh with a gradient fill".into());
            }
            if rec.dash.is_some() {
                return Some("a mesh with a dash".into());
            }
            return None;
        }
        if shaded
            && !ce_light
            && (rec.kind == RecordKind::FaceGroup
                || drawn(&p.fill, None)
                || strokes.iter().any(|(rows, w)| drawn(rows, Some(*w))))
        {
            return Some("a shaded path under a light model other than CE's".into());
        }
        if (gradient(&p.fill)
            || strokes
                .iter()
                .any(|(rows, w)| drawn(rows, Some(*w)) && gradient(rows)))
            && (rec.kind == RecordKind::Instances || (!shaded && ends.is_none()))
        {
            return Some(
                "a gradient paint without ends on an unshaded path, or on an instances record"
                    .into(),
            );
        }
        if let Some(d) = rec.dash
            && !(d.period > 0.0
                && d.period.is_finite()
                && d.duty > 0.0
                && d.duty <= 1.0
                && d.phase.is_finite())
        {
            return Some(
                "a dash whose period is not finite and above 0, duty not in (0, 1] or phase \
                 not finite"
                    .into(),
            );
        }
        None
    }
}

/// What makes a view invalid XF against what the delta leaves, if anything (`XF.md`, apply's check
/// 11, and 10 for a display's): soft diffuse light outside its ranges, or a background image that is
/// not an image shape or that the stream would not hold.
fn view_problem<'a>(
    name: &str,
    v: &View,
    shape: &dyn Fn(&Key) -> Option<&'a Shape>,
) -> Option<String> {
    if let LightModel::SoftDiffuse { ambient, wrap } = &v.light
        && !((0.0..=1.0).contains(ambient) && *wrap >= 0.0)
    {
        return Some(format!(
            "{name}'s soft diffuse light parameters are out of range"
        ));
    }
    let Background::Image(k) = &v.background else {
        return None;
    };
    match shape(k).map(Shape::data) {
        Some(ShapeData::Image { .. }) => None,
        Some(_) => Some(format!(
            "{name} names {} as its background image, not an image shape",
            k.hex()
        )),
        None => Some(format!(
            "{name} names background image {} it lacks",
            k.hex()
        )),
    }
}

/// What makes a record the delta sets invalid XF against what the delta leaves, if anything: the
/// first fault of `XF.md`'s apply check 10, in its order. The reader's checks again (the feature
/// bits and their fields, a tail on a kind that takes none); a kind that needs a shape given none,
/// or a shape of a kind it cannot take (its own and its blend's; a fill image that is not an image,
/// a tail's points that are not a path); a sub-frame display given a shape; a body of another
/// kind's; a shape or paint table the stream would not hold; a display's view and its order against
/// `records`, what the delta leaves (a display set and then set again in its slot is one check 9
/// does not read); a table of other than one row per face, point or placement, placements and
/// objects differing in number; a face group's face ends and widths, and joins.
fn record_problem<'a>(
    r: &Record,
    shape: &dyn Fn(&Key) -> Option<&'a Shape>,
    table: &dyn Fn(&Key) -> Option<&'a Stops>,
    records: &BTreeMap<u32, &Record>,
) -> Option<String> {
    let kind = r.kind.name();
    if (r.features & FEATURE_BACKGROUND_IMAGE != 0) != r.fill_image.is_some() {
        return Some("the background_image bit and fill_image disagree".into());
    }
    if (r.features & FEATURE_PRE_PROJECTED != 0) != r.pre_projected.is_some() {
        return Some("the pre_projected bit and its tail disagree".into());
    }
    if r.pre_projected.is_some()
        && !matches!(
            r.kind,
            RecordKind::Path | RecordKind::Points | RecordKind::Image
        )
    {
        return Some(format!("a pre-projected tail on a {kind} record"));
    }
    let sub_frame = matches!(r.body, Body::Image(ImageSource::SubFrame { .. }));
    type Takes = fn(&ShapeData) -> bool;
    let path: Takes = |s| matches!(s, ShapeData::Path { .. });
    let image: Takes = |s| matches!(s, ShapeData::Image { .. });
    // The shapes a kind takes; none for a kind that needs no shape (a sub-frame display, the
    // reserved kinds).
    let takes: Option<Takes> = match r.kind {
        RecordKind::Path | RecordKind::FaceGroup | RecordKind::Points => Some(path),
        RecordKind::Mesh => Some(|s| matches!(s, ShapeData::Mesh { .. })),
        RecordKind::Instances => {
            Some(|s| matches!(s, ShapeData::Path { .. } | ShapeData::Mesh { .. }))
        }
        RecordKind::Image if !sub_frame => Some(image),
        _ => None,
    };
    let mut kinds: Vec<(Key, Takes)> = Vec::new();
    if let Some(takes) = takes {
        let Some(a) = r.geometry.shape_a else {
            return Some(format!("a {kind} record given no shape"));
        };
        kinds.push((a, takes));
        kinds.extend(r.geometry.blend.as_ref().map(|b| (b.shape_b, takes)));
    }
    kinds.extend(r.fill_image.map(|k| (k, image)));
    kinds.extend(r.pre_projected.as_ref().map(|t| (t.points, path)));
    for (k, takes) in kinds {
        if let Some(s) = shape(&k)
            && !takes(s.data())
        {
            return Some(format!(
                "shape {} is of a kind a {kind} record cannot take there",
                k.hex()
            ));
        }
    }
    if sub_frame && r.geometry.shape_a.is_some() {
        return Some("a sub-frame display given a shape".into());
    }
    let body = match r.kind {
        RecordKind::Path | RecordKind::Mesh => matches!(r.body, Body::None),
        RecordKind::FaceGroup => matches!(r.body, Body::FaceGroup(_)),
        RecordKind::Points => matches!(r.body, Body::Points(_)),
        RecordKind::Image => matches!(r.body, Body::Image(_)),
        RecordKind::Instances => matches!(r.body, Body::Instances(_)),
        _ => matches!(r.body, Body::Opaque(_)),
    };
    if !body {
        return Some(format!("a {kind} record with another kind's body"));
    }
    let own = r
        .geometry
        .shape_a
        .into_iter()
        .chain(r.geometry.blend.as_ref().map(|b| b.shape_b))
        .chain(r.clip.as_ref().map(|(k, _)| *k))
        .chain(r.fill_image)
        .chain(r.pre_projected.as_ref().map(|t| t.points));
    for k in own {
        if shape(&k).is_none() {
            return Some(format!("names shape {} it lacks", k.hex()));
        }
    }
    for k in r.paint_keys() {
        if table(&k).is_none() {
            return Some(format!("names paint table {} it lacks", k.hex()));
        }
    }
    if let Body::Image(ImageSource::SubFrame { view, order, .. }) = &r.body {
        if let Some(what) = view_problem("its display", view, shape) {
            return Some(what);
        }
        if let Some(what) = order_problem(order, records) {
            return Some(format!("its display's order: {what}"));
        }
    }
    // Every table named is held, by the check above.
    let one_per = |name: &str, k: &Key, n: usize, per: &str| -> Option<String> {
        let got = table(k).map_or(0, |t| t.rows().len());
        (got != n).then(|| format!("its {name} table holds {got} rows, not one per {per} ({n})"))
    };
    let tables = match &r.body {
        Body::FaceGroup(f) => one_per("fill", &f.fill, f.uv.len(), "face").or_else(|| {
            f.stroke
                .and_then(|k| one_per("stroke", &k, f.uv.len(), "face"))
        }),
        Body::Points(p) => {
            // The points it draws: its tail's when pre-projected, else its shape's.
            let drawn = r
                .pre_projected
                .as_ref()
                .map(|t| t.points)
                .or(r.geometry.shape_a)
                .and_then(|k| shape(&k))
                .map_or(0, |s| match s.data() {
                    ShapeData::Path { points } => points.len(),
                    _ => 0,
                });
            one_per("colour", &p.colors, drawn, "point")
        }
        Body::Instances(i) => {
            let n = i.placements.len();
            one_per("fill", &i.fill, n, "placement")
                .or_else(|| one_per("stroke", &i.stroke, n, "placement"))
                .or_else(|| {
                    (i.objects.len() != n).then(|| {
                        format!(
                            "its placements ({n}) and objects ({}) differ in number",
                            i.objects.len()
                        )
                    })
                })
        }
        _ => None,
    };
    tables.or_else(|| record_lists(r, shape))
}

/// What makes a record's own lists invalid XF, if anything: a face group's face ends and widths
/// (XF 1.3), and a record's joins, one per cubic of the points a path record draws (XF 1.4).
fn record_lists<'a>(r: &Record, shape: &dyn Fn(&Key) -> Option<&'a Shape>) -> Option<String> {
    if let Body::FaceGroup(f) = &r.body {
        let len = r
            .geometry
            .shape_a
            .and_then(|k| shape(&k))
            .map(|s| match s.data() {
                ShapeData::Path { points } => points.len(),
                _ => 0,
            })
            .unwrap_or(0);
        if !f.face_ends.is_empty()
            && (f.face_ends.len() != f.uv.len()
                || f.face_ends.windows(2).any(|w| w[0] >= w[1])
                || f.face_ends.first() == Some(&0)
                || f.face_ends.last().map(|&e| e as usize) != Some(len))
        {
            return Some(
                "face ends not one per face, not rising, or not ending at the points".into(),
            );
        }
        if !f.face_widths.is_empty()
            && (f.face_widths.len() != f.uv.len()
                || f.face_widths.iter().any(|w| !(w.is_finite() && *w >= 0.0)))
        {
            return Some("face widths not one per face, or not finite and at least 0".into());
        }
    }
    if let Some(joins) = &r.joins {
        if r.kind != RecordKind::Path {
            return Some("joins on a record that is not a path".into());
        }
        let points = match &r.pre_projected {
            Some(tail) => shape(&tail.points).and_then(|s| match s.data() {
                ShapeData::Path { points } => Some(points.len()),
                _ => None,
            }),
            None => draw::cut_len(r, &|k: &Key| {
                shape(k).and_then(|s| match s.data() {
                    ShapeData::Path { points } => Some(points.as_slice()),
                    _ => None,
                })
            }),
        };
        if points.map(|n| n / 4) != Some(joins.len()) {
            return Some("joins not one per cubic of the points the path draws".into());
        }
    }
    None
}

/// Why an order drawn into one view is not valid XF against the records a delta leaves, if it is
/// not: an entry naming a slot or part that does not exist, or two displays drawn in one layer
/// sharing a capture rank (`XF.md`, "Sub-frame capture"; the same rank in another layer or view is
/// allowed, as is one display drawn twice).
fn order_problem(order: &[OrderEntry], records: &BTreeMap<u32, &Record>) -> Option<String> {
    let mut ranks: HashMap<(u8, u32), u32> = HashMap::new();
    for e in order {
        let Some(r) = records.get(&e.slot) else {
            return Some(format!("it names slot {} it lacks", e.slot));
        };
        let parts = match &r.body {
            Body::FaceGroup(f) => f.uv.len() as u32,
            _ => 1,
        };
        if e.part >= parts {
            return Some(format!("it names part {} of slot {}", e.part, e.slot));
        }
        if let Body::Image(ImageSource::SubFrame { rank, .. }) = &r.body
            && let Some(other) = ranks.insert((e.layer as u8, *rank), e.slot)
            && other != e.slot
        {
            return Some(format!(
                "displays in slots {other} and {} share capture rank {rank} in layer {}",
                e.slot, e.layer as u8
            ));
        }
    }
    None
}
