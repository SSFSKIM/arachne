//! A Stream: what a drawer holds across frames. Deltas change it, atomically; `draw` turns it into
//! one frame's pixels, or refuses the whole frame when any drawn record carries what this build
//! cannot draw (`render.unsupported`).

use crate::composite::{self, Frame, Producer, Rendered};
use crate::mesh::{self, MeshDraw};
use crate::xf::*;
use crate::{geometry, instance, path, points, tiles};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

/// Shapes, paint tables, objects, records, the view and the draw order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stream {
    shapes: HashMap<Key, Shape>,
    paints: HashMap<Key, Stops>,
    objects: BTreeMap<u32, ObjectInfo>,
    records: BTreeMap<u32, Record>,
    view: Option<View>,
    order: Vec<OrderEntry>,
}

/// What one delta changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ApplyStats {
    /// Shapes the delta uploaded.
    pub uploaded_shapes: usize,
    /// Records the delta set or removed.
    pub changed_records: usize,
}

/// Why a delta was refused. A refused delta changes nothing.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamError {
    /// A major version this build does not draw.
    Version(String),
    /// A reference to a shape, paint table or slot the stream would not hold.
    Missing(String),
    /// A release of a key something still uses.
    InUse(String),
    /// A record or order entry that contradicts itself or what it references.
    Invalid(String),
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamError::Version(m)
            | StreamError::Missing(m)
            | StreamError::InUse(m)
            | StreamError::Invalid(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for StreamError {}

/// A frame refused because a drawn record (or the view) carries a kind or feature this build does
/// not draw: `render.unsupported`. Nothing is drawn.
#[derive(Clone, Debug, PartialEq)]
pub struct Unsupported {
    /// The record kind's name, or `view`.
    pub kind: String,
    pub slot: Option<u32>,
    pub object: Option<u32>,
    pub reason: String,
}

/// Why `draw` gave no frame.
#[derive(Clone, Debug, PartialEq)]
pub enum DrawError {
    NoView,
    Invalid(String),
    Unsupported(Unsupported),
    /// A drawing thread the system would not start.
    Threads(String),
}

impl fmt::Display for DrawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DrawError::NoView => f.write_str("the stream has no view yet"),
            DrawError::Invalid(m) => f.write_str(m),
            DrawError::Threads(m) => write!(f, "a drawing thread could not start: {m}"),
            DrawError::Unsupported(u) => write!(
                f,
                "render.unsupported: {} (slot {:?}, object {:?}): {}",
                u.kind, u.slot, u.object, u.reason
            ),
        }
    }
}

impl std::error::Error for DrawError {}

/// The stream as a delta would leave it, read without changing anything, by the rules commit
/// applies: a key the stream holds keeps what it holds (a validated delta re-uploads only the same
/// content under it, and uploads each key once); the last record set in a slot wins, then removals.
struct After<'a> {
    stream: &'a Stream,
    uploads: HashMap<Key, &'a Shape>,
    upload_paints: HashMap<Key, &'a Stops>,
    /// The records the delta sets, by slot (the last one set wins).
    set: BTreeMap<u32, &'a Record>,
    released: HashSet<Key>,
    removed: HashSet<u32>,
}

impl<'a> After<'a> {
    fn new(stream: &'a Stream, delta: &'a Delta) -> After<'a> {
        After {
            stream,
            uploads: delta.upload.iter().map(|s| (s.key(), s)).collect(),
            upload_paints: delta.upload_paints.iter().map(|s| (s.key(), s)).collect(),
            set: delta.set.iter().map(|r| (r.slot, r)).collect(),
            released: delta.release.iter().copied().collect(),
            removed: delta.remove.iter().copied().collect(),
        }
    }

    fn shape(&self, key: &Key) -> Option<&'a ShapeData> {
        if self.released.contains(key) {
            return None;
        }
        self.stream
            .shapes
            .get(key)
            .or_else(|| self.uploads.get(key).copied())
            .map(|s| s.data())
    }

    fn paint(&self, key: &Key) -> Option<&'a Stops> {
        if self.released.contains(key) {
            return None;
        }
        self.stream
            .paints
            .get(key)
            .or_else(|| self.upload_paints.get(key).copied())
    }

    fn record(&self, slot: u32) -> Option<&'a Record> {
        if self.removed.contains(&slot) {
            return None;
        }
        self.set
            .get(&slot)
            .copied()
            .or_else(|| self.stream.records.get(&slot))
    }

    /// The records the delta sets that it does not also remove: the ones commit keeps.
    fn kept_sets(&self) -> impl Iterator<Item = &'a Record> + '_ {
        self.set
            .values()
            .copied()
            .filter(|r| !self.removed.contains(&r.slot))
    }

    fn records(&self) -> impl Iterator<Item = &'a Record> + '_ {
        self.stream
            .records
            .values()
            .filter(|r| !self.set.contains_key(&r.slot))
            .chain(self.set.values().copied())
            .filter(|r| !self.removed.contains(&r.slot))
    }
}

fn missing_key(what: &str, key: &Key, slot: Option<u32>) -> StreamError {
    match slot {
        Some(slot) => StreamError::Missing(format!(
            "slot {slot} references {what} {} the stream would not hold",
            key.hex()
        )),
        None => StreamError::Missing(format!(
            "the view references {what} {} the stream would not hold",
            key.hex()
        )),
    }
}

fn shape_kind_name(data: &ShapeData) -> &'static str {
    match data {
        ShapeData::Path { .. } => "path",
        ShapeData::Mesh { .. } => "mesh",
        ShapeData::Image { .. } => "image",
        ShapeData::Reserved { .. } => "reserved",
    }
}

impl After<'_> {
    /// A shape the record needs, of one of the kinds it accepts.
    fn needs_shape(
        &self,
        rec: &Record,
        key: Option<&Key>,
        accepts: &[&str],
    ) -> Result<Option<&ShapeData>, StreamError> {
        let Some(key) = key else {
            return Err(StreamError::Invalid(format!(
                "a {} record (slot {}) needs a shape",
                rec.kind.name(),
                rec.slot
            )));
        };
        let data = self
            .shape(key)
            .ok_or_else(|| missing_key("shape", key, Some(rec.slot)))?;
        if !accepts.is_empty() && !accepts.contains(&shape_kind_name(data)) {
            return Err(StreamError::Invalid(format!(
                "a {} record (slot {}) cannot take a {} shape",
                rec.kind.name(),
                rec.slot,
                shape_kind_name(data)
            )));
        }
        Ok(Some(data))
    }

    fn needs_table(&self, rec: &Record, key: &Key, rows: usize) -> Result<(), StreamError> {
        let stops = self
            .paint(key)
            .ok_or_else(|| missing_key("paint table", key, Some(rec.slot)))?;
        if stops.rows().len() != rows {
            return Err(StreamError::Invalid(format!(
                "slot {}: a table of {} rows where {rows} are needed",
                rec.slot,
                stops.rows().len()
            )));
        }
        Ok(())
    }

    fn check_view(&self, view: &View, slot: Option<u32>) -> Result<(), StreamError> {
        if let LightModel::SoftDiffuse { ambient, wrap } = view.light
            && !((0.0..=1.0).contains(&ambient) && wrap.is_finite() && wrap >= 0.0)
        {
            return Err(StreamError::Invalid(format!(
                "soft diffuse light needs an ambient share in 0..=1 and a wrap of at least 0, \
                 not {ambient} and {wrap}"
            )));
        }
        if let Background::Image(key) = &view.background {
            match self.shape(key) {
                Some(ShapeData::Image { .. }) => {}
                Some(_) => {
                    return Err(StreamError::Invalid(
                        "a background image must be an image shape".into(),
                    ));
                }
                None => return Err(missing_key("background image", key, slot)),
            }
        }
        Ok(())
    }

    /// Checks an order against the records the delta leaves. Displays in each layer capture in
    /// rank order before its entries draw; self capture reads the picture as it then stands.
    fn check_order(&self, order: &[OrderEntry]) -> Result<(), StreamError> {
        let mut ranks = HashMap::new();
        for e in order {
            let rec = self.record(e.slot).ok_or_else(|| {
                StreamError::Missing(format!("the order draws slot {}, which is not set", e.slot))
            })?;
            if let Body::Image(ImageSource::SubFrame { rank, .. }) = &rec.body
                && let Some(other) = ranks.insert((e.layer as u8, *rank), e.slot)
                && other != e.slot
            {
                return Err(StreamError::Invalid(format!(
                    "display slots {other} and {} share capture rank {rank} in layer {}",
                    e.slot, e.layer as u8
                )));
            }
            let parts = match (&rec.kind, &rec.body) {
                (RecordKind::Unknown(_), _) => continue,
                (_, Body::FaceGroup(f)) => f.uv.len(),
                _ => 1,
            };
            if e.part as usize >= parts {
                return Err(StreamError::Invalid(format!(
                    "the order draws part {} of slot {}, which has {parts}",
                    e.part, e.slot
                )));
            }
        }
        Ok(())
    }

    fn check_record(&self, rec: &Record) -> Result<(), StreamError> {
        if let RecordKind::Unknown(_) = rec.kind {
            return Ok(()); // Nothing in it can be read; drawing refuses it.
        }
        if (rec.features & FEATURE_BACKGROUND_IMAGE != 0) != rec.fill_image.is_some() {
            return Err(StreamError::Invalid(
                "background_image feature bit and fill_image field disagree".into(),
            ));
        }
        if let Some(key) = &rec.fill_image {
            self.needs_shape(rec, Some(key), &["image"])?;
        }
        if (rec.features & FEATURE_PRE_PROJECTED != 0) != rec.pre_projected.is_some() {
            return Err(StreamError::Invalid(
                "pre_projected feature bit and tail field disagree".into(),
            ));
        }
        if let Some(p) = &rec.pre_projected {
            if !matches!(
                rec.kind,
                RecordKind::Path | RecordKind::Points | RecordKind::Image
            ) {
                return Err(StreamError::Invalid(format!(
                    "pre_projected is not valid on {} records",
                    rec.kind.name()
                )));
            }
            self.needs_shape(rec, Some(&p.points), &["path"])?;
        }
        let g = &rec.geometry;
        let shape_a = g.shape_a.as_ref();
        let a = match (rec.kind, &rec.body) {
            (RecordKind::Path, Body::None) => self.needs_shape(rec, shape_a, &["path"])?,
            (RecordKind::Mesh, Body::None) => self.needs_shape(rec, shape_a, &["mesh"])?,
            (RecordKind::FaceGroup, Body::FaceGroup(_)) => {
                self.needs_shape(rec, shape_a, &["path"])?
            }
            (RecordKind::Points, Body::Points(_)) => self.needs_shape(rec, shape_a, &["path"])?,
            (RecordKind::Instances, Body::Instances(_)) => {
                self.needs_shape(rec, shape_a, &["path", "mesh"])?
            }
            (RecordKind::Image, Body::Image(ImageSource::Picture { .. })) => {
                self.needs_shape(rec, shape_a, &["image"])?
            }
            (RecordKind::Image, Body::Image(ImageSource::SubFrame { view, order, .. })) => {
                if shape_a.is_some() {
                    return Err(StreamError::Invalid(format!(
                        "slot {}: a sub-frame display has no shape",
                        rec.slot
                    )));
                }
                self.check_view(view, Some(rec.slot))?;
                self.check_order(order)?;
                None
            }
            (RecordKind::Volume | RecordKind::Field | RecordKind::Emission, Body::Opaque(_)) => {
                match shape_a {
                    Some(_) => self.needs_shape(rec, shape_a, &[])?,
                    None => None,
                }
            }
            (kind, _) => {
                return Err(StreamError::Invalid(format!(
                    "slot {}: a {} record with another kind's body",
                    rec.slot,
                    kind.name()
                )));
            }
        };
        if let Some(b) = &g.blend {
            let accepts = match a {
                Some(data) => vec![shape_kind_name(data)],
                None => Vec::new(),
            };
            self.needs_shape(rec, Some(&b.shape_b), &accepts)?;
        }
        if let Some((clip, _)) = &rec.clip {
            self.needs_shape(rec, Some(clip), &["path"])?;
        }
        for key in rec.paint_keys() {
            if self.paint(&key).is_none() {
                return Err(missing_key("paint table", &key, Some(rec.slot)));
            }
        }
        let points = match a {
            Some(ShapeData::Path { points }) => points.len(),
            _ => 0,
        };
        match &rec.body {
            Body::FaceGroup(f) => {
                self.needs_table(rec, &f.fill, f.uv.len())?;
                if let Some(stroke) = &f.stroke {
                    self.needs_table(rec, stroke, f.uv.len())?;
                }
            }
            Body::Points(p) => self.needs_table(rec, &p.colors, points)?,
            Body::Instances(i) => {
                let n = i.placements.len();
                if i.objects.len() != n {
                    return Err(StreamError::Invalid(format!(
                        "slot {}: {n} placements and {} objects",
                        rec.slot,
                        i.objects.len()
                    )));
                }
                self.needs_table(rec, &i.fill, n)?;
                self.needs_table(rec, &i.stroke, n)?;
            }
            Body::None | Body::Image(_) | Body::Opaque(_) => {}
        }
        Ok(())
    }
}

impl Stream {
    pub fn new() -> Stream {
        Stream::default()
    }

    /// Applies a delta: the view, objects, uploads, records set, records removed, keys released,
    /// then the order. Everything is checked first; a refused delta changes nothing.
    pub fn apply(&mut self, delta: &Delta) -> Result<ApplyStats, StreamError> {
        if delta.version.0 != XF_VERSION.0 {
            return Err(StreamError::Version(format!(
                "XF {}.{} cannot be drawn by this build (XF {}.{})",
                delta.version.0, delta.version.1, XF_VERSION.0, XF_VERSION.1
            )));
        }
        self.check(delta)?;

        if let Some(view) = &delta.view {
            self.view = Some(view.clone());
        }
        for o in &delta.objects {
            self.objects.insert(o.object, o.clone());
        }
        for s in &delta.upload {
            self.shapes.entry(s.key()).or_insert_with(|| s.clone());
        }
        for s in &delta.upload_paints {
            self.paints.entry(s.key()).or_insert_with(|| s.clone());
        }
        for r in &delta.set {
            self.records.insert(r.slot, r.clone());
        }
        for slot in &delta.remove {
            self.records.remove(slot);
        }
        for key in &delta.release {
            self.shapes.remove(key);
            self.paints.remove(key);
        }
        if let Some(order) = &delta.order {
            self.order = order.clone();
        }
        Ok(ApplyStats {
            uploaded_shapes: delta.upload.len(),
            changed_records: delta.set.len() + delta.remove.len(),
        })
    }

    fn check(&self, delta: &Delta) -> Result<(), StreamError> {
        let mut uploaded = HashSet::new();
        for key in delta
            .upload
            .iter()
            .map(|s| s.key())
            .chain(delta.upload_paints.iter().map(|s| s.key()))
        {
            if !uploaded.insert(key) {
                return Err(StreamError::Invalid(format!(
                    "a delta uploads {} twice",
                    key.hex()
                )));
            }
        }
        let held_with_other = |key: Key| {
            Err(StreamError::Invalid(format!(
                "an upload of {}, which the stream holds with other content",
                key.hex()
            )))
        };
        for shape in &delta.upload {
            check_shape(shape)?;
            let held = self.shapes.get(&shape.key());
            if self.paints.contains_key(&shape.key())
                || held.is_some_and(|h| !same_content(h.data(), shape.data()))
            {
                return held_with_other(shape.key());
            }
        }
        for stops in &delta.upload_paints {
            if self.shapes.contains_key(&stops.key()) {
                return held_with_other(stops.key());
            }
        }
        let after = After::new(self, delta);
        for key in &delta.release {
            let held = self.shapes.contains_key(key)
                || self.paints.contains_key(key)
                || after.uploads.contains_key(key)
                || after.upload_paints.contains_key(key);
            if !held {
                return Err(StreamError::Missing(format!(
                    "a release of {}, which the stream does not hold",
                    key.hex()
                )));
            }
        }
        for slot in &delta.remove {
            if !self.records.contains_key(slot) && !after.set.contains_key(slot) {
                return Err(StreamError::Missing(format!(
                    "a removal of slot {slot}, which is not set"
                )));
            }
        }
        let view = delta.view.as_ref().or(self.view.as_ref());
        if !after.released.is_empty() {
            let in_use = |key: &Key, by: String| {
                if after.released.contains(key) {
                    Err(StreamError::InUse(format!(
                        "a release of {}, which {by} still uses",
                        key.hex()
                    )))
                } else {
                    Ok(())
                }
            };
            for rec in after.records() {
                for key in rec.shape_keys().iter().chain(&rec.paint_keys()) {
                    in_use(key, format!("slot {}", rec.slot))?;
                }
            }
            if let Some(View {
                background: Background::Image(key),
                ..
            }) = view
            {
                in_use(key, "the view".into())?;
            }
        }
        // Untouched records were checked when set; only a release (above), or a removal or a re-set
        // of a slot a display's sub-frame draws, can strand them.
        if !after.removed.is_empty() || !after.set.is_empty() {
            for rec in after.records() {
                if let Body::Image(ImageSource::SubFrame { order, .. }) = &rec.body {
                    after.check_order(order)?;
                }
            }
        }
        for rec in after.kept_sets() {
            after.check_record(rec)?;
        }
        if let Some(view) = &delta.view {
            after.check_view(view, None)?;
        }
        after.check_order(delta.order.as_deref().unwrap_or(&self.order))
    }

    /// Draws the frame at `width x height` pixels, its tiles spread over `threads` threads.
    pub fn draw(&self, width: u32, height: u32, threads: usize) -> Result<Vec<u8>, DrawError> {
        Ok(self.render(width, height, threads)?.rgba)
    }

    /// Draws the frame, and reports how each pixel was resolved.
    ///
    /// This build draws the background (the view's colour, or its image cropped as CE crops it)
    /// and `mesh`, `path`, `face_group`, `points` and `instances` records. It refuses, whole, any
    /// frame whose order draws what it cannot: `image` records (Cairo's overlay draws them), a
    /// record carrying a field it does not draw yet, reserved or unknown kinds, unknown feature
    /// bits, and unknown light models.
    pub fn render(&self, width: u32, height: u32, threads: usize) -> Result<Rendered, DrawError> {
        self.render_with(width, height, threads, &[])
    }

    /// Draws the frame as [`Stream::render`] does, without the overlay's entries
    /// ([`is_overlay`]): a 3D scene's frame before Cairo draws its overlay over it, in CE's order.
    /// Those entries are neither drawn nor refused here; the others keep their places in the
    /// order, so ties break as in the whole order.
    pub fn render_under_overlay(
        &self,
        width: u32,
        height: u32,
        threads: usize,
    ) -> Result<Rendered, DrawError> {
        self.render_entries(width, height, threads, &[], true)
    }

    /// The order's entries in the overlay ([`is_overlay`]), in order.
    pub fn overlay_entries(&self) -> Vec<OrderEntry> {
        self.order
            .iter()
            .filter(|e| is_overlay(&self.records[&e.slot]))
            .copied()
            .collect()
    }

    /// Draws the frame as [`Stream::render`] does, with `extra` producers drawing into the same
    /// passes after the stream's own: a probe's instrument (a producer that sends every pixel to
    /// the sample resolve, say), never part of what a stream draws.
    pub fn render_with(
        &self,
        width: u32,
        height: u32,
        threads: usize,
        extra: &[&dyn Producer],
    ) -> Result<Rendered, DrawError> {
        self.render_entries(width, height, threads, extra, false)
    }

    fn render_entries(
        &self,
        width: u32,
        height: u32,
        threads: usize,
        extra: &[&dyn Producer],
        without_overlay: bool,
    ) -> Result<Rendered, DrawError> {
        check_size(width, height)?;
        if threads == 0 {
            return Err(DrawError::Invalid(
                "drawing needs at least one thread".into(),
            ));
        }
        self.with_producers(width, height, extra, without_overlay, |frame, producers| {
            tiles::render(frame, producers, threads).map_err(|e| DrawError::Threads(e.to_string()))
        })?
    }

    /// The leaves the core composites at each pixel of `x0..x1` by `y0..y1` (clipped to the
    /// view's frame) that holds a fragment, from the frame [`Stream::render_under_overlay`] draws:
    /// what a pixel's colour is made of, read from the same passes the resolves read
    /// (`contributors.rs`). The overlay's entries are Cairo's and are not reported. Refused as
    /// drawing refuses.
    pub fn contributors(
        &self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
    ) -> Result<BTreeMap<(u32, u32), crate::contributors::PixelContributors>, DrawError> {
        let mut out = BTreeMap::new();
        let leaf = |f: &composite::Fragment| self.leaf_of(f);
        self.visit_contributors(x0, y0, x1, y1, |x, y, frags, coverage, depths, _| {
            out.insert(
                (x, y),
                crate::contributors::of_pixel(frags, coverage, depths, &leaf),
            );
        })?;
        Ok(out)
    }

    /// The edges the rasterizer consumes for every leaf [`Stream::contributors`] reports with
    /// an exact area in the same box, from the same pass: per leaf `(object, part)`, each of its
    /// layers once (a fill, a stroke's outline after the stroker), as `[x0, y0, x1, y1, dir]` in
    /// device pixels (`scanline::Edge`: oriented downward, `dir` the winding it adds). A leaf
    /// drawn by masks (a mesh's triangles, a cloud's blocks) has none.
    #[allow(clippy::type_complexity)]
    pub fn contributor_edges(
        &self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
    ) -> Result<BTreeMap<(u32, u32), Vec<Vec<[f64; 5]>>>, DrawError> {
        let mut out: BTreeMap<(u32, u32), Vec<Vec<[f64; 5]>>> = BTreeMap::new();
        let mut seen = std::collections::BTreeSet::new();
        self.visit_contributors(x0, y0, x1, y1, |_, _, frags, _, _, producers| {
            for f in frags {
                if let composite::Coverage::Area { id, .. } = f.coverage
                    && seen.insert((f.source(), id))
                    && let Some(edges) = producers[f.source() as usize].edges(id)
                {
                    out.entry(self.leaf_of(f)).or_default().push(
                        edges
                            .iter()
                            .map(|e| [e.x0, e.y0, e.x1, e.y1, e.dir as f64])
                            .collect(),
                    );
                }
            }
        })?;
        Ok(out)
    }

    /// For each pixel of `x0..x1` by `y0..y1` [`Stream::contributors`] reports, from the same
    /// pass, each fragment there of a stroke (or background stroke) of a path with no plane of its
    /// own, by its leaf `(object, part)`, with its own share of its key at that pixel
    /// ([`composite::Fragment::own`]): what [`order::lines`] reads.
    #[allow(clippy::type_complexity)]
    pub fn own_shares(
        &self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
    ) -> Result<BTreeMap<(u32, u32), Vec<((u32, u32), f64)>>, DrawError> {
        let mut out: BTreeMap<(u32, u32), Vec<((u32, u32), f64)>> = BTreeMap::new();
        self.visit_contributors(x0, y0, x1, y1, |x, y, frags, _, _, _| {
            for f in frags.iter().filter(|f| f.own >= 0.0) {
                out.entry((x, y))
                    .or_default()
                    .push((self.leaf_of(f), f.own));
            }
        })?;
        Ok(out)
    }

    /// Visits each pixel of `x0..x1` by `y0..y1` (clipped to the view's frame) that holds a
    /// fragment, with its fragments, their coverage at each sub-sample, their depths and the
    /// producers that added them, from the frame [`Stream::render_under_overlay`] draws: the one pass the
    /// contributor reports read.
    #[allow(clippy::type_complexity)]
    fn visit_contributors(
        &self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
        mut visit: impl FnMut(
            u32,
            u32,
            &[composite::Fragment],
            &[[u8; composite::SAMPLES]],
            &[composite::Depths],
            &[&dyn Producer],
        ),
    ) -> Result<(), DrawError> {
        let view = self.view.as_ref().ok_or(DrawError::NoView)?;
        let (width, height) = (view.pixel_width, view.pixel_height);
        let keep = composite::Rect {
            x0: x0.min(width),
            y0: y0.min(height),
            x1: x1.min(width),
            y1: y1.min(height),
        };
        self.with_producers(width, height, &[], true, |_, producers| {
            if keep.x0 >= keep.x1 || keep.y0 >= keep.y1 {
                return;
            }
            for rect in tiles::tile_rects(width, height) {
                if rect.x1 <= keep.x0
                    || rect.x0 >= keep.x1
                    || rect.y1 <= keep.y0
                    || rect.y0 >= keep.y1
                {
                    continue;
                }
                tiles::passes(rect, producers)
                    .visit_pixels(keep, |x, y, frags, coverage, depths| {
                        visit(x, y, frags, coverage, depths, producers)
                    });
            }
        })
    }

    /// The leaf a fragment drawn from this stream belongs to, as `(object, part)`: the object of
    /// the record its order entry draws and the entry's part (a face of a face group), except
    /// that an `instances` record's member reports its own object (instance `i` of a path
    /// prototype draws sub-orders `3i..3i + 3`, of a mesh prototype `t * i..t * (i + 1)` for its
    /// `t` triangles: `instance.rs`).
    fn leaf_of(&self, f: &composite::Fragment) -> (u32, u32) {
        let entry = &self.order[f.order as usize];
        let record = &self.records[&entry.slot];
        if let Body::Instances(body) = &record.body {
            let per = match record
                .geometry
                .shape_a
                .and_then(|k| self.shapes.get(&k))
                .map(|s| s.data())
            {
                Some(ShapeData::Mesh { triangles, .. }) => triangles.len().max(1) as u32,
                _ => 3,
            };
            let i = (f.sub / per) as usize;
            return (body.objects[i], 0);
        }
        (record.object, entry.part)
    }

    /// The frame's background and its producers, lowered from the records the order draws (all,
    /// or all but the overlay's), handed to `f`. Refused as [`Stream::render`] refuses.
    fn with_producers<T>(
        &self,
        width: u32,
        height: u32,
        extra: &[&dyn Producer],
        without_overlay: bool,
        f: impl FnOnce(Frame, &[&dyn Producer]) -> T,
    ) -> Result<T, DrawError> {
        let drawn = |e: &OrderEntry| !(without_overlay && is_overlay(&self.records[&e.slot]));
        check_size(width, height)?;
        let view = self.view.as_ref().ok_or(DrawError::NoView)?;
        if let LightModel::Other { code, .. } = &view.light {
            return Err(DrawError::Unsupported(Unsupported {
                kind: "view".into(),
                slot: None,
                object: None,
                reason: format!("light model {code} is not one this build knows"),
            }));
        }
        let shading = matches!(&view.camera, Camera::ThreeD(c) if c.should_apply_shading);
        let ce_light = view.light == LightModel::Ce;
        if let Some(refused) = self.order.iter().filter(|e| drawn(e)).find_map(|e| {
            refusal(
                &self.records[&e.slot],
                &self.shapes,
                &self.paints,
                shading,
                ce_light,
            )
        }) {
            return Err(DrawError::Unsupported(refused));
        }
        let frame = match &view.background {
            Background::Color(c) => Frame::filled(width, height, composite::background_pixel(*c)),
            Background::Image(key) => {
                let Some(ShapeData::Image {
                    width: iw,
                    height: ih,
                    rgba,
                }) = self.shapes.get(key).map(|s| s.data())
                else {
                    unreachable!("apply checked the background image");
                };
                Frame::cropped(width, height, *iw, *ih, rgba).ok_or_else(|| {
                    DrawError::Invalid(format!(
                        "the background image ({iw} x {ih}) is smaller than the frame \
                         ({width} x {height})"
                    ))
                })?
            }
        };
        let (mut meshes, mut paths, mut clouds) = (Vec::new(), Vec::new(), Vec::new());
        for (order, e) in self.order.iter().enumerate() {
            if !drawn(e) {
                continue;
            }
            let record = &self.records[&e.slot];
            let order = order as u32;
            let shape = record
                .geometry
                .shape_a
                .and_then(|k| self.shapes.get(&k))
                .map(|s| s.data());
            let table = |key: &Key| &self.paints[key];
            match (record.kind, shape, &record.body) {
                (
                    RecordKind::Mesh,
                    Some(ShapeData::Mesh {
                        positions,
                        triangles,
                        normals,
                    }),
                    _,
                ) => {
                    if let Some(fill) = instance::one(&record.paint.fill) {
                        meshes.push(MeshDraw {
                            record,
                            positions,
                            triangles,
                            normals: normals.as_deref(),
                            place: record.geometry.place_a,
                            places: [crate::xf::IDENTITY, record.geometry.place_a],
                            fill,
                            order,
                            sub: 0,
                        });
                    }
                }
                (RecordKind::FaceGroup, Some(ShapeData::Path { points }), Body::FaceGroup(f)) => {
                    paths.push(instance::face(record, points, f, e.part, table, order));
                }
                (RecordKind::Path, Some(ShapeData::Path { points: own }), _)
                    if let Some(tail) = &record.pre_projected =>
                {
                    let placed = path_shape(&self.shapes, &tail.points)
                        .expect("apply checked the tail's points");
                    paths.push(instance::produced_path(
                        record, own, placed, tail, table, order,
                    ));
                }
                (RecordKind::Path, Some(ShapeData::Path { .. }), _) => {
                    let (points, place) =
                        drawn_points(record, &self.shapes).expect("drawing refused the others");
                    let paint = geometry::drawn_paint(&record.paint, table)
                        .expect("drawing refused a paint it cannot mix");
                    paths.push(instance::path_record(record, points, place, paint, order));
                }
                (RecordKind::Points, Some(ShapeData::Path { points }), Body::Points(p)) => {
                    let positions = match &record.pre_projected {
                        Some(tail) => path_shape(&self.shapes, &tail.points)
                            .expect("apply checked the tail's points"),
                        None => points,
                    };
                    clouds.push(instance::points(
                        record,
                        points,
                        positions,
                        table(&p.colors),
                        order,
                    ));
                }
                (RecordKind::Instances, Some(shape), Body::Instances(body)) => match shape {
                    ShapeData::Path { points } => instance::path_members(
                        record,
                        points,
                        body,
                        table(&body.fill),
                        table(&body.stroke),
                        order,
                        &mut paths,
                    ),
                    ShapeData::Mesh {
                        positions,
                        triangles,
                        normals,
                    } => instance::mesh_members(
                        record,
                        positions,
                        triangles,
                        normals.as_deref(),
                        body,
                        table(&body.fill),
                        order,
                        &mut meshes,
                    ),
                    _ => unreachable!("apply checked the instances record's shape"),
                },
                _ => unreachable!("drawing refused every other record"),
            }
        }
        // The meshes' triangles and the paths are offered to one registry, which settles once all
        // are in (`Lowering::coplanar`); then the triangles take their identities.
        let mut registry = crate::depth::Registry::new(view);
        let mut meshes = mesh::lower(view, width, height, &meshes, &mut registry);
        let mut lowering = path::Lowering::new(view, width, height);
        let planes = lowering.coplanar(&paths, &mut registry);
        // Settled, it holds the flat records the producers' fragments carry.
        let registry = std::sync::Arc::new(registry);
        meshes.identify(&registry);
        for (draw, plane) in paths.iter().zip(planes) {
            lowering.path(draw, plane);
        }
        let paths = lowering.finish(&registry);
        let blocks = points::lower(view, width, height, &clouds);
        let mut producers: Vec<&dyn Producer> = vec![&meshes, &paths, &blocks];
        producers.extend_from_slice(extra);
        Ok(f(frame, &producers))
    }

    /// Per slot of `slots`, how each path its record draws lies, decided from the points the
    /// stream holds ([`path::Lowering::planarity`], codes of [`crate::depth::Planarity`]), in the
    /// order of the record's `planes`: what a writer that snaps the record's geometry afterwards
    /// carries with it. A slot holding no record that draws paths (or one the drawer would refuse)
    /// has no entry. Nothing is drawn.
    pub fn planes(&self, slots: &[u32]) -> Result<BTreeMap<u32, Vec<u8>>, DrawError> {
        self.each_paths(slots, |lowering, paths| {
            Some(paths.iter().map(|d| lowering.planarity(d).code()).collect())
        })
    }

    /// Per slot of `slots` holding a record the camera shades, the normals each path it draws is
    /// lit by at its two lit corners ([`path::Lowering::normals`]), decided from the points the
    /// stream holds as a writer that snaps them to `grid` carries them: two per path, in the order
    /// of the record's `planes`. Other slots have no entry. Nothing is drawn.
    pub fn normals(
        &self,
        slots: &[u32],
        grid: f64,
    ) -> Result<BTreeMap<u32, Vec<[f64; 3]>>, DrawError> {
        self.each_paths(slots, |lowering, paths| {
            let mut found = Vec::new();
            for draw in paths {
                found.extend(lowering.normals(draw, grid)?);
            }
            Some(found)
        })
    }

    /// Per slot of `slots` holding a record the camera shades, which of the two lit corners of each
    /// path it draws the drawer lights by the face's Newell normal ([`path::Lowering::newell_corners`]):
    /// two per path, in the order of the record's `planes`. Other slots have no entry. Nothing is
    /// drawn.
    pub fn newell_corners(
        &self,
        slots: &[u32],
    ) -> Result<BTreeMap<u32, Vec<[bool; 2]>>, DrawError> {
        self.each_paths(slots, |lowering, paths| {
            let mut found = Vec::new();
            for draw in paths {
                found.push(lowering.newell_corners(draw)?);
            }
            Some(found)
        })
    }

    /// Per slot of `slots` holding a record the camera shades, the two stops each layer of each
    /// path it draws is painted with ([`path::Lowering::face_stops`]: background stroke, fill and
    /// stroke, lit at the path's lit corners, with the record's group opacity), one entry per path
    /// in the order of the record's `planes`, `None` for a path the drawer draws nothing of. Other
    /// slots (unshaded, pre-projected, refused) have no entry. Nothing is drawn.
    pub fn face_stops(
        &self,
        slots: &[u32],
    ) -> Result<BTreeMap<u32, Vec<Option<path::FaceStops>>>, DrawError> {
        self.each_paths(slots, |lowering, paths| {
            let found: Vec<_> = paths.iter().map(|d| lowering.face_stops(d)).collect();
            found.iter().any(Option::is_some).then_some(found)
        })
    }

    /// `answer(lowering, paths)` per slot of `slots` holding a record that draws paths the drawer
    /// would not refuse, `paths` those it draws, in order; a `None` answer has no entry.
    fn each_paths<T>(
        &self,
        slots: &[u32],
        answer: impl Fn(&path::Lowering<'_>, &[path::PathDraw<'_>]) -> Option<T>,
    ) -> Result<BTreeMap<u32, T>, DrawError> {
        let view = self.view.as_ref().ok_or(DrawError::NoView)?;
        let lowering = path::Lowering::new(view, view.pixel_width, view.pixel_height);
        let shading = matches!(&view.camera, Camera::ThreeD(c) if c.should_apply_shading);
        let ce_light = view.light == LightModel::Ce;
        let mut found = BTreeMap::new();
        for slot in slots {
            let Some(record) = self.records.get(slot) else {
                continue;
            };
            if refusal(record, &self.shapes, &self.paints, shading, ce_light).is_some() {
                continue;
            }
            let shape = record
                .geometry
                .shape_a
                .and_then(|k| self.shapes.get(&k))
                .map(|s| s.data());
            let table = |key: &Key| &self.paints[key];
            let mut paths = Vec::new();
            match (record.kind, shape, &record.body) {
                (RecordKind::FaceGroup, Some(ShapeData::Path { points }), Body::FaceGroup(f)) => {
                    for part in 0..f.uv.len() as u32 {
                        paths.push(instance::face(record, points, f, part, table, 0));
                    }
                }
                (RecordKind::Path, Some(ShapeData::Path { points: own }), _)
                    if let Some(tail) = &record.pre_projected =>
                {
                    if let Some(placed) = path_shape(&self.shapes, &tail.points) {
                        paths.push(instance::produced_path(record, own, placed, tail, table, 0));
                    }
                }
                (RecordKind::Path, Some(ShapeData::Path { .. }), _) => {
                    if let (Some((points, place)), Ok(paint)) = (
                        drawn_points(record, &self.shapes),
                        geometry::drawn_paint(&record.paint, table),
                    ) {
                        paths.push(instance::path_record(record, points, place, paint, 0));
                    }
                }
                (
                    RecordKind::Instances,
                    Some(ShapeData::Path { points }),
                    Body::Instances(body),
                ) => instance::path_members(
                    record,
                    points,
                    body,
                    table(&body.fill),
                    table(&body.stroke),
                    0,
                    &mut paths,
                ),
                _ => continue,
            }
            if let Some(answer) = answer(&lowering, &paths) {
                found.insert(*slot, answer);
            }
        }
        Ok(found)
    }

    /// Per slot the order draws, the pixel box of its record's points as the camera places them
    /// for display and maps them to pixels by the mapping its kind is drawn through
    /// ([`crate::boxes::record_box`]); a record placing no point has no entry. Nothing is drawn,
    /// so this answers for records the drawer would refuse too.
    pub fn screen_boxes(&self) -> Result<BTreeMap<u32, crate::boxes::PixelBox>, DrawError> {
        let view = self.view.as_ref().ok_or(DrawError::NoView)?;
        let mut boxes = BTreeMap::new();
        for entry in &self.order {
            if boxes.contains_key(&entry.slot) {
                continue;
            }
            let record = &self.records[&entry.slot];
            let found = crate::boxes::record_box(view, record, |key| {
                self.shapes.get(key).map(|s| s.data())
            });
            if let Some(b) = found {
                boxes.insert(entry.slot, b);
            }
        }
        Ok(boxes)
    }

    pub fn view(&self) -> Option<&View> {
        self.view.as_ref()
    }

    pub fn record(&self, slot: u32) -> Option<&Record> {
        self.records.get(&slot)
    }

    pub fn order(&self) -> &[OrderEntry] {
        &self.order
    }

    pub fn object(&self, object: u32) -> Option<&ObjectInfo> {
        self.objects.get(&object)
    }

    pub fn has_shape(&self, key: &Key) -> bool {
        self.shapes.contains_key(key)
    }
}

/// Refuses a frame size the drawer does not draw.
fn check_size(width: u32, height: u32) -> Result<(), DrawError> {
    if width == 0 || height == 0 || width > MAX_SIDE || height > MAX_SIDE {
        return Err(DrawError::Invalid(format!(
            "a frame of {width} x {height} pixels (each side 1 to {MAX_SIDE})"
        )));
    }
    Ok(())
}

/// Whether a shape held under a key and one uploaded under it have one content. A key is computed
/// from, or checked against, the contents of every kind this build knows, so two such shapes of one
/// kind are the same; a kind it does not know carries the key it was stated with, so its bytes are
/// compared.
fn same_content(held: &ShapeData, uploaded: &ShapeData) -> bool {
    match (held, uploaded) {
        (
            ShapeData::Reserved { code, payload },
            ShapeData::Reserved {
                code: other,
                payload: other_payload,
            },
        ) => code == other && payload == other_payload,
        (ShapeData::Reserved { .. }, _) | (_, ShapeData::Reserved { .. }) => false,
        (a, b) => std::mem::discriminant(a) == std::mem::discriminant(b),
    }
}

/// The longest side of a frame this build draws, in pixels.
pub const MAX_SIDE: u32 = 16384;

/// A shape that contradicts itself: a triangle naming a vertex the mesh lacks, normals that are not
/// one per vertex, or an image whose pixels are not its width times its height.
fn check_shape(shape: &Shape) -> Result<(), StreamError> {
    let invalid = |why: String| {
        Err(StreamError::Invalid(format!(
            "{}: {why}",
            shape.key().hex()
        )))
    };
    match shape.data() {
        ShapeData::Mesh {
            positions,
            triangles,
            normals,
        } => {
            if triangles
                .iter()
                .flatten()
                .any(|i| *i as usize >= positions.len())
            {
                return invalid(format!(
                    "a mesh's triangle names a vertex past its {}",
                    positions.len()
                ));
            }
            if normals.as_ref().is_some_and(|n| n.len() != positions.len()) {
                return invalid("a mesh needs one normal per vertex".into());
            }
        }
        ShapeData::Image {
            width,
            height,
            rgba,
        } => {
            let size = (*width as usize)
                .checked_mul(*height as usize)
                .and_then(|n| n.checked_mul(4));
            if size != Some(rgba.len()) {
                return invalid(format!(
                    "an image of {width} x {height} pixels in {} bytes",
                    rgba.len()
                ));
            }
        }
        ShapeData::Path { .. } | ShapeData::Reserved { .. } => {}
    }
    Ok(())
}

/// What of a drawn record's fixed fields this build does not draw yet, if anything, its paint
/// tables in `paints`: blends, reveals and paint blends of any kind but a path, a paint blend of
/// row counts numpy does not broadcast (`geometry::drawn_paint`), clips for every kind; for meshes (and instances of a mesh) strokes, gradient fills and dashes; for paths (and
/// instances of a path) CE's lighting under a camera that shades (`shading`) with a light model
/// other than CE's (`ce_light`), a gradient paint without its ends or on an instance, and dashes
/// it cannot read. A pre-projected record is judged by its tail's paints and ends (lit already, so
/// never refused for shading), and refused with a blend, reveal or paint blend.
fn field_refusal(
    rec: &Record,
    paints: &HashMap<Key, Stops>,
    mesh: bool,
    shading: bool,
    ce_light: bool,
) -> Option<String> {
    let kind = rec.kind.name();
    // A pre-projected record's paints are the colours its camera painted with, lit already, and
    // its gradient runs between the ends that camera placed.
    let tail = rec.pre_projected.as_ref();
    let shaded = tail.is_none() && shading && rec.depth.shade_in_3d;
    // A path's blend, reveal and paint blend are drawn as what they make of it (`geometry`); the
    // paint judged below is the one drawn.
    let shaped = rec.geometry.blend.is_some()
        || rec.geometry.reveal.is_some()
        || rec.paint.paint_b.is_some();
    // Tables are checked to exist before drawing.
    let table = |key: &Key| &paints[key];
    let mut p = match geometry::drawn_paint(&rec.paint, table) {
        Ok(p) => p,
        Err(what) => return Some(format!("a {kind}'s {what} is not drawn")),
    };
    let mut ends = rec.paint.gradient;
    if let Some(t) = tail {
        p.fill = geometry::rows(&t.fill, table);
        p.stroke = geometry::rows(&t.stroke, table);
        p.background_stroke = geometry::rows(&t.background_stroke, table);
        ends = t.gradient;
    }
    let p = &p;
    type Rows = Option<Vec<[f64; 4]>>;
    let drawn = |paint: &Rows, width: Option<f64>| paint.is_some() && width.is_none_or(|w| w > 0.0);
    let gradient = |paint: &Rows| paint.as_ref().is_some_and(|r| r.len() > 1);
    let strokes = [
        (&p.stroke, p.stroke_width),
        (&p.background_stroke, p.background_stroke_width),
    ];
    let not_yet = |what: &str| Some(format!("a {kind}'s {what} is not drawn yet"));
    if shaped && rec.kind != RecordKind::Path {
        // CE's blends and reveals are of paths alone; extraction states no other.
        not_yet(if rec.geometry.blend.is_some() {
            "blend"
        } else if rec.geometry.reveal.is_some() {
            "reveal"
        } else {
            "paint blend"
        })
    } else if shaped && tail.is_some() {
        // The tail is what its camera drew of the mobject as it stood; extraction never states
        // both.
        Some(format!(
            "a pre-projected {kind} with a blend, reveal or paint blend is not drawn"
        ))
    } else if rec.clip.is_some() {
        not_yet("clip")
    } else if rec.kind == RecordKind::Points {
        None
    } else if mesh {
        if strokes.iter().any(|(paint, w)| drawn(paint, Some(*w))) {
            not_yet("stroke")
        } else if gradient(&p.fill) {
            not_yet("gradient fill")
        } else if rec.dash.is_some() {
            not_yet("dash")
        } else {
            None
        }
    } else if shaded
        && !ce_light
        && (rec.kind == RecordKind::FaceGroup
            || drawn(&p.fill, None)
            || strokes.iter().any(|(paint, w)| drawn(paint, Some(*w))))
    {
        // CE lights each paint into two stops, at the path's start and end corners
        // (`ThreeDCamera.modified_rgbas`), drawn as a linear gradient; another light model has no
        // look for a path yet.
        not_yet("shading under a light model other than CE's")
    } else if (gradient(&p.fill)
        || strokes
            .iter()
            .any(|(paint, w)| drawn(paint, Some(*w)) && gradient(paint)))
        && (rec.kind == RecordKind::Instances || (!shaded && ends.is_none()))
    {
        // A path's gradient runs between its ends (or a shaded path's corners); an instance's
        // colours are rows of a table, one each.
        not_yet("gradient paint")
    } else if let Some(d) = rec.dash
        && !(d.period > 0.0
            && d.period.is_finite()
            && d.duty > 0.0
            && d.duty <= 1.0
            && d.phase.is_finite())
    {
        Some(format!(
            "a {kind}'s dash needs a finite period above 0, a duty in (0, 1] and a finite phase"
        ))
    } else {
        None
    }
}

/// Whether a 3D scene's frame leaves `rec` to Cairo's overlay, drawn over the core's frame in CE's
/// order: a record fixed in frame, an image (a display's included), or a path filled through a
/// background image.
pub fn is_overlay(rec: &Record) -> bool {
    rec.depth.space == Space::Overlay
        || rec.kind == RecordKind::Image
        || rec.features & FEATURE_BACKGROUND_IMAGE != 0
}

/// The points of the path shape `key` names, if it names one.
fn path_shape<'a>(shapes: &'a HashMap<Key, Shape>, key: &Key) -> Option<&'a [[f64; 3]]> {
    match shapes.get(key).map(|s| s.data()) {
        Some(ShapeData::Path { points }) => Some(points.as_slice()),
        _ => None,
    }
}

/// The points a `path` record draws and the placement they still take: its shape and its own
/// placement, or, under a reveal or a blend, the points those make of it
/// ([`geometry::path_points`]), placed already. `None` when it has no path shape, or blends two
/// shapes of different lengths.
fn drawn_points<'a>(
    rec: &Record,
    shapes: &'a HashMap<Key, Shape>,
) -> Option<(Cow<'a, [[f64; 3]]>, Affine)> {
    let path = |key: &Key| path_shape(shapes, key);
    let a = path(rec.geometry.shape_a.as_ref()?)?;
    let g = &rec.geometry;
    if g.blend.is_none() && g.reveal.is_none() {
        return Some((Cow::Borrowed(a), g.place_a));
    }
    let b = match &g.blend {
        Some(blend) => Some(path(&blend.shape_b)?),
        None => None,
    };
    Some((Cow::Owned(geometry::path_points(g, a, b)?), IDENTITY))
}

/// Why this build refuses to draw a record, if it does, under a camera that shades or not
/// (`shading`) and CE's light model or another (`ce_light`), its paint tables in `paints`. It draws
/// `mesh`, `path`, `face_group`, `points` and `instances` records, pre-projected ones included;
/// `image` records are Cairo's overlay's.
fn refusal(
    rec: &Record,
    shapes: &HashMap<Key, Shape>,
    paints: &HashMap<Key, Stops>,
    shading: bool,
    ce_light: bool,
) -> Option<Unsupported> {
    let prototype = rec
        .geometry
        .shape_a
        .and_then(|k| shapes.get(&k))
        .map(|s| s.data());
    let reason = if rec.features & !KNOWN_FEATURES != 0 {
        format!(
            "feature bits {:#x} are not ones this build knows",
            rec.features & !KNOWN_FEATURES
        )
    } else if rec.features & FEATURE_BACKGROUND_IMAGE != 0 {
        "background_image is drawn by Cairo only".into()
    } else if let RecordKind::Unknown(code) = rec.kind {
        format!("record kind {code} is not one this build knows")
    } else if rec.kind.is_reserved() {
        format!("{} is a reserved kind", rec.kind.name())
    } else if matches!(
        rec.kind,
        RecordKind::Mesh
            | RecordKind::Path
            | RecordKind::FaceGroup
            | RecordKind::Points
            | RecordKind::Instances
    ) {
        let mesh = matches!(prototype, Some(ShapeData::Mesh { .. }));
        if let Some(reason) = field_refusal(rec, paints, mesh, shading, ce_light) {
            reason
        } else if rec.kind == RecordKind::Path && drawn_points(rec, shapes).is_none() {
            "a path's blend of two shapes of different lengths".into()
        } else if let (Body::FaceGroup(faces), Some(ShapeData::Path { points })) =
            (&rec.body, prototype)
        {
            if !faces.uv.is_empty() && points.len() % faces.uv.len() == 0 {
                return None;
            }
            "a face group whose points do not divide into its faces".into()
        } else if let (Body::Instances(body), Some(data)) = (&rec.body, prototype) {
            // Each instance's parts take sub-orders of their own, in 32 bits.
            let per = match data {
                ShapeData::Mesh { triangles, .. } => triangles.len(),
                _ => 3,
            };
            if (body.placements.len() as u64) * (per as u64) <= u32::MAX as u64 {
                return None;
            }
            "an instances record of more parts than 2^32".into()
        } else if let (Body::Points(body), Some(ShapeData::Path { points: own })) =
            (&rec.body, prototype)
        {
            let positions = match &rec.pre_projected {
                Some(tail) => path_shape(shapes, &tail.points).unwrap_or(&[]),
                None => own,
            };
            if rec.pre_projected.is_some()
                && paints.get(&body.colors).map(|t| t.rows().len()) != Some(positions.len())
            {
                return Some(Unsupported {
                    kind: rec.kind.name(),
                    slot: Some(rec.slot),
                    object: Some(rec.object),
                    reason: "a pre-projected points record placing other than one point per \
                             colour"
                        .into(),
                });
            }
            // Each block pixel's write takes a sub-order of its own (its offset, then its point),
            // in 32 bits.
            let side = points::side(body.radius) as u128;
            let writes = side
                .checked_mul(side)
                .and_then(|s| s.checked_mul(positions.len() as u128));
            if writes.is_some_and(|w| w <= u32::MAX as u128 + 1) {
                return None;
            }
            "a points record of more block pixels than 2^32".into()
        } else {
            return None;
        }
    } else {
        format!("this build does not draw {} records yet", rec.kind.name())
    };
    Some(Unsupported {
        kind: rec.kind.name(),
        slot: Some(rec.slot),
        object: Some(rec.object),
        reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::order;

    /// A closed polygon's outline through `corners` in CE's cubic layout.
    fn ring(corners: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let mut out = Vec::new();
        for (i, a) in corners.iter().enumerate() {
            let b = corners[(i + 1) % corners.len()];
            let at = |t: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
            out.extend([*a, at(1.0 / 3.0), at(2.0 / 3.0), b]);
        }
        out
    }

    /// How many times, over every sub-sample of the frame three or more fragments cover, a path's
    /// fill ranks in front of its own stroke drawn after it, in the rank the sample resolve builds
    /// there.
    // The sub-sample `k` indexes each fragment's coverage and depth rows.
    #[allow(clippy::needless_range_loop)]
    fn own_cover(s: &Stream, width: u32, height: u32) -> usize {
        let mut hits = 0;
        let mut out = Vec::new();
        s.visit_contributors(0, 0, width, height, |_, _, frags, cov, d, _| {
            let mut d = d.to_vec();
            composite::strokes_on_own_fills(frags, &mut d, |i| cov[i]);
            for k in 0..composite::SAMPLES {
                let items: Vec<usize> = (0..frags.len()).filter(|i| cov[*i][k] != 0).collect();
                if items.len() < 3 {
                    continue;
                }
                let at = |a: usize, b: usize| {
                    let ((da, ea), (db, eb)) = (
                        d[a].against(&frags[a], &frags[b]),
                        d[b].against(&frags[b], &frags[a]),
                    );
                    (&frags[a], da[k], ea, &frags[b], db[k], eb)
                };
                order::rank_sample(
                    &items,
                    |a, b| {
                        let (fa, da, ea, fb, db, eb) = at(a, b);
                        a != b && order::at_sample(fa, da, ea, fb, db, eb).is_lt()
                    },
                    |a, b| {
                        let (fa, da, ea, fb, db, eb) = at(a, b);
                        order::strong_at(fa, da, ea, fb, db, eb)
                    },
                    |a, b| order::painted(&frags[a], &frags[b]),
                    &mut out,
                );
                for (p, &f) in out.iter().enumerate() {
                    let (fill, behind) = (&frags[f], &out[p + 1..]);
                    hits += behind
                        .iter()
                        .filter(|&&s| {
                            fill.sub % 3 == 1
                                && frags[s].sub > fill.sub
                                && order::own_layers(fill, &frags[s])
                        })
                        .count();
                }
            }
        })
        .expect("the stream draws");
        hits
    }

    /// A small saddle surface of 16 x 16 faces, each a path with its fill and its mesh lines (a
    /// stroke of CE's Surface width), seen from above at a slant: no mesh line ranks behind its
    /// own face's fill at any sub-sample, as CE draws each face's stroke over its fill.
    #[test]
    fn a_dense_surface_s_mesh_lines_never_rank_behind_their_own_face() {
        let (w, h) = (320, 180);
        let view = crate::probes::view(w, h, 65f64.to_radians(), -40f64.to_radians());
        let n = 16;
        let at = |i: usize, j: usize| {
            let (u, v) = (
                2.0 * i as f64 / n as f64 - 1.0,
                2.0 * j as f64 / n as f64 - 1.0,
            );
            [1.2 * u, 1.2 * v, 0.6 * (u * u - v * v)]
        };
        let mut items = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let slot = (j * n + i) as u32 + 1;
                let corners = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
                let shape = Shape::new(ShapeData::Path {
                    points: ring(&corners),
                });
                let mut r = Record::new(RecordKind::Path, slot, 1000 + slot);
                r.geometry.shape_a = Some(shape.key());
                r.paint.fill = Paint::One(if (i + j) % 2 == 0 {
                    [0.16, 0.38, 0.75, 1.0]
                } else {
                    [0.36, 0.62, 0.95, 1.0]
                });
                r.paint.stroke = Paint::One([0.85, 0.85, 0.85, 1.0]);
                r.paint.stroke_width = 0.5;
                r.depth.space = Space::World;
                items.push((shape, r));
            }
        }
        let order: Vec<u32> = items.iter().map(|(_, r)| r.slot).collect();
        let mut s = Stream::new();
        s.apply(&crate::probes::first_delta(view, &items, &order))
            .expect("the delta applies");
        assert_eq!(own_cover(&s, w, h), 0);
    }

    /// A sub-sample of a fill's fringe: its cell reaches into the fill (coverage above 0) while
    /// its point lies outside the fill's outline (the exact point-in-outline mask).
    struct FringeSample {
        point: [f64; 2],
        /// The fill's key there, as the resolve reads it, and its error bound.
        key: f64,
        err: f64,
        plane: composite::Plane,
        /// The outline's point nearest `point`, and how far it is.
        nearest: [f64; 2],
        distance: f64,
    }

    /// The point of `edges` nearest `p`, and its distance.
    fn nearest_on(edges: &[crate::scanline::Edge], p: [f64; 2]) -> ([f64; 2], f64) {
        let mut best = ([f64::NAN; 2], f64::INFINITY);
        for e in edges {
            let (dx, dy) = (e.x1 - e.x0, e.y1 - e.y0);
            let t =
                (((p[0] - e.x0) * dx + (p[1] - e.y0) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
            let q = [e.x0 + t * dx, e.y0 + t * dy];
            let d = libm::hypot(q[0] - p[0], q[1] - p[1]);
            if d < best.1 {
                best = (q, d);
            }
        }
        best
    }

    /// A sub-sample inside a fill's outline: its point, the fill's key there and its plane.
    type InsideSample = ([f64; 2], f64, composite::Plane);

    /// A lone face through `corners`, filled and not stroked, in `view`: its fringe sub-samples
    /// and its inside ones.
    // The sub-sample `k` indexes each fragment's coverage and depth rows.
    #[allow(clippy::needless_range_loop)]
    fn fill_samples(view: View, corners: &[[f64; 3]]) -> (Vec<FringeSample>, Vec<InsideSample>) {
        let (w, h) = (view.pixel_width, view.pixel_height);
        let shape = Shape::new(ShapeData::Path {
            points: ring(corners),
        });
        let r = crate::probes::fill_record(1, &shape, [0.5, 0.5, 0.5, 1.0], Space::World);
        let mut s = Stream::new();
        s.apply(&crate::probes::first_delta(view, &[(shape, r)], &[1]))
            .expect("the delta applies");
        let (mut fringe, mut inside) = (Vec::new(), Vec::new());
        s.visit_contributors(0, 0, w, h, |x, y, frags, cov, d, producers| {
            for (i, f) in frags.iter().enumerate() {
                let composite::Coverage::Area { id, .. } = f.coverage else {
                    continue;
                };
                let producer = producers[f.source() as usize];
                let edges = producer.edges(id).expect("a path gives its edges");
                // Each end of an edge is an end of another: no horizontal edge, which the
                // rasterizer's edges leave out, is missing from the outline they give.
                for e in edges {
                    for end in [[e.x0, e.y0], [e.x1, e.y1]] {
                        let ends = edges
                            .iter()
                            .filter(|o| [[o.x0, o.y0], [o.x1, o.y1]].contains(&end))
                            .count();
                        assert!(ends >= 2, "a horizontal edge at {end:?}");
                    }
                }
                let mask = producer.mask(id, x, y);
                for k in 0..composite::SAMPLES {
                    let point = composite::sample_point(x, y, k);
                    if mask & (1 << k) != 0 {
                        inside.push((point, d[i].samples[k], f.depth));
                    } else if cov[i][k] != 0 {
                        let (nearest, distance) = nearest_on(edges, point);
                        fringe.push(FringeSample {
                            point,
                            key: d[i].samples[k],
                            err: d[i].err,
                            plane: f.depth,
                            nearest,
                            distance,
                        });
                    }
                }
            }
        })
        .expect("the stream draws");
        (fringe, inside)
    }

    /// A face of a regular octahedron seen two degrees from edge-on, a sliver 1.6 to 3 pixels
    /// across: its depth runs steeply across the screen, so its plane read beyond its outline is
    /// nearer or farther than any point of the face. Wherever a sub-sample's cell reaches into
    /// its fill and its point lies outside the outline, the fill's key is its plane read at the
    /// outline's point nearest it: the depth of the face's edge there. Inside, its plane at the
    /// point.
    #[test]
    fn a_steep_fill_s_fringe_sub_samples_read_the_depth_of_its_edge() {
        let view = crate::probes::view(854, 480, 1.3089969158172607, 0.5235987901687622);
        let o = 0.70709228515625;
        let (fringe, inside) = fill_samples(view, &[[0.0, -o, 0.0], [0.0, 0.0, -o], [o, 0.0, 0.0]]);
        assert!(fringe.len() > 100, "{} fringe sub-samples", fringe.len());
        let steep = fringe
            .iter()
            .filter(|f| {
                (f.plane.at(f.point[0], f.point[1]) - f.plane.at(f.nearest[0], f.nearest[1])).abs()
                    > 1e-6
            })
            .count();
        assert!(
            steep > fringe.len() / 2,
            "{steep} sub-samples beyond the outline read the plane off its edge by 1e-6"
        );
        for f in &fringe {
            let edge = f.plane.at(f.nearest[0], f.nearest[1]);
            assert!(
                (f.key - edge).abs() <= f.err,
                "at {:?}, {:.3} px outside the outline: key {} against the edge's {} (err {:e})",
                f.point,
                f.distance,
                f.key,
                edge,
                f.err
            );
        }
        for (p, key, plane) in &inside {
            assert_eq!(*key, plane.at(p[0], p[1]), "inside, at {p:?}");
        }
    }

    /// A face seen at a slant: beyond its outline its fill's key moves from its plane at the point
    /// by at most the plane's slope times the point's distance from the outline, and inside it
    /// stays the plane.
    #[test]
    fn an_ordinary_fill_s_fringe_keys_move_by_at_most_slope_times_distance() {
        let view = crate::probes::view(320, 180, 65f64.to_radians(), -40f64.to_radians());
        let (fringe, inside) = fill_samples(
            view,
            &[[-1.0, -0.5, 0.2], [1.0, -0.8, -0.1], [0.2, 1.0, 0.3]],
        );
        assert!(fringe.len() > 100, "{} fringe sub-samples", fringe.len());
        assert!(inside.len() > 1000, "{} inside sub-samples", inside.len());
        for f in &fringe {
            let slope = libm::hypot(f.plane.b, f.plane.c);
            let moved = (f.key - f.plane.at(f.point[0], f.point[1])).abs();
            assert!(
                moved <= slope * f.distance + f.err,
                "at {:?}, {:.3} px outside: moved {moved:e}, slope {slope:e}",
                f.point,
                f.distance
            );
        }
        for (p, key, plane) in &inside {
            assert_eq!(*key, plane.at(p[0], p[1]), "inside, at {p:?}");
        }
    }

    fn dot3(u: [f64; 3], v: [f64; 3]) -> f64 {
        u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    }

    fn cross3(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    }

    /// The projection of a probe view's 3D camera.
    fn projection_of(view: &View) -> crate::project::Projection {
        let Camera::ThreeD(c) = &view.camera else {
            unreachable!("a probe view is 3D")
        };
        crate::project::Projection::new(c, view.frame_center)
    }

    /// The cubic arc of the circle about `centre` from `a` to `b` (both on it, less than half a
    /// turn apart), its handles along the tangents, `4/3 tan(angle / 4)` of the radius long.
    fn arc(centre: [f64; 3], a: [f64; 3], b: [f64; 3]) -> [[f64; 3]; 4] {
        let (ra, rb): ([f64; 3], [f64; 3]) = (
            std::array::from_fn(|k| a[k] - centre[k]),
            std::array::from_fn(|k| b[k] - centre[k]),
        );
        let r2 = dot3(ra, ra);
        let m = cross3(ra, rb);
        let m = m.map(|v| v / dot3(m, m).sqrt());
        let angle = libm::acos((dot3(ra, rb) / r2).clamp(-1.0, 1.0));
        let k = 4.0 / 3.0 * libm::tan(angle / 4.0);
        // The tangents, as long as the radius, in the direction of travel.
        let (ta, tb) = (cross3(m, ra), cross3(m, rb));
        [
            a,
            std::array::from_fn(|i| a[i] + k * ta[i]),
            std::array::from_fn(|i| b[i] - k * tb[i]),
            b,
        ]
    }

    /// The point of the unit sphere at longitude `lon` and latitude `lat` (radians).
    fn on_sphere(lon: f64, lat: f64) -> [f64; 3] {
        [
            libm::cos(lat) * libm::cos(lon),
            libm::cos(lat) * libm::sin(lon),
            libm::sin(lat),
        ]
    }

    /// The patch of the unit sphere `half_lon` either side of longitude `lon` and `half_lat`
    /// either side of latitude `lat` (radians), as a surface's face is drawn: its outline as four
    /// cubic arcs, along a parallel, a meridian, a parallel and a meridian.
    fn sphere_patch(lon: f64, lat: f64, half_lon: f64, half_lat: f64) -> Vec<[f64; 3]> {
        let (w, e, s, n) = (
            lon - half_lon,
            lon + half_lon,
            lat - half_lat,
            lat + half_lat,
        );
        let parallel = |la: f64| [0.0, 0.0, libm::sin(la)];
        [
            arc(parallel(s), on_sphere(w, s), on_sphere(e, s)),
            arc([0.0; 3], on_sphere(e, s), on_sphere(e, n)),
            arc(parallel(n), on_sphere(e, n), on_sphere(w, n)),
            arc([0.0; 3], on_sphere(w, n), on_sphere(w, s)),
        ]
        .concat()
    }

    /// The exact range over `0..=1` of the cubic Bernstein polynomial with coefficients `c`.
    fn cubic_range(c: [f64; 4]) -> (f64, f64) {
        let at = |t: f64| {
            let u = 1.0 - t;
            u * u * u * c[0] + 3.0 * u * u * t * c[1] + 3.0 * u * t * t * c[2] + t * t * t * c[3]
        };
        // Its derivative over 3: (d0 - 2 d1 + d2) t^2 + 2 (d1 - d0) t + d0.
        let (d0, d1, d2) = (c[1] - c[0], c[2] - c[1], c[3] - c[2]);
        let (qa, qb, qc) = (d0 - 2.0 * d1 + d2, 2.0 * (d1 - d0), d0);
        let mut ts = vec![0.0, 1.0];
        if qa != 0.0 {
            let disc = qb * qb - 4.0 * qa * qc;
            if disc >= 0.0 {
                ts.extend([
                    (-qb + disc.sqrt()) / (2.0 * qa),
                    (-qb - disc.sqrt()) / (2.0 * qa),
                ]);
            }
        } else if qb != 0.0 {
            ts.push(-qc / qb);
        }
        ts.into_iter()
            .filter(|t| (0.0..=1.0).contains(t))
            .map(at)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                (lo.min(v), hi.max(v))
            })
    }

    /// The range of `value` along an outline of cubics, each taken as the cubic through its
    /// control points' values: exact for a value affine in the point (a depth in the camera's
    /// axes), and the depth key along the outline as the core keys a curve (`path::curve_depth`).
    fn outline_range(outline: &[[f64; 3]], value: impl Fn([f64; 3]) -> f64) -> (f64, f64) {
        outline
            .chunks(4)
            .map(|c| cubic_range(std::array::from_fn(|k| value(c[k]))))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (a, b)| {
                (lo.min(a), hi.max(b))
            })
    }

    /// The range of depth keys along `outline` in `view`.
    fn outline_keys(view: &View, outline: &[[f64; 3]]) -> (f64, f64) {
        let p = projection_of(view);
        outline_range(outline, |q| -p.factor(p.project(q)[2]))
    }

    /// The range of depths in the camera's axes (larger is nearer) along `outline` in `view`.
    fn outline_depths(view: &View, outline: &[[f64; 3]]) -> (f64, f64) {
        let p = projection_of(view);
        outline_range(outline, |q| p.project(q)[2])
    }

    /// A square facing the camera at depth `z` in its axes, `half` across each way of its centre
    /// on the sight line through world point `through`.
    fn facing_square(view: &View, through: [f64; 3], z: f64, half: f64) -> Vec<[f64; 3]> {
        let p = projection_of(view);
        let Camera::ThreeD(c) = &view.camera else {
            unreachable!("a probe view is 3D")
        };
        let d = c.focal_distance;
        let q = p.unproject(p.project(through));
        let scale = (d - z) / (d - q[2]);
        let r = p.rotation();
        let world = |a: [f64; 3]| -> [f64; 3] {
            std::array::from_fn(|j| {
                view.frame_center[j] + r[0][j] * a[0] + r[1][j] * a[1] + r[2][j] * a[2]
            })
        };
        let (cx, cy) = (q[0] * scale, q[1] * scale);
        ring(&[
            world([cx - half, cy - half, z]),
            world([cx + half, cy - half, z]),
            world([cx + half, cy + half, z]),
            world([cx - half, cy + half, z]),
        ])
    }

    /// The angle between the unit sphere's normal at `n` and the sight line from `n` to the eye.
    fn sight_angle(view: &View, n: [f64; 3]) -> f64 {
        let p = projection_of(view);
        let Camera::ThreeD(c) = &view.camera else {
            unreachable!("a probe view is 3D")
        };
        let r = p.rotation();
        let eye: [f64; 3] =
            std::array::from_fn(|j| view.frame_center[j] + c.focal_distance * r[2][j]);
        let s: [f64; 3] = std::array::from_fn(|j| eye[j] - n[j]);
        libm::acos(dot3(n, s) / (dot3(n, n) * dot3(s, s)).sqrt())
    }

    /// The camera's direction (from the frame's centre toward the eye) as a longitude and latitude.
    fn toward_camera(view: &View) -> (f64, f64) {
        let v = projection_of(view).rotation()[2];
        (libm::atan2(v[1], v[0]), libm::asin(v[2]))
    }

    /// The longitude at latitude `lat` where the unit sphere's normal stands `angle` from the
    /// sight line, on the far side of the meridian facing the camera.
    fn longitude_at_sight_angle(view: &View, lat: f64, angle: f64) -> f64 {
        let (facing, _) = toward_camera(view);
        let (mut lo, mut hi) = (facing, facing + std::f64::consts::PI);
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if sight_angle(view, on_sphere(mid, lat)) < angle {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// A path record in world space of object `100 + slot`, filled with `rgba`, stroked `stroke`
    /// wide when given, with the planarity `planes` decided for it when given.
    fn face_record(
        slot: u32,
        rgba: [f64; 4],
        stroke: Option<f64>,
        planes: Option<crate::depth::Planarity>,
    ) -> Record {
        let mut r = Record::new(RecordKind::Path, slot, 100 + slot);
        r.paint.fill = Paint::One(rgba);
        r.depth.space = Space::World;
        if let Some(width) = stroke {
            r.paint.stroke = Paint::One([0.9, 0.9, 0.9, 1.0]);
            r.paint.stroke_width = width;
        }
        r.planes = planes.map(|p| vec![p.code()]);
        r
    }

    /// A stream of paths, each an outline and its record, drawn in their order.
    fn paths_stream(view: View, items: Vec<(Vec<[f64; 3]>, Record)>) -> Stream {
        let items: Vec<(Shape, Record)> = items
            .into_iter()
            .map(|(points, r)| {
                let shape = Shape::new(ShapeData::Path { points });
                let mut r = r;
                r.geometry.shape_a = Some(shape.key());
                (shape, r)
            })
            .collect();
        let order: Vec<u32> = items.iter().map(|(_, r)| r.slot).collect();
        let mut s = Stream::new();
        s.apply(&crate::probes::first_delta(view, &items, &order))
            .expect("the delta applies");
        s
    }

    /// A key read beyond a range by more than its error bound: `(x, y, sub-sample, key, err)`.
    type Beyond = (u32, u32, usize, f64, f64);

    /// The keys of `object`'s fragments at the sub-samples they cover, its strokes on its own fill
    /// as the resolve orders them: those beyond `(lo, hi)` by more than their error bound, and how
    /// many keys were read.
    // The sub-sample `k` indexes each fragment's coverage and depth rows.
    #[allow(clippy::needless_range_loop)]
    fn keys_beyond(s: &Stream, object: u32, (lo, hi): (f64, f64)) -> (Vec<Beyond>, usize) {
        let (w, h) = {
            let view = s.view.as_ref().expect("a view");
            (view.pixel_width, view.pixel_height)
        };
        let (mut beyond, mut read) = (Vec::new(), 0);
        s.visit_contributors(0, 0, w, h, |x, y, frags, cov, d, _| {
            let mut d = d.to_vec();
            composite::strokes_on_own_fills(frags, &mut d, |i| cov[i]);
            for (i, f) in frags.iter().enumerate() {
                if s.leaf_of(f).0 != object {
                    continue;
                }
                for k in 0..composite::SAMPLES {
                    if cov[i][k] == 0 {
                        continue;
                    }
                    read += 1;
                    let (key, err) = (d[i].samples[k], d[i].err);
                    if key < lo - err || key > hi + err {
                        beyond.push((x, y, k, key, err));
                    }
                }
            }
        })
        .expect("the stream draws");
        (beyond, read)
    }

    /// The worst of `beyond` (from [`keys_beyond`]) against `(lo, hi)`: how far past the range,
    /// and its error bound.
    fn worst_beyond(beyond: &[Beyond], (lo, hi): (f64, f64)) -> String {
        beyond
            .iter()
            .max_by(|a, b| {
                let past = |e: &Beyond| (lo - e.3).max(e.3 - hi);
                past(a).total_cmp(&past(b))
            })
            .map(|(x, y, k, key, err)| {
                format!(
                    "{} keys beyond [{lo}, {hi}], the worst at ({x}, {y}) sub-sample {k}: key \
                     {key}, {:e} past, err {err:e}",
                    beyond.len(),
                    (lo - key).max(key - hi)
                )
            })
            .unwrap_or_default()
    }

    /// Of the sub-samples where leaves of objects `front` and `back` both lie, in the order the
    /// resolve composites them: how many put a run of `back` above a run of `front`, and how many
    /// there are.
    fn above(s: &Stream, back: u32, front: u32) -> (usize, usize) {
        let (w, h) = {
            let view = s.view.as_ref().expect("a view");
            (view.pixel_width, view.pixel_height)
        };
        let (mut wrong, mut both) = (0, 0);
        for c in s
            .contributors(0, 0, w, h)
            .expect("the stream draws")
            .values()
        {
            let leaf = |object: u32| c.leaves.iter().position(|l| l.object == object);
            let (Some(f), Some(b)) = (leaf(front), leaf(back)) else {
                continue;
            };
            for seq in &c.samples {
                let runs = |leaf: usize| {
                    seq.iter()
                        .enumerate()
                        .filter(move |(_, e)| e.0 == leaf)
                        .map(|(at, _)| at)
                };
                let (Some(lowest_front), Some(highest_back)) = (runs(f).min(), runs(b).max())
                else {
                    continue;
                };
                both += 1;
                if highest_back > lowest_front {
                    wrong += 1;
                }
            }
        }
        (wrong, both)
    }

    /// A patch of the unit sphere 18 by 9 degrees, filled and stroked, whose normal stands 88.5
    /// degrees from the sight line: a sliver whose plane nearest it (it lies on none) runs steeply
    /// across the screen, so read off its middle the plane leaves the depths its outline spans.
    /// Its key's bound holds that departure: every key of its fill and of its stroke on that fill
    /// lies within its bound of its outline's keys, and an opaque square drawn after it, nearer
    /// than every point of its outline by 0.1, is in front of it at every sub-sample (read as its
    /// plane alone, its fill came 0.18 nearer than its outline).
    #[test]
    fn a_grazing_curved_face_keys_within_its_outline_s_depths() {
        let view = crate::probes::view(854, 480, 80f64.to_radians(), 30f64.to_radians());
        let lat = 0.0;
        let lon = longitude_at_sight_angle(&view, lat, 88.5f64.to_radians());
        let patch = sphere_patch(lon, lat, 9f64.to_radians(), 4.5f64.to_radians());
        let (_, nearest) = outline_depths(&view, &patch);
        let square = facing_square(&view, on_sphere(lon, lat), nearest + 0.1, 0.5);
        let keys = outline_keys(&view, &patch);
        let s = paths_stream(
            view,
            vec![
                (
                    patch,
                    face_record(
                        1,
                        [0.2, 0.4, 0.8, 1.0],
                        Some(0.5),
                        Some(crate::depth::Planarity::Nearest),
                    ),
                ),
                (square, face_record(2, [0.8, 0.2, 0.2, 1.0], None, None)),
            ],
        );
        let (beyond, read) = keys_beyond(&s, 101, keys);
        let (wrong, both) = above(&s, 101, 102);
        assert!(read > 20, "{read} keys read");
        assert!(
            both > 20,
            "{both} sub-samples hold the patch and the square"
        );
        assert!(
            beyond.is_empty() && wrong == 0,
            "{}; of {both} sub-samples, the patch above the square at {wrong}",
            worst_beyond(&beyond, keys)
        );
    }

    /// A face whose corners lie on the plane nearest it and whose sides bow out of that plane
    /// between them, up and down in turn, seen at a slant: the far side bows toward the eye, so the
    /// plane read under it lies beyond every depth of the outline. A bound found at the corners
    /// alone is the plane's rounding; the key's bound holds the sides' departure between them, and
    /// every key of the fill lies within it of the outline's keys.
    #[test]
    fn a_face_bowed_between_its_corners_keys_within_its_outline_s_depths() {
        let view = crate::probes::view(320, 180, 60f64.to_radians(), -80f64.to_radians());
        let corners = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let mut outline = Vec::new();
        for i in 0..4 {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            let z = if i % 2 == 0 { 0.5 } else { -0.5 };
            let at = |t: f64| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
            let (h1, h2) = (at(0.25), at(0.75));
            outline.extend([
                [a[0], a[1], 0.0],
                [h1[0], h1[1], z],
                [h2[0], h2[1], z],
                [b[0], b[1], 0.0],
            ]);
        }
        let plane = crate::depth::WorldPlane::fit(&outline, crate::depth::Planarity::Nearest)
            .expect("the nearest plane");
        let off = |p: &[f64; 3]| (dot3(plane.normal, *p) - plane.offset).abs();
        for (i, p) in outline.iter().enumerate() {
            if i % 4 == 0 || i % 4 == 3 {
                assert!(
                    off(p) <= 1e-12,
                    "corner {p:?} lies {:e} off the plane",
                    off(p)
                );
            } else {
                assert!(
                    off(p) >= 0.4,
                    "handle {p:?} lies {:e} off the plane",
                    off(p)
                );
            }
        }
        let keys = outline_keys(&view, &outline);
        let s = paths_stream(
            view,
            vec![(
                outline,
                face_record(
                    1,
                    [0.2, 0.4, 0.8, 1.0],
                    None,
                    Some(crate::depth::Planarity::Nearest),
                ),
            )],
        );
        let (beyond, read) = keys_beyond(&s, 101, keys);
        assert!(read > 1000, "{read} keys read");
        assert!(beyond.is_empty(), "{}", worst_beyond(&beyond, keys));
    }

    /// A face on one plane exactly, at a slant: its fill is keyed on its plane with the bound the
    /// camera's key across that plane carries, bit for bit, wherever it is read.
    #[test]
    fn an_exactly_planar_face_keeps_its_plane_s_key_and_bound() {
        let view = crate::probes::view(320, 180, 65f64.to_radians(), -40f64.to_radians());
        let (w, h) = (view.pixel_width, view.pixel_height);
        let outline = ring(&[[-1.0, -0.5, 0.2], [1.0, -0.8, -0.1], [0.2, 1.0, 0.3]]);
        let p = projection_of(&view);
        let map = crate::project::FrameMap::of_kind(&view, RecordKind::Path, w, h);
        let plane = crate::depth::WorldPlane::fit(&outline, crate::depth::Planarity::Newell)
            .expect("a plane");
        // Where the lowering reads it: the outline's box on screen, a pixel's reach and a pixel
        // beyond (a fill reaches no further than its antialiasing).
        let shown =
            crate::project::pre_display(&outline, Some(&p), crate::project::Placement::Projected);
        let mut region = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for q in &shown {
            let [x, y] = map.apply(*q);
            region = [
                region[0].min(x - 2.0),
                region[1].min(y - 2.0),
                region[2].max(x + 2.0),
                region[3].max(y + 2.0),
            ];
        }
        let expected = p
            .key_plane(&plane, &map, outline[0], 0.0, region)
            .expect("the plane keys");
        let s = paths_stream(
            view,
            vec![(
                outline,
                face_record(
                    1,
                    [0.2, 0.4, 0.8, 1.0],
                    None,
                    Some(crate::depth::Planarity::Newell),
                ),
            )],
        );
        let mut read = 0;
        s.visit_contributors(0, 0, w, h, |_, _, frags, _, _, _| {
            for f in frags {
                assert_eq!(f.depth, expected);
                read += 1;
            }
        })
        .expect("the stream draws");
        assert!(read > 100, "{read} fragments");
    }

    /// A patch of the unit sphere 18 by 9 degrees, filled and stroked, seen face on, between an
    /// opaque square drawn before it and 0.3 nearer than every point of its outline and one drawn
    /// after it and 0.3 farther than every point: its departure from the plane nearest it is a
    /// fraction of that, so depth orders the three at every sub-sample, against their draw order.
    #[test]
    fn a_curved_face_seen_face_on_keeps_a_strict_order_against_faces_clearly_before_and_behind() {
        let view = crate::probes::view(854, 480, 70f64.to_radians(), -110f64.to_radians());
        let (lon, lat) = toward_camera(&view);
        let patch = sphere_patch(lon, lat, 9f64.to_radians(), 4.5f64.to_radians());
        let (farthest, nearest) = outline_depths(&view, &patch);
        let middle = on_sphere(lon, lat);
        let front = facing_square(&view, middle, nearest + 0.3, 0.5);
        let back = facing_square(&view, middle, farthest - 0.3, 0.5);
        let s = paths_stream(
            view,
            vec![
                (front, face_record(1, [0.8, 0.2, 0.2, 1.0], None, None)),
                (
                    patch,
                    face_record(
                        2,
                        [0.2, 0.4, 0.8, 1.0],
                        Some(0.5),
                        Some(crate::depth::Planarity::Nearest),
                    ),
                ),
                (back, face_record(3, [0.2, 0.8, 0.2, 1.0], None, None)),
            ],
        );
        let (wrong, both) = above(&s, 102, 101);
        assert!(
            both > 1000,
            "{both} sub-samples hold the patch and the square before it"
        );
        assert_eq!(
            wrong, 0,
            "of {both} sub-samples, the patch above the square before it at {wrong}"
        );
        let (wrong, both) = above(&s, 103, 102);
        assert!(
            both > 1000,
            "{both} sub-samples hold the patch and the square behind it"
        );
        assert_eq!(
            wrong, 0,
            "of {both} sub-samples, the square behind the patch above it at {wrong}"
        );
    }

    /// The input grid of the views below: 2^-14, the writer's grid for a 14.2-unit frame 854
    /// pixels wide.
    const GRID: f64 = 1.0 / 16384.0;

    /// A view of `phi` and `theta` (in degrees) on geometry snapped to [`GRID`].
    fn snapped_view(phi: f64, theta: f64) -> View {
        let mut view = crate::probes::view(854, 480, phi.to_radians(), theta.to_radians());
        view.input_grid = GRID;
        view
    }

    /// The polygon through `corners`, given in the camera's axes of `view` (larger z is nearer), as
    /// a closed outline in the world.
    fn camera_ring(view: &View, corners: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let r = projection_of(view).rotation();
        let world = |a: &[f64; 3]| -> [f64; 3] {
            std::array::from_fn(|j| {
                view.frame_center[j] + r[0][j] * a[0] + r[1][j] * a[1] + r[2][j] * a[2]
            })
        };
        ring(&corners.iter().map(world).collect::<Vec<_>>())
    }

    /// A square `side` across, centred at `(cx, cy)` at depth `z` in the camera's axes of `view`,
    /// as a closed outline in the world.
    fn camera_square(view: &View, side: f64, (cx, cy): (f64, f64), z: f64) -> Vec<[f64; 3]> {
        let h = side / 2.0;
        camera_ring(
            view,
            &[
                [cx - h, cy - h, z],
                [cx + h, cy - h, z],
                [cx + h, cy + h, z],
                [cx - h, cy + h, z],
            ],
        )
    }

    /// `points` snapped to `grid`, as the writer snaps positions.
    fn snapped(points: &[[f64; 3]], grid: f64) -> Vec<[f64; 3]> {
        points
            .iter()
            .map(|p| p.map(|v| (v / grid).round() * grid + 0.0))
            .collect()
    }

    /// Of the sub-samples objects 101 and 102 both cover: how many there are, and at how many their
    /// own keys part by more than their error bounds (101 nearer: `near`, or 102 nearer: `far`);
    /// with the plane identities of the two.
    // The sub-sample `k` indexes both fragments' coverage and depth rows.
    #[allow(clippy::needless_range_loop)]
    fn two_faces(s: &Stream) -> ([u32; 2], usize, [usize; 2]) {
        let (w, h) = {
            let view = s.view.as_ref().expect("a view");
            (view.pixel_width, view.pixel_height)
        };
        let mut planes = [0u32; 2];
        let (mut shared, mut apart) = (0usize, [0usize; 2]);
        s.visit_contributors(0, 0, w, h, |_, _, frags, cov, depths, _| {
            let find = |id: u32| frags.iter().position(|f| s.leaf_of(f).0 == id);
            let (Some(i), Some(j)) = (find(101), find(102)) else {
                return;
            };
            planes = [frags[i].plane[0], frags[j].plane[0]];
            for k in 0..composite::SAMPLES {
                if cov[i][k] != 0 && cov[j][k] != 0 {
                    shared += 1;
                    let (a, b) = (depths[i].samples[k], depths[j].samples[k]);
                    if !order::tied(a, depths[i].err, b, depths[j].err) {
                        apart[usize::from(b < a)] += 1;
                    }
                }
            }
        })
        .expect("the stream draws");
        (planes, shared, apart)
    }

    /// Two parallel squares 1 across facing the camera of `view`, `steps` grid steps apart in
    /// depth (the nearer, object 101, drawn first and offset a little), their points as given.
    fn parallel_faces(view: &View, steps: f64) -> Stream {
        let near = camera_square(view, 1.0, (0.0, 0.0), steps * GRID);
        let far = camera_square(view, 1.0, (0.3, 0.2), 0.0);
        paths_stream(
            view.clone(),
            vec![
                (near, face_record(1, [0.8, 0.2, 0.2, 1.0], None, None)),
                (far, face_record(2, [0.2, 0.4, 0.8, 1.0], None, None)),
            ],
        )
    }

    /// How far apart, in grid steps along their normal, the two faces [`parallel_faces`] makes may
    /// lie and still be one plane content made: the snap's departure of each
    /// (`depth::snap_departure`, half a step times `|n|_1`), their points' departure from their
    /// planes being rounding alone; and, besides, the most the two keys' own error bounds may
    /// hold, the rounding of each plane's coefficients to [`crate::depth::PLANE_GRID`] at its
    /// largest (half a step of it times `1 + |p|_1`), which each key's bound carries as it happened.
    fn parallel_reach(view: &View, steps: f64) -> (f64, f64) {
        let n = projection_of(view).rotation()[2];
        let none = [IDENTITY, IDENTITY];
        let near = camera_square(view, 1.0, (0.0, 0.0), steps * GRID);
        let far = camera_square(view, 1.0, (0.3, 0.2), 0.0);
        let faces = [near, far];
        let term = faces
            .iter()
            .map(|p| crate::depth::snap_departure(GRID, n, &none, p))
            .sum::<f64>();
        let keys = faces
            .iter()
            .map(|p| {
                let largest = p.iter().map(|q| q.iter().map(|v| v.abs()).sum::<f64>());
                crate::depth::PLANE_GRID / 2.0 * (1.0 + largest.fold(0.0, f64::max))
            })
            .sum::<f64>();
        (term / GRID, keys / GRID)
    }

    /// Two parallel faces facing the camera, on a view whose normal is far from every axis
    /// (`|n|_1` about 1.55), at the edge of what the snap can part one plane by ([`parallel_reach`])
    /// and beyond. Just inside the reach one plane holds each within its own tolerance, so the
    /// registry gives them one identity, and they tie: the farther, drawn later, above the nearer
    /// wherever both cover. Just beyond it and the keys' own bounds, and at four and five steps, no
    /// plane holds both, each has an identity of its own, and depth orders them: the nearer, drawn
    /// first, above the farther.
    #[test]
    fn parallel_faces_tie_within_the_reach_and_keep_depth_order_beyond_it() {
        let view = snapped_view(70.0, -110.0);
        let (reach, keys) = parallel_reach(&view, 0.0);
        assert!((reach - 1.55).abs() < 0.1, "{reach} steps: |n|_1");
        assert!(keys < 0.1, "{keys} steps: the keys' bounds");
        for (steps, tie) in [
            (reach - 0.01, true),
            (reach + keys + 0.01, false),
            (4.0, false),
            (5.0, false),
        ] {
            let s = parallel_faces(&view, steps);
            let (planes, shared, apart) = two_faces(&s);
            assert!(shared > 50, "{steps} steps: {shared} sub-samples hold both");
            assert!(
                planes[0] != 0 && planes[1] != 0 && (planes[0] == planes[1]) == tie,
                "{steps} steps: identities {planes:?}"
            );
            assert_eq!(apart, [shared, 0], "{steps} steps: the own keys part");
            let (wrong, both) = above(&s, 102, 101);
            let expected = if tie { both } else { 0 };
            assert_eq!(
                wrong, expected,
                "{steps} steps: the farther, drawn later, is above at {wrong} of {both}"
            );
        }
    }

    /// The same faces one grid step apart: within what the snap can do to one plane (two faces
    /// content made coplanar can part by up to `|n|_1` steps), so they tie and keep paint order,
    /// the farther, drawn later, above the nearer wherever both cover.
    #[test]
    fn parallel_faces_one_grid_step_apart_tie_by_paint_order() {
        let view = snapped_view(70.0, -110.0);
        let s = parallel_faces(&view, 1.0);
        let (planes, shared, apart) = two_faces(&s);
        assert!(
            planes[0] != 0 && planes[0] == planes[1],
            "identities {planes:?}"
        );
        assert_eq!(apart, [shared, 0], "the own keys part beyond their bounds");
        let (wrong, both) = above(&s, 102, 101);
        assert!(both > 50, "{both} sub-samples hold both");
        assert_eq!(
            wrong, both,
            "the farther, drawn later, is above at {wrong} of {both}"
        );
    }

    /// Faces facing the diagonal, `(1, 1, 1) / sqrt(3)` (`|n|_1 = sqrt(3)`, the largest): the
    /// snap's departure of the two is `sqrt(3)` steps ([`parallel_reach`]), so just inside that
    /// they share an identity, tie and keep paint order, and just beyond it and the keys' own
    /// bounds each has its own and depth orders them.
    #[test]
    fn diagonal_faces_tie_within_the_snap_s_reach_and_part_beyond_it() {
        let view = snapped_view(54.735_610_317_245_35, 45.0);
        let r = projection_of(&view).rotation();
        let third = 1.0 / 3f64.sqrt();
        for v in r[2] {
            assert!((v - third).abs() < 1e-12, "the camera faces {:?}", r[2]);
        }
        let (reach, keys) = parallel_reach(&view, 0.0);
        assert!(
            reach > 3f64.sqrt() && reach < 3f64.sqrt() + 0.1,
            "{reach} steps"
        );
        assert!(keys < 0.1, "{keys} steps: the keys' bounds");
        for (steps, tie) in [(reach - 0.01, true), (reach + keys + 0.01, false)] {
            let s = parallel_faces(&view, steps);
            let (planes, shared, apart) = two_faces(&s);
            assert!(shared > 50, "{steps} steps: {shared} sub-samples hold both");
            assert!(
                planes[0] != 0 && planes[1] != 0 && (planes[0] == planes[1]) == tie,
                "{steps} steps: identities {planes:?}"
            );
            assert_eq!(apart, [shared, 0], "{steps} steps: the own keys part");
            let (wrong, both) = above(&s, 102, 101);
            let expected = if tie { both } else { 0 };
            assert_eq!(
                wrong, expected,
                "{steps} steps: the farther, drawn later, is above at {wrong} of {both}"
            );
        }
    }

    /// The farther of two parallel faces four grid steps apart, beyond the snap's reach, so on
    /// identities of their own, drawn later with a stroke 4 wide: its fill is behind the nearer
    /// face, and so is its stroke, the farther face's outline never showing through the nearer one.
    #[test]
    fn a_stroke_of_a_face_behind_a_parallel_one_stays_behind_it() {
        let view = snapped_view(70.0, -110.0);
        let near = camera_square(&view, 1.0, (0.0, 0.0), 4.0 * GRID);
        let far = camera_square(&view, 1.0, (0.3, 0.2), 0.0);
        let s = paths_stream(
            view.clone(),
            vec![
                (near, face_record(1, [0.8, 0.2, 0.2, 1.0], None, None)),
                (far, face_record(2, [0.2, 0.4, 0.8, 1.0], Some(4.0), None)),
            ],
        );
        let (planes, shared, _) = two_faces(&s);
        assert!(
            planes[0] != 0 && planes[1] != 0 && planes[0] != planes[1],
            "identities {planes:?}"
        );
        assert!(shared > 50, "{shared} sub-samples hold both");
        let (wrong, both) = above(&s, 102, 101);
        assert!(
            both > shared,
            "{both} sub-samples hold the two, strokes included"
        );
        assert_eq!(
            wrong, 0,
            "the farther face, its stroke included, is above the nearer at {wrong} of {both}"
        );
    }

    /// A face inside another on one slanted plane, snapped as the writer snaps, drawn after it or
    /// before it in space but always later: the later is above wherever both cover, along its edge
    /// too. There a sub-sample it covers in part can lie outside its outline, where its own key is
    /// its plane's at the outline's nearest point; against the face it lies on it is read on its
    /// plane at the sub-sample, as that face is, else the slope between the two points parts them.
    #[test]
    fn a_face_on_a_slanted_plane_keeps_paint_order_with_the_face_it_lies_on_along_its_edge() {
        let view = snapped_view(70.0, -110.0);
        let slanted = |corners: &[[f64; 2]]| -> Vec<[f64; 3]> {
            let at: Vec<[f64; 3]> = corners.iter().map(|&[x, y]| [x, y, 0.6 * x]).collect();
            snapped(&camera_ring(&view, &at), GRID)
        };
        let big = slanted(&[[-0.6, -0.5], [0.6, -0.5], [0.6, 0.5], [-0.6, 0.5]]);
        let small = slanted(&[[-0.23, -0.17], [0.31, -0.21], [0.27, 0.19], [-0.19, 0.23]]);
        let flat = Some(crate::depth::Planarity::Newell);
        for (first, second) in [(&big, &small), (&small, &big)] {
            let s = paths_stream(
                view.clone(),
                vec![
                    (
                        first.clone(),
                        face_record(1, [0.8, 0.2, 0.2, 1.0], None, flat),
                    ),
                    (
                        second.clone(),
                        face_record(2, [0.2, 0.4, 0.8, 1.0], None, flat),
                    ),
                ],
            );
            let (planes, shared, _) = two_faces(&s);
            assert!(
                planes[0] != 0 && planes[0] == planes[1],
                "identities {planes:?}"
            );
            assert!(shared > 50, "{shared} sub-samples hold both");
            let (wrong, both) = above(&s, 101, 102);
            assert_eq!(
                wrong, 0,
                "the earlier drawn is above the later at {wrong} of {both}"
            );
        }
    }

    /// Two outlines content made flat on one plane, each with its placement (`None`: unplaced),
    /// drawn in both orders: in each, they share an identity, their own keys part beyond their
    /// error bounds, and they tie, the later drawn above the earlier wherever both cover. (Depth
    /// would put one of the two above the other in one order or the other.)
    fn ties_either_way(view: &View, faces: [(&[[f64; 3]], Option<Affine>); 2]) {
        let flat = Some(crate::depth::Planarity::Newell);
        for (first, second) in [(0, 1), (1, 0)] {
            let record = |slot: u32, (points, place): (&[[f64; 3]], Option<Affine>)| {
                let mut r = face_record(slot, [0.3 * slot as f64, 0.4, 0.5, 1.0], None, flat);
                if let Some(place) = place {
                    r.geometry.place_a = place;
                }
                (points.to_vec(), r)
            };
            let s = paths_stream(
                view.clone(),
                vec![record(1, faces[first]), record(2, faces[second])],
            );
            let (planes, shared, apart) = two_faces(&s);
            assert!(
                planes[0] != 0 && planes[0] == planes[1],
                "face {first} first: identities {planes:?}"
            );
            assert!(
                shared > 50,
                "face {first} first: {shared} sub-samples hold both"
            );
            assert!(
                apart != [0, 0],
                "face {first} first: the keys part beyond their bounds: {apart:?}"
            );
            let (wrong, both) = above(&s, 101, 102);
            assert_eq!(
                wrong, 0,
                "face {first} first: the earlier is above at {wrong} of {both}"
            );
        }
    }

    /// Two faces content made coplanar on a plane far from every axis, a square 0.6 across over one
    /// 1.4 across, each snapped to the grid as the writer snaps them, at a snap found by search to
    /// part them far: each record's own plane, fitted to its snapped points through the first
    /// (Newell's normal over the outline's corners and thirds), stands off the plane content put
    /// them on by more than the snap's reach, and over their overlap the two own planes part by
    /// about 1.07 times the two reaches together (1.65 steps). The points' departure from their own
    /// plane is what bounds that; within it they are one plane, and tie.
    #[test]
    fn an_oblique_coplanar_stack_at_its_worst_snap_still_ties() {
        let view = snapped_view(70.0, -110.0);
        let small = snapped(
            &camera_square(&view, 0.6, (0.1719096441633351, 0.2368793305480134), 0.0),
            GRID,
        );
        let large = snapped(
            &camera_square(&view, 1.4, (-0.1106199564085423, -0.158344315617575), 0.0),
            GRID,
        );
        ties_either_way(&view, [(&small, None), (&large, None)]);
    }

    /// A thin triangle 0.9 long and 0.09 wide, content made flat on the plane of a square 1.4 across
    /// it lies inside, both snapped, at a snap found by search to part them far: Newell's normal of
    /// a sliver's snapped outline tilts by the snap's noise over its small area, so its points stand
    /// off its own plane by about twice the snap's reach, and over the sliver the two own planes
    /// part by about 1.23 times the two reaches together (1.9 steps). They still tie.
    #[test]
    fn an_oblique_sliver_at_its_worst_snap_still_ties_with_its_plane() {
        let view = snapped_view(70.0, -110.0);
        let (x0, y0) = (-0.5393181618522839, -0.4399545868458875);
        let (length, width) = (0.9052188008988994, 0.09220802858741078);
        let sliver = snapped(
            &camera_ring(
                &view,
                &[
                    [x0, y0, 0.0],
                    [x0 + length, y0 + width / 2.0, 0.0],
                    [x0, y0 + width, 0.0],
                ],
            ),
            GRID,
        );
        let square = snapped(&camera_square(&view, 1.4, (0.0, 0.0), 0.0), GRID);
        ties_either_way(&view, [(&sliver, None), (&square, None)]);
    }

    /// A quad content did not make flat (one corner raised a third of a grid step off the plane of
    /// the others: the writer decided it on no plane), lying on a registered plane within its
    /// tolerance, attaches to that plane's identity and is keyed on its plane, its share of a tie
    /// its snap's departure along that plane's normal and its points' departure from it. A flat
    /// face one grid step nearer shares the identity (one plane holds it and the face beneath
    /// within their tolerances), and the two tie: the bent quad, drawn later, is above it wherever
    /// both cover. Two steps nearer (the first whole step past the two faces' tolerances
    /// together, 1.546 steps) and three, the face has an identity of its own, which the bent quad
    /// does not lie on, and depth orders them: the face above.
    #[test]
    fn a_bent_quad_on_a_registered_plane_ties_within_its_reach_and_is_ordered_beyond_it() {
        let view = snapped_view(70.0, -110.0);
        for (steps, tie) in [(1.0, true), (2.0, false), (3.0, false)] {
            let base = camera_square(&view, 1.6, (0.0, 0.0), 0.0);
            let near = camera_square(&view, 1.0, (0.2, 0.1), steps * GRID);
            let bent = camera_ring(
                &view,
                &[
                    [-0.6, -0.5, 0.0],
                    [0.5, -0.6, 0.0],
                    [0.6, 0.5, GRID / 3.0],
                    [-0.5, 0.6, 0.0],
                ],
            );
            let curve = Some(crate::depth::Planarity::Curve);
            let s = paths_stream(
                view.clone(),
                vec![
                    (base, face_record(3, [0.5, 0.5, 0.5, 1.0], None, None)),
                    (near, face_record(1, [0.8, 0.2, 0.2, 1.0], None, None)),
                    (bent, face_record(2, [0.2, 0.4, 0.8, 1.0], None, curve)),
                ],
            );
            let (planes, shared, apart) = two_faces(&s);
            assert!(
                planes[0] != 0 && planes[1] != 0 && (planes[0] == planes[1]) == tie,
                "{steps} steps: identities {planes:?}"
            );
            assert!(shared > 50, "{steps} steps: {shared} sub-samples hold both");
            assert_eq!(
                apart,
                [shared, 0],
                "{steps} steps: the own keys put the flat face in front"
            );
            let (wrong, both) = above(&s, 102, 101);
            let expected = if tie { both } else { 0 };
            assert_eq!(
                wrong, expected,
                "{steps} steps: the bent quad is above at {wrong} of {both}"
            );
        }
    }

    /// A face placed through a placement that scales coordinates in the thousands down to the
    /// scene's, turned a little in its plane (its linear part snapped to 2^-24, its translation and
    /// its shape's points to the grid, as the writer snaps them), and a face content put on the
    /// same plane in world coordinates, snapped: the placement's snapped linear part may move its
    /// points by 2^-25 times the coordinates it places, here up to 7.6 grid steps, and at this
    /// placement (found by search, among those the registry still finds on one plane) the two own
    /// planes part by about 4 steps over their overlap, 1.8 times every other term together. The
    /// two still tie.
    #[test]
    fn a_placed_face_coplanar_with_a_plain_one_still_ties() {
        let view = snapped_view(70.0, -110.0);
        let r = projection_of(&view).rotation();
        let (scale, turn) = (1.0 / 8499.149353149618, 0.004126534451789704f64);
        let origin = [0.31, -0.17, 0.05];
        let (c, s) = (libm::cos(turn), libm::sin(turn));
        let spin = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        // The placement: local x, y turned in the camera's x, y, scaled, about `origin` (in the
        // camera's axes).
        let mut place = crate::xf::IDENTITY;
        for i in 0..3 {
            for j in 0..3 {
                place[i][j] = scale * (0..3).map(|k| r[k][i] * spin[k][j]).sum::<f64>();
            }
            place[i][3] = view.frame_center[i] + (0..3).map(|j| r[j][i] * origin[j]).sum::<f64>();
        }
        let ratio = 1.0 / 16_777_216.0;
        for row in place.iter_mut() {
            for v in row.iter_mut().take(3) {
                *v = (*v / ratio).round() * ratio + 0.0;
            }
            row[3] = (row[3] / GRID).round() * GRID + 0.0;
        }
        // The placed face: a square 10000 across in local units (1.18 in the scene) at local z 0.
        let local = |x: f64, y: f64| [x, y, 0.0];
        let shape = snapped(
            &ring(&[
                local(-5000.0, -5000.0),
                local(5000.0, -5000.0),
                local(5000.0, 5000.0),
                local(-5000.0, 5000.0),
            ]),
            GRID,
        );
        let plain = snapped(
            &camera_square(&view, 1.4, (origin[0] + 0.1, origin[1] - 0.05), origin[2]),
            GRID,
        );
        ties_either_way(&view, [(&shape, Some(place)), (&plain, None)]);
    }

    /// A face 3 units across, facing the camera, drawn after a square 0.05 across near one of its
    /// corners whose plane is tilted from the face's by 1e-4 (as a small record's fitted plane
    /// tilts by the snap's noise over its size), then a line on the face near its far corner. The
    /// square's points lie on the face's plane within the square's own tolerance, but the face's
    /// far points lie off the square's plane by about 4e-4, far beyond the face's: the two share a
    /// plane only because the square's points lie on the face's plane. The face shares the
    /// square's identity, and its fill is keyed on its own plane: its key lies within its bound of
    /// every point of its outline, and the line, with no plane of its own, attaches to the
    /// identity, ties with the face and is drawn over it at every sub-sample both cover (keyed on
    /// the square's plane, the face came about 2e-5 nearer than its own points at its far corner
    /// and hid the line). Unsnapped, the square's tilt passes its tolerance, and the two are apart.
    #[test]
    fn a_face_that_shares_a_smaller_record_s_plane_is_keyed_on_its_own() {
        let view = snapped_view(70.0, -110.0);
        let (w, h) = (view.pixel_width, view.pixel_height);
        let p = projection_of(&view);
        let r = p.rotation();
        // A point given in the camera's axes (larger z is nearer), in the world.
        let world = |a: [f64; 3]| -> [f64; 3] {
            std::array::from_fn(|j| {
                view.frame_center[j] + r[0][j] * a[0] + r[1][j] * a[1] + r[2][j] * a[2]
            })
        };
        let face = ring(&[
            world([-1.5, -1.5, 0.0]),
            world([1.5, -1.5, 0.0]),
            world([1.5, 1.5, 0.0]),
            world([-1.5, 1.5, 0.0]),
        ]);
        // The square's plane rises toward the eye along the diagonal toward the face's far corner.
        let (tilt, centre) = (1e-4, [-1.3, -1.3]);
        let corner = |dx: f64, dy: f64| {
            let rise = tilt * (dx + dy) / 2f64.sqrt();
            world([centre[0] + dx, centre[1] + dy, rise])
        };
        let square = ring(&[
            corner(-0.025, -0.025),
            corner(0.025, -0.025),
            corner(0.025, 0.025),
            corner(-0.025, 0.025),
        ]);
        let (a, b) = (world([0.7, 1.3, 0.0]), world([1.3, 0.7, 0.0]));
        let line: Vec<[f64; 3]> = (0..4)
            .map(|i| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * i as f64 / 3.0))
            .collect();
        // The premise, against each record's own tolerance along the other's plane (the snap's
        // departure of its points, `depth::snap_departure`, their departure from their own plane
        // being rounding alone): the square's points lie on the face's plane, the face's points
        // lie far off the square's.
        let off = |plane: &crate::depth::WorldPlane, points: &[[f64; 3]]| {
            points
                .iter()
                .map(|q| (dot3(plane.normal, *q) - plane.offset).abs())
                .fold(0.0, f64::max)
        };
        let fit = |points: &[[f64; 3]]| {
            crate::depth::WorldPlane::fit(points, crate::depth::Planarity::Newell).expect("a plane")
        };
        let tolerance = |plane: &crate::depth::WorldPlane, points: &[[f64; 3]]| {
            crate::depth::snap_departure(
                GRID,
                plane.unrounded.normal,
                &[IDENTITY, IDENTITY],
                points,
            )
        };
        let (face_plane, square_plane) = (fit(&face), fit(&square));
        let (on, across) = (off(&face_plane, &square), off(&square_plane, &face));
        assert!(
            on < tolerance(&face_plane, &square) / 2.0,
            "the square lies {on:e} off the face's plane"
        );
        assert!(
            across > 4.0 * tolerance(&square_plane, &face),
            "the face lies {across:e} off the square's plane"
        );
        let mut line_record = Record::new(RecordKind::Path, 3, 103);
        line_record.paint.stroke = Paint::One([0.9, 0.9, 0.9, 1.0]);
        line_record.paint.stroke_width = 2.0;
        line_record.depth.space = Space::World;
        let map = crate::project::FrameMap::of_kind(&view, RecordKind::Path, w, h);
        let shown =
            crate::project::pre_display(&face, Some(&p), crate::project::Placement::Projected);
        let s = paths_stream(
            view.clone(),
            vec![
                (
                    square.clone(),
                    face_record(1, [0.8, 0.2, 0.2, 1.0], None, None),
                ),
                (
                    face.clone(),
                    face_record(2, [0.2, 0.4, 0.8, 1.0], None, None),
                ),
                (line, line_record),
            ],
        );
        // Unsnapped, each one's tolerance is its rounding alone, which the square's tilt over its
        // own extent passes: the two are apart.
        let mut unsnapped = view;
        unsnapped.input_grid = 0.0;
        let apart = identities(&paths_stream(
            unsnapped,
            vec![
                (square, face_record(1, [0.8, 0.2, 0.2, 1.0], None, None)),
                (face, face_record(2, [0.2, 0.4, 0.8, 1.0], None, None)),
            ],
        ));
        assert!(
            !apart[&101].is_empty() && apart[&101].is_disjoint(&apart[&102]),
            "unsnapped, the square lies on {:?} and the face on {:?}",
            apart[&101],
            apart[&102]
        );
        let (mut planes, mut key) = ([0u32; 2], None);
        s.visit_contributors(0, 0, w, h, |_, _, frags, _, _, _| {
            for f in frags {
                match s.leaf_of(f).0 {
                    101 => planes[0] = f.plane[0],
                    102 => {
                        planes[1] = f.plane[0];
                        key = Some(f.depth);
                    }
                    _ => {}
                }
            }
        })
        .expect("the stream draws");
        assert!(
            planes[0] != 0 && planes[0] == planes[1],
            "planes {planes:?}"
        );
        let key = key.expect("the face is drawn");
        let (mut worst, mut bound) = (0.0f64, 0.0);
        for q in &shown {
            let [x, y] = map.apply(*q);
            let gap = (key.at(x, y) + p.factor(q[2])).abs();
            if gap > worst {
                (worst, bound) = (gap, key.err + key.read_error(x, y) + p.key_error(*q));
            }
        }
        // Paint order puts the line above the face: the face above it is depth's order.
        let (wrong, both) = above(&s, 102, 103);
        assert!(both > 50, "{both} sub-samples hold the face and the line");
        assert!(
            worst <= bound && wrong == 0,
            "the face's key stands {worst:e} from its own points' keys, against a bound of \
             {bound:e}; of {both} sub-samples, the face above the line drawn after it at {wrong}"
        );
    }

    /// Three wires along one line, drawn in turn: the first two the same, the third moved by
    /// 2^-51 in z, which moves its key by less than the keys' error bound. Over five frames whose
    /// camera angles step on the grid the engine keeps them to (2^-24), paint order decides them
    /// at every sub-sample they cover: depth tells them apart nowhere.
    #[test]
    // The sub-sample `k` indexes each fragment's coverage and depth rows.
    #[allow(clippy::needless_range_loop)]
    fn coincident_wires_keep_paint_order_at_every_sub_sample_of_every_frame() {
        let (w, h) = (320, 180);
        let grid = (1u64 << 24) as f64;
        let wire = |slot: u32, dz: f64, rgba: [f64; 4]| {
            let shape = crate::probes::polyline(&[[-2.0, -1.0, 0.3 + dz], [2.0, 1.0, -0.2 + dz]]);
            let r = crate::probes::stroke_record(slot, &shape, rgba, 10.0);
            (shape, r)
        };
        let items = [
            wire(1, 0.0, [1.0, 0.0, 0.0, 1.0]),
            wire(2, 0.0, [0.0, 1.0, 0.0, 1.0]),
            wire(3, 2.0 * f64::EPSILON, [0.0, 0.0, 1.0, 1.0]),
        ];
        for frame in 0..5 {
            let phi = (1.1 * grid).round() / grid + f64::from(frame) * 37.0 / grid;
            let theta = (-0.7 * grid).round() / grid + f64::from(frame) * 53.0 / grid;
            let view = crate::probes::view(w, h, phi, theta);
            let mut s = Stream::new();
            s.apply(&crate::probes::first_delta(view, &items, &[1, 2, 3]))
                .expect("the delta applies");
            // The third wire's key stands from the first's by less than either's error bound.
            let (mut apart, mut least_err, mut shared) = (0.0f64, f64::INFINITY, 0);
            s.visit_contributors(0, 0, w, h, |_, _, frags, cov, d, _| {
                if frags.len() != 3 {
                    return;
                }
                for k in 0..composite::SAMPLES {
                    if cov.iter().all(|c| c[k] != 0) {
                        shared += 1;
                        apart = apart.max((d[2].samples[k] - d[0].samples[k]).abs());
                        least_err = least_err.min(d[0].err.min(d[2].err));
                    }
                }
            })
            .expect("the stream draws");
            assert!(
                shared > 1000,
                "frame {frame}: {shared} sub-samples all three cover"
            );
            assert!(
                apart < least_err,
                "frame {frame}: keys apart by {apart:e}, error bound {least_err:e}"
            );
            let found = s.contributors(0, 0, w, h).expect("the stream draws");
            let mut seen = 0;
            for ((x, y), pixel) in &found {
                for (k, sample) in pixel.samples.iter().enumerate() {
                    let order: Vec<u32> = sample
                        .iter()
                        .map(|(l, _)| pixel.leaves[*l].object)
                        .collect();
                    if order.len() < 2 {
                        continue;
                    }
                    seen += 1;
                    assert!(
                        order.windows(2).all(|p| p[0] < p[1]),
                        "frame {frame}, pixel ({x}, {y}) sub-sample {k}: {order:?}, bottom first"
                    );
                }
            }
            assert!(
                seen > 1000,
                "frame {frame}: {seen} sub-samples two wires cover"
            );
        }
    }

    /// The identities fragment `f` lies on ([`composite::Fragment::plane`]), the zeros left out.
    fn ids_of(f: &composite::Fragment) -> Vec<u32> {
        f.plane.iter().copied().filter(|p| *p != 0).collect()
    }

    /// Each object's identities over the frame: the union of its fragments' ([`ids_of`]).
    fn identities(s: &Stream) -> BTreeMap<u32, std::collections::BTreeSet<u32>> {
        let (w, h) = {
            let view = s.view.as_ref().expect("a view");
            (view.pixel_width, view.pixel_height)
        };
        let mut out: BTreeMap<u32, std::collections::BTreeSet<u32>> = BTreeMap::new();
        s.visit_contributors(0, 0, w, h, |_, _, frags, _, _, _| {
            for f in frags {
                out.entry(s.leaf_of(f).0).or_default().extend(ids_of(f));
            }
        })
        .expect("the stream draws");
        out
    }

    /// The drawn objects grouped by the identity each lies on (each on exactly one): the
    /// partition of the records, whatever numbers the identities carry.
    fn partition(s: &Stream) -> std::collections::BTreeSet<std::collections::BTreeSet<u32>> {
        let mut by: BTreeMap<u32, std::collections::BTreeSet<u32>> = BTreeMap::new();
        for (object, ids) in identities(s) {
            assert_eq!(ids.len(), 1, "object {object} lies on {ids:?}");
            by.entry(*ids.first().expect("one identity"))
                .or_default()
                .insert(object);
        }
        by.into_values().collect()
    }

    /// `0..n` shuffled by `seed` (xorshift64*, so every run shuffles alike).
    fn shuffled(n: usize, seed: u64) -> Vec<usize> {
        let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut out: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let j = (s.wrapping_mul(0x2545_f491_4f6c_dd1d) % (i as u64 + 1)) as usize;
            out.swap(i, j);
        }
        out
    }

    /// Faces content made flat (each its outline in the world), drawn in `order`, face `i` as
    /// object `101 + i`.
    fn faces_stream(view: &View, faces: &[Vec<[f64; 3]>], order: &[usize]) -> Stream {
        let flat = Some(crate::depth::Planarity::Newell);
        paths_stream(
            view.clone(),
            order
                .iter()
                .map(|&i| {
                    let shade = 0.2 + 0.6 * (i % 7) as f64 / 7.0;
                    let r = face_record(i as u32 + 1, [shade, 0.4, 0.5, 1.0], None, flat);
                    (faces[i].clone(), r)
                })
                .collect(),
        )
    }

    /// A flat strip of square faces `side` across, `cols` long and two rows wide, on the plane
    /// `z = 0.6 x` of the camera's axes of `view` (31 degrees from facing it), each snapped as the
    /// writer snaps its points: a flat surface's faces.
    fn slanted_strip(view: &View, cols: usize, side: f64) -> Vec<Vec<[f64; 3]>> {
        let x0 = -(cols as f64) * side / 2.0;
        let at = |x: f64, y: f64| [x, y, 0.6 * x];
        let mut faces = Vec::new();
        for row in [-1.0, 0.0] {
            for i in 0..cols {
                let (a, b) = (x0 + i as f64 * side, x0 + (i + 1) as f64 * side);
                let (c, d) = (row * side, (row + 1.0) * side);
                let corners = [at(a, c), at(b, c), at(b, d), at(a, d)];
                faces.push(snapped(&camera_ring(view, &corners), GRID));
            }
        }
        faces
    }

    /// A strip of `cols` faces `side` across on the surface `z = kappa x^2` of the world (facing
    /// its z axis, so the snap's departure of each is about half a grid step), each a flat quad on
    /// its corners (on the surface, unsnapped): a slightly curved surface's faces.
    fn curved_strip(cols: usize, side: f64, kappa: f64) -> Vec<Vec<[f64; 3]>> {
        let x0 = -(cols as f64) * side / 2.0;
        let at = |x: f64, y: f64| [x, y, kappa * x * x];
        (0..cols)
            .map(|i| {
                let (a, b) = (x0 + i as f64 * side, x0 + (i + 1) as f64 * side);
                ring(&[at(a, -side), at(b, -side), at(b, side), at(a, side)])
            })
            .collect()
    }

    /// A face's own tolerance against a plane it shares: the snap's departure of its points along
    /// its own normal (unplaced) and their departure from its own plane.
    fn own_tolerance(points: &[[f64; 3]]) -> f64 {
        let own = crate::depth::WorldPlane::fit(points, crate::depth::Planarity::Newell)
            .expect("a plane");
        let none = [IDENTITY, IDENTITY];
        crate::depth::snap_departure(GRID, own.unrounded.normal, &none, points)
            + crate::depth::departure(&own, points)
    }

    /// The least vertical width of a strip between two parallel lines holding every point
    /// `(x, z)`: twice the largest vertical distance the best line through them (in the minimax
    /// sense) leaves a point. The width at a slope is convex in it and least at the slope of an
    /// edge of the points' hull, so the pairs' slopes find it.
    fn vertical_width(points: &[[f64; 2]]) -> f64 {
        let mut least = f64::INFINITY;
        for (i, p) in points.iter().enumerate() {
            for q in &points[i + 1..] {
                if p[0] == q[0] {
                    continue;
                }
                let slope = (q[1] - p[1]) / (q[0] - p[0]);
                let (lo, hi) = points
                    .iter()
                    .map(|r| r[1] - slope * r[0])
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                        (lo.min(v), hi.max(v))
                    });
                least = least.min(hi - lo);
            }
        }
        least
    }

    /// A flat surface's faces, snapped: two rows of thirty squares 0.12 across on a plane slanted
    /// 31 degrees from facing the camera, drawn in order along the strip. Each face's own plane
    /// tilts by its snap's noise over its small size, so a face's plane misses the faces far down
    /// the strip by more than any tolerance, and a registry that merges in draw order splits the
    /// surface wherever the plane a face took misses its neighbour. All lie on the plane content
    /// put them on, within each one's snap: one identity.
    #[test]
    fn a_flat_surface_s_faces_share_one_identity() {
        let view = snapped_view(70.0, -110.0);
        let faces = slanted_strip(&view, 30, 0.12);
        let order: Vec<usize> = (0..faces.len()).collect();
        let parts = partition(&faces_stream(&view, &faces, &order));
        let sizes: Vec<usize> = parts.iter().map(|p| p.len()).collect();
        assert_eq!(
            sizes,
            vec![faces.len()],
            "{} identities, of sizes {sizes:?}",
            parts.len()
        );
    }

    /// The curvature [`curved_strip`] takes for faces `side` across:
    /// such that the best plane through four neighbours leaves them 0.9 of a face's own tolerance
    /// at most, and through five, 1.35 of it. Any two neighbours lie within a fifth of it of one
    /// plane, so a pairwise match chains them all.
    fn curved_kappa(side: f64) -> f64 {
        0.45 * own_tolerance(&curved_strip(1, side, 0.0)[0]) / (side * side)
    }

    /// A slightly curved surface's faces: twenty-four flat quads 0.15 across on `z = kappa x^2`,
    /// neighbours within a fraction of their tolerance of one plane, the strip's ends far off any
    /// plane through its middle. The registry does not make it one identity; each identity's
    /// members lie on one plane within their own tolerances (checked against the best plane
    /// through the members' points, the least the identity's plane can leave them: the curvature
    /// runs along x, so that is the best line through their `(x, z)`, within the cosine of the
    /// faces' slope).
    #[test]
    fn a_slightly_curved_surface_splits_into_identities_each_within_its_members_tolerances() {
        let view = snapped_view(70.0, -110.0);
        let (cols, side) = (24, 0.15);
        let world = curved_strip(cols, side, curved_kappa(side));
        let order: Vec<usize> = (0..cols).collect();
        let parts = partition(&faces_stream(&view, &world, &order));
        assert!(parts.len() > 1, "one identity for a curved surface");
        for part in &parts {
            let members: Vec<usize> = part.iter().map(|o| (o - 101) as usize).collect();
            let mut points: Vec<[f64; 2]> = members
                .iter()
                .flat_map(|&i| world[i].iter().map(|p| [p[0], p[2]]))
                .collect();
            points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
            points.dedup();
            let most = members
                .iter()
                .map(|&i| own_tolerance(&world[i]))
                .fold(0.0, f64::max);
            let half = vertical_width(&points) / 2.0;
            assert!(
                half * (1.0 - 1e-5) <= most,
                "faces {members:?} on one identity: no plane holds them within {most:e}, the best \
                 leaves one {half:e} off"
            );
        }
    }

    /// The flat surface of [`a_flat_surface_s_faces_share_one_identity`] and the curved one of
    /// [`a_slightly_curved_surface_splits_into_identities_each_within_its_members_tolerances`],
    /// each drawn in its order along the strip and in five shuffled orders: the same partition
    /// of the faces every time.
    #[test]
    fn the_draw_order_does_not_change_the_partition() {
        let view = snapped_view(70.0, -110.0);
        let flat = slanted_strip(&view, 30, 0.12);
        let curved = curved_strip(24, 0.15, curved_kappa(0.15));
        for (name, faces) in [("flat", &flat), ("curved", &curved)] {
            let order: Vec<usize> = (0..faces.len()).collect();
            let base = partition(&faces_stream(&view, faces, &order));
            for seed in 1..=5 {
                let order = shuffled(faces.len(), seed);
                let parts = partition(&faces_stream(&view, faces, &order));
                assert_eq!(parts, base, "{name}, shuffled by seed {seed}");
            }
        }
    }

    /// The eye of `view`'s camera, in the world.
    fn eye_of(view: &View) -> [f64; 3] {
        let Camera::ThreeD(c) = &view.camera else {
            unreachable!("a probe view is 3D")
        };
        let r = projection_of(view).rotation();
        std::array::from_fn(|j| view.frame_center[j] + c.focal_distance * r[2][j])
    }

    /// How far every point lies behind the plane of `face` (fitted to its points as the core fits
    /// it, Newell's normal through the first), seen from `eye`: the least of their distances to
    /// it on the far side, negative where one lies in front.
    fn behind(face: &[[f64; 3]], points: &[[f64; 3]], eye: [f64; 3]) -> f64 {
        let own =
            crate::depth::WorldPlane::fit(face, crate::depth::Planarity::Newell).expect("a plane");
        let (n, d) = (own.unrounded.normal, own.unrounded.offset);
        let toward = (dot3(n, eye) - d).signum();
        points
            .iter()
            .map(|q| -(dot3(n, *q) - d) * toward)
            .fold(f64::INFINITY, f64::min)
    }

    /// A stroke record of object `100 + slot` in world space, `width` wide, with no fill, coded as
    /// lying as `planes`.
    fn line_record(slot: u32, width: f64, planes: crate::depth::Planarity) -> Record {
        let mut r = Record::new(RecordKind::Path, slot, 100 + slot);
        r.planes = Some(vec![planes.code()]);
        r.paint.stroke = Paint::One([0.9, 0.9, 0.9, 1.0]);
        r.paint.stroke_width = width;
        r.depth.space = Space::World;
        r
    }

    /// The straight line from `a` to `b` in CE's cubic layout, snapped as the writer snaps it.
    fn snapped_line(a: [f64; 3], b: [f64; 3]) -> Vec<[f64; 3]> {
        let at = |t: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
        snapped(&[a, at(1.0 / 3.0), at(2.0 / 3.0), b], GRID)
    }

    /// A line drawn on a face: the face a square 2 across on the world plane `z = 0.3 x`, the line
    /// across it along y at `x = x0`, all its points sharing x and z (so coded on the axis plane
    /// of z, as the writer codes it; the face coded flat), both snapped, at an `x0` found by search where the snap puts
    /// every point of the line at least a fifth of a grid step behind the face's own plane. The
    /// line lies on the face's plane as content put it, so it shares the face's identity, ties
    /// with it, and, drawn after it, is above it wherever both cover.
    #[test]
    fn a_line_drawn_on_a_face_shares_its_identity_and_is_drawn_over_it() {
        let view = snapped_view(70.0, -110.0);
        let eye = eye_of(&view);
        let at = |x: f64, y: f64| [x, y, 0.3 * x];
        let face = snapped(
            &ring(&[at(-1.0, -1.0), at(1.0, -1.0), at(1.0, 1.0), at(-1.0, 1.0)]),
            GRID,
        );
        let line = (0..200)
            .map(|k| -0.7 + k as f64 * 0.007)
            .map(|x0| snapped_line(at(x0, -0.8), at(x0, 0.8)))
            .find(|line| behind(&face, line, eye) > 0.2 * GRID)
            .expect("a line the snap puts behind the face");
        let axis = crate::depth::Planarity::Axis(2);
        assert_eq!(crate::depth::WorldPlane::decide(&line, false), axis);
        let flat = Some(crate::depth::Planarity::Newell);
        let s = paths_stream(
            view,
            vec![
                (face, face_record(1, [0.2, 0.4, 0.8, 1.0], None, flat)),
                (line, line_record(2, 3.0, axis)),
            ],
        );
        let ids = identities(&s);
        let (on_face, on_line) = (&ids[&101], &ids[&102]);
        assert!(
            !on_face.is_empty() && on_face.intersection(on_line).next().is_some(),
            "the face lies on {on_face:?}, the line on {on_line:?}"
        );
        let (wrong, both) = above(&s, 101, 102);
        assert!(both > 50, "{both} sub-samples hold the face and the line");
        assert_eq!(
            wrong, 0,
            "the face is above the line drawn after it at {wrong} of {both}"
        );
    }

    /// Two lines crossing on the world's plane `z = 0`, their points sharing z (so coded on that
    /// axis plane, as the writer codes them), with no face to attach to: they lie on one axis
    /// plane, so share its identity and tie, and the line drawn second is above the first wherever
    /// both cover (read on their own centrelines, each sub-sample of the crossing keyed at a
    /// different foot on each, depth would order them both ways).
    #[test]
    fn lines_on_one_axis_plane_with_no_face_tie_and_keep_paint_order() {
        let view = snapped_view(70.0, -110.0);
        let axis = crate::depth::Planarity::Axis(2);
        let one = snapped_line([-1.2, -0.3, 0.0], [1.2, 0.4, 0.0]);
        let two = snapped_line([-0.2, -1.1, 0.0], [0.3, 1.2, 0.0]);
        for line in [&one, &two] {
            assert_eq!(crate::depth::WorldPlane::decide(line, false), axis);
        }
        let s = paths_stream(
            view,
            vec![
                (one, line_record(1, 8.0, axis)),
                (two, line_record(2, 8.0, axis)),
            ],
        );
        let ids = identities(&s);
        let (first, second) = (&ids[&101], &ids[&102]);
        assert!(
            !first.is_empty() && first == second,
            "the lines lie on {first:?} and {second:?}"
        );
        let (wrong, both) = above(&s, 101, 102);
        assert!(both > 50, "{both} sub-samples hold both lines");
        assert_eq!(
            wrong, 0,
            "the first line is above the second at {wrong} of {both}"
        );
    }

    /// How far `points` stand off `plane` at most.
    fn off_of(plane: &crate::depth::WorldPlane, points: &[[f64; 3]]) -> f64 {
        let (n, d) = (plane.unrounded.normal, plane.unrounded.offset);
        points
            .iter()
            .map(|q| (dot3(n, *q) - d).abs())
            .fold(0.0, f64::max)
    }

    /// The centre of the `k`th of the spots [`small_face_and_line`] is searched at.
    fn spot(k: usize) -> (f64, f64) {
        (-0.6 + 0.000713 * k as f64, -0.3 + 0.000419 * k as f64)
    }

    /// A face 0.08 across on the world plane `z = 0.3 x` centred at spot `k`, and a line along x
    /// through it and 2.46 past it either way, `dy` off its centre, both snapped; with how far the
    /// line's points stand off the face's own plane (fitted as the core fits it) at most.
    fn small_face_and_line(k: usize, dy: f64) -> (f64, Vec<[f64; 3]>, Vec<[f64; 3]>) {
        let at = |x: f64, y: f64| [x, y, 0.3 * x];
        let ((cx, cy), h) = (spot(k), 0.04);
        let face = snapped(
            &ring(&[
                at(cx - h, cy - h),
                at(cx + h, cy - h),
                at(cx + h, cy + h),
                at(cx - h, cy + h),
            ]),
            GRID,
        );
        let line = snapped_line(at(cx - 2.5, cy + dy), at(cx + 2.5, cy + dy));
        let own =
            crate::depth::WorldPlane::fit(&face, crate::depth::Planarity::Newell).expect("a plane");
        (off_of(&own, &line), face, line)
    }

    /// The spot, of 4000 searched, where the snap tilts the small face's own plane most: the line
    /// through its centre stands farthest off it ([`small_face_and_line`]).
    fn tilted_most() -> usize {
        (0..4000)
            .map(|k| (small_face_and_line(k, 0.0).0, k))
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .expect("spots")
            .1
    }

    /// Where leaves of objects `a` and `b` both lie: each fragment of `a`'s keys and their bound as
    /// read against each of `b`'s ([`composite::Depths::against`]), its strokes on its own fill as
    /// the resolve orders them, at each sub-sample both cover, as bits, by pixel, the two layers'
    /// sub-orders and the sub-sample.
    #[allow(clippy::type_complexity)]
    fn read_against(
        s: &Stream,
        a: u32,
        b: u32,
    ) -> BTreeMap<(u32, u32, u32, u32, usize), (u64, u64)> {
        let (w, h) = {
            let view = s.view.as_ref().expect("a view");
            (view.pixel_width, view.pixel_height)
        };
        let mut out = BTreeMap::new();
        s.visit_contributors(0, 0, w, h, |x, y, frags, cov, d, _| {
            let mut d = d.to_vec();
            composite::strokes_on_own_fills(frags, &mut d, |i| cov[i]);
            let of =
                |object: u32| (0..frags.len()).filter(move |i| s.leaf_of(&frags[*i]).0 == object);
            for i in of(a) {
                for j in of(b) {
                    let (keys, err) = d[i].against(&frags[i], &frags[j]);
                    for k in 0..composite::SAMPLES {
                        if cov[i][k] != 0 && cov[j][k] != 0 {
                            let at = (x, y, frags[i].sub, frags[j].sub, k);
                            out.insert(at, (keys[k].to_bits(), err.to_bits()));
                        }
                    }
                }
            }
        })
        .expect("the stream draws");
        out
    }

    /// The small face and the line through it ([`small_face_and_line`]), its points sharing y (so
    /// coded on the axis plane of y), at the spot where the snap tilts the face's own plane most,
    /// the line's ends standing more than 8 grid steps off it. Content put both on one plane,
    /// which holds each within its snap's departure, and the bound grows outside the face's hull
    /// as fast as that plane may part from the face's own, so the line lies on the face's identity
    /// and ties with it: drawn after it, above it wherever both cover. So at every 500th spot too,
    /// the order not shimmering as the snap's tilt moves. Its share against the face is the bound
    /// within the face's hull ([`crate::depth::Registry::off_members`]), under half its ends'
    /// departure from the identity's plane, which gives its share against what else has no plane
    /// of its own ([`composite::Fragment::wide`]).
    #[test]
    fn a_line_far_past_a_small_face_ties_with_it_wherever_both_cover() {
        let k = tilted_most();
        let (most, face, line) = small_face_and_line(k, 0.0);
        assert!(
            most > 8.0 * GRID,
            "the line's ends stand at most {} steps off the face's plane",
            most / GRID
        );
        let axis = crate::depth::Planarity::Axis(1);
        assert_eq!(crate::depth::WorldPlane::decide(&line, false), axis);
        let content = crate::depth::WorldPlane::canonical([-0.3, 0.0, 1.0], [0.0; 3]).unwrap();
        for points in [&face, &line] {
            let n = content.unrounded.normal;
            let snap = crate::depth::snap_departure(GRID, n, &[IDENTITY, IDENTITY], points);
            assert!(
                off_of(&content, points) <= snap,
                "content's plane holds each"
            );
        }
        let flat = Some(crate::depth::Planarity::Newell);
        let drawn = |face: Vec<[f64; 3]>, line: Vec<[f64; 3]>| {
            paths_stream(
                snapped_view(70.0, -110.0),
                vec![
                    (face, face_record(1, [0.2, 0.4, 0.8, 1.0], None, flat)),
                    (line, line_record(2, 6.0, axis)),
                ],
            )
        };
        for at in std::iter::once(k).chain((0..4000).step_by(500)) {
            let (_, face, line) = small_face_and_line(at, 0.0);
            let s = drawn(face, line);
            let ids = identities(&s);
            assert!(
                !ids[&101].is_empty() && ids[&102] == ids[&101],
                "at spot {at} the face lies on {:?}, the line on {:?}",
                ids[&101],
                ids[&102]
            );
            let (wrong, both) = above(&s, 101, 102);
            assert!(
                both > 20 && wrong == 0,
                "at spot {at} the face is above the line drawn after it at {wrong} of {both}"
            );
        }
        let view = snapped_view(70.0, -110.0);
        let (w, h) = (view.pixel_width, view.pixel_height);
        let mut r = crate::depth::Registry::new(&view);
        let places = [IDENTITY, IDENTITY];
        let newell = crate::depth::Planarity::Newell;
        let (f, ..) = r.offer_path(&face, newell, &places, &face);
        let (l, ..) = r.offer_path(&line, axis, &places, &line);
        r.settle();
        let id = r.identities(f)[0];
        let plane = r.plane(id);
        let whole =
            r.tolerance_along(l, plane.unrounded.normal) + crate::depth::departure(&plane, &line);
        let off = r.off_members(l, id);
        assert!(
            whole > 2.0 * off && r.off_identity(l, id) == whole,
            "{} steps against {}",
            whole / GRID,
            off / GRID
        );
        let p = projection_of(&view);
        let map = crate::project::FrameMap::of_kind(&view, RecordKind::Path, w, h);
        let share = [off, whole].map(|d| p.key_reach(&plane, &map, d));
        let s = drawn(face, line);
        let mut shares = Vec::new();
        s.visit_contributors(0, 0, w, h, |_, _, frags, _, _, _| {
            shares.extend(
                frags
                    .iter()
                    .filter(|f| s.leaf_of(f).0 == 102)
                    .map(|f| [f.snap, f.wide]),
            );
        })
        .expect("the stream draws");
        assert!(
            !shares.is_empty() && shares.iter().all(|v| *v == share),
            "the line's shares {:?}, against {share:?}",
            shares.first()
        );
    }

    /// The small face and the line through it at the spot where the snap tilts the face's plane
    /// most ([`a_line_far_past_a_small_face_ties_with_it_wherever_both_cover`]), and a wall the line
    /// pierces just past the face's hull, on the axis plane of x 0.01 past the face's edge, all
    /// snapped: the line lies on the face's identity, not the wall's, and against the wall it is read
    /// as it is with the face not drawn, on its curve, key and bound bit for bit, so attaching moves
    /// no order there.
    #[test]
    fn an_attached_line_is_read_as_unattached_against_what_shares_none_of_its_identities() {
        let k = tilted_most();
        let (_, face, line) = small_face_and_line(k, 0.0);
        let (cx, cy) = spot(k);
        let x = cx + 0.05;
        let wall = snapped(
            &ring(&[
                [x, cy - 0.2, 0.3 * x - 0.2],
                [x, cy + 0.2, 0.3 * x - 0.2],
                [x, cy + 0.2, 0.3 * x + 0.2],
                [x, cy - 0.2, 0.3 * x + 0.2],
            ]),
            GRID,
        );
        let flat = Some(crate::depth::Planarity::Newell);
        let axis = crate::depth::Planarity::Axis(1);
        let drawn = |with_face: bool| {
            let mut items = Vec::new();
            if with_face {
                items.push((
                    face.clone(),
                    face_record(1, [0.2, 0.4, 0.8, 1.0], None, flat),
                ));
            }
            items.push((line.clone(), line_record(2, 6.0, axis)));
            let on_x = Some(crate::depth::Planarity::Axis(0));
            items.push((
                wall.clone(),
                face_record(3, [0.8, 0.3, 0.2, 1.0], None, on_x),
            ));
            paths_stream(snapped_view(70.0, -110.0), items)
        };
        let (with, without) = (drawn(true), drawn(false));
        let ids = identities(&with);
        assert!(
            !ids[&101].is_empty() && ids[&102] == ids[&101] && ids[&103].is_disjoint(&ids[&102]),
            "the face lies on {:?}, the line on {:?}, the wall on {:?}",
            ids[&101],
            ids[&102],
            ids[&103]
        );
        let read = read_against(&with, 102, 103);
        assert!(read.len() > 20, "{} sub-samples", read.len());
        assert_eq!(read, read_against(&without, 102, 103));
    }

    /// A face 0.2 across on the world plane `z = 0.3 x`, its corners as given, and a line along
    /// its edge at `x = 0.1` running 2.9 past it either way, three ulps of x outside that edge and
    /// a quarter grid step behind the face's plane. No point of the line comes within the snap's
    /// reach of the face's hull and no segment of it crosses the hull, but the segment along the
    /// edge passes within that reach, standing off the plane within the bound there: the line lies
    /// on the face's identity and ties with it, drawn after it, above it wherever both cover.
    #[test]
    fn a_line_along_a_face_s_edge_just_outside_its_hull_lies_on_its_identity() {
        let view = snapped_view(70.0, -110.0);
        let eye = eye_of(&view);
        let (h, at) = (0.1, |x: f64, y: f64| [x, y, 0.3 * x]);
        let face = ring(&[at(-h, -h), at(h, -h), at(h, h), at(-h, h)]);
        let x = f64::from_bits(h.to_bits() + 3);
        let back = if eye[2] > 0.3 * eye[0] { -1.0 } else { 1.0 } * GRID / 4.0;
        let line = [-3.0, -1.0, 1.0, 3.0].map(|y| [x, y, 0.3 * x + back]);
        assert!(
            behind(&face, &line, eye) > GRID / 8.0,
            "the line stands {} steps behind the face",
            behind(&face, &line, eye) / GRID
        );
        assert!(line.iter().all(|p| p[1].abs() - h >= 0.9));
        let flat = Some(crate::depth::Planarity::Newell);
        let coded = crate::depth::WorldPlane::decide(&line, false);
        let s = paths_stream(
            view,
            vec![
                (face, face_record(1, [0.2, 0.4, 0.8, 1.0], None, flat)),
                (line.to_vec(), line_record(2, 6.0, coded)),
            ],
        );
        let ids = identities(&s);
        assert!(
            !ids[&101].is_empty() && ids[&102] == ids[&101],
            "the face lies on {:?}, the line on {:?}",
            ids[&101],
            ids[&102]
        );
        let (wrong, both) = above(&s, 101, 102);
        assert!(
            both > 50 && wrong == 0,
            "the face is above the line drawn after it at {wrong} of {both}"
        );
    }

    /// A face 0.6 across on the world plane `z = 0.3 x`, a square 0.2 across inside it coded as a
    /// curve (so with no plane of its own) and filled, and a third face crossing both on a plane
    /// tilted about x, all snapped: the curve lies on the face's identity, and against the third
    /// face its fill is read as it is with the face not drawn, along its curve, key and bound bit
    /// for bit.
    #[test]
    fn an_attached_fill_is_read_as_unattached_against_what_shares_none_of_its_identities() {
        let at = |x: f64, y: f64| [x, y, 0.3 * x];
        let square = |h: f64| snapped(&ring(&[at(-h, -h), at(h, -h), at(h, h), at(-h, h)]), GRID);
        let tilted = |x: f64, y: f64| [x, y, 0.3 * x + 0.5 * y];
        let third = snapped(
            &ring(&[
                tilted(-0.15, -0.3),
                tilted(0.15, -0.3),
                tilted(0.15, 0.3),
                tilted(-0.15, 0.3),
            ]),
            GRID,
        );
        let flat = Some(crate::depth::Planarity::Newell);
        let curve = Some(crate::depth::Planarity::Curve);
        let drawn = |with_face: bool| {
            let mut items = Vec::new();
            if with_face {
                items.push((
                    square(0.3),
                    face_record(1, [0.2, 0.4, 0.8, 1.0], None, flat),
                ));
            }
            items.push((
                square(0.1),
                face_record(2, [0.9, 0.8, 0.1, 1.0], None, curve),
            ));
            items.push((
                third.clone(),
                face_record(3, [0.8, 0.3, 0.2, 1.0], None, flat),
            ));
            paths_stream(snapped_view(70.0, -110.0), items)
        };
        let (with, without) = (drawn(true), drawn(false));
        let ids = identities(&with);
        assert!(
            !ids[&101].is_empty() && ids[&102] == ids[&101] && ids[&103].is_disjoint(&ids[&102]),
            "the face lies on {:?}, the curve on {:?}, the third on {:?}",
            ids[&101],
            ids[&102],
            ids[&103]
        );
        assert!(identities(&without)[&102].is_empty());
        let read = read_against(&with, 102, 103);
        assert!(read.len() > 20, "{} sub-samples", read.len());
        assert_eq!(read, read_against(&without, 102, 103));
        // Against the face it shares an identity with, the fill is read on that identity's plane.
        let mut rows = 0;
        with.visit_contributors(0, 0, 854, 480, |_, _, frags, _, d, _| {
            let of = |object: u32| frags.iter().position(|f| with.leaf_of(f).0 == object);
            if let (Some(i), Some(j)) = (of(102), of(101)) {
                let row = d[i].on_plane.as_ref().expect("a row on the face's plane");
                assert_eq!(d[i].against(&frags[i], &frags[j]), (&row.0, row.1));
                rows += 1;
            }
        })
        .expect("the stream draws");
        assert!(rows > 20, "{rows} pixels");
    }

    /// Two lines through the small face at the spot where the snap tilts its plane most
    /// ([`small_face_and_line`]), content's plane holding all three: one along x a quarter of the
    /// face's width off its centre (coded on the axis plane of y), one through the other side
    /// turned 0.1 from it (coded a curve), crossing 0.4 past the face's hull. Both lie on the
    /// face's identity, and where they cross each is read on its plane against the other: they
    /// tie, the later drawn above the other wherever both cover.
    #[test]
    fn two_lines_on_a_small_face_s_identity_crossing_outside_its_hull_tie() {
        let k = tilted_most();
        let (_, face, first) = small_face_and_line(k, 0.02);
        let (cx, cy) = spot(k);
        let at = |x: f64, y: f64| [x, y, 0.3 * x];
        let (along, across) = (libm::cos(0.1), libm::sin(0.1));
        let second = snapped_line(
            at(cx - 2.5 * along, cy - 0.02 - 2.5 * across),
            at(cx + 2.5 * along, cy - 0.02 + 2.5 * across),
        );
        let flat = Some(crate::depth::Planarity::Newell);
        let s = paths_stream(
            snapped_view(70.0, -110.0),
            vec![
                (face, face_record(1, [0.2, 0.4, 0.8, 1.0], None, flat)),
                (first, line_record(2, 6.0, crate::depth::Planarity::Axis(1))),
                (second, line_record(3, 6.0, crate::depth::Planarity::Curve)),
            ],
        );
        let ids = identities(&s);
        assert!(
            !ids[&101].is_empty() && ids[&102] == ids[&101] && ids[&103] == ids[&101],
            "the face lies on {:?}, the lines on {:?} and {:?}",
            ids[&101],
            ids[&102],
            ids[&103]
        );
        let (wrong, both) = above(&s, 102, 103);
        assert!(
            both > 20 && wrong == 0,
            "the first line is above the second at {wrong} of {both}"
        );
        let (w, h) = (854, 480);
        let mut rows = 0;
        s.visit_contributors(0, 0, w, h, |_, _, frags, _, d, _| {
            let of = |object: u32| frags.iter().position(|f| s.leaf_of(f).0 == object);
            let (Some(i), Some(j)) = (of(102), of(103)) else {
                return;
            };
            for (m, o) in [(i, j), (j, i)] {
                let row = d[m].on_plane.as_ref().expect("a row on the face's plane");
                assert_eq!(d[m].against(&frags[m], &frags[o]), (&row.0, row.1));
                rows += 1;
            }
        })
        .expect("the stream draws");
        assert!(rows > 0);
    }

    /// A cube turned off every axis, two of its faces drawn, then the edge they share as a line,
    /// all snapped, at a turn found by search where both faces face the eye and the snap puts the
    /// edge at least a fifth of a grid step behind the second face's own plane. The edge lies on
    /// both faces' planes as content put it, so its identities hold both faces', and it ties with
    /// each: drawn after both, it is above each wherever they cover together.
    #[test]
    fn a_line_along_a_cube_s_edge_ties_with_both_faces() {
        let view = snapped_view(70.0, -110.0);
        let eye = eye_of(&view);
        let h = 0.6;
        let turned = |b: f64, c: f64, p: [f64; 3]| -> [f64; 3] {
            let (sb, cb, sc, cc) = (libm::sin(b), libm::cos(b), libm::sin(c), libm::cos(c));
            let q = [p[0], cb * p[1] - sb * p[2], sb * p[1] + cb * p[2]];
            [cc * q[0] - sc * q[1], sc * q[0] + cc * q[1], q[2]]
        };
        let mut found = None;
        'search: for i in 0..24 {
            for j in 0..24 {
                let (b, c) = (0.1 + i as f64 * 0.13, 0.05 + j as f64 * 0.26);
                let at = |p: [f64; 3]| turned(b, c, p);
                let facing = |n: [f64; 3]| dot3(at(n), eye) > 0.3 * dot3(eye, eye).sqrt();
                if !facing([1.0, 0.0, 0.0]) || !facing([0.0, 1.0, 0.0]) {
                    continue;
                }
                let first = snapped(
                    &ring(&[
                        at([h, -h, -h]),
                        at([h, h, -h]),
                        at([h, h, h]),
                        at([h, -h, h]),
                    ]),
                    GRID,
                );
                let second = snapped(
                    &ring(&[
                        at([-h, h, -h]),
                        at([-h, h, h]),
                        at([h, h, h]),
                        at([h, h, -h]),
                    ]),
                    GRID,
                );
                let edge = snapped_line(at([h, h, -h]), at([h, h, h]));
                if behind(&second, &edge, eye) > 0.2 * GRID {
                    found = Some((first, second, edge));
                    break 'search;
                }
            }
        }
        let (first, second, edge) = found.expect("a turn the snap puts the edge behind a face");
        let flat = Some(crate::depth::Planarity::Newell);
        let s = paths_stream(
            view,
            vec![
                (first, face_record(1, [0.8, 0.2, 0.2, 1.0], None, flat)),
                (second, face_record(2, [0.2, 0.4, 0.8, 1.0], None, flat)),
                (edge, line_record(3, 4.0, crate::depth::Planarity::Curve)),
            ],
        );
        let ids = identities(&s);
        let (one, two, line) = (&ids[&101], &ids[&102], &ids[&103]);
        assert!(
            !one.is_empty() && !two.is_empty() && one != two,
            "the faces lie on {one:?} and {two:?}"
        );
        assert!(
            one.is_subset(line) && two.is_subset(line),
            "the faces lie on {one:?} and {two:?}, the edge on {line:?}"
        );
        for face in [101, 102] {
            let (wrong, both) = above(&s, face, 103);
            assert!(
                both > 20,
                "{both} sub-samples hold face {face} and the edge"
            );
            assert_eq!(
                wrong, 0,
                "face {face} is above the edge drawn after it at {wrong} of {both}"
            );
        }
    }

    /// A sliver mesh triangle on a face drawn before it: 1.8 long and six grid steps wide, so its
    /// corners lie on a line within the snap (it has no plane of its own), on the face's world
    /// plane `z = 0.3 x` but a third of a grid step behind it, as a snap could leave it. It
    /// attaches to the face's identity and ties with it: drawn after it, it is above it wherever
    /// both cover, not hidden.
    #[test]
    fn a_sliver_triangle_on_a_face_ties_with_it_and_is_not_hidden() {
        let view = snapped_view(70.0, -110.0);
        let eye = eye_of(&view);
        let at = |x: f64, y: f64| [x, y, 0.3 * x];
        let face = ring(&[at(-1.0, -1.0), at(1.0, -1.0), at(1.0, 1.0), at(-1.0, 1.0)]);
        let length = 1.09f64.sqrt();
        let n = [-0.3 / length, 0.0, 1.0 / length];
        let toward = (dot3(n, eye) - dot3(n, face[0])).signum();
        let back = |p: [f64; 3]| -> [f64; 3] {
            std::array::from_fn(|k| p[k] - toward * GRID / 3.0 * n[k])
        };
        let corners = vec![
            back(at(0.1, -0.9)),
            back(at(0.1, 0.9)),
            back(at(0.1 + 6.0 * GRID, 0.0)),
        ];
        assert!(crate::depth::on_a_line(&corners, GRID));
        let mesh = Shape::new(ShapeData::Mesh {
            positions: corners,
            triangles: vec![[0, 1, 2]],
            normals: None,
        });
        let mut sliver = Record::new(RecordKind::Mesh, 2, 102);
        sliver.paint.fill = Paint::One([0.2, 0.4, 0.8, 1.0]);
        sliver.depth.space = Space::World;
        sliver.geometry.shape_a = Some(mesh.key());
        let path = Shape::new(ShapeData::Path { points: face });
        let flat = Some(crate::depth::Planarity::Newell);
        let mut plain = face_record(1, [0.8, 0.2, 0.2, 1.0], None, flat);
        plain.geometry.shape_a = Some(path.key());
        let items = vec![(path, plain), (mesh, sliver)];
        let order: Vec<u32> = items.iter().map(|(_, r)| r.slot).collect();
        let mut s = Stream::new();
        s.apply(&crate::probes::first_delta(view, &items, &order))
            .expect("the delta applies");
        let ids = identities(&s);
        let (on, sliver) = (&ids[&101], &ids[&102]);
        assert!(
            !on.is_empty() && on.is_subset(sliver),
            "the face lies on {on:?}, the sliver on {sliver:?}"
        );
        let (wrong, both) = above(&s, 101, 102);
        assert!(both > 3, "{both} sub-samples hold the face and the sliver");
        assert_eq!(
            wrong, 0,
            "the face is above the sliver drawn after it at {wrong} of {both}"
        );
    }

    /// Two parallel squares facing the camera 1.4 grid steps apart, within the snap's reach (one
    /// identity, its plane halfway between), and drawn last a sliver mesh triangle (corners on a
    /// line within the snap) on the farther one's plane: 0.7 steps behind the identity's plane,
    /// within its own tolerance, so it attaches to the identity. Its corners stand 1.4 steps
    /// behind the nearer square, more than that square's own share covers; with the share its
    /// identity gives it (`Meshes::identify`) it ties with the square and, drawn later, is above.
    #[test]
    fn a_sliver_triangle_off_its_identity_s_plane_ties_on_its_own_share() {
        let view = snapped_view(70.0, -110.0);
        let r = projection_of(&view).rotation();
        let world = |a: [f64; 3]| -> [f64; 3] {
            std::array::from_fn(|j| {
                view.frame_center[j] + r[0][j] * a[0] + r[1][j] * a[1] + r[2][j] * a[2]
            })
        };
        let near = camera_square(&view, 1.0, (0.0, 0.0), 1.4 * GRID);
        let far = camera_square(&view, 1.0, (0.3, 0.2), 0.0);
        // Across the nearer square at a slope no sub-sample pattern repeats along.
        let (length, wide) = (libm::hypot(0.8, 0.55), 5.5 * GRID);
        let corners = vec![
            world([-0.4, -0.25, 0.0]),
            world([0.4, 0.3, 0.0]),
            world([-wide * 0.55 / length, 0.025 + wide * 0.8 / length, 0.0]),
        ];
        assert!(crate::depth::on_a_line(&corners, GRID));
        let mesh = Shape::new(ShapeData::Mesh {
            positions: corners,
            triangles: vec![[0, 1, 2]],
            normals: None,
        });
        let mut sliver = Record::new(RecordKind::Mesh, 3, 103);
        sliver.paint.fill = Paint::One([0.9, 0.9, 0.2, 1.0]);
        sliver.depth.space = Space::World;
        sliver.geometry.shape_a = Some(mesh.key());
        let mut items: Vec<(Shape, Record)> = [
            (near, face_record(1, [0.8, 0.2, 0.2, 1.0], None, None)),
            (far, face_record(2, [0.2, 0.4, 0.8, 1.0], None, None)),
        ]
        .into_iter()
        .map(|(points, mut r)| {
            let shape = Shape::new(ShapeData::Path { points });
            r.geometry.shape_a = Some(shape.key());
            (shape, r)
        })
        .collect();
        items.push((mesh, sliver));
        let order: Vec<u32> = items.iter().map(|(_, r)| r.slot).collect();
        let mut s = Stream::new();
        s.apply(&crate::probes::first_delta(view.clone(), &items, &order))
            .expect("the delta applies");
        let ids = identities(&s);
        let (near, far, sliver) = (&ids[&101], &ids[&102], &ids[&103]);
        assert!(
            !near.is_empty() && near == far && near.is_subset(sliver),
            "the squares lie on {near:?} and {far:?}, the sliver on {sliver:?}"
        );
        let (wrong, both) = above(&s, 101, 103);
        assert!(
            both > 3,
            "{both} sub-samples hold the nearer square and the sliver"
        );
        assert_eq!(
            wrong, 0,
            "the nearer square is above the sliver drawn after it at {wrong} of {both}"
        );
    }

    /// The world square on z = 0 spanning `[x0, x0 + side]` in x and `[-side / 2, side / 2]` in
    /// y, as a closed outline.
    fn flat_square(x0: f64, side: f64) -> Vec<[f64; 3]> {
        let h = side / 2.0;
        ring(&[
            [x0, -h, 0.0],
            [x0 + side, -h, 0.0],
            [x0 + side, h, 0.0],
            [x0, h, 0.0],
        ])
    }

    /// The writer's grid for a placement's linear part: 2^-24.
    const RATIO_STEP: f64 = 1.0 / 16_777_216.0;

    /// The placement `x -> x + shift`, `z -> z + slope x`, which takes z = 0 onto
    /// `z = slope (x - shift)`: linear, so a flat shape stays flat, turned by `atan(slope)` about
    /// the line `x = shift` of z = 0.
    fn sheared(slope: f64, shift: f64) -> Affine {
        [
            [1.0, 0.0, 0.0, shift],
            [0.0, 1.0, 0.0, 0.0],
            [slope, 0.0, 1.0, 0.0],
        ]
    }

    /// `steps` steps of the ratio grid toward `view`'s eye from z = 0: a slope that turns a face
    /// sheared by it ([`sheared`]) nearer the camera where `x` grows.
    fn toward_eye(view: &View, steps: f64) -> f64 {
        eye_of(view)[2].signum() * steps * RATIO_STEP
    }

    /// Two faces 3 across, content made flat, on the grid: A, object 101, the world square on z = 0
    /// centred on the origin; B, object 102, the square `[-1.5 - shift, 1.5 - shift] x
    /// [-1.5, 1.5]` placed by `sheared(slope, shift)`, so over A, turned from it by `atan(slope)`
    /// about the line `x = shift`: with `shift` -1.5 about A's edge (a hinge), with `shift` 0
    /// about the y axis (a crossing). Drawn A first when `a_first`.
    fn sheared_pair(view: &View, slope: f64, shift: f64, a_first: bool) -> Stream {
        let flat = Some(crate::depth::Planarity::Newell);
        let a = (
            flat_square(-1.5, 3.0),
            face_record(1, [0.8, 0.3, 0.2, 1.0], None, flat),
        );
        let mut placed = face_record(2, [0.2, 0.4, 0.8, 1.0], None, flat);
        placed.geometry.place_a = sheared(slope, shift);
        let b = (flat_square(-1.5 - shift, 3.0), placed);
        paths_stream(view.clone(), if a_first { vec![a, b] } else { vec![b, a] })
    }

    /// The turn the snaps of [`sheared_pair`]'s faces allow about their dihedral's axis (the y
    /// axis), `θ_A + θ_B`, each `2 τ_R / L_R`: `τ_R` the record's tolerance along its own normal
    /// (the snap's departure of its points with its placements, and their departure from its own
    /// plane: `depth::snap_departure`, `depth::departure`), `L_R` its extent across the axis (3);
    /// and the dihedral of the two own planes, each fitted as the core fits a path content made
    /// flat.
    fn sheared_turns(slope: f64, shift: f64) -> (f64, f64) {
        use crate::depth::{Planarity, WorldPlane, departure, snap_departure};
        let place = sheared(slope, shift);
        let a = flat_square(-1.5, 3.0);
        let shape = flat_square(-1.5 - shift, 3.0);
        let b: Vec<[f64; 3]> = shape.iter().map(|p| place_point(&place, *p)).collect();
        let turn = |points: &[[f64; 3]], places: &[Affine; 2], shape: &[[f64; 3]]| {
            let own = WorldPlane::fit(points, Planarity::Newell).expect("a plane");
            let n = own.unrounded.normal;
            let tau = snap_departure(GRID, n, places, shape) + departure(&own, points);
            (2.0 * tau / 3.0, n)
        };
        let (ta, na) = turn(&a, &[IDENTITY, IDENTITY], &a);
        let (tb, nb) = turn(&b, &[IDENTITY, place], &shape);
        let c = cross3(na, nb);
        let dihedral = libm::atan2(dot3(c, c).sqrt(), dot3(na, nb).abs());
        (ta + tb, dihedral)
    }

    /// `p` through placement `m`.
    fn place_point(m: &Affine, p: [f64; 3]) -> [f64; 3] {
        crate::xf::place(m, p)
    }

    /// One sub-sample objects 101 and 102 both cover.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Shared {
        at: (u32, u32, usize),
        /// Their keys part by more than their error bounds.
        apart: bool,
        /// Their keys lie within the tie term of each other, as two fragments on one registered
        /// plane are read (`order::coplanar_tied`): their error bounds and each one's share of its
        /// key ([`composite::Fragment::snap`]) apart, or less.
        within: bool,
        /// 101's key is the nearer.
        a_nearer: bool,
        /// The resolve composites 101 above 102 there.
        a_above: bool,
    }

    /// The sub-samples objects 101 and 102 both cover in `s`, as [`Shared`] reads each.
    // The sub-sample `k` indexes both fragments' coverage and depth rows.
    #[allow(clippy::needless_range_loop)]
    fn shared_samples(s: &Stream) -> Vec<Shared> {
        let (w, h) = {
            let view = s.view.as_ref().expect("a view");
            (view.pixel_width, view.pixel_height)
        };
        let mut keys = BTreeMap::new();
        s.visit_contributors(0, 0, w, h, |x, y, frags, cov, d, _| {
            let find = |id: u32| frags.iter().position(|f| s.leaf_of(f).0 == id);
            let (Some(i), Some(j)) = (find(101), find(102)) else {
                return;
            };
            let (fa, fb) = (&frags[i], &frags[j]);
            let reach = |d: f64, e: f64, share: f64| e + (d.abs() + e) * share;
            for k in 0..composite::SAMPLES {
                if cov[i][k] == 0 || cov[j][k] == 0 {
                    continue;
                }
                let (da, ea, db, eb) = (d[i].samples[k], d[i].err, d[j].samples[k], d[j].err);
                let within = !(fa.snap.is_finite() && fb.snap.is_finite())
                    || order::tied(da, reach(da, ea, fa.snap), db, reach(db, eb, fb.snap));
                keys.insert((x, y, k), (!order::tied(da, ea, db, eb), within, da < db));
            }
        })
        .expect("the stream draws");
        let mut out = Vec::new();
        for ((x, y), pixel) in &s.contributors(0, 0, w, h).expect("the stream draws") {
            let leaf = |object: u32| pixel.leaves.iter().position(|l| l.object == object);
            let (Some(a), Some(b)) = (leaf(101), leaf(102)) else {
                continue;
            };
            for (k, seq) in pixel.samples.iter().enumerate() {
                let at = |l: usize| seq.iter().rposition(|(m, _)| *m == l);
                let (Some(pa), Some(pb)) = (at(a), at(b)) else {
                    continue;
                };
                if let Some(&(apart, within, a_nearer)) = keys.get(&(*x, *y, k)) {
                    out.push(Shared {
                        at: (*x, *y, k),
                        apart,
                        within,
                        a_nearer,
                        a_above: pa > pb,
                    });
                }
            }
        }
        out
    }

    /// Of `shared`: the sub-samples whose keys part beyond their error bounds yet lie within the
    /// tie term (where only the near-parallel rule can tie them), and those beyond the term.
    fn by_term(shared: &[Shared]) -> (Vec<Shared>, Vec<Shared>) {
        let within = shared.iter().filter(|x| x.apart && x.within).copied();
        let beyond = shared.iter().filter(|x| !x.within).copied();
        (within.collect(), beyond.collect())
    }

    /// Two faces 3 across on the writer's grid (2^-14), B turned from A by 4.0e-5 rad about A's
    /// edge (671 steps of the ratio grid), toward the camera: within the turn their snaps allow
    /// (about 6.2e-5: A's own plane may turn 2.0e-5, B's, placed, 4.1e-5), so the input does not
    /// decide how the two are turned to each other, though no one plane holds both within their
    /// tolerances and each has an identity of its own. Where their keys lie within the tie term
    /// they tie, and A, drawn later, is above B; beyond it (the far third of the overlap from the
    /// hinge) depth orders them, and B, nearer, is above A.
    #[test]
    fn faces_turned_apart_within_what_their_snaps_allow_tie_within_the_term_and_not_beyond() {
        let view = snapped_view(30.0, -60.0);
        let slope = toward_eye(&view, 671.0);
        let (room, dihedral) = sheared_turns(slope, -1.5);
        assert!(
            (dihedral - 4.0e-5).abs() < 1e-7 && dihedral / room < 0.7,
            "dihedral {dihedral:e}, room {room:e}"
        );
        let s = sheared_pair(&view, slope, -1.5, false);
        let (planes, ..) = two_faces(&s);
        assert!(
            planes[0] != 0 && planes[1] != 0 && planes[0] != planes[1],
            "identities {planes:?}"
        );
        let shared = shared_samples(&s);
        let (within, beyond) = by_term(&shared);
        assert!(
            within.len() > shared.len() / 2 && beyond.len() > shared.len() / 20,
            "of {} sub-samples, {} within the term and apart, {} beyond it",
            shared.len(),
            within.len(),
            beyond.len()
        );
        assert!(shared.iter().filter(|x| x.apart).all(|x| !x.a_nearer));
        let depth_within = within.iter().filter(|x| !x.a_above).count();
        let paint_beyond = beyond.iter().filter(|x| x.a_above).count();
        assert_eq!(
            (depth_within, paint_beyond),
            (0, 0),
            "of {} within the term, B above at {depth_within}; of {} beyond, A above at \
             {paint_beyond}",
            within.len(),
            beyond.len()
        );
    }

    /// The same hinge just within the turn the two faces' snaps allow (0.998 of it, to the ratio
    /// grid's step) and just past it (1.002): within, the faces tie where their keys lie within
    /// the term, A drawn later above B at every such sub-sample; past it, depth orders them there,
    /// B above A at every one.
    #[test]
    fn hinged_faces_tie_just_within_the_turn_their_snaps_allow_and_not_just_past_it() {
        let view = snapped_view(30.0, -60.0);
        for (target, tie) in [(0.998, true), (1.002, false)] {
            let mut steps = 1000.0;
            for _ in 0..4 {
                let (room, _) = sheared_turns(toward_eye(&view, steps), -1.5);
                steps = (target * room / RATIO_STEP).round();
            }
            let slope = toward_eye(&view, steps);
            let (room, dihedral) = sheared_turns(slope, -1.5);
            let ratio = dihedral / room;
            assert!(
                if tie { ratio < 0.9995 } else { ratio > 1.0005 },
                "{steps} steps: ratio {ratio}"
            );
            let s = sheared_pair(&view, slope, -1.5, false);
            let shared = shared_samples(&s);
            let (within, _) = by_term(&shared);
            assert!(
                within.len() > shared.len() / 4,
                "ratio {ratio}: {} of {} within the term and apart",
                within.len(),
                shared.len()
            );
            let paint = within.iter().filter(|x| x.a_above).count();
            assert_eq!(
                paint,
                if tie { within.len() } else { 0 },
                "ratio {ratio}: A, drawn later, above at {paint} of {} within the term",
                within.len()
            );
        }
    }

    /// Whichever of the two hinged faces is drawn first (so whichever the order reads first), at
    /// 0.8 of the turn their snaps allow, where A's own allowance alone, twice over, falls short
    /// of the dihedral and B's does not: the faces tie at the same sub-samples, the later drawn
    /// above there, and depth orders them alike everywhere else.
    #[test]
    fn the_near_parallel_tie_is_the_same_whichever_face_is_drawn_first() {
        let view = snapped_view(30.0, -60.0);
        let mut steps = 1000.0;
        for _ in 0..4 {
            let (room, _) = sheared_turns(toward_eye(&view, steps), -1.5);
            steps = (0.8 * room / RATIO_STEP).round();
        }
        let slope = toward_eye(&view, steps);
        let runs =
            [false, true].map(|a_first| shared_samples(&sheared_pair(&view, slope, -1.5, a_first)));
        assert_eq!(runs[0].len(), runs[1].len());
        let (within, _) = by_term(&runs[0]);
        assert!(within.len() > runs[0].len() / 4, "{} within", within.len());
        let mut tied = 0;
        for (x, y) in runs[0].iter().zip(&runs[1]) {
            assert_eq!(
                (x.at, x.apart, x.within, x.a_nearer),
                (y.at, y.apart, y.within, y.a_nearer)
            );
            if x.apart && x.within {
                // Tied: the later drawn above, A in the first run, B in the second.
                assert!(x.a_above && !y.a_above, "{x:?} and {y:?}");
                tied += 1;
            } else if x.apart {
                assert_eq!(x.a_above, y.a_above, "{x:?} and {y:?}");
            }
        }
        assert_eq!(tied, within.len());
    }

    /// Two faces 3 across crossing at 2.0e-3 rad (about fifty times the turn their snaps allow),
    /// each on an identity of its own: the input decides how they are turned to each other, so
    /// depth orders them at every sub-sample where their keys part beyond their error bounds,
    /// though along the crossing a band of them (a few pixels) lies within the tie term.
    #[test]
    fn faces_crossing_at_a_dihedral_the_input_decides_keep_depth_order_beyond_their_bounds() {
        let view = snapped_view(30.0, -60.0);
        let slope = toward_eye(&view, (2.0e-3 / RATIO_STEP).round());
        let (room, dihedral) = sheared_turns(slope, 0.0);
        assert!(dihedral / room > 30.0, "{dihedral:e} against {room:e}");
        for a_first in [false, true] {
            let s = sheared_pair(&view, slope, 0.0, a_first);
            let shared = shared_samples(&s);
            let (within, _) = by_term(&shared);
            assert!(
                within.len() > 1000,
                "{} of {} within the term",
                within.len(),
                shared.len()
            );
            let wrong = shared
                .iter()
                .filter(|x| x.apart && x.a_above != x.a_nearer)
                .count();
            assert_eq!(
                wrong,
                0,
                "A first {a_first}: of {} sub-samples, the farther above at {wrong}",
                shared.len()
            );
        }
    }

    /// Two faces 3 across on the writer's grid, one hinged off the other's edge by 3.5e-5 rad
    /// toward the camera, the pair turning slowly about the world x axis (0.0021 rad a frame,
    /// over sixteen frames of one stream), each frame's points snapped as the writer snaps them:
    /// the snap's noise changes from frame to frame, and turns each face's own plane by up to
    /// what the snap allows, so their keys lie within the tie term over nearly all of their
    /// overlap and which is nearer there is the noise's. The input does not decide their order:
    /// they tie, and the face drawn later is above at every sub-sample of every frame, never
    /// flickering under the other.
    #[test]
    fn near_parallel_faces_turning_slowly_keep_paint_order_at_every_sub_sample_of_every_frame() {
        let view = snapped_view(30.0, -60.0);
        let mut s = Stream::new();
        let (mut apart_identities, mut noise_nearer) = (0, 0);
        for frame in 0..16u32 {
            turning_pair(&mut s, &view, frame, 3.5e-5, [1, 2]);
            let (planes, ..) = two_faces(&s);
            apart_identities += usize::from(planes[0] != planes[1]);
            let shared = shared_samples(&s);
            let within = shared.iter().filter(|x| x.within).count();
            assert!(
                shared.len() > 100_000 && within > shared.len() * 4 / 5,
                "frame {frame}: {within} of {} within the term",
                shared.len()
            );
            noise_nearer += usize::from(shared.iter().any(|x| x.apart && x.a_nearer));
            let flicker = shared.iter().filter(|x| x.a_above).count();
            assert_eq!(
                flicker,
                0,
                "frame {frame}: A, drawn first, above B at {flicker} of {} sub-samples",
                shared.len()
            );
        }
        assert!(
            apart_identities >= 8,
            "{apart_identities} frames on two identities"
        );
        assert!(
            noise_nearer >= 4,
            "{noise_nearer} frames with A's key the nearer somewhere"
        );
    }

    /// Frame `frame` of the turning pair applied to `s`: two faces 3 across on the writer's grid,
    /// B hinged off A's edge by `rise` rad toward the camera, the pair turned about the world x
    /// axis by 0.3 + 0.0021 `frame` rad, each frame's points snapped as the writer snaps them,
    /// drawn in `order` (of slots 1, A, and 2, B).
    fn turning_pair(s: &mut Stream, view: &View, frame: u32, rise: f64, order: [u32; 2]) {
        let eye = eye_of(view);
        let flat = Some(crate::depth::Planarity::Newell);
        let phi = 0.3 + 0.0021 * f64::from(frame);
        let (c, sn) = (libm::cos(phi), libm::sin(phi));
        let turned = |p: [f64; 3]| [p[0], p[1] * c - p[2] * sn, p[1] * sn + p[2] * c];
        let up = dot3(eye, [0.0, -sn, c]).signum();
        let face = |rise: f64| -> Vec<[f64; 3]> {
            let points: Vec<[f64; 3]> = flat_square(-1.5, 3.0)
                .into_iter()
                .map(|p| turned([p[0], p[1], up * rise * (p[0] + 1.5)]))
                .collect();
            snapped(&points, GRID)
        };
        let items: Vec<(Shape, Record)> = [(face(0.0), 1u32), (face(rise), 2)]
            .into_iter()
            .map(|(points, slot)| {
                let shape = Shape::new(ShapeData::Path { points });
                let shade = [0.8 - 0.6 * f64::from(slot - 1), 0.3, 0.5, 1.0];
                let mut r = face_record(slot, shade, None, flat);
                r.geometry.shape_a = Some(shape.key());
                (shape, r)
            })
            .collect();
        let delta = if frame == 0 {
            crate::probes::first_delta(view.clone(), &items, &order)
        } else {
            let mut d = Delta::new(
                Time {
                    num: i64::from(frame),
                    den: 30,
                },
                u64::from(frame),
            );
            for (shape, r) in &items {
                d.upload.push(shape.clone());
                d.set.push(r.clone());
            }
            d
        };
        s.apply(&delta).expect("the delta applies");
    }

    /// A measured limit, recorded and not asserted: the turning pair drawn the other way round,
    /// the nearer face (B) first, at the hinge above (3.5e-5 rad) and at one near the turn the
    /// snaps allow (6e-5 rad). Where the pair ties, paint order puts A, the farther, above, and
    /// where its keys leave the term depth puts B above, so the drawn order flips between frames
    /// along the term's edge as the snap's noise moves it; and at 6e-5 the verdict itself is the
    /// noise's (near-parallel in some frames and not in others), which flips the whole overlap.
    /// Sub-samples whose order flips between consecutive frames, of about 7.5 million compared
    /// over the sixteen frames, counted as here: at b4056338, before the near-parallel tie, and at
    /// 4e316ffe, which adds it (03da4caf, which runs the test once per pair, draws the same bits):
    ///
    /// | hinge  | drawn first | b4056338  | 4e316ffe  |
    /// |--------|-------------|-----------|-----------|
    /// | 3.5e-5 | A (farther) | 287,864        | 0         |
    /// | 3.5e-5 | B (nearer)  | 1,767,421      | 249,403   |
    /// | 6e-5   | A (farther) | 118,616        | 31,207    |
    /// | 6e-5   | B (nearer)  | 178,104        | 3,297,310 |
    ///
    /// The test prints its counts (`--nocapture`); it asserts only that each frame draws the two
    /// over the overlap it measures.
    #[test]
    fn near_parallel_faces_drawn_nearer_first_flip_along_the_term_s_edge_measured() {
        let view = snapped_view(30.0, -60.0);
        for (rise, order) in [(3.5e-5, [2, 1]), (6e-5, [2, 1])] {
            let mut s = Stream::new();
            let mut last: Option<BTreeMap<(u32, u32, usize), bool>> = None;
            let (mut flips, mut compared) = (0usize, 0usize);
            for frame in 0..16u32 {
                turning_pair(&mut s, &view, frame, rise, order);
                let shared = shared_samples(&s);
                assert!(
                    shared.len() > 100_000,
                    "frame {frame}: {} shared",
                    shared.len()
                );
                let now: BTreeMap<(u32, u32, usize), bool> =
                    shared.iter().map(|x| (x.at, x.a_above)).collect();
                if let Some(before) = &last {
                    for (k, above) in &now {
                        if let Some(was) = before.get(k) {
                            compared += 1;
                            flips += usize::from(was != above);
                        }
                    }
                }
                last = Some(now);
            }
            eprintln!("hinge {rise:e}, B drawn first: {flips} flips of {compared} compared");
        }
    }

    /// [`sheared_pair`]'s two faces hinged off A's edge at 0.8 of the turn their snaps allow, B
    /// drawn first and A, when `with_a`, after it; and, drawn last when `with_line`, object 103: a
    /// line along y at `x = 1.4`, across both faces and on to 8 either way, rising off z = 0 by
    /// 1.7e-5 per unit of y, coded on the axis plane of x. A plane through the line and A, rising
    /// so, holds A within its tolerances, so the line lies on A's identity; across B it stands
    /// 1.2e-4 or more off B's plane, which no plane holding B within its own tolerances comes near.
    fn hinged_faces_and_line(view: &View, with_a: bool, with_line: bool) -> Stream {
        let mut steps = 1000.0;
        for _ in 0..4 {
            let (room, _) = sheared_turns(toward_eye(view, steps), -1.5);
            steps = (0.8 * room / RATIO_STEP).round();
        }
        let flat = Some(crate::depth::Planarity::Newell);
        let mut placed = face_record(2, [0.2, 0.4, 0.8, 1.0], None, flat);
        placed.geometry.place_a = sheared(toward_eye(view, steps), -1.5);
        let mut items = vec![(flat_square(0.0, 3.0), placed)];
        if with_a {
            items.push((
                flat_square(-1.5, 3.0),
                face_record(1, [0.8, 0.3, 0.2, 1.0], None, flat),
            ));
        }
        if with_line {
            let line = [-8.0, -8.0 / 3.0, 8.0 / 3.0, 8.0].map(|y| [1.4, y, 1.7e-5 * y]);
            items.push((
                line.to_vec(),
                line_record(3, 6.0, crate::depth::Planarity::Axis(0)),
            ));
        }
        paths_stream(view.clone(), items)
    }

    /// A fragment's key read against another, its bound, and its shares: `snap`, `wide` and `own`
    /// ([`composite::Fragment`]).
    type Read = (f64, f64, f64, f64, f64);

    /// One sub-sample objects `a` and `b` both cover, each one's key read against the other
    /// ([`composite::Depths::against`]) with its bound, each fragment's shares, and whether the
    /// resolve composites `a` above `b` there.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Met {
        at: (u32, u32, usize),
        a: Read,
        b: Read,
        a_above: bool,
    }

    /// The sub-samples objects `a` and `b` both cover in `s`, as [`Met`] reads each.
    // The sub-sample `k` indexes both fragments' coverage and depth rows.
    #[allow(clippy::needless_range_loop)]
    fn met(s: &Stream, a: u32, b: u32) -> Vec<Met> {
        let (w, h) = {
            let view = s.view.as_ref().expect("a view");
            (view.pixel_width, view.pixel_height)
        };
        let mut keys = BTreeMap::new();
        s.visit_contributors(0, 0, w, h, |x, y, frags, cov, d, _| {
            let mut d = d.to_vec();
            composite::strokes_on_own_fills(frags, &mut d, |i| cov[i]);
            let find = |id: u32| frags.iter().position(|f| s.leaf_of(f).0 == id);
            let (Some(i), Some(j)) = (find(a), find(b)) else {
                return;
            };
            let (fa, fb) = (&frags[i], &frags[j]);
            let ((ka, ea), (kb, eb)) = (d[i].against(fa, fb), d[j].against(fb, fa));
            for k in 0..composite::SAMPLES {
                if cov[i][k] != 0 && cov[j][k] != 0 {
                    let read = |key: f64, err: f64, f: &composite::Fragment| {
                        (key, err, f.snap, f.wide, f.own)
                    };
                    keys.insert((x, y, k), (read(ka[k], ea, fa), read(kb[k], eb, fb)));
                }
            }
        })
        .expect("the stream draws");
        let mut out = Vec::new();
        for ((x, y), pixel) in &s.contributors(0, 0, w, h).expect("the stream draws") {
            let leaf = |object: u32| pixel.leaves.iter().position(|l| l.object == object);
            let (Some(la), Some(lb)) = (leaf(a), leaf(b)) else {
                continue;
            };
            for (k, seq) in pixel.samples.iter().enumerate() {
                let at = |l: usize| seq.iter().rposition(|(m, _)| *m == l);
                let (Some(pa), Some(pb)) = (at(la), at(lb)) else {
                    continue;
                };
                if let Some(&(ka, kb)) = keys.get(&(*x, *y, k)) {
                    out.push(Met {
                        at: (*x, *y, k),
                        a: ka,
                        b: kb,
                        a_above: pa > pb,
                    });
                }
            }
        }
        out
    }

    /// Whether `m`'s two keys lie within the term each one's `share` of its key and their bounds
    /// make ([`order::coplanar_tied`]'s term).
    fn within_shares(m: &Met, share: impl Fn(Read) -> f64) -> bool {
        let reach = |(d, e, ..): Read, share: f64| e + (d.abs() + e) * share;
        let (sa, sb) = (share(m.a), share(m.b));
        !(sa.is_finite() && sb.is_finite())
            || order::tied(m.a.0, reach(m.a, sa), m.b.0, reach(m.b, sb))
    }

    /// The hinged faces of [`hinged_faces_and_line`], with and without the line on A's identity,
    /// whose whole share past A's hull is more than its share against A: the faces' fragments carry
    /// no whole share, and at every sub-sample both faces cover the two are read, tie and are
    /// ordered as without the line, near-parallel, within their own shares: tied, A, drawn later,
    /// above B; beyond the term, the nearer above.
    #[test]
    fn near_parallel_faces_off_a_line_s_identity_tie_as_without_it() {
        let view = snapped_view(30.0, -60.0);
        let (with, without) = (
            hinged_faces_and_line(&view, true, true),
            hinged_faces_and_line(&view, true, false),
        );
        let ids = identities(&with);
        assert!(
            ids[&101] != ids[&102] && ids[&103] == ids[&101],
            "A lies on {:?}, B on {:?}, the line on {:?}",
            ids[&101],
            ids[&102],
            ids[&103]
        );
        let (mut faces, mut lines) = (0, 0);
        with.visit_contributors(0, 0, 854, 480, |_, _, frags, _, _, _| {
            for f in frags {
                match with.leaf_of(f).0 {
                    103 => {
                        assert!(
                            f.wide > f.snap && f.snap > 0.0,
                            "the line's shares {} and {}",
                            f.snap,
                            f.wide
                        );
                        lines += 1;
                    }
                    _ => {
                        assert_eq!(f.wide, f64::NEG_INFINITY);
                        faces += 1;
                    }
                }
            }
        })
        .expect("the stream draws");
        assert!(
            faces > 1000 && lines > 100,
            "{faces} face and {lines} line fragments"
        );
        let shared = shared_samples(&with);
        assert_eq!(shared, shared_samples(&without));
        let (within, beyond) = by_term(&shared);
        assert!(
            within.len() > shared.len() / 4 && beyond.len() > shared.len() / 20,
            "of {}, {} within the term and apart, {} beyond it",
            shared.len(),
            within.len(),
            beyond.len()
        );
        assert!(within.iter().all(|x| x.a_above));
        assert!(
            beyond
                .iter()
                .filter(|x| x.apart)
                .all(|x| x.a_above == x.a_nearer)
        );
    }

    /// The hinged faces of [`hinged_faces_and_line`] and the line on A's identity, drawn last:
    /// each pair is ordered by its own rule at every sub-sample both cover. The faces, on two
    /// identities and near-parallel, by their own shares ([`order::near_parallel`]); the line and
    /// A, on one identity, by the line's share against A's members, the bound within A's hull,
    /// though its whole share is more: tied, the line above; and the line and B, on no identity in
    /// common, by their keys' bounds alone, the line read as it is with A not drawn.
    #[test]
    fn a_line_on_one_of_two_near_parallel_faces_is_ordered_by_each_pair_s_own_rule() {
        let view = snapped_view(30.0, -60.0);
        let s = hinged_faces_and_line(&view, true, true);
        let ordered = |m: &Met, tie: bool, a_later: bool| {
            if tie {
                m.a_above == a_later
            } else {
                m.a_above == (m.a.0 < m.b.0)
            }
        };
        let faces = met(&s, 101, 102);
        let apart_faces = faces.iter().filter(|m| !within_shares(m, |a| a.2)).count();
        assert!(faces.len() > 100_000 && apart_faces > faces.len() / 20);
        let broken = faces
            .iter()
            .filter(|m| !ordered(m, within_shares(m, |a| a.2), true))
            .count();
        assert_eq!(broken, 0, "of {} sub-samples of the faces", faces.len());
        let on_a = met(&s, 103, 101);
        let snap_tied = on_a.iter().filter(|m| within_shares(m, |a| a.2)).count();
        let wide_only = on_a
            .iter()
            .filter(|m| !within_shares(m, |a| a.2) && within_shares(m, |a| a.3.max(a.2)))
            .count();
        assert!(
            on_a.len() > 100,
            "{} sub-samples of the line and A",
            on_a.len()
        );
        assert!(
            on_a.iter()
                .all(|m| m.a.3 > m.a.2 && m.b.3 == f64::NEG_INFINITY)
        );
        let broken = on_a
            .iter()
            .filter(|m| !ordered(m, within_shares(m, |a| a.2), true))
            .count();
        assert_eq!(
            broken,
            0,
            "of {} sub-samples of the line and A ({snap_tied} tied, {wide_only} tied by the whole share alone)",
            on_a.len()
        );
        let on_b = met(&s, 103, 102);
        assert!(
            on_b.len() > 100,
            "{} sub-samples of the line and B",
            on_b.len()
        );
        let broken = on_b
            .iter()
            .filter(|m| !ordered(m, order::tied(m.a.0, m.a.1, m.b.0, m.b.1), true))
            .count();
        assert_eq!(broken, 0, "of {} sub-samples of the line and B", on_b.len());
        let alone = hinged_faces_and_line(&view, false, true);
        assert_eq!(read_against(&s, 103, 102), read_against(&alone, 103, 102));
    }

    /// Frame `frame` of two lines crossing at a point, applied to `s` (frame 0 the first delta,
    /// drawn A, slot 1, then B, slot 2): content puts each through `(0, 0, lift)`, turned about z
    /// by `0.3 + 0.07 frame` rad (B square to A) and lifted to `lift = 0.3 + 0.37 frame` grid
    /// steps, as an orbit places its fixed-orientation mobjects again each frame; each rising along
    /// itself (A by 0.71 grid steps a unit, B falling by 0.53), so the snap moves the two apart at
    /// their crossing by a different fraction of a step each frame; coded on no plane, so on no
    /// identity.
    fn crossing_lines(s: &mut Stream, view: &View, frame: u32) {
        let turn = 0.3 + 0.07 * f64::from(frame);
        let lift = 0.3 + 0.37 * GRID * f64::from(frame);
        let (c, sn) = (libm::cos(turn), libm::sin(turn));
        let line = |dir: [f64; 2], rise: f64| {
            let at = |t: f64| [t * dir[0], t * dir[1], lift + rise * GRID * t];
            snapped(&[at(-2.0), at(-2.0 / 3.0), at(2.0 / 3.0), at(2.0)], GRID)
        };
        let curve = crate::depth::Planarity::Curve;
        let items: Vec<(Shape, Record)> = [(line([c, sn], 0.71), 1u32), (line([-sn, c], -0.53), 2)]
            .into_iter()
            .map(|(points, slot)| {
                let shape = Shape::new(ShapeData::Path { points });
                let mut r = line_record(slot, 6.0, curve);
                r.geometry.shape_a = Some(shape.key());
                (shape, r)
            })
            .collect();
        let delta = if frame == 0 {
            crate::probes::first_delta(view.clone(), &items, &[1, 2])
        } else {
            let mut d = Delta::new(
                Time {
                    num: i64::from(frame),
                    den: 30,
                },
                u64::from(frame),
            );
            for (shape, r) in &items {
                d.upload.push(shape.clone());
                d.set.push(r.clone());
            }
            d
        };
        s.apply(&delta).expect("the delta applies");
    }

    /// The lines of [`crossing_lines`] over sixteen frames, seen from straight above, so their keys
    /// part only by what the snap does to them: on no identity, each a stroke of a path with no
    /// plane of its own carrying its own finite share of its key, they tie wherever both cover in
    /// every frame, B, drawn later, above A at every sub-sample, so no order there flips from frame
    /// to frame, though the snap makes now one and now the other the nearer by its key.
    #[test]
    fn two_lines_crossing_on_no_identity_tie_in_every_frame_of_an_orbit() {
        let view = snapped_view(0.0, -90.0);
        let mut s = Stream::new();
        let mut nearer = [0usize; 2];
        for frame in 0..16u32 {
            crossing_lines(&mut s, &view, frame);
            let ids = identities(&s);
            assert!(
                ids[&101].is_empty() && ids[&102].is_empty(),
                "frame {frame}: A on {:?}, B on {:?}",
                ids[&101],
                ids[&102]
            );
            let both = met(&s, 101, 102);
            assert!(both.len() > 50, "frame {frame}: {} sub-samples", both.len());
            for m in &both {
                assert!(
                    [m.a.4, m.b.4].iter().all(|s| s.is_finite() && *s > 0.0),
                    "frame {frame}: shares {} and {}",
                    m.a.4,
                    m.b.4
                );
                assert!(!m.a_above, "frame {frame}: A above B at {:?}", m.at);
                if !order::tied(m.a.0, m.a.1, m.b.0, m.b.1) {
                    nearer[usize::from(m.a.0 < m.b.0)] += 1;
                }
            }
        }
        assert!(
            nearer[0] > 0 && nearer[1] > 0,
            "the keys put A nearer at {} sub-samples, B at {}",
            nearer[1],
            nearer[0]
        );
    }

    /// Two level lines along x seen from straight above, one over the other, A on z = 0 and B ten
    /// grid steps above it (the nearer), coded on no plane and so on no identity, B drawn first:
    /// apart by more than their own shares reach, depth orders them, B above A at every sub-sample
    /// both cover, though paint order would put A above.
    #[test]
    fn parallel_lines_apart_beyond_their_own_shares_keep_depth_order() {
        let view = snapped_view(0.0, -90.0);
        let (curve, up) = (crate::depth::Planarity::Curve, 10.0 * GRID);
        let s = paths_stream(
            view,
            vec![
                (
                    snapped_line([-2.0, 0.0, up], [2.0, 0.0, up]),
                    line_record(2, 6.0, curve),
                ),
                (
                    snapped_line([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
                    line_record(1, 6.0, curve),
                ),
            ],
        );
        let both = met(&s, 101, 102);
        assert!(both.len() > 1000, "{} sub-samples", both.len());
        let above = both.iter().filter(|m| m.a_above).count();
        assert_eq!(above, 0, "A above B at {above} of {}", both.len());
    }

    /// A line on none of a flat face's identities, a few times a stroke's own share of its key from
    /// it: the face a square 2 across on z = 0, the line across it two grid steps above (the nearer),
    /// no plane holding both within their tolerances (each half a step along z), coded on no plane
    /// (on the axis planes it holds exactly alone), drawn first. The face has no own share, so
    /// depth orders the two, the line above the face at every sub-sample both cover, though paint
    /// order would put the face above.
    #[test]
    fn a_line_against_a_flat_face_never_ties_by_its_own_share() {
        let view = snapped_view(0.0, -90.0);
        let up = 2.0 * GRID;
        let flat = Some(crate::depth::Planarity::Newell);
        let s = paths_stream(
            view,
            vec![
                (
                    snapped_line([-0.5, 0.2, up], [0.5, 0.2, up]),
                    line_record(2, 6.0, crate::depth::Planarity::Curve),
                ),
                (
                    flat_square(-1.0, 2.0),
                    face_record(1, [0.8, 0.3, 0.2, 1.0], None, flat),
                ),
            ],
        );
        let ids = identities(&s);
        assert!(
            !ids[&101].is_empty() && ids[&101].is_disjoint(&ids[&102]),
            "the face on {:?}, the line on {:?}",
            ids[&101],
            ids[&102]
        );
        let both = met(&s, 101, 102);
        assert!(both.len() > 500, "{} sub-samples", both.len());
        for m in &both {
            assert!(m.a.4 < 0.0 && m.b.4 > 0.0, "shares {} and {}", m.a.4, m.b.4);
            let reach = |(d, e, _, _, s): Read| e + (d.abs() + e) * s;
            assert!(
                (m.a.0 - m.b.0).abs() <= reach(m.b) * 5.0,
                "the keys {} and {} part by more than five times the line's own reach {}",
                m.a.0,
                m.b.0,
                reach(m.b)
            );
        }
        let above = both.iter().filter(|m| m.a_above).count();
        assert_eq!(
            above,
            0,
            "the face above the line at {above} of {}",
            both.len()
        );
    }

    /// Two lines seen from straight above, crossing on screen at the origin: A level on z = 0,
    /// along x; B along y, its control points one grid step above but the last, two steps above
    /// (so through 1.125 steps at the crossing, the nearer), both coded on no plane, so on no
    /// identity in common (A on the axis planes it holds exactly, y = 0 and z = 0, B on x = 0), B
    /// drawn first. Where both cover, their keys part by more than their own shares reach (half a
    /// step each along the sight axis), by less than a quarter more, so depth orders them, B above
    /// A at every sub-sample, though paint order would put A above.
    /// Each share is the stroke's own, finite; the share of a key B carries otherwise
    /// ([`composite::Fragment::snap`]), on no plane the camera keys (the eye lies in x = 0), is
    /// unbounded.
    #[test]
    fn crossing_lines_just_beyond_their_own_shares_keep_depth_order() {
        let view = snapped_view(0.0, -90.0);
        let curve = crate::depth::Planarity::Curve;
        let s = paths_stream(
            view,
            vec![
                (
                    snapped(
                        &[
                            [0.0, -2.0, GRID],
                            [0.0, -2.0 / 3.0, GRID],
                            [0.0, 2.0 / 3.0, GRID],
                            [0.0, 2.0, 2.0 * GRID],
                        ],
                        GRID,
                    ),
                    line_record(2, 6.0, curve),
                ),
                (
                    snapped_line([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
                    line_record(1, 6.0, curve),
                ),
            ],
        );
        let both = met(&s, 101, 102);
        assert!(both.len() > 50, "{} sub-samples", both.len());
        let reach = |(d, e, _, _, s): Read| e + (d.abs() + e) * s;
        for m in &both {
            let (gap, room) = ((m.a.0 - m.b.0).abs(), reach(m.a) + reach(m.b));
            assert!(
                gap > room && gap < 1.25 * room,
                "at {:?} the keys part by {gap:e} against their reach {room:e}",
                m.at
            );
            assert!(m.b.2 == f64::INFINITY, "B's snap share {}", m.b.2);
        }
        let above = both.iter().filter(|m| m.a_above).count();
        assert_eq!(above, 0, "A above B at {above} of {}", both.len());
    }

    /// A segment of the z axis (its points on x = 0 and y = 0, coded on y's plane) and a tick
    /// across it (on y = 0 and z = 0.5, coded on z's), as an axis and its ticks are drawn, seen
    /// over an orbit (the camera 70 degrees from z, turned 7 degrees about it a frame), the
    /// segment drawn first. Both lie on y = 0 exactly, so they share its identity and are read on
    /// its plane against each other, one depth there: the tick above the segment at every
    /// sub-sample both cover in every frame, though along their curves their keys part across the
    /// strokes' overlap, now one way and now the other as the camera turns.
    #[test]
    fn a_tick_and_its_axis_on_one_axis_plane_tie_in_every_frame_of_an_orbit() {
        let (on_y, on_z) = (
            crate::depth::Planarity::Axis(1),
            crate::depth::Planarity::Axis(2),
        );
        let (mut met_any, mut frames) = (0, Vec::new());
        for frame in 0..16u32 {
            let view = snapped_view(70.0, -110.0 + 7.0 * f64::from(frame));
            let s = paths_stream(
                view,
                vec![
                    (
                        snapped_line([0.0, 0.0, -1.0], [0.0, 0.0, 1.0]),
                        line_record(1, 6.0, on_y),
                    ),
                    (
                        snapped_line([-0.25, 0.0, 0.5], [0.25, 0.0, 0.5]),
                        line_record(2, 6.0, on_z),
                    ),
                ],
            );
            let both = met(&s, 101, 102);
            met_any += both.len();
            let above = both.iter().filter(|m| m.a_above).count();
            let ids = identities(&s);
            frames.push((above, both.len(), !ids[&101].is_disjoint(&ids[&102])));
        }
        assert!(met_any > 200, "{met_any} sub-samples");
        assert!(
            frames.iter().all(|(above, _, _)| *above == 0),
            "the segment above the tick, (at, of) a frame: {:?}",
            frames.iter().map(|f| (f.0, f.1)).collect::<Vec<_>>()
        );
        assert!(
            frames.iter().all(|(_, _, shared)| *shared),
            "an identity in common a frame: {:?}",
            frames.iter().map(|f| f.2).collect::<Vec<_>>()
        );
    }

    /// A segment of the z axis (its points on x = 0 and y = 0, coded on y's plane) and a tick across
    /// it on one of those two planes and on z = 0.5, seen over an orbit whose eye lies in the
    /// segment's other plane (theta -90 degrees for x = 0, 0 for y = 0; phi 56 to 86 degrees), the
    /// segment drawn first. The camera keys the plane through its eye nowhere, so the segment is read
    /// on the plane it shares with the tick alone, one depth there: the tick above the segment at
    /// every sub-sample both cover in every frame, whichever of the segment's two planes is its first
    /// identity. Turned 3 degrees off the plane through the eye, the camera keys both, nearly edge-on
    /// the one the eye left, and the segment's snap share is the most of the snap's reach on each of
    /// them: several times that of a line on the shared plane alone.
    #[test]
    fn a_line_on_two_axis_planes_one_holding_the_eye_is_read_on_the_other() {
        let axis = crate::depth::Planarity::Axis;
        let segment = || {
            (
                snapped_line([0.0, 0.0, -1.0], [0.0, 0.0, 1.0]),
                line_record(1, 6.0, axis(1)),
            )
        };
        // The eye's theta, the plane the tick shares with the segment, the tick, and a line on that
        // plane alone (through the origin, rising across it).
        for (theta, shared, tick, alone) in [
            (
                -90.0,
                1u8,
                ([-0.25, 0.0, 0.5], [0.25, 0.0, 0.5]),
                ([-0.5, 0.0, -1.0], [0.5, 0.0, 1.0]),
            ),
            (
                0.0,
                0u8,
                ([0.0, -0.25, 0.5], [0.0, 0.25, 0.5]),
                ([0.0, -0.5, -1.0], [0.0, 0.5, 1.0]),
            ),
        ] {
            let (mut met_any, mut frames) = (0, Vec::new());
            for frame in 0..16u32 {
                let view = snapped_view(56.0 + 2.0 * f64::from(frame), theta);
                let s = paths_stream(
                    view,
                    vec![
                        segment(),
                        (snapped_line(tick.0, tick.1), line_record(2, 6.0, axis(2))),
                    ],
                );
                let both = met(&s, 101, 102);
                met_any += both.len();
                frames.push((both.iter().filter(|m| m.a_above).count(), both.len()));
            }
            assert!(met_any > 200, "theta {theta}: {met_any} sub-samples");
            assert!(
                frames.iter().all(|f| f.0 == 0),
                "theta {theta}: the segment above the tick, (at, of) a frame: {frames:?}"
            );
            let s = paths_stream(
                snapped_view(75.0, theta + 3.0),
                vec![
                    segment(),
                    (
                        snapped_line(alone.0, alone.1),
                        line_record(2, 6.0, axis(shared)),
                    ),
                ],
            );
            let both = met(&s, 101, 102);
            assert!(both.len() > 20, "theta {theta}: {} sub-samples", both.len());
            for m in &both {
                assert!(
                    m.a.2.is_finite() && m.b.2.is_finite() && m.a.2 > 4.0 * m.b.2,
                    "theta {theta}: the segment's snap share {} against the line's {}",
                    m.a.2,
                    m.b.2
                );
            }
        }
    }

    /// Each sub-sample object `a` covers in `s`: its key there read along its curve (its own
    /// depths, whatever identities it lies on), the key's bound, and its fragment's own share at
    /// that pixel.
    // The sub-sample `k` indexes the fragment's coverage and depth rows.
    #[allow(clippy::needless_range_loop)]
    fn curve_keys(s: &Stream, a: u32) -> BTreeMap<(u32, u32, usize), (f64, f64, f64)> {
        let (w, h) = {
            let view = s.view.as_ref().expect("a view");
            (view.pixel_width, view.pixel_height)
        };
        let mut keys = BTreeMap::new();
        s.visit_contributors(0, 0, w, h, |x, y, frags, cov, d, _| {
            let Some(i) = frags.iter().position(|f| s.leaf_of(f).0 == a) else {
                return;
            };
            let f = &frags[i];
            for k in 0..composite::SAMPLES {
                if cov[i][k] != 0 {
                    keys.insert((x, y, k), (d[i].samples[k], d[i].err, f.own));
                }
            }
        })
        .expect("the stream draws");
        keys
    }

    /// A straight line of `view` through `centre`, `length` long, at `angle` degrees from the
    /// camera's sight axis (turned from it toward the screen's x), in CE's cubic layout: its
    /// control points at a third and two thirds, not snapped.
    fn line_off_sight(view: &View, centre: [f64; 3], length: f64, angle: f64) -> Vec<[f64; 3]> {
        let r = projection_of(view).rotation();
        let (c, s) = (libm::cos(angle.to_radians()), libm::sin(angle.to_radians()));
        let dir: [f64; 3] = std::array::from_fn(|j| c * r[2][j] + s * r[0][j]);
        let at = |t: f64| std::array::from_fn(|j| centre[j] + (t - 0.5) * length * dir[j]);
        vec![at(0.0), at(1.0 / 3.0), at(2.0 / 3.0), at(1.0)]
    }

    /// The snap's move of a stroke's key at fixed sub-samples, through the frame's own sampling
    /// (the key along the curve under each sub-sample, [`Centreline::key_under`]): a line in
    /// general position, at 90, 45, 20, 10 and 5 degrees from the sight axis, under a camera
    /// looking down z and two turned ones, drawn where content put it and where a snap within
    /// half a grid step of each of its control points' coordinates moved it (the corners of that
    /// box, alike and opposed between the points, and random moves in it). At every sub-sample
    /// both cover, the moved line's key lies within its reach of the line's own: its bound and
    /// the content's, and its own share at that pixel of its key with its bound
    /// ([`order::lines_term`]'s reach). Each row's tightness, the most the key moved over that
    /// reach, is printed.
    #[test]
    fn a_snapped_line_s_key_at_each_sub_sample_lies_within_its_own_reach_of_the_content_s() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let corners: Vec<[f64; 3]> = (0..8)
            .map(|c| [0, 1, 2].map(|k| if (c >> k) & 1 == 1 { 0.5 } else { -0.5 }))
            .collect();
        let mut moves: Vec<[[f64; 3]; 4]> = Vec::new();
        for c in &corners {
            let o = c.map(|v| -v);
            moves.extend([
                [*c, *c, *c, *c],
                [*c, *c, o, o],
                [*c, o, *c, o],
                [o, *c, *c, o],
            ]);
        }
        for _ in 0..32 {
            moves.push(std::array::from_fn(|_| std::array::from_fn(|_| next())));
        }
        let curve = crate::depth::Planarity::Curve;
        let mut rows = Vec::new();
        for (phi, theta) in [(0.0, -90.0), (70.0, -110.0), (85.0, -100.0)] {
            let view = snapped_view(phi, theta);
            for angle in [90.0, 45.0, 20.0, 10.0, 5.0] {
                let length = if angle < 15.0 { 4.0 } else { 2.0 };
                let content = line_off_sight(&view, [0.31, -0.17, 0.23], length, angle);
                let draw = |points: Vec<[f64; 3]>| {
                    let s = paths_stream(view.clone(), vec![(points, line_record(1, 6.0, curve))]);
                    curve_keys(&s, 101)
                };
                let exact = draw(content.clone());
                let (mut worst, mut met_any) = (0.0f64, 0usize);
                for m in &moves {
                    let moved: Vec<[f64; 3]> = (0..4)
                        .map(|i| std::array::from_fn(|j| content[i][j] + m[i][j] * GRID))
                        .collect();
                    for (at, (k, e, own)) in draw(moved) {
                        let Some((k0, e0, _)) = exact.get(&at) else {
                            continue;
                        };
                        met_any += 1;
                        assert!(own.is_finite() && own > 0.0, "own share {own}");
                        let reach = e + e0 + (k.abs() + e) * own;
                        let moved_by = (k - k0).abs();
                        assert!(
                            moved_by <= reach,
                            "({phi}, {theta}) at {angle} degrees, at {at:?}: the key moved by \
                             {moved_by:e} against its reach {reach:e}"
                        );
                        worst = worst.max(moved_by / reach);
                    }
                }
                assert!(
                    met_any > 1000,
                    "({phi}, {theta}) {angle}: {met_any} sub-samples"
                );
                rows.push((phi, theta, angle, worst, met_any));
            }
        }
        for (phi, theta, angle, worst, n) in rows {
            eprintln!(
                "TIGHT phi {phi} theta {theta} angle {angle}: {worst:.3} over {n} sub-samples"
            );
        }
    }

    /// Two lines content put on one line, steep to the sight axis (5 degrees from it, under a
    /// camera 70 degrees from z), one drawn as one cubic and the other as two halves, so the snap
    /// moves their control points apart though it keeps their ends together; shifted along their
    /// line frame by frame, so it moves them by a different fraction of a step each frame. On no
    /// identity, each a stroke of a path with no plane of its own, they tie wherever both cover,
    /// the one drawn later above at every sub-sample in every frame: the snap's move of their keys
    /// there, along the curves and sliding where their pieces move on screen, lies within their
    /// own shares.
    #[test]
    fn two_lines_on_one_steep_line_tie_in_every_frame_where_the_snap_slides_their_keys() {
        let view = snapped_view(70.0, -110.0);
        let curve = crate::depth::Planarity::Curve;
        let r = projection_of(&view).rotation();
        let (c, s) = (libm::cos(5f64.to_radians()), libm::sin(5f64.to_radians()));
        let dir: [f64; 3] = std::array::from_fn(|j| c * r[2][j] + s * r[0][j]);
        let mut frames = Vec::new();
        for frame in 0..16u32 {
            let shift = 0.0137 * f64::from(frame);
            let at = |t: f64| -> [f64; 3] {
                std::array::from_fn(|j| [0.31, -0.17, 0.23][j] + (t - 2.0 + shift) * dir[j])
            };
            let one = snapped(&[at(0.0), at(4.0 / 3.0), at(8.0 / 3.0), at(4.0)], GRID);
            let halves = snapped(
                &[
                    at(0.0),
                    at(2.0 / 3.0),
                    at(4.0 / 3.0),
                    at(2.0),
                    at(2.0),
                    at(8.0 / 3.0),
                    at(10.0 / 3.0),
                    at(4.0),
                ],
                GRID,
            );
            let s = paths_stream(
                view.clone(),
                vec![
                    (one, line_record(1, 6.0, curve)),
                    (halves, line_record(2, 6.0, curve)),
                ],
            );
            let ids = identities(&s);
            assert!(
                ids[&101].is_empty() && ids[&102].is_empty(),
                "frame {frame}: on {:?} and {:?}",
                ids[&101],
                ids[&102]
            );
            let both = met(&s, 101, 102);
            assert!(
                both.len() > 200,
                "frame {frame}: {} sub-samples",
                both.len()
            );
            frames.push((both.iter().filter(|m| m.a_above).count(), both.len()));
        }
        assert!(
            frames.iter().all(|f| f.0 == 0),
            "the line drawn first above, (at, of) a frame: {frames:?}"
        );
    }

    /// The snap's move of the key of a stroke cut into pieces 2 to 4 px long, as a line slanted
    /// to the sight axis and longer on screen than 128 px is (a cubic is cut into at most 64
    /// pieces), at fixed sub-samples, as for a bent stroke below: two such lines, each 4 long,
    /// meeting at 30, 90 and 150 degrees, the first 40, 50 or 65 degrees from the sight axis (the
    /// second turned from it toward the screen's y); and at each of those angles a thin shape seen
    /// almost edge-on, two such lines 0.01 apart along the sight axis and 0.0001 apart on screen
    /// joined at one end by a short piece, so a read past the stroke's body may change sides as
    /// the snap moves them. Each 6 and 0.3 wide, under a camera looking down z and a turned one,
    /// moved as below. At every sub-sample both cover the moved key lies within its reach of the
    /// content's, and in each row within a tenth of it somewhere; each row's worst and median
    /// move over reach are printed.
    #[test]
    fn a_snapped_stroke_of_short_slanted_pieces_keys_each_sub_sample_within_its_own_reach() {
        let mut seed = 0x6a09_e667_f3bc_c909u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let corners: Vec<[f64; 3]> = (0..8)
            .map(|c| [0, 1, 2].map(|k| if (c >> k) & 1 == 1 { 0.5 } else { -0.5 }))
            .collect();
        let curve = crate::depth::Planarity::Curve;
        let centre = [0.31, -0.17, 0.23];
        let mut rows = Vec::new();
        for (phi, theta) in [(0.0, -90.0), (70.0, -110.0)] {
            let view = snapped_view(phi, theta);
            let r = projection_of(&view).rotation();
            let third = |p: [f64; 3], q: [f64; 3], t: f64| -> [f64; 3] {
                std::array::from_fn(|j| p[j] + (q[j] - p[j]) * t)
            };
            // Each shape: its name, its distinct points, and its points as indices into them.
            let mut shapes: Vec<(String, Vec<[f64; 3]>, Vec<usize>)> = Vec::new();
            for slant in [40.0f64, 50.0, 65.0] {
                let (c, s) = (libm::cos(slant.to_radians()), libm::sin(slant.to_radians()));
                let u: [f64; 3] = std::array::from_fn(|j| c * r[2][j] + s * r[0][j]);
                for bend in [30.0f64, 90.0, 150.0] {
                    let (c, s) = (libm::cos(bend.to_radians()), libm::sin(bend.to_radians()));
                    let w: [f64; 3] = std::array::from_fn(|j| -c * u[j] + s * r[1][j]);
                    let a: [f64; 3] = std::array::from_fn(|j| centre[j] - 4.0 * u[j]);
                    let b: [f64; 3] = std::array::from_fn(|j| centre[j] + 4.0 * w[j]);
                    let distinct = vec![
                        a,
                        third(a, centre, 1.0 / 3.0),
                        third(a, centre, 2.0 / 3.0),
                        centre,
                        third(centre, b, 1.0 / 3.0),
                        third(centre, b, 2.0 / 3.0),
                        b,
                    ];
                    shapes.push((
                        format!("slant {slant} bent {bend}"),
                        distinct,
                        vec![0, 1, 2, 3, 3, 4, 5, 6],
                    ));
                }
                let a: [f64; 3] = std::array::from_fn(|j| centre[j] - 2.0 * u[j]);
                let b: [f64; 3] = std::array::from_fn(|j| centre[j] + 2.0 * u[j]);
                let apart: [f64; 3] = std::array::from_fn(|j| 0.01 * r[2][j] + 0.0001 * r[1][j]);
                let (b2, a2): ([f64; 3], [f64; 3]) = (
                    std::array::from_fn(|j| b[j] + apart[j]),
                    std::array::from_fn(|j| a[j] + apart[j]),
                );
                let distinct = vec![
                    a,
                    third(a, b, 1.0 / 3.0),
                    third(a, b, 2.0 / 3.0),
                    b,
                    third(b, b2, 1.0 / 3.0),
                    third(b, b2, 2.0 / 3.0),
                    b2,
                    third(b2, a2, 1.0 / 3.0),
                    third(b2, a2, 2.0 / 3.0),
                    a2,
                ];
                shapes.push((
                    format!("slant {slant} thin"),
                    distinct,
                    vec![0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9],
                ));
            }
            for (name, distinct, index) in &shapes {
                let n = distinct.len();
                let shared: Vec<bool> = (0..n)
                    .map(|i| index.iter().filter(|&&k| k == i).count() > 1)
                    .collect();
                let mut moves: Vec<Vec<[f64; 3]>> = Vec::new();
                for c in &corners {
                    let o = c.map(|v| -v);
                    moves.push(vec![*c; n]);
                    moves.push((0..n).map(|i| if i % 2 == 0 { *c } else { o }).collect());
                    moves.push((0..n).map(|i| if shared[i] { *c } else { o }).collect());
                }
                for _ in 0..16 {
                    moves.push((0..n).map(|_| std::array::from_fn(|_| next())).collect());
                }
                for width in [6.0, 0.3] {
                    let draw = |points: Vec<[f64; 3]>| {
                        let s = paths_stream(
                            view.clone(),
                            vec![(points, line_record(1, width, curve))],
                        );
                        curve_keys(&s, 101)
                    };
                    let exact = draw(index.iter().map(|&k| distinct[k]).collect());
                    let mut ratios: BTreeMap<(u32, u32, usize), f64> = BTreeMap::new();
                    for m in &moves {
                        let moved: Vec<[f64; 3]> = index
                            .iter()
                            .map(|&k| std::array::from_fn(|j| distinct[k][j] + m[k][j] * GRID))
                            .collect();
                        for (at, (k, e, own)) in draw(moved) {
                            let Some((k0, e0, _)) = exact.get(&at) else {
                                continue;
                            };
                            assert!(own.is_finite() && own > 0.0, "own share {own}");
                            let reach = e + e0 + (k.abs() + e) * own;
                            let moved_by = (k - k0).abs();
                            assert!(
                                moved_by <= reach,
                                "({phi}, {theta}) {name}, {width} wide, at {at:?}: the key moved \
                                 by {moved_by:e} against its reach {reach:e}"
                            );
                            let ratio = ratios.entry(at).or_insert(0.0);
                            *ratio = ratio.max(moved_by / reach);
                        }
                    }
                    let mut sorted: Vec<f64> = ratios.values().copied().collect();
                    sorted.sort_by(f64::total_cmp);
                    assert!(
                        sorted.len() > 200,
                        "({phi}, {theta}) {name}, {width} wide: {} sub-samples",
                        sorted.len()
                    );
                    let (worst, median) = (sorted[sorted.len() - 1], sorted[sorted.len() / 2]);
                    assert!(
                        worst >= 0.1,
                        "({phi}, {theta}) {name}, {width} wide: the reach is {:.1} times the \
                         largest move found",
                        1.0 / worst
                    );
                    rows.push((phi, theta, name.clone(), width, worst, median, sorted.len()));
                }
            }
        }
        for (phi, theta, name, width, worst, median, n) in rows {
            eprintln!(
                "TIGHT phi {phi} theta {theta} {name} width {width}: {worst:.3} at worst, \
                 {median:.3} at the median, over {n} sub-samples"
            );
        }
    }

    /// Each sub-sample a filled object `a`'s stroke covers in `s`: its key there as the resolve
    /// orders by it, the stroke drawn on its own fill taking the fill's key where nearer
    /// ([`composite::strokes_on_own_fills`]), the key's bound, and the stroke's own share.
    // The sub-sample `k` indexes the fragment's coverage and depth rows.
    #[allow(clippy::needless_range_loop)]
    fn resolved_keys(s: &Stream, a: u32) -> BTreeMap<(u32, u32, usize), (f64, f64, f64)> {
        let (w, h) = {
            let view = s.view.as_ref().expect("a view");
            (view.pixel_width, view.pixel_height)
        };
        let mut keys = BTreeMap::new();
        s.visit_contributors(0, 0, w, h, |x, y, frags, cov, d, _| {
            let Some(i) = frags
                .iter()
                .position(|f| s.leaf_of(f).0 == a && f.path_layer && f.sub % 3 == 2)
            else {
                return;
            };
            let mut d = d.to_vec();
            composite::strokes_on_own_fills(frags, &mut d, |j| cov[j]);
            for k in 0..composite::SAMPLES {
                if cov[i][k] != 0 {
                    keys.insert((x, y, k), (d[i].samples[k], d[i].err, frags[i].own));
                }
            }
        })
        .expect("the stream draws");
        keys
    }

    /// The snap's move of a filled stroke's key at fixed sub-samples, as the resolve orders by it
    /// (the stroke drawn on its own fill, read on the stroke's curve with no body, takes the fill's
    /// key where nearer: [`resolved_keys`]), moved as the bent stroke below: a thin shape seen
    /// almost edge-on at 40 and 65 degrees from the sight axis (two lines 4 long, 0.01 apart along
    /// it and 0.0001 apart on screen, closed at both ends); a folded outline (a side ending in a
    /// corner, a leg turning off it, and another part of the outline passing 0.0001 beyond the side
    /// and farther); and a square, convex, whose stroke's own share is the same filled as not.
    /// Each 6 and 0.3 wide, under a camera looking down z and a turned one. At every sub-sample
    /// the stroke covers in both, the moved key lies within its reach of the content's; each row's
    /// worst and median move over reach are printed.
    #[test]
    fn a_snapped_filled_stroke_s_resolved_key_lies_within_its_own_reach_of_the_content_s() {
        let mut seed = 0x243f_6a88_85a3_08d3u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let corners: Vec<[f64; 3]> = (0..8)
            .map(|c| [0, 1, 2].map(|k| if (c >> k) & 1 == 1 { 0.5 } else { -0.5 }))
            .collect();
        let curve = crate::depth::Planarity::Curve;
        let mut rows = Vec::new();
        for (phi, theta) in [(0.0, -90.0), (70.0, -110.0)] {
            let view = snapped_view(phi, theta);
            // Each shape: its name and its corners in the camera's axes (larger z nearer).
            let mut shapes: Vec<(String, Vec<[f64; 3]>)> = Vec::new();
            for slant in [40.0f64, 65.0] {
                let (c, s) = (libm::cos(slant.to_radians()), libm::sin(slant.to_radians()));
                let u = [s, 0.0, c];
                let at = |t: f64, y: f64, z: f64| [0.3 + t * u[0], -0.2 + y, 0.1 + t * u[2] + z];
                shapes.push((
                    format!("thin slant {slant}"),
                    vec![
                        at(-2.0, 0.0, 0.0),
                        at(2.0, 0.0, 0.0),
                        at(2.0, 0.0001, 0.01),
                        at(-2.0, 0.0001, 0.01),
                    ],
                ));
            }
            shapes.push((
                "folded".to_string(),
                vec![
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.05],
                    [0.0, 1.0, 0.0],
                    [0.5, 1.0, 0.0],
                    [0.5, 0.0001, -0.05],
                    [-1.0, 0.0001, -0.05],
                ],
            ));
            shapes.push((
                "square".to_string(),
                vec![
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.2],
                    [1.0, 1.0, 0.0],
                    [-1.0, 1.0, -0.2],
                ],
            ));
            for (name, corners_of) in &shapes {
                let ring = camera_ring(&view, corners_of);
                let n = ring.len();
                let mut moves: Vec<Vec<[f64; 3]>> = Vec::new();
                for c in &corners {
                    let o = c.map(|v| -v);
                    moves.push(vec![*c; n]);
                    moves.push((0..n).map(|i| if i % 2 == 0 { *c } else { o }).collect());
                    moves.push((0..n).map(|i| if i < n / 2 { *c } else { o }).collect());
                }
                for _ in 0..16 {
                    moves.push((0..n).map(|_| std::array::from_fn(|_| next())).collect());
                }
                for width in [6.0, 0.3] {
                    let record = |filled: bool| {
                        let mut r = line_record(1, width, curve);
                        if filled {
                            r.paint.fill = Paint::One([0.2, 0.4, 0.6, 1.0]);
                        }
                        r
                    };
                    let draw = |points: Vec<[f64; 3]>| {
                        let s = paths_stream(view.clone(), vec![(points, record(true))]);
                        resolved_keys(&s, 101)
                    };
                    let exact = draw(ring.clone());
                    if name == "square" {
                        let s = paths_stream(view.clone(), vec![(ring.clone(), record(false))]);
                        let bare = resolved_keys(&s, 101);
                        for (at, (_, _, own)) in &exact {
                            if let Some((_, _, alone)) = bare.get(at) {
                                assert_eq!(
                                    own, alone,
                                    "({phi}, {theta}) square, {width} wide, at {at:?}: the fill \
                                     moved a convex outline's own share"
                                );
                            }
                        }
                    }
                    let mut ratios: BTreeMap<(u32, u32, usize), f64> = BTreeMap::new();
                    for m in &moves {
                        let moved: Vec<[f64; 3]> = ring
                            .iter()
                            .zip(m)
                            .map(|(p, d)| std::array::from_fn(|j| p[j] + d[j] * GRID))
                            .collect();
                        for (at, (k, e, own)) in draw(moved) {
                            let Some((k0, e0, _)) = exact.get(&at) else {
                                continue;
                            };
                            assert!(own.is_finite() && own > 0.0, "own share {own}");
                            let reach = e + e0 + (k.abs() + e) * own;
                            let moved_by = (k - k0).abs();
                            assert!(
                                moved_by <= reach,
                                "({phi}, {theta}) {name}, {width} wide, at {at:?}: the key moved \
                                 by {moved_by:e} against its reach {reach:e}"
                            );
                            let ratio = ratios.entry(at).or_insert(0.0);
                            *ratio = ratio.max(moved_by / reach);
                        }
                    }
                    let mut sorted: Vec<f64> = ratios.values().copied().collect();
                    sorted.sort_by(f64::total_cmp);
                    assert!(
                        sorted.len() > 200,
                        "({phi}, {theta}) {name}, {width} wide: {} sub-samples",
                        sorted.len()
                    );
                    let (worst, median) = (sorted[sorted.len() - 1], sorted[sorted.len() / 2]);
                    rows.push((phi, theta, name.clone(), width, worst, median, sorted.len()));
                }
            }
        }
        for (phi, theta, name, width, worst, median, n) in rows {
            eprintln!(
                "TIGHT filled phi {phi} theta {theta} {name} width {width}: {worst:.3} at worst, \
                 {median:.3} at the median, over {n} sub-samples"
            );
        }
    }

    /// The snap's move of a bent stroke's key at fixed sub-samples, through the frame's own
    /// sampling, as for a straight line above: two straight cubics meeting at 30, 90 and 150
    /// degrees (the first 45 or 10 degrees from the sight axis, the second turned toward the
    /// screen's y), and a closed rectangle outline (its sides along a direction 45 degrees from
    /// the sight axis and along the screen's y), each 6 and 0.5 wide, under the three cameras;
    /// drawn where content put it and where a snap within half a grid step of each of its points'
    /// coordinates moved it, a point two cubics share moved as one (the box's corners alike,
    /// opposed between alternate points and between the shared points and the rest, and random
    /// moves). At a joint or a corner a read changes segment as the snap moves the stroke (the
    /// holder-set jump), and past a thin stroke's body near a corner the nearest segment, clamped
    /// at its end, keys it. Each row's tightness is printed.
    #[test]
    fn a_snapped_bent_stroke_s_key_at_each_sub_sample_lies_within_its_own_reach_of_the_content_s() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let corners: Vec<[f64; 3]> = (0..8)
            .map(|c| [0, 1, 2].map(|k| if (c >> k) & 1 == 1 { 0.5 } else { -0.5 }))
            .collect();
        let curve = crate::depth::Planarity::Curve;
        let centre = [0.31, -0.17, 0.23];
        let mut rows = Vec::new();
        for (phi, theta) in [(0.0, -90.0), (70.0, -110.0), (85.0, -100.0)] {
            let view = snapped_view(phi, theta);
            let r = projection_of(&view).rotation();
            let along = |angle: f64| -> [f64; 3] {
                let (c, s) = (libm::cos(angle.to_radians()), libm::sin(angle.to_radians()));
                std::array::from_fn(|j| c * r[2][j] + s * r[0][j])
            };
            let third = |p: [f64; 3], q: [f64; 3], t: f64| -> [f64; 3] {
                std::array::from_fn(|j| p[j] + (q[j] - p[j]) * t)
            };
            // Each shape: its name, its distinct points, and its points as indices into them.
            let mut shapes: Vec<(String, Vec<[f64; 3]>, Vec<usize>)> = Vec::new();
            for first in [45.0, 10.0] {
                let u = along(first);
                let length = if first < 15.0 { 2.0 } else { 1.0 };
                for bend in [30.0f64, 90.0, 150.0] {
                    let (c, s) = (libm::cos(bend.to_radians()), libm::sin(bend.to_radians()));
                    let w: [f64; 3] = std::array::from_fn(|j| -c * u[j] + s * r[1][j]);
                    let a: [f64; 3] = std::array::from_fn(|j| centre[j] - length * u[j]);
                    let b: [f64; 3] = std::array::from_fn(|j| centre[j] + length * w[j]);
                    let distinct = vec![
                        a,
                        third(a, centre, 1.0 / 3.0),
                        third(a, centre, 2.0 / 3.0),
                        centre,
                        third(centre, b, 1.0 / 3.0),
                        third(centre, b, 2.0 / 3.0),
                        b,
                    ];
                    shapes.push((
                        format!("bent {bend} from {first}"),
                        distinct,
                        vec![0, 1, 2, 3, 3, 4, 5, 6],
                    ));
                }
            }
            let (u, v) = (along(45.0), r[1]);
            let corner = |x: f64, y: f64| -> [f64; 3] {
                std::array::from_fn(|j| centre[j] + x * u[j] + y * v[j])
            };
            let ring = [
                corner(-1.0, -0.6),
                corner(1.0, -0.6),
                corner(1.0, 0.6),
                corner(-1.0, 0.6),
            ];
            let (mut distinct, mut index) = (Vec::new(), Vec::new());
            for (k, &p) in ring.iter().enumerate() {
                let q = ring[(k + 1) % 4];
                distinct.extend([p, third(p, q, 1.0 / 3.0), third(p, q, 2.0 / 3.0)]);
                index.extend([3 * k, 3 * k + 1, 3 * k + 2, (3 * k + 3) % 12]);
            }
            shapes.push(("rectangle".to_string(), distinct, index));
            for (name, distinct, index) in &shapes {
                let n = distinct.len();
                let shared: Vec<bool> = (0..n)
                    .map(|i| index.iter().filter(|&&k| k == i).count() > 1)
                    .collect();
                let mut moves: Vec<Vec<[f64; 3]>> = Vec::new();
                for c in &corners {
                    let o = c.map(|v| -v);
                    moves.push(vec![*c; n]);
                    moves.push((0..n).map(|i| if i % 2 == 0 { *c } else { o }).collect());
                    moves.push((0..n).map(|i| if shared[i] { *c } else { o }).collect());
                }
                for _ in 0..16 {
                    moves.push((0..n).map(|_| std::array::from_fn(|_| next())).collect());
                }
                for width in [6.0, 0.5] {
                    let draw = |points: Vec<[f64; 3]>| {
                        let s = paths_stream(
                            view.clone(),
                            vec![(points, line_record(1, width, curve))],
                        );
                        curve_keys(&s, 101)
                    };
                    let exact = draw(index.iter().map(|&k| distinct[k]).collect());
                    let (mut worst, mut met_any) = (0.0f64, 0usize);
                    for m in &moves {
                        let moved: Vec<[f64; 3]> = index
                            .iter()
                            .map(|&k| std::array::from_fn(|j| distinct[k][j] + m[k][j] * GRID))
                            .collect();
                        for (at, (k, e, own)) in draw(moved) {
                            let Some((k0, e0, _)) = exact.get(&at) else {
                                continue;
                            };
                            met_any += 1;
                            assert!(own.is_finite() && own > 0.0, "own share {own}");
                            let reach = e + e0 + (k.abs() + e) * own;
                            let moved_by = (k - k0).abs();
                            assert!(
                                moved_by <= reach,
                                "({phi}, {theta}) {name}, {width} wide, at {at:?}: the key moved \
                                 by {moved_by:e} against its reach {reach:e}"
                            );
                            worst = worst.max(moved_by / reach);
                        }
                    }
                    assert!(
                        met_any > 200,
                        "({phi}, {theta}) {name}, {width} wide: {met_any} sub-samples"
                    );
                    rows.push((phi, theta, name.clone(), width, worst, met_any));
                }
            }
        }
        for (phi, theta, name, width, worst, n) in rows {
            eprintln!(
                "TIGHT phi {phi} theta {theta} {name} width {width}: {worst:.3} over {n} sub-samples"
            );
        }
    }

    /// As the segment and tick above at theta -90 degrees (the eye in x = 0), the segment's
    /// first identity now the plane through the eye: a short line along y on x = 0 and z = -0.8,
    /// coded on z's plane, comes first in the registry's order and founds x = 0 before the tick
    /// founds y = 0, so the segment numbers x = 0 first and y = 0 after it (seen 3 degrees off
    /// the plane through the eye, where the camera keys both and its fragments carry both). The
    /// camera keys x = 0 nowhere at theta -90; the segment is read on the plane it shares with
    /// the tick, the tick above it at every sub-sample both cover in every frame.
    #[test]
    fn a_line_whose_first_identity_holds_the_eye_is_read_on_the_plane_it_shares() {
        let axis = crate::depth::Planarity::Axis;
        let items = || {
            vec![
                (
                    snapped_line([0.0, 0.0, -1.0], [0.0, 0.0, 1.0]),
                    line_record(1, 6.0, axis(1)),
                ),
                (
                    snapped_line([-0.25, 0.0, 0.5], [0.25, 0.0, 0.5]),
                    line_record(2, 6.0, axis(2)),
                ),
                (
                    snapped_line([0.0, -0.1, -0.8], [0.0, 0.1, -0.8]),
                    line_record(3, 6.0, axis(2)),
                ),
            ]
        };
        let ids = identities(&paths_stream(snapped_view(75.0, -87.0), items()));
        let first = ids[&101].iter().next().copied();
        assert!(
            ids[&101].len() == 2
                && first.is_some_and(|i| ids[&103].contains(&i) && !ids[&102].contains(&i)),
            "the segment on {:?}, the tick on {:?}, the line on x = 0 on {:?}",
            ids[&101],
            ids[&102],
            ids[&103]
        );
        let (mut met_any, mut frames) = (0, Vec::new());
        for frame in 0..16u32 {
            let view = snapped_view(56.0 + 2.0 * f64::from(frame), -90.0);
            let s = paths_stream(view, items());
            let both = met(&s, 101, 102);
            met_any += both.len();
            frames.push((both.iter().filter(|m| m.a_above).count(), both.len()));
        }
        assert!(met_any > 200, "{met_any} sub-samples");
        assert!(
            frames.iter().all(|f| f.0 == 0),
            "the segment above the tick, (at, of) a frame: {frames:?}"
        );
    }
}
