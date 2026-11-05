//! XF read-back as Python values, copied from the delta rather than inferred from its bytes.

use crate::xf::*;
use numpy::{Element, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

fn key<'py>(py: Python<'py>, key: Key) -> Bound<'py, PyBytes> {
    PyBytes::new(py, &key.0)
}

fn point(p: [f64; 3]) -> (f64, f64, f64) {
    (p[0], p[1], p[2])
}

fn rows<'py, T: Element + Copy, const N: usize>(
    py: Python<'py>,
    values: &[[T; N]],
) -> PyResult<Bound<'py, PyArray2<T>>> {
    PyArray1::from_vec(py, values.iter().flatten().copied().collect()).reshape([values.len(), N])
}

pub(super) fn order(entries: &[OrderEntry]) -> Vec<(&'static str, u32, u32)> {
    entries
        .iter()
        .map(|e| {
            let layer = match e.layer {
                Layer::Static => "static",
                Layer::Moving => "moving",
                Layer::Overlay => "overlay",
            };
            (layer, e.slot, e.part)
        })
        .collect()
}

fn view<'py>(py: Python<'py>, v: &View) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("pixel_width", v.pixel_width)?;
    out.set_item("pixel_height", v.pixel_height)?;
    out.set_item("frame_width", v.frame_width)?;
    out.set_item("frame_height", v.frame_height)?;
    out.set_item("frame_center", point(v.frame_center))?;
    out.set_item("mapping_centre", point(v.mapping_centre))?;
    match v.background {
        Background::Color(c) => out.set_item("background", (c[0], c[1], c[2], c[3]))?,
        Background::Image(k) => out.set_item("background", key(py, k))?,
    }
    match &v.camera {
        Camera::Flat => out.set_item("camera", "flat")?,
        Camera::Moving => out.set_item("camera", "moving")?,
        Camera::ThreeD(c) => {
            out.set_item("camera", "three_d")?;
            out.set_item("phi", c.phi)?;
            out.set_item("theta", c.theta)?;
            out.set_item("gamma", c.gamma)?;
            out.set_item("focal_distance", c.focal_distance)?;
            out.set_item("zoom", c.zoom)?;
            out.set_item("exponential_projection", c.exponential_projection)?;
            out.set_item("light_source", point(c.light_source))?;
            out.set_item("shading_factor", c.shading_factor)?;
            out.set_item("should_apply_shading", c.should_apply_shading)?;
        }
    }
    out.set_item("cairo_line_width_multiple", v.cairo_line_width_multiple)?;
    out.set_item("use_z_index", v.use_z_index)?;
    out.set_item("light_model", v.light.code_and_params())?;
    if v.input_grid != 0.0 {
        out.set_item("input_grid", v.input_grid)?;
    }
    Ok(out)
}

fn paint<'py>(py: Python<'py>, p: &Paint) -> PyResult<Bound<'py, PyAny>> {
    Ok(match p {
        Paint::None => py.None().into_bound(py),
        Paint::One(c) => rows(py, &[*c])?.into_any(),
        Paint::Two(cs) => rows(py, cs)?.into_any(),
        Paint::Table(k) => key(py, *k).into_any(),
    })
}

fn paints(out: &Bound<'_, PyDict>, p: &PaintB) -> PyResult<()> {
    let py = out.py();
    out.set_item("fill", paint(py, &p.fill)?)?;
    out.set_item("stroke", paint(py, &p.stroke)?)?;
    out.set_item("background_stroke", paint(py, &p.background_stroke)?)?;
    out.set_item("stroke_width", p.stroke_width)?;
    out.set_item("background_stroke_width", p.background_stroke_width)
}

fn record<'py>(py: Python<'py>, r: &Record) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("features", r.features)?;
    if let RecordKind::Unknown(_) = r.kind {
        if let Body::Opaque(payload) = &r.body {
            out.set_item("payload", PyBytes::new(py, payload))?;
        }
        return Ok(out);
    }
    if let Some(k) = r.geometry.shape_a {
        out.set_item("shape", key(py, k))?;
    }
    out.set_item("place", rows(py, &r.geometry.place_a)?)?;
    if let Some(b) = &r.geometry.blend {
        out.set_item("blend", (key(py, b.shape_b), rows(py, &b.place_b)?, b.t))?;
    }
    if let Some(window) = r.geometry.reveal {
        out.set_item("reveal", window)?;
    }
    let p = &r.paint;
    out.set_item("fill", paint(py, &p.fill)?)?;
    out.set_item("stroke", paint(py, &p.stroke)?)?;
    out.set_item("background_stroke", paint(py, &p.background_stroke)?)?;
    out.set_item("stroke_width", p.stroke_width)?;
    out.set_item("background_stroke_width", p.background_stroke_width)?;
    if let Some([a, b]) = p.gradient {
        out.set_item("gradient", (a[0], a[1], a[2], b[0], b[1], b[2]))?;
    }
    out.set_item(
        "joint",
        match p.joint {
            Joint::Auto => "auto",
            Joint::Round => "round",
            Joint::Bevel => "bevel",
            Joint::Miter => "miter",
        },
    )?;
    out.set_item(
        "cap",
        match p.cap {
            Cap::Auto => "auto",
            Cap::Round => "round",
            Cap::Butt => "butt",
            Cap::Square => "square",
        },
    )?;
    if let Some(b) = &p.paint_b {
        let block = PyDict::new(py);
        paints(&block, b)?;
        out.set_item("paint_b", block)?;
    }
    out.set_item("mix", p.mix)?;
    if let Some(d) = r.dash {
        out.set_item("dash", (d.period, d.duty, d.phase))?;
    }
    out.set_item(
        "space",
        match r.depth.space {
            Space::Screen => "screen",
            Space::World => "world",
            Space::Overlay => "overlay",
        },
    )?;
    out.set_item("z_index", r.depth.z_index)?;
    out.set_item("sort_point", point(r.depth.sort_point))?;
    out.set_item("shade_in_3d", r.depth.shade_in_3d)?;
    if let Some(c) = r.depth.fixed_orientation_centre {
        out.set_item("fixed_orientation_centre", point(c))?;
    }
    if let Some((k, place)) = &r.clip {
        out.set_item("clip", (key(py, *k), rows(py, place)?))?;
    }
    out.set_item("opacity", r.opacity)?;
    out.set_item("layer", r.layer)?;
    if let Some(k) = r.fill_image {
        out.set_item("fill_image", key(py, k))?;
    }
    if let Some(planes) = &r.planes {
        let codes: Vec<u32> = planes.iter().map(|c| u32::from(*c)).collect();
        out.set_item("planes", PyTuple::new(py, codes)?)?;
    }
    if let Some(normals) = &r.normals {
        let rows = normals
            .iter()
            .map(|n| PyTuple::new(py, n))
            .collect::<PyResult<Vec<_>>>()?;
        out.set_item("normals", PyTuple::new(py, rows)?)?;
    }
    if let Some(p) = &r.pre_projected {
        let projected = PyDict::new(py);
        projected.set_item("points", key(py, p.points))?;
        match p.gradient {
            Some([a, b]) => projected.set_item("gradient", (a[0], a[1], a[2], b[0], b[1], b[2]))?,
            None => projected.set_item("gradient", py.None())?,
        }
        projected.set_item("fill", paint(py, &p.fill)?)?;
        projected.set_item("stroke", paint(py, &p.stroke)?)?;
        projected.set_item("background_stroke", paint(py, &p.background_stroke)?)?;
        out.set_item("pre_projected", projected)?;
    }
    match &r.body {
        Body::None => {}
        Body::FaceGroup(f) => {
            out.set_item("uv", rows(py, &f.uv)?)?;
            out.set_item("grid", (f.grid[0], f.grid[1]))?;
            out.set_item("face_fill", key(py, f.fill))?;
            if let Some(k) = f.stroke {
                out.set_item("face_stroke", key(py, k))?;
            }
        }
        Body::Points(p) => {
            out.set_item("colors", key(py, p.colors))?;
            out.set_item("radius", p.radius)?;
        }
        Body::Image(source) => {
            let (corners, resampling, opacity) = match source {
                ImageSource::Picture {
                    corners,
                    resampling,
                    opacity,
                } => (corners, resampling, opacity),
                ImageSource::SubFrame {
                    rank,
                    corners,
                    resampling,
                    opacity,
                    view: v,
                    order: entries,
                } => {
                    out.set_item("sub_frame", (*rank, view(py, v)?, order(entries)))?;
                    (corners, resampling, opacity)
                }
            };
            out.set_item("corners", rows(py, corners)?)?;
            out.set_item("resampling", *resampling)?;
            out.set_item("image_opacity", *opacity)?;
        }
        Body::Instances(i) => {
            let data = i.placements.iter().flatten().flatten().copied().collect();
            out.set_item(
                "placements",
                PyArray1::from_vec(py, data).reshape([i.placements.len(), 3, 4])?,
            )?;
            out.set_item("instance_fill", key(py, i.fill))?;
            out.set_item("instance_stroke", key(py, i.stroke))?;
            out.set_item("objects", &i.objects)?;
        }
        Body::Opaque(payload) => out.set_item("payload", PyBytes::new(py, payload))?,
    }
    Ok(out)
}

pub(super) fn delta<'py>(py: Python<'py>, d: &Delta) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("version", d.version)?;
    out.set_item("t", (d.t.num, d.t.den))?;
    out.set_item("seq", d.seq)?;
    match &d.view {
        Some(v) => out.set_item("view", view(py, v)?)?,
        None => out.set_item("view", py.None())?,
    }
    let objects: Vec<_> = d
        .objects
        .iter()
        .map(|o| (o.object, &o.stable_id, &o.source))
        .collect();
    out.set_item("objects", objects)?;
    let upload = PyList::empty(py);
    for shape in &d.upload {
        let (kind, data) = match shape.data() {
            ShapeData::Path { points } => ("path".to_string(), rows(py, points)?.into_any()),
            ShapeData::Mesh {
                positions,
                triangles,
                normals,
            } => {
                let data = PyDict::new(py);
                data.set_item("positions", rows(py, positions)?)?;
                data.set_item("triangles", rows(py, triangles)?)?;
                match normals {
                    Some(n) => data.set_item("normals", rows(py, n)?)?,
                    None => data.set_item("normals", py.None())?,
                }
                ("mesh".to_string(), data.into_any())
            }
            ShapeData::Image {
                width,
                height,
                rgba,
            } => {
                let data = PyArray1::from_vec(py, rgba.clone()).reshape([
                    *height as usize,
                    *width as usize,
                    4,
                ])?;
                ("image".to_string(), data.into_any())
            }
            ShapeData::Reserved { code, payload } => {
                let name = match code {
                    4 => "volume".to_string(),
                    5 => "field".to_string(),
                    code => format!("kind{code}"),
                };
                (name, PyBytes::new(py, payload).into_any())
            }
        };
        upload.append((kind, key(py, shape.key()), data))?;
    }
    out.set_item("upload", upload)?;
    let upload_paints = PyList::empty(py);
    for table in &d.upload_paints {
        upload_paints.append((key(py, table.key()), rows(py, table.rows())?))?;
    }
    out.set_item("upload_paints", upload_paints)?;
    let released: Vec<_> = d.release.iter().map(|k| key(py, *k)).collect();
    out.set_item("release", released)?;
    let set = PyList::empty(py);
    for r in &d.set {
        set.append((r.kind.name(), r.slot, r.object, record(py, r)?))?;
    }
    out.set_item("set", set)?;
    out.set_item("remove", &d.remove)?;
    match &d.order {
        Some(entries) => out.set_item("order", order(entries))?,
        None => out.set_item("order", py.None())?,
    }
    Ok(out)
}
