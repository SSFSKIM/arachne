//! The Python binding, `arachne_core._core`, re-exported by the `arachne_core` package.

use crate::stream::{DrawError, Stream, StreamError, Unsupported};
use crate::xf::wire::{self, XfError};
use crate::xf::*;
use numpy::{
    AllowTypeChange, PyArray1, PyArrayLike1, PyArrayLike2, PyArrayLike3, PyArrayLikeDyn,
    PyArrayMethods,
};
use pyo3::call::PyCallArgs;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

mod contents;

// ---------------------------------------------------------------------------------------------
// Errors: the classes live in `arachne_core/_errors.py`.

fn python_error<'py, A: PyCallArgs<'py>>(py: Python<'py>, class: &str, args: A) -> PyErr {
    match py
        .import("arachne_core._errors")
        .and_then(|m| m.getattr(class))
        .and_then(|cls| cls.call1(args))
    {
        Ok(exception) => PyErr::from_value(exception),
        Err(err) => err,
    }
}

fn xf_error(py: Python<'_>, message: String) -> PyErr {
    python_error(py, "XfError", (message,))
}

fn wire_error(py: Python<'_>, err: XfError) -> PyErr {
    xf_error(py, err.to_string())
}

fn stream_error(py: Python<'_>, err: StreamError) -> PyErr {
    xf_error(py, err.to_string())
}

fn draw_error(py: Python<'_>, err: DrawError) -> PyErr {
    match err {
        DrawError::Unsupported(Unsupported {
            kind,
            slot,
            object,
            reason,
        }) => python_error(py, "Unsupported", (kind, slot, object, reason)),
        threads @ DrawError::Threads(_) => PyRuntimeError::new_err(threads.to_string()),
        other => xf_error(py, other.to_string()),
    }
}

// ---------------------------------------------------------------------------------------------
// Reading keyword arguments.

/// Keyword arguments taken one by one; whatever is left at the end is an error.
struct Fields<'py> {
    what: String,
    dict: Bound<'py, PyDict>,
}

impl<'py> Fields<'py> {
    fn new(what: String, dict: Option<&Bound<'py, PyDict>>, py: Python<'py>) -> PyResult<Self> {
        let dict = match dict {
            Some(d) => d.copy()?,
            None => PyDict::new(py),
        };
        Ok(Fields { what, dict })
    }

    fn take(&self, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        let value = self.dict.get_item(name)?;
        if value.is_some() {
            self.dict.del_item(name)?;
        }
        Ok(value.filter(|v| !v.is_none()))
    }

    fn get<T: FromPyObjectOwned<'py>>(&self, name: &str) -> PyResult<Option<T>> {
        self.take(name)?
            .map(|v| v.extract::<T>().map_err(|e| self.bad(name, e.into())))
            .transpose()
    }

    fn or<T: FromPyObjectOwned<'py>>(&self, name: &str, default: T) -> PyResult<T> {
        Ok(self.get(name)?.unwrap_or(default))
    }

    fn need<T: FromPyObjectOwned<'py>>(&self, name: &str) -> PyResult<T> {
        self.get(name)?.ok_or_else(|| {
            PyTypeError::new_err(format!("{} needs the keyword argument '{name}'", self.what))
        })
    }

    fn bad(&self, name: &str, err: PyErr) -> PyErr {
        PyTypeError::new_err(format!("{}: '{name}': {err}", self.what))
    }

    fn finish(self) -> PyResult<()> {
        if let Some((name, _)) = self.dict.iter().next() {
            return Err(PyTypeError::new_err(format!(
                "{} got an unexpected keyword argument '{name}'",
                self.what
            )));
        }
        Ok(())
    }
}

fn key_of(value: &Bound<'_, PyAny>) -> PyResult<Key> {
    let bytes: Vec<u8> = value.extract()?;
    let array: [u8; 16] = bytes
        .try_into()
        .map_err(|_| PyValueError::new_err("a key is 16 bytes"))?;
    Ok(Key(array))
}

fn key_field(fields: &Fields, name: &str) -> PyResult<Option<Key>> {
    fields.take(name)?.map(|v| key_of(&v)).transpose()
}

fn need_key(fields: &Fields, name: &str) -> PyResult<Key> {
    key_field(fields, name)?.ok_or_else(|| {
        PyTypeError::new_err(format!(
            "{} needs the keyword argument '{name}'",
            fields.what
        ))
    })
}

fn paint_of(value: Option<Bound<'_, PyAny>>) -> PyResult<Paint> {
    let Some(value) = value else {
        return Ok(Paint::None);
    };
    if value.is_instance_of::<PyBytes>() {
        return Ok(Paint::Table(key_of(&value)?));
    }
    if let Ok(one) = value.extract::<[f64; 4]>() {
        return Ok(Paint::One(one));
    }
    let rows: Vec<[f64; 4]> = value.extract().map_err(|_| {
        PyTypeError::new_err("a paint is an RGBA colour, one or two of them, or a paint table key")
    })?;
    match rows.as_slice() {
        [one] => Ok(Paint::One(*one)),
        [a, b] => Ok(Paint::Two([*a, *b])),
        _ => Err(PyValueError::new_err(
            "an inline paint has one or two stops; upload a table for more",
        )),
    }
}

fn named<T: Copy>(what: &str, value: &str, table: &[(&str, T)]) -> PyResult<T> {
    table
        .iter()
        .find(|(name, _)| *name == value)
        .map(|(_, v)| *v)
        .ok_or_else(|| {
            let names: Vec<&str> = table.iter().map(|(n, _)| *n).collect();
            PyValueError::new_err(format!(
                "{what} '{value}' is not one of {}",
                names.join(", ")
            ))
        })
}

fn layer_of(value: &Bound<'_, PyAny>) -> PyResult<Layer> {
    let name: String = value.extract()?;
    named(
        "layer",
        &name,
        &[
            ("static", Layer::Static),
            ("moving", Layer::Moving),
            ("overlay", Layer::Overlay),
        ],
    )
}

fn order_of(value: &Bound<'_, PyAny>) -> PyResult<Vec<OrderEntry>> {
    value
        .try_iter()?
        .map(|entry| {
            let entry = entry?;
            let (layer, slot, part): (Bound<PyAny>, u32, u32) = entry.extract()?;
            Ok(OrderEntry {
                layer: layer_of(&layer)?,
                slot,
                part,
            })
        })
        .collect()
}

fn rows3(value: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<[f64; 3]>> {
    let array: PyArrayLike2<f64, AllowTypeChange> = value.extract()?;
    let view = array.as_array();
    if view.ncols() != 3 {
        return Err(PyValueError::new_err(format!(
            "{what} must have shape (n, 3), not {:?}",
            view.shape()
        )));
    }
    Ok(view
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect())
}

fn view_of(fields: Fields) -> PyResult<View> {
    let camera: String = fields.or("camera", "flat".to_string())?;
    let camera = match camera.as_str() {
        "flat" => Camera::Flat,
        "moving" => Camera::Moving,
        "three_d" => Camera::ThreeD(ThreeDCamera {
            phi: fields.or("phi", 0.0)?,
            theta: fields.or("theta", -std::f64::consts::FRAC_PI_2)?,
            gamma: fields.or("gamma", 0.0)?,
            focal_distance: fields.or("focal_distance", 20.0)?,
            zoom: fields.or("zoom", 1.0)?,
            exponential_projection: fields.or("exponential_projection", false)?,
            light_source: fields.or("light_source", [-7.0, -9.0, 10.0])?,
            shading_factor: fields.or("shading_factor", 0.2)?,
            should_apply_shading: fields.or("should_apply_shading", true)?,
        }),
        other => {
            return Err(PyValueError::new_err(format!(
                "camera '{other}' is not one of flat, moving, three_d"
            )));
        }
    };
    let background = match fields.take("background")? {
        None => Background::Color([0.0, 0.0, 0.0, 1.0]),
        Some(v) if v.is_instance_of::<PyBytes>() => Background::Image(key_of(&v)?),
        Some(v) => Background::Color(v.extract()?),
    };
    let light = match fields.take("light_model")? {
        None => LightModel::Ce,
        Some(v) => {
            if let Ok(name) = v.extract::<String>() {
                if name != "ce" {
                    return Err(PyValueError::new_err(format!(
                        "light model '{name}' is not one of ce"
                    )));
                }
                LightModel::Ce
            } else {
                let (code, params): (u8, Vec<f64>) = v.extract()?;
                // Read as XF reads it back: (0, []) is CE's, (1, [ambient, wrap]) soft diffuse.
                LightModel::from_code(code, params)
            }
        }
    };
    let frame_center = fields.or("frame_center", [0.0; 3])?;
    let view = View {
        pixel_width: fields.need("pixel_width")?,
        pixel_height: fields.need("pixel_height")?,
        frame_width: fields.need("frame_width")?,
        frame_height: fields.need("frame_height")?,
        frame_center,
        mapping_centre: fields.or("mapping_centre", frame_center)?,
        background,
        camera,
        cairo_line_width_multiple: fields.or("cairo_line_width_multiple", 0.01)?,
        use_z_index: fields.or("use_z_index", true)?,
        light,
    };
    fields.finish()?;
    Ok(view)
}

fn gradient_of(fields: &Fields) -> PyResult<Option<[[f64; 3]; 2]>> {
    fields
        .take("gradient")?
        .map(|v| {
            if let Ok([ax, ay, az, bx, by, bz]) = v.extract::<[f64; 6]>() {
                Ok([[ax, ay, az], [bx, by, bz]])
            } else {
                v.extract::<[[f64; 3]; 2]>()
                    .map_err(|e| fields.bad("gradient", e))
            }
        })
        .transpose()
}

fn pre_projected_of(fields: &Fields) -> PyResult<Option<PreProjected>> {
    fields
        .take("pre_projected")?
        .map(|v| {
            let p = Fields::new("pre_projected".into(), Some(v.cast::<PyDict>()?), v.py())?;
            let projected = PreProjected {
                points: need_key(&p, "points")?,
                gradient: gradient_of(&p)?,
                fill: paint_of(p.take("fill")?)?,
                stroke: paint_of(p.take("stroke")?)?,
                background_stroke: paint_of(p.take("background_stroke")?)?,
            };
            p.finish()?;
            Ok(projected)
        })
        .transpose()
}

fn paint_block(fields: &Fields) -> PyResult<PaintBlock> {
    let joint: String = fields.or("joint", "auto".to_string())?;
    let cap: String = fields.or("cap", "auto".to_string())?;
    let paint_b = match fields.take("paint_b")? {
        None => None,
        Some(v) => {
            let b = Fields::new("paint_b".into(), Some(v.cast::<PyDict>()?), v.py())?;
            let block = PaintB {
                fill: paint_of(b.take("fill")?)?,
                stroke: paint_of(b.take("stroke")?)?,
                background_stroke: paint_of(b.take("background_stroke")?)?,
                stroke_width: b.or("stroke_width", 0.0)?,
                background_stroke_width: b.or("background_stroke_width", 0.0)?,
            };
            b.finish()?;
            Some(block)
        }
    };
    Ok(PaintBlock {
        fill: paint_of(fields.take("fill")?)?,
        stroke: paint_of(fields.take("stroke")?)?,
        background_stroke: paint_of(fields.take("background_stroke")?)?,
        stroke_width: fields.or("stroke_width", 0.0)?,
        background_stroke_width: fields.or("background_stroke_width", 0.0)?,
        gradient: gradient_of(fields)?,
        joint: named(
            "joint",
            &joint,
            &[
                ("auto", Joint::Auto),
                ("round", Joint::Round),
                ("bevel", Joint::Bevel),
                ("miter", Joint::Miter),
            ],
        )?,
        cap: named(
            "cap",
            &cap,
            &[
                ("auto", Cap::Auto),
                ("round", Cap::Round),
                ("butt", Cap::Butt),
                ("square", Cap::Square),
            ],
        )?,
        paint_b,
        mix: fields.or("mix", 0.0)?,
    })
}

fn body_of(kind: RecordKind, fields: &Fields) -> PyResult<Body> {
    Ok(match kind {
        RecordKind::Path | RecordKind::Mesh => Body::None,
        RecordKind::FaceGroup => {
            let uv: PyArrayLike2<u32, AllowTypeChange> = fields.need("uv")?;
            let uv = uv.as_array();
            if uv.ncols() != 2 {
                return Err(PyValueError::new_err("uv must have shape (faces, 2)"));
            }
            Body::FaceGroup(FaceGroup {
                uv: uv.rows().into_iter().map(|r| [r[0], r[1]]).collect(),
                grid: fields.need("grid")?,
                fill: need_key(fields, "face_fill")?,
                stroke: key_field(fields, "face_stroke")?,
            })
        }
        RecordKind::Points => Body::Points(Points {
            colors: need_key(fields, "colors")?,
            radius: fields.need("radius")?,
        }),
        RecordKind::Image => {
            let corners = rows3(&fields.need::<Bound<PyAny>>("corners")?, "corners")?;
            let corners: [[f64; 3]; 4] = corners
                .try_into()
                .map_err(|_| PyValueError::new_err("corners must have shape (4, 3)"))?;
            let resampling = fields.or("resampling", 3)?;
            let opacity = fields.or("image_opacity", 1.0)?;
            Body::Image(match fields.take("sub_frame")? {
                Some(v) => {
                    let (rank, view, order): (u32, Bound<PyDict>, Bound<PyAny>) = v.extract()?;
                    ImageSource::SubFrame {
                        rank,
                        corners,
                        resampling,
                        opacity,
                        view: Box::new(view_of(Fields::new(
                            "a sub-frame's view".into(),
                            Some(&view),
                            v.py(),
                        )?)?),
                        order: order_of(&order)?,
                    }
                }
                None => ImageSource::Picture {
                    corners,
                    resampling,
                    opacity,
                },
            })
        }
        RecordKind::Instances => {
            let placements: PyArrayLikeDyn<f64, AllowTypeChange> = fields.need("placements")?;
            let placements = placements.as_array();
            let &[n, 3, 4] = placements.shape() else {
                return Err(PyValueError::new_err(format!(
                    "placements must have shape (n, 3, 4), not {:?}",
                    placements.shape()
                )));
            };
            let objects: PyArrayLike1<u32, AllowTypeChange> = fields.need("objects")?;
            let objects = objects.as_array().to_vec();
            if objects.len() != n {
                return Err(xf_error(
                    fields.dict.py(),
                    format!(
                        "an instances record has {n} placements and {} objects",
                        objects.len()
                    ),
                ));
            }
            Body::Instances(Instances {
                placements: placements
                    .iter()
                    .copied()
                    .collect::<Vec<f64>>()
                    .chunks(12)
                    .map(|c| {
                        [
                            [c[0], c[1], c[2], c[3]],
                            [c[4], c[5], c[6], c[7]],
                            [c[8], c[9], c[10], c[11]],
                        ]
                    })
                    .collect(),
                fill: need_key(fields, "instance_fill")?,
                stroke: need_key(fields, "instance_stroke")?,
                objects,
            })
        }
        RecordKind::Volume | RecordKind::Field | RecordKind::Emission => {
            Body::Opaque(fields.or("payload", Vec::<u8>::new())?)
        }
        RecordKind::Unknown(_) => unreachable!("kinds are named"),
    })
}

// ---------------------------------------------------------------------------------------------
// Classes.

/// One drawn frame's changes against a stream, built field by field.
#[pyclass(name = "FrameDelta", module = "arachne_core")]
#[derive(Clone)]
struct PyFrameDelta {
    delta: Delta,
    uploaded: HashSet<Key>,
}

impl PyFrameDelta {
    fn wrap(delta: Delta) -> PyFrameDelta {
        let uploaded = delta
            .upload
            .iter()
            .map(|s| s.key())
            .chain(delta.upload_paints.iter().map(|s| s.key()))
            .collect();
        PyFrameDelta { delta, uploaded }
    }

    fn upload(&mut self, py: Python<'_>, data: ShapeData) -> Py<PyBytes> {
        let shape = Shape::new(data);
        let key = shape.key();
        if self.uploaded.insert(key) {
            self.delta.upload.push(shape);
        }
        PyBytes::new(py, &key.0).unbind()
    }
}

fn time_of(value: Option<&Bound<'_, PyAny>>) -> PyResult<Time> {
    let Some(value) = value else {
        return Ok(Time { num: 0, den: 1 });
    };
    let (num, den): (i64, i64) = match value.extract() {
        Ok(pair) => pair,
        Err(_) => (
            value.getattr("numerator")?.extract()?,
            value.getattr("denominator")?.extract()?,
        ),
    };
    if den <= 0 {
        return Err(PyValueError::new_err(
            "a time's denominator must be positive",
        ));
    }
    Ok(Time { num, den })
}

#[pymethods]
impl PyFrameDelta {
    #[new]
    #[pyo3(signature = (t=None, seq=0))]
    fn new(t: Option<&Bound<'_, PyAny>>, seq: u64) -> PyResult<Self> {
        Ok(PyFrameDelta::wrap(Delta::new(time_of(t)?, seq)))
    }

    /// The XF version the delta was written at, (major, minor).
    #[getter]
    fn version(&self) -> (u16, u16) {
        self.delta.version
    }

    /// The frame's time as (numerator, denominator).
    #[getter]
    fn t(&self) -> (i64, i64) {
        (self.delta.t.num, self.delta.t.den)
    }

    #[getter]
    fn seq(&self) -> u64 {
        self.delta.seq
    }

    /// Whether the delta changes nothing (a held frame).
    #[getter]
    fn is_empty(&self) -> bool {
        self.delta.is_empty()
    }

    /// The held XF values as a dictionary. Arrays are independent copies; views and records use
    /// the construction methods' keyword names. No floating-point arithmetic is performed.
    fn contents<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        contents::delta(py, &self.delta)
    }

    /// Sets the view: pixel_width, pixel_height, frame_width and frame_height are required; the
    /// rest default to CE's (camera "flat", "moving" or "three_d" with phi, theta, gamma,
    /// focal_distance, zoom, exponential_projection, light_source, shading_factor,
    /// should_apply_shading; frame_center; mapping_centre (defaults to frame_center);
    /// background as RGBA or an image key;
    /// cairo_line_width_multiple; use_z_index; light_model).
    #[pyo3(signature = (**fields))]
    fn set_view(&mut self, py: Python<'_>, fields: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        self.delta.view = Some(view_of(Fields::new("set_view()".into(), fields, py)?)?);
        Ok(())
    }

    /// Names an object for findings.
    fn add_object(&mut self, object: u32, stable_id: String, source: String) {
        self.delta.objects.push(ObjectInfo {
            object,
            stable_id,
            source,
        });
    }

    /// Uploads a path's points, f64 (n, 3) in CE's cubic layout; returns its key.
    fn upload_path(&mut self, py: Python<'_>, points: &Bound<'_, PyAny>) -> PyResult<Py<PyBytes>> {
        let points = rows3(points, "points")?;
        Ok(self.upload(py, ShapeData::Path { points }))
    }

    /// Uploads a mesh: positions f64 (v, 3), triangles u32 (t, 3), optional normals f64 (v, 3).
    #[pyo3(signature = (positions, triangles, normals=None))]
    fn upload_mesh(
        &mut self,
        py: Python<'_>,
        positions: &Bound<'_, PyAny>,
        triangles: &Bound<'_, PyAny>,
        normals: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyBytes>> {
        let positions = rows3(positions, "positions")?;
        let triangles: PyArrayLike2<u32, AllowTypeChange> = triangles.extract()?;
        let triangles = triangles.as_array();
        if triangles.ncols() != 3 {
            return Err(PyValueError::new_err("triangles must have shape (t, 3)"));
        }
        let triangles: Vec<[u32; 3]> = triangles
            .rows()
            .into_iter()
            .map(|r| [r[0], r[1], r[2]])
            .collect();
        if triangles
            .iter()
            .flatten()
            .any(|i| *i as usize >= positions.len())
        {
            return Err(PyValueError::new_err(
                "a triangle names a vertex that does not exist",
            ));
        }
        let normals = normals.map(|n| rows3(n, "normals")).transpose()?;
        if normals.as_ref().is_some_and(|n| n.len() != positions.len()) {
            return Err(PyValueError::new_err("one normal per vertex"));
        }
        Ok(self.upload(
            py,
            ShapeData::Mesh {
                positions,
                triangles,
                normals,
            },
        ))
    }

    /// Uploads an image, u8 (height, width, 4), straight RGBA; returns its key.
    fn upload_image(&mut self, py: Python<'_>, rgba: &Bound<'_, PyAny>) -> PyResult<Py<PyBytes>> {
        let rgba: PyArrayLike3<u8, AllowTypeChange> = rgba.extract()?;
        let view = rgba.as_array();
        let &[height, width, channels] = view.shape() else {
            unreachable!()
        };
        if channels != 4 {
            return Err(PyValueError::new_err(
                "an image has shape (height, width, 4)",
            ));
        }
        Ok(self.upload(
            py,
            ShapeData::Image {
                width: width as u32,
                height: height as u32,
                rgba: view.iter().copied().collect(),
            },
        ))
    }

    /// Uploads a paint table, f64 (n, 4) RGBA rows; returns its key.
    fn upload_paint(&mut self, py: Python<'_>, rows: &Bound<'_, PyAny>) -> PyResult<Py<PyBytes>> {
        let rows: PyArrayLike2<f64, AllowTypeChange> = rows.extract()?;
        let view = rows.as_array();
        if view.ncols() != 4 {
            return Err(PyValueError::new_err("a paint table has shape (n, 4)"));
        }
        let stops = Stops::new(
            view.rows()
                .into_iter()
                .map(|r| [r[0], r[1], r[2], r[3]])
                .collect(),
        );
        let key = stops.key();
        if self.uploaded.insert(key) {
            self.delta.upload_paints.push(stops);
        }
        Ok(PyBytes::new(py, &key.0).unbind())
    }

    /// Releases a shape or paint table by key.
    fn release(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        self.delta.release.push(key_of(key)?);
        Ok(())
    }

    /// Sets the record in `slot`. `kind` is one of path, mesh, face_group, points, image,
    /// instances, or a reserved kind (volume, field, emission). See `core/XF.md` for the fields.
    #[pyo3(signature = (kind, slot, object, **fields))]
    fn set_record(
        &mut self,
        py: Python<'_>,
        kind: &str,
        slot: u32,
        object: u32,
        fields: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let kind = RecordKind::from_name(kind)
            .ok_or_else(|| PyValueError::new_err(format!("'{kind}' is not a record kind")))?;
        let fields = Fields::new(
            format!("set_record() for a {} record", kind.name()),
            fields,
            py,
        )?;
        let mut record = Record::new(kind, slot, object);
        record.features = fields.or("features", 0u64)?;
        record.fill_image = key_field(&fields, "fill_image")?;
        if record.fill_image.is_some() {
            record.features |= FEATURE_BACKGROUND_IMAGE;
        } else if record.features & FEATURE_BACKGROUND_IMAGE != 0 {
            return Err(xf_error(
                py,
                "background_image feature needs fill_image".into(),
            ));
        }
        record.pre_projected = pre_projected_of(&fields)?;
        if record.pre_projected.is_some() {
            if !matches!(
                kind,
                RecordKind::Path | RecordKind::Points | RecordKind::Image
            ) {
                return Err(xf_error(
                    py,
                    format!("pre_projected is not valid on {} records", kind.name()),
                ));
            }
            record.features |= FEATURE_PRE_PROJECTED;
        } else if record.features & FEATURE_PRE_PROJECTED != 0 {
            return Err(xf_error(
                py,
                "pre_projected feature needs pre_projected".into(),
            ));
        }
        record.geometry = Geometry {
            shape_a: key_field(&fields, "shape")?,
            place_a: fields.or("place", IDENTITY)?,
            blend: match fields.take("blend")? {
                None => None,
                Some(v) => {
                    let (shape_b, place_b, t): (Bound<PyAny>, Affine, f64) = v.extract()?;
                    Some(Blend {
                        shape_b: key_of(&shape_b)?,
                        place_b,
                        t,
                    })
                }
            },
            reveal: fields.get("reveal")?,
        };
        record.paint = paint_block(&fields)?;
        record.dash = fields
            .get::<[f64; 3]>("dash")?
            .map(|[period, duty, phase]| Dash {
                period,
                duty,
                phase,
            });
        let space: String = fields.or("space", "screen".to_string())?;
        record.depth = Depth {
            space: named(
                "space",
                &space,
                &[
                    ("screen", Space::Screen),
                    ("world", Space::World),
                    ("overlay", Space::Overlay),
                ],
            )?,
            z_index: fields.or("z_index", 0.0)?,
            sort_point: fields.or("sort_point", [0.0; 3])?,
            shade_in_3d: fields.or("shade_in_3d", false)?,
            fixed_orientation_centre: fields.get("fixed_orientation_centre")?,
        };
        record.clip = match fields.take("clip")? {
            None => None,
            Some(v) => {
                let (key, place): (Bound<PyAny>, Affine) = v.extract()?;
                Some((key_of(&key)?, place))
            }
        };
        record.opacity = fields.or("opacity", 1.0)?;
        record.layer = fields.or("layer", 0u16)?;
        record.body = body_of(kind, &fields)?;
        fields.finish()?;
        self.delta.set.push(record);
        self.delta.version = wire::encoded_version(&self.delta);
        Ok(())
    }

    /// Removes the record in `slot`.
    fn remove(&mut self, slot: u32) {
        self.delta.remove.push(slot);
    }

    /// Sets the draw order: (layer, slot, part) entries, layer one of static, moving, overlay.
    fn set_order(&mut self, entries: &Bound<'_, PyAny>) -> PyResult<()> {
        self.delta.order = Some(order_of(entries)?);
        Ok(())
    }

    /// The delta in XF's binary form.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &wire::encode_delta(&self.delta))
    }

    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        wire::decode_delta(data)
            .map(PyFrameDelta::wrap)
            .map_err(|e| wire_error(py, e))
    }

    fn __repr__(&self) -> String {
        let d = &self.delta;
        format!(
            "FrameDelta(t={}/{}, seq={}, upload={}, set={}, remove={}, order={})",
            d.t.num,
            d.t.den,
            d.seq,
            d.upload.len() + d.upload_paints.len(),
            d.set.len(),
            d.remove.len(),
            d.order
                .as_ref()
                .map_or("unchanged".to_string(), |o| o.len().to_string()),
        )
    }
}

/// What one delta changed.
#[pyclass(name = "ApplyStats", module = "arachne_core", frozen)]
struct PyApplyStats {
    #[pyo3(get)]
    uploaded_shapes: usize,
    #[pyo3(get)]
    changed_records: usize,
}

#[pymethods]
impl PyApplyStats {
    fn __repr__(&self) -> String {
        format!(
            "ApplyStats(uploaded_shapes={}, changed_records={})",
            self.uploaded_shapes, self.changed_records
        )
    }
}

/// What a drawer holds across frames. Make it in the process that draws (after any fork).
#[pyclass(name = "Stream", module = "arachne_core")]
struct PyStream {
    stream: Stream,
}

#[pymethods]
impl PyStream {
    #[new]
    fn new() -> Self {
        PyStream {
            stream: Stream::new(),
        }
    }

    /// Applies a delta; a refused delta (XfError) changes nothing.
    fn apply(&mut self, py: Python<'_>, delta: PyRef<'_, PyFrameDelta>) -> PyResult<PyApplyStats> {
        let stats = self
            .stream
            .apply(&delta.delta)
            .map_err(|e| stream_error(py, e))?;
        Ok(PyApplyStats {
            uploaded_shapes: stats.uploaded_shapes,
            changed_records: stats.changed_records,
        })
    }

    /// Draws the frame: a uint8 array (height, width, 4), the layout of CE's pixel array. Raises
    /// Unsupported (render.unsupported) when a drawn record carries what this build cannot draw.
    /// With ``overlay=False``, the overlay's entries (``overlay_entries``) are left out, neither
    /// drawn nor refused: the frame Cairo draws a 3D scene's overlay over.
    #[pyo3(signature = (width, height, *, threads=1, overlay=true))]
    fn draw<'py>(
        &self,
        py: Python<'py>,
        width: u32,
        height: u32,
        threads: usize,
        overlay: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let stream = &self.stream;
        let rgba = py
            .detach(|| {
                if overlay {
                    stream.draw(width, height, threads)
                } else {
                    stream
                        .render_under_overlay(width, height, threads)
                        .map(|r| r.rgba)
                }
            })
            .map_err(|e| draw_error(py, e))?;
        Ok(PyArray1::from_vec(py, rgba)
            .reshape([height as usize, width as usize, 4])?
            .into_any())
    }

    /// The order's entries a 3D scene's frame leaves to Cairo's overlay, in order, as
    /// ``(layer, slot, part)``: records fixed in frame, images (displays included), and paths
    /// filled through a background image.
    fn overlay_entries(&self) -> Vec<(&'static str, u32, u32)> {
        contents::order(&self.stream.overlay_entries())
    }

    /// Per slot the order draws, the pixel box ``(x_min, y_min, x_max, y_max)`` (x right, y down)
    /// of its record's points as the camera places them and maps them to pixels: paths, faces
    /// and meshes around the view's mapping centre, point clouds and images around its frame
    /// centre. A record placing no point has no entry. Nothing is drawn, so records the drawer
    /// refuses have boxes too.
    fn screen_boxes(&self, py: Python<'_>) -> PyResult<BTreeMap<u32, (f64, f64, f64, f64)>> {
        let boxes = self.stream.screen_boxes().map_err(|e| draw_error(py, e))?;
        Ok(boxes
            .into_iter()
            .map(|(slot, b)| (slot, (b[0], b[1], b[2], b[3])))
            .collect())
    }

    /// Draws the frame as ``draw`` does, and returns it with each pixel's resolve: a uint8 array
    /// (height, width) indexing ``RESOLVES`` (background, fast, coverage, sample).
    #[pyo3(signature = (width, height, *, threads=1))]
    fn draw_resolves<'py>(
        &self,
        py: Python<'py>,
        width: u32,
        height: u32,
        threads: usize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let stream = &self.stream;
        let drawn = py
            .detach(|| stream.render(width, height, threads))
            .map_err(|e| draw_error(py, e))?;
        let (h, w) = (height as usize, width as usize);
        Ok((
            PyArray1::from_vec(py, drawn.rgba)
                .reshape([h, w, 4])?
                .into_any(),
            PyArray1::from_vec(py, drawn.resolves)
                .reshape([h, w])?
                .into_any(),
        ))
    }
}

/// Reads an XF file into its deltas.
#[pyfunction]
fn read_xf(py: Python<'_>, path: PathBuf) -> PyResult<Vec<PyFrameDelta>> {
    let deltas = wire::read_file(&path).map_err(|e| match e {
        XfError::Io(io) => PyErr::from(io),
        other => wire_error(py, other),
    })?;
    Ok(deltas.into_iter().map(PyFrameDelta::wrap).collect())
}

/// Writes deltas to an XF file.
#[pyfunction]
fn write_xf(path: PathBuf, deltas: &Bound<'_, PyList>) -> PyResult<()> {
    let deltas = deltas
        .iter()
        .map(|d| Ok(d.extract::<PyRef<PyFrameDelta>>()?.delta.clone()))
        .collect::<PyResult<Vec<_>>>()?;
    std::fs::write(&path, wire::encode_file(&deltas))?;
    Ok(())
}

/// The class probes' streams, the parity kit's fixtures: `(name, width, height, deltas)` each,
/// built in code (`parity/probes.rs`), never stored.
#[pyfunction]
fn probe_fixtures() -> Vec<(&'static str, u32, u32, Vec<PyFrameDelta>)> {
    crate::probes::all()
        .into_iter()
        .map(|p| {
            let deltas = p.deltas.into_iter().map(PyFrameDelta::wrap).collect();
            (p.name, p.width, p.height, deltas)
        })
        .collect()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("XF_VERSION", XF_VERSION)?;
    let features = PyDict::new(m.py());
    features.set_item("background_image", FEATURE_BACKGROUND_IMAGE)?;
    features.set_item("pre_projected", FEATURE_PRE_PROJECTED)?;
    m.add("FEATURES", features)?;
    m.add("RESOLVES", ("background", "fast", "coverage", "sample"))?;
    m.add(
        "COVERAGE_TO_SAMPLE_LSB",
        crate::composite::COVERAGE_TO_SAMPLE_LSB,
    )?;
    m.add_function(wrap_pyfunction!(probe_fixtures, m)?)?;
    m.add_class::<PyFrameDelta>()?;
    m.add_class::<PyStream>()?;
    m.add_class::<PyApplyStats>()?;
    m.add_function(wrap_pyfunction!(read_xf, m)?)?;
    m.add_function(wrap_pyfunction!(write_xf, m)?)?;
    Ok(())
}
