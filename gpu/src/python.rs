//! The Python binding, `arachne_gpu._gpu`, re-exported by the `arachne_gpu` package.

use std::sync::Mutex;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::call::PyCallArgs;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::compositor::{Compositor, DrawError, OpenError};
use crate::device::{AdapterSummary, ForkedProcess, NoGpu, Pin};
use crate::frame::{Blur, Focus, Frame, Layer, SoftMask, Source};
use crate::keys::{self, Key};
use crate::object_map::Footprint;
use crate::plan::MAX_NESTING;

// ---------------------------------------------------------------------------------------------
// Errors: the classes live in `arachne_gpu/_errors.py`.

fn python_error<'py, A: PyCallArgs<'py>>(py: Python<'py>, class: &str, args: A) -> PyErr {
    match py
        .import("arachne_gpu._errors")
        .and_then(|m| m.getattr(class))
        .and_then(|cls| cls.call1(args))
    {
        Ok(exception) => PyErr::from_value(exception),
        Err(err) => err,
    }
}

fn no_gpu(py: Python<'_>, err: NoGpu) -> PyErr {
    let seen: PyResult<Vec<Bound<'_, PyDict>>> =
        err.seen.iter().map(|a| adapter_dict(py, a)).collect();
    match seen {
        Ok(seen) => python_error(py, "NoGpu", (err.to_string(), seen)),
        Err(e) => e,
    }
}

fn forked_process(py: Python<'_>, err: ForkedProcess) -> PyErr {
    python_error(py, "ForkedProcessError", (err.to_string(),))
}

fn open_error(py: Python<'_>, err: OpenError) -> PyErr {
    match err {
        OpenError::NoGpu(err) => no_gpu(py, err),
        OpenError::InvalidSize(what) => PyValueError::new_err(what),
        OpenError::ForkedProcess(err) => forked_process(py, err),
    }
}

fn draw_error(py: Python<'_>, err: DrawError) -> PyErr {
    match err {
        DrawError::UnknownKey(_) => PyKeyError::new_err(err.to_string()),
        DrawError::Unsupported(reason) => python_error(py, "Unsupported", (reason,)),
        DrawError::Invalid(_) => PyValueError::new_err(err.to_string()),
        DrawError::SizeMismatch(what) => python_error(py, "SizeMismatch", (what,)),
        DrawError::Device(_) => PyRuntimeError::new_err(err.to_string()),
    }
}

// ---------------------------------------------------------------------------------------------
// Values.

fn adapter_dict<'py>(py: Python<'py>, a: &AdapterSummary) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("name", &a.name)?;
    d.set_item("backend", &a.backend)?;
    d.set_item("device_type", &a.device_type)?;
    d.set_item("driver", &a.driver)?;
    d.set_item("driver_info", &a.driver_info)?;
    d.set_item("vendor", a.vendor)?;
    d.set_item("device", a.device)?;
    Ok(d)
}

fn pin_dict<'py>(py: Python<'py>, pin: &Pin) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("adapter", &pin.adapter)?;
    d.set_item("backend", &pin.backend)?;
    d.set_item("driver", &pin.driver)?;
    d.set_item("wgpu", &pin.wgpu)?;
    Ok(d)
}

fn key_of(value: &Bound<'_, PyAny>) -> PyResult<Key> {
    let bytes: Vec<u8> = value.extract()?;
    bytes
        .try_into()
        .map_err(|_| PyValueError::new_err("a key is 16 bytes"))
}

/// An RGBA texture's size and bytes, from a uint8 array (height, width, 4).
fn rgba_of(rgba: &PyReadonlyArray3<'_, u8>) -> PyResult<(u32, u32, Vec<u8>)> {
    let &[height, width, channels] = rgba.shape() else {
        unreachable!()
    };
    if channels != 4 {
        return Err(PyValueError::new_err(
            "a texture is a uint8 array of shape (height, width, 4)",
        ));
    }
    let size = |n: usize| {
        u32::try_from(n).map_err(|_| PyValueError::new_err("a texture's side does not fit in u32"))
    };
    let view = rgba.as_array();
    let bytes = match view.as_slice() {
        Some(slice) => slice.to_vec(),
        None => view.iter().copied().collect(),
    };
    Ok((size(width)?, size(height)?, bytes))
}

/// Four corners in pixels.
fn corners_of(source: &Bound<'_, PyAny>, what: &str) -> PyResult<[[f64; 2]; 4]> {
    let corners: Vec<(f64, f64)> = source.getattr("corners")?.extract()?;
    let corners: [(f64, f64); 4] = corners
        .try_into()
        .map_err(|_| PyValueError::new_err(format!("{what} has four corners")))?;
    Ok(corners.map(|(x, y)| [x, y]))
}

/// An optional attribute: `None` when absent or None.
fn optional<'py>(value: &Bound<'py, PyAny>, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    let attribute = value.getattr(name)?;
    Ok((!attribute.is_none()).then_some(attribute))
}

/// A `Frame` of `arachne_gpu._frame`, read by its fields, `nesting` windows deep.
fn frame_of(frame: &Bound<'_, PyAny>, nesting: usize) -> PyResult<Frame> {
    let py = frame.py();
    let classes = py.import("arachne_gpu._frame")?;
    let picture = classes.getattr("Picture")?;
    let quad = classes.getattr("Quad")?;
    let window = classes.getattr("Window")?;
    let background: Vec<u8> = frame.getattr("background")?.extract()?;
    let background: [u8; 4] = background
        .try_into()
        .map_err(|_| PyValueError::new_err("a frame's background is 4 bytes, painted RGBA"))?;
    let focus = match optional(frame, "focus")? {
        Some(focus) => Some(Focus {
            depth: focus.getattr("depth")?.extract()?,
            aperture: focus.getattr("aperture")?.extract()?,
        }),
        None => None,
    };
    let mut layers = Vec::new();
    for layer in frame.getattr("layers")?.try_iter()? {
        let layer = layer?;
        let source = layer.getattr("source")?;
        let source = if source.is_instance(&quad)? {
            Source::Quad {
                key: key_of(&source.getattr("key")?)?,
                corners: corners_of(&source, "a quad")?,
                resampling: source.getattr("resampling")?.extract()?,
            }
        } else if source.is_instance(&picture)? {
            Source::Picture {
                key: key_of(&source.getattr("key")?)?,
            }
        } else if source.is_instance(&window)? {
            // Read no deeper than a frame may nest, so a frame that holds itself ends here.
            if nesting >= MAX_NESTING {
                return Err(python_error(
                    py,
                    "Unsupported",
                    (format!("windows nested deeper than {MAX_NESTING}"),),
                ));
            }
            let size = match optional(&source, "size")? {
                Some(size) => {
                    let (width, height): (u32, u32) = size.extract()?;
                    Some([width, height])
                }
                None => None,
            };
            Source::Window {
                frame: Box::new(frame_of(&source.getattr("frame")?, nesting + 1)?),
                corners: corners_of(&source, "a window")?,
                resampling: source.getattr("resampling")?.extract()?,
                size,
            }
        } else {
            return Err(PyTypeError::new_err(
                "a layer's source is a Picture, a Quad or a Window",
            ));
        };
        let blur = match optional(&layer, "blur")? {
            Some(blur) => Some(Blur {
                sigma_px: blur.getattr("sigma_px")?.extract()?,
            }),
            None => None,
        };
        let mask = match optional(&layer, "mask")? {
            Some(mask) => Some(SoftMask {
                key: key_of(&mask.getattr("key")?)?,
                feather_px: mask.getattr("feather_px")?.extract()?,
                invert: mask.getattr("invert")?.extract()?,
            }),
            None => None,
        };
        let depth: Option<f64> = match optional(&layer, "depth")? {
            Some(depth) => Some(depth.extract()?),
            None => None,
        };
        let object_id: Option<String> = match optional(&layer, "object_id")? {
            Some(id) => Some(id.extract()?),
            None => None,
        };
        layers.push(Layer {
            source,
            opacity: layer.getattr("opacity")?.extract()?,
            depth,
            blur,
            mask,
            object_id,
        });
    }
    Ok(Frame {
        background,
        layers,
        focus,
    })
}

/// A `Footprint` of `arachne_gpu._frame`: its mask a bool array (box height, box width).
fn footprint_object<'py>(py: Python<'py>, footprint: Footprint) -> PyResult<Bound<'py, PyAny>> {
    let [x0, y0, x1, y1] = footprint.r#box;
    let mask =
        PyArray1::from_vec(py, footprint.mask).reshape([(y1 - y0) as usize, (x1 - x0) as usize])?;
    py.import("arachne_gpu._frame")?
        .getattr("Footprint")?
        .call1((
            footprint.object_id,
            footprint.layer,
            (x0, y0, x1, y1),
            mask,
            footprint.grow_px,
        ))
}

// ---------------------------------------------------------------------------------------------
// The compositor.

/// A compositor shared by Python threads: every call takes it whole, one at a time, so one draw
/// runs from its first pass to its read-back alone. What never changes is kept beside it.
#[pyclass(name = "Compositor", module = "arachne_gpu", frozen)]
struct PyCompositor {
    inner: Mutex<Compositor>,
    pin: Pin,
    adapter: AdapterSummary,
    software: bool,
    width: u32,
    height: u32,
}

impl PyCompositor {
    fn of(inner: Compositor) -> Self {
        PyCompositor {
            pin: inner.pin(),
            adapter: inner.adapter(),
            software: inner.software(),
            width: inner.width(),
            height: inner.height(),
            inner: Mutex::new(inner),
        }
    }

    /// Runs `f` on the compositor with the GIL released, once every earlier call has ended.
    fn with<T: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut Compositor) -> Result<T, DrawError> + Send,
    ) -> PyResult<T> {
        py.detach(|| match self.inner.lock() {
            Ok(mut compositor) => f(&mut compositor).map_err(Some),
            Err(_) => Err(None),
        })
        .map_err(|err| match err {
            Some(err) => draw_error(py, err),
            None => PyRuntimeError::new_err("the compositor failed in an earlier call"),
        })
    }
}

#[pymethods]
impl PyCompositor {
    /// A compositor drawing frames of ``width`` x ``height`` on a GPU, never on a software
    /// adapter. Raises NoGpu, naming every adapter it saw, and ValueError for a size the device
    /// cannot draw.
    #[new]
    fn new(py: Python<'_>, width: u32, height: u32) -> PyResult<Self> {
        let inner = py
            .detach(|| Compositor::new(width, height))
            .map_err(|e| open_error(py, e))?;
        Ok(PyCompositor::of(inner))
    }

    /// The device it draws on: adapter, backend, driver, wgpu.
    #[getter]
    fn pin<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pin_dict(py, &self.pin)
    }

    /// Everything its adapter reports, as ``adapters()`` lists it.
    #[getter]
    fn adapter<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        adapter_dict(py, &self.adapter)
    }

    /// True only for a compositor from ``arachne_gpu.testing``.
    #[getter]
    fn software(&self) -> bool {
        self.software
    }

    #[getter]
    fn width(&self) -> u32 {
        self.width
    }

    #[getter]
    fn height(&self) -> u32 {
        self.height
    }

    /// Takes an image: straight RGBA8 (height, width, 4) under its Frame IR image-shape key,
    /// converted to painted values here, once.
    fn upload_image(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        rgba: PyReadonlyArray3<'_, u8>,
    ) -> PyResult<()> {
        let key = key_of(key)?;
        let (width, height, bytes) = rgba_of(&rgba)?;
        self.with(py, |c| c.upload_image(key, width, height, &bytes))
    }

    /// Takes a painted picture: painted RGBA8 (height, width, 4) under ``painted_key(rgba)``.
    fn upload_painted(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        rgba: PyReadonlyArray3<'_, u8>,
    ) -> PyResult<()> {
        let key = key_of(key)?;
        let (width, height, bytes) = rgba_of(&rgba)?;
        self.with(py, |c| c.upload_painted(key, width, height, &bytes))
    }

    /// Lets go of the texture under ``key``, if one is held.
    fn release(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let key = key_of(key)?;
        self.with(py, |c| {
            c.release(&key);
            Ok(())
        })
    }

    /// Draws ``frame``: a uint8 array (height, width, 4), painted, in the layout of CE's pixel
    /// array. Raises Unsupported (render.unsupported) for what this build does not draw; a
    /// refused frame draws nothing.
    fn draw<'py>(&self, py: Python<'py>, frame: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let frame = frame_of(frame, 0)?;
        let rgba = self.with(py, |c| c.draw(&frame))?;
        Ok(PyArray1::from_vec(py, rgba)
            .reshape([self.height as usize, self.width as usize, 4])?
            .into_any())
    }

    /// One Footprint per layer that names an object (``object_id``), in the frame's order: the
    /// pixels the layer covers drawn with no effect (a bool mask over a half-open box, y down) and
    /// ``grow_px``, how far its effects spread it. Raises as ``draw`` does for a refused frame.
    fn object_map<'py>(
        &self,
        py: Python<'py>,
        frame: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let frame = frame_of(frame, 0)?;
        let footprints = self.with(py, |c| c.object_map(&frame))?;
        footprints
            .into_iter()
            .map(|f| footprint_object(py, f))
            .collect()
    }
}

/// The testing door: a compositor on a software adapter (``arachne_gpu.testing``).
#[pyfunction]
fn software_compositor(py: Python<'_>, width: u32, height: u32) -> PyResult<PyCompositor> {
    let inner = py
        .detach(|| crate::testing::software_compositor(width, height))
        .map_err(|e| open_error(py, e))?;
    Ok(PyCompositor::of(inner))
}

/// Every adapter this machine offers on Vulkan, of any device type.
#[pyfunction]
fn adapters(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyDict>>> {
    let found = py
        .detach(crate::device::adapters)
        .map_err(|e| forked_process(py, e))?;
    found.iter().map(|a| adapter_dict(py, a)).collect()
}

/// A painted picture's key, over its painted RGBA8 bytes (height, width, 4).
#[pyfunction]
fn painted_key<'py>(
    py: Python<'py>,
    rgba: PyReadonlyArray3<'py, u8>,
) -> PyResult<Bound<'py, PyBytes>> {
    let (width, height, bytes) = rgba_of(&rgba)?;
    Ok(PyBytes::new(py, &keys::painted_key(width, height, &bytes)))
}

#[pymodule]
fn _gpu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("WGPU_VERSION", crate::WGPU_VERSION)?;
    m.add("PAINTED_TAG", keys::PAINTED_TAG)?;
    m.add_class::<PyCompositor>()?;
    m.add_function(wrap_pyfunction!(software_compositor, m)?)?;
    m.add_function(wrap_pyfunction!(adapters, m)?)?;
    m.add_function(wrap_pyfunction!(painted_key, m)?)?;
    Ok(())
}
