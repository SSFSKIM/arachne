//! The Python binding, `arachne_gpu._gpu`, re-exported by the `arachne_gpu` package.

use std::sync::Mutex;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::call::PyCallArgs;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::compositor::{Compositor, DrawError, OpenError};
use crate::device::{AdapterSummary, NoGpu, Pin};
use crate::frame::{Frame, Layer, Source};
use crate::keys::{self, Key};

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

fn open_error(py: Python<'_>, err: OpenError) -> PyErr {
    match err {
        OpenError::NoGpu(err) => no_gpu(py, err),
        OpenError::InvalidSize(what) => PyValueError::new_err(what),
    }
}

fn draw_error(py: Python<'_>, err: DrawError) -> PyErr {
    match err {
        DrawError::UnknownKey(_) => PyKeyError::new_err(err.to_string()),
        DrawError::Unsupported(reason) => python_error(py, "Unsupported", (reason,)),
        DrawError::Invalid(_) => PyValueError::new_err(err.to_string()),
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

/// A `Frame` of `arachne_gpu._frame`, read by its fields.
fn frame_of(frame: &Bound<'_, PyAny>) -> PyResult<Frame> {
    let py = frame.py();
    let classes = py.import("arachne_gpu._frame")?;
    let picture = classes.getattr("Picture")?;
    let quad = classes.getattr("Quad")?;
    let background: Vec<u8> = frame.getattr("background")?.extract()?;
    let background: [u8; 4] = background
        .try_into()
        .map_err(|_| PyValueError::new_err("a frame's background is 4 bytes, painted RGBA"))?;
    let mut layers = Vec::new();
    for layer in frame.getattr("layers")?.try_iter()? {
        let layer = layer?;
        let source = layer.getattr("source")?;
        let key = key_of(&source.getattr("key")?)?;
        let source = if source.is_instance(&quad)? {
            let corners: Vec<(f64, f64)> = source.getattr("corners")?.extract()?;
            let corners: [(f64, f64); 4] = corners
                .try_into()
                .map_err(|_| PyValueError::new_err("a quad has four corners"))?;
            Source::Quad {
                key,
                corners: corners.map(|(x, y)| [x, y]),
                resampling: source.getattr("resampling")?.extract()?,
            }
        } else if source.is_instance(&picture)? {
            Source::Picture { key }
        } else {
            return Err(PyTypeError::new_err(
                "a layer's source is a Picture or a Quad",
            ));
        };
        layers.push(Layer {
            source,
            opacity: layer.getattr("opacity")?.extract()?,
        });
    }
    Ok(Frame { background, layers })
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
        let frame = frame_of(frame)?;
        let rgba = self.with(py, |c| c.draw(&frame))?;
        Ok(PyArray1::from_vec(py, rgba)
            .reshape([self.height as usize, self.width as usize, 4])?
            .into_any())
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
    let found = py.detach(crate::device::adapters);
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
