//! The Python binding, `arachne_sim._sim`, whose `World` the `arachne_sim` package extends.
//!
//! Every number a world is given arrives in the stored type: a `numpy.float32` or a float32 array
//! (anything exposing a float32 buffer), never a Python float, whose one conversion is the
//! caller's. Rows leave as bytes, which the package hands out as read-only arrays. Advancing
//! releases the interpreter's lock.

use std::sync::{Mutex, MutexGuard};

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::spec::{Body, Error, Force, Joint, JointKind, Shape, ShapeKind};
use crate::world::{Integrator, Options, World, finding};

fn value_error(err: Error) -> PyErr {
    PyValueError::new_err(err.0)
}

/// The float32 numbers `value` holds, in order.
fn floats(value: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<f32>> {
    // A numpy.float32 scalar: its value as a Python float is exact, and so is its way back.
    if value.get_type().fully_qualified_name()?.to_cow()? == "numpy.float32" {
        return Ok(vec![value.extract::<f64>()? as f32]);
    }
    let buffer = PyBuffer::<f32>::get(value).map_err(|_| {
        let kind = value
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "?".into());
        PyTypeError::new_err(format!(
            "{what} takes float32 numbers (a numpy.float32 or a float32 array), the stored type, \
             not {kind}: convert each number once, before the world is given it."
        ))
    })?;
    buffer.to_vec(value.py())
}

/// `value`'s numbers, or `default` when it is not given.
fn floats_or(value: Option<&Bound<'_, PyAny>>, default: &[f32], what: &str) -> PyResult<Vec<f32>> {
    match value {
        Some(value) => floats(value, what),
        None => Ok(default.to_vec()),
    }
}

/// One float32 number.
fn float(value: &Bound<'_, PyAny>, what: &str) -> PyResult<f32> {
    let numbers = floats(value, what)?;
    match numbers[..] {
        [x] => Ok(x),
        _ => Err(PyValueError::new_err(format!(
            "{what} takes one number, not {}.",
            numbers.len()
        ))),
    }
}

fn required(value: Option<&Bound<'_, PyAny>>, what: &str) -> PyResult<f32> {
    match value {
        Some(value) => float(value, what),
        None => Err(PyValueError::new_err(format!("{what} is required."))),
    }
}

fn float_or(value: Option<&Bound<'_, PyAny>>, default: f32, what: &str) -> PyResult<f32> {
    match value {
        Some(value) => float(value, what),
        None => Ok(default),
    }
}

#[pyclass(name = "World", module = "arachne_sim", subclass)]
struct PyWorld {
    inner: Mutex<World>,
    dim: usize,
}

impl PyWorld {
    fn world(&self) -> PyResult<MutexGuard<'_, World>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("the world failed in an earlier call"))
    }

    fn zeros(&self) -> Vec<f32> {
        vec![0.0; self.dim]
    }

    fn angular_zeros(&self) -> Vec<f32> {
        vec![0.0; if self.dim == 2 { 1 } else { 3 }]
    }

    fn identity_rotation(&self) -> Vec<f32> {
        if self.dim == 2 {
            vec![0.0]
        } else {
            vec![0.0, 0.0, 0.0, 1.0]
        }
    }

    fn span(&self, start: usize, stop: Option<usize>) -> PyResult<(usize, usize)> {
        let rows = self.world()?.rows();
        let stop = stop.unwrap_or(rows);
        if start > stop || stop > rows {
            return Err(PyValueError::new_err(format!(
                "rows {start} to {stop} are not within the {rows} computed."
            )));
        }
        Ok((start, stop))
    }
}

fn bytes_of(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[pymethods]
impl PyWorld {
    /// A world in ``dimension`` (2 or 3) dimensions. ``integrator`` is ``"solver"`` (contacts,
    /// joints, rotation) or ``"point_masses"`` (no shape and no joint; velocity Verlet). An
    /// ``open`` world also feels forces its caller gives at each tick, through ``step``.
    /// ``speed_limit`` (units per second) and ``angular_speed_limit`` (radians per second) are
    /// the speeds above which a body is unstable, each positive and below the solver's cap
    /// (``SPEED_CAP``, ``ANGULAR_SPEED_CAP``; by default ``SPEED_LIMIT`` and
    /// ``ANGULAR_SPEED_LIMIT``). ``continuous_collision`` sweeps fast bodies against fixed ones.
    #[new]
    #[pyo3(signature = (
        dimension, *, integrator="solver", open=false, speed_limit=None,
        angular_speed_limit=None, continuous_collision=true
    ))]
    fn new(
        dimension: usize,
        integrator: &str,
        open: bool,
        speed_limit: Option<&Bound<'_, PyAny>>,
        angular_speed_limit: Option<&Bound<'_, PyAny>>,
        continuous_collision: bool,
    ) -> PyResult<Self> {
        let integrator = match integrator {
            "solver" => Integrator::Solver,
            "point_masses" => Integrator::PointMasses,
            other => {
                return Err(PyValueError::new_err(format!(
                    "An integrator is \"solver\" or \"point_masses\", not {other:?}."
                )));
            }
        };
        let options = Options {
            speed_limit: speed_limit
                .map(|x| float(x, "A world's speed limit"))
                .transpose()?,
            angular_speed_limit: angular_speed_limit
                .map(|x| float(x, "A world's angular speed limit"))
                .transpose()?,
            continuous_collision,
        };
        let world =
            World::with_options(dimension, integrator, open, options).map_err(value_error)?;
        Ok(PyWorld {
            inner: Mutex::new(world),
            dim: dimension,
        })
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.dim
    }

    #[getter]
    fn integrator(&self) -> PyResult<&'static str> {
        Ok(match self.world()?.integrator() {
            Integrator::Solver => "solver",
            Integrator::PointMasses => "point_masses",
        })
    }

    #[getter]
    fn open(&self) -> PyResult<bool> {
        Ok(self.world()?.is_open())
    }

    /// The speed, units per second, above which a body is unstable.
    #[getter]
    fn speed_limit(&self) -> PyResult<f32> {
        Ok(self.world()?.speed_limit())
    }

    /// The angular speed, radians per second, above which a body is unstable.
    #[getter]
    fn angular_speed_limit(&self) -> PyResult<f32> {
        Ok(self.world()?.angular_speed_limit())
    }

    /// Whether the solver sweeps fast bodies against fixed ones.
    #[getter]
    fn continuous_collision(&self) -> PyResult<bool> {
        Ok(self.world()?.continuous_collision())
    }

    /// The number of bodies.
    #[getter]
    fn bodies(&self) -> PyResult<usize> {
        Ok(self.world()?.bodies())
    }

    /// The number of rows computed.
    #[getter]
    fn rows(&self) -> PyResult<usize> {
        Ok(self.world()?.rows())
    }

    /// Declares a body; returns its number, its position on the bodies axis of every row.
    #[pyo3(signature = (
        *, position=None, rotation=None, velocity=None, angular_velocity=None, mass=None,
        gravity_scale=None, fixed=false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add_body(
        &self,
        position: Option<&Bound<'_, PyAny>>,
        rotation: Option<&Bound<'_, PyAny>>,
        velocity: Option<&Bound<'_, PyAny>>,
        angular_velocity: Option<&Bound<'_, PyAny>>,
        mass: Option<&Bound<'_, PyAny>>,
        gravity_scale: Option<&Bound<'_, PyAny>>,
        fixed: bool,
    ) -> PyResult<u32> {
        let body = Body {
            fixed,
            position: floats_or(position, &self.zeros(), "A body's position")?,
            rotation: floats_or(rotation, &self.identity_rotation(), "A body's rotation")?,
            velocity: floats_or(velocity, &self.zeros(), "A body's velocity")?,
            angular_velocity: floats_or(
                angular_velocity,
                &self.angular_zeros(),
                "A body's angular velocity",
            )?,
            mass: float_or(mass, 0.0, "A body's mass")?,
            gravity_scale: float_or(gravity_scale, 1.0, "A body's gravity scale")?,
        };
        self.world()?.add_body(body).map_err(value_error)
    }

    /// Declares a shape on a body: ``kind`` is ``"ball"`` (``size``: its radius), ``"cuboid"``
    /// (its half extents), ``"capsule"`` (its half height along local y, then its radius) or
    /// ``"convex_hull"`` (its points, flat or one per row).
    #[pyo3(signature = (
        body, kind, size, *, position=None, rotation=None, density=None, friction=None,
        restitution=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add_shape(
        &self,
        body: u32,
        kind: &str,
        size: &Bound<'_, PyAny>,
        position: Option<&Bound<'_, PyAny>>,
        rotation: Option<&Bound<'_, PyAny>>,
        density: Option<&Bound<'_, PyAny>>,
        friction: Option<&Bound<'_, PyAny>>,
        restitution: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let size = floats(size, "A shape's size")?;
        let one = |what: &str| -> PyResult<f32> {
            match size[..] {
                [x] => Ok(x),
                _ => Err(PyValueError::new_err(format!(
                    "{what} takes one number, not {}.",
                    size.len()
                ))),
            }
        };
        let kind = match kind {
            "ball" => ShapeKind::Ball {
                radius: one("A ball's size")?,
            },
            "cuboid" => ShapeKind::Cuboid {
                half_extents: size.clone(),
            },
            "capsule" => match size[..] {
                [half_height, radius] => ShapeKind::Capsule {
                    half_height,
                    radius,
                },
                _ => {
                    return Err(PyValueError::new_err(
                        "A capsule's size is its half height, then its radius.",
                    ));
                }
            },
            "convex_hull" => ShapeKind::ConvexHull {
                points: size.clone(),
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "A shape is \"ball\", \"cuboid\", \"capsule\" or \"convex_hull\", not {other:?}."
                )));
            }
        };
        let shape = Shape {
            body,
            kind,
            position: floats_or(position, &self.zeros(), "A shape's position")?,
            rotation: floats_or(rotation, &self.identity_rotation(), "A shape's rotation")?,
            density: float_or(density, 1.0, "A shape's density")?,
            friction: float_or(friction, 0.0, "A shape's friction")?,
            restitution: float_or(restitution, 0.0, "A shape's restitution")?,
        };
        self.world()?.add_shape(shape).map_err(value_error)
    }

    /// Declares a joint; returns its number. ``kind`` is ``"pin"``, ``"slide"``, ``"weld"``,
    /// ``"ball"`` (3D), ``"rope"`` (``length``) or ``"spring"`` (``rest``, ``stiffness``,
    /// ``damping``). ``axis`` (a slide's, a 3D pin's) is in the first body's frame, of any
    /// length but zero; the joint's angle or offset is zero as the bodies are posed now.
    #[pyo3(signature = (
        kind, body1, body2, *, anchor1=None, anchor2=None, axis=None, limits=None, length=None,
        rest=None, stiffness=None, damping=None, contacts=false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add_joint(
        &self,
        kind: &str,
        body1: u32,
        body2: u32,
        anchor1: Option<&Bound<'_, PyAny>>,
        anchor2: Option<&Bound<'_, PyAny>>,
        axis: Option<&Bound<'_, PyAny>>,
        limits: Option<&Bound<'_, PyAny>>,
        length: Option<&Bound<'_, PyAny>>,
        rest: Option<&Bound<'_, PyAny>>,
        stiffness: Option<&Bound<'_, PyAny>>,
        damping: Option<&Bound<'_, PyAny>>,
        contacts: bool,
    ) -> PyResult<u32> {
        let kind = match kind {
            "pin" => JointKind::Pin,
            "slide" => JointKind::Slide,
            "weld" => JointKind::Weld,
            "ball" => JointKind::Ball,
            "rope" => JointKind::Rope {
                length: required(length, "A rope's length")?,
            },
            "spring" => JointKind::Spring {
                rest: required(rest, "A spring's rest length")?,
                stiffness: required(stiffness, "A spring's stiffness")?,
                damping: float_or(damping, 0.0, "A spring's damping")?,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "A joint is \"pin\", \"slide\", \"weld\", \"ball\", \"rope\" or \"spring\", \
                     not {other:?}."
                )));
            }
        };
        let limits = match limits {
            Some(limits) => match floats(limits, "A joint's limits")?[..] {
                [low, high] => Some([low, high]),
                _ => return Err(PyValueError::new_err("A joint's limits are [min, max].")),
            },
            None => None,
        };
        let joint = Joint {
            kind,
            body1,
            body2,
            anchor1: floats_or(anchor1, &self.zeros(), "A joint's first anchor")?,
            anchor2: floats_or(anchor2, &self.zeros(), "A joint's second anchor")?,
            axis: floats_or(axis, &[], "A joint's axis")?,
            limits,
            contacts,
        };
        self.world()?.add_joint(joint).map_err(value_error)
    }

    /// Declares a uniform gravity: an acceleration on every body, times its gravity scale.
    fn add_gravity(&self, acceleration: &Bound<'_, PyAny>) -> PyResult<()> {
        let acceleration = floats(acceleration, "Gravity")?;
        self.add(Force::Gravity { acceleration })
    }

    /// Declares attraction by the inverse square among ``bodies``:
    /// ``strength * m_a * m_b / r**2`` between each pair.
    fn add_attraction(&self, bodies: Vec<u32>, strength: &Bound<'_, PyAny>) -> PyResult<()> {
        let strength = float(strength, "An attraction's strength")?;
        self.add(Force::Attraction { strength, bodies })
    }

    /// Declares a damped spring between an anchor on each of two bodies, acting along the line
    /// between the anchors (zero for a tick at which they coincide).
    #[pyo3(signature = (
        body_a, body_b, *, stiffness, rest, damping=None, anchor_a=None, anchor_b=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add_spring(
        &self,
        body_a: u32,
        body_b: u32,
        stiffness: &Bound<'_, PyAny>,
        rest: &Bound<'_, PyAny>,
        damping: Option<&Bound<'_, PyAny>>,
        anchor_a: Option<&Bound<'_, PyAny>>,
        anchor_b: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let force = Force::Spring {
            body_a,
            anchor_a: floats_or(anchor_a, &self.zeros(), "A spring's first anchor")?,
            body_b,
            anchor_b: floats_or(anchor_b, &self.zeros(), "A spring's second anchor")?,
            stiffness: float(stiffness, "A spring's stiffness")?,
            damping: float_or(damping, 0.0, "A spring's damping")?,
            rest: float(rest, "A spring's rest length")?,
        };
        self.add(force)
    }

    /// Declares a drag ``-coefficient * v`` on a body.
    fn add_linear_drag(&self, body: u32, coefficient: &Bound<'_, PyAny>) -> PyResult<()> {
        let coefficient = float(coefficient, "A drag's coefficient")?;
        self.add(Force::LinearDrag { body, coefficient })
    }

    /// Declares a drag ``-coefficient * |v| * v`` on a body.
    fn add_quadratic_drag(&self, body: u32, coefficient: &Bound<'_, PyAny>) -> PyResult<()> {
        let coefficient = float(coefficient, "A drag's coefficient")?;
        self.add(Force::QuadraticDrag { body, coefficient })
    }

    /// Declares a constant force at a body's centre of mass and a constant torque.
    #[pyo3(signature = (body, *, force=None, torque=None))]
    fn add_constant(
        &self,
        body: u32,
        force: Option<&Bound<'_, PyAny>>,
        torque: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let force = Force::Constant {
            body,
            force: floats_or(force, &self.zeros(), "A constant force")?,
            torque: floats_or(torque, &self.angular_zeros(), "A constant torque")?,
        };
        self.add(force)
    }

    /// Declares a motor on a pin or a slide, driving its angle or offset (zero as declared)
    /// toward ``position`` and its rate toward ``velocity``, by a force (a torque for a pin) of
    /// ``stiffness`` per unit of offset and ``damping`` per unit of rate, whatever the masses.
    #[pyo3(signature = (joint, *, position=None, velocity=None, stiffness=None, damping=None))]
    fn add_motor(
        &self,
        joint: u32,
        position: Option<&Bound<'_, PyAny>>,
        velocity: Option<&Bound<'_, PyAny>>,
        stiffness: Option<&Bound<'_, PyAny>>,
        damping: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let force = Force::Motor {
            joint,
            position: float_or(position, 0.0, "A motor's position")?,
            velocity: float_or(velocity, 0.0, "A motor's velocity")?,
            stiffness: float_or(stiffness, 0.0, "A motor's stiffness")?,
            damping: float_or(damping, 0.0, "A motor's damping")?,
        };
        self.add(force)
    }

    /// A body's mass: its point mass and its shapes'.
    fn mass(&self, body: u32) -> PyResult<f32> {
        self.world()?.mass(body).map_err(value_error)
    }

    /// Computes the rows up to ``row`` if they are not yet computed; a lookup otherwise. The
    /// interpreter's lock is released meanwhile.
    fn advance_to(&self, py: Python<'_>, row: usize) -> PyResult<()> {
        py.detach(|| match self.inner.lock() {
            Ok(mut world) => world.advance_to(row).map_err(Some),
            Err(_) => Err(None),
        })
        .map_err(|err| match err {
            Some(err) => value_error(err),
            None => PyRuntimeError::new_err("the world failed in an earlier call"),
        })
    }

    /// An open world's next row, after one step under ``forces`` (``[bodies, dimension]``) and
    /// ``torques`` (``[bodies, 1]`` in 2D, ``[bodies, 3]`` in 3D) besides its force laws;
    /// returns its number.
    fn step(
        &self,
        py: Python<'_>,
        forces: &Bound<'_, PyAny>,
        torques: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let forces = floats(forces, "An open world's forces")?;
        let torques = floats(torques, "An open world's torques")?;
        py.detach(|| match self.inner.lock() {
            Ok(mut world) => world.step(&forces, &torques).map_err(Some),
            Err(_) => Err(None),
        })
        .map_err(|err| match err {
            Some(err) => value_error(err),
            None => PyRuntimeError::new_err("the world failed in an earlier call"),
        })
    }

    /// The rows ``start`` to ``stop`` of every body's pose, as little-endian float32 bytes, and
    /// their shape ``(rows, bodies, width)``.
    #[pyo3(signature = (start=0, stop=None))]
    fn _pose<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        stop: Option<usize>,
    ) -> PyResult<(Bound<'py, PyBytes>, (usize, usize, usize))> {
        let (start, stop) = self.span(start, stop)?;
        let world = self.world()?;
        let data = bytes_of(world.pose(start, stop));
        let shape = (stop - start, world.bodies(), world.pose_width());
        Ok((PyBytes::new(py, &data), shape))
    }

    /// As ``_pose``, for every body's velocity.
    #[pyo3(signature = (start=0, stop=None))]
    fn _velocity<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        stop: Option<usize>,
    ) -> PyResult<(Bound<'py, PyBytes>, (usize, usize, usize))> {
        let (start, stop) = self.span(start, stop)?;
        let world = self.world()?;
        let data = bytes_of(world.velocity(start, stop));
        let shape = (stop - start, world.bodies(), world.velocity_width());
        Ok((PyBytes::new(py, &data), shape))
    }

    /// The contact events as little-endian uint32 bytes, and their count.
    fn _contacts<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyBytes>, usize)> {
        let world = self.world()?;
        let events = world.contacts();
        let data: Vec<u8> = events
            .iter()
            .flatten()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        Ok((PyBytes::new(py, &data), events.len()))
    }

    /// What was found, as little-endian uint32 bytes (``row, kind, a, b``), and its count.
    fn _findings<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyBytes>, usize)> {
        let world = self.world()?;
        let found = world.findings();
        let data: Vec<u8> = found
            .iter()
            .flatten()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        Ok((PyBytes::new(py, &data), found.len()))
    }

    /// The units of the ``pose`` and ``velocity`` channels, one per number of a body's row.
    fn units(&self) -> PyResult<(&'static str, &'static str)> {
        Ok(self.world()?.units())
    }

    /// The declaration bytes: everything the world was given, in order.
    fn declaration_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, self.world()?.declaration_bytes()))
    }

    /// The trace bytes: the canonical bytes of the bake content its rows make.
    fn trace_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let data = self.world()?.trace_bytes();
        Ok(PyBytes::new(py, &data))
    }

    /// What computes its rows: producer, producer version, solver, solver version, integrator,
    /// build, and the features that change numbers.
    #[allow(clippy::type_complexity)]
    fn identity(&self) -> PyResult<(String, String, String, String, String, String, Vec<String>)> {
        let id = self.world()?.identity();
        Ok((
            id.producer,
            id.producer_version,
            id.solver,
            id.solver_version,
            id.integrator,
            id.build,
            id.features,
        ))
    }
}

impl PyWorld {
    fn add(&self, force: Force) -> PyResult<()> {
        self.world()?.add_force(force).map_err(value_error)
    }
}

#[pymodule]
fn _sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWorld>()?;
    m.add("DECLARATION_LAYOUT", crate::DECLARATION_LAYOUT)?;
    m.add("CONTENT_LAYOUT", crate::CONTENT_LAYOUT)?;
    m.add("SPEED_LIMIT", crate::SPEED_LIMIT)?;
    m.add("ANGULAR_SPEED_LIMIT", crate::ANGULAR_SPEED_LIMIT)?;
    let settings = crate::encode::Settings::of(2);
    m.add("SPEED_CAP", settings.speed_cap())?;
    m.add("ANGULAR_SPEED_CAP", settings.angular_speed_cap())?;
    m.add("NON_FINITE", finding::NON_FINITE)?;
    m.add("SPEED", finding::SPEED)?;
    m.add("ANGULAR_SPEED", finding::ANGULAR_SPEED)?;
    m.add("TUNNELLED", finding::TUNNELLED)?;
    Ok(())
}
