//! The point-mass integrator: worlds with no shape and no joint (springs, orbits, N bodies, damped
//! oscillators), advanced by velocity Verlet (kick, drift, kick) in a fixed number of substeps per
//! tick, with every force law evaluated at every half kick. It is symplectic for forces that
//! depend on positions only, so a conservative system's energy stays bounded at a step a lesson's
//! periods resolve; a force that depends on velocity (drag, a spring's damping) is evaluated at the
//! half-step velocity.
//!
//! A point mass has no rotational inertia: its rotation stays as declared, and the crate refuses
//! what would turn it (an angular velocity, a torque, a spring anchored off its origin).

use crate::spec::{Body, Force};

pub(crate) struct Points {
    dim: usize,
    substeps: u32,
    dt: f32,
    x: Vec<[f32; 3]>,
    v: Vec<[f32; 3]>,
    /// The declared rotation, normalized in 3D, which never changes.
    rotation: Vec<[f32; 4]>,
    mass: Vec<f32>,
    fixed: Vec<bool>,
    gravity_scale: Vec<f32>,
    gravity: [f32; 3],
    forces: Vec<Force>,
}

fn vec3(v: &[f32]) -> [f32; 3] {
    let mut out = [0.0; 3];
    out[..v.len()].copy_from_slice(v);
    out
}

impl Points {
    pub fn new(dim: usize, dt: f32, substeps: u32) -> Self {
        Points {
            dim,
            substeps,
            dt,
            x: Vec::new(),
            v: Vec::new(),
            rotation: Vec::new(),
            mass: Vec::new(),
            fixed: Vec::new(),
            gravity_scale: Vec::new(),
            gravity: [0.0; 3],
            forces: Vec::new(),
        }
    }

    pub fn add_body(&mut self, b: &Body) {
        self.x.push(vec3(&b.position));
        self.v.push(vec3(&b.velocity));
        let rotation = if self.dim == 2 {
            [b.rotation[0], 0.0, 0.0, 0.0]
        } else {
            let q = &b.rotation;
            let r = 1.0 / (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            [q[0] * r, q[1] * r, q[2] * r, q[3] * r]
        };
        self.rotation.push(rotation);
        self.mass.push(b.mass);
        self.fixed.push(b.fixed);
        self.gravity_scale.push(b.gravity_scale);
    }

    pub fn add_force(&mut self, f: &Force) {
        if let Force::Gravity { acceleration } = f {
            let g = vec3(acceleration);
            for (total, g) in self.gravity.iter_mut().zip(g) {
                *total += g;
            }
        } else {
            self.forces.push(f.clone());
        }
    }

    pub fn mass(&self, body: usize) -> f32 {
        self.mass[body]
    }

    /// The force on each body at positions `x` and velocities `v`, plus `applied` (an open
    /// world's forces, constant over the tick).
    fn net_forces(
        &self,
        x: &[[f32; 3]],
        v: &[[f32; 3]],
        applied: Option<&[f32]>,
        out: &mut [[f32; 3]],
    ) {
        for (i, f) in out.iter_mut().enumerate() {
            let m = self.mass[i] * self.gravity_scale[i];
            *f = [
                self.gravity[0] * m,
                self.gravity[1] * m,
                self.gravity[2] * m,
            ];
        }
        for law in &self.forces {
            match law {
                Force::Attraction { strength, bodies } => {
                    for (k, &a) in bodies.iter().enumerate() {
                        for &b in &bodies[k + 1..] {
                            let (a, b) = (a as usize, b as usize);
                            let r = sub(x[b], x[a]);
                            let d2 = dot(r, r);
                            let s = strength * self.mass[a] * self.mass[b] / (d2 * d2.sqrt());
                            add(&mut out[a], scale(r, s));
                            add(&mut out[b], scale(r, -s));
                        }
                    }
                }
                Force::Spring {
                    body_a,
                    body_b,
                    stiffness,
                    damping,
                    rest,
                    ..
                } => {
                    // Anchored at the origins (the crate refuses other anchors here).
                    let (a, b) = (*body_a as usize, *body_b as usize);
                    let d = sub(x[b], x[a]);
                    let length = dot(d, d).sqrt();
                    let n = scale(d, 1.0 / length);
                    let separating = dot(sub(v[b], v[a]), n);
                    let f = scale(n, stiffness * (length - rest) + damping * separating);
                    add(&mut out[a], f);
                    add(&mut out[b], scale(f, -1.0));
                }
                Force::LinearDrag { body, coefficient } => {
                    let b = *body as usize;
                    add(&mut out[b], scale(v[b], -coefficient));
                }
                Force::QuadraticDrag { body, coefficient } => {
                    let b = *body as usize;
                    let speed = dot(v[b], v[b]).sqrt();
                    add(&mut out[b], scale(v[b], -coefficient * speed));
                }
                Force::Constant { body, force, .. } => {
                    add(&mut out[*body as usize], vec3(force));
                }
                Force::Gravity { .. } | Force::Motor { .. } => {}
            }
        }
        if let Some(applied) = applied {
            for (i, f) in out.iter_mut().enumerate() {
                add(f, vec3(&applied[i * self.dim..(i + 1) * self.dim]));
            }
        }
    }

    /// One tick: `substeps` steps of velocity Verlet.
    pub fn step(&mut self, applied: Option<&[f32]>) {
        let h = self.dt / self.substeps as f32;
        let half = 0.5 * h;
        let mut f = vec![[0.0f32; 3]; self.x.len()];
        for _ in 0..self.substeps {
            self.net_forces(&self.x, &self.v, applied, &mut f);
            self.kick(&f, half);
            for ((x, v), fixed) in self.x.iter_mut().zip(&self.v).zip(&self.fixed) {
                if !fixed {
                    add(x, scale(*v, h));
                }
            }
            self.net_forces(&self.x, &self.v, applied, &mut f);
            self.kick(&f, half);
        }
    }

    fn kick(&mut self, f: &[[f32; 3]], h: f32) {
        for (i, f) in f.iter().enumerate() {
            if !self.fixed[i] {
                let a = scale(*f, h / self.mass[i]);
                add(&mut self.v[i], a);
            }
        }
    }

    /// Each body's pose (`x, y, angle` or `x, y, z, qx, qy, qz, qw`) and velocity (`vx, vy, w` or
    /// `vx, vy, vz, wx, wy, wz`).
    pub fn record(&self, pose: &mut Vec<f32>, velocity: &mut Vec<f32>) {
        for ((x, v), rotation) in self.x.iter().zip(&self.v).zip(&self.rotation) {
            pose.extend_from_slice(&x[..self.dim]);
            if self.dim == 2 {
                pose.push(rotation[0]);
                velocity.extend_from_slice(&[v[0], v[1], 0.0]);
            } else {
                pose.extend_from_slice(rotation);
                velocity.extend_from_slice(v);
                velocity.extend_from_slice(&[0.0; 3]);
            }
        }
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn add(a: &mut [f32; 3], b: [f32; 3]) {
    a[0] += b[0];
    a[1] += b[1];
    a[2] += b[2];
}
