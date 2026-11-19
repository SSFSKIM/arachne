//! Arachne's simulation crate: rigid-body worlds in 2D and 3D, stepped on a fixed clock, whose
//! results are the same bytes on every machine of one architecture.
//!
//! The solver is pinned with its determinism feature, and the arithmetic it can reach is
//! audited (README.md, "Determinism"); `examples/probe` is the audit's program.
