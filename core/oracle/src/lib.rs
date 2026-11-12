//! The truth oracle: what Arachne's core should decide at every sub-sample of a frame, computed by
//! brute force from the XF stream alone.
//!
//! The rules are the truth contract's (`CONTRACT.md` beside this crate), never the core's code. The
//! oracle shares exactly three components with the core, each a contract term: the XF reader
//! (`arachne_core::xf`, the one core path it names), kurbo 0.13.1 (flattening and stroking) and
//! libm 0.2.16. `ci/oracle_check.py` keeps it so.
//!
//! - [`stream`] holds a stream's state across deltas and refuses what the oracle cannot judge;
//!   [`source`] reads each delta of a file alone, with XF 1.5's source section and its checks.
//! - [`camera`] places, projects and maps points (contract §2).
//! - [`draw`] turns the order's entries into what they draw: meshes' triangles, paths' layers and
//!   point blocks, each with its fragment identity.
//! - [`flatten`] makes a path layer's outline and rings through kurbo (contract §4, §8); [`guard`]
//!   draws curves far beyond the frame as chords and lays dashes under that guard (§4's
//!   `guard_chords@1`, §8's dashes), and [`joins`] applies the join decisions a record carries to
//!   kurbo's stroke (§8's `join_mask@1`).
//! - [`shift`] bounds how far the input grid may move a record on screen (contract §6, the screen
//!   shift `ε` that §3's and §4's margins read).
//! - [`cover`] decides coverage at every sample (contract §3, §4).
//! - Depth (contract §5, §6): [`depth`] fits and keys world planes; [`centreline`] keys a curve
//!   along its centreline; [`band`] holds the band's input terms; [`keys`] gives each fragment its
//!   key model and band at a pixel's samples.
//! - The order (contract §7): [`order`] gives a pair's admitted edges and label and tests an order's
//!   admissibility through `cycle_rank@2`; [`lines`] bounds `lines@1`'s share; [`pairs`] runs a
//!   frame's bands and pairs, with both stroke-depth adjustments.
//! - [`decide`] runs a frame; [`out`] writes and reads its decisions (`ORACLE_FORMAT` 2,
//!   `FORMAT.md`).

pub mod band;
pub mod bound;
pub mod camera;
pub mod centreline;
pub mod colour;
pub mod cover;
pub mod decide;
pub mod depth;
pub mod draw;
pub mod flatten;
pub mod guard;
pub mod joins;
pub mod keys;
pub mod label;
pub mod label_classes;
pub mod lines;
pub mod order;
pub mod out;
pub mod pairs;
pub mod shift;
pub mod source;
pub mod stream;
