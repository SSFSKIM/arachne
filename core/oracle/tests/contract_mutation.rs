//! The contract's mutation witnesses (§12, "Cases the contract must reject"), each with its
//! soundness twin: a scene built here as §12 gives it, the core's report of it (or, for a label
//! witness, the report a core keeping §10 gives), the one defect §12 injects, and the verdict the
//! oracle's comparison must give at the samples it names: the defect's, and agreement (or the
//! named finding) for the report without it.
//!
//! A fixture directory under `tests/fixtures/` holds `scene.sha256` (the sha256 of the scene's XF
//! file), `defect.json` (the one quantity the defect changes) and `expected.json` (the verdicts,
//! the pixel and samples). Nothing binary is committed; a test holds the files to what is stated
//! here, and `ORACLE_BLESS_FIXTURES=1` rewrites them from it. The report is read from the core's
//! diagnostics, traced at the one pixel judged.

mod common;
#[path = "common/judge.rs"]
mod judge;
#[path = "common/scenes.rs"]
mod scenes;

use std::path::{Path, PathBuf};

use arachne_core::diagnostics::Layer as CoreLayer;
use arachne_core::stream::Stream;
use arachne_core::xf::{Delta, Paint, View, wire};
use arachne_oracle::cover::sample_cell;
use arachne_oracle::decide::decide_at_wire;
use arachne_oracle::out::Decisions;
use arachne_oracle::source;
use common::sha256_hex;
use judge::{Entry, Identity, Reported, Verdict, judge};
use scenes::*;

/// A fragment as a witness names it: its record's slot and its layer (`ORACLE_FORMAT` codes),
/// and its index within the layer (`sub`).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Frag {
    slot: u32,
    /// Its object is its slot, as the engine numbers them, not `100 + slot`, as the scene kit does.
    engine: bool,
    /// Its order entry's part: a face group's face.
    part: u32,
    layer: u8,
    sub: u32,
}

const BG: u8 = 0;
const FILL: u8 = 1;
const STROKE: u8 = 2;
const MESH: u8 = 3;

fn fill(slot: u32) -> Frag {
    Frag {
        slot,
        engine: false,
        part: 0,
        layer: FILL,
        sub: 1,
    }
}
/// The fill of the engine's record in `slot`.
fn engine_fill(slot: u32) -> Frag {
    Frag {
        engine: true,
        ..fill(slot)
    }
}
/// Face `part` of the face group in `slot`: its fill.
fn face(slot: u32, part: u32) -> Frag {
    Frag { part, ..fill(slot) }
}
fn stroke(slot: u32) -> Frag {
    Frag {
        slot,
        engine: false,
        part: 0,
        layer: STROKE,
        sub: 2,
    }
}
fn background(slot: u32) -> Frag {
    Frag {
        slot,
        engine: false,
        part: 0,
        layer: BG,
        sub: 0,
    }
}
fn tri(slot: u32, i: u32) -> Frag {
    Frag {
        slot,
        engine: false,
        part: 0,
        layer: MESH,
        sub: i,
    }
}

impl Frag {
    fn identity(self) -> Identity {
        let object = if self.engine {
            self.slot
        } else {
            100 + self.slot
        };
        (object, self.part, self.layer, self.sub)
    }
    fn json(self) -> String {
        let layer = ["background_stroke", "fill", "stroke", "mesh", "points"][self.layer as usize];
        let part = if self.part == 0 {
            String::new()
        } else {
            format!(", \"part\": {}", self.part)
        };
        format!(
            "{{\"slot\": {}{part}, \"layer\": \"{layer}\", \"sub\": {}}}",
            self.slot, self.sub
        )
    }
}

/// The one recorded quantity a defect changes, at the witness's samples.
#[derive(Clone, Debug)]
enum Defect {
    /// The sample's order, front first.
    Order(Vec<Frag>),
    /// The fragment reported absent.
    Absent(Frag),
    /// The fragment reported present with these cell bytes.
    Present(Frag, u8),
    /// The fragment's cell bytes.
    Cells(Frag, u8),
}

impl Defect {
    fn json(&self, pixel: (u32, u32), samples: &[usize]) -> String {
        let (q, v) = match self {
            Defect::Order(o) => (
                "order",
                format!(
                    "[{}]",
                    o.iter().map(|f| f.json()).collect::<Vec<_>>().join(", ")
                ),
            ),
            Defect::Absent(f) => ("absent", f.json()),
            Defect::Present(f, b) => (
                "present",
                format!("{{\"fragment\": {}, \"cell_byte\": {b}}}", f.json()),
            ),
            Defect::Cells(f, b) => (
                "cell_bytes",
                format!("{{\"fragment\": {}, \"cell_byte\": {b}}}", f.json()),
            ),
        };
        format!(
            "{{\n  \"quantity\": \"{q}\",\n  \"pixel\": [{}, {}],\n  \"samples\": {samples:?},\n  \"value\": {v}\n}}\n",
            pixel.0, pixel.1
        )
    }
}

/// The report a witness's conforming case is: the core's own, or the core's with the order the
/// label rule gives at the witness's samples.
#[derive(Clone, Debug)]
enum Conforming {
    Core,
    Rule(Vec<Frag>),
}

/// The samples a witness names.
#[derive(Clone, Debug)]
enum Samples {
    All,
    These(Vec<usize>),
    /// Where every one of these fragments is present in the oracle's decisions.
    Covered(Vec<Frag>),
    /// The samples of these cell rows.
    CellRows(Vec<usize>),
    /// The samples of these cell columns.
    CellColumns(Vec<usize>),
}

struct Witness {
    name: &'static str,
    rule: &'static str,
    /// The stream's bytes, and the deltas the core draws (the same drawing, without the source
    /// section the core's reader does not read).
    xf: Vec<u8>,
    deltas: Vec<Delta>,
    seq: u64,
    pixel: (u32, u32),
    samples: Samples,
    conforming: Conforming,
    /// The conforming report's verdict, and the defect with its verdict.
    twin: Verdict,
    defect: Option<(Defect, Verdict)>,
}

fn mismatch() -> Verdict {
    Verdict::Mismatch("mismatch")
}

fn one(b: &Build, view: View) -> (Vec<u8>, Vec<Delta>) {
    let d = vec![b.delta(1, view)];
    (wire::encode_file(&d), d)
}

fn witness(
    name: &'static str,
    rule: &'static str,
    (xf, deltas): (Vec<u8>, Vec<Delta>),
    pixel: (u32, u32),
    samples: Samples,
    defect: Option<(Defect, Verdict)>,
) -> Witness {
    let seq = deltas.last().unwrap().seq;
    Witness {
        name,
        rule,
        xf,
        deltas,
        seq,
        pixel,
        samples,
        conforming: Conforming::Core,
        twin: Verdict::Agree,
        defect,
    }
}

fn square_pair(view: View, world: bool, axis: bool) -> (Vec<u8>, Vec<Delta>) {
    let mut b = Build::new(world);
    let z = 0.0;
    b.fill(rect([-4.0, 2.0], [-4.0, 2.0], z), OPAQUE_RED);
    b.fill(rect([-2.0, 4.0], [-2.0, 4.0], z), OPAQUE_BLUE);
    if axis {
        for r in &mut b.records {
            r.planes = Some(vec![6]);
        }
    }
    one(&b, view)
}

fn surface_and_line(from: [f64; 3], to: [f64; 3]) -> (Vec<u8>, Vec<Delta>) {
    let mut b = Build::new(true);
    b.fill(rect([-6.0, 6.0], [-6.0, 6.0], 0.0), OPAQUE_BLUE);
    b.stroke(open(&[from, to]), WHITE, 0.5);
    one(&b, view_c())
}

/// W1 to W17's mutations (W13, W14 and W17's reference scene apart).
fn depth_witnesses() -> Vec<Witness> {
    let mut out = Vec::new();
    out.push(witness(
        "exact_tie_2d",
        "exact_tie",
        square_pair(view_f(), false, false),
        (8, 8),
        Samples::All,
        Some((Defect::Order(vec![fill(0), fill(1)]), mismatch())),
    ));
    out.push(witness(
        "exact_tie_axis_fills",
        "exact_tie",
        square_pair(view_c(), true, true),
        (32, 32),
        Samples::All,
        Some((Defect::Order(vec![fill(0), fill(1)]), mismatch())),
    ));
    let mut b = Build::new(false);
    b.stroke(open(&[[-6.0, -0.5, 0.0], [6.0, -0.5, 0.0]]), BLACK, 0.25);
    out.push(witness(
        "thin_stroke_present",
        "presence_band",
        one(&b, view_f()),
        (8, 8),
        Samples::CellRows(vec![1, 2]),
        Some((Defect::Absent(stroke(0)), mismatch())),
    ));
    let mut b = Build::new(true);
    b.mesh(
        vec![[-6.0, -6.0, 0.0], [6.0, -6.0, 0.0], [0.0, 6.0, 0.0]],
        vec![[0, 1, 2]],
        OPAQUE_RED,
    );
    b.fill(
        polygon(&[
            [-6.0, -6.0, -3.0],
            [6.0, -6.0, 3.0],
            [6.0, 6.0, 3.0],
            [-6.0, 6.0, -3.0],
        ]),
        OPAQUE_BLUE,
    );
    out.push(witness(
        "mesh_face_reversed",
        "band",
        one(&b, view_c()),
        (36, 32),
        Samples::All,
        Some((Defect::Order(vec![tri(0, 0), fill(1)]), mismatch())),
    ));
    let mut b = Build::new(false);
    b.fill(rect([-4.0, 4.0], [-4.0, 4.0], 0.0), [1.0, 0.0, 0.0, 0.003]);
    out.push(witness(
        "alpha_rounds_to_zero",
        "alpha_round@1",
        one(&b, view_f()),
        (8, 8),
        Samples::All,
        Some((Defect::Present(fill(0), 255), mismatch())),
    ));
    let mut b = Build::new(false);
    b.mesh(
        vec![[-6.0, -6.0, 0.0], [6.0, -6.0, 0.0], [0.0, 6.0, 0.0]],
        vec![[0, 1, 2]],
        OPAQUE_RED,
    );
    out.push(witness(
        "mesh_inside_margin",
        "mesh_margin",
        one(&b, view_f()),
        (8, 8),
        Samples::All,
        Some((Defect::Absent(tri(0, 0)), mismatch())),
    ));
    let mut b = Build::new(false);
    b.fill(rect([21.0 / 32.0, 4.0], [-4.0, 4.0], 0.0), WHITE);
    out.push(witness(
        "cell_byte_outside",
        "presence_band",
        one(&b, view_f()),
        (8, 8),
        Samples::CellColumns(vec![2]),
        Some((Defect::Cells(fill(0), 104), mismatch())),
    ));
    out.push(witness(
        "line_before_surface",
        "curve_s",
        surface_and_line([-4.5, 0.0, 1.0], [4.5, 0.0, 1.0]),
        (32, 31),
        Samples::Covered(vec![fill(0), stroke(1)]),
        Some((Defect::Order(vec![fill(0), stroke(1)]), mismatch())),
    ));
    out.push(witness(
        "line_through_surface",
        "may_share",
        surface_and_line([-4.5, 0.0, -1.5], [4.5, 0.0, 1.5]),
        (20, 31),
        Samples::Covered(vec![fill(0), stroke(1)]),
        Some((Defect::Order(vec![stroke(1), fill(0)]), mismatch())),
    ));
    let mut b = Build::new(true);
    b.fill(rect([-6.0, 6.0], [-6.0, 6.0], 0.0), OPAQUE_RED);
    b.fill(
        polygon(&[
            [-6.0, -6.0, -1.5],
            [6.0, -6.0, 1.5],
            [6.0, 6.0, 1.5],
            [-6.0, 6.0, -1.5],
        ]),
        OPAQUE_BLUE,
    );
    out.push(witness(
        "faces_past_parallel",
        "near_parallel@1",
        one(&b, view_c()),
        (40, 32),
        Samples::All,
        Some((Defect::Order(vec![fill(0), fill(1)]), mismatch())),
    ));
    let mut b = Build::new(true);
    b.stroke(open(&[[-3.1875, 0.0, 0.0], [3.1875, 0.0, 0.0]]), WHITE, 0.5);
    b.stroke(
        open(&[[0.0, -3.1875, 1.0], [0.0, 3.1875, 1.0]]),
        OPAQUE_GREEN,
        0.5,
    );
    out.push(witness(
        "lines_crossing_apart",
        "lines@1",
        one(&b, view_c()),
        (32, 32),
        Samples::Covered(vec![stroke(0), stroke(1)]),
        Some((Defect::Order(vec![stroke(0), stroke(1)]), mismatch())),
    ));
    let mut b = Build::new(true);
    let r = b.fill(rect([-4.0, 4.0], [-4.0, 4.0], 0.0), OPAQUE_RED);
    r.paint.stroke = Paint::One(OPAQUE_BLUE);
    r.paint.stroke_width = 0.5;
    out.push(witness(
        "stroke_under_own_fill",
        "own_layers@1",
        one(&b, view_c()),
        (47, 32),
        Samples::All,
        Some((Defect::Order(vec![fill(0), stroke(0)]), mismatch())),
    ));
    out.push(witness(
        "cycle_breaks_strong",
        "cycle_rank@1",
        cycle_scene(),
        (32, 31),
        Samples::These(vec![12, 13, 15]),
        Some((
            Defect::Order(vec![fill(0), background(0), fill(1)]),
            mismatch(),
        )),
    ));
    let mut b = Build::new(true);
    b.stroke(
        open(&[[-4.25, 0.0, 0.0], [2.125, 0.0, 0.0]]),
        [1.0, 0.0, 0.0, 0.5],
        0.5,
    );
    b.stroke(
        open(&[[-2.125, 0.0, 0.0], [4.25, 0.0, 0.0]]),
        [0.0, 0.0, 1.0, 0.5],
        0.5,
    );
    let r = b.fill(rect([-6.0, 6.0], [-6.0, 6.0], 0.0), [0.0, 1.0, 0.0, 0.0]);
    r.paint.stroke = Paint::One([0.0, 1.0, 0.0, 0.0]);
    r.paint.stroke_width = 0.5;
    out.push(Witness {
        twin: Verdict::Mismatch("invisible_registry_offer"),
        ..witness(
            "invisible_registry_reference",
            "registry_eligibility",
            registry_reference(),
            (80, 45),
            Samples::Covered(vec![stroke(0), stroke(1)]),
            None,
        )
    });
    out.push(witness(
        "invisible_registry_offer",
        "registry_eligibility",
        one(&b, view_c()),
        (32, 32),
        Samples::All,
        Some((
            Defect::Order(vec![stroke(0), stroke(1)]),
            Verdict::Mismatch("invisible_registry_offer"),
        )),
    ));
    out
}

/// W17's reference scene: 160 x 90 px, a 3D camera at `phi = 0.8`, two half-transparent lines
/// 60 wide on `z = 0`, then a polygon on `z = 0` that paints nothing.
fn registry_reference() -> (Vec<u8>, Vec<Delta>) {
    let view = View {
        pixel_width: 160,
        pixel_height: 90,
        frame_width: 14.222222328186035,
        frame_height: 8.0,
        camera: arachne_core::xf::Camera::ThreeD(arachne_core::xf::ThreeDCamera {
            phi: 0.800000011920929,
            theta: -1.5707963109016418,
            gamma: 0.0,
            focal_distance: 20.0,
            zoom: 1.0,
            exponential_projection: false,
            light_source: [-7.0, -9.0, 10.0],
            shading_factor: 0.2,
            should_apply_shading: false,
        }),
        cairo_line_width_multiple: 0.01,
        input_grid: 1.0 / 4096.0,
        ..view_c()
    };
    let mut b = Build::new(true);
    b.stroke(
        open(&[[-3.0, -0.1, 0.0], [3.0, -0.1, 0.0]]),
        [1.0, 0.0, 0.0, 0.5],
        60.0,
    );
    b.stroke(
        open(&[[-3.0, 0.1, 0.0], [3.0, 0.1, 0.0]]),
        [0.0, 0.0, 1.0, 0.5],
        60.0,
    );
    let r = b.fill(
        polygon(&[
            [-4.0, -1.0, 0.0],
            [4.0, -1.0, 0.0],
            [4.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0],
        ]),
        [1.0, 1.0, 1.0, 0.0],
    );
    r.paint.stroke = Paint::One([1.0, 1.0, 1.0, 0.0]);
    r.paint.stroke_width = 0.0;
    one(&b, view)
}

/// The label witnesses the core this contract describes is judged on as it draws them: W5, a
/// label under a surface that covers its centre nearer, and W18, a nearer line over a label.
fn label_core_witnesses() -> Vec<Witness> {
    let mut out = Vec::new();
    let mut b = Build::new(true);
    b.fill(rect([-3.0, 3.0], [-3.0, 3.0], 2.0), OPAQUE_BLUE);
    b.label([0.0; 3]);
    out.push(witness(
        "label_under_surface",
        "label_placement@2",
        one(&b, view_c()),
        (32, 32),
        Samples::All,
        Some((Defect::Order(vec![fill(1), fill(0)]), mismatch())),
    ));
    let mut b = Build::new(true);
    b.label([0.0; 3]);
    b.stroke(
        open(&[[-3.1875, 0.0, 2.0], [3.1875, 0.0, 2.0]]),
        OPAQUE_GREEN,
        0.5,
    );
    out.push(Witness {
        twin: Verdict::Mismatch("label_plate_model"),
        ..witness(
            "label_over_line",
            "label_placement@2",
            one(&b, view_c()),
            (32, 32),
            Samples::All,
            None,
        )
    });
    out
}

const POLYGON: &str = "arachne.mobject.geometry.polygram.Polygon";
const SURFACE: &str = "arachne.mobject.three_d.three_dimensions.Surface";
const CUBE: &str = "arachne.mobject.three_d.three_dimensions.Cube";
const CYLINDER: &str = "arachne.mobject.three_d.three_dimensions.Cylinder";
const VECTOR: &str = "arachne.mobject.geometry.line.Vector";
const CIRCLE: &str = "arachne.mobject.geometry.arc.Circle";

/// A stream of XF 1.5 deltas, each with its source rows, and the same deltas without them for
/// the core, whose drawing they do not change.
fn xf15(parts: Vec<(Delta, Vec<Row>)>) -> (Vec<u8>, Vec<Delta>) {
    let bytes: Vec<Vec<u8>> = parts
        .iter()
        .map(|(d, rows)| delta15(d, &section(rows, None)))
        .collect();
    (file15(&bytes), parts.into_iter().map(|(d, _)| d).collect())
}

/// One XF 1.5 delta of `b`: the label's records (`label`) with a row of facing group 1, every
/// other record with a row of facing group 0 and the class `classes[slot]` (or none, where it is
/// `None`), each its own object.
fn one15(
    b: &Build,
    view: View,
    label: &[u32],
    classes: &[Option<&[(u32, &str)]>],
) -> (Vec<u8>, Vec<Delta>) {
    let mut rows = Vec::new();
    for r in &b.records {
        if label.contains(&r.slot) {
            rows.push(row(r.slot, 1, &[&[(900, POLYGON)]]));
        } else if let Some(Some(path)) = classes.get(r.slot as usize) {
            rows.push(row(r.slot, 0, &[path]));
        }
    }
    xf15(vec![(b.delta(1, view), rows)])
}

/// A label witness: the conforming report is the core's with the rule's order at its samples.
fn by_rule(
    name: &'static str,
    xf: (Vec<u8>, Vec<Delta>),
    pixel: (u32, u32),
    samples: Samples,
    order: Vec<Frag>,
    defect: Option<(Defect, Verdict)>,
) -> Witness {
    Witness {
        conforming: Conforming::Rule(order),
        ..witness(name, "label_placement@2", xf, pixel, samples, defect)
    }
}

/// A witness the oracle leaves open (`label_rule_open`) whatever is reported.
fn open_case(
    name: &'static str,
    xf: (Vec<u8>, Vec<Delta>),
    pixel: (u32, u32),
    samples: Samples,
    pair: u8,
    class: u8,
) -> Witness {
    Witness {
        twin: Verdict::Open(pair, class),
        ..witness(name, "label_placement@2", xf, pixel, samples, None)
    }
}

fn plate() -> Verdict {
    Verdict::Mismatch("label_plate_model")
}

/// View O's sliver scene: `L` over screen `x` 70 to 110 and `y` 28 to 36 at `z = 0`, centre at
/// screen `x` 80; `S` over screen `x` 90 to `e`, `y` 30 to 34, at depth `z = (c − p) / 40`.
fn sliver(c: f64, e: f64) -> (Vec<u8>, Vec<Delta>) {
    let wx = |p: f64| (p - 64.0) / 4.0;
    let z = |p: f64| (c - p) / 40.0;
    let mut b = Build::new(true);
    b.fill(
        polygon(&[
            [wx(90.0), -0.5, z(90.0)],
            [wx(e), -0.5, z(e)],
            [wx(e), 0.5, z(e)],
            [wx(90.0), 0.5, z(90.0)],
        ]),
        OPAQUE_BLUE,
    );
    b.glyph(
        rect([wx(70.0), wx(110.0)], [-1.0, 1.0], 0.0),
        WHITE,
        [4.0, 0.0, 0.0],
    );
    one15(&b, view_o(), &[1], &[Some(&[(1, POLYGON)])])
}

/// Two fills of one surface object on `z = 0`, and `L` at `(0, 0, −4)`: `F1` and `F2`, `F2`'s
/// third corner `δ` above `F1`'s, `F2`'s corners listed backwards where `reversed`.
fn two_sheets(delta: f64, reversed: bool) -> (Vec<u8>, Vec<Delta>) {
    let mut b = Build::new(true);
    b.fill(
        polygon(&[[-2.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]),
        OPAQUE_BLUE,
    );
    let mut f2 = vec![[0.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0 + delta, 0.0]];
    if reversed {
        f2.reverse();
    }
    b.fill(polygon(&f2), OPAQUE_BLUE);
    b.label([0.0, 0.0, -4.0]);
    one15(
        &b,
        view_c(),
        &[2],
        &[
            Some(&[(1, POLYGON), (7, SURFACE)]),
            Some(&[(2, POLYGON), (7, SURFACE)]),
        ],
    )
}

/// W5's scene with `G` given as `g` makes it, and `L` (a row of facing group 1) with `extra`
/// records in its facing group.
fn w5_with(g: impl FnOnce(&mut Build), extra: impl FnOnce(&mut Build)) -> Build {
    let mut b = Build::new(true);
    g(&mut b);
    b.label([0.0; 3]);
    extra(&mut b);
    b
}

fn blue_square_at_2(b: &mut Build) {
    b.fill(rect([-3.0, 3.0], [-3.0, 3.0], 2.0), OPAQUE_BLUE);
}

fn mesh_square_at_2(b: &mut Build) {
    b.mesh(
        vec![
            [-3.0, -3.0, 2.0],
            [3.0, -3.0, 2.0],
            [3.0, 3.0, 2.0],
            [-3.0, 3.0, 2.0],
        ],
        vec![[0, 1, 2], [0, 2, 3]],
        OPAQUE_BLUE,
    );
}

fn pre_projected_in_label(b: &mut Build) {
    b.pre_projected(rect([-6.5, -5.5], [-6.5, -5.5], 0.0), WHITE);
}

/// W20's box: a cube of side 4 about the origin as one face group (front, back, then the four
/// sides), each face wound outward, fill `(0, 0, 1, 0.5)`; then `L` at `centre`. Seen from the
/// camera the front face winds one way, and the back and the sides, facing away, the other.
fn cube(centre: [f64; 3]) -> Build {
    let faces = [
        [
            [-2.0, -2.0, 2.0],
            [2.0, -2.0, 2.0],
            [2.0, 2.0, 2.0],
            [-2.0, 2.0, 2.0],
        ],
        [
            [-2.0, -2.0, -2.0],
            [-2.0, 2.0, -2.0],
            [2.0, 2.0, -2.0],
            [2.0, -2.0, -2.0],
        ],
        [
            [2.0, -2.0, -2.0],
            [2.0, 2.0, -2.0],
            [2.0, 2.0, 2.0],
            [2.0, -2.0, 2.0],
        ],
        [
            [-2.0, -2.0, -2.0],
            [-2.0, -2.0, 2.0],
            [-2.0, 2.0, 2.0],
            [-2.0, 2.0, -2.0],
        ],
        [
            [-2.0, 2.0, -2.0],
            [-2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, -2.0],
        ],
        [
            [-2.0, -2.0, -2.0],
            [2.0, -2.0, -2.0],
            [2.0, -2.0, 2.0],
            [-2.0, -2.0, 2.0],
        ],
    ]
    .map(|f| polygon(&f));
    let mut b = Build::new(true);
    b.face_group(&faces, [0.0, 0.0, 1.0, 0.5]);
    b.label(centre);
    b
}

/// W19 to W26 (W25 apart) and the point reading's witnesses.
fn label_rule_witnesses() -> Vec<Witness> {
    let mut out = Vec::new();
    // W19.
    let mut b = Build::new(true);
    b.fill(rect([-3.0, 3.0], [-3.0, 3.0], 2.0), [0.0, 0.0, 1.0, 0.5]);
    b.stroke(
        open(&[[-3.1875, 0.0, 2.5], [3.1875, 0.0, 2.5]]),
        OPAQUE_GREEN,
        0.5,
    );
    b.label([0.0; 3]);
    out.push(by_rule(
        "label_backed_under_line",
        one(&b, view_c()),
        (32, 32),
        Samples::Covered(vec![fill(0), stroke(1), fill(2)]),
        vec![stroke(1), fill(0), fill(2)],
        Some((Defect::Order(vec![fill(2), stroke(1), fill(0)]), mismatch())),
    ));
    // W20, and its soundness twin with `L` behind the box.
    for (name, centre, order, defect) in [
        (
            "label_inside_box",
            [0.0; 3],
            vec![face(0, 0), fill(1), face(0, 1)],
            Some((
                Defect::Order(vec![face(0, 0), face(0, 1), fill(1)]),
                mismatch(),
            )),
        ),
        (
            "label_behind_box",
            [0.0, 0.0, -4.0],
            vec![face(0, 0), face(0, 1), fill(1)],
            None,
        ),
    ] {
        out.push(by_rule(
            name,
            one15(&cube(centre), view_c(), &[1], &[Some(&[(3, CUBE)])]),
            (32, 32),
            Samples::Covered(vec![face(0, 0), face(0, 1), fill(1)]),
            order,
            defect,
        ));
    }
    // W21.
    let mut b = Build::new(true);
    b.fill(
        polygon(&[
            [0.5, -1.5, -0.25],
            [1.5, -1.5, -2.25],
            [1.5, 1.5, -2.25],
            [0.5, 1.5, -0.25],
        ]),
        OPAQUE_BLUE,
    );
    b.label([0.0; 3]);
    out.push(by_rule(
        "label_off_centre_farther",
        one(&b, view_c()),
        (36, 32),
        Samples::Covered(vec![fill(0), fill(1)]),
        vec![fill(1), fill(0)],
        Some((Defect::Order(vec![fill(0), fill(1)]), mismatch())),
    ));
    // W22.
    let mut b = Build::new(true);
    b.fill(
        polygon(&[
            [0.5, -1.5, 1.2],
            [2.5, -1.5, -2.0],
            [2.5, 1.5, -2.0],
            [0.5, 1.5, 1.2],
        ]),
        OPAQUE_BLUE,
    );
    b.label([0.0; 3]);
    out.push(by_rule(
        "label_genuine_straddle",
        one(&b, view_c()),
        (35, 32),
        Samples::Covered(vec![fill(0), fill(1)]),
        vec![fill(1), fill(0)],
        Some((Defect::Order(vec![fill(0), fill(1)]), plate())),
    ));
    // W23.
    let mut b = Build::new(true);
    b.fill(
        rect([0.5, 1.5], [-1.5, 1.5], 1.0 / 1_073_741_824.0),
        OPAQUE_BLUE,
    );
    b.label([0.0; 3]);
    out.push(open_case(
        "label_interval_straddle",
        one(&b, view_c()),
        (35, 32),
        Samples::Covered(vec![fill(0), fill(1)]),
        1,
        0,
    ));
    // W24.
    let mut b = Build::new(true);
    b.fill(rect([-3.0, 3.0], [-3.0, 3.0], 2.0), [0.0, 0.0, 1.0, 0.5]);
    b.label([0.0, 0.0, 3.0]);
    b.label([0.0; 3]);
    out.push(by_rule(
        "label_placement_over_order",
        one(&b, view_c()),
        (32, 32),
        Samples::All,
        vec![fill(1), fill(0), fill(2)],
        Some((Defect::Order(vec![fill(2), fill(1), fill(0)]), mismatch())),
    ));
    // W26.
    let mut b = Build::new(true);
    b.label([-1.0, 0.0, 0.0]);
    b.fill(rect([-0.75, 2.0], [-3.0, 3.0], 1.0), OPAQUE_BLUE);
    out.push(by_rule(
        "label_cross_tile",
        one(&b, view_c_wide()),
        (64, 32),
        Samples::Covered(vec![fill(0), fill(1)]),
        vec![fill(1), fill(0)],
        Some((Defect::Order(vec![fill(0), fill(1)]), mismatch())),
    ));
    // The point reading.
    for (name, c, e, qualifies) in [
        ("label_points_sliver_between", 100.10, 100.15, true),
        ("label_points_sliver_holds_one", 100.08, 100.12, false),
        ("label_points_sliver_across_cells", 100.225, 100.275, true),
    ] {
        let (order, defect) = if qualifies {
            (vec![fill(0), fill(1)], (vec![fill(1), fill(0)], mismatch()))
        } else {
            (vec![fill(1), fill(0)], (vec![fill(0), fill(1)], plate()))
        };
        out.push(by_rule(
            name,
            sliver(c, e),
            (95, 32),
            Samples::All,
            order,
            Some((Defect::Order(defect.0), defect.1)),
        ));
    }
    out.push(by_rule(
        "insertion_at_points_held",
        sliver(100.225, 100.275),
        (100, 32),
        Samples::CellColumns(vec![0]),
        vec![fill(0), fill(1)],
        Some((Defect::Order(vec![fill(1), fill(0)]), mismatch())),
    ));
    out.push(by_rule(
        "insertion_at_points_not_held",
        sliver(100.225, 100.275),
        (100, 32),
        Samples::CellColumns(vec![1]),
        vec![fill(1), fill(0)],
        Some((Defect::Order(vec![fill(0), fill(1)]), mismatch())),
    ));
    out.push(by_rule(
        "sheet_join_quantum",
        two_sheets(1.0 / 65_536.0, false),
        (30, 32),
        Samples::Covered(vec![fill(0), fill(2)]),
        vec![fill(0), fill(2)],
        Some((Defect::Order(vec![fill(2), fill(0)]), mismatch())),
    ));
    out.push(open_case(
        "sheet_join_near_miss",
        two_sheets(3.0 / 65_536.0, false),
        (30, 32),
        Samples::Covered(vec![fill(0), fill(2)]),
        3,
        0,
    ));
    out.push(open_case(
        "sheet_join_near_miss_second",
        two_sheets(3.0 / 65_536.0, false),
        (34, 32),
        Samples::Covered(vec![fill(1), fill(2)]),
        3,
        0,
    ));
    out.push(by_rule(
        "sheets_opposite_winding",
        two_sheets(0.0, true),
        (34, 32),
        Samples::Covered(vec![fill(1), fill(2)]),
        vec![fill(1), fill(2)],
        Some((Defect::Order(vec![fill(2), fill(1)]), mismatch())),
    ));
    // The ring glyph whose hole holds the centre.
    let mut b = Build::new(true);
    b.fill(rect([-0.25, 0.25], [-0.25, 0.25], 0.0), OPAQUE_BLUE);
    b.fill(rect([1.0, 1.25], [-0.25, 0.25], 0.0), OPAQUE_BLUE);
    let mut ring = rect([-1.5, 1.5], [-1.5, 1.5], -4.0);
    ring.extend(polygon(&[
        [-0.5, -0.5, -4.0],
        [-0.5, 0.5, -4.0],
        [0.5, 0.5, -4.0],
        [0.5, -0.5, -4.0],
    ]));
    b.glyph(ring, WHITE, [0.0, 0.0, -4.0]);
    out.push(by_rule(
        "no_pair_centre_without_points",
        one15(
            &b,
            view_c(),
            &[2],
            &[
                Some(&[(1, POLYGON), (7, SURFACE)]),
                Some(&[(2, POLYGON), (7, SURFACE)]),
            ],
        ),
        (36, 32),
        Samples::Covered(vec![fill(1), fill(2)]),
        vec![fill(1), fill(2)],
        Some((Defect::Order(vec![fill(2), fill(1)]), mismatch())),
    ));
    out.push(open_case(
        "pair_cause_mesh",
        one15(
            &w5_with(mesh_square_at_2, |_| {}),
            view_c(),
            &[1],
            &[Some(&[(1, SURFACE)])],
        ),
        (32, 32),
        Samples::All,
        4,
        0,
    ));
    out.push(open_case(
        "pair_cause_label",
        one15(
            &w5_with(blue_square_at_2, pre_projected_in_label),
            view_c(),
            &[1, 2],
            &[Some(&[(1, POLYGON)])],
        ),
        (32, 32),
        Samples::All,
        5,
        0,
    ));
    out.push(open_case(
        "pair_cause_precedence",
        one15(
            &w5_with(mesh_square_at_2, pre_projected_in_label),
            view_c(),
            &[1, 2],
            &[Some(&[(1, SURFACE)])],
        ),
        (32, 32),
        Samples::All,
        5,
        0,
    ));
    // `label_drawn_otherwise`: `L` with a points record in its facing group, whose block (side 2,
    // at screen (42.5, 31.5)) `S` covers and `L`'s glyph does not reach. The label is read over
    // the whole frame, so `S` meets it there and the pair is open with cause `label`.
    let mut b = Build::new(true);
    b.fill(rect([2.0, 3.0], [-0.5, 0.5], 2.0), OPAQUE_BLUE);
    b.label([0.0; 3]);
    b.points(vec![[2.625, 0.125, 0.0]], WHITE, 2.0);
    out.push(open_case(
        "label_drawn_otherwise",
        one15(&b, view_c(), &[1, 2], &[Some(&[(1, POLYGON)])]),
        (42, 32),
        Samples::All,
        5,
        0,
    ));
    // The engine's two label scenes, drawn at the label head (`engine_stream`): the ring whose
    // hole holds the centre, where `S2` alone makes a pair and qualifies by its points, and W5's
    // scene through a camera class of the scene's own, every record pre-projected, the label's
    // too, so every pair of it is open with cause `label`.
    out.push(by_rule(
        "no_pair_centre_without_points_engine",
        engine_stream("no_pair_centre_without_points_engine"),
        (36, 32),
        Samples::Covered(vec![engine_fill(2), engine_fill(3)]),
        vec![engine_fill(2), engine_fill(3)],
        Some((
            Defect::Order(vec![engine_fill(3), engine_fill(2)]),
            mismatch(),
        )),
    ));
    out.push(open_case(
        "pair_cause_label_engine",
        engine_stream("pair_cause_label_engine"),
        (32, 32),
        Samples::All,
        5,
        0,
    ));
    // Folded (rule 3; no §12 witness names it): two triangles of one surface object, wound one
    // way and sharing two corners, the second folded over the first, both holding the points
    // of pixel (30, 32).
    let mut b = Build::new(true);
    b.fill(
        polygon(&[[-2.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]),
        OPAQUE_BLUE,
    );
    b.fill(
        polygon(&[[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
        OPAQUE_BLUE,
    );
    b.label([0.0, 0.0, -4.0]);
    out.push(open_case(
        "label_sheet_folded",
        one15(
            &b,
            view_c(),
            &[2],
            &[
                Some(&[(1, POLYGON), (7, SURFACE)]),
                Some(&[(2, POLYGON), (7, SURFACE)]),
            ],
        ),
        (30, 32),
        Samples::Covered(vec![fill(0), fill(1), fill(2)]),
        2,
        0,
    ));
    // The centre on a horizontal edge.
    let mut b = Build::new(true);
    b.fill(
        polygon(&[
            [-3.0, 0.0, 2.0],
            [3.0, 0.0, 2.0],
            [3.0, 3.0, -7.0],
            [-3.0, 3.0, -7.0],
        ]),
        OPAQUE_BLUE,
    );
    b.label([0.0; 3]);
    out.push(by_rule(
        "centre_on_horizontal_edge",
        one15(&b, view_c(), &[1], &[Some(&[(1, POLYGON)])]),
        (32, 28),
        Samples::Covered(vec![fill(0), fill(1)]),
        vec![fill(0), fill(1)],
        Some((Defect::Order(vec![fill(1), fill(0)]), plate())),
    ));
    // An ungrouped record kept facing the camera.
    let mut b = Build::new(true);
    b.glyph(
        rect([-3.0, 3.0], [-3.0, 3.0], 2.0),
        OPAQUE_BLUE,
        [0.0, 0.0, 2.0],
    );
    b.label([0.0; 3]);
    out.push(open_case(
        "ungrouped_sheet_open",
        one15(&b, view_c(), &[1], &[Some(&[(1, POLYGON)])]),
        (32, 32),
        Samples::All,
        5,
        0,
    ));
    out
}

/// The witnesses of what a source decides (W27, `held_row_clears_group`,
/// `label_ungrouped_no_row`, the two cylinder witnesses), each over W5's scene: `G` in slot 0,
/// `L` in slot 1.
fn source_witnesses() -> Vec<Witness> {
    let w5 = || w5_with(blue_square_at_2, |_| {});
    let label_row = row(1, 1, &[&[(2, POLYGON)]]);
    let mut out = Vec::new();
    for (name, path) in [
        ("class_cylinder_grouped", vec![(5, CYLINDER), (6, CUBE)]),
        (
            "class_part_follows_cylinder",
            vec![(4, CIRCLE), (5, CYLINDER), (6, SURFACE)],
        ),
    ] {
        out.push(open_case(
            name,
            one15(&w5(), view_c(), &[1], &[Some(&path)]),
            (32, 32),
            Samples::All,
            0,
            5,
        ));
    }
    out.push(by_rule(
        "label_ungrouped_no_row",
        one15(&w5(), view_c(), &[], &[Some(&[(1, POLYGON)])]),
        (32, 32),
        Samples::All,
        vec![fill(0), fill(1)],
        Some((Defect::Order(vec![fill(1), fill(0)]), mismatch())),
    ));
    // W18's scene in XF 1.5 with `L` given no row: the frame draws no facing group, so it is
    // judged as one read as XF 1.4, where `L` is a label by its centre and the line never pushes
    // it back (no §12 witness names it).
    let mut b = Build::new(true);
    b.label([0.0; 3]);
    b.stroke(
        open(&[[-3.1875, 0.0, 2.0], [3.1875, 0.0, 2.0]]),
        OPAQUE_GREEN,
        0.5,
    );
    out.push(by_rule(
        "label_ungrouped_no_row_line",
        one15(&b, view_c(), &[], &[None, Some(&[(1, POLYGON)])]),
        (32, 32),
        Samples::All,
        vec![fill(0), stroke(1)],
        Some((Defect::Order(vec![stroke(1), fill(0)]), plate())),
    ));
    // W27: delta 1 makes `G` a polygon inside a vector (a line); delta 2's held row reparents it
    // out of the vector, its drawing untouched.
    let b = w5();
    let two = || {
        xf15(vec![
            (
                b.delta(1, view_c()),
                vec![
                    row(0, 0, &[&[(1, POLYGON), (3, VECTOR)]]),
                    label_row.clone(),
                ],
            ),
            (bare(2), vec![row(0, 0, &[&[(1, POLYGON)]])]),
        ])
    };
    out.push(Witness {
        seq: 1,
        ..by_rule(
            "source_row_reparent_before",
            two(),
            (32, 32),
            Samples::All,
            vec![fill(1), fill(0)],
            Some((Defect::Order(vec![fill(0), fill(1)]), plate())),
        )
    });
    out.push(by_rule(
        "source_row_reparent",
        two(),
        (32, 32),
        Samples::All,
        vec![fill(0), fill(1)],
        Some((Defect::Order(vec![fill(1), fill(0)]), mismatch())),
    ));
    // `held_row_clears_group`: delta 2 gives `L` facing group 0; frame 2 draws no facing group.
    let held = xf15(vec![
        (
            b.delta(1, view_c()),
            vec![row(0, 0, &[&[(1, POLYGON)]]), row(1, 7, &[&[(2, POLYGON)]])],
        ),
        (bare(2), vec![row(1, 0, &[&[(2, POLYGON)]])]),
    ]);
    out.push(by_rule(
        "held_row_clears_group",
        held,
        (32, 32),
        Samples::All,
        vec![fill(0), fill(1)],
        Some((Defect::Order(vec![fill(1), fill(0)]), mismatch())),
    ));
    out
}

/// A delta at `seq` that sets, removes and orders nothing.
fn bare(seq: u64) -> Delta {
    let mut d = Delta::new(
        arachne_core::xf::Time {
            num: seq as i64,
            den: 1,
        },
        seq,
    );
    d.version = (1, 4);
    d
}

/// W16's scene: `P`, a polygon on no plane of its own, filled green with a red background
/// stroke; then `B`, on the axis plane `z`, half-transparent blue.
fn cycle_scene() -> (Vec<u8>, Vec<Delta>) {
    let mut b = Build::new(true);
    let p = b.fill(
        polygon(&[
            [-3.1875, 0.0, 1.5],
            [3.1875, 0.0, 1.5],
            [3.1875, 3.0 / 64.0, -1.5],
            [-3.1875, 3.0 / 64.0, -1.5],
        ]),
        OPAQUE_GREEN,
    );
    p.paint.background_stroke = Paint::One(OPAQUE_RED);
    p.paint.background_stroke_width = 0.25;
    p.planes = Some(vec![0]);
    let q = b.fill(rect([-6.0, 6.0], [-6.0, 6.0], 0.0), [0.0, 0.0, 1.0, 0.5]);
    q.planes = Some(vec![6]);
    one(&b, view_c())
}

fn layer_code(l: &CoreLayer) -> u8 {
    match l {
        CoreLayer::BackgroundStroke => 0,
        CoreLayer::Fill => 1,
        CoreLayer::Stroke => 2,
        CoreLayer::Surface => 3,
        CoreLayer::Mask => 4,
    }
}

/// The core's report of one pixel: its fragments and, at each sample, every one covering it,
/// front first, before any cull (`samples_all`).
fn core_report(deltas: &[Delta], seq: u64, (x, y): (u32, u32)) -> Reported {
    let mut core = Stream::new();
    for d in deltas {
        core.apply(d).unwrap();
        if d.seq == seq {
            break;
        }
    }
    let px = core.resolve_trace(x, y, x + 1, y + 1, true).unwrap();
    let Some(p) = px.get(&(x, y)) else {
        return Reported {
            x,
            y,
            entries: Vec::new(),
            order: vec![Vec::new(); 16],
        };
    };
    let reads = p.reads.as_ref().expect("the trace's reads");
    Reported {
        x,
        y,
        entries: p
            .entries
            .iter()
            .map(|e| Entry {
                identity: (e.leaf.0, e.leaf.1, layer_code(&e.layer), e.sub),
                cells: e.coverage,
            })
            .collect(),
        order: reads.samples_all.clone(),
    }
}

fn entry_of(rep: &mut Reported, f: Frag) -> usize {
    let id = f.identity();
    if let Some(i) = rep.entries.iter().position(|e| e.identity == id) {
        return i;
    }
    rep.entries.push(Entry {
        identity: id,
        cells: [0; 16],
    });
    rep.entries.len() - 1
}

fn set_order(rep: &mut Reported, s: usize, order: &[Frag]) {
    let idx: Vec<usize> = order.iter().map(|f| entry_of(rep, *f)).collect();
    rep.order[s] = idx;
}

fn apply(rep: &Reported, defect: &Defect, samples: &[usize]) -> Reported {
    let mut r = rep.clone();
    for &s in samples {
        match defect {
            Defect::Order(o) => set_order(&mut r, s, o),
            Defect::Absent(f) => {
                let e = entry_of(&mut r, *f);
                r.entries[e].cells[s] = 0;
                r.order[s].retain(|&i| i != e);
            }
            Defect::Present(f, b) => {
                let e = entry_of(&mut r, *f);
                r.entries[e].cells[s] = *b;
                if !r.order[s].contains(&e) {
                    r.order[s].insert(0, e);
                }
            }
            Defect::Cells(f, b) => {
                let e = entry_of(&mut r, *f);
                r.entries[e].cells[s] = *b;
            }
        }
    }
    r
}

/// A stream the engine wrote at the label head, `30538b2e`, from the scenes of
/// `fixtures/engine_scenes/label_witness_scenes.py` (each rendered on the core at 64 x 64 pixels
/// over a 16 x 16 unit frame, `PYTHONHASHSEED=0`): its bytes, and its deltas as the core draws
/// them, without the source section and read as XF 1.4, whose drawing it does not change. The
/// bytes are kept as hex text (`stream.xf.hex`).
fn engine_stream(name: &str) -> (Vec<u8>, Vec<Delta>) {
    let hex: Vec<u8> = std::fs::read_to_string(dir(name).join("stream.xf.hex"))
        .unwrap()
        .bytes()
        .filter(u8::is_ascii_hexdigit)
        .collect();
    let digit = |c: u8| (c as char).to_digit(16).unwrap() as u8;
    let bytes: Vec<u8> = hex
        .chunks(2)
        .map(|p| digit(p[0]) << 4 | digit(p[1]))
        .collect();
    let (_, wire) = source::read_file_bytes(&bytes).unwrap();
    let deltas = wire
        .into_iter()
        .map(|w| {
            let mut d = w.read.unwrap().0;
            d.version = (1, 4);
            d
        })
        .collect();
    (bytes, deltas)
}

fn oracle(w: &Witness) -> Decisions {
    let (_, wire) = source::read_file_bytes(&w.xf).unwrap();
    decide_at_wire(&wire, w.seq, None).unwrap()
}

fn samples_of(w: &Witness, d: &Decisions) -> Vec<usize> {
    match &w.samples {
        Samples::All => (0..16).collect(),
        Samples::These(s) => s.clone(),
        Samples::CellRows(rows) => (0..16)
            .filter(|s| rows.contains(&sample_cell(*s).1))
            .collect(),
        Samples::CellColumns(cols) => (0..16)
            .filter(|s| cols.contains(&sample_cell(*s).0))
            .collect(),
        Samples::Covered(fs) => (0..16)
            .filter(|&s| {
                fs.iter().all(|f| {
                    let id = d
                        .fragments
                        .iter()
                        .find(|g| (g.object, g.part, g.layer, g.sub) == f.identity());
                    id.is_some_and(|g| {
                        d.covers.iter().any(|c| {
                            (c.x, c.y, c.fragment) == (w.pixel.0, w.pixel.1, g.id)
                                && c.cover[s] != 0
                        })
                    })
                })
            })
            .collect(),
    }
}

fn dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn expected_json(w: &Witness, samples: &[usize]) -> String {
    let defect = w
        .defect
        .as_ref()
        .map_or("null".to_string(), |(_, v)| format!("\"{}\"", v.word()));
    format!(
        "{{\n  \"fixture\": \"{}\",\n  \"kind\": \"witness\",\n  \"rule\": \"{}\",\n  \"seq\": {},\n  \"pixel\": [{}, {}],\n  \"samples\": {samples:?},\n  \"defect_verdict\": {defect},\n  \"twin_verdict\": \"{}\"\n}}\n",
        w.name,
        w.rule,
        w.seq,
        w.pixel.0,
        w.pixel.1,
        w.twin.word()
    )
}

/// Each witness's verdicts at its samples, and its files.
fn run(witnesses: Vec<Witness>) {
    let bless = std::env::var_os("ORACLE_BLESS_FIXTURES").is_some();
    let mut wrong = Vec::new();
    for w in &witnesses {
        let d = oracle(w);
        assert_eq!(d.header.status, 0, "{}: {:?}", w.name, d.refused);
        let samples = samples_of(w, &d);
        assert!(!samples.is_empty(), "{}: no sample named", w.name);
        let mut base = core_report(&w.deltas, w.seq, w.pixel);
        if let Conforming::Rule(order) = &w.conforming {
            for &s in &samples {
                set_order(&mut base, s, order);
            }
        }
        for &s in &samples {
            let v = judge(&d, &base, s);
            if v != w.twin {
                wrong.push(format!(
                    "{} twin s{s}: {} not {}",
                    w.name,
                    v.word(),
                    w.twin.word()
                ));
            }
        }
        if let Some((defect, verdict)) = &w.defect {
            let planted = apply(&base, defect, &samples);
            for &s in &samples {
                let v = judge(&d, &planted, s);
                if v != *verdict {
                    wrong.push(format!(
                        "{} defect s{s}: {} not {}",
                        w.name,
                        v.word(),
                        verdict.word()
                    ));
                }
            }
        }
        // The files.
        let digest = format!("{}\n", sha256_hex(&w.xf));
        let expected = expected_json(w, &samples);
        let defect = w.defect.as_ref().map(|(d, _)| d.json(w.pixel, &samples));
        let path = dir(w.name);
        if bless {
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(path.join("scene.sha256"), &digest).unwrap();
            std::fs::write(path.join("expected.json"), &expected).unwrap();
            if let Some(j) = &defect {
                std::fs::write(path.join("defect.json"), j).unwrap();
            }
        }
        let read = |f: &str| std::fs::read_to_string(path.join(f)).ok();
        if read("scene.sha256").as_deref() != Some(digest.as_str()) {
            wrong.push(format!("{}: scene.sha256 does not pin the scene", w.name));
        }
        if read("expected.json").as_deref() != Some(expected.as_str()) {
            wrong.push(format!("{}: expected.json differs", w.name));
        }
        if read("defect.json") != defect {
            wrong.push(format!("{}: defect.json differs", w.name));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn mutation_depth_witnesses_fail_and_their_twins_agree() {
    run(depth_witnesses());
}

#[test]
fn mutation_label_witnesses_on_the_core_s_own_report() {
    run(label_core_witnesses());
}

#[test]
fn mutation_label_rule_witnesses_fail_and_their_twins_agree() {
    run(label_rule_witnesses());
}

#[test]
fn mutation_source_witnesses_read_the_source_the_frame_holds() {
    run(source_witnesses());
}
