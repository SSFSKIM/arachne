//! `label_classes@2` (contract §10, "Source classes"): what a record is, read from its XF 1.5
//! source: a line, a surface, or `label_rule_open` with a cause. The table is the oracle's own,
//! written from the contract's text alone.

/// A record's class outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    Line,
    Surface,
    Open(Cause),
}

/// Why a class is left open (`label_rule_open`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cause {
    Unmapped,
    GenericOnly,
    PathsDisagree,
    NoSource,
    CylinderGrouped,
    PointCloud,
}

impl Cause {
    pub fn name(self) -> &'static str {
        match self {
            Cause::Unmapped => "unmapped",
            Cause::GenericOnly => "generic_only",
            Cause::PathsDisagree => "paths_disagree",
            Cause::NoSource => "no_source",
            Cause::CylinderGrouped => "cylinder_grouped",
            Cause::PointCloud => "point_cloud",
        }
    }
}

/// What the table says of one class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Line,
    Surface,
    Cylinder,
    Generic,
    PointCloud,
    /// Listed open with `unmapped`, or not listed at all.
    Unmapped,
}

const CYLINDER: &str = "arachne.mobject.three_d.three_dimensions.Cylinder";

const LINES: &[&str] = &[
    "arachne.mobject.three_d.three_dimensions.Arrow3D",
    "arachne.mobject.three_d.three_dimensions.Line3D",
    "arachne.mobject.geometry.line.Angle",
    "arachne.mobject.geometry.line.Elbow",
    "arachne.mobject.geometry.line.RightAngle",
    "arachne.mobject.geometry.tips.ArrowCircleFilledTip",
    "arachne.mobject.geometry.tips.ArrowCircleTip",
    "arachne.mobject.geometry.tips.ArrowSquareFilledTip",
    "arachne.mobject.geometry.tips.ArrowSquareTip",
    "arachne.mobject.geometry.tips.ArrowTip",
    "arachne.mobject.geometry.tips.ArrowTriangleFilledTip",
    "arachne.mobject.geometry.tips.ArrowTriangleTip",
    "arachne.mobject.geometry.tips.StealthTip",
    "arachne.animation.changing.AnimatedBoundary",
    "arachne.animation.changing.TracedPath",
    "arachne.mobject.geometry.arc.CubicBezier",
    "arachne.mobject.graphing.functions.FunctionGraph",
    "arachne.mobject.graphing.functions.ImplicitFunction",
    "arachne.mobject.graphing.functions.ParametricFunction",
    "arachne.mobject.types.vectorized_mobject.CurvesAsSubmobjects",
    "arachne.mobject.types.vectorized_mobject.DashedVMobject",
    "arachne.mobject.vector_field.StreamLine",
    "arachne.mobject.vector_field.ArrowVectorField",
    "arachne.mobject.vector_field.StreamLines",
    "arachne.mobject.vector_field.VectorField",
    "arachne.mobject.geometry.labeled.LabeledArrow",
    "arachne.mobject.geometry.labeled.LabeledLine",
    "arachne.mobject.geometry.line.Arrow",
    "arachne.mobject.geometry.line.DashedLine",
    "arachne.mobject.geometry.line.DoubleArrow",
    "arachne.mobject.geometry.line.Line",
    "arachne.mobject.geometry.line.TangentLine",
    "arachne.mobject.geometry.line.Vector",
    "arachne.mobject.geometry.shape_matchers.Cross",
    "arachne.mobject.geometry.shape_matchers.Underline",
    "arachne.mobject.geometry.arc.Arc",
    "arachne.mobject.geometry.arc.ArcBetweenPoints",
    "arachne.mobject.geometry.arc.CurvedArrow",
    "arachne.mobject.geometry.arc.CurvedDoubleArrow",
    "arachne.mobject.geometry.arc.TangentialArc",
    "arachne.mobject.geometry.arc.TipableVMobject",
    "arachne.mobject.graphing.coordinate_systems.Axes",
    "arachne.mobject.graphing.coordinate_systems.ComplexPlane",
    "arachne.mobject.graphing.coordinate_systems.NumberPlane",
    "arachne.mobject.graphing.coordinate_systems.PolarPlane",
    "arachne.mobject.graphing.coordinate_systems.ThreeDAxes",
    "arachne.mobject.graphing.number_line.NumberLine",
    "arachne.mobject.graphing.number_line.UnitInterval",
    "arachne.mobject.graphing.probability.BarChart",
];

const SURFACES: &[&str] = &[
    "arachne.mobject.geometry.arc.ArcPolygon",
    "arachne.mobject.geometry.arc.ArcPolygonFromArcs",
    "arachne.mobject.geometry.shape_matchers.BackgroundRectangle",
    "arachne.mobject.geometry.shape_matchers.SurroundingRectangle",
    "arachne.mobject.graphing.probability.SampleSpace",
    "arachne.mobject.frame.FullScreenRectangle",
    "arachne.mobject.frame.ScreenRectangle",
    "arachne.mobject.three_d.polyhedra.ConvexHull3D",
    "arachne.mobject.three_d.polyhedra.Dodecahedron",
    "arachne.mobject.three_d.polyhedra.Icosahedron",
    "arachne.mobject.three_d.polyhedra.Octahedron",
    "arachne.mobject.three_d.polyhedra.Polyhedron",
    "arachne.mobject.three_d.polyhedra.Tetrahedron",
    "arachne.mobject.three_d.three_dimensions.Cone",
    "arachne.mobject.three_d.three_dimensions.Cube",
    "arachne.mobject.three_d.three_dimensions.Dot3D",
    "arachne.mobject.three_d.three_dimensions.Prism",
    "arachne.mobject.three_d.three_dimensions.Sphere",
    "arachne.mobject.three_d.three_dimensions.Surface",
    "arachne.mobject.three_d.three_dimensions.Torus",
    "arachne.mobject.geometry.boolean_ops.Difference",
    "arachne.mobject.geometry.boolean_ops.Exclusion",
    "arachne.mobject.geometry.boolean_ops.Intersection",
    "arachne.mobject.geometry.boolean_ops.Union",
    "arachne.mobject.geometry.boolean_ops._BooleanOps",
    "arachne.mobject.geometry.arc.AnnotationDot",
    "arachne.mobject.geometry.arc.Annulus",
    "arachne.mobject.geometry.arc.Circle",
    "arachne.mobject.geometry.arc.Dot",
    "arachne.mobject.geometry.arc.Ellipse",
    "arachne.mobject.geometry.arc.LabeledDot",
    "arachne.mobject.geometry.labeled.LabeledPolygram",
    "arachne.mobject.geometry.polygram.ConvexHull",
    "arachne.mobject.geometry.polygram.Cutout",
    "arachne.mobject.geometry.polygram.Polygon",
    "arachne.mobject.geometry.polygram.Polygram",
    "arachne.mobject.geometry.polygram.Rectangle",
    "arachne.mobject.geometry.polygram.RegularPolygon",
    "arachne.mobject.geometry.polygram.RegularPolygram",
    "arachne.mobject.geometry.polygram.RoundedRectangle",
    "arachne.mobject.geometry.polygram.Square",
    "arachne.mobject.geometry.polygram.Star",
    "arachne.mobject.geometry.polygram.Triangle",
    "arachne.mobject.geometry.arc.AnnularSector",
    "arachne.mobject.geometry.arc.Sector",
];

/// The six generic classes; the list is closed.
const GENERIC: &[&str] = &[
    "arachne.mobject.mobject.Mobject",
    "arachne.mobject.mobject.Group",
    "arachne.mobject.types.vectorized_mobject.VMobject",
    "arachne.mobject.types.vectorized_mobject.VGroup",
    "arachne.mobject.types.vectorized_mobject.VDict",
    "arachne.mobject.three_d.three_dimensions.ThreeDVMobject",
];

const POINT_CLOUDS: &[&str] = &[
    "arachne.frame_ir.cairo_painter._PointsProxy",
    "arachne.mobject.types.point_cloud_mobject.Mobject1D",
    "arachne.mobject.types.point_cloud_mobject.Mobject2D",
    "arachne.mobject.types.point_cloud_mobject.PGroup",
    "arachne.mobject.types.point_cloud_mobject.PMobject",
    "arachne.mobject.types.point_cloud_mobject.Point",
    "arachne.mobject.types.point_cloud_mobject.PointCloudDot",
];

fn kind(class: &str) -> Kind {
    if class == CYLINDER {
        Kind::Cylinder
    } else if LINES.contains(&class) {
        Kind::Line
    } else if SURFACES.contains(&class) {
        Kind::Surface
    } else if GENERIC.contains(&class) {
        Kind::Generic
    } else if POINT_CLOUDS.contains(&class) {
        Kind::PointCloud
    } else {
        // Listed open with `unmapped`, or a class the table does not list: either falls closed.
        Kind::Unmapped
    }
}

/// The cylinder rule (rule 4) for a cylinder at `at` of a path: a solid where it has no
/// container, else `cylinder_grouped`.
fn cylinder(path: &[&str], at: usize) -> Outcome {
    if at + 1 == path.len() {
        Outcome::Surface
    } else {
        Outcome::Open(Cause::CylinderGrouped)
    }
}

/// One path's outcome, its classes maker first, outermost last (rules 1 to 5).
pub fn path_outcome(path: &[&str]) -> Outcome {
    // 1. Line ancestry.
    if path.iter().any(|c| kind(c) == Kind::Line) {
        return Outcome::Line;
    }
    // 2. Generic nodes read through.
    let Some(read) = path.iter().position(|c| kind(c) != Kind::Generic) else {
        return Outcome::Open(Cause::GenericOnly);
    };
    // 4. The cylinder rule decides a cylinder itself.
    if kind(path[read]) == Kind::Cylinder {
        return cylinder(path, read);
    }
    // 3. Parts follow a surface-like container: the walk passes over generic and open nodes and
    //    stops at the first surface-like one, a cylinder included.
    for (at, c) in path.iter().enumerate().skip(read + 1) {
        match kind(c) {
            Kind::Surface => return Outcome::Surface,
            Kind::Cylinder => return cylinder(path, at),
            _ => {}
        }
    }
    // 5. The node's own class.
    match kind(path[read]) {
        Kind::Surface => Outcome::Surface,
        Kind::PointCloud => Outcome::Open(Cause::PointCloud),
        _ => Outcome::Open(Cause::Unmapped),
    }
}

/// A record's outcome from its source's paths: theirs where they agree, `paths_disagree` where
/// they differ, `no_source` where it has none.
pub fn record_outcome<'a>(paths: impl IntoIterator<Item = Vec<&'a str>>) -> Outcome {
    let mut out: Option<Outcome> = None;
    for p in paths {
        let o = path_outcome(&p);
        match out {
            None => out = Some(o),
            Some(prev) if prev != o => return Outcome::Open(Cause::PathsDisagree),
            _ => {}
        }
    }
    out.unwrap_or(Outcome::Open(Cause::NoSource))
}
