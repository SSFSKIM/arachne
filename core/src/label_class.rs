//! The class a camera-facing label's rule reads a record as, from what the record was drawn from
//! (its XF 1.5 `source` and `facing_group`) and nothing else: a label, a line, a surface, or
//! `label_rule_open` with its cause.
//!
//! A record is read by these rules, in order, each path on its own:
//!
//! 1. A record with a facing group is a label, whatever it was drawn from.
//! 2. A path with a line class ([`LINE_CLASSES`]: lines, curves, arrows, tips and axes) anywhere
//!    on it, the record's owner or any container, is a line, whatever lies between.
//! 3. A generic class ([`GENERIC_CLASSES`]) reads as its nearest container whose class is not
//!    generic, and is open ([`OpenCause::GenericOnly`]) with none.
//! 4. A surface class ([`SURFACE_CLASSES`]: surfaces, solids and filled planar shapes) whose
//!    nearest non-generic container is surface-like (a solid's face, a cylinder's or a cone's
//!    cap) reads as that container does.
//! 5. A [`CYLINDER`] that nothing contains is a solid, a surface; inside any container it is open
//!    ([`OpenCause::CylinderGrouped`]): it may be a shaft its user built, and its thinness is
//!    never inferred.
//! 6. Otherwise the class itself decides: a surface class is a surface, a point cloud
//!    ([`POINT_CLOUD_CLASSES`]) is open ([`OpenCause::PointCloud`]), and any class the inventory
//!    does not name (text without a facing group, images, braces, tables) is open
//!    ([`OpenCause::Unmapped`]).
//! 7. A record whose paths read differently is open ([`OpenCause::PathsDisagree`]), and so is a
//!    record with no source ([`OpenCause::NoSource`]).
//!
//! A surface class qualifies by its fill. The class does not say whether a record is filled, so
//! a record drawn only by its outline stroke still reads as a surface here, and its planner, which
//! reads the record's paint, does not count that stroke as a qualifying fill.
//!
//! Nothing drawn calls it.

use crate::xf::SourceNode;

/// What a record is to a camera-facing label's rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelClass {
    /// A camera-facing label: a record with a facing group.
    Label,
    /// A line, a curve or axes, with all they hold.
    Line,
    /// A surface, a solid or a filled planar shape.
    Surface,
    /// `label_rule_open`: the rule decides no order, for this cause.
    LabelRuleOpen(OpenCause),
}

/// Why a record is `label_rule_open`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenCause {
    /// Its class is not in the inventory.
    Unmapped,
    /// Generic classes alone, all the way up its path.
    GenericOnly,
    /// Its paths read differently.
    PathsDisagree,
    /// It carries no source.
    NoSource,
    /// A cylinder inside a container that is not a line, or a part of one.
    CylinderGrouped,
    /// A point cloud.
    PointCloud,
}

impl OpenCause {
    /// The cause's name, as counts report it.
    pub fn name(self) -> &'static str {
        match self {
            OpenCause::Unmapped => "unmapped",
            OpenCause::GenericOnly => "generic_only",
            OpenCause::PathsDisagree => "paths_disagree",
            OpenCause::NoSource => "no_source",
            OpenCause::CylinderGrouped => "cylinder_grouped",
            OpenCause::PointCloud => "point_cloud",
        }
    }
}

/// The line classes: lines, curves, arrows, tips and axes.
pub const LINE_CLASSES: &[&str] = &[
    "arachne.animation.changing.AnimatedBoundary",
    "arachne.animation.changing.TracedPath",
    "arachne.mobject.geometry.arc.Arc",
    "arachne.mobject.geometry.arc.ArcBetweenPoints",
    "arachne.mobject.geometry.arc.CubicBezier",
    "arachne.mobject.geometry.arc.CurvedArrow",
    "arachne.mobject.geometry.arc.CurvedDoubleArrow",
    "arachne.mobject.geometry.arc.TangentialArc",
    "arachne.mobject.geometry.arc.TipableVMobject",
    "arachne.mobject.geometry.labeled.LabeledArrow",
    "arachne.mobject.geometry.labeled.LabeledLine",
    "arachne.mobject.geometry.line.Angle",
    "arachne.mobject.geometry.line.Arrow",
    "arachne.mobject.geometry.line.DashedLine",
    "arachne.mobject.geometry.line.DoubleArrow",
    "arachne.mobject.geometry.line.Elbow",
    "arachne.mobject.geometry.line.Line",
    "arachne.mobject.geometry.line.RightAngle",
    "arachne.mobject.geometry.line.TangentLine",
    "arachne.mobject.geometry.line.Vector",
    "arachne.mobject.geometry.shape_matchers.Cross",
    "arachne.mobject.geometry.shape_matchers.Underline",
    "arachne.mobject.geometry.tips.ArrowCircleFilledTip",
    "arachne.mobject.geometry.tips.ArrowCircleTip",
    "arachne.mobject.geometry.tips.ArrowSquareFilledTip",
    "arachne.mobject.geometry.tips.ArrowSquareTip",
    "arachne.mobject.geometry.tips.ArrowTip",
    "arachne.mobject.geometry.tips.ArrowTriangleFilledTip",
    "arachne.mobject.geometry.tips.ArrowTriangleTip",
    "arachne.mobject.geometry.tips.StealthTip",
    "arachne.mobject.graphing.coordinate_systems.Axes",
    "arachne.mobject.graphing.coordinate_systems.ComplexPlane",
    "arachne.mobject.graphing.coordinate_systems.NumberPlane",
    "arachne.mobject.graphing.coordinate_systems.PolarPlane",
    "arachne.mobject.graphing.coordinate_systems.ThreeDAxes",
    "arachne.mobject.graphing.functions.FunctionGraph",
    "arachne.mobject.graphing.functions.ImplicitFunction",
    "arachne.mobject.graphing.functions.ParametricFunction",
    "arachne.mobject.graphing.number_line.NumberLine",
    "arachne.mobject.graphing.number_line.UnitInterval",
    "arachne.mobject.graphing.probability.BarChart",
    "arachne.mobject.three_d.three_dimensions.Arrow3D",
    "arachne.mobject.three_d.three_dimensions.Line3D",
    "arachne.mobject.types.vectorized_mobject.CurvesAsSubmobjects",
    "arachne.mobject.types.vectorized_mobject.DashedVMobject",
    "arachne.mobject.vector_field.ArrowVectorField",
    "arachne.mobject.vector_field.StreamLine",
    "arachne.mobject.vector_field.StreamLines",
    "arachne.mobject.vector_field.VectorField",
];

/// The surface classes: surfaces, solids and filled planar shapes.
pub const SURFACE_CLASSES: &[&str] = &[
    "arachne.mobject.frame.FullScreenRectangle",
    "arachne.mobject.frame.ScreenRectangle",
    "arachne.mobject.geometry.arc.AnnotationDot",
    "arachne.mobject.geometry.arc.AnnularSector",
    "arachne.mobject.geometry.arc.Annulus",
    "arachne.mobject.geometry.arc.ArcPolygon",
    "arachne.mobject.geometry.arc.ArcPolygonFromArcs",
    "arachne.mobject.geometry.arc.Circle",
    "arachne.mobject.geometry.arc.Dot",
    "arachne.mobject.geometry.arc.Ellipse",
    "arachne.mobject.geometry.arc.LabeledDot",
    "arachne.mobject.geometry.arc.Sector",
    "arachne.mobject.geometry.boolean_ops.Difference",
    "arachne.mobject.geometry.boolean_ops.Exclusion",
    "arachne.mobject.geometry.boolean_ops.Intersection",
    "arachne.mobject.geometry.boolean_ops.Union",
    "arachne.mobject.geometry.boolean_ops._BooleanOps",
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
    "arachne.mobject.geometry.shape_matchers.BackgroundRectangle",
    "arachne.mobject.geometry.shape_matchers.SurroundingRectangle",
    "arachne.mobject.graphing.probability.SampleSpace",
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
];

/// The cylinder: a solid alone, a shaft owned by a line, open anywhere else.
pub const CYLINDER: &str = "arachne.mobject.three_d.three_dimensions.Cylinder";

/// The generic classes, and no others: groups and the faces and paths other objects are made of,
/// which carry no class of their own and read as their owner.
pub const GENERIC_CLASSES: &[&str] = &[
    "arachne.mobject.mobject.Group",
    "arachne.mobject.mobject.Mobject",
    "arachne.mobject.three_d.three_dimensions.ThreeDVMobject",
    "arachne.mobject.types.vectorized_mobject.VDict",
    "arachne.mobject.types.vectorized_mobject.VGroup",
    "arachne.mobject.types.vectorized_mobject.VMobject",
];

/// The point clouds.
pub const POINT_CLOUD_CLASSES: &[&str] = &[
    "arachne.mobject.types.point_cloud_mobject.Mobject1D",
    "arachne.mobject.types.point_cloud_mobject.Mobject2D",
    "arachne.mobject.types.point_cloud_mobject.PGroup",
    "arachne.mobject.types.point_cloud_mobject.PMobject",
    "arachne.mobject.types.point_cloud_mobject.Point",
    "arachne.mobject.types.point_cloud_mobject.PointCloudDot",
];

/// What a node's class says of it, before its owner is read.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Line,
    Surface,
    Cylinder,
    Generic,
    PointCloud,
    Other,
}

fn kind(class: &str) -> Kind {
    if LINE_CLASSES.contains(&class) {
        Kind::Line
    } else if SURFACE_CLASSES.contains(&class) {
        Kind::Surface
    } else if class == CYLINDER {
        Kind::Cylinder
    } else if GENERIC_CLASSES.contains(&class) {
        Kind::Generic
    } else if POINT_CLOUD_CLASSES.contains(&class) {
        Kind::PointCloud
    } else {
        Kind::Other
    }
}

/// The class of a record drawn from `source` (its paths, each owner first, outermost container
/// last) with `facing_group` (`0`: none).
pub fn label_class(source: &[Vec<SourceNode>], facing_group: u32) -> LabelClass {
    if facing_group != 0 {
        return LabelClass::Label;
    }
    let mut classes = source.iter().map(|path| path_class(path.as_slice()));
    match classes.next() {
        None => LabelClass::LabelRuleOpen(OpenCause::NoSource),
        Some(first) if classes.all(|class| class == first) => first,
        Some(_) => LabelClass::LabelRuleOpen(OpenCause::PathsDisagree),
    }
}

/// The class of a path's first node.
fn path_class(path: &[SourceNode]) -> LabelClass {
    if path.iter().any(|node| kind(&node.class) == Kind::Line) {
        return LabelClass::Line;
    }
    let open = LabelClass::LabelRuleOpen;
    // Each node read after its containers. The nearest container of the node at hand whose class
    // is not generic: its kind and class.
    let mut owner: Option<(Kind, LabelClass)> = None;
    let mut class = open(OpenCause::GenericOnly);
    for (at, node) in path.iter().enumerate().rev() {
        let kind = kind(&node.class);
        class = match (kind, owner) {
            (Kind::Generic, Some((_, owned))) => owned,
            (Kind::Generic, None) => open(OpenCause::GenericOnly),
            (Kind::Surface, Some((Kind::Surface | Kind::Cylinder, owned))) => owned,
            (Kind::Cylinder, None) if at + 1 == path.len() => LabelClass::Surface,
            (Kind::Cylinder, _) => open(OpenCause::CylinderGrouped),
            (Kind::Surface, _) => LabelClass::Surface,
            (Kind::PointCloud, _) => open(OpenCause::PointCloud),
            (Kind::Other, _) => open(OpenCause::Unmapped),
            (Kind::Line, _) => unreachable!("a path with a line class is a line"),
        };
        if kind != Kind::Generic {
            owner = Some((kind, class));
        }
    }
    class
}
