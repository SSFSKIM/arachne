//! The class a camera-facing label's rule reads a record as, from what the record was drawn from
//! (its XF 1.5 `source` and `facing_group`) and nothing else: a label, a line, a surface, or open.
//!
//! A record with a facing group is a label, whatever it was drawn from. Any other record is read
//! path by path, and a record whose paths disagree is open. A path is read from its outermost
//! container inwards, each node by its class and its owner's reading, its owner being the nearest
//! container whose class is not generic:
//!
//! - a line class ([`LINE_CLASSES`]: lines, curves, arrows, tips and axes) is a line;
//! - a surface class ([`SURFACE_CLASSES`]: surfaces, solids and filled planar shapes) is a line
//!   when its owner reads as one, as its owner reads when its owner is surface-like (a solid's
//!   face, a cylinder's or a cone's cap), and a surface otherwise;
//! - a [`CYLINDER`] is a line when its owner reads as one (a shaft), a surface when nothing
//!   contains it (a solid), and open otherwise: a cylinder in a plain group or in any other
//!   object may be a shaft its user built, and its thinness is never inferred;
//! - a generic class ([`GENERIC_CLASSES`]) reads as its owner, and is open with none;
//! - any other class (text without a facing group, images, point clouds, braces, tables, a class
//!   these tables do not name) is open.
//!
//! An owner of another class stops a line's ancestry: a plate inside a labelled line's label is
//! owned by the label, so it is a surface.
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
    /// A line, a curve or axes, with its shafts, cones, caps and tips.
    Line,
    /// A surface, a solid or a filled planar shape.
    Surface,
    /// `label_rule_open`: generic or conflicting provenance, a cylinder neither alone nor owned by
    /// a line, or a class the rule does not name.
    Open,
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
    "arachne.mobject.three_d.three_dimensions.Arrow3D",
    "arachne.mobject.three_d.three_dimensions.Line3D",
    "arachne.mobject.types.vectorized_mobject.CurvesAsSubmobjects",
    "arachne.mobject.types.vectorized_mobject.DashedVMobject",
    "arachne.mobject.vector_field.StreamLine",
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

/// The generic classes: groups, points and the faces and path pieces other objects are made of,
/// which carry no class of their own and read as their owner.
pub const GENERIC_CLASSES: &[&str] = &[
    "arachne.mobject.mobject.Group",
    "arachne.mobject.mobject.Mobject",
    "arachne.mobject.svg.svg_mobject.VMobjectFromSVGPath",
    "arachne.mobject.three_d.three_dimensions.ThreeDVMobject",
    "arachne.mobject.types.vectorized_mobject.VDict",
    "arachne.mobject.types.vectorized_mobject.VGroup",
    "arachne.mobject.types.vectorized_mobject.VMobject",
    "arachne.mobject.types.vectorized_mobject.VectorizedPoint",
];

/// What a node's class says of it, before its owner is read.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Line,
    Surface,
    Cylinder,
    Generic,
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
        Some(first) if classes.all(|class| class == first) => first,
        _ => LabelClass::Open,
    }
}

/// The class of a path's first node, each node read after its containers.
fn path_class(path: &[SourceNode]) -> LabelClass {
    // The nearest container of the node at hand whose class is not generic: its kind and class.
    let mut owner: Option<(Kind, LabelClass)> = None;
    let mut class = LabelClass::Open;
    for (at, node) in path.iter().enumerate().rev() {
        let kind = kind(&node.class);
        class = match (kind, owner) {
            (Kind::Line, _) => LabelClass::Line,
            (Kind::Surface | Kind::Cylinder, Some((_, LabelClass::Line))) => LabelClass::Line,
            (Kind::Surface, Some((Kind::Surface | Kind::Cylinder, owned))) => owned,
            (Kind::Surface, _) => LabelClass::Surface,
            (Kind::Cylinder, None) if at + 1 == path.len() => LabelClass::Surface,
            (Kind::Cylinder, _) => LabelClass::Open,
            (Kind::Generic, Some((_, owned))) => owned,
            (Kind::Generic, None) | (Kind::Other, _) => LabelClass::Open,
        };
        if kind != Kind::Generic {
            owner = Some((kind, class));
        }
    }
    class
}
