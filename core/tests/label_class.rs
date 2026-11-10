//! What a camera-facing label's rule reads a record as, from its source and facing group: one
//! case per sentence of the class reading, and the published inventory.

use arachne_core::label_class::{
    CYLINDER, GENERIC_CLASSES, LINE_CLASSES, LabelClass, OpenCause, POINT_CLOUD_CLASSES,
    SURFACE_CLASSES, label_class,
};
use arachne_core::xf::SourceNode;

const LINE3D: &str = "arachne.mobject.three_d.three_dimensions.Line3D";
const ARROW3D: &str = "arachne.mobject.three_d.three_dimensions.Arrow3D";
const CONE: &str = "arachne.mobject.three_d.three_dimensions.Cone";
const SPHERE: &str = "arachne.mobject.three_d.three_dimensions.Sphere";
const CUBE: &str = "arachne.mobject.three_d.three_dimensions.Cube";
const FACE: &str = "arachne.mobject.three_d.three_dimensions.ThreeDVMobject";
const ARROW: &str = "arachne.mobject.geometry.line.Arrow";
const VECTOR: &str = "arachne.mobject.geometry.line.Vector";
const DOUBLE_ARROW: &str = "arachne.mobject.geometry.line.DoubleArrow";
const LINE: &str = "arachne.mobject.geometry.line.Line";
const FILLED_TIP: &str = "arachne.mobject.geometry.tips.ArrowTriangleFilledTip";
const AXES: &str = "arachne.mobject.graphing.coordinate_systems.ThreeDAxes";
const POLAR_PLANE: &str = "arachne.mobject.graphing.coordinate_systems.PolarPlane";
const NUMBER_LINE: &str = "arachne.mobject.graphing.number_line.NumberLine";
const CURVE: &str = "arachne.mobject.graphing.functions.ParametricFunction";
const CIRCLE: &str = "arachne.mobject.geometry.arc.Circle";
const SQUARE: &str = "arachne.mobject.geometry.polygram.Square";
const POLYGON: &str = "arachne.mobject.geometry.polygram.Polygon";
const SURROUNDING: &str = "arachne.mobject.geometry.shape_matchers.SurroundingRectangle";
const PLATE: &str = "arachne.mobject.geometry.shape_matchers.BackgroundRectangle";
const LABEL: &str = "arachne.mobject.geometry.labeled.Label";
const LABELED_LINE: &str = "arachne.mobject.geometry.labeled.LabeledLine";
const VMOBJECT: &str = "arachne.mobject.types.vectorized_mobject.VMobject";
const VGROUP: &str = "arachne.mobject.types.vectorized_mobject.VGroup";
const POINT: &str = "arachne.mobject.types.vectorized_mobject.VectorizedPoint";
const GLYPH: &str = "arachne.mobject.svg.svg_mobject.VMobjectFromSVGPath";
const TEXT: &str = "arachne.mobject.text.text_mobject.Text";
const IMAGE: &str = "arachne.mobject.types.image_mobject.ImageMobject";
const CLOUD: &str = "arachne.mobject.types.point_cloud_mobject.PMobject";

/// `label_rule_open`, for `cause`.
fn open(cause: OpenCause) -> LabelClass {
    LabelClass::LabelRuleOpen(cause)
}

/// One path of containment, owner first, its nodes' tokens their positions.
fn path(classes: &[&str]) -> Vec<SourceNode> {
    classes
        .iter()
        .enumerate()
        .map(|(token, class)| SourceNode {
            token: token as u32 + 1,
            class: (*class).into(),
        })
        .collect()
}

/// The class of a record drawn through one path, with no facing group.
fn class(classes: &[&str]) -> LabelClass {
    label_class(&[path(classes)], 0)
}

/// "Labels first": a facing group makes a label of anything, even with no source.
#[test]
fn a_label_beats_everything() {
    for classes in [
        &[CONE, ARROW3D][..],
        &[CYLINDER],
        &[SQUARE, CUBE],
        &[CYLINDER, VGROUP],
        &[VMOBJECT],
        &[GLYPH, TEXT],
    ] {
        assert_eq!(label_class(&[path(classes)], 9), LabelClass::Label);
    }
    let disagreeing = [path(&[SQUARE, ARROW]), path(&[SQUARE, VGROUP])];
    assert_eq!(label_class(&disagreeing, 9), LabelClass::Label);
    assert_eq!(label_class(&[], 9), LabelClass::Label);
}

/// "Lines/curves/axes, Line3D, Arrow3D, Arrow, Vector and DoubleArrow include their shafts, caps
/// and tips."
#[test]
fn lines_curves_and_axes_include_their_shafts_caps_and_tips() {
    for named in [
        LINE3D,
        ARROW3D,
        ARROW,
        VECTOR,
        DOUBLE_ARROW,
        LINE,
        AXES,
        CURVE,
    ] {
        assert_eq!(class(&[named]), LabelClass::Line, "{named}");
    }
    // Line3D's faces (one face group, or each face alone) and its end caps.
    assert_eq!(class(&[FACE, LINE3D]), LabelClass::Line);
    assert_eq!(class(&[CIRCLE, LINE3D]), LabelClass::Line);
    // An arrow's tip and the points it keeps.
    assert_eq!(class(&[FILLED_TIP, DOUBLE_ARROW]), LabelClass::Line);
    assert_eq!(class(&[POINT, ARROW3D]), LabelClass::Line);
    // Axes' axis lines and ticks, and a polar plane's circles.
    assert_eq!(class(&[LINE, VGROUP, NUMBER_LINE, AXES]), LabelClass::Line);
    assert_eq!(class(&[CIRCLE, VGROUP, POLAR_PLANE]), LabelClass::Line);
    // A line is a line inside anything.
    assert_eq!(class(&[LINE, CUBE]), LabelClass::Line);
}

/// "Enclosing line/arrow ancestry overrides a child's surface-like class: Arrow3D adds a Cone."
#[test]
fn an_enclosing_line_decides_over_a_childs_surface_class() {
    assert_eq!(class(&[CONE]), LabelClass::Surface);
    assert_eq!(class(&[CONE, ARROW3D]), LabelClass::Line);
    // The cone's faces and its base, through the cone.
    assert_eq!(class(&[FACE, CONE, ARROW3D]), LabelClass::Line);
    assert_eq!(class(&[CIRCLE, CONE, ARROW3D]), LabelClass::Line);
    // Through a plain group, which has no class of its own.
    assert_eq!(class(&[SQUARE, VGROUP, ARROW]), LabelClass::Line);
    // Through an owner of any other class: a labelled line's label plate is the line's.
    assert_eq!(class(&[PLATE, LABEL, LABELED_LINE]), LabelClass::Line);
}

/// A line object as a whole is a line: axes' tick-label glyphs, with no facing group, are lines.
#[test]
fn a_tick_label_under_axes_is_a_line() {
    assert_eq!(class(&[GLYPH, TEXT, NUMBER_LINE, AXES]), LabelClass::Line);
    assert_eq!(class(&[GLYPH, TEXT, VGROUP, NUMBER_LINE]), LabelClass::Line);
}

/// "Vector and its filled tip have distinct classes": both are the vector's.
#[test]
fn a_vectors_filled_tip_is_a_line() {
    assert_eq!(class(&[VECTOR]), LabelClass::Line);
    assert_eq!(class(&[FILLED_TIP, VECTOR]), LabelClass::Line);
    assert_eq!(class(&[FILLED_TIP]), LabelClass::Line);
}

/// "Known surfaces, solids and filled planar shapes are surfaces."
#[test]
fn surfaces_solids_and_filled_planar_shapes_are_surfaces() {
    for classes in [
        &[SPHERE][..],
        &[FACE, SPHERE],
        &[FACE, SPHERE, VGROUP],
        &[SQUARE, CUBE],
        &[POLYGON],
        &[CIRCLE],
        &[CIRCLE, CONE],
        &[SQUARE, VGROUP],
        &[PLATE, TEXT],
    ] {
        assert_eq!(class(classes), LabelClass::Surface, "{classes:?}");
    }
}

/// "Their outline strokes alone are not qualifying fills": a class does not say whether a record
/// is filled, so an outline's class is its shape's, and its paint decides whether it qualifies.
#[test]
fn an_outline_reads_as_its_shape() {
    assert_eq!(class(&[SURROUNDING]), LabelClass::Surface);
    assert_eq!(class(&[CIRCLE]), LabelClass::Surface);
}

/// "A standalone Cylinder is a solid; one owned by a line/arrow is a shaft."
#[test]
fn a_standalone_cylinder_is_a_solid_and_a_lines_is_a_shaft() {
    assert_eq!(class(&[CYLINDER]), LabelClass::Surface);
    assert_eq!(class(&[FACE, CYLINDER]), LabelClass::Surface);
    assert_eq!(class(&[CIRCLE, CYLINDER]), LabelClass::Surface);
    for owner in [&[LINE3D][..], &[ARROW3D], &[VGROUP, LINE3D], &[ARROW]] {
        let classes = [&[CYLINDER][..], owner].concat();
        assert_eq!(class(&classes), LabelClass::Line, "{classes:?}");
        let cap = [&[CIRCLE, CYLINDER][..], owner].concat();
        assert_eq!(class(&cap), LabelClass::Line, "{cap:?}");
    }
}

/// "A raw Cylinder used as a shaft carries no such intent: leave it open, never infer thinness."
/// A cylinder in a plain group, or owned by anything but a line, is open, and so are its parts;
/// a cone beside it is a surface still.
#[test]
fn a_cylinder_in_a_plain_group_is_open() {
    let grouped = open(OpenCause::CylinderGrouped);
    assert_eq!(class(&[CYLINDER, VGROUP]), grouped);
    assert_eq!(class(&[FACE, CYLINDER, VGROUP]), grouped);
    assert_eq!(class(&[CIRCLE, CYLINDER, VGROUP]), grouped);
    assert_eq!(class(&[CYLINDER, CUBE]), grouped);
    assert_eq!(class(&[CONE, VGROUP]), LabelClass::Surface);
}

/// "Generic/conflicting provenance also stays label_rule_open."
#[test]
fn generic_or_conflicting_provenance_is_open() {
    // Generic: a leaf and its containers name no class the rule reads.
    let generic = open(OpenCause::GenericOnly);
    assert_eq!(class(&[VMOBJECT]), generic);
    assert_eq!(class(&[VGROUP]), generic);
    assert_eq!(class(&[VMOBJECT, VGROUP, VGROUP]), generic);
    assert_eq!(label_class(&[], 0), open(OpenCause::NoSource));
    // A class the rule does not name, and what it owns; a point cloud.
    assert_eq!(class(&[IMAGE]), open(OpenCause::Unmapped));
    assert_eq!(class(&[GLYPH, TEXT]), open(OpenCause::Unmapped));
    assert_eq!(class(&[CLOUD]), open(OpenCause::PointCloud));
    let unknown = "arachne.mobject.unknown.Shape";
    assert_eq!(class(&[unknown]), open(OpenCause::Unmapped));
    // Conflicting: a child shared by two parents whose paths disagree.
    let shared = [path(&[SQUARE, ARROW]), path(&[SQUARE, VGROUP])];
    assert_eq!(label_class(&shared, 0), open(OpenCause::PathsDisagree));
    let shaft_or_raw = [path(&[CYLINDER, LINE3D]), path(&[CYLINDER, VGROUP])];
    let disagree = open(OpenCause::PathsDisagree);
    assert_eq!(label_class(&shaft_or_raw, 0), disagree);
    // Paths that agree keep their class.
    let agreeing = [path(&[SQUARE, ARROW]), path(&[SQUARE, VECTOR])];
    assert_eq!(label_class(&agreeing, 0), LabelClass::Line);
}

/// "Publish the exact class inventory": every name is a fully qualified library class in one
/// table only, and alone reads as its table says.
#[test]
fn the_inventory_is_exact() {
    let tables = [
        (LINE_CLASSES, LabelClass::Line),
        (SURFACE_CLASSES, LabelClass::Surface),
        (&[CYLINDER][..], LabelClass::Surface),
        (GENERIC_CLASSES, open(OpenCause::GenericOnly)),
        (POINT_CLOUD_CLASSES, open(OpenCause::PointCloud)),
    ];
    let mut seen = std::collections::BTreeSet::new();
    for (table, alone) in tables {
        for &name in table {
            assert!(name.starts_with("arachne."), "{name}");
            assert!(seen.insert(name), "{name} is in two tables");
            assert_eq!(class(&[name]), alone, "{name}");
        }
    }
}
