//! `label_classes@2` (contract §10): a record's class from its source, rule by rule.

use arachne_oracle::label_classes::{Cause, Outcome, path_outcome, record_outcome};

const M: &str = "arachne.mobject.three_d.three_dimensions.";
const G: &str = "arachne.mobject.geometry.";

fn c(module: &str, name: &str) -> String {
    format!("{module}{name}")
}

fn path(nodes: &[String]) -> Outcome {
    path_outcome(&nodes.iter().map(String::as_str).collect::<Vec<_>>())
}

#[test]
fn line_ancestry_makes_any_node_under_a_line_a_line() {
    // A cone, a cylinder and a filled tip inside an arrow are lines.
    for part in [
        c(M, "Cone"),
        c(M, "Cylinder"),
        c(G, "tips.ArrowTriangleFilledTip"),
    ] {
        assert_eq!(path(&[part, c(M, "Arrow3D")]), Outcome::Line);
    }
    // A rectangle behind a label inside a labelled line.
    let rect = c(G, "polygram.Rectangle");
    let vgroup = "arachne.mobject.types.vectorized_mobject.VGroup".to_string();
    assert_eq!(
        path(&[rect, vgroup, c(G, "labeled.LabeledLine")]),
        Outcome::Line
    );
}

#[test]
fn a_generic_maker_reads_through_to_its_nearest_container() {
    let vm = "arachne.mobject.types.vectorized_mobject.VMobject".to_string();
    let group = "arachne.mobject.mobject.Group".to_string();
    assert_eq!(
        path(&[vm.clone(), group.clone(), c(G, "polygram.Square")]),
        Outcome::Surface
    );
    assert_eq!(path(&[vm, group]), Outcome::Open(Cause::GenericOnly));
}

#[test]
fn the_cylinder_rule_reads_a_cylinder_by_where_it_stands() {
    // Alone, a solid; in any container that is not a line, open.
    assert_eq!(path(&[c(M, "Cylinder")]), Outcome::Surface);
    let vgroup = "arachne.mobject.types.vectorized_mobject.VGroup".to_string();
    assert_eq!(
        path(&[c(M, "Cylinder"), vgroup.clone()]),
        Outcome::Open(Cause::CylinderGrouped)
    );
    assert_eq!(
        path(&[c(M, "Cylinder"), c(M, "Surface")]),
        Outcome::Open(Cause::CylinderGrouped)
    );
    // A cap of a cylinder follows its cylinder, even where a surface holds the cylinder.
    let cap = c(G, "arc.Circle");
    assert_eq!(path(&[cap.clone(), c(M, "Cylinder")]), Outcome::Surface);
    assert_eq!(
        path(&[cap, c(M, "Cylinder"), c(M, "Surface")]),
        Outcome::Open(Cause::CylinderGrouped)
    );
}

#[test]
fn parts_follow_a_surface_like_container_past_open_nodes() {
    // A part under an open node inside a surface is a surface.
    let text = "arachne.mobject.text.text_mobject.Text".to_string();
    let pts = "arachne.mobject.types.point_cloud_mobject.PMobject".to_string();
    assert_eq!(path(&[text.clone(), c(M, "Sphere")]), Outcome::Surface);
    assert_eq!(
        path(&[pts.clone(), text.clone(), c(M, "Cube")]),
        Outcome::Surface
    );
    // With no surface-like container, the node's own class.
    assert_eq!(path(&[text]), Outcome::Open(Cause::Unmapped));
    assert_eq!(path(&[pts]), Outcome::Open(Cause::PointCloud));
    assert_eq!(
        path(&["user.module.Unknown".to_string()]),
        Outcome::Open(Cause::Unmapped)
    );
}

#[test]
fn the_key_is_the_class_itself_never_its_base() {
    // Circle derives from Arc (a line) and is a surface; Dot is read as itself.
    assert_eq!(path(&[c(G, "arc.Circle")]), Outcome::Surface);
    assert_eq!(path(&[c(G, "arc.Arc")]), Outcome::Line);
    assert_eq!(path(&[c(G, "arc.Dot")]), Outcome::Surface);
}

#[test]
fn a_record_s_paths_agree_or_it_is_open() {
    let sq = c(G, "polygram.Square");
    let line = c(G, "line.Line");
    assert_eq!(
        record_outcome([vec![sq.as_str()], vec![sq.as_str()]]),
        Outcome::Surface
    );
    assert_eq!(
        record_outcome([vec![sq.as_str()], vec![sq.as_str(), line.as_str()]]),
        Outcome::Open(Cause::PathsDisagree)
    );
    assert_eq!(
        record_outcome(Vec::<Vec<&str>>::new()),
        Outcome::Open(Cause::NoSource)
    );
}
