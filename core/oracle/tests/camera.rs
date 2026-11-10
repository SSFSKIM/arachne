//! The camera against closed forms (contract §2).

mod common;

use arachne_core::xf::{Camera, Record, RecordKind, Space, ThreeDCamera};
use arachne_oracle::camera::{
    BEHIND_EYE, FrameMap, Placement, Projection, map_point_cloud, matmul, rotation_about_x,
    rotation_about_z,
};

const DEG: f64 = std::f64::consts::TAU / 360.0;

fn camera(phi: f64, theta: f64, gamma: f64) -> ThreeDCamera {
    ThreeDCamera {
        phi,
        theta,
        gamma,
        focal_distance: 20.0,
        zoom: 1.0,
        exponential_projection: false,
        light_source: [-7.0, -9.0, 10.0],
        shading_factor: 0.2,
        should_apply_shading: true,
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// The textbook rotation about `x` by `a`.
fn rx(a: f64) -> [[f64; 3]; 3] {
    let (c, s) = (libm::cos(a), libm::sin(a));
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

#[test]
fn the_rotation_about_x_is_the_textbook_rotation() {
    for a in [0.0, 1e-4, 0.3, -75.0 * DEG, 2.5] {
        let (got, want) = (rotation_about_x(a), rx(a));
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    close(got[i][j], want[i][j], 4e-16),
                    "a = {a}, [{i}][{j}]: {got:?}"
                );
            }
        }
    }
}

#[test]
fn the_camera_with_theta_minus_ninety_and_no_tilt_is_the_identity() {
    let p = Projection::new(&camera(0.0, -90.0 * DEG, 0.0), [0.0; 3]);
    for i in 0..3 {
        for j in 0..3 {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(close(p.r[i][j], want, 1e-16), "{:?}", p.r);
        }
    }
}

#[test]
fn a_point_on_the_axis_projects_to_the_centre_at_ces_factor() {
    // R = I: the sight axis is z, and a point on it stays at (0, 0) with its depth.
    let p = Projection::new(&camera(0.0, -90.0 * DEG, 0.0), [0.0; 3]);
    for z in [-5.0, 0.0, 3.0, 19.0] {
        let q = p.project([0.0, 0.0, z]);
        assert_eq!(q, [0.0, 0.0, z]);
        assert_eq!(p.factor(z), 20.0 / (20.0 - z));
        assert_eq!(p.key(z), -(20.0 / (20.0 - z)));
    }
    // Off the axis, x and y scale by the factor.
    let q = p.project([2.0, -1.0, 10.0]);
    assert_eq!(q, [4.0, -2.0, 10.0]);
}

#[test]
fn a_known_rotation_turns_the_axes_as_its_closed_form_does() {
    // Theta 0, phi 75, gamma 0: R = Rx(-75) Rz(-90).
    let cam = camera(75.0 * DEG, 0.0, 0.0);
    let p = Projection::new(&cam, [0.0; 3]);
    let want = matmul(
        &rx(-75.0 * DEG),
        &[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    );
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                close(p.r[i][j], want[i][j], 1e-15),
                "{:?} against {want:?}",
                p.r
            );
        }
    }
    // It is a rotation: orthonormal.
    for i in 0..3 {
        for j in 0..3 {
            let dot: f64 = (0..3).map(|k| p.r[i][k] * p.r[j][k]).sum();
            assert!(close(dot, if i == j { 1.0 } else { 0.0 }, 1e-15));
        }
    }
    // The product is formed as the text writes it: F · R for Rz(−θ − 90°), Rx(−φ), Rz(γ) in turn.
    let mut r = arachne_oracle::camera::IDENTITY3;
    for f in [
        rotation_about_z(-cam.theta - 90.0 * DEG),
        rotation_about_x(-cam.phi),
        rotation_about_z(cam.gamma),
    ] {
        r = matmul(&f, &r);
    }
    assert_eq!(p.r, r);
}

#[test]
fn the_frame_centre_is_subtracted_before_the_rotation() {
    let p = Projection::new(
        &camera(30.0 * DEG, -60.0 * DEG, 10.0 * DEG),
        [1.0, -2.0, 0.5],
    );
    assert_eq!(p.rotated([1.0, -2.0, 0.5]), [0.0, 0.0, 0.0]);
    let q = p.project([1.0, -2.0, 0.5]);
    assert_eq!(q, [0.0, 0.0, 0.0]);
}

#[test]
fn a_point_behind_the_eye_takes_ces_large_factor() {
    let p = Projection::new(&camera(0.0, -90.0 * DEG, 0.0), [0.0; 3]);
    assert_eq!(p.factor(25.0), BEHIND_EYE);
    assert_eq!(p.project([1.0, 1.0, 25.0]), [1e6, 1e6, 25.0]);
    // On the eye's plane: d / 0.
    assert_eq!(p.factor(20.0), f64::INFINITY);
    assert!(!p.keyed_planar(20.0) && !p.keyed_planar(25.0) && p.keyed_planar(19.0));
}

#[test]
fn the_exponential_option_takes_exp_in_front_of_the_eye() {
    let mut cam = camera(0.0, -90.0 * DEG, 0.0);
    cam.exponential_projection = true;
    let p = Projection::new(&cam, [0.0; 3]);
    assert_eq!(p.factor(-4.0), 20.0 / 24.0);
    assert_eq!(p.factor(4.0), libm::exp(4.0 / 20.0));
    assert!(!p.keyed_planar(0.0));
}

#[test]
fn unproject_inverts_project_and_camera_axes_drop_the_rotation() {
    let p = Projection::new(&camera(70.0 * DEG, -40.0 * DEG, 5.0 * DEG), [0.3, 0.0, 0.0]);
    let q = [1.5, -0.5, 2.0];
    let back = p.unproject(p.project(q));
    let rot = p.rotated(q);
    for k in 0..3 {
        assert!(close(back[k], rot[k], 1e-14), "{back:?} against {rot:?}");
    }
    let axes = p.in_camera_axes();
    assert_eq!(axes.rotated(q), q);
}

#[test]
fn the_frame_mapping_is_ces_cairo_matrix() {
    let mut v = common::view(854, 480);
    v.frame_width = 14.0 + 2.0 / 9.0;
    v.frame_height = 8.0;
    let m = FrameMap::of_paths(&v);
    assert_eq!(m.map(0.0, 0.0), [427.0, 240.0]);
    assert_eq!(m.map(0.0, 4.0), [427.0, 0.0]);
    assert_eq!(m.map(0.0, -4.0), [427.0, 480.0]);
    assert!(close(m.map(v.frame_width / 2.0, 0.0)[0], 854.0, 1e-12));
    // Paths map about the mapping centre, point clouds about the frame centre.
    v.mapping_centre = [1.0, 0.0, 0.0];
    v.frame_center = [0.0, 1.0, 0.0];
    let m = FrameMap::of_paths(&v);
    let at = m.map(1.0, 0.0);
    assert!(close(at[0], 427.0, 1e-12) && at[1] == 240.0, "{at:?}");
    assert_eq!(map_point_cloud(&v, [0.0, 1.0, 0.0]), [427.0, 240.0]);
    assert_eq!(m.scale(), m.xx.abs().max(m.yy.abs()));
}

#[test]
fn placement_kinds_follow_the_camera_and_the_space() {
    let mut v = common::view(64, 64);
    let mut r = Record::new(RecordKind::Path, 0, 0);
    r.depth.space = Space::World;
    assert_eq!(Placement::of(&r, &v), Placement::Kept);
    v.camera = Camera::ThreeD(camera(75.0 * DEG, -30.0 * DEG, 0.0));
    assert_eq!(Placement::of(&r, &v), Placement::Projected);
    r.depth.space = Space::Screen;
    assert_eq!(Placement::of(&r, &v), Placement::Kept);
    r.depth.space = Space::Overlay;
    assert_eq!(Placement::of(&r, &v), Placement::FixedInFrame);
    // A fixed-orientation record moves by its centre's projection: its centre shows where the
    // projection puts it, every point by the same offset.
    r.depth.space = Space::World;
    r.depth.fixed_orientation_centre = Some([1.0, 2.0, 0.5]);
    let pl = Placement::of(&r, &v);
    let Camera::ThreeD(cam) = &v.camera else {
        unreachable!()
    };
    let pr = Projection::new(cam, v.frame_center);
    let c = pr.project([1.0, 2.0, 0.5]);
    let shown = pl.show(Some(&pr), [1.0, 2.0, 0.5]);
    for k in 0..3 {
        assert!(close(shown[k], c[k], 1e-15));
    }
    let moved = pl.show(Some(&pr), [2.0, 2.0, 0.5]);
    assert!(close(moved[0] - shown[0], 1.0, 1e-15) && close(moved[1], shown[1], 1e-15));
}
