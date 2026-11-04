//! Where each record lands on the frame, found without drawing: the box of its points as CE's
//! camera places them for display (`transform_points_pre_display`) and maps them to pixels, by the
//! mapping its kind is drawn through ([`FrameMap::of_kind`]). It is the box
//! `arachne.camera.projection.pixel_box` gives the record's mobject (points, not ink), so the two
//! can be compared record by record; nothing here rasterizes.

use crate::project::{FrameMap, Projection, mapping_centre_of, placement, pre_display};
use crate::xf::*;

/// A box in pixels, `(x_min, y_min, x_max, y_max)`: x right, y down, pixel `i` spanning
/// `[i, i + 1)`.
pub type PixelBox = [f64; 4];

/// The pixel box of `rec`'s points under `view`, its shapes found by `shape`; `None` when it
/// places no point (no shape, an empty one, a reserved or unknown kind).
///
/// Its points are those the drawer materialises: the shape through the record's placement, a
/// reveal window cut from it (`pointwise_become_partial`), then a blend with its second shape
/// (`interpolate`); a face group's faces each placed as CE places a face, an instances record's
/// members each through its own placement after the record's. A run of points with a non-finite
/// value is placed, as CE places it, as one point at the origin.
pub fn record_box<'a>(
    view: &View,
    rec: &'a Record,
    shape: impl Fn(&Key) -> Option<&'a ShapeData>,
) -> Option<PixelBox> {
    let geometry = &rec.geometry;
    let place_a = &geometry.place_a;
    let placed = |points: &[[f64; 3]], m: &Affine| -> Vec<[f64; 3]> {
        points.iter().map(|p| place(m, *p)).collect()
    };
    let data = geometry.shape_a.and_then(|k| shape(&k));
    let groups: Vec<Vec<[f64; 3]>> = match (rec.kind, data, &rec.body) {
        (RecordKind::Path, Some(ShapeData::Path { points }), _) => {
            let mut a = placed(points, place_a);
            if let Some((lo, hi)) = geometry.reveal {
                a = revealed(&a, lo, hi);
            }
            if let Some(blend) = &geometry.blend
                && let Some(ShapeData::Path { points: b }) = shape(&blend.shape_b)
            {
                let b = placed(b, &blend.place_b);
                a = a
                    .iter()
                    .zip(&b)
                    .map(|(p, q)| std::array::from_fn(|k| (1.0 - blend.t) * p[k] + blend.t * q[k]))
                    .collect();
            }
            vec![a]
        }
        (RecordKind::FaceGroup, Some(ShapeData::Path { points }), Body::FaceGroup(faces)) => {
            let count = faces.uv.len();
            if count == 0 {
                return None;
            }
            let all = placed(points, place_a);
            let size = all.len() / count;
            if size == 0 {
                return None;
            }
            all.chunks(size).map(<[_]>::to_vec).collect()
        }
        (RecordKind::Points, Some(ShapeData::Path { points }), _) => vec![placed(points, place_a)],
        (RecordKind::Mesh, Some(ShapeData::Mesh { positions, .. }), _) => {
            vec![placed(positions, place_a)]
        }
        (RecordKind::Image, _, Body::Image(source)) => {
            let corners = match source {
                ImageSource::Picture { corners, .. } | ImageSource::SubFrame { corners, .. } => {
                    corners
                }
            };
            vec![placed(corners, place_a)]
        }
        (RecordKind::Instances, Some(prototype), Body::Instances(body)) => {
            let points: &[[f64; 3]] = match prototype {
                ShapeData::Path { points } => points,
                ShapeData::Mesh { positions, .. } => positions,
                _ => return None,
            };
            body.placements
                .iter()
                .map(|m| placed(points, &compose(m, place_a)))
                .collect()
        }
        _ => return None,
    };
    let projection = match &view.camera {
        Camera::ThreeD(c) => Some(Projection::new(c, view.frame_center)),
        _ => None,
    };
    let (pw, ph) = (view.pixel_width, view.pixel_height);
    let map = FrameMap::of_kind(view, rec.kind, pw, ph);
    let (place_kind, _) = placement(
        projection.as_ref(),
        &map,
        rec.depth.space,
        rec.depth.fixed_orientation_centre,
    );
    let centre = mapping_centre_of(view, rec.kind);
    let (pw, ph) = (pw as f64, ph as f64);
    let (x_mult, y_mult) = (pw / view.frame_width, -(ph / view.frame_height));
    let mut found: Option<PixelBox> = None;
    for group in groups.iter().filter(|g| !g.is_empty()) {
        for p in pre_display(group, projection.as_ref(), place_kind) {
            // `projection._placed`: the centre subtracted, then scaled about the frame's middle.
            let x = (p[0] - centre[0]) * x_mult + pw / 2.0;
            let y = (p[1] - centre[1]) * y_mult + ph / 2.0;
            let b = found.get_or_insert([x, y, x, y]);
            *b = [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)];
        }
    }
    found
}

/// `VMobject.pointwise_become_partial(path, lo, hi)` of a path holding no points before: the part
/// of the cubic spline `points` between `lo` and `hi`.
pub fn revealed(points: &[[f64; 3]], lo: f64, hi: f64) -> Vec<[f64; 3]> {
    if lo <= 0.0 && hi >= 1.0 {
        return points.to_vec();
    }
    let curves = points.len() / 4;
    if curves == 0 {
        return Vec::new();
    }
    let (lower, lower_residue) = integer_interpolate(curves, lo);
    let (upper, upper_residue) = integer_interpolate(curves, hi);
    let curve = |i: usize| -> [[f64; 3]; 4] { std::array::from_fn(|k| points[4 * i + k]) };
    if lower == upper {
        return partial_cubic(curve(lower), lower_residue, upper_residue).to_vec();
    }
    let mut out = partial_cubic(curve(lower), lower_residue, 1.0).to_vec();
    out.extend_from_slice(&points[4 * (lower + 1)..4 * upper]);
    out.extend(partial_cubic(curve(upper), 0.0, upper_residue));
    out
}

/// `bezier.integer_interpolate(0, end, alpha)`.
fn integer_interpolate(end: usize, alpha: f64) -> (usize, f64) {
    if alpha >= 1.0 {
        return (end - 1, 1.0);
    }
    if alpha <= 0.0 {
        return (0, 0.0);
    }
    let scaled = end as f64 * alpha;
    (scaled as usize, scaled.rem_euclid(1.0))
}

/// The part of one cubic between `a` and `b` (`bezier.partial_bezier_points`), by de Casteljau.
fn partial_cubic(c: [[f64; 3]; 4], a: f64, b: f64) -> [[f64; 3]; 4] {
    let lerp = |p: [f64; 3], q: [f64; 3], t: f64| -> [f64; 3] {
        std::array::from_fn(|k| p[k] + (q[k] - p[k]) * t)
    };
    // The curve from `t` on.
    let tail = |c: [[f64; 3]; 4], t: f64| -> [[f64; 3]; 4] {
        let (ab, bc, cd) = (
            lerp(c[0], c[1], t),
            lerp(c[1], c[2], t),
            lerp(c[2], c[3], t),
        );
        let (abc, bcd) = (lerp(ab, bc, t), lerp(bc, cd, t));
        [lerp(abc, bcd, t), bcd, cd, c[3]]
    };
    // The curve up to `t`.
    let head = |c: [[f64; 3]; 4], t: f64| -> [[f64; 3]; 4] {
        let (ab, bc, cd) = (
            lerp(c[0], c[1], t),
            lerp(c[1], c[2], t),
            lerp(c[2], c[3], t),
        );
        let (abc, bcd) = (lerp(ab, bc, t), lerp(bc, cd, t));
        [c[0], ab, abc, lerp(abc, bcd, t)]
    };
    if a >= 1.0 {
        return [c[3]; 4];
    }
    let from_a = tail(c, a);
    head(from_a, (b - a) / (1.0 - a))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
        // Each segment a straight cubic: anchor, thirds, anchor.
        let mut out = Vec::new();
        for w in points.windows(2) {
            let (p, q) = (w[0], w[1]);
            let at = |t: f64| -> [f64; 3] { std::array::from_fn(|k| p[k] + (q[k] - p[k]) * t) };
            out.extend([p, at(1.0 / 3.0), at(2.0 / 3.0), q]);
        }
        out
    }

    fn view(camera: Camera) -> View {
        View {
            pixel_width: 160,
            pixel_height: 90,
            frame_width: 16.0,
            frame_height: 9.0,
            frame_center: [0.0; 3],
            mapping_centre: [0.0; 3],
            background: Background::Color([0.0, 0.0, 0.0, 1.0]),
            camera,
            cairo_line_width_multiple: 0.01,
            use_z_index: true,
            light: LightModel::Ce,
        }
    }

    fn path(points: Vec<[f64; 3]>) -> (Record, ShapeData, Key) {
        let mut rec = Record::new(RecordKind::Path, 0, 0);
        let key = Key([1; 16]);
        rec.geometry.shape_a = Some(key);
        (rec, ShapeData::Path { points }, key)
    }

    fn close(a: PixelBox, b: PixelBox) {
        for k in 0..4 {
            assert!((a[k] - b[k]).abs() < 1e-9, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn a_flat_path_s_box_is_its_points_mapped_y_down() {
        let (rec, data, key) = path(line(&[[-1.0, -1.0, 0.0], [2.0, 1.0, 0.0]]));
        let got = record_box(&view(Camera::Flat), &rec, |k| (*k == key).then_some(&data));
        // 10 px a unit, the origin at (80, 45).
        close(got.unwrap(), [70.0, 35.0, 100.0, 55.0]);
    }

    #[test]
    fn a_path_maps_around_the_mapping_centre_a_point_cloud_around_the_frame_s() {
        let mut v = view(Camera::Flat);
        v.mapping_centre = [1.0, 0.0, 0.0];
        v.frame_center = [0.0, 1.0, 0.0];
        let (rec, data, key) = path(line(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]));
        let got = record_box(&v, &rec, |k| (*k == key).then_some(&data)).unwrap();
        close(got, [70.0, 45.0, 70.0, 45.0]);
        let mut cloud = rec.clone();
        cloud.kind = RecordKind::Points;
        cloud.body = Body::Points(Points {
            colors: Key([2; 16]),
            radius: 4.0,
        });
        let got = record_box(&v, &cloud, |k| (*k == key).then_some(&data)).unwrap();
        close(got, [80.0, 55.0, 80.0, 55.0]);
    }

    #[test]
    fn a_reveal_and_a_blend_are_boxed_as_drawn() {
        let (mut rec, data, key) = path(line(&[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]));
        rec.geometry.reveal = Some((0.25, 0.5));
        let got = record_box(&view(Camera::Flat), &rec, |k| (*k == key).then_some(&data));
        close(got.unwrap(), [90.0, 45.0, 100.0, 45.0]);
        let (mut rec, data, key) = path(line(&[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]));
        let other = Key([3; 16]);
        let b = ShapeData::Path {
            points: line(&[[0.0, 2.0, 0.0], [4.0, 2.0, 0.0]]),
        };
        rec.geometry.blend = Some(Blend {
            shape_b: other,
            place_b: IDENTITY,
            t: 0.5,
        });
        let got = record_box(&view(Camera::Flat), &rec, |k| {
            if *k == key {
                Some(&data)
            } else if *k == other {
                Some(&b)
            } else {
                None
            }
        });
        close(got.unwrap(), [80.0, 35.0, 120.0, 35.0]);
    }

    #[test]
    fn a_path_with_a_non_finite_point_is_one_point_at_the_origin() {
        let (rec, data, key) = path(line(&[[1.0, 1.0, 0.0], [f64::NAN, 1.0, 0.0]]));
        let got = record_box(&view(Camera::Flat), &rec, |k| (*k == key).then_some(&data));
        close(got.unwrap(), [80.0, 45.0, 80.0, 45.0]);
    }

    #[test]
    fn an_empty_or_missing_shape_has_no_box() {
        let (rec, data, key) = path(Vec::new());
        assert!(record_box(&view(Camera::Flat), &rec, |k| (*k == key).then_some(&data)).is_none());
        assert!(record_box(&view(Camera::Flat), &rec, |_| None).is_none());
    }

    #[test]
    fn instances_box_every_member() {
        let (mut rec, data, key) = path(line(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]));
        rec.kind = RecordKind::Instances;
        let mut far = IDENTITY;
        far[1][3] = -3.0;
        rec.body = Body::Instances(Instances {
            placements: vec![IDENTITY, far],
            fill: Key([4; 16]),
            stroke: Key([5; 16]),
            objects: vec![0, 1],
        });
        let got = record_box(&view(Camera::Flat), &rec, |k| (*k == key).then_some(&data));
        close(got.unwrap(), [80.0, 45.0, 90.0, 75.0]);
    }

    #[test]
    fn a_projected_point_lands_where_ce_s_projection_puts_it() {
        let camera = ThreeDCamera {
            phi: 0.0,
            theta: -std::f64::consts::FRAC_PI_2,
            gamma: 0.0,
            focal_distance: 20.0,
            zoom: 1.0,
            exponential_projection: false,
            light_source: [0.0; 3],
            shading_factor: 0.2,
            should_apply_shading: true,
        };
        let mut rec = Record::new(RecordKind::Points, 0, 0);
        let key = Key([1; 16]);
        rec.geometry.shape_a = Some(key);
        rec.depth.space = Space::World;
        let data = ShapeData::Path {
            points: vec![[1.0, 0.0, 10.0]],
        };
        let v = view(Camera::ThreeD(camera.clone()));
        let got = record_box(&v, &rec, |k| (*k == key).then_some(&data)).unwrap();
        let p = Projection::new(&camera, [0.0; 3]).project([1.0, 0.0, 10.0]);
        close(
            got,
            [
                80.0 + 10.0 * p[0],
                45.0 - 10.0 * p[1],
                80.0 + 10.0 * p[0],
                45.0 - 10.0 * p[1],
            ],
        );
        // Twice nearer than the focal plane's half: twice as far out.
        assert!((p[0] - 2.0).abs() < 1e-9, "{p:?}");
    }

    #[test]
    fn a_reveal_cuts_whole_curves_and_parts_as_ce_does() {
        let points = line(&[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 3.0, 0.0]]);
        let part = revealed(&points, 0.25, 0.75);
        assert_eq!(part.len(), 8);
        let x = |p: [f64; 3]| (p[0] * 1e9).round() / 1e9;
        assert_eq!(x(part[0]), 1.5);
        assert_eq!(part[3], [3.0, 0.0, 0.0]);
        assert!((part[7][1] - 1.5).abs() < 1e-12);
        assert_eq!(revealed(&points, 0.0, 1.0), points);
        assert_eq!(revealed(&[], 0.2, 0.4), Vec::<[f64; 3]>::new());
    }
}
